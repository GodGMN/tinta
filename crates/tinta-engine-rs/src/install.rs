//! `install_models`: the only path that downloads model files.
//!
//! The sequence matches the Swift engine: one `install_progress` with
//! `state: "downloading"` per component (transcription, vad, diarization), then
//! `all` with `state: "done"`. A file is verified after it is written and again
//! before its component is loaded; a mismatch deletes the file.

use anyhow::{bail, Context, Result};
use bzip2::read::BzDecoder;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::manifest::{installed_path, verify_file, Manifest, ModelFile, Source};
use crate::protocol::{log, Output};
use crate::sherpa;
use crate::threads::Threads;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const READ_TIMEOUT: Duration = Duration::from_secs(300);
const COPY_BLOCK: usize = 64 * 1024;

pub fn install(manifest: &Manifest, root: &Path, out: &Output, threads: &Threads) -> Result<()> {
    fs::create_dir_all(root)
        .with_context(|| format!("cannot create the models folder {}", root.display()))?;
    let provider = sherpa::provider(root)?;
    let work = std::env::temp_dir().join(format!("tinta-engine-install-{}", std::process::id()));
    fs::create_dir_all(&work)
        .with_context(|| format!("cannot create the download folder {}", work.display()))?;

    let result = install_inner(manifest, root, out, threads, &provider, &work);
    if let Err(error) = fs::remove_dir_all(&work) {
        log(&format!("cannot remove {}: {error}", work.display()));
    }
    result
}

fn install_inner(
    manifest: &Manifest,
    root: &Path,
    out: &Output,
    threads: &Threads,
    provider: &str,
    work: &Path,
) -> Result<()> {
    let mut archives: HashMap<String, PathBuf> = HashMap::new();

    for group in &manifest.groups {
        out.event(
            "install_progress",
            serde_json::json!({ "component": group.component, "state": "downloading" }),
        )
        .context("cannot write install progress")?;

        for file in &group.files {
            ensure_file(file, root, work, &mut archives)
                .with_context(|| format!("cannot install {}", file.path))?;
        }

        // Hash every installed file again before loading, so a corrupted file
        // on disk is rejected and deleted instead of reaching sherpa-onnx.
        for file in &group.files {
            verify_file(root, file)?;
        }

        sherpa::load_group(group, root, threads, provider)?;
        log(&format!("loaded the {} models", group.component));
    }

    out.event(
        "install_progress",
        serde_json::json!({ "component": "all", "state": "done" }),
    )
    .context("cannot write install progress")?;
    Ok(())
}

fn ensure_file(
    file: &ModelFile,
    root: &Path,
    work: &Path,
    archives: &mut HashMap<String, PathBuf>,
) -> Result<()> {
    let destination = installed_path(root, file);
    if destination.is_file() {
        if verify_file(root, file).is_ok() {
            return Ok(());
        }
        log(&format!("deleting the corrupt model file {}", file.path));
        let _ = fs::remove_file(&destination);
    }
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }

    match &file.source {
        Source::File { url } => download(url, &destination, file.size, &file.sha256, &file.path),
        Source::Archive {
            url,
            size,
            sha256,
            member,
        } => {
            let archive = archive_for(url, *size, sha256, work, archives)?;
            extract_member(
                &archive,
                member,
                &destination,
                file.size,
                &file.sha256,
                &file.path,
            )
        }
    }
}

fn archive_for(
    url: &str,
    size: u64,
    sha256: &str,
    work: &Path,
    archives: &mut HashMap<String, PathBuf>,
) -> Result<PathBuf> {
    if let Some(path) = archives.get(url) {
        return Ok(path.clone());
    }
    let name = url
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty())
        .unwrap_or("model-archive");
    let path = work.join(name);
    download(url, &path, size, sha256, name)?;
    archives.insert(url.to_string(), path.clone());
    Ok(path)
}

fn download(url: &str, destination: &Path, size: u64, sha256: &str, label: &str) -> Result<()> {
    let part = with_part_suffix(destination);
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(CONNECT_TIMEOUT)
        .timeout_read(READ_TIMEOUT)
        .build();
    let response = agent
        .get(url)
        .call()
        .with_context(|| format!("cannot download {label} from {url}"))?;
    let mut reader = response.into_reader();
    let mut writer =
        File::create(&part).with_context(|| format!("cannot write {}", part.display()))?;
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = vec![0_u8; COPY_BLOCK];

    let outcome = (|| -> Result<()> {
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            writer.write_all(&buffer[..read])?;
            total += read as u64;
        }
        writer.flush()?;
        Ok(())
    })();
    drop(writer);

    if let Err(error) = outcome {
        let _ = fs::remove_file(&part);
        return Err(error).with_context(|| format!("cannot download {label}"));
    }
    let digest = format!("{:x}", hasher.finalize());
    if total != size || digest != sha256.to_lowercase() {
        let _ = fs::remove_file(&part);
        bail!("model file does not match its pinned hash: {label}. Install the models again.");
    }
    fs::rename(&part, destination)
        .with_context(|| format!("cannot move {} into place", destination.display()))?;
    Ok(())
}

fn extract_member(
    archive: &Path,
    member: &str,
    destination: &Path,
    size: u64,
    sha256: &str,
    label: &str,
) -> Result<()> {
    let file = File::open(archive)
        .with_context(|| format!("cannot open the model archive {}", archive.display()))?;
    let mut tar = tar::Archive::new(BzDecoder::new(file));
    let part = with_part_suffix(destination);
    let mut writer =
        File::create(&part).with_context(|| format!("cannot write {}", part.display()))?;
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = vec![0_u8; COPY_BLOCK];
    let mut found = false;

    for entry in tar.entries().context("cannot read the model archive")? {
        let mut entry = entry.context("cannot read the model archive")?;
        let entry_path = entry.path().context("cannot read a model archive path")?;
        if entry_path != Path::new(member) {
            continue;
        }
        found = true;
        loop {
            let read = entry.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
            writer.write_all(&buffer[..read])?;
            total += read as u64;
        }
        break;
    }
    writer.flush()?;
    drop(writer);

    if !found {
        let _ = fs::remove_file(&part);
        bail!("the model archive is missing {member}. Install the models again.");
    }
    let digest = format!("{:x}", hasher.finalize());
    if total != size || digest != sha256.to_lowercase() {
        let _ = fs::remove_file(&part);
        bail!("model file does not match its pinned hash: {label}. Install the models again.");
    }
    fs::rename(&part, destination)
        .with_context(|| format!("cannot move {} into place", destination.display()))?;
    Ok(())
}

fn with_part_suffix(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".part");
    PathBuf::from(name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bzip2::write::BzEncoder;
    use bzip2::Compression;
    use std::fs;

    const HELLO_SHA: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    fn write_archive(path: &Path, member: &str, data: &[u8]) {
        let file = File::create(path).unwrap();
        let encoder = BzEncoder::new(file, Compression::best());
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_size(data.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, member, data).unwrap();
        builder.into_inner().unwrap().finish().unwrap();
    }

    #[test]
    fn extracts_a_verified_member() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("models.tar.bz2");
        write_archive(&archive, "bundle/model.bin", b"hello");
        let destination = dir.path().join("out/model.bin");
        fs::create_dir_all(destination.parent().unwrap()).unwrap();

        extract_member(
            &archive,
            "bundle/model.bin",
            &destination,
            5,
            HELLO_SHA,
            "out/model.bin",
        )
        .unwrap();
        assert_eq!(fs::read(&destination).unwrap(), b"hello");
    }

    #[test]
    fn rejects_a_member_with_the_wrong_hash() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("models.tar.bz2");
        write_archive(&archive, "bundle/model.bin", b"jello");
        let destination = dir.path().join("out/model.bin");
        fs::create_dir_all(destination.parent().unwrap()).unwrap();

        let error = extract_member(
            &archive,
            "bundle/model.bin",
            &destination,
            5,
            HELLO_SHA,
            "out/model.bin",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("does not match its pinned hash"), "{error}");
        assert!(!destination.exists());
        assert!(!with_part_suffix(&destination).exists());
    }

    #[test]
    fn reports_a_missing_member() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("models.tar.bz2");
        write_archive(&archive, "bundle/other.bin", b"hello");
        let destination = dir.path().join("out/model.bin");
        fs::create_dir_all(destination.parent().unwrap()).unwrap();

        let error = extract_member(
            &archive,
            "bundle/model.bin",
            &destination,
            5,
            HELLO_SHA,
            "out/model.bin",
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("is missing bundle/model.bin"), "{error}");
        assert!(!destination.exists());
    }
}
