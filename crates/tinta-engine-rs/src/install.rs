//! `install_models`: the only path that downloads model files.
//!
//! The sequence matches the Swift engine: one `install_progress` with
//! `state: "downloading"` per component (transcription, vad, diarization), then
//! `all` with `state: "done"`. Every file is hashed exactly once per install,
//! after it is written; a mismatch deletes it.
//!
//! Interrupted installs resume. Archive downloads live in a fixed
//! `<models root>/.downloads/` folder and direct files keep a `.part` beside
//! their destination, so a later run asks the server for the remaining bytes
//! with an HTTP `Range` request. The folder is removed after a clean install.

use anyhow::{bail, Context, Result};
use bzip2::read::BzDecoder;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::manifest::{sha256_file, verify_file, Manifest, ModelFile, Source};
use crate::protocol::{log, Output};
use crate::sherpa;
use crate::threads::Threads;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const READ_TIMEOUT: Duration = Duration::from_secs(300);
const COPY_BLOCK: usize = 64 * 1024;

/// Files that share an archive, so the bz2 stream is decompressed once.
struct ArchiveRequest<'a> {
    url: &'a str,
    size: u64,
    sha256: &'a str,
    files: Vec<&'a ModelFile>,
}

pub fn install(manifest: &Manifest, root: &Path, out: &Output, threads: &Threads) -> Result<()> {
    fs::create_dir_all(root)
        .with_context(|| format!("cannot create the models folder {}", root.display()))?;
    let provider = sherpa::provider(root)?;
    let work = root.join(".downloads");
    fs::create_dir_all(&work)
        .with_context(|| format!("cannot create the download folder {}", work.display()))?;

    let result = install_inner(manifest, root, out, threads, &provider, &work);
    if result.is_ok() {
        if let Err(error) = fs::remove_dir_all(&work) {
            log(&format!("cannot remove {}: {error}", work.display()));
        }
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

        install_group(group, root, work, &mut archives)
            .with_context(|| format!("cannot install the {} models", group.component))?;

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

/// Installs one component. Direct files download to their destination; members
/// of the same archive extract together in a single pass over the bz2 stream.
fn install_group(
    group: &crate::manifest::ModelGroup,
    root: &Path,
    work: &Path,
    archives: &mut HashMap<String, PathBuf>,
) -> Result<()> {
    let mut requests: Vec<ArchiveRequest> = Vec::new();

    for file in &group.files {
        match &file.source {
            Source::File { url } => ensure_file(file, root, url)?,
            Source::Archive {
                url, size, sha256, ..
            } => match requests.iter_mut().find(|request| request.url == url) {
                Some(request) => request.files.push(file),
                None => requests.push(ArchiveRequest {
                    url,
                    size: *size,
                    sha256,
                    files: vec![file],
                }),
            },
        }
    }

    for request in requests {
        let mut needed = Vec::new();
        for file in request.files {
            let destination = root.join(&file.path);
            if destination.is_file() {
                if verify_file(root, file).is_ok() {
                    continue;
                }
                log(&format!("deleting the corrupt model file {}", file.path));
                let _ = fs::remove_file(&destination);
            }
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)
                    .with_context(|| format!("cannot create {}", parent.display()))?;
            }
            needed.push(file);
        }
        if needed.is_empty() {
            continue;
        }
        let archive = archive_for(request.url, request.size, request.sha256, work, archives)?;
        extract_members(&archive, &needed, root)?;
    }
    Ok(())
}

fn ensure_file(file: &ModelFile, root: &Path, url: &str) -> Result<()> {
    let destination = root.join(&file.path);
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
    download(url, &destination, file.size, &file.sha256, &file.path)
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
    if path.is_file() {
        if sha256_file(&path)? == sha256.to_lowercase() {
            archives.insert(url.to_string(), path.clone());
            return Ok(path);
        }
        let _ = fs::remove_file(&path);
    }
    download(url, &path, size, sha256, name)?;
    archives.insert(url.to_string(), path.clone());
    Ok(path)
}

/// Downloads one file into `destination`, resuming from a `.part` when the
/// server answers a `Range` request with `206`. A completed file is hashed once;
/// on a mismatch the partial file is deleted. A network drop keeps it so the
/// next install can continue.
fn download(url: &str, destination: &Path, size: u64, sha256: &str, label: &str) -> Result<()> {
    let part = with_part_suffix(destination);
    let mut start = fs::metadata(&part).map(|meta| meta.len()).unwrap_or(0);
    if start > size {
        let _ = fs::remove_file(&part);
        start = 0;
    }
    if start == size && start > 0 {
        if sha256_file(&part)? == sha256.to_lowercase() {
            fs::rename(&part, destination)
                .with_context(|| format!("cannot move {} into place", destination.display()))?;
            return Ok(());
        }
        let _ = fs::remove_file(&part);
        start = 0;
    }

    let agent = ureq::AgentBuilder::new()
        .timeout_connect(CONNECT_TIMEOUT)
        .timeout_read(READ_TIMEOUT)
        .build();
    let response =
        fetch(&agent, url, start).with_context(|| format!("cannot download {label} from {url}"))?;
    let append = start > 0 && response.status() == 206;
    let offset = if append { start } else { 0 };

    let mut writer = if append {
        fs::OpenOptions::new()
            .append(true)
            .open(&part)
            .with_context(|| format!("cannot write {}", part.display()))?
    } else {
        File::create(&part).with_context(|| format!("cannot write {}", part.display()))?
    };
    let mut reader = response.into_reader().take(size - offset + 1);
    let mut total = offset;
    let mut buffer = vec![0_u8; COPY_BLOCK];

    let outcome = (|| -> Result<()> {
        loop {
            let read = reader.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            total += read as u64;
            if total > size {
                bail!("the download of {label} is larger than its pinned size");
            }
            writer.write_all(&buffer[..read])?;
        }
        writer.flush()?;
        Ok(())
    })();
    drop(writer);

    if let Err(error) = outcome {
        return Err(error).with_context(|| format!("cannot download {label}"));
    }
    if total != size {
        bail!(
            "the download of {label} stopped at {total} of {size} bytes. Install the models again to resume."
        );
    }
    if sha256_file(&part)? != sha256.to_lowercase() {
        let _ = fs::remove_file(&part);
        bail!("model file does not match its pinned hash: {label}. Install the models again.");
    }
    fs::rename(&part, destination)
        .with_context(|| format!("cannot move {} into place", destination.display()))?;
    Ok(())
}

fn fetch(agent: &ureq::Agent, url: &str, start: u64) -> Result<ureq::Response> {
    if start == 0 {
        return Ok(agent.get(url).call()?);
    }
    match agent
        .get(url)
        .set("Range", &format!("bytes={start}-"))
        .call()
    {
        Ok(response) => Ok(response),
        // The server rejected the range (a 416 after a truncated response, for
        // example), so start the file from zero.
        Err(ureq::Error::Status(416, _)) => Ok(agent.get(url).call()?),
        Err(error) => Err(error.into()),
    }
}

/// Extracts every needed member of one archive in a single bz2 pass. Each file
/// is written to `.part` and verified while it streams; a failure removes it.
fn extract_members(archive: &Path, files: &[&ModelFile], root: &Path) -> Result<()> {
    let file = File::open(archive)
        .with_context(|| format!("cannot open the model archive {}", archive.display()))?;
    let mut tar = tar::Archive::new(BzDecoder::new(file));
    let mut pending: Vec<&ModelFile> = files.to_vec();

    for entry in tar.entries().context("cannot read the model archive")? {
        let mut entry = entry.context("cannot read the model archive")?;
        let entry_path = entry.path().context("cannot read a model archive path")?;
        let position = pending
            .iter()
            .position(|file| &*entry_path == Path::new(archive_member(file)));
        let Some(position) = position else {
            continue;
        };
        let file = pending.swap_remove(position);
        write_verified(
            &mut entry,
            &root.join(&file.path),
            file.size,
            &file.sha256,
            &file.path,
        )?;
        if pending.is_empty() {
            break;
        }
    }

    if let Some(file) = pending.first() {
        bail!(
            "the model archive is missing {}. Install the models again.",
            archive_member(file)
        );
    }
    Ok(())
}

fn archive_member(file: &ModelFile) -> &str {
    match &file.source {
        Source::Archive { member, .. } => member,
        Source::File { .. } => &file.path,
    }
}

/// Writes a reader to `destination` through `.part`, hashes it while it streams,
/// and renames it into place. Any failure removes the `.part` file.
fn write_verified<R: Read>(
    reader: R,
    destination: &Path,
    size: u64,
    sha256: &str,
    label: &str,
) -> Result<()> {
    let part = with_part_suffix(destination);
    let outcome = (|| -> Result<()> {
        let mut writer =
            File::create(&part).with_context(|| format!("cannot write {}", part.display()))?;
        let mut reader = reader.take(size + 1);
        let mut hasher = Sha256::new();
        let mut total = 0_u64;
        let mut buffer = vec![0_u8; COPY_BLOCK];

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
        drop(writer);

        if total != size {
            bail!("model file does not match its pinned size: {label}. Install the models again.");
        }
        if format!("{:x}", hasher.finalize()) != sha256.to_lowercase() {
            bail!("model file does not match its pinned hash: {label}. Install the models again.");
        }
        fs::rename(&part, destination)
            .with_context(|| format!("cannot move {} into place", destination.display()))?;
        Ok(())
    })();
    if outcome.is_err() {
        let _ = fs::remove_file(&part);
    }
    outcome
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
    use std::io::BufRead;

    const HELLO_SHA: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";
    const HELLO_WORLD_SHA: &str =
        "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9";
    const HELLO_WORLD_JOINED_SHA: &str =
        "936a185caaa266bb9cbe981e9e05cb78cd732b0b3280eb944412bb6f8f8f07af";

    fn archived(path: &str, member: &str, size: u64, sha256: &str) -> ModelFile {
        ModelFile {
            role: "test".to_string(),
            path: path.to_string(),
            size,
            sha256: sha256.to_string(),
            source: Source::Archive {
                url: "http://127.0.0.1:9/models.tar.bz2".to_string(),
                size: 0,
                sha256: String::new(),
                member: member.to_string(),
            },
        }
    }

    fn write_archive(path: &Path, members: &[(&str, &[u8])]) {
        let file = File::create(path).unwrap();
        let encoder = BzEncoder::new(file, Compression::best());
        let mut builder = tar::Builder::new(encoder);
        for (member, data) in members {
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder.append_data(&mut header, member, *data).unwrap();
        }
        builder.into_inner().unwrap().finish().unwrap();
    }

    #[test]
    fn extracts_every_needed_member_in_one_pass() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("models.tar.bz2");
        write_archive(
            &archive,
            &[("bundle/a.bin", b"hello"), ("bundle/b.bin", b"hello world")],
        );
        let root = dir.path().join("out");
        fs::create_dir_all(&root).unwrap();
        let first = archived("a.bin", "bundle/a.bin", 5, HELLO_SHA);
        let second = archived("b.bin", "bundle/b.bin", 11, HELLO_WORLD_SHA);

        extract_members(&archive, &[&first, &second], &root).unwrap();
        assert_eq!(fs::read(root.join("a.bin")).unwrap(), b"hello");
        assert_eq!(fs::read(root.join("b.bin")).unwrap(), b"hello world");
    }

    #[test]
    fn rejects_a_member_with_the_wrong_hash_and_removes_the_part() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("models.tar.bz2");
        write_archive(&archive, &[("bundle/model.bin", b"jello")]);
        let root = dir.path().join("out");
        fs::create_dir_all(&root).unwrap();
        let file = archived("model.bin", "bundle/model.bin", 5, HELLO_SHA);

        let error = extract_members(&archive, &[&file], &root)
            .unwrap_err()
            .to_string();
        assert!(error.contains("does not match its pinned hash"), "{error}");
        assert!(!root.join("model.bin").exists());
        assert!(!with_part_suffix(&root.join("model.bin")).exists());
    }

    #[test]
    fn reuses_a_complete_part_file_without_the_network() {
        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("model.bin");
        fs::write(with_part_suffix(&destination), b"hello").unwrap();

        download(
            "http://127.0.0.1:9/unreachable",
            &destination,
            5,
            HELLO_SHA,
            "model.bin",
        )
        .unwrap();
        assert_eq!(fs::read(&destination).unwrap(), b"hello");
        assert!(!with_part_suffix(&destination).exists());
    }

    #[test]
    fn resumes_a_partial_download_with_a_range_request() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut reader = std::io::BufReader::new(stream.try_clone().unwrap());
            let mut request = String::new();
            loop {
                let mut line = String::new();
                reader.read_line(&mut line).unwrap();
                if line.is_empty() || line == "\r\n" {
                    break;
                }
                request.push_str(&line);
            }
            assert!(
                request.to_lowercase().contains("range: bytes=5-"),
                "{request}"
            );
            stream
                .write_all(
                    b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\nConnection: close\r\n\r\nworld",
                )
                .unwrap();
        });

        let dir = tempfile::tempdir().unwrap();
        let destination = dir.path().join("model.bin");
        fs::write(with_part_suffix(&destination), b"hello").unwrap();
        let url = format!("http://{address}/model.bin");

        download(&url, &destination, 10, HELLO_WORLD_JOINED_SHA, "model.bin").unwrap();
        server.join().unwrap();
        assert_eq!(fs::read(&destination).unwrap(), b"helloworld");
    }

    #[test]
    fn reports_a_missing_member() {
        let dir = tempfile::tempdir().unwrap();
        let archive = dir.path().join("models.tar.bz2");
        write_archive(&archive, &[("bundle/other.bin", b"hello")]);
        let root = dir.path().join("out");
        let file = archived("model.bin", "bundle/model.bin", 5, HELLO_SHA);

        let error = extract_members(&archive, &[&file], &root)
            .unwrap_err()
            .to_string();
        assert!(error.contains("is missing bundle/model.bin"), "{error}");
        assert!(!root.join("model.bin").exists());
    }
}
