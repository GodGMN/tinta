//! The pinned model set. Model choices can change after quality work, so the
//! names, URLs, sizes and hashes live in `assets/models.json`, not in code.

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

const EMBEDDED: &str = include_str!("../assets/models.json");
const MANIFEST_VERSION: u32 = 1;
const HASH_BLOCK: usize = 4 * 1024 * 1024;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Manifest {
    pub schema: u32,
    pub groups: Vec<ModelGroup>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelGroup {
    /// The role the group plays in the pipeline: transcription, vad or diarization.
    pub component: String,
    pub files: Vec<ModelFile>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelFile {
    /// The file's role inside its component, such as encoder or embedding.
    pub role: String,
    /// Path relative to the models root.
    pub path: String,
    /// Size of the installed file.
    pub size: u64,
    /// Lowercase hex SHA-256 of the installed file.
    pub sha256: String,
    pub source: Source,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum Source {
    /// One file served as-is.
    File { url: String },
    /// A release archive that contains the file. The archive itself is verified
    /// before its member is extracted.
    Archive {
        url: String,
        size: u64,
        sha256: String,
        member: String,
    },
}

impl Manifest {
    /// The manifest compiled into the binary.
    pub fn embedded() -> Self {
        Self::parse(EMBEDDED).expect("the embedded model manifest is valid")
    }

    pub fn parse(text: &str) -> Result<Self> {
        let manifest: Self =
            serde_json::from_str(text).context("cannot parse the model manifest")?;
        if manifest.schema != MANIFEST_VERSION {
            bail!(
                "the model manifest schema is {}, expected {}",
                manifest.schema,
                MANIFEST_VERSION
            );
        }
        Ok(manifest)
    }

    /// The cheap check `models_status` uses: file existence only, like the
    /// Swift engine. No model is loaded and no file is hashed.
    pub fn all_files_exist(&self, root: &Path) -> bool {
        self.groups
            .iter()
            .flat_map(|group| &group.files)
            .all(|file| root.join(&file.path).is_file())
    }

    pub fn group(&self, component: &str) -> Result<&ModelGroup> {
        self.groups
            .iter()
            .find(|group| group.component == component)
            .with_context(|| format!("the model manifest has no {component} group"))
    }

    pub fn total_size(&self) -> u64 {
        self.groups
            .iter()
            .flat_map(|group| &group.files)
            .map(|file| file.size)
            .sum()
    }
}

impl ModelGroup {
    pub fn file(&self, role: &str) -> Result<&ModelFile> {
        self.files
            .iter()
            .find(|file| file.role == role)
            .with_context(|| format!("the {} model manifest has no {role} file", self.component))
    }
}

/// Hashes one installed file and compares it to the pin. A file that does not
/// match is deleted so the next install downloads it again.
pub fn verify_file(root: &Path, file: &ModelFile) -> Result<()> {
    let path = root.join(&file.path);
    if !path.is_file() {
        bail!(
            "model file missing: {}. Install the models again.",
            file.path
        );
    }
    let digest = sha256_file(&path).with_context(|| format!("cannot read {}", file.path))?;
    if digest != file.sha256.to_lowercase() {
        let _ = std::fs::remove_file(&path);
        bail!(
            "model file does not match its pinned hash: {}. Install the models again.",
            file.path
        );
    }
    Ok(())
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::with_capacity(HASH_BLOCK, File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; HASH_BLOCK];
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// A copy of the effective model path, for the reply and for sherpa-onnx.
pub fn installed_path(root: &Path, file: &ModelFile) -> PathBuf {
    root.join(&file.path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    const HELLO_SHA: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    fn tiny_manifest() -> Manifest {
        Manifest::parse(&format!(
            r#"{{
              "schema": 1,
              "groups": [
                {{
                  "component": "transcription",
                  "files": [
                    {{
                      "role": "encoder",
                      "path": "tiny/model.bin",
                      "size": 5,
                      "sha256": "{HELLO_SHA}",
                      "source": {{ "kind": "file", "url": "http://127.0.0.1:9/model.bin" }}
                    }}
                  ]
                }}
              ]
            }}"#
        ))
        .unwrap()
    }

    #[test]
    fn embedded_manifest_is_valid_and_pinned() {
        let manifest = Manifest::embedded();
        assert_eq!(manifest.groups.len(), 3);
        assert!(manifest.total_size() > 690_000_000);
        for group in &manifest.groups {
            for file in &group.files {
                assert_eq!(file.sha256.len(), 64);
                assert!(file.size > 0);
                assert!(file.sha256.chars().all(|c| c.is_ascii_hexdigit()));
            }
        }
    }

    #[test]
    fn accepts_a_file_that_matches_its_pin() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = tiny_manifest();
        let file = manifest
            .group("transcription")
            .unwrap()
            .file("encoder")
            .unwrap();
        let path = dir.path().join(&file.path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"hello").unwrap();

        verify_file(dir.path(), file).unwrap();
        assert!(path.is_file());
    }

    #[test]
    fn rejects_and_deletes_a_corrupt_file() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = tiny_manifest();
        let file = manifest
            .group("transcription")
            .unwrap()
            .file("encoder")
            .unwrap();
        let path = dir.path().join(&file.path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut corrupt = fs::File::create(&path).unwrap();
        corrupt.write_all(b"jello").unwrap();
        drop(corrupt);

        let error = verify_file(dir.path(), file).unwrap_err().to_string();
        assert!(error.contains("does not match its pinned hash"), "{error}");
        assert!(!path.exists(), "the corrupt file is deleted");
    }

    #[test]
    fn missing_file_says_to_install_again() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = tiny_manifest();
        let file = manifest
            .group("transcription")
            .unwrap()
            .file("encoder")
            .unwrap();

        let error = verify_file(dir.path(), file).unwrap_err().to_string();
        assert!(error.contains("model file missing"), "{error}");
        assert!(error.contains("Install the models again"), "{error}");
    }

    #[test]
    fn existence_check_ignores_hashes() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = tiny_manifest();
        let file = manifest
            .group("transcription")
            .unwrap()
            .file("encoder")
            .unwrap();
        assert!(!manifest.all_files_exist(dir.path()));

        let path = dir.path().join(&file.path);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(&path, b"jello").unwrap();
        assert!(manifest.all_files_exist(dir.path()));
    }
}
