use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::sync::RwLock;

/// The fixed app folder. It holds `config.json` and the app socket, so the helper binaries
/// always find the app. `TINTA_DATA_DIR` overrides it for tests.
pub fn base_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("TINTA_DATA_DIR") {
        return PathBuf::from(dir);
    }
    dirs::data_dir().unwrap_or_else(|| PathBuf::from(".")).join("Tinta")
}

fn config_path() -> PathBuf {
    base_dir().join("config.json")
}

static LIBRARY: RwLock<Option<PathBuf>> = RwLock::new(None);

/// The folder of the library and the audio. The user can move it. The default is `base_dir()`.
pub fn data_dir() -> PathBuf {
    if let Some(dir) = LIBRARY.read().unwrap().clone() {
        return dir;
    }
    let dir = std::fs::read_to_string(config_path())
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|v| v["library_dir"].as_str().map(PathBuf::from))
        .unwrap_or_else(base_dir);
    *LIBRARY.write().unwrap() = Some(dir.clone());
    dir
}

/// Records a new library folder. The caller moves the files first.
pub fn set_data_dir(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(base_dir())?;
    let config = serde_json::json!({ "library_dir": dir });
    std::fs::write(config_path(), serde_json::to_vec_pretty(&config)?).context("cannot write config.json")?;
    *LIBRARY.write().unwrap() = Some(dir.to_path_buf());
    Ok(())
}

/// The library must stay on this Mac, so cloud-synced folders are not allowed.
pub fn check_local(dir: &Path) -> Result<()> {
    let text = comparable_path(dir);
    let synced = ["/library/mobile documents", "/library/cloudstorage", "/dropbox", "/google drive", "/onedrive", "/box sync"];
    if synced.iter().any(|s| text.contains(s)) {
        bail!("This folder is in a cloud-synced location. Select a folder that stays on this Mac.");
    }
    Ok(())
}

/// The path in lowercase, with `/` separators so that Windows paths match too.
/// Absolute paths on macOS already start with `/` and are returned unchanged.
fn comparable_path(dir: &Path) -> String {
    let text = dir.to_string_lossy().to_lowercase();
    if text.starts_with('/') {
        text
    } else {
        text.replace('\\', "/")
    }
}

pub fn database_path() -> PathBuf {
    data_dir().join("library.db")
}

pub fn meeting_dir(meeting_id: &str) -> PathBuf {
    data_dir().join("meetings").join(meeting_id)
}

pub fn audio_dir(meeting_id: &str) -> PathBuf {
    meeting_dir(meeting_id).join("audio")
}

/// The Unix socket that the MCP binary and the Chrome native host use.
pub fn socket_path() -> PathBuf {
    base_dir().join("tinta.sock")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blocked(path: &str) -> bool {
        check_local(Path::new(path)).is_err()
    }

    #[test]
    fn blocks_cloud_folders_on_macos() {
        assert!(blocked("/Users/me/Library/CloudStorage/Dropbox/Tinta"));
        assert!(blocked("/Users/me/Library/Mobile Documents/com~apple~CloudDocs/Tinta"));
        assert!(blocked("/Users/me/Dropbox/Meetings"));
        assert!(blocked("/Users/me/Google Drive/Tinta"));
        assert!(blocked("/Users/me/Box Sync/Tinta"));
    }

    #[test]
    fn allows_local_folders_on_macos() {
        assert!(!blocked("/Users/me/Documents/Tinta"));
        assert!(!blocked("/Volumes/External/Tinta"));
    }

    #[test]
    fn blocks_cloud_folders_on_windows() {
        assert!(blocked(r"C:\Users\me\Dropbox\Tinta"));
        assert!(blocked(r"C:\Users\me\OneDrive\Tinta"));
        assert!(blocked(r"C:\Users\me\Google Drive\Tinta"));
        assert!(blocked(r"\\server\Box Sync\Tinta"));
    }

    #[test]
    fn allows_local_folders_on_windows() {
        assert!(!blocked(r"C:\Users\me\Documents\Tinta"));
        assert!(!blocked(r"\\server\share\Tinta"));
    }
}
