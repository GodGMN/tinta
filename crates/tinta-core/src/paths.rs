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
///
/// On Linux the socket lives in a folder of its own, so that the helper binaries talk to the
/// app through a folder that only this user can enter. The session runtime directory is
/// private to the user and is removed when the session ends; without one, the data folder is
/// the fallback. Tests keep the socket in their own data folder.
pub fn socket_path() -> PathBuf {
    socket_dir().join("tinta.sock")
}

/// The folder of the app socket. It must be private to the user on Linux; see
/// [`prepare_socket_dir`]. macOS keeps the socket in the data folder, as before.
///
/// The app and both helpers call this one function, so they always agree on the path. When
/// `XDG_RUNTIME_DIR` is unset or relative, the folder of the user session, `/run/user/<uid>`,
/// is used when it exists and belongs to this user; otherwise the data folder is the
/// fallback. A helper started without `XDG_RUNTIME_DIR` (ssh, cron, a system unit) therefore
/// still finds the socket that the desktop app listens on.
pub fn socket_dir() -> PathBuf {
    #[cfg(target_os = "linux")]
    if std::env::var_os("TINTA_DATA_DIR").is_none() {
        let uid = current_uid();
        let runtime = std::env::var_os("XDG_RUNTIME_DIR").map(PathBuf::from);
        if let Some(dir) = runtime_socket_dir(runtime.as_deref(), uid, |dir| owned_by(dir, uid)) {
            return dir;
        }
    }
    base_dir()
}

/// The socket folder for a session runtime directory such as `/run/user/1000`. When the
/// variable is unset or holds a relative path, the default session folder is used if
/// `owned_by` accepts it. `None` sends the caller to the data folder.
#[cfg(target_os = "linux")]
fn runtime_socket_dir(
    runtime: Option<&Path>,
    uid: u32,
    owned_by: impl Fn(&Path) -> bool,
) -> Option<PathBuf> {
    if let Some(dir) = runtime.filter(|dir| dir.is_absolute()) {
        return Some(dir.join("tinta"));
    }
    let session = PathBuf::from(format!("/run/user/{uid}"));
    owned_by(&session).then(|| session.join("tinta"))
}

/// Whether `dir` is a real folder that belongs to `uid`.
#[cfg(target_os = "linux")]
fn owned_by(dir: &Path, uid: u32) -> bool {
    use std::os::unix::fs::MetadataExt;
    std::fs::metadata(dir)
        .map(|meta| meta.is_dir() && meta.uid() == uid)
        .unwrap_or(false)
}

/// The effective user ID of this process.
#[cfg(target_os = "linux")]
fn current_uid() -> u32 {
    // SAFETY: `geteuid` has no arguments and no failure that returns memory.
    unsafe { libc::geteuid() }
}

/// Creates the folder of the app socket and makes it reachable only by this user (0700 on
/// Linux). Called before the app binds the socket. It fails closed when the folder exists
/// with other permissions, or is not a real folder that this user owns.
pub fn prepare_socket_dir() -> Result<PathBuf> {
    let dir = socket_dir();
    #[cfg(target_os = "linux")]
    prepare_private_dir(&dir)?;
    Ok(dir)
}

/// Creates the folder with mode 0700 and makes sure that no other user can enter the folder
/// that holds the socket. No chmod follows: an existing folder with other permissions is an
/// error instead of a folder that another user could already have entered.
#[cfg(target_os = "linux")]
fn prepare_private_dir(dir: &Path) -> Result<()> {
    use std::io::ErrorKind;
    use std::os::unix::fs::{DirBuilderExt, MetadataExt};

    // The parent is the session runtime folder or the data folder, which exist already or
    // which the app creates for its own files.
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("cannot create {}", parent.display()))?;
    }
    let mut builder = std::fs::DirBuilder::new();
    builder.mode(0o700);
    match builder.create(dir) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {}
        Err(error) => {
            return Err(error).with_context(|| format!("cannot create {}", dir.display()));
        }
    }
    // lstat, so a symbolic link is not followed to a folder that this user may not own.
    let meta = std::fs::symlink_metadata(dir)
        .with_context(|| format!("cannot inspect {}", dir.display()))?;
    if !meta.file_type().is_dir() {
        bail!("{} is not a folder", dir.display());
    }
    if meta.uid() != current_uid() {
        bail!("{} belongs to another user", dir.display());
    }
    if meta.mode() & 0o777 != 0o700 {
        bail!(
            "{} is not private to this user (mode {:o})",
            dir.display(),
            meta.mode() & 0o777
        );
    }
    Ok(())
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

    #[cfg(target_os = "linux")]
    #[test]
    fn socket_folder_lives_under_the_session_runtime_dir() {
        let dir = runtime_socket_dir(Some(Path::new("/run/user/1000")), 1000, |_| false);
        assert_eq!(dir, Some(PathBuf::from("/run/user/1000/tinta")));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn socket_folder_uses_the_default_session_dir_when_no_runtime_dir_is_set() {
        assert_eq!(
            runtime_socket_dir(None, 1000, |dir| dir == Path::new("/run/user/1000")),
            Some(PathBuf::from("/run/user/1000/tinta"))
        );
        assert_eq!(runtime_socket_dir(None, 1000, |_| false), None);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn socket_folder_treats_a_relative_runtime_dir_as_unset() {
        let owned = |dir: &Path| dir == Path::new("/run/user/1000");
        assert_eq!(
            runtime_socket_dir(Some(Path::new("run/user/1000")), 1000, owned),
            Some(PathBuf::from("/run/user/1000/tinta"))
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn socket_folder_checks_that_the_session_dir_belongs_to_the_user() {
        assert!(owned_by(Path::new("."), current_uid()));
        assert!(!owned_by(Path::new("."), current_uid() + 1));
        assert!(!owned_by(
            Path::new("/run/user/does-not-exist-here"),
            current_uid()
        ));
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn socket_folder_is_created_private_to_the_user() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().join("tinta");
        prepare_private_dir(&folder).unwrap();
        let mode = std::fs::metadata(&folder).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o700);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn socket_folder_refuses_a_folder_that_others_can_enter() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().join("tinta");
        std::fs::create_dir(&folder).unwrap();
        std::fs::set_permissions(&folder, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert!(prepare_private_dir(&folder).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn socket_folder_refuses_a_symbolic_link() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("elsewhere");
        std::fs::create_dir(&target).unwrap();
        let link = dir.path().join("tinta");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(prepare_private_dir(&link).is_err());
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn socket_folder_refuses_a_file_in_its_place() {
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().join("tinta");
        std::fs::write(&folder, b"not a folder").unwrap();
        assert!(prepare_private_dir(&folder).is_err());
    }
}
