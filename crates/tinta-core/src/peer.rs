//! Authentication of the helper processes that connect to the app socket on Linux.
//!
//! The app accepts a connection only when the peer runs as the same user and the executable of
//! that process is one of the installed helper binaries. The kernel fills `SO_PEERCRED` with
//! the user ID and the process ID that made the connection, so a process cannot claim another
//! identity. The executable comes from `/proc/<pid>/exe`, which the kernel keeps for the
//! lifetime of the process, so a helper that replaces its file on disk cannot change the path
//! that the app sees.
//!
//! The values that the decision needs are arguments, so that the decision has tests and so that
//! the integration test can run the same check with its own list of allowed paths.

use std::path::{Path, PathBuf};

/// The helper binaries that the app trusts, in the folder of the app executable. The Cargo
/// target directory is the same folder in development.
pub fn helpers_in(dir: &Path) -> Vec<PathBuf> {
    ["tinta-native-host", "tinta-mcp"].iter().map(|name| dir.join(name)).collect()
}

/// Whether a peer may use the app socket: it runs as the app user, and its executable is one of
/// `allowed`. `app_uid` is the effective user ID of the app.
pub fn is_trusted(peer_uid: u32, app_uid: u32, peer_exe: &Path, allowed: &[PathBuf]) -> bool {
    peer_uid == app_uid && allowed_exe(peer_exe, allowed)
}

/// Whether the executable is one of `allowed`. Both sides are canonicalized, so a symlink or a
/// relative path does not pass. A path that cannot be resolved does not pass either.
pub fn allowed_exe(peer_exe: &Path, allowed: &[PathBuf]) -> bool {
    let Ok(peer) = peer_exe.canonicalize() else {
        return false;
    };
    allowed.iter().any(|allowed| allowed.canonicalize().map(|path| path == peer).unwrap_or(false))
}

/// The effective user ID of this process.
pub fn current_uid() -> u32 {
    // SAFETY: `geteuid` has no arguments and no failure that returns memory.
    unsafe { libc::geteuid() }
}

/// The user ID and the executable of the process at the other end of `fd`. `None` when the
/// kernel does not give the credentials, the process is gone, or `/proc` does not show its
/// executable, because the check fails closed.
pub fn peer_executable(fd: std::os::unix::io::RawFd) -> Option<(u32, PathBuf)> {
    let mut credentials = libc::ucred { pid: 0, uid: 0, gid: 0 };
    let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: the socket is open, and `credentials` and `length` are the sizes that `SO_PEERCRED` writes.
    let status = unsafe {
        libc::getsockopt(fd, libc::SOL_SOCKET, libc::SO_PEERCRED, &mut credentials as *mut _ as *mut libc::c_void, &mut length)
    };
    if status != 0 || length as usize != std::mem::size_of::<libc::ucred>() || credentials.pid <= 0 {
        return None;
    }
    let exe = std::fs::read_link(format!("/proc/{}/exe", credentials.pid)).ok()?.canonicalize().ok()?;
    Some((credentials.uid, exe))
}

/// The full check for a connected socket: the peer runs as this user and its executable is one
/// of `allowed`. Returns the executable, or `None` when the app must refuse the connection.
pub fn authenticate(fd: std::os::unix::io::RawFd, allowed: &[PathBuf]) -> Option<PathBuf> {
    let (uid, exe) = peer_executable(fd)?;
    is_trusted(uid, current_uid(), &exe, allowed).then_some(exe)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(dir: &Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, b"helper").unwrap();
        path
    }

    #[test]
    fn accepts_the_app_user_running_an_allowed_helper() {
        let dir = tempfile::tempdir().unwrap();
        let helper = file(dir.path(), "tinta-mcp");
        assert!(is_trusted(1000, 1000, &helper, std::slice::from_ref(&helper)));
    }

    #[test]
    fn refuses_another_user_even_with_an_allowed_helper() {
        let dir = tempfile::tempdir().unwrap();
        let helper = file(dir.path(), "tinta-mcp");
        assert!(!is_trusted(1001, 1000, &helper, std::slice::from_ref(&helper)));
    }

    #[test]
    fn refuses_an_executable_that_is_not_a_helper() {
        let dir = tempfile::tempdir().unwrap();
        let helper = file(dir.path(), "tinta-mcp");
        let other = file(dir.path(), "bash");
        assert!(!is_trusted(1000, 1000, &other, &[helper]));
    }

    #[test]
    fn refuses_when_no_helper_paths_are_allowed() {
        let dir = tempfile::tempdir().unwrap();
        let helper = file(dir.path(), "tinta-mcp");
        assert!(!is_trusted(1000, 1000, &helper, &[]));
    }

    #[test]
    fn follows_a_symlink_to_an_allowed_helper() {
        let dir = tempfile::tempdir().unwrap();
        let helper = file(dir.path(), "tinta-mcp");
        let link = dir.path().join("tinta-native-host");
        std::os::unix::fs::symlink(&helper, &link).unwrap();
        assert!(allowed_exe(&link, &[helper]));
    }

    #[test]
    fn refuses_a_path_that_does_not_exist() {
        let dir = tempfile::tempdir().unwrap();
        let helper = file(dir.path(), "tinta-mcp");
        assert!(!allowed_exe(&dir.path().join("gone"), &[helper]));
    }

    #[test]
    fn allows_the_helpers_next_to_the_app() {
        let dir = std::path::Path::new("/opt/tinta");
        assert_eq!(helpers_in(dir), vec![dir.join("tinta-native-host"), dir.join("tinta-mcp")]);
    }
}
