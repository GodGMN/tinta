//! Authentication of the helper processes that connect to the app socket on Linux.
//!
//! The app accepts a connection only when the peer runs as the same user and the process
//! runs the same file as one of the installed helper binaries. `SO_PEERCRED` gives the user
//! ID and the process ID that the kernel recorded at `connect()`. `/proc/<pid>/exe` is the
//! kernel's link to the file that the process runs, so the check compares the file identity
//! (device and inode) of that link with the allowed helper files instead of resolving a path
//! again.
//!
//! This answers "is this the helper build next to the app" and nothing more. A process that
//! already runs as this user can exec the real helper after it connects, inject into it with
//! `LD_PRELOAD` or `ptrace`, or just read the data folder. The same user is the trust
//! boundary, as on many Linux desktop apps.
//!
//! The values that the decision needs are arguments, so that the decision has tests and so
//! that the integration test can run the same check with its own list of allowed paths.

use std::os::unix::io::RawFd;
use std::path::{Path, PathBuf};

/// The helper binaries that the app trusts, in the folder of the app executable. The Cargo
/// target directory is the same folder in development.
pub fn helpers_in(dir: &Path) -> Vec<PathBuf> {
    ["tinta-native-host", "tinta-mcp"]
        .iter()
        .map(|name| dir.join(name))
        .collect()
}

/// Whether a peer may use the app socket: it runs as the app user, and its executable is one
/// of `allowed`. `app_uid` is the effective user ID of the app.
pub fn is_trusted(peer_uid: u32, app_uid: u32, peer_exe: &Path, allowed: &[PathBuf]) -> bool {
    peer_uid == app_uid && allowed_exe(peer_exe, allowed)
}

/// Whether the file at `peer_exe` is the same file as one of `allowed`: same device and
/// inode. A different file at an allowed path, or an old file that an update replaced, does
/// not pass.
pub fn allowed_exe(peer_exe: &Path, allowed: &[PathBuf]) -> bool {
    use std::os::unix::fs::MetadataExt;
    let identity = |path: &Path| {
        std::fs::metadata(path)
            .ok()
            .filter(|meta| meta.is_file())
            .map(|meta| (meta.dev(), meta.ino()))
    };
    let Some(peer) = identity(peer_exe) else {
        return false;
    };
    allowed.iter().any(|path| identity(path) == Some(peer))
}

/// The effective user ID of this process.
pub fn current_uid() -> u32 {
    // SAFETY: `geteuid` has no arguments and no failure that returns memory.
    unsafe { libc::geteuid() }
}

/// The user ID and process ID that the kernel recorded at `connect()` on `fd`.
fn peer_credentials(fd: RawFd) -> Option<(u32, libc::pid_t)> {
    let mut credentials = libc::ucred {
        pid: 0,
        uid: 0,
        gid: 0,
    };
    let mut length = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: the socket is open, and `credentials` and `length` are the sizes that
    // `SO_PEERCRED` writes.
    let status = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_PEERCRED,
            &mut credentials as *mut _ as *mut libc::c_void,
            &mut length,
        )
    };
    if status != 0 || length as usize != std::mem::size_of::<libc::ucred>() || credentials.pid <= 0
    {
        return None;
    }
    Some((credentials.uid, credentials.pid))
}

/// A descriptor that pins the connecting process, so that its pid cannot be reused between
/// the credential read and the executable check. Kernels before 6.5 do not offer
/// `SO_PEERPIDFD`; there the check keeps the pid from `SO_PEERCRED`.
enum PidFd {
    Pinned(RawFd),
    Unsupported,
    Failed,
}

fn peer_pidfd(fd: RawFd) -> PidFd {
    let mut pidfd: libc::c_int = -1;
    let mut length = std::mem::size_of::<libc::c_int>() as libc::socklen_t;
    // SAFETY: the socket is open, and `pidfd` and `length` are the sizes that `SO_PEERPIDFD`
    // writes.
    let status = unsafe {
        libc::getsockopt(
            fd,
            libc::SOL_SOCKET,
            libc::SO_PEERPIDFD,
            &mut pidfd as *mut _ as *mut libc::c_void,
            &mut length,
        )
    };
    if status == 0 && length as usize == std::mem::size_of::<libc::c_int>() {
        return if pidfd >= 0 {
            PidFd::Pinned(pidfd)
        } else {
            PidFd::Failed
        };
    }
    if std::io::Error::last_os_error().raw_os_error() == Some(libc::ENOPROTOOPT) {
        PidFd::Unsupported
    } else {
        PidFd::Failed
    }
}

/// Whether the process that `pidfd` pins still exists. Signal 0 sends nothing and only asks
/// the kernel whether the process is there.
fn pidfd_alive(pidfd: RawFd) -> bool {
    // SAFETY: `pidfd` is an open pid file descriptor, and a null `siginfo` is allowed for
    // signal 0.
    unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal,
            pidfd as libc::c_long,
            0 as libc::c_long,
            std::ptr::null::<libc::siginfo_t>(),
            0 as libc::c_long,
        ) == 0
    }
}

/// Whether the kernel's link to the process executable reports a file that no longer exists.
fn replaced_on_disk(link: &Path) -> bool {
    link.as_os_str().to_string_lossy().ends_with(" (deleted)")
}

/// The full check for a connected socket: the peer runs as this user and its executable is
/// one of `allowed`. Returns the executable, or `None` when the app must refuse the
/// connection.
pub fn authenticate(fd: RawFd, allowed: &[PathBuf]) -> Option<PathBuf> {
    let (uid, pid) = peer_credentials(fd)?;
    let proc_exe = PathBuf::from(format!("/proc/{pid}/exe"));
    let exe = std::fs::read_link(&proc_exe).ok()?;
    // A helper that an update replaced reads "... (deleted)" and is refused: the running
    // old helper gets the refusal until the browser or the MCP client restarts it.
    if replaced_on_disk(&exe) {
        return None;
    }
    // Stat the kernel's link, which follows it to the running inode, and compare that
    // identity with the allowed files.
    if !is_trusted(uid, current_uid(), &proc_exe, allowed) {
        return None;
    }
    // Check the pinned process after reading the executable, so a pid that was reused in
    // the meantime cannot answer.
    match peer_pidfd(fd) {
        PidFd::Unsupported => {}
        PidFd::Pinned(pidfd) => {
            let alive = pidfd_alive(pidfd);
            // SAFETY: `pidfd` is the open descriptor that `SO_PEERPIDFD` returned.
            unsafe { libc::close(pidfd) };
            if !alive {
                return None;
            }
        }
        PidFd::Failed => return None,
    }
    Some(exe)
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
        assert!(is_trusted(
            1000,
            1000,
            &helper,
            std::slice::from_ref(&helper)
        ));
    }

    #[test]
    fn refuses_another_user_even_with_an_allowed_helper() {
        let dir = tempfile::tempdir().unwrap();
        let helper = file(dir.path(), "tinta-mcp");
        assert!(!is_trusted(
            1001,
            1000,
            &helper,
            std::slice::from_ref(&helper)
        ));
    }

    #[test]
    fn refuses_an_executable_that_is_not_a_helper() {
        let dir = tempfile::tempdir().unwrap();
        let helper = file(dir.path(), "tinta-mcp");
        let other = file(dir.path(), "bash");
        assert!(!is_trusted(1000, 1000, &other, &[helper]));
    }

    #[test]
    fn refuses_a_different_file_with_the_same_name() {
        let dir = tempfile::tempdir().unwrap();
        let allowed = file(dir.path(), "tinta-mcp");
        let copy = dir.path().join("elsewhere");
        std::fs::copy(&allowed, &copy).unwrap();
        assert!(!allowed_exe(&copy, &[allowed]));
    }

    #[test]
    fn accepts_a_hard_link_to_an_allowed_helper() {
        let dir = tempfile::tempdir().unwrap();
        let helper = file(dir.path(), "tinta-mcp");
        let link = dir.path().join("tinta-native-host");
        std::fs::hard_link(&helper, &link).unwrap();
        assert!(allowed_exe(&link, std::slice::from_ref(&helper)));
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
    fn refuses_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().join("tinta-mcp");
        std::fs::create_dir(&folder).unwrap();
        assert!(!allowed_exe(&folder, std::slice::from_ref(&folder)));
    }

    #[test]
    fn recognizes_the_deleted_marker_of_a_replaced_helper() {
        assert!(replaced_on_disk(Path::new(
            "/opt/tinta/tinta-mcp (deleted)"
        )));
        assert!(!replaced_on_disk(Path::new("/opt/tinta/tinta-mcp")));
    }

    #[test]
    fn allows_the_helpers_next_to_the_app() {
        let dir = std::path::Path::new("/opt/tinta");
        assert_eq!(
            helpers_in(dir),
            vec![dir.join("tinta-native-host"), dir.join("tinta-mcp")]
        );
    }
}
