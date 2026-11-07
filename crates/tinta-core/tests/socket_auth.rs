//! The app socket accepts a connection only from one of the allowed helper binaries of this
//! user. These tests open a real Unix socket, connect to it from the test binary, and run the
//! real check on the server side. The executable of the test binary is not a helper, so a test
//! that names the test binary as allowed shows the accepted case, and the other tests show the
//! refused cases.

#![cfg(target_os = "linux")]

use std::io::{BufRead, BufReader};
use std::os::unix::io::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Duration;

/// The socket that the child process connects to in `refuses_a_peer_whose_executable_was_deleted`.
const CHILD_SOCKET: &str = "TINTA_TEST_PEER_SOCKET";

/// Connects to a listener that runs the peer check with `allowed`, and returns whether the
/// listener accepted the connection.
fn accepted(allowed: Vec<PathBuf>) -> bool {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tinta.sock");
    let listener = UnixListener::bind(&path).unwrap();
    let server = std::thread::spawn(move || {
        let (stream, _) = listener.accept().unwrap();
        tinta_core::peer::authenticate(stream.as_raw_fd(), &allowed).is_some()
    });
    // The connection stays open until the server has checked it.
    let _client = UnixStream::connect(&path).unwrap();
    server.join().unwrap()
}

#[test]
fn accepts_the_same_user_running_an_allowed_binary() {
    let test_binary = std::env::current_exe().unwrap().canonicalize().unwrap();
    assert!(accepted(vec![test_binary]));
}

#[test]
fn refuses_a_binary_that_is_not_allowed() {
    let dir = tempfile::tempdir().unwrap();
    let helper = dir.path().join("tinta-native-host");
    std::fs::write(&helper, b"not the peer").unwrap();
    assert!(!accepted(vec![helper]));
}

#[test]
fn refuses_when_nothing_is_allowed() {
    assert!(!accepted(Vec::new()));
}

/// The child half of `refuses_a_peer_whose_executable_was_deleted`: connects to the socket
/// from `CHILD_SOCKET` and waits until the parent closes it.
#[test]
fn peer_child_connects() {
    let Ok(path) = std::env::var(CHILD_SOCKET) else {
        return;
    };
    let stream = UnixStream::connect(path).unwrap();
    let mut line = String::new();
    let _ = BufReader::new(stream).read_line(&mut line);
}

#[test]
fn refuses_a_peer_whose_executable_was_deleted() {
    let dir = tempfile::tempdir().unwrap();
    // A second link keeps the inode alive, so only the "(deleted)" marker can refuse it.
    let copy = dir.path().join("peer");
    let allowed = dir.path().join("tinta-mcp");
    std::fs::copy(std::env::current_exe().unwrap(), &copy).unwrap();
    std::fs::hard_link(&copy, &allowed).unwrap();
    let path = dir.path().join("tinta.sock");
    let listener = UnixListener::bind(&path).unwrap();
    listener.set_nonblocking(true).unwrap();
    let mut child = Command::new(&copy)
        .args(["--exact", "peer_child_connects"])
        .env(CHILD_SOCKET, &path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let stream = loop {
        match listener.accept() {
            Ok((stream, _)) => break stream,
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if std::time::Instant::now() > deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    panic!("the child did not connect");
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            Err(error) => panic!("cannot accept the child connection: {error}"),
        }
    };
    let exe = format!("/proc/{}/exe", child.id());
    // Wait until the child runs the copy, then replace the executable on disk.
    for _ in 0..200 {
        if std::fs::read_link(&exe).ok().as_deref() == Some(copy.as_path()) {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    std::fs::remove_file(&copy).unwrap();
    assert!(std::fs::read_link(&exe)
        .unwrap()
        .to_string_lossy()
        .ends_with(" (deleted)"));
    assert!(tinta_core::peer::authenticate(stream.as_raw_fd(), &[allowed]).is_none());
    drop(stream);
    let _ = child.wait();
}
