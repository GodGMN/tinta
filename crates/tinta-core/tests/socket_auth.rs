//! The app socket accepts a connection only from one of the allowed helper binaries of this
//! user. These tests open a real Unix socket, connect to it from the test binary, and run the
//! real check on the server side. The executable of the test binary is not a helper, so a test
//! that names the test binary as allowed shows the accepted case, and the other tests show the
//! refused cases.

#![cfg(target_os = "linux")]

use std::os::unix::io::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;

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
