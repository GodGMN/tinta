//! Spawns the real binary and checks the transport, exactly like the app does.
//! The models folder is an empty temporary directory, so nothing is downloaded
//! and no model is loaded.

use serde_json::Value;
use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};

#[test]
fn ping_and_models_status_over_stdio() {
    let dir = tempfile::tempdir().unwrap();
    let models = dir.path().join("models");

    let mut child = Command::new(env!("CARGO_BIN_EXE_tinta-engine"))
        .env("TINTA_ENGINE_MODELS_DIR", &models)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the engine binary starts");

    let mut stdin = child.stdin.take().unwrap();
    let stdout = child.stdout.take().unwrap();
    let mut lines = BufReader::new(stdout).lines();

    let ready: Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(ready["event"], "ready");
    assert!(ready["capabilities"].is_object());

    writeln!(stdin, "{{\"id\":1,\"cmd\":\"ping\",\"params\":{{}}}}").unwrap();
    stdin.flush().unwrap();
    let ping: Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(ping["id"], 1);
    assert_eq!(ping["ok"], true);
    assert_eq!(ping["result"]["pong"], true);

    writeln!(stdin, "{{\"id\":2,\"cmd\":\"models_status\"}}").unwrap();
    stdin.flush().unwrap();
    let status: Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(status["id"], 2);
    assert_eq!(status["ok"], true);
    assert_eq!(status["result"]["installed"], false);
    assert_eq!(
        status["result"]["path"].as_str().unwrap(),
        models.display().to_string()
    );

    writeln!(stdin, "{{\"id\":0,\"cmd\":\"shutdown\"}}").unwrap();
    stdin.flush().unwrap();
    let shutdown: Value = serde_json::from_str(&lines.next().unwrap().unwrap()).unwrap();
    assert_eq!(shutdown["id"], 0);
    assert_eq!(shutdown["ok"], true);
    drop(stdin);

    let status = child.wait().expect("the engine exits");
    assert!(status.success(), "the engine exits with 0");
}
