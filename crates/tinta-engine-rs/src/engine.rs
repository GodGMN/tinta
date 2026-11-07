//! The engine command loop.
//!
//! Commands are handled one at a time on the stdin reader thread. The app never
//! requires two replies at once for these commands and serial handling keeps the
//! model state simple; slow work (only `install_models` today) therefore delays
//! the next line. Commands whose features are not implemented yet fail with an
//! `unsupported` error, and macOS-only commands return the degraded replies from
//! `docs/engine-protocol.md` so the app does not show errors or retry loops.

use anyhow::{bail, Result};
use serde::Serialize;
use serde_json::{json, Value};
use std::io::{self, BufRead};
use std::path::PathBuf;

use crate::install;
use crate::manifest::Manifest;
use crate::protocol::{log, Output, Request};
use crate::threads::Threads;

/// What the engine can do today. The app does not read this yet; the fields name
/// the features a later app can switch on. Everything is false in the skeleton.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Capabilities {
    pub capture_system_audio: bool,
    pub capture_app_audio: bool,
    pub calls_detection: bool,
    pub call_reader: bool,
    pub calendar: bool,
    pub summaries: bool,
    pub echo_cancellation: bool,
    pub import_audio: bool,
}

/// Commands that belong to the protocol but whose feature is not built yet.
const UNIMPLEMENTED: &[&str] = &[
    "start",
    "pause",
    "resume",
    "mic_muted",
    "stop",
    "finalize",
    "sample",
    "import_audio",
];

pub struct Engine {
    out: Output,
    manifest: Manifest,
    models_root: PathBuf,
    threads: Threads,
}

impl Engine {
    pub fn new(out: Output, manifest: Manifest, models_root: PathBuf, threads: Threads) -> Self {
        Self {
            out,
            manifest,
            models_root,
            threads,
        }
    }

    /// Emits `ready` and answers lines until stdin ends or a `shutdown` request
    /// arrives. Returns `Ok(())` for both clean exits.
    pub fn run<R: BufRead>(&mut self, mut input: R) -> io::Result<()> {
        self.out.event(
            "ready",
            json!({
                "version": env!("CARGO_PKG_VERSION"),
                "capabilities": Capabilities::default(),
            }),
        )?;

        let mut line = String::new();
        loop {
            line.clear();
            if input.read_line(&mut line)? == 0 {
                return Ok(());
            }
            let line = line.trim_end_matches(['\n', '\r']);
            if self.handle_line(line)? {
                return Ok(());
            }
        }
    }

    /// Handles one input line. Returns `true` when the loop should exit.
    pub fn handle_line(&mut self, line: &str) -> io::Result<bool> {
        let Some(request) = Request::parse(line) else {
            self.out
                .event("error", json!({ "message": "invalid request" }))?;
            return Ok(false);
        };

        if request.cmd == "shutdown" {
            self.out.reply(&request.id, json!({}))?;
            log("shutdown requested");
            return Ok(true);
        }

        match self.handle_command(&request.cmd, &request.params) {
            Ok(result) => self.out.reply(&request.id, result)?,
            Err(error) => self.out.fail(&request.id, &error.to_string())?,
        }
        Ok(false)
    }

    fn handle_command(
        &mut self,
        command: &str,
        _params: &serde_json::Map<String, Value>,
    ) -> Result<Value> {
        match command {
            "ping" => Ok(json!({ "pong": true })),
            "models_status" => Ok(json!({
                "installed": self.manifest.all_files_exist(&self.models_root),
                "path": self.models_root.display().to_string(),
            })),
            "install_models" => {
                install::install(&self.manifest, &self.models_root, &self.out, &self.threads)?;
                Ok(json!({ "installed": true }))
            }
            "permissions" => Ok(json!({
                "microphone": "granted",
                "accessibility": false,
            })),
            "request_microphone" => Ok(json!({ "granted": true })),
            "list_sources" => Ok(json!({
                "sources": [],
                "default_input": "",
                "route": "speakers",
            })),
            // macOS-only commands. These replies are the documented degraded
            // shapes, so the app shows no errors and does not retry.
            "calls" => Ok(json!({ "calls": [], "accessibility": false })),
            "request_accessibility" => Ok(json!({ "granted": false })),
            "calendar" => Ok(json!({
                "access": "denied",
                "calendars": [],
                "events": [],
            })),
            "request_calendar" => Ok(json!({ "granted": false })),
            "summary_status" => Ok(json!({
                "available": false,
                "reason": "Summaries are not available on this platform.",
            })),
            "call_report" => bail!("Call reading is not available on this platform."),
            "summarize" => bail!("Summaries are not available on this platform."),
            other if UNIMPLEMENTED.contains(&other) => {
                bail!("unsupported command: {other}")
            }
            other => bail!("unknown command {other}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threads::Threads;
    use std::io::{Cursor, Write};
    use std::sync::{Arc, Mutex};

    const HELLO_SHA: &str = "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    /// A one-file manifest that never reaches the network: the tests only
    /// inspect `models_status` and the protocol replies.
    fn fake_manifest() -> Manifest {
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

    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<u8>>>);

    impl Write for Sink {
        fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buffer);
            Ok(buffer.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Sink {
        fn lines(&self) -> Vec<Value> {
            let text = String::from_utf8(self.0.lock().unwrap().clone()).unwrap();
            text.lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        }
    }

    struct Harness {
        engine: Engine,
        sink: Sink,
        _dir: tempfile::TempDir,
    }

    fn harness() -> Harness {
        let dir = tempfile::tempdir().unwrap();
        let sink = Sink::default();
        let engine = Engine::new(
            Output::new(sink.clone()),
            fake_manifest(),
            dir.path().join("models"),
            Threads::for_tests(),
        );
        Harness {
            engine,
            sink,
            _dir: dir,
        }
    }

    fn run(input: &str) -> Vec<Value> {
        let mut harness = harness();
        harness
            .engine
            .run(Cursor::new(input.as_bytes().to_vec()))
            .unwrap();
        harness.sink.lines()
    }

    #[test]
    fn ready_is_the_first_line_and_lists_capabilities() {
        let lines = run("");
        assert_eq!(lines.len(), 1);
        let ready = &lines[0];
        assert_eq!(ready["event"], "ready");
        assert_eq!(ready["version"], env!("CARGO_PKG_VERSION"));
        let capabilities = ready["capabilities"].as_object().unwrap();
        assert_eq!(capabilities.len(), 8);
        assert!(capabilities
            .values()
            .all(|value| value == &Value::Bool(false)));
    }

    #[test]
    fn replies_echo_the_id() {
        let lines = run("{\"id\":7,\"cmd\":\"ping\",\"params\":{}}\n{\"id\":42,\"cmd\":\"ping\"}");
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[1]["id"], 7);
        assert_eq!(lines[1]["ok"], true);
        assert_eq!(lines[1]["result"]["pong"], true);
        assert_eq!(lines[2]["id"], 42);
        assert_eq!(lines[2]["result"]["pong"], true);
    }

    #[test]
    fn models_status_is_an_existence_check_without_loading() {
        let mut harness = harness();
        let lines = {
            harness
                .engine
                .handle_line("{\"id\":1,\"cmd\":\"models_status\"}")
                .unwrap();
            harness.sink.lines()
        };
        assert_eq!(lines[0]["id"], 1);
        assert_eq!(lines[0]["result"]["installed"], false);
        assert!(lines[0]["result"]["path"]
            .as_str()
            .unwrap()
            .ends_with("models"));
        assert!(!harness.sink.0.lock().unwrap().is_empty());
    }

    #[test]
    fn unknown_command_fails() {
        let lines = run("{\"id\":3,\"cmd\":\"nonsense\"}");
        assert_eq!(lines[1]["id"], 3);
        assert_eq!(lines[1]["ok"], false);
        assert!(lines[1]["error"]
            .as_str()
            .unwrap()
            .contains("unknown command nonsense"));
    }

    #[test]
    fn unimplemented_command_says_unsupported() {
        let lines = run("{\"id\":4,\"cmd\":\"start\",\"params\":{}}");
        assert_eq!(lines[1]["ok"], false);
        assert!(lines[1]["error"]
            .as_str()
            .unwrap()
            .contains("unsupported command: start"));
    }

    #[test]
    fn malformed_lines_get_an_error_event_and_no_reply() {
        let lines = run("this is not json\n{\"id\":5,\"cmd\":3}\n[]");
        assert_eq!(lines.len(), 4);
        for line in &lines[1..] {
            assert_eq!(line["event"], "error");
            assert_eq!(line["message"], "invalid request");
            assert!(line.get("id").is_none());
        }
    }

    #[test]
    fn mac_only_commands_return_degraded_replies() {
        let lines = run("{\"id\":1,\"cmd\":\"calendar\"}\n\
             {\"id\":2,\"cmd\":\"calls\",\"params\":{\"read\":true}}\n\
             {\"id\":3,\"cmd\":\"summary_status\"}\n\
             {\"id\":4,\"cmd\":\"request_calendar\"}");
        assert_eq!(lines[1]["result"]["access"], "denied");
        assert_eq!(lines[2]["result"]["calls"].as_array().unwrap().len(), 0);
        assert_eq!(lines[3]["result"]["available"], false);
        assert_eq!(lines[4]["result"]["granted"], false);
    }

    #[test]
    fn shutdown_replies_then_stops_reading() {
        let lines = run("{\"id\":0,\"cmd\":\"shutdown\"}\n{\"id\":1,\"cmd\":\"ping\"}");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1]["id"], 0);
        assert_eq!(lines[1]["ok"], true);
        assert_eq!(lines[1]["result"], json!({}));
    }

    #[test]
    fn eof_exits_cleanly() {
        let mut harness = harness();
        harness.engine.run(Cursor::new(Vec::new())).unwrap();
        assert_eq!(harness.sink.lines().len(), 1);
    }
}
