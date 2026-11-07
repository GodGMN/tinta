//! The engine command loop.
//!
//! Commands are handled one at a time on the stdin reader thread, except
//! `install_models`, `import_audio` and `finalize`: those run on worker threads
//! and reply when they finish, so `ping`, `models_status`, `calendar`, `sample`
//! and the other quick commands keep answering during a 487 MB install or a long
//! final pass. A second install is rejected while one is in flight. Commands
//! whose features are not implemented yet fail with an `unsupported` error, and
//! macOS-only commands return the degraded replies from `docs/engine-protocol.md`
//! so the app does not show errors or retry loops.

use anyhow::{bail, Context, Result};
use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::io::{self, BufRead};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::install;
use crate::manifest::Manifest;
use crate::protocol::{log, Output, Request};
use crate::speech::SpeechPass;
use crate::threads::Threads;
use tinta_core::chunks::Track;

/// What the engine can do today. The app does not read this yet; the fields name
/// the features a later app can switch on.
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
const UNIMPLEMENTED: &[&str] = &["start", "pause", "resume", "mic_muted", "stop"];

pub struct Engine {
    out: Output,
    manifest: Manifest,
    models_root: PathBuf,
    threads: Threads,
    installing: Arc<AtomicBool>,
}

impl Engine {
    pub fn new(out: Output, manifest: Manifest, models_root: PathBuf, threads: Threads) -> Self {
        Self {
            out,
            manifest,
            models_root,
            threads,
            installing: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Emits `ready` and answers lines until stdin ends or a `shutdown` request
    /// arrives. Returns `Ok(())` for both clean exits.
    pub fn run<R: BufRead>(&mut self, mut input: R) -> io::Result<()> {
        self.out.event(
            "ready",
            json!({
                "version": env!("CARGO_PKG_VERSION"),
                "capabilities": Capabilities {
                    import_audio: true,
                    ..Default::default()
                },
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

        if request.cmd == "install_models" {
            self.start_install(&request.id)?;
            return Ok(false);
        }

        if request.cmd == "finalize" {
            self.start_finalize(&request)?;
            return Ok(false);
        }

        if request.cmd == "import_audio" {
            self.start_import(&request)?;
            return Ok(false);
        }

        match self.handle_command(&request.cmd, &request.params) {
            Ok(result) => self.out.reply(&request.id, result)?,
            Err(error) => self.out.fail(&request.id, &error.to_string())?,
        }
        Ok(false)
    }

    fn handle_command(&mut self, command: &str, params: &Map<String, Value>) -> Result<Value> {
        match command {
            "ping" => Ok(json!({ "pong": true })),
            "models_status" => Ok(json!({
                "installed": self.manifest.all_files_exist(&self.models_root),
                "path": self.models_root.display().to_string(),
            })),
            // A real microphone check needs a capture dependency (cpal, and ALSA
            // headers on Linux), which arrives with the capture task. Until then
            // report `granted` so onboarding is not blocked; the Windows privacy
            // switch is tracked there as well.
            "permissions" => Ok(json!({
                "microphone": "granted",
                "accessibility": false,
            })),
            "request_microphone" => Ok(json!({ "granted": true })),
            // The Swift engine always lists `all`; one entry keeps the record
            // panel's source select populated until system capture exists.
            "list_sources" => Ok(json!({
                "sources": [{ "id": "all", "name": "All system audio", "playing": false }],
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
            "sample" => {
                let dir = param_string(params, "dir")?;
                let track = match param_string(params, "track")?.as_str() {
                    "mic" => Track::Mic,
                    "remote" => Track::Remote,
                    _ => bail!("invalid track"),
                };
                let start = param_f64(params, "s")?;
                let end = param_f64(params, "e")?;
                let key = audio_key(params)?;
                crate::sample::sample(Path::new(&dir), &key, track, start, end)
            }
            other if UNIMPLEMENTED.contains(&other) => {
                bail!("unsupported command: {other}")
            }
            other => bail!("unknown command {other}"),
        }
    }

    /// Runs one final pass on a worker thread. Two passes may overlap; the speech
    /// layer serializes them while the models are loaded.
    fn start_finalize(&mut self, request: &Request) -> io::Result<()> {
        let out = self.out.clone();
        let id = request.id.clone();
        let params = request.params.clone();
        let manifest = self.manifest.clone();
        let root = self.models_root.clone();
        let threads = self.threads;
        std::thread::spawn(move || {
            let result = catching("the final pass stopped unexpectedly", || {
                let dir = param_string(&params, "dir")?;
                let key = audio_key(&params)?;
                let max_speakers = param_usize(&params, "max_speakers");
                let mut pass = SpeechPass::new(manifest, root, threads);
                crate::finalize::finalize(&mut pass, Path::new(&dir), &key, max_speakers, &out)
            });
            reply(&out, &id, result);
        });
        Ok(())
    }

    /// Runs one import on a worker thread, like `Engine.importAudio` on its
    /// detached task.
    fn start_import(&mut self, request: &Request) -> io::Result<()> {
        let out = self.out.clone();
        let id = request.id.clone();
        let params = request.params.clone();
        std::thread::spawn(move || {
            let result = catching("the import stopped unexpectedly", || {
                let dir = param_string(&params, "dir")?;
                let key = audio_key(&params)?;
                let path = param_string(&params, "path")?;
                crate::import::import_audio(Path::new(&dir), &key, Path::new(&path), &out)
            });
            reply(&out, &id, result);
        });
        Ok(())
    }

    /// Starts one install on a worker thread. The reply is written when the
    /// install finishes; a second request while one is running is rejected so
    /// two installs cannot race over the same files.
    fn start_install(&mut self, id: &Value) -> io::Result<()> {
        if self.installing.swap(true, Ordering::SeqCst) {
            return self.out.fail(id, "the models are already being installed");
        }
        let out = self.out.clone();
        let manifest = self.manifest.clone();
        let root = self.models_root.clone();
        let threads = self.threads;
        let installing = self.installing.clone();
        let id = id.clone();
        std::thread::spawn(move || {
            // A panic must still clear the flag and answer the request.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                install::install(&manifest, &root, &out, &threads)
            }));
            installing.store(false, Ordering::SeqCst);
            let _ = match result {
                Ok(Ok(())) => out.reply(&id, json!({ "installed": true })),
                Ok(Err(error)) => out.fail(&id, &error.to_string()),
                Err(_) => out.fail(&id, "the model install stopped unexpectedly"),
            };
        });
        Ok(())
    }
}

/// Reads a required, non-empty string parameter. The message matches Swift's
/// `params.string`.
fn param_string(params: &Map<String, Value>, key: &str) -> Result<String> {
    match params.get(key).and_then(Value::as_str) {
        Some(value) if !value.is_empty() => Ok(value.to_string()),
        _ => bail!("missing parameter: {key}"),
    }
}

fn param_f64(params: &Map<String, Value>, key: &str) -> Result<f64> {
    params
        .get(key)
        .and_then(Value::as_f64)
        .with_context(|| format!("missing parameter: {key}"))
}

/// An optional positive speaker limit; Swift ignores zero and negative values.
fn param_usize(params: &Map<String, Value>, key: &str) -> Option<usize> {
    params
        .get(key)
        .and_then(Value::as_i64)
        .filter(|value| *value > 0)
        .map(|value| value as usize)
}

/// Decodes `audio_key`, a base64 string of exactly 32 bytes. Anything else is
/// `invalid audio key`, like `Controller.key` in Swift.
fn audio_key(params: &Map<String, Value>) -> Result<[u8; 32]> {
    let text = param_string(params, "audio_key")?;
    let bytes = STANDARD
        .decode(text)
        .map_err(|_| anyhow::anyhow!("invalid audio key"))?;
    bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("invalid audio key"))
}

/// Runs a worker body and turns a panic into an error reply. Section 2.5 requires
/// an answer for every request: without this a panic in sherpa-onnx, symphonia or
/// rubato would leave the app waiting for its 60-minute timeout.
fn catching<T>(message: &str, job: impl FnOnce() -> Result<T>) -> Result<T> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(job)) {
        Ok(result) => result,
        Err(_) => Err(anyhow::anyhow!("{message}")),
    }
}

fn reply(out: &Output, id: &Value, result: Result<Value>) {
    let _ = match result {
        Ok(value) => out.reply(id, value),
        Err(error) => out.fail(id, &error.to_string()),
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::threads::Threads;
    use std::io::{Cursor, Write};
    use std::sync::atomic::Ordering;
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
        assert_eq!(capabilities["import_audio"], true);
        for (name, value) in capabilities {
            if name != "import_audio" {
                assert_eq!(value, &Value::Bool(false), "{name}");
            }
        }
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
    fn a_second_install_is_rejected() {
        let mut harness = harness();
        harness.engine.installing.store(true, Ordering::SeqCst);
        harness
            .engine
            .handle_line("{\"id\":9,\"cmd\":\"install_models\"}")
            .unwrap();
        let lines = harness.sink.lines();
        assert_eq!(lines[0]["id"], 9);
        assert_eq!(lines[0]["ok"], false);
        assert!(lines[0]["error"]
            .as_str()
            .unwrap()
            .contains("already being installed"));
    }

    #[test]
    fn list_sources_offers_the_default_output() {
        let lines = run("{\"id\":1,\"cmd\":\"list_sources\"}");
        let sources = lines[1]["result"]["sources"].as_array().unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0]["id"], "all");
        assert_eq!(sources[0]["name"], "All system audio");
        assert_eq!(sources[0]["playing"], false);
        assert_eq!(lines[1]["result"]["route"], "speakers");
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
    fn finalize_reports_a_missing_parameter_from_its_worker() {
        let mut harness = harness();
        harness
            .engine
            .handle_line("{\"id\":8,\"cmd\":\"finalize\",\"params\":{}}")
            .unwrap();
        let reply = wait_for_reply(&harness.sink, 1);
        assert_eq!(reply["id"], 8);
        assert_eq!(reply["ok"], false);
        assert_eq!(reply["error"], "missing parameter: dir");
    }

    #[test]
    fn a_panicking_worker_still_answers_with_an_error() {
        let result = catching(
            "the final pass stopped unexpectedly",
            || -> Result<Value> { panic!("sherpa-onnx panicked") },
        );
        let error = result.unwrap_err().to_string();
        assert_eq!(error, "the final pass stopped unexpectedly");
    }

    /// Builds a request line with serde, so a Windows path's backslashes are
    /// escaped correctly.
    fn line(id: u64, cmd: &str, params: Value) -> String {
        json!({ "id": id, "cmd": cmd, "params": params }).to_string()
    }

    #[test]
    fn sample_validates_its_track_and_rejects_a_short_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().display().to_string();
        let key = STANDARD.encode([1u8; 32]);
        let input = format!(
            "{}\n{}",
            line(
                1,
                "sample",
                json!({ "dir": path, "track": "left", "s": 0, "e": 1, "audio_key": key }),
            ),
            line(
                2,
                "sample",
                json!({ "dir": path, "track": "mic", "s": 0, "e": 1, "audio_key": "AAAA" }),
            ),
        );
        let lines = run(&input);
        assert_eq!(lines[1]["error"], "invalid track");
        assert_eq!(lines[2]["error"], "invalid audio key");
    }

    #[test]
    fn sample_returns_a_wav_for_an_empty_meeting() {
        let dir = tempfile::tempdir().unwrap();
        let key = STANDARD.encode([1u8; 32]);
        let input = line(
            1,
            "sample",
            json!({
                "dir": dir.path().display().to_string(),
                "track": "remote",
                "s": 0,
                "e": 1,
                "audio_key": key,
            }),
        );
        let lines = run(&input);
        let wav = lines[1]["result"]["wav_base64"].as_str().unwrap();
        assert_eq!(STANDARD.decode(wav).unwrap().len(), 44);
    }

    fn wait_for_reply(sink: &Sink, expected_lines: usize) -> Value {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            let lines = sink.lines();
            if lines.len() >= expected_lines {
                return lines[expected_lines - 1].clone();
            }
            assert!(
                std::time::Instant::now() < deadline,
                "timed out waiting for the worker reply"
            );
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
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
