//! End-to-end live recording over the engine protocol.
//!
//! Ignored by default. It needs a running PulseAudio or PipeWire pulse server and
//! `pactl`, `paplay` on `PATH` (or in `TINTA_PACTL` / `TINTA_PAPLAY`). It creates
//! two null sinks and a remapped microphone source, records ~28 s through the
//! real engine, and unloads everything afterwards. Run it with:
//!
//! ```text
//! cargo test --locked -p tinta-engine-rs --test recording_e2e -- --ignored --nocapture
//! ```

#![cfg(target_os = "linux")]

use base64::engine::general_purpose::STANDARD;
use base64::Engine as _;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Lines, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tinta_core::chunks::{read_track, Track};
use tinta_engine_rs::audio::SAMPLE_RATE;

const REMOTE_SINK: &str = "tinta_e2e_remote";
const MIC_SINK: &str = "tinta_e2e_mic_sink";
const MIC_SOURCE: &str = "tinta_e2e_mic";

/// The cases share one pulse server and change its default devices, so they run
/// one at a time even when the test harness uses several threads.
static SERVER: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn serial() -> std::sync::MutexGuard<'static, ()> {
    SERVER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn tool(name: &str) -> PathBuf {
    std::env::var_os(format!("TINTA_{}", name.to_uppercase()))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(name))
}

fn run(program: &Path, args: &[&str]) -> String {
    let output = Command::new(program)
        .args(args)
        .output()
        .unwrap_or_else(|error| panic!("cannot run {}: {error}", program.display()));
    assert!(
        output.status.success(),
        "{} {:?} failed: {}",
        program.display(),
        args,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

/// The virtual devices and the server defaults, restored on drop.
struct Fixture {
    pactl: PathBuf,
    modules: Vec<String>,
    original_sink: String,
    original_source: String,
}

impl Fixture {
    fn setup() -> Self {
        let pactl = tool("pactl");
        let original_sink = run(&pactl, &["get-default-sink"]);
        let original_source = run(&pactl, &["get-default-source"]);
        let mut fixture = Self {
            pactl,
            modules: Vec::new(),
            original_sink,
            original_source,
        };
        fixture.load(&[
            "module-null-sink",
            &format!("sink_name={REMOTE_SINK}"),
            "sink_properties=device.description=tinta_e2e_remote",
        ]);
        fixture.load(&[
            "module-null-sink",
            &format!("sink_name={MIC_SINK}"),
            "sink_properties=device.description=tinta_e2e_mic_sink",
        ]);
        fixture.load(&[
            "module-remap-source",
            &format!("master={MIC_SINK}.monitor"),
            &format!("source_name={MIC_SOURCE}"),
        ]);
        run(&fixture.pactl, &["set-default-sink", REMOTE_SINK]);
        run(&fixture.pactl, &["set-default-source", MIC_SOURCE]);
        // Let the server publish the new nodes before the engine opens them.
        thread::sleep(Duration::from_millis(500));
        fixture
    }

    fn load(&mut self, args: &[&str]) {
        let mut all = vec!["load-module"];
        all.extend_from_slice(args);
        let id = run(&self.pactl, &all);
        self.modules.push(id);
    }

    /// Unloads the most recently loaded module, as unplugging a device does.
    fn unload_last(&mut self) {
        let id = self.modules.pop().expect("a module is loaded");
        run(&self.pactl, &["unload-module", &id]);
    }

    /// Loads a fresh remapped microphone source over the microphone sink, the
    /// way plugging a headset back in publishes a new input.
    fn plug_microphone(&mut self, name: &str) {
        self.load(&[
            "module-remap-source",
            &format!("master={MIC_SINK}.monitor"),
            &format!("source_name={name}"),
        ]);
        run(&self.pactl, &["set-default-source", name]);
        thread::sleep(Duration::from_millis(500));
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        for module in self.modules.drain(..).rev() {
            let _ = Command::new(&self.pactl)
                .args(["unload-module", &module])
                .output();
        }
        if !self.original_sink.is_empty() {
            let _ = Command::new(&self.pactl)
                .args(["set-default-sink", &self.original_sink])
                .output();
        }
        if !self.original_source.is_empty() {
            let _ = Command::new(&self.pactl)
                .args(["set-default-source", &self.original_source])
                .output();
        }
    }
}

struct Engine {
    child: Child,
    stdin: ChildStdin,
    lines: Lines<BufReader<ChildStdout>>,
    seen: Vec<Value>,
}

impl Engine {
    fn spawn() -> Self {
        let models = tempfile::tempdir().unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_tinta-engine"))
            .env("TINTA_ENGINE_MODELS_DIR", models.path())
            // Live drafts need models; this test only checks capture and chunks.
            .env("TINTA_ENGINE_LIVE", "0")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("the engine binary starts");
        let stdin = child.stdin.take().unwrap();
        let lines = BufReader::new(child.stdout.take().unwrap()).lines();
        let mut engine = Self {
            child,
            stdin,
            lines,
            seen: Vec::new(),
        };
        let ready = engine.next_value();
        assert_eq!(ready["event"], "ready");
        engine
    }

    fn next_value(&mut self) -> Value {
        let line = self
            .lines
            .next()
            .expect("the engine line")
            .expect("a readable line");
        serde_json::from_str(&line).expect("a JSON line")
    }

    fn send(&mut self, value: Value) {
        writeln!(self.stdin, "{value}").unwrap();
        self.stdin.flush().unwrap();
    }

    /// Reads until the reply with `id`, keeping the events it passes in `seen`.
    fn reply(&mut self, id: u64) -> Value {
        loop {
            let value = self.next_value();
            if value.get("id").and_then(Value::as_u64) == Some(id) {
                return value;
            }
            self.seen.push(value);
        }
    }

    fn finish(&mut self) {
        self.send(json!({ "id": 0, "cmd": "shutdown" }));
        while let Some(Ok(line)) = self.lines.next() {
            if let Ok(value) = serde_json::from_str::<Value>(&line) {
                if value.get("id").is_none() {
                    self.seen.push(value);
                }
            }
        }
        let status = self.child.wait().unwrap();
        assert!(status.success(), "the engine exits with 0");
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn play(paplay: &Path, sink: &str, file: &Path) {
    let status = start_play(paplay, sink, file).wait().expect("paplay ends");
    assert!(status.success(), "paplay plays {sink}");
}

/// Starts `paplay` and returns the child, so the test can send commands while
/// the tone plays.
fn start_play(paplay: &Path, sink: &str, file: &Path) -> Child {
    Command::new(paplay)
        .arg(format!("--device={sink}"))
        .arg(file)
        .spawn()
        .expect("paplay starts")
}

fn tone(seconds: f32, frequency: f32) -> Vec<f32> {
    let count = (SAMPLE_RATE as f32 * seconds) as usize;
    (0..count)
        .map(|index| {
            let fade =
                ((index as f32 / 400.0).min(1.0)).min(((count - index) as f32 / 400.0).min(1.0));
            (2.0 * std::f32::consts::PI * frequency * index as f32 / SAMPLE_RATE as f32).sin()
                * 0.5
                * fade
        })
        .collect()
}

fn rms(samples: &[f32], from: f32, to: f32) -> f32 {
    let start = (from * SAMPLE_RATE as f32) as usize;
    let end = ((to * SAMPLE_RATE as f32) as usize).min(samples.len());
    if start >= end {
        return 0.0;
    }
    let slice = &samples[start..end];
    (slice.iter().map(|sample| sample * sample).sum::<f32>() / slice.len() as f32).sqrt()
}

fn sleep_until(start: Instant, seconds: f32) {
    let target = start + Duration::from_secs_f32(seconds);
    if let Some(left) = target.checked_duration_since(Instant::now()) {
        thread::sleep(left);
    }
}

#[test]
#[ignore = "needs a PulseAudio/PipeWire server and pulseaudio-utils"]
fn records_two_tracks_with_pause_and_levels() {
    let _serial = serial();
    let fixture = Fixture::setup();
    let paplay = tool("paplay");
    let dir = tempfile::tempdir().unwrap();
    let remote_tone = dir.path().join("remote.wav");
    let mic_tone = dir.path().join("mic.wav");
    let paused_tone = dir.path().join("paused.wav");
    std::fs::write(&remote_tone, tinta_engine_rs::audio::wav(&tone(5.0, 440.0))).unwrap();
    std::fs::write(&mic_tone, tinta_engine_rs::audio::wav(&tone(5.0, 880.0))).unwrap();
    std::fs::write(&paused_tone, tinta_engine_rs::audio::wav(&tone(2.0, 550.0))).unwrap();

    let mut engine = Engine::spawn();
    engine.send(json!({
        "id": 1,
        "cmd": "start",
        "params": {
            "dir": dir.path().display().to_string(),
            "audio_key": STANDARD.encode([7u8; 32]),
            "source": "all",
        },
    }));
    let start = engine.reply(1);
    assert_eq!(start["ok"], true, "{start}");
    assert!(start["result"]["start_wall_ms"].as_i64().unwrap() > 0);
    let t0 = Instant::now();

    // Remote speech in 0-5 s. The microphone must stay silent.
    play(&paplay, REMOTE_SINK, &remote_tone);

    // Microphone speech in 8-13 s. The remote track must stay silent. Mute the
    // microphone by wall-clock time at 9.5 s and unmute at 11 s: the window must
    // be silent even though the audio was captured before the commands arrived.
    sleep_until(t0, 8.0);
    let mut tone = start_play(&paplay, MIC_SINK, &mic_tone);
    sleep_until(t0, 9.5);
    let muted_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    engine.send(json!({ "id": 5, "cmd": "mic_muted",
        "params": { "muted": true, "t": muted_at } }));
    assert_eq!(engine.reply(5)["ok"], true);
    sleep_until(t0, 11.0);
    let unmuted_at = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as i64;
    engine.send(json!({ "id": 6, "cmd": "mic_muted",
        "params": { "muted": false, "t": unmuted_at } }));
    assert_eq!(engine.reply(6)["ok"], true);
    assert!(tone.wait().unwrap().success(), "the microphone tone plays");

    // A remote tone while paused must not reach the chunks.
    sleep_until(t0, 15.0);
    engine.send(json!({ "id": 2, "cmd": "pause" }));
    assert_eq!(engine.reply(2)["ok"], true);
    play(&paplay, REMOTE_SINK, &paused_tone);

    // Resume and speak into the microphone again in 20-25 s.
    sleep_until(t0, 19.5);
    engine.send(json!({ "id": 3, "cmd": "resume" }));
    assert_eq!(engine.reply(3)["ok"], true);
    sleep_until(t0, 20.0);
    play(&paplay, MIC_SINK, &mic_tone);

    sleep_until(t0, 28.0);
    engine.send(json!({ "id": 4, "cmd": "stop" }));
    assert_eq!(engine.reply(4)["ok"], true);
    engine.finish();
    let seen = std::mem::take(&mut engine.seen);

    let key = [7u8; 32];
    let remote = read_track(dir.path(), &key, Track::Remote).unwrap();
    let mic = read_track(dir.path(), &key, Track::Mic).unwrap();
    eprintln!("remote {} samples, mic {} samples", remote.len(), mic.len());
    for (name, track) in [("remote", &remote), ("mic", &mic)] {
        let windows: Vec<String> = (0..28)
            .map(|second| {
                format!(
                    "{second}:{:.3}",
                    rms(track, second as f32, second as f32 + 1.0)
                )
            })
            .collect();
        eprintln!("{name} rms by second: {}", windows.join(" "));
    }
    eprintln!(
        "default sink {} / source {}",
        run(&fixture.pactl, &["get-default-sink"]),
        run(&fixture.pactl, &["get-default-source"])
    );

    // Both tracks open within a few hundred milliseconds of the start.
    assert!(remote.len() > 20 * SAMPLE_RATE as usize, "remote length");
    assert!(mic.len() > 20 * SAMPLE_RATE as usize, "mic length");

    // Remote: audible 1-4 s, silent while only the microphone plays and while
    // the recording is paused.
    assert!(rms(&remote, 1.0, 4.0) > 0.1, "remote tone");
    assert!(
        rms(&remote, 9.0, 12.0) < 0.01,
        "remote hears the microphone"
    );
    assert!(rms(&remote, 16.0, 18.0) < 0.01, "remote through the pause");

    // Microphone: silent while only the remote plays, audible at 8-9.4 s, silent
    // through the mute, audible again until the tone ends, and again after the
    // resume.
    assert!(rms(&mic, 1.0, 4.0) < 0.01, "microphone hears the remote");
    assert!(rms(&mic, 8.2, 9.3) > 0.1, "microphone tone");
    assert!(
        rms(&mic, 9.7, 10.8) < 0.01,
        "microphone through the mute: {:.4}",
        rms(&mic, 9.7, 10.8)
    );
    assert!(rms(&mic, 11.2, 12.6) > 0.1, "microphone after the unmute");
    assert!(rms(&mic, 21.0, 24.0) > 0.1, "microphone after the resume");

    // The levels events report both meters, the capture state and the mute.
    let levels: Vec<&Value> = seen
        .iter()
        .filter(|value| value["event"] == "levels")
        .collect();
    assert!(!levels.is_empty(), "no levels events");
    assert!(levels
        .iter()
        .any(|event| event["remote"].as_f64().unwrap_or(0.0) > 0.02));
    assert!(levels
        .iter()
        .any(|event| event["mic"].as_f64().unwrap_or(0.0) > 0.02));
    assert!(levels.iter().any(|event| event["remote_capturing"] == true));
    assert!(
        levels
            .iter()
            .any(|event| event["mic_muted"] == true && event["mic"].as_f64() == Some(0.0)),
        "no muted levels tick"
    );

    let session = std::fs::read_dir(dir.path().join("audio"))
        .unwrap()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_name().to_string_lossy().starts_with("session-"))
        .count();
    assert_eq!(session, 1, "one session json");

    drop(fixture);
}

#[test]
#[ignore = "needs a PulseAudio/PipeWire server and pulseaudio-utils"]
fn a_monitor_default_source_is_not_recorded_as_the_microphone() {
    let _serial = serial();
    let fixture = Fixture::setup();
    let paplay = tool("paplay");
    // A machine with no input device makes the speaker monitor the default
    // source. Recording it as the microphone duplicates every remote word.
    run(
        &fixture.pactl,
        &["set-default-source", &format!("{REMOTE_SINK}.monitor")],
    );
    thread::sleep(Duration::from_millis(500));

    let dir = tempfile::tempdir().unwrap();
    let remote_tone = dir.path().join("remote.wav");
    std::fs::write(&remote_tone, tinta_engine_rs::audio::wav(&tone(3.0, 440.0))).unwrap();

    let mut engine = Engine::spawn();
    engine.send(json!({
        "id": 1,
        "cmd": "start",
        "params": {
            "dir": dir.path().display().to_string(),
            "audio_key": STANDARD.encode([7u8; 32]),
            "source": "all",
        },
    }));
    let start = engine.reply(1);
    assert_eq!(start["ok"], true, "{start}");
    play(&paplay, REMOTE_SINK, &remote_tone);
    thread::sleep(Duration::from_millis(3_500));
    engine.send(json!({ "id": 2, "cmd": "stop" }));
    assert_eq!(engine.reply(2)["ok"], true);
    engine.finish();
    let seen = std::mem::take(&mut engine.seen);

    let key = [7u8; 32];
    assert!(
        !dir.path().join("audio/mic").exists(),
        "a monitor source must not create a microphone track"
    );
    assert!(read_track(dir.path(), &key, Track::Mic).unwrap().is_empty());
    assert!(
        rms(
            &read_track(dir.path(), &key, Track::Remote).unwrap(),
            0.5,
            2.5
        ) > 0.1
    );

    let warnings: Vec<String> = seen
        .iter()
        .filter(|value| value["event"] == "warning")
        .map(|value| value["message"].as_str().unwrap_or_default().to_string())
        .collect();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(
        warnings[0].contains("No microphone was found"),
        "{:?}",
        warnings[0]
    );
    let levels: Vec<&Value> = seen
        .iter()
        .filter(|value| value["event"] == "levels")
        .collect();
    assert!(
        !levels.is_empty() && levels.iter().all(|event| event["mic"] == 0.0),
        "the microphone meter stays at zero"
    );
}

#[test]
#[ignore = "needs a PulseAudio/PipeWire server and pulseaudio-utils"]
fn a_microphone_that_disappears_and_returns_is_not_filled_with_the_monitor() {
    let _serial = serial();
    let mut fixture = Fixture::setup();
    let paplay = tool("paplay");
    let dir = tempfile::tempdir().unwrap();
    let mic_tone = dir.path().join("mic.wav");
    let remote_tone = dir.path().join("remote.wav");
    std::fs::write(&mic_tone, tinta_engine_rs::audio::wav(&tone(2.5, 880.0))).unwrap();
    std::fs::write(&remote_tone, tinta_engine_rs::audio::wav(&tone(2.5, 440.0))).unwrap();

    let mut engine = Engine::spawn();
    engine.send(json!({
        "id": 1,
        "cmd": "start",
        "params": {
            "dir": dir.path().display().to_string(),
            "audio_key": STANDARD.encode([7u8; 32]),
            "source": "all",
        },
    }));
    let start = engine.reply(1);
    assert_eq!(start["ok"], true, "{start}");
    let t0 = Instant::now();

    // 0-2.5 s: the microphone records its tone.
    play(&paplay, MIC_SINK, &mic_tone);

    // 3 s: unplug the microphone. The server falls back to a speaker monitor,
    // which must never be recorded as the microphone.
    sleep_until(t0, 3.0);
    fixture.unload_last();
    // 4-6.5 s: only the meeting audio plays; the microphone track stays silent.
    sleep_until(t0, 4.0);
    play(&paplay, REMOTE_SINK, &remote_tone);

    // 7 s: plug a new microphone in, as a USB headset appearing or Bluetooth
    // switching to HFP. The track must resume on the meeting clock.
    sleep_until(t0, 7.0);
    fixture.plug_microphone("tinta_e2e_mic2");
    sleep_until(t0, 9.0);
    play(&paplay, MIC_SINK, &mic_tone);

    sleep_until(t0, 13.5);
    engine.send(json!({ "id": 2, "cmd": "stop" }));
    assert_eq!(engine.reply(2)["ok"], true);
    engine.finish();
    let seen = std::mem::take(&mut engine.seen);

    let key = [7u8; 32];
    let mic = read_track(dir.path(), &key, Track::Mic).unwrap();
    let remote = read_track(dir.path(), &key, Track::Remote).unwrap();
    eprintln!("mic {} samples, remote {} samples", mic.len(), remote.len());
    for (name, track) in [("remote", &remote), ("mic", &mic)] {
        let windows: Vec<String> = (0..14)
            .map(|second| {
                format!(
                    "{second}:{:.3}",
                    rms(track, second as f32, second as f32 + 1.0)
                )
            })
            .collect();
        eprintln!("{name} rms by second: {}", windows.join(" "));
    }

    assert!(rms(&mic, 0.3, 2.2) > 0.1, "the first microphone tone");
    assert!(
        rms(&mic, 4.2, 6.2) < 0.01,
        "the monitor is recorded as the microphone after the unplug: {:.3}",
        rms(&mic, 4.2, 6.2)
    );
    assert!(
        rms(&remote, 4.2, 6.2) > 0.1,
        "the meeting audio after the unplug"
    );
    assert!(
        rms(&mic, 9.3, 11.5) > 0.1,
        "the microphone resumed after it was plugged back: {:.3}",
        rms(&mic, 9.3, 11.5)
    );

    let warnings: Vec<String> = seen
        .iter()
        .filter(|value| value["event"] == "warning")
        .map(|value| value["message"].as_str().unwrap_or_default().to_string())
        .collect();
    eprintln!("warnings: {warnings:?}");
    assert!(
        warnings
            .iter()
            .any(|message| message.contains("records the meeting audio only")),
        "{warnings:?}"
    );
    assert!(
        warnings
            .iter()
            .any(|message| message.contains("available again")),
        "{warnings:?}"
    );

    drop(fixture);
}
