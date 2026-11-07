//! Live recording: capture both tracks, write TNTC chunks, emit `levels` and
//! optional `live` drafts.
//!
//! This mirrors `RecordingSession` in `engine/Sources/TintaEngine/Session.swift`:
//!
//! - the meeting clock starts at `start_wall_ms`; each read is placed by the
//!   time its audio was captured, with the sample counter as the primary clock
//!   and the wall clock correcting it in both directions, so a stall is
//!   silence or a small trim, never a permanent shift;
//! - the microphone is held for 0.8 s so a `mic_muted` command that arrives a
//!   little after the click still silences the audio from the exact time;
//! - `pause` drops both tracks and flushes, `stop` releases the held
//!   microphone and flushes both writers;
//! - `levels` is emitted every 250 ms with peak RMS since the previous tick.
//!
//! The two capture threads are the only writers of chunk files. Reopening a
//! failed stream keeps the clock, so the outage is a silent gap; all buffers
//! are bounded, so memory does not grow with the length of the meeting.
//!
//! A machine with no physical microphone makes its speaker monitor the default
//! source. Recording that as the microphone would duplicate every remote
//! utterance as the user, so the microphone open reports it as missing. The
//! microphone thread always runs: while no real input is the default it writes
//! nothing and polls the server, and as soon as a real input appears (a USB
//! headset is plugged in, a Bluetooth headset switches to HFP) it opens it and
//! the microphone track resumes at its true place on the meeting clock. A
//! meeting that never gets a microphone has no `audio/mic` directory at all.

pub mod capture;
pub mod live;

use anyhow::{Context, Result};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::SyncSender;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::manifest::Manifest;
use crate::protocol::{log, Output};
use crate::threads::Threads;
use capture::{CaptureSource, DeviceChange, Opened, FRAGMENT_SAMPLES};
use tinta_core::chunks::{ChunkWriter, Track, GAP_SAMPLES, SAMPLE_RATE};

/// The microphone audio held back before it is stored, in samples (0.8 s).
const MIC_HOLD_SAMPLES: i64 = (SAMPLE_RATE * 8 / 10) as i64;
/// How long remote levels are kept for the live echo check, in samples (60 s).
const ECHO_WINDOW_SAMPLES: i64 = 60 * SAMPLE_RATE as i64;
/// The levels cadence, the same as Swift's 250 ms timer.
const LEVELS_INTERVAL: Duration = Duration::from_millis(250);
/// Reopen attempts back off from this delay to [`REOPEN_LOG_INTERVAL`].
const REOPEN_DELAY: Duration = Duration::from_millis(300);
/// A reopen failure is logged at most once per interval, and the retry delay
/// stops growing here, so a device that is gone overnight cannot flood stderr.
const REOPEN_LOG_INTERVAL: Duration = Duration::from_secs(30);
/// How often a microphone-less track asks the server again for a real input.
const MIC_RETRY_INTERVAL: Duration = Duration::from_secs(2);
/// How long `stop` waits for the capture threads before it replies anyway. A
/// wedged audio server would otherwise hold the command loop forever.
const STOP_TIMEOUT: Duration = Duration::from_secs(5);
/// The largest correction a single read applies to the meeting clock: 5 ms.
/// Dropping or inserting a whole 100 ms read is audible, so clock drift is
/// corrected in small steps instead.
const DRIFT_STEP_SAMPLES: usize = (SAMPLE_RATE / 200) as usize;
/// A quiet read can take the full 5 ms step; a loud one takes 1 ms so the cut
/// or pad stays under a syllable. Quiet means below [`DRIFT_QUIET_RMS`].
const DRIFT_LOUD_STEP_SAMPLES: usize = (SAMPLE_RATE / 1000) as usize;
const DRIFT_QUIET_RMS: f32 = 0.02;
/// A gap this much larger than the deadband is not clock drift: audio was lost,
/// so it is placed at its true time as silence instead of being slid over.
const DRIFT_SNAP_SAMPLES: i64 = 2 * GAP_SAMPLES;

/// Everything `start` needs from its parameters.
pub struct StartOptions {
    pub directory: PathBuf,
    pub key: [u8; 32],
    pub source: String,
    pub mic_muted: bool,
}

/// The model paths a live draft pass needs. Live drafts are only started when
/// the models are installed; the final pass re-reads the same chunks anyway.
#[derive(Clone)]
pub struct LiveConfig {
    pub manifest: Manifest,
    pub models_root: PathBuf,
    pub threads: Threads,
}

/// One recording in progress. Dropping it stops the capture and flushes.
pub struct RecordingSession {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    live: Option<live::LiveDrafts>,
    stopped: AtomicBool,
}

impl RecordingSession {
    /// Opens both capture streams and starts the capture, level and live
    /// threads. A stream that cannot open fails the whole start, like Swift;
    /// a machine with no physical microphone records the meeting audio only.
    pub fn start(options: StartOptions, out: Output, live: Option<LiveConfig>) -> Result<Self> {
        let StartOptions {
            directory,
            key,
            source,
            mic_muted,
        } = options;

        let mut remote = capture::open_remote()
            .map_err(|error| anyhow::anyhow!("cannot capture the meeting audio: {error}"))?;
        let (mut mic, missing_mic) = match capture::open_capture(Track::Mic) {
            Ok(Opened::Source(source)) => (Some(source), None),
            Ok(Opened::Unavailable(reason)) => (None, Some(reason)),
            Err(error) => {
                // The microphone is optional here: a broken input device must
                // not cost the user the meeting audio.
                (None, Some(error.to_string()))
            }
        };
        // Drop what the server buffered while the second stream opened, so
        // both tracks start at the meeting start instead of carrying a
        // different amount of pre-roll.
        if let Some(mic) = &mut mic {
            mic.flush()
                .context("cannot prepare the microphone stream")?;
        }
        remote
            .flush()
            .context("cannot prepare the meeting audio stream")?;

        let start_wall_ms = now_wall_ms();
        let start = Instant::now();
        let recorder = Recorder::new(&directory, &key, mic_muted, start_wall_ms, mic.is_some())?;
        write_session_json(&directory, start_wall_ms, &source)?;

        let shared = Arc::new(Shared {
            recorder: Mutex::new(recorder),
            out,
            start,
            start_wall_ms,
            stop: AtomicBool::new(false),
            remote_capturing: AtomicBool::new(true),
        });
        shared.set_remote_capturing(true);

        if let Some(reason) = &missing_mic {
            log(&format!(
                "no microphone, recording the meeting audio only: {reason}"
            ));
            shared.emit(
                "warning",
                json!({
                    "message": format!(
                        "No microphone was found, so Tinta records the meeting audio only. ({reason})"
                    ),
                }),
            );
        }

        let live = match live {
            Some(config) => Some(live::LiveDrafts::spawn(config, &shared)?),
            None => None,
        };

        // The microphone thread always runs, even without a microphone: it
        // waits for a real input to become the default and then records it.
        // `warned` is true when `start` already emitted the one warning for
        // the missing microphone.
        let warned = missing_mic.is_some();
        let mut workers = Vec::new();
        let mic_shared = shared.clone();
        let mic_worker = thread::Builder::new()
            .name("capture-mic".to_string())
            .spawn(move || run_microphone(mic_shared, mic, warned));
        match mic_worker {
            Ok(worker) => workers.push(worker),
            Err(error) => {
                shared.stop.store(true, Ordering::SeqCst);
                for worker in workers.drain(..) {
                    let _ = worker.join();
                }
                return Err(anyhow::Error::new(error).context("cannot start the capture thread"));
            }
        }
        let remote_shared = shared.clone();
        let spawned = thread::Builder::new()
            .name("capture-remote".to_string())
            .spawn(move || run_capture(remote_shared, Track::Remote, remote));
        match spawned {
            Ok(worker) => workers.push(worker),
            Err(error) => {
                shared.stop.store(true, Ordering::SeqCst);
                for worker in workers.drain(..) {
                    let _ = worker.join();
                }
                return Err(anyhow::Error::new(error).context("cannot start the capture thread"));
            }
        }
        let levels = shared.clone();
        match thread::Builder::new()
            .name("levels".to_string())
            .spawn(move || run_levels(levels))
        {
            Ok(worker) => workers.push(worker),
            Err(error) => {
                shared.stop.store(true, Ordering::SeqCst);
                for worker in workers.drain(..) {
                    let _ = worker.join();
                }
                return Err(anyhow::Error::new(error).context("cannot start the levels thread"));
            }
        }

        Ok(Self {
            shared,
            workers,
            live,
            stopped: AtomicBool::new(false),
        })
    }

    /// The wall-clock start of the meeting, in Unix milliseconds.
    pub fn start_wall_ms(&self) -> i64 {
        self.shared.start_wall_ms
    }

    /// Drops incoming audio on both tracks and flushes what is buffered.
    pub fn pause(&self) {
        self.shared.pause();
    }

    pub fn resume(&self) {
        if let Ok(mut recorder) = self.shared.recorder.lock() {
            recorder.resume();
        }
    }

    /// Records a microphone mute change at a wall-clock time in milliseconds.
    pub fn set_mic_muted(&self, muted: bool, wall_ms: i64) {
        if let Ok(mut recorder) = self.shared.recorder.lock() {
            recorder.set_mic_muted(muted, wall_ms);
        }
    }

    /// Stops capture, releases the held microphone audio and flushes both
    /// tracks. Idempotent.
    ///
    /// Returns a message when the audio server did not let the capture threads
    /// leave within [`STOP_TIMEOUT`]: `stop` still flushes what was written, so
    /// the meeting keeps its audio, but the caller is told the reply was
    /// bounded rather than silently hanging on a wedged server.
    pub fn stop(&mut self) -> Option<String> {
        if self.stopped.swap(true, Ordering::SeqCst) {
            return None;
        }
        self.shared.stop.store(true, Ordering::SeqCst);
        let wedged = join_workers(std::mem::take(&mut self.workers), STOP_TIMEOUT);
        // Closing the recorder first drops the last live sender, so the live
        // worker sees a closed queue and can exit before `finish` waits on it.
        let errors = match self.shared.recorder.lock() {
            Ok(mut recorder) => recorder.stop(),
            Err(_) => Vec::new(),
        };
        if let Some(live) = self.live.take() {
            live.finish();
        }
        for message in errors {
            let _ = self
                .shared
                .out
                .event("error", json!({ "message": message }));
        }
        wedged.then(|| {
            format!(
                "the audio server did not answer; Tinta stopped after {} s and kept the audio written so far",
                STOP_TIMEOUT.as_secs()
            )
        })
    }
}

impl Drop for RecordingSession {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}

/// Joins capture and level threads, waiting at most `timeout` in total.
///
/// Returns true when a thread was still blocked in an audio call that has no
/// timeout. That thread is detached: the stop flag makes it drop anything it
/// reads later, and the writers are flushed by the caller, so the meeting keeps
/// the audio written so far.
fn join_workers(workers: Vec<JoinHandle<()>>, timeout: Duration) -> bool {
    let deadline = Instant::now() + timeout;
    let mut wedged = false;
    for worker in workers {
        while !worker.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(10));
        }
        if worker.is_finished() {
            let _ = worker.join();
        } else {
            wedged = true;
        }
    }
    wedged
}

/// State shared by the session and its threads.
pub(crate) struct Shared {
    recorder: Mutex<Recorder>,
    out: Output,
    start: Instant,
    start_wall_ms: i64,
    stop: AtomicBool,
    remote_capturing: AtomicBool,
}

impl Shared {
    fn ingest(&self, track: Track, samples: &[f32], index: i64) {
        let error = self
            .recorder
            .lock()
            .map(|mut recorder| recorder.ingest(track, samples, index))
            .unwrap_or(None);
        if let Some(message) = error {
            let _ = self.out.event("error", json!({ "message": message }));
        }
    }

    /// Writes `count` samples of silence at `index`, used to nudge the meeting
    /// clock forward for a slow capture device one small step at a time.
    fn ingest_silence(&self, track: Track, count: usize, index: i64) {
        if count == 0 {
            return;
        }
        let silence = vec![0.0f32; count];
        self.ingest(track, &silence, index);
    }

    /// Creates the microphone track's chunk writer, if a real input appeared
    /// after a meeting started without one. The meeting then gains an
    /// `audio/mic` directory; one that never gets a microphone never does.
    fn enable_mic(&self) -> Option<String> {
        match self.recorder.lock() {
            Ok(mut recorder) => recorder
                .enable_mic()
                .err()
                .map(|error| format!("cannot create the microphone track: {error}")),
            Err(_) => {
                Some("the recording state is poisoned; the microphone is not recorded".to_string())
            }
        }
    }

    fn pause(&self) {
        let errors = self
            .recorder
            .lock()
            .map(|mut recorder| recorder.pause())
            .unwrap_or_default();
        for message in errors {
            let _ = self.out.event("error", json!({ "message": message }));
        }
    }

    pub(crate) fn is_echo(&self, level: f32, start: i64, end: i64) -> bool {
        self.recorder
            .lock()
            .map(|recorder| recorder.is_echo(level, start, end))
            .unwrap_or(false)
    }

    pub(crate) fn stopping(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    pub(crate) fn emit(&self, event: &str, fields: Value) {
        let _ = self.out.event(event, fields);
    }

    fn set_remote_capturing(&self, capturing: bool) {
        let previous = self.remote_capturing.swap(capturing, Ordering::SeqCst);
        if previous != capturing {
            let _ = self.out.event(
                "source_state",
                json!({
                    "capturing": capturing,
                    "processes": if capturing { 1 } else { 0 },
                }),
            );
        }
    }

    fn emit_levels(&self) {
        let Ok(mut recorder) = self.recorder.lock() else {
            return;
        };
        let muted = recorder.mic_muted_now();
        let (mic, remote) = recorder.take_levels(muted);
        drop(recorder);
        let _ = self.out.event(
            "levels",
            json!({
                "mic": mic,
                "remote": remote,
                "remote_capturing": self.remote_capturing.load(Ordering::SeqCst),
                "mic_muted": muted,
            }),
        );
    }
}

/// The meeting-audio capture loop. A failed stream is reopened with the default
/// device; the clock keeps running, so the outage is written as silence.
fn run_capture(shared: Arc<Shared>, track: Track, source: Box<dyn CaptureSource>) {
    run_capture_with(shared, track, source, capture::open_capture);
}

/// The testable form of the loop: `open` supplies the replacement after an error
/// or a default-device change.
fn run_capture_with<F>(
    shared: Arc<Shared>,
    track: Track,
    mut source: Box<dyn CaptureSource>,
    open: F,
) where
    F: Fn(Track) -> Result<Opened>,
{
    let mut buffer = vec![0.0f32; FRAGMENT_SAMPLES];
    let mut clock = ClockMap::default();
    let mut warned = false;
    while !shared.stopping() {
        // A changed default device is not a read error. On PipeWire the server
        // moves the stream; on plain PulseAudio it does not move it and does
        // not fail it, so the stream has to be reopened by name.
        if source.device_change() != DeviceChange::None {
            log(&format!(
                "{} capture follows the new default device",
                track.dir_name()
            ));
            let reason = match open(track) {
                Ok(Opened::Source(next)) => {
                    source = next;
                    continue;
                }
                Ok(Opened::Unavailable(reason)) => reason,
                Err(error) => error.to_string(),
            };
            shared.set_remote_capturing(false);
            emit_stopped(&shared, track, &mut warned, &reason);
            match reopen(&shared, track, &open) {
                Some(next) => {
                    source = next;
                    continue;
                }
                None => return,
            }
        }
        match source.read(&mut buffer) {
            Ok(read) if read.count == 0 => continue,
            Ok(read) => {
                if shared.stopping() {
                    break;
                }
                if warned {
                    warned = false;
                    log(&format!("{} capture reconnected", track.dir_name()));
                }
                shared.set_remote_capturing(true);
                let placement = clock.place(
                    shared.start,
                    read.captured,
                    read.count,
                    rms(&buffer[..read.count]) < DRIFT_QUIET_RMS,
                );
                shared.ingest_silence(track, placement.pad, placement.index);
                shared.ingest(
                    track,
                    &buffer[placement.skip..read.count],
                    placement.index + placement.pad as i64,
                );
            }
            Err(error) => {
                shared.set_remote_capturing(false);
                emit_stopped(&shared, track, &mut warned, &error.to_string());
                match reopen(&shared, track, &open) {
                    Some(next) => source = next,
                    None => return,
                }
            }
        }
    }
    shared.set_remote_capturing(false);
}

/// Emits the one warning per outage that tells the user capture stopped.
fn emit_stopped(shared: &Shared, track: Track, warned: &mut bool, error: &str) {
    if *warned {
        return;
    }
    *warned = true;
    let name = if track == Track::Mic {
        "microphone"
    } else {
        "meeting audio"
    };
    shared.emit(
        "warning",
        json!({
            "message": format!("The {name} capture stopped: {error}. Tinta reconnects it."),
        }),
    );
}

/// Reopens a failed stream, backing off from 300 ms to 30 s. A missing device
/// stays silent and is logged at most once per 30 s. Returns `None` on stop.
fn reopen<F>(shared: &Shared, track: Track, open: &F) -> Option<Box<dyn CaptureSource>>
where
    F: Fn(Track) -> Result<Opened>,
{
    let mut delay = REOPEN_DELAY;
    let mut last_log: Option<Instant> = None;
    loop {
        if !sleep_unless_stopped(shared, delay) {
            return None;
        }
        let error = match open(track) {
            Ok(Opened::Source(next)) => return Some(next),
            Ok(Opened::Unavailable(reason)) => reason,
            Err(error) => error.to_string(),
        };
        let now = Instant::now();
        if last_log.is_none_or(|at| now.duration_since(at) >= REOPEN_LOG_INTERVAL) {
            log(&format!(
                "cannot reopen the {} capture: {error}",
                track.dir_name()
            ));
            last_log = Some(now);
        }
        delay = (delay * 2).min(REOPEN_LOG_INTERVAL);
    }
}

/// The microphone capture loop. Unlike the meeting audio, the microphone may
/// legitimately not exist: the default source can be the speaker monitor (no
/// input device), and it can appear later (a USB headset, or Bluetooth
/// switching from A2DP to HFP when a call app opens the input).
///
/// The loop always runs. While there is no real input it writes nothing and
/// asks the server again every [`MIC_RETRY_INTERVAL`]; the first read after an
/// input appears is placed on the meeting clock by its capture time, so the
/// missing span is silence and the track stays aligned with the meeting audio.
fn run_microphone(shared: Arc<Shared>, source: Option<Box<dyn CaptureSource>>, warned: bool) {
    run_microphone_with(
        shared,
        source,
        capture::open_capture,
        warned,
        MIC_RETRY_INTERVAL,
    );
}

/// The testable form of the microphone loop: `open` supplies the next stream and
/// reports when no real input is the default.
fn run_microphone_with<F>(
    shared: Arc<Shared>,
    mut source: Option<Box<dyn CaptureSource>>,
    open: F,
    mut warned: bool,
    retry: Duration,
) where
    F: Fn(Track) -> Result<Opened>,
{
    let mut buffer = vec![0.0f32; FRAGMENT_SAMPLES];
    let mut clock = ClockMap::default();
    // The first open of a missing microphone happens at once; after that the
    // server is polled at the retry interval.
    let mut delay = Duration::ZERO;
    loop {
        if shared.stopping() {
            break;
        }
        let Some(current) = source.as_mut() else {
            if !delay.is_zero() && !sleep_unless_stopped(&shared, delay) {
                break;
            }
            delay = retry;
            match open(Track::Mic) {
                Ok(Opened::Source(next)) => {
                    if let Some(message) = shared.enable_mic() {
                        shared.emit("error", json!({ "message": message }));
                    }
                    log("microphone capture started");
                    if warned {
                        warned = false;
                        shared.emit(
                            "warning",
                            json!({
                                "message": "The microphone is available again. Tinta records it from now on.",
                            }),
                        );
                    }
                    source = Some(next);
                }
                Ok(Opened::Unavailable(reason)) => {
                    if !warned {
                        warned = true;
                        log(&format!("no microphone: {reason}"));
                        shared.emit(
                            "warning",
                            json!({
                                "message": format!(
                                    "The microphone is unavailable, so Tinta records the meeting audio only. ({reason})"
                                ),
                            }),
                        );
                    }
                }
                Err(error) => {
                    if !warned {
                        warned = true;
                        log(&format!("cannot open the microphone: {error}"));
                        shared.emit(
                            "warning",
                            json!({
                                "message": format!(
                                    "The microphone capture stopped: {error}. Tinta records the meeting audio only until it is back."
                                ),
                            }),
                        );
                    }
                }
            }
            continue;
        };
        // A default-device change is not a read error. Reopen by name; if the
        // new default is the monitor, `open` reports it as missing and the
        // stream is closed here (this also covers the server moving the stream
        // onto a monitor without a read error).
        match current.device_change() {
            DeviceChange::None => {}
            change => {
                log(if change == DeviceChange::Monitor {
                    "the default source is the speaker monitor; closing the microphone"
                } else {
                    "microphone capture follows the new default device"
                });
                source = None;
                delay = Duration::ZERO;
                continue;
            }
        }
        match current.read(&mut buffer) {
            Ok(read) if read.count == 0 => {}
            Ok(read) => {
                if shared.stopping() {
                    break;
                }
                let placement = clock.place(
                    shared.start,
                    read.captured,
                    read.count,
                    rms(&buffer[..read.count]) < DRIFT_QUIET_RMS,
                );
                shared.ingest_silence(Track::Mic, placement.pad, placement.index);
                shared.ingest(
                    Track::Mic,
                    &buffer[placement.skip..read.count],
                    placement.index + placement.pad as i64,
                );
            }
            Err(error) => {
                if shared.stopping() {
                    break;
                }
                log(&format!("the microphone capture stopped: {error}"));
                source = None;
                delay = Duration::ZERO;
            }
        }
    }
}

/// Sleeps in short steps, returning false as soon as the session stops.
fn sleep_unless_stopped(shared: &Shared, duration: Duration) -> bool {
    let deadline = Instant::now() + duration;
    loop {
        if shared.stopping() {
            return false;
        }
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return true;
        }
        thread::sleep(left.min(Duration::from_millis(100)));
    }
}

fn run_levels(shared: Arc<Shared>) {
    let mut elapsed = Duration::ZERO;
    while !shared.stop.load(Ordering::SeqCst) {
        thread::sleep(Duration::from_millis(50));
        if shared.stop.load(Ordering::SeqCst) {
            break;
        }
        elapsed += Duration::from_millis(50);
        if elapsed < LEVELS_INTERVAL {
            continue;
        }
        elapsed = Duration::ZERO;
        shared.emit_levels();
    }
}

/// Where one read goes on the meeting clock.
struct Placement {
    /// The meeting-clock index of the first stored sample.
    index: i64,
    /// Samples to drop from the front of the read, correcting a fast device.
    skip: usize,
    /// Samples of silence to write before the read, correcting a slow device.
    pad: usize,
}

/// Maps capture reads to the meeting clock.
///
/// The sample counter is the primary clock, so jitter in the capture timestamps
/// never moves a track. The capture time of each read corrects it when the two
/// clocks disagree by more than [`GAP_SAMPLES`]. A correction of a whole read is
/// audible, though: a device 1 % fast would cut 100 ms out of a syllable, and a
/// slow one would insert 100 ms of silence at once. Instead the excess is
/// trimmed or padded [`DRIFT_STEP_SAMPLES`] at a time (5 ms in quiet audio,
/// 1 ms while loud), and a jump beyond [`DRIFT_SNAP_SAMPLES`] is treated as lost
/// audio and placed at its true time.
#[derive(Default)]
struct ClockMap {
    next: Option<i64>,
}

impl ClockMap {
    fn place(
        &mut self,
        start: Instant,
        captured: Option<Instant>,
        count: usize,
        quiet: bool,
    ) -> Placement {
        let Some(captured) = captured else {
            // No capture time: count samples only.
            let index = self.next.unwrap_or(0);
            self.next = Some(index + count as i64);
            return Placement {
                index,
                skip: 0,
                pad: 0,
            };
        };
        let elapsed = captured.saturating_duration_since(start).as_secs_f64();
        // The read ends at `captured`; its first sample is `count` earlier.
        let wall_start = ((elapsed * SAMPLE_RATE as f64) as i64 - count as i64).max(0);
        let Some(next) = self.next else {
            self.next = Some(wall_start + count as i64);
            return Placement {
                index: wall_start,
                skip: 0,
                pad: 0,
            };
        };
        let drift = next - wall_start;
        let step = if quiet {
            DRIFT_STEP_SAMPLES
        } else {
            DRIFT_LOUD_STEP_SAMPLES
        };
        if drift > DRIFT_SNAP_SAMPLES {
            // Far ahead of the wall clock: not clock drift but excess audio.
            // Drop the excess in one go, as before.
            let skip = (drift as usize).min(count);
            let index = wall_start + skip as i64;
            self.next = Some(index + (count - skip) as i64);
            Placement {
                index,
                skip,
                pad: 0,
            }
        } else if drift > GAP_SAMPLES {
            // Slightly ahead: trim a small step from the front of this read.
            let skip = (drift as usize).min(step).min(count);
            self.next = Some(next + (count - skip) as i64);
            Placement {
                index: next,
                skip,
                pad: 0,
            }
        } else if drift < -DRIFT_SNAP_SAMPLES {
            // Far behind the wall clock: samples were lost, so leave silence
            // rather than shifting the rest of the track.
            self.next = Some(wall_start + count as i64);
            Placement {
                index: wall_start,
                skip: 0,
                pad: 0,
            }
        } else if drift < -GAP_SAMPLES {
            // Slightly behind: insert a small step of silence before this read.
            let pad = ((-drift) as usize).min(step);
            self.next = Some(next + pad as i64 + count as i64);
            Placement {
                index: next,
                skip: 0,
                pad,
            }
        } else {
            self.next = Some(next + count as i64);
            Placement {
                index: next,
                skip: 0,
                pad: 0,
            }
        }
    }
}

/// One track's capture state: chunk writer, held microphone audio, mute changes,
/// level peaks and remote levels for the live echo check. No buffer grows with
/// the length of the meeting.
struct Recorder {
    /// `None` for a track this recording does not have yet: the microphone on a
    /// machine without one. It is created when a real input appears.
    writers: [Option<ChunkWriter>; 2],
    /// Kept so the microphone writer can be created after the meeting starts.
    directory: PathBuf,
    key: [u8; 32],
    start_wall_ms: i64,
    paused: bool,
    mic_hold: VecDeque<(Vec<f32>, i64)>,
    mute_changes: Vec<(i64, bool)>,
    levels: [f32; 2],
    remote_levels: VecDeque<(i64, i64, f32)>,
    live: Option<SyncSender<live::LivePiece>>,
}

fn slot(track: Track) -> usize {
    match track {
        Track::Mic => 0,
        Track::Remote => 1,
    }
}

impl Recorder {
    fn new(
        directory: &Path,
        key: &[u8; 32],
        mic_muted: bool,
        start_wall_ms: i64,
        record_mic: bool,
    ) -> Result<Self> {
        // With no physical microphone (the default source is a speaker
        // monitor) no microphone directory is created at all: the meeting has
        // no mic track, and `finalize` reads it as empty.
        let mic = if record_mic {
            Some(
                ChunkWriter::new(directory, key, Track::Mic)
                    .context("cannot create the microphone track")?,
            )
        } else {
            None
        };
        let remote = Some(
            ChunkWriter::new(directory, key, Track::Remote)
                .context("cannot create the meeting audio track")?,
        );
        let mut mute_changes = Vec::new();
        if mic_muted {
            mute_changes.push((0, true));
        }
        Ok(Self {
            writers: [mic, remote],
            directory: directory.to_path_buf(),
            key: *key,
            start_wall_ms,
            paused: false,
            mic_hold: VecDeque::new(),
            mute_changes,
            levels: [0.0; 2],
            remote_levels: VecDeque::new(),
            live: None,
        })
    }

    /// Whether this recording has a microphone track.
    fn has_mic(&self) -> bool {
        self.writers[0].is_some()
    }

    /// Creates the microphone track when a real input appears mid-meeting.
    fn enable_mic(&mut self) -> Result<()> {
        if self.writers[0].is_none() {
            self.writers[0] = Some(
                ChunkWriter::new(&self.directory, &self.key, Track::Mic)
                    .context("cannot create the microphone track")?,
            );
        }
        Ok(())
    }

    /// Feeds one read. Levels update even while paused, like Swift.
    fn ingest(&mut self, track: Track, samples: &[f32], index: i64) -> Option<String> {
        let level = rms(samples);
        let id = slot(track);
        self.levels[id] = self.levels[id].max(level);
        if track == Track::Remote {
            if !samples.is_empty() {
                self.remote_levels
                    .push_back((index, index + samples.len() as i64, level));
            }
            let cutoff = index - ECHO_WINDOW_SAMPLES;
            while self
                .remote_levels
                .front()
                .is_some_and(|(_, end, _)| *end < cutoff)
            {
                self.remote_levels.pop_front();
            }
        }
        if track == Track::Mic && !self.has_mic() {
            return None;
        }
        if self.paused {
            return None;
        }
        match track {
            Track::Mic => self.hold_mic(samples, index),
            Track::Remote => self.write(Track::Remote, samples, index),
        }
    }

    /// Appends to the 0.8 s hold and stores whatever is older than it, with the
    /// muted intervals replaced by silence.
    fn hold_mic(&mut self, samples: &[f32], index: i64) -> Option<String> {
        self.mic_hold.push_back((samples.to_vec(), index));
        let cutoff = index + samples.len() as i64 - MIC_HOLD_SAMPLES;
        let mut error = None;
        while let Some((held, start)) = self.mic_hold.front() {
            if start + held.len() as i64 > cutoff {
                break;
            }
            let (held, start) = self.mic_hold.pop_front().expect("front exists");
            let held = self.silence_muted(held, start);
            if let Some(message) = self.write(Track::Mic, &held, start) {
                error = Some(message);
            }
        }
        error
    }

    fn release_mic(&mut self) -> Vec<String> {
        let mut errors = Vec::new();
        while let Some((held, start)) = self.mic_hold.pop_front() {
            let held = self.silence_muted(held, start);
            if let Some(message) = self.write(Track::Mic, &held, start) {
                errors.push(message);
            }
        }
        errors
    }

    fn silence_muted(&self, mut samples: Vec<f32>, start: i64) -> Vec<f32> {
        if self.mute_changes.is_empty() {
            return samples;
        }
        for (offset, sample) in samples.iter_mut().enumerate() {
            if self.is_muted(start + offset as i64) {
                *sample = 0.0;
            }
        }
        samples
    }

    fn is_muted(&self, index: i64) -> bool {
        let position = self.mute_changes.partition_point(|(at, _)| *at <= index);
        position > 0 && self.mute_changes[position - 1].1
    }

    fn mic_muted_now(&self) -> bool {
        self.mute_changes.last().is_some_and(|(_, muted)| *muted)
    }

    fn set_mic_muted(&mut self, muted: bool, wall_ms: i64) {
        let index = (wall_ms - self.start_wall_ms).max(0) * SAMPLE_RATE as i64 / 1000;
        // Same index: the later command wins, as in Swift.
        let position = self.mute_changes.partition_point(|(at, _)| *at <= index);
        self.mute_changes.insert(position, (index, muted));
    }

    fn write(&mut self, track: Track, samples: &[f32], start: i64) -> Option<String> {
        if samples.is_empty() {
            return None;
        }
        if let Some(live) = &self.live {
            // A full queue drops the draft piece; chunk writing never waits on
            // the live pass.
            let _ = live.try_send(live::LivePiece {
                track,
                index: start,
                samples: samples.to_vec(),
            });
        }
        self.writers[slot(track)]
            .as_mut()?
            .append(samples, start)
            .err()
            .map(|error| format!("chunk write failed for {}: {error}", track.dir_name()))
    }

    fn flush_all(&mut self) -> Vec<String> {
        let mut errors = Vec::new();
        for (track, writer) in Track::ALL.into_iter().zip(self.writers.iter_mut()) {
            let Some(writer) = writer else { continue };
            if let Err(error) = writer.flush() {
                errors.push(format!(
                    "chunk write failed for {}: {error}",
                    track.dir_name()
                ));
            }
        }
        errors
    }

    fn pause(&mut self) -> Vec<String> {
        let mut errors = self.release_mic();
        self.paused = true;
        errors.extend(self.flush_all());
        errors
    }

    fn resume(&mut self) {
        self.paused = false;
    }

    fn stop(&mut self) -> Vec<String> {
        let mut errors = self.release_mic();
        // Closing the live channel lets its worker finish and drop the models.
        self.live = None;
        errors.extend(self.flush_all());
        errors
    }

    fn set_live(&mut self, live: SyncSender<live::LivePiece>) {
        self.live = Some(live);
    }

    fn take_levels(&mut self, mic_muted: bool) -> (f32, f32) {
        let mic = std::mem::take(&mut self.levels[0]);
        let remote = std::mem::take(&mut self.levels[1]);
        (if mic_muted { 0.0 } else { mic }, remote)
    }

    /// A microphone utterance is echo when the remote track was playing and the
    /// microphone was much quieter at the same time.
    fn is_echo(&self, level: f32, start: i64, end: i64) -> bool {
        let mut sum = 0.0f32;
        let mut count = 0usize;
        for (remote_start, remote_end, remote_level) in &self.remote_levels {
            if *remote_end > start && *remote_start < end {
                sum += remote_level * remote_level;
                count += 1;
            }
        }
        if count == 0 {
            return false;
        }
        crate::echo::is_echo_levels(level, (sum / count as f32).sqrt())
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|sample| sample * sample).sum();
    (sum / samples.len() as f32).sqrt()
}

/// Writes `<dir>/audio/session-<start_wall_ms>.json`, informational like Swift's.
fn write_session_json(directory: &Path, start_wall_ms: i64, source: &str) -> Result<()> {
    let audio = directory.join("audio");
    fs::create_dir_all(&audio).with_context(|| format!("cannot create {}", audio.display()))?;
    let path = audio.join(format!("session-{start_wall_ms}.json"));
    let document = json!({
        "start_wall_ms": start_wall_ms,
        "sample_rate": SAMPLE_RATE,
        "source": source,
    });
    fs::write(&path, serde_json::to_vec(&document)?)
        .with_context(|| format!("cannot write {}", path.display()))
}

/// Unix milliseconds, like `Int64(Date().timeIntervalSince1970 * 1000)`.
pub fn now_wall_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recording::capture::CaptureRead;
    use tinta_core::chunks::read_track;

    fn key() -> [u8; 32] {
        let mut key = [0u8; 32];
        for (index, byte) in key.iter_mut().enumerate() {
            *byte = index as u8;
        }
        key
    }

    fn recorder(dir: &Path, mic_muted: bool) -> Recorder {
        Recorder::new(dir, &key(), mic_muted, 1_000_000, true).unwrap()
    }

    fn constant(value: f32, count: usize) -> Vec<f32> {
        vec![value; count]
    }

    fn track(dir: &Path, track: Track) -> Vec<f32> {
        read_track(dir, &key(), track).unwrap()
    }

    fn rms_between(samples: &[f32], start: usize, end: usize) -> f32 {
        rms(&samples[start..end])
    }

    #[test]
    fn a_buffered_stall_does_not_shift_the_track() {
        let start = Instant::now();
        let mut clock = ClockMap::default();
        // Ten on-time reads, 0-1.0 s, with a few milliseconds of jitter in the
        // capture times. Sample counting keeps every index exact.
        for step in 1..=10 {
            let jitter = if step % 2 == 0 { 0.002 } else { -0.001 };
            let captured = Duration::from_secs_f64(step as f64 * 0.1 + jitter);
            let placement = clock.place(start, Some(start + captured), FRAGMENT_SAMPLES, true);
            assert_eq!(placement.index, (step - 1) as i64 * 1_600);
            assert_eq!(placement.skip, 0);
        }
        // The thread stalls 500 ms while the server keeps buffering. When it
        // resumes, the five fragments are read at once, each carrying the time
        // it was captured.
        for step in 11..=15 {
            let captured = Duration::from_secs_f64(step as f64 * 0.1);
            let placement = clock.place(start, Some(start + captured), FRAGMENT_SAMPLES, true);
            assert_eq!(placement.index, (step - 1) as i64 * 1_600);
            assert_eq!(placement.skip, 0);
        }
        // The track is still 1.5 s long, not 1.9 s: the stall left no shift.
        let placement = clock.place(
            start,
            Some(start + Duration::from_millis(1_600)),
            1_600,
            true,
        );
        let end = placement.index + (1_600 - placement.skip) as i64;
        let wall = (1.6 * SAMPLE_RATE as f64) as i64;
        assert!(
            (end - wall).abs() < 800,
            "residual offset {} samples",
            end - wall
        );
    }

    #[test]
    fn a_real_outage_leaves_a_silent_gap() {
        let start = Instant::now();
        let mut clock = ClockMap::default();
        for step in 1..=5 {
            let placement = clock.place(
                start,
                Some(start + Duration::from_millis(step * 100)),
                FRAGMENT_SAMPLES,
                true,
            );
            assert_eq!(placement.index, (step - 1) as i64 * 1_600);
        }
        // Reads resume 1.1 s later with nothing buffered: the missing audio is
        // silence at its true place on the clock.
        let placement = clock.place(
            start,
            Some(start + Duration::from_millis(1_600)),
            1_600,
            true,
        );
        assert_eq!(placement.index, 24_000);
        assert_eq!(placement.skip, 0);
        assert_eq!(placement.pad, 0);
    }

    /// What one drift run saw.
    struct Drift {
        /// Track length minus the wall-clock length, in samples.
        residual: i64,
        /// The same after the first minute.
        first_residual: i64,
        /// The largest number of samples trimmed from one read.
        max_skip: usize,
        /// The largest silence inserted before one read.
        max_pad: usize,
    }

    /// Feeds `reads` of one fragment each at `rate` times the wall clock, in
    /// quiet audio unless `quiet` is false.
    fn drift_run(rate: f64, reads: usize, quiet: bool) -> Drift {
        let start = Instant::now();
        let mut clock = ClockMap::default();
        let mut wall = 0.0f64;
        let mut first_residual = 0i64;
        let mut end = 0i64;
        let mut max_skip = 0usize;
        let mut max_pad = 0usize;
        for step in 0..reads {
            wall += FRAGMENT_SAMPLES as f64 / (SAMPLE_RATE as f64 * rate);
            let placement = clock.place(
                start,
                Some(start + Duration::from_secs_f64(wall)),
                FRAGMENT_SAMPLES,
                quiet,
            );
            max_skip = max_skip.max(placement.skip);
            max_pad = max_pad.max(placement.pad);
            end =
                placement.index + placement.pad as i64 + (FRAGMENT_SAMPLES - placement.skip) as i64;
            if step == 600 {
                first_residual = end - (wall * SAMPLE_RATE as f64) as i64;
            }
        }
        Drift {
            residual: end - (wall * SAMPLE_RATE as f64) as i64,
            first_residual,
            max_skip,
            max_pad,
        }
    }

    #[test]
    fn a_fast_device_clock_is_corrected_instead_of_stretching_the_track() {
        // 1% fast over ten minutes of 100 ms reads, as a USB headset against
        // the wall clock.
        let drift = drift_run(1.01, 6_000, true);
        assert!(
            drift.residual.abs() <= 2 * GAP_SAMPLES,
            "the track is {} samples off the wall clock",
            drift.residual
        );
        // The error after ten minutes is the same size as after one: it is
        // bounded, not accumulating.
        assert!(
            drift.first_residual.abs() <= 2 * GAP_SAMPLES,
            "after a minute: {}",
            drift.first_residual
        );
        // A correction never cuts more than 5 ms out of one read.
        assert_eq!(drift.max_pad, 0, "a fast clock never pads");
        assert!(
            drift.max_skip <= DRIFT_STEP_SAMPLES,
            "a correction trimmed {} samples",
            drift.max_skip
        );
        assert!(drift.max_skip > 0, "the run never corrected");
    }

    #[test]
    fn a_slow_device_clock_is_corrected_with_silence() {
        let drift = drift_run(0.99, 6_000, true);
        assert!(
            drift.residual.abs() <= 2 * GAP_SAMPLES,
            "the track is {} samples off the wall clock",
            drift.residual
        );
        // Silence is inserted in steps of at most 5 ms, never a whole read.
        assert_eq!(drift.max_skip, 0, "a slow clock never trims");
        assert!(
            drift.max_pad <= DRIFT_STEP_SAMPLES,
            "a correction inserted {} samples",
            drift.max_pad
        );
        assert!(drift.max_pad > 0, "the run never corrected");
    }

    #[test]
    fn a_loud_device_clock_is_corrected_in_smaller_steps() {
        // While speech is playing the correction is 1 ms per read: still
        // bounded, and never audible as a cut.
        let drift = drift_run(1.01, 6_000, false);
        assert!(
            drift.residual.abs() <= 2 * GAP_SAMPLES,
            "the track is {} samples off the wall clock",
            drift.residual
        );
        assert_eq!(drift.max_skip, DRIFT_LOUD_STEP_SAMPLES);
        assert_eq!(drift.max_pad, 0);
    }

    #[test]
    fn pause_drops_audio_and_leaves_a_silent_gap() {
        let dir = tempfile::tempdir().unwrap();
        let mut recorder = recorder(dir.path(), false);
        recorder.ingest(Track::Remote, &constant(0.5, 16_000), 0);
        recorder.pause();
        // Pausing flushes the first second into a chunk.
        assert!(dir.path().join("audio/remote/00000000.tntc").exists());
        recorder.ingest(Track::Remote, &constant(0.5, 16_000), 16_000);
        recorder.resume();
        recorder.ingest(Track::Remote, &constant(0.5, 16_000), 48_000);
        assert!(recorder.stop().is_empty());

        let samples = track(dir.path(), Track::Remote);
        assert_eq!(samples.len(), 64_000);
        assert!(rms_between(&samples, 0, 16_000) > 0.4);
        assert_eq!(rms_between(&samples, 16_000, 48_000), 0.0);
        assert!(rms_between(&samples, 48_000, 64_000) > 0.4);
    }

    #[test]
    fn a_gap_larger_than_100_ms_starts_a_new_chunk() {
        let dir = tempfile::tempdir().unwrap();
        let mut recorder = recorder(dir.path(), false);
        recorder.ingest(Track::Remote, &constant(0.5, 100), 0);
        recorder.ingest(Track::Remote, &constant(0.5, 100), 3_000);
        recorder.stop();
        let chunks = tinta_core::chunks::read_chunks(dir.path(), &key(), Track::Remote).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[1].start, 3_000);
    }

    #[test]
    fn the_microphone_is_held_so_a_late_mute_silences_the_exact_time() {
        let dir = tempfile::tempdir().unwrap();
        let mut recorder = recorder(dir.path(), false);
        // One second of microphone audio is held (0.8 s) with 0.2 s left.
        recorder.ingest(Track::Mic, &constant(0.5, 16_000), 0);
        assert_eq!(recorder.mic_hold.len(), 1);
        // The command arrives "late", after the audio: mute from 0.5 s.
        recorder.set_mic_muted(true, 1_000_500);
        // Another 0.8 s releases the held second.
        recorder.ingest(Track::Mic, &constant(0.0, 12_800), 16_000);
        assert!(recorder
            .mic_hold
            .iter()
            .all(|(held, start)| start + held.len() as i64 > MIC_HOLD_SAMPLES));
        recorder.stop();

        let samples = track(dir.path(), Track::Mic);
        assert!(rms_between(&samples, 0, 8_000) > 0.4, "before the mute");
        assert_eq!(rms_between(&samples, 8_000, 16_000), 0.0, "after it");
    }

    #[test]
    fn mute_changes_are_sorted_by_meeting_time_not_arrival() {
        let dir = tempfile::tempdir().unwrap();
        let mut recorder = recorder(dir.path(), false);
        recorder.set_mic_muted(true, 1_000_700);
        recorder.set_mic_muted(false, 1_000_300);
        assert_eq!(recorder.mute_changes, vec![(4_800, false), (11_200, true)]);
        assert!(!recorder.is_muted(0));
        assert!(!recorder.is_muted(4_800));
        assert!(recorder.is_muted(11_200));
        assert!(recorder.mic_muted_now());
    }

    #[test]
    fn a_muted_start_seeds_silence_from_sample_zero() {
        let dir = tempfile::tempdir().unwrap();
        let mut recorder = recorder(dir.path(), true);
        recorder.ingest(Track::Mic, &constant(0.5, 16_000), 0);
        recorder.ingest(Track::Mic, &constant(0.5, 12_800), 16_000);
        recorder.set_mic_muted(false, 1_002_000);
        recorder.ingest(Track::Mic, &constant(0.5, 16_000), 28_800);
        recorder.ingest(Track::Mic, &constant(0.5, 16_000), 44_800);
        recorder.stop();

        let samples = track(dir.path(), Track::Mic);
        assert_eq!(rms_between(&samples, 0, 32_000), 0.0);
        assert!(rms_between(&samples, 34_000, 44_800) > 0.4);
    }

    #[test]
    fn levels_keep_the_peak_and_zero_the_microphone_while_muted() {
        let dir = tempfile::tempdir().unwrap();
        let mut recorder = recorder(dir.path(), false);
        recorder.ingest(Track::Mic, &constant(0.5, 100), 0);
        recorder.ingest(Track::Remote, &constant(0.25, 100), 0);
        let (mic, remote) = recorder.take_levels(false);
        assert!((mic - 0.5).abs() < 1e-6);
        assert!((remote - 0.25).abs() < 1e-6);
        // The next tick has no new audio.
        assert_eq!(recorder.take_levels(false), (0.0, 0.0));

        recorder.ingest(Track::Mic, &constant(0.9, 100), 0);
        // While muted the microphone peak is reported as zero and discarded.
        assert_eq!(recorder.take_levels(true).0, 0.0);
        assert_eq!(recorder.take_levels(false).0, 0.0);
    }

    #[test]
    fn stopping_releases_held_microphone_audio() {
        let dir = tempfile::tempdir().unwrap();
        let mut recorder = recorder(dir.path(), false);
        recorder.ingest(Track::Mic, &constant(0.5, 8_000), 0);
        assert_eq!(recorder.mic_hold.len(), 1);
        assert!(recorder.stop().is_empty());
        assert!(recorder.mic_hold.is_empty());
        let samples = track(dir.path(), Track::Mic);
        assert_eq!(samples.len(), 8_000);
        assert!(rms_between(&samples, 0, 8_000) > 0.4);
    }

    #[test]
    fn remote_levels_are_bounded_for_the_echo_check() {
        let dir = tempfile::tempdir().unwrap();
        let mut recorder = recorder(dir.path(), false);
        let step = 1_600i64;
        for index in 0..10 {
            recorder.ingest(Track::Remote, &constant(0.1, step as usize), index * step);
        }
        assert!(recorder.remote_levels.len() <= 10);
        // Ten minutes of 100 ms reads keeps only the last 60 seconds.
        for index in 10..6_000i64 {
            recorder.ingest(Track::Remote, &constant(0.1, step as usize), index * step);
        }
        assert!(recorder.remote_levels.len() <= 602);
        let first = recorder.remote_levels.front().unwrap().0;
        let last = recorder.remote_levels.back().unwrap().0;
        assert!(last - first >= 59 * SAMPLE_RATE as i64);
    }

    #[test]
    fn echo_uses_the_remote_levels_in_the_same_range() {
        let dir = tempfile::tempdir().unwrap();
        let mut recorder = recorder(dir.path(), false);
        recorder.ingest(Track::Remote, &constant(0.1, 16_000), 0);
        assert!(recorder.is_echo(0.01, 0, 16_000));
        assert!(!recorder.is_echo(0.02, 0, 16_000));
        assert!(!recorder.is_echo(0.01, 32_000, 48_000));
    }

    #[test]
    fn a_recording_without_a_microphone_writes_no_mic_track() {
        let dir = tempfile::tempdir().unwrap();
        let mut recorder = Recorder::new(dir.path(), &key(), false, 1_000_000, false).unwrap();
        // The monitor-as-microphone audio is ignored, not stored.
        recorder.ingest(Track::Mic, &constant(0.5, 16_000), 0);
        recorder.ingest(Track::Remote, &constant(0.5, 16_000), 0);
        assert!(recorder.stop().is_empty());
        assert!(!dir.path().join("audio/mic").exists(), "no mic directory");
        assert_eq!(track(dir.path(), Track::Remote).len(), 16_000);
        assert!(read_track(dir.path(), &key(), Track::Mic)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn session_json_matches_the_documented_shape() {
        let dir = tempfile::tempdir().unwrap();
        write_session_json(dir.path(), 1_762_400_000_000, "all").unwrap();
        let path = dir.path().join("audio/session-1762400000000.json");
        let value: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        assert_eq!(value["start_wall_ms"], 1_762_400_000_000i64);
        assert_eq!(value["sample_rate"], 16_000);
        assert_eq!(value["source"], "all");
    }

    #[derive(Clone, Default)]
    struct Sink(Arc<Mutex<Vec<u8>>>);

    impl std::io::Write for Sink {
        fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buffer);
            Ok(buffer.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn shared_for_capture(dir: &Path) -> Arc<Shared> {
        shared_with_mic_track(dir, true).0
    }

    /// A shared session and the event sink it writes to. `mic` says whether the
    /// meeting started with a microphone track.
    fn shared_with_mic_track(dir: &Path, mic: bool) -> (Arc<Shared>, Sink) {
        let sink = Sink::default();
        let recorder = Recorder::new(dir, &key(), false, 1_000_000, mic).unwrap();
        let shared = Arc::new(Shared {
            recorder: Mutex::new(recorder),
            out: Output::new(sink.clone()),
            start: Instant::now(),
            start_wall_ms: 1_000_000,
            stop: AtomicBool::new(false),
            remote_capturing: AtomicBool::new(true),
        });
        (shared, sink)
    }

    fn events(sink: &Sink) -> Vec<Value> {
        String::from_utf8_lossy(&sink.0.lock().unwrap())
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }

    /// A virtual capture clock, so read timing does not depend on the load of
    /// the test machine. Each read advances it by its fragment.
    struct Timeline {
        start: Instant,
        micros: std::sync::atomic::AtomicU64,
    }

    impl Timeline {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                start: Instant::now(),
                micros: std::sync::atomic::AtomicU64::new(0),
            })
        }

        /// Advances the clock and returns the capture time of the last sample.
        fn advance(&self, duration: Duration) -> Instant {
            let micros = duration.as_micros() as u64;
            let end = self.micros.fetch_add(micros, Ordering::SeqCst) + micros;
            self.start + Duration::from_micros(end)
        }
    }

    /// A capture source driven by the test: one read every 100 ms, an error after
    /// `fail_after` good reads, a device change after `reopen_after` reads, and
    /// a stop after `limit` calls in total.
    struct FakeSource {
        fail_after: Option<usize>,
        reopen_after: Option<usize>,
        monitor_after: Option<usize>,
        change: DeviceChange,
        reads: usize,
        total: Arc<std::sync::atomic::AtomicUsize>,
        limit: usize,
        shared: Arc<Shared>,
        timeline: Arc<Timeline>,
    }

    impl CaptureSource for FakeSource {
        fn read(&mut self, out: &mut [f32]) -> Result<CaptureRead> {
            thread::sleep(Duration::from_millis(100));
            let calls = self.total.fetch_add(1, Ordering::SeqCst) + 1;
            if calls >= self.limit {
                self.shared.stop.store(true, Ordering::SeqCst);
                return Ok(CaptureRead {
                    count: 0,
                    captured: None,
                });
            }
            if let Some(fail_after) = self.fail_after {
                if self.reads >= fail_after {
                    self.reads = 0;
                    anyhow::bail!("fake device removed");
                }
            }
            self.reads += 1;
            // A device change is visible on the next loop iteration, before the
            // next read, as a real watcher update is.
            if self.reopen_after.is_some_and(|after| self.reads >= after) {
                self.change = DeviceChange::Reopen;
            }
            if self.monitor_after.is_some_and(|after| self.reads >= after) {
                self.change = DeviceChange::Monitor;
            }
            out.fill(0.5);
            Ok(CaptureRead {
                count: out.len(),
                // A real stream reports when the last sample was captured.
                captured: Some(self.timeline.advance(Duration::from_millis(100))),
            })
        }

        fn device_change(&self) -> DeviceChange {
            self.change
        }
    }

    #[test]
    fn a_failed_capture_reopens_and_leaves_a_silent_gap() {
        let dir = tempfile::tempdir().unwrap();
        let shared = shared_for_capture(dir.path());
        let total = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let timeline = Timeline::new();
        let opener = {
            let shared = shared.clone();
            let total = total.clone();
            let timeline = timeline.clone();
            move |_track: Track| -> Result<Opened> {
                // The reopen loop waits 300 ms before it opens; the virtual
                // clock advances by the same amount so the gap is exact.
                timeline.advance(Duration::from_millis(300));
                Ok(Opened::Source(Box::new(FakeSource {
                    fail_after: None,
                    reopen_after: None,
                    monitor_after: None,
                    change: DeviceChange::None,
                    reads: 0,
                    total: total.clone(),
                    limit: 7,
                    shared: shared.clone(),
                    timeline: timeline.clone(),
                })))
            }
        };
        let first = Box::new(FakeSource {
            fail_after: Some(3),
            reopen_after: None,
            monitor_after: None,
            change: DeviceChange::None,
            reads: 0,
            total: total.clone(),
            limit: 7,
            shared: shared.clone(),
            timeline: timeline.clone(),
        });
        run_capture_with(shared.clone(), Track::Remote, first, opener);

        let errors = shared.recorder.lock().unwrap().stop();
        assert!(errors.is_empty(), "{errors:?}");
        let samples = track(dir.path(), Track::Remote);
        let chunks = tinta_core::chunks::read_chunks(dir.path(), &key(), Track::Remote).unwrap();
        assert_eq!(chunks.len(), 2);
        let first_end = (chunks[0].start + chunks[0].samples.len() as i64) as usize;
        let second_start = chunks[1].start as usize;
        assert_eq!(first_end - chunks[0].start as usize, 4_800);
        assert!(
            second_start >= first_end + 4_000,
            "the reopen gap is {} samples",
            second_start - first_end
        );
        // The 300 ms reopen delay (plus read timing) is silence, at the true
        // clock offset; audio resumes after it.
        assert_eq!(
            rms_between(&samples, first_end, second_start),
            0.0,
            "outage"
        );
        assert!(rms_between(&samples, second_start, second_start + 3_200) > 0.4);
    }

    #[test]
    fn a_changed_default_reopens_the_microphone() {
        let dir = tempfile::tempdir().unwrap();
        let shared = shared_for_capture(dir.path());
        let total = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let timeline = Timeline::new();
        let opens = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let opener = {
            let shared = shared.clone();
            let total = total.clone();
            let timeline = timeline.clone();
            let opens = opens.clone();
            move |_track: Track| -> Result<Opened> {
                opens.fetch_add(1, Ordering::SeqCst);
                // The default changed under the stream: the old one was closed
                // and the new device opened a little later.
                timeline.advance(Duration::from_millis(300));
                Ok(Opened::Source(Box::new(FakeSource {
                    fail_after: None,
                    reopen_after: None,
                    monitor_after: None,
                    change: DeviceChange::None,
                    reads: 0,
                    total: total.clone(),
                    limit: 8,
                    shared: shared.clone(),
                    timeline: timeline.clone(),
                })))
            }
        };
        let first = Box::new(FakeSource {
            fail_after: None,
            reopen_after: Some(2),
            monitor_after: None,
            change: DeviceChange::None,
            reads: 0,
            total: total.clone(),
            limit: 8,
            shared: shared.clone(),
            timeline: timeline.clone(),
        });
        run_microphone_with(
            shared.clone(),
            Some(first),
            opener,
            false,
            MIC_RETRY_INTERVAL,
        );
        assert_eq!(opens.load(Ordering::SeqCst), 1, "the microphone reopened");

        assert!(shared.recorder.lock().unwrap().stop().is_empty());
        let samples = track(dir.path(), Track::Mic);
        // Two reads, a 300 ms gap (silence), then the new device.
        assert!(rms_between(&samples, 0, 3_200) > 0.4, "before the change");
        assert_eq!(rms_between(&samples, 3_200, 8_000), 0.0, "the hand-over");
        assert!(rms_between(&samples, 8_000, 11_200) > 0.4, "the new device");
    }

    #[test]
    fn a_microphone_that_appears_later_resumes_the_track() {
        let dir = tempfile::tempdir().unwrap();
        // The meeting started without a microphone: no `audio/mic` directory.
        let (shared, sink) = shared_with_mic_track(dir.path(), false);
        assert!(!dir.path().join("audio/mic").exists());
        let total = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let timeline = Timeline::new();
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let opener = {
            let shared = shared.clone();
            let total = total.clone();
            let timeline = timeline.clone();
            let calls = calls.clone();
            move |_track: Track| -> Result<Opened> {
                // Time passes while the microphone is missing.
                timeline.advance(Duration::from_secs(2));
                if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                    return Ok(Opened::Unavailable(
                        "the default source is \"auto_null.monitor\", the speaker monitor"
                            .to_string(),
                    ));
                }
                Ok(Opened::Source(Box::new(FakeSource {
                    fail_after: None,
                    reopen_after: None,
                    monitor_after: None,
                    change: DeviceChange::None,
                    reads: 0,
                    total: total.clone(),
                    limit: 6,
                    shared: shared.clone(),
                    timeline: timeline.clone(),
                })))
            }
        };
        // `warned` is true: `start` already emitted the missing-microphone
        // warning, so only the recovery is announced.
        run_microphone_with(
            shared.clone(),
            None,
            opener,
            true,
            Duration::from_millis(20),
        );
        assert!(
            shared.recorder.lock().unwrap().has_mic(),
            "the track was created when the input appeared"
        );
        assert!(shared.recorder.lock().unwrap().stop().is_empty());

        let samples = track(dir.path(), Track::Mic);
        assert!(
            samples.len() > 70_000,
            "the track starts where the input appeared: {} samples",
            samples.len()
        );
        assert_eq!(rms_between(&samples, 0, 64_000), 0.0, "before the input");
        assert!(
            rms_between(&samples, 66_000, 70_000) > 0.4,
            "after the input"
        );
        let warnings: Vec<String> = events(&sink)
            .iter()
            .filter(|event| event["event"] == "warning")
            .map(|event| event["message"].as_str().unwrap_or_default().to_string())
            .collect();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("available again"), "{warnings:?}");
    }

    #[test]
    fn stop_does_not_wait_forever_on_a_wedged_capture_thread() {
        let blocked = thread::spawn(|| thread::sleep(Duration::from_millis(400)));
        let started = Instant::now();
        let wedged = join_workers(vec![blocked], Duration::from_millis(50));
        assert!(wedged, "the blocked thread must be reported");
        assert!(
            started.elapsed() < Duration::from_millis(300),
            "stop waited {:?}",
            started.elapsed()
        );
        // A thread that finishes inside the timeout is joined normally.
        let quick = thread::spawn(|| {});
        assert!(!join_workers(vec![quick], Duration::from_secs(1)));
    }

    #[test]
    fn a_microphone_that_becomes_a_monitor_is_closed_and_reported() {
        let dir = tempfile::tempdir().unwrap();
        let (shared, sink) = shared_with_mic_track(dir.path(), true);
        let timeline = Timeline::new();
        let total = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let opens = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let opener = {
            let shared = shared.clone();
            let opens = opens.clone();
            move |_track: Track| -> Result<Opened> {
                if opens.fetch_add(1, Ordering::SeqCst) >= 2 {
                    // The test ends here: stop the loop like a real session.
                    shared.stop.store(true, Ordering::SeqCst);
                }
                Ok(Opened::Unavailable(
                    "the default source is \"auto_null.monitor\", the speaker monitor".to_string(),
                ))
            }
        };
        let first = Box::new(FakeSource {
            fail_after: None,
            reopen_after: None,
            monitor_after: Some(3),
            change: DeviceChange::None,
            reads: 0,
            total: total.clone(),
            limit: 20,
            shared: shared.clone(),
            timeline: timeline.clone(),
        });
        run_microphone_with(
            shared.clone(),
            Some(first),
            opener,
            false,
            Duration::from_millis(20),
        );
        assert_eq!(opens.load(Ordering::SeqCst), 3, "the monitor was checked");
        assert!(shared.recorder.lock().unwrap().stop().is_empty());

        // The three reads before the change are all that reaches the track;
        // nothing from the monitor is stored as the microphone.
        let samples = track(dir.path(), Track::Mic);
        assert_eq!(samples.len(), 4_800, "no audio after the close");
        assert!(rms_between(&samples, 0, 4_800) > 0.4);
        let warnings: Vec<String> = events(&sink)
            .iter()
            .filter(|event| event["event"] == "warning")
            .map(|event| event["message"].as_str().unwrap_or_default().to_string())
            .collect();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(
            warnings[0].contains("records the meeting audio only"),
            "{warnings:?}"
        );
        assert!(warnings[0].contains("speaker monitor"), "{warnings:?}");
    }
}
