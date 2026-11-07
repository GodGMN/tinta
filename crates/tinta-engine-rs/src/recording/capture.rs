//! Audio capture sources for live recording.
//!
//! Linux captures through the PulseAudio simple API. PipeWire's pulse server
//! serves the same API, so one implementation covers both. The two tracks are:
//!
//! - microphone: `@DEFAULT_SOURCE@`, the default input source;
//! - remote: `@DEFAULT_MONITOR@`, the monitor source of the default output sink.
//!
//! Those special names are resolved by the server every time a stream opens.
//! Audio is asked for as 16 kHz mono `S16LE`; the server resamples from the
//! device rate, so the capture loop has nothing to resample.
//!
//! Every read carries the time the audio was captured, not the time the client
//! processed it: the server's record latency is subtracted from the moment the
//! read returns. The meeting clock can then place buffered audio where it was
//! recorded after a stall instead of shifting the whole track later.
//!
//! A machine with no physical microphone makes the monitor of its speakers the
//! default *source*. Recording that as the microphone duplicates every remote
//! utterance as the user, so [`open_microphone`] reports it as missing instead.
//!
//! On plain PulseAudio a stream opened by name is not moved when the default
//! device changes. A small watcher polls the server defaults, and each stream
//! reports [`CaptureSource::needs_reopen`] so the capture loop reopens it.

use anyhow::Result;
use std::time::Instant;
use tinta_core::chunks::Track;

/// The engine's sample rate, the same as `ChunkFormat.sampleRate` in Swift.
pub const SAMPLE_RATE: u32 = 16_000;

/// One read is 100 ms of audio. A shorter period keeps `stop` and pause prompt;
/// a longer one risks a gap larger than the chunk writer's 100 ms snap.
pub const FRAGMENT_SAMPLES: usize = (SAMPLE_RATE / 10) as usize;

/// One finished read.
pub struct CaptureRead {
    /// How many samples were written to the caller's buffer.
    pub count: usize,
    /// The capture time of the last sample, on the caller's `Instant` clock. `None`
    /// when the source cannot report it (a test source, or a server without
    /// timing information); the meeting clock then falls back to sample
    /// counting.
    pub captured: Option<Instant>,
}

/// A blocking source of 16 kHz mono float samples.
pub trait CaptureSource: Send {
    /// Fills up to `out.len()` samples and returns how many were read, with the
    /// time the last sample was captured. A read that lasts the whole fragment
    /// always returns `out.len()`.
    fn read(&mut self, out: &mut [f32]) -> Result<CaptureRead>;

    /// Throws away the audio the server buffered before the meeting started.
    fn flush(&mut self) -> Result<()> {
        Ok(())
    }

    /// True when the server's default device changed under this stream, so the
    /// capture loop should reopen it. False by default.
    fn needs_reopen(&self) -> bool {
        false
    }
}

/// What the microphone lookup found.
pub enum Microphone {
    /// A real input source.
    Capture(Box<dyn CaptureSource>),
    /// No physical microphone: the sentence explains why, for the log and the
    /// one warning the session emits.
    Missing(String),
}

/// Opens the monitor of the default output sink.
pub fn open_remote() -> Result<Box<dyn CaptureSource>> {
    platform::open(Track::Remote)
}

/// Opens the default input source, or reports that the machine has none.
///
/// A default source that is a sink monitor is not a microphone: on a machine
/// without an input device PulseAudio and PipeWire make `<sink>.monitor` the
/// default, and recording it would duplicate the meeting audio as the user.
pub fn open_microphone() -> Result<Microphone> {
    platform::open_microphone()
}

/// Opens a track by name. Used when a failed stream is reopened.
pub fn open_capture(track: Track) -> Result<Box<dyn CaptureSource>> {
    platform::open(track)
}

#[cfg(target_os = "linux")]
mod platform {
    use super::{CaptureRead, CaptureSource, Microphone, FRAGMENT_SAMPLES, SAMPLE_RATE};
    use anyhow::{anyhow, Result};
    use libpulse_binding::context::{Context, FlagSet, State};
    use libpulse_binding::def::BufferAttr;
    use libpulse_binding::mainloop::standard::{IterateResult, Mainloop};
    use libpulse_binding::sample::{Format, Spec};
    use libpulse_binding::stream::Direction;
    use libpulse_binding::time::MicroSeconds;
    use libpulse_simple_binding::Simple;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::thread;
    use std::time::{Duration, Instant};
    use tinta_core::chunks::Track;

    /// Fallbacks for a server that does not resolve the special device names.
    const DEFAULT_INPUT: &str = "@DEFAULT_SOURCE@";
    const DEFAULT_MONITOR: &str = "@DEFAULT_MONITOR@";
    /// How often the watcher asks the server for its default devices.
    const POLL_INTERVAL: Duration = Duration::from_secs(2);
    /// A latency larger than this is a broken answer, not a real backlog.
    const MAX_LATENCY: Duration = Duration::from_secs(5);
    /// The longest a single introspection answer is waited for.
    const ANSWER_TIMEOUT: Duration = Duration::from_secs(2);

    /// The server's default devices, with the monitor check for the source.
    #[derive(Clone, Debug, Default, PartialEq, Eq)]
    struct Defaults {
        sink: Option<String>,
        source: Option<String>,
        /// Whether `source` is a sink monitor rather than a physical input.
        source_is_monitor: bool,
    }

    impl Defaults {
        /// The device this track should be using, by name.
        fn device(&self, track: Track) -> Option<String> {
            match track {
                Track::Mic => self.source.clone(),
                Track::Remote => self.sink.as_ref().map(|sink| format!("{sink}.monitor")),
            }
        }
    }

    /// The last defaults the watcher saw, shared with every open stream.
    struct Devices {
        current: Mutex<Option<Defaults>>,
        /// Bumped whenever the defaults change, so a stream can notice cheaply.
        generation: AtomicU64,
    }

    impl Devices {
        fn snapshot(&self) -> Option<Defaults> {
            self.current
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
        }

        fn store(&self, defaults: Defaults) {
            let mut current = self
                .current
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if current.as_ref() != Some(&defaults) {
                *current = Some(defaults);
                self.generation.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    static DEVICES: OnceLock<Arc<Devices>> = OnceLock::new();

    /// The shared watcher, started on first use. It owns one introspection
    /// connection for the life of the process and polls it every two seconds.
    fn devices() -> Arc<Devices> {
        DEVICES
            .get_or_init(|| {
                let devices = Arc::new(Devices {
                    current: Mutex::new(None),
                    generation: AtomicU64::new(0),
                });
                let watched = devices.clone();
                let spawned = thread::Builder::new()
                    .name("device-watch".to_string())
                    .spawn(move || watch(watched));
                if let Err(error) = spawned {
                    crate::protocol::log(&format!(
                        "cannot start the default-device watcher: {error}"
                    ));
                }
                devices
            })
            .clone()
    }

    fn watch(devices: Arc<Devices>) {
        let mut connection = match Connection::open() {
            Some(connection) => connection,
            None => return,
        };
        loop {
            if let Some(defaults) = connection.query() {
                devices.store(defaults);
            }
            thread::sleep(POLL_INTERVAL);
        }
    }

    /// One introspection connection, driven without blocking so a server that
    /// stops answering cannot hold up a device open.
    struct Connection {
        /// Declared before the main loop on purpose: fields drop in declaration
        /// order, and the context must be released before the loop it drives.
        context: Context,
        mainloop: Mainloop,
    }

    impl Connection {
        fn open() -> Option<Self> {
            let mainloop = Mainloop::new()?;
            let mut context = Context::new(&mainloop, "tinta-engine devices")?;
            context.connect(None, FlagSet::NOFLAGS, None).ok()?;
            let mut connection = Self { context, mainloop };
            if connection.wait_until_ready() {
                Some(connection)
            } else {
                None
            }
        }

        fn wait_until_ready(&mut self) -> bool {
            let deadline = Instant::now() + ANSWER_TIMEOUT;
            loop {
                match self.context.get_state() {
                    State::Ready => return true,
                    State::Failed | State::Terminated => return false,
                    _ => {}
                }
                if Instant::now() >= deadline || !self.iterate() {
                    return false;
                }
                // Do not spin while the server writes the answer.
                thread::sleep(Duration::from_millis(1));
            }
        }

        /// Runs one non-blocking iteration of the main loop.
        fn iterate(&mut self) -> bool {
            match self.mainloop.iterate(false) {
                IterateResult::Quit(_) | IterateResult::Err(_) => false,
                IterateResult::Success(_) => true,
            }
        }

        /// Waits for a callback to fill `slot`, with a deadline.
        fn answer<T>(&mut self, slot: &Arc<Mutex<Option<T>>>) -> Option<T> {
            let deadline = Instant::now() + ANSWER_TIMEOUT;
            loop {
                if let Some(value) = slot
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take()
                {
                    return Some(value);
                }
                if Instant::now() >= deadline {
                    return None;
                }
                // Block in the poll for a few milliseconds instead of spinning.
                if self.mainloop.prepare(Some(MicroSeconds(10_000))).is_err()
                    || self.mainloop.poll().is_err()
                {
                    return None;
                }
                if self.mainloop.dispatch().is_err() {
                    return None;
                }
            }
        }

        /// Asks the server for its defaults, including whether the default
        /// source is a sink monitor.
        fn query(&mut self) -> Option<Defaults> {
            let slot = Arc::new(Mutex::new(None));
            let callback = slot.clone();
            self.context.introspect().get_server_info(move |info| {
                let sink = info.default_sink_name.as_ref().map(|name| name.to_string());
                let source = info
                    .default_source_name
                    .as_ref()
                    .map(|name| name.to_string());
                if let Ok(mut slot) = callback.lock() {
                    *slot = Some((sink, source));
                }
            });
            let (sink, source) = self.answer(&slot)?;
            let source = source?;
            let is_monitor = self.source_is_monitor(&source);
            Some(Defaults {
                sink,
                source: Some(source),
                source_is_monitor: is_monitor,
            })
        }

        /// Looks the source up by name. The monitor flag or a `.monitor` suffix
        /// both mean the source is a speaker monitor.
        fn source_is_monitor(&mut self, name: &str) -> bool {
            if name.ends_with(".monitor") {
                return true;
            }
            let slot: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None));
            let callback = slot.clone();
            self.context
                .introspect()
                .get_source_info_by_name(name, move |result| {
                    use libpulse_binding::callbacks::ListResult;
                    let is_monitor = match result {
                        ListResult::Item(info) => info.monitor_of_sink.is_some(),
                        _ => false,
                    };
                    if let Ok(mut slot) = callback.lock() {
                        *slot = Some(is_monitor);
                    }
                });
            self.answer(&slot).unwrap_or(false)
        }
    }

    pub(super) fn open(track: Track) -> Result<Box<dyn CaptureSource>> {
        let devices = devices();
        let defaults = defaults_now(&devices);
        let (device, description) = match track {
            Track::Mic => (
                defaults
                    .as_ref()
                    .and_then(|defaults| defaults.source.clone())
                    .unwrap_or_else(|| DEFAULT_INPUT.to_string()),
                "tinta microphone",
            ),
            Track::Remote => {
                let monitor = defaults
                    .as_ref()
                    .and_then(|defaults| defaults.sink.as_ref())
                    .map(|sink| format!("{sink}.monitor"))
                    .unwrap_or_else(|| DEFAULT_MONITOR.to_string());
                (monitor, "tinta meeting audio")
            }
        };
        Ok(Box::new(PulseCapture::open(
            track,
            device,
            description,
            devices,
        )?))
    }

    pub(super) fn open_microphone() -> Result<Microphone> {
        let devices = devices();
        let defaults = defaults_now(&devices);
        if let Some(defaults) = &defaults {
            let Some(source) = defaults.source.clone() else {
                return Ok(Microphone::Missing(
                    "the server has no input source".to_string(),
                ));
            };
            if defaults.source_is_monitor || source.ends_with(".monitor") {
                return Ok(Microphone::Missing(format!(
                    "the default source is \"{source}\", the speaker monitor"
                )));
            }
            let capture = PulseCapture::open(Track::Mic, source, "tinta microphone", devices)?;
            return Ok(Microphone::Capture(Box::new(capture)));
        }
        // A server that does not answer may still resolve the special name.
        let capture = PulseCapture::open(
            Track::Mic,
            DEFAULT_INPUT.to_string(),
            "tinta microphone",
            devices,
        )?;
        Ok(Microphone::Capture(Box::new(capture)))
    }

    /// The shared snapshot, filled synchronously when the watcher has not
    /// answered yet (the first open of a fresh process).
    fn defaults_now(devices: &Arc<Devices>) -> Option<Defaults> {
        if let Some(defaults) = devices.snapshot() {
            return Some(defaults);
        }
        let mut connection = Connection::open()?;
        let defaults = connection.query()?;
        devices.store(defaults.clone());
        Some(defaults)
    }

    struct PulseCapture {
        simple: Simple,
        bytes: Vec<u8>,
        track: Track,
        device: String,
        devices: Arc<Devices>,
        /// The watcher generation this stream was opened in, so the check only
        /// costs an atomic load while nothing changed.
        generation: u64,
    }

    impl PulseCapture {
        fn open(
            track: Track,
            device: String,
            description: &str,
            devices: Arc<Devices>,
        ) -> Result<Self> {
            let spec = Spec {
                format: Format::S16le,
                channels: 1,
                rate: SAMPLE_RATE,
            };
            if !spec.is_valid() {
                return Err(anyhow!("the capture sample format is not valid"));
            }
            // 0xFFFFFFFF is -1 for the server: use its defaults for everything
            // except the fragment, which bounds how long one read blocks.
            let attr = BufferAttr {
                maxlength: u32::MAX,
                tlength: u32::MAX,
                prebuf: u32::MAX,
                minreq: u32::MAX,
                fragsize: (FRAGMENT_SAMPLES * 2) as u32,
            };
            // Read the generation before opening: a change that lands during
            // the open must leave this stream stale so it reopens at once.
            let generation = devices.generation.load(Ordering::SeqCst);
            let simple = Simple::new(
                None,
                "tinta-engine",
                Direction::Record,
                Some(&device),
                description,
                &spec,
                None,
                Some(&attr),
            )
            .map_err(|error| anyhow!("{error} (is PulseAudio or PipeWire running?)"))?;
            Ok(Self {
                simple,
                bytes: vec![0u8; FRAGMENT_SAMPLES * 2],
                track,
                device,
                devices,
                generation,
            })
        }
    }

    impl CaptureSource for PulseCapture {
        fn read(&mut self, out: &mut [f32]) -> Result<CaptureRead> {
            let count = out.len().min(FRAGMENT_SAMPLES);
            let bytes = &mut self.bytes[..count * 2];
            self.simple
                .read(bytes)
                .map_err(|error| anyhow!("the audio stream failed: {error}"))?;
            // The record latency tells how far behind the capture head the read
            // head is, so `now - latency` is when the audio we just read was
            // recorded (the read drained the server buffer up to that point).
            let completed = Instant::now();
            let latency = self
                .simple
                .get_latency()
                .map(|micros| Duration::from_micros(micros.0).min(MAX_LATENCY))
                .unwrap_or(Duration::ZERO);
            let captured = completed.checked_sub(latency).unwrap_or(completed);
            let (pairs, _) = bytes.as_chunks::<2>();
            for (sample, pair) in out.iter_mut().zip(pairs) {
                *sample = i16::from_le_bytes(*pair) as f32 / i16::MAX as f32;
            }
            Ok(CaptureRead {
                count,
                captured: Some(captured),
            })
        }

        fn flush(&mut self) -> Result<()> {
            // Record streams buffer audio before the first read; dropping it
            // keeps the meeting start aligned with the other track.
            self.simple
                .flush()
                .map_err(|error| anyhow!("cannot flush the audio stream: {error}"))
        }

        fn needs_reopen(&self) -> bool {
            let generation = self.devices.generation.load(Ordering::SeqCst);
            if generation == self.generation {
                return false;
            }
            // The defaults changed: reopen only when this track's device did.
            self.devices
                .snapshot()
                .and_then(|defaults| defaults.device(self.track))
                .is_some_and(|device| device != self.device)
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::{CaptureSource, Microphone};
    use anyhow::{bail, Result};
    use tinta_core::chunks::Track;

    pub(super) fn open(_track: Track) -> Result<Box<dyn CaptureSource>> {
        bail!("recording is not available on this platform yet")
    }

    pub(super) fn open_microphone() -> Result<Microphone> {
        bail!("recording is not available on this platform yet")
    }
}
