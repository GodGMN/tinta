//! Audio capture sources for live recording.
//!
//! Linux captures through the PulseAudio simple API. PipeWire's pulse server
//! serves the same API, so one implementation covers both. The two tracks are:
//!
//! - microphone: the server's default source name, opened only when it is a
//!   real input;
//! - remote: `<default sink>.monitor`, the monitor source of the default
//!   output sink (`@DEFAULT_MONITOR@` when the server does not answer).
//!
//! The names are resolved from the server defaults on every (re)open. Audio is
//! asked for as 16 kHz mono `S16LE`; the server resamples from the device rate,
//! so the capture loop has nothing to resample.
//!
//! Every read carries the time the audio was captured, not the time the client
//! processed it: the server's record latency is subtracted from the moment the
//! read returns. The meeting clock can then place buffered audio where it was
//! recorded after a stall instead of shifting the whole track later.
//!
//! A machine with no physical microphone makes the monitor of its speakers the
//! default *source*. Recording that as the microphone duplicates every remote
//! utterance as the user, so every microphone open goes through
//! [`Defaults::mic_device`] and reports the monitor as missing instead. That
//! check is not limited to the start: a device that disappears mid-meeting makes
//! the server fall back to the monitor, and the microphone loop closes its
//! stream when it sees that.
//!
//! On plain PulseAudio a stream opened by name is not moved when the default
//! device changes. A small watcher polls the server defaults while a stream is
//! open, and each stream reports [`CaptureSource::device_change`] so the capture
//! loop reopens (or closes) it.

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

    /// What the capture loop should do before the next read because the server's
    /// defaults changed. [`DeviceChange::None`] by default.
    fn device_change(&self) -> DeviceChange {
        DeviceChange::None
    }
}

/// A change in the server defaults that affects an open stream.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DeviceChange {
    /// Nothing this stream cares about changed.
    #[default]
    None,
    /// This track's default device changed: reopen it by name.
    Reopen,
    /// The default input is now a speaker monitor. The stream is not a
    /// microphone any more, whatever its own name says: close it.
    Monitor,
}

/// What opening a capture stream found.
pub enum Opened {
    /// A stream to read.
    Source(Box<dyn CaptureSource>),
    /// No usable microphone: the sentence says why, for the log and the one
    /// warning the session emits. Only the microphone track returns this.
    Unavailable(String),
}

/// Opens the monitor of the default output sink.
pub fn open_remote() -> Result<Box<dyn CaptureSource>> {
    match platform::open_track(Track::Remote)? {
        Opened::Source(source) => Ok(source),
        // The remote track can always be opened by monitor name.
        Opened::Unavailable(reason) => Err(anyhow::anyhow!(reason)),
    }
}

/// Opens a track, or reports that the machine has no usable microphone.
///
/// A default source that is a sink monitor is not a microphone: on a machine
/// without an input device PulseAudio and PipeWire make `<sink>.monitor` the
/// default, and recording it would duplicate the meeting audio as the user.
/// This runs for every microphone open, not just the start, because the server
/// falls back to the monitor when the input device disappears.
pub fn open_capture(track: Track) -> Result<Opened> {
    platform::open_track(track)
}

#[cfg(target_os = "linux")]
mod platform {
    use super::{CaptureRead, CaptureSource, DeviceChange, Opened, FRAGMENT_SAMPLES, SAMPLE_RATE};
    use anyhow::{anyhow, Result};
    use libpulse_binding::context::subscribe::InterestMaskSet;
    use libpulse_binding::context::{Context, FlagSet, State};
    use libpulse_binding::def::BufferAttr;
    use libpulse_binding::mainloop::standard::{IterateResult, Mainloop};
    use libpulse_binding::sample::{Format, Spec};
    use libpulse_binding::stream::Direction;
    use libpulse_binding::time::MicroSeconds;
    use libpulse_simple_binding::Simple;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::thread;
    use std::time::{Duration, Instant};
    use tinta_core::chunks::Track;

    /// The fallback for a server that does not answer its default sink. The
    /// microphone has no such fallback: without an answer there is no way to
    /// tell the monitor from a real input, so it reports as missing.
    const DEFAULT_MONITOR: &str = "@DEFAULT_MONITOR@";
    /// How often the watcher asks the server for its default devices while a
    /// stream is open.
    const POLL_INTERVAL: Duration = Duration::from_secs(2);
    /// How long the watcher sleeps with no stream open, before it checks again.
    const IDLE_INTERVAL: Duration = Duration::from_millis(250);
    /// How long the watcher waits for a server event before looping again.
    /// Server events (a source removed, a default changed) wake it at once.
    const WATCH_PUMP: Duration = Duration::from_millis(200);
    /// The first delay before the watcher reopens a connection to a server that
    /// stopped answering; doubled up to [`POLL_INTERVAL`].
    const RECONNECT_DELAY: Duration = Duration::from_millis(500);
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

        /// The microphone source to open, or why the machine has none.
        ///
        /// A sink monitor is not a microphone, and a server that reports no
        /// default source has no input at all. The name suffix and the
        /// server's `monitor_of_sink` lookup both mean monitor: PipeWire names
        /// its monitors `<sink>.monitor`, and a renamed or virtual monitor is
        /// caught by the introspection flag.
        fn mic_device(&self) -> std::result::Result<String, String> {
            let Some(source) = &self.source else {
                return Err("the server has no input source".to_string());
            };
            if self.source_is_monitor || source.ends_with(".monitor") {
                return Err(format!(
                    "the default source is \"{source}\", the speaker monitor"
                ));
            }
            Ok(source.clone())
        }
    }

    /// The last defaults the watcher saw, shared with every open stream.
    struct Devices {
        current: Mutex<Option<Defaults>>,
        /// Bumped whenever the defaults change, so a stream can notice cheaply.
        generation: AtomicU64,
        /// Set when the last query failed: the snapshot may name devices that
        /// are gone, so an open must ask the server instead of trusting it.
        stale: AtomicBool,
        /// Live streams, i.e. `PulseCapture`s. The watcher only polls while
        /// there is one; a machine that is not recording spends nothing on it.
        users: AtomicUsize,
    }

    impl Devices {
        fn snapshot(&self) -> Option<Defaults> {
            self.current
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
        }

        fn store(&self, defaults: Defaults) {
            self.stale.store(false, Ordering::SeqCst);
            let mut current = self
                .current
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if current.as_ref() != Some(&defaults) {
                *current = Some(defaults);
                self.generation.fetch_add(1, Ordering::SeqCst);
            }
        }

        fn mark_stale(&self) {
            self.stale.store(true, Ordering::SeqCst);
        }

        fn is_stale(&self) -> bool {
            self.stale.load(Ordering::SeqCst)
        }

        fn acquire(self: &Arc<Self>) {
            self.users.fetch_add(1, Ordering::SeqCst);
        }

        fn release(&self) {
            let previous = self.users.fetch_sub(1, Ordering::SeqCst);
            debug_assert!(previous > 0, "released without an acquire");
        }

        fn has_users(&self) -> bool {
            self.users.load(Ordering::SeqCst) > 0
        }
    }

    static DEVICES: OnceLock<Arc<Devices>> = OnceLock::new();

    /// The shared watcher, started on first use. It polls the server for its
    /// default devices while a stream is open, and reconnects after a server
    /// restart instead of querying a dead connection forever.
    fn devices() -> Arc<Devices> {
        DEVICES
            .get_or_init(|| {
                let devices = Arc::new(Devices {
                    current: Mutex::new(None),
                    generation: AtomicU64::new(0),
                    stale: AtomicBool::new(true),
                    users: AtomicUsize::new(0),
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
        let mut connection: Option<Connection> = None;
        let mut delay = RECONNECT_DELAY;
        let mut logged = false;
        let mut queried: Option<Instant> = None;
        loop {
            if !devices.has_users() {
                // Nothing is recording: keep no server connection and stop
                // trusting the snapshot, so the next open asks the server.
                connection = None;
                queried = None;
                devices.mark_stale();
                thread::sleep(IDLE_INTERVAL);
                continue;
            }
            if connection.is_none() {
                connection = Connection::open();
                if connection.is_none() {
                    devices.mark_stale();
                    if !logged {
                        logged = true;
                        crate::protocol::log(
                            "cannot reach the audio server; the device watcher retries",
                        );
                    }
                    thread::sleep(delay);
                    delay = (delay * 2).min(POLL_INTERVAL);
                    continue;
                }
            }
            let Some(current) = connection.as_mut() else {
                continue;
            };
            // Wait for a server event instead of sleeping through the poll
            // interval: a device being unplugged publishes a source change at
            // once, and the microphone must close before the server can move
            // its stream onto the speaker monitor for long.
            if !current.pump(WATCH_PUMP) {
                devices.mark_stale();
                connection = None;
                continue;
            }
            let changed = current.take_event();
            let due = queried.is_none_or(|at| at.elapsed() >= POLL_INTERVAL);
            if !changed && !due {
                continue;
            }
            match current.query() {
                Some(defaults) => {
                    if logged {
                        logged = false;
                        crate::protocol::log("the device watcher reconnected");
                    }
                    delay = RECONNECT_DELAY;
                    queried = Some(Instant::now());
                    devices.store(defaults);
                }
                None => {
                    // The connection is dead or the server restarted: drop it
                    // and reopen with backoff, and stop trusting the snapshot.
                    devices.mark_stale();
                    connection = None;
                    queried = None;
                    thread::sleep(delay);
                    delay = (delay * 2).min(POLL_INTERVAL);
                }
            }
        }
    }

    /// One introspection connection, driven without blocking so a server that
    /// stops answering cannot hold up a device open.
    struct Connection {
        /// Declared before the main loop on purpose: fields drop in declaration
        /// order, and the context must be released before the loop it drives.
        context: Context,
        mainloop: Mainloop,
        /// Set by the subscription callback when the server publishes a change
        /// to its sources, sinks or defaults.
        events: Arc<AtomicBool>,
    }

    impl Connection {
        fn open() -> Option<Self> {
            let mainloop = Mainloop::new()?;
            let mut context = Context::new(&mainloop, "tinta-engine devices")?;
            let events = Arc::new(AtomicBool::new(false));
            context.connect(None, FlagSet::NOFLAGS, None).ok()?;
            let mut connection = Self {
                context,
                mainloop,
                events,
            };
            if !connection.wait_until_ready() {
                return None;
            }
            // Wake the watcher on device changes instead of waiting for the
            // next poll: a source that disappears publishes an event at once.
            let flag = connection.events.clone();
            connection
                .context
                .set_subscribe_callback(Some(Box::new(move |_, _, _| {
                    flag.store(true, Ordering::SeqCst);
                })));
            connection.context.subscribe(
                InterestMaskSet::SINK | InterestMaskSet::SOURCE | InterestMaskSet::SERVER,
                |_| {},
            );
            Some(connection)
        }

        /// Dispatches whatever the server sent, waiting up to `timeout` for an
        /// event. Returns false when the connection has died.
        fn pump(&mut self, timeout: Duration) -> bool {
            if self
                .mainloop
                .prepare(Some(MicroSeconds(timeout.as_micros() as u64)))
                .is_err()
                || self.mainloop.poll().is_err()
            {
                return false;
            }
            self.mainloop.dispatch().is_ok()
        }

        /// Takes the "something changed" flag the subscription set.
        fn take_event(&self) -> bool {
            self.events.swap(false, Ordering::SeqCst)
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
            // A server with no input device has no default source. Keep the
            // answer: a whole answer discarded here would also lose the sink.
            let is_monitor = source
                .as_deref()
                .is_some_and(|source| self.source_is_monitor(source));
            Some(Defaults {
                sink,
                source,
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

    pub(super) fn open_track(track: Track) -> Result<Opened> {
        let devices = devices();
        match track {
            Track::Mic => {
                let Some(defaults) = defaults_now(&devices) else {
                    // Without an answer there is no way to tell a real input
                    // from the speaker monitor, so record no microphone.
                    return Ok(Opened::Unavailable(
                        "the server did not answer its default devices".to_string(),
                    ));
                };
                let device = match defaults.mic_device() {
                    Ok(device) => device,
                    Err(reason) => return Ok(Opened::Unavailable(reason)),
                };
                let capture = PulseCapture::open(Track::Mic, device, "tinta microphone", devices)?;
                Ok(Opened::Source(Box::new(capture)))
            }
            Track::Remote => {
                let monitor = defaults_now(&devices)
                    .and_then(|defaults| defaults.sink)
                    .map(|sink| format!("{sink}.monitor"))
                    .unwrap_or_else(|| DEFAULT_MONITOR.to_string());
                let capture =
                    PulseCapture::open(Track::Remote, monitor, "tinta meeting audio", devices)?;
                Ok(Opened::Source(Box::new(capture)))
            }
        }
    }

    /// The shared snapshot, refreshed synchronously when the watcher has not
    /// answered yet or its last answer failed.
    fn defaults_now(devices: &Arc<Devices>) -> Option<Defaults> {
        if let Some(defaults) = devices.snapshot() {
            if !devices.is_stale() {
                return Some(defaults);
            }
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

    impl Drop for PulseCapture {
        fn drop(&mut self) {
            self.devices.release();
        }
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
            devices.acquire();
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
            //
            // A failed query must not become a zero latency: that would place
            // the read later than it was captured and leave a gap, then a drop
            // on the next good read. `None` makes the meeting clock fall back to
            // sample counting instead. No clamp: a latency of many seconds is a
            // real backlog after a stall, and truncating it would place the
            // buffered audio late and then drop the excess.
            let completed = Instant::now();
            let captured = self
                .simple
                .get_latency()
                .ok()
                .and_then(|micros| completed.checked_sub(Duration::from_micros(micros.0)));
            let (pairs, _) = bytes.as_chunks::<2>();
            for (sample, pair) in out.iter_mut().zip(pairs) {
                *sample = i16::from_le_bytes(*pair) as f32 / i16::MAX as f32;
            }
            Ok(CaptureRead { count, captured })
        }

        fn flush(&mut self) -> Result<()> {
            // Record streams buffer audio before the first read; dropping it
            // keeps the meeting start aligned with the other track.
            self.simple
                .flush()
                .map_err(|error| anyhow!("cannot flush the audio stream: {error}"))
        }

        fn device_change(&self) -> DeviceChange {
            let generation = self.devices.generation.load(Ordering::SeqCst);
            if generation == self.generation {
                return DeviceChange::None;
            }
            let Some(defaults) = self.devices.snapshot() else {
                return DeviceChange::None;
            };
            // The server may have rescued a vanished input by moving this
            // stream onto a monitor without a read error. The stream's own
            // name does not change, so check the default, not `self.device`.
            if self.track == Track::Mic && defaults.source_is_monitor {
                return DeviceChange::Monitor;
            }
            // Otherwise reopen only when this track's device changed.
            match defaults.device(self.track) {
                Some(device) if device != self.device => DeviceChange::Reopen,
                _ => DeviceChange::None,
            }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn defaults(source: Option<&str>, source_is_monitor: bool) -> Defaults {
            Defaults {
                sink: Some("alsa_output.headphones".to_string()),
                source: source.map(str::to_string),
                source_is_monitor,
            }
        }

        #[test]
        fn a_monitor_source_is_not_a_microphone() {
            // A normal input opens.
            assert_eq!(
                defaults(Some("alsa_input.usb-headset"), false)
                    .mic_device()
                    .unwrap(),
                "alsa_input.usb-headset"
            );
            // The `<sink>.monitor` suffix is the common fallback.
            let error = defaults(Some("auto_null.monitor"), false)
                .mic_device()
                .unwrap_err();
            assert!(error.contains("auto_null.monitor"), "{error}");
            assert!(error.contains("speaker monitor"), "{error}");
            // The `monitor_of_sink` lookup catches a monitor whose name does not
            // end in `.monitor`; the suffix alone would miss it.
            let error = defaults(Some("virtual-monitor"), true)
                .mic_device()
                .unwrap_err();
            assert!(error.contains("speaker monitor"), "{error}");
        }

        #[test]
        fn a_server_without_an_input_source_reports_it() {
            let error = defaults(None, false).mic_device().unwrap_err();
            assert_eq!(error, "the server has no input source");
        }

        #[test]
        fn a_failed_query_marks_the_snapshot_stale_and_a_store_clears_it() {
            let devices = Devices {
                current: Mutex::new(None),
                generation: AtomicU64::new(0),
                stale: AtomicBool::new(true),
                users: AtomicUsize::new(0),
            };
            assert!(devices.is_stale());
            devices.store(defaults(Some("alsa_input.usb"), false));
            assert!(!devices.is_stale());
            assert_eq!(devices.generation.load(Ordering::SeqCst), 1);
            // The same answer changes nothing.
            devices.store(defaults(Some("alsa_input.usb"), false));
            assert_eq!(devices.generation.load(Ordering::SeqCst), 1);
            devices.mark_stale();
            assert!(devices.is_stale());
            assert_eq!(
                devices.snapshot().unwrap().source.as_deref(),
                Some("alsa_input.usb")
            );
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::Opened;
    use anyhow::{bail, Result};
    use tinta_core::chunks::Track;

    pub(super) fn open_track(_track: Track) -> Result<Opened> {
        bail!("recording is not available on this platform yet")
    }
}
