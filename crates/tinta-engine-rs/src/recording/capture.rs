//! Audio capture sources for live recording.
//!
//! Linux captures through the PulseAudio simple API. PipeWire's pulse server
//! serves the same API, so one implementation covers both. The two tracks are:
//!
//! - microphone: `@DEFAULT_SOURCE@`, the default input source;
//! - remote: `@DEFAULT_MONITOR@`, the monitor source of the default output sink.
//!
//! Those special names are resolved by the server every time a stream opens, so
//! reopening after a device error picks up a new default device. Audio is asked
//! for as 16 kHz mono `S16LE`; the server resamples from the device rate, so the
//! capture loop has nothing to resample.

use anyhow::Result;
use tinta_core::chunks::Track;

/// The engine's sample rate, the same as `ChunkFormat.sampleRate` in Swift.
pub const SAMPLE_RATE: u32 = 16_000;

/// One read is 100 ms of audio. A shorter period keeps `stop` and pause prompt;
/// a longer one risks a gap larger than the chunk writer's 100 ms snap.
pub const FRAGMENT_SAMPLES: usize = (SAMPLE_RATE / 10) as usize;

/// A blocking source of 16 kHz mono float samples.
pub trait CaptureSource: Send {
    /// Fills up to `out.len()` samples and returns how many were read. A read
    /// that lasts the whole fragment always returns `out.len()`.
    fn read(&mut self, out: &mut [f32]) -> Result<usize>;
}

/// Opens the microphone or the system-output monitor. `Err` carries a sentence
/// the app can show when the server or the device is missing.
pub fn open_capture(track: Track) -> Result<Box<dyn CaptureSource>> {
    platform::open(track)
}

#[cfg(target_os = "linux")]
mod platform {
    use super::{CaptureSource, FRAGMENT_SAMPLES, SAMPLE_RATE};
    use anyhow::{anyhow, Result};
    use libpulse_binding::context::{Context, FlagSet, State};
    use libpulse_binding::def::BufferAttr;
    use libpulse_binding::mainloop::standard::{IterateResult, Mainloop};
    use libpulse_binding::sample::{Format, Spec};
    use libpulse_binding::stream::Direction;
    use libpulse_simple_binding::Simple;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;
    use tinta_core::chunks::Track;

    /// Fallbacks for a server that does not resolve the special device names.
    const DEFAULT_INPUT: &str = "@DEFAULT_SOURCE@";
    const DEFAULT_MONITOR: &str = "@DEFAULT_MONITOR@";

    pub fn open(track: Track) -> Result<Box<dyn CaptureSource>> {
        Ok(Box::new(PulseCapture::open(track)?))
    }

    struct PulseCapture {
        simple: Simple,
        bytes: Vec<u8>,
    }

    impl PulseCapture {
        fn open(track: Track) -> Result<Self> {
            // PipeWire's pulse server does not resolve `@DEFAULT_MONITOR@` the
            // way PulseAudio does, so ask the server for the default devices
            // and open the sink monitor by name. A server that cannot answer
            // falls back to the special names.
            let defaults = server_defaults();
            let (device, description) = match track {
                Track::Mic => (
                    defaults
                        .as_ref()
                        .and_then(|(_, source)| source.clone())
                        .unwrap_or_else(|| DEFAULT_INPUT.to_string()),
                    "tinta microphone",
                ),
                Track::Remote => {
                    let monitor = defaults
                        .as_ref()
                        .and_then(|(sink, _)| sink.as_ref())
                        .map(|sink| format!("{sink}.monitor"))
                        .unwrap_or_else(|| DEFAULT_MONITOR.to_string());
                    (monitor, "tinta meeting audio")
                }
            };
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
            })
        }
    }

    /// Asks the server for its default sink and source names. `None` when the
    /// server does not answer, so opening can still try the special names.
    fn server_defaults() -> Option<(Option<String>, Option<String>)> {
        let mut mainloop = Mainloop::new()?;
        let mut context = Context::new(&mainloop, "tinta-engine devices")?;
        context.connect(None, FlagSet::NOFLAGS, None).ok()?;
        if !wait_until_ready(&mut mainloop, &mut context) {
            return None;
        }

        let slot = Arc::new(Mutex::new(None));
        let callback = slot.clone();
        context.introspect().get_server_info(move |info| {
            let sink = info.default_sink_name.as_ref().map(|name| name.to_string());
            let source = info
                .default_source_name
                .as_ref()
                .map(|name| name.to_string());
            if let Ok(mut slot) = callback.lock() {
                *slot = Some((sink, source));
            }
        });

        // The callback runs on this main loop. Poll rather than block, so a
        // server that never answers cannot hold up `start`.
        let mut defaults = None;
        for _ in 0..2_000 {
            if let Ok(mut slot) = slot.lock() {
                if let Some(value) = slot.take() {
                    defaults = Some(value);
                    break;
                }
            }
            match mainloop.iterate(false) {
                IterateResult::Quit(_) | IterateResult::Err(_) => break,
                _ => {}
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        context.disconnect();
        defaults
    }

    fn wait_until_ready(mainloop: &mut Mainloop, context: &mut Context) -> bool {
        for _ in 0..2_000 {
            match context.get_state() {
                State::Ready => return true,
                State::Failed | State::Terminated => return false,
                _ => {}
            }
            match mainloop.iterate(false) {
                IterateResult::Quit(_) | IterateResult::Err(_) => return false,
                _ => {}
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        false
    }

    impl CaptureSource for PulseCapture {
        fn read(&mut self, out: &mut [f32]) -> Result<usize> {
            let count = out.len().min(FRAGMENT_SAMPLES);
            let bytes = &mut self.bytes[..count * 2];
            self.simple
                .read(bytes)
                .map_err(|error| anyhow!("the audio stream failed: {error}"))?;
            let (pairs, _) = bytes.as_chunks::<2>();
            for (sample, pair) in out.iter_mut().zip(pairs) {
                *sample = i16::from_le_bytes(*pair) as f32 / i16::MAX as f32;
            }
            Ok(count)
        }
    }
}

#[cfg(not(target_os = "linux"))]
mod platform {
    use super::CaptureSource;
    use anyhow::{bail, Result};
    use tinta_core::chunks::Track;

    pub fn open(_track: Track) -> Result<Box<dyn CaptureSource>> {
        bail!("recording is not available on this platform yet")
    }
}
