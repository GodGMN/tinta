//! Live draft transcripts.
//!
//! Swift detects speech with the VAD while recording and transcribes each
//! utterance, emitting a `live` event with provisional text. This port does the
//! same in one background thread: the capture threads push stored audio through
//! a small bounded channel, the thread feeds a streaming Silero VAD per track,
//! and each finished speech segment is decoded with the same recognizer the
//! final pass uses.
//!
//! The worker runs at one or two ASR threads and only when the models are
//! installed (the engine checks before starting it). Set `TINTA_ENGINE_LIVE=0`
//! to turn live drafts off; the final pass still transcribes everything.
//!
//! The channel is bounded, so a slow recognizer drops draft pieces instead of
//! growing memory. A dropped piece becomes a zero-filled gap in the VAD stream,
//! which keeps the `live` times on the meeting clock.

use anyhow::{bail, Result};
use serde_json::json;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Arc, Weak};
use std::thread::{self, JoinHandle};

use crate::audio::SAMPLE_RATE;
use crate::recording::Shared;
use crate::speech::SpeechPass;
use crate::threads::Threads;
use sherpa_onnx::VoiceActivityDetector;
use tinta_core::chunks::Track;

/// 32 pieces of 100 ms; older drafts are dropped rather than buffered.
const LIVE_QUEUE: usize = 32;
/// The Silero window size used by the VAD configuration.
const VAD_WINDOW: usize = 512;
/// A jump beyond this is a pause, not a late piece: reset the VAD instead of
/// feeding half a minute of zeros.
const MAX_GAP_SAMPLES: i64 = 30 * SAMPLE_RATE as i64;

/// One stored piece of audio, in meeting-clock order.
pub(crate) struct LivePiece {
    pub track: Track,
    pub index: i64,
    pub samples: Vec<f32>,
}

/// Owns the live worker and the sender the recorder pushes to.
pub struct LiveDrafts {
    sender: Option<SyncSender<LivePiece>>,
    worker: Option<JoinHandle<()>>,
}

impl LiveDrafts {
    pub(crate) fn spawn(
        config: crate::recording::LiveConfig,
        shared: &Arc<Shared>,
    ) -> Result<Self> {
        let (sender, receiver) = sync_channel(LIVE_QUEUE);
        let recorder_sender = sender.clone();
        match shared.recorder.lock() {
            Ok(mut recorder) => recorder.set_live(recorder_sender),
            Err(_) => bail!("the recording state is poisoned"),
        }
        let weak = Arc::downgrade(shared);
        let worker = thread::Builder::new()
            .name("live-drafts".to_string())
            .spawn(move || run(config, receiver, weak))
            .map_err(|error| anyhow::anyhow!("cannot start the live transcript thread: {error}"))?;
        Ok(Self {
            sender: Some(sender),
            worker: Some(worker),
        })
    }

    /// Closes the channel. The worker finishes the segment it is decoding and
    /// exits; stopping a recording never waits on the model.
    pub fn finish(mut self) {
        self.sender = None;
        if let Some(worker) = self.worker.take() {
            drop(worker);
        }
    }
}

/// One track's streaming VAD state.
struct TrackVad {
    vad: VoiceActivityDetector,
    /// Meeting-clock index of the first sample given to `vad`.
    offset: i64,
    /// Meeting-clock index of the next sample to give to `vad`.
    fed: i64,
}

impl TrackVad {
    fn new(vad: VoiceActivityDetector) -> Self {
        Self {
            vad,
            offset: 0,
            fed: 0,
        }
    }
}

fn run(config: crate::recording::LiveConfig, receiver: Receiver<LivePiece>, shared: Weak<Shared>) {
    let Some(strong) = shared.upgrade() else {
        return;
    };
    // Live drafts are a background nicety: keep them to at most two threads.
    let threads = Threads {
        asr: config.threads.asr.clamp(1, 2),
        ..config.threads
    };
    let mut pass = SpeechPass::new(config.manifest, config.models_root, threads);
    let vads = crate::speech::with_models(|| {
        Ok::<_, anyhow::Error>((
            TrackVad::new(pass.live_vad()?),
            TrackVad::new(pass.live_vad()?),
        ))
    });
    let (mut mic, mut remote) = match vads {
        Ok(vads) => vads,
        Err(_) => {
            strong.emit(
                "warning",
                json!({
                    "message": "Live transcription is off: the speech models are not installed.",
                }),
            );
            return;
        }
    };

    while let Ok(piece) = receiver.recv() {
        if strong.stopping() {
            break;
        }
        let state = match piece.track {
            Track::Mic => &mut mic,
            Track::Remote => &mut remote,
        };
        feed(state, &piece);
        while let Some(segment) = state.vad.front() {
            let start = state.offset + segment.start();
            let end = start + segment.n() as i64;
            let samples = segment.samples().to_vec();
            drop(segment);
            state.vad.pop();

            if samples.is_empty() {
                continue;
            }
            if piece.track == Track::Mic && strong.is_echo(rms(&samples), start, end) {
                continue;
            }
            let offset = start as f64 / SAMPLE_RATE as f64;
            match crate::speech::with_models(|| pass.decode(&samples, offset)) {
                Ok(words) => {
                    let text = words
                        .iter()
                        .map(|word| word.w.as_str())
                        .collect::<Vec<_>>()
                        .join(" ");
                    if !text.trim().is_empty() {
                        strong.emit(
                            "live",
                            json!({
                                "track": piece.track.dir_name(),
                                "s": offset,
                                "e": end as f64 / SAMPLE_RATE as f64,
                                "text": text,
                            }),
                        );
                    }
                }
                Err(error) => strong.emit(
                    "error",
                    json!({ "message": format!("live transcription failed: {error}") }),
                ),
            }
        }
    }
}

/// Aligns a piece with the meeting clock and feeds it to the VAD.
fn feed(state: &mut TrackVad, piece: &LivePiece) {
    let mut samples: &[f32] = &piece.samples;
    let mut start = piece.index;
    if start < state.fed {
        let skip = (state.fed - start) as usize;
        if skip >= samples.len() {
            return;
        }
        samples = &samples[skip..];
        start = state.fed;
    }
    if start > state.fed {
        let gap = start - state.fed;
        if gap > MAX_GAP_SAMPLES {
            state.vad.reset();
            state.offset = start;
        } else {
            feed_zeros(&state.vad, gap);
        }
        state.fed = start;
    }
    feed_samples(&state.vad, samples);
    state.fed += samples.len() as i64;
}

fn feed_samples(vad: &VoiceActivityDetector, samples: &[f32]) {
    for block in samples.chunks(VAD_WINDOW) {
        vad.accept_waveform(block);
    }
}

fn feed_zeros(vad: &VoiceActivityDetector, count: i64) {
    let zeros = vec![0.0f32; VAD_WINDOW];
    let mut left = count;
    while left > 0 {
        let take = left.min(VAD_WINDOW as i64) as usize;
        vad.accept_waveform(&zeros[..take]);
        left -= take as i64;
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|sample| sample * sample).sum();
    (sum / samples.len() as f32).sqrt()
}
