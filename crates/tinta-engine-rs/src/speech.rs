//! Model loading and the two final-pass stages: VAD segmentation + ASR, and
//! speaker diarization.
//!
//! Models load on demand and live only as long as a pass uses them: the
//! recognizer, the VAD and the diarizer are dropped at the end of the pass. A
//! global lock serializes model use, so two final passes never load the 652 MB
//! recognizer twice. The recognizer itself is shared through a weak cache, so a
//! live draft pass and a final pass alive at the same time (the previous
//! meeting finalizing while the next one records) hold one copy, not two; the
//! release happens when the last pass that uses it ends.
//!
//! Threads and the ONNX Runtime provider follow the engine design conditions:
//! `Threads::detect()` and the `SessionConfig.session.intra_op.allow_spinning=0`
//! provider file, applied to the ASR, VAD and embedding sessions.

use anyhow::{bail, Context, Result};
use sherpa_onnx::{
    FastClusteringConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineSpeakerDiarization,
    OfflineSpeakerDiarizationConfig, OfflineSpeakerSegmentationModelConfig,
    OfflineSpeakerSegmentationPyannoteModelConfig, OfflineTransducerModelConfig,
    SileroVadModelConfig, SpeakerEmbeddingExtractorConfig, VadModelConfig, VoiceActivityDetector,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, TryLockError, Weak};

use crate::audio::SAMPLE_RATE;
use crate::manifest::{Manifest, ModelGroup};
use crate::segment::{ensure_recognizer_window, group_pieces, trim_overlaps, vad_segments};
use crate::threads::Threads;
use crate::types::{SpeakerSegment, Word};

/// The cosine-dissimilarity cut from the dev sweep. Higher merges more.
pub const DIARIZATION_THRESHOLD: f32 = 0.7;
/// The VAD closes a continuous run at this length; the splitter keeps the pieces
/// at or below 30 s even if a future VAD setting changes.
const VAD_MAX_SPEECH_SECONDS: f32 = 30.0;
const VAD_BUFFER_SECONDS: f32 = 60.0;
/// `Speech.diarize` skips the diarizer below this much audio.
const MIN_DIARIZATION_SECONDS: usize = 2 * SAMPLE_RATE as usize;

/// The diarizer gets at most this much of a track.
///
/// sherpa-onnx's complete-linkage clustering builds an n × n distance matrix over
/// the segment embeddings, so memory grows quadratically with the segment count.
/// Measured on the task #2 AMI test meetings (pyannote segmentation 3.0, window
/// shift 0.2, complete linkage at 0.7): 172-329 segments for 29.5-55.9 minutes,
/// that is **350-423 segments per audio hour** (mean 382). At 4 h that is about
/// 1,530 segments and a 9 MB f32 matrix; the 12 h codec cap would be about 4,600
/// segments and 85 MB by the same density. The pathological upper bound (one
/// segment per 2 s window for each of 3 local speakers, 5,400 per hour) would be
/// 1.9 GB at 4 h, which is why the input is capped there instead of at 12 h.
/// The high-level sherpa-onnx API only exposes the segments after clustering, so
/// the guard has to be on the audio length. A longer remote track is still fully
/// transcribed; only the speaker labels stop at 4 h. Task #18 (PR #9) replaces
/// `cluster_speakers` with over-cluster + centroid merge + embedding
/// reassignment and removes this cap.
const MAX_DIARIZATION_SECONDS: usize = 4 * 60 * 60;

/// One final pass. `progress` receives the fraction after each unit of work, in
/// the order the Swift engine emits it.
pub trait FinalPass {
    fn transcribe(&mut self, samples: &[f32], progress: &mut dyn FnMut(f64)) -> Result<Vec<Word>>;

    fn diarize(
        &mut self,
        samples: &[f32],
        max_speakers: Option<usize>,
        progress: &mut dyn FnMut(f64),
    ) -> Result<Vec<SpeakerSegment>>;
}

/// The sherpa-onnx implementation of [`FinalPass`].
pub struct SpeechPass {
    manifest: Manifest,
    root: PathBuf,
    threads: Threads,
    /// Kept for the life of the pass: dropping it after every utterance would
    /// free and reload the 652 MB recognizer. It may be the same instance as
    /// another pass's, through [`RECOGNIZER`].
    recognizer: Option<Arc<OfflineRecognizer>>,
}

impl SpeechPass {
    pub fn new(manifest: Manifest, root: PathBuf, threads: Threads) -> Self {
        Self {
            manifest,
            root,
            threads,
            recognizer: None,
        }
    }

    fn provider(&self) -> Result<String> {
        crate::sherpa::provider(&self.root)
    }

    fn group(&self, component: &str) -> Result<&ModelGroup> {
        self.manifest.group(component)
    }

    fn path(&self, group: &ModelGroup, role: &str) -> Result<String> {
        Ok(self
            .root
            .join(&group.file(role)?.path)
            .to_string_lossy()
            .into_owned())
    }

    pub fn check_installed(&self) -> Result<()> {
        if !self.manifest.all_files_exist(&self.root) {
            bail!("models are not installed");
        }
        Ok(())
    }

    /// The recognizer, loaded once per models directory and provider.
    ///
    /// A live draft pass and a final pass can be alive at the same time (the
    /// previous meeting finalizes while the next one records). Both get the same
    /// instance from the weak cache, so the 652 MB Parakeet loads once. The
    /// cache holds a `Weak`, so when the last pass drops its `Arc` the models
    /// are released instead of staying in RSS for the life of the engine.
    ///
    /// Callers hold [`MODEL_LOCK`], so only one thread creates or decodes at a
    /// time. The thread count is whatever the first loader asked for; the live
    /// pass asks for at most two, so a final pass that reuses its instance runs
    /// at two threads. That is slower than the final pass's own budget but
    /// never loads a second copy.
    fn recognizer(&mut self) -> Result<Arc<OfflineRecognizer>> {
        if let Some(recognizer) = &self.recognizer {
            return Ok(recognizer.clone());
        }
        self.check_installed()?;
        let key = RecognizerKey {
            root: self.root.clone(),
            provider: self.provider()?,
        };
        if let Some(recognizer) = cached_recognizer(&key) {
            self.recognizer = Some(recognizer.clone());
            return Ok(recognizer);
        }
        let group = self.group("transcription")?;
        let (encoder, decoder, joiner, tokens) = (
            self.path(group, "encoder")?,
            self.path(group, "decoder")?,
            self.path(group, "joiner")?,
            self.path(group, "tokens")?,
        );
        let config = OfflineRecognizerConfig {
            model_config: sherpa_onnx::OfflineModelConfig {
                transducer: OfflineTransducerModelConfig {
                    encoder: Some(encoder),
                    decoder: Some(decoder),
                    joiner: Some(joiner),
                },
                tokens: Some(tokens),
                model_type: Some("nemo_transducer".to_string()),
                provider: Some(key.provider.clone()),
                num_threads: self.threads.asr as i32,
                debug: false,
                ..Default::default()
            },
            ..Default::default()
        };
        let recognizer = Arc::new(
            OfflineRecognizer::create(&config)
                .context("the transcription model did not load. Install the models again.")?,
        );
        cache_recognizer(&key, &recognizer);
        self.recognizer = Some(recognizer.clone());
        Ok(recognizer)
    }

    pub(crate) fn vad(&self) -> Result<VoiceActivityDetector> {
        self.check_installed()?;
        let group = self.group("vad")?;
        let model = self.path(group, "vad")?;
        let provider = self.provider()?;
        let config = VadModelConfig {
            silero_vad: SileroVadModelConfig {
                model: Some(model),
                // The benchmark's settings, also used for the quality numbers.
                threshold: 0.5,
                min_silence_duration: 0.3,
                min_speech_duration: 0.1,
                window_size: 512,
                max_speech_duration: VAD_MAX_SPEECH_SECONDS,
            },
            sample_rate: SAMPLE_RATE as i32,
            num_threads: self.threads.asr as i32,
            provider: Some(provider),
            debug: false,
            ..Default::default()
        };
        VoiceActivityDetector::create(&config, VAD_BUFFER_SECONDS)
            .context("the voice activity model did not load. Install the models again.")
    }

    fn diarizer(&self, max_speakers: Option<usize>) -> Result<OfflineSpeakerDiarization> {
        self.check_installed()?;
        let group = self.group("diarization")?;
        let segmentation = self.path(group, "segmentation")?;
        let embedding = self.path(group, "embedding")?;
        let provider = self.provider()?;
        let config = OfflineSpeakerDiarizationConfig {
            segmentation: OfflineSpeakerSegmentationModelConfig {
                pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                    model: Some(segmentation),
                    window_shift_ratio: 0.2,
                },
                num_threads: self.threads.diarization as i32,
                debug: false,
                provider: Some(provider.clone()),
            },
            embedding: SpeakerEmbeddingExtractorConfig {
                model: Some(embedding),
                num_threads: self.threads.diarization as i32,
                debug: false,
                provider: Some(provider),
            },
            clustering: cluster_speakers(DIARIZATION_THRESHOLD, max_speakers),
            min_duration_on: 0.3,
            min_duration_off: 0.5,
        };
        OfflineSpeakerDiarization::create(&config)
            .context("the diarization models did not load. Install the models again.")
    }

    /// Decodes one finished speech segment at `offset` seconds.
    pub(crate) fn decode(&mut self, samples: &[f32], offset: f64) -> Result<Vec<Word>> {
        ensure_recognizer_window(samples.len())?;
        let recognizer = self.recognizer()?;
        let stream = recognizer.create_stream();
        stream.accept_waveform(SAMPLE_RATE as i32, samples);
        recognizer.decode(&stream);
        let result = stream
            .get_result()
            .context("the transcription model returned no result")?;
        Ok(tokens_to_words(
            &result.tokens,
            result.timestamps.as_deref(),
            result.durations.as_deref(),
            offset,
        ))
    }
}

impl FinalPass for SpeechPass {
    fn transcribe(&mut self, samples: &[f32], progress: &mut dyn FnMut(f64)) -> Result<Vec<Word>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }
        self.check_installed()?;
        let vad = self.vad()?;
        let segments = vad_segments(&vad, samples);
        let pieces = group_pieces(samples, &segments);
        if pieces.is_empty() {
            return Ok(Vec::new());
        }
        let total = pieces.len();
        let mut words = Vec::with_capacity(total);
        for (index, (start, end)) in pieces.iter().enumerate() {
            let offset = *start as f64 / SAMPLE_RATE as f64;
            words.push(self.decode(&samples[*start..*end], offset)?);
            progress((index + 1) as f64 / total as f64);
        }
        Ok(trim_overlaps(&pieces, &mut words))
    }

    fn diarize(
        &mut self,
        samples: &[f32],
        max_speakers: Option<usize>,
        progress: &mut dyn FnMut(f64),
    ) -> Result<Vec<SpeakerSegment>> {
        if samples.len() <= MIN_DIARIZATION_SECONDS {
            return Ok(Vec::new());
        }
        let samples = if samples.len() > MAX_DIARIZATION_SECONDS * SAMPLE_RATE as usize {
            crate::protocol::log(&format!(
                "diarizing the first {MAX_DIARIZATION_SECONDS} s of a {} s remote track",
                samples.len() / SAMPLE_RATE as usize
            ));
            &samples[..MAX_DIARIZATION_SECONDS * SAMPLE_RATE as usize]
        } else {
            samples
        };
        let diarizer = self.diarizer(max_speakers)?;
        progress(0.0);
        let result = diarizer
            .process(samples)
            .context("the diarizer returned no result")?;
        let segments = result
            .sort_by_start_time()
            .into_iter()
            .map(|segment| SpeakerSegment {
                speaker: format!("S{}", segment.speaker + 1),
                s: segment.start as f64,
                e: segment.end as f64,
            })
            .collect();
        progress(1.0);
        Ok(segments)
    }
}

/// The clustering step, kept in one function so a later task can replace it with
/// over-cluster + centroid merge + reassignment by embedding without touching the
/// diarizer configuration around it. Until then this is sherpa-onnx's complete
/// linkage with the given cosine-dissimilarity threshold (`0.7` over-clusters
/// rather than merging speakers, per the benchmark).
///
/// `max_speakers` is accepted for protocol parity but **ignored**: Swift uses it
/// as a *cap* (`OfflineDiarizerConfig.withSpeakers(min: nil, max:)`), while
/// sherpa-onnx can only force an *exact* cluster count, which design condition 6
/// forbids. The measured effect is visible: the two-speaker test clip comes back
/// with four speakers. Task #18 (PR #9, `spike/18/diarization`) replaces the
/// internals of this function with over-cluster + centroid merge + reassignment
/// by embedding, where a cap fits naturally; the signature stays the same.
pub fn cluster_speakers(threshold: f32, max_speakers: Option<usize>) -> FastClusteringConfig {
    let _ = max_speakers;
    FastClusteringConfig {
        num_clusters: -1,
        threshold,
        compute_confidence: false,
    }
}

/// Turns sentencepiece tokens into words with start and end times. A word spans
/// its tokens; a token with the `▁` marker (or a leading space) starts a new one.
fn tokens_to_words(
    tokens: &[String],
    timestamps: Option<&[f32]>,
    durations: Option<&[f32]>,
    offset: f64,
) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut current: Option<Word> = None;
    for (index, token) in tokens.iter().enumerate() {
        if token.is_empty() {
            continue;
        }
        let start = timestamps
            .and_then(|values| values.get(index))
            .copied()
            .unwrap_or(0.0) as f64
            + offset;
        let end = start
            + durations
                .and_then(|values| values.get(index))
                .copied()
                .unwrap_or(0.0) as f64;
        let starts_word = token.starts_with('▁') || token.starts_with(' ');
        let text = token.trim_start_matches('▁').trim_start();
        if starts_word || current.is_none() {
            if let Some(word) = current.take() {
                words.push(word);
            }
            current = Some(Word {
                w: text.to_string(),
                s: start,
                e: end,
            });
        } else if let Some(word) = current.as_mut() {
            word.w.push_str(text);
            word.e = end;
        }
    }
    if let Some(word) = current {
        words.push(word);
    }
    words.retain(|word| !word.w.trim().is_empty());
    words
}

/// Serializes model work across final passes. The recognizer is shared through
/// [`RECOGNIZER`], so a live pass and a final pass alive at the same time hold
/// one copy; the lock also means only one of them decodes at a time.
static MODEL_LOCK: Mutex<()> = Mutex::new(());

/// The recognizer currently loaded, shared by every pass that needs it.
///
/// The entry is keyed by the models directory and the ONNX provider, not by the
/// thread count: a second copy would double the 652 MB for a cosmetic
/// difference. The weak reference means the models are released when the last
/// pass using them ends.
static RECOGNIZER: Mutex<Option<CachedRecognizer>> = Mutex::new(None);

#[derive(Clone, PartialEq, Eq)]
struct RecognizerKey {
    root: PathBuf,
    provider: String,
}

struct CachedRecognizer {
    key: RecognizerKey,
    recognizer: Weak<OfflineRecognizer>,
}

fn cached_recognizer(key: &RecognizerKey) -> Option<Arc<OfflineRecognizer>> {
    let cache = RECOGNIZER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = cache.as_ref()?;
    if &entry.key != key {
        return None;
    }
    entry.recognizer.upgrade()
}

fn cache_recognizer(key: &RecognizerKey, recognizer: &Arc<OfflineRecognizer>) {
    let mut cache = RECOGNIZER
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *cache = Some(CachedRecognizer {
        key: key.clone(),
        recognizer: Arc::downgrade(recognizer),
    });
}

/// Runs `work` while holding the model lock.
///
/// A panic inside `work` unwinds through the guard and poisons the mutex. The
/// guarded data is just `()`, and the models live in the `SpeechPass` that was
/// unwinding, so recovering the inner value is safe: the next finalize loads its
/// own models and runs instead of failing with "the model lock is poisoned"
/// until the engine restarts.
pub fn with_models<T>(work: impl FnOnce() -> Result<T>) -> Result<T> {
    let _guard = MODEL_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    work()
}

/// Runs `work` while holding the model lock, but returns `Ok(None)` instead of
/// waiting when `keep_going` turns false.
///
/// The live draft worker uses this: during the final pass of the previous
/// meeting the lock is held for minutes, and `stop` must not wait on it. One
/// utterance is skipped instead, and the final pass transcribes it anyway.
pub fn with_models_unless<T>(
    keep_going: impl Fn() -> bool,
    work: impl FnOnce() -> Result<T>,
) -> Result<Option<T>> {
    loop {
        match MODEL_LOCK.try_lock() {
            Ok(_guard) => return work().map(Some),
            Err(TryLockError::Poisoned(poisoned)) => {
                let _guard = poisoned.into_inner();
                return work().map(Some);
            }
            Err(TryLockError::WouldBlock) => {
                if !keep_going() {
                    return Ok(None);
                }
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_words_span_their_tokens() {
        let tokens = vec!["▁Hello".to_string(), "▁world".to_string(), "!".to_string()];
        let timestamps = [0.0f32, 0.4, 0.8];
        let durations = [0.4f32, 0.4, 0.1];
        let words = tokens_to_words(&tokens, Some(&timestamps), Some(&durations), 2.0);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].w, "Hello");
        assert!((words[0].s - 2.0).abs() < 1e-6);
        assert!((words[0].e - 2.4).abs() < 1e-6);
        assert_eq!(words[1].w, "world!");
        assert!((words[1].s - 2.4).abs() < 1e-6);
        assert!((words[1].e - 2.9).abs() < 1e-6);
    }

    #[test]
    fn a_leading_space_also_starts_a_word() {
        let tokens = vec![" Hello".to_string(), " there".to_string()];
        let words = tokens_to_words(&tokens, None, None, 0.0);
        assert_eq!(
            words.iter().map(|word| word.w.as_str()).collect::<Vec<_>>(),
            vec!["Hello", "there"]
        );
    }

    #[test]
    fn a_panic_does_not_poison_the_model_lock() {
        let panicked = std::panic::catch_unwind(|| -> Result<()> {
            with_models(|| -> Result<()> { panic!("a model callback panicked") })
        });
        assert!(panicked.is_err());
        // The next pass still takes the lock and runs.
        assert_eq!(with_models(|| Ok(7)).unwrap(), 7);
    }

    #[test]
    fn the_live_pass_gives_up_instead_of_waiting_on_a_final_pass() {
        // Another thread holds the lock, as a final pass does for minutes.
        let (ready, waiting) = std::sync::mpsc::channel();
        let (release, held) = std::sync::mpsc::channel();
        let holder = std::thread::spawn(move || {
            // Another test may have poisoned the lock with a deliberate panic.
            let _guard = MODEL_LOCK
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            ready.send(()).unwrap();
            let _ = held.recv();
        });
        waiting.recv().unwrap();
        let called = std::sync::atomic::AtomicBool::new(false);
        let result = with_models_unless(
            || false,
            || -> Result<()> {
                called.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok(())
            },
        );
        assert!(matches!(result, Ok(None)), "the live pass gave up");
        assert!(!called.load(std::sync::atomic::Ordering::SeqCst));
        release.send(()).unwrap();
        holder.join().unwrap();
        // With the lock free the same call runs.
        assert_eq!(with_models_unless(|| true, || Ok(5)).unwrap(), Some(5));
    }

    #[test]
    fn clustering_keeps_the_threshold_and_never_forces_a_count() {
        for max_speakers in [None, Some(0), Some(3)] {
            let config = cluster_speakers(0.7, max_speakers);
            assert_eq!(config.threshold, 0.7);
            // sherpa-onnx can only force an exact count; the design conditions
            // forbid that, so the automatic count stays.
            assert_eq!(config.num_clusters, -1);
            assert!(!config.compute_confidence);
        }
    }
}
