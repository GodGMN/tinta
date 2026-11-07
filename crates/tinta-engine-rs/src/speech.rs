//! Model loading and the two final-pass stages: VAD segmentation + ASR, and
//! speaker diarization.
//!
//! Models load on demand and live only for the duration of one `finalize`: the
//! recognizer, the VAD and the diarizer are dropped at the end of the pass. A
//! global lock serializes model use, so two final passes never load the 652 MB
//! recognizer twice; `Speech` in the Swift engine is an actor for the same reason.
//!
//! Diarization is the task #18 recipe: the sherpa-onnx pipeline over-clusters
//! the segments at a cosine-dissimilarity threshold of 0.7, the segments are
//! joined into embedding units, one CAMPPlus embedding is computed per unit, and
//! [`crate::clustering::recluster`] merges the centroids and dissolves tiny
//! clusters by voice. The high-level sherpa-onnx API does not expose the
//! per-segment embeddings it computed while clustering, so the embedding pass is
//! a second pass over the same audio (measured at +14-33 % of the diarization
//! stage in task #18).
//!
//! Threads and the ONNX Runtime provider follow the engine design conditions:
//! `Threads::detect()` and the `SessionConfig.session.intra_op.allow_spinning=0`
//! provider file, applied to the ASR, VAD, diarization and embedding sessions.

use anyhow::{bail, Context, Result};
use sherpa_onnx::{
    FastClusteringConfig, OfflineRecognizer, OfflineRecognizerConfig, OfflineSpeakerDiarization,
    OfflineSpeakerDiarizationConfig, OfflineSpeakerSegmentationModelConfig,
    OfflineSpeakerSegmentationPyannoteModelConfig, OfflineTransducerModelConfig,
    SileroVadModelConfig, SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig,
    VadModelConfig, VoiceActivityDetector,
};
use std::path::PathBuf;
use std::sync::Mutex;

use crate::audio::SAMPLE_RATE;
use crate::clustering::{recluster, Options as ClusterOptions, Unit};
use crate::manifest::{Manifest, ModelGroup};
use crate::segment::{ensure_recognizer_window, group_pieces, trim_overlaps, vad_segments};
use crate::threads::Threads;
use crate::types::{SpeakerSegment, Word};

/// The cosine-dissimilarity cut for the **over-clustering** step. Higher merges
/// more, so 0.7 deliberately splits speakers; [`crate::clustering`] repairs the
/// partition afterwards. Tied to CAMPPlus: another embedding model needs its own
/// value.
pub const DIARIZATION_THRESHOLD: f32 = 0.7;
/// Unit builder thresholds, part of the task #18 recipe: join same-cluster
/// segments no further apart than this, cap a unit at this length, and pad a
/// unit shorter than this before embedding.
const UNIT_JOIN_GAP_SECONDS: f32 = 0.5;
const UNIT_MAX_SECONDS: f32 = 15.0;
const UNIT_MIN_SECONDS: f32 = 0.8;
/// The VAD closes a continuous run at this length; the splitter keeps the pieces
/// at or below 30 s even if a future VAD setting changes.
const VAD_MAX_SPEECH_SECONDS: f32 = 30.0;
const VAD_BUFFER_SECONDS: f32 = 60.0;
/// `Speech.diarize` skips the diarizer below this much audio.
const MIN_DIARIZATION_SECONDS: usize = 2 * SAMPLE_RATE as usize;

/// The diarizer gets at most this much of a track.
///
/// sherpa-onnx's over-clustering still builds an n × n distance matrix over the
/// segment embeddings, so memory grows quadratically with the segment count.
/// Measured on the task #2 AMI test meetings (pyannote segmentation 3.0, window
/// shift 0.2, complete linkage at 0.7): 172-329 segments for 29.5-55.9 minutes,
/// that is **350-423 segments per audio hour** (mean 382). At 4 h that is about
/// 1,530 segments and a 9 MB f32 matrix; the 12 h codec cap would be about 4,600
/// segments and 85 MB by the same density. The pathological upper bound (one
/// segment per 2 s window for each of 3 local speakers, 5,400 per hour) would be
/// 1.9 GB at 4 h, which is why the input is capped there instead of at 12 h.
/// A longer remote track is still fully transcribed; only the speaker labels stop
/// at 4 h. Words after the cap have no diarization segment nearby, so the app
/// labels them with its generic `remote` speaker; the pass also returns a
/// `notes` entry so the app can tell the user that labels stop at 4 h.
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

    /// Non-fatal notes from the last pass, such as speaker labels stopping at
    /// 4 h. They travel as an extra `notes` array in the finalize result; the
    /// app deserializes `FinalResult` without denying unknown fields, so an app
    /// that does not know them ignores them safely. Empty by default.
    fn notes(&self) -> Vec<String> {
        Vec::new()
    }
}

/// The sherpa-onnx implementation of [`FinalPass`].
pub struct SpeechPass {
    manifest: Manifest,
    root: PathBuf,
    threads: Threads,
    recognizer: Option<OfflineRecognizer>,
    notes: Vec<String>,
}

impl SpeechPass {
    pub fn new(manifest: Manifest, root: PathBuf, threads: Threads) -> Self {
        Self {
            manifest,
            root,
            threads,
            recognizer: None,
            notes: Vec::new(),
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

    fn recognizer(&mut self) -> Result<&OfflineRecognizer> {
        if self.recognizer.is_none() {
            self.check_installed()?;
            let group = self.group("transcription")?;
            let (encoder, decoder, joiner, tokens) = (
                self.path(group, "encoder")?,
                self.path(group, "decoder")?,
                self.path(group, "joiner")?,
                self.path(group, "tokens")?,
            );
            let provider = self.provider()?;
            let config = OfflineRecognizerConfig {
                model_config: sherpa_onnx::OfflineModelConfig {
                    transducer: OfflineTransducerModelConfig {
                        encoder: Some(encoder),
                        decoder: Some(decoder),
                        joiner: Some(joiner),
                    },
                    tokens: Some(tokens),
                    model_type: Some("nemo_transducer".to_string()),
                    provider: Some(provider),
                    num_threads: self.threads.asr as i32,
                    debug: false,
                    ..Default::default()
                },
                ..Default::default()
            };
            self.recognizer = Some(
                OfflineRecognizer::create(&config)
                    .context("the transcription model did not load. Install the models again.")?,
            );
        }
        Ok(self.recognizer.as_ref().expect("loaded above"))
    }

    fn vad(&self) -> Result<VoiceActivityDetector> {
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

    /// The CAMPPlus extractor, with the same threads and provider as the
    /// diarizer. It is loaded after the diarizer runs, because the high-level
    /// API does not expose the embeddings it already computed.
    fn embedding_extractor(&self) -> Result<SpeakerEmbeddingExtractor> {
        self.check_installed()?;
        let group = self.group("diarization")?;
        let model = self.path(group, "embedding")?;
        let provider = self.provider()?;
        let config = SpeakerEmbeddingExtractorConfig {
            model: Some(model),
            num_threads: self.threads.diarization as i32,
            debug: false,
            provider: Some(provider),
        };
        SpeakerEmbeddingExtractor::create(&config)
            .context("the speaker embedding model did not load. Install the models again.")
    }

    /// One embedding per unit, in input order. A unit that cannot produce an
    /// embedding (out of range, too short, model failure) becomes a zero vector,
    /// which [`recluster`] excludes from the centroids and sends to the largest
    /// cluster.
    fn unit_embeddings(
        &self,
        samples: &[f32],
        units: &[(f32, f32, usize)],
    ) -> Result<Vec<Vec<f32>>> {
        let extractor = self.embedding_extractor()?;
        let dim = extractor.dim().max(0) as usize;
        let min_samples = (UNIT_MIN_SECONDS * SAMPLE_RATE as f32) as usize;
        let mut out = Vec::with_capacity(units.len());
        for &(start, end, _) in units {
            let (from, to) = padded_unit_range(start, end, samples.len(), min_samples);
            if to <= from {
                out.push(vec![0.0; dim]);
                continue;
            }
            let Some(stream) = extractor.create_stream() else {
                out.push(vec![0.0; dim]);
                continue;
            };
            stream.accept_waveform(SAMPLE_RATE as i32, &samples[from..to]);
            stream.input_finished();
            let embedding = if extractor.is_ready(&stream) {
                extractor.compute(&stream)
            } else {
                None
            };
            out.push(embedding.unwrap_or_else(|| vec![0.0; dim]));
        }
        Ok(out)
    }

    fn decode(&mut self, samples: &[f32], offset: f64) -> Result<Vec<Word>> {
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
        self.notes.clear();
        if samples.len() <= MIN_DIARIZATION_SECONDS {
            return Ok(Vec::new());
        }
        if let Some(note) = truncation_note(samples.len()) {
            crate::protocol::log(&note);
            self.notes.push(note);
        }
        let samples = &samples[..samples
            .len()
            .min(MAX_DIARIZATION_SECONDS * SAMPLE_RATE as usize)];
        let diarizer = self.diarizer(max_speakers)?;
        progress(0.0);
        // Step 1 of the recipe: over-cluster at 0.7 (see `cluster_speakers`).
        let result = diarizer
            .process(samples)
            .context("the diarizer returned no result")?;
        let segments: Vec<(f32, f32, usize)> = result
            .sort_by_start_time()
            .into_iter()
            .map(|segment| (segment.start, segment.end, segment.speaker.max(0) as usize))
            .collect();
        // Step 2: join the over-clustered segments into units and give each one
        // a CAMPPlus embedding. The thresholds are part of the recipe.
        let units = build_units(&segments, UNIT_JOIN_GAP_SECONDS, UNIT_MAX_SECONDS);
        if units.is_empty() {
            progress(1.0);
            return Ok(Vec::new());
        }
        let embeddings = self.unit_embeddings(samples, &units)?;
        let units_with_embeddings: Vec<Unit> = units
            .iter()
            .zip(embeddings)
            .map(|(&(start, end, cluster), embedding)| Unit {
                start,
                end,
                cluster,
                embedding,
            })
            .collect();
        // Step 3: merge centroids and dissolve tiny clusters by voice. The
        // caller's `max_speakers` is a cap, never a forced count.
        let options = ClusterOptions {
            max_speakers,
            ..ClusterOptions::default()
        };
        let outcome = recluster(&units_with_embeddings, &options);
        log_collapse(&outcome);
        let segments = units
            .iter()
            .zip(&outcome.labels)
            .map(|(&(start, end, _), &label)| SpeakerSegment {
                speaker: format!("S{}", label + 1),
                s: start as f64,
                e: end as f64,
            })
            .collect();
        progress(1.0);
        Ok(segments)
    }

    fn notes(&self) -> Vec<String> {
        self.notes.clone()
    }
}

/// Returns the note for a track longer than the diarization cap. Only speaker
/// labels stop there; words are still transcribed and the app labels them with
/// its generic `remote` speaker when no segment is nearby.
fn truncation_note(len: usize) -> Option<String> {
    if len <= MAX_DIARIZATION_SECONDS * SAMPLE_RATE as usize {
        return None;
    }
    Some(format!(
        "speaker labels cover only the first {} hours of a {:.1} hour remote track; \
         later words keep the app's remote label",
        MAX_DIARIZATION_SECONDS / 3600,
        len as f64 / SAMPLE_RATE as f64 / 3600.0
    ))
}

/// Logs when the dissolve step leaves a small fraction of the over-clustered
/// partition, the case where a user-facing merge/split/rename override matters.
/// The engine has no such override yet, so the log is the only signal.
fn log_collapse(outcome: &crate::clustering::Outcome) {
    let collapsed = outcome.clusters_before > 1
        && (outcome.clusters_after == 1
            || (outcome.dissolved >= 3 && outcome.clusters_after * 2 <= outcome.clusters_before));
    if collapsed {
        crate::protocol::log(&format!(
            "diarization collapsed {} over-clustered clusters into {} speakers \
             ({} dissolved, {} units reassigned)",
            outcome.clusters_before, outcome.clusters_after, outcome.dissolved, outcome.reassigned
        ));
    }
}

/// Joins the front end's segments into embedding units: consecutive segments of
/// the same cluster no further apart than `gap` seconds and no longer than
/// `max_len` together. A unit is the smallest thing the post-clustering
/// relabels, so very short fragments never get an embedding of their own.
fn build_units(segments: &[(f32, f32, usize)], gap: f32, max_len: f32) -> Vec<(f32, f32, usize)> {
    let mut out: Vec<(f32, f32, usize)> = Vec::new();
    for &(start, end, cluster) in segments {
        if end <= start {
            continue;
        }
        if let Some(last) = out.last_mut() {
            if last.2 == cluster && start - last.1 <= gap && end.max(last.1) - last.0 <= max_len {
                last.1 = last.1.max(end);
                continue;
            }
        }
        out.push((start, end, cluster));
    }
    out
}

/// Sample range of a unit, padded symmetrically to `min_samples` around very
/// short units: an embedding needs a few hundred ms of voice.
fn padded_unit_range(start: f32, end: f32, len: usize, min_samples: usize) -> (usize, usize) {
    let from = ((start.max(0.0) * SAMPLE_RATE as f32) as usize).min(len);
    let mut to = ((end.max(0.0) * SAMPLE_RATE as f32) as usize).min(len);
    let mut from = from.min(to);
    if to > from && to - from < min_samples {
        let need = min_samples - (to - from);
        from = from.saturating_sub(need / 2);
        to = (to + (need - need / 2)).min(len);
    }
    if to < from {
        to = from;
    }
    (from, to)
}

/// The **over-clustering** step: sherpa-onnx's fast clustering at the given
/// cosine-dissimilarity threshold (`0.7`) with no forced count. Higher values
/// merge more, so 0.7 deliberately splits speakers; [`recluster`] repairs the
/// partition in [`FinalPass::diarize`] afterwards (centroid merge at 0.15 plus
/// dissolve/reassign below 3 % of the speech), which is what turns the 24-60
/// clusters of a meeting into its real speakers.
///
/// `max_speakers` is a **cap**, not a count: the app sends the number of remote
/// participants, and [`crate::clustering::Options::max_speakers`] merges the
/// closest final centroids until the count fits. This function itself keeps
/// `num_clusters = -1`: forcing sherpa-onnx's *exact* count is forbidden by
/// design condition 6 and would split a real speaker when the caller passes a
/// number above the truth. The recipe alone returns the reference count +/-1 on
/// the 90 s `finalize_models` clip; the cap is only an upper bound.
///
/// **0.7 is tuned for CAMPPlus only** (the `diarization` embedding model in the
/// manifest). Another embedding model has a different similarity scale and needs
/// its own over-clustering threshold, merge distance and `min_share`.
pub fn cluster_speakers(threshold: f32, max_speakers: Option<usize>) -> FastClusteringConfig {
    // The cap is applied after `recluster` in `FinalPass::diarize`; this step
    // must never force an exact count.
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

/// Serializes model work across final passes. The 652 MB recognizer is loaded
/// once per pass; two passes run one after the other so they cannot both hold it.
static MODEL_LOCK: Mutex<()> = Mutex::new(());

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

    #[test]
    fn units_join_the_same_cluster_only_inside_the_gap_and_cap() {
        let segments = [
            (0.0, 1.0, 0),
            (1.5, 2.0, 0), // 0.5 s gap: joined
            (2.4, 3.0, 0), // 0.4 s gap: joined
            (4.0, 5.0, 1), // different cluster
            (5.2, 6.0, 0), // same as the first cluster but after cluster 1
            (7.0, 8.0, 0), // 1.0 s gap: not joined
            (9.0, 9.0, 0), // empty: skipped
            (9.0, 9.5, 2),
        ];
        let units = build_units(&segments, 0.5, 15.0);
        assert_eq!(
            units,
            vec![
                (0.0, 3.0, 0),
                (4.0, 5.0, 1),
                (5.2, 6.0, 0),
                (7.0, 8.0, 0),
                (9.0, 9.5, 2),
            ]
        );
    }

    #[test]
    fn a_unit_is_capped_at_fifteen_seconds() {
        // The gap is small enough to join, but 6 + 6 + 6 would pass 15 s.
        let segments = [(0.0, 6.0, 0), (6.1, 12.0, 0), (12.1, 18.0, 0)];
        let units = build_units(&segments, 0.5, 15.0);
        assert_eq!(units, vec![(0.0, 12.0, 0), (12.1, 18.0, 0)]);
    }

    #[test]
    fn short_units_are_padded_symmetrically() {
        let (from, to) = padded_unit_range(10.0, 10.2, 16_000 * 30, 12_800);
        // 0.2 s of voice centred in a 0.8 s window.
        assert_eq!(to - from, 12_800);
        assert_eq!(from, 10 * 16_000 - 4_800);
        assert_eq!(to, 10 * 16_000 + 3_200 + 4_800);
    }

    #[test]
    fn unit_padding_stays_inside_the_track_and_keeps_long_units() {
        let (from, to) = padded_unit_range(0.0, 0.1, 16_000, 12_800);
        // The half that would precede sample 0 is dropped, as in the benchmark.
        assert_eq!(from, 0);
        assert_eq!(to, 7_200);
        // A unit that is already long enough is not padded.
        let (from, to) = padded_unit_range(1.0, 3.0, 16_000 * 30, 12_800);
        assert_eq!((from, to), (16_000, 48_000));
    }

    #[test]
    fn a_track_past_the_diarization_cap_carries_a_note() {
        let cap = MAX_DIARIZATION_SECONDS * SAMPLE_RATE as usize;
        assert!(truncation_note(cap).is_none());
        let note = truncation_note(cap + 1).expect("a note past the cap");
        assert!(note.contains("4 hours"), "{note}");
        assert!(note.contains("remote label"), "{note}");
        assert!(truncation_note(cap * 2).is_some());
    }
}
