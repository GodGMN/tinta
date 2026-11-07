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
use crate::segment::{
    ensure_recognizer_window, group_pieces_with, merge_hole_words, speech_gaps, trim_overlaps,
    vad_segments, Settings,
};
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
/// The VAD buffer must hold a whole segment; sherpa-onnx requires it to be
/// larger than `max_speech_duration` (60 s by default).
const VAD_BUFFER_SECONDS: f32 = 120.0;
/// A decode whose words still leave a hole this long inside VAD speech is
/// treated as failed and retried. The int8 TDT decoder occasionally emits the
/// first word and then jumps its token timeline to the end of the window
/// (measured in task #21), dropping several seconds of speech.
const RETRY_HOLE_SECONDS: f32 = 2.5;
/// How far the start of a window moves for each retry. A change of a few hundred
/// milliseconds leaves the failing feature alignment; a later start is only
/// allowed while the window still begins before its first speech, and an earlier
/// one is extra pre-roll. Both keep the piece covering its speech.
const RETRY_OFFSETS_SECONDS: [f32; 4] = [0.4, -0.4, 0.9, -0.9];
/// At most this many passes of hole detection and re-decode.
const GAP_REDECODE_ROUNDS: usize = 2;
/// The hole re-decode may hand the recognizer at most this many times the first
/// pass's audio, retries included, largest holes first. Without it a VAD false
/// positive over music or typing pays the primary decode plus four retries
/// inside round 1. A stricter 25 % of the track was measured to skip 40 % of the
/// planned holes on EN2002a and cost 1.8 WER points on ES2004b and 3.5 on
/// EN2002a, so the cap sits at twice the first pass: the measured meetings spend
/// 61 % (ES2004b) and 148 % (EN2002a) of the first pass's audio, and a
/// pathological track still stops at 3x the first pass's compute.
const REDECODE_BUDGET_SHARE: f64 = 2.0;
/// The first pass fills this share of the transcription progress; the rest is
/// the hole re-decode, so the bar keeps moving instead of sitting at 100 %
/// while the extra pass runs (measured at +31 % to +137 % of the first pass).
const FIRST_PASS_PROGRESS_SHARE: f64 = 0.8;
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
    settings: Settings,
    recognizer: Option<OfflineRecognizer>,
    notes: Vec<String>,
}

impl SpeechPass {
    pub fn new(manifest: Manifest, root: PathBuf, threads: Threads) -> Self {
        Self::with_settings(manifest, root, threads, Settings::default())
    }

    /// [`SpeechPass::new`] with explicit segmentation settings, used by the
    /// segmentation harness in `scripts/fork/segmentation`.
    pub fn with_settings(
        manifest: Manifest,
        root: PathBuf,
        threads: Threads,
        settings: Settings,
    ) -> Self {
        Self {
            manifest,
            root,
            threads,
            settings,
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
        let config = vad_config(model, self.threads.asr, self.provider()?, &self.settings);
        VoiceActivityDetector::create(&config, VAD_BUFFER_SECONDS)
            .context("the voice activity model did not load. Install the models again.")
    }

    fn diarizer(&self) -> Result<OfflineSpeakerDiarization> {
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
            clustering: cluster_speakers(DIARIZATION_THRESHOLD),
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
            // The spike did not call input_finished(); it changed no labels on
            // ES2004b. The golden tests cover reclustering only, not this step.
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

    /// Decodes one window, retrying from a different start while the result
    /// still leaves a hole of [`RETRY_HOLE_SECONDS`] inside the VAD speech the
    /// window covers. The candidate with the least un-covered speech wins, so a
    /// retry can never make the transcript worse.
    ///
    /// `speech` is the VAD speech inside the window, so silence in the pre- and
    /// post-roll is never mistaken for a dropped stretch. `max_retries` caps the
    /// extra decodes so the re-decode budget holds, and `spent` receives the
    /// samples handed to the recognizer, retries included.
    fn decode_with_retry(
        &mut self,
        samples: &[f32],
        start: usize,
        end: usize,
        speech: &[(usize, usize)],
        max_retries: usize,
        spent: &mut usize,
    ) -> Result<Vec<Word>> {
        let mut best = self.decode(&samples[start..end], start as f64 / SAMPLE_RATE as f64)?;
        *spent += end - start;
        for offset in RETRY_OFFSETS_SECONDS.into_iter().take(max_retries) {
            if speech_gaps(speech, &best, RETRY_HOLE_SECONDS).is_empty() {
                break;
            }
            let from = if offset > 0.0 {
                let speech_start = speech.first().map(|&(a, _)| a).unwrap_or(start);
                (start + seconds(offset)).min(speech_start)
            } else {
                start.saturating_sub(seconds(-offset))
            };
            if from == start || from >= end {
                continue;
            }
            let candidate = self.decode(&samples[from..end], from as f64 / SAMPLE_RATE as f64)?;
            *spent += end - from;
            if missing_seconds(speech, &candidate) < missing_seconds(speech, &best) {
                best = candidate;
            }
        }
        Ok(best)
    }

    /// Decodes the hypothesis holes again: a stretch at least
    /// `gap_redecode_seconds` long inside a VAD speech region with no word from
    /// the previous pass. A hole is planned like any other speech span, so it
    /// starts at a quiet point and is split at the piece length if it is long.
    /// Only words whose midpoint falls inside the hole are kept, and a word that
    /// repeats a hole-edge word is dropped ([`merge_hole_words`]), so the words
    /// at its edges stay single. Up to [`GAP_REDECODE_ROUNDS`] passes run,
    /// because a re-decode can itself drop a stretch; a round that adds nothing
    /// stops.
    ///
    /// The audio handed to the recognizer, retries included, is capped at twice
    /// the first pass's audio ([`REDECODE_BUDGET_SHARE`]), largest holes first;
    /// holes that do not fit stay holes. Progress continues from `first_share`.
    fn redecode_gaps(
        &mut self,
        samples: &[f32],
        segments: &[(usize, usize)],
        mut words: Vec<Word>,
        progress: &mut dyn FnMut(f64),
        first_share: f64,
        first_pass_audio: usize,
    ) -> Result<Vec<Word>> {
        let budget = (first_pass_audio as f64 * REDECODE_BUDGET_SHARE) as usize;
        let mut spent = 0usize;
        let mut planned = 0usize;
        let mut holes_decoded = 0usize;
        let mut last_fraction = first_share;
        // Holes that produced no word at all are not retried in a later round:
        // re-decoding the same window is deterministic, and a VAD false positive
        // would otherwise pay two rounds of retries for silence.
        let mut fruitless: Vec<(usize, usize)> = Vec::new();
        for _ in 0..GAP_REDECODE_ROUNDS {
            let holes = speech_gaps(segments, &words, self.settings.gap_redecode_seconds);
            let holes: Vec<(usize, usize)> = holes
                .into_iter()
                .filter(|hole| !fruitless.contains(hole))
                .collect();
            if holes.is_empty() {
                break;
            }
            let before = words.len();
            // Largest holes first: under the cap they carry the most missing
            // speech per recognizer call.
            for (hole_start, hole_end) in largest_first(holes) {
                let pieces = group_pieces_with(&self.settings, samples, &[(hole_start, hole_end)]);
                let cost: usize = pieces.iter().map(|(start, end)| end - start).sum();
                let remaining = budget.saturating_sub(spent);
                if cost > remaining || cost == 0 {
                    continue;
                }
                // The primary decode fits; spend what is left over on retries,
                // so the cap covers every sample the re-decode hands to the
                // recognizer rather than only the first attempt.
                let retries = ((remaining - cost) / cost).min(RETRY_OFFSETS_SECONDS.len());
                planned += cost;
                holes_decoded += 1;
                let mut decoded = Vec::with_capacity(pieces.len());
                for (start, end) in &pieces {
                    decoded.push(self.decode_with_retry(
                        samples,
                        *start,
                        *end,
                        &[(hole_start, hole_end)],
                        retries,
                        &mut spent,
                    )?);
                }
                let added = merge_hole_words(
                    &mut words,
                    trim_overlaps(&pieces, &mut decoded),
                    (hole_start, hole_end),
                );
                if added == 0 {
                    fruitless.push((hole_start, hole_end));
                }
                // Retries can decode more than planned; the fraction is clamped
                // and still never decreases. The caller emits 1.0 at the end.
                let fraction =
                    first_share + (1.0 - first_share) * (spent as f64 / planned as f64).min(1.0);
                last_fraction = last_fraction.max(fraction.min(0.999));
                progress(last_fraction);
            }
            words.sort_by(|a, b| a.s.partial_cmp(&b.s).unwrap_or(std::cmp::Ordering::Equal));
            if words.len() == before {
                break;
            }
        }
        if planned > 0 {
            let rate = SAMPLE_RATE as f64;
            crate::protocol::log(&format!(
                "hole re-decode: {holes_decoded} holes, {:.1} s planned, {:.1} s decoded with retries, cap {:.1} s",
                planned as f64 / rate,
                spent as f64 / rate,
                budget as f64 / rate
            ));
        }
        Ok(words)
    }
}

/// The holes longest first, so the capped re-decode spends its budget where the
/// most speech is missing.
fn largest_first(mut holes: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    holes.sort_by_key(|&(start, end)| std::cmp::Reverse(end - start));
    holes
}

impl SpeechPass {
    /// Decodes an explicit piece plan with the same first pass and hole
    /// re-decode as [`FinalPass::transcribe`]. `transcribe` builds the plan with
    /// `group_pieces_with`; the task #23 harness passes a plan with speaker
    /// splits so both run exactly the same decode path.
    pub fn transcribe_pieces(
        &mut self,
        samples: &[f32],
        segments: &[(usize, usize)],
        pieces: Vec<(usize, usize)>,
        progress: &mut dyn FnMut(f64),
    ) -> Result<Vec<Word>> {
        self.check_installed()?;
        if pieces.is_empty() {
            return Ok(Vec::new());
        }
        let total = pieces.len();
        let total_audio: usize = pieces.iter().map(|(start, end)| end - start).sum();
        let total_audio = total_audio.max(1);
        let mut decoded_audio = 0usize;
        let mut last_fraction = 0.0f64;
        let mut words = Vec::with_capacity(total);
        for (start, end) in pieces.iter() {
            let offset = *start as f64 / SAMPLE_RATE as f64;
            words.push(self.decode(&samples[*start..*end], offset)?);
            decoded_audio += end - start;
            // The first pass fills a fixed share of the transcription stage, so
            // the hole re-decode below keeps moving the same bar instead of
            // starting after a 100 % event.
            let fraction = decoded_audio as f64 / total_audio as f64 * FIRST_PASS_PROGRESS_SHARE;
            last_fraction = last_fraction.max(fraction);
            progress(last_fraction);
        }
        let mut words = trim_overlaps(&pieces, &mut words);
        if self.settings.gap_redecode_seconds > 0.0 {
            words = self.redecode_gaps(
                samples,
                segments,
                words,
                progress,
                last_fraction,
                total_audio,
            )?;
        }
        progress(1.0);
        Ok(words)
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
        drop(vad);
        let pieces = group_pieces_with(&self.settings, samples, &segments);
        self.transcribe_pieces(samples, &segments, pieces, progress)
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
        let diarizer = self.diarizer()?;
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

/// Seconds as samples, for the retry offsets.
fn seconds(value: f32) -> usize {
    (value.max(0.0) * SAMPLE_RATE as f32) as usize
}

/// Seconds of `speech` not covered by any word span, the score a retry improves.
fn missing_seconds(speech: &[(usize, usize)], words: &[Word]) -> f64 {
    speech_gaps(speech, words, 0.0)
        .iter()
        .map(|&(a, b)| (b - a) as f64 / SAMPLE_RATE as f64)
        .sum()
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
/// The caller's `max_speakers` is a **cap**, not a count, and is applied after
/// `recluster` ([`crate::clustering::Options::max_speakers`] merges the closest
/// final centroids until the count fits). It is deliberately not a parameter
/// here: this function keeps `num_clusters = -1`: forcing sherpa-onnx's *exact* count is forbidden by
/// design condition 6 and would split a real speaker when the caller passes a
/// number above the truth. The recipe alone returns the reference count +/-1 on
/// the 90 s `finalize_models` clip; the cap is only an upper bound.
///
/// **0.7 is tuned for CAMPPlus only** (the `diarization` embedding model in the
/// manifest). Another embedding model has a different similarity scale and needs
/// its own over-clustering threshold, merge distance and `min_share`.
pub fn cluster_speakers(threshold: f32) -> FastClusteringConfig {
    FastClusteringConfig {
        num_clusters: -1,
        threshold,
        compute_confidence: false,
    }
}

/// Turns sentencepiece tokens into words with start and end times. A word spans
/// its tokens; a token with the `▁` marker (or a leading space) starts a new one.
/// Public so the segmentation harness in `scripts/fork/segmentation` builds the
/// same words from a bare recognizer call.
pub fn tokens_to_words(
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

/// The Silero VAD settings the engine uses. Public so the segmentation harness
/// in `scripts/fork/segmentation` measures the same detector as `finalize`.
///
/// The threshold is not a tuning knob: task #19 measured 0.2 vs 0.5 as no
/// difference. `max_speech_duration` comes from [`Settings`] and sits well above
/// the piece length, so the VAD itself never cuts in the middle of speech.
pub fn vad_config(
    model: String,
    threads: usize,
    provider: String,
    settings: &Settings,
) -> VadModelConfig {
    VadModelConfig {
        silero_vad: SileroVadModelConfig {
            model: Some(model),
            threshold: 0.5,
            min_silence_duration: 0.3,
            min_speech_duration: 0.1,
            window_size: 512,
            max_speech_duration: settings.vad_max_speech_seconds,
        },
        sample_rate: SAMPLE_RATE as i32,
        num_threads: threads as i32,
        provider: Some(provider),
        debug: false,
        ..Default::default()
    }
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
    fn missing_seconds_counts_speech_the_words_do_not_cover() {
        let speech = [(10 * SAMPLE_RATE as usize, 20 * SAMPLE_RATE as usize)];
        let word = |s: f64, e: f64| Word {
            w: "word".to_string(),
            s,
            e,
        };
        // Two words cover 10-11 s and 12-13 s, so 8 s are missing.
        let words = vec![word(10.0, 11.0), word(12.0, 13.0)];
        assert!((missing_seconds(&speech, &words) - 8.0).abs() < 1e-3);
        // No word at all: the whole region is missing.
        assert!((missing_seconds(&speech, &[]) - 10.0).abs() < 1e-3);
    }

    #[test]
    fn the_redecode_budget_spends_on_the_largest_holes_first() {
        let holes = vec![(0, 3_000), (10_000, 30_000), (40_000, 50_000)];
        assert_eq!(
            largest_first(holes),
            vec![(10_000, 30_000), (40_000, 50_000), (0, 3_000)]
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
        let config = cluster_speakers(0.7);
        assert_eq!(config.threshold, 0.7);
        // sherpa-onnx can only force an exact count; the design conditions
        // forbid that, so the automatic count stays.
        assert_eq!(config.num_clusters, -1);
        assert!(!config.compute_confidence);
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
