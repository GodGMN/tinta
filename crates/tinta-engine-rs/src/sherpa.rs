//! Loads the manifest's models with sherpa-onnx.
//!
//! `install_models` loads every component once after installing it, so a broken
//! download or a model the CPU cannot run fails inside the install command
//! instead of during the first recording. The models are released afterwards
//! until the transcription and diarization tasks use them.
//!
//! The thread counts and the ONNX Runtime provider file follow the engine design
//! conditions in `docs/fork/benchmark.md`: threads from the CPUs the process
//! actually has, and intra-op spinning off.

use anyhow::{bail, Context, Result};
use sherpa_onnx::{
    FastClusteringConfig, OfflineModelConfig, OfflineRecognizer, OfflineRecognizerConfig,
    OfflineSpeakerDiarization, OfflineSpeakerDiarizationConfig,
    OfflineSpeakerSegmentationModelConfig, OfflineSpeakerSegmentationPyannoteModelConfig,
    OfflineTransducerModelConfig, SileroVadModelConfig, SpeakerEmbeddingExtractorConfig,
    VadModelConfig, VoiceActivityDetector,
};
use std::path::Path;

use crate::manifest::{installed_path, ModelFile, ModelGroup};
use crate::threads::Threads;

const SAMPLE_RATE: i32 = 16_000;
/// The benchmark's VAD settings and the 30 s cap on a piece.
const VAD_MAX_SPEECH_SECONDS: f32 = 30.0;
const VAD_BUFFER_SECONDS: f32 = 60.0;
/// A safe over-segmenting default from the dev sweep.
const DIARIZATION_THRESHOLD: f32 = 0.7;

/// Writes the sherpa-onnx provider config that turns intra-op spinning off and
/// returns the provider string that points at it.
pub fn provider(root: &Path) -> Result<String> {
    std::fs::create_dir_all(root)
        .with_context(|| format!("cannot create the models folder {}", root.display()))?;
    let path = root.join("provider.conf");
    std::fs::write(&path, "SessionConfig.session.intra_op.allow_spinning=0\n")
        .with_context(|| format!("cannot write {}", path.display()))?;
    Ok(format!("cpu:{}", path.display()))
}

/// Loads the component once to prove its files work. Returns after the model is
/// loaded and released.
pub fn load_group(
    group: &ModelGroup,
    root: &Path,
    threads: &Threads,
    provider: &str,
) -> Result<()> {
    match group.component.as_str() {
        "transcription" => load_transcription(group, root, threads, provider),
        "vad" => load_vad(group, root, threads, provider),
        "diarization" => load_diarization(group, root, threads, provider),
        other => bail!("the model manifest names an unknown component: {other}"),
    }
}

fn path_of(root: &Path, file: &ModelFile) -> String {
    installed_path(root, file).to_string_lossy().into_owned()
}

fn load_transcription(
    group: &ModelGroup,
    root: &Path,
    threads: &Threads,
    provider: &str,
) -> Result<()> {
    let encoder = path_of(root, group.file("encoder")?);
    let decoder = path_of(root, group.file("decoder")?);
    let joiner = path_of(root, group.file("joiner")?);
    let tokens = path_of(root, group.file("tokens")?);

    let config = OfflineRecognizerConfig {
        model_config: OfflineModelConfig {
            transducer: OfflineTransducerModelConfig {
                encoder: Some(encoder),
                decoder: Some(decoder),
                joiner: Some(joiner),
            },
            tokens: Some(tokens),
            model_type: Some("nemo_transducer".to_string()),
            provider: Some(provider.to_string()),
            num_threads: threads.asr as i32,
            debug: false,
            ..Default::default()
        },
        ..Default::default()
    };

    let recognizer = OfflineRecognizer::create(&config)
        .context("the transcription model did not load. Install the models again.")?;
    drop(recognizer);
    Ok(())
}

fn load_vad(group: &ModelGroup, root: &Path, threads: &Threads, provider: &str) -> Result<()> {
    let model = path_of(root, group.file("vad")?);
    let config = VadModelConfig {
        silero_vad: SileroVadModelConfig {
            model: Some(model),
            threshold: 0.5,
            min_silence_duration: 0.3,
            min_speech_duration: 0.1,
            window_size: 512,
            max_speech_duration: VAD_MAX_SPEECH_SECONDS,
        },
        sample_rate: SAMPLE_RATE,
        num_threads: threads.asr as i32,
        provider: Some(provider.to_string()),
        debug: false,
        ..Default::default()
    };

    let vad = VoiceActivityDetector::create(&config, VAD_BUFFER_SECONDS)
        .context("the voice activity model did not load. Install the models again.")?;
    drop(vad);
    Ok(())
}

fn load_diarization(
    group: &ModelGroup,
    root: &Path,
    threads: &Threads,
    provider: &str,
) -> Result<()> {
    let segmentation = path_of(root, group.file("segmentation")?);
    let embedding = path_of(root, group.file("embedding")?);

    let config = OfflineSpeakerDiarizationConfig {
        segmentation: OfflineSpeakerSegmentationModelConfig {
            pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
                model: Some(segmentation),
                window_shift_ratio: 0.2,
            },
            num_threads: threads.diarization as i32,
            debug: false,
            provider: Some(provider.to_string()),
        },
        embedding: SpeakerEmbeddingExtractorConfig {
            model: Some(embedding),
            num_threads: threads.diarization as i32,
            debug: false,
            provider: Some(provider.to_string()),
        },
        clustering: FastClusteringConfig {
            num_clusters: -1,
            threshold: DIARIZATION_THRESHOLD,
            compute_confidence: false,
        },
        min_duration_on: 0.3,
        min_duration_off: 0.5,
    };

    let diarizer = OfflineSpeakerDiarization::create(&config)
        .context("the diarization models did not load. Install the models again.")?;
    drop(diarizer);
    Ok(())
}
