//! Decodes one committed audio clip through the engine's own first pass
//! (`SpeechPass::transcribe_pieces`) and prints the words as JSON.
//!
//! The ASR requant portability job (`.github/workflows/ci.yml`) uses this to
//! turn a model file into a word count and a text on every runner. The clip is
//! decoded as a single explicit piece, with the hole re-decode disabled, so the
//! result is one recognizer call and does not depend on the VAD or on the
//! piece planner. That is exactly the failing-window check from task #23.
//!
//! ```text
//! cargo run -p tinta-engine-rs --example decode-clip -- \
//!   --wav crates/tinta-engine-rs/tests/fixtures/es2004b-235.4-248.8.wav \
//!   --models-root /path/to/requant-root --threads 2
//! ```
//!
//! Only the `transcription` group of the embedded manifest is checked, so the
//! models root needs the encoder, decoder, joiner and tokens; the VAD and
//! diarization files are not loaded.

use anyhow::{bail, Context, Result};
use serde_json::json;
use std::path::PathBuf;
use tinta_engine_rs::audio::SAMPLE_RATE;
use tinta_engine_rs::manifest::Manifest;
use tinta_engine_rs::segment::Settings;
use tinta_engine_rs::speech::SpeechPass;
use tinta_engine_rs::threads::Threads;

fn main() -> Result<()> {
    let mut wav = None;
    let mut models_root = None;
    let mut threads = 2usize;
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value =
            || -> Result<String> { args.next().with_context(|| format!("{flag} needs a value")) };
        match flag.as_str() {
            "--wav" => wav = Some(value()?),
            "--models-root" => models_root = Some(PathBuf::from(value()?)),
            "--threads" => threads = value()?.parse().context("--threads is a number")?,
            other => bail!("unknown argument {other}"),
        }
    }
    let wav = wav.context("--wav is required")?;
    let models_root = models_root.context("--models-root is required")?;
    let threads = threads.clamp(1, 2);

    let wave = sherpa_onnx::Wave::read(&wav).with_context(|| format!("cannot read {wav}"))?;
    if wave.sample_rate() != SAMPLE_RATE as i32 {
        bail!(
            "{wav} is {} Hz; the engine needs {SAMPLE_RATE} Hz mono",
            wave.sample_rate()
        );
    }
    let samples = wave.samples().to_vec();
    if samples.is_empty() {
        bail!("{wav} has no samples");
    }

    let mut manifest = Manifest::embedded();
    manifest
        .groups
        .retain(|group| group.component == "transcription");
    let mut pass = SpeechPass::with_settings(
        manifest,
        models_root,
        Threads {
            available: threads,
            physical: threads,
            asr: threads,
            diarization: threads,
        },
        Settings {
            // One explicit piece: no VAD, no piece plan, no hole re-decode.
            gap_redecode_seconds: 0.0,
            ..Settings::default()
        },
    );
    let piece = vec![(0usize, samples.len())];
    let words = pass.transcribe_pieces(&samples, &piece, piece.clone(), &mut |_| {})?;
    let word_list: Vec<&str> = words.iter().map(|word| word.w.as_str()).collect();
    println!(
        "{}",
        json!({
            "wav": wav,
            "audio_seconds": samples.len() as f64 / SAMPLE_RATE as f64,
            "threads": threads,
            "words": words.len(),
            "text": word_list.join(" "),
            "word_list": word_list,
        })
    );
    Ok(())
}
