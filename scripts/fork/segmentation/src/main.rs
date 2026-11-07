//! Offline harness for the engine's own segmentation (`segment.rs` and
//! `SpeechPass::transcribe`), used by task #21 to measure the pipeline WER of a
//! whole meeting against the oracle segments.
//!
//! It links `tinta-engine-rs`, reads one 16 kHz mono meeting wav, runs the
//! engine's VAD + piece planner + recognizer path exactly as `finalize` does,
//! and writes the words plus the VAD ranges and piece count as JSON:
//!
//! ```text
//! tinta-segmentation --wav ES2004b.Mix-Headset.wav --models-root MODELS \
//!   --variant short-gap --output out.json
//! ```
//!
//! The variants are settings for the same code path. `score.py` turns the JSON
//! into the same-reference ablation table.

use anyhow::{bail, Context, Result};
use serde_json::json;
use sherpa_onnx::{VoiceActivityDetector, Wave};
use std::path::PathBuf;
use std::time::Instant;
use tinta_engine_rs::audio::SAMPLE_RATE;
use tinta_engine_rs::manifest::Manifest;
use tinta_engine_rs::segment::{covers_speech, group_pieces_with, vad_segments, Settings};
use tinta_engine_rs::speech::{vad_config, SpeechPass};
use tinta_engine_rs::threads::Threads;

const VAD_BUFFER_SECONDS: f32 = 120.0;

struct Args {
    wav: String,
    models_root: PathBuf,
    variant: Option<String>,
    output: Option<PathBuf>,
    threads: usize,
    /// `START:END` seconds: decode just that window and print its words, for
    /// diagnosing one piece without a whole-meeting run.
    window: Option<(f32, f32)>,
    /// A diarization result JSON: decode each speaker turn on its own instead of
    /// the VAD pieces, for the step 4 experiment.
    turns: Option<PathBuf>,
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1);
    let (mut wav, mut models_root, mut variant, mut output) = (None, None, None, None);
    let mut threads = 2;
    let mut window = None;
    let mut turns = None;
    while let Some(flag) = args.next() {
        let mut value =
            || -> Result<String> { args.next().context(format!("{flag} needs a value")) };
        match flag.as_str() {
            "--wav" => wav = Some(value()?),
            "--models-root" => models_root = Some(PathBuf::from(value()?)),
            "--variant" => variant = Some(value()?),
            "--output" => output = Some(PathBuf::from(value()?)),
            "--turns" => turns = Some(PathBuf::from(value()?)),
            "--threads" => threads = value()?.parse().context("--threads is a number")?,
            "--window" => {
                let raw = value()?;
                let (start, end) = raw.split_once(':').context("--window is START:END")?;
                window = Some((
                    start.parse().context("--window start")?,
                    end.parse().context("--window end")?,
                ));
            }
            other => bail!("unknown argument {other}"),
        }
    }
    Ok(Args {
        wav: wav.context("--wav is required")?,
        models_root: models_root.context("--models-root is required")?,
        variant,
        output,
        threads,
        window,
        turns,
    })
}

/// The settings of one measured variant. Every variant runs the same engine
/// code; only these knobs change, so the table attributes the delta to the
/// segmentation rule and nothing else.
fn variant_settings(name: &str) -> Result<Settings> {
    let base = Settings::default();
    Ok(match name {
        // The engine defaults: the winning combination in this file.
        "default" => base,
        // The engine before task #21: a 30 s VAD run closed by the VAD itself,
        // one piece per run, no gap re-decode. (The 20 s flush that task #19
        // measured lived in the old `tinta-bench` harness, not in the engine.)
        "before" => Settings {
            max_piece_seconds: 30.0,
            pre_roll_seconds: 2.0,
            post_roll_seconds: 1.0,
            min_piece_seconds: 10.0,
            join_gap_seconds: 0.5,
            vad_max_speech_seconds: 30.0,
            gap_redecode_seconds: 0.0,
        },
        // Step 1: let the VAD run past 30 s so it never cuts mid-speech.
        "no-vad-cut" => Settings {
            max_piece_seconds: 30.0,
            vad_max_speech_seconds: 60.0,
            gap_redecode_seconds: 0.0,
            ..base
        },
        // Step 2: 12 s of speech per piece (<= 15 s with the rolls), cut at the
        // quietest point, starting at a quiet point.
        "short" => Settings {
            max_piece_seconds: 12.0,
            gap_redecode_seconds: 0.0,
            ..base
        },
        // Step 3: step 2 plus a re-decode of hypothesis holes >= 3 s.
        "short-gap" => Settings {
            max_piece_seconds: 12.0,
            ..base
        },
        // Pieces only cut at VAD silences: a run is one piece up to the hard
        // cap. This is the "never cut inside speech" reading of step 1.
        "runs" => Settings {
            max_piece_seconds: 57.0,
            vad_max_speech_seconds: 60.0,
            gap_redecode_seconds: 0.0,
            ..base
        },
        // Step 3 on top of silence-only cuts.
        "runs-gap" => Settings {
            max_piece_seconds: 57.0,
            vad_max_speech_seconds: 60.0,
            ..base
        },
        // Silence-only cuts with a 30 s target, for the piece-length sweep.
        "mid-gap" => Settings {
            max_piece_seconds: 30.0,
            vad_max_speech_seconds: 60.0,
            ..base
        },
        // Silence-only cuts that also join pauses up to 1 s.
        "runs-join1-gap" => Settings {
            max_piece_seconds: 57.0,
            join_gap_seconds: 1.0,
            vad_max_speech_seconds: 60.0,
            ..base
        },
        // Silence-only cuts with a 2 s hole threshold for the re-decode.
        "runs-gap2s" => Settings {
            max_piece_seconds: 57.0,
            vad_max_speech_seconds: 60.0,
            gap_redecode_seconds: 2.0,
            ..base
        },
        // Do not extend a short VAD run with its neighbours.
        "runs-min3-gap" => Settings {
            max_piece_seconds: 57.0,
            min_piece_seconds: 3.0,
            vad_max_speech_seconds: 60.0,
            ..base
        },
        // Less context around a run.
        "runs-roll1-gap" => Settings {
            max_piece_seconds: 57.0,
            pre_roll_seconds: 1.0,
            vad_max_speech_seconds: 60.0,
            ..base
        },
        // A shorter piece for tuning.
        "short-10" => Settings {
            max_piece_seconds: 10.0,
            gap_redecode_seconds: 0.0,
            ..base
        },
        // A longer join gap: merge pauses up to 1 s into one piece.
        "short-join1" => Settings {
            join_gap_seconds: 1.0,
            gap_redecode_seconds: 0.0,
            ..base
        },
        other => bail!("unknown variant {other}"),
    })
}

fn settings_json(settings: &Settings) -> serde_json::Value {
    json!({
        "max_piece_seconds": settings.max_piece_seconds,
        "pre_roll_seconds": settings.pre_roll_seconds,
        "post_roll_seconds": settings.post_roll_seconds,
        "min_piece_seconds": settings.min_piece_seconds,
        "join_gap_seconds": settings.join_gap_seconds,
        "vad_max_speech_seconds": settings.vad_max_speech_seconds,
        "gap_redecode_seconds": settings.gap_redecode_seconds,
    })
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let wave = Wave::read(&args.wav).with_context(|| format!("cannot read {}", args.wav))?;
    if wave.sample_rate() != SAMPLE_RATE as i32 {
        bail!(
            "expected a {} Hz wav, got {}",
            SAMPLE_RATE,
            wave.sample_rate()
        );
    }
    let samples = wave.samples().to_vec();
    let threads = Threads {
        available: args.threads,
        physical: args.threads,
        asr: args.threads,
        diarization: args.threads,
    };
    let manifest = Manifest::embedded();
    if !manifest.all_files_exist(&args.models_root) {
        bail!(
            "the models root {} does not have the manifest layout",
            args.models_root.display()
        );
    }
    if let Some((start, end)) = args.window {
        return decode_window(&args, &manifest, threads, &samples, start, end);
    }
    if let Some(turns) = &args.turns {
        return decode_turns(&args, &manifest, threads, &samples, turns);
    }
    let (variant, output) = (
        args.variant.context("--variant is required")?,
        args.output.context("--output is required")?,
    );
    let settings = variant_settings(&variant)?;

    // The planning pass, for the numbers in the report. `transcribe` runs the
    // same detector and planner again; both are deterministic.
    let vad_model = args
        .models_root
        .join(manifest.group("vad")?.file("vad")?.path.clone())
        .to_string_lossy()
        .into_owned();
    let started = Instant::now();
    let vad = VoiceActivityDetector::create(
        &vad_config(
            vad_model,
            args.threads,
            crate::provider(&args.models_root)?,
            &settings,
        ),
        VAD_BUFFER_SECONDS,
    )
    .context("the VAD model did not load")?;
    let ranges = vad_segments(&vad, &samples);
    let pieces = group_pieces_with(&settings, &samples, &ranges);
    drop(vad);
    let speech_seconds: usize = ranges.iter().map(|(start, end)| end - start).sum();
    let vad_seconds = started.elapsed().as_secs_f64();
    if !covers_speech(&pieces, &ranges) {
        bail!("the piece plan does not cover every VAD speech region");
    }
    let longest = pieces
        .iter()
        .map(|(start, end)| (*end - *start) as f32 / SAMPLE_RATE as f32)
        .fold(0.0f32, f32::max);
    eprintln!(
        "{} {}: {} VAD regions, {} pieces, longest {:.2} s",
        variant,
        args.wav,
        ranges.len(),
        pieces.len(),
        longest
    );

    let started = Instant::now();
    let mut pass = SpeechPass::with_settings(manifest, args.models_root.clone(), threads, settings);
    let mut last_tenth = 0u32;
    let words =
        tinta_engine_rs::speech::FinalPass::transcribe(&mut pass, &samples, &mut |fraction| {
            let tenth = (fraction * 10.0) as u32;
            if tenth > last_tenth {
                last_tenth = tenth;
                eprintln!("  transcription {:.0} %", fraction * 100.0);
            }
        })?;
    let pass_seconds = started.elapsed().as_secs_f64();
    let piece_words: Vec<usize> = pieces
        .iter()
        .map(|(start, end)| {
            let (from, to) = (
                *start as f64 / SAMPLE_RATE as f64,
                *end as f64 / SAMPLE_RATE as f64,
            );
            words
                .iter()
                .filter(|word| {
                    let middle = (word.s + word.e) / 2.0;
                    middle >= from && middle < to
                })
                .count()
        })
        .collect();

    let document = json!({
        "wav": args.wav,
        "variant": variant,
        "settings": settings_json(&settings),
        "threads": args.threads,
        "audio_seconds": samples.len() as f64 / SAMPLE_RATE as f64,
        "speech_seconds": speech_seconds as f64 / SAMPLE_RATE as f64,
        "vad_seconds": vad_seconds,
        "pass_seconds": pass_seconds,
        "pieces": pieces.len(),
        "longest_piece_seconds": longest,
        "piece_list": pieces
            .iter()
            .map(|(start, end)| json!([
                *start as f64 / SAMPLE_RATE as f64,
                *end as f64 / SAMPLE_RATE as f64
            ]))
            .collect::<Vec<_>>(),
        "piece_words": piece_words,
        "asr": {
            "words": words
                .iter()
                .map(|word| json!({ "text": word.w, "start": word.s, "end": word.e }))
                .collect::<Vec<_>>(),
            "vad": {
                "ranges": ranges
                    .iter()
                    .map(|(start, end)| json!([
                        *start as f64 / SAMPLE_RATE as f64,
                        *end as f64 / SAMPLE_RATE as f64
                    ]))
                    .collect::<Vec<_>>(),
                "pieces": pieces.len(),
                "max_piece_seconds": longest,
                "speech_seconds": speech_seconds as f64 / SAMPLE_RATE as f64,
                "max_speech_duration": settings.vad_max_speech_seconds,
            },
        },
    });
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, serde_json::to_vec_pretty(&document)?)
        .with_context(|| format!("cannot write {}", output.display()))?;
    eprintln!("wrote {}", output.display());
    Ok(())
}

/// Builds the recognizer with the engine's configuration.
fn build_recognizer(
    args: &Args,
    manifest: &Manifest,
    threads: Threads,
) -> Result<sherpa_onnx::OfflineRecognizer> {
    let group = manifest.group("transcription")?;
    let path = |role: &str| -> Result<String> {
        Ok(args
            .models_root
            .join(&group.file(role)?.path)
            .to_string_lossy()
            .into_owned())
    };
    let config = sherpa_onnx::OfflineRecognizerConfig {
        model_config: sherpa_onnx::OfflineModelConfig {
            transducer: sherpa_onnx::OfflineTransducerModelConfig {
                encoder: Some(path("encoder")?),
                decoder: Some(path("decoder")?),
                joiner: Some(path("joiner")?),
            },
            tokens: Some(path("tokens")?),
            model_type: Some("nemo_transducer".to_string()),
            provider: Some(crate::provider(&args.models_root)?),
            num_threads: threads.asr as i32,
            debug: false,
            ..Default::default()
        },
        ..Default::default()
    };
    sherpa_onnx::OfflineRecognizer::create(&config).context("the transcription model did not load")
}

/// Decodes `[from, to)` samples to engine words, with the recognizer built once.
fn decode_span(
    recognizer: &sherpa_onnx::OfflineRecognizer,
    samples: &[f32],
    from: usize,
    to: usize,
) -> Result<Vec<tinta_engine_rs::types::Word>> {
    let stream = recognizer.create_stream();
    stream.accept_waveform(SAMPLE_RATE as i32, &samples[from..to]);
    recognizer.decode(&stream);
    let result = stream.get_result().context("no result")?;
    Ok(tinta_engine_rs::speech::tokens_to_words(
        &result.tokens,
        result.timestamps.as_deref(),
        result.durations.as_deref(),
        from as f64 / SAMPLE_RATE as f64,
    ))
}

/// Decodes one `[start, end)` window and prints its tokens and timestamps. The
/// words, not the text, are what the segmentation is judged on, so this shows
/// whether a dropped stretch is missing from the decoder output or only
/// mis-timestamped.
fn decode_window(
    args: &Args,
    manifest: &Manifest,
    threads: Threads,
    samples: &[f32],
    start: f32,
    end: f32,
) -> Result<()> {
    let recognizer = build_recognizer(args, manifest, threads)?;
    let (from, to) = (
        (start * SAMPLE_RATE as f32) as usize,
        (end * SAMPLE_RATE as f32) as usize,
    );
    let stream = recognizer.create_stream();
    stream.accept_waveform(SAMPLE_RATE as i32, &samples[from..to]);
    recognizer.decode(&stream);
    let result = stream.get_result().context("no result")?;
    println!(
        "window {start}-{end} s = {:.2} s audio",
        (to - from) as f32 / SAMPLE_RATE as f32
    );
    println!("text: {}", result.text);
    let timestamps = result.timestamps.unwrap_or_default();
    let durations = result.durations.unwrap_or_default();
    for (index, token) in result.tokens.iter().enumerate() {
        println!(
            "{:>7.3} +{:.3} {token}",
            timestamps.get(index).copied().unwrap_or(0.0),
            durations.get(index).copied().unwrap_or(0.0),
        );
    }
    Ok(())
}

/// Reads diarization turns as `(start, end, speaker)` seconds from either the
/// benchmark result shape (`{"diarization": {"segments": [...]}}`) or a bare
/// array. Same-speaker turns closer than 0.5 s are joined.
fn load_turns(path: &PathBuf) -> Result<Vec<(f32, f32, i64)>> {
    let value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).context("cannot read the turns file")?)?;
    let array = value
        .pointer("/diarization/segments")
        .and_then(|segments| segments.as_array())
        .or_else(|| value.as_array())
        .context("the turns file has no diarization.segments")?;
    let mut turns: Vec<(f32, f32, i64)> = array
        .iter()
        .map(|segment| {
            Ok((
                segment["start"].as_f64().context("turn start")? as f32,
                segment["end"].as_f64().context("turn end")? as f32,
                segment["speaker"].as_i64().unwrap_or(0),
            ))
        })
        .collect::<Result<_>>()?;
    turns.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut joined: Vec<(f32, f32, i64)> = Vec::new();
    for (start, end, speaker) in turns {
        if let Some(last) = joined.last_mut() {
            if last.2 == speaker && start - last.1 <= 0.5 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        joined.push((start, end, speaker));
    }
    Ok(joined)
}

/// Step 4 experiment: decode each diarized speaker turn on its own, allowing
/// turns to overlap, and merge the words. The oracle decodes each reference turn
/// separately; this is the closest a single pass can get to that without the
/// reference boundaries.
fn decode_turns(
    args: &Args,
    manifest: &Manifest,
    threads: Threads,
    samples: &[f32],
    turns_path: &PathBuf,
) -> Result<()> {
    let output = args.output.clone().context("--output is required")?;
    let turns = load_turns(turns_path)?;
    let recognizer = build_recognizer(args, manifest, threads)?;
    let pre = (0.5 * SAMPLE_RATE as f32) as usize;
    let post = (0.5 * SAMPLE_RATE as f32) as usize;
    let mut words: Vec<tinta_engine_rs::types::Word> = Vec::new();
    let mut pieces = Vec::new();
    let mut decoded_pieces: Vec<Vec<tinta_engine_rs::types::Word>> = Vec::new();
    let started = Instant::now();
    for (index, (start, end, _speaker)) in turns.iter().enumerate() {
        let from = ((*start * SAMPLE_RATE as f32) as usize).saturating_sub(pre);
        let to = (((*end * SAMPLE_RATE as f32) as usize) + post).min(samples.len());
        if to <= from {
            continue;
        }
        pieces.push((from, to));
        let mut decoded = decode_span(&recognizer, samples, from, to)?;
        // A turn that comes back nearly empty is retried from a later start,
        // like the engine's window retry.
        if decoded.is_empty() && to - from > SAMPLE_RATE as usize {
            let shifted = (from + SAMPLE_RATE as usize / 2).min(to - 1);
            let retry = decode_span(&recognizer, samples, shifted, to)?;
            if retry.len() > decoded.len() {
                decoded = retry;
            }
        }
        words.extend(decoded.clone());
        decoded_pieces.push(decoded);
        if index % 50 == 0 {
            eprintln!("  turn {}/{}", index + 1, turns.len());
        }
    }
    let pass_seconds = started.elapsed().as_secs_f64();
    words.sort_by(|a, b| a.s.partial_cmp(&b.s).unwrap_or(std::cmp::Ordering::Equal));
    let before = words.len();
    words = dedupe_overlaps(words);
    // Also report the engine's own midpoint de-duplication, which drops words at
    // every join instead of only identical overlapping occurrences.
    let joined = tinta_engine_rs::segment::trim_overlaps(&pieces, &mut decoded_pieces);
    eprintln!(
        "{} turns -> {} pieces, {} words ({} duplicate occurrences dropped), {} joined words, {:.0} s",
        turns.len(),
        pieces.len(),
        words.len(),
        before - words.len(),
        joined.len(),
        pass_seconds
    );
    let document = json!({
        "wav": args.wav,
        "variant": "turns",
        "threads": args.threads,
        "audio_seconds": samples.len() as f64 / SAMPLE_RATE as f64,
        "pass_seconds": pass_seconds,
        "turns": turns.len(),
        "pieces": pieces.len(),
        "piece_list": pieces
            .iter()
            .map(|(start, end)| json!([
                *start as f64 / SAMPLE_RATE as f64,
                *end as f64 / SAMPLE_RATE as f64
            ]))
            .collect::<Vec<_>>(),
        "asr": {
            "words": words
                .iter()
                .map(|word| json!({ "text": word.w, "start": word.s, "end": word.e }))
                .collect::<Vec<_>>(),
        },
        "joined": joined
            .iter()
            .map(|word| json!({ "text": word.w, "start": word.s, "end": word.e }))
            .collect::<Vec<_>>(),
    });
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&output, serde_json::to_vec_pretty(&document)?)
        .with_context(|| format!("cannot write {}", output.display()))?;
    eprintln!("wrote {}", output.display());
    Ok(())
}

/// Drops the same word decoded twice from overlapping turns: equal text with
/// overlapping spans. Distinct words of two speakers at the same time stay.
fn dedupe_overlaps(words: Vec<tinta_engine_rs::types::Word>) -> Vec<tinta_engine_rs::types::Word> {
    let mut kept: Vec<tinta_engine_rs::types::Word> = Vec::new();
    for word in words {
        let duplicate = kept.iter().rev().take(40).any(|other| {
            other.s < word.e
                && word.s < other.e
                && other.w.eq_ignore_ascii_case(&word.w)
                && other.e.min(word.e) - other.s.max(word.s) > 0.25
        });
        if !duplicate {
            kept.push(word);
        }
    }
    kept
}

/// The engine's provider string with intra-op spinning off.
fn provider(root: &std::path::Path) -> Result<String> {
    tinta_engine_rs::sherpa::provider(root)
}
