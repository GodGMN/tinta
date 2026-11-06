//! Benchmark harness for sherpa-onnx on CPU: offline ASR (NeMo Parakeet TDT)
//! and offline speaker diarization (pyannote segmentation 3.0 + speaker
//! embedding). See docs/fork/benchmark.md for the reproduction commands.
//!
//! Every stage writes one JSON object. Timings are wall clock. Model load is
//! measured twice: "cold" after evicting the model files from the page cache
//! (posix_fadvise DONTNEED), "warm" by building a second instance right after.

use serde_json::{json, Value};
use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OfflineSpeakerDiarization,
    OfflineSpeakerDiarizationConfig, OfflineSpeakerSegmentationModelConfig,
    OfflineSpeakerSegmentationPyannoteModelConfig, OfflineTransducerModelConfig, SpeakerEmbeddingExtractorConfig,
    Wave,
};
use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::time::Instant;

struct Args {
    cmd: String,
    flags: std::collections::HashMap<String, String>,
}

impl Args {
    fn parse() -> Self {
        let mut it = std::env::args().skip(1);
        let cmd = it.next().unwrap_or_default();
        let mut flags = std::collections::HashMap::new();
        while let Some(a) = it.next() {
            if let Some(key) = a.strip_prefix("--") {
                let val = it.next().unwrap_or_default();
                flags.insert(key.to_string(), val);
            }
        }
        Args { cmd, flags }
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.flags.get(key).map(|s| s.as_str())
    }

    fn req(&self, key: &str) -> String {
        self.get(key)
            .unwrap_or_else(|| panic!("missing --{key}"))
            .to_string()
    }

    fn i32_or(&self, key: &str, default: i32) -> i32 {
        self.get(key)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    fn f32_or(&self, key: &str, default: f32) -> f32 {
        self.get(key)
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }
}

/// Drop the given files from the page cache so the next read is cold.
fn evict(paths: &[&str]) {
    for p in paths {
        if let Ok(f) = File::open(p) {
            let _ = f.sync_all();
            unsafe {
                libc::posix_fadvise(f.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
            }
        }
    }
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn peak_rss_mb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest
                .trim()
                .trim_end_matches(" kB")
                .trim()
                .parse()
                .unwrap_or(0.0);
            return kb / 1024.0;
        }
    }
    0.0
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

struct AsrModels<'a> {
    encoder: &'a str,
    decoder: &'a str,
    joiner: &'a str,
    tokens: &'a str,
}

fn build_recognizer(m: &AsrModels, threads: i32) -> Option<OfflineRecognizer> {
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.transducer = OfflineTransducerModelConfig {
        encoder: Some(m.encoder.to_string()),
        decoder: Some(m.decoder.to_string()),
        joiner: Some(m.joiner.to_string()),
    };
    config.model_config.tokens = Some(m.tokens.to_string());
    config.model_config.model_type = Some("nemo_transducer".to_string());
    config.model_config.provider = Some("cpu".to_string());
    config.model_config.num_threads = threads;
    OfflineRecognizer::create(&config)
}

fn build_diarizer(
    seg_model: &str,
    embedding: &str,
    threads: i32,
    threshold: f32,
    num_speakers: i32,
    min_on: f32,
    min_off: f32,
    window_shift: f32,
) -> Option<OfflineSpeakerDiarization> {
    let mut config = OfflineSpeakerDiarizationConfig::default();
    config.segmentation = OfflineSpeakerSegmentationModelConfig {
        pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
            model: Some(seg_model.to_string()),
            window_shift_ratio: window_shift,
        },
        num_threads: threads,
        debug: false,
        provider: Some("cpu".to_string()),
    };
    config.embedding = SpeakerEmbeddingExtractorConfig {
        model: Some(embedding.to_string()),
        num_threads: threads,
        debug: false,
        provider: Some("cpu".to_string()),
    };
    config.clustering.num_clusters = num_speakers;
    config.clustering.threshold = threshold;
    config.min_duration_on = min_on;
    config.min_duration_off = min_off;
    OfflineSpeakerDiarization::create(&config)
}

#[derive(Clone)]
struct Word {
    start: f32,
    end: f32,
    text: String,
}

/// Turn subword tokens into words using the sentencepiece word marker (▁ or a
/// leading space). Timestamps are per token; a word spans its tokens.
fn tokens_to_words(tokens: &[String], ts: &[f32], dur: Option<&[f32]>) -> Vec<Word> {
    let mut words: Vec<Word> = Vec::new();
    let mut cur: Option<Word> = None;
    for (i, tok) in tokens.iter().enumerate() {
        if tok.is_empty() {
            continue;
        }
        let start = ts.get(i).copied().unwrap_or(0.0);
        let end = start + dur.and_then(|d| d.get(i)).copied().unwrap_or(0.0);
        let is_new = tok.starts_with('▁') || tok.starts_with(' ');
        let text = tok.trim_start_matches('▁').trim_start();
        if is_new || cur.is_none() {
            if let Some(w) = cur.take() {
                words.push(w);
            }
            cur = Some(Word {
                start,
                end,
                text: text.to_string(),
            });
        } else if let Some(w) = cur.as_mut() {
            w.text.push_str(text);
            w.end = end;
        }
    }
    if let Some(w) = cur {
        words.push(w);
    }
    words
}

fn run_asr(
    rec: &OfflineRecognizer,
    samples: &[f32],
    offset_s: f32,
) -> (String, Vec<String>, Vec<f32>, Option<Vec<f32>>) {
    let stream = rec.create_stream();
    stream.accept_waveform(16000, samples);
    rec.decode(&stream);
    let r = stream.get_result().expect("asr result");
    let ts: Vec<f32> = r
        .timestamps
        .clone()
        .unwrap_or_default()
        .iter()
        .map(|t| t + offset_s)
        .collect();
    (r.text, r.tokens, ts, r.durations)
}

fn main() {
    let args = Args::parse();
    let wav_path = args.req("wav");
    let threads = args.i32_or("threads", 4);
    let out_path = args.req("output");

    let wave = Wave::read(&wav_path).expect("read wav");
    assert_eq!(wave.sample_rate(), 16000, "expected 16 kHz mono wav");
    let samples = wave.samples();
    let audio_s = samples.len() as f64 / 16000.0;

    let mut out = json!({
        "command": args.cmd,
        "wav": wav_path,
        "audio_seconds": audio_s,
        "threads": threads,
        "loadavg_start": loadavg(),
        "machine": {
            "cpu": std::fs::read_to_string("/proc/cpuinfo").ok().and_then(|s| s.lines().find(|l| l.starts_with("model name")).map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())).unwrap_or_default(),
            "cores": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0),
        },
    });

    let run_asr_stage = args.cmd == "asr" || args.cmd == "combined";
    let run_diar_stage = args.cmd == "diarize" || args.cmd == "combined";

    let mut asr_seconds = 0.0f64;
    let mut diar_seconds = 0.0f64;

    if run_asr_stage {
        let models = AsrModels {
            encoder: &args.req("encoder"),
            decoder: &args.req("decoder"),
            joiner: &args.req("joiner"),
            tokens: &args.req("tokens"),
        };
        evict(&[models.encoder, models.decoder, models.joiner, models.tokens]);
        let t = Instant::now();
        let rec = build_recognizer(&models, threads).expect("create recognizer");
        let load_cold = ms_since(t);

        let t = Instant::now();
        let rec2 = build_recognizer(&models, threads).expect("create recognizer (warm)");
        let load_warm = ms_since(t);
        drop(rec);
        drop(rec2);

        let rec = build_recognizer(&models, threads).expect("create recognizer");

        let chunk_s = args.f32_or("chunk-seconds", 0.0);
        let overlap = args.f32_or("chunk-overlap", 1.0);
        let t = Instant::now();
        let text;
        let mut tokens = Vec::new();
        let mut ts = Vec::new();
        let mut durations = Vec::<f32>::new();
        let mut words = Vec::<Word>::new();
        let mut chunks = 0;
        if chunk_s <= 0.0 {
            let (t1, tk, tsv, dur) = run_asr(&rec, samples, 0.0);
            text = t1;
            words = tokens_to_words(&tk, &tsv, dur.as_deref());
            tokens = tk;
            ts = tsv;
            durations = dur.unwrap_or_default();
            chunks = 1;
        } else {
            let chunk = (chunk_s * 16000.0) as usize;
            let step = ((chunk_s - overlap).max(0.5) * 16000.0) as usize;
            let mut pos = 0usize;
            while pos < samples.len() {
                let end = (pos + chunk).min(samples.len());
                let off = pos as f32 / 16000.0;
                let (t1, tk, tsv, dur) = run_asr(&rec, &samples[pos..end], off);
                let mut w = tokens_to_words(&tk, &tsv, dur.as_deref());
                // Drop words that start in the overlap the previous chunk
                // already transcribed, so no word is counted twice.
                if pos > 0 {
                    let keep_from = off + overlap / 2.0;
                    w.retain(|x| x.start >= keep_from);
                }
                let _ = t1;
                words.extend(w);
                tokens.extend(tk);
                ts.extend(tsv);
                durations.extend(dur.unwrap_or_default());
                chunks += 1;
                if end == samples.len() {
                    break;
                }
                pos += step;
            }
            text = words.iter().map(|w| w.text.as_str()).collect::<Vec<_>>().join(" ");
        }
        asr_seconds = t.elapsed().as_secs_f64();
        let ts_min = ts.iter().cloned().fold(f32::INFINITY, f32::min);
        let ts_max = ts.iter().cloned().fold(0.0f32, f32::max);
        out["asr"] = json!({
            "load_ms_cold": load_cold,
            "load_ms_warm": load_warm,
            "decode_seconds": asr_seconds,
            "rtf": asr_seconds / audio_s,
            "chunks": chunks,
            "text": text,
            "tokens": tokens,
            "token_timestamps": ts,
            "token_durations": durations,
            "word_count": words.len(),
            "has_word_timestamps": !ts.is_empty() && words.iter().all(|w| w.end >= w.start),
            "timestamp_min": if ts.is_empty() { 0.0 } else { ts_min },
            "timestamp_max": if ts.is_empty() { 0.0 } else { ts_max },
            "words": words.iter().map(|w| json!({"start": w.start, "end": w.end, "text": w.text})).collect::<Vec<_>>(),
        });
    }

    let mut diar_segments: Vec<(f32, f32, i32)> = Vec::new();
    if run_diar_stage {
        let seg_model = args.req("seg-model");
        let embedding = args.req("embedding");
        evict(&[&seg_model, &embedding]);
        let t = Instant::now();
        let d = build_diarizer(
            &seg_model,
            &embedding,
            threads,
            args.f32_or("threshold", 0.5),
            args.i32_or("num-speakers", -1),
            args.f32_or("min-duration-on", 0.3),
            args.f32_or("min-duration-off", 0.5),
            args.f32_or("window-shift", 0.1),
        )
        .expect("create diarizer");
        let load_cold = ms_since(t);
        let t = Instant::now();
        let d2 = build_diarizer(
            &seg_model,
            &embedding,
            threads,
            args.f32_or("threshold", 0.5),
            args.i32_or("num-speakers", -1),
            args.f32_or("min-duration-on", 0.3),
            args.f32_or("min-duration-off", 0.5),
            args.f32_or("window-shift", 0.1),
        )
        .expect("create diarizer (warm)");
        let load_warm = ms_since(t);
        drop(d2);

        let t = Instant::now();
        let result = d.process(samples).expect("diarize");
        diar_seconds = t.elapsed().as_secs_f64();
        let segs = result.sort_by_start_time();
        diar_segments = segs.iter().map(|s| (s.start, s.end, s.speaker)).collect();
        out["diarization"] = json!({
            "seg_model": seg_model,
            "embedding_model": embedding,
            "threshold": args.f32_or("threshold", 0.5),
            "window_shift": args.f32_or("window-shift", 0.1),
            "load_ms_cold": load_cold,
            "load_ms_warm": load_warm,
            "process_seconds": diar_seconds,
            "rtf": diar_seconds / audio_s,
            "num_speakers": result.num_speakers(),
            "num_segments": result.num_segments(),
            "segments": diar_segments.iter().map(|(s, e, sp)| json!({"start": s, "end": e, "speaker": sp})).collect::<Vec<_>>(),
        });
    }

    if args.cmd == "combined" {
        let total = asr_seconds + diar_seconds;
        // Assign every ASR word to the diarization speaker it overlaps most.
        let words: Vec<Value> = out["asr"]["words"].as_array().cloned().unwrap_or_default();
        let mut turns: Vec<Value> = Vec::new();
        for w in &words {
            let ws = w["start"].as_f64().unwrap_or(0.0) as f32;
            let we = w["end"].as_f64().unwrap_or(0.0) as f32;
            let mut best = (-1, 0.0f32);
            for (s, e, sp) in &diar_segments {
                let ov = (we.min(*e) - ws.max(*s)).max(0.0);
                if ov > best.1 {
                    best = (*sp, ov);
                }
            }
            let spk = if best.0 < 0 {
                // No overlap: nearest segment by midpoint distance.
                let mid = (ws + we) / 2.0;
                diar_segments
                    .iter()
                    .min_by(|a, b| {
                        let da = if mid < a.0 { a.0 - mid } else if mid > a.1 { mid - a.1 } else { 0.0 };
                        let db = if mid < b.0 { b.0 - mid } else if mid > b.1 { mid - b.1 } else { 0.0 };
                        da.partial_cmp(&db).unwrap()
                    })
                    .map(|x| x.2)
                    .unwrap_or(-1)
            } else {
                best.0
            };
            let text = w["text"].as_str().unwrap_or("").to_string();
            let append = turns
                .last()
                .map(|t| t["speaker"].as_i64().unwrap_or(-2) == spk as i64)
                .unwrap_or(false);
            if append {
                let last = turns.last_mut().unwrap();
                let t2 = format!("{} {}", last["text"].as_str().unwrap_or(""), text);
                last["text"] = json!(t2.trim());
                last["end"] = json!(we);
            } else {
                turns.push(json!({"speaker": spk, "start": ws, "end": we, "text": text}));
            }
        }
        out["combined"] = json!({
            "asr_seconds": asr_seconds,
            "diarization_seconds": diar_seconds,
            "total_seconds": total,
            "rtf": total / audio_s,
            "turns": turns,
        });
    }

    out["peak_rss_mb"] = json!(peak_rss_mb());
    out["loadavg_end"] = json!(loadavg());
    if let Some(dir) = Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut f = File::create(&out_path).expect("create output");
    f.write_all(serde_json::to_string_pretty(&out).unwrap().as_bytes())
        .expect("write output");
}
