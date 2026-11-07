//! Benchmark harness for sherpa-onnx on CPU: offline ASR (NeMo Parakeet TDT)
//! and offline speaker diarization (pyannote segmentation 3.0 + speaker
//! embedding). See docs/fork/benchmark.md for the reproduction commands.
//!
//! ASR is segmented with the sherpa-onnx Silero VAD into <= 30 s pieces cut at
//! silence, with a hard 60 s cap per model call. The old fixed-chunk mode is
//! kept only for the ablation (`--chunk-seconds`), with the overlap trimmed at
//! the midpoint on both sides.
//!
//! Every stage writes one JSON object. Timings are wall clock. Model load is
//! measured twice: "cold" after evicting the model files from the page cache
//! (posix_fadvise DONTNEED), "warm" by building a second instance right after.

mod clustering;

use clustering::{recluster, speech_shares, Options as ClusterOptions, Unit};
use serde_json::{json, Value};
use sherpa_onnx::{
    OfflineRecognizer, OfflineRecognizerConfig, OfflineSpeakerDiarization,
    OfflineSpeakerDiarizationConfig, OfflineSpeakerSegmentationModelConfig,
    OfflineSpeakerSegmentationPyannoteModelConfig, OfflineTransducerModelConfig,
    SileroVadModelConfig, SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig,
    VadModelConfig, VoiceActivityDetector, Wave,
};
use std::collections::HashSet;
use std::fs::File;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::Path;
use std::time::Instant;

const SAMPLE_RATE: usize = 16000;
const MAX_ASR_SECONDS: f32 = 60.0;
const VAD_PIECE_SECONDS: f32 = 30.0;
const VAD_PRE_ROLL_SECONDS: f32 = 2.0;
const VAD_POST_ROLL_SECONDS: f32 = 1.0;
const VAD_MIN_PIECE_SECONDS: f32 = 10.0;
const VAD_SPLIT_WINDOW_SECONDS: f32 = 0.05;
const VAD_JOIN_GAP_SECONDS: f32 = 0.5;
const VAD_FLUSH_SECONDS: f32 = 20.0;

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

fn cpu_seconds() -> f64 {
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) != 0 {
            return 0.0;
        }
        ru.ru_utime.tv_sec as f64
            + ru.ru_utime.tv_usec as f64 / 1e6
            + ru.ru_stime.tv_sec as f64
            + ru.ru_stime.tv_usec as f64 / 1e6
    }
}

fn ms_since(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

/// CPU ids this process may actually run on.
fn affinity_ids() -> Vec<usize> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &mut set,
        ) != 0
        {
            return Vec::new();
        }
        (0..libc::CPU_SETSIZE as usize)
            .filter(|i| libc::CPU_ISSET(*i, &set))
            .collect()
    }
}

fn read_first<T: std::str::FromStr>(path: &str) -> Option<T> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Physical cores among the CPUs the process may use (package, core) pairs.
fn physical_cores(ids: &[usize]) -> usize {
    let mut cores = HashSet::new();
    for id in ids {
        let pkg: Option<i32> = read_first(&format!(
            "/sys/devices/system/cpu/cpu{id}/topology/physical_package_id"
        ));
        let core: Option<i32> =
            read_first(&format!("/sys/devices/system/cpu/cpu{id}/topology/core_id"));
        if let (Some(p), Some(c)) = (pkg, core) {
            cores.insert((p, c));
        }
    }
    if cores.is_empty() {
        ids.len()
    } else {
        cores.len()
    }
}

/// CPU quota for this process's own cgroup v2 scope, in CPUs, if one is set.
///
/// `/sys/fs/cgroup/cpu.max` is the root cgroup and normally does not exist or
/// says `max`, which is why the first version of this check missed the 6-CPU
/// limit on the bench host. Read the process's own path from
/// `/proc/self/cgroup` (the v2 line is `0::/some/path`), then walk from that
/// path up to the mount root and keep the smallest quota found on the way:
/// a quota may be set on any ancestor slice.
fn cgroup_cpus() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let path = text.lines().find_map(|l| l.strip_prefix("0::"))?.trim();
    let mut parts: Vec<&str> = path.split('/').filter(|p| !p.is_empty()).collect();
    let mut best: Option<usize> = None;
    loop {
        let dir = if parts.is_empty() {
            "/sys/fs/cgroup".to_string()
        } else {
            format!("/sys/fs/cgroup/{}", parts.join("/"))
        };
        if let Some(cpus) = read_cpu_max(&format!("{dir}/cpu.max")) {
            best = Some(best.map_or(cpus, |b| b.min(cpus)));
        }
        if parts.pop().is_none() {
            break;
        }
    }
    best
}

/// One `cpu.max` file: `<quota> <period>`, or `max <period>`.
fn read_cpu_max(path: &str) -> Option<usize> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut it = text.split_whitespace();
    let quota = it.next()?;
    let period: u64 = it.next()?.parse().ok()?;
    if quota == "max" || period == 0 {
        return None;
    }
    let quota: u64 = quota.parse().ok()?;
    Some(quota.div_ceil(period).max(1) as usize)
}

fn machine_info() -> Value {
    let ids = affinity_ids();
    let logical = if ids.is_empty() {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    } else {
        ids.len()
    };
    let physical = physical_cores(&ids).min(logical);
    let available = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let quota = cgroup_cpus();
    // The CPUs the process can really use: the affinity mask, the cgroup quota
    // of its own scope, and what the standard library reports, whichever is
    // smallest.
    let usable = logical.min(quota.unwrap_or(logical)).min(available).max(1);
    let cpu = std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("model name"))
                .map(|l| l.split(':').nth(1).unwrap_or("").trim().to_string())
        })
        .unwrap_or_default();
    json!({
        "cpu": cpu,
        "logical_cpus": logical,
        "physical_cores": physical,
        "usable_cpus": usable,
        "affinity_ids": ids,
        "cgroup_cpus": quota,
        "available_parallelism": available,
    })
}

/// Provider string plus a temp config file that disables ORT intra-op spinning.
fn provider_string(spin: i32) -> (String, Option<String>) {
    if spin < 0 {
        return ("cpu".to_string(), None);
    }
    let path = format!("/tmp/tinta-bench-provider-{}.conf", std::process::id());
    let body = format!(
        "SessionConfig.session.intra_op.allow_spinning={}\n",
        spin.min(1)
    );
    if std::fs::write(&path, body).is_ok() {
        (format!("cpu:{path}"), Some(path))
    } else {
        ("cpu".to_string(), None)
    }
}

struct AsrModels {
    encoder: String,
    decoder: String,
    joiner: String,
    tokens: String,
}

#[allow(clippy::field_reassign_with_default)]
fn build_recognizer(m: &AsrModels, threads: i32, provider: &str) -> Option<OfflineRecognizer> {
    let mut config = OfflineRecognizerConfig::default();
    config.model_config.transducer = OfflineTransducerModelConfig {
        encoder: Some(m.encoder.clone()),
        decoder: Some(m.decoder.clone()),
        joiner: Some(m.joiner.clone()),
    };
    config.model_config.tokens = Some(m.tokens.clone());
    config.model_config.model_type = Some("nemo_transducer".to_string());
    config.model_config.provider = Some(provider.to_string());
    config.model_config.num_threads = threads;
    OfflineRecognizer::create(&config)
}

#[allow(clippy::too_many_arguments, clippy::field_reassign_with_default)]
fn build_diarizer(
    seg_model: &str,
    embedding: &str,
    threads: i32,
    threshold: f32,
    num_speakers: i32,
    min_on: f32,
    min_off: f32,
    window_shift: f32,
    provider: &str,
) -> Option<OfflineSpeakerDiarization> {
    let mut config = OfflineSpeakerDiarizationConfig::default();
    config.segmentation = OfflineSpeakerSegmentationModelConfig {
        pyannote: OfflineSpeakerSegmentationPyannoteModelConfig {
            model: Some(seg_model.to_string()),
            window_shift_ratio: window_shift,
        },
        num_threads: threads,
        debug: false,
        provider: Some(provider.to_string()),
    };
    config.embedding = SpeakerEmbeddingExtractorConfig {
        model: Some(embedding.to_string()),
        num_threads: threads,
        debug: false,
        provider: Some(provider.to_string()),
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

fn run_asr_call(
    rec: &OfflineRecognizer,
    samples: &[f32],
    offset_s: f32,
) -> (String, Vec<String>, Vec<f32>, Option<Vec<f32>>) {
    let seconds = samples.len() as f32 / SAMPLE_RATE as f32;
    assert!(
        seconds <= MAX_ASR_SECONDS + 0.001,
        "ASR call of {seconds:.1} s exceeds the hard cap of {MAX_ASR_SECONDS} s"
    );
    let stream = rec.create_stream();
    stream.accept_waveform(SAMPLE_RATE as i32, samples);
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

/// Silero VAD speech segments in sample indices, absolute in the input.
#[allow(clippy::field_reassign_with_default)]
fn vad_segments(
    samples: &[f32],
    model: &str,
    threads: i32,
    provider: &str,
    threshold: f32,
) -> (Vec<(usize, usize)>, f64, Value) {
    let mut cfg = VadModelConfig::default();
    cfg.silero_vad = SileroVadModelConfig {
        model: Some(model.to_string()),
        threshold,
        min_silence_duration: 0.3,
        min_speech_duration: 0.1,
        window_size: 512,
        max_speech_duration: VAD_PIECE_SECONDS,
    };
    cfg.sample_rate = SAMPLE_RATE as i32;
    cfg.num_threads = threads;
    cfg.provider = Some(provider.to_string());
    cfg.debug = false;

    let t = Instant::now();
    let vad = VoiceActivityDetector::create(&cfg, 60.0).expect("create VAD");
    let load_ms = ms_since(t);
    let mut segs: Vec<(usize, usize)> = Vec::new();
    let flush_n = (VAD_FLUSH_SECONDS * SAMPLE_RATE as f32) as usize;
    let mut since_flush = 0usize;
    for block in samples.chunks(SAMPLE_RATE) {
        vad.accept_waveform(block);
        since_flush += block.len();
        while let Some(s) = vad.front() {
            let start = s.start().max(0) as usize;
            let n = s.n().max(0) as usize;
            if n > 0 {
                segs.push((start, (start + n).min(samples.len())));
            }
            vad.pop();
        }
        // A long continuous speech run would otherwise keep growing the VAD's
        // circular buffer. Close it every 20 s; the pre-roll on the next piece
        // keeps the cut harmless.
        if vad.detected() && since_flush >= flush_n {
            vad.flush();
            since_flush = 0;
            while let Some(s) = vad.front() {
                let start = s.start().max(0) as usize;
                let n = s.n().max(0) as usize;
                if n > 0 {
                    segs.push((start, (start + n).min(samples.len())));
                }
                vad.pop();
            }
        } else if !vad.detected() {
            since_flush = 0;
        }
    }
    vad.flush();
    while let Some(s) = vad.front() {
        let start = s.start().max(0) as usize;
        let n = s.n().max(0) as usize;
        if n > 0 {
            segs.push((start, (start + n).min(samples.len())));
        }
        vad.pop();
    }
    let speech_s: f64 = segs.iter().map(|(a, b)| (b - a) as f64 / SAMPLE_RATE as f64).sum();
    let seg_count = segs.len();
    let ranges: Vec<Value> = segs
        .iter()
        .map(|(a, b)| json!([*a as f64 / SAMPLE_RATE as f64, *b as f64 / SAMPLE_RATE as f64]))
        .collect();
    (
        segs,
        load_ms,
        json!({
            "model": model,
            "load_ms": load_ms,
            "segments": seg_count,
            "speech_seconds": speech_s,
            "max_speech_duration": VAD_PIECE_SECONDS,
            "min_silence_duration": 0.3,
            "threshold": threshold,
            "ranges": ranges,
        }),
    )
}

/// Quietest point in [lo, hi), used to split a long speech run at low energy
/// instead of at an arbitrary 30 s boundary.
fn quietest_point(samples: &[f32], lo: usize, hi: usize) -> usize {
    let win = (VAD_SPLIT_WINDOW_SECONDS * SAMPLE_RATE as f32) as usize;
    if hi <= lo + win {
        return (lo + hi) / 2;
    }
    let mut best = lo + win / 2;
    let mut best_energy = f32::INFINITY;
    let mut pos = lo;
    while pos + win <= hi {
        let energy: f32 = samples[pos..pos + win].iter().map(|x| x * x).sum();
        if energy < best_energy {
            best_energy = energy;
            best = pos + win / 2;
        }
        pos += win;
    }
    best
}

/// Split a speech run into pieces of at most `max_piece` seconds, cutting at
/// the quietest point in the second half of each piece.
fn split_run(samples: &[f32], a: usize, b: usize, max_piece: f32) -> Vec<(usize, usize)> {
    let max_n = (max_piece * SAMPLE_RATE as f32) as usize;
    let mut runs = Vec::new();
    let mut start = a;
    while b.saturating_sub(start) > max_n {
        let hi = start + max_n;
        let lo = start + (max_n as f32 * 0.6) as usize;
        let split = quietest_point(samples, lo, hi);
        runs.push((start, split));
        start = split;
    }
    if b > start {
        runs.push((start, b));
    }
    runs
}

/// Group VAD segments into decode pieces of at most `max_piece` seconds.
/// Every piece gets a pre-roll because Parakeet can return zero words for an
/// input that starts mid-speech; overlapping edges are de-duplicated at the
/// midpoint after decoding.
fn group_vad_pieces(
    samples: &[f32],
    segs: &[(usize, usize)],
    audio_len: usize,
    max_piece: f32,
) -> Vec<(usize, usize)> {
    let max_piece_n = (max_piece * SAMPLE_RATE as f32) as usize;
    let join_gap = (VAD_JOIN_GAP_SECONDS * SAMPLE_RATE as f32) as usize;
    let pre = (VAD_PRE_ROLL_SECONDS * SAMPLE_RATE as f32) as usize;
    let post = (VAD_POST_ROLL_SECONDS * SAMPLE_RATE as f32) as usize;
    let min_piece = (VAD_MIN_PIECE_SECONDS * SAMPLE_RATE as f32) as usize;

    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (a, b) in segs {
        runs.extend(split_run(samples, *a, *b, max_piece));
    }

    let mut groups: Vec<(usize, usize)> = Vec::new();
    for (a, b) in runs {
        if let Some(last) = groups.last_mut() {
            let span = b.saturating_sub(last.0);
            let gap = a.saturating_sub(last.1);
            if span <= max_piece_n && gap <= join_gap {
                last.1 = b;
                continue;
            }
        }
        groups.push((a, b));
    }

    let mut pieces: Vec<(usize, usize)> = Vec::new();
    for (i, (a, b)) in groups.iter().enumerate() {
        let mut start = a.saturating_sub(pre);
        if i == 0 {
            start = 0;
        }
        let end = (b + post).max(start + min_piece).min(audio_len);
        if end <= start {
            continue;
        }
        let cap = (MAX_ASR_SECONDS * SAMPLE_RATE as f32) as usize;
        let end = end.min(start + cap);
        pieces.push((start, end));
    }
    pieces
}

fn fixed_chunks(audio_len: usize, chunk_s: f32, overlap: f32) -> Vec<(usize, usize)> {
    let chunk = (chunk_s * SAMPLE_RATE as f32) as usize;
    let step = ((chunk_s - overlap).max(0.5) * SAMPLE_RATE as f32) as usize;
    let mut pieces = Vec::new();
    let mut pos = 0usize;
    while pos < audio_len {
        let end = (pos + chunk).min(audio_len);
        pieces.push((pos, end));
        if end == audio_len {
            break;
        }
        pos += step;
    }
    pieces
}

struct AsrStage<'a> {
    samples: &'a [f32],
    audio_s: f64,
    models: &'a AsrModels,
    threads: i32,
    provider: &'a str,
    vad_model: Option<&'a str>,
    vad_threshold: f32,
    chunk_s: f32,
    overlap: f32,
}

fn run_asr_stage(cfg: &AsrStage) -> Value {
    let stage_t = Instant::now();
    let cpu_start = cpu_seconds();

    let mut vad_json = Value::Null;
    let mut pieces;
    let mut vad_seconds = 0.0;
    if let Some(vad_model) = cfg.vad_model {
        let t = Instant::now();
        let (segs, _load_ms, info) = vad_segments(cfg.samples, vad_model, 1, cfg.provider, cfg.vad_threshold);
        pieces = group_vad_pieces(cfg.samples, &segs, cfg.samples.len(), VAD_PIECE_SECONDS);
        vad_seconds = t.elapsed().as_secs_f64();
        vad_json = info;
        vad_json["decode_seconds"] = json!(vad_seconds);
        vad_json["pieces"] = json!(pieces.len());
        if pieces.is_empty() {
            pieces = fixed_chunks(cfg.samples.len(), VAD_PIECE_SECONDS, 0.0);
            vad_json["fallback_fixed_chunks"] = json!(true);
        }
    } else {
        pieces = fixed_chunks(cfg.samples.len(), cfg.chunk_s, cfg.overlap);
    }
    let cpu_after_vad = cpu_seconds();

    evict(&[
        &cfg.models.encoder,
        &cfg.models.decoder,
        &cfg.models.joiner,
        &cfg.models.tokens,
    ]);
    let t = Instant::now();
    let rec = build_recognizer(cfg.models, cfg.threads, cfg.provider).expect("create recognizer");
    let load_cold = ms_since(t);

    let t = Instant::now();
    let rec2 = build_recognizer(cfg.models, cfg.threads, cfg.provider)
        .expect("create recognizer (warm)");
    let load_warm = ms_since(t);
    drop(rec);
    drop(rec2);

    let rec = build_recognizer(cfg.models, cfg.threads, cfg.provider).expect("create recognizer");
    let cpu_after_build = cpu_seconds();

    let t = Instant::now();
    let mut all_words: Vec<Word> = Vec::new();
    let mut chunks_out: Vec<(f32, f32, Vec<Word>)> = Vec::new();
    for (a, b) in &pieces {
        let seconds = (*b - *a) as f32 / SAMPLE_RATE as f32;
        assert!(
            seconds <= MAX_ASR_SECONDS + 0.001,
            "piece of {seconds:.1} s exceeds the {MAX_ASR_SECONDS} s cap"
        );
        let off = *a as f32 / SAMPLE_RATE as f32;
        let (_text, tk, tsv, dur) = run_asr_call(&rec, &cfg.samples[*a..*b], off);
        let words = tokens_to_words(&tk, &tsv, dur.as_deref());
        chunks_out.push((off, *b as f32 / SAMPLE_RATE as f32, words));
    }
    // Cut every overlap at its midpoint on both sides, then sort by time.
    for i in 1..chunks_out.len() {
        let boundary = if cfg.vad_model.is_some() {
            (chunks_out[i - 1].1 + chunks_out[i].0) / 2.0
        } else {
            chunks_out[i].0 + cfg.overlap / 2.0
        };
        chunks_out[i - 1].2.retain(|w| w.start < boundary);
        chunks_out[i].2.retain(|w| w.start >= boundary);
    }
    for (_, _, words) in chunks_out {
        all_words.extend(words);
    }
    all_words.sort_by(|a, b| {
        a.start
            .partial_cmp(&b.start)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    all_words.retain(|w| !w.text.trim().is_empty());
    let decode_seconds = t.elapsed().as_secs_f64();
    let cpu_end = cpu_seconds();

    let text = all_words
        .iter()
        .map(|w| w.text.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let max_piece = pieces
        .iter()
        .map(|(a, b)| (*b - *a) as f64 / SAMPLE_RATE as f64)
        .fold(0.0f64, f64::max);
    let stage_seconds = stage_t.elapsed().as_secs_f64();
    json!({
        "load_ms_cold": load_cold,
        "load_ms_warm": load_warm,
        "vad": vad_json,
        "vad_seconds": vad_seconds,
        "vad_cpu_seconds": cpu_after_vad - cpu_start,
        "pieces": pieces.len(),
        "max_piece_seconds": max_piece,
        "decode_seconds": decode_seconds,
        "stage_seconds": stage_seconds,
        "cpu_seconds": cpu_end - cpu_after_build,
        "total_cpu_seconds": cpu_end - cpu_start,
        "rtf": decode_seconds / cfg.audio_s,
        "chunk_seconds": if cfg.vad_model.is_some() { 0.0 } else { cfg.chunk_s },
        "chunk_overlap": cfg.overlap,
        "text": text,
        "word_count": all_words.len(),
        "has_word_timestamps": all_words.iter().all(|w| w.end >= w.start),
        "words": all_words.iter().map(|w| json!({"start": w.start, "end": w.end, "text": w.text})).collect::<Vec<_>>(),
    })
}

struct DiarStage<'a> {
    samples: &'a [f32],
    audio_s: f64,
    seg_model: &'a str,
    embedding: &'a str,
    threads: i32,
    threshold: f32,
    num_speakers: i32,
    min_on: f32,
    min_off: f32,
    window_shift: f32,
    provider: &'a str,
}

fn run_diar_stage(cfg: &DiarStage) -> Value {
    let stage_t = Instant::now();
    let cpu_start = cpu_seconds();
    evict(&[cfg.seg_model, cfg.embedding]);
    let t = Instant::now();
    let d = build_diarizer(
        cfg.seg_model,
        cfg.embedding,
        cfg.threads,
        cfg.threshold,
        cfg.num_speakers,
        cfg.min_on,
        cfg.min_off,
        cfg.window_shift,
        cfg.provider,
    )
    .expect("create diarizer");
    let load_cold = ms_since(t);
    let t = Instant::now();
    let d2 = build_diarizer(
        cfg.seg_model,
        cfg.embedding,
        cfg.threads,
        cfg.threshold,
        cfg.num_speakers,
        cfg.min_on,
        cfg.min_off,
        cfg.window_shift,
        cfg.provider,
    )
    .expect("create diarizer (warm)");
    let load_warm = ms_since(t);
    drop(d2);
    let cpu_after_build = cpu_seconds();

    let t = Instant::now();
    let result = d.process(cfg.samples).expect("diarize");
    let process_seconds = t.elapsed().as_secs_f64();
    let cpu_end = cpu_seconds();
    let segs = result.sort_by_start_time();
    let segments: Vec<(f32, f32, i32)> =
        segs.iter().map(|s| (s.start, s.end, s.speaker)).collect();
    json!({
        "seg_model": cfg.seg_model,
        "embedding_model": cfg.embedding,
        "threshold": cfg.threshold,
        "window_shift": cfg.window_shift,
        "min_duration_on": cfg.min_on,
        "min_duration_off": cfg.min_off,
        "load_ms_cold": load_cold,
        "load_ms_warm": load_warm,
        "process_seconds": process_seconds,
        "stage_seconds": stage_t.elapsed().as_secs_f64(),
        "cpu_seconds": cpu_end - cpu_after_build,
        "total_cpu_seconds": cpu_end - cpu_start,
        "rtf": process_seconds / cfg.audio_s,
        "num_speakers": result.num_speakers(),
        "num_segments": result.num_segments(),
        "segments": segments.iter().map(|(s, e, sp)| json!({"start": s, "end": e, "speaker": sp})).collect::<Vec<_>>(),
    })
}

/// Join the front end's segments into embedding units: consecutive segments of
/// the same cluster no further apart than `gap` seconds and no longer than
/// `max_len` together. A whole unit is the smallest thing the post-clustering
/// relabels, so very short fragments never get an embedding of their own.
fn build_units(segments: &[(f32, f32, i32)], gap: f32, max_len: f32) -> Vec<(f32, f32, usize)> {
    let mut out: Vec<(f32, f32, usize)> = Vec::new();
    for &(start, end, speaker) in segments {
        let cluster = speaker.max(0) as usize;
        if end <= start {
            continue;
        }
        if let Some(last) = out.last_mut() {
            if last.2 == cluster
                && start - last.1 <= gap
                && end.max(last.1) - last.0 <= max_len
            {
                last.1 = last.1.max(end);
                continue;
            }
        }
        out.push((start, end, cluster));
    }
    out
}

/// Embeddings for every unit, cached on disk so a threshold sweep does not pay
/// for the embedding model twice. Returns (embeddings, seconds spent, reused).
fn unit_embeddings(
    samples: &[f32],
    units: &[(f32, f32, usize)],
    model: &str,
    threads: i32,
    provider: &str,
    min_unit: f32,
    cache: Option<&str>,
) -> (Vec<Vec<f32>>, f64, bool) {
    if let Some(path) = cache {
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Ok(v) = serde_json::from_str::<Value>(&text) {
                let same_model = v["model"].as_str() == Some(model);
                let cached: Vec<(f32, f32, usize)> = v["units"]
                    .as_array()
                    .map(|a| {
                        a.iter()
                            .map(|u| {
                                (
                                    u["start"].as_f64().unwrap_or(-1.0) as f32,
                                    u["end"].as_f64().unwrap_or(-1.0) as f32,
                                    u["cluster"].as_u64().unwrap_or(0) as usize,
                                )
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                if same_model && cached == units {
                    let emb: Vec<Vec<f32>> = v["embeddings"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .map(|e| {
                                    e.as_array()
                                        .map(|x| {
                                            x.iter()
                                                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                                                .collect()
                                        })
                                        .unwrap_or_default()
                                })
                                .collect()
                        })
                        .unwrap_or_default();
                    if emb.len() == units.len() && !emb.is_empty() {
                        return (emb, 0.0, true);
                    }
                }
            }
        }
    }

    let config = SpeakerEmbeddingExtractorConfig {
        model: Some(model.to_string()),
        num_threads: threads,
        provider: Some(provider.to_string()),
        ..Default::default()
    };
    let extractor = SpeakerEmbeddingExtractor::create(&config).expect("create embedding extractor");

    let t = Instant::now();
    let min_n = (min_unit * SAMPLE_RATE as f32) as usize;
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(units.len());
    for &(start, end, _) in units {
        let mut a = (start.max(0.0) * SAMPLE_RATE as f32) as usize;
        let mut b = (end.max(0.0) * SAMPLE_RATE as f32) as usize;
        if b > samples.len() {
            b = samples.len();
        }
        // Very short units are padded symmetrically around the segment. An
        // embedding needs a few hundred ms of audio to carry any voice at all.
        if b > a && b - a < min_n {
            let need = min_n - (b - a);
            a = a.saturating_sub(need / 2);
            b = (b + (need - need / 2)).min(samples.len());
        }
        if b <= a {
            out.push(vec![0.0; extractor.dim() as usize]);
            continue;
        }
        let stream = extractor.create_stream().expect("embedding stream");
        stream.accept_waveform(SAMPLE_RATE as i32, &samples[a..b]);
        let emb = if extractor.is_ready(&stream) {
            extractor.compute(&stream).unwrap_or_default()
        } else {
            Vec::new()
        };
        if emb.is_empty() {
            out.push(vec![0.0; extractor.dim() as usize]);
        } else {
            out.push(emb);
        }
    }
    let seconds = t.elapsed().as_secs_f64();

    if let Some(path) = cache {
        let cache_json = json!({
            "model": model,
            "dim": extractor.dim(),
            "units": units.iter().map(|(s, e, c)| json!({"start": s, "end": e, "cluster": c})).collect::<Vec<_>>(),
            "embeddings": out,
        });
        if let Some(dir) = Path::new(path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(mut f) = File::create(path) {
            let _ = f.write_all(serde_json::to_string(&cache_json).unwrap().as_bytes());
        }
    }
    (out, seconds, false)
}

/// Post-cluster an already over-clustered diarization result.
#[allow(clippy::too_many_arguments)]
fn run_postcluster(
    diar_path: &str,
    embedding: &str,
    samples: &[f32],
    audio_s: f64,
    threads: i32,
    provider: &str,
    opts: &ClusterOptions,
    unit_gap: f32,
    unit_max: f32,
    min_unit: f32,
    cache: Option<&str>,
) -> Value {
    let stage_t = Instant::now();
    let raw: Value =
        serde_json::from_str(&std::fs::read_to_string(diar_path).expect("read diarization json"))
            .expect("parse diarization json");
    let diar = if raw.get("diarization").is_some() {
        &raw["diarization"]
    } else {
        &raw
    };
    let mut segments: Vec<(f32, f32, i32)> = diar["segments"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|s| {
                    (
                        s["start"].as_f64().unwrap_or(0.0) as f32,
                        s["end"].as_f64().unwrap_or(0.0) as f32,
                        s["speaker"].as_i64().unwrap_or(0) as i32,
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    segments.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let units = build_units(&segments, unit_gap, unit_max);
    let (embeddings, embedding_seconds, reused) =
        unit_embeddings(samples, &units, embedding, threads, provider, min_unit, cache);
    let clustered: Vec<Unit> = units
        .iter()
        .zip(embeddings)
        .map(|(&(start, end, cluster), embedding)| Unit {
            start,
            end,
            cluster,
            embedding,
        })
        .collect();

    let outcome = recluster(&clustered, opts);
    let shares = speech_shares(&clustered, &outcome.labels);
    let total: f32 = shares.iter().map(|(_, s)| *s).sum();
    let out_segments: Vec<Value> = clustered
        .iter()
        .zip(&outcome.labels)
        .map(|(u, &label)| json!({"start": u.start, "end": u.end, "speaker": label}))
        .collect();

    json!({
        "diar_source": diar_path,
        "embedding_model": embedding,
        "merge_distance": opts.merge_distance,
        "min_share": opts.min_share,
        "unit_gap": unit_gap,
        "unit_max": unit_max,
        "min_unit": min_unit,
        "units": units.len(),
        "clusters_before": outcome.clusters_before,
        "num_speakers": outcome.clusters_after,
        "merges": outcome.merges,
        "dissolved": outcome.dissolved,
        "reassigned": outcome.reassigned,
        "failed_units": outcome.failed_units,
        "shares": shares.iter().map(|(l, s)| json!({"speaker": l, "seconds": s, "share": if total > 0.0 { s / total } else { 0.0 }})).collect::<Vec<_>>(),
        "embedding_seconds": embedding_seconds,
        "embeddings_reused": reused,
        "stage_seconds": stage_t.elapsed().as_secs_f64(),
        "audio_seconds": audio_s,
        "num_segments": out_segments.len(),
        "segments": out_segments,
    })
}

fn combine_turns(asr: &Value, diar_segments: &[(f32, f32, i32)]) -> Vec<Value> {
    let words: Vec<Value> = asr["words"].as_array().cloned().unwrap_or_default();
    let mut turns: Vec<Value> = Vec::new();
    for w in &words {
        let ws = w["start"].as_f64().unwrap_or(0.0) as f32;
        let we = w["end"].as_f64().unwrap_or(0.0) as f32;
        let mut best = (-1, 0.0f32);
        for (s, e, sp) in diar_segments {
            let ov = (we.min(*e) - ws.max(*s)).max(0.0);
            if ov > best.1 {
                best = (*sp, ov);
            }
        }
        let spk = if best.0 < 0 {
            let mid = (ws + we) / 2.0;
            diar_segments
                .iter()
                .min_by(|a, b| {
                    let da = if mid < a.0 {
                        a.0 - mid
                    } else if mid > a.1 {
                        mid - a.1
                    } else {
                        0.0
                    };
                    let db = if mid < b.0 {
                        b.0 - mid
                    } else if mid > b.1 {
                        mid - b.1
                    } else {
                        0.0
                    };
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
    turns
}

fn main() {
    let args = Args::parse();
    let wav_path = args.req("wav");
    let out_path = args.req("output");
    let machine = machine_info();
    let logical = machine["logical_cpus"].as_u64().unwrap_or(1) as i32;
    let physical = (machine["physical_cores"].as_u64().unwrap_or(logical as u64) as i32).min(logical);
    let usable = (machine["usable_cpus"]
        .as_u64()
        .unwrap_or(physical as u64) as i32)
        .clamp(1, physical);

    // Defaults scale with the CPUs the process actually gets: ASR gets half of
    // them, capped at 4; diarization gets 2-4.
    let default_asr_threads = (usable / 2).clamp(1, 4);
    let default_diar_threads = (usable / 2).clamp(2, 4);
    let asr_threads = args.i32_or("threads", default_asr_threads);
    let diar_threads = args.i32_or("diar-threads", default_diar_threads);
    let spin = args.i32_or("spin", -1);

    let wave = Wave::read(&wav_path).expect("read wav");
    assert_eq!(wave.sample_rate(), 16000, "expected 16 kHz mono wav");
    let samples_owned = wave.samples().to_vec();
    let samples = samples_owned.as_slice();
    let audio_s = samples.len() as f64 / SAMPLE_RATE as f64;

    let (provider, provider_conf) = provider_string(spin);
    let process_start = Instant::now();
    if args.cmd == "vad" {
        let model = args.req("vad-model");
        let (segs, _load, info) = vad_segments(samples, &model, asr_threads, &provider, args.f32_or("vad-threshold", 0.5));
        let pieces = group_vad_pieces(samples, &segs, samples.len(), VAD_PIECE_SECONDS);
        let out = json!({
            "command": "vad",
            "wav": wav_path,
            "audio_seconds": audio_s,
            "threads": asr_threads,
            "loadavg_start": loadavg(),
            "vad": info,
            "pieces": pieces.iter().map(|(a, b)| json!([*a as f64 / SAMPLE_RATE as f64, *b as f64 / SAMPLE_RATE as f64])).collect::<Vec<_>>(),
            "max_piece_seconds": pieces.iter().map(|(a, b)| (*b - *a) as f64 / SAMPLE_RATE as f64).fold(0.0f64, f64::max),
            "peak_rss_mb": peak_rss_mb(),
            "loadavg_end": loadavg(),
        });
        if let Some(dir) = Path::new(&out_path).parent() { let _ = std::fs::create_dir_all(dir); }
        let mut f = File::create(&out_path).expect("create output");
        f.write_all(serde_json::to_string_pretty(&out).unwrap().as_bytes()).expect("write output");
        if let Some(path) = provider_conf { let _ = std::fs::remove_file(path); }
        return;
    }
    let cpu_start = cpu_seconds();

    if args.cmd == "postcluster" {
        let opts = ClusterOptions {
            merge_distance: args.f32_or("merge-threshold", 0.15),
            min_share: args.f32_or("min-share", 0.03),
        };
        let block = run_postcluster(
            &args.req("diar"),
            &args.req("embedding"),
            samples,
            audio_s,
            asr_threads,
            &provider,
            &opts,
            args.f32_or("unit-gap", 0.5),
            args.f32_or("unit-max", 15.0),
            args.f32_or("min-unit", 0.8),
            args.get("embeddings"),
        );
        let out = json!({
            "command": "postcluster",
            "wav": wav_path,
            "audio_seconds": audio_s,
            "threads": asr_threads,
            "spin": if spin < 0 { "default".to_string() } else { spin.to_string() },
            "loadavg_start": loadavg(),
            "machine": machine,
            "postcluster": {
                "diar_source": block["diar_source"],
                "clusters_before": block["clusters_before"],
                "num_speakers": block["num_speakers"],
                "merges": block["merges"],
                "dissolved": block["dissolved"],
                "reassigned": block["reassigned"],
                "failed_units": block["failed_units"],
                "units": block["units"],
                "shares": block["shares"],
                "merge_distance": block["merge_distance"],
                "min_share": block["min_share"],
                "embedding_seconds": block["embedding_seconds"],
                "embeddings_reused": block["embeddings_reused"],
                "stage_seconds": block["stage_seconds"],
            },
            "diarization": {
                "segments": block["segments"],
                "num_speakers": block["num_speakers"],
                "num_segments": block["num_segments"],
                "threshold": serde_json::Value::Null,
                "merge_distance": block["merge_distance"],
                "min_share": block["min_share"],
                "failed_units": block["failed_units"],
                "clusters_before": block["clusters_before"],
                "process_seconds": block["stage_seconds"],
                "rtf": block["stage_seconds"].as_f64().unwrap_or(0.0) / audio_s,
                "cpu_seconds": cpu_seconds() - cpu_start,
            },
            "peak_rss_mb": peak_rss_mb(),
            "loadavg_end": loadavg(),
        });
        if let Some(dir) = Path::new(&out_path).parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let mut f = File::create(&out_path).expect("create output");
        f.write_all(serde_json::to_string_pretty(&out).unwrap().as_bytes())
            .expect("write output");
        if let Some(path) = provider_conf {
            let _ = std::fs::remove_file(path);
        }
        return;
    }

    let mut out = json!({
        "command": args.cmd,
        "wav": wav_path,
        "audio_seconds": audio_s,
        "threads": asr_threads,
        "diar_threads": diar_threads,
        "thread_defaults": {
            "asr": default_asr_threads,
            "diarization": default_diar_threads,
            "logical_cpus": logical,
            "physical_cores": physical,
            "usable_cpus": usable,
        },
        "spin": if spin < 0 { "default".to_string() } else { spin.to_string() },
        "provider": provider,
        "loadavg_start": loadavg(),
        "machine": machine,
    });

    let run_asr = args.cmd == "asr" || args.cmd == "combined";
    let run_diar = args.cmd == "diarize" || args.cmd == "combined";
    let vad_model = args.get("vad-model").map(|s| s.to_string());
    let chunk_s = args.f32_or("chunk-seconds", 0.0);

    let asr_models = if run_asr {
        AsrModels {
            encoder: args.req("encoder"),
            decoder: args.req("decoder"),
            joiner: args.req("joiner"),
            tokens: args.req("tokens"),
        }
    } else {
        AsrModels {
            encoder: String::new(),
            decoder: String::new(),
            joiner: String::new(),
            tokens: String::new(),
        }
    };
    let (seg_model, embedding) = if run_diar {
        (args.req("seg-model"), args.req("embedding"))
    } else {
        (String::new(), String::new())
    };
    let threshold = args.f32_or("threshold", 0.5);
    let num_speakers = args.i32_or("num-speakers", -1);
    let min_on = args.f32_or("min-duration-on", 0.3);
    let min_off = args.f32_or("min-duration-off", 0.5);
    let window_shift = args.f32_or("window-shift", 0.2);

    let asr_cfg = AsrStage {
        samples,
        audio_s,
        models: &asr_models,
        threads: asr_threads,
        provider: &provider,
        vad_model: vad_model.as_deref(),
        vad_threshold: args.f32_or("vad-threshold", 0.5),
        chunk_s,
        overlap: args.f32_or("chunk-overlap", 2.0),
    };
    let diar_cfg = DiarStage {
        samples,
        audio_s,
        seg_model: &seg_model,
        embedding: &embedding,
        threads: diar_threads,
        threshold,
        num_speakers,
        min_on,
        min_off,
        window_shift,
        provider: &provider,
    };

    let mut asr_out = Value::Null;
    let mut diar_out = Value::Null;
    if args.cmd == "combined" {
        let (a, d) = std::thread::scope(|scope| {
            let a = scope.spawn(|| run_asr_stage(&asr_cfg));
            let d = scope.spawn(|| run_diar_stage(&diar_cfg));
            (
                a.join().expect("asr thread"),
                d.join().expect("diarization thread"),
            )
        });
        let mut a = a;
        let mut d = d;
        // RUSAGE_SELF is process-wide: the per-stage CPU split is not
        // attributable when both stages run at once. Use the standalone runs
        // for per-stage CPU; report only the process total here.
        a["cpu_seconds"] = serde_json::Value::Null;
        a["total_cpu_seconds"] = serde_json::Value::Null;
        d["cpu_seconds"] = serde_json::Value::Null;
        d["total_cpu_seconds"] = serde_json::Value::Null;
        let asr_wall = a["stage_seconds"].as_f64().unwrap_or(0.0);
        let diar_wall = d["stage_seconds"].as_f64().unwrap_or(0.0);
        let turns = {
            let segs: Vec<(f32, f32, i32)> = d["segments"]
                .as_array()
                .map(|v| {
                    v.iter()
                        .map(|s| {
                            (
                                s["start"].as_f64().unwrap_or(0.0) as f32,
                                s["end"].as_f64().unwrap_or(0.0) as f32,
                                s["speaker"].as_i64().unwrap_or(-1) as i32,
                            )
                        })
                        .collect()
                })
                .unwrap_or_default();
            combine_turns(&a, &segs)
        };
        let total = process_start.elapsed().as_secs_f64();
        out["combined"] = json!({
            "asr_seconds": asr_wall,
            "diarization_seconds": diar_wall,
            "critical_path_seconds": asr_wall.max(diar_wall),
            "total_seconds": total,
            "rtf": total / audio_s,
            "turns": turns,
        });
        asr_out = a;
        diar_out = d;
    } else {
        if run_asr {
            asr_out = run_asr_stage(&asr_cfg);
        }
        if run_diar {
            diar_out = run_diar_stage(&diar_cfg);
        }
    }
    if run_asr {
        out["asr"] = asr_out;
    }
    if run_diar {
        out["diarization"] = diar_out;
    }

    out["peak_rss_mb"] = json!(peak_rss_mb());
    out["process_wall_seconds"] = json!(process_start.elapsed().as_secs_f64());
    out["process_cpu_seconds"] = json!(cpu_seconds() - cpu_start);
    out["loadavg_end"] = json!(loadavg());

    if let Some(dir) = Path::new(&out_path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut f = File::create(&out_path).expect("create output");
    f.write_all(serde_json::to_string_pretty(&out).unwrap().as_bytes())
        .expect("write output");
    if let Some(path) = provider_conf {
        let _ = std::fs::remove_file(path);
    }
}
