//! End-to-end final pass with the real sherpa-onnx models.
//!
//! Ignored by default: it needs the installed models and the reference clip,
//! both outside the repository. Run it with
//!
//! ```text
//! TINTA_ENGINE_MODELS_DIR=/path/to/models \
//! TINTA_ENGINE_TEST_AUDIO=/path/to/clip.wav \
//! cargo test --locked -p tinta-engine-rs --test finalize_models -- --ignored --nocapture
//! ```
//!
//! The clip must be 60-90 s of speech with at least two speakers (the pinned one
//! has three). It is imported
//! as the remote track, so the whole pipeline (decode, resample, TNTC write,
//! read, VAD segmentation, ASR, echo filter, diarization) runs.
//!
//! The pinned clip is ES2004b[120, 210) from the BUTSpeechFIT AMI setup; see
//! `REFERENCE_SPEAKERS` for its reference speakers. The test runs with no
//! `max_speakers` cap, so it checks the recipe itself, not the app's cap.

use std::io::Write;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tinta_engine_rs::finalize::finalize;
use tinta_engine_rs::import::import_audio;
use tinta_engine_rs::manifest::Manifest;
use tinta_engine_rs::protocol::Output;
use tinta_engine_rs::speech::SpeechPass;
use tinta_engine_rs::threads::Threads;

#[derive(Clone, Default)]
struct Sink(Arc<Mutex<Vec<u8>>>);

impl Write for Sink {
    fn write(&mut self, buffer: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buffer);
        Ok(buffer.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Sink {
    fn lines(&self) -> Vec<Value> {
        String::from_utf8(self.0.lock().unwrap().clone())
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
}

fn key() -> [u8; 32] {
    let mut key = [0u8; 32];
    for (index, byte) in key.iter_mut().enumerate() {
        *byte = index as u8;
    }
    key
}

fn required(name: &str) -> PathBuf {
    PathBuf::from(
        std::env::var(name)
            .unwrap_or_else(|_| panic!("{name} must point at the models or the clip")),
    )
}

/// Two threads, per the fork's CPU rule for model runs.
fn threads() -> Threads {
    Threads {
        available: 2,
        physical: 2,
        asr: 2,
        diarization: 2,
    }
}

fn check_track(result: &Value, track: &str) {
    let words = result["tracks"][track]["words"]
        .as_array()
        .unwrap_or_else(|| panic!("{track} has no words array"));
    assert!(!words.is_empty(), "{track} has no words");
    let mut previous = f64::NEG_INFINITY;
    for word in words {
        let start = word["s"].as_f64().unwrap();
        let end = word["e"].as_f64().unwrap();
        let text = word["w"].as_str().unwrap();
        assert!(end >= start, "{track}: {text} ends before it starts");
        assert!(start >= previous, "{track}: {text} is out of order");
        assert!(!text.trim().is_empty(), "{track}: an empty word");
        previous = start;
    }
}

/// Reference speaker count for the pinned clip, from the BUTSpeechFIT AMI
/// `only_words` RTTM of ES2004b (commit 2509d8933721023fab4def2618aabd5c28eb82e9).
///
/// The clip is `ES2004b.Mix-Headset.wav[120.0, 210.0)` (verified sample-exact:
/// Pearson correlation 1.0 with a 16 kHz read at that offset). In that window the
/// reference has four labelled speakers, but one (`MEO015`) holds 0.44 s:
///
/// | speaker | seconds |
/// |---|---|
/// | FEE013 | 49.7 |
/// | MEE014 | 8.2 |
/// | FEE016 | 4.8 |
/// | MEO015 | 0.4 (excluded, under 1 s) |
///
/// so the count to compare against is **3**. The recipe merges nothing on this
/// clip and returns 4 (MEE014 is split in two), which is within the +/-1 check.
/// The reference JSON lives outside the repository (task #18 evidence,
/// `results/ref-ES2004b.json`); no audio or reference data is committed.
const REFERENCE_SPEAKERS: usize = 3;

#[test]
#[ignore = "needs TINTA_ENGINE_MODELS_DIR and TINTA_ENGINE_TEST_AUDIO with the reference clip"]
fn imports_and_finalizes_a_three_speaker_clip() {
    let models = required("TINTA_ENGINE_MODELS_DIR");
    let clip = required("TINTA_ENGINE_TEST_AUDIO");
    assert!(
        clip.is_file(),
        "the clip does not exist: {}",
        clip.display()
    );
    assert!(
        models
            .join("parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx")
            .is_file(),
        "the models are not installed under {}",
        models.display()
    );

    let dir = tempfile::tempdir().unwrap();
    let key = key();
    let sink = Sink::default();
    let out = Output::new(sink.clone());

    let imported = import_audio(dir.path(), &key, &clip, &out).unwrap();
    let imported_seconds = imported["duration"].as_f64().unwrap();
    assert!(
        (30.0..=120.0).contains(&imported_seconds),
        "the clip should be 60-90 s, got {imported_seconds} s"
    );

    let mut pass = SpeechPass::new(Manifest::embedded(), models, threads());
    let result = finalize(&mut pass, dir.path(), &key, None, &out).unwrap();
    let duration = result["duration"].as_f64().unwrap();
    assert!(
        (duration - imported_seconds).abs() < 1.0,
        "duration {duration} does not match the import {imported_seconds}"
    );

    check_track(&result, "remote");
    // The clip was imported as the remote track only, so the microphone track
    // is empty and has no diarization key.
    assert_eq!(
        result["tracks"]["mic"]["words"].as_array().unwrap().len(),
        0
    );
    assert!(result["tracks"]["mic"].get("diarization").is_none());

    let segments = result["tracks"]["remote"]["diarization"]
        .as_array()
        .expect("the remote track has no diarization");
    let speakers: std::collections::BTreeSet<&str> = segments
        .iter()
        .map(|segment| segment["speaker"].as_str().unwrap())
        .collect();
    // The recipe has no count prior; the reference has REFERENCE_SPEAKERS
    // speakers over 1 s in this window. Allow one over- or under-split.
    assert!(
        speakers.len().abs_diff(REFERENCE_SPEAKERS) <= 1,
        "the clip has {REFERENCE_SPEAKERS} reference speakers over 1 s, found {speakers:?}"
    );
    assert!(
        segments
            .iter()
            .all(|segment| { segment["e"].as_f64().unwrap() >= segment["s"].as_f64().unwrap() }),
        "diarization segments are out of order"
    );

    // Progress order: remote transcription, remote diarization, mic transcription.
    let events: Vec<(String, f64)> = sink
        .lines()
        .into_iter()
        .filter(|line| line["event"] == "finalize_progress")
        .map(|line| {
            (
                line["stage"].as_str().unwrap().to_string(),
                line["fraction"].as_f64().unwrap(),
            )
        })
        .collect();
    assert!(!events.is_empty(), "no progress events");
    assert_eq!(events[0].0, "transcription");
    let diarization: Vec<&(String, f64)> = events
        .iter()
        .filter(|event| event.0 == "diarization")
        .collect();
    assert_eq!(diarization.len(), 2, "diarization emits 0.0 then 1.0");
    assert_eq!(diarization[0].1, 0.0);
    assert_eq!(diarization[1].1, 1.0);
    // The remote transcription comes first; the microphone track is empty in an
    // import, so diarization is the last stage. The per-track restart of the
    // transcription fraction is covered by the unit test in `finalize`.
    assert_eq!(
        &events[events.len() - 2..],
        &[
            ("diarization".to_string(), 0.0),
            ("diarization".to_string(), 1.0),
        ]
    );

    let text: String = result["tracks"]["remote"]["words"]
        .as_array()
        .unwrap()
        .iter()
        .map(|word| word["w"].as_str().unwrap())
        .collect::<Vec<_>>()
        .join(" ");
    println!(
        "imported {imported_seconds:.1} s; finalize {duration:.1} s; {} remote words, {} mic words, {} speakers, language {:?}",
        result["tracks"]["remote"]["words"].as_array().unwrap().len(),
        result["tracks"]["mic"]["words"].as_array().unwrap().len(),
        speakers.len(),
        result["language"]
    );
    println!(
        "first words: {}",
        text.chars().take(200).collect::<String>()
    );
    assert!(!text.trim().is_empty());
}
