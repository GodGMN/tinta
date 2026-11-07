//! `finalize`: reads both tracks, transcribes and diarizes, and emits progress
//! in the Swift engine's order.
//!
//! Order (`Engine.swift:214-226`): remote transcription windows, remote
//! diarization only when the remote track has words, then mic transcription
//! windows. The transcription fraction restarts for each track, so the app's
//! progress bar can move backwards.
//!
//! The microphone track is filtered against speaker echo after transcription,
//! with the pure decision ported in [`crate::echo`].
//!
//! # Memory
//!
//! A track at the 12 h codec cap is 691,200,000 samples, 2.76 GB as f32. Both
//! tracks are alive while the microphone is transcribed, so the peak is about
//! **5.5 GB of audio** plus the models (under 1 GB), not the 7 GB the first
//! version used: [`read_track`] now decrypts one 5 s chunk at a time (160 KB of
//! i16) instead of holding every decrypted chunk next to the f32 track. The
//! remote buffer is dropped as soon as the echo filter has read it, and the mic
//! buffer right after. The whole pass holds the model lock while it reads, so a
//! second queued finalize does not hold another 5.5 GB while it waits.

use anyhow::Result;
use serde_json::{json, Map, Value};
use std::path::Path;

use crate::audio::SAMPLE_RATE;
use crate::echo;
use crate::language;
use crate::protocol::Output;
use crate::speech::{with_models, FinalPass};
use crate::types::{segments_json, words_json, Word};
use tinta_core::chunks::{read_track, Track};

/// Runs a whole final pass and returns the command result.
pub fn finalize(
    pass: &mut dyn FinalPass,
    dir: &Path,
    key: &[u8; 32],
    max_speakers: Option<usize>,
    out: &Output,
) -> Result<Value> {
    let mut emit = |stage: &str, fraction: f64| {
        let _ = out.event(
            "finalize_progress",
            json!({ "stage": stage, "fraction": fraction }),
        );
    };
    // Two final passes for different meetings may overlap; the model lock makes
    // them wait for each other instead of loading the recognizer twice. The lock
    // is taken before the tracks are read, so the waiting pass holds no audio.
    with_models(|| {
        let remote = read_track(dir, key, Track::Remote)?;
        let mic = read_track(dir, key, Track::Mic)?;
        finalize_tracks(pass, remote, mic, max_speakers, &mut emit)
    })
}

/// The testable core: takes the two tracks and emits progress through `emit`.
pub fn finalize_tracks(
    pass: &mut dyn FinalPass,
    remote: Vec<f32>,
    mic: Vec<f32>,
    max_speakers: Option<usize>,
    emit: &mut dyn FnMut(&str, f64),
) -> Result<Value> {
    let duration = remote.len().max(mic.len()) as f64 / SAMPLE_RATE as f64;
    let mut tracks = Map::new();
    let mut all_text = String::new();

    let remote_words = pass.transcribe(&remote, &mut |fraction| emit("transcription", fraction))?;
    all_text.push_str(&joined(&remote_words));
    all_text.push(' ');
    let mut remote_entry = json!({ "words": words_json(&remote_words) });
    let mut notes: Vec<String> = Vec::new();
    if !remote_words.is_empty() {
        // Silent audio has no speakers, and the diarizer can fail on it.
        let segments = pass.diarize(&remote, max_speakers, &mut |fraction| {
            emit("diarization", fraction)
        })?;
        remote_entry["diarization"] = segments_json(&segments);
        notes = pass.notes();
    }
    tracks.insert("remote".to_string(), remote_entry);

    let mic_words = pass.transcribe(&mic, &mut |fraction| emit("transcription", fraction))?;
    let mic_words = echo::filter(&mic, &remote, mic_words);
    // The remote samples are not needed once the echo decision is made; at 12 h
    // this releases 2.76 GB before the language guess and the result are built.
    drop(remote);
    drop(mic);
    all_text.push_str(&joined(&mic_words));
    all_text.push(' ');
    tracks.insert(
        "mic".to_string(),
        json!({ "words": words_json(&mic_words) }),
    );

    let mut result = json!({ "tracks": Value::Object(tracks), "duration": duration });
    if let Some(language) = language::detect(&all_text) {
        result["language"] = Value::String(language);
    }
    if !notes.is_empty() {
        // An extra field the app ignores: `FinalResult` deserializes without
        // denying unknown fields, so older apps are unaffected.
        result["notes"] = json!(notes);
    }
    Ok(result)
}

fn joined(words: &[Word]) -> String {
    words
        .iter()
        .map(|word| word.w.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SpeakerSegment;
    use anyhow::bail;

    /// A pass whose work is scripted per call, so the test can look at the order
    /// and count of the progress events without any model. The first `transcribe`
    /// call is the remote track and the second is the microphone track.
    struct Fake {
        remote: Vec<Word>,
        mic: Vec<Word>,
        windows: usize,
        diarization: Vec<SpeakerSegment>,
        notes: Vec<String>,
        calls: usize,
        fail_mic: bool,
    }

    impl FinalPass for Fake {
        fn transcribe(
            &mut self,
            _samples: &[f32],
            progress: &mut dyn FnMut(f64),
        ) -> Result<Vec<Word>> {
            let is_mic = self.calls == 1;
            self.calls += 1;
            if is_mic && self.fail_mic {
                bail!("microphone transcription failed");
            }
            for index in 0..self.windows {
                progress((index + 1) as f64 / self.windows as f64);
            }
            Ok(if is_mic {
                self.mic.clone()
            } else {
                self.remote.clone()
            })
        }

        fn diarize(
            &mut self,
            _samples: &[f32],
            _max_speakers: Option<usize>,
            progress: &mut dyn FnMut(f64),
        ) -> Result<Vec<SpeakerSegment>> {
            progress(0.0);
            progress(1.0);
            Ok(self.diarization.clone())
        }

        fn notes(&self) -> Vec<String> {
            self.notes.clone()
        }
    }

    fn word(text: &str, start: f64, end: f64) -> Word {
        Word {
            w: text.to_string(),
            s: start,
            e: end,
        }
    }

    fn fake(remote: Vec<Word>, mic: Vec<Word>, windows: usize) -> Fake {
        Fake {
            remote,
            mic,
            windows,
            diarization: vec![SpeakerSegment {
                speaker: "S1".to_string(),
                s: 0.0,
                e: 1.0,
            }],
            notes: Vec::new(),
            calls: 0,
            fail_mic: false,
        }
    }

    fn run(remote: Vec<Word>, mic: Vec<Word>, windows: usize) -> (Value, Vec<(String, f64)>) {
        let mut pass = fake(remote, mic, windows);
        let remote_samples = vec![0.5f32; 16_000];
        let mic_samples = vec![0.5f32; 16_000];
        let mut events = Vec::new();
        let result = finalize_tracks(
            &mut pass,
            remote_samples,
            mic_samples,
            None,
            &mut |stage, fraction| events.push((stage.to_string(), fraction)),
        )
        .unwrap();
        (result, events)
    }

    #[test]
    fn progress_order_matches_the_swift_engine() {
        let (_, events) = run(
            vec![word("remote", 0.0, 0.5)],
            vec![word("mic", 0.0, 0.5)],
            2,
        );
        assert_eq!(
            events,
            vec![
                ("transcription".to_string(), 0.5),
                ("transcription".to_string(), 1.0),
                ("diarization".to_string(), 0.0),
                ("diarization".to_string(), 1.0),
                ("transcription".to_string(), 0.5),
                ("transcription".to_string(), 1.0),
            ]
        );
    }

    #[test]
    fn a_silent_remote_track_skips_diarization() {
        let (result, events) = run(Vec::new(), vec![word("mic", 0.0, 0.5)], 1);
        assert!(result["tracks"]["remote"].get("diarization").is_none());
        assert_eq!(
            events,
            vec![
                ("transcription".to_string(), 1.0),
                ("transcription".to_string(), 1.0),
            ]
        );
    }

    #[test]
    fn the_result_has_the_consumed_shape() {
        let (result, _) = run(
            vec![word("hello", 0.0, 0.5), word("everyone", 0.6, 1.0)],
            vec![word("there", 1.0, 1.5), word("again", 1.6, 1.9)],
            1,
        );
        assert_eq!(result["duration"], 1.0);
        assert_eq!(result["tracks"]["remote"]["words"][0]["w"], "hello");
        assert_eq!(
            result["tracks"]["remote"]["diarization"][0]["speaker"],
            "S1"
        );
        assert_eq!(result["tracks"]["mic"]["words"][0]["w"], "there");
        // The concatenated transcript is long enough for whatlang.
        assert_eq!(result["language"], "en");
    }

    #[test]
    fn a_diarization_note_reaches_the_result() {
        let mut pass = fake(vec![word("remote", 0.0, 0.5)], Vec::new(), 1);
        pass.notes = vec!["speaker labels stop at 4 h".to_string()];
        let result = finalize_tracks(
            &mut pass,
            vec![0.5f32; 16_000],
            vec![0.5f32; 16_000],
            None,
            &mut |_, _| {},
        )
        .unwrap();
        assert_eq!(result["notes"][0], "speaker labels stop at 4 h");
    }

    #[test]
    fn microphone_echo_is_filtered_after_transcription() {
        let mut remote = vec![0.1f32; 32_000];
        let mic = {
            let mut mic = vec![0.0f32; 32_000];
            mic[16_000..].fill(0.1);
            mic
        };
        remote[16_000..].fill(0.1);
        let mut pass = fake(
            Vec::new(),
            vec![word("echo", 0.0, 1.0), word("speech", 1.0, 2.0)],
            1,
        );
        let result = finalize_tracks(&mut pass, remote, mic, None, &mut |_, _| {}).unwrap();
        let words = result["tracks"]["mic"]["words"].as_array().unwrap();
        assert_eq!(words.len(), 1);
        assert_eq!(words[0]["w"], "speech");
    }

    #[test]
    fn a_failing_mic_transcription_fails_the_command() {
        let mut pass = fake(Vec::new(), Vec::new(), 1);
        pass.fail_mic = true;
        let error = finalize_tracks(
            &mut pass,
            vec![0.0; 16],
            vec![0.0; 16],
            None,
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("microphone transcription failed"));
    }
}
