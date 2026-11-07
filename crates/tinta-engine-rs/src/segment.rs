//! VAD-driven segmentation for the final transcription pass.
//!
//! The rules come from the binding "Engine design conditions" in
//! `docs/fork/benchmark.md`:
//!
//! - the sherpa-onnx Silero VAD finds speech, with at most 30 s per VAD segment;
//! - a longer speech run is cut at the quietest 50 ms window in the last 40 % of
//!   the allowed span, not at an arbitrary boundary;
//! - every piece gets 2 s of pre-roll and 1 s of post-roll and is extended to at
//!   least 10 s, because Parakeet can drop the first words of a piece that starts
//!   mid-speech;
//! - a hard cap of 60 s is checked in code before every recognizer call;
//! - there is **no forced flush** every 20 s: the VAD closes a segment at 30 s on
//!   its own, so a flush would only punch a hole larger than the 0.5 s join gap.

use crate::audio::SAMPLE_RATE;
use crate::types::Word;
use anyhow::{bail, Result};
use sherpa_onnx::VoiceActivityDetector;

/// No recognizer call may exceed this many seconds. The benchmark measured a
/// 240 s window losing whole stretches of speech in the middle.
pub const MAX_ASR_SECONDS: f32 = 60.0;
/// The longest piece the VAD and the splitter produce.
const MAX_PIECE_SECONDS: f32 = 30.0;
const PRE_ROLL_SECONDS: f32 = 2.0;
const POST_ROLL_SECONDS: f32 = 1.0;
const MIN_PIECE_SECONDS: f32 = 10.0;
const SPLIT_WINDOW_SECONDS: f32 = 0.05;
const JOIN_GAP_SECONDS: f32 = 0.5;

fn samples(seconds: f32) -> usize {
    (seconds * SAMPLE_RATE as f32) as usize
}

/// Fails instead of decoding a window over the hard cap. Called before every
/// recognizer call, so a future segmentation change cannot feed a long window.
pub fn ensure_recognizer_window(len: usize) -> Result<()> {
    let seconds = len as f32 / SAMPLE_RATE as f32;
    if seconds > MAX_ASR_SECONDS + 0.001 {
        bail!("refusing a {seconds:.1} s recognizer window (the cap is {MAX_ASR_SECONDS:.0} s)");
    }
    Ok(())
}

/// Runs the VAD over a whole track and returns speech ranges in samples.
///
/// The VAD's `max_speech_duration` closes a continuous run at 30 s, so no extra
/// flush is needed and no artificial hole is inserted.
pub fn vad_segments(vad: &VoiceActivityDetector, audio: &[f32]) -> Vec<(usize, usize)> {
    let block = SAMPLE_RATE as usize;
    let mut segments = Vec::new();
    for chunk in audio.chunks(block) {
        vad.accept_waveform(chunk);
        drain(vad, &mut segments, audio.len());
    }
    vad.flush();
    drain(vad, &mut segments, audio.len());
    segments
}

fn drain(vad: &VoiceActivityDetector, segments: &mut Vec<(usize, usize)>, len: usize) {
    while let Some(segment) = vad.front() {
        let start = segment.start().max(0) as usize;
        let count = segment.n().max(0) as usize;
        if count > 0 {
            segments.push((start, (start + count).min(len)));
        }
        vad.pop();
    }
}

/// The quietest 50 ms window midpoint in `[lo, hi)`.
fn quietest_point(audio: &[f32], lo: usize, hi: usize) -> usize {
    let window = samples(SPLIT_WINDOW_SECONDS).max(1);
    if hi <= lo + window {
        return (lo + hi) / 2;
    }
    let mut best = lo + window / 2;
    let mut best_energy = f32::INFINITY;
    let mut position = lo;
    while position + window <= hi {
        let energy: f32 = audio[position..position + window]
            .iter()
            .map(|sample| sample * sample)
            .sum();
        if energy < best_energy {
            best_energy = energy;
            best = position + window / 2;
        }
        position += window;
    }
    best
}

/// Splits a speech run into pieces of at most 30 s, cutting at the quietest point
/// in the last 40 % of each piece.
fn split_run(audio: &[f32], start: usize, end: usize) -> Vec<(usize, usize)> {
    let max_piece = samples(MAX_PIECE_SECONDS);
    let mut runs = Vec::new();
    let mut position = start;
    while end.saturating_sub(position) > max_piece {
        let hi = position + max_piece;
        let lo = position + (max_piece as f32 * 0.6) as usize;
        let split = quietest_point(audio, lo, hi);
        runs.push((position, split));
        position = split;
    }
    if end > position {
        runs.push((position, end));
    }
    runs
}

/// Groups VAD speech into decode pieces with pre-roll, post-roll, a 10 s minimum
/// and the 60 s hard cap. Adjacent runs closer than 0.5 s merge; the pieces
/// themselves may overlap, and [`trim_overlaps`] de-duplicates their words.
///
/// Every piece starts at its own speech start minus the pre-roll, never at 0:
/// a track whose first speech is at 5 minutes must not produce a 0–60 s piece.
/// A piece over the hard cap is split at the quietest points, never truncated,
/// so no speech region is ever left out. [`covers_speech`] checks that invariant
/// in debug builds and in the tests.
pub fn group_pieces(audio: &[f32], segments: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let max_piece = samples(MAX_PIECE_SECONDS);
    let join_gap = samples(JOIN_GAP_SECONDS);
    let pre = samples(PRE_ROLL_SECONDS);
    let post = samples(POST_ROLL_SECONDS);
    let min_piece = samples(MIN_PIECE_SECONDS);

    let mut groups: Vec<(usize, usize)> = Vec::new();
    for (segment_start, segment_end) in segments {
        for (a, b) in split_run(audio, *segment_start, *segment_end) {
            if let Some(last) = groups.last_mut() {
                if b.saturating_sub(last.0) <= max_piece && a.saturating_sub(last.1) <= join_gap {
                    last.1 = b;
                    continue;
                }
            }
            groups.push((a, b));
        }
    }

    let mut pieces = Vec::new();
    for (a, b) in groups {
        let start = a.saturating_sub(pre);
        let end = (b + post).max(start + min_piece).min(audio.len());
        if end <= start {
            continue;
        }
        pieces.extend(split_to_cap(audio, start, end));
    }
    debug_assert!(
        covers_speech(&pieces, segments),
        "the decode pieces leave a VAD speech region uncovered"
    );
    pieces
}

/// Splits `[start, end)` into pieces of at most [`MAX_ASR_SECONDS`] at the
/// quietest points. Groups are at most 30 s, so this only fires if a future
/// change lengthens them; it must split rather than truncate, because truncating
/// silently drops whatever comes after the cap.
fn split_to_cap(audio: &[f32], start: usize, end: usize) -> Vec<(usize, usize)> {
    let cap = samples(MAX_ASR_SECONDS);
    let mut pieces = Vec::new();
    let mut position = start;
    while end.saturating_sub(position) > cap {
        let hi = position + cap;
        let lo = position + (cap as f32 * 0.6) as usize;
        let split = quietest_point(audio, lo, hi).clamp(position + 1, hi);
        pieces.push((position, split));
        position = split;
    }
    if end > position {
        pieces.push((position, end));
    }
    pieces
}

/// True when every VAD speech region lies inside the union of the decode pieces.
/// A hole means speech the recognizer never sees, so [`group_pieces`] asserts it
/// and the tests check the awkward cases directly.
pub fn covers_speech(pieces: &[(usize, usize)], segments: &[(usize, usize)]) -> bool {
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for &(start, end) in pieces {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    segments
        .iter()
        .all(|&(start, end)| merged.iter().any(|&(a, b)| a <= start && b >= end))
}

/// De-duplicates words in overlapping pieces at the midpoint of the overlap on
/// both sides, then sorts by start time.
pub fn trim_overlaps(pieces: &[(usize, usize)], words: &mut [Vec<Word>]) -> Vec<Word> {
    let rate = SAMPLE_RATE as f64;
    for index in 1..pieces.len().min(words.len()) {
        let boundary = (pieces[index - 1].1 as f64 + pieces[index].0 as f64) / 2.0 / rate;
        words[index - 1].retain(|word| word.s < boundary);
        words[index].retain(|word| word.s >= boundary);
    }
    let mut all: Vec<Word> = words.iter_mut().flat_map(std::mem::take).collect();
    all.sort_by(|a, b| a.s.partial_cmp(&b.s).unwrap_or(std::cmp::Ordering::Equal));
    all.retain(|word| !word.w.trim().is_empty());
    all
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(len: usize) -> Vec<f32> {
        vec![0.0; len]
    }

    #[test]
    fn the_recognizer_window_cap_is_enforced() {
        assert!(ensure_recognizer_window(samples(60.0)).is_ok());
        assert!(ensure_recognizer_window(samples(30.0)).is_ok());
        let error = ensure_recognizer_window(samples(60.01)).unwrap_err();
        assert!(error.to_string().contains("cap is 60 s"), "{error}");
    }

    #[test]
    fn pieces_are_at_most_thirty_seconds_and_never_over_the_hard_cap() {
        let audio = silence(samples(180.0));
        let run = [(0, samples(180.0))];
        let pieces = group_pieces(&audio, &run);
        assert!(pieces.len() >= 6, "pieces {}", pieces.len());
        for (start, end) in &pieces {
            let seconds = (end - start) as f32 / SAMPLE_RATE as f32;
            // A 30 s group plus its 2 s pre-roll and 1 s post-roll.
            let longest = MAX_PIECE_SECONDS + PRE_ROLL_SECONDS + POST_ROLL_SECONDS;
            assert!(seconds <= longest + 0.01, "piece {seconds} s");
            assert!(seconds <= MAX_ASR_SECONDS, "piece over the hard cap");
            ensure_recognizer_window(end - start).unwrap();
            assert!(end <= &audio.len());
        }
        // No gaps inside the speech run: each piece starts at or before the
        // previous one ends (the pre-roll creates the overlap).
        for pair in pieces.windows(2) {
            assert!(pair[1].0 <= pair[0].1, "hole between pieces");
        }
        // The run is covered from 0 to the end.
        assert_eq!(pieces.first().unwrap().0, 0);
        assert_eq!(pieces.last().unwrap().1, audio.len());
        assert!(covers_speech(&pieces, &run));
    }

    #[test]
    fn a_track_that_starts_with_five_minutes_of_silence_is_covered() {
        let audio = silence(samples(420.0));
        let run = [(samples(300.0), samples(305.0))];
        let pieces = group_pieces(&audio, &run);
        assert_eq!(pieces.len(), 1, "pieces {pieces:?}");
        // The piece starts at the speech minus the 2 s pre-roll, not at 0.
        assert_eq!(pieces[0].0, samples(298.0));
        // Five seconds of speech plus 1 s of post-roll, extended to 10 s.
        assert_eq!(pieces[0].1, samples(308.0));
        assert!(covers_speech(&pieces, &run));
    }

    #[test]
    fn an_opening_run_past_sixty_seconds_is_not_truncated() {
        let audio = silence(samples(120.0));
        let run = [(samples(45.0), samples(74.0))];
        let pieces = group_pieces(&audio, &run);
        assert_eq!(pieces.len(), 1, "pieces {pieces:?}");
        assert_eq!(pieces[0].0, samples(43.0));
        assert_eq!(pieces[0].1, samples(75.0));
        // The speech after 60 s is inside the piece, so the cap never truncates.
        assert!(pieces[0].1 > samples(60.0));
        assert!(covers_speech(&pieces, &run));
    }

    #[test]
    fn a_three_minute_run_has_no_holes_and_covers_every_second() {
        let audio = silence(samples(180.0));
        let run = [(0, samples(180.0))];
        let pieces = group_pieces(&audio, &run);
        assert!(pieces.len() >= 6, "pieces {}", pieces.len());
        for &(start, end) in &pieces {
            ensure_recognizer_window(end - start).unwrap();
        }
        for pair in pieces.windows(2) {
            assert!(pair[1].0 <= pair[0].1, "hole between pieces");
        }
        assert_eq!(pieces.first().unwrap().0, 0);
        assert!(pieces.last().unwrap().1 >= samples(180.0));
        assert!(covers_speech(&pieces, &run));
    }

    #[test]
    fn a_piece_over_the_hard_cap_is_split_at_the_quietest_point() {
        let mut audio = silence(samples(150.0));
        for second in [50usize, 100] {
            for sample in audio
                .iter_mut()
                .skip(second * SAMPLE_RATE as usize)
                .take(samples(0.05))
            {
                *sample = 1.0;
            }
        }
        let pieces = split_to_cap(&audio, 0, audio.len());
        assert!(pieces.len() >= 3, "pieces {pieces:?}");
        assert_eq!(pieces.first().unwrap().0, 0);
        assert_eq!(pieces.last().unwrap().1, audio.len());
        for &(start, end) in &pieces {
            ensure_recognizer_window(end - start).unwrap();
        }
        for pair in pieces.windows(2) {
            assert_eq!(pair[0].1, pair[1].0, "the split must not drop samples");
            let cut = pair[0].1;
            for loud in [50 * SAMPLE_RATE as usize, 100 * SAMPLE_RATE as usize] {
                assert!(
                    cut <= loud || cut >= loud + samples(0.05),
                    "cut landed in the loud window at {loud}"
                );
            }
        }
    }

    #[test]
    fn split_run_cuts_at_the_quietest_point() {
        let mut audio = silence(samples(60.0));
        // Make one 50 ms window in the search range much louder.
        let quiet = 40 * SAMPLE_RATE as usize;
        for sample in audio.iter_mut().skip(quiet).take(samples(0.05)) {
            *sample = 1.0;
        }
        let pieces = split_run(&audio, 0, samples(60.0));
        let cut = pieces[0].1;
        assert!(
            cut < quiet || cut > quiet + samples(0.05),
            "cut landed in the loud window"
        );
        assert!(cut >= (samples(30.0) as f32 * 0.6) as usize);
        assert!(cut <= samples(30.0));
    }

    #[test]
    fn a_short_run_is_extended_to_ten_seconds() {
        let audio = silence(samples(30.0));
        let pieces = group_pieces(&audio, &[(samples(0.5), samples(1.0))]);
        assert_eq!(pieces.len(), 1);
        let (start, end) = pieces[0];
        // First piece starts at 0; the half-second run is extended to ten seconds.
        assert_eq!(start, 0);
        assert_eq!((end - start) as f32 / SAMPLE_RATE as f32, 10.0);
    }

    #[test]
    fn pre_and_post_roll_wrap_later_runs() {
        let audio = silence(samples(60.0));
        let pieces = group_pieces(
            &audio,
            &[
                (samples(10.0), samples(11.0)),
                (samples(20.0), samples(30.0)),
            ],
        );
        assert_eq!(pieces.len(), 2);
        let (start, end) = pieces[1];
        assert_eq!(start, samples(18.0));
        assert_eq!(end, samples(31.0));
    }

    #[test]
    fn close_runs_merge_and_far_runs_stay_apart() {
        let audio = silence(samples(60.0));
        let pieces = group_pieces(
            &audio,
            &[
                (samples(10.0), samples(11.0)),
                (samples(11.4), samples(12.0)),
            ],
        );
        assert_eq!(pieces.len(), 1);

        let pieces = group_pieces(
            &audio,
            &[
                (samples(10.0), samples(11.0)),
                (samples(20.0), samples(21.0)),
            ],
        );
        assert_eq!(pieces.len(), 2);
    }

    #[test]
    fn overlapping_words_are_deduplicated_at_the_midpoint() {
        // Piece 2 starts 1 s inside piece 1, so the boundary is at 9.5 s: a
        // word that ended in both keeps only the earlier copy, and a word that
        // starts in the overlap goes to the later piece.
        let pieces = [(0, samples(10.0)), (samples(9.0), samples(18.0))];
        let mut words = vec![
            vec![
                Word {
                    w: "first".to_string(),
                    s: 0.0,
                    e: 0.5,
                },
                Word {
                    w: "overlap".to_string(),
                    s: 9.2,
                    e: 9.4,
                },
            ],
            vec![
                Word {
                    w: "overlap".to_string(),
                    s: 9.2,
                    e: 9.4,
                },
                Word {
                    w: "late".to_string(),
                    s: 9.8,
                    e: 10.2,
                },
            ],
        ];
        let all = trim_overlaps(&pieces, &mut words);
        let texts: Vec<&str> = all.iter().map(|word| word.w.as_str()).collect();
        assert_eq!(texts, vec!["first", "overlap", "late"]);
        assert!(all.windows(2).all(|pair| pair[0].s <= pair[1].s));
    }

    #[test]
    fn empty_words_are_dropped() {
        let mut words = vec![vec![Word {
            w: "  ".to_string(),
            s: 0.0,
            e: 0.1,
        }]];
        assert!(trim_overlaps(&[(0, 100)], &mut words).is_empty());
    }
}
