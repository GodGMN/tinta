//! VAD-driven segmentation for the final transcription pass.
//!
//! The rules come from the binding "Engine design conditions" in
//! `docs/fork/benchmark.md` and the task #21 evidence in
//! `docs/fork/segmentation.md`:
//!
//! - the sherpa-onnx Silero VAD finds speech. A continuous run is closed by the
//!   VAD only at [`Settings::vad_max_speech_seconds`], so the VAD itself never
//!   cuts speech every 20 or 30 s. That forced cut was the largest measured
//!   source of deletions: the VAD needs about 0.8 s to re-detect, the next piece
//!   starts mid-speech, and Parakeet can return no words for it;
//! - speech runs closer than [`Settings::join_gap_seconds`] are decoded as one
//!   piece, so the recognizer sees a short pause as a pause, not a boundary;
//! - a span longer than [`Settings::max_piece_seconds`] is cut at the quietest
//!   50 ms window in the last 40 % of the allowed span, not at an arbitrary
//!   boundary;
//! - a piece starts at the quietest point of the
//!   [`Settings::pre_roll_seconds`] before its first speech, so no piece starts
//!   mid-word, and every cut after the first backs the next piece up by that
//!   pre-roll, so both recognizer calls see the join;
//! - a shorter span is extended to [`Settings::min_piece_seconds`];
//! - a hard cap of 60 s is checked in code before every recognizer call;
//! - after the first pass, a hypothesis hole of
//!   [`Settings::gap_redecode_seconds`] or more inside a VAD speech region is
//!   decoded again on its own, retrying from a shifted start while it still
//!   leaves a hole. This costs a few percent more ASR time and targets the
//!   44-70 % of deletions that fell inside 3 s+ holes.
//!
//! Cutting *inside* speech at a target of 8-15 s was measured and rejected: it
//! creates more cuts, the decoder drops words at a cut, and the remaining
//! long-form loss is crosstalk, which a shorter single-speaker window does not
//! fix without oracle boundaries. See `docs/fork/segmentation.md` for the table
//! per variant and meeting.

use crate::audio::SAMPLE_RATE;
use crate::types::Word;
use anyhow::{bail, Result};
use sherpa_onnx::VoiceActivityDetector;

/// No recognizer call may exceed this many seconds. The benchmark measured a
/// 240 s window losing whole stretches of speech in the middle.
pub const MAX_ASR_SECONDS: f32 = 60.0;
const SPLIT_WINDOW_SECONDS: f32 = 0.05;

/// The segmentation knobs, all in seconds. [`Settings::default`] is the
/// task #21 combination; the harness in `scripts/fork/segmentation` measures
/// the alternatives against it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// The longest speech span handed to the recognizer in one piece. The audio
    /// of a piece can be up to this plus the pre- and post-roll.
    pub max_piece_seconds: f32,
    /// Audio kept before a piece's first speech, starting at its quietest point.
    pub pre_roll_seconds: f32,
    /// Audio kept after a piece's last speech.
    pub post_roll_seconds: f32,
    /// A span shorter than this is extended to this length.
    pub min_piece_seconds: f32,
    /// VAD runs no further apart than this are decoded as one piece.
    pub join_gap_seconds: f32,
    /// The VAD closes a continuous run at this length. It must stay above
    /// [`Settings::max_piece_seconds`] so the VAD does not cut mid-speech.
    pub vad_max_speech_seconds: f32,
    /// Re-decode a hypothesis hole at least this long when the VAD heard speech
    /// there. Zero disables the extra pass.
    pub gap_redecode_seconds: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            // 57 s of speech plus the 3 s of rolls stays at or under the 60 s
            // hard cap. Task #21 measured longer pieces as better than the
            // reviewed 8-15 s target on all three ablation meetings: a cut in the
            // middle of speech costs more than the shorter window gains, because
            // the recognizer drops words at a cut and the remaining loss is
            // crosstalk, which a shorter single-speaker window does not fix.
            max_piece_seconds: 57.0,
            pre_roll_seconds: 2.0,
            post_roll_seconds: 1.0,
            min_piece_seconds: 10.0,
            join_gap_seconds: 0.5,
            // The VAD's own cut must stay above the piece length: a cut in the
            // middle of speech is the failure mode this segmentation removes.
            vad_max_speech_seconds: 60.0,
            // Task #21: re-decoding the 3 s+ holes the first pass leaves recovers
            // 4.2 points of WER on ES2004b at a few percent more ASR time.
            gap_redecode_seconds: 3.0,
        }
    }
}

fn samples(seconds: f32) -> usize {
    (seconds.max(0.0) * SAMPLE_RATE as f32) as usize
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
/// There is **no forced flush**: the VAD closes a run on its own at
/// `max_speech_duration` (60 s by default), and any earlier flush would insert a
/// hole larger than the VAD's re-detection time, which is the failure mode this
/// segmentation exists to avoid.
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

/// Splits `[start, end)` into decode pieces of at most
/// `max_piece + pre_roll + post_roll` seconds, cutting at the quietest point in
/// the last 40 % of each piece, and backing every cut up by the pre-roll so the
/// join is seen by both recognizer calls. The tail piece keeps at least
/// [`Settings::min_piece_seconds`] of audio.
fn split_span(settings: &Settings, audio: &[f32], start: usize, end: usize) -> Vec<(usize, usize)> {
    let pre = samples(settings.pre_roll_seconds);
    let min_piece = samples(settings.min_piece_seconds);
    let limit = samples(
        (settings.max_piece_seconds + settings.pre_roll_seconds + settings.post_roll_seconds)
            .min(MAX_ASR_SECONDS),
    )
    .max(1);
    let mut pieces = Vec::new();
    let mut position = start;
    while end.saturating_sub(position) > limit {
        let lo = position + (limit as f32 * 0.6) as usize;
        // Keep enough audio after the cut for the last piece to reach the
        // minimum. `max(lo)` only keeps the search range valid if a future
        // setting change breaks that relation.
        let hi = (position + limit)
            .min(end.saturating_sub(min_piece) + pre)
            .max(lo);
        let split = quietest_point(audio, lo, hi).clamp(position + 1, hi);
        pieces.push((position, split));
        // The next piece starts the pre-roll before the cut, never at or before
        // the previous start, so the loop always advances.
        position = split.saturating_sub(pre).max(position + 1);
    }
    if end > position {
        pieces.push((position, end));
    }
    pieces
}

/// Groups VAD speech into decode pieces, with the pre-roll starting at a quiet
/// point and the post-roll, a minimum length and the 60 s hard cap.
///
/// Adjacent runs closer than the join gap merge while the speech span stays
/// under [`Settings::max_piece_seconds`]; a longer continuous run is cut at the
/// quietest points. Every piece starts at or before its own first speech, never
/// at 0: a track whose first speech is at 5 minutes must not produce a 0-15 s
/// piece. [`covers_speech`] asserts on every call that no VAD region is left
/// out, and the tests check the awkward cases directly.
pub fn group_pieces(audio: &[f32], segments: &[(usize, usize)]) -> Vec<(usize, usize)> {
    group_pieces_with(&Settings::default(), audio, segments)
}

/// [`group_pieces`] with explicit settings.
pub fn group_pieces_with(
    settings: &Settings,
    audio: &[f32],
    segments: &[(usize, usize)],
) -> Vec<(usize, usize)> {
    let pre = samples(settings.pre_roll_seconds);
    let post = samples(settings.post_roll_seconds);
    let min_piece = samples(settings.min_piece_seconds);
    let max_speech = samples(settings.max_piece_seconds);
    let join_gap = samples(settings.join_gap_seconds);

    let mut groups: Vec<(usize, usize)> = Vec::new();
    for &(start, end) in segments {
        if end <= start {
            continue;
        }
        if let Some(last) = groups.last_mut() {
            if start.saturating_sub(last.1) <= join_gap && end.max(last.1) - last.0 <= max_speech {
                last.1 = last.1.max(end);
                continue;
            }
        }
        groups.push((start, end));
    }

    let mut pieces = Vec::new();
    for (speech_start, speech_end) in groups {
        let start = quietest_point(audio, speech_start.saturating_sub(pre), speech_start);
        let end = (speech_end + post).max(start + min_piece).min(audio.len());
        if end <= start {
            continue;
        }
        pieces.extend(split_span(settings, audio, start, end));
    }
    // Grouping covers every region today; if a later change breaks that, decode the
    // uncovered regions as extra pieces instead of losing speech or the whole pass.
    if !covers_speech(&pieces, segments) {
        let uncovered: Vec<(usize, usize)> = segments
            .iter()
            .copied()
            .filter(|&seg| !covers_speech(&pieces, &[seg]))
            .collect();
        eprintln!(
            "[tinta-engine] {} VAD regions were not grouped; decoding them separately",
            uncovered.len()
        );
        for (start, end) in uncovered {
            pieces.extend(split_span(settings, audio, start, end.min(audio.len())));
        }
        pieces.sort_unstable();
    }
    pieces
}

/// True when every VAD speech region lies inside the union of the decode pieces.
/// A hole means speech the recognizer never sees, so [`group_pieces_with`]
/// asserts it and the tests check the awkward cases directly.
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

/// Stretches at least `min_seconds` long inside a VAD speech region where the
/// first pass produced no word, in samples.
///
/// A piece that starts mid-speech can come back empty and leave a hole as long
/// as the piece; this finds those holes (and the ones between two words) so
/// [`crate::speech`] can decode them again. A word belongs to a region by its
/// midpoint, so a word on a boundary is not counted twice. Overlapping VAD
/// regions produce merged ranges.
pub fn speech_gaps(
    segments: &[(usize, usize)],
    words: &[Word],
    min_seconds: f32,
) -> Vec<(usize, usize)> {
    let rate = SAMPLE_RATE as f64;
    let min = min_seconds.max(0.0) as f64;
    let mut gaps: Vec<(usize, usize)> = Vec::new();
    for &(region_start, region_end) in segments {
        let (begin, finish) = (region_start as f64 / rate, region_end as f64 / rate);
        let mut inside: Vec<&Word> = words
            .iter()
            .filter(|word| {
                let middle = (word.s + word.e) / 2.0;
                middle >= begin && middle < finish
            })
            .collect();
        inside.sort_by(|a, b| a.s.partial_cmp(&b.s).unwrap_or(std::cmp::Ordering::Equal));
        let mut cursor = begin;
        for word in inside {
            if word.s - cursor >= min {
                gaps.push((samples(cursor as f32), samples(word.s.min(finish) as f32)));
            }
            cursor = cursor.max(word.e);
        }
        if finish - cursor >= min {
            gaps.push((samples(cursor as f32), samples(finish as f32)));
        }
    }
    gaps.sort_unstable();
    let mut merged: Vec<(usize, usize)> = Vec::new();
    for (start, end) in gaps {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged.retain(|(start, end)| end > start);
    merged
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

    fn piece_seconds(piece: &(usize, usize)) -> f32 {
        (piece.1 - piece.0) as f32 / SAMPLE_RATE as f32
    }

    #[test]
    fn the_recognizer_window_cap_is_enforced() {
        assert!(ensure_recognizer_window(samples(60.0)).is_ok());
        assert!(ensure_recognizer_window(samples(30.0)).is_ok());
        let error = ensure_recognizer_window(samples(60.01)).unwrap_err();
        assert!(error.to_string().contains("cap is 60 s"), "{error}");
    }

    #[test]
    fn the_defaults_keep_the_reviewed_conditions() {
        let settings = Settings::default();
        // A piece is at most the hard cap, rolls included.
        assert!(
            settings.max_piece_seconds + settings.pre_roll_seconds + settings.post_roll_seconds
                <= MAX_ASR_SECONDS
        );
        assert!(settings.vad_max_speech_seconds > settings.max_piece_seconds);
        assert!(settings.min_piece_seconds >= 3.0);
        assert!(settings.gap_redecode_seconds >= 3.0);
    }

    #[test]
    fn pieces_are_at_most_the_hard_cap() {
        let audio = silence(samples(180.0));
        let run = [(0, samples(180.0))];
        let pieces = group_pieces(&audio, &run);
        assert!(pieces.len() >= 3, "pieces {}", pieces.len());
        let settings = Settings::default();
        let longest =
            settings.max_piece_seconds + settings.pre_roll_seconds + settings.post_roll_seconds;
        for piece in &pieces {
            let seconds = piece_seconds(piece);
            assert!(seconds <= longest + 0.01, "piece {seconds} s");
            assert!(seconds <= MAX_ASR_SECONDS, "piece over the hard cap");
            ensure_recognizer_window(piece.1 - piece.0).unwrap();
            assert!(piece.1 <= audio.len());
        }
        // The pre-roll makes the pieces overlap, never leaves a hole.
        for pair in pieces.windows(2) {
            assert!(pair[1].0 <= pair[0].1, "hole between pieces");
        }
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
        // The piece starts in the 2 s before the speech, not at 0.
        assert!(pieces[0].0 >= samples(298.0));
        assert!(pieces[0].0 <= samples(300.0));
        // Five seconds of speech plus 1 s of post-roll, extended to 10 s from
        // the quiet point where the piece starts.
        assert_eq!(piece_seconds(&pieces[0]), 10.0);
        assert!(covers_speech(&pieces, &run));
    }

    #[test]
    fn an_opening_run_past_sixty_seconds_is_not_truncated() {
        let audio = silence(samples(160.0));
        // A continuous run over the 60 s cap: the plan must split it, not drop
        // whatever comes after the cap.
        let run = [(samples(45.0), samples(110.0))];
        let pieces = group_pieces(&audio, &run);
        assert!(pieces.len() >= 2, "pieces {pieces:?}");
        assert!(pieces[0].0 >= samples(43.0) && pieces[0].0 <= samples(45.0));
        assert_eq!(pieces.last().unwrap().1, samples(111.0));
        // The speech after 60 s is inside a piece.
        assert!(pieces.last().unwrap().1 > samples(60.0));
        assert!(covers_speech(&pieces, &run));
        for piece in &pieces {
            ensure_recognizer_window(piece.1 - piece.0).unwrap();
        }
    }

    #[test]
    fn a_three_minute_run_has_no_holes_and_covers_every_second() {
        let audio = silence(samples(180.0));
        let run = [(0, samples(180.0))];
        let pieces = group_pieces(&audio, &run);
        assert!(pieces.len() >= 3, "pieces {}", pieces.len());
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
        // A settings value over the hard cap must still split at 60 s.
        let settings = Settings {
            max_piece_seconds: 120.0,
            ..Settings::default()
        };
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
        let pieces = split_span(&settings, &audio, 0, audio.len());
        assert!(pieces.len() >= 3, "pieces {pieces:?}");
        assert_eq!(pieces.first().unwrap().0, 0);
        assert_eq!(pieces.last().unwrap().1, audio.len());
        for &(start, end) in &pieces {
            ensure_recognizer_window(end - start).unwrap();
            assert!(
                piece_seconds(&(start, end)) >= Settings::default().min_piece_seconds,
                "piece {} s under the minimum",
                piece_seconds(&(start, end))
            );
        }
        for pair in pieces.windows(2) {
            // The next piece backs up by the pre-roll: an overlap, never a hole.
            assert!(pair[1].0 <= pair[0].1, "hole between pieces");
        }
    }

    #[test]
    fn a_cut_at_the_quietest_point_avoids_a_loud_window() {
        // A 30 s target splits a 60 s span, so the cut lands in the search range.
        let settings = Settings {
            max_piece_seconds: 30.0,
            ..Settings::default()
        };
        let mut audio = silence(samples(60.0));
        // Make one 50 ms window in the search range much louder.
        let quiet = 25 * SAMPLE_RATE as usize;
        for sample in audio.iter_mut().skip(quiet).take(samples(0.05)) {
            *sample = 1.0;
        }
        let pieces = split_span(&settings, &audio, 0, samples(60.0));
        let cut = pieces[0].1;
        assert!(
            cut < quiet || cut > quiet + samples(0.05),
            "cut landed in the loud window"
        );
    }

    #[test]
    fn a_short_run_is_extended_to_ten_seconds() {
        let audio = silence(samples(30.0));
        let pieces = group_pieces(&audio, &[(samples(0.5), samples(1.0))]);
        assert_eq!(pieces.len(), 1);
        let (start, end) = pieces[0];
        // The half-second run is wrapped by the pre-roll and extended to ten
        // seconds; it starts in the silence before the run, not after it.
        assert!(start <= samples(0.5), "start {start}");
        assert_eq!(piece_seconds(&(start, end)), 10.0);
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
        // The pre-roll starts at or just before the 2 s before the run.
        assert!(start >= samples(18.0) && start <= samples(20.0));
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
    fn a_merged_group_is_never_longer_than_the_max_speech_span() {
        // Eight 1 s runs 0.4 s apart merge while they fit in max_piece_seconds.
        let audio = silence(samples(60.0));
        let settings = Settings {
            max_piece_seconds: 8.0,
            ..Settings::default()
        };
        let runs: Vec<(usize, usize)> = (0..8)
            .map(|index| {
                let start = samples(10.0 + index as f32 * 1.4);
                (start, start + samples(1.0))
            })
            .collect();
        let pieces = group_pieces_with(&settings, &audio, &runs);
        assert!(covers_speech(&pieces, &runs));
        let longest =
            settings.max_piece_seconds + settings.pre_roll_seconds + settings.post_roll_seconds;
        for piece in &pieces {
            assert!(
                piece_seconds(piece) <= longest + 0.01,
                "piece {} s",
                piece_seconds(piece)
            );
        }
    }

    #[test]
    fn the_pre_roll_starts_at_the_quietest_point_before_the_speech() {
        let settings = Settings::default();
        // Loud from 18 s to 19.5 s, silence after it: the pre-roll of a speech
        // run at 20 s must start in that silence, not at the fixed 18 s.
        let mut audio = vec![0.0f32; samples(30.0)];
        for sample in audio
            .iter_mut()
            .skip(18 * SAMPLE_RATE as usize)
            .take(samples(1.5))
        {
            *sample = 1.0;
        }
        let pieces = group_pieces_with(&settings, &audio, &[(samples(20.0), samples(21.0))]);
        let start = pieces[0].0;
        assert!(
            start >= samples(19.5),
            "start {}",
            start as f32 / SAMPLE_RATE as f32
        );
        assert!(start <= samples(20.0));
    }

    #[test]
    fn speech_gaps_find_empty_regions_and_word_holes() {
        let rate = SAMPLE_RATE as f32;
        let word = |s: f32, e: f32| Word {
            w: "word".to_string(),
            s: s as f64,
            e: e as f64,
        };
        let words = vec![word(10.0, 10.5), word(15.0, 15.5), word(30.0, 30.5)];
        // One region 10-16 s (a 4.5 s hole between 10.5 and 15), one region
        // 25-40 s (a 5 s hole at its start and a 9.5 s hole at its end).
        let segments = [
            (samples(10.0), samples(16.0)),
            (samples(25.0), samples(40.0)),
        ];
        let gaps = speech_gaps(&segments, &words, 3.0);
        assert_eq!(
            gaps,
            vec![
                (samples(10.5), samples(15.0)),
                (samples(25.0), samples(30.0)),
                (samples(30.5), samples(40.0)),
            ]
        );
        // Under the largest hole nothing is reported.
        assert!(speech_gaps(&segments, &words, 9.6).is_empty());
        let _ = rate;
    }

    #[test]
    fn speech_gaps_merge_overlapping_regions() {
        let words = vec![Word {
            w: "word".to_string(),
            s: 14.0,
            e: 14.5,
        }];
        let segments = [
            (samples(10.0), samples(20.0)),
            (samples(10.0), samples(20.0)),
        ];
        let gaps = speech_gaps(&segments, &words, 3.0);
        assert_eq!(
            gaps,
            vec![
                (samples(10.0), samples(14.0)),
                (samples(14.5), samples(20.0))
            ]
        );
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
