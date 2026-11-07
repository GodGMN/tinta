//! Port of `Echo.swift`: drops microphone words that are residual speaker echo.
//!
//! Measured on MacBook Pro speakers with voice processing on, echo words reach at
//! most 13% of the remote level at the same time. Speech into the microphone is
//! louder. The Rust engine keeps the same levels and the same decision so the
//! result matches the Swift engine's.

use crate::audio::SAMPLE_RATE;
use crate::types::Word;

/// Below this remote RMS the remote track is silent and nothing is echo.
pub const REMOTE_ACTIVE: f32 = 0.01;
/// A microphone RMS below this fraction of the remote RMS is echo.
pub const MAX_RATIO: f32 = 0.2;

/// Root mean square of `samples[start, end)` in seconds, with Swift's clamping.
pub fn rms(samples: &[f32], start: f64, end: f64) -> f32 {
    let rate = SAMPLE_RATE as f64;
    let a = (start * rate) as i64;
    let b = (end * rate) as i64;
    let a = a.max(0);
    let b = b.min(samples.len() as i64);
    if b <= a {
        return 0.0;
    }
    let slice = &samples[a as usize..b as usize];
    let sum: f32 = slice.iter().map(|sample| sample * sample).sum();
    (sum / slice.len() as f32).sqrt()
}

/// The Swift decision: remote must be audible and the microphone must be much
/// quieter than it at the same time.
pub fn is_echo(mic: &[f32], remote: &[f32], start: f64, end: f64) -> bool {
    let remote_level = rms(remote, start, end);
    if remote_level <= REMOTE_ACTIVE {
        return false;
    }
    rms(mic, start, end) < remote_level * MAX_RATIO
}

/// The level form, used by tests and a future live pass.
pub fn is_echo_levels(mic_level: f32, remote_level: f32) -> bool {
    remote_level > REMOTE_ACTIVE && mic_level < remote_level * MAX_RATIO
}

/// Keeps the words that are not echo. `remote` may be shorter than `mic`; a word
/// past the end of `remote` has no remote level and is kept.
pub fn filter(mic: &[f32], remote: &[f32], words: Vec<Word>) -> Vec<Word> {
    words
        .into_iter()
        .filter(|word| !is_echo(mic, remote, word.s, word.e))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fixtures mirror the Swift logic: remote speech at 0.1 RMS and a
    /// microphone at 0.01 (10%) is echo; 0.02 (20%) is not; silence is not.
    fn constant(value: f32, samples: usize) -> Vec<f32> {
        vec![value; samples]
    }

    fn word(text: &str, start: f64, end: f64) -> Word {
        Word {
            w: text.to_string(),
            s: start,
            e: end,
        }
    }

    #[test]
    fn rms_of_a_constant_is_the_constant() {
        let samples = constant(0.5, 16_000);
        assert!((rms(&samples, 0.0, 1.0) - 0.5).abs() < 1e-6);
        assert!((rms(&samples, 0.5, 1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn rms_clamps_like_swift() {
        let samples = constant(0.5, 1_000);
        assert_eq!(rms(&samples, -1.0, 0.0), 0.0);
        assert!((rms(&samples, -1.0, 1.0) - 0.5).abs() < 1e-6);
        assert_eq!(rms(&samples, 1.0, 2.0), 0.0);
        assert_eq!(rms(&samples, 0.05, 0.01), 0.0);
    }

    #[test]
    fn a_quiet_microphone_under_loud_remote_is_echo() {
        let remote = constant(0.1, 16_000);
        let mic = constant(0.01, 16_000);
        assert!(is_echo(&mic, &remote, 0.0, 1.0));
        assert!(is_echo_levels(0.01, 0.1));
    }

    #[test]
    fn the_boundary_is_strict() {
        let remote = constant(0.1, 16_000);
        // Above 20% the word is speech; below it the decision stays strict.
        let loud = constant(0.021, 16_000);
        assert!(!is_echo(&loud, &remote, 0.0, 1.0));
        assert!(!is_echo_levels(0.021, 0.1));
        assert!(is_echo_levels(0.019, 0.1));
    }

    #[test]
    fn silent_remote_never_drops_a_word() {
        let remote = constant(0.0, 16_000);
        let mic = constant(0.0, 16_000);
        assert!(!is_echo(&mic, &remote, 0.0, 1.0));
        // At or below the activation level the ratio does not matter.
        assert!(!is_echo_levels(0.0, REMOTE_ACTIVE));
    }

    #[test]
    fn filter_keeps_loud_words_and_drops_echo() {
        let remote = constant(0.1, 32_000);
        let mut mic = constant(0.0, 32_000);
        // The second second is loud speech into the microphone.
        mic[16_000..].fill(0.1);
        let words = vec![word("echo", 0.0, 1.0), word("speech", 1.0, 2.0)];
        let kept = filter(&mic, &remote, words);
        assert_eq!(kept.len(), 1);
        assert_eq!(kept[0].w, "speech");
    }
}
