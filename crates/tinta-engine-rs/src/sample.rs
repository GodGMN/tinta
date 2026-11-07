//! `sample`: returns a WAV of a slice of one track, for the speaker samples in
//! the meeting view.
//!
//! Mirrors `Engine.sample` in Swift: read the whole track, clamp `[s, e)` at
//! 16 kHz, write the canonical 44-byte PCM header and base64-encode it. `s > e`,
//! negative values and an empty track all produce a valid zero-length WAV.

use anyhow::Result;
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde_json::{json, Value};
use std::path::Path;

use crate::audio::{slice_seconds, wav};
use tinta_core::chunks::{read_track, Track};

pub fn sample(dir: &Path, key: &[u8; 32], track: Track, start: f64, end: f64) -> Result<Value> {
    let samples = read_track(dir, key, track)?;
    let slice = slice_seconds(&samples, start, end);
    Ok(json!({ "wav_base64": STANDARD.encode(wav(slice)) }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tinta_core::chunks::ChunkWriter;

    fn key() -> [u8; 32] {
        [7u8; 32]
    }

    fn meeting() -> (TempDir, [u8; 32]) {
        let dir = TempDir::new().unwrap();
        let key = key();
        let mut writer = ChunkWriter::new(dir.path(), &key, Track::Mic).unwrap();
        let samples: Vec<f32> = (0..32_000)
            .map(|index| ((index % 100) as f32 - 50.0) / 100.0)
            .collect();
        writer.append(&samples, 0).unwrap();
        writer.flush().unwrap();
        (dir, key)
    }

    fn decode(value: &Value) -> Vec<u8> {
        STANDARD
            .decode(value["wav_base64"].as_str().unwrap())
            .unwrap()
    }

    #[test]
    fn returns_a_wav_for_the_requested_window() {
        let (dir, key) = meeting();
        let value = sample(dir.path(), &key, Track::Mic, 0.5, 1.5).unwrap();
        let wav = decode(&value);
        assert_eq!(wav.len(), 44 + 16_000 * 2);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[40..44], &32_000u32.to_le_bytes());
    }

    #[test]
    fn clamps_odd_ranges_to_a_valid_wav() {
        let (dir, key) = meeting();
        for (start, end, expected) in [(-1.0, 0.001, 16usize), (3.0, 4.0, 0), (1.0, 0.5, 0)] {
            let value = sample(dir.path(), &key, Track::Mic, start, end).unwrap();
            let wav = decode(&value);
            assert_eq!(wav.len(), 44 + expected * 2, "{start}..{end}");
            assert_eq!(&wav[40..44], &(expected as u32 * 2).to_le_bytes());
        }
    }

    #[test]
    fn an_empty_track_is_a_zero_length_wav() {
        let dir = TempDir::new().unwrap();
        let value = sample(dir.path(), &key(), Track::Remote, 0.0, 1.0).unwrap();
        assert_eq!(decode(&value).len(), 44);
    }
}
