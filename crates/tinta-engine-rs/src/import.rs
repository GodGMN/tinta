//! `import_audio`: imports a local file as the **remote** track.
//!
//! Decodes wav/mp3/m4a(aac)/ogg/flac with symphonia, downmixes to mono,
//! resamples to 16 kHz with rubato and writes TNTC chunks with
//! `tinta_core::chunks::ChunkWriter`, starting at sample 0. Only
//! `audio/remote/*.tntc` is written; there is no session JSON and no database
//! change, exactly like `Engine.importAudio` in Swift.
//!
//! A chunk write failure is an `error` event, not a command error: the import
//! continues with the next samples, matching `ChunkWriter.flushLocked`.

use anyhow::{bail, Result};
use serde_json::{json, Value};
use std::path::Path;

use crate::audio::{AudioDecoder, Resampler48, SAMPLE_RATE};
use crate::protocol::Output;
use tinta_core::chunks::{ChunkWriter, Track};

pub fn import_audio(dir: &Path, key: &[u8; 32], path: &Path, out: &Output) -> Result<Value> {
    let mut decoder = AudioDecoder::open(path)?;
    let source_rate = decoder.sample_rate();
    let mut resampler = if source_rate == SAMPLE_RATE {
        None
    } else {
        Some(Resampler48::new(source_rate)?)
    };
    let mut writer = ChunkWriter::new(dir, key, Track::Remote)?;
    let mut index: i64 = 0;
    let mut input_frames = 0usize;
    let mut output = Vec::new();

    while let Some(chunk) = decoder.next_chunk()? {
        if decoder.sample_rate() != source_rate {
            bail!("the audio file changes sample rate mid-stream");
        }
        input_frames += chunk.len();
        match resampler.as_mut() {
            Some(resampler) => resampler.push(&chunk, &mut output)?,
            None => output.extend_from_slice(&chunk),
        }
        append(&mut writer, &output, &mut index, out);
        output.clear();
    }
    if let Some(resampler) = resampler.as_mut() {
        resampler.finish(&mut output, input_frames)?;
    }
    append(&mut writer, &output, &mut index, out);
    if let Err(error) = writer.flush() {
        let _ = out.event(
            "error",
            json!({ "message": format!("chunk write failed for remote: {error}") }),
        );
    }
    Ok(json!({ "duration": index as f64 / SAMPLE_RATE as f64 }))
}

/// Appends resampled samples and advances the meeting clock. A write failure is
/// reported as an event and does not stop the import.
fn append(writer: &mut ChunkWriter, samples: &[f32], index: &mut i64, out: &Output) {
    if samples.is_empty() {
        return;
    }
    if let Err(error) = writer.append(samples, *index) {
        let _ = out.event(
            "error",
            json!({ "message": format!("chunk write failed for remote: {error}") }),
        );
    }
    *index += samples.len() as i64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use tinta_core::chunks::{read_track, CHUNK_SAMPLES};

    fn key() -> [u8; 32] {
        let mut key = [0u8; 32];
        for (index, byte) in key.iter_mut().enumerate() {
            *byte = index as u8;
        }
        key
    }

    #[test]
    fn writes_the_remote_track_and_reports_its_duration() {
        let dir = TempDir::new().unwrap();
        let source = TempDir::new().unwrap();
        let wav_path = source.path().join("clip.wav");
        let samples: Vec<f32> = (0..32_000)
            .map(|index| (index as f32 / 16_000.0 * 440.0 * std::f32::consts::TAU).sin() * 0.25)
            .collect();
        std::fs::write(&wav_path, crate::audio::wav(&samples)).unwrap();

        let out = Output::new(std::io::sink());
        let result = import_audio(dir.path(), &key(), &wav_path, &out).unwrap();
        assert_eq!(result["duration"], 2.0);

        let track = read_track(dir.path(), &key(), Track::Remote).unwrap();
        assert_eq!(track.len(), 32_000);
        assert!(!dir.path().join("audio/mic").exists());
        // Two seconds fit in one 5 s chunk.
        let chunks = std::fs::read_dir(dir.path().join("audio/remote")).unwrap();
        assert_eq!(chunks.count(), 1);
        assert_eq!(CHUNK_SAMPLES, 80_000);
    }

    #[test]
    fn a_missing_file_is_an_error() {
        let dir = TempDir::new().unwrap();
        let out = Output::new(std::io::sink());
        let error =
            import_audio(dir.path(), &key(), Path::new("/does/not/exist.wav"), &out).unwrap_err();
        assert!(error.to_string().contains("cannot open"), "{error}");
    }
}
