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
use tinta_core::chunks::{ChunkWriter, Track, MAX_TRACK_SAMPLES};

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
        ensure_within_limit(index, output.len())?;
        append(&mut writer, &output, &mut index, out);
        output.clear();
    }
    if let Some(resampler) = resampler.as_mut() {
        resampler.finish(&mut output, input_frames)?;
    }
    ensure_within_limit(index, output.len())?;
    append(&mut writer, &output, &mut index, out);
    if let Err(error) = writer.flush() {
        let _ = out.event(
            "error",
            json!({ "message": format!("chunk write failed for remote: {error}") }),
        );
    }
    Ok(json!({ "duration": index as f64 / SAMPLE_RATE as f64 }))
}

/// Rejects an import that would pass the 12 hour track limit. `read_track` uses
/// the same cap, so without this check the import would succeed and `finalize`
/// would later fail with a misleading "chunk header is corrupt".
fn ensure_within_limit(index: i64, samples: usize) -> Result<()> {
    if index + samples as i64 > MAX_TRACK_SAMPLES {
        bail!(
            "the audio is longer than the 12 hour track limit; split or trim the file before importing it"
        );
    }
    Ok(())
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
    use std::path::PathBuf;
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

    /// Zero-crossing frequency estimate, enough for a clean sine.
    fn dominant_frequency(samples: &[f32]) -> f32 {
        let mut crossings = 0;
        for pair in samples.windows(2) {
            if (pair[0] < 0.0 && pair[1] >= 0.0) || (pair[0] >= 0.0 && pair[1] < 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 * SAMPLE_RATE as f32 / (2.0 * samples.len() as f32)
    }

    fn write_wav(dir: &Path, name: &str, rate: u32, seconds: f32) -> PathBuf {
        let count = (rate as f32 * seconds) as usize;
        let samples: Vec<f32> = (0..count)
            .map(|index| (index as f32 / rate as f32 * 300.0 * std::f32::consts::TAU).sin() * 0.25)
            .collect();
        let path = dir.join(name);
        std::fs::write(&path, crate::audio::wav_at(&samples, rate)).unwrap();
        path
    }

    #[test]
    fn a_44k_import_is_resampled_end_to_end() {
        let dir = TempDir::new().unwrap();
        let source = TempDir::new().unwrap();
        let path = write_wav(source.path(), "tone-44k.wav", 44_100, 2.0);
        let out = Output::new(std::io::sink());
        let result = import_audio(dir.path(), &key(), &path, &out).unwrap();
        assert_eq!(result["duration"], 2.0);
        let track = read_track(dir.path(), &key(), Track::Remote).unwrap();
        assert_eq!(track.len(), 32_000);
        let frequency = dominant_frequency(&track[400..31_600]);
        assert!((frequency - 300.0).abs() < 5.0, "frequency {frequency}");
    }

    #[test]
    fn a_48k_import_is_resampled_end_to_end() {
        let dir = TempDir::new().unwrap();
        let source = TempDir::new().unwrap();
        let path = write_wav(source.path(), "tone-48k.wav", 48_000, 2.0);
        let out = Output::new(std::io::sink());
        let result = import_audio(dir.path(), &key(), &path, &out).unwrap();
        assert_eq!(result["duration"], 2.0);
        let track = read_track(dir.path(), &key(), Track::Remote).unwrap();
        assert_eq!(track.len(), 32_000);
        let frequency = dominant_frequency(&track[400..31_600]);
        assert!((frequency - 300.0).abs() < 5.0, "frequency {frequency}");
    }

    /// The fixture is a 2 s, 44.1 kHz mono MP3 made with
    /// `ffmpeg -f lavfi -i "sine=frequency=300:sample_rate=44100:duration=2"
    /// -ac 1 -c:a libmp3lame -b:a 64k -write_xing 1`.
    /// Gapless trimming removes the 1105-sample encoder delay and the padding,
    /// so the timeline matches the 2 s source. Without it the track is about
    /// 25-40 ms longer and every word time shifts against Swift.
    #[test]
    fn a_short_mp3_keeps_its_source_timeline() {
        let dir = TempDir::new().unwrap();
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tone-2s-44k.mp3");
        let out = Output::new(std::io::sink());
        let result = import_audio(dir.path(), &key(), &path, &out).unwrap();
        let duration = result["duration"].as_f64().unwrap();
        assert!(
            (duration - 2.0).abs() <= 0.002,
            "duration {duration} is not the 2 s source"
        );
        let track = read_track(dir.path(), &key(), Track::Remote).unwrap();
        let expected = 32_000i64;
        assert!(
            (track.len() as i64 - expected).abs() <= 32,
            "length {} is not the 2 s source (32,000 samples)",
            track.len()
        );
        let frequency = dominant_frequency(&track[400..31_600]);
        assert!((frequency - 300.0).abs() < 5.0, "frequency {frequency}");
    }

    #[test]
    fn an_import_past_the_twelve_hour_limit_is_rejected() {
        assert!(ensure_within_limit(MAX_TRACK_SAMPLES - 16, 16).is_ok());
        let error = ensure_within_limit(MAX_TRACK_SAMPLES, 1).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("12 hour"), "{message}");
        assert!(!message.contains("corrupt"), "{message}");
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
