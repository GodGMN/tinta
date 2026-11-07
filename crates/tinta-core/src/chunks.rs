//! Encrypted audio chunks, byte-compatible with the Swift engine's `ChunkStore.swift`.
//!
//! A chunk file starts with a 16-byte header followed by a CryptoKit-compatible
//! AES-GCM combined box:
//!
//! | Offset | Size | Content |
//! |---|---|---|
//! | 0 | 4 | ASCII magic `TNTC` |
//! | 4 | 8 | start sample on the meeting clock, `i64` little-endian |
//! | 12 | 4 | sample count, `u32` little-endian |
//! | 16 | rest | 12-byte nonce ‖ ciphertext ‖ 16-byte tag |
//!
//! The plaintext is `count` signed 16-bit PCM samples, little-endian, mono, 16 kHz.
//! Every chunk is sealed on its own with a fresh random nonce and no associated data.
//! The header is not authenticated, exactly as in Swift.
//!
//! Files live under `<meeting dir>/audio/<track>/%08d.tntc` and are written through a
//! `.part` file followed by an atomic rename.

use std::fs;
use std::path::{Path, PathBuf};

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use anyhow::{anyhow, bail, Context, Result};
use rand::rngs::OsRng;
use rand::RngCore;

pub const SAMPLE_RATE: u32 = 16_000;
/// A chunk is flushed when its buffer reaches 5 seconds of audio.
pub const CHUNK_SAMPLES: usize = 5 * SAMPLE_RATE as usize;
/// A jump larger than 100 ms starts a new chunk and leaves a gap.
pub const GAP_SAMPLES: i64 = (SAMPLE_RATE / 10) as i64;

pub const MAGIC: [u8; 4] = *b"TNTC";
pub const HEADER_LEN: usize = 16;
const NONCE_LEN: usize = 12;
const TAG_LEN: usize = 16;

/// The two capture tracks, stored in their own directories.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Track {
    Mic,
    Remote,
}

impl Track {
    pub const ALL: [Track; 2] = [Track::Mic, Track::Remote];

    pub fn dir_name(self) -> &'static str {
        match self {
            Track::Mic => "mic",
            Track::Remote => "remote",
        }
    }

    /// `<meeting dir>/audio/<track>`.
    pub fn directory(self, meeting_dir: &Path) -> PathBuf {
        meeting_dir.join("audio").join(self.dir_name())
    }
}

/// One decrypted chunk: samples on the meeting clock, starting at `start`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub start: i64,
    pub samples: Vec<i16>,
}

/// Writes encrypted chunks for one track. Mirrors `ChunkWriter` in Swift.
pub struct ChunkWriter {
    directory: PathBuf,
    key: [u8; 32],
    track: Track,
    buffer: Vec<i16>,
    buffer_start: i64,
    next_index: Option<i64>,
    file_counter: u32,
}

impl ChunkWriter {
    /// Creates `<meeting dir>/audio/<track>` if needed. The file counter starts at the
    /// number of entries already in that directory, as in Swift.
    pub fn new(meeting_dir: &Path, key: &[u8; 32], track: Track) -> Result<Self> {
        let directory = track.directory(meeting_dir);
        fs::create_dir_all(&directory)
            .with_context(|| format!("cannot create {}", directory.display()))?;
        let file_counter = match fs::read_dir(&directory) {
            Ok(entries) => entries.filter_map(|entry| entry.ok()).count() as u32,
            Err(_) => 0,
        };
        Ok(Self {
            directory,
            key: *key,
            track,
            buffer: Vec::new(),
            buffer_start: 0,
            next_index: None,
            file_counter,
        })
    }

    pub fn track(&self) -> Track {
        self.track
    }

    pub fn directory(&self) -> &Path {
        &self.directory
    }

    /// Appends samples that start at `start_index` on the meeting clock. A jump of more
    /// than 100 ms flushes the current chunk and starts a new one; a smaller drift is
    /// snapped to the expected next index.
    /// A failed write at a gap still appends the new samples, like Swift; the error is
    /// returned after they are buffered.
    pub fn append(&mut self, samples: &[f32], start_index: i64) -> Result<()> {
        let mut start = start_index;
        let mut gap_result = Ok(());
        if let Some(next_index) = self.next_index {
            let drift = start - next_index;
            if drift > GAP_SAMPLES {
                gap_result = self.flush();
            } else {
                start = next_index;
            }
        }
        if self.buffer.is_empty() {
            self.buffer_start = start;
        }
        self.buffer.reserve(samples.len());
        self.buffer.extend(samples.iter().copied().map(f32_to_i16));
        self.next_index = Some(start + samples.len() as i64);
        if self.buffer.len() >= CHUNK_SAMPLES {
            self.flush()?;
        }
        gap_result
    }

    /// Flushes the buffered samples to a chunk file. A failed write drops the buffered
    /// samples and returns the error, like Swift's `flushLocked` drops the chunk.
    pub fn flush(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        let result = self.write_chunk();
        self.buffer.clear();
        result
    }

    fn write_chunk(&mut self) -> Result<()> {
        let bytes = seal(self.buffer_start, &self.buffer, &self.key)?;
        let name = format!("{:08}.tntc", self.file_counter);
        let part = self.directory.join(format!("{name}.part"));
        let path = self.directory.join(&name);
        // Swift's moveItem fails on an existing file; never overwrite a chunk.
        if path.exists() {
            bail!("chunk {} already exists", path.display());
        }
        fs::write(&part, &bytes).with_context(|| format!("cannot write {}", part.display()))?;
        fs::rename(&part, &path)
            .with_context(|| format!("cannot rename {} to {}", part.display(), path.display()))?;
        self.file_counter += 1;
        Ok(())
    }
}

/// The `*.tntc` files of a track, sorted by name. A missing track directory has
/// no chunks.
fn chunk_names(meeting_dir: &Path, track: Track) -> Result<Vec<(String, PathBuf)>> {
    let directory = track.directory(meeting_dir);
    if !directory.exists() {
        return Ok(Vec::new());
    }
    let mut names = Vec::new();
    for entry in
        fs::read_dir(&directory).with_context(|| format!("cannot read {}", directory.display()))?
    {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".tntc") {
            names.push((name, entry.path()));
        }
    }
    names.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(names)
}

/// Decrypts every `*.tntc` file of a track. A missing track directory is an empty
/// track. The sample count in the header is ignored, exactly as `ChunkReader.chunks`.
pub fn read_chunks(meeting_dir: &Path, key: &[u8; 32], track: Track) -> Result<Vec<Chunk>> {
    let names = chunk_names(meeting_dir, track)?;
    let mut chunks = Vec::with_capacity(names.len());
    for (name, path) in names {
        let bytes = fs::read(&path).with_context(|| format!("cannot read {}", path.display()))?;
        chunks.push(open_chunk(&name, &bytes, key)?);
    }
    Ok(chunks)
}

/// Reads a whole track on the meeting clock, filling every gap with silence.
///
/// Chunks are decrypted and copied one at a time, so the f32 track is the only
/// large allocation: the i16 plaintext of a single 5 s chunk is freed before the
/// next file is read. At the 12 h cap that is 2.8 GB per track instead of 2.8 GB
/// plus 1.4 GB of decrypted chunks.
pub fn read_track(meeting_dir: &Path, key: &[u8; 32], track: Track) -> Result<Vec<f32>> {
    let mut output: Vec<f32> = Vec::new();
    for (name, path) in chunk_names(meeting_dir, track)? {
        let bytes = fs::read(&path).with_context(|| format!("cannot read {}", path.display()))?;
        let chunk = open_chunk(&name, &bytes, key)?;
        // The header is not authenticated: a corrupted start must not allocate gigabytes.
        let end = chunk.start.saturating_add(chunk.samples.len() as i64);
        if end > MAX_TRACK_SAMPLES {
            bail!("track is longer than 12 hours ({end} samples); a chunk header is corrupt");
        }
        if chunk.start >= 0 && (chunk.start as usize) > output.len() {
            output.resize(chunk.start as usize, 0.0);
        }
        for (index, &sample) in chunk.samples.iter().enumerate() {
            let position = chunk.start.saturating_add(index as i64);
            if let Ok(position) = usize::try_from(position) {
                if position >= output.len() {
                    output.resize(position + 1, 0.0);
                }
                output[position] = i16_to_f32(sample);
            }
        }
    }
    Ok(output)
}

/// The longest track the engine accepts: 12 hours at 16 kHz. `read_track`
/// rejects a longer one and `import_audio` stops at the same limit.
pub const MAX_TRACK_SAMPLES: i64 = SAMPLE_RATE as i64 * 60 * 60 * 12;

/// Converts a float sample to the stored 16-bit PCM value, clamping to `[-1, 1]`.
/// NaN becomes 0 here (Rust's saturating cast); Swift stores 32767. Both are noise.
pub fn f32_to_i16(sample: f32) -> i16 {
    (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
}

/// Converts a stored 16-bit PCM value back to a float sample.
pub fn i16_to_f32(sample: i16) -> f32 {
    sample as f32 * (1.0 / i16::MAX as f32)
}

/// Seals one chunk with a fresh random nonce.
fn seal(start: i64, samples: &[i16], key: &[u8; 32]) -> Result<Vec<u8>> {
    let mut nonce = [0u8; NONCE_LEN];
    OsRng.fill_bytes(&mut nonce);
    seal_with_nonce(start, samples, key, nonce)
}

/// Seals one chunk with the given nonce. Tests use it for known-answer vectors; the
/// writer always passes a fresh random nonce.
fn seal_with_nonce(
    start: i64,
    samples: &[i16],
    key: &[u8; 32],
    nonce: [u8; NONCE_LEN],
) -> Result<Vec<u8>> {
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| anyhow!("invalid audio key"))?;
    let nonce = Nonce::from_slice(&nonce);
    let mut plaintext = Vec::with_capacity(samples.len() * 2);
    for sample in samples {
        plaintext.extend_from_slice(&sample.to_le_bytes());
    }
    let ciphertext = cipher
        .encrypt(nonce, plaintext.as_slice())
        .map_err(|_| anyhow!("chunk seal failed"))?;

    let mut file = Vec::with_capacity(HEADER_LEN + NONCE_LEN + ciphertext.len());
    file.extend_from_slice(&MAGIC);
    file.extend_from_slice(&start.to_le_bytes());
    file.extend_from_slice(&(samples.len() as u32).to_le_bytes());
    file.extend_from_slice(nonce);
    file.extend_from_slice(&ciphertext);
    Ok(file)
}

/// Decrypts one chunk file. `count` is deliberately not read from the header.
fn open_chunk(name: &str, bytes: &[u8], key: &[u8; 32]) -> Result<Chunk> {
    if bytes.len() <= HEADER_LEN || bytes[..4] != MAGIC {
        bail!("invalid chunk {name}");
    }
    let combined = &bytes[HEADER_LEN..];
    if combined.len() < NONCE_LEN + TAG_LEN {
        bail!("invalid chunk {name}");
    }
    let start = i64::from_le_bytes(bytes[4..12].try_into().expect("8 bytes"));
    let (nonce, ciphertext) = combined.split_at(NONCE_LEN);
    let cipher = Aes256Gcm::new_from_slice(key).map_err(|_| anyhow!("invalid audio key"))?;
    let plaintext = cipher
        .decrypt(Nonce::from_slice(nonce), ciphertext)
        .map_err(|_| anyhow!("cannot decrypt chunk {name}"))?;
    let samples = plaintext
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[low, high]| i16::from_le_bytes([low, high]))
        .collect();
    Ok(Chunk { start, samples })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn key() -> [u8; 32] {
        let mut key = [0u8; 32];
        for (index, byte) in key.iter_mut().enumerate() {
            *byte = index as u8;
        }
        key
    }

    fn write_chunk_file(dir: &Path, name: &str, bytes: &[u8]) {
        let track_dir = dir.join("audio/mic");
        fs::create_dir_all(&track_dir).unwrap();
        fs::write(track_dir.join(name), bytes).unwrap();
    }

    fn read_track_file(dir: &Path, name: &str) -> Vec<u8> {
        fs::read(dir.join("audio/mic").join(name)).unwrap()
    }

    /// AES-256-GCM known answer from the GCM specification (test case 15), used as the
    /// plaintext of a TNTC chunk so both the header and the combined box are pinned.
    #[test]
    fn known_answer_matches_gcm_vector_and_header() {
        let key: [u8; 32] =
            hex::decode("feffe9928665731c6d6a8f9467308308feffe9928665731c6d6a8f9467308308")
                .unwrap()
                .try_into()
                .unwrap();
        let nonce: [u8; 12] = hex::decode("cafebabefacedbaddecaf888")
            .unwrap()
            .try_into()
            .unwrap();
        let plaintext =
            hex::decode("d9313225f88406e5a55909c5aff5269a86a7a9531534f7da2e4c303d8a318a721c3c0c95956809532fcf0e2449a6b525b16aedf5aa0de657ba637b39")
                .unwrap();
        let samples: Vec<i16> = plaintext
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&[low, high]| i16::from_le_bytes([low, high]))
            .collect();

        let start = 0x0102_0304_0506_0708i64;
        let file = seal_with_nonce(start, &samples, &key, nonce).unwrap();

        let mut expected = Vec::new();
        expected.extend_from_slice(b"TNTC");
        expected.extend_from_slice(&start.to_le_bytes());
        expected.extend_from_slice(&(samples.len() as u32).to_le_bytes());
        expected.extend_from_slice(&nonce);
        expected.extend_from_slice(
            &hex::decode("522dc1f099567d07f47f37a32a84427d643a8cdcbfe5c0c97598a2bd2555d1aa8cb08e48590dbb3da7b08b1056828838c5f61e6393ba7a0abcc9f662")
                .unwrap(),
        );
        expected.extend_from_slice(&hex::decode("eb9f796c8d356fc31a8433884b696f4f").unwrap());

        assert_eq!(file, expected);
        assert_eq!(&file[0..4], b"TNTC");
        assert_eq!(&file[4..12], &[8, 7, 6, 5, 4, 3, 2, 1]);
        assert_eq!(&file[12..16], &(30u32).to_le_bytes());
    }

    #[test]
    fn round_trip_preserves_samples_and_names() {
        let dir = TempDir::new().unwrap();
        let key = key();
        let mut writer = ChunkWriter::new(dir.path(), &key, Track::Mic).unwrap();
        assert!(dir.path().join("audio/mic").is_dir());

        writer
            .append(&[0.0, 0.5, -0.5, 1.0, -1.0, 0.25], 0)
            .unwrap();
        writer.flush().unwrap();

        assert_eq!(
            fs::read_dir(dir.path().join("audio/mic"))
                .unwrap()
                .map(|entry| entry.unwrap().file_name().into_string().unwrap())
                .collect::<Vec<_>>(),
            vec!["00000000.tntc"]
        );

        let samples = read_track(dir.path(), &key, Track::Mic).unwrap();
        assert_eq!(
            samples,
            vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25]
                .into_iter()
                .map(|sample| f32_to_i16(sample) as f32 * (1.0 / i16::MAX as f32))
                .collect::<Vec<_>>()
        );
        assert_eq!(f32_to_i16(1.0), i16::MAX);
        assert_eq!(f32_to_i16(-1.0), -i16::MAX);
        assert_eq!(f32_to_i16(2.0), i16::MAX);
        assert_eq!(f32_to_i16(-2.0), -i16::MAX);
    }

    #[test]
    fn wrong_key_is_rejected() {
        let dir = TempDir::new().unwrap();
        let mut writer = ChunkWriter::new(dir.path(), &key(), Track::Mic).unwrap();
        writer.append(&[0.5; 32], 0).unwrap();
        writer.flush().unwrap();

        let mut other = key();
        other[0] ^= 0xff;
        assert!(read_chunks(dir.path(), &other, Track::Mic).is_err());
    }

    #[test]
    fn corrupted_ciphertext_and_tag_are_rejected() {
        for corrupt in ["ciphertext", "tag"] {
            let dir = TempDir::new().unwrap();
            let mut writer = ChunkWriter::new(dir.path(), &key(), Track::Mic).unwrap();
            writer.append(&[0.5; 32], 0).unwrap();
            writer.flush().unwrap();

            let mut bytes = read_track_file(dir.path(), "00000000.tntc");
            match corrupt {
                "ciphertext" => bytes[HEADER_LEN + NONCE_LEN] ^= 0x01,
                _ => *bytes.last_mut().unwrap() ^= 0x01,
            }
            write_chunk_file(dir.path(), "00000000.tntc", &bytes);

            assert!(
                read_chunks(dir.path(), &key(), Track::Mic).is_err(),
                "{corrupt} corruption must be rejected"
            );
        }
    }

    #[test]
    fn truncated_and_invalid_files_are_rejected() {
        let dir = TempDir::new().unwrap();

        write_chunk_file(dir.path(), "00000000.tntc", b"TNTC");
        assert!(read_chunks(dir.path(), &key(), Track::Mic).is_err());

        let mut header_only = Vec::new();
        header_only.extend_from_slice(b"TNTC");
        header_only.extend_from_slice(&0i64.to_le_bytes());
        header_only.extend_from_slice(&1u32.to_le_bytes());
        write_chunk_file(dir.path(), "00000000.tntc", &header_only);
        assert!(read_chunks(dir.path(), &key(), Track::Mic).is_err());

        let mut short_box = header_only;
        short_box.extend_from_slice(&[0u8; 12]);
        write_chunk_file(dir.path(), "00000000.tntc", &short_box);
        assert!(read_chunks(dir.path(), &key(), Track::Mic).is_err());

        let mut bad_magic = vec![b'X'; HEADER_LEN];
        bad_magic.extend_from_slice(&[0u8; NONCE_LEN + TAG_LEN]);
        write_chunk_file(dir.path(), "00000000.tntc", &bad_magic);
        assert!(read_chunks(dir.path(), &key(), Track::Mic).is_err());
    }

    #[test]
    fn gaps_become_silence() {
        let dir = TempDir::new().unwrap();
        let key = key();
        let mut writer = ChunkWriter::new(dir.path(), &key, Track::Mic).unwrap();
        writer.append(&[1.0; 100], 0).unwrap();
        writer.append(&[-1.0; 100], 3_000).unwrap();
        writer.flush().unwrap();

        let chunks = read_chunks(dir.path(), &key, Track::Mic).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].start, 0);
        assert_eq!(chunks[1].start, 3_000);

        let samples = read_track(dir.path(), &key, Track::Mic).unwrap();
        assert_eq!(samples.len(), 3_100);
        assert!(samples[..100]
            .iter()
            .all(|sample| *sample == i16_to_f32(i16::MAX)));
        assert!(samples[100..3_000].iter().all(|sample| *sample == 0.0));
        assert!(samples[3_000..]
            .iter()
            .all(|sample| *sample == i16_to_f32(-i16::MAX)));
    }

    #[test]
    fn append_snaps_small_drift_and_flushes_on_a_gap() {
        let dir = TempDir::new().unwrap();
        let key = key();
        let mut writer = ChunkWriter::new(dir.path(), &key, Track::Mic).unwrap();
        writer.append(&[0.0; 100], 0).unwrap();
        // A drift of 1000 samples is below the 1600-sample gap, so it is snapped.
        writer.append(&[0.0; 100], 1_100).unwrap();
        writer.flush().unwrap();
        // 3000 is far past the expected next index (200), so this starts a new chunk.
        writer.append(&[0.0; 100], 3_000).unwrap();
        writer.flush().unwrap();

        let chunks = read_chunks(dir.path(), &key, Track::Mic).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!((chunks[0].start, chunks[0].samples.len()), (0, 200));
        assert_eq!((chunks[1].start, chunks[1].samples.len()), (3_000, 100));
    }

    #[test]
    fn flushes_at_five_seconds_and_continues_the_file_counter() {
        let dir = TempDir::new().unwrap();
        let key = key();
        // Any existing entry counts, like contentsOfDirectory in Swift.
        let track_dir = dir.path().join("audio/mic");
        fs::create_dir_all(&track_dir).unwrap();
        fs::write(track_dir.join("notes.txt"), b"keep").unwrap();

        let mut writer = ChunkWriter::new(dir.path(), &key, Track::Mic).unwrap();
        writer.append(&[0.0; CHUNK_SAMPLES - 1], 0).unwrap();
        assert!(!track_dir.join("00000001.tntc").exists());
        writer.append(&[0.0], (CHUNK_SAMPLES - 1) as i64).unwrap();

        let names: Vec<_> = fs::read_dir(&track_dir)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        assert!(names.contains(&"00000001.tntc".to_string()));
        assert!(names.contains(&"notes.txt".to_string()));
        assert!(!names.iter().any(|name| name.ends_with(".part")));

        let chunks = read_chunks(dir.path(), &key, Track::Mic).unwrap();
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].samples.len(), CHUNK_SAMPLES);
    }

    #[test]
    fn missing_track_directory_is_an_empty_track() {
        let dir = TempDir::new().unwrap();
        assert!(read_chunks(dir.path(), &key(), Track::Remote)
            .unwrap()
            .is_empty());
        assert!(read_track(dir.path(), &key(), Track::Remote)
            .unwrap()
            .is_empty());
    }

    #[test]
    fn writer_never_overwrites_an_existing_chunk() {
        let dir = TempDir::new().unwrap();
        let mut writer = ChunkWriter::new(dir.path(), &key(), Track::Mic).unwrap();
        let existing = writer.directory().join("00000000.tntc");
        fs::write(&existing, b"keep").unwrap();
        writer.append(&[0.5; 10], 0).unwrap();
        assert!(writer.flush().is_err());
        assert_eq!(fs::read(&existing).unwrap(), b"keep");
    }

    #[test]
    fn corrupt_start_is_rejected_instead_of_allocating() {
        let dir = TempDir::new().unwrap();
        let file = seal_with_nonce(i64::MAX / 4, &[1, 2, 3], &key(), [7; 12]).unwrap();
        write_chunk_file(dir.path(), "00000000.tntc", &file);
        assert!(read_track(dir.path(), &key(), Track::Mic).is_err());
    }
}
