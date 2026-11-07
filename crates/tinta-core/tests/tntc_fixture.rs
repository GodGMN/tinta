//! Reads the fixture written by `engine/scripts/make-tntc-fixture.swift`, which seals
//! chunks with CryptoKit. Ignored by default because Swift is only available on macOS;
//! the `tntc-swift-fixture` CI job runs it with `TINTA_TNTC_FIXTURE_DIR` set.

use std::path::Path;

use tinta_core::chunks::{i16_to_f32, read_chunks, read_track, Track};

fn fixture_dir() -> std::path::PathBuf {
    let dir = std::env::var("TINTA_TNTC_FIXTURE_DIR")
        .expect("set TINTA_TNTC_FIXTURE_DIR to the output directory of make-tntc-fixture.swift");
    std::path::PathBuf::from(dir)
}

fn key() -> [u8; 32] {
    let mut key = [0u8; 32];
    for (index, byte) in key.iter_mut().enumerate() {
        *byte = index as u8;
    }
    key
}

#[test]
#[ignore = "needs the Swift CryptoKit fixture; run with TINTA_TNTC_FIXTURE_DIR"]
fn reads_swift_cryptokit_chunks() {
    let dir = fixture_dir();
    let key = key();

    let chunks = read_chunks(&dir, &key, Track::Mic).unwrap();
    assert_eq!(chunks.len(), 2);
    assert_eq!((chunks[0].start, chunks[0].samples.len()), (0, 1_000));
    assert_eq!((chunks[1].start, chunks[1].samples.len()), (3_000, 500));

    let samples = read_track(&dir, &key, Track::Mic).unwrap();
    assert_eq!(samples.len(), 3_500);
    for (index, sample) in samples.iter().take(1_000).enumerate() {
        let expected = i16_to_f32((index as i16) * 3 - 1_500);
        assert_eq!(*sample, expected, "first chunk sample {index}");
    }
    for (index, sample) in samples[1_000..3_000].iter().enumerate() {
        assert_eq!(*sample, 0.0, "gap sample {}", 1_000 + index);
    }
    for (index, sample) in samples[3_000..].iter().enumerate() {
        let expected = i16_to_f32(2_000 - (index as i16) * 2);
        assert_eq!(*sample, expected, "second chunk sample {index}");
    }

    // The other track was never written.
    assert!(read_track(Path::new(&dir), &key, Track::Remote)
        .unwrap()
        .is_empty());
}
