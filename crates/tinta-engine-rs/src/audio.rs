//! Audio decoding, downmixing, resampling and WAV output.
//!
//! `import_audio` opens a local file with symphonia (wav, mp3, m4a/aac, ogg/vorbis,
//! flac), downmixes every channel to mono and resamples to the engine's 16 kHz with
//! rubato. `sample` writes the canonical 44-byte PCM WAV the Swift engine returns.
//!
//! The resampler uses a windowed-sinc filter. Its internal delay is trimmed from
//! the start of the stream and the filter tail is flushed at the end, so the output
//! has the expected length (`round(input * 16000 / rate)`) and starts at the same
//! instant as the input (within the filter's few-millisecond transient).

use anyhow::{bail, Context, Result};
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use std::fs::File;
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{Decoder as SymphoniaDecoder, DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::{FormatOptions, FormatReader};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

/// The engine's sample rate, the same as `ChunkFormat.sampleRate` in Swift.
pub const SAMPLE_RATE: u32 = 16_000;

/// Input frames per resampler call. Smaller chunks keep the streaming latency low.
const RESAMPLE_CHUNK: usize = 4_096;

/// Opens an audio file and decodes it to mono f32 at the source sample rate.
pub struct AudioDecoder {
    format: Box<dyn FormatReader>,
    track_id: u32,
    decoder: Box<dyn SymphoniaDecoder>,
    sample_rate: u32,
    channels: usize,
    total_frames: Option<u64>,
}

impl AudioDecoder {
    /// Probes the file. `hint` only carries the extension; symphonia sniffs the
    /// container anyway, so an extension-less file still opens.
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
        let stream = MediaSourceStream::new(Box::new(file), Default::default());
        let mut hint = Hint::new();
        if let Some(extension) = path.extension().and_then(|value| value.to_str()) {
            hint.with_extension(extension);
        }
        // Gapless trims the encoder delay and padding that AAC and MP3 carry, so
        // the imported timeline starts where the source audio does, like
        // `AVAudioFile` in the Swift engine.
        let options = FormatOptions {
            enable_gapless: true,
            ..Default::default()
        };
        let probed = symphonia::default::get_probe()
            .format(&hint, stream, &options, &MetadataOptions::default())
            .with_context(|| format!("cannot read the audio format of {}", path.display()))?;
        let format = probed.format;
        let track = format
            .default_track()
            .filter(|track| track.codec_params.codec != CODEC_TYPE_NULL)
            .with_context(|| format!("{} has no audio track", path.display()))?;
        let track_id = track.id;
        let decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &DecoderOptions::default())
            .with_context(|| format!("cannot decode {} with the CPU codecs", path.display()))?;
        let sample_rate = track
            .codec_params
            .sample_rate
            .context("the audio file does not declare a sample rate")?;
        let channels = track
            .codec_params
            .channels
            .map(|channels| channels.count())
            .unwrap_or(1)
            .max(1);
        let total_frames = track.codec_params.n_frames;
        Ok(Self {
            format,
            track_id,
            decoder,
            sample_rate,
            channels,
            total_frames,
        })
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Frames the container declares for the stream, when it declares them.
    /// WAV sizes its data chunk, MP3 may carry a Xing frame count; a streamed
    /// container without a duration has none.
    pub fn total_frames(&self) -> Option<u64> {
        self.total_frames
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Reads the next packet of mono samples. Returns `None` at the end of the file.
    /// A packet that does not decode is skipped, matching AVFoundation's tolerance
    /// for a damaged frame.
    pub fn next_chunk(&mut self) -> Result<Option<Vec<f32>>> {
        loop {
            let packet = match self.format.next_packet() {
                Ok(packet) => packet,
                Err(SymphoniaError::IoError(error))
                    if error.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    return Ok(None)
                }
                Err(SymphoniaError::ResetRequired) => {
                    bail!("the audio file changed format mid-stream")
                }
                Err(error) => return Err(error.into()),
            };
            if packet.track_id() != self.track_id {
                continue;
            }
            let decoded = match self.decoder.decode(&packet) {
                Ok(decoded) => decoded,
                Err(SymphoniaError::DecodeError(_)) => continue,
                Err(error) => return Err(error.into()),
            };
            if decoded.frames() == 0 {
                continue;
            }
            let spec = *decoded.spec();
            self.sample_rate = spec.rate;
            self.channels = spec.channels.count().max(1);
            let mut buffer = SampleBuffer::<f32>::new(decoded.capacity() as u64, spec);
            buffer.copy_interleaved_ref(decoded);
            return Ok(Some(downmix(buffer.samples(), self.channels)));
        }
    }
}

/// Averages interleaved frames into one mono channel.
fn downmix(interleaved: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return interleaved.to_vec();
    }
    let scale = 1.0 / channels as f32;
    interleaved
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() * scale)
        .collect()
}

/// Streaming resampler to 16 kHz mono.
pub struct Resampler48 {
    inner: SincFixedIn<f32>,
    ratio: f64,
    pending: Vec<f32>,
    delay: usize,
    trimmed: bool,
    produced: usize,
}

impl Resampler48 {
    pub fn new(source_rate: u32) -> Result<Self> {
        if source_rate == 0 {
            bail!("the audio file declares a zero sample rate");
        }
        let ratio = SAMPLE_RATE as f64 / source_rate as f64;
        let params = SincInterpolationParameters {
            sinc_len: 128,
            f_cutoff: 0.95,
            oversampling_factor: 256,
            interpolation: SincInterpolationType::Linear,
            window: WindowFunction::BlackmanHarris2,
        };
        let inner = SincFixedIn::<f32>::new(ratio, 4.0, params, RESAMPLE_CHUNK, 1)
            .map_err(|error| anyhow::anyhow!("cannot create the resampler: {error}"))?;
        let delay = inner.output_delay();
        Ok(Self {
            inner,
            ratio,
            pending: Vec::new(),
            delay,
            trimmed: false,
            produced: 0,
        })
    }

    /// Number of output samples the finished stream should have.
    pub fn expected_output(&self, input_frames: usize) -> usize {
        (input_frames as f64 * self.ratio).round() as usize
    }

    /// Resamples `input`, appending output samples to `out`.
    pub fn push(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<()> {
        self.pending.extend_from_slice(input);
        while self.pending.len() >= RESAMPLE_CHUNK {
            let chunk: Vec<f32> = self.pending.drain(..RESAMPLE_CHUNK).collect();
            let waves = self
                .inner
                .process(&[chunk], None)
                .map_err(|error| anyhow::anyhow!("cannot resample the audio: {error}"))?;
            self.emit(&waves[0], out);
        }
        Ok(())
    }

    /// Resamples the remaining samples and flushes the filter tail. Call once.
    ///
    /// A streamed caller clears `out` after every push, so the length check and
    /// the trim use [`Resampler48::produced`] (the whole stream) and never
    /// `out.len()` (just the tail). The flush adds up to one chunk of filter
    /// tail; the trim removes it from the tail buffer before it is written.
    pub fn finish(&mut self, out: &mut Vec<f32>, input_frames: usize) -> Result<()> {
        if !self.pending.is_empty() {
            let tail = std::mem::take(&mut self.pending);
            let waves = self
                .inner
                .process_partial(Some(&[tail.as_slice()]), None)
                .map_err(|error| anyhow::anyhow!("cannot resample the audio: {error}"))?;
            self.emit(&waves[0], out);
        }
        let expected = self.expected_output(input_frames);
        // Two zero-fed calls flush the sinc filter's tail.
        let zeros = vec![0.0f32; RESAMPLE_CHUNK];
        for _ in 0..2 {
            if self.produced >= expected {
                break;
            }
            let waves = self
                .inner
                .process(std::slice::from_ref(&zeros), None)
                .map_err(|error| anyhow::anyhow!("cannot flush the resampler: {error}"))?;
            self.emit(&waves[0], out);
        }
        // Trim to `round(input * 16000 / rate)` so a streamed import has the
        // same length as an unstreamed one. The extra samples are always in this
        // tail (one flush chunk at most); `saturating_sub` is only a safety net.
        let extra = self.produced.saturating_sub(expected);
        let keep = out.len().saturating_sub(extra);
        out.truncate(keep);
        self.produced = expected.min(self.produced);
        Ok(())
    }

    fn emit(&mut self, samples: &[f32], out: &mut Vec<f32>) {
        let mut start = 0;
        if !self.trimmed {
            // The first `delay` output samples are the filter's pre-roll.
            start = self.delay.min(samples.len());
            self.trimmed = true;
        }
        out.extend_from_slice(&samples[start..]);
        self.produced += samples.len() - start;
    }
}

/// Resamples a whole buffer, for tests and short clips.
pub fn resample_to_16k(samples: &[f32], source_rate: u32) -> Result<Vec<f32>> {
    if source_rate == SAMPLE_RATE {
        return Ok(samples.to_vec());
    }
    let mut resampler = Resampler48::new(source_rate)?;
    let mut out = Vec::with_capacity(resampler.expected_output(samples.len()));
    resampler.push(samples, &mut out)?;
    resampler.finish(&mut out, samples.len())?;
    Ok(out)
}

/// Writes a canonical 44-byte PCM header and 16-bit little-endian mono samples,
/// byte for byte like `ChunkReader.wav` in `ChunkStore.swift`.
pub fn wav(samples: &[f32]) -> Vec<u8> {
    wav_at(samples, SAMPLE_RATE)
}

/// Same layout at an arbitrary sample rate, for the import tests.
pub(crate) fn wav_at(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let byte_count = (samples.len() * 2) as u32;
    let mut data = Vec::with_capacity(44 + samples.len() * 2);
    data.extend_from_slice(b"RIFF");
    data.extend_from_slice(&(36 + byte_count).to_le_bytes());
    data.extend_from_slice(b"WAVEfmt ");
    data.extend_from_slice(&16u32.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes());
    data.extend_from_slice(&1u16.to_le_bytes());
    data.extend_from_slice(&sample_rate.to_le_bytes());
    data.extend_from_slice(&(sample_rate * 2).to_le_bytes());
    data.extend_from_slice(&2u16.to_le_bytes());
    data.extend_from_slice(&16u16.to_le_bytes());
    data.extend_from_slice(b"data");
    data.extend_from_slice(&byte_count.to_le_bytes());
    for sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let value = (clamped * i16::MAX as f32) as i16;
        data.extend_from_slice(&value.to_le_bytes());
    }
    data
}

/// Slices `[s, e)` seconds with the clamping `Engine.swift` uses:
/// `from = clamp(0, len, s * rate)` and `to = clamp(from, len, e * rate)`.
pub fn slice_seconds(samples: &[f32], start: f64, end: f64) -> &[f32] {
    let rate = SAMPLE_RATE as f64;
    let from = (start * rate) as i64;
    let from = from.clamp(0, samples.len() as i64) as usize;
    let to = (end * rate) as i64;
    let to = to.clamp(from as i64, samples.len() as i64) as usize;
    &samples[from..to]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, seconds: f32, frequency: f32) -> Vec<f32> {
        let count = (rate as f32 * seconds) as usize;
        (0..count)
            .map(|index| {
                (2.0 * std::f32::consts::PI * frequency * index as f32 / rate as f32).sin() * 0.5
            })
            .collect()
    }

    fn dominant_frequency(samples: &[f32], rate: u32) -> f32 {
        // Zero-crossing estimate; enough for a clean sine and no FFT dependency.
        let mut crossings = 0;
        for pair in samples.windows(2) {
            if (pair[0] < 0.0 && pair[1] >= 0.0) || (pair[0] >= 0.0 && pair[1] < 0.0) {
                crossings += 1;
            }
        }
        crossings as f32 * rate as f32 / (2.0 * samples.len() as f32)
    }

    #[test]
    fn resamples_48k_to_16k_with_the_same_tone() {
        let input = sine(48_000, 1.0, 440.0);
        let output = resample_to_16k(&input, 48_000).unwrap();
        assert!(
            (output.len() as i64 - 16_000).abs() <= 2,
            "length {}",
            output.len()
        );
        let peak = output
            .iter()
            .fold(0.0f32, |peak, sample| peak.max(sample.abs()));
        assert!((peak - 0.5).abs() < 0.05, "peak {peak}");
        let frequency = dominant_frequency(&output[400..15_600], 16_000);
        assert!((frequency - 440.0).abs() < 5.0, "frequency {frequency}");
    }

    #[test]
    fn resamples_44k_to_16k_with_the_expected_length() {
        let input = sine(44_100, 2.0, 300.0);
        let output = resample_to_16k(&input, 44_100).unwrap();
        assert!(
            (output.len() as i64 - 32_000).abs() <= 2,
            "length {}",
            output.len()
        );
        let frequency = dominant_frequency(&output[400..31_600], 16_000);
        assert!((frequency - 300.0).abs() < 5.0, "frequency {frequency}");
    }

    /// Pushes like `import_audio`: every packet is emitted and `out` is cleared,
    /// so only `finish` can trim the filter tail.
    fn stream(rate: u32, seconds: f32, packet: usize) -> Vec<f32> {
        let input = sine(rate, seconds, 440.0);
        let mut resampler = Resampler48::new(rate).unwrap();
        let mut out = Vec::new();
        let mut total = Vec::new();
        for chunk in input.chunks(packet) {
            resampler.push(chunk, &mut out).unwrap();
            total.append(&mut out);
        }
        resampler.finish(&mut out, input.len()).unwrap();
        total.append(&mut out);
        total
    }

    #[test]
    fn a_streamed_48k_import_has_exactly_the_expected_length() {
        // 60 s at 48 kHz in 24 ms packets used to produce 963,882 samples
        // (0.24 s of filter tail) instead of 960,000.
        let output = stream(48_000, 60.0, 1_152);
        assert_eq!(output.len(), 960_000);
    }

    #[test]
    fn a_streamed_44k_import_has_exactly_the_expected_length() {
        let input_frames: usize = 44_100 * 60;
        assert_eq!(input_frames * 16_000 / 44_100, 960_000);
        let output = stream(44_100, 60.0, 1_152);
        assert_eq!(output.len(), 960_000);
    }

    #[test]
    fn a_16k_input_passes_through() {
        let input = sine(16_000, 0.5, 500.0);
        assert_eq!(resample_to_16k(&input, 16_000).unwrap(), input);
    }

    #[test]
    fn downmix_averages_channels() {
        assert_eq!(downmix(&[0.0, 1.0, 0.5, 0.5], 2), vec![0.5, 0.5]);
        assert_eq!(downmix(&[0.25, 0.5], 1), vec![0.25, 0.5]);
    }

    #[test]
    fn wav_header_and_samples_match_the_swift_layout() {
        let wav = wav(&[0.0, 1.0, -1.0, 0.5]);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[4..8], &44u32.to_le_bytes());
        assert_eq!(&wav[8..16], b"WAVEfmt ");
        assert_eq!(&wav[16..20], &16u32.to_le_bytes());
        assert_eq!(&wav[20..22], &1u16.to_le_bytes());
        assert_eq!(&wav[22..24], &1u16.to_le_bytes());
        assert_eq!(&wav[24..28], &16_000u32.to_le_bytes());
        assert_eq!(&wav[28..32], &32_000u32.to_le_bytes());
        assert_eq!(&wav[32..34], &2u16.to_le_bytes());
        assert_eq!(&wav[34..36], &16u16.to_le_bytes());
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(&wav[40..44], &8u32.to_le_bytes());
        assert_eq!(&wav[44..46], &0i16.to_le_bytes());
        assert_eq!(&wav[46..48], &i16::MAX.to_le_bytes());
        assert_eq!(&wav[48..50], &(-i16::MAX).to_le_bytes());
        assert_eq!(&wav[50..52], &16_383i16.to_le_bytes());
        assert_eq!(wav.len(), 52);
    }

    #[test]
    fn slices_clamp_like_swift() {
        let samples: Vec<f32> = (0..32_000).map(|value| value as f32).collect();
        assert_eq!(slice_seconds(&samples, 0.0, 1.0).len(), 16_000);
        assert_eq!(slice_seconds(&samples, -1.0, 0.001).len(), 16);
        assert_eq!(slice_seconds(&samples, 1.5, 2.5).len(), 8_000);
        assert_eq!(slice_seconds(&samples, 3.0, 4.0).len(), 0);
        assert_eq!(slice_seconds(&samples, 1.0, 0.5).len(), 0);
    }
}
