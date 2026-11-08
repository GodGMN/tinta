# Test fixture: AMI ES2004b 235.4-248.8 s

`es2004b-235.4-248.8.wav` is an excerpt of the AMI Meeting Corpus.

- Corpus: **AMI Meeting Corpus** (<https://groups.inf.ed.ac.uk/ami/corpus/>)
- Meeting: `ES2004b`, audio channel `Mix-Headset`
- Excerpt: 235.4 s to 248.8 s (13.4 s), trimmed only, no other processing
- Format: 16 kHz, mono, 16-bit PCM WAV (428,878 B)
- License: **Creative Commons Attribution 4.0 International (CC BY 4.0)**,
  <https://creativecommons.org/licenses/by/4.0/>
- Attribution: AMI Meeting Corpus, created by the AMI Consortium
  (<https://groups.inf.ed.ac.uk/ami/corpus/>), CC BY 4.0.

The excerpt is the window task #23 uses as its first gate: the shipped int8
Parakeet TDT 0.6b v3 encoder returns no words on it, while the fp32 encoder and
the requantized encoder return the sentence below. It is committed so the
`asr-requant-portability` CI job can decode the same audio on every runner.

`es2004b-235.4-248.8.ref.txt` is the fp32 reference text for the clip, produced
with `cargo run -p tinta-engine-rs --example decode-clip` against the fp32
Parakeet TDT 0.6b v3 export pinned in `docs/fork/asr-experiments.md`. The
requantized encoder returns the same text on the measurement host.
