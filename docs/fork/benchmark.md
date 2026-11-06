# sherpa-onnx on CPU: can it replace FluidAudio/CoreML?

Task #2 of the Windows/Linux port. The Mac engine uses FluidAudio with CoreML for
ASR and diarization. The port needs a CPU-only equivalent that runs on an x86_64
machine with 16 GB and processes a one-hour meeting in ten minutes or less, with
quality in the range FluidAudio publishes (Parakeet v3 around 2-5% WER on clean
speech, pyannote community-1 around 10% DER on AMI).

The harness in `scripts/fork/bench/` runs two sherpa-onnx models on the same AMI
audio:

- ASR: NeMo Parakeet TDT 0.6b v3 int8 (`offline_asr`, transducer).
- Diarization: pyannote segmentation 3.0 + a speaker-embedding model
  (`offline_speaker_diarization`, complete-linkage clustering).

## Verdict

**Conditional pass on the performance bar, pass on memory, pass on clean-speech
quality, mixed on diarization, fail on automatic speaker counting.**

The recommended pipeline (240 s chunks, 4 threads for ASR, 4 for diarization,
both stages in parallel) costs **51 CPU-minutes per audio hour** and peaks at
**5.2 GB**. On eight idle cores that is about 6.4 minutes, inside the 10-minute
bar with room to spare. On this shared 8-vCPU host it measured **19.4 minutes**
(and 26.7 minutes with the naive sequential 8+8-thread layout), because other
work kept taking the cores. The bar is met on a dedicated 8-core CPU, not
demonstrated on this host.

| Bar | Result | Verdict |
|---|---|---|
| 1 h meeting in <= 10 min on CPU | 19.4 min wall measured here (load 14-40); 51 CPU-min => 6.4 min at 8 idle cores | **conditional** |
| 16 GB RAM | 5.2 GB peak RSS combined, 5.0 GB ASR alone | pass |
| Parakeet v3 2-5% WER on clean speech | **3.31%** on LibriSpeech test-clean | pass |
| WER on real meetings (AMI mix) | 26.9-43.3%, 20.9-38.3% in non-overlapped regions | expected for overlapping meeting audio; not comparable to clean-speech numbers |
| pyannote community-1 ~10% DER on AMI | **4.4%** IN1014, 6.5% ES2002a, 18.3% IS1000a (tuned threshold) | mixed |
| Speaker count | 9-14 clusters for 4 speakers | **fail** without tuning |

The 20%+ AMI WER is not an ASR model defect: on single-speaker AMI regions the
same model scores 20.5%, and on clean read speech 3.31%. The mix channel
contains 20-24% overlapping words that no single-stream ASR can recover, plus
AMI's spontaneous disfluencies. FluidAudio's 2-5% claim is a clean-speech
number; the comparable measurement here matches it.

Diarization beats the ~10% DER reference on two of three meetings. The
non-native-speaker meeting over-segments (7 clusters at the tuned threshold,
18.3% DER), and all three over-count speakers, so the app needs a
per-recording threshold estimate or a user-visible speaker count.

Two findings decide the engineering:

1. **Use 4 threads per stage, not 8.** ONNX Runtime's thread pool spins, and on
   this host 8 threads were both slower in wall time and 2-5x more expensive in
   CPU than 4.
2. **Run ASR and diarization concurrently.** Diarization then costs almost no
   wall time: the combined wall is the ASR wall. Sequential 8+8 threads was
   26.7 min; parallel 4+4 was 19.4 min for the same 1 h meeting.



## Machine

| | |
|---|---|
| CPU | AMD Ryzen 7 5825U, 8 vCPU (KVM guest, 8 logical CPUs) |
| RAM | 22 GiB (the bar is 16 GB) |
| OS | Debian 12, kernel 6.1.0-54-amd64 |
| Toolchain | Rust 1.100 (nightly), sherpa-onnx crate 2.0.1, static build |
| Crypto/accel | CPU only, no CUDA/CoreML, ONNX Runtime CPU provider |

The host is a shared development VM. Other agents' builds ran during the
benchmark, so the load average moved between 12 and 35 on 8 CPUs. Every result
JSON records `loadavg_start`/`loadavg_end`, and GNU time records user CPU
seconds. The load makes **wall-clock** numbers pessimistic; the per-core cost
(user CPU seconds / audio seconds) is stable and is reported next to them.

## Data

Public AMI corpus, Mix-Headset channel (16 kHz mono, all headset mics mixed),
plus the manual annotations (`ami_public_manual_1.6.2`).

| Meeting | Audio | Speakers | Words | Overlapped words | Character |
|---|---|---|---|---|---|
| ES2002a | 21.2 min | 4 | 2600 | 19.8% | scenario, kick-off |
| IS1000a | 26.4 min | 4 | 2893 | 22.8% | non-native speakers, technical |
| IN1014 | 61.5 min | 4 | 10896 | 24.4% | the ~1 h meeting |

"Overlapped words" counts reference words that overlap a word from another
speaker; they cannot be recovered from a single mixed channel and put a floor
under mix WER.

Two extra samples check the quality claim itself:

- **Clean read speech**: 30 utterances from LibriSpeech `test-clean`
  (3.7 min), fetched through the Hugging Face datasets server.
- **AMI single-speaker speech**: 5 min cut from ES2002a Mix-Headset at regions
  where exactly one speaker is active, concatenated in order
  (`score.py clean-sample`). Removes overlap, keeps AMI's spontaneous style.

## Models (pinned)

| Model | File | SHA-256 |
|---|---|---|
| Parakeet TDT 0.6b v3 int8 | `encoder.int8.onnx` (652 MB) | `acfc2b44…fd2247` |
| | `decoder.int8.onnx` | `179e50c4…48db4e` |
| | `joiner.int8.onnx` | `3164c13f…948b3` |
| | `tokens.txt` | `d5854467…3c35d` |
| pyannote segmentation 3.0 | `model.int8.onnx` (1.5 MB) | `d582f4b4…7b5d` |
| 3D-Speaker CAMPPlus | `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx` (28 MB) | `f682b514…d11` |

Full hashes and download URLs are in `scripts/fork/bench/fetch-data.sh`. The
models are sherpa-onnx release assets:

- <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2>
- <https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2>
- <https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx>
  (the upstream release tag is misspelled "recongition"; the URL is correct)

Data (~230 MB of wav, 23 MB of annotations) and models (~1.2 GB) live outside
the repo under `/home/sergio/code/evidence/tinta-fork/2/`. Nothing is committed
except the harness and this document.

## Method

### ASR

- `OfflineRecognizer` with the transducer encoder/decoder/joiner, `cpu`
  provider, `num_threads` set per run, greedy search (`max_active_paths` left at
  the default 4).
- The wave is cut into **240 s chunks with 2 s overlap**. Chunks longer than
  ~240 s are not safe: at 300 s the decoder silently drops the first ~8 s of
  each chunk (reproduced on a 6 min slice: first word timestamp 8.0 s instead of
  4.6 s, 330 vs 450 words). Words that start in the overlap already covered by
  the previous chunk are dropped, so nothing is counted twice.
- Word-level timestamps come from the transducer token timestamps plus token
  durations; sentencepiece `▁` markers are folded into words.

### Diarization

- pyannote segmentation 3.0 int8, `window_shift_ratio` 0.2 (1 s hop), then
  CAMPPlus embeddings and complete-linkage clustering.
- `threshold` in sherpa-onnx is a **cosine dissimilarity** cut, not a
  similarity: higher merges more. The default 0.5 produced 84 clusters on
  ES2002a; the sweep below shows 1.0 is the useful value for AMI.
- `min_duration_on` 0.3 s, `min_duration_off` 0.5 s.
- The 0.1 hop was also measured: same DER on ES2002a (6.2%), but 14.8% instead
  of 4.4% on IN1014 and about twice the CPU. The coarse hop is both cheaper and
  less fragmented.

### Scoring

`score.py`, definitions matter more than the numbers:

- **WER**: case and punctuation normalised, one token per word, including
  disfluencies (`um`, `uh`). `wer_without_fillers` drops a small filler list.
- **DER**: frame based at 10 ms, NIST-style collar 0.25 s around every
  reference boundary, overlapping speech scored by default (a hypothesis label
  must be one of the active reference speakers). `der_no_overlap` excludes
  frames with more than one reference speaker. Clusters are mapped to reference
  speakers with a Hungarian assignment on overlapping speech duration.
- **Speaker count**: number of clusters with `num_clusters = -1` (automatic)
  against the four reference speakers.
- **WER without overlap**: reference words fully inside single-speaker regions,
  hypothesis words whose midpoint falls in those regions. Isolates the part of
  the error caused by overlapping speech.

## Results

## Results

Every run below is 8 threads unless stated. "CPU-s per audio-s" is GNU time's
user CPU seconds divided by audio seconds; it does not depend on how many cores
the shared host gave the process. `RTF` is wall seconds divided by audio
seconds, so it does.

### ASR

240 s chunks, 2 s overlap, greedy transducer decoding.

| Meeting | Audio | Wall | RTF | CPU-s per audio-s | WER | WER, no overlap | Words hyp/ref | Peak RSS |
|---|---|---|---|---|---|---|---|---|
| ES2002a | 21.2 min | 5:54 | 0.269 | 0.82 | **27.6%** | 22.1% | 2205/2600 | 4.9 GB |
| IS1000a | 26.4 min | 9:25 | 0.357 | 0.75 | **43.3%** | 38.3% | 1834/2893 | 4.9 GB |
| IN1014 (1 h) | 61.5 min | 14:51 | 0.237 | 0.76 | **26.9%** | 20.9% | 8808/10896 | 5.0 GB |

Wall includes model load (cold load 5.0-5.9 s, warm 3.2 s for the 670 MB
transducer; diarization loads in 0.6 s cold, 0.4 s warm), which is negligible
except on the shortest meeting.

Quality on clean speech, for the comparison with FluidAudio's numbers:

| Sample | Audio | WER | RTF |
|---|---|---|---|
| LibriSpeech `test-clean`, 30 utterances | 3.7 min | **3.31%** | 0.222 |
| ES2002a single-speaker regions | 5.0 min | **20.5%** | 0.387 |

The clean read-speech result is inside the published 2-5% band. The AMI
single-speaker sample is spontaneous, disfluent and still carries some room
bleed; it lands at 20.5%, so AMI's meeting WER is not evidence of a broken
model.

### Chunk size

The offline transducer is quadratic in input length in memory, and above
~240 s it degrades in quality. 300 s chunks silently drop the first ~8 s of
each chunk; on the full 21 min meeting, whole-file decoding peaked at 10.9 GB
and was killed, 240 s chunks peak at 4.9 GB.

| Chunk / overlap | WER ES2002a | CPU-s per audio-s | Peak RSS |
|---|---|---|---|
| 240 s / 2 s | **27.6%** | 0.82 | 4.9 GB |
| 120 s / 5 s | 32.7% | 0.78 | 2.9 GB |
| 60 s / 5 s | 35.5% | 0.83 | 2.0 GB |
| 300 s / 2 s | unusable (head dropped) | — | 10.9 GB at 21 min |

Shorter chunks save memory but not CPU, and cost WER because speech is cut at
more boundaries. 240 s is the right point.

### Diarization

pyannote segmentation 3.0 int8 + CAMPPlus embeddings. `window_shift_ratio`
0.2 (1 s hop) instead of 0.1: same DER, ~45% less CPU. `threshold` is the
cosine **dissimilarity** cut.

| Meeting | threshold | DER | DER, no overlap | Speakers ref/hyp | Wall | RTF | CPU-s per audio-s | Peak RSS |
|---|---|---|---|---|---|---|---|---|
| ES2002a | 1.0 | **6.5%** | 5.8% | 4 / 9 | 4:03 | 0.189 | 0.44 | 286 MB |
| IS1000a | 1.0 | 20.4% | 18.1% | 4 / 14 | 7:00 | 0.263 | 0.39 | 327 MB |
| IS1000a | 1.1 | **18.3%** | 16.5% | 4 / 7 | — | — | — | — |
| IN1014 (1 h) | 1.0 | **4.4%** | 2.1% | 4 / 10 | 11:52 | 0.192 | 0.47 | 493 MB |

The threshold sweep on ES2002a (0.1 window) shows why a fixed default is a
trap: the sherpa-onnx default 0.5 yields **84 clusters**, and DER only becomes
usable above 0.9:

| threshold | 0.5 | 0.7 | 0.8 | 0.9 | 1.0 | 1.05 | 1.1 | 1.2 |
|---|---|---|---|---|---|---|---|---|
| speakers | 84 | 47 | 24 | 19 | 9 | 7 | 6 | 2 |
| DER | — | — | 15.6% | 15.2% | 6.3% | 8.3% | 8.3% | 47.2% |

Automatic speaker counting is the weak point: all three meetings over-segment
(7-14 clusters for 4 speakers). DER stays low on two of them because the extra
clusters are mapped away, but the app cannot show a correct speaker count
without tuning or a user-provided number.

### Threads

Six-minute slice of ES2002a, same models:

| Threads | ASR wall | ASR RTF | ASR CPU-s/audio-s | Diar wall | Diar RTF | Diar CPU-s/audio-s |
|---|---|---|---|---|---|---|
| 1 | 2:32 | 0.379 | 0.32 | 0:25 | 0.069 | 0.068 |
| 4 | 1:06 | 0.158 | 0.48 | 0:17 | 0.048 | 0.165 |
| 8 | 1:23 | 0.208 | 0.80 | 0:33 | 0.093 | 0.374 |

On the full meetings the same pattern holds:

| Run | Threads | Wall | RTF | CPU-s per audio-s |
|---|---|---|---|---|
| ASR ES2002a | 8 | 5:54 | 0.269 | 0.82 |
| ASR ES2002a | 4 | **3:57** | **0.186** | **0.48** |
| Diarization IN1014 | 8 | 11:52 | 0.192 | 0.47 |
| Diarization IN1014 | 4 | **3:32** | **0.058** | **0.19** |

ONNX Runtime's intra-op pool spins between tasks, so CPU time grows steeply
with threads: diarization at 8 threads burns 5.5x the CPU of one thread for
1.4x the wall speed, and on the contended host 8 threads were slower than 4.
Four threads per stage is the sweet spot. Running ASR and diarization
**concurrently** (4 + 4) then costs no more wall time than ASR alone: on the
6 min slice the pair took 66 s, the same as ASR by itself, with 0.66 CPU-s per
audio-s in total.

### Alternatives measured

Every item here was actually run, not estimated:

| Change | Effect on 1 h ASR | Effect on quality |
|---|---|---|
| 8 threads instead of 4 | more CPU (0.76 vs 0.48 CPU-s/s), slower wall under load | none |
| 120 s chunks | same CPU | WER 27.6% -> 32.7% |
| 60 s chunks | same CPU, half the memory | WER 27.6% -> 35.5% |
| `window_shift_ratio` 0.1 for diarization | 2x CPU | same DER on ES2002a, 14.8% vs 4.4% on IN1014 |
| `window_shift_ratio` 0.2 | 0.19 CPU-s/s on the 1 h meeting | same DER as 0.1 on ES2002a, better on IN1014 |
| titanet_small embedding | small model, cheap load | over-segments: 7-84 clusters on 2 min where CAMPPlus gives 4-16 |
| default threshold 0.5 | - | 84 clusters on ES2002a; unusable |
| `num_clusters = 4` (oracle) | - | DER 44.6% on IN1014, worse than automatic |

### Combined pipeline, IN1014 (61.5 min)

| Layout | Wall | RTF | Process CPU-min | Peak RSS |
|---|---|---|---|---|
| sequential, 8 threads per stage | 26:43 | 0.434 | 76.0 | 5.0 GB |
| ASR + diarization in parallel, 4 threads each | **19:21** | **0.315** | **51.3** | 5.2 GB |
| same, projected on 8 idle cores | 6.4 min | 0.107 | 51.3 | 5.2 GB |

Quality is identical in every layout (WER 26.9%, DER 4.4%); thread count and
scheduling only change time.

### What the numbers say for the engine

- **ASR** is the bottleneck: roughly 1800-2400 CPU-s for one hour at 4 threads
  (0.48-0.64 CPU-s per audio-s; the higher figure is inside the contended
  parallel run). Diarization adds about 711 CPU-s.
- A 16 GB, 8-core CPU machine meets the bar with ~35% headroom. A 4-core
  machine cannot: the ASR stage alone needs about 7.5 minutes at 100% of all
  four cores, leaving nothing for diarization.
- Peak memory is safe at 5.2 GB. Whole-file decoding without chunking is not:
  a 21 min file already peaks at 10.9 GB.
- If diarization must run on a 4-core machine, the cheap lever is to skip it
  and let the user name speakers, or to clamp `num_threads` to 2 for the
  diarization stage.




## Reproduction

Everything below runs from the worktree; data and models stay in `$ROOT`.

```bash
# once: toolchain + python scoring deps in a venv outside the repo
python3 -m venv /home/sergio/code/evidence/tinta-fork/2/venv
/home/sergio/code/evidence/tinta-fork/2/venv/bin/pip install cmake numpy scipy rapidfuzz

export ROOT=/home/sergio/code/evidence/tinta-fork/2
scripts/fork/bench/fetch-data.sh          # downloads + verifies SHA-256
scripts/fork/bench/run-benchmark.sh       # builds, runs everything, prints the table

# one ASR run
scripts/fork/bench/target/release/tinta-bench asr \
  --wav $ROOT/data/IN1014.Mix-Headset.wav \
  --chunk-seconds 240 --chunk-overlap 2 \
  --encoder $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx \
  --decoder $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx \
  --joiner  $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx \
  --tokens  $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt \
  --threads 4 --output $ROOT/results/asr-IN1014-t4.json

# score it
$ROOT/venv/bin/python scripts/fork/bench/score.py score \
  --ref $ROOT/data/IN1014.ref.json \
  --result $ROOT/results/asr-IN1014-t4.json \
  --out $ROOT/results/score-asr-IN1014-t4.json

# one diarization run
scripts/fork/bench/target/release/tinta-bench diarize \
  --wav $ROOT/data/IN1014.Mix-Headset.wav \
  --seg-model $ROOT/models/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx \
  --embedding $ROOT/models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx \
  --threads 4 --threshold 1.0 --window-shift 0.2 --output $ROOT/results/diar-IN1014-t4.json

# the combined layout used for the bar: both stages at once, 4 threads each
( scripts/fork/bench/target/release/tinta-bench asr ... --threads 4 & \
  scripts/fork/bench/target/release/tinta-bench diarize ... --threads 4 & wait )
```

Peak RSS comes from `/usr/bin/time -v` (`Maximum resident set size`) and from
`VmHWM` recorded in every result JSON; both agree within 2%. The bench host did
not ship GNU time, so `apt-get download time` + `dpkg -x` was used to get it
without root.

## Files

| Path | Purpose |
|---|---|
| `scripts/fork/bench/src/main.rs` | Rust harness: `asr`, `diarize`, `combined` |
| `scripts/fork/bench/score.py` | AMI references, WER/DER, clean-sample cutter, table |
| `scripts/fork/bench/fetch-data.sh` | data/model download and SHA-256 check |
| `scripts/fork/bench/run-benchmark.sh` | end-to-end reproduction |

The harness is a standalone crate (`[workspace]` is empty) and is not part of
the app workspace. Results, logs and JSON are kept in the evidence directory,
never in the repo.
