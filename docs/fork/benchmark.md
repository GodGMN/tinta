# sherpa-onnx on CPU: can it replace FluidAudio/CoreML?

Task #2 of the Windows/Linux port. The Mac engine uses FluidAudio with CoreML for
ASR and diarization. The port needs a CPU-only equivalent that runs on an x86_64
machine with 16 GB and processes a one-hour meeting in ten minutes or less, with
quality in the range FluidAudio publishes.

The harness in `scripts/fork/bench/` runs two sherpa-onnx models on AMI audio:

- ASR: NeMo Parakeet TDT 0.6b v3 int8 (`offline_asr`, transducer), segmented by
  the sherpa-onnx Silero VAD. A hard cap in code refuses any model call over 60 s.
- Diarization: pyannote segmentation 3.0 + CAMPPlus embeddings
  (`offline_speaker_diarization`, complete-linkage clustering).

Scoring is independent of the harness: `pyannote.metrics` for DER (full-file UEM,
collar 0 s with overlap counted as primary, 0.25 s collar reported next to it)
and Whisper's `EnglishTextNormalizer` in addition to the old normalizer for WER.
The clustering threshold and the small-cluster filter were tuned on the official
AMI **dev** split and are reported on the official AMI **test** split, which was
not used for tuning.

## Verdict

**Pass on memory; pass on speed by projection (the host never went idle); fail on
meeting WER, DER and speaker counting. The port can proceed on sherpa-onnx, but
not with this segmenter and a fixed clustering threshold.**

The VAD-segmented pipeline removes the 240 s-chunk failure on some meetings but
not all: Silero VAD misses quiet meeting speech on ES2004b and TS3007b, so long
gaps remain. On the held-out test meetings the Whisper-normalized WER is
**27.3-27.4%** on the two full meetings (22.7% on the IS1009b 6 min sample) and
the no-collar DER with overlap counted is **24.6-29.1%** (mean 27.4%). Peak RSS
is **2.1 GB**. The measured 1 h ASR critical path was **22.8 min** at a 1-minute
load average of 22-45; the host never went below 2 during the session, so that
wall is pessimistic. The same run used 944 CPU-s for 56 min, which projects to
**3.2-4.4 min per hour at 4 threads** on an idle 8-core machine and **6.4-8.8
min for ASR alone at 2 threads** on a 4-core machine.

| Bar | Result | Verdict |
|---|---|---|
| 1 h meeting in <= 10 min on CPU | 22.8 min measured at load 22-45; projection 3.2-4.4 min at 4 threads on an idle 8-core | **pass by projection, not demonstrated** |
| 16 GB RAM | 2.1 GB peak | pass |
| Parakeet v3 2-5% WER on clean speech | 4.1% on LibriSpeech test-clean | pass |
| WER on real meetings (AMI mix) | 27.3-27.4% full, 22.7% 6 min sample; ~22-24% without overlap | **fail** against the remembered ~11% published AMI figure |
| pyannote 3.1-level DER on AMI | 24.6-29.1% no collar, overlap counted, held-out (collar 0.25: 20.8-25.7%) | **worse than the ~18.8% remembered reference** |
| Speaker count | 30/35/33 raw, 6/5/5 after the 2% filter (4 reference) | **fail**; a global threshold cannot fix it, user override needed |

The AMI mix channel contains 10-25% overlapped reference words (40% on TS3006b)
that a single-stream ASR cannot recover, plus spontaneous disfluencies. The
clean read-speech result stays inside the published 2-5% band. The meeting WER is
therefore not evidence of a broken model, but it is also not a clean-speech
number.

Two findings decide the engineering:

1. **Never feed Parakeet an arbitrary long window.** A 240 s input drops whole
   stretches in the middle (IN1014: 1,947 deletions, 163 s of gaps with 254
   reference words). VAD pieces of 10-30 s with a 2 s pre-roll remove that failure
   on IS1009b, but the segmenter's own recall becomes the limit on ES2004b and
   TS3007b. A piece that starts mid-speech with no context can also return *zero*
   words (a 20 s piece at 155.3 s on ES2002a did), while the same audio with 1 s of
   pre-roll returned 39 words.
2. **Set ONNX Runtime's intra-op spinning off and run the stages concurrently.**
   `SessionConfig.session.intra_op.allow_spinning=0` (passed through sherpa-onnx's
   provider config file) cut the 6 min slice from 317.9 to
   78.4 CPU-s and from 759.9 to 93.9 wall
   on the same 4 threads. Diarization then fits under the ASR wall.

## Machine

| | |
|---|---|
| CPU | AMD Ryzen 7 5825U, 8 logical CPUs (KVM guest), 1 thread per core |
| CPUs available to the process | 7 (`sched_getaffinity` mask 1-7); 7 physical cores |
| RAM | 22 GiB (the bar is 16 GB) |
| OS | Debian 12, kernel 6.1.0-54-amd64 |
| Toolchain | Rust 1.100 (nightly), sherpa-onnx crate 2.0.1, static build |
| Crypto/accel | CPU only, no CUDA/CoreML, ONNX Runtime CPU provider |

The host is a shared development VM. Other agents' builds kept the 1-minute load
average between 5 and 66 on 8 CPUs; it never dropped below 2,
so no timing run in this document is clean. Every result JSON records
`loadavg_start`/`loadavg_end` and GNU time records user CPU seconds. Wall times
are pessimistic; CPU seconds per audio second and the critical-path projection
are the stable numbers.

## Data

Public AMI corpus, Mix-Headset channel (16 kHz mono), plus the manual annotations
(`ami_public_manual_1.6.2`). The split is the official AMI **full-corpus**
partition (the same one lhotse and the AMI site publish):

| Split | Meetings used here |
|---|---|
| train | ES2002a, IS1000a, IN1014 (context only) |
| dev (tuning) | ES2003b, IS1008b (TS3006b was started but not needed) |
| test (held out) | ES2004b, IS1009b, TS3007b |

| Meeting | Split | Audio | Speech end | Speakers | Words | Overlapped words | Character |
|---|---|---|---|---|---|---|---|
| ES2002a | train | 21.2 min | 18.5 min | 4 | 2600 | 21.9% | scenario, kick-off |
| IS1000a | train | 26.4 min | 25.2 min | 4 | 2893 | 24.2% | non-native speakers, technical |
| IN1014 | train | 61.5 min | 61.4 min | 4 | 10896 | 26.7% | the ~1 h meeting |
| ES2003b | dev | 35.2 min | 35.0 min | 4 | 5693 | 10.7% | scenario |
| IS1008b | dev | 29.5 min | 28.9 min | 4 | 4314 | 10.6% | non-native speakers |
| TS3006b | dev | 39.2 min | 39.1 min | 4 | 6987 | 40.4% | heavy overlap |
| ES2004b | test | 39.1 min | 39.0 min | 4 | 6763 | 21.2% | scenario |
| IS1009b | test | 34.2 min | 33.8 min | 4 | 6091 | 21.8% | non-native speakers |
| TS3007b | test | 55.9 min | 42.3 min | 4 | 6643 | 18.2% | the ~1 h test meeting |

"Overlapped words" counts reference words whose interval overlaps a word from
another speaker on the 10 ms grid. TS3007b has 14 min of trailing non-speech, so
its audio is 56 min while its last word is at 42 min.

Two extra samples check the quality claim itself:

- **Clean read speech**: 30 utterances from LibriSpeech `test-clean`
  (3.7 min), rows 0-29 of `openslr/librispeech_asr` clean/test.
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
| Silero VAD | `silero_vad.onnx` (629 KB) | `9e2449e1…1fd6` |

Full hashes, audio hashes and download URLs are in
`scripts/fork/bench/fetch-data.sh`. The models are sherpa-onnx release assets:

- <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2>
- <https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2>
- <https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx>
  (the upstream release tag is misspelled "recongition"; the URL is correct)
- <https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx>

Data (~700 MB of wav, 23 MB of annotations) and models (~1.2 GB) live outside
the repo under `/home/sergio/code/evidence/tinta-fork/2/`. Nothing is committed
except the harness and this document.

## Method

### ASR: VAD segmentation, not fixed chunks

- `OfflineRecognizer` with the transducer encoder/decoder/joiner, `cpu`
  provider, greedy search (`max_active_paths` left at the default 4).
- Silero VAD (`threshold` 0.5, `min_silence_duration` 0.3 s,
  `min_speech_duration` 0.1 s, `window_size` 512, `max_speech_duration` 30 s)
  detects speech. Continuous speech is flushed every 20 s so the VAD's circular
  buffer cannot grow without bound.
- Speech runs are cut into pieces of at most 30 s at the **quietest 50 ms
  window** in the last 40% of the allowed span, not at an arbitrary boundary.
- Every piece gets **2 s of pre-roll and 1 s of post-roll**, and a piece is
  extended to at least **10 s**. Both are required by the model, not tuning
  knobs: a 6 s piece cut at 0 s returned "And at this point…" and dropped the
  first two seconds, while the same audio in a 10 s piece returned "Three times.
  Um, and at this point…". A 20 s piece starting exactly at 155.3 s returned
  **zero** words; 1 s earlier it returned 39.
- Overlapping pieces are de-duplicated at the **midpoint of the overlap on both
  sides**, then all words are sorted by time.
- `MAX_ASR_SECONDS = 60.0` is asserted in `run_asr_call` before every decode.
  The harness cannot feed a longer window even if a future VAD returns one.
- The old fixed-chunk mode is kept only for the ablation (`--chunk-seconds`),
  with the same midpoint rule; the benchmark no longer uses it.

### Diarization

- pyannote segmentation 3.0 int8, `window_shift_ratio` 0.2 (1 s hop), then
  CAMPPlus embeddings and complete-linkage clustering.
- `threshold` in sherpa-onnx is a **cosine dissimilarity** cut, not a
  similarity: higher merges more.
- `min_duration_on` 0.3 s, `min_duration_off` 0.5 s.
- After clustering, the scorer applies a **small-cluster filter**: clusters
  under 2% of total speech time are dissolved and each of their segments is
  reassigned to the nearest large cluster in time. The benchmark uses temporal
  proximity because sherpa-onnx does not expose per-segment embeddings; the
  engine should reassign by embedding distance. The filter never forces a fixed
  speaker count and is disabled with fraction 0.
- The threshold is a CLI value (`--threshold`); the engine must keep it
  adjustable, and the user must be able to merge or rename speakers.

### Scoring

`score.py`, definitions matter more than the numbers:

- **DER**: `pyannote.metrics.diarization.DiarizationErrorRate` on the reference
  segments, with a UEM from 0 to the end of the audio so false alarms in silence
  count. Primary is **collar 0 s, `skip_overlap=False`** (overlap is scored, so
  missing the second speaker in an overlap is a miss). The 0.25 s collar number
  is reported next to it. Miss, false alarm and confusion are reported
  separately, each divided by reference speech time.
- **WER**: two normalizers. The legacy one (lowercase, strip punctuation, one
  token per word) and Whisper's `EnglishTextNormalizer`, which handles numbers,
  spellings and fillers. Both report substitutions, insertions and deletions.
- **WER without overlap**: reference words fully inside single-speaker regions,
  hypothesis words whose midpoint falls in those regions. Isolates the part of
  the error caused by overlapping speech.
- **Long hypothesis gaps**: gaps of at least 5 s with no hypothesis word where
  the reference has speech. This is the diagnostic for the old chunk failure.
- **Speaker count**: number of clusters with `num_clusters = -1` (automatic),
  raw and after the small-cluster filter, against the four reference speakers.

### Held-out tuning

The threshold sweep (0.7, 0.8, 0.9, 1.0, 1.1) and the cluster-filter sweep (0,
0.5, 1, 2, 3% of speech time) run on the **dev** meetings only (full IS1008b and
ES2003b, plus 6 min ES2003b samples for the lower thresholds). The selected
setting is frozen and then run once on each **test** meeting. The test results in
this document were not used to choose any value.

## Results

### ASR on the held-out test meetings

VAD segmentation, greedy transducer decoding, 8 threads for the two standalone
runs (the quality does not depend on the thread count; 8 was used to get through
the contended host). The TS3007b row is the combined pipeline at the engine
defaults (3 ASR + 3 diarization threads), so its wall is the critical path.

| Meeting | Audio | Threads | ASR wall | RTF | CPU-s / audio-s | WER legacy | WER Whisper | del/ins/sub | WER no-overlap | Long gaps | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ES2004b | 39.1 min | 8 | 15:26 | 0.395 | 0.228 | 29.7% | **27.4%** | 1606/46/218 | 22.3% | 431 s / 889 words | 1.9 GB |
| IS1009b (6 min sample) | 6.0 min | 8 | 3:05 | 0.515 | 0.240 | 24.2% | **22.7%** | 233/10/47 | 12.9% | 7 s / 11 words | 1.7 GB |
| TS3007b | 55.9 min | 3+3 | 21:13 | 0.379 | — | 28.4% | **27.3%** | 1347/58/305 | 23.7% | 321 s / 485 words | 2.1 GB |

The full IS1009b ASR run did not finish before the host was needed for the other
runs; the 6 min sample is reported and marked as such. The two full meetings
agree: about 27% Whisper-normalized WER on the AMI mix, 22-24% when overlapping
reference words are removed. This is far above Parakeet's published AMI figure
remembered as about 11% (IHM short segments, a different channel and
segmentation), and the long gaps show the VAD still misses quiet meeting speech:
on ES2004b the Silero VAD marks 547-595 s as silence while a plain 60 s decode of
530-600 s returns 18 words. Lowering the VAD threshold to 0.2 did not fix it (a
6 min ES2004b sample: 34.1% vs 33.9% Whisper WER, 132.6 s vs 134.5 s of gaps).
The ASR deletion failure is therefore **reduced but not solved** by Silero VAD.

### What the old 240 s chunks did

The old harness is re-scored with the corrected scorer. The VAD rows are the new
pipeline; the 240 s rows are the same meetings from `results/asr-*-c240.json`.

| Design | Meeting | WER Whisper | Deletions | Long gaps |
|---|---|---|---|---|
| 240 s / 2 s, 8 threads | ES2002a | 25.0% | 463 | 270 s / 48 words |
| 240 s / 2 s, 8 threads | IS1000a | 41.9% | 993 | 488 s / 542 words |
| 240 s / 2 s, 8 threads | IN1014 | 24.1% | 1947 | 163 s / 254 words |
| VAD, 8 threads | ES2004b | 27.4% | 1606 | 431 s / 889 words |
| VAD, 3+3 threads | TS3007b | 27.3% | 1347 | 321 s / 485 words |
| VAD, 8 threads | IS1009b, 6 min | 22.7% | 233 | 7 s / 11 words |

The 240 s windows lose whole mid-chunk stretches; the VAD pieces remove them on
IS1009b but not on ES2004b and TS3007b, where the segmenter misses speech. The
lesson is not "shorter is always better": the segmenter's **recall** is the new
failure mode, and it has to be measured against reference speech.

### Clean speech

| Sample | Audio | WER legacy | WER Whisper | del/ins/sub | RTF |
|---|---|---|---|---|---|
| LibriSpeech `test-clean`, 30 utterances | 3.7 min | 4.0% | **4.1%** | 4/4/17 | 0.346 |
| ES2002a single-speaker regions | 5.0 min | 26.3% | **24.3%** | 125/22/23 | 0.298 |

The read-speech result is inside the published 2-5% band, at its top end. The
single-speaker AMI sample is still 24.3%, and it has 54 s of hypothesis gaps with
50 reference words: the VAD is dropping quiet spontaneous speech, not just
overlap. This is the same recall problem as ES2004b.


### Diarization tuning on dev, then test

The threshold sweep was run on dev only. `threshold` is cosine **dissimilarity**
(higher merges more). The small-cluster filter dissolves clusters under 2% of
speech time and reassigns their segments to the nearest large cluster in time.

| Meeting | Threshold | DER no collar | DER filtered | Speakers raw/filtered/ref |
|---|---|---|---|---|
| IS1008b (full) | 0.7 | 17.2% | **14.9%** | 28 / 4 / 4 |
| IS1008b (full) | 0.9 | 13.8% | 13.7% | 13 / 4 / 4 |
| IS1008b (full) | 1.0 | 13.7% | 13.5% | 8 / 4 / 4 |
| IS1008b (full) | 1.1 | 29.7% | 30.7% | 5 / 3 / 4 |
| ES2003b (full) | 1.0 | 40.9% | 41.9% | 6 / 2 / 4 |
| ES2003b (6 min) | 0.7 | 14.1% | **14.1%** | 4 / 4 / 4 |
| ES2003b (6 min) | 0.8 | 12.4% | 12.4% | 3 / 3 / 4 |
| ES2003b (6 min) | 0.9 | 38.8% | 38.8% | 2 / 2 / 4 |

A single global threshold does not fit both dev meetings: IS1008b wants 0.9-1.0
and ES2003b collapses to two clusters above 0.8. We froze **0.7**: it keeps the
correct filtered count on IS1008b (28 raw clusters -> 4) and on the ES2003b
sample (4/4), at a DER cost of 1.4 points on IS1008b. The engine must estimate
the threshold per recording or let the user adjust it; 0.7 is a safe default
that over-segments rather than merging speakers.

| Meeting (test) | Threshold | DER no collar | DER, 0.25 collar | DER filtered | miss / FA / confusion | Speakers raw/filtered/ref | Wall | CPU-s / audio-s | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|
| ES2004b | 0.7 | 25.5% | 20.8% | **24.6%** | 14.1 / 2.6 / 8.8 | 30 / 6 / 4 | 10:32 | 0.129 | 639 MB |
| IS1009b | 0.7 | 29.6% | 25.0% | **28.4%** | 10.4 / 2.1 / 17.1 | 35 / 5 / 4 | 9:49 | 0.129 | 593 MB |
| TS3007b | 0.7 | 30.2% | 25.7% | **29.1%** | 7.0 / 3.0 / 20.2 | 33 / 5 / 4 | 11:10 | 0.111 | 842 MB |

The corrected no-collar DER is **24.6-29.1%**, not the 4-18% the old frame-based
scorer reported. With the 0.25 s collar it is 20.8-25.7%. For comparison,
pyannote 3.1 on AMI headset-mix is remembered as about 18.8% no collar with
overlap scored, and community-1 about 17%; the sherpa-onnx pipeline is therefore
**worse than the reference on these three held-out meetings**, by roughly 6-11
points. The raw cluster counts are 30-35 and the 2% filter only brings them to
5-6, not 4: clustering is over-segmenting and confusion (17-20 points on
IS1009b/TS3007b) shows speakers being merged as well.

### Threads, ONNX Runtime spinning, and the timing

`sherpa-onnx` 2.0.1 forwards `SessionConfig.*` keys from the provider config
file. Writing `SessionConfig.session.intra_op.allow_spinning=0` and passing
`provider = "cpu:/path/to/config"` is what `--spin 0` does. It is not a small
effect:

| Stage | Slice | Threads | Spin | Wall | CPU-s |
|---|---|---|---|---|---|
| ASR | 6 min | 4 | default | 759.9 s | 317.9 |
| ASR | 6 min | 4 | **0** | **93.9 s** | **78.4** |
| Diarization | 6 min | 4 | default | 125.1 s | 70.5 |
| Diarization | 6 min | 4 | **0** | **111.3 s** | **39.3** |

The 6 min ASR slice is 4x cheaper in CPU and 8x faster in wall with spinning
off. The wall numbers are load-bound; the CPU ratio is the stable result.


The 6 min diarization slice went from 70.5 to 39.3 CPU-s (-44%) and from 125.1
to 111.3 s wall. The 40 s ASR check went from 17.2 to 9.8 CPU-s and from 27.4 to
20.1 s wall at 2 threads. Spinning off is now the default for every benchmark
run and should be the engine default.

### Combined pipeline

The combined run starts ASR and diarization at the same time. On this host the
default is 3 ASR + 3 diarization threads (7 CPUs available, `physical_cores / 2`
clamped to 4).

| Meeting | ASR wall | Diarization wall | Critical path | Total wall | Total CPU | Peak RSS | Load at start/end |
|---|---|---|---|---|---|---|---|
| TS3007b (55.9 min) | 21:13 | 15:19 | **21:13** | 21:13 | 944 s | 2.1 GB | 22.3 / 36.7 |

Diarization finishes under the ASR wall, so the critical path is the ASR wall.
The 1 h projection from that critical path is **22.8 min** under this load. The
same run used 944 CPU-s for 56 min (0.281 CPU-s per audio second including both
stages); the ASR part is about 572 CPU-s after subtracting the standalone
diarization CPU, or **614 CPU-s per audio hour**. At 4 threads with 70-80%
scaling that is **3.2-4.4 min per hour**, and at 2 threads (a 4-core
transcript-first pass) **6.4-8.8 min** for ASR alone. The bar is therefore
reachable on an idle 8-core CPU and even the ASR transcript fits in 10 min on 4
cores; the measured 22.8 min is a busy-host number, not the engine's ceiling.


## Engine design conditions

These are the concrete rules the Rust engine must follow. They are derived from
the measurements above, not from the model cards.

1. **VAD segmentation with a hard cap.** Segment ASR input with the sherpa-onnx
   Silero VAD. Pieces are at most 30 s, cut at the quietest point in the last
   part of the window, with 2 s pre-roll, 1 s post-roll and a 10 s minimum.
   `MAX_ASR_SECONDS = 60` must be asserted in code before every recognizer call.
   Never decode a whole file: the old whole-file attempt hit 10.9 GB at 21 min.
   De-duplicate overlapping pieces at the overlap midpoint and sort by time.
2. **Threads from the CPUs the process actually has.** Read
   `sched_getaffinity` (or `available_parallelism`), then count distinct
   `(physical_package_id, core_id)` pairs and any cgroup quota. Default ASR to
   `min(4, max(1, physical_cores / 2))`; diarization to
   `clamp(physical_cores / 2, 2, 4)`. Run the two stages concurrently. Never
   hard-code 8; on this host 8 threads oversubscribed 7 available CPUs.
3. **Disable ONNX Runtime intra-op spinning.** sherpa-onnx 2.0.1 forwards keys
   prefixed `SessionConfig.` from the provider config file
   (`provider = "cpu:/path/to/config"`). Write
   `SessionConfig.session.intra_op.allow_spinning=0`. Measured on the 6 min
   slice: ASR 6 min 317.9 -> 78.4 CPU-s (-75%), 759.9 -> 93.9 s wall; diarization 70.5 -> 39.3 CPU-s (-44%). Keep the default available for comparison, but make
   spinning off the engine default for predictable CPU on laptops and shared
   machines.
4. **CPU budget per audio second.** The measured rate is ASR **0.17-0.23 CPU-s
   per audio second** and diarization **0.11-0.13**, about **0.28-0.36 total**
   with spinning off. One hour is then **600-850 ASR CPU-s** and **400-470
   diarization CPU-s**. Show a progress/ETA estimate from audio seconds and the
   measured rate, not from a fixed promise.
5. **4-core machines use transcript-first.** ASR needs about 600-850 CPU-s per
   audio hour, which is 6-9 minutes at 2 threads and 80% scaling, before
   diarization. On 4 cores, run ASR first with 2-3 threads, show the transcript,
   then run diarization in the background. A single pass with both stages on 4
   cores is over the ten-minute bar; the transcript alone still fits.
6. **Diarization: per-recording threshold plus small-cluster reassignment.**
   Keep the cosine distance threshold adjustable; the dev sweep selected **0.7**
   as a safe default (it over-segments instead of merging speakers), but a single
   global value does not fit every meeting. Estimate it per recording from the
   cluster-distance distribution, or let the user adjust it. Dissolve clusters
   under 2% of speech time and reassign their segments to the nearest large
   cluster by embedding; never force a fixed `num_clusters`. Let the user merge,
   split or rename speakers. The raw count is not trustworthy; the filtered count
   is the default the UI should show, with an override.
7. **Acceptance test.** A release candidate passes when, on held-out AMI
   full-corpus test meetings (at least ES2004b, IS1009b and one ~1 h meeting
   such as TS3007b), with settings frozen on dev:
   - DER (pyannote.metrics, collar 0, overlap scored, full-file UEM) is at or
     below the pyannote 3.1 reference for AMI headset-mix (from memory about
     18.8%; community-1 about 17% — both marked as remembered figures), plus
     two points of slack;
   - WER with Whisper's English normalizer is within a few points of Parakeet
     TDT v3's published AMI number (from memory about 11% on the Open ASR
     Leaderboard IHM short segments; not the same channel or segmentation);
   - one hour of audio completes in <= 10 minutes on a dedicated 8-core CPU and
     peaks below 8 GB;
   - the filtered speaker count is reported, and the user can correct it.

## Reproduction

Everything below runs from the worktree; data and models stay in `$ROOT`.

```bash
# once: toolchain + python scoring deps in a venv outside the repo
python3 -m venv /home/sergio/code/evidence/tinta-fork/2/venv
/home/sergio/code/evidence/tinta-fork/2/venv/bin/pip install \
  cmake numpy scipy rapidfuzz pyannote.metrics whisper-normalizer

export ROOT=/home/sergio/code/evidence/tinta-fork/2
scripts/fork/bench/fetch-data.sh    # downloads + verifies every SHA-256
ALLOW_BUSY=1 scripts/fork/bench/run-benchmark.sh  # builds, runs, prints tables
```

`run-benchmark.sh` refuses to time when the 1-minute load average is 2 or more.
Set `ALLOW_BUSY=1` to record busy-host numbers anyway; every result JSON then
contains the load at start and end. The script creates the 6 min slice, the clean
samples, the VAD runs, the fixed-240 s ablation, the thread/spin ablation, the
dev threshold sweep, the held-out test runs and the parallel pipeline, and ends
with the markdown tables in `$ROOT/results/repro/summary.md`.

One ASR run by hand:

```bash
scripts/fork/bench/target/release/tinta-bench asr \
  --wav $ROOT/data/IN1014.Mix-Headset.wav \
  --vad-model $ROOT/models/silero_vad.onnx \
  --encoder $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx \
  --decoder $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx \
  --joiner  $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx \
  --tokens  $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt \
  --threads 4 --spin 0 --output $ROOT/results/asr-IN1014.json

$ROOT/venv/bin/python scripts/fork/bench/score.py score \
  --ref $ROOT/data/IN1014.ref.json \
  --result $ROOT/results/asr-IN1014.json \
  --small-cluster-fraction 0.02 \
  --out $ROOT/results/score-asr-IN1014.json
```

One diarization run by hand:

```bash
scripts/fork/bench/target/release/tinta-bench diarize \
  --wav $ROOT/data/TS3007b.Mix-Headset.wav \
  --seg-model $ROOT/models/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx \
  --embedding $ROOT/models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx \
  --threads 4 --threshold 1.0 --window-shift 0.2 --spin 0 \
  --output $ROOT/results/diar-TS3007b.json
```

Peak RSS comes from `/proc/self/status` `VmHWM` in every result JSON; GNU time's
`Maximum resident set size` agrees within a few percent. The bench host did not
ship GNU time, so `apt-get download time` + `dpkg -x` was used to get it without
root.

## Files

| Path | Purpose |
|---|---|
| `scripts/fork/bench/src/main.rs` | Rust harness: `asr`, `diarize`, `combined`, `vad` |
| `scripts/fork/bench/score.py` | AMI references, pyannote DER, both WERs, tuning, tables |
| `scripts/fork/bench/fetch-data.sh` | data/model download and SHA-256 check |
| `scripts/fork/bench/run-benchmark.sh` | end-to-end reproduction |

The harness is a standalone crate (`[workspace]` is empty) and is not part of
the app workspace. Results, logs and JSON are kept in the evidence directory,
never in the repo.

## Limitations

- The timing host never went idle. The measured wall times include heavy
  contention and are not the ten-minute bar; the projection is explicitly
  labelled.
- The small-cluster filter in the benchmark reassigns by temporal proximity
  because sherpa-onnx does not expose embeddings. The engine should use
  embeddings; the benchmark's DER after the filter is therefore an estimate of
  the engine's behavior, not an upper bound.
- AMI reference segments are manual `transcriber_start`/`transcriber_end`
  regions; DER numbers from word references differ. The same reference type is
  used for every meeting and every threshold.
- Published pyannote and Parakeet AMI figures quoted in this document are from
  memory and are marked as such; they are not measured here.
