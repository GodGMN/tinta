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

**Pass on memory. Pass on speed by projection on an idle 8-core machine, but not
on 4 cores for a full pass. The quality gap is now explained rather than
unexplained: the remaining WER is AMI mixed-channel difficulty and the DER gap
is the final cluster merge. The port can proceed on sherpa-onnx.**

The held-out test meetings give a Whisper-normalized WER of **27.0-27.1%** on the
two full meetings (22.5% on the IS1009b 6 min sample) and a no-collar DER with
overlap counted of **24.6-29.1%** (mean 27.4%). Peak RSS is **2.1 GB**. The
measured 1 h ASR critical path was **21.2 min** at a 1-minute load average of
22-45; the host never went idle, so that wall is pessimistic. The same run used
944 CPU-s. Per *decoded* second the ASR costs **0.16 CPU-s idle to 0.21 CPU-s
loaded**, and a dense meeting decodes about **1.1x** its audio, which projects to
**4-6 min per audio hour on an idle 8-core machine** and **about 6-7 min of ASR
on 4 cores**. A full single pass on 4 cores is **10-11 min**, i.e. over the bar.

| Bar | Result | Verdict |
|---|---|---|
| 1 h meeting in <= 10 min on CPU | 21.2 min measured at load 22-45; projection 4-6 min at 4 threads on an idle 8-core | **pass by projection, not demonstrated** |
| 16 GB RAM | 2.1 GB peak | pass |
| Parakeet v3 2-5% WER on clean speech | 4.1% on LibriSpeech test-clean | pass |
| WER on real meetings (AMI mix) | 27.0-27.1% full, 22.5% 6 min sample; 22-24% without overlap | **expected for this channel**; not comparable to the remembered ~11% IHM figure |
| pyannote 3.1-level DER on AMI | 24.6-29.1% no collar, overlap counted, held-out (collar 0.25: 20.8-25.7%) | **worse than the remembered ~18.8%, but the front end is not the cause** |
| Speaker count | 30/35/33 raw, 6/5/5 after the 2% filter (4 reference) | **fail**; a global threshold cannot fix it, user override needed |

Three findings decide the engineering:

1. **The VAD is not losing speech.** 99.6% (ES2004b) and 99.9% (TS3007b) of the
   reference words have their midpoint inside a decoded piece; against the raw
   VAD regions alone the figure is 94.1% and 95.6%, and the 2 s pre-roll, 1 s
   post-roll and 10 s minimum recover the rest. The deletions are Parakeet on
   the mixed channel: in overlap regions the hypothesis has only 0.30-0.35 words
   per reference word, and backchannels (`yeah`, `mm-hmm`, `okay`, `uh`) are
   dropped wholesale. A better segmenter will not close the WER gap.
2. **The DER gap comes from the final cluster merge.** The clustering finds the
   four speakers: the top four clusters are 81-91% pure and each maps to one
   reference speaker. If every raw cluster is mapped to the reference speaker it
   overlaps most, the over-clustered output scores **13.8-16.6% DER** against
   pyannote's word-level reference. The pipeline loses 8-14 points when it
   collapses 30-35 clusters into 4, not in segmentation or embeddings.
3. **Never feed Parakeet an arbitrary long window.** A 240 s input drops whole
   stretches in the middle (IN1014: 1,839 deletions, 163 s of gaps with 254
   reference words). VAD pieces of 10-30 s with a 2 s pre-roll remove that
   failure. Separately, `SessionConfig.session.intra_op.allow_spinning=0` cut the
   6 min ASR slice from 317.9 to 78.4 CPU-s and from 759.9 to 93.9 s wall on the
   same 4 threads; diarization then fits under the ASR wall.

## Machine

| | |
|---|---|
| CPU | AMD Ryzen 7 5825U, 8 logical CPUs exposed (KVM guest), 1 thread per core |
| CPUs available to the process | **6**: affinity mask 1-7 (7 logical), cgroup v2 `cpu.max` `600000 100000` (6), `available_parallelism` 6 |
| RAM | 22 GiB (the bar is 16 GB) |
| OS | Debian 12, kernel 6.1.0-54-amd64 |
| Toolchain | Rust 1.100 (nightly), sherpa-onnx crate 2.0.1, static build |
| Crypto/accel | CPU only, no CUDA/CoreML, ONNX Runtime CPU provider |

The host is a shared development VM. Other agents' builds kept the 1-minute load
average between 5 and 66 on 8 CPUs; it never dropped below 2,
so no timing run in this document is clean. Every result JSON records
`loadavg_start`/`loadavg_end` and GNU time records user CPU seconds. Wall times
are pessimistic; CPU seconds per decoded second and the critical-path projection
are the stable numbers.

The first version of this document reported seven CPUs available. That was the
affinity mask. The process's own cgroup (`user-1000.slice`) has a 6-CPU quota,
and the harness missed it because it read `/sys/fs/cgroup/cpu.max`, which is the
root cgroup and does not exist here. `cgroup_cpus()` now reads the process's own
path from `/proc/self/cgroup` and walks up to the root, taking the smallest
`cpu.max`. The 8-thread ASR test runs therefore oversubscribed a 6-CPU quota;
the defaults (3 ASR + 3 diarization) did not. Inside a VM, `core_id` reflects the
hypervisor's topology, not the physical one, so it is only a fallback.

## Data

Public AMI corpus, Mix-Headset channel (16 kHz mono), plus the manual annotations
(`ami_public_manual_1.6.2`). The split is the official AMI **full-corpus**
partition (the same one lhotse and the AMI site publish):

| Split | Meetings used here |
|---|---|
| train | ES2002a, IS1000a, IN1014 (context only) |
| dev (tuning) | ES2003b, IS1008b (TS3006b was started but not needed) |
| test (held out) | ES2004b, IS1009b, TS3007b |

This is not the split pyannote uses. In BUT's `AMI-diarization-setup`,
**TS3007b and ES2003b are training meetings**, so pyannote segmentation 3.0 saw
them during training. That makes any pyannote figure on TS3007b optimistic and
ES2003b a weak dev meeting. The engine acceptance list in the design conditions
uses BUT's test meetings instead: `ES2004*`, `IS1009*`, `TS3003*`, `EN2002*`.

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

The CAMPPlus model is trained on Chinese (`zh-cn_16k-common`). It is the right
size and licence for the spike, but it is not the last word for English
meetings; see design condition 6.

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
  buffer cannot grow without bound. **That flush is a defect**: it creates 0.83 s
  holes, larger than the 0.5 s join gap, so a word can fall between two pieces.
  The engine should use the existing `split_run` quietest-point split instead, or
  raise the join gap above the flush hole. It is not the main WER gap, but it is
  free to remove.
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
  spellings and fillers. Both map AMI's spelled-out acronyms first: the manual
  annotation writes `T_V_`, Parakeet writes `TV`, and without the mapping each
  one costs a substitution plus two deletions (24-54 tokens per meeting). Every
  WER figure in this document includes the mapping; it moves them by 0.1-1.0
  points from the pre-mapping draft.
- **WER without overlap**: reference words fully inside single-speaker regions,
  hypothesis words whose midpoint falls in those regions. Isolates the part of
  the error caused by overlapping speech.
- **Long hypothesis gaps**: gaps of at least 5 s with no hypothesis word where
  the reference has speech. This is the diagnostic for the old chunk failure.
- **Speaker count**: number of clusters with `num_clusters = -1` (automatic),
  raw and after the small-cluster filter, against the four reference speakers.
- **Oracle marker**: every raw cluster is mapped to the reference speaker it
  overlaps most, then DER is scored again. This bounds what a perfect final
  merge could recover from the existing segmentation and embeddings.

### Held-out tuning

The threshold sweep (0.7, 0.8, 0.9, 1.0, 1.1) and the cluster-filter sweep (0,
0.5, 1, 2, 3% of speech time) run on the **dev** meetings only (full IS1008b and
ES2003b, plus 6 min ES2003b samples for the lower thresholds). The selected
setting is frozen and then run once on each **test** meeting. The test results in
this document were not used to choose any value.

Two caveats belong with that claim:

- `logs/diar-{ES2004b,IS1009b}-t4-th{0.9,1.0}.out` show test-meeting runs that
  started at 00:32 and 00:39, before the freeze at 00:47. No result JSON was
  written for them, so nothing from them reached the tables, but they were
  launched. The frozen run is the 0.7 one.
- ES2003b's low-threshold evidence is a 6 min sample for 0.7-0.9 (the full
  meeting only at 1.0). A full sweep would be better; it is not needed to
  justify 0.7, which fails safe by over-clustering.

## Results

### ASR on the held-out test meetings

VAD segmentation, greedy transducer decoding, 8 threads for the two standalone
runs. Eight threads oversubscribed the 6-CPU quota (see Machine); the quality
does not depend on the thread count, and the engine's own default would be 3.
The TS3007b row is the combined pipeline at the engine defaults (3 ASR + 3
diarization threads), so its wall is the critical path.

| Meeting | Audio | Threads | ASR wall | RTF | CPU-s / audio-s | WER legacy | WER Whisper | del/ins/sub | WER no-overlap | Long gaps | Peak RSS |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ES2004b | 39.1 min | 8 | 15:26 | 0.395 | 0.228 | 29.4% | **27.1%** | 1575/57/207 | 22.1% | 431 s / 889 words | 1.9 GB |
| IS1009b (6 min sample) | 6.0 min | 8 | 3:05 | 0.515 | 0.240 | 23.9% | **22.5%** | 230/11/45 | 12.6% | 7 s / 11 words | 1.7 GB |
| TS3007b | 55.9 min | 3+3 | 21:13 | 0.379 | — | 28.1% | **27.0%** | 1296/70/305 | 23.5% | 321 s / 485 words | 2.1 GB |

The full IS1009b ASR run did not finish before the host was needed for the other
runs; the 6 min sample is reported and marked as such.

The 431 s and 321 s "long gaps" are not silence that the VAD skipped. Re-scoring
the same outputs against the word-level reference shows **99.6% of ES2004b's
reference words and 99.9% of TS3007b's inside a decoded piece** (29 and 5 words
outside, after the pre-roll, post-roll and 10 s minimum). Inside the decoded
audio the hypothesis has only 0.77-0.80 words per reference word, uniformly
across speakers (0.81-0.89 per speaker in single-speaker regions), so no quiet
speaker explains it. The two measured causes are:

- **Overlap.** The hypothesis has 0.35 (ES2004b) and 0.30 (TS3007b) words per
  reference word in overlapping regions: 65-70% of overlapped words are lost to
  a single-stream model.
- **Backchannels and fillers.** `yeah` (-87 on ES2004b, -136 on TS3007b),
  `mm-hmm`, `okay`, `uh`/`um` are not transcribed. Parakeet has a clean-transcript
  style, and that is what these numbers measure.

Raising VAD recall (threshold 0.2) changed nothing (a 6 min ES2004b sample: 34.1%
vs 33.9% Whisper WER, 132.6 s vs 134.5 s of gaps), which is consistent with the
VAD already covering the speech. A 30 s fixed-chunk ablation on one 90 s clip
scored 40.2% Whisper WER versus 45.3% for VAD pieces (n = 179 words, one clip);
the forced 20 s flush may account for part of that, but neither number is a
reason to drop the VAD.

The remembered Parakeet TDT v3 figure of about 11% is the Open ASR Leaderboard
condition: AMI **IHM**, reference-segmented short utterances. Mix-channel
long-form audio with 18-22% overlapped words is a different task, and the
no-overlap column (22.1-23.5%) is the closest this harness gets to it. There is
no cheap ASR knob that closes the rest.

### What the old 240 s chunks did

The old harness is re-scored with the corrected scorer. The VAD rows are the new
pipeline; the 240 s rows are the same meetings from `results/asr-*-c240.json`.

| Design | Meeting | WER Whisper | Deletions | Long gaps |
|---|---|---|---|---|
| 240 s / 2 s, 8 threads | ES2002a | 24.5% | 453 | 270 s / 48 words |
| 240 s / 2 s, 8 threads | IS1000a | 41.8% | 986 | 488 s / 542 words |
| 240 s / 2 s, 8 threads | IN1014 | 23.1% | 1839 | 163 s / 254 words |
| VAD, 8 threads | ES2004b | 27.1% | 1575 | 431 s / 889 words |
| VAD, 3+3 threads | TS3007b | 27.0% | 1296 | 321 s / 485 words |
| VAD, 8 threads | IS1009b, 6 min | 22.5% | 230 | 7 s / 11 words |

The 240 s windows lose whole mid-chunk stretches; VAD pieces at most 22.2 s long
(observed maximum) do not have that failure, and they bound memory. The lesson is
about **input length and compute**, not about VAD recall: the pieces contain
99.6-99.9% of the reference words.

### Clean speech

| Sample | Audio | WER legacy | WER Whisper | del/ins/sub | RTF |
|---|---|---|---|---|---|
| LibriSpeech `test-clean`, 30 utterances | 3.7 min | 4.0% | **4.1%** | 4/4/17 | 0.346 |
| ES2002a single-speaker regions | 5.0 min | 26.1% | **24.1%** | 124/23/21 | 0.298 |

The read-speech result is inside the 2-5% band Parakeet publishes (from memory),
at its top end. The single-speaker AMI sample is still 24.1% because it keeps
AMI's spontaneous speech: fillers, false starts and backchannels that Parakeet
does not write. The
VAD is not the cause here either: with the sample's word times remapped to the
concatenated audio, **100% of its 705 reference words fall inside a decoded
piece**. The 54 s of hypothesis gaps are the model's deletions.

An earlier draft of this document said the VAD was dropping quiet speech on this
sample. That was wrong: `clean-sample` copied the source timestamps into the
concatenated wav, so the gap diagnostic compared two different timelines. The
script now shifts each word by the output offset of its region. The WER is
token-based and was unaffected; the gap and no-overlap figures were.

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
scorer reported. With the 0.25 s collar it is 20.8-25.7%.

The reference choice explains little. Against BUT's `only_words` RTTMs, the
reference pyannote uses, the same filtered outputs score 21.5% / 26.8% / 28.0%,
1-3 points lower. Confusion dominates: 8-20% of reference speech on `only_words`,
against 3-9% miss and 3-5% false alarm. That is a labelling problem between
speakers, not missing or invented speech.

The oracle marker shows where it comes from. Each raw cluster was mapped to the
reference speaker it overlaps most:

| Meeting (th 0.7) | Raw DER | Filtered DER | Oracle DER | Oracle confusion | Speech in top-4 clusters | Largest wrong cluster |
|---|---|---|---|---|---|---|
| ES2004b | 22.4% | 21.5% | **16.6%** | 3.2% | 91% | 51 s |
| IS1009b | 28.0% | 26.8% | **15.3%** | 5.0% | 82% | 216 s (a split of FIO089) |
| TS3007b | 28.9% | 28.0% | **13.8%** | 5.4% | 81% | 252 s (a split of MTD027ID) |

These rows were re-scored against the `only_words` reference by the second
review of this work. Against the manual-segment reference used in the table
above, the same oracle mapping gives 20.0% / 17.3% / 15.5% including miss and
false alarm, with confusion 3.3% / 4.8% / 5.6%; the conclusion is identical.

On all three test meetings the top four clusters are the four reference
speakers, each 81-91% pure. Segmentation and CAMPPlus embeddings work. What
fails is the final merge:

- a fifth cluster holding 9-11% of speech, which the 2%-of-time filter by design
  cannot touch;
- 66-132 s spread over 23-28 tiny clusters, reassigned **by time** rather than
  by voice.

ES2003b at 1.0 shows the opposite failure: complete linkage merged two speakers,
and the oracle is no better (confusion 32.6%). Over-clustering and merging by
centroid is the fix; see design condition 6.

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
off. The wall numbers are load-bound; the CPU ratio is the stable result. The
40 s ASR check went from 17.2 to 9.8 CPU-s and from 27.4 to 20.1 s wall at 2
threads. Spinning off is now the default for every benchmark run and should be
the engine default for the ASR, VAD and embedding sessions.

The 8-thread rows in the ASR table above predate the cgroup fix and oversubscribed
6 CPUs. That is why "8 threads" looked slower than 4 in an earlier draft; it is
not a property of ONNX Runtime.

A separate low-load check on a 90 s clip of ES2004b (480-570 s), run at 1 and 2
threads under `nice -n 19` with a 1-minute load of 1-11:

| Threads | Decode wall | CPU-s |
|---|---|---|
| 1 | 13.4-13.6 s | 13.2-13.3 |
| 2 | **8.0 s** best (13.0 s when load hit 11) | 13.8-15.5 |

That is **1.68x** from one to two threads and about 0.16 CPU-s per decoded
second at low load. The contended-host figures in this document are about 30%
inflated by preemption.

### Combined pipeline

The combined run starts ASR and diarization at the same time. On this host the
default is 3 ASR + 3 diarization threads (6 usable CPUs, `usable_cpus / 2`
clamped to 4).

| Meeting | Threshold | ASR wall | Diarization wall | Critical path | Total wall | Total CPU | Peak RSS | Load at start/end |
|---|---|---|---|---|---|---|---|---|
| TS3007b (55.9 min) | 1.0 | 21:13 | 15:19 | **21:13** | 21:13 | 944 s | 2.1 GB | 22.3 / 36.7 |

**This row ran at threshold 1.0 (8 raw clusters), not at the frozen 0.7.** It was
recorded before the freeze and is kept as the timing measurement; its DER
(31.7% filtered, 3 clusters) is not the pipeline's quality number. The frozen
0.7 diarization of the same meeting is in the table above (29.1%). `run-benchmark.sh`
now feeds the combined run the frozen 0.7.

Diarization finishes under the ASR wall, so the critical path is the ASR wall.
The 1 h projection from that critical path is **21.2 min** under this load. Do
not project from total CPU / 8, and do not subtract a standalone diarization run
from the combined CPU: `cpu_seconds()` is `RUSAGE_SELF`, so each stage's figure
in `combined` mode includes the other stage. Use standalone runs for stage CPU.

The stable rate is per **decoded** second: 0.16 CPU-s at low load and 0.21 on
this contended host. Dense meetings decode 1.07-1.11x their audio because of
pre/post-roll and the 10 s minimum, so one audio hour needs **600-900 ASR
CPU-s**. At 4 threads with 60-75% scaling that is **4-5.5 min**; on an idle
8-core machine the realistic estimate is **4-6 min per hour**, so the bar holds.
At 2 threads the measured 1.68x gives **about 6 min** of ASR, **6-7 min** with
the usual variance, which is what a 4-core transcript-first pass gets. The full
single pass on 4 cores, ASR then diarization, is **10-11 min**: over the bar.

## Engine design conditions

These are the concrete rules the Rust engine must follow. They are derived from
the measurements above, not from the model cards.

1. **VAD segmentation with a hard cap.** Segment ASR input with the sherpa-onnx
   Silero VAD. Pieces are at most 30 s, cut at the quietest point in the last
   part of the window, with 2 s pre-roll, 1 s post-roll and a 10 s minimum.
   `MAX_ASR_SECONDS = 60` must be asserted in code before every recognizer call.
   Never decode a whole file: the old whole-file attempt hit 10.9 GB at 21 min.
   De-duplicate overlapping pieces at the overlap midpoint and sort by time.
   The justification is compute and the long-input failure, not recall: measured
   VAD coverage is 99.6-99.9% of reference words. Budget about 1.1x the audio in
   decoded seconds. Remove the forced `flush()` every 20 s (0.83 s holes, larger
   than the 0.5 s join gap): use `split_run`'s quietest-point split, or make the
   join gap larger than the hole.
2. **Threads from the CPUs the process actually has.**
   `min(affinity, the process's own cgroup quota, available_parallelism)`. Read
   the quota by taking the path from `/proc/self/cgroup` and walking from it up
   to `/sys/fs/cgroup`, keeping the smallest `cpu.max`; the root file alone is
   wrong on this host (the harness reported 7 from the affinity mask, not the
   6-CPU quota). `core_id` inside a VM is the
   hypervisor's topology, so treat the `(package, core)` count as a fallback
   only. Default ASR to `min(4, max(1, usable / 2))`, diarization to
   `clamp(usable / 2, 2, 4)`, and run the stages concurrently. Never hard-code 8.
3. **Disable ONNX Runtime intra-op spinning.** sherpa-onnx 2.0.1 forwards keys
   prefixed `SessionConfig.` from the provider config file
   (`provider = "cpu:/path/to/config"`). Write
   `SessionConfig.session.intra_op.allow_spinning=0`. Measured on the 6 min
   slice: ASR 317.9 -> 78.4 CPU-s (-75%), 759.9 -> 93.9 s wall; diarization
   70.5 -> 39.3 CPU-s (-44%). Apply it to the VAD and embedding sessions too,
   not only the recognizer.
4. **CPU budget per decoded second.** ASR is **0.16 CPU-s per decoded second at
   idle and 0.21 under load**; decoded audio is about 1.1x the audio on dense
   meetings, so one hour is **600-900 ASR CPU-s** and diarization adds roughly
   **0.11-0.13 CPU-s per audio second**. Show a progress/ETA estimate from audio
   seconds and the measured rate, not from a fixed promise. Measure stage CPU
   only in standalone runs; `RUSAGE_SELF` in a combined run is not separable.
5. **4-core machines use transcript-first.** Two threads give 1.68x over one; ASR
   alone is about **6-7 min per audio hour**, then diarization needs about 3-4
   min more. Run ASR first with 2-3 threads, show the transcript, then run
   diarization in the background. A single pass with both stages on 4 cores is
   **10-11 min**, over the ten-minute bar; the transcript alone still fits.
6. **Diarization: over-cluster, then merge by centroid.** Keep the 0.7 cosine
   distance threshold and let complete linkage over-cluster. Compute one
   embedding per segment (sherpa-onnx `SpeakerEmbeddingExtractor` with the same
   model), merge cluster pairs whose centroid cosine distance is below a second
   threshold (average linkage on centroids), then reassign the remaining small
   clusters to the nearest centroid by embedding, not by time. A larger
   time-based filter cannot fix the fifth cluster (9-11% of speech) or the
   216-252 s splits. Test an English speaker-embedding model (sherpa-onnx ships
   `wespeaker_en_voxceleb_resnet34(_LM)`, the embedding pyannote 3.1 uses, and
   English VoxCEleb CAMPPlus/ERes2Net) before freezing CAMPPlus `zh-cn`. Never
   force a fixed `num_clusters`. Let the user merge, split or rename speakers;
   the filtered count is the default the UI should show, with an override.
7. **Acceptance test.** A release candidate passes when, on BUT's
   `AMI-diarization-setup` **test** meetings (`ES2004*`, `IS1009*`, `TS3003*`,
   `EN2002*`), scored against the `only_words` RTTMs with the same
   `der_pyannote`, with settings frozen on dev:
   - DER (pyannote.metrics, collar 0, overlap scored, full-file UEM) is at or
     below the measured baseline on the same meetings plus two points. The
     remembered pyannote 3.1 figure of about 18.8% (community-1 about 17%) is
     context, not a bar on these references.
   - WER with Whisper's English normalizer is set against measured baselines:
     an IHM-utterance sanity check near the remembered ~11%, and a mix
     long-form bar of **no-overlap WER <= baseline + 2**. AMI's 18-22%
     overlapped words make the ~11% leaderboard number unreachable on the mix
     channel.
   - one hour of audio completes in <= 10 minutes on a dedicated 8-core CPU and
     peaks below 8 GB;
   - the filtered speaker count is reported, and the user can correct it.

   `TS3007b` and `ES2003b` must not appear in the acceptance set: they are
   training meetings for pyannote. The normalizer must map AMI's `X_Y_`
   acronyms (done in `score.py`).

## Follow-up

This round corrects the document; it does not re-run anything. The two changes
that the numbers point to are tracked outside this PR:

- **Runewire task #18, diarization clustering and embedding model.** Implement
  over-cluster + merge by centroid + reassign by embedding, and compare CAMPPlus
  `zh-cn` with an English embedding model on the dev meetings. The oracle bounds
  the gain at roughly 21-28% -> 14-17% DER.
- **Runewire task #19, same-data baselines.** Run pyannote
  `speaker-diarization-community-1` and the Parakeet IHM-utterance check on the
  same meetings, so the acceptance bars in condition 7 are measured rather than
  remembered. ES2004b and IS1009b first; skip TS3007b, which pyannote trained on.

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
contains the load at start and end. It defaults to **4 threads**, runs the spin
ablation at 4 threads on both sides so the table is 4-vs-4, and gives the
combined run the frozen 0.7 threshold. The published ASR rows used 8 threads on
a 6-CPU quota; WER does not depend on the thread count, so a 4-thread rerun
reproduces the quality but not the wall times. The script creates the 6 min
slice, the clean samples, the VAD runs, the fixed-240 s ablation, the
thread/spin ablation, the dev threshold sweep, the held-out test runs and the
parallel pipeline, and ends with the markdown tables in
`$ROOT/results/repro/summary.md`.

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
  --threads 4 --threshold 0.7 --window-shift 0.2 --spin 0 \
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
- The cgroup fix in `main.rs` is this round's only harness-code change and is
  syntax-checked only: the machine was reserved for another project, so the
  binary was not rebuilt and no `usable_cpus` value has been observed from it.
  The expected value on this host is 6, and the defaults it produces are the
  3+3 already used by the combined run.
- The harness still forces the VAD's 20 s flush. Design condition 1 says to
  replace it with the quietest-point split or a larger join gap, but that is a
  behavior change that needs a fresh run, so it was not made this round.
- The small-cluster filter in the benchmark reassigns by temporal proximity
  because sherpa-onnx does not expose embeddings. The engine should use
  embeddings; the benchmark's DER after the filter is therefore an estimate of
  the engine's behavior, not an upper bound.
- AMI reference segments are manual `transcriber_start`/`transcriber_end`
  regions; DER numbers from word references differ by 1-3 points. The same
  reference type is used for every meeting and every threshold, and the
  `only_words` figures are labelled where they appear.
- Published pyannote, FluidAudio and Parakeet AMI figures quoted in this
  document are from memory and are marked as such; they are not measured here.
