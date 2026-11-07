# ASR baseline on AMI headset audio: the runtime, the channel, and the segmentation penalty

Task #19 of the Windows/Linux port, ASR half. `docs/fork/benchmark.md` reports
**27.0-27.1%** Whisper-normalized WER with sherpa-onnx Parakeet TDT 0.6b v3 int8
on the AMI **Mix-Headset** test meetings, while the Parakeet v3 model card
publishes **11.31%** on AMI **IHM** test. Both numbers were in the benchmark
document; nothing connected them.

This round decodes the **full 12,643-utterance IHM test split** (8.68 h), scores
everything the way the leaderboard does (summed per-utterance edit distances
over summed reference words), clusters every confidence interval by speaker and
by meeting, and replaces the old cross-reference comparison with a true
same-reference ablation on three full meetings.

Short answer: **the int8 runtime does not reproduce 11.31% on the full split.**
It scores **13.8% [95% CI 12.1-15.6, speaker]**, 2.5 points above the published
figure, and 11.31% falls outside the speaker and meeting intervals. The earlier
10.9% headline was a sampling artifact: the two easiest meetings supplied 69% of
the sample's words but only 14% of the split's. On identical words the room and
the other voices cost **+7.7 points** (Mix 17.6% vs IHM 10.0%). The long-form
pipeline adds a further **+15 to +21 points** on the same reference words in the
standing ablation, so segmentation, not the recognizer, is the dominant product
problem.

## Verdict

**The runtime is a few points behind the published figure on the full split.**
On the leaderboard's IHM condition the int8 ONNX artifact scores **13.8%** over
all 12,643 utterances (95% CI by speaker 12.1-15.6, by meeting 11.7-15.8)
against the published **11.31%**. The gap is **+2.5 points**, and 11.31% is
outside both intervals. The test-set-weighted estimate from the old sample,
**14.7% [12.8-17.0]**, predicted exactly this; the sample's unweighted 10.9%
was not a valid estimate of the split. What this run cannot say is *why* the
2.5 points are there: int8 quantization, the sherpa-onnx greedy TDT decoder, or
a normalizer/reference detail of the leaderboard pipeline are all still on the
table. A ~300-utterance A/B against the fp32 sherpa export or NeMo is the next
step (see "For the engine").

The 27% product number decomposes into two measured parts:

1. **Channel and crosstalk (oracle boundaries, same words):** IHM 10.0% ->
   Mix 17.6%, **+7.7 points [5.6-9.4, speaker]**. Solo windows cost +0.4; the
   windows where two or more speakers cover more than 20% of the span score
   **44.6%** on Mix, +27.0 points over the same IHM windows.
2. **Long-form segmentation (same reference words, full meetings):** oracle
   segments vs the engine VAD pipeline is **+17.5** (ES2004b), **+15.2**
   (IS1009b) and **+20.6** (EN2002a) points; **+18.0 [15.2-20.8]** over the
   three. The earlier 300-600 s ES2004b window gives **+13.4** (16.2% vs
   29.6%). The deletions are not spread evenly: 44-70% of a full meeting's
   deletions fall inside hypothesis holes of 3 s or more, and 82-89 of the
   110-136 decode pieces per meeting are starts forced by the 20 s VAD flush.

| Comparison | Utterances | WER (Whisper, per-utterance) | Source |
|---|---|---|---|
| Parakeet v3, AMI IHM test, published | 12,643 (full split) | **11.31%** | model card `model-index`, pinned below |
| This runtime, AMI IHM test, **full split** | 12,643 | **13.8% [12.1-15.6]** | measured here |
| This runtime, IHM, two paired meetings | 1,291 | 10.0% [8.1-12.3] | measured here |
| This runtime, Mix-Headset, same 1,291 utterances | 1,291 | 17.6% [14.8-20.7] | measured here |
| This runtime, Mix-Headset, full meetings | 2 meetings | 27.1% / 21.6% | `docs/fork/benchmark.md`, re-scored here |
| This runtime, full-meeting pipeline, third meeting | EN2002a | 43.1% | measured here |

## The published figure is pinned, not remembered

`docs/fork/benchmark.md` quotes "about 11%" from memory. The actual reference is
in the model card, fetched 2026-11-07:

- <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>
- raw README: <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/raw/main/README.md>
- `model-index` entry: dataset `AMI (Meetings test)`, type `edinburghcstr/ami`,
  config `ihm`, split `test`, `value: 11.31`, metric `Test WER`.
- The performance section states the WERs come from a "Transducer decoder with
  greedy decoding (without an external language model)" and lists AMI 11.31% in
  both the leaderboard table and the clean-SNR row. That is the same decoding
  strategy as this harness (greedy TDT, no LM, no beam, no punctuation model);
  only the numeric precision and the implementation differ.
- The same card is the source for the LibriSpeech 1.93/3.59 and Earnings-22
  11.42 figures the benchmark document quotes. It also lists AMI among the
  model's training data; that is a caveat on *both* numbers, not on the delta.

## Data

`edinburghcstr/ami`, config `ihm`, split `test`. This is the dataset and split
the Open ASR Leaderboard evaluates for AMI IHM, and the one the Parakeet model
card cites. No Hugging Face token is needed.

| | |
|---|---|
| Dataset | <https://huggingface.co/datasets/edinburghcstr/ami> |
| Config / split | `ihm` / `test` |
| Revision | `46f28f2503e2ec48f8867a84eef356c70476beab` |
| Files | `ihm/test-0000{0..3}-of-00004.parquet`, 1.29 GB |
| SHA-256 | `d95920dc…602e0`, `07a2b1c4…2325`, `83cde21e…f8`, `3312bb79…80d7` (full hashes in `fetch-data.sh`) |
| Rows / audio | 12,643 utterances, 8.68 h, 16 meetings, median 1.5 s |
| Audio | 16 kHz mono, IEEE float32 WAV, one utterance per row, from the speaker's own headset mic (`H00`-`H03`) |
| Text | AMI manual words, uppercase, no punctuation, no digits |

**Full split, not a sample.** Every row of the split is decoded. The old
2,131-utterance sample is kept only as the cross-check that shows why the first
headline was wrong; its test-set-weighted estimate (14.7%) is reported next to
the full-set number.

**Mix pairing.** All 1,291 rows of **ES2004b (688)** and **IS1009b (603)** — the
two test meetings whose Mix-Headset audio exists in the task-#2 evidence
directory — are cut at exactly the same `[begin_time, end_time)` ranges from
`<meeting>.Mix-Headset.wav` and decoded with the same recognizer and settings.
There is no filter: these are every row of those two meetings, and the remaining
840 sample rows are IHM-only (`skipped_mix_missing_audio: []`). The review
checked the pairing against the manifest, the times encoded in the `audio_id`,
and the decoded `audio_seconds` (0 mismatches, max difference 0.2 ms). The
reference text is the same row object on both sides. The only differences are
the microphone (single close headset vs. the room mix), the other speakers in
the mix, and the far-field acoustics.

**Audio and provenance.** Data and models live outside the repo. The parquet
files are pinned by revision and SHA-256 in `fetch-data.sh`; each
`provenance.json` records the hashes actually read. No audio or binary is
committed.

## Method

- **Model**: the same files as the benchmark, pinned by SHA-256 in
  `fetch-data.sh`: Parakeet TDT 0.6b v3 int8 (`encoder.int8.onnx` 652 MB,
  `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt`).
- **Decoding**: `OfflineRecognizer`, CPU provider, greedy TDT, `--spin 0`
  (ONNX Runtime intra-op spinning off), 2 threads. No VAD, no chunking, no
  padding: each IHM row is decoded exactly as the dataset defines it, which is
  what the leaderboard does. The `batch` subcommand builds the recognizer once
  for thousands of clips. Its output is **resume-safe**: it reads the ids
  already in the JSONL, drops a partial last line, and appends only the missing
  rows, so a crash never loses work.
- **Corpus WER**: `sum_e (S_e + D_e + I_e) / sum_e N_e`, i.e. the sum of
  per-utterance edit distances over the sum of reference words (the leaderboard
  method), not one alignment over concatenated text. Every number in this
  document uses it.
- **Normalizer**: Whisper's `EnglishTextNormalizer` (via the
  `whisper-normalizer` package) on both sides. The benchmark's legacy normalizer
  is reported next to it. AMI's `T_V_` acronym rewrite is a no-op on this
  dataset (its text has no underscores).
- **Empty references**: 246 of the 2,131 sample references (990 of the 12,643
  full-split references, 7.8%) normalize to empty, mostly `Mm-hmm` and `Mm`.
  The headline keeps their hypothesis tokens as pure insertions in the edit sum
  and adds 0 reference words, which is what summing per-pair jiwer measures
  does. Dropping those rows instead gives 10.8% on the sample and 13.6% on the
  full split; the paired Mix number moves from 17.6% to 16.9%. The convention is
  stated here so the number is reproducible either way.
- **Confidence intervals**: percentile bootstrap, 2,000 resamples, seed 19.
  The primary cluster is the **speaker** (63 in the full split) and the report
  also gives the **meeting** cluster (16). AMI reuses role-like speaker ids
  across meetings (`MEE073` is a different person in `EN2002a` and `EN2002d`),
  so the speaker cluster is `(meeting, speaker_id)`. An utterance-iid interval
  is kept only for comparison; it is much too narrow.
- **Crosstalk buckets**: the share of the window with **two or more reference
  speakers active, including the target**, on a 10 ms grid. This is an oracle
  diagnostic (it uses the reference transcript). Rows are split by duration
  band, and backchannels under 1 s are reported separately. The old
  "other-speaker > 20%" bucket is kept as a cross-check of the review's
  numbers.
- **Same-reference ablation**: for each meeting, the oracle path decodes the
  dataset's own segments cut from Mix-Headset; the pipeline path is the
  full-meeting engine VAD decode. Each pipeline word is assigned to exactly one
  segment window (the shortest window containing its midpoint); both sides are
  scored **per utterance against the same HF segment text**. The old 300-600 s
  ES2004b check used one `words.xml` reference for both paths and is reported
  separately.
- **No timing claims.** The host is shared; every heavy step ran under
  `nice -n 19` with 2 threads. Throughput is not a result of this document.

## Results

### The full IHM test split

| Set | Utterances | Audio min | WER (Whisper) [95% CI] | del / ins / sub | Ref words |
|---|---|---|---|---|---|
| IHM, full test split | 12,643 | 520.9 | **13.8% [12.1-15.6]** | 6403 / 1694 / 4305 | 89,645 |
| — the two paired meetings | 1,291 | 70.7 | 10.0% [8.1-12.3] | 622 / 221 / 433 | 12,814 |
| — the other 14 meetings | 11,352 | 450.2 | 14.5% [12.6-16.3] | 5781 / 1473 / 3872 | 76,831 |
| Mix-Headset, same 1,291 utterances | 1,291 | 70.7 | 17.6% [14.8-20.7] | 1094 / 440 / 725 | 12,814 |

The interval in brackets is the speaker-cluster one. The meeting-cluster
interval for the full split is 11.7-15.8, and the (wrong) utterance-iid one is
13.4-14.3. 11.31% is below all three. The point estimate is stable: the legacy
normalizer gives 14.9% and the filler-free diagnostic 13.8%.

**Per-meeting** (all 16 meetings are complete now):

| Meeting | Utterances | WER [95% CI, speaker] |
|---|---|---|
| EN2002a | 1367 | 17.1% [10.3-24.6] |
| EN2002b | 955 | 18.5% [12.9-26.3] |
| EN2002c | 1469 | 17.3% [11.5-22.9] |
| EN2002d | 1332 | 19.3% [14.3-24.5] |
| ES2004a | 361 | 10.9% [7.1-17.5] |
| ES2004b | 688 | 9.6% [7.5-13.3] |
| ES2004c | 723 | 9.8% [6.3-14.3] |
| ES2004d | 894 | 11.1% [8.4-15.6] |
| IS1009a | 291 | 16.0% [12.6-27.3] |
| IS1009b | 603 | 10.4% [7.8-13.8] |
| IS1009c | 495 | 10.3% [7.1-12.3] |
| IS1009d | 689 | 11.8% [8.9-20.1] |
| TS3003a | 425 | 17.8% [16.5-36.0] |
| TS3003b | 659 | 10.3% [7.5-18.4] |
| TS3003c | 630 | 11.2% [7.2-17.7] |
| TS3003d | 1062 | 14.8% [11.9-22.8] |

The four `EN2002*` meetings hold 37% of the split's words and score 17-19%,
while ES2004b and IS1009b — 14% of the words — score 9.6% and 10.4%. That is
exactly the bias the first sample had.

### Cross-check: the old sample weighted to the test set

The deterministic 2,131-row sample scored 11.2% unweighted. Re-weighting each
meeting's sample WER by its share of the full split's Whisper-normalized
reference words (stratified bootstrap over utterances within meeting) gives
**14.7% [12.8-17.0]** — the review's number (it gave 14.7 [12.9-17.0]) and
close to the 13.8% the full decode measured. The gap between the sample and the split is entirely the
meeting mix: the sample is 69% paired-meeting words, the split is 14%.

### IHM vs Mix on the same utterances

| Set | Utterances | IHM WER | Mix WER | Delta [95% CI, speaker] |
|---|---|---|---|---|
| ES2004b | 688 | 9.6% | 18.8% | +9.2 [+7.5, +11.0] |
| IS1009b | 603 | 10.4% | 16.3% | +6.0 [+3.5, +8.6] |
| Both | 1,291 | 10.0% | 17.6% | **+7.7 [+5.6, +9.4]** |

The paired delta uses the same rows on both sides, so no sampling caveat
applies. The utterance-cluster interval is 6.2-9.1 (the review's 6.3-9.1) and
the meeting-cluster one is 6.0-9.2; the speaker interval is wider because it
treats the eight speakers as the independent units.

### Crosstalk: share of the window with 2+ speakers (including the target)

| Share | Duration band | Utterances | IHM WER | Mix WER | Delta |
|---|---|---|---|---|---|
| 0 (solo) | <1 s (backchannel) | 78 | 30.8% | 30.0% | -0.8 |
| 0 (solo) | 1-3 s | 78 | 10.2% | 15.3% | +5.1 |
| 0 (solo) | 3-8 s | 154 | 7.3% | 7.2% | -0.0 |
| 0 (solo) | >=8 s | 60 | 5.8% | 5.6% | -0.3 |
| 0-20% | <1 s (backchannel) | 7 | 28.6% | 57.1% | +28.6 |
| 0-20% | 1-3 s | 36 | 12.1% | 14.8% | +2.7 |
| 0-20% | 3-8 s | 90 | 7.3% | 7.1% | -0.2 |
| 0-20% | >=8 s | 73 | 5.5% | 6.9% | +1.3 |
| 20-50% | <1 s (backchannel) | 28 | 39.4% | 45.5% | +6.1 |
| 20-50% | 1-3 s | 43 | 11.1% | 39.9% | +28.8 |
| 20-50% | 3-8 s | 60 | 13.4% | 29.6% | +16.2 |
| 20-50% | >=8 s | 15 | 4.3% | 13.0% | +8.6 |
| 50-100% | <1 s (backchannel) | 391 | 45.3% | 96.3% | +51.0 |
| 50-100% | 1-3 s | 136 | 27.2% | 70.5% | +43.2 |
| 50-100% | 3-8 s | 40 | 10.5% | 37.1% | +26.7 |
| 50-100% | >=8 s | 2 | 11.4% | 38.6% | +27.1 |
| all | all | 1,291 | 10.0% | 17.6% | +7.7 |

Read the duration bands, not just the row counts. In **solo** windows the
far-field room is worth +0.4 points (7.6% vs 8.0% in the old three-bucket cut)
and is negligible in every band of 1 s or more. The +27 effect is in windows
where another speaker covers more than 20% of the span: IHM 17.6%, Mix 44.6%,
+27.0 (the review's 44.6 is reproduced). Within length bands the direction is
the same and the effect is real: at 20-50% share, 3-8 s goes 13.4% -> 29.6%; at
50-100% share, 3-8 s goes 10.5% -> 37.1%.

The `<1 s (backchannel)` rows are not product loss. 419 of the 715 old ">20%"
rows are under 1 s (median 0.73 s): a 0.7 s window assigned to speaker A while
speaker B holds the floor. Mix scores 45.5% in the 20-50% share band and 96.3%
in the 50-100% band because the decoder transcribes the dominant speaker —
correct words for the meeting, wrong words for that window. Report them
separately, as above, and do not add them to the crosstalk penalty.

### Same-reference ablation: oracle segments vs the engine VAD pipeline

Each meeting is decoded twice on the **same reference words** (the HF segment
text): once with the dataset's own segment boundaries cut from Mix-Headset
(oracle), once with the full-meeting engine VAD pipeline, assigning each
pipeline word to the segment window that contains it.

| Meeting | Segments | Oracle WER | VAD WER | Delta [95% CI, speaker] |
|---|---|---|---|---|
| ES2004b | 688 | 18.8% | 36.3% | +17.5 [+16.0, +19.0] |
| IS1009b | 603 | 16.3% | 31.6% | +15.2 [+10.0, +22.8] |
| EN2002a | 1367 | 37.1% | 57.7% | +20.6 [+17.1, +25.0] |
| All three | 2,658 | 25.0% | 43.1% | **+18.0 [+15.2, +20.8]** |

The error shape is the point: the oracle path is 2,359 deletions / 1,088
insertions / 1,728 substitutions; the pipeline path is **6,787 deletions** /
842 insertions / 1,273 substitutions. Segmentation destroys words; it does not
mostly substitute them. The 300-600 s ES2004b check, one `words.xml` reference
for both paths, gives **16.2% vs 29.6% (+13.4)**, 59 vs 197 deletions on 747
reference words — the same direction and a similar size.

Two honest qualifications. The strict per-utterance assignment also charges a
pipeline word to the neighbouring segment when the window is ambiguous, so it is
harsher than a whole-meeting alignment: scored the benchmark's way (one
alignment of the meeting against `words.xml`), the pipeline is 27.1% (ES2004b),
21.6% (IS1009b) and 43.1% (EN2002a), i.e. +8.3 / +5.3 / +6.0 over the oracle
column. Both readings say segmentation is the largest remaining error source;
the standing check uses the per-utterance one because that is the same
estimator as every other WER here. And EN2002a is a hard meeting in the mix
(37.1% oracle), not a misaligned file: its headset IHM WER is 17.1% and its
segments decode correctly.

### Why the long-form path loses words

The engine cuts continuous speech every 20 s (`VAD_FLUSH_SECONDS = 20`). The
VAD then needs about **0.83 s to re-detect**, which is larger than
`VAD_JOIN_GAP_SECONDS = 0.5`, so `group_vad_pieces` never rejoins the two
pieces. Every following piece starts with a 2 s pre-roll that is itself
mid-speech; Parakeet can return zero words for such a piece, and the overlap
de-duplication then leaves a hole of roughly one piece length (~20 s). The
quiet-point split in `split_run` never runs because the pieces are already under
30 s.

Quantified on the three full meetings:

| Meeting | Hyp words | Flush pieces / pieces | Gaps >=3 s | Ref words in gaps | Deletions | In gaps | At piece start (gaps / words) | At flush start (gaps / words) | 15 s+ holes (gaps / words) |
|---|---|---|---|---|---|---|---|---|---|
| ES2004b | 5201 | 89/130 | 53 | 1104 | 1575 | 70% | 44 / 924 | 23 / 685 | 7 / 265 |
| IS1009b | 4972 | 83/110 | 29 | 538 | 1080 | 50% | 27 / 501 | 16 / 440 | 5 / 236 |
| EN2002a | 4628 | 82/136 | 72 | 1334 | 3043 | 44% | 54 / 1000 | 34 / 772 | 2 / 96 |

- **82-89 of the 110-136 pieces per meeting are flush starts**, not natural
  silence boundaries.
- **44-70% of all deletions fall inside a hole of 3 s or more**; **33-59%** of
  the deletions are in holes that contain or start at a piece boundary, and
  **25-43%** in holes at a flush-created boundary.
- The signature is the 20 s hole that starts within 0.5 s of a de-duplication
  midpoint. ES2004b 1495.0-1515.4 s (81 reference words), IS1009b
  1691.1-1711.3 s (60), EN2002a 1897.3-1911.2 s (77) are examples. In the
  300-600 s ES2004b window, 7 holes of 3 s or more hold 162 of the 197
  deletions (the review's count was 165); 5 of the 7 contain or start at a piece
  start, and 59.8-69.1 s is a solo speaker on clean audio that the oracle path
  transcribes verbatim while the pipeline emits nothing until "As I said" at
  69 s.
- The remainder are holes inside a piece (VAD marking speech as silence and
  pieces that return almost no words). The VAD threshold is not the lever:
  threshold 0.2 vs 0.5 on the 6-minute ES2004b sample gives 34.1% vs 33.9%,
  with the same words in gaps (160), and the VAD covers about 99% of reference
  words once pre/post-roll is counted.

## For the engine

These are the reviewer's six recommendations, kept verbatim in substance. They
feed Runewire task #21.

1. **Stop forcing cuts in the middle of speech.** Remove the 20 s `vad.flush()`
   while speech is detected, or treat flush-induced boundaries as joinable: tag
   them, or raise the join gap to >= 1.0 s for them. Then let `split_run` cut
   long runs at the quietest point. This targets the holes above directly.
2. **Shorter pieces, starting at silence.** Target pieces of 8-15 s, cut at VAD
   silences of at least 0.3 s, instead of 20-30 s runs. Choose the pre-roll start
   at the quietest point within the 2 s before the piece rather than a fixed
   -2 s, so no piece starts mid-word. Keep the 1 s post-roll.
3. **Detect holes and re-decode them.** After decoding, if the word timestamps
   show a gap of 3 s or more inside a VAD-speech span, re-decode just that
   sub-span on its own, starting at a quiet point. This costs a few percent more
   ASR time and targets the deletion mode (44-70% of full-meeting errors).
4. **Split on diarization speaker turns.** The pipeline already runs
   diarization. Cut ASR pieces at speaker-change points so each piece is mostly
   one speaker. Do this after steps 1-3, because the biggest holes are in solo
   stretches.
5. **Do not tune the VAD threshold.** 0.2 vs 0.5 makes no difference, and VAD
   coverage loses only about 1% of words.
6. **Make the same-reference ablation a standing regression check** on 3 or more
   meetings, including an `EN2002*` one, scored per utterance. Accept an engine
   change only if it narrows that gap. For the engine bar, keep the full IHM
   split in the loop (it is now decoded once) before fixing a threshold.

## Limitations

- **The 2.5-point IHM gap is unexplained.** int8, the sherpa greedy decoder and
  leaderboard normalizer/reference details are all candidates. This run measures
  the gap; it does not attribute it.
- **Ablation scoring is strict.** Assigning a pipeline word to one segment
  window is harsher than the benchmark's whole-meeting alignment; both numbers
  are given. The ablation reference is the HF segment text, so words outside
  every segment are outside both sides.
- **One of the three ablation meetings is hard.** EN2002a is 17.1% on headset
  and 37.1% on oracle Mix; it is included deliberately as the `EN2002*`
  requested by the review, not as a typical meeting.
- **Oracle boundaries are a best case.** The paired Mix number (17.6%) cuts at
  the dataset's reference boundaries and is not the product's expected WER.
- **int8 vs bf16.** The published 11.31% is not this artifact. That comparison
  is deliberate: it is the CPU artifact the product ships.
- **Overlap buckets use the reference transcript** to decide who is talking;
  they diagnose where the loss is and cannot be computed at runtime.
- **No LM, no punctuation, no casing** on either side. WER only.
- **No timing claims.** The host was shared; see `docs/fork/benchmark.md`.

## Reproduction

```bash
export ROOT=/path/to/evidence/tinta-fork/2
export ROOT19=/path/to/evidence/tinta-fork/19
scripts/fork/bench/fetch-data.sh      # adds the pinned IHM parquet (~1.3 GB)
scripts/fork/bench/run-ihm-baseline.sh
```

`run-ihm-baseline.sh` builds the manifests (sample, full split, oracle
segments), decodes the IHM sample and Mix, then the full split (resume-safe,
seeded from the sample), decodes the missing oracle segments and full-meeting
pipeline runs, and prints every table above. Every heavy step is `nice -n 19`
with `THREADS=2`. One decode by hand:

```bash
scripts/fork/bench/target/release/tinta-bench batch \
  --manifest $ROOT19/data/ihm-baseline-full/manifest-ihm.tsv \
  --output $ROOT19/results/ihm-full-decode.jsonl \
  --encoder $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx \
  --decoder $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx \
  --joiner  $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx \
  --tokens  $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt \
  --threads 2 --spin 0

$ROOT/venv/bin/python scripts/fork/bench/ihm-baseline.py score \
  --rows $ROOT19/data/ihm-baseline-full/rows.json \
  --ihm-jsonl $ROOT19/results/ihm-full-decode.jsonl
```

Re-running the decode is safe: existing ids are skipped, and a partial last line
is dropped.

## Files

| Path | Purpose |
|---|---|
| `scripts/fork/bench/ihm-baseline.py` | prepare (sample/full), per-utterance WER + clustered bootstrap, weighted estimate, ablation, cause analysis |
| `scripts/fork/bench/run-ihm-baseline.sh` | end-to-end: manifests, all decodes, every table |
| `scripts/fork/bench/src/main.rs` | `batch` subcommand (one recognizer, many clips, resume-safe JSONL) |
| `scripts/fork/bench/fetch-data.sh` | IHM parquet download, revision + SHA-256 pinned |

Results and logs: `evidence/tinta-fork/19/results/` and
`evidence/tinta-fork/19/logs/`. The numbers in this document come from
`ihm-full-summary.json`, `ihm-summary.json`, `ihm-weighted-estimate.json`,
`ablation-summary.json` and `cause-analysis.json`, all produced by the script
above.
