# ASR experiments: the fp32 model, speaker splits, and stream-level scoring

Task #23 of the Windows/Linux port, stacked on the task #21 segmentation work
(PR #16). The pipeline now scores 29.4 / 27.4 / 53.3 % per-utterance WER on
ES2004b / IS1009b / EN2002a (37.9 % over the three), **+4.3 points** from the
single-stream oracle, and pays **+31 to +137 %** ASR time for the hole
re-decode. Three questions remained: is the int8 decoder the cause of the
dropped stretches, can splitting pieces at diarized speaker changes recover
backchannels, and does a stream-level metric tell the same story.

**Short answer.** The non-int8 model is the biggest available lever and it is a
model swap, not a segmentation rule:

- the documented failing window (ES2004b 235.4-248.8 s) returns **no words** with
  the int8 artifact and its full 32 words with fp32; the hole re-decode exists to
  paper over that failure;
- the full pipeline with fp32 and the re-decode scores **26.6 / 25.3 / 50.7 %
  (35.4 %)** per utterance, **1.8 points from the single-stream oracle**
  (+0.6 / +1.9 / +2.8 - the re-scoped task #21 target is met), and **20.7 %
  tcpWER** at a 5 s collar against 26.1 % for int8;
- it costs a 2.5 GB model, 2.9 GB peak RSS (1.2 GB for int8) and **1.1-1.6x**
  the int8 ASR time; the re-decode still earns its 30-74 % on top;
- on a 1,000-utterance sample of AMI IHM it moves WER from 14.94 % to 13.79 %
  (**-1.15 points**, 95 % CI -2.63 to +0.48 by speaker), about half of the
  2.5-point gap to the published 11.31 % - and it is not statistically certain.

Speaker splits are a dead end in this form. The VAD already closes every
backchannel into its own piece: the coarse runs are never closer than ~0.93 s,
so the 0.5 s join gap never fires, and a 0.2 s VAD pass finds exactly **one**
extra split per meeting. WER moves by 0.1 points, inside the scoring noise.

The stream metric changes the picture. With meeteval, the same pipeline scores
**26.1 % tcpWER (collar 5 s)** and the *single-stream oracle* **23.6 %**; with
fp32 it is **20.7 %**. The oracle is not an upper bound under a stream metric:
it is a decode of short reference clips, and the pipeline's long-window decode
is simply better per word. **The per-utterance midpoint scorer overstates the
segmentation damage by about 12 points** (35.4 % vs 20.7 % for the same run).
tcpWER should be the standing check for transcript quality; the per-utterance
number stays as the segmentation/attribution diagnostic.

## The non-int8 model (experiment a)

### Pin

The only non-int8 sherpa-onnx export is the fp32 package; the fp16 repository
(`csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp16`) has no model files.

| | |
|---|---|
| Repository | <https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3> |
| Revision | `5fcc0122fe87e8365e339e467af90c67b746218f` |
| Files | `encoder.onnx` (41,838,346 B, sha256 `36b75836...48512`), `encoder.weights` (2,435,420,160 B, `3af3f51a...f39a8`), `decoder.onnx` (47,233,744 B, `caed31af...fdbfd`), `joiner.onnx` (25,286,331 B, `0f294e5b...90340`), `tokens.txt` (93,939 B, `d5854467...3c35d`) |
| Download | `scripts/fork/segmentation/fetch-model-fp32.sh` pins the commit and verifies every hash |
| Layout | the engine manifest names the int8 files, so the fp32 files are linked as `encoder.int8.onnx` etc.; `encoder.weights` sits next to the model, which is where ONNX Runtime finds the external data |

The int8 artifact is unchanged from task #2: `encoder.int8.onnx` 652,184,281 B
(`acfc2b44...`), `decoder.int8.onnx` 11,845,275 B, `joiner.int8.onnx` 6,355,277 B.

### The documented failing window

`--window 235.4:248.8` on ES2004b, 13.40 s, 2 threads:

| Model | Words | Output |
|---|---|---|
| int8 | **0** | empty text |
| fp32 | **32** | `the channel um or the volume, everything is just um some sort of signal to change the state of the T V or other appliance that it's sending the signal to` |

Task #21 measured the same int8 window returning `Okay. Um`; this run returns
nothing at all, so the failure is at least as bad as documented. fp32 recovers
the whole stretch, from 0.24 s to the end of the window, with no hole.

### The full pipeline

Same engine code, same pieces, same hole re-decode; only the model files differ.
`runs` is the same first pass with `gap_redecode_seconds = 0`; `fp32-default`
keeps the shipped 3 s re-decode.

| Variant | ES2004b | IS1009b | EN2002a | All three | vs single-stream oracle | Pieces (ES) |
|---|---|---|---|---|---|---|
| int8 `default` (task #21) | 29.4 % | 27.4 % | 53.3 % | 37.9 % | +4.3 | 79 |
| **int8 `runs`** (no re-decode, task #21) | 33.8 % | 29.8 % | 57.8 % | 41.8 % | +8.2 | 79 |
| fp32 `runs` (no re-decode) | 27.2 % | 26.2 % | 52.4 % | **36.5 %** | +2.9 | 79 |
| **fp32 `default`** (re-decode on) | **26.6 %** | **25.3 %** | **50.7 %** | **35.4 %** | **+1.8** | 79 |
| per-turn oracle | 18.8 % | 16.3 % | 37.1 % | 25.0 % | | |
| single-stream oracle | 26.0 % | 23.4 % | 48.0 % | 33.6 % | | |

Deletion/insertion/substitution counts over the three meetings: int8 default
5,353 / 960 / 1,524; fp32 runs 4,852 / 1,174 / 1,520; fp32 default
**4,449 / 1,244 / 1,625**. fp32 removes 904 deletions and pays 284 more
insertions: it hears words int8 drops, including fillers and repeats.

The hole re-decode still matters with fp32: 35.4 % vs 36.5 % is 1.1 points per
utterance (**2.0 points tcpWER**, 20.7 % vs 22.7 %). On the hard meeting the
re-decode is the difference between missing and meeting the 3-point target:
EN2002a is +2.8 from the single-stream oracle with it and +4.5 without.

The re-decode finds far fewer holes with fp32 - 9 / 6 / 36 holes against 44 / 18
/ 82 for int8 - so most of the int8 re-decode cost existed to repair the int8
decoder.

### The IHM sample

A fixed random sample of 1,000 rows of the AMI IHM test split (seed 23, 6,867
normalized reference words), decoded with the task #19 batch harness (greedy
TDT, no VAD, 2 threads). int8 comes from the cached full-split decode; fp32 was
decoded now on the same files.

| Model | WER | del / ins / sub | Paired errors vs int8 |
|---|---|---|---|
| int8 | 14.94 % | 485 / 160 / 381 | - |
| fp32 | **13.79 %** | 341 / 247 / 359 | **-79** (7.7 % of int8's errors) |

The paired delta is **-1.15 points, 95 % CI -2.63 to +0.48** (speaker-cluster
bootstrap, 2,000 resamples, 63 clusters): the direction is right, the evidence
is not conclusive at 1,000 utterances. Applied to the full-split int8 number
(13.8 %), the sample delta would leave about **12.6 %**, roughly half of the
2.5-point gap to the published 11.31 %. The remaining gap is not obviously
quantization; a matched NeMo decode would be needed to separate the decoder,
the normalizer and the runtime.

### Cost and size

| | int8 | fp32 |
|---|---|---|
| Model files | 641 MB | **2.5 GB** |
| Peak RSS, short-clip batch | 1.2 GB | **2.9 GB** |
| IHM sample RTF, 2 threads (relative) | 0.086 (1.0x) | 0.130 (**1.5x**) |
| ES2004b pass, 2 threads | 310 s | 498 s default / 286 s runs |
| IS1009b pass | 256 s | 341 s / 259 s |
| EN2002a pass | 459 s | 514 s / 396 s |
| Same-round relative cost | 1.0x | **1.1-1.6x** default, re-decode adds 30-74 % |

The host is shared: the same 79-piece int8 plan measured 318 s in the task #21
round and 408 s in an early run of this round (+29 %), so only the ratios are
used; there is no throughput claim here. Memory
is the harder constraint for a 4-core laptop: the model alone is 2.9 GB, and a
one-hour meeting holds about 0.5 GB of audio during finalize, so 8 GB is the
realistic floor. The engine's current worst case (12 h tracks, 5.5 GB of audio)
already needs more than that with either model.

### What fp32 does to the backchannels

Rows shorter than 3 s, scored per utterance (the cells are the whole corpus of
short rows, 1,872 rows / 5,813 reference words):

| Variant | Empty hypotheses | Reference words lost in them | Short-row del / ins / sub |
|---|---|---|---|
| int8 default | 714 | 1,408 | 2,514 / 691 / 927 |
| speaker split (int8) | 715 | 1,409 | 2,510 / 693 / 932 |
| fp32 runs | 574 | 1,037 | 2,088 / 891 / 997 |
| fp32 default | **511** | **831** | **1,938 / 929 / 1,082** |

fp32 recovers **577** of the 1,408 backchannel words int8 drops (41 %), which is
what a precision change should look like. It also inserts more in short rows
(929 vs 691), so the short-row error total moves much less than the deletions.

## Speaker splits (experiment b)

### What was measured

The engine's own diarizer (`SpeechPass::diarize`, the task #15 post-clustered
path) ran on all three meetings: **4 speakers, 269 / 241 / 524 turns**,
138-155 s per meeting at 2 threads. Then the task #21 piece plan was split at
every diarized speaker change that falls inside a silence of at least 0.2 s:

1. a second VAD pass with `min_silence_duration = 0.2` finds the short
   silences the first pass bridged;
2. a boundary is used only when it lies inside a first-pass VAD run (inside
   continuous speech, the place a token can be swallowed), across two different
   diarized speakers;
3. the piece is cut at the midpoint of that silence, and the same first pass
   and hole re-decode run on the new plan.

The rule lives in `scripts/fork/segmentation/src/main.rs` (`--split-turns`,
`--split-gap`), not in the engine, because it lost.

### Result

| Meeting | Pieces default -> split | WER default -> split | tcpWER default -> split | Backchannel words lost |
|---|---|---|---|---|
| ES2004b | 79 -> 80 | 29.4 -> 29.5 | 19.7 -> 19.7 | 262 -> 263 |
| IS1009b | 61 -> 62 | 27.4 -> 27.4 | 17.9 -> 17.9 | 179 -> 178 |
| EN2002a | 84 -> 85 | 53.3 -> 53.3 | 37.8 -> 37.8 | 967 -> 968 |
| All three | 224 -> 227 | 37.9 -> 37.9 | 26.1 -> 26.1 | 1,408 -> 1,409 |

Exactly **one** extra cut per meeting, 36-53 words re-timed at the split, and no
WER change in either metric. The fine VAD at 0.2 s finds 63 / 38 / 66 regions
against the coarse 57 / 37 / 66: Silero needs about 0.93 s after speech stops to
close a run whatever `min_silence_duration` says, so a real 0.2 s pause is
almost never inside a run.

The lesson is that the backchannel hypothesis from the task #21 review was
wrong. The short rows are not swallowed by long pieces: they already have their
own piece. They are missing because the **decoder** drops them (fp32 recovers
41 % of the words) or because the VAD never detected them (the review counted
52 such words). Splitting pieces cannot recover either.

Adopting the rule would also not be cheap: the engine transcribes a track before
it diarizes it, so speaker-aware planning would require reordering `finalize`
(diarize first), which moves the progress events, keeps the diarization model
loaded during transcription, and adds a full diarization pass before the first
word appears. Measured gain: zero. Not adopted.

## Stream-level scoring (experiment c)

### Method

`scripts/fork/segmentation/score-meeteval.py` scores the same reference and the
same three hypothesis streams as `score.py` with meeteval 0.4.3:

- **tcpWER** with a **5 s collar**: time-constrained minimum-permutation WER;
  a hypothesis word may move up to the collar and still match.
- **cpWER**: concatenated minimum-permutation WER (no time constraint).
- Text is normalized with the same Whisper `EnglishTextNormalizer` and AMI
  acronym join as `score.py`.
- Two reference views. **Merged**: every reference row is one speaker, so the
  metric measures word accuracy only - the fair view of a single mixed channel.
  **Speaker**: rows keep their speakers, so tcpWER also scores speaker
  attribution; an oracle that decodes each turn can use it, a single stream
  cannot.

The merged reference still contains both sides of every overlap (9.7 % of the
meeting time on ES2004b), which no single stream can produce. That inflates the
absolute value equally for every stream and is the price of scoring a mixed
channel without diarization.

### Results

Per-utterance WER next to tcpWER/cpWER (merged reference, collar 5 s):

| Variant | ES2004b per-utt / tcp / cp | IS1009b | EN2002a | All per-utt / tcp / cp |
|---|---|---|---|---|
| int8 `before` (task #21) | 38.0 / 30.9 / 30.8 | 31.5 / 23.5 / 23.4 | 57.2 / 44.3 / 44.2 | 43.4 / **33.8** / 33.8 |
| int8 `default` | 29.4 / 19.7 / 19.6 | 27.4 / 17.9 / 17.9 | 53.3 / 37.8 / 37.8 | 37.9 / **26.1** / 26.0 |
| speaker split | 29.5 / 19.7 / 19.7 | 27.4 / 17.9 / 17.9 | 53.3 / 37.8 / 37.8 | 37.9 / 26.1 / 26.0 |
| fp32 `runs` | 27.2 / 15.4 / 15.3 | 26.2 / 15.5 / 15.5 | 52.4 / 34.4 / 34.3 | 36.5 / 22.7 / 22.6 |
| **fp32 `default`** | **26.6 / 14.2 / 14.1** | **25.3 / 14.5 / 14.5** | **50.7 / 30.9 / 30.9** | **35.4 / 20.7 / 20.6** |
| single-stream oracle | 26.0 / 18.3 / 18.3 | 23.4 / 16.6 / 16.6 | 48.0 / 33.5 / 33.5 | 33.6 / **23.6** / 23.6 |
| per-turn oracle | 18.8 / 20.7 / 20.7 | 16.3 / 19.1 / 19.1 | 37.1 / 39.1 / 39.1 | 25.0 / 27.2 / 27.1 |

tcpWER and cpWER agree to within 0.1 points everywhere, so the 5 s collar is not
doing the work and the two can be read as one number at this granularity.

The speaker-reference view, for the record:

| Stream | tcpWER | cpWER |
|---|---|---|
| oracle, per turn, with its speaker | **24.4 %** | **23.3 %** |
| pipeline (one stream) | 119.9 % | 119.1 % |
| single-stream oracle (one stream) | 124.2 % | 123.8 % |

A single-stream hypothesis pays for every reference speaker it does not cover:
the metric charges four speakers against one. That is not a property of the
pipeline, it is what tcpWER is for (diarized hypotheses). The engine does not
attach speakers to words today, so this view cannot be the pipeline's check.

### Which metric should be the standing check

**tcpWER against the merged reference (collar 5 s) should be the standing
check.** It is the literature metric for long-form transcription, it does not
charge a word to the wrong row, and it ranks every variant exactly as the
per-utterance metric does (`before` 33.8 > int8 26.1 > fp32 runs 22.7 > fp32
20.7). It also removes a systematic distortion: the per-utterance midpoint
scorer assigns each pipeline word to one reference window, so it charges a
merge or a shift twice and cannot represent a single stream in an overlap. The
same fp32 run is 35.4 % per utterance and 20.7 % tcpWER - the 14.7-point
difference is assignment, not recognition.

Two consequences:

1. **The "gap to the oracle" target has to be dropped.** Under a stream metric
   the pipeline is *better* than the single-stream oracle (20.7 % vs 23.6 %),
   because the oracle is a decode of short reference clips and the pipeline
   decodes 60 s windows. A decode of the dataset's boundaries is not an upper
   bound on word accuracy. Per utterance the oracle still is (it has the true
   boundaries and both sides of overlaps), but that is a different question.
2. **cpWER and the speaker-reference view become the check once words carry
   speaker labels.** The oracle's 24.4 % tcpWER / 23.3 % cpWER with speakers is
   the number a diarized engine should approach. Until then, per-utterance WER
   stays as the segmentation and attribution diagnostic, because it is the only
   view that shows which rows lose their words (the backchannel table above).

## What the engine should adopt

1. **Switch the transcription artifact to fp32 Parakeet TDT 0.6b v3.** It fixes
   the documented decoder failure, buys 2.5 points per utterance and 5.4 points
   tcpWER over int8 with the same code, recovers 41 % of the dropped
   backchannel words, and costs 1.1-1.6x ASR time and about 1.9 GB more disk.
   The 4-core laptop is the constraint, not the ASR time: 2.9 GB RSS plus the
   meeting audio needs 8 GB. Ship one artifact, not both; if a low-memory mode
   is wanted, int8 stays as the fallback. This is a product decision (installer
   size and minimum RAM), so this branch does not change the manifest.
2. **Keep the hole re-decode.** With fp32 it is still worth 1.1 points per
   utterance and 2.0 tcpWER, and it is what brings EN2002a within 3 points of
   the single-stream oracle. Its cost falls with fp32 (+30 to +74 % over the
   fp32 first pass, against +31 to +137 % over int8) because the decoder leaves
   far fewer holes.
3. **Do not add speaker splits.** One extra cut per meeting, no WER change, and
   it would require diarizing before transcribing.
4. **Score with tcpWER (merged reference, collar 5 s) from now on**, and keep
   the per-utterance table as the segmentation diagnostic. Report cpWER once
   words are speaker-labelled.

## Reproduction

```bash
export ROOT=/path/to/evidence/tinta-fork/2      # meeting wavs, int8 models
export ROOT19=/path/to/evidence/tinta-fork/19   # rows.json, oracle decodes
export EVID=/path/to/evidence/tinta-fork/23     # results for this task
scripts/fork/segmentation/fetch-model-fp32.sh "$EVID/models"
scripts/fork/segmentation/run-asr-experiments.sh
```

The IHM sample was decoded with the task #19 batch harness (it builds the
recognizer directly, no VAD):

```bash
tinta-bench batch --manifest sample-1000.tsv --output ihm-sample-fp32.jsonl \
  --encoder parakeet-tdt-0.6b-v3/encoder.onnx \
  --decoder parakeet-tdt-0.6b-v3/decoder.onnx \
  --joiner  parakeet-tdt-0.6b-v3/joiner.onnx \
  --tokens  parakeet-tdt-0.6b-v3/tokens.txt --threads 2 --spin 0
```

The one-window check:

```bash
tinta-segmentation --wav "$ROOT/data/ES2004b.Mix-Headset.wav" \
  --models-root "$FP32_MODELS" --threads 2 --window 235.4:248.8
```

## Files

| Path | Purpose |
|---|---|
| `scripts/fork/segmentation/fetch-model-fp32.sh` | pinned fp32 download, SHA-256, manifest layout |
| `scripts/fork/segmentation/run-asr-experiments.sh` | the three experiments with the engine's path |
| `scripts/fork/segmentation/score-meeteval.py` | tcpWER (collar 5 s) and cpWER against merged and speaker references |
| `scripts/fork/segmentation/score.py` | per-utterance scoring, now with the short-row/backchannel table |
| `scripts/fork/segmentation/src/main.rs` | `--diarize` (task #15 turns) and `--split-turns` (experiment b) |
| `crates/tinta-engine-rs/src/speech.rs` | `transcribe_pieces`, the explicit piece-plan seam the harness uses (no behavior change) |

Evidence: `evidence/tinta-fork/23/results/` (decodes, scores, meeteval JSON) and
`evidence/tinta-fork/23/logs/`. `turns-<meeting>.json` are the diarizer turns,
`fp32-*-<meeting>.json` and `split-<meeting>.json` the decodes,
`int8-default-<meeting>.json` the same-round int8 baseline.

## Limitations

- **The IHM sample result is not significant.** 1,000 utterances give a -1.15
  point delta with a speaker-cluster CI that crosses zero. Only the full split
  can settle the 13.8 vs 11.31 gap.
- **The fp32 cost is a ratio, not a benchmark.** The host is shared; the same
  79-piece int8 plan measured 318 s and 408 s in two runs of this round
  (+29 %). Memory and model size are exact.
- **The merged reference contains both sides of overlaps**, so tcpWER's
  absolute value is not reachable by any single stream. It is comparable across
  streams, not a product WER.
- **The stream metrics use the same short-clip oracle decodes**, which are not
  an upper bound. The per-turn oracle with speakers (24.4 % tcpWER / 23.3 %
  cpWER) is the only oracle number that behaves like a bound, and only for a
  diarized hypothesis.
- **The speaker split used the engine's diarizer on Mix-Headset**, the same
  audio the pipeline hears, not the close-talk reference. It is the task #15
  post-clustered output (4 speakers on each meeting), which is the intended
  product path.
- **ES2004b is still the tuning meeting** for the segmentation settings; fp32,
  the splits and the metrics were measured on all three, but the settings
  themselves were not re-tuned for the new model.
- **No audio or models are committed.** The fp32 files live in the evidence
  directory and are pinned by hash.
