# ASR baseline on AMI headset audio: is 27% WER the channel or the runtime?

Task #19 of the Windows/Linux port, ASR half. `docs/fork/benchmark.md` reports
**27.0-27.1%** Whisper-normalized WER with sherpa-onnx Parakeet TDT 0.6b v3 int8
on the AMI **Mix-Headset** test meetings, while the Parakeet v3 model card
publishes **11.31%** on AMI **IHM** test. Both numbers were in the benchmark
document; nothing connected them. This document measures the missing middle: the
same model and decoding on the reference-segmented single-headset utterances the
Open ASR Leaderboard uses, and the same words again cut from Mix-Headset.

Short answer: **the runtime is not the problem, and the channel alone does not
explain 27% either.** On headset audio the int8 sherpa-onnx build scores **10.9%**
over a 16-meeting sample and **9.7%** on the utterances we can also score on Mix,
against the published **11.31%**; that is inside sampling noise. On the *same*
1,291 utterances cut from Mix-Headset the same decoder scores **16.4%**, so the
room and the other voices cost **+6.7 points** when the boundaries are ideal.
Nearly all of that penalty sits in windows where another speaker is talking.
The remaining ~10 points to the 27% full-meeting figure are the long-form
condition: continuous mixed audio, no reference boundaries, no overlap handling.
That is a pipeline and product problem, not a recognizer-quality problem.

## Verdict

**Runtime OK.** On the leaderboard's IHM condition this int8 ONNX artifact is
statistically indistinguishable from the published full-precision NeMo number
(10.9% measured, 95% CI 10.2-12.2, versus 11.31% published). A broken conversion
or a bad decoder would not land there on 2,131 utterances.

The 27% is a different task, and it decomposes:

1. **Headset difficulty (the task):** 9.7-10.9% for this sample. Same as
   published.
2. **Crosstalk inside the window:** +6.7 points, 9.7% -> 16.4% on identical
   words. The far-field room alone is worth +0.4 (solo windows); windows where
   another reference speaker talks for more than 20% of the span go from 16.7%
   to 39.6% (+22.9).
3. **Long-form segmentation of mixed audio:** the rest. ES2004b's full-meeting
   run in `docs/fork/benchmark.md` scores 27.1% with 1,575 deletions, while the
   same meeting's oracle-segmented Mix rows score 17.6%. The same model, the
   same audio, the same reference; the 9.5-point difference separates the two
   decode paths (oracle boundaries vs. VAD pieces of continuous multi-speaker
   audio), not two acoustic conditions.

| Comparison | Utterances | WER (Whisper) | Source |
|---|---|---|---|
| Parakeet v3, AMI IHM test, published | 12,643 (full split) | **11.31%** | model card `model-index`, pinned (see below) |
| This runtime, AMI IHM test sample | 2,131 | **10.9%** | measured here |
| This runtime, IHM, the two paired meetings | 1,291 | 9.7% | measured here |
| This runtime, Mix-Headset, the same 1,291 utterances | 1,291 | 16.4% | measured here |
| This runtime, Mix-Headset, full meetings (baseline) | 2 full meetings | 27.0-27.1% | `docs/fork/benchmark.md` |

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
  only the numeric precision differs.
- The same card is the source for the LibriSpeech 1.93/3.59 and Earnings-22
  11.42 figures the benchmark document quotes.

Two differences from this measurement, both nominally in the published number's
favour:

1. **Precision.** The card evaluates the NeMo (PyTorch) model, bf16/fp32. We run
   `sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`: the encoder, decoder and joiner
   are int8 quantized for CPU. Some int8 loss is possible.
2. **Implementation.** NeMo's own transducer decoder vs. sherpa-onnx greedy
   search. Both are greedy TDT, no external LM, no beam.

The measured 10.9% is 0.4 points *below* the published 11.31%, but the 95% CI
(10.2-12.2) covers it, so the honest statement is "the same, within sampling
noise". 11.31% is the reference value, not a bar to beat.

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

**Selection.** Decoding all 12,643 utterances is 8.68 h of audio for a runtime
check; the two-meeting Mix pairing also needs the matching Mix audio. The sample
is deterministic:

- **All 1,291 rows of ES2004b (688) and IS1009b (603).** These are the two test
  meetings whose Mix-Headset audio is already in the task-#2 evidence directory,
  so every row can be scored twice on the same words. That is the core
  comparison and it alone exceeds the 1,000-utterance floor. It is also 96% of
  ES2004b's audio, so it is comparable to that meeting's full-meeting baseline.
- **60 rows from each of the other 14 meetings**, taken at the midpoints of 60
  equal bins over the rows sorted by `audio_id` (840 rows). This is what makes
  the sample span all 16 test meetings, so the headline IHM WER is comparable to
  the published full-split 11.31% and not an ES/IS-only estimate.

Total: **2,131 IHM utterances, 1.75 h** (17% of the rows, 20% of the audio), plus
**1,291 Mix utterances, 1.18 h**. The 95% CI on the headline number is
bootstrapped by resampling utterances (1,000 resamples, seed 19), because the
sampling error matters at the one-point level.

The Mix rows are the **same `[begin_time, end_time)` ranges** cut out of
`<meeting>.Mix-Headset.wav` (16 kHz mono) and decoded with the same recognizer
and settings. Reference text is identical by construction. The only differences
are the microphone (single close headset vs. the room mix), the other speakers
present in the mix, and the far-field acoustics. The utterance boundaries are the
dataset's own, and are the same on both sides, so any boundary clipping is
symmetric.

Data and models live under `/home/sergio/code/evidence/tinta-fork/19/`
(`data/ihm-test/`, `data/ihm-baseline/`, `results/`); the models are the task-#2
ones under `evidence/tinta-fork/2/models/`. No audio is committed. Per-file
SHA-256 of the parquet actually read is recorded in
`data/ihm-baseline/provenance.json`.

## Method

- **Model**: the same files as the benchmark, pinned by SHA-256 in
  `fetch-data.sh`: Parakeet TDT 0.6b v3 int8 (`encoder.int8.onnx` 652 MB,
  `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt`).
- **Decoding**: `OfflineRecognizer`, CPU provider, greedy TDT, `--spin 0`
  (ONNX Runtime intra-op spinning off), 2 threads. No VAD, no chunking, no
  padding: each row is decoded exactly as the dataset defines it, which is what
  the leaderboard does. The 60 s hard cap in `run_asr_call` still applies; the
  longest row in this sample is 26.2 s.
- **Harness**: a new `batch` subcommand in `scripts/fork/bench/src/main.rs` that
  builds the recognizer once and decodes a TSV manifest of
  `id <TAB> wav <TAB> start <TAB> end`. The `asr` command would load the model
  three times and evict it from the page cache per 2 s clip; the quality is the
  same, the throughput is not.
- **Normalizers**: Whisper's `EnglishTextNormalizer` (the leaderboard's
  normalizer, via the `whisper-normalizer` package) is the headline; the
  benchmark's own legacy normalizer (lowercase, strip punctuation) is reported
  next to it. Both are applied to reference and hypothesis. AMI's `T_V_`
  acronym rewrite in `score.py` is a no-op here: this dataset's text has no
  underscores.
- **WER**: corpus level, `(sub + del + ins) / reference words`, summed over
  utterances, with `del`/`ins`/`sub` reported separately. Per-utterance WER
  would be dominated by very short clips (167 of the 2,131 rows are shorter
  than 0.3 s).
- **Overlap buckets**: for the paired rows only, a 10 ms grid over the meeting's
  reference `words.xml` counts how many distinct AMI speakers have annotated
  words inside the window. That is oracle information (it uses the reference
  transcript), so it is a diagnostic, not something the runtime can compute.
- **No timing claims.** The host is shared; the run was `nice -n 19` at 2
  threads under a 1-minute load average of ~28-40. Throughput is reported as a
  ratio only.

## Results

### IHM vs Mix on the same utterances

`ihm_pair` and `mix_pair` are the same 1,291 rows. `ihm_other` is the 840-row
16-meeting sample that has no Mix counterpart. `ihm_all` is the headline IHM
number: all 2,131 rows.

| Set | Utterances | Audio min | WER (Whisper) [95% CI] | del / ins / sub | Ref words |
|---|---|---|---|---|---|
| IHM, paired meetings (ES2004b + IS1009b) | 1,291 | 70.7 | 9.7% [8.8-11.0] | 588 / 187 / 469 | 12,814 |
| Mix-Headset, same utterances | 1,291 | 70.7 | 16.4% [16.1-19.3] | 939 / 285 / 879 | 12,814 |
| IHM, 16-meeting sample | 840 | 34.1 | 13.7% [12.0-15.9] | 410 / 99 / 282 | 5,792 |
| IHM, all sampled utterances | 2,131 | 104.8 | 10.9% [10.2-12.2] | 998 / 286 / 751 | 18,606 |

The difference inside the paired set is the first half of the answer: **+6.7
points of WER from the channel and the other voices**, on identical words, same
model, same decoder, same segment boundaries. The paired IHM and Mix columns are
not two different samples, so no sampling caveat applies to that delta.

### The benchmark's own normalizer agrees

The legacy normalizer keeps more tokens (it does not map spellings, numbers or
fillers the way Whisper's does), so WERs are slightly higher, but the picture is
identical: the paired delta is +6.8 points, mixing costs about as much either
way, and IHM stays near the published figure.

| Set | Utterances | Ref words (legacy) | WER legacy | WER Whisper |
|---|---|---|---|---|
| IHM, paired meetings | 1,291 | 12,945 | 10.7% | 9.7% |
| Mix-Headset, same utterances | 1,291 | 12,945 | 17.5% | 16.4% |
| IHM, 16-meeting sample | 840 | 5,841 | 14.7% | 13.7% |
| IHM, all sampled utterances | 2,131 | 18,786 | 12.0% | 10.9% |

### Per-meeting IHM

All 16 test meetings are represented; ES2004b and IS1009b are complete, the
others are 60-utterance samples.

| Meeting | Utterances | WER (Whisper) |
|---|---|---|
| EN2002a | 60 | 18.9% |
| EN2002b | 60 | 22.6% |
| EN2002c | 60 | 17.4% |
| EN2002d | 60 | 21.9% |
| ES2004a | 60 | 10.6% |
| ES2004b | 688 | 9.5% |
| ES2004c | 60 | 12.4% |
| ES2004d | 60 | 11.2% |
| IS1009a | 60 | 15.9% |
| IS1009b | 603 | 9.9% |
| IS1009c | 60 | 10.1% |
| IS1009d | 60 | 10.4% |
| TS3003a | 60 | 11.0% |
| TS3003b | 60 | 8.7% |
| TS3003c | 60 | 11.6% |
| TS3003d | 60 | 16.3% |

The complete meetings bracket the sampled ones, and the spread across meetings
(8.7-22.6%) is larger than the sampling CI, so the headline number should be
read as "about this", not to a tenth of a point. The EN2002x meetings are the
hard ones for this model; that is a property of the data, since it shows on the
headset channel too.

### Paired IHM vs Mix per meeting

| Meeting | Utterances | IHM WER | Mix WER | Full-meeting Mix WER |
|---|---|---|---|---|
| ES2004b | 688 | 9.5% | 17.6% | 27.1% |
| IS1009b | 603 | 9.9% | 15.1% | 22.5% (6 min sample) |

The last column is `docs/fork/benchmark.md`'s full-meeting run. Same meeting,
same model, same reference text; the difference from the oracle-segmented Mix
column is the long-form decode path. For ES2004b that is **9.5 points**, and it
is larger than the headset-to-mix delta itself.

### Far-field vs crosstalk

The paired Mix rows are oracle-segmented: each is cut at the dataset's own word
boundaries, with no VAD and no chunking. That makes them the *best case* for the
mixed channel, so any remaining gap is acoustic. To split that gap further, the
paired windows are bucketed by whether another AMI reference speaker has
annotated words inside the window. AMI word intervals tile their pauses, so this
counts a backchannel into a pause as well; it is "another voice is in the
window", not strictly simultaneous speech.

| Bucket | Utterances | Audio min | IHM WER | Mix WER | Delta |
|---|---|---|---|---|---|
| solo (no other speaker) | 370 | 27.5 | 7.6% | 7.9% | +0.4 |
| other speaker <= 20% | 206 | 23.9 | 6.6% | 7.3% | +0.7 |
| other speaker > 20% | 715 | 19.4 | 16.7% | 39.6% | +22.9 |

The answer is unambiguous. **Far-field acoustics on their own cost almost
nothing**: in windows where no other speaker is annotated, Mix is +0.4 points
over headset. The penalty is crosstalk: the third bucket holds 715 of 1,291
rows and carries +22.9 points, and its Mix WER (39.6%) alone is worse than the
full-meeting average. The rows are also short (19.4 min for 715 rows), i.e.
mostly backchannels and turn-taking overlap, which is exactly where a
single-stream decoder loses words.

### Error shape

| Set | del | ins | sub | Ref words | Deletions as share of errors |
|---|---|---|---|---|---|
| IHM paired | 588 | 187 | 469 | 12,814 | 47% |
| Mix paired | 939 | 285 | 879 | 12,814 | 45% |
| IHM all | 998 | 286 | 751 | 18,606 | 49% |
| ES2004b full meeting (benchmark) | 1,575 | 57 | 207 | — | 86% |

Oracle-segmented Mix adds **+410 substitutions and +351 deletions** (+98
insertions): a balanced degradation, not one error mode. The full-meeting run is
the outlier, with 86% of its errors deletions. That contrast is the second half
of the answer: when the decoder is handed a clean single-speaker window it
misrecognizes words, as any ASR does; when it is handed a long mixed stretch it
stops emitting whole regions, and those show up as deletions. The latter is a
segmentation/diarization problem, not an acoustic-model problem.

## Why this settles the question

Three checks point the same way:

1. **IHM matches the published number.** 10.9% measured (95% CI 10.2-12.2)
   against 11.31% published, on 2,131 utterances across all 16 test meetings,
   with an int8 CPU artifact and the same greedy decoding. A broken runtime does
   not land there.
2. **The paired delta is crosstalk, not the room.** On identical words the mix
   costs +6.7 points: +0.4 in solo windows, +22.9 in the >20%-other-speaker
   windows, which add 787 errors of their own. Nothing else changed between the
   columns.
3. **The 27% is not reproduced at the segment level.** The same meeting that
   scores 27.1% in the full-meeting pipeline scores 17.6% on its own
   oracle-segmented Mix rows. So 27% is not "the mixed channel"; it is the
   mixed channel *plus* the long-form, no-overlap-handling decode path.

What this does **not** say is that 27% is good, or that the product is done. It
says the gap is not in the port or the model conversion. The engineering levers
are:

- **Overlap handling.** Overlapping speech needs diarization-aware segmentation
  or a multi-speaker model before a single-stream recognizer sees it. The
  benchmark's no-overlap column (22.1-23.5%) already shows how much of the
  full-meeting number is overlap; this document shows the same effect on
  identical words.
- **Long-form segmentation.** VAD pieces that span several speakers are the
  mechanism behind the deletion-heavy full-meeting errors.
- **Fillers/backchannels.** A decoder that emits `Mm-hmm`, `Yeah`, `Okay`
  scores better on AMI's manual transcripts; Parakeet's punctuation/casing-free
  output does not. This is a small effect here (the Whisper normalizer already
  drops most of them) but it is real.

## What 27% means for the acceptance bar

`docs/fork/benchmark.md` design condition 7 asks for an "IHM-utterance sanity
check near the remembered ~11%". That bar is now measured, and the two numbers
must not be compared directly:

- **ASR engine bar (IHM, reference-segmented):** Whisper-normalized WER at or
  below **12%** on this sample (measured 10.9%, plus 1 point for run-to-run and
  sampling noise). The 11.31% model-card figure is context, not the bar. This is
  the number that answers "is the recognizer good enough".
- **Product bar (full meetings, Mix-Headset):** unchanged,
  `no-overlap WER <= baseline + 2`, and it must be measured on the *same*
  pipeline (VAD + segmentation + ASR). 27% is the baseline for that pipeline,
  not a model-quality regression, and it cannot be compared to an IHM number.
  Any improvement here comes from overlap-aware segmentation/diarization, and
  that is where effort should go.
- **Do not** quote "11.31% published vs 27% measured" as an engine gap. The
  engine's gap on its own task is 0 points; the product's gap on the meeting
  task is structural.

## Limitations

- **Oracle boundaries.** The paired Mix number (16.4%) cuts at the dataset's
  reference boundaries. It is a best case for the channel and an *upper bound*
  on what the product's long-form path achieves; it is not the product's
  expected WER. The full-meeting number stays the product number.
- **Two of 16 meetings are complete.** ES2004b and IS1009b are the only test
  meetings with Mix-Headset audio in the existing evidence directory; the other
  14 contribute 60 rows each for the IHM headline. The paired delta is unaffected
  by this; the absolute headline has the stated bootstrap CI.
- **Sample, not the full split.** 2,131 of 12,643 utterances. Decoding all of it
  would cost about four times the audio for a tighter CI; the answer would not
  change direction.
- **int8 vs bf16.** The published 11.31% is not this artifact. The comparison is
  deliberately "published model vs. our CPU artifact", because that is the
  decision at hand.
- **HF segment boundaries.** The dataset's `[begin_time, end_time)` windows are
  what the leaderboard uses; they occasionally include a word from the
  neighbouring segment. That noise is present in the published number too, and
  identical on the IHM and Mix sides.
- **Overlap buckets use the reference transcript** to decide who is talking.
  They diagnose where the loss is; they cannot be computed at runtime.
- **No LM, no punctuation, no casing** on either side. WER only.
- **Shared host.** Throughput was not measured for a timing claim; see
  `docs/fork/benchmark.md` for timing.

## Reproduction

```bash
export ROOT=/home/sergio/code/evidence/tinta-fork/2
export ROOT19=/home/sergio/code/evidence/tinta-fork/19
scripts/fork/bench/fetch-data.sh      # adds the pinned IHM parquet (~1.3 GB)
scripts/fork/bench/run-ihm-baseline.sh
```

`run-ihm-baseline.sh` unpacks the selected utterances, runs the two `batch`
decodes (IHM then Mix, one at a time), and prints the tables above from
`score.py`'s normalizers. One decode by hand:

```bash
scripts/fork/bench/target/release/tinta-bench batch \
  --manifest $ROOT19/data/ihm-baseline/manifest-ihm.tsv \
  --output $ROOT19/results/ihm-decode.jsonl \
  --encoder $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx \
  --decoder $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx \
  --joiner  $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx \
  --tokens  $ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt \
  --threads 2 --spin 0

$ROOT/venv/bin/python scripts/fork/bench/ihm-baseline.py score
```

## Files

| Path | Purpose |
|---|---|
| `scripts/fork/bench/ihm-baseline.py` | unpack the IHM split, build IHM/Mix manifests, score, bootstrap, overlap buckets |
| `scripts/fork/bench/run-ihm-baseline.sh` | end-to-end: prepare, two batch decodes, tables |
| `scripts/fork/bench/src/main.rs` | new `batch` subcommand (one recognizer, many short clips) |
| `scripts/fork/bench/fetch-data.sh` | IHM parquet download, revision + SHA-256 pinned |

Results and logs: `evidence/tinta-fork/19/results/` (`ihm-decode.jsonl`,
`mix-decode.jsonl`, `ihm-summary.json`, `ihm-tables.md`) and
`evidence/tinta-fork/19/logs/`.
