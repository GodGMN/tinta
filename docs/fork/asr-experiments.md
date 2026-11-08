# ASR experiments: the requantized encoder, speaker splits, and stream-level scoring

Task #23 of the Windows/Linux port, stacked on the task #21 segmentation work
(PR #16). The pipeline scored 37.9 % per-utterance WER and 25.7 % tcpWER on
ES2004b / IS1009b / EN2002a with the shipped int8 Parakeet. The task #23 review
found that the dropped stretches come from the int8 **encoder**, not the
decoder or joiner: the sherpa-onnx int8 export quantizes every supported op
(Conv included) per tensor as unsigned int8, the crudest setting for a
FastConformer.

**Short answer.** Re-quantizing the fp32 encoder with MatMul-only, per-channel
int8 weights and Conv left in fp32 fixes the failure at 0.9 GB:

- the 13.4 s failing window (ES2004b 235.4-248.8 s) returns **no words** with
  int8 and the full sentence with the requantized encoder and with fp32;
- the requantized encoder scores **20.2 % tcpWER at a 5 s collar and 26.2 % at
  1 s**, against 20.3 / 26.4 for fp32 and 25.7 / 31.4 for int8; the
  per-utterance WER is 35.3 % against 35.4 % for fp32 and 37.9 % for int8;
- it keeps **102 % of the fp32 tcpWER gain** (5.5 of 5.4 points) at **0.90 GB
  of model instead of 2.55 GB** and **3.0 GiB of engine peak RSS instead of
  4.2 GiB**;
- on the 1,000-utterance AMI IHM sample it is at least as good as fp32:
  **13.16 % WER** against 13.79 % for fp32 and 14.94 % for int8; the paired
  delta against int8 is -1.78 points (95 % CI -3.53 to +0.15, bootstrapped over
  the 16 true speakers), so the direction is right but the sample alone is not
  conclusive, and the fp32 gap is within noise.

**Recommendation: ship the requantized encoder as the default model, keep fp32
as an opt-in "high accuracy" download for machines with at least 16 GB.** The
review's decision rule is met: the window recovers and tcpWER is 20.2 %, below
the 21.8 % threshold (80 % of the fp32 gain).

Speaker splits are still a dead end: the VAD already closes every backchannel
into its own piece, so a 0.2 s pass finds exactly one extra cut per meeting and
moves no metric. Keep the hole re-decode: it is worth 2.1 points tcpWER with
fp32 (22.4 % without it) and it repairs the far fewer holes the new encoder
leaves.

The stream metric changes the picture. tcpWER against the merged reference is
**25.7 % for int8 and 20.2 % for the requantized encoder**; the *single-stream
oracle* is an int8 decode at 23.2 %. Compare like with like: with int8 on both
sides, segmentation costs **+2.5 tcpWER points** (25.7 % pipeline against
23.2 % oracle), while the per-utterance midpoint scorer reports +4.3 points
against the same oracle, so it overstates the damage by **1.8 points**. The
oracle is not an upper bound under a stream metric: it is a decode of short
reference clips, and the pipeline's long-window decode is better per word. The
requantized pipeline's 20.2 % is below that int8 oracle because it is a better
model, not because segmentation helps. tcpWER should be the standing check for
transcript quality; the per-utterance number stays as the
segmentation/attribution diagnostic.

The requantized encoder is **equivalent to fp32 within noise** (at least as
good: 0.1 point better tcpWER at 5 s and 0.2 at 1 s, with a per-utterance delta
of 0.1 point in its favour). It is not a quality improvement over fp32; the
gain over int8 comes from recovering deletions (485 -> 283) at the cost of more
insertions (160 -> 260, against fp32's 247).

## The review's finding and the new encoder

The review ran the failing window with mixed model files:

| Encoder | Decoder + joiner | Words |
|---|---|---|
| int8 | int8 | 0 |
| **int8** | **fp32** | **0** (empty text) |
| **fp32** | **int8** | **32**, identical to full fp32 |

So the encoder is the cause, and the decoder and joiner are not. The shipped
int8 encoder (`csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8`) is
652,184,281 B and its export script calls
`quantize_dynamic(..., weight_type=QuantType.QUInt8)` with the default op list:
**Conv is quantized and the weights are per tensor**. That is the setting the
review flagged.

### The recipe

`scripts/fork/segmentation/requantize-encoder.py` re-quantizes the fp32 encoder
that this branch already downloads (no new model):

```python
quantize_dynamic(
    "encoder.onnx", "encoder.int8.onnx",
    op_types_to_quantize=["MatMul", "Gemm"],  # Conv stays fp32
    per_channel=True,
    reduce_range=True,
    weight_type=QuantType.QInt8,
    use_external_data_format=False,           # 0.9 GB fits one file
)
```

- 217 of the 289 MatMul nodes are quantized; the other 72 have a non-constant
  B. All 77 Conv nodes stay fp32, as do LayerNormalization and the embeddings.
- The 11 `metadata_props` of the fp32 encoder (`model_type`
  `EncDecRNNTBPEModel`, `vocab_size` 8192, `subsampling_factor` 8, ...) are
  copied to the new file, which is what sherpa-onnx reads to pick the NeMo
  transducer path.
- The decoder and joiner stay the shipped int8 files.

**`reduce_range=True` is not optional on this host.** The measurement machine
is an AMD Ryzen 7 5825U (Zen 3, AVX2, **no AVX512-VNNI**). Per-channel int8
weights *without* `reduce_range` produce a broken encoder: the window decodes
to `St.` and the encoder output correlates 0.09 with fp32 on a synthetic
input. ONNX Runtime documents `reduce_range` for per-channel quantization on
non-VNNI machines; with it the same window returns the full sentence. The
ablation (synthetic 200-frame input, correlation with fp32; the window is the
real gate):

| Variant | Correlation | Window |
|---|---|---|
| per-tensor QInt8, MatMul only, Conv fp32 | 0.68 | 32 words |
| per-channel QInt8, no `reduce_range` | 0.09 | 1 word (`St.`) |
| **per-channel QInt8 + `reduce_range`** | **0.90** | **32 words** |
| per-channel QUInt8 | 0.91 | 32 words |

The correlation is a diagnostic on random input, not a quality metric: the
real audio is the window and the three evaluations below.

### Pin and size

| | |
|---|---|
| Input | fp32 `encoder.onnx` 41,838,346 B, sha256 `36b75836...48512`; `encoder.weights` 2,435,420,160 B, `3af3f51a...f39a8` |
| Output | `encoder.int8.onnx` **881,596,689 B (841 MiB)**, sha256 `23c10054...2c336f` |
| Full model | encoder + int8 decoder 11,845,275 B + int8 joiner 6,355,277 B = **899,797,241 B (858 MiB)** |
| int8, for comparison | 652,184,281 + 11,845,275 + 6,355,277 = 670,384,833 B (639 MiB) |
| fp32, for comparison | 2,549,778,581 B (2,432 MiB) |

The script verifies both input hashes and the output hash, and fails if the
recipe changes without updating the pin. It writes `<output>.tmp`, checks the
hash and only then renames it over `--output`, so a failed run never leaves an
unpinned `encoder.int8.onnx` in a live models root. The quantizer versions are
pinned in `scripts/fork/segmentation/requirements-quant.txt`
(`onnxruntime==1.31.1`, `onnx==1.23.2`, `numpy==2.4.6`) and printed at startup,
because the output hash depends on them.

A CI matrix (`asr-requant-portability` in `.github/workflows/ci.yml`) rebuilds
the encoder with those pins on x64 AVX2, x64 with AVX-512/VNNI, Linux arm64 and
Windows arm64, verifies the output hash (falling back to a per-tensor digest if
a platform rounds differently) and decodes the committed 13.4 s clip, asserting
at least 28 words and at most 10 % normalized WER against the fp32 reference.
It runs when the manifest, the script, the quantizer pins, `Cargo.lock` or the
fixture change, and weekly.

## The failing window

`--window 235.4:248.8` on ES2004b, 13.40 s, 2 threads:

| Model | Words | Output |
|---|---|---|
| int8 | **0** | empty text |
| **requantized encoder** | **32** | `the channel um or the volume, everything is just um some sort of signal to change the state of the T V or other appliance that it's sending the signal to` |
| fp32 | **32** | same text |

The review counts 32 words; the engine's word list has 31 whitespace-separated
words because `it's` is one word. The requantized encoder returns exactly the
fp32 text, including the `T V` tokens, so the whole stretch from 0.24 s to the
end of the window is recovered. This is the first gate and it passes.

## The IHM sample (stage 2)

A fixed random sample of 1,000 rows of the AMI IHM test split (seed 23, 6,867
normalized reference words), decoded with the task #19 batch harness (greedy
TDT, no VAD, 2 threads). int8 comes from the cached full-split decode; fp32 and
the requantized encoder were decoded on the same files. The sample generator
and the scorer are now committed (`ihm-sample.py`), and the sample reproduces
byte for byte.

| Model | WER | del / ins / sub | Paired errors vs int8 |
|---|---|---|---|
| int8 | 14.94 % | 485 / 160 / 381 | - |
| fp32 | 13.79 % | 341 / 247 / 359 | -79 |
| **requantized encoder** | **13.16 %** | **283 / 260 / 361** | **-122** |

Paired delta against int8, percentile bootstrap, 2,000 resamples, seed 23:

| Model | Delta | 95 % CI by true speaker (16 clusters) | 95 % CI by meeting |
|---|---|---|---|
| fp32 | -1.15 points | -2.73 to +0.39 | -2.60 to +0.01 |
| **requantized encoder** | **-1.78 points** | **-3.53 to +0.15** | **-3.03 to -0.79** |

The first draft said "63 clusters". AMI IHM test has **16 speakers**; 63 is the
number of meeting x speaker pairs, and resampling those understates the
interval. The corrected speaker-cluster interval still crosses zero for the
requantized encoder, so the sample says "right direction, not conclusive"; the
meeting-cluster interval is tighter but the meeting is not the sampling unit
for speakers. The conclusion does not depend on the cluster choice: the
requantized encoder is at least as good as fp32 on this sample, and both are
better than int8.

## The three meetings (stage 3)

Same engine code, same piece plans (79 / 61 / 84 pieces), same hole re-decode;
only the model files differ. int8, requantized and fp32 ran **back to back per
meeting** in the same round, with `--finalize` (transcribe then diarize in one
process). `fp32-runs` is the same first pass with `gap_redecode_seconds = 0`;
`before` is the engine before task #21.

### Per-utterance WER

| Variant | ES2004b | IS1009b | EN2002a | All three | vs single-stream oracle | Pieces (ES) |
|---|---|---|---|---|---|---|
| int8 `before` (task #21) | 38.0 % | 31.5 % | 57.2 % | 43.4 % | +9.8 | 79 |
| int8 `default` | 29.4 % | 27.4 % | 53.3 % | 37.9 % | +4.3 | 79 |
| **requantized `default`** | **26.6 %** | **25.6 %** | **50.3 %** | **35.3 %** | **+1.7** | 79 |
| fp32 `runs` (no re-decode) | 27.2 % | 26.2 % | 52.4 % | 36.5 % | +2.9 | 79 |
| fp32 `default` | 26.6 % | 25.3 % | 50.7 % | 35.4 % | +1.8 | 79 |
| per-turn oracle | 18.8 % | 16.3 % | 37.1 % | 25.0 % | | |
| single-stream oracle | 26.0 % | 23.4 % | 48.0 % | 33.6 % | | |

The requantized encoder is 2.6 points per utterance better than int8 and 0.1
point better than fp32 over the three meetings. It removes 930 deletions and
pays 290 more insertions than int8; like fp32, it hears words int8 drops,
including fillers and repeats.

### tcpWER and cpWER, collars 5 s and 1 s

Merged reference (one speaker), hypothesis and reference normalized alike:

| Variant | tcpWER 5 s | tcpWER 1 s | cpWER | Per-utterance |
|---|---|---|---|---|
| int8 `before` | 33.5 % | 38.6 % | 33.4 % | 43.4 % |
| int8 `default` | 25.7 % | 31.4 % | 25.7 % | 37.9 % |
| **requantized `default`** | **20.2 %** | **26.2 %** | **20.1 %** | **35.3 %** |
| fp32 `runs` | 22.4 % | 28.1 % | 22.3 % | 36.5 % |
| fp32 `default` | 20.3 % | 26.4 % | 20.3 % | 35.4 % |
| single-stream oracle | 23.2 % | 29.1 % | 23.1 % | 33.6 % |
| per-turn oracle | 26.7 % | 33.3 % | 26.7 % | 25.0 % |

Per meeting (merged reference):

| Variant | ES2004b 5 / 1 | IS1009b 5 / 1 | EN2002a 5 / 1 |
|---|---|---|---|
| int8 `default` | 19.4 / 26.3 | 17.4 / 24.2 | 37.6 / 41.2 |
| **requantized** | **14.1 / 21.1** | **14.0 / 21.5** | **30.2 / 34.3** |
| fp32 `default` | 14.1 / 21.1 | 14.1 / 21.5 | 30.6 / 34.7 |

The ranking is the same at both collars and in both metrics. The 1 s collar
costs about 6 points for every variant and tests the timestamps: the ranking
does not change, so the timestamps are not the weak point.

### Normalization is now symmetric

The first version normalized the reference per utterance row and the hypothesis
per word. Whisper's cross-word rules (number merging, the `T_V_` acronym join)
therefore acted on one side only. The new scorer normalizes the reference per
row and the hypothesis per contiguous run (words no more than 1 s apart), then
re-times the normalized tokens: one token per word when the count is unchanged,
and the span of the words a merged token replaced when it is not. The effect is
small but systematic: int8 moves from 26.1 to **25.7** tcpWER at 5 s, from
31.7 to **31.4** at 1 s, and from 26.0 to **25.7** cpWER; fp32 moves from 20.7
to **20.3**, and the ranking and the int8 -> requant delta do not change.

No UEM is used: hypothesis words outside the annotated regions count as
insertions for every stream. The merged reference still contains both sides of
every overlap (17.9 % self-overlap on the merged reference), so its absolute
value is not reachable by any single stream; it is comparable across streams,
not a product WER.

### What the level shift does and does not mean

For fp32, the same run is **35.4 % per utterance and 20.3 % tcpWER**. The
15-point difference is assignment, not recognition: the midpoint scorer assigns
every pipeline word to one reference window, so it charges a merge or a shift
twice and cannot represent a single stream in an overlap.

The segmentation damage is pipeline minus single-stream oracle, and both sides
must use the same model. The single-stream oracle (`evidence/19`) is an **int8**
decode, so the matched comparison is the int8 pipeline: **+4.3 points per
utterance against +2.5 points tcpWER**, which means the per-utterance scorer
overstates the damage by about **1.8 points**. The first draft compared the
fp32/requantized pipeline (+1.8 points per utterance) with that int8 oracle
(-2.9 points tcpWER) and reported 4.7 points; the round-2 re-review caught that
it mixed a model gain with segmentation damage. The 1.8-point overstatement is
the number that transfers: the fp32 and requantized pipelines show the same
per-utterance-versus-stream difference (int8: +4.3 against +2.5).

Two consequences:

1. **Keep the gap-to-the-oracle target.** With a matched model, segmentation
   still costs 2.5 tcpWER points, and closing that gap is the target. The
   requantized pipeline's 20.2 % is below the int8 oracle (23.2 %) because it
   uses a better model, not because the segmentation became free; a
   requantized-model oracle would move the reference point. Decoding the 1,291
   oracle clips with the requantized encoder would settle it, but that is a
   batch of short clips and not urgent.
2. **cpWER and the speaker-reference view become the check once words carry
   speaker labels.** The oracle's 24.0 % tcpWER / 22.9 % cpWER with speakers is
   the number a diarized engine should approach. A single-stream hypothesis
   scores 119.6 % / 118.7 % there because it is charged for every speaker it
   does not cover; that is what tcpWER is for (diarized hypotheses), not a
   property of the pipeline.

## Cost and size

Same-round runs, `nice -n 19`, 2 threads, `--finalize`; peak RSS is the whole
engine process (VAD + recognizer + one mixed track + diarizer), not the batch
harness.

| | int8 | requantized | fp32 |
|---|---|---|---|
| Model files | 639 MiB | **858 MiB** | 2,432 MiB |
| Engine peak RSS, ES2004b | 2.26 GiB | **2.86 GiB** | 4.16 GiB |
| Engine peak RSS, worst meeting | 2.26 GiB | **3.00 GiB** | 4.21 GiB |
| Batch peak RSS, 1,000 IHM rows | 1.2 GiB | **1.8 GiB** | 2.9 GiB |
| Batch CPU, 200 rows (same round) | 85 s | **73 s** | 149 s |
| Encoder CPU ratio | 1.0x | **0.86x** | 1.75x |
| Engine ASR pass, three meetings | 1,780 s | **851 s** | 1,476 s |
| Holes re-decoded (ES / IS / EN) | 44 / 18 / 82 | **11 / 3 / 33** | 9 / 6 / 36 |
| Re-decode audio with retries | 5,744 s | **1,613 s** | 1,766 s |

The engine ASR pass includes the hole re-decode, which is why int8 is the
slowest in the pipeline even though its encoder is not the slowest: it leaves
44 / 18 / 82 holes against 11 / 3 / 33 for the requantized encoder, and the
re-decode decodes 3.6x as much audio. The encoder-only ratio (batch, same
round) is the cleaner one: **requantized 0.86x int8, fp32 1.75x int8**. The
host is shared and was heavily loaded during these runs (1-minute load average
4-20 on 7 CPUs), so the wall times are ratios, not benchmarks; the CPU seconds,
model sizes and RSS are exact.

The 8 GB floor is now comfortable again: the requantized engine peaks at
3.0 GiB, fp32 at 4.2 GiB, and a one-hour meeting holds about 0.5 GB of audio
during finalize. On an 8 GB machine fp32 leaves no margin for the app, the OS
and the call client; the requantized encoder does.

## Speaker splits (experiment b)

The engine's own diarizer (`SpeechPass::diarize`, the task #15 post-clustered
path) ran on all three meetings: 4 speakers, 269 / 241 / 524 turns. The task
#21 piece plan was split at every diarized speaker change that falls inside a
silence of at least 0.2 s. The rule lives in
`scripts/fork/segmentation/src/main.rs` (`--split-turns`, `--split-gap`), not in
the engine, because it lost.

| Meeting | Pieces default -> split | WER default -> split | tcpWER 5 s default -> split | Backchannel words lost |
|---|---|---|---|---|
| ES2004b | 79 -> 80 | 29.4 -> 29.5 | 19.4 -> 19.4 | 262 -> 263 |
| IS1009b | 61 -> 62 | 27.4 -> 27.4 | 17.4 -> 17.4 | 179 -> 178 |
| EN2002a | 84 -> 85 | 53.3 -> 53.3 | 37.6 -> 37.5 | 967 -> 968 |
| All three | 224 -> 227 | 37.9 -> 37.9 | 25.7 -> 25.7 | 1,408 -> 1,409 |

Exactly **one** extra cut per meeting and no metric change. The fine VAD at
0.2 s finds 63 / 38 / 66 regions against the coarse 57 / 37 / 66: Silero needs
about 0.93 s after speech stops to close a run whatever `min_silence_duration`
says, so a real 0.2 s pause is almost never inside a run. The short rows are
not swallowed by long pieces: they already have their own piece. They are
missing because the decoder drops them or because the VAD never detected them.
Splitting pieces cannot recover either. Adopting the rule would also require
diarizing before transcribing, which moves the progress events and adds a full
diarization pass before the first word appears. Measured gain: zero. Not
adopted.

## What fp32 and the new encoder do to the backchannels

Rows shorter than 3 s, scored per utterance (the cells are the whole corpus of
short rows, 1,872 rows / 5,813 reference words):

| Variant | Empty hypotheses | Reference words lost in them | Short-row del / ins / sub |
|---|---|---|---|
| int8 default | 714 | 1,408 | 2,514 / 691 / 927 |
| fp32 runs | 574 | 1,037 | 2,088 / 891 / 997 |
| fp32 default | 511 | 831 | 1,938 / 929 / 1,082 |
| **requantized default** | **506** | **800** | **1,918 / 920 / 1,079** |

The requantized encoder recovers **608** of the 1,408 backchannel words int8
drops (43 %), slightly more than fp32 (577, 41 %). It also inserts more in
short rows (920 vs 691), so the short-row error total moves less than the
deletions.

## What the engine should adopt

1. **Switch the default transcription artifact to the requantized encoder.**
   MatMul-only per-channel int8, Conv fp32, `reduce_range=True`, metadata
   copied. It fixes the documented window, scores 20.2 / 26.2 tcpWER at
   collars 5 / 1 (against 25.7 / 31.4 for int8 and 20.3 / 26.4 for fp32),
   recovers 43 % of the dropped backchannel words, and costs 858 MiB of model
   and 3.0 GiB of engine RSS. The decision rule from the review is met: the
   window recovers and tcpWER is below 21.8 %.
2. **Offer fp32 only as an opt-in "high accuracy" model for machines with at
   least 16 GB.** It is within noise of the requantized encoder (0.1 point
   worse tcpWER at 5 s and 0.2 at 1 s), and it costs 2.55 GB of model and
   4.2 GiB of engine RSS. That is a RAM decision, not a second default: do not
   offer both side by side for no reason. On 8 GB the requantized encoder is
   the only one with margin.
3. **Keep the hole re-decode.** With fp32 it is still worth 2.1 points tcpWER
   (22.4 % without it, 20.3 % with it) and 1.1 points per utterance. With the
   requantized encoder it repairs 11 / 3 / 33 holes instead of 9 / 6 / 36 for
   fp32, so its cost falls with the better encoder.
4. **Do not add speaker splits.** One extra cut per meeting, no metric change,
   and it would require diarizing before transcribing.
5. **Score with tcpWER (merged reference) at collars 5 s and 1 s**, with the
   same normalization on both sides, and keep the per-utterance table as the
   segmentation diagnostic. Report cpWER once words are speaker-labelled.

## Reproduction

```bash
export ROOT=/path/to/evidence/tinta-fork/2      # meeting wavs, int8 models
export ROOT19=/path/to/evidence/tinta-fork/19   # rows.json, oracle decodes
export EVID=/path/to/evidence/tinta-fork/23     # results for this task
scripts/fork/segmentation/fetch-model-fp32.sh "$EVID/models"
"$EVID/venv-quant/bin/python" scripts/fork/segmentation/requantize-encoder.py \
  --input-dir "$EVID/models/parakeet-tdt-0.6b-v3" \
  --output "$EVID/models/requant-root/parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx"
BENCH=/path/to/tinta-bench scripts/fork/segmentation/run-asr-experiments.sh
```

The script creates the three model roots (`int8-root`, `requant-root`,
`fp32-root`) from the task #2 files and the fetched fp32 files. `INT8_MODELS`
defaults to `$EVID/models/int8-root`; the script creates that directory and its
manifest layout, so the default path exists after the first run. The 1,000-row
sample is regenerated with:

```bash
"$EVID/venv/bin/python" scripts/fork/segmentation/ihm-sample.py make \
  --rows "$ROOT19/data/ihm-baseline-full/rows.json" \
  --out-json "$EVID/data/sample-1000.json" --out-tsv "$EVID/data/sample-1000.tsv"
```

The one-window check:

```bash
tinta-segmentation --wav "$ROOT/data/ES2004b.Mix-Headset.wav" \
  --models-root "$EVID/models/requant-root" --threads 2 --window 235.4:248.8
```

## Files

| Path | Purpose |
|---|---|
| `scripts/fork/segmentation/requantize-encoder.py` | the new encoder: MatMul-only per-channel QInt8, Conv fp32, metadata copy, temp file + atomic rename, pinned input/output SHA-256 |
| `scripts/fork/segmentation/requirements-quant.txt` | pinned `onnxruntime` / `onnx` / `numpy` for the quantizer |
| `scripts/fork/segmentation/fetch-model-fp32.py` | cross-platform pinned fp32 download and manifest layout for CI (the `.sh` is the local equivalent) |
| `scripts/fork/segmentation/verify-requant.py` | encoder hash check with a per-tensor digest fallback for cross-platform rounding |
| `scripts/fork/segmentation/requant-digest.json` | 1,046-tensor digest of the pinned encoder, used by the fallback |
| `scripts/fork/segmentation/check-clip.py` | decode + word-count/WER gate for the committed clip |
| `scripts/fork/segmentation/fetch-model-fp32.sh` | pinned fp32 download, SHA-256, manifest layout |
| `scripts/fork/segmentation/run-asr-experiments.sh` | the three experiments with the engine's path; same-round int8 / requant / fp32 with `--finalize` |
| `scripts/fork/segmentation/ihm-sample.py` | the deterministic 1,000-row sample generator and the speaker-cluster bootstrap scorer |
| `scripts/fork/segmentation/score-meeteval.py` | tcpWER at collars 5 s and 1 s and cpWER, merged and speaker references, symmetric normalization |
| `scripts/fork/segmentation/score.py` | per-utterance scoring, with the short-row/backchannel table |
| `scripts/fork/segmentation/src/main.rs` | `--diarize`, `--split-turns`, `--window`, `--finalize` |
| `crates/tinta-engine-rs/src/speech.rs` | `transcribe_pieces`, the explicit piece-plan seam; empty-input guard in `transcribe` and `transcribe_pieces`, with tests |
| `crates/tinta-engine-rs/examples/decode-clip.rs` | decodes one clip as a single explicit piece and prints words as JSON; used by the portability CI job |
| `crates/tinta-engine-rs/tests/fixtures/es2004b-235.4-248.8.wav` | the 13.4 s AMI excerpt (CC BY 4.0, see `ATTRIBUTION.md`), 428,878 B |
| `crates/tinta-engine-rs/tests/fixtures/es2004b-235.4-248.8.ref.txt` | the fp32 reference text for that clip |

Evidence: `evidence/tinta-fork/23/results/` (decodes, scores, meeteval JSON,
RSS/time logs) and `evidence/tinta-fork/23/logs/`. `turns-<meeting>.json` are
the diarizer turns, `int8-default-<meeting>.json`,
`requant-default-<meeting>.json` and `fp32-default-<meeting>.json` the
same-round decodes, `fp32-runs-<meeting>.json` the no-re-decode pass,
`split-<meeting>.json` the speaker-split pass, and `ihm-sample-<variant>.jsonl`
the 1,000-row IHM decodes. The `r1/` directory keeps the first round's decodes
(int8 copied from task #21, fp32 without the finalize RSS measurement).

## Model licensing

Parakeet TDT 0.6B v3 is **CC BY 4.0**. NVIDIA's model card
(<https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3>) says "Use of this model
is governed by the CC-BY-4.0 license", and the Hub metadata is `cc-by-4.0`. The
`csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3` fp32 export and its int8
sibling carry the same license.

CC BY 4.0 allows redistributing an adapted model, including commercially, if
the redistribution:

1. credits **NVIDIA** as the creator;
2. links the license, <https://creativecommons.org/licenses/by/4.0/>;
3. **states that the model was modified** — here: "re-quantized from the
   sherpa-onnx fp32 ONNX export: MatMul-only per-channel int8 with
   `reduce_range=True`, Conv kept fp32";
4. does not imply that NVIDIA endorses Tinta;
5. adds no further restrictions (no DRM, no extra terms on the file).

Credit the sherpa-onnx export (`csukuangfj`) as the intermediate source as good
practice. The app's third-party notices should carry: "Parakeet TDT 0.6B v3 ©
NVIDIA, CC BY 4.0, modified (int8 re-quantization); ONNX export by
csukuangfj/sherpa-onnx."

A future hosting choice is a Hugging Face model repository under the user's
account with all four files (the requantized encoder plus the int8 decoder,
joiner and `tokens.txt`, about 18 MB extra), so one revision pins the whole
model. The engine would download
`https://huggingface.co/<repo>/resolve/<commit-sha>/<file>` and check the
sha256 in `models.json`, the same pattern as the current csukuangfj source. A
GitHub release asset on `GodGMN/tinta` is a fine mirror or fallback, not the
primary. Do not host the fp32 opt-in: keep pointing at csukuangfj's pinned
revision.

The draft model card (README) for that repository is
`docs/fork/model-card-requant.md`. No repository was created and nothing was
uploaded.

## Limitations

- **The IHM sample result is not conclusive.** 1,000 utterances give -1.78
  points for the requantized encoder against int8 with a speaker-cluster CI
  that crosses zero (-3.53 to +0.15). Only the full split can settle the
  remaining gap to the published 11.31 %.
- **The time numbers are ratios, not benchmarks.** The host is shared and was
  loaded during the runs; the same int8 79-piece plan measured 310 s in the
  task #21 round and 612 s here. The CPU seconds, model sizes and RSS are
  exact; the wall times are not.
- **The merged reference contains both sides of overlaps**, so tcpWER's
  absolute value is not reachable by any single stream. It is comparable
  across streams, not a product WER.
- **No UEM**: hypothesis words outside the annotated regions count as
  insertions for every stream. Acceptable for relative comparison, not for an
  absolute product WER.
- **The stream metrics use the same short-clip oracle decodes**, which are not
  an upper bound. The per-turn oracle with speakers (24.0 % tcpWER / 22.9 %
  cpWER) is the only oracle number that behaves like a bound, and only for a
  diarized hypothesis.
- **The speaker split used the engine's diarizer on Mix-Headset**, the same
  audio the pipeline hears, not the close-talk reference. It is the task #15
  post-clustered output (4 speakers on each meeting), which is the intended
  product path.
- **ES2004b is still the tuning meeting** for the segmentation settings; the
  models and metrics were measured on all three, but the settings themselves
  were not re-tuned for the new encoder.
- **No models are committed.** The fp32 files and the requantized encoder live
  in the evidence directory and are pinned by hash. The only committed audio is
  the 13.4 s AMI excerpt used by the portability CI job, with attribution in
  `crates/tinta-engine-rs/tests/fixtures/ATTRIBUTION.md`.
