# Segmentation: what the long-form pipeline loses and what the engine now does

Task #21 of the Windows/Linux port. Task #19 measured a **+18.0 [15.2-20.8] point**
same-reference gap between the dataset's oracle segment boundaries and the
engine's full-meeting VAD pipeline (25.0% vs 43.1% WER on ES2004b, IS1009b and
EN2002a), found that 82-89 of 110-136 pieces per meeting started at a forced VAD
flush, and recommended removing that cut, shortening pieces, and re-decoding long
hypothesis holes. This round measures each rule on the engine's own code path.

**Result: the gap is down from +18.4 to +12.6 points** (43.4% -> 37.6% on the
same reference). ES2004b, the tuning meeting, goes from 38.0% to **29.1%**
(+19.2 -> +10.3); held-out IS1009b from 31.5% to **27.2%** (+15.2 -> +10.9);
held-out EN2002a from 57.2% to **53.0%** (+20.1 -> +15.9).

**The +3 point target was not reached.** The table below shows why: after the
changes, the pipeline is **+5.0 points** from the oracle on the rows whose
reference window does not overlap another speaker, and **+14.4** on the rows
where it does. The oracle decodes each reference turn separately, so in an
overlap it transcribes both speakers; a single-pass recognizer transcribes the
mix once. That is a crosstalk ceiling of the oracle comparison, not a
segmentation defect. The steps that *are* segmentation defects - the forced
flush, the pieces that start mid-speech, and the decoder holes - are fixed.

## What changed in the engine

`crates/tinta-engine-rs/src/segment.rs` now has a `Settings` value instead of
constants; `Settings::default` is the winning combination:

| Setting | Value | Why |
|---|---|---|
| `vad_max_speech_seconds` | 60 | The VAD must not cut in the middle of speech. It closed a run at 30 s before; the 20 s flush Task #19 measured lived in the old `tinta-bench` harness, and the 30 s cut had the same failure mode. |
| `max_piece_seconds` | 57 | A piece is cut only at a silence (the VAD run) or, above 57 s, at the quietest 50 ms window in the last 40 %. With the rolls a piece is at most 60 s. |
| `join_gap_seconds` | 0.5 | Runs no further apart than a short pause are decoded together. |
| `pre_roll_seconds` | 2 | The pre-roll starts at the quietest 50 ms point in the 2 s before the speech, so a piece begins in silence, not mid-word. |
| `post_roll_seconds` | 1 | Context after the last speech. |
| `min_piece_seconds` | 10 | A short run is extended, so the recognizer has context. |
| `gap_redecode_seconds` | 3 | After the first pass, a 3 s+ hole inside a VAD speech region is decoded again from a quiet point, twice, and each window is retried from a shifted start while it still leaves a hole. |

`vad_config()` in `speech.rs` is now the single source of the VAD settings (the
install check used a copy), and `tokens_to_words()` is public so the harness
builds the same words from a bare recognizer call.

The second change is a measured decoder failure, not a tuning choice. The int8
Parakeet TDT decoder occasionally emits the first word of a window and then jumps
its token timeline to the end: a 13.4 s window at ES2004b 235.4-248.8 returns
only `Okay. Um`, with an 8.6 s blank between two tokens, while the same window
starting 100 ms earlier transcribes fully. In a long meeting this is what the
"empty piece holes" in task #19 were. The gap re-decode finds those holes and the
start-shift retry recovers most of them; on ES2004b the 3 s+ hole count drops
from 47 (342 s) to 8 (38 s).

## Method

- **The engine's own path.** `scripts/fork/segmentation` is a standalone crate
  that links `tinta-engine-rs`. It reads one 16 kHz meeting wav, runs the
  engine's VAD (`vad_segments`), piece planner (`group_pieces_with`) and
  recognizer path (`SpeechPass::transcribe`) exactly as `finalize` does, and
  writes the words plus the VAD ranges and the piece plan. It is not a copy of
  the rules; the measured code is the shipped code.
- **Same reference and scoring as task #19.** The reference is the HF IHM
  segment text of every row of the split (`rows.json`); each pipeline word is
  assigned to exactly one segment window by midpoint (the shortest containing
  window); WER is summed per-utterance edit distance over summed reference words
  with Whisper's `EnglishTextNormalizer`. Oracle hypotheses are the same
  Mix-Headset decodes of the dataset's own segments that task #19 used.
- **Meetings.** ES2004b is the tuning meeting; IS1009b and EN2002a are held out.
  All parameters were chosen on ES2004b only.
- **Variants.** Every row runs the same engine code with a `Settings` value.
  `before` reproduces the engine as it was (a 30 s VAD run, one piece per run, no
  gap re-decode); it scores 38.0 / 31.5 / 57.2 per meeting against task #19's
  36.3 / 31.6 / 57.7 with the old harness, so the harness is calibrated.
- **No timing claims.** Everything ran under `nice -n 19` with 2 threads. The
  ASR seconds per meeting are reported as a cost, not a benchmark.

## Results

### Every variant on the three meetings

688 / 603 / 1,367 segments; oracle 18.8% / 16.3% / 37.1% / 25.0%. Every row ran
the same engine code with a different `Settings` value; the pieces column is
ES2004b. The delta in parentheses is against that meeting's oracle.

| Variant | Pieces | ES2004b (tuning) | IS1009b (held out) | EN2002a (held out) | All three |
|---|---|---|---|---|---|
| `before` - 30 s VAD cut, no gap re-decode | 122 | 38.0% (+19.2) | 31.5% (+15.2) | 57.2% (+20.1) | 43.4% (+18.4) |
| 1. `no-vad-cut` - VAD 60 s, 30 s pieces | 122 | 36.6% (+17.9) | 31.4% (+15.1) | 56.9% (+19.8) | 42.8% (+17.8) |
| 2. `short` - 12 s pieces cut at the quietest point | 229 | 41.2% (+22.4) | 34.8% (+18.5) | 57.5% (+20.4) | 45.5% (+20.5) |
| 2+3. `short-gap` - 12 s pieces + hole re-decode | 229 | 31.1% (+12.3) | 29.1% (+12.7) | 52.8% (+15.7) | 38.8% (+13.7) |
| `mid-gap` - 30 s pieces + hole re-decode | 115 | 29.5% (+10.7) | 26.9% (+10.5) | 52.7% (+15.6) | 37.6% (+12.5) |
| `runs` - cut only at VAD silences, no re-decode | 79 | 33.3% (+14.5) | 29.7% (+13.4) | 57.2% (+20.1) | 41.4% (+16.3) |
| 3. **`default` - silence cuts + hole re-decode** | **79** | **29.1% (+10.3)** | **27.2% (+10.9)** | **53.0% (+15.9)** | **37.6% (+12.6)** |
| 4. `turns` - decode each diarized turn separately | 269 | 34.6% (+15.8) | - | - | - |

Reading the table:

- **Removing the forced cut helps a little; not cutting inside speech helps a
  lot.** VAD 60 s with 30 s pieces buys 0.6 points over `before`; cutting only at
  VAD silences (57 s pieces) buys another 3.5 on top of `no-vad-cut` and 2.0 over
  `before` overall.
- **The 8-15 s target loses on every meeting.** Cutting a continuous run at the
  quietest point doubles the cuts and each cut costs words: `short` is 5.4 points
  *worse* than `runs` on ES2004b and 4.1 worse overall, with insertions dropping
  as deletions rise. `short-gap` recovers 10 points but stays 1.2 points behind
  the default overall and costs about 40% more ASR time.
- **The hole re-decode is the biggest single lever.** It is worth 3.8 points
  overall on top of `runs` (4.2 on ES2004b) and 6.7 on top of `short`, because it
  repairs the decoder jumps.
- **30 s and 57 s pieces tie overall** (37.6% both); the default keeps 57 s
  because it won ES2004b (29.1 vs 29.5) and is 9% cheaper there in ASR seconds
  (312 vs 342 s). The 0.3-point differences on the held-out meetings are inside
  the scoring noise on one meeting.
- **A 2 s hole threshold and a 3 s minimum piece are noise** (+0.2 points on
  ES2004b, where both were measured), so the default keeps 3 s and 10 s, the
  values task #19 recommended.
- **Step 4 loses in this form.** Decoding each over-clustered diarization turn
  with the engine's midpoint join drops 1,085 of 6,027 words at the joins
  (41.0%); keeping overlapping occurrences instead produces 576 insertions
  (34.6%). The turn windows also fragment into 269 pieces. It was only measured
  on ES2004b: the task-2 evidence has no diarization result for the other two
  meetings.

### Final settings on the three meetings

| Meeting | Role | Oracle | Before | After | Delta before -> after |
|---|---|---|---|---|---|
| ES2004b | tuning | 18.8% | 38.0% | **29.1%** | +19.2 -> **+10.3** |
| IS1009b | held out | 16.3% | 31.5% | **27.2%** | +15.2 -> **+10.9** |
| EN2002a | held out | 37.1% | 57.2% | **53.0%** | +20.1 -> **+15.9** |
| All three | | 25.0% | 43.4% | **37.6%** | +18.4 -> **+12.6** |

Deletion counts: 6,984 -> 5,318 over the three meetings (oracle 2,359);
insertions 778 -> 953 (oracle 1,088); substitutions 1,215 -> 1,509 (oracle
1,728). The change removes deletions and leaves the rest.

### Where the remaining gap is

Same scoring, split by whether the reference window overlaps another reference
window by more than 0.2 s (56-70% of the reference words in these meetings are
in overlapping windows):

| Meeting | Rows | Oracle | Pipeline | Delta |
|---|---|---|---|---|
| ES2004b solo | 239 (2,974 words) | 8.8% | 13.9% | **+5.0** |
| ES2004b overlap | 449 (3,859 words) | 26.4% | 40.8% | +14.4 |
| IS1009b solo | 191 (2,374 words) | 7.3% | 12.2% | +4.9 |
| IS1009b overlap | 412 (3,607 words) | 22.3% | 37.1% | +14.8 |
| EN2002a solo | 382 (2,265 words) | 15.1% | 25.2% | +10.1 |
| EN2002a overlap | 985 (5,593 words) | 46.0% | 64.3% | +18.3 |

The oracle decodes each reference turn on its own, including turns that overlap
in time, so it transcribes both sides of a conversation overlap. The pipeline
decodes the mix once. In an overlap the recognizer returns the dominant speaker
and drops the other; no cut rule can recover words the model did not emit. The
solo rows are the honest measure of what segmentation can still buy, and there
the engine is +5 points from a reference-boundary oracle. Closing that would
need a better model or source separation, not segmentation.

The other quantified limit is the long-form decoder itself: the int8 TDT decoder
drops stretches inside long windows (see above). The retry repairs what it can
find, but a hole that produces words again on a shifted start is luck, not a
guarantee. EN2002a still has 26 holes of 3 s+ (133 s), mostly overlapping talk,
and is the meeting where the gap stays widest.

### Cost

ASR seconds for the whole track at 2 threads (not a throughput claim): ES2004b
203 -> 312 s, IS1009b 173 -> 227 s, EN2002a 192 -> 456 s. The increase is the
hole re-decodes and their retries; EN2002a pays the most because its overlap
dense talk produces the most holes. A hole that already came back empty is not
retried in the second round. Pieces per meeting drop (122/94/122 -> 79/61/84),
so the first pass itself is slightly cheaper.

## Reproduction

```bash
export ROOT=/path/to/evidence/tinta-fork/2       # models, meeting wavs
export ROOT19=/path/to/evidence/tinta-fork/19    # rows.json, oracle decodes
export SEG=/path/to/evidence/tinta-fork/21       # results for this task
scripts/fork/segmentation/run.sh
```

`run.sh` links the task-2 models into the manifest layout, builds the harness,
decodes each (variant, meeting) once (it skips an existing JSON), and prints the
score tables. One run by hand:

```bash
scripts/fork/segmentation/target/release/tinta-segmentation \
  --wav "$ROOT/data/ES2004b.Mix-Headset.wav" \
  --models-root "$SEG/models" --variant default --threads 2 \
  --output "$SEG/results/default-ES2004b.json"

"$ROOT/venv/bin/python" scripts/fork/segmentation/score.py \
  --rows "$ROOT19/data/ihm-baseline-full/rows.json" \
  --oracle "ES2004b=$ROOT19/results/mix-decode.jsonl" \
  --pipeline "ES2004b=$SEG/results/default-ES2004b.json"
```

`--window START:END` decodes one window and prints its tokens and timestamps;
that is how the decoder jump was found. `--turns FILE` runs the step 4
experiment against a diarization result.

## Files

| Path | Purpose |
|---|---|
| `crates/tinta-engine-rs/src/segment.rs` | `Settings`, silence-only piece planning, quiet-point pre-roll, `speech_gaps` |
| `crates/tinta-engine-rs/src/speech.rs` | `vad_config`, `with_settings`, hole re-decode with start-shift retries |
| `scripts/fork/segmentation/src/main.rs` | harness: the engine path on one wav, variants, window and turns modes |
| `scripts/fork/segmentation/score.py` | same-reference ablation scoring |
| `scripts/fork/segmentation/run.sh` | models layout, builds, decodes, tables |

Results and logs: `evidence/tinta-fork/21/results/` and
`evidence/tinta-fork/21/logs/`. `default-<meeting>.json` is the engine defaults;
`score-default.json` is its score. The other variant files are the sweep.

## Limitations

- **The +3 point bar is not met** (+10.3 / +10.9 / +15.9 per meeting, +12.6
  overall). The measured cause is the crosstalk advantage of an oracle that
  decodes each reference turn separately; see above.
- **ES2004b chose the parameters.** IS1009b and EN2002a confirm the direction but
  were not used to choose anything.
- **The ablation scoring is strict**: a pipeline word is assigned to one segment
  window, so a word at a boundary is charged to one side, as in task #19.
- **EN2002a is hard**: 37.1% oracle on Mix-Headset and 17.1% on headset. It is
  included because task #19 asked for an `EN2002*` meeting, not because it is
  typical.
- **The step 4 experiment used the task #2 over-clustered diarization** (30
  clusters on ES2004b), not the engine's reclustered turns. It is an upper bound
  on what finer turns can do, and it loses.
- **No timing claims.** The host is shared, and every heavy step ran under
  `nice -n 19` with 2 threads.
