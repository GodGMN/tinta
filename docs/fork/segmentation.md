# Segmentation: what the long-form pipeline loses and what the engine now does

Task #21 of the Windows/Linux port. Task #19 measured a **+18.0 [15.2-20.8] point**
same-reference gap between the dataset's oracle segment boundaries and the
engine's full-meeting VAD pipeline (25.0% vs 43.1% WER on ES2004b, IS1009b and
EN2002a), found that 82-89 of 110-136 pieces per meeting started at a forced VAD
flush, and recommended removing that cut, shortening pieces, and re-decoding long
hypothesis holes. This round measures each rule on the engine's own code path and
closes the review of PR #16.

**Result: 43.4% -> 37.9% on the same reference** (29.4% / 27.4% / 53.3%
per meeting). Against the per-turn oracle the gap goes from +18.4 to **+12.9
points**; against the fair single-stream oracle (below) the pipeline is
**+4.3 points** (3.4 / 4.0 / 5.3).

**The target is re-scoped to "within 3 points of the single-stream oracle".**
The per-turn oracle decodes each reference turn on its own, so in an overlap it
transcribes both speakers; the scorer then charges every word to the shortest
row containing its midpoint, which a single-channel pipeline cannot match. The
reviewer scored the same oracle words as one stream, keeping only the side with
more words in each shared interval: **25.9% / 23.4% / 47.9% (33.6% all three)**.
This repository's scorer reproduces it at **26.0% / 23.4% / 48.0% (33.6%)**.
That is the honest ceiling of a single mixed channel under this metric, and the
engine is now +4.3 points from it (+3.4 / +4.0 / +5.3). The original +3-point
bar, read against the per-turn oracle, was measuring the oracle's overlap
advantage, not a segmentation defect.

The PR #16 review fixes are in this round: re-decoded words are de-duplicated
against the words at the hole edges, the re-decode reports its own progress
instead of sitting at 100%, its audio is capped, the 60.0003 s minimum-length
test is back with the first-piece guard, and the quiet-point pre-roll has a
minimum (0.5 s measured better than 1.0 s; see the pre-roll ablation below).

## What changed in the engine

`crates/tinta-engine-rs/src/segment.rs` now has a `Settings` value instead of
constants; `Settings::default` is the winning combination:

| Setting | Value | Why |
|---|---|---|
| `vad_max_speech_seconds` | 60 | The VAD must not cut in the middle of speech. It closed a run at 30 s before; the 20 s flush Task #19 measured lived in the old `tinta-bench` harness, and the 30 s cut had the same failure mode. |
| `max_piece_seconds` | 57 | A piece is cut only at a silence (the VAD run) or, above 57 s, at the quietest 50 ms window in the last 40 %. With the rolls a piece is at most 60 s. |
| `join_gap_seconds` | 0.5 | Runs no further apart than a short pause are decoded together. |
| `pre_roll_seconds` | 2 | The pre-roll starts at the quietest 50 ms point in the 2 s before the speech, so a piece begins in silence, not mid-word. |
| `min_pre_roll_seconds` | 0.5 | The quietest point can sit milliseconds before the speech (the VAD onset is late on a soft start); the piece start is pulled back so it keeps at least this much audio anyway. Measured at 0.5 and 1.0; 0.5 was better on the tuning meeting. |
| `post_roll_seconds` | 1 | Context after the last speech. |
| `min_piece_seconds` | 10 | A short run is extended, so the recognizer has context. |
| `gap_redecode_seconds` | 3 | After the first pass, a 3 s+ hole inside a VAD speech region is decoded again from a quiet point, twice, and each window is retried from a shifted start while it still leaves a hole. Words that repeat a hole-edge word are dropped, and the pass is capped (below). |

The hole re-decode also reports its own progress. `SpeechPass::transcribe`
fills the first 80% of the transcription fraction with the first pass and 80-100%
with the re-decode, so the bar keeps moving instead of reaching 100% and then
sitting there for the extra pass (measured at +31% to +137% of the first pass on
these three meetings). Fractions only increase, and the final 1.0 still closes
the stage; nothing is emitted outside the transcription stage of the same track.
The re-decoded audio, retries included, is capped at **twice the first pass's
audio**, largest holes first (`REDECODE_BUDGET_SHARE` in `speech.rs`). A stricter
25% of the track was measured first: it skipped 40% of the planned holes on
EN2002a and cost 1.8 points on ES2004b and 3.5 on EN2002a, so the cap was moved
to where it does not bind on these meetings (they spend 61% and 148% of the
first pass's audio) but still stops a pathological track at 3x the first pass.
A hole that repeats a word already at its edge (same normalized text, start
within 0.3 s) is dropped by `segment::merge_hole_words`, so the "no duplicate
words at joins" condition holds: the re-decode now introduces the same number of
repeat pairs as a run without it (see below).

The rest of the PR #16 review is folded in:

- **Duplicate words at hole edges.** `merge_hole_words` and its unit tests cover
  the EN2002a `So` at 411.51/411.56 s case, a genuine repetition inside a hole,
  and a word outside the hole.
- **Cost and progress.** The module and default comments now say +31% to +137%,
  not "a few percent"; the re-decode reports 80-100% of the transcription
  fraction (never decreasing); its audio is capped at 2x the first pass, largest
  holes first.
- **The deleted regression test is back** (`a_cap_sized_span_plus_a_tail...`,
  the 60.0003 s span) and `split_span` again keeps the first piece above
  `min_piece_seconds` for the shorter-piece settings (`the_shorter_piece_settings...`,
  `a_span_that_would_leave_a_short_tail...`).
- **Pre-roll floor.** `min_pre_roll_seconds = 0.5`, measured against 1.0 on all
  three meetings (0.5 wins on the tuning meeting and overall).

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

The re-decode re-times its window, so a word at a hole edge can land just inside
the hole and be emitted twice (`So` at 411.51 s and 411.56 s on EN2002a, `same.`
at 770.04 and 770.06 s). The merge drops a re-decoded word whose normalized text
matches an existing word starting within 0.3 s, and unit tests cover that case,
the true repetition inside a hole, and a word just outside the hole. Measured
repeat pairs whose spans overlap by more than half: **ES2004b 1 / IS1009b 3 /
EN2002a 4**, exactly the same pairs and count as a run of the current code with
no re-decode (1 / 3 / 4; those are the decoder repeating itself inside one
piece). Before the fix the same count was 3 / 3 / 7 against `runs` at 2 / 3 / 4,
so the re-decode no longer adds any.

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
- **Two oracle bounds.** `score.py` prints both. The *per-turn* oracle is the
  task #19 hypothesis, one decode per reference turn. The *single-stream*
  oracle merges those words at absolute time and, for every pair of rows that
  overlaps by more than 0.2 s, keeps only the side with more words in the shared
  interval (that drops 300 / 334 / 1,040 words from ES2004b / IS1009b /
  EN2002a). It asks what the same scorer gives a perfect single-channel
  recognizer, which is the bar the pipeline is now measured against.
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

688 / 603 / 1,367 segments. **Oracles: per-turn 18.8% / 16.3% / 37.1% (25.0%);
single stream 26.0% / 23.4% / 48.0% (33.6%).** Every row ran the same engine code
with a different `Settings` value; the pieces column is ES2004b. The delta in
parentheses is against that meeting's per-turn oracle.

| Variant | Pieces | ES2004b (tuning) | IS1009b (held out) | EN2002a (held out) | All three |
|---|---|---|---|---|---|
| `before` - 30 s VAD cut, no gap re-decode | 122 | 38.0% (+19.2) | 31.5% (+15.2) | 57.2% (+20.1) | 43.4% (+18.4) |
| 1. `no-vad-cut` - VAD 60 s, 30 s pieces | 122 | 36.6% (+17.9) | 31.4% (+15.1) | 56.9% (+19.8) | 42.8% (+17.8) |
| 2. `short` - 12 s pieces cut at the quietest point | 229 | 41.2% (+22.4) | 34.8% (+18.5) | 57.5% (+20.4) | 45.5% (+20.5) |
| 2+3. `short-gap` - 12 s pieces + hole re-decode | 229 | 31.1% (+12.3) | 29.1% (+12.7) | 52.8% (+15.7) | 38.8% (+13.7) |
| `mid-gap` - 30 s pieces + hole re-decode | 115 | 29.5% (+10.7) | 26.9% (+10.5) | 52.7% (+15.6) | 37.6% (+12.5) |
| `runs` - cut only at VAD silences, no re-decode | 79 | 33.8% (+15.0) | 29.8% (+13.5) | 57.8% (+20.7) | 41.8% (+16.7) |
| 3. **`default` - silence cuts + hole re-decode + 0.5 s pre-roll floor** | **79** | **29.4% (+10.6)** | **27.4% (+11.1)** | **53.3% (+16.2)** | **37.9% (+12.9)** |
| 4. `turns` - decode each diarized turn separately | 269 | 34.6% (+15.8) | - | - | - |

Against the single-stream oracle the default is **+3.4 / +4.0 / +5.3 (+4.3
overall)**; the other rows predate the pre-roll floor and were not re-scored
against it. `default` and `runs` were re-decoded with the current code; the
remaining rows were measured in the previous round (they inherit
`Settings::default`, so a re-run now would include the 0.5 s floor, worth at most
0.3 points). The pre-roll floor ablation is separate:

| Pre-roll floor | ES2004b | IS1009b | EN2002a | All three |
|---|---|---|---|---|
| 0 s - quietest point only (previous default) | 29.1% | - | - | - |
| **0.5 s - the default** | **29.4%** | **27.4%** | **53.3%** | **37.9%** |
| 1.0 s | 30.2% | 27.1% | 53.2% | 38.1% |

The floor is the better of the two settings the review asked for, but it did not
buy the ~170 edge words the review expected: on the tuning meeting it costs 0.3
points against the quietest-only start (0.0/1.0 s: 29.1/30.2 on ES2004b), and
1.0 s is 0.2 points worse overall (37.9 vs 38.1) despite a small gain on the two
held-out meetings. The engine keeps 0.5 s so a soft onset is never clipped.

Reading the table:

- **Removing the forced cut helps a little; not cutting inside speech helps a
  lot.** VAD 60 s with 30 s pieces buys 0.6 points over `before`; cutting only at
  VAD silences (57 s pieces) buys another 1.0 on top of `no-vad-cut` and 1.6 over
  `before` overall.
- **The 8-15 s target loses on every meeting.** Cutting a continuous run at the
  quietest point doubles the cuts and each cut costs words: `short` is 7.4 points
  *worse* than `runs` on ES2004b and 3.7 worse overall, with insertions dropping
  as deletions rise. `short-gap` recovers almost all of it but stays 0.9 points
  behind the default overall and costs about 40% more ASR time.
- **The hole re-decode is the biggest single lever.** It is worth 3.9 points
  overall on top of `runs` (4.4 on ES2004b) and 6.7 on top of `short`, because it
  repairs the decoder jumps.
- **30 s and 57 s pieces are within 0.3 points overall** (37.6% for `mid-gap`
  before this round's floor, 37.9% for the default after it; not an
  apples-to-apples pair). The default keeps 57 s because the difference is inside
  the scoring noise on one meeting, 57 s needs fewer pieces (79 vs 115 on
  ES2004b), and the ASR seconds are within a few percent either way. Changing the
  piece length is out of scope for this review round.
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

| Meeting | Role | Per-turn oracle | Single-stream oracle | Before | After | Delta vs per-turn | Delta vs single stream |
|---|---|---|---|---|---|---|---|
| ES2004b | tuning | 18.8% | 26.0% | 38.0% | **29.4%** | +19.2 -> **+10.6** | **+3.4** |
| IS1009b | held out | 16.3% | 23.4% | 31.5% | **27.4%** | +15.2 -> **+11.1** | **+4.0** |
| EN2002a | held out | 37.1% | 48.0% | 57.2% | **53.3%** | +20.1 -> **+16.2** | **+5.3** |
| All three | | 25.0% | 33.6% | 43.4% | **37.9%** | +18.4 -> **+12.9** | **+4.3** |

Deletion counts: 6,984 -> 5,353 over the three meetings (oracle 2,359);
insertions 778 -> 960 (oracle 1,088); substitutions 1,215 -> 1,524 (oracle
1,728). The change removes deletions and leaves the rest.

### Where the remaining gap is

Same scoring, split by whether the reference window overlaps another reference
window by more than 0.2 s (56-70% of the reference words in these meetings are
in overlapping windows):

| Meeting | Rows | Oracle | Pipeline | Delta |
|---|---|---|---|---|
| ES2004b solo | 239 (2,974 words) | 8.8% | 13.9% | **+5.0** |
| ES2004b overlap | 449 (3,859 words) | 26.4% | 41.3% | +14.9 |
| IS1009b solo | 191 (2,374 words) | 7.3% | 12.1% | +4.8 |
| IS1009b overlap | 412 (3,607 words) | 22.3% | 37.5% | +15.3 |
| EN2002a solo | 382 (2,265 words) | 15.1% | 24.9% | +9.8 |
| EN2002a overlap | 985 (5,593 words) | 46.0% | 64.8% | +18.8 |

The oracle decodes each reference turn on its own, including turns that overlap
in time, so it transcribes both sides of a conversation overlap. The pipeline
decodes the mix once. In an overlap the recognizer returns the dominant speaker
and drops the other; no cut rule can recover words the model did not emit. The
single-stream oracle prices that in: on the overlap rows it gives 39.3% / 33.9% /
60.7%, so the pipeline's real excess there is only **+2.0 / +3.6 / +4.1** rather
than +14.9 / +15.3 / +18.8. The solo rows agree between the two oracles and are
the honest measure of what segmentation can still buy: +5.0 / +4.8 / +9.8.
Closing that would need a better model or source separation, not segmentation.

The other quantified limit is the long-form decoder itself: the int8 TDT decoder
drops stretches inside long windows (see above). The retry repairs what it can
find, but a hole that produces words again on a shifted start is luck, not a
guarantee. EN2002a still has 26 holes of 3 s+ (133 s), mostly overlapping talk,
and is the meeting where the gap stays widest.

### Cost

ASR seconds for the whole track at 2 threads (not a throughput claim): ES2004b
203.5 -> 317.9 s, IS1009b 173.3 -> 248.8 s, EN2002a 192.0 -> 455.5 s. That is
**+56% / +44% / +137%**; the previous round measured +54% / +31% / +137% on the
same code path, so the honest range is **+31% to +137%** and the module comments
quote it. The increase is the hole re-decodes and their retries; EN2002a pays the
most because its overlap-dense talk produces the most holes. A hole that already
came back empty is not retried in the second round.

The cap is measured: the re-decode plans 466 / 230 / 915 s of holes and hands the
recognizer 1,475 / 874 / 3,395 s including retries, against the 2x-first-pass cap
(4,698 / 4,054 / 4,586 s), so it does not bind on these meetings. Pieces per
meeting drop (122/94/122 -> 79/61/84), so the first pass itself is slightly
cheaper.

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
| `crates/tinta-engine-rs/src/segment.rs` | `Settings`, silence-only piece planning, quiet-point pre-roll with a floor, `speech_gaps`, `merge_hole_words` |
| `crates/tinta-engine-rs/src/speech.rs` | `vad_config`, `with_settings`, hole re-decode with start-shift retries, the audio cap and its progress |
| `scripts/fork/segmentation/src/main.rs` | harness: the engine path on one wav, variants, window and turns modes |
| `scripts/fork/segmentation/score.py` | same-reference ablation scoring, per-turn and single-stream oracles |
| `scripts/fork/segmentation/run.sh` | models layout, builds, decodes, tables |

Results and logs: `evidence/tinta-fork/21/results/` and
`evidence/tinta-fork/21/logs/`. `default-<meeting>.json` is the engine defaults;
`score-default.json` is its score. The other variant files are the sweep.

## Limitations

- **The re-scoped "within 3 points of the single-stream oracle" bar is not met**
  (+3.4 / +4.0 / +5.3 per meeting, +4.3 overall). ES2004b is closest; EN2002a,
  where the single-stream oracle is itself 48.0%, is furthest.
- **The 0.5 s pre-roll floor is a correctness choice, not a gain.** It fixes
  clipped soft onsets and was the better of the two settings the review asked
  for, but it costs 0.3 points on ES2004b against the quietest-only start (the
  ~170 piece-edge words the review expected it to recover did not move). It is
  0.2 points better overall than the 1.0 s setting (37.9 vs 38.1).
- **ES2004b chose the parameters**, including the pre-roll floor. IS1009b and
  EN2002a confirm the direction but were not used to choose anything.
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
