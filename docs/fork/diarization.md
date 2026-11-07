# Diarization: post-clustering and the speaker-embedding model (task 18)

Task #2 left one clear gap: on the held-out AMI test meetings the sherpa-onnx
diarization pipeline scored **24.6-29.1% DER** (pyannote.metrics, no collar,
overlap counted) while an oracle that only relabelled the same raw clusters
scored **13.8-16.6%**. The front end (pyannote segmentation 3.0 + CAMPPlus
embeddings) was already at that level; the loss happened in the last step, when
complete linkage collapsed 30-60 clusters into a handful of speakers: a fifth
cluster holding 9-11% of the speech survived, and dozens of tiny clusters were
reassigned to whichever large cluster was nearest **in time**.

This is the follow-up. It changes only the post-clustering step and compares two
speaker-embedding models. Nothing in the ASR path changes.

**Result: the no-collar DER on the four AMI test meetings drops from
21.5-41.7% (mean 28.4%) to 14.1-28.9% (mean 18.6%) with the same segmenter and
the same CAMPPlus embeddings.** Three of the four meetings land at 14.1-16.5%
and the speaker count is right on 4/4. The final DER is within 0.5 points of the
many-to-one oracle of the **raw** over-clustered partition on all four meetings
(16.5 vs 16.6%, 14.8 vs 15.3%, 14.1 vs 14.0%, 28.9 vs 28.5%). That oracle is a
fair reference for "how much of the remaining error is clustering", but it is
not a lower bound on turn-level reassignment: the final output beats it on
ES2004b and IS1009b. Confusion falls to 2.1-11.0%; what remains is the front
end, the reference definition and the speaker embeddings.

The English WeSpeaker ResNet34 is **not shown to be better** here: 20.2% mean
and 2/4 speaker counts (it ties on ES2004b and TS3003b, loses on IS1009b and
EN2002a), against 18.6% and 4/4. Its embedding passes also cost **82-133 CPU-s
per meeting** against 30-62 CPU-s for CAMPPlus, which is the stronger reason to
keep CAMPPlus.

## What changed

### Post-clustering (`scripts/fork/bench/src/clustering.rs`)

The engine may not force a fixed speaker count, and no single cosine threshold
separates "one person recorded twice" from "two people who sound alike". So the
pipeline over-clusters and then repairs the partition:

1. **Over-cluster at threshold 0.7** (the task #2 default; in sherpa-onnx
   `threshold` is a cosine *dissimilarity*: higher merges more). This
   deliberately splits speakers.
2. **Merge by centroid.** Every cluster gets a duration-weighted centroid of its
   turn embeddings. The closest pair of centroids is merged while the cosine
   distance stays below `merge_distance`, recomputing the centroid after each
   merge. A cached distance matrix is updated only in the merged row; a long
   meeting does not recompute every pair's dot product on every round.
3. **Reassign by voice, never by time.** Clusters holding less than `min_share`
   of the diarized segment time are dissolved and each of their turns goes to
   the kept centroid it is closest to in embedding space. The old benchmark
   reassigned those segments to whichever large cluster was nearest in time;
   that is what put a 216-252 s split of one speaker into a neighbour. The share
   is a fraction of the summed unit durations, so overlapped units count twice
   and gaps of up to 0.5 s inside a unit count as speech.

The whole algorithm is `recluster(&[Unit], &Options) -> Outcome` in
`scripts/fork/bench/src/clustering.rs`: a pure function over
`(embedding, seconds, initial cluster)`, no I/O, no model handles, no fixed
speaker count. It is meant to move into `crates/tinta-engine-rs`; the unit
builder is part of the recipe but still lives in `main.rs` (see "For the
engine"). It handles failed embeddings without panicking: a zero, empty or
non-finite embedding is excluded from the centroids and its unit joins the
largest final cluster. Unit tests cover merging, keeping speakers apart,
"nearest voice, not nearest time", the cached distance matrix, NaN and failed
embeddings, empty input, and golden extracts of the cached embeddings.

Units, not segments: consecutive segments of the same cluster less than 0.5 s
apart are joined into one *unit* before embedding, so a 0.3 s fragment never
gets an embedding of its own (an embedding needs a few hundred ms of voice).

Cost: the step needs one embedding per turn, which sherpa-onnx's diarization API
does not expose, so the harness computes them again. Measured as process CPU at
2 threads: **30-62 CPU-s per 27-37 min meeting, +14-33% on top of the
diarization stage**. The two measured pairs are IS1008b 30.4/222.9 CPU-s (14%)
and EN2002a 61.9/187.8 (33%); the other runs fall between. The ES2004b
CAMPPlus embedding pass was never measured (its bootstrap file is missing), so
it is not part of that range. In the engine the same pass can produce the
per-turn embeddings the clustering needs from a lower-level pipeline; with
sherpa-onnx's high-level diarization API it is an extra pass, not free.

### Embedding model

| Model | File | Size | Dim | SHA-256 |
|---|---|---|---|---|
| 3D-Speaker CAMPPlus | `3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx` | 28 MB | 512 | `f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11` |
| WeSpeaker ResNet34 | `wespeaker_en_voxceleb_resnet34.onnx` | 26 MB | 256 | `5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94` |

Both are sherpa-onnx release assets under `speaker-recongition-models` (the
upstream tag is misspelled; the URL is correct):

- <https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx>
- <https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx>

The CAMPPlus checkpoint is trained on Mandarin, the WeSpeaker one on English
VoxCeleb; the WeSpeaker family is what pyannote 3.1 uses. Both are pinned by URL
and SHA-256 in `scripts/fork/bench/fetch-data.sh`, which checks the hash before
every run.

## Data and references

Audio: AMI Mix-Headset (16 kHz mono) from the public corpus mirror, the same
files task #2 used.

References: **BUTSpeechFIT `AMI-diarization-setup`, `only_words` RTTMs**, pinned
to commit `2509d8933721023fab4def2618aabd5c28eb82e9` (2023-10-17). This is the
setup pyannote and FluidAudio publish AMI DER against: every word is a speech
region, adjacent words of one speaker are merged, and the UEM is the whole
recording. Task #2 scored against the manual `transcriber_start/end` segments,
which differ by 1-3 points on these meetings; the numbers in this document are
not comparable to that table without re-scoring.

Split per that setup's own lists, not the AMI full-corpus partition:

| Split | Meeting | Recording | Reference speech | Speakers | Overlapped speech |
|---|---|---|---|---|---|
| dev (tuning) | IS1008b | 29.5 min | 22.7 min | 4 | 4.8% |
| dev (tuning) | ES2011c | 26.9 min | 22.2 min | 4 | 14.1% |
| test | ES2004b | 39.1 min | 33.6 min | 4 | 10.1% |
| test | IS1009b | 34.2 min | 29.6 min | 4 | 10.1% |
| test | TS3003b | 36.8 min | 29.0 min | 4 | 4.3% |
| test | EN2002a | 35.7 min | 31.6 min | 4 | 27.4% |

Four test meetings, not the sixteen in the list: one diarization run is 27-37
minutes of audio and 150-300 CPU-s, and the host is shared with another
project. The four cover all four families in the test list (ES, IS, TS, EN) and
146 minutes of audio.

None of these six meetings is in the setup's `lists/train.meetings.txt`:
IS1008b and ES2011c are dev (validation) meetings and the other four are test.
The earlier benchmark document's note that **TS3007b and ES2003b** are training
meetings applies to those two meetings, not to the ones used here.

## Scoring

`scripts/fork/bench/score.py`:

- `der_pyannote`: `pyannote.metrics` `DiarizationErrorRate` over the full-file
  UEM from the setup's `.uem`, `skip_overlap=False`. Primary is collar 0 s.
  pyannote's `collar=0.25` is a **0.25 s total** collar, i.e. **+/-0.125 s**;
  +/-0.25 s is `collar=0.5`. Both are reported next to the no-collar number.
  Miss, false alarm and confusion are reported separately, each divided by
  reference speech time.
- `oracle`: maps every hypothesis cluster to the reference speaker it overlaps
  most (many-to-one, several clusters may map to one speaker) and re-scores.
  This is a reference for what relabelling the existing clusters achieves, not
  a lower bound: reassigning individual turns can beat it. It is reported for
  the raw clusters and for the final output.
- Speaker-count accuracy: final cluster count against the reference count, per
  meeting and per model. No `num_clusters` is ever passed to sherpa-onnx
  (`--num-speakers -1`).

## Tuning on dev

The only knobs are `merge_distance` and `min_share`. The sweep ran on the two
dev meetings only, against the dev RTTMs, 0.10-0.30 in steps of 0.05 and
2/3/5% of speech; `diarization.sh sweep` reproduces that grid.

Both dev meetings, baseline over-clustering at 0.7:

| `merge_distance` | `min_share` | CAMPPlus mean DER | WeSpeaker mean DER | speakers, CAMPPlus (IS1008b / ES2011c) |
|---|---|---|---|---|
| 0.10 | 2% | 18.4% | 17.9% | 4/4, 5/4 |
| 0.10 | 3% | 17.5% | 17.9% | 4/4, 4/4 |
| 0.15 | 2% | 17.9% | 18.3% | 4/4, 5/4 |
| **0.15** | **3%** | **17.4%** | **17.8%** | **4/4, 4/4** |
| 0.15 | 5% | 17.4% | 17.8% | 4/4, 4/4 |
| 0.20 | 2% | 17.5% | 18.3% | 4/4, 4/4 |
| 0.20 | 3% | 17.5% | 18.3% | 4/4, 4/4 |
| 0.25 | 3% | 34.5% | 34.4% | 4/4, **2/4** |
| 0.30 | 3% | 43.1% | 56.2% | 4/4, **1/4** |

`min_share` above 2% is what fixes ES2011c: at 2% a 2.1% cluster survives and
the count is 5. The dev surface is flat between `merge_distance` 0.10 and 0.20
(17.4-17.9% mean, correct count everywhere) and flat between 2% and 5% of
speech for CAMPPlus; the 5-9% band was only swept for WeSpeaker. The frozen
choice is

| Setting | Value |
|---|---|
| over-clustering `threshold` | 0.7 |
| `merge_distance` | **0.15** (cosine distance between centroids) |
| `min_share` | **3%** of speech |
| `num_speakers` | -1 (never fixed) |

These are the code defaults (`Options::default()` and the `postcluster` CLI).

The tie between 0.15 and 0.20 is broken towards the smaller distance: merging
two speakers costs ~25 DER points, failing to merge a split costs ~1, so the
conservative side is worth a tenth of a point.

Two more stopping rules were implemented and measured in the first pass, then
**removed from the code** rather than kept as dead knobs:

- `min_loyalty` dissolved a cluster when fewer than that fraction of its own
  turns were nearest to its own centroid. On CAMPPlus it failed exactly where it
  should help: at `merge_distance` 0.25 the fourth speaker of ES2004b maps to
  85% one reference speaker but only 41% of its turns are individually nearest
  to its own centroid, so loyalty 0.5 dissolved a real speaker (3 clusters,
  27.6% DER, against 16.8% with the guard off). At the frozen 0.15/3% it is
  inert on ES2004b. That decision used ES2004b, a test meeting; see the
  held-out note below.
- `protect_share` (never merge two clusters that both exceed a speech share)
  did nothing the distance threshold did not already do on these meetings,
  because the closest pair is rarely two large clusters before the threshold is
  already too high.

## Held-out discipline (what was touched when)

The test meetings were not untouched before the freeze, and the artifact
timestamps cannot prove the freeze order:

- `score-raw-ES2004b` and `oracle-raw-ES2004b` are from 02:58, before any dev
  sweep. ES2004b was scored before tuning.
- The first test outputs (`pc/bootstrap-{IS1009b,EN2002a,TS3003b}-campplus`,
  03:11-03:45) were produced at the **old defaults** (0.25 / 2% / loyalty 0.5)
  and are not the numbers in this document. The frozen test runs are the
  `m0.15-s0.03` files at 03:48.
- The dev CAMPPlus files behind the dev table were rewritten at 04:30-04:35,
  *after* those test runs. A pre-test dev sweep existed at 03:1x-03:3x but was
  overwritten, so its exact contents cannot be re-checked. The WeSpeaker dev
  sweep (05:00-05:02) also ran after the WeSpeaker test runs (04:58).
- The `min_loyalty` decision described above used ES2004b (test) results at
  `merge_distance` 0.25. The guard was removed, not tuned in.

The impact looks low because the test surface is flat around the frozen point
(see the sensitivity table): the settings within +/-0.05 and +/-1% of speech
move the test mean by at most ~1.1 points. Next time the frozen configuration
should be committed before any test run, and the sweep artifacts kept.

The split itself is clean: dev and test meetings are disjoint, and
`dev.speakers.txt` ∩ `test.speakers.txt` is empty. No reference information
reaches the system: `--num-speakers -1`, threshold 0.7, no speaker count or
labels are passed, and the UEMs are whole-file (0 to the audio end).

## Results

Frozen settings from the dev table above, run once per test meeting. `raw` is
the over-clustered output at threshold 0.7 with no post-processing; `+ time
filter` is task #2's method (dissolve clusters under 2% and reassign by temporal
proximity); `post-cluster` is the new step. Oracle maps the clusters in that
column to their best reference speaker.

### Ablation: which step does the work

CAMPPlus, merge distance 0.15, test meetings:

| Meeting | raw (30-60 clusters) | merge only (`min_share` 0) | merge + reassign (frozen) |
|---|---|---|---|
| ES2004b | 22.4% | 18.7% | **16.5%** |
| IS1009b | 28.0% | 18.1% | **14.8%** |
| TS3003b | 25.0% | 15.6% | **14.1%** |
| EN2002a | 45.5% | 32.0% | **28.9%** |
| mean | 30.2% | 21.1% | **18.6%** |

Merging alone recovers 9 points, and reassigning the remaining small clusters by
voice recovers 2.5 more. Both steps of the recipe are needed; the merge matters
most on IS1009b, where the 11.4% "fifth speaker" is a split of a real one at
centroid distance 0.033.

### Model x clustering variant

One row per model and meeting. Both models start from the **same**
over-clustered partition (see "Why the model comparison uses a shared partition"
below), so the `raw` rows are identical.

| Meeting | Model | Variant | Clusters | DER no collar | DER +/-0.125 s | miss / FA / conf | Oracle |
|---|---|---|---|---|---|---|---|
| ES2004b | CAMPPlus | raw th 0.7 | 30 | 22.4% | 17.6% | 9.1 / 4.4 / 8.9 | 16.6% |
| ES2004b | CAMPPlus | + time filter | 6 | 21.5% | 16.9% | 9.1 / 4.4 / 8.9 | 16.6% |
| ES2004b | CAMPPlus | **post-cluster** | **4** | **16.5%** | 12.2% | 9.1 / 4.4 / 3.0 | 16.5% |
| ES2004b | WeSpeaker | **post-cluster** | **4** | **16.6%** | 12.3% | 9.1 / 4.4 / 3.2 | 16.6% |
| IS1009b | CAMPPlus | raw th 0.7 | 35 | 28.0% | 23.3% | 7.3 / 3.1 / 17.6 | 15.3% |
| IS1009b | CAMPPlus | + time filter | 5 | 26.8% | 22.3% | 7.3 / 3.1 / 17.6 | 15.3% |
| IS1009b | CAMPPlus | **post-cluster** | **4** | **14.8%** | 10.3% | 7.3 / 3.1 / 4.5 | 14.8% |
| IS1009b | WeSpeaker | post-cluster | 5 | 17.8% | 12.9% | 7.3 / 3.1 / 7.4 | 15.9% |
| TS3003b | CAMPPlus | raw th 0.7 | 28 | 25.0% | 20.1% | 5.1 / 6.9 / 13.0 | 14.0% |
| TS3003b | CAMPPlus | + time filter | 5 | 23.8% | 18.9% | 5.1 / 6.9 / 13.0 | 14.0% |
| TS3003b | CAMPPlus | **post-cluster** | **4** | **14.1%** | 10.0% | 5.1 / 6.9 / 2.1 | 14.1% |
| TS3003b | WeSpeaker | **post-cluster** | **4** | **13.8%** | 9.8% | 5.1 / 6.9 / 1.9 | 13.8% |
| EN2002a | CAMPPlus | raw th 0.7 | 60 | 45.5% | 40.4% | 12.2 / 5.8 / 27.5 | 28.5% |
| EN2002a | CAMPPlus | + time filter | 6 | 41.7% | 36.7% | 12.2 / 5.8 / 27.5 | 28.5% |
| EN2002a | CAMPPlus | **post-cluster** | **4** | **28.9%** | 24.2% | 12.2 / 5.8 / 11.0 | 28.9% |
| EN2002a | WeSpeaker | post-cluster | 5 | 32.5% | 27.5% | 12.2 / 5.8 / 14.5 | 29.3% |
| **mean** | **CAMPPlus** | **post-cluster** | | **18.6%** | | | |
| **mean** | **WeSpeaker** | **post-cluster** | | **20.2%** | | | |

### Speaker count

| Model | ES2004b | IS1009b | TS3003b | EN2002a | Exact |
|---|---|---|---|---|---|
| CAMPPlus | 4 | 4 | 4 | 4 | **4/4** |
| WeSpeaker | 4 | 5 | 4 | 5 | 2/4 |
| reference | 4 | 4 | 4 | 4 | |

Reference count from the RTTM; the pipeline is never told it. Before
post-clustering the raw counts were 28-60, and task #2's time filter got 5-6.

### Oracle mapping on the raw clusters

| Model | Meeting | Raw DER (clusters) | Raw oracle | Final DER (clusters) | Final oracle |
|---|---|---|---|---|---|
| CAMPPlus | ES2004b | 22.4% (30) | 16.6% | **16.5% (4)** | 16.5% |
| CAMPPlus | IS1009b | 28.0% (35) | 15.3% | **14.8% (4)** | 14.8% |
| CAMPPlus | TS3003b | 25.0% (28) | 14.0% | **14.1% (4)** | 14.1% |
| CAMPPlus | EN2002a | 45.5% (60) | 28.5% | **28.9% (4)** | 28.9% |
| WeSpeaker | ES2004b | 22.4% (30) | 16.6% | 16.6% (4) | 16.6% |
| WeSpeaker | IS1009b | 28.0% (35) | 15.3% | 17.8% (5) | 15.9% |
| WeSpeaker | TS3003b | 25.0% (28) | 14.0% | 13.8% (4) | 13.8% |
| WeSpeaker | EN2002a | 45.5% (60) | 28.5% | 32.5% (5) | 29.3% |

On CAMPPlus the final DER is within **0.5 points** of the raw-cluster oracle on
all four meetings. It is *below* that oracle on ES2004b (16.5 vs 16.6) and
IS1009b (14.8 vs 15.3), because reassigning individual turns by voice is
finer-grained than mapping whole clusters; that also shows the oracle is not a
lower bound. TS3003b is 0.1 point above it, EN2002a 0.4. The clustering failure
the first review identified is no longer the dominant error on these meetings.

The row for the *final* output in the raw-oracle table is not independent
evidence: with four clusters mapping to four distinct reference speakers, the
many-to-one mapping is just the 1:1 mapping, so final DER and final oracle are
the same number. Only the raw-oracle comparison says something about how much
clustering error is left.

### Sensitivity around the frozen point

Re-scoring cached embeddings, reporting only (this was not used to tune the
frozen value, and no model was re-run). Each cell is DER at `min_share` 2/3/4%;
`(5)` marks a five-speaker output, `(2 spk)` a collapse.

| Meeting | 0.10 | 0.15 | 0.20 | 0.25 |
|---|---|---|---|---|
| ES2004b | 18.7 (5) / 16.5 / 16.5 | 16.5 x3 | 16.5 x3 | 16.8 |
| IS1009b | 14.8 | 14.8 | 15.0 | 15.0 |
| TS3003b | 14.1 | 14.1 | 14.1 | 14.3 |
| EN2002a | 31.7 (5) / 31.7 (5) / 28.8 | 28.9 | 29.0 | **53.1 (2 spk)** |
| ES2011c (dev) | 24.8 (5) / 23.0 / 23.0 | 23.9 (5) / 22.9 / 22.9 | 23.1 | **57.2 (2 spk)** |

Moving `merge_distance` by +/-0.05 and `min_share` by +/-1 point moves the test
mean by at most about **+1.1 points** (0.10 / 2%). Below 0.15 the method fails
gently, with one extra speaker; at 0.25 it fails hard, collapsing EN2002a and
the dev ES2011c to 2 speakers. 0.15 keeps a 0.10 margin on the hard side.

### Why the model comparison uses a shared partition

`sherpa-onnx`'s `threshold` is a cosine dissimilarity in the model's own
embedding space, and the two models do not share a scale. At the frozen 0.7,
CAMPPlus over-clusters to 24-60 clusters, but WeSpeaker ResNet34 already
collapses to **6 clusters** on IS1008b (42.0% DER with the dev reference): its
distances are compressed, so 0.7 is far past the merge point. Re-tuning the
over-clustering threshold per model would need a full diarization run per model
per meeting (25 minutes each on this host), so the comparison holds the
partition fixed instead: both models start from the same CAMPPlus turns at 0.7
and only the embeddings used by the merge and reassign steps differ. That
isolates the question the task asks -- which embedding recovers the speaker
structure better from the same over-clustered input -- and it is not a test of
a full WeSpeaker pipeline.

The consequence for the engine: 0.7 is a safe default **for CAMPPlus only**. If
the engine ever swaps in another embedding model it must re-tune (or estimate)
the over-clustering threshold; the merge distance does not transfer either, but
0.15/3% happens to be the dev optimum for both models.

### What is left

- **EN2002a at 28.9%** is the outlier and its raw-cluster oracle is 28.5%, so
  little of it is clustering. Its reference has 27.4% overlapped speech and the
  pipeline labels 39.5 min of speech where the `only_words` reference has
  31.6 min: backchannels and vocal sounds that the reference excludes become
  false alarms, and overlapped speech becomes misses (12.2%) and confusion
  (11.0%). It also has the worst raw clustering of the four (60 clusters,
  54-69% purity on the top clusters).
- **The +/-0.125 s collar removes 4-5 points** on every meeting (16.5 -> 12.2,
  14.8 -> 10.3, 14.1 -> 10.0, 28.9 -> 24.2). At the usual **+/-0.25 s**
  (pyannote `collar=0.5`) the same runs score **ES2004b 9.2%, IS1009b 7.3%,
  TS3003b 7.5%, EN2002a 20.8%**. That is boundary jitter from the segmenter's
  1 s hop, not a speaker error.
- **False alarm stays at 3-7%** because the segmenter labels non-word vocal
  sounds as speech and the `only_words` reference does not.
- **WeSpeaker leaves 5.9%/7.8% clusters on IS1009b/EN2002a.** Those are splits
  of a real speaker at a distance its compressed space does not reach, and the
  dev meetings have no cluster in the 3-9% band to tune `min_share` against, so
  the frozen 3% keeps them. Raising `min_share` for WeSpeaker on test would be
  tuning on test; the honest result is the table above -- WeSpeaker is not
  shown to be better (2 of 4 tie), and its embedding pass costs 82-133 CPU-s
  per meeting against 30-62 for CAMPPlus.
- **The 3% rule has a real risk: it erases a genuine speaker who holds less
  than 3% of the diarized segment time.** Any speaker under that share is
  dissolved into a neighbour by embedding, however distinct their voice. The
  AMI meetings here all have four roughly balanced speakers and do not test
  this. A user-facing merge/split/rename override is still required for
  meetings with quiet participants.

## For the engine

The engine port (`crates/tinta-engine-rs`, task #9) should carry exactly this:

1. **`recluster` with two settings**: centroid merge at `merge_distance = 0.15`
   plus dissolve/reassign at `min_share = 0.03`. No loyalty guard, no
   protect-share guard; they are absent from the code now. Make 0.15/3% the
   API defaults, not an opt-in.
2. **The unit builder from `main.rs`**, which is part of the recipe:
   - join segments of the same cluster with gaps of 0.5 s or less, capped at
     15 s;
   - pad units shorter than 0.8 s symmetrically before embedding;
   - compute one embedding per unit.
   The thresholds were tuned on these units, not on raw segments.
3. **State the coupling in the API docs**: 0.7 over-clustering and 0.15/3% are
   valid **for CAMPPlus only**. Any other embedding model needs its own
   over-clustering threshold and its own merge distance.
4. **Hardening**:
   - keep a distance matrix between rounds instead of recomputing all pairs
     (done here);
   - make NaN-safe comparisons and never panic on a non-finite embedding
     (done here);
   - send failed-embedding units to the largest cluster (done here);
   - log, and ideally expose to the UI, when the dissolve drops a large share
     of the speech or collapses many clusters to one, so the merge/split/rename
     override can be offered (not implemented);
   - compute `min_share` on non-overlapped time, or document that it is a share
     of summed unit durations with overlap counted twice (documented here).
5. **Per-turn embeddings**: the high-level sherpa diarization API does not
   expose them. Plan either an extra embedding pass, measured at +14-33% of
   diarization CPU, or a lower-level pipeline that keeps the per-turn
   embeddings.

## Reproduction

Everything runs from the worktree; data, models and results stay in `$ROOT`.

```bash
export ROOT=/home/sergio/code/evidence/tinta-fork/18
# once: python deps (reuses task #2's venv)
python3 -m venv "$ROOT/venv"
"$ROOT/venv/bin/pip" install numpy scipy rapidfuzz pyannote.metrics whisper-normalizer

# data + models, every SHA-256 checked (also fetches the RTTMs and both embeddings)
ROOT="$ROOT" scripts/fork/bench/fetch-data.sh

cd scripts/fork/bench && cargo build --release -j2 && cargo test --release -j2 && cd -

# 1. over-cluster every meeting with both models, then cache one embedding per turn
scripts/fork/bench/diarization.sh raw

# 2. sweep the settings on the dev meetings only
#    (merge 0.10-0.30 in steps of 0.05, min_share 2/3/5%)
scripts/fork/bench/diarization.sh sweep

# 3. frozen settings on the test meetings (4 runs per model; the script and the
#    code defaults are already 0.15 / 3%)
scripts/fork/bench/diarization.sh test

# 4. oracle mapping, raw clusters and final output
scripts/fork/bench/diarization.sh oracle

# 5. the tables above
scripts/fork/bench/diarization.sh tables
```

One run by hand:

```bash
nice -n 19 scripts/fork/bench/target/release/tinta-bench diarize \
  --wav "$ROOT/data/ES2004b.Mix-Headset.wav" \
  --seg-model "$ROOT/models/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx" \
  --embedding "$ROOT/models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx" \
  --threads 2 --spin 0 --threshold 0.7 --window-shift 0.2 \
  --output "$ROOT/results/diar-ES2004b-campplus.json"

nice -n 19 scripts/fork/bench/target/release/tinta-bench postcluster \
  --wav "$ROOT/data/ES2004b.Mix-Headset.wav" \
  --diar "$ROOT/results/diar-ES2004b-campplus.json" \
  --embedding "$ROOT/models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx" \
  --threads 2 --spin 0 --merge-threshold 0.15 --min-share 0.03 \
  --embeddings "$ROOT/results/emb-ES2004b-campplus.json" \
  --output "$ROOT/results/pc-ES2004b-campplus-m0.15-s0.03.json"

"$ROOT/venv/bin/python" scripts/fork/bench/score.py score \
  --ref "$ROOT/results/ref-ES2004b.json" \
  --result "$ROOT/results/pc-ES2004b-campplus-m0.15-s0.03.json" \
  --small-cluster-fraction 0 \
  --out "$ROOT/results/score-pc-ES2004b.json"

"$ROOT/venv/bin/python" scripts/fork/bench/score.py oracle \
  --ref "$ROOT/results/ref-ES2004b.json" \
  --result "$ROOT/results/diar-ES2004b-campplus.json" \
  --out "$ROOT/results/oracle-raw-ES2004b.json"
```

The reference JSON is built from the pinned RTTM with
`score.py ref-rttm --rttm ... --uem ... --meeting ... --out ...`; step 1 of
`diarization.sh` (`build_refs`) does it for every meeting from the setup
directory.

The golden fixtures in `scripts/fork/bench/tests/data/` are generated from the
cached embeddings with
`ROOT=... python3 scripts/fork/bench/make_golden_fixtures.py`; they are about
3-5 KB each, contain no audio, one selected unit embedding per row and the
reference speaker of that unit. The reference tag is used only to check that
the pinned assignment is one label per speaker and never by the algorithm.
`cargo test` pins the speaker counts, the exact label sequence and that
partition.

## Remaining gap and what to do next

1. **Clustering is no longer the main lever on these four meetings.** The final
   DER is within 0.5 points of the raw-cluster oracle. Raising it further needs
   a better front end: the segmenter's boundary precision (the +/-0.125 s
   collar is worth 4-5 points) and its false alarms on non-word vocal sounds
   (3-7%).
2. **EN2002a is a front-end/reference problem.** A meeting with 27% overlap and
   many backchannels is scored against `only_words`, so the ceiling is about
   28%. Do not tune the clustering on it; report it separately.
3. **Five or more speakers, and recordings with a long silence, are untested.**
   The AMI meetings here all have four. The algorithm has no speaker-count
   prior, and `min_share` is a share of *speech*, so a real speaker with less
   than 3% of the speech is the first thing it dissolves -- the user override
   (merge/split/rename) still matters.
4. **The engine must expose the per-turn embeddings** so that step 3 costs no
   extra pass; with sherpa-onnx's high-level diarization API it costs one extra
   embedding pass (+14-33% of the stage, 30-62 CPU-s per meeting).

## Files

| Path | Purpose |
|---|---|
| `scripts/fork/bench/src/clustering.rs` | the post-clustering function + unit tests |
| `scripts/fork/bench/src/main.rs` | new `postcluster` subcommand (embeddings, units, merge) |
| `scripts/fork/bench/tests/data/golden-*.json` | small cached-embedding extracts pinned by tests |
| `scripts/fork/bench/make_golden_fixtures.py` | regenerates the golden extracts from the cache |
| `scripts/fork/bench/score.py` | `ref-rttm`, `oracle`, `--uem` |
| `scripts/fork/bench/sweep_diar.py` | dev-only settings sweep on cached embeddings |
| `scripts/fork/bench/diarization.sh` | raw runs, sweep, test runs, oracle |
| `scripts/fork/bench/diarization_tables.py` | the tables in this document |
| `scripts/fork/bench/fetch-data.sh` | new model + RTTM pins |

Results, embeddings caches, logs and audio live in
`/home/sergio/code/evidence/tinta-fork/18/`, never in the repo.

## Limitations

- Four of the sixteen test meetings, and two dev meetings. The host is shared
  with another project and one diarization run takes 10-30 minutes of wall time
  at `nice 19` and two threads; more meetings would tighten the mean and
  exercise the 3% risk, not change the clustering conclusion on these four.
- The reference is `only_words`, so non-word vocal sounds count as false alarms.
  That inflates DER by a few points on the meeting with the most backchannels
  (EN2002a). The same reference is used for every model and every variant.
- Published pyannote 3.1 (about 18.8% on AMI headset-mix) and FluidAudio figures
  are quoted from memory and are not measured here.
