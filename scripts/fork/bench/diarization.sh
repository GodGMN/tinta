#!/usr/bin/env bash
# Diarization post-clustering experiment (task 18).
#
#   diarization.sh raw     # over-cluster + per-unit embeddings, every meeting x model
#   diarization.sh sweep   # settings sweep on the AMI dev meetings
#   diarization.sh test    # evaluate the frozen settings on the AMI test list
#   diarization.sh oracle  # oracle mapping bound, raw and final, for every run
#   diarization.sh tables  # markdown tables from the scored JSON
#
# References are the BUTSpeechFIT AMI-diarization-setup `only_words` RTTMs,
# which is what pyannote and FluidAudio publish AMI DER against. The test list
# is that setup's official test list; four of its meetings are used here.
#
# Data and models live outside the repo; see docs/fork/diarization.md.
set -euo pipefail

ROOT="${ROOT:-/home/sergio/code/evidence/tinta-fork/18}"
BENCH="$(cd "$(dirname "$0")/../../.." && pwd)"
BIN="$BENCH/scripts/fork/bench/target/release/tinta-bench"
PY="${PY:-$ROOT/venv/bin/python}"
SCORE="$BENCH/scripts/fork/bench/score.py"
THREADS="${THREADS:-2}"

DEV="${DEV:-IS1008b ES2011c}"
TEST="${TEST:-ES2004b IS1009b TS3003b EN2002a}"
MODELS="${MODELS:-campplus wespeaker}"

SEG="$ROOT/models/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx"
SETUP="$ROOT/data/ami-diarization-setup"
CAMPPLUS="$ROOT/models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
WESPEAKER="$ROOT/models/wespeaker_en_voxceleb_resnet34.onnx"

embedding_for() {
  case "$1" in
    campplus) echo "$CAMPPLUS" ;;
    wespeaker) echo "$WESPEAKER" ;;
    *) echo "unknown model $1" >&2; exit 1 ;;
  esac
}

# The over-clustered partition every model starts from. CAMPPlus at threshold
# 0.7 over-clusters to 24-60 clusters; the same threshold with WeSpeaker already
# merges to 6, so the model comparison holds the partition fixed and varies only
# the embeddings (see docs/fork/diarization.md).
PARTITION_TAG="${PARTITION_TAG:-campplus}"

# Frozen after the dev sweep; see docs/fork/diarization.md.
MERGE="${MERGE:-0.15}"
LOYALTY="${LOYALTY:-0}"
SHARE="${SHARE:-0.03}"

# Reference JSON from the pinned BUT RTTM. The setup's own test/dev lists decide
# which directory a meeting lives in.
build_refs() {
  local meeting dir
  for meeting in $DEV $TEST; do
    dir=dev
    [ -f "$SETUP/only_words/rttms/test/$meeting.rttm" ] && dir=test
    [ -s "$ROOT/results/ref-$meeting.json" ] && continue
    "$PY" "$SCORE" ref-rttm \
      --rttm "$SETUP/only_words/rttms/$dir/$meeting.rttm" \
      --uem "$SETUP/uems/$dir/$meeting.uem" \
      --meeting "$meeting" --out "$ROOT/results/ref-$meeting.json"
  done
}

partition_for() { # meeting
  echo "$ROOT/results/diar-$1-$PARTITION_TAG.json"
}

run_raw() {
  mkdir -p "$ROOT/results" "$ROOT/logs"
  build_refs
  for model in $MODELS; do
    for meeting in $DEV $TEST; do
      local wav="$ROOT/data/$meeting.Mix-Headset.wav"
      local raw
      raw="$(partition_for "$meeting")"
      local emb="$ROOT/results/emb-$meeting-$model.json"
      [ -s "$wav" ] || { echo "missing $wav" >&2; exit 1; }
      if [ "$model" = "$PARTITION_TAG" ] && [ ! -s "$raw" ]; then
        echo "diarize $meeting ($model)"
        nice -n 19 "$BIN" diarize \
          --wav "$wav" \
          --seg-model "$SEG" \
          --embedding "$(embedding_for "$model")" \
          --threads "$THREADS" --threshold 0.7 --window-shift 0.2 --spin 0 \
          --output "$raw" >"$ROOT/logs/diar-$meeting-$model.out" 2>&1
      fi
      [ -s "$raw" ] || { echo "missing partition $raw" >&2; exit 1; }
      if [ ! -s "$emb" ]; then
        echo "embeddings $meeting ($model)"
        nice -n 19 "$BIN" postcluster \
          --wav "$wav" --diar "$raw" \
          --embedding "$(embedding_for "$model")" \
          --threads "$THREADS" --spin 0 \
          --merge-threshold "$MERGE" --min-share "$SHARE" --min-loyalty "$LOYALTY" \
          --embeddings "$emb" \
          --output "$ROOT/results/pc/bootstrap-$meeting-$model.json" \
          >"$ROOT/logs/emb-$meeting-$model.out" 2>&1
      fi
    done
  done
}

sweep() {
  # Tuning only touches the dev meetings.
  "$PY" "$BENCH/scripts/fork/bench/sweep_diar.py" \
    --meetings "$(echo $DEV | tr ' ' ',')" \
    --diar-tag "$PARTITION_TAG" \
    --merge 0.15,0.20,0.25,0.30,0.35 \
    --loyalty 0,0.3,0.5 \
    --min-share 0.0,0.01,0.02,0.03 \
    --table "$ROOT/results/sweep.md"
}

run_one() { # meeting model merge loyalty share
  local meeting="$1" model="$2" merge="$3" loyalty="$4" share="$5"
  local wav="$ROOT/data/$meeting.Mix-Headset.wav"
  local raw
  raw="$(partition_for "$meeting")"
  local emb="$ROOT/results/emb-$meeting-$model.json"
  local name="pc-$meeting-$model-m${merge}-l${loyalty}-s${share}"
  nice -n 19 "$BIN" postcluster \
    --wav "$wav" --diar "$raw" \
    --embedding "$(embedding_for "$model")" \
    --threads "$THREADS" --spin 0 \
    --merge-threshold "$merge" --min-share "$share" --min-loyalty "$loyalty" \
    --embeddings "$emb" \
    --output "$ROOT/results/$name.json" >"$ROOT/logs/$name.out" 2>&1
  "$PY" "$SCORE" score \
    --ref "$ROOT/results/ref-$meeting.json" \
    --result "$ROOT/results/$name.json" \
    --small-cluster-fraction 0 \
    --out "$ROOT/results/score-$name.json" >/dev/null
  echo "$name"
}

run_test() {
  for model in $MODELS; do
    for meeting in $TEST; do run_one "$meeting" "$model" "$MERGE" "$LOYALTY" "$SHARE"; done
  done
}

run_oracle() {
  for model in $MODELS; do
    for meeting in $TEST; do
      local raw
      raw="$(partition_for "$meeting")"
      local fin="$ROOT/results/pc-$meeting-$model-m${MERGE}-l${LOYALTY}-s${SHARE}.json"
      "$PY" "$SCORE" oracle --ref "$ROOT/results/ref-$meeting.json" \
        --result "$raw" --out "$ROOT/results/oracle-raw-$meeting-$model.json" >/dev/null
      "$PY" "$SCORE" oracle --ref "$ROOT/results/ref-$meeting.json" \
        --result "$fin" --out "$ROOT/results/oracle-fin-$meeting-$model.json" >/dev/null
    done
  done
}

tables() {
  "$PY" "$BENCH/scripts/fork/bench/diarization_tables.py" \
    --merge "$MERGE" --loyalty "$LOYALTY" --share "$SHARE" \
    --models "$(echo $MODELS | tr ' ' ',')" \
    --meetings "$(echo $TEST | tr ' ' ',')" \
    --out "$ROOT/results/tables.md"
}

case "${1:-}" in
  raw) run_raw ;;
  sweep) sweep ;;
  test) run_test ;;
  oracle) run_oracle ;;
  tables) tables ;;
  *) echo "usage: $0 {raw|sweep|test|oracle|tables}" >&2; exit 2 ;;
esac
