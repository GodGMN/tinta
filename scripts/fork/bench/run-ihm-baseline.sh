#!/usr/bin/env bash
# AMI IHM headset baseline for the sherpa-onnx Parakeet TDT 0.6b v3 int8
# runtime, plus the same-utterance Mix-Headset comparison, the full-split
# decode, and the same-reference oracle-vs-VAD ablation (task #19).
# See docs/fork/asr-baseline.md. Everything runs at nice 19 with 2 threads.
#
#   export ROOT=/path/to/evidence/tinta-fork/2
#   export ROOT19=/path/to/evidence/tinta-fork/19
#   scripts/fork/bench/run-ihm-baseline.sh
#
# Every decode is resume-safe: re-running skips the ids already in the output
# JSONL, so a crash never loses work. The full split is about 8.7 h of audio.
set -euo pipefail

ROOT="${ROOT:?export ROOT to the task-2 evidence directory}"
ROOT19="${ROOT19:?export ROOT19 to the task-19 evidence directory}"
THREADS="${THREADS:-2}"
BENCH="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$ROOT/venv/bin/python}"
MODELS="${MODELS:-$ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8}"
VAD_MODEL="${VAD_MODEL:-$ROOT/models/silero_vad.onnx}"
BIN="${BIN:-$BENCH/target/release/tinta-bench}"
DATA="$ROOT19/data/ihm-baseline"
DATA_FULL="$ROOT19/data/ihm-baseline-full"
ABL="$ROOT19/data/ablation"
RESULTS="$ROOT19/results"
MEETINGS="${MEETINGS:-ES2004b,IS1009b,EN2002a}"
# The reviewed ES2004b full-meeting pipeline decode predates this script.
ES2004B_VAD="${ES2004B_VAD:-$ROOT/results/asr-vad-ES2004b-t8s0.json}"

[ -x "$PY" ] || { echo "missing scoring venv: $PY" >&2; exit 1; }
[ -d "$ROOT19/data/ihm-test" ] || {
  echo "missing IHM parquet: run scripts/fork/bench/fetch-data.sh first" >&2
  exit 1
}
if [ ! -x "$BIN" ]; then
  ( cd "$BENCH" && nice -n 19 cargo build --release -j2 )
fi

mkdir -p "$RESULTS" "$ROOT19/logs" "$DATA" "$DATA_FULL" "$ABL"

# References for every meeting the ablation and the crosstalk buckets use.
for m in ${MEETINGS//,/ }; do
  [ -s "$ROOT/data/$m.ref.json" ] && continue
  nice -n 19 "$PY" "$BENCH/score.py" ref \
    --annotations "$ROOT/data/ami_annotations" --meeting "$m" \
    --out "$ROOT/data/$m.ref.json"
done

batch() { # manifest, output, log
  local manifest="$1" output="$2" log="$3"
  nice -n 19 "$BIN" batch \
    --manifest "$manifest" \
    --output "$output" \
    --encoder "$MODELS/encoder.int8.onnx" \
    --decoder "$MODELS/decoder.int8.onnx" \
    --joiner "$MODELS/joiner.int8.onnx" \
    --tokens "$MODELS/tokens.txt" \
    --threads "$THREADS" --spin 0 2>&1 | tee "$log"
}

vad_decode() { # meeting, output
  local m="$1" output="$2"
  [ -s "$output" ] && return 0
  nice -n 19 "$BIN" asr \
    --wav "$ROOT/data/$m.Mix-Headset.wav" \
    --vad-model "$VAD_MODEL" \
    --encoder "$MODELS/encoder.int8.onnx" \
    --decoder "$MODELS/decoder.int8.onnx" \
    --joiner "$MODELS/joiner.int8.onnx" \
    --tokens "$MODELS/tokens.txt" \
    --threads "$THREADS" --spin 0 \
    --output "$output" 2>&1 | tee "$ROOT19/logs/vad-$m-t$THREADS.log"
}

# 1. Manifests: the deterministic sample, the full split, the oracle segments.
nice -n 19 "$PY" "$BENCH/ihm-baseline.py" prepare \
  --out-dir "$DATA" 2>&1 | tee "$ROOT19/logs/prepare-sample.log"
nice -n 19 "$PY" "$BENCH/ihm-baseline.py" prepare --full \
  --out-dir "$DATA_FULL" 2>&1 | tee "$ROOT19/logs/prepare-full.log"
nice -n 19 "$PY" "$BENCH/ihm-baseline.py" ablate-prepare \
  --out-dir "$ABL" --meetings "$MEETINGS" 2>&1 | tee "$ROOT19/logs/ablate-prepare.log"

# 2. IHM sample: every HF utterance decoded exactly as the dataset defines it.
batch "$DATA/manifest-ihm.tsv" "$RESULTS/ihm-decode.jsonl" "$ROOT19/logs/ihm-decode.log"
# Mix: the same words, i.e. the same [begin, end) ranges cut from Mix-Headset.
batch "$DATA/manifest-mix.tsv" "$RESULTS/mix-decode.jsonl" "$ROOT19/logs/mix-decode.log"

# 3. Full split. Seed the resume file with the sample decode, then decode the
#    missing rows; a crash can simply be re-run.
if [ -f "$RESULTS/ihm-decode.jsonl" ] && [ ! -f "$RESULTS/ihm-full-decode.jsonl" ]; then
  cp "$RESULTS/ihm-decode.jsonl" "$RESULTS/ihm-full-decode.jsonl"
fi
batch "$DATA_FULL/manifest-ihm.tsv" "$RESULTS/ihm-full-decode.jsonl" \
  "$ROOT19/logs/ihm-full-decode.log"

# 4. Oracle-segment decodes on Mix. The two paired meetings reuse the sample
#    Mix decode above; EN2002a has no Mix rows in the sample, so decode its
#    segments from the meeting wav.
for m in ${MEETINGS//,/ }; do
  case "$m" in
    ES2004b|IS1009b) ;;
    *) batch "$ABL/manifest-$m.tsv" "$RESULTS/ablation-$m.jsonl" \
         "$ROOT19/logs/ablation-$m.log" ;;
  esac
done

# 5. Full-meeting engine VAD decodes for the ablation. ES2004b is the reviewed
#    full-meeting run; decode the other two at 2 threads.
if [ ! -s "$ES2004B_VAD" ]; then
  ES2004B_VAD="$RESULTS/vad-ES2004b-t$THREADS.json"
  vad_decode ES2004b "$ES2004B_VAD"
fi
vad_decode IS1009b "$RESULTS/vad-IS1009b-t$THREADS.json"
vad_decode EN2002a "$RESULTS/vad-EN2002a-t$THREADS.json"

# 6. Scores. Sample, then the full split, then the weighted cross-check.
nice -n 19 "$PY" "$BENCH/ihm-baseline.py" score \
  --out "$RESULTS/ihm-summary.json" | tee "$RESULTS/ihm-tables.md"
nice -n 19 "$PY" "$BENCH/ihm-baseline.py" score \
  --rows "$DATA_FULL/rows.json" \
  --ihm-jsonl "$RESULTS/ihm-full-decode.jsonl" \
  --out "$RESULTS/ihm-full-summary.json" | tee "$RESULTS/ihm-full-tables.md"
nice -n 19 "$PY" "$BENCH/ihm-baseline.py" weighted \
  --out "$RESULTS/ihm-weighted-estimate.json" | tee "$RESULTS/ihm-weighted-estimate.md"

# 7. Same-reference ablation: oracle segments vs the pipeline, same HF text.
nice -n 19 "$PY" - "$RESULTS" <<'PY'
import json, sys
r = sys.argv[1]
a = json.load(open(f"{r}/ablation-oracle-score.json"))
b = json.load(open(f"{r}/ablation-vad-score.json"))
json.dump({"oracle": a, "vad": b}, open(f"{r}/ablation-window-300-600.json", "w"), indent=1)
PY
vad_args="ES2004b=$ES2004B_VAD,IS1009b=$RESULTS/vad-IS1009b-t$THREADS.json"
vad_args="$vad_args,EN2002a=$RESULTS/vad-EN2002a-t$THREADS.json"
oracle_args="ES2004b=$RESULTS/mix-decode.jsonl,IS1009b=$RESULTS/mix-decode.jsonl"
oracle_args="$oracle_args,EN2002a=$RESULTS/ablation-EN2002a.jsonl"
nice -n 19 "$PY" "$BENCH/ihm-baseline.py" ablate-score \
  --meetings "$MEETINGS" --oracle "$oracle_args" --vad "$vad_args" \
  --window-300-600 "$RESULTS/ablation-window-300-600.json" \
  --out "$RESULTS/ablation-summary.json" | tee "$RESULTS/ablation-tables.md"

# 8. Cause analysis: deletions in long hypothesis holes at 20 s flush starts.
nice -n 19 "$PY" "$BENCH/ihm-baseline.py" cause \
  --meetings "$MEETINGS" --vad "$vad_args" \
  --out "$RESULTS/cause-analysis.json" | tee "$RESULTS/cause-analysis.md"
