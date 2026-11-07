#!/usr/bin/env bash
# AMI IHM headset baseline for the sherpa-onnx Parakeet TDT 0.6b v3 int8
# runtime, plus the same-utterance Mix-Headset comparison (task #19).
# See docs/fork/asr-baseline.md. Everything runs at nice 19 with 2 threads.
#
#   ROOT=/home/sergio/code/evidence/tinta-fork/2 \
#   ROOT19=/home/sergio/code/evidence/tinta-fork/19 \
#   scripts/fork/bench/run-ihm-baseline.sh
set -euo pipefail

ROOT="${ROOT:-/home/sergio/code/evidence/tinta-fork/2}"
ROOT19="${ROOT19:-/home/sergio/code/evidence/tinta-fork/19}"
THREADS="${THREADS:-2}"
BENCH="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$ROOT/venv/bin/python}"
MODELS="${MODELS:-$ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8}"
BIN="${BIN:-$BENCH/target/release/tinta-bench}"
DATA="$ROOT19/data/ihm-baseline"
RESULTS="$ROOT19/results"

[ -x "$PY" ] || { echo "missing scoring venv: $PY" >&2; exit 1; }
[ -d "$ROOT19/data/ihm-test" ] || {
  echo "missing IHM parquet: run scripts/fork/bench/fetch-data.sh first" >&2
  exit 1
}
[ -x "$BIN" ] || (cd "$BENCH" && nice -n 19 cargo build --release -j2)

mkdir -p "$RESULTS" "$ROOT19/logs"
"$PY" "$BENCH/ihm-baseline.py" prepare

# IHM: each HF utterance decoded exactly as the dataset defines it (no VAD).
nice -n 19 "$BIN" batch \
  --manifest "$DATA/manifest-ihm.tsv" \
  --output "$RESULTS/ihm-decode.jsonl" \
  --encoder "$MODELS/encoder.int8.onnx" \
  --decoder "$MODELS/decoder.int8.onnx" \
  --joiner "$MODELS/joiner.int8.onnx" \
  --tokens "$MODELS/tokens.txt" \
  --threads "$THREADS" --spin 0 2>&1 | tee "$ROOT19/logs/ihm-decode.log"

# Mix: the same words, i.e. the same [begin, end) ranges cut from Mix-Headset.
nice -n 19 "$BIN" batch \
  --manifest "$DATA/manifest-mix.tsv" \
  --output "$RESULTS/mix-decode.jsonl" \
  --encoder "$MODELS/encoder.int8.onnx" \
  --decoder "$MODELS/decoder.int8.onnx" \
  --joiner "$MODELS/joiner.int8.onnx" \
  --tokens "$MODELS/tokens.txt" \
  --threads "$THREADS" --spin 0 2>&1 | tee "$ROOT19/logs/mix-decode.log"

"$PY" "$BENCH/ihm-baseline.py" score | tee "$RESULTS/ihm-tables.md"
