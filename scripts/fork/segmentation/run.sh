#!/usr/bin/env bash
# Task #21: measure the engine's own segmentation on the three ablation meetings.
# See docs/fork/segmentation.md. Every heavy step runs at nice 19 with 2 threads.
#
#   export ROOT=/path/to/evidence/tinta-fork/2      # models and meeting wavs
#   export ROOT19=/path/to/evidence/tinta-fork/19   # rows, oracle decodes
#   export SEG=/path/to/evidence/tinta-fork/21      # results for this task
#   scripts/fork/segmentation/run.sh
#
# Re-running skips a (variant, meeting) whose JSON already exists, so a crash
# never loses a finished decode.
set -euo pipefail

ROOT="${ROOT:?export ROOT to the task-2 evidence directory}"
ROOT19="${ROOT19:?export ROOT19 to the task-19 evidence directory}"
SEG="${SEG:?export SEG to the task-21 evidence directory}"
THREADS="${THREADS:-2}"
MEETINGS="${MEETINGS:-ES2004b,IS1009b,EN2002a}"
VARIANTS="${VARIANTS:-before,no-vad-cut,short,short-gap,mid-gap,runs,default,pre-roll-floor0,pre-roll-floor10}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$ROOT/venv/bin/python}"
BIN="${BIN:-$HERE/target/release/tinta-segmentation}"
MODELS="$ROOT/models"
ROOT21="$SEG/models"
RESULTS="$SEG/results"
LOGS="$SEG/logs"

mkdir -p "$RESULTS" "$LOGS"

# The engine reads the manifest's layout under one models root. Link the task-2
# files into it so the harness exercises `Manifest::embedded` unchanged.
link() { # target, link
  mkdir -p "$(dirname "$2")"
  [ -e "$2" ] || ln -s "$1" "$2"
}
link "$MODELS/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx" \
  "$ROOT21/parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx"
link "$MODELS/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx" \
  "$ROOT21/parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx"
link "$MODELS/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx" \
  "$ROOT21/parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx"
link "$MODELS/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt" \
  "$ROOT21/parakeet-tdt-0.6b-v3-int8/tokens.txt"
link "$MODELS/silero_vad.onnx" "$ROOT21/silero-vad/silero_vad.onnx"
link "$MODELS/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx" \
  "$ROOT21/pyannote-segmentation-3-0/model.int8.onnx"
link "$MODELS/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx" \
  "$ROOT21/3dspeaker-campplus/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"

if [ ! -x "$BIN" ]; then
  ( cd "$HERE" && nice -n 19 cargo build --release -j2 )
fi

# The oracle side is the task #19 decode of the dataset's own segments on
# Mix-Headset: the two paired meetings are in the Mix sample decode, EN2002a in
# its own ablation decode. The reference is the full-split rows.json.
ORACLE="ES2004b=$ROOT19/results/mix-decode.jsonl"
ORACLE="$ORACLE,IS1009b=$ROOT19/results/mix-decode.jsonl"
ORACLE="$ORACLE,EN2002a=$ROOT19/results/ablation-EN2002a.jsonl"
ROWS="$ROOT19/data/ihm-baseline-full/rows.json"

for variant in ${VARIANTS//,/ }; do
  vad_args=""
  for meeting in ${MEETINGS//,/ }; do
    output="$RESULTS/$variant-$meeting.json"
    if [ ! -s "$output" ]; then
      nice -n 19 "$BIN" \
        --wav "$ROOT/data/$meeting.Mix-Headset.wav" \
        --models-root "$ROOT21" \
        --variant "$variant" \
        --threads "$THREADS" \
        --output "$output" 2>&1 | tee "$LOGS/$variant-$meeting.log"
    fi
    vad_args="$vad_args${vad_args:+,}$meeting=$output"
  done
  nice -n 19 "$PY" "$HERE/score.py" \
    --rows "$ROWS" --oracle "$ORACLE" --pipeline "$vad_args" \
    --label "$variant" --out "$RESULTS/score-$variant.json" \
    | tee -a "$RESULTS/segmentation-tables.md"
done
