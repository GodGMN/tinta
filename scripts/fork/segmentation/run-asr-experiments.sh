#!/usr/bin/env bash
# Task #23: ASR experiments on top of the task #21 segmentation harness.
#
#   export ROOT=/path/to/evidence/tinta-fork/2      # meeting wavs and int8 models
#   export ROOT19=/path/to/evidence/tinta-fork/19   # rows.json, oracle decodes
#   export EVID=/path/to/evidence/tinta-fork/23     # results for this task
#   export FP32_DEST=$EVID/models                   # fetch-model-fp32.sh target
#   scripts/fork/segmentation/fetch-model-fp32.sh "$FP32_DEST"
#   scripts/fork/segmentation/run-asr-experiments.sh
#
# Measures, with the engine's own decode path:
#   (a) the fp32 Parakeet export, with and without the hole re-decode;
#   (b) pieces split at diarized speaker changes inside >= 0.2 s silences;
#   and scores every run per utterance (score.py) and with meeteval
#   (score-meeteval.py: tcpWER collar 5 s, cpWER).
#
# Every heavy step runs at nice 19 with 2 threads. Decodes are resume-safe: an
# existing JSON is skipped.
set -euo pipefail

ROOT="${ROOT:?export ROOT to the task-2 evidence directory}"
ROOT19="${ROOT19:?export ROOT19 to the task-19 evidence directory}"
EVID="${EVID:?export EVID to the task-23 evidence directory}"
THREADS="${THREADS:-2}"
MEETINGS="${MEETINGS:-ES2004b,IS1009b,EN2002a}"
VARIANT="${VARIANT:-default}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$EVID/venv/bin/python}"        # needs rapidfuzz, whisper-normalizer, meeteval
BIN="${BIN:-$HERE/target/release/tinta-segmentation}"
INT8_MODELS="${INT8_MODELS:-$EVID/models/int8-root}"
FP32_MODELS="${FP32_MODELS:-$EVID/models/fp32-root}"
RESULTS="$EVID/results"
LOGS="$EVID/logs"
SEG21="${SEG21:-$ROOT19/../21}"          # task #21 results, for the int8 baseline

mkdir -p "$RESULTS" "$LOGS"

link() { mkdir -p "$(dirname "$2")"; [ -e "$2" ] || ln -s "$1" "$2"; }

# The manifest layout under each models root. Only the transcription files
# differ between the two roots; VAD and diarization are the same int8 models.
for root in "$INT8_MODELS" "$FP32_MODELS"; do
  link "$ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx" \
    "$root/parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx"
  link "$ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx" \
    "$root/parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx"
  link "$ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx" \
    "$root/parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx"
  link "$ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt" \
    "$root/parakeet-tdt-0.6b-v3-int8/tokens.txt"
  link "$ROOT/models/silero_vad.onnx" "$root/silero-vad/silero_vad.onnx"
  link "$ROOT/models/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx" \
    "$root/pyannote-segmentation-3-0/model.int8.onnx"
  link "$ROOT/models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx" \
    "$root/3dspeaker-campplus/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
done
# The fp32 root keeps its own transcription files; the links above must not
# overwrite them. fetch-model-fp32.sh already placed them, so re-link int8-only
# paths only when missing (the `link` helper never overwrites an existing file).
[ -s "$FP32_MODELS/parakeet-tdt-0.6b-v3-int8/encoder.weights" ] || {
  echo "missing the fp32 model: run scripts/fork/segmentation/fetch-model-fp32.sh $EVID/models" >&2
  exit 1
}

if [ ! -x "$BIN" ]; then
  ( cd "$HERE" && nice -n 19 cargo build --release -j2 )
fi

# Task #15 post-clustered turns, from the engine's own diarizer.
for meeting in ${MEETINGS//,/ }; do
  output="$RESULTS/turns-$meeting.json"
  [ -s "$output" ] || nice -n 19 "$BIN" \
    --wav "$ROOT/data/$meeting.Mix-Headset.wav" \
    --models-root "$INT8_MODELS" --threads "$THREADS" --diarize \
    --output "$output" 2>&1 | tee "$LOGS/turns-$meeting.log"
done

run() { # label, models root, extra args...
  local label="$1" root="$2"; shift 2
  local args=""
  for meeting in ${MEETINGS//,/ }; do
    local output="$RESULTS/$label-$meeting.json"
    if [ ! -s "$output" ]; then
      nice -n 19 "$BIN" --wav "$ROOT/data/$meeting.Mix-Headset.wav" \
        --models-root "$root" --threads "$THREADS" \
        --output "$output" "$@" 2>&1 | tee "$LOGS/$label-$meeting.log"
    fi
    args="$args${args:+,}$meeting=$output"
  done
  nice -n 19 "$PY" "$HERE/score.py" \
    --rows "$ROOT19/data/ihm-baseline-full/rows.json" \
    --oracle "ES2004b=$ROOT19/results/mix-decode.jsonl,IS1009b=$ROOT19/results/mix-decode.jsonl,EN2002a=$ROOT19/results/ablation-EN2002a.jsonl" \
    --pipeline "$args" --label "$label" --out "$RESULTS/score-$label.json" \
    | tee -a "$RESULTS/asr-experiment-tables.md"
  nice -n 19 "$PY" "$HERE/score-meeteval.py" \
    --rows "$ROOT19/data/ihm-baseline-full/rows.json" \
    --oracle "ES2004b=$ROOT19/results/mix-decode.jsonl,IS1009b=$ROOT19/results/mix-decode.jsonl,EN2002a=$ROOT19/results/ablation-EN2002a.jsonl" \
    --pipeline "$args" --label "$label" --out "$RESULTS/meeteval-$label.json" \
    | tee -a "$RESULTS/asr-experiment-tables.md"
}

# (a) fp32, without and with the hole re-decode. `runs` is the same first pass
# as `default` with gap_redecode_seconds = 0.
run fp32-runs "$FP32_MODELS" --variant runs
run fp32-default "$FP32_MODELS" --variant default

# The int8 baseline is the task #21 `default` decode. Re-run it when the cached
# files are not there, so the comparison always comes from this code path.
for meeting in ${MEETINGS//,/ }; do
  src="$SEG21/results/default-$meeting.json"
  [ -s "$RESULTS/default-$meeting.json" ] || cp "$src" "$RESULTS/default-$meeting.json"
done
nice -n 19 "$PY" "$HERE/score.py" \
  --rows "$ROOT19/data/ihm-baseline-full/rows.json" \
  --oracle "ES2004b=$ROOT19/results/mix-decode.jsonl,IS1009b=$ROOT19/results/mix-decode.jsonl,EN2002a=$ROOT19/results/ablation-EN2002a.jsonl" \
  --pipeline "ES2004b=$RESULTS/default-ES2004b.json,IS1009b=$RESULTS/default-IS1009b.json,EN2002a=$RESULTS/default-EN2002a.json" \
  --label default --out "$RESULTS/score-default.json" | tee -a "$RESULTS/asr-experiment-tables.md"
nice -n 19 "$PY" "$HERE/score-meeteval.py" \
  --rows "$ROOT19/data/ihm-baseline-full/rows.json" \
  --oracle "ES2004b=$ROOT19/results/mix-decode.jsonl,IS1009b=$ROOT19/results/mix-decode.jsonl,EN2002a=$ROOT19/results/ablation-EN2002a.jsonl" \
  --pipeline "ES2004b=$RESULTS/default-ES2004b.json,IS1009b=$RESULTS/default-IS1009b.json,EN2002a=$RESULTS/default-EN2002a.json" \
  --label default --out "$RESULTS/meeteval-default.json" | tee -a "$RESULTS/asr-experiment-tables.md"

# (b) speaker splits. The turns come from the engine's diarizer above; each
# meeting uses its own turns file.
for meeting in ${MEETINGS//,/ }; do
  [ -s "$RESULTS/split-$meeting.json" ] || nice -n 19 "$BIN" \
    --wav "$ROOT/data/$meeting.Mix-Headset.wav" --models-root "$INT8_MODELS" \
    --variant "$VARIANT" --split-turns "$RESULTS/turns-$meeting.json" \
    --threads "$THREADS" --output "$RESULTS/split-$meeting.json" \
    2>&1 | tee "$LOGS/split-$meeting.log"
done
nice -n 19 "$PY" "$HERE/score.py" \
  --rows "$ROOT19/data/ihm-baseline-full/rows.json" \
  --oracle "ES2004b=$ROOT19/results/mix-decode.jsonl,IS1009b=$ROOT19/results/mix-decode.jsonl,EN2002a=$ROOT19/results/ablation-EN2002a.jsonl" \
  --pipeline "ES2004b=$RESULTS/split-ES2004b.json,IS1009b=$RESULTS/split-IS1009b.json,EN2002a=$RESULTS/split-EN2002a.json" \
  --label split --out "$RESULTS/score-split.json" | tee -a "$RESULTS/asr-experiment-tables.md"
nice -n 19 "$PY" "$HERE/score-meeteval.py" \
  --rows "$ROOT19/data/ihm-baseline-full/rows.json" \
  --oracle "ES2004b=$ROOT19/results/mix-decode.jsonl,IS1009b=$ROOT19/results/mix-decode.jsonl,EN2002a=$ROOT19/results/ablation-EN2002a.jsonl" \
  --pipeline "ES2004b=$RESULTS/split-ES2004b.json,IS1009b=$RESULTS/split-IS1009b.json,EN2002a=$RESULTS/split-EN2002a.json" \
  --label split --out "$RESULTS/meeteval-split.json" | tee -a "$RESULTS/asr-experiment-tables.md"
