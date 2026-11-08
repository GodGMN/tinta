#!/usr/bin/env bash
# Task #23: ASR experiments on top of the task #21 segmentation harness.
#
#   export ROOT=/path/to/evidence/tinta-fork/2      # meeting wavs and int8 models
#   export ROOT19=/path/to/evidence/tinta-fork/19   # rows.json, oracle decodes
#   export EVID=/path/to/evidence/tinta-fork/23     # results for this task
#   scripts/fork/segmentation/fetch-model-fp32.sh "$EVID/models"
#   venv/bin/python scripts/fork/segmentation/requantize-encoder.py \
#     --input-dir "$EVID/models/parakeet-tdt-0.6b-v3" \
#     --output "$EVID/models/requant-root/parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx"
#   scripts/fork/segmentation/run-asr-experiments.sh
#
# Measures, with the engine's own decode path:
#   (a) the fp32 Parakeet export, with and without the hole re-decode;
#   (b) the requantized encoder (MatMul-only, per-channel int8, Conv fp32);
#   (c) pieces split at diarized speaker changes inside >= 0.2 s silences;
# and scores every run per utterance (score.py) and with meeteval
# (score-meeteval.py: tcpWER at collars 5 s and 1 s, cpWER).
#
# The int8, requant and fp32 `default` decodes run back-to-back per meeting in
# that order, each with `--finalize` (transcribe + diarize in one process), so
# the time ratio and peak RSS come from the same host load and include the
# engine's real finalize shape. `/usr/bin/time -v` records the peak RSS.
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
REQUANT_MODELS="${REQUANT_MODELS:-$EVID/models/requant-root}"
FP32_MODELS="${FP32_MODELS:-$EVID/models/fp32-root}"
RESULTS="$EVID/results"
LOGS="$EVID/logs"
SEG21="${SEG21:-$ROOT19/../21}"          # task #21 results, for the int8 `before` row
TIMEBIN="${TIMEBIN:-/usr/bin/time}"
[ -x "$TIMEBIN" ] || TIMEBIN="$ROOT/tools/root/usr/bin/time"

mkdir -p "$RESULTS" "$LOGS"

link() { mkdir -p "$(dirname "$2")"; [ -e "$2" ] || ln -s "$1" "$2"; }

# The manifest layout under each models root. VAD and diarization are the same
# int8 models in every root; only the transcription files differ.
common_links() { # root
  local root="$1"
  link "$ROOT/models/silero_vad.onnx" "$root/silero-vad/silero_vad.onnx"
  link "$ROOT/models/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx" \
    "$root/pyannote-segmentation-3-0/model.int8.onnx"
  link "$ROOT/models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx" \
    "$root/3dspeaker-campplus/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
  link "$ROOT19/../21/models/provider.conf" "$root/provider.conf"
}

# int8: the shipped artifact, all four files.
common_links "$INT8_MODELS"
P=parakeet-tdt-0.6b-v3-int8
for file in encoder.int8.onnx decoder.int8.onnx joiner.int8.onnx tokens.txt; do
  link "$ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/$file" \
    "$INT8_MODELS/$P/$file"
done

# requant: the encoder is the output of requantize-encoder.py; the decoder,
# joiner and tokens stay the shipped int8 ones, which the review showed are not
# the cause of the failure.
common_links "$REQUANT_MODELS"
[ -s "$REQUANT_MODELS/$P/encoder.int8.onnx" ] || {
  echo "missing the requantized encoder: run requantize-encoder.py first" >&2
  exit 1
}
for file in decoder.int8.onnx joiner.int8.onnx tokens.txt; do
  link "$ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/$file" \
    "$REQUANT_MODELS/$P/$file"
done

# fp32: the manifest names the int8 files, so the fp32 files are linked as
# encoder.int8.onnx etc.; encoder.weights sits next to the model, which is where
# ONNX Runtime finds the external data.
common_links "$FP32_MODELS"
F="$EVID/models/parakeet-tdt-0.6b-v3"
[ -s "$F/encoder.weights" ] || {
  echo "missing the fp32 model: run scripts/fork/segmentation/fetch-model-fp32.sh $EVID/models" >&2
  exit 1
}
link "$F/encoder.onnx" "$FP32_MODELS/$P/encoder.int8.onnx"
link "$F/encoder.weights" "$FP32_MODELS/$P/encoder.weights"
link "$F/decoder.onnx" "$FP32_MODELS/$P/decoder.int8.onnx"
link "$F/joiner.onnx" "$FP32_MODELS/$P/joiner.int8.onnx"
link "$F/tokens.txt" "$FP32_MODELS/$P/tokens.txt"

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

# One decode with the engine's finalize shape. `time -v` records peak RSS;
# `pass_seconds` in the JSON is the ASR time without diarization.
decode() { # label, models root, extra args...
  local label="$1" root="$2"; shift 2
  local output="$RESULTS/$label-$meeting.json"
  if [ ! -s "$output" ]; then
    "$TIMEBIN" -v -o "$LOGS/$label-$meeting.time" \
      nice -n 19 "$BIN" --wav "$ROOT/data/$meeting.Mix-Headset.wav" \
      --models-root "$root" --threads "$THREADS" --variant "$VARIANT" --finalize \
      --output "$output" "$@" 2>&1 | tee "$LOGS/$label-$meeting.log"
  fi
}

# Same-round comparison: int8, requant, fp32 back to back for each meeting.
for meeting in ${MEETINGS//,/ }; do
  decode int8-default "$INT8_MODELS"
  decode requant-default "$REQUANT_MODELS"
  decode fp32-default "$FP32_MODELS"
done

# fp32 without the hole re-decode, for the re-decode's own delta.
for meeting in ${MEETINGS//,/ }; do
  decode fp32-runs "$FP32_MODELS" --variant runs
done

# Speaker splits on int8, from the engine's diarizer turns above.
for meeting in ${MEETINGS//,/ }; do
  output="$RESULTS/split-$meeting.json"
  [ -s "$output" ] || nice -n 19 "$BIN" \
    --wav "$ROOT/data/$meeting.Mix-Headset.wav" --models-root "$INT8_MODELS" \
    --variant "$VARIANT" --split-turns "$RESULTS/turns-$meeting.json" \
    --threads "$THREADS" --output "$output" 2>&1 | tee "$LOGS/split-$meeting.log"
done

SKIP_SCORE="${SKIP_SCORE:-0}"

score_variant() { # label, field (asr/joined), output suffix
  local label="$1" field="$2" suffix="${3:-$1}"
  local args=""
  for meeting in ${MEETINGS//,/ }; do
    args="$args${args:+,}$meeting=$RESULTS/$label-$meeting.json"
  done
  nice -n 19 "$PY" "$HERE/score.py" \
    --rows "$ROOT19/data/ihm-baseline-full/rows.json" \
    --oracle "ES2004b=$ROOT19/results/mix-decode.jsonl,IS1009b=$ROOT19/results/mix-decode.jsonl,EN2002a=$ROOT19/results/ablation-EN2002a.jsonl" \
    --pipeline "$args" --field "$field" --label "$suffix" \
    --out "$RESULTS/score-$suffix.json" \
    | tee -a "$RESULTS/asr-experiment-tables.md"
  nice -n 19 "$PY" "$HERE/score-meeteval.py" \
    --rows "$ROOT19/data/ihm-baseline-full/rows.json" \
    --oracle "ES2004b=$ROOT19/results/mix-decode.jsonl,IS1009b=$ROOT19/results/mix-decode.jsonl,EN2002a=$ROOT19/results/ablation-EN2002a.jsonl" \
    --pipeline "$args" --label "$suffix" \
    --out "$RESULTS/meeteval-$suffix.json" \
    | tee -a "$RESULTS/asr-experiment-tables.md"
}

if [ "$SKIP_SCORE" != "1" ]; then
  score_variant int8-default asr int8-default
  score_variant requant-default asr requant-default
  score_variant fp32-default asr fp32-default
  score_variant fp32-runs asr fp32-runs
  score_variant split asr split

  # The int8 `before` row (task #21 settings) stays in the document as history;
  # copy its decodes when the task #21 evidence is next to this one.
  for meeting in ${MEETINGS//,/ }; do
    src="$SEG21/results/before-$meeting.json"
    [ -s "$src" ] && link "$src" "$RESULTS/before-$meeting.json" || true
  done
  [ -s "$RESULTS/before-${MEETINGS%%,*}.json" ] && score_variant before asr before || true
fi

# The 1,000-utterance IHM sample. `BENCH` is the task #19 batch harness
# (`tinta-bench`); it builds the recognizer directly, with no VAD.
if [ -n "${BENCH:-}" ] && [ -x "$BENCH" ]; then
  for variant in requant int8 fp32; do
    case "$variant" in
      requant) model="$REQUANT_MODELS" ;;
      int8) model="$INT8_MODELS" ;;
      fp32) model="$FP32_MODELS" ;;
    esac
    output="$RESULTS/ihm-sample-$variant.jsonl"
    [ -s "$output" ] || nice -n 19 "$BENCH" batch \
      --manifest "$EVID/data/sample-1000.tsv" --output "$output" \
      --encoder "$model/$P/encoder.int8.onnx" \
      --decoder "$model/$P/decoder.int8.onnx" \
      --joiner "$model/$P/joiner.int8.onnx" \
      --tokens "$model/$P/tokens.txt" --threads "$THREADS" --spin 0 \
      2>&1 | tee "$LOGS/ihm-sample-$variant.log"
  done
  nice -n 19 "$PY" "$HERE/ihm-sample.py" score \
    --rows "$ROOT19/data/ihm-baseline-full/rows.json" \
    --sample "$EVID/data/sample-1000.json" \
    --hyp "int8=$RESULTS/ihm-sample-int8.jsonl,fp32=$RESULTS/ihm-sample-fp32.jsonl,requant=$RESULTS/ihm-sample-requant.jsonl" \
    --baseline int8 --out "$RESULTS/ihm-sample-score.json" \
    | tee "$LOGS/ihm-sample-score.log"
fi
