#!/usr/bin/env bash
# Reproduce the sherpa-onnx CPU benchmark from docs/fork/benchmark.md.
#
# Data and models live outside the repo. Override ROOT to use another location.
# Requirements: cargo, cmake (or `pip install cmake`), python3 + numpy scipy
# rapidfuzz, ffmpeg, GNU time (/usr/bin/time).
set -euo pipefail

ROOT="${ROOT:-/home/sergio/code/evidence/tinta-fork/2}"
BENCH="$(cd "$(dirname "$0")" && pwd)"
BIN="$BENCH/target/release/tinta-bench"
PY="${PY:-$ROOT/venv/bin/python}"
TIME="${TIME:-/usr/bin/time}"

AMI=ES2002a IS1000a IN1014
ASR_MODEL="$ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
SEG_MODEL="$ROOT/models/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx"
EMB_MODEL="$ROOT/models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"

THREADS="${THREADS:-4}"
CHUNK="${CHUNK:-240}"   # seconds; >240 drops words at the head of the chunk
OVERLAP="${OVERLAP:-2}"
THRESHOLD="${THRESHOLD:-1.0}"  # cosine dissimilarity, complete linkage

mkdir -p "$ROOT/results" "$ROOT/logs" "$ROOT/data"

asr() { # $1 meeting, $2 threads
  "$TIME" -v "$BIN" asr \
    --wav "$ROOT/data/$1.Mix-Headset.wav" \
    --chunk-seconds "$CHUNK" --chunk-overlap "$OVERLAP" \
    --encoder "$ASR_MODEL/encoder.int8.onnx" \
    --decoder "$ASR_MODEL/decoder.int8.onnx" \
    --joiner "$ASR_MODEL/joiner.int8.onnx" \
    --tokens "$ASR_MODEL/tokens.txt" \
    --threads "$2" --output "$ROOT/results/asr-$1-t$2-c$CHUNK.json" \
    2> "$ROOT/logs/asr-$1-t$2-c$CHUNK.time"
  "$PY" "$BENCH/score.py" score --ref "$ROOT/data/$1.ref.json" \
    --result "$ROOT/results/asr-$1-t$2-c$CHUNK.json" \
    --out "$ROOT/results/score-asr-$1-t$2-c$CHUNK.json" >/dev/null
}

diar() { # $1 meeting, $2 threads, $3 threshold
  "$TIME" -v "$BIN" diarize \
    --wav "$ROOT/data/$1.Mix-Headset.wav" \
    --seg-model "$SEG_MODEL" --embedding "$EMB_MODEL" \
    --threads "$2" --threshold "$3" --window-shift 0.2 \
    --output "$ROOT/results/diar-$1-t$2-th$3.json" \
    2> "$ROOT/logs/diar-$1-t$2-th$3.time"
  "$PY" "$BENCH/score.py" score --ref "$ROOT/data/$1.ref.json" \
    --result "$ROOT/results/diar-$1-t$2-th$3.json" \
    --out "$ROOT/results/score-diar-$1-t$2-th$3.json" >/dev/null
}

# 1. Build.
( cd "$BENCH" && cargo build --release --locked )

# 2. References from the AMI manual annotations.
for m in $AMI; do
  "$PY" "$BENCH/score.py" ref --annotations "$ROOT/data/ami_annotations" \
    --meeting "$m" --out "$ROOT/data/$m.ref.json"
done

# 3. One heavy job at a time.
for m in $AMI; do asr "$m" "$THREADS"; done
for m in $AMI; do diar "$m" "$THREADS" "$THRESHOLD"; done

# 4. Thread scaling on a 6 minute slice (1, 4, 8 threads).
for t in 1 4 8; do
  "$BIN" asr --wav "$ROOT/data/ES2002a.sample6m.wav" \
    --chunk-seconds "$CHUNK" --chunk-overlap "$OVERLAP" \
    --encoder "$ASR_MODEL/encoder.int8.onnx" \
    --decoder "$ASR_MODEL/decoder.int8.onnx" \
    --joiner "$ASR_MODEL/joiner.int8.onnx" \
    --tokens "$ASR_MODEL/tokens.txt" \
    --threads "$t" --output "$ROOT/results/slice-asr-t$t.json"
  "$BIN" diarize --wav "$ROOT/data/ES2002a.sample6m.wav" \
    --seg-model "$SEG_MODEL" --embedding "$EMB_MODEL" \
    --threads "$t" --threshold "$THRESHOLD" --window-shift 0.2 \
    --output "$ROOT/results/slice-diar-t$t.json"
done

# 5. Clean-speech check (Parakeet v3) and the summary table.
#    The clean sample is built by score.py clean-sample from single-speaker
#    regions, and the LibriSpeech sample from the HF datasets server.
"$PY" "$BENCH/score.py" table "$ROOT"/results/score-*.json
