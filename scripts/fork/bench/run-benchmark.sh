#!/usr/bin/env bash
# Reproduce the sherpa-onnx CPU benchmark from docs/fork/benchmark.md.
#
# Data and models live outside the repo. Override ROOT to use another location.
# Requirements: cargo, cmake (or `pip install cmake`), python3 + numpy scipy
# rapidfuzz pyannote.metrics whisper-normalizer, ffmpeg, GNU time.
#
# Meetings follow the official AMI full-corpus partition. Only DEV is used to
# tune clustering; TEST is scored with the frozen setting.
set -euo pipefail

ROOT="${ROOT:-/home/sergio/code/evidence/tinta-fork/2}"
BENCH="$(cd "$(dirname "$0")" && pwd)"
BIN="$BENCH/target/release/tinta-bench"
PY="${PY:-$ROOT/venv/bin/python}"
TIMEBIN="${TIMEBIN:-/usr/bin/time}"
[ -x "$TIMEBIN" ] || TIMEBIN="$ROOT/tools/root/usr/bin/time"

TRAIN="ES2002a IS1000a IN1014"
DEV="ES2003b IS1008b TS3006b"
TEST="ES2004b IS1009b TS3007b"

ASR_MODEL="$ROOT/models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
SEG_MODEL="$ROOT/models/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx"
EMB_MODEL="$ROOT/models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
VAD_MODEL="$ROOT/models/silero_vad.onnx"

THREADS="${THREADS:-8}"
DIAR_THREADS="${DIAR_THREADS:-4}"
THRESHOLDS="${THRESHOLDS:-0.7 0.9 1.0 1.1}"
FILTER="${FILTER:-0.02}"
ALLOW_BUSY="${ALLOW_BUSY:-0}"
SPIN="${SPIN:-0}"  # 0 disables ONNX Runtime intra-op spinning; -1 keeps the default
RESULTS="${RESULTS:-$ROOT/results/repro}"

mkdir -p "$ROOT/results" "$ROOT/logs" "$ROOT/data" "$RESULTS"
rm -f "$RESULTS"/*.json "$RESULTS"/*.txt 2>/dev/null || true

check_idle() { # before every timed run
  local load
  load="$(awk '{print $1}' /proc/loadavg)"
  if awk "BEGIN{exit !($load < 2)}"; then
    return 0
  fi
  echo "WARNING: 1-min load average is $load (>= 2); timings are not clean." >&2
  if [ "$ALLOW_BUSY" = "1" ]; then
    echo "ALLOW_BUSY=1: continuing and recording the load in every result JSON." >&2
    return 0
  fi
  echo "Wait for an idle machine, or set ALLOW_BUSY=1 to record busy-host numbers." >&2
  return 1
}

asr() { # meeting, threads, tag, extra args...
  local m="$1" t="$2" tag="$3"; shift 3
  check_idle
  "$TIMEBIN" -v "$BIN" asr \
    --wav "$ROOT/data/$m.Mix-Headset.wav" \
    --vad-model "$VAD_MODEL" \
    --encoder "$ASR_MODEL/encoder.int8.onnx" \
    --decoder "$ASR_MODEL/decoder.int8.onnx" \
    --joiner "$ASR_MODEL/joiner.int8.onnx" \
    --tokens "$ASR_MODEL/tokens.txt" \
    --threads "$t" --spin "$SPIN" --output "$RESULTS/asr-$m-$tag.json" "$@" \
    2> "$ROOT/logs/asr-$m-$tag.time"
  "$PY" "$BENCH/score.py" score --ref "$ROOT/data/$m.ref.json" \
    --result "$RESULTS/asr-$m-$tag.json" \
    --out "$RESULTS/score-asr-$m-$tag.json" >/dev/null
}

asr_chunk() { # meeting, chunk, overlap, threads
  local m="$1" c="$2" o="$3" t="$4"
  check_idle
  "$TIMEBIN" -v "$BIN" asr \
    --wav "$ROOT/data/$m.Mix-Headset.wav" \
    --chunk-seconds "$c" --chunk-overlap "$o" \
    --encoder "$ASR_MODEL/encoder.int8.onnx" \
    --decoder "$ASR_MODEL/decoder.int8.onnx" \
    --joiner "$ASR_MODEL/joiner.int8.onnx" \
    --tokens "$ASR_MODEL/tokens.txt" \
    --threads "$t" --spin "$SPIN" --output "$RESULTS/asr-$m-c$c-o$o-t$t.json" \
    2> "$ROOT/logs/asr-$m-c$c-o$o-t$t.time"
  "$PY" "$BENCH/score.py" score --ref "$ROOT/data/$m.ref.json" \
    --result "$RESULTS/asr-$m-c$c-o$o-t$t.json" \
    --out "$RESULTS/score-asr-$m-c$c-o$o-t$t.json" >/dev/null
}

diar_slice() { # meeting, threads, threshold, filter
  local m="$1" t="$2" th="$3" f="$4" tag
  tag="sample6m-t$t-th$th"
  check_idle
  "$TIMEBIN" -v "$BIN" diarize \
    --wav "$ROOT/data/$m.sample6m.wav" \
    --seg-model "$SEG_MODEL" --embedding "$EMB_MODEL" \
    --threads "$t" --threshold "$th" --window-shift 0.2 --spin "$SPIN" \
    --output "$RESULTS/diar-$m-$tag.json" \
    2> "$ROOT/logs/diar-$m-$tag.time"
  "$PY" "$BENCH/score.py" score --ref "$ROOT/data/$m.sample6m.ref.json" \
    --result "$RESULTS/diar-$m-$tag.json" \
    --small-cluster-fraction "$f" \
    --out "$RESULTS/score-diar-$m-$tag-f$f.json" >/dev/null
}

diar() { # meeting, threads, threshold, filter
  local m="$1" t="$2" th="$3" f="$4" tag
  tag="t$t-th$th"
  check_idle
  "$TIMEBIN" -v "$BIN" diarize \
    --wav "$ROOT/data/$m.Mix-Headset.wav" \
    --seg-model "$SEG_MODEL" --embedding "$EMB_MODEL" \
    --threads "$t" --threshold "$th" --window-shift 0.2 --spin "$SPIN" \
    --output "$RESULTS/diar-$m-$tag.json" \
    2> "$ROOT/logs/diar-$m-$tag.time"
  "$PY" "$BENCH/score.py" score --ref "$ROOT/data/$m.ref.json" \
    --result "$RESULTS/diar-$m-$tag.json" \
    --small-cluster-fraction "$f" \
    --out "$RESULTS/score-diar-$m-$tag-f$f.json" >/dev/null
}

combined() { # meeting, optional extra args...
  local m="$1"; shift
  check_idle
  "$TIMEBIN" -v "$BIN" combined \
    --wav "$ROOT/data/$m.Mix-Headset.wav" --vad-model "$VAD_MODEL" \
    --encoder "$ASR_MODEL/encoder.int8.onnx" \
    --decoder "$ASR_MODEL/decoder.int8.onnx" \
    --joiner "$ASR_MODEL/joiner.int8.onnx" \
    --tokens "$ASR_MODEL/tokens.txt" \
    --seg-model "$SEG_MODEL" --embedding "$EMB_MODEL" \
    --threshold "${CHOSEN_TH:-0.7}" --window-shift 0.2 --spin "$SPIN" "$@" \
    --output "$RESULTS/par-$m.json" \
    2> "$ROOT/logs/par-$m.time"
  "$PY" "$BENCH/score.py" score --ref "$ROOT/data/$m.ref.json" \
    --result "$RESULTS/par-$m.json" --small-cluster-fraction "$FILTER" \
    --out "$RESULTS/score-par-$m.json" >/dev/null
}

slice() { # meeting, start, duration, out
  "$PY" "$BENCH/score.py" slice --wav "$ROOT/data/$1.Mix-Headset.wav" \
    --start "$2" --duration "$3" --out-wav "$4"
}

# 1. Build and fetch.
( cd "$BENCH" && cargo build --release --locked )
"$BENCH/fetch-data.sh"

# 2. References for every meeting used below.
for m in $TRAIN $DEV $TEST; do
  "$PY" "$BENCH/score.py" ref --annotations "$ROOT/data/ami_annotations" \
    --meeting "$m" --out "$ROOT/data/$m.ref.json"
done

# 3. Slices and clean-speech samples.
slice ES2002a 0 360 "$ROOT/data/ES2002a.sample6m.wav"
slice ES2003b 600 360 "$ROOT/data/ES2003b.sample6m.wav"
"$PY" "$BENCH/score.py" slice-ref --ref "$ROOT/data/ES2003b.ref.json" \
  --start 600 --duration 360 --out "$ROOT/data/ES2003b.sample6m.ref.json"
"$PY" "$BENCH/score.py" clean-sample --ref "$ROOT/data/ES2002a.ref.json" \
  --wav "$ROOT/data/ES2002a.Mix-Headset.wav" --minutes 5 \
  --out-wav "$ROOT/data/ES2002a.clean5m.wav" \
  --out-ref "$ROOT/data/ES2002a.clean5m.ref.json"
"$PY" "$BENCH/score.py" librispeech-sample \
  --out-wav "$ROOT/data/librispeech_testclean_30.wav" \
  --out-ref "$ROOT/data/librispeech_testclean_30.ref.json"

# 4. ASR: VAD segmentation, 4 threads, train + held-out test.
for m in $TRAIN $TEST; do asr "$m" "$THREADS" "vad-t$THREADS"; done

# 5. ASR ablations: the old fixed 240 s chunk mode (midpoint de-duplication)
#    and the clean-speech samples.
for m in ES2002a IS1000a; do asr_chunk "$m" 240 2 "$THREADS"; done
# The clean samples are not AMI meetings, so call the binary directly.
check_idle
"$TIMEBIN" -v "$BIN" asr --wav "$ROOT/data/ES2002a.clean5m.wav" \
  --vad-model "$VAD_MODEL" \
  --encoder "$ASR_MODEL/encoder.int8.onnx" --decoder "$ASR_MODEL/decoder.int8.onnx" \
  --joiner "$ASR_MODEL/joiner.int8.onnx" --tokens "$ASR_MODEL/tokens.txt" \
  --threads "$THREADS" --output "$RESULTS/asr-clean5m.json" \
  2> "$ROOT/logs/asr-clean5m.time"
"$PY" "$BENCH/score.py" score --ref "$ROOT/data/ES2002a.clean5m.ref.json" \
  --result "$RESULTS/asr-clean5m.json" --out "$RESULTS/score-asr-clean5m.json" >/dev/null
check_idle
"$TIMEBIN" -v "$BIN" asr --wav "$ROOT/data/librispeech_testclean_30.wav" \
  --vad-model "$VAD_MODEL" \
  --encoder "$ASR_MODEL/encoder.int8.onnx" --decoder "$ASR_MODEL/decoder.int8.onnx" \
  --joiner "$ASR_MODEL/joiner.int8.onnx" --tokens "$ASR_MODEL/tokens.txt" \
  --threads "$THREADS" --output "$RESULTS/asr-librispeech30.json" \
  2> "$ROOT/logs/asr-librispeech30.time"
"$PY" "$BENCH/score.py" score --ref "$ROOT/data/librispeech_testclean_30.ref.json" \
  --result "$RESULTS/asr-librispeech30.json" --out "$RESULTS/score-asr-librispeech30.json" >/dev/null

# 6. Threads and ONNX Runtime spin ablation on the 6 min slice.
for t in 1 2 4; do
  check_idle
  "$TIMEBIN" -v "$BIN" asr --wav "$ROOT/data/ES2002a.sample6m.wav" \
    --vad-model "$VAD_MODEL" \
    --encoder "$ASR_MODEL/encoder.int8.onnx" --decoder "$ASR_MODEL/decoder.int8.onnx" \
    --joiner "$ASR_MODEL/joiner.int8.onnx" --tokens "$ASR_MODEL/tokens.txt" \
    --threads "$t" --output "$RESULTS/slice-asr-t$t.json" \
    2> "$ROOT/logs/slice-asr-t$t.time"
  check_idle
  "$TIMEBIN" -v "$BIN" diarize --wav "$ROOT/data/ES2002a.sample6m.wav" \
    --seg-model "$SEG_MODEL" --embedding "$EMB_MODEL" \
    --threads "$t" --threshold 1.0 --window-shift 0.2 \
    --output "$RESULTS/slice-diar-t$t.json" \
    2> "$ROOT/logs/slice-diar-t$t.time"
done
# session.intra_op.allow_spinning=0 through the provider config file.
for stage in asr diar; do
  check_idle
  if [ "$stage" = asr ]; then
    "$TIMEBIN" -v "$BIN" asr --wav "$ROOT/data/ES2002a.sample6m.wav" \
      --vad-model "$VAD_MODEL" \
      --encoder "$ASR_MODEL/encoder.int8.onnx" --decoder "$ASR_MODEL/decoder.int8.onnx" \
      --joiner "$ASR_MODEL/joiner.int8.onnx" --tokens "$ASR_MODEL/tokens.txt" \
      --threads "$THREADS" --spin 0 --output "$RESULTS/slice-asr-t$THREADS-spin0.json" \
      2> "$ROOT/logs/slice-asr-t$THREADS-spin0.time"
  else
    "$TIMEBIN" -v "$BIN" diarize --wav "$ROOT/data/ES2002a.sample6m.wav" \
      --seg-model "$SEG_MODEL" --embedding "$EMB_MODEL" \
      --threads "$THREADS" --threshold 1.0 --window-shift 0.2 --spin 0 \
      --output "$RESULTS/slice-diar-t$THREADS-spin0.json" \
      2> "$ROOT/logs/slice-diar-t$THREADS-spin0.time"
  fi
done

# 7. Diarization tuning on the dev meetings only. The full IS1008b meeting is
#    the threshold sweep; ES2003b at 1.0 shows the high-threshold merge, and its
#    6 min sample covers the lower values. 0.7 is frozen from this table: it is
#    the only value that keeps 4 filtered speakers on both dev meetings.
for th in $THRESHOLDS; do diar IS1008b "$DIAR_THREADS" "$th" "$FILTER"; done
diar ES2003b "$DIAR_THREADS" 1.0 "$FILTER"
for th in 0.7 0.8 0.9; do diar_slice ES2003b "$DIAR_THREADS" "$th" "$FILTER"; done
CHOSEN_TH="${CHOSEN_TH:-0.7}"
CHOSEN_F="${CHOSEN_F:-0.02}"
echo "frozen from dev: threshold=$CHOSEN_TH small-cluster-filter=$CHOSEN_F" | tee "$RESULTS/frozen.txt"

# 8. Diarization on held-out test meetings with the frozen threshold.
for m in $TEST; do diar "$m" "$DIAR_THREADS" "$CHOSEN_TH" "$CHOSEN_F"; done

# 9. Parallel pipeline: ASR and diarization at the same time, default threads
#    from the CPUs the process actually gets.
for m in IN1014 TS3007b; do combined "$m"; done

# 10. Tables.
{
  echo "## ASR, VAD segmented (train)"
  for m in $TRAIN; do echo "$RESULTS/score-asr-$m-vad-t$THREADS.json"; done | \
    xargs "$PY" "$BENCH/score.py" table
  echo
  echo "## ASR, VAD segmented (held-out test)"
  for m in $TEST; do echo "$RESULTS/score-asr-$m-vad-t$THREADS.json"; done | \
    xargs "$PY" "$BENCH/score.py" table
  echo
  echo "## ASR clean speech"
  "$PY" "$BENCH/score.py" table "$RESULTS/score-asr-clean5m.json" \
    "$RESULTS/score-asr-librispeech30.json"
  echo
  echo "## ASR fixed 240 s chunks vs VAD"
  "$PY" "$BENCH/score.py" table "$RESULTS/score-asr-*-c240-*.json"
  echo
  echo "## Diarization, dev tuning"
  "$PY" "$BENCH/score.py" table "$RESULTS"/score-diar-IS1008b-t"$DIAR_THREADS"-th*.json \
    "$RESULTS"/score-diar-ES2003b-*.json
  echo
  echo "## Diarization, held-out test"
  for m in $TEST; do echo "$RESULTS/score-diar-$m-t$DIAR_THREADS-th$CHOSEN_TH-f$CHOSEN_F.json"; done | \
    xargs "$PY" "$BENCH/score.py" table
  echo
  echo "## Combined pipeline"
  "$PY" "$BENCH/score.py" table "$RESULTS"/score-par-*.json
  echo
  echo "## Threads / spin ablation"
  "$PY" "$BENCH/score.py" timing "$RESULTS"/slice-asr-t*.json "$RESULTS"/slice-diar-t*.json
  echo
  echo "## Dev tuning (threshold x small-cluster filter)"
  cat "$RESULTS/tune-dev.txt"
} | tee "$RESULTS/summary.md"
