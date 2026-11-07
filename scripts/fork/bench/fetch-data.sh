#!/usr/bin/env bash
# Fetch the benchmark data and models outside the repo, then verify SHA-256.
# Data: AMI corpus (public, CC BY 4.0). Models: sherpa-onnx release assets.
#
# Meetings follow the official AMI full-corpus partition:
#   train: ES2002a IS1000a IN1014
#   dev:   ES2003b IS1008b TS3006b
#   test:  ES2004b IS1009b TS3007b
# Only dev is used to tune the clustering threshold; test is held out.
set -euo pipefail

ROOT="${ROOT:-/home/sergio/code/evidence/tinta-fork/2}"
BENCH="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-$ROOT/venv/bin/python}"
[ -x "$PY" ] || PY=python3

TRAIN="ES2002a IS1000a IN1014"
DEV="ES2003b IS1008b TS3006b"
TEST="ES2004b IS1009b TS3007b"

AMI_BASE="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
MODELS="https://github.com/k2-fsa/sherpa-onnx/releases/download"
ASR_TAR="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
SEG_TAR="sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB="3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"

mkdir -p "$ROOT/data" "$ROOT/models" "$ROOT/results" "$ROOT/logs"
cd "$ROOT"

# 1. AMI: Mix-Headset audio (16 kHz mono), manual annotations, speaker A headset.
#    A partial file from an interrupted download is rejected by its SHA-256.
audio_hash() {
  case "$1" in
    ES2002a) echo 9c76866990fcc8b84006dc32d273ad99df439090b748ebe72103bb78c3216ee7 ;;
    IS1000a) echo 25c9a48237c1f23bc419702f89234467a1b9efedbac16d9673f52d6a8c1c6a57 ;;
    IN1014) echo ea8c65624b0a4ea4a1fad83fddbc34fcc2c0ff6912dc570e68699ce92da5b00a ;;
    ES2003b) echo 71ba1c132d1957bd9b20f4c0ccae8b05ebf2399302b451897624a6de9611032f ;;
    IS1008b) echo 0d992a210480c6e9652b317ed7f66261e80a1aa86f7e5c3dffd10e81b8e30de2 ;;
    TS3006b) echo 65008347971a4da203260860cbf0266814dcd8db3ac63a5242628e21d06c88c0 ;;
    ES2004b) echo ad0cf07c42b1694ccf7bea8f1a37348d9cfab194787abc7d050b29ba56e365c8 ;;
    IS1009b) echo 07c2891ae6ad7c507b2a4f15b2dcd0a2343491df9e4e6a58f9db0d0b2c5e1382 ;;
    TS3007b) echo 414e49da13f0831166d3b92d62f96eef6b7802f734b356eff570f7e37beffe5d ;;
  esac
}

fetch_audio() {
  local m="$1" f="data/$m.Mix-Headset.wav" h
  h="$(audio_hash "$m")"
  if [ -s "$f" ] && printf '%s  %s\n' "$h" "$f" | sha256sum -c --status; then
    return 0
  fi
  rm -f "$f"
  curl -fL --retry 5 --retry-all-errors --retry-delay 3 -o "$f" "$AMI_BASE/$m/audio/$m.Mix-Headset.wav"
}

for m in $TRAIN $DEV $TEST; do fetch_audio "$m"; done
if [ ! -s data/ES2002a.Headset-0.wav ] || ! printf '%s  %s\n' \
  3bf0627106d6b6f02e5cc8b69f283b70303275c6b354aef3bdc7a50762d3f25e \
  data/ES2002a.Headset-0.wav | sha256sum -c --status; then
  rm -f data/ES2002a.Headset-0.wav
  curl -fL --retry 5 --retry-all-errors -o data/ES2002a.Headset-0.wav \
    "$AMI_BASE/ES2002a/audio/ES2002a.Headset-0.wav"
fi
[ -s data/ami_public_manual_1.6.2.zip ] || curl -fL --retry 5 --retry-all-errors \
  -o data/ami_public_manual_1.6.2.zip \
  "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
[ -d data/ami_annotations ] || unzip -q data/ami_public_manual_1.6.2.zip -d data/ami_annotations

# 2. Models.
[ -s "models/$ASR_TAR" ] || curl -fL --retry 5 --retry-all-errors -o "models/$ASR_TAR" "$MODELS/asr-models/$ASR_TAR"
[ -s "models/$SEG_TAR" ] || curl -fL --retry 5 --retry-all-errors -o "models/$SEG_TAR" \
  "$MODELS/speaker-segmentation-models/$SEG_TAR"
[ -s "models/$EMB" ] || curl -fL --retry 5 --retry-all-errors -o "models/$EMB" \
  "$MODELS/speaker-recongition-models/$EMB"
[ -s models/silero_vad.onnx ] || curl -fL --retry 5 --retry-all-errors -o models/silero_vad.onnx \
  "$MODELS/asr-models/silero_vad.onnx"

[ -d models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 ] || \
  tar xjf "models/$ASR_TAR" -C models
[ -d models/sherpa-onnx-pyannote-segmentation-3-0 ] || \
  tar xjf "models/$SEG_TAR" -C models

# 3. Deterministic derived audio clips (same bytes every time).
if [ ! -s data/ES2002a.sample6m.wav ] || ! printf '%s  %s\n' \
  180f93ee8c7a432007ee79ee3a0822f7a8443d6b485282f6553a00ec32d6f2ec \
  data/ES2002a.sample6m.wav | sha256sum -c --status; then
  "$PY" "$BENCH/score.py" slice --wav data/ES2002a.Mix-Headset.wav \
    --start 0 --duration 360 --out-wav data/ES2002a.sample6m.wav
fi
if [ ! -s data/ES2002a.sample2m.wav ] || ! printf '%s  %s\n' \
  d9db8bd518bc5574c0c48be0372693ed2756d27310494eedb9a70588d63fcbf5 \
  data/ES2002a.sample2m.wav | sha256sum -c --status; then
  "$PY" "$BENCH/score.py" slice --wav data/ES2002a.Mix-Headset.wav \
    --start 0 --duration 120 --out-wav data/ES2002a.sample2m.wav
fi
if [ ! -s data/ES2002a.ref.json ]; then
  "$PY" "$BENCH/score.py" ref --annotations data/ami_annotations \
    --meeting ES2002a --out data/ES2002a.ref.json
fi
if [ ! -s data/ES2002a.clean5m.wav ]; then
  "$PY" "$BENCH/score.py" clean-sample --ref data/ES2002a.ref.json \
    --wav data/ES2002a.Mix-Headset.wav --minutes 5 \
    --out-wav data/ES2002a.clean5m.wav --out-ref data/ES2002a.clean5m.ref.json
fi
for spec in "ES2003b 600 e2d03ff11b9dbb100e59f61ca973a7277135adde1eefa1e2666445bf64e80184" \
            "IS1009b 600 2e5d1d3afa40272698d4abca8a12ce98b5e78f6be84a0331f5458784a7a89ce9" \
            "ES2004b 500 7c4e9154036380e35ed828247f400d1d9a567fb22f253abf86fc8c09cd23f429"; do
  set -- $spec
  m="$1"; start="$2"; want="$3"; f="data/$m.sample6m.wav"
  if [ ! -s "$f" ] || ! printf '%s  %s\n' "$want" "$f" | sha256sum -c --status; then
    "$PY" "$BENCH/score.py" slice --wav "data/$m.Mix-Headset.wav" \
      --start "$start" --duration 360 --out-wav "$f"
  fi
done

if [ ! -s data/librispeech_testclean_30.wav ]; then
  "$PY" "$BENCH/score.py" librispeech-sample \
    --out-wav data/librispeech_testclean_30.wav \
    --out-ref data/librispeech_testclean_30.ref.json
fi

# 4. AMI IHM test split for the ASR baseline (docs/fork/asr-baseline.md, task 19).
#    `edinburghcstr/ami`, config `ihm`, split `test`: the reference-segmented
#    single-headset condition the Open ASR Leaderboard and the Parakeet v3 model
#    card use (11.31% WER). Public, no token. The revision is pinned; the LFS oid
#    of each parquet file is its SHA-256, checked here again.
BASE19="${BASE19:-/home/sergio/code/evidence/tinta-fork/19}"
AMI_REV="46f28f2503e2ec48f8867a84eef356c70476beab"
IHM_BASE="https://huggingface.co/datasets/edinburghcstr/ami/resolve/$AMI_REV/ihm"
mkdir -p "$BASE19/data/ihm-test"
ihm_files="
d95920dccc6924c15215239461bf5d1152fe07c9c61add073bd12da26dd602e0 test-00000-of-00004.parquet
07a2b1c407cf36bc1e66c1362b0248ed32c897edc2a116549d1dc6b456dc2325 test-00001-of-00004.parquet
83cde21e55fc884295585327b0ee334cba47ad53bdb787dd5f97437067aca5f8 test-00002-of-00004.parquet
3312bb79400ddef62d5a0097e124dfabe86a693b4643e7fef7d72a62588280d7 test-00003-of-00004.parquet
"
printf '%s\n' "$ihm_files" | while read -r want f; do
  [ -n "${want:-}" ] || continue
  p="$BASE19/data/ihm-test/$f"
  if [ -s "$p" ] && printf '%s  %s\n' "$want" "$p" | sha256sum -c --status; then
    continue
  fi
  rm -f "$p"
  curl -fL --retry 5 --retry-all-errors --retry-delay 3 -o "$p" "$IHM_BASE/$f"
  printf '%s  %s\n' "$want" "$p" | sha256sum -c --status
done

# 5. Checksums pinned at benchmark time (see docs/fork/benchmark.md).
sha256sum -c <<'EOF'
9c76866990fcc8b84006dc32d273ad99df439090b748ebe72103bb78c3216ee7  data/ES2002a.Mix-Headset.wav
25c9a48237c1f23bc419702f89234467a1b9efedbac16d9673f52d6a8c1c6a57  data/IS1000a.Mix-Headset.wav
ea8c65624b0a4ea4a1fad83fddbc34fcc2c0ff6912dc570e68699ce92da5b00a  data/IN1014.Mix-Headset.wav
3bf0627106d6b6f02e5cc8b69f283b70303275c6b354aef3bdc7a50762d3f25e  data/ES2002a.Headset-0.wav
71ba1c132d1957bd9b20f4c0ccae8b05ebf2399302b451897624a6de9611032f  data/ES2003b.Mix-Headset.wav
0d992a210480c6e9652b317ed7f66261e80a1aa86f7e5c3dffd10e81b8e30de2  data/IS1008b.Mix-Headset.wav
65008347971a4da203260860cbf0266814dcd8db3ac63a5242628e21d06c88c0  data/TS3006b.Mix-Headset.wav
ad0cf07c42b1694ccf7bea8f1a37348d9cfab194787abc7d050b29ba56e365c8  data/ES2004b.Mix-Headset.wav
07c2891ae6ad7c507b2a4f15b2dcd0a2343491df9e4e6a58f9db0d0b2c5e1382  data/IS1009b.Mix-Headset.wav
414e49da13f0831166d3b92d62f96eef6b7802f734b356eff570f7e37beffe5d  data/TS3007b.Mix-Headset.wav
180f93ee8c7a432007ee79ee3a0822f7a8443d6b485282f6553a00ec32d6f2ec  data/ES2002a.sample6m.wav
d9db8bd518bc5574c0c48be0372693ed2756d27310494eedb9a70588d63fcbf5  data/ES2002a.sample2m.wav
6fbef6fc365f962182098ca08f2e56cfcb31720469f898a964f310d883eee349  data/ES2002a.clean5m.wav
b9e82183495d3dfa9b97cb6977e4cf6743c7bce9291003f5270692095c380748  data/librispeech_testclean_30.wav
e2d03ff11b9dbb100e59f61ca973a7277135adde1eefa1e2666445bf64e80184  data/ES2003b.sample6m.wav
2e5d1d3afa40272698d4abca8a12ce98b5e78f6be84a0331f5458784a7a89ce9  data/IS1009b.sample6m.wav
7c4e9154036380e35ed828247f400d1d9a567fb22f253abf86fc8c09cd23f429  data/ES2004b.sample6m.wav
b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d  data/ami_public_manual_1.6.2.zip
5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf  models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488  models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11  models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6  models/silero_vad.onnx
acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247  models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx
179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e  models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx
3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3  models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx
d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d  models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt
d582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d  models/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx
EOF
