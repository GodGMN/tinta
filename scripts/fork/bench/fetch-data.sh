#!/usr/bin/env bash
# Fetch the benchmark data and models outside the repo, then verify SHA-256.
# Data: AMI corpus (public, CC BY 4.0). Models: sherpa-onnx release assets.
set -euo pipefail

ROOT="${ROOT:-/home/sergio/code/evidence/tinta-fork/2}"
mkdir -p "$ROOT/data" "$ROOT/models" "$ROOT/results" "$ROOT/logs"

AMI_BASE="https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
MODELS="https://github.com/k2-fsa/sherpa-onnx/releases/download"
ASR_TAR="sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
SEG_TAR="sherpa-onnx-pyannote-segmentation-3-0.tar.bz2"
EMB="3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"

cd "$ROOT"

# 1. AMI: Mix-Headset audio (16 kHz mono), manual annotations, speaker A headset.
for m in ES2002a IS1000a IN1014; do
  [ -f "data/$m.Mix-Headset.wav" ] || curl -fL -o "data/$m.Mix-Headset.wav" \
    "$AMI_BASE/$m/audio/$m.Mix-Headset.wav"
done
[ -f data/ES2002a.Headset-0.wav ] || curl -fL -o data/ES2002a.Headset-0.wav \
  "$AMI_BASE/ES2002a/audio/ES2002a.Headset-0.wav"
[ -f data/ami_public_manual_1.6.2.zip ] || curl -fL -o data/ami_public_manual_1.6.2.zip \
  "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
[ -d data/ami_annotations ] || unzip -q data/ami_public_manual_1.6.2.zip -d data/ami_annotations

# 2. Models.
[ -f "models/$ASR_TAR" ] || curl -fL -o "models/$ASR_TAR" \
  "$MODELS/asr-models/$ASR_TAR"
[ -f "models/$SEG_TAR" ] || curl -fL -o "models/$SEG_TAR" \
  "$MODELS/speaker-segmentation-models/$SEG_TAR"
[ -f "models/$EMB" ] || curl -fL -o "models/$EMB" \
  "$MODELS/speaker-recongition-models/$EMB"

[ -d models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8 ] || \
  tar xjf "models/$ASR_TAR" -C models
[ -d models/sherpa-onnx-pyannote-segmentation-3-0 ] || \
  tar xjf "models/$SEG_TAR" -C models

# 3. Checksums pinned at benchmark time (see docs/fork/benchmark.md).
sha256sum -c <<'EOF'
9c76866990fcc8b84006dc32d273ad99df439090b748ebe72103bb78c3216ee7  data/ES2002a.Mix-Headset.wav
25c9a48237c1f23bc419702f89234467a1b9efedbac16d9673f52d6a8c1c6a57  data/IS1000a.Mix-Headset.wav
ea8c65624b0a4ea4a1fad83fddbc34fcc2c0ff6912dc570e68699ce92da5b00a  data/IN1014.Mix-Headset.wav
b56e5babb2496b8795deeeda7e71178d7fbc9963f94276cf2a3f4b56ebbc9f9d  data/ami_public_manual_1.6.2.zip
5793d0fd397c5778d2cf2126994d58e9d56b1be7c04d13c7a15bb1b4eafb16bf  models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2
24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488  models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11  models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247  models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx
179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e  models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/decoder.int8.onnx
3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3  models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/joiner.int8.onnx
d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d  models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/tokens.txt
d582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d  models/sherpa-onnx-pyannote-segmentation-3-0/model.int8.onnx
EOF
