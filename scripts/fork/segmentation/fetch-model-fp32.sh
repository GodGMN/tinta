#!/usr/bin/env bash
# Fetch the non-int8 Parakeet TDT 0.6b v3 export (task #23 experiment a) and
# lay it out as an engine models root.
#
#   scripts/fork/segmentation/fetch-model-fp32.sh /path/to/models
#
# The package is the sherpa-onnx fp32 export, pinned by the Hugging Face commit
# and by SHA-256 of every file actually downloaded. The fp16 repository
# (`csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-fp16`) has no model files,
# so fp32 is the only non-int8 choice.
#
# The engine manifest names the int8 files (`encoder.int8.onnx`, ...). The
# downloaded fp32 files are linked under those names; ONNX Runtime finds the
# external weights (`encoder.weights`) next to the model file, so they live in
# the same directory. No file is renamed or rewritten.
set -euo pipefail

DEST="${1:?usage: fetch-model-fp32.sh DEST}"
REV=5fcc0122fe87e8365e339e467af90c67b746218f
REPO=csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3
BASE="https://huggingface.co/$REPO/resolve/$REV"
PKG="$DEST/parakeet-tdt-0.6b-v3"
LAYOUT="$DEST/parakeet-tdt-0.6b-v3-int8"

declare -A SHA=(
  [encoder.onnx]=36b758364dfddce472b9a9f75bf6049bbd35c886d7927e3eda80d4ceb3348512
  [encoder.weights]=3af3f51af5f2d01dbbf5af47d42c7962a2c205f11004254bb4f2b979862f39a8
  [decoder.onnx]=caed31af1bb08edcc09a01a1c3cbb4d4eb5bfdb9c4e5f307012f84bbf5dbfdbc
  [joiner.onnx]=0f294e5ba9021a3d80e492de94abe4c116f0afce78452c7debb5dd07690d0340
  [tokens.txt]=d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d
)

mkdir -p "$PKG" "$LAYOUT"
for file in encoder.onnx encoder.weights decoder.onnx joiner.onnx tokens.txt; do
  if [ ! -s "$PKG/$file" ]; then
    curl -fL --retry 3 -o "$PKG/$file" "$BASE/$file"
  fi
  echo "${SHA[$file]}  $PKG/$file" | sha256sum -c -
done

ln -sfn "$PKG/encoder.onnx" "$LAYOUT/encoder.int8.onnx"
ln -sfn "$PKG/encoder.weights" "$LAYOUT/encoder.weights"
ln -sfn "$PKG/decoder.onnx" "$LAYOUT/decoder.int8.onnx"
ln -sfn "$PKG/joiner.onnx" "$LAYOUT/joiner.int8.onnx"
ln -sfn "$PKG/tokens.txt" "$LAYOUT/tokens.txt"

cat > "$DEST/provenance.txt" <<EOF
repository: https://huggingface.co/$REPO
revision:   $REV
files:      encoder.onnx, encoder.weights, decoder.onnx, joiner.onnx, tokens.txt
sizes:      $(stat -c '%n %s' "$PKG"/* | tr '\n' '; ')
sha256:     $(sha256sum "$PKG"/* | tr '\n' '; ')
EOF

echo "fp32 Parakeet TDT 0.6b v3 ready under $DEST"
echo "VAD, diarization and embedding models still have to be linked into the same root:"
echo "  silero-vad/silero_vad.onnx"
echo "  pyannote-segmentation-3-0/model.int8.onnx"
echo "  3dspeaker-campplus/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
