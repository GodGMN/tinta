#!/usr/bin/env python3
"""Re-quantize the fp32 Parakeet TDT 0.6b v3 encoder the way the review asked.

The shipped sherpa-onnx int8 encoder is exported with onnxruntime's defaults:
`quantize_dynamic(..., weight_type=QuantType.QUInt8)`, which quantizes every
supported op (Conv included) per tensor. That is the setting the task #23
review identified as the cause of the dropped 13.4 s window: int8 encoder +
fp32 decoder/joiner returns no words, fp32 encoder + int8 decoder/joiner
returns all 32.

This script keeps the same fp32 export (no new download), quantizes only the
MatMul weights (Gemm is listed for completeness; this export has none), per
channel, signed int8, and leaves Conv, LayerNormalization and the embeddings
in fp32. It then copies the sherpa-onnx metadata from the fp32 encoder
(`model_type`, `vocab_size`, `subsampling_factor`, ...), because sherpa-onnx
reads those properties to pick the NeMo transducer path.

`reduce_range=True` is not optional here. On x86 without AVX512-VNNI (the
measurement host is a Zen 3 Ryzen 7 5825U, AVX2 only), per-channel int8
weights without it produce a broken encoder: the 235.4-248.8 s window decodes
to "St." instead of the 32 words fp32 returns, and the encoder output
correlates 0.09 with fp32. With `reduce_range=True` (7-bit weights) the same
window returns the full fp32 text. ONNX Runtime documents `reduce_range` for
per-channel quantization on non-VNNI machines; per-tensor and per-channel
uint8 variants do not need it, but they are either less accurate or not the
requested recipe.

Inputs are pinned by SHA-256, and so is the output: run the script, compare the
printed hash with `OUTPUT_SHA256`, and update the constant only when the
quantizer version or the recipe changes.

Usage:
  requantize-encoder.py --input-dir MODELS/parakeet-tdt-0.6b-v3 \
                        --output MODELS/requant-root/parakeet-tdt-0.6b-v3-int8/encoder.int8.onnx

Requires `onnx` and `onnxruntime` in a venv outside the repository; the
versions are pinned in `scripts/fork/segmentation/requirements-quant.txt` and
printed at startup, because the output hash depends on them. The script writes
`<output>.tmp`, verifies the hash, and only then renames it over `--output`.
"""

import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization import QuantType, quantize_dynamic

# The fp32 export of csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3 at
# revision 5fcc0122fe87e8365e339e467af90c67b746218f.
INPUT_SHA256 = {
    "encoder.onnx": "36b758364dfddce472b9a9f75bf6049bbd35c886d7927e3eda80d4ceb3348512",
    "encoder.weights": "3af3f51af5f2d01dbbf5af47d42c7962a2c205f11004254bb4f2b979862f39a8",
}

# Measured with onnxruntime 1.31.1 / onnx 1.23.2: 881,596,689 B.
OUTPUT_SHA256 = "23c100540a23ac714ece37ba69c2c59bad9fc70ab1e0bd5b368f98d71a2c336f"

OP_TYPES = ["MatMul", "Gemm"]  # Conv stays fp32 on purpose


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, expected, label):
    actual = sha256(path)
    if expected and actual != expected:
        raise SystemExit(f"{label}: sha256 {actual} does not match the pin {expected}")
    return actual


def metadata(model):
    return [(prop.key, prop.value) for prop in model.metadata_props]


def op_counts(model):
    return collections.Counter(node.op_type for node in model.graph.node)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, help="the fp32 sherpa export")
    parser.add_argument("--output", required=True, help="the encoder.int8.onnx to write")
    parser.add_argument(
        "--allow-unpinned-output",
        action="store_true",
        help="write even when the output hash differs from OUTPUT_SHA256 (for a recipe change)",
    )
    args = parser.parse_args()

    # The output hash depends on the quantizer, so print exactly what ran.
    print(
        f"onnxruntime {onnxruntime.__version__}, "
        f"onnx {onnx.__version__}, numpy {np.__version__}"
    )

    encoder = os.path.join(args.input_dir, "encoder.onnx")
    weights = os.path.join(args.input_dir, "encoder.weights")
    input_hashes = {
        "encoder.onnx": verify(encoder, INPUT_SHA256["encoder.onnx"], encoder),
        "encoder.weights": verify(weights, INPUT_SHA256["encoder.weights"], weights),
    }
    print(f"input encoder.onnx {input_hashes['encoder.onnx']}")
    print(f"input encoder.weights {input_hashes['encoder.weights']}")

    original = onnx.load(encoder, load_external_data=False)
    original_metadata = metadata(original)
    original_ops = op_counts(original)
    if not original_metadata:
        raise SystemExit("the fp32 encoder has no metadata_props; sherpa-onnx needs them")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    # Never write the pinned path before the pin is checked: a live models root
    # would otherwise keep an unpinned file when the script exits non-zero.
    output = os.path.abspath(args.output)
    tmp = output + ".tmp"
    if os.path.exists(tmp):
        os.unlink(tmp)
    # quantize_dynamic loads the external weights itself and, with
    # use_external_data_format=False, saves one file. The quantized encoder is
    # expected around 0.9 GB, below the 2 GB protobuf limit; if a future
    # variant crosses it, flip the flag and ship encoder.weights next to it.
    try:
        quantize_dynamic(
            encoder,
            tmp,
            op_types_to_quantize=OP_TYPES,
            per_channel=True,
            reduce_range=True,  # non-VNNI x86: without it the encoder is broken
            weight_type=QuantType.QInt8,
            use_external_data_format=False,
        )

        # quantize_dynamic rebuilds the graph but keeps metadata_props in
        # practice; set them explicitly so the file does not depend on that.
        quantized = onnx.load(tmp, load_external_data=False)
        del quantized.metadata_props[:]
        for key, value in original_metadata:
            prop = quantized.metadata_props.add()
            prop.key = key
            prop.value = value
        onnx.save_model(quantized, tmp, save_as_external_data=False)

        ops = op_counts(quantized)
        size = os.path.getsize(tmp)
        output_hash = sha256(tmp)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise

    summary = {
        "input_sha256": input_hashes,
        "output": output,
        "output_sha256": output_hash,
        "output_bytes": size,
        "output_gib": round(size / 2**30, 3),
        "op_types_quantized": OP_TYPES,
        "per_channel": True,
        "reduce_range": True,
        "weight_type": "QInt8",
        "conv_kept_fp32": ops.get("Conv", 0) == original_ops.get("Conv", 0),
        "matmul_nodes": original_ops.get("MatMul", 0),
        "matmul_integer_nodes": ops.get("MatMulInteger", 0),
        "metadata_props": dict(original_metadata),
    }
    print(json.dumps(summary, indent=1))
    if OUTPUT_SHA256 and output_hash != OUTPUT_SHA256 and not args.allow_unpinned_output:
        os.unlink(tmp)
        raise SystemExit(
            f"output sha256 {output_hash} does not match OUTPUT_SHA256 {OUTPUT_SHA256}; "
            "update the pin if the recipe changed"
        )
    # The rename is atomic, so a reader never sees a half-written encoder.
    os.replace(tmp, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
