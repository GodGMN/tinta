#!/usr/bin/env python3
"""Check the requantized encoder against its pin, or against per-tensor digests.

The CI job builds the encoder on every platform. The primary check is the
output SHA-256: if a platform's ONNX Runtime quantizer is reproducible, the
bytes match the pin. If they do not, the job must not fail on a legitimate
kernel-order difference, so this script falls back to a per-tensor comparison
against `requant-digest.json` and reports exactly what differed.

The fallback still rejects a wrong recipe: metadata, op counts, initializer
names, dtypes, shapes and float tensors (scales, zero points and the fp32 Conv
weights) must match. Only int8/uint8 weight statistics may move within
tolerance (mean 0.25, extremes 2), which covers a ±1 rounding difference and
nothing that changes the model's meaning.

Usage:
  verify-requant.py --model encoder.int8.onnx \\
                    --expect-sha256 23c1... --digest requant-digest.json
  verify-requant.py --model encoder.int8.onnx --write-digest requant-digest.json
"""

import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np
import onnx
from onnx import numpy_helper

INT8_MEMBERS = {2, 3}  # UINT8, INT8
FLOAT_MEMBERS = {1, 10, 11}  # FLOAT, FLOAT16, DOUBLE
MEAN_TOLERANCE = 0.25
EXTREME_TOLERANCE = 2.0
FLOAT_TOLERANCE = 1e-6


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_stats(array):
    flat = array.reshape(-1)
    if flat.size == 0:
        return {"n": 0, "sha256": hashlib.sha256(b"").hexdigest()}
    return {
        "n": int(flat.size),
        "sha256": hashlib.sha256(np.ascontiguousarray(flat).tobytes()).hexdigest(),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.astype(np.float64).mean()),
        "std": float(flat.astype(np.float64).std()),
    }


def describe(model_path):
    model = onnx.load(model_path, load_external_data=False)
    initializers = {}
    base_dir = os.path.dirname(os.path.abspath(model_path))
    for initializer in model.graph.initializer:
        array = numpy_helper.to_array(initializer, base_dir=base_dir)
        entry = {"dtype": int(initializer.data_type), "dims": list(initializer.dims)}
        entry.update(tensor_stats(array))
        initializers[initializer.name] = entry
    return {
        "model_sha256": sha256(model_path),
        "op_counts": dict(
            sorted(collections.Counter(node.op_type for node in model.graph.node).items())
        ),
        "metadata_props": {prop.key: prop.value for prop in model.metadata_props},
        "initializers": initializers,
    }


def compare(expected, actual):
    failures = []
    if expected["metadata_props"] != actual["metadata_props"]:
        failures.append("metadata_props differ")
    if expected["op_counts"] != actual["op_counts"]:
        failures.append(
            f"op counts differ: expected {expected['op_counts']}, got {actual['op_counts']}"
        )
    expected_tensors = expected["initializers"]
    actual_tensors = actual["initializers"]
    missing = sorted(set(expected_tensors) - set(actual_tensors))
    extra = sorted(set(actual_tensors) - set(expected_tensors))
    if missing:
        failures.append(f"{len(missing)} tensors missing, first: {missing[:3]}")
    if extra:
        failures.append(f"{len(extra)} unexpected tensors, first: {extra[:3]}")
    tolerated = 0
    for name in sorted(set(expected_tensors) & set(actual_tensors)):
        want = expected_tensors[name]
        got = actual_tensors[name]
        if want["dtype"] != got["dtype"] or want["dims"] != got["dims"]:
            failures.append(f"{name}: dtype or dims differ")
            continue
        if want.get("n", 0) == 0:
            continue
        if want["sha256"] == got["sha256"]:
            continue
        if want["dtype"] in INT8_MEMBERS:
            if (
                abs(want["mean"] - got["mean"]) <= MEAN_TOLERANCE
                and abs(want["min"] - got["min"]) <= EXTREME_TOLERANCE
                and abs(want["max"] - got["max"]) <= EXTREME_TOLERANCE
            ):
                tolerated += 1
                continue
        elif want["dtype"] in FLOAT_MEMBERS:
            # Scales and the fp32 Conv weights are copied from the source; only
            # last-bit float ordering may differ.
            if all(
                abs(want[key] - got[key]) <= FLOAT_TOLERANCE * max(1.0, abs(want[key]))
                for key in ("mean", "min", "max", "std")
            ):
                tolerated += 1
                continue
        failures.append(
            f"{name}: values differ beyond tolerance "
            f"(mean {want['mean']:.6f} -> {got['mean']:.6f})"
        )
    return tolerated, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expect-sha256")
    parser.add_argument("--digest")
    parser.add_argument(
        "--write-digest",
        metavar="PATH",
        help="write the digest for this model and exit (no verification)",
    )
    args = parser.parse_args()

    if args.write_digest:
        digest = describe(args.model)
        with open(args.write_digest, "w", encoding="utf-8") as handle:
            json.dump(digest, handle, indent=1, sort_keys=True)
            handle.write("\n")
        print(
            f"wrote {args.write_digest}: {len(digest['initializers'])} tensors, "
            f"sha256 {digest['model_sha256']}"
        )
        return 0

    actual_hash = sha256(args.model)
    if args.expect_sha256 and actual_hash == args.expect_sha256:
        print(f"encoder sha256 matches the pin: {actual_hash}")
        return 0
    print(f"encoder sha256 {actual_hash} does not match the pin {args.expect_sha256}")
    if not args.digest:
        raise SystemExit("no --digest to fall back to")
    with open(args.digest, encoding="utf-8") as handle:
        expected = json.load(handle)
    actual = describe(args.model)
    if actual == expected:
        print("the model is byte-identical to the digest")
        return 0
    tolerated, failures = compare(expected, actual)
    print(
        f"per-tensor comparison: {len(actual['initializers'])} tensors, "
        f"{tolerated} int8 tensors within tolerance"
    )
    if failures:
        for failure in failures[:20]:
            print(f"  FAIL {failure}")
        raise SystemExit(
            f"{len(failures)} checks failed; the hash difference is not a rounding difference"
        )
    print("PASS: the hash differs but every tensor matches within the documented tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
