#!/usr/bin/env python3
"""Decode the committed AMI clip with `decode-clip` and gate the result.

The ASR requant portability job runs this on every runner. It does not assert
the exact string: kernel-order float differences legitimately flip `T V` /
`TV`. The failure mode this catches is an empty or truncated decode, so the
gates are a minimum word count and a normalized word error rate against the
committed fp32 reference text.

Usage:
  check-clip.py --binary target/debug/examples/decode-clip \\
                --models-root .asr-models/requant-root \\
                --wav crates/tinta-engine-rs/tests/fixtures/es2004b-235.4-248.8.wav \\
                --reference crates/tinta-engine-rs/tests/fixtures/es2004b-235.4-248.8.ref.txt
"""

import argparse
import json
import os
import re
import subprocess
import sys


def normalize(text):
    """Lowercase, turn punctuation into spaces, collapse whitespace."""
    text = text.lower().replace("'", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.split()


def edit_distance(reference, hypothesis):
    previous = list(range(len(hypothesis) + 1))
    for i, ref_word in enumerate(reference, start=1):
        current = [i]
        for j, hyp_word in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (ref_word != hyp_word),  # substitution
                )
            )
        previous = current
    return previous[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, help="the decode-clip executable")
    parser.add_argument("--models-root", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--min-words", type=int, default=28)
    parser.add_argument("--max-wer", type=float, default=0.10)
    args = parser.parse_args()

    command = [
        args.binary,
        "--wav",
        args.wav,
        "--models-root",
        args.models_root,
        "--threads",
        str(args.threads),
    ]
    binary = args.binary
    if not os.path.isfile(binary) and os.path.isfile(binary + ".exe"):
        binary = binary + ".exe"
        command[0] = binary
    result = subprocess.run(command, capture_output=True, text=True)
    sys.stderr.write(result.stderr)
    if result.returncode != 0:
        raise SystemExit(f"{args.binary} exited {result.returncode}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"{args.binary} printed nothing")
    decoded = json.loads(lines[-1])

    reference_words = normalize(open(args.reference, encoding="utf-8").read())
    hypothesis_words = normalize(decoded["text"])
    distance = edit_distance(reference_words, hypothesis_words)
    wer = distance / max(1, len(reference_words))

    print(f"runner decode: {decoded['words']} words, {decoded['audio_seconds']:.2f} s")
    print(f"text: {decoded['text']}")
    print(
        f"reference: {len(reference_words)} normalized words, "
        f"{distance} errors, WER {wer * 100:.1f} %"
    )

    failures = []
    if decoded["words"] < args.min_words:
        failures.append(
            f"{decoded['words']} words is below the {args.min_words}-word floor"
        )
    if wer > args.max_wer:
        failures.append(
            f"normalized WER {wer * 100:.1f} % is above the {args.max_wer * 100:.0f} % ceiling"
        )
    if failures:
        raise SystemExit("; ".join(failures))
    print("PASS: the requantized encoder produces the reference sentence on this runner")


if __name__ == "__main__":
    main()
