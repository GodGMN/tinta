#!/usr/bin/env python3
"""Download the pinned fp32 Parakeet TDT 0.6b v3 export for CI.

This is the cross-platform equivalent of `fetch-model-fp32.sh`: the same five
files, the same Hugging Face revision and the same SHA-256 pins. It also lays
the decoder, joiner and tokens out under the engine manifest's int8 names, so
`requantize-encoder.py` only has to write `encoder.int8.onnx` into the same
directory.

The CI job builds the requantized encoder because it is not hosted anywhere
yet; only the fp32 source is downloaded. Requested by the task #23 re-review.

Usage:
  fetch-model-fp32.py --dest .asr-models
"""

import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

REVISION = "5fcc0122fe87e8365e339e467af90c67b746218f"
REPOSITORY = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3"
BASE = f"https://huggingface.co/{REPOSITORY}/resolve/{REVISION}"

FILES = {
    "encoder.onnx": "36b758364dfddce472b9a9f75bf6049bbd35c886d7927e3eda80d4ceb3348512",
    "encoder.weights": "3af3f51af5f2d01dbbf5af47d42c7962a2c205f11004254bb4f2b979862f39a8",
    "decoder.onnx": "caed31af1bb08edcc09a01a1c3cbb4d4eb5bfdb9c4e5f307012f84bbf5dbfdbc",
    "joiner.onnx": "0f294e5ba9021a3d80e492de94abe4c116f0afce78452c7debb5dd07690d0340",
    "tokens.txt": "d58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d",
}

# The engine manifest names every transcription file `.int8.onnx`, including the
# decoder and joiner. The requantized encoder is written by the quantizer; the
# other three are copied from the fp32 package.
LAYOUT = {
    "decoder.onnx": "decoder.int8.onnx",
    "joiner.onnx": "joiner.int8.onnx",
    "tokens.txt": "tokens.txt",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, path):
    tmp = path + ".part"
    for attempt in range(1, 6):
        try:
            print(f"downloading {url} (attempt {attempt})", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "tinta-ci"})
            with urllib.request.urlopen(request, timeout=300) as source:
                with open(tmp, "wb") as target:
                    shutil.copyfileobj(source, target, 1 << 20)
            os.replace(tmp, path)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            print(f"  failed: {error}", flush=True)
            if os.path.exists(tmp):
                os.unlink(tmp)
            if attempt == 5:
                raise
            time.sleep(5 * attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", required=True, help="directory for models and the layout")
    args = parser.parse_args()

    package = os.path.join(args.dest, "parakeet-tdt-0.6b-v3")
    layout = os.path.join(args.dest, "requant-root", "parakeet-tdt-0.6b-v3-int8")
    os.makedirs(package, exist_ok=True)
    os.makedirs(layout, exist_ok=True)

    for name, expected in FILES.items():
        path = os.path.join(package, name)
        if os.path.isfile(path) and sha256(path) == expected:
            print(f"{name}: cached, sha256 {expected}")
            continue
        download(f"{BASE}/{name}", path)
        actual = sha256(path)
        if actual != expected:
            raise SystemExit(f"{name}: sha256 {actual} does not match {expected}")
        print(f"{name}: {os.path.getsize(path)} B, sha256 {actual}")

    for source, target in LAYOUT.items():
        src = os.path.join(package, source)
        dst = os.path.join(layout, target)
        if os.path.isfile(dst) and sha256(dst) == FILES[source]:
            continue
        shutil.copyfile(src, dst)
        print(f"layout: {target} <- {source}")

    provenance = os.path.join(args.dest, "provenance.txt")
    with open(provenance, "w", encoding="utf-8") as handle:
        handle.write(f"repository: https://huggingface.co/{REPOSITORY}\n")
        handle.write(f"revision:   {REVISION}\n")
        for name, expected in FILES.items():
            handle.write(f"sha256 {expected}  {name}\n")
    print(f"fp32 export ready under {package}; requant layout under {layout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
