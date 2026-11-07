#!/usr/bin/env python3
"""Drive the post-clustering sweep over cached embeddings (fast, no model runs).

Each call runs the harness `postcluster` command, which is pure Rust over a
cached embedding file, then scores the result with pyannote.metrics. The grid
below is the one documented in docs/fork/diarization.md: merge distance
0.10-0.30 in steps of 0.05 (the doc table shows an excerpt) and 2/3/5% of
speech for the dissolve step.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

ROOT = os.environ.get("ROOT", "/home/sergio/code/evidence/tinta-fork/18")
BENCH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BIN = os.path.abspath(os.path.join(BENCH, "scripts/fork/bench/target/release/tinta-bench"))
PY = os.path.join(ROOT, "venv/bin/python")
SCORE = os.path.abspath(os.path.join(BENCH, "scripts/fork/bench/score.py"))


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meetings", required=True, help="comma separated")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--diar-tag", default=None,
                    help="model whose over-clustered partition is the starting point")
    ap.add_argument("--table", default=None)
    ap.add_argument("--merge", default="0.10,0.15,0.20,0.25,0.30")
    ap.add_argument("--min-share", default="0.02,0.03,0.05")
    args = ap.parse_args()

    meetings = args.meetings.split(",")
    merges = [float(x) for x in args.merge.split(",")]
    shares = [float(x) for x in args.min_share.split(",")]

    rows = []
    for merge, share in itertools.product(merges, shares):
        ders = []
        for meeting in meetings:
            emb = os.path.join(ROOT, "results", f"emb-{meeting}-{args.tag}.json")
            raw = os.path.join(
                ROOT, "results",
                f"diar-{meeting}-{args.diar_tag or args.tag}.json",
            )
            if not (os.path.exists(emb) and os.path.exists(raw)):
                print(f"missing {emb} or {raw}", file=sys.stderr)
                sys.exit(1)
            name = f"pc-{meeting}-{args.tag}-m{merge:g}-s{share:g}"
            out = os.path.join(ROOT, "results", name + ".json")
            run([
                "nice", "-n", "19", BIN, "postcluster",
                "--wav", os.path.join(ROOT, "data", f"{meeting}.Mix-Headset.wav"),
                "--diar", raw,
                "--embedding", os.path.join(ROOT, "models",
                    "wespeaker_en_voxceleb_resnet34.onnx" if args.tag == "wespeaker"
                    else "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"),
                "--threads", "2", "--spin", "0",
                "--merge-threshold", str(merge), "--min-share", str(share),
                "--embeddings", emb,
                "--output", out,
            ])
            scored = os.path.join(ROOT, "results", "score-" + name + ".json")
            run([
                PY, SCORE, "score",
                "--ref", os.path.join(ROOT, "results", f"ref-{meeting}.json"),
                "--result", out,
                "--small-cluster-fraction", "0",
                "--out", scored,
            ])
            d = json.load(open(scored))
            ders.append((d["der"], d["hyp_speakers_filtered"], d["ref_speakers"], meeting))
            # Oracle on the final output: residual segmentation floor.
            oracle = os.path.join(ROOT, "results", "oracle-" + name + ".json")
            run([
                PY, SCORE, "oracle",
                "--ref", os.path.join(ROOT, "results", f"ref-{meeting}.json"),
                "--result", out, "--out", oracle,
            ])
        mean = sum(x[0] for x in ders) / len(ders)
        counts = " ".join(f"{x[1]}/{x[2]}" for x in ders)
        rows.append((merge, share, mean, counts, ders))

    print("| merge dist | min share | mean DER | spk/ref | per meeting |")
    print("|---|---|---|---|---|")
    for merge, share, mean, counts, ders in rows:
        per = " ".join(f"{m}={d * 100:.1f}%" for d, _, _, m in ders)
        print(f"| {merge:.2f} | {share * 100:.0f}% | {mean * 100:.1f}% | {counts} | {per} |")
    best = min(rows, key=lambda r: r[2])
    print(f"BEST merge={best[0]:g} share={best[1]:g} der={best[2]:.4f}")
    if args.table:
        with open(args.table, "w") as f:
            f.write("| merge dist | min share | mean DER | spk/ref | per meeting |\n")
            f.write("|---|---|---|---|---|\n")
            for merge, share, mean, counts, ders in rows:
                per = " ".join(f"{m}={d * 100:.1f}%" for d, _, _, m in ders)
                f.write(f"| {merge:.2f} | {share * 100:.0f}% | {mean * 100:.1f}% | {counts} | {per} |\n")


if __name__ == "__main__":
    main()
