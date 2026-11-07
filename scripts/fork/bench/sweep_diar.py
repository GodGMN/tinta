#!/usr/bin/env python3
"""Drive the post-clustering sweep over cached embeddings (fast, no model runs).

Each call runs the harness `postcluster` command, which is pure Rust over a
cached embedding file, then scores the result with pyannote.metrics.
"""
import argparse
import itertools
import json
import os
import subprocess
import sys

ROOT = os.environ.get("ROOT", "/home/sergio/code/evidence/tinta-fork/18")
BENCH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
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
    ap.add_argument("--merge", default="0.20,0.25,0.30,0.35")
    ap.add_argument("--loyalty", default="0,0.3")
    ap.add_argument("--min-share", default="0.01,0.02,0.03")
    args = ap.parse_args()

    meetings = args.meetings.split(",")
    merges = [float(x) for x in args.merge.split(",")]
    loyalties = [float(x) for x in args.loyalty.split(",")]
    shares = [float(x) for x in args.min_share.split(",")]

    rows = []
    for merge, loyalty, share in itertools.product(merges, loyalties, shares):
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
            name = f"pc-{meeting}-{args.tag}-m{merge:g}-l{loyalty:g}-s{share:g}"
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
                "--min-loyalty", str(loyalty),
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
        rows.append((merge, loyalty, share, mean, counts, ders))

    print("| merge dist | min loyalty | min share | mean DER | spk/ref | per meeting |")
    print("|---|---|---|---|---|---|")
    for merge, loyalty, share, mean, counts, ders in rows:
        per = " ".join(f"{m}={d * 100:.1f}%" for d, _, _, m in ders)
        print(
            f"| {merge:.2f} | {loyalty:.2f} | {share * 100:.0f}% | "
            f"{mean * 100:.1f}% | {counts} | {per} |"
        )
    best = min(rows, key=lambda r: r[3])
    print(f"BEST merge={best[0]:g} loyalty={best[1]:g} share={best[2]:g} der={best[3]:.4f}")
    if args.table:
        with open(args.table, "w") as f:
            f.write("| merge dist | min loyalty | min share | mean DER | spk/ref | per meeting |\n")
            f.write("|---|---|---|---|---|---|\n")
            for merge, loyalty, share, mean, counts, ders in rows:
                per = " ".join(f"{m}={d * 100:.1f}%" for d, _, _, m in ders)
                f.write(
                    f"| {merge:.2f} | {loyalty:.2f} | {share * 100:.0f}% | "
                    f"{mean * 100:.1f}% | {counts} | {per} |\n"
                )


if __name__ == "__main__":
    main()
