#!/usr/bin/env python3
"""Build the diarization result tables from the scored JSON files.

Reads the runs produced by `diarization.sh` in $ROOT/results and prints:

  1. model x clustering variant, per test meeting (DER, collar, speakers).
  2. speaker-count accuracy per model and configuration.
  3. the oracle mapping bound (raw clusters vs final clusters).
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.environ.get("ROOT", "/home/sergio/code/evidence/tinta-fork/18")
BENCH = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
PY = os.path.join(ROOT, "venv/bin/python")
SCORE = os.path.join(BENCH, "scripts/fork/bench/score.py")

MODELS = ["campplus", "wespeaker"]
MODEL_NAMES = {
    "campplus": "3D-Speaker CAMPPlus (zh-cn, 512-d)",
    "wespeaker": "WeSpeaker ResNet34 (en VoxCeleb, 256-d)",
}


def score(ref, result, out, fraction):
    if not os.path.exists(out) or os.path.getmtime(result) > os.path.getmtime(out):
        subprocess.run(
            [PY, SCORE, "score", "--ref", ref, "--result", result,
             "--small-cluster-fraction", str(fraction), "--out", out],
            check=True, stdout=subprocess.DEVNULL,
        )
    return json.load(open(out))


def oracle(ref, result, out):
    if not os.path.exists(out) or os.path.getmtime(result) > os.path.getmtime(out):
        subprocess.run(
            [PY, SCORE, "oracle", "--ref", ref, "--result", result, "--out", out],
            check=True, stdout=subprocess.DEVNULL,
        )
    return json.load(open(out))


def pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", type=float, default=0.25)
    ap.add_argument("--loyalty", type=float, default=0.5)
    ap.add_argument("--share", type=float, default=0.02)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--meetings", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meetings = args.meetings.split(",")
    models = args.models.split(",")
    lines = []

    lines.append("### Model x clustering")
    lines.append("")
    lines.append(
        "| Meeting | Model | Variant | Clusters | DER no collar | DER 0.25 s collar "
        "| miss / FA / conf | Oracle DER |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")

    counts = {}
    sums = {}
    for model in models:
        for meeting in meetings:
            ref = os.path.join(ROOT, "results", f"ref-{meeting}.json")
            raw = os.path.join(ROOT, "results", f"diar-{meeting}-{model}.json")
            final = os.path.join(
                ROOT, "results",
                f"pc-{meeting}-{model}-m{args.merge:g}-l{args.loyalty:g}-s{args.share:g}.json",
            )
            # Raw over-clustered output, scored the way task #2 scored it.
            r0 = score(ref, raw, os.path.join(ROOT, "results", f"score-table-raw0-{meeting}-{model}.json"), 0.0)
            rf = score(ref, raw, os.path.join(ROOT, "results", f"score-table-raw2-{meeting}-{model}.json"), 0.02)
            fin = score(ref, final, os.path.join(ROOT, "results", f"score-table-fin-{meeting}-{model}.json"), 0.0)
            oraw = oracle(ref, raw, os.path.join(ROOT, "results", f"oracle-table-raw-{meeting}-{model}.json"))
            ofin = oracle(ref, final, os.path.join(ROOT, "results", f"oracle-table-fin-{meeting}-{model}.json"))

            for variant, der, collar, d, orc, clusters in (
                ("raw th 0.7", r0["der"], r0["der_collar025"], r0, oraw["oracle_der"], d0(r0)),
                ("raw + time filter 2%", rf["der_filtered"], rf["der_filtered_collar025"], rf,
                 oraw["oracle_der"], d0(rf)),
                ("post-cluster", fin["der"], fin["der_collar025"], fin, ofin["oracle_der"], d0(fin)),
            ):
                det = d["der_detail"]
                lines.append(
                    f"| {meeting} | {model} | {variant} | {clusters} | "
                    f"{pct(der)} | {pct(collar)} | "
                    f"{det['miss'] * 100:.1f} / {det['false_alarm'] * 100:.1f} / "
                    f"{det['confusion'] * 100:.1f} | {pct(orc)} |"
                )
                sums.setdefault((model, variant), []).append(der)
            counts.setdefault(model, []).append(
                (meeting, d0(fin), r0["ref_speakers"], d0(r0), d0(rf))
            )

    lines.append("")
    lines.append("| Model | Variant | Mean DER no collar |")
    lines.append("|---|---|---|")
    for (model, variant), vals in sums.items():
        lines.append(f"| {model} | {variant} | {pct(sum(vals) / len(vals))} |")

    lines.append("")
    lines.append("### Speaker count")
    lines.append("")
    lines.append("| Model | Meeting | post-cluster | raw th 0.7 | raw + time filter | reference | exact |")
    lines.append("|---|---|---|---|---|---|---|")
    exact = {}
    for model in models:
        for meeting, n_fin, n_ref, n_raw, n_filt in counts[model]:
            ok = n_fin == n_ref
            exact.setdefault(model, []).append(ok)
            lines.append(
                f"| {model} | {meeting} | {n_fin} | {n_raw} | {n_filt} | {n_ref} | "
                f"{'yes' if ok else 'no'} |"
            )
    for model in models:
        ok = sum(exact.get(model, []))
        total = len(exact.get(model, []))
        lines.append(f"| {model} | **all** | | | | | **{ok}/{total}** |")

    lines.append("")
    lines.append("### Oracle mapping bound")
    lines.append("")
    lines.append("| Model | Meeting | Raw DER (clusters) | Raw oracle | Final DER (clusters) | Final oracle |")
    lines.append("|---|---|---|---|---|---|")
    for model in models:
        for meeting in meetings:
            ref = os.path.join(ROOT, "results", f"ref-{meeting}.json")
            raw = os.path.join(ROOT, "results", f"diar-{meeting}-{model}.json")
            final = os.path.join(
                ROOT, "results",
                f"pc-{meeting}-{model}-m{args.merge:g}-l{args.loyalty:g}-s{args.share:g}.json",
            )
            r0 = score(ref, raw, os.path.join(ROOT, "results", f"score-table-raw0-{meeting}-{model}.json"), 0.0)
            fin = score(ref, final, os.path.join(ROOT, "results", f"score-table-fin-{meeting}-{model}.json"), 0.0)
            oraw = oracle(ref, raw, os.path.join(ROOT, "results", f"oracle-table-raw-{meeting}-{model}.json"))
            ofin = oracle(ref, final, os.path.join(ROOT, "results", f"oracle-table-fin-{meeting}-{model}.json"))
            lines.append(
                f"| {model} | {meeting} | {pct(r0['der'])} ({d0(r0)}) | {pct(oraw['oracle_der'])} | "
                f"{pct(fin['der'])} ({d0(fin)}) | {pct(ofin['oracle_der'])} |"
            )

    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")


def d0(d):
    return d.get("hyp_speakers_filtered", d.get("hyp_speakers"))


if __name__ == "__main__":
    main()
