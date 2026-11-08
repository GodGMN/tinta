#!/usr/bin/env python3
"""The task #23 1,000-utterance AMI IHM sample: generator and scorer.

`make` reproduces `evidence/tinta-fork/23/data/sample-1000.{json,tsv}`: a
deterministic `random.Random(seed).sample(rows, 1000)` of the full AMI IHM test
rows from task #19 (`rows.json`, 12,643 rows). The order is the sampled order,
so the same seed gives the same file.

`score` scores the sample decodes of several models and reports the paired
delta against a baseline with a percentile bootstrap **clustered by true
speaker**. AMI IHM test has 16 speakers, so a speaker-cluster bootstrap has 16
clusters; the 63 "clusters" the first draft used were meeting x speaker pairs,
which understates the interval. `--cluster meeting` is kept for contrast.

Text is normalized with the same Whisper normalizer as `score.py`, on both
sides, and WER is summed per-utterance edits over summed reference words.

Usage:
  ihm-sample.py make --rows ROOT19/data/ihm-baseline-full/rows.json \
                     --out-json data/sample-1000.json --out-tsv data/sample-1000.tsv
  ihm-sample.py score --rows ROOT19/data/ihm-baseline-full/rows.json \
                      --sample data/sample-1000.json \
                      --hyp "int8=...jsonl,fp32=...jsonl,requant=...jsonl" \
                      --baseline int8 --out results/ihm-sample-score.json
"""

import argparse
import importlib.util
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def score_module():
    """`score.py` next to this file, for the shared normalizer and counts."""
    spec = importlib.util.spec_from_file_location(
        "segmentation_score", os.path.join(HERE, "score.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score = score_module()


def pairs(spec):
    return dict(item.split("=", 1) for item in spec.split(",") if item)


def make_sample(rows, count, seed):
    """The sample in the order `random.sample` returns it."""
    return random.Random(seed).sample(rows, count)


def write_sample(sample, out_json, out_tsv):
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    document = [
        {
            "id": row["id"],
            "meeting": row["meeting"],
            "text": row["text"],
            "begin": row["begin"],
            "end": row["end"],
            "wav": row["wav"],
        }
        for row in sample
    ]
    with open(out_json, "w") as handle:
        json.dump(document, handle, indent=1)
    with open(out_tsv, "w") as handle:
        for row in document:
            handle.write(f"{row['id']}\t{row['wav']}\t\t\n")


def row_counts(rows, hypotheses):
    """One cached edit-distance row per utterance.

    Normalizing and aligning 1,000 utterances once per model is cheap; doing it
    again inside 2,000 bootstrap resamples is not. The bootstrap only sums these
    rows.
    """
    counts = []
    missing = 0
    for row in rows:
        hypothesis = hypotheses.get(row["id"])
        if hypothesis is None:
            missing += 1
            counts.append({"errors": 0, "ref_words": 0, "hyp_words": 0, "del": 0, "ins": 0, "sub": 0})
            continue
        counts.append(score.counts(row["text"], hypothesis.get("text", "")))
    return counts, missing


def total_counts(counts, missing=0):
    total = {"errors": 0, "ref_words": 0, "hyp_words": 0, "del": 0, "ins": 0, "sub": 0}
    for item in counts:
        for key in total:
            total[key] += item[key]
    total["wer"] = total["errors"] / total["ref_words"] if total["ref_words"] else 0.0
    total["missing"] = missing
    return total


def clusters_of(rows, key):
    grouped = {}
    for index, row in enumerate(rows):
        grouped.setdefault(row[key], []).append(index)
    return [grouped[name] for name in sorted(grouped)]


def bootstrap_delta(rows, counts_a, counts_b, key, resamples, seed):
    """Paired percentile bootstrap of WER(a) - WER(b), resampling `key` clusters.

    Every resample draws whole clusters with replacement, so all utterances of
    a speaker (or a meeting) move together; that is what makes the interval
    honest for a sample of 1,000 rows from 16 speakers.
    """
    clusters = clusters_of(rows, key)
    rng = random.Random(seed)
    deltas = []
    for _ in range(resamples):
        drawn = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        a = {"errors": 0, "ref_words": 0}
        b = {"errors": 0, "ref_words": 0}
        for cluster in drawn:
            for index in cluster:
                a["errors"] += counts_a[index]["errors"]
                a["ref_words"] += counts_a[index]["ref_words"]
                b["errors"] += counts_b[index]["errors"]
                b["ref_words"] += counts_b[index]["ref_words"]
        if b["ref_words"] and a["ref_words"]:
            deltas.append(a["errors"] / a["ref_words"] - b["errors"] / b["ref_words"])
    deltas.sort()
    lo = deltas[int(0.025 * (len(deltas) - 1))]
    hi = deltas[int(0.975 * (len(deltas) - 1))]
    return {
        "lo": lo,
        "hi": hi,
        "resamples": resamples,
        "clusters": len(clusters),
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    make = sub.add_parser("make", help="write the deterministic sample")
    make.add_argument("--rows", required=True)
    make.add_argument("--count", type=int, default=1000)
    make.add_argument("--seed", type=int, default=23)
    make.add_argument("--out-json", required=True)
    make.add_argument("--out-tsv", required=True)

    scorer = sub.add_parser("score", help="score sample decodes")
    scorer.add_argument("--rows", required=True, help="the full IHM rows.json")
    scorer.add_argument("--sample", required=True, help="sample-1000.json")
    scorer.add_argument("--hyp", required=True, help="label=jsonl,...")
    scorer.add_argument("--baseline", default=None)
    scorer.add_argument("--bootstrap", type=int, default=2000)
    scorer.add_argument("--seed", type=int, default=23)
    scorer.add_argument("--out")
    args = parser.parse_args()

    if args.command == "make":
        rows = json.load(open(args.rows))
        sample = make_sample(rows, args.count, args.seed)
        write_sample(sample, args.out_json, args.out_tsv)
        print(f"{len(sample)} rows -> {args.out_json}, {args.out_tsv}")
        return 0

    full = {row["id"]: row for row in json.load(open(args.rows))}
    sample = json.load(open(args.sample))
    rows = []
    for row in sample:
        # The sample JSON carries no speaker (it is the batch manifest shape);
        # take the true speaker from the full row of the same id.
        rows.append({**row, "speaker": full[row["id"]]["speaker"]})
    paths = pairs(args.hyp)
    hypotheses = {name: score.read_jsonl(path) for name, path in paths.items()}
    cached = {
        name: row_counts(rows, hypothesis)
        for name, hypothesis in hypotheses.items()
    }

    result = {
        "sample": args.sample,
        "rows": len(rows),
        "speakers": len({row["speaker"] for row in rows}),
        "meeting_speakers": len({(row["meeting"], row["speaker"]) for row in rows}),
        "variants": {},
    }
    print(
        f"\n### AMI IHM sample, {len(rows)} utterances, {result['speakers']} true speakers "
        f"({result['meeting_speakers']} meeting x speaker pairs)\n"
    )
    print("| Model | WER | del / ins / sub | Errors / ref words |")
    print("|---|---|---|---|")
    for name in hypotheses:
        counts, missing = cached[name]
        total = total_counts(counts, missing)
        result["variants"][name] = total
        print(
            f"| {name} | **{100 * total['wer']:.2f}%** | "
            f"{total['del']} / {total['ins']} / {total['sub']} | "
            f"{total['errors']} / {total['ref_words']} |"
        )

    if args.baseline:
        if args.baseline not in hypotheses:
            raise SystemExit(f"baseline {args.baseline} is not one of {list(hypotheses)}")
        print(
            f"\nPaired delta against {args.baseline}, percentile bootstrap "
            f"({args.bootstrap} resamples, seed {args.seed}):\n"
        )
        print("| Model | Delta | 95% CI by true speaker (16) | 95% CI by meeting |")
        print("|---|---|---|---|")
        base_counts, base_missing = cached[args.baseline]
        base_wer = total_counts(base_counts, base_missing)["wer"]
        for name in hypotheses:
            if name == args.baseline:
                continue
            counts, missing = cached[name]
            point = total_counts(counts, missing)["wer"] - base_wer
            speaker = bootstrap_delta(
                rows, counts, base_counts, "speaker", args.bootstrap, args.seed
            )
            meeting = bootstrap_delta(
                rows, counts, base_counts, "meeting", args.bootstrap, args.seed
            )
            result["variants"][name]["delta_vs_baseline"] = {
                "baseline": args.baseline,
                "point": point,
                "speaker": speaker,
                "meeting": meeting,
            }
            print(
                f"| {name} | {100 * point:+.2f} points | "
                f"{100 * speaker['lo']:+.2f} to {100 * speaker['hi']:+.2f} | "
                f"{100 * meeting['lo']:+.2f} to {100 * meeting['hi']:+.2f} |"
            )

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
