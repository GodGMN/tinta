#!/usr/bin/env python3
"""Build the golden clustering fixtures from the cached task #18 embeddings.

This script is not part of the run path. It reads the per-unit embeddings that
`diarization.sh raw` cached in `$ROOT/results/emb-<meeting>-campplus.json`,
takes a small extract of units, quantizes every embedding to i8 (scale = that
unit's max absolute value) and writes

    tests/data/golden-<meeting>.json

The extract is chosen to exercise both repair steps: units of a split cluster
that must merge, units of the four dominant clusters that must survive, and one
unit from a small cluster that must dissolve into its nearest voice. Reference
labels are used only to pick units for the fixture, never by the algorithm.

Expected labels are produced by a Python mirror of `recluster`; `cargo test`
checks the Rust implementation against them. Regenerate with:

    ROOT=/home/sergio/code/evidence/tinta-fork/18 \
      python3 scripts/fork/bench/make_golden_fixtures.py
"""
import base64
import json
import math
import os
import sys
from collections import Counter, defaultdict

ROOT = os.environ.get("ROOT", "/home/sergio/code/evidence/tinta-fork/18")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "tests", "data")

# (cluster id, number of units to take), in the order they are picked.
EXTRACTS = {
    "ES2004b": [(6, 5), (19, 5), (3, 1), (1, 1), (21, 1), (0, 1)],
    "ES2011c": [(15, 1), (1, 1), (12, 1), (10, 2), (7, 1), (0, 1)],
}


def l2(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n > 1e-12 else v


def cos(a, b):
    d = sum(x * y for x, y in zip(a, b))
    return math.inf if not math.isfinite(d) else max(0.0, min(2.0, 1.0 - d))


def centroid(units, members):
    dim = len(units[0]["e"])
    acc = [0.0] * dim
    total = 0.0
    for i in members:
        w = max(units[i]["sec"], 1e-3)
        total += w
        for k in range(dim):
            acc[k] += units[i]["e"][k] * w
    if total > 0:
        acc = [x / total for x in acc]
    return l2(acc)


def recluster(units, merge_distance=0.15, min_share=0.03):
    for u in units:
        u["e"] = l2(u["e"])
    by = defaultdict(list)
    for i, u in enumerate(units):
        by[u["cluster"]].append(i)
    live = []
    for c in sorted(by):
        m = by[c]
        live.append({"members": m, "sec": sum(units[i]["sec"] for i in m), "c": centroid(units, m)})
    n = len(live)
    dist = [[math.inf] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist[i][j] = dist[j][i] = cos(live[i]["c"], live[j]["c"])
    active = [True] * n
    merges = 0
    while sum(active) > 1:
        best = (None, None, math.inf)
        for i in range(n):
            if not active[i]:
                continue
            for j in range(i + 1, n):
                if active[j] and dist[i][j] < best[2]:
                    best = (i, j, dist[i][j])
        i, j, d = best
        if not d < merge_distance:
            break
        keep, gone = min(i, j), max(i, j)
        live[keep]["members"] += live[gone]["members"]
        live[keep]["members"].sort()
        live[keep]["sec"] += live[gone]["sec"]
        live[keep]["c"] = centroid(units, live[keep]["members"])
        active[gone] = False
        for k in range(n):
            if active[k] and k != keep:
                dist[keep][k] = dist[k][keep] = cos(live[keep]["c"], live[k]["c"])
        merges += 1

    kept = [i for i in range(n) if active[i]]
    labels = [None] * len(units)
    for label, i in enumerate(kept):
        for m in live[i]["members"]:
            labels[m] = label
    dissolved = 0
    if len(kept) > 1 and min_share > 0:
        total = max(1e-6, sum(u["sec"] for u in units))
        largest = max(kept, key=lambda i: live[i]["sec"])
        drop = [i for i in kept if i != largest and live[i]["sec"] < min_share * total]
        if drop:
            dissolved = len(drop)
            keep = [i for i in kept if i not in drop]
            # Rebuild kept labels with compressed positions before labelling
            # the reassigned units with those same positions.
            for label, k in enumerate(keep):
                for m in live[k]["members"]:
                    labels[m] = label
            for k in drop:
                for m in live[k]["members"]:
                    labels[m] = min(range(len(keep)), key=lambda lab: cos(units[m]["e"], live[keep[lab]]["c"]))
    shares = defaultdict(float)
    for m, label in enumerate(labels):
        shares[label] += units[m]["sec"]
    order = sorted(shares.items(), key=lambda kv: (-kv[1], kv[0]))
    remap = {label: rank for rank, (label, _) in enumerate(order)}
    return {
        "clusters_before": len(by),
        "clusters_after": len(remap),
        "merges": merges,
        "dissolved": dissolved,
        "labels": [remap[label] for label in labels],
    }


def ref_speaker(unit, segments):
    best = (None, 0.0)
    for s in segments:
        overlap = max(0.0, min(unit["end"], s["end"]) - max(unit["start"], s["start"]))
        if overlap > best[1]:
            best = (s["speaker"], overlap)
    return best[0] if best[1] > 0 else None


def quantize(e):
    scale = max(abs(x) for x in e) or 1.0
    return [max(-127, min(127, int(x / scale * 127 + (0.5 if x >= 0 else -0.5)))) for x in e]


def build(meeting):
    with open(os.path.join(ROOT, "results", f"emb-{meeting}-campplus.json")) as f:
        cached = json.load(f)
    with open(os.path.join(ROOT, "results", f"ref-{meeting}.json")) as f:
        ref = json.load(f)
    units = [
        {
            "start": u["start"],
            "end": u["end"],
            "cluster": u["cluster"],
            "e": e,
            "sec": max(0.0, u["end"] - u["start"]),
            "spk": ref_speaker(u, ref["segments"]),
        }
        for u, e in zip(cached["units"], cached["embeddings"])
    ]
    for u in units:
        u["en"] = l2(u["e"][:])
    by_cluster = defaultdict(list)
    for u in units:
        by_cluster[u["cluster"]].append(u)

    picked = []
    for cluster, count in EXTRACTS[meeting]:
        members = by_cluster[cluster]
        majority = Counter(u["spk"] for u in members).most_common(1)[0][0]
        usable = [u for u in members if u["spk"] == majority]
        center = centroid(units, [i for i, u in enumerate(units) if u["cluster"] == cluster and u["spk"] == majority])
        picked += sorted(usable, key=lambda u: cos(u["en"], center))[:count]
    picked.sort(key=lambda u: u["start"])

    body = []
    for u in picked:
        q = quantize(u["e"])
        body.append(
            {
                "start": u["start"],
                "end": u["end"],
                "cluster": u["cluster"],
                # Reference speaker of the unit: fixture selection and the
                # assignment check only, never read by the algorithm.
                "reference": u["spk"],
                "embedding": base64.b64encode(bytes(x & 0xFF for x in q)).decode("ascii"),
            }
        )

    # Expected output from the quantized values, which is what the Rust test
    # actually feeds to `recluster`.
    decoded = [
        {
            "start": u["start"],
            "end": u["end"],
            "cluster": u["cluster"],
            "e": [x / 127.0 for x in quantize(u["e"])],
            "sec": max(0.0, u["end"] - u["start"]),
        }
        for u in picked
    ]
    expected = recluster(decoded)

    doc = {
        "meeting": meeting,
        "source": f"evidence/tinta-fork/18/results/emb-{meeting}-campplus.json",
        "note": (
            "Small extract of cached CAMPPlus per-unit embeddings; selected to cover "
            "centroid merge, a dissolved small cluster and four surviving speakers. "
            "Embeddings are i8-quantized per unit (scale = max abs) and base64-encoded "
            "as 192 bytes. Generated by make_golden_fixtures.py."
        ),
        "dim": len(decoded[0]["e"]),
        "units": body,
        "expected": expected,
    }
    path = os.path.join(OUT, f"golden-{meeting}.json")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w") as f:
        json.dump(doc, f, separators=(",", ":"))
    print(f"{meeting}: {len(body)} units -> {path} ({os.path.getsize(path)} bytes)")
    print(f"  expected {expected}")


def main():
    for meeting in EXTRACTS:
        build(meeting)


if __name__ == "__main__":
    sys.exit(main())
