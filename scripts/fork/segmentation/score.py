#!/usr/bin/env python3
"""Same-reference ablation scoring for the task #21 segmentation harness.

Reference: the HF IHM segment text of the same meeting (`rows.json` from the
task #19 evidence, every row of the split). Oracle hypotheses: the Mix-Headset
decode of the dataset's own segment boundaries. Pipeline hypotheses: the
engine's full-meeting decode from `tinta-segmentation`, with each word assigned
to exactly one segment window by midpoint (the shortest window containing the
midpoint), as in `ihm-baseline.py ablate-score` on the task #19 branch.

Two oracle numbers are reported. The **per-turn oracle** decodes each reference
turn on its own, so it transcribes both sides of an overlap. The **single-stream
oracle** merges those words into one absolute-time stream and, in every overlap,
keeps only the side with more words in the shared interval: it asks what the
same scorer would give a perfect single-channel recognizer, which is the fair
bound for this pipeline. Score deltas against both.

WER is the leaderboard estimator: summed per-utterance edit distances over the
summed reference words, with Whisper's EnglishTextNormalizer on both sides.
An empty normalized reference keeps its hypothesis tokens as insertions.

Usage:
  score.py --rows ROWS --oracle "M=path,..." --pipeline "M=path,..." \
           --label short-gap [--out summary.json]
"""

import argparse
import itertools
import json
import os
import sys

FILLERS_UNUSED = None
_NORMALIZER = None


def ami_acronyms(text):
    """Join AMI's spelled-out acronyms (`T_V_` -> `TV`), as in score.py."""
    import re

    return re.sub(r"\b(?:[A-Za-z]+_)+", lambda m: m.group(0).replace("_", ""), text)


def whisper_tokens(words):
    global _NORMALIZER
    if _NORMALIZER is None:
        from whisper_normalizer.english import EnglishTextNormalizer

        _NORMALIZER = EnglishTextNormalizer()
    text = words if isinstance(words, str) else " ".join(words)
    return _NORMALIZER(ami_acronyms(text)).split()


def counts(ref_text, hyp_text):
    from rapidfuzz.distance import Levenshtein

    ref = whisper_tokens(ref_text)
    hyp = whisper_tokens(hyp_text)
    if not ref:
        # An empty reference keeps its hypothesis tokens as insertions.
        return {
            "errors": len(hyp),
            "ref_words": 0,
            "hyp_words": len(hyp),
            "del": 0,
            "ins": len(hyp),
            "sub": 0,
        }
    sub = ins = dele = 0
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(ref, hyp):
        if tag == "replace":
            n = min(i2 - i1, j2 - j1)
            sub += n
            dele += (i2 - i1) - n
            ins += (j2 - j1) - n
        elif tag == "delete":
            dele += i2 - i1
        elif tag == "insert":
            ins += j2 - j1
    return {
        "errors": sub + ins + dele,
        "ref_words": len(ref),
        "hyp_words": len(hyp),
        "del": dele,
        "ins": ins,
        "sub": sub,
    }


def read_jsonl(path):
    out = {}
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                out[row["id"]] = row
    return out


def load_pipeline(path, field="asr"):
    with open(path) as handle:
        data = json.load(handle)
    if field == "joined":
        return data.get("joined", [])
    if "asr" in data:
        return data["asr"].get("words", [])
    return data.get("words", [])


def assign_rows(rows, words):
    """One hypothesis text per row: the shortest window containing the word midpoint."""
    bounds = sorted(((row["id"], row["begin"], row["end"]) for row in rows), key=lambda x: x[1])
    by_row = {row["id"]: [] for row in rows}
    outside = 0
    for word in words:
        middle = (word["start"] + word["end"]) / 2
        best = None
        for row_id, start, end in bounds:
            if start <= middle < end and (best is None or (end - start) < (best[2] - best[1])):
                best = (row_id, start, end)
        if best is None:
            outside += 1
        else:
            by_row[best[0]].append(word["text"])
    return {key: " ".join(value) for key, value in by_row.items()}, outside


def sum_counts(items):
    total = {"errors": 0, "ref_words": 0, "hyp_words": 0, "del": 0, "ins": 0, "sub": 0}
    for item in items:
        for key in total:
            total[key] += item[key]
    total["wer"] = total["errors"] / total["ref_words"] if total["ref_words"] else 0.0
    return total


def oracle_words(rows, oracle):
    """The oracle hypotheses as one absolute-time word stream."""
    words = []
    for row in rows:
        hypothesis = oracle.get(row["id"], {})
        for word in hypothesis.get("words", []):
            words.append(
                {
                    "text": word["text"],
                    "start": row["begin"] + word["start"],
                    "end": row["begin"] + word["end"],
                    "row": row["id"],
                }
            )
    return words


def single_stream_words(rows, oracle, overlap_seconds=0.2):
    """The fair oracle bound: the per-turn oracle merged into one stream.

    The per-turn oracle decodes each reference turn on its own, so in an overlap
    it transcribes both speakers. A single-channel pipeline hears the mix once.
    Merging the oracle words and, in each overlap, keeping only the side with
    more words in the shared interval (as in the task #21 review) removes that
    advantage. The removed words count is returned for the report.
    """
    words = oracle_words(rows, oracle)
    by_row = {}
    for index, word in enumerate(words):
        by_row.setdefault(word["row"], []).append(index)
    dropped = set()
    for first, second in itertools.combinations(rows, 2):
        shared_start = max(first["begin"], second["begin"])
        shared_end = min(first["end"], second["end"])
        if shared_end - shared_start <= overlap_seconds:
            continue

        def in_shared(row_id):
            return [
                index
                for index in by_row.get(row_id, [])
                if shared_start
                <= (words[index]["start"] + words[index]["end"]) / 2
                < shared_end
            ]

        first_inside = in_shared(first["id"])
        second_inside = in_shared(second["id"])
        if len(first_inside) == len(second_inside):
            loser = by_row.get(second["id"], [])
        elif len(first_inside) > len(second_inside):
            loser = by_row.get(second["id"], [])
        else:
            loser = by_row.get(first["id"], [])
        for index in loser:
            middle = (words[index]["start"] + words[index]["end"]) / 2
            if shared_start <= middle < shared_end:
                dropped.add(index)
    kept = [word for index, word in enumerate(words) if index not in dropped]
    return kept, len(dropped)


def pairs(spec):
    return dict(item.split("=", 1) for item in spec.split(",") if item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--oracle", required=True, help="meeting=jsonl,...")
    parser.add_argument("--pipeline", required=True, help="meeting=json,...")
    parser.add_argument("--label", default="pipeline")
    parser.add_argument(
        "--field",
        default="asr",
        choices=["asr", "joined"],
        help="which word list to score: the engine-shaped asr.words or the "
        "turns experiment's midpoint-joined list",
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    rows = json.load(open(args.rows))
    oracle_paths = pairs(args.oracle)
    pipeline_paths = pairs(args.pipeline)
    meetings = [meeting for meeting in pipeline_paths if meeting in oracle_paths]

    result = {"label": args.label, "per_meeting": {}}
    all_oracle = []
    all_single = []
    all_pipeline = []
    print(f"\n### {args.label}\n")
    print(
        "| Meeting | Segments | Oracle per-turn | Oracle single stream | Pipeline | "
        "Delta vs per-turn | Delta vs single stream | del / ins / sub (pipeline) | Words outside windows |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for meeting in meetings:
        meeting_rows = [row for row in rows if row["meeting"] == meeting]
        oracle = read_jsonl(oracle_paths[meeting])
        hypothesis, outside = assign_rows(
            meeting_rows, load_pipeline(pipeline_paths[meeting], args.field)
        )
        single_words, dropped = single_stream_words(meeting_rows, oracle)
        single_hypothesis, _ = assign_rows(meeting_rows, single_words)
        oracle_counts = [counts(row["text"], oracle.get(row["id"], {}).get("text", "")) for row in meeting_rows]
        single_counts = [counts(row["text"], single_hypothesis.get(row["id"], "")) for row in meeting_rows]
        pipeline_counts = [counts(row["text"], hypothesis.get(row["id"], "")) for row in meeting_rows]
        o = sum_counts(oracle_counts)
        s = sum_counts(single_counts)
        p = sum_counts(pipeline_counts)
        all_oracle.extend(oracle_counts)
        all_single.extend(single_counts)
        all_pipeline.extend(pipeline_counts)
        result["per_meeting"][meeting] = {
            "segments": len(meeting_rows),
            "oracle": o,
            "oracle_single_stream": s,
            "oracle_single_stream_words_dropped": dropped,
            "pipeline": p,
            "delta": p["wer"] - o["wer"],
            "delta_single_stream": p["wer"] - s["wer"],
            "words_outside_windows": outside,
        }
        print(
            f"| {meeting} | {len(meeting_rows)} | {100 * o['wer']:.1f}% | "
            f"{100 * s['wer']:.1f}% | {100 * p['wer']:.1f}% | "
            f"{100 * (p['wer'] - o['wer']):+.1f} | "
            f"{100 * (p['wer'] - s['wer']):+.1f} | "
            f"{p['del']} / {p['ins']} / {p['sub']} | {outside} |"
        )
    o = sum_counts(all_oracle)
    s = sum_counts(all_single)
    p = sum_counts(all_pipeline)
    result["overall"] = {
        "oracle": o,
        "oracle_single_stream": s,
        "pipeline": p,
        "delta": p["wer"] - o["wer"],
        "delta_single_stream": p["wer"] - s["wer"],
    }
    print(
        f"| **All** | {len(all_pipeline)} | **{100 * o['wer']:.1f}%** | "
        f"**{100 * s['wer']:.1f}%** | **{100 * p['wer']:.1f}%** | "
        f"**{100 * (p['wer'] - o['wer']):+.1f}** | "
        f"**{100 * (p['wer'] - s['wer']):+.1f}** | "
        f"{p['del']} / {p['ins']} / {p['sub']} | |"
    )
    print(
        f"\n{args.label}: oracle {100 * o['wer']:.1f}% ({o['del']} del / {o['ins']} ins / "
        f"{o['sub']} sub, {o['ref_words']} ref words), single-stream oracle "
        f"{100 * s['wer']:.1f}%, pipeline {100 * p['wer']:.1f}% "
        f"({p['del']} del / {p['ins']} ins / {p['sub']} sub), delta "
        f"{100 * (p['wer'] - o['wer']):+.1f} / {100 * (p['wer'] - s['wer']):+.1f} points."
    )
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
