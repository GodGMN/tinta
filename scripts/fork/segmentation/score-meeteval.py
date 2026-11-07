#!/usr/bin/env python3
"""Stream-level scoring with meeteval: tcpWER (collar 5 s) and cpWER.

The per-utterance scorer (`score.py`) assigns every word to exactly one
reference row, which mis-prices a single-channel stream in overlaps: the
oracle's own merged words score worse than the pipeline. Time-constrained WER
charges words by time instead, so it is the honest stream-level view of the
same hypotheses.

Two reference views are scored because they answer different questions:

- **merged reference**: every reference row is one speaker, so the metric
  measures word accuracy only. This is the fair view of a single mixed
  channel, and the one to compare the pipeline with the single-stream oracle.
- **speaker reference**: the rows keep their speakers, so tcpWER also scores
  speaker attribution. An oracle that decodes each turn with its speaker can
  use this; a single-stream hypothesis cannot, and every reference speaker it
  does not cover is charged as missed (a metric artifact, reported to show
  why this is not the pipeline's check).

Text is normalized with the same Whisper normalizer as `score.py`. Both
metrics are corpus-level: summed errors over summed reference length.

Usage:
  score-meeteval.py --rows ROWS --oracle "M=path,..." --pipeline "M=path,..." \
                    --label fp32-default [--out summary.json]
"""

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def score_module():
    """`score.py` next to this file, for the shared normalizer and oracle merge."""
    spec = importlib.util.spec_from_file_location(
        "segmentation_score", os.path.join(HERE, "score.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


score = score_module()

from meeteval.io import SegLST  # noqa: E402  (after the path set-up above)
from meeteval.wer import cpwer, tcpwer  # noqa: E402


def normalize(segment):
    """The per-utterance scorer's text form, applied to one segment.

    meeteval's `normalizer` takes a segment dict (it normalizes the `words`
    field), not a string, so this is the same Whisper normalizer as `score.py`
    wrapped for that shape.
    """
    segment = dict(segment)
    segment["words"] = " ".join(score.whisper_tokens(segment["words"]))
    return segment


def reference_segments(rows, merged):
    return [
        {
            "session_id": row["meeting"],
            "speaker": "ALL" if merged else row["speaker"],
            "start_time": float(row["begin"]),
            "end_time": float(row["end"]),
            "words": row["text"],
        }
        for row in rows
    ]


def oracle_per_turn_segments(rows, oracle, merged):
    """The dataset's own turn decodes, re-timed to the meeting clock."""
    segments = []
    for row in rows:
        for word in oracle.get(row["id"], {}).get("words", []):
            segments.append(
                {
                    "session_id": row["meeting"],
                    "speaker": "ALL" if merged else row["speaker"],
                    "start_time": float(row["begin"] + word["start"]),
                    "end_time": float(row["begin"] + word["end"]),
                    "words": word["text"],
                }
            )
    return segments


def word_segments(meeting, words, speaker):
    return [
        {
            "session_id": meeting,
            "speaker": speaker,
            "start_time": float(word["start"]),
            "end_time": float(word["end"]),
            "words": word["text"],
        }
        for word in words
    ]


def summarise(result):
    """Corpus-level errors and reference length from a per-session dict."""
    errors = sum(item.errors for item in result.values())
    length = sum(item.length for item in result.values())
    return {
        "errors": errors,
        "length": length,
        "wer": errors / length if length else 0.0,
        "sessions": len(result),
        "per_session": {
            name: {
                "errors": item.errors,
                "length": item.length,
                "wer": item.errors / item.length if item.length else 0.0,
            }
            for name, item in result.items()
        },
    }


def evaluate(reference, streams, collar):
    """tcpWER and cpWER of each stream against one reference view."""
    output = {}
    for name, segments in streams.items():
        hypothesis = SegLST(segments)
        tcp = summarise(tcpwer(reference, hypothesis, collar=collar, normalizer=normalize))
        cp = summarise(cpwer(reference, hypothesis, normalizer=normalize))
        output[name] = {"tcpwer": tcp, "cpwer": cp}
    return output


def print_table(title, results):
    print(f"\n{title}\n")
    print("| Stream | tcpWER | cpWER | tcp errors / ref words | cp errors / ref words |")
    print("|---|---|---|---|---|")
    for name, metrics in results.items():
        tcp, cp = metrics["tcpwer"], metrics["cpwer"]
        print(
            f"| {name} | {100 * tcp['wer']:.1f}% | {100 * cp['wer']:.1f}% | "
            f"{tcp['errors']} / {tcp['length']} | {cp['errors']} / {cp['length']} |"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--oracle", required=True, help="meeting=jsonl,...")
    parser.add_argument("--pipeline", required=True, help="meeting=json,...")
    parser.add_argument("--label", default="pipeline")
    parser.add_argument("--collar", type=float, default=5.0)
    parser.add_argument("--out")
    args = parser.parse_args()

    rows = json.load(open(args.rows))
    oracle_paths = score.pairs(args.oracle)
    pipeline_paths = score.pairs(args.pipeline)
    meetings = [meeting for meeting in pipeline_paths if meeting in oracle_paths]
    rows = [row for row in rows if row["meeting"] in meetings]

    merged_streams = {"pipeline": [], "oracle_per_turn": [], "oracle_single_stream": []}
    speaker_streams = {"pipeline": [], "oracle_per_turn": [], "oracle_single_stream": []}
    for meeting in meetings:
        meeting_rows = [row for row in rows if row["meeting"] == meeting]
        oracle = score.read_jsonl(oracle_paths[meeting])
        pipeline_words = score.load_pipeline(pipeline_paths[meeting])
        merged_streams["pipeline"].extend(word_segments(meeting, pipeline_words, "H"))
        merged_streams["oracle_single_stream"].extend(
            word_segments(
                meeting, score.single_stream_words(meeting_rows, oracle)[0], "H"
            )
        )
        merged_streams["oracle_per_turn"].extend(
            oracle_per_turn_segments(meeting_rows, oracle, merged=True)
        )
        speaker_streams["pipeline"].extend(word_segments(meeting, pipeline_words, "H"))
        speaker_streams["oracle_single_stream"].extend(
            word_segments(
                meeting, score.single_stream_words(meeting_rows, oracle)[0], "H"
            )
        )
        speaker_streams["oracle_per_turn"].extend(
            oracle_per_turn_segments(meeting_rows, oracle, merged=False)
        )

    merged_reference = SegLST(reference_segments(rows, merged=True))
    speaker_reference = SegLST(reference_segments(rows, merged=False))

    print(f"\n### {args.label} - meeteval, tcpWER collar {args.collar:g} s\n")
    merged = evaluate(merged_reference, merged_streams, args.collar)
    print_table("Merged reference (one speaker; word accuracy only)", merged)
    print("\nPipeline per meeting (merged reference):\n")
    print("| Meeting | tcpWER | cpWER | tcp errors / ref words |")
    print("|---|---|---|---|")
    for meeting, item in merged["pipeline"]["tcpwer"]["per_session"].items():
        cp_item = merged["pipeline"]["cpwer"]["per_session"][meeting]
        print(
            f"| {meeting} | {100 * item['wer']:.1f}% | {100 * cp_item['wer']:.1f}% | "
            f"{item['errors']} / {item['length']} |"
        )
    speaker = evaluate(speaker_reference, speaker_streams, args.collar)
    print_table(
        "Speaker reference (tcpWER also scores speaker attribution; a "
        "single-stream hypothesis is charged for every speaker it does not cover)",
        speaker,
    )
    for view, results in (("merged", merged), ("speaker", speaker)):
        pipeline = results["pipeline"]
        for oracle in ("oracle_per_turn", "oracle_single_stream"):
            pipeline[f"delta_vs_{oracle}"] = {
                "tcpwer": pipeline["tcpwer"]["wer"] - results[oracle]["tcpwer"]["wer"],
                "cpwer": pipeline["cpwer"]["wer"] - results[oracle]["cpwer"]["wer"],
            }
        bound = results["oracle_single_stream"]
        print(
            f"\n{view} reference: pipeline - single-stream oracle is "
            f"{100 * (pipeline['tcpwer']['wer'] - bound['tcpwer']['wer']):+.1f} "
            f"points tcpWER and "
            f"{100 * (pipeline['cpwer']['wer'] - bound['cpwer']['wer']):+.1f} "
            "points cpWER."
        )
    result = {
        "label": args.label,
        "collar": args.collar,
        "merged_reference": merged,
        "speaker_reference": speaker,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
