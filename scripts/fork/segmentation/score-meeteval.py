#!/usr/bin/env python3
"""Stream-level scoring with meeteval: tcpWER (collars 5 s and 1 s) and cpWER.

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

Normalization is symmetric. The task #23 review found the first version
normalized the reference per row and the hypothesis per word, so Whisper's
cross-word rules (number merging, the `T_V_` acronym join) acted on one side
only. Here both sides are normalized before meeteval sees them: the reference
per row, the hypothesis per contiguous run (words no more than `--run-gap`
seconds apart), which is the same unit a speaker would speak. `normalizer=None`
tells meeteval the text is already normalized.

No UEM is used: hypothesis words outside the annotated regions count as
insertions for every stream, which is acceptable for relative comparison. The
merged reference still contains both sides of every overlap (9.7 % of ES2004b
meeting time), so its absolute value is not reachable by a single stream; it
is comparable across streams.

Both metrics are corpus-level: summed errors over summed reference length.

Usage:
  score-meeteval.py --rows ROWS --oracle "M=path,..." --pipeline "M=path,..." \
                    --label requant-default [--out summary.json]
"""

import argparse
import difflib
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

COLLARS = (5.0, 1.0)
RUN_GAP_SECONDS = 1.0


def normalized(text):
    """The shared Whisper normalizer, applied to one piece of text."""
    return " ".join(score.whisper_tokens(text))


def reference_segments(rows, merged):
    return [
        {
            "session_id": row["meeting"],
            "speaker": "ALL" if merged else row["speaker"],
            "start_time": float(row["begin"]),
            "end_time": float(row["end"]),
            "words": normalized(row["text"]),
        }
        for row in rows
    ]


def oracle_per_turn_segments(rows, oracle, merged):
    """The dataset's own turn decodes, re-timed to the meeting clock."""
    segments = []
    for row in rows:
        for segment in run_token_segments(
            row["meeting"],
            oracle.get(row["id"], {}).get("words", []),
            "ALL" if merged else row["speaker"],
            offset=float(row["begin"]),
        ):
            segments.append(segment)
    return segments


def word_runs(words, gap=RUN_GAP_SECONDS):
    """Group words into contiguous runs, so cross-word rules act on both sides."""
    runs = []
    current = []
    for word in sorted(words, key=lambda item: item["start"]):
        if current and word["start"] - current[-1]["end"] > gap:
            runs.append(current)
            current = []
        current.append(word)
    if current:
        runs.append(current)
    return runs


def run_token_segments(meeting, words, speaker, offset=0.0):
    """Normalize each run, then put the tokens back on word-level times.

    A run is the unit the Whisper normalizer sees, so cross-word rules (number
    merging, `T V` -> `tv`) act on the hypothesis exactly as they act on the
    reference. The tokens are then re-timed: one per word when the count is
    unchanged (the usual case), and proportionally across the run when a merge
    or split changed the count. Keeping one segment per token, rather than one
    per run, is what makes the 1 s collar meaningful.
    """
    segments = []
    for run in word_runs(words):
        for token, start, end in run_token_times(run):
            segments.append(
                {
                    "session_id": meeting,
                    "speaker": speaker,
                    "start_time": float(offset + start),
                    "end_time": float(offset + end),
                    "words": token,
                }
            )
    return segments


def run_token_times(run):
    """The run's normalized tokens, each on the time of the word it came from.

    Normalizing the run gives the cross-word rules; normalizing each word alone
    gives a token-to-word map. Aligning the two lists (difflib) keeps the exact
    word time for every token that survived unchanged, and spans the words a
    merged token replaced. Without the alignment, one split word ("cannot" ->
    "can not") would re-space the whole run and destroy the 1 s collar.
    """
    per_word = [score.whisper_tokens(word["text"]) for word in run]
    flat = [(index, token) for index, tokens in enumerate(per_word) for token in tokens]
    tokens = score.whisper_tokens(" ".join(word["text"] for word in run))
    out = []
    matcher = difflib.SequenceMatcher(
        a=[token for _, token in flat], b=tokens, autojunk=False
    )
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for step in range(i2 - i1):
                word = run[flat[i1 + step][0]]
                out.append((tokens[j1 + step], word["start"], word["end"]))
            continue
        if tag == "delete":
            continue
        # "replace" or "insert": the tokens cover the words flat[i1..i2].
        if i2 > i1:
            start, end = run[flat[i1][0]]["start"], run[flat[i2 - 1][0]]["end"]
        elif i1 < len(flat):
            start = end = run[flat[i1][0]]["start"]
        else:
            start = end = run[-1]["end"]
        count = j2 - j1
        span = end - start
        for step in range(count):
            out.append(
                (
                    tokens[j1 + step],
                    start + span * step / count,
                    start + span * (step + 1) / count,
                )
            )
    return out


def word_segments(meeting, words, speaker):
    return run_token_segments(meeting, words, speaker)


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


def evaluate(reference, streams):
    """tcpWER at every collar and cpWER of each stream against one reference."""
    output = {}
    for name, segments in streams.items():
        hypothesis = SegLST(segments)
        collars = {
            f"collar_{collar:g}": summarise(
                tcpwer(reference, hypothesis, collar=collar, normalizer=None)
            )
            for collar in COLLARS
        }
        output[name] = {
            "tcpwer": collars,
            "cpwer": summarise(cpwer(reference, hypothesis, normalizer=None)),
        }
    return output


def print_table(title, results):
    header = " | ".join(f"tcpWER {collar:g} s" for collar in COLLARS)
    print(f"\n{title}\n")
    print(f"| Stream | {header} | cpWER |")
    print("|---|---|---|---|")
    for name, metrics in results.items():
        cells = " | ".join(
            f"{100 * metrics['tcpwer'][f'collar_{collar:g}']['wer']:.1f}%"
            for collar in COLLARS
        )
        print(f"| {name} | {cells} | {100 * metrics['cpwer']['wer']:.1f}% |")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--oracle", required=True, help="meeting=jsonl,...")
    parser.add_argument("--pipeline", required=True, help="meeting=json,...")
    parser.add_argument("--label", default="pipeline")
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

    print(f"\n### {args.label} - meeteval, hypothesis and reference normalized alike\n")
    merged = evaluate(merged_reference, merged_streams)
    print_table("Merged reference (one speaker; word accuracy only)", merged)
    print("\nPipeline per meeting (merged reference):\n")
    print("| Meeting | tcpWER 5 s | tcpWER 1 s | cpWER |")
    print("|---|---|---|---|")
    for meeting in merged["pipeline"]["tcpwer"]["collar_5"]["per_session"]:
        cells = " | ".join(
            f"{100 * merged['pipeline']['tcpwer'][f'collar_{collar:g}']['per_session'][meeting]['wer']:.1f}%"
            for collar in COLLARS
        )
        cp = merged["pipeline"]["cpwer"]["per_session"][meeting]["wer"]
        print(f"| {meeting} | {cells} | {100 * cp:.1f}% |")
    speaker = evaluate(speaker_reference, speaker_streams)
    print_table(
        "Speaker reference (tcpWER also scores speaker attribution; a "
        "single-stream hypothesis is charged for every speaker it does not cover)",
        speaker,
    )
    for view, results in (("merged", merged), ("speaker", speaker)):
        pipeline = results["pipeline"]
        for oracle in ("oracle_per_turn", "oracle_single_stream"):
            pipeline[f"delta_vs_{oracle}"] = {
                f"collar_{collar:g}": (
                    pipeline["tcpwer"][f"collar_{collar:g}"]["wer"]
                    - results[oracle]["tcpwer"][f"collar_{collar:g}"]["wer"]
                )
                for collar in COLLARS
            }
            pipeline[f"delta_vs_{oracle}"]["cpwer"] = (
                pipeline["cpwer"]["wer"] - results[oracle]["cpwer"]["wer"]
            )
        bound = results["oracle_single_stream"]
        for collar in COLLARS:
            print(
                f"\n{view} reference, collar {collar:g} s: pipeline - single-stream "
                f"oracle is "
                f"{100 * (pipeline['tcpwer'][f'collar_{collar:g}']['wer'] - bound['tcpwer'][f'collar_{collar:g}']['wer']):+.1f} "
                "points tcpWER."
            )
    result = {
        "label": args.label,
        "collars": list(COLLARS),
        "run_gap_seconds": RUN_GAP_SECONDS,
        "normalization": "whisper per reference row and per hypothesis run; meeteval normalizer=None",
        "uem": "none; hypothesis words outside annotated regions count as insertions",
        "merged_reference": merged,
        "speaker_reference": speaker,
    }
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
