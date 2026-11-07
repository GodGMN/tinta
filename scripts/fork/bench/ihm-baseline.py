#!/usr/bin/env python3
"""AMI IHM headset baseline for the Parakeet TDT 0.6b v3 int8 runtime.

Task #19. The full-meeting Mix-Headset runs in docs/fork/benchmark.md score about
27% Whisper-normalized WER. The published Parakeet v3 figure for AMI is 11.31%
Test WER on `edinburghcstr/ami`, config `ihm`, split `test` (the Open ASR
Leaderboard condition, pinned in the model card). This script reproduces that
condition with the same sherpa-onnx int8 model and greedy decoding:

  prepare  unpack the HF test split into per-utterance wavs, plus a manifest
           that cuts the same time ranges out of the AMI Mix-Headset audio for
           the meetings we already have, so IHM and Mix can be scored on the
           same words.
  score    WER with Whisper's EnglishTextNormalizer (and the legacy normalizer)
           plus del/ins/sub, per group and per meeting.

The dataset is public and needs no token. Its parquet files are pinned by
revision and SHA-256 in scripts/fork/bench/fetch-data.sh; `provenance.json`
records the hashes actually read.

Audio is never committed. Data lives under $ROOT19/data.

Requires: pyarrow (to read the parquet), numpy, rapidfuzz, whisper-normalizer
(the last two are already the scoring dependencies of score.py).
"""

import argparse
import glob
import hashlib
import json
import os
import struct
import sys

BENCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BENCH)

DATASET = "edinburghcstr/ami"
CONFIG = "ihm"
SPLIT = "test"
REVISION = "46f28f2503e2ec48f8867a84eef356c70476beab"


def _import_pyarrow():
    try:
        import pyarrow.parquet as pq

        return pq
    except ModuleNotFoundError:
        # The scoring venv predates this task and has no pyarrow; the system
        # Python has it, and both are CPython 3.11. Append, do not prepend.
        local = os.path.expanduser("~/.local/lib/python3.11/site-packages")
        if os.path.isdir(local):
            sys.path.append(local)
        import pyarrow.parquet as pq

        return pq


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def wav_info(blob):
    """(channels, rate, bits, format_tag) from a RIFF/WAVE header."""
    assert blob[:4] == b"RIFF" and blob[8:12] == b"WAVE", "not a wav"
    i = blob.find(b"fmt ")
    assert i > 0, "no fmt chunk"
    tag, channels, rate, _byte_rate, _align, bits = struct.unpack(
        "<HHIIHH", blob[i + 8 : i + 8 + 16]
    )
    return channels, rate, bits, tag


def load_metadata(parquet_dir):
    """Rows without audio bytes: one dict per utterance, file order preserved."""
    pq = _import_pyarrow()
    rows = []
    for path in sorted(glob.glob(os.path.join(parquet_dir, "test-*.parquet"))):
        table = pq.read_table(
            path,
            columns=[
                "meeting_id",
                "audio_id",
                "text",
                "begin_time",
                "end_time",
                "microphone_id",
                "speaker_id",
            ],
        )
        d = table.to_pydict()
        for i in range(table.num_rows):
            rows.append(
                {
                    "meeting": d["meeting_id"][i],
                    "id": d["audio_id"][i],
                    "text": d["text"][i],
                    "begin": float(d["begin_time"][i]),
                    "end": float(d["end_time"][i]),
                    "mic": d["microphone_id"][i],
                    "speaker": d["speaker_id"][i],
                }
            )
    return rows


def pick_evenly(items, count, key):
    """`count` items spread evenly over a sorted list (deterministic)."""
    items = sorted(items, key=key)
    n = len(items)
    if count >= n:
        return items
    # Midpoints of n equal bins, so the sample does not depend on file order.
    return [items[min(n - 1, int((i + 0.5) * n / count))] for i in range(count)]


def cmd_prepare(args):
    pq = _import_pyarrow()
    os.makedirs(args.out_dir, exist_ok=True)
    wav_dir = os.path.join(args.out_dir, "ihm")
    os.makedirs(wav_dir, exist_ok=True)

    meta = load_metadata(args.parquet_dir)
    pair_meetings = [m for m in args.pair_meetings.split(",") if m]
    by_meeting = {}
    for r in meta:
        by_meeting.setdefault(r["meeting"], []).append(r)

    selected = {}
    for meeting, rows in sorted(by_meeting.items()):
        if meeting in pair_meetings:
            chosen = rows
            group = "ihm_pair"
        else:
            chosen = pick_evenly(rows, args.sample_per_meeting, key=lambda r: r["id"])
            group = "ihm_other"
        for r in chosen:
            r["group"] = group
            selected[r["id"]] = r

    # Second pass: pull the audio only for the selected ids, byte for byte.
    written = 0
    for path in sorted(glob.glob(os.path.join(args.parquet_dir, "test-*.parquet"))):
        table = pq.read_table(path, columns=["audio_id", "audio"])
        d = table.to_pydict()
        for aid, audio in zip(d["audio_id"], d["audio"]):
            if aid not in selected:
                continue
            blob = audio["bytes"]
            channels, rate, bits, tag = wav_info(blob)
            assert channels == 1 and rate == 16000, (aid, channels, rate)
            assert (tag, bits) in ((3, 32), (1, 16)), (aid, tag, bits)
            out = os.path.join(wav_dir, aid + ".wav")
            if not (os.path.exists(out) and os.path.getsize(out) == len(blob)):
                with open(out, "wb") as f:
                    f.write(blob)
                written += 1
            selected[aid]["wav"] = out

    rows_out = []
    ihm_lines = []
    mix_lines = []
    skipped_mix = []
    for r in selected.values():
        rows_out.append(r)
        ihm_lines.append(f"{r['id']}\t{r['wav']}\t\t")
        mix_wav = os.path.join(args.mix_dir, f"{r['meeting']}.Mix-Headset.wav")
        if r["group"] == "ihm_pair":
            if os.path.exists(mix_wav):
                rows_out[-1]["mix_wav"] = mix_wav
                mix_lines.append(
                    f"{r['id']}\t{mix_wav}\t{r['begin']:.3f}\t{r['end']:.3f}"
                )
            else:
                skipped_mix.append((r["id"], mix_wav))
    rows_out.sort(key=lambda r: r["id"])
    ihm_lines.sort()
    mix_lines.sort()

    with open(os.path.join(args.out_dir, "rows.json"), "w") as f:
        json.dump(rows_out, f, indent=1)
    with open(os.path.join(args.out_dir, "manifest-ihm.tsv"), "w") as f:
        f.write("\n".join(ihm_lines) + "\n")
    with open(os.path.join(args.out_dir, "manifest-mix.tsv"), "w") as f:
        f.write("\n".join(mix_lines) + "\n")

    parquet_files = {
        os.path.basename(p): sha256_file(p)
        for p in sorted(glob.glob(os.path.join(args.parquet_dir, "test-*.parquet")))
    }
    prov = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "revision": args.revision,
        "parquet": parquet_files,
        "selection": {
            "pair_meetings_all_rows": pair_meetings,
            "other_meetings_sample_per_meeting": args.sample_per_meeting,
            "rule": "all rows of the pair meetings; every other meeting sampled "
            "evenly over audio_id (midpoints of n equal bins), deterministic",
        },
        "counts": {
            "ihm_rows": len(ihm_lines),
            "mix_rows": len(mix_lines),
            "meetings": len(by_meeting),
            "rows_in_split": len(meta),
            "wavs_written_this_run": written,
        },
        "skipped_mix_missing_audio": skipped_mix,
    }
    with open(os.path.join(args.out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)

    ihm_s = sum(r["end"] - r["begin"] for r in rows_out)
    mix_s = sum(r["end"] - r["begin"] for r in rows_out if "mix_wav" in r)
    print(
        f"prepare: {len(meta)} rows in the {SPLIT} split, {len(by_meeting)} meetings; "
        f"selected {len(ihm_lines)} IHM rows ({ihm_s / 3600:.2f} h) and "
        f"{len(mix_lines)} Mix rows ({mix_s / 3600:.2f} h), {written} wavs written"
    )
    if skipped_mix:
        print(f"prepare: {len(skipped_mix)} pair rows have no Mix-Headset wav (skipped)")
    print(f"prepare: -> {args.out_dir}")


def read_jsonl(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out[d["id"]] = d
    return out


def aggregate(rows, hyps, normalizer):
    from score import legacy_tokens, wer_tokens, whisper_tokens

    fn = legacy_tokens if normalizer == "legacy" else whisper_tokens
    ref_tokens = []
    hyp_tokens = []
    for r in rows:
        hyp = hyps.get(r["id"], {}).get("text", "")
        ref_tokens.extend(fn(r["text"]))
        hyp_tokens.extend(fn(hyp))
    return wer_tokens(ref_tokens, hyp_tokens)


def bootstrap_ci(rows, hyps, normalizer, n=1000, seed=19):
    """95% percentile CI of the corpus WER, resampling utterances.

    The IHM numbers cover 2,131 of the 12,643 test utterances, so the sampling
    error matters when comparing with the published 11.31%. WER is corpus-level
    (total edits / total reference words), so resample utterances and re-add
    their error and reference counts.
    """
    import random

    from score import legacy_tokens, wer_tokens, whisper_tokens

    fn = legacy_tokens if normalizer == "legacy" else whisper_tokens
    counts = []
    for r in rows:
        hyp = hyps.get(r["id"], {}).get("text", "")
        w = wer_tokens(fn(r["text"]), fn(hyp))
        counts.append((w["errors"], w["ref_words"]))
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        errors = refs = 0
        for i in range(len(counts)):
            e, rw = counts[rng.randrange(len(counts))]
            errors += e
            refs += rw
        if refs:
            out.append(errors / refs)
    out.sort()
    lo = out[int(0.025 * (len(out) - 1))]
    hi = out[int(0.975 * (len(out) - 1))]
    return {"lo": lo, "hi": hi, "resamples": n}



def overlap_buckets(rows, refs_dir):
    """Bucket the paired rows by how many reference speakers are active in them.

    The Mix penalty has two possible sources: the far-field room signal alone,
    and other speakers talking inside the same window. The AMI reference words
    for the meeting say which. A 10 ms grid counts how many distinct speakers
    are active in each window; `multi` is the fraction of the window with two or
    more active speakers.

    AMI word intervals tile their pauses (a word's `end` is the next word's
    `start`), so this counts "another speaker's annotated words fall inside the
    window", which includes a backchannel into a pause. That is the right
    question for an ASR input window even when the two voices are not
    simultaneous.
    """
    import numpy as np

    step = 0.01
    grids = {}
    for meeting in sorted({r["meeting"] for r in rows}):
        path = os.path.join(refs_dir, f"{meeting}.ref.json")
        if not os.path.exists(path):
            continue
        ref = json.load(open(path))
        end = max(w["end"] for w in ref["words"]) + 1.0
        n = int(end / step) + 1
        act = []
        for speaker in ref["speakers"]:
            a = np.zeros(n, dtype=bool)
            for w in ref["words"]:
                if w["speaker"] != speaker:
                    continue
                i = max(0, int(w["start"] / step))
                j = min(n, max(i + 1, int(w["end"] / step)))
                a[i:j] = True
            act.append(a)
        stack = np.stack(act)
        grids[meeting] = (stack, stack.sum(axis=0), n)

    out = []
    for r in rows:
        g = grids.get(r["meeting"])
        if g is None:
            continue
        stack, count, n = g
        i = max(0, int(r["begin"] / step))
        j = min(n, max(i + 1, int(r["end"] / step)))
        window_count = count[i:j]
        if window_count.size == 0:
            continue
        multi = float((window_count > 1).mean())
        speakers = int(stack[:, i:j].any(axis=1).sum())
        if multi == 0.0:
            bucket = "solo (no other speaker)"
        elif multi <= 0.2:
            bucket = "other speaker <= 20%"
        else:
            bucket = "other speaker > 20%"
        out.append((r["id"], bucket, multi, speakers))
    return out


def cmd_score(args):
    from score import FILLERS, legacy_tokens, wer_tokens, whisper_tokens

    rows = json.load(open(args.rows))
    ihm = read_jsonl(args.ihm_jsonl)
    mix = read_jsonl(args.mix_jsonl) if args.mix_jsonl else {}

    def no_fillers(tokens):
        return [t for t in tokens if t.lower().strip(".,?!") not in FILLERS]

    groups = {
        "ihm_pair": [r for r in rows if r["group"] == "ihm_pair"],
        "ihm_other": [r for r in rows if r["group"] == "ihm_other"],
    }
    groups["ihm_all"] = groups["ihm_pair"] + groups["ihm_other"]
    mix_rows = [r for r in rows if "mix_wav" in r and r["id"] in mix]
    groups["mix_pair"] = mix_rows

    result = {"groups": {}, "per_meeting": {}, "pair_per_meeting": {}}
    for name, rs in groups.items():
        if not rs:
            continue
        hyps = mix if name == "mix_pair" else ihm
        block = {
            "utterances": len(rs),
            "audio_seconds": sum(r["end"] - r["begin"] for r in rs),
            "whisper": aggregate(rs, hyps, "whisper"),
            "legacy": aggregate(rs, hyps, "legacy"),
        }
        # Filler-free diagnostic: how much of the gap is backchannels/fillers.
        R = [t for r in rs for t in whisper_tokens(r["text"])]
        H = [t for r in rs for t in whisper_tokens(hyps.get(r["id"], {}).get("text", ""))]
        block["whisper_no_fillers"] = wer_tokens(no_fillers(R), no_fillers(H))
        block["empty_hypotheses"] = sum(
            1 for r in rs if not hyps.get(r["id"], {}).get("text", "").strip()
        )
        block["whisper_ci95"] = bootstrap_ci(rs, hyps, "whisper")
        result["groups"][name] = block

    meetings = sorted({r["meeting"] for r in rows})
    for m in meetings:
        rs = [r for r in rows if r["meeting"] == m and r["id"] in ihm]
        if not rs:
            continue
        result["per_meeting"][m] = {
            "utterances": len(rs),
            "whisper": aggregate(rs, ihm, "whisper"),
        }
    for m in sorted({r["meeting"] for r in mix_rows}):
        rs = [r for r in mix_rows if r["meeting"] == m]
        result["pair_per_meeting"][m] = {
            "utterances": len(rs),
            "ihm_whisper": aggregate(rs, ihm, "whisper"),
            "mix_whisper": aggregate(rs, mix, "whisper"),
        }

    # Far-field vs crosstalk: the same paired rows, split by how much of the
    # window another AMI reference speaker covers.
    buckets = {}
    for rid, bucket, multi, speakers in overlap_buckets(mix_rows, args.refs_dir):
        buckets.setdefault(bucket, []).append(rid)
    by_overlap = {}
    for bucket, ids in buckets.items():
        rs = [r for r in mix_rows if r["id"] in set(ids)]
        by_overlap[bucket] = {
            "utterances": len(rs),
            "audio_seconds": sum(r["end"] - r["begin"] for r in rs),
            "ihm_whisper": aggregate(rs, ihm, "whisper"),
            "mix_whisper": aggregate(rs, mix, "whisper"),
        }
    result["by_overlap"] = by_overlap

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)

    def pct(x):
        return f"{100 * x:.1f}%"

    def wer_row(name, b, key="whisper"):
        w = b[key]
        ci = b.get("whisper_ci95")
        ci_s = f" [{100 * ci['lo']:.1f}-{100 * ci['hi']:.1f}]" if ci else ""
        return (
            f"| {name} | {b['utterances']} | {b['audio_seconds'] / 60:.1f} | "
            f"{pct(w['wer'])}{ci_s} | {w['del']} / {w['ins']} / {w['sub']} | "
            f"{w['ref_words']} |"
        )

    print("\n### IHM vs Mix, same utterances\n")
    print("| Set | Utterances | Audio min | WER (Whisper) [95% CI] | del / ins / sub | Ref words |")
    print("|---|---|---|---|---|---|")
    for name in ("ihm_pair", "mix_pair", "ihm_other", "ihm_all"):
        if name in result["groups"]:
            print(wer_row(name, result["groups"][name]))

    print("\n### Per-meeting IHM (Whisper WER)\n")
    print("| Meeting | Utterances | WER |")
    print("|---|---|---|")
    for m, b in result["per_meeting"].items():
        print(f"| {m} | {b['utterances']} | {pct(b['whisper']['wer'])} |")

    if result["pair_per_meeting"]:
        print("\n### Paired IHM vs Mix per meeting\n")
        print("| Meeting | Utterances | IHM WER | Mix WER |")
        print("|---|---|---|---|")
        for m, b in result["pair_per_meeting"].items():
            print(
                f"| {m} | {b['utterances']} | {pct(b['ihm_whisper']['wer'])} | "
                f"{pct(b['mix_whisper']['wer'])} |"
            )

    print("\n### Filler-free diagnostic (Whisper tokens, fillers removed both sides)\n")
    print("| Set | WER | del / ins / sub | Ref words | Empty hyps |")
    print("|---|---|---|---|---|")
    for name, b in result["groups"].items():
        w = b["whisper_no_fillers"]
        print(
            f"| {name} | {pct(w['wer'])} | {w['del']} / {w['ins']} / {w['sub']} | "
            f"{w['ref_words']} | {b['empty_hypotheses']} |"
        )

    if result["by_overlap"]:
        print("\n### Far-field vs crosstalk (paired rows, by AMI reference speakers in the window)\n")
        print("| Bucket | Utterances | Audio min | IHM WER | Mix WER | Delta |")
        print("|---|---|---|---|---|---|")
        order = [
            "solo (no other speaker)",
            "other speaker <= 20%",
            "other speaker > 20%",
        ]
        for bucket in order:
            b = result["by_overlap"].get(bucket)
            if not b:
                continue
            a = b["ihm_whisper"]["wer"]
            c = b["mix_whisper"]["wer"]
            print(
                f"| {bucket} | {b['utterances']} | {b['audio_seconds'] / 60:.1f} | "
                f"{pct(a)} | {pct(c)} | {pct(c - a)} |"
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("prepare", help="unpack the IHM test split and build manifests")
    q.add_argument("--parquet-dir", default="/home/sergio/code/evidence/tinta-fork/19/data/ihm-test")
    q.add_argument("--out-dir", default="/home/sergio/code/evidence/tinta-fork/19/data/ihm-baseline")
    q.add_argument("--mix-dir", default="/home/sergio/code/evidence/tinta-fork/2/data")
    q.add_argument("--pair-meetings", default="ES2004b,IS1009b")
    q.add_argument("--sample-per-meeting", type=int, default=60)
    q.add_argument("--revision", default=REVISION)
    q.set_defaults(func=cmd_prepare)

    q = sub.add_parser("score", help="WER tables from the batch decode JSONL")
    q.add_argument("--rows", default="/home/sergio/code/evidence/tinta-fork/19/data/ihm-baseline/rows.json")
    q.add_argument("--ihm-jsonl", default="/home/sergio/code/evidence/tinta-fork/19/results/ihm-decode.jsonl")
    q.add_argument("--mix-jsonl", default="/home/sergio/code/evidence/tinta-fork/19/results/mix-decode.jsonl")
    q.add_argument("--refs-dir", default="/home/sergio/code/evidence/tinta-fork/2/data")
    q.add_argument("--out", default="/home/sergio/code/evidence/tinta-fork/19/results/ihm-summary.json")
    q.set_defaults(func=cmd_score)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
