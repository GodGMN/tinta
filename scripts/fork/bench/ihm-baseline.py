#!/usr/bin/env python3
"""AMI IHM headset baseline for the Parakeet TDT 0.6b v3 int8 runtime.

Task #19. The full-meeting Mix-Headset runs in docs/fork/benchmark.md score about
27% Whisper-normalized WER. The published Parakeet v3 figure for AMI is 11.31%
Test WER on `edinburghcstr/ami`, config `ihm`, split `test` (the Open ASR
Leaderboard condition, pinned in the model card). This script reproduces that
condition with the same sherpa-onnx int8 model and greedy decoding.

Subcommands:
  prepare          unpack the HF test split into per-utterance wavs and build
                   the IHM manifest plus the same-ranges Mix manifest. `--full`
                   selects every row of the split; the default is the
                   deterministic 16-meeting sample.
  score            corpus WER as sum(per-utterance edits) / sum(reference
                   words), with del/ins/sub, clustered bootstrap, per-meeting
                   tables and the crosstalk/duration buckets.
  weighted         test-set-weighted estimate from the sample (the cross-check
                   for the full-split number).
  ablate-prepare   oracle-segment manifests for the same-reference check.
  ablate-score     oracle segments vs the engine VAD pipeline, same reference
                   words (the HF segment text), per utterance.
  cause            where the VAD-pipeline deletions come from: hypothesis holes
                   of 3 s or more at 20 s flush piece starts.

The dataset is public and needs no token. Its parquet files are pinned by
revision and SHA-256 in scripts/fork/bench/fetch-data.sh; `provenance.json`
records the hashes actually read.

Audio is never committed. Data lives under $ROOT19/data; models under $ROOT.

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


def env_path(var, *parts):
    """Path under $<var>, or None when the variable is not set.

    The scripts take no machine-specific default; the caller passes an
    argument or exports the variable (see the Reproduction section of the doc).
    """
    root = os.environ.get(var)
    return os.path.join(root, *parts) if root else None


def require(path, arg, env):
    if not path:
        sys.exit(f"error: pass --{arg} or export ${env}")
    return path


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
    out_dir = require(args.out_dir, "out-dir", "ROOT19")
    parquet_dir = require(args.parquet_dir, "parquet-dir", "ROOT19")
    mix_dir = require(args.mix_dir, "mix-dir", "ROOT")
    os.makedirs(out_dir, exist_ok=True)
    wav_dir = os.path.join(out_dir, "ihm")
    os.makedirs(wav_dir, exist_ok=True)

    meta = load_metadata(parquet_dir)
    pair_meetings = [m for m in args.pair_meetings.split(",") if m]
    by_meeting = {}
    for r in meta:
        by_meeting.setdefault(r["meeting"], []).append(r)

    selected = {}
    for meeting, rows in sorted(by_meeting.items()):
        if meeting in pair_meetings:
            chosen = rows
            group = "ihm_pair"
        elif args.full:
            chosen = rows
            group = "ihm_full"
        else:
            chosen = pick_evenly(rows, args.sample_per_meeting, key=lambda r: r["id"])
            group = "ihm_other"
        for r in chosen:
            r["group"] = group
            selected[r["id"]] = r

    # Second pass: pull the audio only for the selected ids, byte for byte.
    written = 0
    total_bytes = 0
    for path in sorted(glob.glob(os.path.join(parquet_dir, "test-*.parquet"))):
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
                total_bytes += len(blob)
            selected[aid]["wav"] = out

    rows_out = []
    ihm_lines = []
    mix_lines = []
    skipped_mix = []
    for r in selected.values():
        rows_out.append(r)
        ihm_lines.append(f"{r['id']}\t{r['wav']}\t\t")
        if r["group"] == "ihm_pair":
            mix_wav = os.path.join(mix_dir, f"{r['meeting']}.Mix-Headset.wav")
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

    with open(os.path.join(out_dir, "rows.json"), "w") as f:
        json.dump(rows_out, f, indent=1)
    with open(os.path.join(out_dir, "manifest-ihm.tsv"), "w") as f:
        f.write("\n".join(ihm_lines) + "\n")
    with open(os.path.join(out_dir, "manifest-mix.tsv"), "w") as f:
        f.write("\n".join(mix_lines) + "\n")

    parquet_files = {
        os.path.basename(p): sha256_file(p)
        for p in sorted(glob.glob(os.path.join(parquet_dir, "test-*.parquet")))
    }
    prov = {
        "dataset": DATASET,
        "config": CONFIG,
        "split": SPLIT,
        "revision": args.revision,
        "parquet": parquet_files,
        "selection": {
            "mode": "full" if args.full else "sample",
            "pair_meetings_all_rows": pair_meetings,
            "other_meetings_sample_per_meeting": None if args.full else args.sample_per_meeting,
            "rule": "all rows of the pair meetings; every other meeting sampled "
            "evenly over audio_id (midpoints of n equal bins), deterministic"
            if not args.full
            else "every row of the split",
        },
        "counts": {
            "ihm_rows": len(ihm_lines),
            "mix_rows": len(mix_lines),
            "meetings": len(by_meeting),
            "rows_in_split": len(meta),
            "wavs_written_this_run": written,
            "bytes_written_this_run": total_bytes,
        },
        "skipped_mix_missing_audio": skipped_mix,
    }
    with open(os.path.join(out_dir, "provenance.json"), "w") as f:
        json.dump(prov, f, indent=1)

    ihm_s = sum(r["end"] - r["begin"] for r in rows_out)
    mix_s = sum(r["end"] - r["begin"] for r in rows_out if "mix_wav" in r)
    print(
        f"prepare ({prov['selection']['mode']}): {len(meta)} rows in the {SPLIT} split, "
        f"{len(by_meeting)} meetings; selected {len(ihm_lines)} IHM rows "
        f"({ihm_s / 3600:.2f} h) and {len(mix_lines)} Mix rows ({mix_s / 3600:.2f} h), "
        f"{written} wavs written"
    )
    if skipped_mix:
        print(f"prepare: {len(skipped_mix)} pair rows have no Mix-Headset wav (skipped)")
    print(f"prepare: -> {out_dir}")


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


# ---------------------------------------------------------------------------
# Scoring: corpus WER is the sum of per-utterance edit distances over the sum
# of reference words (the leaderboard method), never one alignment over
# concatenated text. Every CI resamples clusters, not utterance rows.


def _norm_fn(normalizer):
    from score import legacy_tokens, whisper_tokens

    return legacy_tokens if normalizer == "legacy" else whisper_tokens


def _no_filler_fn(normalizer):
    from score import FILLERS

    base = _norm_fn(normalizer)

    def fn(words):
        return [t for t in base(words) if t.lower().strip(".,?!") not in FILLERS]

    return fn


def utt_counts(ref_text, hyp_text, fn):
    """(errors, ref_words, del, ins, sub) for one utterance.

    An empty normalised reference keeps its hypothesis tokens as pure
    insertions in the edit sum (the denominator gains nothing), which is what
    summing jiwer-style per-pair measures does. `score` reports the alternative
    that drops those rows too.
    """
    from score import wer_tokens

    w = wer_tokens(fn(ref_text), fn(hyp_text))
    c = {
        "errors": w["errors"],
        "ref_words": w["ref_words"],
        "del": w.get("del", 0),
        "ins": w.get("ins", 0),
        "sub": w.get("sub", 0),
        "hyp_words": w["hyp_words"],
    }
    if w["ref_words"] == 0:
        # wer_tokens reports no del/ins/sub for an empty reference; the
        # hypothesis tokens are all insertions. Keep the breakdown equal to
        # the error count so the reported columns add up to the WER.
        c["ins"] = w["errors"]
    return c


def _row_counts(rows, hyps, normalizer):
    fn = _norm_fn(normalizer)
    out = []
    for r in rows:
        hyp = hyps.get(r["id"], {}).get("text", "")
        c = utt_counts(r["text"], hyp, fn)
        c["speaker"] = r.get("speaker", r["id"])
        c["meeting"] = r["meeting"]
        out.append(c)
    return out


def _sum_counts(counts):
    total = {"errors": 0, "ref_words": 0, "del": 0, "ins": 0, "sub": 0, "hyp_words": 0}
    for c in counts:
        for k in total:
            total[k] += c[k]
    total["wer"] = total["errors"] / total["ref_words"] if total["ref_words"] else 0.0
    return total


def aggregate(rows, hyps, normalizer="whisper"):
    return _sum_counts(_row_counts(rows, hyps, normalizer))


def _percentile(values, lo=0.025, hi=0.975):
    values.sort()
    return (
        values[int(lo * (len(values) - 1))],
        values[int(hi * (len(values) - 1))],
    )


def _cluster_key(c, cluster):
    if cluster == "meeting":
        return c["meeting"]
    if cluster == "utterance":
        return id(c)
    # AMI reuses role-like speaker ids across meetings (MEE073 is a different
    # person in EN2002a and EN2002d), so the speaker cluster is the pair.
    return (c["meeting"], c["speaker"])


def bootstrap_ci(rows, hyps, normalizer="whisper", cluster="speaker", n=2000, seed=19):
    """Percentile CI of corpus WER, resampling clusters (speaker or meeting).

    Resampling utterances ignores that one speaker or meeting contributes many
    correlated rows. Clusters are the right unit; `cluster="utterance"` is kept
    only to show how much narrower the old interval was.
    """
    import random

    counts = _row_counts(rows, hyps, normalizer)
    groups = {}
    for c in counts:
        groups.setdefault(_cluster_key(c, cluster), []).append(c)
    keys = sorted(groups)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        err = ref = 0
        for _ in keys:
            for c in groups[keys[rng.randrange(len(keys))]]:
                err += c["errors"]
                ref += c["ref_words"]
        if ref:
            out.append(err / ref)
    lo, hi = _percentile(out)
    return {"lo": lo, "hi": hi, "resamples": n, "clusters": len(keys), "cluster": cluster}


def bootstrap_delta(
    rows, hyps_a, hyps_b, normalizer="whisper", cluster="speaker", n=2000, seed=19
):
    """Paired cluster bootstrap of WER(b) - WER(a) on the same rows."""
    import random

    ca = _row_counts(rows, hyps_a, normalizer)
    cb = _row_counts(rows, hyps_b, normalizer)
    groups = {}
    for a, b in zip(ca, cb):
        groups.setdefault(_cluster_key(a, cluster), []).append((a, b))
    keys = sorted(groups)
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        ea = ra = eb = rb = 0
        for _ in keys:
            for a, b in groups[keys[rng.randrange(len(keys))]]:
                ea += a["errors"]
                ra += a["ref_words"]
                eb += b["errors"]
                rb += b["ref_words"]
        if ra and rb:
            out.append(eb / rb - ea / ra)
    lo, hi = _percentile(out)
    return {"lo": lo, "hi": hi, "resamples": n, "clusters": len(keys), "cluster": cluster}


def group_block(rows, hyps, normalizer, n_boot):
    counts = _row_counts(rows, hyps, normalizer)
    total = _sum_counts(counts)
    total["utterances"] = len(rows)
    total["audio_seconds"] = sum(r["end"] - r["begin"] for r in rows)
    total["empty_hypotheses"] = sum(
        1 for r in rows if not hyps.get(r["id"], {}).get("text", "").strip()
    )
    total["empty_references"] = sum(1 for r in rows if not _norm_fn(normalizer)(r["text"]))
    return total


def cmd_score(args):
    rows = json.load(open(require(args.rows, "rows", "ROOT19")))
    ihm = read_jsonl(require(args.ihm_jsonl, "ihm-jsonl", "ROOT19"))
    mix = read_jsonl(require(args.mix_jsonl, "mix-jsonl", "ROOT19")) if args.mix_jsonl else {}

    pair = [r for r in rows if r["group"] == "ihm_pair"]
    rest = [r for r in rows if r["group"] != "ihm_pair"]
    groups = {"ihm_pair": pair, "ihm_rest": rest, "ihm_all": rows}
    mix_rows = [r for r in rows if "mix_wav" in r and r["id"] in mix]
    groups["mix_pair"] = mix_rows

    n_boot = args.bootstrap
    result = {
        "groups": {},
        "per_meeting": {},
        "pair_per_meeting": {},
        "conventions": {
            "wer": "sum of per-utterance edit distances / sum of reference words",
            "empty_reference": "kept as pure insertions in the edit sum; the "
            "denominator does not grow (jiwer-style per-pair accumulation). "
            "`*_drop_empty` reports the alternative.",
            "ci": f"percentile bootstrap over {n_boot} resamples, seed {args.seed}",
        },
    }
    for name, rs in groups.items():
        if not rs:
            continue
        hyps = mix if name == "mix_pair" else ihm
        block = {
            "utterances": len(rs),
            "audio_seconds": sum(r["end"] - r["begin"] for r in rs),
            "whisper": group_block(rs, hyps, "whisper", n_boot),
            "legacy": group_block(rs, hyps, "legacy", n_boot),
            "whisper_no_fillers": group_block(rs, hyps, "whisper", n_boot),
            "ci95": {},
        }
        # Recompute filler-free with the filler-stripping normalizer.
        fn = _no_filler_fn("whisper")
        block["whisper_no_fillers"] = _sum_counts(
            [utt_counts(r["text"], hyps.get(r["id"], {}).get("text", ""), fn) for r in rs]
        )
        for cluster in ("speaker", "meeting", "utterance"):
            block["ci95"][cluster] = bootstrap_ci(
                rs, hyps, "whisper", cluster=cluster, n=n_boot, seed=args.seed
            )
        # Drop the 246/177 references that normalise to empty.
        kept = [r for r in rs if _norm_fn("whisper")(r["text"])]
        block["whisper_drop_empty"] = _sum_counts(
            [
                utt_counts(
                    r["text"], hyps.get(r["id"], {}).get("text", ""), _norm_fn("whisper")
                )
                for r in kept
            ]
        )
        result["groups"][name] = block

    if "ihm_pair" in result["groups"] and "mix_pair" in result["groups"]:
        result["paired_delta"] = {
            "point": result["groups"]["mix_pair"]["whisper"]["wer"]
            - result["groups"]["ihm_pair"]["whisper"]["wer"],
            "ci95": {
                c: bootstrap_delta(
                    groups["mix_pair"], ihm, mix, "whisper", cluster=c, n=n_boot, seed=args.seed
                )
                for c in ("speaker", "meeting", "utterance")
            },
        }

    for m in sorted({r["meeting"] for r in rows}):
        rs = [r for r in rows if r["meeting"] == m]
        result["per_meeting"][m] = {
            "utterances": len(rs),
            "whisper": _sum_counts(_row_counts(rs, ihm, "whisper")),
            "ci95_speaker": bootstrap_ci(
                rs, ihm, "whisper", cluster="speaker", n=n_boot, seed=args.seed
            ),
        }
    for m in sorted({r["meeting"] for r in mix_rows}):
        rs = [r for r in mix_rows if r["meeting"] == m]
        result["pair_per_meeting"][m] = {
            "utterances": len(rs),
            "ihm_whisper": _sum_counts(_row_counts(rs, ihm, "whisper")),
            "mix_whisper": _sum_counts(_row_counts(rs, mix, "whisper")),
            "delta": _sum_counts(_row_counts(rs, mix, "whisper"))["wer"]
            - _sum_counts(_row_counts(rs, ihm, "whisper"))["wer"],
            "delta_ci95_speaker": bootstrap_delta(
                rs, ihm, mix, "whisper", cluster="speaker", n=n_boot, seed=args.seed
            ),
        }

    result["crosstalk"] = crosstalk_tables(mix_rows, ihm, mix, args.refs_dir)
    result["crosstalk_legacy_buckets"] = crosstalk_legacy(mix_rows, ihm, mix, args.refs_dir)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)

    def pct(x):
        return f"{100 * x:.1f}%"

    def ci(block, key="ci95", cluster="speaker"):
        c = block.get(key, {}).get(cluster)
        return f" [{100 * c['lo']:.1f}-{100 * c['hi']:.1f}]" if c else ""

    def wer_row(name, b, key="whisper"):
        w = b[key]
        return (
            f"| {name} | {b['utterances']} | {b['audio_seconds'] / 60:.1f} | "
            f"{pct(w['wer'])}{ci(b)} | {w['del']} / {w['ins']} / {w['sub']} | "
            f"{w['ref_words']} |"
        )

    print("\n### IHM vs Mix, same utterances (per-utterance WER)\n")
    print(
        "| Set | Utterances | Audio min | WER (Whisper) [95% CI, speaker] | del / ins / sub | Ref words |"
    )
    print("|---|---|---|---|---|---|")
    for name in ("ihm_pair", "mix_pair", "ihm_rest", "ihm_all"):
        if name in result["groups"]:
            print(wer_row(name, result["groups"][name]))
    if "paired_delta" in result:
        d = result["paired_delta"]
        c = d["ci95"]["speaker"]
        print(
            f"\nPaired Mix - IHM: **{100 * d['point']:+.1f} points** "
            f"[{100 * c['lo']:+.1f}, {100 * c['hi']:+.1f}, speakers]"
        )

    print("\n### Per-meeting IHM\n")
    print("| Meeting | Utterances | WER [95% CI, speaker] |")
    print("|---|---|---|")
    for m, b in result["per_meeting"].items():
        print(
            f"| {m} | {b['utterances']} | {pct(b['whisper']['wer'])}"
            f" [{100 * b['ci95_speaker']['lo']:.1f}-{100 * b['ci95_speaker']['hi']:.1f}] |"
        )

    if result["pair_per_meeting"]:
        print("\n### Paired IHM vs Mix per meeting\n")
        print("| Meeting | Utterances | IHM WER | Mix WER | Delta [95% CI, speaker] |")
        print("|---|---|---|---|---|")
        for m, b in result["pair_per_meeting"].items():
            d = b["delta_ci95_speaker"]
            print(
                f"| {m} | {b['utterances']} | {pct(b['ihm_whisper']['wer'])} | "
                f"{pct(b['mix_whisper']['wer'])} | {100 * b['delta']:+.1f} "
                f"[{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] |"
            )

    print("\n### Normalizer and empty-reference diagnostics\n")
    print("| Set | WER Whisper | WER legacy | WER no fillers | drop empty refs | Empty refs |")
    print("|---|---|---|---|---|---|")
    for name, b in result["groups"].items():
        print(
            f"| {name} | {pct(b['whisper']['wer'])} | {pct(b['legacy']['wer'])} | "
            f"{pct(b['whisper_no_fillers']['wer'])} | "
            f"{pct(b['whisper_drop_empty']['wer'])} | {b['whisper']['empty_references']} |"
        )

    print("\n### Crosstalk: share of the window with 2+ speakers (incl. target)\n")
    print("| Share | Duration band | Utterances | IHM WER | Mix WER | Delta |")
    print("|---|---|---|---|---|---|")
    for b in result["crosstalk"]:
        print(
            f"| {b['share']} | {b['duration']} | {b['utterances']} | "
            f"{pct(b['ihm_wer'])} | {pct(b['mix_wer'])} | {100 * b['delta']:+.1f} |"
        )

    print("\n### Legacy crosstalk buckets (review cross-check)\n")
    print("| Bucket | Utterances | IHM WER | Mix WER | Delta |")
    print("|---|---|---|---|---|")
    for b in result["crosstalk_legacy_buckets"]:
        print(
            f"| {b['bucket']} | {b['utterances']} | {pct(b['ihm_wer'])} | "
            f"{pct(b['mix_wer'])} | {100 * b['delta']:+.1f} |"
        )


# ---------------------------------------------------------------------------
# Crosstalk buckets


def _speaker_grids(rows, refs_dir):
    """Per meeting: (active boolean matrix per speaker, 2+ count, frames)."""
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
        two_plus = stack.sum(axis=0) >= 2
        grids[meeting] = (two_plus, n)
    return grids


def _share_two(rows, refs_dir):
    grids = _speaker_grids(rows, refs_dir)
    step = 0.01
    out = {}
    for r in rows:
        g = grids.get(r["meeting"])
        if g is None:
            continue
        two_plus, n = g
        i = max(0, int(r["begin"] / step))
        j = min(n, max(i + 1, int(r["end"] / step)))
        out[r["id"]] = float(two_plus[i:j].mean()) if j > i else 0.0
    return out


def _band(seconds):
    if seconds < 1.0:
        return "<1 s (backchannel)"
    if seconds < 3.0:
        return "1-3 s"
    if seconds < 8.0:
        return "3-8 s"
    return ">=8 s"


def _share_bucket(share):
    if share == 0.0:
        return "0 (solo)"
    if share <= 0.2:
        return "0-20%"
    if share <= 0.5:
        return "20-50%"
    return "50-100%"


def crosstalk_tables(mix_rows, ihm, mix, refs_dir):
    shares = _share_two(mix_rows, refs_dir)
    buckets = {}
    for r in mix_rows:
        if r["id"] not in shares:
            continue
        key = (_share_bucket(shares[r["id"]]), _band(r["end"] - r["begin"]))
        buckets.setdefault(key, []).append(r)
    out = []
    order = {
        "0 (solo)": 0,
        "0-20%": 1,
        "20-50%": 2,
        "50-100%": 3,
    }
    band_order = {"<1 s (backchannel)": 0, "1-3 s": 1, "3-8 s": 2, ">=8 s": 3}
    for key in sorted(buckets, key=lambda k: (order[k[0]], band_order[k[1]])):
        rs = buckets[key]
        a = _sum_counts(_row_counts(rs, ihm, "whisper"))["wer"]
        b = _sum_counts(_row_counts(rs, mix, "whisper"))["wer"]
        out.append(
            {
                "share": key[0],
                "duration": key[1],
                "utterances": len(rs),
                "audio_seconds": sum(r["end"] - r["begin"] for r in rs),
                "ihm_wer": a,
                "mix_wer": b,
                "delta": b - a,
            }
        )
    # Total row for the report.
    a = _sum_counts(_row_counts(mix_rows, ihm, "whisper"))["wer"]
    b = _sum_counts(_row_counts(mix_rows, mix, "whisper"))["wer"]
    out.append(
        {
            "share": "all",
            "duration": "all",
            "utterances": len(mix_rows),
            "audio_seconds": sum(r["end"] - r["begin"] for r in mix_rows),
            "ihm_wer": a,
            "mix_wer": b,
            "delta": b - a,
        }
    )
    return out


def crosstalk_legacy(mix_rows, ihm, mix, refs_dir):
    """The review's three buckets, for the 44.6% cross-check."""
    shares = _share_two(mix_rows, refs_dir)
    buckets = {}
    for r in mix_rows:
        share = shares.get(r["id"])
        if share is None:
            continue
        if share == 0.0:
            key = "solo (no other speaker)"
        elif share <= 0.2:
            key = "2+ speakers <= 20%"
        else:
            key = "2+ speakers > 20%"
        buckets.setdefault(key, []).append(r)
    out = []
    for key in ("solo (no other speaker)", "2+ speakers <= 20%", "2+ speakers > 20%"):
        rs = buckets.get(key)
        if not rs:
            continue
        a = _sum_counts(_row_counts(rs, ihm, "whisper"))["wer"]
        b = _sum_counts(_row_counts(rs, mix, "whisper"))["wer"]
        out.append(
            {
                "bucket": key,
                "utterances": len(rs),
                "audio_seconds": sum(r["end"] - r["begin"] for r in rs),
                "ihm_wer": a,
                "mix_wer": b,
                "delta": b - a,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Test-set-weighted estimate from the sample


def cmd_weighted(args):
    sample = json.load(open(require(args.sample_rows, "sample-rows", "ROOT19")))
    full = json.load(open(require(args.full_rows, "full-rows", "ROOT19")))
    hyps = read_jsonl(require(args.sample_jsonl, "sample-jsonl", "ROOT19"))
    fn = _norm_fn("whisper")

    full_words = {}
    for r in full:
        full_words[r["meeting"]] = full_words.get(r["meeting"], 0) + len(fn(r["text"]))
    total_words = sum(full_words.values())

    import random

    by_meeting = {}
    for r in sample:
        by_meeting.setdefault(r["meeting"], []).append(
            utt_counts(r["text"], hyps.get(r["id"], {}).get("text", ""), fn)
        )
    share = {m: full_words.get(m, 0) / total_words for m in by_meeting}
    per_meeting = {}
    for m, cs in by_meeting.items():
        tot = _sum_counts(cs)
        per_meeting[m] = {
            "sample_utterances": len(cs),
            "sample_wer": tot["wer"],
            "full_split_word_share": share[m],
            "full_split_words": full_words.get(m, 0),
        }
    point = sum(share[m] * per_meeting[m]["sample_wer"] for m in by_meeting)
    # Unweighted sample number, for contrast.
    unweighted = _sum_counts([c for cs in by_meeting.values() for c in cs])["wer"]

    rng = random.Random(args.seed)
    out = []
    for _ in range(args.bootstrap):
        value = 0.0
        for m, cs in by_meeting.items():
            picked = [cs[rng.randrange(len(cs))] for _ in range(len(cs))]
            tot = _sum_counts(picked)
            value += share[m] * tot["wer"]
        out.append(value)
    lo, hi = _percentile(out)
    result = {
        "unweighted_sample_wer": unweighted,
        "weighted_estimate": point,
        "weighted_ci95": {"lo": lo, "hi": hi, "resamples": args.bootstrap, "seed": args.seed},
        "method": "each meeting's per-utterance sample WER weighted by its share "
        "of the full split's Whisper-normalised reference words; utterances "
        "resampled within meeting (stratified), seed fixed",
        "per_meeting": per_meeting,
        "full_split_reference_words": total_words,
    }
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    print(
        f"sample unweighted {100 * unweighted:.2f}% -> test-set-weighted "
        f"{100 * point:.2f}% [95% CI {100 * lo:.2f}-{100 * hi:.2f}]"
    )
    for m in sorted(per_meeting):
        b = per_meeting[m]
        print(
            f"  {m}: sample {100 * b['sample_wer']:.2f}% x "
            f"{100 * b['full_split_word_share']:.1f}% of split words"
        )


# ---------------------------------------------------------------------------
# Same-reference ablation: oracle segments vs the engine VAD pipeline


def cmd_ablate_prepare(args):
    rows = json.load(open(require(args.rows, "rows", "ROOT19")))
    mix_dir = require(args.mix_dir, "mix-dir", "ROOT")
    out_dir = require(args.out_dir, "out-dir", "ROOT19")
    os.makedirs(out_dir, exist_ok=True)
    meetings = [m for m in args.meetings.split(",") if m]
    for m in meetings:
        rs = sorted((r for r in rows if r["meeting"] == m), key=lambda r: r["begin"])
        wav = os.path.join(mix_dir, f"{m}.Mix-Headset.wav")
        if not os.path.exists(wav):
            sys.exit(f"error: missing {wav}")
        lines = [f"{r['id']}\t{wav}\t{r['begin']:.3f}\t{r['end']:.3f}" for r in rs]
        path = os.path.join(out_dir, f"manifest-{m}.tsv")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        audio = sum(r["end"] - r["begin"] for r in rs)
        print(f"ablate-prepare: {m}: {len(rs)} oracle segments, {audio / 60:.1f} min -> {path}")


def _load_vad_words(path):
    d = json.load(open(path))
    if "asr" in d:
        return d["asr"].get("words", []), d["asr"].get("vad", {})
    return d.get("words", []), d.get("vad", {})


def _assign_rows(rows, words, offset=0.0):
    """Assign each hypothesis word to exactly one reference row by midpoint.

    Rows can overlap in time (crosstalk), so pick the shortest window that
    contains the word midpoint. Words outside every window are dropped; the
    reference is the same union of windows on both sides.
    """
    bounds = sorted(((r["id"], r["begin"], r["end"]) for r in rows), key=lambda x: x[1])
    by_row = {r["id"]: [] for r in rows}
    outside = 0
    for w in words:
        mid = (w["start"] + w["end"]) / 2 + offset
        best = None
        for rid, a, b in bounds:
            if a <= mid < b and (best is None or (b - a) < (best[2] - best[1])):
                best = (rid, a, b)
        if best:
            by_row[best[0]].append(w["text"])
        else:
            outside += 1
    return {k: " ".join(v) for k, v in by_row.items()}, outside


def cmd_ablate_score(args):
    rows = json.load(open(require(args.rows, "rows", "ROOT19")))
    meetings = [m for m in args.meetings.split(",") if m]
    oracle_paths = dict(x.split("=", 1) for x in args.oracle.split(",") if x)
    vad_paths = dict(x.split("=", 1) for x in args.vad.split(",") if x)
    fn = _norm_fn("whisper")

    per_meeting = {}
    all_counts = {"oracle": [], "vad": []}
    for m in meetings:
        rs = [r for r in rows if r["meeting"] == m]
        oracle = read_jsonl(oracle_paths[m])
        words, _vad = _load_vad_words(vad_paths[m])
        vad_hyp, outside = _assign_rows(rs, words)
        oc = _row_counts(rs, oracle, "whisper")
        vc = [
            utt_counts(r["text"], vad_hyp.get(r["id"], ""), fn)
            for r in rs
        ]
        for c, r in zip(vc, rs):
            c["speaker"] = r.get("speaker", r["id"])
            c["meeting"] = m
        o = _sum_counts(oc)
        v = _sum_counts(vc)
        per_meeting[m] = {
            "utterances": len(rs),
            "oracle": o,
            "vad": v,
            "delta": v["wer"] - o["wer"],
            "vad_words_outside_segments": outside,
            "delta_ci95_speaker": bootstrap_delta(
                rs,
                oracle,
                {r["id"]: {"text": vad_hyp.get(r["id"], "")} for r in rs},
                "whisper",
                cluster="speaker",
                n=args.bootstrap,
                seed=args.seed,
            ),
        }
        all_counts["oracle"].extend(oc)
        all_counts["vad"].extend(vc)

    o = _sum_counts(all_counts["oracle"])
    v = _sum_counts(all_counts["vad"])
    all_rows = [r for r in rows if r["meeting"] in meetings]
    overall_hyps = {}
    for m in meetings:
        oracle = read_jsonl(oracle_paths[m])
        words, _vad = _load_vad_words(vad_paths[m])
        vad_hyp, _outside = _assign_rows(
            [r for r in rows if r["meeting"] == m], words
        )
        for rid, text in vad_hyp.items():
            overall_hyps[rid] = {"text": text, "oracle": oracle.get(rid, {}).get("text", "")}
    oracle_all = {rid: {"text": x["oracle"]} for rid, x in overall_hyps.items()}
    vad_all = {rid: {"text": x["text"]} for rid, x in overall_hyps.items()}
    result = {
        "reference": "the HF IHM segment text of the same meeting; per-utterance "
        "WER over the segment windows. Oracle hypotheses are the Mix-Headset "
        "decode of the dataset segments; pipeline hypotheses are the full-meeting "
        "VAD decode with each word assigned to the containing segment window.",
        "per_meeting": per_meeting,
        "overall": {
            "oracle": o,
            "vad": v,
            "delta": v["wer"] - o["wer"],
            "delta_ci95_speaker": bootstrap_delta(
                all_rows,
                oracle_all,
                vad_all,
                "whisper",
                cluster="speaker",
                n=args.bootstrap,
                seed=args.seed,
            ),
        },
    }
    if args.window_300_600:
        result["window_300_600_es2004b"] = json.load(open(args.window_300_600))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    print("### Same-reference ablation: oracle segments vs engine VAD pipeline\n")
    print("| Meeting | Segments | Oracle WER | VAD WER | Delta [95% CI, speaker] |")
    print("|---|---|---|---|---|")
    for m, b in per_meeting.items():
        d = b["delta_ci95_speaker"]
        print(
            f"| {m} | {b['utterances']} | {100 * b['oracle']['wer']:.1f}% | "
            f"{100 * b['vad']['wer']:.1f}% | {100 * b['delta']:+.1f} "
            f"[{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}] |"
        )
    print(
        f"\nAll: oracle {100 * o['wer']:.1f}% (del {o['del']} / ins {o['ins']} / sub {o['sub']}, "
        f"{o['ref_words']} words), VAD {100 * v['wer']:.1f}% "
        f"(del {v['del']} / ins {v['ins']} / sub {v['sub']}), "
        f"delta {100 * (v['wer'] - o['wer']):+.1f} points "
        f"[{100 * result['overall']['delta_ci95_speaker']['lo']:+.1f}, "
        f"{100 * result['overall']['delta_ci95_speaker']['hi']:+.1f}, speakers]"
    )
    if args.window_300_600:
        w = result["window_300_600_es2004b"]
        print(
            "300-600 s ES2004b (one words.xml reference, earlier check): "
            f"oracle {100 * w['oracle']['whisper']['wer']:.1f}% vs "
            f"VAD {100 * w['vad']['wer_whisper']:.1f}% "
            f"(+{100 * (w['vad']['wer_whisper'] - w['oracle']['whisper']['wer']):.1f} points, "
            f"{w['oracle']['whisper']['ref_words']} words)"
        )


# ---------------------------------------------------------------------------
# Cause analysis


def _split_run_pieces(ranges, audio_len, max_piece=30.0):
    """Python mirror of group_vad_pieces (no audio energy, so no mid-run split).

    The VAD never emits a run longer than `max_speech_duration` (30 s), so the
    quiet-point split in split_run does not fire; the grouping rule is the
    join gap.
    """
    join_gap = 0.5
    pre = 2.0
    post = 1.0
    min_piece = 10.0
    groups = []
    for a, b in ranges:
        if b - a > max_piece:
            # Conservative: cut in the middle rather than call a quiet point.
            start = a
            while b - start > max_piece:
                mid = start + (max_piece / 2)
                groups.append((start, mid))
                start = mid
            if b > start:
                groups.append((start, b))
        else:
            groups.append((a, b))
    merged = []
    for a, b in groups:
        if merged:
            last = merged[-1]
            if b - last[0] <= max_piece and a - last[1] <= join_gap:
                merged[-1] = (last[0], b)
                continue
        merged.append((a, b))
    pieces = []
    for i, (a, b) in enumerate(merged):
        start = 0.0 if i == 0 else max(0.0, a - pre)
        end = min(float(audio_len), max(b + post, start + min_piece))
        pieces.append((start, end))
    return pieces


def _hyp_gaps(words, min_gap=3.0):
    ws = sorted(words, key=lambda w: w["start"])
    gaps = []
    for prev, cur in zip(ws, ws[1:]):
        start = prev["end"]
        end = cur["start"]
        if end - start >= min_gap:
            gaps.append((start, end))
    return gaps


def cmd_cause(args):
    from score import whisper_tokens, wer_tokens

    rows = json.load(open(require(args.rows, "rows", "ROOT19")))
    meetings = [m for m in args.meetings.split(",") if m]
    vad_paths = dict(x.split("=", 1) for x in args.vad.split(",") if x)
    refs_dir = require(args.refs_dir, "refs-dir", "ROOT")
    fn = _norm_fn("whisper")
    result = {"meetings": {}}
    print("### Where the VAD-pipeline deletions come from\n")
    print(
        "| Meeting | Hyp words | Flush pieces / pieces | Gaps >=3 s | Ref words in gaps "
        "| Deletions | In gaps | At piece start (gaps / words) "
        "| At flush start (gaps / words) | 15 s+ holes (gaps / words) |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for m in meetings:
        rs = [r for r in rows if r["meeting"] == m]
        words, vad = _load_vad_words(vad_paths[m])
        ref = json.load(open(os.path.join(refs_dir, f"{m}.ref.json")))
        audio_len = max(w["end"] for w in ref["words"]) + 1.0
        pieces = _split_run_pieces(vad.get("ranges", []), audio_len)
        # Piece boundaries after the midpoint de-duplication: where the next
        # piece takes over from the previous one. `flush` marks a boundary
        # created by the 20 s VAD flush: the previous run is ~19-20 s and the
        # re-pickup gap is 0.5-1.2 s, larger than VAD_JOIN_GAP_SECONDS (0.5),
        # so group_vad_pieces never rejoins them.
        boundaries = []
        ranges = vad.get("ranges", [])
        flush_flags = []
        for (a0, b0), (a1, b1) in zip(ranges, ranges[1:]):
            flush_flags.append(0.5 < a1 - b0 <= 1.2 and 18.0 <= b0 - a0 <= 20.5)
        for i, ((a0, b0), (a1, b1)) in enumerate(zip(pieces, pieces[1:])):
            # The piece end is b + 1 s post-roll and the next start is a - 2 s
            # pre-roll, so the de-duplication midpoint is (b0 + a1 - 1) / 2.
            boundaries.append(((b0 + a1 - 1.0) / 2, i < len(flush_flags) and flush_flags[i]))
        # Every piece start (after the 2 s pre-roll), with its flush flag.
        starts = [(0.0, False)] + [
            (max(0.0, a - 2.0), i - 1 < len(flush_flags) and flush_flags[i - 1])
            for i, (a, b) in enumerate(pieces[1:], start=1)
        ]
        flush_starts = sum(1 for s, f in starts if f)
        gaps = _hyp_gaps(words)
        ref_words_in = 0
        gap_at_start = 0
        words_at_start = 0
        gap_at_flush = 0
        words_at_flush = 0
        empty_piece = 0
        words_empty_piece = 0
        gap_rows = []
        for gs, ge in gaps:
            n = sum(1 for w in ref["words"] if gs <= (w["start"] + w["end"]) / 2 < ge)
            ref_words_in += n
            # A zero-word piece leaves a hole that contains its start (the
            # hole can begin in the previous piece and end in the next one),
            # so count both a hole that starts at a boundary and one that
            # contains a piece start.
            dist = min(abs(gs - b[0]) for b in boundaries) if boundaries else 1e9
            in_piece = [f for s, f in starts if gs - 2.0 <= s <= ge + 2.0]
            at_start = dist <= 2.0 or bool(in_piece)
            at_flush = any(in_piece) or (
                dist <= 2.0
                and boundaries
                and min(boundaries, key=lambda b: abs(gs - b[0]))[1]
            )
            if at_start:
                gap_at_start += 1
                words_at_start += n
            if at_flush:
                gap_at_flush += 1
                words_at_flush += n
            if at_start and ge - gs >= 15.0:
                empty_piece += 1
                words_empty_piece += n
            gap_rows.append(
                {
                    "start": gs,
                    "end": ge,
                    "seconds": ge - gs,
                    "ref_words": n,
                    "distance_to_piece_boundary": dist,
                    "at_piece_start": at_start,
                    "at_flush_boundary": at_flush,
                }
            )
        ref_tokens = fn(" ".join(w["text"] for w in ref["words"]))
        hyp_tokens = fn(" ".join(w["text"] for w in words))
        total = wer_tokens(ref_tokens, hyp_tokens)
        result["meetings"][m] = {
            "hyp_words": len(words),
            "ref_words_whisper": len(ref_tokens),
            "gaps_ge_3s": len(gaps),
            "ref_words_in_gaps": ref_words_in,
            "total_deletions": total.get("del", 0),
            "gaps_at_piece_start": gap_at_start,
            "ref_words_in_gaps_at_piece_start": words_at_start,
            "gaps_at_flush_start": gap_at_flush,
            "ref_words_in_gaps_at_flush_start": words_at_flush,
            "empty_piece_holes": empty_piece,
            "ref_words_in_empty_piece_holes": words_empty_piece,
            "flush_piece_starts": flush_starts,
            "pieces": len(pieces),
            "gaps": gap_rows,
        }
        print(
            f"| {m} | {len(words)} | {flush_starts}/{len(pieces)} | {len(gaps)} | "
            f"{ref_words_in} | {total.get('del', 0)} | "
            f"{100 * ref_words_in / max(total.get('del', 1), 1):.0f}% | "
            f"{gap_at_start} / {words_at_start} | {gap_at_flush} / {words_at_flush} | "
            f"{empty_piece} / {words_empty_piece} |"
        )
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    q = sub.add_parser("prepare", help="unpack the IHM split and build manifests")
    q.add_argument("--parquet-dir", default=env_path("ROOT19", "data/ihm-test"))
    q.add_argument("--out-dir", default=env_path("ROOT19", "data/ihm-baseline"))
    q.add_argument("--mix-dir", default=env_path("ROOT", "data"))
    q.add_argument("--pair-meetings", default="ES2004b,IS1009b")
    q.add_argument("--sample-per-meeting", type=int, default=60)
    q.add_argument("--full", action="store_true", help="select every row of the split")
    q.add_argument("--revision", default=REVISION)
    q.set_defaults(func=cmd_prepare)

    q = sub.add_parser("score", help="WER tables from the batch decode JSONL")
    q.add_argument("--rows", default=env_path("ROOT19", "data/ihm-baseline/rows.json"))
    q.add_argument(
        "--ihm-jsonl", default=env_path("ROOT19", "results/ihm-decode.jsonl")
    )
    q.add_argument(
        "--mix-jsonl", default=env_path("ROOT19", "results/mix-decode.jsonl")
    )
    q.add_argument("--refs-dir", default=env_path("ROOT", "data"))
    q.add_argument("--out", default=env_path("ROOT19", "results/ihm-summary.json"))
    q.add_argument("--bootstrap", type=int, default=2000)
    q.add_argument("--seed", type=int, default=19)
    q.set_defaults(func=cmd_score)

    q = sub.add_parser("weighted", help="test-set-weighted estimate from the sample")
    q.add_argument(
        "--sample-rows", default=env_path("ROOT19", "data/ihm-baseline/rows.json")
    )
    q.add_argument(
        "--sample-jsonl", default=env_path("ROOT19", "results/ihm-decode.jsonl")
    )
    q.add_argument(
        "--full-rows", default=env_path("ROOT19", "data/ihm-baseline-full/rows.json")
    )
    q.add_argument(
        "--out",
        default=env_path("ROOT19", "results/ihm-weighted-estimate.json"),
    )
    q.add_argument("--bootstrap", type=int, default=4000)
    q.add_argument("--seed", type=int, default=19)
    q.set_defaults(func=cmd_weighted)

    q = sub.add_parser("ablate-prepare", help="oracle-segment manifests for the ablation")
    q.add_argument(
        "--rows", default=env_path("ROOT19", "data/ihm-baseline-full/rows.json")
    )
    q.add_argument("--mix-dir", default=env_path("ROOT", "data"))
    q.add_argument("--out-dir", default=env_path("ROOT19", "data/ablation"))
    q.add_argument("--meetings", default="ES2004b,IS1009b,EN2002a")
    q.set_defaults(func=cmd_ablate_prepare)

    q = sub.add_parser("ablate-score", help="oracle vs VAD on the same reference")
    q.add_argument(
        "--rows", default=env_path("ROOT19", "data/ihm-baseline-full/rows.json")
    )
    q.add_argument("--meetings", default="ES2004b,IS1009b,EN2002a")
    q.add_argument("--oracle", required=True, help="meeting=path,... JSONL")
    q.add_argument("--vad", required=True, help="meeting=path,... result JSON")
    q.add_argument("--window-300-600", default=None)
    q.add_argument(
        "--out", default=env_path("ROOT19", "results/ablation-summary.json")
    )
    q.add_argument("--bootstrap", type=int, default=2000)
    q.add_argument("--seed", type=int, default=19)
    q.set_defaults(func=cmd_ablate_score)

    q = sub.add_parser("cause", help="deletions in hypothesis holes at piece starts")
    q.add_argument(
        "--rows", default=env_path("ROOT19", "data/ihm-baseline-full/rows.json")
    )
    q.add_argument("--meetings", default="ES2004b,IS1009b,EN2002a")
    q.add_argument("--vad", required=True, help="meeting=path,... result JSON")
    q.add_argument("--refs-dir", default=env_path("ROOT", "data"))
    q.add_argument(
        "--out", default=env_path("ROOT19", "results/cause-analysis.json")
    )
    q.set_defaults(func=cmd_cause)

    args = p.parse_args()

    # Resolve defaults that depend on the environment only now, so `--help`
    # works without the variables.
    if args.cmd == "prepare":
        args.parquet_dir = require(args.parquet_dir, "parquet-dir", "ROOT19")
        args.out_dir = require(args.out_dir, "out-dir", "ROOT19")
        args.mix_dir = require(args.mix_dir, "mix-dir", "ROOT")
    elif args.cmd == "score":
        args.rows = require(args.rows, "rows", "ROOT19")
        args.ihm_jsonl = require(args.ihm_jsonl, "ihm-jsonl", "ROOT19")
        args.mix_jsonl = args.mix_jsonl and require(args.mix_jsonl, "mix-jsonl", "ROOT19")
        args.refs_dir = require(args.refs_dir, "refs-dir", "ROOT")
        args.out = args.out and require(args.out, "out", "ROOT19")
    elif args.cmd == "weighted":
        args.sample_rows = require(args.sample_rows, "sample-rows", "ROOT19")
        args.sample_jsonl = require(args.sample_jsonl, "sample-jsonl", "ROOT19")
        args.full_rows = require(args.full_rows, "full-rows", "ROOT19")
    elif args.cmd == "ablate-prepare":
        args.rows = require(args.rows, "rows", "ROOT19")
        args.mix_dir = require(args.mix_dir, "mix-dir", "ROOT")
        args.out_dir = require(args.out_dir, "out-dir", "ROOT19")
    elif args.cmd == "ablate-score":
        args.rows = require(args.rows, "rows", "ROOT19")
    elif args.cmd == "cause":
        args.rows = require(args.rows, "rows", "ROOT19")
        args.refs_dir = require(args.refs_dir, "refs-dir", "ROOT")

    args.func(args)


if __name__ == "__main__":
    main()
