#!/usr/bin/env python3
"""Score tinta-bench JSON output against AMI manual annotations.

Subcommands:
  ref      build a reference JSON from the AMI manual annotation XML files
  score    WER + DER for one tinta-bench result JSON
  table    markdown summary table from several score JSON files

DER is frame based (10 ms), with a NIST-style collar and overlap control.
Requires: numpy, scipy, rapidfuzz (see docs/fork/benchmark.md).
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET

FILLERS = {"uh", "um", "er", "ah", "eh", "mm", "hmm", "uh-huh", "mhm", "oh"}


def norm_tokens(words):
    """Lowercase, strip punctuation and keep one token per word."""
    out = []
    for w in words:
        w = w.lower().replace("’", "'")
        w = re.sub(r"[^a-z0-9'\- ]+", " ", w)
        for piece in w.split():
            if piece:
                out.append(piece)
    return out


def build_ref(annotations, meeting, out_path):
    words = []
    for path in sorted(glob.glob(os.path.join(annotations, "words", f"{meeting}.*.words.xml"))):
        speaker = os.path.basename(path).split(".")[1]
        root = ET.parse(path).getroot()
        for node in root.iter():
            if node.tag.endswith("}w") or node.tag == "w":
                if node.get("punc") == "true":
                    continue
                text = (node.text or "").strip()
                if not text:
                    continue
                words.append(
                    {
                        "start": float(node.get("starttime")),
                        "end": float(node.get("endtime")),
                        "speaker": speaker,
                        "text": text,
                    }
                )
    words.sort(key=lambda w: (w["start"], w["speaker"]))

    segments = []
    for path in sorted(glob.glob(os.path.join(annotations, "segments", f"{meeting}.*.segments.xml"))):
        speaker = os.path.basename(path).split(".")[1]
        root = ET.parse(path).getroot()
        for node in root.iter():
            if node.tag.endswith("}segment") or node.tag == "segment":
                start = node.get("transcriber_start")
                end = node.get("transcriber_end")
                if start is None or end is None:
                    continue
                segments.append(
                    {"start": float(start), "end": float(end), "speaker": speaker}
                )
    segments.sort(key=lambda s: (s["start"], s["speaker"]))

    ref = {
        "meeting": meeting,
        "words": words,
        "segments": segments,
        "duration": max(w["end"] for w in words) if words else 0.0,
        "speakers": sorted({w["speaker"] for w in words}),
    }
    with open(out_path, "w") as f:
        json.dump(ref, f, indent=1)
    print(
        f"ref {meeting}: {len(words)} words, {len(segments)} segments, "
        f"{len(ref['speakers'])} speakers, {ref['duration'] / 60:.1f} min -> {out_path}"
    )


def clean_sample(args):
    """Cut regions with exactly one active speaker into one clean-speech wav.

    The output keeps the words spoken in those regions, in order, so a WER on
    it is comparable to published clean-speech numbers. It is a sanity check,
    not a replacement for scoring the mixed meeting audio.
    """
    import numpy as np
    import wave

    ref = json.load(open(args.ref))
    with wave.open(args.wav, "rb") as w:
        assert w.getnchannels() == 1 and w.getsampwidth() == 2 and w.getframerate() == 16000
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

    step = 0.01
    n = int(len(audio) / 16000 / step) + 1
    cover = np.zeros(n, dtype=np.int16)
    for s in ref["segments"]:
        cover[int(s["start"] / step) : int(s["end"] / step)] += 1

    # Runs with exactly one speaker, at least 2 s long.
    runs = []
    start = None
    for i, c in enumerate(cover):
        if c == 1 and start is None:
            start = i
        elif c != 1 and start is not None:
            if (i - start) * step >= 2.0:
                runs.append((start * step, i * step))
            start = None
    if start is not None and (n - start) * step >= 2.0:
        runs.append((start * step, n * step))

    limit = args.minutes * 60.0
    pieces = []
    words = []
    used = 0.0
    for a, b in runs:
        if used >= limit:
            break
        b = min(b, a + 20.0, a + (limit - used))
        if b - a < 1.0:
            continue
        pieces.append(audio[int(a * 16000) : int(b * 16000)])
        used += b - a
        for w in ref["words"]:
            if w["start"] >= a and w["end"] <= b:
                words.append(w)
        # 0.2 s of silence between regions.
        pieces.append(np.zeros(int(0.2 * 16000), dtype=np.int16))
        used += 0.2

    out = np.concatenate(pieces)
    with wave.open(args.out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(out.tobytes())
    json.dump({"meeting": ref["meeting"], "words": words}, open(args.out_ref, "w"), indent=1)
    print(
        f"clean sample: {used / 60:.2f} min, {len(words)} words, "
        f"{len(runs)} single-speaker regions -> {args.out_wav}"
    )


def wer(ref_words, hyp_words):
    from rapidfuzz.distance import Levenshtein

    r = norm_tokens(ref_words)
    h = norm_tokens(hyp_words)
    if not r:
        return {"wer": 0.0, "ref_words": 0, "hyp_words": len(h), "errors": len(h)}
    ops = Levenshtein.opcodes(r, h)
    sub = ins = dele = 0
    for tag, i1, i2, j1, j2 in ops:
        if tag == "replace":
            n = min(i2 - i1, j2 - j1)
            sub += n
            dele += (i2 - i1) - n
            ins += (j2 - j1) - n
        elif tag == "delete":
            dele += i2 - i1
        elif tag == "insert":
            ins += j2 - j1
    errors = sub + ins + dele
    return {
        "wer": errors / len(r),
        "sub": sub,
        "ins": ins,
        "del": dele,
        "errors": errors,
        "ref_words": len(r),
        "hyp_words": len(h),
    }


def der(ref_segments, hyp_segments, collar, exclude_overlap, step=0.01):
    """Frame-based DER with optional overlap exclusion and NIST collar."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    ref_spk = sorted({s["speaker"] for s in ref_segments})
    hyp_spk = sorted({s["speaker"] for s in hyp_segments})
    if not ref_spk:
        return {"der": 0.0, "note": "no reference segments"}
    dur = max(
        max(s["end"] for s in ref_segments), max((s["end"] for s in hyp_segments), default=0)
    )
    n = int(math.ceil(dur / step)) + 1

    ref = np.zeros((len(ref_spk), n), dtype=bool)
    ridx = {s: i for i, s in enumerate(ref_spk)}
    for s in ref_segments:
        ref[ridx[s["speaker"]], int(round(s["start"] / step)) : int(round(s["end"] / step))] = True

    hyp = np.full(n, -1, dtype=np.int16)
    hidx = {s: i for i, s in enumerate(hyp_spk)}
    for s in hyp_segments:
        hyp[int(round(s["start"] / step)) : int(round(s["end"] / step))] = hidx[s["speaker"]]

    total = ref.sum(axis=0).astype(np.int32)
    ref_any = total > 0
    overlap = total > 1
    hyp_any = hyp >= 0

    mask = ref_any.copy()
    if exclude_overlap:
        mask &= ~overlap

    unheard = np.zeros(n, dtype=bool)
    if collar > 0:
        for s in ref_segments:
            a = int(round((s["start"] - collar) / step))
            b = int(round((s["start"] + collar) / step))
            unheard[max(0, a) : min(n, b)] = True
            a = int(round((s["end"] - collar) / step))
            b = int(round((s["end"] + collar) / step))
            unheard[max(0, a) : min(n, b)] = True
        mask &= ~unheard

    # Map hyp clusters to ref speakers by overlapping speech duration.
    cost = np.zeros((len(ref_spk), len(hyp_spk)))
    for i, rs in enumerate(ref_spk):
        for j, hs in enumerate(hyp_spk):
            both = ref[i] & (hyp == j)
            cost[i, j] = both.sum()
    if len(ref_spk) and len(hyp_spk):
        ri, hi = linear_sum_assignment(-cost)
        mapping = {hyp_spk[j]: ref_spk[i] for i, j in zip(ri, hi)}
    else:
        mapping = {}
    mapped = np.full(n, -1, dtype=np.int16)
    for j, name in enumerate(hyp_spk):
        if name in mapping:
            mapped[hyp == j] = ridx[mapping[name]]
        else:
            mapped[hyp == j] = -2  # spoke, but not matched to any reference speaker

    # Confusion: a hyp label that is not active in the reference at that frame.
    hyp_ok = np.zeros(n, dtype=bool)
    for i in range(len(ref_spk)):
        hyp_ok |= (mapped == i) & ref[i]
    confusion = hyp_any & ref_any & ~hyp_ok
    miss = mask & ref_any & ~hyp_any
    fa = mask & ~ref_any & hyp_any
    conf = mask & confusion
    speaks = mask & ref_any

    total_ref = int(speaks.sum())
    return {
        "der": (int(miss.sum()) + int(fa.sum()) + int(conf.sum())) / max(1, total_ref),
        "miss": int(miss.sum()) / max(1, total_ref),
        "false_alarm": int(fa.sum()) / max(1, total_ref),
        "confusion": int(conf.sum()) / max(1, total_ref),
        "ref_speech_seconds": total_ref * step,
        "collar": collar,
        "exclude_overlap": exclude_overlap,
        "ref_speakers": len(ref_spk),
        "hyp_speakers": len(hyp_spk),
        "mapping": mapping,
    }


def single_speaker_mask(ref, step=0.01):
    """Intervals where exactly one reference speaker is active (10 ms grid)."""
    import numpy as np

    dur = max((s["end"] for s in ref["segments"]), default=0.0)
    n = int(math.ceil(dur / step)) + 1
    cover = np.zeros(n, dtype=np.int16)
    for s in ref["segments"]:
        cover[int(s["start"] / step) : int(s["end"] / step)] += 1
    return cover == 1, step


def score_no_overlap(ref, asr):
    """WER restricted to regions where only one reference speaker talks.

    Both the reference words and the hypothesis words (by timestamp midpoint)
    are filtered to those regions, so words lost to overlapping speech do not
    count as errors. This is the number comparable to clean-speech WER.
    """
    mask, step = single_speaker_mask(ref)
    if not len(mask):
        return None
    ref_words = [
        w["text"]
        for w in ref["words"]
        if mask[min(len(mask) - 1, int(w["start"] / step)) : max(1, int(w["end"] / step))].all()
    ]
    hyp_words = []
    for w in asr.get("words", []):
        mid = (w["start"] + w["end"]) / 2
        if mask[min(len(mask) - 1, max(0, int(mid / step)))]:
            hyp_words.append(w["text"])
    if not ref_words:
        return None
    res = wer(ref_words, hyp_words)
    res["ref_words"] = len(norm_tokens(ref_words))
    return res


def score_one(args):
    ref = json.load(open(args.ref))
    hyp = json.load(open(args.result))
    out = {"meeting": ref["meeting"], "result": os.path.basename(args.result)}

    asr = hyp.get("asr")
    if asr:
        raw = wer([w["text"] for w in ref["words"]], asr.get("text", "").split())
        nofill = wer(
            [w["text"] for w in ref["words"] if w["text"].lower() not in FILLERS],
            [
                t
                for t in asr.get("text", "").split()
                if t.lower().strip(".,?!") not in FILLERS
            ],
        )
        out["wer"] = raw["wer"]
        out["wer_without_fillers"] = nofill["wer"]
        noovl = score_no_overlap(ref, asr)
        if noovl:
            out["wer_no_overlap"] = noovl["wer"]
            out["wer_no_overlap_detail"] = noovl
        out["wer_detail"] = raw
        out["asr_seconds"] = asr.get("decode_seconds")
        out["asr_rtf"] = asr.get("rtf")
        out["load_ms_cold"] = asr.get("load_ms_cold")
        out["load_ms_warm"] = asr.get("load_ms_warm")
        out["has_word_timestamps"] = asr.get("has_word_timestamps")
        out["word_count"] = asr.get("word_count")

    diar = hyp.get("diarization")
    if diar:
        segs = [{"start": s["start"], "end": s["end"], "speaker": s["speaker"]} for s in diar["segments"]]
        refsegs = [{"start": s["start"], "end": s["end"], "speaker": s["speaker"]} for s in ref["segments"]]
        primary = der(refsegs, segs, args.collar, exclude_overlap=False)
        out["der"] = primary["der"]
        out["der_detail"] = primary
        out["der_no_overlap"] = der(refsegs, segs, args.collar, exclude_overlap=True)["der"]
        out["der_no_collar"] = der(refsegs, segs, 0.0, exclude_overlap=False)["der"]
        out["diar_seconds"] = diar.get("process_seconds")
        out["diar_rtf"] = diar.get("rtf")
        out["ref_speakers"] = len(ref["speakers"])
        out["hyp_speakers"] = diar.get("num_speakers")
        out["speaker_count_ok"] = len(ref["speakers"]) == diar.get("num_speakers")

    comb = hyp.get("combined")
    if comb:
        out["total_seconds"] = comb["total_seconds"]
        out["total_rtf"] = comb["rtf"]
    if "peak_rss_mb" in hyp:
        out["peak_rss_mb"] = hyp["peak_rss_mb"]
    out["threads"] = hyp.get("threads")
    out["audio_seconds"] = hyp.get("audio_seconds")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def table(paths):
    rows = [json.load(open(p)) for p in paths]
    cols = [
        ("meeting", "meeting"),
        ("audio_min", "audio_min"),
        ("threads", "thr"),
        ("wer", "WER"),
        ("der", "DER"),
        ("der_no_overlap", "DER no-ovl"),
        ("spk", "spk ref/hyp"),
        ("asr_rtf", "ASR RTF"),
        ("diar_rtf", "DIAR RTF"),
        ("total_rtf", "total RTF"),
        ("load_ms_cold", "load cold ms"),
        ("peak_rss_mb", "peak RSS MB"),
    ]
    print("| " + " | ".join(c[1] for c in cols) + " |")
    print("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        vals = []
        for key, _ in cols:
            if key == "audio_min":
                v = f"{r.get('audio_seconds', 0) / 60:.1f}"
            elif key == "spk":
                v = f"{r.get('ref_speakers', '?')}/{r.get('hyp_speakers', '?')}"
            elif key == "wer":
                v = f"{r.get('wer', 0) * 100:.1f}%"
            elif key in ("der", "der_no_overlap"):
                v = f"{r.get(key, 0) * 100:.1f}%"
            elif key.endswith("_rtf"):
                v = f"{r.get(key, 0):.3f}"
            elif key == "peak_rss_mb":
                v = f"{r.get(key, 0):.0f}"
            elif key == "load_ms_cold":
                v = f"{r.get(key, 0):.0f}"
            else:
                v = str(r.get(key, ""))
            vals.append(v)
        print("| " + " | ".join(vals) + " |")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ref")
    p.add_argument("--annotations", required=True)
    p.add_argument("--meeting", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=lambda a: build_ref(a.annotations, a.meeting, a.out))

    p = sub.add_parser("clean-sample")
    p.add_argument("--ref", required=True)
    p.add_argument("--wav", required=True)
    p.add_argument("--minutes", type=float, default=5.0)
    p.add_argument("--out-wav", required=True)
    p.add_argument("--out-ref", required=True)
    p.set_defaults(func=clean_sample)

    p = sub.add_parser("score")
    p.add_argument("--ref", required=True)
    p.add_argument("--result", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--collar", type=float, default=0.25)
    p.set_defaults(func=score_one)

    p = sub.add_parser("table")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=lambda a: table(a.paths))

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
