#!/usr/bin/env python3
"""Score tinta-bench JSON output against AMI manual annotations.

Subcommands:
  ref                 build a reference JSON from the AMI manual annotation XML
  slice               cut a wav to a deterministic short wav
  clean-sample        cut single-speaker AMI regions into one clean-speech wav
  librispeech-sample  build the 30-utterance LibriSpeech test-clean sample
  score               WER (legacy and Whisper normalizer) + pyannote DER
  tune                mean DER per clustering threshold / small-cluster filter on dev
  table               markdown summary from several score JSON files

DER is pyannote.metrics DiarizationErrorRate over the full file (a UEM from 0 to
the audio end). Primary is collar 0 s with overlapping speech counted; the
0.25 s collar number is reported next to it. WER is reported with the legacy
normalizer and with Whisper's EnglishTextNormalizer, both with del/ins/sub.

Requires: numpy, scipy, rapidfuzz, pyannote.metrics, whisper-normalizer
(see docs/fork/benchmark.md).
"""

import argparse
import glob
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
import xml.etree.ElementTree as ET

FILLERS = {"uh", "um", "er", "ah", "eh", "mm", "hmm", "uh-huh", "mhm", "oh"}
WHISPER_NORMALIZER = None
LEGACY_NORMALIZER = None
AMI_ACRONYM = None


def ami_acronyms(text):
    """Join AMI's spoken acronyms (`T_V_`) into one token (`TV`).

    The manual annotation writes every letter of a spelled-out acronym as its
    own `w` element with trailing underscores; Parakeet transcribes the same
    speech as `TV`. Without this mapping each `T_V_` costs a substitution plus
    two deletions, 24-54 times per meeting.
    """
    global AMI_ACRONYM
    if AMI_ACRONYM is None:
        AMI_ACRONYM = re.compile(r"\b(?:[A-Za-z]+_)+")
    return AMI_ACRONYM.sub(lambda m: m.group(0).replace("_", ""), text)


def legacy_tokens(words):
    """Lowercase, strip punctuation and keep one token per word."""
    global LEGACY_NORMALIZER
    if LEGACY_NORMALIZER is None:
        LEGACY_NORMALIZER = re.compile(r"[^a-z0-9'\- ]+")
    text = words if isinstance(words, str) else " ".join(words)
    text = ami_acronyms(text).lower().replace("\u2019", "'")
    text = LEGACY_NORMALIZER.sub(" ", text)
    return [p for p in text.split() if p]


def whisper_tokens(words):
    """Whisper's English text normalizer: punctuation, numbers, spellings, fillers."""
    global WHISPER_NORMALIZER
    if WHISPER_NORMALIZER is None:
        from whisper_normalizer.english import EnglishTextNormalizer

        WHISPER_NORMALIZER = EnglishTextNormalizer()
    text = words if isinstance(words, str) else " ".join(words)
    return WHISPER_NORMALIZER(ami_acronyms(text)).split()


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
        "duration": max((w["end"] for w in words), default=0.0),
        "speakers": sorted({w["speaker"] for w in words}),
    }
    with open(out_path, "w") as f:
        json.dump(ref, f, indent=1)
    print(
        f"ref {meeting}: {len(words)} words, {len(segments)} segments, "
        f"{len(ref['speakers'])} speakers, {ref['duration'] / 60:.1f} min -> {out_path}"
    )


def cut_slice(args):
    """Cut [start, start+duration) from a wav into a new 16 kHz mono wav."""
    import wave

    with wave.open(args.wav, "rb") as src:
        assert src.getnchannels() == 1 and src.getsampwidth() == 2
        rate = src.getframerate()
        src.setpos(int(args.start * rate))
        data = src.readframes(int(args.duration * rate))
    with wave.open(args.out_wav, "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(rate)
        dst.writeframes(data)
    print(f"slice: {args.start:.1f}s + {args.duration:.1f}s -> {args.out_wav}")


def slice_ref(args):
    """Cut a reference JSON to the same interval as a wav slice."""
    ref = json.load(open(args.ref))
    a, b = args.start, args.start + args.duration
    words = [
        {**w, "start": w["start"] - a, "end": w["end"] - a}
        for w in ref["words"]
        if w["start"] >= a and w["end"] <= b
    ]
    segments = []
    for s in ref["segments"]:
        start, end = max(s["start"], a), min(s["end"], b)
        if end > start:
            segments.append({**s, "start": start - a, "end": end - a})
    out = {
        "meeting": ref["meeting"] + f"@{a:.0f}s",
        "words": words,
        "segments": segments,
        "duration": min(b, ref["duration"]) - a,
        "speakers": ref["speakers"],
    }
    json.dump(out, open(args.out, "w"), indent=1)
    print(f"slice ref: {len(words)} words, {len(segments)} segments -> {args.out}")


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
        # Where this region starts in the concatenated output, before the
        # silence that follows it. Word timings must be shifted by this much:
        # the source regions are not contiguous, so carrying the ES2002a
        # timestamps over would misplace almost every word.
        offset = sum(len(piece) for piece in pieces) / 16000.0
        pieces.append(audio[int(a * 16000) : int(b * 16000)])
        used += b - a
        for w in ref["words"]:
            if w["start"] >= a and w["end"] <= b:
                item = dict(w)
                item["start"] = round(offset + (w["start"] - a), 3)
                item["end"] = round(offset + (w["end"] - a), 3)
                words.append(item)
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


def librispeech_sample(args):
    """Build the same 30 test-clean utterances from the HF datasets server.

    Rows 0..29 of openslr/librispeech_asr, config clean, split test, in order,
    with 0.2 s of silence between utterances. Deterministic, so the SHA-256
    pinned in fetch-data.sh holds.
    """
    import wave

    endpoint = (
        "https://datasets-server.huggingface.co/rows?"
        f"dataset=openslr%2Flibrispeech_asr&config=clean&split=test&offset=0&length={args.count}"
    )
    with urllib.request.urlopen(endpoint, timeout=120) as r:
        payload = json.load(r)

    pieces = []
    refs = []
    sample_rate = 16000
    for row in payload["rows"]:
        audio = row["row"]["audio"][0]["src"]
        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            urllib.request.urlretrieve(audio, tmp_path)
            raw = subprocess.run(
                ["ffmpeg", "-loglevel", "error", "-i", tmp_path, "-f", "s16le",
                 "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(sample_rate), "-"],
                check=True, capture_output=True,
            ).stdout
        finally:
            os.unlink(tmp_path)
        pieces.append(raw)
        pieces.append(b"\x00\x00" * int(0.2 * sample_rate))
        refs.append({"text": row["row"]["text"]})

    with wave.open(args.out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"".join(pieces))
    json.dump({"meeting": "librispeech_test-clean_30", "words": refs}, open(args.out_ref, "w"), indent=1)
    print(f"librispeech sample: {len(refs)} utterances -> {args.out_wav}")


def wer_tokens(ref_tokens, hyp_tokens):
    from rapidfuzz.distance import Levenshtein

    if not ref_tokens:
        return {"wer": 0.0, "ref_words": 0, "hyp_words": len(hyp_tokens), "errors": len(hyp_tokens)}
    ops = Levenshtein.opcodes(ref_tokens, hyp_tokens)
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
        "wer": errors / len(ref_tokens),
        "sub": sub,
        "ins": ins,
        "del": dele,
        "errors": errors,
        "ref_words": len(ref_tokens),
        "hyp_words": len(hyp_tokens),
    }


def wer_pair(ref_words, hyp_words):
    """WER with the legacy normalizer and with Whisper's English normalizer."""
    return {
        "legacy": wer_tokens(legacy_tokens(ref_words), legacy_tokens(hyp_words)),
        "whisper": wer_tokens(whisper_tokens(ref_words), whisper_tokens(hyp_words)),
    }


def single_speaker_mask(ref, step=0.01):
    """Intervals where exactly one reference speaker is active (10 ms grid)."""
    import numpy as np

    if not ref.get("segments"):
        return np.zeros(0, dtype=bool), step
    dur = max((s["end"] for s in ref["segments"]), default=0.0)
    n = int(math.ceil(dur / step)) + 1
    cover = np.zeros(n, dtype=np.int16)
    for s in ref["segments"]:
        cover[int(s["start"] / step) : int(s["end"] / step)] += 1
    return cover == 1, step


def score_no_overlap(ref, asr):
    """WER restricted to regions where only one reference speaker talks."""
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
    return wer_pair(ref_words, hyp_words)


def hyp_gap_stats(ref, asr, min_gap=5.0):
    """Long stretches with no hypothesis word while the reference has speech.

    This is the diagnostic for the old 240 s-chunk failure: whole stretches
    inside a chunk produced no words at all.
    """
    words = sorted(asr.get("words", []), key=lambda w: w["start"])
    if len(words) < 2:
        return {"gaps_seconds": 0.0, "gaps": 0, "ref_words_in_gaps": 0}
    gaps = 0
    seconds = 0.0
    ref_words = 0
    prev = words[0]["end"]
    for w in words[1:]:
        start = w["start"]
        if start - prev >= min_gap:
            gaps += 1
            seconds += start - prev
            for r in ref["words"]:
                mid = (r["start"] + r["end"]) / 2
                if prev <= mid < start:
                    ref_words += 1
        prev = max(prev, w["end"])
    return {"gaps_seconds": seconds, "gaps": gaps, "ref_words_in_gaps": ref_words}


def annotation_from_segments(segments):
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for s in segments:
        start, end = float(s["start"]), float(s["end"])
        if end <= start:
            continue
        label = str(s["speaker"])
        ann[Segment(start, end), label] = label
    return ann


def der_pyannote(ref_segments, hyp_segments, uem_end, collar, skip_overlap):
    """pyannote.metrics DiarizationErrorRate over a UEM covering the whole file."""
    from pyannote.core import Segment
    from pyannote.metrics.diarization import DiarizationErrorRate

    ref = annotation_from_segments(ref_segments)
    hyp = annotation_from_segments(hyp_segments)
    metric = DiarizationErrorRate(collar=collar, skip_overlap=skip_overlap)
    detail = metric(ref, hyp, uem=Segment(0.0, float(uem_end)), detailed=True)
    total = float(detail["total"])
    miss = float(detail.get("missed detection", detail.get("miss", 0.0)))
    fa = float(detail.get("false alarm", 0.0))
    conf = float(detail.get("confusion", 0.0))
    return {
        "der": float(detail["diarization error rate"]),
        "miss": miss / total if total else 0.0,
        "false_alarm": fa / total if total else 0.0,
        "confusion": conf / total if total else 0.0,
        "miss_seconds": miss,
        "false_alarm_seconds": fa,
        "confusion_seconds": conf,
        "ref_speech_seconds": total,
        "collar": collar,
        "skip_overlap": skip_overlap,
    }


def filter_small_clusters(segments, fraction):
    """Reassign clusters under `fraction` of speech time to the nearest large one.

    Segments of a small cluster go to the large cluster whose nearest segment
    is closest in time. No fixed speaker count is ever forced: if every cluster
    is large, or none is, the segments come back unchanged.
    """
    if fraction <= 0 or not segments:
        return list(segments), []
    speech = {}
    for s in segments:
        speech[s["speaker"]] = speech.get(s["speaker"], 0.0) + max(0.0, s["end"] - s["start"])
    total = sum(speech.values())
    if total <= 0:
        return list(segments), []
    large = sorted(k for k, t in speech.items() if t >= fraction * total)
    small = sorted(k for k in speech if k not in set(large))
    if not large or not small:
        return list(segments), []

    by_large = {k: [] for k in large}
    for s in segments:
        if s["speaker"] in by_large:
            by_large[s["speaker"]].append(s)

    def gap(seg, candidate):
        best = math.inf
        for other in by_large[candidate]:
            if seg["end"] < other["start"]:
                d = other["start"] - seg["end"]
            elif other["end"] < seg["start"]:
                d = seg["start"] - other["end"]
            else:
                d = 0.0
            best = min(best, d)
        return best

    out = []
    for s in segments:
        item = dict(s)
        if item["speaker"] in small:
            item["speaker"] = min(large, key=lambda k: (gap(item, k), str(k)))
        out.append(item)
    return out, small


def speaker_totals(segments):
    totals = {}
    for s in segments:
        totals[s["speaker"]] = totals.get(s["speaker"], 0.0) + max(0.0, s["end"] - s["start"])
    return {str(k): round(v, 3) for k, v in sorted(totals.items(), key=lambda kv: -kv[1])}


def score_one(args):
    ref = json.load(open(args.ref))
    hyp = json.load(open(args.result))
    out = {"meeting": ref["meeting"], "result": os.path.basename(args.result)}

    asr = hyp.get("asr")
    if asr:
        ref_text = [w["text"] for w in ref["words"]]
        hyp_text = asr.get("text", "")
        both = wer_pair(ref_text, hyp_text)
        out["wer"] = both["legacy"]["wer"]
        out["wer_detail"] = both["legacy"]
        out["wer_whisper"] = both["whisper"]["wer"]
        out["wer_whisper_detail"] = both["whisper"]
        nofill = wer_pair(
            [w["text"] for w in ref["words"] if w["text"].lower() not in FILLERS],
            [t for t in hyp_text.split() if t.lower().strip(".,?!") not in FILLERS],
        )
        out["wer_without_fillers"] = nofill["legacy"]["wer"]
        noovl = score_no_overlap(ref, asr)
        if noovl:
            out["wer_no_overlap"] = noovl["legacy"]["wer"]
            out["wer_no_overlap_whisper"] = noovl["whisper"]["wer"]
            out["wer_no_overlap_detail"] = noovl
        out["hyp_gap_stats"] = hyp_gap_stats(ref, asr)
        out["asr_seconds"] = asr.get("decode_seconds")
        out["asr_rtf"] = asr.get("rtf")
        out["asr_cpu_seconds"] = asr.get("cpu_seconds")
        out["vad"] = asr.get("vad")
        out["load_ms_cold"] = asr.get("load_ms_cold")
        out["load_ms_warm"] = asr.get("load_ms_warm")
        out["has_word_timestamps"] = asr.get("has_word_timestamps")
        out["word_count"] = asr.get("word_count")

    diar = hyp.get("diarization")
    if diar:
        segs = [
            {"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
            for s in diar["segments"]
        ]
        refsegs = [
            {"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
            for s in ref["segments"]
        ]
        uem_end = float(hyp.get("audio_seconds") or ref.get("duration") or 0.0)
        filtered, dropped = filter_small_clusters(segs, args.small_cluster_fraction)

        def der_block(hyp_segs):
            primary = der_pyannote(refsegs, hyp_segs, uem_end, collar=0.0, skip_overlap=False)
            collar = der_pyannote(refsegs, hyp_segs, uem_end, collar=0.25, skip_overlap=False)
            no_ovl = der_pyannote(refsegs, hyp_segs, uem_end, collar=0.0, skip_overlap=True)
            return primary, collar, no_ovl

        primary, collar, no_ovl = der_block(segs)
        f_primary, f_collar, f_no_ovl = der_block(filtered)
        out["der"] = primary["der"]
        out["der_detail"] = primary
        out["der_collar025"] = collar["der"]
        out["der_collar025_detail"] = collar
        out["der_no_overlap"] = no_ovl["der"]
        out["der_filtered"] = f_primary["der"]
        out["der_filtered_detail"] = f_primary
        out["der_filtered_collar025"] = f_collar["der"]
        out["der_filtered_no_overlap"] = f_no_ovl["der"]
        out["small_cluster_fraction"] = args.small_cluster_fraction
        out["small_clusters"] = [str(x) for x in dropped]
        out["speaker_totals"] = speaker_totals(segs)
        out["diar_seconds"] = diar.get("process_seconds")
        out["diar_rtf"] = diar.get("rtf")
        out["diar_cpu_seconds"] = diar.get("cpu_seconds")
        out["threshold"] = diar.get("threshold")
        out["ref_speakers"] = len(ref["speakers"])
        out["hyp_speakers"] = diar.get("num_speakers")
        out["hyp_speakers_filtered"] = len({s["speaker"] for s in filtered})
        out["speaker_count_ok"] = len(ref["speakers"]) == diar.get("num_speakers")
        out["speaker_count_ok_filtered"] = (
            len(ref["speakers"]) == out["hyp_speakers_filtered"]
        )

    comb = hyp.get("combined")
    if comb:
        out["combined_asr_seconds"] = comb.get("asr_seconds")
        out["combined_diar_seconds"] = comb.get("diarization_seconds")
        out["critical_path_seconds"] = comb.get("critical_path_seconds")
        out["total_seconds"] = comb.get("total_seconds")
        out["total_cpu_seconds"] = hyp.get("process_cpu_seconds")
        out["total_rtf"] = comb.get("rtf")
    if "peak_rss_mb" in hyp:
        out["peak_rss_mb"] = hyp["peak_rss_mb"]
    out["threads"] = hyp.get("threads")
    out["diar_threads"] = hyp.get("diar_threads")
    out["spin"] = hyp.get("spin")
    out["audio_seconds"] = hyp.get("audio_seconds")
    out["loadavg_start"] = hyp.get("loadavg_start")
    out["loadavg_end"] = hyp.get("loadavg_end")
    out["machine"] = hyp.get("machine")

    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def tune(args):
    """Aggregate dev DER over thresholds and small-cluster filters.

    Result files are expected to be named diar-<meeting>-*.json; the AMI
    reference is <ref-dir>/<meeting>.ref.json.
    """
    fractions = [float(x) for x in args.fractions.split(",")]
    by_key = {}
    for result in sorted(glob.glob(args.results_glob)):
        base = os.path.basename(result)
        meeting = base.split("-")[1]
        # 6 min samples have their own time-shifted reference; comparing them
        # with the full-meeting reference would mix two timelines.
        ref_name = meeting + (".sample6m.ref.json" if "-sample" in base else ".ref.json")
        ref = json.load(open(os.path.join(args.ref_dir, ref_name)))
        hyp = json.load(open(result))
        diar = hyp.get("diarization")
        if not diar:
            continue
        threshold = diar.get("threshold")
        segs = diar["segments"]
        refsegs = ref["segments"]
        uem = float(hyp.get("audio_seconds") or ref.get("duration") or 0.0)
        for fraction in fractions:
            filtered, dropped = filter_small_clusters(segs, fraction)
            detail = der_pyannote(refsegs, filtered, uem, collar=0.0, skip_overlap=False)
            key = (threshold, fraction)
            by_key.setdefault(key, []).append(
                {
                    "meeting": meeting,
                    "der": detail["der"],
                    "speakers": len({s["speaker"] for s in filtered}),
                    "ref_speakers": len(ref["speakers"]),
                    "clusters": len({s["speaker"] for s in segs}),
                }
            )
    print("| threshold | cluster filter | mean DER | meetings | speaker count |")
    print("|---|---|---|---|---|")
    for (threshold, fraction) in sorted(by_key, key=lambda k: (k[0], k[1])):
        rows = by_key[(threshold, fraction)]
        mean = sum(r["der"] for r in rows) / len(rows)
        counts = " ".join(f"{r['speakers']}/{r['ref_speakers']}" for r in rows)
        print(
            f"| {threshold} | {fraction * 100:.1f}% | {mean * 100:.1f}% | "
            f"{len(rows)} | {counts} |"
        )
    candidates = [
        (threshold, fraction, sum(r["der"] for r in rows) / len(rows))
        for (threshold, fraction), rows in by_key.items()
        if abs(fraction - args.choose_fraction) < 1e-9
    ]
    if candidates:
        threshold, fraction, mean = min(candidates, key=lambda x: (x[2], x[0]))
        print(f"BEST {threshold:g} {fraction:g} {mean:.6f}")


def fmt_pct(v):
    return "—" if v is None else f"{v * 100:.1f}%"


def table(paths):
    rows = [json.load(open(p)) for p in paths]
    cols = [
        ("meeting", "meeting"),
        ("audio_min", "audio min"),
        ("threads", "thr"),
        ("wer", "WER legacy"),
        ("wer_whisper", "WER Whisper"),
        ("del", "del/ins/sub"),
        ("threshold", "cluster thr"),
        ("der", "DER"),
        ("der_collar025", "DER c25"),
        ("der_filtered", "DER filt"),
        ("spk", "spk raw/filt/ref"),
        ("asr_rtf", "ASR RTF"),
        ("diar_rtf", "DIAR RTF"),
        ("total_rtf", "total RTF"),
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
                v = (
                    f"{r.get('hyp_speakers', '?')}/"
                    f"{r.get('hyp_speakers_filtered', '?')}/"
                    f"{r.get('ref_speakers', '?')}"
                )
            elif key == "del":
                d = r.get("wer_whisper_detail") or {}
                v = f"{d.get('del', '?')}/{d.get('ins', '?')}/{d.get('sub', '?')}"
            elif key == "threshold" and r.get(key) is not None:
                v = f"{r[key]:.2f}"
            elif key in ("wer", "wer_whisper", "der", "der_collar025", "der_filtered"):
                v = fmt_pct(r.get(key))
            elif key.endswith("_rtf"):
                v = f"{r.get(key, 0):.3f}"
            elif key == "peak_rss_mb":
                v = f"{r.get(key, 0):.0f}"
            else:
                v = str(r.get(key, ""))
            vals.append(v)
        print("| " + " | ".join(vals) + " |")


def timing(paths):
    """Timing/ablation table straight from result JSON (no scoring)."""
    print("| run | audio min | threads | spin | wall s | CPU s | RTF | max piece s | peak RSS MB |")
    print("|---|---|---|---|---|---|---|---|---|")
    for p in paths:
        r = json.load(open(p))
        if r.get("command") == "combined":
            wall = r["combined"]["asr_seconds"]
            cpu = r.get("process_cpu_seconds", 0.0)
            rtf = r["combined"].get("rtf", 0.0)
            note = "combined ASR wall"
        else:
            stage = r.get("asr") or r.get("diarization")
            wall = stage.get("decode_seconds", stage.get("process_seconds", 0.0))
            cpu = r.get("process_cpu_seconds", 0.0)
            rtf = stage.get("rtf", 0.0)
            note = os.path.basename(p)
        max_piece = (r.get("asr") or {}).get("max_piece_seconds", 0.0)
        print(
            f"| {note} | {r.get('audio_seconds', 0) / 60:.1f} | {r.get('threads')} | "
            f"{r.get('spin')} | {wall:.1f} | {cpu:.1f} | {rtf:.3f} | "
            f"{max_piece:.1f} | {r.get('peak_rss_mb', 0):.0f} |"
        )


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("ref")
    p.add_argument("--annotations", required=True)
    p.add_argument("--meeting", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=lambda a: build_ref(a.annotations, a.meeting, a.out))

    p = sub.add_parser("slice")
    p.add_argument("--wav", required=True)
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--out-wav", required=True)
    p.set_defaults(func=cut_slice)

    p = sub.add_parser("slice-ref")
    p.add_argument("--ref", required=True)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--duration", type=float, required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=slice_ref)

    p = sub.add_parser("clean-sample")
    p.add_argument("--ref", required=True)
    p.add_argument("--wav", required=True)
    p.add_argument("--minutes", type=float, default=5.0)
    p.add_argument("--out-wav", required=True)
    p.add_argument("--out-ref", required=True)
    p.set_defaults(func=clean_sample)

    p = sub.add_parser("librispeech-sample")
    p.add_argument("--count", type=int, default=30)
    p.add_argument("--out-wav", required=True)
    p.add_argument("--out-ref", required=True)
    p.set_defaults(func=librispeech_sample)

    p = sub.add_parser("score")
    p.add_argument("--ref", required=True)
    p.add_argument("--result", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--small-cluster-fraction", type=float, default=0.02)
    p.set_defaults(func=score_one)

    p = sub.add_parser("tune")
    p.add_argument("--ref-dir", required=True)
    p.add_argument("--results-glob", required=True)
    p.add_argument("--fractions", default="0,0.005,0.01,0.02,0.03")
    p.add_argument("--choose-fraction", type=float, default=0.02)
    p.set_defaults(func=tune)

    p = sub.add_parser("table")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=lambda a: table(a.paths))

    p = sub.add_parser("timing")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=lambda a: timing(a.paths))

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
