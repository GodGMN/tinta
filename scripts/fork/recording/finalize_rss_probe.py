#!/usr/bin/env python3
"""RSS while a final pass of one meeting runs during a recording of another.

Usage: finalize_rss_probe.py ENGINE MODELS WAV OUT

Live drafts load Parakeet lazily, so a silent recording never measures the
recognizer. This imports WAV as meeting A, starts recording meeting B with the
WAV played in a loop into the default sink, waits for the live pass to decode,
then finalizes A while B is still recording and speech is still playing. Both
passes share one recognizer instance (the weak cache), so the reported peak is
the one process holding one copy of the models, the diarization models and the
decoded audio of A.

Samples `ps -o rss=` every 2 s. `Engine.read()` only drains the engine's stdout
while a command waits, so the `live` count below is informational, not a
per-second timeline.
"""

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinta_probe import KEY, Engine, play, rss_mib

SAMPLE_SECONDS = 2.0
# How long meeting B records while the final pass of A runs.
RECORD_SECONDS = 240
# Wait for the live pass to load the recognizer before starting the finalize.
LIVE_WAIT = 90


def main():
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    exe, models, wav, out = sys.argv[1:5]

    meeting_a = tempfile.mkdtemp(prefix="tinta-rss-a-")
    meeting_b = tempfile.mkdtemp(prefix="tinta-rss-b-")
    engine = Engine(exe, models, live=True)
    lines = ["concurrent finalize RSS, %s" % time.strftime("%Y-%m-%d %H:%M")]

    # Meeting A: an import is much faster than recording the same audio.
    reply = engine.command(
        10,
        "import_audio",
        {"dir": meeting_a, "audio_key": KEY, "path": wav},
        timeout=600,
    )
    if not reply.get("ok"):
        raise SystemExit("import failed: %s" % reply.get("error"))
    lines.append("meeting A: imported %s" % os.path.basename(wav))

    # Meeting B: record with speech playing, so the live pass keeps decoding.
    engine.start(meeting_b)
    stop_speech = threading.Event()

    def speak():
        while not stop_speech.is_set():
            play(wav, env=engine.env)
            stop_speech.wait(1.0)

    speech = threading.Thread(target=speak, daemon=True)
    speech.start()

    live_before = 0
    deadline = time.time() + LIVE_WAIT
    while time.time() < deadline:
        live_before = len([e for e in engine.events if e.get("event") == "live"])
        if live_before > 0:
            break
        time.sleep(1.0)
    baseline = rss_mib(engine.process.pid)
    lines.append(
        "live events before the finalize: %d, RSS %.1f MiB" % (live_before, baseline)
    )

    # Finalize A while B records and plays speech. The finalize runs on its own
    # thread because its reply only arrives when the whole pass ends; that
    # thread is the only reader of the engine's stdout.
    result = {}

    def run_finalize():
        result["reply"] = engine.command(
            11,
            "finalize",
            {"dir": meeting_a, "audio_key": KEY},
            timeout=RECORD_SECONDS + 120,
        )

    finalize_start = time.time()
    finalize = threading.Thread(target=run_finalize, daemon=True)
    finalize.start()
    samples = []
    while time.time() - finalize_start < RECORD_SECONDS:
        samples.append(rss_mib(engine.process.pid))
        if "reply" in result:
            break
        time.sleep(SAMPLE_SECONDS)
    finalize_seconds = time.time() - finalize_start
    finalize.join(timeout=180)
    finalize_reply = result.get("reply")
    live_total = len([e for e in engine.events if e.get("event") == "live"])
    peak = max(samples) if samples else rss_mib(engine.process.pid)
    end_rss = rss_mib(engine.process.pid)

    stop_speech.set()
    speech.join(timeout=10)
    engine.stop()
    engine.shutdown()

    lines.append(
        "finalize of A: %s in %.1f s"
        % (
            "ok" if finalize_reply and finalize_reply.get("ok") else "no reply",
            finalize_seconds,
        )
    )
    lines.append("RSS samples during the finalize: %d" % len(samples))
    lines.append("peak during the finalize: %.1f MiB" % peak)
    lines.append("last during the finalize: %.1f MiB" % (samples[-1] if samples else 0.0))
    lines.append("live events read while the probe ran: %d" % live_total)
    lines.append("RSS at the end: %.1f MiB" % end_rss)
    lines.append(
        "warnings/errors: %d"
        % len([e for e in engine.events if e.get("event") in ("warning", "error")])
    )
    text = "\n".join(lines) + "\n"
    open(out, "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
