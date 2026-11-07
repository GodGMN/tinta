#!/usr/bin/env python3
"""RSS over ten minutes of recording with speech playing into the sink.

Usage: rss_probe.py ENGINE MODELS WAV OUT

Live drafts load Parakeet lazily, so a silent recording never measures the
recognizer. This plays the WAV in a loop into the default sink for the whole
run, which keeps the live worker decoding and loads the ASR model; the reported
peak therefore includes it. Samples `ps -o rss=` every 30 s.
"""

import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinta_probe import Engine, play, rss_mib

MINUTES = int(os.environ.get("TINTA_RSS_MINUTES", "10"))


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    exe, models, wav = sys.argv[1:4]
    out = sys.argv[4] if len(sys.argv) > 4 else "rss.txt"

    directory = tempfile.mkdtemp(prefix="tinta-rss-")
    engine = Engine(exe, models, live=True)
    engine.start(directory)
    stop_speech = threading.Event()

    def speak():
        while not stop_speech.is_set():
            play(wav, env=engine.env)
            stop_speech.wait(1.0)

    speech = threading.Thread(target=speak, daemon=True)
    speech.start()

    samples = []
    lines = []
    end = time.time() + MINUTES * 60
    index = 0
    while True:
        value = rss_mib(engine.process.pid)
        samples.append(value)
        lines.append("%4d s  %8.1f MiB" % (index * 30, value))
        print(lines[-1], flush=True)
        if time.time() >= end:
            break
        time.sleep(min(30.0, max(0.0, end - time.time())))
        index += 1

    stop_speech.set()
    engine.stop()
    engine.shutdown()
    live = [event for event in engine.events if event.get("event") == "live"]
    levels = [event for event in engine.events if event.get("event") == "levels"]
    tail = samples[2:]
    summary = "\n".join(
        [
            "%d-minute recording with speech, %s" % (MINUTES, time.strftime("%Y-%m-%d %H:%M")),
            "engine: %s" % exe,
            "models: %s (live drafts on)" % models,
            "speech: %s played in a loop into the default sink" % wav,
            "",
            *lines,
            "",
            "peak %.1f MiB, last %.1f MiB, tail spread %.1f MiB"
            % (max(samples), samples[-1], max(tail) - min(tail)),
            "live events: %d, levels: %d, warnings/errors: %d"
            % (len(live), len(levels), len(engine.errors())),
        ]
    )
    open(out, "w").write(summary + "\n")
    print(summary)


if __name__ == "__main__":
    main()
