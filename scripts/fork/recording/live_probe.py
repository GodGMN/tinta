#!/usr/bin/env python3
"""Live drafts against a real engine with speech playing into the default sink.

Usage: live_probe.py ENGINE MODELS WAV [OUT]

Starts a recording with live drafts on, plays WAV into the default sink, stops
after 40 s, and reports the `live` events. With a machine whose default source
is a monitor, this used to transcribe the same sentence on `mic` and `remote`;
the probe fails when the two tracks carry the same text.
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinta_probe import Engine, play


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    exe, models, wav = sys.argv[1:4]
    out = sys.argv[4] if len(sys.argv) > 4 else None

    directory = tempfile.mkdtemp(prefix="tinta-live-")
    engine = Engine(exe, models, live=True)
    start_wall_ms = engine.start(directory)
    play(wav, env=engine.env)
    time.sleep(40)
    engine.stop()
    engine.shutdown()

    live = [event for event in engine.events if event.get("event") == "live"]
    mic_text = [event["text"] for event in live if event["track"] == "mic"]
    remote_text = [event["text"] for event in live if event["track"] == "remote"]
    duplicated = bool(mic_text) and bool(set(mic_text) & set(remote_text))

    lines = [
        "live probe, %s" % time.strftime("%Y-%m-%d %H:%M"),
        "engine: %s" % exe,
        "models: %s" % models,
        "default source: %s" % os.environ.get("TINTA_DEFAULT_SOURCE", "(unset)"),
        "start_wall_ms: %s" % start_wall_ms,
        "live events: %d (mic %d, remote %d)" % (len(live), len(mic_text), len(remote_text)),
        "events: %s" % {
            name: sum(1 for event in engine.events if event.get("event") == name)
            for name in ("levels", "live", "warning", "error", "source_state")
        },
    ]
    for event in live:
        lines.append("  %s %.2f-%.2f %s" % (event["track"], event["s"], event["e"], event["text"]))
    for event in engine.errors():
        lines.append("  %s: %s" % (event.get("event"), event.get("message")))
    levels = [event for event in engine.events if event.get("event") == "levels"]
    if levels:
        lines.append(
            "levels: max mic %.4f, max remote %.4f"
            % (max(event["mic"] for event in levels), max(event["remote"] for event in levels))
        )
    lines.append("mic duplicates remote: %s" % ("YES" if duplicated else "no"))
    text = "\n".join(lines) + "\n"
    if out:
        open(out, "w").write(text)
    print(text)
    if duplicated:
        raise SystemExit("the microphone track repeats the remote transcript")


if __name__ == "__main__":
    main()
