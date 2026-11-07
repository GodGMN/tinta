#!/usr/bin/env python3
"""kill -9 mid-recording, then finalize from the chunks on disk.

Usage: recovery_probe.py ENGINE MODELS WAV OUT

Records with speech playing, kills the engine with SIGKILL, starts a new engine
and runs `finalize` on the same directory. Reports the chunks that survived and
the words the final pass recovered.
"""

import os
import signal
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinta_probe import Engine, KEY, play


def main():
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    exe, models, wav, out = sys.argv[1:5]
    directory = tempfile.mkdtemp(prefix="tinta-recover-")

    first = Engine(exe, models, live=False)
    first.start(directory)
    play(wav, env=first.env)
    time.sleep(2.0)
    os.kill(first.process.pid, signal.SIGKILL)
    first.process.wait()
    chunks = [
        name
        for root, _, files in os.walk(directory)
        for name in files
        if name.endswith(".tntc")
    ]

    second = Engine(exe, models, live=False)
    reply = second.command(
        2,
        "finalize",
        {"dir": directory, "audio_key": KEY},
        timeout=900,
    )
    second.shutdown()
    result = reply.get("result", {})
    remote = result.get("tracks", {}).get("remote", {})
    mic = result.get("tracks", {}).get("mic", {})
    text = "\n".join(
        [
            "recovery probe, %s" % time.strftime("%Y-%m-%d %H:%M"),
            "chunks after kill -9: %d" % len(chunks),
            "finalize ok: %s" % reply.get("ok"),
            "duration: %s" % result.get("duration"),
            "remote words: %d, mic words: %d"
            % (len(remote.get("words", [])), len(mic.get("words", []))),
            "remote text: %s"
            % " ".join(word["w"] for word in remote.get("words", []))[:300],
            "mic text: %s" % " ".join(word["w"] for word in mic.get("words", []))[:300],
            "directory: %s" % directory,
        ]
    )
    open(out, "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
