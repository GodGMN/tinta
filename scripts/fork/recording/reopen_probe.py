#!/usr/bin/env python3
"""The remote capture follows a changed default sink mid-recording.

Usage: reopen_probe.py ENGINE OUT

Creates two null sinks, records 6.5 s, plays a tone into sink A, then switches
the default sink to B without any read error and plays a tone into B. The remote
track must carry both tones at their true times; a stream that stayed on A would
only have the first.
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinta_probe import Engine, VirtualDevices, play, rms, sample_track, wav_tone


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    exe, out = sys.argv[1:3]
    lines = ["reopen probe, %s" % time.strftime("%Y-%m-%d %H:%M")]
    with VirtualDevices() as devices:
        directory = tempfile.mkdtemp(prefix="tinta-reopen-")
        tone_a = os.path.join(directory, "tone-a.wav")
        tone_b = os.path.join(directory, "tone-b.wav")
        wav_tone(tone_a, 1.5, 440.0)
        wav_tone(tone_b, 1.5, 660.0)
        log_path = os.path.join(directory, "engine.log")
        log = open(log_path, "w")
        engine = Engine(exe, live=False, stderr=log)
        engine.start(directory)
        t0 = time.time()
        play(tone_a, device="tinta_probe_a", env=engine.env, block=False)
        time.sleep(2.5 - (time.time() - t0))
        devices.run(["set-default-sink", "tinta_probe_b"])
        time.sleep(4.0 - (time.time() - t0))
        play(tone_b, device="tinta_probe_b", env=engine.env, block=False)
        time.sleep(6.5 - (time.time() - t0))
        engine.stop()
        remote = sample_track(engine, directory, "remote", 6.0)
        engine.shutdown()
        log.close()
        windows = [(0.5, 2.2), (2.8, 3.8), (4.1, 5.6)]
        for start, end in windows:
            lines.append("  remote rms %.1f-%.1f s: %.4f" % (start, end, rms(remote, start, end)))
        a_tone = rms(remote, 0.5, 2.2) > 0.05
        gap = rms(remote, 2.8, 3.8) < 0.01
        b_tone = rms(remote, 4.1, 5.6) > 0.05
        lines.append("first sink tone: %s" % ("yes" if a_tone else "NO"))
        lines.append("silence during the switch: %s" % ("yes" if gap else "NO"))
        lines.append("new sink tone after the switch: %s" % ("yes" if b_tone else "NO"))
        lines.append("default sink at the end: %s" % devices.run(["get-default-sink"]).strip())
        switch_logs = [
            event
            for event in engine.events
            if event.get("event") in ("warning", "error")
        ]
        lines.append("warnings: %s" % [
            event.get("message") for event in switch_logs
        ])
        switches = [
            line.strip()
            for line in open(log_path)
            if "follows the new default device" in line
        ]
        lines.append("reopen log lines: %d %s" % (len(switches), switches))
        ok = a_tone and gap and b_tone
    text = "\n".join(lines) + "\n"
    open(out, "w").write(text)
    print(text)
    if not ok:
        raise SystemExit("the reopen check failed")


if __name__ == "__main__":
    main()
