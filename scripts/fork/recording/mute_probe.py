#!/usr/bin/env python3
"""Microphone mute at an exact wall-clock time, through the real engine.

Usage: mute_probe.py ENGINE WAV OUT

Creates two null sinks and a remapped microphone source, records 5 s, plays a
tone into the microphone sink from 0.5 s, mutes at 1.5 s and unmutes at 3 s. The
microphone track must be silent between the mute and the unmute even though the
`mic_muted` commands arrive after the audio was captured (the 0.8 s hold).
"""

import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tinta_probe import Engine, VirtualDevices, play, rms, sample_track


def main():
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    exe, wav, out = sys.argv[1:4]
    lines = []
    with VirtualDevices() as devices:
        directory = tempfile.mkdtemp(prefix="tinta-mute-")
        engine = Engine(exe, live=False)
        engine.start(directory)
        t0 = time.time()
        play(wav, device="tinta_probe_b", env=engine.env, block=False)
        time.sleep(1.5 - (time.time() - t0))
        engine.command(
            10,
            "mic_muted",
            {"muted": True, "t": int(time.time() * 1000)},
            timeout=10,
        )
        time.sleep(3.0 - (time.time() - t0))
        engine.command(
            11,
            "mic_muted",
            {"muted": False, "t": int(time.time() * 1000)},
            timeout=10,
        )
        time.sleep(5.0 - (time.time() - t0))
        engine.stop()
        mic = sample_track(engine, directory, "mic", 4.5)
        engine.shutdown()
        lines.append("mute probe, %s" % time.strftime("%Y-%m-%d %H:%M"))
        lines.append(
            "default source while recording: %s"
            % devices.run(["get-default-source"]).strip()
        )
        windows = [(0.5, 1.4), (1.7, 2.9), (3.2, 4.4)]
        for start, end in windows:
            lines.append("  mic rms %.1f-%.1f s: %.4f" % (start, end, rms(mic, start, end)))
        before = rms(mic, 0.5, 1.4)
        during = rms(mic, 1.7, 2.9)
        after = rms(mic, 3.2, 4.4)
        lines.append("mute silenced the microphone: %s" % ("yes" if during < 0.01 else "NO"))
        lines.append("tone before and after the mute: %s" % ("yes" if before > 0.05 and after > 0.05 else "NO"))
        lines.append("levels while muted: %s" % [
            (round(event["mic"], 4), event["mic_muted"])
            for event in engine.events
            if event.get("event") == "levels" and event.get("mic_muted")
        ][:4])
        ok = during < 0.01 and before > 0.05 and after > 0.05
    text = "\n".join(lines) + "\n"
    open(out, "w").write(text)
    print(text)
    if not ok:
        raise SystemExit("the mute check failed")


if __name__ == "__main__":
    main()
