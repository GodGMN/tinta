"""Shared helpers for the live-recording probes.

Every probe runs the real engine over its JSON-lines protocol. The runtime
dependencies are a PulseAudio or PipeWire pulse server plus `pactl` and
`paplay` (Debian: `pulseaudio-utils`). When the binaries are not on `PATH`,
set `TINTA_PACTL` / `TINTA_PAPLAY`, as the end-to-end test does.
"""

import base64
import json
import os
import subprocess
import tempfile
import time

KEY = base64.b64encode(bytes([7] * 32)).decode()


def tool(name):
    return os.environ.get("TINTA_" + name.upper(), name)


def audio_env(models=None, live=True):
    env = dict(os.environ)
    if models:
        env["TINTA_ENGINE_MODELS_DIR"] = models
    if live:
        env.pop("TINTA_ENGINE_LIVE", None)
    else:
        env["TINTA_ENGINE_LIVE"] = "0"
    return env


def play(wav, device="@DEFAULT_SINK@", env=None, block=True):
    command = [tool("paplay"), "--device=" + device, wav]
    if block:
        subprocess.run(command, env=env or os.environ, check=True)
        return None
    return subprocess.Popen(command, env=env or os.environ)


class Engine:
    """The engine under test, with the events it emitted."""

    def __init__(self, exe, models=None, live=True, stderr=None):
        self.env = audio_env(models, live)
        self.stderr = stderr
        stderr = stderr if stderr is not None else subprocess.PIPE
        self.process = subprocess.Popen(
            [exe],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
            bufsize=1,
            env=self.env,
        )
        self.events = []
        self.read(lambda value: value.get("event") == "ready", 30)

    def send(self, value):
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()

    def read(self, predicate, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("the engine stopped: %r" % (self.stderr_text(),))
            value = json.loads(line)
            if "id" not in value:
                self.events.append(value)
            if predicate(value):
                return value
        raise TimeoutError("no matching line in %s s" % timeout)

    def start(self, directory):
        reply = self.send_reply(
            1,
            "start",
            {"dir": directory, "audio_key": KEY, "source": "all"},
            60,
        )
        if not reply.get("ok"):
            raise RuntimeError("start failed: %s" % reply.get("error"))
        return reply["result"]["start_wall_ms"]

    def command(self, request_id, command, params=None, timeout=60):
        return self.send_reply(request_id, command, params or {}, timeout)

    def send_reply(self, request_id, command, params, timeout):
        self.send({"id": request_id, "cmd": command, "params": params})
        return self.read(lambda value: value.get("id") == request_id, timeout)

    def stop(self):
        return self.command(2, "stop", timeout=120)

    def shutdown(self):
        if self.process.poll() is None:
            try:
                self.send({"id": 0, "cmd": "shutdown"})
            except BrokenPipeError:
                pass
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        while True:
            line = self.process.stdout.readline()
            if not line:
                break
            value = json.loads(line)
            if "id" not in value:
                self.events.append(value)

    def stderr_text(self):
        if self.stderr is not None:
            return "<redirected>"
        return self.process.stderr.read()

    def errors(self):
        return [event for event in self.events if event.get("event") in ("warning", "error")]


def rss_mib(pid):
    text = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)]).strip()
    return int(text) / 1024.0


class VirtualDevices:
    """Two null sinks and a remapped microphone source, restored on exit."""

    def __init__(self):
        self.pactl = tool("pactl")
        self.modules = []
        self.sink = self.run(["get-default-sink"]).strip()
        self.source = self.run(["get-default-source"]).strip()

    def run(self, args):
        output = subprocess.run(
            [self.pactl] + args, check=True, capture_output=True, text=True
        )
        return output.stdout

    def load(self, args):
        module = self.run(["load-module"] + args).strip()
        self.modules.append(module)

    def __enter__(self):
        self.load(
            ["module-null-sink", "sink_name=tinta_probe_a",
             "sink_properties=device.description=tinta_probe_a"]
        )
        self.load(
            ["module-null-sink", "sink_name=tinta_probe_b",
             "sink_properties=device.description=tinta_probe_b"]
        )
        self.load(
            ["module-remap-source", "master=tinta_probe_b.monitor",
             "source_name=tinta_probe_mic"]
        )
        self.run(["set-default-sink", "tinta_probe_a"])
        self.run(["set-default-source", "tinta_probe_mic"])
        time.sleep(0.5)
        return self

    def __exit__(self, *error):
        for module in reversed(self.modules):
            subprocess.run([self.pactl, "unload-module", module], capture_output=True)
        if self.sink:
            subprocess.run([self.pactl, "set-default-sink", self.sink], capture_output=True)
        if self.source:
            subprocess.run([self.pactl, "set-default-source", self.source], capture_output=True)
        return False


def wav_tone(path, seconds, frequency):
    """A mono 16 kHz WAV tone, written with the engine's own encoder layout."""
    import math
    import struct
    import wave

    count = int(16000 * seconds)
    frames = bytearray()
    for index in range(count):
        fade = min(1.0, index / 400.0, (count - index) / 400.0)
        sample = int(
            0.5
            * fade
            * math.sin(2.0 * math.pi * frequency * index / 16000.0)
            * 32767
        )
        frames += struct.pack("<h", sample)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(bytes(frames))


def wav_samples(path):
    """Returns mono float samples from a 16-bit WAV."""
    import wave

    with wave.open(path, "rb") as handle:
        assert handle.getnchannels() == 1 and handle.getsampwidth() == 2
        frames = handle.readframes(handle.getnframes())
    import array

    values = array.array("h")
    values.frombytes(frames)
    return [value / 32767.0 for value in values]


def rms(samples, start, end):
    start = max(0, int(start * 16000))
    end = min(len(samples), int(end * 16000))
    if start >= end:
        return 0.0
    window = samples[start:end]
    return (sum(value * value for value in window) / len(window)) ** 0.5


def sample_track(engine, directory, track, seconds):
    """Exports a whole track to a WAV with the `sample` command and reads it."""
    path = os.path.join(directory, "%s.wav" % track)
    reply = engine.command(90, "sample", {"dir": directory, "audio_key": KEY,
                                          "track": track, "s": 0.0, "e": seconds},
                           timeout=120)
    if not reply.get("ok"):
        raise RuntimeError("sample failed: %s" % reply.get("error"))
    with open(path, "wb") as handle:
        handle.write(base64.b64decode(reply["result"]["wav_base64"]))
    return wav_samples(path)
