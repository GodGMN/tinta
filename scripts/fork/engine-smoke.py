#!/usr/bin/env python3
"""Smoke-test a packaged tinta-engine over the JSON-lines protocol.

Pings the engine, installs the pinned models, imports a speech clip as the
remote track, finalizes it, and asserts that the result has words and at least
one speaker. This is what the `integration-smoke` CI job runs on the engine from
the installed .deb, with a real Secret Service in the session.

Usage:
  TINTA_ENGINE_MODELS_DIR=/tmp/models \
    scripts/fork/engine-smoke.py --engine /usr/bin/tinta-engine \
      --work /tmp/meeting --clip /tmp/speech.wav
"""

import argparse
import base64
import json
import os
import queue
import subprocess
import sys
import threading
import time


class Engine:
    def __init__(self, path):
        self.next_id = 0
        self.events = []
        self.ready = None
        self.process = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.lines = queue.Queue()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.process.stdout:
            if line.strip():
                self.lines.put(line)
        self.lines.put(None)

    def _send(self, command, params):
        self.next_id += 1
        request = {"id": self.next_id, "cmd": command, "params": params}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self.next_id

    def request(self, command, params=None, timeout=600, show_progress=False):
        request_id = self._send(command, params or {})
        deadline = time.monotonic() + timeout
        last_progress = 0.0
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"{command}: no reply after {timeout}s")
            try:
                line = self.lines.get(timeout=remaining)
            except queue.Empty:
                raise RuntimeError(f"{command}: no reply after {timeout}s")
            if line is None:
                raise RuntimeError(
                    f"{command}: the engine closed stdout (exit {self.process.poll()})"
                )
            message = json.loads(line)
            if message.get("event"):
                self.events.append(message)
                if message["event"] == "ready":
                    self.ready = message
                if message["event"].endswith("_progress") and show_progress:
                    now = time.monotonic()
                    if now - last_progress >= 10:
                        last_progress = now
                        print(
                            f"  {message['event']}: "
                            f"{message.get('component') or message.get('stage')} "
                            f"{message.get('fraction')}",
                            flush=True,
                        )
                continue
            if message.get("id") != request_id:
                print(f"unexpected reply: {message}", file=sys.stderr)
                continue
            if not message.get("ok"):
                raise RuntimeError(f"{command}: {message.get('error')}")
            return message.get("result")

    def close(self):
        try:
            self.request("shutdown", timeout=30)
        except Exception:
            pass
        try:
            self.process.stdin.close()
        except Exception:
            pass
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="/usr/bin/tinta-engine")
    parser.add_argument("--work", required=True, help="empty folder for the meeting")
    parser.add_argument("--clip", required=True, help="speech audio file to import")
    parser.add_argument("--install-timeout", type=int, default=3600)
    parser.add_argument("--finalize-timeout", type=int, default=3600)
    arguments = parser.parse_args()

    os.makedirs(arguments.work, exist_ok=True)
    meeting = os.path.join(arguments.work, "meeting-smoke")
    os.makedirs(meeting, exist_ok=True)
    audio_key = base64.b64encode(bytes(range(32))).decode()

    engine = Engine(arguments.engine)
    try:
        ping = engine.request("ping", timeout=60)
        assert ping.get("pong") is True, f"ping returned {ping}"
        assert engine.ready is not None, "the engine did not emit ready"
        print(f"engine ready: {engine.ready.get('version')}", flush=True)

        status = engine.request("models_status", timeout=60)
        print(f"models: {status}", flush=True)

        print("installing the pinned models (cached between runs)...", flush=True)
        installed = engine.request(
            "install_models", timeout=arguments.install_timeout, show_progress=True
        )
        assert installed.get("installed") is True, f"install returned {installed}"

        print(f"importing {arguments.clip}...", flush=True)
        imported = engine.request(
            "import_audio",
            {
                "dir": meeting,
                "audio_key": audio_key,
                "path": os.path.abspath(arguments.clip),
            },
            timeout=600,
        )
        duration = float(imported.get("duration", 0))
        assert duration > 0.5, f"import returned duration {duration}"

        print("finalizing...", flush=True)
        result = engine.request(
            "finalize",
            {"dir": meeting, "audio_key": audio_key},
            timeout=arguments.finalize_timeout,
            show_progress=True,
        )

        tracks = result.get("tracks") or {}
        remote = tracks.get("remote") or {}
        words = remote.get("words") or []
        speakers = {
            segment.get("speaker")
            for segment in remote.get("diarization") or []
            if segment.get("speaker")
        }
        assert words, f"finalize returned no words: {result}"
        assert speakers, f"finalize returned no speakers: {result}"

        print(f"duration: {result.get('duration')}s", flush=True)
        print(f"words: {len(words)}", flush=True)
        print(f"speakers: {sorted(speakers)}", flush=True)
        print("transcript: " + " ".join(word.get("w", "") for word in words[:40]), flush=True)
        print("engine smoke test passed", flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
