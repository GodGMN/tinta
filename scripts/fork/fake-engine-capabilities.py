#!/usr/bin/env python3
"""A fake `tinta-engine` that has none of the optional features.

It speaks the line protocol of `docs/engine-protocol.md`: one JSON object per line on
stdin and stdout, logs on stderr. `ready` and `ping` report every capability as false,
and every command it receives is appended to a log, so a test can prove that the app
does not use a feature that the engine said it does not have.

Use it for the manual check in `docs/fork/manual-checks.md`: put it in the place where
the app looks for the engine of a development build, then read the log after using the
app for a while.

    TINTA_FAKE_ENGINE_LOG=/tmp/tinta-fake-engine.log <path to this script>

The log defaults to `fake-engine.log` in the working directory. Every line is the
received command with its parameters, as JSON.
"""

import json
import os
import sys

VERSION = "0.9.3"

CAPABILITIES = {
    "capture_system_audio": False,
    "capture_app_audio": False,
    "calls_detection": False,
    "call_reader": False,
    "calendar": False,
    "summaries": False,
    "echo_cancellation": False,
    "import_audio": False,
}

# Valid, degraded replies, so the app keeps working while it shows no feature. The check
# fails on the log, not on these replies, because a command may arrive from an older app.
RESULTS = {
    "models_status": {"installed": True, "path": "/tmp/tinta-fake-models"},
    "install_models": {"installed": True},
    "permissions": {"microphone": "granted", "accessibility": False},
    "request_microphone": {"granted": True},
    "list_sources": {"sources": [{"id": "all", "name": "System audio", "playing": False}], "default_input": "", "route": "speakers"},
    "calls": {"calls": [], "accessibility": False},
    "request_accessibility": {"granted": False},
    "call_report": {"text": ""},
    "calendar": {"access": "denied", "calendars": [], "events": []},
    "request_calendar": {"granted": False},
    "summary_status": {"available": False, "reason": "Summaries are not available in the fake engine."},
    "start": {"start_wall_ms": 0},
    "pause": {},
    "resume": {},
    "mic_muted": {},
    "stop": {},
    "summarize": {"summary": "", "model": "fake"},
    "finalize": {"tracks": {"mic": {"words": []}, "remote": {"words": []}}, "duration": 0},
    "sample": {"wav_base64": ""},
}


def emit(message):
    sys.stdout.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def record(request):
    path = os.environ.get("TINTA_FAKE_ENGINE_LOG", "fake-engine.log")
    line = json.dumps({"cmd": request.get("cmd"), "params": request.get("params", {})}, separators=(",", ":"))
    with open(path, "a", encoding="utf-8") as log:
        log.write(line + "\n")
    sys.stderr.write("[tinta-engine fake] " + line + "\n")


def reply(request, result):
    emit({"id": request.get("id"), "ok": True, "result": result})


def main():
    emit({"event": "ready", "version": VERSION, "capabilities": CAPABILITIES})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            emit({"event": "error", "message": "invalid request"})
            continue
        if not isinstance(request, dict) or not isinstance(request.get("cmd"), str):
            emit({"event": "error", "message": "invalid request"})
            continue
        record(request)
        command = request["cmd"]
        if command == "shutdown":
            reply(request, {})
            return
        if command == "ping":
            reply(request, {"pong": True, "capabilities": CAPABILITIES})
            continue
        # A command whose feature is off still gets a valid reply, so the check reads the log.
        reply(request, RESULTS.get(command, {}))


if __name__ == "__main__":
    try:
        main()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
