# Recording probes

The manual probes behind `docs/fork/linux-recording.md`. They drive the real
engine over its JSON-lines protocol; the unit tests and the ignored
`recording_e2e` test stay in `crates/tinta-engine-rs/tests/`.

## Environment

The probes need a running PulseAudio or PipeWire pulse server and the
`pulseaudio-utils` binaries (`pactl`, `paplay`). When they are not on `PATH`,
point at them:

```sh
export TINTA_PACTL=/tmp/opencode/audio-prefix/root/usr/bin/pactl
export TINTA_PAPLAY=/tmp/opencode/audio-prefix/root/usr/bin/paplay
export LD_LIBRARY_PATH=/tmp/opencode/audio-prefix/root/usr/lib/x86_64-linux-gnu
```

Build the engine first (`cargo build -p tinta-engine-rs --bin tinta-engine`).
`ENGINE` below is `target/debug/tinta-engine`; `MODELS` is a directory with the
installed models; `WAV` is a speech clip.

## Probes

| script | what it checks |
|---|---|
| `live_probe.py ENGINE MODELS WAV [OUT]` | live drafts arrive and the microphone does not repeat the remote transcript |
| `mute_probe.py ENGINE WAV OUT` | `mic_muted` silences the exact wall-clock window (the 0.8 s hold) |
| `reopen_probe.py ENGINE OUT` | the remote capture follows a changed default sink mid-recording |
| `rss_probe.py ENGINE MODELS WAV OUT` | RSS over `TINTA_RSS_MINUTES` (default 10) with speech in a loop, so the ASR model is loaded |
| `finalize_rss_probe.py ENGINE MODELS WAV OUT` | RSS while a final pass of an imported meeting runs during a recording with speech, the worst case for memory |
| `recovery_probe.py ENGINE MODELS WAV OUT` | `kill -9` mid-recording, then `finalize` from the chunks on disk |

`mute_probe.py` and `reopen_probe.py` create two null sinks and a remapped
microphone source and restore the previous defaults and modules on exit, like
the end-to-end test.

`live_probe.py` plays into the *default* sink, so run it on the host under test:
on a machine whose default source is `<sink>.monitor`, the engine must record no
microphone track and the probe fails if the two transcripts match.
