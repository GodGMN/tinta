# Live recording on Linux

Task #11 of the Windows/Linux port. The Rust engine records two tracks — the
microphone and the sound the computer plays — and writes the same encrypted
TNTC chunks as the Swift engine, so `finalize` and the app work unchanged.

## Capture

Linux uses the **PulseAudio simple API** (`libpulse-simple-binding`). PipeWire's
pulse server serves the same API, so one implementation covers PulseAudio and
PipeWire (the common desktop setup on Linux). The later Windows port can add
WASAPI behind the same `CaptureSource` trait without touching the session code.

| Track | Device | Notes |
|---|---|---|
| microphone | the server's default source name | `@DEFAULT_SOURCE@` when the server does not answer |
| remote | `<default sink>.monitor` | monitor of the default output sink; `@DEFAULT_MONITOR@` fallback |

PipeWire 0.3.65's pulse server does not resolve `@DEFAULT_MONITOR@` like
PulseAudio does (it opened the default *source*), so the engine asks the server
for `default_sink_name` / `default_source_name` with an introspection connection
and opens the monitor by name. The lookup happens on every (re)open, so a new
default device after `pactl set-default-sink` is picked up.

The stream asks for 16 kHz mono `S16LE`; the server resamples from the device
rate. The 100 ms fragment bounds one blocking read, which keeps `pause` and
`stop` prompt.

`list_sources` keeps the single `all` entry: Linux captures the whole system
mix, not per-application audio.

## Clock, gaps and pause

- The meeting clock starts at `start_wall_ms`, taken just before the capture
  threads start.
- Capture reads are counted in samples; the wall clock pulls the index forward
  after a stall (more than 100 ms), exactly when the chunk writer would start a
  new chunk. A stall is therefore silence in the track, never a shift.
- `pause` drops incoming samples on both tracks, releases the held microphone
  audio and flushes both writers. `resume` continues at the true clock offset, so
  the paused interval reads as silence. `stop` does the same and also ends the
  capture threads.
- `levels` is emitted every 250 ms with `mic`, `remote`, `remote_capturing` and
  `mic_muted`, the peak RMS since the previous tick. The microphone level is 0
  while muted, like Swift.

## Microphone mute

`mic_muted` stores the change by meeting-clock sample index
(`max(0, (t - start_wall_ms) * 16000 / 1000)`), kept sorted; the later command
wins at the same index. The microphone is held for 0.8 s before it is written,
so a mute that arrives shortly after the click still silences the audio from the
exact time. `start` with `mic_muted: true` seeds a mute at sample 0.

## Device changes

A read error closes the stream, emits one `warning`, and reopens the default
device every 300 ms while the recording continues. The clock keeps running, so
the outage is a silent gap. A removed output or input device, a changed default,
or a restarted pulse server all take this path. `source_state` follows the
remote capture state.

## Live drafts

When the speech models are installed, a background thread runs a streaming
Silero VAD per track on stored audio and decodes each finished segment with
Parakeet, emitting `live {track, s, e, text}`. The microphone is checked against
the remote level to drop residual speaker echo. Live drafts use at most 2 ASR
threads and the same model lock as the final pass.

- The audio queue is 32 pieces of 100 ms per worker; a slow model drops drafts
  instead of growing memory, and the dropped span becomes a zero gap so times
  stay on the meeting clock.
- `TINTA_ENGINE_LIVE=0` turns live drafts off. The final pass still transcribes
  everything from the chunks.
- `capabilities.live_transcription` is true on Linux; the engine starts the
  worker only when every model file exists.

## Capabilities

```json
{
  "capture_system_audio": true,
  "capture_app_audio": false,
  "echo_cancellation": false,
  "import_audio": true,
  "live_transcription": true,
  "calls_detection": false,
  "call_reader": false,
  "calendar": false,
  "summaries": false
}
```

`echo_cancellation` is false: the engine does not run voice processing on the
microphone. The microphone track can contain speaker output; the final pass and
live drafts drop quiet microphone words while the remote track is loud
(`echo.rs`), the same intent as Swift's echo filter.

## Memory

Every buffer is fixed or bounded by time, not by the length of the meeting:

- capture reads are 100 ms buffers;
- the microphone hold is 0.8 s;
- remote levels kept for the echo check are at most ~60 s;
- the live queue is 32 × 100 ms per worker.

Chunks flush every 5 s, so the writer's buffer is also bounded (a very large
single append excepted, as in Swift).

## Crash recovery

A killed engine leaves the chunks written so far. Chunk files are renamed from
`.part` only after a complete encrypted write, so a crash can lose at most the
current 5 s buffer and leaves a `.part` that the reader ignores. On restart the
app marks the meeting stopped and runs `finalize`, which reads whatever chunks
exist. The `finalize` path is unchanged and needs no session state.

## Tests

Unit tests (`cargo test -p tinta-engine-rs`) cover the clock, gaps, pause, the
microphone hold and late mute, mute ordering, seeded mute, levels, bounded echo
levels, session JSON, and chunk gaps. No audio device is needed.

The end-to-end test is ignored by default because it needs a pulse server and
`pactl` / `paplay` (Debian: `pulseaudio-utils`):

```text
cargo test --locked -p tinta-engine-rs --test recording_e2e -- --ignored --nocapture
```

It creates two null sinks and a remapped microphone source, records 28 s through
the real engine (remote tone, microphone tone, a paused tone, then a post-resume
microphone tone), checks both tracks and the `levels` events, and unloads the
modules and restores the defaults even on failure.

Build dependency for the pulse binding: `libpulse-dev` (CI installs it). With
runtime libraries only, extract the Debian packages outside the repo and point
`PKG_CONFIG_PATH` / `LIBRARY_PATH` at them, as earlier tasks did for OpenSSL.

## Manual test checklist (real Linux laptop)

1. Install Tinta and the speech models. With a microphone and speakers, start a
   Google Meet call in Chrome and record it. Both meters move; `remote` follows
   the call audio and `mic` follows your voice.
2. Pause for ~10 s while the call continues, then resume. The meeting waveform
   shows a silent gap on both tracks, and the clock is not shifted.
3. Mute and unmute in Meet. The microphone track is silent from the click (the
   levels `mic` is 0 while muted); unmuting records again. Repeat with the
   Tinta mute button.
4. Unplug a USB microphone or switch the default output in the desktop sound
   settings mid-recording. The recording continues after a short silence, with
   one warning notice and a new default device.
5. Stop and let the final pass run. The remote track transcribes the call, the
   microphone track your voice, durations match, and the meeting opens in Tinta.
6. Play a video out loud during the meeting. The meeting audio should not be
   transcribed a second time from the microphone track.
7. `kill -9` the engine mid-recording, then restart Tinta. The meeting is
   finalized from the chunks already on disk (at most ~5 s lost).
8. Record 30+ minutes and check RSS stays flat in `ps`/System Monitor; the live
   transcript keeps up or falls behind without growing memory.
