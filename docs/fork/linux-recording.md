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
| microphone | the server's default source name | skipped when it is a speaker monitor; `@DEFAULT_SOURCE@` when the server does not answer |
| remote | `<default sink>.monitor` | monitor of the default output sink; `@DEFAULT_MONITOR@` fallback |

PipeWire 0.3.65's pulse server does not resolve `@DEFAULT_MONITOR@` like
PulseAudio does (it opened the default *source*), so the engine asks the server
for `default_sink_name` / `default_source_name` with an introspection connection
and opens the monitor by name. The lookup happens on every (re)open, so a new
default device is picked up, and a watcher polls it while recording (see
[Device changes](#device-changes)).

Both streams ask for 16 kHz mono `S16LE`; the server resamples from the device
rate. The 100 ms fragment bounds one blocking read, which keeps `pause` and
`stop` prompt. Right before the meeting clock starts, `flush` drops whatever the
server buffered while the second stream was opening, so both tracks begin at the
same instant.

`list_sources` keeps the single `all` entry: Linux captures the whole system
mix, not per-application audio.

`start` opens the two streams on a worker thread and replies when they are
open. A slow server or a two-second introspection timeout therefore does not
stop `ping`, `models_status` or the other quick commands, and a second `start`
is rejected while the first is opening.

## No physical microphone

A machine without an input device makes the **monitor of its speakers** the
default source: a VM, a headless box, and this host (`auto_null.monitor`) all do
it. Recording that as the microphone duplicates every remote utterance as the
user, and the echo filter cannot catch it because the two levels are equal.

The engine therefore looks at the default source before opening it:

- a name ending in `.monitor`, or a source whose PulseAudio info has
  `monitor_of_sink`, is a speaker monitor, not a microphone;
- in that case the recording keeps **no microphone track** — no `audio/mic`
  directory is created, and `finalize` reads it as empty;
- one `warning` event says so and one stderr line records the reason;
- `levels` still arrives every 250 ms with `mic: 0` and the current
  `mic_muted`, so the microphone meter behaves as if it were muted.

The remote track and the rest of the session are unaffected; a start does not
fail just because a VM has no microphone.

## Clock, gaps and drift

The meeting clock starts at `start_wall_ms`, taken just before the capture
threads start. Each read is placed by **when its audio was recorded**, not by
when the client got around to reading it:

- `pa_simple_get_latency` gives how far the read head trails the capture head;
  the read's last sample was captured at `read completion - latency`.
- The **sample counter** is the primary clock: within a 100 ms (one fragment)
  deadband the next read continues exactly where the previous one ended, so
  timestamp jitter never moves a track.
- The capture time corrects the counter in **both directions**:
  - behind the wall clock by more than 100 ms: the missing audio is a silent gap
    at its true place (a real outage);
  - ahead of the wall clock by more than 100 ms: the device produced more
    samples than real time allows, and the excess is dropped from the front of
    the read instead of stretching the track.

This is what makes a stall harmless: PulseAudio keeps buffering while the
capture thread waits on the recorder mutex (a 5 s chunk flush, the live echo
check, a slow `fs::write`). When the thread resumes, the backlogged fragments
still carry their old capture times and land where they were recorded, so the
track is not shifted. Unit tests drive a fake source with capture times: a
500 ms stall leaves the track within 50 ms of the wall clock (measured residual
0 samples), and a 1 % device clock error over ten minutes stays inside the
100 ms deadband in both directions (measured +83 ms for a fast device, -100 ms
for a slow one, the same at one minute and at ten).

Without capture times (a source that cannot report latency) the loop falls back
to sample counting, as before.

## Pause and stop

`pause` drops incoming samples on both tracks, releases the held microphone
audio and flushes both writers. `resume` continues at the true clock offset, so
the paused interval reads as silence. `stop` does the same and also ends the
capture threads.

## Microphone mute

`mic_muted` stores the change by meeting-clock sample index
(`max(0, (t - start_wall_ms) * 16000 / 1000)`), kept sorted; the later command
wins at the same index. The microphone is held for 0.8 s before it is written,
so a mute that arrives shortly after the click still silences the audio from the
exact time. `start` with `mic_muted: true` seeds a mute at sample 0.

The microphone stream stays **open** while muted, as in Swift (`Session.swift`
only stops the microphone in `stop()`). The desktop microphone indicator
therefore stays on during a mute, and up to 0.8 s of muted audio sits in memory
until it is zeroed and written.

## Device changes

A changed default device does **not** raise a read error, and on plain
PulseAudio a stream opened by name is **not moved** by the server: it would keep
recording the old device, or silence, for the rest of the call. PipeWire does
move the stream, but the engine does not rely on that.

A `device-watch` thread polls `get_server_info` every 2 s over one long-lived
introspection connection, and each stream notices a change to its own track's
default. The capture loop then closes that stream and opens the current default
by name; the meeting clock keeps running, so the short hand-over is silence.

A read error takes the same reopen path: one `warning`, then retries back off
from 300 ms to 30 s, and a failure is logged at most once per 30 s (a device
gone overnight writes a log line every 30 s, not every 300 ms). `source_state`
follows the remote capture state.

## Live drafts and the final pass

When the speech models are installed, a background thread runs a streaming
Silero VAD per track on stored audio and decodes each finished segment with
Parakeet, emitting `live {track, s, e, text}`. The microphone is checked against
the remote level to drop residual speaker echo, and muted microphone audio is
already silent.

- The audio queue is 32 pieces of 100 ms per worker; a slow model drops drafts
  instead of growing memory, and the dropped span becomes a zero gap so times
  stay on the meeting clock.
- The recognizer is **shared**, not copied. A weak cache keyed by models
  directory and provider hands the live pass and a concurrent final pass the
  same Parakeet instance, so the previous meeting's finalize running while the
  new meeting records holds one copy (the measured peak RSS below includes it).
  The cache holds a `Weak`, so the models are released when the last pass ends;
  the loader's thread budget applies, and the live pass asks for at most two.
- While a final pass holds the model lock, the live worker does not wait for it:
  it skips that utterance (the final pass transcribes it anyway) and keeps
  feeding the VAD. Drafts are dropped for the duration of a finalize; memory
  does not grow and there is no deadlock.
- `stop` sets the stopping flag before anything else. The worker checks it before
  every emit and gives up on the model lock after it is set, so no `live` event
  can reach the next meeting. `finish` waits up to 1.5 s for the worker; a
  decode still running after that is left to end on its own, named thread
  `live-drafts`, and cannot emit.
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

## Privacy

> The **remote track records all system output**, not just the meeting. Any app
> that plays sound while Tinta records — a video, music, a notification, a
> second call — is in the meeting's remote track, is transcribed, and appears in
> the transcript. Play nothing private while recording, or stop the recording.
> The per-application source selector does not exist on Linux: `list_sources`
> offers the single `all` target.

The microphone track records the default input source. It is muted from the
app; the stream itself stays open (see [Microphone mute](#microphone-mute)), so
the system microphone indicator stays on while muted, exactly as with the Swift
engine. Audio and transcripts stay on disk, encrypted with the meeting key.

## Memory

Every buffer is fixed or bounded by time, not by the length of the meeting:

- capture reads are 100 ms buffers;
- the microphone hold is 0.8 s;
- remote levels kept for the echo check are at most ~60 s;
- the live queue is 32 × 100 ms per worker plus two streaming VADs (~2 MB each);
- the shared Parakeet recognizer loads lazily on the first live utterance or the
  final pass; a silent recording never loads it.

Measured over ten minutes with speech played in a loop into the default sink,
so the live Parakeet was loaded and decoding
(`scripts/fork/recording/rss_probe.py`): 32.6 MiB at start, 1012 MiB after the
first live utterance, **peak 1042.7 MiB**, and 1022.8 MiB for the last seven
minutes (tail spread 22.1 MiB over 650 `levels` ticks and 22 live drafts). The
recognizer dominates a debug build; a silent ten-minute recording stays at
51 MiB. Memory tracks the models in use, not the length of the meeting.

## Crash recovery

A killed engine leaves the chunks written so far. Chunk files are renamed from
`.part` only after a complete encrypted write, so a crash can lose at most the
current 5 s buffer and leaves a `.part` that the reader ignores. On restart the
app marks the meeting stopped and runs `finalize`, which reads whatever chunks
exist. The `finalize` path is unchanged and needs no session state.

## Tests

Unit tests (`cargo test -p tinta-engine-rs`) cover the clock (a buffered stall,
a real outage, fast and slow device clocks over ten simulated minutes), pause,
the microphone hold and late mute, mute ordering, seeded mute, levels, bounded
echo levels, a session with no microphone, the model-lock hand-off and session
JSON. No audio device is needed.

The end-to-end test is ignored by default because it needs a pulse server and
`pactl` / `paplay` (Debian: `pulseaudio-utils`):

```text
cargo test --locked -p tinta-engine-rs --test recording_e2e -- --ignored --nocapture
```

It creates two null sinks and a remapped microphone source, records through the
real engine (remote tone, microphone tone, a mute window, a paused tone, then a
post-resume microphone tone), checks both tracks, the mute silence and the
`levels` events, and unloads the modules and restores the defaults even on
failure.

The manual probes behind the review evidence live in
`scripts/fork/recording/` and are reproducible: `live_probe.py`, `mute_probe.py`,
`reopen_probe.py`, `rss_probe.py` and `recovery_probe.py` (see the README there
for the environment).

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
   transcribed a second time from the microphone track. Remember that the video
   itself is in the remote track (see [Privacy](#privacy)) and is transcribed: a
   video played into the call is part of the meeting.
7. On a machine with no microphone (VM, headless, monitor-only default source),
   start a recording. One warning says no microphone was found, the microphone
   meter stays at 0, and the finished meeting has no microphone track; nothing
   from the speakers is attributed to the user.
8. `kill -9` the engine mid-recording, then restart Tinta. The meeting is
   finalized from the chunks already on disk (at most ~5 s lost).
9. Record 30+ minutes with speech playing and check RSS stays flat in
   `ps`/System Monitor; the live transcript keeps up or falls behind without
   growing memory.
