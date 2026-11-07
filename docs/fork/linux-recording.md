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
| microphone | the server's default source name | never opened when it is a speaker monitor; checked on every (re)open, not only at start |
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
default source: a VM, a headless box, and a machine whose only sink is a null
or dummy sink all do it (`auto_null.monitor` is the usual name). Recording that as the microphone duplicates every remote utterance as the
user, and the echo filter cannot catch it because the two levels are equal.

The engine therefore looks at the default source **before every microphone
open**, not only at start:

- a name ending in `.monitor`, or a source whose PulseAudio info has
  `monitor_of_sink`, is a speaker monitor, not a microphone;
- a server that does not answer its defaults is not assumed to have an input:
  there is no `@DEFAULT_SOURCE@` fallback for the microphone, because that would
  be exactly the monitor on a machine without one;
- a monitor is never recorded: the stream is closed (or not opened), nothing is
  written, and the microphone track stays absent until a real input appears;
- one `warning` event says so and one stderr line records the reason;
- `levels` still arrives every 250 ms with `mic: 0` and the current
  `mic_muted`, so the microphone meter behaves as if it were muted.

The remote track and the rest of the session are unaffected; a start does not
fail just because a VM has no microphone.

The check runs again on **every reopen**. When a USB headset is unplugged, or a
Bluetooth headset turns off, the server makes the speaker monitor the default
source; the watcher sees the change, the microphone loop closes its stream and
reports the monitor instead of opening it. The same happens when the server
moves the stream onto a monitor by itself, without a read error: the check is on
the server's default, not on the stream's own name. This closes the duplication
bug for a device that disappears mid-meeting, not just at start.

## Microphone recovery

The microphone thread keeps running for the whole meeting. While no real input is
the default it writes nothing and asks the server again every 2 s; the device
watcher keeps refreshing the snapshot meanwhile. When a real input becomes the
default — a USB headset is plugged in, a Bluetooth headset switches from A2DP to
HFP when a call app opens the input, or the monitor default is replaced — the
loop opens it and the microphone track resumes:

- the first read is placed by its capture time, so the missing span is silence
  and the track stays aligned with the meeting audio;
- the `audio/mic` directory is created at that moment (it does not exist for a
  meeting that never gets a microphone);
- one `warning` announces the change in both directions and one stderr line
  records it;
- `stop` during the wait returns promptly.

## Stop and a wedged server

`stop` sets the capture flag, joins the threads, releases the held microphone
audio and flushes both writers. A PulseAudio call has no timeout, so a wedged
server could block a capture thread in `Simple::read` or `Simple::new` forever.
`stop` therefore waits at most 5 s for the threads; if one is still blocked it
flushes what was written, detaches that thread (it drops anything it reads after
the flag), and the command replies with an `error` instead of hanging.

## Clock, gaps and drift

The meeting clock starts at `start_wall_ms`, taken just before the capture
threads start. Each read is placed by **when its audio was recorded**, not by
when the client got around to reading it:

- `pa_simple_get_latency` gives how far the read head trails the capture head;
  the read's last sample was captured at `read completion - latency`.
- The **sample counter** is the primary clock: within a 100 ms (one fragment)
  deadband the next read continues exactly where the previous one ended, so
  timestamp jitter never moves a track.
- Past 100 ms of error, the counter is corrected in **small steps**, never by a
  whole read: at most 5 ms per read in quiet audio and 1 ms while the read is
  loud (`RMS >= 0.02`). A fast device has that many samples trimmed from the
  front of the read; a slow one gets that much silence in front of it, written
  as real samples so the chunk writer keeps them.
- A jump beyond 200 ms is not clock drift: audio was lost (a read error, a
  device that stalled), so it is placed at its true time and the span becomes a
  silent gap, as before.

This is what makes a stall harmless: PulseAudio keeps buffering while the
capture thread waits on the recorder mutex (a 5 s chunk flush, the live echo
check, a slow `fs::write`). When the thread resumes, the backlogged fragments
still carry their old capture times and land where they were recorded, so the
track is not shifted. Unit tests drive a fake source with capture times: a
500 ms stall leaves the track within 50 ms of the wall clock (measured residual
0 samples). A 1 % device clock error over ten minutes stays inside the 100 ms
deadband in both directions (measured +83 ms for a fast device, -100 ms for a
slow one, the same at one minute and at ten), and the tests assert that no
single correction trims or pads more than 5 ms (80 samples).

Without capture times (a source that cannot report latency) the loop falls back
to sample counting, as before. A failed `pa_simple_get_latency` returns `None`
for that read instead of zero latency, so it becomes sample counting rather than
a shift. A latency of many seconds is a real backlog, not a broken answer, so
there is no 5 s clamp: a long stall lands its buffered audio where it was
captured, and any still-missing span becomes silence, never a dropped read.

A `pause`, `resume` or `mic_muted` that arrives while `start` is still opening
the devices is kept and replayed in order once the session is active, with the
mute's original wall-clock time. That matches Swift, where `start` blocked the
command loop, so a mute clicked right after Record still starts the meeting
muted instead of being lost.

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

A `device-watch` thread keeps one introspection connection while a stream is
open. It subscribes to source, sink and server changes, so a device being
unplugged wakes it at once, and it also asks `get_server_info` every 2 s as a
fallback. Each stream notices a change to its own track's default. The watcher
only polls while a stream is open, so an idle engine costs nothing. The capture loop then closes that stream and opens the current default
by name; the meeting clock keeps running, so the short hand-over is silence.
The microphone loop also treats a monitor as a reason to close the stream (see
[No physical microphone](#no-physical-microphone)).

If the server restarts (`systemctl --user restart pipewire`), every `query()`
fails once the connection dies. The watcher then drops the connection, reopens
it with a 0.5-2 s backoff and polls again; it writes one log line when it loses
the server and one when it reconnects. It also marks the snapshot **stale**, so
`start` does not open devices by names that may be gone: an open asks the server
fresh instead of trusting the old answer. An answer with no default source is
kept (with `source: None`), so a server that has no input is reported as such
instead of being discarded together with its sink.

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
  new meeting records holds one copy. That case is measured with the finalize
  running (see [Memory](#memory)): 1484.1 MiB peak, one recognizer. The cache
  holds a `Weak`, so the models are released when the last pass ends; the
  loader's thread budget applies, and the live pass asks for at most two.
- While a final pass holds the model lock, the live worker does not wait for it:
  one `try_lock` either takes the models or skips that utterance immediately
  (the final pass transcribes it anyway), and the worker keeps feeding the VAD.
  It never blocks on the lock, so the queue drains and no single utterance is
  queued behind a multi-minute finalize. Drafts are dropped for the duration of
  a finalize; memory does not grow and there is no deadlock.
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
(`scripts/fork/recording/rss_probe.py`): 38.4 MiB at start, 1059 MiB after the
first live utterance, **peak 1061.3 MiB**, and 1061.3 MiB for the last seven
minutes (tail spread 2.3 MiB over 621 `levels` ticks and 33 live drafts). The
recognizer dominates a debug build; a silent ten-minute recording stays at
51 MiB. Memory tracks the models in use, not the length of the meeting.

A concurrent final pass was measured too (`scripts/fork/recording/
finalize_rss_probe.py`, debug build limited to two CPUs with `taskset`): a
5-minute meeting was imported and then finalized in 70.7 s while another
meeting recorded with speech playing into the default sink. The process held
**one** recognizer (the weak cache shares it with the live pass) plus the
diarization models and the decoded audio of the finalized meeting: **peak
1484.1 MiB**, 1398.6 MiB at the end of the finalize, sampled 36 times every
2 s. That is the worst case measured; the finalize's extra ~400 MiB leaves
with the pass, and the live-only figure above is the steady state.

## Crash recovery

A killed engine leaves the chunks written so far. Chunk files are renamed from
`.part` only after a complete encrypted write, so a crash can lose at most the
current 5 s buffer and leaves a `.part` that the reader ignores. On restart the
app marks the meeting stopped and runs `finalize`, which reads whatever chunks
exist. The `finalize` path is unchanged and needs no session state.

## Tests

Unit tests (`cargo test -p tinta-engine-rs`) cover the clock (a buffered stall,
a real outage, fast and slow device clocks over ten simulated minutes with a
5 ms cap on every correction, and a loud audio stream corrected in 1 ms steps),
pause, the microphone hold and late mute, mute ordering, seeded mute, levels,
bounded echo levels, a session with no microphone, the model-lock hand-off and
session JSON. They also cover the microphone paths: the monitor check including
the `monitor_of_sink` flag, a changed default reopening the track, a microphone
that appears later resuming on the meeting clock, and a microphone that becomes
a monitor being closed with one warning. No audio device is needed.

The end-to-end test is ignored by default because it needs a pulse server and
`pactl` / `paplay` (Debian: `pulseaudio-utils`):

```text
cargo test --locked -p tinta-engine-rs --test recording_e2e -- --ignored --nocapture
```

It creates two null sinks and a remapped microphone source, records through the
real engine (remote tone, microphone tone, a mute window, a paused tone, then a
post-resume microphone tone), checks both tracks, the mute silence and the
`levels` events. A second case starts with a monitor default source and checks
that no microphone track is created. A third unloads the microphone module
mid-recording (a real unplug, so the server falls back to the monitor), plays
meeting audio, plugs a new input back in and checks that the microphone track
stays silent during the fallback and resumes afterwards. Every case unloads the
modules and restores the defaults even on failure.

The manual probes behind the review evidence live in
`scripts/fork/recording/` and are reproducible: `live_probe.py`, `mute_probe.py`,
`reopen_probe.py`, `rss_probe.py`, `finalize_rss_probe.py` and
`recovery_probe.py` (see the README there for the environment). The
monitor-fallback probe from the review (`set-default-source <sink>.monitor`, or
unloading the microphone module mid-recording) reports `mic 0.000` after the
change with this revision, where it reported `mic 0.500` before.

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
4. Unplug a USB microphone mid-recording. The recording continues after a short
   silence, with one warning notice, and the microphone track does not pick up
   the speakers. Plug the microphone back in: the track resumes at its true
   place on the meeting clock and one notice says so. Switching the default
   output mid-recording reopens the meeting-audio stream the same way.
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
