# Engine protocol: the Tauri app and the Swift engine

This document is the contract between the Tauri app (`app/src-tauri/`) and the engine
process (`engine/Sources/TintaEngine/`). It is precise enough to implement another engine
— for example a Rust engine for Windows and Linux — that the app accepts **without app
changes**.

The code is the source of truth. Every claim below has a `file:line` reference. Where the
code does not decide a behavior, the text says so instead of guessing; see
[Ambiguities](#11-ambiguities-and-things-the-code-does-not-fix).

> **Not this protocol:** `crates/tinta-core/src/protocol.rs` is the *helper socket*
> protocol between the app and the signed helper binaries (`tinta-mcp`,
> `tinta-native-host`) over a Unix socket. It uses `method` instead of `cmd`, has a
> `hello` handshake, responses with `result`/`error` and no `ok`, a 1 MiB line limit, and
> peer-path trust. None of that applies to the engine. See
> [§10.3](#103-the-helper-socket-protocol-is-a-different-protocol).

---

## 1. Process lifecycle

### 1.1 The binary the app runs

The app runs one executable named `tinta-engine`:

- First choice: a file named `tinta-engine` in the same folder as the app executable
  (`app/src-tauri/src/engine.rs:33-38`). In a Tauri bundle the sidecar is declared as
  `externalBin: ["binaries/tinta-engine", ...]` (`app/src-tauri/tauri.conf.json:39`) and
  the build copies the SwiftPM product there (`scripts/build.sh:14`).
- Fallback in development: `<repo>/engine/.build/release/tinta-engine`
  (`app/src-tauri/src/engine.rs:38`).

There is no environment variable or setting that overrides the path
(`app/src-tauri/src/engine.rs:33-39`). A second engine must therefore be installed under
that name, or the resolver must be changed in the app, or built to the development
fallback path. This is a real constraint, not a convention.

**Windows needs an app change.** The resolver joins exactly `tinta-engine`
(`engine.rs:36`), but a Windows bundle contains `tinta-engine.exe`, so the bundled check
fails and the app falls back to the build machine's development path (`engine.rs:38`).
A console-subsystem engine spawned without `CREATE_NO_WINDOW` would also open a console
window. The "no app changes" goal therefore covers the wire protocol and the file formats
only, not the process launch or packaging. (The app also does not build off macOS today:
`tinta-core` depends on `security-framework` unconditionally.)

Environment: the child inherits the app's environment. The app does not pass any
arguments to the engine (`app/src-tauri/src/engine.rs:61-66`). The engine's own
`--tap-helper <source>` mode is an internal child-process detail, described in
[§8.3](#83-remote-audio-capture-and-the-tap-helper-mac-only); a second engine need not
implement it.

### 1.2 Spawn

Spawn is **lazy** and idempotent per process slot (`app/src-tauri/src/engine.rs:52-98`):

1. If the app is shutting down (`stopped == true`), the call fails with
   `"the engine is shut down"` and no process starts (`engine.rs:53-55`).
2. If a child exists and `try_wait()` is `None` (still running), the app reuses it
   (`engine.rs:56-59`).
3. Otherwise it spawns with **stdin piped, stdout piped, stderr inherited**
   (`engine.rs:61-66`). Stdout is read on a dedicated thread (`engine.rs:72-95`).

The app does not wait for any handshake before writing commands: it starts the reader
thread, then writes the command line right away (`engine.rs:72-123`). In practice the
engine's first line is `ready` (see below), but nothing in the app requires the command
to arrive after it.

### 1.3 Engine start-up sequence

`Engine.main` does this, in order (`engine/Sources/TintaEngine/Engine.swift:10-36`):

1. If called as `tinta-engine --tap-helper <source>`, run the helper instead
   (`Engine.swift:11-14`).
2. Make stdout line-buffered (`Engine.swift:15`). Every message is written with an
   explicit `"\n"` anyway (`Protocol.swift:17-18`).
3. Turn off model downloads (`ModelPins.setOnline(false)`, `Engine.swift:16`;
   see [§9](#9-model-install-and-verification)).
4. Create the `Controller` actor (`Engine.swift:17`).
5. Emit the **`ready`** event as the first line:
   `{"event":"ready","version":"0.9.3","capabilities":{…}}` (`Engine.swift:18`).
   The version must equal the app version (`AGENTS.md:15`). The app reads the
   `capabilities` object (`crates/tinta-core/src/capabilities.rs`) and forwards the whole
   event to the window (`app/src-tauri/src/lib.rs`, `on_engine_event`). See
   [§10.1](#101-capabilities).
6. Start the desktop-call watcher and the calendar observer (`Engine.swift:19-20`).
   Both are Mac-only. Then start model prewarming in a detached task
   (`Engine.swift:21`, `146-150`).
7. Read stdin line by line on a dedicated thread (`Engine.swift:22-32`). Each line is
   parsed and dispatched in a detached task (`Engine.swift:38-61`).
8. On stdin EOF, call `Controller.shutdown()` and `exit(0)` (`Engine.swift:27-30`).

There is no handshake: the app may write commands before the child has read any. The
capabilities are data, reported in `ready` and in the `ping` reply, not a negotiation. The
app sends `ping` on its start and after every engine exit, and starts the watchers of the
optional features only after that reply, and only for what the engine reports as
supported ([§10.1](#101-capabilities)). An engine that reports no capabilities is the Mac
engine from before the field, and the app treats every feature as supported.

### 1.4 Reply/event dispatch on the app side

The reader thread (`app/src-tauri/src/engine.rs:72-95`) parses each line as JSON and
decides **solely by the `id` field**:

- `id` present and a non-negative integer that fits `u64` → **reply**. It is removed from
  the pending map and the caller receives:
  - `Ok(result)` when `message["ok"] == true` (exact boolean) — the `result` value is
    `message["result"]`, which may be missing (`engine.rs:79-81`);
  - `Err(error)` otherwise, using `message["error"]` when it is a string, else
    `"engine error"` (`engine.rs:82-83`).
- Any other line → **event**, passed to `on_event` (`engine.rs:86-88`).
- A line that is not valid JSON → **silently dropped** (`engine.rs:74-75`).
- When stdout ends, pending calls receive `Err("the engine stopped")` and the app
  synthesizes `{"event":"engine_exit"}` (`engine.rs:90-94`).

Consequences for a second engine:

- Reply ids must be JSON numbers, echoed exactly. String ids, `null`, or negative ids are
  treated as events.
- Replies must carry `"ok": true`; a reply with only `result` is treated as an error.
- `result` should be a JSON object. All app call sites index it as an object
  (`app/src-tauri/src/lib.rs:258`, `commands.rs:56-58`, ...). Non-objects are accepted by
  the transport but make every caller fall back to defaults.
- Error values must be strings; errors are shown to the user (directly, or after
  formatting) so they should be short, complete sentences.
- Writing logs to stdout breaks nothing per se but is dropped; logs belong on stderr.

### 1.5 Timeouts

`Engine::call(cmd, params, timeout)` waits on a channel
(`app/src-tauri/src/engine.rs:101-134`). On timeout it forgets the id and returns
`"the engine did not answer <cmd> in time"` (`engine.rs:127-132`). **Nothing is cancelled
or killed on the engine side**: the engine keeps working and its eventual reply is
dropped.

| Command | Timeout | App call site |
|---|---|---|
| `ping` | 20 s | `lib.rs start_engine` (every engine start) |
| `models_status` | 20 s | `commands.rs:25`, `commands.rs:563` |
| `install_models` | 3600 s (1 h) | `commands.rs:147` |
| `permissions` | 10 s | `commands.rs:26` |
| `request_microphone` | 120 s | `commands.rs:152` |
| `list_sources` | 10 s | `commands.rs:157` |
| `calls` | 20 s | `lib.rs watch_calls` (only with `calls_detection`) |
| `request_accessibility` | 10 s | `commands.rs:88` |
| `call_report` | 20 s | `commands.rs:138` |
| `calendar` | 30 s | `calendar.rs:66` |
| `request_calendar` | 300 s | `calendar.rs:81` |
| `start` | 60 s | `lib.rs:257` |
| `pause`, `resume` | 10 s | `lib.rs:295` |
| `mic_muted` | 5 s | `lib.rs:314` |
| `stop` | 30 s | `lib.rs:324` |
| `summary_status` | 10 s | `commands.rs:28`, `lib.rs:600` |
| `summarize` | 1200 s (20 min) | `lib.rs:639` |
| `finalize` | 3600 s (1 h) | `lib.rs:570` |
| `sample` | 60 s | `commands.rs:408` |
| `import_audio` | 600 s (10 min) | `commands.rs:336` |
| `shutdown` | not awaited | `engine.rs:146-149` |

A command that exceeds its timeout leaves the app state inconsistent by design, and the
engine has no way to know: nothing is cancelled. Two examples with different outcomes:

- A timed-out `start` can still finish in the engine. The app has no `Active` entry, so
  it cannot stop that capture; the session keeps recording until the engine exits.
- A timed-out `finalize` makes the app fail the meeting (`lib.rs:547-558`) and drop the
  late reply; the final pass result is lost even though the engine completed it.

Keeping each command inside its budget is part of the contract, and slow work should be
made resumable from the files on disk rather than retried blindly.

### 1.6 Exit, restart and backoff

- The app supervises the engine **only reactively**. On `engine_exit` it clears the known
  calls (emits `calls: []`), forwards the event, and — if a recording was active — takes
  the meeting away from the recording and starts the final pass on the audio that was
  already written (`lib.rs`, `on_recording_lost`).
- Then it schedules one `ping` command after a delay (`lib.rs start_engine`). That
  command spawns a fresh process if none is running (`engine.rs:56-66`), and the app
  starts the supported watchers again after its reply. Any other command (bootstrap,
  `start`, ...) also respawns immediately; the backoff delays only the `ping`.
- Backoff: first wait 1 s, doubled after each run that lasted less than 60 s, capped at
  60 s; a run of at least 60 s resets the next wait to 1 s
  (`lib.rs:165-169`, `429-436`). A `ready` event records the start of the current run
  (`lib.rs:402-405`), and it is the **only** thing that does: `next_restart_delay` resets
  the wait only when a `ready` timestamp exists and is at least 60 s old (`lib.rs:430-435`).
  **An engine must send `ready`**; one that never does gets restart waits that grow to
  60 s and never reset.
- There is **no heartbeat**, no `ping` supervision and no periodic health check. The app
  learns about a dead engine from stdout EOF (`engine.rs:90-94`), not from a timeout.

### 1.7 Shutdown

`Engine::shutdown` (`app/src-tauri/src/engine.rs:142-151`):

1. Sets `stopped`, so later calls fail and never start a process (`engine.rs:53-55`).
2. Writes `{"id":0,"cmd":"shutdown"}` (no `params`, id `0`, below the first id used for
   real calls, which starts at 1) and flushes.
3. Sleeps 300 ms and then **kills the child unconditionally**, without waiting for a
   reply. It neither closes stdin nor checks the exit status.

The engine handles `shutdown` specially, before the command switch
(`engine/Sources/TintaEngine/Engine.swift:49-53`): it stops the recording session,
replies `{"id":0,"ok":true,"result":{}}` and calls `exit(0)`. A second engine must accept
`shutdown` (with or without `params`), may reply, and must exit promptly. Stdin EOF is
equally valid and should also stop capture, flush chunks and exit 0
(`Engine.swift:27-30`); that path is what a pipe-based supervisor would use.

The app calls `shutdown` on `RunEvent::ExitRequested` / `Exit` after stopping a recording
(`lib.rs:1019-1026`). `stop_recording` runs `stop`, then starts the final pass on a new
thread (`lib.rs:320-339`); the app process may exit before that final pass finishes, which
is why unfinished `recording`/`processing` meetings are re-finalized on the next start
(`lib.rs:972-987`).

---

## 2. Wire format

### 2.1 Framing

- One JSON object per line, UTF-8, `\n` terminated, on **stdin** (requests) and
  **stdout** (replies and events). No blank lines, no pretty printing.
  (`Engine.swift:6-7`, `Protocol.swift:8-19`; reader `engine.rs:72-75`.)
- **Flush after every line.** The app blocks on a reply until the newline reaches its
  reader; a buffered writer that only flushes when its buffer fills stalls every command
  until it times out. The Swift engine sets line buffering (`Engine.swift:15`) and writes
  the newline explicitly (`Protocol.swift:17-18`).
- The engine writes each object with a single lock so concurrent writers cannot interleave
  bytes (`Protocol.swift:4-19`). A second engine must serialize writes the same way.
- There is no maximum line length on either side (`engine.rs:73` uses
  `BufReader::lines`; the engine uses `readLine`, `Engine.swift:24`).
- The app never writes anything except request lines and the `shutdown` line
  (`engine.rs:104-115`, `146`).

### 2.2 Requests

```json
{"id": 7, "cmd": "start", "params": {"dir": "/…/meetings/m1", "audio_key": "…", "source": "com.google.Chrome"}}
```

- `id`: `u64`, unique per app run, increasing from 1 (`engine.rs:46`, `102`). It is not
  reset when the engine is respawned, and old pending entries are dropped with the old
  process.
- `cmd`: string. The app sends exactly the command names in
  [§3](#3-commands).
- `params`: object. May be absent; the app sends `{}` for commands without parameters.
  The engine treats a missing or non-object `params` as `{}`
  (`Engine.swift:47`).
- The engine echoes `id` **as received** (`Engine.swift:46`, `Protocol.swift:27-33`).
  It must be a number in the reply for the app to match it.

Requests are dispatched on a detached task per line, so a new request can be handled while
another command is still awaiting I/O or models (`Engine.swift:48-60`).

### 2.3 Replies

Success (`Protocol.swift:27-29`):

```json
{"id": 7, "ok": true, "result": {"start_wall_ms": 1762400000000}}
```

Failure (`Protocol.swift:31-33`):

```json
{"id": 7, "ok": false, "error": "missing parameter: dir"}
```

- `result` is always a JSON object in the Swift engine because `reply` takes
  `[String: Any]` (`Protocol.swift:27`). The app assumes an object.
- `error` is `"\(error)"` of the thrown Swift error (`Engine.swift:57-59`). For
  `EngineError` that is the `description` (`Protocol.swift:40-43`); for system errors it
  is the platform's description. **Error strings are not a stable contract**; the app
  shows them (e.g. `app/src/App.tsx:142-144`) but never matches them. Exception: the app
  replaces one specific failure entirely — a final pass whose result has no words at all
  becomes `NO_SPEECH` (`lib.rs:572-574`, `149-151`).
- One line per request. The engine never sends a reply for an event or a second reply for
  the same id.
- An unparseable request line (no JSON object, or no string `cmd`) gets no reply; the
  engine emits `{"event":"error","message":"invalid request"}` instead
  (`Engine.swift:38-45`).

### 2.4 Events

```json
{"event": "levels", "mic": 0.03, "remote": 0.12, "remote_capturing": true, "mic_muted": false}
```

Events carry `event` plus arbitrary fields, and **no `id`**. The app routes anything
without an integer `id` to `on_engine_event` (`engine.rs:76-88`). See
[§4](#4-events) for the complete list.

Event field values are JSON-safe by construction: `Output.send` checks
`JSONSerialization.isValidJSONObject` and, if invalid, logs to stderr and **drops the
event** (`Protocol.swift:8-13`). A second engine must not emit NaN/Infinity or other
non-JSON values; they vanish silently.

### 2.5 stderr

stderr is inherited by the app (`engine.rs:64`), so engine logs go to the app's terminal
or log stream. The convention is the prefix `[tinta-engine] ` per line
(`Protocol.swift:35-37`). The app parses nothing from stderr. A second engine should keep
the same prefix so Mac and Windows logs read alike, but nothing depends on it.

### 2.6 Parameter type rules (as implemented)

The engine's helpers define what "valid" means (`Protocol.swift:45-66`):

- `string(key)`: must be a JSON string, non-empty. Missing/empty/wrong type throws
  `missing parameter: <key>` (`Protocol.swift:45-51`).
- `optionalString(key)`: string or empty → `nil` (`Protocol.swift:53-56`).
- `double(key)`: accepts a JSON number that Swift bridges to `Double` **or** `Int`;
  otherwise `missing parameter: <key>` (`Protocol.swift:58-62`).
- `bool(key, default:)`: accepts `Bool`, otherwise the default; there is no error
  (`Protocol.swift:64-66`).
- Other parameters are read ad hoc with `as? Bool`, `as? Int`, `as? [String]` and default
  silently (`Engine.swift:94`, `105`, `117`, `199`, `225`; `Summary.swift:35-38`).

A JSON `null` is therefore indistinguishable from an absent optional. An integer and a
float are interchangeable where a number is expected. Extra parameters are ignored.

### 2.7 A minimal exchange

`<` is engine stdout, `>` is app stdin:

```
< {"event":"ready","version":"0.9.3","capabilities":{…}}
> {"id":1,"cmd":"models_status","params":{}}
< {"id":1,"ok":true,"result":{"installed":true,"path":"/Users/sam/Library/Application Support/FluidAudio/Models"}}
> {"id":2,"cmd":"start","params":{"dir":"/Users/sam/Library/Application Support/Tinta/meetings/m1","audio_key":"…","source":"com.google.Chrome","mic_muted":false}}
< {"id":2,"ok":true,"result":{"start_wall_ms":1762400000000}}
< {"event":"levels","mic":0.02,"remote":0.11,"remote_capturing":true,"mic_muted":false}
< {"event":"live","track":"remote","s":4.2,"e":7.9,"text":"Let us start with the roadmap."}
> {"id":3,"cmd":"stop","params":{}}
< {"id":3,"ok":true,"result":{}}
> {"id":4,"cmd":"finalize","params":{"dir":"/Users/sam/Library/Application Support/Tinta/meetings/m1","audio_key":"…"}}
< {"event":"finalize_progress","stage":"transcription","fraction":1.0}
< {"event":"finalize_progress","stage":"diarization","fraction":0.0}
< {"event":"finalize_progress","stage":"diarization","fraction":1.0}
< {"event":"finalize_progress","stage":"transcription","fraction":1.0}
< {"id":4,"ok":true,"result":{"tracks":{"remote":{"words":[{"w":"Let","s":4.2,"e":4.4}],"diarization":[{"speaker":"S1","s":4.0,"e":8.0}]},"mic":{"words":[]}},"duration":42.5,"language":"en"}}
```

Note that `levels` and `live` can arrive while `start`'s reply is still in flight: the app
matches replies by `id`, not by position.

---

## 3. Commands

The switch is `engine/Sources/TintaEngine/Engine.swift:74-138`; `shutdown` is handled
before it (`Engine.swift:49-53`). Unknown commands throw `unknown command <name>`
(`Engine.swift:135-136`).

Concurrency rules in [§5](#5-concurrency) apply to every command. "Mac-only" marks
commands whose implementation in the Swift engine uses macOS APIs. A new engine turns them
off with a capability, which the app honors; an old app calls them regardless, so a second
engine must still answer them (see [§10.1](#101-capabilities)).

### 3.1 `ping`

| | |
|---|---|
| params | none |
| result | `{"pong": true, "capabilities": {…}}` (`Engine.swift:85-86`; see [§10.1](#101-capabilities)) |
| errors | none |
| events | none |
| notes | The app calls it on every engine start, including the restart after an exit: `ping` starts the process without side effects and carries the capabilities, and the app then starts the watchers that the engine supports (`lib.rs start_engine`). A second engine should keep it free of side effects and fast. |

### 3.2 `models_status`

| | |
|---|---|
| params | none |
| result | `{"installed": bool, "path": string}` (`Engine.swift:78-79`) |
| errors | none |
| events | none |
| notes | `installed` is true when every manifest path exists — **file existence only**, no hashing (`Speech.swift:64-68`). `path` is the model root (`ModelPins.modelsRoot`, `Speech.swift:24`); the app sums its size for Settings (`commands.rs:563-572`). Must answer within 20 s and must not load models. |

### 3.3 `install_models`

| | |
|---|---|
| params | none |
| result | `{"installed": true}` (`Engine.swift:80-82`) |
| errors | any download/hash/load error, propagated as text (`Speech.swift:92-109`, `71-90`) |
| events | `install_progress` ×4 (component `transcription`, `vad`, `diarization`, then `all` with state `done`; `Speech.swift:95-108`) |
| notes | Only command allowed to download (`ModelPins.setOnline`, `Speech.swift:19-22`, called at `93-94`). Timeout is 1 h (`commands.rs:147`). Details in [§9](#9-model-install-and-verification). |

### 3.4 `permissions`

| | |
|---|---|
| params | none |
| result | `{"microphone": "granted" \| "denied" \| "undetermined", "accessibility": bool}` (`Engine.swift:83-84`, `159-165`) |
| errors | none |
| events | none |
| notes | Microphone state maps authorized→`granted`, denied/restricted→`denied`, everything else (including not determined)→`undetermined` (`Engine.swift:159-165`). `accessibility` is `AXIsProcessTrusted()` (`CallReader.swift:75`). Non-Mac: return a meaningful microphone state and `accessibility: false`. |

### 3.5 `request_microphone`

| | |
|---|---|
| params | none |
| result | `{"granted": bool}` (`Engine.swift:85-87`) |
| errors | none thrown by the engine |
| events | none |
| notes | Calls `AVCaptureDevice.requestAccess(for: .audio)`, which blocks until the user answers or returns the stored decision (`Engine.swift:86`). 120 s timeout (`commands.rs:152`). |

### 3.6 `list_sources`

| | |
|---|---|
| params | none |
| result | `{"sources": [{"id": string, "name": string, "playing": bool}], "default_input": string, "route": "speakers" \| "headphones"}` (`Engine.swift:88-92`) |
| errors | none |
| events | none |
| notes | `sources` always contains the five known apps and `all` (`Engine.swift:167-182`): `com.google.Chrome`, `us.zoom.xos`, `com.microsoft.teams2`, `com.tinyspeck.slackmacgap`, `com.apple.WebKit`, plus `{"id":"all","name":"All system audio"}`. `playing` is true when any matching process is running output (`Engine.swift:175-181`), and for `all` when **any** process plays (`Engine.swift:180`). `default_input` is `""` when unknown (`Engine.swift:90`). `route` is the echo-handling route (`Echo.swift:6-29`); the UI only switches labels/meters on it (`app/src/api.ts:230`). It is fine for a non-Mac engine to report `"speakers"` for everything. |

### 3.7 `calls`

| | |
|---|---|
| params | optional `read: bool` |
| result | `{"calls": [{"app": string, "name": string, "since": int64 ms}], "accessibility": bool}` (`Engine.swift:93-95`, `CallWatcher.swift:64-70`) |
| errors | none |
| events | the call watcher emits `call_started`, `call_ended`, `call_state` on its own schedule |
| notes | `read` is a side-effect switch for Accessibility reading (`Engine.swift:94` → `CallWatcher.swift:59-61`); the app sends it on every engine start and whenever the setting changes, but only when the engine reports `calls_detection`, and it is true only when the engine also reports `call_reader` (`lib.rs watch_calls`). `calls` is the app's **only** recovery path for calls after an engine restart: it replays the calls that are already in progress (`lib.rs watch_calls`). `since` is wall-clock ms. Mac-only (Core Audio process list). |

### 3.8 `request_accessibility`

| | |
|---|---|
| params | none |
| result | `{"granted": bool}` (`Engine.swift:96-97`) |
| errors | none |
| events | none |
| notes | Shows the macOS Accessibility prompt once and returns the current trust state (`CallReader.swift:78-81`). Mac-only. Non-Mac: `{"granted": false}`. |

### 3.9 `call_report`

| | |
|---|---|
| params | `app: string` (required, bundle ID) (`Engine.swift:99`) |
| result | `{"text": string}` (`Engine.swift:103`) |
| errors | `"Tinta cannot read the app window. Allow Accessibility access, and keep the call open."` when the engine is not trusted or the app is not running (`Engine.swift:100-102`; `CallReader.swift:93-94`). A running app with no windows yields an empty `text`, not an error (`CallReader.swift:95-107`) |
| events | none |
| notes | A depth-first dump of all windows and the menu bar with depth and node limits (`CallReader.swift:93-108`). Output can contain any text from the window; the app writes it to a user-chosen file (`commands.rs:136-143`). Mac-only. Non-Mac: fail with a clear sentence. |

### 3.10 `calendar`

| | |
|---|---|
| params | optional `days: int`, default `7` (`Engine.swift:105`) |
| result | `{"access": string, "calendars": [Calendar], "events": [Event]}` (see below) |
| errors | none thrown in the normal path; the read returns `access` instead (`Calendar.swift:61-63`) |
| events | `calendar_changed` when the EventKit store changes |
| notes | `access` is `"granted"`, `"denied"`, or `"undetermined"`; write-only access counts as `denied` because Tinta must read events (`Calendar.swift:22-29`). Only timed, non-cancelled events that the user did not decline and that have not ended are returned, soonest first, at most 500 (`Calendar.swift:72-80`, `11-14`). Birthdays are excluded (`Calendar.swift:66`). The range is `now .. now + days*86400`, so events already in progress are included (`Calendar.swift:72-75`). Mac-only (EventKit). |

`Calendar` object (`Calendar.swift:87-89`):

| field | type | notes |
|---|---|---|
| `id` | string | `calendarIdentifier` |
| `title` | string | |
| `color` | string | `#rrggbb`, `""` when not convertible |
| `account` | string | source title, `""` when unknown |

`Event` object (`Calendar.swift:91-115`):

| field | type | notes |
|---|---|---|
| `id` | string | `<eventIdentifier or calendarItemIdentifier>@<occurrence start ms>`; the occurrence marker makes repeating events unique |
| `title` | string | trimmed, `""` if none |
| `start`, `end` | int64 | Unix ms |
| `calendar_id` | string | |
| `attendees` | array | see below; rooms and resources removed (`Calendar.swift:95`) |
| `location` | string | optional; omitted when empty |
| `url` | string | optional; `event.url.absoluteString` |
| `notes` | string | optional, truncated to 20 000 characters (`Calendar.swift:14`, `113`) |

Attendee (`Calendar.swift:102-109`, `122-130`): `name` (**required** by the app's serde
type, `crates/tinta-core/src/calendar.rs:27-28`; may be `""` but must be present, or the
whole `calendar` reply fails to parse), `email` (from a
`mailto:` URL, else `""`), `is_self` (bool), `status` (`"accepted"`, `"declined"`,
`"tentative"`, `"pending"`, `"unknown"`).

The app deserializes the result with serde into `tinta_core::calendar::Snapshot`
(`crates/tinta-core/src/calendar.rs:147-167`): `access` is required, everything else
defaults. It then finds the call link in `url`, then `location`, then `notes`
(`calendar.rs:159-167`). Missing `id`, `title`, `start`, `end`, or `calendar_id` on an event
makes parsing fail and the whole `calendar` command fail, so those fields are required in
practice. Unknown extra fields are ignored.

### 3.11 `request_calendar`

| | |
|---|---|
| params | none |
| result | `{"granted": bool}` (`Engine.swift:106-107`) |
| errors | EventKit's error is propagated (`Calendar.swift:52-57`) |
| events | `calendar_changed` may follow from the store change |
| notes | Requests full event access once and replaces the `EKEventStore` afterwards so it can read events (`Calendar.swift:52-57`). The app then requests notification permission and immediately reads the calendar (`calendar.rs:80-86`). 300 s timeout because the user may leave the prompt open. Mac-only. Non-Mac: `{"granted": false}`. |

### 3.12 `start`

| | |
|---|---|
| params | `dir: string` (required), `audio_key: string` (required), `source: string` (optional, default `"com.google.Chrome"`), `mic_muted: bool` (optional, default false) (`Engine.swift:189-199`) |
| result | `{"start_wall_ms": int64}` (`Engine.swift:202`) |
| errors | `"a recording is already active"`, `"Microphone access is off. Allow Tinta in System Settings, Privacy and Security, Microphone."`, `"the recording stopped before it started"`, `"invalid audio key"`, `"missing parameter: dir"` / `"missing parameter: audio_key"`, plus model errors (`Engine.swift:185`, `191-196`; `152-157`) |
| events | after success: `levels` every 250 ms; `source_state` when the remote tap state changes; `error` on write/live failures; `warning` on capture problems |
| notes | The recording only starts after the microphone permission is granted and the speech models are warmed up, so this command can take tens of seconds on a cold start — the timeout is 60 s (`lib.rs:257`). `start_wall_ms` is captured in `RecordingSession.init` **after** model warm-up and just before capture starts (`Session.swift:141-144`), and the app stores it as the meeting start (`lib.rs:258-261`, `db.rs:611`). |

Details that matter to a second engine:

- The session clock starts at `start_wall_ms`; all audio offsets, mute timestamps and live
  event times are on that clock (`Session.swift:143-144`, `190-193`).
- Both tracks are created immediately and their directories exist before capture
  (`Session.swift:146-150`; `ChunkStore.swift:30-37`).
- A metadata file is written: `<dir>/audio/session-<start_wall_ms>.json` with
  `{"start_wall_ms": int, "sample_rate": 16000, "source": string}`
  (`Session.swift:159-163`). **Nothing in the repo reads it**; it is informational. A
  second engine may write it for parity but must not rely on it.
- `mic_muted: true` seeds a mute change at sample 0 (`Session.swift:145`), so the whole
  microphone track is silence until `mic_muted` says otherwise.
- Failure to start capture propagates as an error and cleans up the partial session
  (`Session.swift:174-181`), but chunk files already written are left in place.
- If `stop` arrives while `start` is still waiting for permission or models, the start is
  cancelled and fails with `"the recording stopped before it started"`
  (`Engine.swift:69-72`, `121`, `196`). `stop` itself still returns success.
- Two concurrent `start` calls cannot both succeed: the second sees `session != nil` or
  `starting == true` and fails with `"a recording is already active"`
  (`Engine.swift:185-187`).
- The app calls `start` with `dir` = the meeting folder under the library
  (`lib.rs:225-227`) and `source` = the bundle ID of the app being recorded, or `"all"`
  (`lib.rs:252-257`; sources from `Engine.swift:167-182`).

### 3.13 `pause`

| | |
|---|---|
| params | none |
| result | `{}` (`Engine.swift:110-112`) |
| errors | none |
| events | none (the app emits its own `recording` event) |
| notes | No session → successful no-op, because it is `session?.setPaused(true)` (`Engine.swift:111`). A `pause` that arrives while `start` is still waiting for permission or models is therefore **lost**: the session does not exist yet and will start unpaused (`Engine.swift:111`, `197-201`). The app normally cannot do this, because it enables the pause control only after `start` returns and it stores `Active` (`lib.rs:275-295`). Pausing drops incoming audio on **both** tracks (`Session.swift:203`), releases the held microphone audio through the mute logic (`309-315`, `238-244`), and flushes all chunk writers (`314`). The meeting clock keeps running: the next chunk starts at its true offset, so a paused interval becomes silence when the track is read (`ChunkStore.swift:125-138`). |

### 3.14 `resume`

Same as `pause`, with `setPaused(false)` (`Engine.swift:113-115`). Audio does not backfill;
the gap stays silent. No session → successful no-op.

### 3.15 `mic_muted`

| | |
|---|---|
| params | `muted: bool` (optional, default false), `t: int64` wall-clock ms (optional, default now) (`Engine.swift:117-118`) |
| result | `{}` |
| errors | none |
| events | `levels` will report `mic_muted` and zero the mic level from the next tick |
| notes | The change is recorded at meeting-clock index `max(0, (t - start_wall_ms) * 16000 / 1000)` and kept sorted (`Session.swift:265-272`). Microphone audio is held for 0.8 s before it is written or transcribed (`Session.swift:134-137`, `226-236`), so a mute command that arrives shortly after the click still silences the audio from the exact time. When muted, samples in the affected intervals are replaced with zeros (`Session.swift:246-263`) and the live mic level is forced to 0 (`Session.swift:290-293`). Changes are stored by meeting-clock index, not by arrival order (`Session.swift:265-272`); for the same index the later command wins, and the whole held buffer is re-evaluated when it is released (`Session.swift:238-244`). No session → no-op (`Engine.swift:118`), so a mute sent before `start` returns is lost; use the `mic_muted` parameter of `start` instead (`Engine.swift:199`). |

The app sends `{"muted":…, "t": <now_ms>}` for a click (`commands.rs:281-283`) and for a
call-app change (`lib.rs:489-497`), always with a wall-clock timestamp.

### 3.16 `stop`

| | |
|---|---|
| params | none |
| result | `{}` (`Engine.swift:120-124`) |
| errors | none |
| events | none |
| notes | No session → successful no-op. Cancels a pending start (`Engine.swift:121`). Stops both captures, releases and stores the held microphone audio, flushes both writers, and finishes the live segmenters (`Session.swift:317-324`). It does **not** emit a final event, close stdout, or delete the process: `finalize` and `sample` keep working. The app always follows `stop` with `finalize` even when `stop` failed or timed out (`lib.rs:320-328`). |

### 3.17 `summary_status`

| | |
|---|---|
| params | none |
| result | `{"available": bool}` or `{"available": false, "reason": string}` (`Summary.swift:15-31`) |
| errors | none |
| events | none |
| notes | Decides whether the app runs an automatic summary after a final pass (`lib.rs:589-593`) and what Settings offers (`commands.rs:27-28`). Mac-only in the current engine: it needs macOS 26 and Apple Intelligence (`Summary.swift:16-30`). Non-Mac: `{"available": false, "reason": "<sentence>"}`. |

### 3.18 `summarize`

| | |
|---|---|
| params | `title: string`, `notes: string`, `lines: [string]`, `language: string` (BCP-47, default `"en"`); all are optional-with-defaults (`Summary.swift:35-38`) |
| result | `{"summary": string, "model": string}` (`Summary.swift:44`) |
| errors | `"The meeting has no notes and no transcript to summarize."`; `"Summaries need macOS 26 or later."`; localized generation errors (`Summary.swift:39-47`, `51-64`) |
| events | `summary_progress` with `fraction` (0…1), several times; final value 0.9 before the last model call (`Summary.swift:136`, `164`) |
| notes | The app builds `lines` as `"[mm:ss] Name: text"` (timed meetings) or `"Name: text"` (`lib.rs:621-632`) and sends the meeting language or `"en"` (`lib.rs:637`). `summary` is Markdown: an overview paragraph, then `### Key points`, `### Decisions`, `### Action items` sections (Spanish headings when `language == "es"`) (`Summary.swift:171-192`). `model` is shown in the UI (`"Apple on-device model"`, `Summary.swift:8`). 20-minute timeout (`lib.rs:639`). The app refuses a second summary of the same meeting (`lib.rs:613-617`) but allows summaries of different meetings to overlap; the engine has no guard. Mac-only. Non-Mac: fail with a clear sentence; keep `summary_status.available=false` so it is not called automatically. |

### 3.19 `finalize`

| | |
|---|---|
| params | `dir: string` and `audio_key: string` (required), `max_speakers: int` (optional) (`Engine.swift:205-232`) |
| result | `{"tracks": {"mic": TrackResult, "remote": TrackResult}, "duration": number, "language"?: string}` (`Engine.swift:229-231`) |
| errors | chunk read/decrypt errors (`"invalid chunk <name>"` and CryptoKit failures), invalid key, model missing/hash mismatch (`ChunkStore.swift:102-123`; `Speech.swift:71-90`, `120-133`) |
| events | `finalize_progress` in track order: remote transcription windows, remote diarization (only when the remote track has words), then mic transcription windows; the transcription fraction restarts at `1/n` for each track (`Engine.swift:214-226`; see [§4](#4-events)) |
| notes | **Allowed while a recording is active** — the comment at `Engine.swift:206` states this, and nothing in the app prevents a new recording from starting while another meeting finalizes: `start_recording` checks only the active session and `started_at` (`lib.rs:231-240`), while the final pass is tracked per meeting id (`lib.rs:136`, `529-543`). Read-only on disk. Two `finalize` calls for different meetings may also overlap; the `Speech` actor serializes model work (`Speech.swift:56`, `180-217`), but a second engine should not assume sequential calls. The app sends `max_speakers` only when there are remote participants, equal to their count (`lib.rs:565-569`). Timeout 1 h. |

`TrackResult` (`transcript.rs:8-30` is the consumer):

| field | type | notes |
|---|---|---|
| `words` | `[{"w": string, "s": number, "e": number}]` | seconds on the meeting clock, ascending per track (`Engine.swift:221-222`; `Speech.swift:180-205`) |
| `diarization` | `[{"speaker": string, "s": number, "e": number}]`, optional | only on the remote track, only when that track has words and more than 2 s of audio; `speaker` is an opaque label such as `"S1"` (`Engine.swift:223-226`; `Speech.swift:208-217`) |

Behavioral details:

- `duration` is `max(samples in each track) / 16000` — sample count, not wall clock
  (`Engine.swift:213-216`). The app stores it as the meeting duration.
- `language` is present only when the concatenated transcript is longer than 20 characters
  (`Engine.swift:230`; `Speech.swift:219-224`).
- The microphone track is filtered against speaker echo: microphone words are dropped when
  remote audio is playing and the microphone level is below 20 % of the remote level
  (`Engine.swift:218-220`; `Echo.swift:32-57`). A second engine should keep the intent
  (do not transcribe the meeting coming back through the speakers) but may implement it
  differently.
- Final transcription is VAD-driven: speech regions with at least 0.2 s of speech and
  0.5 s of silence, grouped into windows of at most ~25 s and at most 14 s of continuous
  speech (`Speech.swift:183-196`). Windows are transcribed one at a time
  (`Speech.swift:197-204`).
- Diarization loads a fresh diarizer per final pass and releases it, because Core ML keeps
  GPU memory proportional to audio length (`Speech.swift:135-147`).
- A missing `audio/remote` or `audio/mic` directory reads as an empty track
  (`ChunkStore.swift:104`). A directory with no chunks yields `duration: 0` and empty
  tracks; the app then fails the meeting with `NO_SPEECH` (`lib.rs:572-574`).
- The command has no side effects on files or the database.

### 3.20 `sample`

| | |
|---|---|
| params | `dir: string`, `audio_key: string`, `track: "mic" \| "remote"`, `s: number`, `e: number` (seconds) — all required (`Engine.swift:258-261`) |
| result | `{"wav_base64": string}` (`Engine.swift:266`) |
| errors | `"missing parameter: …"`, `"invalid track"`, chunk/decrypt errors (`Engine.swift:258-261`; `ChunkStore.swift:102-123`) |
| events | none |
| notes | Reads the whole requested track, then slices `[s,e)` at 16 kHz with clamping: `from = clamp(0, len, s*16000)`, `to = clamp(from, len, e*16000)` (`Engine.swift:262-264`). `s > e`, negative values, or an empty track produce a valid WAV of length 0. 60 s timeout (`commands.rs:408`). The app refuses the call when audio was deleted (`commands.rs:399-402`) and asks for at most 8 s per speaker sample (`commands.rs:366-382`). |

### 3.21 `import_audio`

| | |
|---|---|
| params | `dir: string`, `audio_key: string`, `path: string` (a local audio file), all required (`Engine.swift:235-238`) |
| result | `{"duration": number}` (seconds) (`Engine.swift:253`) |
| errors | `"missing parameter: …"`, `"cannot allocate buffer"`, AVFoundation open/read errors, chunk write errors (the latter are events, not errors) |
| events | `error` if a chunk write fails (`ChunkStore.swift:89-91`) |
| notes | Imports a file as the **remote** track so the final pass separates all voices in it: decode with `AVAudioFile`, resample to 16 kHz mono, append sequential samples starting at index 0, flush (`Engine.swift:235-254`). Only `audio/remote/*.tntc` is written; there is no `audio/mic`, no session JSON, and no database change. The app creates the meeting first, calls this, marks it stopped, then finalizes on a new thread (`commands.rs:323-343`). 10-minute timeout. |

---

## 4. Events

### 4.1 Every event the Swift engine emits

| Event | Fields | Emitted at | App handling |
|---|---|---|---|
| `ready` | `version: string`, `capabilities: object` | `Engine.swift:18` | Stores the capabilities, records the run start, forwards to the frontend (`lib.rs`, `on_engine_event`). First line of a run. |
| `error` | `message: string` | Invalid request `Engine.swift:43`; live transcription/VAD failure `Session.swift:64`, `91`; chunk write failure `ChunkStore.swift:90` | Forwarded (`lib.rs:416`); UI shows it as a notice (`app/src/App.tsx:142-144`). |
| `warning` | `message: string` | Echo removal unavailable `Capture.swift:434-437`; tap helper exited `TapHelper.swift:90-91` | Forwarded; UI notice (`App.tsx:142-144`). |
| `install_progress` | `component: "transcription" \| "vad" \| "diarization" \| "all"`, `state: "downloading" \| "done"` | `Speech.swift:95`, `97`, `99`, `108` | Forwarded; onboarding shows step labels (`app/src/Onboarding.tsx:8-12`, `17-21`). |
| `live` | `track: "mic" \| "remote"`, `s: number`, `e: number`, `text: string` | `Session.swift:57-62` (times in seconds on the meeting clock) | Turns into a live turn and a `live_turn` app event (`lib.rs:364-379`). |
| `levels` | `mic: number`, `remote: number`, `remote_capturing: bool`, `mic_muted: bool` | `Session.swift:290-303`; every 250 ms while a session exists (`Session.swift:182-187`), starting 0.25 s after capture starts | Forwarded (`lib.rs:416`); meters use `mic`, `remote`, `remote_capturing` (`app/src/MeetingView.tsx:473-475`, `app/src/Bubble.tsx:74-76`). |
| `source_state` | `capturing: bool`, `processes: int` | `TapHelper.swift:120-125`, when the tap helper reports a state frame | Forwarded; no current consumer besides the event stream. |
| `calendar_changed` | none | `Calendar.swift:38-45`, at most once per second of quiet after a store change | Triggers a calendar refresh on a new thread (`lib.rs:389`; `calendar.rs:197-203`). |
| `call_started` | `app: string`, `name: string`, `since: int64 ms` | `CallWatcher.swift:88-92` | Adds an `AppCall` (`lib.rs:380-384`, `440-461`). |
| `call_ended` | `app: string`, `name: string`, `left_at: int64 ms` | `CallWatcher.swift:83-87`, after 2 s without audio activity | Removes the call and may auto-stop the recording after a 3 s grace (`lib.rs:390-394`, `501-512`, `754-774`). |
| `call_state` | `participants: [{id, name, is_self}]`, `speaking: [string]`, `mic_muted: bool \| null`, `app: string`, `t: int64 ms` | `CallWatcher.swift:96-111` + `CallReader.swift:18-25`; on change, or every 3 s while a call lasts | Updates participants, speaker events, the microphone state (`lib.rs:385-388`, `465-499`). `id` equals `name`; `mic_muted` may be `null`. |
| `finalize_progress` | `stage: "transcription" \| "diarization"`, `fraction: number` | Transcription: after each window, `1/n … 1` (`Speech.swift:202`); diarization: `0.0` then `1.0` (`Speech.swift:211-213`) | Forwards with a meeting `id` added when exactly one final pass runs (`lib.rs:406-415`); UI progress bar (`app/src/MeetingView.tsx:124-127`). Order: remote transcription, remote diarization (skipped when the remote track has no words), then mic transcription; the transcription fraction restarts for each track and the bar can move backwards (`Engine.swift:214-226`). |
| `summary_progress` | `fraction: number` | `Summary.swift:136` (0.9 before the final call), `164` (0.85·i/n per condensed part) | Same id injection and forwarding (`lib.rs:406-415`); UI bar (`app/src/Summary.tsx:63-65`). Not strictly monotonic. |

The app-synthesized `engine_exit` (`engine.rs:94`) is not emitted by the engine; it is
delivered to `on_engine_event` like any other event and triggers restart recovery
(`lib.rs:395-401`). A second engine should never send `engine_exit` itself.

Any event name not listed above is still forwarded to the frontend unchanged as a Tauri
`engine` event (`lib.rs:416`), so adding events is backward compatible, but nothing in the
current app reacts to them.

### 4.2 App events that are *not* engine events

`recording`, `calls`, `calendar`, `extension`, `live_turn`, `final_pass`, `summary`,
`meeting_changed`, `recording_lost`, `auto_stopped`, `engine`, `open_meeting`,
`granola_progress`, `mcp_access` are emitted by the app itself (`lib.rs:196-208`,
`364-427`, `585-610`; `calendar.rs:60-62`). They are out of scope here; only the
`"engine"` channel carries engine events to the web UI.

---

## 5. Concurrency

The app may have several engine calls in flight at once; the transport is built for it:

- Each `Engine::call` gets a unique id and inserts its waiter **before** writing, so a
  reply that races the write is not lost (`engine.rs:106-115`). The process mutex covers
  spawn and write, not the wait (`engine.rs:124-133`).
- Callers that can overlap in the app: the call watcher (`lib.rs:917-932`), the calendar
  thread (`calendar.rs:207-225`), the final-pass thread (`lib.rs:331-338`, `560-595`), the
  summary thread (`lib.rs:589-593`, `605-610`), UI commands such as `sample`
  (`commands.rs:397-410`), and bootstrap during startup (`commands.rs:21-68`).
  `finalize` while recording is explicitly supported (`Engine.swift:206`).
- On the engine side, each request runs in its own detached task
  (`Engine.swift:48-60`), but all commands enter the `Controller` **actor**
  (`Engine.swift:64`). Actor reentrancy means: command bodies do not run in parallel
  inside `Controller`, but a command that `await`s (models, permissions) lets another
  command begin. So `stop` can run while `start` is awaiting permission
  (`Engine.swift:69-72`, `121`), and `pause` can run while `finalize` waits for a model.
- `Speech` is an actor (`Speech.swift:56`): model loading is deduplicated through a single
  in-flight task (`Speech.swift:111-118`), and ASR/VAD calls are serialized. Two final
  passes or a final pass plus a live recording therefore share one model at a time, which
  can make the live pass lag; `LiveSegmenter` buffers up to 2000 chunks and drops the
  oldest beyond that (`Session.swift:23`), and it resets when the sample index jumps more
  than 100 ms (`Session.swift:70-78`).
- Event emission is independent of command handling and may interleave with replies on the
  same stream. The app demultiplexes by `id`, so ordering between events and a reply is
  not meaningful; a second engine must not assume the app has processed an event before
  the reply for a later command.
- The engine must accept commands **before** `ready` has been consumed and after the app
  has stopped reading (shutdown kill), without deadlocking on stdout writes. The Swift
  engine writes synchronously under a lock (`Protocol.swift:15-19`); a slow reader can
  block the engine, but the app's reader always drains stdout until EOF
  (`engine.rs:72-95`).

Rules for a second engine, summarized:

1. Handle every command concurrently or with clearly documented serialization; never
   refuse a command just because another runs, except `start` while a session exists
   (`Engine.swift:185`).
2. Keep per-session state behind a lock/actor: audio, mutes, pause state.
3. Serialize model access or accept queueing; do not load the same model twice.
4. Write each JSON line atomically.

---

## 6. On-disk session layout

The engine owns everything under the meeting's `audio/` folder. The app passes
`dir = <library>/meetings/<meeting id>` (`paths.rs:57-63`; `lib.rs:225-227`) and otherwise
only deletes or sizes that folder (`lib.rs:851-861`; `commands.rs:208`, `578-588`).

### 6.1 Layout

```
<dir>/
  audio/
    session-<start_wall_ms>.json      # informational only (Session.swift:159-163)
    mic/
      00000000.tntc
      00000001.tntc
      ...
    remote/
      00000000.tntc
      ...
```

- Both track directories exist as soon as the session starts
  (`Session.swift:146-150`, `ChunkStore.swift:30-37`). Chunk directories are created with
  intermediate directories, so `dir` need not exist beforehand
  (`ChunkStore.swift:34`); `audio/` is created for the session JSON
  (`Session.swift:160-163`).
- File names are `%08d.tntc`, a per-track counter starting at **the number of entries
  already in that track directory** (`ChunkStore.swift:28`, `35-36`, `83`). The `.part`
  temporary name is used during the atomic write and then renamed (`ChunkStore.swift:84-87`).
- Nothing rotates or compacts the chunks. A track directory is expected to belong to one
  recording; the app refuses to start a second recording of a meeting that already has a
  start time (`lib.rs:239-242`), and `import_audio` is the only other writer.
- Nothing else under a meeting folder is written by the engine.

### 6.2 TNTC chunk format

From `engine/Sources/TintaEngine/ChunkStore.swift:4-13` and `71-88`:

| Offset | Size | Content |
|---|---|---|
| 0 | 4 | ASCII magic `"TNTC"` |
| 4 | 8 | start sample index on the meeting clock, `Int64` little-endian, at 16 kHz |
| 12 | 4 | sample count, `UInt32` little-endian |
| 16 | rest | AES-GCM **combined** box |

- The plaintext is `count` signed 16-bit PCM samples, little-endian, mono, 16 kHz
  (`ChunkStore.swift:55-58`, `73`). `Float` samples are clamped to `[-1, 1]` and scaled by
  `Int16.max` on write (`ChunkStore.swift:56-57`); reading scales back by
  `1/Int16.max` (`ChunkStore.swift:130-135`).
- The AES-GCM key is the 32-byte audio key (see [§7](#7-the-audio-key)). Each chunk is
  sealed separately with a fresh nonce and **no associated data** (`AES.GCM.seal(pcm, using:
  key)`, `ChunkStore.swift:75`; `AES.GCM.open(box, using: key)`, `ChunkStore.swift:116`).
  The 16-byte header is not authenticated; an implementation that passes it as AAD cannot
  read or write Mac-compatible chunks. CryptoKit's
  `SealedBox.combined` — the bytes stored from offset 16 — is nonce (12) ‖ ciphertext ‖
  tag (16), which is what `AES.GCM.SealedBox(combined:)` consumes at
  `ChunkStore.swift:115`.
- A chunk is flushed when its buffer reaches 80 000 samples (5 s), on a gap larger than
  100 ms (1 600 samples), on `pause`, on `stop`, and at the end of an import
  (`ChunkStore.swift:12`, `41-63`, `65-69`; `Session.swift:314`, `322`; `Engine.swift:252`).
  There is no hard cap: a single very large append can make a chunk larger than 5 s.
- Sample offsets are **samples from the meeting start**, not byte offsets and not wall
  clock. The writer snaps a start that drifts by less than or equal to 100 ms to the
  expected next index, and starts a new chunk after a gap of more than 100 ms
  (`ChunkStore.swift:44-52`). A backwards jump of ≤100 ms is therefore silently shifted
  forward (samples are not reordered); a backwards jump of more than 100 ms is also
  snapped forward because the comparison is only `drift > 1600` (`ChunkStore.swift:46-47`).
  The code does not treat a large negative drift specially — do not infer intent from it.
- Files are written atomically: write `<name>.part`, then move to `<name>`
  (`ChunkStore.swift:84-87`). A failed write emits `error` and **drops that chunk**
  (`ChunkStore.swift:89-92`).

### 6.3 Reading and recovery

`ChunkReader` (`ChunkStore.swift:96-138`):

- Lists `audio/<track>`, keeps `*.tntc`, sorts names lexicographically, but **uses each
  chunk's start offset** to place samples; name order is irrelevant.
- A missing track directory is an empty track, not an error (`ChunkStore.swift:104`).
- Each file must be longer than 16 bytes and start with `TNTC`; otherwise the whole read
  fails with `invalid chunk <name>` (`ChunkStore.swift:111-113`). The `count` field is
  written but **ignored** by the reader, which decrypts the rest of the file and treats
  it as `Int16` samples (`ChunkStore.swift:114-119`). A second engine must write the
  header exactly (16 bytes + combined box) but should not trust `count` when reading.
- `track()` builds a `Float` array of length `max(start + decoded sample count)` over all
  chunks and fills them in; all gaps (pauses, dropped audio, missing chunks) are zeros
  (`ChunkStore.swift:125-138`).
- A wrong key or corrupted ciphertext throws a CryptoKit error, which the app shows as the
  command's error. There is no partial recovery.

Crash/recovery behavior in the app:

- If the engine dies mid-recording, the app finalizes the audio written so far
  (`lib.rs:395-401`, `420-427`). Up to one 5 s buffered chunk is lost; a `.part` file may
  remain and is ignored because only `.tntc` is read (`ChunkStore.swift:105-107`).
- On app start, meetings left in `recording` or `processing` are marked stopped and
  re-finalized (`lib.rs:972-987`).
- A final pass is read-only; re-running it is safe and is how retries work
  (`lib.rs:302-305`, `547-558`).

### 6.4 WAV output of `sample`

`ChunkReader.wav` writes a canonical 44-byte PCM header followed by 16-bit little-endian
mono samples: `RIFF`, size `36 + data bytes`, `WAVEfmt `, fmt chunk size 16, format 1
(PCM), 1 channel, 16 000 Hz, byte rate 32 000, block align 2, 16 bits, then `data` and its
size (`ChunkStore.swift:140-163`). The app base64-decodes this into a data URL for playback
(`app/src/Player.tsx:35`; callers `app/src/MeetingView.tsx:726`, `866`).

---

## 7. The audio key

The engine never derives the key. Every audio command receives `audio_key` as a **base64
string of exactly 32 bytes**; anything else fails with `invalid audio key`
(`Engine.swift:152-157`). The app creates the string with standard (padded) base64
(`crates/tinta-core/src/keys.rs:30-33`; `lib.rs:225-227`).

Derivation on the app side (`crates/tinta-core/src/keys.rs:20-28`):

- HKDF with SHA-256.
- Salt: `b"tinta"`.
- IKM: the master key bytes.
- Two independent 32-byte outputs: `info = b"database"` and `info = b"audio"`.
- The audio output is the key the engine gets.

The master key on macOS (`keys.rs:35-66`):

- Stored as a Keychain generic password, service `com.usetinta.tinta`, account
  `library-master-key` (`keys.rs:6-9`).
- Older versions used service `app.tinta`; the app migrates it once (`keys.rs:7-8`,
  `46-54`).
- The stored value is hex text; it is hex-decoded before derivation (`keys.rs:58`, `65`).
- A missing item creates a random 32-byte key (`keys.rs:55-61`). Any other Keychain error
  is fatal on purpose, because a new key would make the library and audio unreadable
  (`keys.rs:74-81`).
- Debug builds can override it with a hex `TINTA_DEV_MASTER_KEY`
  (`keys.rs:40-43`), which is how tests avoid Keychain prompts.

The port must keep the HKDF parameters byte-for-byte if it is to read audio created on a
Mac. Where the Windows/Linux master key lives (DPAPI, Secret Service, a file) is not part
of this protocol; only the derived 32-byte audio key is.

---

## 8. Capture behavior a second engine has to reproduce (Mac reference)

This section describes what the Swift engine does to produce audio and events. A second
engine does not need the same APIs, but it must produce the same file format, event names
and field semantics.

### 8.1 Sample format and clock

- Everything is converted to 16 kHz mono Float32 for processing and Int16 for storage
  (`Capture.swift:21-56`; `ChunkStore.swift:9-12`).
- The meeting clock starts at `RecordingSession.init` (`Session.swift:143-144`). Capture
  callbacks carry a host timestamp; it is converted to an absolute sample index and
  clamped at 0 (`Session.swift:190-193`). Sample indices can therefore be greater than
  the number of samples received (gaps), which is intended.
- The microphone path holds 0.8 s of audio before storing it, to apply mute changes that
  arrive late (`Session.swift:134-137`, `226-244`).

### 8.2 Levels

Every 250 ms the session emits the maximum RMS seen per track since the previous tick
(`Session.swift:195-203`, `290-303`). Microphone level is forced to 0 while muted
(`Session.swift:291-294`). `remote_capturing` is the tap's current capture state
(`Session.swift:300`).

### 8.3 Remote audio capture and the tap helper (Mac-only)

The Swift engine captures other apps through a Core Audio process tap. Because voice
processing on the microphone silences a tap in the same process, the tap runs in a child
process: `tinta-engine --tap-helper <source>`, with raw binary frames on stdout
(`TapHelper.swift:3-8`, `10-41`). The parent turns its state frames into `source_state`
events (`TapHelper.swift:120-125`) and its sample frames into remote chunks. This is an
internal implementation choice: **a second engine must not use this subprotocol**, and the
app never sees it.

Relevant externally visible behavior:

- A source is a bundle ID **prefix** matched against the playing process's bundle ID or
  its responsible app's bundle ID, or `all` for all system audio except the engine
  (`Capture.swift:104-108`, `163-172`, `214-219`).
- Taps are rebuilt when the matching process set, default output device, tap format or
  device rate changes, or when a matching app plays audio but the tap is silent for 2.5 s
  (`Capture.swift:186-188`, `221-258`).
- Microphone capture enables voice processing (echo cancellation) when output is
  speakers; it restarts on configuration changes, silence of 2 s, or route changes
  (`Capture.swift:376-390`, `452-474`).

### 8.4 Live pass

Live transcription is VAD-segmented: a segment starts at probability ≥ 0.5, ends after
0.7 s below 0.35 or 12 s of speech, with 0.25 s of pre-roll padding
(`Session.swift:12-15`, `94-113`). A segment is dropped as echo when it overlaps remote
audio and is quieter than 20 % of it (`Session.swift:48-52`, `280-288`;
`Echo.swift:32-57`). Each accepted segment becomes a `live` event (`Session.swift:53-62`).
Live errors are `error` events, not command errors (`Session.swift:63-65`, `90-93`). The
final pass re-transcribes everything from chunks; live output is only provisional UI
content.

---

## 9. Model install and verification

### 9.1 Pinned inputs

- Three Hugging Face repositories at fixed commits (`Speech.swift:8-13`):

  | Folder | Repository | Revision |
  |---|---|---|
  | `parakeet-tdt-0.6b-v3` | `FluidInference/parakeet-tdt-0.6b-v3-coreml` | `7dd20fe6b1797d35f5e3307e8b1732d9a178edfe` |
  | `silero-vad` | `FluidInference/silero-vad-coreml` | `b419383c55c110e2c9271fa6ee0ea83d03c70d96` |
  | `speaker-diarization` | `FluidInference/speaker-diarization-coreml` | `df2625ac79a7ac6b65ad868fee6d80f320da4232` |

- The Swift package pins FluidAudio itself at revision
  `eabcd9e36dab48f1f7180165396d84b9688650e0` (`engine/Package.swift:10-15`).
- The model root is `MLModelConfigurationUtils.defaultModelsDirectory()` from FluidAudio
  (`Speech.swift:24`), which the manifest script documents as
  `~/Library/Application Support/FluidAudio/Models` (`engine/scripts/model-manifest.py:2`).
  `models_status.path` returns this path (`Engine.swift:79`).
- Files are addressed relative to that root, e.g.
  `parakeet-tdt-0.6b-v3/Encoder.mlmodelc/weights/weight.bin`
  (`ModelManifest.swift:6-58`).

### 9.2 The manifest

`ModelManifest.sha256` maps 53 relative paths to lowercase hex SHA-256 digests:
23 for parakeet, 6 for silero, 24 for diarization (`engine/Sources/TintaEngine/ModelManifest.swift:4-60`).
It is generated from the Hugging Face API at the pinned revisions by
`engine/scripts/model-manifest.py` (header comment at `ModelManifest.swift:1-2`; script
reads the local install and compares against the remote blob hashes, then writes
`model-manifest.json`).

### 9.3 Checks at runtime

- `modelsInstalled()` checks only that every manifest path exists
  (`Speech.swift:64-68`). It is what `models_status.installed` reports, and it is the
  cheap gate before `install_models`/`finalize` do real work.
- `verify()` hashes every manifest file in 4 MiB blocks and compares to the manifest
  (`Speech.swift:71-90`). A missing file fails with
  `"model file missing: <path>. Install the models again."`; a mismatch fails with
  `"model file does not match its pinned hash: <path>. Install the models again."`
  (`Speech.swift:75-87`).
- After a successful verification it writes a `.fluidaudio-revision` file containing the
  pinned revision into each model folder, because FluidAudio re-downloads a folder when
  that marker is missing or different, and other FluidAudio clients share the folder
  (`Speech.swift:33-45`, `88`).
- Verification runs on the first model load and on every diarizer creation, memoized by a
  `verified` flag (`Speech.swift:62`, `71-72`, `111-123`, `137-139`). `install_models`
  clears the flag and verifies again (`Speech.swift:102-103`).

### 9.4 Downloads are blocked outside install

`ModelPins.setOnline(false)` points FluidAudio's registry at `http://127.0.0.1:9`, which
cannot serve models, and installs the revision overrides (`Speech.swift:15-22`). It is set
at startup (`Engine.swift:16`), before every load (`Speech.swift:122`), and before every
diarizer (`Speech.swift:138`). Only `install_models` turns downloads on, and it turns them
off again in a `defer` (`Speech.swift:92-94`).

### 9.5 Install sequence and events

`Speech.install` (`Speech.swift:92-109`):

1. Online registry on.
2. Emit `install_progress` `{component:"transcription", state:"downloading"}`, download
   and load parakeet v3 (`AsrModels.downloadAndLoad(version: .v3)`).
3. Emit the same for `vad`, then create `VadManager()`.
4. Emit the same for `diarization`, then `OfflineDiarizerManager().prepareModels()`.
5. Clear `verified`, run the full SHA-256 verification, write revision markers, load
   ASR/VAD.
6. Emit `install_progress` `{component:"all", state:"done"}`.

Any error aborts the command and is returned as the reply's `error`; the app shows it.

### 9.6 What a second engine must guarantee

- `models_status` is fast and does not load models.
- `install_models` is the only path that writes model files or uses the network.
- Before using models, fail with a sentence that tells the user to install them, rather
  than crashing or timing out. The Swift engine's load fails with "models are not installed"
  (`Speech.swift:121`) and its hash check with "... Install the models again."
  (`Speech.swift:76`, `85`).
- Model format, file names, hashes, and the model root are Mac-specific; a Rust engine may
  choose its own model set and directory, as long as `models_status.path` points at
  something it owns (the app sums its size for Settings, `commands.rs:563-572`).

---

## 10. Porting notes

### 10.1 Capabilities

Every engine reports what it can do in a `capabilities` object, in the `ready` event and in
the `ping` reply:

```json
{"event":"ready","version":"0.9.3","capabilities":{"capture_system_audio":true,"capture_app_audio":true,"calls_detection":true,"call_reader":true,"calendar":true,"summaries":true,"echo_cancellation":true,"import_audio":true}}
```

```json
> {"id":1,"cmd":"ping","params":{}}
< {"id":1,"ok":true,"result":{"pong":true,"capabilities":{…}}}
```

| field | `true` means |
|---|---|
| `capture_system_audio` | The engine records the audio of the whole system (the `all` source). |
| `capture_app_audio` | The engine records the audio of one app, and `list_sources` names the apps. |
| `calls_detection` | The engine detects calls in desktop apps and emits `call_started`, `call_ended`, and `call_state`. |
| `call_reader` | The engine reads names, the active speaker, and the mute state from a call window (`read` in `calls`, `request_accessibility`, `call_report`). |
| `calendar` | The engine reads the calendars (`calendar`, `request_calendar`, `calendar_changed`). |
| `summaries` | The engine writes summaries (`summarize`). |
| `echo_cancellation` | The engine removes the meeting audio from the microphone. The app only changes its hint text; it does not change the audio pipeline. |
| `import_audio` | The engine imports an audio file (`import_audio`). |

Semantics:

- **A missing object means all `true`.** That is the Mac engine from before the field
  existed, so an engine that never sends it keeps working. The Swift engine now sends the
  object with every field `true`.
- A **missing field** inside the object, or a value that is not a boolean, is `true` too.
- The object describes the binary, not the current permission state: `calendar: true`
  still means that `calendar.access` can be `denied`, and `summaries: true` still means
  that `summary_status` can report `available: false`. The app keeps asking those; the
  features that the capability turns off are not asked at all.
- `ready` and `ping` must send the same values. The app reads `ready` on every process
  start and `ping` on every start and restart. A `ping` reply without the object keeps what
  `ready` said.

The app turns a feature off like this (`crates/tinta-core/src/capabilities.rs`,
`app/src-tauri/src/`):

| capability `false` | app behavior |
|---|---|
| `capture_app_audio` | The meeting source is always `all`, and the window shows a fixed "System audio" instead of the per-app picker. This needs `capture_system_audio`. |
| `calls_detection` | No `calls` polling and no call watcher after a restart (the restart uses `ping`), and no call banners or "call detected" hero. |
| `call_reader` | `calls` is sent with `read: false`; `request_accessibility` and `call_report` are not sent; the Zoom and Microsoft Teams names setting is hidden. |
| `calendar` | The calendar thread is not started; `calendar` and `request_calendar` are not sent; the calendar section, the Home cards, and the reminders are hidden. |
| `summaries` | `summary_status` and `summarize` are not sent, no automatic summary runs, and the summary controls are hidden. An existing summary is still shown. |
| `echo_cancellation` | The window does not claim that Tinta removes the echo, and suggests headphones. This only changes the text; the audio pipeline is the engine's. |
| `import_audio` | The import command and button are disabled. |
| `capture_system_audio` | The app never records `all`. With `capture_app_audio` also `false`, no meeting audio can be recorded, so Start is disabled with a reason. |

The two audio capabilities decide whether the app can record at all: with both `false`, there
is no source to pass to `start`, so the recording never starts.

An old app does not know the object and calls the Mac-only commands anyway. A second engine
must therefore still **answer these with valid, degraded results**, not errors, or the old
app will surface errors and retry loops:

| Command | Recommended degraded reply |
|---|---|
| `permissions` | `{"microphone": "granted" \| "denied" \| "undetermined", "accessibility": false}` |
| `request_microphone` | perform the platform permission request; `{"granted": bool}` |
| `calls` | `{"calls": [], "accessibility": false}` (accept and ignore `read`) |
| `request_accessibility` | `{"granted": false}` |
| `call_report` | a clear error sentence |
| `calendar` | `{"access": "denied", "calendars": [], "events": []}` |
| `request_calendar` | `{"granted": false}` |
| `summary_status` | `{"available": false, "reason": "Summaries are not available on this platform."}` |
| `summarize` | a clear error sentence |
| `list_sources` | the platform's recording targets; `route` may always be `"speakers"` |

`calendar.access = "denied"` also keeps reminders inert for an old app, because it only
asks for notification permission when the read reports `granted` (`calendar.rs`, `watch`).
`available:false` from `summary_status` prevents automatic summaries there too
(`lib.rs`, `summaries_available`). A new app never gets this far with a capability of
`false`.

### 10.2 Recording sources and permissions

The app stores the selected source as a string and passes it to `start` unchanged
(`lib.rs:225-257`); it is also persisted as `last_source` and offered back in the UI
(`commands.rs:36`, `156-158`). On macOS it is a bundle ID or `all`. On other platforms it
must be whatever the engine puts in `list_sources[].id`; the app does not interpret it
except to compare it with desktop call app IDs (`lib.rs:243-255`, `457-459`) — a
call-based auto-stop therefore only works if the engine's call app IDs match its source
IDs, as they do on macOS (`CallWatcher.swift:21-24`).

The app maps `permissions.microphone` to its onboarding UI; it asks the OS only through the
engine, via `request_microphone` (`app/src/api.ts:287-289`; `commands.rs:26`, `151-153`).
Audio capture, echo handling, device changes and level metering are entirely the engine's
responsibility.

### 10.3 The helper socket protocol is a different protocol

`crates/tinta-core/src/protocol.rs` documents newline-delimited JSON over the app's Unix
socket for `tinta-mcp` and `tinta-native-host` (`protocol.rs:1-4`; `app/src-tauri/src/socket.rs:1-2`).
Differences from the engine protocol:

- request field is `method`, not `cmd` (`protocol.rs:10-15`);
- responses are `{"id","result"}` or `{"id","error"}` with no `ok` (`protocol.rs:17-24`);
- the first request must be `hello` with `{"client": "mcp" | "native-host"}`
  (`protocol.rs:4`; `socket.rs:74-81`);
- only trusted peer binaries may connect (`socket.rs:37-40`), and lines are capped at
  1 MiB (`socket.rs:15-16`, `59-65`);
- there are no unsolicited events on that channel; the engine protocol has them.

Do not merge the two, and do not reuse `Request`/`Response` for the engine.

### 10.4 Checklist for a second engine

1. Spawn as `tinta-engine`, speak JSON lines on stdin/stdout, logs on stderr, emit
   `ready` first, with the `capabilities` object ([§10.1](#101-capabilities)).
2. Echo numeric ids, reply `{id, ok:true, result}` / `{id, ok:false, error}`.
3. Implement all commands in [§3](#3-commands); at minimum the recording, audio and model
   commands plus the degraded replies above for platform-specific ones.
4. Write `audio/mic` and `audio/remote` as 16 kHz mono `TNTC` chunks with exactly the
   header in [§6.2](#62-tntc-chunk-format), honoring `audio_key`.
5. Emit `levels` at ~4 Hz during recording and the call/calendar/summary events only when
   the feature exists.
6. Keep every command inside its timeout; long work must be resumable/retryable from the
   files on disk, because a timeout does not cancel anything.
7. Exit cleanly on `shutdown` and on stdin EOF, flushing open chunks.

---

## 11. Ambiguities and things the code does not fix

These are real gaps; a second engine should pick a reasonable behavior and document it,
and the app may need a change later.

1. **Capabilities are trusted, not checked.** The app believes the object; an engine that
   reports a feature it does not implement makes the app call that command anyway. See
   [§10.1](#101-capabilities).
2. **No engine binary override.** The path is fixed to the sidecar/development location
   (`engine.rs:33-38`). The port needs either a rename/install step or an app change.
3. **`ping` is not a heartbeat.** The app sends it on every start and restart, but not
   while the engine runs, so a process that hangs without closing stdout is not detected.
   It is still the obvious place to add a future health check.
4. **Strings vs numbers for `id`.** The protocol echoes whatever `id` arrives, but the app
   only accepts `u64` (`engine.rs:76`). A second engine interoperates only with numeric
   ids, even though nothing rejects string ids on the wire.
5. **Ordering of `ready` vs commands.** The app may write before reading `ready`
   (`engine.rs:72-123`), so the engine must process commands that arrive before its
   start-up work finishes. The Swift engine's reader starts after `ready` is queued
   (`Engine.swift:18`, `22`), so in practice `ready` is first on the stream, but no code
   enforces that a second engine do the same.
6. **Restart is not a supervisor.** Restarts happen because a later command spawns a new
   process; only the call watcher delays using backoff (`lib.rs:400`, `917-932`,
   `429-436`). A crash loop triggered by UI commands is not rate-limited.
7. **`install_models` and concurrent model use share global state.** `ModelPins.setOnline`
   mutates static FluidAudio registry state (`Speech.swift:19-22`) while other commands
   may be loading models through the `Speech` actor (`Speech.swift:92-94`, `111-118`).
   Behavior under that overlap is not defined by the code; the app does not coordinate
   these calls, and nothing in the engine guards them.
8. **Interleaving of a final pass with live transcription.** Explicitly allowed
   (`Engine.swift:206`), but only "one window at a time" is promised because `Speech` is an
   actor (`Speech.swift:56`). There is no fairness guarantee; live transcription can be
   starved behind a long final pass.
9. **`finalize` result for a partially written chunk.** The reader ignores the stored
   sample count and treats the remaining bytes as samples (`ChunkStore.swift:114-119`). If
   the plaintext length is not a multiple of 2, `bindMemory(to: Int16.self)` is
   out-of-bounds in principle; the code has no check. A writer must always produce an even
   number of data bytes (it does: Int16 samples).
10. **Chunk file naming on reuse.** The counter starts at the number of existing entries
    (`ChunkStore.swift:35-36`), not at the next free number. Importing or re-recording
    into a folder that already contains files can therefore collide; the atomic move then
    fails and the chunk is dropped with an `error` event (`ChunkStore.swift:84-92`).
11. **Session JSON is unread.** `audio/session-<start_wall_ms>.json`
    (`Session.swift:159-163`) has no consumer in the repository, so its schema is
    descriptive, not a contract.
12. **`calendar` parsing errors are fatal to the command.** The result must contain
    `access`; each calendar needs `id` and `title`; each event needs `id`, `title`,
    `start`, `end`, and `calendar_id` (`tinta-core/src/calendar.rs:12-71`, `147-167`).
    The Swift engine always provides them, but a second engine returning partial objects
    will break the whole refresh.
13. **`source_state` has no consumer.** It is emitted (`TapHelper.swift:120-125`) and
    forwarded (`lib.rs:416`), but no Rust or TypeScript code reads it. Its fields are
    therefore not relied upon today.
14. **The `ready` version is not compared.** `AGENTS.md:15` requires the engine version to
    equal the app version, but the app only forwards the event; a mismatch is invisible.

---

## 12. Quick reference index

| Topic | Location |
|---|---|
| App-side transport | `app/src-tauri/src/engine.rs:1-152` |
| Capabilities | `crates/tinta-core/src/capabilities.rs`; `app/src-tauri/src/lib.rs` (`start_engine`, `watch_calls`) |
| Engine-side transport | `engine/Sources/TintaEngine/Protocol.swift:1-67` |
| Command switch | `engine/Sources/TintaEngine/Engine.swift:74-138` |
| Command dispatch/main loop | `engine/Sources/TintaEngine/Engine.swift:10-61` |
| Recording session | `engine/Sources/TintaEngine/Session.swift:119-325` |
| Chunk format | `engine/Sources/TintaEngine/ChunkStore.swift:4-164` |
| Model pins and verification | `engine/Sources/TintaEngine/Speech.swift:8-151` |
| Model manifest | `engine/Sources/TintaEngine/ModelManifest.swift:4-60` |
| Calendar (EventKit) | `engine/Sources/TintaEngine/Calendar.swift:1-137` |
| Calls (Core Audio + AX) | `engine/Sources/TintaEngine/CallWatcher.swift:1-113`, `CallReader.swift:1-274` |
| Summaries (FoundationModels) | `engine/Sources/TintaEngine/Summary.swift:1-194` |
| Audio capture (Mac) | `engine/Sources/TintaEngine/Capture.swift:1-526`, `TapHelper.swift:1-150` |
| Key derivation | `crates/tinta-core/src/keys.rs:1-82` |
| Final-result consumer | `crates/tinta-core/src/transcript.rs:8-30` |
| Calendar result consumer | `crates/tinta-core/src/calendar.rs:147-178` |
| Helper socket protocol (not this one) | `crates/tinta-core/src/protocol.rs:1-34`, `app/src-tauri/src/socket.rs:1-126` |
