# The Linux integration branch

`integration/linux` is not a port PR. It is one branch that contains the whole
Linux stack, so a `.deb` can be downloaded and tested on a real laptop before
the stacked PRs are merged one by one. It is built by merging the port branches
in dependency order and is never review-merged itself.

- Port PRs: [#1 `feat/3/core-linux`](https://github.com/GodGMN/tinta/pull/1),
  [#2 `docs/4/engine-protocol`](https://github.com/GodGMN/tinta/pull/2),
  [#5 `feat/8/tntc-rust`](https://github.com/GodGMN/tinta/pull/5),
  [#6 `feat/10/linux-keys`](https://github.com/GodGMN/tinta/pull/6),
  [#8 `feat/5/app-linux`](https://github.com/GodGMN/tinta/pull/8),
  [#12 `feat/6/capabilities`](https://github.com/GodGMN/tinta/pull/12),
  [#14 `feat/13/linux-helpers`](https://github.com/GodGMN/tinta/pull/14),
  [#4 `feat/7/engine-skeleton`](https://github.com/GodGMN/tinta/pull/4),
  [#7 `feat/9/engine-finalize`](https://github.com/GodGMN/tinta/pull/7),
  [#15 `feat/20/recluster`](https://github.com/GodGMN/tinta/pull/15),
  [#10 `feat/11/linux-recording`](https://github.com/GodGMN/tinta/pull/10),
  [#13 `feat/12/linux-packaging`](https://github.com/GodGMN/tinta/pull/13).
  The `spike/*` branches are not part of it.

## Rebuild

```sh
cd /home/sergio/code/worktrees/tinta-fork/22-integration
scripts/fork/build-integration.sh              # reset, merge, print heads, push
scripts/fork/build-integration.sh --no-push    # merge only
scripts/fork/build-integration.sh --resume     # continue after a manual conflict fix
```

The script resets the branch to `origin/main` and merges each branch with
`git merge --no-ff --no-edit origin/<branch>` in this dependency order:

1. `feat/3/core-linux`
2. `docs/4/engine-protocol`
3. `feat/8/tntc-rust`
4. `feat/10/linux-keys`
5. `feat/5/app-linux`
6. `feat/6/capabilities`
7. `feat/13/linux-helpers`
8. `feat/7/engine-skeleton`
9. `feat/9/engine-finalize`
10. `feat/20/recluster`
11. `feat/11/linux-recording`
12. `feat/12/linux-packaging`

It prints the head of every merged branch and refuses to finish unless each one
is an ancestor of the result. The only ref it pushes is `integration/linux`, and
only with `--force-with-lease`.

`git rerere` is enabled by the script. A manual resolution is remembered in the
local rerere cache, so the same conflict resolves itself the next time. The
cache is local (`.git/rr-cache`) and is not committed: the branch revisions may
change while other agents work, and a stale rerere postimage could silently
restore old content. The resolutions below are the source of truth.

## Conflicts the current branches produce

The script resolves these two by itself:

- **`Cargo.lock`** (on almost every merge). It takes the current side and runs
  `cargo metadata`, which adds the packages that the merged manifests need and
  keeps existing versions. It never runs `cargo update --workspace`. The final
  `cargo test --locked` proves the lock matches.
- **`CHANGELOG.md` `## Unreleased`** (when two branches both edited it). It
  unions the bullets of both sides per section (`Added`, `Changed`, `Fixed`,
  `Removed`), keeps the current side's order first, and drops exact duplicates.

These need judgement and are resolved by hand:

- **`feat/10`: `Cargo.lock`** — the branch adds the Secret Service stack; the
  union from `cargo metadata` is correct.
- **`feat/6`: `crates/tinta-core/src/lib.rs`** — keep both module declarations:
  `pub mod chunks;` from #5 and `pub mod capabilities;` from #12.
- **`feat/13`: `crates/tinta-core/Cargo.toml`** — #6 and #14 both add a
  `[target.'cfg(target_os = "linux")'.dependencies]` table. A table cannot
  appear twice, so combine them into one with `keyring` (the #6 Secret Service
  client) and `libc` (the #14 peer check).
- **`feat/13`: `docs/fork/linux-build.md`** — keep #6's key-store section and
  #14's socket, peer-check and browser sections in the same document instead of
  taking one side.
- **`feat/7`: `docs/fork/linux-build.md`** — keep the #14 sections and add the
  engine-crate section from #4.
- **`feat/9`: `crates/tinta-core/src/lib.rs`** — #7 adds `#[cfg(feature =
  "database")]` to `db`, `export`, `granola`, `tools` and `transcript` (the Rust
  engine builds `tinta-core` without SQLCipher). Keep those attributes and
  `pub mod capabilities;`, plus `pub mod chunks;` and the Linux `peer` module.
- **`feat/11`: `crates/tinta-engine-rs/src/speech.rs`** — #10 changes the pass
  to keep a shared `Arc<OfflineRecognizer>` and makes `decode` `pub(crate)` for
  the live draft; #15 adds the `notes` field for the 4-hour diarization cap.
  Keep the `Arc` recognizer, the `notes` field and the #15 embedding helpers,
  with the `pub(crate) fn decode` signature from #10.
- **`feat/12`: `docs/fork/linux-build.md`** — the packaging branch rewrites the
  document for the whole port. Keep its bundle and AppImage sections, and keep
  the key-store, socket, peer-check and engine sections resolved above.

When a conflict is not in this list, stop: `git merge` left a mid-merge state,
resolve it, `git add` the files, `git commit --no-edit`, then rerun the script
with `--resume`.

## Integration-only changes

These commits live on `integration/linux` only, because they need code from
both sides of a PR boundary:

- **Stable AppImage helper paths** (`app/src-tauri/src/platform/linux.rs`,
  `app/src-tauri/src/platform/other.rs`): when `$APPIMAGE` is set,
  `helper_path()` copies `tinta-mcp` and `tinta-native-host` once to
  `$XDG_DATA_HOME/Tinta/bin/<version>/` (default
  `~/.local/share/Tinta/bin/<version>/`), mode 0755, and returns those paths.
  The Chrome native host manifest and the MCP path in Settings therefore point
  at a file that survives the AppImage mount. `allowed_helpers()` adds the same
  folder to the peer allow-list, because the running helper is that copy. The
  engine is not copied: the app starts it itself from the mount.
  This is the `TODO (#14)` in `docs/fork/linux-build.md`; it needs the helper
  registration from #14 and the packaging work from #13, so it should be
  upstreamed together with them.
- **`integration-smoke` CI job** (`.github/workflows/ci.yml`) and
  `scripts/fork/engine-smoke.py` / `scripts/fork/integration-session.sh`: the
  end-to-end check of the installed `.deb`.
- **The ignored keyring test** in `crates/tinta-core/src/keys.rs`
  (`creates_a_library_key_through_the_secret_service`): it needs the CI session,
  so it is ignored everywhere else.
- **`integration/linux` on push** in the workflow, so every push builds the
  bundle and runs the smoke test.

## What the smoke test proves

`integration-smoke` runs only on `integration/linux` (and on its PRs, if one is
ever opened). It downloads the `tinta-linux-x86_64` artifact produced by
`bundle-linux`, installs the `.deb` in a clean `ubuntu:24.04` container, and
inside a `dbus-run-session`:

1. Starts `gnome-keyring-daemon --unlock --components=secrets` with an empty
   password, so a real Secret Service is on the session bus.
2. Runs the packaged `/usr/bin/tinta-engine` over the JSON-lines protocol:
   `ping`, `install_models` (the pinned ~600 MB set, cached with
   `actions/cache` on the manifest hash), `import_audio` of a short speech clip
   generated with `espeak-ng` + `ffmpeg`, and `finalize`. It asserts that the
   remote track has words and at least one speaker label.
3. Builds and runs the ignored `tinta-core` test that creates the library key
   through that Secret Service item, reads it back, and removes it.

It proves that the packaged binaries start on a clean supported Ubuntu, that
the model download and verification work, that the audio import and the whole
final pass work end to end, and that the Secret Service path creates and reuses
a key. It cannot prove anything that needs a display, a GPU, audio devices, or a
real Google Meet call; the bundle contents and `Depends` are covered separately
by `bundle-linux`, and the click-through checks are below.

## Download the artifact

Every push to `integration/linux` uploads the `.deb` and the AppImage as the
`tinta-linux-x86_64` artifact, kept for 30 days. GitHub needs a login to
download it.

```sh
# The newest run of the branch and its id.
gh run list -R GodGMN/tinta --branch integration/linux --limit 1

# Download both bundles into ./tinta-linux (deb/ and appimage/).
gh run download -R GodGMN/tinta -n tinta-linux-x86_64 -D tinta-linux

sudo apt-get install -y ./tinta-linux/deb/*.deb
```

Without `gh`, open the run on GitHub, scroll to **Artifacts**, and download
`tinta-linux-x86_64`. The `.deb` is the recommended way to test: the Chrome
extension and an MCP client need a stable helper path, which the `.deb` has
under `/usr/bin` (the AppImage uses the copied helpers described above).
The bundle recommends `gnome-keyring | libkf5wallet-bin`; on Debian 12 and
Ubuntu 24.04 that is `gnome-keyring` or KWallet 5.

## Manual test checklist: recording (from #10)

From `docs/fork/linux-recording.md`, on a real Linux laptop:

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
   transcribed a second time from the microphone track. The video itself is in
   the remote track and is transcribed: a video played into the call is part of
   the meeting.
7. On a machine with no microphone (VM, headless, monitor-only default source),
   start a recording. One warning says no microphone was found, the microphone
   meter stays at 0, and the finished meeting has no microphone track; nothing
   from the speakers is attributed to the user.
8. `kill -9` the engine mid-recording, then restart Tinta. The meeting is
   finalized from the chunks already on disk (at most ~5 s lost).
9. Record 30+ minutes with speech playing and check RSS stays flat in
   `ps`/System Monitor; the live transcript keeps up or falls behind without
   growing memory.

## Manual test checklist: extension, MCP and socket (from #14)

1. From a desktop session, start Tinta and confirm the socket folder and file
   modes: `stat -c '%a %n' "$XDG_RUNTIME_DIR/tinta" "$XDG_RUNTIME_DIR/tinta/tinta.sock"`
   → `700` and `600`.
2. Confirm `~/.config/google-chrome/NativeMessagingHosts/com.usetinta.tinta.json`
   exists (and the same under Chromium/Brave if those profiles exist) with the
   absolute `tinta-native-host` path.
3. In Settings, click **Show extension folder**: the "Chrome extension" folder
   opens in the file manager.
4. Load the extension unpacked in Chrome (`chrome://extensions`, Developer
   mode, **Load unpacked**, select the folder), open the popup and confirm it
   shows the app as connected.
5. Join a Google Meet call, speak in turn, and confirm Tinta suggests the
   participant names for speakers.
6. Configure an MCP client with the `tinta-mcp` path shown in Settings and
   confirm a tool call (e.g. list meetings) works, and that MCP off in Settings
   is refused.
7. Start a second copy of the app or a random program and confirm it cannot use
   the socket (connection refused).

Snap and Flatpak browsers are not supported: their sandboxes cannot start a host
binary and they read native messaging manifests from their own folders. Use
Chrome or Chromium from the distribution's deb/rpm packages.

## Local verification

```sh
export OPENSSL_DIR=/path/to/openssl
cargo test --locked -j2 -p tinta-core
cargo test --locked -j2 -p tinta-engine-rs
cargo test --locked -j2 -p tinta-native-host -p tinta-mcp
cargo clippy --locked -j2 -p tinta-core -p tinta-engine-rs \
  -p tinta-native-host -p tinta-mcp --all-targets -- -D warnings
```

The app cannot be built without the WebKitGTK headers; the `app-linux`,
`bundle-linux` and `integration-smoke` jobs build and check it in CI.
