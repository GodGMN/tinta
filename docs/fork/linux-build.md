# Building Tinta on Linux

This document covers the packaging layer of Tinta's Linux port: the `.deb` and the
AppImage. It does not describe the rest of the port, which lives in other branches.

**The Linux preview needs the whole Linux stack:** the key store
([#6](https://github.com/GodGMN/tinta/pull/6)), the recording engine
([#10](https://github.com/GodGMN/tinta/pull/10)), the Chrome native host and MCP helpers
([#14](https://github.com/GodGMN/tinta/pull/14)), and call detection and capabilities
([#12](https://github.com/GodGMN/tinta/pull/12)). **A bundle built from this branch alone
installs and opens, but it cannot open a library or record a meeting.** On this branch
`load_keys()` always fails off macOS, and the engine answers `start` with `unsupported
command`.

## What the full port does

- Records the monitor of the default output device with PipeWire or PulseAudio and writes
  the same TNTC audio as the Mac app.
- Transcribes and separates speakers on the machine with the bundled sherpa-onnx engine.
- Opens the encrypted library with a key in the Secret Service (`org.freedesktop.secrets`):
  GNOME Keyring, KWallet, or another provider.
- Names Google Meet speakers through the Chrome extension, its native host, and the MCP
  server that the bundles ship.
- Detects calls and reads capabilities, so Home, recording, and the extension follow the
  same flow as on the Mac.

The full port still has these gaps, which the README lists: the calendar and **Next
meetings**, summaries (no local model), names from the Zoom and Microsoft Teams windows,
and per-app audio (it records the whole system output).

## Dependencies (Debian and Ubuntu)

Building `tinta-core` needs the system OpenSSL headers because SQLCipher is bundled but
compiles against OpenSSL, and it needs a C compiler:

```sh
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev
```

Without them the build stops with `fatal error: openssl/crypto.h: No such file or
directory` while compiling `libsqlite3-sys`.

The app bundle itself needs the WebKitGTK and GTK development packages; the `bundle-linux`
job installs `libwebkit2gtk-4.1-dev`, `libayatana-appindicator3-dev`, `librsvg2-dev`,
`libxdo-dev`, and `patchelf`.

## Verify

```sh
cargo test --locked -p tinta-core
cargo clippy --locked -p tinta-core --all-targets -- -D warnings
```

The `core-linux` job in `.github/workflows/ci.yml` runs the same commands on
`ubuntu-latest`.

## The engine crate on Linux, Windows and macOS

`crates/tinta-engine-rs` does not depend on `tinta-core`, so the bundled SQLCipher build
and the OpenSSL headers above do not apply to it. Its dependencies are pure Rust plus the
prebuilt sherpa-onnx native libraries that `sherpa-onnx-sys` downloads in its build
script. The `engine-windows` (`windows-latest`) and `engine-macos` (`macos-15-intel`) jobs
in `.github/workflows/ci.yml` build and test only `-p tinta-engine-rs`, with no extra
system packages:

```sh
cargo test --locked -p tinta-engine-rs
cargo clippy --locked -p tinta-engine-rs --all-targets -- -D warnings
```

On Linux the same two commands run in `core-linux` after the `tinta-core` steps.

## The Linux bundles

`scripts/build-linux.sh` builds the release engine and helpers with `--locked`, copies
them into `app/src-tauri/binaries/` with the target triple suffix that Tauri expects, and
runs `pnpm tauri build`. The script installs the frontend dependencies and lets
`beforeBuildCommand: "pnpm build"` build the frontend once, so the frontend is not built
twice. Tauri reads `app/src-tauri/tauri.linux.conf.json` on top of `tauri.conf.json` on
Linux and leaves `scripts/build.sh` and the macOS config alone.

- The bundle targets are the `.deb` and the AppImage. The `bundle-linux` job in
  `.github/workflows/ci.yml` builds them and uploads both as the `tinta-linux-x86_64`
  artifact. The upload sets `retention-days: 30`, so the artifact expires after 30 days;
  GitHub also needs a login to download it.
- Tauri installs a sidecar as `usr/bin/<name>`, without the triple suffix, which is the
  folder `engine_path()` and `helper_path()` read from `current_exe`. The app finds the
  engine in the `.deb` and inside the AppImage mount alike.
- `app/src-tauri/linux.desktop` replaces Tauri's desktop entry template so that the entry
  has one main freedesktop category, `AudioVideo`, and the additional `Audio` and
  `Recorder` categories: `Categories=AudioVideo;Audio;Recorder;`. Tauri's `bundle.category`
  maps to one freedesktop category, and two main categories make the app appear twice in
  category menus and make `desktop-file-validate` print a hint.
- The job runs on Ubuntu 22.04 for its glibc 2.35, so the bundles start on Debian 12
  (glibc 2.36) and Ubuntu 24.04 (glibc 2.39). `ubuntu-latest` is 24.04 and would tie the
  bundles to glibc 2.39.
- Tauri adds `libwebkit2gtk-4.1-0` and `libgtk-3-0` to the `Depends` of every Linux
  package. The config adds `libdbus-1-3`, `libpulse0`, `libssl3 | libssl3t64` and
  `libstdc++6`. The app links `libdbus` through `tao`, and the bundled SQLCipher links the
  system `libcrypto`; sherpa-onnx links `libstdc++`. `libpulse0` is what the **recording
  engine from #10** needs: `libpulse-binding` and `libpulse-simple-binding` load
  `libpulse.so.0` and `libpulse-simple.so.0` at runtime, and both libraries ship in
  `libpulse0` on Debian and Ubuntu. No binary in this branch needs it yet, which is why it
  stays until #10 lands; `tauri.linux.conf.json` carries that note as a comment next to
  the dependency (`tauri-build` enables its `config-json5` feature, so the file may carry
  comments).
- This branch has no Secret Service client yet; `libdbus-1.so.3` is NEEDED only through
  `tao`. When #6 lands, its client talks D-Bus directly with a vendored libdbus, so the
  package still needs no libsecret. The package recommends the service behind
  `org.freedesktop.secrets` as `gnome-keyring | libkf5wallet-bin`. `libkf5wallet-bin`
  contains `kwalletd5`, the Secret Service daemon on Debian 12 and Ubuntu 24.04. The
  KDE Frameworks 6 package `kwalletd6` does not exist in either (it arrives in Debian 13
  and Ubuntu 24.10), so naming it here would add an alternative that no supported release
  can satisfy. `kwalletmanager` is only the KWallet GUI and is not a Secret Service
  provider.
- `dpkg-deb -c` and `-I` print the contents and the dependencies, and
  `<appimage> --appimage-extract` lists `usr/bin` of the AppImage. The CI job checks that
  every sidecar is present in both bundles.
- The CI job also installs the `.deb` in clean `debian:12` and `ubuntu:24.04`
  containers with `apt-get install -y`, so the entire `Depends` list must resolve there,
  and runs `ldd /usr/bin/tinta*` inside each container. The old `ldd` run on the build
  machine could not catch a missing library, because the build image has every `-dev`
  package installed.

## AppImage helper paths

`helper_path()` returns a sibling of `current_exe()`. In a `.deb` install that is
`/usr/bin/tinta-*`, which is stable. Inside an AppImage it is
`/tmp/.mount_XXXXXX/usr/bin`, which changes on every launch and disappears when the app
quits. PR #14 writes that path into the Chrome native host manifests and shows it as the
MCP command, so a manifest or MCP configuration that points into an AppImage mount works
only while that instance runs.

TODO (#14): when `$APPIMAGE` is set, copy `tinta-engine`, `tinta-mcp`, and
`tinta-native-host` once to `$XDG_DATA_HOME/Tinta/bin/<version>/` (default
`~/.local/share/Tinta/bin/<version>/`), mode 0755, and return those paths from
`helper_path()` and `engine_path()`. The packaging layer cannot do this on its own: the
lookup lives in `app/src-tauri/src/platform/linux.rs`, which #14 owns. Until then, document
that the Chrome extension and MCP need the `.deb`, as the README does.
