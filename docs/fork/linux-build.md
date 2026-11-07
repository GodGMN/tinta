# Building the core crate on Linux

`tinta-core` builds and tests on Linux. The Tauri app (`app/src-tauri`) still uses
macOS-only code, and the Keychain path in `crates/tinta-core/src/keys.rs` is only
compiled on macOS, so a Linux key store does not exist yet.

## Dependencies (Debian and Ubuntu)

SQLCipher is bundled, but it compiles against the system OpenSSL, so the OpenSSL
development headers and a C compiler are required:

```sh
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev
```

Without them the build stops with `fatal error: openssl/crypto.h: No such file or
directory` while compiling `libsqlite3-sys`.

## Verify

```sh
cargo test --locked -p tinta-core
cargo clippy --locked -p tinta-core --all-targets -- -D warnings
```

The `core-linux` job in `.github/workflows/ci.yml` runs the same commands on
`ubuntu-latest`.

## The engine crate on Linux, Windows and macOS

`crates/tinta-engine-rs` does not depend on `tinta-core`, so the bundled
SQLCipher build and the OpenSSL headers above do not apply to it. Its
dependencies are pure Rust plus the prebuilt sherpa-onnx native libraries that
`sherpa-onnx-sys` downloads in its build script. The `engine-windows`
(`windows-latest`) and `engine-macos` (`macos-15-intel`) jobs in
`.github/workflows/ci.yml` build and test only `-p tinta-engine-rs`, with no
extra system packages:

```sh
cargo test --locked -p tinta-engine-rs
cargo clippy --locked -p tinta-engine-rs --all-targets -- -D warnings
```

On Linux the same two commands run in `core-linux` after the `tinta-core` steps.

## The Linux bundles

`scripts/build-linux.sh` builds the release engine and helpers, copies them into
`app/src-tauri/binaries/` with the target triple suffix that Tauri expects, and runs
`pnpm tauri build`. Tauri reads `app/src-tauri/tauri.linux.conf.json` on top of
`tauri.conf.json` on Linux and leaves `scripts/build.sh` and the macOS config alone.

- The bundle targets are the `.deb` and the AppImage. The `bundle-linux` job in
  `.github/workflows/ci.yml` builds them and uploads both as an artifact.
- Tauri installs a sidecar as `usr/bin/<name>`, without the triple suffix, which is the
  folder `engine_path()` and `helper_path()` read from `current_exe`. The app finds the
  engine in the `.deb` and inside the AppImage mount alike.
- `app/src-tauri/linux.desktop` replaces Tauri's desktop entry template so that the
  entry has the two freedesktop categories `Office` and `AudioVideo`. Tauri's
  `bundle.category` maps to one freedesktop category.
- The job runs on Ubuntu 22.04 for its glibc 2.35, so the bundles start on Debian 12
  (glibc 2.36) and Ubuntu 24.04 (glibc 2.39). `ubuntu-latest` is 24.04 and would tie the
  bundles to glibc 2.39.
- Tauri adds `libwebkit2gtk-4.1-0` and `libgtk-3-0` to the `Depends` of every Linux
  package. The config adds `libdbus-1-3`, `libpulse0`, `libssl3` and `libstdc++6`: the
  app links `libdbus` through `tao`, the bundled SQLCipher links the system `libcrypto`,
  and sherpa-onnx links `libstdc++`. The Secret Service client talks D-Bus directly with a
  vendored libdbus, so the package needs no libsecret, and it recommends
  `gnome-keyring | kwalletmanager` for the service behind `org.freedesktop.secrets`.
- `dpkg-deb -c` and `-I` print the contents and the dependencies, and
  `<appimage> --appimage-extract` lists `usr/bin` of the AppImage. The CI job checks both
  and fails when a sidecar is missing or a library is unresolved.

