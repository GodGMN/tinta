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
