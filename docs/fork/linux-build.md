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
