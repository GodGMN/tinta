# Building Tinta on Linux

The Tauri app builds on Linux. `tinta-core`, `tinta-native-host` and `tinta-mcp` also build
and test there without a display. The Keychain path in `crates/tinta-core/src/keys.rs` is
only compiled on macOS, so the encrypted library does not open on Linux yet.

## Dependencies (Debian and Ubuntu)

SQLCipher is bundled, but it compiles against the system OpenSSL, so the OpenSSL
development headers and a C compiler are required:

```sh
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev
```

Without them the build stops with `fatal error: openssl/crypto.h: No such file or
directory` while compiling `libsqlite3-sys`.

The app also needs the WebKitGTK development headers:

```sh
sudo apt-get install -y libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev
```

## Verify

```sh
cargo test --locked -p tinta-core
cargo clippy --locked -p tinta-core --all-targets -- -D warnings
cargo test --locked -p tinta-native-host -p tinta-mcp
cargo clippy --locked -p tinta-native-host -p tinta-mcp --all-targets -- -D warnings
```

The `core-linux` and `helpers-linux` jobs in `.github/workflows/ci.yml` run these commands
on `ubuntu-latest`. The `app-linux` job runs clippy for the app crate with the WebKitGTK
headers installed.

## Speaker names and MCP tools

On Linux the Meet extension and the MCP server reach the app the same way as on macOS:

- `tinta-native-host` is the Chrome native messaging host `com.usetinta.tinta`. The app
  writes the host manifest on each start, with the absolute path of the helper, into the
  `NativeMessagingHosts` folder of every Chromium browser that has a profile folder:
  `~/.config/google-chrome`, `~/.config/chromium` and
  `~/.config/BraveSoftware/Brave-Browser`. When no browser ran yet, it registers with
  Chrome, so a Chrome that starts later finds the manifest.
- `tinta-mcp` is the MCP server over stdio. Add the path that the app shows in Settings.
- Both helpers, and the app, use `tinta_core::paths::socket_path()`.

The socket lives in a folder of its own, `$XDG_RUNTIME_DIR/tinta/tinta.sock`, so that only
this user can enter the folder. The app creates the folder with mode 0700 (and fixes the
mode on each start) and the socket with mode 0600. When `XDG_RUNTIME_DIR` is unset or
relative, the socket stays in the data folder as on macOS. `TINTA_DATA_DIR` (used by tests
and the self-test) always keeps the socket in that data folder. macOS keeps its existing
path.

## Peer authentication

macOS checks the code signature of the helper with `codesign`. Linux has no equivalent that
works for an unsigned development build, so the app checks two things with
`crates/tinta-core/src/peer.rs`:

1. `SO_PEERCRED` on the accepted socket gives the user ID and the process ID of the peer.
   The user ID must be the effective user ID of the app. The kernel fills this value, so a
   process cannot claim it.
2. `/proc/<pid>/exe` gives the executable of that process. The kernel resolves it, and it
   keeps working when the helper replaces its file, so the path cannot be spoofed from the
   peer side. Both this path and the allowed paths are canonicalized, and the peer must
   match one of `tinta-native-host` and `tinta-mcp` next to the app executable. In
   development that folder is the Cargo target directory.

Every error refuses the connection: the process is gone, `/proc` is hidden by a sandbox,
`readlink` is denied, or a path does not resolve. The Windows build has no app socket yet
and stays fail closed; named pipes come with the Windows work.

### Why this is weaker than the macOS check

The Linux check trusts the path of the binary. A process that runs as the same user and can
replace `tinta-native-host` or `tinta-mcp` in the install folder is accepted; the check
would not notice the different file. macOS detects that with the signature.

This is acceptable because the check does not protect a privilege boundary:

- Another user cannot connect: the folder is 0700, the socket is 0600, and the user ID must
  match.
- A web page or another browser extension cannot start the native host. Chrome starts it
  only for the extension ID in the manifest's `allowed_origins`.
- Replacing the installed helper needs write access to the install folder, which belongs to
  root. A path that matches a development build is the developer's own build output.
- A process that already runs as this user can read the meeting library and the app's
  settings directly, so the socket gives it nothing that it could not already reach.

## Not on Linux yet

- The encrypted library does not open, so the app shows the reason and closes (see the
  `## Unreleased` changelog entry).
- Meeting reminders use the macOS Notification Center.
- The recording bubble, Mission Control, Time Machine exclusion, the clipboard file copy,
  and the system privacy settings panes are macOS features.
