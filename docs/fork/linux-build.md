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
this user can enter the folder. The app creates the folder with mode 0700 (without a
following chmod) and the socket with mode 0600. It then checks, with `lstat`, that the
folder is a real folder (not a symbolic link), belongs to this user and still has mode
0700; anything else stops the app with a clear error. When `XDG_RUNTIME_DIR` is unset or
relative, the app and both helpers use `/run/user/<uid>/tinta` when that folder exists and
belongs to the user, and otherwise a private `run` subfolder of the data folder
(`~/.local/share/Tinta/run`, created with mode 0700 and checked the same way; the data
folder itself stays 0755 for the library). An `XDG_RUNTIME_DIR` that does not belong to
this user is ignored. `TINTA_DATA_DIR` (used by tests and the self-test) keeps the socket in
`<TINTA_DATA_DIR>/run`. All three processes call the one
`tinta_core::paths::socket_path()`, so a helper started without `XDG_RUNTIME_DIR` (ssh,
cron, a system unit) still finds the socket that the desktop app listens on. macOS keeps
its existing path.

## Peer authentication

macOS checks the code signature of the helper with `codesign`. Linux has no equivalent that
works for an unsigned development build, so the app checks with
`crates/tinta-core/src/peer.rs`:

1. `SO_PEERCRED` on the accepted socket gives the user ID and the process ID that the
   kernel recorded at `connect()`. The user ID must be the effective user ID of the app, so
   another user cannot connect.
2. `/proc/<pid>/exe` is the kernel's link to the file that the process runs. The app stats
   that link, which follows it to the running inode, and compares the file identity (device
   and inode) with the files of `tinta-native-host` and `tinta-mcp` next to the app
   executable. In development that folder is the Cargo target directory.
3. When the kernel supports `SO_PEERPIDFD` (Linux 6.5 and later), the app pins the
   connecting process with the returned pidfd and checks after reading the executable that
   the process is still alive, so a reused pid cannot answer the check. On older kernels it
   falls back to the pid from `SO_PEERCRED`.

A helper whose file an update replaced reads `/opt/.../tinta-mcp (deleted)`, and the app
refuses it. That is the correct fail-closed behaviour: the running old helper gets
"Refused" until the browser or the MCP client starts it again from the new file.

Every error refuses the connection: the process is gone, `/proc` is hidden by a sandbox, a
read is denied, or the identity does not match a helper file. The Windows build has no app
socket yet and stays fail closed; named pipes come with the Windows work.

### What the check covers, and what it does not

The check answers one question: is the process that connected running the helper file that
this app installed next to itself? It is not a sandbox and it does not verify who built
that file.

- Covered: another user cannot connect (0700 folder, 0600 socket, and the user ID from the
  kernel must match). A random program that runs as this user is refused because it runs a
  different file. A helper file that an update replaced is refused (see above). A web page
  or another browser extension cannot start the native host, because Chrome starts it only
  for the extension ID in the manifest's `allowed_origins`, and replacing an installed
  helper needs write access to the install folder, which belongs to root.
- Not covered: code that already runs as this user can fork after it connects and hand the
  socket to the child while the parent execs the real helper; it can inject into a real
  helper with `LD_PRELOAD` or `ptrace`; it can read the meeting library and the app's
  settings directly. Same-user code is the trust boundary here, as on many Linux desktop
  apps, and the socket gives it nothing that it could not already reach.

## Not on Linux yet

- The encrypted library does not open, so the app shows the reason and closes (see the
  `## Unreleased` changelog entry).
- Meeting reminders use the macOS Notification Center.
- The recording bubble, Mission Control, Time Machine exclusion, the clipboard file copy,
  and the system privacy settings panes are macOS features.
- Snap and Flatpak browsers. Snap Chromium is the default Chromium on Ubuntu, and both it
  and Flatpak Chrome, Chromium and Brave run in a sandbox that cannot start a host binary
  and look for native messaging manifests in their own folders, so the Meet extension
  cannot reach the app there. Install Chrome or Chromium from the distribution's deb/rpm
  packages instead.

