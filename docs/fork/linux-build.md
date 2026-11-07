# Building Tinta on Linux

The Linux port ships as a `.deb` package and an AppImage and contains the whole Linux
stack: the key store, the recording engine, the Chrome native host and MCP helpers, call
detection and capabilities, and the packaging layer. This document covers building,
testing, and packaging it.

## What the port does

- Records the monitor of the default output device with PipeWire or PulseAudio and writes
  the same TNTC audio as the Mac app.
- Transcribes and separates speakers on the machine with the bundled sherpa-onnx engine.
- Opens the encrypted library with a key in the Secret Service (`org.freedesktop.secrets`):
  GNOME Keyring, KWallet, or another provider.
- Names Google Meet speakers through the Chrome extension, its native host, and the MCP
  server that the bundles ship.
- Detects calls and reads capabilities, so Home, recording, and the extension follow the
  same flow as on the Mac.

The port still has these gaps, which the README lists: the calendar and **Next meetings**,
summaries (no local model), names from the Zoom and Microsoft Teams windows, and per-app
audio (it records the whole system output).

## Dependencies (Debian and Ubuntu)

Building `tinta-core` needs the system OpenSSL headers because SQLCipher is bundled but
compiles against OpenSSL, and it needs a C compiler:

```sh
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev
```

Without them the build stops with `fatal error: openssl/crypto.h: No such file or
directory` while compiling `libsqlite3-sys`.

Some distributions install only a versioned `libcrypto.so.3`. SQLCipher links with
`-lcrypto`, so the linker needs a plain `libcrypto.so`; the `libssl-dev` package
provides the unversioned symlink. If OpenSSL is in a custom prefix, point
`OPENSSL_DIR` and `OPENSSL_LIB_DIR` at it (the library directory must be the
prefix's `lib`, with the plain `libcrypto.so` in it). Without it the build stops
with `unable to find library -lcrypto`.

The app bundle itself needs the WebKitGTK and GTK development packages; the `bundle-linux`
job installs `libwebkit2gtk-4.1-dev`, `libayatana-appindicator3-dev`, `librsvg2-dev`,
`libxdo-dev`, and `patchelf`.

The Secret Service client vendors dbus, so `libdbus-1-dev` is not needed at build
time. The `keyring` dependency exists only on Linux and Windows; macOS keeps using
the Keychain and does not compile it.

## Library key store

The encrypted library needs one master key. macOS keeps it in the Keychain (the
code is unchanged). Linux and Windows keep it in the key store of the system:

- **Linux**: the Secret Service (`org.freedesktop.secrets`) over the session bus,
  as provided by GNOME Keyring and by KWallet when its Secret Service bridge is on.
  The item uses the service `com.usetinta.tinta` and the account
  `library-master-key`, the same names as the macOS Keychain item. The value is the
  32-byte master key as hex, the same encoding as macOS.
- **Windows**: the Windows Credential Manager, as a generic credential with the
  target name `library-master-key.com.usetinta.tinta` and the user name
  `library-master-key`. It appears in **Control Panel > Credential Manager >
  Windows Credentials**, and it is protected by the Windows login of the user.
  Tinta stores it as a persistent enterprise credential
  (`CRED_PERSIST_ENTERPRISE`), so it roams with the domain roaming profile of the
  user. It needs an interactive login: a non-interactive logon such as SSH or a
  service account has no Credential Manager vault
  (`ERROR_NO_SUCH_LOGON_SESSION`), and Tinta then stops with an error instead of
  creating a key.

A normal GNOME or KDE desktop starts the Secret Service and unlocks the login
keyring when the user logs in, so Tinta needs no extra setup. KeePassXC (with its
Secret Service integration) and KWallet with its bridge must be unlocked
**before** Tinta starts: their databases are otherwise locked, their search may
return no item, and Tinta reads that as a missing key.

Tinta never creates a key for an existing library. If `library.db` or recorded
meetings already exist but the store reports no item, Tinta stops with:

```text
The key for your existing library was not found. Unlock your keyring (GNOME Keyring, KWallet, KeePassXC) and open Tinta again.
```

Unlock the keyring and start Tinta again. A new key would make the existing library
permanently unreadable.

Two instances started at once do not race: Tinta holds an exclusive lock on
`key.lock` in the per-user Tinta folder (`~/.local/share/Tinta`, or
`%APPDATA%\Tinta`) while it reads or creates the key, and after writing it reads
the stored value back and uses that. There is one key per user, so a key is only
created when neither that folder nor the configured library folder has data
(`library.db*`, `meetings/`, or a `config.json` that names a moved library).
Debug builds accept `TINTA_DEV_MASTER_KEY` (hex) like on macOS, so tests never use
the real key. The lock has no timeout; on NFS mounted with `nolock` it fails and
Tinta refuses to start, which is safe.

If the `default` collection alias points at the in-memory `session` collection (a
minimal setup can do this, and no login keyring is created), the key is written
there and is lost at logout. Tinta then finds the library but no key on the next
start and asks for the keyring, so use a persistent login collection.

### Minimal window managers and headless sessions

A minimal setup may not run a Secret Service provider: i3, sway, a bare X session,
a server without a desktop session, and WSL often have none. In that case Tinta
cannot read or create the key and stops with an error like:

```text
cannot read the library key from the system key store: Platform secure storage failure: ...
```

It never creates a new key when the store is unavailable, locked, or returns any
other error, and it never falls back to a plaintext file. A new key would make the
existing library unreadable. Install and start GNOME Keyring (or KWallet with the
Secret Service bridge), unlock it, and start Tinta again.

To check the provider and the item:

```sh
# Is a Secret Service provider on the session bus? (systemd systems)
busctl --user list | grep org.freedesktop.secrets

# Read the stored key. Install libsecret-tools for secret-tool.
secret-tool lookup service com.usetinta.tinta username library-master-key
```

On a first run the lookup prints nothing and exits 1 because the item does not
exist yet; Tinta creates it. After that it prints the hex key. When no provider is
running, `secret-tool` also prints nothing (or cannot connect to the bus) and
exits non-zero; it does not print a "No such secret" message.

## Verify

```sh
cargo test --locked -p tinta-core
cargo clippy --locked -p tinta-core --all-targets -- -D warnings
cargo test --locked -p tinta-native-host -p tinta-mcp
cargo clippy --locked -p tinta-native-host -p tinta-mcp --all-targets -- -D warnings
```

The `core-linux` and `helpers-linux` jobs in `.github/workflows/ci.yml` run these commands
on `ubuntu-latest`. The `app-linux` job runs clippy for the app crate with the WebKitGTK
headers installed. The engine commands below also run in `core-linux`.

## The engine crate on Linux, Windows and macOS

`crates/tinta-engine-rs` depends on `tinta-core` only through its `chunks` module (the
`database` feature is off), so the bundled SQLCipher build and the OpenSSL headers above
do not apply to it. Its dependencies are pure Rust plus the prebuilt sherpa-onnx native
libraries that `sherpa-onnx-sys` downloads in its build script. The `engine-windows`
(`windows-latest`) and `engine-macos` (`macos-15-intel`) jobs in
`.github/workflows/ci.yml` build and test only `-p tinta-engine-rs`, with no extra system
packages:

```sh
cargo test --locked -p tinta-engine-rs
cargo clippy --locked -p tinta-engine-rs --all-targets -- -D warnings
```

On Linux the same two commands run in `core-linux` after the `tinta-core` steps.

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
  system `libcrypto`; sherpa-onnx links `libstdc++`. `libpulse0` is what the recording
  engine needs: `libpulse-binding` and `libpulse-simple-binding` load `libpulse.so.0` and
  `libpulse-simple.so.0` at runtime, and both libraries ship in `libpulse0` on Debian and
  Ubuntu. `tauri.linux.conf.json` carries a note as a comment next to the dependency
  (`tauri-build` enables its `config-json5` feature, so the file may carry comments).
- The Secret Service client talks D-Bus directly with a vendored libdbus, so the package
  needs no libsecret; `libdbus-1.so.3` is NEEDED through `tao` and the vendored client. The
  package recommends the service behind `org.freedesktop.secrets` as
  `gnome-keyring | libkf5wallet-bin`. `libkf5wallet-bin` contains `kwalletd5`, the Secret
  Service daemon on Debian 12 and Ubuntu 24.04. The KDE Frameworks 6 package `kwalletd6`
  does not exist in either (it arrives in Debian 13 and Ubuntu 24.10), so naming it here
  would add an alternative that no supported release can satisfy. `kwalletmanager` is only
  the KWallet GUI and is not a Secret Service provider.
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
quits, so the Chrome native host manifest and the MCP command cannot point there.

When `$APPIMAGE` is set, the app copies `tinta-engine`, `tinta-mcp` and
`tinta-native-host` once to `$XDG_DATA_HOME/Tinta/bin/<version>/` (default
`~/.local/share/Tinta/bin/<version>/`), mode 0755, and returns those paths from
`helper_path()` and `engine_path()`. The peer check reads the helper files from the same
folder, so the copy is also what the app allows to connect to its socket. The folder is
keyed by the app version, so an update writes a new copy and the old one can still run
until it exits; a regular `.deb` install keeps using `/usr/bin`.

## Not on Linux yet

- Meeting reminders use the macOS Notification Center.
- The recording bubble, Mission Control, Time Machine exclusion, the clipboard file copy,
  and the system privacy settings panes are macOS features.
- Snap and Flatpak browsers. Snap Chromium is the default Chromium on Ubuntu, and both it
  and Flatpak Chrome, Chromium and Brave run in a sandbox that cannot start a host binary
  and look for native messaging manifests in their own folders, so the Meet extension
  cannot reach the app there. Install Chrome or Chromium from the distribution's deb/rpm
  packages instead.
