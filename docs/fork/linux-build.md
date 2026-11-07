# Building Tinta on Linux

The Tauri app, `tinta-core`, `tinta-native-host` and `tinta-mcp` build and test on Linux
without a display. The Keychain path in `crates/tinta-core/src/keys.rs` is only compiled on
macOS; Linux and Windows keep the library key in the key store of the system (see below).

## Dependencies (Debian and Ubuntu)

SQLCipher is bundled, but it compiles against the system OpenSSL, so the OpenSSL
development headers and a C compiler are required:

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

The app also needs the WebKitGTK development headers:

```sh
sudo apt-get install -y libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev
```

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

- Meeting reminders use the macOS Notification Center.
- The recording bubble, Mission Control, Time Machine exclusion, the clipboard file copy,
  and the system privacy settings panes are macOS features.
- Snap and Flatpak browsers. Snap Chromium is the default Chromium on Ubuntu, and both it
  and Flatpak Chrome, Chromium and Brave run in a sandbox that cannot start a host binary
  and look for native messaging manifests in their own folders, so the Meet extension
  cannot reach the app there. Install Chrome or Chromium from the distribution's deb/rpm
  packages instead.
