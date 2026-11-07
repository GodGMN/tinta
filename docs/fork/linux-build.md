# Building the core crate on Linux

`tinta-core` builds and tests on Linux. The Tauri app (`app/src-tauri`) still uses
macOS-only code, so the app does not build yet, but the library key store works on
Linux and Windows.

## Dependencies (Debian and Ubuntu)

SQLCipher is bundled, but it compiles against the system OpenSSL, so the OpenSSL
development headers and a C compiler are required:

```sh
sudo apt-get update
sudo apt-get install -y build-essential libssl-dev
```

Without them the build stops with `fatal error: openssl/crypto.h: No such file or
directory` while compiling `libsqlite3-sys`.

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

A normal GNOME or KDE desktop starts the Secret Service and unlocks the login
keyring when the user logs in, so Tinta needs no extra setup. The keyring must be
unlocked: if it is locked, Tinta asks the Secret Service to unlock it, and the
desktop may show a prompt.

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

On a first run the lookup prints nothing because the item does not exist yet;
Tinta creates it. After that it prints the hex key. `secret-tool` fails with
`No such secret` (or cannot connect to the bus) when no provider is running.

## Verify

```sh
cargo test --locked -p tinta-core
cargo clippy --locked -p tinta-core --all-targets -- -D warnings
```

The `core-linux` job in `.github/workflows/ci.yml` runs the same commands on
`ubuntu-latest`.

There is no Windows CI job yet: the bundled SQLCipher needs an OpenSSL toolchain
for Windows that the workflow does not install, so the Windows check is left to a
machine with a Visual Studio and OpenSSL setup. `cargo check -p tinta-core` is the
command to run there.
