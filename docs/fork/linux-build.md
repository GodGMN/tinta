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

Some distributions install only a versioned `libcrypto.so.3`. SQLCipher links with
`-lcrypto`, so the linker needs a plain `libcrypto.so`; the `libssl-dev` package
provides the unversioned symlink. If OpenSSL is in a custom prefix, point
`OPENSSL_DIR` and `OPENSSL_LIB_DIR` at it (the library directory must be the
prefix's `lib`, with the plain `libcrypto.so` in it). Without it the build stops
with `unable to find library -lcrypto`.

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
`key.lock` in the library folder while it reads or creates the key, and after
writing it reads the stored value back and uses that.

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
```

The `core-linux` job in `.github/workflows/ci.yml` runs the same commands on
`ubuntu-latest`.

There is no Windows CI job yet: the bundled SQLCipher needs an OpenSSL toolchain
for Windows that the workflow does not install, so the Windows check is left to a
machine with a Visual Studio and OpenSSL setup. `cargo check -p tinta-core` is the
command to run there.
