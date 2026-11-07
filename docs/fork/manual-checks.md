# Manual checks for the Windows and Linux port

Checks that need a person in front of the app, because they depend on what the window shows or on
the commands that reach the engine.

## An engine without the optional features

The engine reports what it can do in `capabilities` (`docs/engine-protocol.md` §10.1). The app must
not use a feature that the engine reports as false: no calendar read or reminder, no call polling,
no window reading, no summary, no audio import.

`scripts/fork/fake-engine-capabilities.py` is a fake engine that reports every feature as false and
logs every command it receives.

1. Build the app and start it as usual in development. On macOS the library opens with the Keychain
   key; on Windows and Linux this check waits for the platform key store (task #10).
2. Put the fake engine where the app looks for it, next to the development executable:

   ```sh
   chmod +x scripts/fork/fake-engine-capabilities.py
   ln -sf "$PWD/scripts/fork/fake-engine-capabilities.py" target/debug/tinta-engine
   ```

   On Windows, copy the script to `target\debug\tinta-engine.exe` (it needs a Python launcher) or
   point the engine resolver at it.
3. Start Tinta in development as usual (`pnpm dev` in `app/`, then `cargo run -p tinta`), with a
   fresh log:

   ```sh
   rm -f /tmp/tinta-fake-engine.log
   TINTA_FAKE_ENGINE_LOG=/tmp/tinta-fake-engine.log cargo run -p tinta
   ```

4. Use the app for a few minutes: open Home, Settings, and a meeting; focus and unfocus the window;
   if the window shows no calendar section, that is the expected result.
5. Check that the app sent none of the commands of a disabled feature:

   ```sh
   grep -E '"(calendar|request_calendar|calls|call_report|request_accessibility|summary_status|summarize|import_audio)"' \
     /tmp/tinta-fake-engine.log
   ```

   The command prints nothing. `ping`, `models_status`, `permissions`, `list_sources`, and the
   recording commands (`start`, `stop`, `finalize`, `sample`) may appear: they are not optional
   features.

6. Remove the link with `rm target/debug/tinta-engine` when the check ends.

The log shows `ping` at startup: it starts the engine and carries the capabilities before
the app starts any watcher.
