#!/bin/bash
# Builds the Rust engine, the helper binaries, and the Linux app bundles (.deb and AppImage).
# The macOS build stays in scripts/build.sh.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
triple="$(rustc -vV | sed -n 's/^host: //p')"

echo "Building the Rust engine, the MCP server and the Chrome native host"
(cd "$root" && cargo build --locked --release -p tinta-engine-rs -p tinta-mcp -p tinta-native-host)

# Tauri finds a sidecar as `binaries/<name>-<target triple>` and installs it as `usr/bin/<name>`,
# next to the app, where engine.rs and helper_path() look for it.
mkdir -p "$root/app/src-tauri/binaries"
for name in tinta-engine tinta-mcp tinta-native-host; do
  cp "$root/target/release/$name" "$root/app/src-tauri/binaries/$name-$triple"
done

if [[ "${1:-}" == "--binaries-only" ]]; then
  exit 0
fi

# The frontend is built once: `tauri build` runs `beforeBuildCommand: "pnpm build"` from
# app/. Only the dependencies need installing here.
echo "Installing the frontend dependencies"
(cd "$root/app" && pnpm install --frozen-lockfile)

# TODO(#14): inside an AppImage, helper_path() resolves to /tmp/.mount_*/usr/bin, which is
# gone after this process exits. The native host and MCP registration must copy the helpers
# once to ~/.local/share/Tinta/bin/<version> (0755) when $APPIMAGE is set and use those
# paths. That lookup lives in app/src-tauri/src/platform/linux.rs, which #14 owns; the
# packaging layer only documents it (README, docs/fork/linux-build.md).
echo "Building the app bundles"
(cd "$root/app" && pnpm tauri build)

echo "Built:"
ls -1 "$root/target/release/bundle/deb/"*.deb "$root/target/release/bundle/appimage/"*.AppImage
