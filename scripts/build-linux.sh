#!/bin/bash
# Builds the Rust engine, the helper binaries, and the Linux app bundles (.deb and AppImage).
# The macOS build stays in scripts/build.sh.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
triple="$(rustc -vV | sed -n 's/^host: //p')"

echo "Building the Rust engine, the MCP server and the Chrome native host"
(cd "$root" && cargo build --release -p tinta-engine-rs -p tinta-mcp -p tinta-native-host)

# Tauri finds a sidecar as `binaries/<name>-<target triple>` and installs it as `usr/bin/<name>`,
# next to the app, where engine.rs and helper_path() look for it.
mkdir -p "$root/app/src-tauri/binaries"
for name in tinta-engine tinta-mcp tinta-native-host; do
  cp "$root/target/release/$name" "$root/app/src-tauri/binaries/$name-$triple"
done

if [[ "${1:-}" == "--binaries-only" ]]; then
  exit 0
fi

echo "Building the frontend"
(cd "$root/app" && pnpm install --frozen-lockfile && pnpm build)

echo "Building the app bundles"
(cd "$root/app" && pnpm tauri build)

echo "Built:"
ls -1 "$root/target/release/bundle/deb/"*.deb "$root/target/release/bundle/appimage/"*.AppImage
