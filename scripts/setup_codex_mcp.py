#!/usr/bin/env python3
"""Link this worktree to the primary checkout's private MCP configs (Runewire key).

Fork-only. Links .mcp.json (Claude Code), opencode.json (OpenCode) and
.codex/config.toml (Codex). The files are gitignored and never committed.
"""
from pathlib import Path
import subprocess

CONFIGS = [".mcp.json", "opencode.json", ".codex/config.toml"]


def git_path(option):
    value = subprocess.check_output(["git", "rev-parse", option], text=True).strip()
    return Path(value).resolve()


def main():
    root = git_path("--show-toplevel")
    primary = git_path("--git-common-dir").parent
    if root == primary:
        print("Primary checkout: nothing to link.")
        return
    for name in CONFIGS:
        source = primary / name
        destination = root / name
        if not source.is_file():
            raise SystemExit(f"Primary checkout has no {name}.")
        if "runewire" not in source.read_text():
            raise SystemExit(f"Primary {name} has no Runewire server.")
        if destination.exists() or destination.is_symlink():
            if destination.resolve() == source.resolve():
                print(f"{name} is already linked.")
                continue
            raise SystemExit(f"Existing worktree {name} preserved; reconcile it manually.")
        destination.parent.mkdir(mode=0o700, exist_ok=True)
        destination.symlink_to(source)
        print(f"Linked {name}.")


if __name__ == "__main__":
    main()
