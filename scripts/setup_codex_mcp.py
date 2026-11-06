#!/usr/bin/env python3
"""Link this worktree to the project's private Codex MCP configuration."""
from pathlib import Path
import subprocess
import tomllib


def git_path(option):
    value = subprocess.check_output(
        ["git", "rev-parse", option], text=True
    ).strip()
    return Path(value).resolve()


def main():
    root = git_path("--show-toplevel")
    source = git_path("--git-common-dir").parent / ".codex" / "config.toml"
    destination = root / ".codex" / "config.toml"
    if not source.is_file():
        raise SystemExit("Primary checkout has no project-local Codex config.")
    if "runewire" not in tomllib.loads(source.read_text()).get("mcp_servers", {}):
        raise SystemExit("Primary project-local Codex config has no Runewire server.")
    if destination.exists() or destination.is_symlink():
        if destination.resolve() == source.resolve():
            print("Project-local Codex MCP configuration is already linked.")
            return
        raise SystemExit("Existing worktree Codex config preserved; reconcile it manually.")
    destination.parent.mkdir(mode=0o700, exist_ok=True)
    destination.symlink_to(source)
    print("Linked project-local Codex MCP configuration for this worktree.")


if __name__ == "__main__":
    main()
