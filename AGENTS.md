# Tinta

## Versions and the changelog

Every change to the app gets a new version and a changelog entry, in the same commit as the change.

1. Select the version part from the change, with [Semantic Versioning](https://semver.org):
   - **Patch** (0.1.0 to 0.1.1): bug fixes, text changes, and small UI changes. The behavior that users know stays the same.
   - **Minor** (0.1.0 to 0.2.0): new features, or changes to how a feature works. Old data and settings still work.
   - **Major** (0.1.0 to 1.0.0): changes that break old data, settings, the MCP tools, or the extension protocol.
2. Change the version in all four places. They must always be equal:
   - `app/src-tauri/tauri.conf.json` (`version`). Settings shows this version.
   - `app/package.json` (`version`).
   - `Cargo.toml` (`[workspace.package] version`). Run `cargo build` after the change so `Cargo.lock` updates.
   - `engine/Sources/TintaEngine/Engine.swift` (the `version` of the `ready` event).
3. Add an entry at the top of `CHANGELOG.md`. Create the file if it does not exist. Use this format:

   ```markdown
   ## 0.2.0 (2026-09-25)

   ### Added
   - Folders in the sidebar. Drag a meeting onto a folder to move it.

   ### Changed
   - Speaker names have colors. The initials are gone.

   ### Fixed
   - The sidebar filter no longer stays on a folder that does not exist.
   ```

   - Use only the sections that have items: **Added**, **Changed**, **Fixed**, and **Removed**.
   - Write each item for the user of the app, not for the developer. Say what the user sees or can do.
   - Use the date of the commit.
4. One commit can contain many changes. Give them one version, and use the highest part that one of the changes needs.

## Runewire

This project is managed in Runewire (project slug: tinta-fork).
Use the Runewire MCP tools (get_task, list_tasks, update_task,
create_task_comment) for all task context — do not call
the REST API or GitHub directly for task data.

The Runewire MCP config in this repo holds a personal API key. It is
gitignored and must never be committed.

Before starting Codex in a new worktree, run `python3 scripts/setup_codex_mcp.py`
there to link the ignored MCP configs (`.mcp.json`, `opencode.json`,
`.codex/config.toml`) to the primary checkout's private copies. Do not move the key into global settings.

Follow the fork workflow in [docs/fork/WORKFLOW.md](docs/fork/WORKFLOW.md):
checkout ownership, worktrees, verification, review and merge rules.

Workflow:
1. Read the task with get_task.
2. Move it to doing with update_task before implementation.
3. Implement and verify the change.
4. Open a PR referencing the task (Refs #<taskId>) with manual test notes,
   or post the test plan with create_task_comment.
5. Ensure the task reaches qa with update_task when finished.
