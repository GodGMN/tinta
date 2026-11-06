# Fork workflow (Windows and Linux port)

This fork (`GodGMN/tinta`) adds Windows and Linux support to
[pepebndc/tinta](https://github.com/pepebndc/tinta). The work goes upstream later
as one PR or a stack of PRs. Files under `docs/fork/`, `scripts/setup_codex_mcp.py`
and the Runewire sections of `AGENTS.md`/`CLAUDE.md` are fork-only: never mix them
into a feature commit, so they can be dropped when the upstream PR is prepared.

## Goal and constraints

- Mac behavior must not change. The Swift engine and the Mac Keychain path stay.
- Windows and Linux get a second engine in Rust (`crates/tinta-engine-rs`) that speaks
  the same JSON-lines protocol as `engine/` (Swift).
- Unsupported features (summaries, calendar, Zoom/Teams names) are disabled through
  engine capabilities: no controls, no polling, no calls.
- Simplest long-term maintenance wins over completeness.
- Performance bar: a one-hour meeting processed in 10 minutes or less on a CPU-only
  x86_64 machine with 16 GB (upstream's own target in `docs/TESTING.md`).

## Where knowledge goes

- Design decisions, plans and measured results: `docs/fork/`.
- Behavior: code, types and tests. Short comments for non-obvious reasons only.
- Task history, implementation results and review evidence: Runewire comments and PRs.

## Versions and the changelog in the fork

Upstream bumps the version in four places in every commit (see the top of `AGENTS.md`).
In the fork, parallel PRs would conflict on that, so:

- Do not change the version in task PRs.
- Add each user-facing change to a `## Unreleased` section at the top of `CHANGELOG.md`
  (create it if missing), using upstream's Added/Changed/Fixed/Removed format and tone.
- The version bump happens once, when the upstream PR is prepared.

## Git checkout ownership

- Keep `/home/sergio/code/projects/opensource/tinta-fork` on `main`. Never use it for
  task implementation. Update it only by fast-forward to `origin/main`.
- `upstream` is pepebndc/tinta; `origin` is the fork. Sync upstream into `main` with a
  merge, never a rebase of published history.
- Task worktrees go under `/home/sergio/code/worktrees/tinta-fork/<task>-<slug>`.
  Run every command with an explicit working directory inside the worktree.
- Remove only your own merged, clean worktree with `git worktree remove`. Keep branches.

## Feature workflow

1. Read the task with `get_task`; move it to `doing` with `update_task`.
2. Create the worktree from the latest `origin/main`:
   `git worktree add /home/sergio/code/worktrees/tinta-fork/<task>-<slug> -b <type>/<task>/<slug> origin/main`,
   then run `python3 scripts/setup_codex_mcp.py` in it. It links the gitignored MCP
   configs (`.mcp.json`, `opencode.json`, `.codex/config.toml`) from the primary checkout.
   If you were started inside a worktree that T3 already prepared, use that worktree
   and branch instead of creating another one.
3. Implement. Run the focused tests you touched, then
   `cargo clippy --locked --all-targets -- -D warnings` for the crates you changed.
   Mac-only code cannot be built here; say so in the PR and rely on the macOS CI job.
4. `git fetch origin`; if `main` moved, merge `origin/main` and re-run checks.
5. Open a PR against `GodGMN/tinta` `main` (never against upstream) with
   `Refs #<taskId>`, what changed, how it was verified and the manual test notes.
6. Post a Runewire comment with the change and test performed; move the task to `qa`.
7. Every PR is reviewed by a Claude Opus reviewer before merge. Fix its findings on
   the same branch.
8. Merge only after the user replies `merge` in the conversation.
9. After the merge, remove the task worktree and task-owned temporary data.

## Evidence

- One evidence directory per task under `/home/sergio/code/evidence/tinta-fork/<task>/`.
  Link it from the PR or task comment. Generated evidence older than seven days may be
  deleted. Never commit benchmark audio, models or build outputs.
