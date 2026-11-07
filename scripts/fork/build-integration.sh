#!/usr/bin/env bash
# Build integration/linux out of the stacked Linux-port PRs.
#
# Resets the branch to origin/main, merges every port branch in dependency
# order, and prints the merged heads. Re-runnable: run it again whenever one of
# the port branches moves and it recreates the branch from scratch.
#
#   scripts/fork/build-integration.sh              # merge and push
#   scripts/fork/build-integration.sh --no-push     # merge only
#   scripts/fork/build-integration.sh --resume      # continue after a manual conflict fix
#
# The branch also carries a few integration-only commits (the stable AppImage
# helper paths, the smoke test). They are marked with an
# `Integration-only: true` trailer and cherry-picked back after the merges, so
# a rebuild does not lose them.
#
# Conflict handling:
#   - CHANGELOG.md `## Unreleased` is merged as the union of the entries.
#   - Cargo.lock is regenerated from the merged manifests with `cargo metadata`
#     (never `cargo update -w`); the final `--locked` build verifies it.
#   - Any other conflict stops the script. Resolve it, `git add` the files,
#     `git commit --no-edit`, and run the script again with --resume. Git's
#     rerere cache records the resolution, so the same conflict resolves itself
#     on the next run. docs/fork/integration.md lists the resolutions that the
#     current branches need.
#
# The only ref this script pushes is integration/linux, with --force-with-lease.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Resetting the branch removes this file from the tree. Run from a copy so bash
# can keep reading the script after that.
if [[ "${TINTA_INTEGRATION_COPY:-}" != "1" ]]; then
  copy="$(mktemp "${TMPDIR:-/tmp}/tinta-build-integration.XXXXXX.sh")"
  cp "$0" "$copy"
  TINTA_INTEGRATION_COPY=1 exec bash "$copy" "$@"
fi
trap 'rm -f "$0"' EXIT

REMOTE="${REMOTE:-origin}"
TARGET="${TARGET:-integration/linux}"
PUSH=1
RESUME=0

for argument in "$@"; do
  case "$argument" in
    --no-push) PUSH=0 ;;
    --resume) RESUME=1 ;;
    -h|--help)
      sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown argument: $argument (try --help)" >&2
      exit 2
      ;;
  esac
done

if [[ "$TARGET" != "integration/linux" ]]; then
  echo "refusing to build $TARGET: this script only owns integration/linux" >&2
  exit 2
fi

# Dependency order. A branch that is already contained in the result (because a
# later PR merged it) is skipped by `git merge` itself ("Already up to date").
BRANCHES=(
  feat/3/core-linux
  docs/4/engine-protocol
  feat/8/tntc-rust
  feat/10/linux-keys
  feat/5/app-linux
  feat/6/capabilities
  feat/13/linux-helpers
  feat/7/engine-skeleton
  feat/9/engine-finalize
  feat/20/recluster
  feat/11/linux-recording
  feat/12/linux-packaging
)

resolved_conflicts() {
  local conflicted
  conflicted="$(git diff --name-only --diff-filter=U)"
  [[ -n "$conflicted" ]] || return 0
  local -a unknown=()
  local file
  while IFS= read -r file; do
    case "$file" in
      Cargo.lock)
        echo "  resolving Cargo.lock from the merged manifests"
        git checkout --ours -- Cargo.lock
        if ! cargo metadata --format-version 1 >/dev/null; then
          echo "cargo metadata failed; fix Cargo.lock by hand" >&2
          return 1
        fi
        git add Cargo.lock
        ;;
      CHANGELOG.md)
        echo "  resolving CHANGELOG.md as the union of the Unreleased entries"
        python3 - <<'PY'
import re
import subprocess

def read(stage):
    return subprocess.check_output(
        ["git", "show", f":{stage}:CHANGELOG.md"], text=True
    )

def parse(text):
    match = re.search(r"^## Unreleased\s*$", text, re.M)
    if not match:
        return {}
    rest = text[match.end():]
    next_heading = re.search(r"^## ", rest, re.M)
    section = rest[: next_heading.start()] if next_heading else rest
    out, current = {}, None
    for line in section.splitlines():
        heading = re.match(r"^### (Added|Changed|Fixed|Removed)\s*$", line)
        if heading:
            current = heading.group(1)
            out.setdefault(current, [])
        elif line.startswith("- ") and current:
            out[current].append(line)
    return out

ours = read(2)
theirs = read(3)
merged = {}
for heading in ("Added", "Changed", "Fixed", "Removed"):
    items = []
    for entry in parse(ours).get(heading, []) + parse(theirs).get(heading, []):
        if entry not in items:
            items.append(entry)
    if items:
        merged[heading] = items

match = re.search(r"^## Unreleased\s*$", ours, re.M)
rest = ours[match.end():]
next_heading = re.search(r"^## ", rest, re.M)
block = "## Unreleased\n\n" + "".join(
    f"### {heading}\n" + "\n".join(items) + "\n\n"
    for heading, items in merged.items()
)
with open("CHANGELOG.md", "w") as handle:
    handle.write(ours[: match.start()] + block + rest[next_heading.start():])
PY
        git add CHANGELOG.md
        ;;
      *)
        unknown+=("$file")
        ;;
    esac
  done <<<"$conflicted"

  if ((${#unknown[@]})); then
    echo "" >&2
    echo "These conflicts need a person:" >&2
    printf '  %s\n' "${unknown[@]}" >&2
    echo "" >&2
    echo "Resolve them, then: git add <files> && git commit --no-edit," >&2
    echo "and run this script again with --resume." >&2
    return 1
  fi
}

# Commits that belong to integration/linux only are marked with an
# `Integration-only: true` trailer. They are cherry-picked back after every
# rebuild, because no port branch contains them.
collect_integration_only() {
  git log --reverse --format='%H%x09%(trailers:key=Integration-only,valueonly)' "$1" |
    while IFS=$'\t' read -r sha value; do
      if [[ "$value" == "true" ]]; then
        echo "$sha"
      fi
    done
}

git config rerere.enabled true
git config rerere.autoupdate true
state="$(git rev-parse --path-format=absolute --git-path tinta-integration-commits)"

if git rev-parse -q --verify MERGE_HEAD >/dev/null; then
  if [[ "$RESUME" -eq 0 ]]; then
    echo "a merge is in progress; finish it and rerun with --resume" >&2
    exit 1
  fi
  if [[ -n "$(git diff --name-only --diff-filter=U)" ]]; then
    resolved_conflicts || exit 1
  fi
  git commit --no-edit
fi

if git rev-parse -q --verify CHERRY_PICK_HEAD >/dev/null; then
  if [[ "$RESUME" -eq 0 ]]; then
    echo "a cherry-pick is in progress; finish it and rerun with --resume" >&2
    exit 1
  fi
  if [[ -n "$(git diff --name-only --diff-filter=U)" ]]; then
    resolved_conflicts || exit 1
  fi
  git commit --no-edit
fi

if [[ -n "$(git status --porcelain)" ]]; then
  echo "the worktree is not clean; commit or stash the changes first" >&2
  exit 1
fi

git fetch "$REMOTE" --prune

# Read the integration-only commits from the state file (a resumed run) or from
# the last published result. A first run on a new clone has none yet.
INTEGRATION_COMMITS=()
if [[ -f "$state" ]]; then
  mapfile -t INTEGRATION_COMMITS <"$state"
fi
if ((${#INTEGRATION_COMMITS[@]} == 0)); then
  source=""
  if git rev-parse --verify --quiet "$REMOTE/$TARGET" >/dev/null; then
    source="$REMOTE/$TARGET"
  elif git rev-parse --verify --quiet "$TARGET" >/dev/null; then
    source="$TARGET"
  fi
  if [[ -n "$source" ]]; then
    mapfile -t INTEGRATION_COMMITS < <(collect_integration_only "$source")
  fi
  printf '%s\n' "${INTEGRATION_COMMITS[@]}" >"$state"
fi

if [[ "$RESUME" -eq 0 ]]; then
  git checkout -B "$TARGET" "$REMOTE/main"
fi

for branch in "${BRANCHES[@]}"; do
  if ! git rev-parse --verify --quiet "refs/remotes/$REMOTE/$branch" >/dev/null; then
    echo "missing branch $REMOTE/$branch" >&2
    exit 1
  fi
  if git merge-base --is-ancestor "$REMOTE/$branch" HEAD; then
    echo "== $branch: already in the result"
    continue
  fi
  echo "== merging $REMOTE/$branch"
  if git merge --no-ff --no-edit "$REMOTE/$branch"; then
    continue
  fi
  if ! resolved_conflicts; then
    echo "" >&2
    echo "stopped while merging $REMOTE/$branch" >&2
    exit 1
  fi
  git commit --no-edit
done

for sha in "${INTEGRATION_COMMITS[@]}"; do
  [[ -n "$sha" ]] || continue
  if git merge-base --is-ancestor "$sha" HEAD 2>/dev/null; then
    echo "== integration-only $(git log -1 --format='%h %s' "$sha"): already in the result"
    continue
  fi
  echo "== integration-only $(git log -1 --format='%h %s' "$sha")"
  if ! git cherry-pick --no-edit "$sha"; then
    echo "" >&2
    echo "stopped while applying the integration-only commit $sha" >&2
    echo "Resolve the conflict, git add the files, git commit --no-edit," >&2
    echo "then run this script again with --resume." >&2
    exit 1
  fi
done

# A rerere replay can leave an older Cargo.lock behind. Resolve the lock one
# last time from the merged manifests and commit it if it changed.
if ! cargo metadata --format-version 1 >/dev/null; then
  echo "cargo metadata failed; Cargo.lock needs a manual fix" >&2
  exit 1
fi
if ! git diff --quiet -- Cargo.lock; then
  echo "== regenerating Cargo.lock from the merged manifests"
  git add Cargo.lock
  git commit -m "Cargo.lock: regenerate from the merged manifests"
fi

if git grep -n -e '^<<<<<<< ' -e '^>>>>>>> ' -- . ':!scripts/fork/*' >/tmp/tinta-conflict-markers 2>/dev/null; then
  echo "conflict markers remain:" >&2
  cat /tmp/tinta-conflict-markers >&2
  exit 1
fi

echo ""
echo "merged heads:"
for branch in "${BRANCHES[@]}"; do
  printf '  %-28s %s\n' "$branch" "$(git log -1 --format='%h %s' "$REMOTE/$branch")"
  if ! git merge-base --is-ancestor "$REMOTE/$branch" HEAD; then
    echo "  ^ $branch is not contained in the result" >&2
    exit 1
  fi
done
echo ""
printf 'integration head: %s\n' "$(git log -1 --format='%h %s' HEAD)"

if [[ "$PUSH" -eq 1 ]]; then
  git push --force-with-lease "$REMOTE" "HEAD:refs/heads/$TARGET"
else
  echo "not pushing (--no-push)"
fi
