#!/usr/bin/env bash
# Merge upstream (github.com/volotat/mini-AGI) into our main.
#
#     scripts/sync-upstream.sh
#
# Always a merge, never a rebase: main only moves forward. On a conflict the
# merge is left in progress for a person to resolve - see "SYNC" in
# RIG-TRAINING.md. Never pushes.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

branch=$(git symbolic-ref --quiet --short HEAD || true)
if [ "$branch" != "main" ]; then
    echo "sync-upstream: on '${branch:-detached HEAD}', not main - refusing" >&2
    exit 1
fi
# untracked files are allowed; git merge itself refuses to overwrite one
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "sync-upstream: uncommitted changes - commit them first" >&2
    git status --short --untracked-files=no >&2
    exit 1
fi

git fetch upstream

new=$(git log --oneline main..upstream/main)
if [ -z "$new" ]; then
    echo "sync-upstream: main already has everything in upstream/main"
    exit 0
fi
echo "upstream commits not yet in main:"
echo "$new" | sed 's/^/  /'
echo

if git merge --no-edit upstream/main; then
    echo
    echo "merged: $(git log -1 --oneline)"
    echo "now rerun the tests on the rig (RIG-TRAINING.md, 'Tests')."
    exit 0
fi

conflicted=$(git diff --name-only --diff-filter=U)
if [ -z "$conflicted" ]; then
    echo "sync-upstream: merge failed without conflicts - see above" >&2
    exit 1
fi
echo >&2
echo "sync-upstream: merge stopped on conflicts (left in progress):" >&2
while IFS= read -r f; do
    echo "  $f" >&2
    ours=$(git log --oneline --grep '^fix #' upstream/main..main -- "$f")
    if [ -n "$ours" ]; then
        echo "$ours" | sed 's/^/      our patch: /' >&2
    else
        echo "      (no 'fix #N' commit of ours touches it)" >&2
    fi
done <<< "$conflicted"
echo >&2
echo "if upstream fixed the same issue, take theirs: git checkout --theirs <file>" >&2
echo "then git add the files, git commit, and rerun the tests;" >&2
echo "or abandon with: git merge --abort" >&2
exit 1
