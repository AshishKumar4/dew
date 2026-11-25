#!/bin/sh
# The locks gate (tools/check_gates): Dew's locks are what tools/lock_environment.sh resolves from their
# inputs. A run given a base (BASE, set: CI's lint job, a lane's precheck, a suspect's isolation) resolves
# them again only where an input of theirs differs from its merge-base with it, and else passes, saying so:
# each resolve fetches every staged wheel Dew's index lists from GitHub's releases, and in train 1273's
# suspect 35 lint runs resolving at once were refused (REFUSED_STREAM) on 30, main's own among them, which
# blamed main and hid the heads' reds. A run given none (check.sh by hand) always resolves.
set -eu
INPUTS="tools/lock_environment.sh tools/locks_gate.sh pyproject.toml constraints.txt .python-version locks
uv.lock"
if [ -n "${BASE_GIVEN:-}" ]; then
  sh tools/base_history.sh
  base=$(git merge-base HEAD "$BASE")
  # shellcheck disable=SC2086 # the inputs are one path each
  if git diff --quiet "$base" -- $INPUTS && [ -z "$(git ls-files --others --exclude-standard -- $INPUTS)" ]
  then
    echo "locks: no lock input differs from $(git rev-parse --short=9 "$base"), $BASE's merge-base: kept"
    exit 0
  fi
fi
sh tools/lock_environment.sh
git diff --exit-code --stat locks/
