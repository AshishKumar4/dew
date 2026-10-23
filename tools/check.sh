#!/bin/sh
# Every cheap gate CI runs on a checkout, all of them, reporting every failure before it exits. The gates are
# tools/check_gates' lines, each a name and a command: the locks, flake8, ruff, slop lint and SLOP020's ratchet,
# the fetched fixtures' pins (tools/fetched_fixtures.py), the complexity record (complexity.json), the generated
# API pages, installer, site colors, model list, examples and live model pin, the lazy packages' import
# surfaces (tools/gen_exports.py), the tested Python versions, pyright, and duplicate code
# (tools/duplicates.py), read as text and as structure. PYTHON names the interpreter of the test suite's
# environment (tools/install_environment.sh); default: python. BASE is the ref SLOP020's ratchet holds the
# inventory to; default: origin/main. A lane runs it before every LAND, and CI's lint job runs it as is.
# Usage, from the checkout: sh tools/check.sh [--changed | --gate NAME | --list]
#
# The full run starts the gates side by side, as many at once as this process may use processors (nproc),
# and prints each one's verdict and, where it failed, its output, in the file's order: it takes about as long
# as its slowest gate. --gate NAME runs one gate and exits with its status (the lanes' lint gate runs each
# gate on a container of its own); --list prints the gates' names.
#
# --changed: the gates a head's own Python files decide, in seconds, for iterating on it: ruff and slop lint
# over the Python files that differ from its merge-base with origin/main (committed or not, untracked
# included), SLOP020's ratchet, and the duplicate scan's structural reading held to origin/main. Every gate
# runs, without it, before a LAND.
set -u
gates=tools/check_gates
PYTHON=${PYTHON:-python}
BASE=${BASE:-origin/main}
export PYTHON BASE
names() { grep -v '^#' "$gates" | cut -f1; }
# A gate's command, from the checkout's top, with the lint inputs read; its exit status is the gate's.
run() {
  command=$(grep -v '^#' "$gates" | awk -F'\t' -v name="$1" '$1 == name {print $2; found = 1} END {exit !found}') ||
    { echo "no gate $1 in $gates"; return 2; }
  sh -c ". tools/lint_inputs.sh; $command"
}
failed=""
gate() {
  name=$1
  shift
  if out=$("$@" 2>&1); then
    echo "=== $name ok"
  else
    failed="$failed $name"
    printf '=== %s FAILED\n%s\n' "$name" "$out"
  fi
}
case "${1:-}" in
  --list) names; exit 0 ;;
  --gate) run "$2"; exit ;;
  --changed)
    . tools/lint_inputs.sh
    sh tools/base_history.sh
    since=$(git merge-base origin/main HEAD)
    gate ruff sh -c '. tools/lint_inputs.sh; python_changed "$1" $ruff_inputs | xargs -0r "$PYTHON" -m ruff check' \
      _ "$since"
    # SLOP020's inventory names files across the tree, and a deleted file can leave its entry behind: a head
    # that changes the inventory or deletes a Python file has the whole tree read, as the full run reads it.
    if git diff --quiet "$since" -- tools/tolerance_inventory.json &&
      [ -z "$(git diff --no-renames --name-only --diff-filter=D "$since" -- '*.py')" ]; then
      gate slop sh -c '. tools/lint_inputs.sh; python_changed "$1" $("$PYTHON" tools/lint_slop.py --print-roots) |
        JAX_PLATFORMS=cpu xargs -0r "$PYTHON" tools/lint_slop.py' _ "$since"
    else
      gate slop run slop
    fi
    gate ratchet run ratchet
    gate duplicate-structure run duplicate-structure
    [ -z "$failed" ] || { echo "check --changed: FAILED$failed"; exit 1; }
    echo "check --changed: green"
    exit 0 ;;
esac
# Each gate's output and status in a directory of the run's own, read back in the file's order once all have
# ended. The history the duplicate scan and the ratchet read is fetched once, before either starts.
sh tools/base_history.sh
results=$(mktemp -d)
names | xargs -P "$(nproc)" -I '{}' \
  sh -c 'sh tools/check.sh --gate "$1" > "$2/$1.out" 2>&1; echo $? > "$2/$1.status"' \
  _ '{}' "$results"
for name in $(names); do
  gate "$name" sh -c 'cat "$1/$2.out"; exit "$(cat "$1/$2.status")"' _ "$results" "$name"
done
rm -rf "$results"
if [ -z "$failed" ]; then echo "check: all gates green"; else echo "check: FAILED$failed"; exit 1; fi
