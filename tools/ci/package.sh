#!/bin/sh
# CI's package gate: what a release attaches and Dew's index serves, installed the way a user installs it. The sdist
# and wheel build and pass twine, and name no requirement by URL, which an index refuses and twine passes
# (tools/check_distribution.py); the wheel installs into an empty environment and runs, its requirements at their newest
# and at the lower bounds dewml declares (tools/smoke_wheel.py); an install from PyPI alone stops at resolving jax, on
# which Dew carries fixes; and the accelerator extras resolve against the indexes constraints.txt names. GitHub's CI
# runs it for a pull request and a release on a hosted runner (.github/workflows/ci.yml).
#
# The package is built from the checkout's committed tree staged apart (git archive HEAD), never from the
# checkout: its ignored package data (src/dew/algebra/derived, which a catalogue run writes) reaches no wheel,
# and nothing here writes to it. A release's package carries the template derivations its CI derived
# (DEW_DERIVED: a directory holding one directory a generation, named for it, each holding that generation's
# entries, as tools/ci/derived.sh writes them; DEW_GENERATIONS: the generations it must hold, each with one
# entry at least): their union, one name one content whichever generation derived it (an entry is named by its
# key), staged in the package's place, and its wheel refused unless it holds exactly them. Any other build
# ships none.
#
# Usage: sh tools/ci/package.sh [dist directory]   from the checkout's top, with uv on PATH; the directory (dist by
#                                                   default) is left holding the sdist and wheel it checked.
set -eu
dist=${1:-dist}
python=$(tr -d '\n' < .python-version)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
# Every environment's pip is tools/pip.txt's before it installs anything: older pips trust the roots of eMudhra's
# compromised CAs (2026-10-20).
uv venv -q --seed -p "$python" "$work/build"
"$work/build/bin/python" -m pip install -q -r tools/pip.txt
"$work/build/bin/python" -m pip install -q build twine
tree=$work/tree
mkdir "$tree"
git archive HEAD | tar -x -C "$tree"
into=$tree/src/dew/algebra/derived
mkdir -p "$into"
if [ -n "${DEW_DERIVED:-}" ]; then
  [ -n "${DEW_GENERATIONS:-}" ] || { echo "package: DEW_DERIVED names no DEW_GENERATIONS it must hold"; exit 1; }
  # The generations' union staged and weighed by the one rule of what a package may carry (tools/derived_weight.py).
  # shellcheck disable=SC2086 # Each generation its own word.
  "$work/build/bin/python" tools/derived_weight.py stage "$into" "$DEW_DERIVED" $DEW_GENERATIONS ||
    { echo "package: the release's derivations could not be staged"; exit 1; }
fi
(cd "$into" && ls) | sort > "$work/derived.txt"
rm -rf "$dist"
dist=$(mkdir -p "$dist" && cd "$dist" && pwd)
(cd "$tree" && "$work/build/bin/python" -m build --outdir "$dist")
# The wheel's derivations are exactly the staged ones: every one, and none else.
"$work/build/bin/python" -m zipfile -l "$dist"/*.whl | sed -n 's|.*dew/algebra/derived/\([^/ ]*\.json\) .*|\1|p' \
  | sort > "$work/shipped.txt"
cmp -s "$work/derived.txt" "$work/shipped.txt" || {
  echo "package: the wheel's derivations are not the staged ones:"; diff "$work/derived.txt" "$work/shipped.txt"; exit 1; }
"$work/build/bin/python" -m twine check --strict "$dist"/*
"$work/build/bin/python" tools/check_distribution.py "$dist"/*
"$work/build/bin/python" tools/smoke_wheel.py "$dist"/*.whl
# Without Dew's package index, an install stops at resolving jax.
uv venv -q --seed -p "$python" "$work/plain"
"$work/plain/bin/python" -m pip install -q -r tools/pip.txt
if "$work/plain/bin/python" -m pip install --index-url https://pypi.org/simple "$dist"/*.whl > "$work/plain.txt" 2>&1; then
  cat "$work/plain.txt"; echo "dewml installed from PyPI alone, on a jax without Dew's fixes"; exit 1
fi
grep -q "No matching distribution found for jax" "$work/plain.txt" || { cat "$work/plain.txt"; exit 1; }
# The accelerator extras resolve against the indexes constraints.txt names: Dew's, and a release under test's
# staging, which /simple lists only once the pin's train is green (tools/smoke_wheel.py's INDEXES).
indexes=$(sed -n 's/^--extra-index-url /--extra-index-url=/p' constraints.txt)
wheel=$(realpath "$dist"/*.whl)
for extras in cuda12,quantization,profile cuda13,quantization,profile tpu,quantization,profile; do
  echo "dewml[$extras] @ file://$wheel" | uv pip compile - $indexes --python-version "$python" \
    --python-platform x86_64-manylinux_2_31 --quiet > /dev/null
done
echo "package: green"
