#!/bin/sh
# CI's package gate: what a release attaches and Dew's index serves, installed the way a user installs it. The sdist
# and wheel build and pass twine, and name no requirement by URL, which an index refuses and twine passes
# (tools/check_distribution.py); the wheel installs into an empty environment and runs, its requirements at their newest
# and at the lower bounds dewml declares (tools/smoke_wheel.py); an install from PyPI alone stops at resolving jax, on
# which Dew carries fixes; and the accelerator extras resolve against the indexes constraints.txt names. GitHub's CI
# runs it for a pull request and a release on a hosted runner (.github/workflows/ci.yml).
#
# A release's package carries the template derivations its CI derived for each CUDA generation (DEW_DERIVED, a
# directory of tools/ci/derived.sh's outputs, one a generation, each under its own directory's derived/): put in the
# package's place before it is built, their union, one name one content whichever generation derived it (an entry is
# named by its key), and its wheel refused unless it holds every one.
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
rm -rf "$dist"
if [ -n "${DEW_DERIVED:-}" ]; then
  into=src/dew/algebra/derived
  rm -rf "$into"
  mkdir -p "$into"
  for entry in "$DEW_DERIVED"/*/derived/*.json; do
    [ -e "$entry" ] || { echo "package: DEW_DERIVED ($DEW_DERIVED) holds no derivation"; exit 1; }
    name=$(basename "$entry")
    if [ -e "$into/$name" ]; then
      cmp -s "$entry" "$into/$name" || { echo "package: derivation $name differs between generations"; exit 1; }
    else
      cp "$entry" "$into/$name"
    fi
  done
  derived=$(ls "$into" | wc -l)
fi
"$work/build/bin/python" -m build --outdir "$dist"
if [ -n "${DEW_DERIVED:-}" ]; then
  held=$("$work/build/bin/python" -m zipfile -l "$dist"/*.whl | grep -c 'dew/algebra/derived/[^/]*\.json' \
    || true)
  [ "$held" = "$derived" ] || { echo "package: the wheel holds $held of the $derived derivations"; exit 1; }
fi
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
