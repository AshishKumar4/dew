#!/bin/sh
# CI's package gate: what a release attaches and Dew's index serves, installed the way a user installs it. The sdist
# and wheel build and pass twine, and name no requirement by URL, which an index refuses and twine passes
# (tools/check_distribution.py); the wheel installs into an empty environment and runs (tools/smoke_wheel.py); an
# install from PyPI alone stops at resolving jax, on which Dew carries fixes; and the accelerator extras resolve
# against Dew's package index. The land queue runs it as an armada task (dew-ops queue/gates.sh), and GitHub's CI for
# a pull request and a release on a hosted runner (.github/workflows/ci.yml), both from here.
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
"$work/build/bin/python" -m build --outdir "$dist"
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
# The accelerator extras resolve against Dew's index.
index=$("$work/build/bin/python" -c "import runpy; print(runpy.run_path('src/dew/_builds.py')['PACKAGE_INDEX'])")
wheel=$(realpath "$dist"/*.whl)
for extras in cuda12,quantization,profile cuda13,quantization,profile tpu,quantization,profile; do
  echo "dewml[$extras] @ file://$wheel" | uv pip compile - --extra-index-url "$index" --python-version "$python" \
    --python-platform x86_64-manylinux_2_31 --quiet > /dev/null
done
echo "package: green"
