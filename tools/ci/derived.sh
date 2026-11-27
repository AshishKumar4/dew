#!/bin/sh
# One CUDA generation's template derivations, as a release's wheel ships them (tools/derived_catalogue.py,
# docs/design/algebra.md §6.5), written to OUT: on armada, in the commit's environment (.venv-<python>: its
# locks' egglog and JAX builds, which every entry's key states), with cuDNN beside it as a GPU host has it
# (locks/cuda12.txt's build, its hashes checked, on the path apart from the environment), since a step Dew
# routes to cuDNN's attention lowers for a generation only where cuDNN loads. CI's release run maps it over
# each generation and its package job ships their union (tools/ci/package.sh's DEW_DERIVED).
#
# Usage: sh tools/ci/derived.sh GENERATION OUT   from the checkout's top; OUT absent or holding entries alone
set -eu
generation=$1 out=$2
python=$PWD/.venv-$(tr -d '\n' < .python-version)/bin/python
beside=$(mktemp -d)
trap 'rm -rf "$beside"' EXIT
# The lock's cuDNN, its requirement line and its hashes, as one entry of a requirements file.
awk '/^nvidia-cudnn-cu12==/ { taken = 1 } taken { print; if ($0 !~ /\\$/) exit }' locks/cuda12.txt \
  > "$beside/cudnn.txt"
[ -s "$beside/cudnn.txt" ] || { echo "derived: locks/cuda12.txt pins no nvidia-cudnn-cu12"; exit 1; }
uv pip install -q --no-deps --require-hashes --python "$python" --target "$beside/site" -r "$beside/cudnn.txt"
PYTHONPATH="$beside/site${PYTHONPATH:+:$PYTHONPATH}" "$python" tools/derived_catalogue.py \
  --generations "$generation" --out "$out"
