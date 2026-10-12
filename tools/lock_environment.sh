#!/bin/sh
# The test suite's environment, locked: pyproject.toml's ci and tutorials extras under constraints.txt,
# resolved as PyPI stood at RESOLVED, so the same inputs always give the same locks. The torch build and
# Dew's CUDA extra are their one difference, by the host's driver: locks/cpu.txt for CI and every host
# without a GPU, for each Python and platform; locks/cuda12.txt for a driver below 580 (CUDA 12.8: the
# A100 nodes, Colab's T4 and L4) and locks/cuda13.txt for 580 on (the H100 SXM nodes, Colab's G4), for the
# Python 3.12 of a Linux GPU host. A host installs the one its driver reads with `uv pip sync`.
# Usage, from the checkout: sh tools/lock_environment.sh
set -eu
RESOLVED=2026-10-12T00:00:00Z
lock() {
  name=$1
  shift
  uv pip compile -q --no-header pyproject.toml --extra ci --extra tutorials -c constraints.txt \
    --exclude-newer "$RESOLVED" "$@" -o "locks/$name.txt"
}
lock cpu --universal --torch-backend cpu
gpu="--python-version 3.12 --python-platform x86_64-manylinux_2_35"
lock cuda12 --extra cuda12 --torch-backend cu128 $gpu
lock cuda13 --extra cuda13 --torch-backend cu130 $gpu
