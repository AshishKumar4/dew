#!/bin/sh
# The test suite's environment, locked: pyproject.toml's ci and tutorials extras under constraints.txt,
# resolved as PyPI stood at RESOLVED, so the same inputs always give the same locks; CI regenerates them and
# fails on a difference. The torch build is their one difference: locks/cpu.txt holds CPU torch, for CI and
# every host whose JAX owns the GPU (tools/install_host_torch.sh's rule), and locks/cuda13.txt CUDA 13
# torch, for a driver-580 host that runs the fla reference in Dew's own environment. Usage, from the
# checkout: sh tools/lock_environment.sh
set -eu
RESOLVED=2026-10-12T00:00:00Z
lock() {
  uv pip compile -q --no-header pyproject.toml --extra ci --extra tutorials -c constraints.txt \
    --exclude-newer "$RESOLVED" "$@"
}
lock --universal --torch-backend cpu -o locks/cpu.txt
lock --python-version 3.12 --python-platform x86_64-manylinux_2_35 --torch-backend cu130 -o locks/cuda13.txt
