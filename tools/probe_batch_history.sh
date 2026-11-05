#!/bin/sh
set -eu
probe_directory=$(mktemp -d)
git archive "$1" src | tar -x -C "$probe_directory"
PYTHONPATH="$probe_directory/src" DEW_WIDE_CPU_ARITHMETIC=0 JAX_PLATFORMS=cpu \
  .venv-3.12/bin/python tools/probe_batch_invariance.py
