#!/bin/sh
# The test suite's environment, locked: pyproject.toml's ci, tutorials and mutation extras under constraints.txt,
# resolved as PyPI stood at RESOLVED, so the same inputs always give the same locks. Torch is the CPU
# build in every lock: Dew runs torch only on the host (interop, vision), and a CUDA torch built for
# another CUDA loads its own cuDNN over JAX's. Dew's CUDA extra is the locks' one difference, by driver: locks/cpu.txt for CI and every host
# without a GPU, for each Python and platform; locks/cuda12.txt for a driver below 580 (CUDA 12.8: the
# A100 nodes, Colab's T4 and L4) and locks/cuda13.txt for 580 on (the H100 SXM nodes, Colab's G4), for the
# Python 3.12 of a Linux GPU host. A host installs the one its driver reads: tools/install_environment.sh.
# Usage, from the checkout: sh tools/lock_environment.sh
set -eu
# Every lock resolves for the oldest supported Python, so the interpreter running this changes nothing.
RESOLVED=2026-10-12T00:00:00Z
# Dew's own FlashAttention wheels resolve as they stood when 0.3.4 was published, and jaxlib and its CUDA
# plugins as they stood when 0.12.0 was, past the pins the locks hold (-P); nothing else moves. jax itself,
# which constraints.txt pins to one build of Dew's package index, takes no date: that index's static pages
# give no upload times, which a date would need.
FLASH=2026-10-17T18:46:06Z
JAX=2026-10-19T10:40:00Z
JAXLIB="jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt"
lock() {
  name=$1
  shift
  newer=""
  for package in $JAXLIB; do
    newer="$newer --exclude-newer-package $package=$JAX -P $package"
  done
  # $newer splits into its words.
  # shellcheck disable=SC2086
  uv pip compile -q --no-header pyproject.toml --extra ci --extra tutorials --extra mutation -c constraints.txt \
    --exclude-newer "$RESOLVED" --exclude-newer-package "dew-flash-attn-cu12=$FLASH" \
    --exclude-newer-package "dew-flash-attn-cu13=$FLASH" -P dew-flash-attn-cu12 -P dew-flash-attn-cu13 $newer \
    --exclude-newer-package jax=false \
    --emit-index-url --python-version "$(tr -d '\n' < .python-version)" "$@" -o "locks/$name.txt"
}
lock cpu --universal --torch-backend cpu
gpu="--python-platform x86_64-manylinux_2_35"
lock cuda12 --extra cuda12 --torch-backend cpu $gpu
lock cuda13 --extra cuda13 --torch-backend cpu $gpu
