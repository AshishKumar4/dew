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
# Dew's own FlashAttention wheels resolve as they stood when 0.3.4 was published, past the pins the
# locks hold (-P); nothing else moves.
FLASH=2026-10-17T18:46:06Z
# The agent environments' SDKs resolve as they stood when their adapters were written: OpenEnv
# 0.9.0 and Daytona 0.225.0 with the API clients it pins to its own version.
ENVIRONMENTS=2026-10-17T00:00:00Z
lock() {
  name=$1
  shift
  uv pip compile -q --no-header pyproject.toml --extra ci --extra tutorials --extra mutation -c constraints.txt \
    --exclude-newer "$RESOLVED" --exclude-newer-package "dew-flash-attn-cu12=$FLASH" \
    --exclude-newer-package "dew-flash-attn-cu13=$FLASH" -P dew-flash-attn-cu12 -P dew-flash-attn-cu13 \
    $(for package in openenv daytona daytona-api-client daytona-api-client-async daytona-toolbox-api-client \
        daytona-toolbox-api-client-async daytona-analytics-api-client daytona-analytics-api-client-async; do
      echo --exclude-newer-package "$package=$ENVIRONMENTS"; done) \
    --python-version "$(tr -d '\n' < .python-version)" "$@" -o "locks/$name.txt"
}
lock cpu --universal --torch-backend cpu
gpu="--python-platform x86_64-manylinux_2_35"
lock cuda12 --extra cuda12 --torch-backend cpu $gpu
lock cuda13 --extra cuda13 --torch-backend cpu $gpu
