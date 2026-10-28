#!/bin/sh
# The test suite's environment, locked: pyproject.toml's ci, tutorials and mutation extras under constraints.txt,
# resolved as PyPI stood at RESOLVED, so the same inputs always give the same locks. Torch is the CPU
# build in every lock: Dew runs torch only on the host (interop, vision), and a CUDA torch built for
# another CUDA loads its own cuDNN over JAX's. Dew's CUDA extra is the locks' one difference, by driver: locks/cpu.txt for CI and every host
# without a GPU, for each Python and platform; locks/cuda12.txt for a driver below 580 (CUDA 12.8: the
# A100 nodes, Colab's T4 and L4) and locks/cuda13.txt for 580 on (the H100 SXM nodes, Colab's G4), for the
# Python 3.12 of a Linux GPU host, x86_64 or aarch64 (a GH200's Grace), whose pins are the same. A host installs
# the one its driver reads: tools/install_environment.sh.
# Usage, from the checkout: sh tools/lock_environment.sh
set -eu
# Every lock resolves for the oldest supported Python, so the interpreter running this changes nothing.
RESOLVED=2026-10-12T00:00:00Z
# Dew's own FlashAttention wheels resolve as they stood when 0.3.6 was published, past the pins the
# locks hold (-P); nothing else moves. jax itself, which constraints.txt pins to one build of Dew's package
# index, and qwix, which the quantization extra pins to another, take no date: that index's static pages give
# no upload times, which a date would need.
FLASH=2026-10-27T07:29:32Z
# jaxlib and its CUDA plugins resolve as they stood when 0.12.0.1 was published, which the jax build Dew
# carries its patches on pins exactly; nothing else moves.
JAXLIB=2026-10-21T22:00:00Z
# The agent environments' SDKs resolve as they stood when their adapters were written: OpenEnv
# 0.9.0 and Daytona 0.225.0 with the API clients it pins to its own version.
ENVIRONMENTS=2026-10-17T00:00:00Z
# certifi resolves past RESOLVED to 2026.10.19, the first without the emSign root whose intermediates'
# keys were compromised (constraints.txt holds that floor); nothing else moves.
CERTIFI=2026-10-20T03:00:00Z
lock() {
  name=$1
  shift
  uv pip compile -q --no-header pyproject.toml --extra ci --extra tutorials --extra mutation -c constraints.txt \
    --exclude-newer "$RESOLVED" --exclude-newer-package "dew-flash-attn-cu12=$FLASH" \
    --exclude-newer-package "dew-flash-attn-cu13=$FLASH" -P dew-flash-attn-cu12 -P dew-flash-attn-cu13 \
    --exclude-newer-package jax=false --exclude-newer-package qwix=false \
    --exclude-newer-package "certifi=$CERTIFI" -P certifi --emit-index-url \
    $(for package in jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt libtpu; do
      echo --exclude-newer-package "$package=$JAXLIB" -P "$package"; done) \
    $(for package in openenv daytona daytona-api-client daytona-api-client-async daytona-toolbox-api-client \
        daytona-toolbox-api-client-async daytona-analytics-api-client daytona-analytics-api-client-async; do
      echo --exclude-newer-package "$package=$ENVIRONMENTS"; done) \
    --python-version "$(tr -d '\n' < .python-version)" "$@" -o "locks/$name.txt"
}
# A host with no GPU compiles kernels for one with the CUDA plugin of the kernel-compile extra; a GPU host's
# own CUDA extra holds its plugin, and two CUDA plugins in one environment would both claim 'cuda'.
lock cpu --universal --torch-backend cpu --extra kernel-compile
gpu="--python-platform x86_64-manylinux_2_35"
lock cuda12 --extra cuda12 --torch-backend cpu $gpu
lock cuda13 --extra cuda13 --torch-backend cpu $gpu
# Each GPU lock serves aarch64 too: resolved for it from the lock's own pins (uv prefers an output's pins), it
# pins the same, which this holds, so a GH200 installs it.
# The aarch64 resolutions are the check's alone: gone however it ends.
trap 'rm -f locks/cuda12-aarch64.txt locks/cuda13-aarch64.txt' EXIT
for extra in cuda12 cuda13; do
  cp "locks/$extra.txt" "locks/$extra-aarch64.txt"
  lock "$extra-aarch64" --extra "$extra" --torch-backend cpu --python-platform aarch64-manylinux_2_35
  diff "locks/$extra.txt" "locks/$extra-aarch64.txt" >&2 || {
    echo "locks/$extra.txt resolves otherwise for aarch64: lock it for each machine" >&2; exit 1; }
done
