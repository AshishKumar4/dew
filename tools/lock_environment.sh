#!/bin/sh
# The test suite's environment, locked: pyproject.toml's ci and tutorials extras under constraints.txt,
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
# Dew's own FlashAttention wheels resolve as they stood when 0.3.8 was published, past the pins the
# locks hold (-P); nothing else moves.
FLASH=2026-11-06T20:53:00Z
# jaxlib and its CUDA plugins resolve as they stood when 0.12.0.1 was published, which the jax build Dew
# carries its patches on pins exactly; nothing else moves.
JAXLIB=2026-10-21T22:00:00Z
# The agent environments' SDKs resolve as they stood when their adapters were written: OpenEnv
# 0.9.0 and Daytona 0.225.0 with the API clients it pins to its own version.
ENVIRONMENTS=2026-10-17T00:00:00Z
# certifi resolves past RESOLVED to 2026.10.19, the first without the emSign root whose intermediates'
# keys were compromised (constraints.txt holds that floor); nothing else moves.
CERTIFI=2026-10-20T03:00:00Z
# Every package Dew's package index (constraints.txt's extra index) serves its own builds of, as its root page
# lists them (the fork's patch_stack.py writes it: jax, qwix), takes no date: the index's static pages give no
# upload times, which a date would need. Read at each lock, so a package the index takes on resolves with it,
# and none of the dated lists below dates one (both at once filtered every version of the CUDA plugins once the
# index took them: ledger 3116).
INDEX=$(sed -n 's/^--extra-index-url //p' constraints.txt)
UNDATED=$(curl -fsS "$INDEX/" | sed -n 's|.*<a href="\([^/"]*\)/">.*|\1|p')
[ -n "$UNDATED" ] || { echo "Dew's package index $INDEX lists no package" >&2; exit 1; }
# Each package of a dated list the index does not serve (`UNDATED`), with its date: `--exclude-newer-package
# NAME=DATE -P NAME`.
dated() {
  date=$1
  shift
  for package in "$@"; do
    case " $(echo $UNDATED) " in *" $package "*) ;; *) echo --exclude-newer-package "$package=$date" -P "$package" ;; esac
  done
}
# resolve <python> <output> <uv arguments>: pyproject.toml under constraints.txt, as the locks resolve it.
resolve() {
  python=$1 output=$2
  shift 2
  uv pip compile -q --no-header pyproject.toml -c constraints.txt \
    --exclude-newer "$RESOLVED" --exclude-newer-package "dew-flash-attn-cu12=$FLASH" \
    --exclude-newer-package "dew-flash-attn-cu13=$FLASH" -P dew-flash-attn-cu12 -P dew-flash-attn-cu13 \
    $(for package in $UNDATED; do echo --exclude-newer-package "$package=false"; done) \
    --exclude-newer-package "certifi=$CERTIFI" -P certifi --emit-index-url \
    $(dated "$JAXLIB" jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt libtpu) \
    $(for package in openenv daytona daytona-api-client daytona-api-client-async daytona-toolbox-api-client \
        daytona-toolbox-api-client-async daytona-analytics-api-client daytona-analytics-api-client-async; do
      echo --exclude-newer-package "$package=$ENVIRONMENTS"; done) \
    --python-version "$python" "$@" -o "$output"
}
lock() {
  name=$1
  shift
  resolve "$(tr -d '\n' < .python-version)" "locks/$name.txt" --extra ci --extra tutorials "$@"
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
# Each GPU extra installs on every Python pyproject.toml admits, from requires-python's floor to its newest
# classifier, on each machine: a CUDA plugin built for fewer Pythons fails here, at the pin, and not at a user's
# install (row 3298: post42's plugins came for 3.12 and 3.14 alone, and dewml[cuda12] refused 3.13).
floor=$(sed -n 's/^requires-python = ">=3\.\([0-9]*\)"$/\1/p' pyproject.toml)
newest=$(sed -n 's/.*"Programming Language :: Python :: 3\.\([0-9]*\)".*/\1/p' pyproject.toml | sort -n | tail -1)
[ -n "$floor" ] && [ -n "$newest" ] || { echo "pyproject.toml names no requires-python floor or Python classifier" >&2; exit 1; }
checked=$(mktemp)
trap 'rm -f locks/cuda12-aarch64.txt locks/cuda13-aarch64.txt "$checked"' EXIT
for minor in $(seq "$floor" "$newest"); do
  for extra in cuda12 cuda13; do
    for machine in x86_64 aarch64; do
      : > "$checked"
      resolve "3.$minor" "$checked" --extra "$extra" --torch-backend cpu --python-platform "$machine-manylinux_2_35" || {
        echo "dewml[$extra] does not install on Python 3.$minor for $machine: build its plugin for that Python" >&2
        exit 1; }
    done
  done
done
