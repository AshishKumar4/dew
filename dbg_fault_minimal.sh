#!/bin/bash
set -u
nvidia-smi -q > "$OUT/gpu.txt"
export JAX_PLATFORMS=cuda PYTHONPATH=src CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export CUDA_COREDUMP_GENERATION_FLAGS=skip_global_memory,skip_local_memory,skip_abort
for mode in plain roles; do
  for repeat in 1 2 3 4 5 6; do
    arm="tma-$mode-$repeat"
    export CUDA_COREDUMP_FILE="$OUT/$arm.nvcudmp"
    timeout 600 python -u dbg_fault_tma.py "$mode" > "$OUT/$arm.txt" 2>&1
    echo "$arm $?" >> "$OUT/exits.txt"
  done
done
for variant in swizzle64 nosetmax_swizzle64 consumers2forced; do
  for repeat in 1 2 3 4 5 6; do
    arm="$variant-$repeat"
    export CUDA_COREDUMP_FILE="$OUT/$arm.nvcudmp"
    timeout 600 python -u dbg_fault_probe.py "$variant" > "$OUT/$arm.txt" 2>&1
    echo "$arm $?" >> "$OUT/exits.txt"
  done
done
true
