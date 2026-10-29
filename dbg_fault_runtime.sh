#!/bin/bash
set -u
nvidia-smi -q > "$OUT/gpu.txt"
export JAX_PLATFORMS=cuda PYTHONPATH=src
export CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_SHOW_PROGRESS=1
for variant in nosetmax nograph nosetmax_nograph nosetmax_fullcore; do
  for repeat in 1 2 3 4 5 6; do
    arm="$variant-$repeat"
    export CUDA_COREDUMP_GENERATION_FLAGS=skip_global_memory,skip_local_memory,skip_abort
    if test "$arm" = nosetmax_fullcore-1; then
      export CUDA_COREDUMP_GENERATION_FLAGS=skip_local_memory,skip_abort
    fi
    export CUDA_COREDUMP_FILE="$OUT/$arm.nvcudmp"
    timeout 600 python -u dbg_fault_probe.py "$variant" > "$OUT/$arm.txt" 2>&1
    echo "$arm $?" >> "$OUT/exits.txt"
  done
done
true
