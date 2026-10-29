#!/bin/bash
set -u
nvidia-smi -q > "$OUT/gpu.txt"
export JAX_PLATFORMS=cuda PYTHONPATH=src CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1
export JAX_ENABLE_COMPILATION_CACHE=false
export CUDA_COREDUMP_GENERATION_FLAGS=skip_global_memory,skip_local_memory,skip_abort
for variant in nosetmax nosetmax_coordzero coordzero; do
  for repeat in 1 2 3 4 5 6; do
    arm="$variant-$repeat"
    export CUDA_COREDUMP_FILE="$OUT/$arm.nvcudmp"
    if test "$repeat" = 1; then
      mkdir -p "$OUT/dump-$variant"
      MOSAIC_GPU_DUMP_TO="$OUT/dump-$variant" MOSAIC_GPU_DUMP_PTX=1 MOSAIC_GPU_DUMP_SASS=1 MOSAIC_GPU_DUMP_PTXAS=1 timeout 600 python -u dbg_fault_probe.py "$variant" > "$OUT/$arm.txt" 2>&1
    else
      timeout 600 python -u dbg_fault_probe.py "$variant" > "$OUT/$arm.txt" 2>&1
    fi
    echo "$arm $?" >> "$OUT/exits.txt"
  done
done
true
