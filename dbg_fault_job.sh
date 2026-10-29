#!/bin/bash
set -u
nvidia-smi -q > "$OUT/gpu.txt"
python -m pip list > "$OUT/packages.txt"
mkdir -p "$OUT/debugger"
curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-gdb-12-8_12.8.90-1_amd64.deb -o "$OUT/debugger/gdb.deb"
dpkg-deb -x "$OUT/debugger/gdb.deb" "$OUT/debugger"
GDB="$OUT/debugger/usr/local/cuda-12.8/bin/cuda-gdb"
"$GDB" --version > "$OUT/gdb-version.txt" 2>&1
export JAX_PLATFORMS=cuda PYTHONPATH=src
export CUDA_COREDUMP_ON_EXCEPTION=1 CUDA_COREDUMP_SHOW_PROGRESS=1
export CUDA_COREDUMP_GENERATION_FLAGS=skip_global_memory,skip_shared_memory,skip_local_memory
for variant in base consumers2 stages1 stages3 noresiduals causal d64 d192 d256 nosetmax serialmma; do
  for repeat in 1 2 3 4 5 6; do
    arm="$variant-$repeat"
    export CUDA_COREDUMP_FILE="$OUT/$arm.nvcudmp"
    timeout 600 python -u dbg_fault_probe.py "$variant" > "$OUT/$arm.txt" 2>&1
    code=$?
    echo "$arm $code" >> "$OUT/exits.txt"
    if test -f "$OUT/$arm.nvcudmp"; then
      "$GDB" -nx -batch -ex "target cudacore $OUT/$arm.nvcudmp" -ex "info cuda kernels" -ex "info cuda threads" -ex "info registers" -ex 'x/16i $pc-64' -ex "bt" > "$OUT/$arm-gdb.txt" 2>&1
    fi
  done
done
unset CUDA_COREDUMP_ON_EXCEPTION
SAN=$(command -v compute-sanitizer)
for tool in racecheck synccheck; do
  timeout 1800 "$SAN" --tool "$tool" --print-limit 40 python -u dbg_fault_probe.py base > "$OUT/$tool.txt" 2>&1
  echo "$tool $?" >> "$OUT/exits.txt"
done
true
