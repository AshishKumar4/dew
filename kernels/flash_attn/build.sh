#!/bin/sh
# Builds dist/dew_flash_attn_cu<major>-<version>-py3-none-manylinux_2_28_<machine>.whl,
# the machine x86_64 or aarch64 as the build's is (the release workflow builds both),
# with the CUDA toolkit on PATH (nvcc), cmake >= 3.24, ninja and a Python with
# pip, on glibc 2.28 (the release workflow's nvidia/cuda rockylinux8 image).
set -eu
FLASH_ATTN_JAX=https://github.com/nshepperd/flash_attn_jax
FLASH_ATTN_JAX_COMMIT=24aeb952406a59bd1f2697811f58973d06c2a6e8
FLASH_ATTN3_JAX=https://github.com/kyutai-labs/flash-attn3-jax
FLASH_ATTN3_JAX_COMMIT=4af680283658f467ced12a0811894a672e8d87d1
here=$(cd "$(dirname "$0")" && pwd)
python=${PYTHON:-python3}
major=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
machine=$(uname -m)
# patches/ holds Dew's changes to the two ports' kernels, applied once to a fresh checkout.
if [ ! -d "$here/upstream" ]; then
  git clone --quiet "$FLASH_ATTN_JAX" "$here/upstream"
  git -C "$here/upstream" checkout --quiet "$FLASH_ATTN_JAX_COMMIT"
  git -C "$here/upstream" submodule update --quiet --init --depth 1 csrc/cutlass
  git -C "$here/upstream" apply "$here"/patches/flash-attention-2-*.patch
fi
if [ ! -d "$here/upstream3" ]; then
  git clone --quiet "$FLASH_ATTN3_JAX" "$here/upstream3"
  git -C "$here/upstream3" checkout --quiet "$FLASH_ATTN3_JAX_COMMIT"
  git -C "$here/upstream3" submodule update --quiet --init --depth 1 csrc/cutlass
  git -C "$here/upstream3" apply "$here"/patches/flash-attention-3-*.patch
fi
cmake -S "$here" -B "$here/build" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$here/build" --parallel "${JOBS:-4}"
stage=$(mktemp -d)
cp -r "$here/dew_flash_attn" "$here/pyproject.toml" "$here/README.md" "$stage/"
cp "$here/build/libdew_flash_attn.so" "$here/build/libdew_flash_attn3.so" "$stage/dew_flash_attn/"
cp "$here/upstream/LICENSE" "$stage/LICENSE"
cp "$here/upstream3/LICENSE" "$stage/LICENSE.flash-attn3-jax"
# The libraries also carry CUTLASS's code (each port pins its own) and OpenXLA's FFI headers.
cp "$here/upstream/csrc/cutlass/LICENSE.txt" "$stage/LICENSE.cutlass"
cp "$here/upstream3/csrc/cutlass/LICENSE.txt" "$stage/LICENSE.cutlass-flash-attn3"
cp "$here/LICENSE.openxla" "$stage/LICENSE.openxla"
sed -i "s/dew-flash-attn-cuXX/dew-flash-attn-cu$major/" "$stage/pyproject.toml"
"$python" -m pip install --quiet -r "$here/../../tools/pip.txt"
"$python" -m pip wheel --quiet --no-deps --wheel-dir "$stage/wheel" "$stage"
"$python" -m pip install --quiet "wheel>=0.43"
"$python" -m wheel tags --remove --python-tag py3 --abi-tag none --platform-tag "manylinux_2_28_$machine" \
  "$stage"/wheel/*.whl
mkdir -p "$here/dist"
mv "$stage"/wheel/*"manylinux_2_28_$machine.whl" "$here/dist/"
rm -rf "$stage"
ls -l "$here/dist"
