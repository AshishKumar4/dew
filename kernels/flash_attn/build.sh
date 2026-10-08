#!/bin/sh
# Builds dist/dew_flash_attn_cu<major>-<version>-py3-none-manylinux_2_28_x86_64.whl
# with the CUDA toolkit on PATH (nvcc), cmake >= 3.24, ninja and a Python with
# pip, on glibc 2.28 (the release workflow's nvidia/cuda rockylinux8 image).
set -eu
FLASH_ATTN_JAX=https://github.com/nshepperd/flash_attn_jax
FLASH_ATTN_JAX_COMMIT=24aeb952406a59bd1f2697811f58973d06c2a6e8
here=$(cd "$(dirname "$0")" && pwd)
python=${PYTHON:-python3}
major=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
if [ ! -d "$here/upstream" ]; then
  git clone --quiet "$FLASH_ATTN_JAX" "$here/upstream"
  git -C "$here/upstream" checkout --quiet "$FLASH_ATTN_JAX_COMMIT"
  git -C "$here/upstream" submodule update --quiet --init --depth 1 csrc/cutlass
fi
cmake -S "$here" -B "$here/build" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$here/build" --parallel "${JOBS:-4}"
stage=$(mktemp -d)
cp -r "$here/dew_flash_attn" "$here/pyproject.toml" "$stage/"
cp "$here/build/libdew_flash_attn.so" "$stage/dew_flash_attn/"
cp "$here/upstream/LICENSE" "$stage/LICENSE"
sed -i "s/dew-flash-attn-cuXX/dew-flash-attn-cu$major/" "$stage/pyproject.toml"
"$python" -m pip wheel --quiet --no-deps --wheel-dir "$stage/wheel" "$stage"
"$python" -m pip install --quiet "wheel>=0.43"
"$python" -m wheel tags --remove --python-tag py3 --abi-tag none --platform-tag manylinux_2_28_x86_64 \
  "$stage"/wheel/*.whl
mkdir -p "$here/dist"
mv "$stage"/wheel/*manylinux_2_28_x86_64.whl "$here/dist/"
rm -rf "$stage"
ls -l "$here/dist"
