#!/bin/sh
# Builds dist/dew_cutlass_cu<major>-<version>-py3-none-manylinux_2_28_<machine>.whl, the machine x86_64 or
# aarch64 as the build's is (the release workflow builds both), with the CUDA toolkit on PATH
# (nvcc, 12.8 or newer for sm100a), cmake >= 3.24, ninja and a Python with pip, on glibc 2.28 (the release
# workflow's nvidia/cuda rockylinux8 image). XLA's FFI headers come from the jaxlib PYTHON installs, the
# oldest Dew supports.
set -eu
CUTLASS=https://github.com/NVIDIA/cutlass
CUTLASS_COMMIT=2c85f497547eb441f0fe80bfb0fb2e6f9add1719  # v4.8.1
here=$(cd "$(dirname "$0")" && pwd)
python=${PYTHON:-python3}
major=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')
machine=$(uname -m)
if [ ! -d "$here/upstream" ]; then
  git clone --quiet --filter=blob:none "$CUTLASS" "$here/upstream"
  git -C "$here/upstream" checkout --quiet "$CUTLASS_COMMIT"
fi
"$python" -m pip install --quiet -r "$here/../../tools/pip.txt"
# The oldest jaxlib Dew supports: a handler built against newer FFI headers states an API version an older
# jaxlib refuses to register (0.12's 0.5 against 0.11.2's 0.4), and every newer jaxlib accepts the older.
"$python" -m pip install --quiet "jaxlib==0.11.2"
headers=$("$python" -c "import jax.ffi; print(jax.ffi.include_dir())" 2>/dev/null ||
          "$python" -c "import jaxlib, os; print(os.path.join(os.path.dirname(jaxlib.__file__), 'include'))")
cmake -S "$here" -B "$here/build" -G Ninja -DCMAKE_BUILD_TYPE=Release -DXLA_FFI_INCLUDE="$headers"
cmake --build "$here/build" --parallel "${JOBS:-4}"
stage=$(mktemp -d)
cp -r "$here/dew_cutlass" "$here/pyproject.toml" "$here/README.md" "$stage/"
cp "$here/build/libdew_cutlass.so" "$stage/dew_cutlass/"
cp "$here/upstream/LICENSE.txt" "$stage/LICENSE.cutlass"
cp "$here/LICENSE.openxla" "$stage/LICENSE.openxla"
cp "$here/../../LICENSE" "$stage/LICENSE"
sed -i "s/dew-cutlass-cuXX/dew-cutlass-cu$major/" "$stage/pyproject.toml"
"$python" -m pip wheel --quiet --no-deps --wheel-dir "$stage/wheel" "$stage"
"$python" -m pip install --quiet "wheel>=0.43"
"$python" -m wheel tags --remove --python-tag py3 --abi-tag none --platform-tag "manylinux_2_28_$machine" \
  "$stage"/wheel/*.whl
mkdir -p "$here/dist"
mv "$stage"/wheel/*"manylinux_2_28_$machine.whl" "$here/dist/"
rm -rf "$stage"
ls -l "$here/dist"
