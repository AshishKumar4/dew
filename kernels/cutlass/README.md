# dew-cutlass

CUTLASS's grouped GEMMs for JAX, built for [Dew](https://github.com/AshishKumar4/dew): NVIDIA's CUDA kernels
(BSD-3) as a library with no Python ABI that registers XLA FFI targets with JAX. Dew's mixture-of-experts layers
run them on NVIDIA GPUs where the accuracy gate and the timings show them the better route.

There is one wheel for each CUDA major of your jax, for Linux x86_64 and any Python from 3.11:

    pip install dew-cutlass-cu12   # jax with jax-cuda12-plugin
    pip install dew-cutlass-cu13   # jax with jax-cuda13-plugin

The wheel ships no CUDA runtime: it needs a CUDA build of JAX of the same major in the same environment
(`jax[cuda12]` or `jax[cuda13]`), whose libcudart it loads. It holds code for sm80 (every compute capability
8.x), sm90a (H100, H200) and sm100a (B200, B300); `dew_cutlass.ARCHITECTURES` names what the installed build
holds, and a call on any other GPU answers an error naming its capability.

Two FFI targets, each dispatching on the device's compute capability:

- `dew_cutlass_grouped_gemm`: x `[M, K]` and w `[G, K, N]` (or `[G, N, K]` with `transpose_w`), bf16 or f16, the
  rows of x grouped by `group_sizes` (s32 `[G]`, counts); out `[M, N]`, bf16, f16 or f32. Rows past the groups'
  sum are zero.
- `dew_cutlass_grouped_gemm_k`: x `[M, K]` and dy `[M, N]`, grouped along M by `group_sizes`; out `[G, K, N]`,
  f32 or bf16: each group's rows of x, transposed, times its rows of dy.

Building: `sh build.sh` with the CUDA toolkit (12.8 or newer) on PATH, cmake 3.24 or newer and ninja; the
release workflow builds in NVIDIA's rockylinux8 CUDA images (`.github/workflows/cutlass.yml`).
