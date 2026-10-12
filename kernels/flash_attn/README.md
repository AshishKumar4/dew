# dew-flash-attn

FlashAttention for JAX, built for [Dew](https://github.com/AshishKumar4/dew).
It packages Dao-AILab's CUDA kernels as libraries with no Python ABI that
register XLA FFI targets with JAX: FlashAttention-2 through flash_attn_jax's
port, and FlashAttention-3, Hopper's kernels, through flash-attn3-jax's
torch-free port, all BSD-3.

There is one wheel for each CUDA major of your jax, for Linux x86_64 and any
Python from 3.11:

    pip install dew-flash-attn-cu12   # jax with jax-cuda12-plugin
    pip install dew-flash-attn-cu13   # jax with jax-cuda13-plugin

The wheel ships no CUDA runtime: it needs a CUDA build of JAX of the same major
in the same environment (`jax[cuda12]` or `jax[cuda13]`), whose libcudart it
loads.

With it installed, Dew's `attention_impl="auto"` runs FlashAttention
on an A100 (sm80) and an H100 (sm90) for calls without a window, mask, bias or softcap and with
heads up to 256 wide. `attention_impl="flash"` asks for it on any sm8x, sm90 or sm120 GPU.

FlashAttention-2 is compiled for sm80, which every compute capability 8.x runs, and for sm120,
Blackwell's workstation and consumer parts (the RTX PRO 6000 and the RTX 50 series).
FlashAttention-3 is compiled for sm90a (H100, H200), for bf16 and fp16 heads of 64, 128 and 256
(a narrower head is padded up to the next), with variable lengths and without split keys, paged
caches, softcap, sliding windows or fp8. The same two functions run whichever the GPU has.

Called directly:

```python
from dew_flash_attn import flash_mha

out = flash_mha(query, key, value, is_causal=True)  # [B, S, H, D], bf16 or fp16
```

Key heads may divide the query's (grouped-query attention). The backward pass
is the kernel's own.

Packed sequences, `[T, H, D]` with int32 offsets `cu_seqlens` `[N + 1]` from 0 to
T, attend each within itself:

```python
from dew_flash_attn import flash_mha_varlen

out = flash_mha_varlen(query, key, value, cu_seqlens, max_seqlen, is_causal=True)
```
