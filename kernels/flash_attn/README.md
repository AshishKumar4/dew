# dew-flash-attn

FlashAttention-2 for JAX, built for [Dew](https://github.com/AshishKumar4/dew).
It packages Dao-AILab's CUDA kernels through flash_attn_jax's XLA FFI port, both
BSD-3, as a library with no Python ABI that registers two FFI targets with JAX.

There is one wheel for each CUDA major of your jax, for Linux x86_64 and any
Python from 3.11:

    pip install dew-flash-attn-cu12   # jax with jax-cuda12-plugin
    pip install dew-flash-attn-cu13   # jax with jax-cuda13-plugin

With it installed, Dew's `attention_impl="auto"` runs FlashAttention-2
on an A100 (sm80) for calls without a window, mask, bias or softcap and with
heads up to 256 wide. `attention_impl="flash"` asks for it on any sm8x or sm120 GPU.

The kernels are compiled for sm80, which every compute capability 8.x runs, and for sm120, Blackwell's
workstation and consumer parts (the RTX PRO 6000 and the RTX 50 series).

Called directly:

```python
from dew_flash_attn import flash_mha

out = flash_mha(query, key, value, is_causal=True)  # [B, S, H, D], bf16 or fp16
```

Key heads may divide the query's (grouped-query attention). The backward pass
is the kernel's own.
