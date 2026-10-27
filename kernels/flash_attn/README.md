# dew-flash-attn

FlashAttention for JAX, built for [Dew](https://github.com/AshishKumar4/dew).
It packages Dao-AILab's CUDA kernels as libraries with no Python ABI that
register XLA FFI targets with JAX: FlashAttention-2 through flash_attn_jax's
port, and FlashAttention-3, Hopper's kernels, through flash-attn3-jax's
torch-free port, all BSD-3.

There is one wheel for each CUDA major of your jax, for Linux x86_64 and
aarch64 and any Python from 3.11:

    pip install dew-flash-attn-cu12   # jax with jax-cuda12-plugin
    pip install dew-flash-attn-cu13   # jax with jax-cuda13-plugin

The wheel ships no CUDA runtime: it needs a CUDA build of JAX of the same major
in the same environment (`jax[cuda12]` or `jax[cuda13]`), whose libcudart it
loads.

With it installed, Dew's `attention_impl="auto"` runs FlashAttention
on an A100 (sm80), an Ada GPU (sm89) and an H100 (sm90) for calls without a mask, bias or softcap,
sliding windows and attention sinks included, with heads up to 256 wide. `attention_impl="flash"` asks for it on any sm8x, sm90 or sm120 GPU.

FlashAttention-2 is compiled for sm80, which every compute capability 8.x runs, and for sm120,
Blackwell's workstation and consumer parts (the RTX PRO 6000 and the RTX 50 series).
FlashAttention-3 is compiled for sm90a (H100, H200), for bf16 and fp16 heads of 64, 128, 192 and
256 (a narrower head is padded up to the next), with variable lengths and sliding windows and
without split keys, paged caches, softcap or fp8. The same two functions run whichever the GPU has.

Called directly:

```python
from dew_flash_attn import flash_mha

out = flash_mha(query, key, value, is_causal=True)  # [B, S, H, D], bf16 or fp16
```

Key heads may divide the query's (grouped-query attention). The backward pass
is the kernel's own.

A sliding window and attention sinks, one learned logit a query head in each
row's softmax denominator with no value (gpt-oss, MiMo-V2), go to either function:

```python
out = flash_mha(query, key, value, is_causal=True, window_size=(127, 0), sinks=sinks)  # sinks [H]
```

`window_size` `(left, right)` keeps the `left` keys before each query and the
`right` after it, -1 bounding neither side. The sinks scale the kernel's output
by each row's share of the softmax, `sigmoid(lse - sink)`, and take their
gradient from the kernel's own backward.

Packed sequences, `[T, H, D]` with int32 offsets `cu_seqlens` `[N + 1]` from 0 to
T, attend each within itself:

```python
from dew_flash_attn import flash_mha_varlen

out = flash_mha_varlen(query, key, value, cu_seqlens, max_seqlen, is_causal=True)
```

Keys packed by offsets of their own, `key_cu_seqlens` `[N + 1]` over the keys' `[T_k, H_k, D]`
with at most `key_max_seqlen` a sequence, pair sequence i's queries with sequence i's keys alone:
cross-attention between packed documents, such as a head's slots reading their own document's
tokens.

```python
out = flash_mha_varlen(query, key, value, cu_seqlens, max_seqlen,
                       key_cu_seqlens=key_cu_seqlens, key_max_seqlen=key_max_seqlen)
```

The packed backward launches its CTAs over the batch's blocks, not over `N` times
`max_seqlen`'s: `patches/` changes both ports' backward kernels so that a sequence's blocks start
at `cu_seqlens[i] / block + i` and a CTA finds its sequence by a binary search, so a batch of many
empty or short sequences costs what its tokens do. The patches apply to the pinned checkouts
(build.sh) and are written to go upstream.
