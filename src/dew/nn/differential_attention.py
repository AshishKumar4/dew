"""Differential attention (arXiv 2410.05258; kernel-templates.md §A row 53): `softmax(Q1 K1ᵀ) V - λ softmax(Q2
K2ᵀ) V`, a head's `λ` a scalar, as two calls of the template over one value composed in XLA, the design's
cheaper route (one call holding two softmax states over one V stream is its open lever)."""

import jax

from dew.nn.attention import AttentionImpl, scaled_dot_product_attention


def differential_attention(queries: tuple[jax.Array, jax.Array], keys: tuple[jax.Array, jax.Array],
                           value: jax.Array, lambdas: jax.Array, *, causal: bool = True,
                           segment_ids: jax.Array | None = None,
                           implementation: AttentionImpl = "auto") -> jax.Array:
    """`attend(Q1, K1, V) - λ attend(Q2, K2, V)` (`scaled_dot_product_attention`, each its own softmax),
    `queries` `[B, S, H, D]` each, `keys` `[B, S, Hk, D]` each, `value` `[B, S, Hk, Dv]`, `lambdas` `[H]` a
    head's `λ`: `[B, S, H, Dv]` out, differentiable in every input."""
    first, second = (scaled_dot_product_attention(query, key, value, causal=causal, segment_ids=segment_ids,
                                                  implementation=implementation)
                     for query, key in zip(queries, keys, strict=True))
    return first - lambdas[None, None, :, None].astype(second.dtype) * second
