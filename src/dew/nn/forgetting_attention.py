"""The Forgetting Transformer's attention (FoX, arXiv 2503.02130): a head's forget gate `f = sigmoid(W x)`
a token, each logit `+ G_i - G_j` with `G` the gate's cumulative log, under causality, so a key fades with
the gates between it and the query. The prefix sum is a cross-token reduction, taken in XLA before the
call; the attention is `scaled_dot_product_attention`'s under the pair as a separable bias
(`SeparableBias`: the template reads a query and a key score parameter, every other route the bias written
out), whose gradients are the prefix sum's cotangents, which XLA transposes (docs/design/kernel-templates.md
§A row 52)."""

import jax
import jax.numpy as jnp

from dew.nn.attention import AttentionImpl, SeparableBias, scaled_dot_product_attention


def forget_prefix(forget_logits: jax.Array) -> jax.Array:
    """`G` `[B, H, S]`: each head's cumulative log forget gate, `cumsum(log sigmoid(forget_logits))` along the
    sequence of `forget_logits` `[B, S, H]`, in fp32 at least (fp64 for fp64 gates)."""
    wide = forget_logits.astype(jnp.promote_types(forget_logits.dtype, jnp.float32))
    return jnp.cumsum(jax.nn.log_sigmoid(wide), axis=1).transpose(0, 2, 1)


def forgetting_attention(query: jax.Array, key: jax.Array, value: jax.Array, forget_logits: jax.Array, *,
                         segment_ids: jax.Array | None = None,
                         implementation: AttentionImpl = "auto") -> jax.Array:
    """FoX's causal attention of `query` `[B, S, H, D]` over `key` and `value` `[B, S, Hk, D]`, each logit
    `+ G_i - G_j` (`forget_prefix` of `forget_logits` `[B, S, H]`, the query heads' gates), packed documents
    kept apart by `segment_ids` `[B, S]` where given (a document's prefix its own: the gates before its first
    token cancel in `G_i - G_j`), on the route `implementation` names (`scaled_dot_product_attention`).
    `[B, S, H, Dv]` out, differentiable in every input."""
    prefix = forget_prefix(forget_logits)
    return scaled_dot_product_attention(query, key, value, causal=True, bias=SeparableBias(prefix, -prefix),
                                        segment_ids=segment_ids, implementation=implementation)
