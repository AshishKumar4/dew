"""The Forgetting Transformer's attention (FoX, arXiv 2503.02130): a head's forget gate `f = sigmoid(W x)`
a token, each logit `+ G_i - G_j` with `G` the gate's cumulative log, under causality, so a key fades with
the gates between it and the query. The prefix sum is a cross-token reduction, taken in XLA before the
call; the attention is `scaled_dot_product_attention`'s under the pair as a separable bias
(`SeparableBias`: the template reads a query and a key score parameter, every other route the bias written
out), whose gradients are the prefix sum's cotangents, which XLA transposes (docs/design/kernel-templates.md
§A row 52)."""

import jax
import jax.numpy as jnp

from dew.algebra import scan
from dew.batch import document_starts
from dew.nn.attention import AttentionImpl, SeparableBias, scaled_dot_product_attention
from dew.nn.precision import softplus


def forget_prefix(forget_logits: jax.Array, segment_ids: jax.Array | None = None) -> jax.Array:
    """`G` `[B, H, S]`: each head's cumulative log forget gate, `log sigmoid(forget_logits)` summed along the
    sequence of `forget_logits` `[B, S, H]` in fp32 at least (fp64 for fp64 gates), compensated and each
    packed document's own, restarting at its first token where `segment_ids` `[B, S]` mark documents
    (`scan.compensated`): summed across a row, an earlier document's gates of -2^30 rounded every later
    prefix to theirs, the later documents' `G_i - G_j` to 0 (row 7609's arithmetic, train 1009's F1)."""
    wide = forget_logits.astype(jnp.promote_types(forget_logits.dtype, jnp.float32))
    gates = -softplus(-wide)
    restarts = None if segment_ids is None else jnp.broadcast_to(
        document_starts(segment_ids, jnp.ones(segment_ids.shape, bool))[..., None], gates.shape)
    high, _ = scan.compensated(gates, axis=1, restarts=restarts)
    return high.transpose(0, 2, 1)


def forgetting_attention(query: jax.Array, key: jax.Array, value: jax.Array, forget_logits: jax.Array, *,
                         segment_ids: jax.Array | None = None,
                         implementation: AttentionImpl = "auto") -> jax.Array:
    """FoX's causal attention of `query` `[B, S, H, D]` over `key` and `value` `[B, S, Hk, D]`, each logit
    `+ G_i - G_j` (`forget_prefix` of `forget_logits` `[B, S, H]`, the query heads' gates), packed documents
    kept apart by `segment_ids` `[B, S]` where given (a document's prefix its own, `forget_prefix`), on the
    route `implementation` names (`scaled_dot_product_attention`).
    `[B, S, H, Dv]` out, differentiable in every input."""
    prefix = forget_prefix(forget_logits, segment_ids)
    return scaled_dot_product_attention(query, key, value, causal=True, bias=SeparableBias(prefix, -prefix),
                                        segment_ids=segment_ids, implementation=implementation)
