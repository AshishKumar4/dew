"""The Forgetting Transformer's attention (FoX, arXiv 2503.02130): a head's forget gate `f = sigmoid(W x)`
a token, each logit `+ G_i - G_j` with `G` the gate's cumulative log, under causality, so a key fades with
the gates between it and the query. The prefix sum is a cross-token reduction, taken in XLA before the
call; the attention runs on the template (`attention_route.attend`) under its score hook
(`Forgetting`), a query and a key parameter each, whose gradients are the prefix sum's
cotangents, which XLA transposes (docs/design/kernel-templates.md §A row 52)."""

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp

from dew.nn.kernels import tuning
from dew.nn.kernels.templates import attention as template, attention_route


@dataclasses.dataclass(frozen=True)
class Forgetting(tuning.FieldHook):
    """FoX's logits as the template's score hook: `score_mod`'s (the scaled logits where None) plus the
    query's forget-gate prefix and minus the key's, `G_i - G_j`, its parameters the query's `[q, 1]` (a
    `Kind.QUERY` one) and the key's `[1, k]` (a `Kind.KEY` one): equal for equal inner hooks
    (`tuning.FieldHook`)."""

    score_mod: Callable | None = None

    def __call__(self, logits: jax.Array, q_pos: jax.Array, k_pos: jax.Array, gate_q: jax.Array,
                 gate_k: jax.Array) -> jax.Array:
        capped = logits if self.score_mod is None else self.score_mod(logits, q_pos, k_pos)
        return capped + gate_q - gate_k


def forget_prefix(forget_logits: jax.Array) -> jax.Array:
    """`G` `[B, H, S]`: each head's cumulative log forget gate, `cumsum(log sigmoid(forget_logits))` along the
    sequence of `forget_logits` `[B, S, H]`, in fp32 at least (fp64 for fp64 gates)."""
    wide = forget_logits.astype(jnp.promote_types(forget_logits.dtype, jnp.float32))
    return jnp.cumsum(jax.nn.log_sigmoid(wide), axis=1).transpose(0, 2, 1)


def forgetting_attention(query: jax.Array, key: jax.Array, value: jax.Array, forget_logits: jax.Array, *,
                         segment_ids: jax.Array | None = None) -> jax.Array:
    """FoX's causal attention of `query` `[B, S, H, D]` over `key` and `value` `[B, S, Hk, D]`, each logit
    `+ G_i - G_j` (`forget_prefix` of `forget_logits` `[B, S, H]`, the query heads' gates), packed documents
    kept apart by `segment_ids` `[B, S]` where given (a document's prefix its own: the gates before its first
    token cancel in `G_i - G_j`). `[B, S, H, Dv]` out, differentiable in every input."""
    batch, length = query.shape[:2]
    prefix = forget_prefix(forget_logits)
    spans = template.spans(length, length, batch, causal=True, segment_ids=segment_ids)
    hooks = template.Hooks(Forgetting(), (template.Param(prefix, template.Kind.QUERY),
                                                   template.Param(prefix, template.Kind.KEY)))
    return attention_route.attend(query, key, value, spans, hooks)
