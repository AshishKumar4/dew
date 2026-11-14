"""DeltaFormer's attention (row 1450; fla's `deltaformer_attn`, fla/ops/deltaformer): each token's value
corrected by its earlier tokens' corrected values under a strictly causal softmax of the scores, then read by
causal softmax attention,

    u_t = v_t - beta_t sum_{j < t} P_tj u_j,   P = softmax over j < t of q_t . k_j / sqrt(D),
    o_t = sum_{j <= t} A_tj u_j,               A = softmax over j <= t of q_t . k_j / sqrt(D),

so `u = (I + diag(beta) P)^-1 v`, a unit lower-triangular solve (`deltaformer`); packed documents take each
softmax and the solve within their own tokens."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from dew.nn.documents import with_documents
from dew.nn.precision import at_least_fp32


def masked_softmax(scores, allowed):
    """A row-wise softmax over the `allowed` positions, zero elsewhere and on a row with none (fla's
    `tril_softmax`, its denominator clamped at 1e-20). A position not allowed takes no part in the row's
    maximum, whatever the allowed scores (an allowed score of -2e30 beside none is the row's whole weight),
    and its exponential is of -inf, zero with a zero slope; a row with none takes its maximum as zero."""
    peak = jnp.max(jnp.where(allowed, scores, -jnp.inf), axis=-1, keepdims=True)
    peak = jax.lax.stop_gradient(jnp.where(jnp.isneginf(peak), 0.0, peak))
    weights = jnp.exp(jnp.where(allowed, scores - peak, -jnp.inf))
    return weights / jnp.maximum(jnp.sum(weights, axis=-1, keepdims=True), 1e-20)


def causal_masks(length: int, starts):
    """The strictly causal and causal masks `[B or 1, 1, T, T]`, each within a packed document
    (`documents.with_documents`; `starts` `[B, T]`, a document's first tokens) where given."""
    tokens = jnp.arange(length)
    strict = (tokens[None, :] < tokens[:, None])[None, None]
    causal = (tokens[None, :] <= tokens[:, None])[None, None]
    if starts is None:
        return strict, causal
    # Each token's document an id from one up (zero is padding's, which sees nothing).
    segments = jnp.cumsum(jnp.asarray(starts, jnp.int32), axis=1) + 1
    return with_documents(strict, segments), with_documents(causal, segments)


def deltaformer(query, key, value, beta=None, starts=None):
    """DeltaFormer's attention, `query`, `key` and `value` `[B, T, H, D]`, `beta` `[B, T, H]` (ones where
    None), `starts` `[B, T]` each packed document's first token: its scores, softmaxes and solve in
    `at_least_fp32` of the operands' dtype, the output `[B, T, H, D]` in it."""
    dtype = jnp.result_type(query, key, value)
    work = at_least_fp32(dtype)
    batch, length, _, width = query.shape
    q, k, v = (jnp.moveaxis(jnp.asarray(t, work), 2, 1) for t in (query, key, value))
    beta = jnp.ones((batch, length, q.shape[1]), work) if beta is None else jnp.asarray(beta, work)
    scores = jnp.einsum('bhtd,bhsd->bhts', q, k) * width ** -0.5
    strict, causal = causal_masks(length, starts)
    corrections = jnp.moveaxis(beta, 2, 1)[..., None] * masked_softmax(scores, strict)
    system = jnp.eye(length, dtype=work) + corrections
    corrected = solve_triangular(system, v, lower=True, unit_diagonal=True)
    out = jnp.einsum('bhts,bhsd->bhtd', masked_softmax(scores, causal), corrected)
    return jnp.moveaxis(out, 1, 2).astype(dtype)


def recurrent_deltaformer(query, key, value, beta=None, starts=None):
    """`deltaformer` a token at a time as fla's reference states it (`naive_deltaformer_attn`): each token's
    corrected value from the earlier ones', then the causal attention over them, in at least fp32 (float64
    where the operands are); operands and result as `deltaformer`'s."""
    dtype = jnp.result_type(query, key, value)
    work = at_least_fp32(dtype)
    batch, length, heads, width = query.shape
    q, k, v = (jnp.asarray(t, work) for t in (query, key, value))
    beta = jnp.ones((batch, length, heads), work) if beta is None else jnp.asarray(beta, work)
    scores = jnp.einsum('bthd,bshd->bhts', q, k) * width ** -0.5
    strict, causal = causal_masks(length, starts)
    weights = masked_softmax(scores, strict)

    def one_token(corrected, t):
        earlier = jnp.einsum('bhs,bshd->bhd', weights[:, :, t], corrected)
        token = v[:, t] - beta[:, t, :, None] * earlier
        return corrected.at[:, t].set(token), None

    corrected, _ = jax.lax.scan(one_token, jnp.zeros_like(v), jnp.arange(length))
    out = jnp.einsum('bhts,bshd->bthd', masked_softmax(scores, causal), corrected)
    return out.astype(dtype)
