"""The preconditioned delta rules (rows 1458, 1459; fla's `chunk_precond_gated_delta_rule` and
`chunk_precond_kda`, fla/ops/precond_gated_delta_rule and fla/ops/precond_kda): the gated delta rule, its
decay a head's (PGDN) or a key dimension's (PKDA), writing along keys preconditioned by ATK's running second
moment of the keys and erasing along the keys themselves,

    A_t = exp(g'_t) A_{t-1} + beta'_t k_t^2,    p_t = k_t x^{-s_t},    s = r / (1 + |r|),
    r = log(A_t + eps) - c,    S_t = D_t S_{t-1} + beta_t p_t (v_t - k_t^T D_t S_{t-1})^T,    o_t = q_t^T S_t,

`A` a key head's vector over its key dimensions (shared by its value heads), `c` a key head's learned center
(fla's `log_atk_scale`, -0.2 by default), `x` the squash's range (1.5) and the queries and keys l2-normalized
first, as fla's in-kernel norm takes them. ATK's scan is the diagonal recurrence of every channel decaying
alone (`diagonal.diagonal_scan`); the delta rule's write along `p` corrected along `k` the chunked op's read
erase (`recurrence.Hooks.erase` 'read': its keys `p`, its read `k`, the correction `v - k^T D S` one write).
A packed document's first token enters both states at zero."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from dew.nn.derivatives import Protected, contracted, declared_tangent
from dew.nn.precision import at_least_fp32, broadcast_summed, l2norm

from . import recurrence
from .diagonal import diagonal_scan
from .recurrence_chunk import CHUNK_SIZE, Chunk, chunk


def _squash_value(r):
    """The fast squash `r / (1 + |r|)`, bounded in (-1, 1) (the module's account)."""
    return r / (1.0 + jnp.abs(r))


def _squash_tangent(primals, tangents):
    """`_squash`'s derivative `dr / (1 + |r|)^2`, divided twice, never the quotient rule's `(1 + |r| - |r|)`,
    which cancels to zero where `1 + |r|` rounds to `|r|` (a center of -2^25: fla's ATK backward's 2^-50),
    nor squared past the range where `|r|` passes 2^64; its primal `_squash` itself, so a derivative of a
    derivative takes this rule again rather than the quotient rule's."""
    (r,), (dr,) = primals, tangents
    held = 1.0 + jnp.abs(r)
    return _squash(r), dr / held / held


_squash = contracted("test_the_squash_holds_every_order")(
    declared_tangent(Protected(), _squash_value, _squash_tangent))


def _squashed(key, moment, center, x: float, eps: float):
    """The write keys `k x^{-s}` of keys `k` and their moment (the module's account), `center` a key head's,
    spread over its keys and every token (`precision.broadcast_summed`): the centers' gradient, a key head's
    sum over every row, token and key, summed at least in float32 (float64 on a CPU), where XLA:CPU's
    reduction of a draw's 1280 terms put check 1 at 1.585 against fla's token walk."""
    logs = jnp.log(moment + eps)
    centers = broadcast_summed(jnp.asarray(center, moment.dtype), logs.shape, (logs.ndim - 2,))
    return key * jnp.exp(-np.log(x) * _squash(logs - centers))

CENTER = -0.2
"""fla's default center of ATK's squash, a key head's `log_atk_scale`."""


def preconditioned_keys(key, log_decay, beta, state=None, starts=None, *, center, x: float = 1.5,
                        eps: float = 1e-6):
    """ATK's preconditioned write keys (the module's account): `key` `[B, S, Hk, Dk]` (l2-normalized by the
    caller), `log_decay` and `beta` ATK's `[B, S, Hk]`, `state` the second moment entering `[B, Hk, Dk]`
    (zero where None), `center` `[Hk]`; `(p [B, S, Hk, Dk], final second moment [B, Hk, Dk])` in at least
    fp32, the moment scanned a channel at a time (`diagonal_scan`)."""
    rows, length, heads, width = key.shape
    work = at_least_fp32(key.dtype)
    k = key.astype(work)
    flat = (rows, length, heads * width)
    written = (beta.astype(work)[..., None] * k * k).reshape(flat)
    decay = jnp.broadcast_to(log_decay.astype(work)[..., None], key.shape).reshape(flat)
    entering = None if state is None else state.astype(work).reshape(rows, heads * width)
    moment, final = diagonal_scan(written, decay, entering, starts=starts)
    written = _squashed(k, moment.astype(work).reshape(key.shape), center, x, eps)
    return written, final.reshape(rows, heads, width)


def preconditioned_delta_rule(query, key, value, log_decay, beta, atk_decay, atk_beta, state=None,
                              atk_state=None, starts=None, *, center=None, scale: float | None = None,
                              x: float = 1.5, eps: float = 1e-6, chunk_size: int = CHUNK_SIZE):
    """The preconditioned gated delta rule (the module's account): `query` and `key` `[B, S, Hk, Dk]`,
    `value` `[B, S, Hv, Dv]` (`Hv` a multiple of `Hk`, each key head's ATK moment its value heads'),
    `log_decay` the delta rule's `[B, S, Hv]` (PGDN) or `[B, S, Hv, Dk]` (PKDA), `beta` its write gates
    `[B, S, Hv]`, `atk_decay` and `atk_beta` ATK's `[B, S, Hk]`, `state` `[B, Hv, Dk, Dv]` and `atk_state`
    `[B, Hk, Dk]` entering (zeros where None), `starts` `[B, S]` packed documents' first tokens, `center`
    `[Hk]` (`CENTER` where None), `scale` the queries' (`Dk^-1/2` where None). `(output [B, S, Hv, Dv] in
    the values' dtype, (final state, final ATK moment))`."""
    heads, width = key.shape[2], key.shape[-1]
    center = jnp.full((heads,), CENTER, jnp.float32) if center is None else center
    query, key = l2norm(query), l2norm(key)
    written, moment = preconditioned_keys(key, atk_decay, atk_beta, atk_state, starts, center=center, x=x,
                                          eps=eps)
    hooks = recurrence.Hooks(decay='vector' if log_decay.ndim == 4 else 'scalar', erase='read',
                             scale=width ** -0.5 if scale is None else scale)
    out, final = chunk(hooks, Chunk(query, written, value, log_decay=log_decay, beta=beta, erase_read=key,
                                    state=state, starts=starts), chunk_size=chunk_size)
    return out.astype(value.dtype), (final, moment)


def preconditioned_step(query, key, value, log_decay, beta, atk_decay, atk_beta, state, atk_state, *,
                        center=None, scale: float | None = None, x: float = 1.5, eps: float = 1e-6):
    """A decode token of the preconditioned rule (`preconditioned_delta_rule`'s, each operand without its
    tokens' axis): ATK's moment `[rows, Hk, Dk]` decayed and written, its squashed keys the delta rule's
    write along the read of the token's own (`recurrence.Hooks.erase` 'read', `recurrence.step`).
    `(output [rows, Hv, Dv] in the values' dtype, (state', moment'))`."""
    heads, width = key.shape[1], key.shape[-1]
    center = jnp.full((heads,), CENTER, jnp.float32) if center is None else center
    query, key = l2norm(query), l2norm(key)
    work = at_least_fp32(key.dtype)
    k = key.astype(work)
    moment = (jnp.exp(atk_decay.astype(work))[..., None] * atk_state.astype(work)
              + atk_beta.astype(work)[..., None] * k * k)
    written = _squashed(k, moment, center, x, eps)
    hooks = recurrence.Hooks(decay='vector' if log_decay.ndim == 3 else 'scalar', erase='read',
                             scale=width ** -0.5 if scale is None else scale)
    out, _, held = recurrence.step(hooks, recurrence.Step(state, query=query, key=written, value=value,
                                                          log_decay=log_decay, beta=beta, erase_read=key))
    return out.astype(value.dtype), (held, moment)
