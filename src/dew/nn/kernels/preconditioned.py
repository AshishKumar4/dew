"""The preconditioned delta rules (rows 1458, 1459; fla's `chunk_precond_gated_delta_rule` and
`chunk_precond_kda`, fla/ops/precond_gated_delta_rule and fla/ops/precond_kda): the gated delta rule, its
decay a head's (PGDN) or a key dimension's (PKDA), writing along keys preconditioned by ATK's running second
moment of the keys and erasing along the keys themselves,

    A_t = exp(g'_t) A_{t-1} + beta'_t k_t^2,    p_t = k_t x^{-s_t},    s = r / (1 + |r|),
    r = log(A_t + eps) - c,    S_t = D_t S_{t-1} + beta_t p_t (v_t - k_t^T D_t S_{t-1})^T,    o_t = q_t^T S_t,

`A` a key head's vector over its key dimensions (shared by its value heads), `c` a key head's learned center
(fla's `log_atk_scale`, -0.2 by default), `x` the squash's range (1.5) and the queries and keys l2-normalized
first, as fla's in-kernel norm takes them. ATK's scan is the diagonal recurrence of every channel decaying
alone (`diagonal.diagonal_scan`); the delta rule's write along `p` and erase along `k` the chunked op's untied
erase (`recurrence.Hooks.erase` 'untied': its erase vector the write key `p`, its read `k`). A packed
document's first token enters both states at zero."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from dew.nn.precision import at_least_fp32, l2norm

from . import recurrence
from .diagonal import diagonal_scan
from .recurrence_chunk import CHUNK_SIZE, Chunk, chunk

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
    r = jnp.log(moment.astype(work).reshape(key.shape) + eps) - jnp.asarray(center, work)[:, None]
    squash = r / (1.0 + jnp.abs(r))
    return k * jnp.exp(-np.log(x) * squash), final.reshape(rows, heads, width)


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
    hooks = recurrence.Hooks(decay='vector' if log_decay.ndim == 4 else 'scalar', erase='untied',
                             scale=width ** -0.5 if scale is None else scale)
    out, final = chunk(hooks, Chunk(query, written, value, log_decay=log_decay, beta=beta, erase_key=written,
                                    erase_read=key, state=state, starts=starts), chunk_size=chunk_size)
    return out, (final, moment)
