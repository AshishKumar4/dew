"""Mamba-3's state space recurrence (Lahoti et al., arXiv 2603.15569), single input and single output: its
exponential-trapezoidal discretization (eqs. 5 and 6) and its complex state, each pair of state dimensions a
complex number turned by `dt_t theta_t` a token,

    h_t = e^{-i dt_t theta_t} (alpha_t h_{t-1} + beta_t B_{t-1} x_{t-1}) + gamma_t B_t x_t,
    y_t = Re(C_t^* h_t),

`alpha_t = exp(dt_t A_t)`, `beta_t = (1 - lambda_t) dt_t alpha_t`, `gamma_t = lambda_t dt_t`, a pair
`(2j, 2j + 1)` the complex number `v_2j + i v_2j+1` and the pairs past `theta`'s width real (mamba_ssm's
kernels' convention, `ops/triton/mamba3/mamba3_siso_fwd.py`, and its `rope_fraction`).

On the recurrence template's chunked op (`recurrence_chunk.chunk`), no new kernel
(docs/design/kernel-templates.md, "Mamba1, Mamba3 and Gated Oja"), as mamba_ssm's kernels take it: each
token's `B` and `C` turned by its angle `phi_t`, `dt theta` summed over its document (`rotated`), so the
state's turn between two tokens is their angles' difference, and the previous token's write folded into each
token's own. Unrolled, token `u`'s value reaches a later token's output at weight `gamma_u + delta_{u+1}`
(`delta = (1 - lambda) dt`, the next token's, zero past its document) times the decays between, and its own
at `gamma_u` alone, so the chunked op writes at `gamma_u + delta_{u+1}` and reads apart from each token's own
writes (`Hooks.reads` 'apart'), which the token's output takes at `gamma_u` beside it (`_rank_terms`), as
mamba_ssm's kernels do (strictly lower scores, `gamma q . k` apart). The recurrence a token at a time on the
complex state (`recurrent_mamba3`) is the definitional form and, in float64, check 1's truth."""

from __future__ import annotations

import dataclasses
import functools
from typing import ClassVar, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.typing import Dtype, PrecisionLike
from jax import lax
from jax.ad_checkpoint import checkpoint_name

from dew.algebra import scan
from dew.batch import document_firsts, real_bounds
from dew.mesh import LayoutRefused
from dew.nn.activations import logistic, silu
from dew.nn.blocks import normal_kernel
from dew.nn.dense import Dense
from dew.nn.derivatives import portable_formed
from dew.nn.flops import projection_flops, ssd_flops
from dew.nn.inputs import PLAIN_CALL, MixerCall, coordinates_of
from dew.nn.kernels.recurrence import Hooks, skip_added
from dew.nn.kernels.recurrence_chunk import Chunk, chunk
from dew.nn.kv_cache import Opened, cache_variables, compute_dtype, kept, stated
from dew.nn.linear import forked_scan, sequence_split
from dew.nn.mixer_base import MixerBase, MixerContext
from dew.nn.mixers.mamba import inverse_softplus_step
from dew.nn.mixers.mamba2 import MambaRMSNormGated
from dew.nn.norms import RMSNorm
from dew.nn.precision import at_least_fp32, float32_dot_general, softplus
from dew.nn.residuals import Residual
from dew.nn.scatter import gather_positions

CHUNK_SIZE = 64
"""mamba_ssm's Mamba-3 `chunk_size` for a single input and output."""

MAMBA3 = Hooks(decay='scalar', delta=False, write_gate=True, scale=1.0, reads='apart')
"""Mamba-3's chunked form on the recurrence template: a decay a head (`dt A`), no delta rule, an explicit
write gate (each token's folded trapezoid weight, `chunk_mamba3`), its readout apart from its token's own
writes, a rank an op token each (`Hooks.ranks`, a call's), which `_rank_terms` weighs at the token's own
gate (mamba_ssm's strictly lower scores and `gamma q . k` apart, `mamba3_siso_fwd.py`, and its MIMO's `i //
R > j // R`, `mamba3_mimo_fwd.py`): read whole, the op weighed them at the folded gate and its rank terms
took the next token's share back off, which cancelled at a document's first token, its output there 22
times the recurrence's worst row from float64 (row 7402)."""


def _biased(value, bias):
    """`value` plus the parameter `bias` (broadcast against it) through the rule
    (`derivatives.portable_formed`): its reverse mode returns `value`'s cotangent only with the bias's
    gradient formed, so XLA puts no layer's bias gradient off to the step's end with the cotangents it sums
    (row 6471's guard: a Mamba-3 step's `B` and `C` biases' gradients, put off so, held every layer's rotated
    keys' and queries' cotangents and the rotation's sines to the end)."""
    return portable_formed(jnp.add)(value, bias)


def _per_head(t, heads: int):
    """`t` `[..., G, X]` repeated over the heads each of its `G` groups serves `[..., H, X]`."""
    return jnp.repeat(t, heads // t.shape[-2], axis=-2)


class Carried(NamedTuple):
    """What a call leaves the next: `state` `[B, H, P, N]`, the recurrence's after its last token's turn
    (its own frame; mamba_ssm's `ssm_state` is the same state turned by its `angle_dt_state`), and that
    token's keys `[B, H, R, N]` as given and its values `[B, H, R, P]`, a rank's each, which the next token
    writes again at its `(1 - lambda) dt` (`k_state`, unturned, and `v_state`)."""
    state: jax.Array
    key: jax.Array
    value: jax.Array


CARRIED = ('ssm_state', 'k_state', 'v_state')
"""`Carried`'s leaves in a decode cache, in its order (`Mamba3.cache_spec`)."""


def _ranked(t):
    """`t` `[B, S, R, ...]` as the chunked op's tokens `[B, S R, ...]`, a token's ranks in turn."""
    return t.reshape(t.shape[0], -1, *t.shape[3:])


def _rank_terms(own, x, gamma):
    """What a token's own writes add to each of its ranks' outputs past the chunked op's (`chunk_mamba3`, its
    readout apart from them, `MAMBA3`): every rank `u` of the token at `gamma`, as the recurrence reads them,
    on `own`, the token's queries and keys unturned `(C, B)` `[B, S, R, G, N]` (one angle a token: its ranks'
    overlaps are its turned ones', mamba_ssm's `qk_dot` on `q_pre . k_pre`), a group's overlaps formed once
    for its heads in fp32 and the operands read in their own dtype. `x` `[B, S, R, H, P]` and `gamma` `[B, S,
    H]` in the working dtype."""
    batch, length, rank, heads, width = x.shape
    groups = own[0].shape[-2]
    overlap = jnp.einsum('bsrgn,bsugn->bsgru', *own, preferred_element_type=x.dtype)
    weighed = (x * gamma[:, :, None, :, None]).reshape(batch, length, rank, groups, heads // groups, width)
    out = jnp.einsum('bsgru,bsugjp->bsrgjp', overlap, weighed)
    return out.reshape(x.shape)


def _trapezoid(delta, real, resets):
    """Each token's re-write weight `[B, S, H]`, its next real token's `(1 - lambda) dt` (`delta`, zero at
    a padded slot) where that token is in its document, zero at a padded slot and past the last; and the
    carried token's `[B, H]`, the first real token's where no document begins up to it. A padded slot is
    as if absent: the trapezoid links each real token to the next real one."""
    batch, length = real.shape
    places = jnp.where(real, jnp.arange(length), length)
    # The first real token at or after each, `length` where none is.
    ahead = jnp.flip(lax.cummin(jnp.flip(places, axis=1), axis=1), axis=1)
    following = jnp.concatenate([ahead[:, 1:], jnp.full((batch, 1), length)], axis=1)
    begun = jnp.cumsum(resets.astype(jnp.int32), axis=1)
    at = jnp.minimum(following, length - 1)
    same = (following < length) & (jnp.take_along_axis(begun, at, axis=1) == begun) & real
    weights = jnp.where(same[..., None], jnp.take_along_axis(delta, at[..., None], axis=1), 0)
    first = jnp.minimum(ahead[:, 0], length - 1)
    opens = (ahead[:, 0] < length) & (jnp.take_along_axis(begun, first[:, None], axis=1)[:, 0] == 0)
    entering = jnp.where(opens[:, None], jnp.take_along_axis(delta, first[:, None, None], axis=1)[:, 0], 0)
    return weights, entering


def _stepped(x, dt, B, C, lam, valid):
    """A call's working dtype (`at_least_fp32` of the operands'), its operands', `dt` and `lam` in the working
    dtype, `dt` zero at a padded slot, and its real tokens `[B, S]` (`valid`, every one where None)."""
    batch, length = x.shape[:2]
    work = at_least_fp32(jnp.result_type(x, B, C))
    dt, lam = (jnp.asarray(t, work) for t in (dt, lam))
    real = jnp.ones((batch, length), bool) if valid is None else jnp.asarray(valid, bool)
    return work, jnp.result_type(x, B, C), jnp.where(real[..., None], dt, 0), lam, real


def chunk_mamba3(x, dt, A, B, C, lam, theta, carried: Carried | None = None, starts=None, valid=None,
                 chunk_size: int | None = None):
    """Mamba-3's recurrence over a sequence on the recurrence template's chunked op (`MAMBA3`), of `R` inputs
    and outputs a token (mamba_ssm's MIMO; one, its SISO).

    `x` `[B, S, R, H, P]`, `dt` `[B, S, H]` (positive, through its softplus), `A` `[B, S, H]` or `[H]`
    (negative), `B` and `C` `[B, S, R, G or H, N]` (`N` even), `lam` `[B, S, H]` in [0, 1], `theta` `[B, S,
    G or H, A]` (`A <= N / 2` pairs turning); `carried` the call before's (`Carried`, in at least fp32) or
    None (a zero state), which the call's first document continues; `starts` `[B, S]` each packed document's
    first token, which enters a zero state and no previous token's write; `valid` `[B, S]` the real tokens,
    a padded slot a zero step that neither turns, decays nor writes the state, the carried token the last
    real one. A token's ranks are the op's tokens in turn, its decay on its first (`chunk_size` counts
    them); each rank's output takes the ranks the op read it before or after the recurrence does
    (`_rank_terms`). Returns `(output [B, S, R, H, P], Carried)` in `at_least_fp32` of the operands'
    dtype."""
    batch, length, _, heads, _ = x.shape
    work, operand, dt, lam, real = _stepped(x, dt, B, C, lam, valid)
    # A document's first token enters no state: its turn and decay act on nothing, so neither enters the
    # chunked op, whose pairs would cancel them to rounding where the recurrence's gradient is exactly zero.
    resets = jnp.zeros((batch, length), bool) if starts is None else jnp.asarray(starts, bool)
    unturned = jnp.where(resets[..., None], 0, dt)
    # Each token's turn of the state's key pairs, the chunked op's own (`Hooks.turns`): its keys and queries
    # as given, its turns' gradient the pairs each turn lies between (row 7419).
    turn = unturned[..., None] * _per_head(jnp.asarray(theta, work), heads)
    following, entering_weight = _trapezoid((1 - lam) * dt, real, resets)
    held = None
    if carried is not None:
        # The carried token's writes again at the first real token's `(1 - lambda) dt`, which the chunked op
        # turns and decays with the carried state at that token.
        again = jnp.einsum('bhrn,bhrp->bhnp', carried.key, carried.value)
        held = jnp.swapaxes(carried.state, -1, -2) + entering_weight[:, :, None, None] * again
    out, final = _turned_chunk(x, dt, A, B, C, lam, turn, unturned, following, held, starts, work=work,
                               operand=operand, chunk_size=chunk_size)[:2]
    # The carry (`Carried`): the state after the last token and that real token's keys and values.
    place = (jnp.full((batch,), length - 1) if valid is None
             else jnp.maximum(jnp.max(jnp.where(valid, jnp.arange(length), -1), axis=1), 0))

    def taken(t):
        return jnp.take_along_axis(t, place.reshape(-1, *(1,) * (t.ndim - 1)), axis=1)[:, 0]

    last = (taken(_per_head(jnp.asarray(B, work), heads)), taken(jnp.asarray(x, work)))
    if valid is not None:
        # A row with no real token keeps the carried keys and values.
        before = ((jnp.zeros_like(last[0]), jnp.zeros_like(last[1])) if carried is None
                  else tuple(jnp.swapaxes(t, 1, 2) for t in (carried.key, carried.value)))
        any_real = jnp.any(valid, axis=1)[:, None, None, None]
        last = tuple(jnp.where(any_real, t, b) for t, b in zip(last, before, strict=True))
    return out, Carried(jnp.swapaxes(final, -1, -2), *(jnp.swapaxes(t, 1, 2) for t in last))


def _turned_chunk(x, dt, A, B, C, lam, turn, unturned, following, state, starts, *, work, operand,
                  chunk_size: int | None, forked=None):
    """The chunked op of a call's tokens (`chunk_mamba3`, `entered_mamba3`): `B` and `C` rounded to the
    operands' dtype, its state turned by `turn` `[B, S, H, A]` each token (`Hooks.turns`), the first rank's
    decay `unturned * A`, the write gate `lam dt + following`, from `state` (`[B, H, N, P]`, or a forked
    call's table's given memories `[B, S', H, N, P]`, each the state after its last token's turn) with
    `starts` `[B, S]` each document's first token, its readout apart from each token's own writes, and the
    token's own at `gamma` (`_rank_terms`). A forked call's `forked` is `(loads, stores, held, entries)`,
    `entries(keys)` its documents' entry writes from the keys (`Chunk.entry_key`). Returns `(output, final
    state, keys)`."""
    batch, length, rank, heads, _ = x.shape
    # The op turns its state each token (`Hooks.turns`), a token's turn on its first rank's.
    hooks = dataclasses.replace(MAMBA3, ranks=rank, turns=turn.shape[-1])
    keys, queries = (_per_head(jnp.asarray(t, work), heads).astype(operand) for t in (B, C))
    turns = _ranked(jnp.zeros((batch, length, rank, *turn.shape[2:]), work).at[:, :, 0].set(turn))
    decay = jnp.zeros((batch, length, rank, heads), work).at[:, :, 0].set(unturned * jnp.asarray(A, work))
    gate = jnp.broadcast_to((lam * dt + following)[:, :, None], decay.shape)
    begins = None if starts is None else jnp.zeros((batch, length, rank), bool).at[:, :, 0].set(starts)
    table = {}
    if forked is not None:
        loads, stores, held, entries = forked
        entry_key, entry_value, entry_weight = entries(keys)
        table = {'loads': loads, 'stores': stores, 'held': held, 'entry_key': entry_key,
                 'entry_value': entry_value, 'entry_weight': entry_weight}
    out, final = chunk(hooks,
                       Chunk(_ranked(queries), _ranked(keys), _ranked(jnp.asarray(x, operand)),
                             log_decay=_ranked(decay), beta=_ranked(gate), state=state,
                             starts=None if begins is None else _ranked(begins), turn=turns, **table),
                       **({} if chunk_size is None else {'chunk_size': chunk_size}), dtype=work)
    out = out.reshape(x.shape) + _rank_terms((C, B), jnp.asarray(x, work), lam * dt)
    return out, final, keys


def entered_mamba3(x, dt, A, B, C, lam, theta, carried: Carried, forked, starts, valid=None,
                   chunk_size: int | None = None):
    """`chunk_mamba3` over a forked call (`dew.nn.linear.forked_scan`), in one pass whatever its depth:
    `carried`'s leaves `[B, 1, ...]` the row's start (a cache's carry, or zero), `forked` the call's
    `(loads, stores, held)` (`recurrence_chunk.Chunk.loads`), each of `starts`' documents (the row's first
    slot beginning one, `dew.batch.document_firsts`) continuing table slot `loads[d]` (one, the start; past
    one, the final carry of the earlier document that stored it; zero, none) and storing its own in
    `stores[d]`. Each document's angles run on from its source's (a scan over the row's documents, a carry
    of angles each), its first real token writes its source's last token again at its `(1 - lambda) dt` (the
    chunked op's entry write, `Chunk.entry_key`), and the chunked op enters its source's state from its
    table. Returns `(output, None)`: a forked call leaves the carry as it was."""
    batch, length, _, heads, _ = x.shape
    loads, stores, held = forked
    work, operand, dt, lam, real = _stepped(x, dt, B, C, lam, valid)
    resets = jnp.asarray(starts, bool)
    documents = loads.shape[1]
    firsts = document_firsts(resets, documents)
    slots = jnp.arange(length, dtype=jnp.int32)
    number = jnp.clip(jnp.sum(slots[None, :, None] >= firsts[:, None, :], axis=-1) - 1, 0, documents - 1)
    rows = jnp.arange(batch)[:, None]
    loads, stores = jnp.asarray(loads, jnp.int32), jnp.asarray(stores, jnp.int32)
    # A document entering nothing begins as `chunk_mamba3`'s do, its first token's turn and decay acting on
    # nothing; one continuing a carry turns and decays it at its first token, as its own row's next token.
    fresh = resets & (jnp.take_along_axis(loads, number, axis=1) == 0)
    unturned = jnp.where(fresh[..., None], 0, dt)
    # Each token's turn of the state's key pairs, the chunked op's own (`Hooks.turns`): a document entering
    # a table's memory turns on from it, its turns' gradient the pairs each turn lies between (row 7419).
    turn = unturned[..., None] * _per_head(jnp.asarray(theta, work), heads)
    delta = (1 - lam) * dt
    following, _ = _trapezoid(delta, real, resets)
    # Each document's first and last real tokens, `length` and -1 where it holds none.
    first, last = real_bounds(number + 1, real, documents)
    holding = first < length
    ended = jnp.maximum(last, 0)
    # Each stored slot's document, by the slot it stores in, to read its last token's keys and values.
    stored_by = jnp.zeros((batch, held + 1), jnp.int32).at[rows, stores].set(
        jnp.broadcast_to(jnp.arange(documents, dtype=jnp.int32), stores.shape), mode="drop")
    weight = jnp.where(holding[..., None], gather_positions(delta, jnp.minimum(first, length - 1)), 0)
    given = jnp.swapaxes(carried.state, -1, -2)
    carried_key = jnp.swapaxes(carried.key, 2, 3)

    def entries(keys):
        # Each document's source's last token: the start's carried keys and values, or the storing
        # document's last real token's.
        source = jnp.take_along_axis(stored_by, jnp.maximum(loads - 1, 0), axis=1)
        at = jnp.take_along_axis(ended, source, axis=1)
        key = jnp.where((loads > 1)[..., None, None, None], gather_positions(keys.astype(work), at),
                        carried_key.astype(work))
        value = jnp.where((loads > 1)[..., None, None, None], gather_positions(jnp.asarray(x, work), at),
                          jnp.swapaxes(carried.value, 2, 3).astype(work))
        return key, value, jnp.where((loads > 0)[..., None], weight, 0)

    out, _, _ = _turned_chunk(x, dt, A, B, C, lam, turn, unturned, following, given, resets, work=work,
                              operand=operand, chunk_size=chunk_size, forked=(loads, stores, held, entries))
    return out, None


def step_mamba3(x, dt, A, B, C, lam, theta, carried: Carried, valid=None, begins=None):
    """Mamba-3's recurrence one token on from `carried` (`Carried`), the decode step: `x` `[B, R, H, P]`,
    `dt`, `A` and `lam` `[B, H]` (`A` or `[H]`), `B` and `C` `[B, R, G or H, N]`, `theta` `[B, G or H, A]`,
    `valid` `[B]` the real tokens (a padded slot leaves `carried` as it is), `begins` `[B]` the tokens that
    begin a document (entering no state, their frame restarting there, as `chunk_mamba3`'s). Returns
    `(output [B, R, H, P], Carried)` in at least fp32, the state in the frame of the token's angle as
    `chunk_mamba3` holds it."""
    heads = x.shape[2]
    work = at_least_fp32(jnp.result_type(x, B, C))
    x, dt, lam = (jnp.asarray(t, work) for t in (x, dt, lam))
    if valid is not None:
        dt = jnp.where(valid[:, None], dt, 0)
    unturned = dt
    if begins is not None:
        fresh = jnp.asarray(begins, bool) if valid is None else jnp.asarray(begins, bool) & valid
        carried = Carried(*(jnp.where(fresh.reshape(-1, *(1,) * (t.ndim - 1)), 0, t) for t in carried))
        unturned = jnp.where(fresh[:, None], 0, dt)
    turn = unturned[..., None] * _per_head(jnp.asarray(theta, work), heads)
    key, query = (_per_head(jnp.asarray(t, work), heads) for t in (B, C))
    alpha = jnp.exp(dt * jnp.asarray(A, work))
    again = jnp.einsum('bhrp,bhrn->bhpn', carried.value, carried.key)
    written = jnp.einsum('brhp,brhn->bhpn', x, key)
    # The carried state with its token's writes again, turned by this token's turn (`Hooks.turns`, `e^{-i
    # turn}` on its key pairs, `scan.turned`), decayed, then this token's own writes.
    entering = carried.state + ((1 - lam) * dt)[..., None, None] * again
    state = (alpha[..., None, None] * scan.turned(entering, turn[:, :, None, :])
             + (lam * dt)[..., None, None] * written)
    out = jnp.einsum('bhpn,brhn->brhp', state, query)
    key, x = (jnp.swapaxes(t, 1, 2) for t in (key, x))
    if valid is not None:
        real = valid[:, None, None, None]
        key, x = (jnp.where(real, t, c) for t, c in ((key, carried.key), (x, carried.value)))
    return out, Carried(state, key, x)


def _complex(t, turned: int):
    """`t` `[..., N]` as its first `turned` dimensions' complex pairs `[..., turned / 2]` and the rest."""
    pairs = t[..., :turned].reshape(*t.shape[:-1], -1, 2)
    return jax.lax.complex(pairs[..., 0], pairs[..., 1]), t[..., turned:]


def recurrent_mamba3(x, dt, A, B, C, lam, theta, state=None, starts=None):
    """Mamba-3's recurrence a token at a time on its complex state, eqs. 5 and 6 with each pair turned a
    token and a token's ranks' writes summed, in `at_least_fp32` of the operands' dtype (float64 where they
    are): operands and results as `chunk_mamba3`'s, the final state in its own frame (`Carried`)."""
    work = at_least_fp32(jnp.result_type(x, B, C))
    batch, length, _, heads, head_dim = x.shape
    turned = 2 * theta.shape[-1]
    x, dt, lam = (jnp.asarray(t, work) for t in (x, dt, lam))
    alpha = jnp.exp(jnp.broadcast_to(dt * jnp.asarray(A, work), dt.shape))
    turns = dt[..., None] * _per_head(jnp.asarray(theta, work), heads)               # [B, S, H, A]
    keys, queries = (_complex(_per_head(jnp.asarray(t, work), heads), turned) for t in (B, C))
    if state is None:
        state = jnp.zeros((batch, heads, head_dim, B.shape[-1]), work)
    begins = jnp.zeros((batch, length), jnp.bool_) if starts is None else jnp.asarray(starts, jnp.bool_)

    def one_token(carried, step):
        h, previous = carried                            # each state part [B, H, P, .]
        x_t, dt_t, lam_t, alpha_t, turn, key_t, query_t, begun = step
        kept = ~begun[:, None, None, None]
        written = tuple(jnp.einsum('brhp,brhn->bhpn', x_t, k) for k in key_t)
        entering = tuple(alpha_t[..., None, None] * s + ((1 - lam_t) * dt_t * alpha_t)[..., None, None] * w
                         for s, w in zip(h, previous, strict=True))
        # A document's first token enters a zero state and no previous token's write.
        spun = (jnp.exp(-1j * turn)[:, :, None, :] * entering[0], entering[1])
        h = tuple(jnp.where(kept, s, 0) + (lam_t * dt_t)[..., None, None] * w
                  for s, w in zip(spun, written, strict=True))
        out = (jnp.einsum('bhpa,brha->brhp', h[0], jnp.conj(query_t[0])).real
               + jnp.einsum('bhpn,brhn->brhp', h[1], query_t[1]))
        return (h, written), out

    steps = jax.tree.map(lambda t: jnp.moveaxis(t, 1, 0), (x, dt, lam, alpha, turns, keys, queries, begins))
    entered = _complex(jnp.asarray(state, work), turned)
    (final, _), out = jax.lax.scan(one_token, (entered, jax.tree.map(jnp.zeros_like, entered)), steps)
    pairs = final[0]
    final = jnp.concatenate([jnp.stack([pairs.real, pairs.imag], axis=-1).reshape(*pairs.shape[:-1], turned),
                             final[1]], axis=-1)
    return jnp.moveaxis(out, 0, 1), final


def _heavy_tail(x):
    """mamba_ssm's `heavy_tail_activation` of the data-dependent `A`: `1 + x` from zero up, `1 / (1 - x)`
    below, positive and smooth at zero."""
    return jnp.maximum(x, 0) + 1 / (1 - jnp.minimum(x, 0))


class Mamba3(nn.Module):
    """The token mixer of a Mamba-3 layer (mamba_ssm's `modules/mamba3.py`), of `mimo_rank` inputs and
    outputs a token (its MIMO; one, its SISO).

    `in_proj` yields `[z, x, B, C, dt, A, trap, angles]` in that order, and no conv runs: `dt = softplus(dt
    + dt_bias)`, the data-dependent `A = -heavy_tail(A)` at most `-A_floor`, `lambda = sigmoid(trap)`, the
    angles' rates `tanh(angles) pi` shared by every head over `rope_fraction` of the state's pairs, and `B`
    and `C`, a rank's each, through an RMS norm over the state with a bias a head and rank after (ones at
    init). Of more ranks than one, each rank's `x` and `z` are the token's times their own weights a head
    (`mimo_x`, `mimo_z`, at init a `1 / R` and ones) and the ranks' gated outputs sum by `mimo_o`'s (`1 /
    R`). The recurrence is `chunk_mamba3`'s, a decode token's `step_mamba3`'s; `D x` adds the skip, then
    `silu(z)` gates the output, or with `outproj_norm` a norm a head first (mamba_ssm's `is_outproj_norm`),
    before `out_proj`. Parameter names are the checkpoint's: in_proj, dt_bias, B_bias, C_bias, B_norm,
    C_norm, D, mimo_x, mimo_z, mimo_o, norm, out_proj.

    The decode cache holds `Carried`'s three in at least fp32 (`ssm_state` `[B, H, P, N]`, `k_state` `[B, H,
    R, N]`, `v_state` `[B, H, R, P]`, the state in its own frame): a one-token call runs the step and a longer
    one the chunked form from them. Padded slots (`attention_metadata.valid`) take a zero step and packed
    documents (`starts`) run as if alone; under a mesh sequence axis above one a decode call and one over
    forks run whole (`dew.nn.linear.sequence_split`), and a training call is refused.
    """

    reads: ClassVar[frozenset[str]] = frozenset({"forks"})
    """Of the call's `MIXING` fields (`dew.nn.inputs.MixerCall`), the documents that continue another
    (`dew.batch.Forks`), each from the carry its parent leaves (`entered_mamba3`, `forked_scan`); no mask or
    bias, which a recurrence reading every earlier token of its document has no way to narrow."""

    emb_features: int
    num_heads: int
    head_dim: int
    state_size: int = 128
    n_groups: int = 1
    mimo_rank: int = 1
    rope_fraction: float = 0.5
    chunk_size: int = CHUNK_SIZE
    """The chunked op's tokens a chunk, a token's ranks each one (mamba_ssm's `chunk_size` times its rank)."""
    A_floor: float = 1e-4
    outproj_norm: bool = False
    norm_eps: float = 1e-5
    init_std: float | None = None
    output_init_std: float | None = None
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @property
    def intermediate_size(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def angles(self) -> int:
        """The state's pairs that turn, mamba_ssm's `num_rope_angles`: `rope_fraction` of its dimensions,
        rounded down to even, halved."""
        return int(self.state_size * self.rope_fraction) // 2

    @property
    def widths(self) -> tuple[int, ...]:
        """`in_proj`'s sections: z, x, B, C, dt, A, trap and the angles."""
        inner, heads = self.intermediate_size, self.num_heads
        groups = self.mimo_rank * self.n_groups * self.state_size
        return inner, inner, groups, groups, heads, heads, heads, self.angles

    @nn.nowrap
    def model_flops(self, lengths) -> float:
        """The in and out projections and the state's write and read a rank (`ssd_flops`): the trapezoid's
        previous write folds into each token's own (`chunk_mamba3`), its own writes read apart."""
        return (projection_flops(lengths, self.emb_features, sum(self.widths) + self.intermediate_size)
                + self.mimo_rank * ssd_flops(lengths, self.num_heads, self.state_size, self.head_dim))

    @nn.nowrap
    def parameter_flops(self, lengths) -> float:
        """The learned projections alone, without the state's contractions (`ssd_flops`)."""
        scan = ssd_flops(lengths, self.num_heads, self.state_size, self.head_dim)
        return self.model_flops(lengths) - self.mimo_rank * scan

    def setup(self):
        if self.num_heads % self.n_groups:
            raise ValueError(f"num_heads ({self.num_heads}) must be a multiple of n_groups ({self.n_groups})")
        if self.angles < 1:
            raise ValueError(f"rope_fraction {self.rope_fraction} of a state of {self.state_size} turns no "
                             "pair")
        dense = functools.partial(Dense, use_bias=False, dtype=self.dtype, precision=self.precision,
                                  dot_general=float32_dot_general)
        self.in_proj = dense(sum(self.widths), name='in_proj', kernel_axes=('embed', 'linear'),
                             **normal_kernel(self.init_std))
        self.dt_bias = self.param('dt_bias', inverse_softplus_step, (self.num_heads,))
        ranked = (self.num_heads, self.mimo_rank)
        self.B_bias, self.C_bias = (self.param(name, nn.initializers.ones, (*ranked, self.state_size),
                                               jnp.float32) for name in ('B_bias', 'C_bias'))
        if self.mimo_rank > 1:
            spread = functools.partial(nn.initializers.constant, 1 / self.mimo_rank)
            self.mimo_x, self.mimo_z, self.mimo_o = (
                self.param(name, init, (*ranked, self.head_dim), jnp.float32)
                for name, init in (('mimo_x', spread()), ('mimo_z', nn.initializers.ones),
                                   ('mimo_o', spread())))
        self.B_norm, self.C_norm = (RMSNorm(epsilon=1e-5, dtype=self.dtype, name=name)
                                    for name in ('B_norm', 'C_norm'))
        self.D = self.param('D', nn.initializers.ones, (self.num_heads,), jnp.float32)
        if self.outproj_norm:
            self.norm = MambaRMSNormGated(epsilon=self.norm_eps, groups=self.num_heads, gate_after=True,
                                          dtype=self.dtype, name='norm')
        self.out_proj = dense(self.emb_features, name='out_proj', kernel_axes=('attention', 'embed'),
                              **normal_kernel(self.init_std if self.output_init_std is None
                                              else self.output_init_std))

    def _operands(self, projected):
        """The recurrence's operands from the projection: `x` `[B, S, R, H, P]` and `B` and `C` `[B, S, R, H,
        N]` in its dtype, the step, `A`, `lambda` and the angles' rates `[B, S, 1, A]` in at least fp32, and
        `z` `[B, S, R, H, P]`, a rank's each."""
        batch, length, _ = projected.shape
        wide = at_least_fp32(projected.dtype)
        z, x, B, C, dt, A, trap, angles = jnp.split(projected, np.cumsum(self.widths)[:-1], axis=-1)
        keys = []
        for t, norm, bias in ((B, self.B_norm, self.B_bias), (C, self.C_norm, self.C_bias)):
            ranked = t.reshape(batch, length, self.mimo_rank, self.n_groups, self.state_size)
            normed = _per_head(norm(ranked).astype(wide), self.num_heads)
            keys.append(_biased(normed, jnp.swapaxes(bias, 0, 1)).astype(projected.dtype))
        step = softplus(_biased(dt.astype(wide), self.dt_bias))
        decay = jnp.minimum(-_heavy_tail(A.astype(wide)), -self.A_floor)
        rates = (jnp.tanh(angles.astype(wide)) * jnp.pi)[:, :, None]
        x, z = (t.reshape(batch, length, 1, self.num_heads, self.head_dim) for t in (x, z))
        if self.mimo_rank > 1:
            x, z = ((t.astype(wide) * jnp.swapaxes(weight, 0, 1)).astype(projected.dtype)
                    for t, weight in ((x, self.mimo_x), (z, self.mimo_z)))
        return x, step, decay, *keys, logistic(trap.astype(wide)), rates, z

    @nn.nowrap
    def cache_spec(self, rows: int) -> dict[str, Opened]:
        """The layer's decode cache for `rows` rows: `Carried`'s four leaves in order (`_shapes`), in at
        least fp32, where the scan runs."""
        wide = at_least_fp32(compute_dtype(self, self.dtype))
        return {name: Opened(shape, wide) for name, shape in zip(CARRIED, self._shapes(rows), strict=True)}

    def _shapes(self, batch: int) -> tuple[tuple[int, ...], ...]:
        """`Carried`'s three shapes for `batch` rows: the state, the last token's keys and values."""
        heads = (batch, self.num_heads)
        ranked = (*heads, self.mimo_rank)
        return (*heads, self.head_dim, self.state_size), (*ranked, self.state_size), (*ranked, self.head_dim)

    @nn.compact
    def __call__(self, x, call: MixerCall = PLAIN_CALL):
        batch, length, _ = x.shape
        tokens = coordinates_of(call.metadata, batch, length)
        valid = tokens.valid
        if (shards := sequence_split(call.decode, tokens)) > 1:
            # A decode call and one over forks run whole (`sequence_split`); a training call's split is
            # refused.
            raise LayoutRefused(f"mamba3 runs a training sequence on one shard; this mesh splits it over "
                                f"{shards}. Run it on a mesh with sequence=1.")
        if valid is not None:
            x = jnp.where(valid[:, :, None], x, 0)
        projected = self.in_proj(x)
        wide = at_least_fp32(projected.dtype)
        values, dt, A, B, C, lam, theta, z = self._operands(projected)
        carried, cache = None, ()
        if call.decode:
            spec = self.cache_spec(batch)
            for name, shape in zip(CARRIED, self._shapes(batch), strict=True):
                stated(spec, name, shape, wide)
            held = cache_variables(self, spec)
            cache = tuple(held[name] for name in CARRIED)
            carried = Carried(*(variable.value for variable in cache))
        if tokens.forks is not None:
            # Each document from the carry its parent leaves, a root from the row's (`forked_scan`); a cached
            # call's carry left as it was.
            def forked_rule(values, *, valid, starts, state, loads, stores, held):
                return entered_mamba3(values, dt, A, B, C, lam, theta, state, (loads, stores, held), starts,
                                      valid=valid, chunk_size=self.chunk_size)

            start = Carried(*(jnp.zeros(shape, wide) for shape in self._shapes(batch)) if carried is None
                            else (leaf.astype(wide) for leaf in carried))
            out, carried, cache = forked_scan(values, forked_rule, tokens, start), None, ()
        elif carried is not None and length == 1:
            out, carried = step_mamba3(values[:, 0], dt[:, 0], A[:, 0], B[:, 0], C[:, 0], lam[:, 0],
                                       theta[:, 0], carried, valid=None if valid is None else valid[:, 0],
                                       begins=None if tokens.starts is None else tokens.starts[:, 0])
            out = out[:, None]
        else:
            out, carried = chunk_mamba3(values, dt, A, B, C, lam, theta, carried, starts=tokens.starts,
                                        valid=valid, chunk_size=self.chunk_size)
        if call.decode and carried is not None:
            # The scan's carry kept in the cache's dtype, as it always was.
            for name, variable, value in zip(CARRIED, cache, carried, strict=True):
                kept(variable, value.astype(variable.value.dtype), name)
        # The scan's output under both names a remat policy reads it by (`RESIDUALS`), as `held_scan` marks
        # it.
        out = checkpoint_name(checkpoint_name(out, Residual.SCAN), Residual.CONTEXT)
        # The skip by the shared product (`recurrence.skip_added`): written here, D's gradient was one
        # reduction over every token's products, whose order XLA:CPU took by the input's shape, a packed
        # tiny decoder's 1.65 times the error of its documents laid a row each (row 6479).
        out = skip_added(out, self.D, values, wide)
        ranks = (batch, length, self.mimo_rank, self.intermediate_size)
        if self.outproj_norm:
            out = self.norm(out.reshape(ranks), z.reshape(ranks)).astype(wide).reshape(out.shape)
        else:
            out = out * silu(z.astype(wide))
        if self.mimo_rank > 1:
            out = jnp.einsum('bsrhp,hrp->bshp', out, self.mimo_o)
        out = out.reshape(batch, length, self.intermediate_size).astype(projected.dtype)
        if valid is not None:
            out = jnp.where(valid[:, :, None], out, 0)
        return self.out_proj(out)


@dataclasses.dataclass(frozen=True)
class Mamba3Mixer(MixerBase):
    """The `mamba3` kind, by mamba_ssm's Mamba-3 `ssm_cfg` fields: `expand` times the model's width over
    `head_dim` heads (its `d_inner // headdim`), each over a state of `state_size` (`d_state`), `n_groups`
    distinct B and C (`ngroups`), `mimo_rank` (its `mimo_rank` where `is_mimo`, else one), `rope_fraction`,
    `A_floor`, `chunk_size` (the chunked op's tokens, mamba_ssm's times its rank) and `outproj_norm`
    (`is_outproj_norm`), at its defaults (state-spaces/mamba3-siso-187m's)."""

    residuals = (Residual.CONTEXT, Residual.SCAN)
    # Each document from the carry its parent leaves (`entered_mamba3`).
    reads_forks = True
    # Its cached prefill resets its state at each packed document's start (`chunk_mamba3`'s `starts`).
    packed_cache = True
    draws_init_std = True

    expand: int = 2
    head_dim: int = 64
    state_size: int = 128
    n_groups: int = 1
    mimo_rank: int = 1
    rope_fraction: float = 0.5
    A_floor: float = 1e-4
    chunk_size: int = CHUNK_SIZE
    outproj_norm: bool = False

    def build(self, ctx: MixerContext):
        if not ctx.causal:
            raise ValueError("mamba3 requires causal=True; its recurrence has no bidirectional mode")
        inner = self.expand * ctx.emb_features
        if inner % self.head_dim:
            raise ValueError(f"mamba3's {self.expand} x {ctx.emb_features} channels do not split into heads "
                             f"of {self.head_dim}")
        return functools.partial(
            Mamba3, emb_features=ctx.emb_features, num_heads=inner // self.head_dim, head_dim=self.head_dim,
            state_size=self.state_size, n_groups=self.n_groups, mimo_rank=self.mimo_rank,
            rope_fraction=float(self.rope_fraction),
            chunk_size=self.chunk_size, A_floor=float(self.A_floor), outproj_norm=self.outproj_norm,
            norm_eps=ctx.norm_eps, init_std=ctx.init_std, output_init_std=ctx.output_init_std,
            dtype=ctx.dtype, precision=ctx.precision)
