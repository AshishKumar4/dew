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
at `gamma_u` alone, so the chunked op writes at `gamma_u + delta_{u+1}` and each output takes back its own
token's `delta_{u+1}` term (`chunk_mamba3`). The recurrence a token at a time on the complex state
(`recurrent_mamba3`) is the definitional form and, in float64, check 1's truth."""

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

from dew.mesh import LayoutRefused
from dew.nn import rope
from dew.nn.blocks import normal_kernel
from dew.nn.dense import Dense
from dew.nn.flops import projection_flops, ssd_flops
from dew.nn.inputs import PLAIN_CALL, MixerCall, coordinates_of
from dew.nn.kernels.recurrence import Hooks
from dew.nn.kernels.recurrence_chunk import Chunk, chunk
from dew.nn.mixer_base import MixerBase, MixerContext
from dew.nn.mixers.mamba import inverse_softplus_step
from dew.nn.mixers.mamba2 import MambaRMSNormGated
from dew.nn.norms import RMSNorm
from dew.nn.precision import at_least_fp32, float32_dot_general
from dew.nn.sharding import sequence_shards

CHUNK_SIZE = 64
"""mamba_ssm's Mamba-3 `chunk_size` for a single input and output."""

MAMBA3 = Hooks(decay='scalar', delta=False, write_gate=True, scale=1.0)
"""Mamba-3's chunked form on the recurrence template: a decay a head (`dt A`), no delta rule, an explicit
write gate (each token's folded trapezoid weight, `chunk_mamba3`)."""


def _within(starts, length: int, batch: int):
    """Whether each token `[B, S]` continues its predecessor's document (False at a document's first token
    and at the sequence's first)."""
    if starts is None:
        starts = jnp.zeros((batch, length), jnp.bool_)
    return ~jnp.asarray(starts, jnp.bool_).at[:, 0].set(True)


def _angles(dt, theta, continues):
    """Each token's angle `[B, S, H, R]` over its document: `dt theta` summed from the document's first token
    through the token (`dt` zero where a token's turn acts on no state, `chunk_mamba3`), in `dt`'s dtype,
    held in [0, 2 pi) as mamba_ssm's `angle_dt` holds it, so its rounding stays a turn's however long the
    document."""
    step = dt[..., None] * jnp.asarray(theta, dt.dtype)

    def combine(left, right):
        # A segmented sum: a token that begins a document drops what came before it.
        (sum_left, keep_left), (sum_right, keep_right) = left, right
        summed = jnp.where(keep_right, sum_left + sum_right, sum_right)
        return jnp.remainder(summed, 2 * jnp.pi), keep_left & keep_right

    keep = jnp.broadcast_to(continues[:, :, None, None], step.shape)
    summed, _ = jax.lax.associative_scan(combine, (step, keep), axis=1)
    return summed


def rotated(x, angles):
    """`x` `[B, S, R, H, N]` with each adjacent pair of its first `2 A` dimensions turned by its angle
    (`angles` broadcast against `x` but its last axis, `A` wide), `(v_2j + i v_2j+1) e^{i phi}`, the rest
    as given: `rope.rotate`'s adjacent pairs."""
    return rope.rotate(x, jnp.cos(angles), jnp.sin(angles), pairs='adjacent')


def _per_head(t, heads: int):
    """`t` `[..., G, X]` repeated over the heads each of its `G` groups serves `[..., H, X]`."""
    return jnp.repeat(t, heads // t.shape[-2], axis=-2)


class Carried(NamedTuple):
    """What a call leaves the next (mamba_ssm's `angle_dt_state`, `ssm_state`, `k_state` and `v_state`):
    `state` `[B, H, P, N]` in the frame of each pair's `angle` `[B, H, A]`, its document's through the last
    token in [0, 2 pi), and that token's keys `[B, H, R, N]` turned by it and its values `[B, H, R, P]`, a
    rank's each, which the next token writes again at its `(1 - lambda) dt`."""
    state: jax.Array
    angle: jax.Array
    key: jax.Array
    value: jax.Array


def _ranked(t):
    """`t` `[B, S, R, ...]` as the chunked op's tokens `[B, S R, ...]`, a token's ranks in turn."""
    return t.reshape(t.shape[0], -1, *t.shape[3:])


def _rank_terms(queries, keys, x, gamma, following):
    """What a token's own writes add to each of its ranks' outputs past the chunked op's (`chunk_mamba3`): its
    rank `r` read its ranks through `r` at `gamma + delta` (the next token's `delta`) and none after, where
    the recurrence reads all of them at `gamma` alone. `queries` and `keys` `[B, S, R, H, N]` as the op read
    them, `x` `[B, S, R, H, P]`, `gamma` and `following` `[B, S, H]`."""
    rank = x.shape[2]
    overlap = jnp.einsum('bsrhn,bsuhn->bshru', queries, keys)
    later = jnp.arange(rank)[None, :] > jnp.arange(rank)[:, None]
    weight = jnp.where(later, gamma[..., None, None], -following[..., None, None])
    return jnp.einsum('bshru,bsuhp->bsrhp', overlap * weight, x)


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
    batch, length, rank, heads, _ = x.shape
    work = at_least_fp32(jnp.result_type(x, B, C))
    operand = jnp.result_type(x, B, C)
    continues = _within(starts, length, batch)
    dt, lam = (jnp.asarray(t, work) for t in (dt, lam))
    real = jnp.ones((batch, length), bool) if valid is None else jnp.asarray(valid, bool)
    dt = jnp.where(real[..., None], dt, 0)
    # A document's first token enters no state: its turn and decay act on nothing, so neither enters the
    # chunked op, whose pairs would cancel them to rounding where the recurrence's gradient is exactly zero.
    resets = jnp.zeros((batch, length), bool) if starts is None else jnp.asarray(starts, bool)
    unturned = jnp.where(resets[..., None], 0, dt)
    angles = _angles(unturned, _per_head(jnp.asarray(theta), heads), continues)
    following, entering_weight = _trapezoid((1 - lam) * dt, real, resets)
    held = None
    if carried is not None:
        # The first document's angles from the carried angle on, and the carried token's writes again at
        # its first real token's `(1 - lambda) dt`, which the chunked op decays by the alphas to it.
        first = jnp.cumsum(resets.astype(jnp.int32), axis=1) == 0
        entering = jnp.where(first[:, :, None, None], carried.angle[:, None], 0)
        angles = jnp.remainder(angles + entering, 2 * jnp.pi)
        again = jnp.einsum('bhrn,bhrp->bhnp', carried.key, carried.value)
        held = jnp.swapaxes(carried.state, -1, -2) + entering_weight[:, :, None, None] * again
    keys, queries = (rotated(_per_head(jnp.asarray(t, work), heads), angles[:, :, None]).astype(operand)
                     for t in (B, C))
    decay = jnp.zeros((batch, length, rank, heads), work).at[:, :, 0].set(unturned * jnp.asarray(A, work))
    gate = jnp.broadcast_to((lam * dt + following)[:, :, None], decay.shape)
    begins = None if starts is None else jnp.zeros((batch, length, rank), bool).at[:, :, 0].set(starts)
    out, final = chunk(MAMBA3, Chunk(_ranked(queries), _ranked(keys), _ranked(jnp.asarray(x, operand)),
                                     log_decay=_ranked(decay), beta=_ranked(gate), state=held,
                                     starts=None if begins is None else _ranked(begins)),
                       **({} if chunk_size is None else {'chunk_size': chunk_size}), dtype=work)
    # On the rounded keys and queries the chunked op read, so the terms cancel its own.
    out = out.reshape(x.shape) + _rank_terms(queries.astype(work), keys.astype(work), jnp.asarray(x, work),
                                             lam * dt, following)
    last = (keys[:, -1].astype(work), jnp.asarray(x[:, -1], work))
    if valid is not None:
        # The last real token's keys and values; a row with none keeps the carried ones.
        place = jnp.maximum(jnp.max(jnp.where(valid, jnp.arange(length), -1), axis=1), 0)
        at = place[:, None, None, None, None]
        taken = tuple(jnp.take_along_axis(jnp.asarray(t, work), at, axis=1)[:, 0] for t in (keys, x))
        before = ((jnp.zeros_like(taken[0]), jnp.zeros_like(taken[1])) if carried is None
                  else tuple(jnp.swapaxes(t, 1, 2) for t in (carried.key, carried.value)))
        any_real = jnp.any(valid, axis=1)[:, None, None, None]
        last = tuple(jnp.where(any_real, t, b) for t, b in zip(taken, before, strict=True))
    return out, Carried(jnp.swapaxes(final, -1, -2), angles[:, -1], *(jnp.swapaxes(t, 1, 2) for t in last))


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
    angle = jnp.remainder(carried.angle + turn, 2 * jnp.pi)
    # A token's keys and queries as one step of a sequence (`rotated`'s five axes), the angle a head's.
    key, query = (rotated(_per_head(jnp.asarray(t, work), heads)[:, None], angle[:, None, None])[:, 0]
                  for t in (B, C))
    alpha = jnp.exp(dt * jnp.asarray(A, work))
    again = jnp.einsum('bhrp,bhrn->bhpn', carried.value, carried.key)
    written = jnp.einsum('brhp,brhn->bhpn', x, key)
    state = (alpha[..., None, None] * carried.state + ((1 - lam) * dt * alpha)[..., None, None] * again
             + (lam * dt)[..., None, None] * written)
    out = jnp.einsum('bhpn,brhn->brhp', state, query)
    key, x = (jnp.swapaxes(t, 1, 2) for t in (key, x))
    if valid is not None:
        real = valid[:, None, None, None]
        key, x = (jnp.where(real, t, c) for t, c in ((key, carried.key), (x, carried.value)))
    return out, Carried(state, angle, key, x)


def _complex(t, turned: int):
    """`t` `[..., N]` as its first `turned` dimensions' complex pairs `[..., turned / 2]` and the rest."""
    pairs = t[..., :turned].reshape(*t.shape[:-1], -1, 2)
    return jax.lax.complex(pairs[..., 0], pairs[..., 1]), t[..., turned:]


def recurrent_mamba3(x, dt, A, B, C, lam, theta, state=None, starts=None):
    """Mamba-3's recurrence a token at a time on its complex state, eqs. 5 and 6 with each pair turned a
    token and a token's ranks' writes summed, in `at_least_fp32` of the operands' dtype (float64 where they
    are): operands and results as `chunk_mamba3`'s, the final state turned by its document's angle into
    `chunk_mamba3`'s frame."""
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
        h, previous, angle = carried                     # each state part [B, H, P, .]; angle [B, H, A]
        x_t, dt_t, lam_t, alpha_t, turn, key_t, query_t, begun = step
        kept = ~begun[:, None, None, None]
        written = tuple(jnp.einsum('brhp,brhn->bhpn', x_t, k) for k in key_t)
        entering = tuple(alpha_t[..., None, None] * s + ((1 - lam_t) * dt_t * alpha_t)[..., None, None] * w
                         for s, w in zip(h, previous, strict=True))
        # A document's first token enters a zero state and no previous token's write.
        spun = (jnp.exp(-1j * turn)[:, :, None, :] * entering[0], entering[1])
        h = tuple(jnp.where(kept, s, 0) + (lam_t * dt_t)[..., None, None] * w
                  for s, w in zip(spun, written, strict=True))
        # The frame restarts at a document's first token, its own turn acting on no state.
        angle = jnp.where(begun[:, None, None], 0, angle + turn)
        out = (jnp.einsum('bhpa,brha->brhp', h[0], jnp.conj(query_t[0])).real
               + jnp.einsum('bhpn,brhn->brhp', h[1], query_t[1]))
        return (h, written, angle), out

    steps = jax.tree.map(lambda t: jnp.moveaxis(t, 1, 0), (x, dt, lam, alpha, turns, keys, queries, begins))
    entered = _complex(jnp.asarray(state, work), turned)
    carried = (entered, jax.tree.map(jnp.zeros_like, entered), jnp.zeros((batch, heads, turned // 2), work))
    (final, _, angle), out = jax.lax.scan(one_token, carried, steps)
    pairs = final[0] * jnp.exp(1j * angle)[:, :, None, :]
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

    The decode cache holds `Carried`'s four in at least fp32 (`ssm_state` `[B, H, P, N]`, `angle_state` `[B,
    H, A]`, `k_state` `[B, H, R, N]`, `v_state` `[B, H, R, P]`): a one-token call runs the step and a longer
    one the chunked form from them. Padded slots (`attention_metadata.valid`) take a zero step and packed
    documents (`starts`) run as if alone; a call under a mesh sequence axis above one is refused.
    """

    reads: ClassVar[frozenset[str]] = frozenset()
    """No `MIXING` field (`dew.nn.inputs.MixerCall`): a recurrence reads every earlier token of its document,
    which no mask, bias, tree or entered document narrows here."""

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
        previous write folds into each token's own (`chunk_mamba3`)."""
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
            keys.append((normed + jnp.swapaxes(bias, 0, 1)).astype(projected.dtype))
        step = nn.softplus(dt.astype(wide) + self.dt_bias)
        decay = jnp.minimum(-_heavy_tail(A.astype(wide)), -self.A_floor)
        rates = (jnp.tanh(angles.astype(wide)) * jnp.pi)[:, :, None]
        x, z = (t.reshape(batch, length, 1, self.num_heads, self.head_dim) for t in (x, z))
        if self.mimo_rank > 1:
            x, z = ((t.astype(wide) * jnp.swapaxes(weight, 0, 1)).astype(projected.dtype)
                    for t, weight in ((x, self.mimo_x), (z, self.mimo_z)))
        return x, step, decay, *keys, nn.sigmoid(trap.astype(wide)), rates, z

    def _cache(self, batch: int, dtype) -> tuple[bool, Carried]:
        """Whether the cache was allocated before this call, and its four variables (`Carried`'s)."""
        allocated = self.has_variable('cache', 'ssm_state')
        heads = (batch, self.num_heads)
        ranked = (*heads, self.mimo_rank)
        shapes = ((*heads, self.head_dim, self.state_size), (*heads, self.angles), (*ranked, self.state_size),
                  (*ranked, self.head_dim))
        names = ('ssm_state', 'angle_state', 'k_state', 'v_state')
        return allocated, Carried(*(self.variable('cache', name, jnp.zeros, shape, dtype)
                                    for name, shape in zip(names, shapes, strict=True)))

    @nn.compact
    def __call__(self, x, call: MixerCall = PLAIN_CALL):
        batch, length, _ = x.shape
        tokens = coordinates_of(call.metadata, batch, length)
        valid = tokens.valid
        if (shards := sequence_shards()) > 1:
            raise LayoutRefused(f"mamba3 runs a sequence on one shard; this mesh splits it over {shards}. "
                                "Run it on a mesh with sequence=1.")
        if valid is not None:
            x = jnp.where(valid[:, :, None], x, 0)
        projected = self.in_proj(x)
        wide = at_least_fp32(projected.dtype)
        *operands, z = self._operands(projected)
        carried = None
        if call.decode:
            allocated, cache = self._cache(batch, wide)
            if not allocated:
                # Allocation only: init_cache's dummy token leaves no state behind.
                return self.out_proj(jnp.zeros((batch, length, self.intermediate_size), self.dtype))
            carried = Carried(*(variable.value for variable in cache))
        if call.decode and length == 1:
            out, carried = step_mamba3(*(t[:, 0] for t in operands), carried,
                                       valid=None if valid is None else valid[:, 0],
                                       begins=None if tokens.starts is None else tokens.starts[:, 0])
            out = out[:, None]
        else:
            out, carried = chunk_mamba3(*operands, carried, starts=tokens.starts, valid=valid,
                                        chunk_size=self.chunk_size)
        if call.decode:
            for variable, value in zip(cache, carried, strict=True):
                variable.value = value.astype(variable.value.dtype)
        # The scan's output under both names a remat policy reads it by (`RESIDUALS`), as `held_scan` marks
        # it.
        out = checkpoint_name(checkpoint_name(out, 'scan'), 'context')
        out = out + self.D[:, None] * operands[0].astype(wide)
        ranks = (batch, length, self.mimo_rank, self.intermediate_size)
        if self.outproj_norm:
            out = self.norm(out.reshape(ranks), z.reshape(ranks)).astype(wide).reshape(out.shape)
        else:
            out = out * nn.silu(z.astype(wide))
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

    residuals = ('context', 'scan')
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
