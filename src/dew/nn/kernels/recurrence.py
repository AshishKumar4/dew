"""The linear recurrence every recurrent mixer runs, as one Op whose hooks select its form.

Per head, over a state `S` `[Dk, Dv]`, a token updates and reads

    S' = (I - delta beta k k^T) Gamma S + iota k v^T,    o = S'^T q (+ D v)

where `Gamma` is the decay (none, one per head or one per key dimension), `delta` turns the delta
rule on, and `iota` is the write gate (beta under the delta rule, an input gate or one otherwise).
The gated delta rule (Qwen3.5, Qwen3-Next), Kimi's delta attention and Mamba2's SSD are rows of
it, and a short conv over the token's projections, the queries' and keys' l2 norm, the gates'
activations and a gated RMS norm of the output are its prologue and epilogue.

This module holds the decode token, the chunk of one (`step`): one program a row and value head
reads the state once and writes it once, with its hooks inlined, which is the decode step and the
fused decode step it replaces. Its arithmetic is the XLA reference's (`step_reference`), which is
also its CPU path and, on float64 operands, the float64 truth. Of the chunked form it holds the first
stage every row of it shares, the decay's prefix within each chunk (`cumulated`).
"""

import dataclasses
import functools
import math
from typing import Literal

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from dew._hashing import fmix32
from dew.algebra.scan import Reads, turned
from dew.nn.activations import GATE_ACTIVATIONS, logistic, silu
from dew.nn.aligned import aligned_sum
from dew.nn.derivatives import KernelGlue, Protected, contracted, with_gradients_formed
from dew.nn.kernels import launches
from dew.nn.precision import (
    accurate_exp,
    at_least_fp32,
    compensated_add,
    l2norm,
    rms_statistics,
    rms_weighted,
    softplus,
    stochastic_bf16,
)

from ..sharding import explicit_spec, grouped_heads, manual_map
from . import chunked_launch, short_conv
from .generation import linear_tangent


@dataclasses.dataclass(frozen=True)
class Conv:
    """The prologue's short conv: a token's channels `[C]`, its queries, keys and values head by head in
    the `order` its layer projects them ('qkv', a delta rule's; 'vkq', Mamba2's x, B and C), over `held`
    `[K - 1, C]` history with `taps` `[C, K]`, its `bias` `[C]` added where it has one (`Step.conv_bias`),
    then `activation` ('silu' or None). The history is read rounded to the token's dtype, as the layer's
    conv reads it."""
    activation: str | None = 'silu'
    order: str = 'qkv'
    bias: bool = False

    def __post_init__(self):
        if sorted(self.order) != ['k', 'q', 'v']:
            raise ValueError(f"a conv's order is a permutation of 'qkv', not {self.order!r}")



Erase = Literal['tied', 'read', 'untied', 'additive', 'values']
"""What the delta rule erases a token (`Hooks.erase`)."""

@dataclasses.dataclass(frozen=True)
class Gates:
    """The gates from their pre-activations, a head's: the step `softplus(gate + bias)`, clipped to
    `limit` where one is given, the log decay `-exp(log_a)` times it (the gated delta rule's and
    Mamba2's), and the write gate `write`: 'sigmoid' of a raw `beta` (the gated delta rule's), or
    'step', the step itself (Mamba2's discretized input). Without it the step takes the log decay and
    beta as they are."""
    write: str = 'sigmoid'
    limit: tuple[float, float] | None = None

    def __post_init__(self):
        if self.write not in ('sigmoid', 'step'):
            raise ValueError(f"a write gate is 'sigmoid' or 'step', not {self.write!r}")


@dataclasses.dataclass(frozen=True)
class OutputNorm:
    """The epilogue's gated RMS norm of a head's output: `rms_norm(o) weight act(z)`, the norm in fp32
    with `epsilon`, `activation` the gate's ('silu' or 'sigmoid'), the output first read back in `read_in`
    where it names a dtype: the one the layer's input projection computes in, the projections' dtype,
    where its reference casts the scan's output before its norm (transformers' Qwen3-Next and Kimi
    `.to(query.dtype)`), as the layer's unfused token does. Normalized unrounded, a fused bf16 token sat
    nearer float64 than its reference and apart from the layer's own unfused token (row 6963)."""
    epsilon: float
    activation: str = 'silu'
    read_in: str | None = None


@dataclasses.dataclass(frozen=True)
class Hooks:
    """A step's static form, which enters its launch's key: the prologue's conv, the queries' and
    keys' l2 norm, the queries' `scale` (in working precision; under the l2 norm `Dk^-1/2` where None,
    else none where None), the decay's kind ('none', 'scalar' a head, 'vector' a key dimension), the
    gates, the delta rule, whether a write gate `beta` scales what the token writes (always under the
    delta rule or the gates; without them, an input gate, else one), whether a scalar decay is
    `rated`: the write gate times the head's `rate` (Mamba's discretization, `A dt` with `dt` the
    write gate), a value kind defined on a contraction, `rate <= 0` and the write gate `>= 0`, which
    its callers hold and a port may take as given (mamba_ssm's kernels clamp each decay to at most
    one); a skip `D v`, and the epilogue; and `write_bound`, the largest write gate
    a layer declares of l2-normalized keys (its config's: 1 for a sigmoid's, 2 for twice a sigmoid's, the
    negative-eigenvalue delta rule's), which bounds the delta rule's chunk inverse (`inverse_bound`).
    `unit_queries` declares the queries a layer hands the op l2-normalized, each of at most unit norm (an
    epsilon's normalization gives less), as `write_bound` declares its keys: with both, and write gates at
    most one, every row of the template's own pairs is at most one in size, so it may decay each key once for
    every product (`templates.recurrence.mainloop.Call.unit_pairs`).
    `decay_bound`, a finite negative number where given, is the least log decay a token a layer proves
    (Kimi delta attention's `lower_bound` times a sigmoid), which a decay a key dimension's template prefix
    reads (`templates.recurrence.mainloop.Call.prefix_limbs`); None where it proves none. The
    readout reads the state after the token's update (`reads` 'after'), the state entering it ('before',
    RWKV6's, `o = q^T S`), or the state after the update less the token's own write ('apart', `o = (g
    q)^T S`, mamba_ssm's Mamba-3, whose own write its caller weighs apart: its intra-chunk scores strictly
    lower, the diagonal `gamma q . k` added apart "to prevent non-causal numerical leakage",
    `mamba3_siso_fwd.py`); `own_weight` adds the token's own write to it weighted a key dimension, `o += ((q
    * own * beta k) . 1) v` (`own` `[Hv, Dk]`, RWKV6's bonus `u` read before the update), each alone a form
    of any recurrence. The chunked op (`recurrence_chunk.chunk`) takes the same form. `turns`, where past
    zero, turns the state's first `2 turns` key rows each token, `turns` adjacent pairs each multiplied as a
    complex number by `e^{-i turn}` (`Gated.turn`) before the decay, a head's (Mamba-3's rotating basis; the
    chunked op's every kernel reads it through frames, `templates.recurrence.turned`, each token's in its
    turn's), unrated, without the delta rule or an own write's weight, its readout after the token's turn
    ('after' or 'apart'; base review fbb1c2f71 F1). `erase` is the delta rule's transition beside its decay
    (kernel-templates.md §3's erase kinds): 'tied', its keys' own, `S' = (I - beta k k^T) D S + beta k v^T`;
    'read', its keys erasing along a read of its own (`Gated.erase_read` `b`: the preconditioned delta rules',
    their keys preconditioned, `preconditioned.preconditioned_delta_rule`), `S' = (I - beta k b^T) D S + beta
    k v^T`, the correction `v - b^T D S` one write; 'untied', erase vectors of their own (`Gated.erase_key`
    `a`, `erase_read` `b`: GDN-2's, Comba's), `S' = (I - beta a b^T) D S + beta k v^T`; 'additive', RWKV7's
    diagonal plus low rank, `S' = (D - a b^T) S + beta k v^T`; 'values', Oja's rule's, erasing along the
    values under a decay along them (`log_decay` `[..., Hv, Dv]`), `S' = S D (I - beta v v^T) + beta k v^T`.
    `value_decay`, without the delta rule, decays the values' side too (two-sided GLA's value gates,
    `Gated.value_decay` `[..., Hv, Dv]`), `S' = D S E + beta k v^T`."""
    conv: Conv | None = None
    l2norm: bool = False
    scale: float | None = None
    decay: str = 'vector'
    gates: Gates | None = None
    delta: bool = True
    write_gate: bool = False
    rated: bool = False
    skip: bool = False
    output: OutputNorm | None = None
    write_bound: float | None = None
    unit_queries: bool = False
    decay_bound: float | None = None
    reads: Reads = 'after'
    own_weight: bool = False
    ranks: int = 1
    """The op's tokens a token's writes take, its ranks laid in turn (mamba_ssm's Mamba-3 MIMO), whose
    readout 'apart' reads none of them: its scores keep the pairs of later tokens alone (`i // R > j //
    R`, `mamba3_mimo_fwd.py`)."""
    turns: int = 0
    erase: Erase = 'tied'
    value_decay: bool = False
    substeps: int = 1
    """The delta rule's updates a token takes in turn (DeltaProduct's, fla's `chunk_gated_delta_product`): its
    keys, values and write gates `substeps` of each a token, the chunked op's laid in turn (`[B, S m, ...]`)
    and a decode token's on an axis of their own (`[rows, m, ...]`), its decay taken before the first and its
    query read after the last (the readout's row selector, kernel-templates.md §5); its queries and decays a
    token's."""
    coupled: str | None = None
    """A second state coupled to the memory, None for none: 'momentum', Momentum DeltaNet's (fla's
    momentum_delta_rule), the delta rule's error gathered into a momentum `M` the memory steps along, `w =
    p^T S - v`, `M' = mu M + k w^T`, `S' = alpha S - beta M'`, read `q^T S'`: its decay a head's `alpha =
    exp(log_decay)`, the momentum's `mu = exp(Gated.momentum_decay)`, the write gate `beta` at most
    `write_bound` (which its chunks' forms are bounded by, `dew.algebra.coupled_scan`), and its erase key `p`
    the keys (`erase` 'tied') or its own (`erase` 'untied', `Gated.erase_key`, its caller's to normalize and
    to scale, fla's by alpha); the state `[..., 2, Dk, Dv]`, the memory then the momentum. Its forms are the
    portable math's alone, derived from the pair's token update (`recurrence_chunk._delta_forms`)."""

    @property
    def inverse_bound(self) -> float | None:
        """The largest entry of the delta rule's chunk inverse `(I + A)^-1` its form proves, None where it
        proves none: keys of unit norm (the op's `l2norm`, or as `write_bound` declares) and write gates at
        most `b` (a sigmoid's from the `gates`, else the declared `write_bound`) with `b` at most two. Each
        transition `I - beta k k^T` then has its eigenvalues in [1 - b, 1], within [-1, 1], and a decay
        at most one scales it, so a product of them is no expansion; an entry below the diagonal is
        `-beta_i k_i^T (T_{i-1} ... T_{j+1}) k_j` (decays included), at most `b`, and a product of a chunk's
        partial inverse and its system, `w_i . k_j` with `|w_i| <= b`, too. Unnormalized keys with
        `beta |k|^2` past two grow it geometrically (an eigenvalue past -1)."""
        if not self.delta or self.erase != 'tied' or self.coupled is not None:
            return None
        sigmoid = self.gates is not None and self.gates.write == 'sigmoid'
        bound = 1.0 if self.l2norm and sigmoid else self.write_bound
        return bound if bound is not None and bound <= 2.0 else None

    @property
    def group(self) -> int:
        """The op tokens a readout groups (a token's `ranks` read apart from their writes), whole in every
        chunk a walk cuts (`recurrence_chunk._turned_or_plain`'s, `_portable`'s): one where it reads the
        whole state."""
        return self.ranks if self.reads == 'apart' else 1

    @property
    def apart(self) -> int:
        """The op tokens of a group whose writes a readout apart from them leaves out (`ranks`) that the
        template forms (`mainloop.Call.apart`: a decay a head or none, no delta rule, no own weight), zero
        where it reads the whole state or the portable math alone forms it."""
        formed = self.reads == 'apart' and not self.own_weight and self.decay != 'vector' and not self.delta
        return self.group if formed else 0

    @property
    def gated_write(self) -> bool:
        """Whether the step reads a write gate `beta`."""
        return self.delta or self.write_gate or self.gates is not None or self.rated

    def __post_init__(self):
        if self.reads not in ('after', 'before', 'apart'):
            raise ValueError(f"a readout reads the state 'after' its token's update, 'before' it or 'apart' "
                             f"from its own write, not {self.reads!r}")
        if self.ranks < 1 or (self.ranks > 1 and (self.reads != 'apart' or self.delta)):
            raise ValueError(f"a token's ranks are one or more, more only where its readout reads apart from "
                             f"their writes without the delta rule (whose writes are corrected values a "
                             f"group's read would take), not {self.ranks} reading {self.reads!r} with "
                             f"delta={self.delta}")
        if self.own_weight and self.delta:
            raise ValueError("a readout weighs a token's own write without the delta rule: the delta rule's "
                             "write is its corrected value, which reads the state its readout does not")
        if self.decay not in ('none', 'scalar', 'vector'):
            raise ValueError(f"a decay is 'none', 'scalar' or 'vector', not {self.decay!r}")
        decay_bound(self.decay_bound)
        if not isinstance(self.unit_queries, bool):
            raise TypeError(f"unit_queries declares the queries normalized or not, a bool: not "
                            f"{self.unit_queries!r}")
        if self.gates is not None and self.decay != 'scalar':
            raise ValueError('the gates form one log decay a head: their decay is scalar')
        if self.rated and (self.decay != 'scalar' or self.gates is not None):
            raise ValueError('a rated decay is one a head without the gates, which rate their own')
        if self.erase not in ('tied', 'read', 'untied', 'additive', 'values'):
            raise ValueError(f"an erase is 'tied', 'read', 'untied', 'additive' or 'values', not "
                             f"{self.erase!r}")
        if self.erase != 'tied' and not self.delta:
            raise ValueError(f"an erase of its own is the delta rule's: not {self}")
        if self.value_decay and (self.delta or self.erase != 'tied'):
            raise ValueError(f"a decay along the values is the plain recurrence's, without the delta rule or "
                             f"an erase of its own: not {self}")
        transition = self.erase != 'tied' or self.value_decay
        if self.substeps < 1 or (self.substeps > 1 and (not self.delta or self.reads != 'after' or self.rated
                                                        or self.gates is not None or self.skip
                                                        or self.conv is not None or transition)):
            raise ValueError(f"a token's delta-rule updates are one or more, more read after the last, its "
                             f"decay a given log decay's, unrated, without a skip, a conv or a transition of "
                             f"its own: not {self.substeps} under {self}")
        if transition and (self.reads != 'after' or self.ranks > 1 or self.own_weight or self.turns
                           or self.rated):
            raise ValueError(f"an erase of its own or a decay along the values is read after its token, "
                             f"without ranks, turns, an own write's weight or a rated decay: not {self}")
        if self.erase == 'values' and (self.skip or self.conv is not None or self.gates is not None):
            raise ValueError(f"an erase along the values takes no skip, conv or gates: not {self}")
        uncoupled = (not self.delta or self.decay != 'scalar' or self.gates is not None or self.rated
                     or self.reads != 'after' or self.ranks > 1 or self.own_weight or self.turns
                     or self.value_decay or self.substeps > 1 or self.skip
                     or self.erase not in ('tied', 'untied') or self.write_bound is None)
        if self.coupled is not None and (self.coupled != 'momentum' or uncoupled):
            raise ValueError(f"a coupled momentum ('momentum') is the delta rule's under a decay a head's, "
                             f"read after its token, of one update, its erase key the keys' or its own, its "
                             f"write gate bounded (`write_bound`), without gates, a rate, ranks, turns, an "
                             f"own write's weight, a decay along the values or a skip: not {self}")
        if self.turns < 0 or (self.turns and (self.delta or self.own_weight or self.rated
                                              or self.decay != 'scalar' or self.reads == 'before')):
            raise ValueError(f"a state turns its key pairs under a decay a head's, unrated, without the "
                             f"delta rule or an own write's weight, read after its token's turn, a count of "
                             f"none or more: not {self.turns} under {self}")


def decay_bound(bound: float | None) -> float | None:
    """`bound`, a least log decay a token (`Hooks.decay_bound`), where it is one: None, or a finite negative
    number; else ValueError, so no misstated bound (positive, zero, NaN or infinite) reads as a proof."""
    if bound is not None and not (math.isfinite(bound) and bound < 0):
        raise ValueError(f"a decay bound is the least log decay a token, finite and negative, or None where "
                         f"none is proven: not {bound!r}")
    return bound


def round_to_bf16(state, key, value):
    """`state` `[..., Dk, Dv]` fp32 rounded to bf16 stochastically: up in
    magnitude with the probability of the dropped low 16 bits, so a decay
    smaller than half a bf16 step still moves the state on average, where
    rounding to nearest would hold it. The noise is a hash of the value's
    bits and the token's `key` `[..., Dk]` and `value` `[..., Dv]`, so it is
    deterministic and differs from token to token (`fmix32`, the mixer every Dew noise takes;
    `precision.stochastic_bf16`, its rounding)."""
    return salted_bf16(state, key[..., :, None], value[..., None, :])


def salted_bf16(state, key, value):
    """`round_to_bf16` of `state` with its `key` and `value` broadcast to it as they are (a decode program's
    column of the state, a key's entry and the column's values: `templates.recurrence.step`)."""
    bits = jax.lax.bitcast_convert_type(state, jnp.uint32)
    salt = (jax.lax.bitcast_convert_type(key, jnp.uint32) * jnp.uint32(0x9E3779B1)
            + jax.lax.bitcast_convert_type(value, jnp.uint32) * jnp.uint32(0x85EBCA77))
    return stochastic_bf16(state, fmix32(bits ^ salt))


def update(state, query, key, value, decay, write, *, delta: bool, reads: Reads = 'after', own=None,
           turn=None, erase=None, additive: bool = False, across=None, reader=None):
    """One token over a state `[..., Dk, Dv]` in its working dtype, `decay` its decay per key dimension `[...,
    Dk]` or a head's `[..., 1]` (`exp(g)`) and `write` the write gate `[..., 1]`, a head's one: its output
    `[..., Dv]` and the updated state `S' = g S + (beta k) c^T`, `c = v - (g k)^T S` the delta rule's
    correction.
    The output reads the state before its update and the write apart, `(g q)^T S + q^T ((beta k) c^T)`,
    `q^T S'` as the chunked form reads it (`recurrence_algebra.chunk_output`): from `S'`, each entry's own
    rounding of `g S` put a gated delta net's decode tokens 1.30 to 1.38 times the chunked form's distance
    from float64 through its gated RMS norm, its draws to 6.3 times (qwen35-tiny's widths,
    probes/decode_vs_chunk.py; 1.00 so). The decay scales the query and key before their sums, and the
    query reads the write's every entry, so each sum's terms are `S'`'s parts': the write read as
    `(q . (beta k)) c` put `q . k` past fp32's largest at keys and queries of 2^63 where `q^T S'` stays
    finite, an infinite value gradient where float64's are normal (the decode step's contract).
    The write gate scales the key before any sum it enters (`beta k`, as the chunk form and fla write it):
    the one-pass form `(q decay)^T S + (q . k) write (v - (k decay)^T S)` overflowed where `q^T S'` stays
    finite, its `q . k` past fp32's largest at keys and queries of 2^80 against a write of 2^-80 (rows 1859
    and 1962). The write formed `[Dv, Dk]` and transposed, so its second order's sum over the keys runs
    along their last axis: formed `[Dk, Dv]`, a grad of grad's value gradient summed 128 keys past fp32's
    largest at its prefix (4.4e38) where float64's sum was 3.0e38 (the decode step's contract near the
    largest, 128-wide heads).
    Read `reads` 'before', the output is the state entering the token's, `q^T S`, the decay and the write
    after it; 'apart', the state after the update less the token's own write, `(g q)^T S`; `own` `[...,
    Dk]`, where given, adds the token's own write to the output weighed a key
    dimension, `(q * own)^T ((beta k) c^T)`, RWKV6's bonus (`Hooks.own_weight`), beside the query's whole
    read of it where it reads the state after the update, as the chunked form adds it. `turn` `[..., A]`,
    where given, turns the state's first `A` key pairs by `e^{-i turn}` before its decay (`Hooks.turns`,
    `scan.turned`), which a readout of the state entering the token reads none of. `erase` `(a, b)`, where
    given, erases with vectors of its own (`Hooks.erase` 'untied': `-(beta a) (b^T D S)`; with `additive`,
    'additive': `-a (b^T S)`) where the delta rule's correction takes its keys', the token writing its values
    as they are; `across` `[..., Dv]`, where given, decays the state's values' side too (`Hooks.value_decay`,
    `D S E`); `reader` `[..., Dk]`, where given, the vector the delta rule's correction reads the state along
    in place of its key (`Hooks.erase` 'read', `v - (g b)^T S`)."""
    def read(x, rows):
        return jnp.sum(x[..., :, None] * rows, axis=-2)

    entering = state
    if turn is not None:
        state = turned(state, turn[..., None, :], axis=-2)
    if across is not None:
        state = state * across[..., None, :]
    corrected = (value - read((key if reader is None else reader) * decay, state) if delta and erase is None
                 else value)
    outer = jnp.swapaxes(corrected[..., :, None] * (key * write)[..., None, :], -1, -2)
    if erase is not None:
        target, reader = erase
        weight = target if additive else target * write
        # Read as the delta rule's correction reads (`k^T D S`), the plain formula's order: the erase's
        # contract is its read's (`contract_support.assert_contract`: finite where the plain formula is).
        erased = read(reader if additive else reader * decay, state)
        outer = outer - weight[..., :, None] * erased[..., None, :]
    reading = query if reads == 'before' else query * decay
    out = read(reading, entering if reads == 'before' else state)
    # Read after the update, the token's write is the query's; its own weight, where given, adds to it.
    if reads == 'after':
        out = out + read(query, outer)
    if own is not None:
        out = out + read(query * own, outer)
    return out, state * decay[..., :, None] + outer


def _held(updated, key, value, dtype):
    """The updated state as it is held: bf16 rounded stochastically (`round_to_bf16`), else cast."""
    return round_to_bf16(updated, key, value) if dtype == jnp.bfloat16 else updated.astype(dtype)


def _gated_decay(gate, log_a, bias):
    """The gates' log decay `-exp(log_a) softplus(gate + bias)`, in the operands' dtype."""
    return -accurate_exp(log_a) * softplus(gate + bias)


def gated(gates: Gates, gate, log_a, bias, beta):
    """The log decay and the write gate `gates` forms from its pre-activations, in their dtype: with a
    sigmoid write gate the gated delta rule's (`_gated_decay`, `sigmoid(beta)`), with the step's
    Mamba2's (`-exp(log_a)` times the clipped step, and the step)."""
    if gates.write == 'sigmoid' and gates.limit is None:
        return _gated_decay(gate, log_a, bias), logistic(beta)
    step = softplus(gate + bias)
    if gates.limit is not None:
        step = jnp.clip(step, *gates.limit)
    return -accurate_exp(log_a) * step, step if gates.write == 'step' else logistic(beta)


def gated_rms(out, z, weight, norm: OutputNorm):
    """The epilogue over a head's output `[..., Dv]` in fp32 or wider: `rms_norm(o) weight act(z)`, its
    inverse RMS the one statistic every norm takes (`rms_statistics`)."""
    inverse = rms_statistics(out, norm.epsilon)[1]
    return rms_weighted(out, inverse, weight=weight, gate=GATE_ACTIVATIONS[norm.activation](z))


ROW = {"rows": True}
"""A field's metadata where it holds an entry a row (`Step`'s; the decode step shards those over the rows,
`step`)."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class Gated:
    """The operands the hooks read beside the queries, keys and values, a decode token's (`Step`) and a
    chunked sequence's (`recurrence_chunk.Chunk`) alike, each None where its hook is off: the log decay, the
    gates' pre-activations, the write gate, the skip's `D`, the epilogue's gate and weight, a rated decay's
    rate, the readout's weight of the token's own write (`Hooks.own_weight`), and the state's turn of its key
    pairs (`Hooks.turns`, `[..., Hv, turns]`). A value kind the hooks gain is a field here, an entry a row
    where its metadata says so (`ROW`)."""
    log_decay: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    gate: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    log_a: jax.Array | None = None
    bias: jax.Array | None = None
    beta: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    skip: jax.Array | None = None
    z: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    weight: jax.Array | None = None
    rate: jax.Array | None = None
    own: jax.Array | None = None
    turn: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    erase_key: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    erase_read: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    value_decay: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    momentum_decay: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    """A coupled momentum's log decay a value head's (`Hooks.coupled`), `[..., Hv]`."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Step(Gated):
    """A decode token's operands, each `[rows, ...]`; a hook's are None where it is off.

    Without a conv, `query` and `key` `[rows, Hk, Dk]` and `value` `[rows, Hv, Dv]` (the query scaled unless
    the hooks' l2 norm scales it); with one, `token` `[rows, C]`, `held` `[rows, K - 1, C]` and `taps` `[C,
    K]`, and `conv_bias` `[C]` where the conv has a bias. `log_decay` is `[rows, Hv]` (scalar) or `[rows, Hv,
    Dk]` (vector), or with the gates `gate` `[rows, Hv]`, `log_a` and `bias` `[Hv]`. `beta` `[rows, Hv]` is
    the write gate (a raw beta with the gates), `skip` `[Hv]` the skip's `D`, `z` `[rows, Hv, Dv]` and
    `weight` `[Dv]` the epilogue's, `rate` `[Hv]` a rated decay's, at most zero (`Hooks.rated`), `own`
    `[Hv, Dk]` the readout's weight of the token's own write (`Hooks.own_weight`). Under `Hooks.substeps`
    (DeltaProduct's) `key`, `value` and `beta` hold the token's `m` updates on an axis of their own, `[rows,
    m, ...]`. `state` is
    `[rows, Hv, Dk, Dv]`, fp32 or bf16 (rounded by `round_to_bf16`), or wider for the reference. `active`
    `[rows]` marks the rows that draw. A pytree of its arrays, so it crosses `jit` and a transformation as a
    token's operands do; the `Hooks` are its static form. Its fields are the decode launch's operands
    (`templates.recurrence.step`), those an entry a row marked (`ROW`)."""
    state: jax.Array = dataclasses.field(metadata=ROW)
    query: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    key: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    value: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    token: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    held: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    taps: jax.Array | None = None
    conv_bias: jax.Array | None = None
    active: jax.Array | None = dataclasses.field(default=None, metadata=ROW)


def _pieces(hooks: Hooks, value_heads: int, key_width: int, value_width: int, channels: int):
    """The key heads a conv token holds, from its channels: queries and keys `Hk Dk` each, values
    `Hv Dv`."""
    key_heads = (channels - value_heads * value_width) // (2 * key_width)
    whole = 2 * key_heads * key_width + value_heads * value_width
    if key_heads <= 0 or value_heads % key_heads or whole != channels:
        raise ValueError(f'{channels} conv channels are not queries and keys of {key_width} and values of '
                         f'{value_width} over {value_heads} value heads')
    return key_heads


def _names(hooks: Hooks) -> tuple[str, ...]:
    """The `Step` operands a launch reads, in its order, after `active`; `state` last."""
    streams = (('token', 'held', 'taps', *('conv_bias',) * hooks.conv.bias) if hooks.conv is not None
               else ('query', 'key', 'value'))
    if hooks.gates is not None:
        gates = ('gate', *('beta',) * (hooks.gates.write == 'sigmoid'), 'log_a', 'bias')
    elif hooks.rated:
        gates = ('rate', 'beta')
    else:
        gates = ('log_decay',) * (hooks.decay != 'none') + ('beta',) * hooks.gated_write
    erasing = (('erase_key', 'erase_read') if hooks.erase in ('untied', 'additive')
               else ('erase_read',) if hooks.erase == 'read' else ())
    if hooks.coupled is not None:
        erasing = (*('erase_key',) * (hooks.erase == 'untied'), 'momentum_decay')
    return (*streams, *gates, *erasing, *('value_decay',) * hooks.value_decay, *('turn',) * bool(hooks.turns),
            *('own',) * hooks.own_weight,
            *('skip',) * hooks.skip,
            *('z', 'weight') * (hooks.output is not None), 'state')


def read(hooks: Hooks, operands, *, optional: tuple[str, ...] = ()) -> dict:
    """The operands `hooks` read (`_names`) from a `Step` or a chunked op's operands
    (`recurrence_chunk.Chunk`), by name, each held but those `optional` leaves out, else a ValueError
    naming those it lacks."""
    given = {name: getattr(operands, name) for name in _names(hooks)}
    missing = [name for name, array in given.items() if array is None and name not in optional]
    if missing:
        raise ValueError(f"{hooks} reads {', '.join(missing)}, which its operands do not hold")
    return {name: array for name, array in given.items() if array is not None}


def _activated(pre, conv: Conv):
    """The conv's sums `pre` activated as `conv` says: SiLU, or as they are."""
    return pre if conv.activation is None else silu(pre)


def conv_pieces(conv: Conv, key_heads: int, value_heads: int, key_width: int,
                value_width: int) -> dict[str, tuple[int, int, int]]:
    """Each stream's `(first channel, heads, width)` in a conv's channels, in its `order`: the queries and
    keys `Hk` heads of `Dk`, the values `Hv` of `Dv`."""
    shapes = {'q': (key_heads, key_width), 'k': (key_heads, key_width), 'v': (value_heads, value_width)}
    found, first = {}, 0
    for name in conv.order:
        heads, width = shapes[name]
        found[name] = (first, heads, width)
        first += heads * width
    return found


def _split(convolved, conv: Conv, key_heads: int, value_heads: int, key_width: int, value_width: int):
    """A conv's channels `[rows, C]` as the queries and keys `[rows, Hk, Dk]` and the values `[rows, Hv,
    Dv]` they hold (`conv_pieces`): the one splitter the reference's conv and its one-tap form share."""
    rows = convolved.shape[0]
    pieces = conv_pieces(conv, key_heads, value_heads, key_width, value_width)
    return tuple(convolved[:, first:first + heads * width].reshape(rows, heads, width)
                 for first, heads, width in (pieces[name] for name in 'qkv'))


def _prologue(hooks: Hooks, given: dict, work):
    """The reference's queries `[rows, Hv, Dk]`, keys, values `[rows, Hv, Dv]`, each in `work`, and
    the moved history (None without a conv): the conv's sums in `work` over the history rounded to
    the token's dtype, the activation, the l2 norm, the key heads repeated over their value heads."""
    value_heads, key_width, value_width = given['state'].shape[1:]
    history = None
    if hooks.conv is not None:
        token, held = given['token'], given['held']
        key_heads = _pieces(hooks, value_heads, key_width, value_width, token.shape[-1])
        window = jnp.concatenate([held.astype(token.dtype), token[:, None]], axis=1).astype(work)
        taps, width = given['taps'].astype(work), window.shape[1]
        # Each tap's inputs a slice of the window and its weights a column of the taps, the conv's one
        # multiply and add a tap.
        # A conv's bias spread at each row's token, its gradient summed compensated (`short_conv.conv_bias`).
        bias = (short_conv.conv_bias(given['conv_bias'].astype(work), (token.shape[0], 1))
                if hooks.conv.bias else None)
        pre = short_conv.preactivation([window[:, tap:tap + 1] for tap in range(width)],
                                       [taps[:, tap] for tap in range(width)], bias)[:, 0]
        query, key, value = _split(_activated(pre, hooks.conv), hooks.conv, key_heads, value_heads,
                                   key_width, value_width)
        # The history and the token, then all but the oldest: a one-tap conv holds none.
        history = jnp.concatenate([held, token[:, None].astype(held.dtype)], axis=1)[:, 1:]
    else:
        query, key, value = (given[name].astype(work) for name in ('query', 'key', 'value'))
    if hooks.l2norm:
        query, key = (l2norm(x) for x in (query, key))
        query = query * (key_width ** -0.5 if hooks.scale is None else hooks.scale)
    elif hooks.scale is not None:
        query = query * hooks.scale
    group = value_heads // query.shape[1]
    query, key = (jnp.repeat(x, group, axis=1) for x in (query, key))
    return query, key, value, history


def log_gates(hooks: Hooks, given: dict, work):
    """The log decay and the write gate, each in `work`, from the operands `given` by name (`read`): the
    gates' (`gated`), a rated decay's (the rate times the write gate), or as given; None for a decay
    or a write gate the hooks have not."""
    if hooks.gates is not None:
        beta = given['beta'].astype(work) if hooks.gates.write == 'sigmoid' else None
        return gated(hooks.gates, given['gate'].astype(work), given['log_a'].astype(work),
                     given['bias'].astype(work), beta)
    if hooks.rated:
        beta = given['beta'].astype(work)
        return given['rate'].astype(work) * beta, beta
    return (None if hooks.decay == 'none' else given['log_decay'].astype(work),
            given['beta'].astype(work) if hooks.gated_write else None)


def skipped(hooks: Hooks, out, value, given: dict, work):
    """An output `[..., Hv, Dv]` in `work` with the skip `D v` added where the hooks have one
    (`skip_added`)."""
    return skip_added(out, given['skip'], value, work) if hooks.skip else out


def skip_added(out, skip, value, work):
    """An output `[..., Hv, Dv]` in `work` with the skip `D v` added (`_skip`), `D` `[Hv]` against values
    `[..., Hv, Dv]`, each in `work`: the chunked op's epilogue's and a layer's adding its own skip
    (Mamba-3's), one owner of the skip's product and its gradient's sum."""
    return out + _skip(skip.astype(work), value.astype(work))


def _skip_product(skip, value):
    """`D v`: the skip `D` `[Hv]` against values `[..., Hv, Dv]`."""
    return skip[:, None] * value


def _skip_forward(skip, value):
    return _skip_product(skip, value), (skip, value)


def _skip_backward(saved, operands, grad):
    del operands
    skip, value = saved
    # Rows an Explicit axis splits contract into `D`'s own placement, which JAX cannot infer from a
    # contraction over a split dimension (a data-parallel layout's hybrid, test_parallelism_matrix).
    placed = explicit_spec(value)
    stated = (None if placed is None
              else jax.sharding.NamedSharding(placed.mesh, jax.typeof(skip).sharding.spec))
    leading = 'abcdefg'[:value.ndim - 2]
    summed = jnp.einsum(f'{leading}hv,{leading}hv->h', grad, value, out_sharding=stated)
    # `D`'s gradient formed in its dtype before the value's cotangent leaves (row 6027).
    d_value, d_skip = with_gradients_formed((grad * skip[:, None]).astype(value.dtype),
                                            summed.astype(skip.dtype))
    return d_skip, d_value


_skip = contracted("test_the_decode_step_holds_every_order")(
    linear_tangent(_skip_forward, _skip_backward, _skip_product, kind=Protected()))
"""`D v` (`_skip_product`) differentiated at every order (`generation.linear_tangent`): forward mode the
product's own tangent, reverse mode its gradient in `D` summing every token's and value's product as a
product (at JAX's configured precision) rather than one reduction over them all: XLA:CPU's reduction,
without YNNPACK's fusions, summed a chunked call's `B S Dv` products in turn, 2.1 times the token-by-token
recurrence's distance from float64 (Mamba2's rated decay, test_recurrence_stability after the gate compiled
both arms without them). A custom VJP alone refused `jax.jvp` through a decode step's reference."""


def normed(hooks: Hooks, out, given: dict, work):
    """An output `[..., Hv, Dv]` in `work` through the epilogue's gated RMS norm where the hooks have one."""
    if hooks.output is None:
        return out
    if hooks.output.read_in is not None:
        out = out.astype(hooks.output.read_in).astype(work)
    return gated_rms(out, given['z'].astype(work), given['weight'].astype(work), hooks.output)


def _gates(hooks: Hooks, given: dict, work, key_width: int):
    """The reference's decay `[rows, Hv, Dk]` per key dimension, `[rows, Hv, 1]` a head's (none's one), and
    write gate `[rows, Hv]`."""
    g, beta = log_gates(hooks, given, work)
    rows, heads = given['state'].shape[:2]
    if g is None:
        decay = jnp.ones((rows, heads, 1), work)
    else:
        decay = accurate_exp(g)
        decay = (decay[..., None] if hooks.decay == 'scalar'
                 else jnp.broadcast_to(decay, (rows, heads, key_width)))
    write = jnp.ones((rows, heads), work) if beta is None else beta
    return decay, write


def step_reference(hooks: Hooks, operands: Step, *, dtype=None):
    """`(output [rows, Hv, Dv], history' or None, state')` of a decode token, in XLA: each piece of
    arithmetic in at least fp32 (float64 operands in float64), the state held as it came (bf16
    rounded by `round_to_bf16`), the output in `dtype` (the value's or the token's without it)."""
    if hooks.ranks > 1:
        raise NotImplementedError(f"a decode token of {hooks.ranks} ranks: a token's step takes one op "
                                  "token, its ranks the caller's (Mamba-3's `step_mamba3`)")
    given = read(hooks, operands)
    state = given['state']
    operand = given['token' if hooks.conv is not None else 'value'].dtype
    work = at_least_fp32(jnp.promote_types(operand, state.dtype))
    key_width = state.shape[2]
    if hooks.substeps > 1:
        out, key, value, updated = _substeps(hooks, given, work)
        history = None
    elif hooks.coupled is not None:
        out, updated, key, value, history = _momentum_step(hooks, given, work)
    elif hooks.erase == 'values':
        query, key, value, history = _prologue(hooks, given, work)
        out, updated = _values_erased(hooks, given, state.astype(work), query, key, value, work)
    else:
        query, key, value, history = _prologue(hooks, given, work)
        decay, write = _gates(hooks, given, work, key_width)
        own = given['own'].astype(work) if hooks.own_weight else None
        turn = given['turn'].astype(work) if hooks.turns else None
        out, updated = update(state.astype(work), query, key, value, decay, write[..., None],
                              delta=hooks.delta, reads=hooks.reads, own=own, turn=turn,
                              erase=_erasing(hooks, given, work), additive=hooks.erase == 'additive',
                              across=accurate_exp(given['value_decay'].astype(work)) if hooks.value_decay
                              else None, reader=_reader(hooks, given, work))
    out = normed(hooks, skipped(hooks, out, value, given, work), given, work)
    # A pair's keys and values salt both its states alike (`round_to_bf16`).
    paired = hooks.coupled is not None
    held = _held(updated, key[:, :, None] if paired else key, value[:, :, None] if paired else value,
                 state.dtype)
    if operands.active is not None:
        active = jnp.asarray(operands.active, bool)
        held = jnp.where(active.reshape(-1, *([1] * (held.ndim - 1))), held, state)
        out = jnp.where(active[:, None, None], out, 0.0)
        if history is not None:
            history = jnp.where(active[:, None, None], history, given['held'])
    return out.astype(operand if dtype is None else dtype), history, held


def _substeps(hooks: Hooks, given: dict, work):
    """A decode token's `hooks.substeps` delta-rule updates in turn (`Hooks.substeps`, DeltaProduct's) in
    `work`, the state held there between them (`_substep`). `(output, last keys, last values, state)`, the
    output the state after the last read by the query."""
    counts = {name: given[name].shape[1] for name in ('key', 'value', 'beta') if name in given}
    if any(count != hooks.substeps for count in counts.values()):
        raise ValueError(f"a token of {hooks.substeps} updates takes keys, values and write gates of "
                         f"{hooks.substeps} each, not {counts}")
    found = _substep(hooks, given, work, given['state'].astype(work), 0)
    for substep in range(1, hooks.substeps):
        found = _substep(hooks, given, work, found[3], substep)
    return found


def _substep(hooks: Hooks, given: dict, work, state, substep: int):
    """One of a decode token's delta-rule updates (`_substeps`) on `state`: its keys, values and write gates
    `given`'s at `substep`, its decay the token's on the first alone. `(output, keys, values, state)`."""
    taken = {**given, **{name: given[name][:, substep] for name in ('key', 'value', 'beta') if name in given}}
    if substep and 'log_decay' in given:
        taken['log_decay'] = jnp.zeros_like(given['log_decay'])
    query, key, value, _ = _prologue(hooks, taken, work)
    decay, write = _gates(hooks, taken, work, state.shape[2])
    out, state = update(state, query, key, value, decay, write[..., None], delta=True, reads='after')
    return out, key, value, state


def _momentum_step(hooks: Hooks, given: dict, work):
    """A decode token of a coupled momentum (`Hooks.coupled` 'momentum') in `work` on its pair `[rows, Hv, 2,
    Dk, Dv]`, the memory then the momentum: the erase key's read of the memory less the values, `w = p^T S -
    v`, gathered into the momentum, `M' = mu M + k w^T`, the memory stepping along it, `S' = alpha S - beta
    M'`, read `q^T S'` (fla's recurrent_momentum_delta_rule_ref). `(output, state', keys, values,
    history)`."""
    state = given['state'].astype(work)
    memory, momentum = state[:, :, 0], state[:, :, 1]
    query, key, value, history = _prologue(hooks, {**given, 'state': given['state'][:, :, 0]}, work)
    erase = key
    if hooks.erase == 'untied':
        group = value.shape[1] // given['erase_key'].shape[1]
        erase = jnp.repeat(given['erase_key'].astype(work), group, axis=1)
    alpha, mu = (accurate_exp(given[name].astype(work))[..., None, None]
                 for name in ('log_decay', 'momentum_decay'))
    error = jnp.sum(erase[..., :, None] * memory, axis=-2) - value
    momentum = mu * momentum + key[..., :, None] * error[..., None, :]
    memory = alpha * memory - given['beta'].astype(work)[..., None, None] * momentum
    out = jnp.sum(query[..., :, None] * memory, axis=-2)
    return out, jnp.stack([memory, momentum], axis=2), key, value, history


def _erasing(hooks: Hooks, given: dict, work):
    """A decode token's erase vectors `(a, b)` `[rows, Hv, Dk]` (`Hooks.erase` 'untied' or 'additive'), in
    `work`, each key head's over its value heads as the keys are (`_prologue`); None for the keys' own."""
    if hooks.erase not in ('untied', 'additive'):
        return None
    group = given['state'].shape[1] // given['erase_key'].shape[1]
    return tuple(jnp.repeat(given[name].astype(work), group, axis=1) for name in ('erase_key', 'erase_read'))


def _reader(hooks: Hooks, given: dict, work):
    """A decode token's read vector `b` `[rows, Hv, Dk]` (`Hooks.erase` 'read'), in `work`, each key head's
    over its value heads as the keys are; None for the keys' own."""
    if hooks.erase != 'read':
        return None
    group = given['state'].shape[1] // given['erase_read'].shape[1]
    return jnp.repeat(given['erase_read'].astype(work), group, axis=1)


def _values_erased(hooks: Hooks, given: dict, state, query, key, value, work):
    """A decode token of Oja's rule (`Hooks.erase` 'values'), in `work`: on the transposed state `S^T`, the
    delta rule of keys `v`, values `k` and a decay along its keys, the token's `log_decay` `[rows, Hv, Dv]`
    (a head's `[rows, Hv]` where scalar), read along the transposed state's values, `S'^T q`.
    `(output, state')`. The gated correction `beta k - beta (v^T D S^T)`, `S D` read once as the token reads
    it, is one exact sum of its exact terms (`aligned.aligned_sum`), rounded once: corrected ungated, `k -
    v^T D S^T` passed fp32's largest at a memory of -2^127 and a key of 2^127 under a gate of 1/4, whose
    memory and output stay -2^126; gated before its sum, `beta v` lost a value of 2^-60 under a gate of
    2^-100 to underflow, its read of a memory of 2^120 a normal -2^-40; gated after it, four values of 1/2
    read a memory of 2^127 as 2^128, where the gated read of 1/4 is 2^126. The gate's gradient sums its
    write's and its erase's exactly: apart, each rounded, a key of 2^30 and a read of 2^30 + 1 cancelled to
    zero where it is -1/2."""
    if hooks.decay == 'none':
        decay = jnp.ones(value.shape, work)
    else:
        decay = accurate_exp(given['log_decay'].astype(work))
        decay = jnp.broadcast_to(decay[..., None] if hooks.decay == 'scalar' else decay, value.shape)
    decayed = decay[..., :, None] * jnp.swapaxes(state, -1, -2)
    gate = given['beta'].astype(work)[..., None, None]
    # The gate one factor of both terms, so its gradient is one aligned sum of theirs.
    terms = [(gate, key[..., None, :]), (gate, -value[..., :, None], decayed)]
    corrected = aligned_sum(terms, axis=-2)[..., 0, :]
    transposed = decayed + value[..., :, None] * corrected[..., None, :]
    return jnp.sum(transposed * query[..., None, :], axis=-1), jnp.swapaxes(transposed, -1, -2)


def _one_tap(hooks: Hooks, conv: Conv, operands: Step, given: dict, dtype):
    """`step` of a one-tap `conv`: its token scaled by the taps and activated in at least fp32, as the
    launch's prologue sums one term, kept in that precision as the prologue keeps it, then the launch
    without a conv, the output in the token's dtype unless `dtype` names another, the empty history
    returned as it came."""
    token = given['token']
    value_heads, key_width, value_width = given['state'].shape[1:]
    key_heads = _pieces(hooks, value_heads, key_width, value_width, token.shape[-1])
    work = at_least_fp32(token.dtype)
    pre = token.astype(work) * given['taps'][:, 0].astype(work)
    if conv.bias:
        pre = pre + short_conv.conv_bias(given['conv_bias'].astype(work), (token.shape[0],))
    query, key, value = _split(_activated(pre, conv), conv, key_heads, value_heads, key_width, value_width)
    unconvolved = dataclasses.replace(operands, token=None, held=None, taps=None, conv_bias=None,
                                      query=query, key=key, value=value)
    out, _, state = step(dataclasses.replace(hooks, conv=None), unconvolved,
                         dtype=token.dtype if dtype is None else dtype)
    return out, given['held'], state


def step(hooks: Hooks, operands: Step, *, dtype=None):
    """`(output [rows, Hv, Dv], history' or None, state')` of a decode token in one launch
    (`templates.recurrence.step`), `step_reference`'s arithmetic: the output in `dtype` (the value's, or the
    token's with a conv), the history in its dtype, the state as it is held and updated in place
    where the caller lets it go. A row `active` marks False keeps its history and state, its output
    zeros. A one-tap conv holds no history, so its token is a channel-wise scale: it is taken here
    and the launch steps the queries, keys and values it makes. Differentiated at every order through
    `step_reference` (`generation.linear_tangent`), the launch having no derivative of its own: a token
    traced for `jax.jvp`, however a jit, a scan or a remat staged it, takes the reference's tangent and
    the launch's value (row 2395)."""
    if hooks.ranks > 1:
        raise NotImplementedError(f"a decode token of {hooks.ranks} ranks: a token's step takes one op "
                                  "token, its ranks the caller's (Mamba-3's `step_mamba3`)")
    def portable(operands):
        return step_reference(hooks, operands, dtype=dtype)

    def pulled(saved, operands, cotangent):
        del saved
        # The rows' cotangents leave only with the parameters' formed (`with_gradients_formed`): a field no
        # row marks (`ROW`: the taps, the gates' log_a and bias, the skip's D, the epilogue's weight, a rate).
        (pulled_operands,) = jax.vjp(portable, *operands)[1](cotangent)
        rows = {field.name for field in dataclasses.fields(pulled_operands) if field.metadata == ROW}
        values = {field.name: getattr(pulled_operands, field.name)
                  for field in dataclasses.fields(pulled_operands)}
        inputs, formed = with_gradients_formed(
            {name: value for name, value in values.items() if name in rows},
            {name: value for name, value in values.items() if name not in rows})
        return (dataclasses.replace(pulled_operands, **inputs, **formed),)

    # The template's decode step turns no state (`Hooks.turns`), takes one update a token (`Hooks.substeps`)
    # and the delta rule's own erase beside a decay along the keys alone (`Hooks.erase`, `Hooks.value_decay`):
    # such a token's launch is the reference.
    referenced = (hooks.turns or hooks.substeps > 1 or hooks.erase != 'tied' or hooks.value_decay
                  or hooks.coupled is not None)
    launched = functools.partial(step_reference if referenced else _launched, hooks, dtype=dtype)
    return contracted("test_the_decode_step_holds_every_order")(
        linear_tangent(lambda operands: (launched(operands), ()), pulled, portable,
                       kind=KernelGlue()))(operands)


def _launched(hooks: Hooks, operands: Step, *, dtype):
    """`step`'s launch on its operands: the recurrence template's decode step on Mosaic GPU
    (`templates.recurrence.step`), interpreted off a GPU, on each shard of the rows (`manual_map`): a launch
    left to the partitioner gathers its operands to every device; a one-tap conv's token taken as a scale
    first."""
    from .templates.recurrence import mainloop, step as template

    given = read(hooks, operands)
    if hooks.conv is not None and given['held'].shape[1] == 0:
        return _one_tap(hooks, hooks.conv, operands, given, dtype)
    family, interpret = mainloop.issued(jnp.float32)
    launch = functools.partial(template.step, hooks, dtype=dtype, family=family, interpret=interpret)
    batch, heads, _, width = operands.state.shape
    rows, _ = grouped_heads((batch, 1, heads, width), 1)
    names = [field.name for field in dataclasses.fields(Step) if getattr(operands, field.name) is not None]
    P = jax.sharding.PartitionSpec
    specs = tuple(P(rows) if field.metadata.get('rows') else P() for field in dataclasses.fields(Step)
                  if field.name in names)

    def local(*arrays):
        return launch(dataclasses.replace(operands, **dict(zip(names, arrays, strict=True))))

    moved = P(rows) if hooks.conv is not None else None
    return manual_map(local, specs, (P(rows), moved, P(rows)))(*(getattr(operands, name) for name in names))


def _cumsum_kernel(chunk_ref, s_ref, o_ref, *rest, chunk: int, aligned: bool, scale: float | None,
                   reverse: bool):
    """fla's `chunk_local_cumsum_scalar_kernel` (fla/ops/utils/cumsum.py:33-73) for one chunk of one
    head, of each column of a gate per key dimension (fla's vector kernel, :90-133) too: forward, fla's
    inclusive cumulative sum over its tokens and its `scale`, in float64 and rounded once to fp32 (fla's
    fp32 sum and its fp32 scaling put Kimi delta attention's final memory 1.9 times fla's recorded torch's
    distance from float64, 1.8 with this, SE-dbg-kda-stage-1a95620b8); every kernel takes the difference
    of two positions as a range's exponent. With a second output (`rest`), the float64 sum's remainder
    below its fp32 rounding, so a range's exponent is the his' difference and the los' (`algebra.apart`):
    a steep step's rounding otherwise loses the mild steps after it. `reverse`, the decay's gradient,
    sums from the end compensated, each running sum a `(hi, lo)` pair (`compensated_add`) taken token
    by token and stored rounded once."""
    start, count = chunked_launch._chunk_of(chunk_ref, chunk, aligned)
    if not reverse:
        @pl.when(count > 0)
        def _():
            tokens = chunked_launch.read_rows(chunked_launch._rows(s_ref, start, chunk), count)
            with jax.enable_x64():
                total = jnp.cumsum(tokens.astype(jnp.float64), axis=0)
                total = total if scale is None else total * scale
                rounded = total.astype(jnp.float32)
                remainder = (total - rounded.astype(jnp.float64)).astype(jnp.float32)
            chunked_launch.write_rows(chunked_launch._rows(o_ref, start, chunk), count, rounded)
            for lo_ref in rest:
                chunked_launch.write_rows(chunked_launch._rows(lo_ref, start, chunk), count, remainder)
        return

    def step(i, total):
        row = start + count - 1 - i
        token = chunked_launch._rows(s_ref, row, 1)[...].astype(jnp.float32)
        if scale is not None:
            token = token * scale
        total = compensated_add(total, (token, jnp.zeros_like(token)))
        chunked_launch._rows(o_ref, row, 1)[...] = total[0] + total[1]
        return total

    zero = jnp.zeros((1, *s_ref.shape[1:]), jnp.float32)
    jax.lax.fori_loop(0, count, step, (zero, zero))


@dataclasses.dataclass(frozen=True)
class ChunkLocalCumsum(chunked_launch._Launch):
    """`cumulated`'s launch (tuning.KernelOp), fla's `chunk_local_cumsum`."""

    revision: int = 2

    NAME = "cumulated"
    KEYS, VALUES = None, None
    PICKS = chunked_launch._picks(chunked_launch.Tiles(0, 0, 2, 1))

    def _fwd(self, g: jax.Array, chunk_spans: jax.Array, sequence_spans: jax.Array, *, chunk: int,
             scale: float | None, reverse: bool, interpret: bool, config: chunked_launch.Tiles,
             compensated: bool = False, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans = chunked_launch.Sequences(chunk_spans, sequence_spans)
        warps, stages = config.warps, config.stages
        batch, length, heads, *columns = (int(size) for size in g.shape)
        aligned = chunked_launch._aligned(spans, batch, length, chunk)
        # A decay per key dimension in 64-column blocks, fla's vector kernel's BS.
        blocks = [min(64, width) for width in columns]
        tiles = [width // block for width, block in zip(columns, blocks, strict=True)]
        spec = pl.BlockSpec((batch * length, None, *blocks), lambda i_t, h, *i_k: (0, h, *i_k))
        outputs = 1 + compensated
        out = launches.pallas_call(
            functools.partial(_cumsum_kernel, chunk=chunk, aligned=aligned, scale=scale, reverse=reverse),
            grid=(spans.chunks.shape[0], heads, *tiles),
            in_specs=[chunked_launch._CHUNK, spec], out_specs=[spec] * outputs,
            out_shape=[jax.ShapeDtypeStruct((batch * length, heads, *columns), jnp.float32)] * outputs,
            compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
            interpret=interpret, name="recurrence_cumulated",
        )(spans.chunks, *chunked_launch._flat(g))
        out = tuple(part.reshape(g.shape) for part in out)
        return (out if compensated else out[0]), None


_CHUNK_LOCAL_CUMSUM = ChunkLocalCumsum()


def cumulated(g, *, chunk: int = 64, scale: float | None = None, reverse: bool = False,
              compensated: bool = False, spans: chunked_launch.Sequences | None = None,
              interpret: bool = False):
    """The chunked recurrences' first stage, their decay's prefix: `g` `[B, T, H]`, or `[B, T, H, K]`
    per key dimension, cumulated within each chunk of `chunk` tokens of each sequence (`spans`, the
    rows where None), fp32: fla's `chunk_local_cumsum`
    (cumsum.py:262-341), `scale` RCP_LN2 for the forward's base-2 decay, `reverse` for its gradient.
    Forward is the prefix summed and scaled in float64 and rounded once (fla's is plain fp32), whose
    difference between two positions every kernel takes as a range's exponent, or where `compensated`
    its `(hi, lo)` halves, the remainder below the rounding too (`algebra.apart`); `reverse` the sums
    from the end, each rounded once. fla autotunes only `num_warps` over 1, 2, 4 and 8 here."""
    batch, length = g.shape[:2]
    spans = chunked_launch.sequences(batch, length, chunk=chunk) if spans is None else spans
    return _CHUNK_LOCAL_CUMSUM(g, *spans, chunk=chunk, scale=scale, reverse=reverse, compensated=compensated,
                               interpret=interpret)
