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

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from dew._hashing import fmix32
from dew.nn.derivatives import KernelGlue, Protected
from dew.nn.kernels import launches
from dew.nn.precision import (
    at_least_fp32,
    compensated_add,
    l2norm,
    rms_statistics,
    rms_weighted,
    stochastic_bf16,
)

from ..sharding import explicit_spec
from . import chunked_launch, short_conv
from .generation import linear_tangent


@dataclasses.dataclass(frozen=True)
class Conv:
    """The prologue's short conv: a token's channels `[C]`, its queries, keys and values head by head
    in turn, over `held` `[K - 1, C]` history with `taps` `[C, K]`, then `activation` ('silu' or None).
    The history is read rounded to the token's dtype, as the layer's conv reads it."""
    activation: str | None = 'silu'


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
    with `epsilon`, `activation` the gate's ('silu' or 'sigmoid')."""
    epsilon: float
    activation: str = 'silu'


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
    one); a skip `D v`, and the epilogue; and `floor`, the least log decay a token a layer declares
    (Kimi delta attention's `lower_bound`, -5), which a decay a key dimension's factored chunk form reads
    as its range's bound (`recurrence_algebra.factored_range`); and `write_bound`, the largest write gate
    a layer declares of l2-normalized keys (its config's: 1 for a sigmoid's, 2 for twice a sigmoid's, the
    negative-eigenvalue delta rule's), which bounds the delta rule's chunk inverse (`inverse_bound`). The
    chunked op (`recurrence_chunk.chunk`) takes the same form."""
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
    floor: float | None = None
    write_bound: float | None = None

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
        if not self.delta:
            return None
        sigmoid = self.gates is not None and self.gates.write == 'sigmoid'
        bound = 1.0 if self.l2norm and sigmoid else self.write_bound
        return bound if bound is not None and bound <= 2.0 else None

    @property
    def gated_write(self) -> bool:
        """Whether the step reads a write gate `beta`."""
        return self.delta or self.write_gate or self.gates is not None or self.rated

    def __post_init__(self):
        if self.decay not in ('none', 'scalar', 'vector'):
            raise ValueError(f"a decay is 'none', 'scalar' or 'vector', not {self.decay!r}")
        if self.gates is not None and self.decay != 'scalar':
            raise ValueError('the gates form one log decay a head: their decay is scalar')
        if self.rated and (self.decay != 'scalar' or self.gates is not None):
            raise ValueError('a rated decay is one a head without the gates, which rate their own')


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


def update(state, query, key, value, decay, write, *, delta: bool):
    """One token over a state `[..., Dk, Dv]` in its working dtype, `decay` per key dimension `[...,
    Dk]` (`exp(g)`) and `write` the write gate `[..., 1]`, a head's one: its output `[..., Dv]` and the
    updated state.
    The output reads the updated state, `q^T S'`, as the recurrence states it, with the delta rule and
    without: the one-pass form `(q decay)^T S + (q . k) write (v - (k decay)^T S)` overflowed where
    `q^T S'` stays finite, its `q . k` past fp32's largest at keys and queries of 2^80 against a write of
    2^-80 (rows 1859 and 1962). The write gate scales the key before the write's outer product (`beta k`,
    as the chunk form and fla write it), so the write's transpose, `(beta k) . q` against the output's
    cotangent, takes the gate before the sum: after it, `k . q` past fp32's largest at keys and queries of
    2^63 put an infinite value gradient where the chunk form's was finite (ledger 2470)."""
    corrected = value - jnp.sum((key * decay)[..., :, None] * state, axis=-2) if delta else value
    updated = state * decay[..., :, None] + (key * write)[..., :, None] * corrected[..., None, :]
    return jnp.sum(query[..., :, None] * updated, axis=-2), updated


def _held(updated, key, value, dtype):
    """The updated state as it is held: bf16 rounded stochastically (`round_to_bf16`), else cast."""
    return round_to_bf16(updated, key, value) if dtype == jnp.bfloat16 else updated.astype(dtype)


def _gated_decay(gate, log_a, bias):
    """The gates' log decay `-exp(log_a) softplus(gate + bias)`, in the operands' dtype."""
    return -jnp.exp(log_a) * jax.nn.softplus(gate + bias)


def gated(gates: Gates, gate, log_a, bias, beta):
    """The log decay and the write gate `gates` forms from its pre-activations, in their dtype: with a
    sigmoid write gate the gated delta rule's (`_gated_decay`, `sigmoid(beta)`), with the step's
    Mamba2's (`-exp(log_a)` times the clipped step, and the step)."""
    if gates.write == 'sigmoid' and gates.limit is None:
        return _gated_decay(gate, log_a, bias), jax.nn.sigmoid(beta)
    step = jax.nn.softplus(gate + bias)
    if gates.limit is not None:
        step = jnp.clip(step, *gates.limit)
    return -jnp.exp(log_a) * step, step if gates.write == 'step' else jax.nn.sigmoid(beta)


def gated_rms(out, z, weight, norm: OutputNorm):
    """The epilogue over a head's output `[..., Dv]` in fp32 or wider: `rms_norm(o) weight act(z)`, its
    inverse RMS the one statistic every norm takes (`rms_statistics`)."""
    inverse = rms_statistics(out, norm.epsilon)[1]
    return rms_weighted(out, inverse, weight=weight, gate=z, activation=norm.activation)


ROW = {"rows": True}
"""A field's metadata where it holds an entry a row (`Step`'s; a decode launch keys its rows' count bucketed,
`StepLaunch.ROWS`)."""


@dataclasses.dataclass(frozen=True, kw_only=True)
class Gated:
    """The operands the hooks read beside the queries, keys and values, a decode token's (`Step`) and a
    chunked sequence's (`recurrence_chunk.Chunk`) alike, each None where its hook is off: the log
    decay, the gates' pre-activations, the write gate, the skip's `D`, the epilogue's gate and
    weight, and a rated decay's rate. A value kind the hooks gain is a field here, an entry a row where its
    metadata says so (`ROW`)."""
    log_decay: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    gate: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    log_a: jax.Array | None = None
    bias: jax.Array | None = None
    beta: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    skip: jax.Array | None = None
    z: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    weight: jax.Array | None = None
    rate: jax.Array | None = None


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Step(Gated):
    """A decode token's operands, each `[rows, ...]`; a hook's are None where it is off.

    Without a conv, `query` and `key` `[rows, Hk, Dk]` and `value` `[rows, Hv, Dv]` (the query scaled
    unless the hooks' l2 norm scales it); with one, `token` `[rows, C]`, `held` `[rows, K - 1, C]`
    and `taps` `[C, K]`. `log_decay` is `[rows, Hv]` (scalar) or `[rows, Hv, Dk]` (vector), or with
    the gates `gate` `[rows, Hv]`, `log_a` and `bias` `[Hv]`. `beta` `[rows, Hv]` is the write gate
    (a raw beta with the gates), `skip` `[Hv]` the skip's `D`, `z` `[rows, Hv, Dv]` and `weight`
    `[Dv]` the epilogue's, `rate` `[Hv]` a rated decay's, at most zero (`Hooks.rated`). `state` is
    `[rows, Hv, Dk, Dv]`, fp32 or bf16 (rounded by `round_to_bf16`), or wider for the reference.
    `active` `[rows]` marks the rows that draw. A pytree
    of its arrays, so it crosses `jit` and a transformation as a token's operands do; the `Hooks` are
    its static form. Its fields are the decode launch's operands and their names (`OPERANDS`, `StepLaunch`),
    those an entry a row marked (`ROW`)."""
    state: jax.Array = dataclasses.field(metadata=ROW)
    query: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    key: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    value: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    token: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    held: jax.Array | None = dataclasses.field(default=None, metadata=ROW)
    taps: jax.Array | None = None
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


OPERANDS = tuple(field.name for field in dataclasses.fields(Step) if field.name != 'active')
"""`StepLaunch`'s array operands beside `active`, by name: `Step`'s fields."""


def _names(hooks: Hooks) -> tuple[str, ...]:
    """The `Step` operands a launch reads, in its order, after `active`; `state` last."""
    streams = ('token', 'held', 'taps') if hooks.conv is not None else ('query', 'key', 'value')
    if hooks.gates is not None:
        gates = ('gate', *('beta',) * (hooks.gates.write == 'sigmoid'), 'log_a', 'bias')
    elif hooks.rated:
        gates = ('rate', 'beta')
    else:
        gates = ('log_decay',) * (hooks.decay != 'none') + ('beta',) * hooks.gated_write
    return (*streams, *gates, *('skip',) * hooks.skip, *('z', 'weight') * (hooks.output is not None), 'state')


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
    return pre if conv.activation is None else pre * jax.nn.sigmoid(pre)


def _split(convolved, key_heads: int, value_heads: int, key_width: int, value_width: int):
    """A conv's channels `[rows, C]` as the queries and keys `[rows, Hk, Dk]` and the values `[rows, Hv,
    Dv]` they hold, in that order: the one splitter the reference's conv and its one-tap form share."""
    rows = convolved.shape[0]
    query, key, value = jnp.split(convolved, [key_heads * key_width, 2 * key_heads * key_width], axis=-1)
    return (query.reshape(rows, key_heads, key_width), key.reshape(rows, key_heads, key_width),
            value.reshape(rows, value_heads, value_width))


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
        pre = short_conv.preactivation([window[:, tap:tap + 1] for tap in range(width)],
                                       [taps[:, tap] for tap in range(width)], None)[:, 0]
        query, key, value = _split(_activated(pre, hooks.conv), key_heads, value_heads, key_width,
                                   value_width)
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
    """An output `[..., Hv, Dv]` in `work` with the skip `D v` added where the hooks have one (`_skip`)."""
    return out + _skip(given['skip'].astype(work), value.astype(work)) if hooks.skip else out


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
    return summed.astype(skip.dtype), (grad * skip[:, None]).astype(value.dtype)


_skip = linear_tangent(_skip_forward, _skip_backward, _skip_product, kind=Protected())
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
    return gated_rms(out, given['z'].astype(work), given['weight'].astype(work), hooks.output)


def _gates(hooks: Hooks, given: dict, work, key_width: int):
    """The reference's decay per key dimension `[rows, Hv, Dk]` and write gate `[rows, Hv]`."""
    g, beta = log_gates(hooks, given, work)
    rows, heads = given['state'].shape[:2]
    if g is None:
        decay = jnp.ones((rows, heads, key_width), work)
    else:
        decay = jnp.exp(g)
        decay = decay[..., None] if hooks.decay == 'scalar' else decay
        decay = jnp.broadcast_to(decay, (rows, heads, key_width))
    write = jnp.ones((rows, heads), work) if beta is None else beta
    return decay, write


def step_reference(hooks: Hooks, operands: Step, *, dtype=None):
    """`(output [rows, Hv, Dv], history' or None, state')` of a decode token, in XLA: each piece of
    arithmetic in at least fp32 (float64 operands in float64), the state held as it came (bf16
    rounded by `round_to_bf16`), the output in `dtype` (the value's or the token's without it)."""
    given = read(hooks, operands)
    state = given['state']
    operand = given['token' if hooks.conv is not None else 'value'].dtype
    work = at_least_fp32(jnp.promote_types(operand, state.dtype))
    key_width = state.shape[2]
    query, key, value, history = _prologue(hooks, given, work)
    decay, write = _gates(hooks, given, work, key_width)
    out, updated = update(state.astype(work), query, key, value, decay, write[..., None], delta=hooks.delta)
    out = normed(hooks, skipped(hooks, out, value, given, work), given, work)
    held = _held(updated, key, value, state.dtype)
    if operands.active is not None:
        active = jnp.asarray(operands.active, bool)
        held = jnp.where(active[:, None, None, None], held, state)
        out = jnp.where(active[:, None, None], out, 0.0)
        if history is not None:
            history = jnp.where(active[:, None, None], history, given['held'])
    return out.astype(operand if dtype is None else dtype), history, held


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
    query, key, value = _split(_activated(pre, conv), key_heads, value_heads, key_width, value_width)
    unconvolved = dataclasses.replace(operands, token=None, held=None, taps=None, query=query, key=key,
                                      value=value)
    out, _, state = step(dataclasses.replace(hooks, conv=None), unconvolved,
                         dtype=token.dtype if dtype is None else dtype)
    return out, given['held'], state


def step(hooks: Hooks, operands: Step, *, dtype=None):
    """`(output [rows, Hv, Dv], history' or None, state')` of a decode token in one launch
    (`StepLaunch`), `step_reference`'s arithmetic: the output in `dtype` (the value's, or the
    token's with a conv), the history in its dtype, the state as it is held and updated in place
    where the caller lets it go. A row `active` marks False keeps its history and state, its output
    zeros. A one-tap conv holds no history, so its token is a channel-wise scale: it is taken here
    and the launch steps the queries, keys and values it makes. Differentiated at every order through
    `step_reference` (`generation.linear_tangent`), the launch having no derivative of its own: a token
    traced for `jax.jvp`, however a jit, a scan or a remat staged it, takes the reference's tangent and
    the launch's value (row 2395)."""
    def portable(operands):
        return step_reference(hooks, operands, dtype=dtype)

    def pulled(saved, operands, cotangent):
        del saved
        return jax.vjp(portable, *operands)[1](cotangent)

    launched = functools.partial(_launched, hooks, dtype=dtype)
    return linear_tangent(lambda operands: (launched(operands), ()), pulled, portable,
                          kind=KernelGlue())(operands)


def _launched(hooks: Hooks, operands: Step, *, dtype):
    """`step`'s launch on its operands: the recurrence template's decode step on Mosaic GPU
    (`templates.recurrence.step`), interpreted off a GPU; a one-tap conv's token taken as a scale first."""
    from .templates import bundles
    from .templates.recurrence import mainloop, step as template

    given = read(hooks, operands)
    if hooks.conv is not None and given['held'].shape[1] == 0:
        return _one_tap(hooks, hooks.conv, operands, given, dtype)
    family = mainloop.family_of(jnp.float32)
    return template.step(hooks, operands, dtype=dtype, family=family or bundles.INTERPRETED,
                         interpret=family is None)


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
