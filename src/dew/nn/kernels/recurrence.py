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
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from dew.nn.precision import (
    L2NORM_EPS,
    at_least_fp32,
    compensated_add,
    float32_einsum,
    float32_products,
    l2norm,
    products_dtype,
    stochastic_bf16,
)

from ..sharding import explicit_spec
from . import chunked_launch, norm_swiglu, short_conv, tuning
from .generation import INHERENT, Refusal, triton_runs


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
    as its range's bound (`recurrence_algebra.factored_range`). The chunked op (`recurrence_chunk.chunk`)
    takes the same form."""
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
    deterministic and differs from token to token (`precision.stochastic_bf16`, its rounding)."""
    bits = jax.lax.bitcast_convert_type(state, jnp.uint32)
    salt = (jax.lax.bitcast_convert_type(key, jnp.uint32)[..., :, None] * jnp.uint32(0x9E3779B1)
            + jax.lax.bitcast_convert_type(value, jnp.uint32)[..., None, :] * jnp.uint32(0x85EBCA77))
    mixed = bits ^ salt
    mixed = (mixed ^ (mixed >> 16)) * jnp.uint32(0x7FEB352D)
    mixed = (mixed ^ (mixed >> 15)) * jnp.uint32(0x846CA68B)
    return stochastic_bf16(state, mixed ^ (mixed >> 16))


def update(state, query, key, value, decay, write, *, delta: bool, in_kernel: bool = False):
    """One token over a state `[..., Dk, Dv]` in its working dtype, `decay` per key dimension `[...,
    Dk]` (`exp(g)`) and `write` the write gate, broadcast against the values `[..., Dv]`: its output
    `[..., Dv]` and the updated state.
    Both readouts are products with the old state, `(q decay)^T S + (q . k) d` with `d = write (v -
    (k decay)^T S)` under the delta rule and `write v` without it, so one pass over the state serves
    them and the write. `q . k` is a product (`float32_einsum`), summed as Dew sums one, not a reduction
    whose order XLA chooses by the shape it is batched at: under a skip that cancels it (`(q . k) write`
    against `D`), a reduction a token at a time and the same in the portable rule's batched adjoint
    rounded a last bit apart, which put the portable gradient's cancelled row 4.9 times the token
    recurrence's distance from float64 (test_recurrence_stability's rated site under resets on chunk
    edges, seed 6). A kernel's body (`in_kernel`, the fused step's) sums it in its own registers, where
    Pallas's Triton lowering takes no platform-dependent product."""
    read_query = jnp.sum((query * decay)[..., :, None] * state, axis=-2)
    corrected = value - jnp.sum((key * decay)[..., :, None] * state, axis=-2) if delta else value
    written = corrected * write
    score = (jnp.sum(query * key, axis=-1) if in_kernel else float32_einsum('...d,...d->...', query, key))
    out = read_query + score[..., None] * written
    return out, state * decay[..., :, None] + key[..., :, None] * written[..., None, :]


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
    """The epilogue over a head's output `[..., Dv]` in fp32 or wider: `rms_norm(o) weight act(z)`,
    the mean a product by the reciprocal, as XLA's."""
    inverse = jax.lax.rsqrt(jnp.sum(out * out, axis=-1, keepdims=True) * (1.0 / out.shape[-1]) + norm.epsilon)
    return out * inverse * weight * norm_swiglu.activation_and_slope(z, norm.activation)[0]


@dataclasses.dataclass(frozen=True, kw_only=True)
class Gated:
    """The operands the hooks read beside the queries, keys and values, a decode token's (`Step`) and a
    chunked sequence's (`recurrence_chunk.Chunk`) alike, each None where its hook is off: the log
    decay, the gates' pre-activations, the write gate, the skip's `D`, the epilogue's gate and
    weight, and a rated decay's rate. A value kind the hooks gain is a field here."""
    log_decay: jax.Array | None = None
    gate: jax.Array | None = None
    log_a: jax.Array | None = None
    bias: jax.Array | None = None
    beta: jax.Array | None = None
    skip: jax.Array | None = None
    z: jax.Array | None = None
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
    its static form."""
    state: jax.Array
    query: jax.Array | None = None
    key: jax.Array | None = None
    value: jax.Array | None = None
    token: jax.Array | None = None
    held: jax.Array | None = None
    taps: jax.Array | None = None
    active: jax.Array | None = None


def _pieces(hooks: Hooks, value_heads: int, key_width: int, value_width: int, channels: int):
    """The key heads a conv token holds, from its channels: queries and keys `Hk Dk` each, values
    `Hv Dv`."""
    key_heads = (channels - value_heads * value_width) // (2 * key_width)
    whole = 2 * key_heads * key_width + value_heads * value_width
    if key_heads <= 0 or value_heads % key_heads or whole != channels:
        raise ValueError(f'{channels} conv channels are not queries and keys of {key_width} and values of '
                         f'{value_width} over {value_heads} value heads')
    return key_heads


OPERANDS = ('state', 'query', 'key', 'value', 'token', 'held', 'taps', 'log_decay', 'gate', 'beta', 'log_a',
            'bias', 'skip', 'z', 'weight', 'rate')
"""`StepLaunch`'s array operands after `active`, in its order."""


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
        pre = jnp.sum(window * jnp.swapaxes(given['taps'].astype(work), 0, 1)[None], axis=1)
        convolved = pre if hooks.conv.activation is None else pre * jax.nn.sigmoid(pre)
        rows = token.shape[0]
        split = [key_heads * key_width, 2 * key_heads * key_width]
        query, key, value = jnp.split(convolved, split, axis=-1)
        query, key = (x.reshape(rows, key_heads, key_width) for x in (query, key))
        value = value.reshape(rows, value_heads, value_width)
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


@jax.custom_vjp
def _skip(skip, value):
    """`D v` of the skip `D` `[Hv]` and values `[..., Hv, Dv]`, whose gradient in `D` sums every token's and
    value's product as a product (at `float32_products`' precision) rather than one reduction over them
    all: XLA:CPU's reduction, without YNNPACK's fusions, summed a chunked call's `B S Dv` products in turn,
    2.1 times the token-by-token recurrence's distance from float64 (Mamba2's rated decay,
    test_recurrence_stability after the gate compiled both arms without them)."""
    return skip[:, None] * value


def _skip_forward(skip, value):
    return _skip(skip, value), (skip, value)


def _skip_backward(residual, grad):
    skip, value = residual
    # Rows an Explicit axis splits contract into `D`'s own placement, which JAX cannot infer from a
    # contraction over a split dimension (a data-parallel layout's hybrid, test_parallelism_matrix).
    placed = explicit_spec(value)
    stated = (None if placed is None
              else jax.sharding.NamedSharding(placed.mesh, jax.typeof(skip).sharding.spec))
    leading = 'abcdefg'[:value.ndim - 2]
    summed = jnp.einsum(f'{leading}hv,{leading}hv->h', grad, value, out_sharding=stated,
                        precision=float32_products(products_dtype(grad, value)))
    return summed.astype(skip.dtype), (grad * skip[:, None]).astype(value.dtype)


_skip.defvjp(_skip_forward, _skip_backward)


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


def _kernel(*refs, hooks: Hooks, group: int, key_heads: int, interpret: bool):
    """One row and one value head of a decode token (`step`): the prologue's conv and l2 norm, the
    gates, the update and the epilogue, each piece of arithmetic `step_reference`'s in fp32, its
    quotients and roots rounded once (`short_conv`'s). The conv's channels are read where the row holds
    them, `[K - 1, C]` history and `[C]` token, queries, keys and values in turn; the history the next
    token reads is written apart, its key head's queries and keys by the head's first value head, so no
    program reads a history another moved on; an idle row's as it was."""
    names = _names(hooks)
    inputs = dict(zip(('active', *names), refs[:len(names) + 1], strict=True))
    outputs = refs[len(names) + 1:]
    state_out, out_ref = outputs[0], outputs[-1]
    state_ref = inputs['state']
    head = pl.program_id(1)
    active = inputs['active'][0] != 0
    key_width, value_width = state_ref.shape

    def scalar(name):
        """A head's one entry of a `[1]` block, as a one-entry vector the arithmetic broadcasts."""
        return inputs[name][...].astype(jnp.float32)

    conv = (_ConvRefs(inputs['token'], inputs['held'], inputs['taps'], outputs[1], key_heads, key_width,
                      value_width, head, group) if hooks.conv is not None else None)

    @pl.when(active)
    def _():
        windows = conv.windows() if conv is not None else None
        if hooks.conv is not None and conv is not None and windows is not None:
            query, key, value = conv.convolved(windows, hooks.conv, interpret)
        else:
            query, key, value = (inputs[name][...].astype(jnp.float32) for name in ('query', 'key', 'value'))
        if hooks.l2norm:
            query, key = (x[None] for x in (query, key))
            query = query * short_conv._inverse_norm(query, L2NORM_EPS, interpret=interpret)
            query = query * (key_width ** -0.5 if hooks.scale is None else hooks.scale)
            key = key * short_conv._inverse_norm(key, L2NORM_EPS, interpret=interpret)
            query, key = query[0], key[0]
        elif hooks.scale is not None:
            query = query * hooks.scale
        if hooks.gates is not None and hooks.gates.write == 'sigmoid' and hooks.gates.limit is None:
            decay = jnp.exp(_gated_decay(scalar('gate'), scalar('log_a'), scalar('bias')))
            write_gate = short_conv._sigmoid(scalar('beta'), interpret=interpret)
        elif hooks.gates is not None:
            step = jax.nn.softplus(scalar('gate') + scalar('bias'))
            if hooks.gates.limit is not None:
                step = jnp.clip(step, *hooks.gates.limit)
            decay = jnp.exp(-jnp.exp(scalar('log_a')) * step)
            write_gate = (step if hooks.gates.write == 'step'
                          else short_conv._sigmoid(scalar('beta'), interpret=interpret))
        elif hooks.rated:
            write_gate = scalar('beta')
            decay = jnp.exp(scalar('rate') * write_gate)
        else:
            decay = (jnp.ones((1,), jnp.float32) if hooks.decay == 'none'
                     else jnp.exp(inputs['log_decay'][...].astype(jnp.float32)))
            write_gate = scalar('beta') if hooks.gated_write else jnp.ones((1,), jnp.float32)
        decay = jnp.broadcast_to(decay, key.shape)
        out, updated = update(state_ref[...].astype(jnp.float32), query, key, value, decay, write_gate,
                              delta=hooks.delta, in_kernel=True)
        if hooks.skip:
            out = out + scalar('skip') * value
        if hooks.output is not None:
            gate, weight = (inputs[name][...].astype(jnp.float32) for name in ('z', 'weight'))
            out = gated_rms(out, gate, weight, hooks.output)
        state_out[...] = _held(updated, key, value, state_out.dtype)
        out_ref[...] = out.astype(out_ref.dtype)
        if conv is not None and windows is not None:
            conv.write([stream[1:] for stream in windows])

    @pl.when(jnp.logical_not(active))
    def _():
        out_ref[...] = jnp.zeros(out_ref.shape, out_ref.dtype)
        if conv is not None:
            conv.write(conv.held())


@dataclasses.dataclass(frozen=True)
class _ConvRefs:
    """A decode program's view of the prologue's conv: the row's token `[C]` and history `[K - 1, C]`,
    the taps `[C, K]` and the moved history it writes, its value head's queries, keys and values the
    channels `pieces` names."""
    token: Any
    history: Any
    taps: Any
    moved: Any
    key_heads: int
    key_width: int
    value_width: int
    head: Any
    group: int

    @property
    def pieces(self):
        """Each stream's first channel and width: its key head's queries and keys, its own values."""
        key_head = self.head // self.group
        return ((key_head * self.key_width, self.key_width),
                ((self.key_heads + key_head) * self.key_width, self.key_width),
                (2 * self.key_heads * self.key_width + self.head * self.value_width, self.value_width))

    def windows(self):
        """Each stream's conv window, `K` rows `[1, D]` fp32: the held history rounded to the token's
        dtype, as the layer's conv reads it, then the token."""
        count = self.taps.shape[-1]

        def window(first, width):
            channels = pl.ds(first, width)
            held = [self.history[j, channels][None, :].astype(self.token.dtype).astype(jnp.float32)
                    for j in range(count - 1)]
            return [*held, self.token[channels][None, :].astype(jnp.float32)]

        return [window(*piece) for piece in self.pieces]

    def convolved(self, windows, conv: Conv, interpret: bool):
        """The queries, keys and values `[D]`: each stream's taps over its window, then the activation."""
        streams = []
        for stream, (first, width) in zip(windows, self.pieces, strict=True):
            taps = [self.taps[pl.ds(first, width), j].astype(jnp.float32) for j in range(len(stream))]
            pre = short_conv._preactivation(stream, taps, None)
            streams.append(pre if conv.activation is None
                           else pre * short_conv._sigmoid(pre, interpret=interpret))
        return tuple(x[0] for x in streams)

    def held(self):
        """Each stream's history as it was, for an idle row."""
        count = self.taps.shape[-1]
        return [[self.history[j, pl.ds(*piece)][None, :] for j in range(count - 1)] for piece in self.pieces]

    def write(self, moved):
        """Each stream's moved history, a row at a time: the values, and the queries and keys where the
        program is its key head's first."""
        def store(piece, rows):
            for j, row in enumerate(rows):
                self.moved[j, pl.ds(*piece)] = row[0].astype(self.moved.dtype)

        pieces = self.pieces
        store(pieces[2], moved[2])

        @pl.when(self.head % self.group == 0)
        def _():
            store(pieces[0], moved[0])
            store(pieces[1], moved[1])


@dataclasses.dataclass(frozen=True)
class Launch:
    """A decode launch's program: `block` value columns of a head's state (0 for all of them, which a
    conv or the epilogue's norm needs, each reading a whole head) and its warps."""
    block: int
    warps: int


WARPS = Launch(0, 8)
"""The launch over a whole head."""
COLUMNS = Launch(32, 8)
"""The launch over value columns: 32 a program, 16 to 64 measured within 3% on an RTX 4080."""


@dataclasses.dataclass(frozen=True)
class StepLaunch(tuning.KernelOp):
    """A decode token in one launch over rows and value heads (`_kernel`), its `hooks` inlined. Off a GPU
    it interprets."""

    config_cls: ClassVar[type[Launch]] = Launch

    ROWS: ClassVar[tuple[str, ...]] = ('active', 'query', 'key', 'value', 'token', 'held', 'log_decay',
                                       'gate', 'beta', 'z', 'state')

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        """Normal floats and every row active, so a timed launch steps each row's state and history."""
        return [jnp.ones(spec.shape, spec.dtype) if jnp.issubdtype(spec.dtype, jnp.integer) else array
                for spec, array in zip(specs, super().operands(specs, seed), strict=True)]

    def _fwd(self, active: jax.Array, state: jax.Array, query: jax.Array | None, key: jax.Array | None,
             value: jax.Array | None, token: jax.Array | None, held: jax.Array | None, taps: jax.Array | None,
             log_decay: jax.Array | None, gate: jax.Array | None, beta: jax.Array | None,
             log_a: jax.Array | None, bias: jax.Array | None, skip: jax.Array | None, z: jax.Array | None,
             weight: jax.Array | None, rate: jax.Array | None, *, hooks: Hooks, dtype: str, interpret: bool,
             config: Launch,
             return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        arrays = {'state': state, 'query': query, 'key': key, 'value': value, 'token': token, 'held': held,
                  'taps': taps, 'log_decay': log_decay, 'gate': gate, 'beta': beta, 'log_a': log_a,
                  'bias': bias, 'skip': skip, 'z': z, 'weight': weight, 'rate': rate}
        given = {name: arrays[name] for name in _names(hooks)}
        rows, value_heads, key_width, value_width = state.shape
        key_heads = (_pieces(hooks, value_heads, key_width, value_width, given['token'].shape[-1])
                     if hooks.conv is not None else given['query'].shape[1])

        # A hook that reads a whole head takes a whole-head config (`_whole`).
        block = config.block or value_width

        def whole(array):
            """An array every program reads whole."""
            return pl.BlockSpec(array.shape, lambda r, h, c: (0,) * array.ndim)

        def per_row(array):
            """A row's whole block."""
            return pl.BlockSpec((None, *array.shape[1:]), lambda r, h, c: (r, *(0,) * (array.ndim - 1)))

        def keyed(array):
            """The program's key head's `[rows, Hk, Dk]` block, `[Dk]`."""
            group = value_heads // key_heads
            return pl.BlockSpec((None, None, key_width), lambda r, h, c: (r, h // group, 0))

        def valued(array):
            """The program's head's and columns' block of a `[rows, Hv, Dv]` array, `[block]`."""
            return pl.BlockSpec((None, None, block), lambda r, h, c: (r, h, c))

        def one(array, *, rows: bool):
            """The program's head's one entry, `[1]`."""
            return (pl.BlockSpec((None, 1), lambda r, h, c: (r, h)) if rows
                    else pl.BlockSpec((1,), lambda r, h, c: (h,)))

        def keyed_by_value(array):
            """The program's value head's `[rows, Hv, Dk]` block, `[Dk]`."""
            return pl.BlockSpec((None, None, key_width), lambda r, h, c: (r, h, 0))

        held_state = pl.BlockSpec((None, None, key_width, block), lambda r, h, c: (r, h, 0, c))
        specs = {
            'token': per_row, 'held': per_row, 'taps': whole, 'weight': whole, 'query': keyed, 'key': keyed,
            'value': valued, 'z': valued, 'state': lambda array: held_state,
            'gate': functools.partial(one, rows=True), 'beta': functools.partial(one, rows=True),
            'log_a': functools.partial(one, rows=False), 'bias': functools.partial(one, rows=False),
            'skip': functools.partial(one, rows=False), 'rate': functools.partial(one, rows=False),
            'log_decay': keyed_by_value if hooks.decay == 'vector' else functools.partial(one, rows=True),
        }
        flags = pl.BlockSpec((None, 1), lambda r, h, c: (r, 0))
        in_specs = [flags, *(specs[name](given[name]) for name in given)]
        out_specs = [held_state]
        out_shape = [jax.ShapeDtypeStruct(state.shape, state.dtype)]
        if hooks.conv is not None:
            out_specs.append(per_row(given['held']))
            out_shape.append(jax.ShapeDtypeStruct(given['held'].shape, given['held'].dtype))
        out_specs.append(valued(None))
        out_shape.append(jax.ShapeDtypeStruct((rows, value_heads, value_width), jnp.dtype(dtype)))
        kernel = functools.partial(_kernel, hooks=hooks, group=value_heads // key_heads, key_heads=key_heads,
                                   interpret=interpret)
        return tuple(pl.pallas_call(
            kernel, grid=(rows, value_heads, value_width // block), in_specs=in_specs, out_specs=out_specs,
            out_shape=out_shape, input_output_aliases={len(given): 0},
            compiler_params=plgpu.CompilerParams(num_warps=config.warps),
            interpret=interpret, name="recurrence_step",
        )(active, *given.values())), None

    def _get_heuristics_config(self, ba) -> Launch:
        if _whole(ba.kwargs['hooks']):
            return WARPS
        return dataclasses.replace(COLUMNS, block=min(COLUMNS.block, ba.arguments['state'].shape[-1]))

    def pinnable(self, ba, config) -> bool:
        # A program's block of value columns: a whole head where a hook reads one (`_whole`), else a whole
        # head or a block dividing the state's width.
        width = ba.arguments['state'].shape[-1]
        return isinstance(config, Launch) and (
            config.block == 0 or (not _whole(ba.kwargs['hooks']) and 0 < config.block <= width
                                  and width % config.block == 0))

    def unpinnable(self, ba, config) -> str:
        if not isinstance(config, Launch):
            return super().unpinnable(ba, config)
        return ("its block of value columns is neither the whole head a hook here reads whole nor a block "
                "dividing the state's width")


def _whole(hooks: Hooks) -> bool:
    """Whether a program needs a head's whole state: a conv reads all of a head's values, the epilogue's
    norm all of its output."""
    return hooks.conv is not None or hooks.output is not None


def refusal(state, *, tangent: bool = False) -> Refusal | None:
    """Why the launch turns down a `[rows, heads, Dk, Dv]` `state`, or None where it takes it: it runs
    on a GPU Dew's Triton kernels run on (`triton_runs`), holds the state in fp32 or bf16 and computes
    in fp32 (a float64 state keeps its own precision in `step_reference`), takes widths that are powers
    of two (Triton's blocks), and updates the state in place, which JAX's Pallas gives no tangent, so a
    token traced for forward-mode derivatives (`tangent`) is no launch's."""
    dk, dv = state.shape[-2:]
    return next((Refusal(code, detail, row) for holds, code, detail, row in (
        (not tangent, 'forward_mode', "a JVP trace needs forward-mode derivatives; the launch updates its "
         "state in place, which JAX's Pallas gives no tangent", INHERENT),
        (triton_runs(), 'capability', "the launch needs a GPU of sm80 or later", INHERENT),
        (state.dtype in (jnp.float32, jnp.bfloat16), 'precision',
         f"the launch holds a state in fp32 or bf16 and computes in fp32, not {state.dtype}", INHERENT),
        (all(width > 0 and width & (width - 1) == 0 for width in (dk, dv)), 'widths',
         f"the launch's blocks take power-of-two widths, not {dk} by {dv}",
         'roadmap: recurrence step widths'),
    ) if not holds), None)


def _one_tap(hooks: Hooks, conv: Conv, operands: Step, given: dict, dtype):
    """`step` of a one-tap `conv`: its token scaled by the taps and activated in at least fp32, as the
    launch's prologue sums one term, kept in that precision as the prologue keeps it, then the launch
    without a conv, the output in the token's dtype unless `dtype` names another, the empty history
    returned as it came."""
    token = given['token']
    rows, (value_heads, key_width, value_width) = token.shape[0], given['state'].shape[1:]
    key_heads = _pieces(hooks, value_heads, key_width, value_width, token.shape[-1])
    work = at_least_fp32(token.dtype)
    pre = token.astype(work) * given['taps'][:, 0].astype(work)
    convolved = pre if conv.activation is None else pre * jax.nn.sigmoid(pre)
    query, key, value = jnp.split(convolved, [key_heads * key_width, 2 * key_heads * key_width], axis=-1)
    unconvolved = dataclasses.replace(
        operands, token=None, held=None, taps=None, query=query.reshape(rows, key_heads, key_width),
        key=key.reshape(rows, key_heads, key_width), value=value.reshape(rows, value_heads, value_width))
    out, _, state = step(dataclasses.replace(hooks, conv=None), unconvolved,
                         dtype=token.dtype if dtype is None else dtype)
    return out, given['held'], state


def step(hooks: Hooks, operands: Step, *, dtype=None):
    """`(output [rows, Hv, Dv], history' or None, state')` of a decode token in one launch
    (`StepLaunch`), `step_reference`'s arithmetic: the output in `dtype` (the value's, or the
    token's with a conv), the history in its dtype, the state as it is held and updated in place
    where the caller lets it go. A row `active` marks False keeps its history and state, its output
    zeros. A one-tap conv holds no history, so its token is a channel-wise scale: it is taken here
    and the launch steps the queries, keys and values it makes."""
    given = read(hooks, operands)
    if hooks.conv is not None and given['held'].shape[1] == 0:
        return _one_tap(hooks, hooks.conv, operands, given, dtype)
    rows = given['state'].shape[0]
    flags = (jnp.ones(rows, jnp.int32) if operands.active is None
             else jnp.asarray(operands.active).astype(jnp.int32))[:, None]
    operand = given['token' if hooks.conv is not None else 'value'].dtype
    if 'beta' in given:
        given['beta'] = given['beta'].reshape(rows, -1)
    if 'log_decay' in given and hooks.decay == 'scalar':
        given['log_decay'] = given['log_decay'].reshape(rows, -1)
    named = jnp.dtype(operand if dtype is None else dtype).name
    # Positional, None where a hook is off.
    state, *history, out = StepLaunch()(flags, *(given.get(name) for name in OPERANDS), hooks=hooks,
                                        dtype=named, interpret=not triton_runs())
    return out, history[0] if history else None, state


def _cumsum_kernel(chunk_ref, s_ref, o_ref, *, chunk: int, aligned: bool, scale: float | None, reverse: bool):
    """fla's `chunk_local_cumsum_scalar_kernel` (fla/ops/utils/cumsum.py:33-73) for one chunk of one
    head, of each column of a gate per key dimension (fla's vector kernel, :90-133) too: forward, fla's
    inclusive cumulative sum over its tokens and its `scale`, in float64 and rounded once to fp32 (fla's
    fp32 sum and its fp32 scaling put Kimi delta attention's final memory 1.9 times fla's recorded torch's
    distance from float64, 1.8 with this, SE-dbg-kda-stage-1a95620b8); every kernel takes the difference
    of two positions as a range's exponent. `reverse`, the decay's gradient,
    sums from the end compensated, each running sum a `(hi, lo)` pair (`compensated_add`) taken token
    by token and stored rounded once."""
    start, count = chunked_launch._chunk_of(chunk_ref, chunk, aligned)
    if not reverse:
        @pl.when(count > 0)
        def _():
            tokens = chunked_launch.read_rows(chunked_launch._rows(s_ref, start, chunk), count)
            with jax.enable_x64():
                total = jnp.cumsum(tokens.astype(jnp.float64), axis=0)
                total = (total if scale is None else total * scale).astype(jnp.float32)
            chunked_launch.write_rows(chunked_launch._rows(o_ref, start, chunk), count, total)
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
             return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        spans = chunked_launch.Sequences(chunk_spans, sequence_spans)
        warps, stages = config.warps, config.stages
        batch, length, heads, *columns = (int(size) for size in g.shape)
        aligned = chunked_launch._aligned(spans, batch, length, chunk)
        # A decay per key dimension in 64-column blocks, fla's vector kernel's BS.
        blocks = [min(64, width) for width in columns]
        tiles = [width // block for width, block in zip(columns, blocks, strict=True)]
        spec = pl.BlockSpec((batch * length, None, *blocks), lambda i_t, h, *i_k: (0, h, *i_k))
        out = pl.pallas_call(
            functools.partial(_cumsum_kernel, chunk=chunk, aligned=aligned, scale=scale, reverse=reverse),
            grid=(spans.chunks.shape[0], heads, *tiles),
            in_specs=[chunked_launch._CHUNK, spec], out_specs=spec,
            out_shape=jax.ShapeDtypeStruct((batch * length, heads, *columns), jnp.float32),
            compiler_params=plgpu.CompilerParams(num_warps=warps, num_stages=stages),
            interpret=interpret, name="recurrence_cumulated",
        )(spans.chunks, *chunked_launch._flat(g))
        return out.reshape(g.shape), None


_CHUNK_LOCAL_CUMSUM = ChunkLocalCumsum()


def cumulated(g, *, chunk: int = 64, scale: float | None = None, reverse: bool = False,
              spans: chunked_launch.Sequences | None = None, interpret: bool = False):
    """The chunked recurrences' first stage, their decay's prefix: `g` `[B, T, H]`, or `[B, T, H, K]`
    per key dimension, cumulated within each chunk of `chunk` tokens of each sequence (`spans`, the
    rows where None), fp32: fla's `chunk_local_cumsum`
    (cumsum.py:262-341), `scale` RCP_LN2 for the forward's base-2 decay, `reverse` for its gradient.
    Forward is the prefix summed and scaled in float64 and rounded once (fla's is plain fp32), whose
    difference between two positions every kernel takes as a range's exponent; `reverse` the sums from
    the end, each rounded once. fla autotunes only
    `num_warps` over 1, 2, 4 and 8 here."""
    batch, length = g.shape[:2]
    spans = chunked_launch.sequences(batch, length, chunk=chunk) if spans is None else spans
    return _CHUNK_LOCAL_CUMSUM(g, *spans, chunk=chunk, scale=scale, reverse=reverse, interpret=interpret)
