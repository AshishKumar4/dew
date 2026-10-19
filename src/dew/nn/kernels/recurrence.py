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
also its CPU path and, on float64 operands, the float64 truth.
"""

import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from . import norm_swiglu, short_conv, tuning
from .generation import triton_runs

L2NORM_EPS = 1e-6
"""The queries' and keys' l2 norm's epsilon under the root, the FLA library's (`dew.nn.linear.l2norm`)."""


@dataclasses.dataclass(frozen=True)
class Conv:
    """The prologue's short conv: a token's channels `[C]`, its queries, keys and values head by head
    in turn, over `held` `[K - 1, C]` history with `taps` `[C, K]`, then `activation` ('silu' or None).
    The history is read rounded to the token's dtype, as the layer's conv reads it."""
    activation: str | None = 'silu'


@dataclasses.dataclass(frozen=True)
class Gates:
    """The gates from their pre-activations, a head's: the log decay `-exp(log_a) softplus(gate +
    bias)` (the gated delta rule's and Mamba2's), and beta `sigmoid(beta)`. Without it the step takes
    the log decay and beta as they are."""


@dataclasses.dataclass(frozen=True)
class OutputNorm:
    """The epilogue's gated RMS norm of a head's output: `rms_norm(o) weight act(z)`, the norm in fp32
    with `epsilon`, `activation` the gate's ('silu' or 'sigmoid')."""
    epsilon: float
    activation: str = 'silu'


@dataclasses.dataclass(frozen=True)
class Hooks:
    """A step's static form, which enters its launch's key: the prologue's conv, the queries' and
    keys' l2 norm (and the queries' `Dk^-1/2` scale), the decay's kind ('none', 'scalar' a head,
    'vector' a key dimension), the gates, the delta rule, a skip `D v`, and the epilogue."""
    conv: Conv | None = None
    l2norm: bool = False
    decay: str = 'vector'
    gates: Gates | None = None
    delta: bool = True
    skip: bool = False
    output: OutputNorm | None = None

    def __post_init__(self):
        if self.decay not in ('none', 'scalar', 'vector'):
            raise ValueError(f"a decay is 'none', 'scalar' or 'vector', not {self.decay!r}")
        if self.gates is not None and self.decay != 'scalar':
            raise ValueError('the gates form one log decay a head: their decay is scalar')


def round_to_bf16(state, key, value):
    """`state` `[..., Dk, Dv]` fp32 rounded to bf16 stochastically: up in
    magnitude with the probability of the dropped low 16 bits, so a decay
    smaller than half a bf16 step still moves the state on average, where
    rounding to nearest would hold it. The noise is a hash of the value's
    bits and the token's `key` `[..., Dk]` and `value` `[..., Dv]`, so it is
    deterministic and differs from token to token."""
    bits = jax.lax.bitcast_convert_type(state, jnp.uint32)
    salt = (jax.lax.bitcast_convert_type(key, jnp.uint32)[..., :, None] * jnp.uint32(0x9E3779B1)
            + jax.lax.bitcast_convert_type(value, jnp.uint32)[..., None, :] * jnp.uint32(0x85EBCA77))
    mixed = bits ^ salt
    mixed = (mixed ^ (mixed >> 16)) * jnp.uint32(0x7FEB352D)
    mixed = (mixed ^ (mixed >> 15)) * jnp.uint32(0x846CA68B)
    mixed = mixed ^ (mixed >> 16)
    rounded = (bits + (mixed & jnp.uint32(0xFFFF))) & jnp.uint32(0xFFFF0000)
    return jax.lax.bitcast_convert_type(rounded, jnp.float32).astype(jnp.bfloat16)


def update(state, query, key, value, decay, write, *, delta: bool):
    """One token over a state `[..., Dk, Dv]` in its working dtype, `decay` per key dimension `[...,
    Dk]` (`exp(g)`) and `write` the write gate, broadcast against the values `[..., Dv]`: its output
    `[..., Dv]` and the updated state.
    Both readouts are products with the old state, `(q decay)^T S + (q . k) d` with `d = write (v -
    (k decay)^T S)` under the delta rule and `write v` without it, so one pass over the state serves
    them and the write."""
    read_query = jnp.sum((query * decay)[..., :, None] * state, axis=-2)
    corrected = value - jnp.sum((key * decay)[..., :, None] * state, axis=-2) if delta else value
    written = corrected * write
    out = read_query + jnp.sum(query * key, axis=-1, keepdims=True) * written
    return out, state * decay[..., :, None] + key[..., :, None] * written[..., None, :]


def _held(updated, key, value, dtype):
    """The updated state as it is held: bf16 rounded stochastically (`round_to_bf16`), else cast."""
    return round_to_bf16(updated, key, value) if dtype == jnp.bfloat16 else updated.astype(dtype)


def _gated_decay(gate, log_a, bias):
    """The gates' log decay `-exp(log_a) softplus(gate + bias)`, in the operands' dtype."""
    return -jnp.exp(log_a) * jax.nn.softplus(gate + bias)


def gated_rms(out, z, weight, norm: OutputNorm):
    """The epilogue over a head's output `[..., Dv]` in fp32 or wider: `rms_norm(o) weight act(z)`,
    the mean a product by the reciprocal, as XLA's."""
    inverse = jax.lax.rsqrt(jnp.sum(out * out, axis=-1, keepdims=True) * (1.0 / out.shape[-1]) + norm.epsilon)
    return out * inverse * weight * norm_swiglu.activation_and_slope(z, norm.activation)[0]


@dataclasses.dataclass(frozen=True)
class Step:
    """A decode token's operands, each `[rows, ...]`; a hook's are None where it is off.

    Without a conv, `query` and `key` `[rows, Hk, Dk]` and `value` `[rows, Hv, Dv]` (the query scaled
    unless the hooks' l2 norm scales it); with one, `token` `[rows, C]`, `held` `[rows, K - 1, C]`
    and `taps` `[C, K]`. `log_decay` is `[rows, Hv]` (scalar) or `[rows, Hv, Dk]` (vector), or with
    the gates `gate` `[rows, Hv]`, `log_a` and `bias` `[Hv]`. `beta` `[rows, Hv]` is the write gate
    (a raw beta with the gates), `skip` `[Hv]` the skip's `D`, `z` `[rows, Hv, Dv]` and `weight`
    `[Dv]` the epilogue's. `state` is `[rows, Hv, Dk, Dv]`, fp32 or bf16 (rounded by
    `round_to_bf16`), or wider for the reference. `active` `[rows]` marks the rows that draw."""
    state: jax.Array
    query: jax.Array | None = None
    key: jax.Array | None = None
    value: jax.Array | None = None
    token: jax.Array | None = None
    held: jax.Array | None = None
    taps: jax.Array | None = None
    log_decay: jax.Array | None = None
    gate: jax.Array | None = None
    log_a: jax.Array | None = None
    bias: jax.Array | None = None
    beta: jax.Array | None = None
    skip: jax.Array | None = None
    z: jax.Array | None = None
    weight: jax.Array | None = None
    active: jax.Array | None = None


def _pieces(hooks: Hooks, value_heads: int, key_width: int, value_width: int, channels: int):
    """The key heads a conv token holds, from its channels: queries and keys `Hk Dk` each, values
    `Hv Dv`."""
    key_heads = (channels - value_heads * value_width) // (2 * key_width)
    if key_heads <= 0 or value_heads % key_heads or 2 * key_heads * key_width + value_heads * value_width != channels:
        raise ValueError(f'{channels} conv channels are not queries and keys of {key_width} and values of '
                         f'{value_width} over {value_heads} value heads')
    return key_heads


def _prologue(hooks: Hooks, operands: Step, work):
    """The reference's queries `[rows, Hv, Dk]`, keys, values `[rows, Hv, Dv]`, each in `work`, and
    the moved history (None without a conv): the conv's sums in `work` over the history rounded to
    the token's dtype, the activation, the l2 norm, the key heads repeated over their value heads."""
    state = operands.state
    value_heads, key_width, value_width = state.shape[1:]
    history = None
    if hooks.conv is not None:
        token, held = operands.token, operands.held
        key_heads = _pieces(hooks, value_heads, key_width, value_width, token.shape[-1])
        window = jnp.concatenate([held.astype(token.dtype), token[:, None]], axis=1).astype(work)
        pre = jnp.sum(window * jnp.swapaxes(operands.taps.astype(work), 0, 1)[None], axis=1)
        convolved = pre if hooks.conv.activation is None else pre * jax.nn.sigmoid(pre)
        rows = token.shape[0]
        split = [key_heads * key_width, 2 * key_heads * key_width]
        query, key, value = jnp.split(convolved, split, axis=-1)
        query, key = (x.reshape(rows, key_heads, key_width) for x in (query, key))
        value = value.reshape(rows, value_heads, value_width)
        history = jnp.concatenate([held[:, 1:], token[:, None].astype(held.dtype)], axis=1)
    else:
        query, key, value = (operands.query.astype(work), operands.key.astype(work), operands.value.astype(work))
    if hooks.l2norm:
        query, key = (x * jax.lax.rsqrt(jnp.sum(x * x, axis=-1, keepdims=True) + L2NORM_EPS) for x in (query, key))
        query = query * key_width ** -0.5
    group = value_heads // query.shape[1]
    query, key = (jnp.repeat(x, group, axis=1) for x in (query, key))
    return query, key, value, history


def _gates(hooks: Hooks, operands: Step, work, key_width: int):
    """The reference's decay per key dimension `[rows, Hv, Dk]` and write gate `[rows, Hv]`."""
    if hooks.gates is not None:
        g = _gated_decay(operands.gate.astype(work), operands.log_a.astype(work), operands.bias.astype(work))
        beta = jax.nn.sigmoid(operands.beta.astype(work))
    else:
        g = None if hooks.decay == 'none' else operands.log_decay.astype(work)
        beta = None if operands.beta is None else operands.beta.astype(work)
    rows, heads = operands.state.shape[:2]
    if g is None:
        decay = jnp.ones((rows, heads, key_width), work)
    else:
        decay = jnp.exp(g)
        decay = jnp.broadcast_to(decay[..., None] if hooks.decay == 'scalar' else decay, (rows, heads, key_width))
    write = jnp.ones((rows, heads), work) if beta is None else beta
    return decay, write


def step_reference(hooks: Hooks, operands: Step, *, dtype=None):
    """`(output [rows, Hv, Dv], history' or None, state')` of a decode token, in XLA: each piece of
    arithmetic in at least fp32 (float64 operands in float64), the state held as it came (bf16
    rounded by `round_to_bf16`), the output in `dtype` (the value's or the token's without it)."""
    state = operands.state
    operand = (operands.token if hooks.conv is not None else operands.value).dtype
    work = jnp.promote_types(jnp.promote_types(operand, state.dtype), jnp.float32)
    value_heads, key_width, value_width = state.shape[1:]
    query, key, value, history = _prologue(hooks, operands, work)
    decay, write = _gates(hooks, operands, work, key_width)
    out, updated = update(state.astype(work), query, key, value, decay, write[..., None], delta=hooks.delta)
    if hooks.skip:
        out = out + operands.skip.astype(work)[None, :, None] * value
    if hooks.output is not None:
        out = gated_rms(out, operands.z.astype(work), operands.weight.astype(work), hooks.output)
    held = _held(updated, key, value, state.dtype)
    if operands.active is not None:
        active = jnp.asarray(operands.active, bool)
        held = jnp.where(active[:, None, None, None], held, state)
        out = jnp.where(active[:, None, None], out, 0.0)
        if history is not None:
            history = jnp.where(active[:, None, None], history, operands.held)
    return out.astype(operand if dtype is None else dtype), history, held


def _names(hooks: Hooks) -> tuple[str, ...]:
    """The `Step` operands a launch reads, in its order, after `active`; `state` last."""
    streams = ('token', 'held', 'taps') if hooks.conv is not None else ('query', 'key', 'value')
    gates = (('gate', 'beta', 'log_a', 'bias') if hooks.gates is not None
             else ('log_decay',) * (hooks.decay != 'none') + ('beta',) * hooks.delta)
    return (*streams, *gates, *('skip',) * hooks.skip, *('z', 'weight') * (hooks.output is not None), 'state')


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
    held_out = outputs[1] if hooks.conv is not None else None
    state_ref = inputs['state']
    head = pl.program_id(1)
    active = inputs['active'][0] != 0
    key_width, value_width = state_ref.shape
    key_head = head // group

    def scalar(name):
        """A head's one entry of a `[1]` block, as a one-entry vector the arithmetic broadcasts."""
        return inputs[name][...].astype(jnp.float32)

    if hooks.conv is not None:
        token_ref, history_ref, taps_ref = inputs['token'], inputs['held'], inputs['taps']
        pieces = ((key_head * key_width, key_width), ((key_heads + key_head) * key_width, key_width),
                  (2 * key_heads * key_width + head * value_width, value_width))
        count = taps_ref.shape[-1]

        def window(first, width):
            """A piece's conv window, `K` rows `[1, D]` fp32: the held history rounded to the token's
            dtype, as the layer's conv reads it, then the token."""
            channels = pl.ds(first, width)
            operand = token_ref.dtype
            held = [history_ref[j, channels][None, :].astype(operand).astype(jnp.float32) for j in range(count - 1)]
            return [*held, token_ref[channels][None, :].astype(jnp.float32)]

        def write(moved):
            """Each piece's moved history, a row at a time: the values, and the queries and keys where
            the program is its key head's first."""
            def store(piece, rows):
                for j, row in enumerate(rows):
                    held_out[j, pl.ds(*piece)] = row[0].astype(held_out.dtype)

            store(pieces[2], moved[2])

            @pl.when(head % group == 0)
            def _():
                store(pieces[0], moved[0])
                store(pieces[1], moved[1])

    @pl.when(active)
    def _():
        if hooks.conv is not None:
            windows = [window(*piece) for piece in pieces]
            streams = []
            for stream, (first, width) in zip(windows, pieces, strict=True):
                taps = [taps_ref[pl.ds(first, width), j].astype(jnp.float32) for j in range(count)]
                pre = short_conv._preactivation(stream, taps, None)
                streams.append(pre if hooks.conv.activation is None
                               else pre * short_conv._sigmoid(pre, interpret=interpret))
            query, key, value = (x[0] for x in streams)
        else:
            query, key, value = (inputs[name][...].astype(jnp.float32) for name in ('query', 'key', 'value'))
        if hooks.l2norm:
            query, key = (x[None] for x in (query, key))
            query = query * short_conv._inverse_norm(query, L2NORM_EPS, interpret=interpret) * key_width ** -0.5
            key = key * short_conv._inverse_norm(key, L2NORM_EPS, interpret=interpret)
            query, key = query[0], key[0]
        if hooks.gates is not None:
            decay = jnp.exp(_gated_decay(scalar('gate'), scalar('log_a'), scalar('bias')))
            write_gate = short_conv._sigmoid(scalar('beta'), interpret=interpret)
        else:
            decay = (jnp.ones((1,), jnp.float32) if hooks.decay == 'none'
                     else jnp.exp(inputs['log_decay'][...].astype(jnp.float32)))
            write_gate = scalar('beta') if hooks.delta else jnp.ones((1,), jnp.float32)
        decay = jnp.broadcast_to(decay, key.shape)
        out, updated = update(state_ref[...].astype(jnp.float32), query, key, value, decay, write_gate,
                              delta=hooks.delta)
        if hooks.skip:
            out = out + scalar('skip') * value
        if hooks.output is not None:
            out = gated_rms(out, inputs['z'][...].astype(jnp.float32), inputs['weight'][...].astype(jnp.float32),
                            hooks.output)
        state_out[...] = _held(updated, key, value, state_out.dtype)
        out_ref[...] = out.astype(out_ref.dtype)
        if hooks.conv is not None:
            write([stream[1:] for stream in windows])

    @pl.when(jnp.logical_not(active))
    def _():
        out_ref[...] = jnp.zeros(out_ref.shape, out_ref.dtype)
        if hooks.conv is not None:
            write([[history_ref[j, pl.ds(*piece)][None, :] for j in range(count - 1)] for piece in pieces])


@dataclasses.dataclass(frozen=True)
class Launch:
    """A decode launch's program: `block` value columns of a head's state (None for all of them, which a
    conv or the epilogue's norm needs, each reading a whole head) and its warps."""
    block: int | None
    warps: int


WARPS = Launch(None, 8)
"""The launch without a measurement over a whole head."""
COLUMNS = Launch(32, 8)
"""The launch without a measurement over value columns: 32 a program, 16 to 64 measured within 3% on
an RTX 4080."""


@dataclasses.dataclass(frozen=True)
class StepLaunch(tuning.KernelOp):
    """A decode token in one launch over rows and value heads (`_kernel`), its `hooks` inlined. Off a GPU
    it interprets."""

    config_cls: ClassVar[type[Launch]] = Launch
    ROWS: ClassVar[tuple[str, ...]] = ('active', 'query', 'key', 'value', 'token', 'held', 'log_decay', 'gate',
                                       'beta', 'z', 'state')

    def _fwd(self, active, state, query=None, key=None, value=None, token=None, held=None, taps=None,
             log_decay=None, gate=None, beta=None, log_a=None, bias=None, skip=None, z=None, weight=None, *,
             hooks: Hooks, dtype: str, interpret: bool, config: Launch, return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        arrays = dict(state=state, query=query, key=key, value=value, token=token, held=held, taps=taps,
                      log_decay=log_decay, gate=gate, beta=beta, log_a=log_a, bias=bias, skip=skip, z=z,
                      weight=weight)
        given = {name: arrays[name] for name in _names(hooks)}
        rows, value_heads, key_width, value_width = state.shape
        key_heads = (_pieces(hooks, value_heads, key_width, value_width, given['token'].shape[-1])
                     if hooks.conv is not None else given['query'].shape[1])

        block = value_width if config.block is None else config.block
        if block != value_width and _whole(hooks):
            raise ValueError("a conv or the epilogue's norm reads a head's whole state")

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
            'skip': functools.partial(one, rows=False),
            'log_decay': keyed_by_value if hooks.decay == 'vector' else functools.partial(one, rows=True),
        }
        in_specs = [pl.BlockSpec((None, 1), lambda r, h, c: (r, 0)), *(specs[name](given[name]) for name in given)]
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
        return WARPS if _whole(ba.kwargs['hooks']) else COLUMNS

    def _get_autotuning_configs(self, ba) -> set[Launch]:
        whole = {Launch(None, warps) for warps in (4, 8, 16)}
        if _whole(ba.kwargs['hooks']):
            return whole
        width = ba.arguments['state'].shape[-1]
        return whole | {Launch(block, warps) for block in (16, 32, 64) for warps in (4, 8) if width % block == 0}

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        """Normal floats and every row active, so a timed launch steps each row's state and history."""
        return [jnp.ones(spec.shape, spec.dtype) if jnp.issubdtype(spec.dtype, jnp.integer) else array
                for spec, array in zip(specs, super().operands(specs, seed), strict=True)]


def _whole(hooks: Hooks) -> bool:
    """Whether a program needs a head's whole state: a conv reads all of a head's values, the epilogue's
    norm all of its output."""
    return hooks.conv is not None or hooks.output is not None


def fits(hooks: Hooks, state) -> bool:
    """Whether the launch takes this `[rows, heads, Dk, Dv]` state: a GPU Dew's Triton kernels run on
    (`triton_runs`), held in fp32 or bf16, its widths powers of two (Triton's blocks)."""
    dk, dv = state.shape[-2:]
    return (triton_runs() and state.dtype in (jnp.float32, jnp.bfloat16)
            and all(width > 0 and width & (width - 1) == 0 for width in (dk, dv)))


def step(hooks: Hooks, operands: Step, *, dtype=None):
    """`(output [rows, Hv, Dv], history' or None, state')` of a decode token in one launch
    (`StepLaunch`), `step_reference`'s arithmetic: the output in `dtype` (the value's, or the
    token's with a conv), the history in its dtype, the state as it is held and updated in place
    where the caller lets it go. A row `active` marks False keeps its history and state, its output
    zeros."""
    given = {name: getattr(operands, name) for name in _names(hooks)}
    rows = operands.state.shape[0]
    flags = (jnp.ones(rows, jnp.int32) if operands.active is None
             else jnp.asarray(operands.active).astype(jnp.int32))[:, None]
    operand = (operands.token if hooks.conv is not None else operands.value).dtype
    if 'beta' in given and given['beta'] is not None:
        given['beta'] = given['beta'].reshape(rows, -1)
    if hooks.decay == 'scalar' and hooks.gates is None:
        given['log_decay'] = given['log_decay'].reshape(rows, -1)
    results = StepLaunch()(flags, hooks=hooks, dtype=jnp.dtype(operand if dtype is None else dtype).name,
                           interpret=not triton_runs(), **given)
    state, out = results[0], results[-1]
    return out, results[1] if hooks.conv is not None else None, state
