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
fused decode step it replaces. Its arithmetic is a sequence of stages (`stages`), each a value it
rounds and holds; the launch composes them, and the XLA reference (`step_reference`), which on
float64 operands is the float64 truth, composes the same stages' reference arithmetic. A stage run
as its own launch is a node of the launch's graph (`step_graph`, `dew.nn.kernels.graph`), which the
float64 gate holds where it rounds.
"""

import dataclasses
import functools
from collections.abc import Callable
from typing import Any, ClassVar

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from . import graph, norm_swiglu, short_conv, tuning
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


def _held(updated, key, value, dtype):
    """The updated state as it is held: bf16 rounded stochastically (`round_to_bf16`), else cast."""
    return round_to_bf16(updated, key, value) if dtype == jnp.bfloat16 else updated.astype(dtype)


def _gated_decay(gate, log_a, bias):
    """The gates' log decay `-exp(log_a) softplus(gate + bias)`, in the operands' dtype."""
    return -jnp.exp(log_a) * jax.nn.softplus(gate + bias)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Step:
    """A decode token's operands, each `[rows, ...]`; a hook's are None where it is off.

    Without a conv, `query` and `key` `[rows, Hk, Dk]` and `value` `[rows, Hv, Dv]` (the query scaled
    unless the hooks' l2 norm scales it); with one, `token` `[rows, C]`, `held` `[rows, K - 1, C]`
    and `taps` `[C, K]`. `log_decay` is `[rows, Hv]` (scalar) or `[rows, Hv, Dk]` (vector), or with
    the gates `gate` `[rows, Hv]`, `log_a` and `bias` `[Hv]`. `beta` `[rows, Hv]` is the write gate
    (a raw beta with the gates), `skip` `[Hv]` the skip's `D`, `z` `[rows, Hv, Dv]` and `weight`
    `[Dv]` the epilogue's. `state` is `[rows, Hv, Dk, Dv]`, fp32 or bf16 (rounded by
    `round_to_bf16`), or wider for the reference. `active` `[rows]` marks the rows that draw. A pytree
    of its arrays, so it crosses `jit` and a transformation as a token's operands do; the `Hooks` are
    its static form."""
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
    whole = 2 * key_heads * key_width + value_heads * value_width
    if key_heads <= 0 or value_heads % key_heads or whole != channels:
        raise ValueError(f'{channels} conv channels are not queries and keys of {key_width} and values of '
                         f'{value_width} over {value_heads} value heads')
    return key_heads


OPERANDS = ('state', 'query', 'key', 'value', 'token', 'held', 'taps', 'log_decay', 'gate', 'beta', 'log_a',
            'bias', 'skip', 'z', 'weight')
"""`StepLaunch`'s array operands after `active`, in its order."""


def _names(hooks: Hooks) -> tuple[str, ...]:
    """The `Step` operands a launch reads, in its order, after `active`; `state` last."""
    streams = ('token', 'held', 'taps') if hooks.conv is not None else ('query', 'key', 'value')
    gates = (('gate', 'beta', 'log_a', 'bias') if hooks.gates is not None
             else ('log_decay',) * (hooks.decay != 'none') + ('beta',) * hooks.delta)
    return (*streams, *gates, *('skip',) * hooks.skip, *('z', 'weight') * (hooks.output is not None), 'state')


def _read(hooks: Hooks, operands: Step) -> dict:
    """The operands `hooks` read (`_names`), by name, each held, else a ValueError naming those it lacks."""
    given = {name: getattr(operands, name) for name in _names(hooks)}
    missing = [name for name, array in given.items() if array is None]
    if missing:
        raise ValueError(f"{hooks} reads {', '.join(missing)}, which the step does not hold")
    return given


STREAMS = ('query', 'key', 'value')
"""The conv's streams in its channels' order."""


@dataclasses.dataclass(frozen=True)
class Dims:
    """A decode token's sizes: its rows, value heads, key heads, key and value widths, and the conv's
    taps (0 without one)."""
    rows: int
    value_heads: int
    key_heads: int
    key_width: int
    value_width: int
    taps: int

    @property
    def group(self) -> int:
        """The value heads a key head serves."""
        return self.value_heads // self.key_heads


def _dims(hooks: Hooks, given: dict) -> Dims:
    """The sizes of the operands `hooks` read (`_read`)."""
    rows, value_heads, key_width, value_width = given['state'].shape
    if hooks.conv is None:
        return Dims(rows, value_heads, given['query'].shape[1], key_width, value_width, 0)
    key_heads = _pieces(hooks, value_heads, key_width, value_width, given['token'].shape[-1])
    return Dims(rows, value_heads, key_heads, key_width, value_width, given['taps'].shape[-1])


@dataclasses.dataclass(frozen=True)
class Program:
    """A launch program's place: its value head `head` (traced), the token's sizes, and whether it
    interprets."""
    head: Any
    dims: Dims
    interpret: bool

    def piece(self, stream: str) -> tuple[Any, int]:
        """A stream's first channel and width among the conv's: the program's key head's queries or
        keys, or its own values."""
        dims = self.dims
        key_head = self.head // dims.group
        if stream == 'query':
            return key_head * dims.key_width, dims.key_width
        if stream == 'key':
            return (dims.key_heads + key_head) * dims.key_width, dims.key_width
        return 2 * dims.key_heads * dims.key_width + self.head * dims.value_width, dims.value_width


@dataclasses.dataclass(frozen=True)
class Stage:
    """One value a decode token rounds and holds (`dew.nn.kernels.graph`): its name, its parents' names,
    its kind, its layout (`_spec`), the dtype it is held in, and its arithmetic on its parents' values,
    the launch's on a program's blocks (`kernel`, which takes the `Program` first) and the XLA
    reference's on whole arrays broadcast as a head's (`reference`, `_adapted`)."""
    name: str
    parents: tuple[str, ...]
    kind: graph.Sum | graph.Pointwise
    layout: str
    kernel: Callable
    reference: Callable
    dtype: Any = jnp.float32


def _input_layouts(hooks: Hooks) -> dict[str, str]:
    """Each operand's layout (`_spec`)."""
    return {'query': 'key_head', 'key': 'key_head', 'value': 'valued', 'token': 'row', 'held': 'row',
            'taps': 'taps', 'log_decay': 'keyed' if hooks.decay == 'vector' else 'scalar', 'gate': 'scalar',
            'beta': 'scalar', 'log_a': 'parameter', 'bias': 'parameter', 'skip': 'parameter', 'z': 'valued',
            'weight': 'whole', 'state': 'state'}


def _window(program: Program, held, token, *, stream: str):
    """A stream's conv window in the launch, `K` rows `[1, D]` fp32: the held history rounded to the
    token's dtype, as the layer's conv reads it, then the token."""
    channels = pl.ds(*program.piece(stream))
    rows = [held[j, channels][None, :].astype(token.dtype).astype(jnp.float32) for j in range(held.shape[0])]
    return [*rows, token[channels][None, :].astype(jnp.float32)]


def _streamed(x, stream: str, dims: Dims):
    """A stream's channels of `x` `[..., C]` as `[..., Hv, D]`: its key heads' queries or keys repeated
    over their value heads, or the values."""
    if stream == 'value':
        return x[..., 2 * dims.key_heads * dims.key_width:].reshape(*x.shape[:-1], dims.value_heads,
                                                                   dims.value_width)
    first = 0 if stream == 'query' else dims.key_heads * dims.key_width
    heads = x[..., first:first + dims.key_heads * dims.key_width]
    return jnp.repeat(heads.reshape(*x.shape[:-1], dims.key_heads, dims.key_width), dims.group, axis=-2)


def _window_reference(held, token, *, stream: str, dims: Dims):
    """A stream's conv window `[rows, Hv, K, D]` in at least fp32, the history rounded to the token's
    dtype."""
    window = jnp.concatenate([held.astype(token.dtype), token[:, None]], axis=1)
    window = window.astype(jnp.promote_types(window.dtype, jnp.float32))
    return jnp.moveaxis(_streamed(window, stream, dims), 1, 2)


def _preactivation(program: Program, window, taps, *, stream: str):
    """A stream's conv sum in the launch, tap by tap from the oldest token (`short_conv`'s)."""
    first, width = program.piece(stream)
    columns = [taps[pl.ds(first, width), j].astype(jnp.float32) for j in range(len(window))]
    return short_conv._preactivation(window, columns, None)[0]


def _preactivation_reference(window, taps, *, stream: str, dims: Dims):
    """A stream's conv sum `[rows, Hv, D]` over its window's taps."""
    taps = jnp.moveaxis(_streamed(jnp.swapaxes(taps, 0, 1), stream, dims), 0, 1)
    return jnp.sum(window * taps[None], axis=-2)


def _squares(x):
    """The sum of `x`'s squares over its last axis, kept."""
    return jnp.sum(x * x, axis=-1, keepdims=True)


def _normed(program: Program, x, squares, *, scale: float | None):
    """The l2 norm in the launch, its root and quotient each rounded once (`short_conv`'s), then the
    queries' scale."""
    root = short_conv._sqrt(squares + L2NORM_EPS, interpret=program.interpret)
    x = x * short_conv._divide(1.0, root, interpret=program.interpret)
    return x if scale is None else x * scale


def _normed_reference(x, squares, *, scale: float | None):
    """The l2 norm, then the queries' scale."""
    x = x * jax.lax.rsqrt(squares + L2NORM_EPS)
    return x if scale is None else x * scale


def _contracted(x, state):
    """`x` `[..., Dk]` through the state `[..., Dk, Dv]`: `S^T x`."""
    return jnp.sum(x[..., :, None] * state, axis=-2)


def _updated(state, *operands):
    """The state decayed (where a decay is among `operands`) plus the key's outer product with what
    the token writes."""
    *decay, key, written = operands
    kept = state * decay[0][..., :, None] if decay else state
    return kept + key[..., :, None] * written[..., None, :]


def _gated_rms(out, squares, z, weight, *, norm: OutputNorm):
    """The epilogue's gated RMS norm from the output's sum of squares: `rms_norm(o) weight act(z)`, the
    mean a product by the reciprocal, as XLA's."""
    inverse = jax.lax.rsqrt(squares * (1.0 / out.shape[-1]) + norm.epsilon)
    return out * inverse * weight * norm_swiglu.activation_and_slope(z, norm.activation)[0]


def stages(hooks: Hooks, dims: Dims, *, output, state) -> tuple[Stage, ...]:
    """The values a decode token rounds and holds, each after its parents: the prologue's conv
    windows, sums and activations, the l2 norms' sums and quotients, the gates, the decayed queries and
    keys, the state's readouts, the write, the output and the updated state, the skip and the
    epilogue's sum and norm, with the output in `output` and the state as `state` holds it. Their names
    are the graph's (`step_graph`); a cut never falls between a product and the sum it enters."""
    built = []

    def add(name, parents, kind, layout, reference, kernel=None, dtype=jnp.float32):
        """A stage, its launch running `kernel` or, where None, `reference`'s arithmetic."""
        launch = kernel or (lambda program, *values: reference(*values))
        built.append(Stage(name, tuple(parents), kind, layout, launch, reference, dtype))
        return name

    pointwise = graph.Pointwise()
    streams = {stream: stream for stream in STREAMS}
    if hooks.conv is not None:
        for stream in STREAMS:
            layout = 'valued' if stream == 'value' else 'keyed'
            window = add(f'{stream}_window', ('held', 'token'), pointwise, f'{layout}_window',
                         functools.partial(_window_reference, stream=stream, dims=dims),
                         functools.partial(_window, stream=stream))
            summed = graph.Sum({window: -2, 'taps': -1})
            streams[stream] = add(f'{stream}_preactivation', (window, 'taps'), summed, layout,
                                  functools.partial(_preactivation_reference, stream=stream, dims=dims),
                                  functools.partial(_preactivation, stream=stream))
            if hooks.conv.activation is not None:
                streams[stream] = add(
                    f'{stream}_convolved', (streams[stream],), pointwise, layout,
                    lambda pre: pre * jax.nn.sigmoid(pre),
                    lambda program, pre: pre * short_conv._sigmoid(pre, interpret=program.interpret))
    if hooks.l2norm:
        for stream in ('query', 'key'):
            squares = add(f'{stream}_squares', (streams[stream],), graph.Sum({streams[stream]: -1}), 'unit',
                          _squares)
            scale = dims.key_width ** -0.5 if stream == 'query' else None
            streams[stream] = add(f'{stream}_normed', (streams[stream], squares), pointwise, 'keyed',
                                  functools.partial(_normed_reference, scale=scale),
                                  functools.partial(_normed, scale=scale))
    query, key, value = (streams[stream] for stream in STREAMS)
    decay = None
    if hooks.gates is not None:
        decay = add('decay', ('gate', 'log_a', 'bias'), pointwise, 'unit',
                    lambda gate, log_a, bias: jnp.exp(_gated_decay(gate, log_a, bias)))
        write = add('write', ('beta',), pointwise, 'unit', jax.nn.sigmoid,
                    lambda program, beta: short_conv._sigmoid(beta, interpret=program.interpret))
    else:
        if hooks.decay != 'none':
            decay = add('decay', ('log_decay',), pointwise, 'keyed' if hooks.decay == 'vector' else 'unit',
                        jnp.exp)
        write = 'beta'
    reads = query if decay is None else add('query_decayed', (query, decay), pointwise, 'keyed', jnp.multiply)
    add('read', (reads, 'state'), graph.Sum({reads: -1, 'state': -2}), 'valued', _contracted)
    written = value
    if hooks.delta:
        predicts = key if decay is None else add('key_decayed', (key, decay), pointwise, 'keyed',
                                                 jnp.multiply)
        add('predicted', (predicts, 'state'), graph.Sum({predicts: -1, 'state': -2}), 'valued', _contracted)
        written = add('written', (value, 'predicted', write), pointwise, 'valued',
                      lambda value, predicted, write: (value - predicted) * write)
    add('alignment', (query, key), graph.Sum({query: -1, key: -1}), 'unit',
        lambda query, key: jnp.sum(query * key, axis=-1, keepdims=True))
    out = add('out', ('read', 'alignment', written), pointwise, 'valued',
              lambda read, alignment, written: read + alignment * written)
    decays = () if decay is None else (decay,)
    add('updated', ('state', *decays, key, written), pointwise, 'state', _updated)
    add('next_state', ('updated', key, value), pointwise, 'state',
        lambda updated, key, value: _held(updated, key, value, state), dtype=state)
    if hooks.skip:
        out = add('skipped', (out, 'skip', value), pointwise, 'valued',
                  lambda out, skip, value: out + skip * value)
    if hooks.output is not None:
        add('out_squares', (out,), graph.Sum({out: -1}), 'unit', _squares)
        out = add('normed', (out, 'out_squares', 'z', 'weight'), pointwise, 'valued',
                  functools.partial(_gated_rms, norm=hooks.output))
    add('output', (out,), pointwise, 'valued', lambda out: out.astype(output), dtype=output)
    return tuple(built)


def _layouts(hooks: Hooks, built: tuple[Stage, ...]) -> dict[str, str]:
    """Every operand's and stage's layout (`_spec`)."""
    return {**_input_layouts(hooks), **{stage.name: stage.layout for stage in built}}


def _adapted(layout: str, x, dims: Dims):
    """A value as the reference's arithmetic broadcasts a head's: in at least fp32, key heads repeated
    over their value heads, a head's scalar `[rows, Hv, 1]` and a parameter `[1, Hv, 1]`; the conv's
    history and token as they are, which its window rounds itself."""
    x = jnp.asarray(x)
    if layout == 'row':
        return x
    x = x.astype(jnp.promote_types(x.dtype, jnp.float32))
    if layout == 'key_head':
        return jnp.repeat(x, dims.group, axis=1)
    if layout == 'scalar':
        return x[..., None]
    if layout == 'parameter':
        return x[None, :, None]
    return x


def _referenced(built: tuple[Stage, ...], layouts: dict[str, str], dims: Dims, given: dict) -> dict:
    """Every stage's reference value from the operands `given`, one after another."""
    values = dict(given)
    for stage in built:
        values[stage.name] = stage.reference(*(_adapted(layouts[parent], values[parent], dims)
                                               for parent in stage.parents))
    return values


def step_reference(hooks: Hooks, operands: Step, *, dtype=None):
    """`(output [rows, Hv, Dv], history' or None, state')` of a decode token, in XLA: every stage's
    reference arithmetic (`stages`) in at least fp32 (float64 operands in float64), the state held as it
    came (bf16 rounded by `round_to_bf16`), the output in `dtype` (the value's or the token's without
    it)."""
    given = _read(hooks, operands)
    dims, state = _dims(hooks, given), given['state']
    operand = given['token' if hooks.conv is not None else 'value'].dtype
    built = stages(hooks, dims, output=jnp.dtype(operand if dtype is None else dtype), state=state.dtype)
    values = _referenced(built, _layouts(hooks, built), dims, given)
    out, held, history = values['output'], values['next_state'], None
    if hooks.conv is not None:
        history = jnp.concatenate([given['held'][:, 1:], given['token'][:, None].astype(given['held'].dtype)],
                                  axis=1)
    if operands.active is not None:
        active = jnp.asarray(operands.active, bool)
        held = jnp.where(active[:, None, None, None], held, state)
        out = jnp.where(active[:, None, None], out, jnp.zeros((), out.dtype))
        if history is not None:
            history = jnp.where(active[:, None, None], history, given['held'])
    return out, history, held


def _spec(layout: str, shape: tuple[int, ...], dims: Dims, block: int):
    """A layout's block over the grid of rows, value heads and `block` value columns: a row's whole
    block (`row`), a whole array (`taps`, `whole`), the program's key head's `[Dk]` (`key_head`), its
    value head's `[Dk]` (`keyed`), `[block]` (`valued`) or `[Dk, block]` (`state`), its head's one entry
    (`scalar` `[rows, Hv]`, `unit` `[rows, Hv, 1]`, `parameter` `[Hv]`), or a window's `[K, Dk]` or `[K,
    block]` (`keyed_window`, `valued_window`)."""
    if layout == 'row':
        return pl.BlockSpec((None, *shape[1:]), lambda r, h, c: (r, *(0,) * (len(shape) - 1)))
    if layout in ('taps', 'whole'):
        return pl.BlockSpec(shape, lambda r, h, c: (0,) * len(shape))
    blocks = {
        'key_head': ((None, None, dims.key_width), lambda r, h, c: (r, h // dims.group, 0)),
        'keyed': ((None, None, dims.key_width), lambda r, h, c: (r, h, 0)),
        'valued': ((None, None, block), lambda r, h, c: (r, h, c)),
        'state': ((None, None, dims.key_width, block), lambda r, h, c: (r, h, 0, c)),
        'scalar': ((None, 1), lambda r, h, c: (r, h)),
        'unit': ((None, None, 1), lambda r, h, c: (r, h, 0)),
        'parameter': ((1,), lambda r, h, c: (h,)),
        'keyed_window': ((None, None, dims.taps, dims.key_width), lambda r, h, c: (r, h, 0, 0)),
        'valued_window': ((None, None, dims.taps, block), lambda r, h, c: (r, h, 0, c)),
    }
    return pl.BlockSpec(*blocks[layout])


def _shape(layout: str, dims: Dims) -> tuple[int, ...]:
    """A stage's whole value's shape by its layout."""
    rows, heads = dims.rows, dims.value_heads
    return {'keyed': (rows, heads, dims.key_width), 'valued': (rows, heads, dims.value_width),
            'state': (rows, heads, dims.key_width, dims.value_width), 'unit': (rows, heads, 1),
            'keyed_window': (rows, heads, dims.taps, dims.key_width),
            'valued_window': (rows, heads, dims.taps, dims.value_width)}[layout]


def _loaded(layout: str, ref):
    """A program's block as its arithmetic takes it: a ref the conv slices itself (`row`, `taps`), a
    window's rows `[1, D]`, else the block in fp32."""
    if layout in ('row', 'taps'):
        return ref
    if layout.endswith('_window'):
        return [ref[pl.ds(j, 1), :].astype(jnp.float32) for j in range(ref.shape[0])]
    return ref[...].astype(jnp.float32)


def _stored(layout: str, ref, value) -> None:
    """A program's value written to its block, a window a row at a time."""
    if layout.endswith('_window'):
        for j, row in enumerate(value):
            ref[pl.ds(j, 1), :] = row.astype(ref.dtype)
    else:
        ref[...] = value.astype(ref.dtype)


def _moved(program: Program, moved, windows: dict) -> None:
    """Each stream's moved history, a row at a time: the values, and the queries and keys where the
    program is its key head's first, so no program reads a history another moved on."""
    def store(stream):
        channels = pl.ds(*program.piece(stream))
        for j, row in enumerate(windows[stream]):
            moved[j, channels] = row[0].astype(moved.dtype)

    store('value')

    @pl.when(program.head % program.dims.group == 0)
    def _():
        store('query')
        store('key')


def _kernel(*refs, hooks: Hooks, dims: Dims, built: tuple[Stage, ...], layouts: dict[str, str],
            probes: tuple[str, ...], interpret: bool):
    """One row and one value head of a decode token (`step`): every stage (`stages`) in turn on the
    program's blocks, the conv's channels read where the row holds them, `[K - 1, C]` history and `[C]`
    token, queries, keys and values in turn; the history the next token reads is written apart, its key
    head's queries and keys by the head's first value head; an idle row's as it was. The stages
    `probes` names are written out too, each to its own output after the token's."""
    names = _names(hooks)
    inputs = dict(zip(('active', *names), refs[:len(names) + 1], strict=True))
    state_out, *outputs = refs[len(names) + 1:]
    moved = outputs.pop(0) if hooks.conv is not None else None
    out_ref, *written = outputs
    probed = dict(zip(probes, written, strict=True))
    program = Program(pl.program_id(1), dims, interpret)
    active = inputs['active'][0] != 0

    @pl.when(active)
    def _():
        values = {name: _loaded(layouts[name], inputs[name]) for name in names}
        for stage in built:
            values[stage.name] = stage.kernel(program, *(values[parent] for parent in stage.parents))
            if stage.name in probed:
                _stored(stage.layout, probed[stage.name], values[stage.name])
        state_out[...] = values['next_state']
        out_ref[...] = values['output']
        if moved is not None:
            _moved(program, moved, {stream: values[f'{stream}_window'][1:] for stream in STREAMS})

    @pl.when(jnp.logical_not(active))
    def _():
        out_ref[...] = jnp.zeros(out_ref.shape, out_ref.dtype)
        if moved is not None:
            history = inputs['held']
            _moved(program, moved, {stream: [history[j, pl.ds(*program.piece(stream))][None, :]
                                             for j in range(dims.taps - 1)] for stream in STREAMS})


@dataclasses.dataclass(frozen=True)
class Launch:
    """A decode launch's program: `block` value columns of a head's state (0 for all of them, which a
    conv or the epilogue's norm needs, each reading a whole head) and its warps."""
    block: int
    warps: int


WARPS = Launch(0, 8)
"""The launch without a measurement over a whole head."""
COLUMNS = Launch(32, 8)
"""The launch without a measurement over value columns: 32 a program, 16 to 64 measured within 3% on
an RTX 4080."""


@dataclasses.dataclass(frozen=True)
class StepLaunch(tuning.KernelOp):
    """A decode token in one launch over rows and value heads (`_kernel`), its `hooks` inlined. Off a GPU
    it interprets."""

    config_cls: ClassVar[type[Launch]] = Launch
    ROWS: ClassVar[tuple[str, ...]] = ('active', 'query', 'key', 'value', 'token', 'held', 'log_decay',
                                       'gate', 'beta', 'z', 'state')

    def _fwd(self, active: jax.Array, state: jax.Array, query: jax.Array | None, key: jax.Array | None,
             value: jax.Array | None, token: jax.Array | None, held: jax.Array | None, taps: jax.Array | None,
             log_decay: jax.Array | None, gate: jax.Array | None, beta: jax.Array | None,
             log_a: jax.Array | None, bias: jax.Array | None, skip: jax.Array | None, z: jax.Array | None,
             weight: jax.Array | None, *, hooks: Hooks, dtype: str, interpret: bool, config: Launch,
             probes: tuple[str, ...] = (), return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        arrays = {'state': state, 'query': query, 'key': key, 'value': value, 'token': token, 'held': held,
                  'taps': taps, 'log_decay': log_decay, 'gate': gate, 'beta': beta, 'log_a': log_a,
                  'bias': bias, 'skip': skip, 'z': z, 'weight': weight}
        given = {name: arrays[name] for name in _names(hooks)}
        dims = _dims(hooks, given)
        built = stages(hooks, dims, output=jnp.dtype(dtype), state=state.dtype)
        layouts = _layouts(hooks, built)
        # A hook that reads a whole head has only whole-head configs in its space (`_whole`).
        block = config.block or dims.value_width
        flags = pl.BlockSpec((None, 1), lambda r, h, c: (r, 0))
        in_specs = [flags, *(_spec(layouts[name], array.shape, dims, block) for name, array in given.items())]
        out_specs = [_spec('state', state.shape, dims, block)]
        out_shape = [jax.ShapeDtypeStruct(state.shape, state.dtype)]
        if hooks.conv is not None:
            out_specs.append(_spec('row', given['held'].shape, dims, block))
            out_shape.append(jax.ShapeDtypeStruct(given['held'].shape, given['held'].dtype))
        probed = {stage.name: stage for stage in built if stage.name in probes}
        for stage in (built[-1], *(probed[name] for name in probes)):
            shape = _shape(stage.layout, dims)
            out_specs.append(_spec(stage.layout, shape, dims, block))
            out_shape.append(jax.ShapeDtypeStruct(shape, stage.dtype))
        kernel = functools.partial(_kernel, hooks=hooks, dims=dims, built=built, layouts=layouts,
                                   probes=probes, interpret=interpret)
        return tuple(pl.pallas_call(
            kernel, grid=(dims.rows, dims.value_heads, dims.value_width // block), in_specs=in_specs,
            out_specs=out_specs, out_shape=out_shape, input_output_aliases={len(given): 0},
            compiler_params=plgpu.CompilerParams(num_warps=config.warps),
            interpret=interpret, name="recurrence_step",
        )(active, *given.values())), None

    def _get_heuristics_config(self, ba) -> Launch:
        if _whole(ba.kwargs['hooks']):
            return WARPS
        return dataclasses.replace(COLUMNS, block=min(COLUMNS.block, ba.arguments['state'].shape[-1]))

    def _get_autotuning_configs(self, ba) -> set[Launch]:
        whole = {Launch(0, warps) for warps in (4, 8, 16)}
        if _whole(ba.kwargs['hooks']):
            return whole
        width = ba.arguments['state'].shape[-1]
        return whole | {Launch(block, warps) for block in (16, 32, 64) for warps in (4, 8)
                        if width % block == 0}

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


def _launched(hooks: Hooks, operands: Step, dtype, *, config: Launch | None = None,
              probes: tuple[str, ...] = ()):
    """`StepLaunch`'s outputs for `operands` (`step`), at `config` where given (else its measured or
    heuristic one), with the stages `probes` names written out after them."""
    given = _read(hooks, operands)
    rows = given['state'].shape[0]
    flags = (jnp.ones(rows, jnp.int32) if operands.active is None
             else jnp.asarray(operands.active).astype(jnp.int32))[:, None]
    operand = given['token' if hooks.conv is not None else 'value'].dtype
    if 'beta' in given:
        given['beta'] = given['beta'].reshape(rows, -1)
    if hooks.decay == 'scalar' and hooks.gates is None:
        given['log_decay'] = given['log_decay'].reshape(rows, -1)
    named = jnp.dtype(operand if dtype is None else dtype).name
    # Positional, None where a hook is off: the tuner keys a launch by its positional arrays.
    return StepLaunch(config=config)(flags, *(given.get(name) for name in OPERANDS), hooks=hooks, dtype=named,
                                     interpret=not triton_runs(), probes=probes)


def step(hooks: Hooks, operands: Step, *, dtype=None):
    """`(output [rows, Hv, Dv], history' or None, state')` of a decode token in one launch
    (`StepLaunch`), its stages' arithmetic (`stages`): the output in `dtype` (the value's, or the
    token's with a conv), the history in its dtype, the state as it is held and updated in place
    where the caller lets it go. A row `active` marks False keeps its history and state, its output
    zeros."""
    state, *history, out = _launched(hooks, operands, dtype)
    return out, history[0] if history else None, state


def step_graph(hooks: Hooks, operands: Step, *, dtype=None, config: Launch | None = None) -> graph.Graph:
    """The launch as a graph of the values it rounds and holds (`dew.nn.kernels.graph`), over the arrays
    of `operands` `hooks` read, by name, every row active: each node a stage (`stages`) run as its own
    launch on the parents it is given, at `config` (the launch's heuristic where None), and the witness
    the launch itself, every stage's value written out beside its output and state. Its outputs are
    'output' and 'next_state'."""
    given = _read(hooks, operands)
    dims = _dims(hooks, given)
    operand = given['token' if hooks.conv is not None else 'value'].dtype
    built = stages(hooks, dims, output=jnp.dtype(operand if dtype is None else dtype),
                   state=given['state'].dtype)
    layouts = _layouts(hooks, built)
    if config is None:
        config = WARPS if _whole(hooks) else dataclasses.replace(COLUMNS, block=min(COLUMNS.block,
                                                                                    dims.value_width))
    interpret = not triton_runs()

    def launched(stage: Stage):
        """`stage` as its own launch over the token's programs."""
        from jax.experimental.pallas import triton as plgpu

        block = config.block or dims.value_width
        shape = _shape(stage.layout, dims)

        def body(*refs):
            program = Program(pl.program_id(1), dims, interpret)
            values = [_loaded(layouts[parent], ref)
                      for parent, ref in zip(stage.parents, refs[:-1], strict=True)]
            _stored(stage.layout, refs[-1], stage.kernel(program, *values))

        def run(**parents):
            arrays = [jnp.asarray(parents[parent]) for parent in stage.parents]
            return pl.pallas_call(
                body, grid=(dims.rows, dims.value_heads, dims.value_width // block),
                in_specs=[_spec(layouts[parent], array.shape, dims, block)
                          for parent, array in zip(stage.parents, arrays, strict=True)],
                out_specs=_spec(stage.layout, shape, dims, block),
                out_shape=jax.ShapeDtypeStruct(shape, stage.dtype),
                compiler_params=plgpu.CompilerParams(num_warps=config.warps), interpret=interpret,
                name=f"recurrence_{stage.name}")(*arrays)
        return jax.jit(run)

    probes = tuple(stage.name for stage in built[:-1] if stage.name != 'next_state')

    @jax.jit
    def fused(**inputs):
        launched = _launched(hooks, Step(**inputs), dtype, config=config, probes=probes)
        written = len(launched) - len(probes)
        probed = dict(zip(probes, launched[written:], strict=True))
        return {'output': launched[written - 1], 'next_state': launched[0], **probed}

    nodes = tuple(graph.Node(stage.name, stage.parents, stage.kind, launched(stage)) for stage in built)
    return graph.Graph(tuple(given), nodes, ('output', 'next_state'), fused)


def reference_graph(hooks: Hooks, operands: Step, *, dtype=None, float64: bool = False) -> graph.Graph:
    """`step_graph`'s graph of the XLA reference (`step_reference`): each node a stage's reference
    arithmetic, run op by op as the witness runs them all, and with `float64` the float64 truth, its
    output and state held in float64."""
    given = _read(hooks, operands)
    dims = _dims(hooks, given)
    operand = given['token' if hooks.conv is not None else 'value'].dtype
    held = jnp.float64 if float64 else given['state'].dtype
    output = jnp.float64 if float64 else operand if dtype is None else dtype
    built = stages(hooks, dims, output=jnp.dtype(output), state=held)
    layouts = _layouts(hooks, built)

    def node(stage: Stage):
        def run(**parents):
            return stage.reference(*(_adapted(layouts[parent], parents[parent], dims)
                                     for parent in stage.parents))
        return graph.Node(stage.name, stage.parents, stage.kind, run)

    def fused(**inputs):
        values = _referenced(built, layouts, dims, inputs)
        return {stage.name: values[stage.name] for stage in built}

    return graph.Graph(tuple(given), tuple(node(stage) for stage in built), ('output', 'next_state'), fused)
