"""One decode token of the delta rule, reading the recurrent state once and
writing it once, as a Pallas kernel for CUDA.

The token's update is `S' = S diag(decay) + k delta^T` with
`delta = beta (v - (k decay)^T S)`, and its output `q^T S'`, which is
`(q decay)^T S + (q . k) delta`: both readouts are products with the old
state, so one pass over it serves them and the write. XLA ran the
reference's four passes (decay, read, write, read) as three fusions over
the fp32 state; at Qwen3.5-0.8B's widths and 128 rows on an RTX 4080 they
were two thirds of a decode step (docs/performance.md).

The program is one row-head and one block of value columns, over a state
laid out `[rows * heads, Dk, Dv]`; its body is whole-block array arithmetic
with nothing Triton-specific but the warp count, so a Mosaic GPU kernel (JAX
0.11 deprecates the Triton backend, and Mosaic GPU targets sm90 on) takes
the same body and block specs. Off a GPU it interprets, as the CPU suite
runs it. On a TPU the step stays XLA's: tokamax's
Mosaic TPU `causal_conv1d_gated_delta_rule` covers the conv, the gating and
the rule over a step's tokens in one ragged call, which is the layout of a
serving step's mixed call (`dew.nn.inputs.Admitted`) rather than this one.
"""

import dataclasses
import functools
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from . import norm_swiglu, short_conv, tuning
from .generation import triton_runs


@dataclasses.dataclass(frozen=True)
class Columns:
    """The value columns a program updates, and its warps."""
    block: int
    warps: int


COLUMNS = Columns(block=32, warps=8)
"""The launch without a measurement: 32 value columns a program, 16 to 64 measured within 3% on an RTX
4080."""

SPACE = frozenset(Columns(block, warps) for block in (16, 32, 64) for warps in (4, 8))
"""The configs the tuner measures, each a block that divides the state's columns: each holds the
recurrence's gate (tests/test_linear_attention.py)."""


def fits(state: jax.Array) -> bool:
    """Whether the kernel takes this `[rows, heads, Dk, Dv]` state: a GPU Dew's
    Triton kernels run on (`triton_runs`), held in fp32 or bf16, and
    power-of-two widths with Dv a multiple of `COLUMNS`' block (Triton's blocks)."""
    dk, dv = state.shape[-2:]
    power = dk > 0 and dk & (dk - 1) == 0
    return (triton_runs() and state.dtype in (jnp.float32, jnp.bfloat16) and power
            and dv % COLUMNS.block == 0 and (dv // COLUMNS.block) & (dv // COLUMNS.block - 1) == 0)


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


def _update(state, query, key, value, decay, beta):
    """One token's delta rule over a state `[Dk, Dv]` fp32, `decay` per key dimension `[Dk]`: its
    output `[Dv]` and the updated state, each sum in fp32."""
    read_key = jnp.sum((key * decay)[:, None] * state, axis=0)          # [Dv]
    read_query = jnp.sum((query * decay)[:, None] * state, axis=0)
    delta = (value - read_key) * beta
    out = read_query + jnp.sum(query * key) * delta
    return out, state * decay[:, None] + key[:, None] * delta[None, :]


def _held(updated, key, value, dtype):
    """The updated state as it is held: bf16 rounded stochastically (`round_to_bf16`), else cast."""
    return round_to_bf16(updated, key, value) if dtype == jnp.bfloat16 else updated.astype(dtype)


def _kernel(active_ref, state_ref, query_ref, key_ref, value_ref, decay_ref, beta_ref, state_out, out_ref):
    active = active_ref[0] != 0

    @pl.when(active)
    def _():
        key = key_ref[...]
        out_ref[...], updated = _update(state_ref[...].astype(jnp.float32), query_ref[...], key,
                                        value_ref[...], decay_ref[...], beta_ref[...])
        state_out[...] = _held(updated, key, value_ref[...], state_out.dtype)

    # An idle row's state is neither read nor written: the output aliases it.
    @pl.when(jnp.logical_not(active))
    def _():
        out_ref[...] = jnp.zeros(out_ref.shape, out_ref.dtype)


def step(state, query, key, value, decay, beta, active=None):
    """`(state', out)` for one token: `state` `[rows, heads, Dk, Dv]`, fp32
    or bf16 and read and written as it is held (a bf16 one rounded by
    `round_to_bf16`), `query` (already scaled),
    `key` and `decay` (`exp(g)`, per key dimension) `[rows, heads, Dk]`,
    `value` `[rows, heads, Dv]`, `beta` `[rows, heads]`, all fp32, which the
    rule runs in. The state is updated in place where the caller lets it go.

    A row `active` `[rows]` marks False draws nothing: its state stays as
    it was without being read, and its output is zeros. A serving step
    holds every slot, drawing or not, and the state is most of a decode
    step's bytes."""
    rows, heads, dk, dv = state.shape
    flat = rows * heads
    updated, out = DeltaStep()(
        jnp.repeat(jnp.ones(rows, jnp.int32) if active is None else active.astype(jnp.int32), heads)[:, None],
        state.reshape(flat, dk, dv), query.reshape(flat, dk), key.reshape(flat, dk),
        value.reshape(flat, dv), decay.reshape(flat, dk), jnp.broadcast_to(beta.reshape(flat, 1), (flat, dv)),
        interpret=not triton_runs())
    return updated.reshape(state.shape), out.reshape(rows, heads, dv)


@dataclasses.dataclass(frozen=True)
class DeltaStep(tuning.KernelOp):
    """The token's launch over row-heads (`_kernel`): a program a row-head and a block of its value
    columns, `Columns`, `COLUMNS` without a measurement. Off a GPU it interprets."""

    config_cls: ClassVar[type[Columns]] = Columns
    ROWS: ClassVar[tuple[str, ...]] = ('active', 'state', 'query', 'key', 'value', 'decay', 'beta')

    def _fwd(self, active: jax.Array, state: jax.Array, query: jax.Array, key: jax.Array, value: jax.Array,
             decay: jax.Array, beta: jax.Array, *, interpret: bool, config: Columns,
             return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        flat, dk, dv = state.shape
        flagged = pl.BlockSpec((None, 1), lambda i, j: (i, 0))
        keyed = pl.BlockSpec((None, dk), lambda i, j: (i, 0))
        valued = pl.BlockSpec((None, config.block), lambda i, j: (i, j))
        held = pl.BlockSpec((None, dk, config.block), lambda i, j: (i, 0, j))
        return tuple(pl.pallas_call(
            _kernel, grid=(flat, dv // config.block),
            in_specs=[flagged, held, keyed, keyed, valued, keyed, valued],
            out_specs=[held, valued],
            out_shape=[jax.ShapeDtypeStruct((flat, dk, dv), state.dtype),
                       jax.ShapeDtypeStruct((flat, dv), query.dtype)],
            input_output_aliases={1: 0},
            compiler_params=plgpu.CompilerParams(num_warps=config.warps),
            interpret=interpret, name="delta_rule_step",
        )(active, state, query, key, value, decay, beta)), None

    def _get_heuristics_config(self, ba) -> Columns:
        return COLUMNS

    def _get_autotuning_configs(self, ba) -> set[Columns]:
        return {config for config in SPACE if ba.arguments['state'].shape[-1] % config.block == 0}

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        """Normal floats and every row active, so a timed launch steps each row-head's state."""
        return [jnp.ones(spec.shape, spec.dtype) if jnp.issubdtype(spec.dtype, jnp.integer) else array
                for spec, array in zip(specs, super().operands(specs, seed), strict=True)]



L2NORM_EPS = 1e-6
"""The queries' and keys' l2 norm's epsilon under the root, the FLA library's (`dew.nn.linear.l2norm`)."""


def _fused_kernel(active_ref, token_ref, held_ref, taps_ref, gate_ref, beta_ref, log_decay_ref, bias_ref,
                  z_ref, weight_ref, state_ref, state_out, held_out, out_ref, *, group: int, key_heads: int,
                  epsilon: float, activation: str, interpret: bool):
    """One row and one value head of a gated delta net's decode token (`fused_step`): the conv of its
    key head's queries and keys and of its own values, the gates, the update and the gated norm, each
    piece of arithmetic its kernel's own (`short_conv`'s conv, sigmoid and l2 norm, `_update`,
    `norm_swiglu`'s gated norm). The conv's channels are read where the row holds them, `[K - 1, C]`
    history and `[C]` token, queries, keys and values in turn; the history the next token reads is
    written apart, its key head's queries and keys by the head's first value head, so no program
    reads a history another moved on; an idle row's as it was."""
    head = pl.program_id(1)
    active = active_ref[0] != 0
    key_width, value_width = state_ref.shape
    key_head = head // group
    pieces = ((key_head * key_width, key_width), ((key_heads + key_head) * key_width, key_width),
              (2 * key_heads * key_width + head * value_width, value_width))
    count = taps_ref.shape[-1]

    def window(first, width):
        """A piece's conv window `[K, 1, D]` fp32: the held history as the layer's conv reads it,
        rounded to the token's dtype (`held_scan`), then the token."""
        channels = pl.ds(first, width)
        operand = token_ref.dtype
        held = [held_ref[j, channels][None, :].astype(operand).astype(jnp.float32) for j in range(count - 1)]
        return [*held, token_ref[channels][None, :].astype(jnp.float32)]

    def write(moved):
        """Each piece's moved history, a row of it at a time: the values, and the queries and keys
        where the program is its key head's first."""
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
        windows = [window(*piece) for piece in pieces]
        convolved = []
        for stream, (first, width) in zip(windows, pieces, strict=True):
            taps = [taps_ref[pl.ds(first, width), j].astype(jnp.float32) for j in range(count)]
            pre = short_conv._preactivation(stream, taps, None)
            convolved.append(pre * short_conv._sigmoid(pre, interpret=interpret))
        query, key, value = convolved
        query = query * short_conv._inverse_norm(query, L2NORM_EPS, interpret=interpret) * key_width ** -0.5
        key = key * short_conv._inverse_norm(key, L2NORM_EPS, interpret=interpret)
        # The head's gates as one-entry vectors, which the rounded sigmoid's inline assembly takes.
        gate = gate_ref[...].astype(jnp.float32) + bias_ref[...].astype(jnp.float32)
        decay = jnp.exp(-jnp.exp(log_decay_ref[...].astype(jnp.float32)) * jax.nn.softplus(gate))
        beta = short_conv._sigmoid(beta_ref[...].astype(jnp.float32), interpret=interpret)
        query, key, value = query[0], key[0], value[0]
        out, updated = _update(state_ref[...].astype(jnp.float32), query, key, value,
                               jnp.broadcast_to(decay, key.shape), beta)
        state_out[...] = _held(updated, key, value, state_out.dtype)
        # `gated_rms_normalized`: the norm in fp32, its mean a product by the reciprocal as XLA's.
        inverse = jax.lax.rsqrt(jnp.sum(out * out) * (1.0 / out.shape[-1]) + epsilon)
        gated = norm_swiglu.activated(z_ref[...].astype(jnp.float32), activation)[0]
        out_ref[...] = (out * inverse * weight_ref[...].astype(jnp.float32) * gated).astype(out_ref.dtype)
        write([stream[1:] for stream in windows])

    @pl.when(jnp.logical_not(active))
    def _():
        out_ref[...] = jnp.zeros(out_ref.shape, out_ref.dtype)
        write([[held_ref[j, pl.ds(*piece)][None, :] for j in range(count - 1)] for piece in pieces])


@dataclasses.dataclass(frozen=True)
class Warps:
    """A fused decode launch's warps: a program holds a value head's whole `[Dk, Dv]` state."""
    warps: int


FUSED = Warps(8)
"""The fused launch without a measurement."""


@dataclasses.dataclass(frozen=True)
class FusedStep(tuning.KernelOp):
    """A gated delta net's decode token in one launch over rows and value heads (`_fused_kernel`): the
    conv over the held history, the gates, the update and the gated norm. Off a GPU it interprets."""

    config_cls: ClassVar[type[Warps]] = Warps
    ROWS: ClassVar[tuple[str, ...]] = ('active', 'token', 'held', 'gate', 'beta', 'z', 'state')

    def _fwd(self, active: jax.Array, token: jax.Array, held: jax.Array, taps: jax.Array, gate: jax.Array,
             beta: jax.Array, log_decay: jax.Array, bias: jax.Array, z: jax.Array, weight: jax.Array,
             state: jax.Array, *, epsilon: float, activation: str, dtype: str, interpret: bool, config: Warps,
             return_residuals: bool = False):
        from jax.experimental.pallas import triton as plgpu

        rows, value_heads, key_width, value_width = state.shape
        channels = token.shape[-1]
        key_heads = (channels - value_heads * value_width) // (2 * key_width)

        def row(*shape):
            """A row's whole block."""
            return pl.BlockSpec((None, *shape), lambda r, h: (r, *(0,) * len(shape)))

        def scalar(*, per_row: bool):
            """The program's head's one entry, `[1]`, of each row's where `per_row`."""
            return (pl.BlockSpec((None, 1), lambda r, h: (r, h)) if per_row
                    else pl.BlockSpec((1,), lambda r, h: (h,)))

        headed = pl.BlockSpec((None, None, value_width), lambda r, h: (r, h, 0))
        held_spec = row(*held.shape[1:])
        state_spec = pl.BlockSpec((None, None, key_width, value_width), lambda r, h: (r, h, 0, 0))
        kernel = functools.partial(_fused_kernel, group=value_heads // key_heads, key_heads=key_heads,
                                   epsilon=epsilon, activation=activation, interpret=interpret)
        return tuple(pl.pallas_call(
            kernel, grid=(rows, value_heads),
            in_specs=[row(1), row(channels), held_spec, pl.BlockSpec(taps.shape, lambda r, h: (0, 0)),
                      scalar(per_row=True), scalar(per_row=True), scalar(per_row=False), scalar(per_row=False),
                      headed,
                      pl.BlockSpec((value_width,), lambda r, h: (0,)), state_spec],
            out_specs=[state_spec, held_spec, headed],
            out_shape=[jax.ShapeDtypeStruct(state.shape, state.dtype),
                       jax.ShapeDtypeStruct(held.shape, held.dtype),
                       jax.ShapeDtypeStruct(z.shape, jnp.dtype(dtype))],
            input_output_aliases={10: 0},
            compiler_params=plgpu.CompilerParams(num_warps=config.warps),
            interpret=interpret, name="gated_delta_decode",
        )(active, token, held, taps, gate, beta, log_decay, bias, z, weight, state)), None

    def _get_heuristics_config(self, ba) -> Warps:
        return FUSED

    def _get_autotuning_configs(self, ba) -> set[Warps]:
        return {Warps(warps) for warps in (4, 8, 16)}

    def operands(self, specs: list[jax.ShapeDtypeStruct], seed: int) -> list[jax.Array]:
        """Normal floats and every row active, so a timed launch steps each row's state and history."""
        return [jnp.ones(spec.shape, spec.dtype) if jnp.issubdtype(spec.dtype, jnp.integer) else array
                for spec, array in zip(specs, super().operands(specs, seed), strict=True)]


def fused_fits(state) -> bool:
    """Whether `fused_step` takes this `[rows, heads, Dk, Dv]` state: a GPU Dew's Triton kernels run
    on (`triton_runs`), held in fp32 or bf16, its widths powers of two (Triton's blocks); a program
    holds a head's whole state, with no column blocks to divide."""
    dk, dv = state.shape[-2:]
    return (triton_runs() and state.dtype in (jnp.float32, jnp.bfloat16)
            and all(width > 0 and width & (width - 1) == 0 for width in (dk, dv)))


def fused_step(token, held, taps, gate, beta, log_decay, bias, z, weight, state, *, epsilon: float,
               activation: str, dtype, active=None):
    """`(output, history', state')` of a gated delta net's decode token in one launch: `token` `[rows,
    C]` the conv's input, its queries, keys and values head by head in turn, `held` `[rows, K - 1, C]`
    its history and `taps` `[C, K]`, `gate` and `beta` `[rows, Hv]` the token's decay and write gates
    before their activations, `log_decay` (A_log) and `bias` (dt_bias) `[Hv]`, `z` `[rows, Hv, Dv]`
    the output gate, `weight` `[Dv]` the gated norm's, of `epsilon` and gate `activation`, and
    `state` `[rows, Hv, Dk, Dv]` held as `step`'s is. The output `[rows, Hv, Dv]` is in `dtype`, the
    history in its dtype; the state is updated in place where the caller lets it go. A row `active`
    `[rows]` marks False keeps its history and state, its output zeros. The arithmetic is the unfused
    layer's (the conv and its silu in fp32 over the history rounded to the token's dtype, the queries
    and keys l2-normalized, `step`, then the gated norm in fp32), its sums in another order and its
    quotients and roots rounded once."""
    rows = state.shape[0]
    flags = (jnp.ones(rows, jnp.int32) if active is None else active.astype(jnp.int32))[:, None]
    state, history, out = FusedStep()(flags, token, held, taps, gate, beta, log_decay, bias, z, weight, state,
                                      epsilon=epsilon, activation=activation, dtype=jnp.dtype(dtype).name,
                                      interpret=not triton_runs())
    return out, history, state
