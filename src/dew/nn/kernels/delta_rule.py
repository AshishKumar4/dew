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
the same body and block specs. On a TPU the step stays XLA's: tokamax's
Mosaic TPU `causal_conv1d_gated_delta_rule` covers the conv, the gating and
the rule over a step's tokens in one ragged call, which is the layout of a
serving step's mixed call (`dew.nn.inputs.Admitted`) rather than this one.
"""

import dataclasses
from typing import ClassVar

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from . import tuning
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


def _kernel(active_ref, state_ref, query_ref, key_ref, value_ref, decay_ref, beta_ref, state_out, out_ref):
    active = active_ref[0] != 0

    @pl.when(active)
    def _():
        state = state_ref[...].astype(jnp.float32)               # [Dk, block]
        query, key, decay = query_ref[...], key_ref[...], decay_ref[...]   # [Dk]
        read_key = jnp.sum((key * decay)[:, None] * state, axis=0)          # [block]
        read_query = jnp.sum((query * decay)[:, None] * state, axis=0)
        delta = (value_ref[...] - read_key) * beta_ref[...]
        out_ref[...] = read_query + jnp.sum(query * key) * delta
        updated = state * decay[:, None] + key[:, None] * delta[None, :]
        state_out[...] = (round_to_bf16(updated, key, value_ref[...]) if state_out.dtype == jnp.bfloat16
                          else updated.astype(state_out.dtype))

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
