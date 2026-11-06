"""The recurrence template's decode step on Mosaic GPU: one token over a row's state, a program a row, value
head and block of `scan.THREADS` value columns, each thread one column of the state (kernel-templates.md §5).

A decode token reads its whole state once and writes it once; nothing in it is a product worth an MMA. A
thread holds its value column's `Dk` entries in registers, so every sum over the key dimension is its own
(`recurrence.update`'s `k^T S` and `q^T S'`, no exchange), and the key dimension's vectors (the queries and
keys normed and scaled, the decay) are staged once in shared memory, each read there as a scalar by every
thread. The arithmetic is `recurrence.step_reference`'s, fp32 throughout, the held state rounded as it does
(`recurrence._held`): one hook path, the portable math's, which a call's gradient takes too
(`recurrence.step`'s `linear_tangent`).
"""

import dataclasses

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.kernels import recurrence
from dew.nn.kernels.templates import mosaic, scan

L2NORM_EPS = recurrence.L2NORM_EPS


def serves(hooks: recurrence.Hooks, operands: recurrence.Step) -> bool:
    """Whether a token of `hooks` and `operands` steps here: no conv in its prologue (the fused step's, which
    reads a row's channels whole), its value heads' width a whole number of thread blocks and its key
    dimension the key vectors' (a thread's column of the state)."""
    if hooks.conv is not None:
        return False
    value_width, key_width = operands.state.shape[-1], operands.state.shape[-2]
    # The gated norm sums over a head's whole output: one block of columns a head.
    whole = hooks.output is None or value_width == scan.THREADS
    return value_width % scan.THREADS == 0 and key_width % scan.THREADS == 0 and whole


def step(hooks: recurrence.Hooks, operands: recurrence.Step, *, dtype, family: mosaic.Family,
         interpret: bool = False):
    """`recurrence.step`'s launch on Mosaic GPU: `(output [rows, Hv, Dv], None, state')`, the state as it is
    held, an idle row's (`Step.active`) as it came and its output zeros."""
    given = recurrence.read(hooks, operands)
    state = given['state']
    # A row's entry a head, `[rows, Hv]`, as the token's step takes its write gate and scalar decay.
    for name in ('beta', 'gate') + (('log_decay',) if hooks.decay == 'scalar' else ()):
        if name in given:
            given[name] = given[name].reshape(state.shape[0], -1)
    rows, value_heads, key_width, value_width = state.shape
    key_heads = given['query'].shape[1]
    out_dtype = jnp.dtype(given['value'].dtype if dtype is None else dtype)
    active = (jnp.ones((rows,), jnp.int32) if operands.active is None
              else jnp.asarray(operands.active).astype(jnp.int32))
    names = tuple(given)
    blocks = value_width // scan.THREADS

    def entry(active_ref, *refs):
        inputs = dict(zip(names, refs[:len(names)], strict=True))
        state_out, out_ref, vectors, exchanged = refs[len(names):]
        _program(hooks, inputs, active_ref, state_out, out_ref, vectors, exchanged,
                  group=value_heads // key_heads, interpret=interpret)

    out_type = (jax.ShapeDtypeStruct(state.shape, state.dtype),
                jax.ShapeDtypeStruct((rows, value_heads, value_width), out_dtype))
    launched = mosaic.kernel(entry, out_type=out_type, grid=(rows, value_heads, blocks),
                             grid_names=("rows", "heads", "columns"), warpgroups=1, interpret=interpret,
                             family=family.name, name="recurrence_step",
                             scratch=(plgpu.SMEM((3, key_width), jnp.float32), scan.scratch()))
    state_out, out = launched(active, *given.values())
    return out, None, state_out


@dataclasses.dataclass(frozen=True)
class _Token:
    """A program's token: its row, value head, key head and value columns."""

    row: jax.Array
    head: jax.Array
    keyed: jax.Array
    columns: pl.Slice


def _program(hooks, inputs, active_ref, state_out, out_ref, vectors, exchanged, *, group: int,
             interpret: bool):
    """A program's token (`step`): its key vectors staged, its state column stepped, its output and state
    stored; an idle row's state copied and its output zeros."""
    head = lax.axis_index("heads")
    token = _Token(lax.axis_index("rows"), head, head // group,
                   pl.ds(lax.axis_index("columns") * scan.THREADS, scan.THREADS))
    state_ref = inputs['state']
    key_width = state_ref.shape[2]
    layout = scan.blocked(scan.THREADS)

    def column(ref, k):
        return plgpu.load(ref.at[token.row, token.head, k, token.columns], layout=layout,
                          optimized=False).astype(jnp.float32)

    @pl.when(active_ref[token.row] != 0)
    def _():
        write = _staged(hooks, inputs, token, vectors, exchanged, interpret=interpret)
        value = plgpu.load(inputs['value'].at[token.row, token.head, token.columns], layout=layout,
                           optimized=False).astype(jnp.float32)
        held = [column(state_ref, k) for k in range(key_width)]
        out, updated = _stepped(hooks, held, value, vectors, write)
        out = _epilogue(hooks, inputs, token, out, value, exchanged, interpret=interpret)
        for k, entry in enumerate(updated):
            # Held as `recurrence._held` holds it: bf16 rounded stochastically, salted by its key and value.
            kept = (recurrence.salted_bf16(entry, vectors[1, k], value) if state_out.dtype == jnp.bfloat16
                    else entry.astype(state_out.dtype))
            plgpu.store(state_out.at[token.row, token.head, k, token.columns], kept, optimized=False)
        plgpu.store(out_ref.at[token.row, token.head, token.columns], out.astype(out_ref.dtype),
                    optimized=False)

    @pl.when(active_ref[token.row] == 0)
    def _():
        for k in range(key_width):
            plgpu.store(state_out.at[token.row, token.head, k, token.columns],
                        column(state_ref, k).astype(state_out.dtype), optimized=False)
        plgpu.store(out_ref.at[token.row, token.head, token.columns],
                    jnp.zeros((scan.THREADS,), out_ref.dtype), optimized=False)


def _staged(hooks, inputs, token: _Token, vectors, exchanged, *, interpret: bool):
    """The token's key vectors in `vectors`, fp32: its queries normed and scaled, its keys normed (rows 0
    and 1) and its decay a key dimension (row 2), as `recurrence.step_reference`'s prologue and gates form
    them; its write gate returned, a scalar."""
    key_width = vectors.shape[1]
    layout = scan.blocked(key_width)

    def keyed(name):
        return plgpu.load(inputs[name].at[token.row, token.keyed], layout=layout,
                          optimized=False).astype(jnp.float32)

    query, key = keyed('query'), keyed('key')
    if hooks.l2norm:
        query = query * lax.rsqrt(scan.summed(query * query, exchanged, interpret=interpret) + L2NORM_EPS)
        key = key * lax.rsqrt(scan.summed(key * key, exchanged, interpret=interpret) + L2NORM_EPS)
        query = query * (key_width ** -0.5 if hooks.scale is None else hooks.scale)
    elif hooks.scale is not None:
        query = query * hooks.scale
    decay, write = _gates(hooks, inputs, token, layout)
    for row, value in enumerate((query, key, jnp.broadcast_to(decay, (key_width,)))):
        plgpu.store(vectors.at[row], value, optimized=False)
    mosaic.synced(interpret)
    return write


def _gates(hooks, inputs, token: _Token, layout):
    """The token's decay (a scalar, or a key dimension's vector in `layout`) and write gate (a scalar), as
    `recurrence.step_reference`'s `_gates` form them."""
    def scalar(name):
        """A row's entry a head (`[rows, Hv]`), fp32."""
        return inputs[name][token.row, token.head].astype(jnp.float32)

    def headed(name):
        """A head's entry (`[Hv]`), fp32."""
        return inputs[name][token.head].astype(jnp.float32)

    one = jnp.float32(1.0)
    if hooks.gates is not None and hooks.gates.write == 'sigmoid' and hooks.gates.limit is None:
        decay = jnp.exp(recurrence._gated_decay(scalar('gate'), headed('log_a'), headed('bias')))
        return decay, jax.nn.sigmoid(scalar('beta'))
    if hooks.gates is not None:
        rate = jax.nn.softplus(scalar('gate') + headed('bias'))
        if hooks.gates.limit is not None:
            rate = jnp.clip(rate, *hooks.gates.limit)
        decay = jnp.exp(-jnp.exp(headed('log_a')) * rate)
        return decay, rate if hooks.gates.write == 'step' else jax.nn.sigmoid(scalar('beta'))
    if hooks.rated:
        write = scalar('beta')
        return jnp.exp(headed('rate') * write), write
    write = scalar('beta') if hooks.gated_write else one
    if hooks.decay == 'none':
        return one, write
    if hooks.decay == 'vector':
        decay = plgpu.load(inputs['log_decay'].at[token.row, token.head], layout=layout, optimized=False)
        return jnp.exp(decay.astype(jnp.float32)), write
    return jnp.exp(scalar('log_decay')), write


def _stepped(hooks, held: list, value, vectors, write):
    """`recurrence.update` over a thread's column `held` (a `Dk` list of its entries): the output, `q^T S'`,
    and the updated column, each key dimension's vectors read from `vectors` as scalars."""
    key_width = len(held)
    if hooks.delta:
        corrected = value
        for k in range(key_width):
            corrected = corrected - (vectors[1, k] * vectors[2, k]) * held[k]
    else:
        corrected = value
    updated = [held[k] * vectors[2, k] + (vectors[1, k] * write) * corrected for k in range(key_width)]
    out = jnp.zeros_like(value)
    for k in range(key_width):
        out = out + vectors[0, k] * updated[k]
    return out, updated


def _epilogue(hooks, inputs, token: _Token, out, value, exchanged, *, interpret: bool):
    """The token's output past the update: its skip and its gated norm over the head's columns, as
    `recurrence.step_reference`'s epilogue (`recurrence.skipped`, `recurrence.normed`)."""
    if hooks.skip:
        out = out + inputs['skip'][token.head].astype(jnp.float32) * value
    if hooks.output is None:
        return out
    layout = scan.blocked(scan.THREADS)
    gate = plgpu.load(inputs['z'].at[token.row, token.head, token.columns], layout=layout,
                      optimized=False).astype(jnp.float32)
    weight = plgpu.load(inputs['weight'].at[token.columns], layout=layout,
                        optimized=False).astype(jnp.float32)
    # As `precision.rms_statistics` takes it: the sum times the count's reciprocal.
    mean = scan.summed(out * out, exchanged, interpret=interpret) * (1.0 / inputs['value'].shape[-1])
    return recurrence.rms_weighted(out, lax.rsqrt(mean + hooks.output.epsilon), weight=weight, gate=gate,
                                   activation=hooks.output.activation)
