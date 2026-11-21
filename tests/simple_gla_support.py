"""fla's simple GLA, lightning attention's bar under the run's fast forms (docs/design/testing.md: a cell is
held to its form's release's own code; `recurrence_chunk.simple_gla`, `chunk_rounding`).

Its arithmetic stated in JAX for a host without fla's GPU tier reference (a CPU lane's interpreted cells,
`fla_rounding`), and fla's own `chunk_simple_gla` run in the interpreter `DEW_FLA_PYTHON` names where a
host holds it (`fla_simple_gla`): fla 0.5.2 stores each chunk's entering state bf16 and reads it so for the
output's `q h`, keeps the running state fp32, casts the decayed values bf16 before `k^T v`, and casts the
decayed causal scores bf16 before `A v`, every product bf16 x bf16 into fp32, the output bf16 and the final
state fp32 (ops/common/chunk_h.py:84-151, 319-335; chunk_o.py:88-139; ops/simple_gla/chunk.py:31-43,
171-201)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from delta_rule_support import fla_run

from dew.nn.kernels.recurrence_chunk import Chunk


def fla_rounding(operands: Chunk, *, chunk: int):
    """The chunked recurrence of `operands` (queries, keys and values bf16, a log decay a head fp32, a memory
    entering where given) as fla's `chunk_simple_gla` rounds it, its output and final memory: a value head
    reads its key head's queries and keys, the row padded at its end to whole chunks with no write and no
    decay. Its gradients are this statement's own (autodiff casts at the same products' transposes, where fla
    recomputes its states fp32 and casts them bf16 at each backward product)."""
    query, key, value, decay = operands.query, operands.key, operands.value, operands.log_decay
    length = query.shape[1]
    pad = -length % chunk
    query, key, value, decay = (jnp.pad(x, ((0, 0), (0, pad)) + ((0, 0),) * (x.ndim - 2))
                                for x in (query, key, value, decay))
    query, key = (jnp.repeat(x, value.shape[2] // x.shape[2], axis=2) for x in (query, key))
    rows, laid, heads, width = query.shape

    def bf16(x):
        # A bf16 operand, its products exact in fp32 (bf16 x bf16 into fp32, as fla's `tl.dot`).
        return x.astype(jnp.bfloat16).astype(jnp.float32)

    def product(spec, a, b):
        return jnp.einsum(spec, a, b, precision=jax.lax.Precision.HIGHEST)

    def chunked(x):
        return jnp.moveaxis(x.reshape(rows, laid // chunk, chunk, *x.shape[2:]), 1, 0)

    entered = (jnp.zeros((rows, heads, width, value.shape[-1]), jnp.float32) if operands.state is None
               else operands.state.astype(jnp.float32))

    def step(state, chunk_operands):
        q, k, v, g = chunk_operands
        summed = jnp.cumsum(g.astype(jnp.float32), axis=1)
        last = summed[:, -1]
        carried = product('bchk,bhkv->bchv', bf16(q), bf16(state)) * jnp.exp(summed)[..., None]
        headed = jnp.moveaxis(summed, 1, 2)
        between = jnp.exp(headed[..., :, None] - headed[..., None, :])
        scores = product('bihk,bjhk->bhij', bf16(q), bf16(k)) * between
        scores = jnp.where(jnp.tril(jnp.ones((chunk, chunk), bool)), scores, 0.0)
        out = (carried + product('bhij,bjhv->bihv', bf16(scores), bf16(v))).astype(jnp.bfloat16)
        decayed = bf16(v.astype(jnp.float32) * jnp.exp(last[:, None] - summed)[..., None])
        return state * jnp.exp(last)[..., None, None] + product('bchk,bchv->bhkv', bf16(k), decayed), out

    final, out = jax.lax.scan(step, entered, tuple(chunked(x) for x in (query, key, value, decay)))
    return jnp.moveaxis(out, 0, 1).reshape(rows, laid, heads, -1)[:, :length], final


SIMPLE_GLA_SIDE = '''
import sys, numpy as np, torch
from fla.ops.simple_gla import chunk_simple_gla
given = np.load(sys.argv[1])
load = lambda name, dtype: torch.from_numpy(given[name]).to("cuda", dtype)
q, k, v = (load(name, torch.bfloat16).requires_grad_() for name in ("query", "key", "value"))
g = load("log_decay", torch.float32).requires_grad_()
h0 = load("state", torch.float32).requires_grad_() if "state" in given else None
groups = v.shape[2] // q.shape[2]
o, final = chunk_simple_gla(q.repeat_interleave(groups, 2), k.repeat_interleave(groups, 2), v, g=g,
                            scale=1.0, initial_state=h0, output_final_state=True)
torch.autograd.backward((o, final), (load("d_out", torch.bfloat16), load("d_final", torch.float32)))
leaves = {"query": q, "key": k, "value": v, "log_decay": g, **({"state": h0} if h0 is not None else {})}
np.savez(sys.argv[2], out=o.detach().float().cpu().numpy(), final=final.detach().float().cpu().numpy(),
         **{f"d_{name}": leaf.grad.float().cpu().numpy() for name, leaf in leaves.items()})
'''
"""fla 0.5.2's `chunk_simple_gla` on the operands and cotangents of an npz, its output, final memory and every
leaf's gradient into another: the GPU tier's bar for lightning attention's form (`fla_simple_gla`)."""


def fla_simple_gla(operands: Chunk, cotangents: tuple):
    """fla's own `chunk_simple_gla` on `operands` under `cotangents` (`d_out`, `d_final`), in the interpreter
    `DEW_FLA_PYTHON` names (`SIMPLE_GLA_SIDE`): its output and final memory, and each leaf's gradient as
    `operands`' tree holds it."""
    names = ("query", "key", "value", "log_decay", "state")
    leaves = {name: np.asarray(getattr(operands, name), np.float32)
              for name in names if getattr(operands, name) is not None}
    theirs = fla_run(SIMPLE_GLA_SIDE, {**leaves, "d_out": np.asarray(cotangents[0], np.float32),
                                       "d_final": np.asarray(cotangents[1], np.float32)})
    gradients = {name: jnp.asarray(theirs[f"d_{name}"], getattr(operands, name).dtype) for name in leaves}
    # As `reference_error.with_gradients` returns a call's: its outputs, then its operands' gradients.
    return ((jnp.asarray(theirs["out"], operands.value.dtype), jnp.asarray(theirs["final"])),
            dataclasses.replace(operands, **gradients))
