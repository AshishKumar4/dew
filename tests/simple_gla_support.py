"""fla's simple GLA, lightning attention's bar under the run's fast forms (docs/design/testing.md: a cell is
held to its form's release's own code; `recurrence_chunk.simple_gla`, `chunk_rounding`): fla's own
`chunk_simple_gla` run in the interpreter `DEW_FLA_PYTHON` names where a host holds it (`fla_simple_gla`),
recorded at its sites for a host without one (`fla_sites`' 'lightning')."""

import dataclasses

import jax.numpy as jnp
import numpy as np
from delta_rule_support import fla_run

from dew.nn.kernels.recurrence_chunk import Chunk

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
