"""The float64 oracle the delta rules' tests share: the gated delta rule token by token (`rule_math`),
grouped value heads and packed documents over it (`recurrence`), the gradient harness every rule
is run through (`through`), and fla's own outputs and gradients, the ports' reference (`fla_side`)."""

import itertools
import os
import subprocess
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

CHUNKED = 64
"""The terms a chunk sums in another order than the portable math: a port's fp32 floor over the
math's distance (`assert_leaves_as_exact`'s `reductions`)."""


def rule_math(q, k, v, g, beta, h0):
    """The gated delta rule token by token, in whatever dtype it is given: the reference both paths
    are held to, run in float64. `g` decays per head `[B, T, H]`, or per key dimension `[B, T, H, K]`
    (Kimi's delta attention)."""
    scale = q.shape[-1] ** -0.5

    def step(state, inputs):
        qt, kt, vt, gt, bt = inputs
        state = state * jnp.exp(gt).reshape(*gt.shape[:2], -1, 1)
        delta = (vt - jnp.einsum('bhk,bhkv->bhv', kt, state)) * bt[..., None]
        state = state + jnp.einsum('bhk,bhv->bhkv', kt, delta)
        return state, jnp.einsum('bhk,bhkv->bhv', qt * scale, state)

    final, out = jax.lax.scan(step, h0, tuple(jnp.moveaxis(x, 1, 0) for x in (q, k, v, g, beta)))
    return jnp.moveaxis(out, 0, 1), final


def recurrence(q, k, v, g, beta, h0, cuts=None, rule=rule_math):
    """`rule` (the token recurrence, `rule_math`, by default) with value heads reading their key head,
    each row through its documents (`cuts`, each row's first tokens, the first 0; None, one a row):
    each from a zero memory but a row's first, which takes the row's, the row's final memory its last
    document's."""
    group = v.shape[2] // q.shape[2]
    q, k = jnp.repeat(q, group, axis=2), jnp.repeat(k, group, axis=2)
    if cuts is None:
        return rule(q, k, v, g, beta, h0)
    outs, finals = [], []
    for row, begins in enumerate(cuts):
        state, pieces = h0[row:row + 1], []
        for begin, end in itertools.pairwise((*begins, q.shape[1])):
            state = state if begin == 0 else jnp.zeros_like(state)
            out, state = rule(*(x[row:row + 1, begin:end] for x in (q, k, v, g, beta)), state)
            pieces.append(out)
        outs.append(jnp.concatenate(pieces, axis=1))
        finals.append(state)
    return jnp.concatenate(outs), jnp.concatenate(finals)


def through(rule, do, dht):
    """`rule`'s output, final memory and its operands' gradients under the cotangents `do`, `dht`."""
    def loss(*args):
        out, final = rule(*args)
        return jnp.sum(out.astype(do.dtype) * do) + jnp.sum(final.astype(dht.dtype) * dht), (out, final)

    def run(*args):
        (_, (out, final)), grads = jax.value_and_grad(loss, argnums=range(len(args)), has_aux=True)(*args)
        return out, final, *grads

    return run


FLA_SIDE = '''
import sys, numpy as np, torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.kda import chunk_kda
given, dtype = np.load(sys.argv[1]), getattr(torch, sys.argv[3])
load = lambda name, as_dtype: torch.from_numpy(given[name]).to("cuda", as_dtype)
q, k, v, do, beta = (load(n, dtype) for n in ("q", "k", "v", "do", "beta"))
g, h0, dht = (load(n, torch.float32) for n in ("g", "h0", "dht"))
cu_seqlens = load("cu_seqlens", torch.long) if "cu_seqlens" in given else None
leaves = [x.requires_grad_() for x in (q, k, v, g, beta, h0)]
rule = chunk_kda if g.dim() == 4 else chunk_gated_delta_rule
o, final = rule(*leaves[:5], scale=q.shape[-1] ** -0.5, initial_state=leaves[5], output_final_state=True,
                cu_seqlens=cu_seqlens)
torch.autograd.backward((o, final), (do, dht))
np.savez(sys.argv[2], **{name: x.detach().float().cpu().numpy() for name, x in
                         zip(sys.argv[4:], (o, final, *(leaf.grad for leaf in leaves)), strict=True)})
'''
"""fla's side of the comparison, in the interpreter `DEW_FLA_PYTHON` names: fla 0.5.2's `chunk_kda`, or
`chunk_gated_delta_rule` for a decay per head, on the operands of an npz, its outputs and gradients
into another under the names it is given (`NAMES`); packed documents as its varlen sequences, each
with a memory of its own."""

NAMES = ("o", "final", "dq", "dk", "dv", "dg", "dbeta", "dh0")
"""The outputs and gradients each side gives, in `through`'s order."""


def fla_side(inputs, cuts, dtype) -> list[np.ndarray]:
    """fla's outputs and gradients on `inputs` (one row), run in a process of its own. Packed into
    documents at `cuts`, the row's memory is its first document's and its final memory and cotangent
    its last's: the others start from zero and give theirs no cotangent."""
    python = os.environ.get("DEW_FLA_PYTHON")
    if not python:
        pytest.skip("DEW_FLA_PYTHON names no interpreter with fla 0.5.2 and torch, the GPU tier's reference")
    given = dict(zip(("q", "k", "v", "g", "beta", "h0", "do", "dht"),
                     (np.asarray(x, np.float32) for x in inputs), strict=True))
    if cuts is not None:
        others = np.zeros((len(cuts) - 1, *given["h0"].shape[1:]), np.float32)
        given["h0"] = np.concatenate([given["h0"], others])
        given["dht"] = np.concatenate([others, given["dht"]])
        given["cu_seqlens"] = np.asarray([*cuts, given["q"].shape[1]], np.int64)
    with tempfile.TemporaryDirectory() as root:
        np.savez(Path(root) / "in.npz", **given)
        subprocess.run([python, "-c", FLA_SIDE, Path(root) / "in.npz", Path(root) / "out.npz",
                        jnp.dtype(dtype).name, *NAMES], check=True)
        found = np.load(Path(root) / "out.npz")
        theirs = {name: found[name].astype(np.float64) for name in NAMES}
    theirs["final"], theirs["dh0"] = theirs["final"][-1:], theirs["dh0"][:1]
    return [theirs[name] for name in NAMES]


def factorised_decay_terms(q, k, dq, dk, chunk: int = CHUNKED):
    """The float64 terms of the decay gradient Kimi's port takes as fla factorises it, `[B, T, HV, K]`,
    from each value head's queries and keys and their float64 gradients, `[B, T, HV, K]` each: each
    token's `q dq - k dk` summed back over its chunk of `chunk` tokens from its row's start. A token's
    terms are the root sum of squares of those products from it to its chunk's end, which its
    gradient adds and takes back where earlier keys of zero make it zero."""
    products = np.square(np.asarray(q * dq)) + np.square(np.asarray(k * dk))
    batch, length = products.shape[:2]
    padded = np.pad(products, ((0, 0), (0, -length % chunk), (0, 0), (0, 0)))
    chunks = padded.reshape(batch, -1, chunk, *products.shape[2:])
    summed_back = np.flip(np.cumsum(np.flip(chunks, 2), 2), 2)
    return np.sqrt(summed_back).reshape(padded.shape)[:, :length]
