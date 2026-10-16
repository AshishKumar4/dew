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
from reference_error import float64_terms

from dew.nn import linear
from dew.nn.linear import FORGET

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


def fla_decay_parts(q, k, v, g, beta, h0, do, dht, starts=None, chunk: int = CHUNKED):
    """The decay gradient Kimi's port sums, as fla's `chunk_kda_bwd` decomposes it, a value head at a
    time `[B, T, HV, K]`: within each chunk, each token's parts from the memory and the WY
    representation (`bwd_wy_dqkg`: `q dq_mem`, `-k dk_mem`, the chunk's last row's memory term,
    `k exp(g) dkgb beta`) and from the scores within the chunk (`bwd_intra`: `q dq2`, `k dk2`,
    `-k dkt`), each summed back over its chunk, and their sum: the gradient, the parts as its
    float64 path adds them, for `float64_terms`. `starts` `[B, T]` marks packed documents as the
    port runs them, the decay forgotten at each one's first token. A row's first token's and a
    document's first token's gradients are the port's own (`fla_kda.first_decay_grad`, a gradient
    stopped), not these."""
    dtype = q.dtype
    batch, length, heads, width = k.shape
    if starts is not None:
        g = g + jnp.where(jnp.asarray(starts)[..., None, None], FORGET, 0.0).astype(dtype)
    pad = -length % chunk
    q, k, v, g, do = (jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0))) for x in (q, k, v, g, do))
    beta = jnp.pad(beta, ((0, 0), (0, pad), (0, 0)))
    count = (length + pad) // chunk

    def chunks(x):
        moved = jnp.moveaxis(x, 2, 1)
        return moved.reshape(batch, heads, count, chunk, *moved.shape[3:])

    q, k, v, g, do = chunks(q * width ** -0.5), chunks(k), chunks(v), chunks(g), chunks(do)
    beta = chunks(beta)[..., None]
    gc = jnp.cumsum(g, axis=3)
    gated, last = jnp.exp(gc), gc[..., -1:, :]
    lower = jnp.tril(jnp.ones((chunk, chunk), bool))
    strict = jnp.tril(jnp.ones((chunk, chunk), bool), -1)
    decay = jnp.where(lower[..., None], jnp.exp(gc[..., :, None, :] - gc[..., None, :, :]), 0.0)
    akk = jnp.where(strict, jnp.einsum('...sd,...td,...std->...st', k * beta, k, decay), 0.0)
    aqk = jnp.einsum('...sd,...td,...std->...st', q, k, decay)
    inverse = linear.strictly_lower_inverse(-akk)
    w, u = inverse @ (k * beta * gated), inverse @ (v * beta)
    leaving = k * jnp.exp(last - gc)

    def forward(h, step):
        w_c, u_c, last_c, leaving_c = step
        corrected = u_c - w_c @ h
        written = h * jnp.swapaxes(jnp.exp(last_c), -1, -2) + jnp.swapaxes(leaving_c, -1, -2) @ corrected
        return written, (h, corrected)

    steps = tuple(jnp.moveaxis(x, 2, 0) for x in (w, u, last, leaving))
    _, (entered, corrected) = jax.lax.scan(forward, h0, steps)
    entered, corrected = jnp.moveaxis(entered, 0, 2), jnp.moveaxis(corrected, 0, 2)
    daqk = jnp.where(lower, do @ jnp.swapaxes(corrected, -1, -2), 0.0)
    local = jnp.swapaxes(aqk, -1, -2) @ do

    def backward(dh, step):
        q_c, w_c, do_c, local_c, last_c, leaving_c = step
        dcorrected = local_c + leaving_c @ dh
        entered_grad = (dh * jnp.swapaxes(jnp.exp(last_c), -1, -2) + jnp.swapaxes(q_c, -1, -2) @ do_c
                        - jnp.swapaxes(w_c, -1, -2) @ dcorrected)
        return entered_grad, (dh, dcorrected)

    reversed_ = tuple(jnp.flip(jnp.moveaxis(x, 2, 0), 0) for x in (q * gated, w, do, local, last, leaving))
    _, (written_grad, dcorrected) = jax.lax.scan(backward, dht, reversed_)
    written_grad, dcorrected = (jnp.moveaxis(jnp.flip(x, 0), 0, 2) for x in (written_grad, dcorrected))
    dq_mem = (do @ jnp.swapaxes(entered, -1, -2)) * gated
    dk_mem = (corrected @ jnp.swapaxes(written_grad, -1, -2)) * jnp.exp(last - gc)
    held = jnp.sum(entered * written_grad, axis=-1)[..., None, :] * jnp.exp(last)
    held = held + jnp.sum(k * dk_mem, axis=3, keepdims=True)
    closing = jnp.where(jnp.arange(chunk)[:, None] == chunk - 1, held, 0.0)
    dw = -(dcorrected @ jnp.swapaxes(entered, -1, -2))
    dkgb = jnp.swapaxes(inverse, -1, -2) @ dw
    scores = dcorrected @ jnp.swapaxes(v, -1, -2) + dw @ jnp.swapaxes(k * gated, -1, -2)
    scores = scores * jnp.swapaxes(beta, -1, -2)
    inverse_t = jnp.swapaxes(inverse, -1, -2)
    dakk = jnp.where(strict, -(inverse_t @ jnp.where(strict, scores, 0.0) @ inverse_t), 0.0)
    dq2 = jnp.einsum('...st,...td,...std->...sd', daqk, k, decay)
    dk2 = jnp.einsum('...st,...td,...std->...sd', dakk, k, decay) * beta
    dkt = (jnp.einsum('...st,...sd,...std->...td', daqk, q, decay)
           + jnp.einsum('...st,...sd,...std->...td', dakk, k * beta, decay))
    parts = (q * dq_mem, -k * dk_mem, closing, k * gated * dkgb * beta, q * dq2, k * dk2, -k * dkt)

    def back(x):
        summed = jax.lax.cumsum(x, axis=3, reverse=True)
        return jnp.moveaxis(summed.reshape(batch, heads, count * chunk, width), 1, 2)[:, :length]

    return back(sum(parts)), sum(back(part) for part in parts)


def fla_decay_terms(q, k, v, g, beta, h0, do, dht, starts=None, chunk: int = CHUNKED):
    """The float64 terms of the decay gradient Kimi's port sums, `[B, T, HV, K]`, a value head at a
    time: the larger of its parts' (`fla_decay_parts`) summed back apart and summed back as the
    kernels add them, each position's parts first, so that cancelling parts never shrink a size. A
    row's first token and a packed document's first token take none (zero): their gradients are the
    port's own."""
    _, (summed, apart) = float64_terms(lambda *a: fla_decay_parts(*a, starts=starts, chunk=chunk),
                                       q, k, v, g, beta, h0, do, dht)
    terms = np.maximum(np.asarray(summed), np.asarray(apart))
    first = np.zeros(terms.shape[:2], bool)
    first[:, 0] = True
    if starts is not None:
        first |= np.asarray(starts, bool)
    return np.where(first[..., None, None], 0.0, terms)
