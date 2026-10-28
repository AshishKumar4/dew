"""The float64 oracle the delta rules' tests share: the gated delta rule token by token (`rule_math`), a chunk
at a time for a real shape's gradient (`checkpointed`), grouped value heads and packed documents over it
(`recurrence`), the gradient harness every rule
is run through (`through`), fla's own outputs and gradients, the ports' reference (`fla_side`), the
float64 terms of the decay gradient Kimi's port sums (`fla_decay_terms`), the portable math and its
float64 terms (`math`, `float64_truth`), and the port's plain decay prefix (`plain_prefix`)."""

import functools
import itertools
import os
import subprocess
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import float64_terms, widened, with_gradients

from dew.nn.kernels import fla_gdn, recurrence_algebra, recurrence_chunk

CHUNKED = 64
"""The port's chunk length in tokens: the chunks `fla_decay_parts` sums within and `plain_prefix`
re-cuts the decay by, each by default."""


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


def checkpointed(chunk: int):
    """`rule_math`, its tokens scanned a `chunk` at a time with each chunk's scan recomputed on the way back,
    so its gradient keeps one memory a chunk rather than one a token: the float64 oracle at a model's real
    shape (`gdn_cases`), whose memory a token would not fit a GPU. Its values are `rule_math`'s."""
    def rule(q, k, v, g, beta, h0):
        def blocked(x):
            return jnp.moveaxis(x.reshape(x.shape[0], x.shape[1] // chunk, chunk, *x.shape[2:]), 1, 0)

        @jax.checkpoint
        def block(state, inputs):
            out, state = rule_math(*inputs, state)
            return state, out

        final, out = jax.lax.scan(block, h0, tuple(blocked(x) for x in (q, k, v, g, beta)))
        return jnp.moveaxis(out, 0, 1).reshape(v.shape), final

    return rule


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
    """`rule`'s output, final memory and its operands' gradients under the cotangents `do`, `dht`, each
    cast to its output's dtype, pulled back through the gate (`reference_error.with_gradients`)."""
    def run(*args):
        out, final = jax.eval_shape(rule, *args)
        (out, final), *grads = with_gradients(rule)(*args, (do.astype(out.dtype), dht.astype(final.dtype)))
        return out, final, *grads

    return run


FLA_SIDE = '''
import sys, numpy as np, torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule
from fla.ops.kda import chunk_kda
given, dtype = np.load(sys.argv[1]), getattr(torch, sys.argv[3])
kinds = dict(zip(("q", "k", "v", "g", "beta", "h0", "do", "dht"), str(given["dtypes"]).split(",")))
load = lambda name, as_dtype: torch.from_numpy(given[name]).to("cuda", as_dtype)
q, k, v, g, beta, h0, do, dht = (load(n, getattr(torch, kinds[n])) for n in kinds)
cu_seqlens = load("cu_seqlens", torch.long) if "cu_seqlens" in given else None
leaves = [x.requires_grad_() for x in (q, k, v, g, beta, h0)]
rule = chunk_kda if g.dim() == 4 else chunk_gated_delta_rule
# Each leaf in its own dtype, its copy in the kernels' (fla's layer's common dtype), so a gradient
# rounds back to its leaf's dtype as a mixed call's does.
o, final = rule(*(x.to(dtype) for x in leaves[:3]), leaves[3].float(), leaves[4].to(dtype),
                scale=q.shape[-1] ** -0.5, initial_state=leaves[5].float(), output_final_state=True,
                cu_seqlens=cu_seqlens)
torch.autograd.backward((o, final), (do.to(o.dtype), dht.float()))
np.savez(sys.argv[2], **{name: x.detach().float().cpu().numpy() for name, x in
                         zip(sys.argv[4:], (o, final, *(leaf.grad for leaf in leaves)), strict=True)})
'''
"""fla's side of the comparison, in the interpreter `DEW_FLA_PYTHON` names: fla 0.5.2's `chunk_kda`, or
`chunk_gated_delta_rule` for a decay per head, on the operands of an npz, each leaf in the dtype it
names and the kernels in the given one, its outputs and gradients into another under the names it is
given (`NAMES`); packed documents as its varlen sequences, each with a memory of its own."""

NAMES = ("o", "final", "dq", "dk", "dv", "dg", "dbeta", "dh0")
"""The outputs and gradients each side gives, in `through`'s order."""


def fla_side(inputs, cuts, dtype) -> list[np.ndarray]:
    """fla's outputs and gradients on `inputs` (one row), run in a process of its own, its kernels in
    `dtype` and each input a leaf in its own dtype. Packed into documents at `cuts`, the row's memory
    is its first document's and its final memory and cotangent its last's: the others start from
    zero and give theirs no cotangent."""
    python = os.environ.get("DEW_FLA_PYTHON")
    if not python:
        pytest.fail("DEW_FLA_PYTHON names no interpreter with fla 0.5.2 and torch: the GPU tier's reference, "
                    "which a run selecting fla's gate gives it")
    given = dict(zip(("q", "k", "v", "g", "beta", "h0", "do", "dht"),
                     (np.asarray(x, np.float32) for x in inputs), strict=True))
    given["dtypes"] = np.asarray(",".join(jnp.dtype(x.dtype).name for x in inputs))
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
    float64 path adds them, for `float64_terms`. `starts` `[B, T]` (host) marks packed documents,
    each a sequence of its own as the port runs them (fla's varlen): its chunks from its first token,
    its memory from zero but the row's first, the row's final memory's cotangent its last's. A row's
    first token's and a document's first token's gradients are the port's own
    (`fla_kda.first_decay_grad`), not these."""
    if starts is not None:
        rows = []
        for row, begun in enumerate(np.asarray(starts, bool)):
            cuts = [0, *(np.flatnonzero(begun[1:]) + 1), begun.size]
            pieces = []
            for index, (begin, end) in enumerate(itertools.pairwise(cuts)):
                memory, cotangent = h0[row:row + 1], dht[row:row + 1]
                pieces.append(fla_decay_parts(
                    *(x[row:row + 1, begin:end] for x in (q, k, v, g, beta)),
                    memory if index == 0 else jnp.zeros_like(memory), do[row:row + 1, begin:end],
                    cotangent if end == begun.size else jnp.zeros_like(cotangent), chunk=chunk))
            rows.append(tuple(jnp.concatenate(parts, axis=1) for parts in zip(*pieces, strict=True)))
        return tuple(jnp.concatenate(parts, axis=0) for parts in zip(*rows, strict=True))
    batch, length, heads, width = k.shape
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
    inverse = recurrence_algebra.factor(-akk)
    w, u = inverse @ (k * beta * gated), inverse @ (v * beta)
    last_decay = jnp.exp(last - gc)
    leaving = k * last_decay
    kept = jnp.ones((1, 1, gc.shape[2], 1, 1), gc.dtype)
    entered, corrected, _ = recurrence_chunk.xla_chunk_states(k, w, u, gated, last_decay, h0, kept)
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
    dk_mem = (corrected @ jnp.swapaxes(written_grad, -1, -2)) * last_decay
    held = jnp.sum(entered * written_grad, axis=-1)[..., None, :] * jnp.exp(last)
    held = held + jnp.sum(k * dk_mem, axis=3, keepdims=True)
    # On each chunk's last real token, the row's last chunk's short of its padding.
    real = jnp.minimum(chunk, length - chunk * jnp.arange(count))
    closing = jnp.where((jnp.arange(chunk) == real[:, None] - 1)[..., None], held, 0.0)
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


def math(q, k, v, g, beta, h0, starts=None):
    """The portable rule, value heads reading their key head, a decay per head `[B, T, H]` or per key
    dimension `[B, T, H, K]`, a row's packed documents starting at `starts` (`[B, T]`), as the port
    runs them."""
    group = v.shape[2] // q.shape[2]
    q, k = (jnp.repeat(x, group, axis=2) for x in (q, k))
    starts = None if starts is None else jnp.asarray(starts)
    return recurrence_chunk.chunk_delta_rule(q, k, v, g if g.ndim == 4 else g[..., None], beta, h0,
                                             starts=starts)


HEADS = 8
"""The value heads `float64_truth` evaluates the portable math's terms over at once: fewer operations
dispatched one by one, at a few GB of host memory a block for Kimi's decay per key dimension."""


def float64_truth(inputs, cuts) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The token recurrence's outputs and gradients in float64 (`recurrence`), a value head at a time
    so a head's backward fits, and each one's float64 terms, evaluated op by op on the host: what the
    sums of Dew's portable math add (`float64_terms`, `HEADS` value heads at a time), and for a decay
    per key dimension's gradient those of the parts Kimi's port sums besides (`fla_decay_terms`). A
    key head's gradients and terms are summed over the value heads it serves.

    Packed at `cuts`, the port and fla each run a document as a sequence of its own, chunked from its
    start, the row's memory the first's and its final memory and cotangent the last's; the terms are
    that layout's (`_per_document`), since a chunk's last token reads the inter-chunk sum whole (a dot
    of its key with the state's gradient, which `float64_terms` sizes at that sum's value) and chunks
    cut from the row's start would end at other tokens."""
    q, k, v, g, beta, h0, do, dht = inputs
    group = v.shape[2] // q.shape[2]
    rows = None if cuts is None else (cuts,)
    host = jax.devices("cpu")[0]
    starts = None
    if cuts is not None:
        starts = np.zeros(q.shape[:2], bool)
        starts[0, list(cuts[1:])] = True
    with jax.enable_x64(new_val=True):
        run = jax.jit(lambda q, k, v, g, beta, h0, do, dht: through(
            functools.partial(recurrence, cuts=rows), do, dht)(q, k, v, g, beta, h0))
        heads, decay = [], []
        for head in range(v.shape[2]):
            keyed = slice(head // group, head // group + 1)
            valued = slice(head, head + 1)
            sliced = widened((q[:, :, keyed], k[:, :, keyed], v[:, :, valued], g[:, :, valued],
                              beta[:, :, valued], h0[:, valued], do[:, :, valued], dht[:, valued]))
            heads.append([np.asarray(x) for x in run(*sliced)])
            if g.ndim == 4:
                with jax.default_device(host):
                    decay.append(fla_decay_terms(*jax.device_put(sliced, host), starts=starts))
        sizes = []
        step = max(1, HEADS // group)
        with jax.default_device(host):
            for first in range(0, q.shape[2], step):
                keyed, valued = slice(first, first + step), slice(first * group, (first + step) * group)
                block = jax.device_put(widened((
                    q[:, :, keyed], k[:, :, keyed], v[:, :, valued], g[:, :, valued], beta[:, :, valued],
                    h0[:, valued], do[:, :, valued], dht[:, valued])), host)
                if cuts is None:
                    _, terms = float64_terms(through(math, *block[6:]), *block[:6])
                    sizes.append([np.asarray(x) for x in terms])
                else:
                    sizes.append(_per_document(block, cuts))

    def joined(leaves):
        return [np.concatenate([each[i] for each in leaves], axis=axis)
                for i, axis in enumerate((2, 1, 2, 2, 2, 2, 2, 1))]

    def grouped(x):
        return x.reshape(*x.shape[:2], q.shape[2], group, -1).sum(axis=3)

    found, terms = joined(heads), joined(sizes)
    if decay:
        terms[5] = np.maximum(terms[5], np.concatenate(decay, axis=2))
    for i in (2, 3):
        found[i] = grouped(found[i])
    return found, terms


def _per_document(block, cuts) -> list[np.ndarray]:
    """The float64 terms of `block` (`float64_truth`'s operands and cotangents) with its packed
    documents at `cuts` each a sequence of its own, chunked from its start: the row's memory the
    first's and its final memory and cotangent the last's."""
    q, k, v, g, beta, h0, do, dht = block
    pieces = []
    for start, end in itertools.pairwise([*cuts, q.shape[1]]):
        memory = h0 if start == 0 else jnp.zeros_like(h0)
        cotangent = dht if end == q.shape[1] else jnp.zeros_like(dht)
        _, terms = float64_terms(through(math, do[:, start:end], cotangent),
                                 *(x[:, start:end] for x in (q, k, v, g, beta)), memory)
        pieces.append([np.asarray(x) for x in terms])
    timed = [np.concatenate([piece[i] for piece in pieces], axis=1) for i in (0, 2, 3, 4, 5, 6)]
    return [timed[0], pieces[-1][1], *timed[1:], pieces[0][7]]


def plain_prefix(g, chunk: int = CHUNKED):
    """`g` re-cut from the port's plain fp32 prefix (`recurrence.cumulated`, fla's): each chunk
    of `chunk` tokens from the row's start summed, then scaled to base 2 (`fla_gdn.RCP_LN2`), each
    token's decay the difference of its prefix and the one before carried back to natural log. The
    portable rule, rebuilding the prefixes, then takes every range as the port does, the difference
    of two rounded base-2 prefixes. Its gradient passes to `g` unchanged."""
    batch, length = g.shape[:2]
    padded = jnp.pad(g, ((0, 0), (0, -length % chunk), *((0, 0),) * (g.ndim - 2)))
    prefix = jnp.cumsum(padded.reshape(batch, -1, chunk, *g.shape[2:]), axis=2) * fla_gdn.RCP_LN2
    earlier = jnp.pad(prefix[:, :, :-1], ((0, 0), (0, 0), (1, 0), *((0, 0),) * (g.ndim - 2)))
    recut = ((prefix - earlier) / fla_gdn.RCP_LN2).reshape(padded.shape)[:, :length]
    return g + jax.lax.stop_gradient(recut - g)


def fused_step(token, held, taps, gate, beta, log_a, bias, z, weight, state, *, epsilon: float,
               activation: str, dtype, active=None):
    """`(output, history', state')` of a gated delta net's decode token whole in the recurrence's step
    (`dew.nn.kernels.recurrence.step`), the layer's operands in the order its fused token hands them:
    the conv's token, history and taps, the gates before their activations, A_log and dt_bias, the
    output gate and the gated norm's weight, and the state."""
    from dew.nn.kernels import recurrence

    hooks = recurrence.Hooks(conv=recurrence.Conv('silu'), l2norm=True, decay='scalar',
                             gates=recurrence.Gates(), output=recurrence.OutputNorm(epsilon, activation))
    operands = recurrence.Step(state, token=token, held=held, taps=taps, gate=gate, beta=beta, log_a=log_a,
                               bias=bias, z=z, weight=weight, active=active)
    return recurrence.step(hooks, operands, dtype=dtype)
