"""The float64 oracle the delta rules' tests share: the gated delta rule token by token (`rule_math`),
grouped value heads and packed documents over it (`recurrence`), the gradient harness every rule
is run through (`through`), fla's own outputs and gradients, the ports' reference (`fla_side`), the
float64 terms of the decay gradient Kimi's port sums (`fla_decay_terms`), the port's plain decay
prefix (`plain_prefix`), and the portable rule as the port runs it (`math`)."""

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
from reference_error import float64_terms, widened

from dew.nn import linear
from dew.nn.kernels import fla_gdn
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
    last_decay = jnp.exp(last - gc)
    leaving = k * last_decay
    entered, corrected, _ = linear.xla_chunk_states(k, w, u, gc, last_decay, h0)
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
    return linear.chunk_delta_rule(q, k, v, g if g.ndim == 4 else g[..., None], beta, h0, starts=starts)


HEADS = 8
"""The value heads `float64_truth` evaluates the portable math's terms over at once: fewer operations
dispatched one by one, at a few GB of host memory a block for Kimi's decay per key dimension."""


def float64_truth(inputs, cuts) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The token recurrence's outputs and gradients in float64 (`recurrence`), a value head at a time
    so a head's backward fits, and each one's float64 terms, evaluated op by op on the host: what the
    sums of Dew's portable math add on the whole row, its documents from their starts as the port runs
    them (`float64_terms`, `HEADS` value heads at a time), and for a decay per key dimension's
    gradient those of the parts Kimi's port sums besides (`fla_decay_terms`). A key head's gradients
    and terms are summed over the value heads it serves."""
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
        step, packed = max(1, HEADS // group), functools.partial(math, starts=starts)
        with jax.default_device(host):
            for first in range(0, q.shape[2], step):
                keyed, valued = slice(first, first + step), slice(first * group, (first + step) * group)
                block = jax.device_put(widened((
                    q[:, :, keyed], k[:, :, keyed], v[:, :, valued], g[:, :, valued], beta[:, :, valued],
                    h0[:, valued], do[:, :, valued], dht[:, valued])), host)
                _, terms = float64_terms(through(packed, *block[6:]), *block[:6])
                sizes.append([np.asarray(x) for x in terms])

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


def plain_prefix(g, chunk: int = CHUNKED):
    """`g` re-cut from the port's plain fp32 prefix (`fla_gdn.chunk_local_cumsum`, fla's): each chunk
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


def cumulated(g, chunk: int = CHUNKED, scan=jnp.cumsum):
    """The port's decay prefix as the XLA reference takes it (`fla_gdn.chunk_local_cumsum`'s value):
    each row-aligned chunk of `chunk` tokens summed by `scan`, then scaled to base 2
    (`fla_gdn.RCP_LN2`), in `g`'s dtype."""
    batch, length = g.shape[:2]
    padded = jnp.pad(g, ((0, 0), (0, -length % chunk), *((0, 0),) * (g.ndim - 2)))
    summed = scan(padded.reshape(batch, -1, chunk, *g.shape[2:]), axis=2) * fla_gdn.RCP_LN2
    return summed.reshape(padded.shape)[:, :length]


def _decays(p):
    """`2^(p_i - p_j)` for `j <= i` within each chunk, zero above, `[B, N, H, C, C, D]`, from a
    prefix `[B, N, C, H, D]`; masked before exp2, so no range above the diagonal overflows."""
    p = jnp.moveaxis(p, 3, 2)
    lower = jnp.tril(jnp.ones((p.shape[-2], p.shape[-2]), bool))[..., None]
    return jnp.where(lower, jnp.exp2(jnp.where(lower, p[..., :, None, :] - p[..., None, :, :], 0.0)), 0.0)


def _keyed(prefix):
    """A prefix `[B, N, C, H]` or `[B, N, C, H, K]` with a key axis, one wide for a decay a head."""
    return prefix if prefix.ndim == 5 else prefix[..., None]


def _widened(*xs):
    """Each of `xs` in their common dtype, at least fp32 (float64 for float64)."""
    work = jnp.promote_types(jnp.result_type(*xs), jnp.float32)
    return tuple(jnp.asarray(x).astype(work) for x in xs)


def _inverse(a):
    """`(I + A)^-1` of the strictly lower `A` `[B, N, H, C, C]`, laid out `[B, N, C, H, C]`."""
    return jnp.moveaxis(linear.strictly_lower_inverse(-a), 3, 2)


def _strict(x):
    return jnp.where(jnp.tril(jnp.ones(x.shape[-2:], bool), -1), x, 0.0)


def port_reference(per_key: bool, scale: float) -> dict:
    """The XLA reference of each value the delta rule's port rounds and holds (`port_graph`), from the
    same parents, chunk by chunk `[B, N, C, H, ...]`, in at least fp32 (float64 for float64): the
    prefix, the scores and inverse, `w` and `u`, the gated keys, the memory each chunk enters, its
    corrected values and the last, and the output, every decay between two positions `2^(p_i - p_j)`
    and from a chunk's start `2^p` as the port takes them, by node name."""
    def prefix(g):
        (g,) = _widened(g)
        return jnp.cumsum(g, axis=2) * fla_gdn.RCP_LN2

    def pairs(left, right, p):
        """`sum_d left_id right_jd 2^(p_id - p_jd)` within each chunk, `[B, N, H, C, C]`: for a decay a
        head, the products' sum decayed after it, as fla's kernels take it."""
        decays = _decays(_keyed(p))
        if p.ndim == 4:
            return jnp.einsum('bnihd,bnjhd->bnhij', left, right) * decays[..., 0]
        # Each key decayed to the row, then the row's products summed, as Dew's portable math pairs them.
        gated = jnp.moveaxis(right, 3, 2)[:, :, :, None] * decays
        return jnp.einsum('bnihd,bnhijd->bnhij', left, gated)

    def inverse(k, prefix, beta, q=None):
        k, prefix, beta = _widened(k, prefix, beta)
        if prefix.ndim == 4:
            return _inverse(_strict(pairs(k, k, prefix) * jnp.moveaxis(beta, 3, 2)[..., None]))
        return _inverse(_strict(pairs(k * beta[..., None], k, prefix)))

    def scores(q, k, prefix, beta):
        q, k, prefix = _widened(q, k, prefix)
        return jnp.moveaxis(pairs(q, k, prefix) * scale, 3, 2)

    def u(k, v, beta, inverse, prefix):
        v, beta, inverse = _widened(v, beta, inverse)
        return jnp.einsum('bnihj,bnjhv->bnihv', inverse, v * beta[..., None])

    def w(k, v, beta, inverse, prefix):
        k, beta, inverse, prefix = _widened(k, beta, inverse, prefix)
        return jnp.einsum('bnihj,bnjhk->bnihk', inverse, k * beta[..., None] * jnp.exp2(_keyed(prefix)))

    def kg(k, prefix, **_):
        k, prefix = _widened(k, prefix)
        p = _keyed(prefix)
        return k * jnp.exp2(p[:, :, -1:] - p)

    def recurrence(keys, w, u, prefix, *, gated: bool):
        """The memory across chunks: a chunk's corrected values `u - w S`, then `S 2^(p_last)` plus the
        keys' outer products with them, the decay to the chunk's last row on the gated keys (Kimi's) or,
        for a decay a head, on the values, as fla gates them."""
        keys, w, u, prefix = _widened(keys, w, u, prefix)
        p = _keyed(prefix)
        # A head's decay to the chunk's last row, on the values, [B, N, C, H, 1]; ones for gated keys.
        toward = jnp.ones_like(p[..., :1]) if gated else jnp.exp2(p[:, :, -1:] - p)
        last = jnp.exp2(p[:, :, -1])[..., None]  # [B, N, H, D, 1]

        def one_chunk(state, step):
            w_n, u_n, keys_n, toward_n, last_n = step
            corrected = u_n - jnp.einsum('bihk,bhkv->bihv', w_n, state)
            written = jnp.einsum('bihk,bihv->bhkv', keys_n, corrected if gated else corrected * toward_n)
            return state * last_n + written, (state, corrected)

        state = jnp.zeros((*u.shape[:1], u.shape[3], keys.shape[-1], u.shape[-1]), u.dtype)
        steps = tuple(jnp.moveaxis(x, 1, 0) for x in (w, u, keys, toward, last))
        final, (states, corrected) = jax.lax.scan(one_chunk, state, steps)
        return {'states': jnp.moveaxis(states, 0, 1), 'corrected': jnp.moveaxis(corrected, 0, 1),
                'final': final}

    def output(q, prefix, states, corrected, k=None, scores=None):
        """The output from the memory each chunk entered and the chunk's own corrected values: for
        Kimi, the queries gated per key read the memory, plus the scores given; for a decay a head,
        each product's sum gated after it and both scaled last, as fla's kernels take them."""
        q, prefix, states, corrected = _widened(q, prefix, states, corrected)
        p = _keyed(prefix)
        if scores is not None:
            (scores,) = _widened(scores)
            carried = jnp.einsum('bnihk,bnhkv->bnihv', q * jnp.exp2(p), states) * scale
            return carried + jnp.einsum('bnihj,bnjhv->bnihv', scores, corrected)
        (k,) = _widened(k)
        carried = jnp.einsum('bnihk,bnhkv->bnihv', q, states) * jnp.exp2(p)
        scores = jnp.moveaxis(pairs(q, k, prefix), 3, 2)
        return carried * scale + jnp.einsum('bnihj,bnjhv->bnihv', scores, corrected) * scale

    held = {'prefix': prefix, 'inverse': inverse, 'u': u, 'w': w}
    if per_key:
        def ordered(name):
            return lambda kg, w, u, prefix: recurrence(kg, w, u, prefix, gated=True)[name]
        return {**held, 'scores': scores, 'kg': kg, **{name: ordered(name) for name in ('states', 'corrected',
                                                                                          'final')},
                'output': lambda q, prefix, scores, states, corrected: output(q, prefix, states, corrected,
                                                                              scores=scores)}

    def ordered(name):
        return lambda k, w, u, prefix: recurrence(k, w, u, prefix, gated=False)[name]
    return {**held, **{name: ordered(name) for name in ('states', 'corrected', 'final')},
            'output': lambda q, k, prefix, states, corrected: output(q, prefix, states, corrected, k=k)}
