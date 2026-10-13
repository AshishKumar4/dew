"""fla_gdn's test support: each kernel's math in XLA, per chunk, with the kernel's casts, and its declared
tiles as abstract calls for the compile test."""

import dataclasses
import functools
from collections.abc import Callable, Iterator
from typing import NamedTuple

import jax
import jax.numpy as jnp

from dew.nn.kernels import fla_gdn, generation

COMPILED_SHAPES = {
    "0.8b": {"batch": 8, "length": 3072, "key_heads": 16, "value_heads": 16,
             "width": 128, "columns": 128, "chunk": 64},
    "9b": {"batch": 4, "length": 3072, "key_heads": 16, "value_heads": 32,
           "width": 128, "columns": 128, "chunk": 64},
    "wide keys": {"batch": 1, "length": 512, "key_heads": 2, "value_heads": 4,
                  "width": 256, "columns": 128, "chunk": 64},
}
"""The shapes every declared config compiles at: the training shapes, whose heads are `fla_gdn.WIDEST`
wide, and the widest keys the port takes in 16 bits (`fla_gdn.WIDEST_KEYS`)."""


class KernelCase(NamedTuple):
    """One declared kernel call for the compile test (tests/kernel_compile.py): the kernel with its
    config, and abstract operands at a training shape."""

    name: str
    call: Callable[[], tuple[Callable, tuple[jax.ShapeDtypeStruct, ...]]]


def _case_call(kernel, config, shape, dtype, *, reverse=False):
    batch, length, key_heads, value_heads = (shape[name] for name in
                                            ("batch", "length", "key_heads", "value_heads"))
    width, columns, chunk = (shape[name] for name in ("width", "columns", "chunk"))
    narrow = jnp.dtype(dtype)
    keyed = jax.ShapeDtypeStruct((batch, length, key_heads, width), narrow)
    weighted = jax.ShapeDtypeStruct((batch, length, value_heads, width), narrow)
    valued = jax.ShapeDtypeStruct((batch, length, value_heads, columns), narrow)
    gated = jax.ShapeDtypeStruct((batch, length, value_heads), jnp.float32)
    beta = jax.ShapeDtypeStruct(gated.shape, narrow)
    inverse = jax.ShapeDtypeStruct((batch, length, value_heads, chunk), narrow)
    entered = jax.ShapeDtypeStruct((batch, length // chunk, value_heads, width, columns), narrow)
    state = jax.ShapeDtypeStruct((batch, value_heads, width, columns), jnp.float32)
    operands = {
        "chunk_local_cumsum": (gated,),
        "kkt_solve": (keyed, gated, gated, beta),
        "recompute_w_u": (keyed, valued, beta, inverse, gated),
        "chunk_fwd_h": (keyed, weighted, valued, gated, gated, state),
        "chunk_fwd_o": (keyed, keyed, valued, entered, gated, gated),
        "chunk_bwd_dv_local": (keyed, keyed, gated, gated, valued),
        "chunk_bwd_dhu": (keyed, keyed, weighted, gated, gated, state, valued, valued),
        "chunk_bwd_dqkwg": (keyed, keyed, valued, weighted, gated, gated, entered, valued, valued, entered),
        "prepare_wy_repr_bwd": (keyed, valued, beta, gated, gated, inverse, weighted, valued, keyed),
    }
    kwargs = {**config, "chunk": chunk, "interpret": False}
    if kernel == "chunk_local_cumsum":
        kwargs.update(scale=fla_gdn.RCP_LN2, reverse=reverse)
    elif kernel in ("chunk_fwd_o", "chunk_bwd_dv_local", "chunk_bwd_dhu", "chunk_bwd_dqkwg"):
        kwargs["scale"] = width ** -0.5
    return functools.partial(getattr(fla_gdn, kernel), **kwargs), operands[kernel]


def kernel_cases(target: str) -> Iterator[KernelCase]:
    """Each kernel at the tiles `target` runs it with unmeasured (`fla_gdn.HEURISTICS`) for both dtypes
    at each shape, and the reverse cumsum, for the compile test."""
    with generation.assumed(target):
        wide = fla_gdn._wide()
    for dtype in ("bfloat16", "float32"):
        for kernel, tiles in fla_gdn.HEURISTICS[dtype, wide].items():
            config = {field: value for field, value in dataclasses.asdict(tiles).items() if value}
            for shape_name, shape in COMPILED_SHAPES.items():
                if shape["width"] > fla_gdn.WIDEST and dtype == "float32":
                    continue
                flags = "-".join(f"{key}{value}" for key, value in sorted(config.items()))
                for reverse in ((False, True) if kernel == "chunk_local_cumsum" else (False,)):
                    name = f"{target}/{kernel}/{dtype}/{flags}/{shape_name}" + "/reverse" * reverse
                    yield KernelCase(name, functools.partial(_case_call, kernel, config, shape, dtype,
                                                             reverse=reverse))


def between(later, earlier):
    """A range's exponent from two cumulated decays' `(hi, lo)` halves, as the kernels take it
    (`fla_gdn._between`)."""
    return (later[0] - earlier[0]) + (later[1] - earlier[1])


def pairwise(g, lo):
    """`g_s - g_t` for every pair of a chunk's last-axis positions, `[..., C, C]`, from the halves."""
    return between((g[..., :, None], lo[..., :, None]), (g[..., None, :], lo[..., None, :]))


def reference_o(q, k, v, h, g, g_lo, *, scale: float, chunk: int = 64):
    """`fla_gdn.chunk_fwd_o`'s math in XLA, per chunk, for the tests: the same dtypes and casts."""
    batch, length, key_heads, _ = q.shape
    value_heads = v.shape[2]
    group = value_heads // key_heads
    q, k = (jnp.repeat(x, group, axis=2) for x in (q, k))

    def chunks(x):
        return x.reshape(batch, length // chunk, chunk, *x.shape[2:])

    # Widened before the products: exact for bf16, and XLA:CPU has no bf16 x bf16 -> fp32 dot.
    qc, kc, vc = (chunks(x).astype(jnp.float32) for x in (q, k, v))
    gc, lc = (jnp.moveaxis(chunks(x).astype(jnp.float32), 3, 2) for x in (g, g_lo))
    h = h.astype(jnp.float32)
    out = jnp.einsum('bnchk,bnhkv->bnchv', qc, h, preferred_element_type=jnp.float32)
    out = out * jnp.moveaxis(jnp.exp2(gc), 2, 3)[..., None]
    scores = jnp.einsum('bnshk,bnthk->bnhst', qc, kc, preferred_element_type=jnp.float32)
    scores = scores * jnp.exp2(pairwise(gc, lc))
    index = jnp.arange(chunk)
    scores = jnp.where(index[:, None] >= index[None, :], scores, 0.0)
    out = out * scale + jnp.einsum('bnhst,bnthv->bnshv', scores.astype(v.dtype).astype(jnp.float32), vc,
                                   preferred_element_type=jnp.float32) * scale
    return out.reshape(v.shape).astype(v.dtype)


def reference_w_u(k, v, beta, a, g, *, chunk: int = 64):
    """`recompute_w_u`'s math in XLA, for the tests: the same roundings, products widened (exact)."""
    group = v.shape[2] // k.shape[2]
    k = jnp.repeat(k, group, axis=2)
    batch, length = v.shape[:2]

    def chunks(x):
        return x.reshape(batch, length // chunk, chunk, *x.shape[2:]).astype(jnp.float32)

    inverse = chunks(a)
    scaled_v = (v * beta[..., None]).astype(v.dtype)
    scaled_k = (k * beta[..., None] * jnp.exp2(g.astype(jnp.float32))[..., None]).astype(k.dtype)
    u = jnp.einsum('bnshc,bnchv->bnshv', inverse, chunks(scaled_v), precision=jax.lax.Precision.HIGHEST)
    w = jnp.einsum('bnshc,bnchk->bnshk', inverse, chunks(scaled_k), precision=jax.lax.Precision.HIGHEST)
    return w.reshape(*k.shape[:3], k.shape[-1]).astype(k.dtype), u.reshape(v.shape).astype(v.dtype)


def reference_kkt_solve(k, g, g_lo, beta, *, chunk: int = 64):
    """`kkt_solve`'s math in XLA, for the tests: the inverse in float64 of the same matrix."""
    group = beta.shape[2] // k.shape[2]
    dtype = k.dtype
    k = jnp.repeat(k, group, axis=2).astype(jnp.float32)
    batch, length, heads = beta.shape

    def chunks(x):
        return jnp.moveaxis(x.reshape(batch, length // chunk, chunk, heads, *x.shape[3:]), 3, 2)

    kc, bc = chunks(k), chunks(beta.astype(jnp.float32))
    gc, lc = chunks(g.astype(jnp.float32)), chunks(g_lo.astype(jnp.float32))
    a = jnp.einsum('bnhsk,bnhtk->bnhst', kc, kc, precision=jax.lax.Precision.HIGHEST)
    a = a * jnp.exp2(pairwise(gc, lc))
    a = a * bc[..., :, None]
    a = jnp.tril(a, -1)
    inverse = jnp.linalg.inv(jnp.eye(chunk) + a)
    return jnp.moveaxis(inverse, 2, 3).reshape(batch, length, heads, chunk).astype(dtype)


def reference_h(k, w, u, g, g_lo, h0, *, chunk: int = 64):
    """`chunk_fwd_h`'s math in XLA, a scan over chunks with the same roundings, products widened. A
    `g` per key dimension `[B, T, HV, K]` decays the memory's rows and gates no value (fla's USE_GK)."""
    per_key = g.ndim == 4
    group = u.shape[2] // k.shape[2]
    dtype = k.dtype
    k = jnp.repeat(k, group, axis=2)
    batch, length = u.shape[:2]
    exact = jax.lax.Precision.HIGHEST

    def chunks(x):
        return jnp.moveaxis(x.reshape(batch, length // chunk, chunk, *x.shape[2:]), 1, 0)

    def step(state, inputs):
        keys, weights, values, gate, low = inputs
        entered = state.astype(dtype)
        predicted = jnp.einsum('bchk,bhkv->bchv', weights.astype(jnp.float32),
                               state.astype(weights.dtype).astype(jnp.float32), precision=exact)
        corrected = values.astype(jnp.float32) - predicted
        last = gate[:, -1]
        gated = corrected if per_key else corrected * jnp.exp2(
            between((last[:, None], low[:, -1][:, None]), (gate, low)))[..., None]
        high = gated.astype(dtype)
        update = jnp.einsum('bchk,bchv->bhkv', keys.astype(jnp.float32), high.astype(jnp.float32),
                            precision=exact)
        if jnp.dtype(dtype).itemsize == 2:
            low = (gated - high.astype(jnp.float32)).astype(dtype)
            update += jnp.einsum('bchk,bchv->bhkv', keys.astype(jnp.float32), low.astype(jnp.float32),
                                 precision=exact)
        decay = jnp.exp2(last)[..., None] if per_key else jnp.exp2(last)[..., None, None]
        state = state * decay + update
        return state, (entered, corrected.astype(u.dtype))

    final, (h, corrected) = jax.lax.scan(step, h0.astype(jnp.float32),
                                     (chunks(k), chunks(w), chunks(u), chunks(g.astype(jnp.float32)),
                                      chunks(g_lo.astype(jnp.float32))))
    return jnp.moveaxis(h, 0, 1), jnp.moveaxis(corrected, 0, 1).reshape(u.shape), final


def reference_dhu(q, k, w, g, g_lo, dht, do, dv, *, scale: float, chunk: int = 64):
    """`chunk_bwd_dhu`'s math in XLA, a reverse scan over chunks with the same roundings. A `g` per
    key dimension `[B, T, HV, K]` decays the gradient's rows and gates neither the values' gradient
    nor the queries (fla's USE_GK)."""
    per_key = g.ndim == 4
    group = do.shape[2] // q.shape[2]
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = (jnp.repeat(x, group, axis=2) for x in (q, k))
    batch, length = do.shape[:2]
    exact = jax.lax.Precision.HIGHEST

    def chunks(x):
        return jnp.moveaxis(x.reshape(batch, length // chunk, chunk, *x.shape[2:]), 1, 0)

    def step(grad, inputs):
        queries, keys, weights, gate, low, out_grad, local = inputs
        stored = grad.astype(q_dtype)
        last = gate[:, -1]
        full = jnp.einsum('bchk,bhkv->bchv', keys.astype(jnp.float32),
                          grad.astype(k_dtype).astype(jnp.float32), precision=exact)
        if not per_key:
            full = full * jnp.exp2(between((last[:, None], low[:, -1][:, None]), (gate, low)))[..., None]
        full = full + local.astype(jnp.float32)
        read = queries.astype(jnp.float32) * (1.0 if per_key else jnp.exp2(gate)[..., None])
        grad = (grad * (jnp.exp2(last)[..., None] if per_key else jnp.exp2(last)[..., None, None])
                + jnp.einsum('bchk,bchv->bhkv', read, out_grad.astype(jnp.float32), precision=exact) * scale
                - jnp.einsum('bchk,bchv->bhkv', weights.astype(jnp.float32),
                             full.astype(weights.dtype).astype(jnp.float32), precision=exact))
        return grad, (stored, full.astype(dv.dtype))

    initial, (dh, dv2) = jax.lax.scan(
        step, dht.astype(jnp.float32),
        (chunks(q), chunks(k), chunks(w), chunks(g.astype(jnp.float32)), chunks(g_lo.astype(jnp.float32)),
         chunks(do), chunks(dv)),
        reverse=True)
    return jnp.moveaxis(dh, 0, 1), initial, jnp.moveaxis(dv2, 0, 1).reshape(dv.shape)


def reference_dv_local(q, k, g, g_lo, do, *, scale: float, chunk: int = 64):
    """`chunk_bwd_dv_local`'s math in XLA, per chunk, with the same casts."""
    group = do.shape[2] // q.shape[2]
    q, k = (jnp.repeat(x, group, axis=2).astype(jnp.float32) for x in (q, k))
    batch, length = do.shape[:2]

    def chunks(x):
        return x.reshape(batch, length // chunk, chunk, *x.shape[2:])

    gc, lc = (jnp.moveaxis(chunks(x.astype(jnp.float32)), 3, 2) for x in (g, g_lo))
    exact = jax.lax.Precision.HIGHEST
    scores = jnp.einsum('bnshk,bnthk->bnhst', chunks(k), chunks(q), precision=exact) * scale
    scores = scores * jnp.exp2(-pairwise(gc, lc))
    index = jnp.arange(chunk)
    scores = jnp.where(index[:, None] <= index[None, :], scores, 0.0).astype(do.dtype).astype(jnp.float32)
    dv = jnp.einsum('bnhst,bnthv->bnshv', scores, chunks(do).astype(jnp.float32), precision=exact)
    return dv.reshape(do.shape).astype(do.dtype)


def reference_dqkwg(q, k, v, w, g, g_lo, h, dv, do, dh, *, scale: float, chunk: int = 64):
    """`chunk_bwd_dqkwg`'s math in XLA, per chunk, with the same casts, products widened, and a key
    head's value heads summed in fp32 before `dq` and `dk` round."""
    key_heads = q.shape[2]
    group = v.shape[2] // key_heads
    q_dtype, k_dtype = q.dtype, k.dtype
    batch, length, heads = v.shape[:3]
    exact = jax.lax.Precision.HIGHEST
    f32 = jnp.float32
    q, k = (jnp.repeat(x, group, axis=2) for x in (q, k))

    def chunks(x):
        return x.reshape(batch, length // chunk, chunk, *x.shape[2:])

    qc, kc, vc, doc, dvc = (chunks(x) for x in (q, k, v, do, dv))
    gc, lc = chunks(g.astype(f32)), chunks(g_lo.astype(f32))
    last = gc[:, :, -1]
    dg_last = jnp.sum((h * dh).astype(f32), axis=(-2, -1)) * jnp.exp2(last)
    ds = jnp.einsum('bnshv,bnthv->bnhst', doc.astype(f32), vc.astype(f32), precision=exact)
    dq = jnp.einsum('bnshv,bnhkv->bnshk', doc.astype(f32), h.astype(do.dtype).astype(f32), precision=exact)
    dk = jnp.einsum('bnshv,bnhkv->bnshk', vc.astype(f32), dh.astype(v.dtype).astype(f32), precision=exact)
    dw = -jnp.einsum('bnshv,bnhkv->bnshk', dvc.astype(v.dtype).astype(f32), h.astype(v.dtype).astype(f32),
                     precision=exact)
    dq = dq * jnp.exp2(gc)[..., None] * scale
    dk = dk * jnp.exp2(between((last[:, :, None], lc[:, :, -1][:, :, None]), (gc, lc)))[..., None]
    dg_last = dg_last + jnp.einsum('bnshk,bnshk->bnh', dk, kc.astype(f32))
    gh, lh = jnp.moveaxis(gc, 3, 2), jnp.moveaxis(lc, 3, 2)
    index = jnp.arange(chunk)
    causal = index[:, None] >= index[None, :]
    ds = jnp.where(causal, ds * jnp.exp2(pairwise(gh, lh)), 0.0) * scale
    ds = ds.astype(k_dtype).astype(f32)
    dq = dq + jnp.einsum('bnhst,bnthk->bnshk', ds, kc.astype(f32), precision=exact)
    dk = dk + jnp.einsum('bnhst,bnshk->bnthk', ds, qc.astype(f32), precision=exact)
    dg = jnp.sum(dq * qc.astype(f32), -1) - jnp.sum(dk * kc.astype(f32), -1)
    dg = dg.at[:, :, -1].add(dg_last)
    shape = (batch, length, key_heads, group, q.shape[-1])
    dq, dk = dq.reshape(shape).sum(3).astype(q_dtype), dk.reshape(shape).sum(3).astype(k_dtype)
    return dq, dk, dw.reshape(w.shape).astype(w.dtype), dg.reshape(batch, length, heads)
