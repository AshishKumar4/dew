"""fla_gdn's test support: each kernel's math in XLA, per chunk, with the kernel's casts, and its declared
tiles as abstract calls for the compile test, with Kimi delta attention's and the short conv's kernels at
their training shapes."""

import dataclasses
import functools
import importlib
from collections.abc import Callable, Iterator
from typing import NamedTuple

import jax
import jax.numpy as jnp
from gdn_cases import CASES

from dew.nn.kernels import chunked_launch, fla_gdn, fla_kda, generation, recurrence, short_conv, tuning

COMPILED_SHAPES = {name: CASES[name] for name in ("0.8b", "9b", "27b", "wide keys")}
"""The shapes every declared config compiles at (`gdn_cases`): the training shapes and the widest keys."""


class KernelCase(NamedTuple):
    """One declared kernel call for the compile test (tests/kernel_compile.py): the kernel with its
    config, and abstract operands at a training shape."""

    name: str
    call: Callable[[], tuple[Callable, tuple[jax.ShapeDtypeStruct, ...]]]


def _pinned(launch, tiles, *args, **kwargs):
    """`launch`'s kernel on `args`, pinned to `tiles` (`tuning.pinned`): the function its module
    names as the launch (`NAME`)."""
    with tuning.pinned({type(launch): tiles}):
        return getattr(importlib.import_module(type(launch).__module__), launch.NAME)(*args, **kwargs)


def _case_call(launch, tiles, shape, dtype, *, reverse=False):
    kernel = launch.NAME
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
    entered = jax.ShapeDtypeStruct((batch * (length // chunk), value_heads, width, columns), narrow)
    state = jax.ShapeDtypeStruct((batch, value_heads, width, columns), jnp.float32)
    operands = {
        "cumulated": (gated,),
        "kkt_solve": (keyed, gated, beta),
        "recompute_w_u": (keyed, valued, beta, inverse, gated),
        "chunk_fwd_h": (keyed, weighted, valued, gated, state),
        "chunk_fwd_o": (keyed, keyed, valued, entered, gated),
        "chunk_bwd_dv_local": (keyed, keyed, gated, valued),
        "chunk_bwd_dhu": (keyed, keyed, weighted, gated, state, valued, valued),
        "chunk_bwd_dqkwg": (keyed, keyed, valued, weighted, gated, entered, valued, valued, entered),
        "prepare_wy_repr_bwd": (keyed, valued, beta, gated, inverse, weighted, valued, keyed),
    }
    kwargs = {"chunk": chunk, "interpret": False}
    if kernel == "cumulated":
        kwargs.update(scale=fla_gdn.RCP_LN2, reverse=reverse)
    elif kernel in ("chunk_fwd_o", "chunk_bwd_dv_local", "chunk_bwd_dhu", "chunk_bwd_dqkwg"):
        kwargs["scale"] = width ** -0.5
    return functools.partial(_pinned, launch, tiles, **kwargs), operands[kernel]


def kernel_cases(target: str, dtypes: tuple[str, ...] = ("bfloat16", "float32")) -> Iterator[KernelCase]:
    """Each kernel pinned to the tiles `target` runs it with (its launch's `PICKS`, the reverse
    recurrence's `WIDE_KEYS_PICK` past `WIDEST` keys) for each of `dtypes` at each shape, and the reverse
    cumsum, for the compile test."""
    with generation.assumed(target):
        wide = chunked_launch._wide()
    for dtype in dtypes:
        launches = [*vars(fla_gdn).values(), *vars(recurrence).values()]
        for launch in (x for x in launches if isinstance(x, chunked_launch._Launch)):
            for shape_name, shape in COMPILED_SHAPES.items():
                if shape["width"] > fla_gdn.WIDEST and dtype == "float32":
                    continue
                kernel, tiles = launch.NAME, launch.PICKS[dtype, wide]
                if shape["width"] > fla_gdn.WIDEST and isinstance(launch, fla_gdn.ChunkBwdDhu):
                    tiles = launch.WIDE_KEYS_PICK
                config = {field: value for field, value in dataclasses.asdict(tiles).items() if value}
                flags = "-".join(f"{key}{value}" for key, value in sorted(config.items()))
                for reverse in ((False, True) if kernel == "cumulated" else (False,)):
                    name = f"{target}/{kernel}/{dtype}/{flags}/{shape_name}" + "/reverse" * reverse
                    yield KernelCase(name, functools.partial(_case_call, launch, tiles, shape, dtype,
                                                             reverse=reverse))


def head_chunks(x, chunk: int):
    """`x` `[B, T, H, ...]` as `[B, T / chunk, H, chunk, ...]`: each head's tokens in chunks."""
    batch, length, heads = x.shape[:3]
    return jnp.moveaxis(x.reshape(batch, length // chunk, chunk, heads, *x.shape[3:]), 3, 2)


def scanned_chunks(x, chunk: int):
    """`x` `[B, T, ...]` as `[T / chunk, B, chunk, ...]`: the chunks in the order a scan takes them."""
    batch, length = x.shape[:2]
    return jnp.moveaxis(x.reshape(batch, length // chunk, chunk, *x.shape[2:]), 1, 0)


def pairwise(g):
    """`g_s - g_t` for every pair of a chunk's last-axis positions, `[..., C, C]`."""
    return g[..., :, None] - g[..., None, :]


def _slotted(passing, rows):
    """Memories passing between a row's chunks `[B, C - 1, ...]` and a row's own `[B, ...]` at their
    slots (`chunked_launch.Sequences`): the passing ones row by row, then the rows'."""
    return jnp.concatenate([passing.reshape(-1, *passing.shape[2:]), rows])


def entered_states(per_chunk):
    """Per-chunk memories `[B, C, HV, K, V]`, each the one a chunk enters, at the slots the port
    stores them in: each row's chunks' past its first, then each row's initial memory."""
    return _slotted(per_chunk[:, 1:], per_chunk[:, 0])


def left_grads(per_chunk):
    """Per-chunk memory gradients `[B, C, HV, K, V]`, each of the memory a chunk leaves, at the slots
    the port stores them in: each row's chunks' but its last, then each row's final memory's."""
    return _slotted(per_chunk[:, :-1], per_chunk[:, -1])


def reference_o(q, k, v, h, g, *, scale: float, chunk: int = 64):
    """`fla_gdn.chunk_fwd_o`'s math in XLA, per chunk, for the tests: the same dtypes and casts."""
    batch, length, key_heads, _ = q.shape
    value_heads = v.shape[2]
    group = value_heads // key_heads
    q, k = (jnp.repeat(x, group, axis=2) for x in (q, k))

    def chunks(x):
        return x.reshape(batch, length // chunk, chunk, *x.shape[2:])

    # Widened before the products: exact for bf16, and XLA:CPU has no bf16 x bf16 -> fp32 dot.
    qc, kc, vc = (chunks(x).astype(jnp.float32) for x in (q, k, v))
    gc = jnp.moveaxis(chunks(g).astype(jnp.float32), 3, 2)
    h = h.astype(jnp.float32)
    out = jnp.einsum('bnchk,bnhkv->bnchv', qc, h, preferred_element_type=jnp.float32)
    out = out * jnp.moveaxis(jnp.exp2(gc), 2, 3)[..., None]
    scores = jnp.einsum('bnshk,bnthk->bnhst', qc, kc, preferred_element_type=jnp.float32)
    scores = scores * jnp.exp2(pairwise(gc))
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


def reference_kkt_solve(k, g, beta, *, chunk: int = 64):
    """`kkt_solve`'s math in XLA, for the tests: the inverse in float64 of the same matrix."""
    group = beta.shape[2] // k.shape[2]
    dtype = k.dtype
    k = jnp.repeat(k, group, axis=2).astype(jnp.float32)
    batch, length, heads = beta.shape

    kc, bc = head_chunks(k, chunk), head_chunks(beta.astype(jnp.float32), chunk)
    gc = head_chunks(g.astype(jnp.float32), chunk)
    a = jnp.einsum('bnhsk,bnhtk->bnhst', kc, kc, precision=jax.lax.Precision.HIGHEST)
    a = a * jnp.exp2(pairwise(gc))
    a = a * bc[..., :, None]
    a = jnp.tril(a, -1)
    inverse = jnp.linalg.inv(jnp.eye(chunk) + a)
    return jnp.moveaxis(inverse, 2, 3).reshape(batch, length, heads, chunk).astype(dtype)


def reference_h(k, w, u, g, h0, *, chunk: int = 64):
    """`chunk_fwd_h`'s math in XLA, a scan over chunks with the same roundings, products widened. A
    `g` per key dimension `[B, T, HV, K]` decays the memory's rows and gates no value (fla's USE_GK)."""
    per_key = g.ndim == 4
    group = u.shape[2] // k.shape[2]
    dtype = k.dtype
    k = jnp.repeat(k, group, axis=2)
    exact = jax.lax.Precision.HIGHEST

    def step(state, inputs):
        keys, weights, values, gate = inputs
        entered = state.astype(dtype)
        predicted = jnp.einsum('bchk,bhkv->bchv', weights.astype(jnp.float32),
                               state.astype(weights.dtype).astype(jnp.float32), precision=exact)
        corrected = values.astype(jnp.float32) - predicted
        last = gate[:, -1]
        gated = corrected if per_key else corrected * jnp.exp2(last[:, None] - gate)[..., None]
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

    chunked = tuple(scanned_chunks(x, chunk) for x in (k, w, u, g.astype(jnp.float32)))
    final, (h, corrected) = jax.lax.scan(step, h0.astype(jnp.float32), chunked)
    return jnp.moveaxis(h, 0, 1), jnp.moveaxis(corrected, 0, 1).reshape(u.shape), final


def reference_dhu(q, k, w, g, dht, do, dv, *, scale: float, chunk: int = 64):
    """`chunk_bwd_dhu`'s math in XLA, a reverse scan over chunks with the same roundings. A `g` per
    key dimension `[B, T, HV, K]` decays the gradient's rows and gates neither the values' gradient
    nor the queries (fla's USE_GK)."""
    per_key = g.ndim == 4
    group = do.shape[2] // q.shape[2]
    q_dtype, k_dtype = q.dtype, k.dtype
    q, k = (jnp.repeat(x, group, axis=2) for x in (q, k))
    exact = jax.lax.Precision.HIGHEST

    def step(grad, inputs):
        queries, keys, weights, gate, out_grad, local = inputs
        stored = grad.astype(q_dtype)
        last = gate[:, -1]
        full = jnp.einsum('bchk,bhkv->bchv', keys.astype(jnp.float32),
                          grad.astype(k_dtype).astype(jnp.float32), precision=exact)
        if not per_key:
            full = full * jnp.exp2(last[:, None] - gate)[..., None]
        full = full + local.astype(jnp.float32)
        read = queries.astype(jnp.float32) * (1.0 if per_key else jnp.exp2(gate)[..., None])
        grad = (grad * (jnp.exp2(last)[..., None] if per_key else jnp.exp2(last)[..., None, None])
                + jnp.einsum('bchk,bchv->bhkv', read, out_grad.astype(jnp.float32), precision=exact) * scale
                - jnp.einsum('bchk,bchv->bhkv', weights.astype(jnp.float32),
                             full.astype(weights.dtype).astype(jnp.float32), precision=exact))
        return grad, (stored, full.astype(dv.dtype))

    initial, (dh, dv2) = jax.lax.scan(
        step, dht.astype(jnp.float32),
        tuple(scanned_chunks(x, chunk) for x in (q, k, w, g.astype(jnp.float32), do, dv)),
        reverse=True)
    return jnp.moveaxis(dh, 0, 1), initial, jnp.moveaxis(dv2, 0, 1).reshape(dv.shape)


def reference_dv_local(q, k, g, do, *, scale: float, chunk: int = 64):
    """`chunk_bwd_dv_local`'s math in XLA, per chunk, with the same casts."""
    group = do.shape[2] // q.shape[2]
    q, k = (jnp.repeat(x, group, axis=2).astype(jnp.float32) for x in (q, k))
    batch, length = do.shape[:2]

    def chunks(x):
        return x.reshape(batch, length // chunk, chunk, *x.shape[2:])

    gc = jnp.moveaxis(chunks(g.astype(jnp.float32)), 3, 2)
    exact = jax.lax.Precision.HIGHEST
    scores = jnp.einsum('bnshk,bnthk->bnhst', chunks(k), chunks(q), precision=exact) * scale
    scores = scores * jnp.exp2(-pairwise(gc))
    index = jnp.arange(chunk)
    scores = jnp.where(index[:, None] <= index[None, :], scores, 0.0).astype(do.dtype).astype(jnp.float32)
    dv = jnp.einsum('bnhst,bnthv->bnshv', scores, chunks(do).astype(jnp.float32), precision=exact)
    return dv.reshape(do.shape).astype(do.dtype)


def reference_dqkwg(q, k, v, w, g, h, dv, do, dh, *, scale: float, chunk: int = 64):
    """`chunk_bwd_dqkwg`'s math in XLA, per chunk, with the same casts, products widened, and a key
    head's value heads summed in fp32 before `dq` and `dk` round. The decay's gradient is each
    token's, its terms summed over the ranges through it (`fla_gdn._summed_back`)."""
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
    gc = chunks(g.astype(f32))
    last = gc[:, :, -1]
    dg_last = jnp.sum((h * dh).astype(f32), axis=(-2, -1)) * jnp.exp2(last)
    ds = jnp.einsum('bnshv,bnthv->bnhst', doc.astype(f32), vc.astype(f32), precision=exact)
    dq = jnp.einsum('bnshv,bnhkv->bnshk', doc.astype(f32), h.astype(do.dtype).astype(f32), precision=exact)
    dk = jnp.einsum('bnshv,bnhkv->bnshk', vc.astype(f32), dh.astype(v.dtype).astype(f32), precision=exact)
    dw = -jnp.einsum('bnshv,bnhkv->bnshk', dvc.astype(v.dtype).astype(f32), h.astype(v.dtype).astype(f32),
                     precision=exact)
    dq = dq * jnp.exp2(gc)[..., None] * scale
    dk = dk * jnp.exp2(last[:, :, None] - gc)[..., None]
    gh = jnp.moveaxis(gc, 3, 2)
    index = jnp.arange(chunk)
    causal = index[:, None] >= index[None, :]
    strict = index[:, None] > index[None, :]
    ds = jnp.where(causal, ds * jnp.exp2(pairwise(gh)), 0.0) * scale
    scores = jnp.einsum('bnshk,bnthk->bnhst', qc.astype(f32), kc.astype(f32), precision=exact)
    leaving = jnp.moveaxis(jnp.einsum('bnshk,bnshk->bnsh', dk, kc.astype(f32)), 3, 2)
    pairs = jnp.where(strict, ds * scores, 0.0)
    pairs = pairs.at[..., -1, :].add(jnp.where(index < chunk - 1, leaving, 0.0))
    after = jnp.moveaxis(jnp.einsum('bnshk,bnshk->bnsh', dq, qc.astype(f32)), 3, 2)
    after = after.at[..., -1].add(dg_last)
    through = jnp.cumsum(pairs, axis=-1) - pairs
    dg = jnp.moveaxis(jnp.sum(jnp.where(causal, through + after[..., :, None], 0.0), axis=-2), 2, 3)
    ds = ds.astype(k_dtype).astype(f32)
    dq = dq + jnp.einsum('bnhst,bnthk->bnshk', ds, kc.astype(f32), precision=exact)
    dk = dk + jnp.einsum('bnhst,bnshk->bnthk', ds, qc.astype(f32), precision=exact)
    shape = (batch, length, key_heads, group, q.shape[-1])
    dq, dk = dq.reshape(shape).sum(3).astype(q_dtype), dk.reshape(shape).sum(3).astype(k_dtype)
    return dq, dk, dw.reshape(w.shape).astype(w.dtype), dg.reshape(batch, length, heads)


KDA_TRAINING = (1, 4096, 32, 128)
"""Batch, length, heads, head width: Kimi Linear's KDA layer (32 heads of 128) at a 4096-token row."""


def kda_training_cases() -> Iterator[KernelCase]:
    """The port's forward and backward in bf16 through the rule's VJP: at the training shape, and with
    keys of 256, the widest 16-bit keys the port takes (`fla_gdn.WIDEST_KEYS`)."""
    batch, length, heads, width = KDA_TRAINING

    def case(name, width, columns, heads):
        def call():
            keys = jax.ShapeDtypeStruct((batch, length, heads, width), jnp.bfloat16)
            values = jax.ShapeDtypeStruct((batch, length, heads, columns), jnp.bfloat16)
            decay = jax.ShapeDtypeStruct((batch, length, heads, width), jnp.float32)
            beta = jax.ShapeDtypeStruct((batch, length, heads), jnp.bfloat16)
            state = jax.ShapeDtypeStruct((batch, heads, width, columns), jnp.float32)

            def grads(q, k, v, g, beta, h0):
                def loss(*args):
                    out, final = fla_kda.chunk_kda(*args, None, width ** -0.5, 64, interpret=False)
                    return jnp.sum(out.astype(jnp.float32)) + jnp.sum(final)
                return jax.grad(loss, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, g, beta, h0)

            return grads, (keys, keys, values, decay, beta, state)

        return KernelCase(name, call)

    yield case("kda", width, width, heads)
    yield case("kda/keys of 256", fla_gdn.WIDEST_KEYS, width, 4)


CONV_SHAPES = {
    # Qwen3.5-0.8B: 16 key and 16 value heads of 128.
    "0.8b": ((8, 3072), ((2048, 128), (2048, 128), (2048, None))),
    # Qwen3.5-9B: 16 key heads, 32 value heads.
    "9b": ((4, 3072), ((2048, 128), (2048, 128), (4096, None))),
}


def _conv_case(name, shape, parts, out, bias, valid, packed) -> KernelCase:
    """Both kernels through the conv's VJP over bf16 rows `shape` into `parts` in `out`, each launch at
    its config."""
    def call():
        x = jax.ShapeDtypeStruct((*shape, sum(width for width, _ in parts)), jnp.bfloat16)
        weights = jax.ShapeDtypeStruct((x.shape[-1], 4), jnp.float32)
        shift = jax.ShapeDtypeStruct((x.shape[-1],), jnp.float32) if bias else None
        kept = jax.ShapeDtypeStruct(shape, jnp.int8) if valid else None
        segments = jax.ShapeDtypeStruct(shape, jnp.int32) if packed else None

        def grads(x, weights, shift, kept, segments):
            def loss(x, weights, shift):
                out_ = short_conv.conv(x, weights, shift, kept, segments, parts, activation=True,
                                       eps=1e-6, dtype=jnp.dtype(out), interpret=False)
                return sum(jnp.sum(piece.astype(jnp.float32)) for piece in out_)
            return jax.grad(loss, argnums=(0, 1, 2))(x, weights, shift)
        return grads, (x, weights, shift, kept, segments)
    return KernelCase(name, call)


def conv_training_cases() -> Iterator[KernelCase]:
    """Both kernels, through the conv's VJP, at each training shape: a delta net's bf16 conv into bf16
    pieces over right-padded rows and over packed ones, and Mamba's fp32 output with a bias and no norm
    over whole rows and packed ones."""
    for model, (shape, parts) in sorted(CONV_SHAPES.items()):
        yield _conv_case(f"{model}/delta", shape, parts, jnp.bfloat16, bias=False, valid=True, packed=False)
        yield _conv_case(f"{model}/packed", shape, parts, jnp.bfloat16, bias=False, valid=False, packed=True)
    for packed in (False, True):
        yield _conv_case("mamba/packed" if packed else "mamba", (8, 3072), ((3072, None),), jnp.float32,
                         bias=True, valid=False, packed=packed)
