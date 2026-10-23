"""Measurement script (never lands): dK = dS^T Q of the bf16 small-weight family under candidate forms of
dS for the product, against float64 and the reference's (jax's xla attention, fp32 dS): each form's worst
row over the seeds, |dk - truth| over the row's largest |truth|, beside the reference's at that row."""
import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
import test_attention_stability as suite
from reference_error import with_gradients

family = suite.FAMILIES["a small weight on a large value, bfloat16"]
draw = suite._cast(family, family.draw)
reference = jax.jit(with_gradients(suite._reference(family, anchored=False)))
first, count = int(sys.argv[1]), int(sys.argv[2])


def bf16(x):
    return np.asarray(jnp.asarray(x, jnp.float32).astype(jnp.bfloat16), np.float64)


def fp16_scaled(x):
    # Each query's row of dS scaled by a power of two to fp16's range, rounded, scaled back.
    top = np.abs(x).max(axis=-1, keepdims=True)
    exponent = np.where(top > 0, np.floor(np.log2(np.where(top > 0, top, 1.0))), 0.0)
    scaled = x * 2.0 ** (14.0 - exponent)
    return np.asarray(jnp.asarray(scaled, jnp.float32).astype(jnp.float16), np.float64) * 2.0 ** (exponent - 14.0)


def tf32(x):
    bits = np.asarray(x, np.float32).view(np.uint32)
    rounded = ((bits.astype(np.uint64) + 0x1000) & 0xFFFFE000).astype(np.uint32)
    return rounded.view(np.float32).astype(np.float64)


def split(x, parts):
    out, rest = np.zeros_like(x), np.asarray(np.asarray(x, np.float32), np.float64)
    for _ in range(parts):
        piece = bf16(rest)
        out, rest = out + piece, rest - piece
    return out


def fp16_of(x, axis):
    # `x` scaled by a power of two along `axis` to fp16's range, rounded, scaled back.
    top = np.abs(x).max(axis=axis, keepdims=True)
    exponent = np.where(top > 0, np.floor(np.log2(np.where(top > 0, top, 1.0))), 0.0)
    scaled = np.asarray(jnp.asarray(x * 2.0 ** (14.0 - exponent), jnp.float32).astype(jnp.float16), np.float64)
    return scaled * 2.0 ** (exponent - 14.0)


def hi_bf16_lo_fp16(ds, q64):
    hi = bf16(ds)
    lo = fp16_of(np.asarray(np.asarray(ds, np.float32), np.float64) - hi, -1)
    return np.einsum("bhqk,bqhd->bkhd", hi, q64) + np.einsum("bhqk,bqhd->bkhd", lo, fp16_of(q64, (1, 3)))


FORMS = {"bf16": bf16, "fp16 row-scaled": fp16_scaled, "tf32": tf32, "bf16 hi+lo": lambda x: split(x, 2),
         "bf16x3": lambda x: split(x, 3), "fp32": lambda x: np.asarray(np.asarray(x, np.float32), np.float64)}
worst = {name: (0.0, 0.0) for name in [*FORMS, "bf16 hi + fp16 lo", "reference"]}
for seed in range(first, first + count):
    q, k, v, extra, dout = draw(seed)
    q64, k64, v64, d64 = (np.asarray(x, np.float64) for x in (q, k, v, dout))
    keep = np.asarray(family.keep)
    s = np.where(keep, np.einsum("bqhd,bkhd->bhqk", q64, k64) / 4.0, -np.inf)
    p = np.exp(s - s.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    dp = np.einsum("bqhd,bkhd->bhqk", d64, v64)
    big_d = np.einsum("bqhd,bqhd->bhq", d64, np.einsum("bhqk,bkhd->bqhd", p, v64))
    ds = p * (dp - big_d[..., None]) / 4.0
    truth = np.einsum("bhqk,bqhd->bkhd", ds, q64)
    size = np.abs(truth).max(axis=(1, 2, 3)) + 1e-300
    ref = np.asarray(reference(q, k, v, extra, dout)[2], np.float64)
    ref_err = np.abs(ref - truth).max(axis=(1, 2, 3)) / size
    for name, form in [*FORMS.items(), ("bf16 hi + fp16 lo", "split"), ("reference", None)]:
        if form is None:
            found = ref
        elif form == "split":
            found = hi_bf16_lo_fp16(ds, q64)
        else:
            found = np.einsum("bhqk,bqhd->bkhd", form(ds), q64)
        err = np.abs(found - truth).max(axis=(1, 2, 3)) / size
        rms = float(np.sqrt(np.mean(((found - truth) / size[:, None, None, None]) ** 2)))
        worst[name] = (max(worst[name][0], float(err.max())), worst[name][1] + rms ** 2)
print(json.dumps({"seeds": [first, count], "worst_row_and_rms2": worst}))
