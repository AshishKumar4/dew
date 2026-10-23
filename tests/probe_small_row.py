"""Measurement script (never lands): one bad row of the bf16 small-weight family, its terms."""
import jax
import jax.numpy as jnp
import numpy as np
import test_attention_stability as suite
from reference_error import with_gradients

family = suite.FAMILIES["a small weight on a large value, bfloat16"]
q, k, v, extra, dout = suite._cast(family, family.draw)(2)
row = 122
q64, k64, v64, d64 = (np.asarray(x, np.float64)[row:row + 1] for x in (q, k, v, dout))
keep = np.asarray(family.keep)[row:row + 1]
s = np.einsum("bqhd,bkhd->bhqk", q64, k64) / 4.0
s = np.where(keep, s, -np.inf)
p = np.exp(s - s.max(-1, keepdims=True))
p /= p.sum(-1, keepdims=True)
dp = np.einsum("bqhd,bkhd->bhqk", d64, v64)
o = np.einsum("bhqk,bkhd->bqhd", p, v64)
D = np.einsum("bqhd,bqhd->bhq", d64, o)
ds = p * (dp - D[..., None])
print("size", q64[0, :, 0, 0])
print("p1", p[0, 0, :, 1])
print("dP1", dp[0, 0, :, 1], "dP0", dp[0, 0, :, 0])
print("D", D[0, 0])
print("dS0", ds[0, 0, :, 0], "dS1", ds[0, 0, :, 1])
dk_true = np.einsum("bhqk,bqhd->bkhd", ds, q64) / 4.0
narrowed = np.asarray(jnp.asarray(ds / 4.0, jnp.float32).astype(jnp.bfloat16), np.float64)
dk_narrow = np.einsum("bhqk,bqhd->bkhd", narrowed, q64)
print("dk true", dk_true[0, :2, 0, 0], "dk with dS narrowed", dk_narrow[0, :2, 0, 0])
mainloop = suite._mainloops(family, suite.FORMS["default"])["portable"]
ours = jax.jit(with_gradients(mainloop))(q, k, v, extra, dout)
print("dk ours", np.asarray(ours[2], np.float64)[row, :2, 0, 0])
print("dq ours", np.asarray(ours[1], np.float64)[row, :, 0, 0])
dq_true = np.einsum("bhqk,bkhd->bqhd", ds, k64) / 4.0
print("dq true", dq_true[0, :, 0, 0])
