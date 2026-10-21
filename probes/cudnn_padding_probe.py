"""Probe: the cuDNN band's gradients on padding rows, against the xla band's and the float64 oracle's zero."""
import jax, jax.numpy as jnp, numpy as np
from dew.nn.local_attention import local_attention

batch, length, heads, kv_heads, width, window = 1, 1024, 8, 2, 64, 256
lengths = [300, 200, 380]
segments = np.concatenate([np.repeat(np.arange(1, 4), lengths), np.zeros(length - sum(lengths), int)])[None]
segment_ids = jnp.asarray(segments, jnp.int32)
real = jnp.asarray(segments[0] != 0)
for seed in range(3):
    keys = jax.random.split(jax.random.fold_in(jax.random.key(0), seed), 4)
    q, k, v = (jax.random.normal(key, (batch, length, c, width), jnp.bfloat16) for key, c in zip(keys[:3], (heads, kv_heads, kv_heads)))
    cot = jax.random.normal(keys[3], (batch, length, heads, width), jnp.float32)
    for impl in ("auto", "xla"):
        def run(q, k, v):
            out = local_attention(q, k, v, window=window, segment_ids=segment_ids, implementation=impl)
            return jnp.sum(out.astype(jnp.float32) * cot * real[None, :, None, None])
        grads = jax.jit(jax.grad(run, argnums=(0, 1, 2)))(q, k, v)
        pad = ~np.asarray(real)
        report = []
        for name, g in zip("qkv", grads):
            g = np.asarray(g, np.float32)
            report.append(f"{name}: pad max {np.abs(g[:, pad]).max():.3e} real max {np.abs(g[:, ~pad]).max():.3e} pad nonzero {int((g[:, pad] != 0).sum())}")
        print(seed, impl, " | ".join(report), flush=True)
