"""Probe: the gradient rows whose float64 value is exactly zero (a query that reads one key), cuDNN's band against the xla band's."""
import jax, jax.numpy as jnp, numpy as np
from dew.nn.local_attention import local_attention

batch, length, heads, kv_heads, width, window = 1, 1024, 8, 2, 64, 256
lengths = [300, 200, 380]
segments = np.concatenate([np.repeat(np.arange(1, 4), lengths), np.zeros(length - sum(lengths), int)])[None]
segment_ids = jnp.asarray(segments, jnp.int32)
real = jnp.asarray(segments[0] != 0)
starts = [0, 300, 500]
for seed in range(4):
    keys = jax.random.split(jax.random.fold_in(jax.random.key(0), seed), 4)
    q, k, v = (jax.random.normal(key, (batch, length, c, width), jnp.bfloat16) for key, c in zip(keys[:3], (heads, kv_heads, kv_heads)))
    cot = jax.random.normal(keys[3], (batch, length, heads, width), jnp.float32)
    found = {}
    for impl in ("auto", "xla"):
        def run(q, k, v):
            out = local_attention(q, k, v, window=window, segment_ids=segment_ids, implementation=impl)
            return jnp.sum(out.astype(jnp.float32) * cot * real[None, :, None, None])
        found[impl] = [np.asarray(g, np.float32) for g in jax.jit(jax.grad(run, argnums=(0, 1, 2)))(q, k, v)]
    dq_auto, dq_xla = found["auto"][0], found["xla"][0]
    print(seed, "first-token dq max: auto", f"{np.abs(dq_auto[:, starts]).max():.3e}", "xla", f"{np.abs(dq_xla[:, starts]).max():.3e}",
          "| typical real dq rms", f"{np.sqrt(np.mean(dq_xla[:, np.asarray(real)] ** 2)):.3e}",
          "| other zero rows auto:", int(((np.abs(dq_xla).max(-1) == 0) & (np.abs(dq_auto).max(-1) != 0)).sum()), flush=True)
