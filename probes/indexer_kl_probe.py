"""Probe: where Dew's causal dense indexer KL (its mean) leaves float64 more than MaxText's arithmetic does."""
import sys
sys.path.insert(0, "tools")
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from indexer_reference import PUBLISHED, published_inputs, published_keep, EPS
from dew.nn.mla import indexer_kl

keep = jnp.asarray(published_keep("causal"))
scale = PUBLISHED["head_dim"] ** -0.5
mask_value = -0.7 * float(np.finfo(np.float32).max)

def maxtext_like(scores, query, key, dtype, heads_summed="sum"):
    """The published arithmetic: additive mask, softmaxes, heads summed at once, EPS as published."""
    attention_mask = jnp.where(keep, 0.0, mask_value).astype(dtype)
    q = jnp.asarray(query, dtype) * dtype(scale)
    k = jnp.asarray(key, dtype)
    s = jnp.asarray(scores, dtype) + attention_mask
    indexer = jax.nn.softmax(s, axis=-1)
    logits = jnp.einsum("bthd,bshd->bhts", q, k) + attention_mask[:, None]
    target = jnp.sum(jax.nn.softmax(logits, axis=-1), axis=1)
    target = target / (jnp.sum(target, axis=-1, keepdims=True) + EPS)
    per_query = jnp.sum(target * (jnp.log(target + EPS) - jnp.log(indexer + EPS)), axis=-1)
    return per_query

dew32 = jax.jit(lambda s, q, k: indexer_kl(s, q, k, keep, scale))
ref32 = jax.jit(lambda s, q, k: maxtext_like(s, q, k, jnp.float32))
errs = {"dew": [], "maxtext-like": []}
means = {"dew": [], "maxtext-like": []}
for seed in range(400):
    x = published_inputs(seed)
    truth = np.asarray(indexer_kl(*(jnp.asarray(x[n], jnp.float64) for n in ("scores", "query", "key")), keep, scale))
    for name, fn in (("dew", dew32), ("maxtext-like", ref32)):
        got = np.asarray(fn(x["scores"], x["query"], x["key"]), np.float64)
        errs[name].append(got - truth)
        means[name].append(float(np.mean(np.asarray(fn(x["scores"], x["query"], x["key"]), np.float32)) - np.mean(truth)))
for name in errs:
    e = np.asarray(errs[name])
    print(name, "per-query rms", f"{np.sqrt(np.mean(e**2)):.3e}", "mean-loss rms", f"{np.sqrt(np.mean(np.square(means[name]))):.3e}",
          "bias of per-query error", f"{np.mean(e):.3e}")
    by_position = np.sqrt(np.mean(e ** 2, axis=(0, 1)))
    print("  by query position:", " ".join(f"{v:.1e}" for v in by_position[:8]), "...", f"{by_position[-1]:.1e}")
