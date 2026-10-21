"""Probe 2: which part of Dew's indexer KL carries its per-query bias."""
import sys
sys.path.insert(0, "tools")
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from indexer_reference import PUBLISHED, published_inputs, published_keep, EPS
from dew.nn.mla import indexer_kl

keep = jnp.asarray(published_keep("causal"))
scale = PUBLISHED["head_dim"] ** -0.5
mask_value = -0.7 * float(np.finfo(np.float32).max)

def variant(scores, query, key, *, additive=False, scan=True, log_ratio=False):
    f = jnp.float32
    masked = jnp.finfo(f).min
    q = query.astype(f) * scale
    k = key.astype(f)
    def mask(x):
        return x + jnp.where(keep, 0.0, mask_value).astype(f) if additive else jnp.where(keep, x, masked)
    if scan:
        def head(summed, projections):
            qh, kh = projections
            return summed + jax.nn.softmax(mask(jnp.einsum('bsd,btd->bst', qh, kh)), axis=-1), None
        summed, _ = jax.lax.scan(head, jnp.zeros(scores.shape, f), (jnp.moveaxis(q, 2, 0), jnp.moveaxis(k, 2, 0)))
    else:
        logits = jnp.einsum("bthd,bshd->bhts", q, k)
        summed = jnp.sum(jax.nn.softmax(mask(logits) if not additive else logits + jnp.where(keep, 0.0, mask_value)[:, None].astype(f), axis=-1), axis=1)
    target = summed / (jnp.sum(summed, axis=-1, keepdims=True) + EPS)
    indexer = jax.nn.softmax(mask(scores.astype(f)), axis=-1)
    if log_ratio:
        return jnp.sum(target * jnp.log((target + EPS) / (indexer + EPS)), axis=-1)
    return jnp.sum(target * (jnp.log(target + EPS) - jnp.log(indexer + EPS)), axis=-1)

cases = {"dew": dict(), "additive": dict(additive=True), "no scan": dict(scan=False),
         "additive no scan": dict(additive=True, scan=False)}
fns = {name: jax.jit(lambda s, q, k, kw=kw: variant(s, q, k, **kw)) for name, kw in cases.items()}
fns["indexer_kl"] = jax.jit(lambda s, q, k: indexer_kl(s, q, k, keep, scale))
errs = {name: [] for name in fns}
for seed in range(300):
    x = published_inputs(seed)
    truth = np.asarray(indexer_kl(*(jnp.asarray(x[n], jnp.float64) for n in ("scores", "query", "key")), keep, scale))
    for name, fn in fns.items():
        errs[name].append(np.asarray(fn(x["scores"], x["query"], x["key"]), np.float64) - truth)
for name, e in errs.items():
    e = np.asarray(e)
    print(f"{name:18s} per-query rms {np.sqrt(np.mean(e**2)):.3e} bias {np.mean(e):.3e} mean-loss rms {np.sqrt(np.mean(np.mean(e, axis=(1, 2))**2)):.3e}")
