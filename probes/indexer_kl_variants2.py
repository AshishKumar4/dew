"""Probe 3: the KL's per-head logits against its head sum, apart."""
import sys
sys.path.insert(0, "tools")
import jax, jax.numpy as jnp, numpy as np
jax.config.update("jax_enable_x64", True)
from indexer_reference import PUBLISHED, published_inputs, published_keep, EPS
from dew.nn.mla import indexer_kl

keep = jnp.asarray(published_keep("causal"))
scale = PUBLISHED["head_dim"] ** -0.5
f = jnp.float32
masked = jnp.finfo(f).min

def kl(scores, target_sum):
    target = target_sum / (jnp.sum(target_sum, axis=-1, keepdims=True) + EPS)
    indexer = jax.nn.softmax(jnp.where(keep, scores.astype(f), masked), axis=-1)
    return jnp.sum(target * (jnp.log(target + EPS) - jnp.log(indexer + EPS)), axis=-1)

def per_head_logits(q, k):
    return jnp.stack([jnp.einsum('bsd,btd->bst', q[:, :, h], k[:, :, h]) for h in range(q.shape[2])], 1)

def all_head_logits(q, k):
    return jnp.einsum("bthd,bshd->bhts", q, k)

def summed_at_once(probabilities):
    return jnp.sum(probabilities, axis=1)

def summed_in_turn(probabilities):
    total = jnp.zeros(probabilities.shape[:1] + probabilities.shape[2:], f)
    for h in range(probabilities.shape[1]):
        total = total + probabilities[:, h]
    return total

def variant(logits_of, sum_of):
    def run(scores, query, key):
        q, k = query.astype(f) * scale, key.astype(f)
        probabilities = jax.nn.softmax(jnp.where(keep[:, None], logits_of(q, k), masked), axis=-1)
        return kl(scores, sum_of(probabilities))
    return jax.jit(run)

fns = {"per-head logits, in turn": variant(per_head_logits, summed_in_turn),
       "per-head logits, at once": variant(per_head_logits, summed_at_once),
       "all-head logits, in turn": variant(all_head_logits, summed_in_turn),
       "all-head logits, at once": variant(all_head_logits, summed_at_once),
       "indexer_kl": jax.jit(lambda s, q, k: indexer_kl(s, q, k, keep, scale))}
errs = {name: [] for name in fns}
for seed in range(300):
    x = published_inputs(seed)
    truth = np.asarray(indexer_kl(*(jnp.asarray(x[n], jnp.float64) for n in ("scores", "query", "key")), keep, scale))
    for name, fn in fns.items():
        errs[name].append(np.asarray(fn(x["scores"], x["query"], x["key"]), np.float64) - truth)
for name, e in errs.items():
    e = np.asarray(e)
    print(f"{name:26s} per-query rms {np.sqrt(np.mean(e**2)):.3e} bias {np.mean(e):.3e} mean-loss rms {np.sqrt(np.mean(np.mean(e, axis=(1, 2))**2)):.3e}")
