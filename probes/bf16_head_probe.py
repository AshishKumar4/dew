"""Probe: which hidden states the bf16 tied head multiplies on this backend."""
import sys
sys.path.insert(0, "tests")
import jax, jax.numpy as jnp, numpy as np
from model_support import TINY_DECODER
from dew.nn.backbones.causal_transformer import CausalTransformer

model = CausalTransformer(vocab_size=37, **TINY_DECODER, dtype=jnp.bfloat16)
for seed in range(3):
    ids = jax.random.randint(jax.random.key(seed), (2, 12), 0, 37)
    params = model.init(jax.random.key(100 + seed), ids)
    logits = np.asarray(jax.jit(model.apply)(params, ids), np.float64)
    hidden = jax.jit(lambda p, i: model.apply(p, i, method=CausalTransformer.hidden_states))(params, ids)
    table = params["params"]["embed_tokens"]["embedding"]
    rounded_table = np.asarray(jnp.asarray(table, jnp.bfloat16), np.float64)
    h = np.asarray(hidden, np.float64)
    exact = np.einsum("...d,vd->...v", h, rounded_table)
    as_bf16 = np.asarray(jnp.asarray(exact, jnp.bfloat16), np.float64)
    rel = np.abs(logits - exact) / (np.abs(exact) + 1e-30)
    print(f"seed {seed} backend {jax.default_backend()} hidden dtype {hidden.dtype}: logits==bf16(exact product of hidden_states) "
          f"{np.mean(logits == as_bf16):.3f} of entries; median rel diff {np.median(rel):.2e} max {rel.max():.2e}", flush=True)
