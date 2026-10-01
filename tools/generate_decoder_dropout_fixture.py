"""Record the zero-dropout decoder's CPU fp32 forward and gradients."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.backbones.causal_transformer import CausalTransformer

model = CausalTransformer(vocab_size=17, emb_features=8, num_layers=1, num_heads=2,
                         mlp_features=16, max_seq_len=8, attention_impl="reference",
                         precision=jax.lax.Precision.HIGHEST)
with jax.default_device(jax.devices("cpu")[0]):
    ids = jnp.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], jnp.int32)
    params = model.init(jax.random.key(2026), ids)
    forward = model.apply(params, ids, train=True)
    gradient = jax.grad(lambda p: jnp.sum(model.apply(p, ids, train=True)))(params)
held = {"ids": np.asarray(ids), "forward": np.asarray(forward)}
for kind, values in (("params", params), ("grad", gradient)):
    for path, leaf in jax.tree_util.tree_flatten_with_path(values)[0]:
        held[kind + "/" + "/".join(item.key for item in path)] = np.asarray(leaf)
np.savez(Path(__file__).resolve().parents[1] / "tests/fixtures/decoder-default-dropout.npz", **held)
