"""Probe (not for landing): the memory kind of an offloading rung's residuals under eval_shape."""

import jax
import jax.numpy as jnp

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_block import REMAT_POLICIES


def test_probe():
    for remat in ("minimal", "minimal_offloaded"):
        model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=2, num_heads=2, mlp_features=64,
                                  max_seq_len=16, remat=REMAT_POLICIES[remat])
        ids = jnp.ones((2, 16), jnp.int32)
        variables = jax.eval_shape(lambda: model.init(jax.random.key(0), ids))

        def residuals(params):
            _, linear = jax.linearize(lambda p: model.apply({**variables, "params": p}, ids), params)
            return jax.tree.leaves(linear)

        found = jax.eval_shape(residuals, variables["params"])
        kinds = {}
        for leaf in found:
            space = str(getattr(jax.typeof(leaf), "memory_space", None))
            kinds[space] = kinds.get(space, 0) + leaf.size * leaf.dtype.itemsize
        print("PROBE residuals", remat, kinds)
