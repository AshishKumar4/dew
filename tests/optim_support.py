"""What the optimizer tests share: the learning rate they build at, the small decoder whose parameter
tree they step, and its tree drawn afresh for a seed."""

import jax
import jax.numpy as jnp
import numpy as np

from dew.nn.backbones.causal_transformer import CausalTransformer

LR = 1e-3


def decoder_params():
    model = CausalTransformer(
        vocab_size=64, emb_features=32, num_layers=2, num_heads=2, num_kv_heads=1,
        mlp_features=64, max_seq_len=8, tie_embeddings=False)
    return model.init(jax.random.key(0), jnp.ones((1, 8), jnp.int32))


def drawn_tree(params, seed):
    """`params`'s tree drawn afresh for a seed, its parameters and a gradient of each, both standard
    normal scaled as the decoder's own init and its gradients run (a gradient a hundredth)."""
    keys = iter(jax.random.split(jax.random.key(seed), 2 * len(jax.tree.leaves(params))))

    def normal(scale):
        return lambda leaf: np.asarray(scale * jax.random.normal(next(keys), leaf.shape, leaf.dtype))

    return jax.tree.map(normal(1.0), params), jax.tree.map(normal(1e-2), params)
