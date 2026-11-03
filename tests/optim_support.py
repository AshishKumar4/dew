"""What the optimizer tests share: the learning rate they build at, the small decoder whose parameter
tree they step, the axes its modules draw it in, and its tree drawn afresh for a seed."""

import jax
import jax.numpy as jnp
import numpy as np

from dew.batch import unboxed
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.sharding import boxed, boxed_axes

LR = 1e-3


def decoder_draw():
    """The small decoder's init: its parameters in the boxes its modules declare their axes in."""
    model = CausalTransformer(
        vocab_size=64, emb_features=32, num_layers=2, num_heads=2, num_kv_heads=1,
        mlp_features=64, max_seq_len=8, tie_embeddings=False)
    return model.init(jax.random.key(0), jnp.ones((1, 8), jnp.int32))


def decoder_params():
    """The small decoder's parameters as arrays, the tree an optimizer steps (`dew.batch.unboxed`)."""
    return unboxed(decoder_draw())


def drawn_axes(*draws):
    """The context in which Dew's optimizers read the axes of the trees `draws` were drawn in, as a
    trainer holds its run's (`Objective.parameter_axes`)."""
    return boxed({path: axes for draw in draws for path, axes in boxed_axes(draw).items()})


def drawn_tree(params, seed):
    """`params`'s tree drawn afresh for a seed, its parameters and a gradient of each, both standard
    normal scaled as the decoder's own init and its gradients run (a gradient a hundredth)."""
    keys = iter(jax.random.split(jax.random.key(seed), 2 * len(jax.tree.leaves(params))))

    def normal(scale):
        return lambda leaf: np.asarray(scale * jax.random.normal(next(keys), leaf.shape, leaf.dtype))

    return jax.tree.map(normal(1.0), params), jax.tree.map(normal(1e-2), params)
