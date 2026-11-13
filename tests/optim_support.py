"""What the optimizer tests share: the learning rate they build at, the small decoder whose parameter
tree they step, the axes its modules draw it in, its tree drawn afresh for a seed, and a float64 truth's
numerics (`in_float64`)."""

import jax
import jax.numpy as jnp
import numpy as np

from dew.batch import unboxed
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.kernels.generation import run_precision
from dew.nn.precision import Dtypes, Numerics
from dew.nn.sharding import boxed, boxed_axes

LR = 1e-3

FLOAT64 = Numerics(dtypes=Dtypes(params="float64", compute="float64", output="float64"))
"""A float64 run's numerics, under which an optimizer's matrix iteration multiplies in float64
(`dew.nn.precision.iteration_dtype`)."""

WIDE_MOMENTS = {"mu_dtype": None, "nu_dtype": None}
"""An optimizer's `optimizer_opts` keeping its moments in its parameters' dtype, a float64 truth's, where
`OptimConfig.state_dtype`'s float32 holds them in float32 whatever the parameters'."""


def in_float64(fn):
    """`fn` run under a float64 run's numerics (`FLOAT64`): a float64 truth, an optimizer's Newton-Schulz
    iterating in its parameters' float64 and an objective's scores in a float64 output dtype, where the
    run's default numerics take float32 for both."""
    def wide(*args):
        with run_precision(FLOAT64):
            return fn(*args)
    return wide


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


def optax_orthogonalized(update, steps: int, dimension_numbers=None, *, coeffs=(3.4445, -4.7750, 2.0315),
                         eps: float = 1e-8, preconditioning: str = "frobenius"):
    """optax's own Newton-Schulz of `update` (`orthogonalize_via_newton_schulz`, optax 0.2.8, which it does
    not export), the oracle Dew's iteration (`dew.newton_schulz.newton_schulz`) is held to: `steps`
    iterations at `coeffs`, one triple or a schedule, after `preconditioning`, in the update's dtype; a leaf
    with no `dimension_numbers` one matrix, its rows the reduction axis (a Linen kernel's)."""
    import jax.numpy as jnp
    import optax
    from optax.contrib import _muon

    numbers = optax.contrib.MuonDimensionNumbers() if dimension_numbers is None else dimension_numbers
    return _muon.orthogonalize_via_newton_schulz(update, jnp.asarray(coeffs), steps, preconditioning, eps,
                                                 numbers)
