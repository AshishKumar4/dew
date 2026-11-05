"""DreamerV3's optimizer (`laprop`): LaProp's direction (`scale_by_laprop`) of the update clipped adaptively
(`clip_by_agc`), at DreamerV3's learning rate. `dew.optim` registers `laprop` among its optimizers."""

import jax
import jax.numpy as jnp
import optax

from dew._moments import bias_corrected
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import layer_norms, layers_of


def clip_by_agc(clip: float = 0.3, floor: float = 1e-3) -> optax.GradientTransformation:
    """Adaptive gradient clipping (Brock et al., arXiv 2102.06171) as DreamerV3's code takes it
    (embodied/jax/opt.py:109-123): each parameter's update scaled down until its norm is at most `clip`
    times the parameter's own, which `floor` keeps from reaching zero with the parameter (a bias drawn at
    zero still moves), one norm over each stored layer (`dew.nn.sharding.layers_of`): the whole array, or
    each layer of a stage-local run's stacked leaf. A `clip` of 0 clips nothing. The update reads the
    parameters."""
    if not clip:
        return optax.identity()

    def update_fn(updates, state, params=None):
        if params is None:
            raise ValueError("adaptive gradient clipping reads the parameters, so its update needs them")

        def clipped(path, param, update):
            # Each stored layer clipped by its own norms (`layer_norms`): a stage-local run's stacked leaf
            # clips its layers as the replicated run clips each (row 3933).
            stacked = layers_of(path)[1]
            limit = clip * jnp.maximum(floor, layer_norms(param, stacked))
            return update * (1 / jnp.maximum(1.0, layer_norms(update, stacked) / limit))

        return jax.tree_util.tree_map_with_path(clipped, params, updates), state

    return optax.GradientTransformation(lambda params: optax.EmptyState(), update_fn)


def _dreamer_rate(count: jax.typing.ArrayLike) -> jax.Array:
    """DreamerV3's learning rate after `count` updates (agent.py:342-379, configs.yaml `opt`): 4e-5, ramped
    linearly from zero over the first 1000 updates. Formed in the run's own float dtype, where optax's
    `linear_schedule` returns float32 whatever the run's, its ramp off by 1e-5 of itself at the start."""
    return jnp.minimum(count, 1000) * (4e-5 / 1000)


def scale_by_laprop(b1: float = 0.9, b2: float = 0.999, eps: float = 1e-20) -> optax.GradientTransformation:
    """LaProp's direction (Ziyin et al., arXiv 2002.04839) as DreamerV3's code forms it
    (embodied/jax/opt.py:126-164): each gradient over the root of its bias-corrected mean square (decay
    `b2`, `eps` added outside the root), then the bias-corrected momentum of decay `b1` of those quotients.
    Adam divides the momentum of the gradients; LaProp takes the momentum of the divided ones. Both bias
    corrections form in double (`bias_corrected`). The state is `optax.ScaleByAdamState`'s, its moments in
    float32 at least, as the code keeps them."""

    def moments(params):
        return jax.tree.map(lambda param: jnp.zeros(param.shape, at_least_fp32(param.dtype)), params)

    def init_fn(params):
        return optax.ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=moments(params), nu=moments(params))

    def update_fn(updates, state, params=None):
        del params  # LaProp reads only the gradients
        count = optax.safe_increment(state.count)

        def widened(update, moment):
            # A gradient narrower than its moments is widened before either reads it.
            return update.astype(jnp.promote_types(update.dtype, moment.dtype))

        updates = jax.tree.map(widened, updates, state.nu)
        nu = optax.tree.update_moment_per_elem_norm(updates, state.nu, b2, 2)
        normalized = jax.tree.map(lambda update, mean_square: update / (jnp.sqrt(mean_square) + eps), updates,
                                  bias_corrected(nu, b2, count))
        mu = optax.tree.update_moment(normalized, state.mu, b1, 1)
        return bias_corrected(mu, b1, count), optax.ScaleByAdamState(count=count, mu=mu, nu=nu)

    return optax.GradientTransformation(init_fn, update_fn)


def laprop(learning_rate: float | optax.Schedule | None = None, b1: float = 0.9, b2: float = 0.999,
           eps: float = 1e-20, agc: float = 0.3) -> optax.GradientTransformation:
    """DreamerV3's optimizer (dreamerv3/agent.py:342-379): the update clipped adaptively to `agc` of each
    parameter's norm (`clip_by_agc`), LaProp's direction of it (`scale_by_laprop`, momentum `b1`, mean
    square `b2`, `eps`), stepped at `learning_rate`. Where that is None it is the code's, 4e-5 ramped
    linearly from zero over the first 1000 updates."""
    return optax.chain(
        clip_by_agc(agc),
        scale_by_laprop(b1, b2, eps),
        optax.scale_by_learning_rate(_dreamer_rate if learning_rate is None else learning_rate),
    )


__all__ = ["clip_by_agc", "laprop", "scale_by_laprop"]
