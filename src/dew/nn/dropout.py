"""Dropout whose mask is the same draw in every precision.

`jax.random.bernoulli` draws its uniform at its probability's dtype, so a Python keep probability is a
float32 uniform in a float32 run and a float64 uniform, from other random bits, under
`jax.enable_x64`: a float64 run of a model would drop other units than the float32 run it widens, and
score another forward. Dew's dropout draws its mask at float32 in every mode (`kept`), as its
objectives draw their condition dropout (`dew.objectives.diffusion`)."""

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax import lax


def kept(rng: jax.Array, keep: float, shape) -> jax.Array:
    """Which units a dropout keeping `keep` of them keeps, drawn from `rng`: a float32 uniform below a
    float32 `keep`, the mask a float32 run draws, in every mode."""
    return jax.random.bernoulli(rng, jnp.float32(keep), shape)


class Dropout(nn.Dropout):
    """`flax.linen.Dropout`, its mask drawn by `kept`: a float32 run's mask is flax's own, bit for bit,
    and a float64 run drops exactly the same units."""

    @nn.compact
    def __call__(self, inputs, deterministic: bool | None = None, rng: jax.Array | None = None):
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)
        if self.rate == 0.0 or deterministic:
            return inputs
        # A rate of one drops everything, without dividing by a keep of zero, as flax does.
        if self.rate == 1.0:
            return jnp.zeros_like(inputs)
        keep = 1.0 - self.rate
        if rng is None:
            rng = self.make_rng(self.rng_collection)
        shape = list(inputs.shape)
        for dim in self.broadcast_dims:
            shape[dim] = 1
        mask = jnp.broadcast_to(kept(rng, keep, shape), inputs.shape)
        return lax.select(mask, inputs / keep, jnp.zeros_like(inputs))
