"""A continuum level: the parameters a model's own task loss steps in context, a window at a time.

Nested Learning's continuum memory (Behrouz et al., arXiv 2512.24695, section 7.1) updates a block's
parameters every `C` tokens by the gradient of the task's own objective, equation 71:
theta_{i+1} = theta_i - sum_{t in the window} eta_t grad L(theta_t; x_t), at each window's end. The
level here is that block: a residual two-layer map `x + U act(D x)`, equation 89's memory form, read on
a decoder's final states before its head (`CausalTransformer.continuum`). There each token's loss
reads the level through its own state alone, so a window's gradient is the sum of its tokens' own,
equation 71's per-token term exactly. `U` starts at zero, so a model gains the level as the identity:
its logits are the model's without it. The objective that steps it in context is
`dew.objectives.lm.adaptation`; a plain one trains the level as any other parameters.
"""

import dataclasses

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.typing import Dtype, PrecisionLike

from dew.nn.activations import activation
from dew.nn.dense import Dense
from dew.nn.mlp import GatedMLP
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import logical_axes


@dataclasses.dataclass(frozen=True)
class ContinuumLevel:
    """A continuum level's shape: its `hidden` width and its `activation`, a Hugging Face
    `hidden_act` name (`dew.nn.activations.activation`)."""

    hidden: int
    activation: str = "silu"

    def __post_init__(self):
        if self.hidden < 1:
            raise ValueError(f"a continuum level is at least one unit wide, not {self.hidden}")
        activation(self.activation)


@logical_axes({
    ("continuum", "down"): ("embed", "mlp"),
    ("continuum", "up"): ("mlp", "embed"),
})
class Continuum(nn.Module):
    """The level `ContinuumLevel` describes over `[..., features]` states: `x + up(act(down(x)))`,
    `up` zero at its start."""

    level: ContinuumLevel
    features: int
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    def setup(self):
        # The ungated feed-forward `down`, `up`, the latter zero at its start; its first projection is
        # the feed-forward's `up_proj`, named `down` here, and its output `down_proj`, named `up`.
        self.mlp = GatedMLP(self.level.hidden, self.features, activation(self.level.activation),
                            output_init_std=0.0, linear=Dense, names=("down", "up"), dtype=self.dtype,
                            precision=self.precision)
        nn.share_scope(self, self.mlp)

    def __call__(self, x: jax.Array) -> jax.Array:
        # The residual sums in at least fp32 and rounds once to the states' dtype, the head's input.
        wide = at_least_fp32(x.dtype)
        return (x.astype(wide) + self.mlp(x).astype(wide)).astype(x.dtype)

    def step_terms(self, x: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        """What bounds a step of the level's kernels at states `x` `[..., features]`
        (`dew.objectives.lm.adaptation.damping`): each state's |x|^2 and |f|^2, f = act(down(x)) the
        inputs `up` multiplies, and the Frobenius norm of the map `up` applies, an adapter's factors on it
        included, at least its spectral norm; in at least fp32."""
        wide = at_least_fp32(x.dtype)
        hidden = activation(self.level.activation)(self.mlp.up_proj(x)).astype(wide)
        applied = self.mlp.down_proj(jnp.eye(self.level.hidden, dtype=x.dtype)).astype(wide)
        return (jnp.sum(jnp.square(x.astype(wide)), axis=-1), jnp.sum(jnp.square(hidden), axis=-1),
                jnp.sqrt(jnp.sum(jnp.square(applied))))
