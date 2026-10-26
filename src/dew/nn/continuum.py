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
from flax.typing import Dtype, PrecisionLike

from dew.nn.activations import activation
from dew.nn.dense import Dense
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
    dtype: Dtype | None = None
    precision: PrecisionLike = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        down = Dense(self.level.hidden, use_bias=False, dtype=self.dtype, precision=self.precision,
                     name="down")
        up = Dense(x.shape[-1], use_bias=False, dtype=self.dtype, precision=self.precision, name="up",
                   kernel_init=nn.initializers.zeros_init())
        return x + up(activation(self.level.activation)(down(x))).astype(x.dtype)
