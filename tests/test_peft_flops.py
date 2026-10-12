"""Frozen-base and trainable-factor work, including the actual gradient matrix shapes."""


import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from test_model_flops import forward_products

from dew.lora import LoRA
from dew.nn.flops import dense_training_flops, projection_flops
from dew.objectives.base import FROZEN


class DenseCount(nn.Module):
    inputs: int
    outputs: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.outputs, use_bias=False, name="layer")(x)

    @nn.nowrap
    def model_flops(self, lengths):
        return projection_flops(lengths, self.inputs, self.outputs)

    @nn.nowrap
    def parameter_flops(self, lengths):
        return self.model_flops(lengths)

    @nn.nowrap
    def matrix_usage(self, path, lengths, *, logits=False):
        return np.zeros(np.shape(lengths), np.int64) if logits else np.asarray(lengths)


def test_frozen_dense_rank_factors_equal_the_actual_forward_and_gradient_dots():
    model = DenseCount(8, 16)
    x = jnp.ones((2, 3, 8))
    variables = model.init(jax.random.key(401), x)
    adapted = LoRA(rank=2, alpha=2, modules=("layer",)).apply(model, variables, key=501)
    lengths = np.array([3, 3])
    expected = dense_training_flops(lengths, 8, 16, trainable=False, adapter_rank=2)
    assert adapted.model.training_flops(lengths) == expected

    def forward_backward(params, frozen, x):
        def forward(params, x):
            return adapted.model.apply({"params": params, FROZEN: frozen}, x)
        y, pullback = jax.vjp(forward, params, x)
        return y, *pullback(jnp.ones_like(y))

    assert expected == forward_products(forward_backward, adapted.variables["params"],
                                        adapted.variables[FROZEN], x)
