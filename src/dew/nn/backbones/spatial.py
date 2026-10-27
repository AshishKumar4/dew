"""Execute a spatial architecture on arrays or on its analytical geometry."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class SpatialShape:
    """The dimensions a spatial walk reads without allocating an activation."""

    shape: tuple[int, ...]


class SpatialForward:
    """The walk's ordinary operations, leaving module scopes and arrays unchanged."""

    def project(self, module, x):
        return module(x)

    def residual(self, module, x, time, *, train=False):
        return module(x, time, train=train)

    def attention(self, module, x, context):
        return module(x, context)

    def stage_attention(self, module, x, context):
        return module(x, context)

    def level(self, module, x, time, context, skips=(), *, upsample_shape=None, train=False):
        return module(x, time, context, skips, upsample_shape=upsample_shape, train=train)

    def concatenate(self, x, skip):
        return jnp.concatenate([x, skip], axis=-1)

    def resize(self, x, height: int, width: int):
        from ..blocks import torch_nearest_resize
        return torch_nearest_resize(x, height, width)

    def normalized(self, x, norm):
        return norm(x)

    def activated(self, x, activation):
        return activation(x)


class SpatialCount:
    """The same walk charges module products and transports only their shapes."""

    def __init__(self, context_lengths=None, text_features: int | None = None):
        self.flops = 0.0
        self.context_lengths = context_lengths
        self.text_features = text_features

    @staticmethod
    def positions(x: SpatialShape) -> np.ndarray:
        return np.full((x.shape[0],), np.prod(x.shape[1:-1]), np.int64)

    def project(self, module, x: SpatialShape) -> SpatialShape:
        output = SpatialShape((x.shape[0], *module.spatial_shape(x.shape[1:])))
        self.flops += module.model_flops(self.positions(output), x.shape[-1])
        return output

    def residual(self, module, x: SpatialShape, time: SpatialShape, *, train=False) -> SpatialShape:
        output = SpatialShape((x.shape[0], *module.spatial_shape(x.shape[1:])))
        self.flops += module.model_flops(self.positions(output), x.shape[-1], time.shape[-1])
        return output

    def attention(self, module, x: SpatialShape, context) -> SpatialShape:
        self.flops += module.cross_flops(self.positions(x), self.context_lengths, self.text_features)
        return x

    def stage_attention(self, module, x: SpatialShape, context) -> SpatialShape:
        self.flops += module.cross_flops(self.positions(x), x.shape[-1],
                                         self.context_lengths, self.text_features)
        return x

    def level(self, module, x: SpatialShape, time: SpatialShape, context, skips=(), *,
               upsample_shape=None, train=False):
        return module.walk(x, time, context, skips, upsample_shape=upsample_shape, train=train,
                           operations=self)

    def concatenate(self, x: SpatialShape, skip: SpatialShape) -> SpatialShape:
        return SpatialShape((*x.shape[:-1], x.shape[-1] + skip.shape[-1]))

    def resize(self, x: SpatialShape, height: int, width: int) -> SpatialShape:
        return SpatialShape((x.shape[0], height, width, x.shape[-1]))

    def normalized(self, x: SpatialShape, norm) -> SpatialShape:
        return x

    def activated(self, x: SpatialShape, activation) -> SpatialShape:
        return x
