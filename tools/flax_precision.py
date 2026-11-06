"""The one precision context Diffusers' Flax modules run their float64 oracles under, the Flax counterpart
to tools/torch_precision.py's `float64`.

A Flax module built at float64 (`dtype=jnp.float64`, its parameters widened, JAX's x64 on) still computes in
float32 wherever its code names float32: Diffusers' `get_sinusoidal_embeddings` takes its frequencies'
`arange` and `exp` in float32 whatever the module's dtype, so a UNet's timestep projection and SDXL's time
ids' projection carry float32 arithmetic into a float64 run. `float64` reads the float32 that each of the
named modules names as float64 within the block, a truth's trace inside it; the arm under test keeps the
modules as released.
"""

from __future__ import annotations

import contextlib
import importlib
from collections.abc import Iterator
from unittest.mock import patch

MODELS = ("diffusers.models.embeddings_flax", "diffusers.models.attention_flax", "diffusers.models.resnet_flax",
          "diffusers.models.vae_flax", "diffusers.models.unets.unet_2d_blocks_flax",
          "diffusers.models.unets.unet_2d_condition_flax")
"""Diffusers' Flax model modules: the UNet's, its blocks', the autoencoder's, attention's and the
embeddings'."""


class Float64Numpy:
    """jax.numpy with its float32 read as float64, for a module that names float32."""

    def __getattr__(self, name):
        import jax.numpy as jnp

        return getattr(jnp, "float64" if name == "float32" else name)


@contextlib.contextmanager
def float64(modules: tuple[str, ...] = MODELS) -> Iterator[None]:
    """Within the block, the float32 each of `modules` names (by import name) read as float64."""
    with contextlib.ExitStack() as scope:
        for name in modules:
            scope.enter_context(patch.object(importlib.import_module(name), "jnp", Float64Numpy()))
        yield


__all__ = ["MODELS", "Float64Numpy", "float64"]
