"""The one precision context a JAX reference's modules run their float64 oracles under, the counterpart to
tools/torch_precision.py's `float64`: Diffusers' Flax modules', and MaxText's (tools/maxtext_worker.py).

A Flax module built at float64 (`dtype=jnp.float64`, its parameters widened, JAX's x64 on) still computes in
a narrower dtype wherever its code names one: Diffusers' `get_sinusoidal_embeddings` takes its frequencies'
`arange` and `exp` in float32 whatever the module's dtype, so a UNet's timestep projection and SDXL's time
ids' projection carry float32 arithmetic into a float64 run; MaxText's norms cast to float32 and its tied
table to bfloat16. `float64` reads the dtypes it names, in each of the named modules, as float64 within the
block, a truth's trace inside it; the arm under test keeps the modules as released.
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


class Widened:
    """jax.numpy with each dtype in `names` read as float64, every other name its own."""

    def __init__(self, names: tuple[str, ...]):
        self.names = names

    def __getattr__(self, name):
        import jax.numpy as jnp

        return jnp.float64 if name in self.names else getattr(jnp, name)


@contextlib.contextmanager
def float64(modules: tuple[str, ...] = MODELS,
            names: tuple[str, ...] = ("float32",)) -> Iterator[None]:
    """Within the block, the dtypes in `names` that each of `modules` (by import name) names read as float64:
    float32 by default, the dtype a module pins where it means full precision; another where a module casts
    to a narrower one it means to keep (MaxText's tied table's bfloat16, tools/maxtext_worker.py)."""
    with contextlib.ExitStack() as scope:
        for name in modules:
            scope.enter_context(patch.object(importlib.import_module(name), "jnp", Widened(names)))
        yield


__all__ = ["MODELS", "Widened", "float64"]
