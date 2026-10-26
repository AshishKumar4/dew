"""The Mosaic GPU Family bundles a template's launches derive their kernels on (`mosaic.Family`), one gate
for every template: Hopper's warpgroup MMA's (`mosaic_sm90.WGMMA`) and Ampere's mma.sync's
(`mosaic_sm80.Mma`)."""

import jax.numpy as jnp

from dew.nn.kernels.templates import mosaic, mosaic_sm80, mosaic_sm90


def families(dtype) -> tuple[mosaic.Family, ...]:
    """The bundles a launch over operands of `dtype` may run on; a launch takes its target's (the one whose
    `Family.name` its instruction family names)."""
    return mosaic_sm90.WGMMA, mosaic_sm80.Mma(jnp.dtype(dtype))
