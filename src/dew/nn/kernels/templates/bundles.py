"""The Mosaic GPU Family bundles a template's launches derive their kernels on (`mosaic.Family`), one gate
for every template: Hopper's warpgroup MMA's (`mosaic_sm90.WGMMA`) always, and Ampere's mma.sync's
(`mosaic_sm80.Mma`) where this JAX's layout inference places its tiles (`mosaic_sm80.staging_solved`), or in
a CPU process that interprets them, whose interpreter takes them at any JAX (a CPU's lowering for an assumed
Ampere GPU takes them only where the GPU's own would)."""

import jax
import jax.numpy as jnp

from dew.nn.kernels.templates import mosaic, mosaic_sm80, mosaic_sm90


def families(dtype, *, interpreted: bool) -> tuple[mosaic.Family, ...]:
    """The bundles a launch over operands of `dtype` may run on, `interpreted` or lowered; a launch takes
    its target's (the one whose `Family.name` its instruction family names)."""
    ampere = mosaic_sm80.staging_solved() or (interpreted and jax.default_backend() == 'cpu')
    return mosaic_sm90.WGMMA, *((mosaic_sm80.Mma(jnp.dtype(dtype)),) if ampere else ())
