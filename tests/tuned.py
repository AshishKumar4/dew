"""The configs a kernel's float64 gate holds at: every one its space declares (dew.nn.kernels.tuning)."""

import dataclasses

import jax
import tokamax

from dew.nn.kernels import tuning


def each_config(kernel, function, *args) -> list:
    """Every config of `kernel`'s space at the keys of the shapes
    `function(*args)` launches it with, in field order."""
    with tuning.launches():
        found = tokamax.autotuning.get_bound_args(jax.jit(function), *args)
    return sorted({config for ba in found if type(ba.op) is kernel
                   for config in ba.op.keyed(ba).autotuning_configs}, key=dataclasses.astuple)
