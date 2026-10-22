"""The configs a kernel's float64 gate holds at, every one its space declares, and the launches the
tuner times (dew.nn.kernels.tuning)."""

import dataclasses

import jax
import tokamax

from dew.nn.kernels import tuning


def each_config(kernel, function, *args) -> list:
    """Every config of `kernel`'s space at the keys of the shapes
    `function(*args)` launches it with, in field order."""
    return sorted({config for ba in launches(function, *args) if type(ba.op) is kernel
                   for config in ba.op.keyed(ba).autotuning_configs}, key=dataclasses.astuple)


def launches(function, *args) -> list:
    """The tuner's launches `function(*args)` makes, as the tuner reads them from a program."""
    with tuning.launches():
        return tokamax.autotuning.get_bound_args(jax.jit(function), *args)


def timed(ba) -> tuple:
    """The launch `ba` run on the arrays the tuner times it on, filled into its arguments as the tuner
    fills them (`tuning.materialized`)."""
    arrays, fill = tuning.materialized(ba.op, ba)
    args, kwargs = fill(arrays)
    return ba.op(*args, **kwargs)
