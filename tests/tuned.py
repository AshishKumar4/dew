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
    """The launch `ba` run on the arrays the tuner times it on (`KernelOp.operands`), the way it
    fills them into the launch's arguments."""
    leaves, tree = jax.tree.flatten((ba.args, ba.kwargs),
                                    is_leaf=lambda leaf: isinstance(leaf, jax.ShapeDtypeStruct))
    places = [index for index, leaf in enumerate(leaves) if isinstance(leaf, jax.ShapeDtypeStruct)]
    for index, array in zip(places, ba.op.operands([leaves[index] for index in places], 0), strict=True):
        leaves[index] = array
    args, kwargs = jax.tree.unflatten(tree, leaves)
    return ba.op(*args, **kwargs)
