"""The configs a function's kernel launches take as they trace (dew.nn.kernels.tuning.seen), those a
template's launches fit, every variant an opt-in tuning run may time (`KernelOp.variants`), which a
kernel's float64 gate holds at, and the launches the tuner times (dew.nn.kernels.tuning)."""

import dataclasses

import jax
import tokamax

from dew.nn.kernels import generation, tuning


def _traced(function, *args) -> tuning.Seen:
    """The launches `function(*args)` makes as it traces, and the config each takes."""
    with tuning.seen() as found:
        jax.make_jaxpr(function)(*args)
    return found


def taken(kernel, function, *args) -> list:
    """Each config `function(*args)`'s launches of `kernel` take as they trace, once, in field order."""
    found = _traced(function, *args)
    return sorted({config for config, ba in found.taken if type(ba.op) is kernel}, key=dataclasses.astuple)


def fitting(kernel, function, *args) -> list:
    """Each config of `function(*args)`'s launches of the template `kernel` that fits the device
    (`Template.fitting`): every one of a fixed space, which its cost model ranks, or the one a
    generating launch derives; once, in field order."""
    found, target = _traced(function, *args), generation.gpu_target()
    return sorted({config for op, ba in found if type(op) is kernel for config in op.fitting(ba, target)},
                  key=dataclasses.astuple)


def each_config(kernel, function, *args) -> list:
    """Every variant of `kernel` (`KernelOp.variants`) at the keys of the shapes `function(*args)`
    launches it with, in field order: the configs a tuning run may pick, each of which the kernel's tests
    hold to the gate."""
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
