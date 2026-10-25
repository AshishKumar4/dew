"""The configs a function's kernel launches take as they trace (dew.nn.kernels.tuning.seen), and those
a template's launches fit."""

import dataclasses

import jax

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
