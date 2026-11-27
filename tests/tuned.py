"""The configs a function's kernel launches take as they trace (dew.nn.kernels.tuning.seen), those a
template's launches fit, every variant an opt-in tuning run may time (`KernelOp.variants`), which a
kernel's float64 gate holds at, and the launches the tuner times (dew.nn.kernels.tuning)."""

import dataclasses
from collections.abc import Collection

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


def each_config(kernel, function, *args) -> list:
    """Each pin of `kernel` the gate holds `function(*args)`'s launches of it at, as they trace (no lowering,
    which an interpreter's ordered callbacks refuse): `(pin, config)`, the pin a launch's key
    (`tuning.pinned`'s `(kernel, key)`), the config each its launches at that key take or a tuning run times
    of their variants there (`tuning._candidates`), which each of them may be pinned to and timed at
    (`tuning._timeable`): every launch's configs at its own key, a config one launch refuses no bar to
    another's (a forward's tile its transposed gradient spills); in key and field order
    (`configs_by_kernel`'s)."""
    return configs_by_kernel(function, *args, kernels=(kernel,)).get(kernel, [])


def configs_by_kernel(function, *args, kernels: Collection[type] | None = None) -> dict:
    """`each_config` of every kernel `function(*args)` launches (of `kernels` alone where named, no other's
    candidates certified), traced once."""
    by_key: dict = {}
    for observed in _traced(function, *args).held():
        op, keyed = observed
        if kernels is None or type(op) in kernels:
            by_key.setdefault((type(op), keyed.autotuning_cache_key), []).append((op, keyed, observed))
    pins: dict = {}
    for pin, launches in by_key.items():
        offered = {observed.config for _, _, observed in launches}.union(
            *(tuning._candidates(op, keyed) for op, keyed, _ in launches)) - {None}
        held = [config for config in offered
                if all(tuning._timeable(op, observed.arguments, config) for op, _, observed in launches)]
        pins.setdefault(pin[0], []).extend((pin, config) for config in held)
    return {kernel: sorted(found, key=lambda each: (repr(each[0][1]), dataclasses.astuple(each[1])))
            for kernel, found in pins.items()}


def recorded(function, *args) -> list:
    """The launches `function(*args)` makes as it traces, in that order, each keyed as the tuner reads,
    measures and stores it (`tuning.seen`)."""
    return [ba for _, ba in _traced(function, *args)]


def timed(ba) -> tuple:
    """The launch `ba` run on the arrays the tuner times it on, filled into its arguments as the tuner
    fills them (`tuning.materialized`)."""
    arrays, fill = tuning.materialized(ba.op, ba)
    args, kwargs = fill(arrays)
    return ba.op(*args, **kwargs)
