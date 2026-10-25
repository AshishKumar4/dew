"""Each Pallas launch's config: the one its launch derives from the call's shapes and the device's
facts, the same on every run.

A launch is a `KernelOp`, tokamax's `Op` (tokamax/_src/ops/op.py) for one `pallas_call` on primal
arrays. It declares its config dataclass and the rule that gives its config
(`_get_heuristics_config`); the kernel's own `custom_vjp`, shard placement and remat names stay around
it, and the op is never differentiated. Nothing measures a config or stores one
(docs/design/kernel-templates.md §0). A test names one where its purpose is that config (`pinned`),
checked by the launch's legality (`KernelOp.pinnable`), and reads which launches traced and the config
each took (`seen`).
"""

from __future__ import annotations

import contextlib
import dataclasses
from collections.abc import Hashable, Iterator, Mapping
from typing import ClassVar

import tokamax

from dew import cache


class Unpinnable(ValueError):
    """A config pinned for a kernel (`pinned`) that a launch of it cannot take: past a bound its legality
    checks (`KernelOp.unpinnable`)."""

    def __init__(self, kernel: str, config: Hashable, reason: str):
        super().__init__(f"{kernel} cannot take the pinned config {config}: {reason}")
        self.kernel, self.config, self.reason = kernel, config, reason

    def __reduce__(self):
        # Rebuilt from its own fields, as a pickle or a process boundary carries it.
        return type(self), (self.kernel, self.config, self.reason)


@dataclasses.dataclass(frozen=True)
class KernelOp(tokamax.Op):
    """One Pallas launch. A subclass declares `config_cls` (a frozen dataclass), `_fwd(..., config,
    return_residuals)` returning `(outputs, None)`, and `_get_heuristics_config`, the config a launch
    takes at its own shapes on the device it traces for.
    """

    supports_batched_args_capture: ClassVar[bool] = False
    supports_symbolic_shapes: ClassVar[bool] = False

    def __call__(self, *args, **kwargs):
        config = self.config
        if config is None:
            ba = self.bind(*args, **kwargs)
            if (collected := cache.KERNEL_SEEN.value) is not None:
                collected.found.append((self, ba))
            config = self._chosen(ba)
            if collected is not None:
                collected.found.taken.append((config, ba))
        return self._fwd(*args, config=config, **kwargs)[0]

    def _chosen(self, ba: tokamax.BoundArguments) -> Hashable:
        """The config pinned for the launch's kernel (`pinned`), raising `Unpinnable` where the launch
        of `ba` cannot take it, else its heuristic."""
        pins = dict(cache.KERNEL_PINS.value)
        if type(self) not in pins:
            return ba.heuristics_config
        pin = pins[type(self)]
        if not self.pinnable(ba, pin):
            raise Unpinnable(type(self).__name__, pin, self.unpinnable(ba, pin))
        return pin

    def pinnable(self, ba: tokamax.BoundArguments, config: Hashable) -> bool:
        """Whether a pin may name `config` for the launch of `ba`: any config of its type."""
        del ba
        return isinstance(config, self.config_cls)

    def unpinnable(self, ba: tokamax.BoundArguments, config: Hashable) -> str:
        """Why the launch of `ba` cannot take `config`."""
        del ba, config
        return f"it is no {self.config_cls.__name__}"


Launch = tuple[KernelOp, tokamax.BoundArguments]
"""One launch a trace made, its op and its arguments (`seen`)."""


class Seen(list[Launch]):
    """The launches traced under `seen`, each its op and arguments, in the order they traced, and
    `taken`, the config each took as it traced (`KernelOp._chosen`: a pin or its heuristic) and the
    arguments it took it for, in the same order."""

    def __init__(self) -> None:
        super().__init__()
        self.taken: list[tuple[Hashable, tokamax.BoundArguments]] = []


class _Seen:
    """The launches a lowering traced under `seen` (`Seen`); hashed by identity, as
    `dew.cache.KERNEL_SEEN` keys on it."""

    def __init__(self) -> None:
        self.found = Seen()


@contextlib.contextmanager
def seen() -> Iterator[Seen]:
    """Each kernel launch traced within that takes its config as it traces, its op and arguments and
    the config it took (`Seen.taken`), recorded as it traces: a fresh collection each time, so a trace
    cached before traces again."""
    collected = _Seen()
    with cache.KERNEL_SEEN(collected):
        yield collected.found


@contextlib.contextmanager
def pinned(configs: Mapping[type[KernelOp], Hashable]) -> Iterator[None]:
    """Each launch of a kernel `configs` names, traced within, takes that config where its legality
    holds it (`KernelOp.pinnable`), its heuristic set aside: for a test whose purpose is one config,
    and for a lab that times one."""
    with cache.KERNEL_PINS(tuple(configs.items())):
        yield
