"""The one launch point of Dew's Pallas kernels: `pallas_call` and Mosaic GPU's `kernel`, Pallas's own, each
call held to the per-shard boundary (`dew.nn.sharding.per_shard`), so no route launches a kernel over
operands an automatic mesh axis splits, where every device would run the whole batch (rows 1897, 1923,
4077)."""

import contextvars
import functools
from collections.abc import Callable

from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.sharding import per_shard


def pallas_call(kernel: Callable, *, name: str, **params) -> Callable:
    """`jax.experimental.pallas.pallas_call(kernel, name=name, **params)` held per shard (`_held`), named by
    its `name`, which its profile names it by, its body traced as a launch's (`_traced`)."""
    return _held(pl.pallas_call(_traced(kernel, params.get("interpret")), name=name, **params), name)


def kernel(entry: Callable, *, name: str, **params) -> Callable:
    """`jax.experimental.pallas.mosaic_gpu.kernel(entry, **params)` held per shard (`_held`), named by
    `name`, its body traced as a launch's (`_traced`)."""
    return _held(plgpu.kernel(_traced(entry, params.get("interpret")), **params), name)


_INTERPRETING: contextvars.ContextVar[bool | None] = contextvars.ContextVar("interpreting", default=None)
"""Whether the launch whose body is tracing runs in Pallas's interpreter (`_traced`, every launch's body);
None outside any launch's body."""


def interpreting() -> bool | None:
    """Whether the launch whose body is tracing runs in Pallas's interpreter (`pallas_call`, `kernel`): None
    outside any launch's body."""
    return _INTERPRETING.get()


def _traced(body: Callable, interpret) -> Callable:
    """`body`, tracing within its launch, run in Pallas's interpreter where `interpret` is given
    (`interpreting`)."""

    @functools.wraps(body)
    def traced(*refs, **options):
        token = _INTERPRETING.set(bool(interpret))
        try:
            return body(*refs, **options)
        finally:
            _INTERPRETING.reset(token)

    return traced


def _held(launch: Callable, name: str) -> Callable:
    """`launch`, each call raising unless it runs per shard (`per_shard`), the refusal naming it `name`."""

    @functools.wraps(launch)
    def launched(*operands, **options):
        per_shard(name)
        return launch(*operands, **options)

    return launched
