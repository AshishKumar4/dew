"""The one launch point of Dew's Pallas kernels: `pallas_call` and Mosaic GPU's `kernel`, Pallas's own, each
call held to the per-shard boundary (`dew.nn.sharding.per_shard`), so no route launches a kernel over
operands an automatic mesh axis splits, where every device would run the whole batch (rows 1897, 1923,
4077)."""

import functools
from collections.abc import Callable

from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

from dew.nn.sharding import per_shard


def pallas_call(kernel: Callable, *, name: str, **params) -> Callable:
    """`jax.experimental.pallas.pallas_call(kernel, name=name, **params)` held per shard (`_held`), named by
    its `name`, which its profile names it by."""
    return _held(pl.pallas_call(kernel, name=name, **params), name)


def kernel(entry: Callable, *, name: str, **params) -> Callable:
    """`jax.experimental.pallas.mosaic_gpu.kernel(entry, **params)` held per shard (`_held`), named by
    `name`."""
    return _held(plgpu.kernel(entry, **params), name)


def _held(launch: Callable, name: str) -> Callable:
    """`launch`, each call raising unless it runs per shard (`per_shard`), the refusal naming it `name`."""

    @functools.wraps(launch)
    def launched(*operands, **options):
        per_shard(name)
        return launch(*operands, **options)

    return launched
