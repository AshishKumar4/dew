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
    """`jax.experimental.pallas.pallas_call(kernel, name=name, **params)`, each call raising unless it runs
    per shard (`per_shard`), the refusal naming the launch by its `name`, which its profile names it by."""
    call = pl.pallas_call(kernel, name=name, **params)

    @functools.wraps(call)
    def launched(*operands, **options):
        per_shard(name)
        return call(*operands, **options)

    return launched


def kernel(entry: Callable, *, name: str, **params) -> Callable:
    """`jax.experimental.pallas.mosaic_gpu.kernel(entry, **params)`, each call raising unless it runs per
    shard (`per_shard`), the refusal naming the launch by `name`."""
    built = plgpu.kernel(entry, **params)

    @functools.wraps(built)
    def launched(*operands, **options):
        per_shard(name)
        return built(*operands, **options)

    return launched
