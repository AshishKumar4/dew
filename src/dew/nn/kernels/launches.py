"""The one launch point of Dew's Pallas calls: `pallas_call`, Pallas's own, each call held to the per-shard
boundary (`dew.nn.sharding.per_shard`) as Mosaic GPU's launches are (`templates.mosaic.checked`), so no
route launches a kernel over operands an automatic mesh axis splits, where every device would run the
whole batch (rows 1897, 1923)."""

import functools
from collections.abc import Callable

from jax.experimental import pallas as pl

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
