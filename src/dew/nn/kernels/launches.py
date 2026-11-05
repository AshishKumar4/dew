"""The one launch point of Dew's Pallas calls: `pallas_call`, Pallas's own, each call held to the per-shard
boundary (`dew.nn.sharding.per_shard`) as Mosaic GPU's launches are (`templates.mosaic.checked`), so no
route launches a kernel over operands an automatic mesh axis splits, where every device would run the
whole batch (rows 1897, 1923)."""

import functools
from collections.abc import Callable

from jax.experimental import pallas as pl

from dew.nn.sharding import per_shard


def pallas_call(kernel: Callable, **params) -> Callable:
    """`jax.experimental.pallas.pallas_call(kernel, **params)`, each call raising unless it runs per shard
    (`per_shard`), named by the call's `name` or the kernel's own."""
    call = pl.pallas_call(kernel, **params)
    named = params.get("name") or getattr(kernel, "__name__", None) or getattr(
        getattr(kernel, "func", None), "__name__", "a Pallas kernel")

    @functools.wraps(call)
    def launched(*operands, **options):
        per_shard(named)
        return call(*operands, **options)

    return launched
