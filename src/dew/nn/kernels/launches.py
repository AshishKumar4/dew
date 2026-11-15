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
    its `name`, which its profile names it by, its body traced as a launch's (`_traced`)."""
    return _held(pl.pallas_call(_traced(kernel, params.get("interpret")), name=name, **params), name)


def kernel(entry: Callable, *, name: str, **params) -> Callable:
    """`jax.experimental.pallas.mosaic_gpu.kernel(entry, **params)` held per shard (`_held`), named by
    `name`, its body traced as a launch's (`_traced`)."""
    return _held(plgpu.kernel(_traced(entry, params.get("interpret")), **params), name)


def _traced(body: Callable, interpret) -> Callable:
    """`body`, tracing within its launch, run in Pallas's interpreter where `interpret` is given
    (`bodies.launched`): what its arithmetic reads of the launch (`bodies.subnormals`)."""
    from dew.nn.kernels.templates import bodies  # a template module, which these launches carry

    @functools.wraps(body)
    def traced(*refs, **options):
        with bodies.launched(bool(interpret)):
            return body(*refs, **options)

    return traced


def _held(launch: Callable, name: str) -> Callable:
    """`launch`, each call raising unless it runs per shard (`per_shard`), the refusal naming it `name`."""

    @functools.wraps(launch)
    def launched(*operands, **options):
        per_shard(name)
        return launch(*operands, **options)

    return launched
