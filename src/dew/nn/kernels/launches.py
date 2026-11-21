"""The one launch point of Dew's Pallas kernels: `pallas_call` and Mosaic GPU's `kernel`, Pallas's own, each
call held to the per-shard boundary (`dew.nn.sharding.per_shard`), so no route launches a kernel over
operands an automatic mesh axis splits, where every device would run the whole batch (rows 1897, 1923,
4077)."""

import contextvars
import functools
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.xla_metadata import set_xla_metadata

from dew.nn.kernels.templates import outputs
from dew.nn.sharding import per_shard


def _register_triton_lowerings() -> None:
    """Register Dew's primitive owners' Pallas Triton lowerings on first `pallas_call` (row 9794)."""
    from dew.algebra import scaled
    from dew.nn import derivatives, precision

    derivatives.triton_lowerings()
    precision.triton_lowerings()
    scaled.triton_lowerings()


def pallas_call(kernel: Callable, *, name: str, **params) -> Callable:
    """`jax.experimental.pallas.pallas_call(kernel, name=name, **params)` held per shard (`_held`), named by
    its `name`, which its profile names it by, its body traced as a launch's (`_traced`); an `out_shape`
    stating its outputs as its operands' axes (`outputs.Output`) its shapes, its call declaring their
    dimensions (`outputs.declared`)."""
    _register_triton_lowerings()
    out_shape = params.get("out_shape")
    held = _held(pl.pallas_call(_traced(kernel, params.get("interpret")), name=name,
                                **{**params, "out_shape": outputs.structs(out_shape)}), name)
    if not outputs.dimensioned(out_shape):
        return held

    def declaring(*operands):
        with set_xla_metadata(**outputs.declared(out_shape, operands)):
            return held(*operands)

    return declaring


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


def filled(shape: Sequence[int], dtype, anchor: jax.Array, *anchors: jax.Array, fill: float = 0.0
           ) -> jax.Array:
    """A buffer of `shape` filled with `fill` that one launch alone takes in place (a ref its programs add
    into atomically: an accumulator's zeros, a state's start), its fill passed through an optimization
    barrier with that launch's own operands (`anchor` and `anchors`, one at least) and broadcast after it:
    each launch's broadcast has an operand of its own, which XLA's common-subexpression elimination cannot
    merge with another's, as jax.checkpoint's `prevent_cse` passes a computation's inputs through a barrier
    it then consumes. XLA merges every equal `broadcast(constant)` of one shape (HloCSE sets aside only
    instructions with no operands or with side effects), and a barrier after the broadcast forwards its
    buffer: unbound, one buffer served every layer's launch and lived from the first backward to the last
    (the 4B's split-1 step: the shared-keys backward's dQ zeros, 805 MB, live from layer 31's backward to
    layer 3's). The launch cannot zero its own target in its kernel: other programs add into it with no
    grid-wide barrier before their first add, so the fill is made before the launch, each launch its own,
    on every family (Mosaic's, Pallas-Triton's and the recurrence's alike)."""
    value, _ = jax.lax.optimization_barrier((jnp.asarray(fill, dtype), (anchor, *anchors)))
    return jnp.broadcast_to(value, tuple(shape))
