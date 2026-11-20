"""The one launch point of Dew's Pallas kernels: `pallas_call` and Mosaic GPU's `kernel`, Pallas's own, each
call held to the per-shard boundary (`dew.nn.sharding.per_shard`), so no route launches a kernel over
operands an automatic mesh axis splits, where every device would run the whole batch (rows 1897, 1923,
4077)."""

import contextlib
import contextvars
import functools
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
from jax._src import core as jax_core
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.experimental.xla_metadata import set_xla_metadata

from dew.nn.kernels.templates import outputs
from dew.nn.sharding import checked, computed_from, per_shard


def pallas_call(kernel: Callable, *, name: str, **params) -> Callable:
    """`jax.experimental.pallas.pallas_call(kernel, name=name, **params)` held per shard (`_held`), named by
    its `name`, which its profile names it by, its body traced as a launch's (`_traced`), its outputs typed
    as its operands make them (`_typed`); an `out_shape` stating its outputs as its operands' axes
    (`outputs.Output`) its shapes, its call declaring their dimensions (`outputs.declared`)."""
    out_shape = params.get("out_shape")
    traced = _traced(kernel, params.get("interpret"))
    held = _typed(lambda typed: pl.pallas_call(traced, name=name, **{**params, "out_shape": typed}),
                  outputs.structs(out_shape), name)
    if not outputs.dimensioned(out_shape):
        return held

    def declaring(*operands):
        with set_xla_metadata(**outputs.declared(out_shape, operands)):
            return held(*operands)

    return declaring


def kernel(entry: Callable, *, name: str, out_type, **params) -> Callable:
    """`jax.experimental.pallas.mosaic_gpu.kernel(entry, out_type=out_type, **params)` held per shard
    (`_held`), named by `name`, its body traced as a launch's (`_traced`), its outputs typed as its operands
    make them (`_typed`)."""
    traced = _traced(entry, params.get("interpret"))
    return _typed(lambda typed: plgpu.kernel(traced, out_type=typed, **params), out_type, name)


def _typed(made: Callable, out, name: str) -> Callable:
    """`made(typed)`'s launch held per shard (`_held`), `typed` its outputs `out` (`jax.ShapeDtypeStruct`s)
    as its operands make them where values are typed by the manual axes they vary over
    (`dew.nn.sharding.checked`): each device computes its results from its own operands, so each varies over
    every axis an operand varies over unless it states its own (`computed_from`; row 9808: a result typed
    the same on every device beside a bias each device reads its own of), stated as a typed aval, which
    Pallas takes as it is where Mosaic GPU's wrapper reads a struct with the check off and drops its type.
    `out` as it is elsewhere. One launch made a typing."""
    launches = {}

    def launched(*operands):
        typed = jax.tree.map(lambda leaf: _aval(computed_from(leaf, *operands))
                             if checked() and isinstance(leaf, jax.ShapeDtypeStruct) else leaf, out)
        leaves, structure = jax.tree.flatten(typed)
        key = structure, tuple(leaves)
        if key not in launches:
            launches[key] = _held(made(typed), name)
        return launches[key](*operands)

    return launched


def _aval(struct: jax.ShapeDtypeStruct) -> jax_core.ShapedArray:
    """`struct` as the aval Pallas makes of it where it checks the manual axes values vary over
    (`pallas.core`'s conversion of an output's shape): placed on the context's mesh, varying as it states."""
    return jax_core.ShapedArray(struct.shape, struct.dtype, sharding=jax_core.get_cur_mesh_sharding(),
                                manual_axis_type=struct.manual_axis_type)


_INTERPRETING: contextvars.ContextVar[bool | None] = contextvars.ContextVar("interpreting", default=None)
"""Whether the launch whose body is tracing runs in Pallas's interpreter (`_traced`, every launch's body);
None outside any launch's body."""


def interpreting() -> bool | None:
    """Whether the launch whose body is tracing runs in Pallas's interpreter (`pallas_call`, `kernel`): None
    outside any launch's body."""
    return _INTERPRETING.get()


@contextlib.contextmanager
def tracing_body(interpret):
    """Within the block, a launch's body traces (`interpreting`), run in Pallas's interpreter where
    `interpret` is given: a launch's own (`_traced`), or a probe of what one traces on its tiles (a hook's
    primitives, `templates.gemm.hooks`)."""
    token = _INTERPRETING.set(bool(interpret))
    try:
        yield
    finally:
        _INTERPRETING.reset(token)


def _traced(body: Callable, interpret) -> Callable:
    """`body`, tracing within its launch, run in Pallas's interpreter where `interpret` is given
    (`tracing_body`)."""

    @functools.wraps(body)
    def traced(*refs, **options):
        with tracing_body(interpret):
            return body(*refs, **options)

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
