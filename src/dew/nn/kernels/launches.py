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


def _register_triton_lowerings() -> None:
    """Register Dew's primitive owners' Pallas Triton lowerings on first `pallas_call` (row 9794)."""
    from dew.algebra import scaled
    from dew.nn import derivatives, division

    derivatives.triton_lowerings()
    division.triton_lowerings()
    scaled.triton_lowerings()


_MOSAIC_GPU_REGISTERED: set[jax_core.Primitive] = set()


def _register_mosaic_gpu(prim: jax_core.Primitive, value: Callable) -> None:
    """Register one Dew primitive's Pallas Mosaic GPU warpgroup lowering by tracing `value`: the one
    lowering-by-jaxpr adapter of every Dew primitive a Mosaic GPU kernel runs (rows 12267, 12330)."""
    if prim in _MOSAIC_GPU_REGISTERED:
        return
    from jax._src.pallas.mosaic_gpu import lowering as mosaic_lowering
    from jax.experimental.mosaic import gpu as mgpu

    def _lower(ctx, *args, **params):
        ir_args = tuple(mosaic_lowering._ensure_ir_value(v, a.dtype)
                        for v, a in zip(args, ctx.avals_in, strict=True))
        closed = jax.make_jaxpr(functools.partial(value, **params))(*ctx.avals_in)
        out = mosaic_lowering.lower_jaxpr_to_mosaic_gpu(
            ctx.module_ctx, ctx.launch_ctx, closed.jaxpr, ir_args, closed.consts)
        return out if prim.multiple_results else out[0]

    mosaic_lowering.register_lowering_rule(prim, mgpu.LoweringSemantics.Warpgroup)(_lower)
    _MOSAIC_GPU_REGISTERED.add(prim)


def _register_mosaic_gpu_lowerings() -> None:
    """Register Dew's primitive owners' Pallas Mosaic GPU lowerings on first `kernel` (rows 9795, 12330)."""
    from dew.algebra import scaled
    from dew.nn import division

    _register_mosaic_gpu(division.correctly_rounded_divide_p, division.software_divide)
    for made, (value, _) in scaled.MADE.items():
        _register_mosaic_gpu(made, value)


def pallas_call(kernel: Callable, *, name: str, **params) -> Callable:
    """`jax.experimental.pallas.pallas_call(kernel, name=name, **params)` held per shard (`_held`), named by
    its `name`, which its profile names it by, its body traced as a launch's (`_traced`), its outputs typed
    as its operands make them (`_typed`); an `out_shape` stating its outputs as its operands' axes
    (`outputs.Output`) its shapes, its call declaring their dimensions (`outputs.declared`)."""
    _register_triton_lowerings()
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
    _register_mosaic_gpu_lowerings()
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
    outside any launch's body. A family's products read it so they issue no instruction the interpreter lacks
    (mma.sync's sums, `mosaic_sm80.Mma.products`), whatever a caller states (row 5209's A100 backward:
    products called with no interpret flag issued mma.sync's)."""
    return _INTERPRETING.get()


@contextlib.contextmanager
def tracing_body(interpret):
    """Within the block, a launch's body traces (`interpreting`), run in Pallas's interpreter where
    `interpret` is given: a launch's own (`_traced`), or a probe of what one traces on its tiles (a hook's
    primitives, `templates.gemm.hooks`). A body a GPU or TPU kernel lowers from traces its integers in 32
    bits whatever `jax_enable_x64` holds, as Mosaic's lowering reads them (a barrier's phase, a loop's
    index; row 10121: x64's 64-bit integers failed a GH200's launch); the interpreter runs either."""
    token = _INTERPRETING.set(bool(interpret))
    try:
        with contextlib.nullcontext() if interpret else jax.enable_x64(new_val=False):
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
