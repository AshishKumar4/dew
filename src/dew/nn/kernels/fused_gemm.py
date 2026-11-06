"""The grouped GEMM template under `jax.experimental.pallas.fuser`: a call site's consumers of the
product become the template's epilogue, and its left operand's producer the template's prologue.

Inside a function `fuser.fuse` transforms, `fusible_gemm(lhs, rhs, group_sizes)` is `gemm.gemm`'s
fp32 product, and whatever the function computes from it elementwise or by broadcasting (a bias, a scale,
a residual, an activation, the final cast) runs in the kernel's epilogue on each accumulator tile, the
arrays it reads loaded beside the tile:

    @fuser.fuse
    def layer(x, w, group_sizes, bias):
        return (fused.fusible_gemm(x, w, group_sizes) + bias).astype(jnp.bfloat16)

The fuser hands the consumer over as one function of the product and the arrays it captured
(`fuser.get_fusion_values`); `pull_block_spec` names the launch axes each captured array follows and
evaluates the consumer on tiles, so the epilogue is a plain JAX function that tiles itself at whatever
block the kernel traces it with (`acc.shape`), and runs whole where XLA does. A consumer the kernels
cannot take tile by tile (one that reads the tile's position, a captured array that follows no launch
axis, scalar prefetch) runs in XLA after the template's product, the value the same, and the call reports
it (`generation.route`'s 'gemm_epilogue', XLA's with the refusal's reason), so a model that loses its
fusion shows where it does.

The left operand's producer, where it is one (an RMS or layer norm of the residual stream), is split in
two: what it computes of each whole row or each depth position alone (a row's mean square and its inverse
root, a norm's weight) runs in XLA before the product, a row or a depth vector each; what it computes of
each element of the operand (the scaling of the loaded tile by them) runs in the kernel's prologue on each
loaded tile, so the normalized operand never reaches global memory. A producer the kernels cannot take tile
by tile, or whose operand changes its base's dtype, runs in XLA before the product, reported as the
epilogue's is ('gemm_prologue').

Each stage is its selector's (`EPILOGUE`, `PROLOGUE`, `dew.nn.kernels.selection`), whose one fused kernel
requires the analysis to take the stage tile by tile: `generation.chosen(gemm_epilogue=...,
gemm_prologue=...)` names either stage's route: 'xla' runs it in XLA beside the product whatever the
kernels take, the reference a fused route is checked against, and 'fused' raises where the kernels cannot
take it rather than run XLA (`selection.KernelRefused`).
"""

import contextvars
import dataclasses
import functools
import math

import jax
import jax.numpy as jnp
from flax.typing import PrecisionLike
from jax.custom_derivatives import zero_from_primal
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser
from jax.extend import core as jax_core

from dew import _cache, algebra
from dew.algebra import cost as algebra_cost
from dew.nn import programs
from dew.nn.kernels import gemm, generation
from dew.nn.kernels.selection import Kernel, KernelRefused, Op, Refusal, analysed, select
from dew.nn.kernels.templates import cost

FUSION_ROW = 'ledger 1295'
"""The fusion analysis's limits."""


AXES = ('rows', 'columns')
"""The epilogue's launch axes, in a tile's order."""


def fusible_gemm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array | None = None, *,
                 transpose_rhs: bool = False, interpret_on_cpu: bool = False) -> jax.Array:
    """`gemm.gemm`'s fp32 product of each group's rows of `lhs` and its matrix of `rhs`, its consumers
    fused into the epilogue under `fuser.fuse` (the module's docstring): `product` declared fp32."""
    return product(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, out_dtype=jnp.float32,
                   interpret_on_cpu=interpret_on_cpu)


def product(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array | None = None, *,
            transpose_rhs: bool = False, precision: PrecisionLike = None, out_dtype=None,
            interpret_on_cpu: bool = False) -> jax.Array:
    """The fusible product of each group's rows of `lhs` and its matrix of `rhs` (one matrix where
    `group_sizes` is None, a dense dot), declared as the dot it stands for declares itself: its `precision`
    (`gemm.gemm`'s) and its result dtype (`out_dtype`, `lhs`'s where None; a `preferred_element_type`'s).
    Its products sum in fp32 at least and round once to `out_dtype`, the dot's own rounding, so its
    consumers, fused into the template's epilogue under `fuser.fuse`, read the product exactly as the
    program they came from does: the accumulator tile rounded to `out_dtype`, their first operation."""
    declared = (precision, jnp.dtype(lhs.dtype if out_dtype is None else out_dtype))
    if not all(jnp.issubdtype(operand.dtype, jnp.inexact) for operand in (lhs, rhs)):
        # An integer product (a quantized dot's) is XLA's own, nothing fused: the template sums floating
        # products alone, so a region of one has no stage to analyse or price (`analysis`).
        if group_sizes is None:
            return jax.lax.dot_general(lhs, rhs, (((1,), (1 if transpose_rhs else 0,)), ((), ())),
                                       precision=precision, preferred_element_type=declared[1])
        matrices = jnp.swapaxes(rhs, 1, 2) if transpose_rhs else rhs
        return jax.lax.ragged_dot(lhs, matrices, group_sizes, precision=precision,
                                  preferred_element_type=declared[1])
    if (made := _FUSIBLES.get()) is not None:
        made.append(declared)
    fusible = fuser.fusible(output_fusion_prefix=True)(
        functools.partial(_product, transpose_rhs, interpret_on_cpu, group_sizes is None, declared))
    return fusible(lhs, rhs) if group_sizes is None else fusible(lhs, rhs, group_sizes)


def fused(region, *inputs):
    """`region(*inputs)`, a program's neighbourhood of one product written with `product` in place of its
    dot, its producer and consumers fused into the template's prologue and epilogue where the kernels take
    them tile by tile (`fuser.fuse`): the one entry a call site or a pass fuses a region through, so each
    stage is its selector's decision and report (`EPILOGUE`, `PROLOGUE`). A region whose products are all
    XLA's own (integer ones) has nothing to fuse and runs as written."""
    made: list = []
    token = _FUSIBLES.set(made)
    try:
        jax.eval_shape(region, *inputs)
    finally:
        _FUSIBLES.reset(token)
    return fuser.fuse(region)(*inputs) if made else region(*inputs)


_FUSIBLES: contextvars.ContextVar[list | None] = contextvars.ContextVar('fused_gemm_fusibles', default=None)
"""The fusible products a region makes while `fused` traces it."""


@dataclasses.dataclass(frozen=True)
class Analysis:
    """What `fused` makes of a region (`analysis`): each stage's route as its selector reported it
    (`generation.Route`: the kernel it takes, its refusal's reason where the kernels cannot take it
    tile by tile), and the product's launch's; and the region's seconds both ways on the target
    (`priced`): fused, its consumers in the launch's epilogue, and unfused, the product's launch then
    XLA's fusion of its consumers. None where the region has no consumer, or the target states no rate."""

    epilogue: generation.Route | None
    prologue: generation.Route | None
    launch: generation.Route | None
    fused_seconds: float | None = None
    unfused_seconds: float | None = None
    refused: str | None = None
    """Why a kernel the call names (`generation.chosen`) refuses the region, as `fused` would raise
    (`selection.KernelRefused`); None where nothing named refuses."""


def analysis(region, *inputs) -> Analysis:
    """`fused(region, *inputs)` traced for its shapes alone, nothing run: what its selectors decide, and
    its two prices (`priced`), so a pass fuses where the fused one is the less and no stage refused."""
    priced_calls: list = []
    token = _PRICED.set(priced_calls)
    try:
        # Its routes are its own, no caller's report's: what a pass only weighs is no decision a call took.
        with _cache.KERNEL_ROUTES(()), generation.routes() as found:
            jax.eval_shape(functools.partial(fused, region), *inputs)
    except KernelRefused as refusal:
        return Analysis(None, None, None, refused=str(refusal))
    finally:
        _PRICED.reset(token)

    def last(choice):
        return next((route for route in reversed(found) if route.choice == choice), None)
    seconds = priced(priced_calls[-1], generation.gpu_target()) if priced_calls else (None, None)
    return Analysis(last(EPILOGUE.choice), last(PROLOGUE.choice), last('grouped_matmul'), *seconds)


@dataclasses.dataclass(frozen=True)
class Region:
    """A traced region's product and consumers as `priced` reads them: the product's extents (rows, depth,
    width), its operands' and its result's dtypes, the consumers' elementwise work on the whole product
    (`_chain`), and the bytes of the arrays they capture and of their outputs."""

    rows: int
    depth: int
    width: int
    operands: jnp.dtype
    result: jnp.dtype
    chain: cost.Work
    captured_bytes: float
    output_bytes: float


_PRICED: contextvars.ContextVar[list | None] = contextvars.ContextVar('fused_gemm_priced', default=None)
"""The regions `_product` traces while `analysis` collects them."""


def priced(region: Region, target: generation.GpuTarget) -> tuple[float | None, float | None]:
    """`region`'s seconds on `target`, fused and unfused, from one model (`templates.cost`): the launch's
    units overlapped at the whole device's rates (its products in their tensor mode, its operands, captured
    arrays and outputs in device memory), plus its fixed launch (`GpuTarget.launch_seconds`), with its
    consumers' work in it; and the plain launch writing the product at its result dtype, then XLA's fusion
    of the consumers reading it (`algebra.cost.fused`). None where the target states no tensor rate."""
    rates = cost.rates(target, device=True)
    if rates[0] is None:
        return None, None
    product = algebra_cost.product_work(region.rows, region.depth, region.width, dtype=region.operands)
    operands = (region.rows + region.width) * region.depth * jnp.dtype(region.operands).itemsize
    written = region.rows * region.width * jnp.dtype(region.result).itemsize
    launch = target.launch_seconds or 0.0

    def launched(work: cost.Work) -> float:
        return max(cost.units(work, rates)[0]) + launch

    fused_seconds = launched(product + region.chain + cost.Work(
        device_bytes=operands + region.captured_bytes + region.output_bytes))
    chain = region.chain + cost.Work(device_bytes=written + region.captured_bytes + region.output_bytes,
                                     kernels=1.0)
    unfused_seconds = (launched(product + cost.Work(device_bytes=operands + written))
                       + algebra_cost.fused(target)(chain))
    return fused_seconds, unfused_seconds


TRANSCENDENTAL = frozenset({'exp', 'exp2', 'log', 'log1p', 'tanh', 'logistic', 'erf', 'sin', 'cos', 'pow',
                            'rsqrt', 'sqrt'})
"""The consumers' operations the special-function units take (`_chain`)."""


def _chain(consumer, values, acc) -> cost.Work:
    """The consumers' elementwise work on the whole product: each equation of their program an operation an
    element of its result, a transcendental's on the special-function units, a conversion's apart."""
    closed = jax.make_jaxpr(consumer)(values, acc)
    work = cost.Work()
    for eqn in closed.jaxpr.eqns:
        elements = float(sum(math.prod(var.aval.shape) for var in eqn.outvars if hasattr(var.aval, 'shape')))
        name = eqn.primitive.name
        work = work + (cost.Work(exponentials=elements) if name in TRANSCENDENTAL
                       else cost.Work(conversions=elements) if name == 'convert_element_type'
                       else cost.Work(operations=elements))
    return work


def _product(transpose_rhs: bool, interpret_on_cpu: bool, dense: bool, declared: tuple, *fusions):
    """The fusible product: its operands from their fusions, its consumers (`fusions`' last) as the
    template's epilogue where the kernels take them tile by tile, after its product in XLA otherwise,
    reading it as `declared` (its precision and result dtype, `product`)."""
    precision, out_dtype = declared
    *operand_fusions, out_fusion = fusions
    lhs_fusion, *rest_fusions = operand_fusions
    lhs, prologue = _prologue(lhs_fusion)
    rhs, *rest = (fusion() for fusion in rest_fusions)
    group_sizes = None if dense else rest[0]
    if out_fusion is None:
        return gemm.gemm(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, out_dtype=out_dtype,
                         precision=precision, interpret_on_cpu=interpret_on_cpu, **prologue)
    width = rhs.shape[-2] if transpose_rhs else rhs.shape[-1]
    acc = jax.ShapeDtypeStruct((lhs.shape[0], width), out_dtype)
    consumer, values, prefetch = fuser.get_fusion_values(out_fusion, acc)
    leaves, treedef = jax.tree.flatten(values)
    if (collected := _PRICED.get()) is not None:
        depth = lhs.shape[-1]
        outs = jax.tree.leaves(jax.eval_shape(consumer, values, acc))
        collected.append(Region(
            lhs.shape[0], depth, width, jnp.dtype(lhs.dtype), jnp.dtype(out_dtype),
            _chain(consumer, values, acc), float(sum(leaf.size * leaf.dtype.itemsize for leaf in leaves)),
            float(sum(math.prod(out.shape) * out.dtype.itemsize for out in outs))))
    axes = _stage(EPILOGUE, Refusal('scalar_prefetch', "the consumer reads scalar prefetch")
                  if prefetch else _axes(consumer, values, acc))
    if axes is None:
        summed = gemm.gemm(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, out_dtype=out_dtype,
                           precision=precision, interpret_on_cpu=interpret_on_cpu, **prologue)
        return consumer(values, summed)
    out_leaves, out_treedef = jax.tree.flatten(jax.eval_shape(consumer, values, acc))
    value_types = tuple(jax.typeof(leaf) for leaf in leaves)
    epilogue = functools.partial(_tiled, consumer, treedef, value_types, out_treedef, acc)
    dtypes = tuple(leaf.dtype for leaf in out_leaves)
    several = len(dtypes) > 1
    consumed = gemm.gemm(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, epilogue=epilogue,
                         epilogue_operands=tuple(zip(leaves, axes, strict=True)),
                         out_dtype=dtypes[0], out_dtypes=dtypes if several else None, precision=precision,
                         interpret_on_cpu=interpret_on_cpu, **prologue)
    return jax.tree.unflatten(out_treedef, consumed if several else (consumed,))


@dataclasses.dataclass(frozen=True)
class Tiling:
    """A stage as its selector reads it (`EPILOGUE`, `PROLOGUE`): what its analysis `found` (`_axes`,
    `_split`), a refusal where the kernels cannot take the stage tile by tile."""

    found: tuple | Refusal

    @property
    def refusal(self) -> Refusal | None:
        return self.found if isinstance(self.found, Refusal) else None


def _stage[S: tuple](op: Op[Tiling], found: S | Refusal) -> S | None:
    """The stage its analysis `found`, where `op`'s selector runs it in the kernel; None where it runs in
    XLA's program. The selector decides and reports either way: a refusal is a requirement the fused kernel
    fails, so it selects XLA, or raises where the call names the fused kernel."""
    match found:
        case Refusal():
            select(op, Tiling(found))
            return None
        case _:
            return found if select(op, Tiling(found)).kernel == 'fused' else None


EPILOGUE: Op[Tiling] = Op('gemm_epilogue', (
    Kernel('fused', (analysed('tiled', lambda case: case.refusal, FUSION_ROW),)),
    Kernel('xla', (), generic=True)))
"""The epilogue's selector: the consumers in the kernel's epilogue where the analysis takes them tile by
tile (`_axes`), else after the product in XLA."""

PROLOGUE: Op[Tiling] = Op('gemm_prologue', (
    Kernel('fused', (analysed('tiled', lambda case: case.refusal, FUSION_ROW),)),
    Kernel('xla', (), generic=True)))
"""The prologue's selector: the producer's element stage in the kernel's prologue where the analysis
takes it tile by tile (`_split`), else before the product in XLA."""


PROLOGUE_AXES = ('rows', 'depth')
"""The prologue's launch axes, in a tile's order."""


def _prologue(fusion) -> tuple[jax.Array, dict]:
    """The left operand's base array and the template's prologue arguments that give the operand from it
    (the module's docstring), or the operand itself and none where the kernels cannot take its producer,
    each reported ('gemm_prologue')."""
    producer, values, prefetch = fuser.get_fusion_values(fusion)
    split = (Refusal('scalar_prefetch', "the producer reads scalar prefetch") if prefetch
             else _split(producer, values))
    # An operand as given has no producer to fuse, and no stage to choose.
    if isinstance(split, Refusal) and split.code == 'no_producer':
        return fusion(), {}
    stage = _stage(PROLOGUE, split)
    if stage is None:
        return fusion(), {}
    base, function, sides, axes, jvp = stage
    return base, {'prologue': function, 'prologue_operands': tuple(zip(sides, axes, strict=True)),
                  'prologue_jvp': jvp}


def _split(producer, values):
    """`producer(values)`'s kernel stage and what it reads (`_prologue`): the base array, the prologue on its
    tile and the side arrays' tiles, the side arrays, the launch axes each follows, and the producer's whole
    JVP as the prologue's derivative; the refusal where the kernels cannot take it.

    The prologue's derivative is the whole producer's (`_whole_jvp`), the side arrays it computes (a row's
    statistics) taking none of their own: forward mode takes it, reverse mode transposes it and every
    higher order differentiates it (`gemm.gemm`'s `prologue_jvp`), so every path from an input joins
    where the producer joins it, at its precision, its custom derivatives kept, and none twice."""
    leaves, treedef = jax.tree.flatten(values)

    def flat(*leaves):
        return producer(jax.tree.unflatten(treedef, list(leaves)))

    # The producer as the algebra pass rewrites it, before the fuser stages it (docs/design/algebra.md §2).
    closed = jax.make_jaxpr(functools.partial(_inlined, jax.make_jaxpr(algebra.rewritten(flat))(*leaves)))(
        *leaves)
    jaxpr = closed.jaxpr
    [out] = jaxpr.outvars
    shape = tuple(getattr(out.aval, 'shape', ()))
    if len(shape) != 2 or not isinstance(out, jax_core.Var) or not jaxpr.eqns:
        return Refusal('no_producer', "the operand is an array as given")
    stage, inputs = _tile_stage(jaxpr, out, shape)
    full = [var for var in inputs if _shape(var) == shape and var in jaxpr.invars]
    if not full:
        return Refusal('no_base', "the operand's every element is computed, none loaded")
    base = full[0]
    if _dtype(base) != _dtype(out):
        return Refusal('dtype_changes', f"the producer turns its {_dtype(base)} base into "
                                                   f"{_dtype(out)}, which the kernels would multiply in")
    sides = [var for var in inputs if var is not base]
    given = {var: index for index, var in enumerate(jaxpr.invars)}
    read = {given[var] for var in (base, *sides) if var in given}
    if any(index not in read for index, leaf in enumerate(leaves)
           if jnp.issubdtype(jnp.result_type(leaf), jnp.inexact)):
        return Refusal('input_unstaged',
                                  "an input reaches the operand only through a row or depth value")
    # What the stage reads, each computed whole in XLA (the operand itself, unread, is left out there).
    arrays = jax_core.jaxpr_as_fun(jax_core.Jaxpr(jaxpr.replace(outvars=[base, *sides]),
                                                  closed.consts))(*leaves)
    whole = functools.partial(_staged, jax_core.Jaxpr(
        jax_core.Jaxpr(jaxpr.constvars, [base, *sides], [out], stage), closed.consts))
    axes = _prologue_axes(whole, arrays, shape)
    if isinstance(axes, Refusal):
        return axes
    squeezed = [array.reshape(tuple(size for size, axis in zip(array.shape, side_axes, strict=True)
                                    if axis is not None))
                for array, side_axes in zip(arrays[1:], axes, strict=True)]
    named = tuple(tuple(axis for axis in side_axes if axis is not None) for side_axes in axes)
    function = functools.partial(_prologue_tile, whole, tuple(jax.typeof(array) for array in arrays), axes)
    places = tuple(given.get(var) for var in (base, *sides))
    jvp = functools.partial(_whole_jvp, flat, tuple(leaves), places, tuple(array.shape for array in arrays))
    return arrays[0], function, squeezed, named, jvp


def _whole_jvp(flat, leaves: tuple, places: tuple, shapes: tuple, base, *sides):
    """The producer's own JVP as the prologue's derivative (`gemm.gemm`'s `prologue_jvp`): the
    template's base and side arrays put back among the producer's inputs (`places`, each one's index there,
    None for a computed one), and a function of the base's and each side array's tangent giving the
    operand's, a computed side's tangent unread (the producer's own derivative carries its statistics)."""
    inputs = list(leaves)
    for place, array, shape in zip(places, (base, *sides), shapes, strict=True):
        if place is not None:
            inputs[place] = array.reshape(shape)

    def pushed(d_base, *d_sides):
        tangents = [zero_from_primal(leaf) for leaf in inputs]
        for place, tangent, shape in zip(places, (d_base, *d_sides), shapes, strict=True):
            if place is not None:
                tangents[place] = tangent.reshape(shape)
        return jax.jvp(flat, tuple(inputs), tuple(tangents))[1]

    return pushed


def _inlined(closed, *args):
    """`closed` evaluated with every call (`dew.nn.programs.CALLS`: a jitted, checkpointed or
    custom-derivative call, as Dew's own norms are) read through to its body, so the split sees a norm's
    statistics however its producer wraps them: their values are their bodies'; their derivatives stay the
    producer's (`_whole_jvp`)."""
    return _evaluated(closed.jaxpr, closed.consts, args)


def _evaluated(program, consts, args):
    """`program` evaluated over `args` and its constants `consts`, each call it makes read through its own
    program (`dew.nn.programs.called`, its operands its program's one for one)."""
    env = dict(zip(program.invars, args, strict=True)) | dict(zip(program.constvars, consts, strict=True))

    def read(var):
        return var.val if isinstance(var, jax_core.Literal) else env[var]

    for eqn in program.eqns:
        body = programs.called(eqn)
        invals = [read(var) for var in eqn.invars]
        if body is not None:
            outvals = _evaluated(body, body.consts, invals)
        else:
            outvals = eqn.primitive.bind(*invals, **eqn.params)
            outvals = outvals if eqn.primitive.multiple_results else [outvals]
        env.update(zip(eqn.outvars, outvals, strict=True))
    return [read(var) for var in program.outvars]


def _shape(var) -> tuple:
    """A jaxpr variable's shape (an abstract array's)."""
    return tuple(getattr(var.aval, 'shape', ()))


def _dtype(var):
    """A jaxpr variable's dtype (an abstract array's)."""
    return getattr(var.aval, 'dtype', None)


def _staged(closed, *arrays):
    """The producer's kernel stage evaluated on whole arrays or on tiles (under `pull_block_spec`)."""
    return jax_core.jaxpr_as_fun(closed)(*arrays)[0]


VIEWS = frozenset({'broadcast_in_dim', 'reshape', 'squeeze', 'expand_dims', 'convert_element_type'})
"""The equations that only reshape, broadcast or cast one array: an input viewed through them is the
stage's to read (a norm's weight broadcast to a row), not a computed value."""


def _tile_stage(jaxpr, out, shape: tuple) -> tuple[list, list]:
    """The equations computing `out` element by element, backward from it until a variable is not of its
    `shape` (a row's or a depth's value, which XLA computes whole) or is an input, an input viewed through
    reshapes, broadcasts and casts (`VIEWS`) read through to it; and the stage's inputs, the producer's
    constants aside."""
    producers = {var: eqn for eqn in jaxpr.eqns for var in eqn.outvars}
    constants, given = set(jaxpr.constvars), set(jaxpr.invars)
    stage, inputs, seen = [], [], set()

    def viewed(var) -> list | None:
        """The view equations from an input to `var`, in order, or None where `var` is computed."""
        chain = []
        while var not in given:
            eqn = producers.get(var)
            if eqn is None or eqn.primitive.name not in VIEWS or not isinstance(eqn.invars[0], jax_core.Var):
                return None
            chain.append(eqn)
            var = eqn.invars[0]
        return chain[::-1]

    def visit(var):
        if not isinstance(var, jax_core.Var) or var in seen or var in constants:
            return
        seen.add(var)
        eqn = producers.get(var)
        if eqn is None or _shape(var) != shape:
            chain = None if eqn is None else viewed(var)
            if chain:
                visit(chain[0].invars[0])
                stage.extend(view for view in chain if view not in stage)
            else:
                inputs.append(var)
            return
        for operand in eqn.invars:
            visit(operand)
        if eqn not in stage:
            stage.append(eqn)

    visit(out)
    return stage, inputs


def _prologue_axes(whole, arrays: list, shape: tuple) -> tuple | Refusal:
    """The launch axes each side array follows, one per dimension (None for a dimension of one the tile
    broadcasts), from its block spec at a probe tile of the operand; the refusal where one follows none,
    or the producer reads its base at other than the tile or its tile's position."""
    probe = (_probe(shape[0]), _probe(shape[1]))
    spec = pl.BlockSpec(probe, lambda row, depth: (row, depth))
    pulled = _pulled(whole, spec, *arrays)
    if isinstance(pulled, Refusal):
        return pulled
    tiled, specs, _ = pulled
    if not _is_tile(specs[0], probe):
        return Refusal('base_untiled', "the producer reads its base at other than the tile")
    axes = []
    for array, side_spec in zip(arrays[1:], specs[1:], strict=True):
        block_shape, index_map = _block_and_map(side_spec)
        if block_shape is None or index_map is None:
            return Refusal('value_unblocked', f"a {array.shape} side array has no block")
        named = []
        for block, axis, size in zip(block_shape, _followed(index_map), array.shape, strict=True):
            if axis is not None and block == probe[axis] and size == shape[axis]:
                named.append(PROLOGUE_AXES[axis])
            elif size == 1 and block == 1:
                named.append(None)
            else:
                return Refusal(
                    'value_off_axes', f"a {array.shape} side array follows no launch axis tile by tile")
        if all(axis is None for axis in named):
            return Refusal('value_off_axes', f"a {array.shape} side array follows no launch axis")
        axes.append(tuple(named))
    blocks = [jax.ShapeDtypeStruct(block.block_shape, array.dtype)
              for block, array in zip(specs, arrays, strict=True)]
    ids = (jnp.int32(0), jnp.int32(0))
    closed = jax.make_jaxpr(lambda ids, *blocks: tiled(ids, (), *blocks))(ids, *blocks)
    if any(var in gemm._needed(closed.jaxpr, closed.jaxpr.outvars) for var in closed.jaxpr.invars[:2]):
        return Refusal('reads_position', "the producer reads its tile's position")
    return tuple(axes)


def _prologue_tile(whole, types: tuple, axes: tuple, tile, *side_tiles):
    """The producer's kernel stage on one loaded tile of its base and its side arrays' tiles beside it,
    each side tile given back the dimensions of one the template squeezed (`_split`), by a broadcast:
    Mosaic GPU's layout inference finds a layout for a broadcast's dimension of one, and none for a
    reshape's (FUSEDNORM-h100)."""
    shaped = [jax.lax.broadcast_in_dim(
        side, tuple(1 if axis is None else tile.shape[PROLOGUE_AXES.index(axis)] for axis in side_axes),
        tuple(index for index, axis in enumerate(side_axes) if axis is not None))
        for side, side_axes in zip(side_tiles, axes, strict=True)]
    tiled, _, _ = fuser.pull_block_spec(whole, pl.BlockSpec(tile.shape, lambda row, depth: (row, depth)),
                                        grid_len=2)(*types)
    return tiled((jnp.int32(0), jnp.int32(0)), (), tile, *shaped)


def _axes(consumer, values, acc) -> tuple | Refusal:
    """The launch axes each captured array follows, one per dimension, from its block spec at a probe
    tile; the refusal where the product's own block is not the tile at its place (a transpose), a
    captured array follows no launch axis (a scalar) or its block is not the tile's, or the consumer reads
    the tile's position."""
    probe = (_probe(acc.shape[0]), _probe(acc.shape[1]))
    pulled = _pulled(consumer, _specs(consumer, values, acc, probe), values, acc)
    if isinstance(pulled, Refusal):
        return pulled
    tiled, (value_specs, acc_spec), _ = pulled
    if not _is_tile(acc_spec, probe):
        return Refusal('product_untiled',
                                  "the consumer reads the product at other than its tile")
    axes = []
    specs = jax.tree.leaves(value_specs, is_leaf=_is_spec)
    for leaf, leaf_spec in zip(jax.tree.leaves(values), specs, strict=True):
        block_shape, index_map = _block_and_map(leaf_spec)
        if block_shape is None or index_map is None:
            return Refusal('value_unblocked', f"a captured {leaf.shape} array has no block")
        named = []
        for block, axis, size in zip(block_shape, _followed(index_map), leaf.shape, strict=True):
            if axis is None or block != probe[axis] or size != acc.shape[axis]:
                return Refusal(
                    'value_off_axes', f"a captured {leaf.shape} array follows no launch axis tile by tile")
            named.append(AXES[axis])
        axes.append(tuple(named))
    if _reads_position(tiled, value_specs, values, probe, acc.dtype):
        return Refusal('reads_position', "the consumer reads its tile's position")
    return tuple(axes)


def _pulled(function, specs, *args):
    """`function`'s block specs pulled back from `specs` at `args` (`fuser.pull_block_spec`), or the refusal
    where the fuser has no rule to tile one of its equations (an unsupported primitive, strides), which it
    states by NotImplementedError alone, naming the primitive (its public API has no query): XLA runs it
    whole, reported. Any other error the fuser raises is a fault, and propagates."""
    try:
        return fuser.pull_block_spec(function, specs, grid_len=2)(*args)
    except NotImplementedError as error:
        what = error.args[0] if error.args else "one of its equations"
        return Refusal('untileable', f"the fuser has no rule to tile {what}")


def _specs(consumer, values, acc, block: tuple):
    """One block spec of `block` at the tile's (row, column) for each of the consumer's outputs."""
    spec = pl.BlockSpec(block, lambda row, column: (row, column))
    return jax.tree.map(lambda _: spec, jax.eval_shape(consumer, values, acc))


def _is_tile(spec, probe: tuple) -> bool:
    """Whether `spec` is the probe tile at its own place: a block of `probe` at (row, column), its index map
    the identity (`_followed`)."""
    block_shape, index_map = _block_and_map(spec)
    return (block_shape is not None and index_map is not None and tuple(block_shape) == probe
            and _followed(index_map) == (0, 1))


def _followed(index_map) -> tuple[int | None, ...]:
    """Which launch axis each block index of `index_map` is, read off its program over the two launch
    indices: an index that is a launch index itself names its axis, any other (one computed from them, a
    constant) none. What it gives at one point proves nothing of the map: (row, column) -> (row * row,
    column) gives (1, 2) at (1, 2)."""
    closed = jax.make_jaxpr(index_map)(jnp.int32(0), jnp.int32(0))
    invars = closed.jaxpr.invars
    return tuple(next((axis for axis, var in enumerate(invars) if out is var), None)
                 for out in closed.jaxpr.outvars)


def _block_and_map(spec):
    """A block spec's block and index map; none for a spec that is not a block spec (a scalar's)."""
    if not isinstance(spec, pl.BlockSpec):
        return None, None
    return spec.block_shape, spec.index_map


def _probe(size: int) -> int:
    """A probe block of `size`: a power of two dividing it where one does, so the block spec is exact."""
    for block in (64, 32, 16, 8):
        if size % block == 0 and size > block:
            return block
    return size


def _is_spec(node) -> bool:
    return isinstance(node, pl.BlockSpec)


def _reads_position(tiled, value_specs, values, probe, dtype) -> bool:
    """Whether the tiled consumer's outputs depend on the program ids it is given (an iota, a position
    mask): the template's kernels place ragged tiles where no grid index says. Its product tile is of
    `dtype`, the product's declared one (`product`)."""
    tiles = jax.tree.map(lambda leaf, leaf_spec: jax.ShapeDtypeStruct(leaf_spec.block_shape, leaf.dtype),
                         values, value_specs, is_leaf=_is_spec)
    tile = jax.ShapeDtypeStruct(probe, dtype)
    closed = jax.make_jaxpr(lambda ids, tiles, tile: tiled(ids, (), tiles, tile))(
        (jnp.int32(0), jnp.int32(0)), tiles, tile)
    needed = gemm._needed(closed.jaxpr, closed.jaxpr.outvars)
    return any(var in needed for var in closed.jaxpr.invars[:2])


def _tiled(consumer, treedef, value_types: tuple, out_treedef, whole: jax.ShapeDtypeStruct, acc, *tiles):
    """The consumer on one fp32 accumulator tile of `acc.shape`, rounded first to the product's declared
    dtype (`whole`'s, the product it was traced at), and the captured arrays' tiles beside it: its block
    spec pulled at that block of `whole`, so the same function serves every kernel tile and XLA's whole
    array."""
    values = jax.tree.unflatten(treedef, [jax.ShapeDtypeStruct(t.shape, t.dtype) for t in value_types])
    tiled, _, _ = fuser.pull_block_spec(consumer, _specs(consumer, values, whole, acc.shape),
                                        grid_len=2)(values, whole)
    outs = tiled((jnp.int32(0), jnp.int32(0)), (), jax.tree.unflatten(treedef, list(tiles)),
                 acc.astype(whole.dtype))
    return tuple(jax.tree.leaves(outs)) if out_treedef.num_leaves > 1 else jax.tree.leaves(outs)[0]
