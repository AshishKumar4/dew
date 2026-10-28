"""The grouped GEMM template under `jax.experimental.pallas.fuser`: a call site's consumers of the
product become the template's epilogue, and its left operand's producer the template's prologue.

Inside a function `fuser.fuse` transforms, `fusible_gemm(lhs, rhs, group_sizes)` is `template.gemm`'s
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

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import fuser
from jax.extend import core as jax_core

from dew.nn.kernels import generation
from dew.nn.kernels.selection import Kernel, Op, Requirement, select
from dew.nn.kernels.templates.gemm import template

AXES = ('rows', 'columns')
"""The epilogue's launch axes, in a tile's order."""


def fusible_gemm(lhs: jax.Array, rhs: jax.Array, group_sizes: jax.Array | None = None, *,
                 transpose_rhs: bool = False, interpret_on_cpu: bool = False) -> jax.Array:
    """`template.gemm`'s fp32 product of each group's rows of `lhs` and its matrix of `rhs`, its consumers
    fused into the epilogue under `fuser.fuse` (the module's docstring)."""
    fusible = fuser.fusible(output_fusion_prefix=True)(
        functools.partial(_product, transpose_rhs, interpret_on_cpu, group_sizes is None))
    return fusible(lhs, rhs) if group_sizes is None else fusible(lhs, rhs, group_sizes)


def _product(transpose_rhs: bool, interpret_on_cpu: bool, dense: bool, *fusions):
    """The fusible product: its operands from their fusions, its consumers (`fusions`' last) as the
    template's epilogue where the kernels take them tile by tile, after its product in XLA otherwise."""
    *operand_fusions, out_fusion = fusions
    lhs_fusion, *rest_fusions = operand_fusions
    lhs, prologue = _prologue(lhs_fusion)
    rhs, *rest = (fusion() for fusion in rest_fusions)
    group_sizes = None if dense else rest[0]
    if out_fusion is None:
        return template.gemm(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, out_dtype=jnp.float32,
                             interpret_on_cpu=interpret_on_cpu, **prologue)
    width = rhs.shape[-2] if transpose_rhs else rhs.shape[-1]
    acc = jax.ShapeDtypeStruct((lhs.shape[0], width), jnp.float32)
    consumer, values, prefetch = fuser.get_fusion_values(out_fusion, acc)
    leaves, treedef = jax.tree.flatten(values)
    axes = _stage(EPILOGUE, generation.Refusal('scalar_prefetch', "the consumer reads scalar prefetch")
                  if prefetch else _axes(consumer, values, acc))
    if axes is None:
        summed = template.gemm(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, out_dtype=jnp.float32,
                               interpret_on_cpu=interpret_on_cpu, **prologue)
        return consumer(values, summed)
    out_leaves, out_treedef = jax.tree.flatten(jax.eval_shape(consumer, values, acc))
    value_types = tuple(jax.typeof(leaf) for leaf in leaves)
    epilogue = functools.partial(_tiled, consumer, treedef, value_types, out_treedef, acc.shape)
    dtypes = tuple(leaf.dtype for leaf in out_leaves)
    several = len(dtypes) > 1
    consumed = template.gemm(lhs, rhs, group_sizes, transpose_rhs=transpose_rhs, epilogue=epilogue,
                             epilogue_operands=tuple(zip(leaves, axes, strict=True)),
                             out_dtype=dtypes[0], out_dtypes=dtypes if several else None,
                             interpret_on_cpu=interpret_on_cpu, **prologue)
    return jax.tree.unflatten(out_treedef, consumed if several else (consumed,))


@dataclasses.dataclass(frozen=True)
class Tiling:
    """A stage as its selector reads it (`EPILOGUE`, `PROLOGUE`): what its analysis `found` (`_axes`,
    `_split`), a refusal where the kernels cannot take the stage tile by tile."""

    found: tuple | generation.Refusal

    @property
    def refusal(self) -> generation.Refusal | None:
        return self.found if isinstance(self.found, generation.Refusal) else None


def _stage[S: tuple](op: Op[Tiling], found: S | generation.Refusal) -> S | None:
    """The stage its analysis `found`, where `op`'s selector runs it in the kernel; None where it runs in
    XLA's program. The selector decides and reports either way: a refusal is a requirement the fused kernel
    fails, so it selects XLA, or raises where the call names the fused kernel."""
    match found:
        case generation.Refusal():
            select(op, Tiling(found))
            return None
        case _:
            return found if select(op, Tiling(found)).kernel == 'fused' else None


def _tileable(code: str) -> Requirement[Tiling]:
    """The requirement that a stage's analysis turned it down for no reason `code` names, its detail the
    analysis's."""
    return Requirement(code, lambda case: case.refusal is None or case.refusal.code != code,
                       lambda case: '' if case.refusal is None else case.refusal.detail, row='ledger 1295')


_SHARED = ('scalar_prefetch', 'value_unblocked', 'value_off_axes', 'reads_position', 'untileable')

EPILOGUE: Op[Tiling] = Op('gemm_epilogue', (
    Kernel('fused', tuple(_tileable(code) for code in ('product_untiled', *_SHARED))),
    Kernel('xla', (), generic=True)))
"""The epilogue's selector: the consumers in the kernel's epilogue where the analysis takes them tile by
tile (`_axes`), else after the product in XLA."""

PROLOGUE: Op[Tiling] = Op('gemm_prologue', (
    Kernel('fused', tuple(_tileable(code) for code in ('no_base', 'dtype_changes', 'input_unstaged',
                                                       'base_untiled', *_SHARED))),
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
    split = (generation.Refusal('scalar_prefetch', "the producer reads scalar prefetch") if prefetch
             else _split(producer, values))
    # An operand as given has no producer to fuse, and no stage to choose.
    if isinstance(split, generation.Refusal) and split.code == 'no_producer':
        return fusion(), {}
    stage = _stage(PROLOGUE, split)
    if stage is None:
        return fusion(), {}
    base, function, sides, axes, jvp = stage
    return base, {'prologue': function, 'prologue_operands': tuple(zip(sides, axes, strict=True)),
                  'prologue_jvp': jvp}


CALLS = frozenset({'pjit', 'jit', 'closed_call', 'core_call', 'remat', 'remat2', 'checkpoint',
                   'custom_jvp_call', 'custom_vjp_call', 'custom_vjp_call_jaxpr'})
"""The equations whose bodies the split reads through (a jitted, checkpointed or custom-derivative call, as
Dew's own norms are): their values are their bodies'; their derivatives stay the producer's (`_whole_jvp`)."""


def _split(producer, values):
    """`producer(values)`'s kernel stage and what it reads (`_prologue`): the base array, the prologue on its
    tile and the side arrays' tiles, the side arrays, the launch axes each follows, and the producer's whole
    JVP as the prologue's derivative; the refusal where the kernels cannot take it.

    The prologue's derivative is the whole producer's (`_whole_jvp`), the side arrays it computes (a row's
    statistics) taking none of their own: forward mode takes it, reverse mode transposes it and every
    higher order differentiates it (`template.gemm`'s `prologue_jvp`), so every path from an input joins
    where the producer joins it, at its precision, its custom derivatives kept, and none twice."""
    leaves, treedef = jax.tree.flatten(values)

    def flat(*leaves):
        return producer(jax.tree.unflatten(treedef, list(leaves)))

    closed = jax.make_jaxpr(functools.partial(_inlined, jax.make_jaxpr(flat)(*leaves)))(*leaves)
    jaxpr = closed.jaxpr
    [out] = jaxpr.outvars
    shape = tuple(getattr(out.aval, 'shape', ()))
    if len(shape) != 2 or not isinstance(out, jax_core.Var) or not jaxpr.eqns:
        return generation.Refusal('no_producer', "the operand is an array as given")
    stage, inputs = _tile_stage(jaxpr, out, shape)
    full = [var for var in inputs if _shape(var) == shape and var in jaxpr.invars]
    if not full:
        return generation.Refusal('no_base', "the operand's every element is computed, none loaded")
    base = full[0]
    if _dtype(base) != _dtype(out):
        return generation.Refusal('dtype_changes', f"the producer turns its {_dtype(base)} base into "
                                                   f"{_dtype(out)}, which the kernels would multiply in")
    sides = [var for var in inputs if var is not base]
    given = {var: index for index, var in enumerate(jaxpr.invars)}
    read = {given[var] for var in (base, *sides) if var in given}
    if any(index not in read for index, leaf in enumerate(leaves)
           if jnp.issubdtype(jnp.result_type(leaf), jnp.inexact)):
        return generation.Refusal('input_unstaged',
                                  "an input reaches the operand only through a row or depth value")
    # What the stage reads, each computed whole in XLA (the operand itself, unread, is left out there).
    arrays = jax_core.jaxpr_as_fun(jax_core.Jaxpr(jaxpr.replace(outvars=[base, *sides]),
                                                  closed.consts))(*leaves)
    whole = functools.partial(_staged, jax_core.Jaxpr(
        jax_core.Jaxpr(jaxpr.constvars, [base, *sides], [out], stage), closed.consts))
    axes = _prologue_axes(whole, arrays, shape)
    if isinstance(axes, generation.Refusal):
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
    """The producer's own JVP as the prologue's derivative (`template.gemm`'s `prologue_jvp`): the
    template's base and side arrays put back among the producer's inputs (`places`, each one's index there,
    None for a computed one), and a function of the base's and each side array's tangent giving the
    operand's, a computed side's tangent unread (the producer's own derivative carries its statistics)."""
    inputs = list(leaves)
    for place, array, shape in zip(places, (base, *sides), shapes, strict=True):
        if place is not None:
            inputs[place] = array.reshape(shape)

    def zero(leaf):
        if jnp.issubdtype(jnp.result_type(leaf), jnp.inexact):
            return jnp.zeros_like(leaf)
        return np.zeros(jnp.shape(leaf), jax.dtypes.float0)

    def pushed(d_base, *d_sides):
        tangents = [zero(leaf) for leaf in inputs]
        for place, tangent, shape in zip(places, (d_base, *d_sides), shapes, strict=True):
            if place is not None:
                tangents[place] = tangent.reshape(shape)
        return jax.jvp(flat, tuple(inputs), tuple(tangents))[1]

    return pushed


def _inlined(closed, *args):
    """`closed` evaluated with every call (`CALLS`) read through to its body, so the split sees a norm's
    statistics however its producer wraps them."""
    env = dict(zip(closed.jaxpr.invars, args, strict=True)) | dict(zip(closed.jaxpr.constvars, closed.consts,
                                                                       strict=True))

    def read(var):
        return var.val if isinstance(var, jax_core.Literal) else env[var]

    for eqn in closed.jaxpr.eqns:
        body = (eqn.params.get('jaxpr') or eqn.params.get('call_jaxpr') or eqn.params.get('fun_jaxpr')
                if eqn.primitive.name in CALLS else None)
        invals = [read(var) for var in eqn.invars]
        if body is not None:
            outvals = _inlined(body, *invals[len(invals) - len(body.jaxpr.invars):])
        else:
            outvals = eqn.primitive.bind(*invals, **eqn.params)
            outvals = outvals if eqn.primitive.multiple_results else [outvals]
        env.update(zip(eqn.outvars, outvals, strict=True))
    return [read(var) for var in closed.jaxpr.outvars]


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


def _prologue_axes(whole, arrays: list, shape: tuple) -> tuple | generation.Refusal:
    """The launch axes each side array follows, one per dimension (None for a dimension of one the tile
    broadcasts), from its block spec at a probe tile of the operand; the refusal where one follows none,
    or the producer reads its base at other than the tile or its tile's position."""
    probe = (_probe(shape[0]), _probe(shape[1]))
    spec = pl.BlockSpec(probe, lambda row, depth: (row, depth))
    pulled = _pulled(whole, spec, *arrays)
    if isinstance(pulled, generation.Refusal):
        return pulled
    tiled, specs, _ = pulled
    if not _is_tile(specs[0], probe):
        return generation.Refusal('base_untiled', "the producer reads its base at other than the tile")
    axes = []
    for array, side_spec in zip(arrays[1:], specs[1:], strict=True):
        block_shape, index_map = _block_and_map(side_spec)
        if block_shape is None or index_map is None:
            return generation.Refusal('value_unblocked', f"a {array.shape} side array has no block")
        named = []
        for block, index, size in zip(block_shape, index_map(1, 2), array.shape, strict=True):
            if index in (1, 2) and block == probe[index - 1] and size == shape[index - 1]:
                named.append(PROLOGUE_AXES[index - 1])
            elif size == 1 and block == 1:
                named.append(None)
            else:
                return generation.Refusal(
                    'value_off_axes', f"a {array.shape} side array follows no launch axis tile by tile")
        if all(axis is None for axis in named):
            return generation.Refusal('value_off_axes', f"a {array.shape} side array follows no launch axis")
        axes.append(tuple(named))
    blocks = [jax.ShapeDtypeStruct(block.block_shape, array.dtype)
              for block, array in zip(specs, arrays, strict=True)]
    ids = (jnp.int32(0), jnp.int32(0))
    closed = jax.make_jaxpr(lambda ids, *blocks: tiled(ids, (), *blocks))(ids, *blocks)
    if any(var in template._needed(closed.jaxpr, closed.jaxpr.outvars) for var in closed.jaxpr.invars[:2]):
        return generation.Refusal('reads_position', "the producer reads its tile's position")
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


def _axes(consumer, values, acc) -> tuple | generation.Refusal:
    """The launch axes each captured array follows, one per dimension, from its block spec at a probe
    tile; the refusal where the product's own block is not the tile at its place (a transpose), a
    captured array follows no launch axis (a scalar) or its block is not the tile's, or the consumer reads
    the tile's position."""
    probe = (_probe(acc.shape[0]), _probe(acc.shape[1]))
    pulled = _pulled(consumer, _specs(consumer, values, acc, probe), values, acc)
    if isinstance(pulled, generation.Refusal):
        return pulled
    tiled, (value_specs, acc_spec), _ = pulled
    if not _is_tile(acc_spec, probe):
        return generation.Refusal('product_untiled',
                                  "the consumer reads the product at other than its tile")
    axes = []
    specs = jax.tree.leaves(value_specs, is_leaf=_is_spec)
    for leaf, leaf_spec in zip(jax.tree.leaves(values), specs, strict=True):
        block_shape, index_map = _block_and_map(leaf_spec)
        if block_shape is None or index_map is None:
            return generation.Refusal('value_unblocked', f"a captured {leaf.shape} array has no block")
        named = []
        for block, index, size in zip(block_shape, index_map(1, 2), leaf.shape, strict=True):
            if index not in (1, 2) or block != probe[index - 1] or size != acc.shape[index - 1]:
                return generation.Refusal(
                    'value_off_axes', f"a captured {leaf.shape} array follows no launch axis tile by tile")
            named.append(AXES[index - 1])
        axes.append(tuple(named))
    if _reads_position(tiled, value_specs, values, probe):
        return generation.Refusal('reads_position', "the consumer reads its tile's position")
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
        return generation.Refusal('untileable', f"the fuser has no rule to tile {what}")


def _specs(consumer, values, acc, block: tuple):
    """One block spec of `block` at the tile's (row, column) for each of the consumer's outputs."""
    spec = pl.BlockSpec(block, lambda row, column: (row, column))
    return jax.tree.map(lambda _: spec, jax.eval_shape(consumer, values, acc))


def _is_tile(spec, probe: tuple) -> bool:
    """Whether `spec` is the probe tile at its own place: a block of `probe` at (row, column)."""
    block_shape, index_map = _block_and_map(spec)
    return block_shape is not None and index_map is not None and tuple(block_shape) == probe and tuple(
        index_map(1, 2)) == (1, 2)


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


def _reads_position(tiled, value_specs, values, probe) -> bool:
    """Whether the tiled consumer's outputs depend on the program ids it is given (an iota, a position
    mask): the template's kernels place ragged tiles where no grid index says."""
    tiles = jax.tree.map(lambda leaf, leaf_spec: jax.ShapeDtypeStruct(leaf_spec.block_shape, leaf.dtype),
                         values, value_specs, is_leaf=_is_spec)
    tile = jax.ShapeDtypeStruct(probe, jnp.float32)
    closed = jax.make_jaxpr(lambda ids, tiles, tile: tiled(ids, (), tiles, tile))(
        (jnp.int32(0), jnp.int32(0)), tiles, tile)
    needed = template._needed(closed.jaxpr, closed.jaxpr.outvars)
    return any(var in needed for var in closed.jaxpr.invars[:2])


def _tiled(consumer, treedef, value_types: tuple, out_treedef, whole: tuple, acc, *tiles):
    """The consumer on one accumulator tile of `acc.shape` and the captured arrays' tiles beside it: its
    block spec pulled at that block of the `whole` product it was traced at, so the same function serves
    every kernel tile and XLA's whole array."""
    values = jax.tree.unflatten(treedef, [jax.ShapeDtypeStruct(t.shape, t.dtype) for t in value_types])
    whole_acc = jax.ShapeDtypeStruct(whole, jnp.float32)
    tiled, _, _ = fuser.pull_block_spec(consumer, _specs(consumer, values, whole_acc, acc.shape),
                                        grid_len=2)(values, whole_acc)
    outs = tiled((jnp.int32(0), jnp.int32(0)), (), jax.tree.unflatten(treedef, list(tiles)), acc)
    return tuple(jax.tree.leaves(outs)) if out_treedef.num_leaves > 1 else jax.tree.leaves(outs)[0]
