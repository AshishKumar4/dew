"""Dew: a GEMM launch's hooks (`templates.gemm`): its prologue and epilogue, each a JAX function of a tile and
its side tiles, as `grouped_matmul`'s launches bind them and their kernels trace them (`Hooks`)."""

import dataclasses
import functools
from typing import NamedTuple, Protocol, runtime_checkable

import jax
import jax.extend.core as jax_core
import jax.numpy as jnp
import numpy as np
from jax._src.pallas.mosaic import core as tpu_core, lowering as mosaic_tpu_lowering
from jax._src.pallas.mosaic_gpu import (
    core as plgpu_core,  # the lowering's semantics (`_lowers`)
    lowering as mosaic_gpu_lowering,
)

from dew.nn import programs
from dew.nn.derivatives import marked_call_p
from dew.nn.kernels import generation, launches
from dew.nn.programs import equations


@dataclasses.dataclass(frozen=True)
class Hooks:
    """A launch's prologue and epilogue (`templates.gemm`): each a JAX function of a tile and its side
    tiles, or None, with each side operand's launch axes ('rows', 'depth', 'columns'), the positions of
    each side operand's dimensions of one the launch squeezed out (the hook gets them back), the dtypes of
    the epilogue's outputs where it returns several, and the dtypes a kernel traces each hook at (its tile's
    then each side tile's: `prologue_dtypes`, `epilogue_dtypes`). The side operands themselves are the
    launch's `prologue_operands` and `epilogue_operands`. A launch's picks key each hook by what it
    computes (`_GroupedLaunch.traced`), never by the function object. A `paired` product's epilogue takes its
    two accumulators before its side tiles (`gemm.gemm`'s `paired`)."""
    prologue: object = None
    prologue_axes: tuple = ()
    epilogue: object = None
    epilogue_axes: tuple = ()
    out_dtypes: tuple | None = None
    prologue_broadcast: tuple = ()
    epilogue_broadcast: tuple = ()
    prologue_dtypes: tuple = ()
    epilogue_dtypes: tuple = ()
    accumulators: int = 1
    """The accumulator tiles the epilogue takes before its side tiles: a launch's binding states them from
    its bound operands (two where a second matrix is bound, `gmm_call`'s `paired`; `_GroupedLaunch.bind`),
    never a caller."""
    out_forms: tuple | None = None
    """How each output stores, where the epilogue returns several (`out_dtypes`): None for a tile of its own,
    else `(divisor, transposed)`: one value a row each `divisor` columns, stored `[columns // divisor, rows]`
    where transposed (`stores.Output`; a next product's 1 x `divisor` scales, its rows contiguous, as a
    blockwise FP8 product reads them; row 8547). Its epilogue returns such an output broadcast across each
    of its blocks' columns, so a block's row maximum is its value (`quantized.reduced`)."""
    depth_masked: bool = False
    """Whether the prologue's result is zeroed past the call's depth by its last side vector
    (`mosaic_gpu.depth_masked`), which a launch's alignment binds once (`MosaicGmm._aligned`)."""

    def before(self):
        """The prologue as a kernel traces it: on the squeezed side tiles."""
        return _unsqueezing(self.prologue, self.prologue_broadcast)

    def after(self):
        """The epilogue as a kernel traces it: on the squeezed side tiles."""
        return _unsqueezing(self.epilogue, self.epilogue_broadcast, self.accumulators)

    def kernel_options(self) -> dict:
        return {'prologue': self.before(), 'prologue_axes': self.prologue_axes, 'epilogue': self.after(),
                'epilogue_axes': self.epilogue_axes, 'out_dtypes': self.out_dtypes,
                **({'out_forms': self.out_forms} if self.out_forms else {})}


@runtime_checkable
class Formed(Protocol):
    """An epilogue that states how each of its outputs stores (`Hooks.out_forms`, `quantized.Form`): one
    whose outputs are narrower than its tile (`wide.Gated`'s gated product), which a launch's declared
    outputs and stores read before it is traced."""

    @property
    def out_forms(self) -> tuple: ...


def _unsqueezing(hook, positions: tuple, tiles: int = 1):
    """`hook` given its side tiles' dimensions of one back at `positions` (one tuple per side operand), its
    first `tiles` arguments the launch's tiles."""
    if hook is None or not any(positions):
        return hook
    return functools.partial(_unsqueezed, hook, positions, tiles)


def _unsqueezed(hook, positions: tuple, tiles: int, *arguments):
    """`hook` on its first `tiles` arguments and the side tiles after them with their dimensions of one
    back at `positions`, broadcast across its first tile's extent there (aligned to its trailing axes, as
    the hook's own arithmetic broadcasts them): Mosaic's layout inference lays out a broadcast of a held
    vector to a tile, as a side by group's (`axes.by_rows`), and no dimension of one (GMMGATE-h100-8)."""
    shape = arguments[0].shape

    def restored(side, where):
        if not where:
            return side
        rank = side.ndim + len(where)
        offset = len(shape) - rank
        if offset < 0:
            raise ValueError(f"a side tile of {rank} dimensions given back beside a tile of {len(shape)}")
        kept = [axis for axis in range(rank) if axis not in where]
        target = [0] * rank
        for axis in where:
            target[axis] = shape[offset + axis]
        for place, axis in enumerate(kept):
            target[axis] = side.shape[place]
        return jax.lax.broadcast_in_dim(side, tuple(target), tuple(kept))

    sides = arguments[tiles:]
    return hook(*arguments[:tiles], *(restored(side, where)
                                      for side, where in zip(sides, positions, strict=True)))


NO_HOOKS = Hooks()


def _epilogue_layout(dtype, target):
    """Accumulator register layout of `target`'s primary GPU bundle for `dtype`, or None on TPU/unbudgeted."""
    if generation.tpu(target):
        return None
    from dew.nn.kernels.templates import bundles

    groups = bundles.groups(target.generation, dtype, stated=True, tpu=False)
    if not groups or not groups[0]:
        return None
    family = groups[0][0]
    return family.layout if family.budgets_registers else None

def _lowers(hook, tiles: tuple, sides, rules, *, layout=None) -> bool:
    """Whether a Pallas lowering's `rules` hold a rule for every primitive of `hook` on its `tiles` and its
    side tiles: Mosaic GPU's, at warpgroup semantics, have none for a transpose, which Triton's lowering
    takes. A marked span (`derivatives.marked_call_p`) lowers as its body, whose primitives are read."""
    if hook is None:
        return True
    from dew.nn.kernels.templates.gemm import quantized

    # Traced as a launch's body traces it, on tiles (`launches.tracing_body`).
    with launches.tracing_body(interpret=False), quantized.epilogue_layout(layout):
        traced = jax.make_jaxpr(hook)(*tiles, *sides)
    return all(equation.primitive in rules or equation.primitive is marked_call_p
               for equation in equations(traced.jaxpr))


PROBE_ROWS, PROBE_COLUMNS = 8, 24
"""The extents a hook's locality is read at (`column_local`), apart so a vector's axis reads off its
length."""

SHAPE_PARAMS = frozenset({'shape', 'new_sizes', 'sizes', 'sharding', 'out_sharding'})
"""The parameters an equation states its result's extents in, which a hook traced at two widths differs in
alone (`column_local`)."""


def _traced(hook, accumulators: int, sides: tuple, rows: int, columns: int):
    """`hook` traced on `[rows, columns]` accumulators and its side tiles (`sides`, each its axes and dtype)
    as an epilogue takes them (`axes.by_rows`: along rows or groups, `rows`; along columns, `columns`), None
    where a side's axes name another extent or the hook does not trace."""
    extents = {'rows': rows, 'groups': rows, 'columns': columns}
    try:
        return jax.make_jaxpr(hook)(
            *(jax.ShapeDtypeStruct((rows, columns), jnp.float32),) * accumulators,
            *(jax.ShapeDtypeStruct(tuple(extents[axis] for axis in side_axes), dtype)
              for side_axes, dtype in sides))
    except (KeyError, TypeError, ValueError):
        return None


def _entrywise(jaxpr, rows: int, columns: int) -> bool:
    """Whether every equation of `jaxpr` (traced at `rows` by `columns`, apart), nested ones too, is
    entrywise on tiles, row vectors, column vectors or scalars, or broadcasts a scalar, a column vector along
    the rows or a row vector along the columns."""
    allowed = {(rows, columns), (rows,), (columns,), ()}
    for equation in jaxpr.eqns:
        if any(getattr(variable.aval, 'shape', None) not in allowed
               for variable in (*equation.invars, *equation.outvars)):
            return False
        nested = list(jax_core.jaxprs_in_params(equation.params))
        if nested:
            if programs.control_flow(equation.primitive) or not all(
                    _entrywise(inner, rows, columns) for inner in nested):
                return False
        elif equation.primitive is jax.lax.broadcast_in_dim_p:
            operand = equation.invars[0].aval.shape
            dimensions = tuple(equation.params['broadcast_dimensions'])
            if (operand, dimensions) not in {((), ()), ((rows,), (0,)), ((columns,), (1,)),
                                             ((rows, columns), (0, 1))}:
                return False
        elif not programs.entrywise(equation.primitive):
            return False
    return True


def _same(first, second) -> bool:
    """Whether two traces of one hook at two extents compute alike: equation for equation the same
    primitive, the same parameters but the extents they state (`SHAPE_PARAMS`), nested ones alike, the same
    literals and the same constants; a hook reading its tile's extent (scaling by its width) differs."""
    if len(first.eqns) != len(second.eqns) or len(first.constvars) != len(second.constvars):
        return False
    for one, other in zip(first.eqns, second.eqns, strict=True):
        if one.primitive is not other.primitive or set(one.params) != set(other.params):
            return False
        for name, value in one.params.items():
            if name in SHAPE_PARAMS:
                continue
            twin = other.params[name]
            inner = list(jax_core.jaxprs_in_params({name: value}))
            if inner:
                if not all(_same(a, b) for a, b in zip(inner, jax_core.jaxprs_in_params({name: twin}),
                                                       strict=True)):
                    return False
            elif value != twin:
                return False
        for a, b in zip(one.invars, other.invars, strict=True):
            literal = isinstance(a, jax_core.Literal)
            if literal != isinstance(b, jax_core.Literal) or (literal and not np.array_equal(a.val, b.val)):
                return False
    return True


@functools.lru_cache(maxsize=256)
def column_local(hook, accumulators: int, sides: tuple, rows: int, columns: int, chunk: int) -> bool:
    """Whether an epilogue `hook` computes each output column from that column of its `accumulators` and its
    side tiles alone (`sides`, each its axes and dtype), so a `[rows, columns]` tile may be read out
    `chunk` columns at a time (`pricing.hook_chunk`): traced at a probe's extents its every equation
    entrywise (`_entrywise`: none reduces, gathers, reverses or numbers the columns), and traced at the
    tile's and the chunk's widths alike (`_same`: none reads its width). A hook subtracting its rows'
    maxima, or scaling by its tile's width, is not; one whose constants do not trace is not."""
    probe = _traced(hook, accumulators, sides, PROBE_ROWS, PROBE_COLUMNS)
    whole = _traced(hook, accumulators, sides, rows, columns)
    part = _traced(hook, accumulators, sides, rows, chunk)
    if probe is None or whole is None or part is None or any(
            np.ndim(const) for const in (*probe.consts, *whole.consts, *part.consts)):
        return False
    return (_entrywise(probe.jaxpr, PROBE_ROWS, PROBE_COLUMNS) and _same(probe.jaxpr, whole.jaxpr)
            and _same(whole.jaxpr, part.jaxpr)
            and all(np.array_equal(a, b) for a, b in zip(whole.consts, part.consts, strict=True)))


class ReadOut(NamedTuple):
    """How a launch's epilogue reads its tile out (`read_out`): each output's divisor where its form narrows
    it, 1 elsewhere (`narrowed`; a tile's own columns by `mosaic_gpu._gmm_finished`), and the columns its
    hook takes at a time (`columns`, `pricing.hook_chunk`), None for its issuer's own read-out."""

    narrowed: tuple = ()
    columns: int | None = None


def read_out(family, part: tuple[int, int], dtypes: tuple, forms: tuple, staging, accumulators: int,
             epilogue, side_axes: tuple, side_operands: tuple) -> ReadOut:
    """A launch's `ReadOut` on `family` at a consumer's `part` of its tile (rows, columns), its outputs'
    `dtypes` and `forms` (none where empty), its `staging`, and its `epilogue` over `accumulators` and its
    side operands (`side_operands`, each along its `side_axes`): the one decision its kernel
    (`mosaic_gpu._finished`) and its register model (`grouped_matmul._epilogue_registers`) both read."""
    from dew.nn.kernels.templates.gemm import pricing

    forms = tuple(forms) or (None,) * len(dtypes)
    narrowed = tuple(1 if form is None or not form.narrowed else form.divisor for form in forms)
    if epilogue is None:
        return ReadOut(narrowed)
    sides = tuple((tuple(axes), side.dtype) for axes, side in zip(side_axes, side_operands, strict=True))
    local = functools.partial(column_local, epilogue, accumulators, sides, *part)
    return ReadOut(narrowed, pricing.hook_chunk(family, part[0], part[1], dtypes, forms, staging,
                                                local=local))
