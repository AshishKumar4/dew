"""Dew: a GEMM launch's hooks (`templates.gemm`): its prologue and epilogue, each a JAX function of a tile and
its side tiles, as `grouped_matmul`'s launches bind them and their kernels trace them (`Hooks`)."""

import dataclasses
import functools
from collections.abc import Callable
from typing import NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax._src.pallas.mosaic import core as tpu_core, lowering as mosaic_tpu_lowering
from jax._src.pallas.mosaic_gpu import (
    core as plgpu_core,  # the lowering's semantics (`_lowers`)
    lowering as mosaic_gpu_lowering,
)
from jax.ad_checkpoint import checkpoint_name

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


@runtime_checkable
class Residualized(Protocol):
    """An epilogue whose first `residuals` outputs a reverse pass alone reads (`wide.Gated`'s saved rounded
    product): `gemm` writes them where a derivative is taken, under their names (its `named`), returns the
    rest, and runs `unsaved()`, the epilogue without them, where none is (`linear_tangent`'s `primal`), so
    an inference's launch stores none."""

    @property
    def residuals(self) -> int: ...

    def unsaved(self) -> Callable: ...


def kept_outputs(named: tuple, residuals: int = 0) -> tuple[int, ...]:
    """The outputs `gemm`'s forward saves, by position: each `named` names (`checkpoint_name`) and the first
    `residuals` (`Residualized`'s), named or not."""
    return tuple(at for at in range(max(len(named), residuals))
                 if at < residuals or named[at] is not None)


def held_outputs(out, named: tuple, several: bool, residuals: int = 0) -> tuple:
    """`gemm`'s forward's outputs `out` (several, or one) and what it saves (`kept_outputs`): each output `named`
    names a residual under its name, the one array the output and the residual are; the first `residuals`
    (`Residualized`'s) saved, named or not, and never returned."""
    outs = list(out if several else (out,))
    names = tuple(named) + (None,) * (len(outs) - len(named))
    outs = [checkpoint_name(each, name) if name is not None else each
            for each, name in zip(outs, names, strict=True)]
    saved = tuple(outs[at] for at in kept_outputs(named, residuals))
    return (tuple(outs[residuals:]) if several else outs[0]), saved


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


@dataclasses.dataclass(frozen=True)
class ColumnLocal:
    """An epilogue or prologue `hook` its caller declares column-local: evaluated on any slice of its tile's
    columns (and its side tiles' same columns) it computes that slice of its result, its extents read
    nowhere, and so does its pullback where a reverse pass forms it (`gemm._prologue_pulled`, which declares
    its prologue's VJP only where the prologue is declared; `activations.gated_hook`, entrywise). A launch
    reads such an epilogue's tile out a chunk of columns at a time (`read_out`, `pricing.hook_chunk`); any
    other its issuer's own read-out (a register accumulator's whole, tcgen05's 64 columns)."""

    hook: Callable

    def __call__(self, *tiles):
        return self.hook(*tiles)


def column_local(hook) -> bool:
    """Whether `hook` is declared column-local (`ColumnLocal`) as a launch holds it: through the form every
    launch takes it in (`tuning.Prepared`'s source; the algebra's rewrite computes its function) and its side
    tiles' dimensions of one given back (`_unsqueezed`, across whatever columns its tile has)."""
    from dew.nn.kernels import tuning

    while not isinstance(hook, ColumnLocal):
        if isinstance(hook, tuning.Prepared):
            hook = hook.source
        elif isinstance(hook, functools.partial) and hook.func is _unsqueezed:
            hook = hook.args[0]
        else:
            return False
    return True


class ReadOut(NamedTuple):
    """How a launch's epilogue reads its tile out (`read_out`): each output's divisor where its form narrows
    it, 1 elsewhere (`narrowed`; a tile's own columns by `mosaic_gpu._gmm_finished`), and the columns its
    hook takes at a time (`columns`, `pricing.hook_chunk`), None for its issuer's own read-out."""

    narrowed: tuple = ()
    columns: int | None = None


NO_READ_OUT = ReadOut()
"""A launch's read-out where its caller states none: no output narrowed, its issuer's own chunks."""


def read_out(family, part: tuple[int, int], dtypes: tuple, forms: tuple, staging, epilogue) -> ReadOut:
    """A launch's `ReadOut` on `family` at a consumer's `part` of its tile (rows, columns), its outputs'
    `dtypes` and `forms` (none where empty), its `staging`, and its `epilogue` (a `ColumnLocal` one read out
    a chunk at a time): the one decision its kernel (`mosaic_gpu._finished`) and its register model
    (`grouped_matmul._epilogue_registers`) both read."""
    from dew.nn.kernels.templates.gemm import pricing

    forms = tuple(forms) or (None,) * len(dtypes)
    narrowed = tuple(1 if form is None or not form.narrowed else form.divisor for form in forms)
    if epilogue is None:
        return ReadOut(narrowed)
    return ReadOut(narrowed, pricing.hook_chunk(family, part[0], part[1], dtypes, forms, staging,
                                                local=column_local(epilogue)))


def lowering_rules(target) -> dict:
    """The rules a hook's operations need on `target`'s Mosaic launches: a TPU core's TensorCore rules; a
    GPU's at warpgroup semantics."""
    if generation.tpu(target):
        return mosaic_tpu_lowering.lowering_rules[tpu_core.CoreType.TC]
    return mosaic_gpu_lowering.mosaic_lowering_rules[plgpu_core.WGxWG_SEMANTICS]


def hook_lowers(hook, tiles: tuple, target) -> bool:
    """Whether `target`'s Mosaic launches take `hook` on `tiles` (`_lowers`, `lowering_rules`): a caller's
    hook asked before it is bound to a launch (`dew.nn.gated_mlp`'s gated product)."""
    dtype = tiles[0].dtype if tiles else jnp.bfloat16
    return _lowers(hook, tiles, (), lowering_rules(target), layout=_epilogue_layout(dtype, target))


def hooks_lower(ba, dtype, target) -> bool:
    """Whether `target`'s Mosaic launches take a bound launch's hooks (`_lowers`, `lowering_rules`), traced on
    128-wide tiles: the primitives a hook makes do not depend on its tile's extents."""
    rules = lowering_rules(target)
    hooks = ba.arguments['hooks']

    def tiles(name, axes_of):
        return [jax.ShapeDtypeStruct((128,) * len(axes), side.dtype)
                for side, axes in zip(ba.arguments.get(name, ()), axes_of, strict=True)]

    return (_lowers(hooks.before(), (jax.ShapeDtypeStruct((128, 128), dtype),),
                    tiles('prologue_operands', hooks.prologue_axes), rules, layout=None)
            and _lowers(hooks.after(), (jax.ShapeDtypeStruct((128, 128), jnp.float32),) * hooks.accumulators,
                        tiles('epilogue_operands', hooks.epilogue_axes), rules,
                        layout=_epilogue_layout(dtype, target)))
