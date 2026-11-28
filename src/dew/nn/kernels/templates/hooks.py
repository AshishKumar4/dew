"""Hooks (kernel-templates.md §4): a traced JAX callable and its typed side operands, the one way neighbouring
work joins a template's kernel body. Attention's score, output and prologue hooks, a GEMM's prologue and
epilogue, and the stages the Pallas fuser lifts off a product (`dew.nn.kernels.fused_gemm`) are each a `Hook`:
its function (`Hook.fn`, static) and its side operands (`Side`: an array and each of its dimensions' axis by
launch name, `templates/axes.py`'s vocabulary). A hook is a pytree: a launch flattens it, its leaves the side
operands it loads tiles of, its structure the static hook its kernel traces. What a template's site gives a
hook (the tile, its indices, the side tiles laid as `laid` lays them) is the template's; what a hook computes
is its traced program (`tuning.program`, at `TRACED_TILE`), which keys every launch that takes it, and whether
a launch's kernel takes it is whether its lowering has a rule for each of its primitives (`lowers`)."""

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
from jax._src.pallas.mosaic import core as tpu_core, lowering as mosaic_tpu_lowering
from jax._src.pallas.mosaic_gpu import core as plgpu_core, lowering as mosaic_gpu_lowering

from dew.nn.derivatives import marked_call_p
from dew.nn.kernels import generation, launches
from dew.nn.programs import equations

TRACED_TILE = 16
"""The extent a hook's tile and side tiles are traced at for its identity (`tuning.program`): a launch's keys
and the kernels it names read what a hook computes there, which no tile's extent changes."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Side:
    """A hook's side operand: its array (`value`) and the axis each of its dimensions follows (`axes`,
    static), by launch name (`templates/axes.py`: a GEMM's 'rows', 'depth', 'columns' and 'groups',
    attention's 'batch', 'heads', 'queries', 'keys' and 'width', its `Kind`s), None for a dimension of one.
    A launch reads the tile of it at its own tile's origins along those axes."""

    value: Any
    axes: tuple = dataclasses.field(metadata={"static": True})


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Hook:
    """A hook: `fn` (static), a JAX function of its site's tile, the site's own arguments (attention's
    indices) and a tile of each of its side operands (`sides`, in their order), traced into the kernel's
    body; `fn` None, no hook (`NO_HOOK`), whatever its site. Its identity is what it computes
    (`tuning.program`), never the function object."""

    fn: Callable | None = dataclasses.field(default=None, metadata={"static": True})
    sides: tuple[Side, ...] = ()

    def __post_init__(self):
        if self.fn is None and self.sides:
            raise ValueError("a hook's side operands take its function, none given")


NO_HOOK = Hook()
"""No hook at a site."""


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class RowHook:
    """A row hook: an epilogue a launch runs over each of its tile's rows whole, where its tile's columns come
    a block at a time (a recurrence's output launch's value blocks, `templates.recurrence.output`): its
    `statistic(block_tile, *read_tiles) -> [rows]`, fp32 partials the launch sums across its walk's steps
    (summation the only combiner: another combiner is another seam), and its `fn(block_tile, total,
    *side_tiles) -> out_tile`, applied to each block past the walk with every block's total. Both static, as
    `Hook.fn`, its `sides` (`Side`s on the tile's 'rows' and 'columns') loaded at each block. The launch owns
    the carry, the staging and the read past the walk; a row hook holds no launch's facts. A launch keys its
    build by the two functions (each a frozen dataclass of its form, so two forms key apart)."""

    statistic: Callable = dataclasses.field(metadata={"static": True})
    fn: Callable = dataclasses.field(metadata={"static": True})
    sides: tuple[Side, ...] = ()
    reads: tuple[int, ...] = dataclasses.field(default=(), metadata={"static": True})
    """The sides its statistic reads, by their places in `sides` (its `fn` reads every one): a launch loads
    no other at a step."""

    TILE = ('rows', 'columns')
    """The axes of a row hook's tile: a block's rows and columns."""


def laid(tile: Sequence[str], axes: Sequence[str | None], extents: Mapping[str, int]) -> tuple[int, ...]:
    """The shape of a tile of a side operand on `axes` as a site whose tile is on `tile` lays it: along each
    of the tile's axes the side's extent there (`extents`) where it follows it, else one, so it broadcasts
    against the tile; a scalar where it follows none of them (a head's value at a head's tile). Its other
    axes (a batch row, a head, a group) a launch indexes away at its tile's origin."""
    if not set(tile) & set(axes):
        return ()
    return tuple(extents[axis] if axis in axes else 1 for axis in tile)


def lowering_rules(target) -> dict:
    """The rules a hook's operations need on `target`'s Mosaic launches: a TPU core's TensorCore rules; a
    GPU's at warpgroup semantics."""
    if generation.tpu(target):
        return mosaic_tpu_lowering.lowering_rules[tpu_core.CoreType.TC]
    return mosaic_gpu_lowering.mosaic_lowering_rules[plgpu_core.WGxWG_SEMANTICS]


def lowers(fn: Callable | None, arguments: Sequence[jax.ShapeDtypeStruct], rules: Mapping) -> bool:
    """Whether a Pallas lowering's `rules` (`lowering_rules`) hold a rule for every primitive of `fn` on its
    site's `arguments` (its tiles, then its side tiles), traced as a launch's body traces it
    (`launches.tracing_body`): Mosaic GPU's, at warpgroup semantics, have none for a transpose, which
    Triton's lowering takes. A marked span (`derivatives.marked_call_p`) lowers as its body, whose primitives
    are read. No hook lowers anywhere."""
    if fn is None:
        return True
    with launches.tracing_body(interpret=False):
        traced = jax.make_jaxpr(fn)(*arguments)
    return all(equation.primitive in rules or equation.primitive is marked_call_p
               for equation in equations(traced.jaxpr))

