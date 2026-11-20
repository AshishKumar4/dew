"""Dew: a GEMM launch's hooks (`templates.gemm`): its prologue and epilogue, each a JAX function of a tile and
its side tiles, as `grouped_matmul`'s launches bind them and their kernels trace them (`Hooks`)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
from jax._src.pallas.mosaic import core as tpu_core, lowering as mosaic_tpu_lowering
from jax._src.pallas.mosaic_gpu import (
    core as plgpu_core,  # the lowering's semantics (`_lowers`)
    lowering as mosaic_gpu_lowering,
)

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
                'epilogue_axes': self.epilogue_axes, 'out_dtypes': self.out_dtypes}


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


def _lowers(hook, tiles: tuple, sides, rules) -> bool:
    """Whether a Pallas lowering's `rules` hold a rule for every primitive of `hook` on its `tiles` and its
    side tiles: Mosaic GPU's, at warpgroup semantics, have none for a transpose, which Triton's lowering
    takes. A marked span (`derivatives.marked_call_p`) lowers as its body, whose primitives are read."""
    if hook is None:
        return True
    # Traced as a launch's body traces it, on tiles (`launches.tracing_body`).
    with launches.tracing_body(interpret=False):
        traced = jax.make_jaxpr(hook)(*tiles, *sides)
    return all(equation.primitive in rules or equation.primitive is marked_call_p
               for equation in equations(traced.jaxpr))


def lowering_rules(target) -> dict:
    """The rules a hook's operations need on `target`'s Mosaic launches: a TPU core's TensorCore rules; a
    GPU's at warpgroup semantics."""
    if generation.tpu(target):
        return mosaic_tpu_lowering.lowering_rules[tpu_core.CoreType.TC]
    return mosaic_gpu_lowering.mosaic_lowering_rules[plgpu_core.WGxWG_SEMANTICS]


def hook_lowers(hook, tiles: tuple, target) -> bool:
    """Whether `target`'s Mosaic launches take `hook` on `tiles` (`_lowers`, `lowering_rules`): a caller's
    hook asked before it is bound to a launch (`dew.nn.gated_mlp`'s gated product)."""
    return _lowers(hook, tiles, (), lowering_rules(target))


def hooks_lower(ba, dtype, target) -> bool:
    """Whether `target`'s Mosaic launches take a bound launch's hooks (`_lowers`, `lowering_rules`), traced on
    128-wide tiles: the primitives a hook makes do not depend on its tile's extents."""
    rules = lowering_rules(target)
    hooks = ba.arguments['hooks']

    def tiles(name, axes_of):
        return [jax.ShapeDtypeStruct((128,) * len(axes), side.dtype)
                for side, axes in zip(ba.arguments.get(name, ()), axes_of, strict=True)]

    return (_lowers(hooks.before(), (jax.ShapeDtypeStruct((128, 128), dtype),),
                    tiles('prologue_operands', hooks.prologue_axes), rules)
            and _lowers(hooks.after(), (jax.ShapeDtypeStruct((128, 128), jnp.float32),) * hooks.accumulators,
                        tiles('epilogue_operands', hooks.epilogue_axes), rules))
