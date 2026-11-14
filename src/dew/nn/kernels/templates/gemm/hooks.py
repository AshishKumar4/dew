"""Dew: a GEMM launch's hooks (`templates.gemm`): its prologue and epilogue, each a JAX function of a tile and
its side tiles, as `grouped_matmul`'s launches bind them and their kernels trace them (`Hooks`)."""

import dataclasses
import functools

import jax


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
