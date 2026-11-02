"""Check 1's conditioning (`reference_error.assert_as_exact(..., conditioning=)`, ledger 1714). A site
declares the weighed check's own seeds beside its `seeds`; the unweighted check runs first, whole (its
terms sizing, its declared seeds, a site whose sizing never resolves declaring MOST_SEEDS, and its stages):
a decided failure fails the site and a decided pass passes it. Only where it is still Inconclusive with its
budget spent is the site judged again, weighed: each arm's element e at its truth t read as the residual
(e - t) / c against a truth of zeros, every row of unit size, c the reference program's rounding scale
there, the same for every arm (the element's RMS displacement where every rounding of the program, traced
in the arithmetic the arms run in, moves its value by a standard normal multiple of its bound; tests/
rounding_bound.py's `rounding_scale`). A leaf's heavy elements (`heavy_masks`) are judged as a leaf of
their own beside the light rest, each class's worst rows by the gate's factor for its values a row
(`check1_statistics.row_factor`); that check must decide a pass. An element no rounding reaches (c = 0)
must be exact in every arm. Where a computation amplifies its rounding unevenly (an inner step past its
stable rate), the unweighted pool is the amplified elements' and its interval stays wide; weighed, a fault
on the well-conditioned rest shows, and one on the amplified elements shows on their own class. A sharded
site's map tasks record both checks' draws, and check 2 is never weighed."""

import dataclasses
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from check1_statistics import Finding, GateFailure


def scales(scale) -> list:
    """A rounding scale's leaves, a non-floating output's None among them (`rounding_scale`)."""
    return jax.tree.leaves(scale, is_leaf=lambda leaf: leaf is None)


def float_leaf(leaf) -> bool:
    """Whether `leaf` holds floats, bfloat16's among them, as JAX's dtypes read it."""
    return jnp.issubdtype(jnp.result_type(leaf), jnp.floating)


@dataclasses.dataclass(frozen=True)
class Weighed:
    """One truth's weighing for check 1's conditioning (`Conditioning`): the `truth` on the host, the
    reference program's rounding `scale` at its inputs (`rounding_scale`, each element's RMS displacement
    under its program's roundings), applied to every arm alike, and the truth of zeros and unit sizes
    (`zeros`, `ones`) every weighed arm is held to."""

    truth: object
    scale: object
    zeros: object
    ones: object
    heavy: list | None = None
    """Each floating leaf's heavy elements (`_heavy`), a mask, None for a leaf of none: its residuals
    judged as a leaf of their own beside the light ones' (`classed`)."""

    def widths(self) -> tuple[int, ...]:
        """Each judged leaf's row width (`_Draws.widths`, `row_factor`), a class's the fewest of its values
        any row holds, at least one: a heavy class of one value a row is a row of one."""
        found = []
        truths = jax.tree.leaves(self.truth)
        for truth, mask in zip(truths, self.heavy or [None] * len(truths), strict=True):
            width = int(np.shape(truth)[-1]) if np.ndim(truth) >= 2 else 0
            if mask is None:
                found.append(width)
                continue
            for counts in (np.sum(mask, axis=-1), np.sum(~mask, axis=-1)):
                held = counts[counts > 0]
                found.append(int(held.min()) if held.size else width)
        return tuple(found)

    def classed(self, tree, masked: bool = True):
        """`tree`, of the truth's structure, each floating leaf with heavy elements (`heavy`) split into
        `{"heavy": ..., "light": ...}`, the other class's elements zero in each where `masked` (sizes, not
        masked, stay whole in each), so each class is judged as its own leaf, pooled and by rows, a row of
        no heavy element as it was."""
        if self.heavy is None:
            return tree
        leaves, structure = jax.tree.flatten(tree, is_leaf=lambda leaf: leaf is None)
        split = [leaf if mask is None or leaf is None
                 else {"heavy": np.where(mask, leaf, 0.0) if masked else leaf,
                       "light": np.where(mask, 0.0, leaf) if masked else leaf}
                 for leaf, mask in zip(leaves, self.heavy, strict=True)]
        return structure.unflatten(split)

    def weighed(self, tree, arm: str):
        """`tree`, an arm's outputs, each floating element e at its truth t and scale c read as its residual
        (e - t) / c, c the element's RMS displacement under the reference program's roundings, against a
        truth of zeros: its error in units of the rounding the reference's program does there, every row
        of unit size (`ones`, the
        terms `_prepared` sizes rows by). Each leaf's shape is checked before any arithmetic. An element
        no rounding reaches (c = 0) is exact in the reference's program, and an arm not exact there fails,
        by name, as does an element whose scale is not finite, which no error can be read against."""
        structure = jax.tree.structure(self.truth)
        paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(self.truth)]
        leaves = structure.flatten_up_to(jax.device_get(tree))
        for path, leaf, true in zip(paths, leaves, jax.tree.leaves(self.truth), strict=True):
            if np.shape(leaf) != np.shape(true):
                raise AssertionError(f"{arm}, {path}: its shape is {np.shape(leaf)}, the truth's "
                                     f"{np.shape(true)}")
        weighed = []
        for path, leaf, true, scale in zip(paths, leaves, jax.tree.leaves(self.truth), scales(self.scale),
                                           strict=True):
            if scale is None:
                # A leaf of no float (an index, a count) is judged as it is.
                weighed.append(leaf)
                continue
            value, true, scale = (np.asarray(part, np.float64) for part in (leaf, true, scale))
            unweighable = ~np.isfinite(scale)
            if unweighable.any():
                at = tuple(int(index) for index in np.argwhere(unweighable)[0])
                raise AssertionError(f"{arm}, {path}{list(at)}: the reference's rounding scale is "
                                     f"{scale[at]!r} there, which no error can be weighed against")
            exact = scale == 0
            wrong = exact & ~(value == true)
            if wrong.any():
                # A decided failure of an exact reference's element (`Finding`), the most an arm misses by.
                at = tuple(int(index) for index in np.argwhere(wrong)[0])
                raise GateFailure([Finding(
                    f"{arm}, {path}", "exact reference", float(np.max(np.abs(value - true)[wrong])),
                    f"{arm}, {path}{list(at)}: no rounding of the reference's program reaches it (its "
                    f"rounding scale is zero), and the arm is not exact: {value[at]!r} against "
                    f"{true[at]!r}")])
            weighed.append(np.where(exact, 0.0, (value - true) / np.where(exact, 1.0, scale)))
        return self.classed(structure.unflatten(weighed)), self.classed(self.zeros)


def heavy_masks(runs) -> list:
    """Each floating output leaf's heavy elements over `runs`, the reference's outputs beside their float64
    truths draw by draw, as a mask, None for a leaf of none: those whose share of the reference's squared
    error summed over the draws exceeds their uniform share of their row (1/n, n its length along the leaf's
    last axis). The unweighted RMS, which defines on par, is theirs, so the weighed check judges them apart
    from the light rest (`Weighed.classed`): an error moved onto them never hides behind the rest's."""
    totals = None
    for ran, true in runs:
        squares = [np.square(np.asarray(arm, np.float64) - np.asarray(truth, np.float64))
                   if float_leaf(truth) and np.ndim(truth) else None
                   for arm, truth in zip(jax.tree.leaves(ran), jax.tree.leaves(true), strict=True)]
        totals = squares if totals is None else [None if total is None else total + square
                                                 for total, square in zip(totals, squares, strict=True)]
    masks = []
    for total in totals or []:
        heavy = None if total is None else total * total.shape[-1] > np.sum(total, axis=-1, keepdims=True)
        masks.append(heavy if heavy is not None and heavy.any() else None)
    return masks


@dataclasses.dataclass(frozen=True)
class Conditioning:
    """Check 1's conditioning (`assert_as_exact`'s `conditioning`): `scale_of(inputs, seed)`, the reference
    program's rounding scale at a draw's inputs, which weighs every arm's error there (`Weighed`), and each
    floating output leaf's `heavy` elements (`heavy_masks`), which the weighed check judges apart."""

    scale_of: Callable
    heavy: list | None = None

    def at(self, cases) -> dict[int, Weighed]:
        """Each truth's weighing by its id, from `cases` of (truth, its inputs, its seed), a truth held
        twice weighed once."""
        weighed: dict[int, Weighed] = {}
        for truth, inputs, seed in cases:
            if id(truth) not in weighed:
                weighed[id(truth)] = self.of(truth, inputs, seed)
        return weighed

    def of(self, truth, inputs, seed) -> Weighed:
        """The weighing of `truth`, the truth of `inputs`, at the draw `seed`."""
        truth = jax.device_get(truth)
        scale = self.scale_of(inputs, seed)
        zeros = jax.tree.map(lambda leaf: np.zeros(np.shape(leaf)) if float_leaf(leaf) else leaf, truth)
        ones = jax.tree.map(lambda leaf: np.ones(np.shape(leaf)) if float_leaf(leaf) else None, truth)
        weighing = Weighed(truth, scale, zeros, ones, self.heavy)
        return dataclasses.replace(weighing, ones=weighing.classed(ones, masked=False))
