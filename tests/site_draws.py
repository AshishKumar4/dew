"""A site's draws for check 1 (`reference_error.assert_as_exact`): each leaf's errors over a run of seeds,
the record its rules judge, pooled and by rows."""

import dataclasses
from collections.abc import Sequence

import numpy as np
from check1_statistics import RULED_SHARE, ZeroRows, row_factor


@dataclasses.dataclass
class _Draws:
    """Each leaf's errors over a run of seeds: Dew's and the reference's mean squared error from
    float64 (seed by leaf), the log of each seed's worst sized row over its size (Dew's, the
    reference's; `log_over`), each
    seed's rows of float64 zero with Dew's and the reference's errors on them (`ZeroRows`, seed by leaf;
    `pooled_zero_rows`), whether every value of the leaf was finite on every seed, and each leaf's row
    width (its last axis's length, 0 for a leaf of no rows; NARROW_FACTORS)."""
    names: list[str]
    mine: np.ndarray
    theirs: np.ndarray
    worst: np.ndarray
    zero_rows: list[list[ZeroRows]]
    finite: np.ndarray
    widths: tuple[int, ...]
    examples: list[list[np.ndarray | None]] | None = None
    """A site whose leaves stack independent examples on their first axis (`assert_as_exact`'s
    `examples`): each seed's per-example log RMS ratios of each leaf (`_example_ratios`), else None."""
    example_rows: list[list[np.ndarray | None]] | None = None
    """At such a site, each seed's per-example log ratio of Dew's worst row over the reference's worst row
    on the same example, each over its size (`_example_worst_rows`), else None."""
    shares: np.ndarray | None = None
    """The share of the reference's worst row's squared error its largest element carries (seed by leaf),
    zero where no row is sized; None where the reference keeps no rows' elements (a recording's summary)."""

    def factor(self, index: int) -> float:
        """The worst-row rule's factor for leaf `index`: its rows' width's (`row_factor`), but a row of one
        value's where one element carries past RULED_SHARE of the reference's worst row over these draws."""
        width = self.widths[index]
        if width > 2 and self.shares is not None:
            at = int(np.argmax(self.worst[:, index, 1]))
            if self.shares[at, index] > RULED_SHARE:
                return row_factor(1)
        return row_factor(width)

    def joined(self, other: "_Draws") -> "_Draws":
        examples, example_rows = ((getattr(self, field) + getattr(other, field)
                                   if getattr(self, field) is not None and getattr(other, field) is not None
                                   else None) for field in ("examples", "example_rows"))
        shares = (np.concatenate([self.shares, other.shares])
                  if self.shares is not None and other.shares is not None else None)
        return _Draws(self.names, *(np.concatenate([getattr(self, field), getattr(other, field)])
                                    for field in ("mine", "theirs", "worst")),
                      self.zero_rows + other.zero_rows, np.concatenate([self.finite, other.finite]),
                      self.widths, examples, example_rows, shares)

    def between(self, start: int, stop: int) -> "_Draws":
        """These draws' seeds `start` to `stop`, counted from their first."""
        cut = slice(start, stop)
        examples, example_rows = (None if getattr(self, field) is None else getattr(self, field)[cut]
                                  for field in ("examples", "example_rows"))
        return _Draws(self.names, self.mine[cut], self.theirs[cut], self.worst[cut], self.zero_rows[cut],
                      self.finite[cut], self.widths, examples, example_rows,
                      None if self.shares is None else self.shares[cut])

    def of_examples(self, index: int) -> list[np.ndarray] | None:
        """Leaf `index`'s per-example log ratios a seed, or None where the site stacks none or a seed's
        are not all paired (`_example_ratios`)."""
        if self.examples is None:
            return None
        found = [seed[index] for seed in self.examples]
        return None if any(ratios is None for ratios in found) else found

    def of_example_rows(self, index: int) -> np.ndarray | None:
        """Leaf `index`'s per-example paired worst-row log ratios over every seed, or None where the site
        stacks no examples or the leaf has no rows (`_example_worst_rows`)."""
        if self.example_rows is None or any(seed[index] is None for seed in self.example_rows):
            return None
        return np.concatenate([seed[index] for seed in self.example_rows])

    def scalar(self, index: int) -> bool:
        """Whether leaf `index` holds one value an example (its width, the values a row, one)."""
        return self.widths[index] <= 1

    def against_larger(self, library: "_Draws", leaves: Sequence[str]) -> "_Draws":
        """These draws judged against the larger of their reference's error and `library`'s, the same
        Dew on the same draws against the library arm (`assert_as_exact`'s `library_fn`), on the
        exception's `leaves` alone: seed by seed and leaf by leaf, the reference's evidence (its error,
        its worst row, its rows of float64 zero) is whichever of the two arms' error is the larger there,
        and a value must be finite in both. Every other leaf is the reference's alone."""
        if library.names != self.names or len(library.mine) != len(self.mine):
            raise AssertionError("the library arm's draws are not the reference's")
        unknown = sorted(set(leaves) - set(self.names))
        if unknown or not leaves:
            raise ValueError(f"a library arm names the leaves it covers, of {self.names}; it names "
                             f"{list(leaves)}")
        covered = np.isin(self.names, leaves)
        pick = (library.theirs > self.theirs) & covered
        worst = self.worst.copy()
        worst[..., 1] = np.where(pick, library.worst[..., 1], self.worst[..., 1])
        zero_rows = [[(theirs if chosen else mine)
                      for mine, theirs, chosen in zip(seed, other, picked, strict=True)]
                     for seed, other, picked in zip(self.zero_rows, library.zero_rows, pick, strict=True)]
        # A library arm's leaves are judged by their pooled ratio alone: their examples are not Dew's against
        # one reference.
        shares = (None if self.shares is None or library.shares is None
                  else np.where(pick, library.shares, self.shares))
        return _Draws(self.names, self.mine, np.where(pick, library.theirs, self.theirs), worst, zero_rows,
                      self.finite & (library.finite | ~covered), self.widths, shares=shares)
