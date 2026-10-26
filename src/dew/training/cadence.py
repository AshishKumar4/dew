"""When something recurs in a run: a cadence of steps, or of committed tokens (`Tokens`), as a share of
a run resolves to on a token budget (`dew.training.budget.every_step`)."""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class Tokens:
    """A cadence on the committed-token clock: due at each step whose commit brings the run's committed
    tokens past a multiple of `every`, where a share of a token budget falls (`every_step`)."""

    every: int

    def due(self, real: int, before: int) -> bool:
        """Whether the step that brought the committed tokens from `before` to `real` passed a multiple."""
        return real // self.every > before // self.every


type Cadence = int | Tokens | None
"""When something recurs in a run: every that many steps, on the committed-token clock, or never."""
