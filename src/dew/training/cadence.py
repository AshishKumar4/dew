"""When something recurs in a run: a cadence of steps, of committed tokens (`Tokens`), as a share of a run
resolves to on a token budget, or of consumed examples (`Examples`), as one resolves to on a run bounded
in examples that counts no steps ahead (`dew.training.budget.every_step`)."""

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


@dataclasses.dataclass(frozen=True)
class Examples:
    """A cadence on a run's consumed examples (`dew.state.TrainState.consumed`): due at each step whose
    batch brings the source examples the run has consumed past a multiple of `every`, where a share of a
    run bounded in examples falls when no pass counts its steps ahead (`every_step`)."""

    every: int

    def due(self, consumed: int, before: int) -> bool:
        """Whether the step that brought the consumed examples from `before` to `consumed` passed a
        multiple."""
        return consumed // self.every > before // self.every


type Cadence = int | Tokens | Examples | None
"""When something recurs in a run: every that many steps, on the committed-token clock, on the consumed
examples, or never."""
