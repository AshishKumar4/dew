"""When something recurs in a run: a cadence of steps, or of a counted clock (`Counted`): the committed
tokens, as a share of a run resolves to on a token budget, or the consumed examples, as one resolves to on a
run bounded in examples that counts no steps ahead (`dew.training.budget.every_step`)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Literal

type Clock = Literal["tokens", "examples"]
"""A run's counted clocks: its committed updates' real tokens, and the source examples its batches
consumed (`dew.state.TrainState.consumed`)."""


@dataclasses.dataclass(frozen=True)
class Counted:
    """A cadence on one of a run's counted clocks (`clock`): due at each step that brings the count past a
    multiple of `every`."""

    every: int
    clock: Clock = "tokens"

    def due(self, counts: Mapping[Clock, tuple[int, int]]) -> bool:
        """Whether the step that brought the clock's count from the second of its `counts` to the first
        passed a multiple."""
        now, before = counts[self.clock]
        return now // self.every > before // self.every

    def crossed(self, counts: Mapping[Clock, tuple[int, int]]) -> bool:
        """Whether the step crossed `every` itself, from before it to at or past it: a point once in the
        run, where a cooldown branches."""
        now, before = counts[self.clock]
        return before < self.every <= now


type Cadence = int | Counted | None
"""When something recurs in a run: every that many steps, on a counted clock, or never."""
