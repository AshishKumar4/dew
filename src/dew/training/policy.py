"""What a run asks of its updates and of how its step fits its devices, each one value of its own.

`UpdateBatch` is the optimizer update's size: the batches it pools, or the tokens it trains on, a whole
number of the data's token batches. `FitPolicy` is how the step fits the devices: the microbatches each
step's batch is read in and the recomputation its fit search may take, each "auto" by default, the memory
planner's choice (`dew.training.planning.FitSearch`), or a run's own override.
"""
from __future__ import annotations

import dataclasses
from typing import Literal

Remat = Literal["auto", "none", "selective", "full"]
"""What a step recomputes in its backward: "auto" the least recomputation that fits, the planner's
search of each trained module's remat ladder (`dew.training.memory.recompute_more`); "none" nothing;
"selective" the least that fits short of each module's whole recomputation (the top of its ladder); "full"
each module's whole recomputation, the top of its ladder."""


@dataclasses.dataclass(frozen=True)
class UpdateBatch:
    """One optimizer update's size, stated one way: `batches`, the steps of the data's batch it pools
    (gradient accumulation), or `tokens`, the tokens it trains on, a whole number of the data's token batches
    (`dew.data.Dataset.batch(tokens=)`), from which a fit reads its batches (`Trainer.fit`); neither, one
    batch."""

    batches: int | None = None
    tokens: int | None = None

    def __post_init__(self):
        if self.batches is not None and self.tokens is not None:
            raise ValueError("an update's size is its batches or its tokens, not both")
        for name in ("batches", "tokens"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"an update's {name} is a positive count, not {value}")

    @property
    def stated_batches(self) -> int:
        """The batches it pools where it states them, one where it states tokens or nothing; a fit reads
        tokens' (`of_tokens`)."""
        return 1 if self.batches is None else self.batches

    def of_tokens(self, batch_tokens: tuple[int | None, ...]) -> int:
        """The batches an update of `tokens` pools over data whose phases batch `batch_tokens` tokens
        each (None for a batch of rows), its batches where it states them; refused where its tokens are no
        whole number of one token batch every phase shares."""
        if self.tokens is None:
            return self.stated_batches
        if len(set(batch_tokens)) != 1:
            raise ValueError(f"an update of {self.tokens} tokens is one accumulation over the data's token "
                             f"batches, which its phases batch {list(batch_tokens)} tokens: batch alike or "
                             "state its batches")
        (each,) = set(batch_tokens)
        if each is None or self.tokens % each:
            raise ValueError(f"an update of {self.tokens} tokens is no whole number of the data's token "
                             f"batches ({each}); batch the data by tokens (Dataset.batch(tokens=)) dividing "
                             "it")
        return self.tokens // each


@dataclasses.dataclass(frozen=True)
class FitPolicy:
    """How a run's step fits its devices: `split`, the microbatches each step's batch is read in, its rows
    split evenly (under a mesh of pipeline stages, the microbatches the stages feed, a multiple of them), or
    "auto", the fewest that fit before recomputing anything; and `remat`, what the step recomputes
    (`Remat`), "auto" the least that fits. A step's tokens and its update's stay as stated either way."""

    split: int | Literal["auto"] = "auto"
    remat: Remat = "auto"

    def __post_init__(self):
        if self.split != "auto" and (not isinstance(self.split, int) or self.split < 1):
            raise ValueError(f"a step's split is 'auto' or a positive count of microbatches, not "
                             f"{self.split!r}")
        if self.remat not in ("auto", "none", "selective", "full"):
            raise ValueError(f"a step's remat is auto, none, selective or full, not {self.remat!r}")

    @property
    def stated_split(self) -> int | None:
        """Its split where it states one, None where the planner chooses it."""
        return None if self.split == "auto" else self.split
