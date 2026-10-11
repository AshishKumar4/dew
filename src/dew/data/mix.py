"""Accepted-row accounting and mixtures measured in valid target tokens."""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction

import grain.python as pygrain

from dew.data.dataset import Batch
from dew.data.pipeline import Rejected
from dew.records import integer, integers, record

_log = logging.getLogger(__name__)


@dataclasses.dataclass
class CorpusCounts:
    name: str
    attempted: int = 0
    accepted: int = 0
    truncated: int = 0
    chunks: int = 0
    rejected: Counter[str] = dataclasses.field(default_factory=Counter)

    def metrics(self) -> dict[str, float]:
        total = max(1, self.attempted)
        return {
            f"data/attempted/{self.name}": float(self.attempted),
            f"data/accepted/{self.name}": float(self.accepted),
            f"data/truncated/{self.name}": self.truncated / total,
            f"data/chunks_per_record/{self.name}": self.chunks / total,
            **{
                f"data/rejected/{self.name}/{reason}": float(count) for reason, count in self.rejected.items()
            },
        }

    def state(self) -> Mapping[str, object]:
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "truncated": self.truncated,
            "chunks": self.chunks,
            "rejected": dict(self.rejected),
        }

    def restore(self, state: Mapping[str, object]) -> None:
        self.attempted = integer(state["attempted"], "attempted")
        self.accepted = integer(state["accepted"], "accepted")
        self.truncated = integer(state["truncated"], "truncated")
        self.chunks = integer(state["chunks"], "chunks")
        self.rejected = Counter(
            {name: integer(value, name) for name, value in record(state["rejected"], "rejected").items()}
        )


@dataclasses.dataclass(frozen=True)
class SearchBudget:
    consecutive: int = 10_000
    seconds: float | None = 60.0
    warn_over: float = 0.01

    def __post_init__(self) -> None:
        if self.consecutive < 1 or (self.seconds is not None and self.seconds <= 0):
            raise ValueError("rejection search budgets must be positive")
        if not 0 <= self.warn_over <= 1:
            raise ValueError("warn_over is a share between zero and one")


_DEFAULT_BUDGET = SearchBudget()


class _KeepAccepted:
    def __init__(self, counts: CorpusCounts, budget: SearchBudget, tokens: Callable[[Batch], int] | None):
        self.counts, self.budget, self.tokens = counts, budget, tokens
        self.consecutive = 0
        self.started: float | None = None
        self.warned = False

    def __call__(self, element: Batch | Rejected) -> bool:
        self.counts.attempted += 1
        reason = element.reason if isinstance(element, Rejected) else None
        if reason is None and self.tokens is not None and not isinstance(element, Rejected):
            count = self.tokens(element)
            if count < 0:
                raise ValueError(f"corpus {self.counts.name} has a negative valid-target count: {count}")
            if count == 0:
                reason = "zero_targets"
        if reason is None:
            self.counts.accepted += 1
            if not isinstance(element, Rejected):
                self.counts.truncated += int(bool(element.get("_dew_truncated", False)))
                self.counts.chunks += int(element.get("_dew_chunks", 0))
            self.consecutive, self.started = 0, None
            return True
        self.counts.rejected[reason] += 1
        self.consecutive += 1
        now = time.monotonic()
        if self.started is None:
            self.started = now
        if (
            not self.warned
            and sum(self.counts.rejected.values()) / self.counts.attempted > self.budget.warn_over
        ):
            _log.warning(
                "corpus %s rejected more than %.1f%% of attempted records (%s)",
                self.counts.name,
                self.budget.warn_over * 100,
                reason,
            )
            self.warned = True
        if self.consecutive >= self.budget.consecutive or (
            self.budget.seconds is not None and now - self.started >= self.budget.seconds
        ):
            raise ValueError(
                f"corpus {self.counts.name}: rejection search budget exhausted after "
                f"{self.consecutive} consecutive rejections ({reason}); "
                "this is not a proof that the corpus is empty"
            )
        return False


class _AcceptedIterator(pygrain.DatasetIterator[Batch]):
    def __init__(self, parent: pygrain.DatasetIterator[Batch | Rejected] | pygrain.DatasetIterator[Batch],
                 predicate: _KeepAccepted):
        super().__init__(parent)
        self._read, self._predicate = parent, predicate

    def __next__(self) -> Batch:
        value = next(self._read)
        if isinstance(value, Rejected):
            raise RuntimeError("Grain filter delivered a rejected record")
        return value

    def get_state(self) -> Mapping[str, object]:
        return {
            "parent": self._read.get_state(),
            "corpus": self._predicate.counts.name,
            "counts": self._predicate.counts.state(),
            "consecutive": self._predicate.consecutive,
            "warned": self._predicate.warned,
        }

    def set_state(self, state: Mapping[str, object]) -> None:
        self._read.set_state(record(state["parent"], "accepted parent"))
        self._predicate.counts.restore(record(state["counts"], "corpus counts"))
        self._predicate.consecutive = integer(state["consecutive"], "consecutive")
        self._predicate.started = None
        self._predicate.warned = bool(state["warned"])

    def close(self) -> None:
        self._read.close()


class Accepted(pygrain.IterDataset[Batch]):
    """Grain's filter, with a named search budget and checkpointed counts."""

    def __init__(
        self,
        parent: pygrain.IterDataset[Batch | Rejected] | pygrain.IterDataset[Batch],
        name: str,
        *,
        budget: SearchBudget = _DEFAULT_BUDGET,
        tokens: Callable[[Batch], int] | None = None,
    ):
        super().__init__(parent)
        self._records, self._name, self._budget, self._tokens = parent, name, budget, tokens

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        predicate = _KeepAccepted(CorpusCounts(self._name), self._budget, self._tokens)
        return _AcceptedIterator(iter(self._records.filter(predicate)), predicate)


def token_weights(
    names: Sequence[str], weights: Sequence[float], windows: Sequence[int], targets: Sequence[int]
) -> tuple[float, ...]:
    """Window shares for authoritative per-corpus valid-target counts."""
    adjusted = []
    for name, weight, rows, tokens in zip(names, weights, windows, targets, strict=True):
        if tokens <= 0:
            raise ValueError(
                f"corpus {name} has zero valid target tokens"
                if tokens == 0
                else f"corpus {name} has a negative valid-target count"
            )
        if rows <= 0:
            raise ValueError(f"corpus {name} has no accepted windows")
        adjusted.append(weight * rows / tokens)
    return tuple(adjusted)


class _DeficitIterator(pygrain.DatasetIterator[Batch]):
    def __init__(
        self,
        parents: Sequence[pygrain.DatasetIterator[Batch]],
        weights: tuple[int, ...],
        tokens: Callable[[Batch], int],
    ):
        super().__init__(parents)
        self._reads, self._weights, self._tokens = tuple(parents), weights, tokens
        self._delivered = [0] * len(parents)

    def __next__(self) -> Batch:
        total = sum(self._delivered)
        scale = sum(self._weights)
        chosen = max(range(len(self._reads)),
                     key=lambda index: self._weights[index] * total - self._delivered[index] * scale)
        value = next(self._reads[chosen])
        tokens = self._tokens(value)
        if tokens <= 0:
            raise RuntimeError(
                "token deficit mix received a window without valid targets; filter it before mixing"
            )
        self._delivered[chosen] += tokens
        return value

    def get_state(self) -> Mapping[str, object]:
        return {"parents": [parent.get_state() for parent in self._reads], "delivered": list(self._delivered)}

    def set_state(self, state: Mapping[str, object]) -> None:
        delivered = integers(state["delivered"], "delivered targets")
        parents = state["parents"]
        if (
            not isinstance(parents, (list, tuple))
            or len(parents) != len(self._reads)
            or len(delivered) != len(self._reads)
        ):
            raise ValueError("token deficit state has a different corpus count")
        if min(delivered) < 0:
            raise ValueError("token deficit state has negative delivered targets")
        for parent, saved in zip(self._reads, parents, strict=True):
            parent.set_state(record(saved, "corpus iterator"))
        self._delivered = list(delivered)

    def close(self) -> None:
        for parent in self._reads:
            parent.close()


class TokenDeficit(pygrain.IterDataset[Batch]):
    """Pick the corpus furthest below its token share, with index-order ties."""

    def __init__(
        self,
        parents: Sequence[pygrain.IterDataset[Batch]],
        weights: Sequence[float],
        *,
        tokens: Callable[[Batch], int],
        names: Sequence[str],
        budget: SearchBudget = _DEFAULT_BUDGET,
    ):
        if not parents or len(parents) != len(weights) or len(names) != len(parents):
            raise ValueError("token deficit mix needs one weight and name per corpus")
        if any(not math.isfinite(weight) or weight <= 0 for weight in weights):
            raise ValueError("token deficit weights must be finite and positive")
        filtered = tuple(
            Accepted(parent, name, budget=budget, tokens=tokens)
            for parent, name in zip(parents, names, strict=True)
        )
        super().__init__(filtered)
        ratios = tuple(Fraction(weight) for weight in weights)
        scale = math.lcm(*(ratio.denominator for ratio in ratios))
        self._corpora, self._weights, self._tokens = (
            filtered,
            tuple(ratio.numerator * (scale // ratio.denominator) for ratio in ratios),
            tokens,
        )

    def __iter__(self) -> pygrain.DatasetIterator[Batch]:
        return _DeficitIterator([iter(parent) for parent in self._corpora], self._weights, self._tokens)


def state_counts(state: Mapping[str, object]) -> dict[str, CorpusCounts]:
    """Read per-corpus counters through Grain's nested iterator checkpoints."""
    counted: dict[str, CorpusCounts] = {}

    def visit(branch):
        if isinstance(branch, Mapping):
            corpus, stored = branch.get("corpus"), branch.get("counts")
            if isinstance(corpus, str) and isinstance(stored, Mapping):
                parsed = CorpusCounts(corpus)
                parsed.restore(stored)
                combined = counted.setdefault(corpus, CorpusCounts(corpus))
                combined.accepted = max(combined.accepted, parsed.accepted)
                combined.truncated = max(combined.truncated, parsed.truncated)
                combined.chunks = max(combined.chunks, parsed.chunks)
                combined.rejected.update(parsed.rejected)
            for child in branch.values():
                visit(child)
        elif isinstance(branch, (tuple, list)):
            for child in branch:
                visit(child)

    visit(state)
    for counts in counted.values():
        counts.attempted = counts.accepted + sum(counts.rejected.values())
    return counted
