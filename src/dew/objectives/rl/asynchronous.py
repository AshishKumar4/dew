"""Asynchronous GRPO rollouts from a rollout server, with bounded policy staleness.

`AsyncRollout` is a trainer `Rollout` that keeps generation running while the
trainer updates. Wrap the prompt dataset with `rollout.prompts(dataset)`: the
wrapped stream registers each prompt batch as the trainer's prefetch reads it,
so when the trainer hands over batch `i` the rollout already knows batches
`i + 1 ... i + ahead` and submits them to the server at once, under the
weights it serves now. Batch `i`'s own draws were submitted `ahead` calls
earlier and have been generating and scoring since. Nothing is read ahead of
the trainer's own prefetch, so the checkpointed data position stays the
trainer's: a resumed run re-reads and resubmits whatever was in flight.

Every draw carries the policy version its request was submitted under
(`RolloutServer.version`, the trainer's `updates` count when the weights were
pushed), and each sampled id carries its draw's version under `versions`. The
lag of a batch is the trainer's `updates` minus its oldest draw's version.
Weights are pushed when the served version falls `sync_every` updates behind,
so a batch is at most `ahead + sync_every - 1` updates stale, and
construction refuses a `max_lag` below that. At consumption, a batch
submitted staler than `max_lag` (a resumed run, a stalled push) is discarded
before anyone waits on it, the weights are pushed, and its prompts are drawn
once more; a push that still leaves it too stale raises.

Off-policy correction is decoupled PPO (AReaL, arXiv:2505.24298): the ratio's
old policy is the proximal one, the trainer's current weights, rescored over
the drawn tokens whenever the batch is stale or the server reports no raw
likelihood; the recorded behavior likelihoods stay as the server reported
them, and the objective's `behavior_importance_cap` weights each token by
proximal over behavior. A run that allows any lag must set that cap.
"""

from __future__ import annotations

import dataclasses
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import jax
import numpy as np

from dew.data.dataset import Batch, Dataset, tapped
from dew.data.prompts import PROMPT_KEY
from dew.inference.rollouts import Draw, RolloutServer
from dew.nn.inputs import local_rows
from dew.training.state import TrainState

from .grpo import GRPOObjective
from .rollout import Reward, check_rollout, completion_rows, prompt_rows
from .rollouts import OLD_LOG_PROBS_KEY, sampled_values


@dataclass(frozen=True)
class RolloutRecord:
    """What one trainer call consumed: its `updates` clock, the batch's policy
    version and lag, its mean reward, how many times it was redrawn for
    staleness, and the seconds the trainer waited on it."""

    updates: int
    version: int
    lag: int
    reward: float
    redrawn: int
    waited: float


type Scored = tuple[Draw, float]


@dataclass
class _Entry:
    """One registered prompt batch, host-side, and its submitted draws."""

    serial: int
    prompts: np.ndarray
    lengths: np.ndarray
    sources: list[str]
    truths: list[str]
    infos: list[str]
    submitted: int = -1
    """The served version when the draws were submitted; each draw reports its own."""
    draws: list[list[Future[Scored]]] = field(default_factory=list)


class AsyncRollout:
    """Draw `groups` completions per prompt on a `RolloutServer`, `ahead` batches early.

    `reward` scores decoded completions (EOS excluded) on `scorers` threads
    as each draw finishes, so verification overlaps generation and the
    update. `log`, when given, receives a `RolloutRecord` per call. The
    objective's context must be the prompt width plus `max_new_tokens`
    minus one. Single-process trainers only: the server is one endpoint and
    the weights it receives are one process's full tree.
    """

    def __init__(self, objective: GRPOObjective, server: RolloutServer, reward: Reward, *,
                 decode: Callable[[Sequence[int]], str], groups: int = 4, max_new_tokens: int = 32,
                 max_lag: int = 1, ahead: int = 1, sync_every: int = 1, estimator: str = "group",
                 scorers: int = 16, log: Callable[[RolloutRecord], None] | None = None):
        check_rollout(groups, max_new_tokens, estimator)
        for name, value, least in (("ahead", ahead, 0), ("sync_every", sync_every, 1), ("max_lag", max_lag, 0)):
            if type(value) is not int or value < least:
                raise ValueError(f"{name} must be an integer of at least {least}")
        if ahead + sync_every - 1 > max_lag:
            raise ValueError(
                f"ahead={ahead} and sync_every={sync_every} let a batch fall {ahead + sync_every - 1} "
                f"updates behind, past max_lag={max_lag}")
        if max_lag > 0 and objective.behavior_importance_cap is None:
            raise ValueError("stale rollouts need the objective's behavior_importance_cap: "
                             "the proximal-to-behavior importance weight is the off-policy correction")
        self.objective, self.server, self.reward, self.decode = objective, server, reward, decode
        self.groups, self.max_new_tokens, self.estimator = groups, max_new_tokens, estimator
        self.max_lag, self.ahead, self.sync_every, self.log = max_lag, ahead, sync_every, log
        self._scorers = ThreadPoolExecutor(max_workers=scorers, thread_name_prefix="dew-reward")
        self._lock = threading.Lock()
        self._registered: deque[_Entry] = deque()
        self._serial = 0
        self._rescore = jax.jit(objective.packed_log_probs)

    def prompts(self, dataset: Dataset) -> Dataset:
        """`dataset` with a training stream that registers each batch ahead of the step."""
        opened = tapped(dataset.train, self._register)

        def train() -> Iterator[Batch]:
            with self._lock:
                self._registered.clear()
            return opened()

        return dataclasses.replace(dataset, train=train)

    def close(self) -> None:
        """Stop the reward threads; the server belongs to the caller."""
        self._scorers.shutdown(wait=True, cancel_futures=True)

    def _register(self, batch: Batch) -> None:
        prompts, lengths, sources, truths, infos = prompt_rows(batch, self.objective.seq_len, self.max_new_tokens)
        with self._lock:
            self._registered.append(_Entry(self._serial, prompts, lengths, sources, truths, infos))
            self._serial += 1

    def _scored(self, draw: Future[Draw], entry: _Entry, row: int) -> Future[Scored]:
        """The draw's future, chained into its reward on a scorer thread."""
        scored: Future[Scored] = Future()

        def score() -> None:
            try:
                drawn = draw.result()
                text = self.decode(drawn.tokens[:len(drawn.tokens) - int(drawn.terminated)])
                value = float(self.reward(entry.sources[row], text, entry.truths[row], entry.infos[row]))
                if not math.isfinite(value):
                    raise ValueError("the reward returned a non-finite score")
                scored.set_result((drawn, value))
            except BaseException as failure:
                scored.set_exception(failure)

        def chained(_: Future[Draw]) -> None:
            try:
                self._scorers.submit(score)
            except RuntimeError as closed:
                # The rollout closed while this draw was in flight.
                scored.set_exception(closed)

        draw.add_done_callback(chained)
        return scored

    def _submit(self, entry: _Entry, key: jax.Array, attempt: int) -> None:
        """Submit every row's group of draws under the served weights."""
        rows, width = entry.prompts.shape
        seeds = np.random.default_rng(
            [*np.asarray(jax.random.key_data(key)).ravel().tolist(), entry.serial, attempt]
        ).integers(0, 2 ** 31 - 1, (rows, self.groups))
        entry.submitted = self.server.version
        entry.draws = [[self._scored(self.server.submit(
            entry.prompts[row, width - int(entry.lengths[row]):].tolist(), self.max_new_tokens,
            seed=int(seeds[row, group])), entry, row) for group in range(self.groups)] for row in range(rows)]

    def __call__(self, state: TrainState, batch: Batch, key: jax.Array) -> dict[str, np.ndarray]:
        if jax.process_count() != 1:
            raise ValueError("AsyncRollout serves one trainer process; the server takes one full weight tree")
        updates = int(state.updates)
        if updates - self.server.version >= self.sync_every:
            self.server.load(state.params, updates)
        with self._lock:
            if not self._registered:
                raise ValueError("an AsyncRollout batch comes from the stream of AsyncRollout.prompts(dataset)")
            entry = self._registered.popleft()
            upcoming = list(self._registered)[:self.ahead]
        if not np.array_equal(entry.prompts, local_rows(batch[PROMPT_KEY])):
            raise ValueError("the trainer's batch is not the next registered prompt batch")
        for pending in (entry, *upcoming):
            if not pending.draws:
                self._submit(pending, key, attempt=0)
        began = time.perf_counter()
        redrawn = 0
        # The submission version is known before any draw finishes, so a batch
        # bound for the bin is never waited on or scored, and its failed draws
        # cannot abort the run.
        if updates - entry.submitted > self.max_lag:
            redrawn = 1
            self.server.load(state.params, updates)
            self._submit(entry, key, attempt=redrawn)
            if updates - entry.submitted > self.max_lag:
                raise RuntimeError(f"the weights pushed at update {updates} did not take: the server still "
                                   f"serves version {entry.submitted}, past max_lag={self.max_lag}")
        scored = [[draw.result() for draw in row] for row in entry.draws]
        waited = time.perf_counter() - began
        versions = np.asarray([[draw.version for draw, _ in row] for row in scored], np.int32)
        oldest = int(versions.min())
        if updates - oldest > self.max_lag:
            raise RuntimeError(f"a draw reports version {oldest}, {updates - oldest} updates behind; "
                               f"max_lag is {self.max_lag}")
        packed = self._packed(state, entry, scored, versions, updates - oldest)
        if self.log is not None:
            self.log(RolloutRecord(updates, oldest, updates - oldest,
                                   float(np.mean([[value for _, value in row] for row in scored])), redrawn, waited))
        return packed

    def _packed(self, state: TrainState, entry: _Entry, scored: list[list[Scored]], versions: np.ndarray,
                lag: int) -> dict[str, np.ndarray]:
        rows, width = entry.prompts.shape
        budget, pad = self.max_new_tokens, self.server.sampling.pad_id
        sampled = np.full((rows, self.groups, budget), pad, np.int32)
        lengths = np.zeros((rows, self.groups), np.int32)
        terminated = np.zeros((rows, self.groups), bool)
        behavior = np.zeros((rows, self.groups, budget), np.float32)
        raw: list[tuple[float, ...] | None] = []
        rewards = np.zeros((rows, self.groups), np.float32)
        for row in range(rows):
            prompt = tuple(entry.prompts[row, width - int(entry.lengths[row]):].tolist())
            for group, (draw, value) in enumerate(scored[row]):
                count = len(draw.tokens)
                if draw.prompt != prompt or not 0 < count <= budget:
                    raise ValueError("the server returned a draw for another prompt or past the budget")
                sampled[row, group, :count] = draw.tokens
                lengths[row, group] = count
                terminated[row, group] = draw.terminated
                behavior[row, group, :count] = draw.behavior_log_probs
                raw.append(None if draw.raw_log_probs is None else tuple(draw.raw_log_probs))
                rewards[row, group] = value
        packed, _ = completion_rows(entry.prompts, entry.lengths, sampled, lengths, terminated, behavior,
                                    rewards, versions, self.estimator)
        if lag > 0 or any(values is None for values in raw):
            packed[OLD_LOG_PROBS_KEY] = np.asarray(self._rescore(state.params, packed), np.float32)
        else:
            packed[OLD_LOG_PROBS_KEY] = sampled_values(packed, lambda index, _: raw[index] or ())
        return packed
