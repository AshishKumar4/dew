"""Engine-sourced sessions and the strict prefix-merge packer.

A `Call` is one model call of a session, its `Draw` (`dew.inference.rollouts`)
as any recording gateway can supply it: the prompt ids the engine read, the
ids it sampled, their behavior log-probabilities, the finish reason and the
policy version served when the request was submitted. A `Session` is one harness session: its calls in
submission order, with the verifier's verdict. A `SessionSource` turns tasks
into sessions; Harbor runners, Polar clients and the in-process episode
collector all sit behind it.

`pack` turns sessions into one fixed-shape training batch. It merges call
k + 1 into the row of call k only when call k + 1's prompt ids start with
every id of that row so far: call 1's prompt, then each sampled id and each
interstitial id (tool output, template glue) since. Any other history starts
a new chain; nothing is re-tokenized and no lenient rule is tried, because a
prompt-only prefix check splices sampled ids into a context the model did
not see (research memo section 6). Chains are then packed first-fit into
`[rows, width]` with per-chain segment ids and positions, the layout
`LMObjective.token_scores` reads for packed documents.

Only sampled ids of trained sessions carry loss mass. `COMPLETED` and
`AGENT_ERROR` sessions (the agent broke; the verifier's reward says how
badly) always train on their reward. `INFRA_ERROR` and `CANCELLED` sessions
never do: they take no rows and enter no baseline, and the scheduler retries
them. A `TRUNCATED` session follows the `truncation` policy:

- `mask` (the default) drops it like an infrastructure failure. A context,
  turn or wall-clock limit says little about the task, so agentic runs mask
  them (DeepSWE's compact filtering, SkyRL-Agent, memo section 1.3).
- `score` trains it on its verifier reward, as a completed session. Fits
  single-turn RLVR where a budget-hit completion can still hold a gradable
  answer, and keeps short-budget runs from training on nothing.
- `zero` trains it on reward 0 whatever the verifier said, a penalty on
  running long (SkyRL's `zero_reward_on_non_stop`).
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, field, fields
from enum import Enum
from types import MappingProxyType
from typing import Protocol, SupportsFloat

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from dew._records import finite, json_value, record as section
from dew.batch import Variables
from dew.data.tokens import first_fit
from dew.inference.rollouts import Draw
from dew.nn.inputs import local_rows
from dew.rl import group_advantage, rloo_advantage

IDS_KEY = "input_ids"
RESPONSE_MASK_KEY = "response_mask"
"""1 on every sampled id of a session that trains, the tokens that carry loss mass."""
OLD_LOG_PROBS_KEY = "old_log_probs"
"""The proximal policy's log-probabilities, recorded or rescored before the update.

GRPO's PPO ratio compares the current raw policy with this one. When a batch
has none, the behavior log-probabilities stand in.
"""
BEHAVIOR_LOG_PROBS_KEY = "behavior_log_probs"
"""Actual sampling log-probabilities, including temperature/top-k and greedy selection."""
ADVANTAGES_KEY = "advantages"
TOKEN_REWARDS_KEY = "token_rewards"
"""On a sampled id, the reward that lands on it: each call's step reward on its last sampled id, and
the session's trained reward (`pack`'s `truncation`) on its last call's; 0 elsewhere. verl's
token_level_rewards, which a critic's GAE reads."""
SEGMENT_IDS_KEY = "text_segment_ids"
"""Which chain of its row each token belongs to, from 1; 0 is padding."""
POSITIONS_KEY = "text_positions"
"""Each token's position inside its chain, from 0."""
VERSIONS_KEY = "versions"
"""On a sampled id, the policy version its call was submitted under; -1 elsewhere."""
SESSION_INDEX_KEY = "session_index"
"""Which of `pack`'s sessions each chain token comes from; -1 on padding."""
CALL_INDEX_KEY = "call_index"
"""On a sampled id, which call of its session sampled it; -1 elsewhere."""
SESSION_WEIGHTS_KEY = "session_weights"
"""On a trainable id, one over its session's trainable-id count; 0 elsewhere.
Summed over a batch it counts sessions, so `sum(weights * terms)` over
`sum(weights)` is the mean over sessions of each session's token mean."""
ROUTED_EXPERTS_KEY = "routed_experts"
"""`[rows, width, layers, top_k]` expert ids the engine routed each id to, for replay."""
ROUTED_KEY = "routed"
"""Where `routed_experts` holds a record; elsewhere the trainer's router chooses."""
SUPPORT_KEY = "support_ids"
"""`[rows, capacity]`: each row's kept ids of its filtered sampled ids, back to
back, padded with -1."""
SUPPORT_COLUMNS_KEY = "support_columns"
"""`[rows, capacity]`: for each `support_ids` entry, the column of the sampled
id whose support it belongs to; -1 on padding. A sampled id with no entry was
drawn from the whole vocabulary."""

TRUNCATIONS = ("mask", "score", "zero")
"""What a `TRUNCATED` session trains on; see the module docstring."""
ESTIMATORS = ("group", "mean", "rloo")
"""Advantage families: `group` centres on the group mean and divides by its
deviation (GRPO), `mean` only centres (Dr.GRPO), `rloo` subtracts the mean
of the other members."""
CREDITS = ("session", "turn")
"""What an advantage is computed over: `session`, one per session from its
return, every call alike; `turn`, one per call from the return still ahead of
it, compared with the group's members' at the same turn."""



def finite_reward(value: SupportsFloat) -> float:
    """A verifier's or reward's score as a float: anything `float` takes (a bool's 1.0 and 0.0, a
    NumPy or JAX scalar), refused unless finite (`records.finite`)."""
    return finite(float(value), "a reward")


def local_task_ids(batch: Mapping[str, ArrayLike]) -> np.ndarray:
    """This process's `task_id` rows of `batch`, refused unless a nonempty vector of integers: the
    rollout's task boundary, which the scheduler and native episode collection both read."""
    ids = local_rows(batch["task_id"])
    if ids.ndim != 1 or not ids.size or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("task_id must be a nonempty vector of integer task identities")
    return ids

class Status(Enum):
    """How a session ended, which decides whether it trains.

    `COMPLETED` and `AGENT_ERROR` sessions always train on their reward; an
    agent error means the agent broke, and the verifier's reward says how
    badly. `INFRA_ERROR` and `CANCELLED` sessions never train, and the
    scheduler retries them. A `TRUNCATED` session follows `pack`'s
    `truncation` policy.
    """

    COMPLETED = "completed"
    TRUNCATED = "truncated"
    AGENT_ERROR = "agent_error"
    INFRA_ERROR = "infra_error"
    CANCELLED = "cancelled"

    @property
    def trainable(self) -> bool:
        """Whether a session with this status is scored and always carries loss mass.

        It is False for `TRUNCATED`, because whether a truncated session
        trains depends on `pack`'s `truncation` policy.
        """
        return self in (Status.COMPLETED, Status.AGENT_ERROR)


@dataclass(frozen=True)
class Transcript:
    """One call's conversation in the structure the harness spoke it, as JSON.

    `messages` are the chat messages the call was asked with, those the
    harness sent or those the environment's observations carried so far, and
    `response` is the message the engine answered with, tool calls included,
    or None where the source holds no structured answer (its sampled ids
    are the answer). The call's ids stay its training record: nothing is
    tokenized from a transcript, and it is kept for reading a session, not
    for scoring it.
    """

    messages: tuple[Mapping[str, object], ...]
    response: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "messages", json_objects(self.messages, "a transcript's messages"))
        if self.response is not None:
            object.__setattr__(self, "response", json_objects((self.response,), "a transcript's response")[0])


def json_objects(values: object, name: str) -> tuple[Mapping[str, object], ...]:
    """`values`, a tuple of JSON objects in any `Mapping`, as plain JSON trees (`json_value`), which
    journals copy and write as they are; ValueError for anything else."""
    if not isinstance(values, tuple) or not all(isinstance(value, Mapping) for value in values):
        raise ValueError(f"{name} are a tuple of JSON objects")
    return tuple(section(json_value(value, name), name) for value in values)


@dataclass(frozen=True)
class Call(Draw):
    """One model call of a session: its `Draw` as the engine served it, with what the environment
    paid for it and the conversation it was asked in.

    `reward` is what the environment paid for this call, a step reward, or
    None. `transcript` is the call's conversation in the structure the
    harness spoke it (`Transcript`), where the source has one.
    """

    reward: float | None = None
    transcript: Transcript | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.reward is not None:
            object.__setattr__(self, "reward", finite(self.reward, "a call's reward"))

    @classmethod
    def of(cls, draw: Draw, *, reward: float | None = None, transcript: Transcript | None = None) -> Call:
        """`draw` as a session's call, paid `reward` and asked in `transcript`."""
        return cls(**{name.name: getattr(draw, name.name) for name in fields(Draw)}, reward=reward,
                   transcript=transcript)


@dataclass(frozen=True)
class Task:
    """One unit of work a session source runs, with its id and a source-specific payload in `data`."""

    id: str
    data: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


@dataclass(frozen=True)
class Session:
    """One harness session, with its calls in submission order and how it ended.

    `(task, group)` names the advantage group, and `sample` and `attempt`
    tell its members and their retries apart. `reward` is the verifier's
    score, required when the status is trainable (`Status.trainable`).
    `components` holds the verifier's sub-scores for logging, and `detail`
    says where the verdict or the failure came from.
    """

    task: str
    group: str
    sample: int
    attempt: int
    calls: tuple[Call, ...]
    status: Status
    reward: float | None
    components: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    detail: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.status, Status):
            raise TypeError("session status must be a Status")
        if not isinstance(self.calls, tuple) or not all(isinstance(call, Call) for call in self.calls):
            raise TypeError("session calls must be a tuple of Call records")
        if self.reward is not None:
            object.__setattr__(self, "reward", finite(self.reward, "a session reward"))
        if self.status.trainable and self.reward is None:
            raise ValueError(f"a {self.status.name} session is scored, so it needs a reward")
        for name in ("sample", "attempt"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"session {name} must be a nonnegative integer")


@dataclass(frozen=True)
class Snapshot:
    """The trainer's state at one rollout call, as an acquisition reads it: the `variables` the
    step trains, its attempted `step` and committed `updates`, and the call's `key`."""

    variables: Variables
    step: int
    updates: int
    key: jax.Array


class Collector(Protocol):
    """Draws a call's sessions from the trainer's own variables (`RolloutScheduler.collected`).

    `collect` draws `samples` sessions of each task from `snapshot.variables`
    in one program every process of their mesh enters in the same order, and
    returns them by task; every call it records was drawn under
    `snapshot.updates`, with the raw likelihoods of exactly those variables,
    the proximal policy. A failure that is not the drawn sessions' own (a
    broken environment) raises on every process, and an infrastructure
    failure of a session's reward or environment is that session's
    INFRA_ERROR. On a data share's later reader it enters the draw but
    neither scores nor returns it, an empty list: the share trains on its
    first reader's sessions (`first_reader_batch`). `width` is the packed
    row's ids and `rows(tasks, samples)` the rows a call of that many
    sessions packs into.
    """

    @property
    def width(self) -> int: ...

    def rows(self, tasks: int, samples: int) -> int: ...

    def collect(self, snapshot: Snapshot, tasks: Sequence[Task],
                samples: int) -> Sequence[Sequence[Session]]: ...


class SessionSource(Protocol):
    """Anything that turns tasks into sessions.

    `submit` starts `samples` sessions of one task under the served policy
    `version` and returns one future per session. A source that makes its
    sessions' draws makes session k's under `fold_in(key, k)` and no other
    seed, so the trainer's checkpointed key is the one authority over what
    a run samples; an external harness's draws (Harbor) are its own.
    `cancel` stops the sessions whose results are no longer wanted.
    """

    def submit(self, task: Task, samples: int, *, version: int,
               key: jax.Array) -> Sequence[Future[Session]]: ...

    def cancel(self, futures: Sequence[Future[Session]]) -> None: ...


@dataclass
class _Chain:
    """One append-only token chain of a session, as it is being built."""

    session: int
    tokens: list[int] = field(default_factory=list)
    behavior: list[float] = field(default_factory=list)
    versions: list[int] = field(default_factory=list)
    calls: list[int] = field(default_factory=list)
    """Per token, the call that sampled it, or -1 for a prompt or interstitial id."""
    members: list[tuple[int, int, int]] = field(default_factory=list)
    """Each call this chain holds: its index, where its sampled ids start in
    the chain and how many it sampled, which may be none."""
    routing: np.ndarray | None = None
    """The latest recorded call's `routed_experts`. Once a call is appended
    the chain is exactly that call's ids, so its record row p is chain
    token p, and it covers every row an earlier call's record did."""
    supports: list[tuple[int, ...] | None] = field(default_factory=list)
    """Per token, the ids its sampler kept, or None."""

    def continues(self, call: Call) -> bool:
        """Whether the token prefix and its recorded routing agree with `call`."""
        return (merges(self.tokens, call)
                and (self.routing is None or call.routed_experts is None
                     or np.array_equal(self.routing, call.routed_experts[:len(self.routing)])))

    def extend(self, call: Call, index: int, start: int) -> None:
        """Append `call`'s prompt from `start`, then its sampled ids."""
        tail = len(call.prompt) - start
        self.tokens.extend(call.prompt[start:])
        self.behavior.extend([0.0] * tail)
        self.versions.extend([-1] * tail)
        self.calls.extend([-1] * tail)
        self.tokens.extend(call.tokens)
        self.behavior.extend(call.behavior_log_probs)
        self.versions.extend([call.version] * len(call.tokens))
        self.calls.extend([index] * len(call.tokens))
        self.members.append((index, len(self.tokens) - len(call.tokens), len(call.tokens)))
        self.supports.extend([None] * tail)
        self.supports.extend(call.support if call.support is not None else [None] * len(call.tokens))
        if call.routed_experts is not None:
            self.routing = call.routed_experts


def merges(chain: Sequence[int], call: Call) -> bool:
    """Whether `call` continues `chain` under the strict rule.

    The call's prompt must start with every id of the chain, sampled ids
    included. A prompt that only starts with the previous prompt is refused.
    """
    size = len(chain)
    return len(call.prompt) >= size and tuple(call.prompt[:size]) == tuple(chain)


def chains(session: Session, index: int, width: int) -> list[_Chain]:
    """Split one session's calls into strict append-only chains no wider than `width`.

    Each chain records `index`, the session's number among those `pack`
    reads. A call that does not continue the current chain starts a new
    chain from its own full prompt. A single call wider than `width` is
    refused: dropping it would hide a session the scheduler admitted. A
    merged chain is never wider than its last call, whose prompt holds the
    whole chain, so fitting each call fits the chain.
    """
    built: list[_Chain] = []
    for number, call in enumerate(session.calls):
        size = len(call.prompt) + len(call.tokens)
        if size > width:
            raise ValueError(
                f"session {session.task}/{session.group}/{session.sample} call {number} holds "
                f"{size} ids, wider than the {width}-id row")
        current = built[-1] if built else None
        if current is not None and current.continues(call):
            current.extend(call, number, len(current.tokens))
        else:
            chain = _Chain(index)
            chain.extend(call, number, 0)
            built.append(chain)
    return built


def check_truncation(truncation: str) -> None:
    """Refuse a truncation policy `pack` does not apply."""
    if truncation not in TRUNCATIONS:
        raise ValueError(f"truncation must be one of {TRUNCATIONS}, got {truncation!r}")


def _trains(status: Status, truncation: str) -> bool:
    """Whether a session that ended with `status` trains under `truncation`: a trainable status
    always, a truncated one unless its truncation policy masks it."""
    return status.trainable or (status is Status.TRUNCATED and truncation != "mask")


def _trained_reward(session: Session, truncation: str) -> float | None:
    """The reward `session` trains on under `truncation`, or None when it carries no loss."""
    if session.status.trainable:
        return session.reward
    if not _trains(session.status, truncation):
        return None
    if truncation == "zero":
        return 0.0
    if session.reward is None:
        raise ValueError(f"truncation='score' trains session {session.task}/{session.group}/{session.sample} "
                         "on its reward, and it has none")
    return session.reward


def check_estimator(estimator: str) -> None:
    """Refuse an advantage family `advantages` does not compute."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")


def check_credit(credit: str) -> None:
    """Refuse a credit `advantages` does not assign."""
    if credit not in CREDITS:
        raise ValueError(f"credit must be one of {CREDITS}, got {credit!r}")


def advantages(sessions: Sequence[Session], estimator: str = "group", *,
               truncation: str = "mask", credit: str = "session") -> list[np.ndarray]:
    """Each session's advantage at each of its calls, from the returns of its `(task, group)`.

    A session's return is the reward it trains on plus every reward its
    calls earned (`Call.reward`); from a call on, it is that reward plus the
    rewards of that call and the ones after it. With `credit="session"` the
    estimator compares sessions' returns and every call of a session takes
    its session's advantage; with `credit="turn"` it compares the members'
    returns from their call `t` on, for each turn `t`, and each call takes
    its own. The baseline reads the trained members only, each at its
    trained return. A masked member has no score to compare against, and a
    comparison of fewer than two members has no baseline, so every member
    there gets zero.
    """
    check_estimator(estimator)
    check_truncation(truncation)
    check_credit(credit)
    # The trained sessions' returns from each call on, by session.
    ahead = {index: _returns_ahead(session, reward) for index, session in enumerate(sessions)
             if (reward := _trained_reward(session, truncation)) is not None}
    # Who is compared with whom, and the return each one is compared on.
    compared: dict[tuple[str, str, int], list[tuple[int, int]]] = {}
    for index in ahead:
        session = sessions[index]
        # A session without calls has no turn to compare, and its return stands only for itself.
        turns = range(len(session.calls)) if credit == "turn" else (0,)
        for turn in turns:
            compared.setdefault((session.task, session.group, turn), []).append((index, turn))
    values = {key: 0.0 for members in compared.values() for key in members}
    by_size: dict[int, list[list[tuple[int, int]]]] = {}
    for members in compared.values():
        if len(members) >= 2:
            by_size.setdefault(len(members), []).append(members)
    for size, groups in by_size.items():
        order = [member for members in groups for member in members]
        rewards = jnp.asarray([ahead[index][turn] for index, turn in order], jnp.float32)
        if estimator == "rloo":
            computed = rloo_advantage(rewards, size)
        else:
            computed = group_advantage(rewards, size, normalise_by_std=estimator == "group")
        values.update(zip(order, np.asarray(jax.device_get(computed), np.float32).tolist(), strict=True))
    per_call = []
    for index, session in enumerate(sessions):
        calls = len(session.calls)
        if index not in ahead:
            per_call.append(np.zeros(calls, np.float32))
        elif credit == "turn":
            per_call.append(np.asarray([values[(index, turn)] for turn in range(calls)], np.float32))
        else:
            per_call.append(np.full(calls, values[(index, 0)], np.float32))
    return per_call


def _returns_ahead(session: Session, reward: float) -> list[float]:
    """From each call on, the session's trained `reward` plus the rewards of that call and every
    call after it; a session without calls returns `reward` alone, at turn 0."""
    remaining, returns = reward, []
    for call in reversed(session.calls):
        remaining += 0.0 if call.reward is None else call.reward
        returns.append(remaining)
    return returns[::-1] or [reward]


def _owners(chain: _Chain) -> np.ndarray:
    """Per chain id, the call it belongs to: its sampled ids, and the prompt ids before them that
    no earlier call of the chain holds."""
    owners = np.empty(len(chain.tokens), np.int64)
    start = 0
    for call, first, count in chain.members:
        owners[start:first + count] = call
        start = first + count
    owners[start:] = chain.members[-1][0]
    return owners


def _built(sessions: Sequence[Session], width: int, truncation: str) -> list[_Chain]:
    """The chains of every session that trains under `truncation`, in session order."""
    return [
        chain
        for index, session in enumerate(sessions)
        if _trained_reward(session, truncation) is not None
        for chain in chains(session, index, width)
    ]


def _place(lengths: Sequence[int], width: int) -> list[list[int]]:
    """Place chains first-fit in decreasing length, a stable order: per row, its chain numbers.

    With a window open per chain, `first_fit` never closes one, so it is plain first-fit.
    """
    order = sorted(range(len(lengths)), key=lambda number: -lengths[number])
    grouped, starts = first_fit(np.asarray(lengths, np.int64)[order], width, max(len(order), 1))
    return [[order[index] for index in grouped[start:stop]] for start, stop in itertools.pairwise(starts)]


def chain_lengths(sessions: Sequence[Session], width: int, *, truncation: str) -> list[int]:
    """The length of every chain `pack` builds from `sessions` under `truncation`."""
    return [len(chain.tokens) for chain in _built(sessions, width, truncation)]


def rows_needed(lengths: Sequence[int], width: int) -> int:
    """How many `width`-id rows `pack` fills with chains of `lengths`."""
    return len(_place(lengths, width))


def pack(sessions: Sequence[Session], width: int, *, rows: int | None = None,
         estimator: str = "group", truncation: str = "mask", credit: str = "session",
         support_capacity: int | None = None) -> dict[str, np.ndarray]:
    """Strictly merge each trained session's calls, then pack the chains into `[rows, width]`.

    A call joins the chain of the call before it only when its prompt ids
    start with every id of that chain so far, sampled ids included;
    otherwise it starts a new chain. Nothing is re-tokenized, and a single
    call wider than `width` raises `ValueError`.

    Every array but the engine records below is `[rows, width]` and aligned
    with `input_ids`, so entry t describes id t. `text_segment_ids` numbers
    the chains of a row from 1, with 0 on padding, and `text_positions`
    counts each chain's positions from 0. `response_mask` is 1 on sampled
    ids alone, and `behavior_log_probs`, `versions` and `call_index` are set
    on those ids. `advantages` holds, on each sampled id, its call's
    advantage, which `estimator` computes within its `(task, group)` over
    `credit` (`advantages`), and `token_rewards` the rewards that land on
    each sampled id (`TOKEN_REWARDS_KEY`).
    `session_weights` is one over the session's number of sampled ids on
    each of them and 0 elsewhere. It sums to the number of sessions, so
    `sum(weights * terms) / sum(weights)` is the mean over sessions of each
    session's token mean. Chains are placed first-fit in decreasing length,
    a stable order. `rows` pads the batch to a fixed count and raises
    `ValueError` when the chains need more (`rows_needed` over
    `chain_lengths` counts them).

    `truncation` decides what a `TRUNCATED` session trains on: `mask` (the
    default) drops it, `score` trains it on its verifier reward, and `zero`
    trains it on reward 0. `COMPLETED` and `AGENT_ERROR` sessions always
    train, and `INFRA_ERROR` and `CANCELLED` sessions never do.

    Calls that recorded `routed_experts` add that array,
    `[rows, width, layers, top_k]`, and `routed`, which is true where a
    call's record covers the id. Calls that recorded a `support` add
    `support_ids` and `support_columns`: each row's kept ids back to back,
    and for each one the column of the sampled id it belongs to, both
    padded with -1. `support_capacity`, required when any call recorded a
    support, fixes the per-row length of these two arrays, so every batch
    has one shape and the step compiles once.
    """
    if type(width) is not int or width < 2:
        raise ValueError("a packed row holds at least two ids")
    if rows is not None and (type(rows) is not int or rows < 1):
        raise ValueError("rows is a positive integer, or None for as many as the chains need")
    values = advantages(sessions, estimator, truncation=truncation, credit=credit)
    built = _built(sessions, width, truncation)
    placed = _place([len(chain.tokens) for chain in built], width)
    count = len(placed) if rows is None else rows
    if len(placed) > count:
        raise ValueError(f"the chains need {len(placed)} rows of {width} ids, more than the {rows} asked for")
    count = max(count, 1)
    shape = (count, width)
    ids = np.zeros(shape, np.int32)
    segments = np.zeros(shape, np.int32)
    positions = np.zeros(shape, np.int32)
    mask = np.zeros(shape, np.float32)
    behavior = np.zeros(shape, np.float32)
    versions = np.full(shape, -1, np.int32)
    session_index = np.full(shape, -1, np.int32)
    call_index = np.full(shape, -1, np.int32)
    advantage = np.zeros(shape, np.float32)
    token_rewards = np.zeros(shape, np.float32)
    trainable = np.zeros(len(sessions), np.int64)
    for chain in built:
        trainable[chain.session] += sum(1 for call in chain.calls if call >= 0)
    weights = np.zeros(shape, np.float32)
    for row, segment, start, chain in _spans(built, placed):
        span = slice(start, start + len(chain.tokens))
        sampled = np.asarray(chain.calls) >= 0
        ids[row, span] = chain.tokens
        segments[row, span] = segment
        positions[row, span] = np.arange(len(chain.tokens))
        mask[row, span] = sampled
        behavior[row, span] = chain.behavior
        versions[row, span] = chain.versions
        session_index[row, span] = chain.session
        call_index[row, span] = chain.calls
        advantage[row, span] = values[chain.session][_owners(chain)]
        weights[row, span] = sampled / max(int(trainable[chain.session]), 1)
        session = sessions[chain.session]
        for call, first, count in chain.members:
            if count:
                paid = session.calls[call].reward
                trained = _trained_reward(session, truncation) if call == len(session.calls) - 1 else None
                token_rewards[row, start + first + count - 1] = ((0.0 if paid is None else paid)
                                                                 + (0.0 if trained is None else trained))
    return {
        IDS_KEY: ids, SEGMENT_IDS_KEY: segments, POSITIONS_KEY: positions,
        RESPONSE_MASK_KEY: mask, BEHAVIOR_LOG_PROBS_KEY: behavior, VERSIONS_KEY: versions,
        SESSION_INDEX_KEY: session_index, CALL_INDEX_KEY: call_index,
        ADVANTAGES_KEY: advantage, TOKEN_REWARDS_KEY: token_rewards, SESSION_WEIGHTS_KEY: weights,
        **_engine_records(built, placed, shape, support_capacity),
    }


def _spans(built: Sequence[_Chain], placed: Sequence[Sequence[int]]):
    """Each placed chain as `(row, segment, start, chain)`: its packed row, its 1-based segment in
    the row, the column its first id takes, and the chain, in packing order."""
    for row, numbers in enumerate(placed):
        start = 0
        for segment, number in enumerate(numbers, start=1):
            chain = built[number]
            yield row, segment, start, chain
            start += len(chain.tokens)


def _empty_routing(shape: tuple[int, int], layout: tuple[int, ...], dtype) -> dict[str, np.ndarray]:
    """The routing record's empty form over packed rows of `shape`: expert ids of zero `[*shape,
    layers, top_k]` in the engines' dtype, and `routed` false, which the trainer reads as no record."""
    return {ROUTED_EXPERTS_KEY: np.zeros((*shape, *layout), dtype), ROUTED_KEY: np.zeros(shape, bool)}


def _empty_support(rows: int, capacity: int) -> dict[str, np.ndarray]:
    """The support record's empty form: `rows` rows of `capacity` kept ids and their columns, -1."""
    return {SUPPORT_KEY: np.full((rows, capacity), -1, np.int32),
            SUPPORT_COLUMNS_KEY: np.full((rows, capacity), -1, np.int32)}


def _engine_records(built: Sequence[_Chain], placed: Sequence[Sequence[int]],
                    shape: tuple[int, int], support_capacity: int | None) -> dict[str, np.ndarray]:
    """The packed routing and support arrays, present only when a call recorded them.

    `routed_experts` is `[rows, width, layers, top_k]` in the engines' dtype,
    with `routed` true where a call's record covers the id. The supports are
    ragged within each row, as vLLM's `sampling_mask` and slime's
    `top_p_token_ids` with offsets keep them: row r of `support_ids` holds the
    kept ids of row r's filtered sampled ids back to back and row r of
    `support_columns` the column each belongs to, both padded with -1 to
    `support_capacity`, so the arrays shard with the rows they describe.
    """
    records = [chain.routing for chain in built if chain.routing is not None]
    out: dict[str, np.ndarray] = {}
    if records:
        layout = {record.shape[1:] for record in records}
        if len(layout) != 1:
            raise ValueError(f"routed_experts disagree on [layers, top_k] across calls: {sorted(layout)}")
        out = _empty_routing(shape, layout.pop(), np.result_type(*records))
    kept: list[list[tuple[int, ...]]] = [[] for _ in range(shape[0])]
    owners: list[list[int]] = [[] for _ in range(shape[0])]
    for row, _, start, chain in _spans(built, placed):
        if out and chain.routing is not None:
            rows_covered = len(chain.routing)
            out[ROUTED_EXPERTS_KEY][row, start:start + rows_covered] = chain.routing
            out[ROUTED_KEY][row, start:start + rows_covered] = True
        for position, ids in enumerate(chain.supports):
            if ids is not None:
                kept[row].append(ids)
                owners[row].append(start + position)
    if any(kept):
        totals = [sum(len(group) for group in groups) for groups in kept]
        if support_capacity is None:
            raise ValueError(
                "recorded supports pack to [rows, support_capacity]; pass a support_capacity "
                "(at least top_k times the sampled ids a row holds) so every batch has one shape"
            )
        if max(totals) > support_capacity:
            raise ValueError(f"a row's recorded supports keep {max(totals)} ids, "
                             f"more than support_capacity {support_capacity}")
        support = _empty_support(shape[0], support_capacity)
        for row, (groups, total) in enumerate(zip(kept, totals, strict=True)):
            support[SUPPORT_KEY][row, :total] = np.fromiter((token for group in groups for token in group),
                                                            np.int32, total)
            support[SUPPORT_COLUMNS_KEY][row, :total] = np.repeat(
                np.asarray(owners[row], np.int32), [len(group) for group in groups]
            )
        out.update(support)
    return out


def agreed_records(packed: dict[str, np.ndarray], support_capacity: int | None) -> dict[str, np.ndarray]:
    """`packed`, holding every engine record any process of the pool packed.

    `pack` adds a record only where a call recorded it, so a data share whose
    calls recorded none (all of them truncated, say) packs a different tree
    than a share whose calls did, and the step cannot place the pool's rows
    (`shard_batch` takes one tree from every process). A record another
    process packed and this one did not holds its empty form here: `routed`
    false over expert ids of zero, and supports of -1, which the trainer
    reads as no record. The routing's `[layers, top_k]` and dtype come from
    the processes that packed it, which must agree. A process that packed
    nothing, a share's later reader, keeps nothing. Every process calls
    this, in the order of its other coordination rounds.
    """
    from dew._coordination import from_every_process

    routed = packed.get(ROUTED_EXPERTS_KEY)
    pool = from_every_process({"routing": None if routed is None else [*routed.shape[2:], routed.dtype.name],
                               "support": SUPPORT_KEY in packed})
    layouts = {tuple(share["routing"]) for share in pool if share["routing"] is not None}
    if len(layouts) > 1:
        raise ValueError(f"the pool's routed_experts disagree on [layers, top_k] or dtype: {sorted(layouts)}")
    if not packed:
        return packed
    shape = np.shape(packed[IDS_KEY])
    filled = dict(packed)
    if layouts and routed is None:
        (layers, top_k, dtype), = layouts
        filled.update(_empty_routing(shape, (layers, top_k), np.dtype(dtype)))
    if any(share["support"] for share in pool) and SUPPORT_KEY not in packed:
        assert support_capacity is not None, "a peer packed supports, which need a support_capacity"
        filled.update(_empty_support(shape[0], support_capacity))
    return filled


def sampled_values(batch: Mapping[str, np.ndarray],
                   values: Callable[[int, int], Sequence[float]]) -> np.ndarray:
    """Scatter per-call host values, one per sampled id, into the packed layout.

    `values(index, call)` returns one float per sampled id of call `call`
    of the `index`-th session handed to `pack`, such as the raw-policy
    log-probabilities an in-process sampler recorded. A call's sampled ids
    sit in one chain in order, so row-major order of its
    `(session_index, call_index)` positions is their order in the call.
    """
    session_index = np.asarray(batch[SESSION_INDEX_KEY])
    call_index = np.asarray(batch[CALL_INDEX_KEY])
    out = np.zeros(session_index.shape, np.float32)
    flat_session, flat_call = session_index.reshape(-1), call_index.reshape(-1)
    where = np.flatnonzero(flat_call >= 0)
    keys = flat_session[where].astype(np.int64) * (1 << 32) + flat_call[where]
    order = np.argsort(keys, kind="stable")
    boundaries = np.flatnonzero(np.diff(keys[order])) + 1
    flat = out.reshape(-1)
    for group in np.split(order, boundaries):
        if not group.size:
            continue
        position = where[group[0]]
        given = np.asarray(values(int(flat_session[position]), int(flat_call[position])), np.float32)
        if given.shape != (group.size,):
            raise ValueError("values must return one float per sampled id of the call")
        flat[where[group]] = given
    return out


def session_metrics(sessions: Sequence[Session], batch: Mapping[str, np.ndarray], *,
                    latencies: Sequence[float] | None = None,
                    version: int | None = None, truncation: str = "mask") -> dict[str, float]:
    """Host-side agentic telemetry for one packed batch and the sessions behind it.

    - `merge/calls_per_chain`: trainable calls over packed chains, 1.0 when
      nothing merged; `pack/fill` is the share of row slots holding ids.
    - `status/<name>`: share of sessions per status, and
      `masked/<name>`: share of all sampled ids that status masked, under
      the `truncation` policy `pack` used.
    - `reward/mean` over trained sessions at their trained reward, and
      `reward/component/<name>` per verifier component; `reward/observed`
      the mean of every verifier score, whatever the session trains on, 0.0
      where none was scored.
    - `length/mean`, the mean count of sampled ids per session.
    - `latency/p50`, `p90`, `p99`, `max` over `latencies`, seconds per session.
    - `lag/mean`, `lag/max`: `version` minus each trainable id's version.

    Trainer-versus-engine mismatch is the loss's own metric (`mismatch/*`),
    computed where the proximal policy is known.
    """
    rewards_of = [_trained_reward(session, truncation) for session in sessions]
    metrics: dict[str, float] = {}
    total = max(len(sessions), 1)
    sampled = dict.fromkeys(Status, 0)
    for session in sessions:
        sampled[session.status] += sum(len(call.tokens) for call in session.calls)
    everything = max(sum(sampled.values()), 1)
    for status in Status:
        metrics[f"status/{status.value}"] = sum(session.status == status for session in sessions) / total
        if not _trains(status, truncation):
            metrics[f"masked/{status.value}"] = sampled[status] / everything
    segments = np.asarray(batch[SEGMENT_IDS_KEY])
    rows = np.arange(segments.shape[0])[:, None] * (segments.shape[1] + 1) + segments
    chain_count = np.unique(rows[segments > 0]).size
    calls = sum(len(session.calls) for session, reward in zip(sessions, rewards_of, strict=True)
                if reward is not None)
    metrics["merge/calls_per_chain"] = calls / chain_count if chain_count else 0.0
    metrics["pack/fill"] = float(np.mean(segments > 0))
    scored = [(session, reward) for session, reward in zip(sessions, rewards_of, strict=True)
              if reward is not None]
    observed = [session.reward for session in sessions if session.reward is not None]
    metrics["reward/observed"] = float(np.mean(observed, dtype=np.float32)) if observed else 0.0
    metrics["length/mean"] = float(np.mean([sum(len(call.tokens) for call in session.calls)
                                            for session in sessions])) if sessions else 0.0
    if scored:
        metrics["reward/mean"] = float(np.mean([reward for _, reward in scored]))
        components: dict[str, list[float]] = {}
        for session, _ in scored:
            for name, value in session.components.items():
                components.setdefault(name, []).append(float(value))
        metrics.update(
            {f"reward/component/{name}": float(np.mean(values)) for name, values in components.items()}
        )
    if latencies:
        seconds = np.asarray(latencies, np.float64)
        for label, quantile in (("p50", 50), ("p90", 90), ("p99", 99)):
            metrics[f"latency/{label}"] = float(np.percentile(seconds, quantile))
        metrics["latency/max"] = float(seconds.max())
    mask = np.asarray(batch[RESPONSE_MASK_KEY]) != 0
    if version is not None and mask.any():
        lag = version - np.asarray(batch[VERSIONS_KEY])[mask]
        metrics["lag/mean"], metrics["lag/max"] = float(lag.mean()), float(lag.max())
    return metrics
