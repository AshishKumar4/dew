"""Single-turn prompt tasks, scored draws and the collector that draws them from the trainer's policy."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Sequence

import jax.numpy as jnp
import numpy as np

from dew._coordination import agreed
from dew.batch import Batch
from dew.data.dataset import DataPartition
from dew.data.prompts import INFO_KEY, LENGTH_KEY, PROMPT_KEY, SOURCE_KEY, TRUTH_KEY, PromptRecord
from dew.inference.rollouts import Draw, InfrastructureError, response
from dew.nn.inputs import ModelInputs, local_rows, mesh_of, pad_token_rows
from dew.sampling.text import Sampling

from ..lm import TokenScorer
from .sessions import Call, Session, Snapshot, Status, Task, finite_reward

type Reward = Callable[[str, PromptRecord], float]
"""A function that scores a completion against its prompt's `PromptRecord` and returns the reward.

verl's four-argument `compute_score` enters through `dew.objectives.rl.verl.verl_reward`."""


def scored_draw(task: str, sample: int, call: Call, text: str, record: PromptRecord,
                reward: Reward) -> Session:
    """One draw's one-call session: COMPLETED where its call stopped, TRUNCATED where it ran out,
    scored by `reward(text, record)` on its decoded `text`. A reward whose external boundary failed
    (an `InfrastructureError` or an `OSError`: a sandbox, a service) leaves an INFRA_ERROR session
    naming it, which a scheduler retries; a non-finite score is refused."""
    try:
        score = reward(text, record)
    except (InfrastructureError, OSError) as error:
        return Session(task, "", sample, 0, (call,), Status.INFRA_ERROR, None, {},
                       f"reward: {type(error).__name__}: {error}")
    status = Status.COMPLETED if call.terminated else Status.TRUNCATED
    return Session(task, "", sample, 0, (call,), status, finite_reward(score))


def prompt_rows(batch) -> tuple[np.ndarray, np.ndarray, list[PromptRecord]]:
    """Read one host prompt batch: left-padded ids, lengths and each row's `PromptRecord`.

    Every row needs a length between one and the width. The strings arrive
    as fixed-width UTF-8 byte rows stored as int32.
    """
    prompts = local_rows(batch[PROMPT_KEY])
    prompt_lengths = local_rows(batch[LENGTH_KEY])
    sources, truths, infos = ([bytes(row[row != 0].astype(np.uint8)).decode("utf-8")
                               for row in np.asarray(local_rows(batch[name]), np.int32)]
                              for name in (SOURCE_KEY, TRUTH_KEY, INFO_KEY))
    rows, width = prompts.shape
    if (prompt_lengths.shape != (rows,) or not np.issubdtype(prompt_lengths.dtype, np.integer)
            or np.any(prompt_lengths < 1) or np.any(prompt_lengths > width)):
        raise ValueError("prompt_length must contain one valid integer length per row")
    records = [PromptRecord(*fields) for fields in zip(sources, truths, infos, strict=True)]
    return prompts, prompt_lengths, records


def prompt_tasks(batch: Batch) -> list[Task]:
    """Return one single-turn task per prompt row, with its prompt ids, its `PromptRecord` and the
    width the batch pads its prompts to, which a collector drawing them together pads to again.

    A task's id is a digest of its prompt ids, so repeated prompts share an
    id; their groups keep them apart.
    """
    prompts, lengths, records = prompt_rows(batch)
    rows, width = prompts.shape
    tasks = []
    for row in range(rows):
        ids = tuple(int(token) for token in prompts[row, width - int(lengths[row]):])
        name = hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()[:16]
        tasks.append(Task(name, {"prompt": ids, "record": records[row], "width": width}))
    return tasks


@dataclasses.dataclass(frozen=True)
class PromptTask:
    """What a single-turn prompt task holds (`prompt_tasks`): its prompt ids under `prompt`, the
    `PromptRecord` its reward reads under `record`, and under `width` the width its batch padded
    it to, which a task built outside a batch need not state: its own length is its width."""

    ids: tuple[int, ...]
    record: PromptRecord
    width: int

    @classmethod
    def of(cls, task: Task) -> PromptTask:
        ids, record = task.data.get("prompt"), task.data.get("record")
        if not (isinstance(ids, tuple) and isinstance(record, PromptRecord)):
            raise TypeError(f"task {task.id!r} is not a prompt task: it needs prompt ids and a PromptRecord")
        width = task.data.get("width", len(ids))
        if type(width) is not int or width < len(ids):
            raise TypeError(f"task {task.id!r} states a width of {width!r}, which must hold its {len(ids)} "
                            "prompt ids")
        return cls(ids, record, width)


@dataclasses.dataclass(frozen=True)
class PolicyCollector:
    """Draws completions of `prompt_tasks` tasks from the trainer's own policy, scored on decoded
    text (`sessions.Collector`; `RolloutScheduler.sampled` builds the usual scheduler over it).

    A call draws every task's `samples` continuations in one generation
    (`TextGeneration`'s `n`, one prefill per prompt) from the snapshot's
    variables under its key, and scores each by `reward` on its decoded
    text. EOS counts as an action in the response mask, but the text passed
    to `reward` leaves it out. A completion that ran out of
    `max_new_tokens` is TRUNCATED, and the scheduler's `truncation` decides
    whether it trains. Each call keeps the raw likelihoods the cached model
    recorded at its sampled actions, the proximal policy's, and the
    sampling ones as its behavior likelihoods. The packed rows are the
    objective's `seq_len + 1` ids wide, and a draw longer than a row, its
    prompt and completion together, raises ValueError when packed.
    """

    objective: TokenScorer
    reward: Reward
    decode: Callable[[Sequence[int]], str] = lambda ids: " ".join(str(token) for token in ids)
    max_new_tokens: int = 32
    sampling: Sampling = dataclasses.field(default_factory=Sampling)

    def __post_init__(self) -> None:
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("a rollout generates at least one token")
        # The policy's own: the reward's text trims at the end-of-sequence ids generation stops at.
        object.__setattr__(self, "sampling", self.objective.sampled(self.sampling))

    @property
    def width(self) -> int:
        return self.objective.seq_len + 1

    def rows(self, tasks: int, samples: int) -> int:
        return tasks * samples

    def _prepared(self, tasks: Sequence[Task]) -> tuple[list[PromptTask], int, ModelInputs]:
        """The tasks' prompts, the width they pad to and the inputs the policy is given."""
        prompts = [PromptTask.of(task) for task in tasks]
        width = max(prompt.width for prompt in prompts)
        rows, fields = pad_token_rows([prompt.ids for prompt in prompts], pad_id=self.sampling.pad,
                                      width=width)
        return prompts, width, ModelInputs(jnp.asarray(rows), {name: jnp.asarray(value)
                                                               for name, value in fields.items()})

    def collect(self, snapshot: Snapshot, tasks: Sequence[Task], samples: int) -> list[list[Session]]:
        # Every process of the mesh validates its prompts before generation enters its collectives.
        mesh = mesh_of(snapshot.variables)
        prompts, width, inputs = (self._prepared(tasks) if mesh is None
                                  else agreed("rollout input preparation", lambda: self._prepared(tasks)))
        generated = self.objective.policy(snapshot.variables, self.sampling)(
            inputs, self.max_new_tokens, key=snapshot.key, n=samples).host()
        if mesh is not None and DataPartition.of(mesh).reader != 0:
            # A share's later reader drew its part of the program; the first reader scores the share.
            return []

        def grouped(values: np.ndarray) -> np.ndarray:
            return values.reshape(len(prompts), samples, *values.shape[1:])

        sampled = grouped(generated.tokens[:, width:])
        counts, terminated = grouped(generated.lengths), grouped(generated.terminated)
        raw, behavior = grouped(generated.raw_log_probs), grouped(generated.behavior_log_probs)
        collected = []
        for row, (task, prompt) in enumerate(zip(tasks, prompts, strict=True)):
            sessions = []
            for sample in range(samples):
                count = int(counts[row, sample])
                tokens = tuple(int(token) for token in sampled[row, sample, :count])
                drawn, proximal = (tuple(float(value) for value in values[row, sample, :count])
                                   for values in (behavior, raw))
                finish = "stop" if terminated[row, sample] else "length"
                draw = Draw(prompt.ids, tokens, drawn, finish, snapshot.updates, proximal)
                text = self.decode(list(response(tokens, self.sampling)))
                sessions.append(scored_draw(task.id, sample, Call.of(draw), text, prompt.record, self.reward))
            collected.append(sessions)
        return collected

