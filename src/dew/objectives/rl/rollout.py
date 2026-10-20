"""Host-side grouped completions, packed as one-call sessions with their sampling likelihoods."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from dew.coordination import agreed
from dew.data.prompts import INFO_KEY, LENGTH_KEY, PROMPT_KEY, SOURCE_KEY, TRUTH_KEY, PromptRecord
from dew.nn.inputs import ModelInputs, local_rows, mesh_of
from dew.objectives.base import Shown
from dew.sampling.text import Sampling

from ..lm import TokenScorer
from .sessions import (
    OLD_LOG_PROBS_KEY,
    Call,
    Session,
    Status,
    check_estimator,
    check_truncation,
    pack,
    sampled_values,
)

type Reward = Callable[[str, PromptRecord], float]
"""A function that scores a completion against its prompt's `PromptRecord` and returns the reward.

verl's four-argument `compute_score` enters through `dew.objectives.rl.verl.verl_reward`."""

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


@dataclasses.dataclass(frozen=True)
class SampledRollout:
    """Draws `groups` completions per prompt from the objective's policy and packs them as one-call sessions.

    EOS counts as an action in the response mask, but the text passed to
    `reward` leaves it out. The batch has `pack`'s layout, `prompts * groups`
    rows of the objective's `seq_len + 1`; a draw longer than a row, its
    prompt and completion together, raises ValueError there. Every
    completion is scored. One that ran out of `max_new_tokens` is TRUNCATED,
    and `truncation` decides whether it trains; the default, `score`, trains
    it on its reward. `old_log_probs` holds the raw likelihoods the cached
    model recorded at each sampled action, and `behavior_log_probs` holds
    the sampling ones. `metrics` holds the latest call's mean reward, mean
    completion length and truncated share, which the trainer logs as
    `rollout/<name>`.
    """

    objective: TokenScorer
    reward: Reward
    decode: Callable[[Sequence[int]], str] = lambda ids: " ".join(str(token) for token in ids)
    groups: int = 4
    max_new_tokens: int = 32
    estimator: str = "group"
    truncation: str = "score"
    sampling: Sampling = dataclasses.field(default_factory=Sampling)
    metrics: dict[str, float] = dataclasses.field(default_factory=dict, init=False, compare=False)
    shown: ClassVar[Mapping[str, Shown]] = {"reward/mean": Shown(better="higher"),
                                             "status/truncated": Shown(better="lower", percent=True)}

    def __post_init__(self) -> None:
        if type(self.groups) is not int or self.groups < 2:
            raise ValueError(f"groups is {self.groups}: an advantage needs at least two completions")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("a rollout generates at least one token")
        check_estimator(self.estimator)
        check_truncation(self.truncation)

    def _prepared(self, batch, key: jax.Array):
        """Validate one prompt batch and build the inputs generation reads.

        Returns the prompt ids, their lengths, the `PromptRecord` each row's
        reward is called with, and the `ModelInputs` the policy is given.
        """
        prompts, prompt_lengths, records = prompt_rows(batch)
        width = prompts.shape[1]
        # The lengths are already here on the host, so a batch of whole
        # prompts states its validity by carrying none.
        padded = bool(np.any(prompt_lengths < width))
        inputs = ModelInputs(jnp.asarray(prompts), {
            "attention_mask": jnp.arange(width)[None, :] >= width - jnp.asarray(prompt_lengths)[:, None]
        } if padded else {})
        return prompts, prompt_lengths, records, inputs

    def __call__(self, state, batch, key: jax.Array) -> dict[str, np.ndarray]:
        """Draw `groups` completions per prompt and pack them as GRPO rows.

        A prompt's group is the policy's `groups` continuations of one
        request under `key` (`TextGeneration`'s `n`), drawn from one policy
        snapshot over one prefill and scored by `reward`, and the advantages
        are computed within each prompt's group. The returned columns are the
        batch a GRPO loss reads.
        """
        # Validation completes on every rank before generation enters collectives.
        prepared = (agreed("rollout input preparation", lambda: self._prepared(batch, key))
                    if mesh_of(state.variables) is not None else self._prepared(batch, key))
        prompts, prompt_lengths, records, inputs = prepared
        rows, width = prompts.shape
        generated = self.objective.policy(state.variables, self.sampling)(
            inputs, self.max_new_tokens, key=key, n=self.groups).host()

        def grouped(values: np.ndarray) -> np.ndarray:
            return values.reshape(rows, self.groups, *values.shape[1:])

        sampled = grouped(generated.tokens[:, width:])
        lengths, terminated = grouped(generated.lengths), grouped(generated.terminated)
        raw, behavior = grouped(generated.raw_log_probs), grouped(generated.behavior_log_probs)
        def completion(row: int, group: int) -> str:
            end = int(lengths[row, group]) - int(terminated[row, group])
            return self.decode(sampled[row, group, :end].tolist())

        rewards = np.asarray([[self.reward(completion(row, group), records[row])
                               for group in range(self.groups)] for row in range(rows)], np.float32)
        sessions, version = [], int(state.updates)
        for row in range(rows):
            prompt = tuple(int(token) for token in prompts[row, width - int(prompt_lengths[row]):])
            for group in range(self.groups):
                count, stopped = int(lengths[row, group]), bool(terminated[row, group])
                call = Call(prompt, tuple(int(token) for token in sampled[row, group, :count]),
                            tuple(float(value) for value in behavior[row, group, :count]),
                            "stop" if stopped else "length", version)
                sessions.append(Session(str(row), "", group, 0, (call,),
                                        Status.COMPLETED if stopped else Status.TRUNCATED,
                                        float(rewards[row, group])))
        # Session `row * groups + group` is that draw, which is what `sampled_values` hands its callback.
        # The rows are the objective's, which `pack` refuses a draw too long for.
        packed = pack(sessions, self.objective.seq_len + 1, rows=rows * self.groups,
                      estimator=self.estimator, truncation=self.truncation)
        packed[OLD_LOG_PROBS_KEY] = sampled_values(
            packed, lambda index, _: raw[index // self.groups, index % self.groups,
                                         :int(lengths[index // self.groups, index % self.groups])].tolist())
        self.metrics.update({"reward/mean": float(rewards.mean()), "length/mean": float(lengths.mean()),
                             "status/truncated": float(1 - terminated.mean())})
        return packed
