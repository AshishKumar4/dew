"""Host-side grouped completions, packed as one-call sessions with their sampling likelihoods."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from dew.artifacts import agree_process_phase
from dew.data.prompts import INFO_KEY, LENGTH_KEY, PROMPT_KEY, SOURCE_KEY, TRUTH_KEY
from dew.nn.inputs import ModelInputs, local_rows, mesh_of
from dew.sampling.text import Sampling

from ..lm import LMObjective
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

type Reward = Callable[[str, str, str, str], float]
"""Score ``(data_source, completion, ground_truth, extra_info)``."""

def _texts(rows: np.ndarray) -> list[str]:
    """Decode fixed-width UTF-8 byte rows stored as int32."""
    return [bytes(row[row != 0].astype(np.uint8)).decode("utf-8")
            for row in np.asarray(rows, np.int32)]


def prompt_rows(batch) -> tuple[np.ndarray, np.ndarray, list[str], list[str], list[str]]:
    """Read one host prompt batch: left-padded ids, lengths and the three reward strings.

    Every row needs a length between one and the width.
    """
    prompts = local_rows(batch[PROMPT_KEY])
    prompt_lengths = local_rows(batch[LENGTH_KEY])
    sources, truths, infos = (_texts(local_rows(batch[name])) for name in (SOURCE_KEY, TRUTH_KEY, INFO_KEY))
    rows, width = prompts.shape
    if (prompt_lengths.shape != (rows,) or not np.issubdtype(prompt_lengths.dtype, np.integer)
            or np.any(prompt_lengths < 1) or np.any(prompt_lengths > width)):
        raise ValueError("prompt_length must contain one valid integer length per row")
    return prompts, prompt_lengths, sources, truths, infos


def completion_rows(prompts: np.ndarray, prompt_lengths: np.ndarray, sampled: np.ndarray,
                    lengths: np.ndarray, terminated: np.ndarray, behavior: np.ndarray,
                    rewards: np.ndarray, versions: np.ndarray, estimator: str,
                    truncation: str) -> dict[str, np.ndarray]:
    """Pack `[rows, groups, ...]` completions as one-call sessions through `pack`.

    `prompts` is `[rows, width]` left-padded ids with `prompt_lengths` real
    tokens each; `sampled` and `behavior` are `[rows, groups, R]` with
    `lengths` valid actions per draw, `terminated` whether each stopped at
    EOS; `rewards` and `versions` are `[rows, groups]`. Each prompt's group
    is advantaged by the `estimator` family. A draw that did not stop at EOS
    is TRUNCATED, as `PromptSource` records it, and `truncation` decides
    whether it trains. The batch is `rows * groups` rows
    of `width + R` ids, the packed layout every GRPO batch has; session
    `row * groups + group` is that draw, which is what `sampled_values`
    hands its callback.
    """
    rows, groups, budget = sampled.shape
    width = prompts.shape[1]
    sessions = []
    for row in range(rows):
        prompt = tuple(int(token) for token in prompts[row, width - int(prompt_lengths[row]):])
        for group in range(groups):
            count = int(lengths[row, group])
            call = Call(prompt, tuple(int(token) for token in sampled[row, group, :count]),
                        tuple(float(value) for value in behavior[row, group, :count]),
                        "stop" if bool(terminated[row, group]) else "length", int(versions[row, group]))
            status = Status.COMPLETED if bool(terminated[row, group]) else Status.TRUNCATED
            sessions.append(Session(str(row), "", group, 0, (call,), status, float(rewards[row, group])))
    return pack(sessions, width + budget, rows=rows * groups, estimator=estimator, truncation=truncation)


@dataclasses.dataclass(frozen=True)
class SampledRollout:
    """Draw G completions per prompt and pack them as one-call sessions.

    EOS is a valid action in the response mask but excluded from reward text.
    The batch is `pack`'s layout, `prompts * groups` rows of the prompt
    width plus the response budget. Every completion is scored; one that
    ran out of `max_new_tokens` is TRUNCATED, and `truncation` (default
    `score`, train it on its reward) decides whether it trains.
    `old_log_probs` holds the raw likelihoods the cached model recorded at
    each sampled action, and `behavior_log_probs` the sampling ones.
    """

    objective: LMObjective
    reward: Reward
    decode: Callable[[Sequence[int]], str] = lambda ids: " ".join(str(token) for token in ids)
    groups: int = 4
    max_new_tokens: int = 32
    estimator: str = "group"
    truncation: str = "score"
    sampling: Sampling = Sampling()

    def __post_init__(self) -> None:
        if type(self.groups) is not int or self.groups < 2:
            raise ValueError(f"groups is {self.groups}: an advantage needs at least two completions")
        if type(self.max_new_tokens) is not int or self.max_new_tokens < 1:
            raise ValueError("a rollout generates at least one token")
        check_estimator(self.estimator)
        check_truncation(self.truncation)

    def _prepared(self, batch, key: jax.Array):
        """Validate one prompt batch and build the inputs generation reads.

        Returns the prompt ids, their lengths, the decoded source, truth
        and info strings the reward is called with, and the `ModelInputs`
        the policy is given.
        """
        key = jax.random.wrap_key_data(jax.random.key_data(key), impl=jax.random.key_impl(key))
        if key.shape != ():
            raise ValueError("key must be a single JAX PRNG key")
        prompts, prompt_lengths, sources, truths, infos = prompt_rows(batch)
        width = prompts.shape[1]
        if width + self.max_new_tokens != self.objective.seq_len + 1:
            raise ValueError("size the objective one below the prompt width plus max_new_tokens")
        # The lengths are already here on the host, so a batch of whole
        # prompts states its validity by carrying none.
        padded = bool(np.any(prompt_lengths < width))
        inputs = ModelInputs(jnp.asarray(prompts), {
            "attention_mask": jnp.arange(width)[None, :] >= width - jnp.asarray(prompt_lengths)[:, None]
        } if padded else {})
        return prompts, prompt_lengths, sources, truths, infos, inputs

    def __call__(self, state, batch, key: jax.Array) -> dict[str, np.ndarray]:
        """Draw `groups` completions per prompt and pack them as GRPO rows.

        Every group is drawn from one policy snapshot, scored by `reward`
        and centred within its prompt's group. The returned columns are
        the batch a GRPO loss reads.
        """
        # Validation completes on every rank before generation enters collectives.
        prepared = None
        error = None
        try:
            prepared = self._prepared(batch, key)
        except BaseException as failure:
            error = failure
        if mesh_of(state.params) is not None:
            agree_process_phase(error, phase="rollout input preparation")
        elif error is not None:
            raise error
        assert prepared is not None
        prompts, prompt_lengths, sources, truths, infos, inputs = prepared
        rows, width = prompts.shape
        policy = self.objective.policy(state.params, self.sampling)
        generated = [policy(inputs, self.max_new_tokens, key=jax.random.fold_in(key, group)).host()
                     for group in range(self.groups)]
        sampled = np.stack([generation.tokens[:, width:] for generation in generated], axis=1)
        lengths = np.stack([generation.lengths for generation in generated], axis=1)
        terminated = np.stack([generation.terminated for generation in generated], axis=1)
        raw = np.stack([generation.raw_log_probs for generation in generated], axis=1)
        behavior = np.stack([generation.behavior_log_probs for generation in generated], axis=1)
        rewards = np.asarray([
            [self.reward(sources[row], self.decode(sampled[row, group,
                         :int(lengths[row, group]) - int(terminated[row, group])].tolist()),
                         truths[row], infos[row]) for group in range(self.groups)]
            for row in range(rows)], np.float32)
        versions = np.full((rows, self.groups), int(state.updates), np.int32)
        packed = completion_rows(prompts, prompt_lengths, sampled, lengths, terminated, behavior,
                                    rewards, versions, self.estimator, self.truncation)
        packed[OLD_LOG_PROBS_KEY] = sampled_values(
            packed, lambda index, _: raw[index // self.groups, index % self.groups,
                                         :int(lengths[index // self.groups, index % self.groups])].tolist())
        return packed
