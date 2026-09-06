"""Sampling rollouts for online RL.

A `SampledRollout` is a `dew.training.Rollout`: it takes the trainer's
prompt batch, samples `groups` completions per prompt with
`dew.sampling.generate`, scores each completion with `reward`, and packs the
fixed-shape batch the RL objectives read. Everything runs on the host outside
`jit` with the state's live parameters, including the old log-probabilities,
which come from the objective's own head over the concatenation. The
advantage family is a value: group for GRPO's leave-in mean, RLOO for the
leave-one-out mean, from `dew.rl`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from dew.data.prompts import INFO_KEY, LENGTH_KEY, PROMPT_KEY, SOURCE_KEY, TRUTH_KEY
from dew.rl import group_advantage, rloo_advantage

from ..lm import LMObjective

Reward: TypeAlias = Callable[[str, str, str, str], float]
"""A reward call: `reward(data_source, completion, ground_truth, extra_info)`
is the score of one completion, a plain float the rollout writes into the
`rewards` column."""

IDS_KEY = "input_ids"
"""Batch key holding the `[N, prompt + response]` concatenation, prompts
left-padded and groups contiguous inside each prompt row."""

RESPONSE_MASK_KEY = "response_mask"
"""Batch key holding the `[N, response]` completion marks."""

OLD_LOG_PROBS_KEY = "old_log_probs"
"""Batch key holding the `[N, response]` log-probabilities under the sampling
policy."""

ADVANTAGES_KEY = "advantages"
"""Batch key holding the `[N, response]` advantages broadcast over the width."""

REWARDS_KEY = "rewards"
"""Batch key holding the `[N]` scalar reward of each completion."""


def _texts(rows: np.ndarray) -> list[str]:
    """Fixed-width UTF-8 byte rows back to strings. The rows are int32, so
    each value becomes one byte; reading the raw buffer would pad every
    byte with three zeros."""
    return [bytes(row[row != 0].astype(np.uint8)).decode("utf-8")
            for row in np.asarray(rows, np.int32)]


@dataclasses.dataclass(frozen=True)
class SampledRollout:
    """G completions per prompt, scored and advantaged.

    `objective` is the RL objective training the run; its head rescores the
    concatenation for `old_log_probs`, so the trainer holds one objective for
    rollout and loss, with `seq_len` one below the prompt width plus
    `max_new_tokens`. `decode` renders ids for the reward call; the default
    joins the raw ids, which suits rewards that read ids. `eos_id` stops the
    response mask after the first stop token; None runs every row to
    `max_new_tokens`. Rows keep the prompt order, groups contiguous inside a
    row, and every leaf holds full-bleed rectangles: prompts at their
    left-padded width, responses at `max_new_tokens`, advantages broadcast
    over the response width.
    """

    objective: LMObjective
    reward: Reward
    decode: Callable[[Sequence[int]], str] = lambda ids: " ".join(
        str(token) for token in ids)
    groups: int = 4
    max_new_tokens: int = 32
    sample: str = "group"
    eos_id: int | None = None
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.groups < 2:
            raise ValueError(
                f"groups is {self.groups}: an advantage needs at least two completions")
        if self.max_new_tokens < 1:
            raise ValueError(
                f"max_new_tokens is {self.max_new_tokens}: a rollout generates "
                "at least one token")
        if self.sample not in ("group", "rloo"):
            raise ValueError(
                f"sample is {self.sample!r}: the advantage families are 'group' and 'rloo'")

    def __call__(self, state, batch, key: jax.Array) -> dict[str, np.ndarray]:
        from dew.sampling import generate

        prompts = np.asarray(batch[PROMPT_KEY], np.int32)
        prompt_length = np.asarray(batch[LENGTH_KEY], np.int32).reshape(-1)
        sources = _texts(batch[SOURCE_KEY])
        truths = _texts(batch[TRUTH_KEY])
        infos = _texts(batch[INFO_KEY])
        rows, width = prompts.shape
        if width + self.max_new_tokens != self.objective.seq_len + 1:
            raise ValueError(
                f"the concatenation is {width + self.max_new_tokens} ids wide for a "
                f"seq_len {self.objective.seq_len} objective; size the objective "
                "one below the prompt width plus max_new_tokens")

        completions = [
            np.asarray(generate(
                self.objective.model, state.params, prompts, self.max_new_tokens,
                key=jax.random.fold_in(key, group), temperature=self.temperature))
            [:, width:width + self.max_new_tokens]
            for group in range(self.groups)]
        sampled = np.stack(completions, axis=1)
        rewards = np.asarray(
            [[self.reward(sources[row],
                          self.decode([int(token) for token in sampled[row, group]]),
                          truths[row], infos[row])
              for group in range(self.groups)] for row in range(rows)], np.float32)
        flat = jnp.asarray(rewards.reshape(-1))
        raw = (group_advantage(flat, self.groups) if self.sample == "group"
               else rloo_advantage(flat, self.groups))
        advantages = np.asarray(raw, np.float32)

        if self.eos_id is None:
            mask = np.ones((rows, self.groups, self.max_new_tokens), np.float32)
        else:
            # A response counts through its first stop token. The shifted
            # cumulative sum is zero exactly there, one or more after.
            stopped = (sampled == self.eos_id)
            mask = ((np.cumsum(stopped, axis=-1) - stopped) == 0).astype(np.float32)

        full = np.concatenate(
            [np.broadcast_to(prompts[:, None, :], (rows, self.groups, width)),
             sampled], axis=-1)
        # Position p of the concatenation predicts the token at p + 1, so the
        # response starts at the prompt width minus one.
        old = np.asarray(self.objective.per_token_log_probs(
            state.params, full.reshape(-1, full.shape[-1])),
            np.float32).reshape(rows, self.groups, -1)[:, :, width - 1:width - 1 + self.max_new_tokens]
        return {
            IDS_KEY: full.reshape(-1, full.shape[-1]),
            RESPONSE_MASK_KEY: mask.reshape(-1, self.max_new_tokens),
            OLD_LOG_PROBS_KEY: old.reshape(-1, self.max_new_tokens),
            ADVANTAGES_KEY: np.broadcast_to(
                advantages.reshape(rows, self.groups)[..., None],
                (rows, self.groups, self.max_new_tokens)).reshape(-1, self.max_new_tokens),
            REWARDS_KEY: rewards.reshape(-1),
            LENGTH_KEY: np.broadcast_to(
                prompt_length[:, None], (rows, self.groups)).reshape(-1),
        }
