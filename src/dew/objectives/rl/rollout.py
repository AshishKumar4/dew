"""Host-side grouped rollouts with sampling-policy likelihoods."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Sequence
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np

from dew.data.prompts import INFO_KEY, LENGTH_KEY, PROMPT_KEY, SOURCE_KEY, TRUTH_KEY
from dew.rl import group_advantage, rloo_advantage
from dew.sampling.text import Sampling, generate

from ..lm import LMObjective

Reward: TypeAlias = Callable[[str, str, str, str], float]
"""Score ``(data_source, completion, ground_truth, extra_info)``."""

IDS_KEY = "input_ids"
RESPONSE_MASK_KEY = "response_mask"
RESPONSE_LENGTH_KEY = "response_length"
TERMINATED_KEY = "terminated"
OLD_LOG_PROBS_KEY = "old_log_probs"
"""Raw model-policy log-probabilities recorded before the training update.

GRPO's PPO ratio compares current raw policy to this old raw policy. These
values do not describe the behavior distribution after temperature/top-k.
"""
BEHAVIOR_LOG_PROBS_KEY = "behavior_log_probs"
"""Actual sampling log-probabilities, including temperature/top-k and greedy selection."""
ADVANTAGES_KEY = "advantages"
REWARDS_KEY = "rewards"


def _texts(rows: np.ndarray) -> list[str]:
    """Decode fixed-width UTF-8 byte rows stored as int32."""
    return [bytes(row[row != 0].astype(np.uint8)).decode("utf-8")
            for row in np.asarray(rows, np.int32)]


@dataclasses.dataclass(frozen=True)
class SampledRollout:
    """G completions per prompt, in prompt-major group order.

    EOS is a valid action in the response mask but excluded from reward text.
    Output rectangles preserve the input prompt width and configured response
    budget. Likelihoods come from the cached model at each sampled action.
    """

    objective: LMObjective
    reward: Reward
    decode: Callable[[Sequence[int]], str] = lambda ids: " ".join(str(token) for token in ids)
    groups: int = 4
    max_new_tokens: int = 32
    sample: str = "group"
    sampling: Sampling = Sampling()

    def __post_init__(self) -> None:
        if self.groups < 2:
            raise ValueError(f"groups is {self.groups}: an advantage needs at least two completions")
        if self.max_new_tokens < 1:
            raise ValueError("a rollout generates at least one token")
        if self.sample not in ("group", "rloo"):
            raise ValueError("the advantage families are 'group' and 'rloo'")

    def __call__(self, state, batch, key: jax.Array) -> dict[str, np.ndarray]:
        prompts = np.asarray(batch[PROMPT_KEY])
        prompt_lengths = np.asarray(batch[LENGTH_KEY])
        sources, truths, infos = (_texts(batch[name]) for name in (SOURCE_KEY, TRUTH_KEY, INFO_KEY))
        rows, width = prompts.shape
        if width + self.max_new_tokens != self.objective.seq_len + 1:
            raise ValueError("size the objective one below the prompt width plus max_new_tokens")
        generated = [generate(
            self.objective.model, state.params, prompts, self.max_new_tokens,
            key=jax.random.fold_in(key, group), sampling=self.sampling,
            prompt_lengths=prompt_lengths) for group in range(self.groups)]
        sampled = np.stack([np.asarray(result.tokens)[:, width:] for result in generated], axis=1)
        lengths = np.stack([np.asarray(result.lengths) for result in generated], axis=1)
        terminated = np.stack([np.asarray(result.terminated) for result in generated], axis=1)
        raw = np.stack([np.asarray(result.raw_log_probs) for result in generated], axis=1)
        behavior = np.stack([np.asarray(result.behavior_log_probs) for result in generated], axis=1)
        rewards = np.asarray([
            [self.reward(sources[row], self.decode(sampled[row, group,
                         :int(lengths[row, group]) - int(terminated[row, group])].tolist()),
                         truths[row], infos[row]) for group in range(self.groups)]
            for row in range(rows)], np.float32)
        flat = jnp.asarray(rewards.reshape(-1))
        advantages = np.asarray(
            group_advantage(flat, self.groups) if self.sample == "group"
            else rloo_advantage(flat, self.groups), np.float32)
        mask = np.arange(self.max_new_tokens)[None, None, :] < lengths[..., None]
        full = np.concatenate([
            np.broadcast_to(prompts[:, None, :], (rows, self.groups, width)), sampled], axis=-1)
        repeated_lengths = np.repeat(prompt_lengths, self.groups)

        return {
            IDS_KEY: full.reshape(-1, full.shape[-1]),
            RESPONSE_MASK_KEY: mask.reshape(-1, self.max_new_tokens).astype(np.float32),
            RESPONSE_LENGTH_KEY: lengths.reshape(-1),
            TERMINATED_KEY: terminated.reshape(-1),
            OLD_LOG_PROBS_KEY: raw.reshape(-1, self.max_new_tokens),
            BEHAVIOR_LOG_PROBS_KEY: behavior.reshape(-1, self.max_new_tokens),
            ADVANTAGES_KEY: np.broadcast_to(advantages[:, None],
                                           (rows * self.groups, self.max_new_tokens)),
            REWARDS_KEY: rewards.reshape(-1),
            LENGTH_KEY: repeated_lengths,
        }
