"""Cached generation against transformers `generate` on the same checkpoints.

tools/numerics_reference.py runs `GenerationMixin.generate` (transformers
5.16.1, fp32, eager attention, with its cache) on each tiny fixture from
three left-padded prompts of 4, 7 and 5 tokens for 10 new tokens, once
greedy and once sampled at temperature 0.7, top-k 20, top-p 0.9. It stores
the tokens, the fp32 logits generate scored each step with, and the float64
logits of the same model teacher-forced over the same path.

Greedy: Dew's cached decode from the same padded prompts must emit the same
tokens. That is well posed only where fp32 rounding cannot swap the top two
logits, so each fixture's smallest float64 top-2 margin along the path is
asserted to exceed four times the largest fp32 logit error transformers'
own decode makes (Dew is held to twice that error, and two logits move).

Sampled: two random streams cannot be matched, so Dew is forced along
transformers' sampled path with a strategy that feeds the recorded tokens
through the same cached decode loop and records what `Sample` records: each
token's log probability under the raw model and under the filtered chain.
A log probability log_softmax(z)_i moves by at most 2 max |dz|, and the
filtered one, computed on z / T over the kept support, by at most
2 max |dz| / T; with Dew's logits held to twice transformers' fp32 error E
(tests/reference_error.py), the raw bound is 4 E and the filtered 4 E / T.
Both are measured against the float64 path, where E is transformers' own
largest decode error on that path.

gemma3-tiny and mixtral-tiny have a 4-token sliding window, so every row
decodes past it; deepseek-v3-tiny decodes through MLA's compressed cache and
a routed layer. gemma3-tiny's sampled path equals its greedy one (its logits
leave a single token inside top-p at every step), so its sampled check
reduces to the filtered chain returning log probability zero.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from jax import lax
from reference_error import FACTOR

from dew.interop import load_pretrained
from dew.nn.inputs import ModelInputs
from dew.sampling import Sampling, generate
from dew.sampling.strategies import Draws

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
FAMILIES = ("llama-tiny", "qwen3-tiny", "gemma3-tiny", "mixtral-tiny", "deepseek-v3-tiny")
NEW_TOKENS = 10
SAMPLED = Sampling(temperature=0.7, top_k=20, top_p=0.9)


@pytest.fixture(scope="module", params=FAMILIES)
def family(request):
    directory = FIXTURES / request.param
    pretrained = load_pretrained(str(directory), dtype="float32", attention_impl="reference")
    with np.load(directory / "generate.npz") as stored:
        fixture = {name: stored[name] for name in stored.files}
    return request.param, pretrained, fixture


def prompts(fixture) -> ModelInputs:
    return ModelInputs(jnp.asarray(fixture["prompt"], jnp.int32),
                       {"attention_mask": jnp.asarray(fixture["mask"], bool)})


def decode_error(fixture, path: str) -> float:
    return float(np.max(np.abs(fixture[f"{path}_logits"] - fixture[f"{path}_f64"])))


def selected(log_probs, tokens):
    return np.take_along_axis(log_probs, tokens[..., None], -1)[..., 0]


def float64_log_softmax(logits):
    shifted = logits - logits.max(-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))


def test_greedy_generation_emits_transformers_tokens(family):
    name, pretrained, fixture = family
    error = decode_error(fixture, "greedy")
    ranked = np.sort(fixture["greedy_f64"], -1)
    margin = float(np.min(ranked[..., -1] - ranked[..., -2]))
    assert margin > 2 * FACTOR * error, (name, margin, error)

    result = generate(pretrained.model, pretrained.variables, prompts(fixture), NEW_TOKENS,
                      key=jax.random.key(0), sampling=Sampling(temperature=0))

    width = fixture["prompt"].shape[1]
    np.testing.assert_array_equal(np.asarray(result.tokens)[:, width:], fixture["greedy_tokens"])
    raw = selected(float64_log_softmax(fixture["greedy_f64"]), fixture["greedy_tokens"])
    assert np.max(np.abs(np.asarray(result.raw_log_probs, np.float64) - raw)) <= 2 * FACTOR * error


@struct.dataclass
class Forced:
    """A strategy that emits recorded tokens instead of drawing them, through
    the same decode loop and with the same likelihood records as `Sample`."""

    tokens: jax.Array

    def __call__(self, state, start, ops, transform, stopping, budget, n):
        def step(carry, token):
            state, step_state = carry
            raw = state.logits.astype(jnp.float32)
            scores = transform(step_state, raw)
            behavior = jnp.take_along_axis(jax.nn.log_softmax(scores), token[:, None], -1)[:, 0]
            likelihood = jnp.take_along_axis(jax.nn.log_softmax(raw), token[:, None], -1)[:, 0]
            active = step_state.active
            following = ops.advance(state, token, active)
            return (following, step_state.commit(token, active)), (token, active, behavior, likelihood)

        _, columns = lax.scan(step, (state, start), jnp.swapaxes(self.tokens, 0, 1), length=budget)
        tokens, valid, behavior, raw = (jnp.swapaxes(value, 0, 1) for value in columns)
        return Draws(tokens, valid, behavior, raw, jnp.zeros(tokens.shape[0], bool))


def test_the_sampled_path_scores_as_transformers_scored_it(family):
    name, pretrained, fixture = family
    tokens = fixture["sampled_tokens"]
    error = decode_error(fixture, "sampled")

    result = generate(pretrained.model, pretrained.variables, prompts(fixture), NEW_TOKENS,
                      key=jax.random.key(0), sampling=SAMPLED,
                      strategy=Forced(jnp.asarray(tokens, jnp.int32)))

    width = fixture["prompt"].shape[1]
    np.testing.assert_array_equal(np.asarray(result.tokens)[:, width:], tokens)
    raw = selected(float64_log_softmax(fixture["sampled_f64"]), tokens)
    raw_error = np.max(np.abs(np.asarray(result.raw_log_probs, np.float64) - raw))
    assert raw_error <= 2 * FACTOR * error, (name, raw_error, error)
    behavior_error = np.max(np.abs(np.asarray(result.behavior_log_probs, np.float64)
                                   - fixture["sampled_behavior_f64"]))
    assert behavior_error <= 2 * FACTOR * error / SAMPLED.temperature, (name, behavior_error, error)
