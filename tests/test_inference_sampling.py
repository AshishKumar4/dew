"""Shared sampler filters and source-policy tests reused from inference repair 984d673."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from transformers.generation.logits_process import (
    MinPLogitsWarper, TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper,
)

from dew.inference import TextGeneration
from dew.sampling import Sampling
from dew.sampling.text import _sample_token
from test_text_rollout_contract import decoder


@pytest.fixture
def task():
    model = decoder()
    variables = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    return TextGeneration(model, variables)


def test_single_prompt_processor_does_not_require_an_unneeded_pad_token(task, tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from dew.interop.pretrained import Processor

    backend = Tokenizer(models.WordLevel({"<unk>": 0, "one": 1, "two": 2}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>")
    assert tokenizer.pad_token_id is None
    tokenizer.save_pretrained(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    policy = replace(task, processor=Processor(tokenizer, {}, {}))
    generated = policy("one two", 2, key=jax.random.key(0))
    np.testing.assert_array_equal(generated.tokens[:, :2], [[1, 2]])
    assert generated.lengths[0] == 2


def warped(logits, sampling):
    scores = torch.FloatTensor(np.asarray(logits).copy())
    inputs = torch.LongTensor(np.zeros((scores.shape[0], 1), dtype=np.int64))
    scores = TemperatureLogitsWarper(sampling.temperature)(inputs, scores)
    if sampling.top_k is not None:
        scores = TopKLogitsWarper(sampling.top_k)(inputs, scores)
    if sampling.top_p < 1:
        scores = TopPLogitsWarper(sampling.top_p)(inputs, scores)
    if sampling.min_p > 0:
        scores = MinPLogitsWarper(sampling.min_p)(inputs, scores)
    return scores.log_softmax(-1).numpy()


@pytest.mark.parametrize("sampling", [
    Sampling(temperature=0.7, top_p=0.6),
    Sampling(temperature=1.3, min_p=0.4),
    Sampling(temperature=0.8, top_k=9, top_p=0.8, min_p=0.35),
])
def test_cached_generation_records_the_transformers_filtered_distribution(task, sampling):
    """Transformers 5.16.1 filters in temperature/top-k/top-p/min-p order.
    The selected action's raw and filtered log probabilities agree to 3e-6.
    """
    inputs = jnp.array([[1, 2], [4, 5]])
    actual = task(inputs, 4, key=jax.random.key(1), sampling=sampling)
    for step in range(4):
        logits = task.model.apply(task.variables, actual.tokens[:, :2 + step])[:, -1]
        filtered = warped(logits, sampling)
        selected = np.asarray(actual.tokens[:, 2 + step])
        expected = filtered[np.arange(2), selected]
        assert np.all(np.isfinite(expected))
        np.testing.assert_allclose(actual.behavior_log_probs[:, step], expected, atol=3e-6, rtol=0)
        raw = torch.tensor(np.asarray(logits).copy()).log_softmax(-1).numpy()[np.arange(2), selected]
        np.testing.assert_allclose(actual.raw_log_probs[:, step], raw, atol=3e-6, rtol=0)
    repeated = task(inputs, 4, key=jax.random.key(1), sampling=sampling)
    np.testing.assert_array_equal(repeated.tokens, actual.tokens)


@pytest.mark.parametrize("sampling", [Sampling(top_p=0), Sampling(min_p=1), Sampling(top_k=1, top_p=0.1)])
def test_filters_keep_the_best_token_at_the_boundary(sampling):
    logits = jnp.array([[1.0, 3.0, -2.0]])
    token, behavior, raw = _sample_token(logits, jax.random.split(jax.random.key(3), 1), sampling)
    np.testing.assert_array_equal(token, [1])
    np.testing.assert_allclose(behavior, [0.0], atol=1e-6)
    np.testing.assert_allclose(raw, jax.nn.log_softmax(logits)[0, 1:2], atol=1e-6)


def test_invalid_probability_controls_are_refused():
    for name in ("top_p", "min_p"):
        for value in (-0.1, 1.1, float("nan"), True):
            with pytest.raises(ValueError):
                Sampling(top_p=value) if name == "top_p" else Sampling(min_p=value)


def test_source_policy_preserves_supported_filters_and_requires_an_override_for_others(task):
    from dew.interop.pretrained import Pretrained
    from pathlib import Path

    source = Pretrained(task.model, task.variables, None, {}, Path("."), {},
                        generation_config={"do_sample": True, "temperature": 0.7, "top_p": 0.4, "min_p": 0.1})
    policy = source.text_generation()
    assert policy.sampling.top_p == 0.4 and policy.sampling.min_p == 0.1
    altered = replace(source, generation_config={**source.generation_config, "repetition_penalty": 2.0})
    with pytest.raises(ValueError, match="repetition_penalty"):
        altered.text_generation()
    override = altered.text_generation(sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(override([[1, 2]], 2, key=jax.random.key(1)).behavior_log_probs, 0)
    inactive = replace(source, generation_config={"do_sample": False, "typical_p": 0.1})
    assert inactive.text_generation().sampling.temperature == 0
