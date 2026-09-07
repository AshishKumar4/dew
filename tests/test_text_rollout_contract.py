"""Cached actions, padding and policy likelihoods on real small decoders."""

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.data import Dataset
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import ModelInputs
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.nn.mla import MLAMixer
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.objectives.rl import GRPOObjective, SampledRollout
from dew.rl import clipped_surrogate
from dew.sampling import Sampling, generate
from dew.training import Trainer


def decoder(kind="attention"):
    mixer = None
    if kind == "mla":
        mixer = MLAMixer(q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4,
                         qk_rope_head_dim=4, v_head_dim=8)
    elif kind == "recurrent":
        mixer = GatedDeltaNetMixer(linear_num_key_heads=2, linear_num_value_heads=2,
                                  linear_key_head_dim=8, linear_value_head_dim=8)
    return CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2,
                             head_dim=8, mlp_features=32, max_seq_len=12,
                             dtype="float32", mixer=mixer)


def prompts():
    return {"prompt": np.array([[0, 0, 1, 2], [3, 4, 5, 6]], np.int32),
            "prompt_length": np.array([2, 4], np.int32),
            "data_source": np.array([[97], [98]], np.int32),
            "ground_truth": np.array([[49], [50]], np.int32),
            "extra_info": np.zeros((2, 1), np.int32)}


def model_inputs(batch):
    tokens = jnp.asarray(batch["prompt"])
    mask = jnp.arange(tokens.shape[1])[None, :] >= tokens.shape[1] - jnp.asarray(batch["prompt_length"])[:, None]
    return ModelInputs(tokens, {"attention_mask": mask})


@pytest.mark.parametrize("kind", ["attention", "mla", "recurrent"])
def test_padding_and_cached_likelihoods_match_unpadded_full_forwards(kind):
    model = decoder(kind)
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    batch = prompts()
    sampling = Sampling(temperature=0.7, top_k=5)
    result = generate(model, params, model_inputs(batch), 3, key=jax.random.key(1), sampling=sampling)
    np.testing.assert_array_equal(result.lengths, [3, 3])
    np.testing.assert_array_equal(result.terminated, [False, False])
    for row, length in enumerate(batch["prompt_length"]):
        context = jnp.asarray(batch["prompt"][row:row + 1, 4 - length:])
        for index, action in enumerate(np.asarray(result.tokens)[row, 4:]):
            logits = model.apply(params, context)[0, -1].astype(jnp.float32)
            raw = jax.nn.log_softmax(logits)[action]
            scores = logits / sampling.temperature
            cutoff = jax.lax.top_k(scores, 5)[0][-1]
            behavior = jax.nn.log_softmax(jnp.where(scores >= cutoff, scores, -jnp.inf))[action]
            np.testing.assert_allclose(result.raw_log_probs[row, index], raw, atol=2e-6, rtol=2e-6)
            np.testing.assert_allclose(result.behavior_log_probs[row, index], behavior, atol=2e-6, rtol=2e-6)
            context = jnp.concatenate([context, jnp.array([[action]], jnp.int32)], axis=1)
    objective = GRPOObjective(model, seq_len=6)
    rescored = objective.per_token_log_probs(params, result.tokens,
                                              left_padding=jnp.array([2, 0]))[:, 3:]
    np.testing.assert_allclose(rescored, result.raw_log_probs, atol=2e-6, rtol=2e-6)
    assert np.max(np.abs(np.asarray(result.raw_log_probs - result.behavior_log_probs))) > 0.1


def test_padding_repro_greedy_and_seeded_bucket_independence():
    model = decoder()
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    key = jax.random.key(1)
    for sampling in (Sampling(temperature=0), Sampling(temperature=0.8, top_k=4)):
        inputs = ModelInputs(jnp.array([[0, 0, 1, 2]]), {"attention_mask": jnp.array([[0, 0, 1, 1]], bool)})
        padded = generate(model, params, inputs, 3, key=key, sampling=sampling)
        plain = generate(model, params, [[1, 2]], 3, key=key, sampling=sampling)
        np.testing.assert_array_equal(padded.tokens[:, 4:], plain.tokens[:, 2:])
        np.testing.assert_allclose(padded.behavior_log_probs, plain.behavior_log_probs, atol=1e-6)
        repeated = generate(model, params, inputs, 3, key=key, sampling=sampling)
        for first, second in zip(jax.tree.leaves(padded), jax.tree.leaves(repeated)):
            np.testing.assert_array_equal(first, second)


def test_eos_counts_as_action_and_reward_excludes_eos_and_padding():
    model = decoder()
    objective = GRPOObjective(model, seq_len=7)
    params = objective.init(jax.random.key(0))
    batch = prompts()
    plain = generate(model, params, model_inputs(batch), 4, key=jax.random.key(1),
                     sampling=Sampling(temperature=0))
    eos = int(plain.tokens[0, 4])
    seen = []

    def reward(source, completion, truth, info):
        seen.append((source, completion, truth, info))
        return float(len(completion))

    rollout = SampledRollout(objective, reward, groups=2, max_new_tokens=4,
                             sampling=Sampling(temperature=0, eos_id=eos, pad_id=12))
    result = rollout(SimpleNamespace(params=params), batch, jax.random.key(1))
    np.testing.assert_array_equal(result["response_length"][:2], [1, 1])
    np.testing.assert_array_equal(result["terminated"][:2], [True, True])
    np.testing.assert_array_equal(result["response_mask"][:2], [[1, 0, 0, 0]] * 2)
    np.testing.assert_array_equal(result["input_ids"][:2, 4:], [[eos, 12, 12, 12]] * 2)
    np.testing.assert_array_equal(result["behavior_log_probs"], np.zeros((4, 4)))
    assert seen[:2] == [("a", "", "1", "")] * 2
    for row, (_, text, _, _) in enumerate(seen):
        count = int(result["response_length"][row]) - int(result["terminated"][row])
        assert text == " ".join(str(token) for token in result["input_ids"][row, 4:4 + count])
    assert np.all(result["old_log_probs"][:2, 1:] == 0)
    assert np.all(result["old_log_probs"][:2, 0] < 0)


@pytest.mark.parametrize("kind", ["attention", "recurrent"])
def test_prompt_perplexity_uses_unpadded_context_and_real_transitions(kind):
    model = decoder(kind)
    objective = GRPOObjective(model, seq_len=7)
    params = objective.init(jax.random.key(0))
    batch = prompts()
    scores = objective.evaluate(params, batch, Step(jnp.array(0), jax.random.key(1), None))
    np.testing.assert_array_equal(scores.weights, [[0, 0, 1], [1, 1, 1]])
    for row, length in enumerate(batch["prompt_length"]):
        plain = jnp.asarray(batch["prompt"][row:row + 1, 4 - length:])
        logits = model.apply(params, plain[:, :-1])
        expected = optax.softmax_cross_entropy_with_integer_labels(logits, plain[:, 1:])[0]
        np.testing.assert_allclose(np.asarray(scores.losses)[row, 4 - length:], expected,
                                   atol=2e-6, rtol=2e-6)


def test_real_trainer_update_matches_raw_policy_ratio_with_behavior_recorded():
    model = decoder()
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    objective = GRPOObjective(model, seq_len=7, pretrained=params)
    rollout = SampledRollout(
        objective, lambda source, text, truth, info: float(sum(map(int, text.split()))),
        groups=2, max_new_tokens=4, sampling=Sampling(temperature=1.2, top_k=7))
    key = jax.random.key(5)
    repeats = max(1, jax.device_count() // 2)
    batch = {name: np.tile(value, (repeats,) + (1,) * (value.ndim - 1))
             for name, value in prompts().items()}
    run_key = jax.random.split(key)[1]
    rollout_key = jax.random.fold_in(jax.random.fold_in(run_key, 0), 1)
    rolled = rollout(SimpleNamespace(params=params), batch, rollout_key)
    assert np.any(rolled["advantages"] != 0)
    assert np.max(np.abs(rolled["old_log_probs"] - rolled["behavior_log_probs"])) > 0.1
    def raw_policy_loss(p):
        policy = objective.per_token_log_probs(
            p, jnp.asarray(rolled["input_ids"]),
            left_padding=4 - jnp.asarray(rolled["prompt_length"]))[:, 3:]
        return clipped_surrogate(
            policy - jnp.asarray(rolled["old_log_probs"]),
            jnp.asarray(rolled["advantages"]), jnp.asarray(rolled["response_mask"]))[0]

    loss, grads = jax.value_and_grad(raw_policy_loss)(params)
    assert np.isfinite(loss)
    expected = jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)
    rows = len(batch["prompt"])
    data = Dataset(train=lambda: iter([batch]), val=None, records=rows, batch=rows)
    state = Trainer(objective, optax.sgd(0.01), key=key, rollout=rollout).fit(
        data, steps=1, log_every=1, checkpoint_every=None)
    assert int(state.step) == 1
    movement = 0.0
    for old, actual, reference in zip(jax.tree.leaves(params), jax.tree.leaves(state.params),
                                     jax.tree.leaves(expected)):
        np.testing.assert_allclose(actual, reference, atol=2e-6, rtol=2e-6)
        movement += float(jnp.sum(jnp.abs(actual - old)))
    assert movement > 0.001


def test_recurrent_causality_and_bidirectional_capability_refusal():
    tokens = jnp.array([[1, 2, 3, 4]], jnp.int32)
    changed = tokens.at[0, -1].set(8)
    recurrent = decoder("recurrent")
    params = recurrent.init(jax.random.key(0), tokens)
    np.testing.assert_array_equal(recurrent.apply(params, tokens)[:, 0],
                                  recurrent.apply(params, changed)[:, 0])
    bidirectional = decoder().clone(causal=False)
    weights = bidirectional.init(jax.random.key(0), tokens)
    difference = bidirectional.apply(weights, tokens)[:, 0] - bidirectional.apply(weights, changed)[:, 0]
    assert float(jnp.max(jnp.abs(difference))) > 1e-3
    with pytest.raises(ValueError, match="causal=True"):
        recurrent.clone(causal=False).init(jax.random.key(0), tokens)
    for objective in (LMObjective, GRPOObjective):
        with pytest.raises(ValueError, match="causal"):
            objective(bidirectional, seq_len=3)


def test_any_declared_eos_id_stops_the_generation():
    model = decoder()
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    prompt = jnp.array([[1, 2]])
    baseline = generate(model, params, prompt, 3, key=jax.random.key(1),
                        sampling=Sampling(temperature=0))
    first = int(baseline.tokens[0, 2])
    result = generate(model, params, prompt, 3, key=jax.random.key(1),
                      sampling=Sampling(temperature=0, eos_id=((first + 1) % 13, first), pad_id=12))
    np.testing.assert_array_equal(result.tokens[0, 2:], [first, 12, 12])
    np.testing.assert_array_equal(result.lengths, [1])
    np.testing.assert_array_equal(result.terminated, [True])
    np.testing.assert_allclose(result.raw_log_probs[0, 0], baseline.raw_log_probs[0, 0], atol=1e-6)


def test_sampling_and_prompt_validity_fail_before_execution():
    for kwargs in ({"temperature": -1}, {"temperature": float("nan")}, {"top_k": 0}):
        with pytest.raises(ValueError):
            Sampling(**kwargs)
    model = decoder()
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    for mask in ([[0, 0]], [[1, 2]], [[1]], [[0.5, 1]]):
        with pytest.raises(ValueError):
            generate(model, params, ModelInputs(jnp.array([[1, 2]]), {"attention_mask": jnp.asarray(mask)}),
                     2, key=jax.random.key(1))
