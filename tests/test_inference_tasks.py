"""Generation tasks bind a model, its weights and its host processing."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from dew.inference import BlockGeneration, TextGeneration
from dew.interop import load_pretrained
from dew.nn.inputs import ModelInputs
from dew.sampling import Sampling, generate
from test_text_rollout_contract import decoder

FIXTURES = Path(__file__).parent / "fixtures" / "hf"
SAMPLING = Sampling(temperature=0.8, top_k=5, eos_id=(3, 9))


def assert_same_generation(actual, expected):
    np.testing.assert_array_equal(actual.tokens, expected.tokens)
    np.testing.assert_array_equal(actual.lengths, expected.lengths)
    np.testing.assert_array_equal(actual.terminated, expected.terminated)
    np.testing.assert_allclose(actual.behavior_log_probs, expected.behavior_log_probs, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(actual.raw_log_probs, expected.raw_log_probs, atol=2e-6, rtol=2e-6)


def test_a_bound_task_draws_from_the_weights_it_was_bound_to():
    """Rebinding after an optimizer step samples the updated policy while the
    earlier task keeps sampling the snapshot it holds."""
    model = decoder()
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    rows = [[1, 2, 3], [4, 5, 6]]
    task = TextGeneration(model, params, sampling=SAMPLING)
    before = task(rows, 5, key=jax.random.key(1))
    assert_same_generation(before, generate(model, params, jnp.asarray(rows), 5, key=jax.random.key(1),
                                            sampling=SAMPLING))
    updates, _ = optax.sgd(0.5).update(jax.tree.map(jnp.ones_like, params["params"]),
                                       optax.sgd(0.5).init(params["params"]))
    moved = {"params": optax.apply_updates(params["params"], updates)}
    later = task.bind(moved)
    assert_same_generation(later(rows, 5, key=jax.random.key(1)),
                           generate(model, moved, jnp.asarray(rows), 5, key=jax.random.key(1), sampling=SAMPLING))
    assert_same_generation(task(rows, 5, key=jax.random.key(1)), before)
    greedy = task(ModelInputs(jnp.asarray(rows)), 5, key=jax.random.key(2), sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(greedy.behavior_log_probs, 0)
    assert task.decode(greedy) == ()


def test_prepared_rows_and_text_requests_are_kept_apart():
    model = decoder()
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    task = TextGeneration(model, params)
    with pytest.raises(ValueError, match="processor"):
        task("hello", 3, key=jax.random.key(0))
    with pytest.raises(ValueError, match="images"):
        task([[1, 2]], 3, key=jax.random.key(0), images=[np.zeros((2, 2, 3))])
    with pytest.raises(ValueError, match="vocabulary"):
        task([[99, 2]], 3, key=jax.random.key(0))


def test_a_loaded_source_generates_from_text_with_its_own_policy():
    """The Gemma3 source turns prompts and images into a conditioned greedy
    continuation and decodes it back through its own tokenizer."""
    loaded = load_pretrained(FIXTURES / "gemma3-native-tiny", dtype="float32", attention_impl="reference")
    images = np.load(FIXTURES / "gemma3-native-tiny" / "raw_images.npy")
    prompts = json.loads((FIXTURES / "gemma3-native-tiny" / "prompts.json").read_text())
    task = loaded.text_generation()
    assert task.sampling.temperature == 0 and task.sampling.eos_id is not None
    generated = task(prompts, 3, key=jax.random.key(1), images=[[images[0]], [images[1], images[2]]])
    np.testing.assert_array_equal(generated.tokens[:, -3:],
                                  np.load(FIXTURES / "gemma3-native-tiny" / "continuation.npy"))
    assert loaded.processor is not None
    assert task.decode(generated) == tuple(loaded.processor.decode(generated.tokens[:, -3:]))
    with pytest.raises(TypeError):
        loaded.block_generation()


def test_a_diffusion_gemma_source_generates_canvases_without_likelihood_claims():
    loaded = load_pretrained(FIXTURES / "diffusion-gemma-workflow", dtype="float32", attention_impl="xla",
                             max_seq_len=32)
    task = loaded.block_generation()
    assert isinstance(task, BlockGeneration)
    prompts = ["<bos> t5 t7 t9 t11", "<bos> t6 t8 t10 t12"]
    result = task(prompts, 7, key=jax.random.key(11))
    with np.load(FIXTURES / "diffusion-gemma-workflow" / "reference.npz") as reference:
        np.testing.assert_array_equal(result.tokens, reference["tokens"][:, :12])

    rebound = task.bind(jax.tree.map(lambda leaf: leaf * 0.5, loaded.variables))
    assert not np.array_equal(rebound(prompts, 7, key=jax.random.key(11)).tokens, result.tokens)


def test_seed_is_the_key():
    model = decoder("attention")
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    task = TextGeneration(model, params, sampling=SAMPLING)
    prompt = [[1, 2, 4], [5, 6, 7]]
    assert_same_generation(task(prompt, 3, seed=11), task(prompt, 3, key=jax.random.key(11)))
    with pytest.raises(ValueError, match="exactly one of key and seed"):
        task(prompt, 3)


def test_equal_shape_calls_and_rebinding_do_not_request_compilation(tmp_path):
    from jax import monitoring

    model = decoder("attention")
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    task = TextGeneration(model, params, sampling=SAMPLING)
    rebound = task.bind(jax.tree.map(lambda leaf: leaf * 0.5, params))
    prompt = [[1, 2, 4], [5, 6, 7]]
    events = []
    def record(event, **metadata):
        if event == "/jax/compilation_cache/compile_requests_use_cache":
            events.append(event)
    previous = jax.config.jax_compilation_cache_dir
    jax.config.update("jax_compilation_cache_dir", str(tmp_path))
    monitoring.register_event_listener(record)
    try:
        task(prompt, 3, seed=11).host()
        warmed = len(events)
        rebound(prompt, 3, seed=11).host()
        task([[9, 8, 7], [1, 1, 1]], 3, seed=0).host()
        assert len(events) == warmed
        task(prompt, 6, seed=11).host()
        assert len(events) > warmed
    finally:
        monitoring.unregister_event_listener(record)
        jax.config.update("jax_compilation_cache_dir", previous)

def test_text_decodes_lazily_through_the_bound_processor():
    """`Generation.text` is the processor's decoding of each row's valid
    continuation; without a processor there is nothing to decode with."""
    from dew.inference import RunProcessor

    calls = []

    class Digits:
        vocab_size = 13
        eos_id = 12

        def encode(self, text):
            return [int(character) for character in text]

        def decode(self, ids):
            calls.append(tuple(ids))
            return "".join(str(int(token)) for token in ids)

    model = decoder("attention")
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    bare = TextGeneration(model, params, sampling=Sampling(temperature=0))
    with pytest.raises(ValueError, match="no processor"):
        bare([[1, 2]], 3, seed=0).text
    task = TextGeneration(model, params, RunProcessor(Digits()), sampling=Sampling(temperature=0), max_new_tokens=3)
    result = task(["12", "5"], seed=0)
    assert calls == []
    first = result.text
    assert len(calls) == 2
    assert result.text is first and len(calls) == 2
    rows = result.host()
    assert rows.tokens.shape == (2, 2 + 3) and rows.tokens[1, 0] == 0
    assert result.text == task.decode(result)
    assert result.text == tuple("".join(str(token) for token in row[2:2 + length])
                                for row, length in zip(rows.tokens, rows.lengths))


def test_a_prompt_batch_carries_validity_only_where_it_padded():
    """A host that padded nothing says so by omitting validity. The model
    cannot read the contents of an all-true mask, so it would build one and
    leave its fused attention kernel; ragged prompts still carry theirs, and
    the values say which slots the padding took."""
    from dew.inference import RunProcessor

    class Digits:
        def encode(self, text):
            return [int(character) for character in text]

        def decode(self, ids):
            return "".join(str(int(token)) for token in ids)

    processor = RunProcessor(Digits())

    assert "attention_mask" not in processor(["12", "34"]).token_fields
    assert "attention_mask" not in processor("789").token_fields
    ragged = processor(["12", "5"])
    np.testing.assert_array_equal(ragged.token_fields["attention_mask"],
                                  [[True, True], [False, True]])


@pytest.mark.mesh
def test_a_placed_diffusion_gemma_task_keeps_its_rows_sharded_and_draws_the_same_canvases():
    """Placed under the trainer's layout, a canvas task shards its weights,
    splits rows over the mesh's batch axes and reads them back from `host()`;
    with one row per device there is no padding, so the batch-wide canvas
    draw matches the single-device one row for row."""
    from dew.inference.pipeline import place
    from dew.nn.inputs import BATCH_AXES
    from dew.training import Layout, MeshSpec

    loaded = load_pretrained(FIXTURES / "diffusion-gemma-workflow", dtype="float32", attention_impl="xla",
                             max_seq_len=32)
    plain = loaded.block_generation()
    placed = plain.bind(place(loaded.variables, MeshSpec(fsdp=2), Layout(min_shard=2 ** 6)))
    assert any("fsdp" in str(leaf.sharding.spec) for leaf in jax.tree.leaves(placed.variables))
    prompts = ["<bos> t5 t7 t9 t11", "<bos> t6 t8 t10 t12"] * (jax.device_count() // 2)
    result = placed(prompts, 7, seed=11)
    assert result.tokens.sharding.spec == jax.sharding.PartitionSpec(BATCH_AXES)
    assert result.rows == len(prompts) and result.prompt_width == 5
    rows = result.host()
    np.testing.assert_array_equal(rows.tokens, plain(prompts, 7, seed=11).host().tokens)
    assert result.text == plain.decode(plain(prompts, 7, seed=11))


@pytest.mark.parametrize("kind", ["dpo", "grpo", "ppo"])
def test_pipeline_publishes_the_updated_policy_not_the_frozen_reference(kind, tmp_path):
    from dew.data import Dataset
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.rl import DPOObjective, GRPOObjective, PPOObjective, ValueHead
    from dew.training import Trainer, Checkpoints
    from dew.config import ModelConfig
    from dataclasses import asdict
    import dew

    model = CausalTransformer(vocab_size=8, emb_features=16, num_layers=1, num_heads=2,
                              mlp_features=32, max_seq_len=8, dtype="float32", attention_impl="xla")
    if kind == "dpo":
        objective = DPOObjective(model, seq_len=2)
    elif kind == "grpo":
        objective = GRPOObjective(model, seq_len=2, beta=0.1)
    else:
        objective = PPOObjective(model, seq_len=2, critic=ValueHead(model.clone()), beta=0.1)
    trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(0))
    count = max(2, jax.device_count())
    if kind == "dpo":
        pairs = np.tile(np.asarray([[[1, 2, 3], [1, 2, 4]]], np.int32), (count, 1, 1))
        batch = {"input_ids": pairs, "completion_mask": np.ones_like(pairs, np.float32)}
    else:
        ids = np.tile(np.asarray([[1, 2, 3]], np.int32), (count, 1))
        before = trainer.initial_state()
        actor = objective.actor if kind == "ppo" else objective
        weights = ({collection: value["policy"] for collection, value in before.params.items()}
                   if kind == "ppo" else before.params)
        old = np.asarray(actor.per_token_log_probs(weights, ids))[:, -1:]
        batch = {"input_ids": ids, "old_log_probs": old, "response_mask": np.ones_like(old),
                 "advantages": np.ones_like(old)}
        if kind == "ppo":
            values = np.asarray(objective.values(before.params, batch))
            batch.update(old_values=values, returns=values + 1)
    data = Dataset(train=lambda: iter([batch, batch]), val=None, records=2 * count, batch=count)
    state = trainer.fit(data, steps=2, log_every=100, checkpoint_every=None)
    sampling = Sampling(temperature=0)
    def draw(weights):
        task = objective.policy(weights) if kind == "ppo" else objective.policy(weights, sampling)
        return task([[1, 2]], 1, key=jax.random.key(3), sampling=sampling).host()
    expected = draw(state.params)
    actual = objective.pipeline(state)([[1, 2]], 1, seed=3, sampling=sampling).host()
    reference = draw(state.averaged)
    np.testing.assert_array_equal(actual.tokens, expected.tokens)
    np.testing.assert_allclose(actual.raw_log_probs, expected.raw_log_probs, atol=1e-7, rtol=1e-7)
    assert not np.allclose(actual.raw_log_probs, reference.raw_log_probs, atol=1e-5, rtol=1e-5)
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(int(state.step), state, None)
    checkpoints.wait()
    config = ModelConfig("causal_transformer", dict(vocab_size=8, emb_features=16, num_layers=1,
        num_heads=2, mlp_features=32, max_seq_len=8), dtype="float32", attention_impl="xla")
    (tmp_path / "run.json").write_text(json.dumps({"objective": kind, "model": asdict(config),
        "tokenizer": "byte", "sample_tokens": 1, "sampling": asdict(sampling)}))
    restored = dew.pipeline(str(tmp_path))([[1, 2]], seed=3).host()
    np.testing.assert_array_equal(restored.tokens, expected.tokens)
    np.testing.assert_allclose(restored.raw_log_probs, expected.raw_log_probs, atol=1e-7, rtol=1e-7)
