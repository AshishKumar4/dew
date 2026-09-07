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


def test_seed_is_the_key_and_equal_shapes_reuse_one_executable():
    """`seed=n` draws what `key=jax.random.key(n)` draws; a second request of
    the same shape and controls, and the same task over other weights,
    reuse the compiled decode rather than tracing it again."""
    from dew.sampling import text

    model = decoder("attention")
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    task = TextGeneration(model, params, sampling=SAMPLING)
    prompt = [[1, 2, 4], [5, 6, 7]]
    compiled = text._compiled(None)
    before = compiled._cache_size()
    seeded = task(prompt, 3, seed=11)
    keyed = task(prompt, 3, key=jax.random.key(11))
    assert_same_generation(seeded, keyed)
    assert compiled._cache_size() == before + 1
    task.bind(jax.tree.map(lambda leaf: leaf * 0.5, params))(prompt, 3, seed=11)
    task([[9, 8, 7], [1, 1, 1]], 3, seed=0)
    assert compiled._cache_size() == before + 1
    task(prompt, 2, seed=11)
    assert compiled._cache_size() == before + 2
    with pytest.raises(ValueError, match="exactly one of key and seed"):
        task(prompt, 3)


def test_text_decodes_lazily_through_the_bound_processor():
    """`Generation.text` is the processor's decoding of each row's valid
    continuation; without a processor there is nothing to decode with."""
    from dew.inference import RunProcessor

    class Digits:
        vocab_size = 13
        eos_id = 12

        def encode(self, text):
            return [int(character) for character in text]

        def decode(self, ids):
            return "".join(str(int(token)) for token in ids)

    model = decoder("attention")
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    bare = TextGeneration(model, params, sampling=Sampling(temperature=0))
    with pytest.raises(ValueError, match="no processor"):
        bare([[1, 2]], 3, seed=0).text
    task = TextGeneration(model, params, RunProcessor(Digits()), sampling=Sampling(temperature=0), max_new_tokens=3)
    result = task(["12", "5"], seed=0)
    rows = result.host()
    assert rows.tokens.shape == (2, 2 + 3) and rows.tokens[1, 0] == 0
    assert result.text == task.decode(result)
    assert result.text == tuple("".join(str(token) for token in row[2:2 + length])
                                for row, length in zip(rows.tokens, rows.lengths))


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
    assert result.text == plain.decode(plain(prompts, 7, seed=11), 5)
