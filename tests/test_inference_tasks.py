"""Generation tasks bind a model, its weights and its host processing."""

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from model_support import decoder

from dew.diffusion.block import BlockProcess
from dew.inference import BlockGeneration, TextGeneration
from dew.interop import Pretrained
from dew.nn.inputs import ModelInputs
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling, generate

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
    assert_same_generation(
        later(rows, 5, key=jax.random.key(1)),
        generate(model, moved, jnp.asarray(rows), 5, key=jax.random.key(1), sampling=SAMPLING),
    )
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
    loaded = Pretrained.load(FIXTURES / "gemma3-native-tiny", dtype="float32", attention_impl="reference")
    images = np.load(FIXTURES / "gemma3-native-tiny" / "raw_images.npy")
    prompts = json.loads((FIXTURES / "gemma3-native-tiny" / "prompts.json").read_text())
    task = loaded.text_generation()
    assert task.sampling.temperature == 0 and task.sampling.eos_id is not None
    generated = task(prompts, 3, key=jax.random.key(1), images=[[images[0]], [images[1], images[2]]])
    np.testing.assert_array_equal(generated.tokens[:, -3:],
                                  np.load(FIXTURES / "gemma3-native-tiny" / "continuation.npy"))
    assert loaded.processor is not None
    assert task.decode(generated) == tuple(loaded.processor.decode(generated.tokens[:, -3:]))
    assert not hasattr(loaded, "block_generation")


def test_a_source_loads_as_its_kind_and_a_kind_refuses_another():
    """`Pretrained.load` returns the kind the source is, with the methods
    that kind has; called on a kind, it refuses a source of another."""
    from dew.interop import PretrainedBlockDecoder, PretrainedDecoder, PretrainedMaskedDecoder

    assert type(Pretrained.load(FIXTURES / "llama-tiny", dtype="float32")) is PretrainedDecoder
    assert type(Pretrained.load(FIXTURES / "llada-tiny", dtype="float32")) is PretrainedMaskedDecoder
    gemma = PretrainedBlockDecoder.load(FIXTURES / "diffusion-gemma-workflow", dtype="float32",
                                        max_seq_len=32)
    assert type(gemma) is PretrainedBlockDecoder
    with pytest.raises(TypeError, match="is a PretrainedMaskedDecoder source, not a PretrainedDecoder"):
        PretrainedDecoder.load(FIXTURES / "llada-tiny", dtype="float32")


def test_a_loader_takes_a_dtype_and_records_its_name():
    """`jnp.float32` and "float32" load the same model, and the record the
    model was built from keeps the name, which is what run.json can hold."""
    from dew.interop import PretrainedDecoder

    typed = PretrainedDecoder.load(FIXTURES / "llama-tiny", dtype=jnp.float32, param_dtype=jnp.bfloat16)
    named = PretrainedDecoder.load(FIXTURES / "llama-tiny", dtype="float32", param_dtype="bfloat16")
    assert typed.model_config["dtype"] == "float32"
    assert typed.model_config == named.model_config
    for left, right in zip(jax.tree.leaves(typed.variables), jax.tree.leaves(named.variables), strict=True):
        assert left.dtype == right.dtype
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def test_a_pretrained_bundle_fine_tunes_identically_to_explicit_wiring():
    from dew import Dataset, Trainer
    from dew.objectives.base import Step
    from dew.objectives.lm import LMObjective

    source = Pretrained.load(FIXTURES / "llama-tiny", dtype="float32", attention_impl="xla",
                             max_seq_len=8)
    options = {"ema_decay": None, "head_chunks": 1, "pad_id": 0, "z_loss": 1e-4}
    explicit = LMObjective(source.model, 4, variables=source.variables, **options)
    bundled = LMObjective(source, 4, **options)
    key = jax.random.key(41)
    np.testing.assert_array_equal(jax.tree.leaves(bundled.init(key))[0],
                                  jax.tree.leaves(source.variables)[0])
    count = max(2, jax.device_count())
    batch = {"text": np.tile(np.asarray([[1, 3, 5, 7, 0], [2, 4, 6, 8, 9]], np.int32),
                              (count // 2, 1))}
    step = Step(step=jnp.asarray(0), key=jax.random.key(13), ema=None)
    def loss(objective):
        return jax.jit(objective.scalar_loss)(objective.init(key), batch, step)[0]
    np.testing.assert_array_equal(loss(bundled), loss(explicit))
    data = Dataset(train=lambda partition: iter([batch]), val=None, records=count, batch=count)
    states = [Trainer(objective, optax.adamw(1e-3), key=key).fit(
        data, steps=1, log_every=100, checkpoint_every=None) for objective in (explicit, bundled)]
    assert int(states[1].updates) == 1
    for actual, expected in zip(
        jax.tree.leaves(states[1].variables), jax.tree.leaves(states[0].variables), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    assert any(not np.array_equal(actual, initial) for actual, initial in
               zip(jax.tree.leaves(states[1].variables), jax.tree.leaves(source.variables), strict=True))


def test_a_keyword_overrides_what_a_bundle_supplies():
    """A bundle stands in for the model, its weights and its processor; a
    keyword given beside it wins."""
    source = Pretrained.load(FIXTURES / "llama-tiny", dtype="float32", attention_impl="xla")
    zeros = jax.tree.map(jnp.zeros_like, source.variables)
    objective = LMObjective(source, 4, variables=zeros)
    assert objective.processor is source.text_processor
    assert not any(bool(jnp.any(leaf)) for leaf in jax.tree.leaves(objective.init(jax.random.key(0))))


def test_an_explicit_none_clears_what_a_decoder_bundle_supplies():
    """`variables=None` beside a bundle starts from a fresh init and
    `processor=None` keeps no processor, as they do without one; only an
    omitted keyword takes the bundle's."""
    source = Pretrained.load(FIXTURES / "llama-tiny", dtype="float32", attention_impl="xla")
    cleared = LMObjective(source, 4, variables=None, processor=None)
    assert cleared.variables is None and cleared.processor is None
    drawn = cleared.init(jax.random.key(0))
    assert any(not np.array_equal(np.asarray(ours), np.asarray(theirs)) for ours, theirs in zip(
        jax.tree.leaves(drawn["params"]), jax.tree.leaves(source.variables["params"]), strict=True))
    kept = LMObjective(source, 4)
    assert kept.variables is source.variables and kept.processor is source.text_processor


def test_media_prompts_are_processed_once_and_keep_their_continuations():
    """Text and images reach the processor once per request, whatever the
    continuation count, and the continuations expand afterwards: each row
    carries the prompt, the image features and the conditioned continuation of
    the prompt it sits under."""
    loaded = Pretrained.load(FIXTURES / "gemma3-native-tiny", dtype="float32", attention_impl="reference")
    images = np.load(FIXTURES / "gemma3-native-tiny" / "raw_images.npy")
    prompts = json.loads((FIXTURES / "gemma3-native-tiny" / "prompts.json").read_text())
    expected = np.load(FIXTURES / "gemma3-native-tiny" / "continuation.npy")
    requests = []

    class Counted:
        """The source processor, with the requests it was handed."""

        def __init__(self, inner):
            self.inner = inner

        def __call__(self, text, *, images=None):
            requests.append((tuple(text), None if images is None else len(images)))
            return self.inner(text, images=images)

        def decode(self, tokens):
            return self.inner.decode(tokens)

    assert loaded.processor is not None
    task = replace(loaded.text_generation(), processor=Counted(loaded.processor), n=3)
    generated = task(prompts, 3, key=jax.random.key(1), images=[[images[0]], [images[1], images[2]]])

    assert requests == [(tuple(prompts), 2)]
    rows = generated.host()
    assert rows.tokens.shape[0] == 3 * len(prompts)
    np.testing.assert_array_equal(rows.tokens[:, -3:], np.repeat(expected, 3, axis=0))
    for prompt in range(len(prompts)):
        group = rows.tokens[prompt * 3:(prompt + 1) * 3, :-3]
        np.testing.assert_array_equal(group, np.repeat(group[:1], 3, axis=0))
    text = generated.text
    assert text == task.decode(generated) and len(text) == 3 * len(prompts)
    for prompt in range(len(prompts)):
        assert len(set(text[prompt * 3:(prompt + 1) * 3])) == 1


def test_a_diffusion_gemma_source_generates_canvases_without_likelihood_claims():
    loaded = Pretrained.load(FIXTURES / "diffusion-gemma-workflow", dtype="float32", attention_impl="xla",
                             max_seq_len=32)
    task = loaded.block_generation()
    assert isinstance(task, BlockGeneration)
    prompts = ["<bos> t5 t7 t9 t11", "<bos> t6 t8 t10 t12"]
    result = task(prompts, 7, key=jax.random.key(11))
    with np.load(FIXTURES / "diffusion-gemma-workflow" / "reference.npz") as reference:
        np.testing.assert_array_equal(result.tokens, reference["tokens"][:, :12])

    rebound = task.bind(jax.tree.map(lambda leaf: leaf * 0.5, loaded.variables))
    assert not np.array_equal(rebound(prompts, 7, key=jax.random.key(11)).tokens, result.tokens)


class Counting(nn.Module):
    """A user's own block denoiser, no Dew class: its prefix cache counts the
    clean tokens it encoded, and it scores each canvas slot as its own
    position past them, modulo the vocabulary, so a canvas commits the next
    positions in order."""

    vocab_size: int = 11
    canvas_length: int = 4
    max_seq_len: int = 16

    def init_cache(self, batch_size):
        self.put_variable("cache", "length", jnp.zeros((batch_size,), jnp.int32))

    def encode(self, tokens, **fields):
        self.put_variable("cache", "length", self.get_variable("cache", "length") + tokens.shape[1])
        return jnp.zeros((*tokens.shape, self.vocab_size))

    @nn.compact
    def __call__(self, tokens, *, self_conditioning_logits=None, self_conditioning_mask=None):
        scale = self.param("scale", nn.initializers.constant(30.0), ())
        encoded = (self.get_variable("cache", "length") if self.has_variable("cache", "length")
                   else jnp.zeros(tokens.shape[0], jnp.int32))
        position = encoded[:, None] + jnp.arange(tokens.shape[1])
        return scale * jax.nn.one_hot(position % self.vocab_size, self.vocab_size)


def test_a_user_block_denoiser_generates_through_the_block_task():
    """Block generation runs on what a block denoiser does (`BlockDenoiser`),
    not on DiffusionGemma: a user module that encodes a prompt into its
    cache, refines canvases against it and commits them through `encode`
    continues the prompt's positions canvas after canvas, cropped to the
    budget."""
    model = Counting()
    variables = model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
    task = BlockGeneration(model, variables, BlockProcess(canvas_length=4, vocab_size=11))
    generated = task([[1, 5, 7], [2, 4, 6]], 6, key=0)
    np.testing.assert_array_equal(generated.tokens,
                                  [[1, 5, 7, 3, 4, 5, 6, 7, 8], [2, 4, 6, 3, 4, 5, 6, 7, 8]])
    np.testing.assert_array_equal(generated.lengths, [6, 6])
    assert not bool(generated.terminated.any())


def test_a_causal_decoder_is_no_block_denoiser_and_the_task_names_what_it_lacks():
    """A causal language model scores tokens, but has no canvas to refine or
    clean tokens to commit, so block generation refuses it by name rather
    than fail inside the loop."""
    model = decoder("attention")
    task = BlockGeneration(model, model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)),
                           BlockProcess(canvas_length=4, vocab_size=model.vocab_size))
    with pytest.raises(TypeError, match=r"block denoiser \(BlockDenoiser\).* has no canvas_length, encode"):
        task([[1, 2, 3]], 4, key=0)


def test_seed_is_the_key():
    model = decoder("attention")
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    task = TextGeneration(model, params, sampling=SAMPLING)
    prompt = [[1, 2, 4], [5, 6, 7]]
    assert_same_generation(task(prompt, 3, key=11), task(prompt, 3, key=jax.random.key(11)))
    with pytest.raises(ValueError, match="key must be"):
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
        task(prompt, 3, key=11).host()
        warmed = len(events)
        rebound(prompt, 3, key=11).host()
        task([[9, 8, 7], [1, 1, 1]], 3, key=0).host()
        assert len(events) == warmed
        task(prompt, 6, key=11).host()
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
        _ = bare([[1, 2]], 3, key=0).text
    task = TextGeneration(
        model, params, RunProcessor(Digits()), sampling=Sampling(temperature=0), max_new_tokens=3
    )
    result = task(["12", "5"], key=0)
    assert calls == []
    first = result.text
    assert len(calls) == 2
    assert result.text is first and len(calls) == 2
    rows = result.host()
    assert rows.tokens.shape == (2, 2 + 3) and rows.tokens[1, 0] == 0
    assert result.text == task.decode(result)
    assert result.text == tuple("".join(str(token) for token in row[2:2 + length])
                                for row, length in zip(rows.tokens, rows.lengths, strict=True))


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


@pytest.mark.mesh(devices=1)
def test_a_task_moves_onto_a_mesh_with_the_variables_it_holds():
    """A task holds its variables frozen. Placed on a mesh and bound back,
    they draw what the task drew before: `stream`, which updates a tree's
    dict nodes in place, took the frozen tree's nodes for leaves and refused
    them."""
    from dew.inference.pipeline import place
    from dew.training import Layout, MeshSpec

    model = decoder()
    task = TextGeneration(
        model, model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32)), sampling=SAMPLING
    )
    placed = task.bind(place(task.variables, MeshSpec(), Layout(min_shard=1)))
    assert {len(leaf.sharding.device_set) for leaf in jax.tree.leaves(placed.variables)} == {
        jax.device_count()
    }
    rows = [[1, 2, 3], [4, 5, 6]]
    assert_same_generation(
        placed(rows, 5, key=jax.random.key(1)).host(), task(rows, 5, key=jax.random.key(1)).host()
    )


@pytest.mark.mesh
def test_a_placed_diffusion_gemma_task_keeps_its_rows_sharded_and_draws_the_same_canvases():
    """Placed under the trainer's layout, a canvas task shards its weights,
    splits rows over the mesh's batch axes and reads them back from `host()`;
    with one row per device there is no padding, so the batch-wide canvas
    draw matches the single-device one row for row."""
    from dew.inference.pipeline import place
    from dew.nn.inputs import BATCH_AXES
    from dew.training import Layout, MeshSpec

    loaded = Pretrained.load(FIXTURES / "diffusion-gemma-workflow", dtype="float32", attention_impl="xla",
                             max_seq_len=32)
    plain = loaded.block_generation()
    placed = plain.bind(place(loaded.variables, MeshSpec(fsdp=2), Layout(min_shard=2 ** 6)))
    assert any("fsdp" in str(leaf.sharding.spec) for leaf in jax.tree.leaves(placed.variables))
    prompts = ["<bos> t5 t7 t9 t11", "<bos> t6 t8 t10 t12"] * (jax.device_count() // 2)
    result = placed(prompts, 7, key=11)
    assert result.tokens.sharding.spec == jax.sharding.PartitionSpec(BATCH_AXES)
    assert result.rows == len(prompts) and result.prompt_width == 5
    rows = result.host()
    np.testing.assert_array_equal(rows.tokens, plain(prompts, 7, key=11).host().tokens)
    assert result.text == plain.decode(plain(prompts, 7, key=11))


@pytest.mark.parametrize("kind", ["dpo", "grpo", "ppo"])
def test_pipeline_publishes_the_updated_policy_not_the_frozen_reference(kind, tmp_path):
    import dew
    from dew.data import Dataset
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm import Samples
    from dew.objectives.rl import DPOObjective, GRPOObjective, PPOObjective, ValueHead
    from dew.training import Checkpoints, Trainer

    model = CausalTransformer(vocab_size=8, emb_features=16, num_layers=1, num_heads=2,
                              mlp_features=32, max_seq_len=8, dtype="float32", attention_impl="xla")
    sampling = Sampling(temperature=0)
    # One previewed token under greedy sampling: the budget and policy the run records.
    samples = Samples([1], 1, sampling=sampling)
    if kind == "dpo":
        objective = DPOObjective(model, seq_len=2, samples=samples)
    elif kind == "grpo":
        objective = GRPOObjective(model, seq_len=2, beta=0.1, samples=samples)
    else:
        objective = PPOObjective(model, seq_len=2, critic=ValueHead(model.clone()), beta=0.1, samples=samples)
    trainer = Trainer(objective, optax.sgd(0.1), key=jax.random.key(0))
    count = max(2, jax.device_count())
    if kind == "dpo":
        pairs = np.tile(np.asarray([[[1, 2, 3], [1, 2, 4]]], np.int32), (count, 1, 1))
        batch = {"input_ids": pairs, "completion_mask": np.ones_like(pairs, np.float32)}
    else:
        ids = np.tile(np.asarray([[1, 2, 3]], np.int32), (count, 1))
        before = trainer.initial_state()
        # One packed chain per row, its last id sampled.
        mask = np.tile(np.asarray([[0, 0, 1]], np.float32), (count, 1))
        batch = {"input_ids": ids, "text_segment_ids": np.ones_like(ids),
                 "text_positions": np.tile(np.arange(3, dtype=np.int32), (count, 1)),
                 "response_mask": mask, "advantages": mask}
        actor = objective.actor if kind == "ppo" else objective
        weights = ({collection: value["policy"] for collection, value in before.variables.items()}
                   if kind == "ppo" else before.variables)
        batch["old_log_probs"] = np.asarray(actor.packed_log_probs(weights, batch))
        batch["behavior_log_probs"] = batch["old_log_probs"]
        if kind == "ppo":
            values = np.asarray(objective.values(before.variables, batch))
            batch.update(old_values=values, returns=values + mask)
    data = Dataset(train=lambda partition: iter([batch, batch]), val=None, records=2 * count, batch=count)
    state = trainer.fit(data, steps=2, log_every=100, checkpoint_every=None)
    def draw(weights):
        task = objective.policy(weights) if kind == "ppo" else objective.policy(weights, sampling)
        return task([[1, 2]], 1, key=jax.random.key(3), sampling=sampling).host()
    expected = draw(state.variables)
    actual = objective.pipeline(state)([[1, 2]], 1, key=3, sampling=sampling).host()
    reference = draw(state.averaged)
    np.testing.assert_array_equal(actual.tokens, expected.tokens)
    np.testing.assert_allclose(actual.raw_log_probs, expected.raw_log_probs, atol=1e-7, rtol=1e-7)
    assert not np.allclose(actual.raw_log_probs, reference.raw_log_probs, atol=1e-5, rtol=1e-5)
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(int(state.step), state, None, artifact=objective.inference_record())
    checkpoints.wait()
    restored = dew.pipeline(str(tmp_path))([[1, 2]], key=3).host()
    np.testing.assert_array_equal(restored.tokens, expected.tokens)
    np.testing.assert_allclose(restored.raw_log_probs, expected.raw_log_probs, atol=1e-7, rtol=1e-7)

    from dataclasses import replace

    import jax.numpy as jnp

    from dew.inference import TextGeneration

    baseline = dew.pipeline(str(tmp_path))
    assert isinstance(baseline, TextGeneration)
    params = {**baseline.variables, "params": jax.tree.map(
        lambda leaf: leaf.astype(jnp.bfloat16), baseline.variables["params"])}
    expected_task = replace(baseline, model=baseline.model.clone(dtype=jnp.bfloat16), variables=params)
    converted = dew.pipeline(str(tmp_path), dtype="bfloat16", param_dtype="bfloat16")
    assert isinstance(converted, TextGeneration)
    wanted = expected_task([[1, 2]], 1, key=3).host()
    result = converted([[1, 2]], 1, key=3).host()
    np.testing.assert_array_equal(result.tokens, wanted.tokens)
    np.testing.assert_array_equal(result.raw_log_probs, wanted.raw_log_probs)

