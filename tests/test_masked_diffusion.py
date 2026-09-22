"""The masked-diffusion sampler and the held-weights seam on a loaded checkpoint.

The sampler half runs tokens in and tokens out. The seam half is what lets a
run continue from LLaDA's or Dream's released weights: the objective reports
the loaded tree through `held_variables`, the trainer binds it as the
initializer's argument and builds its state from it. The trained
source-format export of both families lives in test_masked_diffusion_export.py.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import dew.nn.backbones.causal_transformer  # noqa: F401, registers the backbone
from dew.checkpoints import Checkpoints
from dew.config import ModelConfig
from dew.data import Dataset
from dew.diffusion.discrete import MDLM, Unmask
from dew.inference import pipeline
from dew.interop import load_pretrained
from dew.interop.hf_decoders import translate_config, translate_weights
from dew.nn.inputs import BATCH_AXES, ModelInputs
from dew.objectives.base import Step
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.registry import models, with_precision
from dew.sampling import Sampling, sample
from dew.training import Layout, MeshSpec, Trainer
from dew.training.distributed import build_mesh

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"


def loaded(name: str):
    """A committed tiny masked-diffusion checkpoint as its model and variables."""
    from safetensors.numpy import load_file

    directory = FIXTURES / name
    config = translate_config(json.loads((directory / "config.json").read_text()))
    assert config["mask_token_id"] == 120
    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", config, dtype="float32", attention_impl="reference"))
    variables = translate_weights(load_file(str(directory / "model.safetensors")), config)
    return model, variables


def flat(tree):
    return {".".join(str(entry.key) for entry in path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}


def test_unmask_sampler_runs_on_a_loaded_model_end_to_end():
    """Evaluation on llada-tiny unmasks the fully masked rows into vocabulary
    ids: [4, 12] int32 with no mask id left. A sampler returning its input
    would fail on the mask check."""
    model, variables = loaded("llada-tiny")
    objective = MaskedDiffusionObjective(model, MDLM(mask_id=120)(), seq_len=12,
                                        steps=8, samples=4)
    out = objective.preview(
        variables, {}, Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(0), ema=None))
    assert out is not None, "preview returns the artifact on process zero"
    tokens = np.asarray(out.tokens)
    assert tokens.shape == (4, 12) and tokens.dtype == np.int32
    assert bool(((tokens != 120) & (tokens >= 0) & (tokens < 128)).all())
    # Scoring is teacher-forced: every token of the batch carries its weighted
    # masked cross entropy and counts once, so the pass's perplexity is exp of
    # the ELBO bound per token.
    scored = objective.evaluate(
        variables, {"text": np.zeros((7, 12), np.int32)},
        Step(step=jnp.zeros((), jnp.int32), key=jax.random.key(1), ema=None))
    assert scored.losses.shape == (7, 12) and scored.weights.shape == (7, 12)
    np.testing.assert_array_equal(np.asarray(scored.weights), 1.0)
    assert bool(np.isfinite(np.asarray(scored.losses)).all()) and float(np.asarray(scored.losses).max()) > 0


def test_the_objective_reports_the_loaded_tree_and_returns_it_from_init():
    """The held-weights contract. `held_variables` is what the trainer's
    boundary binds as an argument, and `init` returns that tree whether the
    caller passes it or leaves the objective to read its own. An objective
    holding nothing binds nothing and draws the model's tree from the key."""
    model, variables = loaded("llada-tiny")
    holding = MaskedDiffusionObjective(model, MDLM(mask_id=120)(), seq_len=12,
                                       ema_decay=None, pretrained=variables)
    fresh = MaskedDiffusionObjective(model, MDLM(mask_id=120)(), seq_len=12, ema_decay=None)
    tree_bytes = sum(int(np.asarray(leaf).nbytes) for leaf in jax.tree.leaves(variables))

    assert holding.held_variables() is variables and fresh.held_variables() is None
    assert sum(int(np.asarray(leaf).nbytes)
               for leaf in jax.tree.leaves(holding.initializer)) == tree_bytes
    assert jax.tree.leaves(fresh.initializer) == []

    key = jax.random.key(0)
    held = flat(variables)
    for tree in (holding.init(key), holding.initializer(key), fresh.init(key, variables)):
        assert flat(tree).keys() == held.keys()
        for name, leaf in flat(tree).items():
            np.testing.assert_array_equal(np.asarray(leaf), np.asarray(held[name]))
    drawn = flat(fresh.init(key))
    assert drawn.keys() == held.keys()
    assert any(not np.array_equal(np.asarray(leaf), np.asarray(held[name]))
               for name, leaf in drawn.items()), "the fresh init returned the held tree"


def test_a_tree_that_is_not_the_variables_dict_is_refused():
    """`pretrained` is the whole variables dict, so a params collection
    handed over on its own is named rather than initialising a tree whose
    leaves sit one level too high."""
    model, variables = loaded("llada-tiny")
    objective = MaskedDiffusionObjective(model, MDLM(mask_id=120)(), seq_len=12,
                                         ema_decay=None, pretrained=variables["params"])

    with pytest.raises(ValueError, match="params"):
        objective.init(jax.random.key(0))


def test_the_trainer_builds_its_state_from_the_held_checkpoint():
    """What the seam exists for: a real `Trainer` accepts the objective and
    its initial state is the loaded checkpoint, so the run continues from
    Dream's weights instead of a fresh draw."""
    model, variables = loaded("dream-tiny")
    objective = MaskedDiffusionObjective(model, MDLM(mask_id=120)(), seq_len=12,
                                         ema_decay=None, pretrained=variables)

    state = Trainer(objective, optax.sgd(1e-2), key=jax.random.key(0)).initial_state()

    built, held = flat(state.params), flat(variables)
    assert built.keys() == held.keys()
    for name, leaf in built.items():
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(held[name]))


@pytest.fixture(scope="module", params=["llada-tiny", "dream-tiny"])
def masked_source(request):
    source = load_pretrained(FIXTURES / request.param, dtype="float32", attention_impl="xla")
    tokens = jnp.asarray([[0, 120, 5, 6], [1, 2, 3, 4], [0, 0, 7, 8]])
    fields = {"attention_mask": jnp.asarray([[0, 1, 1, 1], [1, 1, 1, 1], [0, 0, 1, 1]], bool),
              "positions": jnp.asarray([[0, 3, 4, 5], [0, 1, 2, 3], [0, 0, 4, 5]])}
    return source, ModelInputs(tokens, fields)


def test_native_masked_task_matches_direct_mdlm_trajectory(masked_source):
    source, inputs = masked_source
    task = source.text_generation()
    key = jax.random.key(7)
    result = task(inputs, 8, key=key, steps=5).host()
    np.testing.assert_array_equal(result.tokens[:, :4], inputs.tokens)
    assert np.all(result.tokens[:, 4:] != 120)
    np.testing.assert_array_equal(result.lengths, 8)
    np.testing.assert_array_equal(result.decoder_steps, 5)
    np.testing.assert_array_equal(result.terminated, False)
    for row in range(3):
        tokens = jnp.concatenate([inputs.tokens[row:row + 1], jnp.full((1, 8), 120)], axis=1)
        valid = jnp.concatenate([inputs.token_fields["attention_mask"][row:row + 1], jnp.ones((1, 8), bool)], axis=1)
        positions = inputs.token_fields["positions"][row:row + 1]
        positions = jnp.concatenate([positions, positions.max() + jnp.arange(1, 9)[None]], axis=1)
        prepared = ModelInputs(tokens, {"attention_mask": valid, "positions": positions})
        mutable = jnp.arange(12)[None] >= 4
        denoise = task.process.denoiser(source.model, source.variables, inputs=prepared, mutable_mask=mutable)
        expected = sample(denoise, tokens, 5, solver=Unmask(), key=jax.random.fold_in(key, row))
        np.testing.assert_array_equal(result.tokens[row], expected[0])


def test_masked_continuations_seed_eos_and_zero_budget(masked_source):
    source, inputs = masked_source
    task = source.text_generation()
    single = task(inputs, 8, seed=7, steps=5).host()
    many = task(inputs, 8, seed=7, steps=5, n=3).host()
    np.testing.assert_array_equal(many.tokens.reshape(3, 3, 12)[:, 0], single.tokens)
    pair = task(inputs, 8, key=jax.random.key(7), steps=5, n=2).host()
    np.testing.assert_array_equal(many.tokens.reshape(3, 3, 12)[:, :2], pair.tokens.reshape(3, 2, 12))
    assert not np.array_equal(task(inputs, 8, seed=8, steps=5).host().tokens, single.tokens)
    np.testing.assert_array_equal(many.tokens[:, :4], np.repeat(np.asarray(inputs.tokens), 3, axis=0))
    eos = int(single.tokens[0, 5])
    stopped = replace(task, eos_token_ids=(eos,))(inputs, 8, seed=7, steps=5).host()
    for row, response in enumerate(single.tokens[:, 4:]):
        hits = np.flatnonzero(response == eos)
        length = int(hits[0]) + 1 if hits.size else 8
        assert stopped.lengths[row] == length
        assert stopped.terminated[row] == bool(hits.size)
        np.testing.assert_array_equal(stopped.tokens[row, 4:4 + length], response[:length])
        np.testing.assert_array_equal(stopped.tokens[row, 4 + length:], task.pad_token_id)
    np.testing.assert_array_equal(stopped.decoder_steps, single.decoder_steps)
    empty = task(inputs, 0, seed=7, n=2).host()
    np.testing.assert_array_equal(empty.tokens, np.repeat(np.asarray(inputs.tokens), 2, axis=0))
    np.testing.assert_array_equal(empty.lengths, 0)
    np.testing.assert_array_equal(empty.decoder_steps, 0)
    with pytest.raises(TypeError):
        source.text_generation(sampling=Sampling())
    with pytest.raises(ValueError, match="num_beams"):
        replace(source, generation_config={"num_beams": 2}).text_generation()
    for controls in ({"sampling": Sampling()}, {"logits": ()}, {"num_beams": 2}):
        with pytest.raises(TypeError):
            task(inputs, 8, seed=7, **controls)


@pytest.mark.mesh
def test_masked_rows_and_continuations_keep_the_mesh_contract(masked_source):
    source, inputs = masked_source
    task = source.text_generation()
    mesh = build_mesh(MeshSpec())
    variables = jax.device_put(source.variables, Layout().shardings(mesh, source.variables))
    expected = task(inputs, 8, seed=7, steps=5, n=2).host()
    placed = task.bind(variables)(inputs, 8, seed=7, steps=5, n=2)
    actual = placed.host()
    assert placed.tokens.sharding.spec == jax.sharding.PartitionSpec(BATCH_AXES)
    assert placed.tokens.shape[0] == jax.device_count() * 2
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(np.asarray(placed.lengths)[6:], 0)
    np.testing.assert_array_equal(np.asarray(placed.decoder_steps)[6:], 0)


def test_masked_training_resume_publish_and_run_pipeline(masked_source, tmp_path):
    source, prompt = masked_source
    model = source.model.clone(dtype=jnp.bfloat16)
    objective = MaskedDiffusionObjective(model, MDLM(mask_id=120)(), 8,
        pretrained=source.variables, ema_decay=0.5)
    rows = jax.device_count()
    batch = {"text": np.full((rows, 8), 7, np.int32)}
    stream = grain.MapDataset.source([batch]).repeat().to_iter_dataset()
    data = Dataset(train=lambda: iter(stream), val=None, records=rows, batch=rows)
    checkpoints = Checkpoints(str(tmp_path / "run"))
    key = jax.random.key(19)
    Trainer(objective, optax.sgd(0.05), key=key, checkpoints=checkpoints).fit(
        data, steps=1, checkpoint_every=1, log_every=1)
    resumed = Trainer(objective, optax.sgd(0.05), key=key,
        checkpoints=Checkpoints(str(tmp_path / "run"))).fit(data, steps=2, checkpoint_every=1, log_every=1)
    direct = Trainer(objective, optax.sgd(0.05), key=key).fit(data, steps=2, log_every=1)
    for left, right in zip(jax.tree.leaves(resumed.params), jax.tree.leaves(direct.params), strict=True):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(jax.tree.leaves(resumed.averaged), jax.tree.leaves(direct.averaged), strict=True):
        np.testing.assert_array_equal(left, right)

    fields = {name: value for name, value in source.model_config.items() if name not in ("dtype", "attention_impl")}
    config = ModelConfig("causal_transformer", fields, dtype="bfloat16", attention_impl="xla")
    (tmp_path / "run" / "run.json").write_text(json.dumps({
        "objective": "masked_diffusion", "model": asdict(config), "tokenizer": "byte", "sample_tokens": 8}))
    live = objective.pipeline(resumed, ema=False)
    averaged = objective.pipeline(resumed)
    restored = pipeline(str(tmp_path / "run"), ema=False)
    restored_ema = pipeline(str(tmp_path / "run"))
    for expected, actual in ((live, restored), (averaged, restored_ema)):
        np.testing.assert_array_equal(actual(prompt, seed=7).host().tokens,
                                      expected(prompt, 8, seed=7).host().tokens)
    masked = jnp.full((1, 8), 120, jnp.int32)
    assert not np.array_equal(model.apply(live.variables, masked), model.apply(averaged.variables, masked))
    assert restored.model.dtype == jnp.bfloat16
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(restored.variables))
    converted = pipeline(str(tmp_path / "run"), ema=False, dtype="float32", param_dtype="bfloat16")
    reference = replace(live, model=model.clone(dtype=jnp.float32),
                        variables=jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), live.variables))
    np.testing.assert_array_equal(converted(prompt, seed=7).host().tokens,
                                  reference(prompt, 8, seed=7).host().tokens)
    assert converted.model.dtype == jnp.float32
    assert all(leaf.dtype == jnp.bfloat16 for leaf in jax.tree.leaves(converted.variables))

    source.save(tmp_path / "published", variables=resumed.params)
    reloaded = load_pretrained(tmp_path / "published", dtype="bfloat16", attention_impl="xla")
    expected = live(prompt, 8, seed=7).host().tokens
    np.testing.assert_array_equal(reloaded.text_generation()(prompt, 8, seed=7).host().tokens, expected)
    np.testing.assert_array_equal(pipeline(str(tmp_path / "published"), dtype="bfloat16")(
        prompt, 8, seed=7).host().tokens, expected)
    raw = restored(["a", "bc"], 3, seed=7, steps=3)
    assert raw.text == restored.decode(raw)


def test_masked_continuation_uses_last_valid_packed_segment(masked_source):
    source, _ = masked_source
    task = source.text_generation()
    final = ModelInputs(jnp.asarray([[7, 8, 9, 0, 0]]), {
        "attention_mask": jnp.asarray([[1, 1, 1, 0, 0]], bool),
        "positions": jnp.asarray([[0, 1, 2, 42, 42]]),
        "segment_ids": jnp.asarray([[2, 2, 2, 0, 0]])})
    packed = ModelInputs(jnp.concatenate([jnp.arange(1, 17)[None], final.tokens], axis=1), {
        "attention_mask": jnp.concatenate([jnp.ones((1, 16), bool), final.token_fields["attention_mask"]], axis=1),
        "positions": jnp.concatenate([jnp.arange(100, 116)[None], final.token_fields["positions"]], axis=1),
        "segment_ids": jnp.concatenate([jnp.ones((1, 16), jnp.int32), final.token_fields["segment_ids"]], axis=1)})
    alone = task(final, 8, seed=7, steps=1).host()
    together = task(packed, 8, seed=7, steps=1).host()
    np.testing.assert_array_equal(together.tokens[:, :21], packed.tokens)
    np.testing.assert_array_equal(together.tokens[:, -8:], alone.tokens[:, -8:])


def test_masked_source_refuses_active_ar_controls_by_name(masked_source):
    source, _ = masked_source
    for name, value in (("repetition_penalty", 2.0), ("suppress_tokens", [7]),
                        ("stop_strings", ["stop"]), ("num_beams", 2),
                        ("temperature", 0.5), ("use_cache", True)):
        with pytest.raises(ValueError, match=name):
            replace(source, generation_config={name: value}).text_generation()


def test_masked_source_accepts_neutral_controls_and_honors_task_metadata(masked_source):
    source, inputs = masked_source
    controls = {"repetition_penalty": 1.0, "suppress_tokens": None, "stop_strings": None,
                "num_beams": 1, "use_cache": False, "temperature": 1.0, "top_k": 0,
                "top_p": 1.0, "min_p": 0.0, "do_sample": True,
                "_from_model_config": True, "transformers_version": "metadata",
                "return_dict_in_generate": True, "max_new_tokens": 8,
                "num_return_sequences": 2, "eos_token_id": [1], "pad_token_id": 0}
    configured = replace(source, generation_config=controls).text_generation()
    expected = replace(source.text_generation(), max_new_tokens=8, n=2, eos_token_ids=(1,), pad_token_id=0)
    actual = configured(inputs, seed=7, steps=5).host()
    wanted = expected(inputs, seed=7, steps=5).host()
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(wanted), strict=True):
        np.testing.assert_array_equal(left, right)


def test_masked_task_refuses_media_and_non_scalar_logical_positions(masked_source):
    source, inputs = masked_source
    task = source.text_generation()
    with pytest.raises(ValueError, match="conditioning"):
        task(replace(inputs, conditioning={"pixel_values": jnp.zeros((3, 1, 3, 4, 4))}), 8, seed=7)
    for name in ("image_indices", "image_groups", "audio_indices"):
        with pytest.raises(ValueError, match=name):
            task(replace(inputs, token_fields={**inputs.token_fields, name: jnp.full((3, 4), -1)}), 8, seed=7)
    with pytest.raises(ValueError, match="positions"):
        task(replace(inputs, token_fields={**inputs.token_fields, "positions": jnp.zeros((3, 4, 3))}), 8, seed=7)


def test_saved_masked_run_refuses_invalid_sample_budget(masked_source, tmp_path):
    source, _ = masked_source
    fields = {name: value for name, value in source.model_config.items() if name not in ("dtype", "attention_impl")}
    config = ModelConfig("causal_transformer", fields, dtype="float32", attention_impl="xla")
    for budget in (-1, True, "8"):
        (tmp_path / "run.json").write_text(json.dumps({"objective": "masked_diffusion",
            "model": asdict(config), "tokenizer": "byte", "sample_tokens": budget}))
        with pytest.raises(ValueError, match="sample_tokens"):
            pipeline(str(tmp_path), ema=False)

