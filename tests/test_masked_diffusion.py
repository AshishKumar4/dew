"""The masked-diffusion sampler and the held-weights seam on a loaded checkpoint.

The sampler half runs tokens in and tokens out. The seam half is what lets a
run continue from LLaDA's or Dream's released weights: the objective reports
the loaded tree through `held_variables`, the trainer binds it as the
initializer's argument and builds its state from it. The trained
source-format export of both families lives in test_masked_diffusion_export.py.
"""

import json
from dataclasses import replace
from importlib import import_module
from pathlib import Path

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from model_support import flat_tree
from reference_error import assert_as_exact

from dew.checkpoints import Checkpoints
from dew.data import ByteTokenizer
from dew.data.dataset import Loader
from dew.diffusion.discrete import MDLM, Unmask
from dew.inference import RunProcessor, pipeline
from dew.interop import Pretrained
from dew.interop.hf_decoders import translate_config, translate_weights
from dew.nn.inputs import ModelInputs, coordinates, request_axes
from dew.nn.protocols import OutputTable
from dew.objectives.base import Step
from dew.objectives.diffusion.masked import MaskedDiffusionObjective
from dew.registry import models
from dew.sampling import Sampling, sample
from dew.training import Layout, MeshSpec, Trainer

import_module("dew.nn.backbones.causal_transformer")  # registers the backbone

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"


def test_biased_masked_objective_scores_the_same_affine_logits_as_its_forward():
    model = models.build('causal_transformer', vocab_size=32, emb_features=8, num_layers=1,
                         num_heads=2, mlp_features=16, max_seq_len=4, causal=False,
                         head_bias=True, qk_norm=False, attention_impl='reference', dtype=jnp.float32)
    tokens = jnp.asarray([[2, 5, 7, 8], [3, 1, 4, 9]], jnp.int32)
    variables = model.init(jax.random.key(0), tokens)
    variables['params']['head_bias'] = jnp.linspace(-.123, .389, 32)
    process = MDLM(mask_id=31)()
    objective = MaskedDiffusionObjective(model, process, seq_len=4, head_chunks=4,
                                         variables=variables, ema_decay=None)
    time_key, mask_key, _ = jax.random.split(jax.random.key(1), 3)
    times = process.sample_t(time_key, tokens.shape[0])
    corrupted, is_masked = process.corrupt(mask_key, tokens, times)
    # The objective reads this corruption (`Step.tape`).
    step = Step(jnp.asarray(0), jax.random.key(1), None, tape={"times": times, "mask": is_masked})
    _, losses, _, _, predicted, _ = objective._token_losses(variables, {'text': tokens}, step, train=False)
    logits = model.apply(variables, corrupted).at[..., 31].set(-jnp.inf)
    expected = optax.softmax_cross_entropy_with_integer_labels(logits, tokens)
    # The same 1e-5 relative loss contract as the bias-free chunked head.
    assert jnp.abs(losses - expected).max() <= 1e-5 * jnp.abs(expected).max()
    np.testing.assert_array_equal(predicted, logits.argmax(-1))


class _MediaReader(nn.Module):
    """A bidirectional model whose every state reads how many media slots its row holds."""

    causal = False
    mask_token_id = None

    @nn.compact
    def hidden_states(self, tokens, train=False, image_indices=None):
        media = jnp.zeros(tokens.shape) if image_indices is None else (image_indices >= 0) * 1.0
        return nn.Embed(32, 8)(tokens) + media.sum(-1)[:, None, None] * self.param("media", nn.zeros, (8,))

    def output_table(self):
        return OutputTable(self.variables["params"]["Embed_0"]["embedding"], vocab_major=True)

    def __call__(self, tokens, **fields):
        return self.hidden_states(tokens, **fields)


def test_a_media_slot_reaches_the_model_and_is_never_masked_or_scored():
    tokens = jnp.tile(jnp.arange(1, 9, dtype=jnp.int32), (8, 1))
    slots = jnp.where(jnp.arange(8) < 2, jnp.arange(8), -1) * jnp.ones((8, 1), jnp.int32)
    objective = MaskedDiffusionObjective(_MediaReader(), MDLM(mask_id=31)(), seq_len=8, ema_decay=None)
    params = objective.init(jax.random.key(0))
    params["params"]["media"] = jnp.ones(8)
    media = {"text": ModelInputs(tokens, {"image_indices": slots})}
    step = Step(jnp.asarray(0), jax.random.key(1), None)
    _, losses, _, counted, _, real = objective._token_losses(params, media, step, train=False)
    plain = objective._token_losses(params, {"text": tokens}, step, train=False)[1]

    np.testing.assert_array_equal(real, slots < 0)
    assert not counted[:, :2].any() and counted[:, 2:].any()
    assert not np.allclose(losses[:, 2:], plain[:, 2:])


@pytest.mark.parametrize("given", ["attention_mask", "coordinates"])
def test_padding_that_holds_the_mask_token_scores_nothing(given):
    """A padded row whose padding slots hold the mask token, as a tokenizer padding with it writes
    them: the loss, its gradient and the evaluation's per-token scores are finite and the same as
    with any other token there, since padding is neither masked nor scored. The padding is
    marked by the attention mask, or by the coordinates' validity the decoder reads."""
    model = models.build('causal_transformer', vocab_size=32, emb_features=8, num_layers=1,
                         num_heads=2, mlp_features=16, max_seq_len=4, causal=False, qk_norm=False,
                         attention_impl='reference', dtype=jnp.float32)
    objective = MaskedDiffusionObjective(model, MDLM(mask_id=31)(), seq_len=4, head_chunks=4,
                                         ema_decay=None)
    variables = objective.init(jax.random.key(0))
    mask = jnp.asarray([[1, 1, 0, 0]], bool)
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)

    def scored(padding):
        tokens = jnp.asarray([[2, 5, padding, padding]], jnp.int32)
        batch = ({"text": ModelInputs(tokens, {"attention_mask": mask})} if given == "attention_mask"
                 else {"text": tokens, "text_coordinates": coordinates(valid=mask)})
        loss, grads = jax.value_and_grad(lambda v: objective.scalar_loss(v, batch, step)[0])(variables)
        scoring = Step(jnp.asarray(0), jax.random.key(2), None)
        return loss, grads, objective._evaluation_scores(variables, batch, scoring)

    mask_padded, other_padded = scored(31), scored(7)
    assert np.isfinite(float(mask_padded[0]))
    assert all(np.all(np.isfinite(leaf)) for leaf in jax.tree.leaves(mask_padded[1]))
    np.testing.assert_array_equal(np.asarray(mask_padded[2].losses), np.asarray(other_padded[2].losses))
    assert float(mask_padded[0]) == float(other_padded[0])


def loaded(name: str):
    """A committed tiny masked-diffusion checkpoint as its model and variables."""
    from safetensors.numpy import load_file

    directory = FIXTURES / name
    config = translate_config(json.loads((directory / "config.json").read_text()))
    assert config["mask_token_id"] == 120
    model = models.build("causal_transformer", config, dtype="float32", attention_impl="reference")
    variables = translate_weights(load_file(str(directory / "model.safetensors")), config)
    return model, variables


def test_compiled_samples_follow_the_substituted_masked_model():
    class FixedToken(nn.Module):
        token: int
        causal = False
        mask_token_id = None

        @nn.compact
        def logits(self, tokens, train=False):
            scale = self.param("scale", nn.initializers.ones, ())
            return jnp.broadcast_to(jax.nn.one_hot(self.token, 4) * 100 * scale, (*tokens.shape, 4))

        def __call__(self, tokens, train=False):
            return self.logits(tokens, train=train)

    process = MDLM(mask_id=3)()
    objective = MaskedDiffusionObjective(FixedToken(0), process, seq_len=4, steps=3, samples=2)
    variables = objective.init(jax.random.key(0))
    step = Step(jnp.array(0), jax.random.key(1), None)
    before = objective.preview(variables, {}, step).tokens
    compiled = objective._sample
    objective.substitute([FixedToken(1)])
    after = objective.preview(variables, {}, step).tokens
    np.testing.assert_array_equal(before, 0)
    np.testing.assert_array_equal(after, 1)
    assert objective._sample is not compiled
    assert objective._sample is objective._sample


@pytest.mark.parametrize("name", ["llada-tiny", "dream-tiny"])
def test_a_masked_family_unmasks_each_position_from_the_logits_its_reference_reads(name):
    """LLaDA predicts each position's token from its own logits; Dream-v0, continued from an
    autoregressive model, from the position before (its reference shifts the logits right, the first
    repeated). One unmasking step's distribution at every position is the softmax of those logits of
    the released model, the mask's column left out, as exact as the release's own float32 forward
    is to float64."""
    from reference_error import assert_as_exact_as_the_reference

    from dew.diffusion.discrete import DiscreteProcess, LogLinear

    model, variables = loaded(name)
    directory = FIXTURES / name
    ids = jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)

    def unmasking(logits):
        logits = np.asarray(logits, np.float64)
        if name == "dream-tiny":
            logits = np.concatenate([logits[:, :1], logits[:, :-1]], axis=1)
        logits[..., 120] = -np.inf
        shifted = logits - logits.max(axis=-1, keepdims=True)
        return np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)

    _, log_probs = DiscreteProcess(LogLinear(), mask_id=120).denoiser(model, variables)(ids, jnp.ones((1,)))
    assert_as_exact_as_the_reference(np.exp(np.asarray(log_probs, np.float64)),
                                      unmasking(np.load(directory / "logits.npy")),
                                      unmasking(np.load(directory / "logits_f64.npy")), f"{name} unmasking")


def test_a_shifted_model_reads_each_document_and_padded_row_as_if_alone():
    """Dream reads position i's prediction from position i-1, but a packed document's first token,
    and a left-padded row's first real one, read their own, as the reference does a row's first: a
    packed row's predictions are its documents' each aligned alone, and a padded row's real ones are
    its text's aligned alone."""
    from dew.diffusion.discrete import aligned

    model, _ = loaded("dream-tiny")
    scores = jnp.arange(6.0).reshape(1, 6, 1) + 10
    alone = [aligned(model, scores[:, :3]), aligned(model, scores[:, 3:])]
    packed = aligned(model, scores, {"segment_ids": jnp.array([[1, 1, 1, 2, 2, 2]])})
    np.testing.assert_array_equal(packed, jnp.concatenate(alone, axis=1))
    padded = aligned(model, scores, {"attention_mask": jnp.array([[False, False, True, True, True, True]])})
    np.testing.assert_array_equal(padded[:, 2:], aligned(model, scores[:, 2:]))
    one = aligned(model, scores[:, :1], {"segment_ids": jnp.array([[1]])})
    np.testing.assert_array_equal(one, scores[:, :1])


def test_a_dream_shaped_record_keeps_how_its_logits_line_up():
    """The alignment is a fact of the checkpoint, so the model's record carries it: a Dream config
    reads to a model that shifts, which its record rebuilds as one that shifts, and LLaDA's to one
    that does not."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.registry import from_record, to_record

    for name, shifted in (("dream-tiny", True), ("llada-tiny", False)):
        model, _ = loaded(name)
        assert model.shifted_logits is shifted
        assert from_record(CausalTransformer, to_record(model, CausalTransformer)).shifted_logits is shifted


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


def test_the_trainer_builds_its_state_from_the_held_checkpoint():
    """What the seam exists for: a real `Trainer` accepts the objective and
    its initial state is the loaded checkpoint, so the run continues from
    Dream's weights instead of a fresh draw."""
    model, variables = loaded("dream-tiny")
    objective = MaskedDiffusionObjective(model, MDLM(mask_id=120)(), seq_len=12,
                                         ema_decay=None, variables=variables)

    state = Trainer(objective, optax.sgd(1e-2), key=jax.random.key(0)).initial_state()

    built, held = flat_tree(state.variables), flat_tree(variables)
    assert built.keys() == held.keys()
    for name, leaf in built.items():
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(held[name]))


@pytest.fixture(scope="module", params=["llada-tiny", "dream-tiny", "llada-moe-tiny",
                                      "llada-moe-v2-tiny", "llada2-tiny", "llada22-tiny"])
def masked_source(request):
    source = Pretrained.load(FIXTURES / request.param, dtype="float32", attention_impl="xla")
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
    np.testing.assert_array_equal(result.terminated, desired=False)
    for row in range(3):
        tokens = jnp.concatenate([inputs.tokens[row:row + 1], jnp.full((1, 8), 120)], axis=1)
        valid = jnp.concatenate(
            [inputs.token_fields["attention_mask"][row : row + 1], jnp.ones((1, 8), bool)], axis=1
        )
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
    single = task(inputs, 8, key=7, steps=5).host()
    many = task(inputs, 8, key=7, steps=5, n=3).host()
    np.testing.assert_array_equal(many.tokens.reshape(3, 3, 12)[:, 0], single.tokens)
    pair = task(inputs, 8, key=jax.random.key(7), steps=5, n=2).host()
    np.testing.assert_array_equal(many.tokens.reshape(3, 3, 12)[:, :2], pair.tokens.reshape(3, 2, 12))
    assert not np.array_equal(task(inputs, 8, key=8, steps=5).host().tokens, single.tokens)
    np.testing.assert_array_equal(many.tokens[:, :4], np.repeat(np.asarray(inputs.tokens), 3, axis=0))
    eos = int(single.tokens[0, 5])
    stopped = replace(task, eos_token_ids=(eos,))(inputs, 8, key=7, steps=5).host()
    for row, response in enumerate(single.tokens[:, 4:]):
        hits = np.flatnonzero(response == eos)
        length = int(hits[0]) + 1 if hits.size else 8
        assert stopped.lengths[row] == length
        assert stopped.terminated[row] == bool(hits.size)
        np.testing.assert_array_equal(stopped.tokens[row, 4:4 + length], response[:length])
        np.testing.assert_array_equal(stopped.tokens[row, 4 + length:], task.pad_token_id)
    np.testing.assert_array_equal(stopped.decoder_steps, single.decoder_steps)
    empty = task(inputs, 0, key=7, n=2).host()
    np.testing.assert_array_equal(empty.tokens, np.repeat(np.asarray(inputs.tokens), 2, axis=0))
    np.testing.assert_array_equal(empty.lengths, 0)
    np.testing.assert_array_equal(empty.decoder_steps, 0)
    with pytest.raises(TypeError):
        source.text_generation(sampling=Sampling())
    with pytest.raises(ValueError):
        replace(source, generation_config={"num_beams": 2}).text_generation()
    for controls in ({"sampling": Sampling()}, {"logits": ()}, {"num_beams": 2}):
        with pytest.raises(TypeError):
            task(inputs, 8, key=7, **controls)


@pytest.mark.mesh
def test_masked_rows_and_continuations_keep_the_mesh_contract(masked_source):
    source, inputs = masked_source
    task = source.text_generation()
    mesh = MeshSpec().build()
    variables = jax.device_put(source.variables, Layout().shardings(mesh, source.variables))
    expected = task(inputs, 8, key=7, steps=5, n=2).host()
    placed = task.bind(variables)(inputs, 8, key=7, steps=5, n=2)
    actual = placed.host()
    assert placed.tokens.sharding.spec == jax.sharding.PartitionSpec(request_axes(mesh))
    assert placed.tokens.shape[0] == jax.device_count() * 2
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(left, right)
    np.testing.assert_array_equal(np.asarray(placed.lengths)[6:], 0)
    np.testing.assert_array_equal(np.asarray(placed.decoder_steps)[6:], 0)


def test_masked_training_resume_publish_and_run_pipeline(masked_source, tmp_path):
    source, prompt = masked_source
    model = source.model.clone(dtype=jnp.bfloat16)
    objective = MaskedDiffusionObjective(model, MDLM(mask_id=120)(), 8,
        variables=source.variables, ema_decay=0.5, processor=RunProcessor(ByteTokenizer()))
    rows = jax.device_count()
    batch = {"text": np.full((rows, 8), 7, np.int32)}
    stream = grain.MapDataset.source([batch]).repeat().to_iter_dataset()
    data = Loader(train=lambda partition: iter(stream), val=None, records=rows, batch=rows)
    checkpoints = Checkpoints(str(tmp_path / "run"))
    key = jax.random.key(19)
    Trainer(objective, optax.sgd(0.05), key=key, checkpoints=checkpoints).fit(
        data, steps=1, checkpoint_every=1, log_every=1)
    resumed = Trainer(objective, optax.sgd(0.05), key=key,
        checkpoints=Checkpoints(str(tmp_path / "run"))).fit(data, steps=2, checkpoint_every=1, log_every=1)
    direct = Trainer(objective, optax.sgd(0.05), key=key).fit(data, steps=2, log_every=1)
    for left, right in zip(jax.tree.leaves(resumed.variables), jax.tree.leaves(direct.variables),
                           strict=True):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(jax.tree.leaves(resumed.averaged), jax.tree.leaves(direct.averaged), strict=True):
        np.testing.assert_array_equal(left, right)

    live = objective.pipeline(resumed, ema=False)
    averaged = objective.pipeline(resumed)
    restored = pipeline(str(tmp_path / "run"), ema=False)
    restored_ema = pipeline(str(tmp_path / "run"))
    for expected, actual in ((live, restored), (averaged, restored_ema)):
        np.testing.assert_array_equal(actual(prompt, key=7).host().tokens,
                                      expected(prompt, 8, key=7).host().tokens)
    masked = jnp.full((1, 8), 120, jnp.int32)
    assert not np.array_equal(model.apply(live.variables, masked), model.apply(averaged.variables, masked))
    assert restored.model.dtype == jnp.bfloat16
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(restored.variables))
    converted = pipeline(str(tmp_path / "run"), ema=False, dtype="float32", param_dtype="bfloat16")
    reference = replace(live, model=model.clone(dtype=jnp.float32),
                        variables={**live.variables, 'params': jax.tree.map(
                            lambda leaf: leaf.astype(jnp.bfloat16), live.variables['params'])})
    np.testing.assert_array_equal(converted(prompt, key=7).host().tokens,
                                  reference(prompt, 8, key=7).host().tokens)
    assert converted.model.dtype == jnp.float32
    assert jax.tree.map(lambda leaf: leaf.dtype, converted.variables) == jax.tree.map(
        lambda leaf: leaf.dtype, reference.variables)
    expected_logits = jax.jit(lambda tokens: reference.model.apply(reference.variables, tokens))
    actual_logits = jax.jit(lambda tokens: converted.model.apply(converted.variables, tokens))
    assert_as_exact(actual_logits, expected_logits,
                    lambda seed: (np.asarray([(np.arange(8) + seed) % model.vocab_size], np.int32),),
                    'converted masked decoder logits', seeds=8,
                    truth_fn=lambda tokens: np.asarray(expected_logits(tokens), np.float64))

    source.save(tmp_path / "published", variables=resumed.variables)
    reloaded = Pretrained.load(tmp_path / "published", dtype="bfloat16", attention_impl="xla")
    expected = live(prompt, 8, key=7).host().tokens
    np.testing.assert_array_equal(reloaded.text_generation()(prompt, 8, key=7).host().tokens, expected)
    np.testing.assert_array_equal(pipeline(str(tmp_path / "published"), dtype="bfloat16")(
        prompt, 8, key=7).host().tokens, expected)
    raw = restored(["a", "bc"], 3, key=7, steps=3)
    assert raw.text == restored.decode(raw)


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
    actual = configured(inputs, key=7, steps=5).host()
    wanted = expected(inputs, key=7, steps=5).host()
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(wanted), strict=True):
        np.testing.assert_array_equal(left, right)

