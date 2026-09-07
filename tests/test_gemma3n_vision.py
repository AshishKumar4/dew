"""Gemma 3n vision against timm 1.0.29 and transformers 5.16.1.

The locally initialized 84-block fixture and its SGD update come from
``tools/gemma3n_vision_reference.py``. No Torch runtime or model download is
needed to run these comparisons.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.experimental import checkify
import numpy as np
import optax
import pytest
from safetensors.numpy import load_file

from dew.interop.hf_decoders import translate_wrapper_config, translate_wrapper_weights
from dew.nn import vision as V
from dew.nn.mobilenet import MobileConvNormAct
from dew.registry import models, with_precision


FIXTURE = Path(__file__).parent / "fixtures" / "hf" / "gemma3n-vision-tiny"


@pytest.fixture(scope="module")
def bundle():
    config = json.loads((FIXTURE / "config.json").read_text())
    record = translate_wrapper_config(config)
    variables = translate_wrapper_weights(load_file(str(FIXTURE / "model.safetensors")), record)
    # Match the fp32 reference arithmetic on GPU as well as CPU.
    precision = jax.lax.Precision.HIGHEST
    tower = V.tower_from_record(record["tower"]).build().clone(precision=precision)
    projector = V.projector_from_record(record["projector"]).build().clone(precision=precision)
    decoder = models.build("causal_transformer", **with_precision(
        "causal_transformer", record["text"], dtype="float32", attention_impl="reference"))
    decoder = decoder.clone(precision=precision)
    return record, variables, tower, projector, decoder


def test_bfloat16_gelu_preserves_the_reference_activation():
    """Torch 2.14 CPU ConvNormAct/RmsNorm2d/tanh-GELU on an identity map.

    The input RMS is exactly one in bf16. timm returns -0.04541015625 for
    the negative channel; a bf16 GELU polynomial returns -0.046875.
    """
    block = MobileConvNormAct(8, dtype=jnp.bfloat16)
    params = {"params": {"conv": {"kernel": jnp.eye(8).reshape(1, 1, 8, 8)},
                         "bn": {"scale": jnp.ones(8)}}}
    inputs = jnp.array([2, -2, 0, 0, 0, 0, 0, 0], jnp.bfloat16).reshape(1, 1, 1, 8)
    output = jax.jit(block.apply)(params, inputs)
    reference = np.array([1.953125, -0.04541015625, 0, 0, 0, 0, 0, 0], np.float32)
    np.testing.assert_array_equal(np.asarray(output, np.float32).ravel(), reference)
    assert output.dtype == jnp.bfloat16


@pytest.mark.parametrize("case", ("odd", "even", "pooled"))
def test_mobilenet_encoder_and_soft_projection_match_reference(bundle, case):
    """Odd ratios, unchanged resolution and average-pooling are separate paths.

    fp32 CPU tolerance 1e-4 covers accumulated convolution/reduction rounding.
    The measured forward error is below 2e-5 on the 84-block fixture.
    """
    _, variables, tower, projector, _ = bundle
    pixels = np.load(FIXTURE / f"pixels_{case}.npy")
    expected = np.load(FIXTURE / f"tower_{case}.npy")
    features = jax.jit(tower.apply)(variables["tower"], pixels)
    np.testing.assert_allclose(features, expected, rtol=0, atol=1e-4)
    assert features.dtype == jnp.float32
    soft = projector.apply(variables["projector"], features)
    np.testing.assert_allclose(soft, np.load(FIXTURE / f"soft_{case}.npy"), rtol=0, atol=1e-4)


def test_encoder_backward_and_sgd_update_match_reference(bundle):
    """timm input-gradient error 8.2e-9 and post-update error 1.5e-5 on CPU.

    A zero gradient would miss the reference input gradient by 0.0021.
    The post-update output checks parameter gradients across the whole tower.
    """
    _, variables, tower, _, _ = bundle
    pixels = jnp.asarray(np.load(FIXTURE / "pixels_odd.npy"))
    meta = json.loads((FIXTURE / "meta.json").read_text())
    coefficients = jnp.linspace(-1, 1, 8192).reshape(1, 4, 2048)

    def loss(params, inputs):
        return jnp.mean(tower.apply({"params": params}, inputs) * coefficients)

    value, (grads, input_grad) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(
        variables["tower"]["params"], pixels)
    np.testing.assert_allclose(value, meta["loss"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(input_grad, np.load(FIXTURE / "input_grad.npy"), rtol=1e-4, atol=1e-7)
    optimizer = optax.sgd(meta["learning_rate"])
    params = variables["tower"]["params"]
    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    stepped = jax.jit(tower.apply)({"params": optax.apply_updates(params, updates)}, pixels)
    np.testing.assert_allclose(stepped, np.load(FIXTURE / "stepped_ref.npy"), rtol=0, atol=1e-4)


def _wrapper_forward(bundle, variables, pixels, ids, positions):
    _, _, tower, projector, decoder = bundle
    features = tower.apply(variables["tower"], pixels)
    soft = projector.apply(variables["projector"], features)
    embeddings = decoder.apply(variables["language_model"], ids,
                                method=lambda model, tokens: model.embed_tokens(tokens))
    embeddings = embeddings * jnp.asarray(decoder.emb_features ** 0.5, embeddings.dtype)
    prepared = projector.apply(
        variables["projector"], embeddings, ids, soft, positions,
        per_layer_input_vocab=decoder.per_layer_input_vocab, method=projector.model_inputs)
    return decoder.apply(variables["language_model"], **prepared)


def test_image_only_wrapper_matches_conditional_reference_and_uses_hard_tokens(bundle):
    """Full image-to-logit fp32 parity, including hard vision IDs and PLE masking.

    The PLE table ends at 48 while image and hard vision IDs exceed it.
    The actual conditional model supplies the reference, with no audio input.
    """
    record, variables, _, projector, _ = bundle
    pixels = jnp.asarray(np.load(FIXTURE / "pixels_odd.npy"))
    ids = jnp.asarray(np.load(FIXTURE / "input_ids.npy"))
    positions = jnp.asarray(np.stack([np.flatnonzero(row == record["image_token_id"])
                                     for row in np.asarray(ids)]), jnp.int32)
    run = jax.jit(checkify.checkify(
        lambda state, tokens: _wrapper_forward(bundle, state, pixels, tokens, positions)))
    error, output = run(variables, ids)
    error.throw()
    logits = np.asarray(output)
    reference = np.load(FIXTURE / "wrapper_ref.npy")
    np.testing.assert_allclose(logits, reference, rtol=0, atol=1e-4)
    np.testing.assert_array_equal(logits.argmax(-1), reference.argmax(-1))
    hard = projector.apply(variables["projector"], jnp.array([[48, 50, 55]]),
                           method=projector.hard_embeddings)
    np.testing.assert_allclose(hard, np.load(FIXTURE / "hard_ref.npy"), rtol=0, atol=1e-4)
    muted = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf) if any(
            getattr(part, "key", None) == "hard_embedding_norm" for part in path) else leaf,
        variables)
    error, muted_output = run(muted, ids)
    error.throw()
    assert np.max(np.abs(np.asarray(muted_output) - reference)) > 1e-2
    audio_ids = jnp.where(ids == 54, 57, ids)
    error, _ = run(variables, audio_ids)
    with pytest.raises(ValueError, match="image-only input_ids"):
        error.throw()


def test_soft_initialization_also_creates_the_hard_vision_path():
    projector = V.Gemma3nProjectorModule(8, 4, vocab_size=3, vocab_offset=16)
    variables = projector.init(jax.random.key(51), jnp.arange(16, dtype=jnp.float32).reshape(1, 2, 8))
    hard = projector.apply(variables, jnp.array([[16, 18]], jnp.int32), method=projector.hard_embeddings)
    np.testing.assert_allclose(jnp.mean(jnp.square(hard), axis=-1), 1.0, atol=1e-4)


def test_drop_path_uses_training_rng_and_is_disabled_for_evaluation(bundle):
    _, variables, tower, _, _ = bundle
    tower = tower.clone(drop_path_rate=0.4)
    pixels = jnp.asarray(np.load(FIXTURE / "pixels_even.npy"))
    training = jax.jit(lambda key: tower.apply(variables["tower"], pixels, train=True,
                                             rngs={"dropout": key}))
    first = training(jax.random.key(1))
    np.testing.assert_array_equal(first, training(jax.random.key(1)))
    assert np.max(np.abs(np.asarray(first - training(jax.random.key(2))))) > 1e-2
    evaluation = jax.jit(tower.apply)(variables["tower"], pixels)
    np.testing.assert_allclose(evaluation, np.load(FIXTURE / "tower_even.npy"), rtol=0, atol=1e-4)


@pytest.mark.parametrize("pixels", (
    np.zeros((1, 32, 32, 3), np.float32),
    np.zeros((3, 32, 32), np.float32),
    np.zeros((1, 3, 32, 32), np.int32),
))
def test_tower_refuses_nonprocessor_pixels(bundle, pixels):
    _, variables, tower, _, _ = bundle
    with pytest.raises(ValueError, match="pixel_values"):
        tower.apply(variables["tower"], pixels)


def test_projector_refuses_incomplete_or_misaligned_image_inputs(bundle):
    _, variables, _, projector, _ = bundle
    embeddings = jnp.ones((1, 4, 32))
    ids = jnp.array([[2, 49, 48, 1]])
    soft = jnp.ones((1, 1, 32))
    with pytest.raises(ValueError, match="arrive together"):
        projector.apply(variables["projector"], embeddings, ids, soft,
                        per_layer_input_vocab=48, method=projector.model_inputs)
    with pytest.raises(ValueError, match="integers"):
        projector.apply(variables["projector"], embeddings, ids, soft, jnp.ones((1, 1)),
                        per_layer_input_vocab=48, method=projector.model_inputs)
    with pytest.raises(ValueError, match="soft_tokens"):
        projector.apply(variables["projector"], embeddings, ids, soft, jnp.array([[1, 2]]),
                        per_layer_input_vocab=48, method=projector.model_inputs)


@pytest.mark.parametrize("change", (
    {"architecture": "resnet50"},
    {"hidden_size": 1024},
    {"do_pooling": True},
    {"model_args": {"features_only": True}},
    {"model_args": {"norm_layer": "batchnorm2d"}},
    {"model_type": "siglip_vision_model"},
    {"unknown_computational_field": True},
))
def test_unsupported_timm_graph_changes_fail_before_loading(change):
    config = json.loads((FIXTURE / "config.json").read_text())
    config["vision_config"].update(change)
    with pytest.raises(ValueError):
        translate_wrapper_config(config)


def test_wrapper_checks_audio_weights_and_wrong_soft_token_count(bundle):
    """An audio config translates to its tower record; its weights must then exist."""
    config = json.loads((FIXTURE / "config.json").read_text())
    config["audio_config"] = {"model_type": "gemma3n_audio", "hidden_size": 32}
    record = translate_wrapper_config(config)
    assert record["audio"]["kind"] == "gemma3n_audio" and record["audio_soft_tokens"] == 188
    with pytest.raises(ValueError, match="audio checkpoint is missing"):
        translate_wrapper_weights(load_file(str(FIXTURE / "model.safetensors")), record)
    config["audio_config"] = None
    config["vision_soft_tokens_per_image"] = 9
    with pytest.raises(ValueError, match="vision_soft_tokens_per_image"):
        translate_wrapper_config(config)
    tensors = load_file(str(FIXTURE / "model.safetensors"))
    tensors["model.audio_tower.linear_start.weight"] = np.ones((4, 4), np.float32)
    with pytest.raises(ValueError, match="unknown tensor"):
        translate_wrapper_weights(tensors, bundle[0])


def test_image_only_released_config_builds_the_actual_encoder():
    """The released geometry is checked abstractly, without allocating 300M weights."""
    config = json.loads((FIXTURE.parent / "gemma-3n-e2b" / "config.json").read_text())
    config["audio_config"] = None
    record = translate_wrapper_config(config)
    tower = V.tower_from_record(record["tower"]).build()
    output, _ = jax.eval_shape(tower.init_with_output, jax.random.key(0),
                               jax.ShapeDtypeStruct((1, 3, 768, 768), jnp.float32))
    assert output.shape == (1, 256, 2048)


@pytest.mark.parametrize("bad_id", (47, 56))
def test_hard_vision_ids_reject_both_sides_of_the_vocabulary_eager_and_compiled(bundle, bad_id):
    _, variables, _, projector, _ = bundle
    ids = jnp.array([[bad_id, 48]], jnp.int32)
    with pytest.raises(ValueError, match="vision token IDs"):
        projector.apply(variables["projector"], ids, method=projector.hard_embeddings)
    checked = jax.jit(checkify.checkify(lambda tokens: projector.apply(
        variables["projector"], tokens, method=projector.hard_embeddings)))
    error, _ = checked(ids)
    with pytest.raises(ValueError, match="vision token IDs"):
        error.throw()
    error, valid = checked(jnp.array([[48, 55]], jnp.int32))
    error.throw()
    np.testing.assert_allclose(valid, np.load(FIXTURE / "hard_ref.npy")[:, [0, 2]], rtol=0, atol=1e-4)


@pytest.mark.parametrize("bad_id", (-1, 56, 57))
def test_image_only_inputs_reject_negative_upper_boundary_and_audio_ids(bundle, bad_id):
    _, variables, _, projector, _ = bundle
    embeddings = jnp.ones((1, 2, 32))
    ids = jnp.array([[2, bad_id]], jnp.int32)

    def prepare(tokens):
        return projector.apply(variables["projector"], embeddings, tokens,
                               per_layer_input_vocab=48, method=projector.model_inputs)

    with pytest.raises(ValueError, match="image-only input_ids"):
        prepare(ids)
    checked = jax.jit(checkify.checkify(prepare))
    error, _ = checked(ids)
    with pytest.raises(ValueError, match="image-only input_ids"):
        error.throw()
    error, valid = checked(jnp.array([[2, 55]], jnp.int32))
    error.throw()
    np.testing.assert_array_equal(valid["input_embeddings"][:, 0], embeddings[:, 0])
    np.testing.assert_allclose(valid["input_embeddings"][:, 1],
                               np.load(FIXTURE / "hard_ref.npy")[:, -1], rtol=0, atol=1e-4)


def test_checked_hard_embedding_gradients_remain_jittable(bundle):
    _, variables, _, projector, _ = bundle
    ids = jnp.array([[48, 50, 55]], jnp.int32)
    coefficients = jnp.linspace(-1, 1, 96).reshape(1, 3, 32)

    def loss(params):
        embeddings = projector.apply({"params": params}, ids, method=projector.hard_embeddings)
        return jnp.mean(embeddings * coefficients)

    params = variables["projector"]["params"]
    expected_value, expected_grad = jax.value_and_grad(loss)(params)
    error, (value, gradients) = jax.jit(checkify.checkify(jax.value_and_grad(loss)))(params)
    error.throw()
    np.testing.assert_allclose(value, expected_value, rtol=0, atol=1e-6)
    for actual, expected in zip(jax.tree.leaves(gradients), jax.tree.leaves(expected_grad), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("field,value", [("vocab_size", "8"), ("vocab_offset", "48"),
                                         ("rms_norm_eps", "x"), ("vocab_size", True)])
def test_wrapper_rejects_mistyped_vision_values_before_building(field, value):
    config = json.loads((FIXTURE / "config.json").read_text())
    config["vision_config"][field] = value
    with pytest.raises(ValueError, match=field):
        translate_wrapper_config(config)
