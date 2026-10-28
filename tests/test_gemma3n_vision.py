"""Gemma 3n vision against timm 1.0.29 and transformers 5.16.1.

The locally initialized 84-block fixture and its SGD update come from
``tools/gemma3n_vision_reference.py``. No Torch runtime or model download is
needed to run these comparisons.
"""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from safetensors.numpy import load_file

from dew.interop.hf_decoders import translate_wrapper_config, translate_wrapper_weights
from dew.interop.pretrained import Pretrained
from dew.nn.inputs import ModelInputs
from dew.nn.mobilenet import MobileConvNormAct
from dew.registry import models, projectors, towers

FIXTURE = Path(__file__).parent / "fixtures" / "hf" / "gemma3n-vision-tiny"


@pytest.fixture(scope="module")
def bundle():
    config = json.loads((FIXTURE / "config.json").read_text())
    record = translate_wrapper_config(config)
    variables = translate_wrapper_weights(load_file(str(FIXTURE / "model.safetensors")), record)
    # Match the fp32 reference arithmetic on GPU as well as CPU.
    precision = jax.lax.Precision.HIGHEST
    tower = towers.from_record(record["tower"]).build().clone(precision=precision)
    projector = projectors.from_record(record["projector"]).clone(precision=precision)
    decoder = models.build("causal_transformer", record["text"], dtype="float32", attention_impl="reference")
    decoder = decoder.clone(precision=precision)
    return record, variables, tower, projector, decoder


def test_a_bfloat16_norm_rounds_as_timm_does():
    """timm's RmsNorm2d on a bf16 tower rounds the square, the mean, the
    inverse root and both products to bf16, the weight included
    (tools/timm_rms_norm_reference.py). The block matches it bit for bit
    under jit, where fusion would otherwise carry fp32 between those steps.
    An fp32 reduction, or flax's weight folded into the inverse root, moves
    about a third of the outputs by one bf16 ulp."""
    fixture = np.load(Path(__file__).parent / "fixtures" / "gemma3n" / "rms_norm2d_bf16.npz")
    width = fixture["weight"].shape[0]
    block = MobileConvNormAct(width, activate=False, dtype=jnp.bfloat16)
    params = {"params": {"conv": {"kernel": jnp.eye(width).reshape(1, 1, width, width)},
                         "bn": {"scale": jnp.asarray(fixture["weight"])}}}
    output = jax.jit(block.apply)(params, jnp.asarray(fixture["inputs"], jnp.bfloat16))
    assert output.dtype == jnp.bfloat16
    np.testing.assert_array_equal(np.asarray(output, np.float32), fixture["output"])


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

    def loss(variables, inputs):
        return jnp.mean(tower.apply({"params": variables}, inputs) * coefficients)

    value, (grads, input_grad) = jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))(
        variables["tower"]["params"], pixels)
    np.testing.assert_allclose(value, meta["loss"], rtol=0, atol=1e-7)
    np.testing.assert_allclose(input_grad, np.load(FIXTURE / "input_grad.npy"), rtol=1e-4, atol=1e-7)
    optimizer = optax.sgd(meta["learning_rate"])
    params = variables["tower"]["params"]
    updates, _ = optimizer.update(grads, optimizer.init(params), params)
    stepped = jax.jit(tower.apply)({"params": optax.apply_updates(params, updates)}, pixels)
    np.testing.assert_allclose(stepped, np.load(FIXTURE / "stepped_ref.npy"), rtol=0, atol=1e-4)


def test_image_only_wrapper_matches_conditional_reference_and_uses_hard_tokens(bundle):
    """Full image-to-logit fp32 parity through the native wrapper, including
    hard vision IDs (48..55 here) and the per-layer table that ends at 48.

    The actual conditional model supplies the reference, with no audio input.
    Silencing the hard embedding norm moves the logits by more than 1e-2.
    """
    record, variables, _, projector, _ = bundle
    loaded = Pretrained.load(FIXTURE, dtype="float32", attention_impl="reference")
    model = loaded.model.clone(precision=jax.lax.Precision.HIGHEST)
    pixels = jnp.asarray(np.load(FIXTURE / "pixels_odd.npy"))
    ids = jnp.asarray(np.load(FIXTURE / "input_ids.npy"))
    marks = ids == record["image_token_id"]
    inputs = ModelInputs(ids, {"image_indices": jnp.where(marks, jnp.cumsum(marks, axis=1) - 1, -1)},
                         {"pixel_values": pixels[:, None]})
    run = jax.jit(lambda state: model.apply(state, inputs.tokens, rngs=None, method=None,
                                            mutable=False, capture_intermediates=False, **inputs.kwargs()))
    logits = np.asarray(run(loaded.variables))
    reference = np.load(FIXTURE / "wrapper_ref.npy")
    np.testing.assert_allclose(logits, reference, rtol=0, atol=1e-4)
    np.testing.assert_array_equal(logits.argmax(-1), reference.argmax(-1))
    hard = projector.apply(variables["projector"], jnp.array([[48, 50, 55]]), method=projector.embed_hard)
    np.testing.assert_allclose(hard, np.load(FIXTURE / "hard_ref.npy"), rtol=0, atol=1e-4)
    muted = jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf) if any(
            getattr(part, "key", None) == "hard_embedding_norm" for part in path) else leaf,
        loaded.variables)
    assert np.max(np.abs(np.asarray(run(muted)) - reference)) > 1e-2


def test_drop_path_uses_training_rng_and_is_disabled_for_evaluation(bundle):
    _, variables, tower, _, _ = bundle
    tower = tower.clone(construction=dataclasses.replace(tower.construction, drop_path_rate=0.4))
    pixels = jnp.asarray(np.load(FIXTURE / "pixels_even.npy"))
    training = jax.jit(lambda key: tower.apply(variables["tower"], pixels, train=True,
                                             rngs={"dropout": key}))
    first = training(jax.random.key(1))
    np.testing.assert_array_equal(first, training(jax.random.key(1)))
    assert np.max(np.abs(np.asarray(first - training(jax.random.key(2))))) > 1e-2
    evaluation = jax.jit(tower.apply)(variables["tower"], pixels)
    np.testing.assert_allclose(evaluation, np.load(FIXTURE / "tower_even.npy"), rtol=0, atol=1e-4)
