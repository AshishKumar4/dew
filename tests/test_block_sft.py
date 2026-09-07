"""Google's actual full-response DiffusionGemma SFT as a committed numerical reference.

Both canvases are evaluated. Row zero selects its second, two-token canvas;
row one selects its first, three-token canvas. Encoder target supports also
 differ. The self-conditioning draw enables row zero and disables row one.
"""

from pathlib import Path
import math

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from safetensors.numpy import load_file

from dew import Dataset, Trainer
from dew.checkpoints import Checkpoints
from dew.interop import load_pretrained
from dew.interop.diffusion_gemma import translate_weights
from dew.objectives.base import Step, scalar_loss
from dew.objectives.diffusion.block import BlockDiffusionObjective

FIXTURE = Path(__file__).resolve().parent / "fixtures/hf/diffusion-gemma-sft"
REFERENCES = FIXTURE / "reference"


@pytest.fixture(scope="module")
def source():
    loaded = load_pretrained(FIXTURE, dtype="float32", attention_impl="xla", max_seq_len=32)
    with np.load(FIXTURE / "reference.npz") as arrays:
        reference = {name: arrays[name] for name in arrays.files}
    batch = {"text": reference["tokens"], "canvas_mask": reference["canvas_mask"],
             "encoder_target_mask": reference["encoder_target_mask"]}
    step = Step(step=jnp.asarray(0, jnp.int32), key=jax.random.wrap_key_data(reference["step_key"]), ema=None)
    return loaded, batch, step, reference


def objective(loaded, **kwargs):
    return BlockDiffusionObjective(loaded.model, prompt_length=4, num_canvases=2,
                                   pretrained=loaded.variables, **kwargs)


def assert_tree_close(actual, expected, tolerance):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_allclose(left, right, atol=tolerance, rtol=0)


def test_official_sft_loss_and_every_parameter_gradient(source):
    """fp32 loss and all parameter gradients against Google's real SFTDiffusion.

    Observed loss error 0, maximum gradient error 3.5e-5, and SGD parameter
    error 6.0e-8. Tolerances are 1e-5, 1e-4, and 2e-6 respectively. The
    reference covers full-response circular overlays and both SC branches;
    dropping the encoder gradient changes parameter gradients by 0.76.
    """
    loaded, batch, step, reference = source
    obj = objective(loaded)
    variables = jax.tree.map(jnp.asarray, loaded.variables)
    (loss, aux), gradient = jax.value_and_grad(
        lambda values: scalar_loss(obj, values, batch, step), has_aux=True)(variables)
    np.testing.assert_allclose(loss, reference["loss"], atol=1e-5, rtol=0)
    np.testing.assert_allclose(aux.metrics["canvas_ce"], reference["canvas_loss"].mean(), atol=1e-5, rtol=0)
    np.testing.assert_allclose(aux.metrics["encoder_ce"], reference["encoder_loss"].mean(), atol=1e-5, rtol=0)
    expected = translate_weights(load_file(str(REFERENCES / "gradient.safetensors")), loaded.config)
    assert_tree_close(gradient, expected, 1e-4)
    updated = jax.tree.map(lambda value, grad: value - 0.001 * grad, variables, gradient)
    reference_update = translate_weights(load_file(str(REFERENCES / "updated.safetensors")), loaded.config)
    assert_tree_close(updated, reference_update, 2e-6)


def test_disabling_encoder_gradient_matches_the_reference_control(source):
    loaded, batch, step, _ = source
    obj = objective(loaded, stop_gradient_from_denoiser_to_encoder=True)
    gradient = jax.grad(lambda values: scalar_loss(obj, values, batch, step)[0])(
        jax.tree.map(jnp.asarray, loaded.variables))
    expected = translate_weights(load_file(str(REFERENCES / "detached_gradient.safetensors")), loaded.config)
    assert_tree_close(gradient, expected, 1e-4)
    full = translate_weights(load_file(str(REFERENCES / "gradient.safetensors")), loaded.config)
    assert max(float(jnp.max(jnp.abs(a - b))) for a, b in zip(jax.tree.leaves(gradient), jax.tree.leaves(full))) > 1e-3


@pytest.mark.parametrize("probability,reference_key", [(0., "sc_off_loss"), (1., "sc_on_loss")])
def test_both_self_conditioning_branches_match_the_official_adapter(source, probability, reference_key):
    loaded, batch, step, reference = source
    obj = objective(loaded, self_cond_prob=probability)
    loss, _ = scalar_loss(obj, jax.tree.map(jnp.asarray, loaded.variables), batch, step)
    np.testing.assert_allclose(loss, reference[reference_key], atol=1e-5, rtol=0)


def test_loss_weights_apply_after_independent_row_normalization(source):
    loaded, batch, step, reference = source
    obj = objective(loaded, encoder_loss_weight=0.3, decoder_loss_weight=2.7)
    loss, _ = scalar_loss(obj, jax.tree.map(jnp.asarray, loaded.variables), batch, step)
    expected = 0.3 * reference["encoder_loss"].mean() + 2.7 * reference["canvas_loss"].mean()
    np.testing.assert_allclose(loss, expected, atol=1e-5, rtol=0)
    canvas_mass = reference["selected_mask"].sum(axis=(1, 2))
    encoder_mass = reference["encoder_target_mask"].sum(axis=-1)
    wrong = (2.7 * np.average(reference["canvas_loss"], weights=canvas_mass)
             + 0.3 * np.average(reference["encoder_loss"], weights=encoder_mass))
    assert abs(float(loss) - wrong) > 1e-3


def dataset(batch):
    rows = next(iter(batch.values())).shape[0]
    records = [{name: value[row] for name, value in batch.items()} for row in range(rows)]
    stream = grain.MapDataset.source(records).repeat().batch(rows, drop_remainder=True).to_iter_dataset()
    return Dataset(train=lambda: iter(stream), val=None, records=rows, batch=rows)


def test_real_trainer_update_and_checkpoint_resume(source, tmp_path):
    loaded, original_batch, step, reference = source
    rows = math.lcm(2, jax.device_count())
    batch = jax.tree.map(lambda value: np.concatenate([value] * (rows // 2), axis=0), original_batch)
    obj = objective(loaded)
    run_key = jax.random.key(int(reference["run_seed"]))
    data = dataset(batch)
    variables = jax.tree.map(jnp.asarray, loaded.variables)
    gradient = jax.grad(lambda values: scalar_loss(obj, values, batch, step)[0])(variables)
    expected = jax.tree.map(lambda value, grad: value - 0.001 * grad, variables, gradient)
    trainer = Trainer(obj, optax.sgd(0.001), key=run_key, checkpoints=Checkpoints(str(tmp_path / "run")))
    updated = trainer.fit(data, steps=1, checkpoint_every=1, log_every=1)
    assert_tree_close(updated.params, expected, 2e-6)
    loaded.save(tmp_path / "published", variables=updated.params)
    readback = load_pretrained(tmp_path / "published", dtype="float32", attention_impl="xla", max_seq_len=32)
    assert_tree_close(readback.variables, updated.params, 0)

    resumed = Trainer(obj, optax.sgd(0.001), key=run_key,
                      checkpoints=Checkpoints(str(tmp_path / "run"))).fit(
                          data, steps=2, checkpoint_every=1, log_every=1)
    direct = Trainer(obj, optax.sgd(0.001), key=run_key).fit(data, steps=2, log_every=1)
    assert_tree_close(resumed.params, direct.params, 0)
    assert int(resumed.updates) == 2
