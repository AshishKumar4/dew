"""Google's actual full-response DiffusionGemma SFT as a committed numerical reference.

Both canvases are evaluated. Row zero selects its second, two-token canvas;
row one selects its first, three-token canvas. Encoder target supports also
differ. The self-conditioning draw enables row zero and disables row one.
"""

import math
from dataclasses import replace
from pathlib import Path

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from diffusion_support import as_a_step, float64_gradient, moved_weights, objective_gradient
from flax.traverse_util import flatten_dict, unflatten_dict
from reference_error import assert_as_exact
from safetensors.numpy import load_file

from dew import Trainer
from dew.checkpoints import Checkpoints
from dew.data.dataset import Loader
from dew.interop import Pretrained
from dew.interop.diffusion_gemma import translate_weights
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.diffusion_gemma import DiffusionGemma
from dew.nn.inputs import ModelInputs
from dew.nn.multimodal import VisionConditioner
from dew.nn.precision import at_least_fp32
from dew.nn.sharding import LOGITS, constrain
from dew.nn.vision import GemmaProjector, SiglipVision
from dew.objectives.base import FROZEN, Step, freeze, model_rngs, select, thaw
from dew.objectives.diffusion.block import BlockDiffusionObjective, _cache_geometry
from dew.objectives.lm.chunked import chunked_tile, model_logits
from dew.optim import scale_by_qk_clip, stepped
from dew.training import MeshSpec
from dew.training.transaction import example_fields

FIXTURE = Path(__file__).resolve().parent / "fixtures/hf/diffusion-gemma-sft"
REFERENCES = FIXTURE / "reference"


@pytest.fixture(scope="module")
def source():
    return loaded_source()


def loaded_source():
    """The SFT fixture's model, batch, the reference's own draws on a step, and its recorded values."""
    loaded = Pretrained.load(FIXTURE, dtype="float32", attention_impl="xla", max_seq_len=32)
    with np.load(FIXTURE / "reference.npz") as arrays:
        reference = {name: arrays[name] for name in arrays.files}
    batch = {"text": reference["tokens"], "canvas_mask": reference["canvas_mask"],
             "encoder_target_mask": reference["encoder_target_mask"]}
    # The step reads the reference's own draws by name (`Step.tape`): each row's corrupted canvas, the
    # canvas it scores and whether it conditions on its first pass, which row zero does and row one not.
    rows, length = reference["noisy"].shape[:2]
    selected = reference["selected_mask"][..., 0].reshape(rows, 2, length // 2).any(-1).argmax(-1)
    tape = {"noised input": jnp.asarray(reference["noisy"][..., 0]), "canvas": jnp.asarray(selected),
            "self conditioning": jnp.asarray([True, False])}
    step = Step(step=jnp.asarray(0, jnp.int32), key=jax.random.wrap_key_data(reference["step_key"]), ema=None,
                tape=tape)
    return loaded, batch, step, reference


def objective(loaded, *, variables=None, **kwargs):
    values = loaded.variables if variables is None else variables
    return BlockDiffusionObjective(loaded.model, prompt_length=4, num_canvases=2,
                                   variables=values, **kwargs)


def test_the_block_objectives_head_takes_the_fit_ladders_tile():
    """The block objective runs its head by the policy every token objective takes
    (`dew.objectives.lm.head`): it keeps the whole logits until the fit ladder tiles it, and its program
    reports the tile, which a resumed run restores."""
    loaded, *_ = loaded_source()
    built = objective(loaded)
    assert built.program_key()[0].head_tile is None
    assert built.tile_head() is not None and built.program_key()[0].head_tile == chunked_tile()
    resumed = objective(loaded)
    resumed.restore_head(chunked_tile(), fused=False)
    assert [module[1:] for module in resumed.program_key()] == [module[1:] for module in built.program_key()]


def reference_variables(loaded, name):
    values, _ = translate_weights(load_file(str(REFERENCES / name)), loaded.config)
    return objective(loaded, variables=values).init(jax.random.key(0))


def trainer_update(obj, batch, run_key, optimizer):
    """The trainer's first step under `optimizer` of a variables tree on one device, compiled once
    over batches of `batch`'s shapes (`stepped`, its stepped state), the step's draws
    (`example_fields`) and the trainer. One device holds the comparison to the trainer's composition
    of the update; its layouts are the parallelism matrix's."""
    trainer = Trainer(obj, optimizer, key=run_key, split=1)
    trainer.device_mesh = MeshSpec().build(jax.devices()[:1])
    initial = trainer.initial_state()
    step = Step(jnp.asarray(0, jnp.int32), jax.random.fold_in(initial.key, 0), None,
                **example_fields(initial, 0, 1, batch))
    run = trainer.compile(jax.tree.map(jnp.copy, initial), batch)

    def stepped(values, rows=batch):
        # The step consumes its state: it steps copies.
        state = jax.tree.map(jnp.copy, initial).replace(variables=jax.tree.map(jnp.copy, values))
        state, *_ = run(state, rows)
        return state

    return stepped, step, trainer


UPDATE_SEEDS = {"text": 8, "image": 16}
"""Check 1's declared seeds for the trainer's gradient of each SFT model, sized paired (`seeds_for` with
the trainer's step as `dew_fn`). The text model's, over drawn batches (`varied`), is the most of three
platforms': its two arms took bitwise the same gradient at every draw on JAX 0.11's and 0.12's CPU and
on an A100, and each sized it at 8 (run-tests sf-1546v-paired-main-c464b5df0 and
sf-1546v-paired-jax012-f65f56d57, node receipts SF-1546v-a100). Its error is heavy-tailed, alike in
both arms: four draws of 320 held at most 72% of a leaf's pooled error on 0.11, 87% on 0.12 and 86% on
the A100 (sf-1546v-tail-main-c464b5df0, sf-1546v-tail-jax012-f65f56d57), so the reference against
itself does not resolve within 4096 seeds and the paired count is the site's (ledger 1546)."""

UPDATE_ROWS = 8
"""Rows of each drawn text batch (`varied`)."""


def varied(batch, rows: int):
    """`batches(seed)` of `rows` rows of `batch`'s layouts, tiled, each real token (not the pad, 0)
    drawn afresh from the fixture's 32-token vocabulary at every seed, the masks the layouts' own: each
    draw's update then averages over conditioning of its own. Over the fixture's one batch, four
    weight draws of 320 carried 83 to 97% of each arm's pooled error at the attention leaves, the same
    draws in both arms, and its truth moved smoothly with the weights there, about 500 times their
    relative noise, as at the other draws (ledger 1546)."""
    tiled = jax.tree.map(lambda value: np.concatenate([np.asarray(value)] * (rows // len(value)), axis=0),
                         batch)

    def drawn(seed):
        tokens = np.asarray(tiled["text"])
        fresh = np.random.default_rng([seed, rows]).integers(1, 32, tokens.shape, dtype=tokens.dtype)
        return {**tiled, "text": np.where(tokens != 0, fresh, tokens)}

    return drawn


def assert_updates_as_the_objective(obj, twin, batch, data, run_key, variables, seeds: int, batches=None):
    """The trainer's update is the objective's at the step's draws. The gradient its step takes, which
    sgd with a momentum of zero keeps as its trace, is on par with the objective's own over drawn
    weights (check 1), and over drawn batches of `batch`'s shapes where `batches(seed)` draws them,
    `twin`, the objective over its model in float64, at the same draws the truth; they are two
    programs, which round otherwise, the objective's compiled with the step's compiler options
    (`as_a_step`). The gradients are compared as they are, not through an update applied to the
    weights, whose rounding at the weights' magnitude would swamp theirs; at each draw the step's
    weights are then sgd(1)'s rule of that trace to the bit, the weights less the trace in float32, so
    the update applies the gradient it holds. The run's own path, its loader, plan and marked rows,
    takes the trainer's own update on one device, to the bit: the same program on the same values."""
    drawn = (lambda seed: batch) if batches is None else batches
    stepped, step, trainer = trainer_update(obj, drawn(0), run_key, optax.sgd(1.0, momentum=0.0))
    weights = moved_weights(variables)

    def draw(seed):
        return (*weights(seed), drawn(seed))

    def trained(values, rows):
        state = stepped(values, rows)
        trace = jax.tree.map(np.asarray, optax.tree_utils.tree_get(state.opt_state, "trace"))
        rule = jax.tree.map(lambda before, gradient: (np.asarray(before) - gradient).astype(np.float32),
                            values["params"], trace)
        assert_tree_equal(jax.tree.map(np.asarray, state.variables["params"]), rule)
        return trace

    gradient = as_a_step(trainer, variables, drawn(0),
                         lambda values, rows: objective_gradient(obj, values, rows, step))
    assert_as_exact(trained, gradient, draw, "trainer update", seeds=seeds,
                    truth_fn=float64_gradient(twin, step))
    taken, _, _ = trainer_update(obj, batch, run_key, optax.sgd(0.001))
    solo = Trainer(obj, optax.sgd(0.001), key=run_key)
    solo.device_mesh = MeshSpec().build(jax.devices()[:1])
    assert_tree_equal(solo.fit(data, steps=1, log_every=1).variables["params"],
                      jax.tree.map(np.asarray, taken(variables).variables["params"]))


def assert_tree_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for (path, left), right in zip(jax.tree_util.tree_leaves_with_path(actual), jax.tree.leaves(expected),
                                   strict=True):
        np.testing.assert_array_equal(left, np.asarray(right), err_msg=jax.tree_util.keystr(path))


def assert_tree_close(actual, expected, tolerance):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for left, right in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
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
    variables = jax.tree.map(jnp.asarray, obj.init(jax.random.key(0)))
    (loss, aux), gradient = jax.value_and_grad(
        lambda values: obj.scalar_loss(values, batch, step), has_aux=True)(variables)
    np.testing.assert_allclose(loss, reference["loss"], atol=1e-5, rtol=0)
    np.testing.assert_allclose(aux.metrics["canvas_ce"], reference["canvas_loss"].mean(), atol=1e-5, rtol=0)
    np.testing.assert_allclose(aux.metrics["encoder_ce"], reference["encoder_loss"].mean(), atol=1e-5, rtol=0)
    expected = reference_variables(loaded, "gradient.safetensors")
    assert_tree_close(gradient, expected, 1e-4)
    updated = jax.tree.map(lambda value, grad: value - 0.001 * grad, variables, gradient)
    reference_update = reference_variables(loaded, "updated.safetensors")
    assert_tree_close(updated, reference_update, 2e-6)


def test_disabling_encoder_gradient_matches_the_reference_control(source):
    loaded, batch, step, _ = source
    obj = objective(loaded, stop_gradient_from_denoiser_to_encoder=True)
    gradient = jax.grad(lambda values: obj.scalar_loss(values, batch, step)[0])(
        jax.tree.map(jnp.asarray, obj.init(jax.random.key(0))))
    expected = reference_variables(loaded, "detached_gradient.safetensors")
    assert_tree_close(gradient, expected, 1e-4)
    full = reference_variables(loaded, "gradient.safetensors")
    assert (
        max(
            float(jnp.max(jnp.abs(a - b)))
            for a, b in zip(jax.tree.leaves(gradient), jax.tree.leaves(full), strict=True)
        )
        > 1e-3
    )


@pytest.mark.parametrize("probability,reference_key", [(0., "sc_off_loss"), (1., "sc_on_loss")])
def test_both_self_conditioning_branches_match_the_official_adapter(source, probability, reference_key):
    loaded, batch, step, reference = source
    obj = objective(loaded, self_cond_prob=probability)
    # Every row's self-conditioning is the probability's, not the reference's mixed draw.
    tape = {name: value for name, value in step.tape.items() if name != "self conditioning"}
    step = step.replace(tape=tape)
    loss, _ = obj.scalar_loss(jax.tree.map(jnp.asarray, obj.init(jax.random.key(0))), batch, step)
    np.testing.assert_allclose(loss, reference[reference_key], atol=1e-5, rtol=0)


def test_loss_weights_apply_after_independent_row_normalization(source):
    loaded, batch, step, reference = source
    obj = objective(loaded, encoder_loss_weight=0.3, decoder_loss_weight=2.7)
    loss, _ = obj.scalar_loss(jax.tree.map(jnp.asarray, obj.init(jax.random.key(0))), batch, step)
    expected = 0.3 * reference["encoder_loss"].mean() + 2.7 * reference["canvas_loss"].mean()
    np.testing.assert_allclose(loss, expected, atol=1e-5, rtol=0)
    canvas_mass = reference["selected_mask"].sum(axis=(1, 2))
    encoder_mass = reference["encoder_target_mask"].sum(axis=-1)
    wrong = (2.7 * np.average(reference["canvas_loss"], weights=canvas_mass)
             + 0.3 * np.average(reference["encoder_loss"], weights=encoder_mass))
    assert abs(float(loss) - wrong) > 1e-3


def test_scoring_reports_the_denoiser_cross_entropy_of_every_canvas_target(source):
    """A validation pass scores each row's selected canvas under the same draw
    training makes from the key, so the weights are the reference's selected
    targets and the per-row means are its canvas losses; `perplexity` over the
    pass is exp of the denoising loss per target."""
    loaded, batch, step, reference = source
    obj = objective(loaded)
    scored = obj.evaluate(jax.tree.map(jnp.asarray, obj.init(jax.random.key(0))), batch, step)
    weights = np.asarray(scored.weights)
    np.testing.assert_array_equal(weights, reference["selected_mask"][..., 0].astype(weights.dtype))
    row_means = (np.asarray(scored.losses) * weights).sum(-1) / weights.sum(-1)
    np.testing.assert_allclose(row_means, reference["canvas_loss"], atol=1e-5, rtol=0)


def test_sparse_canvas_targets_train_and_score_their_physical_canvas(source):
    loaded, _, step, _ = source
    # Another batch than the reference's: the step draws its own, not the reference's tape.
    step = step.replace(tape=None)
    obj = objective(loaded, canvas_size=2, encoder_loss_weight=0)
    variables = jax.tree.map(jnp.asarray, obj.init(jax.random.key(0)))
    tokens = jnp.tile(jnp.arange(1, 9, dtype=jnp.int32), (4, 1))
    mask = jnp.asarray([[0, 0, 1, 1], [0, 0, 0, 1], [1, 1, 1, 1], [0, 0, 0, 0]], bool)
    valid = jnp.ones_like(tokens, bool).at[2, 4:6].set(False)
    batch = {"text": ModelInputs(tokens, {"attention_mask": valid}), "canvas_mask": mask}
    scored = jax.jit(obj.evaluate)(variables, batch, step)
    expected = mask & valid[:, 4:]
    np.testing.assert_array_equal(scored.weights, expected)
    (loss, _), gradient = jax.jit(jax.value_and_grad(obj.scalar_loss, has_aux=True))(
        variables, batch, step)
    row_losses = (scored.losses * expected).sum(-1) / jnp.maximum(expected.sum(-1), 1)
    np.testing.assert_allclose(loss, row_losses.mean(), atol=1e-5, rtol=0)
    assert float(loss) > 0
    assert any(float(jnp.linalg.norm(leaf)) > 1e-6 for leaf in jax.tree.leaves(gradient))


def test_canvas_sampling_is_uniform_over_noncontiguous_partial_canvases(source):
    loaded, _, step, _ = source
    # Another batch than the reference's: the step draws its own, not the reference's tape.
    step = step.replace(tape=None)
    obj = BlockDiffusionObjective(loaded.model, prompt_length=4, num_canvases=4,
                                  canvas_size=2, variables=loaded.variables)
    variables = jax.tree.map(jnp.asarray, obj.init(jax.random.key(0)))
    mask = jnp.tile(jnp.asarray([0, 0, 1, 1, 0, 0, 0, 1], bool), (64, 1))
    batch = {"text": jnp.tile(jnp.arange(1, 13, dtype=jnp.int32), (64, 1)), "canvas_mask": mask}
    scored = jax.jit(obj.evaluate)(variables, batch, step)
    weights = np.asarray(scored.weights).reshape(64, 4, 2)
    selected = weights.any(axis=-1)
    np.testing.assert_array_equal(selected.sum(axis=-1), 1)
    assert not selected[:, [0, 2]].any()
    np.testing.assert_array_equal(weights[:, 1, 0], selected[:, 1])
    assert not weights[:, 3, 0].any()
    assert np.all((selected.sum(axis=0)[[1, 3]] >= 16) & (selected.sum(axis=0)[[1, 3]] <= 48))


def test_the_model_the_trainer_substitutes_is_the_one_the_losses_run(source):
    """The trainer rematerializes or quantizes a step by substituting the
    modules `program_key` names. The losses run the clone sized to the SFT
    sequence, so that is the module named, and a substitute changes what
    they score: one that caps the logits near zero moves both cross
    entropies to the uniform distribution's."""
    loaded, batch, step, _ = source
    obj = objective(loaded)
    variables = jax.tree.map(jnp.asarray, obj.init(jax.random.key(0)))
    before = float(obj.scalar_loss(variables, batch, step)[0])
    (program,) = obj.program_key()
    obj.substitute([program.module.clone(text=program.module.text.clone(final_logit_softcap=1e-6))])
    after = float(obj.scalar_loss(variables, batch, step)[0])
    assert after == pytest.approx(2 * math.log(program.module.vocab_size), abs=1e-4)
    assert abs(after - before) > 1e-3


def dataset(batch):
    rows = next(iter(batch.values())).shape[0]
    records = [{name: value[row] for name, value in batch.items()} for row in range(rows)]
    stream = grain.MapDataset.source(records).repeat().to_iter_dataset().batch(rows, drop_remainder=True)
    return Loader(train=lambda partition: iter(stream), val=None, records=rows, batch=rows)


def test_real_trainer_update_and_checkpoint_resume(source, tmp_path):
    loaded, original_batch, _, reference = source
    rows = math.lcm(2, jax.device_count())
    batch = jax.tree.map(lambda value: np.concatenate([value] * (rows // 2), axis=0), original_batch)
    obj = objective(loaded)
    run_key = jax.random.key(int(reference["run_seed"]))
    data = dataset(batch)
    variables = jax.tree.map(jnp.asarray, obj.init(jax.random.key(0)))
    wide = objective(Pretrained.load(FIXTURE, dtype="float64", attention_impl="xla", max_seq_len=32))
    assert_updates_as_the_objective(obj, wide, batch, data, run_key, variables, UPDATE_SEEDS["text"],
                                    batches=varied(batch, UPDATE_ROWS))
    trainer = Trainer(obj, optax.sgd(0.001), key=run_key, checkpoints=Checkpoints(str(tmp_path / "run")))
    updated = trainer.fit(data, steps=1, checkpoint_every=1, log_every=1)
    # Google's dense text-only model: transformers' DiffusionGemmaForBlockDiffusion
    # builds experts and a vision tower, so no published implementation reads
    # this layout, and the export is refused rather than written.
    with pytest.raises(ValueError):
        replace(loaded, model=obj.model).save(tmp_path / "published", variables=updated.variables)
    assert not (tmp_path / "published").exists()

    resumed = Trainer(obj, optax.sgd(0.001), key=run_key,
                      checkpoints=Checkpoints(str(tmp_path / "run"))).fit(
                          data, steps=2, checkpoint_every=1, log_every=1)
    direct = Trainer(obj, optax.sgd(0.001), key=run_key).fit(data, steps=2, log_every=1)
    assert_tree_close(resumed.variables, direct.variables, 0)
    assert int(resumed.updates) == 2


@pytest.fixture(scope="module")
def image_source():
    return loaded_image_source()


def loaded_image_source():
    """Native SFT behavior over the real loaded multimodal generation fixture."""
    directory = FIXTURE.parent / "diffusion-gemma-workflow"
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="xla", max_seq_len=32)
    with np.load(directory / "reference.npz") as reference:
        pixels = jnp.asarray(reference["pixels"])
    pixels = jnp.concatenate([pixels, pixels[..., ::-1]], axis=-1)
    pixels = jnp.concatenate([pixels, pixels[::-1]], axis=1)
    tokens = jnp.asarray([[0, 2, 60, 60, 5, 0, 6, 7, 8, 9, 0, 0],
                          [2, 60, 60, 60, 60, 6, 7, 8, 9, 10, 11, 12]])
    indices = jnp.asarray([[-1, -1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1],
                           [-1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1]])
    valid = jnp.asarray([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1] * 12], bool)
    positions = jnp.asarray([[0, 0, 1, 2, 4, 5, 7, 8, 9, 10, 10, 10],
                             [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12]])
    inputs = ModelInputs(tokens, {"image_indices": indices, "attention_mask": valid,
                                 "image_groups": jnp.where(indices >= 0, indices // 2, -1),
                                 "positions": positions},
                         {"pixel_values": pixels, "image_lengths": jnp.asarray([1, 2])})
    obj = BlockDiffusionObjective(loaded.model, prompt_length=8, variables=loaded.variables)
    variables = jax.tree.map(jnp.asarray, obj.init(jax.random.key(0)))
    step = Step(step=jnp.asarray(0, jnp.int32), key=jax.random.key(3), ema=None)
    return loaded, inputs, obj, variables, step


def test_model_inputs_text_matches_raw_loss_and_gradients(source):
    loaded, batch, step, _ = source
    obj = objective(loaded)
    variables = jax.tree.map(jnp.asarray, obj.init(jax.random.key(0)))
    evaluate = jax.jit(jax.value_and_grad(lambda values, data: obj.scalar_loss(values, data, step)[0]))
    raw = evaluate(variables, batch)
    prepared = evaluate(variables, {**batch, "text": ModelInputs(jnp.asarray(batch["text"]))})
    assert_tree_close(prepared, raw, 0)


def test_image_sft_uses_media_validity_groups_and_positions(image_source):
    _, inputs, obj, variables, step = image_source
    evaluate = jax.jit(jax.value_and_grad(
        lambda values, data: obj.scalar_loss(values, {"text": data}, step)[0], argnums=(0, 1),
        allow_int=True))
    loss, (gradient, input_gradient) = evaluate(variables, inputs)
    assert np.isfinite(loss)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(gradient))
    media_gradient = input_gradient.conditioning["pixel_values"]
    assert float(jnp.linalg.norm(media_gradient[0, 0])) > 1e-6
    assert float(jnp.linalg.norm(media_gradient[1])) > 1e-6
    np.testing.assert_array_equal(media_gradient[0, 1], 0)
    assert any(float(jnp.linalg.norm(leaf)) > 0
               for leaf in jax.tree.leaves(gradient["params"]["conditioner"]))
    changed = inputs.replace(
        conditioning={**inputs.conditioning, "pixel_values": -inputs.conditioning["pixel_values"]}
    )
    changed_loss = evaluate(variables, changed)[0]
    assert abs(float(changed_loss - loss)) > 1e-5
    for name, replacement in (("attention_mask", inputs.tokens != 0),
                              ("image_groups", jnp.full_like(inputs.tokens, -1)),
                              ("positions", jnp.broadcast_to(jnp.arange(12), inputs.tokens.shape))):
        changed = inputs.replace(token_fields={**inputs.token_fields, name: replacement})
        assert abs(float(evaluate(variables, changed)[0] - loss)) > 1e-6, name


def test_media_placeholders_never_become_text_targets(image_source):
    _, inputs, obj, variables, step = image_source
    indices = inputs.token_fields["image_indices"].at[0, 8].set(0)
    inputs = inputs.replace(token_fields={**inputs.token_fields, "image_indices": indices})
    media = indices >= 0
    weights = jnp.linspace(0.25, 1.25, 12)[None, :] * jnp.ones_like(inputs.tokens)
    evaluate = jax.jit(jax.value_and_grad(lambda values, data: obj.scalar_loss(
        values, {"text": data, "encoder_target_mask": weights}, step)[0]))
    in_vocab = inputs.replace(tokens=jnp.where(media, 60, inputs.tokens))
    out_of_vocab = inputs.replace(tokens=jnp.where(media, obj.model.vocab_size + 100, inputs.tokens))
    expected = evaluate(variables, in_vocab)
    actual = evaluate(variables, out_of_vocab)
    assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(actual))
    assert_tree_close(actual, expected, 0)



def test_denoiser_image_gradient_obeys_encoder_detachment(image_source):
    loaded, inputs, _, variables, step = image_source
    norms = []
    for detach in (False, True):
        obj = BlockDiffusionObjective(loaded.model, prompt_length=8, variables=loaded.variables,
                                      encoder_loss_weight=0,
                                      stop_gradient_from_denoiser_to_encoder=detach)
        def loss(pixels, *, obj=obj):
            conditioned = inputs.replace(conditioning={**inputs.conditioning, "pixel_values": pixels})
            return obj.scalar_loss(variables, {"text": conditioned}, step)[0]
        gradient = jax.jit(jax.grad(loss))(inputs.conditioning["pixel_values"])
        norms.append(float(jnp.linalg.norm(gradient)))
    assert norms[0] > 1e-6
    assert norms[1] == 0

def test_image_sft_trainer_resume_publish_and_generate(image_source, tmp_path):
    """Native lifecycle evidence, not an official image-conditioned SFT oracle."""
    loaded, inputs, obj, variables, _ = image_source
    rows = math.lcm(2, jax.device_count())
    batch = {"text": jax.tree.map(lambda leaf: np.concatenate([leaf] * (rows // 2)), inputs)}
    stream = grain.MapDataset.source([batch]).repeat().to_iter_dataset()
    data = Loader(train=lambda partition: iter(stream), val=None, records=rows, batch=rows)
    run_key = jax.random.key(2)
    wide = Pretrained.load(FIXTURE.parent / "diffusion-gemma-workflow", dtype="float64", attention_impl="xla",
                           max_seq_len=32)
    twin = BlockDiffusionObjective(wide.model, prompt_length=8, variables=wide.variables)
    assert_updates_as_the_objective(obj, twin, batch, data, run_key, variables, UPDATE_SEEDS["image"])
    checkpoints = Checkpoints(str(tmp_path / "run"))
    updated = Trainer(obj, optax.sgd(0.001), key=run_key, checkpoints=checkpoints).fit(
        data, steps=1, checkpoint_every=1, log_every=1)
    assert any(not np.array_equal(before, after) for before, after in zip(
        jax.tree.leaves(variables["params"]["conditioner"]),
        jax.tree.leaves(updated.variables["params"]["conditioner"]), strict=True))
    resumed = Trainer(obj, optax.sgd(0.001), key=run_key,
                      checkpoints=Checkpoints(str(tmp_path / "run"))).fit(
                          data, steps=2, checkpoint_every=1, log_every=1)
    direct = Trainer(obj, optax.sgd(0.001), key=run_key).fit(data, steps=2, log_every=1)
    assert_tree_close(resumed.variables, direct.variables, 0)
    assert_tree_close(resumed.opt_state, direct.opt_state, 0)
    assert int(resumed.updates) == 2

    replace(loaded, model=obj.model).save(tmp_path / "published", variables=resumed.variables)
    readback = Pretrained.load(tmp_path / "published", dtype="float32", attention_impl="xla", max_seq_len=32)
    restored = BlockDiffusionObjective(readback.model, prompt_length=8, variables=readback.variables)
    assert_tree_close(restored.init(jax.random.key(0)), resumed.variables, 0)
    # The published task's loss under the run's draws for the unrepeated rows.
    started = Trainer(obj, optax.sgd(0.001), key=run_key).initial_state()
    step = Step(step=jnp.asarray(0, jnp.int32), key=jax.random.fold_in(started.key, 0), ema=None,
                **example_fields(started, 0, 1, {"text": inputs}))
    original_loss = obj.scalar_loss(resumed.variables, {"text": inputs}, step)[0]
    restored_loss = restored.scalar_loss(restored.init(jax.random.key(0)), {"text": inputs}, step)[0]
    np.testing.assert_array_equal(restored_loss, original_loss)
    prompt = jax.tree.map(jnp.asarray, batch["text"]).slice_tokens(stop=8)
    key = jax.random.key(11)
    published_task = readback.block_generation()
    # The run's task stops where the published one does: at the checkpoint's end-of-sequence ids, which
    # the run's own settings do not carry yet (ledger 895).
    run_task = replace(obj.pipeline(resumed, ema=False, processor=loaded.processor),
                       eos_token_ids=published_task.eos_token_ids)
    generated = run_task(prompt, 4, key=key, process=published_task.process)
    published = published_task(prompt, 4, key=key)
    np.testing.assert_array_equal(generated.tokens, published.tokens)
    np.testing.assert_array_equal(generated.decoder_steps, published.decoder_steps)
    assert np.all(np.asarray(generated.decoder_steps) > 0)


def test_a_frozen_split_moves_only_what_the_filter_keeps(source):
    """Starting variables `freeze` split keep their split, as LMObjective's
    do: the frozen collection holds what the filter rejects, the loss is the
    loss of the whole tree, only the kept leaves take a gradient, and the
    published weights are one `params` collection again."""
    loaded, batch, step, reference = source

    def attention(path):
        return "self_attn" in path

    obj = objective(loaded, variables=freeze(loaded.variables, attention))
    variables = jax.tree.map(jnp.asarray, obj.init(jax.random.key(0)))
    assert set(variables) >= {"params", FROZEN}
    kept = [path for path, _ in jax.tree_util.tree_leaves_with_path(variables["params"])]
    assert kept and all("self_attn" in jax.tree_util.keystr(path) for path in kept)
    assert not any("self_attn" in jax.tree_util.keystr(path)
                   for path, _ in jax.tree_util.tree_leaves_with_path(variables[FROZEN]))
    (loss, _), gradient = jax.value_and_grad(
        lambda moving: obj.scalar_loss({**variables, "params": moving}, batch, step), has_aux=True)(
            variables["params"])
    np.testing.assert_allclose(loss, reference["loss"], atol=1e-5, rtol=0)
    whole = reference_variables(loaded, "gradient.safetensors")
    assert_tree_close(gradient, select(whole["params"], attention), 1e-4)
    trainer = Trainer(obj, optax.sgd(0.001), key=jax.random.key(int(reference["run_seed"])))
    state = trainer.fit(dataset(jax.tree.map(
        lambda value: np.concatenate([value] * (math.lcm(2, jax.device_count()) // 2), axis=0), batch)),
        steps=1, log_every=1)
    assert sorted(state.variables) == sorted(variables)
    assert "frozen" not in obj.pipeline(state, ema=False).variables


def test_a_canvas_rows_context_reads_an_id_its_vocabulary_also_pads_with(source):
    """A conversation's context may hold the id its rows pad with (id 0, Qwen's "!"): the canvas row
    marks its context's own ids (`ResponseCanvases`' prompt_mask), so block SFT reads that id as the
    context's, its loss and gradient those of the row whose validity says so outright."""
    from dew.data.chat import ROLES_KEY, Role
    from dew.data.steps.chat import ResponseCanvases

    loaded, _, step, _ = source
    step = step.replace(tape=None)
    obj = objective(loaded, canvas_size=2, encoder_loss_weight=0)
    variables = jax.tree.map(jnp.asarray, obj.init(jax.random.key(0)))
    roles = [Role.USER] * 3 + [Role.ASSISTANT] * 3
    rows = ResponseCanvases(prompt_length=4, canvas_size=2, count=2, eos_id=7, pad_id=0).flat_map(
        {"text": np.asarray([5, 0, 6, 8, 9, 3], np.int32), ROLES_KEY: np.asarray(roles, np.int8)})
    [row] = rows
    np.testing.assert_array_equal(row["prompt_mask"], [True, True, True, False])
    tokens = jnp.asarray(np.stack([row["text"]] * 2))
    marked = {"text": tokens, "canvas_mask": jnp.asarray(np.stack([row["canvas_mask"]] * 2)),
              "prompt_mask": jnp.asarray(np.stack([row["prompt_mask"]] * 2))}
    valid = jnp.concatenate([marked["prompt_mask"], marked["canvas_mask"]], axis=-1)
    stated = {"text": ModelInputs(tokens, {"attention_mask": valid}), "canvas_mask": marked["canvas_mask"]}
    loss = jax.jit(jax.value_and_grad(lambda values, batch: obj.scalar_loss(values, batch, step)[0]))
    for left, right in zip(jax.tree.leaves(loss(variables, marked)), jax.tree.leaves(loss(variables, stated)),
                           strict=True):
        np.testing.assert_array_equal(np.asarray(left), np.asarray(right))


def merged(passes):
    """The union by module path of several passes' `qk` trees, in NumPy: a path more than one pass sowed
    keeps the elementwise maximum of its sown values, and a path one pass sowed keeps what it sowed."""
    found = {}
    for tree in passes:
        for path, sown in flatten_dict(tree).items():
            found.setdefault(path, []).append(sown)
    return unflatten_dict({path: tuple(np.max(np.stack(column), axis=0) for column in zip(*sown, strict=True))
                           for path, sown in found.items()})


def peaks(tree):
    """The per-layer `[rows, heads]` maxima of a `qk` tree, past the layers' geometry scalars."""
    return [leaf for leaf in jax.tree.leaves(tree) if leaf.ndim == 2]


def sown_passes(obj, variables, batch, step):
    """The attention logit maxima of the encoder's pass and of both denoising passes, each computed by
    calling the training model on the step's own draws as `BlockDiffusionObjective` runs them: the encoder
    over the clean row, the first denoising pass with no self-conditioning signal, and the second with the
    first's logits where the row's draw conditions it. Returns the three `qk` trees, the encoder's first."""
    prepared, tokens, response, _, full_valid, canvas_mask, _ = obj._row(batch)
    noisy, selected, use_sc = obj._corrupted(response, canvas_mask, step)
    fields = prepared.token_fields
    positions, encoder_mask, encoder_keys, decoder_mask, decoder_keys = _cache_geometry(
        full_valid, selected, obj.prompt_length, obj.canvas_size, fields.get("positions"),
        fields.get("image_groups"))
    model, params = obj.training_model, thaw(variables)
    rngs = model_rngs(step.key, training=step.training)
    cache = model.apply(params, tokens.shape[0], method=model.init_cache, mutable=["cache"])[1]["cache"]
    encoder_kwargs = prepared.kwargs()
    encoder_kwargs.update(positions=positions, attention_mask=full_valid)
    encoder_states, encoded = model.apply(
        {**params, "cache": cache}, tokens, **encoder_kwargs, attention_pairwise_mask=encoder_mask,
        attention_key_positions=encoder_keys, method=model.encode, train=step.training, states=True,
        mutable=["cache", "qk"], rngs=rngs, capture_intermediates=False)

    def denoise(sc_logits):
        states, sown = model.apply(
            {**params, "cache": encoded["cache"]}, noisy, self_conditioning_logits=sc_logits,
            train=step.training, rngs=rngs, positions=positions[:, obj.prompt_length:],
            attention_pairwise_mask=decoder_mask, attention_key_positions=decoder_keys, states=True,
            mutable=["qk"])
        return states, sown["qk"]

    zero = jnp.zeros((*response.shape, model.vocab_size), at_least_fp32(encoder_states.dtype))
    first_states, first = denoise(zero)
    first_logits = jax.lax.stop_gradient(constrain(model_logits(
        model, params, jax.lax.stop_gradient(first_states)), LOGITS))
    _, second = denoise(jnp.where(use_sc[:, None, None], first_logits, zero))
    return encoded["qk"], first, second


def test_the_block_objective_observes_the_maxima_of_its_three_passes(source):
    """The fixture's model norms its queries and keys, so QK-Clip refuses it at any threshold (`dew.optim`).
    What the objective owes a clipping step is the observation: `Aux.qk_stats` merges the passes' own maxima
    by module path, which the same draws give when those passes are called directly."""
    loaded, batch, step, _ = source
    obj = objective(loaded)
    variables = jax.tree.map(jnp.asarray, obj.init(jax.random.key(0)))
    _, aux = obj.scalar_loss(variables, batch, step.replace(qk_stats=True))
    assert aux.qk_stats is not None
    expected = merged(sown_passes(obj, variables, batch, step))
    jax.tree.map(np.testing.assert_array_equal, jax.tree.map(np.asarray, aux.qk_stats),
                 jax.tree.map(np.asarray, expected))


def test_an_image_conditioned_run_observes_its_vision_tower_beside_its_passes(image_source):
    """An image's conditioner runs its vision tower in the encoder pass alone, and that tower's attention
    sows its maxima as the text layers' do. The merge keeps them by module path, beside the text layers'
    maxima, which the same draws give when the three passes are called directly."""
    _, inputs, obj, variables, step = image_source
    batch = {"text": inputs}
    _, aux = obj.scalar_loss(variables, batch, step.replace(qk_stats=True))
    assert any(path[0] == "conditioner" for path in flatten_dict(aux.qk_stats))
    expected = merged(sown_passes(obj, variables, batch, step))
    jax.tree.map(np.testing.assert_array_equal, jax.tree.map(np.asarray, aux.qk_stats), expected)


def unnormed_objective(**options):
    """A small DiffusionGemma whose queries and keys are not normed, so QK-Clip may rescale them."""
    text = CausalTransformer(vocab_size=16, emb_features=8, num_layers=2, num_heads=2, mlp_features=16,
                             max_seq_len=8, qk_norm=False, layer_scalar="trainable")
    return BlockDiffusionObjective(DiffusionGemma(text=text, canvas_length=4), prompt_length=4, **options)


def unnormed_image_objective():
    """A small image-conditioned DiffusionGemma with no query or key norm anywhere, so QK-Clip may rescale
    every projection it reads: its text layers and its vision tower's (`SiglipVision`'s layers, built with
    `qk_norm=False`). The fixture's own vision tower is Gemma's, and it norms its queries and keys."""
    text = CausalTransformer(vocab_size=32, emb_features=8, num_layers=2, num_heads=2, mlp_features=16,
                             max_seq_len=16, qk_norm=False, layer_scalar="trainable")
    vision = SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2, image_size=8,
                          patch_size=4)
    projection = GemmaProjector(text_width=8, patches_per_side=2, tokens_per_side=1)
    model = DiffusionGemma(text=text, canvas_length=4, conditioner=VisionConditioner(
        vision, projection, dtype=jnp.float32))
    return BlockDiffusionObjective(model, prompt_length=8)


def unnormed_image_batch(rows):
    """`rows` rows of an image batch over the unnormed model's vocabulary: one image a row, and each row's
    text slots 2 and 3 its image's soft tokens."""
    tokens = jnp.asarray([[0, 2, 9, 9, 5, 0, 6, 7, 8, 9, 0, 0]] * rows, jnp.int32)
    indices = jnp.asarray([[-1, -1, 0, 1, -1, -1, -1, -1, -1, -1, -1, -1]] * rows, jnp.int32)
    valid = jnp.asarray([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]] * rows, bool)
    positions = jnp.asarray([[0, 0, 1, 2, 4, 5, 7, 8, 9, 10, 10, 10]] * rows, jnp.int32)
    pixels = jnp.broadcast_to(jnp.linspace(-0.5, 0.5, 3 * 8 * 8).reshape(1, 1, 3, 8, 8), (rows, 1, 3, 8, 8))
    return {"text": ModelInputs(tokens, {"image_indices": indices, "attention_mask": valid,
                                         "image_groups": jnp.where(indices >= 0, indices // 2, -1),
                                         "positions": positions},
                                {"pixel_values": pixels, "image_lengths": jnp.ones((rows,), jnp.int32)})}


def qk_clip(limit):
    """sgd(0) then `limit`'s QK-Clip: the step's own update is zero, so what it moves is the clip alone."""
    return optax.chain(optax.sgd(0.0), scale_by_qk_clip(limit))


def clipped_run(obj, batch, limit):
    """One step of `obj` over `batch` under `qk_clip(limit)`: the starting variables and optimizer state,
    read before the step consumes them, the passes' `qk` trees at that step's draws, and the stepped
    variables with the updates committed."""
    train = Trainer(obj, qk_clip(limit), key=jax.random.PRNGKey(3), split=1)
    initial = train.initial_state()
    step = Step(jnp.asarray(0, jnp.int32), jax.random.fold_in(initial.key, 0), None,
                **example_fields(initial, 0, 1, batch))
    start = jax.tree.map(np.asarray, initial.variables)
    opt_state = jax.tree.map(np.asarray, initial.opt_state)
    passes = jax.jit(sown_passes, static_argnums=(0,))(obj, jax.tree.map(jnp.asarray, start), batch, step)
    final, *_ = train.compile(initial, batch)(initial, batch)
    return start, opt_state, passes, jax.tree.map(np.asarray, final.variables), int(final.updates)


def assert_clipped_at_the_maxima(start, opt_state, passes, final, limit):
    """`final`'s parameters are `qk_clip(limit)`'s update at the passes' merged maxima, applied to `start` with
    its rescale (`stepped`) bit for bit, and the clip moved some of them. Returns the maxima it clipped at."""
    maxima = jax.tree.map(np.asarray, merged(passes))
    zero = jax.tree.map(np.zeros_like, start["params"])
    applied, after = jax.jit(qk_clip(limit).update)(zero, opt_state, start["params"], qk_stats=maxima)
    expected = jax.tree.map(np.asarray, stepped(start["params"], applied, opt_state, after))
    jax.tree.map(np.testing.assert_array_equal, final["params"], expected)
    assert any(not np.array_equal(after, before) for after, before in
               zip(jax.tree.leaves(final["params"]), jax.tree.leaves(start["params"]), strict=True))
    return maxima


def test_a_block_run_clips_its_attention_at_its_passes_maxima():
    """An optimizer holding a QK-Clip has the run ask the block objective for its maxima, and its update is
    `scale_by_qk_clip`'s at the maxima of its three passes, bit for bit (its own update is zero). Every
    row conditions the second denoising pass, so it differs from the first, and the first's maxima must
    be the strongest somewhere for the merge to matter. A threshold no head reaches moves nothing."""
    threshold = 0.05
    obj = unnormed_objective(self_cond_prob=1.0)
    rows = 8 * jax.device_count()
    batch = {"text": jnp.asarray(np.random.default_rng(5).integers(1, 16, size=(rows, 8)), jnp.int32)}
    start, opt_state, passes, final, updates = clipped_run(obj, batch, threshold)
    assert updates == 1
    encoder, first, second = passes
    maxima = jax.tree.map(np.asarray, merged(passes))
    assert max(float(np.max(peak)) for peak in peaks(maxima)) > threshold
    assert any(np.any(own > np.maximum(before, after)) for own, before, after in
               zip(peaks(first), peaks(encoder), peaks(second), strict=True))
    assert_clipped_at_the_maxima(start, opt_state, passes, final, threshold)
    _, _, _, unclipped, _ = clipped_run(obj, batch, 1e30)
    jax.tree.map(np.testing.assert_array_equal, unclipped["params"], start["params"])


def test_an_image_conditioned_run_clips_its_text_and_vision_attention_at_their_maxima():
    """The image run clips where its projections carry no norm: its text layers and its vision tower alike,
    each at its merged maxima (`merged`), where the vision tower's attention is what only the encoder pass
    observes. A threshold no head reaches moves nothing."""
    obj = unnormed_image_objective()
    rows = math.lcm(2, jax.device_count())
    batch = unnormed_image_batch(rows)
    threshold = 0.05
    start, opt_state, passes, final, updates = clipped_run(obj, batch, threshold)
    assert updates == 1
    maxima = assert_clipped_at_the_maxima(start, opt_state, passes, final, threshold)
    assert max(float(np.max(peak)) for peak in peaks(maxima)) > threshold
    assert any(path[0] == "conditioner" for path in flatten_dict(maxima))
    _, _, _, unclipped, _ = clipped_run(obj, batch, 1e30)
    jax.tree.map(np.testing.assert_array_equal, unclipped["params"], start["params"])
