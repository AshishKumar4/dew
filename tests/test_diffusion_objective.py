"""The diffusion objective on the general trainer.

The loss is checked against a hand computation from the process's own parts,
the frozen encoder's weights are shown to reach the compiled step as an
argument and not as a constant, evaluation produces the typed artifact from
the step's key."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import Replaying
from diffusion_stubs import RES, StubText
from diffusion_support import (
    applied,
    as_a_step,
    float32_update,
    float64_gradient,
    moved_weights,
    objective_gradient,
)
from flax import linen as nn
from reference_error import as_given

from dew.data.dataset import Loader
from dew.diffusion import broadcast_rates, expand, presets
from dew.inputs import CharTable, Condition, Field, InputSpec, unit_range
from dew.nn.backbones import SimpleDiT, SimpleMMDiT
from dew.objectives.base import Recorder, Step
from dew.objectives.diffusion import DiffusionObjective
from dew.sampling import CFG, Euler
from dew.training import Trainer

_DEFAULT_MAKE_OBJECTIVE_GUIDANCE = CFG(2.0)


def make_objective(*, guidance: CFG | None = _DEFAULT_MAKE_OBJECTIVE_GUIDANCE, **fields):
    model = SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1)
    inputs = InputSpec(Field("image", (RES, RES, 3)),
                       {"textcontext": Condition(StubText.from_pretrained("stub"))})
    return DiffusionObjective(model, presets.EDM(P_mean=-0.4, P_std=1.0), inputs, steps=3, guidance=guidance,
                              solver=Euler(), **fields)


def make_batch(count=8):
    images = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, :, None, None],
                     (count, 1, RES, 3)).astype(np.uint8)
    encoder = StubText.from_pretrained("stub")
    return {"image": images,
            "text": encoder.tokenize(["a bird", "cat", "", "two dogs", "x", "y", "zz", "w"][:count])}


def make_batch_of(count: int):
    """`count` rows of drawn images and captions, as `make_batch` makes eight."""
    images = np.random.default_rng(count).integers(0, 256, (count, RES, RES, 3), dtype=np.uint8)
    captions = ["a bird", "cat", "", "two dogs", "x", "y", "zz", "w"]
    return {"image": images,
            "text": StubText.from_pretrained("stub").tokenize([captions[row % 8] for row in range(count)])}


def test_the_steps_blank_is_its_own_towers_encoding_of_the_configured_prompt():
    """The unconditional branch a step reads is the configured prompt encoded by the towers in the
    step's own variables, cast to the conditional branch's dtypes: move the towers and it moves."""
    objective = make_objective()
    params = objective.init(jax.random.PRNGKey(0))
    encoder = objective.inputs.conditions["textcontext"].encoder
    tokens = encoder.tokenize([""])
    given = {"textcontext": encoder.encode(encoder.params, encoder.tokenize(["a bird"]))}
    blank = jax.jit(objective.blank_conditions)(params, given)
    expected = encoder.encode(params["encoders"]["textcontext"], tokens)
    for got, want in zip(jax.tree.leaves(blank["textcontext"]), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(got, want)
    moved = {**params, "encoders": jax.tree.map(lambda leaf: leaf + 1.0, params["encoders"])}
    assert not np.array_equal(jax.tree.leaves(jax.jit(objective.blank_conditions)(moved, given))[0],
                              jax.tree.leaves(blank)[0])


class Zero(nn.Module):
    """A model that outputs zero, so the loss is a closed form of the process."""

    @nn.compact
    def __call__(self, x, temb, textcontext=None, train=False):
        return jnp.zeros_like(x) * self.param("w", nn.initializers.ones, ())


def test_compiled_samples_follow_the_substituted_denoiser():
    class Velocity(nn.Module):
        value: float

        @nn.compact
        def __call__(self, x, temb, train=False):
            return jnp.full_like(x, self.value) * self.param("w", nn.initializers.ones, ())

    inputs = InputSpec(Field("image", (2, 2, 1)))
    objective = DiffusionObjective(Velocity(0.), presets.Flow(), inputs, guidance=None, steps=3)
    variables = objective.init(jax.random.key(0))
    batch = {"image": jnp.zeros((2, 2, 2, 1))}
    step = Step(jnp.array(0), jax.random.key(1), None)
    before = objective.evaluate(variables, batch, step).images
    replacement = Velocity(.5)
    objective.substitute([replacement])
    after = objective.evaluate(variables, batch, step).images
    fresh = DiffusionObjective(replacement, presets.Flow(), inputs, guidance=None, steps=3)
    expected = fresh.evaluate(variables, batch, step).images
    assert not np.allclose(before, expected)
    np.testing.assert_array_equal(after, expected)


def test_compiled_samples_follow_the_autoencoder_they_decode_with():
    """The sample program runs the autoencoder beside the denoiser, whose scale it bakes in, so it
    keys on the autoencoder as on the denoiser (`sampling_modules`): an objective given another
    autoencoder over the same network, scaled apart, decodes its samples with that one, as an
    objective built over it does."""
    from test_autoencoder import IMAGE_SIZE, tiny_vae

    from dew.nn.autoencoders.sd_vae import StableDiffusionVAE

    first = tiny_vae()
    scaled = StableDiffusionVAE(model=first.model, params=first.params, dtype=jnp.float32, latent_scale=2.0)
    assert scaled.model is first.model
    inputs = InputSpec(Field("image", (IMAGE_SIZE, IMAGE_SIZE, 3)))

    def built(autoencoder):
        return DiffusionObjective(Zero(), presets.Flow(), inputs, autoencoder=autoencoder, guidance=None,
                                  steps=2)

    objective = built(first)
    variables = objective.init(jax.random.key(0))
    batch = {"image": jnp.zeros((2, IMAGE_SIZE, IMAGE_SIZE, 3))}
    step = Step(jnp.array(0), jax.random.key(1), None)
    before = objective.evaluate(variables, batch, step).images
    objective.autoencoder = scaled
    after = objective.evaluate(variables, batch, step).images
    expected = built(scaled).evaluate(variables, batch, step).images
    assert not np.allclose(before, expected)
    np.testing.assert_array_equal(after, expected)


def test_a_preset_builds_the_same_loss_and_images_as_its_process():
    from dew.inference import TextToImage

    preset = presets.Flow(shift=2.5, logit_mean=-0.3, min_snr_gamma=4)
    inputs = InputSpec(Field("image", (2, 2, 1)))
    explicit = DiffusionObjective(Zero(), preset(), inputs, guidance=None, steps=3)
    objective = DiffusionObjective(Zero(), preset, inputs, guidance=None, steps=3)
    variables = objective.init(jax.random.key(0))
    batch = {"image": np.arange(8, dtype=np.uint8).reshape(2, 2, 2, 1)}
    step = Step(step=jnp.asarray(0), key=jax.random.key(1), ema=None)
    np.testing.assert_array_equal(objective.scalar_loss(variables, batch, step)[0],
                                  explicit.scalar_loss(variables, batch, step)[0])
    direct = TextToImage.from_objective(objective, variables)
    built = TextToImage.from_objective(explicit, variables)
    np.testing.assert_array_equal(direct(["", ""], key=2).host().images,
                                  built(["", ""], key=2).host().images)


def test_what_an_objective_cannot_train_or_sample_is_refused_at_construction(tmp_path):
    """A masked process; a solver given by name, one that refuses the schedule
    (the sigma integrators hold only where alpha is 1, which the VP cosine is
    not), or one without the real terminal grid's target; and a model that
    cannot run on the inputs (SimpleMMDiT runs its text as a second stream
    through every block, so it declares `RequiresText`): each is refused when
    the objective is built, before a step and before its alignment encoder's
    weights load. The encoder's source is a directory holding no checkpoint,
    whose read raises OSError, so a read before the refusal would raise that
    instead."""
    from dew.diffusion.discrete import MDLM
    from dew.objectives.diffusion import Alignment
    from dew.sampling import RK4, UniPC

    alignment = Alignment(source=str(tmp_path))
    with pytest.raises(OSError):
        alignment.network(None)

    flow, unconditional = presets.Flow(), InputSpec(Field("image", (RES, RES, 3)))
    mmdit = SimpleMMDiT(patch_size=2, emb_features=8, num_layers=1, num_heads=2)
    unipc = UniPC(3, lower_order_final=False)
    for error, _fragment, model, process, settings in (
            (ValueError, "MDLM.*--objective masked_diffusion", Zero(), MDLM(mask_id=1), {}),
            (ValueError, "GeneralizedNoiseScheduler", Zero(), presets.Cosine(), {"solver": RK4()}),
            (TypeError, "names a solver", Zero(), flow, {"solver": "euler"}),
            (ValueError, "sigma=0 target", Zero(), flow, {"solver": unipc, "steps": 7}),
            (ValueError, "unconditionally", mmdit, flow, {})):
        with pytest.raises(error):
            DiffusionObjective(model, process, unconditional, alignment=alignment, **settings)
    DiffusionObjective(Zero(), presets.Karras(), unconditional, solver=RK4())


@pytest.mark.parametrize("order, steps", [(2, 5), (3, 7)])
def test_singlestep_solver_generates_through_the_objective(order, steps):
    from dew.diffusion import DirectPredictionTransform, LinearNoiseScheduler, Process
    from dew.sampling import DPMSolverSinglestep

    process = Process(LinearNoiseScheduler(1000), DirectPredictionTransform())
    objective = DiffusionObjective(Zero(), process, InputSpec(Field("image", (2, 2, 1))),
                                   solver=DPMSolverSinglestep(order), steps=steps, guidance=None)
    params = objective.init(jax.random.key(1))
    result = objective.evaluate(params, {"image": np.zeros((2, 2, 2, 1), np.uint8)},
                                Step(step=jnp.asarray(0), key=jax.random.key(2), ema=None))
    np.testing.assert_array_equal(result.images, np.zeros((2, 2, 2, 1), np.float32))


def test_loss_is_the_weighted_error_of_the_prediction():
    """With a zero output, the Karras parameterization predicts x_0 as
    c_skip x_t, the target is x_0, and the loss is the EDM lambda weighted
    mean of the l2 error, at the t and the noise the step reads (`Step.tape`)."""
    process = presets.EDM(regime="pixel")()
    inputs = InputSpec(Field("image", (RES, RES, 3)))
    objective = DiffusionObjective(Zero(), process, inputs)
    params = objective.init(jax.random.PRNGKey(0))
    batch = make_batch()
    x0 = unit_range(batch["image"])
    time_key, noise_key = jax.random.split(jax.random.PRNGKey(7))
    t = process.schedule.sample_t(time_key, 8)
    noise = jax.random.normal(noise_key, x0.shape)
    step = Step(step=jnp.asarray(3), key=jax.random.PRNGKey(7), ema=None, tape={"times": t, "noise": noise})

    loss, aux = objective.scalar_loss(params, batch, step)

    rates = broadcast_rates(process.schedule, t, x0)
    x_t = rates[0] * x0 + rates[1] * noise
    predicted = process.prediction.pred_transform(x_t, jnp.zeros_like(x_t), rates, t)
    expected = jnp.mean(expand(process.weight(t), x0) * optax.l2_loss(predicted, x0))
    assert float(loss) == pytest.approx(float(expected), rel=1e-6)
    assert aux.metrics == {}


def test_a_checkpoint_of_this_state_resumes_in_place(tmp_path):
    """The state carries no derived leaf, so a run resumes from its own
    checkpoint directory through the trainer, restoring into the template
    `init` describes and taking the next step."""
    from dew.training import Checkpoints

    objective = make_objective()
    batch = make_batch()

    def trainer():
        return Trainer(make_objective(), optax.adam(1e-3), key=jax.random.PRNGKey(0),
                       checkpoints=Checkpoints(str(tmp_path), keep=1))

    data = Loader(train=lambda partition: Replaying(lambda position: batch), val=None, records=None, batch=8)
    first = trainer()
    first.fit(data, steps=1, log_every=100, checkpoint_every=1)
    assert first.checkpoints is not None
    first.checkpoints.wait()

    resumed = trainer().fit(data, steps=2, log_every=100)

    assert int(resumed.step) == 2
    assert set(resumed.variables["encoders"]) == set(objective.inputs.conditions)
    # the frozen encoder came through untouched
    assert jnp.array_equal(resumed.variables["encoders"]["textcontext"]["table"],
                           objective.inputs.conditions["textcontext"].encoder.params["table"])
    leaves = jax.tree.leaves(resumed.variables["params"])
    assert leaves and all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)


def test_validation_samples_follow_the_step_key():
    """Successive validations draw fresh noise while a given key reproduces,
    and the EMA weights are what evaluate samples with when the step has them."""
    objective = make_objective(guidance=None)
    params = objective.init(jax.random.PRNGKey(0))
    batch = make_batch()
    first = Step(step=jnp.asarray(5), key=jax.random.PRNGKey(2), ema=None)
    again = Step(step=jnp.asarray(5), key=jax.random.PRNGKey(2), ema=None)
    later = Step(step=jnp.asarray(6), key=jax.random.PRNGKey(3), ema=None)

    images = objective.evaluate(params, batch, first).images
    assert jnp.array_equal(objective.evaluate(params, batch, again).images, images)
    assert not jnp.allclose(objective.evaluate(params, batch, later).images, images)

    averaged = jax.tree.map(lambda leaf: leaf + 0.1, params)
    with_ema = Step(step=jnp.asarray(5), key=jax.random.PRNGKey(2), ema=averaged)
    assert not jnp.allclose(objective.evaluate(params, batch, with_ema).images, images)
    assert jnp.array_equal(objective.evaluate(params, batch, with_ema).images,
                           objective.evaluate(averaged, batch, first).images)


@pytest.fixture(scope="module")
def conditional_mmdit():
    encoder = CharTable.from_pretrained(tokens=3, features=6, vocab=16)
    inputs = InputSpec(Field("image", (4, 4, 1)), {"textcontext": Condition(encoder)})
    model = SimpleMMDiT(output_channels=1, patch_size=2, emb_features=8,
                               num_layers=1, num_heads=2, mlp_ratio=2, attention_impl="xla")
    preset = presets.EDM(sigma_max=1.0, regime="pixel")
    objective = DiffusionObjective(model, preset, inputs, steps=3, solver=Euler(), guidance=CFG(2.0))
    variables = objective.init(jax.random.key(0))
    # The initialized zero output head otherwise hides conditioning gradients.
    variables = {**variables, "params": jax.tree.map(lambda leaf: leaf + 0.02, variables["params"])}
    table = variables["encoders"]["textcontext"]["table"].at[1].add(jnp.linspace(-1, 1, 6))
    variables = {**variables, "encoders": {"textcontext": {"table": table}}}
    # The objective encodes the unconditional branch from the weights it is
    # built over, so it is rebuilt over the ones these tests sample under.
    objective = DiffusionObjective(model, preset, inputs, steps=3, solver=Euler(),
                                   guidance=CFG(2.0), variables=variables)
    batch = {"image": np.arange(64, dtype=np.uint8).reshape(4, 4, 4, 1) * 3,
             **inputs.tokenize(["ab", "cd", "ef", "gh"])}
    step = Step(jnp.asarray(0), jax.random.key(4), None)
    return objective, variables, batch, step


NULL_DROPOUT_SEEDS = {0.5: 8, 1.0: 8}
"""Check 1's declared seeds for the dropped route at each dropout probability, sized paired
(`seeds_for` with the dropped route as `dew_fn`, 2026-10-23 on armada): 8 at each."""

NULL_DROPOUT_ROWS = 32
"""The rows of each null-dropout draw: each weight's gradient sums over all of them."""


@pytest.mark.parametrize("probability", [0.5, 1.0])
def test_null_dropout_matches_explicit_tokens_under_current_encoder(conditional_mmdit, probability):
    """An objective that drops a row's condition, which its step records, trains as one given that
    row's tokens blanked explicitly: it drops every row at probability one and some at a half, the
    explicit route's tower takes a gradient, and the loss and every trained weight's gradient of the
    dropped route are on par from float64 with the explicit route's (check 1). Each draw is its own
    rows, images and prompts, its weights moved from the fixture's, and its step's key, which draws
    the times, the noise and the dropped rows. The two routes are two programs, which round
    otherwise."""
    from reference_error import as_given, assert_as_exact, widened, without_ynn

    source, variables, _, step = conditional_mmdit
    dropped = DiffusionObjective(source.model, source.process, source.inputs,
                                 unconditional_prob=probability, steps=3, solver=Euler(),
                                 variables=variables)
    conditional = DiffusionObjective(source.model, source.process, source.inputs,
                                     unconditional_prob=0.0, steps=3, solver=Euler(),
                                     variables=variables)
    blank = source.inputs.tokenize([""] * NULL_DROPOUT_ROWS)
    host = jax.tree.map(np.asarray, variables)

    def stepped(seed):
        return replace(step, key=jax.random.key(int(seed)))

    def masked(values, rows, at):
        """The rows the dropped objective's step drops."""
        recorder = Recorder()
        dropped.scalar_loss(values, rows, replace(at, recorder=recorder))
        (_, mask), = recorder.values["condition dropout"]
        return np.asarray(mask)

    def explicit(rows, mask):
        """`rows` with the dropped rows' tokens blanked, which the explicit route reads."""
        return {"image": rows["image"], "text": jax.tree.map(
            lambda empty, given: np.where(mask[:, None], empty, given), blank["text"], rows["text"])}

    def draw(seed):
        """A seed's weights, each trained leaf moved by a draw so every layer takes a gradient, its rows'
        images and prompts, and the seed, its step's key."""
        rng = np.random.default_rng([0xD20, seed])
        params = jax.tree.map(lambda leaf: (leaf + 0.02 * rng.standard_normal(leaf.shape)).astype(leaf.dtype),
                              host["params"])
        prompts = ["".join(rng.choice(list("abcdefgh"), 2)) for _ in range(NULL_DROPOUT_ROWS)]
        rows = {"image": rng.integers(0, 256, (NULL_DROPOUT_ROWS, 4, 4, 1), dtype=np.uint8),
                **jax.tree.map(np.asarray, source.inputs.tokenize(prompts))}
        return {**host, "params": params}, rows, np.int64(seed)

    # Over the seeds, every row is dropped at probability one, and at a half some are and some not.
    masks = np.stack([masked(*draw(seed)[:2], stepped(seed)) for seed in range(4)])
    assert masks.all() if probability == 1.0 else (masks.any() and not masks.all())
    values, rows, _ = draw(0)
    tokens = explicit(rows, masked(values, rows, stepped(0)))
    given = jax.jit(jax.grad(lambda values: conditional.scalar_loss(values, tokens, stepped(0))[0]))
    assert float(jnp.linalg.norm(given(values)["encoders"]["textcontext"]["table"])) > 1e-6

    def trained(loss_and_grads):
        loss, grads = loss_and_grads
        return np.asarray(loss), jax.tree.map(np.asarray, grads["params"])

    def through(objective, values, rows, at):
        return jax.value_and_grad(lambda values: objective.scalar_loss(values, rows, at)[0])(values)

    # Both arms are Dew's, so both compile without YNNPACK's fusions (`without_ynn`).
    through_dropped = without_ynn(lambda values, rows, at: through(dropped, values, rows, at))
    through_explicit = without_ynn(lambda values, rows, at: through(conditional, values, rows, at))

    def dew(values, rows, seed):
        return trained(through_dropped(values, rows, stepped(seed)))

    def reference(values, rows, seed):
        at = stepped(seed)
        return trained(through_explicit(values, explicit(rows, masked(values, rows, at)), at))

    def truth(values, rows, seed):
        at = stepped(seed)
        tokens = explicit(rows, masked(jax.tree.map(lambda leaf: np.asarray(leaf, np.float32)
                                                    if np.issubdtype(np.asarray(leaf).dtype, np.floating)
                                                    else leaf, values), rows, at))
        tokens = float64_pixels(tokens)
        with jax.enable_x64(new_val=True):
            wide = DiffusionObjective(source.model.clone(dtype=jnp.float64), source.process, source.inputs,
                                      unconditional_prob=0.0, steps=3, solver=Euler(),
                                      variables=widened(variables))
            return trained(through(wide, widened(values), tokens, at))

    assert_as_exact(as_given(dew), as_given(reference), draw, f"null dropout at {probability}",
                    seeds=NULL_DROPOUT_SEEDS[probability], truth_fn=truth)


def test_a_dropped_row_is_conditioned_on_the_steps_own_towers(conditional_mmdit):
    """A fully dropped batch reads the blank its step's towers encode: move the towers in the step's
    variables and the loss moves with them, as a resumed tree's would."""
    source, variables, batch, step = conditional_mmdit
    dropped = DiffusionObjective(source.model, source.process, source.inputs,
                                 unconditional_prob=1.0, steps=3, solver=Euler(),
                                 variables=variables)
    before = float(dropped.scalar_loss(variables, batch, step)[0])
    moved = {**variables, "encoders": jax.tree.map(
        lambda leaf: leaf + 1.0 if np.issubdtype(leaf.dtype, np.floating) else leaf, variables["encoders"])}

    assert float(dropped.scalar_loss(moved, batch, step)[0]) != pytest.approx(before)


def test_guided_samples_use_bound_encoder_not_constructor_weights(conditional_mmdit):
    objective, variables, batch, step = conditional_mmdit
    condition = objective.inputs.conditions["textcontext"]
    rebound = replace(condition.encoder, params=variables["encoders"]["textcontext"])
    inputs = replace(objective.inputs, conditions={"textcontext": replace(condition, encoder=rebound)})
    reconstructed = DiffusionObjective(objective.model, objective.process, inputs,
                                       steps=3, solver=Euler(), guidance=CFG(2.0))
    expected = reconstructed.evaluate(variables, batch, step).images
    actual = objective.evaluate(variables, batch, step).images
    np.testing.assert_array_equal(actual, expected)


def test_each_example_draws_its_own_times_noise_and_dropout_on_every_layout():
    """A step's draws are each example's own (`Step.sample`): the batch whole on one device, its two
    halves pooled in an accumulation window, and the batch read in two microbatches give every
    example the same times, noise and dropped condition, and so the same noised input, target and
    loss weight, bitwise, as the transaction places each microbatch's rows among the update's
    examples (`example_places`). The model's dropout stays the attempt's (`Step.key`)."""
    from dew._distributions import counter_key
    from dew.objectives.base import Recorder, stream
    from dew.training.distributed import microbatch
    from dew.training.transaction import example_places

    objective = make_objective(unconditional_prob=0.5)
    params = objective.init(jax.random.PRNGKey(0))
    batch = make_batch(8)
    update = jax.random.key(7)

    def taped(parts):
        """The tape of the update's eight examples when the loss reads `parts`, (slot, split, rows) each."""
        recorder = Recorder()
        for slot, split, rows in parts:
            index = example_places(slot, split, len(jax.tree.leaves(rows)[0]))
            step = Step(jnp.asarray(0), jax.random.key(slot), None, example_key=update,
                        example_index=index, recorder=recorder)
            objective.loss(params, rows, step)
        return recorder.tape(8)

    halves = [jax.tree.map(lambda value, rows=rows: value[rows], batch)
              for rows in (slice(0, 4), slice(4, 8))]
    microbatches = [microbatch(batch, jnp.asarray(part), 2) for part in range(2)]
    whole = taped([(0, 1, batch)])
    pooled = taped([(slot, 1, half) for slot, half in enumerate(halves)])
    split = taped([(part, 2, rows) for part, rows in enumerate(microbatches)])
    assert set(whole) == {"condition dropout", "times", "noise", "noised input", "target", "loss weights"}
    for name in whole:
        for layout in (pooled, split):
            np.testing.assert_array_equal(np.asarray(layout[name]), np.asarray(whole[name]), err_msg=name)
    # Both halves of the dropout draw occur: each example's draw is its own, not one for the batch.
    assert 0 < int(np.sum(np.asarray(whole["condition dropout"]))) < 8
    assert len({float(np.asarray(whole["times"])[row]) for row in range(8)}) == 8
    # An example's noise is its documented draw: the update's key folded with the draw's name, then the
    # example's place.
    third = jax.random.fold_in(jax.random.fold_in(counter_key(update), stream("noise")), 3)
    np.testing.assert_array_equal(np.asarray(whole["noise"])[3],
                                  np.asarray(jax.random.normal(third, (RES, RES, 3), jnp.float32)))


def float64_pixels(batch):
    """`batch` with its pixels in float64, so a float64 truth's loss normalizes them (`unit_range`) and
    runs its rates and noising in float64; the step's draws stay its float32 ones."""
    return {**batch, "image": np.asarray(batch["image"], np.float64)}


def float64_twin(objective):
    """`objective` over its model in float64: the truth of its float32 step at the same draws."""
    import copy

    twin = copy.copy(objective)
    twin.substitute([objective.model.clone(dtype=jnp.float64)])
    return twin


POOLED_SEEDS = 16
"""Check 1's declared seeds for the pooled update, sized by `seeds_for` (2026-10-21, on armada)."""


def test_an_accumulated_update_of_a_diffusion_objective_is_its_batch_whole_on_one_device():
    """Two microbatches pooled by the trainer step as their batch whole does: each example draws its
    time, noise and dropped condition by its place among the update's examples, which the halves
    give as the whole gives them, so the pooled update is on par with the whole batch's float32 step
    over drawn weights (check 1), its float64 twin the truth. The weights are the init's, each leaf
    moved by a draw, so the DiT's zero-initialized layers take a gradient too. Drawn by the attempt,
    the halves read other noise than the whole, and the step another update."""
    import optax
    from reference_error import assert_as_exact

    from dew.training import Trainer
    from dew.training.transaction import example_fields

    objective = make_objective(unconditional_prob=0.5)
    rows = 4 * jax.device_count()
    whole = make_batch_of(2 * rows)
    halves = [jax.tree.map(lambda value, part=part: value[part * rows:(part + 1) * rows], whole)
              for part in (0, 1)]
    trainer = Trainer(objective, optax.sgd(1.0), key=jax.random.PRNGKey(3), accumulation=2)
    initial = trainer.initial_state()
    template = jax.tree.map(jnp.copy, initial)
    run = trainer.compile(initial, halves[0])
    fields = example_fields(initial, 0, 1, whole)
    info = Step(jnp.asarray(0), jax.random.fold_in(initial.key, 0), None, **fields)
    twin = float64_twin(objective)

    @as_given
    def pooled(variables):
        # The step donates its state: each draw steps a copy of the untouched template.
        state = jax.tree.map(jnp.copy, template).replace(variables=jax.tree.map(jnp.asarray, variables))
        partial, *_ = run(state, halves[0])
        stepped, *_ = run(partial, halves[1])
        assert int(stepped.updates) == 1
        after = jax.tree.map(np.asarray, stepped.variables["params"])
        return jax.tree.map(applied, variables["params"], after)

    # The whole batch's step.
    alone = float32_update(as_a_step(trainer, initial.variables, halves[0],
                                     lambda variables: objective_gradient(objective, variables, whole, info)))
    truth = float64_gradient(twin, info)
    assert_as_exact(pooled, alone, moved_weights(template.variables), "pooled update", seeds=POOLED_SEEDS,
                    truth_fn=lambda variables: truth(variables, float64_pixels(whole)))


HOST_MASTER_SEEDS = 16
"""Check 1's declared seeds for the host-master update, sized paired (`seeds_for` with the host step as
`dew_fn`)."""


def test_a_host_master_step_draws_each_examples_noise_as_the_resident_step_does():
    """A step whose variables live on the host runs its loss on the accelerator with the examples'
    draws the step carries, so it takes the update a resident step takes: on par with it over drawn
    weights (check 1), the objective's float64 twin at the step's draws the truth. The two steps are
    two programs, which round otherwise: the host's update is its CPU's, though its accelerator
    programs compile under the resident step's options (`step_compiler_options`). A host-master run's
    draws by the attempt would read other times, noise and dropped conditions, and step elsewhere."""
    import optax
    from reference_error import as_given, assert_as_exact

    from dew.training import Layout, Trainer
    from dew.training.transaction import example_fields

    objective = make_objective(unconditional_prob=0.5)
    batch = make_batch_of(4 * jax.device_count())

    def stepping(layout):
        """The layout's step of drawn weights, its update applied (`applied`), and its placed state."""
        trainer = Trainer(objective, optax.sgd(1.0), key=jax.random.PRNGKey(3), layout=layout)
        placed, _, _ = trainer.place()
        template = jax.tree.map(jnp.copy, placed)
        run = trainer.compile(placed, batch)
        where = jax.tree.map(lambda leaf: leaf.sharding, template.variables)

        @as_given
        def stepped(variables):
            # The step donates its state: each draw steps a copy of the untouched template.
            state = jax.tree.map(jnp.copy, template).replace(variables=jax.device_put(variables, where))
            state, *_ = run(state, batch)
            after = jax.tree.map(np.asarray, state.variables["params"])
            return jax.tree.map(applied, variables["params"], after)

        return stepped, template

    host, _ = stepping(Layout(host=("variables",), min_shard=1, tolerance=1.))
    resident, template = stepping(Layout(min_shard=1, tolerance=1.))
    info = Step(jnp.asarray(0), jax.random.fold_in(template.key, 0), None,
                **example_fields(template, 0, 1, batch))
    truth = float64_gradient(float64_twin(objective), info)
    assert_as_exact(host, resident, moved_weights(template.variables), "host-master update",
                    seeds=HOST_MASTER_SEEDS,
                    truth_fn=lambda variables: truth(variables, float64_pixels(batch)))


def test_a_fit_validates_on_an_unbatched_dataset_in_trainings_batches_its_caption_steps_after_them():
    """A fit of a captioned diffusion objective over a batched training chain validates on a dataset that
    states no batch of its own, read in training's batches, the objective's caption-encoding batch steps
    after its batch as in training (row 2085: the validation chain was measured before its batch, so its
    batch steps stood ahead of it and preparation refused the fit)."""
    from dew.data import Memory

    batch = make_batch_of(8)
    held = make_batch_of(5)
    objective = make_objective(guidance=None)
    state = Trainer(objective, optax.sgd(0.0), key=0).fit(
        Memory(batch).batch(4), validation=Memory(held), steps=1, eval_every=1, log_every=1)
    assert state is not None
