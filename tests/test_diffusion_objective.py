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
from flax import linen as nn

from dew.data.dataset import Loader
from dew.diffusion import broadcast_rates, expand, presets
from dew.inputs import CharTable, Condition, Field, InputSpec, unit_range
from dew.nn.backbones import SimpleDiT, SimpleMMDiT
from dew.objectives.base import Step
from dew.objectives.diffusion import DiffusionObjective
from dew.sampling import CFG, Euler
from dew.training import Trainer

_DEFAULT_MAKE_OBJECTIVE_GUIDANCE = CFG(2.0)


def make_objective(*, guidance: CFG | None = _DEFAULT_MAKE_OBJECTIVE_GUIDANCE):
    model = SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1)
    inputs = InputSpec(Field("image", (RES, RES, 3)),
                       {"textcontext": Condition(StubText.from_pretrained("stub"))})
    return DiffusionObjective(model, presets.EDM(P_mean=-0.4, P_std=1.0), inputs, steps=3, guidance=guidance,
                              solver=Euler())


def make_batch(count=8):
    images = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, :, None, None],
                     (count, 1, RES, 3)).astype(np.uint8)
    encoder = StubText.from_pretrained("stub")
    return {"image": images,
            "text": encoder.tokenize(["a bird", "cat", "", "two dogs", "x", "y", "zz", "w"][:count])}


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
    from dew.sampling import TextToImage

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
    mean of the l2 error, with t and the noise drawn from the step's key in
    the objective's order."""
    process = presets.EDM(regime="pixel")()
    inputs = InputSpec(Field("image", (RES, RES, 3)))
    objective = DiffusionObjective(Zero(), process, inputs)
    params = objective.init(jax.random.PRNGKey(0))
    batch = make_batch()
    step = Step(step=jnp.asarray(3), key=jax.random.PRNGKey(7), ema=None)

    loss, aux = objective.scalar_loss(params, batch, step)

    _, _, time_key, noise_key, _ = jax.random.split(step.key, 5)
    x0 = unit_range(batch["image"])
    t = process.schedule.sample_t(time_key, 8)
    noise = jax.random.normal(noise_key, x0.shape)
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


@pytest.mark.parametrize("probability", [0.5, 1.0])
def test_null_dropout_matches_explicit_tokens_under_current_encoder(conditional_mmdit, probability):
    source, variables, batch, step = conditional_mmdit
    dropped = DiffusionObjective(source.model, source.process, source.inputs,
                                 unconditional_prob=probability, steps=3, solver=Euler(),
                                 variables=variables)
    conditional = DiffusionObjective(source.model, source.process, source.inputs,
                                     unconditional_prob=0.0, steps=3, solver=Euler(),
                                     variables=variables)
    mask = jax.random.bernoulli(jax.random.split(step.key, 5)[1], probability, (4,))
    explicit = source.inputs.tokenize([""] * 4)
    explicit = {"image": batch["image"], "text": jax.tree.map(
        lambda blank, given: jnp.where(mask[:, None], blank, given), explicit["text"], batch["text"])}
    expected = jax.jit(jax.value_and_grad(
        lambda values: conditional.scalar_loss(values, explicit, step)[0]))(variables)
    actual = jax.jit(jax.value_and_grad(
        lambda values: dropped.scalar_loss(values, batch, step)[0]))(variables)
    assert float(jnp.linalg.norm(expected[1]["encoders"]["textcontext"]["table"])) > 1e-6
    # The trained weights, on the loss the two routes agree on. Both encode the blank with the step's
    # frozen tower; nothing applies a gradient to it.
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-6, rtol=1e-6)
    for left, right in zip(jax.tree.leaves(actual[1]["params"]),
                           jax.tree.leaves(expected[1]["params"]), strict=True):
        np.testing.assert_array_equal(left, right)


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
