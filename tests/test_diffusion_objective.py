"""The diffusion objective on the general trainer.

The loss is checked against a hand computation from the process's own parts,
the frozen encoder's weights are shown to reach the compiled step as an
argument and not as a constant, evaluation produces the typed artifact from
the step's key."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from diffusion_stubs import FEATURES, RES, TOKENS, VOCAB, StubText
from flax import linen as nn

from dew.artifacts import ImageGrid, VideoGrid
from dew.data import Dataset
from dew.diffusion import broadcast_rates, expand, presets
from dew.inputs import CharTable, CLIPText, Condition, Field, InputSpec, unit_range
from dew.nn.backbones import SimpleDiT, SimpleMMDiT
from dew.nn.dit import TextContext
from dew.objectives.base import Step
from dew.objectives.diffusion import VALIDATION_SAMPLES, DiffusionObjective
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


def test_build_defers_unconditional_encoding_and_reuses_its_exact_snapshot(monkeypatch):
    """Building binds the towers; their first use encodes the fixed prompt once."""
    encoder = StubText.from_pretrained("stub")
    tokens = encoder.tokenize([""])
    expected = encoder.encode(encoder.params, tokens)
    original = StubText.encode
    calls = []

    def encoded(self, params, tokens):
        assert not any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(params))
        calls.append(True)
        return original(self, params, tokens)

    monkeypatch.setattr(StubText, "encode", encoded)
    objective = make_objective()
    assert not calls
    given = jax.tree.map(jnp.asarray, expected)
    with jax.default_matmul_precision("highest"):
        first = jax.jit(objective.blank_conditions)({"textcontext": given})
        second = jax.jit(objective.blank_conditions)({"textcontext": given})
    assert len(calls) == 1
    for actual in (first, second):
        for got, want in zip(jax.tree.leaves(actual["textcontext"]), jax.tree.leaves(expected), strict=True):
            np.testing.assert_array_equal(got, want)


def test_first_blank_reads_saved_weights_without_constant_folding_their_initializers():
    """Flax's shape validation may trace an initializer, but must not execute it."""
    initialized = []

    def initializer(key, shape, dtype=jnp.float32):
        initialized.append(isinstance(key, jax.core.Tracer))
        return jax.random.normal(key, shape, dtype)

    class Tower(nn.Module):
        @nn.compact
        def __call__(self, ids):
            table = self.param("table", initializer, (VOCAB, FEATURES))
            return table[ids]

    class Encoder(StubText):
        def encode(self, params, tokens):
            return TextContext(hidden=Tower().apply({"params": params}, jnp.asarray(tokens["input_ids"])),
                               mask=jnp.asarray(tokens["attention_mask"]))

    encoder = Encoder.from_pretrained("stub")
    objective = DiffusionObjective(Zero(), presets.Flow(), InputSpec(
        Field("image", (2, 2, 1)), {"textcontext": Condition(encoder)}))
    given = TextContext(jnp.zeros((1, TOKENS, FEATURES)), jnp.ones((1, TOKENS), jnp.int32))
    actual = jax.jit(objective.blank_conditions)({"textcontext": given})
    np.testing.assert_array_equal(actual["textcontext"].hidden,
                                  np.asarray(encoder.params["table"])[encoder.tokenize([""])["input_ids"]])
    assert initialized and all(initialized), "shape checking executed a random weight initializer"


@pytest.mark.parametrize("kind", ["clip", "char"])
def test_lazy_blank_keeps_the_eager_towers_bits_and_construction_precision(kind):
    """A later trace's matmul policy must not change the original eager snapshot."""
    if kind == "clip":
        encoder = CLIPText.from_pretrained(str(Path(__file__).parent / "fixtures/clip/tiny"),
                                           dtype="float32")
    else:
        encoder = CharTable.from_pretrained(dtype="float32")
    inputs = InputSpec(Field("image", (2, 2, 1)),
                       {"textcontext": Condition(encoder, unconditional="a bird")})
    tokens = encoder.tokenize(["a bird"])
    with jax.default_matmul_precision("highest"):
        expected = encoder.encode(encoder.params, tokens)
        objective = DiffusionObjective(Zero(), presets.Flow(), inputs)
    # As in the old constructor, the fixed prompt reads the construction
    # policy. A differently configured caller only casts this saved result.
    with jax.default_matmul_precision("bfloat16"):
        actual = jax.jit(objective.blank_conditions)({"textcontext": expected})
    for got, want in zip(jax.tree.leaves(actual["textcontext"]), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(np.ascontiguousarray(got).view(np.uint8),
                                      np.ascontiguousarray(want).view(np.uint8))


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
    compiled = objective._sample
    replacement = Velocity(.5)
    objective.substitute([replacement])
    after = objective.evaluate(variables, batch, step).images
    fresh = DiffusionObjective(replacement, presets.Flow(), inputs, guidance=None, steps=3)
    expected = fresh.evaluate(variables, batch, step).images
    assert not np.allclose(before, expected)
    np.testing.assert_array_equal(after, expected)
    assert objective._sample is not compiled
    assert objective._sample is objective._sample


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


def test_what_an_objective_cannot_train_or_sample_is_refused_at_construction():
    """A masked process; a solver given by name, one that refuses the schedule
    (the sigma integrators hold only where alpha is 1, which the VP cosine is
    not), or one without the real terminal grid's target; and a model that
    cannot run on the inputs (SimpleMMDiT runs its text as a second stream
    through every block, so it declares `RequiresText`): each is refused when
    the objective is built, before a step."""
    from dew.diffusion.discrete import MDLM
    from dew.sampling import RK4, UniPC

    flow, unconditional = presets.Flow(), InputSpec(Field("image", (RES, RES, 3)))
    mmdit = SimpleMMDiT(patch_size=2, emb_features=8, num_layers=1, num_heads=2)
    unipc = UniPC(3, lower_order_final=False)
    for error, fragment, model, process, settings in (
            (ValueError, "MDLM.*--objective masked_diffusion", Zero(), MDLM(mask_id=1), {}),
            (ValueError, "GeneralizedNoiseScheduler", Zero(), presets.Cosine(), {"solver": RK4()}),
            (TypeError, "names a solver", Zero(), flow, {"solver": "euler"}),
            (ValueError, "sigma=0 target", Zero(), flow, {"solver": unipc, "steps": 7}),
            (ValueError, "unconditionally", mmdit, flow, {})):
        with pytest.raises(error, match=fragment):
            DiffusionObjective(model, process, unconditional, **settings)
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


def test_the_compiled_step_carries_no_frozen_tower_constants():
    """T19: the text encoder's table and the VAE's weights arrive through the
    tree's `encoders` and `autoencoder`, so the loss's jaxpr has no constant
    of the table's or the VAE encoder kernel's shape. A loss that reads them
    off the objective's own towers instead bakes them in, and this catches it."""
    from dew.nn.autoencoders import AutoencoderKL, StableDiffusionVAE
    model = AutoencoderKL(channels=(8, 8), latent_channels=2, blocks_per_level=1, norm_groups=4,
                          dtype=jnp.float32)
    autoencoder = StableDiffusionVAE(
        model=model, params=model.init(jax.random.PRNGKey(0), jnp.zeros((1, RES, RES, 3)))["params"],
        dtype=jnp.float32, latent_shift=0.0, latent_scale=1.0)
    text = make_objective()
    objective = DiffusionObjective(text.model.clone(output_channels=2), text.process, text.inputs, steps=3,
                                   autoencoder=autoencoder)
    params = objective.init(jax.random.PRNGKey(0))
    batch, step = make_batch(), Step(step=jnp.asarray(0), key=jax.random.PRNGKey(1), ema=None)

    def constants(loss):
        return {np.shape(const) for const in jax.make_jaxpr(loss)(params, batch, step).consts}

    class Leaky(DiffusionObjective):
        def loss(self, variables, batch, step):
            towers = {"encoders": self.encoder_params(), "autoencoder": autoencoder.params}
            return super().loss({**variables, **towers}, batch, step)

    frozen = {(VOCAB, FEATURES), (3, 3, 3, 8)}
    assert not frozen & constants(objective.loss)
    leaky = Leaky(objective.model, objective.process, objective.inputs, steps=3, autoencoder=autoencoder)
    assert frozen <= constants(leaky.loss)


def encode_calls(monkeypatch, encoder) -> list:
    """The token batches `encoder`'s class encodes, recorded as they happen."""
    calls: list = []
    original = type(encoder).encode

    def counted(self, params, tokens):
        calls.append(tokens)
        return original(self, params, tokens)

    monkeypatch.setattr(type(encoder), "encode", counted)
    return calls


def test_the_text_tower_runs_once_a_step(monkeypatch):
    """The unconditional branch is a pure function of the frozen tower and a
    fixed prompt, so the objective caches its first encoding and the
    compiled step encodes the batch and nothing else. Encoding it in the step
    instead ran the tower twice a step, the second time over one row of
    padding."""
    objective = make_objective()
    params = objective.init(jax.random.PRNGKey(0))
    batch = make_batch()
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(1), ema=None)
    # First use prepares the fixed prompt independently of the compiled step.
    _ = objective.unconditional_conditions
    calls = encode_calls(monkeypatch, objective.inputs.conditions["textcontext"].encoder)

    jax.make_jaxpr(objective.loss)(params, batch, step)

    assert len(calls) == 1
    assert np.shape(calls[0]["input_ids"])[0] == batch["image"].shape[0]


def test_the_unconditional_branch_is_cached_as_host_arrays_on_first_use():
    """What the objective holds is what encoding the tower again produces, to
    the bit, and it is host arrays rather than a leaf of the state: the state
    an objective initializes has the collections it always had."""
    objective = make_objective()
    for held, encoded in zip(jax.tree.leaves(objective.unconditional_conditions),
                             jax.tree.leaves(objective.encode(objective.encoder_params())),
                             strict=True):
        np.testing.assert_array_equal(held, encoded)

    params = objective.init(jax.random.PRNGKey(0))
    assert set(params["encoders"]) == set(objective.inputs.conditions)


def test_a_checkpoint_of_this_state_resumes_in_place(tmp_path):
    """The state carries no derived leaf, so a run resumes from its own
    checkpoint directory through the trainer, restoring into the template
    `init` describes and taking the next step."""
    from dew.training import Checkpoints

    objective = make_objective()
    batch = make_batch()

    class Stream:
        def __init__(self):
            self.position = 0

        def __iter__(self):
            return self

        def __next__(self):
            self.position += 1
            return batch

        def get_state(self):
            return str(self.position).encode()

        def set_state(self, state):
            self.position = int(state.decode())

    def trainer():
        return Trainer(make_objective(), optax.adam(1e-3), key=jax.random.PRNGKey(0),
                       checkpoints=Checkpoints(str(tmp_path), keep=1))

    data = Dataset(train=lambda partition: Stream(), val=None, records=None, batch=8)
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


def test_a_sampling_call_does_not_encode_the_tasks_own_unconditional_prompt(monkeypatch):
    """The pipeline's own unconditional prompt is the one the task was built
    with, so preparing a call traces the tower over the prompts alone.
    Negatives a caller passes are their own text, and are encoded."""
    from dew.sampling import TextToImage

    objective = make_objective()
    params = objective.init(jax.random.PRNGKey(0))
    pipe = TextToImage.from_objective(objective, params)
    # The first use caches the fixed prompt; later requests only encode their text.
    _ = objective.unconditional_conditions
    calls = encode_calls(monkeypatch, objective.inputs.conditions["textcontext"].encoder)

    prepared = pipe.prepare(["a bird", "a cat"], steps=3, key=0)
    assert len(calls) == 1
    for used, held in zip(jax.tree.leaves(prepared.unconditional),
                          jax.tree.leaves(objective.unconditional_conditions), strict=True):
        np.testing.assert_array_equal(used, held)

    pipe.prepare(["a bird", "a cat"], steps=3, key=0, unconditional="a blurry photo")
    assert len(calls) == 2
    assert np.shape(calls[1]["input_ids"]) == (1, TOKENS)


def test_uint8_pixels_reach_the_sampler_as_training_normalizes_them():
    """An img2img or inpainting call's uint8 image normalizes exactly as
    training reads the same pixels through `unit_range`: subtracting 127.5
    and then dividing, where dividing and then subtracting 1 differs in
    float32 on 128 of the 256 levels, by up to 6e-8."""
    from dew.inputs import unit_range
    from dew.sampling import TextToImage

    objective = make_objective()
    pipe = TextToImage.from_objective(objective, objective.init(jax.random.PRNGKey(0)))
    levels = np.resize(np.arange(256, dtype=np.uint8), (2, RES, RES, 3))
    supplied = pipe._supplied(2, pipe.latent_shape, image=levels, image_latents=None, mask=None,
                              noise=None, initial=None)
    np.testing.assert_array_equal(supplied["image"], np.asarray(unit_range(levels)))


def test_scoring_covers_all_conditions_and_preview_decodes_only_its_small_draw():
    objective = make_objective()
    params = objective.init(jax.random.PRNGKey(0))
    batch = make_batch()
    step = Step(step=jnp.asarray(5), key=jax.random.PRNGKey(2), ema=None)

    artifact = objective.evaluate(params, batch, step)
    assert isinstance(artifact, ImageGrid)
    assert artifact.images.shape == (batch[objective.inputs.sample.key].shape[0], RES, RES, 3)
    assert artifact.captions == ()
    preview = objective.preview(params, batch, step)
    assert isinstance(preview, ImageGrid)
    encoder = objective.inputs.conditions["textcontext"].encoder
    assert preview.captions == encoder.captions(
        {"input_ids": batch["text"]["input_ids"][:VALIDATION_SAMPLES]})
    assert len(preview.captions) == VALIDATION_SAMPLES and preview.captions[2] == ""
    assert preview.images.shape == (VALIDATION_SAMPLES, RES, RES, 3)


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


def test_a_video_objective_returns_a_video_grid():
    class ZeroVideo(nn.Module):
        @nn.compact
        def __call__(self, x, temb, train=False):
            return jnp.zeros_like(x) * self.param("w", nn.initializers.ones, ())

    objective = DiffusionObjective(ZeroVideo(), presets.Flow(),
                                   InputSpec(Field("video", (2, RES, RES, 3))), steps=2, guidance=None)
    assert objective.artifact is VideoGrid
    params = objective.init(jax.random.PRNGKey(0))
    batch = {"video": np.zeros((3, 2, RES, RES, 3), np.uint8)}
    artifact = objective.evaluate(params, batch, Step(jnp.asarray(0), jax.random.PRNGKey(0), None))
    assert isinstance(artifact, VideoGrid) and artifact.videos.shape == (3, 2, RES, RES, 3)


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
    # The trained weights, on the loss the two routes agree on. The dropped
    # rows read a blank the objective encoded once, so the frozen tower is a
    # constant on that route and takes no gradient through it; it is frozen,
    # and nothing applies the gradient the explicit route happens to produce.
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-6, rtol=1e-6)
    for left, right in zip(jax.tree.leaves(actual[1]["params"]),
                           jax.tree.leaves(expected[1]["params"]), strict=True):
        np.testing.assert_allclose(left, right, atol=1e-6, rtol=1e-6)


def test_a_dropped_row_is_conditioned_on_what_the_objective_holds(conditional_mmdit):
    """The step takes the unconditional branch from the objective: move what
    it holds and a fully dropped batch's loss moves with it. A step that
    encoded the prompt again would not notice."""
    source, variables, batch, step = conditional_mmdit
    dropped = DiffusionObjective(source.model, source.process, source.inputs,
                                 unconditional_prob=1.0, steps=3, solver=Euler(),
                                 variables=variables)
    held = dropped.unconditional_conditions
    before = float(dropped.scalar_loss(variables, batch, step)[0])
    # The branch is encoded once and held; moving it in place moves the step.
    held.update(jax.tree.map(
        lambda leaf: leaf + 1.0 if np.issubdtype(leaf.dtype, np.floating) else leaf, dict(held)))

    assert float(dropped.scalar_loss(variables, batch, step)[0]) != pytest.approx(before)


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
