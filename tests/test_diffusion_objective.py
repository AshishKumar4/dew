"""The diffusion objective on the general trainer.

The loss is checked against a hand computation from the process's own parts,
the frozen encoder's weights are shown to reach the compiled step as an
argument and not as a constant, evaluation produces the typed artifact from
the step's key, and a golden fingerprint of five real steps pins the numbers
of the objective and the trainer together.
"""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn

from dew.artifacts import ImageGrid, VideoGrid
from dew.data import Dataset
from dew.diffusion import broadcast_rates, expand, presets
from dew.inputs import CharTable, Condition, ConditionEncoder, Field, InputSpec, unit_range
from dew.nn.dit import TextContext
from dew.objectives.base import Step, Variables, scalar_loss
from dew.objectives.diffusion import VALIDATION_SAMPLES, DiffusionObjective
from dew.registry import encoders, models
from dew.sampling import CFG, Euler
from dew.training import Trainer

RES = 8
TOKENS = 5
FEATURES = 6
VOCAB = 11


@encoders("stub_text")
@dataclass(frozen=True, eq=False)
class StubText(ConditionEncoder[str]):
    """A text encoder with a table of `VOCAB` vectors: tokenize maps a prompt to
    ids by character behind a start token, encode looks them up. Small, and
    shaped like CLIP's output, so the models' text keyword takes it; registered,
    so a run's text condition can name it."""

    checkpoint: str
    params: Variables

    @classmethod
    def from_pretrained(cls, checkpoint: str, *, params=None, **fields):
        if params is None:
            params = {"table": jnp.asarray(
                np.random.RandomState(0).normal(size=(VOCAB, FEATURES)).astype(np.float32))}
        return cls(checkpoint=checkpoint, params=params)

    def tokenize(self, data):
        ids = np.zeros((len(data), TOKENS), np.int32)
        mask = np.zeros((len(data), TOKENS), np.int32)
        for row, text in enumerate(data):
            codes = [1] + [2 + (ord(char) % (VOCAB - 2)) for char in text[:TOKENS - 1]]
            ids[row, :len(codes)] = codes
            mask[row, :len(codes)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def encode(self, params, tokens):
        return TextContext(hidden=params["table"][jnp.asarray(tokens["input_ids"])],
                           mask=jnp.asarray(tokens["attention_mask"]))

    def captions(self, tokens):
        return tuple("".join(chr(97 + int(i)) for i in row[row > 1])
                     for row in np.asarray(tokens["input_ids"]))

    def to_json(self):
        return {"checkpoint": self.checkpoint}


def make_objective(*, guidance: CFG | None = CFG(2.0)):
    model = models.SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1)
    inputs = InputSpec(Field("image", (RES, RES, 3)),
                       {"textcontext": Condition(StubText.from_pretrained("stub"))})
    return DiffusionObjective(model, presets.EDM()(), inputs, steps=3, guidance=guidance, sampler=Euler())


def make_batch(count=8):
    images = np.tile(np.linspace(0, 255, RES, dtype=np.float32)[None, :, None, None],
                     (count, 1, RES, 3)).astype(np.uint8)
    encoder = StubText.from_pretrained("stub")
    return {"image": images,
            "text": encoder.tokenize(["a bird", "cat", "", "two dogs", "x", "y", "zz", "w"][:count])}


def tree_fingerprint(tree):
    # per-leaf sums accumulated in python floats, so the golden values below
    # do not depend on float32 reduction order
    return sum(float(jnp.sum(leaf)) for leaf in jax.tree.leaves(tree)
               if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating))


def tree_magnitude(tree):
    """Sum of absolute values: no cancellation, so a relative tolerance means what it says."""
    return sum(float(jnp.sum(jnp.abs(leaf))) for leaf in jax.tree.leaves(tree)
               if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating))



class Zero(nn.Module):
    """A model that outputs zero, so the loss is a closed form of the process."""

    @nn.compact
    def __call__(self, x, temb, textcontext=None, train=False):
        return jnp.zeros_like(x) * self.param("w", nn.initializers.ones, ())


def test_a_solver_that_refuses_the_schedule_is_refused_at_construction():
    """The sigma integrators hold only when alpha is 1; the cosine
    preset is VP, and the mismatch surfaces when the objective is built."""
    from dew.sampling import RK4
    unconditional = InputSpec(Field("image", (RES, RES, 3)))
    with pytest.raises(ValueError, match="GeneralizedNoiseScheduler"):
        DiffusionObjective(Zero(), presets.Cosine()(), unconditional, sampler=RK4())
    DiffusionObjective(Zero(), presets.Karras()(), unconditional, sampler=RK4())


@pytest.mark.parametrize("order, steps", [(2, 5), (3, 7)])
def test_singlestep_solver_generates_through_the_objective(order, steps):
    from dew.diffusion import DirectPredictionTransform, LinearNoiseScheduler, Process
    from dew.sampling import DPMSolverSinglestep

    process = Process(LinearNoiseScheduler(1000), DirectPredictionTransform())
    objective = DiffusionObjective(Zero(), process, InputSpec(Field("image", (2, 2, 1))),
                                   sampler=DPMSolverSinglestep(order), steps=steps, guidance=None)
    params = objective.init(jax.random.key(1))
    result = objective.evaluate(params, {"image": np.zeros((2, 2, 2, 1), np.uint8)},
                                Step(step=jnp.asarray(0), key=jax.random.key(2), ema=None))
    np.testing.assert_array_equal(result.images, np.zeros((2, 2, 2, 1), np.float32))


def test_objective_validates_the_real_terminal_grid_before_sampling():
    from dew.sampling import UniPC

    with pytest.raises(ValueError, match="sigma=0 target"):
        DiffusionObjective(Zero(), presets.Flow()(), InputSpec(Field("image", (2, 2, 1))),
                           sampler=UniPC(3, lower_order_final=False), steps=7, guidance=None)


def test_loss_is_the_weighted_error_of_the_prediction():
    """With a zero output, the Karras parameterization predicts x_0 as
    c_skip x_t, the target is x_0, and the loss is the EDM lambda weighted
    mean of the l2 error, with t and the noise drawn from the step's key in
    the objective's order."""
    process = presets.EDM()()
    inputs = InputSpec(Field("image", (RES, RES, 3)))
    objective = DiffusionObjective(Zero(), process, inputs)
    params = objective.init(jax.random.PRNGKey(0))
    batch = make_batch()
    step = Step(step=jnp.asarray(3), key=jax.random.PRNGKey(7), ema=None)

    loss, aux = scalar_loss(objective, params, batch, step)

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


def test_the_compiled_step_carries_no_encoder_constants():
    """T19: the encoder's table arrives through `params["encoders"]`, so the
    loss's jaxpr has no constant of its shape. The mutation that reads the
    table off the encoder object instead bakes it in, and this assertion
    catches that."""
    objective = make_objective()
    params = objective.init(jax.random.PRNGKey(0))
    batch = make_batch()
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(1), ema=None)

    def shapes_of_constants(fn):
        closed = jax.make_jaxpr(fn)(params, batch, step)
        return {np.shape(const) for const in closed.consts}

    assert (VOCAB, FEATURES) not in shapes_of_constants(objective.loss)

    class Leaky(DiffusionObjective):
        def encode(self, encoders, tokens=None):
            return super().encode(self.encoder_params(), tokens)

    leaky = Leaky(objective.model, objective.process, objective.inputs, steps=3)
    assert (VOCAB, FEATURES) in shapes_of_constants(leaky.loss)


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
    fixed prompt, so the objective encodes it when it is built and the
    compiled step encodes the batch and nothing else. Encoding it in the step
    instead ran the tower twice a step, the second time over one row of
    padding."""
    objective = make_objective()
    params = objective.init(jax.random.PRNGKey(0))
    batch = make_batch()
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(1), ema=None)
    calls = encode_calls(monkeypatch, objective.inputs.conditions["textcontext"].encoder)

    jax.make_jaxpr(objective.loss)(params, batch, step)

    assert len(calls) == 1
    assert np.shape(calls[0]["input_ids"])[0] == batch["image"].shape[0]


def test_the_unconditional_branch_is_encoded_when_the_objective_is_built():
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
    assert set(resumed.params["encoders"]) == set(objective.inputs.conditions)
    leaves = jax.tree.leaves(resumed.params["params"])
    assert leaves and all(np.all(np.isfinite(np.asarray(leaf))) for leaf in leaves)


def test_a_sampling_call_does_not_encode_the_tasks_own_unconditional_prompt(monkeypatch):
    """The pipeline's own unconditional prompt is the one the task was built
    with, so preparing a call traces the tower over the prompts alone.
    Negatives a caller passes are their own text, and are encoded."""
    from dew.sampling import TextToImage

    objective = make_objective()
    params = objective.init(jax.random.PRNGKey(0))
    pipe = TextToImage.from_objective(objective, params)
    calls = encode_calls(monkeypatch, objective.inputs.conditions["textcontext"].encoder)

    prepared = pipe.prepare(["a bird", "a cat"], steps=3, seed=0)
    assert len(calls) == 1
    for used, held in zip(jax.tree.leaves(prepared.unconditional),
                          jax.tree.leaves(objective.unconditional_conditions), strict=True):
        np.testing.assert_array_equal(used, held)

    pipe.prepare(["a bird", "a cat"], steps=3, seed=0, unconditional="a blurry photo")
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


def test_the_compiled_step_carries_no_autoencoder_constants():
    """T19, the VAE half: the autoencoder weights arrive through
    `params["autoencoder"]`, so the loss's jaxpr has no constant of the
    encoder kernel's shape. The mutation that reads them off the autoencoder
    object instead bakes them in, and this assertion catches that."""
    from dew.nn.autoencoders import SimpleAutoEncoder
    autoencoder = SimpleAutoEncoder(latent_channels=2, feature_depths=(8,))
    inputs = InputSpec(Field("image", (RES, RES, 3)))
    objective = DiffusionObjective(Zero(), presets.EDM()(), inputs,
                                   autoencoder=autoencoder)
    params = objective.init(jax.random.PRNGKey(0))
    assert set(params) == {"params", "encoders", "autoencoder"}
    batch = make_batch()
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(1), ema=None)

    def shapes_of_constants(fn):
        closed = jax.make_jaxpr(fn)(params, batch, step)
        return {np.shape(const) for const in closed.consts}

    assert (3, 3, 3, 8) not in shapes_of_constants(objective.loss)

    class Leaky(DiffusionObjective):
        def loss(self, params, batch, step):
            params = dict(params, autoencoder=autoencoder.params)
            return super().loss(params, batch, step)

    leaky = Leaky(Zero(), presets.EDM()(), inputs, autoencoder=autoencoder)
    assert (3, 3, 3, 8) in shapes_of_constants(leaky.loss)


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

    objective = DiffusionObjective(ZeroVideo(), presets.Flow()(),
                                   InputSpec(Field("video", (2, RES, RES, 3))), steps=2, guidance=None)
    assert objective.artifact is VideoGrid
    params = objective.init(jax.random.PRNGKey(0))
    batch = {"video": np.zeros((3, 2, RES, RES, 3), np.uint8)}
    artifact = objective.evaluate(params, batch, Step(jnp.asarray(0), jax.random.PRNGKey(0), None))
    assert isinstance(artifact, VideoGrid) and artifact.videos.shape == (3, 2, RES, RES, 3)


def batches(count=8):
    batch = make_batch(count)
    while True:
        yield batch


def test_diffusion_objective_reproduces_the_golden_fingerprint(tmp_path):
    """Five real steps of the tiny conditional DiT on the EDM process pin the
    parameters, the EMA and the optimizer state together.

    The values were captured from this implementation. The fingerprint of the
    inlined train step this objective was lifted out of (8.209761425852776)
    does not carry over: that step chained one random state object through
    the schedule, the noise and the dropout and seeded itself from the
    trainer's own derivation, while every draw here comes from the step's
    fold_in(run_key, step) key split once (design decision 4), and the EDM
    weight is Eq. 8 of Karras et al. without the epsilon guard (T21). Any
    real change in what the objective computes moves these by orders of
    magnitude more than the 1e-6 XLA reassociation leaves between CPUs.
    """
    objective = make_objective()
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.PRNGKey(0))
    data = Dataset(train=lambda partition: batches(), val=None, records=32, batch=8)
    state = trainer.fit(data, steps=5, log_every=100)

    assert int(state.step) == 5
    assert tree_fingerprint(state.params["params"]) == pytest.approx(GOLDEN["params"], rel=1e-6)
    assert tree_fingerprint(state.ema) == pytest.approx(GOLDEN["ema"], rel=1e-6)
    assert tree_magnitude(state.opt_state) == pytest.approx(GOLDEN["opt_state"], rel=1e-6)
    # the frozen encoder came through untouched
    assert jnp.array_equal(state.params["encoders"]["textcontext"]["table"],
                           objective.inputs.conditions["textcontext"].encoder.params["table"])


# Captured on one CPU at c0f4156 (JAX_PLATFORMS=cpu, the eight simulated
# devices of conftest move the third figure after the decimal point by 2e-9).
GOLDEN = {"params": 15.044008062570356, "ema": 15.049092350082788,
          "opt_state": 2.391809580367163}



@pytest.fixture(scope="module")
def conditional_mmdit():
    encoder = CharTable.from_pretrained(tokens=3, features=6, vocab=16)
    inputs = InputSpec(Field("image", (4, 4, 1)), {"textcontext": Condition(encoder)})
    model = models.SimpleMMDiT(output_channels=1, patch_size=2, emb_features=8,
                               num_layers=1, num_heads=2, mlp_ratio=2, attention_impl="xla")
    process = presets.EDM(sigma_max=1.0)()
    objective = DiffusionObjective(model, process, inputs, steps=3, sampler=Euler(), guidance=CFG(2.0))
    variables = objective.init(jax.random.key(0))
    # The initialized zero output head otherwise hides conditioning gradients.
    variables = {**variables, "params": jax.tree.map(lambda leaf: leaf + 0.02, variables["params"])}
    table = variables["encoders"]["textcontext"]["table"].at[1].add(jnp.linspace(-1, 1, 6))
    variables = {**variables, "encoders": {"textcontext": {"table": table}}}
    # The objective encodes the unconditional branch from the weights it is
    # built over, so it is rebuilt over the ones these tests sample under.
    objective = DiffusionObjective(model, process, inputs, steps=3, sampler=Euler(),
                                   guidance=CFG(2.0), pretrained=variables)
    batch = {"image": np.arange(64, dtype=np.uint8).reshape(4, 4, 4, 1) * 3,
             **inputs.tokenize(["ab", "cd", "ef", "gh"])}
    step = Step(jnp.asarray(0), jax.random.key(4), None)
    return objective, variables, batch, step


@pytest.mark.parametrize("probability", [0.5, 1.0])
def test_null_dropout_matches_explicit_tokens_under_current_encoder(conditional_mmdit, probability):
    source, variables, batch, step = conditional_mmdit
    dropped = DiffusionObjective(source.model, source.process, source.inputs,
                                 unconditional_prob=probability, steps=3, sampler=Euler(),
                                 pretrained=variables)
    conditional = DiffusionObjective(source.model, source.process, source.inputs,
                                     unconditional_prob=0.0, steps=3, sampler=Euler(),
                                     pretrained=variables)
    mask = jax.random.bernoulli(jax.random.split(step.key, 5)[1], probability, (4,))
    explicit = source.inputs.tokenize([""] * 4)
    explicit = {"image": batch["image"], "text": jax.tree.map(
        lambda blank, given: jnp.where(mask[:, None], blank, given), explicit["text"], batch["text"])}
    expected = jax.jit(jax.value_and_grad(
        lambda values: scalar_loss(conditional, values, explicit, step)[0]))(variables)
    actual = jax.jit(jax.value_and_grad(
        lambda values: scalar_loss(dropped, values, batch, step)[0]))(variables)
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
                                 unconditional_prob=1.0, steps=3, sampler=Euler(),
                                 pretrained=variables)
    held = dropped.unconditional_conditions
    before = float(scalar_loss(dropped, variables, batch, step)[0])
    dropped.unconditional_conditions = jax.tree.map(
        lambda leaf: leaf + 1.0 if np.issubdtype(leaf.dtype, np.floating) else leaf, held)

    assert float(scalar_loss(dropped, variables, batch, step)[0]) != pytest.approx(before)


def test_guided_samples_use_bound_encoder_not_constructor_weights(conditional_mmdit):
    objective, variables, batch, step = conditional_mmdit
    condition = objective.inputs.conditions["textcontext"]
    rebound = replace(condition.encoder, params=variables["encoders"]["textcontext"])
    inputs = replace(objective.inputs, conditions={"textcontext": replace(condition, encoder=rebound)})
    reconstructed = DiffusionObjective(objective.model, objective.process, inputs,
                                       steps=3, sampler=Euler(), guidance=CFG(2.0))
    expected = reconstructed.evaluate(variables, batch, step).images
    actual = objective.evaluate(variables, batch, step).images
    np.testing.assert_array_equal(actual, expected)

