"""The diffusion objective on the general trainer.

The loss is checked against a hand computation from the process's own parts,
the frozen encoder's weights are shown to reach the compiled step as an
argument and not as a constant, evaluation produces the typed artifact from
the step's key, and a golden fingerprint of five real steps pins the numbers
of the objective and the trainer together.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn

import dew.nn.backbones  # registers the models
from dew.artifacts import ImageGrid, VideoGrid
from dew.data import Dataset
from dew.diffusion import Process, broadcast_rates, expand, presets
from dew.inputs import Condition, ConditionEncoder, Field, InputSpec, unit_range
from dew.nn.dit import TextContext
from dew.objectives.base import Step, select
from dew.objectives.diffusion import VALIDATION_SAMPLES, DiffusionObjective
from dew.registry import encoders, models
from dew.sampling import CFG, DDIM, Euler
from dew.training import Trainer

RES = 8
TOKENS = 5
FEATURES = 6
VOCAB = 11


@encoders("stub_text")
@dataclass(frozen=True, eq=False)
class StubText(ConditionEncoder):
    """A text encoder with a table of `VOCAB` vectors: tokenize maps a prompt to
    ids by character behind a start token, encode looks them up. Small, and
    shaped like CLIP's output, so the models' text keyword takes it; registered,
    so a run's text condition can name it."""

    checkpoint: str
    params: dict

    @classmethod
    def from_pretrained(cls, checkpoint: str, **fields):
        return cls(checkpoint=checkpoint, params={"table": jnp.asarray(
            np.random.RandomState(0).normal(size=(VOCAB, FEATURES)).astype(np.float32))})

    def tokenize(self, texts):
        ids = np.zeros((len(texts), TOKENS), np.int32)
        mask = np.zeros((len(texts), TOKENS), np.int32)
        for row, text in enumerate(texts):
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


def make_objective(**kwargs):
    model = models.SimpleDiT(patch_size=4, emb_features=16, num_layers=1, num_heads=2, mlp_ratio=1)
    inputs = InputSpec(Field("image", (RES, RES, 3)),
                       {"textcontext": Condition(StubText.from_pretrained("stub"))})
    settings = dict(steps=3, guidance=CFG(2.0), sampler=Euler())
    settings.update(kwargs)
    return DiffusionObjective(model, presets.EDM()(), inputs, **settings)


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


def test_the_tree_holds_the_model_and_the_frozen_encoders():
    objective = make_objective()
    tree = objective.init(jax.random.PRNGKey(0))
    assert set(tree) == {"params", "encoders"}
    assert tree["encoders"]["textcontext"]["table"].shape == (VOCAB, FEATURES)
    # the EMA tracks the model alone: a frozen encoder needs no average
    assert set(select(tree, objective.ema.select)) == {"params"}
    assert objective.artifact is ImageGrid


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

    loss, aux = objective.loss(params, batch, step)

    _, _, time_key, noise_key, _ = jax.random.split(step.key, 5)
    x0 = unit_range(batch["image"])
    t = process.schedule.sample_t(time_key, 8)
    noise = jax.random.normal(noise_key, x0.shape)
    rates = broadcast_rates(process.schedule, t, x0)
    x_t = rates[0] * x0 + rates[1] * noise
    predicted = process.prediction.pred_transform(x_t, jnp.zeros_like(x_t), rates)
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
        def encode(self, encoders, tokens):
            return {keyword: condition.encoder.encode(condition.encoder.params, tokens[keyword])
                    for keyword, condition in self.inputs.conditions.items()}

    leaky = Leaky(objective.model, objective.process, objective.inputs, steps=3)
    assert (VOCAB, FEATURES) in shapes_of_constants(leaky.loss)


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
            params = dict(params, autoencoder=self.autoencoder.params)
            return super().loss(params, batch, step)

    leaky = Leaky(Zero(), presets.EDM()(), inputs, autoencoder=autoencoder)
    assert (3, 3, 3, 8) in shapes_of_constants(leaky.loss)


def test_evaluate_samples_from_the_batch_conditions():
    """The artifact is `VALIDATION_SAMPLES` images in [-1, 1] captioned with
    the batch's text, from the averaged weights when the step carries them."""
    objective = make_objective()
    params = objective.init(jax.random.PRNGKey(0))
    batch = make_batch()
    step = Step(step=jnp.asarray(5), key=jax.random.PRNGKey(2), ema=None)

    artifact = objective.evaluate(params, batch, step)
    assert isinstance(artifact, ImageGrid)
    assert artifact.images.shape == (VALIDATION_SAMPLES, RES, RES, 3)
    assert float(artifact.images.min()) >= -1.0 and float(artifact.images.max()) <= 1.0
    encoder = objective.inputs.conditions["textcontext"].encoder
    assert artifact.captions == encoder.captions(
        {"input_ids": batch["text"]["input_ids"][:VALIDATION_SAMPLES]})
    assert len(artifact.captions) == VALIDATION_SAMPLES and artifact.captions[2] == ""


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
    data = Dataset(train=batches, val=None, records=32, batch=8)
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
