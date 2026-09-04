"""Masked discrete diffusion: the algebra in dew.diffusion.discrete and the
objective that trains a full-attention CausalTransformer with it on the LM
data path, through the general trainer.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn

import dew.nn.backbones  # registers the models
from dew.artifacts import TextSamples
from dew.data import Dataset
from dew.diffusion import EpsilonPredictionTransform, Process
from dew.diffusion.discrete import MDLM, DiscreteProcess, LogLinear, Unmask
from dew.diffusion.schedules import CosineNoiseScheduler
from dew.objectives.base import Step
from dew.objectives.diffusion import MaskedDiffusionObjective
from dew.registry import models, presets, samplers
from dew.sampling import sample
from dew.training import Trainer

VOCAB = 7
MASK = VOCAB - 1
TIMES = jnp.array([0.1, 0.5, 0.9])


def test_log_linear_schedule_and_its_nelbo_weight():
    """alpha(t) = 1 - (1 - eps) t runs from 1 to eps, and the weight
    -alpha'(t) / (1 - alpha(t)) is 1 / t."""
    process = DiscreteProcess(LogLinear(eps=1e-3), mask_id=MASK)
    assert float(process.schedule.alpha(0.0)) == 1.0
    assert float(process.schedule.alpha(1.0)) == pytest.approx(1e-3, rel=1e-4)
    assert jnp.allclose(process.weight(TIMES), 1 / TIMES, rtol=1e-5)


def test_corrupt_masks_the_schedules_fraction_and_keeps_the_rest(rng):
    process = DiscreteProcess(LogLinear(), mask_id=MASK)
    tokens = jax.random.randint(rng, (3, 4000), 0, MASK)
    masked, is_masked = process.corrupt(jax.random.fold_in(rng, 1), tokens, TIMES)
    fraction = is_masked.mean(axis=1)
    assert jnp.allclose(fraction, 1 - process.schedule.alpha(TIMES), atol=0.03)
    assert jnp.all(jnp.where(is_masked, masked == MASK, masked == tokens))


def test_training_times_are_stratified_over_the_batch(rng):
    process = DiscreteProcess(LogLinear(), mask_id=MASK)
    t = process.sample_t(rng, 10)
    assert jnp.all((t >= 0) & (t < 1))
    # one draw per tenth, so the weights 1 / t of a batch cover the trajectory
    assert jnp.array_equal(jnp.floor(jnp.sort(t) * 10), jnp.arange(10))


class Peaked(nn.Module):
    """Logits that put every position on the token equal to its index mod
    (VOCAB - 1), so a revealed token is predictable."""

    @nn.compact
    def __call__(self, tokens):
        scale = self.param("scale", nn.initializers.constant(20.0), ())
        target = jnp.arange(tokens.shape[1]) % (VOCAB - 1)
        return scale * jax.nn.one_hot(target, VOCAB)[None]


def test_unmask_reveals_the_schedules_share_with_the_models_token(rng):
    """From t to s, a masked position is revealed with probability
    (alpha(s) - alpha(t)) / (1 - alpha(t)) and takes the model's draw; s = t
    reveals nothing and s = 0 reveals everything."""
    process = DiscreteProcess(LogLinear(), mask_id=MASK)
    model = Peaked()
    denoise = process.denoiser(model, model.init(rng, jnp.zeros((1, 8), jnp.int32)))
    x = jnp.full((200, 8), MASK, jnp.int32)
    t = jnp.full((200,), 0.8)
    filled, log_probs = denoise(x, t)
    assert jnp.array_equal(filled, jnp.broadcast_to(jnp.arange(8) % (VOCAB - 1), (200, 8)))

    s = jnp.full((200,), 0.3)
    stepped, _ = Unmask().step(x, t, s, filled, log_probs, (), rng, process, denoise)
    revealed = stepped != MASK
    share = (process.schedule.alpha(0.3) - process.schedule.alpha(0.8)) / (1 - process.schedule.alpha(0.8))
    assert abs(float(revealed.mean()) - float(share)) < 0.03
    assert jnp.all(jnp.where(revealed, stepped == filled, True))

    same, _ = Unmask().step(x, t, t, filled, log_probs, (), rng, process, denoise)
    assert jnp.all(same == MASK)
    done, _ = Unmask().step(x, t, jnp.zeros((200,)), filled, log_probs, (), rng, process, denoise)
    assert jnp.all(done != MASK)


def test_sample_walks_the_grid_to_a_fully_revealed_row(rng):
    process = DiscreteProcess(LogLinear(), mask_id=MASK)
    model = Peaked()
    denoise = process.denoiser(model, model.init(rng, jnp.zeros((1, 8), jnp.int32)))
    x_T = process.noise(rng, (5, 8))
    assert jnp.all(x_T == MASK)
    out = sample(denoise, x_T, 6, solver=Unmask(), key=rng)
    assert jnp.array_equal(out, jnp.broadcast_to(jnp.arange(8) % (VOCAB - 1), (5, 8)))


def test_unmask_refuses_a_gaussian_process(rng):
    process = Process(CosineNoiseScheduler(10), EpsilonPredictionTransform())
    x = jnp.zeros((2, 4))
    with pytest.raises(ValueError, match="DiscreteProcess"):
        Unmask().step(x, jnp.ones((2,)), jnp.zeros((2,)), x, x, (), rng, process, None)


def test_the_mdlm_preset_is_registered_and_takes_no_conditions():
    assert presets["mdlm"] is MDLM and samplers["unmask"] is Unmask
    process = presets.build("mdlm", mask_id=MASK, eps=1e-2)()
    assert process.mask_id == MASK and process.schedule == LogLinear(eps=1e-2)
    with pytest.raises(ValueError, match="no conditions"):
        process.denoiser(Peaked(), {}, {"label": jnp.zeros((1,))})


############################################################################################################
# Full attention on the CausalTransformer
############################################################################################################

def transformer(causal):
    return models.CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=1, num_heads=2,
                                    max_seq_len=8, causal=causal)


def test_causal_false_lets_a_position_read_the_future(rng):
    tokens = jnp.array([[1, 2, 3, 4, 5, 1, 2, 3]])
    changed = tokens.at[0, 6].set(4)
    for causal in (True, False):
        model = transformer(causal)
        params = model.init(rng, tokens)
        moved = not jnp.allclose(model.apply(params, tokens)[0, 0], model.apply(params, changed)[0, 0])
        assert moved is (not causal)


def test_full_attention_has_no_cache(rng):
    model = transformer(causal=False)
    params = model.init(rng, jnp.zeros((1, 8), jnp.int32))
    with pytest.raises(ValueError, match="no KV cache"):
        model.apply(params, 2, method=type(model).init_cache, mutable=["cache"])


def test_the_masked_objective_refuses_a_causal_model():
    with pytest.raises(ValueError, match="causal=False"):
        MaskedDiffusionObjective(transformer(causal=True), MDLM(mask_id=MASK)(), 8)


############################################################################################################
# A masked diffusion LM trains on the LM data path with no trainer change
############################################################################################################

SENTENCES = ["the cat sat on the mat.", "a dog ran in the park.", "rain fell on the roof.",
             "she read a long book.", "birds sing at dawn.", "we ate bread and jam.",
             "the sun set over hills.", "he wrote a short note."]
ROW = 24
BYTE_MASK = 256
ROWS = np.array([[ord(char) for char in text.ljust(ROW)] for text in SENTENCES], np.int32)


def corpus_batches():
    rng = np.random.RandomState(0)
    while True:
        yield {"text": ROWS[rng.randint(0, len(ROWS), 16)]}


def test_masked_diffusion_lm_memorises_the_toy_corpus():
    """1000 steps on eight sentences: the model fills half-masked rows of the
    corpus at over 80% accuracy where a random byte would be right 1 in 257
    times, and rows unmasked from nothing match a corpus sentence at a rate
    above the 0.19 that rows of random corpus characters reach (measured,
    tests/test_discrete.py at c0f4156). The loss reports the masked accuracy
    and the masked fraction beside the NELBO."""
    process = MDLM(mask_id=BYTE_MASK)()
    model = models.CausalTransformer(vocab_size=257, emb_features=64, num_layers=2, num_heads=4,
                                     max_seq_len=ROW, causal=False)
    objective = MaskedDiffusionObjective(model, process, ROW, steps=48, samples=16,
                                         decode=lambda ids: "".join(chr(min(i, 255)) for i in ids))
    assert objective.artifact is TextSamples and objective.inputs.sample.shape == (ROW,)

    trainer = Trainer(objective, optax.adam(3e-3), key=jax.random.PRNGKey(0))
    state = trainer.fit(Dataset(train=corpus_batches, val=None, records=None, batch=16),
                        steps=1000, log_every=500)
    params = state.params

    loss, aux = objective.loss(params, {"text": ROWS}, Step(state.step, jax.random.PRNGKey(1), None))
    assert set(aux.metrics) == {"masked_accuracy", "masked_fraction"}
    assert jnp.isfinite(loss)

    t = jnp.full((len(ROWS),), 0.5)
    masked, is_masked = process.corrupt(jax.random.PRNGKey(5), jnp.asarray(ROWS), t)
    filled, _ = process.denoiser(model, params)(masked, t)
    accuracy = float(jnp.sum((filled == ROWS) & is_masked) / jnp.sum(is_masked))
    assert accuracy > 0.8, accuracy

    artifact = objective.evaluate(params, {"text": ROWS}, Step(state.step, jax.random.PRNGKey(3), None))
    assert isinstance(artifact, TextSamples) and len(artifact.texts) == 16
    generated = np.asarray(artifact.tokens)
    assert generated.shape == (16, ROW) and not np.any(generated == BYTE_MASK)
    match = (generated[:, None, :] == ROWS[None, :, :]).mean(-1).max(-1)
    assert float(match.mean()) > 0.3, match
