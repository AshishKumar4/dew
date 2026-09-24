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

from dew.data import Dataset
from dew.diffusion import EpsilonPredictionTransform, Process
from dew.diffusion.discrete import MDLM, DiscreteProcess, LogLinear, Unmask
from dew.diffusion.schedules import CosineNoiseScheduler
from dew.objectives.base import Step, scalar_loss
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


def test_a_zero_time_row_contributes_nothing_to_the_loss(rng, monkeypatch):
    """t = 0 masks nothing, so its NELBO contribution is exactly zero: the
    loss stays finite and equals the batch with that row removed. The
    stratified offset draws t = 0 about once in 2^24 rows, and a weight of
    1 / t there would be a NaN the trainer aborts the run over."""
    process = MDLM(mask_id=MASK)()
    objective = MaskedDiffusionObjective(transformer(causal=False), process, 8)
    params = objective.init(rng)
    rows = jnp.array([[1, 2, 3, 4, 5, 1, 2, 3], [3, 2, 1, 0, 4, 5, 1, 2]])
    monkeypatch.setattr(DiscreteProcess, "sample_t",
                        lambda self, key, n: jnp.zeros((n,)))
    full, _ = scalar_loss(objective, params, {"text": rows}, Step(jnp.asarray(0), rng, None))
    rest, _ = scalar_loss(objective, params, {"text": rows[1:]}, Step(jnp.asarray(0), rng, None))
    assert jnp.all(jnp.isfinite(full))
    assert float(full) == pytest.approx(0.0, abs=1e-12)
    assert float(full) == pytest.approx(float(rest), abs=1e-12)


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


def test_the_denoiser_normalizes_bf16_logits_in_fp32():
    """The reveal draws from these log probabilities. Against a float64
    log-softmax of the same bf16 values, fp32 misses by 9.5e-7, within
    4 * eps32 * max|log p| = 1.4e-5; a bf16 reduction misses by 0.11."""
    logits = (4 * jax.random.normal(jax.random.key(0), (2, 4, 1024))).astype(jnp.bfloat16)

    class Fixed(nn.Module):
        @nn.compact
        def __call__(self, tokens):
            return logits

    mask = 1023
    _, log_probs = DiscreteProcess(LogLinear(), mask_id=mask).denoiser(Fixed(), {})(
        jnp.full((2, 4), mask, jnp.int32), jnp.full((2,), 0.5))

    exact = np.asarray(logits[..., :mask].astype(jnp.float32), np.float64)
    shifted = exact - exact.max(axis=-1, keepdims=True)
    expected = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    assert np.all(np.isneginf(np.asarray(log_probs)[..., mask]))
    error = np.abs(np.asarray(log_probs[..., :mask], np.float64) - expected).max()
    assert error <= 4 * float(np.finfo(np.float32).eps) * float(np.abs(expected).max())


def test_unmasking_never_emits_the_mask_id(rng):
    """The mask token marks corruption; it is not a token a sample can end
    with, however the model scores it. A model that puts all its mass there
    must still unmask to real tokens."""
    class MaskLoving(nn.Module):
        @nn.compact
        def __call__(self, tokens):
            scale = self.param("scale", nn.initializers.constant(20.0), ())
            return scale * jnp.broadcast_to(jax.nn.one_hot(MASK, VOCAB),
                                            (*tokens.shape, VOCAB))

    process = DiscreteProcess(LogLinear(), mask_id=MASK)
    model = MaskLoving()
    denoise = process.denoiser(model, model.init(rng, jnp.zeros((1, 8), jnp.int32)))
    out = sample(denoise, process.noise(rng, (5, 8)), 6, solver=Unmask(), key=rng)
    assert not jnp.any(out == MASK)


def test_a_revealed_token_is_never_drawn_again(rng):
    """MDLM carries a revealed token over: a reverse step draws only where
    the row is still masked. The model here scores every token alike, so a
    step that drew at every position would change five revealed tokens in
    six."""
    class Flat(nn.Module):
        @nn.compact
        def __call__(self, tokens):
            return jnp.zeros((*tokens.shape, VOCAB))

    process = DiscreteProcess(LogLinear(), mask_id=MASK)
    model = Flat()
    denoise = process.denoiser(model, model.init(rng, jnp.zeros((1, 8), jnp.int32)))
    t, s, done = jnp.full((200,), 1.0), jnp.full((200,), 0.5), jnp.zeros((200,))
    x = process.noise(rng, (200, 8))
    half, _ = Unmask().step(x, t, s, *denoise(x, t), (), jax.random.fold_in(rng, 1), process, denoise)
    revealed = np.asarray(half != MASK)
    assert 0.3 < revealed.mean() < 0.7
    final, _ = Unmask().step(half, s, done, *denoise(half, s), (), jax.random.fold_in(rng, 2),
                             process, denoise)
    assert not jnp.any(final == MASK)
    np.testing.assert_array_equal(np.asarray(final)[revealed], np.asarray(half)[revealed])


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


def test_the_loss_is_the_nelbo_of_the_row_the_model_saw(rng, monkeypatch):
    """At fixed times and a fixed corruption, the loss is the cross entropy
    of each masked token under the model's logits for the corrupted row,
    weighted by 1/t and averaged over every position of the batch.
    Evaluation reports each position's term, zero where the token stayed
    visible, under the averaged weights it is handed, so a validation pass's
    perplexity is exp of this bound per token."""
    model = transformer(causal=False)
    objective = MaskedDiffusionObjective(model, MDLM(mask_id=MASK)(), 8)
    params = objective.init(rng)
    rows = jnp.array([[1, 2, 3, 4, 5, 1, 2, 3], [3, 2, 1, 0, 4, 5, 1, 2]])
    times = jnp.array([0.25, 0.5])
    hidden = jnp.array([[1, 0, 0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 0, 0, 0, 1]], bool)
    monkeypatch.setattr(DiscreteProcess, "sample_t", lambda self, key, n: times)
    monkeypatch.setattr(DiscreteProcess, "corrupt",
                        lambda self, key, tokens, t: (jnp.where(hidden, MASK, tokens), hidden))

    def terms(variables):
        log_probs = jax.nn.log_softmax(model.apply(variables, jnp.where(hidden, MASK, rows)), axis=-1)
        cross_entropy = -jnp.take_along_axis(log_probs, rows[..., None], axis=-1)[..., 0]
        return jnp.where(hidden, cross_entropy / times[:, None], 0.0)

    loss, _ = scalar_loss(objective, params, {"text": rows}, Step(jnp.asarray(0), rng, None))
    np.testing.assert_allclose(loss, terms(params).sum() / rows.size, rtol=1e-5)

    averaged = jax.tree.map(lambda leaf: 1.5 * leaf, params)
    scored = objective.evaluate(params, {"text": rows}, Step(jnp.asarray(0), rng, averaged))
    np.testing.assert_allclose(scored.losses, terms(averaged), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(scored.weights, 1.0)


def test_a_packed_window_scores_as_its_documents_would_one_by_one(rng, monkeypatch):
    """Two documents and a padded tail in one window, against each document
    alone in a window of its own: every masked token scores the same, since
    a document attends only to itself, in both directions, from its own
    positions. The tail is never masked or counted, so the loss is the bound
    per real token."""
    objective = MaskedDiffusionObjective(transformer(causal=False), MDLM(mask_id=MASK)(), 8)
    params = objective.init(rng)
    first, second = [1, 2, 3], [4, 5]
    packed = {"text": jnp.array([first + second + [0] * 3]),
              "text_segment_ids": jnp.array([[1, 1, 1, 2, 2, 0, 0, 0]]),
              "text_positions": jnp.array([[0, 1, 2, 0, 1, 0, 0, 0]])}
    alone = {"text": jnp.array([first + [0] * 5, second + [0] * 6]),
             "text_segment_ids": jnp.array([[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]]),
             "text_positions": jnp.array([[0, 1, 2, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]])}
    # One noise level, and a masking that hides one token of each document
    # and every padded slot, which the objective has to leave out.
    hidden = {1: jnp.array([[0, 1, 0, 1, 0, 1, 1, 1]], bool),
              2: jnp.array([[0, 1, 0, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 1, 1, 1]], bool)}
    monkeypatch.setattr(DiscreteProcess, "sample_t", lambda self, key, n: jnp.full((n,), 0.5))
    monkeypatch.setattr(DiscreteProcess, "corrupt", lambda self, key, tokens, t: (
        jnp.where(hidden[tokens.shape[0]], MASK, tokens), hidden[tokens.shape[0]]))
    step = Step(jnp.asarray(0), rng, None)

    together = objective.evaluate(params, packed, step)
    apart = objective.evaluate(params, alone, step)

    np.testing.assert_allclose(together.losses[0, :3], apart.losses[0, :3], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(together.losses[0, 3:5], apart.losses[1, :2], rtol=1e-5, atol=1e-6)
    assert float(together.losses[0, 1]) > 0 and float(together.losses[0, 3]) > 0
    np.testing.assert_array_equal(together.losses[0, 5:], 0.0)
    np.testing.assert_array_equal(together.weights, [[1, 1, 1, 1, 1, 0, 0, 0]])
    loss, _ = scalar_loss(objective, params, packed, step)
    np.testing.assert_allclose(loss, together.losses.sum() / 5, rtol=1e-5)


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
    objective = MaskedDiffusionObjective(model, process, ROW, steps=48, samples=16)

    trainer = Trainer(objective, optax.adam(3e-3), key=jax.random.PRNGKey(0))
    state = trainer.fit(Dataset(train=lambda partition: corpus_batches(), val=None, records=None, batch=16),
                        steps=1000, log_every=500)
    params = state.params

    loss, aux = scalar_loss(objective, params, {"text": ROWS}, Step(state.microstep, jax.random.PRNGKey(1), None))
    assert jnp.isfinite(loss)
    assert set(aux.metrics) == {"masked_accuracy", "masked_fraction"}

    t = jnp.full((len(ROWS),), 0.5)
    masked, is_masked = process.corrupt(jax.random.PRNGKey(5), jnp.asarray(ROWS), t)
    filled, _ = process.denoiser(model, params)(masked, t)
    accuracy = float(jnp.sum((filled == ROWS) & is_masked) / jnp.sum(is_masked))
    assert accuracy > 0.8, accuracy

    artifact = objective.preview(params, {"text": ROWS}, Step(state.step, jax.random.PRNGKey(3), None))
    generated = np.asarray(artifact.tokens)
    assert generated.shape == (16, ROW) and not np.any(generated == BYTE_MASK)
    match = (generated[:, None, :] == ROWS[None, :, :]).mean(-1).max(-1)
    assert float(match.mean()) > 0.3, match
