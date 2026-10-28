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
from reference_error import assert_as_exact
from sampling_gate import assert_draws_as_declared
from scipy import stats

from dew.data.dataset import Loader
from dew.diffusion import EpsilonPredictionTransform, Process
from dew.diffusion.discrete import MDLM, DiscreteProcess, LogLinear, Unmask
from dew.diffusion.schedules import CosineNoiseScheduler
from dew.nn.backbones import CausalTransformer
from dew.objectives.base import Step
from dew.objectives.diffusion import MaskedDiffusionObjective
from dew.registry import presets, solvers
from dew.sampling import sample
from dew.training import Trainer

VOCAB = 7
MASK = VOCAB - 1
TIMES = jnp.array([0.1, 0.5, 0.9])

CORRUPT_SEEDS = 8
"""The corrupt site's declared draws of 4000 tokens at each of TIMES (the sampling gate's): a masked
share off its chance by 0.03, the bound the site held before the gate, fails at every one of them."""


def test_corrupt_masks_the_schedules_fraction_and_keeps_the_rest(rng):
    """Each token is masked by its own draw at the schedule's chance 1 - alpha(t), every other token
    kept: the masked counts at each of TIMES, independent rows of one verdict (the sampling gate)."""
    process = DiscreteProcess(LogLinear(), mask_id=MASK)
    tokens = jax.random.randint(rng, (3, 4000), 0, MASK)
    masked, is_masked = process.corrupt(jax.random.fold_in(rng, 1), tokens, TIMES)
    assert jnp.all(jnp.where(is_masked, masked == MASK, masked == tokens))

    def counts(seed):
        masks = np.asarray(process.corrupt(jax.random.key(seed), tokens, TIMES)[1]).sum(axis=1)
        return np.stack([masks, tokens.shape[1] - masks], axis=-1)

    kept = np.asarray(process.schedule.alpha(TIMES), np.float64)
    assert_draws_as_declared(counts, np.stack([1 - kept, kept], axis=-1), "corrupt's masks",
                             seeds=CORRUPT_SEEDS)


PLACE_DRAWS = 80_000
"""The strata site's draws of eight rows' training times: at one in a billion the Kolmogorov-Smirnov test
over their 640,000 places fails a law off the uniform by 3.27 / 800, 0.0041, where a mean off a half by
0.01, the bound the site held before, moves it by at least 0.01; and the t test of two rows' places fails
a correlation of 6.1 / sqrt(PLACE_DRAWS), 0.022, against the old 0.08."""


def test_training_times_are_mdlms_antithetic_draw():
    """MDLM's `_sample_t` (kuleshov-group/mdlm @c112c52, diffusion.py:800-808,
    with configs/config.yaml's antithetic_sampling True and sampling_eps
    1e-3): row i draws its own uniform u_i in the i-th of n strata,
    (u_i + i) / n, and t = (1 - 1e-3) of that + 1e-3, so no row's weight
    1 / t exceeds 1000. Each row's place in its stratum is its own draw;
    one offset shared by the batch would put every row at the same place.
    Within a stratum the place is uniform, by the Kolmogorov-Smirnov test,
    and two rows' places unrelated, by the t test of their correlation, each
    at one in a billion."""
    process = DiscreteProcess(LogLinear(), mask_id=MASK)
    n, sampling_eps = 8, 1e-3
    keys = jax.random.split(jax.random.key(0), PLACE_DRAWS)
    draws = np.asarray(jax.vmap(lambda key: process.sample_t(key, n))(keys))
    assert draws.min() >= sampling_eps and draws.max() < 1
    # Each row's time lies in its own stratum, between the times its uniform's two ends make, as rounded.
    edges = np.asarray(process.stratified(jnp.zeros(n + 1), jnp.arange(n + 1), n))
    assert np.all((edges[:-1] <= draws) & (draws <= edges[1:]))
    places = (draws.astype(np.float64) - sampling_eps) / (1 - sampling_eps) * n - np.arange(n)
    assert np.all(np.ptp(places, axis=1) > 0)
    assert stats.kstest(places.ravel(), stats.uniform.cdf).pvalue > 1e-9
    assert stats.pearsonr(places[:, 0], places[:, 1]).pvalue > 1e-9


class Peaked(nn.Module):
    """Logits that put every position on the token equal to its index mod
    (VOCAB - 1), so a revealed token is predictable."""

    @nn.compact
    def __call__(self, tokens):
        scale = self.param("scale", nn.initializers.constant(20.0), ())
        target = jnp.arange(tokens.shape[1]) % (VOCAB - 1)
        return scale * jax.nn.one_hot(target, VOCAB)[None]


REVEALS = ((0.8, 0.3), (1.0, 0.5))
"""The reveal site's steps from t to s: one inside the trajectory and one from its fully masked end."""

REVEAL_SEEDS = 16
"""The reveal site's declared steps of 200 rows of 8 masked positions at each of REVEALS (the sampling
gate's): a revealed share off its chance by 0.03, the bound the site held before the gate, fails."""


def test_unmask_reveals_the_schedules_share_with_the_models_token(rng):
    """From t to s, a masked position is revealed with probability
    (alpha(s) - alpha(t)) / (1 - alpha(t)), its own draw (the sampling gate,
    each of REVEALS a row of one verdict), and takes the model's draw; s = t
    reveals nothing and s = 0 reveals everything."""
    process = DiscreteProcess(LogLinear(), mask_id=MASK)
    model = Peaked()
    denoise = process.denoiser(model, model.init(rng, jnp.zeros((1, 8), jnp.int32)))
    x = jnp.full((200, 8), MASK, jnp.int32)
    t = jnp.full((200,), 0.8)
    filled, log_probs = denoise(x, t)
    assert jnp.array_equal(filled, jnp.broadcast_to(jnp.arange(8) % (VOCAB - 1), (200, 8)))

    def counts(seed):
        # Each step its own key: the rows of one verdict are independent draws.
        keys = jax.random.split(jax.random.fold_in(rng, seed), len(REVEALS))
        shown = [int(jnp.sum(Unmask().step(x, jnp.full((200,), start), jnp.full((200,), end), filled,
                                           log_probs, (), key, process, denoise)[0] != MASK))
                 for key, (start, end) in zip(keys, REVEALS, strict=True)]
        return np.stack([shown, [x.size - count for count in shown]], axis=-1)

    alpha = {time: float(process.schedule.alpha(time)) for pair in REVEALS for time in pair}
    shares = np.asarray([(alpha[end] - alpha[start]) / (1 - alpha[start]) for start, end in REVEALS])
    assert_draws_as_declared(counts, np.stack([shares, 1 - shares], axis=-1), "unmask's reveals",
                             seeds=REVEAL_SEEDS)

    s = jnp.full((200,), 0.3)
    stepped, _ = Unmask().step(x, t, s, filled, log_probs, (), rng, process, denoise)
    revealed = stepped != MASK
    unobserved = True
    assert jnp.all(jnp.where(revealed, stepped == filled, unobserved))

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


NORMALIZED_SEEDS = 8
"""The bf16 normalization site's declared draws of bf16 logits (check 1's, `seeds_for`)."""


def test_the_denoiser_normalizes_bf16_logits_in_fp32():
    """The reveal draws from these log probabilities: over drawn bf16 logits, the
    tokens' but the mask's on par with float32's log-softmax of the same bf16
    values from float64's (check 1), which a bf16 reduction misses by 0.11
    where float32 misses by 9.5e-7; the mask's at minus infinity."""
    mask = 1023

    class Held(nn.Module):
        @nn.compact
        def __call__(self, tokens):
            return self.param("logits", nn.initializers.zeros, (2, 4, 1024), jnp.bfloat16)

    process = DiscreteProcess(LogLinear(), mask_id=mask)
    tokens, t = jnp.full((2, 4), mask, jnp.int32), jnp.full((2,), 0.5)

    def normalized(logits):
        return process.denoiser(Held(), {"params": {"logits": logits}})(tokens, t)[1]

    def draw(seed):
        return ((4 * jax.random.normal(jax.random.key(seed), (2, 4, 1024))).astype(jnp.bfloat16),)

    assert np.all(np.isneginf(np.asarray(jax.jit(normalized)(*draw(0)))[..., mask]))
    assert_as_exact(lambda logits: normalized(logits)[..., :mask],
                    lambda logits: jax.nn.log_softmax(logits[..., :mask].astype(jnp.float32)),
                    draw, "bf16 log-probabilities", seeds=NORMALIZED_SEEDS,
                    truth_fn=lambda logits: jax.nn.log_softmax(logits[..., :mask]))


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
    # The step revealed some positions to carry over and left some to draw (its share, the reveal site's).
    assert revealed.any() and not revealed.all()
    final, _ = Unmask().step(half, s, done, *denoise(half, s), (), jax.random.fold_in(rng, 2),
                             process, denoise)
    assert not jnp.any(final == MASK)
    np.testing.assert_array_equal(np.asarray(final)[revealed], np.asarray(half)[revealed])


def test_unmask_refuses_a_gaussian_process(rng):
    process = Process(CosineNoiseScheduler(10), EpsilonPredictionTransform())
    x = jnp.zeros((2, 4))
    with pytest.raises(ValueError):
        Unmask().step(x, jnp.ones((2,)), jnp.zeros((2,)), x, x, (), rng, process, None)


def test_the_mdlm_preset_is_registered_and_takes_no_conditions():
    assert presets["mdlm"] is MDLM and solvers["unmask"] is Unmask
    process = presets.build("mdlm", mask_id=MASK, eps=1e-2)()
    assert process.mask_id == MASK and process.schedule == LogLinear(eps=1e-2)
    with pytest.raises(ValueError):
        process.denoiser(Peaked(), {}, {"label": jnp.zeros((1,))})


############################################################################################################
# Full attention on the CausalTransformer
############################################################################################################

def transformer(causal):
    return CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=1, num_heads=2,
                                    max_seq_len=8, causal=causal)


def test_causal_false_lets_a_position_read_the_future(rng):
    tokens = jnp.array([[1, 2, 3, 4, 5, 1, 2, 3]])
    changed = tokens.at[0, 6].set(4)
    for causal in (True, False):
        model = transformer(causal)
        params = model.init(rng, tokens)
        moved = not np.array_equal(model.apply(params, tokens)[0, 0], model.apply(params, changed)[0, 0])
        assert moved is (not causal)


def test_full_attention_has_no_cache(rng):
    model = transformer(causal=False)
    params = model.init(rng, jnp.zeros((1, 8), jnp.int32))
    with pytest.raises(ValueError):
        model.apply(params, 2, method=type(model).init_cache, mutable=["cache"])


def test_the_masked_objective_refuses_a_causal_model():
    with pytest.raises(ValueError):
        MaskedDiffusionObjective(transformer(causal=True), MDLM(mask_id=MASK)(), 8)


LOSS_SEEDS = 8
"""The masked loss site's declared draws of a model, two rows and their corruption (check 1's,
`seeds_for`, paired: both arms read the same per-token terms, whose NELBO weights 1 / t run to a
thousand where a row's time is drawn near zero, so one arm's error alone resolves in no 4096 seeds)."""


def test_the_loss_is_the_mean_of_the_terms_evaluation_reports_under_the_weights_it_is_handed(rng):
    """The training loss is the batch's per-token NELBO terms averaged over
    every position, and evaluation reports those terms, zero where a token
    stayed visible, under the averaged weights it is handed, so a validation
    pass's perplexity is exp of this bound per token. The terms themselves
    are MDLM's (tests/test_mdlm_reference.py)."""
    model = transformer(causal=False)
    objective = MaskedDiffusionObjective(model, MDLM(mask_id=MASK)(), 8)
    params = objective.init(rng)
    rows = jnp.array([[1, 2, 3, 4, 5, 1, 2, 3], [3, 2, 1, 0, 4, 5, 1, 2]])
    step = Step(jnp.asarray(0), rng, None)

    terms = objective.evaluate(params, {"text": rows}, step)
    np.testing.assert_array_equal(terms.weights, 1.0)

    # Over drawn models, rows and corruptions, the loss on par with the terms' mean from its float64 run.
    # The corruption and its loss weights are the draw's, at the objective's boundary (`Step.tape`), as
    # the objective draws them in float32, so the float64 run scores the same masks under the same
    # weights and computes every term and sum in float64.
    wide = MaskedDiffusionObjective(model.clone(dtype=jnp.float64, attention_impl="reference"),
                                    MDLM(mask_id=MASK)(), 8)
    process = objective.process

    def draw(seed):
        key = jax.random.PRNGKey(seed)
        drawn = jax.random.randint(jax.random.fold_in(key, 1), rows.shape, 0, MASK)
        times = process.sample_t(jax.random.fold_in(key, 2), rows.shape[0])
        corruption = {"times": times, "mask": process.corrupt(jax.random.fold_in(key, 3), drawn, times)[1],
                      "loss weights": process.weight(times)}
        return objective.init(jax.random.fold_in(key, 0)), drawn, corruption

    def taped(corruption):
        return Step(jnp.asarray(0), jax.random.PRNGKey(0), None, tape=corruption)

    def loss(params, drawn, corruption):
        return objective.scalar_loss(params, {"text": drawn}, taped(corruption))[0]

    def mean_of_terms(of):
        def mean(params, drawn, corruption):
            return of.evaluate(params, {"text": drawn}, taped(corruption)).losses.sum() / drawn.size
        return mean

    assert_as_exact(loss, mean_of_terms(objective), draw, "masked loss",
                    seeds=LOSS_SEEDS, truth_fn=mean_of_terms(wide))

    averaged = jax.tree.map(lambda leaf: 1.5 * leaf, params)
    handed = objective.evaluate(params, {"text": rows}, Step(jnp.asarray(0), rng, averaged))
    np.testing.assert_array_equal(handed.losses, objective.evaluate(averaged, {"text": rows}, step).losses)


############################################################################################################
# Masked generation runs on what a model reads, not on its class
############################################################################################################

class Reversed(nn.Module):
    """A user's own token model, no Dew class: each slot scores how many
    slots follow it in the row, which it knows only by reading to the end."""

    vocab_size: int = VOCAB
    max_seq_len: int = 8

    def setup(self):
        self.scale = self.param("scale", nn.initializers.constant(30.0), ())

    def __call__(self, tokens, **fields):
        return self.logits(tokens, **fields)

    def logits(self, tokens, *, train=False, **fields):
        following = tokens.shape[1] - 1 - jnp.arange(tokens.shape[1])
        return self.scale * jnp.broadcast_to(jax.nn.one_hot(following % MASK, VOCAB), (*tokens.shape, VOCAB))


class Mirrored(Reversed):
    """`Reversed`, saying that it reads the whole row (`TokenModel`)."""

    causal: bool = False
    mask_token_id: int | None = None


def test_a_user_bidirectional_model_fills_a_masked_response():
    """Masked generation needs full-sequence logits that read the whole row,
    not a CausalTransformer: a user module that says so fills each response
    slot with what it scores there, the slots after it."""
    from dew.inference import MaskedGeneration

    model = Mirrored()
    task = MaskedGeneration(model, model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)),
                            DiscreteProcess(LogLinear(), mask_id=MASK), steps=8)
    generated = task([[1, 2, 3], [4, 5, 0]], 4, key=0)
    np.testing.assert_array_equal(generated.tokens, [[1, 2, 3, 3, 2, 1, 0], [4, 5, 0, 3, 2, 1, 0]])
    np.testing.assert_array_equal(generated.lengths, [4, 4])


@pytest.mark.parametrize("model,lacks", [
    (transformer(causal=True), "this CausalTransformer is causal"),
    (Reversed(), r"this Reversed does not say whether a position reads the ones after it \(TokenModel\)"),
    (Peaked(), r"this Peaked gives no full-sequence logits \(Logits\)")])
def test_masked_generation_names_what_a_model_lacks(rng, model, lacks):
    """A causal decoder has every capability masked generation reads but one
    of its positions never sees the masked ones after it; a model that does
    not say how it reads, or gives no full-sequence logits, is refused for
    that, by name."""
    process = DiscreteProcess(LogLinear(), mask_id=MASK)
    with pytest.raises(ValueError):
        process.generate(model, model.init(rng, jnp.zeros((1, 4), jnp.int32)), [[1, 2]], 3, key=0)


def test_a_masked_walk_whose_model_scores_nothing_refuses_rather_than_reveal_a_token(rng):
    """A model whose scores are NaN holds no distribution to reveal a token from, so masked generation
    raises instead of committing an arbitrary one; the same model with finite scores generates."""
    model = transformer(causal=False)
    variables = model.init(rng, jnp.zeros((1, 4), jnp.int32))
    process = DiscreteProcess(LogLinear(), mask_id=MASK)
    process.generate(model, variables, [[1, 2]], 3, key=0, steps=4)
    poisoned = jax.tree.map(lambda leaf: jnp.full_like(leaf, jnp.nan), variables)
    with pytest.raises(ValueError):
        process.generate(model, poisoned, [[1, 2]], 3, key=0, steps=4)


def test_masked_generation_reads_its_ids_as_every_parser_of_ids_does(rng):
    """NumPy integers are token ids as Python's are (`dew.batch.token_id`), so a request built
    from a tokenizer's arrays generates as one built from Python ints; a flag or a fraction where
    an id goes is refused."""
    model = transformer(causal=False)
    variables = model.init(rng, jnp.zeros((1, 4), jnp.int32))
    process = DiscreteProcess(LogLinear(), mask_id=MASK)
    python = process.generate(model, variables, [[1, 2]], 3, key=0, pad_token_id=0, eos_token_ids=(3,))
    arrays = process.generate(model, variables, [[1, 2]], 3, key=0, pad_token_id=np.int64(0),
                              eos_token_ids=(np.int32(3),))
    np.testing.assert_array_equal(arrays.tokens, python.tokens)
    for wrong in ({"pad_token_id": True}, {"eos_token_ids": (3.0,)}):
        with pytest.raises(ValueError):
            process.generate(model, variables, [[1, 2]], 3, key=0, **wrong)


class FieldReader(nn.Module):
    """A user's bidirectional model that spells out what each slot is handed:
    a text slot scores its logical position, one past it when the row
    carries pixels, and a media slot scores 14."""

    vocab_size: int = 16
    max_seq_len: int = 16
    causal: bool = False
    mask_token_id: int | None = None

    def setup(self):
        self.scale = self.param("scale", nn.initializers.constant(30.0), ())

    def __call__(self, tokens, **fields):
        return self.logits(tokens, **fields)

    def logits(self, tokens, *, train=False, positions=None, image_indices=None, conditioning=None,
               **fields):
        text = (positions + (conditioning is not None)) % 14
        target = text if image_indices is None else jnp.where(image_indices < 0, text, 14)
        return self.scale * jax.nn.one_hot(target, self.vocab_size)


def test_a_masked_response_continues_the_prompts_fields_and_keeps_its_media():
    """A media prompt fills a masked response too (`ModelInputs.extended`):
    the response's slots are text at the logical positions after the
    prompt's last real token, and the row keeps its pixels. Each row here
    reads back as its next positions, shifted by the pixels it holds; a
    left-padded row with an image continues from its own positions."""
    from dew.nn.inputs import ModelInputs

    model = FieldReader()
    inputs = ModelInputs(jnp.array([[0, 1, 3], [2, 4, 6]]),
                         {"attention_mask": jnp.array([[False, True, True], [True, True, True]]),
                          "positions": jnp.array([[0, 7, 8], [0, 1, 2]]),
                          "image_indices": jnp.array([[-1, 0, -1], [-1, -1, -1]])},
                         {"pixel_values": jnp.ones((2, 1, 3, 2, 2))})
    variables = model.init(jax.random.key(0), inputs.tokens, **inputs.kwargs())
    generated = DiscreteProcess(LogLinear(), mask_id=15).generate(model, variables, inputs, 3, key=0, steps=4)
    np.testing.assert_array_equal(generated.tokens, [[0, 1, 3, 10, 11, 12], [2, 4, 6, 4, 5, 6]])
    unruled = ModelInputs(inputs.tokens, {"token_type_ids": jnp.zeros((2, 3), jnp.int32)})
    with pytest.raises(ValueError):
        DiscreteProcess(LogLinear(), mask_id=15).generate(model, variables, unruled, 3, key=0)


def test_a_prompt_laid_out_by_its_coordinates_generates_as_its_fields_do(rng):
    """A request's `Coordinates` continue into its masked response as every other
    extension continues them (`ModelInputs.extended`): a left-padded prompt at
    given positions generates, token for token, what the same prompt as a
    validity and a positions field does, its response's positions read from and
    written to its `Coordinates` (audit AT-04)."""
    from dew.batch import COORDINATES_FIELD, coordinates
    from dew.nn.inputs import ModelInputs

    model = transformer(causal=False)
    tokens = jnp.array([[0, 1, 3], [2, 4, 5]])
    valid = jnp.array([[False, True, True], [True, True, True]])
    positions = jnp.array([[0, 2, 3], [0, 1, 2]])
    fields = ModelInputs(tokens, {"attention_mask": valid, "positions": positions})
    laid = ModelInputs(tokens, {COORDINATES_FIELD: coordinates((2, 3), valid=valid, positions=positions)})
    variables = model.init(rng, tokens)
    process = DiscreteProcess(LogLinear(), mask_id=MASK)
    expected = process.generate(model, variables, fields, 3, key=0, steps=4)
    generated = process.generate(model, variables, laid, 3, key=0, steps=4)
    np.testing.assert_array_equal(generated.tokens, expected.tokens)
    np.testing.assert_array_equal(generated.lengths, expected.lengths)


def test_a_packed_prompt_continues_its_last_document_as_if_alone():
    """A packed prompt numbers its documents from 1 and its padding 0
    (`dew.batch.real_slots`), and its tokens sit at their places in their
    documents (`Coordinates.places`): a row of two documents and a row padded
    by segment 0 each continue their last document's positions, reading back
    what that document alone does."""
    from dew.nn.inputs import ModelInputs

    model = FieldReader()
    process = DiscreteProcess(LogLinear(), mask_id=15)
    packed = ModelInputs(jnp.array([[1, 2, 3, 4, 6], [3, 4, 6, 0, 0]]),
                         {"segment_ids": jnp.array([[1, 1, 2, 2, 2], [1, 1, 1, 0, 0]])})
    alone = ModelInputs(jnp.array([[3, 4, 6], [3, 4, 6]]))
    variables = model.init(jax.random.key(0), alone.tokens, positions=jnp.zeros((2, 3), jnp.int32))
    generated = process.generate(model, variables, packed, 3, key=0, steps=4)
    expected = process.generate(model, variables, alone, 3, key=0, steps=4)
    np.testing.assert_array_equal(np.asarray(generated.tokens)[:, 5:], np.asarray(expected.tokens)[:, 3:])
    np.testing.assert_array_equal(generated.lengths, expected.lengths)


def test_a_bidirectional_media_wrapper_fills_a_response_after_an_image():
    """A multimodal wrapper around a bidirectional decoder answers the same
    capabilities its decoder does, so it fills a masked response after an
    image prompt: the prompt stays, and the response holds real tokens."""
    from dew.inference import MaskedGeneration
    from dew.nn.inputs import ModelInputs
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision import GemmaProjector, SiglipVision

    text = CausalTransformer(vocab_size=32, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                             mlp_features=32, max_seq_len=16, causal=False, mask_token_id=31)
    model = MultimodalTransformer(
        text, SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2,
                           image_size=8, patch_size=4),
        GemmaProjector(text_width=16, patches_per_side=2, tokens_per_side=1),
        family="gemma3", image_token_id=1)
    tokens = jnp.array([[2, 1, 3, 4]])
    inputs = ModelInputs(tokens, {"image_indices": jnp.where(tokens == 1, 0, -1)},
                         {"pixel_values": jnp.linspace(-0.5, 0.5, 3 * 8 * 8).reshape(1, 1, 3, 8, 8)})
    task = MaskedGeneration(model, model.init(jax.random.key(0), tokens, **inputs.kwargs()),
                            DiscreteProcess(LogLinear(), mask_id=31), steps=4)
    generated = task(inputs, 5, key=0).host()
    np.testing.assert_array_equal(generated.tokens[:, :4], tokens)
    assert generated.tokens.shape == (1, 9) and not np.any(generated.tokens == 31)
    np.testing.assert_array_equal(generated.lengths, [5])


def test_a_media_wrapper_around_a_shifted_decoder_reads_the_logits_its_decoder_reads():
    """A multimodal wrapper around a decoder whose predictions are the position before's logits
    (Dream's, `shifted_logits`) reads its scores as its decoder does: over text alone, the masked
    walk's unmasking distribution and the masked objective's loss are the decoder's own."""
    from dew.nn.multimodal import MultimodalTransformer
    from dew.nn.vision import GemmaProjector, SiglipVision
    from dew.objectives.base import Step
    from dew.objectives.diffusion import MaskedDiffusionObjective

    text = CausalTransformer(vocab_size=32, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                             mlp_features=32, max_seq_len=16, causal=False, mask_token_id=31,
                             shifted_logits=True)
    model = MultimodalTransformer(
        text, SiglipVision(hidden_size=16, intermediate_size=32, num_layers=1, num_heads=2,
                           image_size=8, patch_size=4),
        GemmaProjector(text_width=16, patches_per_side=2, tokens_per_side=1),
        family="gemma3", image_token_id=1)
    tokens = jnp.array([[2, 31, 3, 31, 4, 31]])
    variables = model.init(jax.random.key(0), tokens)
    alone = {"params": variables["params"]["language_model"]}
    process = DiscreteProcess(LogLinear(), mask_id=31)
    _, wrapped = process.denoiser(model, variables)(tokens, jnp.ones((1,)))
    _, decoded = process.denoiser(text, alone)(tokens, jnp.ones((1,)))
    np.testing.assert_array_equal(wrapped, decoded)
    batch, step = {"text": np.array([[2, 5, 3, 6, 4, 7]])}, Step(jnp.array(0), jax.random.key(1), None)
    wrapped_loss, decoded_loss = (
        MaskedDiffusionObjective(network, process, 6, ema_decay=None).loss(held, batch, step)[0]
        for network, held in ((model, variables), (text, alone)))
    for leaf, own in zip(jax.tree.leaves(wrapped_loss), jax.tree.leaves(decoded_loss), strict=True):
        np.testing.assert_array_equal(leaf, own)


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
    model = CausalTransformer(vocab_size=257, emb_features=64, num_layers=2, num_heads=4,
                                     max_seq_len=ROW, causal=False)
    objective = MaskedDiffusionObjective(model, process, ROW, steps=48, samples=16)

    trainer = Trainer(objective, optax.adam(3e-3), key=jax.random.PRNGKey(0))
    state = trainer.fit(Loader(train=lambda partition: corpus_batches(), val=None, records=None, batch=16),
                        steps=1000, log_every=500)
    params = state.variables

    loss, aux = objective.scalar_loss(
        params, {"text": ROWS}, Step(state.microstep, jax.random.PRNGKey(1), None)
    )
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


@pytest.mark.parametrize("impl", ["rbg", "unsafe_rbg", "threefry2x32"])
def test_a_rows_reveals_are_its_own_under_any_key_implementation(impl):
    """`Unmask.step_rows` draws each row's reveals and tokens under its own key, the same whatever rows it
    steps beside: `vmap` would draw every row of an RBG key implementation from the first key's stream,
    so each row key is made counter-based first (`dew._distributions.counted`)."""
    process, model, rows = DiscreteProcess(LogLinear(), mask_id=MASK), Peaked(), 6
    denoise = process.denoiser(model, model.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32)))
    x = jnp.full((rows, 8), MASK, jnp.int32)
    t, s = jnp.full((rows,), 0.9), jnp.full((rows,), 0.4)
    filled, log_probs = (jnp.broadcast_to(value, (rows, *value.shape[1:])) for value in denoise(x, t))
    keys = jax.random.split(jax.random.key(3, impl=impl), rows)

    def stepped(part):
        return np.asarray(Unmask().step_rows(x[part], t[part], s[part], filled[part], log_probs[part], (),
                                             keys[part], process, denoise)[0])

    alone = np.concatenate([stepped(slice(row, row + 1)) for row in range(rows)])
    np.testing.assert_array_equal(stepped(slice(None)), alone)
