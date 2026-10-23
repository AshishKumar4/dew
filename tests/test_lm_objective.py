"""Language modelling: the shifted cross entropy, what evaluation produces,
and the objective through the general trainer.

The model here is a small causal stack that honors the backbone's contract
(int32 ids in, float32 logits out, a `train` flag for dropout, the head split
off behind `hidden_states` and `output_table`) and `dew.sampling.text.generate`
is recorded, not run, so what is under test is the objective: that the
loss is the cross entropy of the shifted sequence, that padding is excluded
only when a pad id is named, that a pass is scored per token, and that the
trainer drives it on both a data-parallel and an FSDP mesh. The real
sampler runs in test_lm_recipe.
"""


import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from affine_run import dataset
from recording import RecordingTracker
from reference_error import assert_as_exact, widened, without_ynn
from token_support import token_batch_of

from dew.artifacts import TokenScores
from dew.data.chat import ROLES_KEY, Role
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.inputs import ModelInputs, coordinates
from dew.objectives.base import Step
from dew.objectives.lm import TEXT_KEY, LMObjective, Perplexity, Samples
from dew.sampling import Sampling
from dew.training import Checkpoints, Layout, MeshSpec, Trainer
from dew.training.memory import climb_to

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh


VOCAB = 8
SEQ = 16
BATCH = 8
STEPS = 200
# The test model's parameters are far below the production shard threshold, so
# the threshold is lowered; at the production value "FSDP on" would replicate
# everything.
TINY = 64


def tiny_decoder(**options):
    """The native decoder at test size: two layers of 16 features in float32,
    attention through XLA so the CPU runs the same kernel everywhere."""
    return CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=2, num_heads=2, head_dim=8,
                             mlp_features=32, max_seq_len=64, dtype="float32", attention_impl="xla",
                             **options)


def make_objective(seq=SEQ, model=None, **kwargs):
    return LMObjective(model or tiny_decoder(), seq, **kwargs)


def step_at(index=0, key=1, ema=None):
    return Step(step=jnp.asarray(index), key=jax.random.key(key), ema=ema)


def token_batch(batch=4, seq=SEQ, seed=0):
    return token_batch_of(VOCAB, seq, batch, seed)


def cycle_batches(batch=BATCH, seq=SEQ, seed=0):
    """Rows counting upwards mod the vocabulary from a random offset.

    Every target is the input token plus one, so a model that learns to read
    the token in front of it drives the loss to zero, while one that only
    learns the marginal distribution sits at log(VOCAB).
    """
    rs = np.random.RandomState(seed)
    positions = np.arange(seq + 1)[None, :]
    while True:
        offsets = rs.randint(0, VOCAB, (batch, 1))
        yield {TEXT_KEY: ((offsets + positions) % VOCAB).astype(np.int32)}


def reference_cross_entropy(logits, targets, pad_id=None, weights=None):
    """The shifted cross entropy in NumPy at the logits' own width, from the model's own logits: each
    target's negative log softmax, averaged over the targets `weights` counts (those past `pad_id`,
    or all)."""
    logits = np.asarray(logits)
    largest = logits.max(axis=-1, keepdims=True)
    log_probs = logits - largest - np.log(np.exp(logits - largest).sum(axis=-1, keepdims=True))
    picked = -np.take_along_axis(log_probs, np.asarray(targets)[..., None], axis=-1)[..., 0]
    if weights is None:
        weights = np.ones_like(picked) if pad_id is None else np.asarray(targets) != pad_id
    weights = np.asarray(weights, logits.dtype)
    return (picked * weights).sum() / weights.sum()


ROWS = 32
"""A check-1 draw's rows: 32 of SEQ targets, where `token_batch`'s 4 leave the reference's own rounding
of a scalar resolving slowly (the z-loss gradient site's sizing, `seeds_for`)."""

LM_SEEDS = {
    "loss": 16,
    "validity attention_mask": 16,
    "validity coordinates": 16,
    "padded loss": 16,
    "perplexity": 16,
    "evaluation": 16,
    "ema evaluation": 16,
    "scores with samples": 16,
    "perplexity of counted targets": 16,
    "perplexity of the mean": 16,
    "validation perplexity": 16,
    "packed scores": 16,
    "packed against alone": 16,
    "packed against its documents' heads": 16,
    "packed batch dict": 16,
    "loss role": 16,
    "roles ignored": 16,
    "z loss": 16,
    "z-loss cross entropy": 16,
    "router z-loss": 16,
    "prediction": 16,
    "tiled head fused 4": 16,
    "tiled head unfused 4": 16,
    "tiled head fused 16": 16,
    "tiled head unfused 16": 16,
}
"""Check 1's declared draws by site (`seeds_for`): each draw is the objective's parameters initialized
from the seed's key and the site's batch drawn from the seed."""


def forwards(model):
    """`model.apply`, compiled once at the model's width and once with the model widened to float64
    (`clone(dtype=jnp.float64)`), the program chosen by its parameters' width: a reference reads its
    logits through it, and its truth the widened program's on widened parameters."""
    compiled = {np.dtype(np.float32): jax.jit(model.apply),
                np.dtype(np.float64): jax.jit(model.clone(dtype=jnp.float64).apply)}

    def apply(params, *arguments, **options):
        return compiled[np.dtype(jax.tree.leaves(params)[0].dtype)](params, *arguments, **options)
    return apply


def initialized(objective):
    """The objective's initialization, compiled once: a draw's parameters from its seed's key."""
    init = jax.jit(objective.init)
    return lambda seed: init(jax.random.key(seed))


def assert_objective_as_exact(objective, label, batch_of, expected, dew=None, *, step=None):
    """`dew(params, batch)`, by default the objective's loss of the batch, on par from float64 with
    `expected(forward, params, batch)`, a NumPy closed form over the model's own logits read through
    `forward` (`forwards`), over draws of the objective's parameters and `batch_of(seed)` (check 1,
    `LM_SEEDS[label]`)."""
    step = step_at() if step is None else step
    dew = dew or (lambda params, batch: objective.scalar_loss(params, batch, step)[0])
    forward, init = forwards(objective.model), initialized(objective)
    assert_as_exact(jax.jit(dew), lambda params, batch: expected(forward, params, batch),
                    lambda seed: (init(seed), batch_of(seed)), label, seeds=LM_SEEDS[label])


def shifted(forward, params, batch, **weighting):
    """The cross entropy of the batch's targets over the logits of the inputs before them."""
    tokens = np.asarray(batch[TEXT_KEY])
    return reference_cross_entropy(forward(params, jnp.asarray(tokens[:, :-1])), tokens[:, 1:], **weighting)


ARGMAX_MARGIN = 1e-4
"""The least gap a draw admits between a position's two largest float64 logits where an accuracy is held
exactly: a hundred times the float32 rounding of logits of order one, so two programs' argmaxes agree.
Below it the argmax is a tie the claim does not decide."""


def clear_of_ties(objective, batch_of, tries=64):
    """The first of seeds 0 to `tries` whose drawn parameters and batch hold every position's two largest
    float64 logits `ARGMAX_MARGIN` apart, and those parameters and that batch."""
    forward, init = forwards(objective.model), initialized(objective)
    for seed in range(tries):
        params, batch = init(seed), batch_of(seed)
        with jax.enable_x64(new_val=True):
            logits = np.asarray(forward(widened(params), jnp.asarray(batch[TEXT_KEY])[:, :-1]))
        top = np.sort(logits, axis=-1)
        if np.min(top[..., -1] - top[..., -2]) > ARGMAX_MARGIN:
            return params, batch
    raise AssertionError(f"no draw of {tries} clears every argmax by {ARGMAX_MARGIN}")


def accuracy_of(logits, targets, kept=None):
    """The argmax accuracy over the targets `kept` counts (all), as float32 divides the two counts."""
    hits = np.argmax(np.asarray(logits), axis=-1) == targets
    kept = np.ones_like(hits) if kept is None else kept
    return np.float32((hits & kept).sum()) / np.float32(kept.sum())


# --- the loss --------------------------------------------------------------

def test_loss_is_the_cross_entropy_of_the_shifted_sequence():
    """The loss is the cross entropy of each target given the tokens before it, on par with NumPy's over
    the model's own logits (check 1); the unshifted cross entropy, the mistake this guards, differs."""
    objective = make_objective()
    assert_objective_as_exact(objective, "loss", lambda seed: token_batch(ROWS, seed=seed), shifted)

    params, batch = objective.init(jax.random.key(0)), token_batch(ROWS)
    tokens = np.asarray(batch[TEXT_KEY])
    logits = objective.model.apply(params, jnp.asarray(tokens[:, :-1]))
    assert reference_cross_entropy(logits, tokens[:, 1:]) != reference_cross_entropy(logits, tokens[:, :-1])


@pytest.mark.parametrize("given", ["attention_mask", "coordinates"])
def test_the_validity_the_decoder_reads_decides_which_targets_count(given):
    """Three tokens whose last is marked invalid, by the attention mask or
    by the coordinates' validity the decoder reads: the loss is the cross
    entropy of the one target between valid tokens, which the model gives
    the row of its first token alone (check 1, over drawn ids)."""
    objective = make_objective(seq=2)
    valid = jnp.asarray([[True, True, False]] * ROWS)

    def batch_of(seed):
        ids = jnp.asarray(np.random.RandomState(seed).randint(0, VOCAB, (ROWS, 3)), jnp.int32)
        return ({TEXT_KEY: ModelInputs(ids, {"attention_mask": valid})} if given == "attention_mask"
                else {TEXT_KEY: ids, "text_coordinates": coordinates(valid=valid)})

    def expected(forward, params, batch):
        ids = np.asarray(batch[TEXT_KEY].tokens if given == "attention_mask" else batch[TEXT_KEY])
        return reference_cross_entropy(forward(params, jnp.asarray(ids[:, :1])), ids[:, 1:2])

    assert_objective_as_exact(objective, f"validity {given}", batch_of, expected)


def padded_batch(rows=ROWS, seed=0):
    """`token_batch` with every fourth token from the third the pad id, 0."""
    tokens = np.asarray(token_batch(rows, seed=seed)[TEXT_KEY]).copy()
    tokens[:, 3::4] = 0
    return {TEXT_KEY: jnp.asarray(tokens)}


def test_padded_targets_are_left_out_of_the_average():
    """A named pad id leaves its targets out: the loss is the cross entropy of the rest (check 1), which
    the padded targets would move."""
    pad_id = 0
    objective = make_objective(pad_id=pad_id)
    assert_objective_as_exact(objective, "padded loss", lambda seed: padded_batch(seed=seed),
                              lambda forward, params, batch: shifted(forward, params, batch, pad_id=pad_id))

    params, batch = objective.init(jax.random.key(0)), padded_batch()
    tokens = np.asarray(batch[TEXT_KEY])
    logits = objective.model.apply(params, jnp.asarray(tokens[:, :-1]))
    assert reference_cross_entropy(logits, tokens[:, 1:], pad_id=pad_id) != reference_cross_entropy(
        logits, tokens[:, 1:]), "masking made no difference to check"


def test_aux_reports_perplexity_and_token_accuracy():
    """The aux reports the loss as its cross entropy, the same value of the same program, its exponential
    as the perplexity (check 1), and the argmax accuracy, exactly, on a draw whose argmaxes clear any tie
    by `ARGMAX_MARGIN`."""
    objective = make_objective()
    params, batch = objective.init(jax.random.key(0)), token_batch()

    loss, aux = objective.scalar_loss(params, batch, step_at())

    assert set(aux.metrics) == {"ce", "perplexity", "token_accuracy"}
    assert aux.variables is None
    np.testing.assert_array_equal(aux.metrics["ce"], loss)
    assert_objective_as_exact(
        objective, "perplexity", lambda seed: token_batch(ROWS, seed=seed),
        lambda forward, params, batch: np.exp(shifted(forward, params, batch)),
        lambda params, batch: objective.scalar_loss(params, batch, step_at())[1].metrics["perplexity"])

    params, batch = clear_of_ties(objective, lambda seed: token_batch(seed=seed))
    _, aux = objective.scalar_loss(params, batch, step_at())
    tokens = np.asarray(batch[TEXT_KEY])
    logits = objective.model.apply(params, jnp.asarray(tokens[:, :-1]))
    assert np.float32(aux.metrics["token_accuracy"]) == accuracy_of(logits, tokens[:, 1:])


def test_dropout_runs_on_the_step_key():
    """The loss passes the step key as the dropout rng, so a model with
    dropout is stochastic across keys and reproducible under one: the same
    program on the same key gives the same bits, another key others."""
    objective = make_objective(model=tiny_decoder(dropout_rate=0.5))
    params = objective.init(jax.random.key(0))
    batch = token_batch()

    first, _ = objective.scalar_loss(params, batch, step_at(key=1))
    again, _ = objective.scalar_loss(params, batch, step_at(key=1))
    other, _ = objective.scalar_loss(params, batch, step_at(key=2))

    np.testing.assert_array_equal(first, again)
    assert float(first) != float(other)


def test_padded_tokens_are_left_out_of_the_accuracy_too():
    """The accuracy counts the targets past the pad id alone, exactly, on a draw whose argmaxes clear any
    tie by `ARGMAX_MARGIN`."""
    objective = make_objective(pad_id=0)
    params, batch = clear_of_ties(objective, lambda seed: padded_batch(4, seed=seed))
    targets = np.asarray(batch[TEXT_KEY][:, 1:])

    _, aux = objective.scalar_loss(params, batch, step_at())

    logits = objective.model.apply(params, batch[TEXT_KEY][:, :-1])
    assert np.float32(aux.metrics["token_accuracy"]) == accuracy_of(logits, targets, targets != 0)


# --- what the trainer needs ------------------------------------------------

def test_pretrained_weights_are_what_init_returns():
    """Continued pretraining: the trainer's whole initial state is the
    checkpoint, not a fresh init that happens to have the same shapes."""
    objective = make_objective()
    trained = objective.init(jax.random.key(0))
    loaded = jax.tree.map(lambda leaf: leaf + 1.0, trained)

    resumed = make_objective(variables=loaded).init(jax.random.key(1))

    for restored, expected in zip(jax.tree.leaves(resumed), jax.tree.leaves(loaded), strict=True):
        assert jnp.array_equal(restored, expected)

def make_trainer(tmp_path=None, fsdp=1, learning_rate=3e-3, tracker=None, **objective_kwargs):
    return Trainer(
        make_objective(**objective_kwargs),
        optax.adam(learning_rate),
        key=jax.random.key(0),
        mesh=MeshSpec(fsdp=fsdp),
        layout=Layout(min_shard=TINY),
        checkpoints=None if tmp_path is None else Checkpoints(str(tmp_path / "lm")),
        tracker=tracker,
    )


@pytest.mark.parametrize("fsdp", [1, 4])
def test_the_objective_trains_through_the_trainer(tmp_path, fsdp):
    """The whole seam on the simulated mesh: 8x1 data parallel, then 2x4 FSDP.

    Compiling is not the claim. The loss on the copy task has to actually fall,
    the parameters have to stay on the mesh they were built on, and the run has
    to leave a checkpoint behind.
    """
    trainer = make_trainer(tmp_path, fsdp=fsdp)
    batch = next(cycle_batches(seed=7))
    scored = jax.jit(lambda params: trainer.objective.scalar_loss(params, batch, step_at())[0])
    before = float(scored(trainer.initial_state().variables))

    state = trainer.fit(dataset(cycle_batches), steps=STEPS, log_every=50)
    after = float(scored(state.variables))

    specs = [p.sharding.spec for p in jax.tree.leaves(state.variables)]
    if fsdp > 1:
        assert any('fsdp' in str(spec) for spec in specs), "no parameter was sharded"
    assert before > 1.5, "the untrained model already knew the task"
    assert after < 0.3, f"the loss did not fall on the copy task: {before} -> {after}"
    assert Checkpoints(str(tmp_path / "lm")).latest == STEPS


# --- evaluation ------------------------------------------------------------

def test_evaluation_correctness_is_the_same_full_forward_argmax():

    model = CausalTransformer(vocab_size=8, emb_features=16, num_layers=1, num_heads=2,
                              mlp_features=32, max_seq_len=8, attention_impl="reference")
    objective = LMObjective(model, seq_len=4, ema_decay=None, token_accuracy=False)
    tokens = jnp.asarray([[1, 2, 3, 4, 1], [4, 3, 2, 1, 4]], jnp.int32)
    params = objective.init(jax.random.key(0))
    scores = objective.evaluate(params, {"text": tokens}, Step(jnp.asarray(0), jax.random.key(1), None))
    logits = model.apply(params, tokens[:, :-1], train=False)
    expected = jnp.argmax(logits, axis=-1) == tokens[:, 1:]
    np.testing.assert_array_equal(np.asarray(scores.correct), np.asarray(expected))
    assert scores.correct.shape == scores.losses.shape == scores.weights.shape


def test_evaluation_scores_every_target_of_the_batch():
    """Evaluation scores every target of the batch, each counted once, and their mean is the cross
    entropy of the shifted sequence (check 1)."""
    objective = make_objective()
    params, batch = objective.init(jax.random.key(0)), token_batch()

    scores = objective.evaluate(params, batch, step_at())

    assert scores.losses.shape == scores.weights.shape == (4, SEQ)
    np.testing.assert_array_equal(scores.weights, 1.0)
    def scored(params, batch):
        return jnp.mean(objective.evaluate(params, batch, step_at()).losses)

    assert_objective_as_exact(objective, "evaluation", lambda seed: token_batch(ROWS, seed=seed), shifted,
                              scored)


def test_evaluation_reads_the_ema_copy():
    """Evaluation under an EMA scores the EMA copy: the cross entropy of its logits (check 1, over drawn
    parameters and an EMA a unit from them), which the live weights' scores are not."""
    objective = make_objective()
    forward, init = forwards(objective.model), initialized(objective)

    def draw(seed):
        params = init(seed)
        return params, jax.tree.map(lambda leaf: leaf + 1, params), token_batch(ROWS, seed=seed)

    @jax.jit
    def averaged(params, ema, batch):
        return jnp.mean(objective.evaluate(params, batch, step_at(ema=ema)).losses)

    assert_as_exact(averaged, lambda params, ema, batch: shifted(forward, ema, batch), draw, "ema evaluation",
                    seeds=LM_SEEDS["ema evaluation"])
    params, ema, batch = draw(0)
    live = jnp.mean(objective.evaluate(params, batch, step_at()).losses)
    assert float(live) != float(shifted(forward, ema, batch))


def test_scoring_with_samples_configured_does_not_decode():
    """Scoring never decodes a preview, and its compiled losses are the eager per-token scores' on par
    from float64 (check 1)."""
    def fail_decode(ids):
        raise AssertionError("scoring decoded a preview")

    samples = Samples(prompt=[1, 2, 3], max_new_tokens=2, decode=fail_decode)
    objective = make_objective(samples=samples)
    wide = make_objective(model=tiny_decoder().clone(dtype=jnp.float64), samples=samples)
    init = initialized(objective)

    def eager(params, batch):
        scoring = wide if np.dtype(jax.tree.leaves(params)[0].dtype) == np.float64 else objective
        return scoring.token_scores(params, batch[TEXT_KEY]).losses

    assert_as_exact(jax.jit(lambda params, batch: objective.evaluate(params, batch, step_at()).losses), eager,
                    lambda seed: (init(seed), token_batch(ROWS, seed=seed)), "scores with samples",
                    seeds=LM_SEEDS["scores with samples"])


def test_preview_generates_reproducible_text_from_ema():

    model = CausalTransformer(vocab_size=8, emb_features=16, num_layers=1,
                              num_heads=2, mlp_features=32, max_seq_len=32,
                              dtype="float32", attention_impl="xla")
    objective = LMObjective(
        model,
        seq_len=SEQ,
        samples=Samples(prompt=[1, 2, 3], max_new_tokens=4, sampling=Sampling(temperature=0.0)),
    )
    params = objective.init(jax.random.key(0))
    ema = jax.tree.map(lambda leaf: leaf + 0.1, params)
    averaged = objective.preview(params, token_batch(), step_at(key=5, ema=ema))
    direct = objective.preview(ema, token_batch(), step_at(key=5))
    np.testing.assert_array_equal(averaged.tokens, direct.tokens)
    assert averaged.texts == direct.texts
    np.testing.assert_array_equal(averaged.tokens[:, :3], [[1, 2, 3]])
    assert averaged.tokens.shape == (1, 7)


def test_a_preview_over_prompts_of_different_lengths_draws_what_each_draws_alone():
    """Two prompts of three and five ids preview together, left-padded as one
    request: each row holds the ids its prompt draws alone, and each text is
    that prompt and its draw, with no padding."""
    model = CausalTransformer(vocab_size=8, emb_features=16, num_layers=1,
                              num_heads=2, mlp_features=32, max_seq_len=32,
                              dtype="float32", attention_impl="xla")
    prompts = [[1, 2, 3], [4, 5, 6, 7, 1]]

    def objective_for(prompt):
        return LMObjective(model, seq_len=SEQ, samples=Samples(
            prompt=prompt, max_new_tokens=4, sampling=Sampling(temperature=0.0), decode=lambda ids: str(ids)))

    together = objective_for(prompts)
    params = together.init(jax.random.key(0))
    drawn = together.preview(params, token_batch(), step_at(key=5))
    # One prompt as a flat array previews as one row, as a list does.
    flat = objective_for(jnp.asarray(prompts[0])).preview(params, token_batch(), step_at(key=5))
    np.testing.assert_array_equal(flat.tokens, objective_for(prompts[0]).preview(
        params, token_batch(), step_at(key=5)).tokens)
    for index, prompt in enumerate(prompts):
        alone = objective_for(prompt).preview(params, token_batch(), step_at(key=5))
        np.testing.assert_array_equal(drawn.tokens[index, -len(prompt) - 4:], alone.tokens[0])
        assert drawn.texts[index] == alone.texts[0] == str(alone.tokens[0].tolist())
    assert drawn.prompt == str(prompts[0])


# --- perplexity ------------------------------------------------------------

def scores_of(losses, weights):
    """A pass's per-token artifact over `losses` and `weights`, none of its targets correct."""
    return TokenScores(losses=jnp.asarray(losses), weights=jnp.asarray(weights),
                       correct=jnp.zeros(np.shape(losses), bool))


def perplexity_of(*batches):
    """exp of the counted targets' mean cross entropy over `(losses, weights)` batches, in NumPy at the
    losses' own width: the closed form the metric is held to."""
    total = sum((losses * weights).sum() for losses, weights in batches)
    count = sum(weights.sum() for _, weights in batches)
    return np.exp(total / count)


def test_perplexity_weighs_every_batch_by_its_counted_targets():
    """exp(sum(loss * weight) / sum(weight)) over the pass: a batch with more
    counted targets moves the score more, which the mean of per-batch means
    gets wrong the moment counts differ. On par with the closed form over
    drawn losses and counts (check 1)."""
    metric = Perplexity()
    assert metric.reads is TokenScores

    def draw(seed):
        rng = np.random.default_rng([0x9E7, seed])
        losses = rng.exponential(2.0, (2, ROWS, SEQ)).astype(np.float32)
        weights = (rng.random((2, ROWS, SEQ)) < [[[0.9]], [[0.2]]]).astype(np.float32)
        return losses[0], weights[0], losses[1], weights[1]

    def score(heavy, heavy_weights, light, light_weights):
        return metric.finalize(metric.merge(metric(scores_of(heavy, heavy_weights), {}),
                                            metric(scores_of(light, light_weights), {})))

    assert_as_exact(score, lambda *values: perplexity_of(values[:2], values[2:]), draw,
                    "perplexity of counted targets", seeds=LM_SEEDS["perplexity of counted targets"])
    # The mean of the per-batch means is not it: a batch of four counted losses of 1 and one of a
    # single counted 3.
    heavy = scores_of(np.full((1, 4), 1.0, np.float32), np.ones((1, 4), np.float32))
    light = scores_of(np.full((1, 4), 3.0, np.float32), np.asarray([[1.0, 0, 0, 0]], np.float32))
    assert metric.finalize(metric.merge(metric(heavy, {}), metric(light, {}))) != np.exp(2.0)


def test_perplexity_is_exp_of_the_mean_cross_entropy_not_the_mean_of_exps():
    """exp of the mean, not the mean of the exponentials: on par with the closed form over drawn
    batches of equal counts (check 1), which the mean of exps of two cross entropies is not."""
    metric = Perplexity()

    def draw(seed):
        rng = np.random.default_rng([0x9E8, seed])
        return tuple(rng.exponential(2.0, (ROWS, SEQ)).astype(np.float32) for _ in range(2))

    def score(first, second):
        return metric.finalize(metric.merge(*(metric(scores_of(losses, np.ones_like(losses)), {})
                                              for losses in (first, second))))

    assert_as_exact(score, lambda first, second: perplexity_of((first, np.ones_like(first)),
                                                               (second, np.ones_like(second))),
                    draw, "perplexity of the mean", seeds=LM_SEEDS["perplexity of the mean"])
    assert np.exp(np.mean([0.0, 2.0])) != np.mean(np.exp([0.0, 2.0]))


def test_the_validation_pass_scores_perplexity_per_token_and_logs_it():
    """Objective artifacts through the trainer's loop and into the metric:
    two batches of different padding weigh by their counted targets. Each
    draw is a run validating its drawn batches, whose logged perplexity is on
    par with exp of the counted targets' mean cross entropy over the run's
    own state (check 1): the run computes Dew's value, so the draw carries it."""
    wide = make_objective(model=tiny_decoder().clone(dtype=jnp.float64), pad_id=0)

    def draw(seed):
        padded = {TEXT_KEY: jnp.asarray(np.where(np.arange(SEQ + 1) < 9,
                                                 token_batch(BATCH, seed=2 * seed)[TEXT_KEY], 0))}
        batches = [token_batch(BATCH, seed=2 * seed + 1), padded]
        tracker = RecordingTracker()
        trainer = make_trainer(pad_id=0, tracker=tracker)
        state = trainer.fit(dataset(cycle_batches, val=lambda: iter(batches)), steps=1, log_every=1,
                            eval_every=1, metrics=(Perplexity(),))
        logged = [scalars["val/perplexity"] for _, scalars in tracker.scalars if "val/perplexity" in scalars]
        assert len(logged) == 1
        return np.float32(logged[0]), jax.device_get(state.variables), batches

    objective = make_objective(pad_id=0)

    def expected(logged, variables, batches):
        scoring = wide if np.dtype(jax.tree.leaves(variables)[0].dtype) == np.float64 else objective
        scores = [scoring.evaluate(variables, batch, step_at()) for batch in batches]
        # The pass scores the weights the step left, so it matches them.
        assert 0 < sum(float(jnp.sum(score.weights)) for score in scores) < 2 * BATCH * SEQ
        return perplexity_of(*((np.asarray(score.losses), np.asarray(score.weights)) for score in scores))

    assert_as_exact(lambda logged, variables, batches: logged, expected, draw, "validation perplexity",
                    seeds=LM_SEEDS["validation perplexity"])


# --- packed batches --------------------------------------------------------

def packed_objective(seq=SEQ, model=None, **kwargs):
    return LMObjective(model or tiny_decoder(), seq, **kwargs)

def documents(first=7, second=3, seq=SEQ, rows=1, seed=11):
    """`rows` packed rows of two documents each, drawn from `seed`, plus the padding that follows them."""
    rs = np.random.RandomState(seed)
    doc_a = rs.randint(1, VOCAB, (rows, first))
    doc_b = rs.randint(1, VOCAB, (rows, second))
    pad = np.zeros((rows, seq + 1 - first - second), np.int64)
    tokens = jnp.asarray(np.concatenate([doc_a, doc_b, pad], axis=1), jnp.int32)
    segment_ids = jnp.asarray(np.concatenate(
        [np.full((rows, first), 1), np.full((rows, second), 2), pad], axis=1), jnp.int32)
    positions = jnp.asarray(np.concatenate(
        [np.broadcast_to(np.arange(first), (rows, first)), np.broadcast_to(np.arange(second), (rows, second)),
         pad], axis=1), jnp.int32)
    if rows == 1:
        doc_a, doc_b = doc_a[0], doc_b[0]
    return tokens, segment_ids, positions, (doc_a, doc_b)


def only_document(document, seq=SEQ):
    """One document a row (a row of each of `document`'s rows), right-padded to the same window."""
    document = np.atleast_2d(document)
    rows, length = document.shape
    pad = np.zeros((rows, seq + 1 - length), np.int64)
    tokens = jnp.asarray(np.concatenate([document, pad], axis=1), jnp.int32)
    segment_ids = jnp.asarray(np.concatenate([np.ones((rows, length)), pad], axis=1), jnp.int32)
    positions = jnp.asarray(np.concatenate([np.broadcast_to(np.arange(length), (rows, length)), pad], axis=1),
                            jnp.int32)
    return tokens, segment_ids, positions


def counted_losses(objective, params, tokens, segment_ids, positions, forward=None):
    """Per-token losses and the weight the objective gives each of them, over the model's logits read
    through `forward` (`forwards`; its apply by default), in NumPy at their width."""
    forward = forward or objective.model.apply
    inputs, targets = tokens[:, :-1], np.asarray(tokens[:, 1:])
    logits = np.asarray(forward(params, inputs, positions=positions[:, :-1], segment_ids=segment_ids[:, :-1]))
    largest = logits.max(axis=-1, keepdims=True)
    log_probs = logits - largest - np.log(np.exp(logits - largest).sum(axis=-1, keepdims=True))
    losses = -np.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    segments = np.asarray(segment_ids)
    weights = ((segments[:, 1:] == segments[:, :-1]) & (segments[:, 1:] != 0)).astype(losses.dtype)
    return losses, weights


def weighted_mean(losses, weights):
    return jnp.sum(losses * weights) / jnp.sum(weights)


PACKED_ROWS = 8
"""A packed draw's rows, each two documents of 7 and 3 and its padding."""


def packed_draw(objective):
    """A packed site's draw: the objective's parameters from the seed's key and `PACKED_ROWS` packed rows
    drawn from it."""
    init = initialized(objective)
    return lambda seed: (init(seed), *documents(rows=PACKED_ROWS, seed=seed)[:3])


def test_a_packed_batch_ignores_the_boundary_and_the_padding():
    """A packed row's weights drop the transition between its documents and its padding, exactly, and the
    scored mean is the cross entropy of the rest (check 1)."""
    objective = packed_objective()
    params = objective.init(jax.random.key(0))
    tokens, segment_ids, positions, (doc_a, _) = documents()

    scored_weights = objective.token_scores(params, tokens, segment_ids=segment_ids, positions=positions)[1]

    _, weights = counted_losses(objective, params, tokens, segment_ids, positions)
    weights = weights[0]
    # The last token of the first document predicts the first of the second,
    # which is the one transition a packed row must not train on.
    boundary = len(doc_a) - 1
    assert weights[boundary] == 0
    assert weights[:boundary].all() and weights[boundary + 1] == 1
    assert weights[len(doc_a) + 2:].sum() == 0, "padding was trained on"
    np.testing.assert_array_equal(np.asarray(scored_weights[0]), weights)

    forward = forwards(objective.model)

    def expected(params, tokens, segment_ids, positions):
        losses, weights = counted_losses(objective, params, tokens, segment_ids, positions, forward)
        return (losses * weights).sum() / weights.sum()

    @jax.jit
    def scored(params, tokens, segment_ids, positions):
        return weighted_mean(*objective.token_scores(params, tokens, segment_ids=segment_ids,
                                                     positions=positions)[:2])

    assert_as_exact(scored, expected, packed_draw(objective), "packed scores",
                    seeds=LM_SEEDS["packed scores"])


def test_a_packed_batch_scores_like_its_documents_alone():
    """Packing is only a layout: the two documents' losses have to be the ones
    they get on their own, from the whole logits and through the same chunked
    head, and the boundary target is dropped on both sides (check 1 each, the
    documents alone the reference)."""
    objective = packed_objective()
    params = objective.init(jax.random.key(0))
    tokens, segment_ids, positions, (doc_a, doc_b) = documents()
    for document in (doc_a, doc_b):
        alone_tokens, alone_segments, alone_positions = only_document(document)
        head_weights = objective.token_scores(params, alone_tokens, segment_ids=alone_segments,
                                              positions=alone_positions)[1]
        # Every transition inside the document; its last target is padding.
        assert float(jnp.sum(head_weights)) == len(document) - 1
    _, packed_weights = counted_losses(objective, params, tokens, segment_ids, positions)
    assert packed_weights.sum() == len(doc_a) + len(doc_b) - 2

    forward = forwards(objective.model)

    @jax.jit
    def packed(params, tokens, segment_ids, positions):
        return weighted_mean(*objective.token_scores(params, tokens, segment_ids=segment_ids,
                                                     positions=positions)[:2])

    def documents_of(tokens, segment_ids):
        tokens, segments = np.asarray(tokens), np.asarray(segment_ids)
        return [tokens[:, segments[0] == index] for index in (1, 2)]

    def alone(params, tokens, segment_ids, positions):
        total = count = 0
        for document in documents_of(tokens, segment_ids):
            losses, weights = counted_losses(objective, params, *only_document(document), forward)
            total, count = total + (losses * weights).sum(), count + weights.sum()
        return total / count

    def heads(params, tokens, segment_ids, positions):
        wide = np.dtype(jax.tree.leaves(params)[0].dtype) == np.float64
        scoring = (packed_objective(model=tiny_decoder().clone(dtype=jnp.float64)) if wide else objective)
        total = count = 0
        for document in documents_of(tokens, segment_ids):
            alone_tokens, alone_segments, alone_positions = only_document(document)
            losses, weights = scoring.token_scores(params, alone_tokens, segment_ids=alone_segments,
                                                   positions=alone_positions)[:2]
            total, count = total + jnp.sum(losses * weights), count + jnp.sum(weights)
        return total / count

    draw = packed_draw(objective)
    assert_as_exact(packed, alone, draw, "packed against alone", seeds=LM_SEEDS["packed against alone"])
    assert_as_exact(packed, heads, draw, "packed against its documents' heads",
                    seeds=LM_SEEDS["packed against its documents' heads"])


def test_a_packed_batch_reaches_the_objective_through_the_batch_dict():
    """The batch's coordinates carry the packing to the loss: its value is the packed per-token scores'
    mean under the same dropout key (check 1, the scores the reference and in float64 the truth), and
    differs from the unpacked row's."""
    objective = packed_objective()
    wide = packed_objective(model=tiny_decoder().clone(dtype=jnp.float64))
    params = objective.init(jax.random.key(0))
    tokens, segment_ids, positions, _ = documents()

    def batch_of(tokens, segment_ids, positions):
        return {TEXT_KEY: tokens, "text_coordinates": coordinates(
            segments=segment_ids, positions=positions, max_segments=tokens.shape[1], right_padded=True,
            xp=np)}

    @jax.jit
    def packed(params, tokens, segment_ids, positions):
        return objective.scalar_loss(params, batch_of(tokens, segment_ids, positions), step_at())[0]

    def scored(params, tokens, segment_ids, positions):
        scoring = wide if np.dtype(jax.tree.leaves(params)[0].dtype) == np.float64 else objective
        return weighted_mean(*scoring.token_scores(params, tokens, train=True,
                                                   rngs={"dropout": jax.random.key(1)},
                                                   segment_ids=segment_ids, positions=positions)[:2])

    assert_as_exact(packed, scored, packed_draw(objective), "packed batch dict",
                    seeds=LM_SEEDS["packed batch dict"])
    unpacked, _ = objective.scalar_loss(params, {TEXT_KEY: tokens}, step_at())
    assert float(packed(params, tokens, segment_ids, positions)) != float(unpacked), (
        "the packed keys made no difference to the loss")


@pytest.mark.parametrize("stale", [("text_segment_ids",), ("text_positions",),
                                  ("text_segment_ids", "text_positions")])
def test_a_packed_batch_refuses_the_old_packing_columns(stale):
    """A batch carrying the old packing columns is refused by the loss and
    the evaluation, where the same documents under `text_coordinates` are
    read (`test_a_packed_batch_reaches_the_objective_through_the_batch_dict`)."""
    objective = packed_objective()
    params = objective.init(jax.random.key(0))
    tokens, segments, positions, _ = documents()
    columns = {"text_segment_ids": segments, "text_positions": positions}
    batch = {TEXT_KEY: tokens, **{name: columns[name] for name in stale}}

    objective.evaluate(params, {TEXT_KEY: tokens}, step_at())
    with pytest.raises(ValueError):
        objective.scalar_loss(params, batch, step_at())
    with pytest.raises(ValueError):
        objective.evaluate(params, batch, step_at())


# --- loss_role: SFT on the chat data path --------------------------------------


def assistant_batch(every=2, batch=4, seq=SEQ, seed=0):
    """token_batch with a roles column marking every `every`-th target."""
    batch_dict = token_batch(batch=batch, seq=seq, seed=seed)
    roles = np.zeros((batch, seq + 1), np.int8)
    roles[:, 1::every] = Role.ASSISTANT
    batch_dict[ROLES_KEY] = jnp.asarray(roles)
    return batch_dict


def test_loss_role_counts_only_assistant_targets():
    """With loss_role set, the loss is the cross entropy over the targets
    whose role matches (check 1), and the reported accuracy counts the same
    targets, exactly, on a draw clear of argmax ties."""
    objective = make_objective(loss_role=Role.ASSISTANT)

    def kept(batch):
        return np.asarray(batch[ROLES_KEY])[:, 1:] == Role.ASSISTANT

    def expected(forward, params, batch):
        return shifted(forward, params, batch, weights=kept(batch))

    assert_objective_as_exact(objective, "loss role", lambda seed: assistant_batch(batch=ROWS, seed=seed),
                              expected)

    params, batch = clear_of_ties(objective, lambda seed: assistant_batch(seed=seed))
    _, aux = objective.scalar_loss(params, batch, step_at())
    tokens = np.asarray(batch[TEXT_KEY])
    logits = objective.model.apply(params, batch[TEXT_KEY][:, :-1])
    assert kept(batch).sum() > 0 and (~kept(batch)).sum() > 0, "the batch counts nothing out"
    assert reference_cross_entropy(logits, tokens[:, 1:], weights=kept(batch)) != reference_cross_entropy(
        logits, tokens[:, 1:]), "the role mask made no difference to check"
    assert np.float32(aux.metrics["token_accuracy"]) == accuracy_of(logits, tokens[:, 1:], kept(batch))


def test_a_roles_column_is_ignored_without_loss_role():
    """The same packed SFT batch trains every target under a plain
    LMObjective: the column only matters when a role is asked for (check 1)."""
    assert_objective_as_exact(make_objective(), "roles ignored",
                              lambda seed: assistant_batch(batch=ROWS, seed=seed), shifted)


def test_evaluation_weights_follow_loss_role():
    objective = make_objective(loss_role=Role.ASSISTANT)
    params = objective.init(jax.random.key(0))
    batch = assistant_batch()

    scores = objective.evaluate(params, batch, step_at())

    assert scores.losses.shape == scores.weights.shape == (4, SEQ)
    expected = (np.asarray(batch[ROLES_KEY])[:, 1:] == Role.ASSISTANT).astype(np.float32)
    np.testing.assert_array_equal(np.asarray(scores.weights), expected)


# --- z-loss: PaLM's log-partition penalty ---------------------------------


def test_z_loss_adds_the_squared_log_partition_of_the_counted_targets():
    """`z_loss * log(Z)^2` per counted target, masked and averaged with the
    cross entropy, as MaxText's `cross_entropy_with_logits` and
    `train.py` sum it (tools/distillation_reference.py:z_loss_reference): the
    loss and its z term on par with the oracle's on the model's own logits
    (check 1), the cross entropy it reports beside them the plain one's
    (check 1), and the term moves the loss."""
    from tools.distillation_reference import z_loss_reference

    pad_id, coefficient = 0, 1e-2
    plain = make_objective(pad_id=pad_id)
    zed = make_objective(pad_id=pad_id, z_loss=coefficient)
    forward, init = forwards(plain.model), initialized(plain)

    @jax.jit
    def dew(params, batch):
        loss, aux = zed.scalar_loss(params, batch, step_at())
        return loss, aux.metrics["z_loss"]

    def oracle(params, batch):
        tokens = np.asarray(batch[TEXT_KEY])
        logits = np.asarray(forward(params, jnp.asarray(tokens[:, :-1])))
        loss, z_term, _ = z_loss_reference(logits, tokens[:, 1:], tokens[:, 1:] != pad_id, coefficient,
                                           dtype=logits.dtype)
        return np.asarray(loss, logits.dtype), np.asarray(z_term, logits.dtype)

    assert_as_exact(dew, oracle, lambda seed: (init(seed), padded_batch(seed=seed)), "z loss",
                    seeds=LM_SEEDS["z loss"])
    def reported(params, batch):
        return zed.scalar_loss(params, batch, step_at())[1].metrics["ce"]

    assert_objective_as_exact(zed, "z-loss cross entropy", lambda seed: padded_batch(seed=seed),
                              lambda forward, params, batch: shifted(forward, params, batch, pad_id=pad_id),
                              reported)
    params, batch = init(0), padded_batch()
    without, plain_aux = plain.scalar_loss(params, batch, step_at())
    assert "z_loss" not in plain_aux.metrics
    assert float(dew(params, batch)[0]) != float(without), "the term made no difference to check"


Z_LOSS_SEEDS = 256
"""Check 1's declared seeds for the z-loss gradient, sized by `seeds_for` (2026-10-21, on armada)."""


def test_z_loss_gradient_carries_the_reference_factor():
    """The oracle's logit gradient, `(1 + 2 z log Z) softmax - onehot`, pulled
    back through the model, holds the objective's parameter gradient on par
    with the oracle's own (check 1), over drawn parameters and tokens: the
    reference pulls the oracle at the model's logits back through the model,
    in float32, and in float64 for the truth, the model then widened too."""
    from tools.distillation_reference import z_loss_reference

    coefficient = 1e-2
    objective = make_objective(z_loss=coefficient)
    model = objective.model

    initialized = jax.jit(lambda key: objective.init(key)["params"])

    def draw(seed):
        # 32 rows: at the 4 of `token_batch`'s default the reference's own rounding of the embedding
        # and norm gradients does not resolve its parity with itself within 512 seeds (`seeds_for`).
        return initialized(jax.random.key(seed)), np.asarray(token_batch(batch=32, seed=seed)[TEXT_KEY])

    def programs(forward):
        # The forward and its pullback compiled once each, the oracle's cotangent between them.
        apply = jax.jit(lambda p, ids: forward.apply({"params": p}, ids))
        pull = jax.jit(lambda p, ids, cotangent: jax.vjp(lambda p: forward.apply({"params": p}, ids), p)[1](
            cotangent)[0])
        return apply, pull

    compiled = {np.dtype(np.float32): programs(model),
                np.dtype(np.float64): programs(model.clone(dtype=jnp.float64))}

    def reference(params, tokens):
        wide = jax.tree.leaves(params)[0].dtype == np.float64
        apply, pull = compiled[np.dtype(np.float64 if wide else np.float32)]
        ids = jnp.asarray(tokens[:, :-1], jnp.int32)
        _, _, cotangent = z_loss_reference(np.asarray(apply(params, ids)), tokens[:, 1:],
                                           np.ones(tokens[:, 1:].shape), coefficient,
                                           dtype=None if wide else np.float32)
        return pull(params, ids, jnp.asarray(cotangent))

    @jax.jit
    def dew(params, tokens):
        def loss(p):
            return objective.scalar_loss({"params": p}, {TEXT_KEY: tokens}, step_at())[0]

        return jax.grad(loss)(params)

    assert_as_exact(dew, reference, draw, "z-loss gradient", seeds=Z_LOSS_SEEDS)


def test_router_z_loss_adds_each_routers_mean_squared_log_partition():
    """The term is the coefficient times each router's squared logsumexp of
    its gate logits, averaged over the positions it routed and summed over
    routers, beside the cross entropy and not inside its token mean: the loss
    and the term on par with the oracle's, the cross entropy of the model's
    logits plus the term recomputed from each layer's kernel and the normed
    input its router saw (check 1)."""
    from dew.nn.backbones.decoder_block import Mixture
    model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=2, num_heads=2,
                              max_seq_len=8, mixture=Mixture(experts=4, top_k=2, expert_features=16))
    zed = LMObjective(model, 8, ema_decay=None, router_z_loss=0.3)
    init = initialized(zed)
    captured = {np.dtype(np.float32): model, np.dtype(np.float64): model.clone(dtype=jnp.float64)}

    @jax.jit
    def dew(params, tokens):
        stats, aux = zed.loss(params, {TEXT_KEY: tokens}, step_at())
        return zed.reduce_loss(stats)[0], aux.metrics["router_z_loss"]

    def oracle(params, tokens):
        width = np.dtype(jax.tree.leaves(params)[0].dtype)
        # Each layer's gate logits recomputed from its kernel and the normed input the router saw,
        # captured by flax's intermediates; the block's norm owns its residual seam, which returns
        # the stream beside the normed input.
        logits, state = captured[width].apply(params, tokens[:, :-1], mutable=["intermediates"],
                                              capture_intermediates=lambda module, method: method == "seam")
        term = 0
        for index in range(2):
            layer = state["intermediates"][f"layers_{index}"]
            normed = np.asarray(layer["post_attention_layernorm"]["seam"][0][1], width)
            gates = normed @ np.asarray(params["params"][f"layers_{index}"]["mlp"]["gate"]["kernel"])
            largest = gates.max(axis=-1)
            log_z = largest + np.log(np.exp(gates - largest[..., None]).sum(axis=-1))
            term = term + width.type(0.3) * np.mean(np.square(log_z))
        return reference_cross_entropy(logits, np.asarray(tokens[:, 1:])) + term, term

    def draw(seed):
        return init(seed), jax.random.randint(jax.random.key(seed), (ROWS, 9), 0, 64)

    assert_as_exact(dew, oracle, draw, "router z-loss", seeds=LM_SEEDS["router z-loss"])


def vocabulary_products(jaxpr, vocab: int) -> int:
    """The products in `jaxpr` and every program inside it that read or write a vocabulary-sized array."""
    count = 0
    for eqn in jaxpr.eqns:
        shapes = [getattr(var.aval, "shape", ()) for var in (*eqn.invars, *eqn.outvars)]
        count += eqn.primitive.name == "dot_general" and any(vocab in shape for shape in shapes)
        for value in jax.tree.leaves(eqn.params, is_leaf=lambda leaf: hasattr(leaf, "eqns")
                                     or hasattr(getattr(leaf, "jaxpr", None), "eqns")):
            inner = value if hasattr(value, "eqns") else getattr(value, "jaxpr", None)
            if hasattr(inner, "eqns"):
                count += vocabulary_products(inner, vocab)
    return count


def head_scored(vocab: int, **options):
    """A two-layer decoder whose untied head spans `vocab` columns, and two
    9-token rows of its ids, for the tests that count the head's products."""
    model = CausalTransformer(vocab_size=vocab, emb_features=24, num_layers=2, num_heads=2, mlp_features=32,
                              max_seq_len=8, tie_embeddings=False, qk_norm=False, attention_impl="reference",
                              **options)
    return model, jax.random.randint(jax.random.key(1), (2, 9), 0, vocab)


def test_a_head_that_is_no_matrix_is_never_fused():
    """A head past a nonlinear transform is no one matrix (`chunked.head_table`), so a training step
    tiles it unfused, its program says so once traced, and the fit ladder has no rung to unfuse."""
    vocab = 61
    model, tokens = head_scored(vocab, head_transform="gelu")
    objective = LMObjective(model, 8, ema_decay=None, pad_id=0, head_tile=(4, 16))
    params = objective.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(1), None, training=True)
    jax.make_jaxpr(lambda params: objective.scalar_loss(params, {TEXT_KEY: tokens}, step)[0])(params)
    assert not objective.program_key()[0].head_fused
    assert objective.unfuse_head() is None


def test_a_head_put_back_unfused_stays_so_whatever_decoder_was_traced_before():
    """A record's unfused rung, put back (`climb_to`) on a decoder the objective has not traced,
    holds once the decoder's head turns out to be one matrix: a head whose transform was traced, then
    the decoder replaced by one with a matrix head, as a run's substitution does."""
    vocab = 61
    model, tokens = head_scored(vocab, head_transform="gelu")
    objective = LMObjective(model, 8, ema_decay=None, pad_id=0, head_tile=(4, 16))
    step = Step(jnp.asarray(0), jax.random.key(1), None, training=True)

    def trace():
        params = objective.init(jax.random.key(0))
        jax.make_jaxpr(lambda params: objective.scalar_loss(params, {TEXT_KEY: tokens}, step)[0])(params)

    trace()
    objective.substitute([model.clone(head_transform=None)])
    climb_to(objective, {"head_tile": [4, 16], "head_fused": False, "remat": None})
    trace()
    assert not objective.program_key()[0].head_fused and objective.unfuse_head() is None


def test_a_decoder_put_back_keeps_what_its_trace_found_of_its_head():
    """Each decoder a loss traced keeps what that trace found of its head, so a decoder put back
    after another was traced, as a half-rung probe that missed puts its decoder back, is still no
    fused head where its head is no matrix."""
    vocab = 61
    model, tokens = head_scored(vocab, head_transform="gelu")
    objective = LMObjective(model, 8, ema_decay=None, pad_id=0, head_tile=(4, 16))
    step = Step(jnp.asarray(0), jax.random.key(1), None, training=True)
    params = objective.init(jax.random.key(0))

    def trace():
        jax.make_jaxpr(lambda params: objective.scalar_loss(params, {TEXT_KEY: tokens}, step)[0])(params)

    trace()
    objective.substitute([model.clone(remat="full")])
    trace()
    objective.substitute([model])
    assert not objective.program_key()[0].head_fused and objective.unfuse_head() is None


def test_a_prediction_applies_the_head_once_and_scores_the_logits_it_returns():
    """A prediction that returns its logits applies the head once, as the
    whole logits, and scores its losses from them: its forward contracts the
    vocabulary once and its gradient three times, and its losses and logits
    are the tiled head's and the model's own (check 1). A padding id past the
    vocabulary as the last target picks no column, as the tiled head's, so
    every loss stays finite."""
    from dew.objectives.lm.chunked import model_logits
    vocab = 61
    model, tokens = head_scored(vocab)
    tokens = tokens.at[:, -1].set(vocab)
    objective = LMObjective(model, 8, ema_decay=None, pad_id=vocab, head_tile=(4, 16))
    wide = LMObjective(model.clone(dtype=jnp.float64), 8, ema_decay=None, pad_id=vocab, head_tile=(4, 16))
    params = objective.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(1), None, training=False)

    def predicted(params, tokens=tokens):
        return objective.predict(params, {TEXT_KEY: tokens}, step, train=False)[2]

    def pulled(params):
        prediction = predicted(params)
        return jnp.sum(prediction.logits * 0.5) + jnp.sum(prediction.losses * prediction.weights)

    assert vocabulary_products(jax.make_jaxpr(predicted)(params).jaxpr, vocab) == 1
    assert vocabulary_products(jax.make_jaxpr(jax.grad(pulled))(params).jaxpr, vocab) == 3
    assert np.all(np.isfinite(predicted(params).losses))

    def tiled(scoring):
        def run(params, tokens):
            scores = scoring.token_scores(params, tokens)
            return scores.losses, model_logits(scoring.model, params, scores.hidden)
        return without_ynn(run)

    references = {np.dtype(np.float32): tiled(objective), np.dtype(np.float64): tiled(wide)}
    init = initialized(objective)

    def draw(seed):
        drawn = jax.random.randint(jax.random.key(seed), (ROWS, 9), 0, vocab)
        return init(seed), drawn.at[:, -1].set(vocab)

    def scored(params, tokens):
        prediction = predicted(params, tokens)
        return prediction.losses, prediction.logits

    def reference(params, tokens):
        return references[np.dtype(jax.tree.leaves(params)[0].dtype)](params, tokens)

    with jax.default_matmul_precision("highest"):
        assert_as_exact(without_ynn(scored), reference, draw, "prediction", seeds=LM_SEEDS["prediction"])


@pytest.mark.parametrize("tile_tokens", [4, 16])
def test_a_training_steps_tiled_head_contracts_the_vocabulary_three_times_and_matches_the_tiled_head(
        tile_tokens):
    """A training step whose head is tiled takes the head's gradient in the
    forward (`fused_head_loss`), a tile's tokens a chunk, all 16 in one where
    the tile holds them: beside blocks recomputed in the backward, the step
    contracts the whole vocabulary three times, the logits, the states'
    gradient and the head's, none again, and its loss and gradients are the
    per-token tiled head's, z-loss and padding included (check 1). Unfused
    (`unfuse_head`, the fit ladder's rung for a fused chunk that does not
    fit), the step takes the tiled head's gradient in its backward, a tile
    of the vocabulary at a time, to the same loss and gradients (check 1)."""
    vocab = 61
    model, tokens = head_scored(vocab, remat="full")
    tokens = tokens.at[1, 6:].set(0)
    options = {"ema_decay": None, "z_loss": 1e-3, "pad_id": 0, "head_tile": (tile_tokens, 16)}
    fused, unfused = LMObjective(model, 8, **options), LMObjective(model, 8, **options)
    wide = LMObjective(model.clone(dtype=jnp.float64), 8, **options)
    assert unfused.unfuse_head() is not None and unfused.unfuse_head() is None
    params = fused.init(jax.random.key(0))

    def loss(objective, *, training):
        step = Step(jnp.asarray(0), jax.random.key(1), None, training=training)

        def scalar(params, tokens=tokens):
            stats, _ = objective.loss(params, {TEXT_KEY: tokens}, step)
            return objective.reduce_loss(stats)[0]
        return scalar

    def step_products(objective) -> int:
        program = jax.make_jaxpr(jax.grad(loss(objective, training=True)))(params)
        return vocabulary_products(program.jaxpr, vocab)

    assert step_products(fused) == 3
    # The tiled head contracts the vocabulary a tile of its columns at a time, so no product spans it.
    assert step_products(unfused) == 0

    references = {np.dtype(width): without_ynn(jax.value_and_grad(loss(objective, training=False)))
                  for width, objective in ((np.float32, fused), (np.float64, wide))}
    init = initialized(fused)

    def draw(seed):
        drawn = jax.random.randint(jax.random.key(seed), (ROWS, 9), 0, vocab)
        # Half the rows padded from their seventh token.
        return init(seed), drawn.at[::2, 6:].set(0)

    with jax.default_matmul_precision("highest"):
        for name, objective in (("fused", fused), ("unfused", unfused)):
            label = f"tiled head {name} {tile_tokens}"
            assert_as_exact(without_ynn(jax.value_and_grad(loss(objective, training=True))),
                            lambda params, tokens: references[np.dtype(jax.tree.leaves(params)[0].dtype)](
                                params, tokens), draw, label, seeds=LM_SEEDS[label])
