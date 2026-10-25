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
from reference_error import assert_as_exact, assert_as_exact_as_the_reference, widened, without_ynn
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


def written_cross_entropy(model, counted=None, shift=1):
    """The cross entropy written from `model`'s own logits of every token but the last, in their dtype:
    each target `shift` tokens on, its negative log probability averaged over the targets
    `counted(tokens, *given)` keeps (every one where it is None). The reference arm of the loss's
    check 1, and on the model in float64 its truth."""
    def loss(params, tokens, *given):
        logits = model.apply(params, tokens[:, :-1])
        targets = tokens[:, shift:tokens.shape[1] - 1 + shift]
        picked = jnp.take_along_axis(jax.nn.log_softmax(logits, axis=-1), targets[..., None], axis=-1)[..., 0]
        weights = jnp.ones_like(picked) if counted is None else counted(tokens, *given).astype(picked.dtype)
        return -jnp.sum(picked * weights) / jnp.sum(weights)
    return loss


def assert_scores_the_written_cross_entropy(dew, draw, label, *, seeds, counted=None, model=None):
    """Check 1: `dew(params, tokens, *given)` on par with the cross entropy written from the decoder's
    logits (`written_cross_entropy`), both from that cross entropy's float64 run, over `draw`'s
    parameters, tokens and what each draw gives beside them."""
    model = model or tiny_decoder()
    assert_as_exact(without_ynn(dew), without_ynn(written_cross_entropy(model, counted)), draw, label,
                    seeds=seeds, truth_fn=written_cross_entropy(model.clone(dtype=jnp.float64), counted))


def drawn_tokens(counted=None, **options):
    """Each seed's draw: the decoder's parameters and a batch of its tokens (`token_batch`'s, of
    `options`), and what `counted` gives beside them."""
    initialized = jax.jit(make_objective().init)

    def draw(seed):
        tokens = token_batch(seed=seed, **options)[TEXT_KEY]
        return (initialized(jax.random.key(seed)), tokens) if counted is None else (
            initialized(jax.random.key(seed)), tokens, *counted(seed))
    return draw


SHIFTED_SEEDS = 8
"""Check 1's declared seeds for the loss against the written shifted cross entropy (`seeds_for`)."""


# --- the loss --------------------------------------------------------------

def test_loss_is_the_cross_entropy_of_the_shifted_sequence():
    """Over drawn decoders and tokens, the loss on par with the cross entropy of each token's next
    (check 1), which differs from the unshifted one, the mistake this guards."""
    objective = make_objective()

    def loss(params, tokens):
        return objective.scalar_loss(params, {TEXT_KEY: tokens}, step_at())[0]

    assert_scores_the_written_cross_entropy(loss, drawn_tokens(), "the shifted cross entropy",
                                            seeds=SHIFTED_SEEDS)
    params, tokens = drawn_tokens()(0)
    unshifted = written_cross_entropy(objective.model, shift=0)(params, tokens)
    assert written_cross_entropy(objective.model)(params, tokens) != unshifted


VALID_SEEDS = 8
"""Check 1's declared seeds for the loss of rows whose last token is invalid (`seeds_for`)."""


@pytest.mark.parametrize("given", ["attention_mask", "coordinates"])
def test_the_validity_the_decoder_reads_decides_which_targets_count(given):
    """Rows of three tokens whose last is marked invalid, by the attention
    mask or by the coordinates' validity the decoder reads: over drawn
    decoders and rows, the loss on par with the cross entropy of each row's
    one target between valid tokens, which the model gives the row of its
    first token alone (check 1)."""
    objective = make_objective(seq=2)

    def loss(params, ids):
        valid = jnp.broadcast_to(jnp.asarray([True, True, False]), ids.shape)
        batch = ({TEXT_KEY: ModelInputs(ids, {"attention_mask": valid})} if given == "attention_mask"
                 else {TEXT_KEY: ids, "text_coordinates": coordinates(valid=valid)})
        return objective.scalar_loss(params, batch, step_at())[0]

    assert_scores_the_written_cross_entropy(
        loss, drawn_tokens(batch=16, seq=2), f"the valid target, by {given}", seeds=VALID_SEEDS,
        model=objective.model, counted=lambda ids: jnp.arange(2)[None] == 0)


def counted_share(right, counted=None):
    """The share of the counted answers that are `right`, as the objective reports it: two exact counts
    and one float32 division."""
    counted = np.ones_like(right) if counted is None else counted
    return np.float32((right & counted).sum()) / np.float32(counted.sum())


def unpadded(tokens, *_):
    """The targets a pad id of zero leaves counted."""
    return tokens[:, 1:] != 0


PADDED_SEEDS = 8
"""Check 1's declared seeds for the loss with its padded targets left out (`seeds_for`)."""


def test_padded_targets_are_left_out_of_the_average():
    """Over drawn decoders and tokens, some of them the pad id, the loss on par with the cross entropy of
    the targets that are not (check 1), which differs from every target's."""
    objective = make_objective(pad_id=0)

    def loss(params, tokens):
        return objective.scalar_loss(params, {TEXT_KEY: tokens}, step_at())[0]

    assert_scores_the_written_cross_entropy(loss, drawn_tokens(), "the unpadded targets",
                                            seeds=PADDED_SEEDS, counted=unpadded)
    params, tokens = drawn_tokens()(0)
    assert np.any(~unpadded(tokens)), "this batch has no padding to skip"
    masked = written_cross_entropy(objective.model, unpadded)(params, tokens)
    assert masked != written_cross_entropy(objective.model)(params, tokens), "masking made no difference"


def test_aux_reports_perplexity_and_token_accuracy():
    objective = make_objective()
    params = objective.init(jax.random.key(0))
    batch = token_batch()

    loss, aux = objective.scalar_loss(params, batch, step_at())

    assert set(aux.metrics) == {"ce", "perplexity", "token_accuracy"}
    assert aux.variables is None
    # The cross entropy is the loss, and the perplexity its exponential, each the value itself.
    np.testing.assert_array_equal(aux.metrics["ce"], loss)
    np.testing.assert_array_equal(aux.metrics["perplexity"], jnp.exp(loss))

    logits = objective.model.apply(params, batch[TEXT_KEY][:, :-1])
    right = np.argmax(np.asarray(logits), axis=-1) == np.asarray(batch[TEXT_KEY][:, 1:])
    np.testing.assert_array_equal(aux.metrics["token_accuracy"], counted_share(right))


def test_dropout_runs_on_the_step_key():
    """The loss passes the step key as the dropout rng, so a model with
    dropout is stochastic across keys and reproducible under one."""
    objective = make_objective(model=tiny_decoder(dropout_rate=0.5))
    params = objective.init(jax.random.key(0))
    batch = token_batch()

    first, _ = objective.scalar_loss(params, batch, step_at(key=1))
    again, _ = objective.scalar_loss(params, batch, step_at(key=1))
    other, _ = objective.scalar_loss(params, batch, step_at(key=2))

    np.testing.assert_array_equal(first, again)
    assert first != other


def test_padded_tokens_are_left_out_of_the_accuracy_too():
    objective = make_objective(pad_id=0)
    params = objective.init(jax.random.key(0))
    batch = token_batch(seed=3)
    targets = np.asarray(batch[TEXT_KEY][:, 1:])
    assert (targets == 0).any(), "this batch has no padding to skip"

    _, aux = objective.scalar_loss(params, batch, step_at())

    logits = np.asarray(objective.model.apply(params, batch[TEXT_KEY][:, :-1]))
    np.testing.assert_array_equal(aux.metrics["token_accuracy"],
                                  counted_share(np.argmax(logits, axis=-1) == targets, targets != 0))


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


EVALUATED_SEEDS = 8
"""Check 1's declared seeds for evaluation's per-target losses against the written cross entropy
(`seeds_for`)."""


def test_evaluation_scores_every_target_of_the_batch():
    """Evaluation scores each target at weight one, and over drawn decoders and tokens the targets'
    mean loss on par with their written cross entropy (check 1)."""
    objective = make_objective()
    params = objective.init(jax.random.key(0))
    scores = objective.evaluate(params, token_batch(), step_at())
    assert scores.losses.shape == scores.weights.shape == (4, SEQ)
    np.testing.assert_array_equal(scores.weights, 1.0)

    def mean_loss(params, tokens):
        return jnp.mean(objective.evaluate(params, {TEXT_KEY: tokens}, step_at()).losses)

    assert_scores_the_written_cross_entropy(mean_loss, drawn_tokens(), "evaluation's losses",
                                            seeds=EVALUATED_SEEDS)


def test_evaluation_reads_the_ema_copy():
    """Over drawn live and averaged decoders and tokens, evaluation's mean loss on par with the written
    cross entropy of the averaged copy (check 1), and not the live one's."""
    objective = make_objective()
    initialized = jax.jit(objective.init)
    tokens = drawn_tokens()

    def draw(seed):
        params, ids = tokens(seed)
        return params, ids, initialized(jax.random.key(seed + (1 << 20)))

    def averaged(params, ids, ema):
        return jnp.mean(objective.evaluate(params, {TEXT_KEY: ids}, step_at(ema=ema)).losses)

    written = written_cross_entropy(objective.model)
    wide = written_cross_entropy(objective.model.clone(dtype=jnp.float64))
    assert_as_exact(without_ynn(averaged), without_ynn(lambda params, ids, ema: written(ema, ids)), draw,
                    "evaluation of the averaged copy", seeds=EVALUATED_SEEDS,
                    truth_fn=lambda params, ids, ema: wide(ema, ids))
    params, ids, ema = draw(0)
    live = jnp.mean(objective.evaluate(params, {TEXT_KEY: ids}, step_at()).losses)
    assert live != averaged(params, ids, ema)


def test_scoring_with_samples_configured_does_not_decode():
    def fail_decode(ids):
        raise AssertionError("scoring decoded a preview")

    samples = Samples(prompt=[1, 2, 3], max_new_tokens=2, decode=fail_decode)
    objective = make_objective(samples=samples)
    wide = make_objective(model=tiny_decoder().clone(dtype=jnp.float64), samples=samples)

    def scored(params, tokens):
        return objective.evaluate(params, {TEXT_KEY: tokens}, step_at()).losses

    # Evaluation's losses, scored without a preview, on par with the per-token scores (check 1).
    assert_as_exact(without_ynn(scored), without_ynn(lambda params, tokens: objective.token_scores(
        params, tokens).losses), drawn_tokens(), "scoring beside samples", seeds=EVALUATED_SEEDS,
                    truth_fn=lambda params, tokens: wide.token_scores(params, tokens).losses)


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

def test_perplexity_weighs_every_batch_by_its_counted_targets():
    """exp(sum(loss * weight) / sum(weight)) over the pass: a batch with more
    counted targets moves the score more, which the mean of per-batch means
    gets wrong the moment counts differ."""
    metric = Perplexity()
    assert metric.reads is TokenScores
    heavy = TokenScores(
        losses=jnp.full((1, 4), 1.0),
        weights=jnp.ones((1, 4)),
        correct=jnp.zeros_like(jnp.full((1, 4), 1.0), dtype=bool),
    )
    light = TokenScores(
        losses=jnp.full((1, 4), 3.0),
        weights=jnp.array([[1.0, 0, 0, 0]]),
        correct=jnp.zeros_like(jnp.full((1, 4), 3.0), dtype=bool),
    )

    score = metric.finalize(metric.merge(metric(heavy, None), metric(light, None)))

    # The metric's float64 sums of these values are exact, so the score is this exponential itself.
    assert score == np.exp((4 * 1.0 + 1 * 3.0) / 5)
    assert score != np.exp(np.mean([1.0, 3.0]))


def test_perplexity_is_exp_of_the_mean_cross_entropy_not_the_mean_of_exps():
    metric = Perplexity()
    values = [
        metric(
            TokenScores(
                losses=jnp.full((1, 2), ce),
                weights=jnp.ones((1, 2)),
                correct=jnp.zeros_like(jnp.full((1, 2), ce), dtype=bool),
            ),
            None,
        )
        for ce in (0.0, 2.0)
    ]
    expected = np.exp(np.mean([0.0, 2.0]))
    wrong = np.mean(np.exp([0.0, 2.0]))
    assert expected != wrong
    assert metric.finalize(metric.merge(*values)) == expected


def test_the_validation_pass_scores_perplexity_per_token_and_logs_it():
    """Objective artifacts through the trainer's loop and into the metric:
    two batches of different padding weigh by their counted targets."""
    padded = {TEXT_KEY: jnp.asarray(np.where(np.arange(SEQ + 1) < 9, token_batch(BATCH)[TEXT_KEY], 0))}
    batches = [token_batch(BATCH, seed=1), padded]
    tracker = RecordingTracker()
    trainer = make_trainer(pad_id=0, tracker=tracker)
    state = trainer.fit(dataset(cycle_batches, val=lambda: iter(batches)), steps=1, log_every=1,
                        eval_every=1, metrics=(Perplexity(),))

    metric = Perplexity()
    scored = [(trainer.objective.evaluate(state.variables, batch, step_at()), batch) for batch in batches]
    count = sum(float(jnp.sum(scores.weights)) for scores, _ in scored)
    logged = [s["val/perplexity"] for _, s in tracker.scalars if "val/perplexity" in s]
    assert len(logged) == 1
    # The pass scores the weights the step left, so it logs their perplexity, by the metric's own sums.
    assert 0 < count < 2 * BATCH * SEQ
    assert logged[0] == metric.finalize(metric.merge(*(metric(scores, batch) for scores, batch in scored)))


# --- packed batches --------------------------------------------------------

def packed_objective(seq=SEQ, **kwargs):
    return LMObjective(tiny_decoder(), seq, **kwargs)


def documents(first=7, second=3, seq=SEQ, seed=11):
    """A packed row of two documents, plus the padding that follows them."""
    rs = np.random.RandomState(seed)
    doc_a = rs.randint(1, VOCAB, first)
    doc_b = rs.randint(1, VOCAB, second)
    pad = np.zeros(seq + 1 - first - second, np.int64)
    tokens = jnp.asarray(np.concatenate([doc_a, doc_b, pad])[None], jnp.int32)
    segment_ids = jnp.asarray(
        np.concatenate([np.full(first, 1), np.full(second, 2), pad])[None], jnp.int32)
    positions = jnp.asarray(
        np.concatenate([np.arange(first), np.arange(second), pad])[None], jnp.int32)
    return tokens, segment_ids, positions, (doc_a, doc_b)


def only_document(document, seq=SEQ):
    """One document in a row of its own, right-padded to the same window."""
    pad = np.zeros(seq + 1 - len(document), np.int64)
    tokens = jnp.asarray(np.concatenate([document, pad])[None], jnp.int32)
    segment_ids = jnp.asarray(
        np.concatenate([np.full(len(document), 1), pad])[None], jnp.int32)
    positions = jnp.asarray(
        np.concatenate([np.arange(len(document)), pad])[None], jnp.int32)
    return tokens, segment_ids, positions


def counted_losses(objective, params, tokens, segment_ids, positions):
    """Per-token losses written from the model's logits, in their dtype, and the weight the objective
    gives each of them."""
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    logits = objective.model.apply(params, inputs, positions=positions[:, :-1],
                                   segment_ids=segment_ids[:, :-1])
    losses = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    weights = ((segment_ids[:, 1:] == segment_ids[:, :-1])
               & (segment_ids[:, 1:] != 0)).astype(losses.dtype)
    return losses[0], weights[0]


def weighted_mean(losses, weights):
    return jnp.sum(losses * weights) / jnp.sum(weights)


PACKED_SEEDS = 8
"""Check 1's declared seeds for a packed row's loss against its documents' (`seeds_for`)."""


def drawn_documents(seed):
    """A seed's decoder parameters, a packed row of two documents of its own and the two documents in
    rows of their own (`documents`, `only_document`)."""
    tokens, segment_ids, positions, documents_drawn = documents(seed=seed)
    alone = [piece for document in documents_drawn for piece in only_document(document)]
    return jax.jit(packed_objective().init)(jax.random.key(seed)), tokens, segment_ids, positions, *alone


def float64_packed():
    """The packed objective on its decoder in float64, for the truths of the packed rows' check 1."""
    return LMObjective(tiny_decoder().clone(dtype=jnp.float64), SEQ)


def test_a_packed_batch_ignores_the_boundary_and_the_padding():
    """A packed row trains on every transition inside a document and on none across the boundary or
    into the padding, and over drawn decoders and documents its loss is on par with the cross entropy
    written from the model's logits of the packed row (check 1)."""
    objective = packed_objective()
    params = objective.init(jax.random.key(0))
    tokens, segment_ids, positions, (doc_a, _) = documents()

    scored_weights = objective.token_scores(params, tokens, segment_ids=segment_ids, positions=positions)[1]

    _, weights = (np.asarray(value) for value in counted_losses(objective, params, tokens, segment_ids,
                                                                  positions))
    # The last token of the first document predicts the first of the second,
    # which is the one transition a packed row must not train on.
    boundary = len(doc_a) - 1
    assert weights[boundary] == 0
    assert weights[:boundary].all() and weights[boundary + 1] == 1
    assert weights[len(doc_a) + 2:].sum() == 0, "padding was trained on"
    np.testing.assert_array_equal(np.asarray(scored_weights[0]), weights)

    def scored(params, tokens, segment_ids, positions, *_):
        return weighted_mean(*objective.token_scores(params, tokens, segment_ids=segment_ids,
                                                     positions=positions)[:2])

    def written(of):
        return lambda params, tokens, segment_ids, positions, *_: weighted_mean(
            *counted_losses(of, params, tokens, segment_ids, positions))

    assert_as_exact(without_ynn(scored), without_ynn(written(objective)), drawn_documents,
                    "a packed row's written loss", seeds=PACKED_SEEDS, truth_fn=written(float64_packed()))


def test_a_packed_batch_scores_like_its_documents_alone():
    """Packing is only a layout: over drawn decoders and documents, a packed
    row's loss is on par with the two documents' on their own, from the whole
    logits and through the same chunked head (check 1; the packed row's own
    written loss is the test before's), and the boundary target is dropped on
    both sides."""
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
    assert float(jnp.sum(packed_weights)) == len(doc_a) + len(doc_b) - 2

    def scored(params, tokens, segment_ids, positions, *_):
        return weighted_mean(*objective.token_scores(params, tokens, segment_ids=segment_ids,
                                                     positions=positions)[:2])

    def alone(score):
        def mean(params, tokens, segment_ids, positions, *pieces):
            parts = [score(params, *pieces[index:index + 3]) for index in (0, 3)]
            return sum(jnp.sum(losses * weights) for losses, weights in parts) / sum(
                jnp.sum(weights) for _, weights in parts)
        return mean

    def written(of):
        return alone(lambda params, *row: counted_losses(of, params, *row))

    def chunked(of):
        return alone(lambda params, tokens, segment_ids, positions: of.token_scores(
            params, tokens, segment_ids=segment_ids, positions=positions)[:2])

    wide = float64_packed()
    assert_as_exact(without_ynn(scored), without_ynn(written(objective)), drawn_documents,
                    "the documents' written losses", seeds=PACKED_SEEDS, truth_fn=written(wide))
    assert_as_exact(without_ynn(scored), without_ynn(chunked(objective)), drawn_documents,
                    "the documents' chunked losses", seeds=PACKED_SEEDS, truth_fn=chunked(wide))


def test_a_packed_batch_reaches_the_objective_through_the_batch_dict():
    """Over drawn decoders and documents, the loss of a batch whose packing comes as `text_coordinates`
    on par with the per-token scores of the packed row (check 1), and not the loss of the same row
    unpacked."""
    objective = packed_objective()

    def batch_of(tokens, segment_ids, positions):
        return {TEXT_KEY: tokens, "text_coordinates": coordinates(
            segments=segment_ids, positions=positions, max_segments=tokens.shape[1], right_padded=True)}

    def packed(params, tokens, segment_ids, positions, *_):
        return objective.scalar_loss(params, batch_of(tokens, segment_ids, positions), step_at())[0]

    def scored(of):
        return lambda params, tokens, segment_ids, positions, *_: weighted_mean(*of.token_scores(
            params, tokens, train=True, rngs={"dropout": jax.random.key(1)}, segment_ids=segment_ids,
            positions=positions)[:2])

    assert_as_exact(without_ynn(packed), without_ynn(scored(objective)), drawn_documents,
                    "a packed batch's loss", seeds=PACKED_SEEDS, truth_fn=scored(float64_packed()))
    params, tokens, segment_ids, positions, *_ = drawn_documents(0)
    unpacked, _ = objective.scalar_loss(params, {TEXT_KEY: tokens}, step_at())
    assert packed(params, tokens, segment_ids, positions) != unpacked, (
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


ROLES_SEEDS = 8
"""Check 1's declared seeds for the loss of a batch with a roles column (`seeds_for`)."""


def drawn_roles(seed):
    """A seed's decoder parameters, its tokens and the roles column marking every second target."""
    batch = assistant_batch(seed=seed)
    return jax.jit(make_objective().init)(jax.random.key(seed)), batch[TEXT_KEY], batch[ROLES_KEY]


def assistant_targets(tokens, roles):
    """The targets whose role is the assistant's."""
    return roles[:, 1:] == Role.ASSISTANT


def test_loss_role_counts_only_assistant_targets():
    """With loss_role set, over drawn decoders and tokens the loss is on par
    with the cross entropy over the targets whose role matches (check 1),
    which differs from every target's, and the reported accuracy counts the
    same targets."""
    objective = make_objective(loss_role=Role.ASSISTANT)

    def loss(params, tokens, roles):
        return objective.scalar_loss(params, {TEXT_KEY: tokens, ROLES_KEY: roles}, step_at())[0]

    assert_scores_the_written_cross_entropy(loss, drawn_roles, "the assistant's targets", seeds=ROLES_SEEDS,
                                            counted=assistant_targets)
    params, tokens, roles = drawn_roles(0)
    kept = np.asarray(assistant_targets(tokens, roles))
    assert kept.sum() > 0 and (~kept).sum() > 0, "the batch counts nothing out"
    masked = written_cross_entropy(objective.model, assistant_targets)(params, tokens, roles)
    assert masked != written_cross_entropy(objective.model)(params, tokens), "the mask made no difference"
    _, aux = objective.scalar_loss(params, {TEXT_KEY: tokens, ROLES_KEY: roles}, step_at())
    logits = np.asarray(objective.model.apply(params, tokens[:, :-1]))
    right = np.argmax(logits, axis=-1) == np.asarray(tokens[:, 1:])
    np.testing.assert_array_equal(aux.metrics["token_accuracy"], counted_share(right, kept))


def test_a_roles_column_is_ignored_without_loss_role():
    """The same SFT batch trains every target under a plain LMObjective: over
    drawn decoders and tokens its loss is on par with every target's cross
    entropy (check 1); the column only matters when a role is asked for."""
    objective = make_objective()

    def loss(params, tokens, roles):
        return objective.scalar_loss(params, {TEXT_KEY: tokens, ROLES_KEY: roles}, step_at())[0]

    assert_scores_the_written_cross_entropy(loss, drawn_roles, "every target beside roles", seeds=ROLES_SEEDS)


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
    loss and its z term as near the oracle on the float64 forward's logits as
    the oracle's own float32 run on the float32 logits is
    (`assert_as_exact_as_the_reference`), and zero keeps the loss the plain
    cross entropy to the bit."""
    from tools.distillation_reference import z_loss_reference

    pad_id, coefficient = 0, 1e-2
    batch = token_batch()
    tokens = np.asarray(batch[TEXT_KEY])
    plain = make_objective(pad_id=pad_id)
    params = plain.init(jax.random.key(0))
    ids = jnp.asarray(tokens[:, :-1], jnp.int32)
    logits = np.asarray(plain.model.apply(params, ids))
    weights = (tokens[:, 1:] != pad_id).astype(np.float64)
    rounded, rounded_z, _ = z_loss_reference(logits, tokens[:, 1:], weights, coefficient, dtype=np.float32)
    with jax.enable_x64(new_val=True):
        wide_logits = np.asarray(plain.model.clone(dtype=jnp.float64).apply(widened(params), ids))
        expected, z_term, _ = z_loss_reference(wide_logits, tokens[:, 1:], weights, coefficient)

    loss, aux = make_objective(pad_id=pad_id, z_loss=coefficient).scalar_loss(params, batch, step_at())
    without, plain_aux = plain.scalar_loss(params, batch, step_at())

    for dew, reference, truth, label in ((loss, rounded, expected, "loss"),
                                         (aux.metrics["z_loss"], rounded_z, z_term, "z term")):
        assert_as_exact_as_the_reference(np.atleast_1d(np.float64(dew)), np.atleast_1d(reference),
                                         np.atleast_1d(truth), label)
    # The term leaves the cross entropy the plain objective's, which is the padded targets' own
    # (`test_padded_targets_are_left_out_of_the_average`), and adds to the loss.
    np.testing.assert_array_equal(aux.metrics["ce"], plain_aux.metrics["ce"])
    assert "z_loss" not in plain_aux.metrics
    assert loss != without, "the term made no difference to check"


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


ROUTER_SEEDS = 8
"""Check 1's declared seeds for the router z-loss against its recomputed oracle (`seeds_for`)."""


def test_router_z_loss_adds_each_routers_mean_squared_log_partition():
    """The term is the coefficient times each router's squared logsumexp of
    its gate logits, averaged over the positions it routed and summed over
    routers, beside the cross entropy and not inside its token mean: over
    drawn decoders and tokens, the loss on par with the plain loss plus that
    term, and the term it reports on par with the term (check 1), the oracle
    each layer's gate logits recomputed from its kernel and the normed input
    the router saw, captured by flax's intermediates."""
    from dew.nn.backbones.decoder_block import Mixture
    model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=2, num_heads=2,
                              max_seq_len=8, mixture=Mixture(experts=4, top_k=2, expert_features=16))
    zed = LMObjective(model, 8, ema_decay=None, router_z_loss=0.3)
    initialized = jax.jit(zed.init)

    def draw(seed):
        return initialized(jax.random.key(seed)), jax.random.randint(jax.random.key(seed + (1 << 20)),
                                                                     (8, 9), 0, 64)

    def scored(params, tokens):
        stats, aux = zed.loss(params, {TEXT_KEY: tokens}, step_at())
        return zed.reduce_loss(stats)[0], aux.metrics["router_z_loss"]

    def oracle(of):
        plain = LMObjective(of, 8, ema_decay=None)

        def recomputed(params, tokens):
            # The block's norm owns its residual seam, which returns the stream beside the normed input.
            _, state = of.apply(params, tokens[:, :-1], mutable=["intermediates"],
                                capture_intermediates=lambda module, method: method == "seam")
            term = 0.0
            for index in range(2):
                normed = state["intermediates"][f"layers_{index}"]["post_attention_layernorm"]["seam"][0][1]
                kernel = params["params"][f"layers_{index}"]["mlp"]["gate"]["kernel"]
                term += 0.3 * jnp.mean(jnp.square(jax.nn.logsumexp(normed @ kernel, -1)))
            base, _ = plain.reduce_loss(plain.loss(params, {TEXT_KEY: tokens}, step_at())[0])
            return base + term, term
        return recomputed

    assert_as_exact(without_ynn(scored), without_ynn(oracle(model)), draw, "router z-loss",
                    seeds=ROUTER_SEEDS, truth_fn=oracle(model.clone(dtype=jnp.float64)))


def vocabulary_products(jaxpr, vocab: int) -> int:
    """The products in `jaxpr` and every program inside it that read or write a vocabulary-sized array,
    a cond counted by the branch with the most, the one a run takes at most (as test_model_flops'
    `jaxpr_products` counts): a product's platform choice is a cond of both."""
    count = 0
    for eqn in jaxpr.eqns:
        shapes = [getattr(var.aval, "shape", ()) for var in (*eqn.invars, *eqn.outvars)]
        count += eqn.primitive.name == "dot_general" and any(vocab in shape for shape in shapes)
        inner = [value if hasattr(value, "eqns") else value.jaxpr
                 for value in jax.tree.leaves(eqn.params, is_leaf=lambda leaf: hasattr(leaf, "eqns")
                                              or hasattr(getattr(leaf, "jaxpr", None), "eqns"))
                 if hasattr(value, "eqns") or hasattr(getattr(value, "jaxpr", None), "eqns")]
        counts = [vocabulary_products(program, vocab) for program in inner]
        count += max(counts, default=0) if eqn.primitive.name == "cond" else sum(counts)
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


PREDICTION_SEEDS = 8
"""Check 1's declared seeds for a prediction against the tiled head's scores (`seeds_for`)."""

HEAD_STEP_SEEDS = 8
"""Check 1's declared seeds for a training step's tiled head against the per-token tiled head's loss
and gradients (`seeds_for`)."""


def test_a_prediction_applies_the_head_once_and_scores_the_logits_it_returns():
    """A teacher's or a student's prediction (`LMObjective.predict`) holds the
    whole logits, and scores its losses from them: its forward contracts the
    vocabulary once and its gradient three times, and its losses and logits
    are the tiled head's and the model's own. A padding id past the
    vocabulary as the last target picks no column, as the tiled head's, so
    every loss stays finite."""
    from dew.objectives.lm.chunked import model_logits
    vocab = 61
    model, tokens = head_scored(vocab)
    tokens = tokens.at[:, -1].set(vocab)
    objective = LMObjective(model, 8, ema_decay=None, pad_id=vocab, head_tile=(4, 16))
    params = objective.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(1), None, training=False)

    def predicted(params):
        return objective.predict(params, {TEXT_KEY: tokens}, step, train=False)[2]

    def pulled(params):
        prediction = predicted(params)
        return jnp.sum(prediction.logits * 0.5) + jnp.sum(prediction.losses * prediction.weights)

    assert vocabulary_products(jax.make_jaxpr(predicted)(params).jaxpr, vocab) == 1
    assert vocabulary_products(jax.make_jaxpr(jax.grad(pulled))(params).jaxpr, vocab) == 3
    assert np.all(np.isfinite(predicted(params).losses))

    initialized = jax.jit(objective.init)

    def draw(seed):
        drawn = jax.random.randint(jax.random.key(seed + (1 << 20)), tokens.shape, 0, vocab)
        return initialized(jax.random.key(seed)), drawn.at[:, -1].set(vocab)

    def prediction_of(params, ids):
        prediction = objective.predict(params, {TEXT_KEY: ids}, step, train=False)[2]
        return prediction.losses, prediction.logits

    def scored(of):
        def tiled(params, ids):
            scores = of.token_scores(params, ids)
            return scores.losses, model_logits(of.model, params, scores.hidden)
        return tiled

    # Over drawn decoders and rows, the prediction's losses and logits on par with the tiled head's
    # losses and the model's own logits of its states (check 1).
    wide = LMObjective(model.clone(dtype=jnp.float64), 8, ema_decay=None, pad_id=vocab, head_tile=(4, 16))
    with jax.default_matmul_precision("highest"):
        assert_as_exact(without_ynn(prediction_of), without_ynn(scored(objective)), draw, "a prediction",
                        seeds=PREDICTION_SEEDS, truth_fn=scored(wide))


@pytest.mark.parametrize("tile_tokens", [4, 16])
def test_a_training_steps_tiled_head_contracts_the_vocabulary_three_times_and_matches_the_tiled_head(
        tile_tokens):
    """A training step whose head is tiled takes the head's gradient in the
    forward (`fused_head_loss`), a tile's tokens a chunk, all 16 in one where
    the tile holds them: beside blocks recomputed in the backward, the step
    contracts the whole vocabulary three times, the logits, the states'
    gradient and the head's, none again, and its loss and gradients are the
    per-token tiled head's, z-loss and padding included. Unfused
    (`unfuse_head`, the fit ladder's rung for a fused chunk that does not
    fit), the step takes the tiled head's gradient in its backward, a tile
    of the vocabulary at a time, to the same loss and gradients."""
    vocab = 61
    model, tokens = head_scored(vocab, remat="full")
    tokens = tokens.at[1, 6:].set(0)
    fused = LMObjective(model, 8, ema_decay=None, z_loss=1e-3, pad_id=0, head_tile=(tile_tokens, 16))
    unfused = LMObjective(model, 8, ema_decay=None, z_loss=1e-3, pad_id=0, head_tile=(tile_tokens, 16))
    assert unfused.unfuse_head() is not None and unfused.unfuse_head() is None
    params = fused.init(jax.random.key(0))

    def loss(objective, *, training):
        step = Step(jnp.asarray(0), jax.random.key(1), None, training=training)

        def scalar(params):
            stats, _ = objective.loss(params, {TEXT_KEY: tokens}, step)
            return objective.reduce_loss(stats)[0]
        return scalar

    def step_products(objective) -> int:
        program = jax.make_jaxpr(jax.grad(loss(objective, training=True)))(params)
        return vocabulary_products(program.jaxpr, vocab)

    assert step_products(fused) == 3
    # The tiled head contracts the vocabulary a tile of its columns at a time, so no product spans it.
    assert step_products(unfused) == 0
    initialized = jax.jit(fused.init)

    def draw(seed):
        drawn = jax.random.randint(jax.random.key(seed + (1 << 20)), tokens.shape, 0, vocab)
        return initialized(jax.random.key(seed)), drawn.at[1, 6:].set(0)

    def stepped(objective, *, training):
        def value_and_grad(params, ids):
            step = Step(jnp.asarray(0), jax.random.key(1), None, training=training)

            def scalar(params):
                stats, _ = objective.loss(params, {TEXT_KEY: ids}, step)
                return objective.reduce_loss(stats)[0]
            return jax.value_and_grad(scalar)(params)
        return value_and_grad

    # Over drawn decoders and rows, each step's loss and gradients on par with the per-token tiled
    # head's (check 1).
    wide = LMObjective(model.clone(dtype=jnp.float64), 8, ema_decay=None, z_loss=1e-3, pad_id=0,
                       head_tile=(tile_tokens, 16))
    with jax.default_matmul_precision("highest"):
        for objective, label in ((fused, "fused"), (unfused, "unfused")):
            assert_as_exact(without_ynn(stepped(objective, training=True)),
                            without_ynn(stepped(fused, training=False)), draw,
                            f"a {label} head's step, tiles of {tile_tokens}", seeds=HEAD_STEP_SEEDS,
                            truth_fn=stepped(wide, training=False))
