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
from flax import linen as nn
from recording import RecordingTracker

from dew.artifacts import TokenScores
from dew.data.chat import ROLES_KEY, Role
from dew.nn.protocols import OutputTable
from dew.objectives.base import Step
from dew.objectives.lm import TEXT_KEY, LMObjective, Perplexity, Samples
from dew.sampling import Sampling
from dew.training import Checkpoints, Layout, MeshSpec, Trainer

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


class TinyCausalLM(nn.Module):
    """A small causal transformer, standing in for `causal_transformer`.

    Same contract as the real backbone: int32 ids `[B, S]` in, float32 logits
    `[B, S, vocab]` out, `train` gating dropout, no path from a position to a
    later one, and the head split off behind `hidden_states` and
    `output_table` so the loss can score a vocabulary slice at a time. The
    head carries no bias, so the table's product is the whole projection.
    """

    vocab_size: int
    emb_features: int = 16
    num_layers: int = 2
    max_seq_len: int = 64
    dropout_rate: float = 0.0
    final_logit_softcap: float | None = None

    def setup(self):
        self.lm_head = nn.Dense(self.vocab_size, use_bias=False, dtype=jnp.float32,
                                name="lm_head")

    def __call__(self, tokens, train: bool = False, **packing):
        logits = self.lm_head(self.hidden_states(tokens, train=train, **packing))
        if self.final_logit_softcap is not None:
            cap = jnp.asarray(self.final_logit_softcap, jnp.float32)
            logits = cap * jnp.tanh(logits / cap)
        return logits.astype(jnp.float32)

    @nn.compact
    def hidden_states(self, tokens, train: bool = False):
        length = tokens.shape[1]
        positions = self.param("positions", nn.initializers.normal(0.02),
                               (self.max_seq_len, self.emb_features))
        x = nn.Embed(self.vocab_size, self.emb_features)(tokens) + positions[:length]
        causal = jnp.tril(jnp.ones((length, length), bool))
        dropout = nn.Dropout(self.dropout_rate, deterministic=not train)

        for _ in range(self.num_layers):
            h = nn.LayerNorm()(x)
            query = nn.Dense(self.emb_features)(h)
            key = nn.Dense(self.emb_features)(h)
            value = nn.Dense(self.emb_features)(h)
            scores = query @ jnp.swapaxes(key, -1, -2) / np.sqrt(self.emb_features)
            attention = jax.nn.softmax(jnp.where(causal, scores, -1e9), axis=-1)
            x = x + dropout(nn.Dense(self.emb_features)(attention @ value))

            h = nn.LayerNorm()(x)
            x = x + nn.Dense(self.emb_features)(nn.gelu(nn.Dense(2 * self.emb_features)(h)))

        return nn.LayerNorm()(x)

    def output_table(self):
        return OutputTable(self.lm_head.variables["params"]["kernel"], vocab_major=False,
                           softcap=self.final_logit_softcap)


def make_objective(seq=SEQ, model=None, **kwargs):
    return LMObjective(model or TinyCausalLM(vocab_size=VOCAB), seq, **kwargs)


def step_at(index=0, key=1, ema=None):
    return Step(step=jnp.asarray(index), key=jax.random.key(key), ema=ema)


def token_batch(batch=4, seq=SEQ, seed=0):
    """A `[B, seq + 1]` batch of ids, as the token pipeline packs them."""
    ids = np.random.RandomState(seed).randint(0, VOCAB, (batch, seq + 1))
    return {TEXT_KEY: jnp.asarray(ids, jnp.int32)}


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
    """Shifted cross entropy in numpy, from the model's own logits."""
    logits = np.asarray(logits, np.float64)
    largest = logits.max(axis=-1, keepdims=True)
    log_probs = logits - largest - np.log(np.exp(logits - largest).sum(axis=-1, keepdims=True))
    picked = -np.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    if weights is None:
        weights = np.ones_like(picked) if pad_id is None else (targets != pad_id).astype(np.float64)
    return float((picked * weights).sum() / weights.sum())


# --- the loss --------------------------------------------------------------

def test_loss_is_the_cross_entropy_of_the_shifted_sequence():
    objective = make_objective()
    params = objective.init(jax.random.key(0))
    batch = token_batch()
    tokens = np.asarray(batch[TEXT_KEY])

    loss, _ = objective.scalar_loss(params, batch, step_at())

    logits = objective.model.apply(params, jnp.asarray(tokens[:, :-1], jnp.int32))
    expected = reference_cross_entropy(logits, tokens[:, 1:])
    assert float(loss) == pytest.approx(expected, rel=1e-5)
    # Not the unshifted cross entropy, which is the mistake this guards
    unshifted = reference_cross_entropy(logits, tokens[:, :-1])
    assert abs(expected - unshifted) > 1e-3


def test_padded_targets_are_left_out_of_the_average():
    pad_id = 0
    objective = make_objective(pad_id=pad_id)
    params = objective.init(jax.random.key(0))
    batch = token_batch(seed=3)
    tokens = np.asarray(batch[TEXT_KEY])
    assert (tokens[:, 1:] == pad_id).any(), "this batch has no padding to skip"

    loss, _ = objective.scalar_loss(params, batch, step_at())

    logits = objective.model.apply(params, jnp.asarray(tokens[:, :-1], jnp.int32))
    masked = reference_cross_entropy(logits, tokens[:, 1:], pad_id=pad_id)
    unmasked = reference_cross_entropy(logits, tokens[:, 1:])
    assert float(loss) == pytest.approx(masked, rel=1e-5)
    assert abs(masked - unmasked) > 1e-4, "masking made no difference to check"


def test_aux_reports_perplexity_and_token_accuracy():
    objective = make_objective()
    params = objective.init(jax.random.key(0))
    batch = token_batch()

    loss, aux = objective.scalar_loss(params, batch, step_at())

    assert set(aux.metrics) == {"ce", "perplexity", "token_accuracy"}
    assert aux.variables is None
    assert float(aux.metrics["ce"]) == pytest.approx(float(loss))
    assert float(aux.metrics["perplexity"]) == pytest.approx(float(jnp.exp(loss)), rel=1e-5)

    logits = objective.model.apply(params, batch[TEXT_KEY][:, :-1])
    predicted = np.argmax(np.asarray(logits), axis=-1)
    accuracy = (predicted == np.asarray(batch[TEXT_KEY][:, 1:])).mean()
    assert float(aux.metrics["token_accuracy"]) == pytest.approx(accuracy)


def test_dropout_runs_on_the_step_key():
    """The loss passes the step key as the dropout rng, so a model with
    dropout is stochastic across keys and reproducible under one."""
    objective = make_objective(model=TinyCausalLM(vocab_size=VOCAB, dropout_rate=0.5))
    params = objective.init(jax.random.key(0))
    batch = token_batch()

    first, _ = objective.scalar_loss(params, batch, step_at(key=1))
    again, _ = objective.scalar_loss(params, batch, step_at(key=1))
    other, _ = objective.scalar_loss(params, batch, step_at(key=2))

    assert float(first) == pytest.approx(float(again))
    assert float(first) != pytest.approx(float(other))


def test_padded_tokens_are_left_out_of_the_accuracy_too():
    objective = make_objective(pad_id=0)
    params = objective.init(jax.random.key(0))
    batch = token_batch(seed=3)
    targets = np.asarray(batch[TEXT_KEY][:, 1:])
    assert (targets == 0).any(), "this batch has no padding to skip"

    _, aux = objective.scalar_loss(params, batch, step_at())

    logits = np.asarray(objective.model.apply(params, batch[TEXT_KEY][:, :-1]))
    kept = targets != 0
    expected = ((np.argmax(logits, axis=-1) == targets) & kept).sum() / kept.sum()
    assert float(aux.metrics["token_accuracy"]) == pytest.approx(expected)


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
    from dew.nn.backbones import CausalTransformer

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
    objective = make_objective()
    params = objective.init(jax.random.key(0))
    batch = token_batch()

    scores = objective.evaluate(params, batch, step_at())

    assert scores.losses.shape == scores.weights.shape == (4, SEQ)
    np.testing.assert_array_equal(scores.weights, 1.0)
    logits = objective.model.apply(params, batch[TEXT_KEY][:, :-1])
    expected = reference_cross_entropy(logits, np.asarray(batch[TEXT_KEY][:, 1:]))
    assert float(jnp.mean(scores.losses)) == pytest.approx(expected, rel=1e-5)


def test_evaluation_reads_the_ema_copy():
    objective = make_objective()
    params = objective.init(jax.random.key(0))
    ema = jax.tree.map(lambda leaf: leaf + 1, params)
    batch = token_batch()

    live = objective.evaluate(params, batch, step_at())
    averaged = objective.evaluate(params, batch, step_at(ema=ema))

    logits = objective.model.apply(ema, batch[TEXT_KEY][:, :-1])
    expected = reference_cross_entropy(logits, np.asarray(batch[TEXT_KEY][:, 1:]))
    assert float(jnp.mean(averaged.losses)) == pytest.approx(expected, rel=1e-4)
    assert float(jnp.mean(live.losses)) != pytest.approx(expected)


def test_scoring_with_samples_configured_does_not_decode():
    def fail_decode(ids):
        raise AssertionError("scoring decoded a preview")

    objective = make_objective(samples=Samples(prompt=[1, 2, 3], max_new_tokens=2,
                                              decode=fail_decode))
    params = objective.init(jax.random.key(0))
    scores = objective.evaluate(params, token_batch(), step_at())
    expected = objective.token_scores(params, token_batch()[TEXT_KEY])
    # Jitted scoring and eager scoring differ by <= 4.8e-7 in float32 on CPU.
    np.testing.assert_allclose(scores.losses, expected.losses, rtol=1e-6, atol=1e-6)


def test_preview_generates_reproducible_text_from_ema():
    from dew.nn.backbones.causal_transformer import CausalTransformer

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

    assert score == pytest.approx(np.exp((4 * 1.0 + 1 * 3.0) / 5))
    assert score != pytest.approx(np.exp(np.mean([1.0, 3.0])))


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
    assert expected != pytest.approx(wrong)
    assert metric.finalize(metric.merge(*values)) == pytest.approx(expected)


def test_the_validation_pass_scores_perplexity_per_token_and_logs_it():
    """Objective artifacts through the trainer's loop and into the metric:
    two batches of different padding weigh by their counted targets."""
    padded = {TEXT_KEY: jnp.asarray(np.where(np.arange(SEQ + 1) < 9, token_batch(BATCH)[TEXT_KEY], 0))}
    batches = [token_batch(BATCH, seed=1), padded]
    tracker = RecordingTracker()
    trainer = make_trainer(pad_id=0, tracker=tracker)
    state = trainer.fit(dataset(cycle_batches, val=lambda: iter(batches)), steps=1, log_every=1,
                        eval_every=1, metrics=(Perplexity(),))

    total, count = 0.0, 0.0
    for batch in batches:
        scores = trainer.objective.evaluate(state.variables, batch, step_at())
        total += float(jnp.sum(scores.losses * scores.weights))
        count += float(jnp.sum(scores.weights))
    logged = [s["val/perplexity"] for _, s in tracker.scalars if "val/perplexity" in s]
    assert len(logged) == 1
    # The pass scores the weights the step left, so it matches them.
    assert 0 < count < 2 * BATCH * SEQ
    assert logged[0] == pytest.approx(np.exp(total / count), rel=1e-6)


# --- packed batches --------------------------------------------------------

class PackedTinyLM(TinyCausalLM):
    """TinyCausalLM that honours the packing arguments the real backbone takes.

    The loss reads `hidden_states`, so that is where the packing arguments
    have to arrive; `__call__` forwards them, as `CausalTransformer` does.
    """

    @nn.compact
    def hidden_states(self, tokens, train: bool = False, positions=None,
                      segment_ids=None):
        length = tokens.shape[1]
        table = self.param("positions", nn.initializers.normal(0.02),
                           (self.max_seq_len, self.emb_features))
        index = jnp.arange(length) if positions is None else positions
        x = nn.Embed(self.vocab_size, self.emb_features)(tokens) + table[index]
        mask = jnp.tril(jnp.ones((length, length), bool))[None]
        if segment_ids is not None:
            mask = mask & ((segment_ids[:, :, None] == segment_ids[:, None, :])
                           & (segment_ids[:, :, None] != 0))
        dropout = nn.Dropout(self.dropout_rate, deterministic=not train)

        for _ in range(self.num_layers):
            h = nn.LayerNorm()(x)
            query, key, value = (nn.Dense(self.emb_features)(h) for _ in range(3))
            scores = query @ jnp.swapaxes(key, -1, -2) / np.sqrt(self.emb_features)
            attention = jax.nn.softmax(jnp.where(mask, scores, -1e9), axis=-1)
            x = x + dropout(nn.Dense(self.emb_features)(attention @ value))

            h = nn.LayerNorm()(x)
            x = x + nn.Dense(self.emb_features)(nn.gelu(nn.Dense(2 * self.emb_features)(h)))

        return nn.LayerNorm()(x)


def packed_objective(seq=SEQ, **kwargs):
    return LMObjective(PackedTinyLM(vocab_size=VOCAB), seq, **kwargs)


def documents(first=7, second=3, seq=SEQ):
    """A packed row of two documents, plus the padding that follows them."""
    rs = np.random.RandomState(11)
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
    """Per-token losses and the weight the objective gives each of them."""
    inputs, targets = tokens[:, :-1], tokens[:, 1:]
    logits = objective.model.apply(params, inputs, positions=positions[:, :-1],
                                   segment_ids=segment_ids[:, :-1])
    losses = optax.softmax_cross_entropy_with_integer_labels(
        logits.astype(jnp.float32), targets)
    weights = ((segment_ids[:, 1:] == segment_ids[:, :-1])
               & (segment_ids[:, 1:] != 0)).astype(losses.dtype)
    return np.asarray(losses[0]), np.asarray(weights[0])


def weighted_mean(losses, weights):
    return float(jnp.sum(losses * weights) / jnp.sum(weights))


def test_a_packed_batch_ignores_the_boundary_and_the_padding():
    objective = packed_objective()
    params = objective.init(jax.random.key(0))
    tokens, segment_ids, positions, (doc_a, _) = documents()

    scored_losses, scored_weights = objective.token_scores(
        params, tokens, segment_ids=segment_ids, positions=positions)[:2]

    losses, weights = counted_losses(objective, params, tokens, segment_ids, positions)
    # The last token of the first document predicts the first of the second,
    # which is the one transition a packed row must not train on.
    boundary = len(doc_a) - 1
    assert weights[boundary] == 0
    assert weights[:boundary].all() and weights[boundary + 1] == 1
    assert weights[len(doc_a) + 2:].sum() == 0, "padding was trained on"
    np.testing.assert_array_equal(np.asarray(scored_weights[0]), weights)
    assert weighted_mean(scored_losses, scored_weights) == pytest.approx(
        float((losses * weights).sum() / weights.sum()), rel=1e-6)


def test_a_packed_batch_scores_like_its_documents_alone():
    """Packing is only a layout: the two documents' losses have to be the ones
    they get on their own, from the whole logits and through the same chunked
    head, and the boundary target is dropped on both sides."""
    objective = packed_objective()
    params = objective.init(jax.random.key(0))
    tokens, segment_ids, positions, (doc_a, doc_b) = documents()

    losses, weights = objective.token_scores(
        params, tokens, segment_ids=segment_ids, positions=positions)[:2]
    ce = weighted_mean(losses, weights)

    total, chunked, count = 0.0, 0.0, 0.0
    for document in (doc_a, doc_b):
        alone_tokens, alone_segments, alone_positions = alone = only_document(document)
        alone_losses, alone_weights = counted_losses(objective, params, *alone)
        total += float((alone_losses * alone_weights).sum())
        count += float(alone_weights.sum())
        head_losses, head_weights = objective.token_scores(
            params, alone_tokens, segment_ids=alone_segments, positions=alone_positions)[:2]
        # Every transition inside the document; its last target is padding.
        assert float(jnp.sum(head_weights)) == len(document) - 1
        chunked += weighted_mean(head_losses, head_weights) * (len(document) - 1)

    packed_losses, packed_weights = counted_losses(
        objective, params, tokens, segment_ids, positions)
    assert count == packed_weights.sum() == len(doc_a) + len(doc_b) - 2
    assert ce == pytest.approx(total / count, rel=1e-5)
    assert ce == pytest.approx(float((packed_losses * packed_weights).sum() / count), rel=1e-6)
    # Largest relative difference observed on CPU: 7.1e-08.
    assert ce == pytest.approx(chunked / count, rel=1e-6)


def test_a_packed_batch_reaches_the_objective_through_the_batch_dict():
    objective = packed_objective()
    params = objective.init(jax.random.key(0))
    tokens, segment_ids, positions, _ = documents()
    batch = {TEXT_KEY: tokens, "text_segment_ids": segment_ids,
             "text_positions": positions}

    packed, _ = objective.scalar_loss(params, batch, step_at())
    unpacked, _ = objective.scalar_loss(params, {TEXT_KEY: tokens}, step_at())

    losses, weights = objective.token_scores(
        params, tokens, train=True, rngs={"dropout": jax.random.key(1)},
        segment_ids=segment_ids, positions=positions)[:2]
    assert float(packed) == pytest.approx(weighted_mean(losses, weights))
    assert abs(float(packed) - float(unpacked)) > 1e-3, (
        "the packed keys made no difference to the loss")


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
    whose role matches, and the reported accuracy counts the same targets."""
    objective = make_objective(loss_role=Role.ASSISTANT)
    params = objective.init(jax.random.key(0))
    batch = assistant_batch()
    tokens = np.asarray(batch[TEXT_KEY])

    loss, aux = objective.scalar_loss(params, batch, step_at())

    logits = objective.model.apply(params, batch[TEXT_KEY][:, :-1])
    kept = np.asarray(batch[ROLES_KEY])[:, 1:] == Role.ASSISTANT
    assert kept.sum() > 0 and (~kept).sum() > 0, "the batch counts nothing out"
    expected = reference_cross_entropy(logits, tokens[:, 1:], weights=kept)
    assert float(loss) == pytest.approx(expected, rel=1e-5)
    full = reference_cross_entropy(logits, tokens[:, 1:])
    assert abs(expected - full) > 1e-4, "the role mask made no difference to check"
    accuracy = ((np.argmax(np.asarray(logits), axis=-1) == tokens[:, 1:]) & kept).sum() / kept.sum()
    assert float(aux.metrics["token_accuracy"]) == pytest.approx(accuracy)


def test_a_roles_column_is_ignored_without_loss_role():
    """The same packed SFT batch trains every target under a plain
    LMObjective: the column only matters when a role is asked for."""
    objective = make_objective()
    params = objective.init(jax.random.key(0))
    batch = assistant_batch()

    loss, _ = objective.scalar_loss(params, batch, step_at())

    logits = objective.model.apply(params, batch[TEXT_KEY][:, :-1])
    expected = reference_cross_entropy(logits, np.asarray(batch[TEXT_KEY][:, 1:]))
    assert float(loss) == pytest.approx(expected, rel=1e-5)


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
    `train.py` sum it (tools/distillation_reference.py:z_loss_reference);
    loss within 1e-6 of the float64 oracle (observed 3.1e-7), and zero keeps the loss the
    plain cross entropy to the bit."""
    from tools.distillation_reference import z_loss_reference

    pad_id, coefficient = 0, 1e-2
    batch = token_batch()
    tokens = np.asarray(batch[TEXT_KEY])
    plain = make_objective(pad_id=pad_id)
    params = plain.init(jax.random.key(0))
    logits = np.asarray(plain.model.apply(params, jnp.asarray(tokens[:, :-1], jnp.int32)))
    weights = (tokens[:, 1:] != pad_id).astype(np.float64)
    expected, z_term, _ = z_loss_reference(logits, tokens[:, 1:], weights, coefficient)

    loss, aux = make_objective(pad_id=pad_id, z_loss=coefficient).scalar_loss(params, batch, step_at())
    without, plain_aux = plain.scalar_loss(params, batch, step_at())

    assert float(loss) == pytest.approx(expected, abs=1e-6)
    assert float(aux.metrics["z_loss"]) == pytest.approx(z_term, abs=1e-6)
    assert float(aux.metrics["ce"]) == pytest.approx(float(plain_aux.metrics["ce"]))
    assert "z_loss" not in plain_aux.metrics
    assert float(without) == pytest.approx(reference_cross_entropy(logits, tokens[:, 1:], pad_id))
    assert abs(float(loss) - float(without)) > 1e-4, "the term made no difference to check"


def test_z_loss_gradient_carries_the_reference_factor():
    """The oracle's logit gradient, `(1 + 2 z log Z) softmax - onehot`, pulled
    back through the model agrees with the objective's parameter gradient
    within 1e-4 scaled; observed at most 6.8e-8."""
    from tools.distillation_reference import z_loss_reference

    coefficient = 1e-2
    batch = token_batch()
    tokens = np.asarray(batch[TEXT_KEY])
    objective = make_objective(z_loss=coefficient)
    params = objective.init(jax.random.key(0))
    ids = jnp.asarray(tokens[:, :-1], jnp.int32)
    logits, pullback = jax.vjp(lambda p: objective.model.apply({"params": p}, ids), params["params"])
    _, _, cotangent = z_loss_reference(np.asarray(logits), tokens[:, 1:],
                                       np.ones(tokens[:, 1:].shape), coefficient)
    expected, = pullback(jnp.asarray(cotangent, jnp.float32))
    actual = jax.grad(lambda p: objective.scalar_loss({"params": p}, batch, step_at())[0])(params["params"])

    for want, have in zip(jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True):
        scale = max(float(jnp.abs(want).max()), 1.0)
        assert float(jnp.abs(have - want).max()) <= 1e-4 * scale


def test_router_z_loss_adds_each_routers_mean_squared_log_partition():
    """The term is the coefficient times each router's squared logsumexp of
    its gate logits, averaged over the positions it routed and summed over
    routers, beside the cross entropy and not inside its token mean."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.decoder_block import Mixture
    model = CausalTransformer(vocab_size=64, emb_features=32, num_layers=2, num_heads=2,
                              max_seq_len=8, mixture=Mixture(experts=4, top_k=2, expert_features=16))
    tokens = jax.random.randint(jax.random.key(1), (2, 9), 0, 64)
    plain = LMObjective(model, 8, ema_decay=None)
    zed = LMObjective(model, 8, ema_decay=None, router_z_loss=0.3)
    params = plain.init(jax.random.key(0))
    base, _ = plain.reduce_loss(plain.loss(params, {TEXT_KEY: tokens}, step_at())[0])
    stats, aux = zed.loss(params, {TEXT_KEY: tokens}, step_at())
    total, _ = zed.reduce_loss(stats)

    # The oracle: each layer's gate logits recomputed from its kernel and
    # the normed input the router saw, captured by flax's intermediates.
    # The block's norm owns its residual seam, which returns the stream beside the normed input.
    _, state = model.apply(params, tokens[:, :-1], mutable=["intermediates"],
                           capture_intermediates=lambda module, method: method == "seam")
    expected = 0.0
    for index in range(2):
        layer = state["intermediates"][f"layers_{index}"]
        normed = layer["post_attention_layernorm"]["seam"][0][1]
        kernel = params["params"][f"layers_{index}"]["mlp"]["gate"]["kernel"]
        logits = normed.astype(jnp.float32) @ kernel
        expected += 0.3 * float(jnp.mean(jnp.square(jax.nn.logsumexp(logits, -1))))
    np.testing.assert_allclose(float(total - base), expected, rtol=1e-5)
    np.testing.assert_allclose(float(aux.metrics["router_z_loss"]), expected, rtol=1e-5)


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


def test_a_prediction_applies_the_head_once_and_scores_the_logits_it_returns():
    """A teacher's or a student's prediction (`LMObjective.predict`) holds the
    whole logits, and scores its losses from them: its forward contracts the
    vocabulary once and its gradient three times, and its losses and logits
    are the tiled head's and the model's own."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.objectives.lm.chunked import model_logits
    vocab = 61
    model = CausalTransformer(vocab_size=vocab, emb_features=24, num_layers=2, num_heads=2, mlp_features=32,
                              max_seq_len=8, tie_embeddings=False, qk_norm=False, attention_impl="reference")
    tokens = jax.random.randint(jax.random.key(1), (2, 9), 0, vocab).at[1, 6:].set(0)
    objective = LMObjective(model, 8, ema_decay=None, pad_id=0, head_tile=(4, 16))
    params = objective.init(jax.random.key(0))
    step = Step(jnp.asarray(0), jax.random.key(1), None, training=False)

    def predicted(params):
        return objective.predict(params, {TEXT_KEY: tokens}, step, train=False)[2]

    def pulled(params):
        prediction = predicted(params)
        return jnp.sum(prediction.logits * 0.5) + jnp.sum(prediction.losses * prediction.weights)

    assert vocabulary_products(jax.make_jaxpr(predicted)(params).jaxpr, vocab) == 1
    assert vocabulary_products(jax.make_jaxpr(jax.grad(pulled))(params).jaxpr, vocab) == 3
    with jax.default_matmul_precision("highest"):
        prediction = predicted(params)
        scores = objective.token_scores(params, tokens)
        whole = model_logits(model, params, scores.hidden)
    np.testing.assert_allclose(prediction.losses, scores.losses, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(prediction.logits, whole, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("tile_tokens", [4, 16])
def test_a_training_steps_tiled_head_contracts_the_vocabulary_three_times_and_matches_the_tiled_head(
        tile_tokens):
    """A training step whose head is tiled takes the head's gradient in the
    forward (`fused_head_loss`), a tile's tokens a chunk: beside blocks
    recomputed in the backward, the step contracts the whole vocabulary
    three times, the logits, the states' gradient and the head's, none again,
    and its loss and gradients are the per-token tiled head's, z-loss and
    padding included. A tile that holds all 16 tokens would make one chunk
    holding every logit, so the step keeps the tiled head, contracting the
    vocabulary as it does without a z-loss, which never fuses."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    vocab = 61
    model = CausalTransformer(vocab_size=vocab, emb_features=24, num_layers=2, num_heads=2, mlp_features=32,
                              max_seq_len=8, tie_embeddings=False, remat="full", qk_norm=False,
                              attention_impl="reference")
    tokens = jax.random.randint(jax.random.key(1), (2, 9), 0, vocab).at[1, 6:].set(0)
    fused = LMObjective(model, 8, ema_decay=None, z_loss=1e-3, pad_id=0, head_tile=(tile_tokens, 16))
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

    if tile_tokens == 4:
        assert step_products(fused) == 3
    else:
        tiled = LMObjective(model, 8, ema_decay=None, pad_id=0, head_tile=(tile_tokens, 16))
        assert step_products(fused) == step_products(tiled) != 3
    with jax.default_matmul_precision("highest"):
        mine = jax.value_and_grad(loss(fused, training=True))(params)
        theirs = jax.value_and_grad(loss(fused, training=False))(params)
    for left, right in zip(jax.tree.leaves(mine), jax.tree.leaves(theirs), strict=True):
        np.testing.assert_allclose(left, right, rtol=2e-5, atol=2e-6)
