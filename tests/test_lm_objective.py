"""Language modelling: the shifted cross entropy, and a trainer with no input config.

The model here is a small causal stack that honors the backbone's contract
(int32 ids in, float32 logits out, a `train` flag for dropout) and
`dew.sampling.text.generate` is recorded rather than run, so what is under
test is the objective: that the loss is the cross entropy of the shifted
sequence and nothing else, that padding is excluded only when a pad id is
named, and that the trainer drives it on both a data-parallel and an FSDP mesh
without a DiffusionInputConfig to describe the inputs. The real sampler runs
in test_lm_recipe.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn

from dew.eval import get_perplexity_metric
from dew.objectives.base import EMASpec, Objective
from dew.objectives.lm import LMObjective, TEXT_KEY
from dew.training import ObjectiveTrainer
from dew.training.objective_trainer import TrainState

VOCAB = 8
SEQ = 16
BATCH = 8
STEPS = 200
# The test model's parameters are far below the production shard threshold, so
# lower it or "FSDP on" would silently mean "everything replicated".
TINY = 64


class TinyCausalLM(nn.Module):
    """A small causal transformer, standing in for `causal_transformer`.

    Same call contract as the real backbone: int32 ids `[B, S]` in, float32
    logits `[B, S, vocab]` out, `train` gating dropout, and no path from a
    position to a later one.
    """

    vocab_size: int
    emb_features: int = 16
    num_layers: int = 2
    max_seq_len: int = 64
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, tokens, train: bool = False):
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

        return nn.Dense(self.vocab_size)(nn.LayerNorm()(x)).astype(jnp.float32)


class WandbRecorder:
    """Just enough of a wandb run to see what validation logged."""

    def __init__(self):
        self.logged = {}

    def log(self, values, step=None):
        self.logged.update(values)


class ShapelessObjective(Objective):
    """An objective that says nothing about its inputs."""

    tag = "shapeless"

    def __init__(self):
        self.ema = EMASpec(decay=lambda step: 0.0)

    def init_params(self, rng):
        return {"params": {"w": jnp.zeros(())}}

    def loss(self, params, ema_params, batch, rng, step):
        return jnp.zeros(()), {}

    def make_validation_step(self, **kwargs):
        return lambda val_state, batch: None


def make_objective(seq=SEQ, model=None, **kwargs):
    return LMObjective(model or TinyCausalLM(vocab_size=VOCAB), seq,
                       vocab_size=VOCAB, **kwargs)


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


def reference_cross_entropy(logits, targets, pad_id=None):
    """Shifted cross entropy in numpy, from the model's own logits."""
    logits = np.asarray(logits, np.float64)
    largest = logits.max(axis=-1, keepdims=True)
    log_probs = logits - largest - np.log(np.exp(logits - largest).sum(axis=-1, keepdims=True))
    picked = -np.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    weights = np.ones_like(picked) if pad_id is None else (targets != pad_id).astype(np.float64)
    return float((picked * weights).sum() / weights.sum())


def make_val_state(params, ema_params=None, rngs=None):
    return TrainState.create(
        apply_fn=None, params=params,
        ema_params=params if ema_params is None else ema_params,
        tx=optax.sgd(0.0), rngs=jax.random.PRNGKey(0) if rngs is None else rngs,
        metrics=None, dynamic_scale=None)


@pytest.fixture
def recorded_generate(monkeypatch):
    """Stand in for `dew.sampling.text.generate` and record how it was called."""
    calls = []

    def generate(model, params, prompt, max_new_tokens, *, rng, temperature=1.0,
                 top_k=None):
        calls.append({"model": model, "params": params, "prompt": prompt,
                      "max_new_tokens": max_new_tokens, "rng": rng,
                      "temperature": temperature, "top_k": top_k})
        return jnp.concatenate(
            [prompt, jnp.zeros((prompt.shape[0], max_new_tokens), jnp.int32)], axis=1)

    import dew.sampling.text as text_sampler
    monkeypatch.setattr(text_sampler, "generate", generate)
    return calls


# --- the loss --------------------------------------------------------------

def test_loss_is_the_cross_entropy_of_the_shifted_sequence():
    objective = make_objective()
    params = objective.init_params(jax.random.PRNGKey(0))
    batch = token_batch()
    tokens = np.asarray(batch[TEXT_KEY])

    loss, _ = objective.loss(params, params, batch, jax.random.PRNGKey(1), 0)

    logits = objective.model.apply(params, jnp.asarray(tokens[:, :-1], jnp.int32))
    expected = reference_cross_entropy(logits, tokens[:, 1:])
    assert float(loss) == pytest.approx(expected, rel=1e-5)
    # Not the unshifted cross entropy, which is the mistake this guards
    unshifted = reference_cross_entropy(logits, tokens[:, :-1])
    assert abs(expected - unshifted) > 1e-3


def test_padded_targets_are_left_out_of_the_average():
    pad_id = 0
    objective = make_objective(pad_id=pad_id)
    params = objective.init_params(jax.random.PRNGKey(0))
    batch = token_batch(seed=3)
    tokens = np.asarray(batch[TEXT_KEY])
    assert (tokens[:, 1:] == pad_id).any(), "this batch has no padding to skip"

    loss, _ = objective.loss(params, params, batch, jax.random.PRNGKey(1), 0)

    logits = objective.model.apply(params, jnp.asarray(tokens[:, :-1], jnp.int32))
    masked = reference_cross_entropy(logits, tokens[:, 1:], pad_id=pad_id)
    unmasked = reference_cross_entropy(logits, tokens[:, 1:])
    assert float(loss) == pytest.approx(masked, rel=1e-5)
    assert abs(masked - unmasked) > 1e-4, "masking made no difference to check"


def test_a_batch_of_only_padding_does_not_divide_by_zero():
    objective = make_objective(pad_id=5)
    params = objective.init_params(jax.random.PRNGKey(0))
    batch = {TEXT_KEY: jnp.full((2, SEQ + 1), 5, jnp.int32)}

    loss, aux = objective.loss(params, params, batch, jax.random.PRNGKey(1), 0)
    assert float(loss) == 0.0 and bool(jnp.isfinite(aux["perplexity"]))


def test_aux_reports_perplexity_and_token_accuracy():
    objective = make_objective()
    params = objective.init_params(jax.random.PRNGKey(0))
    batch = token_batch()

    loss, aux = objective.loss(params, params, batch, jax.random.PRNGKey(1), 0)

    assert set(aux) == {"ce", "perplexity", "token_accuracy"}
    assert float(aux["ce"]) == pytest.approx(float(loss))
    assert float(aux["perplexity"]) == pytest.approx(float(jnp.exp(loss)), rel=1e-5)

    logits = objective.model.apply(params, batch[TEXT_KEY][:, :-1])
    predicted = np.argmax(np.asarray(logits), axis=-1)
    accuracy = (predicted == np.asarray(batch[TEXT_KEY][:, 1:])).mean()
    assert float(aux["token_accuracy"]) == pytest.approx(accuracy)


def test_a_batch_with_no_room_for_the_shift_is_rejected():
    objective = make_objective()
    params = objective.init_params(jax.random.PRNGKey(0))
    batch = {TEXT_KEY: jnp.zeros((2, SEQ), jnp.int32)}

    with pytest.raises(ValueError, match=f"{SEQ + 1} ids per row"):
        objective.loss(params, params, batch, jax.random.PRNGKey(0), 0)


def test_dropout_runs_on_the_step_key():
    """The loss passes a dropout rng, so a model with dropout is stochastic."""
    objective = make_objective(model=TinyCausalLM(vocab_size=VOCAB, dropout_rate=0.5))
    params = objective.init_params(jax.random.PRNGKey(0))
    batch = token_batch()

    first, _ = objective.loss(params, params, batch, jax.random.PRNGKey(1), 0)
    again, _ = objective.loss(params, params, batch, jax.random.PRNGKey(1), 0)
    other, _ = objective.loss(params, params, batch, jax.random.PRNGKey(2), 0)

    assert float(first) == pytest.approx(float(again))
    assert float(first) != pytest.approx(float(other))


def test_cross_entropy_is_computed_in_float32_under_bfloat16():
    """Params stay fp32 and the loss is fp32 even when the model computes in bf16."""
    class Bf16LM(TinyCausalLM):
        @nn.compact
        def __call__(self, tokens, train: bool = False):
            embedded = nn.Embed(self.vocab_size, self.emb_features,
                                dtype=jnp.bfloat16)(tokens)
            return nn.Dense(self.vocab_size, dtype=jnp.bfloat16)(embedded)

    objective = make_objective(model=Bf16LM(vocab_size=VOCAB))
    params = objective.init_params(jax.random.PRNGKey(0))
    logits = objective.model.apply(params, token_batch()[TEXT_KEY][:, :-1])
    loss, aux = objective.loss(params, params, token_batch(), jax.random.PRNGKey(0), 0)

    assert logits.dtype == jnp.bfloat16
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))
    assert loss.dtype == jnp.float32 and aux["ce"].dtype == jnp.float32


# --- what the trainer needs ------------------------------------------------

def test_input_shapes_declare_one_int32_token_sequence():
    assert make_objective().input_shapes == {"tokens": ((SEQ,), jnp.int32)}


def test_the_ema_tracks_the_whole_parameter_tree():
    ema = make_objective(ema_decay=0.995).ema
    assert ema.path == ()
    assert float(ema.decay(0)) == pytest.approx(0.995)
    assert float(ema.decay(10_000)) == pytest.approx(0.995)


def make_trainer(tmp_path, fsdp_size=1, seq=SEQ, learning_rate=3e-3, **objective_kwargs):
    model = TinyCausalLM(vocab_size=VOCAB)
    objective = LMObjective(model, seq, vocab_size=VOCAB, **objective_kwargs)
    return ObjectiveTrainer(
        model=model,
        optimizer=optax.adam(learning_rate),
        rngs=jax.random.PRNGKey(0),
        input_config=None,
        objective=objective,
        name=f"lm-smoke-fsdp{fsdp_size}",
        wandb_config=None,
        distributed_training=True,
        fsdp_size=fsdp_size,
        fsdp_min_param_size=TINY,
        checkpoint_base_path=str(tmp_path),
        eval_metrics=[get_perplexity_metric()],
    )


def test_the_trainer_takes_its_input_shapes_from_the_objective(tmp_path):
    trainer = make_trainer(tmp_path)
    assert trainer.input_shapes == {"tokens": ((SEQ,), jnp.int32)}
    ones = trainer.get_input_ones()
    assert ones["tokens"].shape == (1, SEQ)
    assert ones["tokens"].dtype == jnp.int32, "an init batch of floats is not token ids"


def test_a_trainer_with_neither_an_input_config_nor_input_shapes_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="input_shapes"):
        ObjectiveTrainer(
            model=TinyCausalLM(vocab_size=VOCAB),
            optimizer=optax.adam(1e-3),
            rngs=jax.random.PRNGKey(0),
            objective=ShapelessObjective(),
            wandb_config=None,
            distributed_training=False,
            checkpoint_base_path=str(tmp_path),
        )


@pytest.mark.parametrize("fsdp_size", [1, 4])
def test_the_objective_trains_through_the_trainer(tmp_path, fsdp_size):
    """The whole seam on the simulated mesh: 8x1 data parallel, then 2x4 FSDP.

    Compiling is not the claim. The loss on the copy task has to actually fall,
    the parameters have to stay on the mesh they were built on, and the run has
    to leave a checkpoint behind.
    """
    trainer = make_trainer(tmp_path, fsdp_size=fsdp_size)
    assert trainer.mesh.devices.shape == (jax.device_count() // fsdp_size, fsdp_size)
    specs = [p.sharding.spec for p in jax.tree.leaves(trainer.state.params)]
    if fsdp_size > 1:
        sharded = [p for p in jax.tree.leaves(trainer.state.params)
                   if 'fsdp' in str(p.sharding.spec)]
        assert sharded, "no parameter was sharded over the fsdp axis"

    batch = next(cycle_batches(seed=7))
    scored = jax.jit(lambda params: trainer.objective.loss(
        params, params, batch, jax.random.PRNGKey(0), 0)[0])
    before = float(scored(trainer.state.params))

    state = trainer.fit(
        {"train": cycle_batches, "train_len": 64, "local_batch_size": BATCH},
        training_steps_per_epoch=STEPS, epochs=1, val_steps_per_epoch=0)
    after = float(scored(state.params))

    assert before > 1.5, "the untrained model already knew the task"
    assert after < 0.3, f"the loss did not fall on the copy task: {before} -> {after}"
    assert specs == [p.sharding.spec for p in jax.tree.leaves(state.params)]
    trainer.wait_for_checkpoints()
    assert trainer.checkpointer.latest_step() == STEPS


# --- validation ------------------------------------------------------------

def test_validation_reports_the_teacher_forced_cross_entropy():
    objective = make_objective()
    params = objective.init_params(jax.random.PRNGKey(0))
    batch = token_batch()

    artifacts = objective.make_validation_step()(make_val_state(params), batch)

    assert set(artifacts) == {"ce"}
    logits = objective.model.apply(params, batch[TEXT_KEY][:, :-1])
    expected = reference_cross_entropy(logits, np.asarray(batch[TEXT_KEY][:, 1:]))
    assert float(artifacts["ce"]) == pytest.approx(expected, rel=1e-5)


def test_validation_generates_from_the_ema_copy(recorded_generate):
    objective = make_objective(samples={
        "prompt": [1, 2, 3], "max_new_tokens": 4, "temperature": 0.0,
        "decode": lambda ids: "".join(str(i) for i in ids)})
    params = objective.init_params(jax.random.PRNGKey(0))
    ema_params = jax.tree.map(lambda leaf: leaf + 1, params)

    artifacts = objective.make_validation_step()(
        make_val_state(params, ema_params), token_batch())

    assert artifacts["tokens"].shape == (1, 3 + 4)
    call, = recorded_generate
    assert call["params"] is ema_params, "samples were drawn from the live params"
    assert call["model"] is objective.model
    assert np.array_equal(np.asarray(call["prompt"]), [[1, 2, 3]])
    assert call["max_new_tokens"] == 4 and call["temperature"] == 0.0
    assert call["top_k"] is None


def test_validation_falls_back_to_the_live_params_without_an_ema(recorded_generate):
    objective = make_objective(samples={"prompt": [[1, 2], [3, 4]], "max_new_tokens": 2})
    params = objective.init_params(jax.random.PRNGKey(0))
    state = make_val_state(params).replace(ema_params=None)

    artifacts = objective.make_validation_step()(state, token_batch())

    assert artifacts["tokens"].shape == (2, 4)
    assert recorded_generate[0]["params"] is params
    assert recorded_generate[0]["temperature"] == 1.0


def test_prompts_that_cannot_be_batched_are_rejected():
    ragged = make_objective(samples={"prompt": [[1, 2], [3]], "max_new_tokens": 1})
    with pytest.raises(ValueError, match="equal length"):
        ragged.make_validation_step()

    empty = make_objective(samples={"prompt": [], "max_new_tokens": 1})
    with pytest.raises(ValueError, match="non-empty"):
        empty.make_validation_step()


def test_log_validation_artifacts_reports_decoded_text():
    objective = make_objective(samples={
        "prompt": [1, 2], "max_new_tokens": 3,
        "decode": lambda ids: "|".join(str(i) for i in ids)})
    run = WandbRecorder()

    objective.log_validation_artifacts(run, {
        "ce": jnp.asarray(1.5),
        "tokens": jnp.asarray([[1, 2, 0, 1, 2]], jnp.int32),
    }, step=7)

    assert list(run.logged) == ["val/samples"]
    assert run.logged["val/samples"].data == [[0, "1|2|0|1|2"]]


def test_log_validation_artifacts_without_samples_is_empty():
    run = WandbRecorder()
    make_objective().log_validation_artifacts(run, {"ce": jnp.asarray(0.5)}, step=1)
    assert run.logged == {}


def test_perplexity_metric_reads_the_cross_entropy_artifact():
    metric = get_perplexity_metric()
    assert metric.name == "perplexity" and metric.higher_is_better is False
    assert metric.function({"ce": jnp.asarray(2.0)}, None) == pytest.approx(2.0)
    assert metric.reducer([0.0, 2.0]) == pytest.approx(np.exp(1.0))


def test_the_validation_loop_scores_perplexity(tmp_path):
    """Objective artifacts through the trainer's loop and into the metric."""
    trainer = make_trainer(tmp_path)
    trainer.validation_loop(trainer.state, trainer._define_validation_step(),
                            lambda: iter([token_batch(batch=BATCH)]), 1, 0)

    ce = float(trainer.objective.make_validation_step()(
        trainer.state, token_batch(batch=BATCH))["ce"])
    assert trainer.best_val_metrics["val/perplexity"] == pytest.approx(
        float(np.exp(ce)), rel=1e-4)


def test_validation_exponentiates_after_averaging_cross_entropy(tmp_path):
    trainer = make_trainer(tmp_path)
    cross_entropies = iter([jnp.asarray(0.0), jnp.asarray(2.0)])

    trainer.validation_loop(
        trainer.state,
        lambda state, batch: {"ce": next(cross_entropies)},
        None,
        val_steps_per_epoch=2,
        current_step=0,
    )

    expected = np.exp(np.mean([0.0, 2.0]))
    wrong = np.mean(np.exp([0.0, 2.0]))
    assert expected != pytest.approx(wrong)
    assert trainer.best_val_metrics["val/perplexity"] == pytest.approx(expected)


# --- packed batches --------------------------------------------------------

class PackedTinyLM(TinyCausalLM):
    """TinyCausalLM that honours the packing arguments the real backbone takes."""

    @nn.compact
    def __call__(self, tokens, train: bool = False, positions=None, segment_ids=None):
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

        return nn.Dense(self.vocab_size)(nn.LayerNorm()(x)).astype(jnp.float32)


def packed_objective(seq=SEQ, **kwargs):
    return LMObjective(PackedTinyLM(vocab_size=VOCAB), seq, vocab_size=VOCAB, **kwargs)


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


def test_a_packed_batch_ignores_the_boundary_and_the_padding():
    objective = packed_objective()
    params = objective.init_params(jax.random.PRNGKey(0))
    tokens, segment_ids, positions, (doc_a, _) = documents()

    ce, aux = objective.shifted_cross_entropy(
        params, tokens, segment_ids=segment_ids, positions=positions)

    losses, weights = counted_losses(objective, params, tokens, segment_ids, positions)
    # The last token of the first document predicts the first of the second,
    # which is the one transition a packed row must not train on.
    boundary = len(doc_a) - 1
    assert weights[boundary] == 0
    assert weights[:boundary].all() and weights[boundary + 1] == 1
    assert weights[len(doc_a) + 2:].sum() == 0, "padding was trained on"
    assert float(ce) == pytest.approx(
        float((losses * weights).sum() / weights.sum()), rel=1e-6)
    assert float(aux["token_accuracy"]) <= 1.0


def test_a_packed_batch_scores_like_its_documents_alone():
    """Packing is only a layout: the two documents' losses have to be the ones
    they get on their own, and the boundary target is dropped on both sides."""
    objective = packed_objective()
    params = objective.init_params(jax.random.PRNGKey(0))
    tokens, segment_ids, positions, (doc_a, doc_b) = documents()

    ce, _ = objective.shifted_cross_entropy(
        params, tokens, segment_ids=segment_ids, positions=positions)

    total, count = 0.0, 0.0
    for document in (doc_a, doc_b):
        alone = only_document(document)
        losses, weights = counted_losses(objective, params, *alone)
        total += float((losses * weights).sum())
        count += float(weights.sum())

    packed_losses, packed_weights = counted_losses(
        objective, params, tokens, segment_ids, positions)
    assert count == packed_weights.sum() == len(doc_a) + len(doc_b) - 2
    # Position by position, so a mismatch says which target moved.
    assert float(ce) == pytest.approx(total / count, rel=1e-5)
    assert float(ce) == pytest.approx(
        float((packed_losses * packed_weights).sum() / count), rel=1e-6)


def test_a_packed_batch_reaches_the_objective_through_the_batch_dict():
    objective = packed_objective()
    params = objective.init_params(jax.random.PRNGKey(0))
    tokens, segment_ids, positions, _ = documents()
    batch = {TEXT_KEY: tokens, "text_segment_ids": segment_ids,
             "text_positions": positions}

    packed, _ = objective.loss(params, params, batch, jax.random.PRNGKey(1), 0)
    unpacked, _ = objective.loss(params, params, {TEXT_KEY: tokens},
                                 jax.random.PRNGKey(1), 0)

    expected, _ = objective.shifted_cross_entropy(
        params, tokens, train=True, rngs={"dropout": jax.random.PRNGKey(1)},
        segment_ids=segment_ids, positions=positions)
    assert float(packed) == pytest.approx(float(expected))
    assert abs(float(packed) - float(unpacked)) > 1e-3, (
        "the packed keys made no difference to the loss")


def test_a_fixed_window_batch_is_scored_exactly_as_before():
    """A batch without the packing keys has to give the loss it gave before
    the packed path existed: the model is called without them, and every
    target counts."""
    objective = make_objective()
    params = objective.init_params(jax.random.PRNGKey(0))
    batch = token_batch()

    loss, _ = objective.loss(params, params, batch, jax.random.PRNGKey(1), 0)

    tokens = np.asarray(batch[TEXT_KEY])
    logits = objective.model.apply(params, jnp.asarray(tokens[:, :-1], jnp.int32))
    assert float(loss) == pytest.approx(
        reference_cross_entropy(logits, tokens[:, 1:]), rel=1e-5)


def test_a_packed_row_of_only_padding_does_not_divide_by_zero():
    objective = packed_objective()
    params = objective.init_params(jax.random.PRNGKey(0))
    tokens = jnp.zeros((2, SEQ + 1), jnp.int32)
    segment_ids = jnp.zeros((2, SEQ + 1), jnp.int32)

    loss, aux = objective.shifted_cross_entropy(
        params, tokens, segment_ids=segment_ids, positions=segment_ids)
    assert float(loss) == 0.0 and bool(jnp.isfinite(aux["perplexity"]))
