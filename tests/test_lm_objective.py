"""Language modelling: the shifted cross entropy, what evaluation produces,
and the objective through the general trainer.

The model here is a small causal stack that honors the backbone's contract
(int32 ids in, float32 logits out, a `train` flag for dropout, the head split
off behind `hidden_states` and `head_weight`) and `dew.sampling.text.generate`
is recorded rather than run, so what is under test is the objective: that the
loss is the cross entropy of the shifted sequence and nothing else, that
padding is excluded only when a pad id is named, that a pass is scored per
token, and that the trainer drives it on both a data-parallel and an FSDP
mesh. The real sampler runs in test_lm_recipe.
"""

from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh
from flax import linen as nn

from dew.artifacts import TextSamples, TokenScores
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective, Perplexity, Samples, TEXT_KEY
from dew.registry import metrics
from dew.training import Checkpoints, Layout, MeshSpec, Trainer

VOCAB = 8
SEQ = 16
BATCH = 8
STEPS = 200
# The test model's parameters are far below the production shard threshold, so
# lower it or "FSDP on" would silently mean "everything replicated".
TINY = 64


class TinyCausalLM(nn.Module):
    """A small causal transformer, standing in for `causal_transformer`.

    Same contract as the real backbone: int32 ids `[B, S]` in, float32 logits
    `[B, S, vocab]` out, `train` gating dropout, no path from a position to a
    later one, and the head split off behind `hidden_states` and
    `head_weight` so the loss can score a vocabulary slice at a time. The
    head carries no bias, which is what makes `hidden @ head_weight` the
    whole projection.
    """

    vocab_size: int
    emb_features: int = 16
    num_layers: int = 2
    max_seq_len: int = 64
    dropout_rate: float = 0.0
    final_logit_softcap: Optional[float] = None
    precision = None

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

    def head_weight(self, params):
        return params["lm_head"]["kernel"].astype(jnp.float32)


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


class Data:
    def __init__(self, train, val=None, batch=BATCH):
        self._train, self.val, self.batch, self.records = train, val, batch, None

    def train(self):
        return self._train()

    steps_per_epoch = None


def reference_cross_entropy(logits, targets, pad_id=None):
    """Shifted cross entropy in numpy, from the model's own logits."""
    logits = np.asarray(logits, np.float64)
    largest = logits.max(axis=-1, keepdims=True)
    log_probs = logits - largest - np.log(np.exp(logits - largest).sum(axis=-1, keepdims=True))
    picked = -np.take_along_axis(log_probs, targets[..., None], axis=-1)[..., 0]
    weights = np.ones_like(picked) if pad_id is None else (targets != pad_id).astype(np.float64)
    return float((picked * weights).sum() / weights.sum())


@pytest.fixture
def recorded_generate(monkeypatch):
    """Stand in for `dew.sampling.text.generate` and record how it was called."""
    calls = []

    def generate(model, params, prompt, max_new_tokens, *, key, temperature=1.0,
                 top_k=None):
        calls.append({"model": model, "params": params, "prompt": prompt,
                      "max_new_tokens": max_new_tokens, "key": key,
                      "temperature": temperature, "top_k": top_k})
        return jnp.concatenate(
            [prompt, jnp.zeros((prompt.shape[0], max_new_tokens), jnp.int32)], axis=1)

    import dew.sampling.text as text_sampler
    monkeypatch.setattr(text_sampler, "generate", generate)
    return calls


# --- the loss --------------------------------------------------------------

def test_loss_is_the_cross_entropy_of_the_shifted_sequence():
    objective = make_objective()
    params = objective.init(jax.random.key(0))
    batch = token_batch()
    tokens = np.asarray(batch[TEXT_KEY])

    loss, _ = objective.loss(params, batch, step_at())

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

    loss, _ = objective.loss(params, batch, step_at())

    logits = objective.model.apply(params, jnp.asarray(tokens[:, :-1], jnp.int32))
    masked = reference_cross_entropy(logits, tokens[:, 1:], pad_id=pad_id)
    unmasked = reference_cross_entropy(logits, tokens[:, 1:])
    assert float(loss) == pytest.approx(masked, rel=1e-5)
    assert abs(masked - unmasked) > 1e-4, "masking made no difference to check"


def test_a_batch_of_only_padding_does_not_divide_by_zero():
    objective = make_objective(pad_id=5)
    params = objective.init(jax.random.key(0))
    batch = {TEXT_KEY: jnp.full((2, SEQ + 1), 5, jnp.int32)}

    loss, aux = objective.loss(params, batch, step_at())
    assert float(loss) == 0.0 and bool(jnp.isfinite(aux.metrics["perplexity"]))


def test_aux_reports_perplexity_and_token_accuracy():
    objective = make_objective()
    params = objective.init(jax.random.key(0))
    batch = token_batch()

    loss, aux = objective.loss(params, batch, step_at())

    assert set(aux.metrics) == {"ce", "perplexity", "token_accuracy"}
    assert aux.variables is None
    assert float(aux.metrics["ce"]) == pytest.approx(float(loss))
    assert float(aux.metrics["perplexity"]) == pytest.approx(float(jnp.exp(loss)), rel=1e-5)

    logits = objective.model.apply(params, batch[TEXT_KEY][:, :-1])
    predicted = np.argmax(np.asarray(logits), axis=-1)
    accuracy = (predicted == np.asarray(batch[TEXT_KEY][:, 1:])).mean()
    assert float(aux.metrics["token_accuracy"]) == pytest.approx(accuracy)


def test_a_batch_with_no_room_for_the_shift_is_rejected():
    objective = make_objective()
    params = objective.init(jax.random.key(0))
    batch = {TEXT_KEY: jnp.zeros((2, SEQ), jnp.int32)}

    with pytest.raises(ValueError, match=f"{SEQ + 1} ids per row"):
        objective.loss(params, batch, step_at())


def test_dropout_runs_on_the_step_key():
    """The loss passes the step key as the dropout rng, so a model with
    dropout is stochastic across keys and reproducible under one."""
    objective = make_objective(model=TinyCausalLM(vocab_size=VOCAB, dropout_rate=0.5))
    params = objective.init(jax.random.key(0))
    batch = token_batch()

    first, _ = objective.loss(params, batch, step_at(key=1))
    again, _ = objective.loss(params, batch, step_at(key=1))
    other, _ = objective.loss(params, batch, step_at(key=2))

    assert float(first) == pytest.approx(float(again))
    assert float(first) != pytest.approx(float(other))


def test_cross_entropy_is_computed_in_float32_under_bfloat16():
    """Params stay fp32 and the loss is fp32 even when the model computes in bf16."""
    class Bf16LM(TinyCausalLM):
        @nn.compact
        def hidden_states(self, tokens, train: bool = False):
            return nn.Embed(self.vocab_size, self.emb_features,
                            dtype=jnp.bfloat16)(tokens)

    objective = make_objective(model=Bf16LM(vocab_size=VOCAB))
    params = objective.init(jax.random.key(0))
    inputs = token_batch()[TEXT_KEY][:, :-1]
    hidden = objective.model.apply(params, inputs, method=Bf16LM.hidden_states)
    loss, aux = objective.loss(params, token_batch(), step_at())

    assert hidden.dtype == jnp.bfloat16
    assert objective.model.apply(params, inputs).dtype == jnp.float32
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(params))
    assert loss.dtype == jnp.float32 and aux.metrics["ce"].dtype == jnp.float32


def test_the_compiled_step_never_builds_a_tokens_by_vocabulary_tensor():
    """The point of chunking, read off the optimized HLO.

    The vocabulary here is 512 wide over 16 tokens, and the head is 32 wide,
    so a `[tokens, vocab]` tensor is unmistakable in the text. The
    full-vocabulary loss below is compiled too: it is what makes this grep a
    test rather than a string that happens not to appear.
    """
    from dew.nn.backbones.causal_transformer import CausalTransformer

    vocab, batch, seq = 512, 2, 8
    model = CausalTransformer(vocab_size=vocab, emb_features=32, num_layers=1,
                              num_heads=4, mlp_features=64, max_seq_len=16,
                              dtype=jnp.bfloat16)
    objective = LMObjective(model, seq, head_chunks=4)
    params = objective.init(jax.random.key(0))
    tokens = jnp.zeros((batch, seq + 1), jnp.int32)
    batch_dict = {TEXT_KEY: tokens}

    def chunked(p):
        loss, aux = objective.loss(p, batch_dict, step_at())
        return loss, aux.metrics

    def full_vocabulary(p):
        logits = model.apply(p, tokens[:, :-1])
        losses = optax.softmax_cross_entropy_with_integer_labels(logits, tokens[:, 1:])
        correct = (jnp.argmax(logits, axis=-1) == tokens[:, 1:]).astype(losses.dtype)
        return jnp.mean(losses), {"token_accuracy": jnp.mean(correct)}

    def compiled_text(loss_fn):
        return jax.jit(jax.value_and_grad(loss_fn, has_aux=True)).lower(
            params).compile().as_text()

    whole_row = (f"f32[{batch * seq},{vocab}]", f"f32[{batch},{seq},{vocab}]")
    chunked_text, full_text = compiled_text(chunked), compiled_text(full_vocabulary)

    assert any(shape in full_text for shape in whole_row), (
        "the full-vocabulary step lost its logits tensor, so this grep proves nothing")
    assert not any(shape in chunked_text for shape in whole_row)
    assert f"f32[{batch * seq},{vocab // 4}]" in chunked_text, "no chunk tile either"

    reduced_over_the_vocabulary = [
        line for line in chunked_text.splitlines()
        if ("reduce" in line and f",{vocab}]" in line)]
    assert reduced_over_the_vocabulary == []


def test_token_accuracy_is_the_argmax_the_full_pass_would_have_taken():
    objective = make_objective()
    params = objective.init(jax.random.key(0))
    batch = token_batch()
    targets = np.asarray(batch[TEXT_KEY][:, 1:])

    _, aux = objective.loss(params, batch, step_at())

    logits = objective.model.apply(params, batch[TEXT_KEY][:, :-1])
    expected = (np.argmax(np.asarray(logits), axis=-1) == targets).mean()
    assert float(aux.metrics["token_accuracy"]) == pytest.approx(expected)


def test_padded_tokens_are_left_out_of_the_accuracy_too():
    objective = make_objective(pad_id=0)
    params = objective.init(jax.random.key(0))
    batch = token_batch(seed=3)
    targets = np.asarray(batch[TEXT_KEY][:, 1:])
    assert (targets == 0).any(), "this batch has no padding to skip"

    _, aux = objective.loss(params, batch, step_at())

    logits = np.asarray(objective.model.apply(params, batch[TEXT_KEY][:, :-1]))
    kept = targets != 0
    expected = ((np.argmax(logits, axis=-1) == targets) & kept).sum() / kept.sum()
    assert float(aux.metrics["token_accuracy"]) == pytest.approx(expected)


# --- what the trainer needs ------------------------------------------------

def test_the_inputs_declare_one_token_row():
    inputs = make_objective().inputs
    assert inputs.sample.key == TEXT_KEY and inputs.sample.shape == (SEQ + 1,)
    assert dict(inputs.conditions) == {}


def test_init_builds_the_tree_from_int32_ids():
    """An init batch of floats is not token ids."""
    objective = make_objective()
    params = objective.init(jax.random.key(0))
    assert set(params) == {"params"}
    assert params["params"]["lm_head"]["kernel"].shape == (16, VOCAB)


def test_the_ema_tracks_the_whole_parameter_tree():
    ema = make_objective(ema_decay=0.995).ema
    assert ema.select(("params", "anything"))
    assert float(ema.decay(0)) == pytest.approx(0.995)
    assert float(ema.decay(10_000)) == pytest.approx(0.995)


def test_pretrained_weights_are_what_init_returns():
    """Continued pretraining: the trainer's whole initial state is the
    checkpoint, not a fresh init that happens to have the same shapes."""
    objective = make_objective()
    trained = objective.init(jax.random.key(0))
    loaded = jax.tree.map(lambda leaf: leaf + 1.0, trained)

    resumed = make_objective(pretrained=loaded).init(jax.random.key(1))

    for restored, expected in zip(jax.tree.leaves(resumed), jax.tree.leaves(loaded)):
        assert jnp.array_equal(restored, expected)


def test_pretrained_params_without_the_variables_dict_are_refused():
    params = make_objective().init(jax.random.key(0))["params"]
    with pytest.raises(ValueError, match="variables dict"):
        make_objective(pretrained=params).init(jax.random.key(0))


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
    scored = jax.jit(lambda params: trainer.objective.loss(params, batch, step_at())[0])
    before = float(scored(trainer.initial_state().params))

    state = trainer.fit(Data(cycle_batches), steps=STEPS, log_every=50)
    after = float(scored(state.params))

    specs = [p.sharding.spec for p in jax.tree.leaves(state.params)]
    if fsdp > 1:
        assert any('fsdp' in str(spec) for spec in specs), "no parameter was sharded"
    assert before > 1.5, "the untrained model already knew the task"
    assert after < 0.3, f"the loss did not fall on the copy task: {before} -> {after}"
    assert Checkpoints(str(tmp_path / "lm")).latest == STEPS


# --- evaluation ------------------------------------------------------------

def test_evaluation_scores_every_target_of_the_batch():
    objective = make_objective()
    params = objective.init(jax.random.key(0))
    batch = token_batch()

    scores = objective.evaluate(params, batch, step_at())

    assert isinstance(scores, TokenScores)
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


def test_evaluation_writes_text_from_the_ema_copy(recorded_generate):
    objective = make_objective(samples=Samples(
        prompt=[1, 2, 3], max_new_tokens=4, temperature=0.0,
        decode=lambda ids: "".join(str(i) for i in ids)))
    params = objective.init(jax.random.key(0))
    ema = jax.tree.map(lambda leaf: leaf + 1, params)

    scores, samples = objective.evaluate(params, token_batch(), step_at(key=5, ema=ema))

    assert isinstance(scores, TokenScores) and isinstance(samples, TextSamples)
    assert samples.tokens.shape == (1, 3 + 4)
    assert samples.prompt == "123" and samples.texts == ("1230000",)
    call, = recorded_generate
    assert call["params"] is ema, "samples were drawn from the live params"
    assert call["model"] is objective.model
    assert np.array_equal(np.asarray(call["prompt"]), [[1, 2, 3]])
    assert call["max_new_tokens"] == 4 and call["temperature"] == 0.0
    assert call["top_k"] is None
    assert jnp.array_equal(jax.random.key_data(call["key"]),
                           jax.random.key_data(jax.random.key(5)))


def test_evaluation_without_an_ema_writes_from_the_live_params(recorded_generate):
    objective = make_objective(samples=Samples(prompt=[[1, 2], [3, 4]], max_new_tokens=2))
    params = objective.init(jax.random.key(0))

    _, samples = objective.evaluate(params, token_batch(), step_at())

    assert samples.tokens.shape == (2, 4)
    assert recorded_generate[0]["params"] is params
    assert recorded_generate[0]["temperature"] == 1.0


def test_prompts_that_cannot_be_batched_are_rejected():
    with pytest.raises(ValueError, match="equal length"):
        make_objective(samples=Samples(prompt=[[1, 2], [3]], max_new_tokens=1))
    with pytest.raises(ValueError, match="non-empty"):
        make_objective(samples=Samples(prompt=[], max_new_tokens=1))


# --- perplexity ------------------------------------------------------------

def test_perplexity_weighs_every_batch_by_its_counted_targets():
    """exp(sum(loss * weight) / sum(weight)) over the pass: a batch with more
    counted targets moves the score more, which the mean of per-batch means
    gets wrong the moment counts differ."""
    metric = metrics.perplexity()
    assert isinstance(metric, Perplexity) and metric.reads is TokenScores
    heavy = TokenScores(losses=jnp.full((1, 4), 1.0), weights=jnp.ones((1, 4)))
    light = TokenScores(losses=jnp.full((1, 4), 3.0), weights=jnp.array([[1.0, 0, 0, 0]]))

    score = metric.reduce([metric(heavy, None), metric(light, None)])

    assert score == pytest.approx(np.exp((4 * 1.0 + 1 * 3.0) / 5))
    assert score != pytest.approx(np.exp(np.mean([1.0, 3.0])))


def test_a_batch_with_no_counted_target_weighs_nothing():
    metric = metrics.perplexity()
    scored = TokenScores(losses=jnp.full((1, 4), 2.0), weights=jnp.ones((1, 4)))
    empty = TokenScores(losses=jnp.zeros((1, 4)), weights=jnp.zeros((1, 4)))

    assert metric.reduce([metric(scored, None), metric(empty, None)]) == pytest.approx(np.exp(2.0))
    with pytest.raises(ValueError, match="no counted target"):
        metric.reduce([metric(empty, None)])


def test_perplexity_is_exp_of_the_mean_cross_entropy_not_the_mean_of_exps():
    metric = metrics.perplexity()
    values = [metric(TokenScores(losses=jnp.full((1, 2), ce), weights=jnp.ones((1, 2))), None)
              for ce in (0.0, 2.0)]
    expected = np.exp(np.mean([0.0, 2.0]))
    wrong = np.mean(np.exp([0.0, 2.0]))
    assert expected != pytest.approx(wrong)
    assert metric.reduce(values) == pytest.approx(expected)


class RecordingTracker:
    def __init__(self):
        self.scalars = []
        self.artifacts = []

    def log(self, scalars, step):
        self.scalars.append((step, dict(scalars)))

    def artifact(self, value, step):
        self.artifacts.append((step, value))


def test_the_validation_pass_scores_perplexity_per_token_and_logs_it():
    """Objective artifacts through the trainer's loop and into the metric:
    two batches of different padding weigh by their counted targets."""
    padded = {TEXT_KEY: jnp.asarray(np.where(np.arange(SEQ + 1) < 9, token_batch(BATCH)[TEXT_KEY], 0))}
    batches = [token_batch(BATCH, seed=1), padded]
    tracker = RecordingTracker()
    trainer = make_trainer(pad_id=0, tracker=tracker)
    trainer.fit(Data(cycle_batches, val=lambda: iter(batches)), steps=1, log_every=1,
                eval_every=1, metrics=(metrics.perplexity(),))

    state = trainer.initial_state()
    total, count = 0.0, 0.0
    for batch in batches:
        scores = trainer.objective.evaluate(state.params, batch, step_at())
        total += float(jnp.sum(scores.losses * scores.weights))
        count += float(jnp.sum(scores.weights))
    logged = [s["val/perplexity"] for _, s in tracker.scalars if "val/perplexity" in s]
    assert len(logged) == 1
    # The initial parameters moved one step before the pass, so the pass
    # scores are close, not equal; what the assertion pins is the weighting.
    assert 0 < count < 2 * BATCH * SEQ
    assert logged[0] == pytest.approx(np.exp(total / count), rel=5e-2)


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

    scored_losses, scored_weights, _, _, _ = objective.token_scores(
        params, tokens, segment_ids=segment_ids, positions=positions)

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
    they get on their own, and the boundary target is dropped on both sides."""
    objective = packed_objective()
    params = objective.init(jax.random.key(0))
    tokens, segment_ids, positions, (doc_a, doc_b) = documents()

    losses, weights, _, _, _ = objective.token_scores(
        params, tokens, segment_ids=segment_ids, positions=positions)
    ce = weighted_mean(losses, weights)

    total, count = 0.0, 0.0
    for document in (doc_a, doc_b):
        alone = only_document(document)
        alone_losses, alone_weights = counted_losses(objective, params, *alone)
        total += float((alone_losses * alone_weights).sum())
        count += float(alone_weights.sum())

    packed_losses, packed_weights = counted_losses(
        objective, params, tokens, segment_ids, positions)
    assert count == packed_weights.sum() == len(doc_a) + len(doc_b) - 2
    assert ce == pytest.approx(total / count, rel=1e-5)
    assert ce == pytest.approx(float((packed_losses * packed_weights).sum() / count), rel=1e-6)


def test_a_packed_batch_scores_like_its_documents_through_the_chunked_head():
    """Both rearrangements at once: the row is packed and the vocabulary is
    scored in chunks. The loss has to be the one the documents get on their
    own through the same chunked head, weighted by the targets each counts."""
    objective = packed_objective()
    params = objective.init(jax.random.key(0))
    tokens, segment_ids, positions, (doc_a, doc_b) = documents()

    losses, weights, _, _, _ = objective.token_scores(
        params, tokens, segment_ids=segment_ids, positions=positions)
    packed = weighted_mean(losses, weights)

    total, count = 0.0, 0
    for document in (doc_a, doc_b):
        alone_tokens, alone_segments, alone_positions = only_document(document)
        alone_losses, alone_weights, _, _, _ = objective.token_scores(
            params, alone_tokens, segment_ids=alone_segments, positions=alone_positions)
        # Every transition inside the document; its last target is padding.
        counted = len(document) - 1
        assert float(jnp.sum(alone_weights)) == counted
        total += weighted_mean(alone_losses, alone_weights) * counted
        count += counted

    assert count == len(doc_a) + len(doc_b) - 2
    # Largest relative difference observed on CPU: 7.1e-08.
    assert packed == pytest.approx(total / count, rel=1e-6)


def test_a_packed_batch_reaches_the_objective_through_the_batch_dict():
    objective = packed_objective()
    params = objective.init(jax.random.key(0))
    tokens, segment_ids, positions, _ = documents()
    batch = {TEXT_KEY: tokens, "text_segment_ids": segment_ids,
             "text_positions": positions}

    packed, _ = objective.loss(params, batch, step_at())
    unpacked, _ = objective.loss(params, {TEXT_KEY: tokens}, step_at())

    losses, weights, _, _, _ = objective.token_scores(
        params, tokens, train=True, rngs={"dropout": jax.random.key(1)},
        segment_ids=segment_ids, positions=positions)
    assert float(packed) == pytest.approx(weighted_mean(losses, weights))
    assert abs(float(packed) - float(unpacked)) > 1e-3, (
        "the packed keys made no difference to the loss")


def test_a_packed_evaluation_carries_the_document_weights():
    """The pass scores what the loss counts: TokenScores of a packed batch
    carry zero weight on the boundary and the padding."""
    objective = packed_objective()
    params = objective.init(jax.random.key(0))
    tokens, segment_ids, positions, (doc_a, doc_b) = documents()
    batch = {TEXT_KEY: tokens, "text_segment_ids": segment_ids, "text_positions": positions}

    scores = objective.evaluate(params, batch, step_at())

    assert float(jnp.sum(scores.weights)) == len(doc_a) + len(doc_b) - 2
    _, weights = counted_losses(objective, params, tokens, segment_ids, positions)
    np.testing.assert_array_equal(np.asarray(scores.weights[0]), weights)


def test_a_fixed_window_batch_is_scored_exactly_as_before():
    """A batch without the packing keys has to give the loss it gave before
    the packed path existed: the model is called without them, and every
    target counts."""
    objective = make_objective()
    params = objective.init(jax.random.key(0))
    batch = token_batch()

    loss, _ = objective.loss(params, batch, step_at())

    tokens = np.asarray(batch[TEXT_KEY])
    logits = objective.model.apply(params, jnp.asarray(tokens[:, :-1], jnp.int32))
    assert float(loss) == pytest.approx(
        reference_cross_entropy(logits, tokens[:, 1:]), rel=1e-5)


def test_a_packed_row_of_only_padding_does_not_divide_by_zero():
    objective = packed_objective()
    params = objective.init(jax.random.key(0))
    tokens = jnp.zeros((2, SEQ + 1), jnp.int32)
    segment_ids = jnp.zeros((2, SEQ + 1), jnp.int32)
    batch = {TEXT_KEY: tokens, "text_segment_ids": segment_ids, "text_positions": segment_ids}

    loss, aux = objective.loss(params, batch, step_at())
    assert float(loss) == 0.0 and bool(jnp.isfinite(aux.metrics["perplexity"]))
    scores = objective.evaluate(params, batch, step_at())
    assert float(jnp.sum(scores.weights)) == 0.0, "an all-padding row must weigh nothing"
