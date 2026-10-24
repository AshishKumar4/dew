"""Cached actions, padding and policy likelihoods on real small decoders.

Each likelihood a generation records, cached and padded, is on par with a
fresh full forward over its row's unpadded tokens from that forward's float64
evaluation over drawn prompts and keys (check 1, tests/reference_error.py).
The raw model's likelihoods are held over every draw. A policy's, which read
its top-k support, and a comparison of two generations' tokens hold over draws
whose steps clear MARGIN at the policy's cutoff, so the float64 forward keeps
the support and the tokens the float32 one kept; the selection at a near tie is
the sampler's, held by tests/test_decoding_strategies.py's planted ties, and the
logits it reads are the raw likelihoods'. What is one computation, or a token,
is held exactly.
"""

import functools
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from model_support import decoder
from reference_error import assert_as_exact, widened
from rl_support import clipped_surrogate, prompt_inputs
from test_mixers import KIND_FIELDS, tiny

from dew.data.dataset import Loader
from dew.inference import TextGeneration
from dew.nn.inputs import ModelInputs
from dew.objectives.base import Step
from dew.objectives.rl import GRPOObjective, RolloutScheduler
from dew.objectives.rl.episodes import EpisodeStatus, Observation
from dew.registry import mixers
from dew.sampling import Sampling, generate
from dew.training import Trainer

ROWS = 16
"""A draw's prompts."""
MARGIN = 1e-4
"""The least gap a draw admits between the scores either side of a policy's cutoff, the top-k's last
and the first outside it, or the two largest under greedy decoding: past it a float64 forward keeps
the tokens the float32 policy kept."""
SEEDS = {'attention cache capacity': 32, 'attention cached scores raw': 8,
         'attention cached scores policy': 16, 'attention perplexity': 16, 'attention positions': 16,
         'attention rescored generation': 8, 'greedy padded generation': 32, 'mla cache capacity': 8,
         'mla cached scores raw': 8, 'mla cached scores policy': 64, 'mla positions': 8,
         'mla rescored generation': 8, 'recurrent cache capacity': 32, 'recurrent cached scores raw': 64,
         'recurrent cached scores policy': 64, 'recurrent perplexity': 64, 'recurrent positions': 64,
         'recurrent rescored generation': 64, 'sampled padded generation': 32, 'trainer update': 32,
         'attention registered scores raw': 16, 'attention registered scores policy': 16,
         'deepseek_v4 registered scores raw': 16, 'deepseek_v4 registered scores policy': 16,
         'gated_delta_net registered scores raw': 32, 'gated_delta_net registered scores policy': 64,
         'kimi_delta_attention registered scores raw': 64,
         'kimi_delta_attention registered scores policy': 64,
         'kpool_sparse_attention registered scores raw': 8,
         'kpool_sparse_attention registered scores policy': 32, 'llama4 registered scores raw': 8,
         'llama4 registered scores policy': 16, 'mamba registered scores raw': 8,
         'mamba registered scores policy': 16, 'mamba2 registered scores raw': 8,
         'mamba2 registered scores policy': 8, 'mla registered scores raw': 16,
         'mla registered scores policy': 16, 'mlp registered scores raw': 8,
         'mlp registered scores policy': 16}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada)."""


def prompts():
    return {"prompt": np.array([[0, 0, 1, 2], [3, 4, 5, 6]], np.int32),
            "prompt_length": np.array([2, 4], np.int32),
            "data_source": np.array([[97], [98]], np.int32),
            "ground_truth": np.array([[49], [50]], np.int32),
            "extra_info": np.zeros((2, 1), np.int32)}


def drawn_prompts(seed: int, attempt: int, vocab: int, *, width: int = 4, shortest: int = 1):
    """ROWS left-padded prompts of `width`, each of a drawn length from `shortest` and drawn tokens."""
    rng = np.random.default_rng([0x7E, seed, attempt])
    lengths = rng.integers(shortest, width + 1, ROWS).astype(np.int32)
    prompt = rng.integers(0, vocab, (ROWS, width)).astype(np.int32)
    prompt[np.arange(width)[None, :] < (width - lengths)[:, None]] = 0
    return prompt, lengths


@functools.cache
def forward(model):
    """`model`'s forward, one compiled program a model (equal clones share it), so every seed, attempt and
    row reuses its compile."""
    return jax.jit(model.apply)


def row_logits(model, params, prompt, lengths, tokens, positions=None):
    """Each row's logits for its drawn `tokens`, from one full forward over every row's unpadded prompt and
    the tokens drawn before each, right-padded to the longest as a learner scores a batch of rollouts, at
    `positions` where given (a row each, the prompt's and the tokens', continued past the row's end): a
    causal model reads no padding at a real position, and the batch's products take a learner's shapes.
    In the model's dtype: an array a row, a step each."""
    width, lengths = prompt.shape[1], np.asarray(lengths)
    contexts = [np.concatenate([prompt[row, width - length:], tokens[row, :-1]])
                for row, length in enumerate(lengths)]
    longest = max(len(context) for context in contexts)
    rows = np.zeros((len(contexts), longest), np.int32)
    for row, context in enumerate(contexts):
        rows[row, :len(context)] = context
    extra = {}
    if positions is not None:
        placed = np.asarray(positions)
        extra["positions"] = jnp.asarray(np.stack([
            np.concatenate([placed[row, :len(context)],
                            placed[row, len(context) - 1] + np.arange(1, longest - len(context) + 1)])
            for row, context in enumerate(contexts)]))
    logits = forward(model)(params, jnp.asarray(rows), **extra)
    return [logits[row, length - 1:length - 1 + tokens.shape[1]] for row, length in enumerate(lengths)]


def policy_scores(model, params, prompt, lengths, tokens, sampling, positions=None) -> dict:
    """Each drawn token's log probability under the raw model (`raw`) and, for a sampled policy, under
    the policy that drew it, its temperature and top-k (`behavior`), from `row_logits`."""
    found = {"raw": []} | ({"behavior": []} if sampling.temperature else {})
    for logits, picked in zip(row_logits(model, params, prompt, lengths, tokens, positions),
                              jnp.asarray(tokens)[..., None], strict=True):
        found["raw"].append(jnp.take_along_axis(jax.nn.log_softmax(logits), picked, -1)[:, 0])
        if sampling.temperature:
            scores = logits / sampling.temperature
            cutoff = jnp.sort(scores, -1)[:, -sampling.top_k][:, None]
            policy = jax.nn.log_softmax(jnp.where(scores >= cutoff, scores, -jnp.inf))
            found["behavior"].append(jnp.take_along_axis(policy, picked, -1)[:, 0])
    return {name: jnp.stack(values) for name, values in found.items()}


def cutoff_gaps(model, params, prompt, lengths, tokens, sampling, positions=None) -> np.ndarray:
    """Each drawn step's gap between the scores either side of its policy's cutoff, `[rows, steps]`."""
    kept = sampling.top_k if sampling.temperature else 1
    gaps = []
    for logits in row_logits(model, params, prompt, lengths, tokens, positions):
        scores = np.sort(np.asarray(logits / (sampling.temperature or 1)), -1)[:, ::-1]
        gaps.append(scores[:, kept - 1] - scores[:, kept])
    return np.stack(gaps)


def clears_margin(model, params, prompt, lengths, tokens, sampling, positions=None) -> bool:
    """Whether every drawn step's scores clear MARGIN at its policy's cutoff."""
    return bool(np.min(cutoff_gaps(model, params, prompt, lengths, tokens, sampling, positions)) > MARGIN)


def generation_draw(model, params, sampling, steps: int, *, margin: bool, **prompt_shape):
    """Check 1's draw of drawn prompts, a key and the tokens `generate` draws from them: with `margin`,
    the first of the seed's attempts whose steps clear MARGIN."""
    def draw(seed):
        for attempt in range(64 if margin else 1):
            prompt, lengths = drawn_prompts(seed, attempt, int(model.vocab_size), **prompt_shape)
            key = jax.random.key_data(jax.random.fold_in(jax.random.key(seed), attempt))
            result = generate(model, params, prompt_inputs({"prompt": prompt, "prompt_length": lengths}),
                              steps, key=jax.random.wrap_key_data(key), sampling=sampling)
            tokens = np.asarray(result.tokens)[:, prompt.shape[1]:]
            if not margin or clears_margin(model, params, prompt, lengths, tokens, sampling):
                return prompt, lengths, np.asarray(key), tokens
        raise AssertionError(f"no draw of seed {seed} clears MARGIN")
    return draw


def only(name: str, run):
    """`run`'s leaf `name` alone."""
    return lambda *draw: {name: run(*draw)[name]}


def generated(model, params, sampling, steps: int):
    """The generation's own recorded likelihoods of a draw, its tokens the draw's, bit for bit."""
    def run(prompt, lengths, key, tokens):
        result = generate(model, params, prompt_inputs({"prompt": prompt, "prompt_length": lengths}), steps,
                          key=jax.random.wrap_key_data(key), sampling=sampling)
        np.testing.assert_array_equal(np.asarray(result.tokens)[:, prompt.shape[1]:], tokens)
        np.testing.assert_array_equal(result.lengths, np.full(len(prompt), steps))
        np.testing.assert_array_equal(result.terminated, np.zeros(len(prompt), bool))
        if not sampling.temperature:
            # A greedy policy's choice is certain.
            np.testing.assert_array_equal(result.behavior_log_probs, 0)
        return {"raw": result.raw_log_probs} | ({"behavior": result.behavior_log_probs}
                                                if sampling.temperature else {})
    return run


def scored(model, params, sampling, dtype=None):
    """`policy_scores` of a draw's tokens, the model computing at `dtype` where given, on its
    parameters widened to it under x64: a draw's reference, and at float64 its truth. The model and its
    parameters are fixed once, so only the draw goes through."""
    if dtype is not None:
        model, params = model.clone(dtype=dtype), widened(params)

    def run(prompt, lengths, key, tokens):
        with jax.enable_x64(new_val=True) if dtype is not None else nullcontext():
            return policy_scores(model, params, prompt, lengths, tokens, sampling)
    return run


def assert_generation_scores_as_full_forwards(model, label: str, steps: int = 3) -> None:
    """Check 1: a left-padded batch drawn through the cache scores each drawn token, under the raw
    model over every draw (`{label} raw`) and under the top-k policy that drew it over draws whose
    steps clear MARGIN (`{label} policy`), on par with a fresh full forward over that row's unpadded
    tokens, from its float64 evaluation, over drawn prompts and keys."""
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    sampling = Sampling(temperature=0.7, top_k=5)
    for leaf, margin in (("raw", False), ("behavior", True)):
        name = f"{label} {'raw' if leaf == 'raw' else 'policy'}"
        assert_as_exact(only(leaf, generated(model, params, sampling, steps)),
                        only(leaf, scored(model, params, sampling)),
                        generation_draw(model, params, sampling, steps, margin=margin), name,
                        seeds=SEEDS[name], truth_fn=only(leaf, scored(model, params, sampling, "float64")))


@pytest.mark.parametrize("kind", ["attention", "mla", "recurrent"])
def test_padding_and_cached_likelihoods_match_unpadded_full_forwards(kind):
    """The cached, left-padded generation's raw and policy likelihoods on par with full forwards
    (`assert_generation_scores_as_full_forwards`); GRPO's rescoring of the padded rows is the
    generation's raw likelihoods, and the top-k policy's differ from the raw model's."""
    model = decoder(kind)
    assert_generation_scores_as_full_forwards(model, f"{kind} cached scores")
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    batch = prompts()
    sampling = Sampling(temperature=0.7, top_k=5)
    result = generate(model, params, prompt_inputs(batch), 3, key=jax.random.key(1), sampling=sampling)
    np.testing.assert_array_equal(result.lengths, [3, 3])
    np.testing.assert_array_equal(result.terminated, [False, False])
    objective = GRPOObjective(model, seq_len=6)

    def rescored(prompt, lengths, key, tokens):
        rows = np.concatenate([prompt, tokens], axis=1)
        padding = prompt.shape[1] - lengths
        return {"raw": objective.per_token_log_probs(params, rows, left_padding=padding)[:, 3:]}

    label = f"{kind} rescored generation"
    assert_as_exact(rescored, only("raw", scored(model, params, sampling)),
                    generation_draw(model, params, sampling, 3, margin=False), label, seeds=SEEDS[label],
                    truth_fn=only("raw", scored(model, params, sampling, "float64")))
    assert not np.array_equal(result.raw_log_probs, result.behavior_log_probs)


@pytest.mark.parametrize("kind", sorted(mixers.paths))
def test_every_registered_mixer_draws_padded_rows_as_their_unpadded_full_forwards_score(kind):
    """Every mixer in the registry, in a decoder of its own: a left-padded batch drawn through the
    cache scores each drawn token, under the raw model and under the top-k policy that drew it, on par
    with a fresh full forward over that row's unpadded tokens
    (`assert_generation_scores_as_full_forwards`)."""
    fields, overrides = KIND_FIELDS.get(kind, ({}, {}))
    model = tiny(mixer=mixers.build(kind, **fields), dtype="float32", **overrides)
    assert_generation_scores_as_full_forwards(model, f"{kind} registered scores")


@pytest.mark.parametrize("kind", ["attention", "mla", "recurrent"])
def test_right_padding_and_logical_positions_survive_cached_generation(kind):
    """Right-padded prompts at their own logical positions: the greedy cached generation picks each
    full forward's largest logit at those positions wherever the two largest are MARGIN apart, and
    its raw likelihoods are on par with that forward's from its float64 evaluation over drawn tokens
    and positions (check 1)."""
    model = decoder(kind)
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    greedy, steps = Sampling(temperature=0), 3

    def placed(seed, attempt):
        rng = np.random.default_rng([0x7F, seed, attempt])
        tokens = np.zeros((ROWS, 4), np.int32)
        tokens[:, :2] = rng.integers(0, int(model.vocab_size), (ROWS, 2))
        start = rng.integers(0, 12, ROWS)
        positions = np.zeros((ROWS, 4), np.int32)
        positions[:, 0], positions[:, 1] = start, start + rng.integers(1, 6, ROWS)
        return tokens, positions

    def inputs(tokens, positions):
        mask = jnp.asarray(np.broadcast_to(np.arange(4)[None, :] < 2, tokens.shape))
        return ModelInputs(jnp.asarray(tokens), {"attention_mask": mask, "positions": jnp.asarray(positions)})

    def walked(positions):
        """The prompt's two positions and each drawn token's after them."""
        return np.concatenate([positions[:, :2], positions[:, 1:2] + np.arange(1, steps)], axis=1)

    def draw(seed):
        tokens, positions = placed(seed, 0)
        result = generate(model, params, inputs(tokens, positions), steps, key=jax.random.key(1),
                          sampling=greedy)
        return tokens, positions, np.asarray(result.tokens)[:, 4:]

    def cached(tokens, positions, drawn):
        result = generate(model, params, inputs(tokens, positions), steps, key=jax.random.key(1),
                          sampling=greedy)
        np.testing.assert_array_equal(np.asarray(result.tokens)[:, 4:], drawn)
        np.testing.assert_array_equal(result.behavior_log_probs, 0)
        return {"raw": result.raw_log_probs}

    def full(dtype=None):
        at, weights = (model, params) if dtype is None else (model.clone(dtype=dtype), widened(params))

        def run(tokens, positions, drawn):
            with jax.enable_x64(new_val=True) if dtype is not None else nullcontext():
                return policy_scores(at, weights, tokens[:, :2], np.full(ROWS, 2), drawn, greedy,
                                     walked(positions))
        return run

    label = f"{kind} positions"
    assert_as_exact(cached, full(), draw, label, seeds=SEEDS[label], truth_fn=full("float64"))
    for seed in range(SEEDS[label]):
        tokens, positions, drawn = draw(seed)
        prompt, lengths = tokens[:, :2], np.full(ROWS, 2)
        clear = cutoff_gaps(model, params, prompt, lengths, drawn, greedy, walked(positions)) > MARGIN
        largest = np.stack([np.argmax(np.asarray(logits), -1) for logits in
                            row_logits(model, params, prompt, lengths, drawn, walked(positions))])
        np.testing.assert_array_equal(drawn[clear], largest[clear])


@pytest.mark.parametrize("sampling,name", [(Sampling(temperature=0), "greedy"),
                                           (Sampling(temperature=0.8, top_k=4), "sampled")])
def test_padding_repro_greedy_and_seeded_reproducibility(sampling, name):
    """A left-padded prompt draws the tokens its unpadded copy draws from the same key, bit for bit,
    and likelihoods on par with the unpadded copy's from the full forward's float64 evaluation over
    drawn prompts and keys (check 1); the same call twice is the same generation."""
    model = decoder()
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    steps = 3

    def unpadded(prompt, key):
        return generate(model, params, prompt[:, 2:], steps, key=jax.random.wrap_key_data(key),
                        sampling=sampling)

    def padded(prompt, key):
        mask = jnp.asarray(np.broadcast_to(np.arange(4)[None, :] >= 2, prompt.shape))
        return generate(model, params, ModelInputs(jnp.asarray(prompt), {"attention_mask": mask}), steps,
                        key=jax.random.wrap_key_data(key), sampling=sampling)

    def draw(seed):
        for attempt in range(64):
            prompt, _ = drawn_prompts(seed, attempt, int(model.vocab_size), shortest=2, width=4)
            prompt[:, :2] = 0
            key = np.asarray(jax.random.key_data(jax.random.fold_in(jax.random.key(seed), attempt)))
            tokens = np.asarray(unpadded(prompt, key).tokens)[:, 2:]
            if clears_margin(model, params, prompt, np.full(ROWS, 2), tokens, sampling):
                return prompt, key, tokens

        raise AssertionError(f"no draw of seed {seed} clears MARGIN")

    def recorded(generation):
        def run(prompt, key, tokens):
            result = generation(prompt, key)
            np.testing.assert_array_equal(np.asarray(result.tokens)[:, -steps:], tokens)
            if not sampling.temperature:
                np.testing.assert_array_equal(result.behavior_log_probs, 0)
            return {"raw": result.raw_log_probs} | ({"behavior": result.behavior_log_probs}
                                                    if sampling.temperature else {})
        return run

    wide = scored(model, params, sampling, "float64")

    def full(prompt, key, tokens):
        return wide(prompt, np.full(ROWS, 2), key, tokens)

    label = f"{name} padded generation"
    assert_as_exact(recorded(padded), recorded(unpadded), draw, label, seeds=SEEDS[label], truth_fn=full)
    prompt, key, _ = draw(0)
    for first, second in zip(jax.tree.leaves(padded(prompt, key)), jax.tree.leaves(padded(prompt, key)),
                             strict=True):
        np.testing.assert_array_equal(first, second)


def test_eos_counts_as_action_and_reward_excludes_eos_and_padding():
    model = decoder()
    objective = GRPOObjective(model, seq_len=7)
    params = objective.init(jax.random.key(0))
    batch = prompts()
    plain = generate(model, params, prompt_inputs(batch), 4, key=jax.random.key(1),
                     sampling=Sampling(temperature=0))
    eos = int(plain.tokens[0, 4])
    seen = []

    def reward(completion, record):
        seen.append((record.data_source, completion, record.ground_truth, record.extra_info))
        return float(len(completion))

    rollout = RolloutScheduler.sampled(objective, reward, groups=2, max_new_tokens=4,
                                       sampling=Sampling(temperature=0, eos_token_ids=eos, pad_token_id=12))
    result = rollout(SimpleNamespace(variables=params, step=0, updates=0), batch, jax.random.key(1))
    assert seen[:2] == [("a", "", "1", "")] * 2
    for index, (_, text, _, _) in enumerate(seen):
        where = result["session_index"] == index
        chain = result["input_ids"][where]
        mask = result["response_mask"][where] != 0
        width = int(batch["prompt_length"][index // 2])
        np.testing.assert_array_equal(chain[:width], batch["prompt"][index // 2, 4 - width:])
        assert not mask[:width].any() and mask[width:].all()
        # The mask counts EOS as an action; the reward text stops before it.
        actions = chain[width:]
        count = len(actions) - int(actions[-1] == eos)
        assert text == " ".join(str(token) for token in actions[:count])
        np.testing.assert_array_equal(result["behavior_log_probs"][where], 0)
        old = result["old_log_probs"][where]
        assert np.all(old[~mask] == 0) and np.all(old[mask] < 0)
    for index in (0, 1):
        actions = result["input_ids"][(result["session_index"] == index) & (result["response_mask"] != 0)]
        np.testing.assert_array_equal(actions, [eos])


@pytest.mark.parametrize("kind", ["attention", "recurrent"])
def test_prompt_perplexity_uses_unpadded_context_and_real_transitions(kind):
    """The evaluation scores a prompt's real transitions only, and each on par with the cross entropy
    of a full forward over that row's unpadded prompt from its float64 evaluation over drawn prompts
    (check 1)."""
    model = decoder(kind)
    objective = GRPOObjective(model, seq_len=7)
    params = objective.init(jax.random.key(0))
    step = Step(jnp.array(0), jax.random.key(1), None)
    scores = objective.evaluate(params, prompts(), step)
    np.testing.assert_array_equal(scores.weights, [[0, 0, 1], [1, 1, 1]])

    def batch_of(prompt, lengths):
        return {"prompt": prompt, "prompt_length": lengths, "data_source": np.zeros((ROWS, 1), np.int32),
                "ground_truth": np.zeros((ROWS, 1), np.int32), "extra_info": np.zeros((ROWS, 1), np.int32)}

    def evaluated(prompt, lengths):
        found = objective.evaluate(params, batch_of(prompt, lengths), step)
        real = np.arange(3)[None, :] >= 4 - lengths[:, None]
        np.testing.assert_array_equal(found.weights, real)
        return jnp.where(real, found.losses, 0)

    def full(model, params):
        apply = forward(model)

        def run(prompt, lengths):
            losses = []
            for row, length in enumerate(np.asarray(lengths)):
                plain = jnp.asarray(prompt[row:row + 1, 4 - length:])
                entropy = optax.softmax_cross_entropy_with_integer_labels(apply(params, plain[:, :-1]),
                                                                          plain[:, 1:])[0]
                losses.append(jnp.pad(entropy, (4 - length, 0)))
            return jnp.stack(losses)
        return run

    wide = full(model.clone(dtype="float64"), widened(params))

    def truth(prompt, lengths):
        with jax.enable_x64(new_val=True):
            return wide(prompt, lengths)

    def draw(seed):
        return drawn_prompts(seed, 0, int(model.vocab_size), shortest=2)

    label = f"{kind} perplexity"
    assert_as_exact(evaluated, full(model, params), draw, label, seeds=SEEDS[label], truth_fn=truth)


def test_real_trainer_update_matches_raw_policy_ratio_with_behavior_recorded():
    """The trainer's one SGD step from a rollout is on par with the step of the raw policy ratio's
    clipped surrogate taken by hand over the same rollout, from its float64 evaluation over drawn
    keys (check 1); the rollout records a top-k behavior apart from the raw policy, and the step
    moves the parameters."""
    model = decoder()
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    objective = GRPOObjective(model, seq_len=7, variables=params)
    rollout = RolloutScheduler.sampled(
        objective, lambda text, record: float(sum(map(int, text.split()))),
        groups=2, max_new_tokens=4, sampling=Sampling(temperature=1.2, top_k=7))
    repeats = max(1, jax.device_count() // 2)
    batch = {name: np.tile(value, (repeats,) + (1,) * (value.ndim - 1))
             for name, value in prompts().items()}
    rows = len(batch["prompt"])

    def rolled_from(key):
        run_key = jax.random.split(key)[1]
        rollout_key = jax.random.fold_in(jax.random.fold_in(run_key, 0), 1)
        return rollout(SimpleNamespace(variables=params, step=0, updates=0), batch, rollout_key)

    def draw(seed):
        """A key and the rollout the trainer's first step takes from it, realized once, in float32."""
        key = np.asarray(jax.random.key_data(jax.random.fold_in(jax.random.key(5), seed)))
        return key, rolled_from(jax.random.wrap_key_data(key))

    def trained(key, rolled):
        data = Loader(train=lambda partition: iter([batch]), val=None, records=rows, batch=rows)
        trainer = Trainer(objective, optax.sgd(0.01), key=jax.random.wrap_key_data(key), rollout=rollout)
        state = trainer.fit(data, steps=1, log_every=1, checkpoint_every=None)
        assert int(state.step) == 1
        return state.variables

    def by_hand(dtype=None):
        if dtype is None:
            return lambda key, rolled: stepped(objective, params, rolled)
        weights = widened(params)
        # Built under x64, as it computes, so nothing it holds narrows to float32.
        with jax.enable_x64(new_val=True):
            wide = GRPOObjective(model.clone(dtype=dtype), seq_len=7, variables=weights)

        def run(key, rolled):
            with jax.enable_x64(new_val=True):
                return stepped(wide, weights, widened(rolled))
        return run

    def stepped(objective, params, rolled):
        def raw_policy_loss(p):
            policy = objective.packed_log_probs(p, rolled)
            return clipped_surrogate(
                policy - jnp.asarray(rolled["old_log_probs"]),
                jnp.asarray(rolled["advantages"]), jnp.asarray(rolled["response_mask"]))[0]

        loss, grads = jax.value_and_grad(raw_policy_loss)(params)
        assert np.isfinite(loss)
        return jax.tree.map(lambda p, g: p - 0.01 * g, params, grads)

    rolled = rolled_from(jax.random.key(5))
    assert np.any(rolled["advantages"] != 0)
    assert not np.array_equal(rolled["old_log_probs"], rolled["behavior_log_probs"])
    assert_as_exact(trained, by_hand(), draw, "trainer update", seeds=SEEDS['trainer update'],
                    truth_fn=by_hand("float64"))
    moved = trained(*draw(0))
    assert any(not np.array_equal(old, new)
               for old, new in zip(jax.tree.leaves(params), jax.tree.leaves(moved), strict=True))


@pytest.mark.parametrize("kind", ["attention", "mla", "recurrent"])
def test_padding_slots_do_not_consume_cache_capacity(kind):
    """A row behind six padding slots, past a five-token cache, draws the tokens its unpadded copy
    draws, bit for bit, and raw likelihoods on par with the unpadded copy's from a full forward's
    float64 evaluation over drawn prompts (check 1)."""
    model = decoder(kind).clone(max_seq_len=5)
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    greedy, steps = Sampling(temperature=0), 3
    mask = jnp.asarray(np.arange(8)[None, :] >= 6)

    def draw(seed):
        for attempt in range(64):
            prompt = np.random.default_rng([0x80, seed, attempt]).integers(
                0, int(model.vocab_size), (ROWS, 2)).astype(np.int32)
            tokens = np.asarray(generate(model, params, prompt, steps, key=jax.random.key(1),
                                         sampling=greedy).tokens)[:, 2:]
            if clears_margin(model, params, prompt, np.full(ROWS, 2), tokens, greedy):
                return prompt, tokens
        raise AssertionError(f"no draw of seed {seed} clears MARGIN")

    def behind_padding(prompt, tokens):
        rows = jnp.asarray(np.pad(prompt, ((0, 0), (6, 0))))
        inputs = ModelInputs(rows, {"attention_mask": jnp.broadcast_to(mask, rows.shape)})
        result = generate(model, params, inputs, steps, key=jax.random.key(1), sampling=greedy)
        np.testing.assert_array_equal(np.asarray(result.tokens)[:, 8:], tokens)
        np.testing.assert_array_equal(result.behavior_log_probs, 0)
        return {"raw": result.raw_log_probs}

    def unpadded(prompt, tokens):
        result = generate(model, params, prompt, steps, key=jax.random.key(1), sampling=greedy)
        return {"raw": result.raw_log_probs}

    wide = scored(model, params, greedy, "float64")

    def full(prompt, tokens):
        return wide(prompt, np.full(ROWS, 2), None, tokens)

    label = f"{kind} cache capacity"
    assert_as_exact(behind_padding, unpadded, draw, label, seeds=SEEDS[label], truth_fn=full)


def test_any_declared_eos_id_stops_the_generation():
    model = decoder()
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    prompt = jnp.array([[1, 2]])
    baseline = generate(model, params, prompt, 3, key=jax.random.key(1),
                        sampling=Sampling(temperature=0))
    first = int(baseline.tokens[0, 2])
    result = generate(model, params, prompt, 3, key=jax.random.key(1),
                      sampling=Sampling(temperature=0, eos_token_ids=((first + 1) % 13, first),
                                        pad_token_id=12))
    np.testing.assert_array_equal(result.tokens[0, 2:], [first, 12, 12])
    np.testing.assert_array_equal(result.lengths, [1])
    np.testing.assert_array_equal(result.terminated, [True])
    # The first step is the same computation under either stopping rule.
    np.testing.assert_array_equal(result.raw_log_probs[0, 0], baseline.raw_log_probs[0, 0])


class Answered:
    """An environment whose episode ends on the first action the policy finishes."""

    def reset(self):
        return Observation((1, 2))

    def step(self, action):
        return Observation((), EpisodeStatus.COMPLETED, " ".join(map(str, action.tokens)))


@contextmanager
def answered(identity):
    yield Answered()


def test_a_rollout_draws_the_same_rows_from_a_legacy_key_as_from_its_typed_one():
    """A rollout takes a single key in either of JAX's spellings, a typed key or a legacy uint32
    pair, which the generation it runs reads alike (`request_key`): sampled prompts and native
    episodes give the same rows from both, where the legacy key was once refused for its shape."""
    model = decoder()
    objective = GRPOObjective(model, seq_len=7)
    params = objective.init(jax.random.key(0))
    sampling = Sampling(temperature=1.2, eos_token_ids=(3, 7), pad_token_id=12)
    sampled = RolloutScheduler.sampled(objective, lambda text, record: float(len(text) % 3),
                                       groups=2, max_new_tokens=4, sampling=sampling)
    state = Trainer(objective, optax.sgd(.01), key=jax.random.key(1)).initial_state()
    episodes = RolloutScheduler.episodes(
        TextGeneration(model, params), answered,
        lambda episode: float(sum(sum(turn.action.tokens) for turn in episode.transitions) % 5),
        max_prompt_tokens=4, max_new_tokens=4, max_turns=1, groups=2, sampling=sampling)
    for rollout, batch in ((sampled, prompts()), (episodes, {"task_id": np.array([31], np.int32)})):
        typed = rollout(state, batch, jax.random.key(3))
        legacy = rollout(state, batch, jax.random.PRNGKey(3))
        assert typed.keys() == legacy.keys()
        for name in typed:
            np.testing.assert_array_equal(np.asarray(typed[name]), np.asarray(legacy[name]))
