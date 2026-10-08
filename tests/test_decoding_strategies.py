"""The decode loop with transforms, criteria and a caller's own components.

These run the whole compiled path, so a component that were only recorded and
never executed, or executed on the host, would not change what comes back.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify
from test_text_rollout_contract import decoder

from dew.sampling import Sample, Sampling, decoding, generate
from dew.sampling.decoding import StepState
from dew.sampling.strategies import draw

VOCAB = 13


def test_terminal_greedy_matches_the_categorical_point_mass_and_float64_likelihood():
    """The final processor selects a point mass, including first-index ties.

    Compare the complete draw to the existing categorical path and score its
    selected model logits with a float64 log partition. Earlier processors
    affect the selection but not the model's raw likelihood.
    """
    values = np.asarray([[1.0, 1.0, -2.0, 0.5], [-5.0, 3.0, 2.0, 1.0], [0.0, -1.0, 2.0, 2.0]], np.float32)
    bias = jnp.asarray([0.0, 0.0, 0.0, 1.5], jnp.float32)

    def shifted(state, scores):
        return scores + bias

    chain = decoding.chain((shifted, decoding.Greedy()))
    for seed in (0, 19):
        state = StepState(jnp.zeros((3, 2), jnp.int32), jnp.zeros((3, 2), bool),
                          jnp.asarray([0, 1, 7], jnp.int32), jnp.ones(3, bool),
                          jax.random.split(jax.random.key(seed), 3), prompt_width=1)
        checked = checkify.checkify(draw)
        error, actual = jax.jit(lambda state, scores, checked=checked: checked(state, scores, chain))(
            state, jnp.asarray(values)
        )
        error.throw()
        # A plain callable retains the categorical point-mass implementation.
        error, categorical = jax.jit(lambda state, scores, checked=checked: checked(
            state, scores, lambda step, logits: chain(step, logits)))(state, jnp.asarray(values))
        error.throw()
        for mine, reference in zip(actual, categorical, strict=True):
            np.testing.assert_array_equal(mine, reference)
        tokens, behavior, raw = map(np.asarray, actual)
        np.testing.assert_array_equal(tokens, np.argmax(values + np.asarray(bias), axis=-1))
        np.testing.assert_array_equal(behavior, 0.0)
        wide = values.astype(np.float64)
        maximum = wide.max(axis=-1)
        partition = maximum + np.log(np.exp(wide - maximum[:, None]).sum(axis=-1))
        expected = wide[np.arange(len(values)), tokens] - partition
        np.testing.assert_allclose(raw, expected, atol=2e-6, rtol=2e-6)


def test_a_transform_after_greedy_can_restore_a_sampled_distribution():
    """Greedy is not a property of a chain with a later arbitrary rewrite."""
    values = jnp.asarray([[1.0, 2.0, -3.0, 4.0]], jnp.float32)
    state = StepState(jnp.zeros((1, 2), jnp.int32), jnp.zeros((1, 2), bool),
                      jnp.zeros(1, jnp.int32), jnp.ones(1, bool),
                      jax.random.split(jax.random.key(0), 1), prompt_width=1)

    def uniform(step, scores):
        return jnp.zeros_like(scores)

    checked = checkify.checkify(draw)
    error, (_, behavior, _) = jax.jit(lambda state: checked(
        state, values, decoding.chain((decoding.Greedy(), uniform))))(state)
    error.throw()
    np.testing.assert_allclose(behavior, -np.log(4.0), atol=2e-6, rtol=2e-6)


def test_an_array_backed_logits_chain_can_be_jitted_directly():
    """A callable chain captures processor arrays as the old closure did."""
    state = StepState(jnp.zeros((2, 2), jnp.int32), jnp.zeros((2, 2), bool),
                      jnp.zeros(2, jnp.int32), jnp.ones(2, bool),
                      jax.random.split(jax.random.key(0), 2), prompt_width=1)
    scores = jnp.asarray([[1.0, 9.0, 5.0, 7.0], [8.0, 2.0, 4.0, 1.0]], jnp.float32)
    chain = decoding.chain((decoding.SuppressTokens(jnp.asarray([1, 3], jnp.int32)), decoding.Greedy()))
    actual = jax.jit(chain)(state, scores)
    expected = np.full((2, 4), -np.inf, np.float32)
    expected[np.arange(2), [2, 0]] = 0.0
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_terminal_greedy_still_refuses_an_undefined_processed_distribution(invalid):
    state = StepState(jnp.zeros((1, 2), jnp.int32), jnp.zeros((1, 2), bool),
                      jnp.zeros(1, jnp.int32), jnp.ones(1, bool),
                      jax.random.split(jax.random.key(0), 1), prompt_width=1)

    def undefined(step, scores):
        return jnp.full_like(scores, invalid)

    checked = checkify.checkify(draw)
    error, _ = jax.jit(lambda state: checked(
        state, jnp.asarray([[1.0, 2.0, 3.0]], jnp.float32),
        decoding.chain((undefined, decoding.Greedy()))))(state)
    with pytest.raises(Exception, match="without a distribution to draw from"):
        error.throw()


@pytest.fixture(scope="module")
def model():
    module = decoder()
    return module, module.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))


def walk(module, params, prompt, steps, transform=None):
    """Greedy continuation without a cache, one full forward per token."""
    sequence = np.asarray(prompt)
    for _ in range(steps):
        logits = np.asarray(module.apply(params, jnp.asarray(sequence))[:, -1])
        if transform is not None:
            logits = transform(sequence, logits)
        sequence = np.concatenate([sequence, logits.argmax(-1)[:, None].astype(np.int32)], axis=1)
    return sequence


def test_defaults_reproduce_the_plain_greedy_walk_when_no_component_is_given(model):
    """Absent components change nothing: the default call, explicit `None`
    components and an explicit `Sample()` all draw the same tokens. An empty
    chain is a different request, because it runs no transform at all."""
    module, params = model
    prompt = jnp.asarray([[1, 2, 3], [4, 5, 6]], jnp.int32)
    policy = Sampling(temperature=0)
    plain = generate(module, params, prompt, 4, key=jax.random.key(1), sampling=policy)
    explicit = generate(module, params, prompt, 4, key=jax.random.key(1), sampling=policy,
                        logits=None, stopping=None, strategy=None)
    chosen = generate(module, params, prompt, 4, key=jax.random.key(1), sampling=policy,
                      strategy=Sample())
    bare = generate(module, params, prompt, 4, key=jax.random.key(1), sampling=policy, logits=())
    assert not np.array_equal(np.asarray(bare.tokens), np.asarray(plain.tokens))
    np.testing.assert_array_equal(np.asarray(plain.tokens), walk(module, params, prompt, 4))
    np.testing.assert_array_equal(np.asarray(explicit.tokens), np.asarray(plain.tokens))
    np.testing.assert_array_equal(np.asarray(chosen.tokens), np.asarray(plain.tokens))
    np.testing.assert_array_equal(np.asarray(plain.behavior_log_probs), 0.0)
    assert plain.lengths.tolist() == [4, 4] and plain.terminated.tolist() == [False, False]


def test_a_plain_callable_transform_runs_inside_the_compiled_loop(model):
    """A user function is not a marker: the same bias, added on the host to a
    full forward walk, has to reproduce the tokens the device loop drew."""
    module, params = model
    prompt = jnp.asarray([[1, 2, 3]], jnp.int32)
    bias = jnp.asarray(np.linspace(4.0, -4.0, VOCAB, dtype=np.float32))

    def lean(state, logits):
        return logits + bias

    drawn = generate(module, params, prompt, 5, key=jax.random.key(0),
                     sampling=Sampling(temperature=0), logits=(lean, decoding.Greedy()))

    expected = walk(module, params, prompt, 5,
                    transform=lambda ids, logits: logits + np.asarray(bias))
    np.testing.assert_array_equal(np.asarray(drawn.tokens), expected)
    assert not np.array_equal(np.asarray(drawn.tokens), walk(module, params, prompt, 5))


def test_a_partial_carries_its_array_configuration_without_recompiling(model):
    """`jax.tree_util.Partial` puts the array in the tree, not in the cache
    key, so two different biases give two different answers."""
    module, params = model
    prompt = jnp.asarray([[1, 2, 3]], jnp.int32)

    def shifted(bias, state, logits):
        return logits + bias

    first = jnp.asarray(np.eye(VOCAB, dtype=np.float32)[2] * 20.0)
    second = jnp.asarray(np.eye(VOCAB, dtype=np.float32)[7] * 20.0)
    def draw(bias):
        return generate(
            module, params, prompt, 3, key=jax.random.key(0), sampling=Sampling(temperature=0),
            logits=(jax.tree_util.Partial(shifted, bias), decoding.Greedy()))
    np.testing.assert_array_equal(np.asarray(draw(first).tokens)[0, 3:], [2, 2, 2])
    np.testing.assert_array_equal(np.asarray(draw(second).tokens)[0, 3:], [7, 7, 7])


def test_an_explicit_chain_runs_in_the_order_it_is_written(model):
    """A chain is complete and ordered. The bias is sized so that adding it
    before the temperature divides picks the runner-up and adding it after
    picks the leader, so the two orders name different tokens and each order
    returns its own."""
    module, params = model
    prompt = jnp.asarray([[1, 2, 3]], jnp.int32)
    logits = np.asarray(module.apply(params, prompt)[:, -1])[0]
    leader, runner_up = np.argsort(logits)[::-1][:2]
    bias = np.zeros(VOCAB, np.float32)
    bias[runner_up] = float(logits[leader] - logits[runner_up]) + 0.01
    before = int(np.argmax((logits + bias) / 0.5))
    after = int(np.argmax(logits / 0.5 + bias))
    assert before == runner_up and after == leader and before != after

    def add(state, values):
        return values + jnp.asarray(bias)

    scaled = decoding.Temperature(0.5)
    first = generate(module, params, prompt, 1, key=jax.random.key(0),
                     logits=(add, scaled, decoding.TopK(1)))
    second = generate(module, params, prompt, 1, key=jax.random.key(0),
                      logits=(scaled, add, decoding.TopK(1)))
    assert int(np.asarray(first.tokens)[0, -1]) == before
    assert int(np.asarray(second.tokens)[0, -1]) == after


def test_a_callers_criterion_ends_the_row_it_names(model):
    """A stopping callable runs after every committed token, and the row that
    matched keeps the token that ended it."""
    module, params = model
    prompt = jnp.asarray([[1, 2, 3], [4, 5, 6]], jnp.int32)

    def after_two(state, tokens):
        return state.step >= 2

    drawn = generate(module, params, prompt, 5, key=jax.random.key(0),
                     sampling=Sampling(temperature=0, pad_token_id=11), stopping=(after_two,))
    assert drawn.lengths.tolist() == [2, 2]
    assert drawn.terminated.tolist() == [True, True]
    np.testing.assert_array_equal(np.asarray(drawn.tokens)[:, 5:], 11)
    np.testing.assert_array_equal(np.asarray(drawn.behavior_log_probs)[:, 2:], 0.0)
    np.testing.assert_array_equal(np.asarray(drawn.raw_log_probs)[:, 2:], 0.0)
    full = generate(module, params, prompt, 5, key=jax.random.key(0), sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(drawn.tokens)[:, :5], np.asarray(full.tokens)[:, :5])


def test_raw_likelihoods_stay_pre_transform_while_behavior_follows_the_chain(model):
    """The RL contract: `raw_log_probs` scores the model's own distribution,
    `behavior_log_probs` the one that drew, and a transform moves only the
    second."""
    module, params = model
    prompt = jnp.asarray([[1, 2, 3]], jnp.int32)
    penalty = decoding.RepetitionPenalty(1.6)
    drawn = generate(module, params, prompt, 4, key=jax.random.key(3),
                     sampling=Sampling(temperature=1.0), logits=(penalty,))
    tokens = np.asarray(drawn.tokens)
    for step in range(4):
        logits = np.asarray(module.apply(params, jnp.asarray(tokens[:, :3 + step]))[:, -1])
        selected = tokens[0, 3 + step]
        raw = jax.nn.log_softmax(jnp.asarray(logits))[0, selected]
        seen = np.unique(tokens[0, :3 + step])
        shaped = logits.copy()
        shaped[0, seen] = np.where(shaped[0, seen] < 0, shaped[0, seen] * 1.6, shaped[0, seen] / 1.6)
        behavior = jax.nn.log_softmax(jnp.asarray(shaped))[0, selected]
        np.testing.assert_allclose(drawn.raw_log_probs[0, step], raw, atol=3e-6, rtol=0)
        np.testing.assert_allclose(drawn.behavior_log_probs[0, step], behavior, atol=3e-6, rtol=0)
    assert not np.allclose(np.asarray(drawn.raw_log_probs), np.asarray(drawn.behavior_log_probs))


def test_an_undefined_distribution_raises_instead_of_returning_a_token(model):
    """Suppressing the whole vocabulary leaves no distribution. The loop has
    to say so rather than hand back index zero."""
    module, params = model
    prompt = jnp.asarray([[1, 2, 3]], jnp.int32)
    everything = decoding.SuppressTokens(jnp.arange(VOCAB, dtype=jnp.int32))
    with pytest.raises(Exception, match="without a distribution"):
        generate(module, params, prompt, 2, key=jax.random.key(0), logits=(everything,))
    # A row that keeps one token is defined, and that token is what it draws.
    kept = decoding.SuppressTokens(jnp.asarray([index for index in range(VOCAB) if index != 5],
                                               jnp.int32))
    drawn = generate(module, params, prompt, 2, key=jax.random.key(0), logits=(kept,))
    np.testing.assert_array_equal(np.asarray(drawn.tokens)[0, 3:], [5, 5])
    np.testing.assert_allclose(np.asarray(drawn.behavior_log_probs), 0.0, atol=1e-6)


def test_min_new_tokens_holds_the_eos_back_and_the_row_still_terminates(model):
    """A suppressed EOS cannot be drawn, so the row runs past the minimum and
    only then terminates, which separates the criterion from the budget."""
    module, params = model
    prompt = jnp.asarray([[1, 2, 3]], jnp.int32)
    eos = int(np.asarray(generate(module, params, prompt, 1, key=jax.random.key(0),
                                  sampling=Sampling(temperature=0)).tokens)[0, -1])
    policy = Sampling(temperature=0, eos_token_ids=eos, pad_token_id=12)
    immediate = generate(module, params, prompt, 5, key=jax.random.key(0), sampling=policy)
    assert immediate.lengths.tolist() == [1] and immediate.terminated.tolist() == [True]
    held = generate(module, params, prompt, 5, key=jax.random.key(0), sampling=policy,
                    logits=(decoding.MinNewTokens(3, jnp.asarray([eos], jnp.int32)),))
    tokens = np.asarray(held.tokens)[0, 3:]
    assert int(held.lengths[0]) >= 3
    assert not np.any(tokens[:3] == eos)


def test_stop_strings_end_a_row_on_text_it_never_decodes_on_the_host(tmp_path, model):
    """The criterion compiles the tokenizer once and then runs on device: the
    row that spells the string stops on the token that completed it, and the
    token and its likelihood are still returned."""
    from tokenizers import Tokenizer, decoders, models
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    module, params = model
    pieces = ["st", "op", "sto", "pper", "x", "yy", "a", "b", "c", "d", "e", "f"]
    vocabulary = {"<unk>": 0}
    vocabulary.update({piece: index for index, piece in enumerate(pieces, start=1)})
    backend = Tokenizer(models.BPE(vocabulary, [], unk_token="<unk>"))
    backend.decoder = decoders.Fuse()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>").save_pretrained(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    criterion = decoding.stop_strings(tokenizer, "stop", VOCAB)

    prompt = jnp.asarray([[1, 2, 3]], jnp.int32)
    # A transform that forces "st" then "op" spells the stop string on the
    # second drawn token, and nothing else in the vocabulary completes it.
    order = [vocabulary["st"], vocabulary["op"], vocabulary["yy"]]

    def scripted(state, logits):
        pick = jnp.take(jnp.asarray(order, jnp.int32), jnp.clip(state.step, 0, len(order) - 1))
        return jnp.where(jnp.arange(VOCAB)[None, :] == pick[:, None], 0.0, -jnp.inf)

    drawn = generate(module, params, prompt, 4, key=jax.random.key(0),
                     sampling=Sampling(pad_token_id=0), logits=(scripted,),
                     stopping=(criterion,))
    assert drawn.lengths.tolist() == [2] and drawn.terminated.tolist() == [True]
    np.testing.assert_array_equal(np.asarray(drawn.tokens)[0, 3:5], order[:2])
    np.testing.assert_allclose(np.asarray(drawn.behavior_log_probs)[0, :2], 0.0, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(drawn.behavior_log_probs)[0, 2:], 0.0)


def test_a_padded_prompt_gives_a_history_transform_the_same_row_as_an_unpadded_one(model):
    """Components read the buffer through the validity mask. A left-padded
    prompt and the same prompt on its own draw the same continuation under
    transforms that read order and membership."""
    from dew.nn.inputs import ModelInputs

    module, params = model
    chain = (decoding.RepetitionPenalty(1.7), decoding.NoRepeatNGram(2),
             decoding.PromptNoRepeatNGram(2))
    padded = ModelInputs(jnp.asarray([[0, 1, 2, 3], [4, 5, 6, 7]], jnp.int32),
                         {"attention_mask": jnp.asarray([[False, True, True, True],
                                                         [True, True, True, True]])})
    together = generate(module, params, padded, 4, key=jax.random.key(0),
                        sampling=Sampling(temperature=0), logits=chain)
    alone = generate(module, params, jnp.asarray([[1, 2, 3]], jnp.int32), 4,
                     key=jax.random.key(0), sampling=Sampling(temperature=0), logits=chain)
    np.testing.assert_array_equal(np.asarray(together.tokens)[0, 4:],
                                  np.asarray(alone.tokens)[0, 3:])
    np.testing.assert_allclose(np.asarray(together.behavior_log_probs)[0],
                               np.asarray(alone.behavior_log_probs)[0], atol=2e-6, rtol=0)
    # The padding slot holds token 0, which the row must not treat as history.
    unpenalized = generate(module, params, padded, 4, key=jax.random.key(0),
                           sampling=Sampling(temperature=0))
    assert not np.array_equal(np.asarray(together.tokens)[0, 4:],
                              np.asarray(unpenalized.tokens)[0, 4:])


@pytest.mark.parametrize("factory,values", [
    (decoding.Temperature, (0.5, 2.0)),
    (decoding.TopP, (0.35, 0.95)),
    (decoding.MinP, (0.1, 0.8)),
])
def test_scalar_values_change_outputs_without_retracing_a_fixed_chain(factory, values):
    from dew.sampling.text import _digest, resolve

    traces = []

    @jax.jit
    def apply(transforms, state, logits):
        traces.append(None)
        return decoding.chain(transforms)(state, logits)

    state = StepState(tokens=jnp.ones((1, 1), jnp.int32), valid=jnp.ones((1, 1), bool),
                      step=jnp.zeros(1, jnp.int32), active=jnp.ones(1, bool),
                      keys=jax.random.split(jax.random.key(0), 1), prompt_width=1)
    logits = jnp.asarray([[-2.0, -1.0, 0.0, 1.0]], jnp.float32)
    first = resolve(Sampling(), (factory(values[0]),), None, None)
    second = resolve(Sampling(), (factory(values[1]),), None, None)
    a = np.asarray(apply(first[0], state, logits))
    b = np.asarray(apply(second[0], state, logits))
    assert traces == [None]
    assert not np.array_equal(a, b)
    assert _digest(first) != _digest(second), "pooled ranks must detect different effective policies"


def test_sampling_defaults_and_explicit_policies_generate_the_same_outputs(model):
    module, params = model
    sampling = Sampling(temperature=0.5, top_k=5, top_p=0.95, min_p=0.2)
    explicit = (decoding.Temperature(0.5), decoding.TopK(5), decoding.TopP(0.95), decoding.MinP(0.2))
    prompt = jnp.asarray([[1, 2, 3]], jnp.int32)
    a = generate(module, params, prompt, 3, key=jax.random.key(0), sampling=sampling)
    b = generate(module, params, prompt, 3, key=jax.random.key(0), sampling=sampling, logits=explicit)
    for field in ("tokens", "lengths", "terminated", "behavior_log_probs", "raw_log_probs"):
        np.testing.assert_array_equal(np.asarray(getattr(a, field)), np.asarray(getattr(b, field)))
