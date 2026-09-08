"""The decode loop with transforms, criteria and a caller's own components.

These run the whole compiled path, so a component that were only recorded and
never executed, or executed on the host, would not change what comes back.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.sampling import Sample, Sampling, decoding, generate
from dew.sampling.decoding import StepState
from test_text_rollout_contract import decoder

VOCAB = 13


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
    """Absent components change nothing: the default call, an explicit pair of
    empty tuples and an explicit `Sample()` all draw the same tokens."""
    module, params = model
    prompt = jnp.asarray([[1, 2, 3], [4, 5, 6]], jnp.int32)
    policy = Sampling(temperature=0)
    plain = generate(module, params, prompt, 4, key=jax.random.key(1), sampling=policy)
    explicit = generate(module, params, prompt, 4, key=jax.random.key(1), sampling=policy,
                        logits=(), stopping=(), strategy=None)
    chosen = generate(module, params, prompt, 4, key=jax.random.key(1), sampling=policy,
                      strategy=Sample())
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
                     sampling=Sampling(temperature=0), logits=(lean,))

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
    draw = lambda bias: generate(  # noqa: E731
        module, params, prompt, 3, key=jax.random.key(0), sampling=Sampling(temperature=0),
        logits=(jax.tree_util.Partial(shifted, bias),))
    np.testing.assert_array_equal(np.asarray(draw(first).tokens)[0, 3:], [2, 2, 2])
    np.testing.assert_array_equal(np.asarray(draw(second).tokens)[0, 3:], [7, 7, 7])


def test_the_policy_tail_runs_after_a_callers_transform(model):
    """Order is observable. The bias is sized so that adding it before the
    temperature divides picks the runner-up, and adding it after picks the
    leader, so the two orders name different tokens."""
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

    drawn = generate(module, params, prompt, 1, key=jax.random.key(0),
                     sampling=Sampling(temperature=0.5, top_k=1), logits=(add,))
    assert int(np.asarray(drawn.tokens)[0, -1]) == before


def test_a_callers_criterion_ends_the_row_it_names(model):
    """A stopping callable runs after every committed token, and the row that
    matched keeps the token that ended it."""
    module, params = model
    prompt = jnp.asarray([[1, 2, 3], [4, 5, 6]], jnp.int32)

    def after_two(state, tokens):
        return state.step >= 2

    drawn = generate(module, params, prompt, 5, key=jax.random.key(0),
                     sampling=Sampling(temperature=0, pad_id=11), stopping=(after_two,))
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
    policy = Sampling(temperature=0, eos_id=eos, pad_id=12)
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
    from dew.interop.pretrained import Processor

    module, params = model
    pieces = ["st", "op", "sto", "pper", "las", "x", "yy", "zz", "a", "b", "c", "d"]
    vocabulary = {"<unk>": 0}
    vocabulary.update({piece: index for index, piece in enumerate(pieces, start=1)})
    backend = Tokenizer(models.BPE(vocabulary, [], unk_token="<unk>"))
    backend.decoder = decoders.Fuse()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>").save_pretrained(tmp_path)
    tokenizer = AutoTokenizer.from_pretrained(tmp_path, local_files_only=True)
    processor = Processor(tokenizer, {}, {}, VOCAB)
    criterion = decoding.stop_strings(processor, "stop", VOCAB,
                                      probe=tokenizer.encode("abcd"))

    prompt = jnp.asarray([[1, 2, 3]], jnp.int32)
    # A transform that forces "st" then "op" spells the stop string on the
    # second drawn token, and nothing else in the vocabulary completes it.
    order = [vocabulary["st"], vocabulary["op"], vocabulary["yy"]]

    def scripted(state, logits):
        pick = jnp.take(jnp.asarray(order, jnp.int32), jnp.clip(state.step, 0, len(order) - 1))
        return jnp.where(jnp.arange(VOCAB)[None, :] == pick[:, None], 0.0, -jnp.inf)

    drawn = generate(module, params, prompt, 4, key=jax.random.key(0),
                     sampling=Sampling(temperature=1.0, pad_id=0), logits=(scripted,),
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
