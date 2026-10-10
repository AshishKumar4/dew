"""Request identities, bound policy snapshots and filtered draw likelihoods.

Bindings, shape buckets and cache pause/gather are Dew policies; Transformers oracles their tokens.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from model_support import decoder
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from dew.inference import TextGeneration, tasks
from dew.nn.inputs import ModelInputs
from dew.sampling import Sampling, text
from dew.sampling.decoding import StepState, chain
from dew.sampling.strategies import draw
from dew.sampling.text import generate


@pytest.fixture
def task():
    model = decoder()
    variables = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    return TextGeneration(model, variables)


@pytest.fixture
def roomy():
    """The same decoder with a context wide enough to hold a shape bucket."""
    model = decoder().clone(max_seq_len=1024)
    variables = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    return TextGeneration(model, variables, sampling=Sampling(temperature=0))


def ramp(width):
    """A prompt of `width` in-vocabulary ids."""
    return np.tile(np.arange(1, 13, dtype=np.int32), width)[None, :width]


@pytest.mark.parametrize("bad", [
    [[1, 2], "discard me", [3, 4]], [[1.9, 2.8]], [["1", "2"]],
    [[True, False]], [[1, True]], [[1, 2], np.array([True, False])], np.array([[1.5, 2.0]]),
    ModelInputs(jnp.asarray([[1.2, 2.0]])), [[1], [2, 3]],
])
def test_requests_are_rejected_without_dropping_or_coercing_rows(task, bad):
    with pytest.raises(ValueError):
        task(bad, 1, key=jax.random.key(0))


def test_bind_freezes_mapping_structure_but_preserves_array_leaves(task):
    original = jax.tree.map(lambda leaf: leaf, task.variables.unfreeze())
    bound = task.bind(original)
    prompt = [[1, 2]]
    before = bound(prompt, 3, key=jax.random.key(2))
    original["params"].clear()
    original["params"] = jax.tree.map(jnp.zeros_like, task.variables["params"])
    after = bound(prompt, 3, key=jax.random.key(2))
    np.testing.assert_array_equal(after.tokens, before.tokens)
    np.testing.assert_array_equal(after.raw_log_probs, before.raw_log_probs)
    changed = task.bind(original)(prompt, 3, key=jax.random.key(2))
    assert np.max(np.abs(changed.raw_log_probs - before.raw_log_probs)) > 0.01



@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_padless_tokenizer_batches_match_unpadded_rows_without_changing_exports(task, tmp_path, padding_side):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    from dew.interop.pretrained import Processor

    vocabulary = {"<unk>": 0, "one": 1, "two": 2, "three": 3, "<eos>": 4}
    vocabulary.update({f"t{index}": index for index in range(5, task.model.vocab_size)})
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>",
                            padding_side=padding_side).save_pretrained(tmp_path / "source")
    tokenizer = AutoTokenizer.from_pretrained(tmp_path / "source", local_files_only=True)
    processor = Processor(tokenizer, {}, {}, task.model.vocab_size)
    processor.save_pretrained(tmp_path / "before")
    policy = replace(task, processor=processor, sampling=Sampling(temperature=0))
    prompts = ["one", "one two three"]
    inputs = processor(prompts)
    generated = policy(inputs, 2, key=0).host()
    for row, prompt in enumerate(prompts):
        valid = np.asarray(inputs.token_fields["attention_mask"])[row]
        ids = np.asarray(inputs.tokens)[row, valid]
        np.testing.assert_array_equal(ids, tokenizer.encode(prompt))
        alone = policy(prompt, 2, key=0).host()
        np.testing.assert_array_equal(generated.tokens[row, -2:], alone.tokens[0, -2:])
        np.testing.assert_allclose(generated.raw_log_probs[row], alone.raw_log_probs[0], atol=2e-6, rtol=2e-6)
    assert tokenizer.pad_token_id is None and tokenizer.padding_side == padding_side
    assert len(tokenizer) == len(vocabulary)
    processor.save_pretrained(tmp_path / "after")
    before = {path.name: path.read_bytes() for path in (tmp_path / "before").iterdir()}
    after = {path.name: path.read_bytes() for path in (tmp_path / "after").iterdir()}
    assert after == before


def warped(logits, sampling):
    scores = torch.FloatTensor(np.asarray(logits).copy())
    inputs = torch.LongTensor(np.zeros((scores.shape[0], 1), dtype=np.int64))
    scores = TemperatureLogitsWarper(sampling.temperature)(inputs, scores)
    if sampling.top_k is not None:
        scores = TopKLogitsWarper(sampling.top_k)(inputs, scores)
    if sampling.top_p < 1:
        scores = TopPLogitsWarper(sampling.top_p)(inputs, scores)
    if sampling.min_p > 0:
        scores = MinPLogitsWarper(sampling.min_p)(inputs, scores)
    return scores.log_softmax(-1).numpy()


@pytest.mark.parametrize("sampling", [
    Sampling(temperature=0.7, top_p=0.6),
    Sampling(temperature=1.3, min_p=0.4),
    Sampling(temperature=0.8, top_k=9, top_p=0.8, min_p=0.35),
])
def test_cached_generation_records_the_transformers_filtered_distribution(task, sampling):
    """Transformers 5.16.1 filters in temperature/top-k/top-p/min-p order.
    The selected action's raw and filtered log probabilities agree to 3e-6.
    """
    inputs = jnp.array([[1, 2], [4, 5]])
    actual = task(inputs, 4, key=jax.random.key(1), sampling=sampling)
    for step in range(4):
        logits = task.model.apply(task.variables, actual.tokens[:, :2 + step])[:, -1]
        filtered = warped(logits, sampling)
        selected = np.asarray(actual.tokens[:, 2 + step])
        expected = filtered[np.arange(2), selected]
        assert np.all(np.isfinite(expected))
        np.testing.assert_allclose(actual.behavior_log_probs[:, step], expected, atol=3e-6, rtol=0)
        raw = torch.tensor(np.asarray(logits).copy()).log_softmax(-1).numpy()[np.arange(2), selected]
        np.testing.assert_allclose(actual.raw_log_probs[:, step], raw, atol=3e-6, rtol=0)
    repeated = task(inputs, 4, key=jax.random.key(1), sampling=sampling)
    np.testing.assert_array_equal(repeated.tokens, actual.tokens)


def test_every_continuation_records_the_likelihoods_of_its_own_draw(task):
    """One row's likelihoods describe that row's action under that row's
    prefix. Rebuilt from a full forward over each realized sequence, the six
    rows of two prompts agree to 3e-6 on both the filtered and the raw
    distribution, which a row paired with another continuation's prefix or
    another prompt's logits cannot do."""
    sampling = Sampling(temperature=0.8, top_k=9, top_p=0.8)
    inputs = jnp.array([[1, 2], [4, 5]])

    actual = task(inputs, 3, key=jax.random.key(1), sampling=sampling, n=3)

    assert actual.tokens.shape == (6, 5)
    rows = np.arange(6)
    for step in range(3):
        logits = task.model.apply(task.variables, actual.tokens[:, :2 + step])[:, -1]
        selected = np.asarray(actual.tokens[:, 2 + step])
        expected = warped(logits, sampling)[rows, selected]
        assert np.all(np.isfinite(expected))
        np.testing.assert_allclose(actual.behavior_log_probs[:, step], expected, atol=3e-6, rtol=0)
        raw = torch.tensor(np.asarray(logits).copy()).log_softmax(-1).numpy()[rows, selected]
        np.testing.assert_allclose(actual.raw_log_probs[:, step], raw, atol=3e-6, rtol=0)


@pytest.mark.parametrize("sampling", [Sampling(top_p=0), Sampling(min_p=1), Sampling(top_k=1, top_p=0.1)])
def test_filters_keep_the_best_token_at_the_boundary(sampling):
    logits = jnp.array([[1.0, 3.0, -2.0]])
    state = StepState(tokens=jnp.zeros((1, 1), jnp.int32), valid=jnp.ones((1, 1), bool),
                      step=jnp.zeros(1, jnp.int32), active=jnp.ones(1, bool),
                      keys=jax.random.split(jax.random.key(3), 1), prompt_width=1)
    token, behavior, raw = draw(state, logits, chain(sampling.transforms()))
    np.testing.assert_array_equal(token, [1])
    np.testing.assert_allclose(behavior, [0.0], atol=1e-6)
    np.testing.assert_allclose(raw, jax.nn.log_softmax(logits)[0, 1:2], atol=1e-6)


def test_invalid_probability_controls_are_refused():
    for value in (-0.1, 1.1, float("nan"), True):
        with pytest.raises(ValueError):
            Sampling(top_p=value)
        with pytest.raises(ValueError):
            Sampling(min_p=value)


def test_a_source_binds_a_chain_only_for_controls_its_policy_lacks(task):
    """A source's common controls are its `Sampling` value. A control the
    policy does not carry (epsilon sampling) binds the complete chain,
    built in the reference's order with the policy's own transforms in it.
    An explicit policy replaces both, while the row count stays the source's."""
    from pathlib import Path

    from dew.interop.pretrained import PretrainedDecoder
    from dew.sampling import decoding

    source = PretrainedDecoder(task.model, task.variables, None, {}, Path("."), {},
                        generation_config={"do_sample": True, "temperature": 0.7, "top_p": 0.4,
                                           "min_p": 0.1, "num_return_sequences": 2})
    altered = replace(source, generation_config={**source.generation_config,
                                                 "repetition_penalty": 2.0, "typical_p": 0.9})
    bound = altered.text_generation()
    assert bound.logits is None and bound.n == 2
    assert (bound.sampling.repetition_penalty, bound.sampling.typical_p) == (2.0, 0.9)
    rare = replace(altered, generation_config={**altered.generation_config, "epsilon_cutoff": 0.01})
    assert [type(transform) for transform in rare.text_generation().logits] == [
        decoding.RepetitionPenalty, decoding.Temperature, decoding.TopP, decoding.MinP, decoding.Typical,
        decoding.EpsilonCutoff]
    override = rare.text_generation(sampling=Sampling(temperature=0))
    assert override.logits is None and override.sampling.repetition_penalty == 1.0 and override.n == 2
    plain = override([[1, 2]], 6, key=jax.random.key(1), n=1)
    np.testing.assert_array_equal(plain.behavior_log_probs, 0)
    penalized = override([[1, 2]], 6, key=jax.random.key(1), n=1,
                         sampling=Sampling(temperature=0, repetition_penalty=2.0))
    assert not np.array_equal(np.asarray(plain.tokens), np.asarray(penalized.tokens))


def test_one_policy_holds_the_common_controls_in_the_reference_order():
    """Penalties, n-gram bans and the EOS floor run before the warpers, and
    typical filtering after min-p, which is `_get_logits_processor`'s order;
    presence and frequency sit beside the repetition penalty, as vLLM applies
    its penalties together. Every control at its neutral value adds nothing."""
    from dew.sampling import decoding

    policy = Sampling(temperature=0.7, top_k=5, top_p=0.9, min_p=0.05, eos_token_ids=5,
                      repetition_penalty=1.1, presence_penalty=0.5, frequency_penalty=0.25,
                      no_repeat_ngram_size=3, min_new_tokens=2, typical_p=0.8)
    assert [type(transform) for transform in policy.transforms()] == [
        decoding.RepetitionPenalty, decoding.PresencePenalty, decoding.FrequencyPenalty,
        decoding.NoRepeatNGram, decoding.MinNewTokens, decoding.Temperature, decoding.TopK,
        decoding.TopP, decoding.MinP, decoding.Typical]
    assert [type(transform) for transform in replace(policy, temperature=0).transforms()] == [
        decoding.RepetitionPenalty, decoding.PresencePenalty, decoding.FrequencyPenalty,
        decoding.NoRepeatNGram, decoding.MinNewTokens, decoding.Greedy]
    assert Sampling(temperature=0.7).transforms() == (decoding.Temperature(0.7),)
    with pytest.raises(ValueError, match="eos_token_ids"):
        Sampling(min_new_tokens=2).transforms()
    for bad in ({"repetition_penalty": 0.0}, {"no_repeat_ngram_size": -1}, {"min_new_tokens": -1},
                {"typical_p": 1.5}, {"stop": ("",)}, {"stop": "\n\n"}, {"presence_penalty": float("nan")}):
        with pytest.raises(ValueError):
            Sampling(**bad)


def test_a_sources_generation_config_becomes_its_sampling_value(task):
    """The common controls a source declares land in `task.sampling`, so the
    task binds no chain of its own, and a caller changes one control by
    replacing it on that value while the others, EOS included, still apply."""
    from pathlib import Path

    from dew.interop.pretrained import PretrainedDecoder
    from dew.sampling import decoding

    source = PretrainedDecoder(task.model, task.variables, None, {}, Path("."), {}, generation_config={
        "do_sample": True, "temperature": 0.7, "top_k": 5, "eos_token_id": 5, "pad_token_id": 3,
        "repetition_penalty": 1.3, "no_repeat_ngram_size": 2, "min_new_tokens": 2, "typical_p": 0.9})
    bound = source.text_generation()
    assert bound.logits is None
    assert bound.sampling == Sampling(temperature=0.7, top_k=5, eos_token_ids=5, pad_token_id=3,
                                      repetition_penalty=1.3, no_repeat_ngram_size=2, min_new_tokens=2,
                                      typical_p=0.9)
    greedy = bound([[1, 2]], 6, key=1, n=1, sampling=replace(bound.sampling, temperature=0))
    explicit = bound([[1, 2]], 6, key=1, n=1, logits=(
        decoding.RepetitionPenalty(1.3), decoding.NoRepeatNGram(2),
        decoding.MinNewTokens(2, jnp.asarray([5], jnp.int32)), decoding.Greedy()))
    for name in ("tokens", "lengths", "terminated", "behavior_log_probs"):
        np.testing.assert_array_equal(getattr(greedy, name), getattr(explicit, name))


def test_a_call_policy_takes_the_tasks_eos_and_pad_ids(task):
    """EOS and padding are the model's and the tokenizer's, so a call-level
    policy that names neither stops and pads where the task does."""
    first = task([[1, 2]], 4, key=1, sampling=Sampling(temperature=0))
    eos = int(first.tokens[0, 3])
    bound = replace(task, sampling=Sampling(temperature=0, eos_token_ids=eos, pad_token_id=11))

    drawn = bound([[1, 2]], 4, key=1, sampling=Sampling(temperature=0))

    assert drawn.lengths.tolist() == [2] and drawn.terminated.tolist() == [True]
    np.testing.assert_array_equal(drawn.tokens[0, 4:], 11)
    named = bound([[1, 2]], 4, key=1, sampling=Sampling(temperature=0, eos_token_ids=12, pad_token_id=0))
    assert named.lengths.tolist() == [4] and not named.terminated.any()


def test_stop_strings_in_the_policy_end_a_row_as_the_compiled_criterion_does(task, tmp_path):
    """`stop=` compiles against the task's processor, once for the bound
    policy and on the call for a call's own; plain `generate` has no
    vocabulary to compile them against and says so."""
    from tokenizers import Tokenizer, decoders, models
    from transformers import PreTrainedTokenizerFast

    from dew.data import HFTokenizer
    from dew.inference import RunProcessor
    from dew.sampling import decoding

    pieces = ["st", "op", "sto", "pper", "x", "yy", "a", "b", "c", "d", "e", "f"]
    vocabulary = {"<unk>": 0, **{piece: index for index, piece in enumerate(pieces, start=1)}}
    backend = Tokenizer(models.BPE(vocabulary, [], unk_token="<unk>"))
    backend.decoder = decoders.Fuse()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>").save_pretrained(tmp_path)
    tokenizer = HFTokenizer(str(tmp_path))
    order = jnp.asarray([vocabulary["st"], vocabulary["op"], vocabulary["yy"]], jnp.int32)

    def scripted(state, logits):
        pick = jnp.take(order, jnp.clip(state.step, 0, len(order) - 1))
        return jnp.where(jnp.arange(logits.shape[-1])[None, :] == pick[:, None], 0.0, -jnp.inf)

    processed = replace(task, processor=RunProcessor(tokenizer))
    expected = processed([[1, 2]], 4, key=0, logits=(scripted,),
                         stopping=(decoding.stop_strings(tokenizer.tokenizer, "stop", 13),))
    called = processed([[1, 2]], 4, key=0, logits=(scripted,), sampling=Sampling(stop=("stop",)))
    bound = replace(processed, sampling=Sampling(stop=("stop",)))([[1, 2]], 4, key=0, logits=(scripted,))
    assert expected.lengths.tolist() == [2] and expected.terminated.tolist() == [True]
    for drawn in (called, bound):
        np.testing.assert_array_equal(drawn.tokens, expected.tokens)
        assert drawn.lengths.tolist() == [2] and drawn.terminated.tolist() == [True]
    with pytest.raises(ValueError, match="processor"):
        replace(task, sampling=Sampling(stop=("stop",)))
    with pytest.raises(ValueError, match="stop"):
        generate(task.model, task.variables, [[1, 2]], 2, sampling=Sampling(stop=("stop",)))


def test_unsupported_source_controls_report_their_reason(task):
    """An unsupported active source control names itself and the missing behavior."""
    from pathlib import Path

    from dew.interop.pretrained import PretrainedDecoder
    source = PretrainedDecoder(task.model, task.variables, None, {}, Path("."), {}, generation_config={})
    refusals = {
        "stochastic beam": ({"num_beams": 2, "do_sample": True}, "marginal probability"),
        "beams below rows": ({"num_beams": 2, "num_return_sequences": 3}, "exceeds num_beams"),
        "num_beam_groups": ({"num_beams": 2, "num_beam_groups": 2}, "group beam search"),
        "penalty_alpha": ({"do_sample": True, "penalty_alpha": 0.6}, "contrastive search"),
        "unknown": ({"a_future_control": 3}, "does not know this control"),
        "max_time": ({"max_time": 5.0}, "host clock"),
        "token_healing": ({"token_healing": True}, "prompt construction"),
        "guidance_scale": ({"guidance_scale": 2.0}, "second time per step"),
        "dola_layers": ({"dola_layers": "high"}, "DoLa"),
        "watermark": ({"watermarking_config": {"greenlist_ratio": 0.5}}, "watermarking"),
        "constraints": ({"force_words_ids": [[3]]}, "constrained beam search"),
        "ensemble": ({"assistant_ensemble_weight": 0.5}, "biased distribution"),
        "lookup": ({"prompt_lookup_num_tokens": 3}, "prompt lookup"),
        "early exit": ({"assistant_early_exit": 2}, "early-exit"),
        "no depths": ({"use_mtp": True}, "no prediction-depth weights"),
        "other proposer": ({"speculation_type": "ngram"}, "names no native proposer"),
        "both searches": ({"num_beams": 2, "use_mtp": True}, "at once"),
        "cache": ({"use_cache": False}, "its own cache"),
        "quantized cache": ({"cache_config": {"backend": "quanto"}}, "quantized and offloaded"),
        "chunked prefill": ({"prefill_chunk_size": 8}, "one call"),
        "continuous batching": (
            {"continuous_batching_config": {"max_batch_tokens": 8}},
            "continuous batching",
        ),
        "scores": ({"output_scores": True}, "per-step distributions"),
        "hidden states": ({"output_hidden_states": True}, "hidden states"),
        "no compile": ({"disable_compile": True}, "always runs compiled"),
        "capacity": ({"max_cache_len": 4096}, "max_seq_len"),
    }
    for (active, reason) in refusals.values():
        with pytest.raises(ValueError, match=reason):
            replace(source, generation_config=active).text_generation()


def test_a_source_asking_for_several_sequences_binds_them_as_the_task_default(task):
    """num_return_sequences counts returned rows rather than filtering the
    distribution, so the loaded task returns that many per prompt whichever
    strategy runs. An explicit call count overrides it in either direction and
    an explicit sampling policy leaves it alone."""
    from pathlib import Path

    from dew.interop.pretrained import PretrainedDecoder

    source = PretrainedDecoder(task.model, task.variables, None, {}, Path("."), {},
                        generation_config={"do_sample": True, "temperature": 0.9,
                                           "num_return_sequences": 3})
    policy = source.text_generation()
    rows = policy([[1, 2], [3, 4]], 4, key=5).host()
    assert rows.tokens.shape == (6, 6) and rows.lengths.shape == (6,)
    np.testing.assert_array_equal(rows.tokens[:, :2], np.repeat([[1, 2], [3, 4]], 3, axis=0))
    assert policy([[1, 2], [3, 4]], 4, n=1, key=5).host().tokens.shape == (2, 6)
    overridden = source.text_generation(sampling=Sampling(temperature=0))
    assert overridden([[1, 2], [3, 4]], 1, key=5).lengths.shape == (6,)
    searched = replace(source, generation_config={"num_return_sequences": 3, "num_beams": 4})
    beamed = searched.text_generation()
    found = beamed([[1, 2], [3, 4]], 4, key=5).host()
    assert found.tokens.shape == (6, 6)
    np.testing.assert_array_equal(found.tokens[:, :2], np.repeat([[1, 2], [3, 4]], 3, axis=0))
    with pytest.raises(ValueError, match="num_return_sequences"):
        replace(source, generation_config={"num_return_sequences": 0}).text_generation()


def test_source_total_length_and_explicit_continuation_budget_have_defined_precedence(task):
    from pathlib import Path

    from dew.interop.pretrained import PretrainedDecoder

    source = PretrainedDecoder(task.model, task.variables, None, {}, Path("."), {},
                        generation_config={"do_sample": False, "max_length": 5})
    policy = source.text_generation()
    full = policy([[1, 2]], key=1).host()
    assert full.tokens.shape == (1, 5) and full.lengths.tolist() == [3]
    overridden = policy([[1, 2]], 1, key=1).host()
    np.testing.assert_array_equal(overridden.tokens, full.tokens[:, :3])
    with pytest.raises(ValueError, match="prompt width"):
        policy([[1, 2, 3, 4, 5, 6]], key=1)


def test_a_source_forced_eos_follows_the_budget_the_call_asks_for(task):
    """`forced_eos_token_id` accepts several ids and fires one step before the
    end of the request, so a call that changes the budget moves it. Pinning
    the position to the source's own length would force at the wrong step."""
    from pathlib import Path

    from dew.interop.pretrained import PretrainedDecoder

    source = PretrainedDecoder(task.model, task.variables, None, {}, Path("."), {},
                        generation_config={"forced_eos_token_id": [2, 5],
                                           "max_new_tokens": 3, "max_length": 18,
                                           "pad_token_id": 0})
    policy = source.text_generation()
    from dew.sampling import decoding
    plain = replace(policy, logits=(decoding.Greedy(),))
    moved = False
    for budget in (3, 5):
        drawn = policy([[1, 2]], budget, key=0).host()
        assert drawn.tokens.shape == (1, 2 + budget)
        # The last step allows either id and nothing else.
        assert int(drawn.tokens[0, 2 + budget - 1]) in (2, 5)
        free = plain([[1, 2]], budget, key=0).host()
        moved = moved or int(free.tokens[0, -1]) != int(drawn.tokens[0, -1])
    assert moved, "the control changed nothing, so the position it fires at is untested"
    # Without an explicit budget the source's own max_new_tokens decides.
    assert policy([[1, 2]], key=0).host().tokens.shape == (1, 5)


def test_an_explicit_policy_replaces_the_chain_the_source_could_not_build(task):
    """The override replaces the basic policy and the chain, so a control that
    only shapes the distribution is neither built nor judged: a watermark the
    caller just replaced cannot block the call. Everything the task keeps is
    still judged, and an unknown name still refuses because nothing says who
    would own it."""
    from pathlib import Path

    from dew.interop.pretrained import PretrainedDecoder

    blocked = {"watermarking_config": {"greenlist_ratio": 0.5}, "guidance_scale": 2.0,
               "temperature": "warm", "num_return_sequences": 2}
    source = PretrainedDecoder(task.model, task.variables, None, {}, Path("."), {},
                        generation_config=blocked)
    with pytest.raises(ValueError):
        source.text_generation()
    overridden = source.text_generation(sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(overridden([[1, 2]], 2, key=jax.random.key(0), n=1
                                             ).behavior_log_probs, 0)
    # A control the task still owns keeps refusing under the same override.
    for active, reason in (({"max_time": 5.0}, "host clock"),
                           ({"output_scores": True}, "per-step distributions"),
                           ({"a_future_control": 3}, "does not know this control")):
        with pytest.raises(ValueError, match=reason):
            replace(source, generation_config={**blocked, **active}).text_generation(
                sampling=Sampling(temperature=0))
    # A constant proposal length is the fixed block size, so it is not adaptive.
    steady = replace(source, generation_config={"num_assistant_tokens_schedule": "constant"})
    assert steady.text_generation().strategy is None
    with pytest.raises(ValueError, match="constant proposal length"):
        replace(source, generation_config={"num_assistant_tokens_schedule": "heuristic"}
                ).text_generation()


def test_neutral_beam_controls_preserve_the_search(task):
    """Serialized beam defaults remain inert while beam search is active."""
    from pathlib import Path

    from dew.interop.pretrained import PretrainedDecoder

    config = {"num_beams": 2, "num_return_sequences": 2, "eos_token_id": 5}
    source = PretrainedDecoder(task.model, task.variables, None, {}, Path("."), {},
                        generation_config=config)
    expected = source.text_generation()([[1, 2]], 3, key=7)
    declared = replace(source, generation_config={
        **config, "num_beam_groups": 1, "diversity_penalty": 0.0,
        "early_stopping": False, "length_penalty": 1.0})
    actual = declared.text_generation()([[1, 2]], 3, key=7)
    for name in ("tokens", "lengths", "terminated", "raw_log_probs", "behavior_log_probs"):
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@pytest.mark.parametrize("strategy", ["sample", "beam"])
def test_a_bucketed_budget_ends_where_the_request_does(roomy, strategy):
    """A three-token request scans a bucket of four trips and still ends at
    three: ForcedEOS forces EOS at its third token, and beam search ends and
    scores its hypotheses there, as generation at exactly three does."""
    from dew.sampling import decoding
    from dew.sampling.strategies import Beam

    forced = (decoding.Greedy(), decoding.ForcedEOS(jnp.array([5], jnp.int32)))
    search = None if strategy == "sample" else Beam(width=2)
    exact = generate(roomy.model, roomy.variables, ramp(10), 3, key=0, logits=forced, strategy=search)
    drawn = roomy(ramp(10), 3, key=0, logits=forced, strategy=search)
    for field in ("tokens", "lengths", "terminated"):
        np.testing.assert_array_equal(getattr(drawn, field), getattr(exact, field), err_msg=field)
    np.testing.assert_allclose(drawn.raw_log_probs, exact.raw_log_probs, rtol=0, atol=1e-6)
    assert int(np.asarray(drawn.tokens)[0, -1]) == 5


def test_prompts_inside_one_bucket_trace_once_and_draw_what_their_own_width_draws(roomy):
    """A 100-token and a 120-token prompt both pad to 128, so the second call
    reuses the first's executable, and both draw what generation at the exact
    width draws, token for token.

    The likelihoods agree to fp32 rounding, not bit for bit: the padded
    width is a different attention reduction (128 keys against 100 or 120,
    the extra ones masked to zero weight), which a GPU kernel may tile and
    sum in another order (measured on CUDA: 1 ulp). log_softmax rounds in
    the logits' absolute scale, so the bound is 16 fp32 eps of the row's
    largest |logit|: sqrt(128) eps, 11, rounded up. Padding the model did
    attend to, the bug the masking prevents, moves the same log
    probabilities by 0.07 or more, four orders above the bound.
    """
    exact = {width: generate(roomy.model, roomy.variables, ramp(width), 8, key=0,
                             sampling=roomy.sampling) for width in (100, 120)}
    compiled = text._compiled(None)
    traced = compiled._cache_size()
    for width, reference in exact.items():
        drawn = roomy(ramp(width), 8, key=0)
        assert drawn.tokens.shape == (1, width + 8)
        np.testing.assert_array_equal(drawn.tokens, reference.tokens)
        tokens = np.asarray(reference.tokens)
        logits = roomy.model.apply(roomy.variables, jnp.asarray(tokens))
        bound = 16 * np.finfo(np.float32).eps * float(jnp.max(jnp.abs(logits)))
        for field in ("raw_log_probs", "behavior_log_probs"):
            np.testing.assert_allclose(getattr(drawn, field), getattr(reference, field),
                                       rtol=0, atol=bound, err_msg=field)
        attended = jnp.concatenate([jnp.zeros((1, 128 - width), jnp.int32), jnp.asarray(tokens)], axis=1)
        unmasked = roomy.model.apply(roomy.variables, attended)[:, 128 - width:]
        drawn_ids = jnp.asarray(tokens[:, width:, None])
        moved = jnp.max(jnp.abs(
            jnp.take_along_axis(jax.nn.log_softmax(logits[:, width - 1:-1]), drawn_ids, axis=-1)
            - jnp.take_along_axis(jax.nn.log_softmax(unmasked[:, width - 1:-1]), drawn_ids, axis=-1)))
        assert float(moved) > 1e3 * bound, (float(moved), bound)
    assert compiled._cache_size() - traced == 1


def processor_rows(width):
    """`ramp(width)` as a source processor hands it over: whole rows carry the
    token positions and no validity field (`Processor.from_hf`)."""
    tokens = jnp.asarray(ramp(width))
    return ModelInputs(tokens, {"positions": jnp.arange(width, dtype=jnp.int32)[None, :]})


def test_a_processor_prompt_is_bucketed_like_bare_ids(roomy):
    """Every text prompt a loaded source's processor builds carries its token
    positions. One-axis positions say nothing a left-padded row loses: the
    filler is invalid, and the real tokens keep their positions. So two such
    prompts inside one bucket share one executable, and each draws what bare
    ids of its width draw at the exact width: the same tokens, and the same
    likelihoods to the bound the bare-id test above justifies.
    """
    # The references draw bare ids at their exact widths, an executable each, and
    # whether an earlier test in this process already compiled them is no concern
    # of this test's: the count starts after them.
    references = {width: generate(roomy.model, roomy.variables, ramp(width), 8, key=0,
                                  sampling=roomy.sampling) for width in (100, 120)}
    compiled = text._compiled(None)
    traced = compiled._cache_size()
    for width, reference in references.items():
        drawn = roomy(processor_rows(width), 8, key=0)
        assert drawn.tokens.shape == (1, width + 8)
        np.testing.assert_array_equal(drawn.tokens, reference.tokens)
        logits = roomy.model.apply(roomy.variables, jnp.asarray(reference.tokens))
        bound = 16 * np.finfo(np.float32).eps * float(jnp.max(jnp.abs(logits)))
        for field in ("raw_log_probs", "behavior_log_probs"):
            np.testing.assert_allclose(getattr(drawn, field), getattr(reference, field),
                                       rtol=0, atol=bound, err_msg=field)
    assert compiled._cache_size() - traced == 1


def test_the_cache_a_call_builds_holds_the_request_not_the_model_context(roomy, monkeypatch):
    """A 200-token prompt with a 100-token budget runs over 512 cache slots:
    the buckets are 256 and 128, and the model's own 1024 is the ceiling that
    refuses a request, not the capacity every request pays for."""
    seen = {}
    unwrapped = tasks.generate

    def record(model, *args, **kwargs):
        seen["model"] = model
        return unwrapped(model, *args, **kwargs)

    monkeypatch.setattr(tasks, "generate", record)
    roomy(ramp(200), 100, key=0)
    cache = seen["model"].apply(roomy.variables, 1, method="init_cache", mutable=["cache"])[1]["cache"]
    slots = {path[-1].key: leaf.shape[1]
             for path, leaf in jax.tree_util.tree_flatten_with_path(cache)[0] if leaf.ndim > 1}
    assert slots == {"cached_key": 512, "cached_value": 512}


def test_a_request_the_ceiling_refuses_keeps_refusing_at_its_own_shapes(roomy):
    """A prompt and budget over `max_seq_len` cannot be bucketed into one that
    fits, so the request keeps its own shapes and meets the cache ceiling."""
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        roomy(ramp(1000), 100, key=0)


def test_a_one_token_request_scans_one_trip(roomy):
    """The budget ladder starts at one: a scoring probe that wants a single
    token pays for a single decode step, not the smallest serving bucket."""
    shaped, trips, capacity = tasks._bucketed(ModelInputs.from_value(ramp(40)), 1,
                                              roomy.model.max_seq_len)
    assert (shaped.tokens.shape[1], trips, capacity) == (64, 1, 128)


def test_a_budget_inside_a_bucket_returns_the_budget_and_what_the_budget_draws(roomy):
    """A 100-token budget scans the 128-trip bucket and hands back 100 tokens
    per row: the same tokens, lengths and likelihoods the 100-trip scan over
    the same cache produces, and one executable for both budgets."""
    prompt, budget = ramp(40), 100
    shaped, trips, capacity = tasks._bucketed(ModelInputs.from_value(prompt), budget,
                                              roomy.model.max_seq_len)
    assert (shaped.tokens.shape[1], trips, capacity) == (64, 128, 256)
    exact = generate(tasks.cache_sized(roomy.model, capacity), roomy.variables, shaped, budget,
                     key=3, sampling=roomy.sampling)
    compiled = text._compiled(None)
    traced = compiled._cache_size()
    drawn = roomy(prompt, budget, key=3)
    assert drawn.tokens.shape == (1, 40 + budget) and drawn.behavior_log_probs.shape == (1, budget)
    np.testing.assert_array_equal(drawn.tokens[:, -budget:], exact.tokens[:, -budget:])
    np.testing.assert_array_equal(drawn.lengths, exact.lengths)
    np.testing.assert_array_equal(drawn.terminated, exact.terminated)
    np.testing.assert_array_equal(drawn.raw_log_probs, exact.raw_log_probs)
    assert compiled._cache_size() - traced == 1
    roomy(prompt, 128, key=3)
    assert compiled._cache_size() - traced == 1


def stop_at_110(state, token):
    """A criterion no draw inside a 100-token budget can reach."""
    return state.step >= 110


def test_a_criterion_the_budget_never_reaches_leaves_the_row_unterminated(roomy):
    """The 128-trip bucket runs 28 trips past a 100-token budget. A row that
    stops in one of them stopped outside the request: it comes back at the
    budget's length, unterminated, as it does without the bucket."""
    drawn = roomy(ramp(40), 100, key=3, stopping=stop_at_110)
    np.testing.assert_array_equal(drawn.lengths, [100])
    np.testing.assert_array_equal(drawn.terminated, [False])


def test_prefill_scores_the_sampled_position_and_no_other(roomy, monkeypatch):
    """The head runs on the gathered state, not on every prompt position, and
    the two give the same numbers: the row of the full head the decoder would
    have kept, bit for bit, the prediction states untouched, and the same
    greedy continuation as the path that scores every position."""
    prompt = np.concatenate([ramp(6), ramp(6) + 1], axis=0)
    slots = jnp.asarray([3, 5], jnp.int32)
    states, picked = roomy.model.apply(roomy.variables, jnp.asarray(prompt), slots,
                                       method="states_and_logits_at")
    whole_states, every = roomy.model.apply(roomy.variables, jnp.asarray(prompt),
                                            method="states_and_logits")
    assert picked.shape == (2, roomy.model.vocab_size) and every.shape[:2] == prompt.shape
    np.testing.assert_array_equal(picked, every[jnp.arange(2), slots])
    np.testing.assert_array_equal(states, whole_states)
    gathered = roomy(prompt, 8, key=0)
    # Nothing satisfies this stand-in, so the prefill takes the path that
    # scores every prompt position.
    monkeypatch.setattr(text, "Selective", type("NotSelective", (), {}))
    scored_everywhere = roomy(prompt, 8, key=0)
    np.testing.assert_array_equal(gathered.tokens, scored_everywhere.tokens)
    np.testing.assert_array_equal(gathered.raw_log_probs, scored_everywhere.raw_log_probs)
