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
from dew.sampling import Sampling
from dew.sampling.text import generate
from dew.telemetry.instrumentation import Compiles


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

    from dew.interop.processors import Processor

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


@pytest.mark.parametrize(("stated", "cut"), [({}, 50), ({"top_k": None}, 50), ({"top_k": 0}, None),
                                             ({"top_k": 7}, 7)], ids=["absent", "null", "zero", "seven"])
def test_a_sampled_source_cuts_to_the_top_k_transformers_generate_samples_from(task, stated, cut):
    """A sampled source that leaves top_k unset samples from transformers' top 50, as its `generate`
    does; a stated 0 asks for no cut, and any other count is its cut."""
    from pathlib import Path

    from dew.interop.pretrained import PretrainedDecoder

    source = PretrainedDecoder(task.model, task.variables, None, {}, Path("."), {},
                               generation_config={"do_sample": True, **stated})
    assert source.text_generation().sampling.top_k == cut


@pytest.mark.parametrize("changed", [{"temperature": True}, {"top_p": np.True_}, {"top_k": True},
                                     {"repetition_penalty": "1.1"}, {"eos_token_ids": True}])
def test_a_policy_refuses_as_it_is_built_a_flag_or_a_non_number_where_a_setting_goes(changed):
    """A policy checks its settings by the controls' own rules, so a flag where a rate or a count
    goes, or a value that is no number, is refused when the policy is built, not run at a rate
    nobody asked for."""
    valid = {"temperature": 0.7, "top_p": 0.9, "top_k": 5, "repetition_penalty": 1.1, "eos_token_ids": 3}
    Sampling(**valid)
    with pytest.raises(ValueError):
        Sampling(**{**valid, **changed})


def test_a_policy_built_from_numpy_numbers_is_the_policy_of_the_numbers_they_hold():
    """A config's or a sweep's NumPy scalars are ordinary settings: the policy holds the numbers
    they hold, as Python's, so it records and compares as one built from those."""
    given = {"temperature": np.float32(0.7), "top_p": np.float64(0.9), "min_p": np.float32(0.05),
             "top_k": np.int64(5), "repetition_penalty": np.float32(1.1), "presence_penalty": np.int32(1)}
    assert Sampling(**given) == Sampling(**{name: value.item() for name, value in given.items()})


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
    with pytest.raises(ValueError):
        policy([[1, 2, 3, 4, 5, 6]], key=1)


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
    with Compiles("_checked") as compiled:
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
    assert compiled.count == 1


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
    with Compiles("_checked") as compiled:
        for width, reference in references.items():
            drawn = roomy(processor_rows(width), 8, key=0)
            assert drawn.tokens.shape == (1, width + 8)
            np.testing.assert_array_equal(drawn.tokens, reference.tokens)
            logits = roomy.model.apply(roomy.variables, jnp.asarray(reference.tokens))
            bound = 16 * np.finfo(np.float32).eps * float(jnp.max(jnp.abs(logits)))
            for field in ("raw_log_probs", "behavior_log_probs"):
                np.testing.assert_allclose(getattr(drawn, field), getattr(reference, field),
                                           rtol=0, atol=bound, err_msg=field)
    assert compiled.count == 1


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
    with Compiles("_checked") as compiled:
        drawn = roomy(prompt, budget, key=3)
    assert drawn.tokens.shape == (1, 40 + budget) and drawn.behavior_log_probs.shape == (1, budget)
    np.testing.assert_array_equal(drawn.tokens[:, -budget:], exact.tokens[:, -budget:])
    np.testing.assert_array_equal(drawn.lengths, exact.lengths)
    np.testing.assert_array_equal(drawn.terminated, exact.terminated)
    np.testing.assert_array_equal(drawn.raw_log_probs, exact.raw_log_probs)
    assert compiled.count == 1
    with Compiles("_checked") as again:
        roomy(prompt, 128, key=3)
    assert again.count == 0
