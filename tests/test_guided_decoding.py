"""A guided row only ever spells text its grammar accepts, and ends where it may.

The token automaton is outlines-core's own `Index`, state by state: every
reachable state's allowed tokens and every transition match, over a
vocabulary whose merged pieces cross states. The model is random, so left
alone it spells noise; the remaining checks are that the
grammar, not the model, decides the shape of the text: a JSON schema's
documents parse and validate, a regex's strings match in full, a served
guided request draws what the same request draws alone, and the raw
likelihood stays the model's own.
"""

import json
import re

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

from dew.inference import RunProcessor, TextGeneration
from dew.inference.serving import Server
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.sampling import Sample, Sampling, decoding, guided

pytest.importorskip("outlines_core")

EOS = 256


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory):
    """A byte-level BPE over all 256 bytes, so any text has a spelling."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    alphabet = bytes_to_unicode()
    backend = Tokenizer(models.BPE({alphabet[byte]: byte for byte in range(256)}, [], unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    path = tmp_path_factory.mktemp("bytes")
    PreTrainedTokenizerFast(tokenizer_object=backend).save_pretrained(path)
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


_DEFAULT_TASK_SAMPLING = Sampling(temperature=0, eos_token_ids=EOS)


def task(tokenizer, grammar, sampling=_DEFAULT_TASK_SAMPLING):
    model = CausalTransformer(vocab_size=EOS + 1, emb_features=16, num_layers=1, num_heads=2,
                              head_dim=8, mlp_features=32, max_seq_len=256, dtype="float32")
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    return TextGeneration(model, params, RunProcessor(tokenizer), sampling=sampling,
                          strategy=Sample(grammar))


def continuation(tokenizer, generation):
    rows = generation.host()
    width = rows.tokens.shape[1] - rows.behavior_log_probs.shape[1]
    drawn = rows.tokens[0, width:width + int(rows.lengths[0])]
    return tokenizer.decode([int(token) for token in drawn if token != EOS])


@pytest.fixture(scope="module")
def merged(tmp_path_factory):
    """A byte-level BPE over all 256 bytes with merges into longer pieces and
    an EOS special token, so tokens of different lengths cross the same
    automaton states and one token can spell what two others spell."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    alphabet = bytes_to_unicode()
    vocab = {alphabet[byte]: byte for byte in range(256)}
    space = alphabet[ord(" ")]
    merges = [("a", "b"), ("ab", "c"), ('"', ":"), ("t", "r"), ("tr", "u"), ("tru", "e"), ("1", "2"),
              (":", space), ('"', ",")]
    for left, right in merges:
        vocab[left + right] = len(vocab)
    backend = Tokenizer(models.BPE(vocab, merges, unk_token=None))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    path = tmp_path_factory.mktemp("merged")
    PreTrainedTokenizerFast(tokenizer_object=backend, eos_token="<eos>").save_pretrained(path)
    return AutoTokenizer.from_pretrained(path, local_files_only=True)


def outlines_index(tokenizer, pattern):
    """outlines-core's own `Index` of `pattern`, over the bytes each token
    spells by GPT-2's byte alphabet, which is what a byte-level BPE token
    means; the EOS spells nothing."""
    from outlines_core import Index, Vocabulary
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    byte = {char: value for value, char in bytes_to_unicode().items()}
    spelled: dict[bytes, list[int]] = {}
    for token, index in tokenizer.get_vocab().items():
        if index != tokenizer.eos_token_id:
            spelled.setdefault(bytes(byte[char] for char in token), []).append(index)
    return Index(pattern, Vocabulary(tokenizer.eos_token_id, spelled))


SCHEMA = {"type": "object", "properties": {"name": {"type": "string", "maxLength": 3},
                                           "ok": {"type": "boolean"}, "n": {"type": "integer"}},
          "required": ["name", "ok", "n"]}


@pytest.mark.parametrize("pattern", [r"(abc|ab)+[0-9]{1,2}", r"true|tr[a-z]?u", "schema"])
def test_every_reachable_state_allows_and_leads_where_outlines_index_does(merged, pattern):
    """Walking outlines-core's `Index` breadth first from its initial state,
    Dew's grammar at the matching state masks exactly the tokens `Index`
    disallows (EOS allowed exactly in its final states, the head's padded ids
    nowhere) and every allowed token leads to the state matching the one
    `Index` names; the matching is one to one and reaches every Dew state.
    A second stop id, one of the padded ids, ends a row wherever EOS does.
    The schema compiles through outlines-core's schema-to-regex translation
    on both sides."""
    from outlines_core import json_schema as schemas

    regex = schemas.build_regex_from_schema(json.dumps(SCHEMA), None) if pattern == "schema" else pattern
    index = outlines_index(merged, regex)
    eos, width = merged.eos_token_id, len(merged) + 3
    stops = (eos, len(merged) + 1)
    grammar = (guided.json_schema(merged, SCHEMA, stops, vocab_size=width) if pattern == "schema"
               else guided.regex(merged, pattern, stops, vocab_size=width))
    states = grammar.transitions.shape[0]
    allowed = np.isfinite(np.asarray(grammar.masked(jnp.arange(states), jnp.zeros((states, width)))))
    matching = {index.get_initial_state(): 0}
    queue = [index.get_initial_state()]
    while queue:
        state = queue.pop()
        here = matching[state]
        want = set(index.get_allowed_tokens(state))
        assert (eos in want) == index.is_final_state(state)
        want |= {stops[1]} if eos in want else set()
        assert set(np.flatnonzero(allowed[here]).tolist()) == want, (state, here)
        tokens = jnp.asarray(sorted(want), jnp.int32)
        drawn = jnp.ones(tokens.shape, bool)
        following = np.asarray(grammar.advanced(jnp.full(tokens.shape, here), tokens, drawn))
        for token, there in zip(sorted(want), following.tolist(), strict=True):
            if token in stops:
                assert there == here
                continue
            reached = index.get_next_state(state, token)
            if reached not in matching:
                matching[reached] = there
                queue.append(reached)
            assert matching[reached] == there, (state, token)
    assert sorted(matching.values()) == list(range(states))


def test_a_json_schema_guided_row_writes_a_document_the_schema_accepts(tokenizer):
    """Greedy and sampled rows both end in EOS with a parseable document whose
    fields take the enum values and types the schema names."""
    schema = {"type": "object",
              "properties": {"color": {"enum": ["red", "green"]}, "ok": {"type": "boolean"}},
              "required": ["color", "ok"]}
    grammar = guided.json_schema(tokenizer, schema, EOS, vocab_size=EOS + 1)
    for sampling in (Sampling(temperature=0, eos_token_ids=EOS),
                     Sampling(temperature=1.0, eos_token_ids=EOS)):
        for seed in range(3):
            generation = task(tokenizer, grammar, sampling)("hi", 64, key=seed)
            assert bool(generation.host().terminated[0])
            document = json.loads(continuation(tokenizer, generation))
            assert document["color"] in ("red", "green") and isinstance(document["ok"], bool)


def test_a_served_guided_request_draws_what_it_draws_alone(tokenizer):
    """The server carries each row's grammar state beside its cache, so a
    batch of guided requests draws token for token what each draws alone,
    and every continuation matches the pattern in full."""
    pattern = r"[0-9]{3}-[a-c]{2,4}"
    grammar = guided.regex(tokenizer, pattern, EOS, vocab_size=EOS + 1)
    bound = task(tokenizer, grammar, Sampling(temperature=1.0, eos_token_ids=EOS))
    prompts = ["a", "bcd", "hello there", "x"]
    alone = [bound(prompt, 16, key=index) for index, prompt in enumerate(prompts)]
    server = Server.from_task(bound, slots=2, capacity=64)
    tickets = [server.submit(prompt, 16, key=index) for index, prompt in enumerate(prompts)]
    server.run()
    for ticket, lone in zip(tickets, alone, strict=True):
        served = ticket.result().host()
        np.testing.assert_array_equal(served.tokens, lone.host().tokens)
        np.testing.assert_allclose(served.raw_log_probs, lone.host().raw_log_probs, atol=2e-6)
        assert re.fullmatch(pattern, continuation(tokenizer, served))


def test_the_raw_likelihood_is_the_models_own_and_the_behaviour_one_the_masked(tokenizer):
    """The mask shapes the distribution the row draws from, not the one it
    reports as the model's: forcing a single allowed token makes the
    behaviour likelihood one while the raw one is the model's log-softmax."""
    grammar = guided.regex(tokenizer, "7", EOS, vocab_size=EOS + 1)
    rows = task(tokenizer, grammar, Sampling(temperature=1.0, eos_token_ids=EOS))("q", 4, key=0).host()
    assert continuation(tokenizer, rows) == "7" and bool(rows.terminated[0])
    np.testing.assert_allclose(rows.behavior_log_probs[0, :2], 0.0, atol=1e-6)
    assert np.all(rows.raw_log_probs[0, :2] < 0)


BUILT_INS = {
    "temperature": decoding.Temperature(0.7), "top_k": decoding.TopK(3), "top_p": decoding.TopP(0.8),
    "min_p": decoding.MinP(0.1), "typical": decoding.Typical(0.9), "epsilon": decoding.EpsilonCutoff(0.01),
    "eta": decoding.EtaCutoff(0.01), "top_h": decoding.TopH(0.5), "renormalize": decoding.Renormalize(),
    "repetition": decoding.RepetitionPenalty(1.3), "prompt_repetition": decoding.PromptRepetitionPenalty(1.3),
    "frequency": decoding.FrequencyPenalty(0.5), "presence": decoding.PresencePenalty(0.5),
    "length_decay": decoding.ExponentialDecayLengthPenalty(0, 1.5, jnp.array([0, 1], jnp.int32)),
}


@pytest.mark.parametrize("name", sorted(BUILT_INS))
def test_every_transform_keeps_a_masked_token_masked(name):
    """The grammar masks before the chain runs, so every built-in transform
    sees -inf scores: none may turn one into a NaN or revive it."""
    tokens = jnp.array([[1, 2, 3, 4, 5, 6, 0, 0], [3, 3, 4, 0, 0, 0, 0, 0]])
    state = decoding.StepState(tokens, tokens > 0, jnp.array([2, 1]), jnp.ones(2, bool),
                               jax.random.split(jax.random.key(0), 2), prompt_width=4)
    logits = jax.random.normal(jax.random.key(1), (2, 16)).at[:, ::3].set(-jnp.inf)
    out = BUILT_INS[name](state, logits)
    assert not bool(jnp.any(jnp.isnan(out)))
    assert bool(jnp.all(jnp.isneginf(out[:, ::3])))


def test_a_length_penalty_on_a_masked_eos_leaves_the_row_drawing(tokenizer):
    """ExponentialDecayLengthPenalty raises EOS's score; where the grammar
    forbids EOS the score stays -inf, so the row keeps drawing inside the
    pattern and ends where the pattern completes."""
    grammar = guided.regex(tokenizer, "[0-9]{4}", EOS, vocab_size=EOS + 1)
    bound = task(tokenizer, grammar, Sampling(temperature=1.0, eos_token_ids=EOS))
    rows = bound("q", 8, key=0, logits=(decoding.ExponentialDecayLengthPenalty(
        1, 1.5, jnp.array([EOS], jnp.int32)),)).host()
    assert re.fullmatch("[0-9]{4}", continuation(tokenizer, rows)) and bool(rows.terminated[0])


def test_a_transform_forcing_a_token_the_grammar_forbids_fails_the_request(tokenizer):
    """ForcedEOS draws EOS at the last slot whatever the mask allowed; a
    grammar that cannot end there refuses the draw instead of leaving the
    text outside its language."""
    grammar = guided.regex(tokenizer, "[0-9]{6}", EOS, vocab_size=EOS + 1)
    bound = task(tokenizer, grammar)
    with pytest.raises(checkify.JaxRuntimeError, match="forbids"):
        bound("q", 4, key=0, logits=(decoding.Greedy(), decoding.ForcedEOS(jnp.array([EOS], jnp.int32))))
