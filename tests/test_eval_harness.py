"""The lm-evaluation-harness adapter over a saved run.

The numbers are checked against lm-eval's own `HFLM` running the same
weights in transformers: the tiny Llama fixture, loaded into Dew and into
torch, read through one byte-level vocabulary on both sides. Every step a
harness adapter can get wrong on its own (the context and continuation
split, the rolling windows, which logits slot reads which target) is then
compared with the harness's own. The end-to-end run of a real task needs
the task's dataset from the Hub, so it carries the `network` marker; lm_eval
0.4 ships no task whose data is in the package.
"""

from pathlib import Path

import pytest
from test_inference import make_lm_run

pytest.importorskip("lm_eval")

from lm_eval.api.instance import Instance

from dew.eval.harness import DewLM
from dew.inference import TextGeneration

LLAMA = Path(__file__).parent / "fixtures" / "hf" / "llama-tiny"


def instance(*arguments, request_type="loglikelihood"):
    return Instance(request_type=request_type, doc={}, arguments=arguments, idx=0)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    directory = tmp_path_factory.mktemp("lm-run")
    make_lm_run(directory)
    return directory


@pytest.fixture(scope="module")
def adapter(run):
    return DewLM(TextGeneration.from_run(str(run)), batch_size=2)


BOS = 1
"""The byte `\\x01` doubles as BOS in the BOS-vocabulary case; only the text that starts with the
prefix token on purpose holds it."""


def _byte_tokenizer(bos: str):
    """Dew's `ByteTokenizer` as a transformers tokenizer: one id per utf-8 byte,
    id = byte value, EOS 255, which is what `HFLM` needs to be handed. `bos`
    is "none"; "inserted", where every encoding starts with BOS, as Llama,
    Mistral and Gemma vocabularies do, and `HFLM` conditions first tokens on
    it; "declared", a BOS the tokenizer names but never inserts, as
    GPT-2's; or "closing", no BOS and an EOS every encoding ends with, as
    T5's."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    symbols = bytes_to_unicode()
    vocabulary = {symbols[byte]: byte for byte in range(256)}
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    if bos == "none":
        return PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token=symbols[255])
    if bos == "closing":
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"$A {symbols[255]}", special_tokens=[(symbols[255], 255)])
        return PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token=symbols[255])
    if bos == "inserted":
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{symbols[BOS]} $A", special_tokens=[(symbols[BOS], BOS)])
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token=symbols[255],
                                   bos_token=symbols[BOS])


def _pair(max_length: int, directory=None, *, bos: str = "none", source: bool = False):
    """The same weights behind `DewLM` and behind lm-eval's `HFLM`, both
    scoring windows of `max_length` ids and reading the `_byte_tokenizer(bos)`
    vocabulary. Dew reads it as a run does, through a `RunProcessor` over
    `ByteTokenizer` or, with a BOS, over the files saved in `directory`; or,
    with `source`, as a loaded checkpoint does, through the tokenizer's own
    `dew.interop.Processor`."""
    import torch
    from lm_eval.models.huggingface import HFLM
    from transformers import LlamaForCausalLM

    from dew.data.text import ByteTokenizer, HFTokenizer
    from dew.inference.loading import RunProcessor
    from dew.interop.pretrained import Pretrained
    from dew.interop.processors import Processor
    from dew.sampling import Sampling

    reference = _byte_tokenizer(bos)
    if source:
        processor = Processor(reference, {}, {}, 256)
    elif bos == "none":
        processor = RunProcessor(ByteTokenizer())
    else:
        reference.save_pretrained(str(directory))
        processor = RunProcessor(HFTokenizer(str(directory), local_files_only=True))
    loaded = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference",
                             max_seq_len=max_length)
    ours = DewLM(TextGeneration(loaded.model, loaded.variables, processor,
                                sampling=Sampling(eos_token_ids=255)), batch_size=2)
    model = LlamaForCausalLM.from_pretrained(LLAMA, dtype=torch.float32).eval()
    theirs = HFLM(pretrained=model, tokenizer=reference, max_length=max_length, batch_size=2)
    return ours, theirs


# Both sides run fp32 over the same weights, torch against XLA; summed
# log-probabilities agree to 2.6e-7 relative at worst (-89.84966 against
# -89.84964 for ("", "empty context"), which fits the window), so 1e-6
# relative is the float32 bound.
RELATIVE = 1e-6


def _text(length: int) -> str:
    return "".join(chr(ord("a") + (7 * step) % 26) for step in range(length))


def test_loglikelihood_equals_lm_evals_own_model_on_the_same_weights():
    """Contexts with trailing whitespace, where lm-eval moves the spaces into
    the continuation; an empty context, conditioned on the prefix token; a
    pair longer than the window, truncated from the left; one batch holding
    rows of different lengths."""
    ours, theirs = _pair(16)
    requests = [instance(*pair) for pair in (
        ("the ", "quick"), ("a  ", "b"), ("hello", " world"), ("", "empty context"),
        ("a context much longer than sixteen bytes, ", "then the end"))]
    expected = theirs.loglikelihood(requests)
    got = ours.loglikelihood(requests)
    for (score, greedy), (reference, reference_greedy) in zip(got, expected, strict=True):
        assert score == pytest.approx(reference, rel=RELATIVE)
        assert greedy == reference_greedy


def test_a_bos_vocabulary_conditions_first_tokens_on_bos_as_lm_eval_does(tmp_path):
    """`HFLM.prefix_token_id` is the tokenizer's BOS when it has one. An empty
    context and a rolling window's first token are conditioned on it, so a
    BOS vocabulary is scored after BOS, not after EOS."""
    ours, theirs = _pair(16, tmp_path, bos="inserted")
    assert ours.prefix_token_id == theirs.prefix_token_id == BOS
    requests = [instance("", "empty context"), instance("the ", "quick")]
    for (score, greedy), (reference, reference_greedy) in zip(
            ours.loglikelihood(requests), theirs.loglikelihood(requests), strict=True):
        assert score == pytest.approx(reference, rel=RELATIVE)
        assert greedy == reference_greedy
    text = [instance(_text(40), request_type="loglikelihood_rolling")]
    assert ours.loglikelihood_rolling(text) == pytest.approx(theirs.loglikelihood_rolling(text),
                                                             rel=RELATIVE)


@pytest.mark.parametrize("source", [False, True])
@pytest.mark.parametrize("bos", ["none", "inserted", "declared"])
def test_tokenization_controls_encode_as_lm_evals_own_model_does(bos, source, tmp_path):
    """`add_special_tokens` and `left_truncate_len` encode as `HFLM.tok_encode`
    does, through a run's processor and a loaded checkpoint's, over a
    vocabulary without BOS, one that inserts it and one that names it
    without inserting it: the inserted BOS left off when the caller asks for
    no special tokens, or when the text already starts with the prefix
    token's decoded text, and the last ids kept. With a BOS, a text that starts
    with the BOS's own symbol, which the tokenizer reads as the BOS while its
    decoded text is another byte, encodes as `HFLM`'s too."""
    ours, theirs = _pair(16, tmp_path, bos=bos, source=source)
    prefix = theirs.tokenizer.decode([theirs.prefix_token_id])
    texts = ["the quick", prefix + "the quick"]
    if bos != "none":
        texts.append(theirs.tokenizer.convert_ids_to_tokens(theirs.prefix_token_id) + "the quick")
    for text in texts:
        for special in (None, True, False):
            for keep in (None, 3):
                assert (ours.tok_encode(text, add_special_tokens=special, left_truncate_len=keep)
                        == theirs.tok_encode(text, add_special_tokens=special, left_truncate_len=keep))


@pytest.mark.parametrize("source", [False, True])
def test_a_vocabulary_that_closes_a_sequence_scores_no_eos_it_never_wrote(source, tmp_path):
    """A tokenizer that ends every encoding with an EOS, as T5's does: a scored context and its
    continuation split at the context's end, the continuation its own ids and no EOS in either, since
    a pair encodes as one row (`tok_encode`'s default opens a text and closes it with nothing). Asked
    for its special tokens, the encoding is the tokenizer's own, its EOS after the text; asked for
    none, the text's ids alone. (`HFLM`'s default keeps the EOS, so its context ends in an EOS and its
    continuation starts one id late.)"""
    ours, _ = _pair(16, tmp_path, bos="closing", source=source)
    bare = list(b" quick")
    context, continuation = ours._encode_pair("the ", "quick")
    assert continuation == bare and context == list(b"the")
    assert ours.tok_encode("the", add_special_tokens=True) == [*b"the", 255]
    assert ours.tok_encode("the", add_special_tokens=False) == list(b"the")


def test_a_task_that_names_no_eos_has_no_end_of_text_id_to_condition_on():
    """The id an empty context is conditioned on is the task's EOS; a task whose sampling names none
    is refused, rather than conditioned on id 0, some vocabulary's ordinary token."""
    from dew.data.text import ByteTokenizer
    from dew.inference.loading import RunProcessor
    from dew.interop.pretrained import Pretrained
    from dew.sampling import Sampling

    loaded = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference", max_seq_len=16)
    adapter = DewLM(TextGeneration(loaded.model, loaded.variables, RunProcessor(ByteTokenizer()),
                                   sampling=Sampling()), batch_size=1)
    with pytest.raises(ValueError):
        _ = adapter.eot_token_id


def test_options_the_adapter_cannot_honour_are_refused(run, adapter):
    """A torch `--device`, and options `HFLM` does not read, raise instead of being dropped."""
    with pytest.raises(ValueError):
        DewLM.create_from_arg_string(f"run={run}", {"device": "cuda:0"})
    assert DewLM.create_from_arg_string(f"run={run},batch_size=2", {"device": None}).batch_size == 2
    with pytest.raises(TypeError):
        adapter.tok_encode("the", truncation=True)
    with pytest.raises(TypeError):
        adapter._loglikelihood_tokens([(None, [1], [2])], override_bs=1)


@pytest.mark.mesh(devices=2)
def test_a_harness_loads_a_run_sharded_and_stored_narrower(run):
    """A harness loads a run over a mesh, by a layout, and stores its weights in the dtype `param_dtype`
    names, as `TextGeneration.from_run` does (row 2244): `--model_args` names the mesh by its sizes,
    `fsdp=2,param_dtype=bfloat16` placing bfloat16 weights on a mesh of two fsdp shards that scores a
    request, and `DewLM.from_run`'s layout splits the largest of them over those shards."""
    import math

    import jax
    import jax.numpy as jnp

    from dew.training import Layout, MeshSpec

    narrow = DewLM.create_from_arg_string(f"run={run},fsdp=2,param_dtype=bfloat16")
    leaves = jax.tree.leaves(narrow.task.variables["params"])
    assert {leaf.dtype for leaf in leaves} == {jnp.dtype(jnp.bfloat16)}
    assert {leaf.sharding.mesh.shape["fsdp"] for leaf in leaves} == {2}
    ((score, _),) = narrow.loglikelihood([instance("the ", "quick")])
    assert math.isfinite(score)
    split = DewLM.from_run(str(run), mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=1))
    largest = max(jax.tree.leaves(split.task.variables["params"]), key=lambda leaf: leaf.size)
    assert not largest.sharding.is_fully_replicated


def test_generation_controls_shape_the_draws_as_they_shape_lm_evals_own_model():
    """On the same weights, a request that samples under `top_k=1` draws what
    lm-eval's own model draws there and what greedy decoding draws; a total
    `max_length`, a request's own EOS and an explicit zero temperature return
    lm-eval's own answers. A beam search, which lm-eval's model stops by its
    leading candidate for every beam together, a control `generate` would
    not use, and sampling at temperature 0, which `generate` refuses, raise
    instead of drawing."""
    ours, theirs = _pair(64)
    # Only printable ids are drawn, so no answer is cut at the EOS's replacement-character text.
    greedy = {"until": ["\n\n"], "do_sample": False, "max_gen_toks": 8,
              "suppress_tokens": [*range(32), *range(127, 256)]}
    sampled = {**greedy, "do_sample": True, "temperature": 1.0, "top_k": 1}

    def asked(controls):
        return [instance("the quick ", controls, request_type="generate_until")]

    context = len(ours.tok_encode("the quick "))
    answer = ours.generate_until(asked(sampled))
    assert answer == theirs.generate_until(asked(sampled))
    assert answer == ours.generate_until(asked(greedy))
    # A request's own EOS, the greedy answer's second id, ends that answer early.
    assert len(answer[0]) > 2
    eos = ours.tok_encode(answer[0][1], add_special_tokens=False)[0]
    ended = {**greedy, "eos_token_id": eos}
    assert ours.generate_until(asked(ended)) != answer
    for controls in ({**greedy, "max_length": context + 2}, ended, {**greedy, "temperature": 0.0}):
        assert ours.generate_until(asked(controls)) == theirs.generate_until(asked(controls)), controls
    for refused in ({**greedy, "num_beams": 3, "until": []}, {**greedy, "not_a_control": 1},
                    {**greedy, "do_sample": True}, {**greedy, "do_sample": True, "temperature": 0.0}):
        with pytest.raises(ValueError):
            ours.generate_until(asked(refused))


@pytest.mark.parametrize("length, max_length", [(3, 16), (50, 16), (5000, 2048)])
def test_rolling_likelihood_scores_every_token_once_as_lm_eval_does(length, max_length):
    """lm-eval's rolling windows score all `length` tokens, the first one
    conditioned on the prefix token: 5000 of 5000 at `max_length` 2048,
    where cutting the string into consecutive rows scored 4997. A
    dropped token moves the sum by its whole log-probability, about 5.5
    nats for this vocabulary, so equal sums mean the same tokens scored."""
    text = _text(length)
    ours, theirs = _pair(max_length)
    (reference,) = theirs.loglikelihood_rolling([instance(text, request_type="loglikelihood_rolling")])
    (score,) = ours.loglikelihood_rolling([instance(text, request_type="loglikelihood_rolling")])
    assert score == pytest.approx(reference, rel=RELATIVE)


def test_a_batch_scores_every_request_as_it_would_alone_and_keeps_the_order(adapter):
    """Right padding cannot reach backwards through a causal model, so a
    short row in a batch scores what it scores alone, and the answers come
    back paired with the requests that asked for them."""
    pairs = [("the ", "quick"), ("a ", "b"), ("hello ", "world")]
    together = adapter.loglikelihood([instance(*pair) for pair in pairs])
    assert len(together) == len(pairs)
    for pair, (score, _) in zip(pairs, together, strict=True):
        (alone, _), = adapter.loglikelihood([instance(*pair)])
        assert score == pytest.approx(alone, rel=RELATIVE)
    reordered = adapter.loglikelihood([instance(*pair) for pair in reversed(pairs)])
    assert [score for score, _ in reordered] == pytest.approx(
        [score for score, _ in reversed(together)], rel=RELATIVE)


@pytest.mark.network
def test_a_real_task_suite_runs_against_a_run(tmp_path):
    """One tiny suite end to end, over a run whose 512-id window holds
    hellaswag's endings; the module's 16-id run refuses them, as HFLM would.
    lm_eval 0.4 ships no task whose data is in the package, so hellaswag's
    four documents come from the Hub."""
    import lm_eval

    make_lm_run(tmp_path, max_seq_len=512)
    results = lm_eval.simple_evaluate(
        model=DewLM(TextGeneration.from_run(str(tmp_path)), batch_size=4),
        tasks=["hellaswag"], limit=4, bootstrap_iters=0)
    scores = results["results"]["hellaswag"]
    assert 0.0 <= scores["acc,none"] <= 1.0
