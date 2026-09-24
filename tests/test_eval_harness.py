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

import dataclasses
from pathlib import Path

import pytest
from test_inference import make_lm_run

pytest.importorskip("lm_eval")

from lm_eval.api.instance import Instance  # noqa: E402  the extra has to be there first

from dew.eval.harness import DewLM  # noqa: E402
from dew.inference import TextGeneration  # noqa: E402

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
"""The byte `\\x01` doubles as BOS in the BOS-vocabulary case; no test text holds it."""


def _byte_tokenizer(bos: bool):
    """Dew's `ByteTokenizer` as a transformers tokenizer: one id per utf-8 byte,
    id = byte value, EOS 255, which is what `HFLM` needs to be handed. With
    `bos`, every encoding starts with BOS, as Llama, Mistral and Gemma
    vocabularies do, and `HFLM` conditions first tokens on it."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    symbols = bytes_to_unicode()
    vocabulary = {symbols[byte]: byte for byte in range(256)}
    tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    if not bos:
        return PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token=symbols[255])
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{symbols[BOS]} $A", special_tokens=[(symbols[BOS], BOS)])
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, eos_token=symbols[255],
                                   bos_token=symbols[BOS])


def _pair(max_length: int, bos_directory=None):
    """The same weights behind `DewLM` and behind lm-eval's `HFLM`, both
    scoring windows of `max_length` ids. With `bos_directory`, both read the
    BOS vocabulary: Dew as the run's `HFTokenizer` over the saved files."""
    import torch
    from lm_eval.models.huggingface import HFLM
    from transformers import LlamaForCausalLM

    from dew.data.text import ByteTokenizer, HFTokenizer
    from dew.inference.pipeline import RunProcessor
    from dew.interop.pretrained import load_pretrained
    from dew.sampling import Sampling

    reference = _byte_tokenizer(bos_directory is not None)
    if bos_directory is None:
        run_tokenizer = ByteTokenizer()
    else:
        reference.save_pretrained(str(bos_directory))
        run_tokenizer = HFTokenizer(str(bos_directory), local_files_only=True)
    loaded = load_pretrained(LLAMA, dtype="float32", attention_impl="reference",
                             max_seq_len=max_length)
    ours = DewLM(TextGeneration(loaded.model, loaded.variables, RunProcessor(run_tokenizer),
                                sampling=Sampling(eos_id=255)), batch_size=2)
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


def test_a_source_processor_reads_bos_off_the_tokenizer_it_holds():
    """A source's processor, however it was built (the GGUF loader builds one
    from its four fields), answers the BOS its tokenizer has, so a harness
    over it does not fall back to EOS."""
    from dew.interop.pretrained import Processor

    assert Processor(_byte_tokenizer(True), {}, {}, 256).bos_id == BOS
    assert Processor(_byte_tokenizer(False), {}, {}, 256).bos_id is None


def test_a_bos_vocabulary_conditions_first_tokens_on_bos_as_lm_eval_does(tmp_path):
    """`HFLM.prefix_token_id` is the tokenizer's BOS when it has one. An empty
    context and a rolling window's first token are conditioned on it, so a
    BOS vocabulary is scored after BOS, not after EOS."""
    ours, theirs = _pair(16, tmp_path)
    assert ours.prefix_token_id == theirs.prefix_token_id == BOS
    requests = [instance("", "empty context"), instance("the ", "quick")]
    for (score, greedy), (reference, reference_greedy) in zip(
            ours.loglikelihood(requests), theirs.loglikelihood(requests), strict=True):
        assert score == pytest.approx(reference, rel=RELATIVE)
        assert greedy == reference_greedy
    text = [instance(_text(40), request_type="loglikelihood_rolling")]
    assert ours.loglikelihood_rolling(text) == pytest.approx(theirs.loglikelihood_rolling(text),
                                                             rel=RELATIVE)


def test_a_continuation_longer_than_the_window_is_refused_as_lm_eval_refuses_it():
    """No row of `max_length` ids holds such a continuation with any of its
    context, so `HFLM` asserts against it and the adapter refuses it rather
    than score it without the context the request conditions on."""
    ours, theirs = _pair(16)
    request = [instance("The capital of France is", _text(21))]
    with pytest.raises(AssertionError):
        theirs.loglikelihood(request)
    with pytest.raises(ValueError, match="is 21 ids, longer than this model's 16-id window"):
        ours.loglikelihood(request)


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


def test_a_continuation_that_adds_no_token_is_refused_not_scored_as_certain(adapter):
    """lm-eval's HFLM asserts every continuation has ids. Scored, an empty
    one would be probability 1 and greedy, and win every multiple-choice
    comparison it is in."""
    with pytest.raises(ValueError, match="'' adds no token to its context"):
        adapter.loglikelihood([instance("The sky is", " blue"), instance("The sky is", "")])


def test_generate_until_cuts_the_answer_at_the_first_stop_string(adapter):
    """The budget is the request's, and a stop string cuts the decoded text."""
    full, = adapter.generate_until([instance("the ", {"until": [], "max_gen_toks": 4})])
    assert isinstance(full, str) and len(full) > 2
    again, = adapter.generate_until([instance("the ", {"until": [], "max_gen_toks": 4})])
    assert again == full, "the default policy is greedy, so one request draws one answer"
    stop = full[1:3]
    cut, = adapter.generate_until([instance("the ", {"until": [stop], "max_gen_toks": 4})])
    assert cut == full[:full.index(stop)]
    assert len(cut) < len(full)


def test_generate_until_reads_each_request_as_lm_evals_own_model_does():
    """A budget named `max_new_tokens`, `do_sample=False` beside a
    temperature, which is greedy, and a context longer than the window less
    the budget, which keeps its last ids: each answers what `HFLM` answers
    on the same weights, the EOS token's text among the stop strings."""
    ours, theirs = _pair(32)
    for context, controls in (("the ", {"until": [], "max_new_tokens": 6}),
                              ("the ", {"until": [], "max_gen_toks": 6, "do_sample": False,
                                        "temperature": 0.7}),
                              (_text(60), {"until": [], "max_gen_toks": 6})):
        request = [instance(context, controls, request_type="generate_until")]
        assert ours.generate_until(request) == theirs.generate_until(request), controls


def test_the_adapter_refuses_a_task_it_cannot_score(run):
    """It scores next-token likelihoods, so it takes the task that has them,
    and it needs the processor that turns the harness's text into tokens."""
    task = TextGeneration.from_run(str(run))
    with pytest.raises(TypeError, match="TextToImage is a different task"):
        DewLM(_NotText())
    with pytest.raises(ValueError, match="needs the processor"):
        DewLM(dataclasses.replace(task, processor=None))
    with pytest.raises(ValueError, match="batch_size is a positive integer"):
        DewLM(task, batch_size=0)


class _NotText:
    """Stands in for another task kind; only its class name is read."""


_NotText.__name__ = "TextToImage"


def test_the_registry_answers_dew_with_this_adapter_built_from_a_run(run):
    """`--model dew --model_args run=<directory>` is this construction."""
    from lm_eval.api.registry import get_model

    assert get_model("dew") is DewLM
    built = get_model("dew").create_from_arg_string(f"run={run},batch_size=2")
    assert isinstance(built, DewLM) and built.batch_size == 2
    with pytest.raises(ValueError, match="name it with --model_args run="):
        get_model("dew").create_from_arg_string("batch_size=2")


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
