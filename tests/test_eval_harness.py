"""The lm-evaluation-harness adapter over a saved run.

The arithmetic is checked against a direct log-softmax over the model's own
logits, which is the only thing a harness adapter can get wrong on its own:
everything else is the suite's. The end-to-end run of a real task needs the
task's dataset from the Hub, so it carries the `network` marker; lm_eval
0.4 ships no task whose data is in the package.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_inference import make_lm_run

pytest.importorskip("lm_eval")

from lm_eval.api.instance import Instance  # noqa: E402  the extra has to be there first

from dew.eval.harness import DewLM  # noqa: E402
from dew.inference import TextGeneration  # noqa: E402


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


def _reference(task, context: str, continuation: str) -> tuple[float, bool]:
    """The same number by hand: log-softmax over the whole row's logits, read
    at the continuation's targets."""
    prefix = np.asarray(task.processor([context]).tokens)[0]
    whole = np.asarray(task.processor([context + continuation]).tokens)[0]
    logits = task.model.apply(task.variables, jnp.asarray(whole[None, :-1], jnp.int32))
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)[0]
    width = len(whole) - len(prefix)
    targets = whole[1:]
    picked = [float(log_probs[slot, targets[slot]]) for slot in range(len(targets) - width, len(targets))]
    greedy = [int(np.argmax(np.asarray(log_probs[slot]))) == int(targets[slot])
              for slot in range(len(targets) - width, len(targets))]
    return float(sum(picked)), all(greedy)


def test_loglikelihood_is_the_log_softmax_over_the_models_own_logits(adapter):
    """The one number the adapter computes itself, against the direct one.

    The pair fits the model's 16-id context, so no window is taken and the
    two read the same row."""
    expected, greedy = _reference(adapter.task, "the ", "quick")
    (score, is_greedy), = adapter.loglikelihood([instance("the ", "quick")])
    assert score == pytest.approx(expected, abs=1e-5)
    assert is_greedy == greedy


def test_a_batch_scores_every_request_as_it_would_alone_and_keeps_the_order(adapter):
    """Right padding cannot reach backwards through a causal model, so a
    short row in a batch scores what it scores alone, and the answers come
    back paired with the requests that asked for them."""
    pairs = [("the ", "quick"), ("a ", "b"), ("hello ", "world")]
    together = adapter.loglikelihood([instance(*pair) for pair in pairs])
    assert len(together) == len(pairs)
    for pair, (score, _) in zip(pairs, together, strict=True):
        alone, _ = _reference(adapter.task, *pair)
        assert score == pytest.approx(alone, abs=1e-5)
    reordered = adapter.loglikelihood([instance(*pair) for pair in reversed(pairs)])
    assert [score for score, _ in reordered] == pytest.approx(
        [score for score, _ in reversed(together)], abs=1e-6)


def test_rolling_likelihood_scores_every_token_after_the_first(adapter):
    """One window: the sum of the row's own per-target log-probabilities."""
    text = "the quick brown"
    tokens = np.asarray(adapter.task.processor([text]).tokens)[0]
    assert len(tokens) <= adapter.max_length
    logits = adapter.task.model.apply(adapter.task.variables,
                                      jnp.asarray(tokens[None, :-1], jnp.int32))
    log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)[0]
    expected = sum(float(log_probs[slot, tokens[slot + 1]]) for slot in range(len(tokens) - 1))
    (score,) = adapter.loglikelihood_rolling([instance(text)])
    assert score == pytest.approx(expected, abs=1e-5)


def test_a_string_longer_than_the_context_scores_in_consecutive_windows(adapter):
    """The model declares 16 ids, so a longer string is scored in windows
    rather than refused, and each window scores its own tokens."""
    text = "x" * (adapter.max_length * 2 + 1)
    tokens = adapter.tok_encode(text)
    assert len(tokens) > adapter.max_length
    (score,) = adapter.loglikelihood_rolling([instance(text)])
    windows = [tokens[start:start + adapter.max_length]
               for start in range(0, len(tokens), adapter.max_length)]
    counted = sum(len(window) - 1 for window in windows if len(window) > 1)
    assert counted == len(tokens) - len(windows)
    assert score < 0.0


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


def test_the_adapter_refuses_a_task_it_cannot_score(run):
    """It scores next-token likelihoods, so it takes the task that has them,
    and it needs the processor that turns the harness's text into tokens."""
    import dataclasses

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
def test_a_real_task_suite_runs_against_the_run(run):
    """One tiny suite end to end. lm_eval 0.4 ships no task whose data is in
    the package, so hellaswag's four documents come from the Hub."""
    import lm_eval

    results = lm_eval.simple_evaluate(
        model=DewLM(TextGeneration.from_run(str(run)), batch_size=4),
        tasks=["hellaswag"], limit=4, bootstrap_iters=0)
    scores = results["results"]["hellaswag"]
    assert 0.0 <= scores["acc,none"] <= 1.0
