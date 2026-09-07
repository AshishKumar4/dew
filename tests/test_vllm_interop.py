"""A decoder Dew trained and exported, served by a live vLLM server.

Network-marked, so an ordinary `-m "not network"` run deselects it, and it
skips rather than fails without a server. Nothing downloads: the served
model is a tiny Llama-layout export from a real Dew run, and the reference
side reads those same files back through `load_pretrained`.

What is held to account is `OpenAICompletion(provider="vllm")` against a
real vLLM engine: asked for the model's own argmax, the server reproduces
Dew's greedy draw token for token, and the vLLM-only controls the client
places in `extra_body` change what the server's sampler does. The launch:

    vllm serve <export> --host 127.0.0.1 --port <port> --dtype float32 \
        --enforce-eager --gpu-memory-utilization 0.2 --max-num-seqs 8 \
        --trust-request-chat-template
    OPENAI_BASE_URL=http://127.0.0.1:<port>/v1 pytest tests/test_vllm_interop.py

The last flag is what lets a chat request carry the template the export
does not ship; a host whose gcc is newer than the CUDA toolkit allows also
needs VLLM_USE_FLASHINFER_SAMPLER=0, or the engine fails to build
FlashInfer's sampling kernels at startup.

`--dtype float32` is what the logprob tolerance below is calibrated to.
The export records no `torch_dtype`, so vLLM's own default serves it in
bfloat16 through FlashAttention instead of fp32 through Triton: the argmax
survives that on these prompts, but the reported logprobs move from within
0.0013 of Dew's own to within 0.017, and the tolerance refuses the second.
This checkpoint's margin between first and second choice reaches down to
0.07 in logits, so neither dtype flips a token here; a wider model or a
longer draw is where that stops being true.

The base URL comes from the SDK's own variable, and the model id vLLM
reports is the path it was started with, which is how the reference finds
the same weights. Without `OPENAI_BASE_URL` the SDK would address
api.openai.com, so an unset variable skips.
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

openai = pytest.importorskip("openai", reason="optional inference-clients extra")

from dew.inference import OpenAICompletion
from dew.interop import Pretrained, load_pretrained
from dew.nn.inputs import ModelInputs
from dew.sampling.text import Sampling, generate

pytestmark = pytest.mark.network

PROMPTS = ("The trainer", "Dew trains", "The model", "This recipe")
DRAWN = 8
GREEDY = Sampling(temperature=0.0)
"""Argmax on both sides. vLLM overrides top-p/top-k/min-p at temperature
zero, so a control's effect is only observable at a live temperature."""
TEMPLATE = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
"""A template that renders one user turn as the bare prompt, so a chat
answer is comparable with the completion the same text produces."""


def greedy(loaded: Pretrained, head: list[int], count: int) -> list[int]:
    """Dew's fp32 argmax continuation of one prompt, as token ids."""
    drawn = generate(loaded.model, loaded.variables,
                     ModelInputs(tokens=jnp.asarray([head], jnp.int32)), count,
                     key=jax.random.key(0), sampling=GREEDY)
    return [int(token) for token in np.asarray(drawn.tokens)[0, len(head):]]


@pytest.fixture(scope="module")
def client() -> OpenAICompletion:
    """`OpenAICompletion` bound to whatever model the live server serves."""
    if not os.environ.get("OPENAI_BASE_URL"):
        pytest.skip("set OPENAI_BASE_URL to a live vLLM server's /v1 endpoint")
    sdk = openai.OpenAI(api_key="dew", max_retries=0)
    try:
        served = sdk.models.list().data
    except openai.APIConnectionError as unreachable:
        pytest.skip(f"no server at {os.environ['OPENAI_BASE_URL']}: {unreachable}")
    if not (Path(served[0].id) / "config.json").is_file():
        pytest.skip(f"served model {served[0].id!r} is not a local export directory")
    return OpenAICompletion(served[0].id, sdk, provider="vllm")


@pytest.fixture(scope="module")
def reference(client):
    """The served export read back into Dew: its tokenizer and fp32 model."""
    from transformers import AutoTokenizer

    return (AutoTokenizer.from_pretrained(client.model, local_files_only=True),
            load_pretrained(client.model, dtype="float32", attention_impl="xla"))


@pytest.fixture(scope="module")
def continuation(client) -> str:
    """The server's greedy answer to the first prompt, which the argmax test
    below ties to Dew's own draw; the control tests compare against it."""
    return client(PROMPTS[0], DRAWN, sampling=GREEDY).texts[0]


def test_the_server_tokenizes_and_draws_dews_own_greedy_continuation(client, reference):
    """The export describes one computation and both engines run it.

    `return_token_ids` is a vLLM extension the client passes through
    untouched, so the comparison is over ids and needs no detokenizing:
    the prompt ids the server charged against the tokenizer beside the
    weights, and the continuation against Dew's fp32 argmax. A transposed
    kernel or the other rope convention parts on the first token.
    """
    tokenizer, loaded = reference

    for prompt in PROMPTS:
        head = tokenizer.encode(prompt, add_special_tokens=False)
        answer = client(prompt, DRAWN, sampling=GREEDY, extra_body={"return_token_ids": True})
        choice = answer.responses[0].choices[0]
        ours = greedy(loaded, head, DRAWN)

        assert choice.prompt_token_ids == head, prompt
        assert choice.token_ids == ours, (f"{prompt!r}: vllm {choice.token_ids} "
                                          f"({answer.texts[0]!r}) against dew {ours} "
                                          f"({tokenizer.decode(ours)!r})")


def test_choices_stay_in_prompt_order_under_one_aggregate_usage(client, reference):
    """Many prompts are one request, and its usage is one total.

    The completions API reports no per-choice usage, so the client fills a
    per-choice count only where the aggregate is unambiguous: a single
    choice. The batch's answers still have to arrive in prompt order, which
    is the client's own reordering of the server's `index` fields.
    """
    tokenizer, _ = reference
    batch = client(list(PROMPTS), DRAWN, sampling=GREEDY)
    apart = tuple(client(prompt, DRAWN, sampling=GREEDY).texts[0] for prompt in PROMPTS)
    single = client(PROMPTS[0], DRAWN, sampling=GREEDY)

    assert batch.texts == apart
    assert batch.finish_reasons == ("length",) * len(PROMPTS)
    assert batch.usage is not None and batch.usage.completion_tokens == DRAWN * len(PROMPTS)
    assert batch.usage.prompt_tokens == sum(
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in PROMPTS)
    assert batch.token_counts == (None,) * len(PROMPTS)
    assert single.token_counts == (single.usage.completion_tokens,) == (DRAWN,)


def test_top_k_top_p_and_min_p_reach_the_servers_sampler(client, continuation):
    """The controls are the reason the client has a vLLM provider at all.

    Each of the three, set to the value that admits only the most likely
    token, has to collapse a live-temperature request onto the greedy
    answer. Temperature alone does not: this checkpoint's first steps are
    flat enough that sampling walks away from the argmax immediately, which
    is what makes the three agreements evidence of forwarding rather than
    of a model with nothing to choose.
    """
    def drawn(**controls: int | float) -> str:
        return client(PROMPTS[0], DRAWN, seed=1234,
                      sampling=Sampling(temperature=1.0, **controls)).texts[0]

    assert drawn(top_k=1) == continuation
    assert drawn(top_p=1e-6) == continuation
    assert drawn(min_p=1.0) == continuation
    assert drawn() != continuation


def test_a_stop_token_ends_the_completion_and_names_itself(client, reference):
    """`Sampling.eos_id` is the client's only termination control, and it
    reaches the server as `stop_token_ids`.

    Stopping the model on the token it was about to draw first is a
    termination the server cannot reach by accident. vLLM answers with the
    stop reason it acted on, and keeps the stop token out of the text, where
    Dew's own sampler counts EOS as a drawn action and returns it.
    """
    tokenizer, loaded = reference
    head = tokenizer.encode(PROMPTS[0], add_special_tokens=False)
    first = greedy(loaded, head, DRAWN)[0]

    answer = client(PROMPTS[0], DRAWN, sampling=Sampling(temperature=0.0, eos_id=first))
    ours = generate(loaded.model, loaded.variables,
                    ModelInputs(tokens=jnp.asarray([head], jnp.int32)), DRAWN,
                    key=jax.random.key(0), sampling=Sampling(temperature=0.0, eos_id=first))

    assert answer.finish_reasons == ("stop",)
    assert answer.responses[0].choices[0].stop_reason == first
    assert answer.texts[0] == ""
    assert int(ours.lengths[0]) == 1 and bool(ours.terminated[0])
    assert int(np.asarray(ours.tokens)[0, len(head)]) == first


def test_the_server_reports_dews_own_logprobs(client, reference):
    """Tighter than the token identity, and it drifts first.

    A logprob is a number rather than a decision, so it moves before an
    argmax flips. Both engines score in fp32 here and the kernels still
    differ, which is worth about 0.0013; the tolerance sits between that
    and the 0.07 logit margin this model keeps between its first and second
    choice, where drift would start flipping tokens.
    """
    tokenizer, loaded = reference
    worst = 0.0

    for prompt in PROMPTS:
        head = tokenizer.encode(prompt, add_special_tokens=False)
        answer = client(prompt, DRAWN, sampling=GREEDY, logprobs=1,
                        extra_body={"return_token_ids": True})
        choice = answer.responses[0].choices[0]
        reported = choice.logprobs.token_logprobs
        assert len(reported) == DRAWN, prompt

        logits = np.asarray(loaded.model.apply(
            loaded.variables, jnp.asarray([head + choice.token_ids], jnp.int32)))[0]
        for offset, token in enumerate(choice.token_ids):
            row = logits[len(head) + offset - 1].astype(np.float64)
            ours = float(row[token] - np.logaddexp.reduce(row))
            worst = max(worst, abs(ours - float(reported[offset])))

    assert worst < 0.01, f"max |dew - vllm| logprob {worst:.5f}"


def test_streaming_replays_the_completion_it_would_have_returned(client, continuation):
    """Streaming is the SDK's own iterator, and the client hands it back
    unwrapped. What the chunks have to add up to is the answer the same
    request returns whole, with the finish reason arriving on the chunk
    that carries the last token."""
    chunks = list(client.stream(PROMPTS[0], DRAWN, sampling=GREEDY))

    assert "".join(chunk.choices[0].text for chunk in chunks) == continuation
    assert [chunk.choices[0].finish_reason for chunk in chunks[:-1]] == [None] * (len(chunks) - 1)
    assert chunks[-1].choices[0].finish_reason == "length"


def test_chat_carries_the_template_the_export_does_not(client, continuation):
    """The gap between a served export and a chat endpoint.

    `save_pretrained_decoder` writes no chat template, so vLLM has nothing
    to render a conversation with and refuses one. A request that carries
    its own template is answered, whole or streamed, and with a template
    that renders the turn as the bare prompt the answer is the completion's
    answer: chat reaches the same weights under the same policy.
    """
    messages = [{"role": "user", "content": PROMPTS[0]}]
    template = {"chat_template": TEMPLATE}

    with pytest.raises(openai.BadRequestError):
        client.chat(messages, DRAWN, sampling=GREEDY)
    answered = client.chat(messages, DRAWN, sampling=GREEDY, extra_body=template)
    parts = [chunk.choices[0].delta.content for chunk
             in client.chat(messages, DRAWN, sampling=GREEDY, stream=True, extra_body=template)
             if chunk.choices]

    assert answered.choices[0].message.content == continuation
    assert answered.choices[0].finish_reason == "length"
    assert "".join(part for part in parts if part is not None) == continuation


def test_the_server_refuses_impossible_controls_but_ignores_unknown_ones(client):
    """Where the client's checks end and the server's begin.

    `Sampling` admits top-p zero and Dew's own sampler keeps the argmax
    under it; vLLM's sampler rejects the value outright, so a policy that
    runs natively is a 400 here. The context the export declares and the
    server's logprob cap are refused the same way. An unrecognised control
    is not: vLLM's request models allow extra fields, so a misspelled
    vLLM-only key in `extra_body` is dropped silently, which is why the
    client builds the controls it supports itself.
    """
    with pytest.raises(openai.BadRequestError):
        client(PROMPTS[0], DRAWN, sampling=Sampling(temperature=1.0, top_p=0.0))
    with pytest.raises(openai.BadRequestError):
        client(PROMPTS[0], 200, sampling=GREEDY)
    with pytest.raises(openai.BadRequestError):
        client(PROMPTS[0], DRAWN, sampling=GREEDY, logprobs=1000)

    ignored = client(PROMPTS[0], DRAWN, sampling=GREEDY, extra_body={"top_kk": 1})

    assert ignored.finish_reasons == ("length",)
