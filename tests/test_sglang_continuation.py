"""A policy's stop ids on SGLang, which also ends a draw on its model's own EOS ids
(`dew.inference.continuation`), through the official OpenAI SDK against a stand-in for SGLang
v0.5.20's OpenAI routes at the HTTP transport.

The stand-in draws as SGLang does where it matters here: each token from logits over the ids
before it, with a frequency penalty over the ids it drew, a logit bias, and its stop ids masked
while `min_tokens` is still owed (`BatchedMinNewTokensPenalizer`), under a seed keyed by the seed
and the token's place in the sequence (`multinomial_with_seed`), the same for every parallel
sample of a request. It ends a draw on `stop_token_ids` and on its model's own EOS, `NATIVE`,
unless the request ignores every token stop (`Req._check_token_based_finish`); answers a batch
prompt by prompt; decodes a draw after its prompt's last ids, as SGLang's detokenizer does, with a
tokenizer that drops the leading space of what it decodes, as SentencePiece does; trims the stop
it ended on unless told not to; reports every text offset as -1 and an empty top-logprob list for
`logprobs=0` (openai/utils.py); and streams a body read as it arrives. The same stand-in told its
model has no EOS of its own is the policy's own engine, the reference a continued draw is held to."""

import asyncio
import json
import math
from collections import Counter
from importlib import import_module

import numpy as np
import pytest

from dew.inference import OpenAICompletion
from dew.sampling import Sampling

openai = pytest.importorskip("openai", reason="optional inference-clients extra")
httpx2 = import_module("httpx2")

NATIVE = 2
"""The served model's own EOS, which no policy here names."""
EOS = 7
"""The policy's stop id."""
SPECIAL = (NATIVE, EOS)
"""The ids a decoding that skips special tokens leaves out."""
CONTEXT = 5
"""How many prompt ids SGLang decodes a draw after (`INIT_INCREMENTAL_DETOKENIZATION_OFFSET`)."""
VOCABULARY = 10
TEMPLATE_HEAD = [9, 9]
"""The ids a server's completion template puts before a token row."""
POLICY = Sampling(eos_token_ids=EOS)


def logits(sequence, adapter):
    """A model whose next id depends on the last one: its own EOS likely after an odd id, the
    policy's stop rarely, and the adapter `b` leaning to 4."""
    scores = np.full(VOCABULARY, -1.0)
    scores[[0, 3, 5, 6]] = 0.5
    scores[NATIVE] = 1.5 if sequence[-1] % 2 else -0.5
    scores[EOS] = -0.3
    if adapter == "b":
        scores[4] = 1.0
    return scores


FRAGMENT = 8
"""An id that is the first byte of a character the next id completes, decoded alone as U+FFFD."""


def decoded(ids, skip=True):
    """A tokenizer's decoding of `ids`: each id a word, the first word's leading space dropped, as
    SentencePiece drops it, and `FRAGMENT` joined to the id after it, as a byte-level tokenizer
    joins a character's bytes, or read as U+FFFD where nothing completes it."""
    kept = [token for token in ids if not (skip and token in SPECIAL)]
    words = []
    for at, token in enumerate(kept):
        if at and kept[at - 1] == FRAGMENT:
            continue
        words.append(f" ü{kept[at + 1]}" if token == FRAGMENT and at + 1 < len(kept)
                     else " \ufffd" if token == FRAGMENT else f" w{token}")
    return "".join(words).lstrip()


def generated(prompt, ids, skip=True):
    """A draw's text as SGLang decodes it: after the prompt's last `CONTEXT` ids."""
    context = list(prompt[-CONTEXT:])
    return decoded([*context, *ids], skip)[len(decoded(context, skip)):]


def streamed_body(records, *, asynchronous):
    """An SSE body the transport hands over as it is read, so the response stays open until then."""
    lines = [*(f"data: {json.dumps(record)}\n\n".encode() for record in records), b"data: [DONE]\n\n"]

    async def arriving():
        for line in lines:
            yield line

    return arriving() if asynchronous else iter(lines)


class SGLang:
    """SGLang's `/v1/completions`, `/v1/chat/completions` and `/detokenize` over `logits`, its
    model ending a draw on `NATIVE` where `own_eos`; every request it answered is in `requests`
    with its headers."""

    def __init__(self, own_eos=True, asynchronous=False, template=False):
        self.own_eos, self.asynchronous, self.template, self.requests = own_eos, asynchronous, template, []
        self.unseeded = np.random.default_rng(0)

    def __call__(self, request):
        body = json.loads(request.content)
        path = request.url.path
        self.requests.append((path, body, dict(request.headers)))
        if path.endswith("/detokenize"):
            # One row of ids decodes to one text, a batch of rows to one text each (DetokenizeRequest).
            skip, tokens = body.get("skip_special_tokens", True), body["tokens"]
            batched = bool(tokens) and isinstance(tokens[0], list)
            return httpx2.Response(200, json={"text": [decoded(row, skip) for row in tokens] if batched
                                              else decoded(tokens, skip)})
        chat = path.endswith("/chat/completions")
        budget = body.get("max_completion_tokens" if chat else "max_tokens")
        asked = [[9, 9]] if chat else [list(row) for row in body["prompt"]]
        # A completion template the server holds wraps each token row, as SGLang's may, unless
        # the request's empty suffix passes it through as it is.
        prompts = asked if chat or body.get("suffix") == "" or not self.template else [
            [*TEMPLATE_HEAD, *row] for row in asked]
        adapters = body.get("lora_path")
        n = body.get("n", 1)
        choices = []
        for index in range(len(prompts) * n):
            row = index // n
            adapter = adapters[row] if isinstance(adapters, list) else adapters
            choices.append({**self.draw(index, prompts[row], adapter, body, budget), "asked": asked[row]})
        if body.get("stream"):
            return self.stream(choices, chat, body)
        produced = sum(len(choice["ids"]) for choice in choices)
        prompt_tokens = sum(len(prompts[index // n]) for index in range(len(prompts) * n))
        return httpx2.Response(200, json={
            "id": "sglang", "model": "tiny", "created": 1,
            "object": "chat.completion" if chat else "text_completion",
            "choices": [self.rendered(choice, chat, body) for choice in choices],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": produced,
                      "total_tokens": prompt_tokens + produced}})

    def draw(self, index, prompt, adapter, body, budget):
        stops = set() if body.get("ignore_eos") else set(body.get("stop_token_ids") or ()) | (
            {NATIVE} if self.own_eos else set())
        bias = {int(token): value for token, value in (body.get("logit_bias") or {}).items()}
        seed, ids, scores = body.get("seed"), [], []
        while len(ids) < budget:
            sequence = [*prompt, *ids]
            weights = logits(sequence, adapter) - body.get("frequency_penalty", 0) * np.bincount(
                ids, minlength=VOCABULARY)
            for token, value in bias.items():
                weights[token] += value
            if len(ids) < (body.get("min_tokens") or 0):
                # SGLang's own EOS is masked too, whichever stops the request names.
                weights[sorted(stops | {NATIVE})] = -np.inf
            probabilities = np.exp(weights - weights.max())
            probabilities /= probabilities.sum()
            rng = self.unseeded if seed is None else np.random.default_rng((seed, len(sequence)))
            token = int(rng.choice(VOCABULARY, p=probabilities))
            ids.append(token)
            scores.append(float(np.log(probabilities[token])))
            if token in stops:
                return {"index": index, "ids": ids, "scores": scores, "prompt": prompt, "reason": "stop",
                        "matched": token}
        return {"index": index, "ids": ids, "scores": scores, "prompt": prompt, "reason": "length",
                "matched": None}

    @staticmethod
    def shown(choice, body):
        ids = choice["ids"]
        if choice["matched"] is not None and not body.get("no_stop_trim"):
            ids = ids[:-1]
        text = generated(choice["prompt"], ids, body.get("skip_special_tokens", True))
        return decoded(choice["asked"]) + text if body.get("echo") else text

    def rendered(self, choice, chat, body):
        if chat:
            return {"index": choice["index"], "finish_reason": choice["reason"],
                    "matched_stop": choice["matched"],
                    "message": {"role": "assistant", "content": self.shown(choice, body)}}
        entry = {"index": choice["index"], "text": self.shown(choice, body),
                 "finish_reason": choice["reason"], "matched_stop": choice["matched"]}
        if body.get("return_token_ids"):
            entry.update(token_ids=choice["ids"], prompt_token_ids=choice["prompt"])
        if body.get("logprobs") is not None:
            entry["logprobs"] = {"tokens": [f"w{token}" for token in choice["ids"]],
                                 "token_logprobs": choice["scores"], "top_logprobs": [],
                                 "text_offset": [-1] * len(choice["ids"])}
        return entry

    def stream(self, choices, chat, body):
        records = []
        for choice in choices:
            for at in range(len(choice["ids"])):
                last = at == len(choice["ids"]) - 1
                reason, matched = (choice["reason"], choice["matched"]) if last else (None, None)
                piece = generated(choice["prompt"], choice["ids"][:at + 1])[len(
                    generated(choice["prompt"], choice["ids"][:at])):]
                part = ({"index": choice["index"], "delta": {"content": piece}} if chat
                        else {"index": choice["index"], "text": piece})
                records.append({"id": "sglang", "model": "tiny", "created": 1,
                                "object": "chat.completion.chunk" if chat else "text_completion",
                                "choices": [{**part, "finish_reason": reason, "matched_stop": matched}]})
        return httpx2.Response(200, content=streamed_body(records, asynchronous=self.asynchronous),
                               headers={"Content-Type": "text/event-stream"})


def sglang(own_eos=True, asynchronous=False, template=False):
    engine = SGLang(own_eos, asynchronous, template)
    if asynchronous:
        async def answer(request):
            await request.aread()
            return engine(request)

        client = openai.AsyncOpenAI(base_url="http://sglang/v1", api_key="fixture", max_retries=0,
                                   http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(answer)))
    else:
        client = openai.OpenAI(base_url="http://sglang/v1", api_key="fixture", max_retries=0,
                               http_client=httpx2.Client(transport=httpx2.MockTransport(engine)))
    return OpenAICompletion("tiny", client, provider="sglang"), engine


def completions(engine):
    return [body for path, body, _ in engine.requests if path.endswith("/completions")]


@pytest.mark.parametrize("policy, fields", [
    (POLICY, {}),
    (Sampling(eos_token_ids=EOS, frequency_penalty=0.7), {}),
    (Sampling(eos_token_ids=EOS, frequency_penalty=0.7), {"extra_body": {"logit_bias": {"5": 0.4}}}),
    (POLICY, {"n": 2, "extra_body": {"lora_path": ["a", "b"]}}),
    (POLICY, {"extra_body": {"no_stop_trim": True, "skip_special_tokens": False}}),
])
def test_a_continued_draw_is_the_policys_uninterrupted_draw_under_its_seed(policy, fields):
    """Where SGLang ends a draw on its model's own EOS, the draw goes on from its prompt and the ids it
    drew on the rest of its budget, and under a seed it is the draw an engine without that EOS
    draws: the same ids, log-probabilities, text, finish and usage, for a frequency penalty beside
    a logit bias in extra_body, two samples of two rows on their own adapters, and special tokens
    and the stop's text kept too. A prompt ends on a character's first byte, which its draw's first
    id completes, as SGLang decodes it. Most draws here go on at least once."""
    completion, engine = sglang()
    reference, _ = sglang(own_eos=False)
    continued = 0
    for key in range(12):
        engine.requests.clear()
        done = completion([[1, 4, FRAGMENT], [3]], 8, key=key, sampling=policy, logprobs=0, **fields)
        expected = reference([[1, 4, FRAGMENT], [3]], 8, key=key, sampling=policy, logprobs=0, **fields)
        assert done.tokens == expected.tokens and done.texts == expected.texts
        assert done.finish_reasons == expected.finish_reasons and done.log_probs == expected.log_probs
        assert done.usage.completion_tokens == sum(len(row) for row in done.tokens)
        continued += len(completions(engine)) - 1
    assert continued >= 6


def test_a_continuation_sends_what_the_policy_draws_by():
    """A continuation's request carries the policy and its seed, its own row's adapter, an empty
    suffix past any completion template, a frequency penalty's counts of the ids drawn as logit bias
    where the SDK sends it, and the request's headers on every call."""
    completion, engine = sglang()
    policy = Sampling(eos_token_ids=EOS, frequency_penalty=0.5)
    for key in range(40):
        engine.requests.clear()
        completion([[1], [3]], 8, key=key, sampling=policy, extra_headers={"x-tenant": "t"}, timeout=None,
                   extra_body={"lora_path": ["a", "b"], "suffix": "tail"})
        if len(completions(engine)) > 1:
            break
    first, then = completions(engine)[:2]
    (row,) = then["prompt"]
    drawn = row[1:]
    assert then["seed"] == first["seed"] and then["n"] == 1 and then["suffix"] == ""
    assert then["max_tokens"] == 8 - len(drawn) and then["stop_token_ids"] == [EOS]
    assert then["lora_path"] == ["a", "b"][row[0] != 1]
    assert then["logit_bias"] == {str(token): -0.5 * count for token, count in Counter(drawn).items()}
    assert all(headers.get("x-tenant") == "t" for _, _, headers in engine.requests)
    assert any(path.endswith("/detokenize") for path, _, _ in engine.requests)


PROBABILITIES = {0: 0.25, NATIVE: 0.5, EOS: 0.25}
"""A model that draws each id on its own, its own EOS first in half the draws."""


class Independent(SGLang):
    def draw(self, index, prompt, adapter, body, budget):
        stops = set(body.get("stop_token_ids") or ()) | {NATIVE}
        seed, ids = body.get("seed"), []
        while len(ids) < budget:
            rng = np.random.default_rng((seed, len(prompt) + len(ids)))
            ids.append(int(rng.choice(list(PROBABILITIES), p=list(PROBABILITIES.values()))))
            if ids[-1] in stops:
                return {"index": index, "ids": ids, "scores": [0.0] * len(ids), "prompt": prompt,
                        "reason": "stop", "matched": ids[-1]}
        return {"index": index, "ids": ids, "scores": [0.0] * len(ids), "prompt": prompt,
                "reason": "length", "matched": None}


def test_a_continued_draw_has_the_policys_distribution():
    """Each id is NATIVE with probability 1/2, EOS 1/4 and 0 1/4, on a budget of three: the policy
    ends a draw on EOS alone, so a draw is each id sequence up to the first EOS or the budget, with
    the product of its ids' probabilities. Over N draws each sequence's frequency is within five
    binomial standard deviations sqrt(p (1 - p) / N) of its probability, and every drawn sequence is
    one the policy draws. Drawing again from scratch when SGLang ends on NATIVE would weigh the
    sequences without NATIVE by more than their probability: [EOS] at about 0.42 for 0.25."""
    engine = Independent()
    client = openai.OpenAI(base_url="http://sglang/v1", api_key="fixture", max_retries=0,
                           http_client=httpx2.Client(transport=httpx2.MockTransport(engine)))
    completion = OpenAICompletion("tiny", client, provider="sglang")
    count = 4000
    outcomes = Counter(completion([[1]], 3, key=key, sampling=POLICY).tokens[0] for key in range(count))
    exact = {}
    for length in (1, 2, 3):
        for head in np.ndindex(*(2,) * (length - 1)):
            ids = tuple((0, NATIVE)[at] for at in head)
            exact[(*ids, EOS)] = math.prod(PROBABILITIES[token] for token in (*ids, EOS))
            if length == 3:
                for last in (0, NATIVE):
                    exact[(*ids, last)] = math.prod(PROBABILITIES[token] for token in (*ids, last))
    assert math.isclose(sum(exact.values()), 1.0)
    assert set(outcomes) <= set(exact)
    for ids, probability in exact.items():
        spread = math.sqrt(probability * (1 - probability) / count)
        assert abs(outcomes[ids] / count - probability) <= 5 * spread, ids


@pytest.mark.parametrize("sampling, fields", [
    (Sampling(eos_token_ids=EOS, presence_penalty=0.5), {}),
    (Sampling(eos_token_ids=EOS, repetition_penalty=1.2), {}),
    (POLICY, {"extra_body": {"regex": "w3.*"}}),
    (POLICY, {"stop": ["w5"]}),
])
def test_a_control_a_continuation_cannot_carry_is_refused_where_a_draw_needs_one(sampling, fields):
    """A presence or repetition penalty counts each id drawn before the cut once, and stop strings,
    patterns and constraints match from where a draw starts, none of which a continuation from the
    ids before it holds: a draw that needs one is refused, and the same request whose draws all end
    on the policy's stop is drawn."""
    completion, _ = sglang()
    with pytest.raises(ValueError):
        for key in range(40):
            completion([[1]], 8, key=key, sampling=sampling, **fields)
    stopping = Sampling(eos_token_ids=(NATIVE, EOS), presence_penalty=sampling.presence_penalty,
                        repetition_penalty=sampling.repetition_penalty)
    assert completion([[1]], 8, key=0, sampling=stopping, **fields).tokens


@pytest.mark.parametrize("template", [False, True])
def test_an_echoed_prompt_heads_the_continued_draw_as_sglang_writes_it(template):
    """An echoed prompt heads each choice's text, the request's own token row decoded with special
    tokens skipped, not the model prompt a server's completion template made of it: a draw that
    goes on is the policy's own engine's, echo included."""
    completion, engine = sglang(template=template)
    reference, _ = sglang(own_eos=False, template=template)
    continued = 0
    for key in range(12):
        engine.requests.clear()
        done = completion([[1, 4, NATIVE], [3]], 8, key=key, sampling=POLICY, echo=True)
        assert done.texts == reference([[1, 4, NATIVE], [3]], 8, key=key, sampling=POLICY, echo=True).texts
        continued += len(completions(engine)) - 1
    assert continued >= 3 and all("echo" not in body for body in completions(engine)[1:])


@pytest.mark.parametrize("policy", [POLICY, Sampling()])
def test_an_owed_minimum_output_is_refused_before_a_draw(policy):
    """SGLang masks its model's own EOS ids while min_tokens is owed, ids the policy draws as any
    other, and a policy with no stop ids of its own too, so a request owing one is refused before it
    is sent, on every route."""
    completion, engine = sglang()
    for draw in (lambda: completion([[1]], 8, key=0, sampling=policy, extra_body={"min_tokens": 2}),
                 lambda: completion.chat([{"role": "user", "content": "hi"}], 8, key=0, sampling=policy,
                                         extra_body={"min_tokens": 2})):
        with pytest.raises(ValueError):
            draw()
    assert engine.requests == []
    assert completion([[1]], 8, key=0, sampling=POLICY, extra_body={"min_tokens": 0}).tokens


def test_a_chat_or_stream_sglang_ends_on_its_own_eos_is_refused_and_its_stream_closed():
    """Only a whole completion goes on past SGLang's own EOS, so a chat or a stream it ends there is
    refused, a stream at that chunk after the chunks before it and its response closed; one whose
    draws end on the policy's stop is the engine's own, and a stream closed before its first read
    closes its open response."""
    completion, _ = sglang()
    messages = [{"role": "user", "content": "hi"}]
    refusals = Counter()
    for key in range(20):
        stream = completion.stream([[1]], 8, key=key, sampling=POLICY)
        assert not stream.response.is_closed
        try:
            chunks = list(stream)
        except ValueError:
            refusals["stream"] += 1
            assert stream.response.is_closed
        else:
            assert chunks[-1].choices[0].finish_reason in ("stop", "length")
        try:
            completion.chat(messages, 8, key=key, sampling=POLICY)
        except ValueError:
            refusals["chat"] += 1
    assert 0 < refusals["stream"] < 20 and 0 < refusals["chat"] < 20
    unread = completion.stream([[1]], 8, key=0, sampling=POLICY)
    unread.close()
    assert unread.response.is_closed


def test_the_async_client_goes_on_as_the_sync_one_does():
    """`acall` goes on as `__call__` does, and `astream` is checked as `stream` is, closed by
    `aclose` before its first read too."""
    async def run():
        completion, _ = sglang(asynchronous=True)
        try:
            done = [await completion.acall([[1]], 8, key=key, sampling=POLICY) for key in range(6)]
            stream = await completion.astream([[1]], 8, key=0, sampling=Sampling(eos_token_ids=(NATIVE, EOS)))
            chunks = [chunk async for chunk in stream]
            unread = await completion.astream([[1]], 8, key=0, sampling=POLICY)
            open_before = not unread.response.is_closed
            await unread.aclose()
            closed = unread.response.is_closed
        finally:
            await completion.client.close()
        return done, chunks, open_before and closed

    done, chunks, closed = asyncio.run(run())
    completion, _ = sglang()
    assert [draw.tokens for draw in done] == [completion([[1]], 8, key=key, sampling=POLICY).tokens
                                              for key in range(6)]
    assert chunks and chunks[-1].choices[0].finish_reason in ("stop", "length") and closed
