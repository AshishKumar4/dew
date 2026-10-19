"""A policy's stop ids on SGLang, which also ends a draw on its model's own EOS ids
(`dew.inference.continuation`), through the official OpenAI SDK against a stand-in for SGLang
v0.5.20's OpenAI routes at the HTTP transport.

The stand-in draws as SGLang does where it matters here: each token from logits over the ids
before it, a frequency penalty over the ids it drew and a logit bias added to them, under a seed
keyed by the seed and the token's place in the sequence (`multinomial_with_seed`); it ends a draw
on `stop_token_ids` and on its model's own EOS, `NATIVE`, unless the request ignores every token
stop or still owes `min_tokens` (`Req._check_token_based_finish`); it trims the stop it ended on
from the text unless told not to, reports every text offset as -1 and an empty top-logprob list
for `logprobs=0` (openai/utils.py), and refuses the chat route's token-id flags on a stream. The
same stand-in told its model has no EOS of its own is the policy's own engine, the reference a
continued draw is held to."""

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
"""The ids SGLang's text leaves out, as it skips special tokens."""
TEMPLATE = [9, 9]
"""The ids a chat request's messages render to."""
VOCABULARY = 10


def logits(sequence, adapter):
    """A model whose next id depends on the last one: its own EOS likely after an odd id, the
    policy's stop rarely, and the adapter `b` leaning to 4."""
    last = sequence[-1]
    scores = np.full(VOCABULARY, -1.0)
    scores[[0, 3, 5, 6]] = 0.5
    scores[NATIVE] = 1.5 if last % 2 else -0.5
    scores[EOS] = -0.3
    if adapter == "b":
        scores[4] = 1.0
    return scores


def text(ids, skip=True):
    return "".join(f"<{token}>" for token in ids if not (skip and token in SPECIAL))


class SGLang:
    """SGLang's `/v1/completions`, `/v1/chat/completions` and `/v1/detokenize` over `logits`, its
    model ending a draw on `NATIVE` where `own_eos`; every request it answered is in `requests`."""

    def __init__(self, own_eos=True):
        self.own_eos, self.requests = own_eos, []
        self.unseeded = np.random.default_rng(0)

    def __call__(self, request):
        body = json.loads(request.content)
        path = request.url.path
        self.requests.append((path, body))
        if path.endswith("/detokenize"):
            decoded = text(body["tokens"], body.get("skip_special_tokens", True))
            return httpx2.Response(200, json={"text": decoded})
        chat = path.endswith("/chat/completions")
        flagged = body.get("return_token_ids") or body.get("return_prompt_token_ids")
        if chat and body.get("stream") and flagged:
            return httpx2.Response(400, json={"error": {"message": "no token ids on a stream"}})
        budget = body.get("max_completion_tokens" if chat else "max_tokens")
        if (body.get("min_tokens") or 0) > budget:
            return httpx2.Response(400, json={"error": {"message": "min_tokens past max_tokens"}})
        prompts = [list(TEMPLATE)] if chat else [list(row) for row in body["prompt"]]
        adapters = body.get("lora_path")
        n = body.get("n", 1)
        choices = []
        for index in range(len(prompts) * n):
            adapter = adapters[index % len(prompts)] if isinstance(adapters, list) else adapters
            choices.append(self.draw(index, prompts[index % len(prompts)], adapter, body, budget))
        if body.get("stream"):
            return self.stream(choices, chat)
        generated = sum(len(choice["ids"]) for choice in choices)
        return httpx2.Response(200, json={
            "id": "sglang", "model": "tiny", "created": 1,
            "object": "chat.completion" if chat else "text_completion",
            "choices": [self.rendered(choice, chat, body) for choice in choices],
            "usage": {"prompt_tokens": sum(len(row) for row in prompts), "completion_tokens": generated,
                      "total_tokens": sum(len(row) for row in prompts) + generated}})

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
            probabilities = np.exp(weights - weights.max())
            probabilities /= probabilities.sum()
            rng = self.unseeded if seed is None else np.random.default_rng((seed, len(sequence)))
            token = int(rng.choice(VOCABULARY, p=probabilities))
            ids.append(token)
            scores.append(float(np.log(probabilities[token])))
            if token in stops and len(ids) >= (body.get("min_tokens") or 0):
                return {"index": index, "ids": ids, "scores": scores, "prompt": prompt, "reason": "stop",
                        "matched": token}
        return {"index": index, "ids": ids, "scores": scores, "prompt": prompt, "reason": "length",
                "matched": None}

    @staticmethod
    def shown(choice, body):
        ids = choice["ids"]
        if choice["matched"] is not None and not body.get("no_stop_trim"):
            ids = ids[:-1]
        return text(ids, body.get("skip_special_tokens", True))

    def rendered(self, choice, chat, body):
        if chat:
            entry = {"index": choice["index"], "finish_reason": choice["reason"],
                     "matched_stop": choice["matched"],
                     "message": {"role": "assistant", "content": self.shown(choice, body)}}
            if body.get("return_prompt_token_ids"):
                entry["prompt_token_ids"] = choice["prompt"]
            if body.get("return_token_ids"):
                entry["response_token_ids"] = choice["ids"]
            if body.get("logprobs"):
                entry["logprobs"] = {"content": [
                    {"token": f"<{token}>", "logprob": score, "bytes": list(f"<{token}>".encode()),
                     "top_logprobs": []}
                    for token, score in zip(choice["ids"], choice["scores"], strict=True)]}
            return entry
        entry = {"index": choice["index"], "text": self.shown(choice, body),
                 "finish_reason": choice["reason"], "matched_stop": choice["matched"]}
        if body.get("return_token_ids"):
            entry.update(token_ids=choice["ids"], prompt_token_ids=choice["prompt"])
        if body.get("logprobs") is not None:
            entry["logprobs"] = {"tokens": [f"<{token}>" for token in choice["ids"]],
                                 "token_logprobs": choice["scores"], "top_logprobs": [],
                                 "text_offset": [-1] * len(choice["ids"])}
        return entry

    def stream(self, choices, chat):
        records = []
        for choice in choices:
            for at, token in enumerate(choice["ids"]):
                last = at == len(choice["ids"]) - 1
                reason, matched = (choice["reason"], choice["matched"]) if last else (None, None)
                part = ({"index": choice["index"], "delta": {"content": text([token])}} if chat
                        else {"index": choice["index"], "text": text([token])})
                records.append({"id": "sglang", "model": "tiny", "created": 1,
                                "object": "chat.completion.chunk" if chat else "text_completion",
                                "choices": [{**part, "finish_reason": reason, "matched_stop": matched}]})
        content = "".join(f"data: {json.dumps(record)}\n\n" for record in records) + "data: [DONE]\n\n"
        return httpx2.Response(200, content=content, headers={"Content-Type": "text/event-stream"})


def sglang(own_eos=True, asynchronous=False):
    engine = SGLang(own_eos)
    if asynchronous:
        client = openai.AsyncOpenAI(base_url="http://sglang/v1", api_key="fixture", max_retries=0,
                                   http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(engine)))
    else:
        client = openai.OpenAI(base_url="http://sglang/v1", api_key="fixture", max_retries=0,
                               http_client=httpx2.Client(transport=httpx2.MockTransport(engine)))
    return OpenAICompletion("tiny", client, provider="sglang"), engine


@pytest.mark.parametrize("policy, fields", [
    (Sampling(eos_token_ids=EOS), {}),
    (Sampling(eos_token_ids=EOS, frequency_penalty=0.7), {}),
    (Sampling(eos_token_ids=EOS), {"extra_body": {"lora_path": ["a", "b"]}}),
    (Sampling(eos_token_ids=EOS), {"extra_body": {"min_tokens": 3}}),
    (Sampling(eos_token_ids=EOS), {"extra_body": {"no_stop_trim": True, "skip_special_tokens": False}}),
])
def test_a_continued_draw_is_the_policys_uninterrupted_draw_under_its_seed(policy, fields):
    """Where SGLang ends a draw on its model's own EOS, the draw goes on from its prompt and the ids it
    drew on the rest of its budget, and under a seed it is the draw an engine without that EOS
    draws: the same ids, log-probabilities, text, finish and usage, for a frequency penalty, per-row
    adapters, a minimum output and the stop's text kept too. Most draws here go on at least once."""
    completion, engine = sglang()
    reference, _ = sglang(own_eos=False)
    continued = 0
    for key in range(12):
        engine.requests.clear()
        done = completion([[1], [3]], 8, key=key, sampling=policy, logprobs=0, **fields)
        expected = reference([[1], [3]], 8, key=key, sampling=policy, logprobs=0, **fields)
        assert done.tokens == expected.tokens and done.texts == expected.texts
        assert done.finish_reasons == expected.finish_reasons and done.log_probs == expected.log_probs
        assert done.usage.completion_tokens == sum(len(row) for row in done.tokens)
        continued += sum(path.endswith("/completions") for path, _ in engine.requests) - 1
    assert continued >= 6


def test_a_continuation_sends_what_the_policy_draws_by():
    """A continuation's request carries the policy and its seed, its own row's adapter, the minimum
    output still owed, an empty suffix past any completion template, and a frequency penalty's
    counts of the ids drawn as logit bias."""
    completion, engine = sglang()
    policy = Sampling(eos_token_ids=EOS, frequency_penalty=0.5)
    for key in range(40):
        engine.requests.clear()
        completion([[1], [3]], 8, key=key, sampling=policy,
                   extra_body={"lora_path": ["a", "b"], "min_tokens": 1})
        follow = [(body, engine.requests[0][1]) for path, body in engine.requests[1:]
                  if path.endswith("/completions")]
        if follow:
            break
    (then, first), *_ = follow
    (row,) = then["prompt"]
    drawn = row[1:]
    assert then["seed"] == first["seed"] and then["n"] == 1 and then["suffix"] == ""
    assert then["max_tokens"] == 8 - len(drawn) and then["stop_token_ids"] == [EOS]
    assert then["lora_path"] == ["a", "b"][row[0] != 1] and then["min_tokens"] == 0
    assert then["logit_bias"] == {str(token): -0.5 * count for token, count in Counter(drawn).items()}


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
    outcomes = Counter(completion([[1]], 3, key=key, sampling=Sampling(eos_token_ids=EOS)).tokens[0]
                       for key in range(count))
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


@pytest.mark.parametrize("control", [{"presence_penalty": 0.5}, {"repetition_penalty": 1.2},
                                     {"extra_body": {"regex": "<3>.*"}}, {"stop": ["<5>"]}])
def test_a_control_a_continuation_cannot_carry_is_refused_where_a_draw_needs_one(control):
    """A presence or repetition penalty counts each id drawn before the cut once, and stop strings,
    patterns and constraints match from where a draw starts, none of which a continuation from the
    ids before it can hold: a draw that needs one is refused, and the same request whose draws all
    end on the policy's stop is drawn."""
    completion, _ = sglang()
    policy = (Sampling(eos_token_ids=EOS, presence_penalty=0.5) if "presence_penalty" in control
              else Sampling(eos_token_ids=EOS, repetition_penalty=1.2) if "repetition_penalty" in control
              else Sampling(eos_token_ids=EOS))
    fields = {name: value for name, value in control.items()
              if name not in ("presence_penalty", "repetition_penalty")}
    with pytest.raises(ValueError):
        for key in range(40):
            completion([[1]], 8, key=key, sampling=policy, **fields)
    stopping = Sampling(eos_token_ids=(NATIVE, EOS), presence_penalty=policy.presence_penalty,
                        repetition_penalty=policy.repetition_penalty)
    assert completion([[1]], 8, key=0, sampling=stopping, **fields).tokens


def test_a_stream_is_refused_where_sglang_ends_on_its_own_eos_and_closes_its_response():
    """A stream's text deltas would start their decoding again at a continuation's cut, so a stream
    SGLang ends on its own EOS is refused at that chunk, after the chunks before it, and its
    response is closed; one that ends on the policy's stop streams whole, and a stream closed before
    its first read closes its response."""
    completion, _ = sglang()
    for key in range(40):
        stream = completion.stream([[1]], 8, key=key, sampling=Sampling(eos_token_ids=EOS))
        chunks = []
        try:
            chunks.extend(stream)
        except ValueError:
            assert stream.response.is_closed
            break
        assert chunks[-1].choices[0].finish_reason in ("stop", "length")
    else:
        pytest.fail("no stream ended on SGLang's own EOS")
    unread = completion.stream([[1]], 8, key=0, sampling=Sampling(eos_token_ids=EOS))
    unread.close()
    assert unread.response.is_closed


def test_a_chat_goes_on_as_the_policys_chat_and_a_chat_stream_asks_for_no_ids():
    """A chat SGLang ended on its own EOS goes on from its rendered prompt on the completions route:
    its message and token ids are the policy's own chat's under the same seed. A chat stream sends
    none of the token-id flags SGLang's stream refuses, and a chat with media is refused where its
    draw would go on, as the completions route takes none."""
    completion, engine = sglang()
    reference, _ = sglang(own_eos=False)
    messages = [{"role": "user", "content": "hi"}]
    policy = Sampling(eos_token_ids=EOS)
    continued = 0
    for key in range(12):
        engine.requests.clear()
        (choice,) = completion.chat(messages, 8, key=key, sampling=policy, logprobs=True).choices
        (expected,) = reference.chat(messages, 8, key=key, sampling=policy, logprobs=True).choices
        assert choice.message.content == expected.message.content
        assert choice.model_extra["response_token_ids"] == expected.model_extra["response_token_ids"]
        assert [entry.logprob for entry in choice.logprobs.content] == [
            entry.logprob for entry in expected.logprobs.content]
        assert choice.finish_reason == expected.finish_reason
        continued += len(engine.requests) > 1
    assert continued >= 3
    for key in range(3):
        try:
            list(completion.chat(messages, 8, key=key, sampling=policy, stream=True))
        except ValueError:
            pass
    assert all("return_token_ids" not in body for path, body in engine.requests
               if path.endswith("/chat/completions") and body.get("stream"))
    pictured = [{"role": "user", "content": [{"type": "text", "text": "hi"},
                                             {"type": "image_url", "image_url": {"url": "data:,"}}]}]
    with pytest.raises(ValueError):
        for key in range(40):
            completion.chat(pictured, 8, key=key, sampling=policy)


def test_the_async_client_goes_on_as_the_sync_one_does():
    """`acall` and `achat` go on as `__call__` and `chat` do, and `astream` is checked as `stream` is."""
    async def run():
        completion, _ = sglang(asynchronous=True)
        try:
            done = [await completion.acall([[1]], 8, key=key, sampling=Sampling(eos_token_ids=EOS))
                    for key in range(6)]
            chats = [await completion.achat([{"role": "user", "content": "hi"}], 8, key=key,
                                            sampling=Sampling(eos_token_ids=EOS)) for key in range(6)]
            stream = await completion.astream([[1]], 8, key=0,
                                              sampling=Sampling(eos_token_ids=(NATIVE, EOS)))
            chunks = [chunk async for chunk in stream]
        finally:
            await completion.client.close()
        return done, chats, chunks

    done, chats, chunks = asyncio.run(run())
    completion, _ = sglang()
    assert [draw.tokens for draw in done] == [
        completion([[1]], 8, key=key, sampling=Sampling(eos_token_ids=EOS)).tokens for key in range(6)]
    assert [chat.choices[0].model_extra["response_token_ids"] for chat in chats] == [
        completion.chat([{"role": "user", "content": "hi"}], 8, key=key,
                        sampling=Sampling(eos_token_ids=EOS)).choices[0].model_extra["response_token_ids"]
        for key in range(6)]
    assert chunks and chunks[-1].choices[0].finish_reason in ("stop", "length")
