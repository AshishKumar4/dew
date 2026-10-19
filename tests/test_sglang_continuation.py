"""A policy's stop ids on SGLang, which also ends a draw on its model's own EOS ids
(`dew.inference.continuation`), through the official OpenAI SDK against a stand-in for SGLang
v0.5.20's OpenAI routes at the HTTP transport. The stand-in draws each token from what came before
it and ends a draw as SGLang's `Req._check_token_based_finish` does: on `stop_token_ids` and on its
model's own EOS, `NATIVE`, unless the request ignores every token stop."""

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


def scripted(sequence, rng):
    """A model that draws the same id at each place of the sequence: 3, NATIVE, 5, EOS, 6, 6, ..."""
    script = [3, NATIVE, 5, EOS]
    at = len(sequence) - 1
    return script[at] if at < len(script) else 6


PROBABILITIES = {0: 0.25, NATIVE: 0.5, EOS: 0.25}
"""A model that draws each id on its own, its own EOS first in half the draws."""


def drawn(sequence, rng):
    return int(rng.choice(list(PROBABILITIES), p=list(PROBABILITIES.values())))


class SGLang:
    """SGLang's `/v1/completions` and `/v1/chat/completions` over the model `draw`, which picks the
    next id from the prompt and the ids drawn so far, its random draws keyed by the request's seed.
    It reports `matched_stop`, `token_ids` and `prompt_token_ids` (chat: `response_token_ids`) as
    SGLang does, and every request it answered in `requests`."""

    def __init__(self, draw):
        self.draw, self.requests = draw, []

    def __call__(self, request):
        body = json.loads(request.content)
        self.requests.append((request.url.path, body))
        chat = request.url.path.endswith("/chat/completions")
        prompt = list(TEMPLATE) if chat else list(body["prompt"][0])
        rng = np.random.default_rng(body.get("seed"))
        stops = set(body.get("stop_token_ids") or ()) | (set() if body.get("ignore_eos") else {NATIVE})
        budget = body.get("max_completion_tokens" if chat else "max_tokens")
        choices = [self.choice(index, prompt, rng, stops, budget) for index in range(body.get("n", 1))]
        if body.get("stream"):
            return self.stream(choices, chat, body)
        generated = sum(len(choice["ids"]) for choice in choices)
        return httpx2.Response(200, json={
            "id": "sglang", "model": "tiny", "created": 1,
            "object": "chat.completion" if chat else "text_completion",
            "choices": [self.rendered(choice, chat, body) for choice in choices],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": generated,
                      "total_tokens": len(prompt) + generated}})

    def choice(self, index, prompt, rng, stops, budget):
        ids = []
        while len(ids) < budget:
            ids.append(self.draw([*prompt, *ids], rng))
            if ids[-1] in stops:
                return {"index": index, "ids": ids, "prompt": prompt, "reason": "stop", "matched": ids[-1]}
        return {"index": index, "ids": ids, "prompt": prompt, "reason": "length", "matched": None}

    @staticmethod
    def text(ids):
        return "".join(f"<{token}>" for token in ids if token not in SPECIAL)

    def rendered(self, choice, chat, body):
        extra = body.get("return_token_ids")
        if chat:
            entry = {"index": choice["index"], "finish_reason": choice["reason"],
                     "matched_stop": choice["matched"],
                     "message": {"role": "assistant", "content": self.text(choice["ids"])}}
            if body.get("return_prompt_token_ids"):
                entry["prompt_token_ids"] = choice["prompt"]
            if extra:
                entry["response_token_ids"] = choice["ids"]
            return entry
        entry = {"index": choice["index"], "text": self.text(choice["ids"]),
                 "finish_reason": choice["reason"], "matched_stop": choice["matched"]}
        if extra:
            entry.update(token_ids=choice["ids"], prompt_token_ids=choice["prompt"])
        if body.get("logprobs") is not None:
            tokens = [f"<{token}>" for token in choice["ids"]]
            offsets = np.cumsum([0, *(len(self.text([token])) for token in choice["ids"])])[:-1]
            entry["logprobs"] = {"tokens": tokens, "token_logprobs": [-0.5] * len(tokens),
                                 "top_logprobs": None, "text_offset": offsets.tolist()}
        return entry

    def stream(self, choices, chat, body):
        records = []
        for choice in choices:
            for at, token in enumerate(choice["ids"]):
                last = at == len(choice["ids"]) - 1
                reason, matched = (choice["reason"], choice["matched"]) if last else (None, None)
                if chat:
                    part = {"index": choice["index"], "delta": {"content": self.text([token])},
                            "finish_reason": reason, "matched_stop": matched}
                else:
                    part = {"index": choice["index"], "text": self.text([token]),
                            "finish_reason": reason, "matched_stop": matched}
                    if body.get("return_token_ids"):
                        part["token_ids"] = [token]
                        if at == 0:
                            part["prompt_token_ids"] = choice["prompt"]
                records.append({"id": "sglang", "model": "tiny", "created": 1,
                                "object": "chat.completion.chunk" if chat else "text_completion",
                                "choices": [part]})
        content = "".join(f"data: {json.dumps(record)}\n\n" for record in records) + "data: [DONE]\n\n"
        return httpx2.Response(200, content=content, headers={"Content-Type": "text/event-stream"})


def sglang(draw, asynchronous=False):
    engine = SGLang(draw)
    if asynchronous:
        client = openai.AsyncOpenAI(base_url="http://sglang/v1", api_key="fixture", max_retries=0,
                                   http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(engine)))
    else:
        client = openai.OpenAI(base_url="http://sglang/v1", api_key="fixture", max_retries=0,
                               http_client=httpx2.Client(transport=httpx2.MockTransport(engine)))
    return OpenAICompletion("tiny", client, provider="sglang"), engine


POLICY = Sampling(eos_token_ids=EOS)


def test_a_draw_sglang_ends_on_its_own_eos_goes_on_to_the_policys_stop():
    """SGLang ends the draw 3, NATIVE, 5, EOS on its own NATIVE, which the policy does not stop on, so
    the draw goes on from the prompt and the ids it drew, on the rest of its budget, and ends on
    the policy's EOS: ids, text, finish and usage are the whole draw's, and a policy whose stop SGLang
    ends on first takes one request."""
    completion, engine = sglang(scripted)
    done = completion([[1]], 6, key=0, sampling=POLICY, logprobs=0)
    assert done.tokens == ((3, NATIVE, 5, EOS),) and done.finish_reasons == ("stop",)
    assert done.texts == ("<3><5>",) and done.log_probs == ((-0.5,) * 4,)
    assert done.usage.completion_tokens == 4 and len(done.responses) == 2
    (_, first), (_, then) = engine.requests
    assert first["stop_token_ids"] == [EOS] and first["max_tokens"] == 6
    assert then["prompt"] == [[1, 3, NATIVE]] and then["max_tokens"] == 4 and then["n"] == 1
    assert then["stop_token_ids"] == [EOS] and then["seed"] != first["seed"]
    native = Sampling(eos_token_ids=(NATIVE, EOS))
    engine.requests.clear()
    assert completion([[1]], 6, key=0, sampling=native).tokens == ((3, NATIVE),) and len(engine.requests) == 1


def test_a_draw_that_ends_on_sglangs_own_eos_at_its_budget_is_cut_by_length():
    """The budget ends at the NATIVE SGLang stopped on, which the policy draws past: the draw is the
    budget's, ended by length, and nothing goes on."""
    completion, engine = sglang(scripted)
    done = completion([[1]], 2, key=0, sampling=POLICY)
    assert done.tokens == ((3, NATIVE),) and done.finish_reasons == ("length",) and len(engine.requests) == 1


def test_a_continued_draw_has_the_policys_distribution():
    """Each id is NATIVE with probability 1/2, EOS 1/4 and 0 1/4, on a budget of three: the policy
    ends a draw on EOS alone, so a draw is each id sequence up to the first EOS or the budget, with
    the product of its ids' probabilities. Over N draws each sequence's frequency is within five
    binomial standard deviations sqrt(p (1 - p) / N) of its probability, and every drawn sequence is
    one the policy draws. Drawing again from scratch when SGLang ends on NATIVE would weigh the
    sequences without NATIVE by more than their probability: [EOS] at about 0.42 for 0.25."""
    completion, _ = sglang(drawn)
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


def test_a_penalized_policy_is_refused_where_its_draw_needs_continuing():
    """A continuation reads the ids before it as prompt, which SGLang's repetition, frequency and
    presence penalties do not count, so a penalized policy whose draw SGLang ended on NATIVE is
    refused; one SGLang ends on the policy's own stop is drawn."""
    completion, _ = sglang(scripted)
    with pytest.raises(ValueError):
        completion([[1]], 6, key=0, sampling=Sampling(eos_token_ids=EOS, presence_penalty=0.5))
    done = completion([[1]], 6, key=0, sampling=Sampling(eos_token_ids=(NATIVE, EOS), presence_penalty=0.5))
    assert done.tokens == ((3, NATIVE),)


def test_a_stream_goes_on_past_sglangs_own_eos_under_the_choices_index():
    """A completion stream SGLang ended on NATIVE goes on in a stream of its own: its chunks follow
    under the choice's index, the chunk SGLang ended on carries no finish, and the last carries the
    policy's stop."""
    completion, engine = sglang(scripted)
    chunks = list(completion.stream([[1]], 6, key=0, sampling=POLICY))
    assert [choice.index for chunk in chunks for choice in chunk.choices] == [0] * 4
    assert [chunk.choices[0].model_extra["token_ids"] for chunk in chunks] == [[3], [NATIVE], [5], [EOS]]
    assert [chunk.choices[0].finish_reason for chunk in chunks] == [None, None, None, "stop"]
    assert "".join(chunk.choices[0].text for chunk in chunks) == "<3><5>"
    assert [body["prompt"] for _, body in engine.requests] == [[[1]], [[1, 3, NATIVE]]]


def test_a_chat_goes_on_past_sglangs_own_eos_and_a_chat_stream_is_refused_there():
    """A chat SGLang ended on NATIVE goes on from its rendered prompt on the completions route: the
    message holds the whole draw's text, its token ids the whole draw. A chat stream reports no ids
    to go on from, so it is refused where SGLang ends on NATIVE, after the chunks before it."""
    completion, engine = sglang(scripted)
    chat = completion.chat([{"role": "user", "content": "hi"}], 6, key=0, sampling=POLICY)
    (choice,) = chat.choices
    assert choice.message.content == "<5>" and choice.finish_reason == "stop"
    assert choice.model_extra["response_token_ids"] == [NATIVE, 5, EOS]
    (path, then) = engine.requests[1]
    assert path.endswith("/v1/completions") and then["prompt"] == [[*TEMPLATE, NATIVE]]
    stream = completion.chat([{"role": "user", "content": "hi"}], 6, key=0, sampling=POLICY, stream=True)
    with pytest.raises(ValueError):
        list(stream)


def test_the_async_client_goes_on_past_sglangs_own_eos_as_the_sync_one_does():
    """`acall`, `astream` and `achat` go on as `__call__`, `stream` and `chat` do."""
    async def run():
        completion, _ = sglang(scripted, asynchronous=True)
        try:
            done = await completion.acall([[1]], 6, key=0, sampling=POLICY)
            chunks = [chunk async for chunk in await completion.astream([[1]], 6, key=0, sampling=POLICY)]
            chat = await completion.achat([{"role": "user", "content": "hi"}], 6, key=0, sampling=POLICY)
        finally:
            await completion.client.close()
        return done, chunks, chat

    done, chunks, chat = asyncio.run(run())
    assert done.tokens == ((3, NATIVE, 5, EOS),)
    assert [chunk.choices[0].model_extra["token_ids"] for chunk in chunks] == [[3], [NATIVE], [5], [EOS]]
    assert chat.choices[0].model_extra["response_token_ids"] == [NATIVE, 5, EOS]
