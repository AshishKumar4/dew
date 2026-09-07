"""Official SDK requests/responses with mocked network I/O, not a Dew server."""

import asyncio
import json

import pytest

ollama = pytest.importorskip("ollama", reason="optional inference-clients extra")
openai = pytest.importorskip("openai", reason="optional inference-clients extra")
import httpx
import httpx2

from dew.inference import OllamaCompletion, OpenAICompletion, Usage
from dew.sampling import Sampling


def response(choices, **fields):
    return {"id": "fixture", "model": "tiny", "created": 1, "object": "text_completion",
            "choices": choices, **fields}


def choice(index, text, reason=None, **fields):
    return {"index": index, "text": text, "finish_reason": reason, **fields}


@pytest.fixture
def clients():
    owned = []

    def make(kind, handler):
        calls = []
        module = httpx if kind == "ollama" else httpx2

        def network(request):
            body = json.loads(request.content)
            calls.append(body)
            return handler(module, request, body)

        transport = module.MockTransport(network)
        if kind == "ollama":
            client = ollama.Client(host="http://fixture", transport=transport)
            adapter = OllamaCompletion("tiny", client)
        else:
            client = openai.OpenAI(base_url="http://fixture/v1", api_key="fixture", max_retries=0,
                                   http_client=module.Client(transport=transport))
            adapter = OpenAICompletion("tiny", client)
        owned.append(client)
        return adapter, calls

    yield make
    for client in owned:
        client.close()


def test_requested_sampling_overrides_hidden_ollama_penalties(clients):
    task, calls = clients("ollama", lambda http, request, body: http.Response(200, json={"response": "ok"}))
    task("prompt", 5, sampling=Sampling(temperature=0), options={"num_gpu": 0})
    options = calls[0]["options"]
    assert options["repeat_penalty"] == 1.0 and options["top_k"] == 0
    assert options["top_p"] == 1.0 and options["min_p"] == 0.0
    with pytest.raises(ValueError, match="conflict"):
        task("prompt", 5, sampling=Sampling(), options={"repeat_penalty": 1.1})
    with pytest.raises(ValueError, match="request fields"):
        task("prompt", 5, options={"logprobs": True})
    assert len(calls) == 1


def test_vllm_sampling_controls_are_explicit_and_generic_openai_is_not_guessed(clients):
    task, calls = clients("openai", lambda http, request, body: http.Response(200, json=response([choice(0, "ok")])))
    from dataclasses import replace
    sampling = Sampling(temperature=0.7, top_k=4, top_p=0.8, min_p=0.1, eos_id=2)
    with pytest.raises(ValueError, match="vllm"):
        task("prompt", 5, sampling=sampling)
    vllm = replace(task, provider="vllm")
    vllm("prompt", 5, sampling=sampling)
    assert calls[0]["repetition_penalty"] == 1.0
    assert calls[0]["stop_token_ids"] == [2]
    assert calls[0]["top_k"] == 4 and calls[0]["min_p"] == 0.1
    # The SDK writes extra_body over the named parameters, so a policy field
    # hidden there would reach the backend after the named checks passed.
    policy = Sampling(temperature=0, top_p=0.5)
    for hidden in ({"temperature": 1.5}, {"top_p": 1.0}, {"presence_penalty": 2}, {"repetition_penalty": 1.3}):
        with pytest.raises(ValueError, match="conflict"):
            vllm("prompt", 5, sampling=policy, extra_body=hidden)
        with pytest.raises(ValueError, match="conflict"):
            vllm.chat([{"role": "user", "content": "hi"}], 5, sampling=policy, extra_body=hidden)
    with pytest.raises(ValueError, match="vllm"):
        task("prompt", 5, sampling=policy, extra_body={"top_k": 5})
    assert len(calls) == 1
    vllm("prompt", 5, sampling=policy, extra_body={"temperature": 0.0, "guided_regex": "[a-z]+"})
    assert calls[1]["temperature"] == 0.0 and calls[1]["top_p"] == 0.5
    assert calls[1]["presence_penalty"] == 0.0 and calls[1]["guided_regex"] == "[a-z]+"


def test_ollama_retains_sdk_metadata_and_full_options(clients):
    def network(http, request, body):
        return http.Response(200, json={"response": body["prompt"].upper(), "done_reason": "length",
                                        "context": [4, 5], "logprobs": [{"token": "A", "logprob": -0.7}]})
    task, calls = clients("ollama", network)
    result = task(["a", "b"], 7, seed=3,
                  options={"top_p": 0.6, "min_p": 0.1, "repeat_penalty": 1.5},
                  raw=True, system="system", format={"type": "object"},
                  images=[b"image bytes"], logprobs=True, think=True)
    assert result.texts == ("A", "B")
    assert result.finish_reasons == ("length", "length")
    assert result.token_counts == (None, None) and result.usage is None
    assert result.responses[0].context == [4, 5]
    assert result.responses[0].logprobs[0].logprob == -0.7
    assert [item["options"]["seed"] for item in calls] == [3, 4]
    assert calls[0]["options"] == {"top_p": 0.6, "min_p": 0.1, "repeat_penalty": 1.5,
                                    "num_predict": 7, "seed": 3}
    assert calls[0]["images"] == ["aW1hZ2UgYnl0ZXM="] and calls[0]["format"] == {"type": "object"}


@pytest.mark.parametrize("answer", [
    {}, {"response": None}, {"response": {}}, {"response": "", "eval_count": -1},
    {"response": "", "eval_count": True}, {"response": "", "eval_count": 1.5},
    {"response": "", "eval_count": "3"},
])
def test_ollama_rejects_bad_wire_fields_before_sdk_coercion(clients, answer):
    task, _ = clients("ollama", lambda http, request, body: http.Response(200, json=answer))
    with pytest.raises(ValueError):
        task("a", 3)


def test_genuine_empty_text_zero_usage_and_absent_reason_remain_distinct(clients):
    task, _ = clients("ollama", lambda http, request, body: http.Response(200, json={"response": "", "eval_count": 0}))
    result = task("a", 0)
    assert result.texts == ("",) and result.token_counts == (0,) and result.finish_reasons == (None,)


def test_openai_keeps_aggregate_usage_separate_and_associates_choices(clients):
    def network(http, request, body):
        return http.Response(200, json=response([
            choice(1, "second", "length", token_ids=[2]), choice(0, "first", "stop", token_ids=[1])],
            usage={"prompt_tokens": 3, "completion_tokens": 7, "total_tokens": 10}))
    task, calls = clients("openai", network)
    result = task(["a", "b"], 8, seed=9, top_p=0.7, logprobs=2,
                  extra_body={"top_k": 3, "min_p": 0.1, "return_tokens_as_token_ids": True})
    assert result.texts == ("first", "second") and result.finish_reasons == ("stop", "length")
    assert result.token_counts == (None, None)
    assert result.usage == Usage(3, 7, 10)
    assert result.responses[0].choices[0].model_extra["token_ids"] == [2]
    assert calls[0]["top_k"] == 3 and calls[0]["min_p"] == 0.1
    assert calls[0]["top_p"] == 0.7 and calls[0]["logprobs"] == 2


@pytest.mark.parametrize("choices", [
    [choice(0, "a"), choice(0, "b")], [choice(0.9, "a"), choice(0.1, "b")],
    [choice(True, "a"), choice(0, "b")], [choice(2, "a"), choice(0, "b")],
    [{"text": "a"}, choice(1, "b")], [choice(0, None), choice(1, "b")], [{}, {}],
])
def test_openai_refuses_ambiguous_or_malformed_prompt_associations(clients, choices):
    task, _ = clients("openai", lambda http, request, body: http.Response(200, json=response(choices)))
    with pytest.raises(ValueError):
        task(["a", "b"], 4)


@pytest.mark.parametrize("count", [-1, True, 1.2, "4"])
def test_openai_refuses_coerced_or_negative_usage(clients, count):
    task, _ = clients("openai", lambda http, request, body: http.Response(200, json=response(
        [choice(0, "ok")], usage={"completion_tokens": count})))
    with pytest.raises(ValueError):
        task("a", 4)


def test_multiple_choices_and_absent_usage_are_preserved(clients):
    task, _ = clients("openai", lambda http, request, body: http.Response(200, json=response(
        [choice(1, "second"), choice(0, "first")], model_extension="kept")))
    result = task("prompt", 5, n=2)
    assert result.texts == ("first", "second")
    assert result.finish_reasons == (None, None) and result.usage is None
    assert result.token_counts == (None, None)
    assert result.responses[0].model_extra["model_extension"] == "kept"


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_bad_budgets_fail_before_network_and_sdk_errors_propagate(clients, kind):
    task, calls = clients(kind, lambda http, request, body: http.Response(404, json={"error": "missing model"}))
    for budget in (-1, True, 1.5):
        with pytest.raises(ValueError):
            task("prompt", budget)
    assert calls == []
    if kind == "openai":
        with pytest.raises(ValueError):
            task("prompt", 1, extra_body={"max_tokens": -1})
        assert calls == []
    error = ollama.ResponseError if kind == "ollama" else openai.NotFoundError
    with pytest.raises(error):
        task("prompt", 1)


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_native_streams_and_chat_keep_tools_structured_outputs_and_media(clients, kind):
    def network(http, request, body):
        if request.url.path.endswith("chat") or request.url.path.endswith("chat/completions"):
            message = {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call0", "type": "function", "function": {"name": "verify", "arguments": "{}"}}]}
            if kind == "ollama":
                message["tool_calls"][0]["function"]["arguments"] = {}
                return http.Response(200, json={"message": message, "done": True, "done_reason": "stop"})
            return http.Response(200, json={"id": "chat0", "model": "tiny", "object": "chat.completion",
                                            "created": 1, "choices": [{"index": 0, "message": message,
                                                                       "finish_reason": "tool_calls"}]})
        if kind == "ollama":
            return http.Response(200, content='{"response":"part","done":false}\n{"response":"","done":true,"eval_count":1}\n')
        records = [response([choice(0, "part")]), response([choice(0, "", "stop")])]
        return http.Response(200, headers={"Content-Type": "text/event-stream"},
                             content="".join(f"data: {json.dumps(item)}\n\n" for item in records) + "data: [DONE]\n\n")
    task, calls = clients(kind, network)
    streamed = list(task.stream("prompt", 5, seed=7))
    if kind == "ollama":
        assert [chunk.response for chunk in streamed] == ["part", ""]
        options = {"format": {"type": "object"}}
        messages = [{"role": "tool", "content": "verified", "tool_name": "verify", "images": ["aW1hZ2U="]}]
    else:
        assert [chunk.choices[0].text for chunk in streamed] == ["part", ""]
        options = {"response_format": {"type": "json_object"}, "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
        messages = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,aW1hZ2U="}}]}]
    tool = {"type": "function", "function": {"name": "verify", "parameters": {"type": "object"}}}
    chat = task.chat(messages, 9, tools=[tool], **options)
    message = chat.message if kind == "ollama" else chat.choices[0].message
    assert message.tool_calls[0].function.name == "verify"
    assert calls[-1]["tools"][0]["function"]["name"] == "verify"
    assert calls[-1]["messages"][0]["content"] == messages[0]["content"]


@pytest.mark.parametrize("kind", ["ollama", "openai"])
def test_async_sdk_completion_and_streaming(kind):
    async def run():
        http = httpx if kind == "ollama" else httpx2

        def network(request):
            body = json.loads(request.content)
            if body["stream"]:
                content = '{"response":"async","done":true}\n' if kind == "ollama" else (
                    f"data: {json.dumps(response([choice(0, 'async', 'stop')]))}\n\ndata: [DONE]\n\n")
                return http.Response(200, content=content, headers={"Content-Type": "text/event-stream"})
            answer = {"response": "async", "eval_count": 1} if kind == "ollama" else response(
                [choice(0, "async", "stop")], usage={"completion_tokens": 1})
            return http.Response(200, json=answer)

        transport = http.MockTransport(network)
        if kind == "ollama":
            client = ollama.AsyncClient(host="http://fixture", transport=transport)
            task = OllamaCompletion("tiny", client)
        else:
            client = openai.AsyncOpenAI(base_url="http://fixture/v1", api_key="fixture", max_retries=0,
                                       http_client=http.AsyncClient(transport=transport))
            task = OpenAICompletion("tiny", client)
        try:
            result = await task.acall("prompt", 5)
            assert result.texts == ("async",) and result.token_counts == (1,)
            stream = await task.astream("prompt", 5)
            chunks = [part async for part in stream]
            assert len(chunks) == 1
            assert (chunks[0].response if kind == "ollama" else chunks[0].choices[0].text) == "async"
        finally:
            await client.close()
    asyncio.run(run())
