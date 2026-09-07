"""Text completion from an engine outside Dew: Ollama, vLLM or any OpenAI-style server.

A client is the serving counterpart of `TextGeneration`: text in, text out,
one `Sampling` value for the controls the engine accepts. What the engine
does not report is absent from the result. A completion carries no
autoregressive likelihoods, so it cannot stand in for a native draw in a
policy-gradient rollout; it is for evaluation, data generation and
interactive use against a checkpoint Dew exported.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from urllib import error, request

from dew.sampling.text import Sampling


@dataclass(frozen=True)
class Completion:
    """One engine response per prompt, in prompt order."""

    texts: tuple[str, ...]
    finished: tuple[bool, ...]
    token_counts: tuple[int, ...]


def _controls(sampling: Sampling, max_new_tokens: int, engine: str) -> dict[str, object]:
    if sampling.pad_id != 0:
        raise ValueError(f"{engine} does not take a pad id; it returns text, not padded rows")
    if sampling.eos_id is not None:
        raise ValueError(f"{engine} stops at its own tokenizer's end tokens; eos_id cannot be sent")
    return {"temperature": sampling.temperature, "max_new_tokens": max_new_tokens,
            "top_k": sampling.top_k}


def _post(url: str, payload: dict[str, object], timeout: float) -> dict[str, object]:
    body = json.dumps(payload).encode()
    call = request.Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with request.urlopen(call, timeout=timeout) as response:
            decoded = json.loads(response.read())
    except error.HTTPError as failure:
        raise RuntimeError(f"{url} answered {failure.code}: {failure.read().decode(errors='replace')}") from failure
    if not isinstance(decoded, dict):
        raise RuntimeError(f"{url} did not answer with a JSON object")
    return decoded


@dataclass(frozen=True)
class OllamaCompletion:
    """`POST /api/generate` of an Ollama server, one call per prompt.

    Ollama takes `temperature` and `top_k` as options; `Sampling.eos_id` and
    `pad_id` have no counterpart and are refused. Each call sends its own
    seed so a batch is reproducible prompt by prompt.
    """

    model: str
    url: str = "http://127.0.0.1:11434"
    timeout: float = 600.0

    def __call__(self, prompts: str | Sequence[str], max_new_tokens: int, *,
                 seed: int, sampling: Sampling = Sampling()) -> Completion:
        controls = _controls(sampling, max_new_tokens, "Ollama")
        options: dict[str, object] = {"temperature": controls["temperature"],
                                      "num_predict": max_new_tokens, "seed": seed}
        if controls["top_k"] is not None:
            options["top_k"] = controls["top_k"]
        texts, finished, counts = [], [], []
        for index, prompt in enumerate([prompts] if isinstance(prompts, str) else prompts):
            answer = _post(f"{self.url}/api/generate", {
                "model": self.model, "prompt": prompt, "stream": False,
                "options": {**options, "seed": seed + index}}, self.timeout)
            texts.append(str(answer.get("response", "")))
            finished.append(answer.get("done_reason") == "stop")
            count = answer.get("eval_count", 0)
            counts.append(count if isinstance(count, int) else 0)
        return Completion(tuple(texts), tuple(finished), tuple(counts))


@dataclass(frozen=True)
class OpenAICompletion:
    """`POST /v1/completions` of vLLM or any OpenAI-compatible server.

    vLLM accepts `temperature`, `top_k` (its extension), `max_tokens` and a
    `seed`; `Sampling.eos_id` and `pad_id` are refused. Prompts go in one
    request; the server orders choices by prompt index.
    """

    model: str
    url: str = "http://127.0.0.1:8000"
    api_key: str | None = None
    timeout: float = 600.0

    def __call__(self, prompts: str | Sequence[str], max_new_tokens: int, *,
                 seed: int, sampling: Sampling = Sampling()) -> Completion:
        controls = _controls(sampling, max_new_tokens, "an OpenAI-compatible server")
        batch = [prompts] if isinstance(prompts, str) else list(prompts)
        payload: dict[str, object] = {"model": self.model, "prompt": batch, "max_tokens": max_new_tokens,
                                      "temperature": controls["temperature"], "seed": seed}
        if controls["top_k"] is not None:
            payload["top_k"] = controls["top_k"]
        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = f"Bearer {self.api_key}"
        call = request.Request(f"{self.url}/v1/completions", data=json.dumps(payload).encode(),
                               headers=headers, method="POST")
        try:
            with request.urlopen(call, timeout=self.timeout) as response:
                answer = json.loads(response.read())
        except error.HTTPError as failure:
            raise RuntimeError(f"{self.url} answered {failure.code}: "
                               f"{failure.read().decode(errors='replace')}") from failure
        choices = answer.get("choices") if isinstance(answer, dict) else None
        if not isinstance(choices, list) or len(choices) != len(batch):
            raise RuntimeError(f"{self.url} returned {0 if not isinstance(choices, list) else len(choices)} "
                               f"choices for {len(batch)} prompts")
        ordered = sorted(choices, key=lambda choice: int(choice.get("index", 0)))
        usage = answer.get("usage") if isinstance(answer, dict) else None
        total = usage.get("completion_tokens") if isinstance(usage, dict) else None
        counts = tuple([int(total)] if isinstance(total, int) and len(batch) == 1 else [0] * len(batch))
        return Completion(tuple(str(choice.get("text", "")) for choice in ordered),
                          tuple(choice.get("finish_reason") == "stop" for choice in ordered), counts)
