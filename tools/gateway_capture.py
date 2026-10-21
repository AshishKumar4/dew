"""Regenerate tests/fixtures/gateway: live SGLang and vLLM traces recorded by rllm-model-gateway, and
the draws Dew's rollout servers make on each engine under its own stops.

    python tools/gateway_capture.py tests/fixtures/gateway --venvs /content/capture-venvs
    python tools/gateway_capture.py tests/fixtures/gateway --only stops --python vllm=PATH --python sglang=PATH

Needs one GPU and `uv`. Installs SGLang 0.5.20 (with ninja), vLLM 0.30.0 and rllm-model-gateway at
3b40c37 in their own venvs under --venvs, or runs each engine with the interpreter `--python` names.
For each engine, on 127.0.0.1: start the engine on Qwen/Qwen2.5-0.5B-Instruct and wait for /health;
warm it with one chat request (SGLang compiles for minutes on its first one, stalling /health past
the gateway's 5 s probe).

`gateway`: start the gateway (sync_traces, health_check_interval 60) and wait until
`dew.interop.gateway.Gateway.ready` passes; run one three-turn tool-calling session and one prompt
past the context length through /sessions/{sid}/v1; write the session's traces as
`{engine}_session_traces.json` and the overflow's as `{engine}_overflow_trace.json`, keeping only
the fields `dew.interop.gateway.calls` reads. The fixtures committed were captured on a Colab L4 on
2026-09-23.

`stops`: draw from the engine through Dew's `OpenAIRolloutServer`, behind a proxy that records each
request Dew sends and the engine's answer, for a chat prompt the model answers in a few tokens and
ends on its own EOS: under a policy with no stop, one that stops at the model's EOS, and one that
stops at a token the model draws after its EOS (the first such seed's). Write them as
`{engine}_stops.json` with the model's EOS ids and what each draw gave: its ids and whether it
ended on a stop, or the error Dew raised. It also records the engine's raw answer to a draw from
the same seed that stops at a string its answer reaches (`stop_strings`), Dew's own draws from that
seed under a policy with that stop string, with the model's EOS and without an EOS
(`string_draws`), and on SGLang, which Dew
streams, how many ids the engine generated for a draw under a long budget that Dew's server cut at
an early stop (`release`).
"""

from __future__ import annotations

import os

# The engines hold the GPU; the capture's own JAX draws keys alone, on the host.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import contextlib
import json
import subprocess
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from dew.interop.gateway import Gateway

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ENGINE, GATEWAY, PROXY = "http://127.0.0.1:8011", "http://127.0.0.1:9090", ("127.0.0.1", 8012)
PACKAGES = {"sglang": ("sglang==0.5.20", "ninja"), "vllm": ("vllm==0.30.0",),
            "gateway": ("git+https://github.com/rllm-org/rllm@3b40c37cf6a262cf4d28cc987ebe4f4cf797956c"
                        "#subdirectory=rllm-model-gateway",)}
TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Run a bash command.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
KEPT = ("prompt_token_ids", "completion_token_ids", "logprobs", "finish_reason", "weight_version", "timestamp",
        "latency_ms", "messages", "response_message")
RETURN_IDS = {"vllm": {"return_tokens_as_token_ids": True, "return_token_ids": True},
              "sglang": {"return_token_ids": True}}
"""The fields that make each engine's completions route answer with its sampled ids."""
STOPS_BUDGET = 48
"""Tokens a stops draw may take: the answer, its EOS and what the model draws after it."""
RELEASE_BUDGET = 2048
"""The budget of the streamed draw whose early stop shows whether closing a stream ends SGLang's request."""


def venv(root: Path, name: str) -> Path:
    path = root / name
    if not (path / "bin/python").exists():
        subprocess.run(["uv", "venv", str(path), "--python", "3.12", "-q"], check=True)
        subprocess.run(["uv", "pip", "install", "-q", "--python", str(path / "bin/python"), *PACKAGES[name]],
                       check=True)
    return path / "bin"


def engine(name: str, bin: Path) -> list[str]:
    if name == "sglang":
        return [str(bin / "python"), "-m", "sglang.launch_server", "--model-path", MODEL, "--host", "127.0.0.1",
                "--port", "8011", "--context-length", "4096", "--mem-fraction-static", "0.6",
                "--tool-call-parser", "qwen25", "--enable-metrics"]
    return [str(bin / "python"), "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--host",
            "127.0.0.1", "--port", "8011", "--max-model-len", "4096", "--gpu-memory-utilization", "0.6",
            "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--enforce-eager"]


@contextlib.contextmanager
def served(name: str, bin: Path, out: Path) -> Iterator[None]:
    """The engine `name` started from `bin`, healthy and warm, stopped after; its output goes to
    `{name}_engine.log` in `out`."""
    log = (out / f"{name}_engine.log").open("w")
    # The engine's own environment first on PATH: it runs build tools (SGLang's ninja) from there.
    server = subprocess.Popen(engine(name, bin), stdout=log, stderr=subprocess.STDOUT,
                              env={**os.environ, "PATH": os.pathsep.join((str(bin), os.environ["PATH"]))})
    try:
        deadline = time.monotonic() + 1200
        while not _answers(f"{ENGINE}/health"):
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"{name} did not start")
            time.sleep(3)
        httpx.post(f"{ENGINE}/v1/chat/completions", timeout=900,
                   json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
        yield
    finally:
        server.terminate()
        server.wait(60)
        log.close()


def compact(trace: dict) -> dict:
    """The fields `calls` reads: the gateway's extraction, SGLang's choice ids and likelihoods, error bodies."""
    raw = trace.get("raw_response") or {}
    kept_raw = {key: raw[key] for key in ("object", "message", "error") if key in raw}
    if raw.get("choices"):
        choice = raw["choices"][0]
        kept_raw["choices"] = [{"prompt_token_ids": choice.get("prompt_token_ids"),
                                "response_token_ids": choice.get("response_token_ids"),
                                "logprobs": {"content": [{"logprob": entry["logprob"]}
                                                         for entry in choice["logprobs"]["content"]]}}]
    return {**{key: trace.get(key) for key in KEPT}, "raw_response": kept_raw}


def chat(sid: str, messages: list, **extra) -> httpx.Response:
    return httpx.post(f"{GATEWAY}/sessions/{sid}/v1/chat/completions", timeout=900,
                      json={"model": MODEL, "messages": messages, **extra})


def capture_gateway(name: str, bin: Path, gateway_bin: Path, venvs: Path, out: Path) -> None:
    config = venvs / "gateway.yaml"
    config.write_text(json.dumps({"host": "127.0.0.1", "port": 9090, "store_worker": "memory", "sync_traces": True,
                                  "health_check_interval": 60,
                                  "workers": [{"url": f"{ENGINE}/v1", "model_name": MODEL}]}))
    with served(name, bin, out):
        gateway = subprocess.Popen([str(gateway_bin / "rllm-model-gateway"), "--config", str(config)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            Gateway(GATEWAY).ready(300)
            messages = [{"role": "system", "content": "You solve tasks with the bash tool, one command per turn."},
                        {"role": "user", "content": "Create hello.txt containing 'Hello, world!', then check it."}]
            for turn in range(3):
                message = chat(f"{name}-session", messages, tools=TOOLS, max_tokens=256, temperature=1.0).json()[
                    "choices"][0]["message"]
                messages.append({key: message[key] for key in ("role", "content", "tool_calls") if message.get(key)})
                messages.extend({"role": "tool", "tool_call_id": call["id"], "content": f"(turn {turn}) ok"}
                                for call in message.get("tool_calls") or [])
                if not message.get("tool_calls"):
                    messages.append({"role": "user", "content": "Use the bash tool."})
            chat(f"{name}-overflow", [{"role": "user", "content": "word " * 8000}], max_tokens=16)
            traces = {sid: httpx.get(f"{GATEWAY}/sessions/{name}-{sid}/traces", timeout=60).json()
                      for sid in ("session", "overflow")}
            (out / f"{name}_session_traces.json").write_text(
                json.dumps([compact(trace) for trace in traces["session"]]))
            (out / f"{name}_overflow_trace.json").write_text(json.dumps(compact(traces["overflow"][0])))
        finally:
            gateway.terminate()
            gateway.wait(60)


class _Recorded(BaseHTTPRequestHandler):
    """Forwards each request to the engine and logs its path, body and the engine's answer in `exchanges`."""

    def _forward(self, method: str) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        answer = httpx.request(method, ENGINE + self.path, content=raw, timeout=900,
                               headers={"Content-Type": self.headers.get("Content-Type", "application/json")})
        kind = answer.headers.get("Content-Type", "application/json")
        # A streamed answer is kept as the server-sent events it was, a whole one as its JSON.
        streamed = kind.startswith("text/event-stream")
        self.server.exchanges.append({"path": self.path, "request": json.loads(raw) if raw else None,
                                      "status": answer.status_code,
                                      **({"stream": answer.text} if streamed else {"response": answer.json()})})
        self.send_response(answer.status_code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(answer.content)))
        self.end_headers()
        self.wfile.write(answer.content)

    def do_GET(self):
        self._forward("GET")

    def do_POST(self):
        self._forward("POST")

    def log_message(self, *args):
        pass


def capture_stops(name: str, bin: Path, out: Path) -> None:
    import openai
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    from dew.inference import OpenAICompletion, OpenAIRolloutServer
    from dew.sampling import Sampling

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # The chat template's ids as a list: Transformers 5 answers a BatchEncoding unless told not to.
    prompt = list(tokenizer.apply_chat_template([{"role": "user", "content": "Say hello."}],
                                                add_generation_prompt=True, tokenize=True, return_dict=False))
    generation = json.loads(Path(hf_hub_download(MODEL, "generation_config.json")).read_text())
    listed = generation["eos_token_id"]
    native = sorted({*(listed if isinstance(listed, list) else [listed]), tokenizer.eos_token_id})

    def published(variables, version):
        raise AssertionError("no weights are pushed while capturing")

    proxy = ThreadingHTTPServer(PROXY, _Recorded)
    proxy.exchanges = []
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    client = openai.OpenAI(base_url=f"http://{PROXY[0]}:{PROXY[1]}/v1", api_key="capture", max_retries=0)
    completion = OpenAICompletion(MODEL, client, provider=name)

    def draw(eos: int | None, seed: int, stop: str | None = None) -> dict:
        proxy.exchanges.clear()
        sampling = Sampling(temperature=1.0, eos_token_ids=eos, stop=() if stop is None else (stop,))
        server = OpenAIRolloutServer(completion, sampling, published, decode=tokenizer.decode)
        try:
            drawn = server.submit(prompt, STOPS_BUDGET, key=seed).result()
            outcome = {"tokens": list(drawn.tokens), "terminated": drawn.terminated}
        except ValueError as error:
            outcome = {"refused": f"{type(error).__name__}: {error}"}
        finally:
            server.close()
        return {"eos": eos, "seed": seed, **({} if stop is None else {"stop": stop}),
                "exchanges": list(proxy.exchanges), **outcome}

    def raw(seed: int, stop: str) -> dict:
        """The engine's own answer, through the proxy, to a draw from the same seed that stops at the
        string `stop`: which ids it returns and how it says the string matched."""
        proxy.exchanges.clear()
        httpx.post(f"http://{PROXY[0]}:{PROXY[1]}/v1/completions", timeout=900, json={
            "model": MODEL, "prompt": [list(prompt)], "max_tokens": STOPS_BUDGET, "temperature": 1.0,
            "seed": seed, "logprobs": 0, "stop": [stop], **RETURN_IDS[name]})
        return {"stop": stop, "seed": seed, "exchanges": list(proxy.exchanges)}

    def released(seed: int, stop: int) -> dict:
        """A streamed draw straight to SGLang under a long budget, stopped at `stop`: how many ids the
        engine generated for it, which says whether closing the stream at the stop ended the request."""
        direct = OpenAICompletion(MODEL, openai.OpenAI(base_url=f"{ENGINE}/v1", api_key="capture", max_retries=0),
                                  provider=name)
        server = OpenAIRolloutServer(direct, Sampling(temperature=1.0, eos_token_ids=stop), published)
        # The tokenizer counts a request's generated tokens as it collects the request, finished or
        # aborted, which is when it counts the request itself (sglang/srt/managers/tokenizer_manager.py
        # at v0.5.20), so the draw's tokens are counted once the request count moves past its baseline.
        requests, before = _metric("sglang:num_requests_total"), _metric("sglang:generation_tokens_total")
        try:
            drawn = server.submit(prompt, RELEASE_BUDGET, key=seed).result()
        finally:
            server.close()
        deadline = time.monotonic() + 120
        while _metric("sglang:num_requests_total") <= requests:
            if time.monotonic() > deadline:
                raise RuntimeError(f"{name} counted no finished request 120 s after the released draw closed")
            time.sleep(0.5)
        return {"budget": RELEASE_BUDGET, "stop": stop, "seed": seed, "drawn": len(drawn.tokens),
                "terminated": drawn.terminated, "generated": _metric("sglang:generation_tokens_total") - before}

    draws: list[dict] = []
    stopped: list[dict] = []
    strings: list[dict] = []
    release: dict | None = None
    with served(name, bin, out):
        try:
            # The first seed whose unstopped draw ends its answer on the model's EOS and goes on to a
            # token that is no EOS, which the third policy stops at.
            for seed in range(16):
                unstopped = draw(None, seed)
                if "tokens" not in unstopped:
                    draws.append(unstopped)
                    raise RuntimeError(f"{name} refused a draw without stops: {unstopped['refused']}")
                after = next((index for index, token in enumerate(unstopped["tokens"][:-1]) if token in native),
                             None)
                later = [] if after is None else [token for token in unstopped["tokens"][after + 1:]
                                                  if token not in native]
                if later:
                    break
            else:
                raise RuntimeError(f"{name} drew no token past its EOS in 16 seeds")
            draws = [unstopped, draw(native[0], seed), draw(later[0], seed)]
            # A stop string the same seed's answer reaches a few tokens in.
            pieces = [tokenizer.decode([token]).strip() for token in unstopped["tokens"][2:after]]
            stop = next((piece for piece in pieces if piece), None)
            if stop is not None:
                stopped.append(raw(seed, stop))
                strings.extend((draw(native[0], seed, stop), draw(None, seed, stop)))
            if name == "sglang":
                release = released(seed, later[0])
        finally:
            proxy.shutdown()
            (out / f"{name}_stops.json").write_text(json.dumps(
                {"engine": name, "model": MODEL, "native_eos": native, "prompt": prompt, "budget": STOPS_BUDGET,
                 "draws": draws, "stop_strings": stopped, "string_draws": strings, "release": release},
                indent=1))


def _metric(name: str) -> float:
    """The engine's Prometheus series `name`, summed over its labels; an engine that exports no such
    series raises, rather than reading as zero."""
    response = httpx.get(f"{ENGINE}/metrics", timeout=30)
    response.raise_for_status()
    values = [float(line.rsplit(" ", 1)[1]) for line in response.text.splitlines()
              if line.startswith((f"{name}{{", f"{name} "))]
    if not values:
        raise RuntimeError(f"the engine exports no {name} series")
    return sum(values)


def _answers(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--venvs", type=Path, default=Path("/tmp/gateway-capture"))
    parser.add_argument("--only", choices=("gateway", "stops"), action="append",
                        help="the captures to run (default: both)")
    parser.add_argument("--python", action="append", default=[], metavar="ENGINE=PATH",
                        help="run ENGINE (vllm or sglang) with this interpreter instead of a venv under --venvs")
    args = parser.parse_args()
    args.venvs.mkdir(parents=True, exist_ok=True)
    given = dict(entry.split("=", 1) for entry in args.python)
    if not set(given) <= {"vllm", "sglang"}:
        parser.error("--python names vllm or sglang")
    captures = args.only or ["gateway", "stops"]
    for name in ("sglang", "vllm"):
        bin = Path(given[name]).parent if name in given else venv(args.venvs, name)
        if "gateway" in captures:
            capture_gateway(name, bin, venv(args.venvs, "gateway"), args.venvs, args.out)
        if "stops" in captures:
            capture_stops(name, bin, args.out)


if __name__ == "__main__":
    main()
