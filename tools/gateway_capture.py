"""Regenerate tests/fixtures/gateway: live SGLang and vLLM traces recorded by rllm-model-gateway.

    python tools/gateway_capture.py tests/fixtures/gateway --venvs /content/capture-venvs

Needs one GPU and `uv`. Installs SGLang 0.5.20 (with ninja), vLLM 0.30.0 and rllm-model-gateway at
3b40c37 in their own venvs under --venvs. For each engine, on 127.0.0.1: start the engine on
Qwen/Qwen2.5-0.5B-Instruct and wait for /health; warm it with one chat request (SGLang compiles for
minutes on its first one, stalling /health past the gateway's 5 s probe); start the gateway
(sync_traces, health_check_interval 60) and wait until `dew.objectives.rl.harbor.Gateway.ready`
passes; run one three-turn tool-calling session and one prompt past the context length through
/sessions/{sid}/v1; write the session's traces as `{engine}_session_traces.json` and the overflow's
as `{engine}_overflow_trace.json`, keeping only the fields `dew.objectives.rl.harbor.calls` reads.
The fixtures committed were captured on a Colab L4 on 2026-09-23.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import httpx

from dew.objectives.rl.harbor import Gateway

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
ENGINE, GATEWAY = "http://127.0.0.1:8011", "http://127.0.0.1:9090"
PACKAGES = {"sglang": ("sglang==0.5.20", "ninja"), "vllm": ("vllm==0.30.0",),
            "gateway": ("git+https://github.com/rllm-org/rllm@3b40c37cf6a262cf4d28cc987ebe4f4cf797956c"
                        "#subdirectory=rllm-model-gateway",)}
TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Run a bash command.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
KEPT = ("prompt_token_ids", "completion_token_ids", "logprobs", "finish_reason", "weight_version", "timestamp",
        "latency_ms")


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
                "--tool-call-parser", "qwen25"]
    return [str(bin / "python"), "-m", "vllm.entrypoints.openai.api_server", "--model", MODEL, "--host",
            "127.0.0.1", "--port", "8011", "--max-model-len", "4096", "--gpu-memory-utilization", "0.6",
            "--enable-auto-tool-choice", "--tool-call-parser", "hermes", "--enforce-eager"]


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


def capture(name: str, venvs: Path, out: Path) -> None:
    server = subprocess.Popen(engine(name, venv(venvs, name)), stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    config = venvs / "gateway.yaml"
    config.write_text(json.dumps({"host": "127.0.0.1", "port": 9090, "store_worker": "memory", "sync_traces": True,
                                  "health_check_interval": 60,
                                  "workers": [{"url": f"{ENGINE}/v1", "model_name": MODEL}]}))
    gateway = None
    try:
        deadline = time.monotonic() + 1200
        while not _answers(f"{ENGINE}/health"):
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"{name} did not start")
            time.sleep(3)
        httpx.post(f"{ENGINE}/v1/chat/completions", timeout=900,
                   json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
        gateway = subprocess.Popen([str(venv(venvs, "gateway") / "rllm-model-gateway"), "--config", str(config)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
        (out / f"{name}_session_traces.json").write_text(json.dumps([compact(trace) for trace in traces["session"]]))
        (out / f"{name}_overflow_trace.json").write_text(json.dumps(compact(traces["overflow"][0])))
    finally:
        for process in (gateway, server):
            if process is not None:
                process.terminate()
                process.wait(60)


def _answers(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--venvs", type=Path, default=Path("/tmp/gateway-capture"))
    args = parser.parse_args()
    args.venvs.mkdir(parents=True, exist_ok=True)
    for name in ("sglang", "vllm"):
        capture(name, args.venvs, args.out)


if __name__ == "__main__":
    main()
