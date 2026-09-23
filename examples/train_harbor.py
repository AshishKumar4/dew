"""Agentic GRPO: a harness runs in Harbor sandboxes, a gateway records its model calls, and Dew trains on them.

    python examples/train_harbor.py --tasks path/to/harbor/task ... --gateway http://127.0.0.1:9090 \
        --sandbox-gateway http://proxy:9091 --engines http://127.0.0.1:8011 --served runs/harbor/served
    JAX_PLATFORMS=cpu python examples/train_harbor.py --smoke --out /tmp/harbor-smoke

Each update draws `--prompts` Harbor tasks and runs `--groups` trials of
each through `HarborSource`: Harbor starts the task's sandbox and the
harness (mini-swe-agent by default), the harness's model calls go through
rllm-model-gateway to the engines, and the gateway's traces become the
session's calls. `RolloutScheduler` admits complete groups at most
`--max-lag` updates stale and packs them; the GRPO objective trains on the
sampled ids only. `Publication` pushes each version to every engine with
`SafetensorsReload` and then stamps the gateway, so every call carries the
version it was sampled under.

The example writes the policy to `--served` and then waits, up to
`--ready-timeout` seconds, for the gateway to route to an engine:
launch the engines on `--served` (vLLM with `VLLM_SERVER_DEV_MODE=1`) and
the gateway in front of them while it waits, as
docs/concepts/post_training.md describes (warm each engine, size the
gateway's health interval, keep its admin routes away from the sandboxes).

`--smoke` needs neither Harbor nor an engine. A stand-in `harbor` script
runs a two-turn harness against an in-process gateway whose engine is
Dew's own server on the committed tiny Qwen2, so the ids and likelihoods
it records are the policy's own draws. The reward is the share of vowels
in the replies. Two updates on CPU.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jax
import numpy as np
import optax
import tyro

from dew.data import tokenizer_for
from dew.data.dataset import Dataset
from dew.inference import NativeRolloutServer, Publication, SafetensorsReload, Server, TextGeneration
from dew.interop import load_pretrained
from dew.objectives.rl import GRPOObjective, RolloutScheduler, SchedulerRecord
from dew.objectives.rl.harbor import HARBOR_KEY, Gateway, HarborSource
from dew.objectives.rl.scheduler import task_ids
from dew.objectives.rl.sessions import Task
from dew.sampling import Sampling
from dew.training import Trainer

FIXTURES = Path(__file__).resolve().parents[1] / "tests/fixtures/hf"


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tasks: tuple[Path, ...] = ()
    """Harbor task directories; each update samples `prompts` of them."""
    gateway: str = "http://127.0.0.1:9090"
    """The gateway's root as this process reaches it."""
    sandbox_gateway: str | None = None
    """Where sandboxes reach the gateway: a proxy that forwards only session chat calls."""
    engines: tuple[str, ...] = ("http://127.0.0.1:8011",)
    engine: str = "vllm"
    served: Path = Path("runs/harbor/served")
    harbor: str = "harbor"
    agent: str = "mini-swe-agent"
    harness_model: str = "openai/policy"
    """The harness's --model: its provider prefix and the name the engines serve."""
    harbor_arguments: tuple[str, ...] = ("-e", "docker")
    out: Path = Path("runs/harbor")
    steps: int = 20
    prompts: int = 4
    groups: int = 4
    width: int = 16384
    learning_rate: float = 1e-6
    max_lag: int = 1
    truncation: str = "mask"
    ready_timeout: float = 900.0
    """Seconds to wait for the gateway to route to an engine."""
    smoke: bool = False
    seed: int = 0
    workers: int = 16


def main(config: Config) -> dict:
    if config.smoke:
        config = replace(config, model=str(FIXTURES / "qwen2-tiny"), steps=2, prompts=2, groups=2, width=128,
                         truncation="score", workers=4)
    config.out.mkdir(parents=True, exist_ok=True)
    source = load_pretrained(config.model, dtype="float32" if config.smoke else "bfloat16", param_dtype="float32",
                             max_seq_len=config.width)
    objective = GRPOObjective(source.model, config.width - 1, pretrained=source.variables,
                              behavior_importance=2.0, epsilon_high=0.28)
    fake = None
    if config.smoke:
        fake, harbor, tasks = smoke_setup(config, source)
        gateway = Gateway(f"http://127.0.0.1:{fake.server_port}")
        push = fake.policy.load
    else:
        harbor, tasks = config.harbor, config.tasks
        gateway = Gateway(config.gateway, sandbox_url=config.sandbox_gateway)
        push = SafetensorsReload(source, config.served, config.engines, config.engine)
        push.write(source.variables)
        # The engines launch on --served now; the Publication below stamps the gateway at once.
        gateway.ready(config.ready_timeout)
    trials = HarborSource(gateway, harbor=harbor, model=config.harness_model, trials=config.out / "trials",
                          agent=config.agent, environment={"OPENAI_API_KEY": "none", "MSWEA_API_KEY": "none"},
                          arguments=config.harbor_arguments, workers=config.workers,
                          ready_timeout=config.ready_timeout)
    history: list[SchedulerRecord] = []

    def log(record: SchedulerRecord) -> None:
        history.append(record)
        print(f"update {record.updates:3d}  reward {record.metrics.get('reward/mean', 0.0):.3f}  "
              f"version {record.version}  lag {record.lag}  resubmitted {sum(record.resubmitted.values())}",
              flush=True)

    def harbor_tasks(batch) -> list[Task]:
        return [Task(task.id, {HARBOR_KEY: str(tasks[int(task.id)])}) for task in task_ids(batch)]

    rollout = RolloutScheduler(objective, trials, Publication(push, stamp=gateway.stamp), width=config.width,
                               rows=config.prompts * config.groups, tasks=harbor_tasks, groups=config.groups,
                               max_lag=config.max_lag, ahead=config.max_lag, truncation=config.truncation, log=log)
    draw = np.random.default_rng(config.seed)
    data = Dataset(train=lambda partition: iter(
                       lambda: {"task_id": draw.integers(0, len(tasks), config.prompts, np.int32)}, None),
                   val=None, records=len(tasks), batch=config.prompts)
    trainer = Trainer(objective, optax.adamw(config.learning_rate, b2=0.99, weight_decay=0.0),
                      key=jax.random.key(config.seed), rollout=rollout)
    try:
        with trials:
            try:
                state = trainer.fit(rollout.tasks(data), steps=config.steps, log_every=1)
            finally:
                rollout.close()
    finally:
        if fake is not None:
            fake.shutdown()
            fake.policy.close()
    summary = {"updates": int(state.updates), "rewards": [record.metrics.get("reward/mean") for record in history],
               "max_lag": max(record.lag for record in history)}
    (config.out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


HARNESS = '''\
import json, os, sys, urllib.request
arguments = sys.argv[1:]
def value(flag):
    return arguments[arguments.index(flag) + 1]
agent = dict(arguments[index + 1].split("=", 1) for index, flag in enumerate(arguments) if flag == "--ae")
trial = os.path.join(value("--trials-dir"), value("--trial-name"))
os.makedirs(trial)
messages, replies = [{"role": "user", "content": "say hello"}], []
for turn in range(2):
    request = urllib.request.Request(agent["OPENAI_BASE_URL"] + "/chat/completions", method="POST",
                                     data=json.dumps({"messages": messages, "max_tokens": 8}).encode())
    reply = json.load(urllib.request.urlopen(request))["choices"][0]["message"]["content"]
    replies.append(reply)
    messages += [{"role": "assistant", "content": reply}, {"role": "user", "content": "go on"}]
text = "".join(replies)
reward = sum(character in "aeiou" for character in text) / max(len(text), 1)
with open(os.path.join(trial, "result.json"), "w") as out:
    json.dump({"exception_info": None, "verifier_result": {"rewards": {"reward": reward}}}, out)
'''


class SmokeGateway(ThreadingHTTPServer):
    """rllm-model-gateway's routes a HarborSource reads, over Dew's own server as the engine."""

    def __init__(self, policy: NativeRolloutServer, words):
        super().__init__(("127.0.0.1", 0), _SmokeRoutes)
        self.policy, self.words = policy, words
        self.traces: dict[str, list[dict]] = {}
        self.stamp: int | None = None
        self.lock = threading.Lock()

    def handle_error(self, request, client_address) -> None:
        """Quiet: a trial cancelled mid-call leaves its request's socket and draw behind."""


class _SmokeRoutes(BaseHTTPRequestHandler):
    server: SmokeGateway

    def answer(self, body) -> None:
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health/workers":
            self.answer({"healthy": 1, "total": 1})
        elif self.path == "/v1/models":
            self.answer({"data": [{"id": "policy"}]})
        else:
            with self.server.lock:
                self.answer(self.server.traces.get(self.path.removeprefix("/sessions/").removesuffix("/traces"), []))

    def do_DELETE(self) -> None:
        with self.server.lock:
            self.server.traces.pop(self.path.removeprefix("/sessions/"), None)
        self.answer({})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/admin/weight_version":
            self.server.stamp = body["weight_version"]
            self.answer(body)
            return
        session = self.path.removeprefix("/sessions/").removesuffix("/v1/chat/completions")
        # The engine's chat rendering: role-tagged text, tokenized once; the ids are what the policy reads.
        text = "".join(f"<{message['role']}>{message['content']}\n" for message in body["messages"])
        prompt = self.server.words.encode(text)
        began, version = time.time(), self.server.stamp
        draw = self.server.policy.submit(prompt, body["max_tokens"], seed=hash(session) % 2 ** 31).result()
        trace = {"prompt_token_ids": list(prompt), "completion_token_ids": list(draw.tokens),
                 "logprobs": list(draw.behavior_log_probs), "finish_reason": "stop" if draw.terminated else "length",
                 "weight_version": version, "timestamp": time.time(), "latency_ms": (time.time() - began) * 1000}
        with self.server.lock:
            self.server.traces.setdefault(session, []).append(trace)
        self.answer({"choices": [{"message": {"role": "assistant", "content": self.server.words.decode(draw.tokens)},
                                  "finish_reason": trace["finish_reason"]}]})

    def log_message(self, *arguments) -> None:
        pass


def smoke_setup(config: Config, source) -> tuple[SmokeGateway, Path, tuple[Path, ...]]:
    """The stand-in harbor, two task directories and a gateway over Dew's server on the tiny policy."""
    harbor = config.out / "harbor"
    harbor.write_text(f"#!{sys.executable}\n" + HARNESS)
    harbor.chmod(0o755)
    tasks = tuple(config.out / "tasks" / name for name in ("t0", "t1"))
    for task in tasks:
        task.mkdir(parents=True, exist_ok=True)
    stock = source.text_generation().sampling
    sampling = Sampling(temperature=1.0, eos_id=stock.eos_id, pad_id=stock.pad_id)
    engine = Server.from_task(TextGeneration(source.model, source.variables, source.processor, sampling=sampling),
                              slots=config.prompts * config.groups, capacity=config.width)
    fake = SmokeGateway(NativeRolloutServer(engine), tokenizer_for(str(FIXTURES / "diffusion-gemma-workflow")))
    threading.Thread(target=fake.serve_forever, daemon=True).start()
    return fake, harbor, tasks


if __name__ == "__main__":
    main(tyro.cli(Config))
