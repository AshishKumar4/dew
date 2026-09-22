"""RLVR: GRPO on programs that must pass their tests, rolled out asynchronously.

    python examples/train_rlvr.py --backend native --steps 40 --out runs/rlvr-native
    python examples/train_rlvr.py --backend vllm --steps 40 --out runs/rlvr-vllm

Every prompt asks for a Python program that reads two integers from stdin
and prints a stated function of them. A completion's reward is the fraction
of three hidden test cases its program passes, run in a `SandboxFleet` of
resource-limited processes with a wall clock, CPU time and memory cap.

Rollouts run on a rollout server while the trainer updates. `--backend
native` serves them from Dew's own continuous-batching `Server` in this
process, with weights pushed in place. `--backend vllm` starts a vLLM
OpenAI-compatible server on an export of the same checkpoint, samples from
it by token ids, and pushes weights by writing safetensors and asking vLLM to
reload them (its development endpoints, `VLLM_SERVER_DEV_MODE=1`); `--vllm`
names the executable, which may live in its own environment. The rollout
draws one batch ahead of the update, so each batch is at most one update
stale; the GRPO objective's importance cap corrects for it.

The run prints one line per update and writes `rewards.json` to `--out`
with the per-update reward, policy version and lag, and the mean reward of
the first and last `--window` updates.

    JAX_PLATFORMS=cpu python examples/train_rlvr.py --smoke --out /tmp/rlvr-smoke
"""

from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import httpx
import jax
import jax.numpy as jnp
import optax
import tyro

from dew.data import Loading, tokenizer_for
from dew.data.prompts import Prompts
from dew.inference.tasks import SHAPE_BUCKETS
from dew.inference import (
    NativeRolloutServer,
    OpenAICompletion,
    OpenAIRolloutServer,
    SafetensorsReload,
    Server,
    TextGeneration,
)
from dew.interop import load_pretrained
from dew.objectives.rl import (
    AsyncRollout,
    CodeReward,
    GRPOObjective,
    ProcessRunner,
    RolloutRecord,
    SandboxFleet,
    SandboxLimits,
)
from dew.sampling import Sampling
from dew.training import Trainer

SMOKE_MODEL = Path(__file__).resolve().parents[1] / "tests/fixtures/hf/qwen2-tiny"

# (what to print, how to compute it from a and b, a valid input range)
TASKS = (
    ("the sum of a and b", lambda a, b: a + b, (-50, 50)),
    ("the product of a and b", lambda a, b: a * b, (-30, 30)),
    ("a minus b", lambda a, b: a - b, (-50, 50)),
    ("the larger of a and b", max, (-99, 99)),
    ("the absolute difference between a and b", lambda a, b: abs(a - b), (-99, 99)),
    ("the sum of the squares of a and b", lambda a, b: a * a + b * b, (-20, 20)),
    ("the sum of all integers from min(a, b) to max(a, b) inclusive",
     lambda a, b: sum(range(min(a, b), max(a, b) + 1)), (-30, 30)),
    ("a integer-divided by b, rounded down (b is never zero)", lambda a, b: a // b, (1, 60)),
    ("the remainder of a divided by b (b is never zero)", lambda a, b: a % b, (1, 60)),
    ("the number of multiples of b between 1 and a inclusive (both are positive)",
     lambda a, b: a // b, (1, 90)),
    ("the sum of the decimal digits of a times b", lambda a, b: sum(map(int, str(abs(a * b)))), (1, 99)),
    ("2 * a + 3 * b", lambda a, b: 2 * a + 3 * b, (-40, 40)),
)


def records(count: int, seed: int) -> tuple[str, ...]:
    """`count` prompt rows in the verl layout, each with three hidden test cases."""
    draw = random.Random(seed)
    rows = []
    for _ in range(count):
        text, function, (low, high) = draw.choice(TASKS)
        cases = []
        for _ in range(3):
            a, b = draw.randint(low, high), draw.randint(low, high)
            cases.append({"stdin": f"{a}\n{b}\n", "stdout": str(function(a, b))})
        prompt = (f"Write a Python program that reads two integers a and b from standard input, "
                  f"one per line, and prints {text}. Print nothing but that number. "
                  f"Reply with one ```python code block.")
        rows.append(json.dumps({"prompt": [{"role": "user", "content": prompt}], "data_source": "code",
                                "ground_truth": json.dumps(cases)}))
    return tuple(rows)


@dataclass
class Config:
    model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    backend: str = "native"
    """native: Dew's own server in this process; vllm: a vLLM server this run starts."""
    out: Path = Path("runs/rlvr")
    steps: int = 40
    prompts: int = 8
    """Prompts per update; each gets `groups` completions."""
    groups: int = 8
    prompt_tokens: int = 128
    new_tokens: int = 128
    learning_rate: float = 2e-6
    max_lag: int = 1
    tasks: int = 2048
    window: int = 10
    """Updates averaged at each end of the run for the reward comparison."""
    vllm: str = "vllm"
    port: int = 8011
    vllm_memory: float = 0.12
    """vLLM's --gpu-memory-utilization; the trainer takes the rest."""
    seed: int = 0
    smoke: bool = False
    """Two updates of the committed tiny Qwen2 on CPU, native backend."""


def launch_vllm(config: Config, directory: Path) -> subprocess.Popen:
    """Start vLLM on the exported checkpoint and wait until it answers."""
    log = open(config.out / "vllm.log", "w")
    process = subprocess.Popen(
        [config.vllm, "serve", str(directory), "--served-model-name", "policy", "--port", str(config.port),
         "--gpu-memory-utilization", str(config.vllm_memory), "--dtype", "bfloat16",
         "--max-model-len", str(config.prompt_tokens + config.new_tokens), "--generation-config", "vllm",
         "--enable-prefix-caching", "--seed", str(config.seed)],
        # vLLM runs build tools (ninja) from its own environment's bin directory.
        env={**os.environ, "VLLM_SERVER_DEV_MODE": "1",
             "PATH": os.pathsep.join((str(Path(shutil.which(config.vllm) or config.vllm).parent),
                                      os.environ.get("PATH", "")))},
        stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 900
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited with {process.returncode}; see {config.out / 'vllm.log'}")
        try:
            if httpx.get(f"http://127.0.0.1:{config.port}/health", timeout=2).status_code == 200:
                return process
        except httpx.HTTPError:
            pass
        time.sleep(2)
    process.kill()
    raise TimeoutError("vLLM did not come up within fifteen minutes")


def main(config: Config) -> dict:
    if config.smoke:
        config = replace(config, model=str(SMOKE_MODEL), backend="native", steps=2, prompts=2, groups=2,
                         prompt_tokens=56, new_tokens=8, tasks=8, window=1)
    config.out.mkdir(parents=True, exist_ok=True)
    tokenizer = str(SMOKE_MODEL.parents[0] / "diffusion-gemma-workflow") if config.smoke else config.model
    width = config.prompt_tokens + config.new_tokens
    # The server rounds its cache up to a power-of-two shape bucket; the model's context covers it.
    context = next(bucket for bucket in SHAPE_BUCKETS if bucket >= width)
    source = load_pretrained(config.model, dtype="float32" if config.smoke else "bfloat16",
                             param_dtype="float32", max_seq_len=context)
    stock = source.text_generation().sampling
    # Temperature one without filters: the engine's reported likelihoods are
    # then the behavior policy's, on either backend.
    sampling = Sampling(temperature=1.0, eos_id=stock.eos_id, pad_id=stock.pad_id)
    words = tokenizer_for(tokenizer)

    objective = GRPOObjective(source.model, width - 1, pretrained=source.variables,
                              behavior_importance_cap=2.0, epsilon_high=0.28)
    if config.backend == "native":
        served = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.bfloat16) if jnp.issubdtype(leaf.dtype, jnp.floating)
                              else jnp.asarray(leaf), source.variables)
        engine = Server.from_task(TextGeneration(source.model, served, source.processor, sampling=sampling),
                                  slots=config.prompts * config.groups, capacity=width)
        server = NativeRolloutServer(engine)
        vllm = None
    elif config.backend == "vllm":
        import openai

        directory = config.out / "served"
        root = f"http://127.0.0.1:{config.port}"
        weights = SafetensorsReload(source, directory, root)
        weights.write(source.variables)
        vllm = launch_vllm(config, directory)
        # A seeded request is safe to resend, so the SDK's retries cover a
        # keep-alive connection the server closed between requests.
        completion = OpenAICompletion("policy", openai.OpenAI(base_url=f"{root}/v1", api_key="none", max_retries=3,
                                                              timeout=600), provider="vllm")
        server = OpenAIRolloutServer(completion, sampling, weights, workers=config.prompts * config.groups * 2)
    else:
        raise ValueError(f"backend is native or vllm, got {config.backend!r}")

    history: list[RolloutRecord] = []

    def log(record: RolloutRecord) -> None:
        history.append(record)
        print(f"update {record.updates:3d}  reward {record.reward:.3f}  version {record.version}  "
              f"lag {record.lag}  redrawn {record.redrawn}  waited {record.waited:.1f}s", flush=True)

    limits = SandboxLimits(wall_seconds=5.0, cpu_seconds=2, memory_bytes=512 * 1024 ** 2, message_bytes=65536)
    fleet = SandboxFleet(ProcessRunner(), limits=limits, workers=max(os.cpu_count() or 1, 4))
    rollout = AsyncRollout(objective, server, CodeReward(fleet), decode=words.decode, groups=config.groups,
                           max_new_tokens=config.new_tokens, max_lag=config.max_lag, ahead=config.max_lag,
                           log=log)
    data = Prompts(tokenizer=tokenizer, records=records(config.tasks, config.seed),
                   max_prompt_len=config.prompt_tokens, pad_id=sampling.pad_id, val_batches=None,
                   loading=Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1),
                   seed=config.seed).load(batch=config.prompts)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.adamw(config.learning_rate, b2=0.99, weight_decay=0.0))
    trainer = Trainer(objective, optimizer, key=jax.random.key(config.seed), rollout=rollout)
    began = time.perf_counter()
    try:
        state = trainer.fit(rollout.prompts(data), steps=config.steps, log_every=1)
    finally:
        server.close()
        rollout.close()
        fleet.close()
        if vllm is not None:
            vllm.terminate()
            vllm.wait(timeout=60)
    rewards = [record.reward for record in history]
    window = min(config.window, len(rewards) // 2 or 1)
    summary = {
        "backend": config.backend, "model": config.model, "updates": int(state.updates),
        "seconds": time.perf_counter() - began, "device": jax.devices()[0].device_kind,
        "first_reward": sum(rewards[:window]) / window, "last_reward": sum(rewards[-window:]) / window,
        "max_lag": max(record.lag for record in history), "redrawn": sum(record.redrawn for record in history),
        "history": [asdict(record) for record in history],
    }
    (config.out / "rewards.json").write_text(json.dumps(summary, indent=1))
    print(f"{config.backend}: reward {summary['first_reward']:.3f} over the first {window} updates, "
          f"{summary['last_reward']:.3f} over the last {window}; largest lag {summary['max_lag']}")
    return summary


if __name__ == "__main__":
    main(tyro.cli(Config))
