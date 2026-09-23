"""RLVR: GRPO on programs that must pass their tests, rolled out asynchronously.

    python examples/train_rlvr.py --backend native --steps 40 --out runs/rlvr-native
    python examples/train_rlvr.py --backend vllm --steps 40 --out runs/rlvr-vllm
    python examples/train_rlvr.py --backend sglang --steps 40 --out runs/rlvr-sglang

Every prompt asks for a Python program that reads two integers from stdin
and prints a stated function of them. A completion's reward is the fraction
of three hidden test cases its program passes, run in a `SandboxFleet`.
`--runner container` runs each program in a network-less `python:3.12-slim`
container. `--runner process` (the default, and the only choice where Docker
is absent, as on Colab) runs it as a process with a wall clock, CPU time and
memory cap; that process has your user's filesystem and network.

Rollouts run on a rollout server while the trainer updates. `--backend
native` serves them from Dew's own continuous-batching `Server` in this
process, with weights pushed in place. `--backend vllm` and `--backend
sglang` start that engine's OpenAI-compatible server on an export of the
same checkpoint, sample from it by token ids, and push weights by writing
safetensors and asking the engine to reload them (vLLM's development
endpoints, `VLLM_SERVER_DEV_MODE=1`; SGLang's `/update_weights_from_disk`).
`--vllm` and `--sglang` name the executables, which may live in their own
environments. A `RolloutScheduler` over a `PromptSource` draws one batch
ahead of the update, so each batch is at most one update stale; the GRPO
objective's importance cap corrects for it. A completion that runs out of
`--new-tokens` is still scored and trained on (`truncation="score"`): a
closed code block followed by cut-off prose can pass every test.

The run prints one line per update and writes `rewards.json` to `--out`
with the per-update reward, policy version and lag, the mean reward of the
first and last `--window` updates, and the seconds each weight push took.

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
from dew.inference import (
    NativeRolloutServer,
    OpenAICompletion,
    OpenAIRolloutServer,
    SafetensorsReload,
    Server,
    TextGeneration,
)
from dew.inference.tasks import SHAPE_BUCKETS
from dew.interop import load_pretrained
from dew.objectives.rl import (
    CodeReward,
    ContainerRunner,
    GRPOObjective,
    ProcessRunner,
    PromptSource,
    RolloutScheduler,
    SandboxFleet,
    SandboxLimits,
    SchedulerRecord,
    prompt_tasks,
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
    """native: Dew's own server in this process; vllm or sglang: that engine's server, started by this run."""
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
    runner: str = "process"
    """process: limited local processes; container: network-less Docker containers of --image."""
    image: str = "python:3.12-slim"
    vllm: str = "vllm"
    sglang: str = "sglang"
    port: int = 8011
    vllm_memory: float = 0.12
    """vLLM's --gpu-memory-utilization, a fraction of the whole GPU; the trainer takes the rest."""
    sglang_memory: float = 0.8
    """SGLang's --mem-fraction-static, a fraction of the memory the trainer left free."""
    seed: int = 0
    smoke: bool = False
    """Two updates of the committed tiny Qwen2 on CPU, native backend."""


def engine_context(config: Config) -> int:
    """The context window the engine is started with: the rollout width, and SGLang's reserve on top.

    SGLang 0.5.20 caps a request's budget at `context - input - 2` and
    refuses an input of `context - 6` ids or more, so at a context equal to
    the width a prompt at the window would draw fewer than `new_tokens` ids
    and end with a "length" the rollout refuses. vLLM's `--max-model-len`
    admits the full width.
    """
    width = config.prompt_tokens + config.new_tokens
    return width + 6 if config.backend == "sglang" else width


def engine_command(config: Config, directory: Path) -> tuple[list[str], dict[str, str]]:
    """The command line and extra environment that serve `directory` on `config.backend`."""
    context = str(engine_context(config))
    if config.backend == "vllm":
        return ([config.vllm, "serve", str(directory), "--served-model-name", "policy", "--port", str(config.port),
                 "--gpu-memory-utilization", str(config.vllm_memory), "--dtype", "bfloat16",
                 "--max-model-len", context, "--generation-config", "vllm",
                 "--enable-prefix-caching", "--seed", str(config.seed)],
                {"VLLM_SERVER_DEV_MODE": "1"})
    return ([config.sglang, "serve", "--model-path", str(directory), "--served-model-name", "policy",
             "--port", str(config.port), "--mem-fraction-static", str(config.sglang_memory), "--dtype", "bfloat16",
             "--context-length", context, "--random-seed", str(config.seed)], {})


def launch_engine(config: Config, directory: Path) -> subprocess.Popen:
    """Start the remote engine on the exported checkpoint and wait until it answers."""
    command, extra = engine_command(config, directory)
    with open(config.out / f"{config.backend}.log", "w") as log:
        process = subprocess.Popen(
            command,
            # Engines run build tools (ninja) from their own environment's bin directory.
            env={**os.environ, **extra,
                 "PATH": os.pathsep.join((str(Path(shutil.which(command[0]) or command[0]).parent),
                                          os.environ.get("PATH", "")))},
            stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 900
    unanswered: httpx.HTTPError | None = None
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{config.backend} exited with {process.returncode}; "
                               f"see {config.out / f'{config.backend}.log'}")
        try:
            if httpx.get(f"http://127.0.0.1:{config.port}/health", timeout=2).status_code == 200:
                return process
        except httpx.HTTPError as error:
            unanswered = error
        time.sleep(2)
    process.kill()
    raise TimeoutError(f"{config.backend} did not come up within fifteen minutes") from unanswered


def main(config: Config) -> dict:
    if config.smoke:
        config = replace(config, model=str(SMOKE_MODEL), backend="native", steps=2, prompts=2, groups=2,
                         prompt_tokens=56, new_tokens=8, tasks=8, window=1)
    config.out.mkdir(parents=True, exist_ok=True)
    tokenizer = str(SMOKE_MODEL.parents[0] / "diffusion-gemma-workflow") if config.smoke else config.model
    width = config.prompt_tokens + config.new_tokens
    # The server rounds its cache up to a power-of-two shape bucket, and an
    # engine refuses a context past the export's; the model's context covers both.
    context = next(bucket for bucket in SHAPE_BUCKETS if bucket >= engine_context(config))
    source = load_pretrained(config.model, dtype="float32" if config.smoke else "bfloat16",
                             param_dtype="float32", max_seq_len=context)
    stock = source.text_generation().sampling
    # Temperature one without filters: the engine's reported likelihoods are
    # then the behavior policy's, on either backend.
    sampling = Sampling(temperature=1.0, eos_id=stock.eos_id, pad_id=stock.pad_id)
    words = tokenizer_for(tokenizer)

    objective = GRPOObjective(source.model, width - 1, pretrained=source.variables,
                              behavior_importance=2.0, epsilon_high=0.28)
    pushes: list[float] = []
    if config.backend == "native":
        served = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.bfloat16) if jnp.issubdtype(leaf.dtype, jnp.floating)
                              else jnp.asarray(leaf), source.variables)
        engine = Server.from_task(TextGeneration(source.model, served, source.processor, sampling=sampling),
                                  slots=config.prompts * config.groups, capacity=width)
        server = NativeRolloutServer(engine)
        remote = None
    elif config.backend in ("vllm", "sglang"):
        import openai

        directory = config.out / "served"
        root = f"http://127.0.0.1:{config.port}"
        reload = SafetensorsReload(source, directory, root, config.backend)
        reload.write(source.variables)
        remote = launch_engine(config, directory)

        def push(variables) -> None:
            began = time.perf_counter()
            reload(variables)
            pushes.append(time.perf_counter() - began)

        # A seeded request is safe to resend, so the SDK's retries cover a
        # keep-alive connection the server closed between requests.
        completion = OpenAICompletion("policy", openai.OpenAI(base_url=f"{root}/v1", api_key="none", max_retries=3,
                                                              timeout=600), provider=config.backend)
        server = OpenAIRolloutServer(completion, sampling, push, workers=config.prompts * config.groups * 2)
    else:
        raise ValueError(f"backend is native, vllm or sglang, got {config.backend!r}")

    history: list[SchedulerRecord] = []

    def log(record: SchedulerRecord) -> None:
        history.append(record)
        print(f"update {record.updates:3d}  reward {record.metrics.get('reward/mean', 0.0):.3f}  "
              f"version {record.version}  "
              f"lag {record.lag}  resubmitted {sum(record.resubmitted.values())}  "
              f"truncated {record.metrics['status/truncated']:.2f}  waited {record.waited:.1f}s", flush=True)

    limits = SandboxLimits(wall_seconds=5.0, cpu_seconds=2, memory_bytes=512 * 1024 ** 2, message_bytes=65536)
    if config.runner not in ("process", "container"):
        raise ValueError(f"runner is process or container, got {config.runner!r}")
    runner = ProcessRunner() if config.runner == "process" else ContainerRunner(config.image)
    fleet = SandboxFleet(runner, limits=limits, workers=max(os.cpu_count() or 1, 4))
    prompts = PromptSource(server, CodeReward(fleet), decode=words.decode, max_new_tokens=config.new_tokens,
                           seed=config.seed)
    # One chain per completion at most, each within the prompt and response width.
    rollout = RolloutScheduler(objective, prompts, server, width=width, rows=config.prompts * config.groups,
                               tasks=prompt_tasks, groups=config.groups, max_lag=config.max_lag,
                               ahead=config.max_lag, truncation="score", log=log)
    data = Prompts(tokenizer=tokenizer, records=records(config.tasks, config.seed),
                   max_prompt_len=config.prompt_tokens, pad_id=sampling.pad_id, val_batches=None,
                   loading=Loading(workers=0, threads=1, read_buffer=2, worker_buffer=1),
                   seed=config.seed).load(batch=config.prompts)
    optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.adamw(config.learning_rate, b2=0.99, weight_decay=0.0))
    trainer = Trainer(objective, optimizer, key=jax.random.key(config.seed), rollout=rollout)
    began = time.perf_counter()
    try:
        state = trainer.fit(rollout.tasks(data), steps=config.steps, log_every=1)
    finally:
        rollout.close()
        prompts.close()
        server.close()
        fleet.close()
        if remote is not None:
            remote.terminate()
            remote.wait(timeout=60)
    # A batch with no scored rollout (every one truncated) reports no reward; it counts as zero.
    rewards = [record.metrics.get("reward/mean", 0.0) for record in history]
    window = min(config.window, len(rewards) // 2 or 1)
    summary = {
        "backend": config.backend, "model": config.model, "updates": int(state.updates),
        "seconds": time.perf_counter() - began, "device": jax.devices()[0].device_kind,
        "first_reward": sum(rewards[:window]) / window, "last_reward": sum(rewards[-window:]) / window,
        "max_lag": max(record.lag for record in history),
        "resubmitted": sum(sum(record.resubmitted.values()) for record in history), "push_seconds": pushes, "history": [asdict(record) for record in history],
    }
    (config.out / "rewards.json").write_text(json.dumps(summary, indent=1))
    print(f"{config.backend}: reward {summary['first_reward']:.3f} over the first {window} updates, "
          f"{summary['last_reward']:.3f} over the last {window}; largest lag {summary['max_lag']}"
          + (f"; median weight push {sorted(pushes)[len(pushes) // 2]:.2f}s over {len(pushes)}" if pushes else ""))
    return summary


if __name__ == "__main__":
    main(tyro.cli(Config))
