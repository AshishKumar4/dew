"""A receipt of an NCCL weight push into real vLLM replicas.

    python tools/vllm_publication.py OUT --base DIR --policy DIR --python VLLM_PYTHON [--engine-gpu 1]

Needs two GPUs, vLLM 0.30.0 in its own environment (`--python` is its interpreter), and two
checkpoints of one architecture, such as Qwen/Qwen2.5-0.5B and Qwen/Qwen2.5-0.5B-Instruct. On the
engine GPU it starts two replicas from `--base` with NCCL weight transfer and a reference engine from
`--policy`, each on 127.0.0.1. Dew, on GPU 0, loads both checkpoints and serves each replica through
an `OpenAIRolloutServer` whose weights are one `NCCLPush` to both; loading `--policy`'s variables as
version 1 pushes them. Every engine draws the same prompts under one seed before and after, and
`OUT/receipt.json` records each engine's draws, each replica's `/weight_info` version and the
push's timings. A delivered push makes both replicas draw what the reference engine draws, token
for token and likelihood for likelihood, where they drew otherwise before it; the tool exits
nonzero unless they do. Each engine's output goes to `OUT/{name}.log`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

# The trainer's JAX leaves the engine GPU's memory to the engines.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import httpx

PROMPTS = ("Name three prime numbers.", "What is the capital of France?", "Write a haiku about rain.")
BUDGET = 24
SEED = 7
PORTS = {"first": 8021, "second": 8022, "reference": 8023}


def nccl_library(python: str) -> str:
    """The NCCL library the engine environment loads, which the push must load too."""
    found = subprocess.run([python, "-c", "import importlib.util, pathlib; print(pathlib.Path(next(iter("
                            "importlib.util.find_spec('nvidia.nccl').submodule_search_locations))) "
                            "/ 'lib' / 'libnccl.so.2')"], check=True, capture_output=True, text=True)
    return found.stdout.strip()


@contextlib.contextmanager
def engines(python: str, base: Path, policy: Path, gpu: int, out: Path) -> Iterator[dict[str, str]]:
    """The two replicas from `base` and the reference from `policy`, healthy, stopped after."""
    bin = str(Path(python).parent)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "VLLM_SERVER_DEV_MODE": "1",
                   "PATH": os.pathsep.join((bin, os.environ["PATH"]))}
    started = []
    try:
        for name, port in PORTS.items():
            model = policy if name == "reference" else base
            # Every engine holds bfloat16 weights, the dtype the push sends, so a delivered push
            # leaves the replicas' weights equal to the reference's.
            command = [python, "-m", "vllm.entrypoints.openai.api_server", "--model", str(model), "--host",
                       "127.0.0.1", "--port", str(port), "--gpu-memory-utilization", "0.25",
                       "--max-model-len", "2048", "--enforce-eager", "--dtype", "bfloat16"]
            if name != "reference":
                command += ["--weight-transfer-config", json.dumps({"backend": "nccl"})]
            log = (out / f"{name}.log").open("w")
            started.append((subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT),
                            log))
            # One at a time: each engine sizes its KV cache from the memory the ones before it left.
            healthy(f"http://127.0.0.1:{port}", started[-1][0], name)
        yield {name: f"http://127.0.0.1:{port}" for name, port in PORTS.items()}
    finally:
        for process, log in started:
            process.terminate()
            process.wait(60)
            log.close()


def healthy(root: str, process: subprocess.Popen[bytes], name: str) -> None:
    """Wait until the engine at `root` answers its health check, the readiness its server reports;
    an engine still loading refuses the connection."""
    deadline = time.monotonic() + 900
    while not _answers(f"{root}/health"):
        if process.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError(f"the {name} engine did not start; see its log")
        time.sleep(3)


def _answers(url: str) -> bool:
    try:
        return httpx.get(url, timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--python", required=True, help="the interpreter of vLLM's environment")
    parser.add_argument("--engine-gpu", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    import openai
    from transformers import AutoTokenizer

    from dew.inference import OpenAICompletion, OpenAIRolloutServer
    from dew.inference.nccl import NCCLPush
    from dew.interop.pretrained import Pretrained
    from dew.sampling import Sampling

    tokenizer = AutoTokenizer.from_pretrained(args.policy)
    def asked(text: str) -> tuple[int, ...]:
        """The chat template's ids for one user turn, a list (not Transformers 5's BatchEncoding)."""
        turn = [{"role": "user", "content": text}]
        return tuple(tokenizer.apply_chat_template(turn, add_generation_prompt=True, tokenize=True,
                                                   return_dict=False))

    prompts = [asked(text) for text in PROMPTS]
    sampling = Sampling(temperature=1.0, eos_token_ids=tokenizer.eos_token_id)
    base = Pretrained.load(args.base, dtype="bfloat16")
    policy = Pretrained.load(args.policy, dtype="bfloat16")
    with engines(args.python, args.base, args.policy, args.engine_gpu, args.out) as roots:
        push = NCCLPush(base, (roots["first"], roots["second"]), nccl_library(args.python))
        servers = {name: OpenAIRolloutServer(
            OpenAICompletion(str(args.policy if name == "reference" else args.base),
                             openai.OpenAI(base_url=f"{root}/v1", api_key="receipt", max_retries=0),
                             provider="vllm"),
            sampling, push if name == "first" else (lambda variables, version: None))
            for name, root in roots.items()}

        def drawn() -> dict[str, list[dict[str, object]]]:
            return {name: [{"tokens": list(draw.tokens), "log_probs": list(draw.behavior_log_probs),
                            "version": draw.version}
                           for draw in (server.submit(prompt, BUDGET, key=SEED).result()
                                        for prompt in prompts)]
                    for name, server in servers.items()}

        try:
            before = drawn()
            # The first replica's server publishes to both replicas; the second's learns the version.
            servers["first"].load(policy.variables, 1)
            servers["second"].load(policy.variables, 1)
            after = drawn()
            versions = {name: httpx.get(f"{roots[name]}/weight_info", timeout=30).json()
                        for name in ("first", "second")}
        finally:
            push.close()
            for server in servers.values():
                server.close()
    def sampled(draws: list[dict[str, object]]) -> list[tuple[object, object]]:
        return [(draw["tokens"], draw["log_probs"]) for draw in draws]

    replicas = ("first", "second")
    delivered = all(sampled(after[name]) == sampled(after["reference"]) for name in replicas)
    distinct = all(sampled(before[name]) != sampled(before["reference"]) for name in replicas)
    stamped = all(str(versions[name].get("weight_version")) == "1"
                  and {draw["version"] for draw in after[name]} == {1} for name in replicas)
    receipt = {"base": str(args.base), "policy": str(args.policy),
               "prompts": [list(prompt) for prompt in prompts],
               "budget": BUDGET, "seed": SEED, "before": before, "after": after, "weight_info": versions,
               "timings": push.timings, "delivered": delivered, "distinct_before": distinct,
               "stamped": stamped}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=1))
    print(json.dumps({key: receipt[key] for key in ("delivered", "distinct_before", "stamped", "timings")}))
    sys.exit(0 if delivered and distinct and stamped else 1)


if __name__ == "__main__":
    main()
