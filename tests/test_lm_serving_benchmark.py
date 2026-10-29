"""The serving benchmark (tools/benchmark_lm_serving.py), run end to end on the CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from dew.interop.pretrained import PretrainedDecoder
from dew.nn.backbones.causal_transformer import CausalTransformer

ROOT = Path(__file__).resolve().parents[1]
WIDTH, MLP, VOCABULARY = 256, 512, 512
"""A decoder wide enough that its embeddings, projections and MLP reach the
layout's `min_shard` (2**16 elements), so a mesh splits them."""
SLOTS, REQUESTS, PROMPT, OUTPUT = 2, 4, 8, 8


def checkpoint(directory: Path) -> Path:
    """A random decoder of `WIDTH`, saved in its Hugging Face layout by Dew's exporter."""
    model = CausalTransformer(vocab_size=VOCABULARY, emb_features=WIDTH, num_layers=2, num_heads=4,
                              head_dim=WIDTH // 4, mlp_features=MLP, max_seq_len=64, dtype="float32")
    variables = model.init(jax.random.key(0), jnp.zeros((1, PROMPT), jnp.int32))
    PretrainedDecoder.from_model(model, variables).save(directory)
    return directory


def served(model: Path, out: Path, *options: str) -> dict:
    """The benchmark's record, served on two CPU devices."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT / "src"),
           "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
    subprocess.run([sys.executable, str(ROOT / "tools" / "benchmark_lm_serving.py"), "--backend", "dew",
                    "--model", str(model), "--slots", str(SLOTS), "--requests", str(REQUESTS),
                    "--prompt", str(PROMPT), "--output", str(OUTPUT), "--vocab-limit", str(VOCABULARY),
                    "--repeats", "1", "--out", str(out), *options],
                   check=True, env=env, cwd=ROOT)
    return json.loads(out.read_text())


@pytest.mark.whole_suite
def test_the_benchmark_serves_weights_a_mesh_splits(tmp_path):
    """`--mesh` places the weights on that mesh before the server is built: the record names the
    mesh and the axis that splits them, which the default placement over the same two devices
    (data parallelism) leaves whole, and serves every requested token. What a server on a mesh
    draws is tests/test_serving.py's contract
    (`test_a_server_on_a_mesh_draws_what_each_request_draws_alone`)."""
    model = checkpoint(tmp_path / "decoder")
    whole = served(model, tmp_path / "whole.json")
    record = served(model, tmp_path / "split.json", "--mesh", '{"fsdp": 2}')
    assert (whole["mesh"], whole["hardware"]["weights"]) == (None, {"split_over": [], "devices": 2})
    assert (record["mesh"], record["hardware"]["weights"]) == ({"fsdp": 2},
                                                               {"split_over": ["fsdp"], "devices": 2})
    assert all(repeat["output_tokens"] == REQUESTS * OUTPUT
               for run in (whole, record) for point in run["sweep"] for repeat in point["repeats"])
    # Each slot count's measured peak, in a process of its own that loaded every program from the cache a
    # first run filled (tools/peak_memory.py); on the CPU, whose allocator keeps no high-water, unmeasured.
    assert all(point["peak_device_bytes"] is None for run in (whole, record) for point in run["sweep"])


VLLM = {
    "__init__.py": '''import json, os

class SamplingParams:
    def __init__(self, **policy):
        json.dump(policy, open(os.environ["POLICY"], "w"))
        self.max_tokens = policy["max_tokens"]
''',
    "inputs.py": "class TokensPrompt(dict):\n    pass\n",
    "engine/__init__.py": "",
    "engine/arg_utils.py": ("class AsyncEngineArgs:\n    def __init__(self, **settings):\n"
                            "        self.settings = settings\n"),
    "v1/__init__.py": "", "v1/engine/__init__.py": "",
    "v1/engine/async_llm.py": '''import asyncio, types

class AsyncLLM:
    @classmethod
    def from_engine_args(cls, args):
        return cls()

    async def generate(self, prompt, policy, request_id):
        for count in range(1, policy.max_tokens + 1):
            await asyncio.sleep(0)
            yield types.SimpleNamespace(outputs=[types.SimpleNamespace(token_ids=[0] * count)])

    def shutdown(self):
        pass
''',
}


def test_vllms_engine_streams_token_ids_as_the_other_engines_do(tmp_path):
    """vllm-engine asks for token ids, not text (`detokenize=False`), as Dew's and SGLang's engines give,
    so no engine's clock holds text decoding the others skip; and it times on the same closed loop."""
    for name, text in VLLM.items():
        (tmp_path / "engine" / "vllm" / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "engine" / "vllm" / name).write_text(text)
    policy = tmp_path / "policy.json"
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "engine"), "POLICY": str(policy)}
    subprocess.run([sys.executable, str(ROOT / "tools" / "benchmark_lm_serving.py"),
                    "--backend", "vllm-engine", "--model", "a/model", "--slots", "2", "--prompt", "8",
                    "--output", "5", "--vocab-limit", "200", "--repeats", "1", "--requests", "4",
                    "--out", str(tmp_path / "record.json")], check=True, env=env, cwd=ROOT)
    assert json.loads(policy.read_text()) == {
        "temperature": 0, "top_p": 1, "top_k": -1, "min_p": 0, "repetition_penalty": 1, "max_tokens": 5,
        "ignore_eos": True, "detokenize": False}
    record = json.loads((tmp_path / "record.json").read_text())
    assert record["sweep"][0]["repeats"][0]["output_tokens"] == 4 * 5


@pytest.mark.parametrize("kv", ["dense", "paged"])
@pytest.mark.parametrize("fixture", ["qwen35-tiny-mm", "gemma3-tiny-mm"])
def test_a_multimodal_model_serves_the_layout_asked_for(tmp_path, fixture, kv):
    """A multimodal model's wrapper hands the layout asked for to its text decoder, which holds the
    cache: `--kv dense` and `--kv paged` serve a Qwen3.5 wrapper, whose delta-net layers keep each
    row's state beside the attention layers' pages, and a Gemma 3 one, every requested token."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT / "src"),
           "XLA_FLAGS": "--xla_force_host_platform_device_count=1"}
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "benchmark_lm_serving.py"),
                           "--backend", "dew", "--model", str(ROOT / "tests" / "fixtures" / "hf" / fixture),
                           "--slots", "2", "--requests", "4", "--prompt", "8", "--output", "8",
                           "--vocab-limit", "200", "--repeats", "1", "--kv", kv,
                           "--out", str(tmp_path / "out.json")],
                          env=env, cwd=ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-3000:]
    record = json.loads((tmp_path / "out.json").read_text())
    assert record["sweep"][0]["repeats"][0]["output_tokens"] == 4 * 8
