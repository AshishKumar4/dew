"""The serving benchmark (tools/benchmark_lm_serving.py), run end to end on the CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

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


def served(model: Path, out: Path, *options: str) -> tuple[dict, dict]:
    """The benchmark's record and its first repeat's generations, served on two CPU devices."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT / "src"),
           "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
    subprocess.run([sys.executable, str(ROOT / "tools" / "benchmark_lm_serving.py"), "--backend", "dew",
                    "--model", str(model), "--slots", str(SLOTS), "--requests", str(REQUESTS),
                    "--prompt", str(PROMPT), "--output", str(OUTPUT), "--vocab-limit", str(VOCABULARY),
                    "--repeats", "1", "--generations", "--out", str(out), *options],
                   check=True, env=env, cwd=ROOT)
    with np.load(out.with_name(f"{out.stem}-slots{SLOTS}.npz")) as generations:
        return json.loads(out.read_text()), dict(generations)


def test_a_model_served_over_a_mesh_decodes_what_it_decodes_with_its_weights_whole(tmp_path):
    """`--mesh` places the weights on that mesh before the server is built: the record names the
    mesh and the axis that splits them, which the default placement over the same two devices
    (data parallelism) leaves whole. FSDP gathers each weight before its product, and splits the
    requests over the devices as data parallelism does, so each device computes what it computes
    with the weights whole: the same tokens and log-probabilities, bit for bit."""
    model = checkpoint(tmp_path / "decoder")
    whole, generated = served(model, tmp_path / "whole.json")
    record, split = served(model, tmp_path / "split.json", "--mesh", '{"fsdp": 2}')
    assert (whole["mesh"], whole["hardware"]["weights"]) == (None, {"split_over": [], "devices": 2})
    assert (record["mesh"], record["hardware"]["weights"]) == ({"fsdp": 2},
                                                               {"split_over": ["fsdp"], "devices": 2})
    for name in ("tokens", "raw", "behavior"):
        np.testing.assert_array_equal(split[name], generated[name], err_msg=name)


SGLANG = '''"""SGLang's in-process engine, standing in: a request streams its output ids, one chunk a token."""
import asyncio, json, os

class Engine:
    def __init__(self, **settings):
        json.dump(settings, open(os.environ["SETTINGS"], "w"))

    async def async_generate(self, input_ids, sampling_params, stream):
        assert stream and sampling_params["temperature"] == 0 and sampling_params["ignore_eos"]
        async def chunks():
            for count in range(1, sampling_params["max_new_tokens"] + 1):
                await asyncio.sleep(0)
                yield {"output_ids": [0] * count, "meta_info": {"completion_tokens": count}}
        return chunks()

    def shutdown(self):
        pass
'''


def test_the_benchmark_serves_through_sglangs_engine_on_its_clocks(tmp_path):
    """--backend sglang-engine builds SGLang's engine as the vLLM engine is built (bf16, the prompt and
    output as its context, the most slots as its running requests, no prefix cache) and times each
    request from SGLang's stream of growing output, every token counted."""
    (tmp_path / "engine").mkdir()
    (tmp_path / "engine" / "sglang.py").write_text(SGLANG)
    settings = tmp_path / "settings.json"
    env = {**os.environ, "PYTHONPATH": str(tmp_path / "engine"), "SETTINGS": str(settings)}
    subprocess.run([sys.executable, str(ROOT / "tools" / "benchmark_lm_serving.py"),
                    "--backend", "sglang-engine", "--model", "a/model", "--slots", "2,4", "--prompt", "8",
                    "--output", "5", "--vocab-limit", "200", "--repeats", "2", "--requests", "6",
                    "--out", str(tmp_path / "record.json")], check=True, env=env, cwd=ROOT)
    record = json.loads((tmp_path / "record.json").read_text())
    assert json.loads(settings.read_text()) == {
        "model_path": "a/model", "dtype": "bfloat16", "context_length": 13, "max_running_requests": 4,
        "mem_fraction_static": 0.9, "disable_radix_cache": True, "skip_tokenizer_init": True,
        "log_level": "error"}
    assert (record["backend"], record["transport"]) == ("sglang-engine", "in-process")
    assert [point["slots"] for point in record["sweep"]] == [2, 4]
    assert all(repeat["output_tokens"] == 6 * 5 for point in record["sweep"] for repeat in point["repeats"])
