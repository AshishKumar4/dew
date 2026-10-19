"""The serving benchmark (tools/benchmark_lm_serving.py), run end to end on the CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "hf" / "llama-tiny"


def served(out: Path, *options: str, devices: int = 1) -> tuple[dict, np.ndarray]:
    """The benchmark's record and its first repeat's greedy tokens, on `devices` CPU devices."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT / "src"),
           "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}"}
    subprocess.run([sys.executable, str(ROOT / "tools" / "benchmark_lm_serving.py"), "--backend", "dew",
                    "--model", str(FIXTURE), "--slots", "2", "--prompt", "8", "--output", "8",
                    "--vocab-limit", "200", "--repeats", "1", "--generations", "--out", str(out), *options],
                   check=True, env=env, cwd=ROOT)
    return json.loads(out.read_text()), np.load(out.with_name(f"{out.stem}-slots2.npz"))["tokens"]


def test_a_model_served_over_a_mesh_decodes_the_tokens_it_decodes_on_one_device(tmp_path):
    """Weights placed over two devices (`--mesh`) serve the greedy tokens one device serves, and
    the record names the mesh and the devices the weights span."""
    alone, tokens = served(tmp_path / "one.json")
    record, sharded = served(tmp_path / "mesh.json", "--mesh", '{"fsdp": 2}', devices=2)
    assert (alone["mesh"], alone["hardware"]["weights_on"]) == (None, 1)
    assert (record["mesh"], record["hardware"]["weights_on"]) == ({"fsdp": 2}, 2)
    assert record["sweep"][0]["repeats"][0]["output_tokens"] == tokens.size
    np.testing.assert_array_equal(sharded, tokens)
