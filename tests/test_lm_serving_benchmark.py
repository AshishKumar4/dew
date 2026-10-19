"""The serving benchmark (tools/benchmark_lm_serving.py), run end to end on the CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "tests" / "fixtures" / "hf" / "llama-tiny"
WIDTH, MLP, VOCABULARY, LAYERS = 256, 512, 512, 2
"""A Llama wide enough that its embeddings, projections and MLP reach the
layout's `min_shard` (2**16 elements), so a mesh splits them."""


def checkpoint(directory: Path) -> Path:
    """A random Llama checkpoint of `WIDTH`, llama-tiny's config at these widths."""
    config = json.loads((LLAMA / "config.json").read_text())
    config.update(hidden_size=WIDTH, intermediate_size=MLP, vocab_size=VOCABULARY,
                  num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2, head_dim=WIDTH // 4,
                  attention_bias=False)
    heads = WIDTH // 4
    shapes = {"model.embed_tokens.weight": (VOCABULARY, WIDTH), "model.norm.weight": (WIDTH,),
              "lm_head.weight": (VOCABULARY, WIDTH)}
    for layer in range(LAYERS):
        prefix = f"model.layers.{layer}."
        shapes |= {f"{prefix}self_attn.q_proj.weight": (4 * heads, WIDTH),
                   f"{prefix}self_attn.k_proj.weight": (2 * heads, WIDTH),
                   f"{prefix}self_attn.v_proj.weight": (2 * heads, WIDTH),
                   f"{prefix}self_attn.o_proj.weight": (WIDTH, 4 * heads),
                   f"{prefix}mlp.gate_proj.weight": (MLP, WIDTH),
                   f"{prefix}mlp.up_proj.weight": (MLP, WIDTH), f"{prefix}mlp.down_proj.weight": (WIDTH, MLP),
                   f"{prefix}input_layernorm.weight": (WIDTH,),
                   f"{prefix}post_attention_layernorm.weight": (WIDTH,)}
    generator = np.random.default_rng(0)
    weights = {name: (np.ones(shape) if len(shape) == 1 else generator.normal(0, 0.05, shape))
               .astype(np.float32) for name, shape in shapes.items()}
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(config))
    save_file(weights, directory / "model.safetensors")
    return directory


def served(model: Path, out: Path, *options: str) -> dict:
    """The benchmark's record, served on two CPU devices."""
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT / "src"),
           "XLA_FLAGS": "--xla_force_host_platform_device_count=2"}
    subprocess.run([sys.executable, str(ROOT / "tools" / "benchmark_lm_serving.py"), "--backend", "dew",
                    "--model", str(model), "--slots", "2", "--requests", "4", "--prompt", "8",
                    "--output", "8", "--vocab-limit", str(VOCABULARY), "--repeats", "1", "--out", str(out),
                    *options],
                   check=True, env=env, cwd=ROOT)
    return json.loads(out.read_text())


def test_the_benchmark_serves_weights_a_mesh_splits(tmp_path):
    """`--mesh` places the weights on that mesh before the server is built, as the record shows:
    the mesh, the devices the weights span and the axis that splits them, which the default
    placement over the same two devices (data parallelism) leaves whole. That a server on a mesh
    draws what one device draws is tests/test_serving.py's contract."""
    model = checkpoint(tmp_path / "llama")
    default = served(model, tmp_path / "default.json")
    record = served(model, tmp_path / "mesh.json", "--mesh", '{"tensor": 2}')
    assert (default["mesh"], default["hardware"]["weights"]["split_over"]) == (None, [])
    assert (record["mesh"], record["hardware"]["weights"]) == ({"tensor": 2},
                                                               {"devices": 2, "split_over": ["tensor"]})
    assert record["sweep"][0]["repeats"][0]["output_tokens"] == 4 * 8
