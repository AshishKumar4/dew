"""The serving benchmark (tools/benchmark_lm_serving.py), run end to end on the CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_as_exact_as_the_reference, float64_forward
from tool_support import load

from dew.interop import Pretrained
from dew.interop.pretrained import PretrainedDecoder
from dew.nn.backbones import CausalTransformer

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


def raw_truth(model: Path, prompts: np.ndarray, tokens: np.ndarray) -> np.ndarray:
    """Each generated token's log-probability under the model's own distribution, in float64."""
    loaded = Pretrained.load(model, dtype="float32", attention_impl="reference")
    logits, _ = float64_forward(loaded.model, loaded.variables, np.concatenate([prompts, tokens], 1))
    logits = np.asarray(logits, np.float64)[:, prompts.shape[1] - 1:-1]
    shifted = logits - logits.max(-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(-1, keepdims=True))
    return np.take_along_axis(log_probs, tokens[..., None].astype(np.int64), -1)[..., 0]


def test_a_model_served_over_a_mesh_decodes_what_it_decodes_with_its_weights_whole(tmp_path):
    """`--mesh` places the weights on that mesh before the server is built: the record names the
    mesh and the axis that splits them, which the default placement over the same two devices
    (data parallelism) leaves whole. Served either way in bf16, the greedy tokens agree, and the
    split run's log-probabilities are as near float64 as the whole run's."""
    model = checkpoint(tmp_path / "decoder")
    whole, generated = served(model, tmp_path / "whole.json")
    record, split = served(model, tmp_path / "split.json", "--mesh", '{"tensor": 2}')
    assert (whole["mesh"], whole["hardware"]["weights"]["split_over"]) == (None, [])
    assert (record["mesh"], record["hardware"]["weights"]) == ({"tensor": 2},
                                                               {"split_over": ["tensor"], "devices": 2})
    np.testing.assert_array_equal(split["tokens"], generated["tokens"])
    # Greedy decoding draws each token with certainty: both runs' behavior log-probabilities are zero.
    np.testing.assert_array_equal(split["behavior"], generated["behavior"])
    prompts = load("benchmark_lm_serving").prompts_for(SLOTS, REQUESTS, PROMPT, VOCABULARY)
    truth = raw_truth(model, prompts, generated["tokens"])
    assert_as_exact_as_the_reference(split["raw"], generated["raw"], truth, "raw log-probabilities")
