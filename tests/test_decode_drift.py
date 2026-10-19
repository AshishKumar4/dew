"""The decode noise-floor gate (tools/decode_drift.py), run end to end on the CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

from dew.interop.pretrained import PretrainedDecoder
from dew.nn.backbones.causal_transformer import CausalTransformer

ROOT = Path(__file__).resolve().parents[1]


def test_the_gate_measures_the_server_against_the_references_own_noise(tmp_path):
    """A decoder's greedy decode, its reference replay at batch 1 and 4 and with its embedding one ulp
    up, and its prefill rescoring: the record holds each drift, the floor is the larger of the
    reference's two, and the verdict is the server's drift held to it."""
    model = CausalTransformer(vocab_size=256, emb_features=64, num_layers=2, num_heads=4, head_dim=16,
                              mlp_features=128, max_seq_len=64, dtype="float32")
    variables = model.init(jax.random.key(0), jnp.zeros((1, 8), jnp.int32))
    PretrainedDecoder.from_model(model, variables).save(tmp_path / "decoder")
    out = tmp_path / "drift.json"
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "decode_drift.py"),
                           "--model", str(tmp_path / "decoder"), "--steps", "24", "--prompt", "8",
                           "--vocab-limit", "200", "--out", str(out)],
                          env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT / "src")},
                          cwd=ROOT)
    record = json.loads(out.read_text())
    assert done.returncode == (0 if record["passes"] else 1)
    assert record["reference_embedding_ulp"]["rms"] > 0
    for key in ("rms", "max"):
        assert record["floor"][key] == max(record["reference_batch_4"][key],
                                           record["reference_embedding_ulp"][key])
    measured = record["server_from_reference"]
    assert record["passes"] == (measured["rms"] <= record["floor"]["rms"]
                                and measured["max"] <= record["floor"]["max"])
    assert record["greedy_paths_part_at"] is None or 0 <= record["greedy_paths_part_at"] < 24
