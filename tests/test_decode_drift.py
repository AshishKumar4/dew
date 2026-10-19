"""The decode drift gate (tools/decode_drift.py), run end to end on the CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import float64_forward

from dew.interop.pretrained import PretrainedDecoder
from dew.nn.backbones.causal_transformer import CausalTransformer

ROOT = Path(__file__).resolve().parents[1]
STEPS, PROMPT = 70, 8


def test_an_unchanged_decoder_meets_the_gate_on_the_tokens_it_decoded(tmp_path):
    """A decoder's greedy decode, its reference replays and its float64 truth: the truth the record holds is
    the model's float64 log-probability of each decoded token after the prompt and the tokens before it,
    computed here apart; the reference's replay is the decode it scored; and Dew's server, which decodes
    by the reference's own arithmetic on the CPU, passes the gate over the whole decode and each block."""
    model = CausalTransformer(vocab_size=256, emb_features=64, num_layers=2, num_heads=4, head_dim=16,
                              mlp_features=128, max_seq_len=128, dtype="float32")
    variables = model.init(jax.random.key(0), jnp.zeros((1, PROMPT), jnp.int32))
    PretrainedDecoder.from_model(model, variables).save(tmp_path / "decoder")
    out = tmp_path / "drift.json"
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "decode_drift.py"),
                           "--model", str(tmp_path / "decoder"), "--steps", str(STEPS), "--prompt", str(PROMPT),
                           "--vocab-limit", "200", "--out", str(out)],
                          env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT / "src"),
                               "XLA_FLAGS": "--xla_force_host_platform_device_count=1"}, cwd=ROOT)
    record = json.loads(out.read_text())
    scored = record["steps_scored"]
    prompt = np.random.default_rng(0).integers(0, 200, (4, PROMPT), dtype=np.int32)[0]
    sequence = np.concatenate([prompt, scored["tokens"]])[None]
    logits, _ = float64_forward(model, variables, jnp.asarray(sequence[:, :-1]))
    rows = np.asarray(logits, np.float64)[0, PROMPT - 1:]
    shifted = rows - rows.max(-1, keepdims=True)
    oracle = (shifted - np.log(np.exp(shifted).sum(-1, keepdims=True)))[np.arange(STEPS), scored["tokens"]]
    np.testing.assert_allclose(scored["truth"], oracle, rtol=1e-9, atol=1e-9)
    assert record["gate"]["whole"]["reference"] > 0, "the bf16 reference must sit off the truth to measure it"
    assert (done.returncode, record["passes"], record["gate"]["blocks_failed"]) == (0, True, [])
    assert record["reference_embedding_ulp"]["rms"] > 0
    assert record["reference_delta_rule"] is None
