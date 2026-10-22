"""The decode drift gate (tools/decode_drift.py), run end to end on the CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from lane_environment import cpu_numerics
from reference_error import FACTOR, distance, float64_forward
from tool_support import load

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
                           "--model", str(tmp_path / "decoder"), "--steps", str(STEPS),
                           "--prompt", str(PROMPT), "--vocab-limit", "200", "--out", str(out)],
                          env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT / "src"),
                               "XLA_FLAGS": cpu_numerics("--xla_force_host_platform_device_count=1")},
                          cwd=ROOT)
    record = json.loads(out.read_text())
    scored = record["steps_scored"]
    prompt = np.random.default_rng(0).integers(0, 200, (4, PROMPT), dtype=np.int32)[0]
    sequence = np.concatenate([prompt, scored["tokens"]])[None]
    logits, _ = float64_forward(model, variables, jnp.asarray(sequence[:, :-1]))
    rows = np.asarray(logits, np.float64)[0, PROMPT - 1:]
    shifted = rows - rows.max(-1, keepdims=True)
    oracle = (shifted - np.log(np.exp(shifted).sum(-1, keepdims=True)))[np.arange(STEPS), scored["tokens"]]
    np.testing.assert_allclose(scored["truth"], oracle, rtol=1e-9, atol=1e-9)
    # The cached reference replay, scored apart by one bf16 forward of the same weights over the same tokens:
    # the replay no further from float64 than that forward, at every step from the first target to the last.
    bf16 = model.clone(dtype=jnp.bfloat16)
    narrow = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16), variables)
    rows = np.asarray(bf16.apply(narrow, jnp.asarray(sequence[:, :-1]))[0, PROMPT - 1:], np.float64)
    shifted = rows - rows.max(-1, keepdims=True)
    forward = (shifted - np.log(np.exp(shifted).sum(-1, keepdims=True)))[np.arange(STEPS), scored["tokens"]]
    assert distance(scored["reference"], oracle) <= FACTOR * distance(forward, oracle)
    assert record["gate"]["accuracy"]["reference"] > 0, "a reference at the truth measures nothing"
    assert (done.returncode, record["passes"]) == (0, True), record["gate"]
    assert record["reference_embedding_ulp"]["rms"] > 0
    assert record["reference_delta_rule"] is None


def test_the_gate_fails_a_server_further_from_float64_or_from_the_reference_than_it_allows():
    """The gate's two halves on arrays: a server twice as far from float64 as the reference fails the
    accuracy half while it sits within the reference's noise; a server within accuracy but further from
    the reference than the reference wavers fails the noise half; and a NaN fails both."""
    gate = load("decode_drift").gate
    rng = np.random.default_rng(0)
    true = rng.normal(-3, 1, 256)
    reference = true + rng.normal(0, 1e-3, 256)
    noise = np.full(256, 1e-2)
    assert gate(reference, reference, noise, true)["passes"]
    far = true + 2.5 * (reference - true)
    assert not gate(far, reference, noise, true)["accuracy"]["passes"]
    assert gate(far, reference, noise, true)["noise"]["passes"]
    quiet = np.full(256, 1e-4)
    assert not gate(true - (reference - true), reference, quiet, true)["noise"]["passes"]
    broken = reference.copy()
    broken[100] = np.nan
    assert not gate(broken, reference, noise, true)["passes"]
