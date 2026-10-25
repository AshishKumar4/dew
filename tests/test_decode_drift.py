"""The decode drift tool (tools/decode_drift.py), run end to end on the CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from reference_error import assert_computes_the_oracle, chain_roundings, widened
from tool_support import load

from dew.interop.pretrained import PretrainedDecoder
from dew.nn.backbones.causal_transformer import CausalTransformer

ROOT = Path(__file__).resolve().parents[1]
STEPS, PROMPT, SEEDS = 128, 8, 8


def test_an_unchanged_decoder_holds_check_1_over_its_long_decodes(tmp_path):
    """A decoder's greedy decodes of the tool's drawn prompts, Dew's server's log-probabilities held to the
    reference path's from float64 by check 1, each 64 steps a row: on the CPU, where the server decodes by
    the reference's own arithmetic, the verdict passes and the exit code says so. The record's first draw is
    the verdict's seed 0: its truth is the decoder's own float64 log-probability of each decoded token after
    the tool's prompt and the tokens before it, computed here apart, and the reference with its embedding
    one ulp up, which the record reports, moves."""
    model = CausalTransformer(vocab_size=256, emb_features=64, num_layers=2, num_heads=4, head_dim=16,
                              mlp_features=128, max_seq_len=256, dtype="float32")
    variables = model.init(jax.random.key(0), jnp.zeros((1, PROMPT), jnp.int32))
    PretrainedDecoder.from_model(model, variables).save(tmp_path / "decoder")
    out = tmp_path / "drift.json"
    done = subprocess.run([sys.executable, str(ROOT / "tools" / "decode_drift.py"),
                           "--model", str(tmp_path / "decoder"), "--steps", str(STEPS),
                           "--prompt", str(PROMPT), "--seeds", str(SEEDS), "--vocab-limit", "200",
                           "--out", str(out)],
                          env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(ROOT / "src"),
                               "XLA_FLAGS": "--xla_force_host_platform_device_count=1"}, cwd=ROOT)
    record = json.loads(out.read_text())
    assert (done.returncode, record["passes"], record["finding"]) == (0, True, None)
    assert record["reference_embedding_ulp"]["rms"] > 0, "the reference's embedding one ulp up moved nothing"
    tokens = np.asarray(record["steps_scored"]["tokens"])
    prompt = load("decode_drift").prompts(200, PROMPT, 0)[0]
    sequence = jnp.asarray(np.concatenate([prompt, tokens])[None, :-1])

    def scored(held):
        return jax.nn.log_softmax(model.clone(dtype=jnp.float64).apply(held, sequence), axis=-1)

    with jax.enable_x64(new_val=True):
        wide = widened(variables)
        roundings = chain_roundings(jax.make_jaxpr(scored)(wide))
        oracle = np.asarray(scored(wide))[0, PROMPT - 1:][np.arange(STEPS), tokens]
    assert_computes_the_oracle(np.asarray(record["steps_scored"]["truth"]), oracle, "the record's truth",
                               roundings=roundings)
