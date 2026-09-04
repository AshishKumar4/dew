"""Multi-head latent attention parity against transformers 5.16.1.

The fixtures come from tools/mla_reference.py: tiny DeepseekV3Attention
blocks with and without the query LoRA (YaRN rope at the released
spelling, interleaved pairs), a tiny DeepseekV32Attention with its DSA
indexer and biased projections, and the YaRN derivation standalone.
Everything runs at fp32 on CPU, and each parity test states its tolerance
and the largest difference observed.
"""

import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop.hf_decoders import _yarn_record, translate_weights
from dew.nn.mla import (
    MultiHeadLatentAttention,
    YarnScaling,
    mla_rope_freqs,
    yarn_attention_factor,
    yarn_inv_freq,
    yarn_query_scale,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mla"
CONFIG = json.loads((FIXTURES / "config.json").read_text())


def fixture(name: str) -> dict:
    with np.load(FIXTURES / f"{name}.npz") as data:
        return {key: np.asarray(value) for key, value in data.items()}


def yarn_of(settings: dict) -> YarnScaling:
    """The fixture's rope spelling as the mixer's yarn record."""
    record = _yarn_record(
        dict(settings["rope_scaling"], rope_theta=settings["rope_theta"]),
        "rope_scaling", float(settings["rope_theta"]),
        int(settings["max_position_embeddings"]))
    return YarnScaling(**record)


def mla_module(settings: dict, max_seq_len: int = 64
               ) -> MultiHeadLatentAttention:
    """The fixture's reference config as a dew mixer."""
    return MultiHeadLatentAttention(
        emb_features=settings["hidden_size"],
        num_heads=settings["num_attention_heads"],
        max_seq_len=max_seq_len,
        q_lora_rank=settings["q_lora_rank"],
        kv_lora_rank=settings["kv_lora_rank"],
        qk_nope_head_dim=settings["qk_nope_head_dim"],
        qk_rope_head_dim=settings["qk_rope_head_dim"],
        v_head_dim=settings["v_head_dim"],
        rope_theta=float(settings["rope_theta"]),
        rope_interleave=settings.get("rope_interleave", True),
        yarn=yarn_of(settings),
        norm_eps=float(settings["rms_norm_eps"]),
        attention_bias=bool(settings["attention_bias"]),
        index_topk=settings.get("index_topk"),
        index_n_heads=settings.get("index_n_heads"),
        index_head_dim=settings.get("index_head_dim"))


def block_variables(tensors: dict) -> dict:
    """Block-relative fixture tensors as the mixer's parameter tree.

    The fixtures hold one attention block's tensors under their torch
    names; a checkpoint carries them under model.layers.N.self_attn.*,
    so the test prefixes that path and runs the real weight translation.
    """
    names = {f"model.layers.0.self_attn.{name}": tensor
             for name, tensor in tensors.items()
             if name not in ("hidden", "output")}
    tree = translate_weights(names, {"tie_embeddings": False})["params"]
    return {"params": tree["layers_0"]["self_attn"]}


def block_output(name: str, settings: dict) -> float:
    """Largest difference between the dew mixer and the fixture block."""
    tensors = fixture(name)
    module = mla_module(settings)
    output = module.apply(block_variables(tensors),
                          jnp.asarray(tensors["hidden"]))
    return float(np.max(np.abs(np.asarray(output) - tensors["output"])))


def test_yarn_matches_the_reference_derivation():
    """Inverse frequencies, cos/sin amplitude and positions, standalone."""
    tensors = fixture("yarn")
    yarn = yarn_of(CONFIG["v3"])

    np.testing.assert_allclose(
        np.asarray(yarn_inv_freq(8, 10000.0, yarn)), tensors["inv_freq"],
        rtol=1e-6, atol=1e-7)
    assert yarn_attention_factor(yarn) == pytest.approx(
        float(tensors["attention_scaling"]), rel=1e-6)
    cos, sin = mla_rope_freqs(
        jnp.arange(CONFIG["length"]), 8, 10000.0, yarn)
    half = tensors["cos"].shape[-1] // 2
    # The reference batches identical position rows; the first row compares.
    np.testing.assert_allclose(np.asarray(cos), tensors["cos"][0, ..., :half],
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sin), tensors["sin"][0, ..., :half],
                               rtol=1e-5, atol=1e-6)
    # (0.1 * ln(40) + 1) ** 2.
    assert yarn_query_scale(yarn) == pytest.approx(
        (0.1 * math.log(40.0) + 1.0) ** 2, rel=1e-6)


def decode_loop(module, variables, tokens):
    """Prefill plus one-token steps, the sampler's decode protocol."""
    state = module.init(jax.random.key(0), tokens[:, :1], decode=True)
    state = {**state, "params": variables["params"]}
    outputs = []
    for position in range(tokens.shape[1]):
        out, mutated = module.apply(
            state, tokens[:, position:position + 1],
            decode=True, mutable=["cache"])
        state = {**state, "cache": mutated["cache"]}
        outputs.append(out)
    return jnp.concatenate(outputs, axis=1)


SETTINGS = {"mla_v3": "v3", "mla_v3n": "v3n", "mla_v32": "v32"}


@pytest.mark.parametrize("name", ["mla_v3", "mla_v3n", "mla_v32"])
def test_decode_matches_prefill(name):
    """Tolerance 1e-5; observed 2.4e-6, 9.5e-7 and 1.9e-6."""
    settings = CONFIG[SETTINGS[name]]
    tensors = fixture(name)
    module = mla_module(settings)
    variables = block_variables(tensors)
    hidden = jnp.asarray(tensors["hidden"])
    full = module.apply(variables, hidden)
    assert np.max(np.abs(np.asarray(decode_loop(module, variables, hidden))
                         - np.asarray(full))) < 1e-5


def test_mla_reproduces_the_v3_block():
    """DeepseekV3Attention: q LoRA, compressed KV, interleaved YaRN rope.

    Tolerance 2e-5; observed 2.9e-6.
    """
    assert block_output("mla_v3", CONFIG["v3"]) < 2e-5


def test_mla_reproduces_the_v3_block_without_the_query_lora():
    """The plain-q_proj variant no released checkpoint uses, same reference.

    Tolerance 2e-5; observed 2.1e-6.
    """
    assert block_output("mla_v3n", CONFIG["v3n"]) < 2e-5


def test_mla_reproduces_the_v32_block():
    """DeepseekV32Attention: biased projections, indexer top-k mask fold.

    Tolerance 2e-5; observed 1.9e-6. The dense mixer on the same weights
    differs from the fixture by 3.4, so the fixture is the sparse block and
    the parity covers the selection; a generator that lost the eager mask
    fold again would fail here on the dense side.
    """
    assert block_output("mla_v32", CONFIG["v32"]) < 2e-5
    dense = dict(CONFIG["v32"], index_topk=None, index_n_heads=None,
                 index_head_dim=None)
    assert block_output("mla_v32", dense) > 1.0

