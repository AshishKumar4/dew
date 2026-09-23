"""Tier 2 of loading: an unregistered model_type through the verified Llama convention.

The offline cases write a tiny random checkpoint of a transformers class Dew
registers no family for and load it with `load_pretrained`. CWM is the
Llama block with per-layer sliding windows and a llama3 rope ramp, so it
loads, and its logits match transformers at the tier-1 fixtures' 1e-4
(tests/test_hf_decoders.py).
Granite scales its embeddings, residuals, attention and logits by config
fields the convention does not read. SmolLM3 drops the rotary on every fourth
layer, which a probe shrunk to two layers would never reach.

The network cases load real repos pinned to a commit. No unregistered type in
the top-200 census computes the Llama convention, so each census repo below
is refused, for its own reason, before its weights download. The one real
repo that loads is a random-weight CWM (Meta's Code World Model is the Llama
block with per-layer sliding windows); its released 32B checkpoint passes the
same probe from its config, at 65 GB beyond what this suite downloads.
Measured on CPU against transformers 5.16.1 in fp32, over two rows of 64
random ids:

                             max |Δlogits|   bound (twice the measured)
  Dew fp32                   2.09e-07        4.2e-07
  Dew bfloat16 compute and
  storage                    4.02e-03        8.1e-03

on logits of magnitude 0.59.
"""

import json
import sys
import warnings

import jax
import numpy as np
import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from dew.interop import load_pretrained
from dew.interop.verify import VerifiedMappingWarning, probe_ids, reference_logits, scatter_weights

TIER2 = "tier 2: verified mapping"
TORCHAX = 'fallback="torchax"'
TINY = {"hidden_size": 64, "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "intermediate_size": 128, "vocab_size": 256, "max_position_embeddings": 64,
        "tie_word_embeddings": False, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2}

# Pinned: each outcome below is this commit's config and weights.
CWM = ("hf-tiny-v2/tiny-random-CwmForCausalLM", "e0acc36e9533b85c5af948f7662edbd88012a2f3")
FP32, BF16 = 4.2e-7, 8.1e-3
REFUSED = {
    # Every fourth layer without rotary positions.
    "smollm3": ("HuggingFaceTB/SmolLM3-3B", "a07cc9a04f16550a088caea529712d1d335b0ac1",
                r"SmolLM3ForCausalLM and the convention disagree .* \['no_rope_layer_interval', 'no_rope_layers'\]"),
    # Embedding, residual, attention and logit multipliers.
    "granite": ("ibm-granite/granite-4.1-3b", "c0650403e44e78ec0262dab1c90914c65b196c4e",
                r"GraniteForCausalLM and the convention disagree .* 'residual_multiplier'"),
    # Llama's names and fields, a different rotary: the modeling alone differs.
    "ernie4_5": ("baidu/ERNIE-4.5-0.3B-PT", "b565cf6caebdb7a1eadf00100857b1ed5e044f12",
                 r"Ernie4_5ForCausalLM and the convention disagree"),
    # Phi-3's longrope and fused projections.
    "phi3": ("microsoft/Phi-4-mini-instruct", "cfbefacb99257ffa30c83adab238a50856ac3083",
             r"rope_type 'longrope'"),
    # OLMo 2's q/k norms over the whole projection and its post-norms.
    "olmo2": ("allenai/OLMo-2-0425-1B", "a1847dff35000b4271fa70afc5db10fd29fedbdf",
              r"Olmo2ForCausalLM holds \d+ parameters at the probe's sizes"),
}


def write_tiny(directory, model_type, **fields):
    """Save a random tiny checkpoint of transformers' class for `model_type`; return its logits."""
    config = AutoConfig.for_model(model_type, **{**TINY, **fields})
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    scatter_weights(model)
    model.save_pretrained(directory)
    ids = probe_ids(config.vocab_size)
    return ids, reference_logits(model, ids.astype(np.int64))


def test_a_llama_convention_type_loads_with_one_tier2_warning(tmp_path):
    ids, expected = write_tiny(tmp_path, "cwm", sliding_window=4,
                               layer_types=["sliding_attention", "full_attention"] * 2)
    with pytest.warns(VerifiedMappingWarning) as caught:
        loaded = load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    assert [str(entry.message).startswith(TIER2) for entry in caught] == [True]
    assert "transformers 5.16.1's CwmForCausalLM" in str(caught[0].message)
    actual = np.asarray(loaded.model.apply(loaded.variables, ids))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)


@pytest.mark.skipif(jax.default_backend() != "gpu", reason="TF32 matmuls exist on a GPU alone")
def test_the_probe_holds_its_bound_at_a_gpus_default_tf32_precision(tmp_path):
    """The bound was measured at fp32 matmul precision. A GPU's default runs
    fp32 matmuls in TF32, which puts 1e-2 between two correct models, so the
    probe computes Dew's side at the highest precision whatever the caller's.
    The load runs as a user's would, outside the suite's precision and XLA
    flags."""
    import os
    import subprocess

    write_tiny(tmp_path, "cwm", sliding_window=4)
    script = ("import sys\nfrom dew.interop import load_pretrained\n"
              "load_pretrained(sys.argv[1], dtype='float32', attention_impl='reference')")
    env = {name: value for name, value in os.environ.items()
           if name not in ("JAX_DEFAULT_MATMUL_PRECISION", "XLA_FLAGS")}
    run = subprocess.run([sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr[-1500:]
    assert TIER2 in run.stderr


def test_a_verified_load_saves_the_source_config_and_reloads(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "export"
    ids, _ = write_tiny(source, "cwm", sliding_window=4)
    with pytest.warns(VerifiedMappingWarning):
        loaded = load_pretrained(source, dtype="float32", attention_impl="reference")
    loaded.save(destination)
    assert json.loads((destination / "config.json").read_text()) == loaded.config
    with pytest.warns(VerifiedMappingWarning):
        restored = load_pretrained(destination, dtype="float32", attention_impl="reference")
    np.testing.assert_array_equal(np.asarray(restored.model.apply(restored.variables, ids)),
                                  np.asarray(loaded.model.apply(loaded.variables, ids)))


@pytest.mark.parametrize("model_type, fields", [
    ("granite", {"embedding_multiplier": 12.0, "residual_multiplier": 0.22,
                 "attention_multiplier": 0.015625, "logits_scaling": 8.0}),
    ("smollm3", {"no_rope_layer_interval": 4}),
])
def test_a_type_that_computes_something_else_is_refused_with_its_error(tmp_path, model_type, fields):
    write_tiny(tmp_path, model_type, **fields)
    with warnings.catch_warnings():
        warnings.simplefilter("error", VerifiedMappingWarning)
        with pytest.raises(ValueError, match=r"max \|Δlogits\| \S+ in fp32, over the bound") as refused:
            load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    assert TORCHAX in str(refused.value)


def test_without_torch_the_refusal_names_the_extra_and_the_generic_route(tmp_path, monkeypatch):
    write_tiny(tmp_path, "cwm")
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ValueError, match=r"pip install 'dew-ml\[torch\]'") as refused:
        load_pretrained(tmp_path, dtype="float32", attention_impl="reference")
    assert TORCHAX in str(refused.value)


@pytest.mark.network
def test_a_released_cwm_loads_verified_and_matches_transformers():
    repo, revision = CWM
    with pytest.warns(VerifiedMappingWarning, match=TIER2):
        fp32 = load_pretrained(repo, revision=revision, dtype="float32", attention_impl="reference")
    with pytest.warns(VerifiedMappingWarning, match=TIER2):
        bf16 = load_pretrained(repo, revision=revision, dtype="bfloat16", param_dtype="bfloat16",
                               attention_impl="reference")
    reference = AutoModelForCausalLM.from_pretrained(repo, revision=revision, dtype=torch.float32)
    ids = np.random.RandomState(0).randint(0, reference.config.vocab_size, (2, 64)).astype(np.int32)
    expected = reference_logits(reference, ids.astype(np.int64))
    np.testing.assert_allclose(np.asarray(fp32.model.apply(fp32.variables, ids)), expected,
                               atol=FP32, rtol=0)
    np.testing.assert_allclose(np.asarray(bf16.model.apply(bf16.variables, ids)).astype(np.float32),
                               expected, atol=BF16, rtol=0)


@pytest.mark.network
@pytest.mark.parametrize("model_type", sorted(REFUSED))
def test_a_census_repo_that_deviates_is_refused_before_its_weights(model_type):
    repo, revision, reason = REFUSED[model_type]
    with pytest.raises(ValueError, match=reason) as refused:
        load_pretrained(repo, revision=revision, dtype="float32")
    assert TORCHAX in str(refused.value)
