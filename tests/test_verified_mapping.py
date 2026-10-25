"""Tier 2 of loading: an unregistered model_type through a registered family's verified convention.

The offline cases write a tiny random checkpoint of a transformers class Dew
registers no family for and load it with `Pretrained.load`. CWM is the
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
same probe from its config, at 65 GB beyond what this suite downloads. Over
every registered family, `tools/hf_coverage.py` on the 400 most downloaded
text-generation repos (2026-10-05) finds one more type that loads:
GLM-4.7-Flash, as kimi_k2, which the offline GLM_FLASH case checks.
Measured on CPU against transformers 5.16.1 in fp32, over two rows of 64
random ids:

                             max |Δlogits|   bound (twice the measured)
  Dew fp32                   2.09e-07        4.2e-07
  Dew bfloat16 compute and
  storage                    4.02e-03        8.1e-03

on logits of magnitude 0.59.
"""

import json

import numpy as np
import pytest
import torch
from interop_support import tiny_config
from safetensors.numpy import load_file
from transformers import AutoModelForCausalLM

from dew.interop import Pretrained
from dew.interop.verify import VerifiedMappingWarning, probe_ids, reference_logits, scatter_weights

# CI runs this file in tasks of its own: torchax renames torch's privateuse1 backend process-wide.
pytestmark = pytest.mark.torchax


def write_tiny(directory, model_type, **fields):
    """Save a random tiny checkpoint of transformers' class for `model_type` (`tiny_config`); return
    its logits."""
    config = tiny_config(model_type, **fields)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    scatter_weights(model)
    model.save_pretrained(directory)
    ids = probe_ids(config.vocab_size)
    return ids, reference_logits(model, ids.astype(np.int64))


# GLM-4.7-Flash is DeepSeek V3's MLA and routed experts under its own name;
# its released config states as many key/value heads as query heads, and no
# head_dim: the rope is qk_rope_head_dim wide.
GLM_FLASH = {"num_key_value_heads": 4, "n_routed_experts": 4, "num_experts_per_tok": 2, "n_shared_experts": 1,
             "moe_intermediate_size": 64, "kv_lora_rank": 16, "q_lora_rank": 32, "qk_nope_head_dim": 16,
             "qk_rope_head_dim": 8, "v_head_dim": 16, "first_k_dense_replace": 1, "n_group": 1,
             "topk_group": 1, "head_dim": None}


@pytest.mark.parametrize("head_dim", ["absent", "null", "explicit"])
def test_an_optional_size_stated_null_states_nothing(head_dim):
    """CWM derives `head_dim` from its width and heads where its config leaves it out or states it
    null, and the probe reads both alike, as an explicit one, verifying the same convention."""
    from dew.interop.verify import VerifiedMapping, verify_mapping

    config = tiny_config("cwm", sliding_window=4).to_dict()
    config.pop("head_dim", None)
    if head_dim == "null":
        config["head_dim"] = None
    elif head_dim == "explicit":
        config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]
    assert isinstance(verify_mapping(config), VerifiedMapping)


@pytest.mark.parametrize("model_type, fields", [("cwm", {"sliding_window": 4}), ("glm4_moe_lite", GLM_FLASH)])
def test_a_verified_load_saves_the_source_config_and_reloads(tmp_path, model_type, fields):
    """A verified type is not its family: it saves under its own config and
    tensor names, which load as the same type again."""
    source, destination = tmp_path / "source", tmp_path / "export"
    ids, _ = write_tiny(source, model_type, **fields)
    with pytest.warns(VerifiedMappingWarning):
        loaded = Pretrained.load(source, dtype="float32", attention_impl="reference")
    loaded.save(destination)
    assert json.loads((destination / "config.json").read_text()) == loaded.config
    assert set(load_file(destination / "model.safetensors")) == set(load_file(source / "model.safetensors"))
    with pytest.warns(VerifiedMappingWarning):
        restored = Pretrained.load(destination, dtype="float32", attention_impl="reference")
    np.testing.assert_array_equal(np.asarray(restored.model.apply(restored.variables, ids)),
                                  np.asarray(loaded.model.apply(loaded.variables, ids)))
