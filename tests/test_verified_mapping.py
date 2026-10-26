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
GLM-4.7-Flash, as kimi_k2, which the offline GLM_FLASH case checks. Its
classes registered as a type of their own verify as DeepSeek V3's convention
too, and the probe refuses each of two changes its eval logits over the
first positions would not show: a table that pads no row, and a rotary that
takes no ramp.
Measured on CPU against transformers 5.16.1 in fp32, over two rows of 64
random ids:

                             max |Δlogits|   bound (twice the measured)
  Dew fp32                   2.09e-07        4.2e-07
  Dew bfloat16 compute and
  storage                    4.02e-03        8.1e-03

on logits of magnitude 0.59.
"""

import copy
import json

import numpy as np
import pytest
import torch
from interop_support import tiny_config
from safetensors.numpy import load_file
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import Glm4MoeLiteConfig
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteRotaryEmbedding,
)

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
    """CwmConfig reads a null `head_dim` as its width over its heads and an absent one as its default,
    128 (configuration_cwm.py:82, :109-110): the probe reads a null as stating nothing, so a null and an
    explicit width over heads verify alike as the convention that computes them, and an absent one,
    which no convention reads as 128, is a type none computes (`Unmapped`), never a probe that cannot
    run."""
    from dew.interop.verify import Unmapped, VerifiedMapping, verify_mapping

    config = tiny_config("cwm", sliding_window=4).to_dict()
    config.pop("head_dim", None)
    if head_dim == "null":
        config["head_dim"] = None
    elif head_dim == "explicit":
        config["head_dim"] = config["hidden_size"] // config["num_attention_heads"]
    verdict = verify_mapping(config)
    assert isinstance(verdict, Unmapped if head_dim == "absent" else VerifiedMapping), verdict


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


class _FlashConfig(Glm4MoeLiteConfig):
    model_type = "registered_flash"


class _Flash(Glm4MoeLiteForCausalLM):
    """GLM-4.7-Flash's own causal LM, under a type of its own."""
    config_class = _FlashConfig


class _UnpaddedConfig(Glm4MoeLiteConfig):
    model_type = "unpadded_flash"


class _UnpaddedFlash(Glm4MoeLiteForCausalLM):
    """GLM-4.7-Flash whose table pads no row, whatever pad id its config states."""
    config_class = _UnpaddedConfig

    def __init__(self, config):
        super().__init__(config)
        self.model.embed_tokens.padding_idx = None


class _UngeneratingConfig(Glm4MoeLiteConfig):
    model_type = "ungenerating_flash"


class _UngeneratingFlash(Glm4MoeLiteForCausalLM):
    """GLM-4.7-Flash that does not generate, so its config's pad id is not a generation config's, which
    transformers holds nonnegative when it saves (generation/configuration_utils.py:879)."""
    config_class = _UngeneratingConfig

    @classmethod
    def can_generate(cls) -> bool:
        return False


class _RamplessConfig(Glm4MoeLiteConfig):
    model_type = "rampless_flash"


class _RamplessFlash(Glm4MoeLiteForCausalLM):
    """GLM-4.7-Flash whose rotary turns at the plain frequencies, whatever ramp its config states."""
    config_class = _RamplessConfig

    def __init__(self, config):
        super().__init__(config)
        plain = copy.deepcopy(config)
        plain.rope_parameters = {"rope_type": "default", "rope_theta": config.rope_parameters["rope_theta"]}
        self.model.rotary_emb = Glm4MoeLiteRotaryEmbedding(plain)


FLASH_TYPES = {"flash": (_FlashConfig, _Flash), "unpadded": (_UnpaddedConfig, _UnpaddedFlash),
               "ungenerating": (_UngeneratingConfig, _UngeneratingFlash),
               "rampless": (_RamplessConfig, _RamplessFlash)}
"""GLM-4.7-Flash's classes as types Dew registers no family for: unchanged, and changed where the
probe's eval logits over its first positions do not show it."""


def _registered(kind: str, **fields) -> dict:
    """The `kind` of `FLASH_TYPES` registered as transformers registers a new model type (its config and
    causal LM through the auto classes, and GLM-4.7-Flash's checkpoint conversion, one tensor per expert,
    which transformers keys by model type), and its config at GLM_FLASH's sizes and `fields`."""
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )

    config_class, model_class = FLASH_TYPES[kind]
    AutoConfig.register(config_class.model_type, config_class, exist_ok=True)
    AutoModelForCausalLM.register(config_class, model_class, exist_ok=True)
    register_checkpoint_conversion_mapping(config_class.model_type,
                                           get_checkpoint_conversion_mapping("glm4_moe_lite"), overwrite=True)
    return tiny_config(config_class.model_type, **GLM_FLASH, **fields).to_dict()


@pytest.mark.parametrize(("kind", "pad"), [("flash", 0), ("ungenerating", -1), ("unpadded", 0)])
def test_a_convention_pads_the_row_its_source_pads(kind, pad):
    """GLM-4.7-Flash's classes under a type of their own verify as DeepSeek V3's convention, and the same
    classes whose table pads no row, under a config stating a pad id, do not: the probe holds each
    candidate's pad row to the source's own table's, which takes no gradient there (ledger 1593) and
    which no eval logit shows. A negative pad id, which a source that does not generate may state, names
    the row it counts back from the table's end, as torch's lookup and Dew's both read it, so -1 is the
    last row in each."""
    from dew.interop.verify import Unmapped, VerifiedMapping, verify_mapping

    config = _registered(kind, pad_token_id=pad)
    assert isinstance(verify_mapping(config), Unmapped if kind == "unpadded" else VerifiedMapping)


FAR_RAMP = {"rope_type": "yarn", "rope_theta": 1e6, "factor": 40.0, "original_max_position_embeddings": 2**28,
            "beta_fast": 32.0, "beta_slow": 1.0, "mscale": 1.0, "mscale_all_dim": 1.0}
"""A YaRN record of amplitude one whose stated original context is past every wavelength of the
probe's rotary, so that at it the ramp rescales no frequency the probe turns."""


@pytest.mark.parametrize("kind", ["flash", "rampless"])
def test_a_convention_turns_the_ramp_its_source_turns_past_the_original_context(kind):
    """Under a YaRN record, GLM-4.7-Flash's classes under a type of their own verify as DeepSeek V3's
    convention, and the same classes whose rotary turns at the plain frequencies do not: the probe runs
    past the record's original context, shrunk below its length as a window is, where the ramp rescales
    the frequencies it turns."""
    from dew.interop.verify import Unmapped, VerifiedMapping, verify_mapping

    config = _registered(kind, rope_parameters=FAR_RAMP, max_position_embeddings=2**29)
    assert isinstance(verify_mapping(config), Unmapped if kind == "rampless" else VerifiedMapping)
