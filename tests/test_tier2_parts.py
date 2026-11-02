"""Tier 2 from parts (ledger 1607): an unregistered decoder whose every module is a registered family's
part by its code loads natively, each part read from the source's own modeling code.

Each source here is a repository of remote code, loaded only under `trust_remote_code=True`: a Llama
whose code renames every class and attribute (`renamed`), and a decoder built of Llama's attention,
Mixtral's routed experts and Gemma's (1 + w) norms (`mixed`). Each loads natively and its logits hold
check 1 against its own code run by transformers, from that code's float64 twin.
"""
import json

import jax
import numpy as np
import pytest
import torch
from interop_support import assert_pads_as, clearing_selections, token_draw, torch_logits
from reference_error import as_given, assert_as_exact

from dew.interop import Pretrained, parts
from dew.interop.verify import VerifiedMapping, scatter_weights, verify_mapping
from tools.remote_code import renamed_module
from tools.torch_precision import float64, float64_twin

# CI runs this file in tasks of its own beside the other tier-2 files: tier 2's probe runs torch.
pytestmark = pytest.mark.torchax

TINY = {"hidden_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "intermediate_size": 96, "vocab_size": 256, "max_position_embeddings": 64,
        "tie_word_embeddings": False, "pad_token_id": 0, "bos_token_id": 1, "eos_token_id": 2}
"""Each source's sizes, which the probe shrinks no further."""

RENAMES = {"q_proj": "wq", "k_proj": "wk", "v_proj": "wv", "o_proj": "wo", "gate_proj": "w1",
           "up_proj": "w3", "down_proj": "w2", "self_attn": "attention", "input_layernorm": "attention_norm",
           "post_attention_layernorm": "ffn_norm", "mlp": "feed_forward", "embed_tokens": "tok_embeddings"}
"""The renamed Llama's attribute names, by Llama's."""

MIXED = '''
from torch import nn
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM, LlamaModel
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

from .configuration_mixed import MixedConfig


class MixedDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self.mlp = MixtralSparseMoeBlock(config)
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class MixedModel(LlamaModel):
    config_class = MixedConfig

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(MixedDecoderLayer(config, index)
                                    for index in range(config.num_hidden_layers))
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_init()


class MixedForCausalLM(LlamaForCausalLM):
    config_class = MixedConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MixedModel(config)
        self.post_init()
'''
"""A decoder of Llama's attention, Mixtral's routed experts and Gemma's norms, as remote code."""

MIXED_CONFIG = '''
from transformers.models.llama.configuration_llama import LlamaConfig


class MixedConfig(LlamaConfig):
    model_type = "mixed_parts"
    num_local_experts: int = 4
    num_experts_per_tok: int = 2
    router_jitter_noise: float = 0.0
'''

RENAMED_CONFIG = '''
from transformers.models.llama.configuration_llama import LlamaConfig


class RenamedConfig(LlamaConfig):
    model_type = "renamed_llama"
'''


def _renamed_llama() -> str:
    """transformers' Llama modeling as a repository's own module: every class and attribute renamed
    (`RENAMES`, the classes' `Llama` prefix to `Renamed`), its imports absolute, its code the same, and
    the table's attribute, which the base class finds by name, declared."""
    from transformers.models.llama import modeling_llama

    source = renamed_module(modeling_llama, ("Llama", "Renamed"), RENAMES,
                            ("from .configuration_llama import LlamaConfig",
                             "from .configuration_renamed import RenamedConfig"))
    return source.replace("class RenamedPreTrainedModel(PreTrainedModel):",
                          "class RenamedPreTrainedModel(PreTrainedModel):\n"
                          "    _input_embed_layer = \"tok_embeddings\"", 1)


MOE_RENAMES = {"experts": "moe_experts", "gate": "router", "embed_tokens": "tok_embeddings"}
"""The renamed Mixtral's attribute names, by Mixtral's: its experts module, whose fused tensors are
stored under it, its router and its table, which its head is tied to (`FIELDS`). The experts' own
tensors keep their names, which transformers' experts implementations read."""

RENAMED_MOE_CONFIG = '''
from transformers.models.mixtral.configuration_mixtral import MixtralConfig


class RenamedMoeConfig(MixtralConfig):
    model_type = "renamed_moe"
    num_local_experts: int = 4
'''


def _renamed_mixtral() -> str:
    """transformers' Mixtral modeling as a repository's own module, its classes and the attributes
    `MOE_RENAMES` names renamed, its head and table found by their new names."""
    from transformers.models.mixtral import modeling_mixtral

    source = renamed_module(modeling_mixtral, ("Mixtral", "Renamedmoe"), MOE_RENAMES,
                            ("from .configuration_mixtral import MixtralConfig",
                             "from .configuration_renamedmoe import RenamedMoeConfig as RenamedmoeConfig"))
    return source.replace("class RenamedmoePreTrainedModel(PreTrainedModel):",
                          "class RenamedmoePreTrainedModel(PreTrainedModel):\n"
                          "    _input_embed_layer = \"tok_embeddings\"", 1)


SOURCES = {
    "renamed": ("renamed_llama", "renamed", _renamed_llama, RENAMED_CONFIG, "RenamedConfig",
                "RenamedForCausalLM"),
    "mixed": ("mixed_parts", "mixed", lambda: MIXED, MIXED_CONFIG, "MixedConfig", "MixedForCausalLM"),
    "renamed-moe": ("renamed_moe", "renamedmoe", _renamed_mixtral, RENAMED_MOE_CONFIG, "RenamedMoeConfig",
                    "RenamedmoeForCausalLM"),
}
"""Each source: its model_type, its modules' stem (`modeling_<stem>.py` and `configuration_<stem>.py`), its
modeling code, its configuration code, and its config and causal LM classes."""

FIELDS = {"renamed-moe": {"tie_word_embeddings": True}}
"""Each source's config fields beside `TINY`."""

SEEDS = {"renamed": 8, "mixed": 32, "renamed-moe": 8}
"""Each source's declared check-1 seeds for its logits, sized by `seeds_for` on armada."""


def write_source(directory, name: str, edit=lambda code: code) -> None:
    """The source's repository at `directory`: its code (`edit` of its modeling code), its config naming
    its classes by `auto_map`, and its weights at the probe's scatter (`scatter_weights`), saved by its
    own class."""
    from transformers import AutoConfig, AutoModelForCausalLM

    model_type, stem, modeling, configuration, config_class, model_class = SOURCES[name]
    directory.mkdir()
    (directory / f"modeling_{stem}.py").write_text(edit(modeling()))
    (directory / f"configuration_{stem}.py").write_text(configuration)
    (directory / "config.json").write_text(json.dumps({
        **TINY, **FIELDS.get(name, {}), "model_type": model_type,
        "auto_map": {"AutoConfig": f"configuration_{stem}.{config_class}",
                     "AutoModelForCausalLM": f"modeling_{stem}.{model_class}"}}))
    config = AutoConfig.from_pretrained(directory, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, dtype=torch.float32)
    scatter_weights(model)
    model.save_pretrained(directory)
    # save_pretrained writes the module of the class the auto class built over the repository's own,
    # and the auto class builds the class registered for the config's: within one process, that is
    # an earlier repository's of the same folder name and configuration code, whose `edit` may differ.
    (directory / f"modeling_{stem}.py").write_text(edit(modeling()))


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_a_model_of_known_parts_loads_natively_from_its_own_code(tmp_path, name):
    """The source verifies from parts (no registered family's whole convention computes it), loads as a
    native decoder, and its logits over drawn ids hold check 1 against its own code's."""
    from transformers import AutoModelForCausalLM

    directory = tmp_path / name
    write_source(directory, name)
    config = json.loads((directory / "config.json").read_text())
    verified = verify_mapping(config, remote=directory)
    assert isinstance(verified, VerifiedMapping) and verified.family == "parts"
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference", trust_remote_code=True)
    forward = jax.jit(loaded.model.apply)

    def built(dtype):
        return AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=True, dtype=dtype,
                                                    attn_implementation="eager",
                                                    experts_implementation="eager").eval()

    reference = built(torch.float32)
    wide = float64_twin(reference, lambda: built(torch.float64)).eval()

    # A routed source's draws clear its routers' margin, so every run routes alike.
    draw = clearing_selections(loaded, token_draw(0x1607, TINY["vocab_size"], (4, 24)), name)
    assert_as_exact(as_given(lambda ids: np.asarray(forward(loaded.variables, ids), np.float64)),
                    torch_logits(reference), draw, f"{name} logits", seeds=SEEDS[name],
                    truth_fn=torch_logits(wide, float64))


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_a_model_loaded_from_parts_pads_the_row_its_own_table_pads(tmp_path, name):
    """A decoder assembled from parts reads its pad row off its source's own table, the nn.Embedding
    the code builds with padding_idx=config.pad_token_id (ledger 1593), and its lookup's gradient is
    that table's (`assert_pads_as`)."""
    from transformers import AutoModelForCausalLM

    directory = tmp_path / name
    write_source(directory, name)
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference", trust_remote_code=True)

    def built(dtype):
        return AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=True, dtype=dtype).eval()

    reference = built(torch.float32)
    wide = float64_twin(reference, lambda: built(torch.float64))
    assert reference.get_input_embeddings().padding_idx == TINY["pad_token_id"]
    assert_pads_as(loaded.model, loaded.variables, reference.get_input_embeddings(),
                   wide.get_input_embeddings(), TINY["pad_token_id"], name, seeds=8)


def test_a_remote_type_is_refused_without_its_code(tmp_path):
    """A type transformers has no class for runs its repository's code only where the load opts in."""
    write_source(tmp_path / "renamed", "renamed")
    with pytest.raises(ValueError):
        Pretrained.load(tmp_path / "renamed", dtype="float32", attention_impl="reference")


def test_a_part_whose_code_differs_is_not_assembled(tmp_path):
    """One statement changed in one part's code (the renamed norm's epsilon added twice) leaves the
    source a part no known one computes, so tier 2 refuses its remote code rather than assembling it."""
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

    def doubled(code):
        changed = code.replace("variance + self.variance_epsilon", "variance + 2 * self.variance_epsilon", 1)
        assert changed != code
        return changed

    # A directory of its own name, so transformers imports this code, not the unchanged module of that name.
    write_source(tmp_path / "doubled", "renamed", doubled)
    config = AutoConfig.from_pretrained(tmp_path / "doubled", trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    assert isinstance(model, PreTrainedModel)
    with pytest.raises(parts.Ambiguous):
        parts.assemble(model)
    with pytest.raises(ValueError):
        verify_mapping(json.loads((tmp_path / "doubled" / "config.json").read_text()),
                       remote=tmp_path / "doubled")


_LOADED = """
import sys
from dew.interop import Pretrained
Pretrained.load(sys.argv[1], subfolder=sys.argv[2] or None, dtype="float32", attention_impl="reference",
                trust_remote_code=True)
"""
"""A load of the source `argv[1]` names, at the folder `argv[2]` selects, running its own code."""


def _offline_load(root, source, subfolder: str = ""):
    """A load of `source` (at `subfolder`) in a process of its own, offline over the Hub cache at
    `root`/hub with Dew's cache at `root`/cache."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "HF_HUB_CACHE": str(root / "hub"), "HF_HUB_OFFLINE": "1",
           "XDG_CACHE_HOME": str(root / "cache"), "JAX_PLATFORMS": "cpu"}
    return subprocess.run([sys.executable, "-c", _LOADED, str(source), subfolder], capture_output=True,
                          text=True, env=env)


@pytest.mark.whole_suite
def test_a_verification_of_remote_code_is_recorded_under_its_commit(tmp_path):
    """A passed verification of a repository's own code is recorded under the Hub commit the load
    fetched that code at, beside the config, transformers, torch and Dew it ran under, so the same code
    at another commit is verified and recorded again; a local directory's code, which no commit names,
    is verified at each load and recorded under none. Each load runs offline over a Hub cache holding
    the repository at a commit, in a process of its own."""
    from interop_support import hub_cache

    write_source(tmp_path / "repo", "renamed")
    # The same code, under two repositories at two commits.
    repos = [f"dew/renamed-{index}" for index in range(2)]
    for index, repo in enumerate(repos):
        hub_cache(tmp_path / "hub", repo, tmp_path / "repo", commit=f"{index}" * 40)
    verified = tmp_path / "cache" / "dew" / "verified"

    def load(source) -> None:
        run = _offline_load(tmp_path, source)
        assert run.returncode == 0, run.stderr[-2000:]

    load(tmp_path / "repo")
    assert not list(verified.glob("*/mapping.json"))
    for count, repo in enumerate(repos, 1):
        load(repo)
        assert len(list(verified.glob("*/mapping.json"))) == count


def test_each_code_folder_at_one_commit_is_verified_on_its_own(tmp_path):
    """A commit names a whole repository, whose folders (`subfolder`) may each hold code of their own:
    two folders at one commit, each beside the same config, one the renamed Llama and the other the same
    code that trains otherwise than it evaluates (`_hooked`), are verified each, so the second is refused
    after the first's verification is recorded, as it is refused alone."""
    from interop_support import hub_cache

    (tmp_path / "repo").mkdir()
    write_source(tmp_path / "repo" / "plain", "renamed")
    write_source(tmp_path / "repo" / "hooked", "renamed", _hooked)
    configs = [json.loads((tmp_path / "repo" / folder / "config.json").read_text())
               for folder in ("plain", "hooked")]
    assert configs[0] == configs[1]
    hub_cache(tmp_path / "hub", "dew/renamed", tmp_path / "repo")
    plain = _offline_load(tmp_path, "dew/renamed", "plain")
    assert plain.returncode == 0, plain.stderr[-2000:]
    assert _offline_load(tmp_path, "dew/renamed", "hooked").returncode != 0


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_a_model_loaded_from_parts_saves_under_its_own_names_and_reloads(tmp_path, name):
    """A decoder assembled from parts saves under its source's config and tensor names (each tensor at
    its stored name, the experts fused as stored), which load from parts again to the same logits."""
    from safetensors.numpy import load_file

    source, destination = tmp_path / name, tmp_path / "export"
    write_source(source, name)
    loaded = Pretrained.load(source, dtype="float32", attention_impl="reference", trust_remote_code=True)
    loaded.save(destination)
    assert set(load_file(destination / "model.safetensors")) == set(load_file(source / "model.safetensors"))
    restored = Pretrained.load(destination, dtype="float32", attention_impl="reference",
                               trust_remote_code=True)
    ids = np.random.default_rng(1607).integers(0, TINY["vocab_size"], (2, 12)).astype(np.int32)
    np.testing.assert_array_equal(np.asarray(restored.model.apply(restored.variables, ids)),
                                  np.asarray(loaded.model.apply(loaded.variables, ids)))


def test_a_helper_whose_default_differs_is_not_the_known_one(tmp_path):
    """A helper the known forward calls without one of its arguments computes by that argument's
    default (`apply_rotary_pos_emb`'s `unsqueeze_dim`), so a copy whose default differs is another
    computation, and the part calling it is not assembled."""
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

    def moved(code):
        changed = code.replace("def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)",
                               "def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2)", 1)
        assert changed != code
        return changed

    write_source(tmp_path / "moved", "renamed", moved)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(tmp_path / "moved", trust_remote_code=True), trust_remote_code=True)
    assert isinstance(model, PreTrainedModel)
    with pytest.raises(parts.Ambiguous):
        parts.assemble(model)


def test_a_built_in_type_loads_as_it_does_whether_remote_code_is_trusted(tmp_path):
    """Trusting remote code permits a repository's own classes; it does not require them, so a type
    transformers itself builds verifies as it does without the flag."""
    from interop_support import tiny_config
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_config(tiny_config("cwm", sliding_window=4), dtype=torch.float32)
    scatter_weights(model)
    model.save_pretrained(tmp_path)
    config = json.loads((tmp_path / "config.json").read_text())
    trusted, plain = verify_mapping(config, remote=tmp_path), verify_mapping(config)
    assert isinstance(trusted, VerifiedMapping) and isinstance(plain, VerifiedMapping)
    assert (trusted.family, trusted.source) == (plain.family, None)


def _hooked(code: str) -> str:
    """The renamed Llama whose causal LM, after building its parts, registers a forward hook on each
    query projection that doubles its output in training alone."""
    changed = code.replace(
        "        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)\n",
        "        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)\n"
        "        for layer in self.model.layers:\n"
        "            layer.attention.wq.register_forward_hook(\n"
        "                lambda module, inputs, output: 2 * output if module.training else output)\n", 1)
    assert changed != code
    return changed


def test_a_training_only_behavior_is_not_assembled(tmp_path):
    """A source whose parts are the known ones by their code but which trains otherwise than it
    evaluates (a hook doubling a projection in training) is refused, whatever its eval logits."""
    write_source(tmp_path / "hooked", "renamed", _hooked)
    config = json.loads((tmp_path / "hooked" / "config.json").read_text())
    with pytest.raises(ValueError):
        verify_mapping(config, remote=tmp_path / "hooked")


def test_an_embedding_dews_lookup_does_not_compute_is_refused_at_load(tmp_path):
    """A source whose table scales its gradient by token frequency, its tensors under Llama's own
    names, is refused by the load whatever convention its eval logits would match: remote code is
    admitted by its code alone."""
    def scaled(code):
        changed = code.replace("nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)",
                               "nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx, "
                               "scale_grad_by_freq=True)", 1)
        assert changed != code
        return changed

    write_source(tmp_path / "scaled", "renamed", scaled)
    with pytest.raises(ValueError):
        Pretrained.load(tmp_path / "scaled", dtype="float32", attention_impl="reference",
                        trust_remote_code=True)


BUILT_IN = {
    "granite": {"embedding_multiplier": 12.0, "residual_multiplier": 0.22, "attention_multiplier": 0.0078125,
                "logits_scaling": 8.0},
    "smollm3": {},
    "olmo2": {},
    "hy_v3": {"num_experts": 8, "num_experts_per_tok": 2, "num_shared_experts": 1,
              "moe_intermediate_size": 32, "router_scaling_factor": 2.826,
              "num_nextn_predict_layers": 1},
    "glm4_moe_lite": {"head_dim": None, "num_key_value_heads": 4, "n_routed_experts": 8,
                      "num_experts_per_tok": 2, "n_shared_experts": 1, "moe_intermediate_size": 32,
                      "q_lora_rank": 24, "kv_lora_rank": 16, "qk_nope_head_dim": 16, "qk_rope_head_dim": 8,
                      "v_head_dim": 16, "n_group": 2, "topk_group": 1, "routed_scaling_factor": 1.8,
                      "num_nextn_predict_layers": 1},
    "olmoe": {"num_experts": 8, "num_experts_per_tok": 2},
    "stablelm": {"head_dim": None, "num_key_value_heads": 4, "use_qkv_bias": True,
                 "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                     "partial_rotary_factor": 0.25}},
    "stablelm-parallel": {"model_type": "stablelm", "head_dim": None, "use_parallel_residual": True,
                          "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                              "partial_rotary_factor": 0.5}},
    "hunyuan_v1_moe": {"num_experts": 8, "moe_topk": 2,
                       "rope_parameters": {"rope_type": "dynamic", "rope_theta": 10000.0, "alpha": 1000.0,
                                           "factor": 1.0}},
    "ernie4_5_moe": {"head_dim": None, "moe_num_experts": 8, "moe_k": 2, "moe_intermediate_size": 32,
                     "moe_num_shared_experts": 1, "moe_layer_start_index": 1},
    "lfm2": {"head_dim": None, "layer_types": ["conv", "full_attention", "conv", "full_attention"],
             "conv_L_cache": 3, "block_auto_adjust_ff_dim": False},
    "lfm2_moe": {"head_dim": None, "layer_types": ["conv", "full_attention", "conv", "full_attention"],
                 "conv_L_cache": 3, "num_dense_layers": 1, "num_experts": 8, "num_experts_per_tok": 2,
                 "moe_intermediate_size": 32, "routed_scaling_factor": 1.7},
}
"""transformers' own types that no registered family reads, each made of known parts, by their fields beside
`tiny_config`'s: Granite's multipliers, SmolLM3's layers without positions (its fourth), OLMo 2's
output-normed sublayers, HY-3's routed experts beside a shared one, and GLM-4.7-Flash's latent attention
(DeepSeek V3's, whose `head_dim` names the rotated slice, so it is left to the config) with its grouped
sigmoid router, each of these two's first layer dense and the others routed, and stating one prediction
depth, which their releases store past the trunk (`with_prediction_depth`); OLMoE's every layer routed
under OLMo 2's whole-projection query and key norms; StableLM's layer-normed blocks turning a part of
each head, in sequence or (`stablelm-parallel`) beside each other off one norm; and LFM2's gated short
convolutions between its attention layers, its modules under names of their own (`operator_norm`,
`feed_forward.w1`, `out_proj`), and LFM2-MoE's, its layers past the first routed by sigmoid scores and a
balancing bias, renormalized over a sum with its own epsilon; and ERNIE 4.5's routed layers past its
first, a softmax router selecting with a `[1, experts]` correction bias beside a shared expert, its
rotary turning adjacent pairs; and Hunyuan's every layer routed beside a shared MLP, each head's queries
and keys normed after their rotation, at its dynamic-alpha base."""

BUILT_IN_SEEDS = {"granite": 8, "smollm3": 8, "olmo2": 8, "hy_v3": 8, "glm4_moe_lite": 8, "olmoe": 8,
                  "stablelm": 8, "stablelm-parallel": 8, "lfm2": 8, "lfm2_moe": 8, "ernie4_5_moe": 8,
                  "hunyuan_v1_moe": 8}
"""Each built-in type's declared check-1 seeds for its logits, sized by `seeds_for` on armada."""

STORED_EXPERT = {"hy_v3": "model.layers.1.mlp.experts.0.gate_proj.weight",
                 "glm4_moe_lite": "model.layers.1.mlp.experts.0.gate_proj.weight",
                 "olmoe": "model.layers.1.mlp.experts.0.gate_proj.weight",
                 "lfm2_moe": "model.layers.1.feed_forward.experts.0.w1.weight",
                 "ernie4_5_moe": "model.layers.1.mlp.experts.0.gate_proj.weight",
                 "hunyuan_v1_moe": "model.layers.1.mlp.experts.0.gate_proj.weight"}
"""A routed type's first stored expert projection, as `save_pretrained` stores its experts one by one."""


def with_prediction_depth(directory, layers: int, width: int, extra: tuple[str, ...] = (), *,
                          fused: bool = False) -> None:
    """The checkpoint at `directory` with a prediction depth past its `layers` trunk layers, as
    GLM-4.7-Flash's and HY-3's releases store one (`model.layers.{layers}.*`): its normed inputs, their
    projection, its output norm and a block, and GLM-4.7-Flash's copies of the trunk's table and head,
    beside the trunk the source's own save writes; `extra` names more tensors under it. `fused` stores
    the depth's experts as transformers' fused module holds them (`mlp.experts.gate_up_proj`,
    `mlp.experts.down_proj`), not one by one."""
    import re

    from safetensors.torch import load_file, save_file

    path = directory / "model.safetensors"
    tensors = load_file(path)
    last, depth = f"model.layers.{layers - 1}.", f"model.layers.{layers}."
    block = {depth + name.removeprefix(last): tensor.clone() for name, tensor in tensors.items()
             if name.startswith(last)}
    if fused:
        experts = {}
        for name in list(block):
            if (found := re.fullmatch(rf"{re.escape(depth)}mlp\.experts\.(\d+)\.(\w+)\.weight", name)):
                experts.setdefault(found.group(2), {})[int(found.group(1))] = block.pop(name)
        count = len(experts["gate_proj"])
        block[depth + "mlp.experts.gate_up_proj"] = torch.stack(
            [torch.cat([experts["gate_proj"][index], experts["up_proj"][index]]) for index in range(count)])
        block[depth + "mlp.experts.down_proj"] = torch.stack([experts["down_proj"][index]
                                                             for index in range(count)])
    tensors.update(block)
    generator = torch.Generator().manual_seed(7)
    tensors.update({depth + "eh_proj.weight": torch.randn(width, 2 * width, generator=generator),
                    depth + "enorm.weight": torch.ones(width), depth + "hnorm.weight": torch.ones(width),
                    depth + "shared_head.norm.weight": torch.ones(width),
                    depth + "embed_tokens.weight": tensors["model.embed_tokens.weight"].clone(),
                    depth + "shared_head.head.weight": tensors["lm_head.weight"].clone(),
                    **{depth + name: torch.ones(width) for name in extra}})
    save_file(tensors, path, metadata={"format": "pt"})


def test_a_released_prediction_depth_is_left_unread_and_its_copies_follow_the_trunk(tmp_path):
    """GLM-4.7-Flash's release stores a prediction depth past its trunk, which its own code builds none
    of and its load leaves unread: a load reads the trunk alone, whether it keeps the depth or not
    (`prediction_depths=0`), and computes the same logits; the depth's copies of the table and head are
    the trunk's, so a save after the table changes writes the copy that reloads. A tensor under the depth
    that is none of its own and no block's is refused; a depth stored with fused experts loads
    too, its fused tensors kept whole (ledger 3141)."""
    from interop_support import tiny_config
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    config = tiny_config("glm4_moe_lite", **BUILT_IN["glm4_moe_lite"])
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    scatter_weights(model)
    model.save_pretrained(tmp_path / "release")
    with_prediction_depth(tmp_path / "release", config.num_hidden_layers, config.hidden_size)
    kept = Pretrained.load(tmp_path / "release", dtype="float32", attention_impl="reference")
    trunk = Pretrained.load(tmp_path / "release", dtype="float32", attention_impl="reference",
                            prediction_depths=0)
    ids = np.random.default_rng(0).integers(0, config.vocab_size, (2, 12)).astype(np.int32)
    np.testing.assert_array_equal(np.asarray(kept.model.apply(kept.variables, ids)),
                                  np.asarray(trunk.model.apply(trunk.variables, ids)))
    table = kept.variables["params"]["embed_tokens"]["embedding"]
    kept.save(tmp_path / "saved", variables={**kept.variables, "params": {
        **kept.variables["params"], "embed_tokens": {"embedding": np.asarray(table) * 2}}})
    reloaded = Pretrained.load(tmp_path / "saved", dtype="float32", attention_impl="reference")
    np.testing.assert_array_equal(np.asarray(reloaded.variables["params"]["embed_tokens"]["embedding"]),
                                  np.asarray(table) * 2)
    copy = f"model.layers.{config.num_hidden_layers}.embed_tokens.weight"
    with safe_open(tmp_path / "saved" / "model.safetensors", "np") as stored:
        np.testing.assert_array_equal(stored.get_tensor(copy), np.asarray(table) * 2)
    # A depth stored with fused experts loads as the per-expert one does, its fused tensors kept whole
    # for the save (ledger 3141).
    model.save_pretrained(tmp_path / "fused")
    with_prediction_depth(tmp_path / "fused", config.num_hidden_layers, config.hidden_size, fused=True)
    fused = Pretrained.load(tmp_path / "fused", dtype="float32", attention_impl="reference")
    np.testing.assert_array_equal(np.asarray(fused.model.apply(fused.variables, ids)),
                                  np.asarray(trunk.model.apply(trunk.variables, ids)))
    fused.save(tmp_path / "fused-saved")
    gate_up = f"model.layers.{config.num_hidden_layers}.mlp.experts.gate_up_proj"
    with (safe_open(tmp_path / "fused" / "model.safetensors", "np") as source,
          safe_open(tmp_path / "fused-saved" / "model.safetensors", "np") as saved):
        np.testing.assert_array_equal(saved.get_tensor(gate_up), source.get_tensor(gate_up))
    model.save_pretrained(tmp_path / "unknown")
    with_prediction_depth(tmp_path / "unknown", config.num_hidden_layers, config.hidden_size,
                          ("unknown.weight",))
    with pytest.raises(ValueError):
        Pretrained.load(tmp_path / "unknown", dtype="float32", attention_impl="reference")


@pytest.mark.parametrize("case", sorted(BUILT_IN))
def test_a_built_in_type_of_known_parts_loads_natively(tmp_path, case):
    """A type transformers builds and no registered family reads verifies from parts and loads as a
    native decoder, its logits holding check 1 against transformers' own. A routed type's checkpoint
    holds each expert's projections apart, as transformers saves them, which its load stacks into the
    fused tensors its experts module holds (its conversion mapping): the load reads them as stored. A
    release's prediction depth past the trunk, which the source's own code builds none of and its load
    leaves unread, is left unread too."""
    from interop_support import tiny_config
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    fields = dict(BUILT_IN[case])
    model_type = fields.pop("model_type", case)
    torch.manual_seed(0)
    config = tiny_config(model_type, **fields)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    scatter_weights(model)
    model.save_pretrained(tmp_path)
    if fields.get("num_nextn_predict_layers"):
        with_prediction_depth(tmp_path, config.num_hidden_layers, config.hidden_size)
    with safe_open(tmp_path / "model.safetensors", "pt") as stored:
        names = set(stored.keys())
    assert model_type not in STORED_EXPERT or STORED_EXPERT[model_type] in names
    verified = verify_mapping(json.loads((tmp_path / "config.json").read_text()))
    assert isinstance(verified, VerifiedMapping) and verified.family == "parts"
    loaded = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")
    forward = jax.jit(loaded.model.apply)

    def built(dtype):
        return AutoModelForCausalLM.from_pretrained(tmp_path, dtype=dtype, attn_implementation="eager",
                                                    experts_implementation="eager").eval()

    reference = built(torch.float32)
    wide = float64_twin(reference, lambda: built(torch.float64)).eval()
    draw = clearing_selections(loaded, token_draw(0x3070, model.config.vocab_size, (4, 24)), case)
    assert_as_exact(as_given(lambda ids: np.asarray(forward(loaded.variables, ids), np.float64)),
                    torch_logits(reference), draw, f"{case} logits", seeds=BUILT_IN_SEEDS[case],
                    truth_fn=torch_logits(wide, float64))


def test_an_lfm2_of_convolutions_alone_assembles():
    """An LFM2 whose layers are all gated short convolutions (Lfm2Config allows no attention layer) states
    its head geometry through its model's config, as its rotary reads it, so it assembles: every layer a
    `conv` kind, the head tied. transformers' own forward runs it only without a cache (a cache of
    convolutions alone has no sequence length), as the probe runs every source."""
    from interop_support import tiny_config
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(tiny_config("lfm2", head_dim=None, conv_L_cache=4,
                                                             layer_types=["conv"] * 4))
    fields = parts.assemble(model).fields
    assert fields["layer_types"] == ("conv",) * 4 and (fields["num_heads"], fields["num_kv_heads"]) == (4, 2)


def _aliased(first, second):
    def compute(x):
        return x if first is second else -x
    return compute


def test_closures_whose_containers_alias_apart_compute_apart():
    """Two closures of one body over lists of equal items compute alike only where the lists alias alike
    (`code_identity._paired`): one binding the same list twice returns its input, one binding two lists
    negates it; and two kernel-decorated copies of a helper, each holding its own tuple of parameter
    names and list of child names (LFM2-MoE's of LFM2's convolution), compute alike."""
    from transformers.models.lfm2.modeling_lfm2 import Lfm2ShortConv
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeShortConv

    from dew.interop.code_identity import Renaming, matched_class, same_code

    shared, apart = [], ([], [])
    assert not same_code(_aliased(shared, shared), _aliased(*apart), Renaming("", ""))
    assert same_code(_aliased(*apart), _aliased([], []), Renaming("", ""))
    assert matched_class(Lfm2MoeShortConv, Lfm2ShortConv) is not None


def test_a_biased_short_convolution_is_left_to_tier_3():
    """LFM2's biased convolution zeroes its padded slots before its input projection, so they still convolve
    their bias terms into the next real tokens' history, which Dew's held convolution keeps empty: tier 2
    assembles none of it."""
    from interop_support import tiny_config
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(tiny_config("lfm2", head_dim=None, conv_bias=True,
                                                             layer_types=["conv", "full_attention"] * 2))
    with pytest.raises(parts.Ambiguous):
        parts.assemble(model)


ERNIE_CLAMP_SEEDS = 8
"""The clamped router's declared check-1 seeds, sized by `seeds_for` on armada."""


def test_ernies_router_divides_by_its_clamped_sum_where_its_bias_picks_a_faint_expert():
    """ERNIE 4.5's router picks by score plus its correction bias and divides the chosen scores by their sum
    clamped below at `moe_norm_min`: where the bias picks an expert of faint score (its logit near -40),
    the clamp binds. Dew's router as tier 2 reads it (`Router.weight_sum_min`) picks alike, and its
    weights and their gradient into the tokens are on par with the source's from its float64 twin
    (check 1)."""
    import jax.numpy as jnp
    from transformers import Ernie4_5_MoeConfig
    from transformers.models.ernie4_5_moe.modeling_ernie4_5_moe import Ernie4_5_MoeTopKRouter

    from dew.nn.moe import EXPERT_BIAS, EXPERT_BIAS_COLLECTION, Router

    config = Ernie4_5_MoeConfig(hidden_size=4, moe_num_experts=2, moe_k=1)
    source = Ernie4_5_MoeTopKRouter(config)
    with torch.no_grad():
        source.weight.copy_(torch.tensor([[0.0, 0.0, 0.0, 0.0], [-40.0, 0.0, 0.0, 0.0]]))
        source.moe_statics.e_score_correction_bias.copy_(torch.tensor([[0.0, 2.0]]))
    wide = float64_twin(source, lambda: Ernie4_5_MoeTopKRouter(config))
    router = Router(num_experts=2, in_features=4, top_k=1, expert_bias=True, weight_sum_epsilon=0.0,
                    weight_sum_min=float(source.norm_min))
    variables = {**router.init(jax.random.key(0), jnp.zeros((1, 4))),
                 "params": {"kernel": jnp.asarray(source.weight.detach().numpy().T)},
                 EXPERT_BIAS_COLLECTION: {EXPERT_BIAS: jnp.asarray([0.0, 2.0])}}

    def draw(seed):
        rng = np.random.default_rng([0x3078, seed])
        tokens = (np.array([1.0, 0.0, 0.0, 0.0]) + 0.05 * rng.standard_normal((6, 4))).astype(np.float32)
        return tokens, rng.standard_normal((6, 1)).astype(np.float32)

    @as_given
    def ours(tokens, cotangent):
        def weighed(tokens):
            return jnp.sum(router.apply(variables, tokens)[0] * cotangent)
        return (np.asarray(router.apply(variables, jnp.asarray(tokens))[0]),
                np.asarray(jax.grad(weighed)(jnp.asarray(tokens))))

    def torch_arm(model, dtype):
        def run(tokens, cotangent):
            held = torch.from_numpy(np.asarray(tokens)).to(dtype).requires_grad_()
            _, weights, chosen = model(held)
            assert chosen.flatten().tolist() == [1] * len(chosen)
            (weights * torch.from_numpy(np.asarray(cotangent)).to(dtype)).sum().backward()
            return weights.detach().numpy(), held.grad.numpy()
        return run

    def truth(tokens, cotangent):
        with float64():
            return torch_arm(wide, torch.float64)(tokens, cotangent)

    assert_as_exact(ours, as_given(torch_arm(source, torch.float32)), draw, "ERNIE clamped router",
                    seeds=ERNIE_CLAMP_SEEDS, truth_fn=truth)


def test_a_biased_ernie_moe_is_left_to_tier_3():
    """ERNIE 4.5's `use_bias` biases its shared experts beside its unbiased routed ones, which Dew's routed
    layers do not hold (their gated MLPs are bias-free): tier 2 assembles none of it, saying so."""
    from interop_support import tiny_config

    config = tiny_config("ernie4_5_moe", head_dim=None, use_bias=True, moe_num_experts=8, moe_k=2,
                         moe_intermediate_size=32, moe_num_shared_experts=1, moe_layer_start_index=0)
    verdict = verify_mapping(config.to_dict())
    assert not isinstance(verdict, VerifiedMapping)
    assert "shared experts are biased" in verdict.reasons["parts"]


def test_attention_fields_join_the_kinds_of_the_layers_that_state_them():
    """A part's attention fields (`_with_attention_fields`) join each stated layer type's kind beside the
    fields it holds, a NoPE kind's among them, and leave a convolution layer's kind as it is; layers of
    one type that state different fields are `Ambiguous`."""
    nope = {"mixer": {"class": "attention", "fields": {"nope": True}}}
    conv = {"mixer": {"class": "short_conv", "fields": {"conv_L_cache": 3}}}
    fields = {"num_layers": 4, "layer_types": ("conv", "full_attention", "conv", "full_attention_nope"),
              "kinds": {"conv": conv, "full_attention_nope": nope}}
    pairs = {"rotary_pairs": "adjacent"}
    joined = parts._with_attention_fields(fields, {1: pairs, 3: pairs})
    assert joined["kinds"]["conv"] == conv
    assert joined["kinds"]["full_attention_nope"]["mixer"]["fields"] == {"nope": True, **pairs}
    assert joined["kinds"]["full_attention"]["mixer"] == {"class": "attention", "fields": pairs}
    with pytest.raises(parts.Ambiguous):
        parts._with_attention_fields({**fields, "layer_types": ("conv", "full_attention") * 2}, {1: pairs})


def test_a_decoder_norming_its_rotated_heads_decodes_through_its_cache_as_transformers_generates(tmp_path):
    """Hunyuan's attention norms each head's queries and keys after their rotation, so a cached key is
    normed as it is written: a cached greedy continuation of a decoder assembled from its parts draws the
    tokens transformers' own generate draws."""
    from interop_support import tiny_config
    from transformers import AutoModelForCausalLM

    from dew.sampling import Sampling, generate

    torch.manual_seed(0)
    config = tiny_config("hunyuan_v1_moe", **BUILT_IN["hunyuan_v1_moe"])
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    scatter_weights(model)
    model.save_pretrained(tmp_path)
    loaded = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")
    source = AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32, attn_implementation="eager",
                                                  experts_implementation="eager").eval()
    ids = torch.as_tensor(np.random.default_rng(0x3078).integers(3, config.vocab_size, (1, 6)))
    with torch.no_grad():
        drawn = source.generate(ids, max_new_tokens=8, min_new_tokens=8, do_sample=False)
    generated = generate(loaded.model, loaded.variables, np.asarray(ids), 8, key=0,
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(generated.tokens), drawn.numpy())


def test_a_stored_name_reads_as_the_sources_load_reads_it():
    """An assembly reads a stored tensor's name as transformers' load does (`get_model_conversion_mapping`):
    Mixtral's release stores each expert's projections apart, as `w1`, `w3` and `w2` under
    `block_sparse_moe`, which its load renames and stacks into the fused tensors its experts hold; each
    reads as its expert's canonical gate, up or down projection by its place among the merge's sources. A
    backbone's name without the base model's prefix reads under it, and a name the source holds as stored
    reads as stored."""
    from interop_support import tiny_config
    from transformers import AutoModelForCausalLM

    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(tiny_config("mixtral", num_local_experts=4))
    assembly = parts.assemble(model)
    layer = "model.layers.1"
    assert [assembly.canonical(f"{layer}.block_sparse_moe.experts.2.{stored}.weight")
            for stored in ("w1", "w3", "w2")] == [
        f"{layer}.mlp.experts.2.{projection}.weight" for projection in ("gate_proj", "up_proj", "down_proj")]
    assert assembly.canonical("layers.1.self_attn.q_proj.weight") == f"{layer}.self_attn.q_proj.weight"
    assert assembly.canonical(f"{layer}.mlp.gate.weight") == f"{layer}.mlp.gate.weight"
    assert assembly.canonical(f"{layer}.block_sparse_moe.experts.2.w4.weight") is None


def test_a_remote_causal_lm_over_a_built_in_config_is_its_own(tmp_path):
    """A repository naming its own causal LM beside transformers' config for a type transformers
    builds verifies its own class, as transformers resolves the auto map: the renamed Llama over CWM's
    config, whose own model attends windows the renamed Llama does not, verifies as the renamed one."""
    write_source(tmp_path / "renamed", "renamed")
    config = json.loads((tmp_path / "renamed" / "config.json").read_text())
    config = {**config, "model_type": "cwm", "sliding_window": 4,
              "auto_map": {"AutoModelForCausalLM": config["auto_map"]["AutoModelForCausalLM"]}}
    (tmp_path / "renamed" / "config.json").write_text(json.dumps(config))
    verified = verify_mapping(config, remote=tmp_path / "renamed")
    assert isinstance(verified, VerifiedMapping) and verified.family == "parts"
    assert verified.source is not None and set(verified.source.references) == {"AutoModelForCausalLM"}


@pytest.mark.parametrize("layer", [0, -1])
def test_a_backward_its_forward_does_not_show_is_not_assembled(tmp_path, layer):
    """A source whose parts are the known ones by their code but whose one query projection's weight
    takes no gradient (a tensor hook zeroing it, in training and evaluation alike) is refused: that
    one parameter's gradient is far from the truth, however small beside the others'. So it is on the
    last of four layers, which a probe cut to fewer would not reach: the probe cuts no depth of a
    source's own code (`verify.shallow`)."""
    def frozen(code):
        changed = code.replace(
            "        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)\n",
            "        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)\n"
            f"        self.model.layers[{layer}].attention.wq.weight.register_hook(\n"
            "            lambda gradient: 0 * gradient)\n", 1)
        assert changed != code
        return changed

    write_source(tmp_path / "frozen", "renamed", frozen)
    config = {**json.loads((tmp_path / "frozen" / "config.json").read_text()), "num_hidden_layers": 4}
    with pytest.raises(ValueError):
        verify_mapping(config, remote=tmp_path / "frozen")


FACTORY_CONFIG = """
from transformers import LlamaConfig


class FactoryConfig(LlamaConfig):
    model_type = "factory"
"""

FACTORY = """
from transformers import LlamaForCausalLM, PreTrainedModel

from .configuration_factory import FactoryConfig


class FactoryForCausalLM(PreTrainedModel):
    config_class = FactoryConfig

    def __new__(cls, config, *args, **kwargs):
        model = LlamaForCausalLM(config)
        for layer in model.model.layers[2:]:
            layer.self_attn.q_proj.weight.register_hook(lambda gradient: 0 * gradient)
        return model
"""
"""A remote causal LM whose constructor returns transformers' own Llama, its later layers' query
projections taking no gradient."""


def test_a_remote_constructor_returning_transformers_classes_is_probed_at_its_depth(tmp_path):
    """A remote source whose constructor returns transformers' own Llama, a hook zeroing the gradient of
    each query projection past its second layer, is refused: the probe cuts no depth of a model a
    source's own code built, whatever classes it holds (`verify.shallow`), so its four layers run and
    the hooked ones show."""
    directory = tmp_path / "factory"
    directory.mkdir()
    (directory / "configuration_factory.py").write_text(FACTORY_CONFIG)
    (directory / "modeling_factory.py").write_text(FACTORY)
    config = {**TINY, "num_hidden_layers": 4, "model_type": "factory",
              "auto_map": {"AutoConfig": "configuration_factory.FactoryConfig",
                           "AutoModelForCausalLM": "modeling_factory.FactoryForCausalLM"}}
    (directory / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError):
        verify_mapping(config, remote=directory)


def _slot_release(tmp_path, edit=lambda code: code):
    """The renamed source's repository alone (`edit` of its modeling code), and a local slot release over
    it: the same repository, its remote code included, with a tokenizer and a slot readout beside it."""
    import shutil

    from test_slot_sources import BACKBONE

    alone, release = tmp_path / "renamed", tmp_path / "release"
    write_source(alone, "renamed", edit)
    shutil.copytree(alone, release)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(BACKBONE / name, release / name)
    (release / "readout.json").write_text(json.dumps({"readout": "slot_lm", "head_dim": 8, "n_slots": 16}))
    return alone, release


def test_a_slot_release_over_a_model_of_parts_loads_its_backbone_by_its_own_code(tmp_path):
    """A slot release whose backbone is a repository of remote code reads that code where the load opts
    in, as the backbone alone does: its decoder verifies from parts and holds the backbone's own
    variables, and it computes the backbone's logits."""
    alone, release = _slot_release(tmp_path)
    backbone = Pretrained.load(alone, dtype="float32", attention_impl="reference", trust_remote_code=True)
    loaded = Pretrained.load(release, dtype="float32", attention_impl="reference", trust_remote_code=True)
    for ours, theirs in zip(jax.tree.leaves(loaded.variables), jax.tree.leaves(backbone.variables),
                            strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    ids = np.random.default_rng(1693).integers(0, TINY["vocab_size"], (2, 12)).astype(np.int32)
    np.testing.assert_array_equal(np.asarray(loaded.model.apply(loaded.variables, ids)),
                                  np.asarray(backbone.model.apply(backbone.variables, ids)))


def _counted(path):
    """An edit of the renamed Llama whose causal LM, built to run (off torch's meta device, where a probe
    sizes it and an assembly reads its parts), appends a line to `path`: the lines count the times the
    source's model is built to run."""
    def edit(code: str) -> str:
        changed = code.replace(
            "        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)\n",
            "        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)\n"
            "        if not self.lm_head.weight.is_meta:\n"
            f"            with open({str(path)!r}, 'a') as file:\n"
            "                file.write('built\\n')\n", 1)
        assert changed != code
        return changed
    return edit


@pytest.mark.parametrize("kind", ["alone", "release"])
def test_one_load_of_local_remote_code_probes_it_once(tmp_path, kind):
    """A load verifies the remote code it runs once: a local repository's code, which no commit names,
    has no recorded verification to read back, so each probe of it builds and runs the source again. One
    load of the repository alone, and of a slot release over it, which routes its backbone before
    reading its weights, builds the source's model to run as often as one probe of it does
    (`verify_mapping`), and Dew's own load builds none."""
    built = tmp_path / "built"
    sources = dict(zip(("alone", "release"), _slot_release(tmp_path, _counted(built)), strict=True))

    def ran(run) -> int:
        built.unlink(missing_ok=True)
        run()
        return len(built.read_text().splitlines()) if built.exists() else 0

    config = json.loads((sources["alone"] / "config.json").read_text())
    probe = ran(lambda: verify_mapping(config, remote=sources["alone"]))
    assert probe > 0
    assert ran(lambda: Pretrained.load(sources[kind], dtype="float32", attention_impl="reference",
                                       trust_remote_code=True)) == probe
