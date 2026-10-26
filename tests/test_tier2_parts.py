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
from interop_support import clearing_selections, token_draw, torch_logits
from reference_error import assert_as_exact

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
    assert_as_exact(lambda ids: np.asarray(forward(loaded.variables, ids), np.float64),
                    torch_logits(reference), draw, f"{name} logits", seeds=SEEDS[name],
                    truth_fn=torch_logits(wide, float64))


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


def test_a_passed_verification_is_read_back_by_its_key(tmp_path, monkeypatch):
    """A verification is recorded under its key in Dew's cache, and a later one under the same config,
    transformers, Dew and code reads the record rather than probing again: here the record's own
    reference string comes back."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    write_source(tmp_path / "renamed", "renamed")
    config = json.loads((tmp_path / "renamed" / "config.json").read_text())
    first = verify_mapping(config, remote=tmp_path / "renamed")
    assert isinstance(first, VerifiedMapping)
    (record,) = (tmp_path / "cache" / "dew" / "verified").glob("*/mapping.json")
    stored = json.loads(record.read_text())
    record.write_text(json.dumps({**stored, "reference": "the recorded reference"}))
    second = verify_mapping(config, remote=tmp_path / "renamed")
    assert isinstance(second, VerifiedMapping) and second.reference == "the recorded reference"


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


def test_a_backward_its_forward_does_not_show_is_not_assembled(tmp_path):
    """A source whose parts are the known ones by their code but whose one query projection's weight
    takes no gradient (a tensor hook zeroing it, in training and evaluation alike) is refused: that
    one parameter's gradient is far from the truth, however small beside the others'."""
    def frozen(code):
        changed = code.replace(
            "        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)\n",
            "        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)\n"
            "        self.model.layers[0].attention.wq.weight.register_hook(\n"
            "            lambda gradient: 0 * gradient)\n", 1)
        assert changed != code
        return changed

    write_source(tmp_path / "frozen", "renamed", frozen)
    config = json.loads((tmp_path / "frozen" / "config.json").read_text())
    with pytest.raises(ValueError):
        verify_mapping(config, remote=tmp_path / "frozen")


def test_a_slot_release_over_a_model_of_parts_loads_its_backbone_by_its_own_code(tmp_path):
    """A slot release whose backbone is a repository of remote code reads that code where the load opts
    in, as the backbone alone does: its decoder verifies from parts and holds the backbone's own
    variables, and it computes the backbone's logits."""
    import shutil

    from test_slot_sources import BACKBONE

    alone, release = tmp_path / "renamed", tmp_path / "release"
    write_source(alone, "renamed")
    shutil.copytree(alone, release)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(BACKBONE / name, release / name)
    (release / "readout.json").write_text(json.dumps({"readout": "slot_lm", "head_dim": 8, "n_slots": 16}))
    backbone = Pretrained.load(alone, dtype="float32", attention_impl="reference", trust_remote_code=True)
    loaded = Pretrained.load(release, dtype="float32", attention_impl="reference", trust_remote_code=True)
    assert type(loaded.model) is type(backbone.model)
    for ours, theirs in zip(jax.tree.leaves(loaded.variables), jax.tree.leaves(backbone.variables),
                            strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    ids = np.random.default_rng(1693).integers(0, TINY["vocab_size"], (2, 12)).astype(np.int32)
    np.testing.assert_array_equal(np.asarray(loaded.model.apply(loaded.variables, ids)),
                                  np.asarray(backbone.model.apply(backbone.variables, ids)))
