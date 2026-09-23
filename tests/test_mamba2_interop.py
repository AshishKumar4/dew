"""A `Mamba2ForCausalLM` checkpoint through `dew.interop.mamba2` against the
transformers 5.16.1 logits in tests/fixtures/hf/mamba2-tiny.

Observed on CPU: fp32 logits to 9.4e-07 on logits of magnitude 1.7,
tolerance 1e-5; a bfloat16 run of the same weights to 1.5e-02 from the fp32
reference and 2.9e-02 from the reference's own bfloat16 forward (which sits
2.3e-02 from its fp32), tolerance 1e-1 on each.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from safetensors.numpy import load_file

from dew.interop.mamba2 import config_from_hf, export_path, translate, weight_path
from dew.interop.pretrained import load_pretrained
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers.mamba2 import Mamba2Mixer

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hf" / "mamba2-tiny"


@pytest.fixture(scope="module")
def hf_config():
    return json.loads((FIXTURE / "config.json").read_text())


@pytest.fixture(scope="module")
def tensors():
    return load_file(FIXTURE / "model.safetensors")


def largest(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left, np.float32) - np.asarray(right, np.float32))))


def test_the_config_reads_the_references_fields(hf_config):
    used = set()
    config = config_from_hf(hf_config, used)
    assert config["mlp_features"] == 0 and config["emb_features"] == 8 and config["num_layers"] == 2
    assert config["tie_embeddings"] is False and config["norm_eps"] == 1e-5
    assert config["mixer"] == Mamba2Mixer(
        num_heads=2, head_dim=8, state_size=4, n_groups=1, conv_kernel=4, chunk_size=4,
        use_bias=False, use_conv_bias=True, time_step_limit=(0.0, float("inf")))
    # Every field the forward reads is accounted for; what is left is
    # token ids, init policy and metadata.
    assert set(hf_config) - used <= {
        "architectures", "bos_token_id", "eos_token_id", "pad_token_id", "dtype", "initializer_range",
        "use_cache", "transformers_version", "model_type", "num_hidden_layers"}
    assert CausalTransformer(**config).mixer == config["mixer"]


@pytest.mark.parametrize("field, value, message", [
    ("hidden_act", "gelu", "hidden_act"),
    ("head_dim", 4, "must equal"),
    ("model_type", "mamba", "model_type"),
])
def test_a_field_the_mixer_cannot_compute_is_refused(hf_config, field, value, message):
    with pytest.raises(ValueError, match=message):
        config_from_hf({**hf_config, field: value})


def test_every_fixture_tensor_has_a_place_and_an_unknown_one_none(hf_config, tensors):
    config = config_from_hf(hf_config)
    paths = {name: weight_path(name, config) for name in tensors}
    assert paths["backbone.embeddings.weight"] == ("params", "embed_tokens", "embedding")
    assert paths["backbone.norm_f.weight"] == ("params", "norm", "scale")
    assert paths["lm_head.weight"] == ("params", "lm_head", "kernel")
    assert paths["backbone.layers.1.norm.weight"] == ("params", "layers_1", "input_layernorm", "scale")
    assert paths["backbone.layers.0.mixer.conv1d.bias"] == ("params", "layers_0", "self_attn", "conv1d", "bias")
    assert paths["backbone.layers.0.mixer.in_proj.weight"] == ("params", "layers_0", "self_attn", "in_proj", "kernel")
    assert weight_path("lm_head.weight", {"tie_embeddings": True}) is None
    with pytest.raises(ValueError, match="no place"):
        weight_path("backbone.layers.0.mlp.up_proj.weight", config)


def test_the_translated_weights_reproduce_the_reference_logits(hf_config, tensors):
    """The whole checkpoint: the fixture's tensors through `translate`, the
    model `config_from_hf` names, against the reference's fp32 logits on
    the fixture's own tokens. Largest observed difference 9.4e-07 on logits
    of magnitude 1.7; every argmax equal."""
    config = config_from_hf(hf_config)
    model = CausalTransformer(**config, max_seq_len=16)
    variables = translate(tensors, config)
    ids = jnp.asarray(np.load(FIXTURE / "input_ids.npy"))
    template = jax.eval_shape(model.init, jax.random.key(0), ids)
    assert jax.tree.map(jnp.shape, template) == jax.tree.map(jnp.shape, variables)

    logits = model.apply(variables, ids)
    reference = np.load(FIXTURE / "logits.npy")

    assert largest(logits, reference) < 1e-5
    assert np.array_equal(np.asarray(logits).argmax(-1), reference.argmax(-1))


def test_a_tied_head_is_checked_and_dropped(hf_config, tensors):
    config = config_from_hf({**hf_config, "tie_word_embeddings": True})
    tied = {**tensors, "lm_head.weight": tensors["backbone.embeddings.weight"]}
    variables = translate(tied, config)
    assert "lm_head" not in variables["params"]
    with pytest.raises(ValueError, match="tied"):
        translate(tensors, config)


def test_the_source_pins_the_reference():
    source = json.loads((FIXTURE / "source.json").read_text())
    assert source["transformers"]["version"] == "5.16.1"


def test_export_path_inverts_weight_path(hf_config, tensors):
    """Every fixture tensor's place maps back to its own name, and the tied
    head's copy is the one name with no place and no export."""
    config = config_from_hf(hf_config)
    for name in tensors:
        path = weight_path(name, config)
        if path is None:
            assert export_path("lm_head.kernel", config) is None
            continue
        assert export_path(".".join(path[1:]), config) == name


def test_the_public_loader_reads_the_fixture(tensors):
    """`load_pretrained` dispatches the `mamba2` model type through the
    decoder family table and reproduces the reference logits."""
    source = load_pretrained(FIXTURE, dtype="float32", attention_impl="reference")
    ids = jnp.asarray(np.load(FIXTURE / "input_ids.npy"))
    logits = source.model.apply(source.variables, ids)
    reference = np.load(FIXTURE / "logits.npy")
    assert largest(logits, reference) < 1e-5
