"""Hugging Face decoder checkpoints, loaded into CausalTransformer and back out.

The claim these tests defend is parity, not plausibility: the same weights and
the same token ids through the reference implementation and through dew have to
produce the same logits. tools/hf_reference.py writes the fixtures under torch
and transformers (the reference), including two tiny random-weight checkpoints
whose logits are committed, so the comparison runs in CI without a download.

Tolerances and the differences actually observed, fp32 on CPU:

- qwen3-tiny  : max |logit difference| 8.3e-06, tolerance 1e-4
- gemma3-tiny : max |logit difference| 3.3e-06, tolerance 1e-4
- llama-tiny  : max |logit difference| 6.1e-06, tolerance 1e-4 (untied head,
  biased projections)
- Qwen3-0.6B  : max |top-32 logit difference| 1.4e-04, mean 1.2e-05, tolerance
  5e-3, and the argmax of all 48 positions equal. The larger residue is 28
  layers of a real checkpoint accumulating fp32 rounding, not a different
  computation.
"""

import json
import os
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.interop.hf_decoders import (
    load_pretrained_decoder, save_pretrained_decoder, translate_config,
    translate_weights,
)
from dew.registry import models, with_precision

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"
TINY = ("qwen3-tiny", "gemma3-tiny", "llama-tiny")
TORCH_VENV = Path("/tmp/hfref/bin/python")
REAL = FIXTURES / "qwen3-0.6b"


def fixture_config(name):
    return json.loads((FIXTURES / name / "config.json").read_text())


def fp32_decoder(directory, **kwargs):
    """The fixture as a model plus variables, in fp32 on the reference kernel."""
    return load_pretrained_decoder(str(directory), dtype='float32',
                                   attention_impl='reference', **kwargs)


def flat_tree(tree):
    leaves, _ = jax.tree_util.tree_flatten_with_path(tree)
    return {'.'.join(str(entry.key) for entry in path): leaf for path, leaf in leaves}


def test_qwen3_config_translates_field_by_field():
    config = translate_config(fixture_config("qwen3-tiny"))

    assert config == {
        'vocab_size': 256, 'emb_features': 64, 'num_layers': 2, 'num_heads': 4,
        'num_kv_heads': 2, 'head_dim': 16, 'mlp': 'swiglu', 'mlp_features': 128,
        'max_seq_len': 64, 'rope_theta': 1e6, 'rope_local_theta': None,
        'layer_types': ('full_attention', 'full_attention'), 'sliding_window': None,
        'norm_eps': 1e-6, 'scale_after_cast': True, 'qk_norm': True, 'attention_bias': False,
        'tie_embeddings': True,
    }


def test_gemma3_config_carries_the_gemma_switches():
    config = translate_config(fixture_config("gemma3-tiny"))

    assert config['sandwich_norms'] and config['scale_offset']
    assert config['embedding_scale'] and config['mlp'] == 'geglu'
    assert config['qk_norm'] and config['num_kv_heads'] == 1
    # query_pre_attn_scalar 16, not the head_dim of 32
    assert config['attention_scale'] == pytest.approx(0.25)
    assert config['final_logit_softcap'] == 30.0
    assert config['layer_types'] == ('sliding_attention', 'full_attention')
    assert config['sliding_window'] == 4
    # rope_local_base_freq for the sliding layer, rope_theta for the full one
    assert (config['rope_theta'], config['rope_local_theta']) == (1e6, 1e4)


def test_a_multimodal_gemma3_config_is_refused():
    """Only a text decoder maps, and no published multimodal Gemma 3 has a
    text_config that would: gemma-3-4b, 12b and 27b all carry rope_scaling
    {'rope_type': 'linear', 'factor': 8}, which the field map refuses. So the
    refusal names the model_type instead of loading the text half of a
    checkpoint whose vision tower nothing here runs."""
    wrapped = {'model_type': 'gemma3', 'text_config': fixture_config("gemma3-tiny"),
               'vision_config': {'hidden_size': 8}, 'mm_tokens_per_image': 256,
               'boi_token_index': 255999, 'eoi_token_index': 256000,
               'image_token_index': 262144}

    with pytest.raises(ValueError, match="model_type 'gemma3'"):
        translate_config(wrapped)


def test_the_real_gemma3_1b_config_translates():
    """google/gemma-3-1b-pt is gated, so the fixture is the identical config
    from a mirror. Nothing in it is beyond the field map."""
    config = translate_config(fixture_config("gemma3-1b"))

    assert config['emb_features'] == 1152 and config['num_layers'] == 26
    assert (config['num_heads'], config['num_kv_heads']) == (4, 1)
    assert config['head_dim'] == 256 and config['mlp_features'] == 6912
    # the config names no tie_word_embeddings, and Gemma3TextConfig ties by
    # default where Qwen and Llama do not
    assert config['vocab_size'] == 262144 and config['tie_embeddings'] is True
    assert config['attention_scale'] == pytest.approx(256 ** -0.5)
    assert config['sliding_window'] == 512
    assert (config['rope_theta'], config['rope_local_theta']) == (1e6, 1e4)
    assert config['sandwich_norms'] and config['scale_offset']
    # sliding_window_pattern 6: five sliding layers, then a full one
    assert config['layer_types'][:6] == (
        'sliding_attention',) * 5 + ('full_attention',)
    assert config['layer_types'].count('full_attention') == 4
    # the cache is clamped, the config asks for 32768
    assert config['max_seq_len'] == 8192


@pytest.mark.parametrize("field, value, message", [
    ('model_type', 'mamba', "model_type 'mamba'"),
    ('model_type', 'qwen2', "o_proj does not"),
    ('attn_logit_softcapping', 50.0, "attn_logit_softcapping"),
    ('use_bidirectional_attention', True, "use_bidirectional_attention"),
    ('hidden_activation', 'relu', "hidden_act 'relu'"),
    ('rope_parameters', {'rope_type': 'linear', 'factor': 8.0, 'rope_theta': 1e6},
     "rope_type 'linear'"),
    ('sliding_window', None, "sliding_window is not set"),
    ('quantization_config', {'bits': 4}, "quantization_config"),
])
def test_a_config_field_with_no_counterpart_is_refused(field, value, message):
    config = {**fixture_config("gemma3-tiny"), field: value}
    with pytest.raises(ValueError, match=message):
        translate_config(config)


def test_a_rope_scaling_spelled_the_old_way_is_refused():
    """transformers reads rope_type from the older 'type' key too
    (modeling_rope_utils.py:785, 839), so a config that spells its Yarn
    scaling that way scales there and must not load here as plain rope. A
    bare factor names no type at all and is still a scaling."""
    yarn = {**fixture_config("llama-tiny"),
            'rope_scaling': {'type': 'yarn', 'factor': 4.0,
                             'original_max_position_embeddings': 32}}
    with pytest.raises(ValueError, match="rope_type 'yarn'"):
        translate_config(yarn)

    factor_only = {**fixture_config("llama-tiny"), 'rope_scaling': {'factor': 8.0}}
    with pytest.raises(ValueError, match=r"rope_scaling scaling fields \['factor'\]"):
        translate_config(factor_only)


@pytest.mark.parametrize("name", TINY)
def test_translated_weights_are_exactly_the_models_param_tree(name, rng):
    """Same paths, same shapes, same dtypes as a freshly initialised model."""
    config = translate_config(fixture_config(name))
    built = with_precision('causal_transformer', dict(config),
                                   dtype='float32', attention_impl='reference')
    model = models.build('causal_transformer', **built)
    initialised = flat_tree(model.init(rng, jnp.zeros((1, 4), jnp.int32))['params'])

    from dew.interop.hf_decoders import _load_shards
    loaded = flat_tree(translate_weights(_load_shards(FIXTURES / name), config)['params'])

    assert set(loaded) == set(initialised)
    for path, leaf in loaded.items():
        assert leaf.shape == initialised[path].shape, path
        assert leaf.dtype == jnp.float32, path


@pytest.mark.parametrize("name", TINY)
def test_fp32_logits_match_the_reference_implementation(name):
    """The parity claim: transformers' logits, our logits, same weights."""
    directory = FIXTURES / name
    model, variables, _ = fp32_decoder(directory)
    ids = np.load(directory / "input_ids.npy")
    reference = np.load(directory / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


def test_the_bf16_gemma_forward_still_tracks_the_reference():
    """gemma3-tiny is the fixture with an attention scale (query_pre_attn_scalar
    16 on head_dim 32), and bf16 is the recipe's default compute dtype. Against
    the fp32 reference logits the observed difference is 5.7e-02 on the
    reference kernel, tolerance 1e-01; dropping the scale moves them by 1.06."""
    directory = FIXTURES / "gemma3-tiny"
    model, variables, _ = load_pretrained_decoder(
        str(directory), dtype='bfloat16', attention_impl='reference')
    reference = np.load(directory / "logits.npy")

    logits = np.asarray(model.apply(
        variables, jnp.asarray(np.load(directory / "input_ids.npy"), jnp.int32)),
        np.float32)

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-1, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


def test_the_tied_head_is_the_embedding_and_untied_heads_load():
    """A tied checkpoint carries lm_head as a copy; the tree keeps one leaf."""
    config = translate_config(fixture_config("qwen3-tiny"))
    embedding = np.arange(256 * 64, dtype=np.float32).reshape(256, 64)
    tensors = {'model.embed_tokens.weight': embedding,
               'lm_head.weight': embedding.copy()}

    tied = translate_weights(tensors, config)['params']
    assert 'lm_head' not in tied
    assert np.array_equal(tied['embed_tokens']['embedding'], embedding)

    untied = translate_weights(tensors, {**config, 'tie_embeddings': False})['params']
    assert np.array_equal(untied['lm_head']['kernel'], embedding.T)

    # Qwen3-0.6B really does ship both, byte for byte identical; a head that
    # is not that copy means the checkpoint is not the tied model it says
    different = {**tensors, 'lm_head.weight': embedding + 1.0}
    with pytest.raises(ValueError, match="not the embedding it claims to copy"):
        translate_weights(different, config)


def test_an_unfamiliar_tensor_name_is_refused():
    config = translate_config(fixture_config("qwen3-tiny"))
    with pytest.raises(ValueError, match="unknown tensor name"):
        translate_weights({'model.layers.0.self_attn.rotary_emb.inv_freq':
                           np.zeros((8,), np.float32)}, config)


@pytest.mark.parametrize("name", TINY)
def test_export_round_trips_the_weights_and_the_config(name, tmp_path):
    model, variables, _ = fp32_decoder(FIXTURES / name)
    export = tmp_path / name

    save_pretrained_decoder(model, variables, export, tokenizer_name="byte")
    again, reloaded, _ = fp32_decoder(export)

    # against the fixture's config, not the exported one read twice: a field
    # the export changes and the model does not read back (the context length,
    # the hidden_act spelling) shows up here
    assert (translate_config(json.loads((export / "config.json").read_text()))
            == translate_config(fixture_config(name)))
    assert again == model, "the exported config rebuilds a different model"
    for path, leaf in flat_tree(reloaded['params']).items():
        assert np.array_equal(np.asarray(leaf),
                              np.asarray(flat_tree(variables['params'])[path])), path
    generation = json.loads((export / "generation_config.json").read_text())
    assert generation['tokenizer_name'] == "byte"


def biased_qwen3(rng):
    """The qwen3-tiny shape with the q/k/v/o biases its own config leaves off.

    nn.Dense starts a bias at zero, and zeros would let a mismapped bias name
    through the round-trip, so every bias gets its own draw.
    """
    config = {**translate_config(fixture_config("qwen3-tiny")), 'attention_bias': True}
    built = with_precision('causal_transformer', dict(config),
                                   dtype='float32', attention_impl='reference')
    model = models.build('causal_transformer', **built)
    leaves, structure = jax.tree_util.tree_flatten_with_path(
        model.init(rng, jnp.zeros((1, 4), jnp.int32)))
    return model, jax.tree_util.tree_unflatten(structure, [
        jax.random.normal(jax.random.fold_in(rng, index), leaf.shape, leaf.dtype) * 0.2
        if path[-1].key == 'bias' else leaf
        for index, (path, leaf) in enumerate(leaves)])


def test_a_biased_qwen3_round_trips_through_an_export(tmp_path, rng):
    """attention_bias is one flag for all four projections, which is what
    Qwen3Attention builds from config.attention_bias (modeling_qwen3.py:225-236)
    and Gemma3Attention from the same field (modeling_gemma3.py:322-333), so a
    biased qk_norm model is exportable rather than a refusal."""
    model, variables = biased_qwen3(rng)
    export = tmp_path / "biased"

    save_pretrained_decoder(model, variables, export)
    again, reloaded, _ = fp32_decoder(export)

    assert json.loads((export / "config.json").read_text())['attention_bias'] is True
    assert again == model, "the exported config rebuilds a different model"
    for path, leaf in flat_tree(reloaded['params']).items():
        assert np.array_equal(np.asarray(leaf),
                              np.asarray(flat_tree(variables['params'])[path])), path


def test_the_real_checkpoints_tensor_table_matches_the_built_tree(rng):
    """No weights: the 311 names and shapes of Qwen3-0.6B against our tree.

    This is the check that a config translation and a key map stay honest
    about a checkpoint nobody wants to download in CI.
    """
    from dew.interop.hf_decoders import _dew_path

    table = json.loads((REAL / "tensors.json").read_text())
    config = translate_config(json.loads((REAL / "config.json").read_text()))
    built = with_precision('causal_transformer', dict(config),
                                   dtype='float32', attention_impl='reference')
    model = models.build('causal_transformer', **built)

    expected = {}
    for name, info in table['tensors'].items():
        path = _dew_path(name, config)
        if path is None:
            continue  # the tied lm_head copy
        shape = tuple(info['shape'])
        expected['.'.join(path)] = (tuple(reversed(shape))
                                    if path[-1] == 'kernel' else shape)

    tree = jax.eval_shape(
        lambda: model.init(rng, jnp.zeros((1, 4), jnp.int32))['params'])
    initialised = {path: leaf.shape for path, leaf in flat_tree(tree).items()}

    assert initialised == expected
    assert len(expected) == len(table['tensors']) - 1, "the tied head was not skipped"


def qwen3_is_available() -> bool:
    if os.environ.get("DEW_NETWORK_TESTS") == "1":
        return True
    from huggingface_hub import try_to_load_from_cache
    cached = try_to_load_from_cache("Qwen/Qwen3-0.6B", "model.safetensors")
    return isinstance(cached, str)


@pytest.mark.network
@pytest.mark.skipif(not qwen3_is_available(),
                    reason="Qwen3-0.6B is neither cached nor DEW_NETWORK_TESTS=1")
def test_qwen3_0_6b_matches_the_reference_on_the_real_weights():
    """The real thing: 28 layers of Qwen3-0.6B, fp32, against torch's top 32."""
    prompt = json.loads((REAL / "prompt.json").read_text())
    reference = np.load(REAL / "reference.npz")
    ids = np.asarray(prompt['input_ids'], np.int32)[None]

    model, variables, _ = load_pretrained_decoder(
        prompt['repo'], dtype='float32', attention_impl='reference',
        max_seq_len=int(ids.shape[1]))
    logits = np.asarray(model.apply(variables, jnp.asarray(ids)), np.float32)[0]

    assert np.array_equal(np.argmax(logits, axis=-1), reference['argmax'])
    ours = np.take_along_axis(logits, reference['top_ids'], axis=-1)
    difference = float(np.max(np.abs(ours - reference['top_logits'])))
    assert difference < 5e-3, f"max |top-32 logit difference| {difference:.3e}"


def transformers_logits(export, ids, tmp_path):
    """What transformers computes for `ids` on the checkpoint at `export`."""
    script = """
import sys
import numpy as np, torch
from transformers import AutoModelForCausalLM
directory, ids_path, out = sys.argv[1:4]
model = AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32)
model.eval()
model.set_attn_implementation("eager")
with torch.no_grad():
    logits = model(input_ids=torch.from_numpy(np.load(ids_path).astype(np.int64))).logits
np.save(out, logits.to(torch.float32).numpy())
"""
    ids_path, out = tmp_path / "ids.npy", tmp_path / "theirs.npy"
    np.save(ids_path, ids)
    subprocess.run([str(TORCH_VENV), "-c", script, str(export), str(ids_path), str(out)],
                   check=True, capture_output=True)
    return np.load(out)


@pytest.mark.skipif(not TORCH_VENV.exists(),
                    reason="no torch venv at /tmp/hfref to load the export with")
def test_our_export_loads_in_transformers_with_the_same_logits(tmp_path):
    """The export is a real HF checkpoint: transformers reads it and agrees."""
    model, variables, _ = fp32_decoder(FIXTURES / "qwen3-tiny")
    export = tmp_path / "exported"
    save_pretrained_decoder(model, variables, export)

    ids = np.load(FIXTURES / "qwen3-tiny" / "input_ids.npy")
    ours = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(transformers_logits(export, ids, tmp_path) - ours)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"


@pytest.mark.skipif(not TORCH_VENV.exists(),
                    reason="no torch venv at /tmp/hfref to load the export with")
def test_a_biased_qwen3_export_carries_its_biases_into_transformers(tmp_path, rng):
    """Qwen3Attention builds q, k, v and o with bias=config.attention_bias
    (modeling_qwen3.py:225-236), so the reference applies the biases where dew
    does and a biased export is a checkpoint it reads."""
    model, variables = biased_qwen3(rng)
    export = tmp_path / "biased"
    save_pretrained_decoder(model, variables, export)

    ids = np.load(FIXTURES / "qwen3-tiny" / "input_ids.npy")
    ours = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(transformers_logits(export, ids, tmp_path) - ours)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"


# --------------------------------------------------------------------------
# Gemma 4 gaps: per-layer input embeddings and cross-layer KV sharing
# --------------------------------------------------------------------------

GEMMA4 = Path(__file__).resolve().parent / "fixtures" / "gemma4"

# The Dew fields for a fixture. The translator does not cover gemma4 (no
# v-norm, no partial rotary), so the map is written out here, next to the
# assertion it serves.
GEMMA4_FIELDS = dict(
    vocab_size=64, emb_features=32, num_layers=4, num_heads=4, num_kv_heads=2,
    mlp_features=64, max_seq_len=64, rope_theta=10000.0, sliding_window=32,
    norm_eps=1e-6, qk_norm=True, attention_bias=False, attention_scale=1.0,
    embedding_scale=True, tie_embeddings=True, sandwich_norms=True)

# Reference norm names to the Dew names that sit in the same place: Dew's
# post_attention_layernorm is the pre-feedforward norm, applied after the
# residual, and attention_output_norm is the reference's post-attention norm.
GEMMA4_NORMS = {"input_layernorm": "input_layernorm",
                "attention_output_norm": "post_attention_layernorm",
                "post_attention_layernorm": "pre_feedforward_layernorm",
                "mlp_output_norm": "post_feedforward_layernorm"}


def gemma4_tensors(name):
    """Fixture weights, input ids and reference logits.

    The fixtures were generated under torch with transformers 5.16.1 (see the
    meta json beside each npz for the seed, the versions and the config):
    a tiny Gemma4TextConfig, random init, the v_proj of every owning layer
    zeroed. Zeroing the values isolates the new feature paths from the
    values norm, which Dew does not implement; the queries, the shared keys
    and the scores stay live. The attention scale is 1.0, the head dim is
    uniform, full layers run standard rope by config override.
    """
    data = np.load(GEMMA4 / f"{name}.npz")
    tensors = {key: data[key] for key in data.files if key not in ("ids", "expected")}
    return data["ids"], data["expected"], tensors


def gemma4_variables(tensors, per_layer_input_dim, zero_values=True):
    """Reference tensors into a CausalTransformer variables dict.

    zero_values repeats the fixture's own isolation: the committed expected
    logits were generated with the values zeroed, so the comparison zeroes
    the translated v_proj kernels too."""
    tensors = dict(tensors)
    if zero_values:
        for key in [key for key in tensors if key.endswith("self_attn.v_proj.weight")]:
            tensors[key] = np.zeros_like(tensors[key])
    params = {"embed_tokens": {"embedding": tensors["model.embed_tokens.weight"]}}
    for index in range(4):
        prefix = f"model.layers.{index}."
        layer = {dew: {"scale": tensors[prefix + hf + ".weight"]}
                 for dew, hf in GEMMA4_NORMS.items()}
        attention = {
            "q_proj": {"kernel": tensors[prefix + "self_attn.q_proj.weight"].T},
            "o_proj": {"kernel": tensors[prefix + "self_attn.o_proj.weight"].T},
            "q_norm": {"scale": tensors[prefix + "self_attn.q_norm.weight"]}}
        for projection in ("k_proj", "v_proj"):
            key = prefix + f"self_attn.{projection}.weight"
            if key in tensors:
                attention[projection] = {"kernel": tensors[key].T}
        key = prefix + "self_attn.k_norm.weight"
        if key in tensors:
            attention["k_norm"] = {"scale": tensors[key]}
        layer["self_attn"] = attention
        layer["mlp"] = {
            projection: {"kernel": tensors[prefix + f"mlp.{projection}.weight"].T}
            for projection in ("gate_proj", "up_proj", "down_proj")}
        if per_layer_input_dim:
            layer["per_layer_input_gate"] = {
                "kernel": tensors[prefix + "per_layer_input_gate.weight"].T}
            layer["per_layer_projection"] = {
                "kernel": tensors[prefix + "per_layer_projection.weight"].T}
            layer["post_per_layer_input_norm"] = {
                "scale": tensors[prefix + "post_per_layer_input_norm.weight"]}
        params[f"layers_{index}"] = layer
    params["norm"] = {"scale": tensors["model.norm.weight"]}
    if per_layer_input_dim:
        params["embed_tokens_per_layer"] = {
            "embedding": tensors["model.embed_tokens_per_layer.weight"]}
        params["per_layer_model_projection"] = {
            "kernel": tensors["model.per_layer_model_projection.weight"].T}
        params["per_layer_projection_norm"] = {
            "scale": tensors["model.per_layer_projection_norm.weight"]}
    return {"params": params}


def gemma4_model(name, **overrides):
    fields = dict(GEMMA4_FIELDS, **overrides)
    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", fields, dtype="float32", attention_impl="xla"))
    ids, expected, tensors = gemma4_tensors(name)
    variables = gemma4_variables(tensors, fields.get("per_layer_input_dim", 0))
    return model, variables, jnp.asarray(ids, jnp.int32), expected


def test_per_layer_input_embeddings_match_gemma4():
    """PLE parity: the packed table, its projection and norm, the per-layer
    gate, product, projection, norm and residual, against Gemma4TextModel
    with per-layer inputs on and KV sharing off. Largest observed max |logit
    difference| on CPU: 2.5e-07."""
    model, variables, ids, expected = gemma4_model(
        "ple", layer_types=("sliding_attention",) * 3 + ("full_attention",),
        per_layer_input_dim=8, per_layer_input_vocab=64)

    logits = np.asarray(model.apply(variables, ids))

    difference = float(np.max(np.abs(logits - expected)))
    assert difference < 1e-5, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(expected, axis=-1))


def test_kv_sharing_matches_gemma4():
    """Sharing parity: layers 2 and 3 read the K/V of the last earlier layer
    of their own type, against Gemma4TextModel with two shared layers and no
    per-layer inputs. Largest observed max |logit difference| on CPU:
    2.3e-07."""
    model, variables, ids, expected = gemma4_model(
        "kvshare", layer_types=("sliding_attention", "full_attention") * 2,
        num_kv_shared_layers=2)

    logits = np.asarray(model.apply(variables, ids))

    difference = float(np.max(np.abs(logits - expected)))
    assert difference < 1e-5, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(expected, axis=-1))


def test_sharing_layers_own_no_kv_and_name_their_provider():
    """The tree shape of sharing: layers past the cutoff keep q_proj, o_proj
    and q_norm but lose k_proj, v_proj and k_norm; the provider map follows
    the layer type, not the position."""
    model, variables, _, _ = gemma4_model(
        "kvshare", layer_types=("sliding_attention", "full_attention") * 2,
        num_kv_shared_layers=2)
    params = variables["params"]

    assert set(model.kv_sharing) == {2, 3}
    assert model.kv_sharing[2] == 0 and model.kv_sharing[3] == 1
    for index in (2, 3):
        attention = params[f"layers_{index}"]["self_attn"]
        assert set(attention) == {"q_proj", "o_proj", "q_norm"}, set(attention)
    for index in (0, 1):
        attention = params[f"layers_{index}"]["self_attn"]
        assert {"k_proj", "v_proj", "k_norm"} <= set(attention)


def test_sharing_without_a_provider_and_sharing_everything_are_refused():
    base = dict(GEMMA4_FIELDS, layer_types=("sliding_attention", "full_attention") * 2)
    with pytest.raises(ValueError, match="no earlier full_attention layer"):
        models.build("causal_transformer", **base, num_kv_shared_layers=3).kv_sharing
    with pytest.raises(ValueError, match="leave a provider"):
        models.build("causal_transformer", **base, num_kv_shared_layers=4).kv_sharing


def test_the_features_leave_a_plain_tree_unchanged():
    """Off by default: no PLE leaves, no missing K/V, same leaves as before."""
    model, variables, _, _ = gemma4_model("ple", per_layer_input_dim=0)
    assert model.kv_sharing == {}
    flat = flat_tree(variables["params"])
    assert not [name for name in flat if "per_layer" in name]
    assert "layers_0.self_attn.k_proj.kernel" in flat


def test_new_leaves_are_declared_or_heuristic():
    """The coverage sweep builds default configs only, so the new leaves are
    asserted here: the packed table and the projections are declared, the
    scalar norms fall under rank one."""
    from dew.nn.sharding import declared_axes, is_heuristic

    model = models.build("causal_transformer", **dict(
        GEMMA4_FIELDS, per_layer_input_dim=8, num_kv_shared_layers=2,
        layer_types=("sliding_attention", "full_attention") * 2))
    variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, 8), jnp.int32))
    uncovered = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(variables)[0]:
        if leaf.ndim < 2:
            continue
        if declared_axes(path, leaf.ndim) is None and not is_heuristic(path):
            uncovered.append(jax.tree_util.keystr(path))
    assert uncovered == []


def test_a_sharing_model_decodes_like_it_prefills(rng):
    """The decode path of sharing: prefill writes the provider's cache, each
    single-token step reads it, and the tokens match a full forward."""
    model = models.build("causal_transformer", **dict(
        GEMMA4_FIELDS, layer_types=("sliding_attention", "full_attention") * 2,
        num_kv_shared_layers=2, max_seq_len=16))
    params = model.init(rng, jnp.ones((1, 4), jnp.int32))
    prompt = jax.random.randint(rng, (1, 3), 0, 64)

    cache = model.apply(params, 1, method=type(model).init_cache, mutable=["cache"])[1]["cache"]
    variables = {**params, "cache": cache}
    logits, mutated = model.apply(variables, prompt, decode=True, mutable=["cache"])
    first = jnp.argmax(logits[:, -1], axis=-1)
    token, variables = first[:, None], {**params, "cache": mutated["cache"]}
    generated = [first]
    for _ in range(3):
        logits, mutated = model.apply(variables, token, decode=True, mutable=["cache"])
        token = jnp.argmax(logits[:, -1], axis=-1)[:, None]
        variables = {**params, "cache": mutated["cache"]}
        generated.append(token[:, 0])
    decoded = jnp.concatenate([prompt, jnp.stack(generated, axis=1)], axis=1)

    assert jnp.array_equal(
        model.apply(params, decoded)[:, -1].argmax(axis=-1),
        jnp.asarray(generated[-1]))


def test_export_refuses_the_new_features(tmp_path, rng):
    """The three exported families have neither, so a model with either set
    is refused instead of silently dropping its leaves."""
    model = models.build("causal_transformer", **dict(
        GEMMA4_FIELDS, per_layer_input_dim=8, num_kv_shared_layers=2,
        layer_types=("sliding_attention", "full_attention") * 2))
    variables = model.init(rng, jnp.ones((1, 4), jnp.int32))
    with pytest.raises(ValueError, match="per_layer_input_dim"):
        save_pretrained_decoder(model, variables, str(tmp_path))


def test_kv_sharing_scores_live_values_like_gemma4():
    """Sharing with live values: the committed fixture zeroes the values, so
    this runs the transformers model live with its values norms replaced by
    the identity (the one Gemma 4 norm Dew does not implement) and the same
    live weights on both sides. What is under test is the shared-K scores
    with nonzero values behind them. Largest observed max |logit difference|
    on CPU: 3.5e-07."""
    torch = pytest.importorskip("torch")
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    meta = json.loads((GEMMA4 / "kvshare.json").read_text())
    config = Gemma4TextConfig(**meta["config"])
    reference = Gemma4ForCausalLM(config).eval()
    _, _, tensors = gemma4_tensors("kvshare")
    # npz arrays are read-only; torch takes ownership on conversion.
    tensors = {name: array.copy() for name, array in tensors.items()}
    # Buffers (layer_scalar) and the tied head carry no weights in the
    # fixture; both are ones and copies at init, so a non-strict load keeps
    # them and every other leaf comes from the fixture.
    reference.load_state_dict({name: torch.from_numpy(tensors[name])
                               for name, _ in reference.named_parameters()
                               if name in tensors}, strict=False)
    for layer in reference.model.layers:
        layer.self_attn.v_norm = torch.nn.Identity()

    model, _, dew_ids, _ = gemma4_model(
        "kvshare", layer_types=("sliding_attention", "full_attention") * 2,
        num_kv_shared_layers=2)
    live = gemma4_variables(tensors, 0, zero_values=False)
    with torch.no_grad():
        expected = reference(torch.tensor(np.asarray(dew_ids), dtype=torch.long)
                             ).logits.detach().cpu().numpy()
    logits = np.asarray(model.apply(live, dew_ids))

    difference = float(np.max(np.abs(logits - expected)))
    assert difference < 1e-5, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(expected, axis=-1))
