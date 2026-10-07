"""Hugging Face decoders with routed experts or latent attention: GPT OSS,
DeepSeek V2/V4, Kimi K2 and K2.5, GLM 4.5 and GLM-5, Llama 4 and Gemma 4's
routed size, each against transformers' logits (see test_hf_decoders).
"""

import dataclasses
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from model_support import flat_tree
from reference_error import assert_as_exact_as_the_reference, distance
from test_hf_decoders import GEMMA4_MOE, fixture_config, fp32_decoder

from dew.interop import Pretrained, PretrainedDecoder
from dew.interop.codecs import dequantize_mxfp4, quantize_mxfp4
from dew.interop.hf_decoders import translate_config, translate_weights
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.registry import models, with_precision

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "hf"

QWEN2_MOE = FIXTURES / 'qwen2-moe-tiny'


def test_qwen2_moe_released_config_keeps_unnormalized_routing_and_shared_gate():
    config = translate_config(fixture_config('qwen1.5-moe-a2.7b'))
    assert config['mixture'] == {'experts': 60, 'top_k': 4, 'layers': tuple(range(24)),
                                'norm_topk_prob': False, 'expert_features': 1408,
                                'shared_features': 5632, 'shared_gate': True}
    assert config['attention_bias'] and config['o_proj_bias'] is False and not config['qk_norm']
    assert config['layer_types'] == ('full_attention',) * 24


@pytest.mark.network
def test_qwen2_moe_pinned_config_reads_as_the_committed_release():
    from huggingface_hub import hf_hub_download

    source = json.loads((FIXTURES / 'qwen1.5-moe-a2.7b/source.json').read_text())
    config = json.loads(Path(hf_hub_download(source['repo'], 'config.json',
                                           revision=source['revision'])).read_text())
    assert config == fixture_config('qwen1.5-moe-a2.7b')
    assert translate_config(config)['mixture']['shared_gate']


def test_qwen2_moe_windows_and_dense_exclusions_follow_the_reference():
    config = translate_config(fixture_config('qwen2-moe-tiny'))
    assert config['layer_types'] == ('sliding_attention', 'full_attention', 'sliding_attention')
    assert config['mixture']['layers'] == (0, 2)
    assert config['kinds']['sliding_attention']['window'] == 4
    model, variables = fp32_decoder(QWEN2_MOE)
    assert 'experts' not in variables['params']['layers_1']['mlp']
    assert 'shared_experts' in variables['params']['layers_2']['mlp']
    assert model.mixture.shared_features == 24


def test_qwen2_moe_logits_match_the_reference():
    model, variables = fp32_decoder(QWEN2_MOE)
    ids = np.load(QWEN2_MOE / 'input_ids.npy')
    reference = np.load(QWEN2_MOE / 'logits.npy')
    truth = np.load(QWEN2_MOE / 'logits_f64.npy')
    logits = np.asarray(model.apply(variables, ids))
    assert_as_exact_as_the_reference(logits, reference, truth, 'Qwen2-MoE logits')
    np.testing.assert_array_equal(logits.argmax(-1), reference.argmax(-1))
    print('Qwen2-MoE RMS ratio', distance(logits, truth) / distance(reference, truth))


def test_qwen2_moe_decode_matches_the_reference():
    model, variables = fp32_decoder(QWEN2_MOE)
    ids = np.load(QWEN2_MOE / 'input_ids.npy')
    state = model.apply(variables, ids.shape[0], method='init_cache', mutable=['cache'])[1]
    out, state = model.apply({**variables, **state}, ids[:, :4], decode=True, mutable=['cache'])
    pieces = [np.asarray(out)]
    for index in range(4, ids.shape[1]):
        out, state = model.apply({**variables, **state}, ids[:, index:index + 1],
                                 decode=True, mutable=['cache'])
        pieces.append(np.asarray(out))
    assert_as_exact_as_the_reference(np.concatenate(pieces, axis=1),
        np.load(QWEN2_MOE / 'logits.npy'), np.load(QWEN2_MOE / 'logits_f64.npy'), 'Qwen2-MoE cache')


def test_qwen2_moe_padded_routing_matches_the_reference():
    model, variables = fp32_decoder(QWEN2_MOE)
    arrays = np.load(QWEN2_MOE / 'padded.npz')
    logits = np.asarray(model.apply(
        variables, arrays['input_ids'], positions=arrays['position_ids'],
        attention_mask=arrays['attention_mask']))
    valid = arrays['attention_mask']
    assert_as_exact_as_the_reference(logits[valid], arrays['logits'][valid],
                                     arrays['logits_f64'][valid], 'Qwen2-MoE padded')
    print('Qwen2-MoE padded RMS ratio', distance(logits[valid], arrays['logits_f64'][valid])
          / distance(arrays['logits'][valid], arrays['logits_f64'][valid]))


def test_qwen2_moe_export_keeps_fused_experts_and_shared_gate(tmp_path):
    loaded = Pretrained.load(str(QWEN2_MOE), dtype='float32', attention_impl='reference')
    loaded.save(tmp_path / 'export')
    again = Pretrained.load(str(tmp_path / 'export'), dtype='float32', attention_impl='reference')
    for path, leaf in flat_tree(loaded.variables).items():
        np.testing.assert_array_equal(leaf, flat_tree(again.variables)[path])
    assert json.loads((tmp_path / 'export/config.json').read_text()) == fixture_config('qwen2-moe-tiny')


def test_qwen2_moe_runs_a_backward_update():
    model, variables = fp32_decoder(QWEN2_MOE)
    ids = jnp.asarray(np.load(QWEN2_MOE / 'input_ids.npy'))

    def loss(params):
        return jnp.mean(jnp.square(model.apply({'params': params}, ids)))

    before, gradients = jax.value_and_grad(loss)(variables['params'])
    updated = jax.tree_util.tree_map(lambda p, g: p - 1e-3 * g, variables['params'], gradients)
    assert np.isfinite(float(before)) and float(loss(updated)) < float(before)


def test_qwen2_moe_reads_qkv_bias_without_biasing_the_output():
    config = translate_config({**fixture_config('qwen2-moe-tiny'), 'qkv_bias': False})
    assert config['attention_bias'] is False and config['o_proj_bias'] is False


GRANITEMOE = FIXTURES / 'granitemoe-tiny'


def test_granitemoe_released_config_keeps_all_four_multipliers():
    config = translate_config(fixture_config('powermoe-3b'))
    assert config['embedding_multiplier'] == 12.0 and config['residual_multiplier'] == 0.22
    assert config['logits_scaling'] == 6.0 and config['attention_scale'] == 0.015625
    assert config['mixture'] == {'experts': 40, 'top_k': 8}
    assert (config['num_layers'], config['num_heads'], config['num_kv_heads']) == (32, 24, 8)
    assert config['mlp'] == 'swiglu' and config['tie_embeddings']


@pytest.mark.network
def test_granitemoe_pinned_config_reads_as_the_committed_release():
    from huggingface_hub import hf_hub_download

    source = json.loads((FIXTURES / 'powermoe-3b/source.json').read_text())
    config = json.loads(Path(hf_hub_download(source['repo'], 'config.json',
                                           revision=source['revision'])).read_text())
    assert config == fixture_config('powermoe-3b')
    assert translate_config(config)['residual_multiplier'] == 0.22


@pytest.mark.parametrize('changes, field', (
    ({'activation_function': 'gelu'}, 'activation_function'),
    ({'router_jitter_noise': 0.1}, 'router_jitter_noise'),
))
def test_granitemoe_refuses_fields_that_disagree_with_the_reference(changes, field):
    with pytest.raises(ValueError, match=field):
        translate_config({**fixture_config('powermoe-3b'), **changes})


def test_granitemoe_logits_match_the_reference():
    model, variables = fp32_decoder(GRANITEMOE)
    ids = np.load(GRANITEMOE / 'input_ids.npy')
    reference = np.load(GRANITEMOE / 'logits.npy')
    truth = np.load(GRANITEMOE / 'logits_f64.npy')
    logits = np.asarray(model.apply(variables, ids))
    assert_as_exact_as_the_reference(logits, reference, truth, 'Granite MoE logits')
    np.testing.assert_array_equal(logits.argmax(-1), reference.argmax(-1))
    print('Granite MoE RMS ratio', distance(logits, truth) / distance(reference, truth))


def test_granitemoe_prefill_and_steps_match_the_reference():
    model, variables = fp32_decoder(GRANITEMOE)
    ids = np.load(GRANITEMOE / 'input_ids.npy')
    state = model.apply(variables, ids.shape[0], method='init_cache', mutable=['cache'])[1]
    out, state = model.apply({**variables, **state}, ids[:, :4], decode=True, mutable=['cache'])
    pieces = [np.asarray(out)]
    for index in range(4, ids.shape[1]):
        out, state = model.apply({**variables, **state}, ids[:, index:index + 1],
                                 decode=True, mutable=['cache'])
        pieces.append(np.asarray(out))
    assert_as_exact_as_the_reference(np.concatenate(pieces, axis=1),
        np.load(GRANITEMOE / 'logits.npy'), np.load(GRANITEMOE / 'logits_f64.npy'), 'Granite MoE cache')


def test_granitemoe_padded_routing_matches_the_reference():
    model, variables = fp32_decoder(GRANITEMOE)
    arrays = np.load(GRANITEMOE / 'padded.npz')
    logits = np.asarray(model.apply(
        variables, arrays['input_ids'], positions=arrays['position_ids'],
        attention_mask=arrays['attention_mask']))
    valid = arrays['attention_mask']
    assert_as_exact_as_the_reference(logits[valid], arrays['logits'][valid],
                                     arrays['logits_f64'][valid], 'Granite MoE padded')
    print('Granite MoE padded RMS ratio', distance(logits[valid], arrays['logits_f64'][valid])
          / distance(arrays['logits'][valid], arrays['logits_f64'][valid]))


def test_granitemoe_export_keeps_packed_experts_and_tied_head(tmp_path):
    from safetensors.numpy import load_file

    loaded = Pretrained.load(str(GRANITEMOE), dtype='float32', attention_impl='reference')
    loaded.save(tmp_path / 'export')
    tensors = load_file(str(tmp_path / 'export/model.safetensors'))
    assert 'model.layers.0.block_sparse_moe.input_linear.weight' in tensors
    assert 'lm_head.weight' not in tensors
    again = Pretrained.load(str(tmp_path / 'export'), dtype='float32', attention_impl='reference')
    for path, leaf in flat_tree(loaded.variables).items():
        np.testing.assert_array_equal(leaf, flat_tree(again.variables)[path])
    assert json.loads((tmp_path / 'export/config.json').read_text()) == fixture_config('granitemoe-tiny')


def test_granitemoe_runs_a_backward_update():
    model, variables = fp32_decoder(GRANITEMOE)
    ids = jnp.asarray(np.load(GRANITEMOE / 'input_ids.npy'))

    def loss(params):
        return jnp.mean(jnp.square(model.apply({'params': params}, ids)))

    before, gradients = jax.value_and_grad(loss)(variables['params'])
    updated = jax.tree_util.tree_map(lambda p, g: p - 1e-3 * g, variables['params'], gradients)
    assert np.isfinite(float(before)) and float(loss(updated)) < float(before)


MINIMAX_M2 = FIXTURES / 'minimax-m2-tiny'


@pytest.mark.parametrize('name', ('minimax-m2', 'minimax-m2.5', 'minimax-m2.7'))
def test_minimax_m2_released_configs_read_partial_rotary_and_sigmoid_routing(name):
    config = translate_config(fixture_config(name))
    assert config['qk_norm_scope'] == 'projection'
    assert config['partial_rotary_factor'] == 0.5
    assert config['partial_rotary_type'] == 'default'
    assert config['mixture'] == {'experts': 256, 'top_k': 8, 'score_function': 'sigmoid', 'bias': True}
    assert (config['num_layers'], config['num_heads'], config['num_kv_heads']) == (62, 48, 8)


@pytest.mark.network
@pytest.mark.parametrize('name', ('minimax-m2', 'minimax-m2.5', 'minimax-m2.7'))
def test_minimax_m2_pinned_configs_and_indexes_have_no_prediction_weights(name):
    from huggingface_hub import hf_hub_download

    source = json.loads((FIXTURES / name / 'source.json').read_text())
    config = json.loads(Path(hf_hub_download(source['repo'], 'config.json',
                                           revision=source['revision'])).read_text())
    assert config == fixture_config(name)
    assert translate_config(config)['partial_rotary_factor'] == 0.5
    index = json.loads(Path(hf_hub_download(source['repo'], 'model.safetensors.index.json',
                                          revision=source['revision'])).read_text())
    assert not any('mtp' in key for key in index['weight_map'])


@pytest.mark.parametrize('changes, reason', (
    ({'use_qk_norm': False}, 'query and key projections'),
    ({'qk_norm_type': 'per_head'}, 'whole projections before splitting heads'),
    ({'use_routing_bias': False}, 'balancing bias'),
    ({'scoring_func': 'softmax'}, 'selected sigmoid probabilities'),
    ({'router_jitter_noise': 0.1}, 'training-time input jitter'),
    ({'partial_rotary_factor': 1.0}, 'rotary_dim'),
    ({'shared_intermediate_size': 4}, 'no shared expert branch'),
))
def test_minimax_m2_refuses_settings_that_change_the_released_block(changes, reason):
    with pytest.raises(ValueError, match=reason):
        translate_config({**fixture_config('minimax-m2'), **changes})


@pytest.mark.parametrize('name', ('minimax-m2-tiny', 'minimax-m2-fp8-tiny'))
def test_minimax_m2_logits_match_the_corrected_reference(name):
    directory = FIXTURES / name
    model, variables = fp32_decoder(directory)
    ids = np.load(directory / 'input_ids.npy')
    reference = np.load(directory / 'logits.npy')
    truth = np.load(directory / 'logits_f64.npy')
    logits = np.asarray(model.apply(variables, ids))
    assert_as_exact_as_the_reference(logits, reference, truth, 'MiniMax-M2 logits')
    np.testing.assert_array_equal(logits.argmax(-1), reference.argmax(-1))
    print(name, 'RMS ratio', distance(logits, truth) / distance(reference, truth))


def test_minimax_m2_prefill_and_token_steps_match_the_corrected_reference():
    model, variables = fp32_decoder(MINIMAX_M2)
    ids = np.load(MINIMAX_M2 / 'input_ids.npy')
    state = model.apply(variables, ids.shape[0], method='init_cache', mutable=['cache'])[1]
    out, state = model.apply({**variables, **state}, ids[:, :4], decode=True, mutable=['cache'])
    pieces = [np.asarray(out)]
    for index in range(4, ids.shape[1]):
        out, state = model.apply({**variables, **state}, ids[:, index:index + 1],
                                 decode=True, mutable=['cache'])
        pieces.append(np.asarray(out))
    assert_as_exact_as_the_reference(np.concatenate(pieces, axis=1),
        np.load(MINIMAX_M2 / 'logits.npy'), np.load(MINIMAX_M2 / 'logits_f64.npy'), 'MiniMax-M2 cache')


def test_minimax_m2_padded_routing_matches_the_corrected_reference():
    model, variables = fp32_decoder(MINIMAX_M2)
    arrays = np.load(MINIMAX_M2 / 'padded.npz')
    logits = np.asarray(model.apply(
        variables, arrays['input_ids'], positions=arrays['position_ids'],
        attention_mask=arrays['attention_mask']))
    valid = arrays['attention_mask']
    assert_as_exact_as_the_reference(logits[valid], arrays['logits'][valid],
                                     arrays['logits_f64'][valid], 'MiniMax-M2 padded')
    print('MiniMax-M2 padded RMS ratio', distance(logits[valid], arrays['logits_f64'][valid])
          / distance(arrays['logits'][valid], arrays['logits_f64'][valid]))


def test_minimax_m2_export_keeps_source_weights_and_router_state(tmp_path):
    loaded = Pretrained.load(str(MINIMAX_M2), dtype='float32', attention_impl='reference')
    loaded.save(tmp_path / 'export')
    again = Pretrained.load(str(tmp_path / 'export'), dtype='float32', attention_impl='reference')
    for path, leaf in flat_tree(loaded.variables).items():
        np.testing.assert_array_equal(leaf, flat_tree(again.variables)[path])
    assert json.loads((tmp_path / 'export/config.json').read_text()) == fixture_config('minimax-m2-tiny')


def test_minimax_m2_runs_a_backward_update_without_moving_router_bias():
    model, variables = fp32_decoder(MINIMAX_M2)
    ids = jnp.asarray(np.load(MINIMAX_M2 / 'input_ids.npy'))

    def loss(params):
        return jnp.mean(jnp.square(model.apply({**variables, 'params': params}, ids)))

    before, gradients = jax.value_and_grad(loss)(variables['params'])
    updated = jax.tree_util.tree_map(lambda p, g: p - 1e-3 * g, variables['params'], gradients)
    assert np.isfinite(float(before)) and float(loss(updated)) < float(before)
    assert set(variables['moe']['layers_0']['mlp']['gate']) == {'e_score_correction_bias'}


def test_minimax_m2_refuses_unimplemented_prediction_tensors():
    config = translate_config(fixture_config('minimax-m2-tiny'))
    with pytest.raises(ValueError, match='mtp'):
        translate_weights({'mtp.0.weight': np.ones((2, 2), np.float32)}, config, 'minimax_m2')

# --------------------------------------------------------------------------
# GPT OSS: attention sinks, biased interleaved experts, YaRN over GQA
# --------------------------------------------------------------------------

GPT_OSS = FIXTURES / "gpt-oss-tiny"


def test_gpt_oss_config_translates_field_by_field():
    config = translate_config(fixture_config("gpt-oss-tiny"))

    assert config["mlp"] == "swigluoai" and config["attention_sinks"]
    assert config["layer_types"] == ("sliding_attention", "full_attention")
    assert config["kinds"] == {"sliding_attention": {"window": 4}}
    assert config["mixture"] == {"experts": 4, "top_k": 2}
    assert config["yarn"]["factor"] == 4.0 and config["yarn"]["truncate"] is False
    assert config["rope_theta"] == 150000.0 and config["attention_bias"]
    assert not config["qk_norm"] and not config["scale_after_cast"]


def test_the_real_gpt_oss_20b_config_translates():
    """openai/gpt-oss-20b, released with MXFP4 experts: 24 alternating
    layers, 64 query and 8 key/value heads of 64, 32 experts with 4 per
    token, YaRN factor 32 over 4096 positions, and the sliding window 128."""
    config = translate_config(fixture_config("gpt-oss-20b"))

    assert config["num_layers"] == 24 and config["layer_types"][:2] == (
        "sliding_attention", "full_attention")
    assert (config["num_heads"], config["num_kv_heads"], config["head_dim"]) == (64, 8, 64)
    assert config["mixture"] == {"experts": 32, "top_k": 4}
    assert config["kinds"] == {"sliding_attention": {"window": 128}}
    assert config["yarn"]["factor"] == 32.0
    assert config["yarn"]["original_max_position_embeddings"] == 4096
    assert config["vocab_size"] == 201088 and config["max_seq_len"] == 8192


@pytest.mark.parametrize("field, value, message", [
    ("swiglu_limit", 3.0, "swiglu_limit"),
    ("quantization_config", {"quant_method": "awq"}, "quantization_config"),
    ("output_router_logits", True, "output_router_logits"),
])
def test_a_gpt_oss_field_with_no_counterpart_is_refused(field, value, message):
    with pytest.raises(ValueError, match=message):
        translate_config({**fixture_config("gpt-oss-tiny"), field: value})


def test_gpt_oss_logits_match_the_reference_implementation():
    """fp32 parity through xla sink attention: tolerance 1e-4, observed
    max |logit difference| 2.5e-06 with identical argmax."""
    pretrained = Pretrained.load(str(GPT_OSS), dtype="float32", attention_impl="xla")
    model, variables = pretrained.model, pretrained.variables
    ids = np.load(GPT_OSS / "input_ids.npy")
    reference = np.load(GPT_OSS / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))
    flat = flat_tree(variables["params"])
    assert flat["layers_0.self_attn.sinks"].shape == (4,)
    assert flat["layers_1.mlp.experts.gate_up_proj_bias"].shape == (4, 128)


def test_gpt_oss_export_round_trips_sinks_and_fused_experts(tmp_path):
    pretrained = Pretrained.load(str(GPT_OSS), dtype="float32", attention_impl="xla")
    model, variables = pretrained.model, pretrained.variables
    export = tmp_path / "gpt-oss"
    PretrainedDecoder.from_model(model, variables).save(export)
    round_trip = Pretrained.load(str(export), dtype="float32", attention_impl="xla")
    again, reloaded = round_trip.model, round_trip.variables
    assert again == model
    for path, leaf in flat_tree(reloaded["params"]).items():
        assert np.array_equal(np.asarray(leaf), np.asarray(flat_tree(variables["params"])[path])), path


def test_an_mxfp4_gpt_oss_checkpoint_loads_through_the_dequantization(tmp_path):
    """The released layout: uint8 blocks and scales in place of the expert
    matrices. Loading them must give the logits of the fixture whose
    experts are those exact bf16 values, so the packed checkpoint is
    written from the fixture's own weights."""
    from safetensors.numpy import load_file, save_file

    tensors = load_file(str(GPT_OSS / "model.safetensors"))
    packed = {}
    for name, tensor in tensors.items():
        if not (name.endswith(("gate_up_proj", "down_proj"))):
            packed[name] = tensor
            continue
        blocks, scales = (np.asarray(part) for part in quantize_mxfp4(jnp.asarray(tensor)))
        packed[name + "_blocks"], packed[name + "_scales"] = blocks, scales
        tensors[name] = np.asarray(dequantize_mxfp4(jnp.asarray(blocks), jnp.asarray(scales))
                                   .astype(jnp.float32))
    directory = tmp_path / "packed"
    directory.mkdir()
    save_file(packed, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(
        {**fixture_config("gpt-oss-tiny"), "quantization_config": {"quant_method": "mxfp4"}}))

    pretrained = Pretrained.load(str(directory), dtype="float32", attention_impl="xla")
    _model, variables = pretrained.model, pretrained.variables
    expected = translate_weights(tensors, translate_config(fixture_config("gpt-oss-tiny")))
    for path, leaf in flat_tree(variables["params"]).items():
        assert np.array_equal(np.asarray(leaf), flat_tree(expected["params"])[path]), path


@pytest.mark.network
@pytest.mark.skipif(not os.environ.get("DEW_NETWORK_TESTS"),
                    reason="DEW_NETWORK_TESTS=1 downloads openai/gpt-oss-20b")
def test_gpt_oss_20b_matches_transformers_on_the_real_weights():
    """The released MXFP4 checkpoint through the loader against transformers'
    own dequantized bf16 forward at the same prompt. Tolerance on the top-32
    logits 5e-2 for bf16 accumulation over 24 layers; the argmax must agree
    everywhere. The largest observed difference is not recorded here: the
    test needs the download."""
    import torch
    from transformers import AutoTokenizer, GptOssForCausalLM

    prompt = "The Cascade Range runs from northern California through Oregon"
    ids = AutoTokenizer.from_pretrained("openai/gpt-oss-20b")(
        prompt, return_tensors="np")["input_ids"].astype(np.int32)
    pretrained = Pretrained.load("openai/gpt-oss-20b", dtype="bfloat16",
                                 attention_impl="xla",
                                 max_seq_len=int(ids.shape[1]))
    model, variables = pretrained.model, pretrained.variables
    logits = np.asarray(model.apply(variables, jnp.asarray(ids)), np.float32)[0]

    reference = GptOssForCausalLM.from_pretrained("openai/gpt-oss-20b", dtype=torch.bfloat16)
    reference.eval()
    reference.set_attn_implementation("eager")
    with torch.no_grad():
        theirs = reference(input_ids=torch.from_numpy(ids.astype(np.int64))).logits[0].float().numpy()

    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(theirs, axis=-1))
    top_ids = np.argsort(-theirs, axis=-1)[:, :32]
    difference = float(np.max(np.abs(np.take_along_axis(logits, top_ids, axis=-1)
                                     - np.take_along_axis(theirs, top_ids, axis=-1))))
    assert difference < 5e-2, f"max |top-32 logit difference| {difference:.3e}"



# --------------------------------------------------------------------------
# DeepSeek V2 and Kimi K2: the V2 router under MLA, and Kimi's own release
# --------------------------------------------------------------------------

DEEPSEEK_V2 = FIXTURES / "deepseek-v2-tiny"


def test_deepseek_v2_config_translates_field_by_field():
    """The tiny V2: MLA without the query LoRA, YaRN with the 0.707 mscales,
    a dense first layer over a softmax router under group_limited_greedy
    that never renormalises, and two shared experts of width 16."""
    config = translate_config(fixture_config("deepseek-v2-tiny"))

    assert config["mixer"]["class"] == "mla" and config["mixer"]["fields"]["q_lora_rank"] is None
    assert config["mixer"]["fields"]["yarn"]["mscale"] == 0.707
    assert config["mixture"] == {
        "experts": 8, "top_k": 4, "layers": (1,), "scaling": 2.5, "shared_features": 32,
        "expert_features": 16, "score_function": "softmax", "norm_topk_prob": False,
        "groups": 4, "groups_per_token": 2, "group_score": "max"}
    assert config["head_dim"] == 16 and config["scale_after_cast"]


def test_the_real_deepseek_v2_lite_config_translates():
    """deepseek-ai/DeepSeek-V2-Lite: 27 layers with one dense, 64 experts
    with 6 per token under greedy selection, two shared experts of 1408,
    kv_lora_rank 512 with no query LoRA, YaRN factor 40 at mscale 0.707."""
    config = translate_config(fixture_config("deepseek-v2-lite"))

    assert config["num_layers"] == 27 and config["mixture"]["layers"] == tuple(range(1, 27))
    assert config["mixture"]["experts"] == 64 and config["mixture"]["top_k"] == 6
    assert config["mixture"]["shared_features"] == 2816
    assert config["mixture"]["groups"] == 1 and config["mixture"]["group_score"] == "max"
    assert config["mixture"]["norm_topk_prob"] is False
    mixer = config["mixer"]["fields"]
    assert mixer["kv_lora_rank"] == 512 and mixer["q_lora_rank"] is None
    assert config["mixer"]["fields"]["yarn"]["factor"] == 40.0
    assert config["mixer"]["fields"]["yarn"]["mscale_all_dim"] == 0.707
    assert config["vocab_size"] == 102400 and config["head_dim"] == 192


@pytest.mark.parametrize("field, value, message", [
    ("norm_topk_prob", True, "norm_topk_prob=True"),
    ("topk_method", "noaux_tc", "topk_method 'noaux_tc'"),
    ("scoring_func", "sigmoid", "scoring_func 'sigmoid'"),
])
def test_a_deepseek_v2_field_the_reference_does_not_run_is_refused(field, value, message):
    with pytest.raises(ValueError, match=message):
        translate_config({**fixture_config("deepseek-v2-tiny"), field: value})


def test_deepseek_v2_logits_match_the_reference_implementation():
    """fp32 parity: tolerance 1e-4, observed max |logit difference| 2.3e-06
    with identical argmax."""
    model, variables = fp32_decoder(DEEPSEEK_V2)
    ids = np.load(DEEPSEEK_V2 / "input_ids.npy")
    reference = np.load(DEEPSEEK_V2 / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))
    assert "moe" not in variables, "V2 keeps no balancing bias"


def test_the_real_kimi_k2_config_translates():
    """moonshotai/Kimi-K2-Instruct is DeepSeek V3's computation: 61 layers
    with one dense, 384 sigmoid-scored experts with 8 per token in one
    group, scaled by 2.827, one shared expert, MLA with the 1536 query LoRA,
    YaRN factor 32 at theta 50000, under a 163840-token vocabulary."""
    config = translate_config(fixture_config("kimi-k2"))

    assert config["vocab_size"] == 163840 and config["num_layers"] == 61
    assert config["mixture"]["experts"] == 384 and config["mixture"]["top_k"] == 8
    assert config["mixture"]["scaling"] == 2.827 and config["mixture"]["bias"]
    assert config["mixture"]["groups"] == 1 and config["mixture"]["shared_features"] == 2048
    assert config["mixer"]["fields"]["q_lora_rank"] == 1536
    assert config["mixer"]["fields"]["yarn"]["factor"] == 32.0 and config["rope_theta"] == 50000.0


def test_the_kimi_k2_fixture_translates_the_releases_own_choices():
    """kimi-k2-tiny scales the release's widths and keeps its choices, so
    both configs translate to the same routing, rope and MLA record, and
    neither of them is DeepSeek V3's.

    The release's own values the fixture holds unscaled are the sigmoid
    scores under the balancing bias, one group holding every expert, the
    2.827 scaling, rope theta 50000 and YaRN factor 32 at beta_fast and
    beta_slow 1.0 over an original context of 4096.
    """
    tiny = translate_config(fixture_config("kimi-k2-tiny"))
    released = translate_config(fixture_config("kimi-k2"))
    v3 = translate_config(fixture_config("deepseek-v3-tiny"))

    shared = ("score_function", "bias", "groups", "groups_per_token", "scaling")
    assert ({name: tiny["mixture"][name] for name in shared}
            == {name: released["mixture"][name] for name in shared}
            == {"score_function": "sigmoid", "bias": True, "groups": 1,
                "groups_per_token": 1, "scaling": 2.827})
    assert tiny["mixer"]["fields"]["yarn"] == released["mixer"]["fields"]["yarn"]
    assert tiny["mixer"]["fields"]["yarn"]["factor"] == 32.0
    assert (tiny["mixer"]["fields"]["yarn"]["beta_fast"],
            tiny["mixer"]["fields"]["yarn"]["beta_slow"]) == (1.0, 1.0)
    assert tiny["mixer"]["fields"]["yarn"]["original_max_position_embeddings"] == 4096
    assert tiny["rope_theta"] == released["rope_theta"] == 50000.0
    assert tiny["mixture"]["layers"] == (1,) and tiny["num_layers"] == 2

    # A copy of the V3 fixture under Kimi's model_type is what the audit
    # found here. V3 groups its experts four ways, scales by 2.5 and runs a
    # factor-40 ramp over theta 10000, so none of these agree.
    differ = ("groups", "groups_per_token", "scaling")
    for name in differ:
        assert tiny["mixture"][name] != v3["mixture"][name], name
    assert tiny["rope_theta"] != v3["rope_theta"]
    assert tiny["mixer"]["fields"]["yarn"]["factor"] != v3["mixer"]["fields"]["yarn"]["factor"]


def test_the_kimi_k2_fixture_keeps_the_releases_mla_proportions():
    """The release makes qk_nope twice qk_rope and the values as wide as
    qk_nope. The V3 fixture's three equal widths hide that split, so a
    query head that halved its nope and rope slices evenly would load
    there and fail here."""
    tiny = translate_config(fixture_config("kimi-k2-tiny"))
    released = translate_config(fixture_config("kimi-k2"))

    for config in (tiny, released):
        assert config["mixer"]["class"] == "mla"
        mixer = config["mixer"]["fields"]
        assert mixer["qk_nope_head_dim"] == 2 * mixer["qk_rope_head_dim"]
        assert mixer["v_head_dim"] == mixer["qk_nope_head_dim"]
        assert config["head_dim"] == mixer["qk_nope_head_dim"] + mixer["qk_rope_head_dim"]
    assert tiny["mixer"]["fields"]["q_lora_rank"] == 8 and released["mixer"]["fields"]["q_lora_rank"] == 1536


# --------------------------------------------------------------------------
# Kimi K2.5: a vision wrapper whose decoder is Kimi K2's, text half only
# --------------------------------------------------------------------------

KIMI_K25 = FIXTURES / "kimi-k25-tiny"


def test_kimi_k25_translates_the_wrapper_into_its_text_decoder():
    """The wrapper's own fields name a vision tower, a projector and the
    media marks it fills, none of which reach the decoder; what translates
    is text_config, and the head is the wrapper's.

    The tiny fixture scales the release's widths and keeps its choices, so
    the two configs agree on the routing, the rope and the MLA record. What
    K2.5's text config changes from K2-Instruct's, and both of these hold,
    is rms_norm_eps 1e-5 and the YaRN factor 64 at beta_fast 32.
    """
    tiny = translate_config(fixture_config("kimi-k25-tiny"))
    released = translate_config(fixture_config("kimi-k25"))
    k2 = translate_config(fixture_config("kimi-k2-tiny"))

    shared = ("score_function", "bias", "groups", "groups_per_token", "scaling")
    assert ({name: tiny["mixture"][name] for name in shared}
            == {name: released["mixture"][name] for name in shared}
            == {"score_function": "sigmoid", "bias": True, "groups": 1,
                "groups_per_token": 1, "scaling": 2.827})
    assert tiny["mixer"]["fields"]["yarn"] == released["mixer"]["fields"]["yarn"]
    assert (tiny["mixer"]["fields"]["yarn"]["factor"], tiny["mixer"]["fields"]["yarn"]["beta_fast"],
            tiny["mixer"]["fields"]["yarn"]["beta_slow"]) == (64.0, 32.0, 1.0)
    assert tiny["mixer"]["fields"]["yarn"]["original_max_position_embeddings"] == 4096
    assert tiny["rope_theta"] == released["rope_theta"] == 50000.0
    assert tiny["norm_eps"] == released["norm_eps"] == 1e-5
    assert tiny["mixer"]["class"] == "mla" and tiny["mixture"]["layers"] == (1,)
    assert tiny["mixer"]["fields"]["qk_nope_head_dim"] == 2 * tiny["mixer"]["fields"]["qk_rope_head_dim"]
    assert tiny["mixer"]["fields"]["v_head_dim"] == tiny["mixer"]["fields"]["qk_nope_head_dim"]
    # The wrapper's tie_word_embeddings decides the head, not the nested
    # text config's, which describes a DeepseekV3Model with no head.
    assert tiny["tie_embeddings"] is False and released["tie_embeddings"] is False
    assert translate_config({**fixture_config("kimi-k25-tiny"),
                             "tie_word_embeddings": True})["tie_embeddings"] is True

    # K2.5 is not K2-Instruct: the ramp and the norm epsilon moved.
    assert k2["mixer"]["fields"]["yarn"]["factor"] == 32.0 and k2["norm_eps"] == 1e-6


def test_the_real_kimi_k25_config_translates():
    """moonshotai/Kimi-K2.5 releases the remote-code spelling: a wrapper
    whose text_config is model_type kimi_k2 with every PreTrainedConfig
    attribute transformers 4.56.2 serialized beside it, and whose media
    mark is media_placeholder_token_id rather than image_token_id. Its
    decoder is K2's 61 layers with one dense, 384 sigmoid-scored experts
    with 8 per token in one group, and MLA with the 1536 query LoRA over a
    192-wide query head."""
    config = translate_config(fixture_config("kimi-k25"))

    assert config["vocab_size"] == 163840 and config["num_layers"] == 61
    assert config["mixture"]["experts"] == 384 and config["mixture"]["top_k"] == 8
    assert config["mixture"]["shared_features"] == 2048
    assert config["mixture"]["layers"] == tuple(range(1, 61))
    assert (config["num_heads"], config["num_kv_heads"]) == (64, 64)
    mixer = config["mixer"]["fields"]
    assert mixer["q_lora_rank"] == 1536 and mixer["kv_lora_rank"] == 512
    assert mixer["qk_nope_head_dim"] == 128 and mixer["qk_rope_head_dim"] == 64
    assert config["head_dim"] == 192 and config["norm_eps"] == 1e-5


@pytest.mark.parametrize("field, value, message", [
    ("text_config", {"model_type": "qwen3_moe"}, "text_config model_type 'qwen3_moe'"),
    ("text_config", 7, "the wrapper carries its decoder under text_config"),
    ("tie_word_embeddings", "yes", "tie_word_embeddings 'yes'"),
    ("audio_config", {"hidden_size": 8}, r"config fields \['audio_config'\]"),
])
def test_a_kimi_k25_wrapper_field_with_no_counterpart_is_refused(field, value, message):
    with pytest.raises(ValueError, match=message):
        translate_config({**fixture_config("kimi-k25-tiny"), field: value})


@pytest.mark.parametrize("field, value", [
    ("add_cross_attention", True),
    ("cross_attention_hidden_size", 128),
    ("tie_encoder_decoder", True),
    ("pruned_heads", {"0": [1]}),
])
def test_a_kimi_k25_text_config_that_names_an_encoder_is_refused(field, value):
    """The release's text_config carries transformers 4.x's whole
    PreTrainedConfig serialization at its defaults. The four fields of it
    that would name another model are read by value, not accepted by
    name."""
    config = fixture_config("kimi-k25")
    text = {**config["text_config"], field: value}
    with pytest.raises(ValueError, match=f"text_config {field}"):
        translate_config({**config, "text_config": text})


def test_kimi_k25_refuses_to_load_as_a_multimodal_wrapper():
    """The tower and the projector have no counterpart here, so the
    wrapper path names the model_type rather than building half a model."""
    from dew.interop.hf_decoders import translate_wrapper_config

    with pytest.raises(ValueError, match="model_type 'kimi_k25'"):
        translate_wrapper_config(fixture_config("kimi-k25-tiny"))


def test_kimi_k25_logits_match_the_reference_implementation():
    """fp32 parity of the text half against Kimi_K25ForConditionalGeneration
    on input_ids alone: tolerance 1e-4, observed max |logit difference|
    the bound applies to the complete vocabulary at every input position."""
    model, variables = fp32_decoder(KIMI_K25)
    ids = np.load(KIMI_K25 / "input_ids.npy")
    reference = np.load(KIMI_K25 / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


def test_the_kimi_k25_fixture_carries_the_releases_ramp_and_not_kimi_k2s():
    """K2.5's text config differs from K2-Instruct's in the YaRN factor
    (64 against 32), beta_fast (32 against 1) and rms_norm_eps (1e-5
    against 1e-6). Each of K2's values on the same weights disagrees with
    the fixture by far more than the tolerance, so the fixture is K2.5's
    choices and not a relabelled K2."""
    model, variables = fp32_decoder(KIMI_K25)
    ids = jnp.asarray(np.load(KIMI_K25 / "input_ids.npy"), jnp.int32)
    reference = np.load(KIMI_K25 / "logits.npy")

    def moved(swapped):
        logits = np.asarray(swapped.apply(variables, ids))
        return float(np.max(np.abs(logits - reference)))

    yarn = model.mixer.yarn
    assert (yarn.factor, yarn.beta_fast, yarn.beta_slow) == (64.0, 32.0, 1.0)
    kimi_k2_ramp = {"factor": 32.0, "beta_fast": 1.0}
    for field, value in kimi_k2_ramp.items():
        swapped = model.clone(mixer=dataclasses.replace(
            model.mixer, yarn=dataclasses.replace(yarn, **{field: value})))
        assert moved(swapped) > 1e-2, f"{field}={value} moved {moved(swapped):.3e}"
    assert moved(model.clone(norm_eps=1e-6)) > 1e-4



def test_a_kimi_k25_load_binds_the_decoder_and_retains_the_vision_halves():
    """The release nests its decoder under `language_model.model.*` beside
    a `vision_tower.*` and a `mm_projector.*` this has no counterpart for.
    Every decoder tensor binds to a leaf, every vision tensor is retained
    by name, and the retained bytes reach the export untouched."""
    loaded = Pretrained.load(str(KIMI_K25), dtype="float32", attention_impl="reference")
    from dew.interop.sources import load_shards

    tensors = load_shards(KIMI_K25)
    bound = {layout.name for layout in loaded.weight_layouts}
    retained = set(loaded.retained_tensors)

    assert bound | retained == set(tensors)
    assert not bound & retained
    assert all(name.startswith(("language_model.model.", "language_model.lm_head."))
               for name in bound)
    assert all(name.startswith(("vision_tower.", "mm_projector.")) for name in retained)
    assert len(retained) == 35 and len(bound) == 65
    for name in retained:
        np.testing.assert_array_equal(loaded.retained_tensors[name], tensors[name])


def test_a_kimi_k25_export_writes_the_source_names_and_the_vision_bytes(tmp_path):
    """The whole checkpoint comes back out: the decoder from the parameter
    tree under the release's names, the tower and the projector from the
    bytes they were retained as, and the source's own config beside them."""
    loaded = Pretrained.load(str(KIMI_K25), dtype="float32", attention_impl="reference")
    from dew.interop.sources import load_shards

    destination = tmp_path / "export"
    loaded.save(destination)
    exported = load_shards(destination)
    source = load_shards(KIMI_K25)

    assert set(exported) == set(source)
    for name, tensor in source.items():
        np.testing.assert_array_equal(exported[name], tensor, err_msg=name)
    assert (json.loads((destination / "config.json").read_text())
            == fixture_config("kimi-k25-tiny"))
    again = Pretrained.load(str(destination), dtype="float32", attention_impl="reference")
    ids = jnp.asarray(np.load(KIMI_K25 / "input_ids.npy"), jnp.int32)
    np.testing.assert_array_equal(np.asarray(again.model.apply(again.variables, ids)),
                                  np.asarray(loaded.model.apply(loaded.variables, ids)))


def test_a_kimi_k25_tied_head_binds_to_the_nested_embedding(tmp_path):
    """Kimi_K25ForConditionalGeneration ties `lm_head.weight` to
    `model.language_model.embed_tokens.weight` (modeling_kimi_k25.py:727),
    which the checkpoint spells `language_model.lm_head.weight` and
    `language_model.model.embed_tokens.weight`. Under tying the head is the
    embedding's leaf, and a copy that is a different matrix is refused
    under those names rather than the unnested ones."""
    from shutil import copytree

    from dew.interop.safetensors_io import write_file
    from dew.interop.sources import load_shards

    source = Path(copytree(KIMI_K25, tmp_path / "tied"))
    config = fixture_config("kimi-k25-tiny")
    config["tie_word_embeddings"] = True
    (source / "config.json").write_text(json.dumps(config))
    tensors = load_shards(source)
    embedding = tensors["language_model.model.embed_tokens.weight"]
    write_file({**tensors, "language_model.lm_head.weight": embedding},
               source / "model.safetensors", {"format": "pt"})

    loaded = Pretrained.load(str(source), dtype="float32", attention_impl="reference")
    assert "lm_head" not in loaded.variables["params"]
    head = next(layout for layout in loaded.weight_layouts
                if layout.name == "language_model.lm_head.weight")
    assert head.paths == (("params", "embed_tokens", "embedding"),)

    write_file({**tensors, "language_model.lm_head.weight": embedding + 1.0},
               source / "model.safetensors", {"format": "pt"})
    with pytest.raises(ValueError,
                       match=r"language_model.lm_head.weight is not the embedding"):
        Pretrained.load(str(source), dtype="float32", attention_impl="reference")


# --------------------------------------------------------------------------
# GLM 4.5: a half rotary over biased GQA, DeepSeek V3 routing, an MTP depth
# --------------------------------------------------------------------------

GLM4_MOE = FIXTURES / "glm4-moe-tiny"


def test_glm4_moe_config_translates_field_by_field():
    config = translate_config(fixture_config("glm4-moe-tiny"))

    assert config["attention_bias"] and config["o_proj_bias"] is False
    assert config["qk_norm"] and config["scale_after_cast"]
    assert config["partial_rotary_factor"] == 0.5 and config["partial_rotary_type"] == "default"
    assert config["rope_theta"] == 1e6 and config["head_dim"] == 8
    assert config["mixture"] == {
        "experts": 8, "top_k": 2, "layers": (1,), "scaling": 1.5, "shared_features": 16,
        "expert_features": 16, "score_function": "sigmoid", "groups": 1,
        "groups_per_token": 1, "bias": True}
    assert config["num_nextn_predict_layers"] == 1


def test_the_real_glm_4_5_air_config_translates():
    """zai-org/GLM-4.5-Air: 46 layers with one dense, 128 sigmoid-scored
    experts with 8 per token, one shared expert of 1408, 96 query and 8
    key/value heads of 128 with biased q/k/v and a bias-free o_proj, a half
    rotary at theta 1e6, no q/k norms, and one MTP depth."""
    config = translate_config(fixture_config("glm-4.5-air"))

    assert config["num_layers"] == 46 and config["mixture"]["layers"] == tuple(range(1, 46))
    assert config["mixture"]["experts"] == 128 and config["mixture"]["top_k"] == 8
    assert config["mixture"]["shared_features"] == 1408 and config["mixture"]["bias"]
    assert (config["num_heads"], config["num_kv_heads"], config["head_dim"]) == (96, 8, 128)
    assert config["attention_bias"] and config["o_proj_bias"] is False
    assert not config["qk_norm"]
    assert config["partial_rotary_factor"] == 0.5 and config["rope_theta"] == 1e6
    assert config["num_nextn_predict_layers"] == 1 and config["vocab_size"] == 151552


@pytest.mark.parametrize("field, value, message", [
    ("rope_scaling", {"rope_type": "yarn", "factor": 4.0}, "plain rotary"),
    ("norm_topk_prob", False, "norm_topk_prob=False"),
    ("topk_method", "greedy", "topk_method 'greedy'"),
])
def test_a_glm4_moe_field_with_no_counterpart_is_refused(field, value, message):
    with pytest.raises(ValueError, match=message):
        translate_config({**fixture_config("glm4-moe-tiny"), field: value})


def test_glm4_moe_logits_match_the_reference_implementation():
    """fp32 parity of the trunk: tolerance 1e-4, observed max |logit
    difference| 3.3e-06 with identical argmax."""
    model, variables = fp32_decoder(GLM4_MOE)
    ids = np.load(GLM4_MOE / "input_ids.npy")
    reference = np.load(GLM4_MOE / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


def test_the_glm4_moe_mtp_depth_matches_the_reference_composition():
    """The checkpoint's model.layers.2.* depth loads as mtp_0 with the
    trunk's routing and computes what the engines running the released
    weights compute (tools/hf_reference_b.py Glm4MoeMTP): tolerance 1e-4,
    observed max |logit difference| 3.2e-06 with identical argmax. Swapping
    the two norms, the order today's from-scratch code had, disagrees."""
    from dew.nn.backbones.causal_transformer import CausalTransformer

    model, variables = fp32_decoder(GLM4_MOE)
    ids = jnp.asarray(np.load(GLM4_MOE / "input_ids.npy"), jnp.int32)
    reference = np.load(GLM4_MOE / "mtp_logits.npy")
    depth = variables["params"]["mtp_0"]
    assert set(depth) == {"enorm", "hnorm", "eh_proj", "block", "final_norm"}
    assert depth["block"]["mlp"]["experts"]["gate_proj"]["kernel"].shape == (8, 32, 16)
    assert "mtp_0" in variables["moe"]

    hidden = model.apply(variables, ids, method=CausalTransformer.hidden_states)
    logits = np.asarray(model.apply(variables, hidden, ids, method=CausalTransformer.mtp_logits)[0])
    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |mtp logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))

    swapped = {**variables, "params": {**variables["params"], "mtp_0": {
        **depth, "enorm": depth["hnorm"], "hnorm": depth["enorm"]}}}
    other = np.asarray(model.apply(swapped, hidden, ids, method=CausalTransformer.mtp_logits)[0])
    assert float(np.max(np.abs(other - reference))) > 1e-2


def test_a_glm4_moe_depth_with_its_own_head_is_refused(tmp_path):
    """The depth shares the trunk's embedding and head; a checkpoint whose
    copies differ would load as a model computing something else."""
    from safetensors.numpy import load_file, save_file

    tensors = load_file(str(GLM4_MOE / "model.safetensors"))
    tensors["model.layers.2.shared_head.head.weight"] = tensors["lm_head.weight"] * 1.5
    directory = tmp_path / "glm"
    directory.mkdir()
    save_file(tensors, str(directory / "model.safetensors"))
    (directory / "config.json").write_text(json.dumps(fixture_config("glm4-moe-tiny")))
    with pytest.raises(ValueError, match=r"shared_head.head.weight differs from lm_head.weight"):
        fp32_decoder(directory)



# --------------------------------------------------------------------------
# GLM-5 (glm_moe_dsa): V3.2's sparse MLA with an interleaved indexer,
# IndexShare layers and the MTP depth
# --------------------------------------------------------------------------

GLM_MOE_DSA = FIXTURES / "glm-moe-dsa-tiny"


def test_kimi_k25_text_only_placeholders_match_reference():
    import torch

    from tools.decoder_export_reference import Case, reference_model

    directory = FIXTURES / "kimi-k25-tiny"
    source = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    ids = np.load(directory / "input_ids.npy").copy()
    config = fixture_config("kimi-k25-tiny")
    ids[0, 0] = config["image_token_id"]
    ids[1, 2] = config["video_token_id"]
    reference, _ = reference_model(
        Case("kimi_k25", "kimi-k25-tiny", reference_class="Kimi_K25ForConditionalGeneration"),
        directory)
    reference.eval()
    reference.set_attn_implementation("eager")
    with torch.no_grad():
        expected = reference(input_ids=torch.from_numpy(ids.astype(np.int64)), use_cache=False).logits.numpy()
    actual = np.asarray(source.model.apply(source.variables, jnp.asarray(ids)))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    np.testing.assert_array_equal(actual.argmax(-1), expected.argmax(-1))


def test_kimi_k25_nested_quantization_is_refused_before_loading_shards(tmp_path):
    config = fixture_config("kimi-k25-tiny")
    config["text_config"]["quantization_config"] = {"quant_method": "compressed-tensors"}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=r"text_config\.quantization_config"):
        Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")


def test_glm_moe_dsa_config_translates_field_by_field():
    """The tiny GLM-5.3: every layer sparse MLA whose indexer rotates
    interleaved pairs and norms its latents at the hardcoded 1e-6, the
    IndexShare schedule as sharing layers, three dense layers ahead of the
    routed ones, and the released head_dim (qk_nope_head_dim) read as
    the stale field it is."""
    config = translate_config(fixture_config("glm-moe-dsa-tiny"))

    assert config["layer_types"] == ("deepseek_sparse_attention",) * 8
    assert config["mixer"] == {
        "class": "mla", "fields": {"q_lora_rank": 12, "kv_lora_rank": 8, "qk_nope_head_dim": 12,
        "qk_rope_head_dim": 4, "v_head_dim": 16, "rope_interleave": True, "yarn": None,
        "index_topk": 4, "index_n_heads": 4, "index_head_dim": 8,
        "index_rope_interleave": True}}
    assert config["kv_shared_layers"] == (3, 4, 5, 7)
    assert config["mixture"]["layers"] == (3, 4, 5, 6, 7)
    assert config["mixture"]["bias"] and config["mixture"]["shared_features"] == 12
    assert config["rope_theta"] == 8e6 and config["norm_eps"] == 1e-5
    assert config["head_dim"] == 16 and config["num_nextn_predict_layers"] == 1


@pytest.mark.parametrize("name, sharing", [("glm-5", None), ("glm-5.3", tuple(
    index for index in range(78) if index >= 3 and (index - 2) % 4 != 0))])
def test_the_released_glm_5_configs_translate(name, sharing):
    """zai-org/GLM-5 runs its indexer on all 78 layers; GLM-5.3 ships the
    IndexShare list its schedule (freq 4 off offset 3) produces, so layers
    6, 10, ... 74 select and the rest of the tail share. Both: 256 experts
    with 8 per token in one group scaled by 2.5, one shared expert, MLA
    with the 2048 query LoRA, plain rope, one prediction depth."""
    config = translate_config(fixture_config(name))

    assert config["num_layers"] == 78 and config["mixture"]["layers"] == tuple(range(3, 78))
    assert (config["mixture"]["experts"], config["mixture"]["top_k"]) == (256, 8)
    assert config["mixture"]["scaling"] == 2.5 and config["mixture"]["shared_features"] == 2048
    mixer = config["mixer"]["fields"]
    assert (mixer["q_lora_rank"], mixer["kv_lora_rank"]) == (2048, 512)
    assert (mixer["qk_nope_head_dim"], mixer["qk_rope_head_dim"], mixer["v_head_dim"]) == (192, 64, 256)
    assert (mixer["index_topk"], mixer["index_n_heads"], mixer["index_head_dim"]) == (2048, 32, 128)
    assert mixer["index_rope_interleave"] and mixer["yarn"] is None
    assert config.get("kv_shared_layers") == sharing
    assert config["num_nextn_predict_layers"] == 1 and config["vocab_size"] == 154880


def test_the_glm_5_3_schedule_derives_from_its_freq_and_offset():
    """Dropping the list leaves the schedule GlmMoeDsaConfig.__post_init__
    derives (configuration_glm_moe_dsa.py:144-148), which is the list."""
    config = fixture_config("glm-5.3")
    listed = translate_config(config)
    del config["indexer_types"]
    assert translate_config(config)["kv_shared_layers"] == listed["kv_shared_layers"]
    config["index_topk_pattern"] = "F" * 78
    assert "kv_shared_layers" not in translate_config(config)


@pytest.mark.parametrize("field, value, message", [
    ("indexer_rope_interleave", False, "always rotates interleaved"),
    ("indexer_types", ["shared"] + ["full"] * 7, "starting with 'shared'"),
    ("indexer_types", ["full"] * 7, "8 layers"),
    ("indexer_types", ["full"] * 7 + ["skip"], "entries \\['skip'\\]"),
    ("norm_topk_prob", False, "norm_topk_prob=False"),
    ("rope_parameters", {"rope_type": "linear", "factor": 4.0, "rope_theta": 8e6}, "plain or YaRN"),
    ("layer_types", ["full_attention"] * 8, "every layer is deepseek_sparse_attention"),
])
def test_a_glm_moe_dsa_field_with_no_counterpart_is_refused(field, value, message):
    with pytest.raises(ValueError, match=message):
        translate_config({**fixture_config("glm-moe-dsa-tiny"), field: value})


def test_glm_moe_dsa_logits_match_the_reference_implementation():
    """fp32 parity of the trunk: tolerance 1e-4, observed max |logit
    difference| 2.7e-05 with identical argmax, over four selecting layers
    and four that share the previous full layer's top-k."""
    model, variables = fp32_decoder(GLM_MOE_DSA)
    ids = np.load(GLM_MOE_DSA / "input_ids.npy")
    reference = np.load(GLM_MOE_DSA / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


def test_the_glm_moe_dsa_sharing_layers_carry_no_indexer_and_read_the_last_full_one():
    """The fixture's sharing layers own no indexer leaves, and the plan
    reads the last earlier selecting layer: layer 7 attends layer 6's
    top-k, not layer 2's. Rewiring the tail onto layer 2's selection, or
    letting the sharing layers select with layer 6's indexer weights,
    both disagree with the reference by more than the tolerance."""
    model, variables = fp32_decoder(GLM_MOE_DSA)
    assert model.kv_sharing == {3: 2, 4: 2, 5: 2, 7: 6}
    params = variables["params"]
    for layer in (3, 4, 5, 7):
        assert "indexer" not in params[f"layers_{layer}"]["self_attn"]
    for layer in (0, 1, 2, 6):
        assert params[f"layers_{layer}"]["self_attn"]["indexer"]["wq_b"]["kernel"].shape == (12, 32)
    assert "indexer" in params["mtp_0"]["block"]["self_attn"]

    ids = jnp.asarray(np.load(GLM_MOE_DSA / "input_ids.npy"), jnp.int32)
    reference = np.load(GLM_MOE_DSA / "logits.npy")
    tail_on_layer_2 = model.clone(kv_shared_layers=(3, 4, 5, 6, 7))
    rewired = {**variables, "params": {**params, "layers_6": {
        **params["layers_6"], "self_attn": {
            key: value for key, value in params["layers_6"]["self_attn"].items()
            if key != "indexer"}}}}
    difference = float(np.max(np.abs(
        np.asarray(tail_on_layer_2.apply(rewired, ids)) - reference)))
    assert difference > 1e-2, f"the tail on layer 2's selection differs by only {difference:.3e}"

    unshared = model.clone(kv_shared_layers=None)
    own = {**variables, "params": {**params, **{
        f"layers_{layer}": {**params[f"layers_{layer}"], "self_attn": {
            **params[f"layers_{layer}"]["self_attn"],
            "indexer": params["layers_6"]["self_attn"]["indexer"]}}
        for layer in (3, 4, 5, 7)}}}
    difference = float(np.max(np.abs(np.asarray(unshared.apply(own, ids)) - reference)))
    assert difference > 1e-2, f"sharing layers selecting on their own differ by only {difference:.3e}"


def test_the_glm_moe_dsa_mtp_depth_matches_the_reference_composition():
    """The checkpoint's model.layers.8.* depth loads as mtp_0 with its own
    indexer and the trunk's routing, and computes what the engines running
    the released weights compute (tools/hf_reference_b.py GlmMTP over a
    GlmMoeDsaDecoderLayer): tolerance 1e-4, observed max |logit difference|
    6.7e-06 with identical argmax."""
    from dew.nn.backbones.causal_transformer import CausalTransformer

    model, variables = fp32_decoder(GLM_MOE_DSA)
    ids = jnp.asarray(np.load(GLM_MOE_DSA / "input_ids.npy"), jnp.int32)
    reference = np.load(GLM_MOE_DSA / "mtp_logits.npy")
    depth = variables["params"]["mtp_0"]
    assert set(depth) == {"enorm", "hnorm", "eh_proj", "block", "final_norm"}
    assert depth["block"]["mlp"]["experts"]["gate_proj"]["kernel"].shape == (8, 32, 12)

    hidden = model.apply(variables, ids, method=CausalTransformer.hidden_states)
    logits = np.asarray(model.apply(variables, hidden, ids, method=CausalTransformer.mtp_logits)[0])
    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |mtp logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


def test_glm_moe_dsa_decodes_what_it_computes_in_one_pass():
    """Cached decode over the sparse layers, the sharing ones included:
    the prompt then one token reaches the one-pass logits of that token,
    with the sharing layers' cache holding no indexer keys."""
    from dew.nn.backbones.causal_transformer import CausalTransformer

    model, variables = fp32_decoder(GLM_MOE_DSA)
    ids = jnp.asarray(np.load(GLM_MOE_DSA / "input_ids.npy"), jnp.int32)
    full = np.asarray(model.apply(variables, ids))
    cache = model.apply(variables, 2, method=CausalTransformer.init_cache, mutable=["cache"])[1]["cache"]
    state = {**variables, "cache": cache}
    _, updated = model.apply(state, ids[:, :-1], decode=True, mutable=["cache"])
    step, _ = model.apply({**state, "cache": updated["cache"]}, ids[:, -1:], decode=True,
                          mutable=["cache"])
    assert "cached_index" in updated["cache"]["layers_6"]["self_attn"]
    assert "cached_index" not in updated["cache"]["layers_7"]["self_attn"]
    difference = float(np.max(np.abs(np.asarray(step)[:, 0] - full[:, -1])))
    assert difference < 1e-4, f"decode differs from the one-pass logits by {difference:.3e}"


# --------------------------------------------------------------------------
# DeepSeek V4: mHC's residual streams over the sliding, compressed sparse
# and heavily compressed kinds, with hash-routed first layers
# --------------------------------------------------------------------------

DEEPSEEK_V4 = FIXTURES / "deepseek-v4-tiny"
V4_TYPES = ("heavily_compressed_attention", "heavily_compressed_attention",
            "compressed_sparse_attention", "heavily_compressed_attention",
            "sliding_attention", "compressed_sparse_attention")
V4_YARN = {"rope_type": "yarn", "rope_theta": 160000.0, "factor": 16.0,
           "original_max_position_embeddings": 65536, "beta_fast": 32.0,
           "beta_slow": 1.0, "mscale": None, "mscale_all_dim": None,
           "truncate": True, "attention_factor": 1.0}
V4_MIXER = {"class": "deepseek_v4", "fields": {"q_lora_rank": 8, "o_groups": 2, "o_lora_rank": 8,
            "rope_head_dim": 4, "compressor": None, "compress_rate": None,
            "index_topk": None, "index_n_heads": None, "index_head_dim": None}}


def test_deepseek_v4_config_translates_field_by_field():
    """The tiny V4: the legacy `compress_ratios` read as the attention
    schedule, one key/value head, the rotated quarter head derived from
    `qk_rope_head_dim`, every kind naming the window and the compressed
    ones the compress rope under its YaRN ramp with the reference's forced
    attention_factor, the mixer carrying the query LoRA and the grouped
    output projection, every layer routed with the first three by the hash
    table, the gate clamp and mHC's stack."""
    config = translate_config(fixture_config("deepseek-v4-tiny"))

    assert config["layer_types"] == V4_TYPES
    assert config["num_kv_heads"] == 1 and config["head_dim"] == 16
    assert config["num_heads"] == 4 and config["emb_features"] == 32
    assert config["mlp"] == "swiglu" and config["mlp_features"] == 16
    assert config["qk_norm"] is False and config["attention_bias"] is False
    assert config["rope_theta"] == 10000.0 and config.get("yarn") is None
    assert config["mixer"] == V4_MIXER
    assert config["kinds"] == {
        "heavily_compressed_attention": {
            "window": 4, "rope_theta": 160000.0, "yarn": V4_YARN,
            "mixer": {"class": "deepseek_v4", "fields": {**V4_MIXER["fields"], "compressor": "hca",
                                                        "compress_rate": 4}}},
        "compressed_sparse_attention": {
            "window": 4, "rope_theta": 160000.0, "yarn": V4_YARN,
            "mixer": {"class": "deepseek_v4", "fields": {**V4_MIXER["fields"], "compressor": "csa",
                                                        "compress_rate": 2, "index_topk": 2,
                                                        "index_n_heads": 2, "index_head_dim": 8}}},
        "sliding_attention": {"window": 4},
        "mtp_attention": {"window": 4, "rope_theta": 10000.0, "mixer": V4_MIXER}}
    assert config["mixture"] == {
        "experts": 8, "top_k": 2, "layers": (0, 1, 2, 3, 4, 5),
        "score_function": "sqrtsoftplus", "bias": True, "scaling": 1.5,
        "shared_features": 16, "expert_features": 16, "hash_layers": (0, 1, 2)}
    assert config["swiglu_limit"] == 2.0
    assert config["hyper_connections"] == {"hc_mult": 2, "hc_eps": 1e-6,
                                           "hc_sinkhorn_iters": 20, "head": "weighted"}
    assert config['mtp_hyper_connections'] == config['hyper_connections']
    assert config["norm_eps"] == 1e-6 and config["tie_embeddings"] is False
    assert config["max_seq_len"] == 64 and config["vocab_size"] == 256
    assert config["scale_after_cast"] and config["num_nextn_predict_layers"] == 1
    assert config["mtp_layer_type"] == 'mtp_attention'


def test_the_released_deepseek_v4_flash_config_translates():
    """deepseek-ai/DeepSeek-V4-Flash: 43 layers whose `compress_ratios`
    name two sliding layers and then the CSA/HCA interleave, with the list
    carrying one entry more than the stack for its prediction depth; 256
    experts, 6 per token, one shared expert of the routed width, sinks over
    64 heads of 512 with the rope on 64 of them, the 128-token window and
    four residual streams."""
    config = translate_config(fixture_config("deepseek-v4-flash"))

    assert config["num_layers"] == 43 and len(config["layer_types"]) == 43
    assert config["layer_types"][:4] == (
        "sliding_attention", "sliding_attention",
        "compressed_sparse_attention", "heavily_compressed_attention")
    assert config["layer_types"][-1] == "compressed_sparse_attention"
    assert (config["num_heads"], config["head_dim"], config["num_kv_heads"]) == (64, 512, 1)
    mixer = config["mixer"]["fields"]
    assert (mixer["q_lora_rank"], mixer["o_groups"], mixer["o_lora_rank"]) == (1024, 8, 1024)
    assert mixer["rope_head_dim"] == 64 and mixer["compressor"] is None
    sparse = config["kinds"]["compressed_sparse_attention"]["mixer"]["fields"]
    heavy = config["kinds"]["heavily_compressed_attention"]["mixer"]["fields"]
    assert (sparse["compress_rate"], heavy["compress_rate"]) == (4, 128)
    assert (sparse["index_topk"], sparse["index_n_heads"], sparse["index_head_dim"]) == (512, 64, 128)
    assert heavy["index_topk"] is None
    assert config["kinds"]["sliding_attention"] == {"window": 128}
    assert config["kinds"]["compressed_sparse_attention"]["yarn"] == V4_YARN
    assert config["mixture"]["experts"] == 256 and config["mixture"]["top_k"] == 6
    assert config["mixture"]["shared_features"] == 2048
    assert config["mixture"]["hash_layers"] == (0, 1, 2)
    assert config["mixture"]["layers"] == tuple(range(43))
    assert config["swiglu_limit"] == 10.0
    assert config["hyper_connections"]["hc_mult"] == 4
    assert config["vocab_size"] == 129280 and config["rope_theta"] == 10000.0


def test_the_deepseek_v4_legacy_fields_fold_into_the_modern_spelling():
    """The fixture is the released legacy spelling: `compress_ratios` per
    layer, the two `compress_rate_*` scalars, `num_hash_layers`,
    `qk_rope_head_dim` and one flat `rope_scaling`. Stating the modern
    fields instead (`layer_types`, `mlp_layer_types`, `compress_rates`,
    `partial_rotary_factor` and the nested `rope_parameters` transformers
    writes back) reaches the same record
    (configuration_deepseek_v4.py:239-321)."""
    legacy = fixture_config("deepseek-v4-tiny")
    modern = {key: value for key, value in legacy.items()
              if key not in ("compress_ratios", "compress_rate_csa", "compress_rate_hca",
                             "num_hash_layers", "qk_rope_head_dim", "rope_scaling")}
    modern.update(
        layer_types=list(V4_TYPES),
        mlp_layer_types=["hash_moe"] * 3 + ["moe"] * 3,
        compress_rates={"compressed_sparse_attention": 2,
                        "heavily_compressed_attention": 4},
        partial_rotary_factor=0.25,
        rope_parameters={
            "main": {"rope_type": "default", "rope_theta": 10000,
                     "partial_rotary_factor": 0.25},
            "compress": {"rope_type": "yarn", "rope_theta": 160000, "factor": 16,
                         "beta_fast": 32, "beta_slow": 1, "attention_factor": 1.0,
                         "original_max_position_embeddings": 65536,
                         "partial_rotary_factor": 0.25}})

    assert translate_config(modern) == translate_config(legacy)


@pytest.mark.parametrize("field, value, message", [
    ("num_key_value_heads", 4, "num_key_value_heads 4"),
    ("scoring_func", "relu", "scoring_func 'relu'"),
    ("norm_topk_prob", False, "norm_topk_prob=False"),
    ("n_shared_experts", 2, "n_shared_experts 2"),
    ("topk_method", "greedy", "topk_method 'greedy'"),
    ("layer_types", ["full_attention"] * 6, r"layer_types entries \['full_attention'\]"),
    ("layer_types", ["sliding_attention"] * 5, "layer_types of 5 entries"),
    ("mlp_layer_types", ["dense"] * 6, r"mlp_layer_types entries \['dense'\]"),
    ("mlp_layer_types", ["moe"] * 5, "mlp_layer_types of 5 entries"),
    ("compress_ratios", [7] * 6, r"compress_ratios entries \[7\]"),
    ("compress_rates", {"full_attention": 4}, r"compress_rates keys \['full_attention'\]"),
    ("compress_rate_hca", 0, "compress_rates\\['heavily_compressed_attention'\\] 0"),
    ("mlp_bias", True, "mlp_bias=True"),
    ("attention_bias", True, "attention_bias=True"),
    ("sliding_window", None, "sliding_window"),
    ("o_groups", 3, "o_groups 3"),

    ("partial_rotary_factor", 0.1875, "partial_rotary_factor 0.1875"),
    ("rope_scaling", {"type": "linear", "factor": 4.0}, "plain or YaRN"),
    ("rope_scaling", {"type": "yarn", "factor": 16,
                      "original_max_position_embeddings": 65536,
                      "attention_factor": 1.3}, "attention_factor 1.3"),
])
def test_a_deepseek_v4_field_with_no_counterpart_is_refused(field, value, message):
    with pytest.raises(ValueError, match=message):
        translate_config({**fixture_config("deepseek-v4-tiny"), field: value})


def test_a_deepseek_v4_yarn_ramp_that_scales_its_table_is_refused():
    """A nested compress entry naming no `attention_factor` derives one
    that scales cos and sin (modeling_deepseek_v4.py:151-152), which this
    rotary does not do, so it refuses naming the field."""
    config = {key: value for key, value in fixture_config("deepseek-v4-tiny").items()
              if key != "rope_scaling"}
    config["rope_parameters"] = {
        "main": {"rope_type": "default", "rope_theta": 10000,
                 "partial_rotary_factor": 0.25},
        "compress": {"rope_type": "yarn", "rope_theta": 160000, "factor": 16,
                     "original_max_position_embeddings": 65536,
                     "partial_rotary_factor": 0.25}}
    with pytest.raises(ValueError, match="attention_factor None"):
        translate_config(config)


@pytest.mark.parametrize("released, saved", [
    ("embed.weight", "model.embed_tokens.weight"),
    ("norm.weight", "model.norm.weight"),
    ("hc_head_fn", "model.hc_head.hc_fn"),
    ("hc_head_base", "model.hc_head.hc_base"),
    ("hc_head_scale", "model.hc_head.hc_scale"),
    ("layers.0.attn.kv_norm.weight", "model.layers.0.attn.norm.weight"),
    ("layers.0.attn.wq_a.weight", "model.layers.0.attn.wq_a.weight"),
    ("layers.0.hc_attn_fn", "model.layers.0.hc_attn_fn"),
    ("layers.2.attn.indexer.wq_b.weight", "model.layers.2.attn.indexer.wq_b.weight"),
    ("layers.0.ffn.experts.3.w2.weight", "model.layers.0.ffn.experts.3.w2.weight"),
])
def test_a_deepseek_v4_released_tensor_name_reaches_the_same_leaf(released, saved):
    """A released checkpoint holds the stack at `layers.N.*` with no
    `model.` prefix, a flat `hc_head_*` and the latent norm as `kv_norm`,
    while a checkpoint transformers wrote back carries the prefix, the
    nested head and `norm` (conversion_mapping.py:489-508 is ^-anchored on
    the first three, so the reverse leaves them). Both spellings are the
    same weights and land on the same leaf."""
    from dew.interop.families.deepseek import _deepseek_v4_path

    config = translate_config(fixture_config("deepseek-v4-tiny"))
    path = _deepseek_v4_path(released, config)
    assert path is not None and path == _deepseek_v4_path(saved, config)


def test_both_spellings_of_one_deepseek_v4_tensor_are_refused():
    """The two spellings land on one leaf, so a checkpoint carrying both
    with different values would load whichever came last."""
    from dew.interop.sources import load_shards

    config = translate_config(fixture_config("deepseek-v4-tiny"))
    tensors = dict(load_shards(DEEPSEEK_V4))
    tensors["model.embed_tokens.weight"] = tensors["embed.weight"] + 1

    with pytest.raises(ValueError, match=r"model.embed_tokens.weight lands on params/embed_tokens"):
        translate_weights(tensors, config)


def test_the_deepseek_v4_tree_is_exactly_the_models_variables(rng):
    """Same collections, paths and shapes as a freshly initialised model,
    with the `moe` collection holding the hash layers' int32 table and the
    top-k layers' fp32 balancing bias."""
    from dew.interop.sources import load_shards

    config = translate_config(fixture_config("deepseek-v4-tiny"))
    built = with_precision("causal_transformer", dict(config),
                           dtype="float32", attention_impl="reference")
    model = models.build("causal_transformer", **built)
    initialised = flat_tree(model.init(rng, jnp.zeros((1, 4), jnp.int32)))
    loaded = flat_tree(translate_weights(load_shards(DEEPSEEK_V4), config))

    assert set(loaded) == set(initialised)
    for path, leaf in loaded.items():
        assert leaf.shape == initialised[path].shape, path
        assert leaf.dtype == (jnp.int32 if path.endswith("tid2eid") else jnp.float32), path
    assert loaded["params.layers_0.self_attn.o_a_proj.kernel"].shape == (2, 32, 8)
    assert sorted(name for name in loaded if name.startswith("moe.")) == [
        f"moe.layers_{index}.mlp.gate."
        + ("tid2eid" if index < 3 else "e_score_correction_bias")
        for index in range(6)] + ['moe.mtp_0.block.mlp.gate.e_score_correction_bias']


def test_deepseek_v4_logits_match_the_reference_implementation():
    """fp32 parity over all three attention kinds and both routers:
    tolerance 1e-4 on logits of magnitude 4.4."""
    model, variables = fp32_decoder(DEEPSEEK_V4)
    ids = np.load(DEEPSEEK_V4 / "input_ids.npy")
    reference = np.load(DEEPSEEK_V4 / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))


def test_the_deepseek_v4_hash_layers_route_by_their_token_table():
    """The first three layers select their experts at `tid2eid[input_ids]`
    rather than on the scores (modeling_deepseek_v4.py:1062-1073): rolling
    the table along the vocabulary sends every token to another token's
    experts, and the logits move far past the parity tolerance."""
    model, variables = fp32_decoder(DEEPSEEK_V4)
    ids = jnp.asarray(np.load(DEEPSEEK_V4 / "input_ids.npy"), jnp.int32)
    reference = np.load(DEEPSEEK_V4 / "logits.npy")
    moe = variables["moe"]
    assert [
        layer
        for layer in sorted(moe)
        if layer.startswith("layers_") and "tid2eid" in moe[layer]["mlp"]["gate"]
    ] == ["layers_0", "layers_1", "layers_2"]

    rolled = jax.tree.map(lambda value: value, moe)
    for layer in ("layers_0", "layers_1", "layers_2"):
        table = np.asarray(moe[layer]["mlp"]["gate"]["tid2eid"])
        rolled[layer]["mlp"]["gate"]["tid2eid"] = np.roll(table, 1, axis=0)
    difference = float(np.max(np.abs(
        np.asarray(model.apply({**variables, "moe": rolled}, ids)) - reference)))
    assert difference > 1e-2, f"rolling the table moved the logits {difference:.3e}"


@pytest.mark.parametrize('saved_spelling', [False, True], ids=['released', 'hf_saved'])
def test_deepseek_v4_tied_source_names_roundtrip(tmp_path, saved_spelling):
    from dew.interop.safetensors_io import save_hf_layout
    from dew.interop.sources import load_shards

    tensors = load_shards(DEEPSEEK_V4)
    tensors['head.weight'] = tensors['embed.weight'].copy()
    if saved_spelling:
        # HF's reverse converter prefixes the trunk and misses the anchored
        # embedding/head renames. Keep the released prediction payload too.
        renamed = {}
        for name, tensor in tensors.items():
            if name == 'embed.weight':
                name = 'model.embed_tokens.weight'
            elif name == 'norm.weight':
                name = 'model.norm.weight'
            elif name.startswith('hc_head_'):
                name = 'model.hc_head.hc_' + name.removeprefix('hc_head_')
            elif name.startswith('layers.'):
                name = 'model.' + name.replace('.attn.kv_norm.', '.attn.norm.')
            renamed[name] = tensor
        tensors = renamed
    config = fixture_config('deepseek-v4-tiny')
    config['tie_word_embeddings'] = True
    directory = tmp_path / 'source'
    save_hf_layout(tensors, config, directory)
    source = Pretrained.load(directory, dtype='float32', attention_impl='reference')
    export = tmp_path / 'export'
    source.save(export)
    emitted = load_shards(export)
    assert set(emitted) == set(tensors)
    for name, tensor in tensors.items():
        np.testing.assert_array_equal(emitted[name], tensor, err_msg=name)
    reloaded = Pretrained.load(export, dtype='float32', attention_impl='reference')
    ids = np.load(DEEPSEEK_V4 / 'input_ids.npy')
    np.testing.assert_array_equal(source.model.apply(source.variables, ids),
                                  reloaded.model.apply(reloaded.variables, ids))
    save_hf_layout({**tensors, 'head.weight': tensors['head.weight'] + 1}, config, directory)
    with pytest.raises(ValueError, match=r'head\.weight is not the embedding'):
        Pretrained.load(directory, dtype='float32', attention_impl='reference')


@pytest.mark.parametrize('invalid', [2**32, 8])
def test_deepseek_v4_hash_table_refuses_invalid_expert_indices(tmp_path, invalid):
    from dew.interop.safetensors_io import save_hf_layout
    from dew.interop.sources import load_shards

    tensors = load_shards(DEEPSEEK_V4)
    name = 'layers.0.ffn.gate.tid2eid'
    tensors[name] = tensors[name].copy()
    tensors[name][0, 0] = invalid
    save_hf_layout(tensors, fixture_config('deepseek-v4-tiny'), tmp_path)
    with pytest.raises(ValueError, match=r'tid2eid.*(int32 range|outside its 8 experts)'):
        Pretrained.load(tmp_path, dtype='float32', attention_impl='reference')


def test_deepseek_v4_public_speculation_preserves_padded_rows():
    from dew.nn.inputs import ModelInputs
    from dew.sampling import Sampling, Speculative

    source = Pretrained.load(DEEPSEEK_V4, dtype='float32', attention_impl='reference')
    tokens = np.load(DEEPSEEK_V4 / 'input_ids.npy')[[0, 1, 0], :5].copy()
    valid = np.arange(5)[None, :] >= np.asarray([0, 2, 4])[:, None]
    tokens[~valid] = 0
    inputs = ModelInputs(jnp.asarray(tokens), {'attention_mask': jnp.asarray(valid)})
    task = source.text_generation(sampling=Sampling(temperature=0))
    ordinary = task(inputs, max_new_tokens=7, key=0).host()
    speculative = task(inputs, max_new_tokens=7, key=0, strategy=Speculative(block=3)).host()
    # Three rows differ from the two residual streams: a row mask must not
    # accidentally broadcast along the stream axis during predictor reseeding.
    np.testing.assert_array_equal(speculative.tokens, ordinary.tokens)
    np.testing.assert_array_equal(speculative.lengths, ordinary.lengths)
    np.testing.assert_array_equal(speculative.terminated, ordinary.terminated)


def test_deepseek_v4_cached_chunks_match_transformers():
    import torch

    from tools.decoder_export_reference import Case, reference_model

    source = Pretrained.load(DEEPSEEK_V4, dtype='float32', attention_impl='reference')
    ids = np.load(DEEPSEEK_V4 / 'input_ids.npy')
    cache = source.model.apply(source.variables, ids.shape[0], method=source.model.init_cache,
                               mutable=['cache'])[1]['cache']
    reference, _ = reference_model(Case('deepseek_v4', 'deepseek-v4-tiny'), DEEPSEEK_V4)
    reference.eval()
    reference.set_attn_implementation('eager')
    past, begin = None, 0
    # End calls both inside and on compressor boundaries; the third call
    # closes several windows and must retain the preceding CSA Ca series.
    for size in (3, 1, 5, 3):
        tokens = ids[:, begin:begin + size]
        actual, updated = source.model.apply(
            {**source.variables, 'cache': cache}, jnp.asarray(tokens),
            decode=True, mutable=['cache'])
        cache = updated['cache']
        with torch.no_grad():
            output = reference(input_ids=torch.from_numpy(tokens.astype(np.int64)),
                               past_key_values=past, use_cache=True)
        past = output.past_key_values
        np.testing.assert_allclose(actual, output.logits.numpy(), atol=1e-4, rtol=0)
        begin += size


def test_deepseek_v4_public_prediction_states_preserve_raw_streams():
    source = Pretrained.load(DEEPSEEK_V4, dtype='float32', attention_impl='reference')
    ids = np.load(DEEPSEEK_V4 / 'input_ids.npy')
    model, variables = source.model, source.variables
    states, logits = model.apply(variables, ids, method=model.states_and_logits)
    states = jnp.asarray(states)
    assert states.shape == (*ids.shape, 2, model.emb_features)
    normalized, expected = model.apply(variables, ids, method=model.hidden_and_mtp_inputs)
    np.testing.assert_array_equal(states, expected)
    assert jnp.asarray(normalized).shape == (*ids.shape, model.emb_features)
    doubled = {**variables, 'params': {**variables['params'], 'norm': {
        **variables['params']['norm'], 'scale': variables['params']['norm']['scale'] * 2}}}
    raw, louder = model.apply(doubled, ids, method=model.states_and_logits)
    np.testing.assert_array_equal(raw, states)
    np.testing.assert_allclose(jnp.asarray(louder), 2 * jnp.asarray(logits), atol=1e-4, rtol=0)


def test_deepseek_v4_mtp_matches_released_raw_stream_composition():
    source = Pretrained.load(DEEPSEEK_V4, dtype='float32', attention_impl='reference')
    ids = np.load(DEEPSEEK_V4 / 'input_ids.npy')
    hidden, streams = source.model.apply(source.variables, ids, method=source.model.hidden_and_mtp_inputs)
    streams = jnp.asarray(streams)
    actual = source.model.apply(source.variables, streams, ids, method=source.model.mtp_logits)[0]
    expected = np.load(DEEPSEEK_V4 / 'mtp_logits.npy')
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=0)
    cache = source.model.apply(source.variables, ids.shape[0], method=source.model.init_mtp_cache,
                               mutable=['cache'])[1]['cache']
    start = 0
    for size in (3, 1, 5, 2):
        (logits, prediction), updated = source.model.apply(
            {**source.variables, 'cache': cache}, streams[:, start:start + size],
            ids[:, start + 1:start + size + 1], method=source.model.mtp_step,
            decode=True, mutable=['cache'])
        np.testing.assert_allclose(logits, expected[:, start:start + size], atol=1e-4, rtol=0)
        assert prediction.shape == streams[:, start:start + size].shape
        cache = updated['cache']
        start += size
    with pytest.raises(ValueError, match='uncollapsed residual streams'):
        source.model.apply(source.variables, hidden, ids, method=source.model.mtp_logits)


def test_the_deepseek_v4_swiglu_limit_clamps_the_gated_mlps():
    """`swiglu_limit` caps the gate from above and the up projection on
    both sides before the activation, in the shared MLP as in the routed
    experts (modeling_deepseek_v4.py:978-979, :1014-1022). The fixture's
    pre-activations reach its 2.0, so the same weights unclamped disagree
    with the reference by more than 1."""
    model, variables = fp32_decoder(DEEPSEEK_V4)
    ids = jnp.asarray(np.load(DEEPSEEK_V4 / "input_ids.npy"), jnp.int32)
    reference = np.load(DEEPSEEK_V4 / "logits.npy")
    assert model.swiglu_limit == 2.0

    unclamped = np.asarray(model.clone(swiglu_limit=None).apply(variables, ids))

    difference = float(np.max(np.abs(unclamped - reference)))
    assert difference > 1.0, f"dropping the clamp moved the logits {difference:.3e}"


# --------------------------------------------------------------------------
# Llama 4 (text): iRoPE with chunked local layers, input-scaled experts
# --------------------------------------------------------------------------

LLAMA4 = FIXTURES / "llama4-tiny"


def test_llama4_config_translates_field_by_field():
    """The tiny text config: three chunked rotated layers and one global
    layer, each kind naming the llama4 mixer under its own rule, the dense
    layers at intermediate_size_mlp and the routed layers 1 and 3 with a
    shared expert at intermediate_size, sigmoid weights on the inputs."""
    config = translate_config(fixture_config("llama4-tiny"))

    assert config["layer_types"] == ("chunked_attention",) * 3 + ("full_attention",)
    rule = {"use_qk_norm": True, "attn_temperature_tuning": True, "floor_scale": 4.0, "attn_scale": 0.1}
    assert config["kinds"] == {
        "full_attention": {"mixer": {"class": "llama4", "fields": {**rule, "use_rope": False}}},
        "chunked_attention": {"chunk": 4, "mixer": {"class": "llama4", "fields": {**rule, "use_rope": True}}}}
    assert config["mixture"] == {
        "experts": 4, "top_k": 2, "score_function": "sigmoid", "norm_topk_prob": False,
        "scale_inputs": True, "expert_features": 48, "shared_features": 48, "layers": (1, 3)}
    assert config["mlp_features"] == 64 and not config["qk_norm"]
    assert config["rope_theta"] == 500000.0 and config["scale_after_cast"]


def test_the_real_llama_4_scout_text_config_translates():
    """meta-llama/Llama-4-Scout-17B-16E's text_config, from a mirror: 48
    layers with every fourth global, 40 query and 8 key/value heads of 128,
    16 experts with one per token on every layer beside a shared expert of
    8192, dense layers of 16384, chunks of 8192 with temperature tuning at
    floor_scale 8192, and the llama3 ramp at theta 500000 on the rotated
    layers."""
    config = translate_config(fixture_config("llama-4-scout")["text_config"])

    assert config["num_layers"] == 48
    assert config["layer_types"][:4] == ("chunked_attention",) * 3 + ("full_attention",)
    assert config["layer_types"].count("full_attention") == 12
    assert (config["num_heads"], config["num_kv_heads"], config["head_dim"]) == (40, 8, 128)
    assert config["mixture"] == {
        "experts": 16, "top_k": 1, "score_function": "sigmoid", "norm_topk_prob": False,
        "scale_inputs": True, "expert_features": 8192, "shared_features": 8192,
        "layers": tuple(range(48))}
    assert config["mlp_features"] == 16384 and config["vocab_size"] == 202048
    local = config["kinds"]["chunked_attention"]
    assert local["chunk"] == 8192 and local["mixer"]["fields"]["floor_scale"] == 8192.0
    assert config["rope_theta"] == 500000.0
    assert config["rope_scaling"] == {
        "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
        "high_freq_factor": 4.0, "original_max_position_embeddings": 8192}


def test_a_llama4_wrapper_config_is_refused_by_name():
    """meta-llama/Llama-4-Scout-17B-16E's config.json wraps the text decoder
    beside a vision tower; the text_config alone is what translates."""
    with pytest.raises(ValueError, match="model_type 'llama4'"):
        translate_config(fixture_config("llama-4-scout"))


@pytest.mark.parametrize("field, value, message", [
    ("layer_types", ["full_attention"] * 4, "disagrees with no_rope_layers"),
    ("router_jitter_noise", 0.1, "router_jitter_noise"),
    ("output_router_logits", True, "output_router_logits"),
    ("no_rope_layers", [1, 1, 2, 0], "no_rope_layers"),
])
def test_a_llama4_field_with_no_counterpart_is_refused(field, value, message):
    with pytest.raises(ValueError, match=message):
        translate_config({**fixture_config("llama4-tiny"), field: value})


def test_llama4_logits_match_the_reference_implementation():
    """fp32 parity: tolerance 1e-4, observed max |logit difference| 3.5e-06
    with identical argmax. The fused expert kernels arrive split in place."""
    model, variables = fp32_decoder(LLAMA4)
    ids = np.load(LLAMA4 / "input_ids.npy")
    reference = np.load(LLAMA4 / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))
    experts = variables["params"]["layers_1"]["mlp"]["experts"]
    assert experts["gate_proj"]["kernel"].shape == (4, 32, 48)
    assert experts["down_proj"]["kernel"].shape == (4, 48, 32)
    assert "shared_experts" in variables["params"]["layers_1"]["mlp"]
    assert "experts" not in variables["params"]["layers_0"]["mlp"]


def test_llama4_export_is_refused_by_name(tmp_path):
    model, variables = fp32_decoder(LLAMA4)
    with pytest.raises(ValueError, match="lacks 'num_local_experts'"):
        PretrainedDecoder.from_model(model, variables).save(str(tmp_path))


def test_a_chunked_kind_no_config_carries_is_refused_on_export(tmp_path):
    """Only Llama 4's config spells a chunk; a llama-family export of a
    chunked attention kind would write a model that attends everything."""
    model = CausalTransformer(
        vocab_size=32, num_layers=2, emb_features=16, num_heads=2, max_seq_len=16,
        layer_types=("chunked_attention", "full_attention"),
        kinds={"chunked_attention": {"chunk": 4}})
    variables = model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))
    with pytest.raises(ValueError, match="chunk"):
        PretrainedDecoder.from_model(model, variables).save(str(tmp_path))



# ---------------------------------------------------------------------------
# Gemma 4's routed size: the parallel experts, keys as values, layer scalars
# ---------------------------------------------------------------------------



def test_gemma4_moe_config_translates_field_by_field():
    """The 26B-A4B shape at toy width: enable_moe_block names the parallel
    mixture of four experts with two per token at width 16, attention_k_eq_v
    lands as a field with the global kind's own head dim and single
    key/value head, and every layer carries its output scalar."""
    config = translate_config(fixture_config("gemma4-moe-tiny"))

    assert config["mixture"] == {"experts": 4, "top_k": 2, "expert_features": 16,
                                 "parallel": True}
    assert config["attention_k_eq_v"] and config["layer_scalar"]
    assert config["num_kv_heads"] == 2 and config["head_dim"] == 8
    assert config["kinds"]["full_attention"] == {"head_dim": 16, "num_kv_heads": 1}
    assert config["kinds"]["sliding_attention"] == {"window": 4, "rope_theta": 10000.0}
    assert config["partial_rotary_factor"] == 0.25
    assert config["mlp"] == "geglu" and config["mlp_features"] == 48


def test_the_real_gemma_4_26b_a4b_text_config_translates():
    """google/gemma-4-26B-A4B's text_config, from a mirror: 30 layers in the
    5:1 pattern, 16 query heads of 256 with 8 key/value heads on the sliding
    layers and 2 of 512 reading their values off the keys on the global ones,
    128 experts with 8 per token at width 704 beside a dense MLP of 2112, a
    quarter rotary at theta 1e6 over a local theta of 1e4, and softcap 30."""
    config = translate_config(fixture_config("gemma4-26b-a4b")["text_config"])

    assert config["num_layers"] == 30
    assert config["layer_types"].count("sliding_attention") == 25
    assert config["layer_types"][-1] == "full_attention"
    assert config["emb_features"] == 2816 and config["vocab_size"] == 262144
    assert config["num_heads"] == 16 and config["num_kv_heads"] == 8
    assert config["head_dim"] == 256
    assert config["kinds"]["full_attention"] == {"head_dim": 512, "num_kv_heads": 2}
    assert config["kinds"]["sliding_attention"] == {"window": 1024, "rope_theta": 10000.0}
    assert config["attention_k_eq_v"] and config["layer_scalar"]
    assert config["mixture"] == {"experts": 128, "top_k": 8, "expert_features": 704,
                                 "parallel": True}
    assert config["mlp_features"] == 2112 and config["mlp"] == "geglu"
    assert config["partial_rotary_factor"] == 0.25
    assert config["rope_theta"] == 1000000.0
    assert config["final_logit_softcap"] == 30.0
    assert config["v_norm"] and not config["use_double_wide_mlp"]
    assert config["per_layer_input_dim"] is None and config["kv_shared_layers"] is None


def test_the_released_diffusiongemma_26b_text_config_derives_what_it_does_not_name():
    """google/diffusiongemma-26B-A4B-it's text_config is the only committed
    DiffusionGemma config whose global geometry differs from its sliding one:
    every tiny fixture sets global_head_dim and num_global_key_value_heads to
    the sliding values, so none of them can tell the two apart.

    The three decisions this family makes for itself, none of them a field of
    the config: the reference builds v_proj only where a layer slides, so the
    record reads values off keys and its global layers take the global key
    count; it names no enable_moe_block, so the three routed widths route
    every layer beside the dense MLP; and the canvas decoder is bidirectional.
    """
    config = translate_config(fixture_config("diffusiongemma-26b")["text_config"])

    assert config["attention_k_eq_v"] and not config["causal"]
    assert config["kinds"]["full_attention"] == {"head_dim": 512, "num_kv_heads": 2}
    assert config["kinds"]["sliding_attention"] == {"window": 1024, "rope_theta": 10000.0}
    assert config["layer_types"] == (("sliding_attention",) * 5 + ("full_attention",)) * 5
    assert config["mixture"] == {"experts": 128, "top_k": 8, "expert_features": 704,
                                 "parallel": True}
    assert config["final_logit_softcap"] == 30.0
    assert config["partial_rotary_factor"] == 0.25 and config["rope_theta"] == 1000000.0


def test_the_released_diffusiongemma_26b_builds_its_split_global_geometry():
    """The released config as a model, shapes only, no weights and no
    download: the sliding layers project 16 heads of 256 and keep their own
    values, the global layers project 16 of 512 and read values off 2 key
    heads, and every layer routes 128 experts of 704 beside its dense 2112.

    These are the shapes a released checkpoint's tensors have to land in, so a
    translation that carried the model's 8 key/value heads onto the global
    layers, or lost the routed branch the config never flags, builds a tree
    the checkpoint cannot load.
    """
    config = translate_config(fixture_config("diffusiongemma-26b")["text_config"])
    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", config, dtype="bfloat16", attention_impl="reference"))
    params = jax.eval_shape(
        lambda: model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)))["params"]

    sliding, full = params["layers_0"], params["layers_5"]
    assert config["layer_types"][0] == "sliding_attention"
    assert config["layer_types"][5] == "full_attention"
    assert sliding["self_attn"]["q_proj"]["kernel"].shape == (2816, 16 * 256)
    assert sliding["self_attn"]["k_proj"]["kernel"].shape == (2816, 8 * 256)
    assert sliding["self_attn"]["v_proj"]["kernel"].shape == (2816, 8 * 256)
    assert full["self_attn"]["q_proj"]["kernel"].shape == (2816, 16 * 512)
    assert full["self_attn"]["k_proj"]["kernel"].shape == (2816, 2 * 512)
    assert "v_proj" not in full["self_attn"]
    experts = full["moe"]["experts"]
    assert experts["gate_proj"]["kernel"].shape == (128, 2816, 704)
    assert experts["down_proj"]["kernel"].shape == (128, 704, 2816)
    assert full["moe"]["router"]["proj"]["kernel"].shape == (2816, 128)
    assert full["mlp"]["gate_proj"]["kernel"].shape == (2816, 2112)
    assert params["embed_tokens"]["embedding"].shape == (262144, 2816)
    assert "lm_head" not in params


def gated_text_config(repo: str):
    """The text_config of a gated release, from the runner's own hub access.

    google gates the Gemma 4 repositories, so the config cannot be committed
    as a fixture the way a mirrored one is. These two shapes are covered
    portably nowhere else: E2B is the only Gemma 4 that shares key/value
    layers and carries per-layer inputs at released width, and the dense 31B
    is the only released Gemma 4 with no routed branch.
    """
    from huggingface_hub import hf_hub_download

    return json.loads(Path(hf_hub_download(repo, "config.json")).read_text())["text_config"]


def gemma4_release_is_available(repo: str) -> bool:
    if os.environ.get("DEW_NETWORK_TESTS") == "1":
        return True
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(repo, "config.json"), str)


@pytest.mark.network
@pytest.mark.skipif(not gemma4_release_is_available("google/gemma-4-E2B"),
                    reason="google/gemma-4-E2B is gated: neither cached nor DEW_NETWORK_TESTS=1")
def test_the_released_e2b_config_translates_and_shares_the_layers_it_names():
    """google/gemma-4-E2B: partial rotary 0.25, head dims 256 and 512, 20
    shared key/value layers, per-layer inputs of 256, the double-wide MLP,
    scale 1.0 and softcap 30. The shared layers then own no k_proj, v_proj or
    k_norm in the built tree, which is what makes the count load-bearing."""
    config = translate_config(gated_text_config("google/gemma-4-E2B"))

    assert config["partial_rotary_factor"] == 0.25
    assert config["head_dim"] == 256
    assert config["num_kv_heads"] == 1
    assert config["kinds"]["full_attention"] == {"head_dim": 512}
    assert config["kinds"]["sliding_attention"] == {"window": 512, "rope_theta": 10000.0}
    assert config["kv_shared_layers"] == tuple(range(config["num_layers"] - 20, config["num_layers"]))
    assert config["per_layer_input_dim"] == 256
    assert config["use_double_wide_mlp"]
    assert config["attention_scale"] == 1.0
    assert config["final_logit_softcap"] == 30.0
    assert config["rope_theta"] == 1000000.0

    model = models.build("causal_transformer", **with_precision(
        "causal_transformer", config, dtype="bfloat16", attention_impl="reference"))
    params = jax.eval_shape(
        lambda: model.init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)))["params"]
    shared = [index for index in range(config["num_layers"])
              if set(params[f"layers_{index}"]["self_attn"]) == {"q_proj", "o_proj", "q_norm"}]
    assert shared == sorted(model.kv_sharing)
    assert tuple(shared) == config["kv_shared_layers"]


@pytest.mark.network
@pytest.mark.skipif(not gemma4_release_is_available("google/gemma-4-31B"),
                    reason="google/gemma-4-31B is gated: neither cached nor DEW_NETWORK_TESTS=1")
def test_the_released_dense_31b_config_translates_without_a_routed_branch():
    """The dense 31B: 60 layers, 32 query heads of 256 with 16 key/value
    heads, the global layers reading their values off 4 key/value heads of
    512, no routed branch and no per-layer inputs."""
    config = translate_config(gated_text_config("google/gemma-4-31B"))

    assert config["num_layers"] == 60
    assert config["num_heads"] == 32 and config["num_kv_heads"] == 16
    assert config["head_dim"] == 256
    assert config["kinds"]["full_attention"] == {"head_dim": 512, "num_kv_heads": 4}
    assert config["attention_k_eq_v"] and config["layer_scalar"]
    assert "mixture" not in config
    assert config["per_layer_input_dim"] is None


def test_gemma4_moe_logits_match_the_reference_implementation():
    """fp32 parity: tolerance 1e-4, observed max |logit difference| 4.9e-06
    with identical argmax. The fused expert kernels arrive split in place,
    the global layer holds no v_proj, and every layer holds its scalar."""
    model, variables = fp32_decoder(GEMMA4_MOE)
    ids = np.load(GEMMA4_MOE / "input_ids.npy")
    reference = np.load(GEMMA4_MOE / "logits.npy")

    logits = np.asarray(model.apply(variables, jnp.asarray(ids, jnp.int32)))

    difference = float(np.max(np.abs(logits - reference)))
    assert difference < 1e-4, f"max |logit difference| {difference:.3e}"
    assert np.array_equal(np.argmax(logits, axis=-1), np.argmax(reference, axis=-1))
    params = variables["params"]
    experts = params["layers_0"]["moe"]["experts"]
    assert experts["gate_proj"]["kernel"].shape == (4, 32, 16)
    assert experts["down_proj"]["kernel"].shape == (4, 16, 32)
    assert params["layers_0"]["moe"]["router"]["per_expert_scale"].shape == (4,)
    assert "v_proj" not in params["layers_2"]["self_attn"]
    assert params["layers_2"]["self_attn"]["k_proj"]["kernel"].shape == (32, 16)


def test_the_layer_scalars_are_what_the_parity_tests():
    """The fixture's scalars are the reference's ones, so the parity above
    cannot tell a model that reads them from one that ignores them; a model
    fed other scalars disagrees with it, so the tree's leaf is live."""
    model, variables = fp32_decoder(GEMMA4_MOE)
    ids = jnp.asarray(np.load(GEMMA4_MOE / "input_ids.npy"), jnp.int32)
    reference = np.load(GEMMA4_MOE / "logits.npy")
    constants = dict(variables["constants"])
    for index in range(3):
        constants[f"layers_{index}"] = {**constants[f"layers_{index}"],
                                        "layer_scalar": jnp.full((1,), 0.5, jnp.float32)}
    logits = np.asarray(model.apply({**variables, "constants": constants}, ids))
    assert float(np.max(np.abs(logits - reference))) > 1e-3


def test_a_global_layer_without_k_eq_v_needs_its_v_proj(tmp_path):
    """The same weights under a config with the flag off raise on the
    global layer's missing v_proj leaf; no value projection the checkpoint
    never had is loaded."""
    directory = tmp_path / "gemma4"
    directory.mkdir()
    (directory / "model.safetensors").symlink_to(GEMMA4_MOE / "model.safetensors")
    (directory / "config.json").write_text(json.dumps(
        {**fixture_config("gemma4-moe-tiny"), "attention_k_eq_v": False}))
    with pytest.raises(ValueError, match=r"layers_2.self_attn.v_proj.kernel"):
        fp32_decoder(directory)


def test_a_routed_gemma4_without_its_expert_fields_is_refused():
    config = fixture_config("gemma4-moe-tiny")
    del config["moe_intermediate_size"]
    with pytest.raises(ValueError, match="moe_intermediate_size"):
        translate_config(config)


def test_a_gemma4_wrapper_around_the_routed_text_config_is_refused_by_name():
    """The 26B-A4B repo's config.json is the multimodal wrapper; its text
    half alone is what translates."""
    with pytest.raises(ValueError, match="multimodal wrapper"):
        translate_config(fixture_config("gemma4-26b-a4b"))


@pytest.mark.parametrize("changes, field", [
    ({"partial_rotary_type": "default"}, "partial_rotary_type"),
    ({"layer_types": ("sliding_attention",) * 3}, "layer_types"),
    ({"kv_shared_layers": (1,)}, "kv_shared_layers"),
    ({"attention_scale": 0.25}, "attention_scale"),
])
def test_standalone_gemma4_refuses_unrepresentable_native_computation(changes, field, tmp_path):
    model, variables = fp32_decoder(GEMMA4_MOE)
    with pytest.raises(ValueError, match=field):
        PretrainedDecoder.from_model(model.clone(**changes), variables).save(str(tmp_path))
    assert not (tmp_path / "config.json").exists()
