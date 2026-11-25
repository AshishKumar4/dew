#!/usr/bin/env python3
"""The Inkling 2 fixture tests/test_inkling2.py checks against, from transformers' own Inkling 2
(huggingface/transformers at REVISION, models/inkling2_loop/modeling_inkling2_loop.py), which the suite's
transformers predates: it runs in the suite's environment with that transformers over it
(tools/inkling2_fixture.requirements.txt):

    uv venv --system-site-packages -p .venv-3.12/bin/python .venv-inkling2
    uv pip install -p .venv-inkling2 -r tools/inkling2_fixture.requirements.txt
    PYTHONPATH=src:tools:tests .venv-inkling2/bin/python tools/inkling2_fixture.py [--quantized] [--recording]

tests/fixtures/hf/inkling2-tiny/ is a random Inkling2LoopForCausalLM at `TEXT`, the release's text config
(thinkingmachines/Inkling-2-Mini's config.json) at eight layers: two global, one inside the loop's core
[2, 6) and one past it, its three passes merging their messages; blocks of two layers; windows shorter
than the drawn ids; eight experts; and a head scoring fewer rows than the embedding holds. Its config.json
is the release's wrapper over it and its model.safetensors holds the weights in the release's own names and
layout (`release_tensors`), which transformers' conversion (conversion_mapping.py `inkling2_loop_text`)
reads back to the same weights, as the write checks. `--recording` records instead its logits over check 1's
seeds, whole and at the shapes a cached decode computes them (`STEPPED`), against its float64 twin, as
recorded_logits.npz there (`recorded_references.record_logits`). `--quantized` writes or records
inkling2-tiny-nvfp4 instead: the same model with its experts stored as the release stores them
(`quantized_experts`), which transformers dequantizes on read (`Inkling2LoopDequantizeExperts`).
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

REVISION = '3a96649547593bacad4b65d43442ccdf15d0ae0f'
TINY = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'hf' / 'inkling2-tiny'
QUANTIZED = TINY.parent / 'inkling2-tiny-nvfp4'
NVFP4_LAYERS = 6
"""The layers whose experts the quantized fixture stores NVFP4, the rest MXFP8, as the release stores its
first 20 of 24."""
SHAPE = (2, 40)
"""The ids' shape check 1 draws in."""
STEPPED = 24
"""The tokens a cached decode prefills before it steps a token at a time."""

TEXT = {
    'model_type': 'inkling2_loop_text', 'model_max_length': 4096, 'hidden_size': 64, 'num_hidden_layers': 8,
    'vocab_size': 136, 'unpadded_vocab_size': 128, 'num_attention_heads': 4, 'num_key_value_heads': 2,
    'head_dim': 16, 'swa_head_dim': 16, 'swa_num_attention_heads': 4, 'swa_num_key_value_heads': 2,
    'd_rel': 4, 'rel_extent': 32, 'q_bias': False, 'o_bias': False, 'log_scaling_n_floor': 8,
    'log_scaling_alpha': 0.1, 'rms_norm_eps': 1e-06, 'use_value_norm_no_gamma': True,
    'qk_l2_clip_threshold': 22.627416997969522, 'rel_l2_clip_threshold': 8.0,
    'local_layer_ids': [0, 1, 2, 4, 5, 6],
    'layer_types': ['hybrid_sliding'] * 3 + ['hybrid'] + ['hybrid_sliding'] * 3 + ['hybrid'],
    'sconv_kernel_size': 4, 'final_logit_softcapping': 30.0, 'sliding_window_size': 16,
    'n_layers_per_attn_res_block': 2, 'attn_res_normalize_values': True, 'n_routed_experts': 8,
    'num_experts_per_tok': 2, 'n_shared_experts': 0, 'intermediate_size': 32, 'mlp_gate_softcap': 4.0,
    'mlp_up_softcap': 25.0, 'route_scale': 8.0, 'use_gate_bias': True,
    'loop_config': {'core_start': 2, 'core_end': 6, 'num_passes': 3, 'gate_rank': 8, 'is_varloop': False}}
"""The tiny text config, in the release's fields. transformers reads its layers' kinds off `layer_types`
(each sixth layer global where it states none) and the release's code off `local_layer_ids`: both are
stated, alike. `log_scaling_n_floor` is low enough that the global layers' query temperature grows within
the drawn ids, and `rel_extent` reaches past the window, where a sliding layer's bank spans the window."""

_SPELLINGS = (
    (r'^model\.final_input\.query$', 'model.attn_res.output_query'),
    (r'^model\.layers\.(\d+)\.mlp_input\.query$', r'model.attn_res.mlp_queries.\1'),
    (r'\.embed_tokens\.embed_norm\.', '.embed_norm.'), (r'\.embed_tokens\.', '.embed.'),
    (r'\.mlp\.experts\.gate_up_proj$', '.mlp.experts.w13_weight'),
    (r'\.mlp\.experts\.down_proj$', '.mlp.experts.w2_weight'),
    (r'\.mlp\.gate\.e_score_correction_bias$', '.mlp.gate.bias'),
    (r'\.self_attn\.q_proj\.linear\.', '.attn.wq_du.'), (r'\.self_attn\.k_proj\.linear\.', '.attn.wk_dv.'),
    (r'\.self_attn\.v_proj\.linear\.', '.attn.wv_dv.'), (r'\.self_attn\.q_proj\.norm\.', '.attn.q_l2_clip.'),
    (r'\.self_attn\.(k|v)_proj\.sconv\.conv1d\.weight$', r'.attn.\1_sconv.weight'),
    (r'\.self_attn\.(k|v)_proj\.sconv\.static_weight$', r'.attn.\1_sconv.static_w'),
    (r'\.self_attn\.(k|v)_proj\.sconv\.weight_proj\.weight$', r'.attn.\1_sconv.weight_proj'),
    (r'\.self_attn\.relative_position_bias\.linear\.', '.attn.wr_du.'),
    (r'\.self_attn\.relative_position_bias\.norm\.', '.attn.rel_l2_clip.'),
    (r'\.self_attn\.relative_position_bias\.proj$', '.attn.rel_logits_proj.proj'),
    (r'\.self_attn\.o_proj\.linear\.', '.attn.wo_ud.'),
    (r'\.self_attn\.o_proj\.gate_proj\.', '.attn.gate_attn.'),
    (r'\.self_attn\.sink_proj\.', '.attn.attn_sink_proj.'),
    (r'\.post_attention_convolution\.conv1d\.', '.attn_sconv.'),
    (r'\.post_mlp_convolution\.conv1d\.', '.mlp_sconv.'),
    (r'\.post_attention_layernorm\.', '.mlp_norm.'), (r'\.(q|k|v)_input_layernorm\.', r'.\1_input_norm.'),
    (r'\.input_layernorm\.', '.attn_norm.'),
    (r'\.q_message_gate\.', '.controllers.attn.'), (r'\.(k|v|mlp)_message_gate\.', r'.controllers.\1.'),
    (r'\.down_proj\.weight$', '.down_weight'), (r'\.forget_gate_proj\.weight$', '.forget_weight'),
    (r'\.message_gate_proj\.weight$', '.gain_weight'),
    (r'^model\.', 'model.llm.'),
)
"""transformers' names onto the release's: conversion_mapping.py's `inkling2_text` and `inkling2_loop_text`
renames read backwards, in an order where none rewrites another's result."""


def _modeling():
    from transformers.models.inkling2_loop import configuration_inkling2_loop, modeling_inkling2_loop

    return configuration_inkling2_loop, modeling_inkling2_loop


def release_tensors(model, vocabulary: int) -> dict[str, torch.Tensor]:
    """`model`'s weights in the release's names and layout, its conversion read backwards: each attention
    query one layer down (`Inkling2IndexOffsetWeightRename`), the loop's message queries as the two stacks
    the release keeps (`Inkling2LoopMergeMessageQueries`), each expert's gate and up rows interleaved
    (`Interleave(dim=1)`), and the head at the embedding's `vocabulary` rows, the release's padding rows
    random, which a load must leave unscored (`Inkling2UnpadVocab`)."""
    stored = {}
    for name, tensor in model.state_dict().items():
        tensor = tensor.detach()
        if name == 'lm_head.weight':
            rows, width = tensor.shape
            padding = torch.randn(vocabulary - rows, width, generator=torch.Generator().manual_seed(7)) * 0.05
            stored['model.llm.unembed.weight'] = torch.cat([tensor, padding.to(tensor.dtype)])
            continue
        if name == 'model.message_input.query':
            stored['model.llm.middle_cycle_cross_pass_queries'] = tensor[:, [0, 3]].flatten(0, 1)
            stored['model.llm.middle_cycle_cross_pass_kv_queries'] = tensor[:, 1:3].flatten(0, 1)
            continue
        if (found := re.fullmatch(r'model\.layers\.(\d+)\.attention_input\.query', name)) is not None:
            stored[f'model.llm.attn_res.attn_queries.{int(found[1]) - 1}'] = tensor
            continue
        if name.endswith('.mlp.experts.gate_up_proj'):
            experts, rows, width = tensor.shape
            halves = tensor.reshape(experts, 2, rows // 2, width)
            tensor = halves.transpose(1, 2).reshape(experts, rows, width)
        for pattern, spelling in _SPELLINGS:
            name = re.sub(pattern, spelling, name)
        stored[name] = tensor
    return {name: tensor.contiguous() for name, tensor in stored.items()}


def write_tiny() -> None:
    """The tiny model, random, written in the release's layout and read back through transformers."""
    from safetensors.torch import save_file

    from dew.interop.verify import scatter_weights

    configuration, modeling = _modeling()
    TINY.mkdir(parents=True, exist_ok=True)
    config = configuration.Inkling2LoopTextConfig(**TEXT)
    config._attn_implementation = 'eager'
    torch.manual_seed(0)
    model = modeling.Inkling2LoopForCausalLM(config).float().eval()
    scatter_weights(model, 1234)
    save_file(release_tensors(model, TEXT['vocab_size']), TINY / 'model.safetensors',
              metadata={'format': 'pt'})
    wrapper = {'architectures': ['Inkling2LoopForConditionalGeneration'], 'model_type': 'inkling2_loop',
               'text_config': TEXT}
    (TINY / 'config.json').write_text(json.dumps(wrapper, indent=2) + '\n')
    read = modeling.Inkling2LoopForCausalLM.from_pretrained(TINY, config=config,
                                                            dtype=torch.float32).state_dict()
    written = model.state_dict()
    differing = sorted(set(read) ^ set(written)) or [name for name in written
                                                     if not torch.equal(read[name], written[name])]
    if differing:
        raise SystemExit(f"transformers reads the written release back otherwise: {differing}")


def quantized_experts(name: str, shape: tuple[int, int, int], nvfp4: bool,
                      generator: torch.Generator) -> dict[str, torch.Tensor]:
    """One random expert bank `[experts, rows, columns]` in the release's storage of it: NVFP4's packed E2M1
    pairs under an F8_E4M3 scale per 16 and a per-expert `amax` over `outer_scale_denominator`, or MXFP8's
    F8_E4M3 values under a U8 exponent per 32, at the magnitudes `scatter_weights` draws."""
    experts, rows, columns = shape
    if not nvfp4:
        values = torch.randn(experts, rows, columns, generator=generator).to(torch.float8_e4m3fn)
        exponents = torch.randint(121, 125, (experts, rows, columns // 32), generator=generator)
        return {name: values, f'{name}_scale': exponents.to(torch.uint8)}
    codes = torch.randint(0, 256, (experts, rows, columns // 2), generator=generator).to(torch.uint8)
    scales = torch.rand(experts, rows, columns // 16, generator=generator) * 2 + 0.25
    return {name: codes, f'{name}_scale': scales.to(torch.float8_e4m3fn),
            f'{name}_amax': torch.rand(experts, generator=generator) + 1,
            f'{name}_outer_scale_denominator': torch.tensor([6.0 * 448.0])}


def write_quantized() -> None:
    """inkling2-tiny with its experts stored quantized (`quantized_experts`), its config stating the
    release's `quantization_config`."""
    from safetensors.torch import load_file, save_file

    QUANTIZED.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(11)
    stored = {}
    for name, tensor in load_file(TINY / 'model.safetensors').items():
        found = re.fullmatch(r'model\.llm\.layers\.(\d+)\.mlp\.experts\.w(13|2)_weight', name)
        if found is None:
            stored[name] = tensor
        else:
            nvfp4 = int(found[1]) < NVFP4_LAYERS
            stored.update(quantized_experts(name, tuple(tensor.shape), nvfp4, generator))
    save_file(stored, QUANTIZED / 'model.safetensors', metadata={'format': 'pt'})
    wrapper = {**json.loads((TINY / 'config.json').read_text()),
               'quantization_config': {'quant_method': 'modelopt', 'quant_algo': 'NVFP4'}}
    (QUANTIZED / 'config.json').write_text(json.dumps(wrapper, indent=2) + '\n')


def draw(seed: int) -> dict[str, np.ndarray]:
    """Check 1's ids, uniform over the embedding's rows."""
    return {'input_ids': np.random.default_rng(seed).integers(0, TEXT['vocab_size'], SHAPE).astype(np.int32)}


def write_recording(directory: Path) -> None:
    """The fixture at `directory` as transformers reads it, its experts dequantized where stored quantized,
    fp32 on eager attention, and its float64 twin, recorded over check 1's seeds, whole and stepped."""
    import transformers
    from recorded_references import record_logits

    from _dew_torch.precision import float64_twin

    configuration, modeling = _modeling()
    config = configuration.Inkling2LoopTextConfig(**TEXT)
    model = modeling.Inkling2LoopForCausalLM.from_pretrained(
        directory, config=config, dtype=torch.float32, attn_implementation='eager').eval()
    # The experts' own loop (`Inkling2LoopExperts.forward`), which runs in float64 as torch's grouped
    # products do not; its modules read the choice off the config at each call (integrations/moe.py:633).
    model.config._experts_implementation = 'eager'
    truth = float64_twin(model, lambda: modeling.Inkling2LoopForCausalLM(model.config))
    truth.config._experts_implementation = 'eager'
    flag = ' --quantized' if directory == QUANTIZED else ''
    command = ('PYTHONPATH=src:tools:tests .venv-inkling2/bin/python tools/inkling2_fixture.py'
               f'{flag} --recording')
    record_logits(directory, model, truth, command, draw=draw, stepped=STEPPED,
                  reference='Inkling2LoopForCausalLM (transformers)',
                  revision=f'huggingface/transformers@{REVISION}', license='Apache-2.0',
                  transformers=transformers.__version__, torch=torch.__version__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quantized', action='store_true',
                        help="the fixture whose experts are stored as the release stores them")
    parser.add_argument('--recording', action='store_true',
                        help="record the tiny model's logits over the seeds")
    arguments = parser.parse_args()
    if arguments.recording:
        write_recording(QUANTIZED if arguments.quantized else TINY)
    elif arguments.quantized:
        write_quantized()
    else:
        write_tiny()
