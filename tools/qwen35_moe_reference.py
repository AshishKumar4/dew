"""Write fp32 text and image references for the Qwen 3.5 MoE wrapper."""

import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import (
    AutoProcessor,
    Qwen3_5MoeConfig,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3VLProcessor,
    Qwen3VLVideoProcessor,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from dew.interop.verify import scatter_weights
from multimodal_reference import _row_slices, _tokenizer


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    directory = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'hf' / 'qwen35-moe-native-tiny'
    directory.mkdir(parents=True, exist_ok=True)
    config = Qwen3_5MoeConfig.from_dict({
        'image_token_id': 200, 'video_token_id': 201,
        'vision_start_token_id': 202, 'vision_end_token_id': 203,
        'tie_word_embeddings': False,
        'text_config': {
            'model_type': 'qwen3_5_moe_text', 'vocab_size': 256,
            'hidden_size': 64, 'num_hidden_layers': 3, 'num_attention_heads': 4,
            'num_key_value_heads': 2, 'head_dim': 16,
            'layer_types': ['linear_attention', 'full_attention', 'linear_attention'],
            'linear_num_key_heads': 2, 'linear_num_value_heads': 2,
            'linear_key_head_dim': 4, 'linear_value_head_dim': 4, 'linear_conv_kernel_dim': 3,
            'moe_intermediate_size': 16, 'shared_expert_intermediate_size': 32,
            'num_experts': 4, 'num_experts_per_tok': 2, 'mtp_num_hidden_layers': 0,
            'max_position_embeddings': 128, 'partial_rotary_factor': .5,
            'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000.,
                                'partial_rotary_factor': .5, 'mrope_interleaved': True,
                                'mrope_section': [2, 1, 1]},
            'bos_token_id': 2, 'eos_token_id': None, 'pad_token_id': 0,
        },
        'vision_config': {
            'model_type': 'qwen3_5_moe_vision', 'hidden_size': 32,
            'intermediate_size': 48, 'depth': 2, 'num_heads': 4,
            'patch_size': 8, 'temporal_patch_size': 2, 'spatial_merge_size': 2,
            'num_position_embeddings': 64, 'out_hidden_size': 64,
        },
    })
    config._attn_implementation = 'eager'
    model = Qwen3_5MoeForConditionalGeneration(config).float().eval()
    scatter_weights(model)
    model = model.to('cuda')
    model.save_pretrained(directory)
    special = {'<pad>': 0, '<eos>': 1, '<bos>': 2, '<unk>': 3,
               '<image_soft_token>': 200, '<video>': 201,
               '<start_of_image>': 202, '<end_of_image>': 203}
    tokenizer = _tokenizer(256, special, video_token='<video>',
                           vision_start_token='<start_of_image>', vision_end_token='<end_of_image>')
    processor = Qwen3VLProcessor(
        Qwen2VLImageProcessor(patch_size=8, temporal_patch_size=2, merge_size=2, do_resize=False),
        tokenizer, Qwen3VLVideoProcessor(patch_size=8, temporal_patch_size=2, merge_size=2))
    processor.save_pretrained(directory)
    processor = AutoProcessor.from_pretrained(directory, local_files_only=True)
    images = np.random.default_rng(1735).integers(0, 256, (3, 32, 32, 3), dtype=np.uint8)
    prompts = ['token7 <start_of_image><image_soft_token><end_of_image> token9',
               'token5 <start_of_image><image_soft_token><end_of_image> token8 '
               '<start_of_image><image_soft_token><end_of_image> token6']
    encoded = dict(processor(text=prompts, images=[[images[0]], [images[1], images[2]]],
                             return_tensors='pt', padding=True))
    valid = encoded['attention_mask'].bool()
    logits = torch.zeros((*encoded['input_ids'].shape, 256))
    continuation = []
    for row, single in enumerate(_row_slices(encoded, model)):
        single = {name: value.to('cuda') for name, value in single.items()}
        with torch.no_grad():
            logits[row, valid[row]] = model(**single, use_cache=False).logits[0].cpu()
            generated = model.generate(**single, max_new_tokens=3, do_sample=False, eos_token_id=None)
            continuation.append(generated[0, single['input_ids'].shape[1]:].cpu())
    plain = dict(tokenizer(['token7 token9', 'token5 token8'], return_tensors='pt', padding=True))
    with torch.no_grad():
        text_logits = model(**{key: value.to('cuda') for key, value in plain.items()}, use_cache=False).logits.cpu()
    np.save(directory / 'raw_images.npy', images)
    np.save(directory / 'logits.npy', logits.numpy())
    np.save(directory / 'continuation.npy', torch.stack(continuation).numpy())
    np.save(directory / 'text_logits.npy', text_logits.numpy())
    np.save(directory / 'text_input_ids.npy', plain['input_ids'].numpy())
    np.save(directory / 'text_attention_mask.npy', plain['attention_mask'].numpy())
    for name, value in encoded.items():
        np.save(directory / f'{name}.npy', value.numpy())
    (directory / 'prompts.json').write_text(json.dumps(prompts) + '\n')
    (directory / 'meta.json').write_text(json.dumps({
        'transformers': transformers.__version__, 'torch': torch.__version__,
        'dtype': 'float32', 'attention': 'eager', 'device': torch.cuda.get_device_name(),
        'seed': 1234, 'command': 'PYTHONPATH=src python tools/qwen35_moe_reference.py',
    }, indent=2) + '\n')
    print(directory, flush=True)


if __name__ == '__main__':
    main()
