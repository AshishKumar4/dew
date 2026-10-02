"""Write a tiny GPT-NeoX reference with head-interleaved fused qkv tensors."""

import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from dew.interop.verify import probe_ids, scatter_weights


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = GPTNeoXConfig(vocab_size=128, hidden_size=32, num_hidden_layers=2,
                          num_attention_heads=4, intermediate_size=48,
                          max_position_embeddings=64, hidden_dropout=0, attention_dropout=0,
                          rope_parameters={'rope_type': 'default', 'rope_theta': 10000.,
                                           'partial_rotary_factor': .5},
                          use_parallel_residual=True, bos_token_id=1, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = 'eager'
    model = GPTNeoXForCausalLM(config).float().eval()
    scatter_weights(model)
    model = model.to('cuda')
    directory = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'hf' / 'gpt-neox-tiny'
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(directory)
    ids = probe_ids(config.vocab_size)
    with torch.no_grad():
        logits = model(torch.tensor(ids, device='cuda'), use_cache=False).logits.cpu().numpy()
        generated = model.generate(torch.tensor(ids[:1, :4], device='cuda'),
                                   max_new_tokens=6, do_sample=False, eos_token_id=None, pad_token_id=0).cpu().numpy()
    np.save(directory / 'input_ids.npy', ids)
    np.save(directory / 'logits.npy', logits)
    np.save(directory / 'generated.npy', generated)
    (directory / 'meta.json').write_text(json.dumps({
        'transformers': transformers.__version__, 'torch': torch.__version__,
        'dtype': 'float32', 'attention': 'eager', 'device': torch.cuda.get_device_name(),
        'seed': 1234, 'command': 'PYTHONPATH=src python tools/gpt_neox_reference.py',
    }, indent=2) + '\n')
    print(directory, flush=True)


if __name__ == '__main__':
    main()
