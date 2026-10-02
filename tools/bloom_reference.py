"""Write BLOOM's fp32 same-weight decoder, ALiBi and cached generation fixture."""

import json
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import BloomConfig, BloomForCausalLM

from dew.interop.verify import probe_ids, scatter_weights


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    config = BloomConfig(vocab_size=128, hidden_size=32, n_layer=2, n_head=4,
                         bos_token_id=1, eos_token_id=None, pad_token_id=0,
                         attention_dropout=0, hidden_dropout=0)
    config._attn_implementation = 'eager'
    model = BloomForCausalLM(config).float().eval()
    scatter_weights(model)
    model = model.to('cuda')
    directory = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'hf' / 'bloom-tiny'
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
        'seed': 1234, 'command': 'PYTHONPATH=src python tools/bloom_reference.py',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
