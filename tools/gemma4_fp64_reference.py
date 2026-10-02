"""Pin the published tiny Gemma 4 text logits in float64.

Reproduce with Transformers 5.16.1 and Torch 2.14.0 on CPU:
    PYTHONPATH=src ~/.cache/dew/dew-cpu-run python tools/gemma4_fp64_reference.py

The fp32 checkpoint weights are widened exactly. RMSNorm, softmax and RoPE
also need widening: the reference pins these operations to fp32 even in a
double model. RoPE frequencies are recomputed rather than upcast.
"""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm, Gemma4TextRotaryEmbedding

CHECKPOINT = 'trl-internal-testing/tiny-Gemma4ForConditionalGeneration'
REVISION = '0dc1746b7f9f623b748e735ed5a4302eb4baf346'
PROMPT = 'The capital of France is'
OUTPUT = (Path(__file__).resolve().parents[1] / 'tests' / 'fixtures' / 'hf'
          / 'gemma4-published-tiny' / 'text_f64.npz')


def norm64(self, hidden_states):
    output = self._norm(hidden_states)
    return output * self.weight if self.with_scale else output


def rotary64(self, x, position_ids, layer_type):
    params = self.config.rope_parameters[layer_type]
    assert params['rope_type'] in ('default', 'proportional')
    dim = self.config.per_layer_config[layer_type].head_dim
    pairs = dim // 2
    if params['rope_type'] == 'proportional':
        pairs = int(dim * params['partial_rotary_factor']) // 2
    exponents = torch.arange(0, 2 * pairs, 2, dtype=torch.float64, device=x.device) / dim
    inverse = 1.0 / (params['rope_theta'] ** exponents)
    if pairs < dim // 2:
        inverse = torch.cat([inverse, torch.zeros(dim // 2 - pairs, dtype=torch.float64,
                                                 device=x.device)])
    angles = position_ids.double()[..., None] * inverse
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def write_reference(output):
    torch.set_num_threads(1)
    processor = AutoProcessor.from_pretrained(CHECKPOINT, revision=REVISION)
    batch = dict(processor(text=[PROMPT], return_tensors='pt'))
    model = AutoModelForImageTextToText.from_pretrained(
        CHECKPOINT, dtype=torch.float32, attn_implementation='eager', revision=REVISION).eval().double()
    original_softmax = torch.nn.functional.softmax

    def softmax64(x, *args, **kwargs):
        if x.dtype == torch.float64:
            kwargs['dtype'] = torch.float64
        return original_softmax(x, *args, **kwargs)

    with patch.object(Gemma4RMSNorm, 'forward', norm64), \
            patch.object(Gemma4TextRotaryEmbedding, 'forward', rotary64), \
            patch.object(torch.nn.functional, 'softmax', softmax64), torch.no_grad():
        logits = model(**batch, use_cache=False).logits.numpy()
    assert logits.dtype == np.float64
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, logits=logits, input_ids=batch['input_ids'].numpy(),
                        checkpoint=CHECKPOINT, revision=REVISION, prompt=PROMPT,
                        transformers=transformers.__version__, torch=torch.__version__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    write_reference(parser.parse_args().output)


if __name__ == '__main__':
    main()
