"""Compare a small published multimodal checkpoint at highest fp32 precision."""

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from dew.interop import load_pretrained
from dew.interop.verify import _ROUNDING
from dew.sampling.text import Sampling

# The existing qwen35-tiny fp32 fixture measures 28 ulp per layer per unit
# logit at 12 tokens and key width 12 (9.1e-5 over four layers, logits 6.8).
# The delta rule's normalized keys, beta in [0, 1] and decay in [0, 1] keep
# its state-update operator nonexpansive. Local update rounding therefore
# accumulates over tokens; the query/key reductions contribute gamma_key_dim.
QWEN35_ROUNDING = 28.0
QWEN35_CALIBRATION_LENGTH = 12
QWEN35_CALIBRATION_KEY_DIM = 12


def rounding_bound(config, reference):
    text = config.get_text_config()
    scale = float(np.max(np.abs(reference)))
    epsilon = np.finfo(np.float32).eps
    if config.model_type not in ('qwen3_5', 'qwen3_5_moe'):
        return float(2 * _ROUNDING * epsilon * text.num_hidden_layers * scale)
    def gamma(terms):
        unit_roundoff = epsilon / 2
        return terms * unit_roundoff / (1 - terms * unit_roundoff)
    length_ratio = reference.shape[1] / QWEN35_CALIBRATION_LENGTH
    contraction_ratio = gamma(text.linear_key_head_dim) / gamma(QWEN35_CALIBRATION_KEY_DIM)
    recurrent_rounding = (QWEN35_ROUNDING - _ROUNDING) * length_ratio * contraction_ratio
    return float(2 * (_ROUNDING + recurrent_rounding) * epsilon * text.num_hidden_layers * scale)


def check_checkpoint(checkpoint, output, revision=None):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = AutoConfig.from_pretrained(checkpoint, revision=revision)
    processor = AutoProcessor.from_pretrained(checkpoint, revision=revision)
    device = 'cuda' if jax.default_backend() == 'gpu' else 'cpu'
    reference = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation='eager', revision=revision).eval().to(device)
    image = np.random.default_rng(67).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    prompt = (f'The image shows {processor.image_token} a' if config.model_type == 'gemma4' else
              'The image shows <|vision_start|><|image_pad|><|vision_end|> a')
    batches = [dict(processor(text=['The capital of France is'], return_tensors='pt')),
               dict(processor(text=[prompt], images=[image], return_tensors='pt'))]
    expected = []
    continuations = []
    with torch.no_grad():
        for batch in batches:
            placed = {name: value.to(device) for name, value in batch.items()}
            expected.append(reference(**placed, use_cache=False).logits.cpu().numpy())
            continuations.append(reference.generate(**placed, max_new_tokens=3,
                                                     do_sample=False, eos_token_id=None).cpu().numpy()[:, -3:])
    del reference
    torch.cuda.empty_cache()
    loaded = load_pretrained(checkpoint, dtype='float32', attention_impl='reference', max_seq_len=256,
                             revision=revision)
    language = loaded.model.language_model.clone(precision=jax.lax.Precision.HIGHEST)
    model = loaded.model.clone(language_model=language, precision=jax.lax.Precision.HIGHEST)
    loaded = dataclasses.replace(loaded, model=model)
    observations = []
    for index, batch in enumerate(batches):
        inputs = loaded.processor.from_hf({name: value.numpy() for name, value in batch.items()})
        actual = np.asarray(model.apply(loaded.variables, inputs.tokens, **inputs.kwargs()))
        error = float(np.max(np.abs(actual - expected[index])))
        bound = rounding_bound(config, expected[index])
        argmax = bool(np.array_equal(actual.argmax(-1), expected[index].argmax(-1)))
        generated = loaded.text_generation()(inputs, 3, key=jax.random.key(0), sampling=Sampling(temperature=0))
        agreement = bool(np.array_equal(np.asarray(generated.tokens)[:, -3:], continuations[index]))
        observations.append({'modality': 'text' if index == 0 else 'image',
                             'max_abs_error': error, 'bound': bound,
                             'sequence_length': int(expected[index].shape[1]),
                             'key_dim': int(getattr(config.get_text_config(), 'linear_key_head_dim',
                                                     config.get_text_config().head_dim)),
                             'max_reference_logit': float(np.max(np.abs(expected[index]))),
                             'argmax_agreement': argmax, 'generation_agreement': agreement})
    metadata = Path(checkpoint) / '.cache' / 'huggingface' / 'download' / 'config.json.metadata'
    revision = config._commit_hash or revision or (metadata.read_text().splitlines()[0] if metadata.is_file() else None)
    result = {'checkpoint': checkpoint, 'revision': revision,
              'dtype': 'float32', 'precision': 'highest', 'tf32': False,
              'bound': 'Qwen3.5 fixture calibration, linear token count and gamma_key_dim scaling',
              'device': jax.devices()[0].device_kind, 'observations': observations}
    Path(output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    assert all(row['max_abs_error'] < row['bound'] and row['argmax_agreement']
               and row['generation_agreement'] for row in observations), result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--revision')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check_checkpoint(args.checkpoint, args.output, args.revision)


if __name__ == '__main__':
    main()
