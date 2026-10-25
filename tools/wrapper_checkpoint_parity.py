"""Measure a small published multimodal checkpoint at highest fp32 precision against its reference.

Dew's logits, the reference's fp32 logits and the reference's float64 logits on the same batches, and
both greedy continuations with the reference's logits at each step; the accuracy verdict is the gate's
(tests/reference_error.py), read by the tests that call `measure_checkpoint`.
"""

import argparse
import dataclasses
import json
from pathlib import Path

import jax
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from dew.accuracy import distance
from dew.interop import Pretrained
from dew.sampling.text import Sampling

CONTINUATION = 3
"""Greedy tokens each batch continues by, Dew's and the reference's."""


@dataclasses.dataclass(frozen=True)
class Measured:
    """One batch measured: Dew's fp32 logits, the reference's fp32 and float64 logits, the token ids, and
    each greedy continuation with the reference's fp32 logits at every generated step."""
    modality: str
    dew: np.ndarray
    reference: np.ndarray
    truth: np.ndarray
    ids: np.ndarray
    dew_continuation: np.ndarray
    reference_continuation: np.ndarray
    continuation_logits: np.ndarray


def parity_batches(processor, model_type: str) -> list[dict]:
    """The two batches every parity measurement reads: a text prompt, and an image prompt over one fixed
    32 x 32 image (seed 67)."""
    image = np.random.default_rng(67).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    prompt = (f'The image shows {processor.image_token} a' if model_type == 'gemma4' else
              'The image shows <|vision_start|><|image_pad|><|vision_end|> a')
    return [dict(processor(text=['The capital of France is'], return_tensors='pt')),
            dict(processor(text=[prompt], images=[image], return_tensors='pt'))]


def float64_logits(checkpoint, revision, weights, batches) -> list[np.ndarray]:
    """Each batch's logits in float64, on the CPU: the model built in float64 from its config, so its
    rotary tables, the vision tower's among them, are computed in float64 rather than widened from
    float32, and every float32 pin in it (norms, softmax, a pooler's or standardizer's casts) is
    widened; its experts, where it has them, run one at a time, as Torch's grouped GEMM takes no
    float64; the fp32 checkpoint `weights` load into it exactly."""
    from tools.torch_precision import float64

    torch.set_num_threads(1)
    with float64(), torch.no_grad():
        model = AutoModelForImageTextToText.from_config(
            AutoConfig.from_pretrained(checkpoint, revision=revision), dtype=torch.float64,
            attn_implementation='eager', experts_implementation='eager').eval()
        model.load_state_dict(weights)
        return [model(**batch, use_cache=False).logits.double().numpy() for batch in batches]


def measure_checkpoint(checkpoint, output, revision=None) -> list[Measured]:
    """Each parity batch measured (`Measured`), and the measurement's record written to `output`. The
    reference runs on the CPU whatever JAX's platform, so a CPU-only Torch build supplies it beside a GPU
    and no TF32 enters it."""
    config = AutoConfig.from_pretrained(checkpoint, revision=revision)
    processor = AutoProcessor.from_pretrained(checkpoint, revision=revision)
    reference = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation='eager', revision=revision).eval()
    batches = parity_batches(processor, config.model_type)
    expected, continuations = [], []
    with torch.no_grad():
        for batch in batches:
            expected.append(reference(**batch, use_cache=False).logits.numpy())
            generated = reference.generate(**batch, max_new_tokens=CONTINUATION, do_sample=False,
                                           eos_token_id=None, output_logits=True,
                                           return_dict_in_generate=True)
            continuations.append((generated.sequences.numpy()[:, -CONTINUATION:],
                                  np.stack([step.numpy() for step in generated.logits], axis=1)))
    truths = float64_logits(checkpoint, revision, reference.state_dict(), batches)
    del reference
    loaded = Pretrained.load(checkpoint, dtype='float32', attention_impl='reference', max_seq_len=256,
                             revision=revision)
    language = loaded.model.language_model.clone(precision=jax.lax.Precision.HIGHEST)
    model = loaded.model.clone(language_model=language, precision=jax.lax.Precision.HIGHEST)
    loaded = dataclasses.replace(loaded, model=model)
    measured = []
    for batch, logits, truth, (tokens, steps), modality in zip(
            batches, expected, truths, continuations, ('text', 'image'), strict=True):
        inputs = loaded.processor.from_hf({name: value.numpy() for name, value in batch.items()})
        actual = np.asarray(model.apply(loaded.variables, inputs.tokens, **inputs.kwargs()))
        generated = loaded.text_generation()(
            inputs, CONTINUATION, key=jax.random.key(0), sampling=Sampling(temperature=0))
        measured.append(Measured(modality, actual, logits, truth, batch['input_ids'].numpy(),
                                 np.asarray(generated.tokens)[:, -CONTINUATION:], tokens, steps))
    metadata = Path(checkpoint) / '.cache' / 'huggingface' / 'download' / 'config.json.metadata'
    revision = (config._commit_hash or revision
                or (metadata.read_text().splitlines()[0] if metadata.is_file() else None))
    record = {'checkpoint': checkpoint, 'revision': revision, 'dtype': 'float32', 'precision': 'highest',
              'tf32': False, 'device': jax.devices()[0].device_kind,
              'observations': [{'modality': row.modality, 'sequence_length': int(row.truth.shape[1]),
                                'dew_rms_error': distance(row.dew, row.truth),
                                'reference_rms_error': distance(row.reference, row.truth)}
                               for row in measured]}
    Path(output).write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)
    return measured


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--revision')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    measure_checkpoint(args.checkpoint, args.output, args.revision)


if __name__ == '__main__':
    main()
