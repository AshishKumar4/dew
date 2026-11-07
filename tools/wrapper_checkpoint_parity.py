"""Measure a small published multimodal checkpoint at highest fp32 precision against its reference.

Dew's logits, the reference's fp32 logits and the reference's float64 logits on the same batches, and
both greedy continuations with the reference's logits at each step; the accuracy verdict is the gate's
(tests/reference_error.py), whose tests draw the batches over its seeds through `checkpoint_runs`.
"""

import argparse
import dataclasses
import functools
import json
from pathlib import Path

import jax
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

from dew._accuracy import distance
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


PROMPT_WORDS = ("the", "image", "shows", "a", "red", "bird", "over", "blue", "water", "and", "small", "tree")
"""The words a drawn prompt's text is made of."""


def parity_batches(processor, model_type: str, seed: int = 0) -> list[dict]:
    """The two batches a parity measurement reads at `seed`, as NumPy arrays: a text prompt of five words
    drawn from PROMPT_WORDS, and an image prompt over a drawn 32 x 32 image."""
    generator = np.random.default_rng([67, seed])
    text = " ".join(generator.choice(PROMPT_WORDS, 5))
    image = generator.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    prompt = (f'{text} {processor.image_token} a' if model_type == 'gemma4' else
              f'{text} <|vision_start|><|image_pad|><|vision_end|> a')
    return [{name: value.numpy() for name, value in batch.items()}
            for batch in (processor(text=[text], return_tensors='pt'),
                          processor(text=[prompt], images=[image], return_tensors='pt'))]


def tensors(batch) -> dict[str, torch.Tensor]:
    return {name: torch.from_numpy(np.asarray(value)) for name, value in batch.items()}


@dataclasses.dataclass(frozen=True)
class Runs:
    """A published checkpoint loaded once as the reference's fp32 model, its float64 model and Dew's at
    highest fp32 precision: each a run of a batch (`parity_batches`) to its logits."""
    processor: object
    model_type: str
    reference_model: torch.nn.Module
    wide_model: torch.nn.Module
    loaded: Pretrained

    def reference(self, batch) -> np.ndarray:
        with torch.no_grad():
            return self.reference_model(**tensors(batch), use_cache=False).logits.numpy()

    def truth(self, batch) -> np.ndarray:
        """The batch's logits in float64, on the CPU: the float64 model's (`checkpoint_runs`)."""
        from tools.torch_precision import float64

        with float64(), torch.no_grad():
            return self.wide_model(**tensors(batch), use_cache=False).logits.double().numpy()

    @functools.cached_property
    def forward(self):
        """Dew's whole forward, compiled once for every batch."""
        return jax.jit(self.loaded.model.apply)

    def dew(self, batch) -> np.ndarray:
        """Dew's logits of `batch`, its processor's host reading of it outside the compiled forward."""
        inputs = self.loaded.processor.from_hf(dict(batch))
        return np.asarray(self.forward(self.loaded.variables, inputs.tokens, **inputs.kwargs()))

    def continued(self, batch) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dew's greedy continuation of `batch` by CONTINUATION tokens, the reference's, and the reference's
        logits at each of its steps."""
        with torch.no_grad():
            generated = self.reference_model.generate(
                **tensors(batch), max_new_tokens=CONTINUATION, do_sample=False, eos_token_id=None,
                output_logits=True, return_dict_in_generate=True)
        inputs = self.loaded.processor.from_hf(dict(batch))
        mine = self.loaded.text_generation()(inputs, CONTINUATION, key=jax.random.key(0),
                                             sampling=Sampling(temperature=0))
        return (np.asarray(mine.tokens)[:, -CONTINUATION:], generated.sequences.numpy()[:, -CONTINUATION:],
                np.stack([step.numpy() for step in generated.logits], axis=1))


def checkpoint_runs(checkpoint, revision) -> Runs:
    """`checkpoint` at its pinned `revision` loaded for its runs (`Runs`). The reference runs on the CPU
    whatever JAX's platform, so a CPU-only Torch build supplies it beside a GPU and no TF32 enters it. Its
    float64 model is built in float64 from its config, so its rotary tables, the vision tower's among
    them, are computed in float64 rather than widened from float32, and every float32 pin in it (norms,
    softmax, a pooler's or standardizer's casts) is widened; its experts, where it has them, run one at a
    time, as Torch's grouped GEMM takes no float64; the fp32 checkpoint's weights load into it exactly."""
    from tools.torch_precision import float64

    config = AutoConfig.from_pretrained(checkpoint, revision=revision)
    reference = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.float32, attn_implementation='eager', revision=revision).eval()
    torch.set_num_threads(1)
    with float64():
        wide = AutoModelForImageTextToText.from_config(
            config, dtype=torch.float64, attn_implementation='eager', experts_implementation='eager').eval()
        wide.load_state_dict(reference.state_dict())
    loaded = Pretrained.load(checkpoint, dtype='float32', attention_impl='reference', max_seq_len=256,
                             revision=revision)
    language = loaded.model.language_model.clone(precision=jax.lax.Precision.HIGHEST)
    model = loaded.model.clone(language_model=language, precision=jax.lax.Precision.HIGHEST)
    return Runs(AutoProcessor.from_pretrained(checkpoint, revision=revision), config.model_type, reference,
                wide, dataclasses.replace(loaded, model=model))


def measure_checkpoint(checkpoint, output, revision) -> list[Measured]:
    """Each parity batch of `checkpoint` at its pinned `revision` measured (`Measured`), and the
    measurement's record written to `output`."""
    runs = checkpoint_runs(checkpoint, revision)
    measured = [Measured(modality, runs.dew(batch), runs.reference(batch), runs.truth(batch),
                         batch['input_ids'], *runs.continued(batch))
                for batch, modality in zip(parity_batches(runs.processor, runs.model_type), ('text', 'image'),
                                           strict=True)]
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
    parser.add_argument('--revision', required=True, help='the commit the checkpoint is pinned at')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    measure_checkpoint(args.checkpoint, args.output, args.revision)


if __name__ == '__main__':
    main()
