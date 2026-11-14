"""The native multimodal families' reference models (tests/multimodal_families.py: each family's tiny
checkpoint and its processor, under transformers) run on drawn images and audio, in float32 and float64:
the reference arm of tools/venv_reference.py's `multimodal-<family>` sites (multimodal_families.DRAWN; its
protocol tools/reference_worker.py), run in an environment of transformers 5.16.1, torch and torchvision,
on the CPU.

    python tools/multimodal_worker.py multimodal-FAMILY[-text]    `IN OUT` lines on stdin, each `done`

IN holds each draw's `<draw>.images` `[3, H, W, 3]` uint8, under the family's prompts, and for a family
that hears, its `<draw>.audio_<clip>` waveforms; a text site's (multimodal_families.TEXT_DRAWN), its
`<draw>.ids` token rows alone (`multimodal_families.text_rows`). The processor encodes the padded batch
(`multimodal_reference._encode`: one image in the first row, two in the second), and the model runs one
row of it at a time (`multimodal_reference.numerics`). OUT gets `<draw>.reference.<leaf>` and
`<draw>.truth.<leaf>` (multimodal_families.OUTPUTS) in float32 and float64, the latter the model's float64
twin (`_dew_torch.precision.float64_twin`) inside `float64` over the encoding widened.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from multimodal_reference import _encode, numerics
from reference_worker import draws, paired, serve

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"


def run(family: str, text: bool, cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's leaves under the family's checkpoint, in float32 and float64, over its images and audio,
    or with `text` its token rows alone (`multimodal_families.text_rows`)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from _dew_torch.precision import float64, float64_twin

    references, checkpoint = FAMILIES[family]
    source = ROOT / checkpoint
    processor = AutoProcessor.from_pretrained(source, local_files_only=True)
    prompts = json.loads((ROOT / references / "prompts.json").read_text())
    out = {}
    for name, inputs in draws(cases).items():
        clips = sorted(key for key in inputs if key.startswith("audio_"))
        encoded = ({key: torch.from_numpy(value) for key, value in text_rows(inputs["ids"]).items()}
                   if text else _encode(processor, inputs["images"], prompts,
                                        [inputs[key] for key in clips] if clips else None))
        model = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.float32,
                                                            attn_implementation="eager").eval()
        twin = float64_twin(model)
        with float64():
            widened = {key: value.double() if value.is_floating_point() else value
                       for key, value in encoded.items()}
            truth = numerics(twin, widened)
        # The float32 run after its truth, as each trains its own model in place; the answer's arrays in that
        # order, the truth's first.
        answer = paired(numerics(model, encoded), truth)
        ordered = sorted(answer, key=lambda key: key.startswith("reference."))
        out.update({f"{name}.{key}": answer[key] for key in ordered})
    return out


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].startswith("multimodal-"):
        raise SystemExit(__doc__)
    sys.path[:0] = [str(ROOT.parents[2]), str(ROOT.parents[1]), str(ROOT.parents[2] / "src")]
    from multimodal_families import FAMILIES, text_rows

    torch.set_num_threads(1)
    family, _, kind = sys.argv[1].removeprefix("multimodal-").partition("-")
    serve(lambda cases: run(family, kind == "text", cases))
