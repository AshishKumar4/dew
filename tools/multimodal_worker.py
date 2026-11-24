"""The native multimodal families' reference models (tests/multimodal_families.py: each family's tiny
checkpoint and its processor, under transformers) run on drawn images and audio, in float32 and float64:
the reference arm of tools/venv_reference.py's `multimodal-<family>` sites (multimodal_families.DRAWN; its
protocol tools/reference_worker.py), run in an environment of transformers 5.16.1, torch and torchvision,
on the device `DEVICE` names (`DEW_REFERENCE_DEVICE`: the CPU by default, `cuda` a GPU's float64 units,
which a weighed site's rounding scale takes: its tapped runs, a dozen float64 passes a draw, took hours a
draw on one CPU thread, rows 5233-5237). On a GPU every operation runs its deterministic algorithm, its
products in float32 or float64 as asked (no TF32), so a draw repeats to the bit on its GPU, as a resumed
recording reads it; a recording runs on one device throughout.

    python tools/multimodal_worker.py multimodal-FAMILY[-text]    `IN OUT` lines on stdin, each `done`

IN holds each draw's `<draw>.images` `[3, H, W, 3]` uint8, under the family's prompts, and for a family
that hears, its `<draw>.audio_<clip>` waveforms; a text site's (multimodal_families.TEXT_DRAWN), its
`<draw>.ids` token rows alone (`multimodal_families.text_rows`). The processor encodes the padded batch
(`multimodal_reference._encode`: one image in the first row, two in the second), and the model runs one
row of it at a time (`multimodal_reference.numerics`). OUT gets `<draw>.reference.<leaf>` and
`<draw>.truth.<leaf>` (multimodal_families.OUTPUTS) in float32 and float64, the latter the model's float64
twin (`_dew_torch.precision.float64_twin`) inside `float64` over the encoding widened. A draw asked for its
rounding scale (`<draw>.weighed`, tools/venv_reference.py's weighed site's) gets that alone,
`<draw>.scale.<leaf>`: the float64 twin's own (`_dew_torch.rounding.rounding_scale`, every operation tapped
in float32's unit, its draws keyed by the draw's bytes), each tapped run on its own copy of the twin, as
`numerics` trains the model it runs in place.
"""

import copy
import functools
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from multimodal_reference import _encode, numerics
from reference_worker import draws, paired, serve

DEVICE = os.environ.get("DEW_REFERENCE_DEVICE", "cpu")
"""The device the worker's models run on: `cpu` (one thread) or `cuda`."""

ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"


def run(family: str, text: bool, cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's leaves under the family's checkpoint, in float32 and float64, over its images and audio,
    or with `text` its token rows alone (`multimodal_families.text_rows`); a draw asked for its rounding
    scale (`weighed`), that alone (`_scale`)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from _dew_torch.precision import float64

    processor, prompts, loaded = _loaded(family)
    widest = _twin(family)
    out = {}
    for name, inputs in draws(cases).items():
        weighed = bool(inputs.pop("weighed", False))
        clips = sorted(key for key in inputs if key.startswith("audio_"))
        encoded = ({key: torch.from_numpy(value) for key, value in text_rows(inputs["ids"]).items()}
                   if text else _encode(processor, inputs["images"], prompts,
                                        [inputs[key] for key in clips] if clips else None))
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        # A fused attention (Gemma 3n's vision tower calls torch's scaled dot product directly) runs in its
        # eager form in every run: a GPU's fused kernels take no float64 (cutlassF: no kernel found to launch,
        # GH200, IG-MMGPU15), nor run deterministically backward, and a float32 run so formed is the
        # host's, whose recordings these resume.
        with sdpa_kernel(SDPBackend.MATH):
            with float64():
                widened = {key: value.double() if value.is_floating_point() else value
                           for key, value in encoded.items()}
                if weighed:
                    out.update({f"{name}.scale.{leaf}": value
                                for leaf, value in _scale(widest, widened, _keyed(inputs)).items()})
                    continue
                truth = numerics(copy.deepcopy(widest), widened)
            # The float32 run after its truth, as each trains its own model in place, each on its own copy
            # of the worker's; the answer's arrays in that order, the truth's first.
            answer = paired(numerics(copy.deepcopy(loaded), encoded), truth)
        ordered = sorted(answer, key=lambda key: key.startswith("reference."))
        out.update({f"{name}.{key}": answer[key] for key in ordered})
    return out


@functools.cache
def _loaded(family: str):
    """The family's processor, prompts and float32 checkpoint, loaded once a worker: every request reads
    them, each draw running its own copy of the model (`run`), which `numerics` trains in place, so the one
    loaded stays as stored (a weighed recording's requests, a draw or a scale each, loaded the checkpoint's
    shards again every request: 37 loads a slice, IG-MMGPU22 and 23)."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    references, checkpoint = FAMILIES[family]
    source = ROOT / checkpoint
    processor = AutoProcessor.from_pretrained(source, local_files_only=True,
                                              **PROCESSOR_OPTIONS.get(family, {}))
    prompts = json.loads((ROOT / references / "prompts.json").read_text())
    loaded = AutoModelForImageTextToText.from_pretrained(source, dtype=torch.float32,
                                                         attn_implementation="eager").eval()
    # `Module.to` itself: transformers types its model's `to` as its generation mixin's, which no device is.
    torch.nn.Module.to(loaded, DEVICE)
    return processor, prompts, loaded


@functools.cache
def _twin(family: str):
    """The family's float64 twin (`_dew_torch.precision.float64_twin`) of the checkpoint `_loaded` holds,
    made once a worker and never run: each draw runs copies of it (`run`, `_scale`), which `numerics` trains
    in place. A draw made its own twin of its own copy of the checkpoint, the twin a weighed draw never ran,
    two model copies a draw beside the scale's ten (a recording's cost gate, 2026-11-23)."""
    from _dew_torch.precision import float64_twin

    return float64_twin(copy.deepcopy(_loaded(family)[2]))


def _keyed(inputs: dict[str, np.ndarray]) -> int:
    """A draw's key for its tap's draws, read from its arrays' bytes: the request carries no seed, and the
    same draw keys alike in every recording of it."""
    digest = hashlib.sha256()
    for key in sorted(inputs):
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(inputs[key]).tobytes())
    return int.from_bytes(digest.digest()[:4], "little")


def _scale(twin, widened: dict, seed: int) -> dict[str, np.ndarray]:
    """The float64 `twin`'s rounding scale of each `numerics` leaf over `widened` (`rounding_scale`, float32's
    unit, its draws keyed by `seed`), each of its runs on a copy of the twin made here, outside the tap: a
    copy is no step of the computation weighed, and `numerics` steps its model in place. Its caller runs it
    with a fused attention in its eager form (`run`), as the tap reads one, whose backward's several results
    no rule weighs apart."""
    from _dew_torch.rounding import DRAWS, rounding_scale

    copies = [copy.deepcopy(twin) for _ in range(DRAWS + 1)]
    return rounding_scale(lambda: numerics(copies.pop(), widened), float(np.finfo(np.float32).eps) / 2,
                          seed)[1]


if __name__ == "__main__":
    if len(sys.argv) != 2 or not sys.argv[1].startswith("multimodal-"):
        raise SystemExit(__doc__)
    sys.path[:0] = [str(ROOT.parents[2]), str(ROOT.parents[1]), str(ROOT.parents[2] / "src")]
    from multimodal_families import FAMILIES, PROCESSOR_OPTIONS, text_rows

    torch.set_num_threads(1)
    if DEVICE == "cuda":
        # Each operation's deterministic algorithm, its products at the dtype asked: a draw repeats exactly.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(mode=True)
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    family, _, kind = sys.argv[1].removeprefix("multimodal-").partition("-")
    serve(lambda cases: run(family, kind == "text", cases))
