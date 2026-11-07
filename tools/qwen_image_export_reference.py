#!/usr/bin/env python3
"""Diffusers reading what Dew wrote: Qwen-Image 2.1's own classes over a Dew export, run on drawn inputs in
float32 and float64, the reference arm of tools/venv_reference.py's `qwen-image-export` site (its protocol
tools/reference_worker.py).

Qwen-Image 2.1's classes exist only in Diffusers at 6256aa7666cedd47443adc8f82da9a10e110b09c (with
transformers 5.17.0), which the Dew environment does not install, so its reading of a Dew export is recorded
rather than run in the suite. Two halves, in two environments:

- The Dew environment: the committed tiny Qwen-Image 2.1 pipeline (tests/fixtures/qwen_image_source.tar.xz)
  loaded with `Pretrained.load`, every transformer parameter moved by a seeded draw (a stand-in for training
  that every machine reproduces bit for bit), and saved with `Pretrained.save` (`perturbed_export`), which
  tests/test_qwen_image_source.py reruns and whose files' SHA-256 (tests/fixtures/qwen_image_export.json)
  it holds; the recording writes it to EXPORT.
- The worker (the isolated Qwen-Image environment, tools/diffusers_qwen_image_reference.py's):
  `QwenImage21Pipeline` loads EXPORT whole, each weighted component loads with `output_loading_info` and
  has to report nothing, and the transformer runs the pipeline's own call (`transformer_call`) on each
  draw's `packed`, `context` and `times` in float32 and in float64 (`torch_precision.float64`), its
  `output` packed.

The source's own float32 sinusoid table is kept as published: the float64 run builds its own, so check 1
measures it as rounding.

    QWEN=~/.cache/dew/reference-venvs/qwen-image/bin/python
    python tools/venv_reference.py qwen-image-export $QWEN
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "tests" / "fixtures" / "qwen_image_source.tar.xz"
GRID = (4, 6)
PROMPT = 5
SEED = 89
EXPORT = Path(tempfile.gettempdir()) / "dew-qwen-image-export" / "export"
"""Where the recording's Dew half writes the export its worker reads."""


def perturbed_variables(variables, seed: int = SEED):
    """Every transformer parameter moved by 0.05 of a seeded normal draw, in
    float32, leaf by leaf in tree order."""
    import jax

    rng = np.random.default_rng(seed)
    params = jax.tree.map(
        lambda leaf: (np.asarray(leaf, np.float32)
                      + np.float32(0.05) * rng.standard_normal(np.shape(leaf)).astype(np.float32)),
        variables["params"])
    return {**variables, "params": params}


def perturbed_export(destination: Path, scratch: Path):
    """The pipeline loaded from the archive into `scratch`, its transformer
    perturbed and saved to `destination`; returns the bundle and the
    variables it saved."""
    from dew.interop.pretrained import Pretrained

    with tarfile.open(ARCHIVE) as archive:
        archive.extractall(scratch, filter="data")
    loaded = Pretrained.load(str(scratch / "pipeline"), dtype="float32", attention_impl="xla")
    variables = perturbed_variables(loaded.variables)
    loaded.save(destination, variables=variables)
    return loaded, variables


def digests(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(directory.rglob("*")) if path.is_file()}


def checked(export: Path):
    """The export loaded whole by `QwenImage21Pipeline`, each weighted component loading with nothing in its
    loading report; returns its transformer in float32 and float64."""
    import torch
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel

    from tools.torch_precision import float64

    index = json.loads((export / "model_index.json").read_text())
    for name, entry in index.items():
        weighted = isinstance(entry, list) and entry[0] is not None
        if not (weighted and any((export / name).glob("*.safetensors"))):
            continue
        module = __import__(entry[0], fromlist=[entry[1]])
        loaded = getattr(module, entry[1]).from_pretrained(str(export), subfolder=name, local_files_only=True,
                                                           output_loading_info=True)
        problems = {key: value for key, value in loaded[1].items() if value}
        if problems:
            raise SystemExit(f"{name} loads with {problems}")
    QwenImage21Pipeline.from_pretrained(str(export), torch_dtype=torch.float32, local_files_only=True)

    def transformer(dtype):
        return QwenImage21Transformer2DModel.from_pretrained(
            str(export), subfolder="transformer", torch_dtype=dtype, local_files_only=True).eval()

    narrow = transformer(torch.float32)
    with float64():
        wide = transformer(torch.float64)
    return narrow, wide


def run(models, cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's output under the export's transformer, in float32 and float64."""
    import torch
    from diffusers_qwen_image_reference import transformer_call

    from tools.torch_precision import float64

    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        for kind, model, dtype in (("reference", models[0], torch.float32),
                                   ("truth", models[1], torch.float64)):
            given = [torch.from_numpy(cases[f"{name}.{key}"]).to(dtype)
                     for key in ("packed", "context", "times")]
            with float64() if dtype == torch.float64 else contextlib.nullcontext(), torch.no_grad():
                out[f"{name}.{kind}.output"] = transformer_call(model, *given, GRID).numpy()
    return out


if __name__ == "__main__":
    if sys.argv[1:] != ["qwen-image-export"]:
        raise SystemExit(__doc__)
    sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
    from reference_worker import serve

    models = checked(EXPORT)
    serve(lambda cases: run(models, cases))
