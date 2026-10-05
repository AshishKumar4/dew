#!/usr/bin/env python3
"""Write the Qwen-Image 2.1 export fixture: Diffusers reading what Dew wrote.

Qwen-Image 2.1's classes exist only in Diffusers at
6256aa7666cedd47443adc8f82da9a10e110b09c (with transformers 5.17.0), which
the Dew environment does not install, so its reading of a Dew export is
recorded rather than run in the suite. Two halves, in two environments:

- `export DIR` (the Dew environment): the committed tiny Qwen-Image 2.1
  pipeline (tests/fixtures/qwen_image_source.tar.xz) loaded with
  `Pretrained.load`, every transformer parameter moved by a seeded draw (a
  stand-in for training that every machine reproduces bit for bit), and
  saved with `Pretrained.save` into DIR/export. `perturbed_export` is what
  tests/test_qwen_image_source.py reruns.
- `consume DIR OUTPUT.npz` (the isolated Qwen-Image environment,
  tools/diffusers_qwen_image_reference.py's): `QwenImage21Pipeline` loads
  the export whole, each weighted component loads with
  `output_loading_info` and has to report nothing, and the transformer runs
  the pipeline's own call (`transformer_call`) on fixed inputs in float32
  and in float64 (`diffusers_wan_reference.float64`). The fixture holds the
  SHA-256 of every exported file, the inputs and both outputs.

The source's own float32 sinusoid table is kept as published: the float64
run builds its own, so the rule measures it as rounding.

    python tools/qwen_image_export_reference.py export DIR
    QWEN=~/.cache/dew/reference-venvs/qwen-image/bin/python
    $QWEN tools/qwen_image_export_reference.py consume DIR OUTPUT.npz
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "tests" / "fixtures" / "qwen_image_source.tar.xz"
GRID = (4, 6)
PROMPT = 5
SEED = 89


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


def inputs(config: dict) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(SEED + 1)
    rows, columns = GRID
    return {"packed": rng.standard_normal((1, rows * columns, config["in_channels"])).astype(np.float32),
            "context": rng.standard_normal((1, PROMPT, config["context_in_dim"])).astype(np.float32),
            "times": np.asarray([500.0], np.float32)}


def consume(directory: Path, output: Path) -> None:
    import torch
    from diffusers import QwenImage21Pipeline, QwenImage21Transformer2DModel

    sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
    from diffusers_qwen_image_reference import transformer_call

    from tools.diffusers_wan_reference import float64

    export = directory / "export"
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

    config = json.loads((export / "transformer" / "config.json").read_text())
    given = inputs(config)
    arrays = dict(given)
    for precision, dtype in (("fp32", torch.float32), ("fp64", torch.float64)):
        with float64() if dtype == torch.float64 else contextlib.nullcontext():
            model = QwenImage21Transformer2DModel.from_pretrained(
                str(export), subfolder="transformer", torch_dtype=dtype, local_files_only=True).eval()
            with torch.no_grad():
                out = transformer_call(model, torch.from_numpy(given["packed"]).to(dtype),
                                       torch.from_numpy(given["context"]).to(dtype),
                                       torch.from_numpy(given["times"]).to(dtype), GRID)
        arrays[f"{precision}.output"] = out.numpy()
    import diffusers
    import transformers

    meta = {"digests": digests(export), "grid": GRID, "diffusers": diffusers.__version__,
            "transformers": transformers.__version__, "torch": torch.__version__}
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    np.savez_compressed(output, **arrays)
    gap = float(np.abs(arrays["fp32.output"] - arrays["fp64.output"]).max())
    print(f"{output}: fp32 off float64 by {gap:.3g}, {len(meta['digests'])} files")


def main() -> None:
    if sys.argv[1:2] == ["export"] and len(sys.argv) == 3:
        import tempfile

        destination = Path(sys.argv[2]) / "export"
        with tempfile.TemporaryDirectory(dir=sys.argv[2]) as scratch:
            perturbed_export(destination, Path(scratch))
        print(f"{destination}: {len(digests(destination))} files")
    elif sys.argv[1:2] == ["consume"] and len(sys.argv) == 4:
        consume(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
