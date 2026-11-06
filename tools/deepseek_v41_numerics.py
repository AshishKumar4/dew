"""What tests/test_deepseek_v41.py shares about DeepSeek-V4.1-Flash's tiny fixture
(tests/fixtures/hf/deepseek-v41-tiny, tools/deepseek_v41_reference.py): the quantizers as the release's
kernels compute them (`QUANTIZERS`), the model with them off (`unquantized`), the fixture with its vision
half as the release ships a bundle (`vision_bundle`), and the fixture over the weights the release holds,
as it runs them (`as_released`, row 354).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path

from flax.core import freeze

from dew.nn.fake_quant import fake_quant_fp4, fake_quant_fp8

TINY = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf" / "deepseek-v41-tiny"


# The release's three quantizer sites: FP8 over 32 channels under
# power-of-two scales, FP4 over 16 under E4M3 scales, FP4 over 32 under
# power-of-two scales.
QUANTIZERS = {"window": lambda v: fake_quant_fp8(v, 32),
              "entries": lambda v: fake_quant_fp4(v, 16, e4m3_scale=True),
              "index": lambda v: fake_quant_fp4(v, 32, e4m3_scale=False)}


def unquantized(model):
    """The model with its quantization-aware rounding off, as the reference's
    architecture outputs were taken."""
    kinds = {name: dataclasses.replace(kind, mixer=dataclasses.replace(kind.mixer, kv_qat=False))
             for name, kind in model.kinds.items()}
    return model.clone(kinds=freeze(kinds))


def vision_bundle(directory: Path) -> Path:
    """The fixture with its vision half in `directory` as the release ships a
    bundle: vision_config inside config.json, one safetensors file."""
    from safetensors.numpy import load_file

    from dew.interop.safetensors_io import write_file

    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        shutil.copy(TINY / name, directory / name)
    config = json.loads((TINY / "config.json").read_text())
    config["vision_config"] = json.loads((TINY / "vision_config.json").read_text())
    (directory / "config.json").write_text(json.dumps(config))
    write_file({**load_file(TINY / "model.safetensors"), **load_file(TINY / "vision.safetensors")},
               directory / "model.safetensors", {"format": "pt"})
    return directory


def as_released(directory: Path, *, vision: bool = False) -> Path:
    """The fixture in `directory` with the weights the release holds (release.safetensors: bf16 but its
    head, norms and mixes, as tools/deepseek_v41_reference.py's `--release` keeps them), its vision half
    with them where `vision` (vision_config inside config.json, as `vision_bundle`): loaded with
    `param_dtype="auto"`, the parameters the release computes with (row 354)."""
    from dew.interop.safetensors_io import read_file, write_file

    for name in ("tokenizer.json", "tokenizer_config.json", "generation_config.json"):
        shutil.copy(TINY / name, directory / name)
    config = json.loads((TINY / "config.json").read_text())
    if vision:
        config["vision_config"] = json.loads((TINY / "vision_config.json").read_text())
    (directory / "config.json").write_text(json.dumps(config))
    held, _ = read_file(TINY / "release.safetensors")
    seeing, _ = read_file(TINY / "vision.safetensors")
    write_file({name: value for name, value in held.items() if vision or name not in seeing},
               directory / "model.safetensors", {"format": "pt"})
    return directory
