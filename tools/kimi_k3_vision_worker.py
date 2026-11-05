"""Kimi K3's vision tower and projector as its release computes them (moonshotai/Kimi-K3 at f831ab6,
modeling_kimi_k3.py: MoonViT3dPretrainedModel and PatchMergerMLPV2), over drawn media, in float32 and as
its float64 twin, on the CPU: the reference arm of tools/venv_reference.py's `kimi-k3-vision` site, run in
an environment holding the release's transformers (4.56.2) and torch (2.8.0, CPU).

    python tools/kimi_k3_vision_worker.py SITE    `IN OUT` lines on stdin, each answered `done`

IN holds cases `<case>.<kind>`, each kind's patches `[t*h*w, 3, 14, 14]` of its grid in
tests/test_kimi_k3_vision.py's `GRIDS`; OUT gets `<case>.reference.<kind>` and `<case>.truth.<kind>`, the
item's projected merge blocks `[h*w/4 * (t if an image), text width]`, the tower's frames pooled
(`sd2_tpool`). The release's vision definitions run as they are written, selected from the file without
the language model it imports (`KimiLinearForCausalLM`, whose kernels need fla on a CUDA device), its
attention eager (`eager_attention`, the vision config's `_attn_implementation`). Two changes, both its
precision policy: the position table's interpolation (`get_rope_shape`) runs eagerly, not through
`torch.compile`; and the float64 twin, built and run inside `torch_precision.float64`, reads its rotary
table as the complex128 its widened arange makes, where the release asserts complex64
(`_apply_rope_input_validation`), its attention's `torch.softmax(..., dtype=torch.float32)` in float64,
and its RMS norms at the epsilon the float32 run's take (`RMSNorm(eps=None)`, float32's, which a float64
input would take as float64's), so the twin computes the float32 run's function.
"""

from __future__ import annotations

import ast
import collections.abc
import contextlib
import copy
import json
import math
import sys
import types
import typing
from pathlib import Path

import numpy as np
import torch
from reference_source import ran, selected
from reference_worker import serve
from torch_precision import dtype_widened, float64, replaced

REPO, REVISION = "moonshotai/Kimi-K3", "f831ab66814297da540d832a5235f8e904f29d06"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf" / "kimi-k3-tiny"
GRIDS = {"image": (1, 4, 6), "video": (2, 4, 4)}
"""Each drawn item's `(t, h, w)` grid in 14-pixel patches, as tests/test_kimi_k3_vision.py draws it."""
VISION = {"eager_attention", "multihead_attention", "VL_VISION_ATTENTION_FUNCTIONS",
          "_apply_rope_input_validation", "get_rope_shape_decorate", "get_rope_shape", "apply_rope",
          "get_1d_sincos_pos_embed_from_grid", "get_1d_sincos_pos_embed",
          "Learnable2DInterpPosEmbDivided_fixed", "MoonVision3dPatchEmbed", "Rope2DPosEmbRepeated", "MLP2",
          "MoonViTEncoderLayer", "MoonViT3dEncoder", "tpool_patch_merger", "MoonViT3dPretrainedModel",
          "PatchMergerMLPV2", "VisionTowerConfig", "ProjectorConfig"}
"""The release's vision definitions, every one the tower and the projector reach."""


def released() -> types.ModuleType:
    """The release's vision definitions (`VISION`) as one module, under the names its file imports them
    by, the position table's interpolation eager."""
    from huggingface_hub import hf_hub_download
    from transformers import activations
    from transformers.configuration_utils import PretrainedConfig
    from transformers.modeling_utils import PreTrainedModel

    path = Path(hf_hub_download(REPO, "modeling_kimi_k3.py", revision=REVISION))
    statements = selected(path.read_text(), VISION, filename=str(path))
    for node in statements:
        if isinstance(node, ast.FunctionDef) and node.name == "get_rope_shape":
            node.decorator_list = [decorator for decorator in node.decorator_list
                                   if "torch.compile" not in ast.unparse(decorator)]
    module = types.ModuleType("kimi_k3_vision")
    # The file's own imports (modeling_kimi_k3.py:25-56); its config class annotates alone here.
    module.__dict__.update(
        math=math, np=np, torch=torch, nn=torch.nn, F=torch.nn.functional, deepcopy=copy.deepcopy,
        Optional=typing.Optional, Sequence=collections.abc.Sequence, PretrainedConfig=PretrainedConfig,
        PreTrainedModel=PreTrainedModel, KimiK3Config=types.SimpleNamespace, flash_attn_varlen_func=None,
        PytorchGELUTanh=getattr(activations, "PytorchGELUTanh", None) or activations.GELUTanh)
    ran(statements, module.__dict__, str(path))
    return module


def models(release: types.ModuleType, dtype: torch.dtype) -> tuple[torch.nn.Module, torch.nn.Module]:
    """The fixture's tower and projector at `dtype`, as KimiK3ForConditionalGeneration builds them from
    its vision_config (modeling_kimi_k3.py:899-915), holding the fixture's tensors."""
    from safetensors.torch import load_file

    vision = types.SimpleNamespace(**json.loads((FIXTURE / "config.json").read_text())["vision_config"])
    tower = release.MoonViT3dPretrainedModel(release.VisionTowerConfig(vision))
    projector = release.PatchMergerMLPV2(release.ProjectorConfig(vision))
    tensors = load_file(str(FIXTURE / "model.safetensors"))
    for module, prefix in ((tower, "vision_tower."), (projector, "mm_projector.")):
        module.load_state_dict({name.removeprefix(prefix): value.to(dtype) for name, value in tensors.items()
                                if name.startswith(prefix)}, strict=True)
        module.to(dtype).eval()
    return tower, projector


@contextlib.contextmanager
def twin_policy(release: types.ModuleType):
    """The float64 twin's two changes to the release (the module docstring's): its complex128 rotary table
    read as the release reads its complex64 one, and its attention's float32 softmax pin widened."""
    check = release._apply_rope_input_validation

    def validated(x, freqs_cis):
        # The release's shape checks, over the table's complex64 rounding where the twin holds complex128.
        return check(x, freqs_cis.to(torch.complex64) if freqs_cis.dtype == torch.complex128 else freqs_cis)

    with replaced((release, "_apply_rope_input_validation", validated),
                  (torch, "softmax", dtype_widened(torch.softmax))):
        yield


def projected(tower, projector, patches: np.ndarray, grid: tuple[int, int, int]) -> np.ndarray:
    """One item's projected merge blocks."""
    dtype = next(tower.parameters()).dtype
    with torch.no_grad():
        features = tower(torch.tensor(patches, dtype=dtype), torch.tensor([grid]))
        return projector(features)[0].numpy()


def main() -> None:
    if sys.argv[1] != "kimi-k3-vision":
        raise SystemExit(f"no site {sys.argv[1]!r} here: kimi-k3-vision")
    release = released()
    narrow = models(release, torch.float32)
    with float64(), twin_policy(release):
        wide = models(release, torch.float64)
    # The float32 run's RMS norms' epsilon, which `eps=None` resolves by its input's dtype.
    for module in wide[0].modules():
        if isinstance(module, torch.nn.RMSNorm) and module.eps is None:
            module.eps = torch.finfo(torch.float32).eps

    def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for name in sorted({key.split(".", 1)[0] for key in cases}):
            for kind, grid in GRIDS.items():
                patches = cases[f"{name}.{kind}"]
                out[f"{name}.reference.{kind}"] = projected(*narrow, patches, grid)
                with float64(), twin_policy(release):
                    out[f"{name}.truth.{kind}"] = projected(*wide, patches, grid)
        return out

    serve(run)


if __name__ == "__main__":
    main()
