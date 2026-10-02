"""Actual Diffusers 0.34.0 Wan 2.1 objects, saved and walked, for tests/fixtures.

The transformer half constructs tiny `WanTransformer3DModel` instances with
every parameter moved off its initialization, saves each with
`save_pretrained`, and calls each as `WanPipeline` calls it: latents
`[B, C, F, H, W]`, the timestep, and the prompt states. Against a fixed
cotangent it records the gradients of the latents, the prompt states and
every parameter. Each call runs twice over the same float32 weights: in
float32, the reference, and in float64 (`widened`), the truth
tests/reference_error.py measures both runs from. The weights are rounded to
bfloat16-representable values so the fixture compresses.

Run with the Dew test environment's diffusers 0.34.0 and torch, on CPU:

    python tools/diffusers_wan_reference.py transformer OUTPUT_DIR
    python tools/diffusers_wan_reference.py bundle OUTPUT_DIR tests/fixtures/wan_transformer.tar.xz
"""

from __future__ import annotations

import contextlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

DIFFUSERS = "0.34.0"
SOURCE = Path(__file__).resolve().parents[1] / "tests/fixtures/hf/wan-source"
BASE = {"patch_size": [1, 2, 2], "num_attention_heads": 2, "attention_head_dim": 12, "in_channels": 4,
        "out_channels": 4, "text_dim": 20, "freq_dim": 16, "ffn_dim": 40, "num_layers": 2,
        "cross_attn_norm": True, "qk_norm": "rms_norm_across_heads", "eps": 1e-6, "rope_max_seq_len": 32}
SEED = 43


@dataclass(frozen=True)
class Case:
    """One transformer: the config controls that differ from the tiny base,
    the latent's (frames, height, width), the prompt length and each row's
    timestep."""

    config: dict = field(default_factory=dict)
    size: tuple[int, int, int] = (3, 6, 8)
    length: int = 7
    times: tuple[float, ...] = (250.0, 612.5)


CASES: dict[str, Case] = {
    # The published controls: one frame per patch, norms on.
    "published": Case(),
    # A head of 20 splits its pairs 4/3/3 rather than 2/2/2; two frames per
    # patch; neither norm; three layers at a coarse eps.
    "variant": Case({"patch_size": [2, 2, 2], "attention_head_dim": 20, "qk_norm": None,
                     "cross_attn_norm": False, "num_layers": 3, "eps": 1e-3, "out_channels": 6},
                    size=(4, 4, 6), length=5, times=(999.0,)),
}


@contextlib.contextmanager
def widened():
    """The reference in float64: every float32 pin of the Wan modules
    widened, which are their `.float()` and `.to(torch.float32)` casts and
    the sinusoids' float32 `arange`, and float64 the default dtype."""
    float_, to, arange = torch.Tensor.float, torch.Tensor.to, torch.arange

    def wide(dtype):
        return torch.float64 if dtype == torch.float32 else dtype

    def widened_to(self, *args, **kwargs):
        if "dtype" in kwargs:
            kwargs["dtype"] = wide(kwargs["dtype"])
        return to(self, *(wide(arg) if isinstance(arg, torch.dtype) else arg for arg in args), **kwargs)

    torch.Tensor.float = lambda self, *args, **kwargs: self.double()
    torch.Tensor.to = widened_to
    torch.arange = lambda *args, dtype=None, **kwargs: arange(*args, dtype=wide(dtype), **kwargs)
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        yield
    finally:
        torch.Tensor.float, torch.Tensor.to, torch.arange = float_, to, arange
        torch.set_default_dtype(default)


def walk(model, inputs: dict[str, torch.Tensor], probe: torch.Tensor, dtype) -> dict[str, np.ndarray]:
    """One call at `dtype` and the gradients of `sum(output * probe)` with
    respect to the latents, the prompt states and every parameter."""
    model = model.to(dtype)
    latents, context = (inputs[name].to(dtype).requires_grad_() for name in ("latents", "context"))
    named = list(model.named_parameters())
    output = model(latents, inputs["times"].to(dtype), context, return_dict=False)[0]
    grads = torch.autograd.grad((output * probe.to(dtype)).sum(),
                                [latents, context] + [value for _, value in named])
    arrays = {"output": output, "grad_latents": grads[0], "grad_context": grads[1]}
    for (key, _), gradient in zip(named, grads[2:], strict=True):
        arrays[f"grad_param.{key}"] = gradient
    return {key: value.detach().numpy() for key, value in arrays.items()}


def build(name: str, case: Case, root: Path) -> dict[str, np.ndarray]:
    from diffusers import WanTransformer3DModel

    config = {**BASE, **case.config}
    torch.manual_seed(SEED)
    model = WanTransformer3DModel(**config).eval()
    generator = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
            parameter.copy_(parameter.to(torch.bfloat16).float())
    model.save_pretrained(root / name / "transformer", safe_serialization=True)
    batch = len(case.times)
    inputs = {"latents": torch.randn((batch, config["in_channels"], *case.size), generator=generator),
              "context": torch.randn((batch, case.length, config["text_dim"]), generator=generator),
              "times": torch.tensor(case.times)}
    with torch.no_grad():
        shape = model(inputs["latents"], inputs["times"], inputs["context"], return_dict=False)[0].shape
    probe = torch.randn(shape, generator=generator)
    arrays = {key: value.numpy() for key, value in inputs.items()}
    arrays["probe"] = probe.numpy()
    arrays.update({f"fp32.{key}": value for key, value in walk(model, inputs, probe, torch.float32).items()})
    with widened():
        widest = walk(model, inputs, probe, torch.float64)
        arrays.update({f"fp64.{key}": value for key, value in widest.items()})
    gap = np.abs(arrays["fp32.output"] - arrays["fp64.output"]).max()
    print(f"{name}: latent {case.size} prompt {case.length} "
          f"|output| <= {np.abs(arrays['fp64.output']).max():.4g}, fp32 off float64 by {gap:.3g}")
    return arrays


def transformer(destination: str) -> None:
    import diffusers

    if diffusers.__version__ != DIFFUSERS:
        raise RuntimeError(f"recorded against diffusers {DIFFUSERS}, not {diffusers.__version__}")
    torch.set_num_threads(2)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, case in CASES.items():
        arrays.update({f"{name}.{key}": value for key, value in build(name, case, root).items()})
    record = {"diffusers": DIFFUSERS, "torch": torch.__version__, "base": BASE, "seed": SEED,
              "cases": {name: {"config": {**BASE, **case.config}, "size": list(case.size),
                               "length": case.length, "times": list(case.times)}
                        for name, case in CASES.items()}}
    np.savez_compressed(root / "wan_transformer.npz", **arrays)
    (root / "wan_transformer.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB, {len(CASES)} cases")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "bundle":
        from diffusers_dc_ae_reference import bundle

        bundle(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] == "transformer":
        transformer(sys.argv[2])
    else:
        raise SystemExit(__doc__)
