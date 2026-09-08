"""Actual Diffusers 0.34.0 Flux objects, saved and walked, for tests/fixtures.

Tiny `FluxTransformer2DModel` instances are constructed and saved with
`save_pretrained` - their real config and their real safetensors - and each
is then run on packed latents the way its pipeline runs it: the ids the
pipeline lays out, the timestep it divides by a thousand, and the distilled
guidance a guidance-embedded checkpoint takes. Every case records the
forward, the vector-Jacobian products against a fixed cotangent for the
latent, the text tokens and the pooled vector, and the gradient of every
parameter, in float32.

The variants are the ones whose wiring differs: the distilled checkpoint's
guidance embedder against the schnell-style model without one, a rectangular
packed grid, and a deeper single-stream stack.

Run in the isolated reference environment on CPU:

    PYTHONPATH=src python tools/diffusers_flux_reference.py OUTPUT_DIR
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import torch
import transformers.utils as transformers_utils

for _name, _value in (("FLAX_WEIGHTS_NAME", "flax_model.msgpack"),
                      ("WEIGHTS_INDEX_NAME", "pytorch_model.bin.index.json")):
    if not hasattr(transformers_utils, _name):
        setattr(transformers_utils, _name, _value)

BASE = dict(patch_size=1, in_channels=16, num_layers=2, num_single_layers=2,
            attention_head_dim=12, num_attention_heads=2, joint_attention_dim=16,
            pooled_projection_dim=10, guidance_embeds=False, axes_dims_rope=(4, 4, 4))
TOKENS = 5
SEED = 23


@dataclass(frozen=True)
class Case:
    """One transformer: the config controls that differ from the tiny base and
    the packed latent grid the walk runs at."""

    config: dict = field(default_factory=dict)
    grid: tuple[int, int] = (4, 4)
    batch: int = 2


CASES: dict[str, Case] = {
    "schnell": Case(),
    "dev": Case(dict(guidance_embeds=True)),
    "rect": Case(dict(guidance_embeds=True), grid=(6, 2)),
    "deep": Case(dict(num_layers=1, num_single_layers=3), grid=(2, 6)),
}


def image_ids(rows: int, columns: int) -> torch.Tensor:
    """`FluxPipeline._prepare_latent_image_ids` over a packed grid."""
    ids = torch.zeros(rows, columns, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(rows)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(columns)[None, :]
    return ids.reshape(rows * columns, 3)


def build(name: str, case: Case, root: Path) -> dict[str, np.ndarray]:
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

    config = {**BASE, **case.config}
    torch.manual_seed(SEED)
    model = FluxTransformer2DModel(**config).eval()
    directory = root / name / "transformer"
    model.save_pretrained(directory, safe_serialization=True)
    generator = torch.Generator().manual_seed(SEED + 1)
    rows, columns = case.grid
    packed = torch.randn((case.batch, rows * columns, config["in_channels"]), generator=generator,
                         dtype=torch.float32, requires_grad=True)
    context = torch.randn((case.batch, TOKENS, config["joint_attention_dim"]),
                          generator=generator, dtype=torch.float32, requires_grad=True)
    pooled = torch.randn((case.batch, config["pooled_projection_dim"]), generator=generator,
                         dtype=torch.float32, requires_grad=True)
    # The pipeline divides the scheduler's timestep by the training count
    # before the call, and the model multiplies it back up.
    times = torch.tensor([0.731, 0.042][: case.batch], dtype=torch.float32)
    guidance = (torch.full((case.batch,), 3.5, dtype=torch.float32)
                if config["guidance_embeds"] else None)
    output = model(hidden_states=packed, encoder_hidden_states=context, pooled_projections=pooled,
                   timestep=times, guidance=guidance, txt_ids=torch.zeros(TOKENS, 3),
                   img_ids=image_ids(rows, columns), return_dict=False)[0]
    probe = torch.randn(output.shape, generator=generator, dtype=torch.float32)
    named = [(key, value) for key, value in model.named_parameters()]
    grads = torch.autograd.grad((output * probe).sum(), [packed, context, pooled]
                                + [value for _, value in named])
    arrays = {
        "config": np.asarray(json.dumps(json.loads((directory / "config.json").read_text()))),
        "packed": packed.detach().numpy(), "context": context.detach().numpy(),
        "pooled": pooled.detach().numpy(), "times": times.numpy(),
        "guidance": np.zeros(0, np.float32) if guidance is None else guidance.numpy(),
        "output": output.detach().numpy(), "probe": probe.numpy(),
        "grad_packed": grads[0].numpy(), "grad_context": grads[1].numpy(),
        "grad_pooled": grads[2].numpy(),
    }
    for (key, _), gradient in zip(named, grads[3:]):
        arrays[f"grad_param.{key}"] = gradient.numpy()
    print(f"{name}: packed {tuple(packed.shape)} grid {case.grid} tokens {TOKENS} "
          f"|output| <= {float(output.detach().abs().max()):.4g} parameters {len(named)}")
    return arrays


def bundle(directory: str, destination: str) -> None:
    """Pack the saved transformers and the recorded arrays for the suite."""
    import tarfile

    root = Path(directory)
    with tarfile.open(destination, "w:xz") as archive:
        for path in sorted(root.iterdir()):
            if path.name.endswith((".npz", ".json")) or path.is_dir():
                archive.add(path, arcname=path.name)
    print(f"{destination}: {Path(destination).stat().st_size / 1e6:.2f} MB")


def main(destination: str) -> None:
    import diffusers

    if diffusers.__version__ != "0.34.0":
        raise RuntimeError("Requires diffusers==0.34.0")
    torch.set_num_threads(2)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    record: dict[str, object] = {"diffusers": diffusers.__version__, "base": BASE,
                                 "tokens": TOKENS, "cases": {}}
    cases: dict[str, dict[str, object]] = record["cases"]  # type: ignore[assignment]
    for name, case in CASES.items():
        for key, value in build(name, case, root).items():
            arrays[f"{name}.{key}"] = value
        cases[name] = {"config": {**BASE, **case.config}, "grid": list(case.grid),
                       "batch": case.batch}
    np.savez_compressed(root / "flux_transformer.npz", allow_pickle=False, **arrays)
    (root / "flux_transformer.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB, {len(CASES)} cases")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "bundle":
        bundle(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dew-flux-reference")
