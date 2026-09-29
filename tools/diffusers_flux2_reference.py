"""Actual Diffusers 0.40.0 FLUX.2 objects, saved and walked, for tests/fixtures.

Tiny `Flux2Transformer2DModel` instances are constructed and saved with
`save_pretrained` - their real config and their real safetensors - and each
is run the way `Flux2Pipeline` runs it: the latent flattened one token per
position, the four-axis ids the pipeline lays out for the text and the grid,
the timestep it divides by a thousand, and the distilled guidance. Every case
records the forward, the vector-Jacobian products against a fixed cotangent
for the latent and the text states, and the gradient of every parameter, in
float32, with the source's sinusoidal frequency table rounded from float64
as `tools/diffusers_flux_reference.py` records Flux's (torch's float32 `exp`
is one unit in the last place off at some entries, and one ulp of a frequency
is one ulp of the 3500-radian angle a distilled guidance embeds).

Every parameter is moved off its initialization, so the RMS norms' scales
are not all ones.

Run in the isolated reference environment (diffusers 0.40.0, transformers
5.17.0, torch 2.8.0, CPU):

    python tools/diffusers_flux2_reference.py OUTPUT_DIR
    python tools/diffusers_flux2_reference.py bundle OUTPUT_DIR tests/fixtures/flux2_source.tar.xz
"""

from __future__ import annotations

import contextlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

DIFFUSERS = "0.40.0"
BASE = dict(patch_size=1, in_channels=8, num_layers=2, num_single_layers=2, attention_head_dim=16,
            num_attention_heads=2, joint_attention_dim=12, timestep_guidance_channels=32, mlp_ratio=3.0,
            axes_dims_rope=(4, 4, 4, 4), rope_theta=2000, eps=1e-6, guidance_embeds=True)
TOKENS = 5
SEED = 29


@dataclass(frozen=True)
class Case:
    """One transformer: the config controls that differ from the tiny base and
    the latent grid the walk runs at."""

    config: dict = field(default_factory=dict)
    grid: tuple[int, int] = (4, 4)
    guidance: tuple[float, ...] = (4.0, 2.5)


CASES: dict[str, Case] = {
    "dev": Case(),
    "rect": Case(grid=(3, 5)),
    # FLUX.2 [klein]'s base models embed no guidance.
    "unguided": Case(dict(guidance_embeds=False, num_layers=1, num_single_layers=3)),
    "narrow": Case(dict(mlp_ratio=2.0, eps=1e-3, out_channels=4, rope_theta=10000), grid=(2, 6)),
}


def rounded_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False,
                               downscale_freq_shift=1, scale=1, max_period=10000):
    """The source's `get_timestep_embedding` with its exponential taken in
    float64 and rounded once to float32, the table Dew builds on the host."""
    half = embedding_dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32)
    table = torch.exp((exponent / (half - downscale_freq_shift)).double()).float()
    angle = scale * (timesteps[:, None].float() * table[None, :])
    embedded = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
    if flip_sin_to_cos:
        embedded = torch.cat([embedded[:, half:], embedded[:, :half]], dim=-1)
    return embedded


@contextlib.contextmanager
def rounded_frequency_table():
    from diffusers.models import embeddings

    original = embeddings.get_timestep_embedding
    embeddings.get_timestep_embedding = rounded_timestep_embedding
    try:
        yield
    finally:
        embeddings.get_timestep_embedding = original


def ids(rows: int, columns: int) -> tuple[torch.Tensor, torch.Tensor]:
    """`Flux2Pipeline._prepare_text_ids` and `_prepare_latent_ids`."""
    text = torch.cartesian_prod(torch.arange(1), torch.arange(1), torch.arange(1), torch.arange(TOKENS))
    image = torch.cartesian_prod(torch.arange(1), torch.arange(rows), torch.arange(columns), torch.arange(1))
    return text, image


def build(name: str, case: Case, root: Path) -> dict[str, np.ndarray]:
    from diffusers import Flux2Transformer2DModel

    config = {**BASE, **case.config}
    torch.manual_seed(SEED)
    model = Flux2Transformer2DModel(**config).eval()
    generator = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
    model.save_pretrained(root / name / "transformer", safe_serialization=True)
    rows, columns = case.grid
    batch = len(case.guidance)
    latent = torch.randn((batch, rows * columns, config["in_channels"]), generator=generator, requires_grad=True)
    context = torch.randn((batch, TOKENS, config["joint_attention_dim"]), generator=generator, requires_grad=True)
    times = torch.tensor([731.0, 42.0])
    guidance = torch.tensor(case.guidance) if config["guidance_embeds"] else None
    text_ids, image_ids = ids(rows, columns)
    probe = torch.randn((batch, rows * columns, config.get("out_channels") or config["in_channels"]),
                        generator=generator)
    named = list(model.named_parameters())
    with rounded_frequency_table():
        output = model(hidden_states=latent, encoder_hidden_states=context, timestep=times / 1000,
                       guidance=guidance, img_ids=image_ids, txt_ids=text_ids, return_dict=False)[0]
        grads = torch.autograd.grad((output * probe).sum(), [latent, context] + [value for _, value in named])
    arrays = {"latent": latent.detach().numpy(), "context": context.detach().numpy(), "times": times.numpy(),
              "guidance": np.zeros(0, np.float32) if guidance is None else guidance.numpy(),
              "output": output.detach().numpy(), "probe": probe.numpy(),
              "grad_latent": grads[0].numpy(), "grad_context": grads[1].numpy()}
    for (key, _), gradient in zip(named, grads[2:], strict=True):
        arrays[f"grad_param.{key}"] = gradient.numpy()
    print(f"{name}: grid {case.grid} tokens {TOKENS} |output| <= {float(output.abs().max()):.4g} "
          f"parameters {len(named)}")
    return arrays


def bundle(directory: str, destination: str) -> None:
    """Pack the saved transformers and the recorded arrays for the suite."""
    import tarfile

    root = Path(directory)
    with tarfile.open(destination, "w:xz") as archive:
        for path in sorted(root.iterdir()):
            archive.add(path, arcname=path.name)
    print(f"{destination}: {Path(destination).stat().st_size / 1e6:.2f} MB")


def main(destination: str) -> None:
    import diffusers

    if diffusers.__version__ != DIFFUSERS:
        raise RuntimeError(f"recorded against diffusers {DIFFUSERS}, not {diffusers.__version__}")
    torch.set_num_threads(2)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for name, case in CASES.items():
        arrays.update({f"{name}.{key}": value for key, value in build(name, case, root).items()})
    record = {"diffusers": DIFFUSERS, "base": BASE, "tokens": TOKENS, "seed": SEED,
              "cases": {name: {"config": {**BASE, **case.config}, "grid": list(case.grid),
                               "guidance": list(case.guidance)} for name, case in CASES.items()}}
    np.savez_compressed(root / "flux2_transformer.npz", **arrays)
    (root / "flux2_transformer.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB, {len(CASES)} cases")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "bundle":
        bundle(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dew-flux2-reference")
