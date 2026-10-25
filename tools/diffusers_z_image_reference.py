"""Actual Diffusers 0.40.0 Z-Image objects, saved and walked, for tests/fixtures.

Two halves. The transformer half constructs tiny `ZImageTransformer2DModel`
instances with every parameter moved off its initialization, saves each with
`save_pretrained`, and calls each as `ZImagePipeline` calls it: a list of
latents `[C, 1, H, W]`, the time `1 - sigma`, and a list of prompt states
of each prompt's own length. It records the output and, against a fixed
cotangent, the gradients of the latents, the prompt states and every
parameter (the weights rounded to bfloat16-representable values so the
fixture compresses), unmodified, in float32, the reference, and in float64,
the truth (`torch_precision.float64`: the source's float32 pins
widened).

The pipeline half saves one tiny `ZImagePipeline` over the published
configs - a Qwen3 text encoder saved as the release stores it, the Flux VAE,
the transformer, the published scheduler config and a byte-level tokenizer
with the published chat template - and walks the unmodified call from fixed
latents, recording the prompt states, the latent it ends on and the image,
in float32 and, the saved pipeline loaded widened, in float64.

Run in the isolated reference environment (diffusers 0.40.0, transformers
5.17.0, torch 2.8.0, CPU):

    python tools/diffusers_z_image_reference.py OUTPUT_DIR
    python tools/diffusers_z_image_reference.py bundle OUTPUT_DIR tests/fixtures/z_image_source.tar.xz
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from diffusers_reference_helpers import perturb, recording, saved_pipeline, tiny_qwen3_encoder
from torch_precision import float64
from diffusers_wan_reference import float64_scheduler
from reference_releases import RELEASES

DIFFUSERS = RELEASES["diffusers-0.40"]["diffusers"]
SOURCE = Path(__file__).resolve().parents[1] / "tests/fixtures/hf/z-image-source"
# Every case records every parameter's gradient in float32 and in float64, so the
# width is as narrow as two heads with three rotary axes allow.
BASE = {"all_patch_size": [2], "all_f_patch_size": [1], "in_channels": 4, "dim": 32, "n_layers": 2,
        "n_refiner_layers": 1, "n_heads": 2, "n_kv_heads": 2, "norm_eps": 1e-5, "qk_norm": True,
        "cap_feat_dim": 24, "rope_theta": 256.0, "t_scale": 1000.0, "axes_dims": [4, 6, 6],
        "axes_lens": [160, 32, 32]}
SEED = 31


@dataclass(frozen=True)
class Case:
    """One transformer: the config controls that differ from the tiny base,
    the latent size, and each row's prompt length."""

    config: dict = field(default_factory=dict)
    size: tuple[int, int] = (8, 12)
    lengths: tuple[int, ...] = (7, 40)
    times: tuple[float, ...] = (0.25, 0.625)


CASES: dict[str, Case] = {
    # A row's prompt padded to 32 and the other to 64: the source pads the
    # batch to the longer, masked; the image's 24 patches pad to 32.
    "padded": Case(),
    # 32 patches and 32 prompt tokens: no padding anywhere.
    "exact": Case(size=(8, 16), lengths=(32,), times=(0.5,)),
    "deep": Case({"n_layers": 3, "n_refiner_layers": 2, "norm_eps": 1e-3, "rope_theta": 64.0}, size=(6, 10),
                 lengths=(33,), times=(0.875,)),
}


def build(name: str, case: Case, root: Path) -> dict[str, np.ndarray]:
    """The case's saved transformer and its walk in float32 and in float64
    (`fp64.`), the same weights and inputs widened."""
    from diffusers import ZImageTransformer2DModel

    config = {**BASE, **case.config}
    torch.manual_seed(SEED)
    model = ZImageTransformer2DModel(**config).eval()
    generator = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
            parameter.copy_(parameter.to(torch.bfloat16).float())
    directory = root / name / "transformer"
    model.save_pretrained(directory, safe_serialization=True)
    height, width = case.size
    batch = len(case.lengths)
    latents = torch.randn((batch, config["in_channels"], 1, height, width), generator=generator)
    captions = [torch.randn((length, config["cap_feat_dim"]), generator=generator) for length in case.lengths]
    times = torch.tensor(case.times)
    probe = torch.randn((batch, config["in_channels"], 1, height, width), generator=generator)

    def walk(model, dtype) -> dict[str, np.ndarray]:
        inputs = [tensor.detach().to(dtype).requires_grad_() for tensor in (latents, *captions)]
        named = list(model.named_parameters())
        output = torch.stack(model(list(inputs[0].unbind(0)), times.to(dtype), inputs[1:],
                                   return_dict=False)[0])
        grads = torch.autograd.grad((output * probe.to(dtype)).sum(), inputs + [value for _, value in named])
        return {"output": output.detach().numpy(), "grad_latents": grads[0].numpy(),
                **{f"grad_caption.{row}": grads[1 + row].numpy() for row in range(batch)},
                **{f"grad_param.{key}": gradient.numpy()
                   for (key, _), gradient in zip(named, grads[1 + batch:], strict=True)}}

    narrow = walk(model, torch.float32)
    with float64():
        from diffusers import ZImageTransformer2DModel

        wide = walk(ZImageTransformer2DModel.from_pretrained(directory, torch_dtype=torch.float64).eval(),
                    torch.float64)
    arrays = {"latents": latents.numpy(), "times": times.numpy(), "probe": probe.numpy(),
              **{f"caption.{row}": caption.numpy() for row, caption in enumerate(captions)},
              **narrow, **{f"fp64.{key}": value for key, value in wide.items()}}
    print(f"{name}: size {case.size} prompts {case.lengths} "
          f"|output| <= {float(np.abs(narrow['output']).max()):.4g} parameters {len(narrow) - 2 - batch}")
    return arrays


PIPELINE = {**BASE, "cap_feat_dim": 16, "n_layers": 1}
PIPELINE_VAE = {"block_out_channels": [8, 16], "down_block_types": ["DownEncoderBlock2D"] * 2,
                "up_block_types": ["UpDecoderBlock2D"] * 2, "layers_per_block": 1, "latent_channels": 4,
                "norm_num_groups": 4, "use_quant_conv": False, "use_post_quant_conv": False}
PROMPTS = ["a red cat on a mat", "tiny photo"]
HEIGHT, WIDTH, STEPS, GUIDANCE = 16, 24, 4, 5.0


def text_encoder(tok):
    """The published Qwen3 config, narrowed and three layers deep; a causal
    language model with tied embeddings, which saves as the release is
    stored (`model.` names, no head)."""
    return tiny_qwen3_encoder(SOURCE, tok, hidden_size=PIPELINE["cap_feat_dim"], layers=3, seed=SEED)


def build_pipeline(root: Path):
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        ZImagePipeline,
        ZImageTransformer2DModel,
    )
    from make_tiny_tokenizer import qwen_byte_tokenizer

    metadata = json.loads((SOURCE / "tokenizer" / "tokenizer_config.json").read_text())
    tok = qwen_byte_tokenizer(metadata, chat_template=metadata["chat_template"])
    encoder = text_encoder(tok)
    published = json.loads((SOURCE / "vae" / "config.json").read_text())
    torch.manual_seed(SEED + 7)
    vae = AutoencoderKL(**PIPELINE_VAE, scaling_factor=published["scaling_factor"],
                        shift_factor=published["shift_factor"]).eval()
    transformer = ZImageTransformer2DModel(**PIPELINE).eval()
    perturb(torch.Generator().manual_seed(SEED + 8), 0.05, vae, transformer)
    scheduler = json.loads((SOURCE / "scheduler" / "scheduler_config.json").read_text())
    pipe = ZImagePipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler), vae=vae,
                          text_encoder=encoder, tokenizer=tok, transformer=transformer)
    return saved_pipeline(pipe, root / "pipeline", HEIGHT, WIDTH)


def pipeline_record(root: Path) -> dict[str, np.ndarray]:
    """The prompt states each prompt (and the empty negative) encodes to, and
    the unmodified call's walk from fixed latents at `STEPS` steps, guided at
    its default scale: the latent it ends on and the image it decodes; in
    float32 and, the saved pipeline loaded widened, in float64 (`fp64.`)."""
    from diffusers import ZImagePipeline
    from diffusers.pipelines.z_image import pipeline_z_image
    from diffusers.schedulers import scheduling_flow_match_euler_discrete

    pipe = build_pipeline(root)
    generator = torch.Generator().manual_seed(SEED + 9)
    latents = torch.randn((len(PROMPTS), PIPELINE["in_channels"], HEIGHT // 2, WIDTH // 2), generator=generator)

    def walk(pipe, dtype) -> dict[str, np.ndarray]:
        arrays = {}
        with torch.no_grad():
            for row, prompt in enumerate([*PROMPTS, ""]):
                (states,) = pipe._encode_prompt(prompt=prompt, device=torch.device("cpu"))
                arrays[f"context.{row}"] = states.numpy()
            for row, prompt in enumerate(PROMPTS):
                call = {"prompt": prompt, "height": HEIGHT, "width": WIDTH, "num_inference_steps": STEPS,
                        "guidance_scale": GUIDANCE}
                start = latents[row:row + 1].to(dtype)
                walked = pipe(**call, latents=start.clone(), output_type="latent").images
                arrays[f"latents.{row}"] = walked.numpy()
                arrays[f"images.{row}"] = pipe(**call, latents=start.clone(), output_type="np").images
        return arrays

    narrow = walk(pipe, torch.float32)
    # The call keeps its latents in float32, which it names and asserts in its own module.
    with (float64(), float64_scheduler(pipeline_z_image),
          float64_scheduler(scheduling_flow_match_euler_discrete)):
        wide_pipe = ZImagePipeline.from_pretrained(root / "pipeline", torch_dtype=torch.float64,
                                                   local_files_only=True)
        wide_pipe.set_progress_bar_config(disable=True)
        wide = walk(wide_pipe, torch.float64)
    print(f"pipeline: latents {narrow['latents.0'].shape} images {narrow['images.0'].shape}")
    return {"pipeline.x_T": latents.numpy(), **{f"pipeline.{key}": value for key, value in narrow.items()},
            **{f"pipeline.fp64.{key}": value for key, value in wide.items()}}


def main(destination: str) -> None:
    root = recording(destination, "diffusers-0.40")
    arrays: dict[str, np.ndarray] = {}
    for name, case in CASES.items():
        arrays.update({f"{name}.{key}": value for key, value in build(name, case, root).items()})
    arrays.update(pipeline_record(root))
    record = {"diffusers": DIFFUSERS, "base": BASE, "seed": SEED,
              "cases": {name: {"config": {**BASE, **case.config}, "size": list(case.size),
                               "lengths": list(case.lengths), "times": list(case.times)}
                        for name, case in CASES.items()},
              "pipeline": {"config": PIPELINE, "vae": PIPELINE_VAE, "prompts": PROMPTS, "height": HEIGHT,
                           "width": WIDTH, "steps": STEPS, "guidance": GUIDANCE}}
    np.savez_compressed(root / "z_image.npz", **arrays)
    (root / "z_image.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB, {len(CASES)} cases")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "bundle":
        from diffusers_dc_ae_reference import bundle

        bundle(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dew-z-image-reference")
