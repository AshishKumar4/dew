"""DiffusionGemma's denoiser under transformers (tests/fixtures/hf/diffusion-gemma-workflow, its shared text
tree with routed experts, a sliding and a full layer, self-conditioning and the vision conditioner) run on
drawn prompts, canvases, conditioning logits and pictures, in bfloat16 and float64: the reference arm of
tools/venv_reference.py's `diffusion-gemma-bf16` site (its protocol tools/reference_worker.py), run in an
environment of transformers 5.16.1 and torch, on the CPU.

    python tools/diffusion_gemma_worker.py diffusion-gemma-bf16    `IN OUT` lines on stdin, each `done`

IN holds each draw's `prompt` and `image_prompt` ids `[B, P]`, `canvas` ids `[B, C]`, `previous` logits
`[B, C, V]` and `pixels` `[B, 1, 3, 4, 4]`. OUT gets `<draw>.reference.<leaf>` and `<draw>.truth.<leaf>`:
the checkpoint read in bfloat16 (its weights and activations, stored as float32, which holds every bfloat16)
and in float64 (every float32 pin widened, `torch_precision.float64`), each of the denoiser's three forwards
(`workflow_outputs`): the canvas over the prompt (`bare`), with the previous step's logits
(`conditioned`), and over the image prompt with its pictures (`image_logits`).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from reference_worker import serve

WORKFLOW = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf" / "diffusion-gemma-workflow"


def eager(model):
    """Eager attention, as tools/hf_reference.py ran it, and the per-expert loop, the one routed kernel that
    runs in float64 on CPU (grouped_mm takes no double), so every precision runs one order of operations."""
    model.set_attn_implementation("eager")
    model.set_experts_implementation("eager")
    return model.eval()


def media_inputs(pixels) -> dict[str, torch.Tensor]:
    """The image inputs DiffusionGemma's forward reads for a prompt's `pixels` ([batch, 1, 3, 4, 4], torch or
    NumPy): each image's Gemma 4 patches at patch size 2 (`convert_image_to_patches`), in the pixels' dtype,
    and the patches' places on their 2 by 2 grid."""
    from transformers.models.gemma4.image_processing_pil_gemma4 import convert_image_to_patches

    pixels = torch.as_tensor(pixels)
    # Patched in float64, which holds every narrower float exactly (NumPy holds no bfloat16), then cast back.
    patches = torch.stack([torch.as_tensor(convert_image_to_patches(image.to(torch.float64).numpy(), 2))
                           for image in pixels[:, 0]])
    positions = torch.tensor([[[0, 0], [1, 0], [0, 1], [1, 1]]]).expand(len(pixels), -1, -1)
    return {"pixel_values": patches.to(pixels.dtype), "image_position_ids": positions}


def workflow(dtype: torch.dtype):
    """The workflow checkpoint read in `dtype`."""
    from transformers.models.diffusion_gemma.modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion

    model = DiffusionGemmaForBlockDiffusion.from_pretrained(WORKFLOW, dtype=dtype, local_files_only=True)
    return eager(model)


def workflow_outputs(model, inputs: dict[str, np.ndarray], dtype: torch.dtype) -> dict[str, np.ndarray]:
    """The denoiser's three forwards over `inputs`, in float64."""
    prompt, canvas = (torch.from_numpy(np.asarray(inputs[key], np.int64)) for key in ("prompt", "canvas"))
    previous = torch.from_numpy(inputs["previous"]).to(dtype)
    media = media_inputs(torch.from_numpy(inputs["pixels"]).to(dtype))
    image_prompt = torch.from_numpy(np.asarray(inputs["image_prompt"], np.int64))
    with torch.no_grad():
        outputs = {
            "bare": model(input_ids=prompt, decoder_input_ids=canvas).logits,
            "conditioned": model(input_ids=prompt, decoder_input_ids=canvas,
                                 self_conditioning_logits=previous).logits,
            "image_logits": model(input_ids=image_prompt, decoder_input_ids=canvas, **media).logits,
        }
    return {key: value.to(torch.float64).numpy() for key, value in outputs.items()}


def run(cases: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Every draw's three forwards, in bfloat16 and float64."""
    from torch_precision import float64

    narrow = workflow(torch.bfloat16)
    with float64():
        wide = workflow(torch.float64)
    out = {}
    for name in sorted({key.split(".", 1)[0] for key in cases}):
        inputs = {key.removeprefix(f"{name}."): value for key, value in cases.items()
                  if key.startswith(f"{name}.")}
        for leaf, value in workflow_outputs(narrow, inputs, torch.bfloat16).items():
            out[f"{name}.reference.{leaf}"] = value.astype(np.float32)
        with float64():
            for leaf, value in workflow_outputs(wide, inputs, torch.float64).items():
                out[f"{name}.truth.{leaf}"] = value
    return out


if __name__ == "__main__":
    if sys.argv[1:] != ["diffusion-gemma-bf16"]:
        raise SystemExit(__doc__)
    torch.set_num_threads(1)
    serve(run)
