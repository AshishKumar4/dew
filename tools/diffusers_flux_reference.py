"""Actual Diffusers 0.34.0 Flux objects, saved and walked, for tests/fixtures.

Tiny `FluxTransformer2DModel` instances are constructed and saved with
`save_pretrained` - their real config and their real safetensors - and each
is then run on packed latents the way its pipeline runs it: the ids the
pipeline lays out, the timestep it divides by a thousand, and the distilled
guidance a guidance-embedded checkpoint takes. Every case records the
forward, the vector-Jacobian products against a drawn cotangent for the
latent, the text tokens and the pooled vector, and the gradient of every
parameter, unmodified, in float32, the reference, and in float64, the truth
(`diffusers_wan_reference.float64`: the source's float32 pins widened). The
pipeline half records the prompt states and the walk the same two ways.

The variants are the ones whose wiring differs: the distilled checkpoint's
guidance embedder against the schnell-style model without one, a rectangular
packed grid, and a deeper single-stream stack.

Run in the isolated reference environment on CPU:

    PYTHONPATH=src python tools/diffusers_flux_reference.py OUTPUT_DIR
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import torch
from diffusers_reference_helpers import restore_transformers_names
from diffusers_sd3_reference import bundle, clip_tokenizers, t5_tokenizer
from diffusers_wan_reference import float64
from recorded_references import provenance, record_case

BASE = {"patch_size": 1, "in_channels": 16, "num_layers": 2, "num_single_layers": 2,
        "attention_head_dim": 12, "num_attention_heads": 2, "joint_attention_dim": 16,
        "pooled_projection_dim": 10, "guidance_embeds": False, "axes_dims_rope": (4, 4, 4)}
TOKENS = 5
SEED = 23


@dataclass(frozen=True)
class Case:
    """One transformer: the config controls that differ from the tiny base and
    the packed latent grid the walk runs at."""

    config: dict = field(default_factory=dict)
    grid: tuple[int, int] = (4, 4)
    batch: int = 2
    guidance: tuple[float, ...] = (3.5, 3.5)


CASES: dict[str, Case] = {
    "schnell": Case(),
    "dev": Case({"guidance_embeds": True}),
    "rect": Case({"guidance_embeds": True}, grid=(6, 2)),
    "deep": Case({"num_layers": 1, "num_single_layers": 3}, grid=(2, 6)),
    # Each row walked at its own distilled guidance, which the source takes as
    # a per-row tensor rather than one scalar.
    "mixed": Case({"guidance_embeds": True}, guidance=(3.5, 7.0)),
}


def image_ids(rows: int, columns: int) -> torch.Tensor:
    """`FluxPipeline._prepare_latent_image_ids` over a packed grid."""
    ids = torch.zeros(rows, columns, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(rows)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(columns)[None, :]
    return ids.reshape(rows * columns, 3)


def build(name: str, case: Case, root: Path) -> dict[str, np.ndarray]:
    """The saved transformer and its float32 and float64 walks over recorded seeds.

    Seed zero keeps the original draw. Further seeds redraw standard-normal inputs and cotangents,
    with unit-normal guidance perturbations about the case's per-row guidance. Weights stay fixed.
    """
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
    from test_flux_source import flux_walk

    config = {**BASE, **case.config}
    torch.manual_seed(SEED)
    model = FluxTransformer2DModel(**config).eval()
    directory = root / name / "transformer"
    model.save_pretrained(directory, safe_serialization=True)
    rows, columns = case.grid
    shape = (case.batch, rows * columns, config["in_channels"])

    def draw(seed):
        generator = torch.Generator().manual_seed(SEED + 1 + seed)
        inputs = {
            "packed": torch.randn(shape, generator=generator),
            "context": torch.randn((case.batch, TOKENS, config["joint_attention_dim"]), generator=generator),
            "pooled": torch.randn((case.batch, config["pooled_projection_dim"]), generator=generator),
            "times": torch.tensor([731.0, 42.0][:case.batch]),
            "probe": torch.randn(shape, generator=generator),
        }
        guidance = torch.tensor(case.guidance[:case.batch]) if config["guidance_embeds"] else torch.zeros(0)
        if seed and guidance.numel():
            guidance = guidance + torch.randn(guidance.shape, generator=generator)
        return {**{key: value.numpy() for key, value in inputs.items()}, "guidance": guidance.numpy()}

    def walk(model, dtype, drawn) -> dict[str, np.ndarray]:
        inputs = [torch.from_numpy(drawn[key]).to(dtype).requires_grad_()
                  for key in ("packed", "context", "pooled")]
        times, guidance, probe = (torch.from_numpy(drawn[key]).to(dtype)
                                  for key in ("times", "guidance", "probe"))
        named = list(model.named_parameters())
        output = model(hidden_states=inputs[0], encoder_hidden_states=inputs[1],
                       pooled_projections=inputs[2], timestep=times / 1000,
                       guidance=guidance if guidance.numel() else None,
                       txt_ids=torch.zeros(TOKENS, 3, dtype=dtype),
                       img_ids=image_ids(rows, columns).to(dtype), return_dict=False)[0]
        gradients = torch.autograd.grad((output * probe).sum(),
                                        inputs + [value for _, value in named])
        return {"output": output.detach().numpy(), "grad_packed": gradients[0].numpy(),
                "grad_context": gradients[1].numpy(), "grad_pooled": gradients[2].numpy(),
                **{f"grad_param.{key}": gradient.numpy()
                   for (key, _), gradient in zip(named, gradients[3:], strict=True)}}

    with float64():
        wide_model = FluxTransformer2DModel.from_pretrained(directory, torch_dtype=torch.float64).eval()

    def evaluate(inputs):
        narrow = walk(model, torch.float32, inputs)
        with float64():
            wide = walk(wide_model, torch.float64, inputs)
        return narrow, wide

    original = draw(0)
    narrow, wide = evaluate(original)
    arrays = {
        "config": np.asarray(json.dumps(json.loads((directory / "config.json").read_text()))),
        **original, **narrow, **{f"fp64.{key}": value for key, value in wide.items()},
    }
    print(f"{name}: packed {shape} grid {case.grid} tokens {TOKENS} "
          f"|output| <= {float(np.abs(narrow['output']).max()):.4g} parameters {len(narrow) - 4}")
    proxy, _ = flux_walk(root, name, case.grid, dtype=None)
    return {**{f"{name}.{key}": value for key, value in arrays.items()},
            **record_case(name, draw, evaluate, proxy)}


def reload(directory: str, recorded: str) -> None:
    """Read a native export with the actual source classes.

    `directory` is a directory Dew wrote after a native training step and
    `recorded` the arrays that step left behind. The published pipeline is
    loaded from those files and the actual transformer recomputes the forward
    the native model computed with the trained weights.
    """
    from diffusers import FluxPipeline

    arrays = np.load(recorded)
    pipe = FluxPipeline.from_pretrained(directory, torch_dtype=torch.float32,
                                        local_files_only=True)
    model = pipe.transformer.eval()
    rows = int(arrays["rows"])
    with torch.no_grad():
        output = model(hidden_states=torch.from_numpy(arrays["packed"]),
                       encoder_hidden_states=torch.from_numpy(arrays["context"]),
                       pooled_projections=torch.from_numpy(arrays["pooled"]),
                       timestep=torch.from_numpy(arrays["times"]) / 1000,
                       guidance=torch.from_numpy(arrays["guidance"]),
                       txt_ids=torch.zeros(arrays["context"].shape[1], 3),
                       img_ids=image_ids(rows, rows), return_dict=False)[0]
    native = arrays["native"]
    gap = float(np.abs(output.numpy() - native).max() / max(1.0, float(np.abs(native).max())))
    trained = float(np.abs(model.proj_out.weight.detach().numpy().T - arrays["proj_out"]).max())
    print(f"reimported forward gap {gap:.3g}; trained kernel gap {trained:.3g}")
    if not (gap < 1e-5 and trained == 0.0):
        raise SystemExit("the source did not read the native update")
    print("the source reads the native update")


def main(destination: str) -> None:
    import diffusers
    from reference_releases import require

    require("diffusers")
    # Diffusers 0.34.0's pipeline modules import names transformers 5 dropped.
    restore_transformers_names()
    torch.set_num_threads(2)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    cases: dict[str, dict[str, object]] = {}
    record: dict[str, object] = {"diffusers": diffusers.__version__, "base": BASE, "tokens": TOKENS,
                                 "cases": cases}
    for name, case in CASES.items():
        arrays.update(build(name, case, root))
        cases[name] = {"config": {**BASE, **case.config}, "grid": list(case.grid),
                       "batch": case.batch, "guidance": list(case.guidance[: case.batch])}
    import inspect

    from diffusers import FluxPipeline

    defaults = inspect.signature(FluxPipeline.__call__).parameters
    arrays.update(pipeline_record(root))
    record["pipeline"] = {"config": PIPELINE, "prompts": PROMPTS, "size": PIPELINE_SIZE,
                          "sequence": SEQUENCE,
                          "default_steps": defaults["num_inference_steps"].default,
                          "default_guidance": defaults["guidance_scale"].default,
                          "true_cfg": defaults["true_cfg_scale"].default}
    arrays["provenance"] = provenance(
        "diffusers.FluxTransformer2DModel", diffusers.__version__, "cpu",
        "PYTHONPATH=src python tools/diffusers_flux_reference.py OUTPUT_DIR",
        generator_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        torch=torch.__version__)
    np.savez_compressed(root / "flux_transformer.npz", **arrays)
    (root / "flux_transformer.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB, {len(CASES)} cases")


# The pipeline half: one tiny Flux pipeline, saved the way a published
# checkpoint is saved and walked through its own call.
PIPELINE = dict(BASE, guidance_embeds=True, num_layers=1, num_single_layers=1)
PROMPTS = [{"text": "a red cat", "second": "a green bird"},
           {"text": "tiny photo", "second": "a blue dog"}]
PIPELINE_SIZE = 16
SEQUENCE = 512


def build_pipeline(root: Path):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxPipeline
    from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel
    from transformers import CLIPTextConfig, CLIPTextModel, T5Config, T5EncoderModel

    directory = root / "pipeline"
    tokenizer = clip_tokenizers(directory, count=1)[0]
    tokenizer_2 = t5_tokenizer()
    torch.manual_seed(SEED + 5)
    # Flux reads its CLIP tower's pooled row, so the special ids must be the
    # tokenizer's own for that row to be the end-of-text one.
    text_encoder = CLIPTextModel(CLIPTextConfig(
        vocab_size=len(tokenizer.get_vocab()), hidden_size=PIPELINE["pooled_projection_dim"],
        intermediate_size=2 * PIPELINE["pooled_projection_dim"], num_hidden_layers=2,
        num_attention_heads=2, projection_dim=PIPELINE["pooled_projection_dim"],
        max_position_embeddings=tokenizer.model_max_length, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)).eval()
    text_encoder_2 = T5EncoderModel(T5Config(
        vocab_size=tokenizer_2.vocab_size, d_model=PIPELINE["joint_attention_dim"], d_ff=32,
        num_layers=2, num_heads=2, d_kv=8, relative_attention_num_buckets=8,
        feed_forward_proj="gated-gelu")).eval()
    vae = AutoencoderKL(in_channels=3, out_channels=3, block_out_channels=(4, 8),
                        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
                        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
                        layers_per_block=1, latent_channels=PIPELINE["in_channels"] // 4,
                        norm_num_groups=2, sample_size=PIPELINE_SIZE, shift_factor=0.1159,
                        scaling_factor=0.3611, use_quant_conv=False,
                        use_post_quant_conv=False).eval()
    pipe = FluxPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, use_dynamic_shifting=True, base_shift=0.5, max_shift=1.15,
            base_image_seq_len=256, max_image_seq_len=4096),
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
        text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2,
        transformer=FluxTransformer2DModel(**PIPELINE).eval())
    pipe.save_pretrained(directory, safe_serialization=True)
    pipe.set_progress_bar_config(disable=True)
    # The class declares no sample size, so the directory declares the
    # geometry it is read at, which is what these keys are for.
    index = json.loads((directory / "model_index.json").read_text())
    index.update(dew_height=PIPELINE_SIZE, dew_width=PIPELINE_SIZE)
    (directory / "model_index.json").write_text(json.dumps(index, indent=2))
    return pipe, directory


def pipeline_record(root: Path) -> dict[str, np.ndarray]:
    """The prompt states and the unmodified call's walk from fixed latents, in
    float32 and, the saved pipeline loaded widened, in float64 (`fp64.`)."""
    from diffusers import FluxPipeline

    pipe, directory = build_pipeline(root)
    prompts = [row["text"] for row in PROMPTS]
    seconds = [row["second"] for row in PROMPTS]
    generator = torch.Generator().manual_seed(SEED + 7)
    rows = PIPELINE_SIZE // 2 // 2
    latents = torch.randn((len(PROMPTS), rows * rows, PIPELINE["in_channels"]),
                          generator=generator, dtype=torch.float32)

    def walk(pipe, dtype) -> dict[str, np.ndarray]:
        with torch.no_grad():
            embeds, pooled, ids = pipe.encode_prompt(prompt=prompts, prompt_2=seconds,
                                                     device=torch.device("cpu"), num_images_per_prompt=1,
                                                     max_sequence_length=SEQUENCE)
            walked = pipe(prompt=prompts, prompt_2=seconds, height=PIPELINE_SIZE, width=PIPELINE_SIZE,
                          latents=latents.to(dtype), output_type="latent").images
            images = pipe(prompt=prompts, prompt_2=seconds, height=PIPELINE_SIZE, width=PIPELINE_SIZE,
                          latents=latents.to(dtype), output_type="np").images
        return {"context": embeds.numpy(), "pooled": pooled.numpy(), "text_ids": ids.numpy(),
                "latents": walked.numpy(), "images": images}

    narrow = walk(pipe, torch.float32)
    with float64():
        wide_pipe = FluxPipeline.from_pretrained(directory, torch_dtype=torch.float64, local_files_only=True)
        wide_pipe.set_progress_bar_config(disable=True)
        wide = walk(wide_pipe, torch.float64)
    print(f"pipeline: context {narrow['context'].shape} pooled {narrow['pooled'].shape} "
          f"latents {narrow['latents'].shape} |image| <= {float(np.abs(narrow['images']).max()):.4g}")
    return {"pipeline.x_T": latents.numpy(), **{f"pipeline.{key}": value for key, value in narrow.items()},
            **{f"pipeline.fp64.{key}": value for key, value in wide.items() if key != "text_ids"}}

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "bundle":
        bundle(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 2 and sys.argv[1] == "reload":
        reload(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dew-flux-reference")
