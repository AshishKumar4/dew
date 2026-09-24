"""Actual Diffusers 0.34.0 Flux objects, saved and walked, for tests/fixtures.

Tiny `FluxTransformer2DModel` instances are constructed and saved with
`save_pretrained` - their real config and their real safetensors - and each
is then run on packed latents the way its pipeline runs it: the ids the
pipeline lays out, the timestep it divides by a thousand, and the distilled
guidance a guidance-embedded checkpoint takes. Every case records the
forward, the vector-Jacobian products against a fixed cotangent for the
latent, the text tokens and the pooled vector, and the gradient of every
parameter, in float32, with the source's sinusoidal frequency table rounded
from float64 (`rounded_timestep_embedding`); the pipeline walk is the
unmodified source.

The variants are the ones whose wiring differs: the distilled checkpoint's
guidance embedder against the schnell-style model without one, a rectangular
packed grid, and a deeper single-stream stack.

Run in the isolated reference environment on CPU:

    PYTHONPATH=src python tools/diffusers_flux_reference.py OUTPUT_DIR
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import torch

# The SD3 tool restores the transformers names diffusers 0.34.0 imports.
from diffusers_sd3_reference import clip_tokenizers, t5_tokenizer

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
    guidance: tuple[float, ...] = (3.5, 3.5)


CASES: dict[str, Case] = {
    "schnell": Case(),
    "dev": Case(dict(guidance_embeds=True)),
    "rect": Case(dict(guidance_embeds=True), grid=(6, 2)),
    "deep": Case(dict(num_layers=1, num_single_layers=3), grid=(2, 6)),
    # Each row walked at its own distilled guidance, which the source takes as
    # a per-row tensor rather than one scalar.
    "mixed": Case(dict(guidance_embeds=True), guidance=(3.5, 7.0)),
}


def frequency_exponent(half: int, shift: float, max_period: float) -> torch.Tensor:
    """`get_timestep_embedding`'s float32 exponent, its own arithmetic."""
    exponent = -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32)
    return exponent / (half - shift)


def rounded_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False,
                               downscale_freq_shift=1, scale=1, max_period=10000):
    """The source's `get_timestep_embedding` with its exponential taken in
    float64 and rounded once to float32.

    Torch's float32 `exp` is off by one unit in the last place at some of the
    entries, and one ulp of a frequency is one ulp of the 3500-radian angle a
    distilled guidance embeds; the rounded table is the one Dew builds on the
    host, so a walk over it holds the rest of the source to the suite's bound
    with nothing else altered.
    """
    half = embedding_dim // 2
    table = torch.exp(frequency_exponent(half, downscale_freq_shift, max_period).double()).float()
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


def torch_exp_disagreements(half: int) -> int:
    """How many of the table's entries torch's float32 `exp` rounds differently
    from the float64 one; the attribution the record carries."""
    exponent = frequency_exponent(half, 0, 10000)
    return int((torch.exp(exponent) != torch.exp(exponent.double()).float()).sum())


def image_ids(rows: int, columns: int) -> torch.Tensor:
    """`FluxPipeline._prepare_latent_image_ids` over a packed grid."""
    ids = torch.zeros(rows, columns, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(rows)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(columns)[None, :]
    return ids.reshape(rows * columns, 3)


def build(name: str, case: Case, root: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """The case's saved transformer and recorded walk, and what the source's
    own float32 `exp` costs it: the worst parameter gradient's scaled gap
    between the unmodified walk and the recorded one."""
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
    # The schedule's own model times, which the pipeline divides by the
    # training count before the call and the transformer multiplies back.
    times = torch.tensor([731.0, 42.0][: case.batch], dtype=torch.float32)
    guidance = (torch.tensor(case.guidance[: case.batch], dtype=torch.float32)
                if config["guidance_embeds"] else None)
    probe = torch.randn(packed.shape, generator=generator, dtype=torch.float32)
    named = [(key, value) for key, value in model.named_parameters()]

    def walk():
        output = model(hidden_states=packed, encoder_hidden_states=context,
                       pooled_projections=pooled, timestep=times / 1000, guidance=guidance,
                       txt_ids=torch.zeros(TOKENS, 3), img_ids=image_ids(rows, columns),
                       return_dict=False)[0]
        gradients = torch.autograd.grad((output * probe).sum(), [packed, context, pooled]
                                        + [value for _, value in named])
        return output.detach().numpy(), [gradient.numpy() for gradient in gradients]

    _, unmodified = walk()
    with rounded_frequency_table():
        output, grads = walk()
    arrays = {
        "config": np.asarray(json.dumps(json.loads((directory / "config.json").read_text()))),
        "packed": packed.detach().numpy(), "context": context.detach().numpy(),
        "pooled": pooled.detach().numpy(), "times": times.numpy(),
        "guidance": np.zeros(0, np.float32) if guidance is None else guidance.numpy(),
        "output": output, "probe": probe.numpy(),
        "grad_packed": grads[0], "grad_context": grads[1], "grad_pooled": grads[2],
    }
    gaps = {}
    for (key, _), gradient, theirs in zip(named, grads[3:], unmodified[3:]):
        arrays[f"grad_param.{key}"] = gradient
        gaps[key] = float(np.abs(gradient - theirs).max() / max(1.0, float(np.abs(gradient).max())))
    tensor, gap = max(gaps.items(), key=lambda item: item[1])
    print(f"{name}: packed {tuple(packed.shape)} grid {case.grid} tokens {TOKENS} "
          f"|output| <= {float(np.abs(output).max()):.4g} parameters {len(named)}; "
          f"torch's own exp moves {tensor} by {gap:.3g}")
    return arrays, {"tensor": tensor, "gap": gap}


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
    cases: dict[str, dict[str, object]] = {}
    record: dict[str, object] = {
        "diffusers": diffusers.__version__, "base": BASE, "tokens": TOKENS, "cases": cases,
        # The transformer walks are recorded over the rounded frequency table;
        # this is what the source's own float32 `exp` differs from it by.
        "torch_exp_off_by_one_ulp": {"entries": 128, "count": torch_exp_disagreements(128)}}
    for name, case in CASES.items():
        built, unmodified = build(name, case, root)
        for key, value in built.items():
            arrays[f"{name}.{key}"] = value
        cases[name] = {"config": {**BASE, **case.config}, "grid": list(case.grid),
                       "batch": case.batch, "guidance": list(case.guidance[: case.batch]),
                       "unmodified_exp_gap": unmodified}
    import inspect

    from diffusers import FluxPipeline

    defaults = inspect.signature(FluxPipeline.__call__).parameters
    arrays.update(pipeline_record(root))
    record["pipeline"] = {"config": PIPELINE, "prompts": PROMPTS, "size": PIPELINE_SIZE,
                          "sequence": SEQUENCE,
                          "default_steps": defaults["num_inference_steps"].default,
                          "default_guidance": defaults["guidance_scale"].default,
                          "true_cfg": defaults["true_cfg_scale"].default}
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
    pipe, _ = build_pipeline(root)
    prompts = [row["text"] for row in PROMPTS]
    seconds = [row["second"] for row in PROMPTS]
    with torch.no_grad():
        embeds, pooled, ids = pipe.encode_prompt(prompt=prompts, prompt_2=seconds,
                                                 device=torch.device("cpu"),
                                                 num_images_per_prompt=1,
                                                 max_sequence_length=SEQUENCE)
    generator = torch.Generator().manual_seed(SEED + 7)
    rows = PIPELINE_SIZE // 2 // 2
    latents = torch.randn((len(PROMPTS), rows * rows, PIPELINE["in_channels"]),
                          generator=generator, dtype=torch.float32)
    with torch.no_grad():
        walked = pipe(prompt=prompts, prompt_2=seconds, height=PIPELINE_SIZE,
                      width=PIPELINE_SIZE, latents=latents.clone(),
                      output_type="latent").images
        images = pipe(prompt=prompts, prompt_2=seconds, height=PIPELINE_SIZE,
                      width=PIPELINE_SIZE, latents=latents.clone(), output_type="np").images
    print(f"pipeline: context {tuple(embeds.shape)} pooled {tuple(pooled.shape)} "
          f"latents {tuple(walked.shape)} |image| <= {float(np.abs(images).max()):.4g}")
    return {"pipeline.context": embeds.numpy(), "pipeline.pooled": pooled.numpy(),
            "pipeline.text_ids": ids.numpy(), "pipeline.x_T": latents.numpy(),
            "pipeline.latents": walked.numpy(), "pipeline.images": images}

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "bundle":
        bundle(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 2 and sys.argv[1] == "reload":
        reload(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dew-flux-reference")
