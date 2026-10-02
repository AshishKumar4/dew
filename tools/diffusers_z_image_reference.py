"""Actual Diffusers 0.40.0 Z-Image objects, saved and walked, for tests/fixtures.

Two halves. The transformer half constructs tiny `ZImageTransformer2DModel`
instances with every parameter moved off its initialization, saves each with
`save_pretrained`, and calls each as `ZImagePipeline` calls it: a list of
latents `[C, 1, H, W]`, the time `1 - sigma`, and a list of prompt states
of each prompt's own length. It records the output and, against a fixed
cotangent, the gradients of the latents and the prompt states, and for one
case of every parameter, in float32 (the weights rounded to bfloat16-representable values
so the fixture compresses), with the time embedding's frequency table rounded
from float64 (torch's float32 `exp` is one ulp off at some entries).

The pipeline half saves one tiny `ZImagePipeline` over the published
configs - a Qwen3 text encoder saved as the release stores it, the Flux VAE,
the transformer, the published scheduler config and a byte-level tokenizer
with the published chat template - and walks the unmodified call from fixed
latents, recording the prompt states, the latent it ends on and the image.

Run in the isolated reference environment (diffusers 0.40.0, transformers
5.17.0, torch 2.8.0, CPU):

    python tools/diffusers_z_image_reference.py OUTPUT_DIR
    python tools/diffusers_z_image_reference.py bundle OUTPUT_DIR tests/fixtures/z_image_source.tar.xz
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
SOURCE = Path(__file__).resolve().parents[1] / "tests/fixtures/hf/z-image-source"
BASE = {"all_patch_size": [2], "all_f_patch_size": [1], "in_channels": 4, "dim": 64, "n_layers": 2,
        "n_refiner_layers": 1, "n_heads": 2, "n_kv_heads": 2, "norm_eps": 1e-5, "qk_norm": True,
        "cap_feat_dim": 24, "rope_theta": 256.0, "t_scale": 1000.0, "axes_dims": [8, 12, 12],
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
    gradients: bool = False
    """Whether every parameter's gradient is recorded too; one case's are."""


CASES: dict[str, Case] = {
    # A row's prompt padded to 32 and the other to 64: the source pads the
    # batch to the longer, masked; the image's 24 patches pad to 32.
    "padded": Case(gradients=True),
    # 32 patches and 32 prompt tokens: no padding anywhere.
    "exact": Case(size=(8, 16), lengths=(32,), times=(0.5,)),
    "deep": Case({"n_layers": 3, "n_refiner_layers": 2, "norm_eps": 1e-3, "rope_theta": 64.0}, size=(6, 10),
                 lengths=(33,), times=(0.875,)),
}


def rounded_timestep_embedding(t, dim, max_period=10000):
    """`TimestepEmbedder.timestep_embedding` with its exponential taken in
    float64 and rounded once to float32, the table Dew builds on the host."""
    half = dim // 2
    exponent = -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    args = t[:, None].float() * torch.exp(exponent.double()).float()[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


@contextlib.contextmanager
def rounded_frequency_table():
    from diffusers.models.transformers import transformer_z_image

    original = transformer_z_image.TimestepEmbedder.timestep_embedding
    transformer_z_image.TimestepEmbedder.timestep_embedding = staticmethod(rounded_timestep_embedding)
    try:
        yield
    finally:
        transformer_z_image.TimestepEmbedder.timestep_embedding = original


def build(name: str, case: Case, root: Path) -> dict[str, np.ndarray]:
    from diffusers import ZImageTransformer2DModel

    config = {**BASE, **case.config}
    torch.manual_seed(SEED)
    model = ZImageTransformer2DModel(**config).eval()
    generator = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
            parameter.copy_(parameter.to(torch.bfloat16).float())
    model.save_pretrained(root / name / "transformer", safe_serialization=True)
    height, width = case.size
    batch = len(case.lengths)
    latents = torch.randn((batch, config["in_channels"], 1, height, width), generator=generator,
                          requires_grad=True)
    captions = [torch.randn((length, config["cap_feat_dim"]), generator=generator, requires_grad=True)
                for length in case.lengths]
    times = torch.tensor(case.times)
    probe = torch.randn((batch, config["in_channels"], 1, height, width), generator=generator)
    named = list(model.named_parameters())
    with rounded_frequency_table():
        outputs = model(list(latents.unbind(0)), times, captions, return_dict=False)[0]
        output = torch.stack(outputs)
        grads = torch.autograd.grad((output * probe).sum(), [latents, *captions] + [value for _, value in named])
    arrays = {"latents": latents.detach().numpy(), "times": times.numpy(), "output": output.detach().numpy(),
              "probe": probe.numpy(), "grad_latents": grads[0].numpy()}
    for row, caption in enumerate(captions):
        arrays[f"caption.{row}"] = caption.detach().numpy()
        arrays[f"grad_caption.{row}"] = grads[1 + row].numpy()
    if case.gradients:
        for (key, _), gradient in zip(named, grads[1 + batch:], strict=True):
            arrays[f"grad_param.{key}"] = gradient.numpy()
    print(f"{name}: size {case.size} prompts {case.lengths} |output| <= {float(output.abs().max()):.4g} "
          f"parameters {len(named)}")
    return arrays


PIPELINE = {**BASE, "cap_feat_dim": 16, "n_layers": 1}
PIPELINE_VAE = {"block_out_channels": [8, 16], "down_block_types": ["DownEncoderBlock2D"] * 2,
                "up_block_types": ["UpDecoderBlock2D"] * 2, "layers_per_block": 1, "latent_channels": 4,
                "norm_num_groups": 4, "use_quant_conv": False, "use_post_quant_conv": False}
PROMPTS = ["a red cat on a mat", "tiny photo"]
HEIGHT, WIDTH, STEPS, GUIDANCE = 16, 24, 4, 5.0


def tokenizer():
    """A byte-level Qwen2 tokenizer carrying the published special tokens and
    chat template: every character is one piece."""
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import Qwen2Tokenizer

    metadata = json.loads((SOURCE / "tokenizer" / "tokenizer_config.json").read_text())
    special = [item["content"] for item in metadata["added_tokens_decoder"].values()]
    vocab = {word: index for index, word in enumerate(sorted(ByteLevel.alphabet()))}
    for word in special:
        vocab.setdefault(word, len(vocab))
    made = Qwen2Tokenizer(vocab=vocab, merges=[], eos_token=metadata["eos_token"], pad_token=metadata["pad_token"],
                          additional_special_tokens=special)
    made.chat_template = metadata["chat_template"]
    return made


def text_encoder(tok):
    """The published Qwen3 config, narrowed and three layers deep; a causal
    language model with tied embeddings, which saves as the release is
    stored (`model.` names, no head)."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = json.loads((SOURCE / "text_encoder" / "config.json").read_text())
    config.pop("torch_dtype", None)
    config.pop("dtype", None)
    config.update(hidden_size=PIPELINE["cap_feat_dim"], intermediate_size=32, num_hidden_layers=3,
                  layer_types=["full_attention"] * 3, max_window_layers=3, num_attention_heads=2,
                  num_key_value_heads=1, head_dim=8, vocab_size=len(tok), max_position_embeddings=1024,
                  bos_token_id=None, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    torch.manual_seed(SEED + 5)
    model = Qwen3ForCausalLM(Qwen3Config(**config)).eval()
    generator = torch.Generator().manual_seed(SEED + 6)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return model


def build_pipeline(root: Path):
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        ZImagePipeline,
        ZImageTransformer2DModel,
    )

    tok = tokenizer()
    encoder = text_encoder(tok)
    published = json.loads((SOURCE / "vae" / "config.json").read_text())
    torch.manual_seed(SEED + 7)
    vae = AutoencoderKL(**PIPELINE_VAE, scaling_factor=published["scaling_factor"],
                        shift_factor=published["shift_factor"]).eval()
    transformer = ZImageTransformer2DModel(**PIPELINE).eval()
    generator = torch.Generator().manual_seed(SEED + 8)
    with torch.no_grad():
        for module in (vae, transformer):
            for parameter in module.parameters():
                parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)
    scheduler = json.loads((SOURCE / "scheduler" / "scheduler_config.json").read_text())
    pipe = ZImagePipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler), vae=vae,
                          text_encoder=encoder, tokenizer=tok, transformer=transformer)
    directory = root / "pipeline"
    pipe.save_pretrained(directory, safe_serialization=True)
    pipe.set_progress_bar_config(disable=True)
    index = json.loads((directory / "model_index.json").read_text())
    index.update(dew_height=HEIGHT, dew_width=WIDTH)
    (directory / "model_index.json").write_text(json.dumps(index, indent=2))
    return pipe


def pipeline_record(root: Path) -> dict[str, np.ndarray]:
    """The prompt states each prompt (and the empty negative) encodes to, and
    the unmodified call's walk from fixed latents at `STEPS` steps, guided at
    its default scale: the latent it ends on and the image it decodes."""
    pipe = build_pipeline(root)
    generator = torch.Generator().manual_seed(SEED + 9)
    latents = torch.randn((len(PROMPTS), PIPELINE["in_channels"], HEIGHT // 2, WIDTH // 2), generator=generator)
    arrays: dict[str, np.ndarray] = {"pipeline.x_T": latents.numpy()}
    for row, prompt in enumerate([*PROMPTS, ""]):
        with torch.no_grad():
            (states,) = pipe._encode_prompt(prompt=prompt, device=torch.device("cpu"))
        arrays[f"pipeline.context.{row}"] = states.numpy()
    with rounded_frequency_table():
        for row, prompt in enumerate(PROMPTS):
            with torch.no_grad():
                walked = pipe(prompt=prompt, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
                              guidance_scale=GUIDANCE, latents=latents[row:row + 1].clone(),
                              output_type="latent").images
                images = pipe(prompt=prompt, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
                              guidance_scale=GUIDANCE, latents=latents[row:row + 1].clone(),
                              output_type="np").images
            arrays[f"pipeline.latents.{row}"] = walked.numpy()
            arrays[f"pipeline.images.{row}"] = images
            print(f"pipeline {prompt!r}: latents {tuple(walked.shape)} images {images.shape}")
    return arrays


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
