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

A tiny Mistral-3 encoder over Mistral Small 3.1's config and chat template
(FLUX.2 [dev]'s own files are gated) records the prompt states [dev]'s
pipeline reads. Two tiny `AutoencoderKLFlux2`s, one with the small decoder's narrower
levels, are walked as the pipeline encodes a reference image and decodes its
result, the latent folded 2x2 and normalized by the batch norm's statistics.

Run in the isolated reference environment (diffusers 0.40.0, transformers
5.17.0, torch 2.8.0, CPU):

    python tools/diffusers_flux2_reference.py OUTPUT_DIR
    python tools/diffusers_flux2_reference.py bundle OUTPUT_DIR tests/fixtures/flux2_source.tar.xz
    python tools/diffusers_flux2_reference.py published tests/fixtures/flux2_published.npz
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from diffusers_reference_helpers import (
    rounded_frequency_table as rounded_frequency_table,
    rounded_timestep_embedding as rounded_timestep_embedding,
)

DIFFUSERS = "0.40.0"
BASE = {"patch_size": 1, "in_channels": 8, "num_layers": 2, "num_single_layers": 2, "attention_head_dim": 16,
        "num_attention_heads": 2, "joint_attention_dim": 12, "timestep_guidance_channels": 32, "mlp_ratio": 3.0,
        "axes_dims_rope": (4, 4, 4, 4), "rope_theta": 2000, "eps": 1e-6, "guidance_embeds": True}
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
    "unguided": Case({"guidance_embeds": False, "num_layers": 1, "num_single_layers": 3}),
    "narrow": Case({"mlp_ratio": 2.0, "eps": 1e-3, "out_channels": 4, "rope_theta": 10000}, grid=(2, 6)),
}


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


VAE = {"block_out_channels": [8, 16], "down_block_types": ["DownEncoderBlock2D"] * 2,
       "up_block_types": ["UpDecoderBlock2D"] * 2, "layers_per_block": 1, "latent_channels": 4,
       "norm_num_groups": 4}
VAES = {"vae": {}, "small_decoder": {"decoder_block_out_channels": [4, 12]}}


def build_vae(name: str, root: Path) -> dict[str, np.ndarray]:
    """A tiny `AutoencoderKLFlux2`, every parameter and the batch norm's
    running statistics moved off their init, walked as `Flux2Pipeline` walks
    it: `_encode_vae_image` (posterior mode, 2x2 fold, batch-norm
    normalization) and the call's decode (the inverse, then `vae.decode`),
    with the gradients of fixed probes against the pixels and the latent."""
    from diffusers import AutoencoderKLFlux2, Flux2Pipeline

    torch.manual_seed(SEED)
    model = AutoencoderKLFlux2(**VAE, **VAES[name]).eval()
    generator = torch.Generator().manual_seed(SEED + 2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
        model.bn.running_mean.copy_(0.3 * torch.randn(model.bn.running_mean.shape, generator=generator))
        model.bn.running_var.copy_(torch.rand(model.bn.running_var.shape, generator=generator) + 0.2)
    model.save_pretrained(root / name / "vae", safe_serialization=True)
    statistics = (model.bn.running_mean.view(1, -1, 1, 1),
                  torch.sqrt(model.bn.running_var.view(1, -1, 1, 1) + model.config.batch_norm_eps))
    image = (torch.rand((2, 3, 16, 24), generator=generator) * 2 - 1).requires_grad_()
    latent = (Flux2Pipeline._patchify_latents(model.encode(image).latent_dist.mode()) - statistics[0]) / statistics[1]
    probe_latent = torch.randn(latent.shape, generator=generator)
    (grad_image,) = torch.autograd.grad((latent * probe_latent).sum(), [image])
    code = torch.randn(latent.shape, generator=generator).requires_grad_()
    pixels = model.decode(Flux2Pipeline._unpatchify_latents(code * statistics[1] + statistics[0])).sample
    probe = torch.randn(pixels.shape, generator=generator)
    (grad_code,) = torch.autograd.grad((pixels * probe).sum(), [code])
    print(f"{name}: latent {tuple(latent.shape)} pixels {tuple(pixels.shape)}")
    return {"image": image.detach().numpy(), "latent": latent.detach().numpy(), "probe_latent": probe_latent.numpy(),
            "grad_image": grad_image.numpy(), "code": code.detach().numpy(), "pixels": pixels.detach().numpy(),
            "probe": probe.numpy(), "grad_code": grad_code.numpy()}


PUBLISHED_VAE = ("black-forest-labs/FLUX.2-klein-4B", "e7b7dc27f91deacad38e78976d1f2b499d76a294")
CROP = 64


def smooth_image() -> np.ndarray:
    """A smooth `[1, 3, 128, 192]` image in [-1, 1] with a little noise."""
    y, x = np.mgrid[0:128, 0:192] / 32.0
    image = np.stack([np.sin(x + y), np.cos(2 * x - y), np.sin(3 * y) * np.cos(x)])[None]
    return (0.8 * image + 0.05 * np.random.default_rng(0).standard_normal(image.shape)).astype(np.float32)


def published(destination: str) -> None:
    """The published VAE's pipeline latent of `smooth_image()` and its
    decode's top-left `CROP` pixels, in float64, and how far the source's
    own float32 run lands from each, for `tests/fixtures/flux2_published.npz`."""
    from diffusers import AutoencoderKLFlux2, Flux2Pipeline

    model = AutoencoderKLFlux2.from_pretrained(PUBLISHED_VAE[0], subfolder="vae", revision=PUBLISHED_VAE[1]).eval()

    def walk(dtype):
        with torch.no_grad():
            mean = model.bn.running_mean.view(1, -1, 1, 1).to(dtype)
            std = torch.sqrt(model.bn.running_var.view(1, -1, 1, 1).to(dtype) + model.config.batch_norm_eps)
            folded = Flux2Pipeline._patchify_latents(
                model.encode(torch.from_numpy(smooth_image()).to(dtype)).latent_dist.mode())
            pixels = model.decode(Flux2Pipeline._unpatchify_latents(folded)).sample
            return ((folded - mean) / std).numpy(), pixels.numpy()[:, :, :CROP, :CROP]

    single = walk(torch.float32)
    model.double()
    exact = walk(torch.float64)
    arrays = {}
    for key, value, rounded in zip(("latent", "pixels"), exact, single, strict=True):
        arrays[f"vae.{key}"] = value
        arrays[f"vae.{key}.float32_gap"] = np.float64(np.abs(rounded - value).max() / max(1.0, np.abs(value).max()))
    print({key: float(value) for key, value in arrays.items() if key.endswith("gap")})
    np.savez_compressed(destination, **arrays)
    print(f"{destination}: {Path(destination).stat().st_size / 1e6:.2f} MB")


# The pipeline half: one tiny FLUX.2 [klein] pipeline over the published
# configs, shrunk, saved the way the release is saved and walked through its
# own call. The release is step-distilled; this one is not, so its call
# guides two branches, the empty prompt the negative.
SOURCE = Path(__file__).resolve().parents[1] / "tests/fixtures/hf/flux2-klein-4b-source"
PIPELINE = {**BASE, "in_channels": 16, "joint_attention_dim": 48, "guidance_embeds": False, "num_layers": 1,
            "num_single_layers": 1}
PIPELINE_VAE = {"block_out_channels": [8, 16], "down_block_types": ["DownEncoderBlock2D"] * 2,
                "up_block_types": ["UpDecoderBlock2D"] * 2, "layers_per_block": 1, "latent_channels": 4,
                "norm_num_groups": 4}
PROMPTS = ["a red cat on a mat", "tiny photo"]
HEIGHT, WIDTH, STEPS = 16, 24, 4


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
    made.chat_template = (SOURCE / "tokenizer" / "chat_template.jinja").read_text()
    return made


def text_encoder(tok):
    """The published Qwen3 config, narrowed but as deep as the layers the
    pipeline reads (9, 18, 27); every field it does not name is the
    release's own."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = json.loads((SOURCE / "text_encoder" / "config.json").read_text())
    config.pop("dtype")
    config.update(hidden_size=PIPELINE["joint_attention_dim"] // 3, intermediate_size=32, num_hidden_layers=28,
                  layer_types=["full_attention"] * 28, max_window_layers=28, num_attention_heads=2,
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
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
    )

    tok = tokenizer()
    encoder = text_encoder(tok)
    torch.manual_seed(SEED + 7)
    vae = AutoencoderKLFlux2(**PIPELINE_VAE).eval()
    transformer = Flux2Transformer2DModel(**PIPELINE).eval()
    generator = torch.Generator().manual_seed(SEED + 8)
    with torch.no_grad():
        for module in (vae, transformer):
            for parameter in module.parameters():
                parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)
        vae.bn.running_mean.copy_(0.3 * torch.randn(vae.bn.running_mean.shape, generator=generator))
        vae.bn.running_var.copy_(torch.rand(vae.bn.running_var.shape, generator=generator) + 0.2)
    scheduler = json.loads((SOURCE / "scheduler" / "scheduler_config.json").read_text())
    pipe = Flux2KleinPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler), vae=vae,
                              text_encoder=encoder, tokenizer=tok, transformer=transformer, is_distilled=False)
    directory = root / "pipeline"
    pipe.save_pretrained(directory, safe_serialization=True)
    pipe.set_progress_bar_config(disable=True)
    # The class declares no sample size, so the directory declares the
    # geometry it is read at.
    index = json.loads((directory / "model_index.json").read_text())
    index.update(dew_height=HEIGHT, dew_width=WIDTH)
    (directory / "model_index.json").write_text(json.dumps(index, indent=2))
    return pipe


def pipeline_record(root: Path) -> dict[str, np.ndarray]:
    """The prompt states each prompt encodes to, and the unmodified call's
    walk from fixed latents at `STEPS` steps: the raw VAE latent it ends on
    and the image it decodes."""
    pipe = build_pipeline(root)
    rows, columns = HEIGHT // 4, WIDTH // 4
    generator = torch.Generator().manual_seed(SEED + 9)
    latents = torch.randn((len(PROMPTS), PIPELINE["in_channels"], rows, columns), generator=generator)
    arrays: dict[str, np.ndarray] = {"pipeline.x_T": latents.numpy()}
    for row, prompt in enumerate([*PROMPTS, ""]):
        with torch.no_grad():
            embeds, _ = pipe.encode_prompt(prompt=prompt, device=torch.device("cpu"))
        arrays[f"pipeline.context.{row}"] = embeds.numpy()
    for row, prompt in enumerate(PROMPTS):
        with torch.no_grad():
            walked = pipe(prompt=prompt, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
                          latents=latents[row:row + 1].clone(), output_type="latent").images
            images = pipe(prompt=prompt, height=HEIGHT, width=WIDTH, num_inference_steps=STEPS,
                          latents=latents[row:row + 1].clone(), output_type="np").images
        arrays[f"pipeline.latents.{row}"] = walked.numpy()
        arrays[f"pipeline.images.{row}"] = images
        print(f"pipeline {prompt!r}: latents {tuple(walked.shape)} images {images.shape}")
    return arrays


# FLUX.2 [dev]'s Mistral-3 encoder: a tiny `Mistral3ForConditionalGeneration`
# over Mistral Small 3.1's config and chat template (the release's own files
# are gated), one layer deeper than the last the pipeline stacks (10, 20, 30;
# transformers' last hidden state is after the final norm), and its
# prompt states as `Flux2Pipeline._get_mistral_3_small_prompt_embeds` reads them.
MISTRAL = Path(__file__).resolve().parents[1] / "tests/fixtures/hf/mistral-small-3.1-source"


def mistral_tokenizer():
    """A byte-level tokenizer carrying Mistral Small 3.1's named special
    tokens and chat template: every character is one piece."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    metadata = json.loads((MISTRAL / "tokenizer_config.json").read_text())
    special = [item["content"] for item in metadata["added_tokens_decoder"].values()
               if not item["content"].startswith("<SPECIAL_")]
    vocab = {word: index for index, word in enumerate(special)}
    for word in sorted(pre_tokenizers.ByteLevel.alphabet()):
        vocab.setdefault(word, len(vocab))
    model = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    model.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    model.decoder = decoders.ByteLevel()
    made = PreTrainedTokenizerFast(tokenizer_object=model, bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                   unk_token="<unk>", additional_special_tokens=special)
    made.chat_template = json.loads((MISTRAL / "chat_template.json").read_text())["chat_template"]
    return made


def build_mistral3(root: Path) -> dict[str, np.ndarray]:
    from diffusers import Flux2Pipeline
    from transformers import AutoModelForImageTextToText, Mistral3Config

    tok = mistral_tokenizer()
    config = json.loads((MISTRAL / "config.json").read_text())
    config.pop("torch_dtype", None)
    config["text_config"].update(hidden_size=16, intermediate_size=32, num_hidden_layers=31, num_attention_heads=2,
                                 num_key_value_heads=1, head_dim=8, vocab_size=len(tok),
                                 max_position_embeddings=1024)
    config["vision_config"].update(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                                   head_dim=8, image_size=56)
    config["image_token_index"] = tok.convert_tokens_to_ids("[IMG]")
    torch.manual_seed(SEED + 10)
    model = AutoModelForImageTextToText.from_config(Mistral3Config(**config)).eval()
    generator = torch.Generator().manual_seed(SEED + 11)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)
    model.save_pretrained(root / "mistral3" / "text_encoder", safe_serialization=True)
    tok.save_pretrained(root / "mistral3" / "tokenizer")
    with torch.no_grad():
        embeds = Flux2Pipeline._get_mistral_3_small_prompt_embeds(model, tok, list(PROMPTS), dtype=torch.float32,
                                                                  device=torch.device("cpu"))
    print(f"mistral3: context {tuple(embeds.shape)}")
    return {"mistral3.context": embeds.numpy()}


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
    for name in VAES:
        arrays.update({f"{name}.{key}": value for key, value in build_vae(name, root).items()})
    arrays.update(pipeline_record(root))
    arrays.update(build_mistral3(root))
    record = {"diffusers": DIFFUSERS, "vae": VAE, "vaes": VAES, "base": BASE,
              "pipeline": {"config": PIPELINE, "vae": PIPELINE_VAE, "prompts": PROMPTS, "height": HEIGHT,
                           "width": WIDTH, "steps": STEPS}, "tokens": TOKENS, "seed": SEED,
              "cases": {name: {"config": {**BASE, **case.config}, "grid": list(case.grid),
                               "guidance": list(case.guidance)} for name, case in CASES.items()}}
    np.savez_compressed(root / "flux2_transformer.npz", **arrays)
    (root / "flux2_transformer.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB, {len(CASES)} cases")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "bundle":
        from diffusers_dc_ae_reference import bundle

        bundle(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 2 and sys.argv[1] == "published":
        published(sys.argv[2])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dew-flux2-reference")
