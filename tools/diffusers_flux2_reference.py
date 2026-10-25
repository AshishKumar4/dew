"""Actual Diffusers 0.40.0 FLUX.2 objects, saved and walked, for tests/fixtures.

Tiny `Flux2Transformer2DModel` instances are constructed and saved with
`save_pretrained` - their real config and their real safetensors - and each
is run the way `Flux2Pipeline` runs it: the latent flattened one token per
position, the four-axis ids the pipeline lays out for the text and the grid,
the timestep it divides by a thousand, and the distilled guidance. Every case
records the forward, the vector-Jacobian products against a fixed cotangent
for the latent and the text states, and the gradient of every parameter,
unmodified, in float32, the reference, and in float64, the truth
(`torch_precision.float64`: the source's float32 pins widened). Every
other walk below is recorded the same two ways.

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

import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from diffusers_reference_helpers import perturb, recording, saved_pipeline, tiny_qwen3_encoder
from torch_precision import float64, float64_twin
from reference_releases import RELEASES

DIFFUSERS = RELEASES["diffusers-0.40"]["diffusers"]
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


def widened_walk(walk, model) -> dict[str, np.ndarray]:
    """`walk(model, torch.float64)` inside `float64`, the flow-matching scheduler's own tables widened
    around every call into it (`float64_scheduler`)."""
    from diffusers.schedulers import scheduling_flow_match_euler_discrete
    from diffusers_wan_reference import float64_scheduler

    with float64(), float64_scheduler(scheduling_flow_match_euler_discrete):
        return walk(model, torch.float64)


def twice(walk, model) -> dict[str, np.ndarray]:
    """`walk(model, dtype)` in float32 and, on `model`'s float64 twin (`float64_twin`), in float64 (`fp64.`)."""
    narrow = walk(model, torch.float32)
    wide = widened_walk(walk, float64_twin(model))
    return {**narrow, **{f"fp64.{key}": value for key, value in wide.items()}}


def build(name: str, case: Case, root: Path) -> dict[str, np.ndarray]:
    """The case's saved transformer and its walk in float32 and in float64."""
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
    latent = torch.randn((batch, rows * columns, config["in_channels"]), generator=generator)
    context = torch.randn((batch, TOKENS, config["joint_attention_dim"]), generator=generator)
    times = torch.tensor([731.0, 42.0])
    guidance = torch.tensor(case.guidance) if config["guidance_embeds"] else None
    text_ids, image_ids = ids(rows, columns)
    probe = torch.randn((batch, rows * columns, config.get("out_channels") or config["in_channels"]),
                        generator=generator)

    def walk(model, dtype) -> dict[str, np.ndarray]:
        inputs = [tensor.detach().to(dtype).requires_grad_() for tensor in (latent, context)]
        named = list(model.named_parameters())
        output = model(hidden_states=inputs[0], encoder_hidden_states=inputs[1],
                       timestep=times.to(dtype) / 1000, guidance=None if guidance is None else guidance.to(dtype),
                       img_ids=image_ids, txt_ids=text_ids, return_dict=False)[0]
        grads = torch.autograd.grad((output * probe.to(dtype)).sum(), inputs + [value for _, value in named])
        return {"output": output.detach().numpy(), "grad_latent": grads[0].numpy(),
                "grad_context": grads[1].numpy(),
                **{f"grad_param.{key}": gradient.numpy()
                   for (key, _), gradient in zip(named, grads[2:], strict=True)}}

    arrays = {"latent": latent.numpy(), "context": context.numpy(), "times": times.numpy(),
              "guidance": np.zeros(0, np.float32) if guidance is None else guidance.numpy(),
              "probe": probe.numpy(), **twice(walk, model)}
    print(f"{name}: grid {case.grid} tokens {TOKENS} |output| <= {float(np.abs(arrays['output']).max()):.4g}")
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
    image = torch.rand((2, 3, 16, 24), generator=generator) * 2 - 1
    with torch.no_grad():
        folded = Flux2Pipeline._patchify_latents(model.encode(image).latent_dist.mode()).shape
    probe_latent, code = (torch.randn(folded, generator=generator) for _ in range(2))
    probe = torch.randn(image.shape, generator=generator)

    def walk(model, dtype) -> dict[str, np.ndarray]:
        statistics = (model.bn.running_mean.view(1, -1, 1, 1),
                      torch.sqrt(model.bn.running_var.view(1, -1, 1, 1) + model.config.batch_norm_eps))
        pixels_in = image.to(dtype).requires_grad_()
        latent = (Flux2Pipeline._patchify_latents(model.encode(pixels_in).latent_dist.mode())
                  - statistics[0]) / statistics[1]
        (grad_image,) = torch.autograd.grad((latent * probe_latent.to(dtype)).sum(), [pixels_in])
        code_in = code.to(dtype).requires_grad_()
        unfolded = Flux2Pipeline._unpatchify_latents(code_in * statistics[1] + statistics[0])
        pixels = model.decode(unfolded).sample
        (grad_code,) = torch.autograd.grad((pixels * probe.to(dtype)).sum(), [code_in])
        return {"latent": latent.detach().numpy(), "grad_image": grad_image.numpy(),
                "pixels": pixels.detach().numpy(), "grad_code": grad_code.numpy()}

    arrays = {"image": image.numpy(), "probe_latent": probe_latent.numpy(), "code": code.numpy(),
              "probe": probe.numpy(), **twice(walk, model)}
    print(f"{name}: latent {arrays['latent'].shape} pixels {arrays['pixels'].shape}")
    return arrays


PUBLISHED_VAE = ("black-forest-labs/FLUX.2-klein-4B", "e7b7dc27f91deacad38e78976d1f2b499d76a294")
CROP = 64


def smooth_image() -> np.ndarray:
    """A smooth `[1, 3, 128, 192]` image in [-1, 1] with a little noise."""
    y, x = np.mgrid[0:128, 0:192] / 32.0
    image = np.stack([np.sin(x + y), np.cos(2 * x - y), np.sin(3 * y) * np.cos(x)])[None]
    return (0.8 * image + 0.05 * np.random.default_rng(0).standard_normal(image.shape)).astype(np.float32)


def published(destination: str) -> None:
    """The published VAE's pipeline latent of `smooth_image()` and its
    decode's top-left `CROP` pixels, in float32 and in float64 (`.fp64`),
    for `tests/fixtures/flux2_published.npz`."""
    from diffusers import AutoencoderKLFlux2, Flux2Pipeline

    model = AutoencoderKLFlux2.from_pretrained(PUBLISHED_VAE[0], subfolder="vae", revision=PUBLISHED_VAE[1]).eval()

    def walk(model, dtype) -> dict[str, np.ndarray]:
        with torch.no_grad():
            mean = model.bn.running_mean.view(1, -1, 1, 1)
            std = torch.sqrt(model.bn.running_var.view(1, -1, 1, 1) + model.config.batch_norm_eps)
            folded = Flux2Pipeline._patchify_latents(
                model.encode(torch.from_numpy(smooth_image()).to(dtype)).latent_dist.mode())
            pixels = model.decode(Flux2Pipeline._unpatchify_latents(folded)).sample
            return {"vae.latent": ((folded - mean) / std).numpy(), "vae.pixels": pixels.numpy()[:, :, :CROP, :CROP]}

    arrays = {key.replace("fp64.vae.", "vae.") + (".fp64" if key.startswith("fp64.") else ""): value
              for key, value in twice(walk, model).items()}
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


def text_encoder(tok):
    """The published Qwen3 config, narrowed but as deep as the layers the
    pipeline reads (9, 18, 27); every field it does not name is the
    release's own."""
    return tiny_qwen3_encoder(SOURCE, tok, hidden_size=PIPELINE["joint_attention_dim"] // 3, layers=28,
                              seed=SEED)


def build_pipeline(root: Path):
    from diffusers import (
        AutoencoderKLFlux2,
        FlowMatchEulerDiscreteScheduler,
        Flux2KleinPipeline,
        Flux2Transformer2DModel,
    )
    from make_tiny_tokenizer import qwen_byte_tokenizer

    tok = qwen_byte_tokenizer(json.loads((SOURCE / "tokenizer" / "tokenizer_config.json").read_text()),
                              chat_template=(SOURCE / "tokenizer" / "chat_template.jinja").read_text())
    encoder = text_encoder(tok)
    torch.manual_seed(SEED + 7)
    vae = AutoencoderKLFlux2(**PIPELINE_VAE).eval()
    transformer = Flux2Transformer2DModel(**PIPELINE).eval()
    generator = torch.Generator().manual_seed(SEED + 8)
    perturb(generator, 0.05, vae, transformer)
    with torch.no_grad():
        vae.bn.running_mean.copy_(0.3 * torch.randn(vae.bn.running_mean.shape, generator=generator))
        vae.bn.running_var.copy_(torch.rand(vae.bn.running_var.shape, generator=generator) + 0.2)
    scheduler = json.loads((SOURCE / "scheduler" / "scheduler_config.json").read_text())
    pipe = Flux2KleinPipeline(scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler), vae=vae,
                              text_encoder=encoder, tokenizer=tok, transformer=transformer, is_distilled=False)
    return saved_pipeline(pipe, root / "pipeline", HEIGHT, WIDTH)


def pipeline_record(root: Path) -> dict[str, np.ndarray]:
    """The prompt states each prompt encodes to, and the unmodified call's
    walk from fixed latents at `STEPS` steps: the raw VAE latent it ends on
    and the image it decodes; in float32 and, widened, in float64."""
    pipe = build_pipeline(root)
    rows, columns = HEIGHT // 4, WIDTH // 4
    generator = torch.Generator().manual_seed(SEED + 9)
    latents = torch.randn((len(PROMPTS), PIPELINE["in_channels"], rows, columns), generator=generator)

    def walk(pipe, dtype) -> dict[str, np.ndarray]:
        arrays = {}
        with torch.no_grad():
            for row, prompt in enumerate([*PROMPTS, ""]):
                states, _ = pipe.encode_prompt(prompt=prompt, device=torch.device("cpu"))
                arrays[f"context.{row}"] = states.numpy()
            for row, prompt in enumerate(PROMPTS):
                call = {"prompt": prompt, "height": HEIGHT, "width": WIDTH, "num_inference_steps": STEPS}
                start = latents[row:row + 1].to(dtype)
                walked = pipe(**call, latents=start.clone(), output_type="latent").images
                arrays[f"latents.{row}"] = walked.numpy()
                arrays[f"images.{row}"] = pipe(**call, latents=start.clone(), output_type="np").images
        return arrays

    narrow = walk(pipe, torch.float32)
    # The pipeline's float64 run: each of its modules' float64 twin (`float64_twin`).
    twin = copy.deepcopy(pipe)
    for name, component in pipe.components.items():
        if isinstance(component, torch.nn.Module):
            setattr(twin, name, float64_twin(component))
    wide = widened_walk(walk, twin)
    print(f"pipeline: latents {narrow['latents.0'].shape} images {narrow['images.0'].shape}")
    return {"pipeline.x_T": latents.numpy(), **{f"pipeline.{key}": value for key, value in narrow.items()},
            **{f"pipeline.fp64.{key}": value for key, value in wide.items()}}


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

    def walk(model, dtype) -> dict[str, np.ndarray]:
        with torch.no_grad():
            return {"context": Flux2Pipeline._get_mistral_3_small_prompt_embeds(
                model, tok, list(PROMPTS), dtype=dtype, device=torch.device("cpu")).numpy()}

    arrays = twice(walk, model)
    print(f"mistral3: context {arrays['context'].shape}")
    return {f"mistral3.{key}": value for key, value in arrays.items()}


def main(destination: str) -> None:
    root = recording(destination, "diffusers-0.40")
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
