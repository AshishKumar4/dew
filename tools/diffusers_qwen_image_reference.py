"""Actual Qwen-Image 2.1 objects from Diffusers 6256aa76, saved and walked, for tests/fixtures.

Two halves. The transformer half constructs tiny `QwenImage21Transformer2DModel`
instances, saves each with `save_pretrained`, and runs each the way
`QwenImage21Pipeline` runs it for text-to-image: packed latents, one token per
latent position; the prompt states; the timestep divided by a thousand; the
`img_mask` with one image slot per 2x2 group of target tokens appended. It
records the prediction at the image's tokens and, against a fixed cotangent,
the gradients of the latent, the prompt states and every parameter, in
float32. The source's sinusoidal frequency table is replaced by the one
rounded from float64, which is Dew's; the gap the source's own float32 `exp`
makes is recorded beside it.

A row padded behind a shorter prompt is recorded as the source's call for
that prompt alone, which is what each native row reproduces (see
`dew.nn.backbones.qwen_image`), so its forward and its gradients are those
of a separate call and the parameter gradients are summed over the calls.

The pipeline half saves one tiny `QwenImage21Pipeline` - a Qwen3-VL text
encoder, the 2.1 VAE, the transformer, the published scheduler config and a
processor over a byte-level vocabulary carrying the published special tokens
and chat template - and walks the unmodified pipeline once per prompt from
fixed latents, recording the prompt states, the walked latents and the
decoded images.

Run in the isolated reference environment on CPU (torch 2.14.0, transformers
5.17.0, diffusers at 6256aa7666cedd47443adc8f82da9a10e110b09c):

    python tools/diffusers_qwen_image_reference.py OUTPUT_DIR
    python tools/diffusers_qwen_image_reference.py bundle OUTPUT_DIR tests/fixtures/qwen_image_source.tar.xz
    python tools/diffusers_qwen_image_reference.py reload EXPORT_DIR RECORDED.npz
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests" / "fixtures" / "hf" / "qwen-image-2.1-source"
DIFFUSERS_COMMIT = "6256aa7666cedd47443adc8f82da9a10e110b09c"
BASE = dict(patch_size=1, in_channels=8, out_channels=8, num_layers=2, attention_head_dim=12,
            num_attention_heads=2, context_in_dim=16, mlp_ratio=3, axes_dims_rope=(4, 4, 4),
            eps=1e-6, causal_condition=True)
SEED = 29


@dataclass(frozen=True)
class Case:
    """One transformer: the config controls that differ from the tiny base,
    the latent grid, each row's real prompt length out of the padded one,
    and each row's model time."""

    config: dict = field(default_factory=dict)
    grid: tuple[int, int] = (4, 4)
    lengths: tuple[int, ...] = (7, 7)
    times: tuple[float, ...] = (731.0, 42.0)


CASES: dict[str, Case] = {
    "square": Case(),
    "rect": Case(grid=(2, 6)),
    # An odd side centres its grid one position off zero.
    "odd": Case(grid=(3, 4)),
    "padded": Case(lengths=(9, 5)),
    "acausal": Case(dict(causal_condition=False), lengths=(8, 6)),
}


def rounded_frequencies(model) -> None:
    """Replace the source's float32 `exp` frequency table with the one rounded
    once from float64, which Dew builds on the host."""
    projector = model.time_text_embed.time_proj
    half = projector.timestep_dim // 2
    exponent = -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
    projector.freqs = torch.exp(exponent.double()).float()


def exp_disagreements() -> int:
    exponent = -math.log(10000) * torch.arange(128, dtype=torch.float32) / 128
    return int((torch.exp(exponent) != torch.exp(exponent.double()).float()).sum())


def transformer_call(model, packed, context, times, grid, mask=None):
    """The pipeline's own call for text-to-image: the target's image slots
    follow the prompt in `img_mask`, and the prediction is the last tokens."""
    rows, columns = grid
    slots = torch.cat([torch.zeros(1, context.shape[1], dtype=torch.bool),
                       torch.ones(1, rows * columns // 4, dtype=torch.bool)], dim=1)
    output = model(hidden_states=packed, encoder_hidden_states=context,
                   encoder_hidden_states_mask=mask, timestep=times / 1000,
                   img_shapes=[[(1, rows, columns)]] * packed.shape[0],
                   img_mask=slots.expand(packed.shape[0], -1), return_dict=False)[0]
    return output[:, -rows * columns:]


def build(name: str, case: Case, root: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    from diffusers import QwenImage21Transformer2DModel

    config = {**BASE, **case.config}
    torch.manual_seed(SEED)
    model = QwenImage21Transformer2DModel(**config).eval()
    # Every published tensor away from its initializer, the zero-centred
    # norms included, so no wiring hides behind an identity.
    generator = torch.Generator().manual_seed(SEED + 2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)
    directory = root / name / "transformer"
    model.save_pretrained(directory, safe_serialization=True)
    rows, columns = case.grid
    batch, tokens = len(case.lengths), max(case.lengths)
    packed = torch.randn((batch, rows * columns, config["in_channels"]), generator=generator)
    context = torch.randn((batch, tokens, config["context_in_dim"]), generator=generator)
    mask = torch.arange(tokens)[None] < torch.tensor(case.lengths)[:, None]
    context = context * mask[..., None]
    times = torch.tensor(case.times, dtype=torch.float32)
    probe = torch.randn((batch, rows * columns, config["out_channels"]), generator=generator)
    named = list(model.named_parameters())

    def walk():
        outputs, grad_packed, grad_context = [], [], []
        grad_params = [torch.zeros_like(value) for _, value in named]
        for row, length in enumerate(case.lengths):
            latent = packed[row:row + 1].clone().requires_grad_(True)
            text = context[row:row + 1, :length].clone().requires_grad_(True)
            output = transformer_call(model, latent, text, times[row:row + 1], case.grid)
            gradients = torch.autograd.grad((output * probe[row:row + 1]).sum(),
                                            [latent, text] + [value for _, value in named])
            outputs.append(output.detach())
            grad_packed.append(gradients[0])
            grad_context.append(torch.nn.functional.pad(gradients[1], (0, 0, 0, tokens - length)))
            for index, gradient in enumerate(gradients[2:]):
                grad_params[index] += gradient
        return (torch.cat(outputs).numpy(), torch.cat(grad_packed).numpy(),
                torch.cat(grad_context).numpy(), [gradient.numpy() for gradient in grad_params])

    _, _, _, unmodified = walk()
    rounded_frequencies(model)
    output, grad_packed, grad_context, grads = walk()
    arrays = {"packed": packed.numpy(), "context": context.numpy(), "mask": mask.numpy(),
              "times": times.numpy(), "output": output, "probe": probe.numpy(),
              "grad_packed": grad_packed, "grad_context": grad_context}
    gaps = {}
    for (key, _), gradient, theirs in zip(named, grads, unmodified, strict=True):
        arrays[f"grad_param.{key}"] = gradient
        gaps[key] = float(np.abs(gradient - theirs).max() / max(1.0, float(np.abs(gradient).max())))
    tensor, gap = max(gaps.items(), key=lambda item: item[1])
    print(f"{name}: grid {case.grid} lengths {case.lengths} |output| <= "
          f"{float(np.abs(output).max()):.4g} parameters {len(named)}; torch's own exp moves "
          f"{tensor} by {gap:.3g}")
    return arrays, {"tensor": tensor, "gap": gap}


# The pipeline half.
PIPELINE = dict(BASE)
PROMPTS = ["a red cat on a mat", "sky"]
HEIGHT, WIDTH = 64, 96
VAE = dict(base_dim=4, decoder_base_dim=6, z_dim=8, dim_mult=[1, 2, 2, 2, 2], num_res_blocks=1,
           attn_scales=[], temperal_downsample=[False, True, True, True], dropout=0.0,
           is_residual=True, in_channels=4, out_channels=4, patch_size=None,
           scale_factor_spatial=16, scale_factor_temporal=8)


def tokenizer():
    """A byte-level Qwen2 tokenizer carrying the published special tokens,
    padding side and chat template: every character is one piece."""
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import Qwen2Tokenizer

    metadata = json.loads((SOURCE / "processor" / "tokenizer_config.json").read_text())
    special = [item["content"] for item in metadata["added_tokens_decoder"].values()]
    vocab = {word: index for index, word in enumerate(sorted(ByteLevel.alphabet()))}
    for word in special:
        vocab.setdefault(word, len(vocab))
    return Qwen2Tokenizer(vocab=vocab, merges=[], eos_token=metadata["eos_token"],
                          pad_token=metadata["pad_token"], additional_special_tokens=special,
                          padding_side="left")


def processor():
    from transformers import AutoImageProcessor, AutoVideoProcessor, Qwen3VLProcessor

    source = SOURCE / "processor"
    return Qwen3VLProcessor(image_processor=AutoImageProcessor.from_pretrained(source),
                            tokenizer=tokenizer(),
                            video_processor=AutoVideoProcessor.from_pretrained(source),
                            chat_template=(source / "chat_template.jinja").read_text())


def text_encoder(tok):
    """The published Qwen3-VL config, shrunk: every field it does not name
    is the release's own."""
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = json.loads((SOURCE / "text_encoder" / "config.json").read_text())
    # The release is stored in bfloat16; the reference computes in float32.
    for record in (config, config["text_config"], config["vision_config"]):
        record.pop("dtype")
    config["text_config"].update(
        hidden_size=PIPELINE["context_in_dim"], intermediate_size=32, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8, vocab_size=len(tok),
        max_position_embeddings=512, bos_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    config["text_config"]["rope_scaling"]["mrope_section"] = [2, 1, 1]
    config["vision_config"].update(depth=2, hidden_size=16, intermediate_size=32, num_heads=2,
                                   out_hidden_size=PIPELINE["context_in_dim"],
                                   num_position_embeddings=16, deepstack_visual_indexes=[0, 1])
    for key, symbol in (("image_token_id", "<|image_pad|>"), ("video_token_id", "<|video_pad|>"),
                        ("vision_start_token_id", "<|vision_start|>"),
                        ("vision_end_token_id", "<|vision_end|>")):
        config[key] = tok.convert_tokens_to_ids(symbol)
    torch.manual_seed(SEED + 5)
    model = Qwen3VLForConditionalGeneration(Qwen3VLConfig(**config)).eval()
    generator = torch.Generator().manual_seed(SEED + 6)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return model


def build_pipeline(root: Path):
    from diffusers import (AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler,
                           QwenImage21Pipeline, QwenImage21Transformer2DModel)

    directory = root / "pipeline"
    proc = processor()
    encoder = text_encoder(proc.tokenizer)
    scheduler = json.loads((SOURCE / "scheduler" / "scheduler_config.json").read_text())
    published = json.loads((SOURCE / "vae" / "config.json").read_text())
    torch.manual_seed(SEED + 7)
    vae = AutoencoderKLQwenImage21(
        **VAE, latents_mean=published["latents_mean"][:VAE["z_dim"]],
        latents_std=published["latents_std"][:VAE["z_dim"]]).eval()
    transformer = QwenImage21Transformer2DModel(**PIPELINE).eval()
    generator = torch.Generator().manual_seed(SEED + 8)
    with torch.no_grad():
        for module in (vae, transformer):
            for parameter in module.parameters():
                parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)
    pipe = QwenImage21Pipeline(
        scheduler=FlowMatchEulerDiscreteScheduler.from_config(scheduler), vae=vae,
        text_encoder=encoder, processor=proc, transformer=transformer)
    pipe.save_pretrained(directory, safe_serialization=True)
    pipe.set_progress_bar_config(disable=True)
    # The class declares no sample size, so the directory declares the
    # geometry it is read at, which is what these keys are for.
    index = json.loads((directory / "model_index.json").read_text())
    index.update(dew_height=HEIGHT, dew_width=WIDTH)
    (directory / "model_index.json").write_text(json.dumps(index, indent=2))
    return pipe


def pipeline_record(root: Path) -> dict[str, np.ndarray]:
    pipe = build_pipeline(root)
    rounded_frequencies(pipe.transformer)
    rows, columns = HEIGHT // 16, WIDTH // 16
    generator = torch.Generator().manual_seed(SEED + 9)
    arrays: dict[str, np.ndarray] = {}
    latents = torch.randn((len(PROMPTS), rows * columns, PIPELINE["in_channels"]),
                          generator=generator)
    arrays["pipeline.x_T"] = latents.numpy()
    for row, prompt in enumerate(PROMPTS):
        with torch.no_grad():
            embeds, mask, slots = pipe.encode_prompt(prompt=prompt, device=torch.device("cpu"))
            walked = pipe(prompt=prompt, height=HEIGHT, width=WIDTH,
                          latents=latents[row:row + 1].clone(), output_type="latent").images
            images = pipe(prompt=prompt, height=HEIGHT, width=WIDTH,
                          latents=latents[row:row + 1].clone(), output_type="np").images
        assert mask is None and not bool(slots.any())
        arrays[f"pipeline.context.{row}"] = embeds.numpy()
        arrays[f"pipeline.latents.{row}"] = walked.numpy()
        arrays[f"pipeline.images.{row}"] = images
        print(f"pipeline {prompt!r}: context {tuple(embeds.shape)} latents {tuple(walked.shape)} "
              f"images {images.shape}")
    return arrays


def reload(directory: str, recorded: str) -> None:
    """Read a native export with the actual source classes.

    `directory` is a directory Dew wrote after a native training step and
    `recorded` the arrays that step left behind. The pipeline is loaded from
    those files and the actual transformer recomputes the native forward
    with the trained weights, and the actual text encoder the prompt states.
    """
    from diffusers import QwenImage21Pipeline

    arrays = np.load(recorded)
    pipe = QwenImage21Pipeline.from_pretrained(directory, torch_dtype=torch.float32,
                                               local_files_only=True)
    rounded_frequencies(pipe.transformer)
    grid = tuple(int(size) for size in arrays["grid"])
    with torch.no_grad():
        output = transformer_call(pipe.transformer.eval(), torch.from_numpy(arrays["packed"]),
                                  torch.from_numpy(arrays["context"]),
                                  torch.from_numpy(arrays["times"]), grid)
    native = arrays["native"]
    gap = float(np.abs(output.numpy() - native).max() / max(1.0, float(np.abs(native).max())))
    trained = float(np.abs(pipe.transformer.proj_out.weight.detach().numpy().T
                           - arrays["proj_out"]).max())
    print(f"reimported forward gap {gap:.3g}; trained kernel gap {trained:.3g}")
    if not (gap < 1e-5 and trained == 0.0):
        raise SystemExit("the source did not read the native update")
    print("the source reads the native update")


def bundle(directory: str, destination: str) -> None:
    """Pack the saved components and the recorded arrays for the suite."""
    import tarfile

    root = Path(directory)
    with tarfile.open(destination, "w:xz") as archive:
        for path in sorted(root.iterdir()):
            archive.add(path, arcname=path.name)
    print(f"{destination}: {Path(destination).stat().st_size / 1e6:.2f} MB")


def main(destination: str) -> None:
    import inspect

    import diffusers
    import transformers
    from diffusers import QwenImage21Pipeline

    if transformers.__version__ != "5.17.0" or torch.__version__.split("+")[0] != "2.14.0":
        raise RuntimeError("Requires transformers==5.17.0 and torch==2.14.0")
    torch.set_num_threads(2)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    cases: dict[str, dict[str, object]] = {}
    for name, case in CASES.items():
        built, unmodified = build(name, case, root)
        arrays.update({f"{name}.{key}": value for key, value in built.items()})
        cases[name] = {"config": {**BASE, **case.config}, "grid": list(case.grid),
                       "lengths": list(case.lengths), "times": list(case.times),
                       "unmodified_exp_gap": unmodified}
    arrays.update(pipeline_record(root))
    defaults = inspect.signature(QwenImage21Pipeline.__call__).parameters
    record = {"diffusers": diffusers.__version__, "diffusers_commit": DIFFUSERS_COMMIT,
              "transformers": transformers.__version__, "torch": torch.__version__,
              "base": BASE, "cases": cases,
              "torch_exp_off_by_one_ulp": {"entries": 128, "count": exp_disagreements()},
              "pipeline": {"config": PIPELINE, "vae": VAE, "prompts": PROMPTS,
                           "height": HEIGHT, "width": WIDTH,
                           "default_steps": defaults["num_inference_steps"].default,
                           "true_cfg": defaults["true_cfg_scale"].default}}
    np.savez_compressed(root / "qwen_image.npz", allow_pickle=False, **arrays)
    (root / "qwen_image.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB, {len(CASES)} cases")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "bundle":
        bundle(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 3 and sys.argv[1] == "reload":
        reload(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dew-qwen-image-reference")
