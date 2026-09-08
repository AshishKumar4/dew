"""Actual Diffusers 0.34.0 SD3 objects, saved and walked, for tests/fixtures.

G0 and G1 of the SD3 qualification: tiny `SD3Transformer2DModel` instances are
constructed, their stored position buffer is perturbed away from its own sin/cos
initializer so a load has to read it, and each is saved with
`save_pretrained` - its real config and its real safetensors, nothing
fabricated. Every case then runs the actual forward and records the output, the
vector-Jacobian product against a fixed cotangent for the latent, the text
tokens and the pooled vector, and the gradient of every parameter, all in
float32, which is what the native model runs in.

The variants are the ones whose block wiring differs: plain, `qk_norm`,
SD3.5's dual attention, and a rectangular latent whose patch grid is smaller
than the position buffer, so the centred crop is exercised.

Run in the isolated reference environment on CPU:

    PYTHONPATH=src python tools/diffusers_sd3_reference.py OUTPUT_DIR
"""

from __future__ import annotations

import inspect
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np
import torch
import transformers.utils as transformers_utils

# Diffusers 0.34.0's pipeline modules import two names Transformers dropped
# after 4.x. The pipelines under test compute the sigma preparation this tool
# records, so the names are restored rather than that preparation copied.
for _name, _value in (("FLAX_WEIGHTS_NAME", "flax_model.msgpack"),
                      ("WEIGHTS_INDEX_NAME", "pytorch_model.bin.index.json")):
    if not hasattr(transformers_utils, _name):
        setattr(transformers_utils, _name, _value)

BASE = dict(sample_size=16, patch_size=2, in_channels=4, num_layers=2, attention_head_dim=8,
            num_attention_heads=2, joint_attention_dim=12, caption_projection_dim=16,
            pooled_projection_dim=10, out_channels=4, pos_embed_max_size=8)
TOKENS = 5
SEED = 19


@dataclass(frozen=True)
class Case:
    """One transformer: the config controls that differ from the tiny base and
    the latent grid the walk runs at."""

    config: dict = field(default_factory=dict)
    latent: tuple[int, int] = (8, 8)
    batch: int = 2


CASES: dict[str, Case] = {
    "plain": Case(),
    "qknorm": Case(dict(qk_norm="rms_norm")),
    "dual": Case(dict(qk_norm="rms_norm", dual_attention_layers=(0,))),
    "rect_cropped": Case(dict(qk_norm="rms_norm"), latent=(12, 6)),
    "deep": Case(dict(num_layers=3, dual_attention_layers=(1,)), latent=(6, 10)),
}


def build(name: str, case: Case, root: Path) -> dict[str, np.ndarray]:
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel

    torch.manual_seed(SEED)
    model = SD3Transformer2DModel(**{**BASE, **case.config}).eval()
    with torch.no_grad():
        # The buffer ships sin/cos values; perturbing the stored array means a
        # load that rebuilt it from the initializer would not match.
        model.pos_embed.pos_embed.add_(
            torch.randn_like(model.pos_embed.pos_embed, dtype=torch.float32) * 0.05)
    directory = root / name / "transformer"
    model.save_pretrained(directory, safe_serialization=True)
    generator = torch.Generator().manual_seed(SEED + 1)
    height, width = case.latent
    latent = torch.randn((case.batch, BASE["in_channels"], height, width), generator=generator,
                         dtype=torch.float32, requires_grad=True)
    context = torch.randn((case.batch, TOKENS, BASE["joint_attention_dim"]), generator=generator,
                          dtype=torch.float32, requires_grad=True)
    pooled = torch.randn((case.batch, BASE["pooled_projection_dim"]), generator=generator,
                         dtype=torch.float32, requires_grad=True)
    times = torch.tensor([731.0, 42.0][: case.batch], dtype=torch.float32)
    output = model(hidden_states=latent, encoder_hidden_states=context,
                   pooled_projections=pooled, timestep=times, return_dict=False)[0]
    probe = torch.randn(output.shape, generator=generator, dtype=torch.float32)
    named = [(key, value) for key, value in model.named_parameters()]
    grads = torch.autograd.grad((output * probe).sum(), [latent, context, pooled]
                                + [value for _, value in named])
    arrays = {
        "config": np.asarray(json.dumps(json.loads((directory / "config.json").read_text()))),
        "latent": latent.detach().numpy(), "context": context.detach().numpy(),
        "pooled": pooled.detach().numpy(), "times": times.numpy(),
        "output": output.detach().numpy(), "probe": probe.numpy(),
        "grad_latent": grads[0].numpy(), "grad_context": grads[1].numpy(),
        "grad_pooled": grads[2].numpy(),
        "position_buffer": model.pos_embed.pos_embed.detach().numpy(),
    }
    for (key, _), gradient in zip(named, grads[3:]):
        arrays[f"grad_param.{key}"] = gradient.numpy()
    print(f"{name}: latent {tuple(latent.shape)} tokens {TOKENS} "
          f"|output| <= {float(output.abs().max()):.4g} parameters {len(named)}")
    return arrays


FLOW_CASES: dict[str, dict] = {
    # The files SD3 and SD3.5 ship, plus the controls the class declares that
    # change where its sigmas land.
    "flow.plain": dict(num_train_timesteps=1000),
    "flow.shift3": dict(num_train_timesteps=1000, shift=3.0),
    "flow.dynamic": dict(num_train_timesteps=1000, use_dynamic_shifting=True, base_shift=0.5,
                         max_shift=1.15, base_image_seq_len=256, max_image_seq_len=4096),
    "flow.linear_dynamic": dict(num_train_timesteps=1000, use_dynamic_shifting=True,
                                time_shift_type="linear"),
    "flow.terminal": dict(num_train_timesteps=1000, shift=3.0, shift_terminal=0.1),
    "flow.karras": dict(num_train_timesteps=1000, shift=3.0, use_karras_sigmas=True),
    "flow.exponential": dict(num_train_timesteps=1000, use_exponential_sigmas=True),
}
# Steps and the latent token counts a pipeline computes for a resolution: SD3
# counts (h/p)(w/p) patches, Flux (h/2)(w/2) packed positions.
FLOW_STEPS = (4, 7)
FLOW_TOKENS = (256, 1024, 4096)
DATA_STD = 0.4


def flow_walk(scheduler, latent, sigmas):
    """The Euler walk the source takes: one model call per interval, the
    velocity of the optimal transport path for x0 ~ N(0, DATA_STD^2)."""
    latents = []
    for index, time in enumerate(scheduler.timesteps):
        sigma = scheduler.sigmas[index]
        # E[eps - x0 | x_t] for a Gaussian data law under x = (1-s) x0 + s eps
        variance = (1 - sigma) ** 2 * DATA_STD ** 2 + sigma ** 2
        x0 = latent * (1 - sigma) * DATA_STD ** 2 / variance
        eps = latent * sigma / variance
        velocity = eps - x0
        latent = scheduler.step(velocity, time, latent, return_dict=False)[0]
        latents.append(latent)
    return latents


def flow_record(name: str, config: dict) -> dict[str, np.ndarray]:
    from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
        FlowMatchEulerDiscreteScheduler)
    from diffusers.pipelines.flux.pipeline_flux import calculate_shift

    arrays: dict[str, np.ndarray] = {}
    for steps in FLOW_STEPS:
        for tokens in FLOW_TOKENS:
            for origin in ("scheduler", "linspace"):
                scheduler = FlowMatchEulerDiscreteScheduler(**config)
                kwargs: dict = {}
                if config.get("use_dynamic_shifting"):
                    kwargs["mu"] = calculate_shift(
                        tokens, config.get("base_image_seq_len", 256),
                        config.get("max_image_seq_len", 4096), config.get("base_shift", 0.5),
                        config.get("max_shift", 1.15))
                if origin == "linspace":
                    kwargs["sigmas"] = np.linspace(1.0, 1 / steps, steps)
                elif config.get("use_dynamic_shifting"):
                    # The SD3 pipeline only passes mu; its sigmas stay the
                    # scheduler's own.
                    pass
                scheduler.set_timesteps(steps, **kwargs)
                tag = f"{steps}.{tokens}.{origin}"
                generator = torch.Generator().manual_seed(SEED + steps)
                latent = torch.randn((2, 3, 4), generator=generator, dtype=torch.float32,
                                     requires_grad=True)
                cotangent = torch.randn(latent.shape, generator=generator, dtype=torch.float32)
                latents = flow_walk(scheduler, latent, scheduler.sigmas)
                (gradient,) = torch.autograd.grad((latents[-1] * cotangent).sum(), latent)
                arrays[f"{tag}.sigmas"] = scheduler.sigmas.numpy()
                arrays[f"{tag}.times"] = scheduler.timesteps.numpy()
                arrays[f"{tag}.mu"] = np.asarray(kwargs.get("mu", np.nan), np.float64)
                arrays[f"{tag}.x_T"] = latent.detach().numpy()
                arrays[f"{tag}.latents"] = np.stack([v.detach().numpy() for v in latents])
                arrays[f"{tag}.cotangent"] = cotangent.numpy()
                arrays[f"{tag}.grad"] = gradient.numpy()
    print(f"{name}: {len(FLOW_STEPS) * len(FLOW_TOKENS) * 2} grids")
    return arrays


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
        cases[name] = {"config": {**BASE, **case.config}, "latent": list(case.latent),
                       "batch": case.batch}
    flow: dict[str, np.ndarray] = {}
    for name, config in FLOW_CASES.items():
        flow[f"{name}.config"] = np.asarray(json.dumps(
            {"_class_name": "FlowMatchEulerDiscreteScheduler", **config}))
        for key, value in flow_record(name, config).items():
            flow[f"{name}.{key}"] = value
    record["flow"] = {"cases": sorted(FLOW_CASES), "steps": list(FLOW_STEPS),
                      "tokens": list(FLOW_TOKENS), "data_std": DATA_STD}
    pipeline = pipeline_record(root)
    from diffusers import StableDiffusion3Pipeline

    defaults = inspect.signature(StableDiffusion3Pipeline.__call__).parameters
    record["pipeline"] = {"config": PIPELINE, "prompts": PROMPTS, "negatives": NEGATIVES,
                          "t5_tokens": T5_TOKENS, "steps": PIPELINE_STEPS,
                          "guidance": PIPELINE_GUIDANCE, "size": PIPELINE_SIZE,
                          "default_steps": defaults["num_inference_steps"].default,
                          "default_guidance": defaults["guidance_scale"].default}
    np.savez_compressed(root / "sd3_pipeline.npz", allow_pickle=False, **pipeline)
    np.savez_compressed(root / "sd3_flow.npz", allow_pickle=False, **flow)
    np.savez_compressed(root / "sd3_transformer.npz", allow_pickle=False, **arrays)
    (root / "sd3_transformer.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB, {len(CASES)} cases")


def reload(directory: str, recorded: str) -> None:
    """Read a native export with the actual source classes.

    `directory` is a directory Dew wrote after a native training step and
    `recorded` the arrays that step left behind. The published pipeline is
    loaded from those files - the transformer, both CLIP towers, the T5 tower,
    its tokenizer and the scheduler - and the actual transformer recomputes
    the forward the native model computed with the trained weights.
    """
    from diffusers import StableDiffusion3Pipeline

    arrays = np.load(recorded)
    pipe = StableDiffusion3Pipeline.from_pretrained(directory, torch_dtype=torch.float32,
                                                   local_files_only=True)
    model = pipe.transformer.eval()
    with torch.no_grad():
        output = model(hidden_states=torch.from_numpy(arrays["latent"]).permute(0, 3, 1, 2),
                       encoder_hidden_states=torch.from_numpy(arrays["context"]),
                       pooled_projections=torch.from_numpy(arrays["pooled"]),
                       timestep=torch.from_numpy(arrays["times"]), return_dict=False)[0]
    native = arrays["native"]
    theirs = output.permute(0, 2, 3, 1).numpy()
    gap = float(np.abs(theirs - native).max() / max(1.0, float(np.abs(native).max())))
    buffer = float(np.abs(model.pos_embed.pos_embed.detach().numpy() - arrays["buffer"]).max())
    trained = float(np.abs(model.proj_out.weight.detach().numpy().T - arrays["proj_out"]).max())
    print(f"reimported forward gap {gap:.3g}; position buffer gap {buffer:.3g}; "
          f"trained kernel gap {trained:.3g}")
    if not (gap < 1e-5 and buffer == 0.0 and trained == 0.0):
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




# G2/G4/G5: one tiny SD3 pipeline, saved the way a published checkpoint is
# saved, walked through the source's own prompt encoding and its own call.
PIPELINE = dict(sample_size=8, patch_size=2, in_channels=16, num_layers=2, attention_head_dim=8,
                num_attention_heads=2, joint_attention_dim=32, caption_projection_dim=16,
                pooled_projection_dim=10, out_channels=16, pos_embed_max_size=16,
                qk_norm="rms_norm", dual_attention_layers=(0,))
# Distinct text in every slot: the source routes `prompt` to CLIP-L,
# `prompt_2` to CLIP-G and `prompt_3` to T5, so identical prompts would hide a
# native encoder that crossed them.
PROMPTS = [{"text": "a red cat", "second": "a blue dog", "third": "a green bird"},
           {"text": "tiny photo", "second": "a red dog", "third": "a blue cat"}]
NEGATIVES = [{"text": "", "second": "", "third": ""},
             {"text": "a green cat", "second": "tiny dog", "third": "a red bird"}]
T5_WORDS = ("<pad>", "</s>", "<unk>", "a", "red", "blue", "green", "cat", "dog", "bird",
            "tiny", "photo")
T5_TOKENS = 256  # the source's own `max_sequence_length` default
PIPELINE_STEPS = 3
PIPELINE_GUIDANCE = 3.5
PIPELINE_SIZE = 16


def clip_tokenizers(root: Path):
    """The committed tiny CLIP tokenizer files, which are real published
    byte-level BPE vocabularies rather than anything this tool invents."""
    import tarfile
    from transformers import CLIPTokenizer

    fixture = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "tiny_diffusers.tar.xz"
    with tarfile.open(fixture) as archive:
        for name, source in (("tokenizer", "xl/tokenizer"), ("tokenizer_2", "xl/tokenizer_2")):
            members = [member for member in archive.getmembers()
                       if member.name.startswith(source + "/")]
            for member in members:
                member.name = name + "/" + member.name.split("/")[-1]
            archive.extractall(root, members=members, filter="data")
    return [CLIPTokenizer.from_pretrained(root / name) for name in ("tokenizer", "tokenizer_2")]


def t5_tokenizer():
    """A T5-style fast tokenizer over a fixed word vocabulary: whitespace
    pieces and the sentinel the source's `add_special_tokens` appends."""
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {word: index for index, word in enumerate(T5_WORDS)}
    inner = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    inner.pre_tokenizer = pre_tokenizers.Whitespace()
    inner.post_processor = processors.TemplateProcessing(
        single="$A </s>", special_tokens=[("</s>", vocab["</s>"])])
    return PreTrainedTokenizerFast(tokenizer_object=inner, eos_token="</s>", unk_token="<unk>",
                                   pad_token="<pad>", model_max_length=T5_TOKENS)


def build_pipeline(root: Path, *, t5: bool):
    from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
    from transformers import (CLIPTextConfig, CLIPTextModelWithProjection, T5Config,
                              T5EncoderModel)

    directory = root / ("pipeline" if t5 else "pipeline_no_t5")
    tokenizers = clip_tokenizers(directory)
    torch.manual_seed(SEED + 5)
    widths = (8, 12)
    towers = [CLIPTextModelWithProjection(CLIPTextConfig(
        vocab_size=len(tokenizer.get_vocab()), hidden_size=width, intermediate_size=2 * width,
        num_hidden_layers=2, num_attention_heads=2, projection_dim=5,
        max_position_embeddings=tokenizer.model_max_length)).eval()
        for tokenizer, width in zip(tokenizers, widths)]
    encoder_3 = tokenizer_3 = None
    if t5:
        tokenizer_3 = t5_tokenizer()
        encoder_3 = T5EncoderModel(T5Config(
            vocab_size=len(T5_WORDS), d_model=PIPELINE["joint_attention_dim"], d_ff=64,
            num_layers=2, num_heads=2, d_kv=16, relative_attention_num_buckets=8,
            feed_forward_proj="gated-gelu")).eval()
    vae = AutoencoderKL(in_channels=3, out_channels=3, block_out_channels=(4, 8),
                        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
                        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
                        layers_per_block=1, latent_channels=PIPELINE["in_channels"],
                        norm_num_groups=2, sample_size=PIPELINE_SIZE, shift_factor=0.0609,
                        scaling_factor=1.5305, use_quant_conv=False,
                        use_post_quant_conv=False).eval()
    transformer = SD3Transformer2DModel(**PIPELINE).eval()
    with torch.no_grad():
        transformer.pos_embed.pos_embed.add_(
            torch.randn_like(transformer.pos_embed.pos_embed, dtype=torch.float32) * 0.05)
    pipe = StableDiffusion3Pipeline(
        transformer=transformer, scheduler=FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=3.0),
        vae=vae, text_encoder=towers[0], tokenizer=tokenizers[0],
        text_encoder_2=towers[1], tokenizer_2=tokenizers[1],
        text_encoder_3=encoder_3, tokenizer_3=tokenizer_3)
    pipe.save_pretrained(directory, safe_serialization=True)
    pipe.set_progress_bar_config(disable=True)
    return pipe, directory


def pipeline_record(root: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for t5 in (True, False):
        label = "pipeline" if t5 else "pipeline_no_t5"
        pipe, directory = build_pipeline(root, t5=t5)
        with torch.no_grad():
            embeds, negative_embeds, pooled, negative_pooled = pipe.encode_prompt(
                prompt=[row["text"] for row in PROMPTS],
                prompt_2=[row["second"] for row in PROMPTS],
                prompt_3=[row["third"] for row in PROMPTS],
                negative_prompt=[row["text"] for row in NEGATIVES],
                negative_prompt_2=[row["second"] for row in NEGATIVES],
                negative_prompt_3=[row["third"] for row in NEGATIVES],
                device=torch.device("cpu"), num_images_per_prompt=1,
                do_classifier_free_guidance=True)
        arrays[f"{label}.context"] = embeds.numpy()
        arrays[f"{label}.pooled"] = pooled.numpy()
        arrays[f"{label}.negative_context"] = negative_embeds.numpy()
        arrays[f"{label}.negative_pooled"] = negative_pooled.numpy()
        generator = torch.Generator().manual_seed(SEED + 7)
        latents = torch.randn((len(PROMPTS), PIPELINE["in_channels"],
                               PIPELINE_SIZE // 2 // PIPELINE["patch_size"] * PIPELINE["patch_size"],
                               PIPELINE_SIZE // 2), generator=generator, dtype=torch.float32)
        with torch.no_grad():
            walked = pipe(prompt=[row["text"] for row in PROMPTS],
                          prompt_2=[row["second"] for row in PROMPTS],
                          prompt_3=[row["third"] for row in PROMPTS],
                          negative_prompt=[row["text"] for row in NEGATIVES],
                          negative_prompt_2=[row["second"] for row in NEGATIVES],
                          negative_prompt_3=[row["third"] for row in NEGATIVES],
                          num_inference_steps=PIPELINE_STEPS, guidance_scale=PIPELINE_GUIDANCE,
                          height=PIPELINE_SIZE, width=PIPELINE_SIZE,
                          latents=latents.clone(), output_type="latent").images
            images = pipe(prompt=[row["text"] for row in PROMPTS],
                          prompt_2=[row["second"] for row in PROMPTS],
                          prompt_3=[row["third"] for row in PROMPTS],
                          negative_prompt=[row["text"] for row in NEGATIVES],
                          negative_prompt_2=[row["second"] for row in NEGATIVES],
                          negative_prompt_3=[row["third"] for row in NEGATIVES],
                          num_inference_steps=PIPELINE_STEPS, guidance_scale=PIPELINE_GUIDANCE,
                          height=PIPELINE_SIZE, width=PIPELINE_SIZE,
                          latents=latents.clone(), output_type="np").images
        with torch.no_grad():
            # The same call with its policy omitted: the pipeline's own step
            # count and guidance scale, which a native task has to take too.
            omitted = pipe(prompt=[row["text"] for row in PROMPTS],
                           prompt_2=[row["second"] for row in PROMPTS],
                           prompt_3=[row["third"] for row in PROMPTS],
                           negative_prompt=[row["text"] for row in NEGATIVES],
                           negative_prompt_2=[row["second"] for row in NEGATIVES],
                           negative_prompt_3=[row["third"] for row in NEGATIVES],
                           height=PIPELINE_SIZE, width=PIPELINE_SIZE,
                           latents=latents.clone(), output_type="latent").images
        arrays[f"{label}.default_latents"] = omitted.permute(0, 2, 3, 1).numpy()
        arrays[f"{label}.x_T"] = latents.permute(0, 2, 3, 1).numpy()
        arrays[f"{label}.latents"] = walked.permute(0, 2, 3, 1).numpy()
        arrays[f"{label}.images"] = images
        print(f"{label}: context {tuple(embeds.shape)} pooled {tuple(pooled.shape)} "
              f"latents {tuple(walked.shape)} |image| <= {float(np.abs(images).max()):.4g}")
    return arrays

if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "bundle":
        bundle(sys.argv[2], sys.argv[3])
    elif len(sys.argv) > 2 and sys.argv[1] == "reload":
        reload(sys.argv[2], sys.argv[3])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dew-sd3-reference")
