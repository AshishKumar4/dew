#!/usr/bin/env python3
"""What diffusers' own `from_single_file` converts a single-file checkpoint to.

    <diffusers venv>/bin/python tools/single_file_reference.py tiny
    <diffusers venv>/bin/python tools/single_file_reference.py released

(~/.cache/dew/reference-venvs/qwen-image carries a diffusers whose pipelines
import.) Both record a SHA-256 of every tensor `from_single_file` converts,
per component, which tests/test_single_file.py compares Dew's conversion to.

`released` converts Comfy-Org/stable-diffusion-v1-5-archive's
v1-5-pruned-emaonly-fp16.safetensors (2.13 GB) at a pinned revision, with the
configs diffusers infers for it, into
tests/fixtures/single_file/sd15_from_single_file.json.

`tiny` makes the offline fixture.
Builds a random tiny Stable Diffusion 1.x pipeline at SD 1.5's block and
layer counts (the reverse script hard-codes them), with the committed tiny
pipeline's 514-token tokenizer (tests/fixtures/tiny_diffusers.tar.xz), and converts it into one
LDM-format file with diffusers' own reverse script, scripts/convert_diffusers_to_original_stable_diffusion.py at
v0.34.0 (fetched, not copied here), and records what diffusers'
`from_single_file(file, config=<that pipeline>)` converts it back to: a
SHA-256 per stored tensor, per component. Writes tests/fixtures/single_file/:
tiny-sd-ldm.safetensors, the pipeline's configs and tokenizer (configs/),
and from_single_file.json.
"""

import hashlib
import importlib.util
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "single_file"
RELEASED = ("Comfy-Org/stable-diffusion-v1-5-archive", "v1-5-pruned-emaonly-fp16.safetensors",
            "9cfd069101959ca3828bf9c04a4419870832b74f")
COMPONENTS = ("unet", "vae", "text_encoder")
SCRIPT = ("https://raw.githubusercontent.com/huggingface/diffusers/v0.34.0/"
          "scripts/convert_diffusers_to_original_stable_diffusion.py")


def digests(state) -> dict[str, str]:
    return {name: hashlib.sha256(value.contiguous().cpu().numpy().tobytes()).hexdigest() + f":{value.dtype}"
            for name, value in state.items()}


def tiny_pipeline(directory: Path, tokenizer_from: Path) -> None:
    """A random Stable Diffusion 1.x pipeline at SD 1.5's topology, the one
    the reverse script's fixed block and layer counts describe, at small widths."""
    import torch
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    torch.manual_seed(0)
    unet = UNet2DConditionModel(
        sample_size=8, block_out_channels=(8, 16, 16, 16), layers_per_block=2, norm_num_groups=4,
        down_block_types=("CrossAttnDownBlock2D",) * 3 + ("DownBlock2D",),
        up_block_types=("UpBlock2D",) + ("CrossAttnUpBlock2D",) * 3,
        cross_attention_dim=16, attention_head_dim=4)
    vae = AutoencoderKL(block_out_channels=(8, 8, 16, 16), layers_per_block=2, norm_num_groups=4,
                        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                        latent_channels=4, sample_size=16)
    text = CLIPTextModel(CLIPTextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                        num_attention_heads=2, vocab_size=514, max_position_embeddings=77))
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_from)
    pipe = StableDiffusionPipeline(vae=vae, text_encoder=text, tokenizer=tokenizer, unet=unet,
                                   scheduler=DDIMScheduler(), safety_checker=None, feature_extractor=None,
                                   requires_safety_checker=False)
    pipe.save_pretrained(directory, safe_serialization=True)


def released() -> None:
    import torch
    from diffusers import StableDiffusionPipeline
    from huggingface_hub import hf_hub_download

    repo, name, revision = RELEASED
    # torch_dtype keeps each tensor in the dtype the file stores it in.
    pipe = StableDiffusionPipeline.from_single_file(hf_hub_download(repo, name, revision=revision),
                                                    torch_dtype=torch.float16)
    record = {component: digests(getattr(pipe, component).state_dict()) for component in COMPONENTS}
    (FIXTURE / "sd15_from_single_file.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    print({component: len(values) for component, values in record.items()})


def tiny() -> None:
    import torch
    from diffusers import StableDiffusionPipeline
    from safetensors.torch import load_file, save_file

    scratch = Path(tempfile.mkdtemp())
    with tarfile.open(ROOT / "tests" / "fixtures" / "tiny_diffusers.tar.xz") as archive:
        archive.extractall(scratch / "committed", filter="data")
    pipeline = scratch / "sd"
    tiny_pipeline(pipeline, scratch / "committed" / "sd" / "tokenizer")
    script = scratch / "convert.py"
    urllib.request.urlretrieve(SCRIPT, script)
    spec = importlib.util.spec_from_file_location("convert", script)
    assert spec is not None and spec.loader is not None
    convert = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["convert"]
    spec.loader.exec_module(convert)
    sys.argv = argv
    unet = convert.convert_unet_state_dict(load_file(pipeline / "unet" / "diffusion_pytorch_model.safetensors"))
    vae = convert.convert_vae_state_dict(load_file(pipeline / "vae" / "diffusion_pytorch_model.safetensors"))
    # A released SD 1.x file names the text encoder as transformers' CLIPTextModel
    # did before 5.6 flattened it: under `text_model.`.
    text = {name if name.startswith("text_model.") else f"text_model.{name}": value for name, value in
            convert.convert_text_enc_state_dict(load_file(pipeline / "text_encoder" / "model.safetensors")).items()}
    state = {**{f"model.diffusion_model.{k}": v for k, v in unet.items()},
             **{f"first_stage_model.{k}": v for k, v in vae.items()},
             **{f"cond_stage_model.transformer.{k}": v for k, v in text.items()}}
    FIXTURE.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in state.items()}, FIXTURE / "tiny-sd-ldm.safetensors")
    configs = FIXTURE / "configs"
    shutil.rmtree(configs, ignore_errors=True)
    shutil.copytree(pipeline, configs, ignore=shutil.ignore_patterns("*.safetensors"))
    pipe = StableDiffusionPipeline.from_single_file(FIXTURE / "tiny-sd-ldm.safetensors", config=str(configs),
                                                    torch_dtype=torch.float32)
    record = {name: digests(getattr(pipe, name).state_dict()) for name in COMPONENTS}
    (FIXTURE / "from_single_file.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    print(FIXTURE, {name: len(values) for name, values in record.items()})


if __name__ == "__main__":
    {"tiny": tiny, "released": released}[sys.argv[1]]()
