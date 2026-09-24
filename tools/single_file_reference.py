#!/usr/bin/env python3
"""What diffusers' own `from_single_file` converts a single-file checkpoint to.

    <diffusers venv>/bin/python tools/single_file_reference.py sd | sdxl | released

(~/.cache/dew/reference-venvs/qwen-image carries a diffusers whose pipelines
import.) Each records a SHA-256 of every tensor `from_single_file` converts,
per component, which tests/test_single_file.py compares Dew's conversion to.

`sd` and `sdxl` make the offline fixtures in tests/fixtures/single_file/<kind>/.
Each builds a random tiny pipeline at the block and layer counts diffusers'
reverse script hard-codes (Stable Diffusion 1.5's, SDXL base's), small widths,
and the committed tiny pipeline's 514-token tokenizer
(tests/fixtures/tiny_diffusers.tar.xz). The reverse script, fetched at
diffusers v0.34.0 rather than copied here, writes it as one original-format
file (<kind>.safetensors), next to the pipeline's configs (configs/) and what
`from_single_file(file, config=configs)` converts it back to
(from_single_file.json).

`released` records the same for Comfy-Org/stable-diffusion-v1-5-archive's
v1-5-pruned-emaonly-fp16.safetensors (2.13 GB) at a pinned revision, with the
configs diffusers infers for it (sd15_from_single_file.json).
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
SCRIPTS = {"sd": "convert_diffusers_to_original_stable_diffusion.py", "sdxl": "convert_diffusers_to_original_sdxl.py"}
COMPONENTS = {"sd": ("unet", "vae", "text_encoder"), "sdxl": ("unet", "vae", "text_encoder", "text_encoder_2")}


def digests(state) -> dict[str, str]:
    import torch

    return {name: hashlib.sha256(value.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
            + f":{value.dtype}" for name, value in state.items()}


def tiny_pipeline(kind: str, directory: Path, tokenizer_from: Path) -> None:
    """A random pipeline at the topology the reverse script's fixed block and
    layer counts describe, at small widths."""
    import torch
    from diffusers import (
        AutoencoderKL,
        DDIMScheduler,
        EulerDiscreteScheduler,
        StableDiffusionPipeline,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    torch.manual_seed(0)
    vae = AutoencoderKL(block_out_channels=(8, 8, 16, 16), layers_per_block=2, norm_num_groups=4,
                        down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
                        latent_channels=4, sample_size=16)
    text = CLIPTextModel(CLIPTextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                                        num_attention_heads=2, vocab_size=514, max_position_embeddings=77))
    tokenizer = CLIPTokenizer.from_pretrained(tokenizer_from)
    if kind == "sd":
        unet = UNet2DConditionModel(
            sample_size=8, block_out_channels=(8, 16, 16, 16), layers_per_block=2, norm_num_groups=4,
            down_block_types=("CrossAttnDownBlock2D",) * 3 + ("DownBlock2D",),
            up_block_types=("UpBlock2D",) + ("CrossAttnUpBlock2D",) * 3,
            cross_attention_dim=16, attention_head_dim=4)
        pipe = StableDiffusionPipeline(vae=vae, text_encoder=text, tokenizer=tokenizer, unet=unet,
                                       scheduler=DDIMScheduler(), safety_checker=None, feature_extractor=None,
                                       requires_safety_checker=False)
    else:
        # OpenCLIP-G's projection is as wide as its hidden size, which the
        # OpenCLIP conversion splits the fused qkv by.
        text_2 = CLIPTextModelWithProjection(CLIPTextConfig(
            hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2, vocab_size=514,
            max_position_embeddings=77, projection_dim=32, hidden_act="gelu"))
        unet = UNet2DConditionModel(
            sample_size=8, block_out_channels=(8, 16, 16), layers_per_block=2, norm_num_groups=4,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
            transformer_layers_per_block=(1, 2, 2), attention_head_dim=(2, 4, 4), use_linear_projection=True,
            cross_attention_dim=16 + 32, addition_embed_type="text_time", addition_time_embed_dim=8,
            projection_class_embeddings_input_dim=6 * 8 + 32)
        pipe = StableDiffusionXLPipeline(vae=vae, text_encoder=text, text_encoder_2=text_2, tokenizer=tokenizer,
                                         tokenizer_2=tokenizer, unet=unet, scheduler=EulerDiscreteScheduler())
    pipe.save_pretrained(directory, safe_serialization=True)


def _reverse_script(kind: str, scratch: Path):
    script = scratch / SCRIPTS[kind]
    urllib.request.urlretrieve(f"https://raw.githubusercontent.com/huggingface/diffusers/v0.34.0/scripts/"
                               f"{SCRIPTS[kind]}", script)
    spec = importlib.util.spec_from_file_location("convert", script)
    assert spec is not None and spec.loader is not None
    convert = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["convert"]
    spec.loader.exec_module(convert)
    sys.argv = argv
    return convert


def _original(kind: str, pipeline: Path, convert) -> dict:
    """The reverse script's original-format state dict (its `__main__`, which
    is not importable, composed from its functions the same way)."""
    from safetensors.torch import load_file

    unet = convert.convert_unet_state_dict(load_file(pipeline / "unet" / "diffusion_pytorch_model.safetensors"))
    vae = convert.convert_vae_state_dict(load_file(pipeline / "vae" / "diffusion_pytorch_model.safetensors"))
    # A released file names CLIP-L as transformers' CLIPTextModel did before
    # 5.6 flattened it: under `text_model.`.
    text = {name if name.startswith("text_model.") else f"text_model.{name}": value
            for name, value in load_file(pipeline / "text_encoder" / "model.safetensors").items()}
    state = {**{f"model.diffusion_model.{k}": v for k, v in unet.items()},
             **{f"first_stage_model.{k}": v for k, v in vae.items()}}
    if kind == "sd":
        return {**state, **{f"cond_stage_model.transformer.{k}": v
                            for k, v in convert.convert_text_enc_state_dict(text).items()}}
    text_2 = convert.convert_openclip_text_enc_state_dict(
        load_file(pipeline / "text_encoder_2" / "model.safetensors"))
    text_2 = {f"conditioner.embedders.1.model.{k}": v for k, v in text_2.items()}
    text_2["conditioner.embedders.1.model.text_projection"] = text_2.pop(
        "conditioner.embedders.1.model.text_projection.weight").T.contiguous()
    return {**state, **{f"conditioner.embedders.0.transformer.{k}": v
                        for k, v in convert.convert_openai_text_enc_state_dict(text).items()}, **text_2}


def tiny(kind: str) -> None:
    import torch
    from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
    from safetensors.torch import save_file

    scratch = Path(tempfile.mkdtemp())
    with tarfile.open(ROOT / "tests" / "fixtures" / "tiny_diffusers.tar.xz") as archive:
        archive.extractall(scratch / "committed", filter="data")
    pipeline = scratch / kind
    tiny_pipeline(kind, pipeline, scratch / "committed" / "sd" / "tokenizer")
    state = _original(kind, pipeline, _reverse_script(kind, scratch))
    out = FIXTURE / kind
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(pipeline, out / "configs", ignore=shutil.ignore_patterns("*.safetensors"))
    save_file({k: v.contiguous() for k, v in state.items()}, out / f"{kind}.safetensors")
    loader = StableDiffusionPipeline if kind == "sd" else StableDiffusionXLPipeline
    pipe = loader.from_single_file(out / f"{kind}.safetensors", config=str(out / "configs"),
                                   torch_dtype=torch.float32)
    record = {name: digests(getattr(pipe, name).state_dict()) for name in COMPONENTS[kind]}
    (out / "from_single_file.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    print(out, {name: len(values) for name, values in record.items()})


def released() -> None:
    import torch
    from diffusers import StableDiffusionPipeline
    from huggingface_hub import hf_hub_download

    repo, name, revision = RELEASED
    # torch_dtype keeps each tensor in the dtype the file stores it in.
    pipe = StableDiffusionPipeline.from_single_file(hf_hub_download(repo, name, revision=revision),
                                                    torch_dtype=torch.float16)
    record = {component: digests(getattr(pipe, component).state_dict()) for component in COMPONENTS["sd"]}
    (FIXTURE / "sd15_from_single_file.json").write_text(json.dumps(record, indent=1, sort_keys=True))
    print({component: len(values) for component, values in record.items()})


if __name__ == "__main__":
    released() if sys.argv[1] == "released" else tiny(sys.argv[1])
