#!/usr/bin/env python3
"""Write the fixtures of PEFT and Diffusers reading the LoRA adapters Dew writes.

Each case adapts a committed tiny source through Dew, moves every factor by
a seeded draw (a stand-in for training that every machine reproduces bit
for bit) and writes the adapter with `LoRA.save`: PEFT's directory for a
decoder, the Diffusers file for a pipeline. Its consumer then reads exactly
those files and runs the adapted and the merged model on fixed inputs, in
float32 and in float64. Two halves, in two environments:

- `export DIR` (the Dew environment) writes every case's adapter into
  DIR/<case>; tests/test_lora.py reruns `export_case` and requires the same
  files, byte for byte.
- `consume DIR` (a PEFT 0.20.0 environment over the Dew one, whose
  transformers 5.16.1, Diffusers 0.34.0 and torch it uses) loads each
  adapter, requires every tensor in the file to land in the model and no
  other adapter state, and writes tests/fixtures/lora_exports/<case>.npz:
  the inputs, the adapted and the merged (PEFT's `merge_and_unload`,
  Diffusers' `fuse_lora`) outputs in float32 and float64
  (`diffusers_wan_reference.float64`), the output without the adapter, and
  the SHA-256 of every adapter file (`digest`).

The decoders are tests/fixtures/hf/llama-tiny under PEFT's `PeftModel`;
the pipelines are the tiny SD (its PyTorch UNet), FLUX and SD3 sources
under their Diffusers pipelines' `load_lora_weights`.

    uv venv --python .venv/bin/python ~/.cache/dew/reference-venvs/peft
    echo $PWD/.venv/lib/python3.12/site-packages \\
        > ~/.cache/dew/reference-venvs/peft/lib/python3.12/site-packages/dew-env.pth
    uv pip install --python ~/.cache/dew/reference-venvs/peft/bin/python --no-deps peft==0.20.0
    PYTHONPATH=src python tools/lora_export_reference.py export DIR
    ~/.cache/dew/reference-venvs/peft/bin/python tools/lora_export_reference.py consume DIR
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import sys
import tarfile
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "lora_exports"
LLAMA = ROOT / "tests" / "fixtures" / "hf" / "llama-tiny"
PATTERNED = ROOT / "tests" / "fixtures" / "lora" / "llama-tiny" / "adapter"
DENOISER = ("to_q", "to_k", "to_v", "to_out.0")


@dataclasses.dataclass(frozen=True)
class Case:
    source: str
    """"llama", or the pipeline family: "sd", "flux" or "sd3"."""
    seed: int
    rank: int = 4
    modules: tuple[str, ...] = DENOISER
    alpha: float | None = None
    rslora: bool = False
    dropout: float = 0.0
    patterned: bool = False
    """Start from the committed PEFT adapter, whose rank and alpha patterns
    Dew then writes back, rather than a fresh one."""


CASES = {
    "llama": Case("llama", 0, modules=("q_proj", "v_proj", "down_proj"), alpha=8.0, dropout=0.1),
    "llama-rslora": Case("llama", 1, rank=2, modules=("k_proj", "o_proj", "gate_proj", "up_proj"), alpha=6.0,
                         rslora=True),
    "llama-patterns": Case("llama", 2, patterned=True),
    "sd": Case("sd", 3, alpha=6.0),
    "flux": Case("flux", 4, alpha=8.0, rslora=True),
    "sd3": Case("sd3", 5, rank=2, alpha=3.0),
}


@contextlib.contextmanager
def source_directory(source: str) -> Iterator[Path]:
    """The committed source of a case, extracted for the duration of the block."""
    if source == "llama":
        yield LLAMA
        return
    with tempfile.TemporaryDirectory() as extracted:
        if source == "sd":
            with tarfile.open(ROOT / "tests/fixtures/tiny_diffusers.tar.xz") as archive:
                archive.extractall(extracted, members=[m for m in archive.getmembers()
                                                       if m.name.startswith("sd/")], filter="data")
            yield Path(extracted) / "sd"
        else:
            with tarfile.open(ROOT / f"tests/fixtures/{source}_source.tar.xz") as archive:
                archive.extractall(extracted, filter="data")
            yield Path(extracted) / "pipeline"


def moved(variables, seed: int):
    """`variables` with every LoRA factor moved by 0.2 of a seeded normal draw."""
    import jax

    flat = jax.tree_util.tree_flatten_with_path(variables)[0]
    count = sum(path[-1].key in ("lora_A", "lora_B") for path, _ in flat)
    keys = iter(jax.random.split(jax.random.key(seed), count))
    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: (leaf + 0.2 * jax.random.normal(next(keys), leaf.shape, leaf.dtype)
                            if path[-1].key in ("lora_A", "lora_B") else leaf), variables)


def export_case(case: Case, source, directory: Path):
    """Adapt the loaded `source` as `case` says, move the factors and write
    the adapter into `directory`; returns the adapter and its variables."""
    from dew.lora import LoRA

    if case.patterned:
        adapter = LoRA.load(source.model, source.variables, PATTERNED, layouts=source.layouts)
    else:
        adapter = source.adapt(LoRA(rank=case.rank, modules=case.modules, alpha=case.alpha,
                                    rslora=case.rslora, dropout=case.dropout), key=case.seed).adapter
    variables = moved(adapter.variables, 100 + case.seed)
    adapter.save(variables, directory)
    return adapter, variables


def digest(path: Path) -> str:
    """The SHA-256 of a file's bytes, or of a safetensors file's contents:
    safetensors 0.8.0 writes the header's metadata from a hash map, in an
    order that changes from process to process, and no reader sees it. The
    contents are every tensor's name, dtype, shape and data, and the
    metadata, in sorted order."""
    if path.suffix != ".safetensors":
        return hashlib.sha256(path.read_bytes()).hexdigest()
    from safetensors import safe_open

    hashed = hashlib.sha256()
    with safe_open(str(path), "numpy") as file:
        hashed.update(json.dumps(file.metadata(), sort_keys=True).encode())
        for name in sorted(file.keys()):
            tensor = file.get_tensor(name)
            hashed.update(f"{name} {tensor.dtype} {tensor.shape}".encode())
            hashed.update(np.ascontiguousarray(tensor).tobytes())
    return hashed.hexdigest()


def digests(directory: Path) -> dict[str, str]:
    return {path.name: digest(path) for path in sorted(directory.iterdir()) if path.is_file()}


def pipeline_inputs(source: Path, family: str) -> dict[str, np.ndarray]:
    """Fixed inputs in Dew's layout: a channels-last latent, the model time
    (a training timestep), a text context, and for FLUX and SD3 a pooled
    row and FLUX's distilled guidance."""
    component = "unet" if family == "sd" else "transformer"
    config = json.loads((source / component / "config.json").read_text())
    generator = np.random.default_rng(7)
    if family == "sd":
        return {"latent": generator.standard_normal((1, 8, 8, config["in_channels"]), dtype=np.float32),
                "context": generator.standard_normal((1, 6, config["cross_attention_dim"]), dtype=np.float32),
                "times": np.asarray([731.0], np.float32)}
    channels = config["in_channels"] // (4 if family == "flux" else 1)
    return {"latent": generator.standard_normal((1, 8, 8, channels), dtype=np.float32),
            "context": generator.standard_normal((1, 8, config["joint_attention_dim"]), dtype=np.float32),
            "pooled": generator.standard_normal((1, config["pooled_projection_dim"]), dtype=np.float32),
            "guidance": (np.full((1,), 3.5, np.float32) if config.get("guidance_embeds")
                         else np.zeros((0,), np.float32)),
            "times": np.asarray([731.0], np.float32)}


# --------------------------------------------------------------------------
# The consumers' half, in the PEFT environment
# --------------------------------------------------------------------------


def assert_landed(model, tensors: Mapping[str, np.ndarray], prefix: str = "") -> None:
    """Every tensor of the file is the model's adapter state under its name,
    and the model holds no adapter state the file does not."""
    import torch
    from peft.utils import get_peft_model_state_dict

    [name] = model.peft_config  # PEFT's "default", Diffusers' "default_0"
    held = {prefix + key: value for key, value in get_peft_model_state_dict(model, adapter_name=name).items()}
    if held.keys() != tensors.keys():
        missing, unexpected = sorted(tensors.keys() - held.keys()), sorted(held.keys() - tensors.keys())
        raise AssertionError(f"adapter state differs from the file: missing {missing[:4]}, "
                             f"unexpected {unexpected[:4]}")
    for name, value in held.items():
        if not torch.equal(value.detach().double(), torch.from_numpy(tensors[name]).double()):
            raise AssertionError(f"{name} holds other values than the file's")


def decoder_outputs(adapter: Path, wide: bool) -> dict[str, np.ndarray]:
    """llama-tiny under PEFT with the adapter, and merged into the weights."""
    import torch
    from peft import PeftModel
    from safetensors.numpy import load_file
    from transformers import AutoModelForCausalLM

    from tools.diffusers_consumer import precision

    ids = torch.from_numpy(np.load(LLAMA / "input_ids.npy")).long()
    with precision(wide):
        base = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float64 if wide else torch.float32,
                                                    attn_implementation="eager")
        model = PeftModel.from_pretrained(base, adapter).eval()
        assert_landed(model, load_file(str(adapter / "adapter_model.safetensors")))
        with torch.no_grad():
            adapted = model(ids).logits.numpy()
            with model.disable_adapter():
                plain = model(ids).logits.numpy()
            merged = model.merge_and_unload()(ids).logits.numpy()
    return {"adapted": adapted, "merged": merged, "base": plain}


def _prediction(family: str, model, arrays: Mapping[str, np.ndarray], wide: bool) -> np.ndarray:
    """The denoiser's call on the inputs, read back into Dew's layout: the
    SD UNet and SD3 take channels first, FLUX packs 2x2 latent patches into
    tokens and takes a timestep over the training count."""
    import torch

    def tensor(name):
        return torch.from_numpy(np.asarray(arrays[name], np.float64 if wide else np.float32))

    latent, times, context = arrays["latent"], tensor("times"), tensor("context")
    if family == "sd":
        output = model(tensor("latent").permute(0, 3, 1, 2), times, encoder_hidden_states=context).sample
        return output.permute(0, 2, 3, 1).numpy()
    if family == "sd3":
        output = model(hidden_states=tensor("latent").permute(0, 3, 1, 2), encoder_hidden_states=context,
                       pooled_projections=tensor("pooled"), timestep=times, return_dict=False)[0]
        return output.permute(0, 2, 3, 1).numpy()
    channels = latent.shape[-1]
    rows, columns = latent.shape[1] // 2, latent.shape[2] // 2
    packed = tensor("latent").reshape(1, rows, 2, columns, 2, channels).permute(0, 1, 3, 5, 2, 4)
    ids = torch.zeros(rows, columns, 3, dtype=times.dtype)
    ids[..., 1] += torch.arange(rows)[:, None]
    ids[..., 2] += torch.arange(columns)[None, :]
    output = model(hidden_states=packed.reshape(1, rows * columns, -1), encoder_hidden_states=context,
                   pooled_projections=tensor("pooled"), timestep=times / 1000, guidance=tensor("guidance"),
                   txt_ids=torch.zeros(context.shape[1], 3, dtype=times.dtype),
                   img_ids=ids.reshape(rows * columns, 3), return_dict=False)[0].numpy()
    return output.reshape(1, rows, columns, channels, 2, 2).transpose(0, 1, 4, 2, 5, 3).reshape(latent.shape)


def _pipeline(family: str, source: Path, dtype):
    """The family's Diffusers pipeline over its denoiser alone: the files
    adapt nothing else, and the tiny SD source declares Flax classes, whose
    pipeline loads no adapter, so its PyTorch UNet stands in a PyTorch one."""
    import diffusers

    if family == "sd":
        unet = diffusers.UNet2DConditionModel.from_pretrained(source, subfolder="unet", torch_dtype=dtype)
        scheduler = diffusers.DDIMScheduler.from_pretrained(source, subfolder="scheduler")
        return diffusers.StableDiffusionPipeline(
            vae=None, text_encoder=None, tokenizer=None, unet=unet, scheduler=scheduler, safety_checker=None,
            feature_extractor=None, requires_safety_checker=False)
    towers = {"flux": ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"),
              "sd3": ("text_encoder", "text_encoder_2", "text_encoder_3", "tokenizer", "tokenizer_2",
                      "tokenizer_3")}[family]
    cls = {"flux": diffusers.FluxPipeline, "sd3": diffusers.StableDiffusion3Pipeline}[family]
    return cls.from_pretrained(source, torch_dtype=dtype, **dict.fromkeys(towers))


def pipeline_outputs(family: str, adapter: Path, arrays: Mapping[str, np.ndarray],
                     wide: bool) -> dict[str, np.ndarray]:
    """The pipeline's denoiser with the adapter `load_lora_weights` reads,
    without it, and with it fused into the weights."""
    import torch
    from safetensors.numpy import load_file

    from tools.diffusers_consumer import precision

    component = "unet" if family == "sd" else "transformer"
    with source_directory(family) as source, precision(wide):
        pipe = _pipeline(family, source, torch.float64 if wide else torch.float32)
        pipe.load_lora_weights(adapter)
        model = getattr(pipe, component)
        assert_landed(model, load_file(str(adapter / "pytorch_lora_weights.safetensors")), f"{component}.")
        with torch.no_grad():
            adapted = _prediction(family, model, arrays, wide)
            pipe.disable_lora()
            plain = _prediction(family, model, arrays, wide)
            pipe.enable_lora()
            pipe.fuse_lora()
            merged = _prediction(family, model, arrays, wide)
    return {"adapted": adapted, "merged": merged, "base": plain}


def consume(directory: Path) -> None:
    sys.path[:0] = [str(ROOT)]
    import diffusers
    import peft
    import torch
    import transformers

    # Before Diffusers' pipelines: restores the transformers names they import.
    import tools.diffusers_consumer  # noqa: F401

    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, case in CASES.items():
        adapter = directory / name
        if case.source == "llama":
            arrays = {"input_ids": np.load(LLAMA / "input_ids.npy")}
        else:
            with source_directory(case.source) as source:
                arrays = pipeline_inputs(source, case.source)
        for precision, wide in (("fp32", False), ("fp64", True)):
            outputs = (decoder_outputs(adapter, wide) if case.source == "llama"
                       else pipeline_outputs(case.source, adapter, arrays, wide))
            arrays.update({f"{precision}.{kind}": value for kind, value in outputs.items()
                           if kind != "base" or not wide})
        meta = {"digests": digests(adapter), "peft": peft.__version__,
                "transformers": transformers.__version__, "diffusers": diffusers.__version__,
                "torch": torch.__version__}
        arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
        np.savez_compressed(FIXTURES / f"{name}.npz", **arrays)
        gap = float(np.abs(arrays["fp32.adapted"] - arrays["fp64.adapted"]).max())
        moved_by = float(np.abs(arrays["fp32.adapted"] - arrays["fp32.base"]).max())
        print(f"{name}: fp32 off float64 by {gap:.3g}, the adapter moves the output by {moved_by:.3g}")


def export(directory: Path) -> None:
    from dew.interop.pretrained import Pretrained

    for name, case in CASES.items():
        with source_directory(case.source) as source:
            loaded = Pretrained.load(source, dtype="float32", attention_impl="xla")
            export_case(case, loaded, directory / name)
        print(f"{directory / name}: {sorted(digests(directory / name))}")


def main() -> None:
    if sys.argv[1:2] == ["export"] and len(sys.argv) == 3:
        export(Path(sys.argv[2]))
    elif sys.argv[1:2] == ["consume"] and len(sys.argv) == 3:
        consume(Path(sys.argv[2]))
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
