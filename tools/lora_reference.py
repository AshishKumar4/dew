#!/usr/bin/env python3
"""Write the LoRA fixtures tests/test_lora.py checks against.

Everything here runs under torch with PEFT, which dew does not depend on, so
this is the only place the reference adapters are executed. Two isolated
environments, both sharing the baseline site-packages through a `.pth` file
(see /home/mrwhite0racle/.cache/dew/vision-check/lib/python3.12/site-packages/dew-baseline.pth):

    uv venv --python <py312> ~/.cache/dew/peft-reference
    uv pip install --python ~/.cache/dew/peft-reference/bin/python --no-deps peft==0.20.0 accelerate psutil
    PYTHONPATH=src ~/.cache/dew/peft-reference/bin/python tools/lora_reference.py decoder

    uv venv --python <py312> ~/.cache/dew/peft-diffusers-reference
    uv pip install --python ~/.cache/dew/peft-diffusers-reference/bin/python --no-deps \\
        peft==0.20.0 accelerate psutil transformers==4.49.0 "tokenizers>=0.21,<0.22" "huggingface-hub>=0.26,<1"
    PYTHONPATH=src ~/.cache/dew/peft-diffusers-reference/bin/python tools/lora_reference.py pipeline

The FLUX and SD3 round trip runs in two halves: Dew writes its adapter in
Dew's own environment, whose transformers reads the fixtures' tokenizers,
and Diffusers reads it in the pipeline reference environment:

    PYTHONPATH=src .venv/bin/python tools/lora_reference.py pipeline-adapters
    PYTHONPATH=src ~/.cache/dew/peft-diffusers-reference/bin/python \\
        tools/lora_reference.py check-pipeline-adapters

The decoder reference runs on transformers 5.16.1, the release every decoder
fixture is calibrated against; the pipeline reference needs Diffusers 0.34.0's
pipelines, which import only under transformers 4.49.0, the pair the native
diffusion fixtures were written with.

What lands in tests/fixtures/lora:

- llama-tiny/adapter: a PEFT adapter on tests/fixtures/hf/llama-tiny, rank 4
  alpha 8 on q_proj, v_proj and down_proj with a `rank_pattern` and an
  `alpha_pattern`, its B factors drawn at random so the adapter moves the
  logits. llama-tiny/reference.npz holds the base and adapted logits on the
  fixture's input_ids, the merged weights of every target, the gradient of
  the mean next-token cross entropy with respect to every adapter parameter,
  and the adapter and logits after one SGD step on those parameters alone.
- sd-tiny: a Diffusers LoRA file on the `sd` pipeline of
  tests/fixtures/tiny_diffusers.tar.xz, rank 4 alpha 6 on the UNet's to_q,
  to_v and to_out.0 and rank 2 alpha 5 on the text encoder's q_proj and
  k_proj, with the PEFT configs in the header. sd-tiny/reference.npz holds a
  latent, timestep and prompt ids, the adapted text encoder's hidden states,
  the UNet's prediction on them with the adapter on, off, and fused (the
  fused prediction equals the adapted one), and the fused UNet weights of
  every target.
- flux-tiny and sd3-tiny: the round trip of an adapter Dew writes. Dew loads
  the `pipeline` of tests/fixtures/{flux,sd3}_source.tar.xz, adapts its
  transformer's to_q, to_k, to_v and to_out.0 at rank 4 through
  `PretrainedPipeline.lora`, draws every B away from zero and writes the
  Diffusers file with `adapter.save`. Diffusers then reads that file with
  the pipeline's own `load_lora_weights`; reference.npz holds a latent, a
  model time, a text context, a pooled row and FLUX's distilled guidance,
  all in Dew's layout, with the transformer's prediction on them with the
  adapter on and off, also in Dew's layout, and Dew's own prediction.

`check DIRECTORY` loads a PEFT adapter dew exported next to a `logits.npy`
into the reference decoder and compares; `check-pipeline DIRECTORY` loads a
Diffusers file dew exported next to a `context.npy` and a `prediction.npy`
into the reference pipeline and compares on the fixture's inputs.
"""

import argparse
import contextlib
import copy
import json
import sys
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "lora"
LLAMA = ROOT / "tests" / "fixtures" / "hf" / "llama-tiny"
LEARNING_RATE = 0.05


def _random_b(model, seed: int) -> None:
    """PEFT starts B at zero; a nonzero B makes the adapter observable."""
    generator = torch.Generator().manual_seed(seed)
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            parameter.data = torch.randn(parameter.shape, generator=generator) * 0.2


def decoder(out: Path) -> None:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    import transformers
    if transformers.__version__ != "5.16.1":
        raise RuntimeError("the decoder reference runs on transformers 5.16.1")

    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float32)
    input_ids = torch.from_numpy(np.load(LLAMA / "input_ids.npy")).long()
    with torch.no_grad():
        base_logits = base(input_ids).logits.numpy()
    config = LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "down_proj"],
                        rank_pattern={"layers.1.self_attn.v_proj": 2}, alpha_pattern={"down_proj": 3.0},
                        lora_dropout=0.1)
    model = get_peft_model(base, config)
    _random_b(model, 1)
    model.eval()
    adapter = out / "adapter"
    model.save_pretrained(adapter)
    (adapter / "README.md").unlink()
    # The config names the machine the adapter was made on; the fixture
    # names the repository's copy.
    saved = json.loads((adapter / "adapter_config.json").read_text())
    saved["base_model_name_or_path"] = str(LLAMA.relative_to(ROOT))
    (adapter / "adapter_config.json").write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")

    with torch.no_grad():
        adapted_logits = model(input_ids).logits.numpy()
    trained = copy.deepcopy(model)
    loss = trained(input_ids, labels=input_ids).loss
    loss.backward()
    arrays = {"input_ids": input_ids.numpy(), "base_logits": base_logits, "adapted_logits": adapted_logits,
              "loss": np.float32(loss.item())}
    for name, parameter in trained.named_parameters():
        if parameter.requires_grad:
            key = name.removeprefix("base_model.model.").replace(".default", "")
            arrays["grad/" + key] = parameter.grad.numpy().copy()
    optimizer = torch.optim.SGD([p for p in trained.parameters() if p.requires_grad], lr=LEARNING_RATE)
    optimizer.step()
    with torch.no_grad():
        arrays["updated_logits"] = trained(input_ids).logits.numpy()
    for name, parameter in trained.named_parameters():
        if parameter.requires_grad:
            key = name.removeprefix("base_model.model.").replace(".default", "")
            arrays["updated/" + key] = parameter.detach().numpy().copy()
    merged = model.merge_and_unload()
    with torch.no_grad():
        arrays["merged_logits"] = merged(input_ids).logits.numpy()
    for name, parameter in merged.named_parameters():
        if any(name.endswith(f"{target}.weight") for target in config.target_modules):
            arrays["merged/" + name] = parameter.detach().numpy().copy()
    np.testing.assert_allclose(arrays["merged_logits"], arrays["adapted_logits"], atol=1e-5)
    np.savez_compressed(out / "reference.npz", **arrays)
    (out / "meta.json").write_text(json.dumps({"peft": _peft_version(), "transformers": transformers.__version__,
                                               "learning_rate": LEARNING_RATE}, indent=2) + "\n")
    print(f"wrote {out}")


def check(directory: Path) -> None:
    """Load an adapter dew exported into PEFT and compare to the logits beside it."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float32)
    model = PeftModel.from_pretrained(base, directory)
    input_ids = torch.from_numpy(np.load(LLAMA / "input_ids.npy")).long()
    with torch.no_grad():
        logits = model(input_ids).logits.numpy()
    expected = np.load(directory / "logits.npy")
    error = float(np.max(np.abs(logits - expected)))
    print(json.dumps({"max |logit difference|": error, "argmax equal": bool(
        np.array_equal(logits.argmax(-1), expected.argmax(-1)))}))
    if error > 1e-4:
        raise SystemExit(1)


def _peft_version() -> str:
    import peft
    return peft.__version__


@contextlib.contextmanager
def _tiny_sd():
    """The `sd` pipeline of the bundle, in torch, for the duration of the block."""
    import transformers
    from diffusers import AutoencoderKL, DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    if transformers.__version__ != "4.49.0":
        raise RuntimeError("the pipeline reference runs on transformers 4.49.0")
    with tempfile.TemporaryDirectory() as extracted:
        with tarfile.open(ROOT / "tests/fixtures/tiny_diffusers.tar.xz") as archive:
            archive.extractall(extracted, members=[m for m in archive.getmembers() if m.name.startswith("sd/")],
                               filter="data")
        root = Path(extracted) / "sd"
        unet = UNet2DConditionModel.from_pretrained(root, subfolder="unet", torch_dtype=torch.float32)
        text_encoder = CLIPTextModel.from_pretrained(root, subfolder="text_encoder")
        tokenizer = CLIPTokenizer.from_pretrained(root, subfolder="tokenizer")
        yield StableDiffusionPipeline(
            vae=AutoencoderKL.from_pretrained(root, subfolder="vae"), text_encoder=text_encoder,
            tokenizer=tokenizer, unet=unet, scheduler=DDIMScheduler.from_pretrained(root, subfolder="scheduler"),
            safety_checker=None, feature_extractor=None, requires_safety_checker=False)


def check_pipeline(directory: Path) -> None:
    """Load a Diffusers file dew exported and compare the pipeline's text
    context and UNet prediction on the fixture's inputs to the `context.npy`
    and `prediction.npy` dew wrote beside it."""
    reference = np.load(FIXTURES / "sd-tiny" / "reference.npz")
    with _tiny_sd() as pipe:
        pipe.load_lora_weights(directory)
        with torch.no_grad():
            context = pipe.text_encoder(torch.from_numpy(reference["prompt_ids"]))[0]
            predicted = pipe.unet(torch.from_numpy(reference["latent"]).permute(0, 3, 1, 2),
                                  torch.from_numpy(reference["time"]), encoder_hidden_states=context).sample
    errors = {"context": float(np.max(np.abs(context.numpy() - np.load(directory / "context.npy")))),
              "prediction": float(np.max(np.abs(predicted.permute(0, 2, 3, 1).numpy()
                                                - np.load(directory / "prediction.npy"))))}
    print(json.dumps(errors))
    if max(errors.values()) > 1e-4:
        raise SystemExit(1)


def pipeline(out: Path) -> None:
    import transformers
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict

    with _tiny_sd() as pipe:
        unet, text_encoder, tokenizer = pipe.unet, pipe.text_encoder, pipe.tokenizer
        torch.manual_seed(0)
        unet_config = LoraConfig(r=4, lora_alpha=6, target_modules=["to_q", "to_v", "to_out.0"])
        text_config = LoraConfig(r=2, lora_alpha=5, target_modules=["q_proj", "k_proj"])
        unet.add_adapter(unet_config)
        text_encoder.add_adapter(text_config)
        _random_b(unet, 2)
        _random_b(text_encoder, 3)
        # The configs name the temporary directory the pipeline was read
        # from; the file names no base.
        metadata = [{**config.to_dict(), "base_model_name_or_path": None} for config in (unet_config, text_config)]
        pipe.save_lora_weights(out, unet_lora_layers=get_peft_model_state_dict(unet),
                               text_encoder_lora_layers=get_peft_model_state_dict(text_encoder),
                               unet_lora_adapter_metadata=metadata[0], text_encoder_lora_adapter_metadata=metadata[1])

        prompt_ids = tokenizer(["a cat"], padding="max_length", max_length=tokenizer.model_max_length,
                               return_tensors="pt").input_ids
        generator = torch.Generator().manual_seed(4)
        latent = torch.randn((1, 4, 8, 8), generator=generator)
        time = torch.tensor([7])
        with torch.no_grad():
            context = text_encoder(prompt_ids)[0]
            adapted = unet(latent, time, encoder_hidden_states=context).sample
            unet.disable_adapters()
            base = unet(latent, time, encoder_hidden_states=context).sample
            unet.enable_adapters()
            pipe.fuse_lora()
            fused_context = text_encoder(prompt_ids)[0]
            fused = unet(latent, time, encoder_hidden_states=context).sample
        np.testing.assert_allclose(fused.numpy(), adapted.numpy(), atol=1e-5)
        np.testing.assert_allclose(fused_context.numpy(), context.numpy(), atol=1e-5)
        arrays = {"prompt_ids": prompt_ids.numpy(), "context": context.numpy(),
                  "latent": latent.numpy().transpose(0, 2, 3, 1), "time": time.numpy(),
                  "adapted": adapted.numpy().transpose(0, 2, 3, 1), "base": base.numpy().transpose(0, 2, 3, 1)}
        for component, module, config in (("unet", unet, unet_config), ("text_encoder", text_encoder, text_config)):
            for name, parameter in module.named_parameters():
                name = name.replace(".base_layer", "")
                if "lora" not in name and any(name.endswith(f"{t}.weight") for t in config.target_modules):
                    arrays[f"merged/{component}.{name}"] = parameter.detach().numpy().copy()
    np.savez_compressed(out / "reference.npz", **arrays)
    (out / "meta.json").write_text(json.dumps({"peft": _peft_version(), "transformers": transformers.__version__,
                                               "diffusers": __import__("diffusers").__version__}, indent=2) + "\n")
    print(f"wrote {out}")


ADAPTED = ("to_q", "to_k", "to_v", "to_out.0")
FAMILIES = ("flux", "sd3")
PIPELINES = {"flux": "FluxPipeline", "sd3": "StableDiffusion3Pipeline"}
TEXT = {"flux": ("text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2"),
        "sd3": ("text_encoder", "text_encoder_2", "text_encoder_3",
                "tokenizer", "tokenizer_2", "tokenizer_3")}


@contextlib.contextmanager
def _source_pipeline(family: str):
    """The `pipeline` directory of the family's source fixture, for the duration of the block."""
    with tempfile.TemporaryDirectory() as extracted:
        with tarfile.open(ROOT / f"tests/fixtures/{family}_source.tar.xz") as archive:
            archive.extractall(extracted, filter="data")
        yield Path(extracted) / "pipeline"


def dew_adapter(family: str, directory: Path) -> None:
    """Dew's half of the round trip, in Dew's own environment: adapt the
    family's tiny pipeline through `PretrainedPipeline.lora`, draw every B
    away from zero, write the Diffusers file with `adapter.save`, and record
    the inputs with Dew's prediction on them."""
    import jax
    import jax.numpy as jnp

    from dew.diffusion.process import DenoisingCondition
    from dew.interop.pretrained import Pretrained

    directory.mkdir(parents=True, exist_ok=True)
    with _source_pipeline(family) as root:
        source = Pretrained.load(root, dtype="float32", attention_impl="xla")
        config = json.loads((root / "transformer" / "config.json").read_text())
    tuned = source.lora(rank=4, modules=ADAPTED, key=jax.random.key(0))
    moved = iter(jax.random.split(jax.random.key(1), len(tuned.adapter.targets)))
    variables = jax.tree_util.tree_map_with_path(
        lambda path, leaf: (0.2 * jax.random.normal(next(moved), leaf.shape, leaf.dtype)
                            if path[-1].key == "lora_B" else leaf), tuned.variables)
    tuned.adapter.save(variables, directory)
    generator = np.random.default_rng(2)
    channels = config["in_channels"] // (4 if family == "flux" else 1)
    arrays = {
        "latent": generator.standard_normal((1, 8, 8, channels), dtype=np.float32),
        "context": generator.standard_normal((1, 8, config["joint_attention_dim"]), dtype=np.float32),
        "pooled": generator.standard_normal((1, config["pooled_projection_dim"]), dtype=np.float32),
        "guidance": (np.full((1,), 3.5, np.float32) if config.get("guidance_embeds")
                     else np.zeros((0,), np.float32)),
        "times": np.asarray([731.0], np.float32)}
    own = {name: tree for name, tree in variables.items() if name not in ("encoders", "autoencoder")}
    guidance = jnp.asarray(arrays["guidance"]) if arrays["guidance"].size else None
    condition = DenoisingCondition(jnp.asarray(arrays["context"]), jnp.asarray(arrays["pooled"]),
                                   guidance=guidance)
    arrays["dew"] = np.asarray(tuned.model.apply(own, jnp.asarray(arrays["latent"]),
                                                 jnp.asarray(arrays["times"]), condition))
    np.savez_compressed(directory / "reference.npz", **arrays)
    print(f"wrote {directory}")


def _transformer_prediction(family: str, transformer, arrays) -> np.ndarray:
    """The source transformer's call on the recorded inputs, read back into Dew's layout:
    FLUX packs 2x2 latent patches into tokens and takes a timestep over the
    training count, SD3 takes channels first."""
    latent, times = arrays["latent"], torch.from_numpy(arrays["times"])
    context, pooled = torch.from_numpy(arrays["context"]), torch.from_numpy(arrays["pooled"])
    if family == "sd3":
        output = transformer(hidden_states=torch.from_numpy(latent).permute(0, 3, 1, 2),
                             encoder_hidden_states=context, pooled_projections=pooled, timestep=times,
                             return_dict=False)[0]
        return output.permute(0, 2, 3, 1).numpy()
    channels = latent.shape[-1]
    rows, columns = latent.shape[1] // 2, latent.shape[2] // 2
    packed = latent.reshape(1, rows, 2, columns, 2, channels).transpose(0, 1, 3, 5, 2, 4)
    ids = torch.zeros(rows, columns, 3)
    ids[..., 1] += torch.arange(rows)[:, None]
    ids[..., 2] += torch.arange(columns)[None, :]
    output = transformer(
        hidden_states=torch.from_numpy(np.ascontiguousarray(packed).reshape(1, rows * columns, -1)),
        encoder_hidden_states=context, pooled_projections=pooled, timestep=times / 1000,
        guidance=torch.from_numpy(arrays["guidance"]), txt_ids=torch.zeros(context.shape[1], 3),
        img_ids=ids.reshape(rows * columns, 3), return_dict=False)[0].numpy()
    return output.reshape(1, rows, columns, channels, 2, 2).transpose(0, 1, 4, 2, 5, 3).reshape(latent.shape)


def diffusers_check(family: str, directory: Path) -> None:
    """Diffusers' half, in the pipeline reference environment: the family's
    pipeline reads the file Dew wrote with its own `load_lora_weights`, and
    its transformer predicts on the recorded inputs with the adapter on and
    off. Both land in reference.npz once the adapted one agrees with Dew's.
    The text towers and their tokenizers are not loaded: the file adapts the
    transformer alone."""
    import diffusers
    import transformers

    arrays = dict(np.load(directory / "reference.npz"))
    with _source_pipeline(family) as root:
        pipe = getattr(diffusers, PIPELINES[family]).from_pretrained(
            root, torch_dtype=torch.float32, **dict.fromkeys(TEXT[family]))
    pipe.load_lora_weights(directory)
    with torch.no_grad():
        adapted = _transformer_prediction(family, pipe.transformer, arrays)
        pipe.disable_lora()
        base = _transformer_prediction(family, pipe.transformer, arrays)
    gap = float(np.abs(arrays["dew"] - adapted).max() / np.abs(adapted).max())
    moved_by = float(np.abs(adapted - base).max())
    print(json.dumps({"family": family, "relative gap to Dew": gap, "max |adapted - base|": moved_by}))
    if gap > 1e-5 or moved_by < 1e-2:
        raise SystemExit(f"{family}: Diffusers does not predict what Dew does with Dew's adapter")
    np.savez_compressed(directory / "reference.npz", **arrays, adapted=adapted, base=base)
    versions = {"peft": _peft_version(), "transformers": transformers.__version__,
                "diffusers": diffusers.__version__}
    (directory / "meta.json").write_text(json.dumps(versions, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("target",
                        choices=["decoder", "pipeline", "pipeline-adapters", "check", "check-pipeline",
                                 "check-pipeline-adapters"])
    parser.add_argument("directory", nargs="?", type=Path)
    parser.add_argument("--out", type=Path, default=FIXTURES)
    arguments = parser.parse_args(argv)
    if arguments.target in ("pipeline-adapters", "check-pipeline-adapters"):
        half = dew_adapter if arguments.target == "pipeline-adapters" else diffusers_check
        for family in FAMILIES:
            half(family, arguments.out / f"{family}-tiny")
    elif arguments.target.startswith("check"):
        if arguments.directory is None:
            parser.error("check needs the exported adapter directory")
        (check if arguments.target == "check" else check_pipeline)(arguments.directory)
    elif arguments.target == "decoder":
        (arguments.out / "llama-tiny").mkdir(parents=True, exist_ok=True)
        decoder(arguments.out / "llama-tiny")
    else:
        (arguments.out / "sd-tiny").mkdir(parents=True, exist_ok=True)
        pipeline(arguments.out / "sd-tiny")


if __name__ == "__main__":
    sys.exit(main())
