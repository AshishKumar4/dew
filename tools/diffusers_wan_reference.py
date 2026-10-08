"""Actual Diffusers 0.34.0 Wan 2.1 objects, saved and walked, for tests/fixtures.

The transformer half constructs tiny `WanTransformer3DModel` instances with
every parameter moved off its initialization, saves each with
`save_pretrained`, and calls each as `WanPipeline` calls it: latents
`[B, C, F, H, W]`, the timestep, and the prompt states. Against a fixed
cotangent it records the gradients of the latents, the prompt states and
every parameter. Each call runs twice over the same float32 weights: in
float32, the reference, and in float64 (`widened`), the truth
tests/reference_error.py measures both runs from; and once more forward in
bfloat16 as `from_pretrained(torch_dtype=torch.bfloat16)` loads it, its
`_keep_in_fp32_modules` in float32, on bfloat16 inputs. The weights are
rounded to bfloat16-representable values so the fixture compresses.

The pipeline half saves one tiny `WanPipeline` over the published configs -
a UMT5 text encoder saved as the release stores it, a character-level T5
tokenizer, the Wan VAE, the transformer and the published UniPC scheduler
config with its flow shift - and walks the unmodified call from fixed
latents, recording the prompt states, the latent it ends on and the frames
it decodes. Its prompts carry what `prompt_clean` repairs: curly quotes,
HTML entities and runs of whitespace. Cleaning needs ftfy installed. Beside
the walk it steps the published UniPC scheduler alone, at 10 and 50 steps,
on a fixed sequence of model outputs that no sample feeds back into: its
timesteps and the sample after every step, in float32 and in float64.

Run with the Dew test environment's diffusers 0.34.0 and torch, on CPU:

    python tools/diffusers_wan_reference.py transformer OUTPUT_DIR
    python tools/diffusers_wan_reference.py bundle OUTPUT_DIR tests/fixtures/wan_transformer.tar.xz
    python tools/diffusers_wan_reference.py pipeline OUTPUT_DIR
    python tools/diffusers_wan_reference.py bundle OUTPUT_DIR tests/fixtures/wan_pipeline.tar.xz
"""

from __future__ import annotations

import contextlib
import copy
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import numpy as np
import torch
import transformers.utils as transformers_utils

# Diffusers 0.34.0's pipeline modules import two names Transformers dropped
# after 4.x; the pipeline walk needs `WanPipeline` itself, so the names are
# restored, as tools/diffusers_sd3_reference.py restores them.
for _name, _value in (("FLAX_WEIGHTS_NAME", "flax_model.msgpack"),
                      ("WEIGHTS_INDEX_NAME", "pytorch_model.bin.index.json")):
    if not hasattr(transformers_utils, _name):
        setattr(transformers_utils, _name, _value)

DIFFUSERS = "0.34.0"
SOURCE = Path(__file__).resolve().parents[1] / "tests/fixtures/hf/wan-source"
BASE = {"patch_size": [1, 2, 2], "num_attention_heads": 2, "attention_head_dim": 12, "in_channels": 4,
        "out_channels": 4, "text_dim": 20, "freq_dim": 16, "ffn_dim": 40, "num_layers": 2,
        "cross_attn_norm": True, "qk_norm": "rms_norm_across_heads", "eps": 1e-6, "rope_max_seq_len": 32}
SEED = 43


@dataclass(frozen=True)
class Case:
    """One transformer: the config controls that differ from the tiny base,
    the latent's (frames, height, width), the prompt length and each row's
    timestep."""

    config: dict = field(default_factory=dict)
    size: tuple[int, int, int] = (3, 6, 8)
    length: int = 7
    times: tuple[float, ...] = (250.0, 612.5)


CASES: dict[str, Case] = {
    # The published controls: one frame per patch, norms on.
    "published": Case(),
    # A head of 20 splits its pairs 4/3/3 rather than 2/2/2; two frames per
    # patch; neither norm; three layers at a coarse eps.
    "variant": Case({"patch_size": [2, 2, 2], "attention_head_dim": 20, "qk_norm": None,
                     "cross_attn_norm": False, "num_layers": 3, "eps": 1e-3, "out_channels": 6},
                    size=(4, 4, 6), length=5, times=(999.0,)),
}


class Float64Library:
    """A library (numpy, torch) with its float32 read as float64, for a
    scheduler module whose tables round through an explicit float32."""

    def __init__(self, library):
        self.library = library

    def __getattr__(self, name):
        return getattr(self.library, "float64" if name == "float32" else name)


@contextlib.contextmanager
def float64_scheduler(module: ModuleType):
    """Evaluate the published formulas in float64, including table creation.

    Casting already-built tables would retain their float32 cumprod and
    interpolation errors. Some step methods also explicitly upcast a sample
    to float32; that minimum precision must become float64 for this oracle.
    Only this scheduler module sees the widened libraries, and the default
    dtype is restored before the native float32 reference runs.
    """
    from unittest.mock import patch

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with contextlib.ExitStack() as scope:
            for name, library in (("torch", torch), ("np", np)):
                if hasattr(module, name):
                    scope.enter_context(patch.object(module, name, Float64Library(library)))
            yield
    finally:
        torch.set_default_dtype(previous)


@contextlib.contextmanager
def widened():
    """The reference in float64: every float32 pin of the Wan modules
    widened, which are their `.float()` and `.to(torch.float32)` casts, the
    sinusoids' float32 `arange` and the UniPC scheduler's float32 sigma
    table, and float64 the default dtype."""
    from unittest.mock import patch

    from diffusers.schedulers import scheduling_unipc_multistep
    from torch.utils._device import _device_constructors

    # `torch.device(...)` as a context places only the constructors in a set
    # torch builds once, on first use. Built inside this block it would hold
    # the arange below for the rest of the process, and every later
    # meta-device init (transformers' from_pretrained) would make arange's
    # tensors on the CPU beside the meta ones; so it is built first.
    _device_constructors()
    float_, to, arange = torch.Tensor.float, torch.Tensor.to, torch.arange

    def wide(dtype):
        return torch.float64 if dtype == torch.float32 else dtype

    def widened_to(self, *args, **kwargs):
        if "dtype" in kwargs:
            kwargs["dtype"] = wide(kwargs["dtype"])
        return to(self, *(wide(arg) if isinstance(arg, torch.dtype) else arg for arg in args), **kwargs)

    torch.Tensor.float = lambda self, *args, **kwargs: self.double()
    torch.Tensor.to = widened_to
    torch.arange = lambda *args, dtype=None, **kwargs: arange(*args, dtype=wide(dtype), **kwargs)
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        with patch.object(scheduling_unipc_multistep, "np", Float64Library(np)):
            yield
    finally:
        torch.Tensor.float, torch.Tensor.to, torch.arange = float_, to, arange
        torch.set_default_dtype(default)


@contextlib.contextmanager
def widened_softmax():
    """`softmax(..., dtype=torch.float32)`, the pin transformers' eager
    attention and routers carry, in float64."""
    softmax = torch.nn.functional.softmax

    def wide(*args, dtype=None, **kwargs):
        return softmax(*args, dtype=torch.float64 if dtype == torch.float32 else dtype, **kwargs)

    torch.nn.functional.softmax = wide
    try:
        yield
    finally:
        torch.nn.functional.softmax = softmax


@contextlib.contextmanager
def float64():
    """A transformers model built and run in float64: `widened` and
    `widened_softmax` together."""
    with widened(), widened_softmax():
        yield


def walk(model, inputs: dict[str, torch.Tensor], probe: torch.Tensor, dtype) -> dict[str, np.ndarray]:
    """One call at `dtype` and the gradients of `sum(output * probe)` with
    respect to the latents, the prompt states and every parameter."""
    model = model.to(dtype)
    latents, context = (inputs[name].to(dtype).requires_grad_() for name in ("latents", "context"))
    named = list(model.named_parameters())
    output = model(latents, inputs["times"].to(dtype), context, return_dict=False)[0]
    grads = torch.autograd.grad((output * probe.to(dtype)).sum(),
                                [latents, context] + [value for _, value in named])
    arrays = {"output": output, "grad_latents": grads[0], "grad_context": grads[1]}
    for (key, _), gradient in zip(named, grads[2:], strict=True):
        arrays[f"grad_param.{key}"] = gradient
    return {key: value.detach().numpy() for key, value in arrays.items()}


def build(name: str, case: Case, root: Path) -> dict[str, np.ndarray]:
    from diffusers import WanTransformer3DModel

    config = {**BASE, **case.config}
    torch.manual_seed(SEED)
    model = WanTransformer3DModel(**config).eval()
    generator = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1 * torch.randn(parameter.shape, generator=generator))
            parameter.copy_(parameter.to(torch.bfloat16).float())
    model.save_pretrained(root / name / "transformer", safe_serialization=True)
    batch = len(case.times)
    inputs = {"latents": torch.randn((batch, config["in_channels"], *case.size), generator=generator),
              "context": torch.randn((batch, case.length, config["text_dim"]), generator=generator),
              "times": torch.tensor(case.times)}
    with torch.no_grad():
        shape = model(inputs["latents"], inputs["times"], inputs["context"], return_dict=False)[0].shape
    probe = torch.randn(shape, generator=generator)
    arrays = {key: value.numpy() for key, value in inputs.items()}
    arrays["probe"] = probe.numpy()
    arrays.update({f"fp32.{key}": value for key, value in walk(model, inputs, probe, torch.float32).items()})
    with widened():
        widest = walk(model, inputs, probe, torch.float64)
        arrays.update({f"fp64.{key}": value for key, value in widest.items()})
    half = WanTransformer3DModel.from_pretrained(root / name / "transformer", torch_dtype=torch.bfloat16)
    with torch.no_grad():
        output = half.eval()(inputs["latents"].bfloat16(), inputs["times"], inputs["context"].bfloat16(),
                             return_dict=False)[0]
    arrays["bf16.output"] = output.float().numpy()
    gap = np.abs(arrays["fp32.output"] - arrays["fp64.output"]).max()
    print(f"{name}: latent {case.size} prompt {case.length} "
          f"|output| <= {np.abs(arrays['fp64.output']).max():.4g}, fp32 off float64 by {gap:.3g}")
    return arrays


def output_root(destination: str) -> Path:
    import diffusers

    if diffusers.__version__ != DIFFUSERS:
        raise RuntimeError(f"recorded against diffusers {DIFFUSERS}, not {diffusers.__version__}")
    torch.set_num_threads(2)
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    return root


def transformer(destination: str) -> None:
    root = output_root(destination)
    arrays: dict[str, np.ndarray] = {}
    for name, case in CASES.items():
        arrays.update({f"{name}.{key}": value for key, value in build(name, case, root).items()})
    record = {"diffusers": DIFFUSERS, "torch": torch.__version__, "base": BASE, "seed": SEED,
              "cases": {name: {"config": {**BASE, **case.config}, "size": list(case.size),
                               "length": case.length, "times": list(case.times)}
                        for name, case in CASES.items()}}
    np.savez_compressed(root / "wan_transformer.npz", **arrays)
    (root / "wan_transformer.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB, {len(CASES)} cases")


PIPELINE = {**BASE, "text_dim": 16}
PIPELINE_VAE = {"base_dim": 8, "z_dim": 4, "dim_mult": [1, 2, 2, 2], "num_res_blocks": 1}
# ftfy unescapes the entities of text with no `<` itself; with one, only the
# source's own two unescapes reach them.
PROMPTS = ["a red cat on a mat", "  tiny \u201cboat\u201d <&amp;amp;>   sea\n"]
FRAMES, HEIGHT, WIDTH, STEPS, GUIDANCE = 9, 32, 48, 4, 5.0


def tokenizer():
    """A T5 tokenizer over single characters: the release's umt5 vocabulary
    is 256k pieces, and every character here is one."""
    import string

    from transformers import T5Tokenizer

    pieces = ["<pad>", "</s>", "<unk>", "\u2581", *string.ascii_lowercase, *string.digits,
              *string.punctuation]
    return T5Tokenizer(vocab=[(piece, 0.0) for piece in pieces], extra_ids=0)


def text_encoder(vocab: int):
    """The published UMT5 config, narrowed and two layers deep, as
    `UMT5EncoderModel`, which saves as the release is stored."""
    from transformers import UMT5Config, UMT5EncoderModel

    config = json.loads((SOURCE / "text_encoder" / "config.json").read_text())
    for key in ("torch_dtype", "dtype", "_name_or_path", "architectures", "transformers_version"):
        config.pop(key, None)
    config.update(vocab_size=vocab, d_model=PIPELINE["text_dim"], d_kv=8, d_ff=32, num_layers=2,
                  num_decoder_layers=2, num_heads=2, relative_attention_num_buckets=8,
                  relative_attention_max_distance=32)
    torch.manual_seed(SEED + 5)
    model = UMT5EncoderModel(UMT5Config(**config)).eval()
    generator = torch.Generator().manual_seed(SEED + 6)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)
    return model


def build_pipeline(root: Path):
    from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel

    tok = tokenizer()
    encoder = text_encoder(len(tok))
    published = json.loads((SOURCE / "vae" / "config.json").read_text())
    vae_config = {key: value for key, value in published.items() if not key.startswith("_")}
    generator = torch.Generator().manual_seed(SEED + 7)
    vae_config.update(PIPELINE_VAE, latents_mean=torch.randn(4, generator=generator).mul(0.5).tolist(),
                      latents_std=torch.rand(4, generator=generator).add(0.5).tolist())
    torch.manual_seed(SEED + 8)
    vae = AutoencoderKLWan(**vae_config).eval()
    transformer = WanTransformer3DModel(**PIPELINE).eval()
    with torch.no_grad():
        for module in (vae, transformer):
            for name, parameter in module.named_parameters():
                if module is transformer or name.endswith(("gamma", "bias")):
                    parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.05)
    scheduler = json.loads((SOURCE / "scheduler" / "scheduler_config.json").read_text())
    pipe = WanPipeline(tokenizer=tok, text_encoder=encoder, transformer=transformer, vae=vae,
                       scheduler=UniPCMultistepScheduler.from_config(scheduler))
    directory = root / "pipeline"
    pipe.save_pretrained(directory, safe_serialization=True)
    pipe.set_progress_bar_config(disable=True)
    index = json.loads((directory / "model_index.json").read_text())
    index.update(dew_frames=FRAMES, dew_height=HEIGHT, dew_width=WIDTH)
    (directory / "model_index.json").write_text(json.dumps(index, indent=2))
    return pipe


REPLAY_STEPS = (10, 50)


def scheduler_replay() -> dict[str, np.ndarray]:
    """The published scheduler stepped on recorded model outputs, the one
    model evaluation per step a guided Wan walk makes, from a fixed latent:
    its timesteps and every step's sample, in float32 and widened."""
    from diffusers import UniPCMultistepScheduler

    config = json.loads((SOURCE / "scheduler" / "scheduler_config.json").read_text())
    arrays: dict[str, np.ndarray] = {}
    for steps in REPLAY_STEPS:
        generator = torch.Generator().manual_seed(SEED + 10 + steps)
        x_T = torch.randn((2, 4, 3, 4, 6), generator=generator)
        outputs = torch.randn((steps, *x_T.shape), generator=generator)
        arrays[f"replay.{steps}.x_T"], arrays[f"replay.{steps}.outputs"] = x_T.numpy(), outputs.numpy()
        for precision, dtype in (("fp32", torch.float32), ("fp64", torch.float64)):
            with widened() if dtype == torch.float64 else contextlib.nullcontext():
                scheduler = UniPCMultistepScheduler.from_config(config)
                scheduler.set_timesteps(steps)
                x, latents = x_T.to(dtype), []
                for index, time in enumerate(scheduler.timesteps):
                    x = scheduler.step(outputs[index].to(dtype), time, x).prev_sample
                    latents.append(x)
            arrays[f"replay.{steps}.{precision}.latents"] = torch.stack(latents).numpy()
            arrays[f"replay.{steps}.{precision}.timesteps"] = scheduler.timesteps.numpy()
            arrays[f"replay.{steps}.{precision}.sigmas"] = scheduler.sigmas.numpy()
    return arrays


def encoded_and_walked(pipe, latents: torch.Tensor, suffix: str = "") -> dict[str, np.ndarray]:
    """The prompt states each prompt (and the empty negative) encodes to, and
    the unmodified call's walk from `latents` at `STEPS` steps, guided at
    its default scale: the latent it ends on and the frames it decodes."""
    arrays: dict[str, np.ndarray] = {}
    for row, prompt in enumerate([*PROMPTS, ""]):
        with torch.no_grad():
            # `encode_prompt` defaults to 226 tokens; the call passes its own 512.
            states, _ = pipe.encode_prompt(prompt, do_classifier_free_guidance=False, max_sequence_length=512,
                                           device=torch.device("cpu"))
        arrays[f"context{suffix}.{row}"] = states[0].numpy()
    for row, prompt in enumerate(PROMPTS):
        call = {"prompt": prompt, "height": HEIGHT, "width": WIDTH, "num_frames": FRAMES,
                "num_inference_steps": STEPS, "guidance_scale": GUIDANCE}
        with torch.no_grad():
            walked = pipe(**call, latents=latents[row:row + 1].clone(), output_type="latent").frames
            frames = pipe(**call, latents=latents[row:row + 1].clone(), output_type="np").frames
        arrays[f"latents{suffix}.{row}"] = walked[0].numpy()
        arrays[f"frames{suffix}.{row}"] = frames[0]
        print(f"pipeline{suffix} {prompt!r}: latents {tuple(walked.shape)} frames {frames.shape}")
    return arrays


def pipeline(destination: str) -> None:
    """`encoded_and_walked` as the pipeline runs, in float32, and its truth,
    the whole pipeline in float64 (`float64`: the Wan modules' float32 pins,
    the call's float32 latents and the UniPC tables widened with it), under
    `_f64` names."""
    root = output_root(destination)
    pipe = build_pipeline(root)
    generator = torch.Generator().manual_seed(SEED + 9)
    shape = (len(PROMPTS), PIPELINE_VAE["z_dim"], (FRAMES - 1) // 4 + 1, HEIGHT // 8, WIDTH // 8)
    latents = torch.randn(shape, generator=generator)
    arrays: dict[str, np.ndarray] = {"x_T": latents.numpy(), **encoded_and_walked(pipe, latents)}
    with float64():
        arrays.update(encoded_and_walked(copy.deepcopy(pipe).to(torch.float64), latents.double(), "_f64"))
    arrays.update(scheduler_replay())
    record = {"diffusers": DIFFUSERS, "torch": torch.__version__, "seed": SEED, "transformer": PIPELINE,
              "vae": PIPELINE_VAE, "prompts": PROMPTS, "frames": FRAMES, "height": HEIGHT, "width": WIDTH,
              "steps": STEPS, "guidance": GUIDANCE, "replay_steps": list(REPLAY_STEPS)}
    np.savez_compressed(root / "wan_pipeline.npz", **arrays)
    (root / "wan_pipeline.json").write_text(json.dumps(record, indent=1) + "\n")
    size = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    print(f"{root}: {size / 1e6:.2f} MB")


if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[1] == "bundle":
        from diffusers_dc_ae_reference import bundle

        bundle(sys.argv[2], sys.argv[3])
    elif len(sys.argv) == 3 and sys.argv[1] in ("transformer", "pipeline"):
        {"transformer": transformer, "pipeline": pipeline}[sys.argv[1]](sys.argv[2])
    else:
        raise SystemExit(__doc__)
