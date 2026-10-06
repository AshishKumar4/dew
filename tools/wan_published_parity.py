"""Published Wan 2.1 T2V-1.3B against Diffusers 0.34.0, in float32, stage by stage.

Diffusers runs first and is freed, then Dew loads the pipeline as a user
does (`load_diffusion_source` at the clip's geometry) and runs the same
stages. Each gap is the largest difference over the larger of 1 and the
reference's largest value, and beside it the RMS difference over the
reference's RMS:

- `states`: two prompts and the empty negative through
  `WanPipeline._get_t5_prompt_embeds` at 512 tokens, against
  `WanConditioner` (UMT5-XXL, 5.7B parameters);
- `prediction`: one transformer call on fixed noise at timestep 700 for
  both prompts, each side on the reference's prompt states;
- `walk`: the unmodified `WanPipeline` call from fixed latents at `--steps`,
  guided at its default 5.0 over its UniPC grid, on the reference's prompt
  states, in float32 and again widened to float64 (scheduler tables
  included, `diffusers_wan_reference.widened`), against Dew's
  `text_to_image()` from the same latents and states: the latent each ends
  on.

Both sides compute in true float32: TF32 is off in torch and JAX multiplies
at HIGHEST. A single call is held to an absolute bound (`BOUNDS`). A walk
carries every step's rounding into the next through the model, so its two
float32 runs part further than any one call does; it is held to
tests/reference_error.py's rule instead, Dew's RMS distance from the
float64 walk at most twice the float32 source's (`FACTOR`). The scheduler
alone is held to the same rule on recorded model outputs in
tests/test_wan_source.py. The run exits 1 when a stage fails its bound, so
a command after it stops on a failed parity.

The prompt states alone are 22.7 GB, so the reference runs on a GPU with
that room (CUDA torch), or on the host. `--stages prediction` needs only
the transformer; it reads a fixed stand-in for the prompt states
(unit-variance states on the first 24 positions, zeros after, as a padded
prompt looks). `--reference FILE` keeps the reference arrays: written when
absent, read when present, so the two sides can run as two processes, the
first with `--reference-only`.

    python tools/wan_published_parity.py [--frames 17 --height 480 --width 832 --steps 10]
        [--stages states,prediction,walk] [--reference FILE [--reference-only]] > wan_parity.json
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import transformers.utils as transformers_utils
from diffusers_wan_reference import widened

# Diffusers 0.34.0's pipeline modules import two names Transformers dropped
# after 4.x, as tools/diffusers_sd3_reference.py restores them.
for _name, _value in (("FLAX_WEIGHTS_NAME", "flax_model.msgpack"),
                      ("WEIGHTS_INDEX_NAME", "pytorch_model.bin.index.json")):
    if not hasattr(transformers_utils, _name):
        setattr(transformers_utils, _name, _value)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
PROMPTS = ["A red fox trotting through fresh snow at sunrise, its breath steaming in the cold air",
           "Waves rolling onto a black sand beach under a stormy sky"]
TIMESTEP, GUIDANCE, SEED = 700.0, 5.0, 0
# One call measured 2.0e-5 (RTX 4080 against torch on the host) and 1.7e-5
# (A100); a wrong shift, scale, sign or mask moves it by 1e-2 and more.
BOUNDS = {"states": 1e-4, "prediction": 1e-4}
FACTOR = 2.0
"""tests/reference_error.py's: Dew's RMS distance from float64 over the source's."""


def distance(value, truth) -> float:
    """The root-mean-square difference over every entry, in float64."""
    difference = np.asarray(value, np.float64) - np.asarray(truth, np.float64)
    return float(np.sqrt(np.mean(np.square(difference))))


def gap(actual, expected) -> dict[str, float]:
    actual, expected = np.asarray(actual, np.float64), np.asarray(expected, np.float64)
    difference = actual - expected
    return {"max": float(np.abs(difference).max() / max(1.0, float(np.abs(expected).max()))),
            "rms": float(np.sqrt(np.mean(difference ** 2) / np.mean(expected ** 2)))}


def channels_last(value):
    return np.moveaxis(np.asarray(value), 1, -1)


def free():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def device_weights(directory) -> dict:
    """The transformer's parameters, translated and placed one tensor at a
    time, so the host holds one float32 tensor rather than all 5.7 GB."""
    import jax

    from dew.interop.diffusion import component_tensors, translate_wan_weights
    from dew.interop.weights import insert

    tree: dict = {}
    for name, tensor in component_tensors(directory, "transformer").items():
        part, _ = translate_wan_weights({name: tensor})
        for path, leaf in jax.tree_util.tree_leaves_with_path(part):
            insert(tree, tuple(key.key for key in path), jax.device_put(leaf), name)
    return {"params": tree}


def stand_in_states(width: int) -> np.ndarray:
    states = np.zeros((len(PROMPTS), 512, width), np.float32)
    states[:, :24] = np.random.default_rng(SEED + 1).standard_normal((len(PROMPTS), 24, width))
    return states


def reference(directory, args, stages) -> dict[str, np.ndarray]:
    """Every stage's reference arrays, with Diffusers' own modules."""
    import types

    import torch
    from diffusers import UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel
    from transformers import AutoTokenizer, UMT5EncoderModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays: dict[str, np.ndarray] = {"x_T": args.latents}
    if stages & {"states", "walk"}:
        encoder = UMT5EncoderModel.from_pretrained(directory / "text_encoder", torch_dtype=torch.float32)
        held = types.SimpleNamespace(text_encoder=encoder.to(device), _execution_device=device,
                                     tokenizer=AutoTokenizer.from_pretrained(directory / "tokenizer"))
        with torch.no_grad():
            arrays["states"] = WanPipeline._get_t5_prompt_embeds(
                held, [*PROMPTS, ""], max_sequence_length=512, device=device).cpu().numpy()
        del encoder, held
        free()
    states = arrays.get("states")
    if states is None:
        width = json.loads((directory / "transformer" / "config.json").read_text())["text_dim"]
        states = np.concatenate([stand_in_states(width), np.zeros((1, 512, width), np.float32)])
        arrays["states"] = states
    transformer = WanTransformer3DModel.from_pretrained(directory / "transformer", torch_dtype=torch.float32)
    transformer = transformer.to(device)
    if "prediction" in stages:
        noise = torch.from_numpy(np.repeat(args.latents[:1], len(PROMPTS), axis=0)).to(device)
        with torch.no_grad():
            arrays["prediction"] = transformer(
                noise, torch.full((len(PROMPTS),), TIMESTEP, device=device),
                torch.from_numpy(states[:len(PROMPTS)]).to(device), return_dict=False)[0].cpu().numpy()
    if "walk" in stages:
        # float32 first: the float64 run widens the transformer in place.
        for precision, dtype in (("fp32", torch.float32), ("fp64", torch.float64)):
            with widened() if dtype == torch.float64 else contextlib.nullcontext():
                transformer = transformer.to(dtype)
                scheduler = UniPCMultistepScheduler.from_pretrained(directory / "scheduler")
                pipe = WanPipeline(tokenizer=None, text_encoder=None, transformer=transformer, vae=None,
                                   scheduler=scheduler)
                pipe.set_progress_bar_config(disable=True)
                for row in range(len(PROMPTS)):
                    with torch.no_grad():
                        start = time.perf_counter()
                        latents = pipe(
                            prompt_embeds=torch.from_numpy(states[row:row + 1]).to(device, dtype),
                            negative_prompt_embeds=torch.from_numpy(states[-1:]).to(device, dtype),
                            height=args.height, width=args.width, num_frames=args.frames,
                            num_inference_steps=args.steps, guidance_scale=GUIDANCE,
                            latents=torch.from_numpy(args.latents[row:row + 1]).to(device, dtype),
                            output_type="latent").frames
                    arrays[f"{precision}.seconds.{row}"] = np.asarray(time.perf_counter() - start)
                    arrays[f"{precision}.latents.{row}"] = latents.cpu().numpy()
                del pipe
    del transformer
    free()
    return arrays


def native(directory, args, stages, expected) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Dew's side of every stage: each gap against the reference, and the
    stages that failed their bound."""
    import jax
    import jax.numpy as jnp

    from dew.diffusion.process import DenoisingCondition
    from dew.interop.diffusion_components import _denoiser
    from dew.interop.pretrained import load_diffusion_source
    from dew.nn.autoencoders.wan import load_wan_vae
    from dew.sampling.pipelines import DenoisingInputs

    gaps: dict[str, dict[str, float]] = {}
    if "prediction" in stages:
        # The transformer alone, which is all this stage reads.
        denoiser = _denoiser(directory, dtype="float32", attention_impl=args.attention)
        params = device_weights(directory)
        noise = jnp.asarray(channels_last(np.repeat(args.latents[:1], len(PROMPTS), axis=0)))
        states = DenoisingCondition(jnp.asarray(expected["states"][:len(PROMPTS)]))
        flow = jax.jit(denoiser.model.apply)(params, noise, jnp.full((len(PROMPTS),), TIMESTEP), states)
        gaps["prediction"] = gap(flow, channels_last(expected["prediction"]))
        del denoiser, params, flow
        gc.collect()
    size = (args.frames, args.height, args.width)
    if "states" in stages:
        pipeline = load_diffusion_source(str(directory), dtype="float32", param_dtype="float32",
                                         attention_impl=args.attention, size=size)
        encoder = pipeline.inputs.conditions["conditioning"].encoder
        states = encoder.encode(pipeline.variables["encoders"]["conditioning"],
                                encoder.tokenize([*PROMPTS, ""])).context
        gaps["states"] = gap(states, expected["states"])
        del pipeline, encoder, states
        gc.collect()
    failed = [name for name, value in gaps.items() if value["max"] > BOUNDS[name]]
    if "walk" in stages:
        # The walk reads the reference's states, so it never runs UMT5: the
        # transformer streams to the device and the VAE is bound unread.
        _, vae, _, _ = load_wan_vae(directory)
        pipeline = load_diffusion_source(
            str(directory), dtype="float32", param_dtype="float32", attention_impl=args.attention, size=size,
            variables={**device_weights(directory), "encoders": {"conditioning": {"text_encoder": {}}},
                       "autoencoder": vae})
        task = pipeline.text_to_image()
        states = expected["states"]
        dew, source, truth = [], [], []
        for row in range(len(PROMPTS)):
            inputs = DenoisingInputs(
                jnp.asarray(channels_last(args.latents[row:row + 1])),
                {"conditioning": DenoisingCondition(jnp.asarray(states[row:row + 1]))},
                {"conditioning": DenoisingCondition(jnp.asarray(states[-1:]))}, rows=1, grid_steps=args.steps)
            dew.append(np.asarray(task(inputs, decode=False, key=0).latents))
            source.append(channels_last(expected[f"fp32.latents.{row}"]))
            truth.append(channels_last(expected[f"fp64.latents.{row}"]))
            gaps[f"walk.{row}"] = gap(dew[-1], source[-1])
        mine, theirs = distance(dew, truth), distance(source, truth)
        gaps["walk.rule"] = {"dew_from_float64": mine, "source_from_float64": theirs, "ratio": mine / theirs}
        if not mine <= FACTOR * theirs:
            failed.append("walk.rule")
    return gaps, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--stages", default="states,prediction,walk")
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--attention", default="auto", help="Dew's attention_impl")
    parser.add_argument("--source", help="a pipeline directory instead of the published repo")
    args = parser.parse_args()
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    stages = set(args.stages.split(","))
    from dew.interop import sources

    if args.source is None:
        directory = sources.snapshot(REPO, args.revision, weights=("text_encoder", "transformer", "vae")
                                     if stages & {"states", "walk"} else ("transformer", "vae"))
    else:
        directory = Path(args.source)
    channels = json.loads((directory / "transformer" / "config.json").read_text())["in_channels"]
    latent = ((args.frames - 1) // 4 + 1, args.height // 8, args.width // 8)
    args.latents = np.random.default_rng(SEED).standard_normal((len(PROMPTS), channels, *latent)).astype(
        np.float32)
    start = time.perf_counter()
    if args.reference is not None and args.reference.is_file():
        with np.load(args.reference) as held:
            expected = dict(held)
    else:
        expected = reference(directory, args, stages)
        if args.reference is not None:
            np.savez(args.reference, **expected)
    reference_seconds = time.perf_counter() - start
    if args.reference_only:
        print(json.dumps({"reference": str(args.reference), "seconds": reference_seconds}))
        return
    start = time.perf_counter()
    gaps, failed = native(directory, args, stages, expected)

    import jax

    print(json.dumps({
        "repo": args.source or REPO, "revision": None if args.source else args.revision,
        "attention_impl": args.attention, "geometry": [args.frames, args.height, args.width],
        "steps": args.steps, "guidance": GUIDANCE, "stages": sorted(stages), "gaps": gaps,
        "seconds": {"reference": reference_seconds, "dew": time.perf_counter() - start},
        "precision": {"jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
                      "torch_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                      "torch_cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                      "jax_devices": [str(device) for device in jax.devices()],
                      "torch_device": "cuda" if torch.cuda.is_available() else "cpu",
                      "versions": {"jax": jax.__version__, "torch": torch.__version__}}}, indent=1))
    if failed:
        raise SystemExit(f"failed {failed}: single calls are held to {BOUNDS}, the walk to {FACTOR} times "
                         "the source's RMS distance from float64")


if __name__ == "__main__":
    main()
