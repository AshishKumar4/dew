"""Sample a published text-to-image pipeline through Dew's native loader, for the landing page.

Z-Image-Turbo's text encoder (Qwen3, 8 GB in bf16) and its transformer (6B, 12 GB
in bf16) do not fit a 16 GB card together, and the workstation's GPU jobs run under
an 8 GB host-memory cap that Dew's diffusion loader, which builds the whole
translated tree on the host, would exceed. So this runs in two jobs, each with the
loader's own translation (`record_layouts`: same names, casts and transposes) but
every leaf placed on the device as soon as it is translated:

    encode  load only the conditioner (`HiddenStatesConditioner.from_pretrained`),
            encode every prompt as the pipeline's `prepare` does, save the states
    sample  load the pipeline without its text encoder, walk each prompt as the
            single call `task([prompt], steps=9, guidance=None, key=seed)` would,
            from the saved states and that call's own starting noise, then free
            the transformer and decode the latents with the pipeline's VAE

    dew-gpu-run env PYTHONPATH=src python tools/frontier_samples.py encode --out DIR
    dew-gpu-run env PYTHONPATH=src python tools/frontier_samples.py sample --out DIR --size 1024x1024
"""

import argparse
import contextlib
import gc
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

CHECKPOINT = "Tongyi-MAI/Z-Image-Turbo"
REVISION = "f332072aa78be7aecdf3ee76d5c247082da564a6"
# The model card's call: 9 inference steps and guidance_scale 0.0, which Dew spells
# guidance=None (the plain conditional prediction).
STEPS = 9
PROMPTS = {
    "dew-web": (
        "macro photograph of morning dew drops on a spider web at sunrise, golden backlight, "
        "shallow depth of field"
    ),
    "dew-leaf": (
        "a single dew drop on a bright green leaf reflecting a forest, macro photograph, crisp detail"
    ),
    "peaks": (
        "snow-capped mountain peaks above a sea of clouds at sunrise, aerial photograph, warm golden light"
    ),
    "aurora": (
        "green and purple northern lights over a frozen lake in Iceland, reflections on the "
        "ice, long exposure photograph"
    ),
    "fox": "a red fox sitting in fresh snow in a pine forest, soft morning light, wildlife photograph",
    "lighthouse": (
        "a lighthouse on a rocky coast during a storm, crashing waves, dramatic clouds, cinematic photograph"
    ),
    "reading-nook": (
        "a cozy reading nook by a rain-streaked window at night, warm lamp light, houseplants, "
        "film photograph"
    ),
    "neon": (
        'a neon sign that reads "dew" glowing in a rainy Tokyo alley at night, reflections on wet pavement'
    ),
    "fisherman": (
        "portrait of an old fisherman with a weathered face and a knitted wool cap, natural "
        "window light, 85mm photograph"
    ),
    "ramen": (
        "a bowl of ramen with a soft-boiled egg and steam rising, on a dark wooden table, food photography"
    ),
    "astronaut": "an astronaut floating above Earth at dawn, the thin blue atmosphere glowing, photograph",
    "kyoto": (
        "a Japanese garden in autumn with a red maple tree and a stone lantern beside a still "
        "pond, photograph"
    ),
    "watercolor": (
        "a watercolor painting of a Mediterranean harbor village with colorful houses and sailboats"
    ),
    "greenhouse": "a glass greenhouse full of lush plants on Mars at dusk, red desert outside, concept art",
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


@contextlib.contextmanager
def streamed(skip=()):
    """Dew's diffusion loader with each translated leaf put on the device as it
    is inserted, so the host holds one tensor at a time; a component in `skip`
    reads as having no tensors."""
    import jax

    from dew.interop import diffusion

    insert, component_tensors = diffusion.insert, diffusion.component_tensors

    def placed(tree, path, value, name):
        return insert(tree, path, jax.device_put(value), name)

    def tensors(directory, component):
        return {} if component in skip else component_tensors(directory, component)

    diffusion.insert, diffusion.component_tensors = placed, tensors
    try:
        yield
    finally:
        diffusion.insert, diffusion.component_tensors = insert, component_tensors


def device_record():
    import jax

    stats = jax.devices()[0].memory_stats() or {}
    return {
        "device": jax.devices()[0].device_kind,
        "peak_bytes_in_use": stats.get("peak_bytes_in_use"),
        "bytes_limit": stats.get("bytes_limit"),
    }


def environment(args):
    import jax

    return {
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "dew_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip(),
        "jax": jax.__version__,
        "dtype": "bfloat16",
        "param_dtype": "bfloat16",
    }


def encode(args):
    import jax

    from dew.inputs.diffusion import HiddenStatesConditioner

    tick = time.monotonic()
    with streamed():
        encoder = HiddenStatesConditioner.from_pretrained(
            args.checkpoint, revision=args.revision, dtype="bfloat16", param_dtype="bfloat16"
        )
    loaded = round(time.monotonic() - tick, 1)
    run = jax.jit(encoder.encode)
    states = {}
    for name in args.prompts:
        condition = run(encoder.params, encoder.tokenize([PROMPTS[name]]))
        states[f"{name}.context"] = np.asarray(condition.context.astype("float32"))
        states[f"{name}.mask"] = np.asarray(condition.mask)
        print(
            name,
            condition.context.shape,
            condition.context.dtype,
            int(condition.mask.sum()),
            "tokens",
            flush=True,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez(args.out / "states.npz", **states)
    save_json(
        args.out / "states.json",
        {
            **environment(args),
            "prompts": {name: PROMPTS[name] for name in args.prompts},
            "context_dtype": "bfloat16, saved as float32 (exact)",
            "load_seconds": loaded,
            "seconds": round(time.monotonic() - tick, 1),
            **device_record(),
        },
    )


def sample(args):
    import jax
    import jax.numpy as jnp

    from dew.artifacts import uint8_pixels
    from dew.diffusion.process import DenoisingCondition
    from dew.interop.pretrained import load_diffusion_source
    from dew.sampling.pipelines import DenoisingInputs

    height, width = (int(part) for part in args.size.lower().split("x"))
    states = np.load(args.out / "states.npz")
    started = time.monotonic()
    with streamed(skip=("text_encoder",)):
        pipeline = load_diffusion_source(
            args.checkpoint,
            revision=args.revision,
            dtype="bfloat16",
            param_dtype="bfloat16",
            size=(height, width),
        )
    task = pipeline.text_to_image()
    loaded = round(time.monotonic() - started, 1)
    print(
        f"loaded {height}x{width} in {loaded}s; solver {type(task.solver).__name__}; "
        f"pipeline default steps {task.steps}, guidance {task.guidance}",
        flush=True,
    )
    process, _ = task.prepared_process(args.steps)
    latents, records = {}, []
    for name in args.prompts:
        condition = DenoisingCondition(
            jnp.asarray(states[f"{name}.context"], jnp.bfloat16), mask=jnp.asarray(states[f"{name}.mask"])
        )
        for seed in args.seeds:
            if time.monotonic() - started > args.deadline:
                print("deadline: decoding what was sampled", flush=True)
                break
            tick = time.monotonic()
            # The noise `task([prompt], key=seed)` starts from: row 0 of that key.
            noise = process.noise(jax.random.fold_in(jax.random.key(seed), 0), task.latent_shape)[None]
            prepared = DenoisingInputs(
                noise, {"conditioning": condition}, {"conditioning": condition}, rows=1
            )
            result = task(prepared, steps=args.steps, guidance=None, key=seed, decode=False)
            latents[(name, seed)] = np.asarray(result.latents)
            seconds = round(time.monotonic() - tick, 2)
            records.append(
                {
                    "name": name,
                    "prompt": PROMPTS[name],
                    "seed": seed,
                    "steps": args.steps,
                    "guidance": None,
                    "solver": type(task.solver).__name__,
                    "height": height,
                    "width": width,
                    "denoise_seconds": seconds,
                }
            )
            print(name, seed, seconds, "s", flush=True)
    walked = device_record()
    autoencoder, vae = task.autoencoder, task.params["autoencoder"]
    finish = task.finish
    del task, pipeline
    gc.collect()
    decode = jax.jit(lambda params, z: jnp.clip(autoencoder.decode(params, z), -1.0, 1.0))
    if finish is not None:
        raise ValueError("this pipeline ships an output transform; decode it with the pipeline")
    for record in records:
        tick = time.monotonic()
        pixels = uint8_pixels(decode(vae, jnp.asarray(latents[(record["name"], record["seed"])])))[0]
        target = args.out / f"{record['name']}-s{record['seed']}-{width}x{height}.png"
        Image.fromarray(pixels).save(target)
        record.update(
            file=target.name,
            sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            decode_seconds=round(time.monotonic() - tick, 2),
        )
    manifest = args.out / f"manifest-{width}x{height}.json"
    save_json(
        manifest,
        {
            **environment(args),
            "load_seconds": loaded,
            "sampling": walked,
            "decoding": device_record(),
            "seconds": round(time.monotonic() - started, 1),
            "images": records,
        },
    )
    print("complete", len(records), "images", round(time.monotonic() - started), "s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("encode", "sample"))
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", default="1024x1024", help="HEIGHTxWIDTH")
    parser.add_argument("--prompts", nargs="+", choices=PROMPTS, default=list(PROMPTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--deadline", type=int, default=1300, help="seconds of sampling before decoding")
    args = parser.parse_args()
    if Path(args.checkpoint).is_dir():
        args.revision = None
    {"encode": encode, "sample": sample}[args.command](args)


if __name__ == "__main__":
    main()
