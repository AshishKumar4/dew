"""Text-to-video samples from Wan 2.1 T2V-1.3B through Dew, with their provenance.

Loads the published pipeline with `load_diffusion_source` at the clip's
geometry, computing in bfloat16 over float32 weights, which is what the
source's bfloat16 pipeline computes, its float32-kept modules included.
Every prompt is encoded first; the text encoder's weights then leave the
task, and the transformer's and the VAE's go to the device once. Each clip
is sampled with the source's own policy (UniPC over its flow-shifted grid,
guidance 5.0) and the negative prompt Diffusers' Wan example uses, then
decoded, and written as an H.264 MP4 at Wan's 16 frames per second, its
first frame as a PNG poster, and one entry in `manifest.json`: prompt,
seed, steps, guidance, solver, geometry, revision, Dew commit, hardware and
the seconds each part took. A clip already written is skipped, so a run
that stopped resumes.

    python tools/wan_samples.py OUTPUT_DIR [--frames 49 --height 480 --width 832 --steps 50]
        [--source DIR] [--only NAME ...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
REVISION = "0fad780a534b6463e45facd96134c9f345acfa5b"
FPS = 16
NEGATIVE = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra "
    "fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, "
    "still picture, messy background, three legs, many people in the background, walking backwards"
)
CLIPS = {
    "forest-sunrise": (0, "A slow aerial shot gliding over a misty pine forest at sunrise, golden light "
                          "breaking through the fog between the trees, cinematic, highly detailed"),
    "ocean-sunset": (1, "Ocean waves crashing against dark volcanic rocks at sunset, the spray glowing "
                        "orange in the backlight, slow motion, cinematic"),
    "hummingbird": (2, "A hummingbird hovering beside bright red flowers in a sunlit garden, its wings a "
                       "soft blur, shallow depth of field, nature documentary footage"),
    "aurora-lake": (3, "Green northern lights rippling across a starry night sky above a snowy mountain "
                       "lake, the aurora reflected and shimmering on the still water"),
    "koi-pond": (4, "Orange and white koi fish gliding through clear water among lily pads, sunlight "
                    "dappling the surface, top-down view, calm and serene"),
}


def commit() -> str:
    found = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                           cwd=Path(__file__).resolve().parent, check=False)
    return found.stdout.strip() if found.returncode == 0 else "unknown (not a git checkout)"


def write_clip(frames: np.ndarray, path: Path) -> None:
    """`[T, H, W, 3]` uint8 frames as H.264 in a yuv420p MP4 that browsers play."""
    _, height, width, _ = frames.shape
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                    "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(path)], input=frames.tobytes(), check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--frames", type=int, default=49)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--source", default=REPO, help="a pipeline directory instead of the published repo")
    parser.add_argument("--only", nargs="*", choices=sorted(CLIPS), default=sorted(CLIPS))
    args = parser.parse_args()

    import jax
    from PIL import Image

    from dew.artifacts import uint8_pixels
    from dew.interop.pretrained import load_diffusion_source

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {"clips": {}}
    pending = [name for name in args.only if not (args.output / f"{name}.mp4").is_file()]
    if not pending:
        return
    revision = REVISION if args.source == REPO else None
    start = time.perf_counter()
    pipeline = load_diffusion_source(args.source, revision=revision, dtype="bfloat16", param_dtype="float32",
                                     size=(args.frames, args.height, args.width))
    load_seconds = time.perf_counter() - start
    task = pipeline.text_to_image()
    prepared, encode_seconds = {}, {}
    for name in pending:
        seed, prompt = CLIPS[name]
        start = time.perf_counter()
        prepared[name] = task.prepare(prompt, unconditional=NEGATIVE, key=seed, steps=args.steps)
        jax.block_until_ready(prepared[name])
        encode_seconds[name] = time.perf_counter() - start
    # The prompts are encoded: the 5.7B-parameter encoder leaves, and what
    # sampling reads goes to the device once.
    resident = jax.device_put({name: value for name, value in pipeline.variables.items()
                               if name != "encoders"})
    task = task.bind({**resident, "encoders": {}})
    device = jax.devices()[0]
    for index, name in enumerate(pending):
        seed, prompt = CLIPS[name]
        start = time.perf_counter()
        latents = jax.block_until_ready(task(prepared[name], decode=False, key=seed).latents)
        sample_seconds = time.perf_counter() - start
        start = time.perf_counter()
        pixels = jax.block_until_ready(pipeline.autoencoder.decode(resident["autoencoder"], latents))
        decode_seconds = time.perf_counter() - start
        frames = uint8_pixels(np.asarray(pixels[0], np.float32))
        write_clip(frames, args.output / f"{name}.mp4")
        Image.fromarray(frames[0]).save(args.output / f"{name}.png")
        manifest["clips"][name] = {
            "video": f"{name}.mp4", "poster": f"{name}.png", "prompt": prompt, "negative_prompt": NEGATIVE,
            "seed": seed, "steps": args.steps, "guidance": task.guidance.scale if task.guidance else None,
            "solver": {"name": type(task.solver).__name__,
                       **{key: value for key, value in vars(task.solver).items()
                          if isinstance(value, (bool, int, float, str))}},
            "scheduler": dict(pipeline.config["scheduler"]),
            "frames": args.frames, "fps": FPS, "height": args.height, "width": args.width,
            "repo": args.source, "revision": pipeline.revision, "dew_commit": commit(),
            "dtype": {"compute": "bfloat16", "params": "float32"},
            "hardware": device.device_kind, "platform": device.platform, "jax": jax.__version__,
            "seconds": {"load": load_seconds, "encode": encode_seconds[name],
                        "sample": sample_seconds, "decode": decode_seconds,
                        "includes_compile": index == 0},
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        manifest_path.write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"{name}: sampled {sample_seconds:.1f}s, decoded {decode_seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
