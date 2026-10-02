"""Sample, score and pick landing-page images from dewml/hybrid-dit-176m.

Every image is the single call the live cell makes, `pipe([prompt], key=seed, ...)`
with `prepare(..., unconditional=negative)`: rows are batched for speed, but each
row's starting noise is the noise that single call draws (row 0 of its key), so a
manifest entry reproduces with one call. `verify` checks that on a sample of rows;
`finals` re-runs the picks one at a time and writes those pixels.

    dew-gpu-run env PYTHONPATH=src python tools/landing_visuals.py generate dpm40_cfg6i_negB --seeds 0 1 2 3
    python tools/landing_visuals.py sheets
"""

import argparse
import hashlib
import json
import math
import subprocess
import textwrap
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/mnt/scratch/dew/site-samples/landing-visuals/176m")
REPO = Path(__file__).resolve().parents[1]
MODEL = "dewml/hybrid-dit-176m"
REVISION = "90d1717ac262e1b33a65ac602812ef442129c7ea"
CLIP = "openai/clip-vit-large-patch14"
AESTHETIC = Path("/mnt/scratch/dew/site-samples/curation/models/sac+logos+ava1-l14-linearMSE.pth")
AESTHETIC_SHA256 = "21dd590f3ccdc646f0d53120778b296013b096a035a2718c9cb0d511bff0f1e0"
NEGATIVES = {
    "negA": "letterbox, white border, black border, border, frame, blurry, text, watermark, lowres, collage",
    "negB": (
        "letterbox, white border, black border, frame, text, watermark, collage, blurry, lowres, "
        "low quality, dull colors, washed out, low contrast, grainy"
    ),
    "negP": "white border, frame, text, watermark, collage, blurry, low quality",
}
SUFFIX = {"plain": "", "vivid": ", vivid colors, dramatic lighting, highly detailed"}
PROMPTS = {
    "aurora": "the northern lights over a frozen lake at night",
    "aurora2": "green and purple northern lights reflected in a frozen lake, snowy mountains at night",
    "lake": "a mountain lake reflecting snowy peaks",
    "waterfall": "a waterfall in a lush green forest",
    "alpine": "a turquoise alpine lake surrounded by pine trees and rugged mountains",
    "roerich": "a red and gold mountain landscape, a painting in the style of Nicholas Roerich",
    "turner": "a misty mountain valley at dawn, a watercolor painting by J. M. W. Turner",
    "autumn-oil": "an oil painting of a forest in autumn, golden leaves and a winding stream",
    "lavender": "a lavender field at sunset with distant mountains and a pink sky",
    "lighthouse": "a lighthouse on a rocky coast at sunset, dramatic waves and golden light",
    "castle": "a castle on a rocky hill above a misty valley at sunrise",
    "dunes": "rolling sand dunes in the desert at sunset, deep orange sand and purple sky",
    "milkyway": "the milky way above snowy mountains, a clear starry night sky",
    "cathedral": "a Gothic cathedral interior with glowing stained glass windows and stone arches",
    "vangogh": "a swirling starry night over a quiet village, an oil painting by Vincent van Gogh",
    "hokusai": "a Japanese woodblock print of ocean waves with Mount Fuji, in the style of Hokusai",
    "canyon": "a canyon with towering red rock cliffs and a winding river at sunset",
    "balloons": "colorful hot air balloons floating above a green valley at sunrise",
    "village": "a snowy mountain village at dusk, warm glowing windows and blue snow",
    "bridge": "an old stone bridge crossing a river in autumn, golden trees and mist",
    "poppies": "a field of red poppies under a dramatic blue sky with white clouds",
    "bamboo": "a bamboo forest with golden sunlight streaming between tall green stalks",
    "lagoon": "a tropical lagoon with turquoise water and palm trees, aerial view",
    "volcano": "a volcano erupting at night, glowing red lava beneath a starry sky",
    "sunset": "a dramatic sunset over the ocean, fiery orange and pink clouds",
    "temple": "a Japanese temple beside a cherry blossom tree on a spring morning",
    "santorini": "white houses with blue domes above the sea in Santorini at sunset",
    "cabin": "a cozy wooden cabin in a snowy forest at night, warm light in the windows",
    "ship": "a tall ship sailing on a stormy sea, an oil painting",
    "skycastle": "a fantasy castle on a cliff above the clouds, digital painting",
    "venice": "a canal in Venice at sunset with gondolas and old buildings",
    "ghibli": "a lush green valley with a winding river and white clouds, Studio Ghibli style landscape",
    "jungle-temple": "an ancient temple in the jungle covered with vines, golden light",
    "icecave": "a glowing blue ice cave with sunlight shining through the ice",
    "terraces": "green rice terraces on the hills at sunrise with morning mist",
    "meadow": "the Swiss Alps in summer, a green meadow with wildflowers below snowy peaks",
    "autumn-road": "a road through an autumn forest with red and orange trees",
    "inkwash": "a Chinese ink wash painting of misty mountains and pine trees",
    "monet": "a garden with water lilies and a wooden bridge, an impressionist painting by Claude Monet",
    "sunflowers": "a field of sunflowers in southern France, an oil painting by Vincent van Gogh",
    "maple": "a red maple tree beside a calm lake in autumn, mountains in the background",
    "cloudsea": "a snowy mountain peak at sunrise above a sea of clouds",
    "tulips": "a field of colorful tulips with a windmill in the Netherlands under a blue sky",
    "lanterns": "a medieval town square at night lit by warm lanterns, cobblestone streets",
    "iceland": "a waterfall in Iceland at sunset, green hills and a golden sky",
    "pagoda": "a pagoda on a misty mountain surrounded by pine trees, a Chinese landscape painting",
    "fireworks": "colorful fireworks over a harbor city at night, reflections in the water",
    "cottage": "a stone cottage with a flower garden in the English countryside on a summer afternoon",
    "glacier": "a turquoise glacier lagoon with floating icebergs under a pink sky",
    "fjord": "a Norwegian fjord with steep green cliffs and a small red village in summer",
}
CONFIGS = {
    "dpm40_cfg6i_negA": {
        "solver": "DPMSolverMultistep",
        "steps": 40,
        "cfg": 6.0,
        "interval": [0.15, 0.9],
        "negative": "negA",
    },
    "dpm40_cfg6i_negB": {
        "solver": "DPMSolverMultistep",
        "steps": 40,
        "cfg": 6.0,
        "interval": [0.15, 0.9],
        "negative": "negB",
    },
    "dpm40_cfg7.5i_negB": {
        "solver": "DPMSolverMultistep",
        "steps": 40,
        "cfg": 7.5,
        "interval": [0.15, 0.9],
        "negative": "negB",
    },
    "unipc30_cfg6i2_negB": {
        "solver": "UniPC",
        "steps": 30,
        "cfg": 6.0,
        "interval": [0.2, 0.9],
        "negative": "negB",
    },
    "dpm60_cfg5i_negB": {
        "solver": "DPMSolverMultistep",
        "steps": 60,
        "cfg": 5.0,
        "interval": [0.15, 0.9],
        "negative": "negB",
    },
    # Live-cell candidates: the cell's current default, then the strong recipe at CPU step counts.
    "dpm15_cfg5": {
        "solver": "DPMSolverMultistep",
        "steps": 15,
        "cfg": 5.0,
        "interval": [0.0, 1.0],
        "negative": None,
    },
    "dpm15_cfg6i_negB": {
        "solver": "DPMSolverMultistep",
        "steps": 15,
        "cfg": 6.0,
        "interval": [0.15, 0.9],
        "negative": "negB",
    },
    "dpm20_cfg6i_negB": {
        "solver": "DPMSolverMultistep",
        "steps": 20,
        "cfg": 6.0,
        "interval": [0.15, 0.9],
        "negative": "negB",
    },
    "dpm20_cfg7.5i_negB": {
        "solver": "DPMSolverMultistep",
        "steps": 20,
        "cfg": 7.5,
        "interval": [0.15, 0.9],
        "negative": "negB",
    },
    "unipc15_cfg6i2_negB": {
        "solver": "UniPC",
        "steps": 15,
        "cfg": 6.0,
        "interval": [0.2, 0.9],
        "negative": "negB",
    },
}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def full_prompt(name, variant):
    return PROMPTS[name] + SUFFIX[variant]


def border_bands(pixels):
    """Flat edge strips, counted per edge (ObedientKite's curation filter)."""
    pixels = pixels.astype(np.float32)
    columns = pixels.transpose(1, 0, 2)
    result = {}
    for name, lines in (
        ("top", pixels),
        ("bottom", pixels[::-1]),
        ("left", columns),
        ("right", columns[::-1]),
    ):
        count = 0
        for line in lines[: len(lines) // 4]:
            spatial_std = line.std(axis=0).mean()
            spatial_range = (np.percentile(line, 95, axis=0) - np.percentile(line, 5, axis=0)).mean()
            if spatial_std > 7.0 or spatial_range > 25.0:
                break
            count += 1
        result[name] = count
    return result


def content_box(bands, size=256, margin=2):
    """(left, top, right, bottom) inside the flat bands, with `margin` more
    pixels off a banded edge for the soft line the VAE decodes beside a band."""

    def inset(edge):
        return bands[edge] + margin if bands[edge] else 0

    return inset("left"), inset("top"), size - inset("right"), size - inset("bottom")


def framing(bands, size=256):
    """'square' with no band on any edge; 'wide' when only the top and bottom
    carry bands and what is left still fills a 3:2 figure (within 4%);
    'portrait' the same for side bands at 3:4; else 'letterboxed'."""
    if not any(bands.values()):
        return "square"
    left, top, right, bottom = content_box(bands, size)
    width, height = right - left, bottom - top
    if left == 0 and right == size and width / height <= 1.5 * 1.04:
        return "wide"
    if top == 0 and bottom == size and width / height >= 0.75 / 1.04:
        return "portrait"
    return "letterboxed"


def display_box(bands, aspect=1.5, size=256):
    """The largest centred `aspect` box inside the content box."""
    left, top, right, bottom = content_box(bands, size)
    width, height = right - left, bottom - top
    if width / height > aspect:
        cut = width - round(height * aspect)
        left, right = left + cut // 2, right - (cut - cut // 2)
    else:
        cut = height - round(width / aspect)
        top, bottom = top + cut // 2, bottom - (cut - cut // 2)
    return left, top, right, bottom


def colorfulness(pixels):
    """Hasler and Suesstrunk (2003) colourfulness M over RGB uint8 pixels."""
    pixels = pixels.astype(np.float32)
    rg = pixels[..., 0] - pixels[..., 1]
    yb = 0.5 * (pixels[..., 0] + pixels[..., 1]) - pixels[..., 2]
    return float(np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean()))


def contrast(pixels):
    luminance = pixels.astype(np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return float(luminance.std())


class Scorer:
    """CLIPScore against the subject prompt and the LAION aesthetic predictor
    (sac+logos+ava1-l14-linearMSE) over Dew's CLIP ViT-L/14 image features."""

    def __init__(self):
        import jax
        import jax.numpy as jnp
        import torch

        from dew.data.text import load_tokenizer
        from dew.eval.images import _get_clip

        if hashlib.sha256(AESTHETIC.read_bytes()).hexdigest() != AESTHETIC_SHA256:
            raise ValueError("aesthetic head checksum mismatch")
        self.tokenizer = load_tokenizer(CLIP)
        self.model, self.processor = _get_clip(CLIP)
        state = torch.load(AESTHETIC, map_location="cpu", weights_only=True)
        weights = [
            (
                jnp.asarray(state[f"layers.{index}.weight"].numpy()),
                jnp.asarray(state[f"layers.{index}.bias"].numpy()),
            )
            for index in (0, 2, 4, 6, 7)
        ]

        @jax.jit
        def aesthetic(features):
            features = features / jnp.linalg.norm(features, axis=-1, keepdims=True)
            for weight, bias in weights:
                features = jnp.matmul(features, weight.T, precision=jax.lax.Precision.HIGHEST) + bias
            return features[:, 0]

        self.aesthetic = aesthetic

    def __call__(self, pixels, prompts):
        from dew.eval.images import clip_image_text_cosine

        tokens = self.tokenizer(
            list(prompts),
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="np",
        )
        cosine = np.asarray(clip_image_text_cosine(pixels, tokens["input_ids"], tokens["attention_mask"]))
        processed = self.processor(images=pixels, return_tensors="np")["pixel_values"]
        aesthetic = np.asarray(self.aesthetic(self.model.get_image_features(processed)))
        return 100.0 * np.maximum(cosine, 0.0), aesthetic


def load_pipe():
    from dew.interop.hub import pull_from_hub
    from dew.sampling import TextToImage

    directory = pull_from_hub(MODEL, revision=REVISION)
    return TextToImage.from_run(str(directory)), directory


def provenance(pipe, directory):
    import jax

    return {
        "model": MODEL,
        "revision": REVISION,
        "loaded_with": f"dew.sampling.TextToImage.from_run(pull_from_hub({MODEL!r}, revision=...)), "
        "the run from_pretrained loads",
        "checkpoint_step": 1350000,
        "parameters": int(sum(math.prod(leaf.shape) for leaf in jax.tree.leaves(pipe.params["params"]))),
        "dew_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "device": jax.devices()[0].device_kind,
        "jax": jax.__version__,
        "compute": "published run dtypes, unquantized",
        "run_json_sha256": hashlib.sha256((Path(directory) / "run.json").read_bytes()).hexdigest(),
    }


def call_settings(config):
    import dew.sampling as sampling

    solver = getattr(sampling, config["solver"])()
    guidance = sampling.CFG(config["cfg"], interval=tuple(config["interval"]))
    return solver, guidance, None if config["negative"] is None else NEGATIVES[config["negative"]]


def single_call_noise(pipe, seeds, steps):
    """The starting noise `pipe([prompt], key=seed)` draws, one row per seed."""
    import jax

    process, _ = pipe.prepared_process(steps)
    keys = jax.numpy.stack([jax.random.fold_in(jax.random.key(int(seed)), 0) for seed in seeds])
    return jax.vmap(lambda key: process.noise(key, pipe.latent_shape))(keys)


def batched(pipe, rows, config, decode=True):
    """`rows` of (prompt, seed) sampled together, each as its own single call."""
    import jax

    solver, guidance, negative = call_settings(config)
    prompts = [prompt for prompt, _ in rows]
    prepared = pipe.prepare(prompts, key=int(rows[0][1]), steps=config["steps"], unconditional=negative)
    noise = single_call_noise(pipe, [seed for _, seed in rows], config["steps"])
    prepared = prepared.replace(
        noise=jax.device_put(noise.astype(prepared.noise.dtype), prepared.noise.sharding)
    )
    return pipe(
        prepared, key=int(rows[0][1]), steps=config["steps"], solver=solver, guidance=guidance, decode=decode
    )


def single(pipe, prompt, seed, config, decode=True):
    """Exactly the live cell's call, with the negative prompt and interval guidance."""
    solver, guidance, negative = call_settings(config)
    prepared = pipe.prepare([prompt], key=seed, steps=config["steps"], unconditional=negative)
    return pipe(prepared, key=seed, steps=config["steps"], solver=solver, guidance=guidance, decode=decode)


def generate(args):
    from dew.artifacts import uint8_pixels

    started = time.monotonic()
    config = CONFIGS[args.config]
    pipe, directory = load_pipe()
    save_json(ROOT / "provenance.json", provenance(pipe, directory))
    save_json(
        ROOT / "experiment.json",
        {"prompts": PROMPTS, "suffix": SUFFIX, "negatives": NEGATIVES, "configs": CONFIGS},
    )
    print("loaded", round(time.monotonic() - started, 1), flush=True)
    scorer = Scorer()
    scorefile = ROOT / "scores" / f"{args.config}.jsonl"
    scorefile.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if scorefile.exists():
        done = {row["id"] for row in map(json.loads, scorefile.read_text().splitlines())}
    names = args.prompts or list(PROMPTS)
    todo = [
        (name, variant, seed)
        for name in names
        for variant in args.variants
        for seed in args.seeds
        if f"{args.config}/{name}.{variant}.s{seed:02d}" not in done
    ]
    with scorefile.open("a", buffering=1) as scores:
        for start in range(0, len(todo), args.batch_size):
            if time.monotonic() - started > args.deadline:
                print("deadline; resume same command", flush=True)
                return
            group = todo[start : start + args.batch_size]
            tick = time.monotonic()
            images = batched(
                pipe, [(full_prompt(name, variant), seed) for name, variant, seed in group], config
            )
            pixels = uint8_pixels(images.host().images)
            clip, aesthetic = scorer(pixels, [PROMPTS[name] for name, _, _ in group])
            for row, (name, variant, seed) in enumerate(group):
                identifier = f"{args.config}/{name}.{variant}.s{seed:02d}"
                target = ROOT / "pool" / f"{identifier}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels[row]).save(target)
                bands = border_bands(pixels[row])
                box = display_box(bands)
                scores.write(
                    json.dumps(
                        {
                            "id": identifier,
                            "name": name,
                            "variant": variant,
                            "seed": seed,
                            "config": args.config,
                            "prompt": full_prompt(name, variant),
                            "subject": PROMPTS[name],
                            "settings": config,
                            "file": str(target),
                            "clip_score": float(clip[row]),
                            "aesthetic": float(aesthetic[row]),
                            "colorfulness": colorfulness(pixels[row]),
                            "contrast": contrast(pixels[row]),
                            "colorfulness_wide": colorfulness(pixels[row][box[1] : box[3]]),
                            "border_bands": bands,
                            "framing": framing(bands),
                        }
                    )
                    + "\n"
                )
            print(
                args.config,
                len(group),
                "rows",
                round(time.monotonic() - tick, 2),
                "s; total",
                round(time.monotonic() - started),
                flush=True,
            )
    print("complete", args.config, flush=True)


def records():
    found = {}
    for path in sorted((ROOT / "scores").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            row["framing"] = framing(row["border_bands"])
            found[row["id"]] = row
    excluded = ROOT / "manual-exclusions.json"
    if excluded.exists():
        for identifier, reason in json.loads(excluded.read_text()).items():
            if identifier in found:
                found[identifier].update(framing="excluded", exclusion=reason)
    return list(found.values())


def rank(rows):
    """ObedientKite's ranking plus a small colourfulness term; the eye makes the picks."""
    for row in rows:
        row["rank_score"] = row["aesthetic"] + 0.07 * row["clip_score"] + 0.01 * row["colorfulness"]
    return sorted(rows, key=lambda row: row["rank_score"], reverse=True)


def font(size):
    for path in ("/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def contact_sheet(rows, path, columns=6, title="", size=224, wide=False):
    big, small = font(13), font(11)
    height = round(size / 1.5) if wide else size
    cell_h, margin, heading = height + 76, 12, 40
    sheet = Image.new(
        "RGB",
        (columns * (size + margin) + margin, math.ceil(len(rows) / columns) * cell_h + heading),
        (18, 22, 27),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, 12), title, font=big, fill="white")
    for slot, row in enumerate(rows):
        x, y = margin + (slot % columns) * (size + margin), heading + (slot // columns) * cell_h
        image = Image.open(row["file"]).convert("RGB")
        if wide:
            image = image.crop(display_box(row["border_bands"]))
        sheet.paste(image.resize((size, height), Image.Resampling.LANCZOS), (x, y))
        label = row.get("label") or (
            f"{row['name']}.{row['variant']} s{row['seed']}  C{row['clip_score']:.1f} "
            f"A{row['aesthetic']:.2f} M{row['colorfulness']:.0f}"
        )
        draw.text((x, y + height + 4), label, font=big, fill=(242, 211, 132))
        draw.text(
            (x, y + height + 21),
            row.get("sublabel") or f"{row['config']}  [{row['framing']}]",
            font=small,
            fill=(165, 207, 225),
        )
        for number, line in enumerate(textwrap.wrap(row.get("prompt", ""), width=size // 6)[:3]):
            draw.text((x, y + height + 36 + number * 12), line, font=small, fill=(220, 226, 232))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    print(path, sheet.size, flush=True)


def summary(args):
    rows = records()
    table = {}
    for key in ("config", "variant"):
        for value in sorted({row[key] for row in rows}):
            subset = [row for row in rows if row[key] == value]
            usable = [row for row in subset if row["framing"] in ("square", "wide")]
            table[f"{key}={value}"] = {
                "generated": len(subset),
                "square": sum(row["framing"] == "square" for row in subset),
                "wide": sum(row["framing"] == "wide" for row in subset),
                **{
                    f"mean_{metric}": round(float(np.mean([row[metric] for row in usable])), 3)
                    for metric in ("aesthetic", "clip_score", "colorfulness", "contrast")
                },
                "mean_rank_top25pct": round(
                    float(
                        np.mean(
                            sorted((row["rank_score"] for row in rank(usable)), reverse=True)[
                                : max(1, len(usable) // 4)
                            ]
                        )
                    ),
                    3,
                ),
            }
    save_json(ROOT / "summary.json", table)
    for key, value in table.items():
        print(key, value)


def sheets(args):
    rows = rank([row for row in records() if row["framing"] in ("square", "wide")])
    rows = [row for row in rows if not args.config or row["config"] in args.config]
    save_json(ROOT / "ranked.json", rows)
    seen, best = {}, []
    for row in rows:
        key = (
            (row["name"], row["seed"]) if args.distinct_seeds else (row["name"], row["seed"], row["variant"])
        )
        if key in seen:
            continue
        seen[key] = True
        best.append(row)
    per_prompt = {}
    for row in best:
        per_prompt.setdefault(row["name"], []).append(row)
    names = args.prompts or list(PROMPTS)
    top = [row for name in names for row in per_prompt.get(name, [])[: args.per_prompt]]
    tag = args.tag or "all"
    for start in range(0, len(top), args.per_sheet):
        contact_sheet(
            top[start : start + args.per_sheet],
            ROOT / "sheets" / f"{tag}-{start // args.per_sheet:02d}.png",
            columns=args.per_prompt * 2,
            wide=args.wide,
            title=f"{tag}: top {args.per_prompt} per prompt by rank (A + 0.07 C + 0.01 M); "
            "band-free (square) or bands outside the 3:2 crop (wide)",
        )


def verify(args):
    """Batched rows against the single call they stand for."""
    from dew.artifacts import uint8_pixels

    pipe, _ = load_pipe()
    rows = [row for row in records() if row["config"] in CONFIGS]
    picks = [rows[index] for index in np.random.default_rng(0).choice(len(rows), args.count, replace=False)]
    report = []
    for row in picks:
        pixels = uint8_pixels(single(pipe, row["prompt"], row["seed"], CONFIGS[row["config"]]).host().images)[
            0
        ]
        stored = np.asarray(Image.open(row["file"]).convert("RGB"))
        gap = np.abs(pixels.astype(np.int16) - stored.astype(np.int16))
        report.append({"id": row["id"], "max_abs": int(gap.max()), "mean_abs": float(gap.mean())})
        print(report[-1], flush=True)
    save_json(ROOT / "verify.json", report)


def trajectory(pipe, prompt, seed, config):
    """Every step's clean prediction on the walk `single` takes: `sample`'s scan,
    with the predictions kept, and the closing prediction last."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    solver, guidance, negative = call_settings(config)
    steps = config["steps"]
    prepared = pipe.prepare([prompt], key=seed, steps=steps, unconditional=negative)
    process, grid = pipe.prepared_process(steps)
    if grid is not None or process.interval:
        raise ValueError("this walk covers a process on its own schedule only")
    key = jax.random.fold_in(jax.random.key(seed), 1)

    @jax.jit
    def run(params, given, null, x_T):
        variables = {name: value for name, value in params.items() if name not in ("encoders", "autoencoder")}
        walk = guidance.walk(process.denoiser(pipe.model, variables, given, null))
        with jax.ensure_compile_time_eval():
            times = process.times(steps)
            initial = solver.init(x_T, times, process, key=key)
        batch = x_T.shape[0]

        def body(carry, inputs):
            x, state, guided = carry
            t, t_next, index = inputs
            t, t_next = jnp.full((batch,), t), jnp.full((batch,), t_next)
            (denoised, eps), guided = walk.step(x, t, guided)
            x, state = solver.step(
                x, t, t_next, denoised, eps, state, jax.random.fold_in(key, index), process, walk.at(guided)
            )
            return (x, state, guided), denoised

        (x, _, guided), x0s = lax.scan(
            body, (x_T, initial, walk.init(x_T)), (times[:-1], times[1:], jnp.arange(times.shape[0] - 1))
        )
        final = walk.at(guided)(x, jnp.full((batch,), times[-1]))[0]
        return jnp.concatenate([x0s, final[None]])

    walked = run(pipe.params, prepared.conditions, prepared.unconditional, prepared.noise)
    reference = single(pipe, prompt, seed, config, decode=False).latents
    gap = float(jnp.max(jnp.abs(reference.astype(jnp.float32) - walked[-1].astype(jnp.float32))))
    return walked[:, 0], gap


def finals(args):
    """Re-run every pick as its single call; heroes also keep each step's prediction."""
    import jax
    import jax.numpy as jnp

    from dew.artifacts import uint8_pixels

    selection = json.loads(Path(args.selection).read_text())
    pipe, directory = load_pipe()
    record = provenance(pipe, directory)
    out = ROOT / "finals"
    save_json(out / "provenance.json", record)
    scorer = Scorer()
    decode = jax.jit(lambda params, z: jnp.clip(pipe.autoencoder.decode(params, z), -1.0, 1.0))
    picks = []
    for pick in selection["picks"]:
        name, variant, seed, config_name = pick["name"], pick["variant"], pick["seed"], pick["config"]
        config, prompt = CONFIGS[config_name], full_prompt(name, variant)
        tick = time.monotonic()
        pixels = uint8_pixels(single(pipe, prompt, seed, config).host().images)[0]
        seconds = round(time.monotonic() - tick, 2)
        stem = f"{name}.{variant}.s{seed:02d}.{config_name}"
        target = out / "png" / f"{stem}.png"
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(target)
        clip, aesthetic = scorer(pixels[None], [PROMPTS[name]])
        bands = border_bands(pixels)
        row = {
            "id": f"{config_name}/{name}.{variant}.s{seed:02d}",
            "name": name,
            "variant": variant,
            "seed": seed,
            "config": config_name,
            "prompt": prompt,
            "settings": {
                **config,
                "negative": None if config["negative"] is None else NEGATIVES[config["negative"]],
            },
            "file": str(target),
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "clip_score": float(clip[0]),
            "aesthetic": float(aesthetic[0]),
            "colorfulness": colorfulness(pixels),
            "border_bands": bands,
            "framing": framing(bands),
            "display_box_3x2": list(display_box(bands)),
            "roles": pick.get("roles", []),
            "seconds_on_device": seconds,
        }
        if "hero" in pick.get("roles", []):
            walked, gap = trajectory(pipe, prompt, seed, config)
            if gap > 1e-4:
                raise ValueError(f"{stem}: the kept walk ends {gap} from the pipeline's latent")
            replay = out / "replays" / pick["hero"]
            replay.mkdir(parents=True, exist_ok=True)
            frames = []
            for start in range(0, walked.shape[0], 8):
                frames.extend(uint8_pixels(decode(pipe.params["autoencoder"], walked[start : start + 8])))
            for step, frame in enumerate(frames):
                Image.fromarray(frame).save(replay / f"x0_{step:03d}.png")
            pixel_gap = int(np.abs(frames[-1].astype(np.int16) - pixels.astype(np.int16)).max())
            save_json(
                replay / "trajectory.json",
                {
                    "prompt": prompt,
                    "seed": seed,
                    "sampler": config["solver"],
                    "steps": config["steps"],
                    "cfg": config["cfg"],
                    "guidance_interval": config["interval"],
                    "negative_prompt": row["settings"]["negative"],
                    "batch_prompts": None,
                    "batch_row": None,
                    "call": (
                        "pipe.prepare([prompt], key=seed, steps=steps, unconditional=negative_prompt), "
                        "then pipe(prepared, key=seed, steps=steps, solver=sampler, "
                        "guidance=CFG(cfg, interval=...))"
                    ),
                    "walk_vs_pipe_max_abs": gap,
                    "final_vs_pick_pixel_max_abs": pixel_gap,
                },
            )
            row.update(hero=pick["hero"], walk_vs_pipe_max_abs=gap, final_vs_pick_pixel_max_abs=pixel_gap)
        picks.append(row)
        print(
            stem,
            row["framing"],
            f"C{row['clip_score']:.1f} A{row['aesthetic']:.2f}",
            seconds,
            "s",
            flush=True,
        )
    save_json(out / "picks.json", {"provenance": record, "picks": picks})


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("config", choices=CONFIGS)
    generate_parser.add_argument("--seeds", nargs="+", type=int, default=list(range(4)))
    generate_parser.add_argument("--prompts", nargs="+", choices=PROMPTS)
    generate_parser.add_argument("--variants", nargs="+", choices=SUFFIX, default=list(SUFFIX))
    generate_parser.add_argument("--batch-size", type=int, default=16)
    generate_parser.add_argument("--deadline", type=int, default=1620)
    sheets_parser = commands.add_parser("sheets")
    sheets_parser.add_argument("--per-prompt", type=int, default=3)
    sheets_parser.add_argument("--per-sheet", type=int, default=60)
    sheets_parser.add_argument("--config", nargs="+")
    sheets_parser.add_argument("--tag")
    sheets_parser.add_argument("--wide", action="store_true")
    sheets_parser.add_argument("--distinct-seeds", action="store_true")
    sheets_parser.add_argument("--prompts", nargs="+", choices=PROMPTS)
    commands.add_parser("summary")
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--count", type=int, default=6)
    finals_parser = commands.add_parser("finals")
    finals_parser.add_argument("selection")
    args = parser.parse_args()
    {"generate": generate, "sheets": sheets, "summary": summary, "verify": verify, "finals": finals}[
        args.command
    ](args)


if __name__ == "__main__":
    main()
