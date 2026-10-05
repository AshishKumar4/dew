#!/usr/bin/env python3
"""Write tests/fixtures/flaxdiff_pipeline/reference.npz: FlaxDiff's own
preview sampling of a tiny text-to-image run, for `TextToImage.from_flaxdiff`.

The reference is AshishKumar4/FlaxDiff at 3e3497e924fe58ade3fbb4e3e67c5a33d5f6623a,
the commit the importer was written for, its source archive checked against
its SHA-256 and imported as written. The run is a `SimpleUDiT` of the
fixture size tools/flaxdiff_reference.py uses, every weight redrawn from
N(0, 0.05^2), over the tiny SD pipeline of tests/fixtures/tiny_diffusers.tar.xz:
its CLIP text tower and its VAE, which Diffusers 0.34.0's FlaxAutoencoderKL
reads under FlaxDiff's `StableDiffusionVAE`. The run config is the record
FlaxDiff's training.py logs, its input config FlaxDiff's own `serialize()`.

The sampling is `GeneralDiffusionTrainer`'s preview, as written:
`EulerAncestralSampler` over `KarrasVENoiseScheduler(1, sigma_max=80, rho=7,
sigma_data=0.5)` with `KarrasPredictionTransform`, guidance 3, the prompts
encoded by FlaxDiff's `CLIPTextEncoder` and its unconditional "", and
`generate_samples(..., diffusion_steps=200, start_step=1000, end_step=0)`,
decoding through the VAE and clipping. Every weight of the run is a draw
from N(0, 0.05^2) rounded to a bfloat16-representable value. transformers 5 has no
FlaxCLIPTextModel, so the encoder's model is transformers' PyTorch
CLIPTextModel over the same weights, at the precision FlaxDiff ran its
Flax one (bfloat16).

The random draws are Dew's: the initial unit noise and each ancestral
step's noise, captured from `TextToImage.from_flaxdiff`'s own call on the
same run and handed to FlaxDiff's sampler in its order. Three runs land:
as FlaxDiff ran (towers in bfloat16), with the towers in float32, and the
truth, everything in float64.

    PYTHONPATH=src python tools/flaxdiff_pipeline_reference.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tarfile
import tempfile
import types
import urllib.request
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

# Dew sets its numerical policy before the JAX backend opens, as in a test.
import dew  # noqa: F401

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "flaxdiff_pipeline" / "reference.npz"
COMMIT = "3e3497e924fe58ade3fbb4e3e67c5a33d5f6623a"
ARCHIVE_SHA256 = "93535e536346b5112751cd581fc6d94ba041fcf46365a81abbeaf572ff3682fd"
SOURCE = Path.home() / ".cache" / "dew" / "upstream" / "AshishKumar4" / "FlaxDiff" / COMMIT
MODEL = {"output_channels": 4, "patch_size": 2, "emb_features": 32, "num_layers": 4,
         "num_heads": 4, "mlp_ratio": 4, "dropout_rate": 0.1, "norm_groups": 0,
         "use_hilbert": False, "use_flash_attention": False,
         "activation": "jax._src.nn.functions.silu", "dtype": "jax.numpy.float32",
         "precision": "DEFAULT"}
PROMPTS = ("a red cat", "two blue birds")
IMAGE, STEPS, KEY, SEED = 32, 200, 3, 0


def fetch() -> Path:
    """The pinned source, fetched once and checked against its SHA-256."""
    if (SOURCE / "flaxdiff" / "__init__.py").is_file():
        return SOURCE
    SOURCE.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/AshishKumar4/FlaxDiff/archive/{COMMIT}.tar.gz"
    with urllib.request.urlopen(url, timeout=300) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != ARCHIVE_SHA256:
        raise SystemExit(f"{url} is not the pinned archive")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as file:
        file.write(archive)
        file.flush()
        with tarfile.open(file.name) as source:
            members = [member for member in source.getmembers() if "/" in member.name]
            for member in members:
                member.name = member.name.split("/", 1)[1]
            source.extractall(SOURCE, members=members, filter="data")
    return SOURCE


def towers(root: Path) -> tuple[Path, Path]:
    """The tiny SD pipeline's CLIP (its text encoder and tokenizer in one
    directory, as a CLIP checkpoint holds them) and VAE, under `root`."""
    with tarfile.open(ROOT / "tests" / "fixtures" / "tiny_diffusers.tar.xz") as archive:
        parts = (["sd", "text_encoder"], ["sd", "tokenizer"], ["sd", "vae"])
        members = [member for member in archive.getmembers() if member.name.split("/")[:2] in parts]
        archive.extractall(root, members=members, filter="data")
    clip = root / "clip"
    clip.mkdir()
    for part in ("text_encoder", "tokenizer"):
        for file in (root / "sd" / part).iterdir():
            file.rename(clip / file.name)
    return clip, root / "sd" / "vae"


def run_config(input_config: dict, clip: Path, vae: Path) -> dict:
    """The record FlaxDiff's training.py logs for the run, the CLIP and VAE
    named by their local directories."""
    conditions = copy.deepcopy(input_config["conditions"])
    for condition in conditions:
        condition["encoder"]["modelname"] = str(clip)
    return {"model": MODEL, "architecture": "simple_udit",
            "input_config": {**input_config, "conditions": conditions},
            "arguments": {"noise_schedule": "edm", "architecture": "simple_udit"},
            "autoencoder": "stable_diffusion", "autoencoder_opts": json.dumps({"modelname": str(vae)})}


def checkpoint(step: Path, weights) -> None:
    """The weights as FlaxDiff's trainer saves a step, the averaged ones
    under `ema_params` and the live ones moved off them."""
    import orbax.checkpoint as ocp

    live = jax.tree.map(lambda leaf: leaf + 0.01, weights)
    state = {"params": {"params": live}, "ema_params": {"params": weights}, "step": np.asarray(1)}
    tree = {"state": state, "best_state": state, "best_loss": np.asarray(0.5)}
    ocp.PyTreeCheckpointer().save((step / "default").resolve(), tree)


class TorchCLIP:
    """transformers' PyTorch CLIPTextModel where FlaxDiff's encoder calls a
    FlaxCLIPTextModel: the same call, its last hidden state as a jax array."""

    def __init__(self, directory: Path, dtype: str):
        import torch
        from transformers import CLIPTextModel

        self.dtype = dtype
        self.model = CLIPTextModel.from_pretrained(directory, dtype=getattr(torch, dtype)).eval()

    def __call__(self, input_ids, attention_mask):
        import torch

        with torch.no_grad():
            ids, mask = (torch.from_numpy(np.asarray(array)).long() for array in (input_ids, attention_mask))
            hidden = self.model(input_ids=ids, attention_mask=mask).last_hidden_state
        values = hidden.float().numpy() if self.dtype == "bfloat16" else hidden.numpy()
        return types.SimpleNamespace(last_hidden_state=jnp.asarray(values, getattr(jnp, self.dtype)))


def draws(step: Path, config: dict) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
    """Dew's own call on the run: its initial state, each ancestral step's
    unit noise in order, and its images."""
    import dew.sampling.solvers as solvers
    from dew.sampling import TextToImage

    task = TextToImage.from_flaxdiff(step, config, jax_version=jax.__version__)
    noise: list[np.ndarray] = []

    def normal(key, shape, dtype=jnp.float32):
        value = jax.random.normal(key, shape, dtype)
        jax.debug.callback(lambda drawn: noise.append(np.asarray(drawn)), value, ordered=True)
        return value

    recorded = types.SimpleNamespace(**{**vars(jax), "random": types.SimpleNamespace(
        **{**vars(jax.random), "normal": normal})})
    original, solvers.jax = solvers.jax, recorded
    try:
        images = np.asarray(task(list(PROMPTS), key=KEY).images)
    finally:
        solvers.jax = original
    start = np.asarray(task.prepare(list(PROMPTS), key=KEY).noise)
    return start, noise, images


def condition(encoder):
    """The text condition as FlaxDiff's training.py configures it."""
    from flaxdiff.inputs import ConditionalInputConfig

    return ConditionalInputConfig(encoder=encoder, conditioning_data_key="text", unconditional_input="",
                                  model_key_override="textcontext")


def flaxdiff_samples(weights, clip: Path, vae: Path, unit, noise, *, towers: str, wide: bool) -> np.ndarray:
    """`generate_samples` as FlaxDiff's trainer previews a run, its draws Dew's."""
    import flaxdiff.samplers.common as common
    import flaxdiff.samplers.euler as euler
    from flaxdiff.inputs import CLIPTextEncoder, DiffusionInputConfig
    from flaxdiff.models.autoencoder.diffusers import StableDiffusionVAE
    from flaxdiff.models.simple_vit import SimpleUDiT
    from flaxdiff.predictors import KarrasPredictionTransform
    from flaxdiff.schedulers import KarrasVENoiseScheduler
    from transformers import AutoTokenizer

    dtype = jnp.float64 if wide else jnp.float32
    model = SimpleUDiT(**{key: value for key, value in MODEL.items()
                          if key not in ("activation", "dtype", "precision")}, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(clip)
    encoder = CLIPTextEncoder(model=TorchCLIP(clip, "float64" if wide else towers), tokenizer=tokenizer,
                              modelname=str(clip), backend="jax")
    input_config = DiffusionInputConfig(sample_data_key="image", sample_data_shape=(IMAGE, IMAGE, 3),
                                        conditions=[condition(encoder)])
    autoencoder = StableDiffusionVAE(modelname=str(vae), dtype=jnp.float64 if wide else getattr(jnp, towers))
    sampler = euler.EulerAncestralSampler(
        model=model, noise_schedule=KarrasVENoiseScheduler(1, sigma_max=80, rho=7, sigma_data=0.5),
        model_output_transform=KarrasPredictionTransform(sigma_data=0.5), input_config=input_config,
        autoencoder=autoencoder, guidance_scale=3.0)

    steps = iter(noise)
    initial = types.SimpleNamespace(**{**vars(jax), "random": types.SimpleNamespace(
        **{**vars(jax.random), "normal": lambda key, shape, *_, **__: jnp.asarray(unit, dtype)})})
    stepped = types.SimpleNamespace(**{**vars(jax), "random": types.SimpleNamespace(
        **{**vars(jax.random), "normal": lambda key, shape, *_, **__: jnp.asarray(next(steps), dtype)})})
    common.jax, euler.jax = initial, stepped
    try:
        params = {"params": jax.tree.map(lambda leaf: jnp.asarray(leaf, dtype), weights)}
        samples = sampler.generate_samples(
            params=params, num_samples=len(PROMPTS), resolution=IMAGE, diffusion_steps=STEPS,
            start_step=1000, end_step=0, priors=None, model_conditioning_inputs=(encoder(list(PROMPTS)),))
    finally:
        common.jax, euler.jax = jax, jax
    assert next(steps, None) is None, "FlaxDiff took fewer ancestral steps than Dew"
    return np.asarray(samples, np.float64 if wide else np.float32)


def main() -> None:
    sys.path.insert(0, str(fetch()))
    # FlaxDiff's `StableDiffusionVAE` imports Diffusers' Flax SD pipeline and
    # never uses it; that module imports FlaxCLIPTextModel, which
    # transformers 5 dropped, so an empty module stands in for it.
    pipeline = types.ModuleType("diffusers.pipelines.stable_diffusion.pipeline_flax_stable_diffusion")
    pipeline.FlaxStableDiffusionPipeline = None
    sys.modules.setdefault(pipeline.__name__, pipeline)
    from flaxdiff.inputs import CLIPTextEncoder, DiffusionInputConfig
    from flaxdiff.models.simple_vit import SimpleUDiT
    from transformers import AutoTokenizer, CLIPTextConfig

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        clip, vae = towers(root)
        text = CLIPTextConfig.from_pretrained(clip)
        tokens = AutoTokenizer.from_pretrained(clip).model_max_length
        model = SimpleUDiT(**{key: value for key, value in MODEL.items()
                              if key not in ("activation", "dtype", "precision")}, dtype=jnp.float32)
        rng = np.random.default_rng(SEED)
        latent = IMAGE // 4
        shapes = jax.eval_shape(model.init, jax.random.key(0), jnp.zeros((1, latent, latent, 4)),
                                jnp.zeros((1,)), jnp.zeros((1, tokens, text.hidden_size)))
        # bfloat16-representable float32 values, so the fixture compresses.
        weights = jax.tree.map(lambda leaf: np.asarray(
            jnp.asarray(0.05 * rng.standard_normal(leaf.shape), jnp.bfloat16), np.float32), shapes["params"])

        tokenizer = AutoTokenizer.from_pretrained(clip)
        encoder = CLIPTextEncoder(model=TorchCLIP(clip, "float32"), tokenizer=tokenizer, modelname=str(clip),
                                  backend="jax")
        serialized = DiffusionInputConfig(sample_data_key="image", sample_data_shape=(IMAGE, IMAGE, 3),
                                          conditions=[condition(encoder)]).serialize()
        step = root / "1"
        checkpoint(step, weights)
        start, noise, ours = draws(step, run_config(serialized, clip, vae))

        from dew.diffusion.presets import EDM
        scale = float(EDM(regime="latent")().sampler_schedule.prior_scale())
        unit = (start / np.float32(scale)).astype(np.float32)
        arrays = {"start": start}
        for name, precision in (("as_run", "bfloat16"), ("fp32_towers", "float32")):
            arrays[f"{name}.images"] = flaxdiff_samples(weights, clip, vae, unit, noise, towers=precision,
                                                        wide=False)
        with jax.enable_x64(new_val=True):
            arrays["fp64.images"] = flaxdiff_samples(weights, clip, vae, unit.astype(np.float64),
                                                     [draw.astype(np.float64) for draw in noise],
                                                     towers="float64", wide=True)

    serialized["conditions"][0]["encoder"]["modelname"] = None
    meta = {"input_config": serialized, "jax_version": jax.__version__, "flaxdiff": COMMIT}
    arrays.update({f"params/{'/'.join(path)}": leaf for path, leaf in
                   ((tuple(str(key.key) for key in path), np.asarray(leaf))
                    for path, leaf in jax.tree_util.tree_flatten_with_path(weights)[0])})
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE, **arrays)
    for name, images in (("FlaxDiff as run", arrays["as_run.images"]),
                         ("FlaxDiff, float32 towers", arrays["fp32_towers.images"]), ("Dew as run", ours)):
        gap = float(np.sqrt(np.mean((np.clip(images, -1, 1) - arrays["fp64.images"]) ** 2)))
        print(f"{name}: {gap:.3g} rms from float64")


if __name__ == "__main__":
    main()
