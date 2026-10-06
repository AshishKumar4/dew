"""REPA-E by its official code, for tests/fixtures/repae.

regularizer.npz, the autoencoder regularizer and the latent batch norm:

- The KL: End2End-Diffusion/REPA-E's `DiagonalGaussianDistribution`
  (models/autoencoder.py, read at a pinned commit, the class extracted and
  run as published), summed over each latent and averaged over the batch
  as `ReconstructionLoss_Single_Stage` does, beside the L1 reconstruction,
  at configs/l1_lpips_kl_gan.yaml's weights.
- The latent normalization: `models/sit.py`'s `BatchNorm2d(eps=1e-4,
  momentum=0.1, affine=False)`, started from `init_bn`'s statistics, in
  training mode and then in evaluation mode.

step.npz, one training step of `train_repae.py` as published: the
statements of its loop body from `vae.train()` to the SiT update, run on
REPA-E's own `AutoencoderKL.forward` and `DiagonalGaussianDistribution`,
`SiT.forward`, `interpolant`, `unpatchify`, `init_bn`, `LabelEmbedder`
and `build_mlp`, `ReconstructionLoss_Single_Stage` and the script's
`requires_grad`, `update_ema` and `preprocess_imgs_vae`, with the loss's
`PerceptualLoss("lpips")` (loss/perceptual_loss.py) over REPA-E's `LPIPS`
(loss/lpips.py) and its `NLayerDiscriminator` and `weights_init`
(loss/discriminator.py), all as published. The autoencoder's encoder,
decoder and 1x1 convolutions are tests/fixtures/tiny_diffusers' SD VAE,
run by diffusers; LPIPS's VGG16 and linear heads are
`lpips_reference.drawn_weights()`, which the test draws too; the
discriminator is built at `ndf=8` (the published 64 at an eighth of the
width) and the fixture carries its weights; the SiT's embedders, blocks
and final layer are small stand-ins (`Patches`, `Times`, `Block`, `Final`)
whose weights the fixture carries, and the representation features are a
stand-in encoder's over the ImageNet-normalized pixels (REPA's
`preprocess_raw_image` is held in tests/test_alignment.py; at 32 pixels
it would resize to none).

The loss config is configs/l1_lpips_kl_gan.yaml as published: LPIPS at
weight 1.0 and the PatchGAN at 0.1 from step 0, so the step's
discriminator update runs too. The draws are `DiffusionObjective.loss`'s from
`jax.random.key(KEY)`, which the step's `torch.randn`, `rand` and
`randn_like` replay in its order: the posterior's sample, the training
times, the noise and the label-dropout uniforms. The gradients are each
network's as the step's `clip_grad_norm_` reads them, before clipping, in
float64 and in float32. After the step, `SiT.extract_latents_stats` gives
the latent scale and bias the script's sampling denormalizes with; the
posterior's sample `z`, which the batch norm reads, is kept too.

    PYTHONPATH=src python tools/repae_reference.py
"""

from __future__ import annotations

import ast
import contextlib
import copy
import functools
import json
import math
import tarfile
import types
import urllib.request
from collections import OrderedDict
from pathlib import Path

import jax
import lpips_reference
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from einops import rearrange
from safetensors.torch import load as load_safetensors
from torchvision import models

from dew.diffusion import presets

REPOSITORY = "https://raw.githubusercontent.com/End2End-Diffusion/REPA-E/2ad4e9f69234c109497fb41d3e5e555de7b4b0de/"
FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "repae"
KL_WEIGHT, SHIFT, SCALE = 1e-6, 0.1, 0.8

KEY = 11
BATCH, SIDE, LATENT, PATCH, WIDTH, FEATURES, PROJECTOR, CLASSES = 8, 32, 4, 2, 12, 6, 10, 10
DOWNSCALE, DROPOUT = 4, 0.1
"""The step's sizes: eight 32-pixel RGB images, the tiny VAE's four latent
channels at a quarter of the side, two-pixel patches into 16 tokens of
width 12, which the stand-in encoder's 8-pixel patches match; the
projector's width, ten classes and `--cfg-prob`'s label dropout."""
TOKENS = (SIDE // DOWNSCALE // PATCH) ** 2

LOSSES = {"discriminator_start": 0, "discriminator_factor": 1.0, "discriminator_weight": 0.1,
          "quantizer_weight": 1.0, "perceptual_loss": "lpips", "perceptual_weight": 1.0,
          "reconstruction_loss": "l1", "reconstruction_weight": 1.0, "lecam_regularization_weight": 0.0,
          "kl_weight": KL_WEIGHT, "logvar_init": 0.0}
"""configs/l1_lpips_kl_gan.yaml."""
DISCRIMINATOR_WIDTH = 8

ARGS = types.SimpleNamespace(path_type="linear", prediction="v", weighting="uniform", proj_coeff=0.5,
                             vae_align_proj_coeff=1.5, bn_momentum=0.1, max_grad_norm=1.0, compile=False)
"""The README's launch flags the step reads."""


def source(path: str) -> str:
    return urllib.request.urlopen(REPOSITORY + path).read().decode()


def definitions(text: str, names: set[str], scope: dict) -> dict:
    """Run the module's top-level classes and functions named `names` as
    published, in `scope`."""
    for node in ast.parse(text).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            exec(compile(ast.Module(body=[node], type_ignores=[]), "<repa-e>", "exec"), scope)
    missing = names - set(scope)
    assert not missing, missing
    return scope


def regularizer() -> None:
    scope = definitions(source("models/autoencoder.py"), {"DiagonalGaussianDistribution"},
                        {"torch": torch, "np": np})
    generator = torch.Generator().manual_seed(0)
    moments = torch.randn(3, 8, 4, 4, generator=generator, dtype=torch.float64) * 2
    images = torch.rand(3, 3, 16, 16, generator=generator, dtype=torch.float64) * 2 - 1
    reconstruction = images + 0.1 * torch.randn(3, 3, 16, 16, generator=generator, dtype=torch.float64)
    kl = scope["DiagonalGaussianDistribution"](moments).kl()
    total = F.l1_loss(images, reconstruction) + KL_WEIGHT * torch.sum(kl) / kl.shape[0]

    latents = torch.randn(3, 4, 4, 4, generator=generator, dtype=torch.float64) * 1.7 + 0.4
    norm = torch.nn.BatchNorm2d(4, eps=1e-4, momentum=0.1, affine=False).double()
    norm.running_mean = torch.full((4,), SHIFT, dtype=torch.float64)
    norm.running_var = torch.full((4,), 1 / SCALE, dtype=torch.float64).pow(2)
    trained = norm.train()(latents)
    evaluated = norm.eval()(latents)
    channels_last = (0, 2, 3, 1)
    FIXTURE.mkdir(parents=True, exist_ok=True)
    np.savez(FIXTURE / "regularizer.npz",
             moments=moments.permute(*channels_last).numpy(), images=images.permute(*channels_last).numpy(),
             reconstruction=reconstruction.permute(*channels_last).numpy(), regularizer=total.numpy(),
             kl=(torch.sum(kl) / kl.shape[0]).numpy(), latents=latents.permute(*channels_last).numpy(),
             trained=trained.permute(*channels_last).numpy(),
             evaluated=evaluated.permute(*channels_last).numpy(),
             running_mean=norm.running_mean.numpy(), running_var=norm.running_var.numpy())
    print(f"{FIXTURE}: the regularizer and one training and one evaluation batch norm")


class Patches(torch.nn.Module):
    """SiT's `x_embedder`: a patch convolution into raster-order tokens."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Conv2d(LATENT, WIDTH, PATCH, stride=PATCH)
        self.patch_size = (PATCH, PATCH)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Times(torch.nn.Module):
    """SiT's `t_embedder`: the time through one linear layer."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, WIDTH)

    def forward(self, t):
        return self.linear(t[:, None])


class Block(torch.nn.Module):
    """A block conditioned on `c`: tokens plus tanh of their and the
    condition's linear maps."""

    def __init__(self, out: int = WIDTH):
        super().__init__()
        self.x, self.c = torch.nn.Linear(WIDTH, out), torch.nn.Linear(WIDTH, out)

    def forward(self, x, c):
        return x + torch.tanh(self.x(x) + self.c(c)[:, None])


class Final(Block):
    """SiT's `final_layer`: the tokens' and the condition's linear maps into
    a patch of every latent channel."""

    def __init__(self):
        super().__init__(PATCH * PATCH * LATENT)

    def forward(self, x, c):
        return self.x(x) + self.c(c)[:, None]


class Config(dict):
    """OmegaConf's reading of the loss config, attributes and `get`, and
    the `dictdot` the autoencoder's `decode` returns."""

    __getattr__ = dict.__getitem__


class Replayed:
    """torch as the step reads it: `randn`, `rand` and `randn_like` return
    Dew's draws in the step's order, each checked against the shape asked
    for, and `tensor` builds in the run's precision."""

    def __init__(self, dtype, draws: list[np.ndarray]):
        self.dtype, self.draws = dtype, [torch.as_tensor(np.array(draw)) for draw in draws]

    def __getattr__(self, name):
        return getattr(torch, name)

    def _next(self, shape):
        draw = self.draws.pop(0)
        assert tuple(draw.shape) == tuple(shape), (tuple(draw.shape), tuple(shape))
        return draw.to(self.dtype)

    def randn(self, *size, device=None):
        return self._next(size[0] if len(size) == 1 and not isinstance(size[0], int) else size)

    def rand(self, *size, device=None):
        return self.randn(*size)

    def randn_like(self, like):
        return self._next(like.shape)

    def tensor(self, data, device=None, dtype=None):
        return torch.tensor(data, dtype=dtype or self.dtype)


class Accelerator:
    """accelerate's `Accelerator` on one process: `backward` is the loss's,
    and `clip_grad_norm_` records each parameter's gradient as it reads
    it."""

    sync_gradients = True

    def __init__(self):
        self.gradients: dict[torch.Tensor, torch.Tensor] = {}

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, max_norm):
        for parameter in parameters:
            if parameter.grad is not None:
                self.gradients[parameter] = parameter.grad.detach().clone()
        return torch.zeros(())

    def unwrap_model(self, model):
        return model


@contextlib.contextmanager
def precision(dtype):
    """`Tensor.float()`, which `preprocess_imgs_vae` and the loss's forward
    call, as the run's precision."""
    published = torch.Tensor.float
    torch.Tensor.float = lambda self, *args, **kwargs: self.to(dtype)
    try:
        yield
    finally:
        torch.Tensor.float = published


def tiny_vae() -> AutoencoderKL:
    """tests/fixtures/tiny_diffusers' SD VAE, read from the archive."""
    with tarfile.open(FIXTURE.parent / "tiny_diffusers.tar.xz") as archive:
        config, weights = (archive.extractfile(f"sd/vae/{name}")
                           for name in ("config.json", "diffusion_pytorch_model.safetensors"))
        assert config is not None and weights is not None
        vae = AutoencoderKL.from_config(json.load(config))
        assert isinstance(vae, AutoencoderKL)
        vae.load_state_dict(load_safetensors(weights.read()))
    return vae


def published_discriminator() -> torch.nn.Module:
    """REPA-E's `NLayerDiscriminator(input_nc=3, n_layers=3)` at
    `DISCRIMINATOR_WIDTH`, drawn by its `weights_init`."""
    scope = definitions(source("loss/discriminator.py"), {"ActNorm", "weights_init", "NLayerDiscriminator"},
                        {"torch": torch, "nn": torch.nn, "functools": functools})
    return scope["NLayerDiscriminator"](input_nc=3, ndf=DISCRIMINATOR_WIDTH, n_layers=3).apply(
        scope["weights_init"])


def perceptual(vgg: dict[str, np.ndarray], linear: dict[str, np.ndarray]) -> type:
    """REPA-E's `PerceptualLoss` over its `LPIPS`, whose weights are these."""
    scope = definitions(source("loss/perceptual_loss.py"), {"PerceptualLoss"},
                        {"torch": torch, "models": models,
                         "LPIPS": lpips_reference.lpips_class(
                             {key: torch.as_tensor(value) for key, value in vgg.items()},
                             {key: torch.as_tensor(value) for key, value in linear.items()})})
    text = source("loss/perceptual_loss.py")
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(getattr(target, "id", "").startswith("_IMAGENET")
                                                  for target in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), "perceptual_loss.py", "exec"), scope)
    return scope["PerceptualLoss"]


def weights() -> dict[str, np.ndarray]:
    """Every stand-in's and published module's weights, drawn once."""
    torch.manual_seed(0)
    modules = {"model.x_embedder": Patches(), "model.t_embedder": Times(),
               "model.y_embedder.embedding_table": torch.nn.Embedding(CLASSES + 1, WIDTH),
               "model.blocks.0": Block(), "model.blocks.1": Block(), "model.final_layer": Final(),
               "model.projectors.0": torch.nn.Sequential(
                   torch.nn.Linear(WIDTH, PROJECTOR), torch.nn.SiLU(), torch.nn.Linear(PROJECTOR, PROJECTOR),
                   torch.nn.SiLU(), torch.nn.Linear(PROJECTOR, FEATURES)),
               "representation": torch.nn.Conv2d(3, FEATURES, 8, stride=8),
               "discriminator": published_discriminator()}
    drawn = {f"{prefix}.{name}": value.detach().numpy().copy()
             for prefix, module in modules.items() for name, value in module.state_dict().items()}
    drawn["model.pos_embed"] = torch.randn(1, TOKENS, WIDTH).numpy()
    drawn["bn.latents_bias"] = (torch.randn(LATENT) * 0.3).numpy()
    drawn["bn.latents_scale"] = (0.6 + torch.rand(LATENT)).numpy()
    return drawn


def draws(pixels: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    """DiffusionObjective.loss's draws from `jax.random.key(KEY)`, channels
    first, in the step's order, and the rows its label dropout blanks."""
    encode, drop, time, noise, _ = jax.random.split(jax.random.key(KEY), 5)
    latents = (pixels.shape[0], SIDE // DOWNSCALE, SIDE // DOWNSCALE, LATENT)
    uniforms = np.asarray(jax.random.uniform(drop, (pixels.shape[0],)))
    dropped = np.asarray(jax.random.bernoulli(drop, DROPOUT, (pixels.shape[0],)))
    assert np.array_equal(dropped, uniforms < DROPOUT) and 0 < dropped.sum() < len(dropped)
    t = np.asarray(presets.Flow(density="uniform")().schedule.sample_t(time, pixels.shape[0]))
    first = (0, 3, 1, 2)
    return [np.asarray(jax.random.normal(encode, latents)).transpose(first), t.reshape(-1, 1, 1, 1),
            np.asarray(jax.random.normal(noise, latents)).transpose(first), uniforms], dropped


def run(dtype, drawn: dict[str, np.ndarray], pixels: np.ndarray, classes: np.ndarray,
        replayed: list[np.ndarray]) -> dict[str, np.ndarray]:
    """One published step at `dtype`: each network's gradients, the batch
    norm's running statistics after it and the step's losses."""
    random = Replayed(dtype, replayed)
    sit = definitions(source("models/sit.py"), {"mean_flat", "build_mlp", "LabelEmbedder", "SiT"},
                      {"torch": random, "nn": torch.nn, "np": np, "math": math})
    autoencoder = definitions(
        source("models/autoencoder.py"), {"DiagonalGaussianDistribution", "AutoencoderKL"},
        {"torch": random, "nn": torch.nn, "np": np, "dictdot": Config})
    losses = definitions(
        source("loss/losses.py"),
        {"hinge_d_loss", "compute_lecam_loss", "ReconstructionLoss_Stage2",
         "ReconstructionLoss_Single_Stage"},
        {"torch": torch, "nn": torch.nn, "F": F, "rearrange": rearrange,
         "autocast": lambda enabled=True: (lambda method: method),
         "PerceptualLoss": perceptual(*lpips_reference.drawn_weights()),
         "NLayerDiscriminator": functools.partial(
             definitions(source("loss/discriminator.py"), {"ActNorm", "NLayerDiscriminator"},
                         {"torch": torch, "nn": torch.nn, "functools": functools})["NLayerDiscriminator"],
             ndf=DISCRIMINATOR_WIDTH),
         "weights_init": definitions(source("loss/discriminator.py"), {"weights_init"},
                                     {"torch": torch, "nn": torch.nn})["weights_init"]})
    train = source("train_repae.py")
    script = definitions(train, {"requires_grad", "update_ema"}, {"torch": torch, "OrderedDict": OrderedDict})
    definitions(source("utils.py"), {"preprocess_imgs_vae"}, script)

    vae = autoencoder["AutoencoderKL"].__new__(autoencoder["AutoencoderKL"])
    torch.nn.Module.__init__(vae)
    tiny = tiny_vae()
    vae.encoder, vae.decoder, vae.use_variational = tiny.encoder, tiny.decoder, True
    vae.quant_conv, vae.post_quant_conv = tiny.quant_conv, tiny.post_quant_conv

    model = sit["SiT"].__new__(sit["SiT"])
    torch.nn.Module.__init__(model)
    model.out_channels, model.encoder_depth = LATENT, 1
    model.x_embedder, model.t_embedder = Patches(), Times()
    model.y_embedder = sit["LabelEmbedder"](CLASSES, WIDTH, DROPOUT)
    model.register_buffer("pos_embed", torch.zeros(1, TOKENS, WIDTH))
    model.blocks = torch.nn.ModuleList([Block(), Block()])
    model.projectors = torch.nn.ModuleList([sit["build_mlp"](WIDTH, PROJECTOR, FEATURES)])
    model.final_layer = Final()
    model.bn = torch.nn.BatchNorm2d(LATENT, eps=1e-4, momentum=ARGS.bn_momentum, affine=False,
                                    track_running_stats=True)
    model.bn.reset_running_stats()
    vae_loss_fn = losses["ReconstructionLoss_Single_Stage"](
        Config(losses=Config(LOSSES), model=Config(vq_model=Config(quantize_mode="vae"))))
    for module in (vae, model, vae_loss_fn):
        module.to(dtype)
    for prefix, module in (("model.", model), ("discriminator.", vae_loss_fn.discriminator)):
        missing, _ = module.load_state_dict({name: torch.as_tensor(drawn[prefix + name]).to(dtype)
                                             for name in module.state_dict() if prefix + name in drawn},
                                            strict=False)
        assert all(name.startswith("bn.") for name in missing), missing
    model.init_bn(latents_scale=torch.as_tensor(drawn["bn.latents_scale"]).to(dtype),
                  latents_bias=torch.as_tensor(drawn["bn.latents_bias"]).to(dtype))
    before = {name: getattr(model.bn, name).clone() for name in ("running_mean", "running_var")}

    representation = torch.nn.Conv2d(3, FEATURES, 8, stride=8).to(dtype)
    representation.load_state_dict({name: torch.as_tensor(drawn["representation." + name]).to(dtype)
                                     for name in ("weight", "bias")})
    mean, std = (torch.tensor(value, dtype=dtype).view(1, 3, 1, 1)
                 for value in ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
    raw_image = torch.as_tensor(pixels).permute(0, 3, 1, 2)
    with torch.no_grad():
        zs = [representation((raw_image.to(dtype) / 255 - mean) / std).flatten(2).transpose(1, 2)]

    accelerator = Accelerator()
    scope = {**script, "raw_image": raw_image, "labels": torch.as_tensor(classes), "zs": zs, "vae": vae,
             "model": model, "vae_loss_fn": vae_loss_fn, "args": ARGS, "accelerator": accelerator,
             "global_step": 0, "ema": copy.deepcopy(model),
             **{name: torch.optim.SGD(module.parameters(), lr=0.0)
                for name, module in (("optimizer_vae", vae), ("optimizer_loss_fn", vae_loss_fn),
                                     ("optimizer", model))}}
    loop = next(node for node in ast.walk(ast.parse(train)) if isinstance(node, ast.For)
                and isinstance(node.target, ast.Tuple) and ast.unparse(node.target) == "(raw_image, y)")
    start = next(index for index, node in enumerate(loop.body) if ast.unparse(node) == "vae.train()")
    accumulate = next(node for node in loop.body if isinstance(node, ast.With)
                      and ast.unparse(node.items[0]).startswith("accelerator.accumulate"))
    statements = [*loop.body[start:loop.body.index(accumulate)], *accumulate.body]
    with precision(dtype):
        exec(compile(ast.Module(body=statements, type_ignores=[]), "train_repae.py", "exec"), scope)
    assert not random.draws, "the step drew fewer times than DiffusionObjective.loss"

    tail = "_f64" if dtype == torch.float64 else ""
    result = {f"grad/vae.{name}{tail}": accelerator.gradients[value].numpy()
              for name, value in vae.named_parameters()}
    result.update({f"grad/model.{name}{tail}": accelerator.gradients[value].numpy()
                   for name, value in model.named_parameters()})
    result.update({f"grad/discriminator.{name}{tail}": accelerator.gradients[value].numpy()
                   for name, value in vae_loss_fn.discriminator.named_parameters()})
    for name, value in before.items():
        result[f"bn/{name}_before{tail}"] = value.numpy()
        result[f"bn/{name}{tail}"] = getattr(model.bn, name).numpy()
    result.update({f"latents/{name}{tail}": value.numpy()
                   for name, value in model.extract_latents_stats().items()})
    result[f"latents/sample{tail}"] = scope["z"].detach().permute(0, 2, 3, 1).numpy()
    terms = scope["vae_loss_dict"]
    result.update({
        f"loss/vae{tail}": scope["vae_loss"].detach().numpy(),
        f"loss/sit{tail}": scope["sit_loss"].detach().numpy(),
        f"loss/reconstruction{tail}": terms["reconstruction_loss"].numpy(),
        f"loss/kl{tail}": (terms["kl_loss"] / KL_WEIGHT).numpy(),
        f"loss/perceptual{tail}": terms["perceptual_loss"].numpy(),
        f"loss/generator{tail}": terms["gan_loss"].numpy(),
        f"loss/discriminator{tail}": scope["d_loss"].detach().numpy(),
        f"loss/autoencoder_alignment{tail}": scope["vae_align_outputs"]["proj_loss"].detach().numpy(),
        f"loss/denoising{tail}": scope["sit_outputs"]["denoising_loss"].mean().detach().numpy(),
        f"loss/alignment{tail}": scope["sit_outputs"]["proj_loss"].detach().numpy()})
    return result


def step() -> None:
    drawn = weights()
    generator = np.random.default_rng(0)
    pixels = generator.integers(0, 256, (BATCH, SIDE, SIDE, 3), dtype=np.uint8)
    classes = generator.integers(0, CLASSES, BATCH)
    replayed, dropped = draws(pixels)
    arrays = {f"weights/{name}": value for name, value in drawn.items()}
    arrays.update(pixels=pixels, classes=classes, key=np.asarray(KEY), dropped=dropped)
    for dtype in (torch.float64, torch.float32):
        arrays.update(run(dtype, drawn, pixels, classes, replayed))
    np.savez(FIXTURE / "step.npz", **arrays)
    print(f"{FIXTURE / 'step.npz'}: one step, {int(dropped.sum())} of {BATCH} labels dropped, "
          f"{sum(1 for name in arrays if name.startswith('grad/') and name.endswith('_f64'))} gradients")


def main() -> None:
    regularizer()
    step()


if __name__ == "__main__":
    main()
