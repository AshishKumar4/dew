"""REPA and iREPA against their official code (`tools/repa_reference.py`),
and the alignment inside the diffusion objective."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference, assert_as_exact_over_orders, distance

from dew.diffusion import presets
from dew.inputs import Field, InputSpec
from dew.nn.backbones import SimpleDiT
from dew.objectives.base import Step
from dew.objectives.diffusion import Alignment, DiffusionObjective
from dew.objectives.diffusion.alignment import ALIGNMENT, REPRESENTATION, spatial_zscore
from dew.sampling import Euler, TextToImage

CASES = np.load(Path(__file__).resolve().parent / "fixtures" / "repa" / "losses.npz")
SETTINGS = json.loads(str(CASES["settings"]))


def projector(kind: str) -> dict:
    prefix = f"{kind}."
    tree: dict = {}
    for key in CASES.files:
        if key.startswith(prefix):
            module, leaf = key[len(prefix):].split(".")
            tree.setdefault(module, {})[leaf] = jnp.asarray(CASES[key], jnp.float32)
    return {"params": tree}


# A float32 mean of 32 cosines, each a few roundings of an O(1) value: its
# error is a small multiple of 2^-24, so 1e-6 separates rounding from any
# difference in what is computed (a wrong normalization or projection moves
# the loss by more than 1e-3 here).
LOSS_ATOL = 1e-6


def test_the_repa_loss_is_the_official_one():
    """REPA's MLP projector and its -cos loop over tokens and examples."""
    alignment = Alignment("block", width=SETTINGS["width"])
    loss = alignment.loss(projector("mlp"), jnp.asarray(CASES["hidden"]), jnp.asarray(CASES["features"]), {})
    np.testing.assert_allclose(float(loss), float(CASES["repa"]), rtol=0, atol=LOSS_ATOL)


def test_the_irepa_loss_is_the_official_one():
    """iREPA's 3x3 convolution projector against spatially z-scored targets."""
    alignment = Alignment("block", projector="conv", spatial_norm=SETTINGS["gamma"])
    targets = spatial_zscore(jnp.asarray(CASES["features"]), SETTINGS["gamma"])
    loss = alignment.loss(projector("conv"), jnp.asarray(CASES["hidden"]), targets, {})
    np.testing.assert_allclose(float(loss), float(CASES["irepa"]), rtol=0, atol=LOSS_ATOL)


class Patches(nn.Module):
    """A frozen encoder of 4-pixel patches, one token per DiT patch."""

    @nn.compact
    def __call__(self, pixels):
        return nn.Conv(5, (4, 4), strides=(4, 4))(pixels)


def aligned(kind: str = "mlp"):
    model = SimpleDiT(patch_size=4, emb_features=16, num_layers=2, num_heads=2, mlp_ratio=1)
    encoder = Patches()
    variables = encoder.init(jax.random.PRNGKey(9), jnp.zeros((1, 8, 8, 3)))
    alignment = Alignment("dit_block_0", encoder=encoder, weight=0.5, projector=kind, width=8,
                          spatial_norm=0.6 if kind == "conv" else None, resolution=None)
    return DiffusionObjective(model, presets.Flow()(), InputSpec(Field("image", (8, 8, 3))),
                              guidance=None, solver=Euler(), steps=2, alignment=alignment,
                              variables={REPRESENTATION: variables})


@pytest.mark.parametrize("kind", ["mlp", "conv"])
def test_the_objective_adds_the_weighted_alignment_to_the_denoising_mean(kind):
    """The loss is the denoising mean plus `weight` times the alignment, whose
    gradient reaches the projector and the layers up to the aligned one; the
    encoder is held beside the model, and a published task drops it and the
    projector."""
    objective = aligned(kind)
    params = objective.init(jax.random.PRNGKey(0))
    batch = {"image": np.asarray(jax.random.randint(jax.random.PRNGKey(1), (4, 8, 8, 3), 0, 256), np.uint8)}
    step = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(2), ema=None)
    loss, aux = objective.scalar_loss(params, batch, step)

    plain = DiffusionObjective(objective.model, objective.process, objective.inputs, guidance=None,
                               solver=Euler(), steps=2)
    denoising, _ = plain.scalar_loss({**objective.model_variables(params), "encoders": params["encoders"]},
                               batch, step)
    # REPA's total is mse + proj_coeff * alignment; Dew's L2 halves the
    # first, so the second is halved with it: 0.5 / 2.
    assert float(loss) == pytest.approx(float(denoising) + 0.25 * float(aux.metrics["alignment"]), rel=1e-6)
    assert -1.0 <= float(aux.metrics["alignment"]) <= 1.0

    def alignment_only(tree):
        return objective.scalar_loss({**params, "params": tree}, batch, step)[1].metrics["alignment"]

    grads = jax.grad(alignment_only)(params["params"])
    assert float(jnp.abs(jax.tree.leaves(grads[ALIGNMENT])[0]).sum()) > 0
    assert float(sum(jnp.abs(leaf).sum() for leaf in jax.tree.leaves(grads["dit_block_0"]))) > 0
    assert float(sum(jnp.abs(leaf).sum() for leaf in jax.tree.leaves(grads["dit_block_1"]))) == 0
    published = TextToImage.from_objective(objective, params).variables
    assert REPRESENTATION not in published and ALIGNMENT not in published["params"]


COMPOSED = np.load(Path(__file__).resolve().parent / "fixtures" / "repa" / "composed.npz")
PREPROCESSED = np.load(Path(__file__).resolve().parent / "fixtures" / "repa" / "preprocessed.npz")


def patches(images, patch: int):
    """`[B, H, W, C]` as `[B, N, p * p * C]` tokens in raster order."""
    b, h, w, c = images.shape
    grid = images.reshape(b, h // patch, patch, w // patch, patch, c).transpose(0, 1, 3, 2, 4, 5)
    return grid.reshape(b, (h // patch) * (w // patch), patch * patch * c)


class Block(nn.Module):
    @nn.compact
    def __call__(self, hidden):
        return jnp.tanh(hidden @ self.param("kernel", nn.initializers.zeros, (hidden.shape[-1],) * 2))


class Network(nn.Module):
    """`tools/repa_reference.py`'s stand-in model at Dew's model time (t times
    1000): patch tokens plus the time, the aligned `block`, a head."""

    patch: int
    width: int

    @nn.compact
    def __call__(self, x, time, train=False):
        b, side, _, c = x.shape
        tokens = patches(x, self.patch)
        hidden = (tokens @ self.param("embed", nn.initializers.zeros, (tokens.shape[-1], self.width))
                  + (time / 1000).reshape(-1, 1, 1)
                  * self.param("time", nn.initializers.zeros, (self.width,)))
        block = Block(name="block")(hidden)
        out = block @ self.param("head", nn.initializers.zeros, (self.width, tokens.shape[-1]))
        grid = out.reshape(b, side // self.patch, side // self.patch, self.patch, self.patch, c)
        return grid.transpose(0, 1, 3, 2, 4, 5).reshape(b, side, side, c)


class Projected(nn.Module):
    """The stand-in encoder: a projection of the preprocessed pixels' patches."""

    patch: int

    @nn.compact
    def __call__(self, pixels):
        tokens = patches(pixels, self.patch)
        return tokens @ self.param("encoder", nn.initializers.zeros, (tokens.shape[-1], 6))


def test_repa_trains_on_repas_composed_loss_and_its_gradient():
    """REPA's whole loss as its train.py takes it, `loss_mean +
    proj_loss_mean * proj_coeff` over `SILoss` (linear path, v prediction,
    uniform times) with `build_mlp`'s projector and the encoder's input
    from `preprocess_raw_image`, run as published on a stand-in network and
    encoder (`tools/repa_reference.py`), on the times and noise
    `DiffusionObjective.loss` draws: Dew's loss is half of it, as Dew's L2
    halves the denoising error, and its gradient in every weight, the
    model's up to and past the aligned block and the projector's, is half
    of REPA's, held to its float64 run by the K-order rule: over ORDERS
    orders of the stand-in's hidden units, an exact symmetry, against REPA's
    fp32 runs over the same orders. One run against one is a coin flip on
    the time weight's gradient, a sum over every token: Dew's runs spread
    from 0.65 to 1.87 times REPA's identity run, and with XLA capped to AVX
    the identity run reached 2.23."""
    from dew.diffusion import FlowMatchingScheduler, FlowMatchPredictionTransform, Process

    settings = json.loads(str(COMPOSED["settings"]))
    patch, width = settings["patch"], settings["width"]
    encoder = Projected(patch)
    alignment = Alignment("block", encoder=encoder, weight=settings["proj_coeff"],
                          width=settings["projector"], resolution=None)
    process = Process(FlowMatchingScheduler(density="uniform"), FlowMatchPredictionTransform())
    pixels = COMPOSED["pixels"]
    objective = DiffusionObjective(Network(patch, width), process,
                                   InputSpec(Field("image", pixels.shape[1:])), guidance=None, solver=Euler(),
                                   steps=2, alignment=alignment, unconditional_prob=0.0, ema_decay=None,
                                   variables={REPRESENTATION: {"params": {
                                       "encoder": jnp.asarray(COMPOSED["encoder"])}}})
    variables = objective.init(jax.random.PRNGKey(0))
    projector = {name: {leaf: jnp.asarray(COMPOSED[f"projector/{name}/{leaf}"], jnp.float32)
                        for leaf in ("kernel", "bias")} for name in ("Dense_0", "Dense_1", "Dense_2")}
    params = {"embed": COMPOSED["weights/embed"], "time": COMPOSED["weights/time"],
              "head": COMPOSED["weights/head"], "block": {"kernel": COMPOSED["weights/block"]},
              ALIGNMENT: projector}
    params = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float32), params)
    step = Step(step=jnp.asarray(0), key=jax.random.key(settings["key"]), ema=None)

    def loss(params):
        return objective.scalar_loss({**variables, "params": params}, {"image": pixels}, step)[0]

    value = loss(params)
    np.testing.assert_allclose(2 * float(value), float(COMPOSED["loss_f64"]), rtol=2e-6)
    gradient = jax.jit(jax.grad(loss))
    distances = {}
    projector = params[ALIGNMENT]
    for order in COMPOSED["orders"].astype(np.intp):
        back = np.argsort(order)
        first = {**projector["Dense_0"], "kernel": projector["Dense_0"]["kernel"][order]}
        moved = {**params, "embed": params["embed"][:, order], "time": params["time"][order],
                 "head": params["head"][order],
                 "block": {"kernel": params["block"]["kernel"][np.ix_(order, order)]},
                 ALIGNMENT: {**projector, "Dense_0": first}}
        got = jax.tree.map(lambda leaf: 2 * np.asarray(leaf), gradient(moved))
        found = {"grad/embed": got["embed"][:, back], "grad/time": got["time"][back],
                 "grad/head": got["head"][back], "grad/block": got["block"]["kernel"][np.ix_(back, back)]}
        for name in ("Dense_0", "Dense_1", "Dense_2"):
            kernel = got[ALIGNMENT][name]["kernel"]
            found[f"grad/projector/{name}/kernel"] = kernel[back] if name == "Dense_0" else kernel
            found[f"grad/projector/{name}/bias"] = got[ALIGNMENT][name]["bias"]
        for key, leaf in found.items():
            distances.setdefault(key, []).append(distance(leaf, COMPOSED[f"{key}_f64"]))
    recorded = {name.removeprefix("orders/") for name in COMPOSED.files if name.startswith("orders/")}
    assert set(distances) == recorded
    for key, mine in distances.items():
        assert_as_exact_over_orders(mine, COMPOSED[f"orders/{key}"], key)


class Unchanged(nn.Module):
    """An encoder whose features are the pixels it is handed."""

    @nn.compact
    def __call__(self, pixels):
        return pixels


@pytest.mark.parametrize("spatial_norm", [None, 0.6], ids=["repa", "irepa"])
def test_the_encoders_input_is_repas_and_irepas_dinov2_preprocessing(spatial_norm):
    """256-pixel images to an encoder of 224: REPA's `preprocess_raw_image`
    ("dinov2") and iREPA's `DINOv2Encoder.preprocess`, which agree (the
    reference checks), /255, ImageNet's normalization and torch's bicubic
    resize, and under iREPA its `spatial_zscore` (gamma 0.6) of the
    features, held to the published float64 run by the float64 rule."""
    from dew.inputs import unit_range

    alignment = Alignment("block", encoder=Unchanged(), resolution=224, spatial_norm=spatial_norm)
    targets = alignment.targets({}, unit_range(PREPROCESSED["pixels"]))
    name = "dinov2" if spatial_norm is None else "zscore"
    assert_as_exact_as_the_reference(targets, PREPROCESSED[name], PREPROCESSED[f"{name}_f64"], name)


def test_a_layer_the_model_lacks_is_refused():
    objective = aligned()
    alignment = Alignment("dit_block_9", encoder=objective.alignment.encoder, resolution=None)
    with pytest.raises(ValueError, match="no submodule 'dit_block_9'"):
        DiffusionObjective(objective.model, objective.process, objective.inputs, guidance=None,
                           solver=Euler(), steps=2, alignment=alignment,
                           variables={REPRESENTATION: objective.representation}).init(jax.random.PRNGKey(0))


def test_from_run_publishes_the_model_without_the_alignment_head(tmp_path):
    """An inference record restores the denoiser, not the frozen encoder or
    projector that only its training loss reads, as the objective's own
    pipeline does."""
    import optax

    from dew.checkpoints import Checkpoints
    from dew.training import Trainer

    objective = aligned()
    trainer = Trainer(objective, optax.adam(1e-3), key=jax.random.key(0))
    state = trainer.initial_state()
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(0, state, None, artifact=objective.inference_record())
    checkpoints.wait()

    restored = TextToImage.from_run(str(tmp_path), ema=False)
    published = objective.pipeline(state, ema=False)
    assert REPRESENTATION not in restored.variables
    assert ALIGNMENT not in restored.variables["params"]
    assert jax.tree.structure(restored.variables) == jax.tree.structure(published.variables)
    for actual, expected in zip(jax.tree.leaves(restored.variables), jax.tree.leaves(published.variables),
                                strict=True):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    np.testing.assert_array_equal(restored([{}, {}], key=9).host().images,
                                  published([{}, {}], key=9).host().images)


def test_a_published_pipeline_aligns_beside_its_own_tree(tmp_path):
    """A REPA run on the tiny Flux holds the encoder its tree lacks and draws
    the projector; a step moves both sides. REPA-E needs a run's own autoencoder."""
    import dataclasses
    import tarfile

    import optax
    from test_diffusion_run_sources import batch_for, precision

    from dew.config import ObjectiveConfig
    from dew.data import TFDSImages
    from dew.objectives.diffusion import DiffusionRunConfig, EndToEnd
    from dew.training import Trainer

    for name in ("flux_source", "rae"):
        with tarfile.open(Path(__file__).resolve().parent / "fixtures" / f"{name}.tar.xz") as archive:
            archive.extractall(tmp_path / name, filter="data")
    alignment = Alignment("x_embedder", source=str(tmp_path / "rae" / "dinov2_plain"), width=8, resolution=56)
    config = DiffusionRunConfig(pretrained=str(tmp_path / "flux_source" / "pipeline"), preset=None,
                                model=precision(), data=TFDSImages(image_size=16), val_metrics=(),
                                objective=ObjectiveConfig("diffusion", {
                                    "alignment": alignment, "solver": Euler(), "guidance": None, "steps": 2,
                                    "ema_decay": None}))
    objective = config.build()
    trainer = Trainer(objective, optax.sgd(1e-1), key=jax.random.PRNGKey(3))
    state = trainer.initial_state()
    before = jax.tree.map(np.asarray, state.variables["params"])  # the step donates these buffers
    held = state.variables[REPRESENTATION], objective.representation
    assert jax.tree.all(jax.tree.map(lambda got, want: np.array_equal(got, want), *held))
    batch = batch_for(objective, 16)
    state, *_ = trainer.compile(state, batch)(state, batch)
    moved = jax.tree.map(lambda got, want: not np.array_equal(got, want), state.variables["params"], before)
    assert all(any(jax.tree.leaves(moved[name])) for name in (ALIGNMENT, "x_embedder"))
    with pytest.raises(ValueError, match="own `autoencoder`"):
        dataclasses.replace(config, objective=ObjectiveConfig("diffusion", {
            **config.objective.fields, "end_to_end": EndToEnd()}))
