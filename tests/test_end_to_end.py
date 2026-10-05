"""REPA-E: the autoencoder's regularizer and latent batch norm against the
official code (`tools/repae_reference.py`), one end-to-end step against
`train_repae.py`'s, and the gradient split of a step: the diffusion loss
trains the model and not the autoencoder, and the autoencoder's own loss
trains it and not the model."""

import tarfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference, assert_computes_the_oracle
from test_mean_flow import CLASSES, labelled

from dew.diffusion import presets
from dew.inputs import Condition, Field, InputSpec, unit_range
from dew.nn.autoencoders.kl import AutoencoderKL
from dew.nn.autoencoders.sd_vae import StableDiffusionVAE
from dew.nn.autoencoders.vae import translate_vae_weights
from dew.nn.backbones import SimpleDiT
from dew.objectives.base import Step
from dew.objectives.diffusion import Alignment, DiffusionObjective
from dew.objectives.diffusion.alignment import ALIGNMENT
from dew.objectives.diffusion.end_to_end import AUTOENCODER, LATENT_STATS, EndToEnd
from dew.sampling import Euler, TextToImage
from dew.training import Trainer

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CASE = np.load(FIXTURES / "repae" / "regularizer.npz")
STEPPED = np.load(FIXTURES / "repae" / "step.npz")


def test_the_regularizer_is_repa_es():
    total, terms = EndToEnd().regularizer(jnp.asarray(CASE["images"], jnp.float32),
                                          jnp.asarray(CASE["reconstruction"], jnp.float32),
                                          jnp.asarray(CASE["moments"], jnp.float32))
    # float32 sums of a few hundred O(1) terms: 1e-5 relative is rounding.
    np.testing.assert_allclose(float(terms["kl"]), float(CASE["kl"]), rtol=1e-5)
    np.testing.assert_allclose(float(total), float(CASE["regularizer"]), rtol=1e-5)


def test_the_latent_batch_norm_is_repa_es():
    """Started from `init_bn`'s statistics, one training batch normalizes by
    its own and moves the running ones; evaluation then reads those."""
    end_to_end = EndToEnd()
    latents = jnp.asarray(CASE["latents"], jnp.float32)
    trained, statistics = end_to_end.batch_normalized(latents, end_to_end.initial_statistics(0.1, 0.8, 4))
    np.testing.assert_allclose(np.asarray(trained), CASE["trained"], rtol=0, atol=2e-6)
    np.testing.assert_allclose(np.asarray(statistics["mean"]), CASE["running_mean"], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(statistics["var"]), CASE["running_var"], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(end_to_end.normalized(latents, statistics)), CASE["evaluated"],
                               rtol=0, atol=2e-6)


class Patches(nn.Module):
    @nn.compact
    def __call__(self, pixels):
        return nn.Conv(5, (4, 4), strides=(4, 4))(pixels)


def objective(end_to_end: EndToEnd) -> DiffusionObjective:
    module = AutoencoderKL(channels=(8, 16), latent_channels=4, blocks_per_level=1, norm_groups=4)
    autoencoder = StableDiffusionVAE(
        model=module, params=module.init(jax.random.PRNGKey(5), jnp.zeros((1, 8, 8, 3)))["params"],
        dtype=jnp.float32, latent_shift=0.1, latent_scale=0.8)
    encoder = Patches()
    alignment = Alignment(encoder, encoder.init(jax.random.PRNGKey(9), jnp.zeros((1, 8, 8, 3))),
                          "dit_block_0", width=8)
    model = SimpleDiT(patch_size=2, emb_features=16, num_layers=2, num_heads=2, mlp_ratio=1,
                             output_channels=4)
    return DiffusionObjective(model, presets.Flow()(), InputSpec(Field("image", (8, 8, 3))), guidance=None,
                              solver=Euler(), steps=2, autoencoder=autoencoder, alignment=alignment,
                              end_to_end=end_to_end, ema_decay=None)


# Eight rows, which the test mesh's eight devices divide.
BATCH = {"image": np.asarray(jax.random.randint(jax.random.PRNGKey(1), (8, 8, 8, 3), 0, 256), np.uint8)}
STEP = Step(step=jnp.asarray(0), key=jax.random.PRNGKey(2), ema=None)


def gradients(end_to_end: EndToEnd):
    task = objective(end_to_end)
    params = task.init(jax.random.PRNGKey(0))
    return params, jax.grad(lambda tree: task.scalar_loss({**params, "params": tree}, BATCH, STEP)[0])(
        params["params"])


def total(tree) -> float:
    return float(sum(jnp.abs(leaf).sum() for leaf in jax.tree.leaves(tree)))


def test_each_loss_trains_its_own_side():
    """With the autoencoder's own loss weighted to zero, nothing reaches it:
    the denoising and alignment losses read a detached latent. With it on,
    the model's and projector's gradients are unchanged: the autoencoder's
    alignment reads them frozen."""
    params, joint = gradients(EndToEnd())
    _, model_only = gradients(EndToEnd(align_weight=0.0, reconstruction_weight=0.0, kl_weight=0.0))
    assert total(model_only[AUTOENCODER]) == 0
    assert total(joint[AUTOENCODER]) > 0
    for name in joint:
        if name != AUTOENCODER:
            for got, want in zip(
                jax.tree.leaves(joint[name]), jax.tree.leaves(model_only[name]), strict=True
            ):
                np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    _, aligned_only = gradients(EndToEnd(reconstruction_weight=0.0, kl_weight=0.0))
    assert total(aligned_only[AUTOENCODER]) > 0
    assert set(params) >= {LATENT_STATS} and "autoencoder" not in params


def test_a_step_moves_the_running_statistics_and_the_task_decodes_with_them():
    task = objective(EndToEnd())
    trainer = Trainer(task, optax.adam(1e-3), key=jax.random.PRNGKey(3))
    state = trainer.initial_state()
    before = jax.tree.map(np.asarray, state.variables[LATENT_STATS])
    state, *_ = trainer.compile(state, BATCH)(state, BATCH)
    after = state.variables[LATENT_STATS]
    assert not np.allclose(np.asarray(after["mean"]), np.asarray(before["mean"]))

    published = TextToImage.from_objective(task, state.variables)
    assert AUTOENCODER not in published.variables["params"] and LATENT_STATS not in published.variables
    for got, want in zip(jax.tree.leaves(published.variables["autoencoder"]),
                         jax.tree.leaves(state.variables["params"][AUTOENCODER]), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    np.testing.assert_allclose(np.asarray(published.autoencoder.latent_scale),
                               1 / np.sqrt(np.asarray(after["var"])), rtol=1e-6)


SIDE, LATENT, PATCH, WIDTH, FEATURES, PROJECTOR, DROPOUT = 32, 4, 2, 12, 6, 10, 0.1


class Block(nn.Module):
    """The tool's `Block`, tokens plus tanh of their and the condition's
    linear maps, or without `residual` its `Final`, the maps' sum."""

    out: int = WIDTH
    residual: bool = True

    @nn.compact
    def __call__(self, x, c):
        mixed = nn.Dense(self.out, name="x")(x) + nn.Dense(self.out, name="c")(c)[:, None]
        return x + jnp.tanh(mixed) if self.residual else mixed


class StandIn(nn.Module):
    """The tool's SiT around its stand-ins: `Patches`, the position table,
    `Times` on the flow time (Dew passes it times 1000) plus the class's
    `LabelEmbedder` row, two blocks, `Final` and SiT's `unpatchify`. The
    class is the condition's second token, whose table entry is the class,
    the blank prompt's padding reading the null class."""

    @nn.compact
    def __call__(self, x, time, textcontext, train=False):
        tokens = nn.Conv(WIDTH, (PATCH, PATCH), strides=(PATCH, PATCH), padding="VALID", name="x_embedder")(x)
        n, rows, columns, _ = tokens.shape
        tokens = tokens.reshape(n, rows * columns, WIDTH) + STEPPED["weights/model.pos_embed"]
        label = textcontext.hidden[:, 1, 0].astype(jnp.int32)
        c = (nn.Dense(WIDTH, name="t_embedder")(time[:, None] / 1000)
             + nn.Embed(CLASSES + 1, WIDTH, name="y_embedder")(label))
        for index in range(2):
            tokens = Block(name=f"block_{index}")(tokens, c)
        patches = Block(PATCH * PATCH * LATENT, residual=False, name="final_layer")(tokens, c)
        patches = patches.reshape(n, rows, columns, PATCH, PATCH, LATENT).transpose(0, 1, 3, 2, 4, 5)
        return patches.reshape(n, rows * PATCH, columns * PATCH, LATENT)


class Representation(nn.Module):
    """The tool's representation encoder: 8-pixel patches."""

    @nn.compact
    def __call__(self, pixels):
        return nn.Conv(FEATURES, (8, 8), strides=(8, 8), padding="VALID")(pixels)


LAYOUT = (("model.x_embedder.proj", ("x_embedder",)), ("model.t_embedder.linear", ("t_embedder",)),
          ("model.y_embedder.embedding_table", ("y_embedder",)),
          *((f"model.blocks.{index}.{side}", (f"block_{index}", side)) for index in (0, 1) for side in "xc"),
          *((f"model.final_layer.{side}", ("final_layer", side)) for side in "xc"),
          *((f"model.projectors.0.{2 * index}", (ALIGNMENT, f"Dense_{index}")) for index in range(3)))
"""Each of the reference's modules and where Dew's tree holds it."""


def linen(weight: np.ndarray) -> np.ndarray:
    """A torch weight in linen's layout: a convolution's [out, in, kh, kw]
    as [kh, kw, in, out], a linear layer's [out, in] as [in, out], an
    embedding table as it is."""
    return weight.transpose(2, 3, 1, 0) if weight.ndim == 4 else weight.T


def torch_layout(kernel: np.ndarray) -> np.ndarray:
    return kernel.transpose(3, 2, 0, 1) if kernel.ndim == 4 else kernel.T


def module(name: str) -> dict:
    """The reference module `name`'s weights in linen's names and layout."""
    if name.endswith("embedding_table"):
        return {"embedding": STEPPED[f"weights/{name}.weight"]}
    return {"kernel": linen(STEPPED[f"weights/{name}.weight"]), "bias": STEPPED[f"weights/{name}.bias"]}


def nested(entries) -> dict:
    """A tree of `(path, value)` pairs."""
    tree: dict = {}
    for path, value in entries:
        node = tree
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return tree


def autoencoder_gradients(tail: str) -> dict:
    """The reference's autoencoder gradient in Dew's tree.
    `translate_vae_weights` moves each diffusers tensor's entries (and reads
    them as float32), so it moves their indices into one flat table, which
    the float64 gradients are then read from."""
    names = sorted(key.removeprefix("grad/vae.") for key in STEPPED.files
                   if key.startswith("grad/vae.") and not key.endswith("_f64"))
    flat = np.concatenate([STEPPED[f"grad/vae.{name}{tail}"].ravel() for name in names])
    starts = np.cumsum([0] + [STEPPED[f"grad/vae.{name}"].size for name in names])
    assert starts[-1] < 2 ** 24
    indices = {name: np.arange(start, end, dtype=np.float32).reshape(STEPPED[f"grad/vae.{name}"].shape)
               for name, start, end in zip(names, starts[:-1], starts[1:], strict=True)}
    return jax.tree.map(lambda index: flat[index.astype(np.int64)], translate_vae_weights(indices))


def gamma(roundings: int) -> float:
    """Higham's gamma_k for float32: k roundings of unit roundoff 2^-24 bound
    a relative error by k u / (1 - k u)."""
    unit = float(np.finfo(np.float32).eps) / 2
    return roundings * unit / (1 - roundings * unit)


def batch_norm(latents, before: dict, momentum: float = 0.1) -> dict:
    """torch `BatchNorm2d`'s training-mode update of the running statistics
    `before` by `latents` `[..., C]`, in float64: the batch's mean and
    unbiased variance over every axis but the channels, mixed in at
    `momentum`."""
    rows = np.asarray(latents, np.float64).reshape(-1, np.shape(latents)[-1])
    count = rows.shape[0]
    return {"mean": (1 - momentum) * np.asarray(before["mean"], np.float64) + momentum * rows.mean(0),
            "var": (1 - momentum) * np.asarray(before["var"], np.float64)
            + momentum * rows.var(0) * count / (count - 1)}


def assert_the_batch_norm_of(latents, before: dict, after: dict, momentum: float = 0.1) -> None:
    """Dew's float32 running statistics `after` against `batch_norm` of the
    same float32 `latents` in float64, within the float32 rounding that
    computation can make. Summing n terms in any order errs by at most
    gamma_{n-1} times the sum of their magnitudes, so the batch mean, the
    sum and one division, is within gamma_n of the mean magnitude S. The
    centered variance's terms (x - mean)^2 take three roundings each before
    the n-term sum and the division, gamma_{n+3} of the variance, plus the
    square of the mean's own error, which shifts the centering. Mixing
    adds the stored statistics' float32 conversion, each momentum constant's
    float32 rounding, the multiplies, the variance's n / (n - 1) (two more
    roundings) and the sum. A dropped or extra term (a biased variance, a
    momentum off its value) moves a statistic by a share of 1 / n or more,
    hundreds of times these bounds; the rounding a runner's vector width
    reorders stays inside them."""
    rows = np.asarray(latents, np.float64).reshape(-1, np.shape(latents)[-1])
    count = rows.shape[0]
    exact = batch_norm(latents, before, momentum)
    magnitude = np.abs(rows).mean(0)
    drift = gamma(count) * magnitude
    stored = {name: np.abs(np.asarray(before[name], np.float64)) for name in ("mean", "var")}
    bounds = {"mean": gamma(count + 4) * ((1 - momentum) * stored["mean"] + momentum * magnitude),
              "var": gamma(count + 9) * ((1 - momentum) * stored["var"]
                                         + momentum * count / (count - 1) * (rows.var(0) + drift ** 2))
              + momentum * count / (count - 1) * drift ** 2}
    for name, bound in bounds.items():
        error = np.abs(np.asarray(after[name], np.float64) - exact[name])
        assert np.all(error <= bound), (name, error.tolist(), bound.tolist())


def test_a_step_is_train_repae_s_step(tmp_path):
    """One DiffusionObjective step under `EndToEnd` against
    `train_repae.py`'s loop body run as published on DiffusionObjective's
    own draws (`tools/repae_reference.py`): its two passes, the VAE's
    update through the frozen SiT in evaluation mode (the batch norm on its
    running statistics, no label dropped) and then the SiT's on the
    detached latent in training mode (the batch norm on the batch, the
    running statistics moved, a label dropped), on the same posterior
    sample, times and noise, with the tiny SD VAE and a stand-in SiT.

    This is Dew's current scope, the L1 reconstruction and the KL: the
    reference runs l1_lpips_kl_gan.yaml with `perceptual_weight` and
    `discriminator_weight` at 0, where the published recipe has LPIPS at 1
    and the PatchGAN at 0.1 from step 0.

    The autoencoder's gradient is the VAE update's, and the model's and
    the projector's are half the SiT update's, as Dew's L2 halves the
    denoising error and REPA's term with it; each network's is held to the
    reference by the float64 rule, the running statistics too, and the loss
    and its terms within 1e-6 of the reference's float64 run. The tuned
    autoencoder a run then decodes with takes the latent scale and bias of
    REPA-E's `extract_latents_stats`, by the float64 rule too."""
    with tarfile.open(FIXTURES / "tiny_diffusers.tar.xz") as archive:
        archive.extractall(tmp_path, filter="data")
    alignment = Alignment(Representation(), {"params": {"Conv_0": module("representation")}}, "block_0",
                          width=PROJECTOR)
    inputs = InputSpec(Field("image", (SIDE, SIDE, 3)), {"textcontext": Condition(labelled())})
    task = DiffusionObjective(
        StandIn(), presets.Flow(density="uniform")(), inputs, guidance=None, solver=Euler(), steps=2,
        autoencoder=StableDiffusionVAE(modelname=str(tmp_path / "sd" / "vae"), dtype=jnp.float32),
        alignment=alignment, end_to_end=EndToEnd(), ema_decay=None, unconditional_prob=DROPOUT)
    variables = task.init(jax.random.PRNGKey(0))
    params = {**nested((path, module(name)) for name, path in LAYOUT),
              AUTOENCODER: variables["params"][AUTOENCODER]}
    assert jax.tree.structure(params) == jax.tree.structure(variables["params"])
    variables = {**variables, LATENT_STATS: {"mean": STEPPED["bn/running_mean_before"],
                                             "var": STEPPED["bn/running_var_before"]}}
    batch = {"image": STEPPED["pixels"], **inputs.tokenize([str(label) for label in STEPPED["classes"]])}
    step = Step(step=jnp.asarray(0), key=jax.random.key(int(STEPPED["key"])), ema=None)
    (value, aux), gradients = jax.value_and_grad(
        lambda tree: task.scalar_loss({**variables, "params": tree}, batch, step), has_aux=True)(params)

    def flat(tree):
        return np.concatenate([np.ravel(leaf) for leaf in jax.tree.leaves(tree)])

    assert jax.tree.structure(gradients[AUTOENCODER]) == jax.tree.structure(autoencoder_gradients(""))
    assert_as_exact_as_the_reference(flat(gradients[AUTOENCODER]), flat(autoencoder_gradients("")),
                                     flat(autoencoder_gradients("_f64")), "the autoencoder's gradient")
    for label, entries in (("the model's gradient", LAYOUT[:-3]), ("the projector's gradient", LAYOUT[-3:])):
        dew, reference, truth = [], [], []
        for name, path in entries:
            held = gradients
            for key in path:
                held = held[key]
            for leaf, gradient in held.items():
                part = "bias" if leaf == "bias" else "weight"
                gradient = np.asarray(gradient)
                dew.append(2 * (torch_layout(gradient) if leaf == "kernel" else gradient))
                reference.append(STEPPED[f"grad/{name}.{part}"])
                truth.append(STEPPED[f"grad/{name}.{part}_f64"])
        got, want, exact = (np.concatenate([np.ravel(part) for part in parts])
                            for parts in (dew, reference, truth))
        assert_as_exact_as_the_reference(got, want, exact, label)
    # The running statistics are eight numbers, too few for the float64
    # rule's RMS to settle: a runner's vector width alone moved its ratio
    # from 1.00 to 2.46. So the rule holds the posterior's sample they are
    # computed from, 2048 entries, and they are held to the reference's
    # batch norm of Dew's own sample within that computation's float32
    # rounding, the reference's batch norm being that function in float64.
    statistics = aux.variables[LATENT_STATS]
    tuned = task._end_to_end_latents({**variables, "params": params}, unit_range(batch["image"]),
                                     jax.random.split(step.key, 5)[0])
    for name in ("mean", "var"):
        np.testing.assert_array_equal(np.asarray(tuned.statistics[name]), np.asarray(statistics[name]))
    assert_as_exact_as_the_reference(np.asarray(tuned.raw), STEPPED["latents/sample"],
                                     STEPPED["latents/sample_f64"], "the posterior's sample")
    before = {name: STEPPED[f"bn/running_{name}_before"] for name in ("mean", "var")}
    published = batch_norm(STEPPED["latents/sample_f64"], {name: STEPPED[f"bn/running_{name}_before_f64"]
                                                           for name in ("mean", "var")})
    assert_computes_the_oracle(
        np.concatenate([STEPPED["bn/running_mean_f64"], STEPPED["bn/running_var_f64"]]),
        np.concatenate([published["mean"], published["var"]]), "the reference's batch norm",
        roundings=2 * 512)
    assert_the_batch_norm_of(tuned.raw, before, statistics)
    np.testing.assert_allclose(float(value), 0.5 * STEPPED["loss/sit_f64"] + STEPPED["loss/vae_f64"],
                               rtol=1e-6)
    for term in ("alignment", "autoencoder_alignment", "reconstruction", "kl"):
        np.testing.assert_allclose(float(aux.metrics[term]), STEPPED[f"loss/{term}_f64"], rtol=1e-6,
                                   err_msg=term)
    # REPA-E's `extract_latents_stats` is the running mean and the running
    # variance's reciprocal square root, with no epsilon; the tuned
    # autoencoder takes Dew's own statistics through the same, within the
    # square root's and the division's roundings.
    np.testing.assert_array_equal(STEPPED["latents/latents_bias_f64"], STEPPED["bn/running_mean_f64"])
    np.testing.assert_allclose(STEPPED["latents/latents_scale_f64"],
                               1 / np.sqrt(STEPPED["bn/running_var_f64"]), rtol=4 * np.finfo(np.float64).eps)
    decoder, _ = task.published_autoencoder({**variables, "params": params, LATENT_STATS: statistics})
    assert decoder is not None
    np.testing.assert_array_equal(np.asarray(decoder.latent_shift), np.asarray(statistics["mean"]))
    np.testing.assert_allclose(np.asarray(decoder.latent_scale, np.float64),
                               1 / np.sqrt(np.asarray(statistics["var"], np.float64)), rtol=gamma(2), atol=0)


def test_end_to_end_needs_alignment_and_a_kl_autoencoder():
    task = objective(EndToEnd())
    with pytest.raises(ValueError, match="needs `alignment`"):
        DiffusionObjective(task.model, task.process, task.inputs, autoencoder=task.autoencoder,
                           end_to_end=EndToEnd())


def test_a_run_config_tunes_its_autoencoder_and_from_run_decodes_with_the_tuned_one(tmp_path):
    """REPA-E through `DiffusionRunConfig` on the committed tiny DINOv2 and
    SD VAE: a saved run's task restores the tuned autoencoder and its
    running statistics, and samples exactly as the trained objective's own
    task does."""
    from test_diffusion_run_sources import batch_for

    from dew.checkpoints import Checkpoints
    from dew.config import ModelConfig, TrainerConfig
    from dew.data import TFDSImages
    from dew.objectives.diffusion import DiffusionRunConfig, PretrainedAutoencoder, TextCondition
    from dew.objectives.diffusion.config import RepresentationAlignment

    for name in ("tiny_diffusers", "rae"):
        with tarfile.open(FIXTURES / f"{name}.tar.xz") as archive:
            archive.extractall(tmp_path / name, filter="data")
    config = DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 1, "emb_features": 16, "num_layers": 2, "num_heads": 2,
                                         "mlp_ratio": 1}, dtype="float32", attention_impl="xla"),
        data=TFDSImages(image_size=32), preset=presets.Flow(), solver=Euler(), guidance=None,
        sampling_steps=2, ema_decay=None, val_metrics=(), trainer=TrainerConfig(checkpoint_dir=str(tmp_path)),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        autoencoder=PretrainedAutoencoder(modelname=str(tmp_path / "tiny_diffusers/sd/vae"), dtype="float32"),
        alignment=RepresentationAlignment(encoder=str(tmp_path / "rae/dinov2_plain"), layer="dit_block_0",
                                          width=8, resolution=112, end_to_end=EndToEnd()))
    task = config.build()
    trainer = Trainer(task, optax.adam(1e-2), key=jax.random.PRNGKey(3))
    state = trainer.initial_state()
    batch = batch_for(task, 32)
    state, *_ = trainer.compile(state, batch)(state, batch)
    run = tmp_path / "run"
    checkpoints = Checkpoints(str(run))
    checkpoints.save(1, state, None, artifact=task.inference_record())
    checkpoints.wait()
    config.save(str(run))

    # The saved tree carries the encoder's weights, so restoring reads its
    # checkpoint's config and nothing else.
    for weights in (tmp_path / "rae/dinov2_plain").glob("*.safetensors"):
        weights.unlink()
    restored = TextToImage.from_run(str(run))
    for got, want in zip(jax.tree.leaves(restored.variables["autoencoder"]),
                         jax.tree.leaves(state.variables["params"][AUTOENCODER]), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
    expected = task.pipeline(state, ema=False)(["a red bird"], key=9).host().images
    np.testing.assert_array_equal(restored(["a red bird"], key=9).host().images, expected)
