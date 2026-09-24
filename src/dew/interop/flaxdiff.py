"""FlaxDiff checkpoints as Dew models.

FlaxDiff (github.com/AshishKumar4/FlaxDiff) is the project Dew grew out of.
Its trainer saved an orbax tree at every checkpoint step, `{"state":
{"params", "ema_params", "opt_state", ...}, "best_state": {...}, ...}`, and
logged the run's config to wandb, so the architecture lives in the config and
not in the tree. `load_flaxdiff` reads one step and that config and returns
the run as a `TextToImage` over Dew's own model, built with the fields that
compute what FlaxDiff computed. This module maps names and config and
nothing else.

It knows `simple_udit`, FlaxDiff 0.2's U-shaped DiT (commit 3e3497e, the
code flaxdiff 0.2.8 shipped). Its blocks project the conditioning vector
without a SiLU and it averages the text over every position, so it loads as
`SimpleUDiT(adaln_silu=False, text_pooling="all")`. FlaxDiff drew the Fourier
table of its time embedding with `jax.random.normal` at PRNGKey(42) and never
saved it, and jax 0.5.0 changed that stream, so the loader draws the table
the way the run's jax did. From then on the table is the `constants` variable
every Dew checkpoint of the model carries.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import jax
import jax.numpy as jnp
import numpy as np

from dew import records

if TYPE_CHECKING:
    from dew.sampling.pipelines import TextToImage

# FlaxDiff 0.2's SimpleUDiT held these at its root; Dew's nests them under the
# conditioning embed and the output head.
_MOVED = {"time_embed": ("conditioning", "time_embed"),
          "text_proj": ("conditioning", "text_context_proj"),
          "final_norm": ("output", "final_norm"),
          "final_proj": ("output", "final_proj")}

_READ = frozenset({"output_channels", "patch_size", "emb_features", "num_layers", "num_heads",
                   "mlp_ratio", "norm_epsilon"})
# Model config keys FlaxDiff 0.2's SimpleUDiT computed nothing with on a CPU
# or GPU: its blocks take `dropout_rate` and never apply it, and it reads
# none of the others.
_UNREAD = frozenset({"dropout_rate", "activation", "norm_groups", "use_flash_attention",
                     "dtype", "precision"})
# Variants this port was never checked against FlaxDiff with.
_REFUSED = ("use_hilbert", "learn_sigma")


class SimpleUDiTFields(TypedDict):
    """The `simple_udit` fields a FlaxDiff 0.2 run config sets."""

    output_channels: int
    patch_size: int
    emb_features: int
    num_layers: int
    num_heads: int
    mlp_ratio: int
    norm_epsilon: float
    adaln_silu: bool
    text_pooling: Literal["real", "all"]


def read_checkpoint(directory: str | os.PathLike) -> dict:
    """One FlaxDiff checkpoint step, the directory holding `default/`, as the
    nested dict of host arrays FlaxDiff saved there with orbax.

    FlaxDiff's 2024 runs saved orbax's older aggregate file
    (`default/checkpoint`), which this does not read: no model of those runs
    has a loader.
    """
    import orbax.checkpoint as ocp

    restored = ocp.PyTreeCheckpointer().restore((Path(directory) / "default").resolve())
    if not isinstance(restored, dict) or "state" not in restored:
        raise ValueError(f"{directory} is not a FlaxDiff checkpoint step: it holds no 'state'")
    return restored


def flaxdiff_weights(tree: Mapping, *, ema: bool = True, best: bool = False) -> dict:
    """The model parameters a FlaxDiff checkpoint tree holds: the averaged
    copy (`ema`) or the live one, of the last state or of `best_state`."""
    state = tree["best_state" if best else "state"]
    return dict(state["ema_params" if ema else "params"]["params"])


def fourier_table(features: int, jax_version: str, scale: float = 16) -> np.ndarray:
    """The frequencies FlaxDiff's FourierEmbedding drew under `jax_version`,
    `jax.random.normal(PRNGKey(42), (features // 2,)) * scale`, drawn on the
    backend this runs on.

    jax 0.5.0 made the partitionable threefry stream the default, which
    changed every draw; a run's own `requirements.txt` names its version.
    The stream's bits are the same on every backend, but the normal
    transform (`erf_inv`) rounds its last bit the backend's way, so a table
    drawn on a GPU can differ from a CPU's by an ulp in an entry, as
    FlaxDiff's own draw followed the device the run trained on. A run
    trained on a TPU or a GPU drew its table there, so loading it on
    another backend can hand it a table one ulp off the one it trained
    with in an entry.
    """
    release = tuple(int(part) for part in jax_version.split(".")[:2])
    with jax.threefry_partitionable(release >= (0, 5)):
        draw = jax.random.normal(jax.random.PRNGKey(42), (features // 2,), dtype=jnp.float32)
    return np.asarray(draw * scale)


def simple_udit_fields(model: Mapping[str, object]) -> SimpleUDiTFields:
    """The `simple_udit` fields that compute what FlaxDiff 0.2's SimpleUDiT
    did, from the `model` entry of its run config."""
    unknown = set(model) - _READ - _UNREAD - set(_REFUSED)
    if unknown:
        raise ValueError(f"unknown FlaxDiff simple_udit config keys {sorted(unknown)}")
    for name in _REFUSED:
        if records.boolean(model.get(name, False), name):
            raise ValueError(f"a FlaxDiff simple_udit with {name} has no Dew counterpart")
    return SimpleUDiTFields(
        output_channels=records.integer(model["output_channels"], "output_channels"),
        patch_size=records.integer(model["patch_size"], "patch_size"),
        emb_features=records.integer(model["emb_features"], "emb_features"),
        num_layers=records.integer(model["num_layers"], "num_layers"),
        num_heads=records.integer(model["num_heads"], "num_heads"),
        mlp_ratio=records.integer(model.get("mlp_ratio", 4), "mlp_ratio"),
        norm_epsilon=records.number(model.get("norm_epsilon", 1e-5), "norm_epsilon"),
        adaln_silu=False, text_pooling="all")


def simple_udit_variables(weights: Mapping, model: Mapping[str, object], *,
                          jax_version: str) -> dict:
    """FlaxDiff 0.2 SimpleUDiT weights as the variables of Dew's model: the
    params under Dew's names and the Fourier table the run's jax drew."""
    params: dict = {}
    for name, value in weights.items():
        *parents, leaf = _MOVED.get(name, (name,))
        branch = params
        for parent in parents:
            branch = branch.setdefault(parent, {})
        branch[leaf] = value
    table = fourier_table(records.integer(model["emb_features"], "emb_features"), jax_version)
    return {"params": params,
            "constants": {"conditioning": {"time_embed": {"layers_0": {"frequencies": table}}}}}


def _condition(input_config: Mapping[str, object]) -> tuple[str, str, str]:
    """The one text condition of a FlaxDiff run: the keyword the model takes
    it under, the CLIP checkpoint that encodes it, and the unconditional text."""
    conditions = input_config.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 1:
        raise ValueError(f"a FlaxDiff simple_udit run conditions on one text; got {conditions!r}")
    condition = records.record(conditions[0], "conditions")
    encoder = records.record(condition["encoder"], "encoder")
    return (records.text(condition.get("model_key_override") or "textcontext", "model_key_override"),
            records.text(encoder["modelname"], "modelname"),
            records.text(condition.get("unconditional_input", ""), "unconditional_input"))


def load_flaxdiff(directory: str | os.PathLike, config: Mapping[str, object], *, jax_version: str,
                  ema: bool = True, best: bool = False, dtype: str | None = None) -> TextToImage:
    """A FlaxDiff text-to-image run as a Dew `TextToImage`.

    `directory` is one checkpoint step, `config` the run config FlaxDiff's
    trainer logged (`wandb.Api().run(path).config`), and `jax_version` the jax
    the run trained under, from its `requirements.txt`. `ema` and `best` pick
    the weights (`flaxdiff_weights`); `dtype` is the model's compute dtype.

    The text tower and the VAE load from the Hub under the names the config
    records, both computing in bfloat16 as FlaxDiff's did. A call samples the
    way FlaxDiff's trainer previewed the run: Euler ancestral over 200 steps
    of the Karras grid, classifier-free guidance 3.
    """
    from dew import models
    from dew.diffusion.presets import EDM
    from dew.inputs import Field, InputSpec
    from dew.nn.dit import TextContext
    from dew.nn.text_encoders import check_tree
    from dew.objectives.diffusion.config import StableDiffusionAutoencoder, TextCondition
    from dew.registry import resolve_dtype
    from dew.sampling import CFG, EulerAncestral, TextToImage

    architecture = config.get("architecture")
    if architecture != "simple_udit":
        raise ValueError(f"load_flaxdiff knows simple_udit runs, not {architecture!r}")
    arguments = records.record(config.get("arguments") or {}, "arguments")
    schedule = config.get("noise_schedule") or arguments.get("noise_schedule")
    if schedule != "edm":
        raise ValueError(f"a FlaxDiff simple_udit run trains on the EDM schedule, not {schedule!r}")
    autoencoder = config.get("autoencoder")
    if autoencoder != "stable_diffusion":
        raise ValueError(f"load_flaxdiff knows latent runs on the SD VAE, not {autoencoder!r}")
    options = config.get("autoencoder_opts") or {}
    options = records.record(json.loads(options) if isinstance(options, str) else options,
                             "autoencoder_opts")

    input_config = records.record(config["input_config"], "input_config")
    keyword, clip, unconditional = _condition(input_config)
    height, width, channels = records.integers(input_config["sample_data_shape"], "sample_data_shape")
    model_config = records.record(config["model"], "model")
    model = models.build("simple_udit", simple_udit_fields(model_config), dtype=resolve_dtype(dtype))
    variables = simple_udit_variables(flaxdiff_weights(read_checkpoint(directory), ema=ema, best=best),
                                      model_config, jax_version=jax_version)

    # FlaxDiff's encoders ran in bfloat16, and it read the VAE's main branch.
    condition = TextCondition(checkpoint=clip, dtype="bfloat16", unconditional=unconditional).build()
    vae = StableDiffusionAutoencoder(modelname=records.text(options["modelname"], "modelname"),
                                     revision="main").build()
    encoder = condition.encoder
    context = encoder.encode(encoder.params, encoder.tokenize([unconditional]))
    if not isinstance(context, TextContext):
        raise TypeError(f"{clip} encodes to {type(context).__name__}, not a TextContext")
    factor = vae.downscale_factor
    check_tree(variables, model,
               jnp.zeros((1, height // factor, width // factor, vae.latent_channels)), jnp.zeros((1,)),
               TextContext(jnp.zeros(context.hidden.shape), jnp.ones(context.mask.shape, jnp.int32)))

    inputs = InputSpec(sample=Field("image", (height, width, channels)), conditions={keyword: condition})
    params = {**variables, "encoders": {keyword: encoder.params}, "autoencoder": vae.params}
    return TextToImage(model, EDM()(), inputs, params, vae, steps=200, guidance=CFG(3.0),
                       sampler=EulerAncestral())


__all__ = ["SimpleUDiTFields", "flaxdiff_weights", "fourier_table", "load_flaxdiff",
           "read_checkpoint", "simple_udit_fields", "simple_udit_variables"]
