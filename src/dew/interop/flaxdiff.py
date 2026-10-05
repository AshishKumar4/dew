"""FlaxDiff checkpoints as Dew models.

FlaxDiff (github.com/AshishKumar4/FlaxDiff) is the project Dew grew out of.
Its trainer saved an orbax tree at every checkpoint step, `{"state":
{"params", "ema_params", "opt_state", ...}, "best_state": {...}, ...}`, and
logged the run's config to wandb, so the architecture lives in the config and
not in the tree. `TextToImage.from_flaxdiff` reads one step and that config
and returns the run as a `TextToImage` over Dew's own model, built with the
fields that compute what FlaxDiff computed. This module maps names and config
and nothing else.

It knows `simple_udit`, the U-shaped DiT, and `hybrid_dit`, the S5-attention
DiT. Both project the conditioning vector without a SiLU and average the
text over every position, so Dew builds them with
`adaln_silu=False, text_pooling="all"`. FlaxDiff drew the Fourier
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
from jax.typing import DTypeLike

from dew import records

if TYPE_CHECKING:
    from dew.diffusion.process import Process
    from dew.registry import DtypeName
    from dew.sampling.pipelines import TextToImage

# FlaxDiff 0.2's DiTs held these at their root; Dew nests them under the
# conditioning embed and the output head.
_MOVED = {"time_embed": ("conditioning", "time_embed"),
          "text_proj": ("conditioning", "text_context_proj"),
          "final_norm": ("output", "final_norm"),
          "final_proj": ("output", "final_proj")}
_HYBRID_MOVED = {"time_embed": ("conditioning", "time_embed"),
                 "text_context_proj": ("conditioning", "text_context_proj"),
                 "hilbert_projection": ("embed", "hilbert_projection"),
                 "patch_embed": ("embed", "patch_embed"),
                 "final_norm": ("output", "final_norm"),
                 "final_proj": ("output", "final_proj")}

_READ = frozenset({"output_channels", "patch_size", "emb_features", "num_layers", "num_heads",
                   "mlp_ratio", "norm_epsilon"})
_HYBRID_READ = frozenset({"ssm_state_dim", "ssm_attention_ratio", "use_2d_fusion", "use_zigzag"})
# Model config keys FlaxDiff 0.2's DiTs computed nothing with on a CPU
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


class HybridDiTFields(SimpleUDiTFields):
    """The `hybrid_dit` fields a FlaxDiff 0.2 run config sets."""

    ssm_state_dim: int
    ssm_attention_ratio: str
    use_2d_fusion: bool
    scan_order: Literal["raster", "zigzag"]


def read_checkpoint(directory: str | os.PathLike, *, ema: bool = True, best: bool = False) -> dict:
    """The model weights from one FlaxDiff checkpoint step holding `default/`.

    Read the averaged weights of the last state by default; `ema=False`
    selects live weights and `best=True` selects `best_state`. Return the
    parameter dict under FlaxDiff's model names as host arrays. Only that
    copy is restored, without allocating the optimizer or other states.

    The older 2024 aggregate format (`default/checkpoint`) is not supported.
    """
    import orbax.checkpoint as ocp
    from etils import epath

    path = epath.Path((Path(directory) / "default").resolve())
    state = "best_state" if best else "state"
    weights = "ema_params" if ema else "params"
    pytree = ocp.PyTreeCheckpointHandler()
    with ocp.Checkpointer(pytree) as checkpointer:
        metadata = pytree.metadata(path)
        selected = {state: {weights: metadata.tree[state][weights]}}
        restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), selected)
        tree = checkpointer.restore(path, args=ocp.args.PyTreeRestore(
            item=selected, restore_args=restore_args, partial_restore=True))
    return dict(tree[state][weights]["params"])


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
    return _variables(weights, model, _MOVED, jax_version)


def hybrid_dit_fields(model: Mapping[str, object]) -> HybridDiTFields:
    """The `hybrid_dit` fields that compute FlaxDiff 0.2's S5-attention DiT."""
    common = simple_udit_fields({name: value for name, value in model.items()
                                 if name not in _HYBRID_READ})
    return HybridDiTFields(
        **common,
        ssm_state_dim=records.integer(model.get("ssm_state_dim", 64), "ssm_state_dim"),
        ssm_attention_ratio=records.text(model.get("ssm_attention_ratio", "3:1"), "ssm_attention_ratio"),
        use_2d_fusion=records.boolean(model.get("use_2d_fusion", False), "use_2d_fusion"),
        scan_order="zigzag" if records.boolean(model.get("use_zigzag", False), "use_zigzag") else "raster")


def hybrid_dit_variables(weights: Mapping, model: Mapping[str, object], *,
                         jax_version: str) -> dict:
    """FlaxDiff 0.2 hybrid DiT weights and Fourier table under Dew's names."""
    return _variables(weights, model, _HYBRID_MOVED, jax_version)


def _variables(weights: Mapping, model: Mapping[str, object],
               moved: Mapping[str, tuple[str, ...]], jax_version: str) -> dict:
    params: dict = {}
    for name, value in weights.items():
        *parents, leaf = moved.get(name, (name,))
        branch = params
        for parent in parents:
            branch = branch.setdefault(parent, {})
        branch[leaf] = value
    table = fourier_table(records.integer(model["emb_features"], "emb_features"), jax_version)
    return {"params": params,
            "constants": {"conditioning": {"time_embed": {"layers_0": {"frequencies": table}}}}}


_ARCHITECTURES = {"simple_udit": (simple_udit_fields, simple_udit_variables),
                  "hybrid_dit": (hybrid_dit_fields, hybrid_dit_variables)}


def _condition(input_config: Mapping[str, object]) -> tuple[str, str, str]:
    """The one text condition of a FlaxDiff run: the keyword the model takes
    it under, the CLIP checkpoint that encodes it, and the unconditional text."""
    conditions = input_config.get("conditions")
    if not isinstance(conditions, list) or len(conditions) != 1:
        raise ValueError(f"a FlaxDiff run conditions on one text; got {conditions!r}")
    condition = records.record(conditions[0], "conditions")
    encoder = records.record(condition["encoder"], "encoder")
    return (records.text(condition.get("model_key_override") or "textcontext", "model_key_override"),
            records.text(encoder["modelname"], "modelname"),
            records.text(condition.get("unconditional_input", ""), "unconditional_input"))


def _towers(clip: str, unconditional: str, vae: str, dtype: DtypeName):
    """A run's CLIP condition and SD VAE, computing in `dtype`; FlaxDiff read
    the VAE's main branch."""
    from dew.objectives.diffusion.config import PretrainedAutoencoder, TextCondition

    return (TextCondition(checkpoint=clip, dtype=dtype, unconditional=unconditional).build(),
            PretrainedAutoencoder(modelname=vae, revision="main", dtype=dtype).build())


def text_to_image(directory: str | os.PathLike, config: Mapping[str, object], *, jax_version: str,
                  ema: bool = True, best: bool = False, dtype: DTypeLike | None = None) -> TextToImage:
    """A FlaxDiff text-to-image run as a Dew `TextToImage` (`TextToImage.from_flaxdiff`).

    `directory` is one checkpoint step, `config` the run config FlaxDiff's
    trainer logged (`wandb.Api().run(path).config`), and `jax_version` the jax
    the run trained under, from its `requirements.txt`. `ema` and `best` pick
    the weights (`read_checkpoint`); `dtype` is the model's compute dtype.

    The text tower and the VAE load from the Hub under the names the config
    records, both computing in bfloat16 as FlaxDiff's did. A call samples the
    way FlaxDiff's trainer previewed the run: Euler ancestral with
    classifier-free guidance 3 over FlaxDiff's own grid of the Karras
    schedule, 200 steps unless the call names another count.
    """
    from dew.diffusion.presets import EDM
    from dew.inputs import Field, InputSpec
    from dew.nn.dit import TextContext
    from dew.nn.text_encoders import check_tree
    from dew.registry import models, resolve_dtype
    from dew.sampling import CFG, EulerAncestral, TextToImage

    architecture = records.text(config.get("architecture"), "architecture")
    if architecture not in _ARCHITECTURES:
        raise ValueError(f"from_flaxdiff knows simple_udit and hybrid_dit runs, not {architecture!r}")
    arguments = records.record(config.get("arguments") or {}, "arguments")
    schedule = config.get("noise_schedule") or arguments.get("noise_schedule")
    if schedule != "edm":
        raise ValueError(f"a FlaxDiff {architecture} run trains on the EDM schedule, not {schedule!r}")
    autoencoder = config.get("autoencoder")
    if autoencoder != "stable_diffusion":
        raise ValueError(f"from_flaxdiff knows latent runs on the SD VAE, not {autoencoder!r}")
    options = config.get("autoencoder_opts") or {}
    options = records.record(json.loads(options) if isinstance(options, str) else options,
                             "autoencoder_opts")

    input_config = records.record(config["input_config"], "input_config")
    keyword, clip, unconditional = _condition(input_config)
    height, width, channels = records.integers(input_config["sample_data_shape"], "sample_data_shape")
    model_config = records.record(config["model"], "model")
    fields, convert = _ARCHITECTURES[architecture]
    model = models.build(architecture, fields(model_config), dtype=resolve_dtype(dtype))
    variables = convert(read_checkpoint(directory, ema=ema, best=best),
                         model_config, jax_version=jax_version)

    # FlaxDiff's encoders ran in bfloat16.
    condition, vae = _towers(clip, unconditional, records.text(options["modelname"], "modelname"), "bfloat16")
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
    process = EDM(regime="latent")()

    def grid(steps: int) -> tuple[Process, jax.Array]:
        # FlaxDiff's `get_steps(1000, 0, steps)` truncates an evenly spaced
        # per-mille grid to int16 and `scale_steps` maps it back onto the
        # schedule's [0, 1], so its times sit up to a thousandth below an
        # even ramp's.
        return process, jnp.linspace(0, 1000, steps, dtype=jnp.int16)[::-1] * (1 / 1000)

    return TextToImage(model, process, inputs, params, vae, steps=200, guidance=CFG(3.0),
                       solver=EulerAncestral(), grid=grid)
