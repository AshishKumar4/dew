"""A trained model, rebuilt from its run, that turns prompts into images."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, replace
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.core import freeze

from dew.diffusion.process import Process
from dew.inputs import InputSpec
from dew.nn.autoencoders import AutoEncoder
from dew.nn.inputs import RowPlan, local_rows, mesh_of, request_key
from dew.objectives.base import Variables
from dew.sampling.guidance import CFG
from dew.sampling.sample import sample
from dew.sampling.solvers import DDIM, Solver


if TYPE_CHECKING:
    from dew.objectives.diffusion import DiffusionObjective
    from dew.training.distributed import Layout, MeshSpec

_UNSET: object = object()


@struct.dataclass
class DenoisingInputs:
    """Encoded conditioning and initial noise, placed the way a call runs them.

    ``rows`` counts this process's real prompts; on a mesh the arrays carry
    the padded, row-sharded batch a call consumes directly.
    """

    noise: jax.Array
    conditions: Mapping[str, object] = struct.field(default_factory=dict)
    unconditional: Mapping[str, object] = struct.field(default_factory=dict)
    rows: int = struct.field(pytree_node=False, default=0)


@struct.dataclass
class Images:
    """Decoded samples in [-1, 1], NHWC, keeping the placement the task ran with.

    ``host()`` reads this process's ``rows`` real rows back as a host array.
    """

    images: jax.Array
    rows: int = struct.field(pytree_node=False, default=0)

    def host(self) -> Images:
        return jax.tree.map(lambda leaf: local_rows(leaf)[:self.rows], self)


@dataclass(frozen=True, eq=False)
class TextToImage:
    """`pipe(prompts, seed=0)` or `pipe(prompts, steps=40, guidance=4.0, sampler=samplers.Heun(), key=key)`.

    `params` is the objective's whole tree, the EMA copy merged over the live
    weights when the run kept one, so a sample comes from the weights a run
    publishes. `steps`, `guidance` and `sampler` are the defaults a call
    omits; an objective or a loaded source sets them. `finish` runs on the
    decoded images under the same placement, for a source that ships a
    checker or an output transform.

    Weights keep their placement. On a mesh, prompts split into per-process
    rows over its batch axes and the result keeps that sharding; each row's
    initial noise comes from its global row index, so a pool draws what one
    process draws for the same prompts.
    """

    model: nn.Module
    process: Process
    inputs: InputSpec
    params: Variables
    autoencoder: AutoEncoder | None = None
    steps: int = 50
    guidance: CFG | None = None
    sampler: Solver[object] = DDIM()
    finish: Callable[[Variables, jax.Array], jax.Array] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", freeze(dict(self.params)))

    def bind(self, variables: Variables) -> TextToImage:
        """Bind another variables snapshot without rebuilding the model or encoders."""
        return replace(self, params=variables)

    @classmethod
    def from_objective(cls, objective: DiffusionObjective, variables: Variables) -> TextToImage:
        """The objective's model over `variables`, sampling the way its evaluation does."""
        return cls(objective.model, objective.process, objective.inputs, variables, objective.autoencoder,
                   steps=objective.steps, guidance=objective.guidance, sampler=objective.sampler)

    @classmethod
    def from_run(cls, directory: str, *, ema: bool = True, step: int | None = None,
                 mesh: MeshSpec | None = None, layout: Layout | None = None,
                 dtype: str | None = None) -> TextToImage:
        """The run in `directory`: its `run.json` built the way the recipe
        built it, and the weights of its latest checkpoint (or `step`).

        `ema` reads the averaged weights when the run kept them. With `mesh`
        the weights restore straight onto that mesh under `layout`, the way
        the trainer places them; without one they land on the default device.
        """
        from dew.objectives.diffusion import DiffusionRunConfig

        objective = DiffusionRunConfig.load(directory).build()
        params = restore_variables(directory, ema=ema, step=step, mesh=mesh, layout=layout, dtype=dtype)
        return cls.from_objective(objective, params)

    @classmethod
    def from_pretrained(cls, repo_id: str, *, ema: bool = True, mesh: MeshSpec | None = None,
                        layout: Layout | None = None, dtype: str | None = None) -> TextToImage:
        """A run directory published to the Hugging Face Hub, as
        `dew.interop.hub.push_to_hub` writes it."""
        from dew.interop.hub import pull_from_hub

        return cls.from_run(os.fspath(pull_from_hub(repo_id)), ema=ema, mesh=mesh, layout=layout, dtype=dtype)

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """The per-example shape the model denoises: the sample field's, or
        its latent when an autoencoder sits in front of the model."""
        shape = self.inputs.sample.shape
        if self.autoencoder is None:
            return shape
        *lead, height, width, _ = shape
        factor = self.autoencoder.downscale_factor
        return (*lead, height // factor, width // factor, self.autoencoder.latent_channels)

    @property
    def _conditions(self) -> tuple[tuple[str, object], ...]:
        return tuple((keyword, condition.encoder) for keyword, condition in self.inputs.conditions.items())

    def _unconditional(self) -> dict:
        tokens = {keyword: condition.encoder.tokenize([condition.unconditional])
                  for keyword, condition in self.inputs.conditions.items()}
        return _encode(None)(self._conditions, self.params, jax.tree.map(jnp.asarray, tokens))

    def prepare(self, prompts: str | Sequence[str], *, key: jax.Array | None = None,
                seed: int | None = None) -> DenoisingInputs:
        """Encode prompts once and draw their noise; reuse with other solvers."""
        rows = [prompts] if isinstance(prompts, str) else list(prompts)
        if not rows or not all(isinstance(prompt, str) for prompt in rows):
            raise ValueError("prompts must be a non-empty string sequence")
        request = request_key(key, seed)
        plan = RowPlan.over(mesh_of(self.params), len(rows))
        tokens = {keyword: condition.encoder.tokenize(rows)
                  for keyword, condition in self.inputs.conditions.items()}
        placed = plan.place(plan.pad(tokens))
        given = _encode(plan.sharding)(self._conditions, self.params, placed)
        noise = _noise(plan.sharding)(self.process, plan.keys(request), self.latent_shape)
        return DenoisingInputs(noise, given, self._unconditional(), rows=plan.rows)

    def __call__(self, prompts: str | Sequence[str] | DenoisingInputs, *, steps: int | None = None,
                 guidance: CFG | float | None | object = _UNSET, sampler: Solver[object] | None = None,
                 key: jax.Array | None = None, seed: int | None = None) -> Images:
        """Images in [-1, 1], `[rows, H, W, C]`. `guidance` is a classifier-free
        guidance scale, or a `CFG` with its interval, or None for the plain
        conditional prediction; omitted, it is the task's default."""
        chosen = self.guidance if guidance is _UNSET else guidance
        if isinstance(chosen, (int, float)) and not isinstance(chosen, bool):
            chosen = CFG(float(chosen))
        if chosen is not None and not isinstance(chosen, CFG):
            raise ValueError("guidance must be a scale, a CFG value or None")
        request = request_key(key, seed)
        prepared = prompts if isinstance(prompts, DenoisingInputs) else self.prepare(prompts, key=request)
        if prepared.noise.ndim != len(self.latent_shape) + 1 or prepared.noise.shape[1:] != self.latent_shape:
            raise ValueError(f"initial noise must have shape [batch, {self.latent_shape}]")
        plan = RowPlan.over(mesh_of(self.params), prepared.rows)
        if prepared.noise.shape[0] != plan.global_rows:
            raise ValueError("the prepared inputs were placed for a different mesh")
        images = _run(plan.sharding)(self.model, self.process, self.autoencoder, self.finish,
                                     self.steps if steps is None else steps,
                                     self.sampler if sampler is None else sampler, chosen,
                                     self.params, prepared.conditions, prepared.unconditional, prepared.noise,
                                     jax.random.fold_in(request, 1))
        return Images(images, rows=plan.rows)


def restore_variables(directory: str, *, ema: bool, step: int | None, mesh: MeshSpec | None,
                      layout: Layout | None, dtype: str | None) -> Variables:
    """A run's published variables, restored onto a mesh under a layout or onto the default device.

    The checkpoint is its own template. `ema` merges the averaged copy over
    the live weights when the run kept one; `dtype` casts the floating leaves.
    """
    from dew.checkpoints import Checkpoints
    from dew.objectives.base import merge
    from dew.training.distributed import Layout as DefaultLayout, build_mesh

    checkpoints = Checkpoints(directory)
    stored = checkpoints.stored(step)
    template = {"params": stored["params"]}
    averaged = ema and stored["ema"] is not None
    if averaged:
        template["ema"] = stored["ema"]
    if mesh is not None:
        device_mesh = build_mesh(mesh)
        placement = (DefaultLayout() if layout is None else layout).shardings(device_mesh, template)
        template = jax.tree.map(
            lambda leaf, sharding: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype, sharding=sharding),
            template, placement)
    values, _ = checkpoints.restore(template, step=step)
    params = values["params"]
    if averaged:
        params = merge(params, values["ema"])
    if dtype is not None:
        params = cast_floating(params, dtype)
    return params


def cast_floating(tree, dtype: str):
    """Every floating leaf as `dtype`, keeping each leaf's placement."""
    target = jnp.dtype(dtype)
    return jax.tree.map(
        lambda leaf: leaf.astype(target) if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, tree)


@functools.lru_cache(maxsize=None)
def _encode(rows: jax.sharding.NamedSharding | None):
    def encode(conditions, params, tokens):
        return {keyword: encoder.encode(params["encoders"][keyword], tokens[keyword])
                for keyword, encoder in conditions}
    return jax.jit(encode, static_argnums=(0,), in_shardings=(None, rows), out_shardings=rows)


@functools.lru_cache(maxsize=None)
def _noise(rows: jax.sharding.NamedSharding | None):
    def noise(process, keys, shape):
        return jax.vmap(lambda key: process.noise(key, shape))(keys)
    return jax.jit(noise, static_argnums=(0, 2), in_shardings=(rows,), out_shardings=rows)


@functools.lru_cache(maxsize=None)
def _run(rows: jax.sharding.NamedSharding | None):
    # Rebinding weights must not change the static compilation identity.
    def run(model, process, autoencoder, finish, steps, sampler, guidance, params, given, null, x_T, key):
        variables = {name: value for name, value in params.items() if name not in ("encoders", "autoencoder")}
        denoise = process.denoiser(model, variables, given, None if guidance is None else null)
        samples = sample(denoise, x_T, steps, solver=sampler, guidance=guidance, key=key)
        if autoencoder is not None:
            samples = autoencoder.decode(params["autoencoder"], samples)
        samples = jnp.clip(samples, -1.0, 1.0)
        return samples if finish is None else finish(params, samples)
    return jax.jit(run, static_argnums=(0, 1, 2, 3, 4, 5, 6),
                   in_shardings=(None, rows, None, rows, None), out_shardings=rows)
