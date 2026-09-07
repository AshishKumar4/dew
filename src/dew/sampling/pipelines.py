"""A trained model, rebuilt from its run, that turns prompts into images."""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, replace
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Generic, overload
from enum import Enum

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.core import freeze
from jax.typing import ArrayLike

from dew.diffusion.process import Process
from dew.inputs import InputSpec
from dew.nn.autoencoders import AutoEncoder
from dew.nn.inputs import ArrayT, RowPlan, generation_signature, local_rows, mesh_of, request_key
from dew.artifacts import agree_process_phase
from jax.experimental import multihost_utils
from dew.objectives.base import Variables
from dew.sampling.guidance import CFG
from dew.sampling.sample import sample
from dew.sampling.solvers import DDIM, Solver


if TYPE_CHECKING:
    from dew.objectives.diffusion import DiffusionObjective
    from dew.training.distributed import Layout, MeshSpec

class _Default(Enum):
    GUIDANCE = "guidance"


@struct.dataclass
class DenoisingInputs:
    """Encoded conditioning and initial noise, placed the way a call runs them.

    ``rows`` counts this process's real prompts; on a mesh the arrays carry
    the padded, row-sharded batch a call consumes directly.
    """

    noise: jax.Array
    conditions: Mapping[str, object] = struct.field(default_factory=dict)
    unconditional: Mapping[str, object] = struct.field(default_factory=dict)
    rows: int | None = struct.field(pytree_node=False, default=None)
    grid_steps: int | None = struct.field(pytree_node=False, default=None)
    process: Process | None = struct.field(pytree_node=False, default=None)
    times: tuple[float, ...] | None = struct.field(pytree_node=False, default=None)


@struct.dataclass
class Images(Generic[ArrayT]):
    """Decoded samples in [-1, 1], NHWC, keeping the placement the task ran with.

    ``host()`` reads this process's ``rows`` real rows back as a host array.
    """

    images: ArrayT | None
    rows: int | None = struct.field(pytree_node=False, default=None)
    latents: ArrayT | None = None

    def host(self) -> Images[np.ndarray]:
        return jax.tree.map(lambda leaf: local_rows(leaf)[:self.rows], self)


@dataclass(frozen=True, eq=False)
class TextToImage:
    """`pipe(prompts, seed=0)` or `pipe(prompts, steps=40, guidance=4.0, sampler=samplers.Heun(), key=key)`.

    `params` is the objective's whole tree, the EMA copy merged over the live
    weights when the run kept one, so a sample comes from the weights a run
    publishes. `steps`, `guidance` and `sampler` are the defaults a call
    omits; an objective or a loaded source sets them. `grid` prepares the
    process and its explicit time grid for a step count, for a source whose
    sampler pairs its own sigma and model-time tables; `final_denoise`
    False ends a trajectory the way those samplers do. `finish` runs on the
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
    grid: Callable[[int], tuple[Process, jax.Array]] | None = None
    final_denoise: bool = True
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
        the trainer places them; without one the default mesh uses the current pool.
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

    def prepared_process(self, steps: int) -> tuple[Process, tuple[float, ...] | None]:
        """The process and explicit time grid a `steps` call walks; the grid
        is concrete, so the compiled trajectory has its length and values."""
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        if self.grid is None:
            return self.process, None
        process, times = self.grid(steps)
        return process, _time_grid(times)

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

    def _unconditional(self, tokens, plan: RowPlan) -> dict:
        leaves = jax.tree.leaves(tokens)
        if leaves and leaves[0].shape[0] != 1:
            return _encode(plan.sharding)(self._conditions, self.params, plan.place(plan.pad(tokens)))
        return _encode(None)(self._conditions, self.params, jax.tree.map(jnp.asarray, tokens))

    @overload
    def prepare(self, prompts: str | Sequence[str | Mapping[str, object]], *,
                key: jax.Array, seed: None = None, steps: int | None = None,
                unconditional: str | Sequence[str | Mapping[str, object]] | None = None,
                image: ArrayLike | None = None, image_latents: ArrayLike | None = None,
                mask: ArrayLike | None = None, noise: ArrayLike | None = None, initial: ArrayLike | None = None,
                times: ArrayLike | Sequence[float] | None = None,
                encode_key: jax.Array | None = None) -> DenoisingInputs: ...

    @overload
    def prepare(self, prompts: str | Sequence[str | Mapping[str, object]], *,
                key: None = None, seed: int, steps: int | None = None,
                unconditional: str | Sequence[str | Mapping[str, object]] | None = None,
                image: ArrayLike | None = None, image_latents: ArrayLike | None = None,
                mask: ArrayLike | None = None, noise: ArrayLike | None = None, initial: ArrayLike | None = None,
                times: ArrayLike | Sequence[float] | None = None,
                encode_key: jax.Array | None = None) -> DenoisingInputs: ...

    def prepare(self, prompts: str | Sequence[str | Mapping[str, object]], *,
                key: jax.Array | None = None, seed: int | None = None, steps: int | None = None,
                unconditional: str | Sequence[str | Mapping[str, object]] | None = None,
                image: ArrayLike | None = None, image_latents: ArrayLike | None = None,
                mask: ArrayLike | None = None, noise: ArrayLike | None = None, initial: ArrayLike | None = None,
                times: ArrayLike | Sequence[float] | None = None,
                encode_key: jax.Array | None = None) -> DenoisingInputs:
        """Encode conditions and construct the initial state on a concrete grid.

        Images are uint8 or normalized floating NHWC pixels at the task's
        geometry. image_latents skips VAE encoding. A mask adds spatial
        conditioning to both guidance branches. noise is unit Gaussian noise
        for noising a clean image; initial is an already-noisy latent state
        for a continuation or refiner handoff and is never noised again.
        Explicit times select a partial trajectory in the prepared process.
        encode_key samples a VAE posterior; None uses its mean.
        """
        mesh = mesh_of(self.params)
        prepared = None
        error = None
        try:
            rows = [prompts] if isinstance(prompts, str) else list(prompts)
            if not rows or not all(isinstance(prompt, (str, Mapping)) for prompt in rows):
                raise ValueError("prompts must be a non-empty sequence of strings or conditioning records")
            request = request_key(key, seed)
            count = self.steps if steps is None else steps
            process, source_times = self.prepared_process(count)
            selected = _time_grid(times) if times is not None else source_times
            plan = RowPlan.over(mesh, len(rows))
            if unconditional is None:
                negatives = None
            else:
                negatives = [unconditional] if isinstance(unconditional, str) else list(unconditional)
                if len(negatives) not in (1, len(rows)):
                    raise ValueError("unconditional inputs need one row or one row per prompt")
            tokens = {keyword: condition.encoder.tokenize(rows)
                      for keyword, condition in self.inputs.conditions.items()}
            null_tokens = {keyword: condition.encoder.tokenize(
                [condition.unconditional] if negatives is None else negatives)
                for keyword, condition in self.inputs.conditions.items()}
            for leaf in jax.tree.leaves(tokens):
                if leaf.ndim < 1 or leaf.shape[0] != len(rows):
                    raise ValueError("tokenized conditions must have one row per prompt")
            for leaf in jax.tree.leaves(null_tokens):
                if leaf.ndim < 1 or leaf.shape[0] not in (1, len(rows)):
                    raise ValueError("unconditional tokens must have one row or one per prompt")
            shape = self.latent_shape
            if image is not None and image_latents is not None:
                raise ValueError("pass image or image_latents, not both")
            if mask is not None and self.autoencoder is None:
                raise ValueError("masked-image conditioning requires an autoencoder")
            if noise is not None and (image is None and image_latents is None or initial is not None):
                raise ValueError("noise is for noising a clean image; initial is already noisy")
            if mask is not None and image is None:
                raise ValueError("a mask requires its image pixels")
            if encode_key is not None:
                encode_key = request_key(encode_key, None)
            data = {}
            if image is not None:
                pixels = _image_rows(image, len(rows), self.inputs.sample.shape, "image")
                data["image"] = pixels.astype(np.float32) / 127.5 - 1 if pixels.dtype == np.uint8 else pixels
            for name, value in (("image_latents", image_latents), ("noise", noise), ("initial", initial)):
                if value is not None:
                    data[name] = _image_rows(value, len(rows), shape, name)
            if mask is not None:
                value = _image_rows(mask, len(rows), (*self.inputs.sample.shape[:-1], 1), "mask")
                data["mask"] = (value >= (128 if value.dtype == np.uint8 else 0.5)).astype(np.float32)
            controls = (plan.rows, count, selected, shape,
                        tuple(np.asarray(jax.random.key_data(request))),
                        None if encode_key is None else tuple(np.asarray(jax.random.key_data(encode_key))))
            signature = generation_signature((tokens, null_tokens, data), controls)
            prepared = plan, process, request, tokens, null_tokens, shape, count, selected, data, signature
        except Exception as failure:
            error = failure
        if mesh is not None:
            agree_process_phase(error, phase="image input preparation")
        elif error is not None:
            raise error
        assert prepared is not None
        plan, process, request, tokens, null_tokens, shape, count, selected, data, signature = prepared
        if plan.processes > 1:
            multihost_utils.assert_equal(signature, "image input shapes and sampling must agree across processes")
        given = _encode(plan.sharding)(self._conditions, self.params, plan.place(plan.pad(tokens)))
        null = self._unconditional(null_tokens, plan)
        if data:
            start = process.times(count)[0] if selected is None else selected[0]
            initial_state, spatial = _image_start(plan.sharding)(
                self.autoencoder, process, shape, self.params, plan.place(plan.pad(data)),
                plan.keys(request), encode_key, start)
            if spatial:
                given = {**given, **spatial}
                null = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (plan.global_rows, *leaf.shape[1:]))
                                    if leaf.shape[0] == 1 else leaf, null)
                null = {**null, **spatial}
        else:
            initial_state = _noise(plan.sharding)(process, plan.keys(request), shape)
        owns_grid = self.grid is not None or times is not None
        return DenoisingInputs(initial_state, given, null, rows=plan.rows,
                               grid_steps=count if owns_grid else None,
                               process=process if owns_grid else None, times=selected)

    @overload
    def __call__(self, prompts: str | Sequence[str | Mapping[str, object]] | DenoisingInputs, *,
                 steps: int | None = None, guidance: CFG | float | None | _Default = _Default.GUIDANCE,
                 sampler: Solver[object] | None = None, key: jax.Array,
                 seed: None = None, decode: bool = True) -> Images: ...

    @overload
    def __call__(self, prompts: str | Sequence[str | Mapping[str, object]] | DenoisingInputs, *,
                 steps: int | None = None, guidance: CFG | float | None | _Default = _Default.GUIDANCE,
                 sampler: Solver[object] | None = None, key: None = None,
                 seed: int, decode: bool = True) -> Images: ...

    def __call__(self, prompts: str | Sequence[str | Mapping[str, object]] | DenoisingInputs, *,
                 steps: int | None = None, guidance: CFG | float | None | _Default = _Default.GUIDANCE,
                 sampler: Solver[object] | None = None, key: jax.Array | None = None,
                 seed: int | None = None, decode: bool = True) -> Images:
        """Images in [-1, 1], `[rows, H, W, C]`. `guidance` is a classifier-free
        guidance scale, or a `CFG` with its interval, or None for the plain
        conditional prediction; omitted, it is the task's default."""
        mesh = mesh_of(self.params)
        settings = None
        error = None
        try:
            chosen = self.guidance if guidance is _Default.GUIDANCE else guidance
            if isinstance(chosen, (int, float)) and not isinstance(chosen, bool):
                chosen = CFG(float(chosen))
            if chosen is not None and not isinstance(chosen, CFG):
                raise ValueError("guidance must be a scale, a CFG value or None")
            request = request_key(key, seed)
            prepared = prompts if isinstance(prompts, DenoisingInputs) else None
            default_count = (prepared.grid_steps if prepared is not None and prepared.grid_steps is not None
                             else self.steps)
            count = default_count if steps is None else steps
            if prepared is not None and prepared.times is not None:
                times = _time_grid(prepared.times)
                process = self.process if prepared.process is None else prepared.process
            else:
                process, times = self.prepared_process(count)
            if type(decode) is not bool:
                raise ValueError("decode must be a boolean")
            solver = self.sampler if sampler is None else sampler
            if prepared is not None:
                if prepared.grid_steps is not None and count != prepared.grid_steps:
                    raise ValueError("prepared noise belongs to a different source grid; prepare it for these steps")
                shape = self.latent_shape
                if prepared.noise.ndim != len(shape) + 1 or prepared.noise.shape[1:] != shape:
                    raise ValueError(f"initial noise must have shape [batch, {shape}]")
                if prepared.rows is None:
                    if isinstance(prepared.noise, jax.Array) and not prepared.noise.is_fully_addressable:
                        raise ValueError("global prepared arrays need the number of real local rows")
                    prepared = replace(prepared, rows=prepared.noise.shape[0])
                if type(prepared.rows) is not int or prepared.rows < 1:
                    raise ValueError("prepared inputs must declare a positive number of local rows")
                plan = RowPlan.over(mesh, prepared.rows)
                if prepared.noise.shape[0] != plan.global_rows or mesh_of(prepared.noise) != mesh:
                    raise ValueError("the prepared inputs were placed for a different mesh")
                for leaf in jax.tree.leaves(prepared.conditions):
                    if leaf.ndim < 1 or leaf.shape[0] != plan.global_rows:
                        raise ValueError("prepared conditions must match the noise batch")
            controls = (count, times, solver, chosen, self.final_denoise, decode,
                        tuple(np.asarray(jax.random.key_data(request))), prepared is not None)
            arrays = None if prepared is None else (prepared.noise, prepared.conditions, prepared.unconditional)
            signature = generation_signature(arrays, controls)
            settings = prepared, request, count, process, times, solver, chosen, signature
        except Exception as failure:
            error = failure
        if mesh is not None:
            agree_process_phase(error, phase="image sampling setup")
        elif error is not None:
            raise error
        assert settings is not None
        prepared, request, count, process, times, solver, chosen, signature = settings
        if mesh is not None and jax.process_count() > 1:
            multihost_utils.assert_equal(signature, "image execution controls must agree across processes")
        if prepared is None:
            assert not isinstance(prompts, DenoisingInputs)
            prepared = self.prepare(prompts, key=request, steps=count)
        assert prepared.rows is not None
        plan = RowPlan.over(mesh, prepared.rows)
        result = _run(plan.sharding)(self.model, process, self.autoencoder, self.finish, count,
                                     solver, chosen, self.final_denoise, times, decode, self.params,
                                     prepared.conditions, prepared.unconditional,
                                     prepared.noise, jax.random.fold_in(request, 1))
        return replace(result, rows=plan.rows)


def _time_grid(times) -> tuple[float, ...]:
    values = np.asarray(times, np.float32)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all() or np.any(np.diff(values) > 0):
        raise ValueError("times must be a finite descending grid with at least one point")
    return tuple(float(value) for value in values)


def _image_rows(value, rows: int, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = local_rows(value)
    if array.shape == shape:
        array = array[None]
    if array.ndim != len(shape) + 1 or array.shape[1:] != shape or array.shape[0] not in (1, rows):
        raise ValueError(f"{name} must have shape [{rows}, {shape}] or one broadcast row; got {array.shape}")
    if not jnp.issubdtype(array.dtype, jnp.number) and array.dtype != np.bool_:
        raise ValueError(f"{name} must be a numeric array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values")
    if name == "image" and array.dtype != np.uint8 and np.any((array < -1) | (array > 1)):
        raise ValueError("floating image pixels must be normalized to [-1, 1]; uint8 pixels are also accepted")
    return np.broadcast_to(array, (rows, *shape))


@functools.lru_cache(maxsize=None)
def _image_start(rows: jax.sharding.NamedSharding | None):
    def prepare(autoencoder, process, shape, params, data, keys, encode_key, start):
        spatial = {}
        pixels = data.get("image")
        if "initial" in data:
            value = data["initial"]
        else:
            clean = data.get("image_latents")
            if clean is None:
                if autoencoder is None:
                    clean = pixels
                else:
                    clean = autoencoder.encode(params["autoencoder"], pixels, encode_key)
            noise = data.get("noise")
            if noise is None:
                noise = jax.vmap(lambda key: jax.random.normal(key, shape))(keys)
            alpha, sigma = process.sampler_schedule.rates(start)
            value = alpha * clean + sigma * noise
        if "mask" in data:
            from dew.inputs.diffusion import latent_image_conditions
            spatial = latent_image_conditions(autoencoder, params["autoencoder"], pixels, data["mask"], encode_key)
        return value, spatial
    return jax.jit(prepare, static_argnums=(0, 1, 2),
                   in_shardings=(None, rows, rows, None, None), out_shardings=rows)


def restore_variables(directory: str, *, ema: bool, step: int | None, mesh: MeshSpec | None,
                      layout: Layout | None, dtype: str | None) -> Variables:
    """A run's published variables, restored onto the current mesh under a layout.

    The checkpoint is its own template. `ema` merges the averaged copy over
    the live weights when the run kept one; `dtype` casts the floating leaves.
    """
    from dew.checkpoints import Checkpoints
    from dew.objectives.base import merge
    from dew.training.distributed import Layout as DefaultLayout, MeshSpec as DefaultMesh, build_mesh

    checkpoints = Checkpoints(directory)
    stored = checkpoints.stored(step)
    template = {"params": stored["params"]}
    if ema and stored.get("ema") is None:
        raise ValueError("the run keeps no EMA; request the live policy with ema=False")
    averaged = ema
    if averaged:
        template["ema"] = stored["ema"]
    device_mesh = build_mesh(DefaultMesh() if mesh is None else mesh)
    chosen_layout = DefaultLayout() if layout is None else layout
    placement = chosen_layout.shardings(device_mesh, template)
    chosen_layout.check(template["params"], placement["params"], device_mesh)
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
    def run(model, process, autoencoder, finish, steps, sampler, guidance, final_denoise, times, decode,
            params, given, null, x_T, key):
        variables = {name: value for name, value in params.items() if name not in ("encoders", "autoencoder")}
        denoise = process.denoiser(model, variables, given, None if guidance is None else null)
        with jax.ensure_compile_time_eval():
            grid = None if times is None else jnp.asarray(times, jnp.float32)
        if grid is None:
            latents = sample(denoise, x_T, steps, solver=sampler, guidance=guidance,
                             key=key, final_denoise=final_denoise)
        else:
            latents = sample(denoise, x_T, solver=sampler, guidance=guidance,
                             key=key, times=grid, final_denoise=final_denoise)
        if not decode:
            return Images(None, latents=latents)
        images = autoencoder.decode(params["autoencoder"], latents) if autoencoder is not None else latents
        images = jnp.clip(images, -1.0, 1.0)
        if finish is not None:
            images = finish(params, images)
        return Images(images, latents=latents)
    return jax.jit(run, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                   in_shardings=(None, rows, None, rows, None), out_shardings=rows)
