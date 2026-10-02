"""A trained model, rebuilt from its run, that turns prompts into images."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from importlib import import_module
from typing import TYPE_CHECKING, Generic

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.core import freeze
from jax.experimental import multihost_utils
from jax.typing import ArrayLike
from typing_extensions import TypeVar

from dew.artifacts import agreed, uint8_pixels
from dew.diffusion.process import Conditioning, Process
from dew.inputs import InputSpec, unit_range
from dew.nn.autoencoders import AutoEncoder
from dew.nn.inputs import RowPlan, generation_signature, local_rows, mesh_of, request_key
from dew.objectives.base import FROZEN, Variables
from dew.registry import dtype_name, resolve_dtype
from dew.sampling.guidance import CFG, Guidance
from dew.sampling.sample import sample
from dew.sampling.solvers import DDIM, Solver
from dew.telemetry.profile import region

ArrayT = TypeVar("ArrayT", bound=jax.Array | np.ndarray, default=jax.Array, covariant=True)

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from dew.objectives.diffusion import DiffusionObjective
    from dew.training.distributed import Layout, MeshSpec
    from dew.training.quantization import Quantization

class _Default(Enum):
    """The sentinel that tells `guidance=None` from an omitted `guidance=`.

    None is a value a caller passes to turn guidance off, so the default
    cannot be None.
    """

    GUIDANCE = "guidance"


@dataclass(frozen=True)
class _Resolved:
    """What `prepare` settles on the host before any process runs a model.

    `plan` is this process's rows of the request, `process` and `times` the
    trajectory it walks, `request` the key its noise is drawn from, and
    `posterior` the key a VAE encode samples with. `tokens` and `null_tokens`
    are the two conditioning branches, `samples` whatever image, mask, noise
    or latent state the caller handed over, and `signature` the value the
    pool compares before any of it reaches a device.
    """

    plan: RowPlan
    process: Process
    request: jax.Array
    tokens: dict
    null_tokens: dict
    shape: tuple[int, ...]
    count: int
    times: tuple[float, ...] | None
    samples: dict
    posterior: jax.Array | None
    signature: object


@struct.dataclass
class DenoisingInputs:
    """Encoded conditioning and initial noise, placed the way a call runs them.

    ``rows`` counts this process's real prompts; on a mesh the arrays carry
    the padded, row-sharded batch a call consumes directly.
    """

    noise: jax.Array
    conditions: Mapping[str, Conditioning] = struct.field(default_factory=dict)
    unconditional: Mapping[str, Conditioning] = struct.field(default_factory=dict)
    rows: int | None = struct.field(pytree_node=False, default=None)
    grid_steps: int | None = struct.field(pytree_node=False, default=None)
    process: Process | None = struct.field(pytree_node=False, default=None)
    times: tuple[float, ...] | None = struct.field(pytree_node=False, default=None)


@struct.dataclass
class Images(Generic[ArrayT]):
    """Decoded samples in [-1, 1], NHWC (NTHWC for a video field), keeping the
    placement the task ran with.

    ``host()`` reads this process's ``rows`` real rows back as a host array;
    ``pil()`` reads an image batch's back as 8-bit images.
    """

    images: ArrayT | None
    rows: int | None = struct.field(pytree_node=False, default=None)
    latents: ArrayT | None = None

    def host(self) -> Images[np.ndarray]:
        """This process's real rows as host arrays, without the padding a
        row plan added to fill the devices."""
        return jax.tree.map(lambda leaf: local_rows(leaf)[:self.rows], self)

    def pil(self) -> list[PILImage]:
        """This process's real rows of an NHWC image batch as RGB images (or
        grayscale, for one channel), their pixels quantized by
        `dew.artifacts.uint8_pixels`."""
        from PIL import Image

        if self.images is None:
            raise ValueError("these samples kept only their latents; call the task with decode=True")
        if self.images.ndim != 4 or self.images.shape[-1] not in (1, 3):
            raise ValueError(f"pil() takes [N, H, W, 3] or [N, H, W, 1] images, not samples of shape "
                             f"{self.images.shape}")
        pixels = uint8_pixels(local_rows(self.images)[:self.rows])
        return [Image.fromarray(row[..., 0] if row.shape[-1] == 1 else row) for row in pixels]


@dataclass(frozen=True, eq=False)
class TextToImage:
    """`pipe(prompts, key=0)` or `pipe(prompts, steps=40, guidance=4.0, solver=Heun(), key=key)`.

    `params` is the objective's whole tree, the EMA copy merged over the live
    weights when the run kept one, so a sample comes from the weights a run
    publishes. `steps`, `guidance` and `solver` are the defaults a call
    omits; an objective or a loaded source sets them. `grid` prepares the
    process and its explicit time grid for a step count, for a source whose
    solver pairs its own sigma and model-time tables; `final_denoise`
    False ends a trajectory the way those solvers do. `finish` runs on the
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
    guidance: Guidance | None = None
    solver: Solver[object] = field(default_factory=DDIM)
    grid: Callable[[int], tuple[Process, jax.Array]] | None = None
    final_denoise: bool = True
    finish: Callable[[Variables, jax.Array], jax.Array] | None = None
    blank: Callable[[dict], dict] | None = None
    """The task's own unconditional branch in the dtypes of a conditional one,
    encoded once by whoever built this task (`DiffusionObjective.blank_conditions`);
    None encodes it on every call, for a source that has none."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", freeze(dict(self.params)))

    def bind(self, variables: Variables) -> TextToImage:
        """Bind another variables snapshot without rebuilding the model or encoders."""
        return replace(self, params=variables)

    def quantized(self, spec: Quantization) -> TextToImage:
        """This task with its denoiser's weights stored quantized as `spec`
        says and its matmuls computing with them
        (`dew.training.quantization.quantize_for_serving`). The encoders and
        the autoencoder keep their weights."""
        from dew.training.quantization import quantize_for_serving

        example = self.prepare("", key=0, steps=1)
        denoiser = {
            name: value for name, value in self.params.items() if name not in ("encoders", "autoencoder")
        }
        model, variables = quantize_for_serving(self.model, denoiser, spec, example.noise,
                                                jnp.zeros(example.noise.shape[:1]), **example.conditions)
        return replace(self, model=model, params={**self.params, **variables})

    @classmethod
    def from_objective(cls, objective: DiffusionObjective, variables: Variables) -> TextToImage:
        """The objective's model over `variables`, sampling the way its
        evaluation does; a loss-only head the objective trains is dropped."""
        from dew.objectives.diffusion.objective import _without_loss_heads

        autoencoder, variables = objective.published_autoencoder(variables)
        return cls(objective.model, objective.process, objective.inputs,
                   _without_loss_heads(variables), autoencoder,
                   steps=objective.steps, guidance=objective.guidance, solver=objective.solver,
                   blank=objective.blank_conditions)

    @classmethod
    def from_run(cls, directory: str, *, ema: bool | None = None, step: int | str | None = None,
                 mesh: MeshSpec | None = None, layout: Layout | None = None,
                 dtype: str | None = None, param_dtype: str | None = None) -> TextToImage:
        """The run in `directory`: its `run.json` built the way the recipe
        built it, and the weights of its latest checkpoint (or `step`).

        `ema` None reads the averaged weights when the run kept them and the
        live ones when it kept none; True requires the averaged ones and False
        reads the live ones. A run whose average is a reference policy rather
        than the trained one (Flow-GRPO's frozen KL reference) reads its live
        policy. With `mesh` the weights restore straight onto that mesh under
        `layout`, the way the trainer places them; without one the default
        mesh uses the current pool.
        dtype overrides computation in the model, encoders and VAE. param_dtype
        overrides parameter storage; None preserves checkpoint storage exactly.
        """
        from dew.objectives.diffusion import DiffusionRunConfig
        from dew.registry import objectives

        import_module("dew.objectives.rl.flow")  # registers the flow_grpo a record names
        config = DiffusionRunConfig.load(directory)
        compute = dtype_name(resolve_dtype(dtype))
        if compute is not None:
            config = replace(config, model=replace(config.model, dtype=compute),
                             text=None if config.text is None else replace(config.text, dtype=compute),
                             audio=None if config.audio is None else replace(config.audio, dtype=compute),
                             autoencoder=None if config.autoencoder is None else
                             replace(config.autoencoder, dtype=compute))
        averaged = False if objectives[config.objective]._ema_is_reference else ema
        params = restore_variables(directory, ema=averaged, step=step, mesh=mesh, layout=layout,
                                   param_dtype=param_dtype, parameter_roots=config.parameter_roots)
        objective = config.build(variables=params)
        return cls.from_objective(objective, params)

    @classmethod
    def from_pretrained(cls, repo_id: str, *, ema: bool | None = None, mesh: MeshSpec | None = None,
                        layout: Layout | None = None, dtype: str | None = None,
                        param_dtype: str | None = None) -> TextToImage:
        """A run directory published to the Hugging Face Hub, as
        `HfApi().upload_folder` of the run directory writes it."""
        from dew.interop.hub import pull_from_hub

        return cls.from_run(os.fspath(pull_from_hub(repo_id)), ema=ema, mesh=mesh, layout=layout,
                            dtype=dtype, param_dtype=param_dtype)

    @classmethod
    def from_flaxdiff(cls, directory: str | os.PathLike, config: Mapping[str, object], *, jax_version: str,
                      ema: bool = True, best: bool = False, dtype: str | None = None) -> TextToImage:
        """A FlaxDiff text-to-image run (`simple_udit` or `hybrid_dit` on the
        SD VAE) over Dew's own model.

        `directory` is one checkpoint step, `config` the run config FlaxDiff's
        trainer logged, and `jax_version` the jax the run trained under, from
        its `requirements.txt`. `ema` and `best` pick the weights; `dtype` is
        the model's compute dtype. `dew.interop.flaxdiff` reads the format.
        """
        from dew.interop import flaxdiff

        return flaxdiff.text_to_image(directory, config, jax_version=jax_version, ema=ema, best=best,
                                      dtype=dtype)

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
        return shape if self.autoencoder is None else self.autoencoder.latent_shape(shape)

    @property
    def _conditions(self) -> tuple[tuple[str, object], ...]:
        return tuple((keyword, condition.encoder) for keyword, condition in self.inputs.conditions.items())

    def _unconditional(self, tokens, plan: RowPlan, given: dict, *, configured: bool) -> dict:
        """The unconditional branch: the value already encoded for the task's
        own unconditional prompt, and an encode of the caller's negatives."""
        if configured and self.blank is not None:
            return self.blank(given)
        leaves = jax.tree.leaves(tokens)
        if leaves and leaves[0].shape[0] != 1:
            return _encode(plan.sharding)(self._conditions, self.params, plan.place(plan.pad(tokens)))
        return _encode(None)(self._conditions, self.params, jax.tree.map(jnp.asarray, tokens))


    def prepare(
        self,
        prompts: str | Sequence[str | Mapping[str, object]],
        *,
        key: int | jax.Array | None = None,
        steps: int | None = None,
        unconditional: str | Sequence[str | Mapping[str, object]] | None = None,
        image: ArrayLike | None = None,
        image_latents: ArrayLike | None = None,
        mask: ArrayLike | None = None,
        noise: ArrayLike | None = None,
        initial: ArrayLike | None = None,
        times: ArrayLike | Sequence[float] | None = None,
        encode_key: int | jax.Array | None = None,
    ) -> DenoisingInputs:
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

        def resolve() -> _Resolved:
            return self._resolved(mesh, prompts, key=key, steps=steps,
                                  unconditional=unconditional, image=image,
                                  image_latents=image_latents, mask=mask, noise=noise,
                                  initial=initial, times=times, encode_key=encode_key)

        settled = (agreed("image input preparation", resolve) if mesh is not None else resolve())
        plan, process, count, selected = settled.plan, settled.process, settled.count, settled.times
        if plan.processes > 1:
            multihost_utils.assert_equal(settled.signature,
                                         "image input shapes and sampling must agree across processes")
        with region("inference.image.prepare"):
            given, null, initial_state = self._encoded(settled, configured=unconditional is None)
        owns_grid = self.grid is not None or times is not None
        return DenoisingInputs(initial_state, given, null, rows=plan.rows,
                               grid_steps=count if owns_grid else None,
                               process=process if owns_grid else None, times=selected)

    def _settings(self, mesh, prompts, *, steps, guidance, solver, key, decode):
        """Everything one call settles on the host before it runs the model.

        A caller who hands over `DenoisingInputs` gets them checked against
        this task's geometry and this mesh here; everything else resolves
        the grid, the solver and the guidance a call runs with. The value
        ends with the signature a pool compares.
        """
        chosen = self.guidance if guidance is _Default.GUIDANCE else guidance
        if isinstance(chosen, (int, float)) and not isinstance(chosen, bool):
            chosen = CFG(float(chosen))
        if chosen is not None and not isinstance(chosen, Guidance):
            raise ValueError("guidance must be a scale, a guidance value or None")
        request = request_key(key)
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
        solver = self.solver if solver is None else solver
        if prepared is not None:
            prepared = self._checked_inputs(prepared, mesh, count)
        controls = (count, times, solver, chosen, self.final_denoise, decode,
                    tuple(jax.device_get(jax.random.key_data(request))), prepared is not None,
                    None if prepared is None else prepared.rows)
        arrays = None if prepared is None else (prepared.noise, prepared.conditions, prepared.unconditional)
        signature = generation_signature(arrays, controls)
        return prepared, request, count, process, times, solver, chosen, signature

    def _checked_inputs(self, prepared: DenoisingInputs, mesh, count: int) -> DenoisingInputs:
        """`prepared` with its row count filled in, checked against this call.

        A caller may hand over inputs prepared for another step count, at
        another geometry, or placed for another mesh; each is refused by
        name. Global arrays carry no local row count of their own, so one
        that reports none has to declare it.
        """
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
        return prepared

    def _resolved(self, mesh, prompts, *, key, steps, unconditional, image,
                  image_latents, mask, noise, initial, times, encode_key) -> _Resolved:
        """Everything `prepare` settles on the host, in one value.

        This is the half a pool has to agree on: every refusal a caller can
        earn is raised here, and the signature at the end is what the ranks
        compare before any of them touches a device.
        """
        rows = [prompts] if isinstance(prompts, str) else list(prompts)
        if not rows or not all(isinstance(prompt, (str, Mapping)) for prompt in rows):
            raise ValueError("prompts must be a non-empty sequence of strings or conditioning records")
        request = request_key(key)
        count = self.steps if steps is None else steps
        process, source_times = self.prepared_process(count)
        selected = _time_grid(times) if times is not None else source_times
        plan = RowPlan.over(mesh, len(rows))
        tokens, null_tokens = self._tokenized(rows, unconditional)
        shape = self.latent_shape
        if image is not None and image_latents is not None:
            raise ValueError("pass image or image_latents, not both")
        if mask is not None and self.autoencoder is None:
            raise ValueError("masked-image conditioning requires an autoencoder")
        if noise is not None and ((image is None and image_latents is None) or initial is not None):
            raise ValueError("noise is for noising a clean image; initial is already noisy")
        if mask is not None and image is None:
            raise ValueError("a mask requires its image pixels")
        posterior = encode_key
        if posterior is not None:
            posterior = request_key(posterior)
        samples = self._supplied(len(rows), shape, image=image, image_latents=image_latents,
                                 mask=mask, noise=noise, initial=initial)
        controls = (plan.rows, count, selected, shape,
                    tuple(jax.device_get(jax.random.key_data(request))),
                    None if posterior is None else tuple(np.asarray(jax.random.key_data(posterior))))
        signature = generation_signature((tokens, null_tokens, samples), controls)
        return _Resolved(plan, process, request, tokens, null_tokens, shape, count,
                         selected, samples, posterior, signature)

    def _tokenized(self, rows: list, unconditional) -> tuple[dict, dict]:
        """The conditional and unconditional token fields, one row per prompt.

        `unconditional` None is the task's own blank prompt; a caller's
        negatives are one row or one per prompt, and an encoder that answers
        a different count is refused here.
        """
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
        return tokens, null_tokens

    def _supplied(self, rows: int, shape: tuple[int, ...], *, image, image_latents,
                  mask, noise, initial) -> dict[str, np.ndarray]:
        """The caller's own arrays, checked and broadcast to `rows` rows.

        Pixels arrive at the task's geometry and normalize to [-1, 1];
        latents, noise and an already-noisy state arrive at the latent
        shape, and a mask is thresholded to a float indicator. Which
        combinations are allowed is settled before this runs; nothing is
        drawn or encoded here.
        """
        samples: dict[str, np.ndarray] = {}
        if image is not None:
            pixels = _image_rows(image, rows, self.inputs.sample.shape, "image")
            samples["image"] = np.asarray(unit_range(pixels)) if pixels.dtype == np.uint8 else pixels
        for name, value in (("image_latents", image_latents), ("noise", noise), ("initial", initial)):
            if value is not None:
                samples[name] = _image_rows(value, rows, shape, name)
        if mask is not None:
            value = _image_rows(mask, rows, (*self.inputs.sample.shape[:-1], 1), "mask")
            samples["mask"] = (value >= (128 if value.dtype == np.uint8 else 0.5)).astype(np.float32)
        return samples

    def _encoded(self, settled: _Resolved, *, configured: bool
                 ) -> tuple[dict, dict, jax.Array]:
        """The two encoded conditioning branches and the initial state.

        With no caller-supplied pixels the state is fresh noise. With any,
        the autoencoder runs over them, and a mask adds its spatial
        conditioning to both branches, the unconditional one broadcast to the
        batch first because its single row would not carry the mask.
        """
        plan, process = settled.plan, settled.process
        given = _encode(plan.sharding)(self._conditions, self.params,
                                       plan.place(plan.pad(settled.tokens)))
        null = self._unconditional(settled.null_tokens, plan, given, configured=configured)
        if not settled.samples:
            return given, null, _noise(plan.sharding)(process, plan.keys(settled.request),
                                                      settled.shape)
        start = process.times(settled.count)[0] if settled.times is None else settled.times[0]
        initial_state, spatial = _image_start(plan.sharding)(
            self.autoencoder, process, settled.shape, self.params,
            plan.place(plan.pad(settled.samples)), plan.keys(settled.request),
            settled.posterior, start)
        if spatial:
            given = {**given, **spatial}
            null = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (plan.global_rows, *leaf.shape[1:]))
                                if leaf.shape[0] == 1 else leaf, null)
            null = {**null, **spatial}
        return given, null, initial_state


    def __call__(
        self,
        prompts: str | Sequence[str | Mapping[str, object]] | DenoisingInputs,
        *,
        steps: int | None = None,
        guidance: Guidance | float | None | _Default = _Default.GUIDANCE,
        solver: Solver | None = None,
        key: int | jax.Array | None = None,
        decode: bool = True,
    ) -> Images:
        """Images in [-1, 1], `[rows, H, W, C]`. `guidance` is a classifier-free
        guidance scale, or a `CFG` with its interval, or None for the plain
        conditional prediction; omitted, it is the task's default."""
        mesh = mesh_of(self.params)

        def resolve():
            return self._settings(mesh, prompts, steps=steps, guidance=guidance,
                                  solver=solver, key=key, decode=decode)

        settings = (agreed("image sampling setup", resolve) if mesh is not None else resolve())
        prepared, request, count, process, times, solver, chosen, signature = settings
        if mesh is not None and jax.process_count() > 1:
            multihost_utils.assert_equal(signature, "image execution controls must agree across processes")
        if prepared is None:
            assert not isinstance(prompts, DenoisingInputs)
            prepared = self.prepare(prompts, key=request, steps=count)
        with region("inference.image"):
            assert prepared.rows is not None
            plan = RowPlan.over(mesh, prepared.rows)
            generated = _run(plan.sharding)(self.model, process, self.autoencoder, self.finish, count,
                                         solver, chosen, self.final_denoise, times, decode, self.params,
                                         prepared.conditions, prepared.unconditional,
                                         prepared.noise, jax.random.fold_in(request, 1))
        return replace(generated, rows=plan.rows)


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
        raise ValueError(
            "floating image pixels must be normalized to [-1, 1]; uint8 pixels are also accepted"
        )
    return np.broadcast_to(array, (rows, *shape))


@functools.cache
def _image_start(rows: jax.sharding.NamedSharding | None):
    def prepare(autoencoder, process, shape, params, samples, keys, encode_key, start):
        spatial = {}
        pixels = samples.get("image")
        if "initial" in samples:
            value = samples["initial"]
        else:
            clean = samples.get("image_latents")
            if clean is None:
                if autoencoder is None:
                    clean = pixels
                else:
                    clean = autoencoder.encode(params["autoencoder"], pixels, encode_key)
            noise = samples.get("noise")
            if noise is None:
                noise = jax.vmap(lambda key: jax.random.normal(key, shape))(keys)
            alpha, sigma = process.sampler_schedule.rates(start)
            value = alpha * clean + sigma * noise
        if "mask" in samples:
            from dew.inputs.diffusion import latent_image_conditions

            spatial = latent_image_conditions(
                autoencoder, params["autoencoder"], pixels, samples["mask"], encode_key
            )
        return value, spatial
    return jax.jit(prepare, static_argnums=(0, 1, 2),
                   in_shardings=(None, rows, rows, None, None), out_shardings=rows)


def restore_variables(directory: str, *, ema: bool | None, step: int | str | None, mesh: MeshSpec | None,
                      layout: Layout | None, param_dtype: str | None,
                      parameter_roots: tuple[tuple[str, ...], ...] = (("params",), (FROZEN,))) -> Variables:
    """A run's published variables, restored onto the current mesh under a layout.

    The checkpoint is its own template. Owner-declared parameter roots select
    floating weights for param_dtype; other leaves keep their stored dtype.
    EMA uses the live tree's selection, restricted to the leaves it contains.
    `ema` None takes the averaged weights when the run kept them; True
    requires them.
    """
    from dew.checkpoints import Checkpoints
    from dew.objectives.base import merge
    from dew.training.distributed import Layout as DefaultLayout, MeshSpec as DefaultMesh

    target = resolve_dtype(param_dtype)
    checkpoints = Checkpoints(directory)
    stored = checkpoints.stored(step)
    template = {"params": stored["params"]}
    if ema and stored.get("ema") is None:
        raise ValueError("the run keeps no EMA; request the live policy with ema=False")
    averaged = stored.get("ema") is not None if ema is None else ema
    if averaged:
        template["ema"] = stored["ema"]
    device_mesh = (DefaultMesh() if mesh is None else mesh).build()
    chosen_layout = DefaultLayout() if layout is None else layout
    placement = chosen_layout.shardings(device_mesh, template)
    chosen_layout.check(template["params"], placement["params"], device_mesh)
    selected = set()
    if target is not None:
        roots = tuple(tuple(jax.tree_util.DictKey(name) for name in root) for root in parameter_roots)
        selected = {path for path, leaf in jax.tree_util.tree_flatten_with_path(stored["params"])[0]
                    if jnp.issubdtype(leaf.dtype, jnp.floating) and
                    any(path[:len(root)] == root for root in roots)}
    template = jax.tree_util.tree_map_with_path(
        lambda path, leaf, sharding: jax.ShapeDtypeStruct(
            leaf.shape, target if path[1:] in selected else leaf.dtype, sharding=sharding),
        template, placement)
    values, _ = checkpoints.restore(template, step=step)
    params = values["params"]
    if averaged:
        params = merge(params, values["ema"])

    return params


@functools.cache
def _encode(rows: jax.sharding.NamedSharding | None):
    def encode(conditions, params, tokens):
        return {keyword: encoder.encode(params["encoders"][keyword], tokens[keyword])
                for keyword, encoder in conditions}
    return jax.jit(encode, static_argnums=(0,), in_shardings=(None, rows), out_shardings=rows)


@functools.cache
def _noise(rows: jax.sharding.NamedSharding | None):
    def noise(process, keys, shape):
        return jax.vmap(lambda key: process.noise(key, shape))(keys)
    return jax.jit(noise, static_argnums=(0, 2), in_shardings=(rows,), out_shardings=rows)


@functools.cache
def _run(rows: jax.sharding.NamedSharding | None):
    # Rebinding weights must not change the static compilation identity.
    def run(model, process, autoencoder, finish, steps, solver, guidance, final_denoise, times, decode,
            params, given, null, x_T, key):
        variables = {name: value for name, value in params.items() if name not in ("encoders", "autoencoder")}
        denoise = process.denoiser(model, variables, given, None if guidance is None else null)
        with jax.ensure_compile_time_eval():
            grid = None if times is None else jnp.asarray(times, jnp.float32)
        if grid is None:
            latents = sample(denoise, x_T, steps, solver=solver, guidance=guidance,
                             key=key, final_denoise=final_denoise)
        else:
            latents = sample(denoise, x_T, solver=solver, guidance=guidance,
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
