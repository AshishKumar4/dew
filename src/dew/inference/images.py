"""The text-to-image task: a trained diffusion model, rebuilt from its run, that turns prompts into
images, beside `TextGeneration`. The reverse walk it runs is `dew.sampling`'s (`walk_and_decode`)."""

from __future__ import annotations

import contextlib
import functools
import itertools
import os
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Generic

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.core import freeze
from jax.experimental import multihost_utils
from jax.typing import ArrayLike, DTypeLike
from typing_extensions import TypeVar

from dew._coordination import agreed
from dew._records import JSON
from dew.artifacts import uint8_pixels
from dew.batch import Variables, thaw
from dew.diffusion.process import Conditioning, Process
from dew.inference.scoring import ScoringRequest
from dew.inference.tasks import InItsForm, SavedRun, SavedRunTask, traced_in_its_form
from dew.inputs import InputSpec, encode_conditions, unit_range
from dew.nn.autoencoders import AutoEncoder
from dew.nn.inputs import RowPlan, generation_signature, host_rows, local_rows, mesh_of, request_key
from dew.nn.kernels.options import program
from dew.objectives.base import OMITTED, Omitted
from dew.registry import dtype_name, resolve_dtype
from dew.sampling.guidance import CFG, Guidance
from dew.sampling.solvers import DDIM, Solver
from dew.sampling.walks import sampling_modules, walk_and_decode
from dew.telemetry.profile import region

ArrayT = TypeVar("ArrayT", bound=jax.Array | np.ndarray, default=jax.Array, covariant=True)

if TYPE_CHECKING:
    from jax.sharding import Mesh
    from PIL.Image import Image as PILImage

    from dew.distributed.placement import Layout, MeshSpec
    from dew.objectives.diffusion import DiffusionObjective
    from dew.quantization import Quantization

@dataclass(frozen=True)
class _Resolved:
    """What `prepare` settles on the host before any process runs a model.

    `plan` is this process's rows of the request, `process` and `times` the
    trajectory it walks, `request` the key its noise is drawn from, and
    `posterior` the key a VAE encode samples with. `tokens` and `null_tokens`
    are the two conditioning branches, as tokens to encode or, where
    `encoded` names the branch, already encoded (`null_tokens` None for an
    encoded call with no unconditional branch), `samples` whatever image, mask, noise
    or latent state the caller handed over, and `signature` the value the
    pool compares before any of it reaches a device.
    """

    plan: RowPlan
    process: Process
    request: jax.Array
    tokens: dict
    null_tokens: dict | None
    shape: tuple[int, ...]
    count: int
    times: tuple[float, ...] | None
    samples: dict
    posterior: jax.Array | None
    signature: object
    encoded: frozenset[str] = frozenset()


@struct.dataclass
class DenoisingInputs:
    """Encoded conditioning and initial noise, placed on the devices the way a call uses them.

    `TextToImage.prepare` builds them, and a `TextToImage` call takes them in
    place of prompts. ``rows`` is the number of this process's real prompts. On
    a mesh the arrays hold the padded batch, sharded by row, which a call uses
    as it is. A call refuses inputs prepared for another step count, another
    geometry or another mesh.
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
    """Decoded samples in [-1, 1], NHWC (NTHWC for a video field), placed as the task ran them.

    ``latents`` holds the denoised latents, and ``images`` is None when the
    call ran with ``decode=False``. ``host()`` copies this process's ``rows``
    real rows back to host arrays, and ``pil()`` returns an image batch's real
    rows as 8-bit images.
    """

    images: ArrayT | None
    rows: int | None = struct.field(pytree_node=False, default=None)
    latents: ArrayT | None = None

    def host(self) -> Images[np.ndarray]:
        """Return this process's real rows as host arrays, without the padding added to fill the devices."""
        return jax.tree.map(lambda leaf: host_rows(leaf, self.rows), self)

    def pil(self) -> list[PILImage]:
        """Return this process's real rows of an NHWC image batch as PIL images.

        Three channels give RGB images and one channel gives grayscale. The
        pixels are quantized by `dew.artifacts.uint8_pixels`. Raises ValueError
        when the samples kept only their latents, or when they are not
        `[N, H, W, 3]` or `[N, H, W, 1]` images."""
        from PIL import Image

        if self.images is None:
            raise ValueError("these samples kept only their latents; call the task with decode=True")
        if self.images.ndim != 4 or self.images.shape[-1] not in (1, 3):
            raise ValueError(f"pil() takes [N, H, W, 3] or [N, H, W, 1] images, not samples of shape "
                             f"{self.images.shape}")
        pixels = uint8_pixels(host_rows(self.images, self.rows))
        return [Image.fromarray(row[..., 0] if row.shape[-1] == 1 else row) for row in pixels]


@dataclass(frozen=True)
class ImageSettings:
    """What a text-to-image run records beside its model, and its task is built with
    (`TextToImage.from_objective`, `TextToImage.from_run`): the process, the solver, the
    guidance and the step count, and the records of the input encoders and the autoencoder
    (`dew.registry.to_record`, read back by `InputSpec.bound` and `AutoEncoder.bound`) and an
    end-to-end tuning (`EndToEnd`), which a saved run rebuilds over its own weights at the dtype it
    asks for.
    A run trained from a loaded pipeline samples on that pipeline's own grid, recorded as `grid`
    (`SourceGrid`), and ends as its solver does, with no closing denoise (`TextToImage.grid`)."""

    process: Process
    inputs: JSON
    autoencoder: JSON
    solver: Solver
    guidance: Guidance | None
    sampling_steps: int
    # dewml/hybrid-dit-176m's published record predates the fields below.
    end_to_end: JSON = None
    grid: JSON = None

    def __post_init__(self) -> None:
        from dew._records import integer

        integer(self.sampling_steps, "sampling_steps")


@dataclass(frozen=True, eq=False)
class TextToImage(SavedRunTask, InItsForm):
    """Generates images from text prompts with a trained diffusion model and its encoders.

    Call it as `pipe(prompts, key=0)` or
    `pipe(prompts, steps=40, guidance=4.0, solver=Heun(), key=key)`.

    `variables` is the objective's whole tree, with the EMA copy merged over
    the live weights when the run kept one, so a sample comes from the weights
    the run publishes. `steps`, `guidance` and `solver` are the defaults for a
    call that omits them; an objective or a loaded source sets them. `grid` is
    for a source whose solver pairs its own sigma and model-time tables, and
    returns the process and its explicit time grid for a step count. With
    `final_denoise` False a trajectory ends the way those solvers end it.
    `finish` runs on the decoded images under the same placement, for a source
    that ships a checker or an output transform.

    The weights stay where they are placed. On a mesh, the prompts are split
    into per-process rows over its batch axes, and the result keeps that
    sharding. Each row's initial noise comes from its global row index, so a
    pool of processes draws the same noise as one process does for the same
    prompts.
    """

    model: nn.Module
    process: Process
    inputs: InputSpec
    variables: Variables
    autoencoder: AutoEncoder | None = None
    steps: int = 50
    guidance: Guidance | None = None
    solver: Solver[object] = field(default_factory=DDIM)
    grid: Callable[[int], tuple[Process, jax.Array]] | None = None
    final_denoise: bool = True
    finish: Callable[[Variables, jax.Array], jax.Array] | None = None

    def __post_init__(self) -> None:
        # A tree split for training, an adapter's, is read whole.
        object.__setattr__(self, "variables", freeze(dict(thaw(self.variables))))

    def bind(self, variables: Variables) -> TextToImage:
        """Return this task with `variables` as its weights, its encoders' included: each call
        encodes its prompts and its blank with the towers it holds."""
        return replace(self, variables=variables)

    def quantized(self, spec: Quantization) -> TextToImage:
        """Return this task with its denoiser's weights quantized as `spec` says.

        The denoiser's matmuls then compute with the quantized weights
        (`dew.quantization.quantize_for_serving`). The encoders and
        the autoencoder keep their weights."""
        from dew.quantization import quantize_for_serving

        example = self.prepare("", key=0, steps=1)
        denoiser = {
            name: value for name, value in self.variables.items() if name not in ("encoders", "autoencoder")
        }
        model, variables = quantize_for_serving(self.model, denoiser, spec, example.noise,
                                                jnp.zeros(example.noise.shape[:1]), **example.conditions)
        return replace(self, model=model, variables={**self.variables, **variables})

    @classmethod
    def from_objective(cls, objective: DiffusionObjective, variables: Variables) -> TextToImage:
        """Build a task over the objective's model and `variables` that samples the way its evaluation does.

        A loss-only head the objective trains is dropped."""
        from dew.objectives.diffusion.objective import without_loss_heads

        autoencoder, variables = objective.published_autoencoder(variables)
        # The live encoders and autoencoder need no record (`task_settings` writes theirs for a save).
        return cls(objective.model, objective.process, objective.inputs,
                   without_loss_heads(variables), autoencoder,
                   steps=objective.steps, guidance=objective.guidance, solver=objective.solver,
                   grid=objective.grid, final_denoise=objective.final_denoise)

    @classmethod
    def from_saved(cls, run: SavedRun, *, ema: bool | None = None, mesh: MeshSpec | Mesh | None = None,
                   layout: Layout | None = None, dtype: DTypeLike | None = None,
                   param_dtype: DTypeLike | None = None, variables: Variables | None = None) -> TextToImage:
        """Load the run `run` (its checkpoint pinned, `SavedRun.pinned`), built from the inference record
        its checkpoint holds (`Checkpoints.artifact`) the way the recipe built it; the run's `run.json`
        is not read. `from_run` loads one from its directory.

        `ema` None reads the averaged weights when the run kept them and the
        live ones when it kept none; True requires the averaged weights, and
        False reads the live ones. A run whose averaged weights are a reference
        policy, such as Flow-GRPO's frozen KL reference, always reads its live
        policy. With `mesh`, the weights are restored directly onto that mesh
        under `layout`, the way the trainer places them. Without it, they go
        onto a default `MeshSpec()` over the current pool's devices.

        `dtype` sets the compute dtype of the model, the encoders and the VAE.
        `param_dtype` sets the dtype the parameters are stored in; None keeps
        the checkpoint's dtypes exactly. Supplied `variables`, a tree of the
        run's own form (a run that started from it saved them), are bound in
        place of the checkpoint's, which is not read.
        """
        from dew._records import record as fields
        from dew.diffusion.schedules.source import SourceGrid
        from dew.inference.tasks import saved_model, saved_settings
        from dew.nn.autoencoders import AutoEncoder
        from dew.objectives.diffusion.objective import without_loss_heads
        from dew.registry import from_record

        settings = saved_settings(run.record, ImageSettings)
        compute = dtype_name(resolve_dtype(dtype))
        config = saved_model(run.record, compute)
        inputs_record = fields(settings.inputs, 'inputs')
        autoencoder_record = None if settings.autoencoder is None else fields(settings.autoencoder,
                                                                              'autoencoder')
        if compute is not None:
            conditions = {keyword: fields(condition, keyword) for keyword, condition
                          in fields(inputs_record['conditions'], 'conditions').items()}
            inputs_record = {**inputs_record, 'conditions': {
                keyword: {**condition,
                          'encoder': _computing(fields(condition['encoder'], 'encoder'), compute)}
                for keyword, condition in conditions.items()}}
            if autoencoder_record is not None:
                autoencoder_record = _computing(autoencoder_record, compute)
        params = (restored_tree(run, ema=ema, mesh=mesh, layout=layout, param_dtype=param_dtype)
                  if variables is None else variables)
        inputs = InputSpec.bound(inputs_record, params=params.get('encoders', {}))
        end_to_end = settings.end_to_end
        autoencoder = None
        if autoencoder_record is not None and end_to_end is not None:
            from dew.objectives.diffusion.end_to_end import AUTOENCODER, EndToEnd
            tuning = from_record(EndToEnd, fields(end_to_end, 'end_to_end'), dtypes=False)
            frozen = AutoEncoder.bound(autoencoder_record, params=params['params'][AUTOENCODER])
            autoencoder, params = tuning.tuned(frozen, params)
        elif autoencoder_record is not None:
            autoencoder = AutoEncoder.bound(autoencoder_record, params=params['autoencoder'])
        grid = None if settings.grid is None else from_record(SourceGrid, fields(settings.grid, 'grid'))
        return cls(config.build(), settings.process, inputs, without_loss_heads(params), autoencoder,
                   steps=settings.sampling_steps, guidance=settings.guidance, solver=settings.solver,
                   grid=grid, final_denoise=grid is None)

    @classmethod
    def from_flaxdiff(cls, directory: str | os.PathLike, config: Mapping[str, object], *, jax_version: str,
                      ema: bool = True, best: bool = False, dtype: DTypeLike | None = None) -> TextToImage:
        """Load a FlaxDiff text-to-image run (`simple_udit` or `hybrid_dit` on the SD VAE) into Dew's model.

        `directory` is one checkpoint step, `config` the run config FlaxDiff's
        trainer logged, and `jax_version` the jax version the run trained
        under, from its `requirements.txt`. `ema` and `best` pick the weights;
        `dtype` is the model's compute dtype. `dew.interop.flaxdiff` reads the
        format.
        """
        from dew.interop import flaxdiff

        return flaxdiff.text_to_image(directory, config, jax_version=jax_version, ema=ema, best=best,
                                      dtype=dtype)

    def prepared_process(self, steps: int) -> tuple[Process, tuple[float, ...] | None]:
        """Return the process and explicit time grid that a call with `steps` steps runs.

        The grid is a tuple of concrete times, so the compiled trajectory has a
        fixed length and fixed values. It is None when the task has no `grid`.
        Raises ValueError unless `steps` is a positive int."""
        if type(steps) is not int or steps < 1:
            raise ValueError("steps must be a positive integer")
        if self.grid is None:
            return self.process, None
        process, times = self.grid(steps)
        return process, _time_grid(times)

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """The per-example shape the model denoises.

        It is the sample field's shape, or that field's latent shape when the
        task has an autoencoder in front of the model."""
        shape = self.inputs.sample.shape
        return shape if self.autoencoder is None else self.autoencoder.latent_shape(shape)

    def _unconditional(self, tokens, plan: RowPlan) -> dict:
        """The unconditional branch: the task's own blank prompt or the caller's negatives, encoded
        by the task's towers as the training step encodes its blank (`DiffusionObjective.encode`)."""
        leaves = jax.tree.leaves(tokens)
        if leaves and leaves[0].shape[0] != 1:
            encode = _encoder(plan.sharding, self.inputs.encoders)
            return encode(self.inputs.encoders, self.variables, plan.place(plan.pad(tokens)))
        # Host tokens, which reach the encoders' devices as the call places them.
        return _encoder(None, self.inputs.encoders)(self.inputs.encoders, self.variables,
                                                    jax.tree.map(np.asarray, tokens))


    def prepare(
        self,
        prompts: str | Sequence[str | Mapping[str, object]] | None = None,
        *,
        conditions: Mapping[str, Conditioning] | None = None,
        key: int | jax.Array,
        steps: int | None = None,
        unconditional: str | Sequence[str | Mapping[str, object]] | Mapping[str, Conditioning] | None = None,
        image: ArrayLike | None = None,
        image_latents: ArrayLike | None = None,
        mask: ArrayLike | None = None,
        noise: ArrayLike | None = None,
        initial: ArrayLike | None = None,
        times: ArrayLike | Sequence[float] | None = None,
        encode_key: int | jax.Array | None = None,
    ) -> DenoisingInputs:
        """Encode the conditions and build the initial state for a call, on a concrete grid.

        Pass either `prompts` or `conditions`. The result is `DenoisingInputs`,
        which a call takes in place of prompts. Each array argument has one row
        or one row per sample. The optional inputs:

        - `image`: uint8 or normalized floating NHWC pixels at the task's
          geometry.
        - `image_latents`: the image already encoded, which skips VAE encoding.
        - `mask`: spatial conditioning added to both guidance branches; it
          needs `image` and an autoencoder.
        - `noise`: unit Gaussian noise for noising a clean image.
        - `initial`: an already-noisy latent state for a continuation or
          refiner handoff, which is never noised again.
        - `times`: an explicit grid that selects a partial trajectory in the
          prepared process.
        - `encode_key`: the key that samples a VAE posterior; None uses its
          mean.

        `conditions` are prompts already encoded, `{keyword: condition}` with
        one row per sample. A pipeline's conditioner loaded on its own can
        encode them, so the text encoder does not have to share the device with
        the denoiser (`load_diffusion_source(text=False)`). `unconditional`
        encoded the same way, one row or one per sample, is the branch guidance
        reads. It can also be negative prompts as text, which the text encoder
        encodes. Without it, the task's own blank prompt is encoded if the text
        encoder is loaded. If it is not loaded, the result has no unconditional
        branch, and a call with it must pass `guidance=None`. Each row's noise
        is the noise a call with prompts draws.

        Raises ValueError for a combination the task cannot run, such as
        `image` together with `image_latents`, or `noise` without a clean
        image.
        """
        mesh = mesh_of(self.variables)

        def resolve() -> _Resolved:
            count = self.steps if steps is None else steps
            process, source_times = self.prepared_process(count)
            return self._resolved(mesh, prompts, conditions, request=request_key(key), count=count,
                                  process=process,
                                  selected=_time_grid(times) if times is not None else source_times,
                                  unconditional=unconditional, image=image, image_latents=image_latents,
                                  mask=mask, noise=noise, initial=initial, encode_key=encode_key)

        settled = (agreed("image input preparation", resolve) if mesh is not None else resolve())
        if settled.plan.processes > 1:
            multihost_utils.assert_equal(settled.signature,
                                         "image input shapes and sampling must agree across processes")
        return self._inputs(settled, owns_grid=self.grid is not None or times is not None)

    @traced_in_its_form
    def _inputs(self, settled: _Resolved, *, owns_grid: bool) -> DenoisingInputs:
        """The conditions encoded and the initial state built for `settled`, placed for its rows. A grid
        the task or the caller chose (`owns_grid`) travels with them."""
        with region("inference.image.prepare"):
            given, null, initial_state = self._encoded(settled)
        return DenoisingInputs(initial_state, given, null, rows=settled.plan.rows,
                               grid_steps=settled.count if owns_grid else None,
                               process=settled.process if owns_grid else None, times=settled.times)

    def _guidance(self, guidance: Guidance | float | None | Omitted) -> Guidance | None:
        """The guidance a call runs: the task's where it is omitted, a scale as `CFG`'s."""
        chosen = self.guidance if guidance is OMITTED else guidance
        if isinstance(chosen, (int, float)) and not isinstance(chosen, bool):
            chosen = CFG(float(chosen))
        if chosen is not None and not isinstance(chosen, Guidance):
            raise ValueError("guidance must be a scale, None, or a guidance (`dew.sampling.Guidance`: its "
                             "walk, and the modules it runs besides the denoiser, `modules()`)")
        return chosen

    def scoring(self, *, steps: int | None = None, guidance: Guidance | float | None | Omitted = OMITTED,
                solver: Solver | None = None, decode: bool = True, images: int = 8
                ) -> tuple[ImageScoring, Variables]:
        """This task as a scoring task of `ImageRequest`s (`dew.inference.scoring.Scoring`), which a
        `ScoringServer` serves beside the other kinds of an `Engine`, and the weights it samples with.
        `steps`, `guidance` and `solver` are a request's where it names none, the task's where these are
        omitted. A request is answered with the `Images` its lone call `task(prompts, key=key, ...)` draws
        with its settings: its noise and its walk's draws are its key's, whatever requests share its pass.
        A pass holds requests alike (as many prompts and negatives, the same settings) up to `images`
        images, and walks them as one program mapped over the requests, each request's walk its own. The
        scoring task holds no weights: a server holds them resident and swaps both whole on a reload
        (`ScoringServer.reload`)."""
        if type(decode) is not bool:
            raise ValueError("decode must be a boolean")
        if type(images) is not int or images < 1:
            raise ValueError(f"a pass holds an image or more, not {images}")
        self._text_encoder("a served request's prompts are encoded with it")
        held = ImageScoring(replace(self, variables={}), self.steps if steps is None else steps,
                            self._guidance(guidance), self.solver if solver is None else solver, decode,
                            images)
        return held, self.variables

    @traced_in_its_form
    def _served(self, requests: Sequence[_Drawn]) -> list[Images]:
        """One served pass over `requests`, alike (`_Drawn.alike`): each request's conditions and noise
        prepared as its lone call prepares them, and the walks run as one program mapped over the
        requests, each with its own key (`TextToImage.scoring`)."""
        mesh = mesh_of(self.variables)
        walking = requests[0].walking

        def inputs(request: _Drawn) -> DenoisingInputs:
            """The request's noise and conditions as `prepare` prepares them for its own key and steps."""
            process, times = self.prepared_process(request.preparing)
            return self._inputs(self._resolved(
                mesh, list(request.prompts), None, request=request.prepared, count=request.preparing,
                process=process, selected=times,
                unconditional=None if request.negative is None else list(request.negative), image=None,
                image_latents=None, mask=None, noise=None, initial=None, encode_key=None),
                owns_grid=self.grid is not None)

        prepared = [inputs(request) for request in requests]
        with _streams(requests, [inputs.rows for inputs in prepared]) as streams:
            return self._walked(prepared, walking, [request.key for request in requests], streams)

    def _walked(self, prepared: Sequence[DenoisingInputs], walking: _Walking, requests: Sequence[jax.Array],
                streams: jax.Array | None = None) -> list[Images]:
        """The walks of `prepared` inputs under `walking`, each drawing its request's: the solver's state
        built on the grid, then the walk, each step's clean prediction sent to its request's stream where
        `streams` names them (`ImageRequest.on_step`). One request walks through a lone call's own
        programs, so a served request alone draws its lone call's pixels; several through those programs
        mapped over the requests (`_initials`, `_walks`), the weights and the grid shared."""
        walks = [jax.random.fold_in(request, 1) for request in requests]
        grid = _default_grid(walking.process, walking.count) if walking.times is None else walking.times
        models = sampling_modules(self.model, walking.guidance, self.autoencoder)
        statics = (self.model, walking.process, self.autoencoder, self.finish, self.final_denoise,
                   walking.decode)
        times = np.asarray(grid, np.float32)
        with region("inference.image"):
            if len(prepared) == 1:
                (inputs,), (walk,) = prepared, walks
                assert inputs.rows is not None
                plan = RowPlan.over(mesh_of(self.variables), inputs.rows)
                initial = program(_initial, static_argnums=(0, 1, 2), in_shardings=(plan.sharding, None))(
                    walking.solver, walking.process, grid, inputs.noise, walk)
                walked = program(_walk, *models, static_argnums=(0, 1, 2, 3, 4, 5),
                                 in_shardings=(None, None, None, None, None, plan.sharding, None,
                                               plan.sharding, None, None),
                                 out_shardings=plan.sharding)
                generated = walked(*statics, walking.solver, walking.guidance, times, initial, self.variables,
                                   inputs.conditions, inputs.unconditional, inputs.noise, walk,
                                   None if streams is None else streams[0])
                return [replace(generated, rows=plan.rows)]
            noise, conditions, unconditional = (jax.tree.map(lambda *leaves: jnp.stack(leaves), *branch)
                                                for branch in zip(*((inputs.noise, dict(inputs.conditions),
                                                                     dict(inputs.unconditional))
                                                                    for inputs in prepared), strict=True))
            keys = jnp.stack(walks)
            initial = program(_initials, static_argnums=(0, 1, 2))(walking.solver, walking.process, grid,
                                                                   noise, keys)
            generated = program(_walks, *models, static_argnums=(0, 1, 2, 3, 4, 5))(
                *statics, walking.solver, walking.guidance, times, initial, self.variables, conditions,
                unconditional, noise, keys, streams)
        return [replace(jax.tree.map(lambda leaf, index=index: leaf[index], generated), rows=inputs.rows)
                for index, inputs in enumerate(prepared)]

    def _settings(self, mesh, prompts, *, steps, guidance, solver, key, decode):
        """Everything one call settles on the host before it runs the model.

        A caller who hands over `DenoisingInputs` gets them checked against
        this task's geometry and this mesh here; everything else resolves
        the grid, the solver and the guidance a call runs with. The value
        ends with the signature a pool compares.
        """
        chosen = self._guidance(guidance)
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
            if chosen is not None and prepared.conditions and not prepared.unconditional:
                raise ValueError("guidance reads the unconditional branch, which these inputs lack; prepare "
                                 "them with an encoded unconditional= too, or call with guidance=None")
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

    def _resolved(self, mesh, prompts, conditions, *, request, count, process, selected, unconditional,
                  image, image_latents, mask, noise, initial, encode_key) -> _Resolved:
        """Everything `prepare` settles on the host, in one value, over the request key, step count,
        process and grid its caller resolved once.

        This is the half a pool has to agree on: every refusal a caller can
        earn is raised here, and the signature at the end is what the ranks
        compare before any of them touches a device.
        """
        given: list | Mapping[str, Conditioning]
        if prompts is not None and conditions is None:
            given = [prompts] if isinstance(prompts, str) else list(prompts)
            if not given or not all(isinstance(prompt, (str, Mapping)) for prompt in given):
                raise ValueError("prompts must be a non-empty sequence of strings or conditioning records")
            samples_count = len(given)
        elif conditions is not None and prompts is None:
            given, samples_count = conditions, _encoded_rows(conditions)
        else:
            raise ValueError("pass the prompts or their encoded conditions, one of the two")
        plan = RowPlan.over(mesh, samples_count)
        tokens, null_tokens, encoded = self._branches(given, unconditional, samples_count)
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
        samples = self._supplied(samples_count, shape, image=image, image_latents=image_latents,
                                 mask=mask, noise=noise, initial=initial)
        controls = (plan.rows, count, selected, shape,
                    tuple(jax.device_get(jax.random.key_data(request))),
                    None if posterior is None else tuple(np.asarray(jax.random.key_data(posterior))))
        signature = generation_signature((tokens, null_tokens, samples), controls)
        return _Resolved(plan, process, request, tokens, null_tokens, shape, count,
                         selected, samples, posterior, signature, encoded)

    def _branches(self, given: list | Mapping[str, Conditioning], unconditional,
                  count: int) -> tuple[dict, dict | None, frozenset[str]]:
        """The conditional and unconditional branches, one row per sample, as
        tokens the text encoder encodes or as the encodings a caller passed,
        which the returned set names ("given", "null"). `given` is the prompt
        rows, or the encodings `{keyword: condition}`.

        Negatives are one row or one per sample. `unconditional` None is the
        task's own blank prompt, encoded where the text encoder is loaded; an
        encoded call without the text encoder then has no unconditional
        branch (None), which only an unguided call takes.
        """
        encoded = set()
        if isinstance(given, Mapping):
            tokens = self._encodings(given, (count,), "conditions")
            encoded.add("given")
        else:
            self._text_encoder("a prompt is encoded with it")
            tokens = {keyword: condition.encoder.tokenize(given)
                      for keyword, condition in self.inputs.conditions.items()}
            for leaf in jax.tree.leaves(tokens):
                if leaf.ndim < 1 or leaf.shape[0] != count:
                    raise ValueError("tokenized conditions must have one row per prompt")
        if isinstance(unconditional, Mapping):
            null = self._encodings(unconditional, (1, count), "unconditional")
            return tokens, null, frozenset({*encoded, "null"})
        if unconditional is None and isinstance(given, Mapping) and not self._text_held():
            return tokens, None, frozenset(encoded)
        self._text_encoder("an unconditional prompt is encoded with it")
        negatives = None
        if unconditional is not None:
            negatives = [unconditional] if isinstance(unconditional, str) else list(unconditional)
            if len(negatives) not in (1, count):
                raise ValueError("unconditional inputs need one row or one row per prompt")
        null_tokens = {keyword: condition.encoder.tokenize(
            [condition.unconditional] if negatives is None else negatives)
            for keyword, condition in self.inputs.conditions.items()}
        for leaf in jax.tree.leaves(null_tokens):
            if leaf.ndim < 1 or leaf.shape[0] not in (1, count):
                raise ValueError("unconditional tokens must have one row or one per prompt")
        return tokens, null_tokens, frozenset(encoded)

    def _text_held(self) -> bool:
        """Whether every condition's encoder has its weights in this task."""
        held = self.variables.get("encoders", {})
        return all(keyword in held for keyword in self.inputs.conditions)

    def _text_encoder(self, needed: str) -> None:
        if not self._text_held():
            raise ValueError(
                f"this task's text encoder is not loaded (load_diffusion_source(text=False)), and {needed}; "
                "encode the prompts with the pipeline's conditioner alone and pass "
                "prepare(conditions=..., unconditional=...)")

    def _encodings(self, given: Mapping[str, Conditioning], rows: tuple[int, ...], name: str) -> dict:
        """A caller's encoded branch, checked against the task's conditions."""
        if set(given) != set(self.inputs.conditions):
            raise ValueError(f"encoded {name} are keyed by {sorted(self.inputs.conditions)}, "
                             f"not {sorted(given)}")
        for leaf in jax.tree.leaves(dict(given)):
            if leaf.ndim < 1 or leaf.shape[0] not in rows:
                raise ValueError(f"encoded {name} need {' or '.join(map(str, rows))} rows, got {leaf.shape}")
        return dict(given)

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

    @traced_in_its_form
    def _encoded(self, settled: _Resolved) -> tuple[dict, dict, jax.Array]:
        """The two encoded conditioning branches and the initial state.

        With no caller-supplied pixels the state is fresh noise. With any,
        the autoencoder runs over them, and a mask adds its spatial
        conditioning to both branches, the unconditional one broadcast to the
        batch first because its single row would not carry the mask.
        """
        plan, process = settled.plan, settled.process
        given = plan.place(plan.pad(settled.tokens))
        if "given" not in settled.encoded:
            given = _encoder(plan.sharding, self.inputs.encoders)(self.inputs.encoders, self.variables, given)
        if settled.null_tokens is None:
            null = {}
        elif "null" in settled.encoded:
            single = jax.tree.leaves(settled.null_tokens)[0].shape[0] == 1
            null = (jax.device_put(settled.null_tokens) if single
                    else plan.place(plan.pad(settled.null_tokens)))
        else:
            null = self._unconditional(settled.null_tokens, plan)
        if not settled.samples:
            noise = program(_noised, static_argnums=(0, 2), in_shardings=(plan.sharding,),
                            out_shardings=plan.sharding)
            return given, null, noise(process, plan.keys(settled.request), settled.shape)
        start = process.times(settled.count)[0] if settled.times is None else settled.times[0]
        starting = program(_image_start, *(() if self.autoencoder is None else self.autoencoder.towers()),
                           static_argnums=(0, 1, 2),
                           in_shardings=(None, plan.sharding, plan.sharding, None, None),
                           out_shardings=plan.sharding)
        initial_state, spatial = starting(
            self.autoencoder, process, settled.shape, self.variables,
            plan.place(plan.pad(settled.samples)), plan.keys(settled.request),
            settled.posterior, start)
        if spatial:
            given = {**given, **spatial}
            null = jax.tree.map(lambda leaf: jnp.broadcast_to(leaf, (plan.global_rows, *leaf.shape[1:]))
                                if leaf.shape[0] == 1 else leaf, null)
            null = {**null, **spatial}
        return given, null, initial_state


    @traced_in_its_form
    def __call__(
        self,
        prompts: str | Sequence[str | Mapping[str, object]] | DenoisingInputs,
        *,
        steps: int | None = None,
        guidance: Guidance | float | None | Omitted = OMITTED,
        solver: Solver | None = None,
        key: int | jax.Array,
        decode: bool = True,
    ) -> Images:
        """Generate images for `prompts`, as `Images` in [-1, 1] of shape `[rows, H, W, C]`.

        `prompts` can also be `DenoisingInputs` from `prepare`. `guidance` is a
        classifier-free guidance scale, a `CFG` with its interval, or None for
        the plain conditional prediction; when it is omitted, the task's
        default applies, and so do the task's `steps` and `solver`. With
        `decode=False` the result holds only the latents."""
        mesh = mesh_of(self.variables)

        def resolve():
            settings = self._settings(mesh, prompts, steps=steps, guidance=guidance,
                                      solver=solver, key=key, decode=decode)
            prepared, request, count, process, times = settings[:5]
            # Prompts are prepared over the request, count, process and grid just resolved.
            settled = None if prepared is not None else self._resolved(
                mesh, prompts, None, request=request, count=count, process=process, selected=times,
                unconditional=None, image=None, image_latents=None, mask=None, noise=None, initial=None,
                encode_key=None)
            return settings, settled

        settings, settled = (agreed("image sampling setup", resolve) if mesh is not None else resolve())
        prepared, request, count, process, times, solver, chosen, signature = settings
        if mesh is not None and jax.process_count() > 1:
            # The controls carry the request's kind, so once they agree every rank enters the same
            # collectives: the prompts' preparation is compared only where every rank resolved one.
            multihost_utils.assert_equal(signature, "image execution controls must agree across processes")
            if settled is not None:
                multihost_utils.assert_equal(settled.signature,
                                             "image input shapes and sampling must agree across processes")
        if prepared is None:
            assert settled is not None, "prompts are resolved in the agreement above"
            prepared = self._inputs(settled, owns_grid=self.grid is not None)
        return self._walked([prepared], _Walking(count, process, times, chosen, solver, decode), [request])[0]


@dataclass(frozen=True)
class ImageRequest:
    """One request a served text-to-image task answers (`TextToImage.scoring`): its prompts, an image
    each, the key its draws come from, and the settings of its lone call `task(prompts, key=key, ...)`:
    `steps`, `guidance` and `solver` (the scoring task's where omitted), and `negative`, its negative
    prompts, one or one a prompt, in place of the task's blank. A request whose noise and conditions are
    prepared apart from its walk, as `task(task.prepare(prompts, key=prepare_key, steps=prepare_steps,
    unconditional=negative), key=key, steps=steps, ...)` prepares them, names `prepare_key` and
    `prepare_steps` (its `key` and `steps` where None). `on_step`, where given, streams the walk: it is
    called on the runtime's callback thread with each solver step's number, from 1, and the clean
    prediction that step reads, the request's rows as a host array, as the walk runs."""

    prompts: str | Sequence[str]
    key: int | jax.Array
    steps: int | None = None
    guidance: Guidance | float | None | Omitted = OMITTED
    solver: Solver | None = None
    negative: str | Sequence[str] | None = None
    prepare_key: int | jax.Array | None = None
    prepare_steps: int | None = None
    on_step: Callable[[int, np.ndarray], None] | None = None


_STREAMS: dict[int, Callable[[np.ndarray], None]] = {}
"""Each served request's step stream that a walk running now feeds (`_streams`), by its number."""
_STREAMING = threading.Lock()
_STREAMED = itertools.count()


@contextlib.contextmanager
def _streams(requests: Sequence[_Drawn], rows: Sequence[int | None]) -> Iterator[jax.Array | None]:
    """The stream numbers of a served pass's requests, -1 for a request that streams none, each request's
    `on_step` registered while the pass runs with its real `rows` alone, its steps counted from 1; None
    where none streams."""
    if all(request.on_step is None for request in requests):
        yield None
        return
    numbers = [-1 if request.on_step is None else next(_STREAMED) for request in requests]

    def counted(on_step: Callable[[int, np.ndarray], None], real: int | None) -> Callable[[np.ndarray], None]:
        steps = itertools.count(1)
        return lambda held: on_step(next(steps), held[:real])

    with _STREAMING:
        _STREAMS.update({number: counted(request.on_step, real)
                         for number, request, real in zip(numbers, requests, rows, strict=True)
                         if request.on_step is not None})
    try:
        yield jnp.asarray(numbers, jnp.int32)
        # Every step's callback has run once the pass's effects have, its stream whole before its answer.
        jax.effects_barrier()
    finally:
        with _STREAMING:
            for number in numbers:
                _STREAMS.pop(number, None)


def _streamed(numbers: np.ndarray, denoised: np.ndarray) -> None:
    """A served step's clean predictions to their requests' streams: one request's, or a pass's mapped
    over its requests, as the runtime calls a callback under `vmap`."""
    numbers, denoised = np.asarray(numbers), np.asarray(denoised)
    if numbers.ndim == 0:
        numbers, denoised = numbers[None], denoised[None]
    with _STREAMING:
        streams = [_STREAMS.get(int(number)) for number in numbers]
    for stream, rows in zip(streams, denoised, strict=True):
        if stream is not None:
            stream(rows)


@dataclass(frozen=True)
class _Walking:
    """What one served pass walks with, which every request in it shares."""

    count: int
    process: Process
    times: tuple[float, ...] | None
    guidance: Guidance | None
    solver: Solver
    decode: bool


@dataclass(frozen=True, eq=False)
class _Drawn:
    """An admitted request: its prompts, negatives, the typed keys its noise is prepared and its walk
    draws under, the step count it is prepared for and its walk's settings."""

    prompts: tuple[str, ...]
    negative: tuple[str, ...] | None
    prepared: jax.Array
    key: jax.Array
    preparing: int
    walking: _Walking
    on_step: Callable[[int, np.ndarray], None] | None = None

    def alike(self, other: _Drawn) -> bool:
        """Whether `other` walks in one program beside this one: as many prompts and negatives, and the
        same settings."""
        return (len(self.prompts) == len(other.prompts) and self.walking == other.walking
                and (self.negative is None) == (other.negative is None)
                and (self.negative is None or len(self.negative) == len(other.negative or ())))


@dataclass
class _ImagePass:
    """A served pass of requests alike its first, up to `images` images (`PassPlan`)."""

    images: int
    first: _Drawn | None = None
    held: int = 0

    def add(self, request: _Drawn) -> bool:
        if self.first is not None and (not self.first.alike(request)
                                       or self.held + len(request.prompts) > self.images):
            return False
        self.first = self.first or request
        self.held += len(request.prompts)
        return True


@dataclass(frozen=True, eq=False)
class ImageScoring:
    """A text-to-image task served by a `ScoringServer` (`TextToImage.scoring`): each `ImageRequest` one
    row, answered with its `Images`, every pass over the weights its server hands it, with the settings
    a request names none of."""

    task: TextToImage
    steps: int
    guidance: Guidance | None
    solver: Solver
    decode: bool
    images: int

    def admit(self, request: ImageRequest) -> ScoringRequest[_Drawn, None]:
        task = self.task
        prompts = (request.prompts,) if isinstance(request.prompts, str) else tuple(request.prompts)
        negative = (None if request.negative is None else (request.negative,)
                    if isinstance(request.negative, str) else tuple(request.negative))
        if not prompts or not all(isinstance(prompt, str) for prompt in (*prompts, *(negative or ()))):
            raise ValueError("an image request holds a prompt or more, each a string, and its negatives too")
        if negative is not None and len(negative) not in (1, len(prompts)):
            raise ValueError("an image request's negatives are one or one a prompt")
        count = self.steps if request.steps is None else request.steps
        preparing = count if request.prepare_steps is None else request.prepare_steps
        if task.grid is not None and preparing != count:
            raise ValueError("prepared noise belongs to a different source grid; prepare it for these steps")
        process, times = task.prepared_process(count)
        task.prepared_process(preparing)
        walking = _Walking(count, process, times,
                           self.guidance if request.guidance is OMITTED else task._guidance(request.guidance),
                           self.solver if request.solver is None else request.solver, self.decode)
        key = request_key(request.key)
        prepared = key if request.prepare_key is None else request_key(request.prepare_key)
        drawn = _Drawn(prompts, negative, prepared, key, preparing, walking, request.on_step)
        return ScoringRequest((drawn,), None)

    def planned(self) -> _ImagePass:
        return _ImagePass(self.images)

    def forward(self, variables: Variables, rows: Sequence[_Drawn]) -> list[Images]:
        # A walk places its rows over the weights' mesh itself, as a lone call does, outside the server's
        # mesh context, under which its eager key folds would take the whole mesh for a key's one device.
        with jax.set_mesh(None):
            return self.task.bind(variables)._served(rows)

    def compile(self, variables: Variables) -> Sequence[object]:
        """Nothing up front: a pass's programs compile at its first settings and count of requests and
        prompts, as a lone call's do at its first settings and count of prompts."""
        return ()

    def answer(self, payload: None, outputs: Sequence[Images]) -> Images:
        (images,) = outputs
        return images


def restored_tree(run: SavedRun, *, ema: bool | None, mesh: MeshSpec | Mesh | None, layout: Layout | None,
                  param_dtype: DTypeLike | None) -> Variables:
    """The diffusion run `run`'s tree as its checkpoint holds it, an adapter's split of it kept
    (`SavedRun.variables`), each parameter root its record names (`_parameter_roots`) stored in
    `param_dtype` (as saved where None), restored where `mesh` and `layout` place it:
    what `TextToImage.from_saved` binds, and what a run that starts from `run` trains
    (`dew.interop.saved_sources.pipeline_of_run`)."""
    from dew._records import record as fields
    from dew.inference.tasks import saved_settings

    settings = saved_settings(run.record, ImageSettings)
    autoencoder = None if settings.autoencoder is None else fields(settings.autoencoder, 'autoencoder')
    return run.variables(ema=ema, mesh=mesh, layout=layout, param_dtype=param_dtype,
                         parameter_roots=_parameter_roots(fields(settings.inputs, 'inputs'), autoencoder))


def _parameter_roots(inputs: Mapping[str, object], autoencoder: Mapping[str, object] | None
                     ) -> tuple[tuple[str, ...], ...]:
    """The parameter roots of a recorded run's tree, which a storage override
    casts: the denoiser's, each condition encoder's own collections and the
    autoencoder's. The rest keep the dtypes they were saved in."""
    from dew._records import record
    from dew.batch import PARAMETER_ROOTS
    from dew.inputs.condition import ConditionEncoder
    from dew.registry import component

    roots: list[tuple[str, ...]] = list(PARAMETER_ROOTS)
    for keyword, condition in record(inputs['conditions'], 'conditions').items():
        encoder, _ = component(ConditionEncoder, record(record(condition, keyword)['encoder'], 'encoder'))
        collections = encoder.parameter_collections
        roots.extend([("encoders", keyword)] if collections is None else
                     [("encoders", keyword, collection) for collection in collections])
    if autoencoder is not None:
        roots.append(("autoencoder",))
    return tuple(roots)


def _computing(owner: Mapping[str, object], compute: str) -> Mapping[str, object]:
    """A recorded encoder or autoencoder (a class record, `dew.registry.component`) computing in
    `compute`: its own `dtype`, and the dtype of the model it wraps where it records one."""
    from dew._records import record

    recorded = dict(record(owner['fields'], 'fields'))
    if 'dtype' in recorded:
        recorded['dtype'] = compute
    model = recorded.get('model')
    if isinstance(model, Mapping) and 'dtype' in record(model.get('fields', {}), 'model.fields'):
        recorded['model'] = {**model, 'fields': {**record(model['fields'], 'model.fields'), 'dtype': compute}}
    return {**owner, 'fields': recorded}


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


def _image_start(autoencoder, process, shape, params, samples, keys, encode_key, start):
    """The image a walk starts from, and the spatial conditions its mask adds: the given initial
    latents, or the clean image (encoded by `autoencoder` where there is one) noised to `start`.
    Compiled over the rows with the autoencoder's own XLA options (`program`)."""
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




def _encoded(encoders, params, tokens):
    """The conditions' encodings of `tokens` (`encode_conditions`), each row's where its rows sit."""
    # A task without conditions holds no encoder weights.
    return encode_conditions(encoders, params["encoders"] if encoders else {}, tokens)


def _encoder(rows: jax.sharding.NamedSharding | None, encoders):
    """The program that encodes rows placed by `rows` (`_encoded`) with `encoders`, placed as they are,
    under the options its towers compile with (`compiler_options`)."""
    return program(_encoded, encoders, static_argnums=(0,), in_shardings=(None, rows), out_shardings=rows)


def _encoded_rows(conditions: Mapping[str, Conditioning]) -> int:
    """The samples an encoded branch holds: its leaves' common leading axis."""
    counts = {leaf.shape[0] for leaf in jax.tree.leaves(dict(conditions)) if leaf.ndim}
    if len(counts) != 1:
        raise ValueError(f"encoded conditions need one row count across their leaves, got {sorted(counts)}")
    return counts.pop()


def _noised(process, keys, shape):
    """Each row's starting noise of `shape`, drawn under its own key."""
    return jax.vmap(lambda key: process.noise(key, shape))(keys)


@functools.lru_cache(maxsize=64)
def _default_grid(process: Process, count: int) -> tuple[float, ...]:
    """The process's own grid of `count` steps, read once per process and count."""
    return _time_grid(process.times(count))


def _initial(solver, process, grid, x_T, key):
    """The solver's state before a walk's first step, on the concrete `grid`: its checks and tables
    read the grid's values, so this small program is a grid's own and the walk's is every grid's."""
    with jax.ensure_compile_time_eval():
        times = jnp.asarray(grid, jnp.float32)
    return solver.init(x_T, times, process, key=key)


def _initials(solver, process, grid, x_T, keys):
    """`_initial` of each request's rows, mapped over the requests (`TextToImage._served`)."""
    return jax.vmap(lambda x, key: _initial(solver, process, grid, x, key))(x_T, keys)


def _walks(model, process, autoencoder, finish, final_denoise, decode,
           solver, guidance, times, initial, params, given, null, x_T, keys, streams):
    """`_walk` of each request's rows, mapped over the requests, the weights and the grid shared, so a
    served pass runs each request's walk as its lone call does under one program, each request's steps
    sent to its own of `streams` where they stream."""
    def walked(initial, given, null, x_T, key, stream):
        return _walk(model, process, autoencoder, finish, final_denoise, decode, solver, guidance, times,
                     initial, params, given, null, x_T, key, stream)

    return jax.vmap(walked)(initial, given, null, x_T, keys, streams)


def _walk(model, process, autoencoder, finish, final_denoise, decode,
          solver, guidance, times, initial, params, given, null, x_T, key, stream=None):
    """The walk over the rows, compiled with the XLA options of its denoiser, its guidance's guide and
    its autoencoder (`program`), so a quantized model and another never share it. Rebinding weights
    must not change the static compilation identity, nor a solver's or guidance's numbers, nor the
    grid's values: the solver and the guidance are pytrees whose structure is static and whose numbers
    (a guidance scale, a Brownian seed) are data, and the grid is data with the solver state
    `_initial` built from it. Each step's clean prediction goes to `stream` where one is given
    (`ImageRequest.on_step`), through the walk's own step reader (`sample`'s `record`)."""
    variables = {name: value for name, value in params.items() if name not in ("encoders", "autoencoder")}
    denoise = process.denoiser(model, variables, given, None if guidance is None else null)

    def streamed(x, state, denoised) -> None:
        jax.debug.callback(_streamed, stream, denoised)

    images, latents = walk_and_decode(denoise, autoencoder, params, x_T, solver=solver, guidance=guidance,
                                      key=key, times=times, final_denoise=final_denoise, initial=initial,
                                      decode=decode, record=None if stream is None else streamed)
    if finish is not None and images is not None:
        images = finish(params, images)
    return Images(images, latents=latents)
