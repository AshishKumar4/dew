"""A trained model, rebuilt from its run, that turns prompts into images."""

from __future__ import annotations

import functools
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import TYPE_CHECKING, Generic

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn, struct
from flax.core import freeze
from jax.experimental import multihost_utils
from jax.typing import ArrayLike, DTypeLike
from typing_extensions import TypeVar

from dew.artifacts import uint8_pixels
from dew.coordination import agreed
from dew.diffusion.process import Conditioning, Process
from dew.inputs import InputSpec, unit_range
from dew.nn.autoencoders import AutoEncoder
from dew.nn.inputs import RowPlan, generation_signature, local_rows, mesh_of, request_key
from dew.objectives.base import Variables, thaw
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
        return jax.tree.map(lambda leaf: local_rows(leaf)[:self.rows], self)

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
        pixels = uint8_pixels(local_rows(self.images)[:self.rows])
        return [Image.fromarray(row[..., 0] if row.shape[-1] == 1 else row) for row in pixels]


@dataclass(frozen=True, eq=False)
class TextToImage:
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
    blank: Callable[[dict], dict] | None = None
    """A function that returns the task's own unconditional branch in the
    dtypes of the conditional branch it is given. Whoever built this task
    encoded the branch once (`DiffusionObjective.blank_conditions`). None
    means each call encodes the unconditional prompt itself, for a source
    that has no encoded branch."""

    def __post_init__(self) -> None:
        # A tree split for training, an adapter's, is read whole.
        object.__setattr__(self, "variables", freeze(dict(thaw(self.variables))))

    def bind(self, variables: Variables) -> TextToImage:
        """Return this task with `variables` as its weights.

        When encoder leaves change, the unconditional branch is encoded again,
        lazily on first use. When only the denoiser's weights change, the
        branch already encoded is kept."""
        from dew.objectives.diffusion.objective import FixedBlank

        blank = self.blank
        if isinstance(blank, FixedBlank):
            blank = blank.rebind(variables.get("encoders", {}))
        return replace(self, variables=variables, blank=blank)

    def quantized(self, spec: Quantization) -> TextToImage:
        """Return this task with its denoiser's weights quantized as `spec` says.

        The denoiser's matmuls then compute with the quantized weights
        (`dew.training.quantization.quantize_for_serving`). The encoders and
        the autoencoder keep their weights."""
        from dew.training.quantization import quantize_for_serving

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
        from dew.objectives.diffusion.objective import _without_loss_heads

        autoencoder, variables = objective.published_autoencoder(variables)
        return cls(objective.model, objective.process, objective.inputs,
                   _without_loss_heads(variables), autoencoder,
                   steps=objective.steps, guidance=objective.guidance, solver=objective.solver,
                   blank=objective._fixed_blank.rebind(variables.get("encoders", {})))

    @classmethod
    def from_run(cls, directory: str, *, ema: bool | None = None, step: int | str | None = None,
                 mesh: MeshSpec | None = None, layout: Layout | None = None,
                 dtype: DTypeLike | None = None, param_dtype: DTypeLike | None = None,
                 trust: Sequence[str] = ()) -> TextToImage:
        """Load the run in `directory`, built from its `run.json` the way the recipe built it.

        The weights come from the run's latest checkpoint, or from `step`.

        `ema` None reads the averaged weights when the run kept them and the
        live ones when it kept none; True requires the averaged weights, and
        False reads the live ones. A run whose averaged weights are a reference
        policy, such as Flow-GRPO's frozen KL reference, always reads its live
        policy. With `mesh`, the weights are restored directly onto that mesh
        under `layout`, the way the trainer places them. Without it, they go
        onto a default `MeshSpec()` over the current pool's devices.

        The configured unconditional prompt is encoded the way an objective's
        pipeline encodes it: eagerly, at the matmul precision recorded when the
        objective was built. The record must have a `condition_precision`
        field, which is None when the objective used JAX's default, whatever
        precision the caller's context sets.

        `dtype` sets the compute dtype of the model, the encoders and the VAE.
        `param_dtype` sets the dtype the parameters are stored in; None keeps
        the checkpoint's dtypes exactly. `trust` names the packages outside
        Dew the run's record may import (`TextGeneration.from_run`).
        """
        from dew.config import ModelConfig
        from dew.diffusion.process import Process
        from dew.inference.tasks import run_record
        from dew.nn.autoencoders import AutoEncoder
        from dew.objectives.diffusion.objective import FixedBlank, _without_loss_heads
        from dew.records import integer, record as fields, text
        from dew.registry import from_record, objectives, solvers

        record = run_record(directory, step, trust)
        config = ModelConfig.from_dict(fields(record['model'], 'model'))
        inputs_record = fields(record['inputs'], 'inputs')
        autoencoder_record = None if record['autoencoder'] is None else fields(record['autoencoder'],
                                                                               'autoencoder')
        compute = dtype_name(resolve_dtype(dtype))
        if compute is not None:
            config = config.with_dtype(compute)
            conditions = {keyword: fields(condition, keyword) for keyword, condition
                          in fields(inputs_record['conditions'], 'conditions').items()}
            inputs_record = {**inputs_record, 'conditions': {
                keyword: {**condition,
                          'encoder': _computing(fields(condition['encoder'], 'encoder'), compute)}
                for keyword, condition in conditions.items()}}
            if autoencoder_record is not None:
                autoencoder_record = _computing(autoencoder_record, compute)
        params = objectives[text(record['objective'], 'objective')]._saved_variables(
            directory, ema=ema, step=step, mesh=mesh, layout=layout, param_dtype=param_dtype,
            parameter_roots=_parameter_roots(inputs_record, autoencoder_record))
        inputs = InputSpec.from_json(inputs_record, params=params.get('encoders', {}))
        end_to_end = record.get('end_to_end')
        autoencoder = None
        if autoencoder_record is not None and end_to_end is not None:
            from dew.objectives.diffusion.end_to_end import AUTOENCODER, EndToEnd
            tuning = from_record(EndToEnd, fields(end_to_end, 'end_to_end'), dtypes=False)
            frozen = AutoEncoder.from_json(autoencoder_record, params=params['params'][AUTOENCODER])
            autoencoder, params = tuning.tuned(frozen, params)
        elif autoencoder_record is not None:
            autoencoder = AutoEncoder.from_json(autoencoder_record, params=params['autoencoder'])
        solver_record = fields(record['solver'], 'solver')
        solver = solvers.build(text(solver_record['class'], 'solver class'),
                                fields(solver_record['fields'], 'solver fields'))
        guidance = from_record(Guidance | None, record['guidance'], dtypes=False)
        precision = record['condition_precision']
        precision = None if precision is None else text(precision, 'condition_precision')
        return cls(config.build(), Process.from_json(fields(record['process'], 'process')),
                   inputs, _without_loss_heads(params), autoencoder,
                   steps=integer(record['sampling_steps'], 'sampling_steps'),
                   guidance=guidance, solver=solver,
                   blank=FixedBlank(inputs, params.get("encoders", {}), precision))

    @classmethod
    def from_pretrained(cls, repo_id: str, *, revision: str | None = None,
                        ema: bool | None = None, mesh: MeshSpec | None = None,
                        layout: Layout | None = None, dtype: DTypeLike | None = None,
                        param_dtype: DTypeLike | None = None, trust: Sequence[str] = ()) -> TextToImage:
        """Download a run directory from the Hugging Face Hub and load it with `from_run`.

        The repository holds the run directory as `HfApi().upload_folder`
        writes it. `revision` is the Hub revision to download; the other
        arguments are `from_run`'s."""
        from dew.interop.hub import pull_from_hub

        return cls.from_run(os.fspath(pull_from_hub(repo_id, revision=revision)),
                            ema=ema, mesh=mesh, layout=layout,
                            dtype=dtype, param_dtype=param_dtype, trust=trust)

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
            return _encode(plan.sharding)(self._conditions, self.variables, plan.place(plan.pad(tokens)))
        return _encode(None)(self._conditions, self.variables, jax.tree.map(jnp.asarray, tokens))


    def prepare(
        self,
        prompts: str | Sequence[str | Mapping[str, object]] | None = None,
        *,
        conditions: Mapping[str, Conditioning] | None = None,
        key: int | jax.Array | None = None,
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
            return self._resolved(mesh, prompts, conditions, key=key, steps=steps,
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

    def _resolved(self, mesh, prompts, conditions, *, key, steps, unconditional, image,
                  image_latents, mask, noise, initial, times, encode_key) -> _Resolved:
        """Everything `prepare` settles on the host, in one value.

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
        request = request_key(key)
        count = self.steps if steps is None else steps
        process, source_times = self.prepared_process(count)
        selected = _time_grid(times) if times is not None else source_times
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

    def _encoded(self, settled: _Resolved, *, configured: bool
                 ) -> tuple[dict, dict, jax.Array]:
        """The two encoded conditioning branches and the initial state.

        With no caller-supplied pixels the state is fresh noise. With any,
        the autoencoder runs over them, and a mask adds its spatial
        conditioning to both branches, the unconditional one broadcast to the
        batch first because its single row would not carry the mask.
        """
        plan, process = settled.plan, settled.process
        given = plan.place(plan.pad(settled.tokens))
        if "given" not in settled.encoded:
            given = _encode(plan.sharding)(self._conditions, self.variables, given)
        if settled.null_tokens is None:
            null = {}
        elif "null" in settled.encoded:
            single = jax.tree.leaves(settled.null_tokens)[0].shape[0] == 1
            null = (jax.device_put(settled.null_tokens) if single
                    else plan.place(plan.pad(settled.null_tokens)))
        else:
            null = self._unconditional(settled.null_tokens, plan, given, configured=configured)
        if not settled.samples:
            return given, null, _noise(plan.sharding)(process, plan.keys(settled.request),
                                                      settled.shape)
        start = process.times(settled.count)[0] if settled.times is None else settled.times[0]
        initial_state, spatial = _image_start(plan.sharding)(
            self.autoencoder, process, settled.shape, self.variables,
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
        """Generate images for `prompts`, as `Images` in [-1, 1] of shape `[rows, H, W, C]`.

        `prompts` can also be `DenoisingInputs` from `prepare`. `guidance` is a
        classifier-free guidance scale, a `CFG` with its interval, or None for
        the plain conditional prediction; when it is omitted, the task's
        default applies, and so do the task's `steps` and `solver`. With
        `decode=False` the result holds only the latents."""
        mesh = mesh_of(self.variables)

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
                                         solver, chosen, self.final_denoise, times, decode, self.variables,
                                         prepared.conditions, prepared.unconditional,
                                         prepared.noise, jax.random.fold_in(request, 1))
        return replace(generated, rows=plan.rows)


def _parameter_roots(inputs: Mapping[str, object], autoencoder: Mapping[str, object] | None
                     ) -> tuple[tuple[str, ...], ...]:
    """The parameter roots of a recorded run's tree, which a storage override
    casts: the denoiser's, each condition encoder's own collections and the
    autoencoder's. The rest keep the dtypes they were saved in."""
    from dew.objectives.base import FROZEN
    from dew.records import record, text
    from dew.registry import encoders

    roots: list[tuple[str, ...]] = [("params",), (FROZEN,)]
    for keyword, condition in record(inputs['conditions'], 'conditions').items():
        name = text(record(record(condition, keyword)['encoder'], 'encoder')['class'], 'encoder class')
        collections = encoders[name].parameter_collections
        roots.extend([("encoders", keyword)] if collections is None else
                     [("encoders", keyword, collection) for collection in collections])
    if autoencoder is not None:
        roots.append(("autoencoder",))
    return tuple(roots)


def _computing(owner: Mapping[str, object], compute: str) -> Mapping[str, object]:
    """A recorded encoder or autoencoder computing in `compute`: its own
    `dtype`, and the dtype of the model it wraps where it records one."""
    from dew.records import record

    recorded = dict(record(owner['fields'], 'fields'))
    if 'dtype' in recorded:
        recorded['dtype'] = compute
    model = recorded.get('model')
    if isinstance(model, Mapping) and 'dtype' in model:
        recorded['model'] = {**record(model, 'model'), 'dtype': compute}
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






@functools.cache
def _encode(rows: jax.sharding.NamedSharding | None):
    def encode(conditions, params, tokens):
        return {keyword: encoder.encode(params["encoders"][keyword], tokens[keyword])
                for keyword, encoder in conditions}
    return jax.jit(encode, static_argnums=(0,), in_shardings=(None, rows), out_shardings=rows)


def _encoded_rows(conditions: Mapping[str, Conditioning]) -> int:
    """The samples an encoded branch holds: its leaves' common leading axis."""
    counts = {leaf.shape[0] for leaf in jax.tree.leaves(dict(conditions)) if leaf.ndim}
    if len(counts) != 1:
        raise ValueError(f"encoded conditions need one row count across their leaves, got {sorted(counts)}")
    return counts.pop()


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
