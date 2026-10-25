"""The denoising diffusion objective.

Sample a noise level, corrupt, predict, weight. The convention (schedule,
parameterization, weighting) is the `Process`; the sample field and the
conditions are the `InputSpec`; every draw comes from the step's key. The
frozen encoders' weights live in the tree's `encoders` collection, so they
reach the compiled step as arguments and the optimizer never sees them. The
unconditional branch is a pure function of those frozen weights and a fixed
prompt, so the objective encodes it once, on first use, and the step reads
that: the tower runs over the batch and nothing else, once a step.

Evaluation samples one image per validation row from its conditions, with the
averaged weights when the run keeps them, through the same `sample` inference
uses; the preview hook limits itself to the display count.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn

from dew.artifacts import ImageGrid, VideoGrid
from dew.coordination import agreed, collective_host
from dew.diffusion.presets import Preset, build_process
from dew.diffusion.process import Process, aligned_conditions
from dew.diffusion.schedules import FlowMatchingScheduler, expand
from dew.diffusion.schedules.source import SourceGrid
from dew.diffusion.transforms import FlowMatchPredictionTransform, broadcast_rates
from dew.inputs import InputSpec, encode_conditions, unit_range
from dew.lora import unadapted
from dew.nn.autoencoders import AutoEncoder
from dew.nn.autoencoders.kl import posterior_latent
from dew.nn.flops import MissingFlopsCount
from dew.nn.mp import Uncertainty
from dew.nn.protocols import (
    CountedCondition,
    CountedConditionedSpatial,
    CountedConditionLayout,
    CountedImages,
    CountedMultimodal,
    CountedSequences,
    CountedSpatial,
    CountedVideo,
    RequiresText,
)
from dew.objectives.base import (
    FROZEN,
    OMITTED,
    TEACHER,
    Aux,
    Batch,
    EMASpec,
    Objective,
    Omitted,
    ProgramModule,
    Ratio,
    Step,
    Variables,
    merge,
    model_rngs,
    thaw,
    under,
)
from dew.objectives.diffusion.alignment import ALIGNMENT, REPRESENTATION, Alignment
from dew.objectives.diffusion.end_to_end import AUTOENCODER, LATENT_STATS, PERCEPTUAL, EndToEnd
from dew.sampling.guidance import CFG, Guidance
from dew.sampling.pipelines import ImageSettings, TextToImage, sampling_modules, walk_and_decode
from dew.sampling.solvers import DDIM, Consistency, Solver

if TYPE_CHECKING:
    from dew.data.pipeline import ObjectiveData
    from dew.inference.tasks import Processor

# Samples a validation batch draws, conditioned or not.
VALIDATION_SAMPLES = 4

UNCERTAINTY = "loss_uncertainty"
"""Where the learned loss weighting's head lives in the `params` and
`constants` collections, beside the model's own modules."""


FAKE_SCORE = "fake_score"
"""Where rCM's fake score network lives in `params` and `constants`."""

DISCRIMINATOR = "discriminator"
"""Where an adversarial objective's discriminator heads live in `params`."""

SPECTRAL = "spectral"
"""The collection of an adversarial objective's spectral-norm vectors."""

LOSS_HEADS = (UNCERTAINTY, ALIGNMENT, AUTOENCODER, FAKE_SCORE, DISCRIMINATOR)
"""What trains beside the model under `params` and the model never reads."""


def _own_loss(name: str, kwargs: dict) -> None:
    """Refuse the denoising loss's extras, which an objective with its own
    loss would leave unused, and a guidance: its model samples unguided, the
    few steps it learns or the guidance it was trained with standing in."""
    unused = sorted(key for key in ("uncertainty", "alignment", "end_to_end", "guidance")
                    if kwargs.get(key) is not None)
    if unused:
        raise ValueError(f"{name} trains on its own loss and samples unguided, which reads none of {unused}")
    kwargs["guidance"] = None


def teacher_weights(variables: Variables | None, given: Variables | None, *,
                    whole: bool = False) -> Variables:
    """A distillation's frozen teacher: the starting tree's `TEACHER` when it
    holds one, as a saved student's does; else the teacher's tree `given`, as
    a run config reads it from the teacher's run (`DiffusionRunConfig.teacher_run`);
    else the model the starting tree holds, which the student then starts as.
    The model's own collections alone (`model_part`) unless `whole`."""
    if variables is not None and TEACHER in variables:
        return variables[TEACHER]
    if given is not None:
        tree = given
    elif variables is not None and "params" in variables:
        tree = variables
    else:
        raise ValueError(f"the teacher's weights come from the starting tree's {TEACHER!r}, the teacher's "
                         f"tree (teacher_variables=) or the model the starting tree holds, and none is given")
    return thaw(tree) if whole else model_part(tree)


def without_loss_heads(variables: Variables) -> Variables:
    """`variables` without what the model never reads: the uncertainty head,
    the alignment projector, the frozen representation encoder, an
    autoencoder trained end to end with its latent statistics, perceptual
    network and discriminator, a fake score and a teacher."""
    return {name: ({key: value for key, value in tree.items() if key not in LOSS_HEADS}
                   if name in ("params", "constants") else tree)
            for name, tree in variables.items()
            if name not in (REPRESENTATION, LATENT_STATS, PERCEPTUAL, TEACHER, SPECTRAL)}


def model_part(tree: Variables) -> Variables:
    """The model's own collections of `tree`, a frozen split merged back: the
    frozen towers and the loss's own heads left out."""
    return without_loss_heads({name: value for name, value in thaw(tree).items()
                                if name not in ("encoders", "autoencoder")})


def sampling_grid(process: Process, steps: int, grid: SourceGrid | None = None) -> tuple[Process, jax.Array]:
    """The process and the concrete descending times `steps` steps sample on: a loaded pipeline's
    own (`grid`, `SourceTask.grid`), else `process`'s grid."""
    with jax.ensure_compile_time_eval():
        return (process, process.times(steps)) if grid is None else grid(steps)


def check_solver(process, solver, steps: int, grid: SourceGrid | None = None) -> None:
    """Trace a solver step on the objective's actual sampling grid (`sampling_grid`)."""
    if isinstance(solver, str):
        raise TypeError(f"solver={solver!r} names a solver; pass the solver itself, as Euler()")
    x = jnp.zeros((1, 1), jnp.float32)
    process, times = sampling_grid(process, steps, grid)
    state = solver.init(x, times, process, key=jax.random.PRNGKey(0))
    if times.shape[0] < 2:
        return
    t, t_next = times[:1], times[1:2]
    key = jax.ShapeDtypeStruct((2,), jnp.uint32)
    jax.eval_shape(
        lambda x, key: solver.step(x, t, t_next, x, x, state, key, process,
                                    lambda x_t, t_: (x_t, x_t)),
        x, key)


def _check_end_to_end(end_to_end: EndToEnd | None, alignment: Alignment | None,
                      autoencoder: AutoEncoder | None, inputs: InputSpec) -> None:
    """Refuse an end-to-end tuning the objective cannot train, and masked-image inputs without the
    autoencoder that encodes them."""
    if end_to_end is not None:
        if alignment is None or autoencoder is None or inputs.mask is not None:
            raise ValueError("end-to-end tuning trains a KL autoencoder through REPA's loss; it "
                             "needs `alignment`, a KL autoencoder and no masked-image input")
        # An autoencoder without the posterior the step trains through refuses it by name.
        jax.eval_shape(autoencoder.moments, autoencoder.params, jnp.zeros((1, *inputs.sample.shape)))
    if inputs.mask is not None and autoencoder is None:
        raise ValueError("Masked-image conditioning requires an autoencoder")


class TunedLatents(NamedTuple):
    """An end-to-end step's latents: the autoencoder's raw draw, the
    normalized samples and the running statistics after the pass,
    the autoencoder's regularizer, the discriminator's hinge
    loss, and their terms."""

    raw: jax.Array
    samples: jax.Array
    statistics: Variables
    regularizer: jax.Array
    hinge: jax.Array
    terms: dict[str, jax.Array]


_DEFAULT_SOLVER = DDIM()
_DEFAULT_GUIDANCE = CFG(3.0)
_DEFAULT_STEPS = 200
_CONSISTENCY = Consistency()


class _SourceSolver(Protocol):
    @property
    def solver(self) -> Solver: ...


class _SourceCall(Protocol):
    @property
    def steps(self) -> int: ...

    @property
    def guidance(self) -> Guidance | None: ...

    @property
    def grid(self) -> SourceGrid: ...


@runtime_checkable
class PipelineSource(Protocol):
    """A loaded diffusion pipeline a `DiffusionObjective` can train in place
    of a bare denoiser: its denoiser `model`, starting `variables`, `process`,
    `inputs` and `autoencoder`, and how it samples, the solver of its
    `schedule` and the steps, guidance and grid of its `task`.
    `dew.interop.PretrainedPipeline` is one; the protocol keeps
    `dew.objectives` from importing `dew.interop`."""

    @property
    def model(self) -> nn.Module: ...

    @property
    def variables(self) -> Variables: ...

    @property
    def process(self) -> Process: ...

    @property
    def inputs(self) -> InputSpec: ...

    @property
    def autoencoder(self) -> AutoEncoder | None: ...

    @property
    def schedule(self) -> _SourceSolver: ...

    @property
    def task(self) -> _SourceCall: ...


class DiffusionObjective(Objective[Ratio]):
    """Trains a denoising diffusion model: draw a noise level, corrupt, predict, and weight the loss."""

    saved_task = TextToImage
    settings = ImageSettings
    # A pipeline's own sampling policy, else DDIM, CFG(3.0) and `_DEFAULT_STEPS`.
    resolved = ("solver", "guidance", "steps")

    counts_tokens = False

    def data(self, tokenizer=None) -> ObjectiveData:
        """Each batch's captions tokenized for the conditions, on the host, as
        the step reads them, and kept beside as host strings, which a metric
        or a reward reads with its own tokenizer (`dew.eval.images.CLIPScore`);
        and the fields other conditions read (a clip's sound) encoded."""
        from dew.data.images import EncodeCaptions, EncodeFields
        from dew.data.pipeline import ObjectiveData

        conditions = self.inputs.conditions.values()
        steps = (EncodeCaptions(self.inputs.tokenize if conditions else None, keep=True),
                 *((EncodeFields(self.inputs.encode_fields),)
                   if any(not condition.encoder.reads_captions for condition in conditions) else ()))
        return ObjectiveData(train_batch=steps, evaluate_batch=steps)

    def real_lengths(self, batch: Batch) -> np.ndarray:
        """The denoiser's real patch-token lengths from the objective's latent shape."""
        rows = np.shape(batch[self.inputs.sample.key])[0]
        if isinstance(self.model, CountedVideo):
            frames, patches = self.model.video_lengths(self.latent_shape, rows)
            return frames * patches
        if not isinstance(self.model, CountedImages | CountedSequences | CountedMultimodal):
            raise MissingFlopsCount(f"{type(self.model).__name__} has no declared image_lengths geometry")
        return self.model.image_lengths(self.latent_shape, rows)

    def model_flops(self, model, batch: Batch) -> float:
        """Assemble the denoiser's patch/time/text projections and its layers."""
        denoiser, encoder = self._forward_flops(model, batch)
        return denoiser + encoder

    def training_flops(self, model, batch: Batch) -> float:
        """The denoiser trains; the frozen condition encoder executes only forward."""
        denoiser, encoder = self._forward_flops(model, batch)
        return 3 * denoiser + encoder

    def _forward_flops(self, model, batch: Batch) -> tuple[float, float]:
        model = self.model if model is None else model
        if type(self) is not DiffusionObjective or self.alignment is not None or self.end_to_end is not None:
            raise MissingFlopsCount(f"{type(self).__name__} needs counts for its "
                                    "additional forwards and heads")
        if self.autoencoder is not None or self.uncertainty is not None:
            raise MissingFlopsCount("DiffusionObjective needs auxiliary autoencoder/uncertainty work")
        if len(self.inputs.conditions) > 1:
            raise MissingFlopsCount("DiffusionObjective needs the composition "
                                    "of multiple condition geometries")
        if isinstance(model, RequiresText) and not self.inputs.conditions:
            raise MissingFlopsCount(f"{type(model).__name__} needs real text context lengths")
        conditions = []
        for condition in self.inputs.conditions.values():
            encoder = condition.encoder
            if not isinstance(encoder, CountedCondition):
                raise MissingFlopsCount(f"{type(encoder).__name__} has no conditioning_flops(tokens)")
            work = encoder.conditioning_flops(batch[condition.field])
            conditions.append(work)
        encoded = sum(condition.flops for condition in conditions)
        context = None
        if conditions:
            if not isinstance(model, CountedConditionLayout):
                raise MissingFlopsCount(f"{type(model).__name__} has no condition_lengths layout declaration")
            work = conditions[0]
            context = model.condition_lengths(work.lengths, work.spans)
            if self.unconditional_prob:
                condition = next(iter(self.inputs.conditions.values()))
                encoder = condition.encoder
                if not isinstance(encoder, CountedCondition):
                    raise MissingFlopsCount(f"{type(encoder).__name__} has no conditioning_flops(tokens)")
                blank = encoder.conditioning_flops(encoder.tokenize([condition.unconditional]))
                dropped = replace(work, lengths=np.broadcast_to(blank.lengths, work.lengths.shape))
                blank_context = model.condition_lengths(dropped.lengths, dropped.spans)
                if self.unconditional_prob == 1:
                    work, context = dropped, blank_context
                elif np.any(context != blank_context):
                    raise MissingFlopsCount("DiffusionObjective needs post-dropout context lengths; "
                                            "the host count does not know the conditioning-dropout draw")
            conditions[0] = work
        features = None if not conditions else conditions[0].features
        rows = np.shape(batch[self.inputs.sample.key])[0]
        if isinstance(model, CountedConditionedSpatial) and conditions:
            return model.conditioned_spatial_flops(self.latent_shape, rows, conditions[0]), encoded
        if isinstance(model, CountedSpatial):
            return model.spatial_flops(self.latent_shape, rows, context, features), encoded
        if isinstance(model, CountedVideo):
            frames, patches = model.video_lengths(self.latent_shape, rows)
            return model.spatiotemporal_flops(frames, patches, self.latent_shape[-1], features), encoded
        if isinstance(model, CountedMultimodal):
            lengths = model.image_lengths(self.latent_shape, rows)
            return model.multimodal_flops(lengths, np.zeros(lengths.shape, np.int64)
                                          if context is None else context,
                                          self.latent_shape[-1], features or 0), encoded
        if isinstance(model, CountedSequences):
            if context is None or features is None:
                raise MissingFlopsCount(f"{type(model).__name__} needs real text context lengths")
            lengths = model.image_lengths(self.latent_shape, rows)
            return model.multisequence_flops(lengths, context, input_channels=self.latent_shape[-1],
                                             text_features=features,
                                             pooled_features=conditions[0].pooled_features), encoded
        if not isinstance(model, CountedImages):
            raise MissingFlopsCount(f"{type(model).__name__} has no declared image_flops geometry")
        return model.image_flops(model.image_lengths(self.latent_shape, rows),
                                  self.latent_shape[-1], features), encoded

    @property
    def inputs(self) -> InputSpec:
        if self._inputs is None:
            raise ValueError("a diffusion objective requires an InputSpec")
        return self._inputs

    @inputs.setter
    def inputs(self, inputs: InputSpec | None) -> None:
        self._inputs = inputs

    def __init__(
        self,
        model: nn.Module | PipelineSource,
        process: Process | Preset | None = None,
        inputs: InputSpec | None = None,
        *,
        autoencoder: AutoEncoder | None | Omitted = OMITTED,
        unconditional_prob: float = 0.12,
        ema_decay: float | optax.Schedule | None = 0.999,
        solver: Solver = _DEFAULT_SOLVER,
        guidance: Guidance | None = _DEFAULT_GUIDANCE,
        steps: int | None = None,
        variables: Variables | None | Omitted = OMITTED,
        uncertainty: int | None = None,
        alignment: Alignment | None = None,
        end_to_end: EndToEnd | None = None,
        grid: SourceGrid | None | Omitted = OMITTED,
    ):
        """Build a denoising objective over `model` for the `inputs` field.

        `process` is a preset or a custom `Process`; a preset is built once here.
        `solver`, `guidance` and `steps` set how evaluation samples, by default DDIM,
        `CFG(3.0)` and 200 steps. `guidance` None samples the plain conditional
        prediction.

        `variables` is the tree training starts from. It is the denoiser's variables
        next to its frozen towers (`encoders`, `autoencoder`), as a loaded pipeline
        holds them, or an adapter's split of them. A split tree is kept as given, so
        the optimizer updates only what is in `params`. None draws the denoiser and
        uses the towers as built.

        `model` may be a loaded pipeline instead of the denoiser
        (`DiffusionObjective(flux)`). The pipeline then supplies the denoiser,
        `variables`, `process`, `inputs`, `autoencoder` and its own sampling policy,
        and any of them passed here overrides the pipeline's, an explicit None
        included (`autoencoder=None` trains in pixel space, `variables=None` draws
        the denoiser). Evaluation and the published task sample on the pipeline's
        own grid and end as its solver does (`SourceTask.grid`), whatever process
        trains. `grid` names that grid for a denoiser built apart from its
        pipeline, as a run config builds one; None samples on the process's own
        times with a closing denoise. The pipeline's text encoder trains nothing but encodes every
        caption, so a pipeline loaded without it is refused.

        `ema_decay` is the EMA's decay, either a number or a schedule over the number
        of updates so far, such as EDM2's power EMA
        (`dew.training.posthoc.power_decay`). None keeps no EMA.

        `uncertainty` learns EDM2's loss weighting (equation 21 of Karras et al.
        2024). It adds a head u of the model time, `dew.nn.mp.Uncertainty` with this
        many Fourier channels, and replaces the loss w ||D - y||^2 with
        w / e^u ||D - y||^2 + u, halved as Dew's L2 loss is. The loss is smallest
        when u is the log of the weighted error each noise level leaves, so every
        level contributes about equally. The head trains next to the model under
        `params`, as `UNCERTAINTY`, and a published task drops it. None keeps the
        preset's fixed weighting.

        `alignment` adds REPA's or iREPA's representation alignment (`Alignment`)
        between the model's hidden tokens at one layer and a frozen encoder's
        features of the clean sample. Its projector trains under `params` as
        `ALIGNMENT`, the encoder's weights stay frozen under `REPRESENTATION`, taken
        from `variables` when it holds them and from the alignment's source
        otherwise, and a published task drops both.

        `end_to_end` trains the autoencoder together with the model, as REPA-E does
        (`EndToEnd`). It needs `alignment` and a KL autoencoder. The autoencoder's
        weights then train under `params` as `AUTOENCODER`, and a batch norm
        normalizes its latents, with the running statistics in the `LATENT_STATS`
        collection. A published task includes the tuned autoencoder, with those
        statistics as its latent normalization.
        """
        if isinstance(model, PipelineSource):
            source = model
            model = source.model
            process = source.process if process is None else process
            inputs = source.inputs if inputs is None else inputs
            autoencoder = source.autoencoder if autoencoder is OMITTED else autoencoder
            variables = source.variables if variables is OMITTED else variables
            solver = source.schedule.solver if solver is _DEFAULT_SOLVER else solver
            guidance = source.task.guidance if guidance is _DEFAULT_GUIDANCE else guidance
            steps = source.task.steps if steps is None else steps
            grid = source.task.grid if grid is OMITTED else grid
            held = source.variables.get("encoders", {})
            if any(keyword not in held for keyword in inputs.conditions):
                raise ValueError("this pipeline was loaded without its text encoder (text=False), and "
                                 "training encodes its captions; load it with its text encoder")
        grid = None if grid is OMITTED else grid
        if process is None or inputs is None:
            raise ValueError("a denoiser needs its `process` and `inputs`; a loaded pipeline carries both")
        if isinstance(model, RequiresText) and model.text_keyword not in inputs.conditions:
            raise ValueError(f"{type(model).__name__} reads text as {model.text_keyword!r} on every call and "
                             f"cannot run unconditionally; its inputs give {sorted(inputs.conditions)}")
        autoencoder = None if autoencoder is OMITTED else autoencoder
        variables = None if variables is OMITTED else variables
        steps = _DEFAULT_STEPS if steps is None else steps
        self.model = model
        self.process = build_process(process)
        self.inputs = inputs
        self.autoencoder = autoencoder
        self.variables = variables
        self.uncertainty = None if uncertainty is None else Uncertainty(uncertainty)
        _check_end_to_end(end_to_end, alignment, autoencoder, inputs)
        check_solver(self.process, solver, steps, grid)
        # Refusals come first: resolving the alignment's encoder may download its weights.
        self.alignment, self.representation = (None, None) if alignment is None else alignment.network(
            None if variables is None else variables.get(REPRESENTATION))
        self.end_to_end = end_to_end
        self.unconditional_prob = unconditional_prob
        self.solver = solver
        self.guidance = guidance
        self.steps = steps
        self.grid = grid
        self.ema = (None if ema_decay is None else EMASpec(
            decay=ema_decay if callable(ema_decay) else optax.constant_schedule(ema_decay),
            select=under("params")))
        self._grid = VideoGrid if len(inputs.sample.shape) == 4 else ImageGrid

    @property
    def final_denoise(self) -> bool:
        """Whether a sample ends on a closing denoise: on the process's own times, and not on a
        pipeline's grid, whose solver ends the walk itself (`PretrainedPipeline.text_to_image`)."""
        return self.grid is None

    @property
    def _sample(self):
        return self.program('sample', self._sample_impl, static_argnames=("count",),
                            holds=tuple(held for held in (self.guidance, self.autoencoder)
                                        if held is not None),
                            modules=sampling_modules(self.model, self.guidance, self.autoencoder))

    def task_settings(self) -> ImageSettings:
        """The process, input encoders, autoencoder and sampling convention."""
        from dew.registry import to_record
        return ImageSettings(self.process, self.inputs.to_json(),
                             None if self.autoencoder is None else self.autoencoder.to_json(),
                             self.solver, self.guidance, self.steps,
                             # A tuned autoencoder's weights and statistics sit in the run's own tree.
                             None if self.end_to_end is None else to_record(self.end_to_end, EndToEnd),
                             grid=None if self.grid is None else to_record(self.grid, SourceGrid))

    def build_task(self, variables: Variables, *,
                   processor: Processor | None | Omitted = OMITTED) -> TextToImage:
        """Return the model over `variables`' published weights as a `TextToImage` task.

        The task samples the same way this objective's evaluation does. Its
        conditions are encoded by the run's own towers, so it takes no processor.
        """
        from dew.sampling.pipelines import TextToImage

        if processor is not OMITTED:
            raise TypeError("a text-to-image task encodes its conditions with the run's own towers, "
                            "so it takes no processor")
        return TextToImage.from_objective(self, thaw(variables))

    @property
    def latent_shape(self) -> tuple[int, ...]:
        """The per-example shape the model denoises.

        That is the sample field's shape, or its latent's shape when an autoencoder
        sits in front of the model.
        """
        shape = self.inputs.sample.shape
        return shape if self.autoencoder is None else self.autoencoder.latent_shape(shape)

    def encoder_params(self) -> dict:
        if self.variables is not None and "encoders" in self.variables:
            return dict(self.variables["encoders"])
        return {keyword: condition.encoder.params
                for keyword, condition in self.inputs.conditions.items()}

    def encode(self, encoders, tokens: dict | None = None) -> dict:
        """Encode the conditions with the given parameters.

        A condition whose tokens are omitted uses its configured unconditional input.
        """
        if tokens is None:
            tokens = {keyword: condition.encoder.tokenize([condition.unconditional])
                      for keyword, condition in self.inputs.conditions.items()}
        return encode_conditions(self.inputs.encoders, encoders, tokens)

    def blank_conditions(self, params, like: dict) -> dict:
        """The configured unconditional conditions, encoded by `params`' own towers and cast to the
        conditional branch's dtypes in `like`: the one function a task over these weights encodes
        its blank with too, so a step and the task read the same branch."""
        return jax.tree.map(lambda blank, value: blank.astype(value.dtype),
                            self.encode(params["encoders"]), like)

    def held_variables(self, *, beside: Variables | None = None) -> Variables:
        """Return every array `init` starts from instead of drawing: the starting tree, or the frozen towers.

        A text tower and a VAE are released weights of hundreds of megabytes, which a
        trace without arguments would compile into the state executable as
        constants. The loss's frozen networks ride beside a starting tree without them.
        This is one mapping, so an objective that starts from more than the towers
        extends both this and `init`. A part `beside` holds, a caller's own, is
        taken as given and not built: LPIPS's published weights are not fetched for
        a caller that brings its own.
        """
        held: dict[str, Any] = {**(self.variables or {}), **(beside or {})}
        if "params" not in held:
            # Parts a draw goes beside: the towers they leave out are the built ones.
            held.setdefault("encoders", self.encoder_params())
            if self.autoencoder is not None:
                held.setdefault("autoencoder", self.autoencoder.params)
        if self.representation is not None:
            held.setdefault(REPRESENTATION, self.representation)
        if self.end_to_end is not None and self.end_to_end.perceptual_weight and PERCEPTUAL not in held:
            from dew.eval.lpips import LPIPSNetwork

            held[PERCEPTUAL] = LPIPSNetwork.published()[1]
        return held

    def optimizer(self, tx: optax.GradientTransformation) -> optax.GradientTransformation:
        """Return `tx`, or under `end_to_end` one copy of it per network.

        REPA-E updates the model (with its alignment projector), the autoencoder and
        the discriminator with three optimizers, each clipping its own gradient, so
        each gets its own copy of `tx` with its own clip, moments and update count.
        """
        if self.end_to_end is None:
            return tx

        def network(params):
            return {name: name if name in (AUTOENCODER, DISCRIMINATOR) else "model" for name in params}

        return optax.multi_transform({"model": tx, AUTOENCODER: tx, DISCRIMINATOR: tx}, network)

    # The held tree is the starting tree or the frozen towers a draw goes beside.
    _held_parts = True

    def fresh_variables(self, key: jax.Array, held: Variables | None) -> Variables:
        """Draw the denoiser beside the held frozen towers."""
        held = self.held_variables() if held is None else held
        conditions = self.encode(held["encoders"])
        if self.inputs.mask is not None:
            conditions = {**conditions, "mask": jnp.zeros((1, *self.latent_shape[:-1], 1)),
                          "masked_image": jnp.zeros((1, *self.latent_shape))}
        drawn = self.model.init(key, jnp.ones((1, *self.latent_shape)), jnp.ones((1,)),
                                **conditions)
        state = {**drawn, "encoders": held["encoders"]}
        built: Variables | None = None
        for frozen in ("autoencoder", REPRESENTATION, PERCEPTUAL, DISCRIMINATOR, TEACHER):
            # The frozen weights are state, like the encoders'. They ride in
            # as an argument to the compiled step for the layout to place.
            # A caller holding only some towers takes the rest as built, and
            # a pretrained discriminator and a teacher ride in the same way.
            if frozen in held:
                value = held[frozen]
            else:
                built = self.held_variables(beside=held) if built is None else built
                value = built.get(frozen)
            if value is not None:
                state[frozen] = value
        return state

    def complete_variables(self, key: jax.Array, tree: Variables) -> Variables:
        """Add the loss's own heads the tree lacks: EDM2's uncertainty head,
        REPA-E's tuned autoencoder, statistics and discriminator, and REPA's
        projector. A split is kept: the optimizer moves what it leaves in `params`."""
        state: dict[str, Any] = dict(tree)
        head_key = jax.random.fold_in(key, 1)
        if self.uncertainty is not None and UNCERTAINTY not in state["params"]:
            head = self.uncertainty.init(head_key, jnp.ones((1,)))
            for collection, value in head.items():
                state[collection] = {**state.get(collection, {}), UNCERTAINTY: value}
        if self.end_to_end is not None and AUTOENCODER not in state["params"]:
            assert self.autoencoder is not None
            state["params"] = {
                **state["params"],
                AUTOENCODER: state.pop("autoencoder", self.autoencoder.params),
            }
            state[LATENT_STATS] = self.end_to_end.initial_statistics(
                self.autoencoder.latent_shift, self.autoencoder.latent_scale, self.autoencoder.latent_channels
            )
            network = self.end_to_end.discriminator
            if network is not None:
                given = state.pop(DISCRIMINATOR, None)
                if given is None:
                    given = network.init(jax.random.fold_in(key, 3),
                                         jnp.zeros((2, *self.inputs.sample.shape)), {})
                state["params"] = {**state["params"], DISCRIMINATOR: given["params"]}
        if self.alignment is not None and ALIGNMENT not in state["params"]:
            state["params"] = {**state["params"], ALIGNMENT: self._projector_init(
                jax.random.fold_in(key, 2), state)}
        return state

    def _projector_init(self, key, state) -> Variables:
        """The projector's parameters, shaped by the model's hidden tokens at
        the aligned layer and the encoder's feature width."""
        alignment = self.alignment
        assert alignment is not None
        conditions = jax.tree.map(lambda value: value[:1], self.encode(state["encoders"]))

        def hidden(variables):
            _, captured = self.model.apply(
                variables, jnp.ones((1, *self.latent_shape)), jnp.ones((1,)), **conditions,
                capture_intermediates=alignment.captures, mutable=["intermediates"])
            return self._captured(captured)

        tokens = jax.eval_shape(hidden, self.model_variables(state))
        features = jax.eval_shape(alignment.targets, state[REPRESENTATION],
                                  jnp.zeros((1, *self.inputs.sample.shape)))
        return alignment.init(key, tokens.shape[-1], tokens.shape[1], features.shape[-1])["params"]

    def _captured(self, captured) -> jax.Array:
        assert self.alignment is not None
        kept = captured.get("intermediates", {}).get(self.alignment.layer)
        if kept is None:
            raise ValueError(f"the model has no submodule {self.alignment.layer!r} to align")
        return kept["__call__"][0]

    def model_variables(self, variables: Variables) -> Variables:
        """Return the model's own collections (`model_part`)."""
        return model_part(variables)

    def encoded_conditions(self, params, batch) -> dict:
        """Encode each condition's own batch field with the tree's frozen towers."""
        tokens = {keyword: batch[condition.field]
                  for keyword, condition in self.inputs.conditions.items()}
        return self.encode(params["encoders"], tokens)

    def clean_samples(self, variables, batch, key, step: Step | None = None) -> jax.Array:
        """Return the batch's samples in [-1, 1], or under an autoencoder their latents drawn with `key`:
        in a step, each example's posterior draw its own (`Step.sample`)."""
        samples = unit_range(batch[self.inputs.sample.key])
        if self.autoencoder is None:
            return samples
        return self.autoencoder.encode(variables["autoencoder"], samples, key,
                                       noise=self._posterior_noise(step, "posterior", samples))

    def _posterior_noise(self, step: Step | None, name: str, pixels) -> jax.Array | None:
        """Each example's standard normal draw for its latent's posterior (`posterior_latent`), None
        outside a step or for a deterministic encoder (`AutoEncoder.stochastic`)."""
        if step is None or self.autoencoder is None or not self.autoencoder.stochastic:
            return None
        shape = self.autoencoder.latent_shape(tuple(pixels.shape[1:]))
        return step.sample(name, jax.random.normal, shape, dtype=jnp.float32, rows=pixels.shape[0])

    def denoiser(self, params, given, unconditional, process: Process | None = None):
        """Build `process`'s denoiser, the training process's by default, over the model's own
        collections.

        The unconditional branch is passed only when this objective is
        guided; without guidance the solver never evaluates it.
        """
        return (self.process if process is None else process).denoiser(
            self.model, self.model_variables(params), given, None if self.guidance is None else unconditional)

    def _conditions(self, params, batch, key, step: Step | None = None):
        """The batch's conditions and the blank ones, the blank aligned to and
        broadcast over the batch's rows."""
        given = self.encoded_conditions(params, batch)
        unconditional = jax.tree.map(lambda value, null: jnp.broadcast_to(null, value.shape),
                                     given, aligned_conditions(given, self.blank_conditions(params, given)))
        spatial = self._spatial(params, batch, key, step)
        return {**given, **spatial}, {**unconditional, **spatial}

    def _spatial(self, params, batch, key, step: Step | None = None) -> dict[str, jax.Array]:
        """The masked-image inputs an inpainting model reads in both guidance branches
        (`latent_image_conditions`), none where the inputs name no mask: in a step, each example's
        posterior draw of its masked image its own (`Step.sample`)."""
        if self.inputs.mask is None:
            return {}
        from dew.inputs.diffusion import latent_image_conditions

        pixels = unit_range(batch[self.inputs.sample.key])
        return latent_image_conditions(self.autoencoder, params["autoencoder"], pixels,
                                       batch[self.inputs.mask.key], jax.random.fold_in(key, 1),
                                       noise=self._posterior_noise(step, "masked image posterior", pixels))

    def _dropped(self, given, blank, batch, step: Step):
        """`given` with a drawn share of the batch's rows `blank`, classifier-free guidance's training
        dropout, each example's own draw (`Step.sample`)."""
        count = batch[self.inputs.sample.key].shape[0]
        # A float32 draw, as the step's others are, at any precision.
        probability = jnp.float32(self.unconditional_prob)
        dropped = step.sample("condition dropout", lambda key: jax.random.bernoulli(key, probability),
                              rows=count)
        return jax.tree.map(lambda value, null: jnp.where(expand(dropped, value), null, value), given, blank)

    def _sampling_batch(self, batch):
        fields = [condition.field for condition in self.inputs.conditions.values()]
        if self.inputs.mask is not None:
            fields.extend((self.inputs.sample.key, self.inputs.mask.key))
        return {name: batch[name] for name in fields}

    def loss(self, variables, batch, step: Step):
        images = unit_range(batch[self.inputs.sample.key])
        # The times, the noise, the posterior's draws and the dropped conditions are each example's own
        # (`Step.sample`); the model's dropout is the attempt's (`Step.key`).
        encode_key, drop_key, _, _, dropout_key = jax.random.split(step.key, 5)
        given, blank = self._conditions(variables, batch, drop_key, step)
        conditions = self._dropped(given, blank, batch, step)
        schedule = self.process.schedule
        count = images.shape[0]
        t = step.sample("times", lambda key: schedule.sample_t(key, 1)[0], rows=count)
        end_to_end = None
        if self.end_to_end is not None:
            end_to_end = self._end_to_end_latents(variables, images, encode_key, step, batch)
            samples = end_to_end.samples
        else:
            samples = self.clean_samples(variables, batch, encode_key, step)
        noise = step.sample("noise", jax.random.normal, samples.shape[1:], dtype=jnp.float32, rows=count)
        # The times are drawn in float32 and read at the samples' precision,
        # so a float64 run interpolates in float64.
        t = t.astype(jnp.promote_types(samples.dtype, t.dtype))

        call = {**conditions, "train": step.training, "rngs": model_rngs(dropout_key, training=step.training)}
        # The representation's features of the clean images, which both alignments read.
        targets = (None if self.alignment is None
                   else self.alignment.targets(variables[REPRESENTATION], images))
        losses, aligned = self._denoised(variables, self.model_variables(variables), samples, t, noise, call,
                                         targets, batch, step)
        weighted = losses * expand(step.boundary("loss weights", self.process.weight(t)), losses)
        if self.uncertainty is not None:
            head = {collection: variables[collection][UNCERTAINTY] for collection in ("params", "constants")}
            logvar = expand(self.uncertainty.apply(head, schedule.model_time(t)), losses)
            weighted = weighted * jnp.exp(-logvar) + logvar / 2
        denoising = self.row_mean(weighted, batch)
        total, mass = denoising.total, denoising.mass
        metrics: dict[str, jax.Array] = {}
        if self.alignment is not None and aligned is not None:
            # REPA adds proj_coeff times its mean to the denoising mean; the
            # denoising term here is halved, as Dew's L2 is, and so is this.
            metrics["alignment"] = aligned
            total = total + self.alignment.weight / 2 * aligned * mass
        if self.end_to_end is None or end_to_end is None:
            return Ratio(total, mass), Aux(metrics=metrics)
        # The autoencoder's update: its regularizer and the alignment of its
        # latent, read through the frozen model and projector in evaluation
        # mode (the batch norm on its running statistics, no condition
        # dropped), as REPA-E's `align_only` pass reads them, on the same
        # times and noise. That pass draws nothing, and its noised input and
        # target are its own, so its step marks no boundary a tape replaces.
        latents = self.end_to_end.normalized(end_to_end.raw, variables[LATENT_STATS])
        frozen = jax.lax.stop_gradient(variables)
        _, through = self._denoised(frozen, self.model_variables(frozen), latents, t, noise,
                                    {**given, "train": False}, targets, batch,
                                    Step(step.step, step.key, None))
        assert through is not None
        metrics.update(end_to_end.terms, autoencoder_alignment=through)
        autoencoder = end_to_end.regularizer + self.end_to_end.align_weight * through
        total = total + (autoencoder + end_to_end.hinge) * mass
        updates = {LATENT_STATS: end_to_end.statistics} if step.training else None
        return Ratio(total, mass), Aux(metrics=metrics, variables=updates)

    def _denoised(self, params, variables, samples, t, noise, call, targets, batch: Batch, step: Step):
        """The per-element denoising loss at `(t, noise)` and, under
        `alignment`, the REPA loss of the model's hidden tokens against the
        representation's `targets`. The noised input and the target are where
        the model's computation starts (`Step.boundary`)."""
        schedule = self.process.schedule
        rates = broadcast_rates(schedule, t, samples)
        noisy, c_in, target = self.process.prediction.forward_diffusion(samples, noise, rates)
        noisy, target = step.boundary("noised input", noisy), step.boundary("target", target)
        inputs = (variables, noisy * c_in, schedule.model_time(t))
        aligned = None
        if self.alignment is None:
            preds = self.model.apply(*inputs, **call)
        else:
            preds, captured = self.model.apply(*inputs, **call, mutable=["intermediates"],
                                               capture_intermediates=self.alignment.captures)
            aligned = self.alignment.loss({"params": params["params"][ALIGNMENT]}, self._captured(captured),
                                          targets, batch)
        preds = self.process.prediction.pred_transform(noisy, preds, rates, t)
        return optax.l2_loss(preds, target), aligned

    def _end_to_end_latents(self, params, images, key, step: Step, batch: Batch) -> TunedLatents:
        """The trained autoencoder's posterior draw of `images` and what the
        step reads of it."""
        assert self.end_to_end is not None and self.autoencoder is not None
        weights = params["params"][AUTOENCODER]
        moments = self.autoencoder.moments(weights, images)
        mean_shape = (*moments.shape[1:-1], moments.shape[-1] // 2)
        raw = posterior_latent(moments, key, step.sample("posterior", jax.random.normal, mean_shape,
                                                         dtype=jnp.float32, rows=moments.shape[0]))
        reconstruction = self.autoencoder.decode_raw(weights, raw)
        discriminator = params["params"].get(DISCRIMINATOR)
        regularizer, hinge, terms = self.end_to_end.regularizer(
            images, reconstruction, moments, perceptual=params.get(PERCEPTUAL),
            discriminator=None if discriminator is None else {"params": discriminator}, step=step.step,
            batch=batch)
        samples, statistics = jax.lax.stop_gradient(raw), params[LATENT_STATS]
        if step.training:
            samples, statistics = self.end_to_end.batch_normalized(samples, statistics, batch)
        else:
            samples = self.end_to_end.normalized(samples, statistics)
        return TunedLatents(raw, samples, statistics, regularizer, hinge, terms)

    def published_autoencoder(self, variables: Variables) -> tuple[AutoEncoder | None, Variables]:
        """Return the autoencoder a task over `variables` decodes with, and its weights under `autoencoder`.

        That is the frozen autoencoder, or under `end_to_end` the tuned one, with its
        latents normalized by the running statistics.
        """
        if self.end_to_end is None or self.autoencoder is None:
            return self.autoencoder, variables
        return self.end_to_end.tuned(self.autoencoder, variables)

    def _sample_impl(self, params, batch, key, *, count: int):
        given, unconditional = self._conditions(params, batch, key)
        process, times = sampling_grid(self.process, self.steps, self.grid)
        noise_key, sample_key = jax.random.split(key)
        x_T = process.noise(noise_key, (count, *self.latent_shape))
        denoise = self.denoiser(params, given, unconditional, process)
        autoencoder, published = self.published_autoencoder(params)
        walked = walk_and_decode(denoise, autoencoder, published, x_T, solver=self.solver,
                                 guidance=self.guidance, key=sample_key, times=np.asarray(times),
                                 final_denoise=self.final_denoise)
        return walked.images

    def _rows(self, batch) -> int:
        """How many samples a batch draws, one per real row: its images' count."""
        return batch[self.inputs.sample.key].shape[0]

    def _draw(self, params, batch, step: Step, limit: int | None = None) -> tuple[jax.Array, dict]:
        """Sample the batch's conditions on every rank, agreeing at each phase.

        The weights are the EMA copy when the step carries one and it is not
        the objective's reference (`_ema_is_reference`), else the live ones.
        `limit` caps the rows drawn, which is what a preview takes. Returns
        the samples and the condition tokens behind them.
        """
        weights = self.evaluation_variables(params, step)

        def setup() -> tuple[int, dict]:
            count, selected = self._rows(batch), self._sampling_batch(batch)
            if limit is not None:
                count = min(limit, count)
                selected = jax.tree.map(lambda value: value[:count], selected)
            return count, selected

        count, selected = agreed("diffusion sample setup", setup)
        samples = agreed("diffusion sample generation",
                         lambda: self._sample(weights, selected, step.key, count=count))
        return samples, {keyword: selected[condition.field]
                         for keyword, condition in self.inputs.conditions.items()}

    def evaluate(self, params, batch, step: Step):
        """Generate one sample for every real row, without decoding for display."""
        samples, _ = self._draw(params, batch, step)
        return self._grid(samples)

    def preview(self, params, batch, step: Step, *, scored=None):
        """Draw a separate small batch for display.

        It draws up to `VALIDATION_SAMPLES` rows on every process and gathers them to
        the host, and process zero decodes the captions; the other processes return
        None.
        """
        drawn = self._draw(params, batch, step, VALIDATION_SAMPLES)
        hosted = collective_host(drawn, phase="diffusion preview", held_by="first")
        if hosted is None:
            return None
        samples, tokens = hosted
        captions = ()
        for keyword, condition in self.inputs.conditions.items():
            captions = condition.encoder.captions(tokens[keyword])
            if captions:
                break
        return self._grid(samples, captions)


class FlowDistillationObjective(DiffusionObjective):
    """A few-step student of a flow teacher's model, trained on a loss of its own: rCM's and LADD's.

    `teacher` is the teacher's model, by default the student's own without
    its adapter. Its weights, frozen under `TEACHER`, are the starting tree's
    when it holds them, else `teacher_variables`, as a run config reads them
    from the teacher's run, else the model the starting tree holds
    (`teacher_weights`). A
    tree that lacks the loss's own `network` starts the student as the
    teacher, in buffers the step may donate, the weights under `FROZEN` in an
    adapter's split, which freezes all but its factors, beside that network
    (`network_variables`). Sampling runs
    `Consistency`, unguided.
    """

    network: ClassVar[str]
    """The network the loss trains beside the student under `params`."""

    def __init__(self, model: nn.Module, process: Process | Preset, inputs: InputSpec, *,
                 teacher: nn.Module | None = None, teacher_variables: Variables | None = None,
                 solver: Solver = _CONSISTENCY, **kwargs):
        name = type(self).__name__
        _own_loss(name, kwargs)
        super().__init__(model, process, inputs, solver=solver, **kwargs)
        if not (isinstance(self.process.schedule, FlowMatchingScheduler) and not self.process.interval
                and isinstance(self.process.prediction, FlowMatchPredictionTransform)):
            raise ValueError(f"{name} distills a velocity model on the linear path; build the process with "
                             "presets.Flow()")
        self.teacher = unadapted(self.model) if teacher is None else teacher
        self.teacher_variables = teacher_weights(self.variables, teacher_variables)

    def program_key(self) -> tuple[ProgramModule, ...]:
        """The student, then the frozen teacher."""
        return (*super().program_key(), ProgramModule(self.teacher, None, trained=False))

    def substitute(self, modules: Sequence[nn.Module]) -> None:
        self.model, self.teacher = modules

    def held_variables(self, *, beside: Variables | None = None) -> Variables:
        return {**super().held_variables(beside=beside), TEACHER: self.teacher_variables}

    def complete_variables(self, key: jax.Array, tree: Variables) -> Variables:
        """Start the student as the tree's teacher and add the loss's `network`, unless the tree holds it."""
        state = super().complete_variables(key, tree)
        if self.network in state["params"]:
            return state
        copied = dict(jax.tree.map(jnp.copy, dict(state[TEACHER])))
        if FROZEN in state:
            copied[FROZEN] = copied.pop("params")
        state = merge(state, copied)
        return merge(state, self.network_variables(key, state))

    @abstractmethod
    def network_variables(self, key: jax.Array, state: Variables) -> Variables:
        """Return the loss's `network`'s variables by collection, beside `state`'s student started
        as the teacher."""
