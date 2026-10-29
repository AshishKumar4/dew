"""The diffusion run as one typed record, and the one function that builds it.

`DiffusionRunConfig` is what a diffusion recipe parses from its command line
and writes as `run.json` next to the checkpoints. `build()` is the single
function that turns it into a `DiffusionObjective`: the recipe trains the
objective it returns, and `TextToImage.from_run` samples from the objective it
returns for the same file.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, ClassVar, TypedDict

import jax
import numpy as np
import tyro

from dew._records import number, record
from dew.batch import Variables, merge
from dew.checkpoints import Checkpoints
from dew.config import MetricConfig, ObjectiveConfig, Prepared, RunConfig
from dew.config.media import media_field
from dew.diffusion.presets import EDM, Flow, build_process
from dew.diffusion.process import Process
from dew.inputs import AudioCondition, Field, InputSpec, TextCondition
from dew.nn.attention import attention_implementation
from dew.nn.autoencoders import AutoEncoder
from dew.nn.autoencoders.pretrained import PretrainedAutoencoder
from dew.nn.protocols import IntervalModel, TimeScaled
from dew.objectives.base import TEACHER, Metric
from dew.objectives.diffusion.alignment import Alignment
from dew.objectives.diffusion.end_to_end import AUTOENCODER
from dew.objectives.diffusion.few_step import SMOOTH_TIME_SCALE, MeanFlowObjective
from dew.objectives.diffusion.guidance_distillation import GuidanceDistillationObjective
from dew.objectives.diffusion.objective import (
    LOSS_HEADS,
    DiffusionObjective,
    FlowDistillationObjective,
    teacher_weights,
)
from dew.registry import (
    Configured,
    DtypeName,
    ModelConfig,
    dtype_name,
    encoders,
    from_record,
    metrics,
    models,
    objectives,
    presets,
    resolve_dtype,
    to_record,
)
from dew.sampling.solvers import EulerAncestral, Solver

if TYPE_CHECKING:

    from dew.diffusion.presets import Preset
    from dew.objectives.rl.flow import FlowGRPOObjective, FlowRollout

    # A kind's `union` imports every aliased class, so the run builds it at
    # the end of the module, where the annotations, read when a command line
    # is parsed, find it. Statically the fields are typed as the kinds' base
    # classes, which a reader and a checker need.
    PresetSpec = Preset

# Every other dial of a stage is `dew.nn.attention_blocks.Stage`'s own default, and
# a stage that names one restates it.
ATTENTION = {"heads": 8}

# The default unet has attention everywhere but the full-resolution stage,
# where it costs the most. Every other architecture takes its own kwargs as
# JSON.
DEFAULT_MODEL_CONFIG = {
    "attention_configs": [None, ATTENTION, ATTENTION, ATTENTION],
    "precision": "default",
    "dtype": "bfloat16",
}


@dataclasses.dataclass(frozen=True)
class FlowRolloutConfig:
    """A run's Flow-GRPO rollout: `groups` samples of each prompt over `steps` time points, the first
    `train_steps` transitions trained (all of them when None), each scored by the image metric
    `reward` names, an alias or an import path, against its own prompt, higher being better
    (`clip_score`). `build` makes the `FlowRollout` over an objective."""

    reward: str = "clip_score"
    groups: int = 4
    steps: int = 11
    train_steps: int | None = None

    def __post_init__(self) -> None:
        if ":" not in self.reward and self.reward not in metrics:
            raise ValueError(f"reward names {self.reward!r}, which no metric alias names; the aliases are "
                             f"{sorted(metrics)}, or name a metric by its import path")

    def build(self, objective: FlowGRPOObjective) -> FlowRollout:
        """The rollout over `objective`, scoring each sample with the metric's per-sample measure."""
        from dew.artifacts import ImageGrid
        from dew.eval.common import ImageMetric
        from dew.objectives.rl.flow import FlowRollout

        metric = metrics[self.reward]()
        if not isinstance(metric, ImageMetric):
            raise ValueError(f"reward {self.reward!r} is a {type(metric).__name__}, and Flow-GRPO "
                             "scores each sample with an image metric's per-sample measure")

        def reward(images, batch):
            return np.asarray(metric.fn(ImageGrid(images), batch))

        return FlowRollout(objective, reward, groups=self.groups, steps=self.steps,
                           train_steps=self.train_steps)


class TeacherArguments(TypedDict, total=False):
    """What a distillation objective takes of its teacher's run (`DiffusionRunConfig.teacher_run`)."""

    teacher_variables: Variables
    """The tree the teacher's run published, where the starting tree holds no teacher."""
    teacher: DiffusionObjective
    """Guidance distillation's teacher, its objective rebuilt from the run's record."""


@dataclasses.dataclass(frozen=True)
class DiffusionRunConfig(RunConfig):
    """A diffusion run's configuration: the shared run fields plus the diffusion objective's own settings."""

    _FLAG_SELECTED: ClassVar[Mapping[str, str]] = {**RunConfig._FLAG_SELECTED, "rollout": "rollout"}

    model: ModelConfig = dataclasses.field(
        default_factory=lambda: ModelConfig("unet", dict(DEFAULT_MODEL_CONFIG)))
    preset: PresetSpec | None = dataclasses.field(default_factory=EDM)
    """The convention the model is trained and sampled with.

    None uses the one the `pretrained` pipeline's scheduler reads, which a preset
    may restate.
    """
    objective: ObjectiveConfig = dataclasses.field(default_factory=lambda: ObjectiveConfig(
        "diffusion", {"solver": to_record(EulerAncestral(), Solver)}))
    """The objective and its arguments: the denoising loss (`diffusion`) with its
    EMA, condition dropout and validation sampling (`--objective.ema-decay`,
    `--objective.guidance`, `--objective.steps`), or a loss of its own in its
    place, as `--objective mean_flow --objective.omega 2` or `--objective rcm
    --teacher-run runs/teacher`. Each objective refuses a preset or a
    guidance its loss cannot train or sample with.
    """
    text: TextCondition | None = dataclasses.field(default_factory=TextCondition)
    """The text condition, passed under its encoder's keyword (`ConditionEncoder.keyword`).

    None trains unconditionally, or on `audio`.
    """
    audio: AudioCondition | None = None
    """The audio condition, passed under its encoder's keyword in place of text.

    It needs `text` to be None and a chain of clips that hold their audio
    (`dew.data.presets.clips(sample_rate=)`).
    """
    autoencoder: PretrainedAutoencoder | None = None
    """The autoencoder for latent diffusion; None trains in pixel space."""
    pretrained: str | None = None
    """A published diffusion pipeline to fine-tune.

    It is a Hub repo, `repo@revision` or a local directory in the diffusers layout.
    The checkpoint decides the model, its text conditioning and its autoencoder, so
    `--model` holds only the precision settings and `text` and `autoencoder` stay
    unset.
    """
    pretrained_param_dtype: DtypeName = "float32"
    """The dtype `pretrained`'s parameters are stored and trained in."""
    teacher_run: str | None = None
    """The run directory a distillation's teacher trained in. Its published weights are the frozen
    teacher's (`teacher_variables=`), unless the starting tree holds the teacher; guidance distillation
    also rebuilds the teacher's objective from the run's record (`teacher=`). Only a distillation
    takes one."""
    rollout: Annotated[FlowRolloutConfig, tyro.conf.subcommand("rollout")] | None = None
    """The rollout a Flow-GRPO run trains on: its reward metric, group size, time points and trained
    transitions, `--rollout.reward clip_score --rollout.groups 4`. A Flow-GRPO run without one takes
    `FlowRolloutConfig()`, which its record then states; another objective trains on its data and
    takes none."""
    def __post_init__(self) -> None:
        super().__post_init__()
        if not issubclass(objectives[self.objective.name], DiffusionObjective):
            raise ValueError(f"--objective {self.objective.name} trains no diffusion model")
        from dew.objectives.rl.flow import FlowGRPOObjective

        flow = issubclass(objectives[self.objective.name], FlowGRPOObjective)
        if flow and self.rollout is None:
            object.__setattr__(self, "rollout", FlowRolloutConfig())
        if not flow and self.rollout is not None:
            raise ValueError(f"--rollout configures a Flow-GRPO run's rollout, and --objective "
                             f"{self.objective.name} trains on its data")
        if self.pretrained is None and self.pretrained_param_dtype != "float32":
            raise ValueError("--pretrained-param-dtype stores a --pretrained pipeline's parameters; "
                             "this run loads none")

        if self.pretrained is not None:
            # The pipeline's own denoiser trains; the model flags it reads are
            # its compute dtype and attention kernel.
            scratch = ModelConfig("unet", dict(DEFAULT_MODEL_CONFIG))
            reads = ("dtype", "attention_impl")
            chosen = [name for name, named in (
                ("model", self.model.name != scratch.name or any(
                    self.model.fields.get(key) != scratch.fields.get(key)
                    for key in set(self.model.fields) | set(scratch.fields) if key not in reads)),
                ("text", self.text not in (None, TextCondition())),
                ("audio", self.audio is not None),
                ("autoencoder", self.autoencoder is not None)) if named]
            if chosen:
                raise ValueError(
                    f"{self.pretrained} decides the model, its text conditioning and its "
                    f"autoencoder; leave {', '.join(chosen)} unset")
            object.__setattr__(self, "text", None)
        elif self.preset is None:
            raise ValueError("preset None trains on the convention a pretrained pipeline's "
                             "scheduler reads; name a preset or --pretrained")
        # EDM's sigma draw is the space's: pixels without an autoencoder,
        # latents with one. A regime or sigmas the preset states win.
        if (isinstance(self.preset, EDM) and self.preset.regime is None
                and (self.preset.P_mean is None or self.preset.P_std is None)):
            latent = self.autoencoder is not None or self.pretrained is not None
            object.__setattr__(self, "preset", dataclasses.replace(
                self.preset, regime="latent" if latent else "pixel"))
        # A resolution shift is the data's: the token count of its images.
        if (isinstance(self.preset, Flow) and self.preset.resolution_shift is not None
                and self.preset.resolution_shift.tokens is None and self.data is not None):
            height, width = self.sample_field().shape[-3:-1]
            object.__setattr__(self, "preset", dataclasses.replace(
                self.preset, resolution_shift=self.preset.resolution_shift.at(height, width)))
        if self.audio is not None and self.text is not None:
            raise ValueError(
                "the models take one context under textcontext, and this run names both "
                "text and audio; set text to None to condition on audio")
        if self.audio is not None and self.data is not None and "audio" not in record(
                self.chain().facts()["fields"], "the chain's fields"):
            raise ValueError("audio conditioning reads its clips' audio, and this chain's clips carry none "
                             "(dew.data.presets.clips(sample_rate=))")

    def sample_field(self) -> Field:
        """Return the batch field the model generates, at the resolution the data comes in
        (`Dataset.facts`): a clip's video, else an image. Raises ValueError for a chain of neither, or
        one whose size varies."""
        return media_field(self.chain(), "diffusion")

    def model_fields(self, autoencoder: AutoEncoder | None) -> dict[str, Configured]:
        """Return the fields the run derives for the model, which `ModelConfig.build`
        takes over the model's `arguments`.

        These are the channels the model denoises when the architecture takes them
        as `output_channels`. The published families name theirs as their sources
        do, in `model.fields`. On the model those build, an `IntervalModel` embeds
        the duration under an interval process, and MeanFlow, whose loss
        differentiates in time, turns a `TimeScaled` model's time features at
        `SMOOTH_TIME_SCALE` unless `model.fields` names a scale.
        """
        fields: dict[str, Configured] = {}
        if "output_channels" in {field.name for field in dataclasses.fields(models[self.model.name])}:
            fields["output_channels"] = (self.sample_field().shape[-1] if autoencoder is None
                                         else autoencoder.latent_channels)
        model = self.model.build(**fields)
        if isinstance(model, IntervalModel) and self.preset is not None:
            built = self.preset()
            fields["interval"] = isinstance(built, Process) and built.interval
        if (issubclass(objectives[self.objective.name], MeanFlowObjective) and isinstance(model, TimeScaled)
                and "time_scale" not in self.model.fields):
            fields["time_scale"] = SMOOTH_TIME_SCALE
        return fields

    @property
    def context(self) -> TextCondition | AudioCondition | None:
        """Return the condition the model reads, text or audio, if any."""
        return self.text if self.text is not None else self.audio

    def build(self, *, variables: Variables | None = None) -> DiffusionObjective:
        """Build the objective, with each component built around its parameters.

        Supplied variables are the authoritative saved snapshot. The encoders and the
        VAE read only configuration and tokenizer metadata, and use their subtrees of
        the supplied variables without loading source weights or casting storage.

        A `lora` goes on the denoiser that `pretrained` loads, or, from scratch, on a
        fresh draw of it from the run's key. The objective then trains the adapter's
        factors and any loss head of its own.
        """
        if self.lora is not None and variables is not None:
            raise ValueError("a --lora run's saved variables hold its factors, which "
                             "TextToImage.from_run binds through the run's own adapter record")
        objective = self._objective(variables)
        if self.lora is None or self.pretrained is not None:
            return objective
        # The objective's own init draws the denoiser beside its heads and
        # towers; the adapter freezes the denoiser's weights, and the heads
        # stay under `params` to train, beside their own `constants`. The
        # adapted denoiser takes the drawn one's place (`substitute`), and the
        # towers, the autoencoder and any frozen teacher stay as built.
        key = jax.random.key(self.trainer.key)
        drawn = objective.init(key)
        adapter = self.lora.apply(objective.model, objective.model_variables(drawn),
                                  key=jax.random.fold_in(key, 1))
        heads = {collection: {name: tree for name, tree in drawn[collection].items() if name in LOSS_HEADS}
                 for collection in ("params", "constants") if collection in drawn}
        objective.substitute((adapter.model, *(entry.module for entry in objective.program_key()[1:])))
        objective.variables = merge({**drawn, **adapter.variables}, heads)
        return objective

    def _objective(self, variables: Variables | None) -> DiffusionObjective:
        """The configured objective over `variables`."""
        if self.pretrained is None:
            self.built_fresh()
            model, conditions, autoencoder = self._scratch(variables)
            sample, convention = self.sample_field(), None
        else:
            # A REPA-E run keeps its tuned autoencoder under `params`, where the pipeline's
            # metadata binds it at the pipeline's own `autoencoder`; the trained tree is kept whole.
            trained = variables
            source = self._source(None if variables is None else
                                  {**variables, "autoencoder": self._autoencoder_params(variables)})
            if self.lora is not None:
                # The adapter binds to the denoiser and the pipeline's
                # weights, so the objective trains its factors alone.
                source = source.adapt(self.lora, key=self.trainer.key)
            if source.inputs is None:
                raise ValueError(f"{self.pretrained} loads no diffusion inputs to train on")
            if source.inputs.mask is not None:
                raise ValueError(f"{self.pretrained} is an inpainting pipeline, which trains "
                                 "on masks an image dataset does not carry")
            # The objective is built over the pipeline itself, which brings the sampling policy it
            # samples by (`DiffusionObjective`'s `model`); the run's own process, inputs and tree
            # stand over the pipeline's.
            model, conditions, autoencoder = source, dict(source.inputs.conditions), source.autoencoder
            variables, convention = source.variables if trained is None else trained, source.process
            # The data's geometry, in the autoencoder's own pixel channels:
            # Qwen-Image 2.1's are RGBA.
            sample = source.inputs.sample
        inputs = InputSpec(sample=sample, conditions=conditions)
        objective = self.objective.build(model=model, process=self._process(convention), inputs=inputs,
                                         autoencoder=autoencoder, variables=variables,
                                         **self._teacher(variables))
        assert isinstance(objective, DiffusionObjective)
        return objective

    def _teacher(self, variables: Variables | None) -> TeacherArguments:
        """The teacher a distillation reads from `teacher_run`: the tree its run published, unless the
        starting tree holds the teacher (`TEACHER`), and for guidance distillation the teacher's
        objective rebuilt from the run's record over the tree the student holds frozen
        (`teacher_weights`)."""
        if self.teacher_run is None:
            return {}
        kind = objectives[self.objective.name]
        if not issubclass(kind, FlowDistillationObjective | GuidanceDistillationObjective):
            raise ValueError(f"teacher_run names a distillation's teacher, and {self.objective.name} "
                             f"distills none")
        published: Variables | None = None
        if variables is None or TEACHER not in variables:
            # A resumed student holds its teacher, so only a fresh one reads the teacher's run.
            with Checkpoints(self.teacher_run) as checkpoints:
                published = checkpoints.variables(ema=None, step=None, mesh=None, layout=None,
                                                  param_dtype=None)
        given: TeacherArguments = {} if published is None else {"teacher_variables": published}
        if issubclass(kind, GuidanceDistillationObjective):
            held = teacher_weights(variables, published, whole=True)
            given["teacher"] = DiffusionRunConfig.load(self.teacher_run).build(variables=held)
        return given

    def build_run(self) -> Prepared:
        """The objective `build` makes, with the run's Hub sources pinned
        (`pinned`), on its data, whose captions the objective's conditions
        tokenize. `val_metrics` score each validation, and Flow-GRPO samples
        the batches it trains on (`rollout`)."""
        from dew.objectives.rl.flow import FlowGRPOObjective

        run = self
        objective = run.build()
        # Compiled once: `train` reads the run's length and cadences off this loader and trains on it.
        dataset, held = run.compiled(objective.data(), run.chain(), run.validation)
        metrics = run.validation_metrics(validates=bool(held))
        rollout = None
        if isinstance(objective, FlowGRPOObjective) and run.rollout is not None:
            rollout = run.rollout.build(objective)
        return Prepared(run, lambda name: run.train(objective, dataset, name=name, metrics=metrics,
                                                    rollout=rollout, validation=held))

    def pinned(self) -> DiffusionRunConfig:
        """Return this run with its Hub sources pinned to the commits they resolve to now: an imported
        adapter's (`RunConfig.pinned`), `pretrained` and the alignment's encoder, so the record names
        the weights the run started from.
        """
        from dew.interop import sources

        def pin(source: str) -> str:
            return sources.SourceReference.parse(source).pinned().spelled()

        objective = self.objective
        stated = objective.fields.get("alignment")
        if isinstance(stated, Mapping):
            alignment = from_record(Alignment, stated)
            if alignment.source is not None:
                objective = dataclasses.replace(objective, fields={**objective.fields, "alignment": to_record(
                    dataclasses.replace(alignment, source=pin(alignment.source)), Alignment)})
        return dataclasses.replace(super().pinned(), objective=objective,
                                   pretrained=None if self.pretrained is None else pin(self.pretrained))

    def _scratch(self, variables: Variables | None):
        """The registry's model, the run's text or audio condition and its
        autoencoder."""
        autoencoder = (None if self.autoencoder is None else self.autoencoder.build(
            params=None if variables is None else self._autoencoder_params(variables)))
        conditions = {}
        if self.context is not None:
            keyword = encoders[self.context.encoder].keyword
            params = None if variables is None else variables["encoders"][keyword]
            if self.text is not None:
                conditions[keyword] = self.text.build(params=params, dtype=self._compute)
            else:
                assert self.audio is not None
                seconds = number(self.chain().facts().get("audio_seconds"), "the clips' audio_seconds")
                conditions[keyword] = self.audio.build(seconds, params=params, dtype=self._compute)
        model = self.model.build(**self.model_fields(autoencoder))
        return model, conditions, autoencoder

    def _autoencoder_params(self, variables: Variables) -> Variables:
        """The autoencoder's weights in a saved tree: frozen beside the
        model, or trained under `params` when REPA-E tuned it."""
        if self.objective.fields.get("end_to_end") is not None:
            return variables["params"][AUTOENCODER]
        return variables["autoencoder"]

    @property
    def _compute(self) -> DtypeName | None:
        """The compute dtype the model's record names, which the text and audio
        encoders and a pretrained pipeline compute in too."""
        held = self.model.fields.get("dtype")
        return None if held is None else dtype_name(resolve_dtype(str(held)))

    def _source(self, variables: Variables | None):
        """The `pretrained` pipeline at the data's resolution: its own
        weights, or `variables` bound over its metadata."""
        from dew.interop.pretrained import load_diffusion_source
        from dew.interop.sources import SourceReference

        assert self.pretrained is not None
        reference = SourceReference.parse(self.pretrained)
        # An adapter keeps the pipeline's weights as they are: they load where the start keeps them.
        # The pipeline reads its weights on a resume too, which binds the checkpoint over them.
        placement = self.source_keywords(frozen=self.lora is not None)
        return load_diffusion_source(
            reference.name, revision=reference.revision, dtype=self._compute or "bfloat16",
            param_dtype=self.pretrained_param_dtype,
            attention_impl=attention_implementation(str(self.model.fields.get("attention_impl", "auto"))),
            size=self.sample_field().shape[:-1],
            variables=variables, mesh=placement.get("mesh"), layout=placement.get("layout"),
            text_only=placement.get("text_only", False), prediction_depths=placement.get("prediction_depths"),
            modeling_overrides=placement.get("modeling_overrides"))

    def _process(self, convention: Process | None) -> Process:
        """The preset's process, which must be of the kind the pretrained
        pipeline's scheduler reads: the same schedule family and the same
        prediction. None is that scheduler's own."""
        if self.preset is None:
            assert convention is not None
            return convention
        process = build_process(self.preset)
        if convention is not None and (
                type(process.schedule) is not type(convention.schedule)
                or type(process.prediction) is not type(convention.prediction)):
            raise ValueError(
                f"{self.pretrained}'s scheduler reads {type(convention.schedule).__name__} with "
                f"{type(convention.prediction).__name__}, and preset "
                f"{type(self.preset).__name__} is "
                f"{type(process.schedule).__name__} with {type(process.prediction).__name__}; "
                "name a preset of its kind, or none for the scheduler's own")
        return process

    def default_metrics(self, *, validates: bool) -> tuple[MetricConfig, ...]:
        """CLIP for an image run whose data `validates`, and nothing otherwise."""
        return (MetricConfig("clip"),) if validates and len(self.sample_field().shape) != 4 else ()

    def metric(self, config: MetricConfig) -> Metric:
        """`config` built, loading its own weights: a video run scores a `VideoGrid` against its `video`
        field, which a PSNR or SSIM that states no field of its own reads there, and an image-only
        metric raises a ValueError naming it here, before the trainer starts."""
        from dew.artifacts import ImageGrid, VideoGrid

        video = len(self.sample_field().shape) == 4
        member = metrics[config.name]
        if video and member in (metrics["fid"], metrics["clip"], metrics["clip_score"]):
            raise ValueError(
                f"metric {config.name!r} reads an ImageGrid and this run samples video; "
                "validate a video run with psnr or ssim")
        grids = member in (metrics["psnr"], metrics["ssim"])
        return config.build(**({"field": "video" if video else "image",
                                "reads": VideoGrid if video else ImageGrid} if grids else {}))

if not TYPE_CHECKING:
    PresetSpec = presets.union


__all__ = ["DiffusionRunConfig", "FlowRolloutConfig"]
