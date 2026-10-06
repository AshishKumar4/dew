"""The diffusion run as one typed record, and the one function that builds it.

`DiffusionRunConfig` is what a diffusion recipe parses from its command line
and writes as `run.json` next to the checkpoints. `build()` is the single
function that turns it into a `DiffusionObjective`: the recipe trains the
objective it returns, and `TextToImage.from_run` samples from the objective it
returns for the same file.
"""

from __future__ import annotations

import dataclasses
import os
from importlib import import_module
from typing import TYPE_CHECKING, ClassVar

import jax
import numpy as np

from dew.config import ModelConfig, RunConfig
from dew.data import ImageDataset, OnlineImages, OnlineVideos, TFDSImages, VideoDataset
from dew.diffusion.presets import EDM, Flow, build_process
from dew.diffusion.process import Process
from dew.inputs import Condition, Field, InputSpec, rebuild
from dew.nn.autoencoders import AutoEncoder
from dew.nn.protocols import IntervalModel, TimeScaled
from dew.nn.text_encoders import DEFAULT_MODEL
from dew.objectives.base import FROZEN, Variables, merge
from dew.registry import DtypeName, datasets, encoders, metrics, models, presets, solvers, trainings
from dew.sampling.guidance import CFG
from dew.sampling.solvers import EulerAncestral

from .end_to_end import AUTOENCODER
from .few_step import SMOOTH_TIME_SCALE, MeanFlowTraining
from .objective import LOSS_HEADS, Denoising, DiffusionObjective, Training

import_module("dew.eval")  # registers the image metrics
import_module("dew.nn.backbones")  # registers the models before the config's unions are built
for _mode in ("adversarial", "consistency", "guidance_distillation"):
    # Each registers its training mode before `mode`'s union is built.
    import_module(f"dew.objectives.diffusion.{_mode}")

if TYPE_CHECKING:
    from flax import linen as nn

    from dew.diffusion.presets import Preset
    from dew.objectives.rl.flow import FlowGRPOObjective, FlowRollout
    from dew.sampling.solvers import Solver

    # A registry's `union` is built from what has registered by import time, so
    # only the run sees it. Statically the fields are typed as every member
    # of those tables, which a reader and a checker need.
    PresetSpec = Preset
    SolverSpec = Solver
    TrainingSpec = Training
    # The run reads captions through `load(tokenize=)`, which the token
    # datasets do not take; `sample_field` refuses those at runtime.
    CaptionedSpec = ImageDataset | OnlineImages | VideoDataset
else:
    PresetSpec = presets.union
    SolverSpec = solvers.union
    TrainingSpec = trainings.union
    CaptionedSpec = datasets.union

# Every other dial of a stage is `dew.nn.attention.Stage`'s own default, and
# a stage that names one restates it.
ATTENTION = {"heads": 8}

# The default unet has attention everywhere but the full-resolution stage,
# where it costs the most. Every other architecture takes its own kwargs as
# JSON.
DEFAULT_MODEL_CONFIG = {
    "attention_configs": [None, ATTENTION, ATTENTION, ATTENTION],
    "precision": "default",
}


@dataclasses.dataclass(frozen=True)
class TextCondition:
    """Text conditioning: which registered encoder reads the batch's tokens, and from which checkpoint."""

    encoder: str = "clip_text"
    checkpoint: str = DEFAULT_MODEL
    dtype: DtypeName | None = None
    """The encoder's compute dtype; None follows the model's compute dtype."""
    param_dtype: DtypeName = "float32"
    """The storage dtype when loading source weights; supplied params keep theirs."""
    field: str = "text"
    """The batch field holding the tokenized text."""
    unconditional: str = ""
    """The prompt the unconditional branch is encoded from."""
    max_length: int | None = None
    """The token length every prompt is padded to.

    None keeps the encoder's own default, which for CLIP is the checkpoint's
    context length.
    """
    revision: str | None = None
    """The checkpoint's git revision.

    With it, a rerun conditions on the weights the run named, even after the
    branch has moved on.
    """

    def build(self, *, params: Variables | None = None,
              dtype: DtypeName | None = None) -> Condition:
        """Build the encoder over supplied params, without loading source weights or casting storage.

        `dtype` is the run's own compute dtype, which an unset `self.dtype` follows:
        the tower runs next to the model it conditions in every step, so a checkpoint
        stored in float32 is no reason to run it in float32.
        """
        fields = {name: value for name, value in
                  (("max_length", self.max_length), ("revision", self.revision))
                  if value is not None}
        return Condition(
            rebuild(self.encoder, {"checkpoint": self.checkpoint,
                                   "dtype": dtype if self.dtype is None else self.dtype,
                                   "param_dtype": self.param_dtype, **fields}, params=params),
            field=self.field, unconditional=self.unconditional)


@dataclasses.dataclass(frozen=True)
class AudioCondition:
    """Audio conditioning on each clip, through the video dataset's own audio model.

    The dataset's `audio_model` names the `hf_audio` tower, whose feature
    extractor already wrote the batch's `audio` field. The dataset's clip length
    sets the waveform length used to encode every clip and the silent
    unconditional input, so the two always agree.
    """

    encoder: ClassVar[str] = "hf_audio"
    dtype: DtypeName | None = None
    """The tower's compute dtype; None follows the model's compute dtype."""
    param_dtype: DtypeName = "float32"
    """The storage dtype when loading source weights; supplied params keep theirs."""

    def build(self, clips: VideoDataset, *, params: Variables | None = None,
              dtype: DtypeName | None = None) -> Condition:
        """Build the tower over supplied params, without loading source weights or casting storage."""
        return Condition(
            rebuild(self.encoder, {"checkpoint": clips.audio_model, "seconds": clips.audio_seconds,
                                   "dtype": dtype if self.dtype is None else self.dtype,
                                   "param_dtype": self.param_dtype}, params=params),
            field="audio", unconditional=0.0)


@dataclasses.dataclass(frozen=True)
class PretrainedAutoencoder:
    """A published autoencoder for latent diffusion.

    It is a Stable Diffusion AutoencoderKL, or a DC-AE when the checkpoint's
    config names `AutoencoderDC` (`dew.nn.autoencoders.pretrained.load_autoencoder`).
    """

    modelname: str = "pcuenq/sd-vae-ft-mse-flax"
    revision: str = "bf16"
    dtype: DtypeName = "bfloat16"
    latent_shift: float | None = None
    latent_scale: float | None = None
    """Per-dataset latent statistics; None keeps the checkpoint's."""

    def build(self, *, params: Variables | None = None) -> AutoEncoder:
        """Build the autoencoder from its config over supplied params."""
        import jax.numpy as jnp

        from dew.nn.autoencoders.pretrained import load_autoencoder

        return load_autoencoder(self.modelname, revision=self.revision, dtype=jnp.dtype(self.dtype),
                                latent_shift=self.latent_shift, latent_scale=self.latent_scale,
                                params=params)


@trainings("flow_grpo")
@dataclasses.dataclass(frozen=True)
class FlowGRPO(Training):
    """Flow-GRPO training of the model as a policy on an image reward.

    Flow-GRPO is from Liu et al. (2025). For each prompt, a group of rollouts runs
    through the flow SDE; their rewards are normalized within the group, and the
    model trains on the clipped likelihood ratio, with the conditional KL to the
    initial model at weight `beta`.

    The fields are `FlowGRPOObjective`'s and `FlowRollout`'s, which document them.
    `reward` names a registered image metric that scores each sample against its
    own prompt, where higher must be better (`clip_score`).

    Under Flow-GRPO, the run's `ema_decay` and `unconditional_prob` are unused: the
    EMA slot holds the frozen KL reference when `beta` > 0 and nothing otherwise,
    and no training row drops its condition.
    """

    reward: str = "clip_score"
    noise_level: float = 0.7
    beta: float = 0.0
    clip_range: float = 1e-4
    adv_clip_max: float = 5.0
    groups: int = 4
    rollout_steps: int = 11
    train_steps: int | None = None

    def __post_init__(self) -> None:
        if self.reward not in metrics:
            raise ValueError(f"reward names {self.reward!r}, which no metric is registered "
                             f"under; the registered metrics are {sorted(metrics)}")

    def objective(self, run: DiffusionRunConfig, model: nn.Module, process: Process, inputs: InputSpec, *,
                  base: nn.Module, autoencoder: AutoEncoder | None,
                  variables: Variables | None) -> FlowGRPOObjective:
        from dew.objectives.rl.flow import FlowGRPOObjective
        from dew.sampling.flow import FlowSDE

        return FlowGRPOObjective(
            model, process, inputs, sde=FlowSDE(self.noise_level), beta=self.beta, clip_range=self.clip_range,
            adv_clip_max=self.adv_clip_max, autoencoder=autoencoder, guidance=run.guidance, solver=run.solver,
            steps=run.sampling_steps, variables=variables)

    def rollout(self, objective: DiffusionObjective) -> FlowRollout:
        """Return the trainer's rollout over `objective`, scored by the named metric."""
        from dew.artifacts import ImageGrid
        from dew.eval.common import ImageMetric
        from dew.objectives.rl.flow import FlowGRPOObjective, FlowRollout

        if not isinstance(objective, FlowGRPOObjective):
            raise TypeError(f"Flow-GRPO rolls out the FlowGRPOObjective it builds, "
                            f"not a {type(objective).__name__}")

        metric = metrics[self.reward]()
        if not isinstance(metric, ImageMetric):
            raise ValueError(f"reward {self.reward!r} is a {type(metric).__name__}, and Flow-GRPO "
                             "scores each sample with an image metric's per-sample measure")

        def reward(images, batch):
            return np.asarray(metric.fn(ImageGrid(images), batch))

        return FlowRollout(objective, reward, groups=self.groups, steps=self.rollout_steps,
                           train_steps=self.train_steps)



@dataclasses.dataclass(frozen=True)
class DiffusionRunConfig(RunConfig):
    """A diffusion run's configuration: the shared run fields plus the diffusion objective's own settings."""

    objective: str = "diffusion"
    """The name of the objective `build` returns, which is the name `mode` is registered under.

    It is set from `mode` when the config is created, so a saved record names what
    trained.
    """
    model: ModelConfig = dataclasses.field(
        default_factory=lambda: ModelConfig("unet", dict(DEFAULT_MODEL_CONFIG)))
    data: CaptionedSpec = dataclasses.field(default_factory=TFDSImages)
    preset: PresetSpec | None = dataclasses.field(default_factory=EDM)
    """The convention the model is trained and sampled with.

    None uses the one the `pretrained` pipeline's scheduler reads, which a preset
    may restate.
    """
    solver: SolverSpec = dataclasses.field(default_factory=EulerAncestral)
    """The solver validation samples with."""
    guidance: CFG | None = dataclasses.field(default_factory=lambda: CFG(3.0))
    """How validation samples are guided, with scale and interval.

    None samples the conditional prediction alone.
    """
    sampling_steps: int = 200
    unconditional_prob: float = 0.12
    """The fraction of training examples whose condition is dropped."""
    ema_decay: float | None = 0.999
    """The EMA decay; None disables the EMA and 1.0 keeps a frozen copy."""
    text: TextCondition | None = dataclasses.field(default_factory=TextCondition)
    """The text condition, passed under its encoder's keyword (`ConditionEncoder.keyword`).

    None trains unconditionally, or on `audio`.
    """
    audio: AudioCondition | None = None
    """The audio condition, passed under its encoder's keyword in place of text.

    It needs `text` to be None and a `VideoDataset`, whose clips hold the audio.
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
    mode: TrainingSpec = dataclasses.field(default_factory=Denoising)
    """How the run trains, with the denoising loss or with another loss in its place.

    `Denoising` is the denoising loss, optionally with EDM2's learned weighting or
    representation alignment. `FlowGRPO`, `MeanFlowTraining`, `ShortcutTraining`,
    `ConsistencyDistillation`, `GuidanceDistillation` and `AdversarialDistillation`
    train on their own losses. Each mode refuses a preset or guidance that its loss
    cannot train or sample with, and `objective` is set to the name the mode is
    registered under. On the command line, pick one with
    `mode:mean-flow-training --mode.omega 2`.
    """
    val_metrics: tuple[str, ...] = ("clip",)
    """Names in the metrics registry, scored on every validation pass.

    The registry lists what a run can name, so a metric registered elsewhere can
    be named here without this class knowing about it. `__post_init__` refuses a
    name that nothing is registered under.
    """

    def __post_init__(self) -> None:
        # A record carries every sequence as a JSON list and a command line
        # writes one too; the field is a tuple, so the value is one.
        object.__setattr__(self, "val_metrics", tuple(self.val_metrics))
        object.__setattr__(self, "objective", trainings.name_of(type(self.mode)))
        self.mode.check(self)

        if self.pretrained is not None:
            scratch = ModelConfig("unet", dict(DEFAULT_MODEL_CONFIG))
            chosen = [name for name, named in (
                ("model.architecture", self.model.architecture != scratch.architecture),
                ("model.config", self.model.config != scratch.config),
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
                and self.preset.resolution_shift.tokens is None):
            height, width = self.sample_field().shape[-3:-1]
            object.__setattr__(self, "preset", dataclasses.replace(
                self.preset, resolution_shift=self.preset.resolution_shift.at(height, width)))
        unknown = [name for name in self.val_metrics if name not in metrics]
        if unknown:
            raise ValueError(
                f"val_metrics names {unknown}, which no metric is registered under; "
                f"the registered metrics are {sorted(metrics)}")
        if self.audio is not None and self.text is not None:
            raise ValueError(
                "the models take one context under textcontext, and this run names both "
                "text and audio; set text to None to condition on audio")
        if self.audio is not None and not isinstance(self.data, VideoDataset):
            raise ValueError(
                f"audio conditioning reads the audio of a VideoDataset's clips, and "
                f"{datasets.name_of(type(self.data))} carries none")

    def sample_field(self) -> Field:
        """Return the batch field the model generates, at the resolution the data comes in."""
        spec = self.data
        if isinstance(spec, VideoDataset):
            return Field("video", (spec.frames, spec.frame_size, spec.frame_size, 3))
        if isinstance(spec, OnlineVideos):
            return Field("video", (spec.frames, spec.image_size, spec.image_size, 3))
        if isinstance(spec, (ImageDataset, OnlineImages)):
            return Field("image", (spec.image_size, spec.image_size, 3))
        raise ValueError(
            f"the diffusion recipe trains on image or video datasets, not "
            f"{datasets.name_of(type(spec))}")

    def scratch_model(self, autoencoder: AutoEncoder | None) -> nn.Module:
        """Return the registry's model over the run's precision settings and `model.config`.

        An architecture that takes `output_channels` gets the channels it denoises (the
        published families name theirs as their sources do, in `model.config`). An
        `IntervalModel` embeds the duration under an interval process, and MeanFlow,
        whose loss differentiates in time, turns a `TimeScaled` model's time features at
        `SMOOTH_TIME_SCALE` unless `model.config` names a scale.
        """
        fields = dict(self.model.fields())
        if "output_channels" in {field.name for field in dataclasses.fields(models[self.model.architecture])}:
            fields["output_channels"] = (self.sample_field().shape[-1] if autoencoder is None
                                         else autoencoder.latent_channels)
        model = models.build(self.model.architecture, fields)
        if isinstance(model, IntervalModel) and self.preset is not None:
            built = self.preset()
            model = model.clone(interval=isinstance(built, Process) and built.interval)
        if isinstance(self.mode, MeanFlowTraining) and isinstance(model, TimeScaled) \
                and "time_scale" not in self.model.config:
            model = model.clone(time_scale=SMOOTH_TIME_SCALE)
        return model

    @property
    def context(self) -> TextCondition | AudioCondition | None:
        """Return the condition the model reads, text or audio, if any."""
        return self.text if self.text is not None else self.audio

    @property
    def parameter_roots(self) -> tuple[tuple[str, ...], ...]:
        """Return which parts of the variables tree this config builds own which parameters."""
        roots: list[tuple[str, ...]] = [("params",), (FROZEN,)]
        if self.pretrained is not None:
            # Every published pipeline's conditioner owns a bare tree.
            from dew.inputs.diffusion import DiffusionConditioner

            return (*roots, ("encoders", DiffusionConditioner.keyword), ("autoencoder",))
        if self.context is not None:
            encoder = encoders[self.context.encoder]
            prefix = ("encoders", encoder.keyword)
            collections = encoder.parameter_collections
            roots.extend((prefix,) if collections is None else
                         ((*prefix, collection) for collection in collections))
        if self.autoencoder is not None:
            roots.append(("autoencoder",))
        return tuple(roots)

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
        objective = self._objective(variables, None)
        if self.lora is None or self.pretrained is not None:
            return objective
        # The objective's own init draws the denoiser beside its heads and
        # towers; the adapter freezes the denoiser's weights, and the heads
        # stay under `params` to train, beside their own `constants`.
        key = jax.random.key(self.trainer.key)
        drawn = objective.init(key)
        adapter = self.lora.apply(objective.model, objective.model_variables(drawn),
                                  key=jax.random.fold_in(key, 1))
        heads = {collection: {name: tree for name, tree in drawn[collection].items() if name in LOSS_HEADS}
                 for collection in ("params", "constants") if collection in drawn}
        return self._objective(merge({**drawn, **adapter.variables}, heads), adapter.model)

    def _objective(self, variables: Variables | None, adapted: nn.Module | None) -> DiffusionObjective:
        """The configured objective over `variables`, with `adapted` in place
        of the model a run from scratch builds."""
        if self.pretrained is None:
            base, conditions, autoencoder = self._scratch(variables)
            model = base if adapted is None else adapted
            sample, convention = self.sample_field(), None
        else:
            source = self._source(variables)
            base = source.model
            if self.lora is not None:
                # The adapter binds to the denoiser and the pipeline's
                # weights, so the objective trains its factors alone.
                source = source.adapt(self.lora, key=self.trainer.key)
            if source.inputs is None:
                raise ValueError(f"{self.pretrained} loads no diffusion inputs to train on")
            if source.inputs.mask is not None:
                raise ValueError(f"{self.pretrained} is an inpainting pipeline, which trains "
                                 "on masks an image dataset does not carry")
            model, conditions, autoencoder = source.model, dict(source.inputs.conditions), source.autoencoder
            variables, convention = source.variables, source.process
            # The data's geometry, in the autoencoder's own pixel channels:
            # Qwen-Image 2.1's are RGBA.
            sample = source.inputs.sample
        inputs = InputSpec(sample=sample, conditions=conditions)
        process = self._process(convention)
        return self.mode.objective(self, model, process, inputs, base=base, autoencoder=autoencoder,
                                   variables=variables)

    def rollout(self, objective: DiffusionObjective):
        """Return the mode's rollout for `objective`, which the trainer runs.

        `FlowGRPO` returns its rollout, and a mode whose loss trains on each batch as it
        comes returns None.
        """
        return self.mode.rollout(objective)

    def pinned(self) -> DiffusionRunConfig:
        """Return this run with its Hub sources pinned to the commits they resolve to now.

        The sources are `pretrained` and the alignment's encoder, so the record names
        the weights the run started from.
        """
        from dew.interop import sources
        from dew.interop.pretrained import split_revision

        def pin(source: str) -> str:
            if os.path.isdir(source):
                return source
            name, revision = split_revision(source)
            return f"{name}@{sources.snapshot(name, revision, weights=False).name}"

        mode = self.mode
        if isinstance(mode, Denoising) and mode.alignment is not None:
            mode = dataclasses.replace(mode, alignment=dataclasses.replace(
                mode.alignment, encoder=pin(mode.alignment.encoder)))
        return dataclasses.replace(self, pretrained=None if self.pretrained is None else pin(self.pretrained),
                                   mode=mode)

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
                conditions[keyword] = self.text.build(params=params, dtype=self.model.dtype)
            else:
                # __post_init__ holds audio to a VideoDataset.
                assert self.audio is not None and isinstance(self.data, VideoDataset)
                conditions[keyword] = self.audio.build(self.data, params=params, dtype=self.model.dtype)
        return self.scratch_model(autoencoder), conditions, autoencoder

    def _autoencoder_params(self, variables: Variables) -> Variables:
        """The autoencoder's weights in a saved tree: frozen beside the
        model, or trained under `params` when REPA-E tuned it."""
        if isinstance(self.mode, Denoising) and self.mode.alignment is not None \
                and self.mode.alignment.end_to_end is not None:
            return variables["params"][AUTOENCODER]
        return variables["autoencoder"]

    def _source(self, variables: Variables | None):
        """The `pretrained` pipeline at the data's resolution: its own
        weights, or `variables` bound over its metadata."""
        from dew.interop.pretrained import load_diffusion_source, split_revision

        assert self.pretrained is not None
        name, revision = split_revision(self.pretrained)
        return load_diffusion_source(
            name, revision=revision, dtype=self.model.dtype or "bfloat16",
            param_dtype=self.model.param_dtype or "float32",
            attention_impl=self.model.attention_impl, size=self.sample_field().shape[:-1],
            variables=variables)

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
                f"{presets.name_of(type(self.preset))!r} is "
                f"{type(process.schedule).__name__} with {type(process.prediction).__name__}; "
                "name a preset of its kind, or none for the scheduler's own")
        return process

    def build_eval_metrics(self) -> list:
        """Build the validation metrics `val_metrics` names, each loading its own weights.

        A video run scores a `VideoGrid` against its `video` field, so `psnr` and
        `ssim` read that grid there, and an image-only metric raises a ValueError
        naming it here, before the trainer starts.
        """
        from dew.artifacts import ImageGrid, VideoGrid

        video = len(self.sample_field().shape) == 4
        built = []
        for name in self.val_metrics:
            if video and name in ("fid", "clip", "clip_score"):
                raise ValueError(
                    f"metric {name!r} reads an ImageGrid and this run samples video; "
                    "validate a video run with psnr or ssim")
            if name in ("psnr", "ssim"):
                factory = metrics[name]
                built.append(factory(field="video" if video else "image",
                                     reads=VideoGrid if video else ImageGrid))
            else:
                built.append(metrics[name]())
        return built
