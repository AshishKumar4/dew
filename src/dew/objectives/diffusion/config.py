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
from typing import TYPE_CHECKING, ClassVar, Literal

import jax
import numpy as np

from dew.config import ModelConfig, RunConfig
from dew.data import ImageDataset, OnlineImages, OnlineVideos, TFDSImages, VideoDataset
from dew.diffusion.presets import EDM, Flow, MeanFlow, Shortcut, build_process
from dew.diffusion.process import Process
from dew.inputs import Condition, Field, InputSpec, rebuild
from dew.nn.autoencoders import AutoEncoder
from dew.nn.text_encoders import DEFAULT_MODEL
from dew.objectives.base import FROZEN, Variables
from dew.registry import DtypeName, datasets, encoders, metrics, models, presets, solvers
from dew.sampling.guidance import CFG
from dew.sampling.solvers import EulerAncestral

from .alignment import REPRESENTATION, Alignment
from .end_to_end import AUTOENCODER, EndToEnd
from .objective import LOSS_HEADS, DiffusionObjective

import_module("dew.eval")  # registers the image metrics
import_module("dew.nn.backbones")  # registers the models before the config's unions are built

if TYPE_CHECKING:
    from flax import linen as nn

    from dew.diffusion.presets import Preset
    from dew.sampling.solvers import Solver

    # A registry's `union` is built from what has registered by import time, so
    # only the run sees it. Statically the fields are typed as every member
    # of those tables, which a reader and a checker need.
    PresetSpec = Preset
    SolverSpec = Solver
    # The run reads captions through `load(tokenize=)`, which the token
    # datasets do not take; `sample_field` refuses those at runtime.
    CaptionedSpec = ImageDataset | OnlineImages | VideoDataset
else:
    PresetSpec = presets.union
    SolverSpec = solvers.union
    CaptionedSpec = datasets.union

# Every other dial of a stage is `dew.nn.attention.Stage`'s own default, and
# a stage that names one restates it.
ATTENTION = {"heads": 8}

# Architectures that run the text as a second stream through every block's
# joint attention. With no text there is no sequence to project, so `build`
# raises for an unconditional run on one of these before the first attention
# softmax over an empty slice.
TEXT_STREAM_MODELS = ("simple_mmdit", "hierarchical_mmdit")

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


@dataclasses.dataclass(frozen=True)
class FlowGRPO:
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

    def rollout(self, objective):
        """Return the trainer's rollout over `objective`, scored by the named metric."""
        from dew.artifacts import ImageGrid
        from dew.eval.common import ImageMetric
        from dew.objectives.rl.flow import FlowRollout

        metric = metrics[self.reward]()
        if not isinstance(metric, ImageMetric):
            raise ValueError(f"reward {self.reward!r} is a {type(metric).__name__}, and Flow-GRPO "
                             "scores each sample with an image metric's per-sample measure")

        def reward(images, batch):
            return np.asarray(metric.fn(ImageGrid(images), batch))

        return FlowRollout(objective, reward, groups=self.groups, steps=self.rollout_steps,
                           train_steps=self.train_steps)



@dataclasses.dataclass(frozen=True)
class MeanFlowTraining:
    """MeanFlow training under the `mean_flow` preset, where one step samples the model.

    `MeanFlowObjective` documents the fields.
    """

    instantaneous: float = 0.75
    omega: float = 1.0
    kappa: float = 0.0
    guidance_interval: tuple[float, float] = (0.0, 1.0)
    norm_p: float = 1.0
    norm_eps: float = 0.01

    def __post_init__(self) -> None:
        # A record carries the interval as a JSON list.
        start, stop = (float(edge) for edge in self.guidance_interval)
        object.__setattr__(self, "guidance_interval", (start, stop))


@dataclasses.dataclass(frozen=True)
class ShortcutTraining:
    """Shortcut-model training under the `shortcut` preset.

    `ShortcutObjective` documents the fields.
    """

    sections: int = 128
    bootstrap_every: int = 8


def teacher_model(directory: str, variables: Variables | None) -> Variables:
    """A distilled run's teacher model variables: a saved distilled tree's
    own copy, else the teacher run's published ones."""
    from dew.checkpoints import Checkpoints

    from .objective import TEACHER, _without_loss_heads

    if variables is not None:
        return variables[TEACHER]
    restored = Checkpoints(directory).variables(ema=None, step=None, mesh=None, layout=None, param_dtype=None)
    return _without_loss_heads({name: tree for name, tree in restored.items()
                                if name not in ("encoders", "autoencoder")})


SMOOTH_TIME_SCALE = 0.002
"""The Fourier time scale a model trained through a derivative in time takes
when its config names none. On a 2-D two-class toy (RTX 4080), one-step
class accuracy at simple_dit's default 16 against 0.002 was MeanFlow 23%
against 99%, an sCM student 11% against 98.6%."""


@dataclasses.dataclass(frozen=True)
class AdversarialDistillation:
    """Adversarial distillation of a saved flow run into a few-step student.

    This is LADD with ADD's R1 penalty and distillation term
    (`AdversarialDistillationObjective`, which documents the fields). `teacher` is
    the teacher run's directory, and its model is this run's `model`.
    """

    teacher: str = ""
    feature_layers: tuple[str, ...] = ()
    student_times: tuple[float, ...] = (1.0, 0.75, 0.5, 0.25)
    renoise_times: tuple[float, float] = (1.0, 1.0)
    distillation_weight: float = 2.5
    r1_weight: float = 1e-5
    cmap_dim: int = 64
    kernel_size: tuple[int, int] = (9, 9)

    def __post_init__(self) -> None:
        if not self.teacher or not self.feature_layers:
            raise ValueError("adversarial distillation names a teacher run and the layers its "
                             "discriminator reads")
        object.__setattr__(self, "feature_layers", tuple(self.feature_layers))
        object.__setattr__(self, "student_times", tuple(float(time) for time in self.student_times))
        mean, std = (float(value) for value in self.renoise_times)
        object.__setattr__(self, "renoise_times", (mean, std))
        height, width = (int(size) for size in self.kernel_size)
        object.__setattr__(self, "kernel_size", (height, width))


@dataclasses.dataclass(frozen=True)
class ConsistencyDistillation:
    """rCM distillation of a saved flow run into a few-step student.

    rCM is sCM's consistency loss regularized by DMD2's, or either one alone when
    the other's weight is 0 (`ConsistencyDistillationObjective` documents the other
    fields). `teacher` is the teacher run's directory. Its model is this run's
    `model`, and the student and the fake score start from its weights.
    """

    teacher: str = ""
    consistency_weight: float = 100.0
    dmd_weight: float = 1.0
    teacher_guidance: float = 1.0
    tangent_warmup: int = 0
    student_update_freq: int = 5
    max_simulation_steps: int = 4
    student_times: tuple[float, float] = (-0.8, 1.6)
    critic_times: tuple[float, float] = (0.0, 1.6)
    consistency: Literal["continuous", "discrete"] = "continuous"
    discrete_steps: int = 48
    discrete_skip: int = 1
    discrete_shift: float = 5.0

    def __post_init__(self) -> None:
        if not self.teacher:
            raise ValueError("rCM distills a teacher; name its run directory")
        for name in ("student_times", "critic_times"):
            mean, std = (float(value) for value in getattr(self, name))
            object.__setattr__(self, name, (mean, std))

    def check_teacher(self, architecture: str) -> None:
        """Refuse sCM over a teacher whose time embedding changes too fast in time.

        The student starts from the teacher's variables, including its Fourier table,
        so it trains with the teacher's time scale whatever this run's model config
        says.
        """
        if self.consistency != "continuous" or self.consistency_weight <= 0:
            return
        if "time_scale" not in {field.name for field in dataclasses.fields(models[architecture])}:
            return
        teacher = DiffusionRunConfig.load(self.teacher)
        scale = teacher.model_fields(None).get("time_scale",
                                               {f.name: f.default for f in dataclasses.fields(
                                                   models[teacher.model.architecture])}["time_scale"])
        if scale != SMOOTH_TIME_SCALE:
            raise ValueError(
                f"sCM differentiates the student in time, and the student starts from a teacher trained at "
                f"time_scale={scale}, whose time embedding is too fast in it to learn from; "
                "train the teacher "
                f"with time_scale={SMOOTH_TIME_SCALE}, or distill with dmd only (consistency_weight=0)"
            )

    def teacher_variables(self, variables: Variables | None) -> Variables:
        return teacher_model(self.teacher, variables)


@dataclasses.dataclass(frozen=True)
class GuidanceDistillation:
    """Distillation of a saved run's classifier-free guidance into this run's model.

    The student reads the guidance scale as its conditioning's guidance input
    (`GuidanceDistillationObjective`). `teacher` is the teacher run's directory, and
    `scales` is the range each row's scale is drawn from.
    """

    teacher: str = ""
    scales: tuple[float, float] = (1.0, 8.0)

    def __post_init__(self) -> None:
        if not self.teacher:
            raise ValueError("guidance distillation distills a teacher; name its run directory")
        low, high = (float(value) for value in self.scales)
        object.__setattr__(self, "scales", (low, high))

    def teacher_objective(self, variables: Variables | None) -> tuple[DiffusionObjective, Variables]:
        """Return the teacher run's objective over its variables.

        The variables are a saved student tree's copy of them if there is one, and the
        teacher run's published ones otherwise.
        """
        from dew.checkpoints import Checkpoints

        from .objective import TEACHER

        held = (
            variables[TEACHER]
            if variables is not None
            else Checkpoints(self.teacher).variables(
                ema=None, step=None, mesh=None, layout=None, param_dtype=None
            )
        )
        return DiffusionRunConfig.load(self.teacher).build(variables=held), held


@dataclasses.dataclass(frozen=True)
class RepresentationAlignment:
    """Alignment of the model's hidden tokens with a frozen DINOv2's patch features.

    This is REPA (Yu et al. 2025) or iREPA (Singh et al. 2026), optionally with the
    autoencoder tuned end to end through it, as in REPA-E (Leng et al. 2025).

    `encoder` is a transformers `Dinov2Model` checkpoint given as `repo`,
    `repo@revision` or a directory, by default REPA's DINOv2-B/14, read at
    `resolution` pixels; a run's record pins it to a commit. `layer` names the
    model's submodule whose output is aligned: REPA aligns after the eighth block,
    which is `dit_block_7` on `simple_dit`. The other fields are `Alignment`'s, and
    `end_to_end` is REPA-E's `EndToEnd`, which needs a KL `autoencoder`.
    """

    encoder: str = "facebook/dinov2-base"
    layer: str = "dit_block_7"
    weight: float = 0.5
    projector: Literal["mlp", "conv"] = "mlp"
    width: int = 2048
    kernel_size: int = 3
    spatial_norm: float | None = None
    resolution: int = 224
    end_to_end: EndToEnd | None = None

    def build(self, variables: Variables | None = None) -> Alignment:
        """Return the alignment over the encoder's weights.

        It uses the `representation` subtree of `variables` when a saved tree supplies
        it, and the checkpoint's weights otherwise.
        """
        from dew.interop.pretrained import split_revision
        from dew.nn.autoencoders.rae import load_dinov2

        name, revision = split_revision(self.encoder)
        supplied = None if variables is None else variables[REPRESENTATION]["params"]
        module, params, _ = load_dinov2(name, revision=revision, params=supplied)
        return Alignment(module.clone(input_size=self.resolution), {"params": params}, self.layer,
                         weight=self.weight, projector=self.projector, width=self.width,
                         kernel_size=self.kernel_size, spatial_norm=self.spatial_norm,
                         resolution=self.resolution)


@dataclasses.dataclass(frozen=True)
class DiffusionRunConfig(RunConfig):
    """A diffusion run's configuration: the shared run fields plus the diffusion objective's own settings."""

    objective: str = "diffusion"
    """The objective `build` returns: `flow_grpo` under `rl`, `diffusion` otherwise.

    It follows `rl`, so a saved record names what trained.
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
    rl: FlowGRPO | None = None
    """Flow-GRPO training on a reward, in place of the denoising loss."""
    uncertainty: int | None = None
    """The number of Fourier channels of a head that learns EDM2's loss weighting.

    This is `DiffusionObjective(uncertainty=...)`; EDM2 uses 128. None keeps the
    preset's fixed weighting.
    """
    alignment: RepresentationAlignment | None = None
    """Alignment of the model's hidden tokens with a frozen DINOv2's (REPA or iREPA).

    With `end_to_end`, it also tunes the autoencoder through the alignment (REPA-E).
    """
    mean_flow: MeanFlowTraining | None = None
    """MeanFlow training in place of the denoising loss.

    The preset is `mean_flow`, and sampling is unguided because the guidance is
    trained in.
    """
    shortcut: ShortcutTraining | None = None
    """Shortcut-model training in place of the denoising loss.

    The preset is `shortcut`, and sampling is unguided.
    """
    distill: ConsistencyDistillation | None = None
    """Few-step distillation of a saved flow run (rCM, sCM or DMD2), in place of the denoising loss.

    Sampling is unguided.
    """
    guidance_distill: GuidanceDistillation | None = None
    adversarial: AdversarialDistillation | None = None
    """Adversarial distillation of a saved flow run (LADD, ADD); sampling is unguided."""
    """Distill a saved run's classifier-free guidance into this model's
    guidance input; sampling reads the conditioner's guidance value and no
    second branch."""
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
        object.__setattr__(self, "objective", "flow_grpo" if self.rl is not None else
                           "mean_flow" if self.mean_flow is not None else
                           "shortcut" if self.shortcut is not None else
                           "rcm" if self.distill is not None else
                           "guidance_distillation" if self.guidance_distill is not None else
                           "ladd" if self.adversarial is not None else "diffusion")

        extras = [name for name in ("rl", "alignment", "mean_flow", "shortcut", "distill", "uncertainty")
                  if getattr(self, name) is not None]
        if self.adversarial is not None and (extras or self.guidance_distill is not None
                                             or self.guidance is not None
                                             or not isinstance(self.preset, Flow)):
            raise ValueError("adversarial distillation trains on its own losses under the flow preset and "
                             "samples unguided: set guidance None, and leave the other training modes unset")
        if self.guidance_distill is not None and (extras or self.guidance is not None):
            raise ValueError("guidance distillation trains on its own loss and samples one branch: set "
                             f"guidance None, and leave {extras or 'the other training modes'} unset")
        others = [name for name in ("rl", "alignment", "mean_flow", "shortcut", "uncertainty")
                  if getattr(self, name) is not None]
        if self.distill is not None and (others or self.guidance is not None
                                         or not isinstance(self.preset, Flow)):
            raise ValueError(
                "rCM distills on its own losses under the flow preset and samples unguided: "
                f"set guidance None, and leave {others or 'rl, alignment, mean_flow, shortcut, uncertainty'}"
                " unset"
            )
        if (self.mean_flow is not None or self.shortcut is not None) and self.uncertainty is not None:
            raise ValueError("MeanFlow and shortcut models train on their own losses, which read no "
                             "learned uncertainty weighting; leave uncertainty unset")
        if self.mean_flow is not None and (self.rl is not None or self.alignment is not None
                                           or self.guidance is not None
                                           or not isinstance(self.preset, MeanFlow)):
            raise ValueError("MeanFlow trains on its own loss under the mean_flow preset, guided in "
                             "training (omega, kappa): set guidance None, and neither rl nor alignment")
        if self.shortcut is not None and (self.rl is not None or self.alignment is not None
                                          or self.mean_flow is not None or self.guidance is not None
                                          or not isinstance(self.preset, Shortcut)):
            raise ValueError("a shortcut model trains on its own loss under the shortcut preset and "
                             "samples unguided: set guidance None, and none of rl, alignment, mean_flow")
        if self.alignment is not None and (self.rl is not None or self.pretrained is not None):
            raise ValueError("representation alignment trains a scratch model on the denoising "
                             "loss; it takes neither `rl` nor `pretrained`")
        if self.alignment is not None and self.alignment.end_to_end is not None and self.autoencoder is None:
            raise ValueError("end-to-end tuning trains the run's autoencoder; set `autoencoder`")

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

    def model_fields(self, autoencoder: AutoEncoder | None) -> dict:
        """Return the fields the registry builds the model from.

        These are the run's precision settings over `model.config`, plus the channels
        the model denoises when the architecture takes them as `output_channels`. The
        published families name theirs as their sources do, in `model.config`.
        """
        fields = dict(self.model.fields())
        declared = {field.name for field in dataclasses.fields(models[self.model.architecture])}
        if "interval" in declared and self.preset is not None:
            # An interval process's model reads the interval's duration.
            built = self.preset()
            fields["interval"] = isinstance(built, Process) and built.interval
        if self.mean_flow is not None and "time_scale" in declared and "time_scale" not in self.model.config:
            # MeanFlow's loss differentiates the model in time; the default
            # time embedding is far too fast in it to learn from.
            fields["time_scale"] = SMOOTH_TIME_SCALE
        if "output_channels" in declared:
            sample = self.sample_field()
            fields["output_channels"] = (sample.shape[-1] if autoencoder is None
                                         else autoencoder.latent_channels)
        return fields

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
        # stay under `params` to train.
        key = jax.random.key(self.trainer.key)
        drawn = objective.init(key)
        adapter = self.lora.apply(objective.model, objective.model_variables(drawn),
                                  key=jax.random.fold_in(key, 1))
        heads = {name: tree for name, tree in drawn["params"].items() if name in LOSS_HEADS}
        start = {**drawn, **adapter.variables, "params": {**heads, **adapter.variables["params"]}}
        return self._objective(start, adapter.model)

    def _objective(self, variables: Variables | None, adapted: nn.Module | None) -> DiffusionObjective:
        """The configured objective over `variables`, with `adapted` in place
        of the model a run from scratch builds."""
        if self.pretrained is None:
            model, conditions, autoencoder = self._scratch(variables)
            model = model if adapted is None else adapted
            sample, convention = self.sample_field(), None
        else:
            source = self._source(variables)
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
        if self.rl is not None:
            from dew.objectives.rl.flow import FlowGRPOObjective
            from dew.sampling.flow import FlowSDE

            return FlowGRPOObjective(
                model, process, inputs, sde=FlowSDE(self.rl.noise_level), beta=self.rl.beta,
                clip_range=self.rl.clip_range, adv_clip_max=self.rl.adv_clip_max,
                autoencoder=autoencoder, guidance=self.guidance, solver=self.solver,
                steps=self.sampling_steps, variables=variables)
        if self.mean_flow is not None:
            from .few_step import MeanFlowObjective

            return MeanFlowObjective(
                model, process, inputs, **dataclasses.asdict(self.mean_flow),
                autoencoder=autoencoder, variables=variables, unconditional_prob=self.unconditional_prob,
                ema_decay=self.ema_decay, solver=self.solver, guidance=None, steps=self.sampling_steps)
        if self.adversarial is not None:
            from .adversarial import AdversarialDistillationObjective

            fields = {field.name: getattr(self.adversarial, field.name)
                      for field in dataclasses.fields(self.adversarial) if field.name != "teacher"}
            return AdversarialDistillationObjective(
                model, process, inputs, teacher=teacher_model(self.adversarial.teacher, variables), **fields,
                autoencoder=autoencoder, variables=variables, unconditional_prob=self.unconditional_prob,
                ema_decay=self.ema_decay, solver=self.solver, guidance=None, steps=self.sampling_steps)
        if self.guidance_distill is not None:
            from .guidance_distillation import GuidanceDistillationObjective

            teacher, held = self.guidance_distill.teacher_objective(variables)
            return GuidanceDistillationObjective(
                model, process, inputs, teacher=teacher, teacher_variables=held,
                scales=self.guidance_distill.scales, autoencoder=autoencoder, variables=variables,
                ema_decay=self.ema_decay, solver=self.solver, guidance=None, steps=self.sampling_steps)
        if self.distill is not None:
            from .consistency import ConsistencyDistillationObjective

            self.distill.check_teacher(self.model.architecture)

            fields = {
                field.name: getattr(self.distill, field.name)
                for field in dataclasses.fields(self.distill)
                if field.name != "teacher"
            }
            return ConsistencyDistillationObjective(
                model, process, inputs, teacher=self.distill.teacher_variables(variables), **fields,
                autoencoder=autoencoder, variables=variables, ema_decay=self.ema_decay, solver=self.solver,
                guidance=None, steps=self.sampling_steps)
        if self.shortcut is not None:
            from .few_step import ShortcutObjective

            return ShortcutObjective(
                model, process, inputs, **dataclasses.asdict(self.shortcut),
                autoencoder=autoencoder, variables=variables, unconditional_prob=self.unconditional_prob,
                ema_decay=self.ema_decay, solver=self.solver, guidance=None, steps=self.sampling_steps)
        return DiffusionObjective(
            model, process, inputs,
            autoencoder=autoencoder, variables=variables,
            unconditional_prob=self.unconditional_prob,
            ema_decay=self.ema_decay,
            solver=self.solver,
            guidance=self.guidance,
            steps=self.sampling_steps,
            uncertainty=self.uncertainty,
            alignment=None if self.alignment is None else self.alignment.build(variables),
            end_to_end=None if self.alignment is None else self.alignment.end_to_end,
        )

    def rollout(self, objective: DiffusionObjective):
        """Return the trainer's rollout for `objective`.

        That is Flow-GRPO's rollout, or None for the denoising loss, which trains on
        each batch as it comes.
        """
        return None if self.rl is None else self.rl.rollout(objective)

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

        alignment = (None if self.alignment is None else
                     dataclasses.replace(self.alignment, encoder=pin(self.alignment.encoder)))
        return dataclasses.replace(self, pretrained=None if self.pretrained is None else pin(self.pretrained),
                                   alignment=alignment)

    def _scratch(self, variables: Variables | None):
        """The registry's model, the run's text or audio condition and its
        autoencoder."""
        if self.context is None and self.model.architecture in TEXT_STREAM_MODELS:
            raise ValueError(
                f"an unconditional run needs a model that attends without text, and "
                f"{self.model.architecture!r} runs the text as a second stream through "
                "every block")
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
        model = models.build(self.model.architecture, self.model_fields(autoencoder))
        return model, conditions, autoencoder

    def _autoencoder_params(self, variables: Variables) -> Variables:
        """The autoencoder's weights in a saved tree: frozen beside the
        model, or trained under `params` when REPA-E tuned it."""
        if self.alignment is not None and self.alignment.end_to_end is not None:
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
