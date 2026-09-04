"""The diffusion run, as one typed record and one construction.

`DiffusionRunConfig` is what a diffusion recipe parses from its command line
and writes as `run.json` next to the checkpoints, and `build()` is the one
function that turns it into the `DiffusionObjective`: the recipe trains what
it returns, and `TextToImage.from_run` samples from what it returns for the
same file, so the two cannot drift.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Optional

from dew.config import JsonDict, ModelConfig, RunConfig
from dew.data import ImageDataset, OnlineImages, VideoDataset
from dew.inputs import Condition, Field, InputSpec
from dew.nn.autoencoders import AutoEncoder
from dew.nn.text_encoders import DEFAULT_MODEL
from dew.registry import datasets, encoders, metrics, models, presets, samplers
from dew.sampling.guidance import CFG
from .objective import DiffusionObjective

import dew.eval  # noqa: F401  registers the image metrics
import dew.nn.backbones  # noqa: F401  registers the models

ATTENTION = {
    "heads": 8, "use_projection": False,
    "use_self_and_cross": True, "only_pure_attention": True,
}

# The default unet: attention everywhere but the full-resolution stage, where
# it costs the most. Every other architecture takes its own kwargs as JSON.
DEFAULT_MODEL_CONFIG = {
    "attention_configs": [None, ATTENTION, ATTENTION, ATTENTION],
    "precision": "default",
}


@dataclasses.dataclass(frozen=True)
class TextCondition:
    """The text the model is conditioned on: which registered encoder reads
    the batch's tokens, from which checkpoint."""

    encoder: str = "clip_text"
    checkpoint: str = DEFAULT_MODEL
    dtype: Optional[Literal["float32", "bfloat16"]] = None
    """The encoder's compute dtype; None keeps the checkpoint's."""
    field: str = "text"
    """The batch field holding the tokenized text."""
    unconditional: str = ""
    """The prompt the unconditional branch is encoded from."""

    def build(self) -> Condition:
        return Condition(encoders[self.encoder].from_pretrained(self.checkpoint, dtype=self.dtype),
                         field=self.field, unconditional=self.unconditional)


@dataclasses.dataclass(frozen=True)
class StableDiffusionAutoencoder:
    """Latent diffusion behind the vendored Stable Diffusion VAE."""

    modelname: str = "pcuenq/sd-vae-ft-mse-flax"
    revision: str = "bf16"
    dtype: Literal["float32", "bfloat16"] = "bfloat16"
    latent_shift: Optional[float] = None
    latent_scale: Optional[float] = None
    """Per-dataset latent statistics; None keeps the checkpoint's."""

    def build(self) -> AutoEncoder:
        from dew.nn.autoencoders.sd_vae import StableDiffusionVAE
        from dew.registry import resolve_dtype

        return StableDiffusionVAE(self.modelname, revision=self.revision,
                                  dtype=resolve_dtype(self.dtype),
                                  latent_shift=self.latent_shift, latent_scale=self.latent_scale)


@dataclasses.dataclass(frozen=True)
class DiffusionRunConfig(RunConfig):
    """A run, plus the diffusion objective's own knobs."""

    model: ModelConfig = dataclasses.field(
        default_factory=lambda: ModelConfig("unet", dict(DEFAULT_MODEL_CONFIG)))
    preset: presets.union = dataclasses.field(default_factory=presets.EDM)
    """The convention the model is trained and sampled with."""
    sampler: samplers.union = dataclasses.field(default_factory=samplers.EulerAncestral)
    """The solver validation samples with."""
    guidance: float = 3.0
    """Classifier-free guidance scale for validation samples; 0 samples the
    conditional prediction alone."""
    sampling_steps: int = 200
    unconditional_prob: float = 0.12
    """Fraction of training examples whose condition is dropped."""
    ema_decay: float = 0.999
    text: Optional[TextCondition] = dataclasses.field(default_factory=TextCondition)
    """The text condition, under the models' `textcontext` keyword; None
    trains unconditionally."""
    autoencoder: Optional[StableDiffusionAutoencoder] = None
    """Set for latent diffusion; None trains in pixel space."""
    val_metrics: list[Literal["clip", "clip_score", "fid", "psnr", "ssim"]] = dataclasses.field(
        default_factory=lambda: ["clip"])

    def sample_field(self) -> Field:
        """The batch field the model generates, at the resolution the data comes in."""
        spec = self.data
        if isinstance(spec, (ImageDataset, OnlineImages)):
            return Field("image", (spec.image_size, spec.image_size, 3))
        if isinstance(spec, VideoDataset):
            return Field("video", (spec.frames, spec.frame_size, spec.frame_size, 3))
        raise ValueError(
            f"the diffusion recipe trains on image or video datasets, not "
            f"{datasets.name_of(type(spec))}")

    def model_fields(self, autoencoder: Optional[AutoEncoder]) -> dict:
        """The fields the registry builds the model from: the run's precision
        settings and the channels the model denoises, over `model.config`."""
        fields = self.model.fields()
        sample = self.sample_field()
        if autoencoder is None:
            channels, size = sample.shape[-1], sample.shape[-2]
        else:
            channels = autoencoder.latent_channels
            size = sample.shape[-2] // autoencoder.downscale_factor
        if self.model.architecture == "diffusers_unet_simple":
            fields.update(sample_size=size, in_channels=channels, out_channels=channels)
        else:
            fields["output_channels"] = channels
        return fields

    def build(self) -> DiffusionObjective:
        """The objective this run trains: the model, the process, the inputs
        and the autoencoder, with validation sampling as configured. The one
        construction, shared by the recipe and by `TextToImage.from_run`."""
        autoencoder = None if self.autoencoder is None else self.autoencoder.build()
        conditions = {} if self.text is None else {"textcontext": self.text.build()}
        inputs = InputSpec(sample=self.sample_field(), conditions=conditions)
        model = models.build(self.model.architecture, **self.model_fields(autoencoder))
        return DiffusionObjective(
            model, self.preset(), inputs,
            autoencoder=autoencoder,
            unconditional_prob=self.unconditional_prob,
            ema_decay=self.ema_decay,
            sampler=self.sampler,
            guidance=CFG(self.guidance) if self.guidance else None,
            steps=self.sampling_steps,
        )

    def build_eval_metrics(self) -> list:
        """Validation metrics for `val_metrics`, each pulling its own weights
        on construction. A video run scores `VideoGrid` against its `video`
        field, so `psnr` and `ssim` read that grid there and the image-only
        metrics are refused by name instead of failing in the trainer."""
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
