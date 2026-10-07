"""The JEPA run, as one typed record: I-JEPA over images, V-JEPA over video.

The encoder is the run's model. The predictor takes the encoder's width and
heads, and its own depth and width from `predictor`: a predictor as wide as
the encoder makes the objective too easy. The probes score the frozen
encoder at every validation.
"""

from __future__ import annotations

import dataclasses

from dew.config import JsonDict, ModelConfig, ObjectiveConfig, OptimConfig, Prepared, RunConfig
from dew.data import ImageDataset, VideoDataset
from dew.inputs import Field
from dew.registry import models

from .masking import MultiBlockMask
from .probes import KnnProbe, LinearProbe

SHARED_MODEL_KEYS = ("emb_features", "num_heads", "mlp_ratio", "ssm_attention_ratio",
                     "ssm_state_dim", "dropout_rate", "precision")
"""What the predictor takes from the encoder unless `predictor` overrides it."""


@dataclasses.dataclass(frozen=True)
class JepaRunConfig(RunConfig):
    """A run, plus the JEPA objective's own knobs."""

    objective: ObjectiveConfig = dataclasses.field(default_factory=lambda: ObjectiveConfig("jepa"))
    """The JEPA objective and its arguments (`--objective.momentum 0.996 1.0`).
    Its EMA ramps over the whole run unless it names `momentum_steps`."""
    model: ModelConfig = dataclasses.field(
        default_factory=lambda: ModelConfig("jepa_encoder", {"precision": "default", "dtype": "bfloat16"}))
    optim: OptimConfig = dataclasses.field(default_factory=lambda: OptimConfig(learning_rate=1e-3))
    predictor: JsonDict = dataclasses.field(default_factory=dict)
    """Predictor kwargs, over the encoder's shared ones."""
    num_target_blocks: int = 4
    target_scale: tuple[float, float] = (0.15, 0.2)
    target_aspect: tuple[float, float] = (0.75, 1.5)
    probe_classes: int | None = None
    """Number of classes for the frozen-encoder probes, which are what a
    validation pass scores; a run without them schedules no pass."""
    knn_k: int = 20

    def __post_init__(self) -> None:
        if self.probe_classes is None and self.trainer.eval_every is not None:
            raise ValueError(
                "a JEPA validation pass scores the frozen-encoder probes; set probe_classes, "
                "or --trainer.eval-every None for a run without them")

    def sample(self) -> Field:
        """The batch field the encoder reads, at the resolution the data comes in."""
        spec = self.data
        if isinstance(spec, ImageDataset):
            return Field("image", (spec.image_size, spec.image_size, 3))
        if isinstance(spec, VideoDataset):
            return Field("video", (spec.frames, spec.frame_size, spec.frame_size, 3))
        raise ValueError(f"a JEPA run trains on image or video datasets, not {type(spec).__name__}")

    def prepare(self) -> Prepared:
        """The encoder, the predictor over its patch grid, the target-block mask
        and the probes, around the objective the run names. The predictor
        computes as the encoder does, its compute dtype and attention kernel."""
        dataset = self.data.load(batch=self.trainer.batch_size)
        sample = self.sample()
        video = sample.key == "video"
        if video != (models[self.model.name] is models["jepa_video_encoder"]):
            raise ValueError("a video dataset and --model jepa_video_encoder go together")
        encoder = self.model.build()
        grid = (sample.shape[-2] // encoder.patch_size,) * 2
        fields = self.model.fields
        predictor = ModelConfig("jepa_predictor", {
            **{key: value for key, value in fields.items() if key in SHARED_MODEL_KEYS}, **self.predictor,
            "grid": grid, "factorized": video, "scan_order": encoder.scan_order,
            **{key: fields[key] for key in ("dtype", "attention_impl") if key in fields}}).build()
        mask = MultiBlockMask.for_grid(grid, num_targets=self.num_target_blocks, scale=self.target_scale,
                                       aspect=self.target_aspect, scan_order=encoder.scan_order)
        ramp = ({} if "momentum_steps" in self.objective.fields
                else {"momentum_steps": self.trainer.total_steps(dataset)})
        objective = self.objective.build(encoder=encoder, predictor=predictor, mask=mask, sample=sample,
                                         **ramp)
        probes = () if not self.probe_classes else (
            LinearProbe(self.probe_classes), KnnProbe(self.probe_classes, k=self.knn_k))
        return Prepared(self, objective, dataset, metrics=probes)


__all__ = ["JepaRunConfig"]
