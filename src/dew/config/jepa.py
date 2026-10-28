"""The JEPA run, as one typed record: I-JEPA over images, V-JEPA over video.

The encoder is the run's model. The predictor takes the encoder's width and
heads, and its own depth and width from `predictor`: a predictor as wide as
the encoder makes the objective too easy. Optional probes score the frozen
encoder beside the validation loss.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import jax
import jax.numpy as jnp

from dew.config import MetricConfig, ObjectiveConfig, Prepared, RunConfig, media_field
from dew.data.pipeline import ObjectiveData
from dew.inputs import Field
from dew.nn.protocols import ReadsClips
from dew.objectives.jepa.masking import MultiBlockMask
from dew.optim import OptimConfig
from dew.registry import JsonDict, ModelConfig

SHARED_MODEL_KEYS = ("emb_features", "num_heads", "mlp_ratio", "ssm_attention_ratio",
                     "ssm_state_dim", "dropout_rate", "precision")
"""What the predictor takes from the encoder unless `predictor` overrides it."""


@dataclasses.dataclass(frozen=True)
class JepaRunConfig(RunConfig):
    """A run, plus the JEPA objective's own knobs."""

    RESUMABLE: ClassVar[tuple[str, ...]] = (*RunConfig.RESUMABLE, "probe_classes", "knn_k")

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
    """Number of classes for optional frozen-encoder probes; without them,
    validation scores the loss alone."""
    knn_k: int = 20

    def sample(self) -> Field:
        """The batch field the encoder reads, at the resolution the data comes in (`Dataset.facts`): a
        clip's video, else an image. Raises ValueError for a chain of neither, or one whose size
        varies."""
        return media_field(self.chain(), "JEPA")

    def prepare(self) -> Prepared:
        """The encoder, the predictor over its patch grid, the target-block mask
        and the probes, around the objective the run names. The predictor
        computes as the encoder does, its compute dtype and attention kernel."""
        # Compiled once, before the objective, whose EMA ramps over the run's length: JEPA reads its
        # batches as the chain makes them, with no stages of its own.
        self.built_fresh()
        dataset, held = self.compiled(ObjectiveData(), self.chain(), self.validation)
        sample = self.sample()
        video = sample.key == "video"
        encoder = self.model.build()
        if video != (isinstance(encoder, ReadsClips) and encoder.reads_clips):
            raise ValueError("a video dataset and an encoder that reads clips (--model jepa_video_encoder) "
                             "go together")
        # A frame is [height, width, channels], an image's or each of a clip's.
        grid = (sample.shape[-3] // encoder.patch_size, sample.shape[-2] // encoder.patch_size)
        # The encoder's effective arguments, its class's defaults among them, as the encoder was built.
        fields = self.model.arguments
        predictor = ModelConfig("jepa_predictor", {
            **{key: value for key, value in fields.items() if key in SHARED_MODEL_KEYS}, **self.predictor,
            "grid": grid, "factorized": video, "scan_order": encoder.scan_order,
            **{key: fields[key] for key in ("dtype", "attention_impl") if key in fields}}).build()
        mask = MultiBlockMask.for_grid(grid, num_targets=self.num_target_blocks, scale=self.target_scale,
                                       aspect=self.target_aspect, scan_order=encoder.scan_order)
        ramp = {}
        if "momentum_steps" not in self.objective.fields:
            updates = self.trainer.total_updates(dataset)
            if updates is None:
                raise ValueError("a run bounded by tokens alone counts no updates ahead for the target "
                                 "encoder's momentum to ramp over; set --objective.momentum_steps")
            ramp = {"momentum_steps": updates}
        adapted = {}
        if self.lora is not None:
            # A fresh encoder's draw from the run's key, frozen under the factors that train.
            key = jax.random.key(self.trainer.key)
            base = encoder.init(key, jnp.zeros((1, *sample.shape)),
                                jnp.arange(mask.num_context, dtype=jnp.int32)[None])
            adapter = self.lora.apply(encoder, base, key=jax.random.fold_in(key, 1))
            encoder, adapted = adapter.model, {"encoder_variables": adapter.variables}
        objective = self.objective.build(encoder=encoder, predictor=predictor, mask=mask, sample=sample,
                                         **ramp, **adapted)
        probes = self.validation_metrics(validates=bool(held))
        return Prepared(self, lambda name: self.train(objective, dataset, name=name, metrics=probes,
                                                      validation=held))

    def default_metrics(self, *, validates: bool) -> tuple[MetricConfig, ...]:
        """A linear and a k-nearest-neighbour probe of the frozen encoder, where `probe_classes` names
        the labels' classes, whatever the data holds: a run without a validation split refuses them."""
        del validates
        if not self.probe_classes:
            return ()
        return (MetricConfig("linear_probe", {"num_classes": self.probe_classes}),
                MetricConfig("knn_probe", {"num_classes": self.probe_classes, "k": self.knn_k}))


__all__ = ["JepaRunConfig"]
