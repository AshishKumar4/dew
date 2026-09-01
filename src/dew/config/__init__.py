"""Typed configuration for a training run.

One dataclass tree describes a run: what to build, what to feed it, how to
optimize it, and where the checkpoints go. Recipes parse it with tyro and hand
the pieces to the trainer, so `to_dict()` is a full record of a run and
`from_dict()` puts it back together.

Model kwargs stay an opaque JSON dict. The registry already knows which
architecture takes which fields, and mirroring them here would be a second
place to keep in sync.
"""

import dataclasses
import json
from typing import Annotated, Any, Literal, Mapping, Optional, Self

import tyro

JsonDict = Annotated[
    dict[str, Any],
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=lambda args: json.loads(args[0]),
        is_instance=lambda value: isinstance(value, dict),
        str_from_instance=lambda value: [json.dumps(value)],
    ),
]
"""A dict, written as a single JSON string on the command line."""


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Architecture name and the kwargs `dew.registry.build_model` receives.

    The name may carry the +2d/+hilbert/+zigzag suffixes the registry
    canonicalizes.
    """

    architecture: str = "unet"
    config: JsonDict = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """Which dataset, at what resolution, through which loader."""

    dataset: str = "laiona_coco"
    dataset_path: str = "/home/mrwhite0racle/gcs_mount"
    dataset_seed: int = 0
    batch_size: int = 32
    image_size: int = 128
    val_steps_per_epoch: int = 4
    # 'auto' reads the dataset name: anything containing 'online' streams
    loader: Literal["auto", "grain", "online"] = "auto"
    augmentation_mode: Literal["none", "flip_only", "flip_jitter"] = "flip_jitter"
    worker_count: int = 32
    read_thread_count: int = 140
    read_buffer_size: int = 96
    worker_buffer_size: int = 100


@dataclasses.dataclass(frozen=True)
class OptimConfig:
    """Optimizer, learning-rate schedule and gradient handling."""

    optimizer: Literal["adam", "adamw", "lamb"] = "adamw"
    optimizer_opts: JsonDict = dataclasses.field(default_factory=dict)
    learning_rate: float = 2.7e-4
    learning_rate_schedule: Optional[Literal["cosine"]] = None
    learning_rate_peak: float = 3e-4
    learning_rate_end: float = 2e-4
    learning_rate_warmup_steps: int = 10000
    learning_rate_decay_epochs: int = 1
    weight_decay: Optional[float] = None
    clip_grads: float = 0.0
    grad_accum_steps: int = 1
    use_dynamic_scale: bool = False


@dataclasses.dataclass(frozen=True)
class TrainerConfig:
    """Loop length, checkpointing, sharding and run tracking."""

    name: Optional[str] = None
    epochs: int = 100
    steps_per_epoch: Optional[int] = None
    checkpoint_dir: str = "./checkpoints"
    checkpoint_fs: Literal["local", "gcs"] = "local"
    checkpoint_step: Optional[int] = None
    load_from_checkpoint: Optional[str] = None
    resume_last_run: Optional[str] = None
    max_checkpoints_to_keep: int = 1
    distributed_training: bool = True
    fsdp_size: int = 1
    fsdp_min_param_size: Optional[int] = None
    ema_decay: float = 0.999
    best_tracker_metric: Optional[str] = None
    profile_steps: int = 0
    compilation_cache_dir: Optional[str] = None
    log_every: int = 100
    wandb_project: str = "mlops-msml605-project"
    wandb_entity: str = "umd-projects"
    wandb_offline: bool = False


def _rebuild(cls, values: Mapping[str, Any]):
    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name not in values:
            continue
        value = values[field.name]
        kwargs[field.name] = (_rebuild(field.type, value)
                              if dataclasses.is_dataclass(field.type) else value)
    return cls(**kwargs)


@dataclasses.dataclass(frozen=True)
class RunConfig:
    """A whole run. Recipes add their objective's knobs by subclassing this."""

    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    data: DataConfig = dataclasses.field(default_factory=DataConfig)
    optim: OptimConfig = dataclasses.field(default_factory=OptimConfig)
    trainer: TrainerConfig = dataclasses.field(default_factory=TrainerConfig)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe record of the run."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Self:
        """Inverse of `to_dict`, for subclasses too."""
        return _rebuild(cls, values)
