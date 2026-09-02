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

from dew.telemetry.instrumentation import default_compilation_cache_dir

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
    dtype: Literal["float32", "bfloat16"] = "bfloat16"
    """Compute dtype; params stay float32."""
    attention_impl: Literal["auto", "reference", "xla", "cudnn", "tpu"] = "auto"
    """Attention kernel; 'auto' is cudnn on gpu, xla elsewhere."""


@dataclasses.dataclass(frozen=True)
class DataConfig:
    """Which dataset, at what resolution, through which loader."""

    dataset: str = "oxford_flowers102"
    dataset_path: Optional[str] = None
    """Root the dataset's source resolves its files against; TFDS datasets
    ignore it and read from the TFDS data dir."""
    dataset_seed: int = 0
    batch_size: int = 32
    image_size: int = 128
    val_steps_per_epoch: int = 4
    loader: Literal["auto", "grain", "online"] = "auto"
    """'auto' reads the registries: a name registered only for streaming
    streams, anything else goes through grain."""
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
    checkpoint_every_steps: Optional[int] = None
    """Save a checkpoint every N global steps, not only at epoch boundaries."""
    distributed_training: bool = True
    multi_host: bool = False
    """Join the JAX process pool with jax.distributed.initialize(); failures
    raise. Required on TPU pods and any other multi-process run."""
    fsdp_size: int = 1
    fsdp_min_param_size: Optional[int] = None
    ema_decay: float = 0.999
    best_tracker_metric: Optional[str] = None
    profile_steps: int = 0
    compilation_cache_dir: Optional[str] = dataclasses.field(
        default_factory=default_compilation_cache_dir)
    """Persisted XLA cache, so a restart skips recompiling the step. None
    compiles from scratch every run."""
    log_every: int = 100
    wandb_project: Optional[str] = None
    """Unset runs without wandb: nothing is logged and nothing is published."""
    wandb_entity: Optional[str] = None
    wandb_offline: bool = False

    def __post_init__(self):
        if self.resume_last_run is not None and self.wandb_project is None:
            raise ValueError(
                "resume_last_run is a wandb run id and needs wandb_project set "
                "to resolve it")


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
