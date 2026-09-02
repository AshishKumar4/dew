#!/usr/bin/env python3
"""Time the real training step, one architecture at a time.

tools/benchmark_data.py answers "is the loader the bottleneck". This answers
the other half: what a step of ObjectiveTrainer costs for a given
architecture, batch size and fsdp width. The step measured here is the one the
trainer compiles for a real run - same objective, same sharding, same donated
state - so a number from this tool is a number from training, not from a
hand-written forward pass.

FLOPs are counted off the compiled executable's optimized HLO
(dew.telemetry.instrumentation), never from a parameter-count formula, and the
utilisation is the same figure the trainer logs as train/mfu. Each case is
timed twice over the same number of steps: once with the asynchronous dispatch
a real run uses, which gives ms/step, and once waiting on every step, which
gives the p10/p50/p90 spread.

Usage:
    python tools/benchmark_step.py --preset cpu-smoke
    python tools/benchmark_step.py --preset small --json-out /tmp/bench.json
    python tools/benchmark_step.py --preset small --architectures simple_dit unet
    python tools/benchmark_step.py --cases '[{"architecture": "simple_dit",
        "config": {"patch_size": 2, "emb_features": 512, "num_layers": 12,
        "num_heads": 8}, "batch_size": 32, "image_size": 32, "fsdp_size": 2}]'
"""

import contextlib
import dataclasses
import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro

from dew.diffusion.transforms import get_diffusion_preset
from dew.inputs import ConditionalInputConfig, DiffusionInputConfig
from dew.inputs.encoders import ConditioningEncoder
from dew.objectives.jepa import JepaObjective, multi_block_mask
from dew.objectives.lm import LMObjective
from dew.registry import MODEL_REGISTRY, build_model
from dew.telemetry.instrumentation import compiled_flops
from dew.training import ObjectiveTrainer
from dew.training.distributed import DevicePrefetchIterator

JsonCases = Annotated[
    list[dict[str, Any]],
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=lambda args: json.loads(args[0]),
        is_instance=lambda value: isinstance(value, list),
        str_from_instance=lambda value: [json.dumps(value)],
    ),
]
"""A list of case dicts, written as one JSON string on the command line."""

# Stand-in for the CLIP-L/14 context: a benchmark of the model should not spend
# its first minute downloading a text encoder, and the step cost only depends
# on the context's shape.
TEXT_TOKENS = 77
TEXT_FEATURES = 768


@dataclass
class StubTextEncoder(ConditioningEncoder):
    """Pretokenized text of a fixed width, embedded by broadcast."""

    @property
    def key(self):
        return "text"

    def tokenize(self, data):
        return np.zeros((len(data), TEXT_TOKENS), np.int32)

    def encode_from_tokens(self, tokens):
        embedded = jnp.asarray(tokens, jnp.float32)[..., None] / TEXT_TOKENS
        return jnp.broadcast_to(embedded, (*embedded.shape[:2], TEXT_FEATURES))

    def serialize(self):
        return {}

    @staticmethod
    def deserialize(serialized_config):
        raise NotImplementedError("benchmark-only encoder")


@dataclass(frozen=True)
class Case:
    """One measurement: what to build, how much to feed it, how to shard it."""

    architecture: str
    config: dict = field(default_factory=dict)
    batch_size: int = 8
    fsdp_size: int = 1
    image_size: int = 32
    frames: int = 0
    """Video models take (frames, H, W, C) samples; 0 means images."""
    predictor: Optional[dict] = None
    """Set for JEPA: the architecture is an encoder and this builds its predictor."""
    seq_len: int = 0
    """Set for language models: batches are token windows of this length, not images."""
    fsdp_min_param_size: int = 2 ** 16

    @property
    def is_jepa(self) -> bool:
        return self.predictor is not None

    @property
    def is_lm(self) -> bool:
        return self.seq_len > 0

    @property
    def sample_shape(self) -> tuple:
        square = (self.image_size, self.image_size, 3)
        return square if self.frames == 0 else (self.frames, *square)

    @property
    def label(self) -> str:
        return f"{self.architecture} b{self.batch_size} fsdp{self.fsdp_size}"


def cpu_smoke_cases() -> list[Case]:
    """Tiny enough to run anywhere, real enough to compile the same step."""
    return [
        Case("simple_dit", {"patch_size": 4, "emb_features": 64, "num_layers": 2,
                            "num_heads": 2, "mlp_ratio": 2},
             batch_size=8, image_size=16, fsdp_min_param_size=256),
        Case("jepa_encoder", {"patch_size": 4, "emb_features": 32, "num_layers": 2,
                              "num_heads": 2, "mlp_ratio": 2},
             predictor={"grid": (4, 4), "emb_features": 32, "predictor_features": 16,
                        "num_layers": 1, "num_heads": 2, "mlp_ratio": 2},
             batch_size=8, image_size=16, fsdp_min_param_size=256),
        Case("causal_transformer", {"vocab_size": 256, "emb_features": 32, "num_layers": 2,
                                    "num_heads": 2, "mlp_ratio": 2, "max_seq_len": 16},
             batch_size=8, seq_len=16, fsdp_min_param_size=256),
    ]


def small_cases(dtype: str) -> list[Case]:
    """Every registry architecture at a size that fits one 16 GB card in bf16.

    Sized so the whole sweep is minutes rather than hours: real token counts
    (256 image tokens at 64px/patch 4) and real widths, but few layers.
    """
    dit = {"patch_size": 4, "emb_features": 384, "num_layers": 6, "num_heads": 6,
           "mlp_ratio": 4, "dtype": dtype}
    unet = {"emb_features": 256, "feature_depths": [64, 128, 256],
            "attention_configs": [None, {"heads": 4, "dtype": dtype},
                                  {"heads": 4, "dtype": dtype}],
            "num_res_blocks": 2, "num_middle_res_blocks": 1, "dtype": dtype}
    encoder = {"patch_size": 4, "emb_features": 384, "num_layers": 6, "num_heads": 6,
               "mlp_ratio": 4, "dtype": dtype}
    predictor = {"grid": (16, 16), "emb_features": 384, "predictor_features": 192,
                 "num_layers": 3, "num_heads": 6, "mlp_ratio": 4, "dtype": dtype}

    cases = [
        Case("unet", unet, batch_size=16, image_size=64),
        Case("uvit", {**dit, "num_layers": 6}, batch_size=16, image_size=64),
        Case("simple_udit", {**dit, "num_layers": 6}, batch_size=16, image_size=64),
        Case("simple_dit", dit, batch_size=16, image_size=64),
        Case("simple_mmdit", dit, batch_size=16, image_size=64),
        Case("hierarchical_mmdit",
             {"base_patch_size": 2, "emb_features": (192, 384, 576),
              "num_layers": (2, 2, 2), "num_heads": (3, 6, 9), "mlp_ratio": 4,
              "dtype": dtype},
             batch_size=16, image_size=64),
        Case("hybrid_dit", {**dit, "ssm_state_dim": 64, "ssm_attention_ratio": "3:1"},
             batch_size=16, image_size=64),
        Case("video_dit", {**dit, "num_layers": 4}, batch_size=4, image_size=64, frames=8),
        Case("unet_3d", {**unet, "temporal_heads": 4},
             batch_size=4, image_size=64, frames=8),
        Case("jepa_encoder", encoder, predictor=predictor, batch_size=16, image_size=64),
        Case("jepa_video_encoder", {**encoder, "num_layers": 4},
             predictor={**predictor, "num_layers": 2, "factorized": True},
             batch_size=4, image_size=64, frames=8),
        # GPT-2 small's width and heads at a quarter of its depth, 512-token windows
        Case("causal_transformer", {"vocab_size": 50304, "emb_features": 768, "num_layers": 3,
                                    "num_heads": 12, "mlp_ratio": 4, "max_seq_len": 512,
                                    "dtype": dtype},
             batch_size=16, seq_len=512),
    ]
    # jepa_predictor has no step of its own: it is built through the registry
    # inside the two JEPA cases above.
    covered = {case.architecture for case in cases} | {"jepa_predictor"}
    missing = set(MODEL_REGISTRY) - covered
    if missing:
        raise ValueError(
            f"--preset small does not cover {sorted(missing)}; add a case for every "
            "architecture in dew.registry.MODEL_REGISTRY")
    return cases


@dataclass(frozen=True)
class BenchmarkConfig:
    """Which cases to time, and how."""

    preset: Literal['small', 'cpu-smoke'] = 'small'
    cases: JsonCases = field(default_factory=list)
    """Explicit cases as a JSON list of Case fields; replaces the preset."""
    architectures: Optional[list[str]] = None
    """Keep only these cases from the preset."""
    warmup: int = 2
    steps: int = 100
    """Measured steps per case, timed twice: once dispatched asynchronously for
    ms/step, once waiting per step for the p10/p50/p90 spread."""
    dtype: Literal['bfloat16', 'float32'] = 'bfloat16'
    """Model compute dtype for --preset small; losses stay fp32 either way."""
    batch_size: Optional[int] = None
    """Override every case's batch size."""
    fsdp_size: Optional[int] = None
    image_size: Optional[int] = None
    frames: Optional[int] = None
    """Frame count for the video cases; image cases are left alone."""
    checkpoint_dir: str = "/tmp/dew-benchmark-step"
    json_out: Optional[str] = None
    quiet: bool = True
    """Silence the trainer's own prints, which are per-run noise here."""


def build_cases(config: BenchmarkConfig) -> list[Case]:
    if config.cases:
        fields = {f.name for f in dataclasses.fields(Case)}
        for spec in config.cases:
            unknown = sorted(set(spec) - fields)
            if unknown:
                raise ValueError(
                    f"--cases entry has no such field {unknown}; "
                    f"valid fields are {sorted(fields)}")
        cases = [Case(**spec) for spec in config.cases]
    elif config.preset == 'cpu-smoke':
        cases = cpu_smoke_cases()
    else:
        cases = small_cases(config.dtype)

    if config.architectures:
        wanted = set(config.architectures)
        unknown = wanted - {case.architecture for case in cases}
        if unknown:
            raise ValueError(f"--architectures {sorted(unknown)} not in preset {config.preset}")
        cases = [case for case in cases if case.architecture in wanted]

    overrides = {name: getattr(config, name)
                 for name in ('batch_size', 'fsdp_size', 'image_size')
                 if getattr(config, name) is not None}

    def apply(case: Case) -> Case:
        # An image model handed a (T, H, W, C) sample is not a shorter
        # benchmark, it is a rank error, so --frames only resizes the video
        # cases.
        frames = {} if config.frames is None or case.frames == 0 else {'frames': config.frames}
        return dataclasses.replace(case, **overrides, **frames)

    return [apply(case) for case in cases]


def text_condition() -> ConditionalInputConfig:
    return ConditionalInputConfig(
        encoder=StubTextEncoder(model=None, tokenizer=None),
        conditioning_data_key="text",
        pretokenized=True,
        unconditional_input="",
        model_key_override="textcontext",
    )


def build_trainer(case: Case, checkpoint_dir: str) -> ObjectiveTrainer:
    """The trainer a recipe would build for this case, minus wandb and data."""
    model = build_model(case.architecture, case.config)
    sample_key = "video" if case.frames else "image"
    optimizer = optax.adam(1e-4)
    common = dict(
        model=model,
        optimizer=optimizer,
        rngs=jax.random.PRNGKey(0),
        name=f"bench-{case.architecture}",
        wandb_config=None,
        distributed_training=True,
        fsdp_size=case.fsdp_size,
        fsdp_min_param_size=case.fsdp_min_param_size,
        checkpoint_base_path=checkpoint_dir,
    )

    if case.is_lm:
        objective = LMObjective(model, case.seq_len, vocab_size=case.config["vocab_size"])
        return ObjectiveTrainer(input_config=None, objective=objective, **common)

    if case.is_jepa:
        patch = case.config.get("patch_size", 16)
        grid = (case.image_size // patch, case.image_size // patch)
        predictor_config = {**case.predictor, "grid": grid}
        objective = JepaObjective(
            model, build_model("jepa_predictor", predictor_config),
            multi_block_mask(grid, num_targets=2, scale=(0.2, 0.3)),
            sample_key, case.sample_shape,
        )
        input_config = DiffusionInputConfig(
            sample_data_key=sample_key, sample_data_shape=case.sample_shape, conditions=[])
        return ObjectiveTrainer(input_config=input_config, objective=objective, **common)

    train_schedule, _, transform = get_diffusion_preset("edm")
    input_config = DiffusionInputConfig(
        sample_data_key=sample_key,
        sample_data_shape=case.sample_shape,
        conditions=[text_condition()],
    )
    return ObjectiveTrainer(
        input_config=input_config,
        noise_schedule=train_schedule,
        model_output_transform=transform,
        **common,
    )


def batches(case: Case):
    """One host batch, reused: the loader is benchmarked by benchmark_data.py."""
    rng = np.random.default_rng(0)
    if case.is_lm:
        batch = {"text": rng.integers(0, case.config["vocab_size"],
                                      size=(case.batch_size, case.seq_len + 1)).astype(np.int32)}
    else:
        sample_key = "video" if case.frames else "image"
        batch = {sample_key: rng.integers(
            0, 256, size=(case.batch_size, *case.sample_shape)).astype(np.float32)}
        if not case.is_jepa:
            batch["text"] = np.ones((case.batch_size, TEXT_TOKENS), np.int32)
    while True:
        yield batch


def device_peak_bytes() -> Optional[int]:
    """The allocator's high-water mark, where the backend reports one (not CPU).

    Monotonic for the life of the process and with no reset hook, so in a sweep
    it is this case's own peak only for the first case; every later case gets
    an upper bound plus its own delta.
    """
    try:
        stats = jax.local_devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    return stats.get('peak_bytes_in_use') or stats.get('bytes_in_use')


def parameter_count(params) -> int:
    return int(sum(np.prod(leaf.shape, dtype=np.int64) for leaf in jax.tree.leaves(params)))


def measure(case: Case, config: BenchmarkConfig) -> dict[str, Any]:
    """Warm up, then time the compiled step over a fixed number of steps."""
    peak_before = device_peak_bytes()
    trainer = build_trainer(case, config.checkpoint_dir)
    trainer.global_batch_size = case.batch_size

    train_step = trainer._define_train_step(batch_size=case.batch_size)
    source = DevicePrefetchIterator(batches(case), trainer.batch_sharding)
    state, rng = trainer.state, trainer.rngstate

    compile_start = time.perf_counter()
    compiled = trainer._compiled_step(train_step, state, rng, next(source))
    compile_seconds = time.perf_counter() - compile_start

    loss = None
    for _ in range(max(config.warmup, 1)):
        state, loss, _, rng, is_finite = compiled(state, rng, next(source))
    loss.block_until_ready()

    start = time.perf_counter()
    for _ in range(config.steps):
        state, loss, _, rng, is_finite = compiled(state, rng, next(source))
    loss.block_until_ready()
    elapsed = time.perf_counter() - start

    # A second window of the same length, waiting on every step, for the
    # spread. The loop above dispatches asynchronously on purpose - that is
    # how a run behaves - so timing its individual iterations would time the
    # dispatch, not the step. These per-step numbers are therefore a different
    # quantity from ms_per_step above, and each carries one synchronisation.
    synced = []
    for _ in range(config.steps):
        step_start = time.perf_counter()
        state, loss, _, rng, is_finite = compiled(state, rng, next(source))
        loss.block_until_ready()
        synced.append((time.perf_counter() - step_start) * 1e3)
    p10, p50, p90 = np.percentile(synced, [10, 50, 90])

    flops = compiled_flops(compiled)
    throughput = trainer._throughput_metrics(elapsed, config.steps)
    peak = device_peak_bytes()
    row = {
        "architecture": case.architecture,
        "batch_size": case.batch_size,
        "fsdp_size": case.fsdp_size,
        "sample_shape": [case.seq_len] if case.is_lm else list(case.sample_shape),
        "devices": trainer.mesh.devices.size,
        "device_kind": jax.devices()[0].device_kind,
        "params": parameter_count(state.params),
        "measured_steps": config.steps,
        "compile_seconds": round(compile_seconds, 2),
        "ms_per_step": round(elapsed / config.steps * 1e3, 3),
        "p10_ms": round(float(p10), 3),
        "p50_ms": round(float(p50), 3),
        "p90_ms": round(float(p90), 3),
        "samples_per_sec": round(throughput["train/samples_per_sec"], 2),
        "flops_per_step": flops,
        "utilization": throughput.get("train/mfu"),
        "peak_device_bytes": peak,
        "case_peak_delta_bytes": (
            None if peak_before is None else max(0, peak - peak_before)),
        "loss": float(loss),
        "finite": bool(is_finite),
    }
    # The state holds every device buffer this case allocated; drop it before
    # the next case builds its own.
    del state, compiled, trainer
    return row


TABLE_COLUMNS = (
    ("architecture", "architecture", 20, "{}"),
    ("batch_size", "batch", 6, "{}"),
    ("fsdp_size", "fsdp", 5, "{}"),
    ("params", "params", 12, "{:,}"),
    ("ms_per_step", "ms/step", 9, "{:.1f}"),
    ("p10_ms", "p10", 7, "{:.1f}"),
    ("p50_ms", "p50", 7, "{:.1f}"),
    ("p90_ms", "p90", 7, "{:.1f}"),
    ("samples_per_sec", "samples/s", 10, "{:.1f}"),
    ("flops_per_step", "GFLOP/step", 11, "{:.1f}"),
    ("utilization", "util %", 7, "{:.1f}"),
    ("peak_device_bytes", "peak GiB", 9, "{:.2f}"),
)


def format_table(rows: list[dict]) -> str:
    scale = {"flops_per_step": 1e-9, "utilization": 100.0, "peak_device_bytes": 2 ** -30}
    header = " ".join(title.rjust(width) if key != "architecture" else title.ljust(width)
                      for key, title, width, _ in TABLE_COLUMNS)
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = []
        for key, _, width, fmt in TABLE_COLUMNS:
            value = row.get(key)
            if value is None:
                text = "n/a"
            else:
                text = fmt.format(value * scale[key] if key in scale else value)
            cells.append(text.ljust(width) if key == "architecture" else text.rjust(width))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def run(config: BenchmarkConfig) -> list[dict]:
    rows = []
    for case in build_cases(config):
        # The trainer narrates state generation and input shapes per case,
        # which buries the numbers this tool exists to print.
        sink = (contextlib.redirect_stdout(io.StringIO()) if config.quiet
                else contextlib.nullcontext())
        with sink:
            row = measure(case, config)
        rows.append(row)
        print(f"{case.label}: {row['ms_per_step']} ms/step, "
              f"{row['samples_per_sec']} samples/s")
        if config.json_out:
            # A GPU sweep is minutes of compilation per case; rewriting the
            # file as each case lands means an interrupted sweep still keeps
            # the cases it did measure.
            write_json(rows, config.json_out)
    return rows


def write_json(rows: list[dict], path: str):
    with open(path, "w") as handle:
        json.dump(rows, handle, indent=2)


def main(config: BenchmarkConfig):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    rows = run(config)
    print()
    print(format_table(rows))
    if config.json_out:
        print(f"\nWrote {config.json_out}")
    else:
        print()
        print(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    main(tyro.cli(BenchmarkConfig))
