#!/usr/bin/env python3
"""Time the real training step, one architecture at a time.

tools/benchmark_data.py measures the loader. This measures what a step of the
Trainer costs for a given architecture, batch size and fsdp width. The step
is the one the trainer compiles for a real run (same objective, same
sharding, same donated state), so a number from this tool is a number from
training.

FLOPs are read off the compiled executable's optimized HLO
(dew.telemetry.instrumentation), and the utilisation is the figure the
trainer logs as train/mfu. Each case is timed twice over the same number of
steps: once with the asynchronous dispatch a real run uses, which gives
ms/step, and once waiting on every step, which gives the p10/p50/p90 spread.

Usage:
    python tools/benchmark_step.py --preset cpu-smoke
    python tools/benchmark_step.py --preset small --json-out /tmp/bench.json
    python tools/benchmark_step.py --preset small --architectures simple_dit unet
    python tools/benchmark_step.py --architectures causal_transformer \\
        --attention-impl cudnn
    python tools/benchmark_step.py --architectures unet \\
        --xla-flags=--xla_gpu_triton_gemm_any=true
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
from typing import Annotated, Iterator, Literal, Mapping

import jax
import numpy as np
import optax
import tyro

from dew.diffusion import presets
from dew.inputs import CharTable, Condition, Field, InputSpec
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.jepa import JepaObjective, multi_block_mask
from dew.objectives.lm import LMObjective
from dew import models  # naming a registry fills it
from dew.registry import with_precision
from dew.telemetry.devices import apply_xla_flags
from dew.telemetry.instrumentation import compiled_flops, model_flops_utilization
from dew.training import Layout, MeshSpec, Trainer
from dew.training.distributed import DevicePrefetchIterator

# The CLIP-L/14 context's shape, from the library's table encoder: a benchmark
# of the model should not spend its first minute downloading a text tower, and
# the step cost depends only on the context's shape.
TEXT_TOKENS = 77
TEXT_FEATURES = 768

Batch = dict[str, np.ndarray | Mapping[str, np.ndarray]]
Row = dict[str, object]


def text_condition() -> Condition:
    return Condition(CharTable.from_pretrained(tokens=TEXT_TOKENS, features=TEXT_FEATURES))


@dataclass(frozen=True)
class Case:
    """One measurement: what to build, how much to feed it, how to shard it."""

    architecture: str
    config: dict[str, object] = field(default_factory=dict)
    dtype: str = 'float32'
    """Compute dtype, written into the model config by the precision policy."""
    batch_size: int = 8
    fsdp_size: int = 1
    expert_size: int = 1
    """Devices the expert dimension of an MoE layer is split over, as
    --trainer.expert-size; 1 replicates every expert."""
    image_size: int = 32
    frames: int = 0
    """Video models take (frames, H, W, C) samples; 0 means images."""
    predictor: dict[str, object] | None = None
    """Set for JEPA: the architecture is an encoder and this builds its predictor."""
    seq_len: int = 0
    """Set for language models: batches are token windows of this length, not images."""
    packed_documents: int = 0
    """Set for language models: documents packed into every row, with the
    segment ids and positions the packed loader emits; 0 feeds a fixed
    window, which is what a stream of tokens gives."""
    fsdp_min_param_size: int = 2 ** 16

    @property
    def is_jepa(self) -> bool:
        return self.predictor is not None

    @property
    def is_lm(self) -> bool:
        return self.seq_len > 0

    @property
    def sample_shape(self) -> tuple[int, ...]:
        square = (self.image_size, self.image_size, 3)
        return square if self.frames == 0 else (self.frames, *square)

    @property
    def label(self) -> str:
        mixture = self.config.get("mixture")
        experts = f" x{mixture['experts']}experts" if isinstance(mixture, dict) else ""
        return (f"{self.architecture}{experts} b{self.batch_size} fsdp{self.fsdp_size} "
                f"expert{self.expert_size}")


def cases_from_json(text: str) -> list[Case]:
    """`--cases`: a JSON list of objects whose keys are Case fields."""
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise ValueError("--cases is a JSON list of objects, one per case")
    fields = {f.name for f in dataclasses.fields(Case)}
    cases = []
    for spec in parsed:
        if not isinstance(spec, dict):
            raise ValueError(f"--cases entry {spec!r} is not a JSON object")
        unknown = sorted(set(spec) - fields)
        if unknown:
            raise ValueError(
                f"--cases entry has no such field {unknown}; valid fields are {sorted(fields)}")
        cases.append(Case(**spec))
    return cases


JsonCases = Annotated[
    list[Case],
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=lambda args: cases_from_json(args[0]),
        is_instance=lambda value: isinstance(value, list),
        str_from_instance=lambda cases: [json.dumps([dataclasses.asdict(c) for c in cases])],
    ),
]
"""A list of cases, written as one JSON string on the command line."""


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
                                    "num_heads": 2, "mlp_features": 64, "max_seq_len": 16},
             batch_size=8, seq_len=16, fsdp_min_param_size=256),
    ]


def small_cases(dtype: str) -> list[Case]:
    """Every registry architecture at a size that fits one 16 GB card in bf16.

    Sized so the whole sweep is minutes rather than hours: real token counts
    (256 image tokens at 64px/patch 4) and real widths, but few layers.
    """
    dit: dict[str, object] = {"patch_size": 4, "emb_features": 384, "num_layers": 6,
                              "num_heads": 6, "mlp_ratio": 4}
    unet: dict[str, object] = {"emb_features": 256, "feature_depths": [64, 128, 256],
                               "attention_configs": [None, {"heads": 4}, {"heads": 4}],
                               "num_res_blocks": 2, "num_middle_res_blocks": 1}
    encoder: dict[str, object] = {"patch_size": 4, "emb_features": 384, "num_layers": 6,
                                  "num_heads": 6, "mlp_ratio": 4}
    predictor: dict[str, object] = {"grid": (16, 16), "emb_features": 384,
                                    "predictor_features": 192, "num_layers": 3,
                                    "num_heads": 6, "mlp_ratio": 4}

    cases = [
        Case("unet", unet, batch_size=16, image_size=64),
        Case("uvit", {**dit, "num_layers": 6}, batch_size=16, image_size=64),
        Case("simple_udit", {**dit, "num_layers": 6}, batch_size=16, image_size=64),
        Case("simple_dit", dit, batch_size=16, image_size=64),
        Case("simple_mmdit", dit, batch_size=16, image_size=64),
        Case("hierarchical_mmdit",
             {"base_patch_size": 2, "emb_features": (192, 384, 576),
              "num_layers": (2, 2, 2), "num_heads": (3, 6, 9), "mlp_ratio": 4},
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
                                    "num_heads": 12, "mlp_features": 3072, "max_seq_len": 512},
             batch_size=16, seq_len=512),
        # The same decoder with an 8-expert, top-2 feed-forward on every second
        # layer, which is the sparse shape the 4.7 acceptance run trains
        Case("causal_transformer", {"vocab_size": 50304, "emb_features": 768, "num_layers": 3,
                                    "num_heads": 12, "mlp_features": 3072, "max_seq_len": 512,
                                    "mixture": {"experts": 8, "top_k": 2, "every": 2}},
             batch_size=16, seq_len=512),
    ]
    cases = [dataclasses.replace(case, dtype=dtype) for case in cases]
    # jepa_predictor has no step of its own: it is built through the registry
    # inside the two JEPA cases above.
    covered = {case.architecture for case in cases} | {"jepa_predictor"}
    missing = set(models) - covered
    if missing:
        raise ValueError(
            f"--preset small does not cover {sorted(missing)}; add a case for every "
            "architecture in dew.registry.models")
    return cases


@dataclass(frozen=True)
class BenchmarkConfig:
    """Which cases to time, and how."""

    preset: Literal['small', 'cpu-smoke'] = 'small'
    cases: JsonCases = field(default_factory=list)
    """Explicit cases as a JSON list of Case fields; replaces the preset."""
    architectures: list[str] | None = None
    """Keep only these cases from the preset."""
    warmup: int = 2
    steps: int = 100
    """Measured steps per case, timed twice: once dispatched asynchronously for
    ms/step, once waiting per step for the p10/p50/p90 spread."""
    dtype: Literal['bfloat16', 'float32'] = 'bfloat16'
    """Model compute dtype for --preset small; losses stay fp32 either way."""
    attention_impl: Literal['auto', 'reference', 'xla', 'cudnn', 'tpu'] = 'auto'
    """Attention kernel, through the same precision policy a recipe uses."""
    xla_flags: str | None = None
    """Appended to XLA_FLAGS before the first JAX call, as TrainerConfig.xla_flags
    is. A flag only takes effect in a process that has not opened a backend
    yet, so a sweep runs one configuration per process."""
    batch_size: int | None = None
    """Override every case's batch size."""
    fsdp_size: int | None = None
    image_size: int | None = None
    frames: int | None = None
    """Frame count for the video cases; image cases are left alone."""
    packed_documents: int | None = None
    """Documents per row for the language-model cases; the others are left
    alone. This is the packed loader's batch, which reroutes attention off the
    fused kernel."""
    json_out: str | None = None
    quiet: bool = True
    """Silence the trainer's own prints, which are per-run noise here."""


def build_cases(config: BenchmarkConfig) -> list[Case]:
    if config.cases:
        cases = config.cases
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

    overrides: dict[str, int] = {}
    for name in ('batch_size', 'fsdp_size', 'image_size'):
        value = getattr(config, name)
        if value is not None:
            overrides[name] = value

    def apply(case: Case) -> Case:
        # An image model handed a (T, H, W, C) sample is a rank error, so
        # --frames only resizes the video cases, and packing is a language
        # model's batch.
        frames = {} if config.frames is None or case.frames == 0 else {'frames': config.frames}
        packed = ({'packed_documents': config.packed_documents}
                  if config.packed_documents is not None and case.is_lm else {})
        return dataclasses.replace(case, **overrides, **frames, **packed)

    return [apply(case) for case in cases]


def build_trainer(case: Case, attention_impl: str = 'auto') -> Trainer:
    """The trainer a recipe would build for this case, minus the tracker and the
    checkpoints.

    The model goes through the same precision function the recipes use, so the
    dtype and the attention kernel land in the nested unet attention configs
    too, and a row of this table is a row a real run would produce.
    """
    def built(architecture: str, config: Mapping[str, object]):
        return models.build(architecture, **with_precision(
            architecture, config, dtype=case.dtype, attention_impl=attention_impl))

    model = built(case.architecture, case.config)
    sample_key = "video" if case.frames else "image"

    if case.is_lm:
        objective = LMObjective(model, case.seq_len)
    elif case.predictor is not None:
        patch = case.config.get("patch_size", 16)
        if not isinstance(patch, int):
            raise ValueError(f"{case.architecture}'s patch_size is {patch!r}, not an int")
        grid = (case.image_size // patch, case.image_size // patch)
        objective = JepaObjective(
            model, built("jepa_predictor", {**case.predictor, "grid": grid}),
            multi_block_mask(grid, num_targets=2, scale=(0.2, 0.3)),
            sample=Field(sample_key, case.sample_shape))
    else:
        inputs = InputSpec(Field(sample_key, case.sample_shape),
                           {"textcontext": text_condition()})
        objective = DiffusionObjective(model, presets.EDM()(), inputs)

    return Trainer(
        objective, optax.adam(1e-4), key=jax.random.key(0),
        mesh=MeshSpec(fsdp=case.fsdp_size, expert=case.expert_size),
        layout=Layout(min_shard=case.fsdp_min_param_size),
        checkpoints=None, tracker=None)


def batches(case: Case) -> Iterator[Batch]:
    """One host batch, reused: the loader is benchmarked by benchmark_data.py."""
    rng = np.random.default_rng(0)
    batch: Batch = {}
    if case.is_lm:
        vocab = case.config["vocab_size"]
        if not isinstance(vocab, int):
            raise ValueError(f"{case.architecture}'s vocab_size is {vocab!r}, not an int")
        width = case.seq_len + 1
        batch["text"] = rng.integers(0, vocab, size=(case.batch_size, width)).astype(np.int32)
        if case.packed_documents:
            # Equal documents tiling the row. A packed row from the loader is
            # ragged and can end in padding; what the mask and the kernel cost
            # see is how many segments the row carries, and equal ones make
            # the case reproducible.
            per_document = -(-width // case.packed_documents)
            document = np.repeat(np.arange(case.packed_documents), per_document)[:width]
            rows = (case.batch_size, 1)
            batch["text_segment_ids"] = np.tile(document + 1, rows).astype(np.int32)
            batch["text_positions"] = np.tile(
                np.arange(width) - document * per_document, rows).astype(np.int32)
    else:
        sample_key = "video" if case.frames else "image"
        batch[sample_key] = rng.integers(
            0, 256, size=(case.batch_size, *case.sample_shape)).astype(np.float32)
        if not case.is_jepa:
            batch["text"] = text_condition().encoder.tokenize(["a flower"] * case.batch_size)
    while True:
        yield batch


def device_peak_bytes() -> int | None:
    """The allocator's high-water mark, where the backend reports one (not CPU).

    Monotonic for the life of the process and with no reset hook, so in a sweep
    it is this case's own peak only for the first case; every later case gets
    an upper bound plus its own delta.
    """
    stats = jax.local_devices()[0].memory_stats()
    if not stats:
        return None
    return stats.get('peak_bytes_in_use') or stats.get('bytes_in_use')


def parameter_count(params) -> int:
    return int(sum(np.prod(leaf.shape, dtype=np.int64) for leaf in jax.tree.leaves(params)))


def measure(case: Case, config: BenchmarkConfig) -> Row:
    """Warm up, then time the compiled step over a fixed number of steps."""
    if config.steps < 1:
        raise ValueError(f"--steps must be at least 1, got {config.steps}")
    peak_before = device_peak_bytes()
    trainer = build_trainer(case, config.attention_impl)

    source = DevicePrefetchIterator(batches(case), trainer.device_mesh)
    abstract = jax.eval_shape(trainer.initial_state)
    state = jax.jit(trainer.initial_state, out_shardings=trainer.shardings(abstract))()
    scale = None

    compile_start = time.perf_counter()
    compiled = trainer.compile(state, next(source))
    compile_seconds = time.perf_counter() - compile_start

    def step(state, scale):
        state, scale, loss, _, finite = compiled(state, scale, next(source))
        return state, scale, loss, finite

    # At least one warm step, so the timed window never holds the first
    # dispatch of the executable.
    state, scale, loss, is_finite = step(state, scale)
    for _ in range(config.warmup - 1):
        state, scale, loss, is_finite = step(state, scale)
    loss.block_until_ready()

    start = time.perf_counter()
    for _ in range(config.steps):
        state, scale, loss, is_finite = step(state, scale)
    loss.block_until_ready()
    elapsed = time.perf_counter() - start

    # A second window of the same length, waiting on every step, for the
    # spread. The loop above dispatches asynchronously on purpose, as a run
    # does, so timing its individual iterations would time the dispatch and
    # not the step. These per-step numbers are a different quantity from
    # ms_per_step above, and each carries one synchronisation.
    synced = []
    for _ in range(config.steps):
        step_start = time.perf_counter()
        state, scale, loss, is_finite = step(state, scale)
        loss.block_until_ready()
        synced.append((time.perf_counter() - step_start) * 1e3)
    p10, p50, p90 = np.percentile(synced, [10, 50, 90])

    flops = compiled_flops(compiled)
    step_time = elapsed / config.steps
    utilization = model_flops_utilization(flops, step_time)
    peak = device_peak_bytes()
    row: Row = {
        "architecture": case.architecture,
        "batch_size": case.batch_size,
        "fsdp_size": case.fsdp_size,
        "expert_size": case.expert_size,
        "sample_shape": [case.seq_len] if case.is_lm else list(case.sample_shape),
        "packed_documents": case.packed_documents,
        "dtype": case.dtype,
        "attention_impl": config.attention_impl,
        "xla_flags": config.xla_flags,
        "devices": trainer.device_mesh.devices.size,
        "device_kind": jax.devices()[0].device_kind,
        "params": parameter_count(state.params),
        "measured_steps": config.steps,
        "compile_seconds": round(compile_seconds, 2),
        "ms_per_step": round(step_time * 1e3, 3),
        "p10_ms": round(float(p10), 3),
        "p50_ms": round(float(p50), 3),
        "p90_ms": round(float(p90), 3),
        "samples_per_sec": round(case.batch_size / step_time, 2),
        "flops_per_step": flops,
        "utilization": utilization,
        "peak_device_bytes": peak,
        "case_peak_delta_bytes": (
            None if peak is None or peak_before is None else max(0, peak - peak_before)),
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
    ("expert_size", "expert", 7, "{}"),
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
# Units the table shows a column in: FLOPs as GFLOP, a fraction as a
# percentage, bytes as GiB.
TABLE_SCALE = {"flops_per_step": 1e-9, "utilization": 100.0, "peak_device_bytes": 2 ** -30}


def format_table(rows: list[Row]) -> str:
    header = " ".join(title.rjust(width) if key != "architecture" else title.ljust(width)
                      for key, title, width, _ in TABLE_COLUMNS)
    lines = [header, "-" * len(header)]
    for row in rows:
        cells = []
        for key, _, width, fmt in TABLE_COLUMNS:
            value = row.get(key)
            if value is None:
                text = "n/a"
            elif isinstance(value, (int, float)):
                text = fmt.format(value * TABLE_SCALE.get(key, 1))
            else:
                text = fmt.format(value)
            cells.append(text.ljust(width) if key == "architecture" else text.rjust(width))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def run(config: BenchmarkConfig) -> list[Row]:
    rows: list[Row] = []
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


def write_json(rows: list[Row], path: str) -> None:
    with open(path, "w") as handle:
        json.dump(rows, handle, indent=2)


def main(config: BenchmarkConfig) -> list[Row]:
    apply_xla_flags(config.xla_flags)
    print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}")
    print(f"dtype {config.dtype}, attention_impl {config.attention_impl}, "
          f"XLA_FLAGS {os.environ.get('XLA_FLAGS', '')!r}")
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
