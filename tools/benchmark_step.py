#!/usr/bin/env python3
"""Time the real training step, one architecture at a time.

tools/benchmark_data.py measures the loader. This measures what a step of the
Trainer costs for a given architecture, batch size and mesh. The step is
the one the trainer compiles for a real run (same objective, same sharding
and state ownership), so a number from this tool is a number from training.
Under `dew launch` every process measures its own devices and process 0
reports.

FLOPs are read off the compiled executable's optimized HLO
(dew.telemetry.instrumentation), and their utilisation is the figure the
trainer logs as train/hfu. MFU is the trainer's train/mfu: the objective's
training FLOPs on the case's batch (`Objective.training_flops`) over the
mesh's peak, or None where the model has no count. Each case is timed twice over the same number of
steps: once with the asynchronous dispatch a real run uses, which gives
ms/step, and once waiting on every step, which gives the p10/p50/p90 spread.

A case's peak memory is measured, not planned: each device's allocator
high-water over the case's run, native workspaces (cuBLAS, cuDNN, Mosaic
scratch) included, read in a process of the case's own that loads every
program from a compilation cache and compiles nothing (`measured_peak`,
tools/peak_memory.py). XLA's plan of the step's arguments, outputs and
temporaries stands beside it (`planned_device_bytes`).

Two registry architectures are composites: `multimodal_transformer` wraps a
decoder in an image tower and projector, and `diffusion_gemma` reads one
decoder both ways for the official block-diffusion loss. A case builds those
from its trunk `config` plus the `media` or `canvas` record the wrapper
needs, so their rows measure the tower and the canvas passes, not the plain
decoder step.

Usage:
    python tools/benchmark_step.py --preset cpu-smoke
    python tools/benchmark_step.py --preset small --json-out /tmp/bench.json
    python tools/benchmark_step.py --preset small --architectures simple_dit unet
    python tools/benchmark_step.py --architectures causal_transformer \\
        --attention-impl cudnn
    python tools/benchmark_step.py --architectures unet \\
        --xla-flags=--xla_gpu_triton_gemm_any=true
    python tools/benchmark_step.py --architectures simple_dit \\
        --profile-dir /tmp/dew-trace --profile-steps 5
    python tools/benchmark_step.py --cases '[{"architecture": "simple_dit",
        "config": {"patch_size": 2, "emb_features": 512, "num_layers": 12,
        "num_heads": 8}, "batch_size": 32, "image_size": 32, "mesh": {"fsdp": 2}}]'
    python tools/benchmark_step.py --preset small --architectures causal_transformer \\
        --mesh '{"fsdp": 2, "tensor": 2}' --profile-dir /tmp/dew-trace
    dew launch --processes-per-host 4 --devices-per-process 1 -- \\
        python tools/benchmark_step.py --mesh '{"fsdp": 2, "replicas": 2}' ...
    python tools/benchmark_step.py --preset small \\
        --architectures multimodal_transformer diffusion_gemma
"""

import contextlib
import dataclasses
import importlib
import io
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

import jax
import numpy as np
import tyro

from dew.files import write_atomically
from dew.nn.attention import AttentionImpl
from dew.nn.flops import MissingFlopsCount
from dew.pool import joins_pool, pool_cluster
from dew.registry import models
from dew.telemetry.benchmark import carried
from dew.telemetry.instrumentation import model_flops_utilization
from dew.telemetry.profile import capture_options
from dew.telemetry.trace import communication_totals, device_events, window_split
from dew.training import Trainer
from dew.training.memory import recompute_record
from dew.training.runtime import prepare_process

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peak_memory
from benchmark_cases import (
    Case,
    canvas_split,
    cpu_smoke_cases,
    images_per_row,
    media_pixels,
    mesh_label,
    small_cases,
)
from benchmark_models import build_trainer, compiled_step, global_batch, timed_steps, trainer_mesh
from trace_window import kernel_category, kernel_totals

Row = dict[str, object]


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


def mesh_from_json(text: str) -> dict[str, int]:
    """`--mesh`: one JSON object of MeshSpec fields, checked by building it."""
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("--mesh is a JSON object of MeshSpec fields")
    trainer_mesh(parsed)
    return parsed


JsonMesh = Annotated[
    dict[str, int] | None,
    tyro.constructors.PrimitiveConstructorSpec(
        nargs=1,
        metavar="JSON",
        instance_from_str=lambda args: mesh_from_json(args[0]),
        is_instance=lambda value: value is None or isinstance(value, dict),
        str_from_instance=lambda mesh: [json.dumps(mesh)],
    ),
]
"""A mesh, written as one JSON object on the command line."""


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
    """Model compute dtype for every case that names none of its own; losses
    stay fp32 either way."""
    attention_impl: AttentionImpl = 'auto'
    """Attention kernel, through the same precision policy a recipe uses."""
    fixed_batch: bool = False
    """Reuse one placed batch in every step, as tools/benchmark_torch.py does
    without --h2d. Unset, each step takes a fresh placement of the host batch
    from DevicePrefetchIterator, as Trainer.fit does."""
    xla_flags: str | None = None
    """Appended to XLA_FLAGS before the first JAX call, as TrainerConfig.xla_flags
    is. A flag only takes effect in a process that has not opened a backend
    yet, so a sweep runs one configuration per process."""
    batch_size: int | None = None
    """Override every case's batch size."""
    mesh: JsonMesh = None
    """Override every case's mesh with these MeshSpec fields."""
    image_size: int | None = None
    frames: int | None = None
    """Frame count for the video cases; image cases are left alone."""
    packed_documents: int | None = None
    """Documents per row for the language-model cases; the others are left
    alone. This is the packed loader's batch, which reroutes attention off the
    fused kernel."""
    profile_dir: str | None = None
    """Trace `profile_steps` more steps into this directory with jax.profiler
    after the timed windows, and read the device timeline back into the row:
    the fraction of the window the device was busy, kernels per step and
    kernel milliseconds per step by category. The trace itself stays on disk
    for TensorBoard or Perfetto."""
    profile_steps: int = 5
    json_out: str | None = None
    quiet: bool = True
    """Silence the trainer's own prints, which are per-run noise here."""
    within: str | None = None
    """`module:function` (a module beside this tool), whose context manager every case runs inside, in this
    process and each peak process: a diagnostic's swapped implementation (tools/benchmark_depthwise.py's
    lax form)."""
    peak_run: bool = False
    """A case run in a process of its own for its measured peak (`measured_peak`), its compiles taken
    from the cache that run names, measuring no peak of its own."""


def build_cases(config: BenchmarkConfig) -> list[Case]:
    if config.cases:
        cases = config.cases
    elif config.preset == 'cpu-smoke':
        cases = cpu_smoke_cases()
    else:
        cases = small_cases(config.dtype)
        # jepa_predictor has no step of its own: it is built through the registry
        # inside the two JEPA cases above.
        covered = {case.architecture for case in cases} | {"jepa_predictor"}
        missing = set(models) - covered
        if missing:
            raise ValueError(
                f"--preset small does not cover {sorted(missing)}; add a case for every "
                "architecture in dew.registry.models")

    if config.architectures:
        wanted = set(config.architectures)
        unknown = wanted - {case.architecture for case in cases}
        if unknown:
            raise ValueError(f"--architectures {sorted(unknown)} not in preset {config.preset}")
        cases = [case for case in cases if case.architecture in wanted]

    overrides: dict[str, object] = {}
    for name in ('batch_size', 'mesh', 'image_size'):
        value = getattr(config, name)
        if value is not None:
            overrides[name] = value

    def apply(case: Case) -> Case:
        # An image model handed a (T, H, W, C) sample is a rank error, so
        # --frames only resizes the video cases, and packing is a plain token
        # window's batch.
        frames = {} if config.frames is None or case.frames == 0 else {'frames': config.frames}
        packed = ({'packed_documents': config.packed_documents}
                  if config.packed_documents is not None and case.packs_documents else {})
        dtype = {'dtype': config.dtype} if case.dtype is None else {}
        return dataclasses.replace(case, **overrides, **frames, **packed, **dtype)

    return [apply(case) for case in cases]


def step_bytes(executable: jax.stages.Compiled | None) -> int | None:
    """The bytes one device holds while the compiled step runs, as XLA planned
    them: its arguments, the outputs not written over them, and the
    temporaries; native workspaces (cuBLAS, cuDNN, a kernel's scratch) are
    outside the plan, which `measured_peak` reads."""
    stats = None if executable is None else executable.memory_analysis()
    if stats is None:
        return None
    return (stats.argument_size_in_bytes + stats.output_size_in_bytes
            - stats.alias_size_in_bytes + stats.temp_size_in_bytes)


def parameter_count(params) -> int:
    return int(sum(np.prod(leaf.shape, dtype=np.int64) for leaf in jax.tree.leaves(params)))


def device_timeline(directory: str, steps: int, *, planes=None) -> dict[str, Any]:
    """What the device did during the traced `steps`, from the newest trace
    under `directory`.

    Busy is the union of kernel intervals across the traced devices and
    streams, divided by the earliest-start to latest-end window. For a
    multi-device trace this measures time when any device ran a kernel,
    not average utilization across devices. Category timings sum events
    and can overlap; they are not an additional wall-clock measurement.
    """
    if planes is None:
        planes = device_events(directory)[0]
    kernels = [(event.name, event.start_ns, event.end_ns)
               for events in planes.values() for event in events]
    if not kernels:
        raise ValueError(
            f"the trace under {directory} holds no device kernels: the profiler "
            "saw no accelerator, and a CPU run has no device timeline to read")
    split = window_split(kernels)
    busy, window = split["busy"], split["window"]
    by_category, _ = kernel_totals(kernels, kernel_category)
    by_name, _ = kernel_totals(kernels)
    per_step = 1e-6 / steps
    return {
        "profiled_steps": steps,
        "kernels_per_step": len(kernels) / steps,
        "device_busy_percent": 100.0 * busy / window,
        "device_busy_ms_per_step": busy * per_step,
        "device_window_ms_per_step": window * per_step,
        "kernel_ms_by_category": {
            category: ms * per_step
            for category, ms in sorted(by_category.items(), key=lambda item: -item[1])},
        "top_kernels": [(name[:100], ms * per_step)
                        for name, ms in sorted(by_name.items(), key=lambda item: -item[1])[:12]],
    }


def communication(directory: str, steps: int, *, planes=None) -> dict[str, Any]:
    """Each device's traced window split by `dew.telemetry.trace.window_split` into
    compute, every collective, the communication no compute kernel
    overlapped, and idle time ended by a host-to-device copy (input) or by
    anything else (host), averaged over the devices.

    `exposed_communication_ms_per_step` is the collective time no compute
    kernel on the same device ran beside: what overlap did not hide. Each
    HLO collective (`dew.telemetry.trace.COLLECTIVES`) gets its own time and exposed time,
    `all_to_all_ms_per_step` and `all_to_all_exposed_ms_per_step`, from the
    op each NCCL kernel ran for.
    """
    if planes is None:
        planes = device_events(directory)[0]
    devices = [kernels for kernels in planes.values() if kernels]
    if not devices:
        raise ValueError(f"the trace under {directory} holds no device kernels")

    totals = communication_totals(devices)
    totals.pop("busy")  # device_timeline's, over every device
    per_step = 1e-6 / steps / len(devices)
    window = totals.pop("window")
    return {
        "traced_devices": len(devices),
        "window_ms_per_step": window * per_step,
        **{f"{key}_ms_per_step": value * per_step for key, value in totals.items()},
        "exposed_communication_percent": 100.0 * totals["exposed_communication"] / window,
    }


def training_work(trainer: Trainer, case: Case, step_time: float) -> tuple[float | None, float | None]:
    """The objective's training FLOPs on the case's batch, train/mfu's numerator, and that MFU over the
    mesh's peak; both None where the model has no count."""
    try:
        flops = trainer.objective.training_flops(None, global_batch(case))
    except MissingFlopsCount:
        return None, None
    return flops, model_flops_utilization(flops / trainer.device_mesh.devices.size, step_time)


def measured_peak(case: Case, config: BenchmarkConfig) -> list[int | None]:
    """Each local device's high-water over the case's run, measured (tools/peak_memory.py) in a process of
    the case's own that loads its programs from a cache and compiles nothing. Run before this process
    opens a backend, which would hold the device it needs."""
    return peak_memory.measured(
        [os.path.abspath(__file__), "--cases", json.dumps([dataclasses.asdict(case)]),
         "--dtype", config.dtype, "--attention-impl", config.attention_impl, "--warmup", str(config.warmup),
         "--steps", str(config.steps), "--fixed-batch" if config.fixed_batch else "--no-fixed-batch",
         *(["--xla-flags", config.xla_flags] if config.xla_flags else []),
         *(["--within", config.within] if config.within else []),
         # The profiled steps are the case's too; their traces go apart from this process's.
         *(["--profile-dir", os.path.join(config.profile_dir, "peak-run"), "--profile-steps",
            str(config.profile_steps)] if config.profile_dir else []), "--peak-run"])


def measure(case: Case, config: BenchmarkConfig, peaks: list[int | None] | None) -> Row:
    """Warm up, then time the compiled step over a fixed number of steps. `peaks` is each device's
    measured high-water (`measured_peak`), None in a pool."""
    if config.steps < 1:
        raise ValueError(f"--steps must be at least 1, got {config.steps}")
    trainer = build_trainer(case, config.attention_impl)

    with compiled_step(case, trainer, fixed_batch=config.fixed_batch) as compiled:
        state, step, compile_seconds = compiled.state, compiled.step, compiled.compile_seconds
        # At least one warm step, so the first dispatch of the executable is
        # outside the timed window.
        state, loss, is_finite, _ = timed_steps(step, state, max(config.warmup, 1))
        state, loss, is_finite, elapsed = timed_steps(step, state, config.steps)

        # A second window of the same length, waiting on every step, for the
        # spread. The loop above dispatches asynchronously on purpose, as a run
        # does, so timing its individual iterations would time the dispatch and
        # not the step. These per-step numbers are a different quantity from
        # ms_per_step above, and each carries one synchronisation.
        (state, loss, is_finite), synced = carried(lambda carry: step(carry[0]), (state, loss, is_finite),
                                                   config.steps, each=True)
        p10, p50, p90 = np.percentile(np.asarray(synced) * 1e3, [10, 50, 90])

        timeline = {}
        if config.profile_dir:
            # After the timed windows, so the trace's own overhead is not in
            # them. Each process traces its own devices into its own
            # directory, since the trace file is named after the host.
            directory = os.path.join(config.profile_dir, case.label.replace(" ", "_"),
                                     f"process{jax.process_index()}")
            jax.profiler.start_trace(directory, profiler_options=capture_options())
            try:
                for _ in range(config.profile_steps):
                    state, loss, is_finite = step(state)
                loss.block_until_ready()
            finally:
                primary = sys.exception()
                try:
                    jax.profiler.stop_trace()
                except BaseException as error:
                    if primary is None:
                        raise
                    primary.add_note(f"Profiler stop failed: {error!r}")
            # A peak process traces the steps for their memory; their timeline is the case process's.
            if not config.peak_run:
                planes = device_events(directory)[0]
                timeline = {**device_timeline(directory, config.profile_steps, planes=planes),
                            **communication(directory, config.profile_steps, planes=planes)}
        flops = trainer.step_flops and trainer.step_flops.hardware
        step_time = elapsed / config.steps
        utilization = model_flops_utilization(flops, step_time)
        training_flops, mfu = training_work(trainer, case, step_time)
        planned = step_bytes(trainer.executable)
        row: Row = {
            "architecture": case.architecture,
            "batch_size": case.batch_size,
            "accumulation": case.accumulation,
            "mesh": case.mesh,
            "device_order": case.device_order,
            "mesh_shape": {axis: int(size) for axis, size in trainer.device_mesh.shape.items()},
            "processes": jax.process_count(),
            "sample_shape": [case.seq_len] if case.is_lm else list(case.sample_shape),
            "packed_documents": case.packed_documents,
            # A row's own extra work, so a number is readable without its
            # case: the images each row carries and their processed shape,
            # and the canvases the block loss splits the response into.
            "images_per_row": 0 if case.media is None else images_per_row(case),
            "image_pixels": None if case.media is None else list(media_pixels(case)),
            "canvases_per_row": 0 if case.canvas is None else canvas_split(case)[2],
            "dtype": case.dtype,
            "attention_impl": config.attention_impl,
            "fixed_batch": config.fixed_batch,
            "xla_flags": config.xla_flags,
            "devices": trainer.device_mesh.devices.size,
            "device_kind": jax.devices()[0].device_kind,
            "params": parameter_count(state.variables),
            "measured_steps": config.steps,
            "compile_seconds": round(compile_seconds, 2),
            # The rung the trainer compiled the step under, the model's own
            # or a stronger one where the step did not fit.
            "remat": recompute_record(trainer.objective),
            "ms_per_step": round(step_time * 1e3, 3),
            "p10_ms": round(float(p10), 3),
            "p50_ms": round(float(p50), 3),
            "p90_ms": round(float(p90), 3),
            "samples_per_sec": round(case.batch_size / step_time, 2),
            "tokens_per_sec": round(case.batch_size * case.seq_len / step_time, 1) if case.is_lm else None,
            "flops_per_step": flops,
            "utilization": utilization,
            "training_flops": training_flops,
            "mfu": mfu,
            "planned_device_bytes": planned,
            "loss": float(loss),
            "finite": bool(is_finite),
            **timeline,
        }
    measured = None if peaks is None or None in peaks else max(peaks)
    return {**row, "peak_device_bytes": measured, "peak_bytes_by_device": peaks}


TABLE_COLUMNS = (
    # Wide enough for the longest registry name, multimodal_transformer.
    ("architecture", "architecture", 22, "{}"),
    ("batch_size", "batch", 6, "{}"),
    ("mesh", "mesh", 24, "{}"),
    ("params", "params", 12, "{:,}"),
    ("ms_per_step", "ms/step", 9, "{:.1f}"),
    ("p10_ms", "p10", 7, "{:.1f}"),
    ("p50_ms", "p50", 7, "{:.1f}"),
    ("p90_ms", "p90", 7, "{:.1f}"),
    ("samples_per_sec", "samples/s", 10, "{:.1f}"),
    ("flops_per_step", "GFLOP/step", 11, "{:.1f}"),
    ("utilization", "util %", 7, "{:.1f}"),
    ("mfu", "mfu %", 7, "{:.1f}"),
    ("peak_device_bytes", "peak GiB", 9, "{:.2f}"),
    ("planned_device_bytes", "plan GiB", 9, "{:.2f}"),
)
# Units the table shows a column in: FLOPs as GFLOP, a fraction as a
# percentage, bytes as GiB.
TABLE_SCALE = {"flops_per_step": 1e-9, "utilization": 100.0, "mfu": 100.0, "peak_device_bytes": 2 ** -30,
               "planned_device_bytes": 2 ** -30}


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
            elif key == "mesh" and isinstance(value, dict):
                text = mesh_label(value)
            elif isinstance(value, (int, float)):
                text = fmt.format(value * TABLE_SCALE.get(key, 1))
            else:
                text = fmt.format(value)
            cells.append(text.ljust(width) if key == "architecture" else text.rjust(width))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def run(config: BenchmarkConfig, peaks: list[list[int | None]] | None) -> list[Row]:
    """Every case measured, with its devices' measured high-water (`measured_peak`) from `peaks`, one a
    case in order; None in a pool."""
    rows: list[Row] = []
    # Every process of a pool measures its own devices; process 0 speaks.
    speaker = jax.process_index() == 0
    for index, case in enumerate(build_cases(config)):
        # The trainer narrates state generation and input shapes per case,
        # which buries the numbers this tool exists to print.
        sink = (contextlib.redirect_stdout(io.StringIO()) if config.quiet
                else contextlib.nullcontext())
        with sink:
            row = measure(case, config, None if peaks is None else peaks[index])
        rows.append(row)
        if not speaker:
            continue
        print(f"{case.label}: {row['ms_per_step']} ms/step, "
              f"{row['samples_per_sec']} samples/s")
        categories = row.get("kernel_ms_by_category")
        if isinstance(categories, dict):
            categories = ", ".join(f"{category} {ms:.2f}" for category, ms in categories.items())
            print(f"  device busy {row['device_busy_percent']:.1f}% of "
                  f"{row['device_window_ms_per_step']:.2f} ms/step, "
                  f"{row['kernels_per_step']:.0f} kernels/step; ms/step by category: "
                  f"{categories}")
            print(f"  per device: compute {row['compute_ms_per_step']:.2f}, communication "
                  f"{row['communication_ms_per_step']:.2f}, exposed "
                  f"{row['exposed_communication_ms_per_step']:.2f} ms/step "
                  f"({row['exposed_communication_percent']:.1f}% of the window), idle on the host "
                  f"{row['idle_host_ms_per_step']:.2f}, on input {row['idle_input_ms_per_step']:.2f}")
        if config.json_out:
            # A GPU sweep is minutes of compilation per case; rewriting the
            # file as each case lands means an interrupted sweep still keeps
            # the cases it did measure.
            write_atomically(config.json_out, json.dumps(rows, indent=2))
    return rows


def entered(within: str | None) -> contextlib.AbstractContextManager:
    """The context `within` (`BenchmarkConfig.within`) names, or none."""
    if within is None:
        return contextlib.nullcontext()
    module, _, function = within.partition(":")
    return getattr(importlib.import_module(module), function)()


def main(config: BenchmarkConfig) -> list[Row]:
    if not config.peak_run and not jax.config.jax_compilation_cache_dir:
        # Run alone, a step's compile seconds are a cold compile's, which no cached executable stands in for.
        # A caller that names its cache (JAX_COMPILATION_CACHE_DIR: perf_gate's per-tree cache, whose later
        # rounds load what its first compiled) keeps it, and its own switch (JAX_ENABLE_COMPILATION_CACHE).
        jax.config.update("jax_enable_compilation_cache", val=False)
    # Before this process opens a backend, which would hold the device each case's peak process needs.
    measuring = not (joins_pool(None, pool_cluster(None)) or config.peak_run)
    peaks = [measured_peak(case, config) for case in build_cases(config)] if measuring else None
    # Joins a `dew launch` pool when one launched this process, and applies
    # the flags before the backend opens either way.
    prepare_process(xla_flags=config.xla_flags)
    speaker = jax.process_index() == 0
    if speaker:
        print(f"Devices: {jax.device_count()} x {jax.devices()[0].device_kind}, "
              f"{jax.process_count()} process(es)")
        print(f"dtype {config.dtype}, attention_impl {config.attention_impl}, "
              f"XLA_FLAGS {os.environ.get('XLA_FLAGS', '')!r}")
    with entered(config.within):
        rows = run(config, peaks)
    if not speaker:
        return rows
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
