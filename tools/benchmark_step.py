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

A case runs in processes of its own (`measured_row`, tools/peak_memory.py): the
first compiles its programs into a cache, its compile the row's
`compile_seconds`; the second loads every program from it, compiling nothing,
and is the case's measurement: its timed windows, its profile, and its peak
memory, measured, not planned: each device's allocator high-water over the
case's run, native workspaces (cuBLAS, cuDNN, Mosaic scratch) included, read
as the process exits, after its windows. XLA's plan of the step's arguments,
outputs and temporaries stands beside it (`planned_device_bytes`). The
process that starts them opens no backend; a `dew launch` pool's processes
measure their own devices, timing in process, with no peak.

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
    python tools/benchmark_step.py --cases '[{"architecture": "causal_transformer",
        "hf_config": "Qwen/Qwen3.5-4B", "dtype": "bfloat16", "batch_size": 2,
        "seq_len": 2048}]'
    python tools/benchmark_step.py --preset small --architectures causal_transformer \\
        --mesh '{"fsdp": 2, "tensor": 2}' --profile-dir /tmp/dew-trace
    dew launch --processes-per-host 4 --devices-per-process 1 -- \\
        python tools/benchmark_step.py --mesh '{"fsdp": 2, "replicas": 2}' ...
    python tools/benchmark_step.py --preset small \\
        --architectures multimodal_transformer diffusion_gemma
"""

import contextlib
import dataclasses
import datetime
import hashlib
import importlib
import io
import json
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import jax
import numpy as np
import tyro

from dew._files import write_atomically
from dew._pool import joins_pool
from dew.clusters import detected_environment
from dew.nn.attention import AttentionImpl
from dew.nn.flops import MissingFlopsCount
from dew.nn.kernels.selection import Unlowerable
from dew.registry import models
from dew.telemetry.benchmark import carried
from dew.telemetry.instrumentation import (
    BOUND_MARKS,
    compilation_cache_counts,
    count_compilation_cache,
    logged_utilization,
)
from dew.telemetry.profile import Profiler
from dew.telemetry.trace import communication_totals, device_events, window_split
from dew.training import Trainer
from dew.training.ahead import AheadOfTime
from dew.training.interval import throughput
from dew.training.memory import recompute_record
from dew.training.runtime import prepare_process
from dew.training.tracker import Tracker, WandbTracker

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
from benchmark_models import ahead_of, build_trainer, compiled_step, global_batch, timed_steps, trainer_mesh
from trace_window import kernel_category, kernel_totals

Row = dict[str, object]


def cases_to_json(cases: Sequence[Case]) -> str:
    """`cases` as the JSON list `cases_from_json` reads back, each its record (`Case.record`)."""
    return json.dumps([case.record() for case in cases])


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
        str_from_instance=lambda cases: [cases_to_json(cases)],
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
    tracked: str | None = None
    """A W&B project each case's row is logged into once every case is measured, outside every timed
    window, one step a case, its numbers named by its architecture (`tracked`): an operational run's MFU
    and step time where a training run's land. Unset, nothing is logged."""
    profile_dir: str | None = None
    """Trace `profile_steps` more steps under this directory with `dew.Profiler`
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
    process that measures a case: a diagnostic's swapped implementation (tools/benchmark_depthwise.py's
    lax form)."""
    peak_run: bool = False
    """A case run in a process of its own (`measured_row`), its stage tools/peak_memory.py's
    (`PEAK_STAGE`): the warm stage compiles and profiles nothing; the measure stage measures, its peak read
    by peak_memory as it exits. Its rows go to `json_out` with the stage's name after it."""
    ahead: str | None = None
    """A directory of steps compiled ahead (`compile_ahead`, a case's file named by `ahead_name`): each case
    runs its step as its first where it is that program, so its GPU compiles none of it; a case it lacks is
    refused (`measure`)."""
    compile_ahead: str | None = None
    """Compile each case's step ahead of its run into this directory, for one compile-only GPU of
    `generation` holding `capacity_gib` (`benchmark_models.ahead_of`), and measure nothing: a GPU node's
    prepare, on its CPU, for its run's `ahead`. A step holding a kernel this host cannot lower is refused
    (`selection.Unlowerable`): the rest saved, the prepare fails, and no run takes the directory as whole."""
    generation: str | None = None
    """The CUDA generation `compile_ahead` compiles for (sm80, sm90)."""
    capacity_gib: int | None = None
    """The GPU memory `compile_ahead` plans for, in GiB."""


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
    outside the plan, which `measured_row` reads."""
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


def training_work(trainer: Trainer, case: Case) -> tuple[float | None, str | None]:
    """The objective's training FLOPs on the case's batch, train/mfu's numerator; None and the model's
    reason where it has no count (`MissingFlopsCount`)."""
    try:
        return trainer.objective.training_flops(None, global_batch(case)), None
    except MissingFlopsCount as missing:
        return None, str(missing)


def rates(trainer: Trainer, rows: int, steps: int, elapsed: float,
          work: float | None) -> tuple[float | None, float | None]:
    """`steps` steps of `rows` rows over `elapsed` seconds, each `work` training FLOPs, rated as the
    trainer's interval rates them (`dew.training.interval.throughput`): the step's HLO utilization, a
    bound named by what its count is (`StepFlops.bound`: a lower bound where launches' work went
    uncounted, an upper bound where a loop was counted at its stated bound), and its MFU, the
    work over the mesh's peak; each None where nothing counts it."""
    flops = trainer.step_flops
    scalars = throughput(elapsed, steps, rows * steps, None if flops is None else steps * flops,
                         trainer.device_mesh.devices.size,
                         training_flops=None if work is None else steps * work)
    hfu = logged_utilization(scalars, "hfu")
    return None if hfu is None else hfu[0], scalars.get("train/mfu")


def measured_row(case: Case, config: BenchmarkConfig) -> Row:
    """A case's row, measured in processes of its own (tools/peak_memory.py), its step handed to both where
    `ahead` holds it: the first compiles the rest into a cache, its compile the row's (`compile_cached` where
    that cache gave it its programs: a later round of a gate's arm); the second loads every program from it,
    times the step, profiles it, and reads each device's high-water as it exits. Where the second compiled a
    program its cache did not give it, its row stands with no peak, the compiles counted
    (`compiled_while_measured`): the compile joined the high-water, and its warm-up the timing's. This process
    opens no backend, which would hold the device the case needs."""
    compiled = 0
    with tempfile.TemporaryDirectory(prefix="dew-step-") as directory:
        out = os.path.join(directory, "rows.json")
        try:
            peaks = peak_memory.measured(
                [os.path.abspath(__file__), "--cases", cases_to_json([case]),
                 "--dtype", config.dtype, "--attention-impl", config.attention_impl, "--warmup",
                 str(config.warmup), "--steps", str(config.steps),
                 "--fixed-batch" if config.fixed_batch else "--no-fixed-batch", "--json-out", out,
                 *(["--xla-flags", config.xla_flags] if config.xla_flags else []),
                 *(["--within", config.within] if config.within else []),
                 *(["--ahead", config.ahead] if config.ahead else []),
                 *(["--profile-dir", config.profile_dir, "--profile-steps", str(config.profile_steps)]
                   if config.profile_dir else []), "--peak-run"])
        except peak_memory.CompiledWhileMeasured as missed:
            peaks, compiled = None, missed.compiled
        (warm,), (row,) = (json.loads(Path(f"{out}.{stage}").read_text()) for stage in ("warm", "measure"))
    return {**row, "compile_seconds": warm["compile_seconds"],
            "compile_cached": warm["compile_cache_hits"] > 0, "compiled_while_measured": compiled,
            "peak_device_bytes": None if peaks is None or None in peaks else max(
                peak for peak in peaks if peak is not None),
            "peak_bytes_by_device": peaks}


def ahead_name(case: Case) -> str:
    """The file of `case`'s step compiled ahead in a directory of them (`compile_ahead`, `ahead`): its
    case's own JSON's digest, so the run reads the step its prepare compiled for exactly that case."""
    return f"{hashlib.sha256(cases_to_json([case]).encode()).hexdigest()[:16]}.dew-aot"


WINDOW_PARTS = 6
"""How many parts a timed window is cut into, each closed by one wait (`window_parts`)."""


def window_parts(steps: int) -> list[int]:
    """`steps` cut into at most `WINDOW_PARTS` parts of as near equal sizes, the larger first, none empty."""
    count = min(WINDOW_PARTS, steps)
    return [steps // count + (index < steps % count) for index in range(count)]


GPU_IDLE = 0x1
"""nvidia-smi's clocks_event_reasons bit for a GPU with nothing to run (`DeviceSampler.summary` masks it)."""


class DeviceSampler:
    """Each GPU sampled by nvidia-smi every `PERIOD_MS` while the context is open (its SM clock, power
    draw, temperature and active clock event reasons), so a window's GPU is read under its load and not at
    the idle moments before or after it; nothing sampled on a host without nvidia-smi. Its process runs on
    the host's CPU and touches no timed work."""

    PERIOD_MS = 250
    FIELDS = ("sm_mhz", "power_w", "temperature_c", "event_reasons")

    def __enter__(self) -> "DeviceSampler":
        try:
            self.process = subprocess.Popen(
                ["nvidia-smi",
                 "--query-gpu=timestamp,index,clocks.sm,power.draw,temperature.gpu,clocks_event_reasons.active",
                 "--format=csv,noheader,nounits", f"-lms={self.PERIOD_MS}"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except OSError:
            self.process = None
        self.lines = ""
        return self

    def __exit__(self, *exc) -> None:
        if self.process is not None:
            # SIGINT, as a user's Ctrl-C ends nvidia-smi's loop: it flushes what it sampled before it exits.
            self.process.send_signal(signal.SIGINT)
            self.lines, _ = self.process.communicate(timeout=30)

    def summary(self, window: tuple[datetime.datetime, datetime.datetime] | None = None
                ) -> dict[str, dict[str, object]] | None:
        """Each GPU's samples taken within `window` (host-clock times, nvidia-smi stamping each sample in
        local time; every sample where none is given): their count, the least and greatest of each reading,
        and every clock event reason any sample held (nvidia-smi's bitmask, or'd); None where nothing was
        sampled in it."""
        gpus: dict[str, list[list[str]]] = {}
        for line in self.lines.splitlines():
            stamp, index, *values = (value.strip() for value in line.split(",")) if line.count(",") > 1 \
                else ("", "", "")
            if len(values) != len(self.FIELDS):
                continue
            try:
                taken = datetime.datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f")
            except ValueError:
                continue
            if window is None or window[0] <= taken <= window[1]:
                gpus.setdefault(index, []).append(values)
        if not gpus:
            return None

        def span(rows: list[list[str]], column: int) -> list[float] | None:
            numbers = [float(row[column]) for row in rows if row[column].replace(".", "", 1).isdigit()]
            return [min(numbers), max(numbers)] if numbers else None

        # GpuIdle (0x1) said by a sample taken between steps is no reason the clock was held down.
        def reasons(rows: list[list[str]]) -> str:
            held = 0
            for row in rows:
                held |= int(row[3], 16) & ~GPU_IDLE if row[3].startswith("0x") else 0
            return hex(held)

        return {index: {"samples": len(rows),
                        **{name: span(rows, column) for column, name in enumerate(self.FIELDS[:3])},
                        "event_reasons": reasons(rows)}
                for index, rows in gpus.items()}


def timed_parts(step, state, parts: list[int]):
    """`sum(parts)` steps dispatched back to back from `state`, each part's last loss handed as it is
    dispatched to a thread that awaits each in turn and stamps the seconds since the first dispatch as it
    becomes ready: no wait inside the window, and a part's stamp its readiness whether the host's dispatch or
    the device is the slower (a host-bound step's parts all ready before a wait after the whole dispatch
    could read them). Returns the last state, loss and finiteness, and each part's (steps, seconds, the
    host's seconds dispatching it): a part slow in both was the host's, one slow only to be ready the
    device's."""
    ready: queue.Queue = queue.Queue()
    ends: list[float] = []

    def waited() -> None:
        while (loss := ready.get()) is not None:
            jax.block_until_ready(loss)
            ends.append(time.perf_counter() - started)

    waiter = threading.Thread(target=waited, name="timed-parts")
    started = time.perf_counter()
    waiter.start()
    carry: tuple[Any, Any, Any] = (state, None, None)
    dispatched = []
    try:
        for size in parts:
            for _ in range(size):
                carry = step(carry[0])
            dispatched.append(time.perf_counter() - started)
            ready.put(carry[1])
    finally:
        ready.put(None)
        waiter.join()
    jax.block_until_ready(carry)
    begins, sent = [0.0, *ends[:-1]], [0.0, *dispatched[:-1]]
    return (*carry, [(size, end - begun, out - went) for size, end, begun, out, went
                     in zip(parts, ends, begins, dispatched, sent, strict=True)])


def tuning_digest() -> dict[str, object] | None:
    """The XLA per-fusion autotuning results' directory XLA_FLAGS names
    (--xla_gpu_per_fusion_autotune_cache_dir) as it stands at the row: its result files' count and the
    sha256 of their names and contents. The directory is a node's, shared by its jobs, so the digest names
    what it held then, not the picks this step alone used; None where no such directory is named."""
    flags = os.environ.get("XLA_FLAGS", "").split()
    option = "--xla_gpu_per_fusion_autotune_cache_dir="
    named = [flag.removeprefix(option) for flag in flags if flag.startswith(option)]
    if not named or not named[-1]:
        return None
    directory = Path(named[-1])
    files = sorted(path for path in directory.rglob("*.textproto")
                   if path.is_file()) if directory.is_dir() else []
    digest = hashlib.sha256()
    for path in files:
        digest.update(str(path.relative_to(directory)).encode() + b"\0" + path.read_bytes() + b"\0")
    return {"directory": str(directory), "results": len(files), "sha256": digest.hexdigest()}


def timed_window(step, state, config: BenchmarkConfig):
    """`config.steps` steps of `step` from `state` timed as a run dispatches them, after at least one warm
    step: dispatched whole and read in parts (`WINDOW_PARTS`, `timed_parts`), each part's last loss
    awaited in turn, so no wait drains the device inside the window, and a window that runs slow where it
    starts (the device or its allocator warming) shows as a series that falls, where one running another
    program stays slow throughout. The GPUs sampled from before the warm step, so nvidia-smi's start
    (NVML's, up to half a second) is behind it, and summarized over the window alone (`DeviceSampler`).
    Returns the last state, loss and finiteness, each part's (steps, seconds, dispatch seconds) and the
    GPUs' summary."""
    with DeviceSampler() as sampler:
        state, _, _, _ = timed_steps(step, state, max(config.warmup, 1))
        began = datetime.datetime.now()
        state, loss, is_finite, parts = timed_parts(step, state, window_parts(config.steps))
        ended = datetime.datetime.now()
    return state, loss, is_finite, parts, sampler.summary((began, ended))


def window_record(parts, synced: list[float], device, hits: float) -> dict:
    """A row's diagnostics of its timed window, beside its step's time (the window's whole elapsed time):
    each part's ms a step in order and its host dispatch's (a part slow in both was the host's, one slow
    only to be ready the device's), every synced step's ms in order (2026-11-14: a perf gate round's first
    arm read 40% slow on the very tree its others read), the GPUs under the window's load (`DeviceSampler`),
    JAX's persistent-cache hits while the step was compiled (its state's and other programs' among them),
    and the XLA autotuning results' directory as it stood at the row (`tuning_digest`)."""
    return {"window_ms_series": [round(seconds / size * 1e3, 3) for size, seconds, _ in parts],
            "window_dispatch_ms_series": [round(sent / size * 1e3, 3) for size, _, sent in parts],
            "synced_ms_series": [round(seconds * 1e3, 3) for seconds in synced],
            "device_window": device, "compile_cache_hits": hits, "tuning": tuning_digest()}


def measure(case: Case, config: BenchmarkConfig) -> Row:
    """Warm up, then time the compiled step over a fixed number of steps, in this process."""
    if config.steps < 1:
        raise ValueError(f"--steps must be at least 1, got {config.steps}")
    found = None if config.ahead is None else Path(config.ahead) / ahead_name(case)
    # A directory handed holds each case's step, its prepare failing any it could not compile: a case it
    # lacks is refused, never compiled here in its place (row 10796).
    if found is not None and not found.is_file():
        raise SystemExit(f"{found}: no step compiled ahead for {case.architecture}, in the directory handed")
    ahead = AheadOfTime.load(found) if found is not None else None
    # A step compiled for as many devices as this run's is its program; one for another count is not handed.
    handed = ahead if ahead is not None and ahead.devices == jax.device_count() else None
    trainer = build_trainer(case, config.attention_impl, ahead=handed)

    count_compilation_cache()
    hits = compilation_cache_counts().get("train/cache_hits", 0.0)
    with compiled_step(case, trainer, fixed_batch=config.fixed_batch) as compiled:
        state, step, compile_seconds = compiled.state, compiled.step, compiled.compile_seconds
        hits = compilation_cache_counts().get("train/cache_hits", 0.0) - hits
        state, loss, is_finite, parts, device = timed_window(step, state, config)
        elapsed = sum(seconds for _, seconds, _ in parts)

        # A second window of the same length, waiting on every step, for the
        # spread. The loop above dispatches asynchronously on purpose, as a run
        # does, so timing its individual iterations would time the dispatch and
        # not the step. These per-step numbers are a different quantity from
        # ms_per_step above, and each carries one synchronisation.
        (state, loss, is_finite), synced = carried(lambda carry: step(carry[0]), (state, loss, is_finite),
                                                   config.steps, each=True)
        p10, p50, p90 = np.percentile(np.asarray(synced) * 1e3, [10, 50, 90])
        # Every step of the warm, timed and synced windows valid, read once they are over (`nonfinite`): a
        # timing from a window any step of which went non-finite is no measurement.
        nonfinite = compiled.nonfinite()

        timeline = {}
        # A case's warm stage (`measured_row`) fills the cache; its measure stage profiles.
        if config.profile_dir and os.environ.get(peak_memory.PEAK_STAGE) != "warm":
            # After the timed windows, so the trace's own overhead is not in
            # them. Each process of a pool traces its own devices into a
            # directory of its own (`Profiler`), with its reports, which
            # tools/attribute_step.py reads (`--trace`); the timeline reads
            # the trace alone.
            with Profiler(os.path.join(config.profile_dir, case.label.replace(" ", "_"))) as capture:
                for _ in range(config.profile_steps):
                    state, loss, is_finite = step(state)
                loss.block_until_ready()
            (directory,) = (str(captured) for captured in capture.captures)
            planes = device_events(directory)[0]
            timeline = {**device_timeline(directory, config.profile_steps, planes=planes),
                        **communication(directory, config.profile_steps, planes=planes)}
        flops = trainer.step_flops
        step_time = elapsed / config.steps
        training_flops, uncounted = training_work(trainer, case)
        utilization, mfu = rates(trainer, case.batch_size, config.steps, elapsed, training_flops)
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
            # Whether the step timed is the one compiled ahead (`ahead`), which compiled nothing here.
            "ahead_step": handed is not None,
            # The rung the trainer compiled the step under, the model's own
            # or a stronger one where the step did not fit.
            "remat": recompute_record(trainer.objective),
            "ms_per_step": round(step_time * 1e3, 3),
            **window_record(parts, synced, device, hits),
            "p10_ms": round(float(p10), 3),
            "p50_ms": round(float(p50), 3),
            "p90_ms": round(float(p90), 3),
            "samples_per_sec": round(case.batch_size / step_time, 2),
            "tokens_per_sec": round(case.batch_size * case.seq_len / step_time, 1) if case.is_lm else None,
            "flops_per_step": None if flops is None else flops.hardware,
            "utilization": utilization,
            # What `utilization` and `flops_per_step` are (`StepFlops.bound`), as the trainer names its
            # train/hfu: a lower bound where a launch's work went uncounted (`unsized_kernels`, the launches
            # a step runs whose work no count holds), an upper bound where a loop was counted at its stated
            # bound, an estimate where both, else "" (the step's own count).
            "utilization_bound": None if flops is None else flops.bound,
            "unsized_kernels": None if flops is None else flops.unsized,
            "training_flops": training_flops,
            # Why the model counts no training work, where it counts none, so no MFU is reported.
            "training_flops_missing": uncounted,
            "mfu": mfu,
            "planned_device_bytes": planned,
            "loss": float(loss),
            "finite": nonfinite == 0 and bool(is_finite),
            "nonfinite_steps": nonfinite,
            **timeline,
        }
    return row


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
                # A count shown as the bound it is (`BOUND_MARKS`).
                if key in ("flops_per_step", "utilization"):
                    text = BOUND_MARKS[str(row["utilization_bound"])] + text
            else:
                text = fmt.format(value)
            cells.append(text.ljust(width) if key == "architecture" else text.rjust(width))
        lines.append(" ".join(cells))
    return "\n".join(lines)


def run(config: BenchmarkConfig, apart: bool) -> list[Row]:
    """Every case measured: each in processes of its own (`measured_row`) where `apart`, this process
    opening no backend; else in this one, a pool's process or a case's own (`peak_run`), with no peak."""
    rows: list[Row] = []
    # Every process of a pool measures its own devices; process 0 speaks.
    speaker = apart or jax.process_index() == 0
    out = config.json_out
    stage = os.environ.get(peak_memory.PEAK_STAGE, "measure") if config.peak_run else None
    if stage and out:
        out = f"{out}.{stage}"
    for case in build_cases(config):
        # The trainer narrates state generation and input shapes per case,
        # which buries the numbers this tool exists to print.
        sink = (contextlib.redirect_stdout(io.StringIO()) if config.quiet
                else contextlib.nullcontext())
        with sink:
            row = (measured_row(case, config) if apart
                   else {**measure(case, config), "peak_device_bytes": None, "peak_bytes_by_device": None})
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
        if out:
            # A GPU sweep is minutes of compilation per case; rewriting the
            # file as each case lands means an interrupted sweep still keeps
            # the cases it did measure.
            write_atomically(out, json.dumps(rows, indent=2))
    return rows


def entered(within: str | None) -> contextlib.AbstractContextManager:
    """The context `within` (`BenchmarkConfig.within`) names, or none."""
    if within is None:
        return contextlib.nullcontext()
    module, _, function = within.partition(":")
    return getattr(importlib.import_module(module), function)()


def tracked(rows: list[Row], tracker: Tracker) -> None:
    """`rows`' numbers logged into `tracker` and it closed, one step a row in their order, each named by its
    row's architecture: after every case's measurement, so no timed window holds it (row 285)."""
    for step, row in enumerate(rows):
        tracker.log({f"{row['architecture']}/{key}": float(value) for key, value in row.items()
                     if isinstance(value, (int, float)) and not isinstance(value, bool)}, step)
    tracker.close()


def main(config: BenchmarkConfig) -> list[Row]:
    if config.compile_ahead is not None:
        if config.generation is None or config.capacity_gib is None:
            raise SystemExit("--compile-ahead names the GPU it compiles for: --generation and --capacity-gib")
        Path(config.compile_ahead).mkdir(parents=True, exist_ok=True)
        refused = []
        for case in build_cases(config):
            try:
                ahead = ahead_of(case, config.generation, config.capacity_gib << 30, config.attention_impl)
            except Unlowerable as refusal:
                refused.append(f"{ahead_name(case)} ({case.architecture}): {refusal}")
                continue
            ahead.save(Path(config.compile_ahead) / ahead_name(case))
        # Each case this host compiled is kept, and a refused one fails the prepare: a run handed this
        # directory takes each case's step from it, never its own compile in its place (row 10796).
        if refused:
            raise SystemExit(f"not compiled ahead, so this prepare fails: {'; '.join(refused)}")
        return []
    # Each case in processes of its own (`measured_row`), this one opening no backend, which would hold the
    # device they need; a pool's processes and a case's own measure in process.
    apart = not (joins_pool(None, detected_environment()) or config.peak_run)
    if not apart and not config.peak_run and not jax.config.jax_compilation_cache_dir:
        # A pool's step compile seconds are a cold compile's, which no cached executable stands in for. A
        # caller that names its cache (JAX_COMPILATION_CACHE_DIR) keeps it, and its own switch
        # (JAX_ENABLE_COMPILATION_CACHE).
        jax.config.update("jax_enable_compilation_cache", val=False)
    if not apart:
        # Joins a `dew launch` pool when one launched this process, and applies
        # the flags before the backend opens either way.
        prepare_process(xla_flags=config.xla_flags)
    speaker = apart or jax.process_index() == 0
    if speaker:
        print(f"dtype {config.dtype}, attention_impl {config.attention_impl}, "
              f"XLA_FLAGS {os.environ.get('XLA_FLAGS', '')!r}")
    with contextlib.nullcontext() if apart else entered(config.within):
        rows = run(config, apart)
    if not speaker:
        return rows
    if rows:
        print(f"Devices: {rows[0]['devices']} x {rows[0]['device_kind']}, {rows[0]['processes']} process(es)")
    print()
    print(format_table(rows))
    if config.tracked:
        tracked(rows, WandbTracker(config.tracked, name=f"benchmark_step {config.preset}"))
    if config.json_out:
        print(f"\nWrote {config.json_out}")
    else:
        print()
        print(json.dumps(rows, indent=2))
    return rows


if __name__ == "__main__":
    # A row any of whose steps went non-finite fails the run, its rows written first.
    if invalid := [row["architecture"] for row in main(tyro.cli(BenchmarkConfig)) if not row["finite"]]:
        sys.exit(f"steps went non-finite in {', '.join(map(str, invalid))}: no timing of theirs is a measurement")
