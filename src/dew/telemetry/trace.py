"""Where a device's traced time went: the one reading of a capture's kernel intervals, for Dew and its tools.

A capture's kernels are `(name, start, end)` records a device, JAX's from its xplane
(`device_events`), torch.profiler's from its trace (tools/reference_runs). Busy time is their
union, so kernels on overlapping streams count once; the window runs from the first kernel's start
to the last one's end, and idle is what no kernel covers (`window_split`). XProf's self-time
tables sum each op's own time, which counts concurrent kernels twice: they attribute time to ops,
they never measure a device's busy time.

The standard library only at import (jax.profiler loads inside `device_events`): a torch environment without
Dew loads this file by its path (tools/reference_runs/common.py), and Dew's trainer imports telemetry/."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

NCCL_COLLECTIVES = (
    "Broadcast", "Reduce", "AllGather", "AllGatherV", "ReduceScatter", "AllReduce", "SendRecv")
"""The collectives NCCL names its kernels after (`ncclDevKernel_AllGather_RING_LL`,
`ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL`, `ncclDevKernel_SendRecv` for a collective-permute and an
all-to-all; `ncclKernel_` before NCCL 2.19). NCCL states them in its build alone, so they are read from its
source at v2.32.3-1, the release the GPU locks install: src/device/generate.py's `all_colls`."""

COLLECTIVES = ("all-reduce", "all-gather", "reduce-scatter", "all-to-all", "ragged-all-to-all",
               "collective-broadcast", "collective-permute", "collective-reduce", "send", "recv")
"""The HLO ops that move data between devices, as XLA prints them. XLA runs some as an async pair,
`<op>-start` and `<op>-done`, and wraps others in `async-start` over a computation that holds the op
itself. jaxlib states an op's opcode (`HloOpcode`) but neither its printed name nor which ones communicate,
so they are read from XLA's source at d867969800d4 (the revision jaxlib 0.12.0.1 builds):
xla/hlo/utils/hlo_query.cc's `IsCollectiveCommunicationOp` and the send and recv its async
collectives take (`IsAsyncCollectiveStartOp`), each spelled as xla/hlo/ir/hlo_opcode.h prints it."""


REDUCING = ("all-reduce", "reduce-scatter", "collective-reduce")
"""The collectives (`COLLECTIVES`) that compute as they move data, reducing it; every other moves alone."""

TRANSFERS = ("copy", "infeed", "outfeed")
"""The HLO ops other than a collective that move data and compute none: a copy, and a host's infeed and
outfeed."""


def moves_only(op: str) -> bool:
    """Whether the HLO op `op` (an opcode as XLA prints it) moves data and computes none, or runs half of one
    (`<op>-start`, `<op>-done`): a transfer (`TRANSFERS`) or a collective (`collective`) that reduces nothing
    (`REDUCING`). An `async-start` is none, its wrapped op unnamed by its opcode."""
    kind = collective(op)
    if kind is not None:
        return kind not in REDUCING
    return op.removesuffix("-start").removesuffix("-done") in TRANSFERS


def collective(op: str) -> str | None:
    """The collective (`COLLECTIVES`) HLO op `op` is, or runs half of; None for any other."""
    kind = op.removesuffix("-start").removesuffix("-done")
    return kind if kind in COLLECTIVES else None


def communicates(name: str) -> bool:
    """Whether a kernel named `name` moves data between devices: NCCL's (torch's and a GPU's), or an HLO
    collective's (`collective`), as a TPU's `XLA Ops` line names its ops (`all-reduce.1`)."""
    return "nccl" in name.lower() or collective(name.split(".", 1)[0]) is not None


@dataclass(frozen=True)
class Kernel:
    """One kernel's event as a trace holds it, read out of JAX's native ProfileEvent, whose fields live
    only as long as the ProfileData it came from: read after that is freed, they crash the reader."""

    name: str
    start_ns: float
    end_ns: float
    duration_ns: float
    stats: tuple[tuple[str, object], ...]


def device_events(directory: str | Path, tpu_line: str = "XLA Ops"
                  ) -> tuple[dict[str, list[Kernel]], list[str]]:
    """The kernel events of the newest JAX trace under `directory`, by device
    plane (`/device:GPU:0`) in the trace's order, and the names of every
    device line.

    A GPU plane holds one line per CUDA stream; its `XLA Ops` and `XLA
    Modules` lines are derived from the same kernels, and a module's span
    runs across the idle gaps between them, so only stream lines count. A TPU
    plane runs its HLO ops on one `XLA Ops` line, named after the HLO
    instruction (`fusion.12`, `convolution.3`); `tpu_line` reads another of its
    lines instead (`XLA Modules`, a program's runs, each named after its module)."""
    return planes_of(newest(directory), tpu_line)


def planes_of(capture, tpu_line: str = "XLA Ops") -> tuple[dict[str, list[Kernel]], list[str]]:
    """`device_events`' events and lines of one read capture (`newest`'s), so several views of one capture
    are read from it alone."""
    planes: dict[str, list[Kernel]] = {}
    lines = set()
    # The trace's planes, lines and events are views into `capture`, which outlives the loop.
    for plane in capture.planes:
        match = re.match(r"/device:(GPU|TPU):\d+", plane.name)
        if match is None:
            continue
        for line in plane.lines:
            lines.add(f"{plane.name}:{line.name}")
            if (line.name.lower().startswith("stream") if match.group(1) == "GPU"
                    else line.name == tpu_line):
                planes.setdefault(plane.name, []).extend(
                    Kernel(event.name, event.start_ns, event.end_ns, event.duration_ns, tuple(event.stats))
                    for event in line.events)
    return planes, sorted(lines)


def newest(directory: str | Path):
    """The newest JAX trace under `directory`, read (`jax.profiler.ProfileData`)."""
    from jax.profiler import ProfileData

    traces = sorted(Path(directory).rglob("*.xplane.pb"), key=os.path.getmtime)
    if not traces:
        raise FileNotFoundError(f"no xplane.pb under {directory}")
    return ProfileData.from_file(str(traces[-1]))


EXECUTE = "PJRT_LoadedExecutable_Execute"
"""The PJRT C API's execute as a host's XProf names it: its caller's thread's linkage event (`EXECUTE
linkage`) the flow's producer, the runtime thread's execute its consumer (`enqueued`)."""


def enqueued(capture, annotation: str) -> dict[int, set[int]]:
    """The TPU programs a host call enqueued within its event `annotation` (a `jax.profiler.TraceAnnotation`)
    in `capture` (`newest`'s), by device ordinal: each run id its `DoEnqueueProgram`s state, which the
    device's run of it on `XLA Modules` states too. A v5e's XProf links the call to them by its flows: the
    annotation's own thread holds, within it, the PJRT C API's execute linkage (its `_pt`/`_p` flow), the
    runtime's thread the execute that consumes it (its `_ct`/`_c`), and the call's work is that execute's
    span and, flow by flow, the span of each event consuming a flow an event of it produces, on whatever
    thread: an enqueue the execute issues itself (M-FS-104) or one a queue's thread issues once its inputs'
    transfer is done (`tpu::System::Execute` to `=>IssueSequencedEvent`, M-FS-105, M-FS-106: an executable
    taking a tuple index table). Another call's linkage, on another thread, roots none of it. The host's
    clock is not the device's, so the host's events alone are read by time (`_flows`)."""
    roots, consumers, marks = _flows(capture, annotation)
    work = [(thread, event) for flow in roots for thread, event in consumers.get(flow, ())
            if event.name == EXECUTE]
    followed = set(roots)
    found: dict[int, set[int]] = {}
    while work:
        thread, outer = work.pop()
        for event, produced, run in marks[thread]:
            if not _within(event, outer):
                continue
            if run is not None:
                found.setdefault(run[0], set()).add(run[1])
            if produced is not None and produced not in followed:
                followed.add(produced)
                work += consumers.get(produced, ())
    return found


def _within(event, outer) -> bool:
    return outer.start_ns <= event.start_ns and event.start_ns + event.duration_ns <= (
        outer.start_ns + outer.duration_ns)


def _flows(capture, annotation: str) -> tuple[set, dict[tuple, list], list[list]]:
    """What `enqueued` walks of the host's events starting within an `annotation` event's time, each one's
    stats read once: the flows of the execute linkages within the annotation on its own thread (the call's
    roots), every event consuming a flow by that flow (with its thread's index), and each thread's events
    that produce a flow or enqueue a run, each `(event, flow or None, (device, run id) or None)`."""
    threads = [list(line.events) for plane in capture.planes if plane.name.startswith("/host:")
               for line in plane.lines]
    annotated = [[event for event in events if event.name == annotation] for events in threads]
    spans = [(event.start_ns, event.start_ns + event.duration_ns) for events in annotated for event in events]
    roots: set = set()
    consumers: dict[tuple, list] = {}
    marks: list[list] = []
    for thread, events in enumerate(threads):
        held = []
        for event in events:
            if not any(lo <= event.start_ns <= hi for lo, hi in spans):
                continue
            stats = dict(event.stats)
            if "_c" in stats:
                consumers.setdefault((stats.get("_ct"), stats["_c"]), []).append((thread, event))
            produced = (stats.get("_pt"), stats["_p"]) if "_p" in stats else None
            if produced is not None and event.name == f"{EXECUTE} linkage" and any(
                    _within(event, outer) for outer in annotated[thread]):
                roots.add(produced)
            device, run = stats.get("device_ordinal"), stats.get("run_id")
            ran = ((device, run) if event.name == "DoEnqueueProgram" and isinstance(device, int)
                   and isinstance(run, int) else None)
            if produced is not None or ran is not None:
                held.append((event, produced, ran))
        marks.append(held)
    return roots, consumers, marks


def run_of(event: Kernel) -> int | None:
    """The run a TPU program's `XLA Modules` event is, by its `run_id` stat (`enqueued`'s)."""
    return next((value for name, value in event.stats if name == "run_id" and isinstance(value, int)), None)


def program_of(event: Kernel) -> str | None:
    """The XLA program a kernel ran for, from its `program_id` stat (XProf's hlo_stats name programs so); none
    for a host transfer, which no program launches."""
    return next((str(value) for name, value in event.stats if name == "program_id"), None)


def union[Time: (int, float)](intervals: Sequence[tuple[Time, Time]]) -> list[tuple[Time, Time]]:
    merged: list[tuple[Time, Time]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def overlap[Time: (int, float)](first: list[tuple[Time, Time]], second: list[tuple[Time, Time]]) -> Time:
    """Time two disjoint, sorted interval lists share."""
    shared, i, j = 0, 0, 0
    while i < len(first) and j < len(second):
        start, end = max(first[i][0], second[j][0]), min(first[i][1], second[j][1])
        shared += max(0, end - start)
        if first[i][1] < second[j][1]:
            i += 1
        else:
            j += 1
    return shared


def length[Time: (int, float)](intervals: list[tuple[Time, Time]]) -> Time:
    return sum(end - start for start, end in intervals)


def busy_time[Time: (int, float)](intervals: Sequence[tuple[Time, Time]]) -> Time:
    """Time covered by at least one kernel, counting overlapping streams once."""
    return length(union(intervals))


def _host_to_device(name: str) -> bool:
    """JAX's `MemcpyH2D`, torch's `Memcpy HtoD (Pinned -> Device)`."""
    lowered = name.lower()
    return "memcpyh2d" in lowered or "htod" in lowered


def window_split[Time: (int, float)](events: Sequence[tuple[str, Time, Time]]) -> dict[str, Time]:
    """One device's traced window from its `(name, start, end)` kernels.

    Times keep the input unit, JAX nanoseconds or torch microseconds.

    `window` runs from the first kernel's start to the last one's end and
    `busy` is the union of every kernel. A kernel is a collective where it moves
    data between devices (`communicates`: NCCL's, or an HLO collective a TPU's
    ops are named after) and compute otherwise: `compute` and `communication` are
    their unions, and `exposed_communication` the collective time no compute
    kernel ran beside, what overlap did not hide. Each of `NCCL_COLLECTIVES` gets
    the union of its own kernels. The idle rest is split by what ends each gap: a
    host-to-device copy (`idle_input`, the device waited on a batch) or
    anything else (`idle_host`, it waited on dispatch or host work). So the
    window is compute + exposed_communication + idle_input + idle_host.

    A collective's kernel spans its wait for the slowest peer as well as the
    transfer, so communication includes the skew between devices."""
    if not events:
        raise ValueError("no device kernels to split")
    ordered = sorted(events, key=lambda event: event[1])
    idle: dict[str, list[tuple[Time, Time]]] = {"idle_input": [], "idle_host": []}
    high = ordered[0][2]
    for name, start, end in ordered[1:]:
        if start > high:
            idle["idle_input" if _host_to_device(name) else "idle_host"].append((high, start))
        high = max(high, end)
    collectives = [(name, start, end) for name, start, end in ordered if communicates(name)]
    compute = union([(start, end) for name, start, end in ordered if not communicates(name)])
    every = union([(start, end) for _, start, end in collectives])
    figures = {
        "window": high - ordered[0][1], "busy": busy_time([(start, end) for _, start, end in ordered]),
        "compute": length(compute), "communication": length(every),
        "exposed_communication": length(every) - overlap(every, compute),
        **{key: length(spans) for key, spans in idle.items()}}
    for kind in NCCL_COLLECTIVES:
        figures[kind] = length(union([
            (start, end) for name, start, end in collectives
            if f"_{kind.lower()}_" in f"_{name.lower()}_".replace("(", "_")]))
    return figures


def window_totals[Time: (int, float)](
    devices: Iterable[Sequence[tuple[str, Time, Time]]],
) -> dict[str, Time]:
    """Sum each device's window split; device averaging belongs to the report."""
    totals: dict[str, Time] = {}
    for events in devices:
        for key, value in window_split(events).items():
            totals[key] = totals.get(key, 0) + value
    return totals


def hlo_collective(event: Kernel) -> str | None:
    """The HLO collective (`COLLECTIVES`) an event ran for: the op it is named after, as a TPU's ops are
    (`all-reduce.1`), or, for an NCCL kernel, XLA's hlo_op stat, which tells apart the all-to-all and
    collective-permute that NCCL runs as the same SendRecv kernel."""
    named = collective(event.name.split(".", 1)[0])
    if named is not None:
        return named
    for name, value in event.stats:
        if name == "hlo_op" and isinstance(value, str):
            spelled = value.replace("_", "-")
            return next((op for op in COLLECTIVES if spelled.startswith(op)), None)
    return None


def communication_totals(devices: Sequence[Sequence[Kernel]]) -> dict[str, float]:
    """Per-device window totals and the busy and exposed time of each HLO collective (`COLLECTIVES`)."""
    kernels = [[(event.name, event.start_ns, event.end_ns) for event in events] for events in devices]
    totals = window_totals(kernels)
    for events, records in zip(devices, kernels, strict=True):
        compute = union([(start, end) for name, start, end in records if not communicates(name)])
        ops = [hlo_collective(event) for event in events]
        for op in COLLECTIVES:
            spans = union([(start, end) for (_, start, end), of in zip(records, ops, strict=True)
                           if of == op])
            key = op.replace("-", "_")
            total = length(spans)
            totals[key] = totals.get(key, 0) + total
            totals[f"{key}_exposed"] = totals.get(f"{key}_exposed", 0) + (total - overlap(spans, compute))
    return totals
