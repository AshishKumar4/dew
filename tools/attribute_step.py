"""Where a profiled training step's device time and peak memory go, by model part.

    PYTHONPATH=src python tools/attribute_step.py --trace PROFILE_DIR --hlo HLO_DIR_OR_TGZ --out OUT.json
                                                  [--steps N] [--entry entry.json] [--module jit_step]

`--trace` is a dew.Profiler directory (or a capture inside one); the newest capture holding
`reports/hlo_stats.json` is read. Profile it with XLA_FLAGS=--xla_gpu_enable_command_buffer= :
inside CUDA command buffers XProf attributes every kernel to its module and hlo_stats is empty.
`--steps` is how many steps the capture traced; totals are divided by it.

`--hlo` is an XLA dump (`--xla_dump_to=... --xla_dump_hlo_as_text`), a directory or a .tgz of
one. The `<module>` compile the capture ran is read, the newest without `--trace`: its
optimized HLO (each op's opcode, a custom call's target, the ops a fusion runs), its
live-range dump's peak (the values live at the peak and XLA's breakdown of them by op_name)
and its buffer assignment (parameters, outputs, temporaries).

Time, from the XProf reports: device time a step by model part (GDN, attention, LoRA, MLP, head
and loss, embedding, norm, optimizer, other) and direction (forward, backward, neither); by kind,
read from the op itself (`kind_of`): GEMM, attention, Pallas/Triton, layout (transposes and
copies), slicing, gathers and scatters, reductions, elementwise, collectives; both together; GEMMs
by the time they lose against the device's peak (time - flops / peak); the costliest layout
moves and ops with framework name and source line, each with its HBM traffic a step (XProf's
measured HBM bandwidth over the op's time). Those are XProf's summed self times, which count
concurrent kernels each: where the time went, not how long the device was busy. The device's
time is the step program's kernel intervals in the capture's trace (`program_window`): its window,
busy time (their union), compute, exposed communication, other programs' time in its gaps and
idle; with `--entry`, the step's model FLOPs over that busy time against the peak, a diagnostic
beside the canonical MFU over wall time. Without `--hlo`, a custom call or a
fusion is kinded from its name alone, which cannot tell FlashAttention's call from any other.

Memory, from the dump: the peak's bytes by role (arguments: parameters, optimizer state, inputs;
saved forward activations; backward working set; rematerialized forward; unattributed) and model
part, and its largest buffers.

It reads text and JSON only, through Dew's own readers: the HLO parser the FLOP count uses
(`dew.telemetry.instrumentation.hlo_instructions`, `custom_call_kind`) and XProf's report tables
(`dew.telemetry.profile.report_table`).
"""

from __future__ import annotations

import argparse
import contextlib
import html
import itertools
import json
import re
import resource
import shutil
import sys
import tarfile
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

from dew.nn.kernels.generation import gpu_target
from dew.telemetry.instrumentation import HloInstruction, custom_call_kind, hlo_instructions
from dew.telemetry.profile import report_table
from dew.telemetry.trace import collective, device_events, overlap, program_of, union, window_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_window import program_times

PARTS = (
    # The decision head's own attention and norms are the head's; Qwen3.5 names every mixer
    # `self_attn`, so a delta net's ops are told apart by the file that traced them.
    ("head_loss", r"Head\)|Head/|JointSchemaHead|lm_head|logits|[Ll]oss|cross_entropy|log_softmax|"
                  r"[Dd]ecision"),
    ("optimizer", r"optax|adam|[Mm]uon|apply_updates|clip_by|update_moment|scale_by|opt_state|"
                  r"training/transaction\.py"),
    ("gdn", r"nn/linear\.py|nn/kernels/(fla_gdn|recurrence)|nn/kda\.py|GatedDeltaNet|gated_delta|chunk_delta|"
            r"KimiDelta|fla_chunk|fla_kkt|fla_recompute|fla_prepare"),
    ("lora", r"[Ll]o[Rr][Aa]|lora_a|lora_b"),
    ("norm", r"layernorm|_norm/|/norm\d?/|RMSNorm|LayerNorm"),
    ("attention", r"nn/attention\.py|nn/mixers/attention\.py|flash_attn|decode_attention|nn/rope\.py|"
                  r"[Aa]ttention|_attended|dot_product_attention|fmha|splash"),
    ("mlp", r"/mlp/|gate_proj|up_proj|down_proj|SwiGLU|experts|nn/moe\.py"),
    ("embedding", r"[Ee]mbed|table_rows|wte|tok_emb"),
)
CALL_KINDS = {"gemm": "gemm", "convolution": "gemm", "attention": "attention", "triton": "pallas"}
"""A custom call's kind here, from `custom_call_kind`'s: Pallas lowers to Triton calls."""
FUSED_KINDS = (
    ("gemm", {"dot", "convolution"}),
    ("gather_scatter", {"gather", "scatter"}),
    ("layout", {"transpose", "copy", "bitcast-convert"}),
    ("reduce", {"reduce", "reduce-window"}),
    ("slicing", {"concatenate", "slice", "dynamic-slice", "dynamic-update-slice", "pad"}),
)
"""A fusion's kind by the first of these its ops hold, an op itself's the same way; anything else
is elementwise. A transpose or a copy that a fusion carries is a layout move however much
arithmetic rides with it."""


def part_of(text: str) -> str:
    """An op's model part from its framework name and the source line that traced it."""
    for part, pattern in PARTS:
        if re.search(pattern, text):
            return part
    return "other"


def direction_of(name: str) -> str:
    if "transpose(jvp" in name:
        return "backward"
    if "jvp(" in name:
        return "forward"
    return "neither"


# ---- time, from XProf's reports --------------------------------------------------------------


def kind_of(row: dict, ops: dict[str, HloInstruction] | None = None) -> str:
    """An op's kind, read from the op: its opcode, a custom call's target, the ops a fusion runs.
    From the module (`hlo_instructions`) where it was given, else from the op's own expression,
    and a fusion from its name; never from the names of its operands, which the expression also
    holds."""
    name = str(row.get("hlo_op_name") or "")
    known = (ops or {}).get(name)
    if known is None:
        # The row's expression is one instruction, cut short: read as a module of it alone.
        alone = hlo_instructions(f"ENTRY %row () -> () {{\n  {row.get('hlo_op_expression') or ''}\n}}")
        found = next(iter(alone.values()), None)
        op, target = (found.op, found.target) if found else ("", "")
        fused = frozenset(re.sub(r"\.\d+$", "", name).split("_")) if op == "fusion" else frozenset()
    else:
        op, target, fused = known.op, known.target, known.fused
    if collective(op) is not None:
        return "collective"
    if op == "custom-call":
        if not target and name.startswith("pallas_call"):
            return "pallas"
        return CALL_KINDS.get(custom_call_kind(target), "custom-call")
    for kind, members in FUSED_KINDS:
        if (fused if op == "fusion" else {op}) & members:
            return kind
    if row.get("category") == "data formatting":
        return "layout"
    return "elementwise"


def source_of(row: dict) -> str:
    """The first frame in Dew's own source, else the first frame, of the op's source cell."""
    cell = " ".join(str(value) for value in row.values() if "source-info" in str(value))
    match = re.search(r"title='([^']*)'", cell) or re.search(r'title="([^"]*)"', cell)
    frames = html.unescape(match.group(1)).split("\n") if match else []
    own = [frame for frame in frames if "/dew/" in frame and "site-packages" not in frame]
    return (own or frames or [""])[0].split("/src/")[-1]


def capture_of(trace: Path) -> Path:
    found = sorted(trace.rglob("reports/hlo_stats.json"), key=lambda path: path.stat().st_mtime)
    if not found:
        raise SystemExit(f"no reports/hlo_stats.json under {trace}")
    return found[-1].parent.parent


def ms(times: dict) -> dict:
    """`times` in µs as ms, the largest first."""
    return {key: round(us / 1e3, 3) for key, us in sorted(times.items(), key=lambda item: -item[1])}


def by_majority(found: list[str], unknown: str) -> str:
    """The most common of `found` other than `unknown`, or `unknown`."""
    named = [each for each in found if each != unknown]
    return max(set(named), key=named.count) if named else unknown


def heaviest(groups: dict) -> list:
    """`groups` of times by key, the largest total first."""
    return sorted(groups.items(), key=lambda item: -sum(item[1].values()))


def program_runs(rows: list, step: str, steps: int, ops: dict[str, HloInstruction] | None) -> float:
    """How many times a step ran its program `step`: the count most of its ops show over the ops'
    own calls a run (`HloInstruction.runs`). XProf counts an op's kernels, not its calls; a
    cuBLAS GEMM that splits its reduction launches two, which read as twice its work."""
    known = {row_name: found for row in rows if str(row.get("program_id")) == step
             and (found := (ops or {}).get(row_name := str(row.get("hlo_op_name") or ""))) is not None
             and found.runs}
    counts = Counter(round(float(row.get("occurrences") or 0) / steps / known[name].runs, 3)
                     for row in rows if (name := str(row.get("hlo_op_name") or "")) in known)
    return counts.most_common(1)[0][0] if counts else 0.0


def program_window(capture: Path, step: str) -> dict[str, float] | None:
    """The step program's own device time, in nanoseconds, from the capture's kernel intervals
    (`dew.telemetry.trace`), averaged over the devices that ran it: the window from its first
    kernel's start to its last one's end; `busy`, their union, where concurrent kernels count once;
    its `compute` and the `exposed_communication` compute did not hide; and the window's gaps, split
    into the time other programs' kernels ran in them (`other_programs`) and the device's own
    `idle`. A host transfer, which no program launches, counts where it falls in the window. None
    where the capture holds no accelerator timeline (a CPU's)."""
    if not any(capture.rglob("*.xplane.pb")):
        return None
    planes, _ = device_events(capture)
    found = []
    for events in planes.values():
        mine = [(event.name, event.start_ns, event.end_ns) for event in events if program_of(event) == step]
        if not mine:
            continue
        low, high = min(start for _, start, _ in mine), max(end for _, _, end in mine)
        inside = [event for event in events if event.start_ns < high and event.end_ns > low]
        transfers = [(event.name, max(event.start_ns, low), min(event.end_ns, high)) for event in inside
                     if program_of(event) is None]
        split = window_split(mine + transfers)
        covered = union([(start, end) for _, start, end in mine + transfers])
        gaps = [(end, start) for (_, end), (start, _) in itertools.pairwise(covered)]
        others = overlap(gaps, union([(max(event.start_ns, low), min(event.end_ns, high)) for event in inside
                                      if program_of(event) not in (None, step)]))
        found.append({"window": split["window"], "busy": split["busy"], "compute": split["compute"],
                      "exposed_communication": split["exposed_communication"], "other_programs": others,
                      "idle": split["window"] - split["busy"] - others})
    if not found:
        return None
    return {key: sum(each[key] for each in found) / len(found) for key in found[0]}


def attribute_time(trace: Path, steps: int, model_flops: float | None,
                   ops: dict[str, HloInstruction] | None = None, device: str | None = None) -> dict:
    capture = capture_of(trace)
    reports = capture / "reports"
    rows = report_table(reports / "hlo_stats.json")[1]
    if not rows:
        return {"capture": str(capture), "error": "hlo_stats.json has no ops: profile with "
                "XLA_FLAGS=--xla_gpu_enable_command_buffer= so XProf can attribute kernels"}
    roofline = json.loads((reports / "roofline_model.json").read_text())
    roofline = roofline[0] if isinstance(roofline, list) else roofline
    # Dew's dense bf16 peak, the MFU denominator every run reports. XProf's roofline peak is a
    # different kind of figure (on an RTX PRO 6000, 3,742 TFLOP/s, sparse low precision), kept for
    # reference only: a device Dew's table does not name gets no peak-derived field at all.
    target = gpu_target(device) if device else None
    known = None if target is None else target.bf16_peak
    peak = known / 1e9 if known else None
    fp8_peak = None if target is None or target.fp8_peak is None else target.fp8_peak / 1e9
    peak_source = (f"dew.nn.kernels.generation.gpu_target for {device}" if known
                   else f"unknown device: {device}" if device else "no device named")
    xprof_peak = float(roofline.get("p", {}).get("peak_flop_rate", 0) or 0) or None
    programs = program_times(rows)
    step = max(programs, key=programs.__getitem__)
    runs = program_runs(rows, step, steps, ops)
    by_part: dict = defaultdict(lambda: defaultdict(float))
    by_kind: dict = defaultdict(float)
    by_part_kind: dict = defaultdict(lambda: defaultdict(float))
    by_source: dict = defaultdict(float)
    gemms, layouts, everything = [], [], []
    fp8_gflop = 0.0
    for row in (row for row in rows if str(row.get("program_id")) == step):
        name = str(row.get("tf_op_name") or "")
        us = float(row["total_self_time"]) / steps
        source = source_of(row)
        part, direction, kind = part_of(f"{name} {source}"), direction_of(name), kind_of(row, ops)
        known = (ops or {}).get(str(row.get("hlo_op_name") or ""))
        if not name and known is not None and known.traced:
            # A multi-output fusion carries no name of its own; its ops say what it computes.
            part = by_majority([part_of(each) for each in known.traced], "other")
            direction = by_majority([direction_of(each) for each in known.traced], "neither")
        if known is not None and any(re.search(dict(PARTS)["optimizer"], each) for each in known.reads):
            # An update is named by its root, the parameter it writes narrowed for the next step's
            # compute (training/narrow.py); the optimizer state it reads says what it is.
            part = "optimizer"
        by_part[part][direction] += us
        by_kind[kind] += us
        by_part_kind[part][kind] += us
        by_source[kind, source] += us
        # The compiled FLOPs over the measured time: the work done, counted from the op itself as the
        # step's FLOP count counts it. XProf's own rates are no measure of it: its model rate put a
        # cuBLAS weight-gradient GEMM of C6's Qwen3.5-0.8B at 550 TFLOP/s on an A100 (4x its work), and its
        # normalized rate guesses each op's precision (the later Qwen3.5-0.8B step's GEMMs at 1.127 of the peak).
        calls = runs * known.runs if known is not None else 0.0
        gflops = known.flops * calls / (us * 1e-6) / 1e9 if known is not None and us else 0.0
        # XProf's HBM bandwidth is GiB/s over the op's own time, so the op moves that much a step.
        hbm_mib = float(row.get("hbm_bw") or 0) * us * 1e-6 * 1024
        record = {"us": round(us, 1), "part": part, "direction": direction, "kind": kind,
                  "hlo": row.get("hlo_op_name"), "framework": name[:200], "count": row.get("occurrences"),
                  "gflops": round(gflops, 1), "hbm_mib": round(hbm_mib, 1), "bound_by": row.get("bound_by"),
                  "source": source}
        everything.append(record)
        if kind == "gemm":
            record["precision"] = gemm_precision(row, known)
            # Model FLOPs exclude remat's recompute, so an FP8 GEMM's recomputed copy is not charged.
            if record["precision"] == "fp8" and not (known is not None and known.recomputed):
                fp8_gflop += gflops * us * 1e-6
            # Each GEMM against its own precision's peak, as mfu_mixed counts it.
            limit = fp8_peak if record["precision"] == "fp8" else peak
            if limit:
                record["fraction_of_peak"] = round(gflops / limit, 3)
                record["lost_us"] = round(us * max(0.0, 1 - gflops / limit), 1)
                gemms.append(record)
        if kind == "layout":
            layouts.append(record)
    timeline = program_window(capture, step)
    figure = {} if timeline is None else {key: round(ns / steps / 1e6, 3) for key, ns in timeline.items()}
    judged_us = sum(record["us"] for record in gemms)
    judged = sum(record["us"] * record["fraction_of_peak"] for record in gemms)
    result = {
        "capture": str(capture), "steps": steps, "peak_gflops": peak, "peak_source": peak_source,
        "xprof_peak_gflops": xprof_peak, "step_program": step,
        "programs_ms": {program: round(us / 1e3 / steps, 3)
                        for program, us in sorted(programs.items(), key=lambda item: -item[1])[:6]},
        # The step program's device time from its kernel intervals (`program_window`): busy is their
        # union, the window busy + other programs' time in its gaps + idle. None without an accelerator
        # timeline. The by-part and by-kind times below are XProf's summed self times, which count
        # concurrent kernels each: attribution of where time went, not the device's busy time.
        **{f"device_{key}_ms_per_step": figure.get(key) for key in (
            "window", "busy", "compute", "exposed_communication", "other_programs", "idle")},
        "op_self_time_ms_per_step": round(sum(by_kind.values()) / 1e3, 3),
        "by_part_ms": {part: {direction: round(us / 1e3, 3) for direction, us in directions.items()}
                       for part, directions in heaviest(by_part)},
        "by_kind_ms": ms(by_kind),
        "by_part_kind_ms": {part: ms(kinds) for part, kinds in heaviest(by_part_kind)},
        "gemm_fraction_of_peak": round(judged / judged_us, 3) if judged_us else None,
        "gemm_lost_ms": round(sum(record["lost_us"] for record in gemms) / 1e3, 3) if peak else None,
        "gemms_by_lost_time": sorted(gemms, key=lambda record: -record["lost_us"])[:25],
        "layout_moves": sorted(layouts, key=lambda record: -record["us"])[:25],
        "top_ops": sorted(everything, key=lambda record: -record["us"])[:40],
        # A layer's op is one of as many as there are layers, each too small to be a top op: summed by
        # the line that wrote it, the time a reduction or a convert costs every layer shows.
        "by_source_ms": [{"kind": kind, "source": source, "ms": round(us / 1e3, 3)} for (kind, source), us
                         in sorted(by_source.items(), key=lambda item: -item[1])[:40]],
        "xprof_steps": report_table(reports / "input_pipeline_analyzer.json", 1)[1][:8],
    }
    if model_flops:
        # Utilization of the step's busy device time, the union of its kernels (`program_window`), a
        # diagnostic beside the canonical MFU over wall time (docs/design/training.md), none without an
        # accelerator timeline. It names its precision: mfu_bf16, model FLOPs over the bf16 peak, which
        # is utilization only when every GEMM is bf16 and otherwise a bf16-equivalent throughput;
        # mfu_mixed, each GEMM's FLOPs over its own precision's peak, the true utilization, null where
        # an FP8 GEMM runs on a device with no FP8 peak in Dew's table.
        seconds = timeline["busy"] / steps / 1e9 if timeline else None
        fp8 = fp8_gflop * 1e9
        throughput = round(model_flops / seconds / 1e12, 2) if seconds else None
        result.update(model_flops_per_step=model_flops, fp8_gemm_flops_per_step=fp8,
                      bf16_equiv_throughput_tflops=throughput)
        result["mfu_bf16_of_device_time"] = (
            round(model_flops / seconds / (peak * 1e9), 4) if peak and seconds else None)
        if seconds and peak and not fp8:
            result["mfu_mixed_of_device_time"] = result["mfu_bf16_of_device_time"]
        elif seconds and peak and fp8_peak and fp8 <= model_flops:
            share = ((model_flops - fp8) / (peak * 1e9) + fp8 / (fp8_peak * 1e9)) / seconds
            result["mfu_mixed_of_device_time"] = round(share, 4)
        else:
            result["mfu_mixed_of_device_time"] = None
    return result


def gemm_precision(row: dict, known: HloInstruction | None) -> str:
    """'fp8' for a GEMM that multiplies FP8 operands (cuBLASLt's f8 matmul, or f8 operand types in
    its expression), else 'bf16': the peak its FLOPs count against."""
    target = known.target if known is not None else ""
    expression = str(row.get("hlo_op_expression") or "")
    return "fp8" if "$f8" in target or re.search(r"\bf8e[45]m", expression) else "bf16"


# ---- memory, from XLA's dump ------------------------------------------------------------------


@contextlib.contextmanager
def dump_files(hlo: Path, module: str, program: str | None = None) -> Iterator[dict]:
    """The after-optimizations HLO, live-range, buffer-assignment and memory-usage-report
    files of `module`'s compile numbered `program`, the one a capture spent most time in,
    or of its newest without a capture, by suffix (the HLO's is `.txt`), as paths read a
    line at a time (`lines`): from a .tgz only those four are extracted, to a directory that
    lasts the context. A dump holds every compile of the step, each tens of megabytes: a
    plan compiles several candidates, and the newest need not be the one that ran."""
    suffixes = (".txt", "-live-range.txt", "-buffer-assignment.txt", "-memory-usage-report.txt")
    pattern = re.compile(rf"module_(\d+)\.{re.escape(module)}\..*after_optimizations((?:-[a-z-]+)?\.txt)$")

    def wanted(name: str) -> tuple[int, str] | None:
        match = pattern.match(name.rsplit("/", 1)[-1])
        return (int(match.group(1)), match.group(2)) if match and match.group(2) in suffixes else None

    with contextlib.ExitStack() as stack:
        if hlo.is_file():
            archive = stack.enter_context(tarfile.open(hlo))
            found = [(key, member) for member in archive.getmembers()
                     if member.isfile() and (key := wanted(member.name))]
        else:
            found = [(key, path) for path in hlo.rglob(f"module_*.{module}.*after_optimizations*.txt")
                     if (key := wanted(str(path)))]
        if not found:
            raise SystemExit(f"no {module} after-optimizations dumps under {hlo}")
        numbers = sorted({number for (number, _), _ in found})
        if program is not None and int(program) not in numbers:
            raise SystemExit(f"the capture ran program {program}, and {hlo} holds {module} "
                             f"compiles {numbers} only")
        chosen = numbers[-1] if program is None else int(program)
        directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        files: dict = {"module_number": str(chosen)}
        for (number, suffix), source in found:
            if number != chosen:
                continue
            if isinstance(source, Path):
                files[suffix] = source
                continue
            target = directory / f"module{suffix}"
            with archive.extractfile(source) as reader, target.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
            files[suffix] = target
        yield files


def lines(path: Path | None) -> Iterator[str]:
    """`path`'s lines without their newlines, read as they stream; none for no path."""
    if path is None:
        return
    with path.open(errors="replace") as stream:
        for line in stream:
            yield line.rstrip("\n")


def peak_breakdown(text: str | Iterable[str]) -> tuple[int, list[tuple[str, int]]]:
    """The live-range dump's peak bytes and its "Stack trace breakdown for peak usage" as
    (label, bytes the node itself holds): an op's op_name path, or an argument's tree path
    (`p/params/in_proj_qkv/kernel`)."""
    marker = "Stack trace breakdown for peak usage:"
    peak, leaves, stack, inside = 0, [], [], False
    for line in text.splitlines() if isinstance(text, str) else text:
        if not inside:
            if marker in line:
                inside, peak = True, int(re.sub(r"\D", "", line.partition(marker)[2]) or 0)
            continue
        match = re.match(r"^([\s│├└─]*)(.+?) \(\d+\.\d+%, total: [\d ]+ bytes, current: ([\d ]+) bytes", line)
        if not match:
            if line.strip() and not line.startswith(" "):
                break
            continue
        depth = len(match.group(1))
        label = match.group(2).strip()
        while stack and stack[-1][0] >= depth:
            stack.pop()
        stack.append((depth, label))
        current = int(match.group(3).replace(" ", ""))
        if current:
            labels = [entry[1] for entry in stack[1:]]  # the first is the entry computation
            ops = [label for label in labels if "jit(" in label]
            leaves.append((ops[-1] if ops else "/".join(labels), current))
    return peak, leaves


def role_of(label: str) -> str:
    if "jit(" not in label:
        if re.search(r"opt|adam|\bmu\b|\bnu\b|momentum|moment|ema", label):
            return "argument: optimizer state"
        if re.search(r"^p(/|$)|param|frozen|lora|kernel|bias|weight|scale|embedding", label):
            return "argument: parameters"
        if re.match(r"^[a-z_]+(/|$)", label) and not re.search(r"fusion|copy|\.\d", label):
            return "argument: inputs"
        return "unattributed"
    if "rematted_computation" in label:
        return "rematerialized forward"
    if "transpose(jvp" in label:
        return "backward working set"
    if "jvp(" in label:
        return "saved forward activations"
    return "other ops"


def allocations(text: str | Iterable[str]) -> dict[str, int]:
    kinds: dict[str, int] = defaultdict(int)
    for line in text.splitlines() if isinstance(text, str) else text:
        match = re.match(r"^allocation \d+: size (\d+), (.*)$", line)
        if match is None:
            continue
        kind = next((name for name in ("preallocated-temp", "parameter", "constant", "thread-local",
                                       "maybe-live-out") if name in match.group(2)), "other")
        kinds[kind] += int(match.group(1))
    return dict(kinds)


def attribute_memory(files: dict) -> dict:
    """The peak's bytes by role and part, from `dump_files`'s files, each read as it streams."""
    peak, leaves = peak_breakdown(lines(files.get("-live-range.txt")))
    groups: dict = defaultdict(int)
    for label, size in leaves:
        groups[(role_of(label), part_of(label))] += size
    total = next(filter(None, (re.search(r"Total bytes: (\d+)", line)
                               for line in lines(files.get("-memory-usage-report.txt")))), None)
    held = allocations(lines(files.get("-buffer-assignment.txt")))
    gib = 1 << 30
    return {
        "module_number": files["module_number"],
        "peak_gib": round(peak / gib, 3),
        "total_bytes_gib": round(int(total.group(1)) / gib, 3) if total else None,
        "allocations_gib": {kind: round(size / gib, 3)
                            for kind, size in held.items()},
        "peak_by_role_and_part_gib": [
            {"role": role, "part": part, "gib": round(size / gib, 3),
             "share": round(size / peak, 4) if peak else None}
            for (role, part), size in sorted(groups.items(), key=lambda item: -item[1])],
        "peak_largest_buffers": [{"gib": round(size / gib, 4), "role": role_of(label), "part": part_of(label),
                                  "label": label[:220]}
                                 for label, size in sorted(leaves, key=lambda item: -item[1])[:30]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--hlo", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--entry", type=Path, help="entry.json carrying model_flops_per_step")
    parser.add_argument("--module", default="jit_step")
    parser.add_argument("--device", help="the device kind, where --entry does not name it")
    args = parser.parse_args(argv)
    flops, device = None, args.device
    if args.entry and args.entry.exists():
        entry = json.loads(args.entry.read_text())
        flops, device = entry.get("model_flops_per_step"), device or entry.get("device")
    result = {}
    rows = report_table(capture_of(args.trace) / "reports" / "hlo_stats.json")[1] if args.trace else []
    programs = program_times(rows)
    program = max(programs, key=programs.__getitem__) if programs else None
    with dump_files(args.hlo, args.module, program) if args.hlo else contextlib.nullcontext({}) as files:
        if args.trace:
            ops = hlo_instructions(lines(files[".txt"])) if ".txt" in files else None
            result["time"] = attribute_time(args.trace, args.steps, flops, ops, device)
            result["time"]["hlo_module"] = files.get("module_number")
        if files:
            result["memory"] = attribute_memory(files)
    result["peak_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 3)
    args.out.write_text(json.dumps(result, indent=1))
    time = result.get("time", {})
    if "by_part_ms" in time:
        print(f"time: {time['device_window_ms_per_step']} ms a step on the device = "
              f"{time['device_busy_ms_per_step']} busy + {time['device_other_programs_ms_per_step']} other "
              f"programs + {time['device_idle_ms_per_step']} idle, "
              f"GEMMs at {time['gemm_fraction_of_peak']} of peak, {time['gemm_lost_ms']} ms lost to it, "
              f"MFU of device time: bf16 {time.get('mfu_bf16_of_device_time')}, "
              f"mixed {time.get('mfu_mixed_of_device_time')}")
        for part, directions in time["by_part_ms"].items():
            print(f"  {part:10s} " + "  ".join(f"{d} {ms:.2f}" for d, ms in directions.items()))
        print("  kinds: " + ", ".join(f"{k} {ms:.2f}" for k, ms in time["by_kind_ms"].items()))
    elif time:
        print("time:", time.get("error"))
    memory = result.get("memory")
    if memory:
        print(f"memory: peak {memory['peak_gib']} GiB (module {memory['module_number']}), "
              f"allocations {memory['allocations_gib']}")
        for group in memory["peak_by_role_and_part_gib"][:12]:
            print(f"  {group['gib']:8.3f} GiB  {group['role']:28s} {group['part']}")
    # A part asked for and not attributed fails the run, so a driver's gate sees it.
    return 1 if "error" in time else 0


if __name__ == "__main__":
    raise SystemExit(main())
