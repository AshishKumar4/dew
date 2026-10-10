"""Where a profiled training step's device time and peak memory go, by model part.

    python tools/attribute_step.py --trace PROFILE_DIR --hlo HLO_DIR_OR_TGZ --out OUT.json
                                   [--steps N] [--entry entry.json] [--module jit_step]

`--trace` is a dew.Profiler directory (or a capture inside one); the newest capture holding
`reports/hlo_stats.json` is read. Profile it with XLA_FLAGS=--xla_gpu_enable_command_buffer= :
inside CUDA command buffers XProf attributes every kernel to its module and hlo_stats is empty.
`--steps` is how many steps the capture traced; totals are divided by it.

`--hlo` is an XLA dump (`--xla_dump_to=... --xla_dump_hlo_as_text`), a directory or a .tgz of
one. The newest `<module>` compiled is read: its optimized HLO (each op's opcode, a custom call's
target, the ops a fusion runs), its live-range dump's peak (the values live at the peak and XLA's
breakdown of them by op_name) and its buffer assignment (parameters, outputs, temporaries).

Time, from the XProf reports: device time a step by model part (GDN, attention, LoRA, MLP, head
and loss, embedding, norm, optimizer, other) and direction (forward, backward, neither); by kind,
read from the op itself (`kind_of`): GEMM, attention, Pallas/Triton, layout (transposes and
copies), slicing, gathers and scatters, reductions, elementwise, collectives; both together; GEMMs
by the time they lose against the device's peak (time - flops / peak); the costliest layout
moves and ops with framework name and source line; the device's idle time; with `--entry`, the
step's model FLOPs over its device time against the peak. Without `--hlo`, a custom call or a
fusion is kinded from its name alone, which cannot tell FlashAttention's call from any other.

Memory, from the dump: the peak's bytes by role (arguments: parameters, optimizer state, inputs;
saved forward activations; backward working set; rematerialized forward; unattributed) and model
part, and its largest buffers.

Reads only text and JSON: no JAX, so it runs anywhere.
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import tarfile
from collections import defaultdict
from pathlib import Path

PARTS = (
    # The decision head's own attention and norms are the head's; Qwen3.5 names every mixer
    # `self_attn`, so a delta net's ops are told apart by the file that traced them.
    ("head_loss", r"Head\)|Head/|JointSchemaHead|lm_head|logits|[Ll]oss|cross_entropy|log_softmax|"
                  r"[Dd]ecision"),
    ("optimizer", r"optax|adam|[Mm]uon|apply_updates|clip_by|update_moment|scale_by|opt_state|"
                  r"training/transaction\.py"),
    ("gdn", r"nn/linear\.py|nn/kernels/delta_|nn/kda\.py|GatedDeltaNet|gated_delta|chunk_delta|KimiDelta"),
    ("lora", r"[Ll]o[Rr][Aa]|lora_a|lora_b"),
    ("norm", r"layernorm|_norm/|/norm\d?/|RMSNorm|LayerNorm"),
    ("attention", r"nn/attention\.py|nn/mixers/attention\.py|flash_attn|decode_attention|nn/rope\.py|"
                  r"[Aa]ttention|_attended|dot_product_attention|fmha|splash"),
    ("mlp", r"/mlp/|gate_proj|up_proj|down_proj|SwiGLU|experts|nn/moe\.py"),
    ("embedding", r"[Ee]mbed|table_rows|wte|tok_emb"),
)
TARGETS = (
    ("gemm", r"^__cublas|^__cudnn\$conv"),
    ("attention", r"^__cudnn\$fmha|^dew_flash_mha"),
    ("pallas", r"^__gpu\$xla\.gpu\.triton$|mosaic"),
)
"""A custom call's kind by its target: cuBLAS and cuDNN's convolutions, the fused attentions
(cuDNN's and Dew's FlashAttention-2), and the Triton calls Pallas lowers to."""
COLLECTIVES = {"all-reduce", "all-gather", "reduce-scatter", "collective-permute", "all-to-all",
               "all-reduce-start", "all-gather-start", "collective-permute-start", "all-reduce-done",
               "all-gather-done", "collective-permute-done"}
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


def table(path: Path, index: int = 0) -> list[dict]:
    """The rows of one of an XProf report's DataTables, as dicts by column id."""
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    tables = data if isinstance(data, list) else [data]
    if index >= len(tables) or "cols" not in tables[index]:
        return []
    columns = [column.get("id") for column in tables[index]["cols"]]
    return [{name: (cell.get("v") if isinstance(cell, dict) else cell)
             for name, cell in zip(columns, row["c"], strict=False)}
            for row in tables[index].get("rows", [])]


_INSTRUCTION = re.compile(r"^\s*(?:ROOT\s+)?%(?P<name>[\w.\-]+) = (?:\([^()]*\)|[^\s(]+) "
                          r"(?P<op>[a-z][a-z0-9-]*)\((?P<rest>.*)$")
_COMPUTATION = re.compile(r"^\s*(?:ENTRY\s+)?%(?P<name>[\w.\-]+) \(.*\{\s*$")


def instructions(text: str) -> dict[str, tuple[str, str, frozenset[str]]]:
    """Each instruction of an optimized HLO module by name: its opcode, a custom call's target, and
    the opcodes of what a fusion calls (the fused computation's, nested ones included)."""
    computations: dict[str, set[str]] = defaultdict(set)
    calls: dict[str, list[str]] = defaultdict(list)
    found: dict[str, tuple[str, str, list[str]]] = {}
    current = ""
    for line in text.splitlines():
        computation = _COMPUTATION.match(line)
        if computation:
            current = computation.group("name")
            continue
        match = _INSTRUCTION.match(line)
        if not match:
            continue
        op, rest = match.group("op"), match.group("rest")
        computations[current].add(op)
        called = re.findall(r"calls=%([\w.\-]+)", rest)
        calls[current].extend(called)
        target = re.search(r'custom_call_target="([^"]+)"', rest)
        found[match.group("name")] = (op, target.group(1) if target else "", called)

    def fused(name: str, seen: frozenset = frozenset()) -> set[str]:
        ops = set(computations.get(name, ()))
        for inner in calls.get(name, ()):
            if inner not in seen:
                ops |= fused(inner, seen | {name})
        return ops

    return {name: (op, target, frozenset().union(*(fused(called) for called in called_list)) if called_list
                   else frozenset())
            for name, (op, target, called_list) in found.items()}


def kind_of(row: dict, ops: dict[str, tuple[str, str, frozenset[str]]] | None = None) -> str:
    """An op's kind, read from the op: its opcode, a custom call's target, the ops a fusion runs.
    From the module (`instructions`) where it was given, else from the op's expression and name;
    never from the names of its operands, which the expression also holds."""
    name = str(row.get("hlo_op_name") or "")
    expression = str(row.get("hlo_op_expression") or "")
    known = (ops or {}).get(name)
    if known:
        op, target, fused = known
    else:
        match = _INSTRUCTION.match(expression)
        op = match.group("op") if match else ""
        target = (re.search(r'custom_call_target="([^"]+)"', expression) or re.search(r"()", "")).group(1)
        base = re.sub(r"\.\d+$", "", name)
        fused = frozenset(base.split("_")) if op == "fusion" else frozenset()
    if op in COLLECTIVES:
        return "collective"
    if op == "custom-call":
        if not target and name.startswith("pallas_call"):
            return "pallas"
        return next((kind for kind, pattern in TARGETS if re.search(pattern, target)), "custom-call")
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


def heaviest(groups: dict) -> list:
    """`groups` of times by key, the largest total first."""
    return sorted(groups.items(), key=lambda item: -sum(item[1].values()))


def attribute_time(trace: Path, steps: int, model_flops: float | None,
                   ops: dict[str, tuple[str, str, frozenset[str]]] | None = None) -> dict:
    capture = capture_of(trace)
    reports = capture / "reports"
    rows = table(reports / "hlo_stats.json")
    if not rows:
        return {"capture": str(capture), "error": "hlo_stats.json has no ops: profile with "
                "XLA_FLAGS=--xla_gpu_enable_command_buffer= so XProf can attribute kernels"}
    roofline = json.loads((reports / "roofline_model.json").read_text())
    roofline = roofline[0] if isinstance(roofline, list) else roofline
    peak = float(roofline.get("p", {}).get("peak_flop_rate", 0) or 0)
    programs = defaultdict(float)
    for row in rows:
        programs[str(row.get("program_id"))] += float(row["total_self_time"])
    step = max(programs, key=programs.__getitem__)
    by_part: dict = defaultdict(lambda: defaultdict(float))
    by_kind: dict = defaultdict(float)
    by_part_kind: dict = defaultdict(lambda: defaultdict(float))
    gemms, layouts, everything = [], [], []
    for row in (row for row in rows if str(row.get("program_id")) == step):
        name = str(row.get("tf_op_name") or "")
        us = float(row["total_self_time"]) / steps
        source = source_of(row)
        part, direction, kind = part_of(f"{name} {source}"), direction_of(name), kind_of(row, ops)
        by_part[part][direction] += us
        by_kind[kind] += us
        by_part_kind[part][kind] += us
        gflops = float(row.get("normalized_flop_rate") or row.get("model_flop_rate") or 0)
        record = {"us": round(us, 1), "part": part, "direction": direction, "kind": kind,
                  "hlo": row.get("hlo_op_name"), "framework": name[:200], "count": row.get("occurrences"),
                  "gflops": round(gflops, 1), "bound_by": row.get("bound_by"), "source": source}
        everything.append(record)
        if kind == "gemm" and peak:
            record["lost_us"] = round(us * max(0.0, 1 - gflops / peak), 1)
            gemms.append(record)
        if kind == "layout":
            layouts.append(record)
    device_us = sum(by_kind.values())
    idle_us = sum(float(row["total_self_time"]) for row in table(reports / "framework_op_stats.json")
                  if row.get("host_or_device") == "Device" and row.get("type") == "IDLE") / steps
    gemm_us = by_kind.get("gemm", 0.0)
    gemm_work = sum(record["us"] * record["gflops"] for record in gemms)
    result = {
        "capture": str(capture), "steps": steps, "peak_gflops": peak, "step_program": step,
        "programs_ms": {program: round(us / 1e3 / steps, 3)
                        for program, us in sorted(programs.items(), key=lambda item: -item[1])[:6]},
        "device_ms_per_step": round(device_us / 1e3, 3),
        "device_idle_ms_per_step": round(idle_us / 1e3, 3),
        "by_part_ms": {part: {direction: round(us / 1e3, 3) for direction, us in directions.items()}
                       for part, directions in heaviest(by_part)},
        "by_kind_ms": ms(by_kind),
        "by_part_kind_ms": {part: ms(kinds) for part, kinds in heaviest(by_part_kind)},
        "gemm_fraction_of_peak": round(gemm_work / gemm_us / peak, 3) if gemm_us and peak else None,
        "gemm_lost_ms": round(sum(record["lost_us"] for record in gemms) / 1e3, 3),
        "gemms_by_lost_time": sorted(gemms, key=lambda record: -record["lost_us"])[:25],
        "layout_moves": sorted(layouts, key=lambda record: -record["us"])[:25],
        "top_ops": sorted(everything, key=lambda record: -record["us"])[:40],
        "xprof_steps": table(reports / "input_pipeline_analyzer.json", 1)[:8],
    }
    if model_flops and peak:
        result["model_flops_per_step"] = model_flops
        result["mfu_of_device_time"] = round(model_flops / (device_us * 1e-6) / (peak * 1e9), 4)
    return result


# ---- memory, from XLA's dump ------------------------------------------------------------------


def dump_files(hlo: Path, module: str) -> dict[str, str]:
    """The newest `module`'s after-optimizations HLO, live-range, buffer-assignment and
    memory-usage-report texts, by suffix (the HLO's is `.txt`)."""
    suffixes = (".txt", "-live-range.txt", "-buffer-assignment.txt", "-memory-usage-report.txt")
    texts: dict[str, dict[str, str]] = defaultdict(dict)

    def take(name: str, read) -> None:
        base = name.rsplit("/", 1)[-1]
        match = re.match(rf"module_(\d+)\.{re.escape(module)}\..*after_optimizations((?:-[a-z-]+)?\.txt)$",
                         base)
        if match and match.group(2) in suffixes:
            texts[match.group(1)][match.group(2)] = read()

    if hlo.is_file():
        with tarfile.open(hlo) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    take(member.name, lambda member=member: io.TextIOWrapper(
                        archive.extractfile(member), errors="replace").read())
    else:
        for path in hlo.rglob(f"module_*.{module}.*after_optimizations*.txt"):
            take(str(path), lambda path=path: path.read_text(errors="replace"))
    if not texts:
        raise SystemExit(f"no {module} after-optimizations dumps under {hlo}")
    newest = max(texts, key=int)
    return {"module_number": newest, **texts[newest]}


def peak_breakdown(text: str) -> tuple[int, list[tuple[str, int]]]:
    """The live-range dump's peak bytes and its "Stack trace breakdown for peak usage" as
    (label, bytes the node itself holds): an op's op_name path, or an argument's tree path
    (`p/params/in_proj_qkv/kernel`)."""
    _, _, tree = text.partition("Stack trace breakdown for peak usage:")
    if not tree:
        return 0, []
    first, _, rest = tree.partition("\n")
    peak = int(re.sub(r"\D", "", first))
    leaves, stack = [], []
    for line in rest.splitlines():
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


def allocations(text: str) -> dict[str, int]:
    kinds: dict[str, int] = defaultdict(int)
    for size, attributes in re.findall(r"^allocation \d+: size (\d+), (.*)$", text, re.M):
        kind = next((name for name in ("preallocated-temp", "parameter", "constant", "thread-local",
                                       "maybe-live-out") if name in attributes), "other")
        kinds[kind] += int(size)
    return dict(kinds)


def attribute_memory(files: dict[str, str]) -> dict:
    peak, leaves = peak_breakdown(files.get("-live-range.txt", ""))
    groups: dict = defaultdict(int)
    for label, size in leaves:
        groups[(role_of(label), part_of(label))] += size
    total = re.search(r"Total bytes: (\d+)", files.get("-memory-usage-report.txt", ""))
    gib = 1 << 30
    return {
        "module_number": files["module_number"],
        "peak_gib": round(peak / gib, 3),
        "total_bytes_gib": round(int(total.group(1)) / gib, 3) if total else None,
        "allocations_gib": {kind: round(size / gib, 3)
                            for kind, size in allocations(files.get("-buffer-assignment.txt", "")).items()},
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
    args = parser.parse_args(argv)
    flops = None
    if args.entry and args.entry.exists():
        flops = json.loads(args.entry.read_text()).get("model_flops_per_step")
    result = {}
    files = dump_files(args.hlo, args.module) if args.hlo else {}
    if args.trace:
        result["time"] = attribute_time(args.trace, args.steps, flops,
                                        instructions(files[".txt"]) if ".txt" in files else None)
    if files:
        result["memory"] = attribute_memory(files)
    args.out.write_text(json.dumps(result, indent=1))
    time = result.get("time", {})
    if "by_part_ms" in time:
        print(f"time: {time['device_ms_per_step']} ms device a step, "
              f"{time['device_idle_ms_per_step']} ms idle, "
              f"GEMMs at {time['gemm_fraction_of_peak']} of peak, {time['gemm_lost_ms']} ms lost to it, "
              f"MFU of device time {time.get('mfu_of_device_time')}")
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
