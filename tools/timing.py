"""The measurement lane's timing harness: a manifest of cases, timed in one exclusive window per GPU kind.

A case is a builder (an import path, `builder(cell, implementation)` returning `(call, draw)`: the callable
and a draw of its arguments by index, so every arm meets the same inputs in the same order), its cells (the
shapes and products it is built for), and its arms. An arm is {name, tree, env, setup, implementation}: `tree`
a commit whose checkout it runs in (None, the window's own), `env` the variables its process starts with,
`setup` a module its process imports first (a patch, a registration), `implementation` what the builder is
given. Arms of one tree and env share a process; each process keeps a compilation cache of its own, so a
revision's compiles are its own.

The window builds every case's every arm first (each compile, and a first call whose output's digest is kept,
so arms are compared bitwise), then times them in interleaved rounds: every round passes every case, its arms
in an order reversed from the round before (ABBA), each arm timed `repeats` calls. Each round also times the
case's reference arm against itself (`<reference>~aa`, its own build in its own place in the order), so a row
carries the window's own noise. Time is the device's (dew.telemetry.benchmark.device_time: CUPTI, each kernel
and their sum, an eviction's own left out where the case evicts), with each call's wall clock beside it; a
case whose question is host-bound latency asks for the wall clock alone (`"clock": "wall"`). Profiles (Nsight
Compute) run after every round, each arm's in a process of its own, never in a timed one.

    python tools/timing.py run MANIFEST.json --out DIR    # DIR/timing.jsonl's rows, DIR/profiles' reports
    python tools/timing.py worker                         # one process's arms, JSON lines on stdin and stdout

A row: case, cell, arm, tree, jax, implementation, round, repeat, order (its place in the window's run order),
device_ms (the sum), kernels ([name, ms], launch order), wall_ms, digest, evicted, cache ("cold": each
process's compilation cache is new to its window), gpu (name, uuid, driver, clocks as the round began)."""
import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
INSTRUMENT = TOOLS.parent / "src/dew/telemetry/benchmark.py"
"""The window's own device timer, which every arm's process loads by its path (`instrument`), never its
tree's: one instrument for every arm, whatever the arm's commit holds."""
ROUNDS = 5
"""A case's rounds unless it names its own: an arm's range and the A/A band are its rounds' ratios, which
three leave coarse enough for one outlier round to set the band."""
NCU_SECTIONS = ("SpeedOfLight", "MemoryWorkloadAnalysis", "SchedulerStats", "WarpStateStats", "LaunchStats",
                "Occupancy")
"""The Nsight Compute sections a profile reads unless its case names its own."""


def digest(output) -> str:
    """The sha256 of an output's leaves' bytes, in order: two arms' outputs agree bitwise where it does."""
    import jax
    import numpy as np

    hashed = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(output):
        hashed.update(np.asarray(leaf).tobytes())
    return hashed.hexdigest()


def built(case: dict, cell: dict, arm: dict) -> tuple:
    """`case`'s builder's (call, draw) for `cell` and `arm`'s implementation. A builder is `file.py:function`,
    a file the window holds (`main` makes its path absolute), loaded by its path so every arm builds alike,
    its imports of Dew from the arm's tree; or `module:function`, imported where the worker runs (the arm's
    tree)."""
    source, _, function = case["builder"].partition(":")
    if source.endswith(".py"):
        spec = importlib.util.spec_from_file_location(f"_timing_builder_{Path(source).stem}", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module = importlib.import_module(source)
    return getattr(module, function)(cell, arm.get("implementation"))


def instrument():
    """dew.telemetry.benchmark as the window holds it, loaded from its file under a name of its own (it
    imports only jax, numpy and the standard library), so an arm's tree's own copy is never read."""
    spec = importlib.util.spec_from_file_location("_timing_instrument",
                                                  os.environ.get("DEW_TIMING_INSTRUMENT", INSTRUMENT))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def worker(clock: str) -> int:
    """One process's arms: `build` (an arm compiled and called once: its digest), `time` (`repeats` calls,
    their device and wall times), `once` (one call, for a profiler wrapping this process), `quit`. Each
    request a JSON line on stdin, each answer one on stdout."""
    import jax

    for module in filter(None, os.environ.get("DEW_TIMING_SETUP", "").split(",")):
        importlib.import_module(module)
    held: dict[str, tuple] = {}
    with contextlib.ExitStack() as stack:
        timer = None
        if clock == "device":
            timer = stack.enter_context(instrument().device_time())
        out = sys.stdout
        for line in sys.stdin:
            request = json.loads(line)
            if request["op"] == "quit":
                break
            if request["op"] == "build":
                call, draw = built(request["case"], request["cell"], request["arm"])
                held[request["key"]] = (call, draw)
                answer = {"digest": digest(jax.block_until_ready(call(*draw(0)))), "jax": jax.__version__,
                          "tree": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                                 text=True).stdout.strip()}
            elif request["op"] == "time":
                call, draw = held[request["key"]]
                answer = {"calls": [timed(timer, call, draw((request["index"] + repeat) % request["draws"]),
                                          request["evict"]) for repeat in range(request["repeats"])]}
            elif request["op"] == "once":
                # A profiler started off (`--profile-from-start off`) sees the second call alone: the first's
                # compile and its autotuning launches are never profiled.
                import ctypes

                call, draw = built(request["case"], request["cell"], request["arm"])
                jax.block_until_ready(call(*draw(0)))
                cuda = ctypes.CDLL("libcuda.so.1")
                cuda.cuProfilerStart()
                jax.block_until_ready(call(*draw(0)))
                cuda.cuProfilerStop()
                answer = {}
            elif request["op"] == "release":
                held.pop(request["key"], None)
                import gc

                gc.collect()
                answer = {}
            else:
                raise ValueError(f"no request {request['op']!r}")
            out.write(json.dumps(answer) + "\n")
            out.flush()
    return 0


def timed(timer, call, args: tuple, evict: bool) -> dict:
    """One call's device time (each kernel, and their sum) where `timer` reads it, and a second call's wall
    clock: the device read runs its call once untimed first (`DeviceTime`), so the two are separate calls."""
    import jax

    kernels = [] if timer is None else [[kernel.name, kernel.milliseconds]
                                        for kernel in timer.kernels(call, *args, evicted=evict)]
    start = time.perf_counter()
    jax.block_until_ready(call(*args))
    wall = (time.perf_counter() - start) * 1e3
    return {"device_ms": sum(ms for _, ms in kernels) if kernels else None, "kernels": kernels,
            "wall_ms": wall}


class Process:
    """A worker (`worker`) in an arm's tree, its env and setup its own, and a compilation cache of its own."""

    def __init__(self, tree: Path, env: dict, setup: list[str], clock: str, cache: Path):
        cache.mkdir(parents=True, exist_ok=True)
        # Several workers share one GPU: none preallocates it unless an arm's env says otherwise.
        variables = {**os.environ, "XLA_PYTHON_CLIENT_PREALLOCATE": "false", **env,
                     "DEW_TIMING_INSTRUMENT": str(INSTRUMENT), "JAX_COMPILATION_CACHE_DIR": str(cache),
                     "PYTHONPATH": os.pathsep.join([str(tree / "src"), str(tree), str(tree / "tools")]),
                     "DEW_TIMING_SETUP": ",".join(setup)}
        self.child = subprocess.Popen([sys.executable, str(TOOLS / "timing.py"), "worker", "--clock", clock],
                                      cwd=tree, env=variables, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      text=True)

    def ask(self, request: dict) -> dict:
        self.child.stdin.write(json.dumps(request) + "\n")
        self.child.stdin.flush()
        line = self.child.stdout.readline()
        if not line:
            raise RuntimeError(f"the worker for {request.get('key', request['op'])} ended with "
                               f"{self.child.wait()}")
        return json.loads(line)

    def close(self) -> None:
        with contextlib.suppress(BrokenPipeError):
            self.child.stdin.write(json.dumps({"op": "quit"}) + "\n")
            self.child.stdin.close()
        self.child.wait()


def fetchable(manifest: dict, repo: Path) -> None:
    """Every arm's tree a commit the window can check out: held here and on a remote branch (pushed, so a
    node's clone fetches it), or fetched from origin now; else refused before the window starts, naming the
    arm."""
    for case in manifest["cases"]:
        for arm in case["arms"]:
            tree = arm.get("tree")
            if tree is None:
                continue

            def git(*arguments: str) -> subprocess.CompletedProcess:
                return subprocess.run(["git", "-C", str(repo), *arguments], capture_output=True, text=True,
                                      check=False)

            held = git("cat-file", "-e", f"{tree}^{{commit}}").returncode == 0
            if held and git("branch", "-r", "--contains", tree).stdout.strip():
                continue
            if not held and git("fetch", "-q", "origin", tree).returncode == 0:
                continue
            why = "held here but on no remote branch: push it" if held else "not fetchable from origin"
            raise SystemExit(f"timing: case {case['name']} arm {arm['name']}: tree {tree} is {why}, so no "
                             "node can check it out")


def checkout(tree: str | None, out: Path, repo: Path) -> Path:
    """The checkout an arm of `tree` runs in: the window's own, or a worktree of that commit under `out`."""
    if tree is None:
        return repo
    where = out / "trees" / tree[:12]
    if not where.exists():
        subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", "--detach", str(where), tree],
                       check=True)
    return where


def arms_of(case: dict) -> list[dict]:
    """A case's arms, and its reference's A/A arm (`<reference>~aa`: the reference built again, in its
    process)."""
    reference = case.get("reference", case["arms"][0]["name"])
    found = next(arm for arm in case["arms"] if arm["name"] == reference)
    return [*case["arms"], {**found, "name": f"{reference}~aa"}]


def gpu() -> dict:
    """The window's GPU as nvidia-smi reads it now, or {} where there is none (a CPU window)."""
    answer = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,clocks.sm,clocks.mem",
                             "--format=csv,noheader"], capture_output=True, text=True, check=False) \
        if Path("/usr/bin/nvidia-smi").exists() else None
    if answer is None or answer.returncode != 0:
        return {}
    name, uuid, driver, sm, memory = [field.strip() for field in answer.stdout.splitlines()[0].split(",")]
    return {"name": name, "uuid": uuid, "driver": driver, "sm_clock": sm, "memory_clock": memory}


def run(manifest: dict, out: Path, repo: Path) -> int:
    """Every case of `manifest` built, timed in interleaved rounds and profiled: its rows in
    out/timing.jsonl."""
    clock = manifest.get("clock", "device")
    out.mkdir(parents=True, exist_ok=True)
    processes: dict[str, Process] = {}

    def process(arm: dict, clock_of: str) -> Process:
        group = json.dumps([arm.get("tree"), arm.get("env", {}), arm.get("setup"), clock_of], sort_keys=True)
        if group not in processes:
            name = hashlib.sha256(group.encode()).hexdigest()[:12]
            processes[group] = Process(checkout(arm.get("tree"), out, repo), arm.get("env", {}),
                                       [arm["setup"]] if arm.get("setup") else [], clock_of,
                                       out / "caches" / name)
        return processes[group]

    order = 0
    try:
        with open(out / "timing.jsonl", "w") as rows:
            # A case at a time: its arms built, timed and released, so a window holds one case's arms at once.
            for case in manifest["cases"]:
                units = []
                for index, cell in enumerate(case["cells"]):
                    for arm in arms_of(case):
                        key = f"{case['name']}/{index}/{arm['name']}"
                        worker_of = process(arm, case.get("clock", clock))
                        answer = worker_of.ask({"op": "build", "key": key, "case": case, "cell": cell,
                                                "arm": arm})
                        units.append({"key": key, "index": index, "cell": cell, "arm": arm,
                                      "process": worker_of, "built": answer})
                for round_ in range(case.get("rounds", ROUNDS)):
                    facts = gpu()
                    for index, _ in enumerate(case["cells"]):
                        mine = [unit for unit in units if unit["index"] == index]
                        for unit in mine if round_ % 2 == 0 else mine[::-1]:
                            answer = unit["process"].ask({
                                "op": "time", "key": unit["key"], "repeats": case.get("repeats", 5),
                                "evict": case.get("evict", False), "draws": case.get("draws", 1),
                                "index": round_ * case.get("repeats", 5)})
                            for repeat, called in enumerate(answer["calls"]):
                                rows.write(json.dumps({
                                    "case": case["name"], "cell": unit["cell"], "arm": unit["arm"]["name"],
                                    "tree": unit["built"]["tree"], "jax": unit["built"]["jax"],
                                    "implementation": unit["arm"].get("implementation"), "round": round_,
                                    "repeat": repeat, "order": order, **called,
                                    "digest": unit["built"]["digest"], "evicted": case.get("evict", False),
                                    "cache": "cold", "gpu": facts}) + "\n")
                            order += 1
                for unit in units:
                    unit["process"].ask({"op": "release", "key": unit["key"]})
    finally:
        for worker_of in processes.values():
            worker_of.close()
    for case in manifest["cases"]:
        if "profile" in case:
            profiled(case, out, repo)
    summarized(manifest, out)
    return 0


def profiled(case: dict, out: Path, repo: Path) -> None:
    """Each of `case`'s cells and arms profiled by Nsight Compute (`profile`: its kernel regex, count,
    sections and source-level sampling), each in a process of its own, its report under out/profiles."""
    profile = case["profile"]
    (out / "profiles").mkdir(exist_ok=True)
    for index, cell in enumerate(case["cells"]):
        for arm in case["arms"]:
            tree = checkout(arm.get("tree"), out, repo)
            sections = [word for section in profile.get("sections", NCU_SECTIONS)
                        for word in ("--section", section)]
            sampling = ["--section", "SourceCounters"] if profile.get("pcsamp") else []
            report = out / "profiles" / f"{case['name']}-{index}-{arm['name']}"
            request = json.dumps({"op": "once", "case": case, "cell": cell, "arm": arm})
            variables = {**os.environ, **arm.get("env", {}),
                         "PYTHONPATH": os.pathsep.join([str(tree / "src"), str(tree), str(tree / "tools")]),
                         "DEW_TIMING_SETUP": arm.get("setup") or ""}
            # Profiling off until the worker's `once` starts it around its second call, so no compile,
            # autotuning or warm-up launch is what ncu captures.
            subprocess.run(["ncu", "--profile-from-start", "off", "-k", f"regex:{profile['kernel']}",
                            "-c", str(profile.get("count", 1)),
                            *sections, *sampling, "-o", str(report), sys.executable, str(TOOLS / "timing.py"),
                            "worker", "--clock", "wall"],
                           input=request + "\n" + json.dumps({"op": "quit"}) + "\n", text=True,
                           cwd=tree, env=variables, check=True)


def summarized(manifest: dict, out: Path) -> None:
    """The window's verdict, out/summary.json and its table out/summary.md, from its rows, so every window
    decides alike. Per case, cell and arm: the median of its measure (device ms, or wall ms where the case's
    clock is the wall's); its paired ratio to the reference each round (the arm's round median over the
    reference's), the ratios' median, least and most; the A/A arm's least and most ratio, the window's noise
    band; `differs` only where the arm's whole range of ratios lies outside the band; and whether its digest
    is the reference's. Beside them, the window's run order (each `order`'s case, cell and arm) and each
    round's GPU."""
    import statistics

    rows = [json.loads(line) for line in (out / "timing.jsonl").read_text().splitlines()]
    cases, lines = [], []
    for case in manifest["cases"]:
        measure = "wall_ms" if case.get("clock", manifest.get("clock", "device")) == "wall" else "device_ms"
        reference = case.get("reference", case["arms"][0]["name"])
        lines += [f"### {case['name']} ({measure}, reference {reference})", "",
                  "| cell | arm | median | ratio median | ratio range | differs | digest |",
                  "|---|---|---|---|---|---|---|"]
        cells = []
        for index, cell in enumerate(case["cells"]):
            mine = [row for row in rows if row["case"] == case["name"] and row["cell"] == cell]

            def by_round(arm: str, mine: list[dict] = mine, measure: str = measure) -> dict[int, float]:
                found: dict[int, list[float]] = {}
                for row in mine:
                    if row["arm"] == arm and row[measure] is not None:
                        found.setdefault(row["round"], []).append(row[measure])
                return {round_: statistics.median(values) for round_, values in found.items()}

            base = by_round(reference)
            digest = next(row["digest"] for row in mine if row["arm"] == reference)
            aa = [value / base[round_] for round_, value in by_round(f"{reference}~aa").items()]
            band = (min(aa), max(aa)) if aa else None
            arms = []
            for arm in [found["name"] for found in arms_of(case)]:
                rounds = by_round(arm)
                ratios = [value / base[round_] for round_, value in rounds.items()]
                values = [row[measure] for row in mine if row["arm"] == arm and row[measure] is not None]
                # The reference and its A/A are the band's own ends, never judged against it.
                differs = bool(band and ratios and arm not in (reference, f"{reference}~aa")
                               and (min(ratios) > band[1] or max(ratios) < band[0]))
                same = next(row["digest"] for row in mine if row["arm"] == arm) == digest
                arms.append({"arm": arm, "median": statistics.median(values) if values else None,
                             "ratio_median": statistics.median(ratios) if ratios else None,
                             "ratio_range": [min(ratios), max(ratios)] if ratios else None,
                             "differs": differs, "digest_is_reference": same})
                ranged = f"{min(ratios):.4f}-{max(ratios):.4f}" if ratios else "-"
                median = f"{arms[-1]['median']:.4f}" if values else "-"
                ratio = f"{arms[-1]['ratio_median']:.4f}" if ratios else "-"
                lines.append(f"| {index}: {json.dumps(cell)} | {arm} | {median} | {ratio} | {ranged} | "
                             f"{'yes' if differs else 'no'} | {'same' if same else 'differs'} |")
            cells.append({"cell": cell, "noise_band": list(band) if band else None, "arms": arms})
        cases.append({"case": case["name"], "measure": measure, "reference": reference, "cells": cells})
        lines.append("")
    order = [{"order": row["order"], "case": row["case"], "cell": row["cell"], "arm": row["arm"]}
             for row in rows if row["repeat"] == 0]
    gpus = {f"{row['case']}/{row['round']}": row["gpu"] for row in rows if row["repeat"] == 0}
    (out / "summary.json").write_text(json.dumps({"cases": cases, "run_order": order, "gpu_by_round": gpus},
                                                 indent=1))
    (out / "summary.md").write_text("\n".join(lines))


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser("run")
    running.add_argument("manifest", type=Path)
    running.add_argument("--out", type=Path, required=True)
    serving = commands.add_parser("worker")
    serving.add_argument("--clock", choices=("device", "wall"), default="device")
    arguments = parser.parse_args(argv)
    if arguments.command == "worker":
        return worker(arguments.clock)
    repo = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                               check=True).stdout.strip())
    manifest = json.loads(arguments.manifest.read_text())
    for case in manifest["cases"]:
        source, _, function = case["builder"].partition(":")
        if source.endswith(".py"):
            case["builder"] = f"{(repo / source).resolve()}:{function}"
    fetchable(manifest, repo)
    return run(manifest, arguments.out.resolve(), repo)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
