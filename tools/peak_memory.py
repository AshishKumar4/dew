"""A program's measured peak device memory: each device's allocator high-water over the program's run.

JAX resets no high-water (`memory_stats`' peak_bytes_in_use only grows over a process), and a compile's
autotuning allocates through the same allocator, so a program measured in a process of its own that
compiled nothing reads its own peak: its state, its inputs, its temporaries and native workspaces
(cuBLAS, cuDNN, a kernel's scratch), and no compile. `measured` runs the program twice, each in a fresh
process, the two sharing a JAX compilation cache: the first fills it, and the second loads every program
it runs from it and reports each local device's high-water as the program exits. A program the second
needed but did not load from the cache (`dew.telemetry.compiles.Compiles` against the cache's
hits) it compiled, which refuses the measurement: one the cache keeps no entry for compiles every time.
A probe measures one row a process, so the high-water is that row's.

    peaks = measured(["tools/benchmark_step.py", "--cases", ..., "--peak-run"])

A device whose allocator keeps no high-water (JAX's CPU devices, as this JAX builds them) reports None,
which a caller may know before it runs a process (`keeps_high_water`).
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MEASURED = """
import atexit, json, os, runpy, sys
import jax
from dew.telemetry.instrumentation import compilation_cache_counts, count_compilation_cache
from dew.telemetry.compiles import Compiles
stage, out, *argv = sys.argv[1:]
# The script's own directory first, as `python script.py` puts it, for the modules beside it.
sys.path[0] = os.path.dirname(os.path.abspath(argv[0]))
count_compilation_cache()
needed = Compiles(None).__enter__()


def report():
    if stage == "measure":
        loaded = compilation_cache_counts().get("train/cache_hits", 0)
        peaks = [(device.memory_stats() or {}).get("peak_bytes_in_use") for device in jax.local_devices()]
        with open(out, "w") as written:
            json.dump({"compiled": needed.count - int(loaded), "peaks": peaks}, written)


atexit.register(report)
sys.argv = argv
runpy.run_path(argv[0], run_name="__main__")
"""
"""A stage's process: the program run as its own `__main__`, the programs it needed and those it loaded
from the cache counted from its start, and, in the measure stage, how many it compiled and each local
device's high-water written as it exits."""


class CompiledWhileMeasured(RuntimeError):
    """The measured process compiled a program its cache did not hold, whose compile, and autotuning,
    joined the high-water it read: the program compiles otherwise on its second run than its first."""


def keeps_high_water() -> bool:
    """Whether this process's devices may keep a high-water: not where JAX runs on its CPU devices alone
    (`jax_platforms`, read without opening a backend), whose every peak `measured` reports is None. A
    caller that skips its peak processes where none is kept asks here, not by a check of its own:
    benchmark_lm_serving skips a slot count's two, its test running `measured` over a peak run where
    they run. Every other tool runs its peak processes on every platform, the CPU's runs its tests of
    them (ledger 3856)."""
    import jax

    return jax.config.jax_platforms != "cpu"


PEAK_CACHE = "DEW_PEAK_CACHE"
"""The environment variable naming a compile cache the peak runs keep across calls (tools/perf_gate.py
names one an arm, so each of its rounds' first runs loads what its first round compiled); a cache of the
call's own, removed after it, where unset."""


def measured(argv: list[str]) -> list[int | None]:
    """Each local device's high-water over `python <argv>`'s run, in a process that loaded every program
    from the cache a first run of it filled, so compiled nothing; a `CalledProcessError` where either run
    fails, `CompiledWhileMeasured` where the second compiled. The cache is PEAK_CACHE's where it names one."""
    with tempfile.TemporaryDirectory(prefix="dew-peak-") as directory:
        # JAX's settings by their environment names: every compile kept, however small or quick.
        cache = os.environ.get(PEAK_CACHE) or str(Path(directory, "cache"))
        environment = {**os.environ, "JAX_ENABLE_COMPILATION_CACHE": "true",
                       "JAX_COMPILATION_CACHE_DIR": cache,
                       "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
                       "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1"}
        out = Path(directory, "peaks.json")
        for stage in ("warm", "measure"):
            done = subprocess.run([sys.executable, "-c", MEASURED, stage, str(out), *argv], env=environment,
                                  capture_output=True, text=True)
            try:
                done.check_returncode()
            except subprocess.CalledProcessError as error:
                error.add_note(f"its {stage} run's stderr:\n{done.stderr[-4000:]}")
                raise
        report = json.loads(out.read_text())
    if report["compiled"]:
        raise CompiledWhileMeasured(f"{' '.join(argv)}: {report['compiled']} programs compiled in its "
                                    "measured run, which the cache did not give it")
    return report["peaks"]
