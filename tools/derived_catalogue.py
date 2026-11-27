"""Write the template derivations a built Dew package ships (`dew.algebra.stored.shipped`,
docs/design/algebra.md §6.5): each step the perf gate declares (`perf_gate.STEP_CASES`) lowered, never
compiled, for a GPU of each CUDA generation (`TrainerAdvanced(lower_for=...)` over an abstract state, as
tools/kernel_picks.py lowers), each pair in a process of its own, its templates deriving their forms into one
store as they trace (`dew.algebra.stored`); that store's entries become `--out`'s, put in its place whole and
refused where it holds anything but entries. A pair a generation cannot lower (`Unlowerable`: a dtype or a
kernel the generation lacks) is skipped and said. It fails where a pair fails otherwise, or where the entries
pass `--limit-mib`.

    python tools/derived_catalogue.py                                  # every step, every generation
    python tools/derived_catalogue.py --generations sm90 --jobs 4      # one generation's
    python tools/derived_catalogue.py --steps "step lm-dense 4x1024"   # one step's

Run from the repository's root, as the package is built, before its wheel is made: the entries are the
package data under src/dew/algebra/derived, which the tree never holds (.gitignore). A key states the
algebra's code and egglog's and JAX's builds (`stored.key`), so the tool runs on the release's own.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from dew import _files
from dew.nn.kernels import generation

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
OUT = ROOT / "src" / "dew" / "algebra" / "derived"
GENERATIONS = tuple(name for name, target in generation.TARGETS.items() if not generation.tpu(target))
"""The CUDA generations a package's entries are derived for by default: every one Dew's table states."""
LIMIT_MIB = 16.0
"""The most the entries may weigh, in MiB: a package carries them to every install."""


def lowered(step: str, generation: str) -> dict:
    """`step` lowered for a GPU of `generation` in this process, its derivations written to the store Dew's
    cache holds (the caller's): the store's counts of them, or why the generation cannot lower it."""
    sys.path[:0] = [str(TOOLS), str(ROOT / "tests")]
    import jax
    import perf_gate
    from benchmark_cases import Case
    from benchmark_models import build_trainer, global_batch

    from dew.nn.kernels.selection import Unlowerable
    from dew.nn.kernels.templates import report
    from dew.training._lowering import Lowered
    from dew.training.ahead import step_batch

    case = Case(**perf_gate.STEP_CASES[step])
    trainer = build_trainer(case)
    trainer.lower_for = generation
    state = jax.eval_shape(trainer.initial_state, trainer.objective.initializer, trainer.key)
    with report.recording() as recorded:
        try:
            trainer.compile(state, step_batch(trainer, global_batch(case)))
        except Lowered as lowered:
            # Lowering the step is the whole compile under `lower_for`, which it ends by raising.
            programs = [name for name, _ in lowered.programs]
        except Unlowerable as refused:
            return {"refused": str(refused)}
        else:
            raise AssertionError("a trainer under lower_for raises its lowered step")
    return {"programs": programs, "stored": dict(recorded.stored)}


def _pair(step: str, generation: str, cache: str, empty: str) -> dict:
    """`lowered` in a process of its own, over the shared store `cache` and no read-only one (`empty`), so
    each derivation is derived afresh where no pair wrote it before: its result, its wall, or its failure."""
    env = {**os.environ, "XDG_CACHE_HOME": cache, "DEW_DERIVATIONS": empty, "JAX_PLATFORMS": "cpu"}
    began = time.perf_counter()
    done = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--lower", step, generation],
                          capture_output=True, text=True, env=env, cwd=ROOT)
    result = {"step": step, "generation": generation, "seconds": round(time.perf_counter() - began, 1)}
    lines = [line for line in done.stdout.splitlines() if line.startswith("{")]
    if done.returncode != 0 or not lines:
        return {**result, "failed": done.stderr[-4000:]}
    return {**result, **json.loads(lines[-1])}


DESCRIPTION = "Write the template derivations a built Dew package ships."


def _foreign(written: _files.Entries, held: _files.Entries, directory: Path) -> set[Path]:
    """What `--out` holds that is no store's entry (`dew.algebra.stored.entry`: a file at its top, named by a
    key, its text its own), which it never replaces."""
    from dew.algebra import stored

    return {entry for entry, kind in held.items()
            if not (kind == "file" and entry.parent == Path(".") and stored.entry(directory / entry))}


def main() -> None:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("--steps", nargs="+", help="perf_gate.STEP_CASES' names (default: every one)")
    parser.add_argument("--generations", nargs="+", default=list(GENERATIONS))
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--jobs", type=int, default=os.cpu_count() or 1, help="pairs lowered at once")
    parser.add_argument("--limit-mib", type=float, default=LIMIT_MIB)
    parser.add_argument("--lower", nargs=2, metavar=("STEP", "GENERATION"), help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.lower is not None:
        print(json.dumps(lowered(*arguments.lower)), flush=True)
        return
    sys.path[:0] = [str(TOOLS)]
    import perf_gate

    steps = arguments.steps or [name for name in perf_gate.STEP_CASES if name.startswith("step ")]
    unknown = sorted(set(steps) - set(perf_gate.STEP_CASES))
    if unknown:
        raise SystemExit(f"no perf-gate step {unknown}; it declares {sorted(perf_gate.STEP_CASES)}")
    began = time.perf_counter()
    with tempfile.TemporaryDirectory() as cache, tempfile.TemporaryDirectory() as empty:
        pairs = [(step, generation) for generation in arguments.generations for step in steps]
        with concurrent.futures.ThreadPoolExecutor(max(1, arguments.jobs)) as pool:
            running = [pool.submit(_pair, step, generation, cache, empty) for step, generation in pairs]
            results = [future.result() for future in running]
        for result in results:
            print(json.dumps(result), flush=True)
        failed = [result for result in results if "failed" in result]
        if failed:
            raise SystemExit(f"{len(failed)} of {len(results)} pairs failed to lower: "
                             f"{[(result['step'], result['generation']) for result in failed]}")
        derived = Path(cache) / "dew" / "algebra" / "derived"
        entries = sorted(derived.glob("*.json")) if derived.is_dir() else []
        weight = sum(entry.stat().st_size for entry in entries)
        if weight > arguments.limit_mib * 2**20:
            raise SystemExit(f"the entries weigh {weight / 2**20:.2f} MiB, past --limit-mib "
                             f"{arguments.limit_mib}")
        # Written whole beside `--out` and put in its place in one call, refused where `--out` holds anything
        # but entries (`_foreign`).
        with _files.swapped(arguments.out, _foreign) as staged:
            for entry in entries:
                shutil.copyfile(entry, staged / entry.name)
    print(json.dumps({"entries": len(entries), "mib": round(weight / 2**20, 3), "pairs": len(results),
                      "refused": sum("refused" in result for result in results),
                      "seconds": round(time.perf_counter() - began, 1), "out": str(arguments.out)}))


if __name__ == "__main__":
    main()
