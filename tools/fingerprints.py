"""Behavioral fingerprints of src/dew's pure numeric helpers: a signal of code that computes alike whatever
its text, which tools/duplicates.py, reading text and structure, cannot see.

Each module-level function whose body reads `jnp`, `lax`, `jax` or `np`, names no I/O, logging or
clock, takes one to four positional arrays (unannotated, or annotated as an array) and is decorated by
nothing but JAX's `jit` or custom derivatives, is called on canonical draws: every shape of `SHAPES`,
float32 at each scale of `SCALES` and bfloat16 at the first, the same draw in every argument position.
Where its outputs are floats it is differentiated too, along a drawn tangent (`jax.jvp`). Two
functions match when, on every draw both run on, covering at least two shapes, their outputs land on
the same points of the draw's grid (`_grid`), and their derivatives do wherever both have one.

A match ranks a candidate for a reader; it proves no equivalence and authorizes no dedupe by itself:
equal draws miss what they do not reach (other ranks, dtypes, regimes, arguments not drawn alike), and
two forms of one formula that each match a different reference's rounding are separate on purpose. The
report says what each match covers: forward alone (no float derivative) or forward and derivative, and
the shapes, dtypes and scales both ran on. It leaves out, and counts, a function that returns an input
unchanged or the same outputs at every scale, and a pair where one function names the other (a call,
a dispatch table's entry), which is delegation and not a copy.

Calls run with the Hub offline (HF_HUB_OFFLINE), since a helper's purity is read from its text alone;
the precheck bounds the whole run's time.

    python tools/fingerprints.py [--base=REF] [--all]

With `--base` (origin/main by default) it reports the groups holding a function the checkout adds or
changes against REF; `--all` reports every group. It always exits 0: it is a signal, which a lane's
precheck prints, and never a gate.
"""

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import ast
import collections
import importlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src"

NUMERIC = frozenset({"jnp", "lax", "jax", "np"})
"""Names a numeric helper's body reads."""
IMPURE = frozenset({
    "open", "Path", "requests", "subprocess", "threading", "os", "shutil", "print", "logging", "_log",
    "hf_hub_download", "snapshot_download", "socket", "time", "pickle", "json", "orbax", "tensorstore",
    "input", "sys"})
"""Names that mark a body as more than arithmetic: files, processes, threads, the network, logs, clocks."""
ARRAYS = frozenset({"", "jax.Array", "jnp.ndarray", "np.ndarray", "Array", "ArrayLike",
                    "jax.typing.ArrayLike"})
"""A drawn parameter's annotations: none, or an array's."""
DECORATORS = frozenset({"jit", "custom_jvp", "custom_vjp"})
"""The decorators a candidate may carry: JAX's, which keep it a function of its arrays."""
SHAPES = ((2, 3, 8), (4, 16), (3, 2, 5, 6))
SCALES = (1.0, 1e-4, 30.0)
"""The draws' scales: unit normals, values near zero, and values large enough to leave a function's
linear regime (a saturating activation, an exponential)."""
STEP = {"float32": 2e-3, "bfloat16": 3e-2}
"""A draw's grid step, relative to its outputs' largest magnitude: coarse enough that two orders of one
sum land on the same points, fine enough that another epsilon's floor or another constant does not."""
MOST_ARGUMENTS = 4


@dataclass
class Candidate:
    module: str
    name: str
    arguments: int
    names: frozenset[str]
    """Every name its body reads, a callee's or a table entry's: what delegation is read from."""
    source: str
    draws: dict = field(default_factory=dict)
    """Each draw's key, (shape, dtype, scale), to its outputs' grid (`_grid`), its derivative's (None
    without one), its inputs' grids and its outputs as host arrays."""
    refusals: dict = field(default_factory=dict)
    """Each shape it did not run on to the exception its first draw raised."""
    underived: str | None = None
    """The exception JAX raised taking its derivative, where it could not."""

    @property
    def label(self) -> str:
        return f"{self.module}:{self.name}"


def _decorator(node: ast.expr) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Call):
        target = target.func
    return target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")


def candidates(root: Path = SOURCE) -> list[Candidate]:
    """Every pure numeric helper under `root`, read from its text (the module docstring's rule)."""
    found = []
    for path in sorted(root.rglob("*.py")):
        module = ".".join(path.relative_to(root).with_suffix("").parts).removesuffix(".__init__")
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if any(_decorator(decorator) not in DECORATORS for decorator in node.decorator_list):
                continue
            names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
            if not names & NUMERIC or names & IMPURE:
                continue
            arguments = node.args
            positional = [*arguments.posonlyargs, *arguments.args]
            required = positional[:len(positional) - len(arguments.defaults)]
            if (arguments.vararg or not required or len(required) > MOST_ARGUMENTS
                    or any(default is None for default in arguments.kw_defaults)
                    or any((ast.unparse(argument.annotation) if argument.annotation else "") not in ARRAYS
                           for argument in required)):
                continue
            read = names | {child.attr for child in ast.walk(node) if isinstance(child, ast.Attribute)}
            found.append(Candidate(module, node.name, len(required), frozenset(read), ast.unparse(node)))
    return found


def _draws():
    """Each draw's key and its arguments and tangents, at most MOST_ARGUMENTS of each."""
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    for shape in SHAPES:
        values = rng.standard_normal((2 * MOST_ARGUMENTS, *shape))
        for dtype, scales in (("float32", SCALES), ("bfloat16", SCALES[:1])):
            for scale in scales:
                arrays = [jnp.asarray(value * scale, dtype) for value in values]
                yield (shape, dtype, scale), arrays[:MOST_ARGUMENTS], arrays[MOST_ARGUMENTS:]


def _host(tree) -> list[np.ndarray] | None:
    """`tree`'s leaves as float64 host arrays, or None where one is not a finite number."""
    import jax

    leaves = []
    for leaf in jax.tree.leaves(tree):
        if not hasattr(leaf, "dtype") or not (np.issubdtype(leaf.dtype, np.number) or leaf.dtype == bool):
            return None
        array = np.asarray(leaf, np.float64)
        if not np.all(np.isfinite(array)):
            return None
        leaves.append(array)
    return leaves or None


def _grid(leaves: list[np.ndarray], dtype: str) -> tuple:
    """`leaves` on a grid: their largest magnitude's power of two, and each leaf's shape and values in
    steps of STEP[dtype] of that power, as integers. Two computations alike to the step land on the same
    points, but where a value sits at a step's edge or the largest magnitude at a power of two."""
    largest = max(np.max(np.abs(leaf), initial=0.0) for leaf in leaves)
    power = int(np.ceil(np.log2(largest))) if largest > 0 else None
    step = STEP[dtype] * 2.0 ** (power or 0)
    return power, tuple((leaf.shape, np.rint(leaf / step).astype(np.int64).tobytes()) for leaf in leaves)


def fingerprint(candidate: Candidate, draws) -> None:
    """`candidate`'s grids on each draw it runs on, into `candidate.draws`, and why it did not run on a
    shape, into `candidate.refusals`: a shape whose first draw raises is not tried again."""
    import jax
    import jax.numpy as jnp

    function = getattr(importlib.import_module(candidate.module), candidate.name)
    for key, arguments, tangents in draws:
        if key[0] in candidate.refusals:
            continue
        given, along = arguments[:candidate.arguments], tangents[:candidate.arguments]
        try:
            returned = function(*given)
            outputs, derivative = _host(returned), None
            if outputs is not None and all(jnp.issubdtype(leaf.dtype, jnp.floating)
                                           for leaf in jax.tree.leaves(returned)):
                try:
                    derivative = _host(jax.jvp(function, tuple(given), tuple(along))[1])
                except (TypeError, ValueError, NotImplementedError) as error:
                    # An integer path or a custom rule without a forward derivative: forward alone.
                    candidate.underived = type(error).__name__
        except Exception as error:
            candidate.refusals[key[0]] = type(error).__name__
            continue
        if outputs is None:
            candidate.refusals[key[0]] = "no finite numeric output"
            continue
        inputs = [_grid([np.asarray(value, np.float64)], key[1]) for value in given]
        candidate.draws[key] = (_grid(outputs, key[1]), derivative and _grid(derivative, key[1]), inputs,
                                outputs)


def _trivial(candidate: Candidate) -> bool:
    """Whether `candidate` returns an input as it is (or cast) on every draw, or the same outputs at
    every scale of a shape: no computation of its own to compare."""
    passes = all(any((grid[0], (leaf,)) in inputs for leaf in grid[1])
                 for grid, _, inputs, _ in candidate.draws.values())
    by_shape: dict = {}
    for (shape, dtype, _), (_, _, _, outputs) in candidate.draws.items():
        if dtype == "float32":
            by_shape.setdefault(shape, []).append(outputs)
    constant = any(len(runs) > 1 and all(len(run) == len(runs[0]) and all(
        np.array_equal(a, b) for a, b in zip(run, runs[0], strict=True)) for run in runs[1:])
        for runs in by_shape.values())
    return passes or constant


def match(first: Candidate, second: Candidate) -> str | None:
    """What `first` and `second` agree on, as the report states it, or None where they part or share
    fewer than two shapes."""
    shared = sorted(set(first.draws) & set(second.draws), key=str)
    if len({key[0] for key in shared}) < 2:
        return None
    derived = True
    for key in shared:
        (outputs, derivative, _, _), (theirs, their_derivative, _, _) = first.draws[key], second.draws[key]
        if outputs != theirs:
            return None
        if derivative is None or their_derivative is None:
            derived = False
        elif derivative != their_derivative:
            return None
    shapes = ", ".join(str(shape) for shape in sorted({key[0] for key in shared}, key=str))
    dtypes = ", ".join(sorted({key[1] for key in shared}))
    scales = ", ".join(f"{scale:g}" for scale in sorted({key[2] for key in shared}))
    kind = "forward and derivative" if derived else "forward only"
    return f"{kind}; shapes {shapes}; {dtypes}; scales {scales}"


def changed(found: list[Candidate], base: str) -> set[str]:
    """The labels of the candidates the checkout adds or changes against `base`'s tree."""
    changed_labels = set()
    by_path: dict = {}
    for candidate in found:
        by_path.setdefault(candidate.module, []).append(candidate)
    for module, members in by_path.items():
        relative = Path(*module.split("."))
        path = next((p for p in (relative.with_suffix(".py"), relative / "__init__.py")
                     if (SOURCE / p).exists()), relative.with_suffix(".py"))
        shown = subprocess.run(["git", "-C", str(ROOT), "show", f"{base}:src/{path.as_posix()}"],
                               capture_output=True, text=True)
        before = {}
        if shown.returncode == 0:
            before = {node.name: ast.unparse(node) for node in ast.parse(shown.stdout).body
                      if isinstance(node, ast.FunctionDef)}
        changed_labels |= {member.label for member in members if before.get(member.name) != member.source}
    return changed_labels


def groups(found: list[Candidate]) -> tuple[list[tuple[list[str], list[str]]], int]:
    """The matching functions in groups, each with its pairs' coverage, and the pairs left out as
    delegation (one names the other). A group is the functions linked by matching pairs, transitively."""
    linked: dict = collections.defaultdict(set)
    covered: dict = {}
    delegating = 0
    for index, first in enumerate(found):
        for second in found[index + 1:]:
            coverage = match(first, second)
            if coverage is None:
                continue
            if first.name in second.names or second.name in first.names:
                delegating += 1
                continue
            linked[first.label].add(second.label)
            linked[second.label].add(first.label)
            covered[(first.label, second.label)] = coverage
    result, seen = [], set()
    for label in linked:
        if label in seen:
            continue
        group, frontier = set(), [label]
        while frontier:
            current = frontier.pop()
            if current not in group:
                group.add(current)
                frontier.extend(linked[current] - group)
        seen |= group
        pairs = [f"{a} ~ {b}: {coverage}" for (a, b), coverage in covered.items() if a in group]
        result.append((sorted(group), pairs))
    return sorted(result), delegating


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--all", action="store_true")
    arguments = parser.parse_args(argv)
    sys.path.insert(0, str(SOURCE))
    # Dew's numerical policy, before JAX opens a backend.
    importlib.import_module("dew")
    found = candidates()
    draws = list(_draws())
    drawn, unimported = [], {}
    for candidate in found:
        try:
            fingerprint(candidate, draws)
        except Exception as error:
            unimported[candidate.label] = type(error).__name__
            continue
        if candidate.draws and not _trivial(candidate):
            drawn.append(candidate)
    matched, delegating = groups(drawn)
    fresh = None if arguments.all else changed(drawn, arguments.base)
    shown = [(labels, pairs) for labels, pairs in matched if fresh is None or set(labels) & fresh]
    scope = "every group" if fresh is None else f"groups holding a function changed against {arguments.base}"
    refused = collections.Counter(next(iter(candidate.refusals.values())) for candidate in found
                                  if candidate.refusals and not candidate.draws)
    reasons = ", ".join(f"{count} {name}" for name, count in refused.most_common(4))
    print(f"fingerprints: {len(found)} candidates ({len(unimported)} not imported, {sum(refused.values())} "
          f"running on no draw: {reasons}), {len(drawn)} drawn with a computation of their own, "
          f"{len(matched)} groups, {delegating} delegating pairs left out; {scope}: {len(shown)}")
    for labels, pairs in shown:
        print(f"- {', '.join(labels)}")
        for pair in pairs:
            print(f"    {pair}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
