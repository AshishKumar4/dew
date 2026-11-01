"""Every program Dew compiles passes its one compile point (ledger 3322): `dew.nn.kernels.options.program`,
whose executables `dew.nn.kernels.templates.report.compiled` admits against the room NCCL's communicators
need (`dew.communicators.admit`) and reports. A `jax.jit` of src/dew stands apart from it only where
`COMPILED_APART` names it with why: the compile point's own jit, a program compiled through
`report.compiled` another way, a nested program inside one that passes it, a trace that compiles nothing,
a timing probe measuring its own dispatch, or a step's bookkeeping on replicated scalars. A new `jax.jit`
fails here until it is the compile point's or named apart."""

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "src" / "dew"

NESTED = "a kernel's own jit, a nested program inside the outer one the compile point admits"
TRACED = "traces the function's shapes, compiling nothing"
TIMED = "a timing probe, which measures its candidates' own dispatch"
BOOKKEEPING = ("a step's bookkeeping on replicated scalars, on the step's hot path: no collective, no math "
               "the algebra pass reads")

COMPILED_APART = {
    "src/dew/nn/kernels/options.py::OuterProgram._jitted": "the compile point's own jit",
    "src/dew/training/trainer.py::Trainer._jitted":
        "the resident step, which the planner compiles through report.compiled (planning.Candidate)",
    "src/dew/nn/kernels/ragged_dot.py::gmm": NESTED,
    "src/dew/nn/kernels/ragged_dot.py::tgmm": NESTED,
    "src/dew/nn/kernels/recurrence_algebra.py::_cumulated": NESTED,
    "src/dew/nn/kernels/splash.py::_splash": NESTED,
    "src/dew/nn/kernels/templates/recurrence/mainloop.py::prefix": NESTED,
    "src/dew/nn/precision.py::_traced_once": NESTED,
    "src/dew/nn/kernels/tuning.py::lowered_once.decorate.jitted": TIMED,
    "src/dew/nn/kernels/tuning.py::_measured": TIMED,
    "src/dew/nn/kernels/tuning.py::_measured.compiled": TIMED,
    "src/dew/quantization.py::_fp8_faster": TIMED,
    "src/dew/nn/flops.py::traced_flops": TRACED,
    "src/dew/training/trainer.py::Trainer._model_shapes": TRACED,
    "src/dew/training/interval.py::bookkeep": BOOKKEEPING,
    "src/dew/training/interval.py::_added_tokens": BOOKKEEPING,
    "src/dew/training/interval.py::held": BOOKKEEPING,
    "src/dew/training/planning.py::whole_step": BOOKKEEPING,
}
"""Each `jax.jit` of src/dew that stands apart from the compile point, `path::qualified scope`, and why."""


def _is_jit(node) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == "jit" and isinstance(node.value, ast.Name)
            and node.value.id == "jax")


def jits_of(relative: str, text: str) -> list[str]:
    """`relative`'s uses of `jax.jit` in `text`, each `relative::qualified scope` (`<module>` at its top
    level): a call, a decorator (scoped by the function it decorates), and one passed on (to
    functools.partial)."""
    found = []

    def walk(node, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                walk(child, [*scope, child.name])
                continue
            if _is_jit(child):
                found.append(".".join(scope) or "<module>")
            walk(child, scope)

    walk(ast.parse(text), [])
    return [f"{relative}::{name}" for name in dict.fromkeys(found)]


def jits() -> list[str]:
    """Every use of `jax.jit` in src/dew (`jits_of`)."""
    root = SOURCE.parents[1]
    return [jit for path in sorted(SOURCE.rglob("*.py"))
            for jit in jits_of(str(path.relative_to(root)), path.read_text())]


def test_every_program_dew_compiles_passes_its_compile_point():
    """Each `jax.jit` of src/dew is named apart (`COMPILED_APART`), and each named one exists, so the list
    never outlives its code."""
    found = set(jits())
    assert found - set(COMPILED_APART) == set(), "a jax.jit outside the compile point, named apart by none"
    assert set(COMPILED_APART) - found == set(), "a jax.jit named apart no longer in src/dew"


def test_the_scan_finds_each_form_of_a_jit():
    """`jits_of` on each form a jit takes: a decorator, one within functools.partial, a call, a call in a
    nested function, a method's, and one at a module's top level; a function of no jit is not found."""
    text = (
        "import functools\nimport jax\n"
        "@jax.jit\ndef a(x): return x\n"
        "@functools.partial(jax.jit, static_argnums=1)\ndef b(x, n): return x\n"
        "def c(x): return jax.jit(a)(x)\n"
        "def outer(x):\n    def inner(x): return jax.jit(a)(x)\n    return inner(x)\n"
        "class K:\n    def method(self, x): return jax.jit(a)(x)\n"
        "top = jax.jit(a)\n"
        "def plain(x): return x\n")
    assert jits_of("m.py", text) == ["m.py::a", "m.py::b", "m.py::c", "m.py::outer.inner", "m.py::K.method",
                                     "m.py::<module>"]
