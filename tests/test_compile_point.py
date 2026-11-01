"""Dew's one compile point (`dew.nn.kernels.options.OuterProgram`, whose every compile passes
`dew.nn.kernels.templates.report.compiled`: its communicators admitted, the algebra pass's seam) holds the
programs of the packages that own their compiles: RL's rollouts and simulators, and serving's steps,
state moves and admissions (ledger 3314)."""

import ast
from pathlib import Path

import dew

PACKAGES = ("rl", "objectives/rl", "inference")
"""The packages whose every program compiles through `OuterProgram`."""


def bare_jits(root: Path, packages: tuple[str, ...]) -> list[str]:
    """Each `jax.jit` named in `packages` under `root`, as `path:line`."""
    return [f"{path.relative_to(root)}:{node.lineno}"
            for package in packages for path in sorted((root / package).rglob("*.py"))
            for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Attribute) and node.attr == "jit"
            and isinstance(node.value, ast.Name) and node.value.id == "jax"]


def test_every_rl_and_serving_program_compiles_through_dews_compile_point():
    """No program of RL's or serving's packages compiles by a bare `jax.jit`: a simulator's policy step,
    a host race's programs, a server's state moves, its state's opening and its admissions' keys are
    `OuterProgram`s, as the rollouts' (3287) and the decode steps are."""
    assert bare_jits(Path(dew.__file__).parent, PACKAGES) == []


def test_the_scan_finds_a_bare_jit_in_each_form(tmp_path):
    """`bare_jits` finds `jax.jit` as a decorator, a call and within functools.partial, and nothing in
    a program that compiles through `OuterProgram`."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text(
        "import functools\nimport jax\n@jax.jit\ndef f(x): return x\ng = jax.jit(f)\n"
        "h = functools.partial(jax.jit, static_argnums=0)\n")
    (tmp_path / "pkg" / "b.py").write_text("from dew.nn.kernels.options import OuterProgram\n"
                                           "@OuterProgram\ndef f(x): return x\n")
    assert bare_jits(tmp_path, ("pkg",)) == ["pkg/a.py:3", "pkg/a.py:5", "pkg/a.py:6"]
