"""Where a parameter splits, declared on the module that owns it.

A module names the logical axes of the parameters its submodules create,
keyed by the trailing module path, outermost dimension first:

    @logical_axes({("q_proj",): ("embed", "heads"), ("o_proj",): ("attention", "embed")})
    class CausalSelfAttention(nn.Module): ...

A parameter takes the trailing names its rank can hold, so a kernel takes all
of them and its bias the output ones. The declarations of every decorated
module merge into one table the `Layout` reads when it places a train state
and Muon reads when it picks a parameter's matrix axes, which is what keeps
the models plain Flax modules whose init returns arrays. The optimizer's
moments and the EMA copy have paths ending in their parameter's, so one
declaration reaches them as well.

`heuristic` lists what a class leaves to the shape heuristic on purpose (a
convolution, a state matrix, a projection with no side worth naming): their
parameters are placed on their largest divisible axis. Each entry is a run
of `fnmatch` patterns matched against consecutive names of the parameter
path, so `("time_embed",)` covers every parameter under that module and
`("up_dense_*",)` a numbered family. A declared or heuristic name that no
parameter carries any more is what the coverage test in
tests/test_architectures.py reports, so a renamed submodule fails there
rather than silently stopping to match.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Mapping
from typing import TypeAlias

import jax

LogicalAxes: TypeAlias = tuple[str | None, ...]
Suffix: TypeAlias = tuple[str, ...]

DECLARED: dict[Suffix, LogicalAxes] = {}
"""Every decorated module's declarations, merged."""

HEURISTIC: set[Suffix] = set()
"""Runs of name patterns whose parameters take the shape heuristic on purpose."""


def logical_axes(declared: Mapping[Suffix, LogicalAxes], *,
                 heuristic: Iterable[Suffix] = ()):
    """Declare the parameter axes of the modules `cls` creates."""
    declared = {tuple(suffix): tuple(axes) for suffix, axes in declared.items()}
    heuristic = tuple(tuple(suffix) for suffix in heuristic)
    for suffix, axes in declared.items():
        names = [name for name in axes if name is not None]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            # The rules hand a mesh axis to a logical name once per array, so
            # a second dimension of the same name could never be placed.
            raise ValueError(
                f"{'/'.join(suffix)} is declared {axes}, which names "
                f"{', '.join(repeated)} twice; a kernel whose two sides share a "
                f"width names one of them or leaves the other None")

    def decorate(cls):
        for suffix, axes in declared.items():
            held = DECLARED.get(suffix)
            if held is not None and held != axes:
                raise ValueError(
                    f"{'/'.join(suffix)} is declared {axes} by {cls.__name__} and "
                    f"{held} elsewhere; one module path has one set of axes")
            DECLARED[suffix] = axes
        HEURISTIC.update(heuristic)
        cls.__logical_axes__ = declared
        cls.__heuristic_axes__ = heuristic
        return cls

    return decorate


def parameter_path(path) -> Suffix:
    """The parameter's own path: the trailing run of dict keys under a leaf.

    An optimizer state nests a copy of the parameter tree inside its own
    structure, so what identifies a parameter is where its path ends.
    """
    names = []
    for entry in reversed(path):
        if not isinstance(entry, jax.tree_util.DictKey) or not isinstance(entry.key, str):
            break
        names.append(entry.key)
    return tuple(reversed(names))


def _matching(table, module: Suffix):
    for length in range(len(module), 0, -1):
        if module[-length:] in table:
            return module[-length:]
    return None


def declared_axes(path, ndim: int) -> LogicalAxes | None:
    """The declared axes of the parameter at `path`, or None for an unnamed one."""
    module = parameter_path(path)[:-1]
    suffix = _matching(DECLARED, module)
    if suffix is None:
        return None
    axes = DECLARED[suffix]
    if ndim > len(axes):
        raise ValueError(
            f"{'/'.join(suffix)} is declared {axes}, which cannot name the "
            f"{ndim} dimensions of {'/'.join(parameter_path(path))}")
    return axes[len(axes) - ndim:]


def is_heuristic(path) -> bool:
    """Whether the parameter at `path` is one a module left to the shape heuristic."""
    names = parameter_path(path)
    for pattern in HEURISTIC:
        for start in range(len(names) - len(pattern) + 1):
            if all(fnmatch.fnmatchcase(name, glob)
                   for name, glob in zip(names[start:], pattern)):
                return True
    return False
