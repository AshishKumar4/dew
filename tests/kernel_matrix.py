"""The template kernel families' check-1 matrices (Main, 2026-11-02, after the attention dK escape on main).

Each family declares its axes (`AXES`) and one matrix cell function (`CELLS`): a route-level training flow,
the forward and its backward in one program as a user calls them, held to check 1 with the float64 terms,
its launch form pinned. Its cells cover every pair of axis values the family's route serves
(docs/design/testing.md's covering scenarios): what it serves is read from the route's own analyses
(`supported`, the cell module's), never a second list. Main's GPU tier runs `FLOWS` on an H100 and an A100
after a publish whose diff reaches the kernels, and tests/test_kernel_matrix.py holds each family's cells to
its supported pairs (`covered`). Nothing here needs a GPU.
"""

import functools
import importlib
import itertools

import gpu_flow

AXES: dict[str, dict[str, tuple[str, ...]]] = {
    "attention": {
        "generation": ("sm90", "sm80"),
        "width": ("64", "128", "256"),
        "heads": ("mha", "gqa-8-2", "gqa-8-4"),
        "mask": ("causal", "full", "documents"),
        "hook": ("plain", "key-bias", "tile-bias", "softcap"),
        "launch": ("one", "two", "shared-keys"),
        "dtype": ("bfloat16", "float16", "float32"),
        "form": ("fast", "accurate"),
    },
    "recurrence": {
        "generation": ("sm90", "sm80"),
        "form": ("gated delta net", "kimi delta attention", "write gate", "rated scan"),
        "kernel": ("mosaic_gpu", "port", "xla", "derived"),
        "dtype": ("bfloat16", "float32"),
        "documents": ("one", "packed"),
    },
}
"""Each family's axes and their values. Attention's: the GPU; the head width; 8 query heads over 8 key heads,
2, or 4; a causal, a full or a causal call over three documents and padding; no hook, two key or two tile
parameters of one shape added to the logits (two in-place gradients of one shape, row 3180's), or a softcap;
the backward in one launch, two, or one whose consumers share a key block; the operands' dtype; the fast or
the accurate form. The recurrence's: the GPU; the form (tests/recurrence_matrix.py's `FORMS`); the kernel
the call names (the port fla's or mamba_ssm's by the form); the operands' dtype; one sequence or packed
documents."""

SEEDED: dict[str, tuple[tuple[int, tuple[str, ...]], ...]] = {
    "attention": (
        (3522, ("sm90", "256", "gqa-8-4", "causal", "plain", "two", "float32", "fast")),
    ),
}
"""Each family's cases a ledger row names, by the row, its axis values in `AXES`' order: its cells take them
whatever the covering takes besides (`covering`), so the row's own case proves it closed."""

CELLS: dict[str, tuple[str, str]] = {
    "attention": ("attention_matrix", "a_routes_training_step_is_on_par"),
    "recurrence": ("recurrence_matrix", "a_recurrences_training_step_is_on_par"),
}
"""Each family's matrix cell function, by its module and name (`gpu_flow.cell`)."""

FLOWS: tuple[str, ...] = (
    "tests/test_gpu_attention.py::test_the_attention_matrix_holds_on_this_gpu",
    "tests/test_gpu_recurrence.py::test_the_recurrence_matrix_holds_on_this_gpu",
)
"""The flows that run the matrices' cells, by pytest node id."""


def check(family: str):
    """`family`'s matrix cell function (`CELLS`)."""
    module, name = CELLS[family]
    return getattr(importlib.import_module(module), name)


def combinations(family: str) -> list[dict[str, str]]:
    """Every combination of `family`'s axis values, in `AXES`' order."""
    axes = AXES[family]
    return [dict(zip(axes, values, strict=True)) for values in itertools.product(*axes.values())]


@functools.cache
def supported(family: str, combination: tuple[tuple[str, str], ...]) -> bool:
    """Whether `family`'s route serves `combination` (its axis-value pairs), read from the route's own
    analyses by the family's cell module (`supported` there)."""
    return importlib.import_module(CELLS[family][0]).supported(dict(combination))


def supported_pairs(family: str) -> set[tuple[tuple[str, str], tuple[str, str]]]:
    """Every pair of axis values (each an axis and its value) that some combination the route serves holds."""
    pairs = set()
    for combination in combinations(family):
        if supported(family, tuple(combination.items())):
            pairs |= set(itertools.combinations(combination.items(), 2))
    return pairs


def covered(family: str) -> list[tuple[str, dict[str, str]]]:
    """Each registered cell of `family` by its label beside the combination it covers (`gpu_flow.covered`)."""
    return gpu_flow.covered(check(family))


def covering(family: str) -> list[tuple[str, ...]]:
    """Combinations the route serves in which every supported pair occurs (`supported_pairs`), each as its
    values in `AXES`' order: the family's seeded cases (`SEEDED`) first, then the combination holding the most
    pairs not yet covered, a tie to the first in `combinations`' order. A family's cell module takes these as
    its cases, regenerated where its axes or its route's support change (tests/test_kernel_matrix.py fails
    until they are)."""
    left = supported_pairs(family)
    served = [combination for combination in combinations(family)
              if supported(family, tuple(combination.items()))]
    chosen = []
    for _, case in SEEDED.get(family, ()):
        combination = dict(zip(AXES[family], case, strict=True))
        chosen.append(case)
        left -= set(itertools.combinations(combination.items(), 2))
    while left:
        best = max(served, key=lambda combination: len(
            left & set(itertools.combinations(combination.items(), 2))))
        chosen.append(tuple(best.values()))
        left -= set(itertools.combinations(best.items(), 2))
    return chosen
