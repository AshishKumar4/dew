"""The scenario contract's own functions (tests/scenario_contract.py) on their subjects."""

import itertools

import pytest
from scenario_contract import covering

AXES = [
    {"head": ("joint", "question", "letters"), "context": (False, True), "consistency": (0.0, 0.5),
     "bound": ("examples", "share")},
    {"precision": ("32-true", "bf16-mixed", "fp8"), "rows": ("padded", "packed"), "memory": (None, "full"),
     "devices": (1, 2), "quantization": (None, "qat", "ptq"), "read": ("memory", "indexed", "streamed")},
    {"mixer": ("gdn", "kda", "mamba")},
]


@pytest.mark.parametrize("axes", AXES, ids=["decision", "six axes", "one axis"])
def test_a_covering_set_meets_every_pair_of_values_and_takes_each_value_once_at_least(axes):
    """Every two values of two axes meet in some cell, every value of an axis is taken, each cell names one
    value of every axis, and no two runs of the generator differ."""
    cells = covering(axes)
    assert all(set(cell) == set(axes) and all(cell[name] in axes[name] for name in axes) for cell in cells)
    for one, other in itertools.combinations(axes, 2):
        met = {(cell[one], cell[other]) for cell in cells}
        assert met == set(itertools.product(axes[one], axes[other])), (one, other)
    assert all({cell[name] for cell in cells} == set(axes[name]) for name in axes)
    assert covering(axes) == cells



def test_a_covering_set_over_what_a_flow_admits_meets_every_pair_an_admitted_cell_holds():
    """Where a scenario's flow admits some combinations alone (a block-diffusion LM reads the stream's
    windows, a masked one packed or padded documents), none of its cells is one the flow refuses, and every
    two values of two axes that an admitted combination holds meet in some cell."""
    axes = {"objective": ("masked", "block"), "rows": ("padded", "packed", "windows"), "devices": (1, 2),
            "precision": ("32-true", "bf16-mixed")}

    def admits(cell: dict) -> bool:
        return (cell["objective"] == "block") == (cell["rows"] == "windows")

    cells = covering(axes, admits)
    assert cells and all(admits(cell) for cell in cells)
    admitted = [cell for values in itertools.product(*axes.values())
                for cell in [dict(zip(axes, values, strict=True))] if admits(cell)]
    for one, other in itertools.combinations(axes, 2):
        assert ({(cell[one], cell[other]) for cell in cells}
                == {(cell[one], cell[other]) for cell in admitted}), (one, other)
