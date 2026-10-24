"""Selected upstream definitions execute with the globals they declare."""

import pytest
from tool_support import load


def test_selected_functions_keep_annotated_global_values_in_source_order():
    source = load("reference_source")
    upstream = """
offset = 3
rate: float = offset / 2
unused = object()
def project(values):
    return [rate * value + offset for value in values]
"""
    scope = source.ran(source.selected(upstream, ("offset", "rate", "project")), {})
    values = [-4, 0, 2, 7]
    assert scope["project"](values) == [1.5 * value + 3 for value in values]
    scope["rate"] = -2
    assert scope["project"](values) == [-2 * value + 3 for value in values]


def test_a_bare_annotation_is_not_a_reference_value():
    source = load("reference_source")
    with pytest.raises(AssertionError):
        source.selected("declared: float\n", ("declared",))
