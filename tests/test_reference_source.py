"""Selected upstream definitions execute with the globals they declare; a script run from its directory
imports its own siblings."""

import sys

import pytest
from tool_support import load, run_from


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


def test_a_script_run_from_its_directory_imports_its_sibling_over_a_cached_namesake():
    # A session that collected tests/test_kernel_outputs_written.py holds tests/kernel_outputs.py as
    # `kernel_outputs`; the live bridge imports its own sibling of that name (rows 7329 to 7333).
    import kernel_outputs

    with run_from("shared_bridge", "site/live/container") as bridge:
        assert callable(bridge.outputs_of)
    assert sys.modules["kernel_outputs"] is kernel_outputs


def test_a_script_run_from_its_directory_leaves_no_sibling_for_a_namesakes_later_import(monkeypatch):
    # The other order: the bridge runs first, and tests/test_gemm_paired.py imports tests/kernel_outputs.py
    # inside a test after it.
    monkeypatch.delitem(sys.modules, "kernel_outputs", raising=False)
    with run_from("shared_bridge", "site/live/container"):
        pass
    from kernel_outputs import launched_outputs

    assert callable(launched_outputs)
