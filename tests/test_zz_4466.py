"""Scratch probe (not for landing): row 4466's derived/xla distance from float64 (derived4466.py)."""
import runpy
import sys

import pytest


@pytest.mark.parametrize("form", ["gated delta net", "write gate"])
def test_probe(form, capsys):
    sys.argv = ["derived4466.py", form, "drawn", "one"]
    runpy.run_path("tests/zz_derived4466.py", run_name="__main__")
    raise AssertionError(capsys.readouterr().out)
