"""Real lint failures at the library boundary, with no per-file suppressions."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def lint(source: str, *, library: bool = True):
    filename = "src/dew/quality_probe.py" if library else "examples/quality_probe.py"
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--output-format", "json", "--stdin-filename", filename, "-"],
        input=source, capture_output=True, text=True, cwd=ROOT, check=False, timeout=20)
    assert result.returncode in (0, 1), result.stderr
    return result.returncode, {finding["code"] for finding in json.loads(result.stdout)}


@pytest.mark.parametrize("source,rule", [
    ('print("library diagnostic")\n', "T201"),
    ('caption = "' + "one more word " * 10 + '"\n', "E501"),
    ('value = 1\nimport math\nassert math.pi > value\n', "E402"),
    ('class Options:\n    pass\n\n\ndef read(options=Options()):\n    return options\n', "B008"),
    ('class Options:\n    rows = []\n', "RUF012"),
    ('from dataclasses import dataclass\n\n\nclass Value:\n    pass\n\n\n'
     '@dataclass\nclass Row:\n    value: Value = Value()\n', "RUF009"),
    ('def read(x):\n    if x:\n        return 1\n    else:\n        return 2\n', "RET505"),
    ('def read(x):\n    if x:\n        y = 1\n    else:\n        y = 2\n    return y\n', "SIM108"),
    ('functions = []\nfor row in range(3):\n    functions.append(lambda: row)\n', "B023"),
    ('caption = "Cyrillic \u0430 is not Latin a"\n', "RUF001"),
    ('from typing import Mapping\n\n\ndef count(rows: Mapping[str, int]) -> int:\n'
     '    return len(rows)\n', "UP035"),
])
def test_strict_library_rules_reject_real_violations(source, rule):
    status, findings = lint(source)
    assert status == 1
    assert rule in findings


def test_plain_print_is_the_scripts_interface_not_the_librarys():
    assert lint('print("result")\n', library=False) == (0, set())
    assert "T201" in lint('print("result")\n')[1]


def test_imported_generics_are_replaced_by_local_defaulted_parameters():
    source = """from typing import Generic

from typing_extensions import TypeVar

Payload = TypeVar("Payload", default=int)


class Box(Generic[Payload]):
    payload: Payload
"""
    assert lint(source) == (0, set())
    status, findings = lint(source.replace(', default=int', ''))
    assert status == 1 and "UP046" in findings


@pytest.mark.parametrize("branches,statements,rule", [
    (25, 0, "C901"), (26, 0, "PLR0912"), (0, 76, "PLR0915"),
])
def test_complexity_limits_reject_functions_beyond_the_gate(branches, statements, rule):
    source = "def read(x):\n"
    source += "".join(f"    if x == {branch}:\n        x += 1\n" for branch in range(branches))
    source += "    x += 1\n" * statements
    source += "    return x\n"
    status, findings = lint(source)
    assert status == 1 and rule in findings


def test_explicit_loop_binding_keeps_each_closures_own_value():
    source = "functions = [lambda row=row: row for row in range(3)]\n"
    assert lint(source) == (0, set())
    namespace = {}
    exec(source, namespace)
    assert [call() for call in namespace["functions"]] == [0, 1, 2]
