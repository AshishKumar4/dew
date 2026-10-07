"""A config mapping's keys survive a run record: written as JSON's string keys
and read back, in a fresh process, as the key type the mapping declares."""

import dataclasses
import json
import math
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import pytest

from dew.registry import from_record, to_record


@dataclasses.dataclass(frozen=True)
class Keyed:
    """Every key type a record reads back: numbers, literals and tree paths of them."""

    weights: dict[int, float] = dataclasses.field(default_factory=dict)
    scales: dict[float, str] = dataclasses.field(default_factory=dict)
    paths: Mapping[tuple[str, ...], int] = dataclasses.field(default_factory=dict)
    places: Mapping[tuple[str, int], float] = dataclasses.field(default_factory=dict)
    sides: Mapping[Literal["left", "right"], int] = dataclasses.field(default_factory=dict)
    either: Mapping[int | str, int] = dataclasses.field(default_factory=dict)


KEYED = Keyed(weights={0: 0.5, -3: 2.0, 10**20: 1.0},
              scales={0.1: "a", -0.0: "b", 1e-300: "c", math.inf: "d"},
              paths={("params", "layers_0", "q"): 1, ("",): 2, ("a", "", "b"): 3},
              places={("layer", 7): 0.25}, sides={"left": 1, "right": 2}, either={1: 1, "x": 2})

READ_BACK = """
import json, sys
from dew.registry import from_record
from test_record_keys import Keyed
held = from_record(Keyed, json.loads(sys.stdin.read()), dtypes=False)
print(repr({name: [(type(key).__name__, key) for key in getattr(held, name)] for name in
            ("weights", "scales", "paths", "places", "sides", "either")}))
print(held == Keyed(**{f: getattr(held, f) for f in Keyed.__dataclass_fields__}))
"""


def test_a_record_reads_its_keys_back_as_their_declared_types_in_a_fresh_process():
    written = json.dumps(to_record(KEYED, Keyed))
    root = Path(__file__).resolve().parents[1]
    path = os.pathsep.join([str(root / "src"), str(root / "tests")])
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": path}
    done = subprocess.run([sys.executable, "-c", READ_BACK], input=written, capture_output=True, text=True,
                          env=env, timeout=300, check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    expected = {name: [(type(key).__name__, key) for key in getattr(KEYED, name)] for name in
                ("weights", "scales", "paths", "places", "sides", "either")}
    assert done.stdout.splitlines()[0] == repr(expected)
    assert from_record(Keyed, json.loads(written), dtypes=False) == KEYED


@pytest.mark.parametrize(("value", "message"), [
    (Keyed(weights={True: 1.0}), "would read back as"),
    (Keyed(scales={math.nan: "x"}), "would read back as"),
    (Keyed(paths={("a/b", "c"): 1}), "would read back as"),
    (Keyed(either={"1": 1}), "would read back as"),
])
def test_a_key_its_declared_type_would_not_read_back_is_refused_at_write(value, message):
    with pytest.raises(ValueError, match=message):
        to_record(value, Keyed)


def test_a_number_keyed_mapping_with_no_declared_key_type_is_refused_at_write():
    with pytest.raises(ValueError, match="declare the key type"):
        to_record({1: 2.0}, dict)
