"""Every parameter of every model Dew builds is placed as the snapshot recorded at its main
(tests/placement_snapshot.py; tools/placement_snapshot.py writes it): its path, shape, declared logical axes,
partition spec and memory kind, and a trainer's optimizer state and EMA beside its parameters. A change to
how axes are declared moves no placement but those its head writes into the snapshot again."""

from pathlib import Path

import pytest
from placement_snapshot import cases, snapshot

SNAPSHOT = Path(__file__).with_name("fixtures") / "placement" / "snapshot.tsv"
RECORDED: dict[str, list[str]] = {}
for line in SNAPSHOT.read_text().splitlines()[1:]:
    RECORDED.setdefault(line.split("\t")[0].removesuffix(":state"), []).append(line)


def test_the_snapshot_records_every_case():
    """The snapshot holds each case the catalogue builds, and no case it no longer builds."""
    assert sorted(RECORDED) == sorted(cases())


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("name", sorted(cases()))
def test_each_parameter_is_placed_as_the_snapshot_records(name):
    """Each of the case's leaves, its optimizer state and EMA where a trainer's are recorded, has the
    shape, axes, spec and memory kind its line in the snapshot records."""
    assert snapshot(name, cases()[name]) == RECORDED[name]
