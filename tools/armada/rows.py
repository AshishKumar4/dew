"""How tools/armada/ci.py names a verdict's rows and where the test cache keeps each one's record: a row
`<python>:<test file>[#<group>]`, its group a split's `G/N`, a GPU lane's process variant after it
(`<file>#G/N@users`), and its record's place on a device class, in the cache Worker's key alphabet (no colon,
no hash)."""

from collections.abc import Mapping


def row(python: str, name: str, split: str) -> str:
    return f"{python}:{name}" + (f"#{split}" if split else "")


def parsed_row(row_name: str) -> tuple[str, str, str]:
    """`row`'s (python, name, split) back from its name."""
    python, rest = row_name.split(":", 1)
    name, _, split = rest.partition("#")
    return python, name, split


GROUPED: list = [None, {}]
"""The rows `grouped_rows` last grouped, and their groups: a plan asks of one run's timings for every file."""


def grouped_rows(measured: Mapping[str, float]) -> dict[tuple[str, str], dict[str, float]]:
    """Armada's row timings `measured` by their Python and file, each row's seconds under its group; a map
    task's seconds are kept under its task's name, no row's. Grouped once for the rows it was last asked
    of."""
    if GROUPED[0] is not measured:
        found: dict[tuple[str, str], dict[str, float]] = {}
        for row_name, seconds in measured.items():
            if ":" in row_name:
                python, name, group = parsed_row(row_name)
                found.setdefault((python, name), {})[group] = seconds
        GROUPED[:] = [measured, found]
    return GROUPED[1]


VARIED = "@"
"""What sets a process variant's name apart in a GPU lane's test name and in its rows' group (`varied`,
`lane_units`): `<file>#G/N@users`, a whole file's `<file>#@users`."""


def varied(group: str, variant: str) -> str:
    """A row's group (`row`) for the process `variant` of it: the group's own, the lane's process."""
    return f"{group}{VARIED}{variant}" if variant else group


def cut_of(group: str) -> str:
    """The cut a row's group names ("G/N"; none, a whole file's), whichever process ran it (`varied`)."""
    return group.partition(VARIED)[0]


def row_place(row_name: str, held: str) -> str:
    """Row `row_name`'s place on class `held` under a record prefix: `<class>/<test file>/<group>`, a group
    `whole` or a split's `<index>-of-<count>` (`key_parts`), a process variant's (`varied`) with
    `-as-<variant>` after it, in the Worker's key alphabet."""
    _, name, split = parsed_row(row_name)
    variant = split.partition(VARIED)[2]
    return (f"{held}/{name}/" + ("-of-".join(cut.split("/")) if (cut := cut_of(split)) else "whole")
            + (f"-as-{variant}" if variant else ""))


def key_parts(key: str) -> tuple[str, str, str]:
    """A record's key under its Python's prefix (`record_key`'s `<class>/<test file>/<group>`) as its class,
    its test file and its group (`whole`, or a split's `<index>-of-<count>`)."""
    held, rest = key.split("/", 1)
    name, group = rest.rsplit("/", 1)
    return held, name, group


def place_variant(group: str) -> str:
    """The process variant a record's group (`key_parts`) is of (`row_place`: `whole-as-<variant>`,
    `<index>-of-<count>-as-<variant>`); none, the lane's own process's."""
    return group.partition("-as-")[2]


def release_plan(verdict: dict, python: str) -> list[dict]:
    """The release checks of a whole suite's `verdict` (`armada verdict <sha> --json`), a task each
    requirement: the files whose rows name checks under it (`release_env`, as pytest resolved each test's
    mark before any selection: executed.py), which the run's own tasks set aside, as the suite's environment
    lacks it (ci.py's `selected`), each run on `python` (the oldest tested). ci.py's `release` runs them."""
    checks: dict[str, set[str]] = {}
    for found in verdict["rows"]:
        for requirement in found.get("release_env", {}):
            checks.setdefault(requirement, set()).add(parsed_row(found["name"])[1])
    return [{"release": requirement, "python": python, "tests": " ".join(sorted(names))}
            for requirement, names in sorted(checks.items())]
