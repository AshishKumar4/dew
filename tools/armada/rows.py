"""How tools/armada/ci.py names a verdict's rows and where the test cache keeps each one's record: a row
`<python>:<test file>[#<group>]`, its group a split's `G/N`, a GPU lane's process variant after it
(`<file>#G/N@users`), and its record's place on a device class, in the cache Worker's key alphabet (no colon,
no hash)."""


def row(python: str, name: str, split: str) -> str:
    return f"{python}:{name}" + (f"#{split}" if split else "")


def parsed_row(row_name: str) -> tuple[str, str, str]:
    """`row`'s (python, name, split) back from its name."""
    python, rest = row_name.split(":", 1)
    name, _, split = rest.partition("#")
    return python, name, split


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
