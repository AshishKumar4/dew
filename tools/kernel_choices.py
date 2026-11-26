"""The kernel choices `dew.kernels.Kernels` holds, written from what Dew's dispatches serve.

`Kernels` states each op a block can name (`generation.CHOICES`) as a field whose values are exactly the
kernels its op's selectors hold (`selection.Op`, every one the production modules under `dew.nn` define) or
a site decides by name (`generation.SITE_KERNELS`): a type a checker and a command line's parser read, so
written out in the module, never computed when it imports. This tool writes that block from the selectors
(`--write`), between its markers in `src/dew/kernels.py`, and otherwise says whether the module's block is
the one it would write, exiting 1 where it is not (row 12776: a hand-kept copy of the table drifted).
A choice new to the block is keyword-only, so a positional call of the fields before it keeps naming
the ones it named.

    python tools/kernel_choices.py            # check
    python tools/kernel_choices.py --write    # rewrite the block
"""

import argparse
import dataclasses
import importlib
import pkgutil
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "src" / "dew" / "kernels.py"
BEGIN = "    # The kernel choices, written by tools/kernel_choices.py from the selectors: edit none by hand."
END = "    # End of the kernel choices."
KEYWORD_ONLY = "    # Keyword-only: a positional call of the fields before it keeps naming the ones it named."
WIDTH = 110


def ops() -> dict[str, list]:
    """Every kernel op Dew's dispatches serve (`selection.Op`), by its choice, read from the production
    modules under `dew.nn` as they define them, a module's own or held in a module's mapping or sequence (a
    layer's selectors by choice, `joined.JOINS`): none is listed here."""
    import dew.nn
    from dew.nn.kernels import selection

    def held(value) -> Iterator[selection.Op]:
        if isinstance(value, selection.Op):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from held(item)
        elif isinstance(value, list | tuple):
            for item in value:
                yield from held(item)

    found: dict[str, list] = {}
    for module in pkgutil.walk_packages(dew.nn.__path__, "dew.nn."):
        for value in vars(importlib.import_module(module.name)).values():
            for op in held(value):
                if all(op is not seen for seen in found.get(op.choice, [])):
                    found.setdefault(op.choice, []).append(op)
    return found


def served() -> dict[str, frozenset[str]]:
    """Each kernel choice (`generation.CHOICES`) and the kernels it chooses among: its op's selectors' and
    the names a site decides by (`generation.SITE_KERNELS`)."""
    from dew.nn.kernels import generation

    named: dict[str, set[str]] = {str(choice): set() for choice in generation.CHOICES}
    for choice, found in ops().items():
        named.setdefault(str(choice), set()).update(kernel.name for op in found for kernel in op.kernels)
    for choice, kernels in generation.SITE_KERNELS.items():
        named.setdefault(str(choice), set()).update(kernels)
    return {choice: frozenset(kernels) for choice, kernels in named.items()}


def _field(name: str, kernels: frozenset[str], keyword_only: bool) -> list[str]:
    """`name`'s field, its kernels a `Literal` wrapped at `WIDTH` under its first value."""
    head = f"    {name}: Literal["
    tail = "] | None = " + ("dataclasses.field(default=None, kw_only=True)" if keyword_only else "None")
    values = [repr(kernel) for kernel in sorted(kernels)]
    lines, line = [], head
    for index, value in enumerate(values):
        piece = value + (", " if index < len(values) - 1 else "")
        last = index == len(values) - 1
        if len(line + piece.rstrip() + (tail if last else "")) > WIDTH and line.strip() != head.strip():
            lines.append(line.rstrip())
            line = " " * len(head)
        line += piece
    return [*lines, line + tail]


def rendered(choices: Mapping[str, frozenset[str]], keyword_only: frozenset[str]) -> list[str]:
    """The block's lines for `choices`, by name, those in `keyword_only` keyword-only, the first of them
    under the comment saying why."""
    lines = [BEGIN]
    said = False
    for name in sorted(choices):
        if name in keyword_only and not said:
            lines.append(KEYWORD_ONLY)
            said = True
        lines.extend(_field(name, choices[name], name in keyword_only))
    return [*lines, END]


def keyword_only_now() -> frozenset[str]:
    """The fields `Kernels` holds keyword-only as it stands; every choice it does not hold yet joins them."""
    from dew import kernels
    from dew.nn.kernels import generation

    fields = {field.name: field.kw_only for field in dataclasses.fields(kernels.Kernels)}
    return frozenset(name for name in map(str, generation.CHOICES) if fields.get(name, True))


def block(text: str) -> tuple[int, int]:
    """The lines `text`'s block spans, its markers' indices, refused where either is missing."""
    lines = text.splitlines()
    if BEGIN not in lines or END not in lines:
        raise ValueError(f"{MODULE} holds no block between {BEGIN!r} and {END!r}")
    return lines.index(BEGIN), lines.index(END)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--write", action="store_true", help="rewrite the block in place")
    args = parser.parse_args(argv)
    text = MODULE.read_text()
    first, last = block(text)
    lines = text.splitlines()
    wanted = rendered(served(), keyword_only_now())
    if lines[first:last + 1] == wanted:
        return 0
    if not args.write:
        print(f"{MODULE}: the kernel choices drifted from the selectors; run python tools/kernel_choices.py "
              "--write", file=sys.stderr)
        return 1
    MODULE.write_text("\n".join([*lines[:first], *wanted, *lines[last + 1:]]) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
