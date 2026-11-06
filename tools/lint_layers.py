"""Dew's layering contracts (docs/design/architecture.md; row 1557), read from the source: each contract's
modules import, directly, none of the packages it forbids, whether the import is absolute, relative,
function-local or under TYPE_CHECKING. A crossing names its contract, its module and the package it
reaches. tools/check_gates runs it on every head:

    python tools/lint_layers.py [SOURCE]

SOURCE is the directory that holds the `dew` package (`src`, the default).
"""

import ast
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lint_scopes as scopes

FEATURES = ("dew.nn", "dew.data", "dew.inputs", "dew.objectives", "dew.training", "dew.inference",
            "dew.interop", "dew.rl", "dew.decision", "dew._cli", "dew.config", "dew.checkpoints",
            "dew.diffusion", "dew.sampling", "dew.eval", "dew.lora", "dew.quantization", "dew.telemetry",
            "dew.io", "dew._cleanup", "dew.figures")
"""Dew's feature packages, which the layers below them import none of."""


class Contract(NamedTuple):
    """The modules a contract holds (each package and its modules), those among them it leaves out, and the
    packages none of the rest imports."""

    modules: tuple[str, ...]
    excepted: tuple[str, ...]
    forbidden: tuple[str, ...]


LAYERS: Mapping[str, Contract] = {
    "conditioning inputs import no objective or trainer": Contract(
        ("dew.inputs",), (), ("dew.objectives", "dew.training", "dew.inference", "dew.rl", "dew.decision",
                              "dew._cli", "dew.config", "dew.checkpoints", "dew.sampling", "dew.eval",
                              "dew.lora", "dew.quantization", "dew.io", "dew._cleanup", "dew.figures")),
    "the data layer imports no feature package": Contract(
        ("dew.data",), (), tuple(name for name in FEATURES if name != "dew.data")),
    "telemetry imports nothing above the network layer": Contract(
        ("dew.telemetry",), (), tuple(name for name in FEATURES if name not in ("dew.nn", "dew.telemetry"))),
    "training imports no front end": Contract(("dew.training",), (), ("dew._cli", "dew.decision")),
    "checkpoints import no front end or configuration": Contract(
        ("dew.checkpoints",), (), ("dew._cli", "dew.decision", "dew.config")),
    # A run's source is loaded as its record states it, so inference reads the run record's own types
    # (`SourceOptions`, dew.config): the contract allows dew.config (row 4268, Serving & RL).
    "inference imports no trainer or front end": Contract(
        ("dew.inference",), (), ("dew.training", "dew._cli", "dew.decision", "dew.io")),
    "reinforcement learning imports no trainer": Contract(("dew.rl",), (), ("dew.training",)),
    # An environment run's record is the configuration's (`dew.config.env`), not the objectives' (row 4267).
    "objectives import no trainer, configuration or front end": Contract(
        ("dew.objectives",), (), ("dew.training", "dew.config", "dew.checkpoints", "dew._cli")),
    "only the reinforcement-learning packages import dew.rl": Contract(
        ("dew",), ("dew.rl", "dew.objectives.rl"), ("dew.rl",)),
}
"""The contracts, by name. Each holds on main; a layer whose crossings are still open joins when its owner
settles them (rows 4264-4270, records/themes.md's RT07)."""


class Crossing(NamedTuple):
    """A module's import of a package its contract forbids."""

    contract: str
    module: str
    package: str


def _within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


def crossings(root: Path, layers: Mapping[str, Contract] | None = None) -> Iterator[Crossing]:
    """Every crossing of `layers` (`LAYERS`) in the `dew` package under `root`, by contract, module and
    package."""
    layers = LAYERS if layers is None else layers
    for path in sorted((root / "dew").rglob("*.py")):
        module = ".".join(path.relative_to(root).with_suffix("").parts).removesuffix(".__init__")
        held = [(name, contract) for name, contract in layers.items()
                if any(_within(module, prefix) for prefix in contract.modules)
                and not any(_within(module, left) for left in contract.excepted)]
        if not held:
            continue
        names = scopes.modules_imported(ast.parse(path.read_text()),
                                        scopes.package_of(module, path.name == "__init__.py"))
        for name, contract in held:
            for package in sorted({package for package in contract.forbidden
                                   if any(_within(found, package) for found in names)}):
                yield Crossing(name, module, package)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = Path(arguments[0] if arguments else "src")
    found = list(crossings(root))
    for crossing in found:
        print(f"{crossing.module} imports {crossing.package}: {crossing.contract}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
