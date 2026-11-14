"""Dew's layering contracts (docs/design/architecture.md; row 1557), read from the source: each contract's
modules import none of the packages it forbids, directly, whether the import is absolute, relative,
function-local or under TYPE_CHECKING, and a contract that holds over its modules' closure (`closure`), nor
through any module they import in turn (rows 5988-5990): the modules a module's import runs, its package's
initializers and a lazy package's export it names among them, each import of theirs followed but a type
checker's. A contract joins its closure as the bridges its modules' imports cross by are cut (theme RT07's
rows). A crossing names its contract, its module, the package it reaches and the chain of imports that
reaches it. tools/check_gates runs it on every head:

    python tools/lint_layers.py [SOURCE]

SOURCE is the directory that holds the `dew` package (`src`, the default).
"""

import ast
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_exports
import lint_scopes as scopes

FEATURES = ("dew.nn", "dew.data", "dew.inputs", "dew.objectives", "dew.training", "dew.inference",
            "dew.interop", "dew.rl", "dew.decision", "dew._cli", "dew.config", "dew.checkpoints",
            "dew.diffusion", "dew.sampling", "dew.eval", "dew.lora", "dew.quantization", "dew.telemetry",
            "dew.io", "dew._cleanup", "dew.figures")
"""Dew's feature packages, which the layers below them import none of."""


class Contract(NamedTuple):
    """The modules a contract holds (each package and its modules), those among them it leaves out, the
    packages none of the rest imports, and whether none reaches them through the modules it imports in turn
    either (`closure`)."""

    modules: tuple[str, ...]
    excepted: tuple[str, ...]
    forbidden: tuple[str, ...]
    closure: bool = False


LAYERS: Mapping[str, Contract] = {
    "conditioning inputs import no objective or trainer": Contract(
        ("dew.inputs",), (), ("dew.objectives", "dew.training", "dew.inference", "dew.rl", "dew.decision",
                              "dew._cli", "dew.config", "dew.checkpoints", "dew.sampling", "dew.eval",
                              "dew.lora", "dew.quantization", "dew.io", "dew._cleanup", "dew.figures")),
    "the data layer imports no feature package": Contract(
        ("dew.data",), (), tuple(name for name in FEATURES if name != "dew.data")),
    "telemetry imports nothing above the network layer": Contract(
        ("dew.telemetry",), (), tuple(name for name in FEATURES if name not in ("dew.nn", "dew.telemetry")),
        closure=True),
    # The algebra rewrites the network layer's programs and binds its operations (`recognize`'s attention
    # among them), so the two are one layer, nothing above which either reaches (rows 5988, 5989).
    "the algebra imports nothing above the network layer": Contract(
        ("dew.algebra",), (), tuple(name for name in FEATURES if name not in ("dew.nn", "dew.telemetry")),
        closure=True),
    "the network layer imports no trainer": Contract(("dew.nn",), (), ("dew.training",), closure=True),
    "kernels import nothing above the network layer": Contract(
        ("dew.nn.kernels",), (), tuple(name for name in FEATURES if name not in ("dew.nn", "dew.telemetry"))),
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
    # A model computes; where a release stores its tensors and config is dew.interop's (row 4264).
    "models import no weight interop": Contract(("dew.nn",), (), ("dew.interop",), closure=True),
    # A model's inputs are its own types (`dew.nn.inputs`, `dew.nn.flops`), which the diffusion process and
    # the condition encoders read; a release's preprocessing is interop's (row 4265).
    "models import no diffusion, conditioning or data module": Contract(
        ("dew.nn",), (), ("dew.diffusion", "dew.inputs", "dew.data"), closure=True),
    # A readout's source states its own axes (`HeadComponent.axes`), so a release's reader names no task
    # (row 4269).
    "interop imports no decision module": Contract(("dew.interop",), (), ("dew.decision",)),
}
"""The contracts, by name. Each holds on main; a layer whose crossings are still open joins when its owner
settles them (rows 4264-4270, records/themes.md's RT07)."""


class Crossing(NamedTuple):
    """A module's import of a package its contract forbids, and the chain of modules its import runs to it
    (the module's own direct import, its last a module of that package; empty for an import a type checker
    alone reads)."""

    contract: str
    module: str
    package: str
    chain: tuple[str, ...] = ()


def _within(name: str, prefix: str) -> bool:
    return name == prefix or name.startswith(prefix + ".")


class Source(NamedTuple):
    """The `dew` package under a root, read once: each module's path, the names every import of its reads
    (`lint_scopes.modules_imported`) and the modules its run imports in turn (`runs`)."""

    paths: Mapping[str, Path]
    named: Mapping[str, set[str]]
    runs: Mapping[str, frozenset[str]]


def read(root: Path) -> Source:
    """The `dew` package under `root` (`Source`)."""
    paths = {".".join(path.relative_to(root).with_suffix("").parts).removesuffix(".__init__"): path
             for path in sorted((root / "dew").rglob("*.py"))}
    trees = {module: ast.parse(path.read_text()) for module, path in paths.items()}
    parents = {module: scopes.package_of(module, path.name == "__init__.py")
               for module, path in paths.items()}
    named = {module: scopes.modules_imported(tree, parents[module]) for module, tree in trees.items()}
    exported = {module: gen_exports.exported(tree) for module, tree in trees.items()}
    runs = {module: frozenset(_run(_executed(tree), parents[module], paths, exported))
            for module, tree in trees.items()}
    return Source(paths, named, runs)


def _executed(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    """The imports a module's run may execute: every one, a function's included, but those under
    TYPE_CHECKING, which a type checker alone reads."""
    checking = {id(inner) for node in ast.walk(tree) if isinstance(node, ast.If) and _type_checking(node.test)
                for statement in node.body for inner in ast.walk(statement)}
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Import | ast.ImportFrom) and id(node) not in checking]


def _type_checking(test: ast.expr) -> bool:
    return ((isinstance(test, ast.Name) and test.id == "TYPE_CHECKING")
            or (isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"))


def _run(imports: Sequence[ast.Import | ast.ImportFrom], parent: Sequence[str], paths: Mapping[str, Path],
         exported: Mapping[str, Mapping[str, str]]) -> Iterator[str]:
    """The `dew` modules `imports` run: each module an import names and every package above it (whose
    initializer runs first), and of a `from` import's names, each a module of its package or a lazy
    package's export (`_exported`); a star import of a lazy package reads every name it exports (its
    generated `__all__`, tools/gen_exports.py)."""
    for node in imports:
        if isinstance(node, ast.Import):
            named, taken = [alias.name for alias in node.names], []
        else:
            base = scopes.imported_from(node, parent)
            named = [base]
            taken = [(base, name) for alias in node.names
                     for name in (exported.get(base, {}) if alias.name == "*" else (alias.name,))]
        for name in named:
            yield from _prefixes(name, paths)
        for base, name in taken:
            if f"{base}.{name}" in paths:
                yield f"{base}.{name}"
            else:
                yield from _exported(base, name, paths, exported)


def _prefixes(name: str, paths: Mapping[str, Path]) -> Iterator[str]:
    """Module `name` and every package above it, as importing it runs them, those under the root alone."""
    parts = name.split(".")
    return (prefix for prefix in (".".join(parts[:at]) for at in range(1, len(parts) + 1)) if prefix in paths)


def _exported(package: str, name: str, paths: Mapping[str, Path],
              exported: Mapping[str, Mapping[str, str]]) -> Iterator[str]:
    """The modules reading `name` off `package` runs where `package` exports it lazily
    (`gen_exports.exported`, read by `dew._exports.lookup`): the module that defines it, with every package
    above it, and where that module is the package's own module of that name, nothing more; else the name
    read off that module in turn, a lazy package's export of it followed as its lookup's `getattr` follows
    it."""
    seen = set()
    while (package, name) not in seen and (target := exported.get(package, {}).get(name)) is not None:
        seen.add((package, name))
        yield from _prefixes(target, paths)
        if target == f"{package}.{name}":
            return
        package = target


def _chains(source: Source, start: str) -> Mapping[str, tuple[str, ...]]:
    """Every module `start`'s run imports, each by the shortest chain of imports from `start` to it (`start`
    excluded, the module itself last)."""
    chains: dict[str, tuple[str, ...]] = {start: ()}
    frontier = [start]
    while frontier:
        following = []
        for module in frontier:
            for imported in sorted(source.runs.get(module, ())):
                if imported not in chains:
                    chains[imported] = (*chains[module], imported)
                    following.append(imported)
        frontier = following
    del chains[start]
    return chains


def crossings(root: Path, layers: Mapping[str, Contract] | None = None) -> Iterator[Crossing]:
    """Every crossing of `layers` (`LAYERS`) in the `dew` package under `root`, by contract, module and
    package: a module's direct import of a forbidden package, a type checker's included, else, under a
    contract over its closure, the shortest chain of imports its run takes to one."""
    layers = LAYERS if layers is None else layers
    source = read(root)
    for module in source.paths:
        held = [(name, contract) for name, contract in layers.items()
                if any(_within(module, prefix) for prefix in contract.modules)
                and not any(_within(module, left) for left in contract.excepted)]
        if not held:
            continue
        chains = _chains(source, module) if any(contract.closure for _, contract in held) else {}
        for name, contract in held:
            for package in sorted(contract.forbidden):
                if any(_within(found, package) for found in source.named[module]):
                    yield Crossing(name, module, package)
                    continue
                reached = sorted((chain for found, chain in chains.items()
                                  if contract.closure and _within(found, package)),
                                 key=lambda chain: (len(chain), chain))
                if reached:
                    yield Crossing(name, module, package, reached[0])


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = Path(arguments[0] if arguments else "src")
    found = list(crossings(root))
    for crossing in found:
        through = f" through {' -> '.join(crossing.chain)}" if crossing.chain else ""
        print(f"{crossing.module} imports {crossing.package}{through}: {crossing.contract}")
    return 1 if found else 0


if __name__ == "__main__":
    sys.exit(main())
