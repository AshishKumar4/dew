"""The products one request of a saved run's task multiplies, in a fresh interpreter of its own, for
tests/saved_run_kinds.py's `computed`: `python tests/saved_run_products.py <run directory> [<step>]`
prints, as JSON, each module's dtypes by its variable scope's path (`products`), a packed projection's at
each of its members (`unpacked`), beside the projections the task's model packs, by path (`packs`).

The task loads at the step named, else its run's default, and answers one request (`answered`) with
JAX writing the StableHLO of each program it lowers (`jax_dump_ir_to`): a fresh process holds no
program another task compiled, so every one lowers. Each Flax module method the request runs is
intercepted (`flax.linen.intercept_methods`, `scoping`) to name its scope in the name stack (`SCOPE`):
the path of the variables it reads in the task's own tree, a shared scope's its owner's, whatever its
class or method; a root applied over a branch of the tree (a decision model's backbone or head) reads
beneath that branch's path. Each module is parsed into MLIR objects, and each `stablehlo.dot_general`
read for its operands' element types and the innermost scope its location names."""

import json
import sys
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

import jax
from flax import linen as nn
from flax.core.scope import Scope
from jax.extend.mlir import ir
from jax.interpreters import mlir
from saved_run_kinds import answered

from dew.batch import Variables, thaw
from dew.config import RunConfig
from dew.registry import objectives

SCOPE = "scope|"
"""What a module's name-stack segment begins with: its variable scope's path follows, joined by `|`."""

DTYPES = {"bf16": "bfloat16", "f16": "float16", "f32": "float32", "f64": "float64"}
"""The dtype each MLIR element type names."""


def scoping(variables) -> Callable:
    """The interceptor that runs each module method under a name naming its variable scope's path
    (`SCOPE`) in `variables`, the task's tree, beneath the branch its root was applied over: the one
    subtree of `variables["params"]` whose leaves' paths, shapes and dtypes are the root's own, empty
    where none is (a root over variables of its own). Two branches alike in every leaf cannot be told
    apart by what a root reads, and are refused rather than merged."""
    branches: dict[frozenset, list[tuple[str, ...]]] = {}

    def walk(tree, path: tuple[str, ...]) -> None:
        if isinstance(tree, Mapping):
            branches.setdefault(signature(tree), []).append(path)
            for name, child in tree.items():
                walk(child, (*path, str(name)))

    # A trunk held frozen is read as its whole tree is (`thaw`).
    walk(thaw(variables).get("params", {}), ())

    def branch(root: Scope) -> tuple[str, ...]:
        # Read off the root's own variables at each call, never kept by the root: a root made after an
        # earlier one is freed may take its address (row 2867).
        found = branches.get(signature(thaw(dict(root.variables())).get("params", {})), [()])
        if len(found) > 1:
            raise ValueError(f"a root reads one of the branches {found}, alike in every leaf")
        return found[0]

    def scoped(next_fun, args, kwargs, context):
        scope = context.module.scope
        path = () if scope is None else (*branch(scope.root), *scope.path)
        with jax.named_scope(SCOPE + "|".join(path)):
            return next_fun(*args, **kwargs)

    return scoped


def packed(task) -> dict[str, list[str]]:
    """The projections the task's decoder may serve packed (`dew.inference.projections.declared_groups`),
    each packed projection's members, both by their paths in the task's tree: its model's, or its
    backbone's under `backbone`, the decoder a decision model reads its rows through and packs to serve
    (`dew.decision.task._served`). A trunk held frozen declares its groups as its whole tree does."""
    from dew.batch import part
    from dew.inference.projections import declared_groups

    held: list[tuple[tuple[str, ...], nn.Module, Variables]] = [((), task.model, thaw(task.variables))]
    if hasattr(task.model, "backbone"):
        held.append((("backbone",), task.model.backbone, thaw(part(task.variables, "backbone"))))
    return {"/".join((*prefix, *group.path, group.packed)): ["/".join((*prefix, *group.path, member))
                                                             for member in group.members]
            for prefix, model, tree in held for group in declared_groups(model, tree)}


def signature(tree) -> frozenset:
    """A tree of variables by what a root reads of it: each leaf's path, shape and dtype."""
    leaves = jax.tree_util.tree_flatten_with_path(tree)[0]
    return frozenset((jax.tree_util.keystr(path), tuple(leaf.shape), str(leaf.dtype))
                     for path, leaf in leaves)


def module_path(location: ir.Location) -> str:
    """The variable scope of the innermost module a product's location names (`scoped`), its path
    joined by `/`; empty where it runs in none."""
    # A lowered operation's location names its name stack (`ir.NameLoc`), the call sites beneath it.
    segments = location.name_str.split("/") if isinstance(location, ir.NameLoc) else []
    scopes = [segment.removeprefix(SCOPE) for segment in segments if segment.startswith(SCOPE)]
    return scopes[-1].replace("|", "/") if scopes else ""


def products(modules: list[str]) -> dict[str, list[str]]:
    """Each module path's dtypes its products' operands hold, over `modules`, StableHLO text."""
    found: dict[str, set[str]] = {}

    def walk(operation: ir.Operation) -> None:
        for region in operation.regions:
            for block in region.blocks:
                for inner in block.operations:
                    if inner.operation.name == "stablehlo.dot_general":
                        found.setdefault(module_path(inner.location), set()).update(
                            DTYPES[str(ir.RankedTensorType(operand.type).element_type)]
                            for operand in inner.operands)
                    walk(inner.operation)

    with mlir.make_ir_context():
        for text in modules:
            walk(ir.Module.parse(text).operation)
    return {path: sorted(dtypes) for path, dtypes in sorted(found.items())}


def unpacked(products: Mapping[str, list[str]], packs: Mapping[str, list[str]]) -> dict[str, list[str]]:
    """`products` with each packed projection's dtypes counted at each of its members instead, `packs`
    naming each packed projection's members by their paths (`packed`): a task that serves a decoder's q,
    k and v as one `qkv_proj` multiplies them in one product, which a task that does not pack them
    multiplies in three, so the evidence reads alike whichever a main's task does. A module of the same
    name that no group declares keeps its own path."""
    found: dict[str, set[str]] = {}
    for path, dtypes in products.items():
        for member in packs.get(path, [path]):
            found.setdefault(member, set()).update(dtypes)
    return {path: sorted(dtypes) for path, dtypes in sorted(found.items())}


def main(directory: str, step: int | None) -> None:
    assert jax.default_backend() == "cpu", "a saved run's products are read on the suite's CPU"
    task = objectives[RunConfig.load(directory).objective.name].saved_task
    assert task is not None
    loaded = task.from_run(directory) if step is None else task.from_run(directory, step=step)
    with tempfile.TemporaryDirectory() as dump:
        jax.config.update("jax_dump_ir_to", dump)
        with nn.intercept_methods(scoping(loaded.variables)):
            answered(loaded)
        jax.config.update("jax_dump_ir_to", "")
        # The groups the task's model may serve packed, by name: each packed projection's members.
        found = products([path.read_text() for path in sorted(Path(dump).glob("*.mlir"))])
        packs = packed(loaded)
        print(json.dumps({"products": unpacked(found, packs), "packs": packs}))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
