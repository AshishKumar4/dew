"""The products one request of a saved run's task multiplies, in a fresh interpreter of its own, for
tests/saved_run_kinds.py's `computed`: `python tests/saved_run_products.py <run directory> [<step>]`
prints, as JSON, each module path's dtypes (`products`).

The task loads at the step named, else its run's default, and answers one request (`answered`) with
JAX writing the StableHLO of each program it lowers (`jax_dump_ir_to`): a fresh process holds no
program another task compiled, so every one lowers. Each module is parsed into MLIR objects, and each
`stablehlo.dot_general` read for its operands' element types and the module path its location's name
stack states."""

import json
import re
import sys
import tempfile
from pathlib import Path

import jax
from jax.extend.mlir import ir
from jax.interpreters import mlir
from saved_run_kinds import answered

from dew.config import RunConfig
from dew.registry import objectives

NAMED = re.compile(r'loc\("([^"]*)"')
"""A location's name stack, which an operation's printed location leads with."""

CONTROL = re.compile(r"while|body|cond|scan|eval_jaxpr|checkpoint|remat|rematted_computation|branch_\d+_fun")
"""The name stack's segments JAX's own control flow and transforms push, no module's."""

DTYPES = {"bf16": "bfloat16", "f16": "float16", "f32": "float32", "f64": "float64"}
"""The dtype each MLIR element type names."""


def module_path(location: ir.Location) -> str:
    """The module path a product's location names: its name stack's module scopes, each an identifier
    that is neither a method's (`module._method`), a transform's (`jit(...)`, an einsum's subscripts) nor
    JAX's control flow (`CONTROL`), its primitive's last segment dropped."""
    named = NAMED.match(str(location))
    segments = [] if named is None else named.group(1).split("/")[:-1]
    return "/".join(segment for segment in segments
                    if segment.isidentifier() and CONTROL.fullmatch(segment) is None)


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


def main(directory: str, step: int | None) -> None:
    assert jax.default_backend() == "cpu", "a saved run's products are read on the suite's CPU"
    task = objectives[RunConfig.load(directory).objective.name].saved_task
    assert task is not None
    loaded = task.from_run(directory) if step is None else task.from_run(directory, step=step)
    with tempfile.TemporaryDirectory() as dump:
        jax.config.update("jax_dump_ir_to", dump)
        answered(loaded)
        jax.config.update("jax_dump_ir_to", "")
        print(json.dumps(products([path.read_text() for path in sorted(Path(dump).glob("*.mlir"))])))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else None)
