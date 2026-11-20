"""Precision's error-free transforms marked to round alone where the lowering's compiler honours the fork's
no-contract mark, in place of the optimization barrier (row 8378; `dew.nn.precision.rounding_alone`)."""

import jax
import jax.numpy as jnp


def float_arithmetic(lowered) -> list[str | None]:
    """The no-contract mark (`jax.extend.backend.no_contract_attribute`'s attribute, its value) of every float
    add, subtract and multiply of `lowered`'s StableHLO module, a reduction's body and every other nested
    region included, None where an operation carries none."""
    from jax.extend.backend import no_contract_attribute
    from jax.extend.mlir import ir

    name = no_contract_attribute()
    module = lowered.compiler_ir(dialect="stablehlo")
    found = []

    def walk(operation):
        if operation.name in ("stablehlo.add", "stablehlo.subtract", "stablehlo.multiply") and isinstance(
                ir.RankedTensorType(operation.results[0].type).element_type, ir.FloatType):
            attributes = operation.attributes
            marks = ({entry.name: ir.StringAttr(entry.attr).value
                      for entry in ir.DictAttr(attributes["mhlo.frontend_attributes"])}
                     if "mhlo.frontend_attributes" in attributes else {})
            found.append(marks.get(name) if name is not None else None)
        for region in operation.regions:
            for block in region.blocks:
                for inner in block.operations:
                    walk(inner.operation)

    with module.context:
        walk(module.operation)
    return found


def test_an_error_free_transform_marks_its_arithmetic_where_the_jaxlib_marks():
    """Where the jaxlib marks a float operation to round alone and the lowering's compiler fences the mark (a
    CPU's, `dew.nn.precision.MARKED_BACKENDS`), `two_sum`, `two_product`, a compensated sum (its reduction's
    body among them) and a forward derivative of `two_sum` lower with every float add, subtract and multiply
    so marked and no optimization barrier, which split the fusions around every transform (row 8378); on the
    barrier path (`dew._cache.BARRIERED`), and on a jaxlib that marks nothing, they lower as before, their
    operands behind a barrier and nothing marked. The same jitted callables lowered before, under and after
    the barrier path: a trace of one path is never run on the other."""
    from jax.extend.backend import no_contract_attribute

    from dew import _cache
    from dew.nn.kernels.lowered import operations
    from dew.nn.precision import compensated_sum, two_product, two_sum

    x, y = jnp.linspace(1.0, 2.0, 64, dtype=jnp.float32), jnp.linspace(3.0, 6.0, 64, dtype=jnp.float32)
    transforms = [jax.jit(two_sum), jax.jit(two_product), jax.jit(lambda x, y: compensated_sum(y, (0,))),
                  jax.jit(lambda x, y: jax.jvp(two_sum, (x, y), (y, x)))]

    def lowered():
        return [transform.lower(x, y) for transform in transforms]

    def barriered(programs):
        for program in programs:
            assert "stablehlo.optimization_barrier" in operations(program)
            assert not any(float_arithmetic(program))

    if no_contract_attribute() is None:
        barriered(lowered())
        return
    for _ in range(2):
        for program in lowered():
            assert "stablehlo.optimization_barrier" not in operations(program)
            marks = float_arithmetic(program)
            assert marks and all(mark == "true" for mark in marks), marks
        with _cache.BARRIERED(new_value=True):
            barriered(lowered())
