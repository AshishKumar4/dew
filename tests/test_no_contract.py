"""Precision's error-free transforms marked to round alone where the lowering's compiler honours the fork's
no-contract mark, in place of the optimization barrier (row 8378; `dew.nn.precision.rounding_alone`)."""

import jax
import jax.numpy as jnp
import numpy as np


def float_arithmetic(lowered, *, paired: bool = False) -> list[str | None]:
    """The no-contract mark (`jax.extend.backend.no_contract_attribute`'s attribute, its value) of every float
    add, subtract and multiply of `lowered`'s StableHLO module (`dew.nn.kernels.lowered.read_operations`, a
    reduction's body among them), None where an operation carries none; with `paired`, those in a reduction
    of pairs alone (a compensated sum's `(hi, lo)` accumulator, two operands and two inits), whose plain sum
    beside it and its pair's final addition (`compensated_total`) no transform makes."""
    from jax.extend.backend import no_contract_attribute
    from jax.extend.mlir import ir

    from dew.nn.kernels.lowered import frontend_attributes, read_operations

    name = no_contract_attribute()

    def in_pairs(operation) -> bool:
        parent = operation.parent
        while parent is not None and parent.name != "stablehlo.reduce":
            parent = parent.parent
        return parent is not None and len(parent.operands) == 4

    def read(operation):
        arithmetic = operation.name in ("stablehlo.add", "stablehlo.subtract", "stablehlo.multiply")
        if not arithmetic or not isinstance(ir.RankedTensorType(operation.results[0].type).element_type,
                                            ir.FloatType):
            return None
        if paired and not in_pairs(operation):
            return None
        return (frontend_attributes(operation).get(name) if name is not None else None,)

    return [mark for (mark,) in read_operations(lowered, read)]


def test_an_error_free_transform_marks_its_arithmetic_where_the_jaxlib_marks():
    """Where the jaxlib marks a float operation to round alone, `two_sum`, `two_product`, a compensated sum
    and a forward derivative of `two_sum` trace with every float add, subtract and multiply so marked (a
    compensated sum's in its reduction of pairs); traced for a CPU and lowered for a CPU or for CUDA, whose
    compilers round a marked operation alone (`dew.nn.precision.MARKED_BACKENDS`), with no optimization
    barrier, which split the fusions around every transform; lowered from the same trace for a TPU, whose
    compiler is no such one, with its barrier (row 8378). Traced for a GPU they are marked nowhere and keep
    their barrier, as a raw Pallas body a GPU traces must (row 10138). On the barrier path
    (`dew._cache.BARRIERED`), and on a jaxlib that marks nothing, they lower as before, their operands behind
    a barrier and nothing marked. The same jitted callables lowered before, under and after the barrier
    path: a trace of one path is never run on the other."""
    from jax.extend.backend import no_contract_attribute

    from dew import _cache
    from dew.nn.kernels import generation
    from dew.nn.kernels.lowered import operations
    from dew.nn.precision import compensated_sum, two_product, two_sum

    cpu = jax.devices("cpu")[0]
    x = jax.device_put(jnp.linspace(1.0, 2.0, 64, dtype=jnp.float32), cpu)
    y = jax.device_put(jnp.linspace(3.0, 6.0, 64, dtype=jnp.float32), cpu)
    # Each transform, and whether its marks are read in its reduction of pairs alone (`float_arithmetic`).
    transforms = [(jax.jit(two_sum), False), (jax.jit(two_product), False),
                  (jax.jit(lambda x, y: compensated_sum(y, (0,))), True),
                  (jax.jit(lambda x, y: jax.jvp(two_sum, (x, y), (y, x))), False)]

    def lowered(platform: str, *, assumed: str | None = None):
        # Traced for a CPU (`jax.default_device`, the platform a call's code is chosen for), whatever the
        # process's default, or for a GPU of the `assumed` generation, then lowered for `platform`.
        with jax.default_device(cpu), generation.assumed(assumed):
            return [(transform.trace(x, y).lower(lowering_platforms=(platform,)), paired)
                    for transform, paired in transforms]

    def held(programs, *, barrier: bool, marked: bool):
        for program, paired in programs:
            assert ("stablehlo.optimization_barrier" in operations(program)) == barrier
            marks = float_arithmetic(program, paired=paired)
            assert marks and all((mark == "true") == marked for mark in marks), marks

    if no_contract_attribute() is None:
        held(lowered("cpu"), barrier=True, marked=False)
        return
    for _ in range(2):
        held(lowered("cpu"), barrier=False, marked=True)
        held(lowered("cuda"), barrier=False, marked=True)
        held(lowered("cuda", assumed="sm90"), barrier=True, marked=False)
        held(lowered("tpu"), barrier=True, marked=True)
        with _cache.BARRIERED(new_value=True):
            held(lowered("cpu"), barrier=True, marked=False)


def a_marked_multiply_add_rounds_each_step():
    """On a GPU, `a * b + c` with its add and multiply marked to round alone (the jaxlib's no-contract
    attribute, `set_xla_metadata`, as `dew.nn.precision.rounding_alone` marks a transform a CPU traces for
    CUDA) over 65536 normal draws is bitwise the product rounded and then the sum rounded (NumPy's float32
    steps), and the draws tell that from a fused multiply-add's one rounding: should a later XLA:GPU contract
    a marked operation, as precision's transforms lowered for CUDA without their barrier take it not to
    (`dew.nn.precision.MARKED_BACKENDS`), this fails (row 10135: none contracted on an A100 or a GH200 at
    post115). The GPU's own compiler's (test_gpu_numerics), its program compiled where the cell's arm runs
    (`ahead_arm`)."""
    from jax.experimental.xla_metadata import set_xla_metadata
    from reference_error import ahead_arm

    from dew.nn.precision import no_contract

    name = no_contract()
    assert name is not None, "this jaxlib marks no operation to round alone"
    rng = np.random.default_rng(2026_1120)
    a, b, c = (rng.standard_normal(65536).astype(np.float32) for _ in range(3))

    def marked(a, b, c):
        with set_xla_metadata(**{name: True}):
            return a * b + c

    marks = float_arithmetic(jax.jit(marked).trace(a, b, c).lower())
    assert marks and all(mark == "true" for mark in marks), marks
    found = np.asarray(ahead_arm(jax.jit(marked), "a marked multiply-add")(a, b, c))
    stepped = a * b + c
    fused = (a.astype(np.float64) * b + c).astype(np.float32)
    assert np.any(fused != stepped), "no draw tells a fused multiply-add from its steps"
    np.testing.assert_array_equal(found.view(np.uint32), stepped.view(np.uint32))
