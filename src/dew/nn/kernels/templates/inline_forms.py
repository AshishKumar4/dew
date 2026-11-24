"""A bit form's decode emitted inline (docs/design/algebra.md §7.4): each output register of a `bits.BitForm`
over a register tile of stored words, one PTX block a word (`BitForm.ptx`), through Mosaic GPU's inline
assembly (`plgpu.inline_mgpu`, LLVM's `inline_asm` on each 32-bit word of each register), so the decoder is
the form's own instructions on the pipes its steps name rather than XLA's lowering of their `jax.numpy`
reading (`BitForm.traced`, which an interpreter runs, where no inline assembly does)."""


import numpy as np
from jax import numpy as jnp
from jax.experimental.mosaic.gpu import fragmented_array as mgpu_fragments
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir as mgpu_ir
from jax.extend.mlir.dialects import vector as mgpu_vector
from jaxlib.mlir.dialects import llvm as mgpu_llvm

from dew.algebra import bits


def registers(form: bits.BitForm, words, layout) -> tuple:
    """Each output register of `form` over `words` (uint32, a register tile in `layout`), as `form.traced`
    gives them, each a uint32 tile in `layout`: one PTX block a word computing every register
    (`BitForm.ptx_registers`), each step once, which a kernel's emission counts as emitted
    (`inline_work.body_work`)."""
    typed = plgpu.ShapeDtypeStruct(words.shape, jnp.dtype(jnp.uint32), layout=layout)

    def decoded(ctx, words):
        return _asm(words, form.ptx_registers(), len(form.outputs))

    return plgpu.inline_mgpu(arg_types=(layout,), return_type=(typed,) * len(form.outputs))(decoded)(words)


def _asm(array, ptx: str, results: int) -> tuple:
    """`ptx` (`results` 32-bit results of one 32-bit word, constraints `=r,...,r`) on each word of
    `array`'s registers, a word a block: each result's array."""
    i32 = mgpu_ir.IntegerType.get_signless(32)
    returned = mgpu_ir.Type.parse(f"!llvm.struct<({', '.join(['i32'] * results)})>")
    constraints = ",".join(["=r"] * results + ["r"])
    outs = [np.empty_like(array.registers) for _ in range(results)]
    for index, register in np.ndenumerate(array.registers):
        [count] = mgpu_ir.VectorType(register.type).shape
        decoded = [mgpu_llvm.mlir_undef(register.type) for _ in range(results)]
        for at in range(count):
            position = mgpu_ir.DenseI64ArrayAttr.get([at])
            word = mgpu_vector.extract(register, dynamic_position=[], static_position=position)
            done = mgpu_llvm.inline_asm(returned, [word], ptx, constraints)
            for result in range(results):
                value = mgpu_llvm.extractvalue(i32, done, [result])
                decoded[result] = mgpu_vector.insert(value, decoded[result], dynamic_position=[],
                                                     static_position=position)
        for result in range(results):
            outs[result][index] = decoded[result]
    return tuple(mgpu_fragments.FragmentedArray(_registers=out, _layout=array.layout, _is_signed=False)
                 for out in outs)
