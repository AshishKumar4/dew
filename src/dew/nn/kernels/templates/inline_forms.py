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
    gives them, each a uint32 tile in `layout`."""
    typed = plgpu.ShapeDtypeStruct(words.shape, jnp.dtype(jnp.uint32), layout=layout)
    return plgpu.inline_mgpu(arg_types=(layout,), return_type=(typed,) * len(form.outputs))(
        lambda ctx, words: tuple(_asm(words, form.ptx(output)) for output in range(len(form.outputs))))(words)


def _asm(array, ptx: str):
    """`ptx` (constraints `=r,r`: one 32-bit result of one 32-bit word) on each word of `array`'s registers,
    a register's words each their own block."""
    i32 = mgpu_ir.IntegerType.get_signless(32)
    out = np.empty_like(array.registers)
    for index, register in np.ndenumerate(array.registers):
        [count] = mgpu_ir.VectorType(register.type).shape
        decoded = mgpu_llvm.mlir_undef(register.type)
        for at in range(count):
            position = mgpu_ir.DenseI64ArrayAttr.get([at])
            word = mgpu_vector.extract(register, dynamic_position=[], static_position=position)
            done = mgpu_llvm.inline_asm(i32, [word], ptx, "=r,r")
            decoded = mgpu_vector.insert(done, decoded, dynamic_position=[], static_position=position)
        out[index] = decoded
    return mgpu_fragments.FragmentedArray(_registers=out, _layout=array.layout, _is_signed=False)
