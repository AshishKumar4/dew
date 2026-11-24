"""A bit form's decode emitted inline (docs/design/algebra.md §7.4): each output register of a `bits.BitForm`
over a register tile of stored words, one PTX block a word (`BitForm.ptx`), through Mosaic GPU's inline
assembly (`plgpu.inline_mgpu`, LLVM's `inline_asm` on each 32-bit word of each register), so the decoder is
the form's own instructions on the pipes its steps name rather than XLA's lowering of their `jax.numpy`
reading (`BitForm.traced`, which an interpreter runs, where no inline assembly does); and a register tile's
IEEE fp32 quotients the same way (`quotients`: PTX's `div.rn.f32`)."""

import math

import numpy as np
from jax import numpy as jnp
from jax.experimental.mosaic.gpu import fragmented_array as mgpu_fragments
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir as mgpu_ir
from jax.extend.mlir.dialects import vector as mgpu_vector
from jaxlib.mlir.dialects import llvm as mgpu_llvm

from dew.algebra import bits
from dew.nn.kernels.templates import emission
from dew.nn.kernels.templates.cost import Work


def registers(form: bits.BitForm, words, layout) -> tuple:
    """Each output register of `form` over `words` (uint32, a register tile in `layout`), as `form.traced`
    gives them, each a uint32 tile in `layout`; its work the form's over every word (`BitForm.work`), which
    a kernel's emission counts where the body, opaque to it, is inlined (`emission.stating`)."""
    typed = plgpu.ShapeDtypeStruct(words.shape, jnp.dtype(jnp.uint32), layout=layout)

    @emission.stating(lambda stored: form.work(math.prod(stored.shape)))
    def decoded(ctx, words):
        return tuple(_asm((words,), form.ptx(output), "r") for output in range(len(form.outputs)))

    return plgpu.inline_mgpu(arg_types=(layout,), return_type=(typed,) * len(form.outputs))(decoded)(words)


def quotients(numerator, denominator, layout):
    """`numerator / denominator` (an fp32 register tile in `layout`, by one of its shape and layout or by a
    Python number, the instruction's immediate) as IEEE fp32 division rounds it, subnormals kept: PTX's
    `div.rn.f32` an element, the correctly rounded quotient (`precision.correctly_rounded_divide`'s), where
    its software form issued dozens of instructions an element and spilled a quantizing epilogue on tcgen05
    (row 12721). A constant's tile is no operand: Mosaic GPU casts no splat to a register layout for an
    inline body (witness/b1-probe-6, sm100)."""
    typed = plgpu.ShapeDtypeStruct(numerator.shape, jnp.dtype(jnp.float32), layout=layout)
    constant = isinstance(denominator, (int, float))
    divisor = f"0f{int(np.float32(denominator).view(np.uint32)):08X}" if constant else "$2"
    operands = (numerator,) if constant else (numerator, denominator)

    @emission.stating(lambda numerator, *_: Work(operations=math.prod(numerator.shape)))
    def divided(ctx, *tiles):
        del ctx
        return (_asm(tiles, f"div.rn.f32 $0, $1, {divisor};", "f", signed=None),)

    return plgpu.inline_mgpu(arg_types=(layout,) * len(operands), return_type=(typed,))(divided)(*operands)[0]


def _asm(arrays: tuple, ptx: str, kind: str, signed: bool | None = False):
    """`ptx` on each element of `arrays`' registers (one shape and layout), one 32-bit result of one 32-bit
    element of each, of LLVM constraint `kind` (`r` an integer word's, `f` an fp32's), an element at a time;
    the result `signed` as an integer tile's is (None for a float's)."""
    element = (mgpu_ir.IntegerType.get_signless(32) if kind == "r" else mgpu_ir.F32Type.get())
    constraints = ",".join([f"={kind}", *(kind for _ in arrays)])
    first = arrays[0]
    out = np.empty_like(first.registers)
    for index, register in np.ndenumerate(first.registers):
        [count] = mgpu_ir.VectorType(register.type).shape
        done = mgpu_llvm.mlir_undef(register.type)
        for at in range(count):
            position = mgpu_ir.DenseI64ArrayAttr.get([at])
            values = [mgpu_vector.extract(array.registers[index], dynamic_position=[],
                                          static_position=position) for array in arrays]
            result = mgpu_llvm.inline_asm(element, values, ptx, constraints)
            done = mgpu_vector.insert(result, done, dynamic_position=[], static_position=position)
        out[index] = done
    return mgpu_fragments.FragmentedArray(_registers=out, _layout=first.layout, _is_signed=signed)
