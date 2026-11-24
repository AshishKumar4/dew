"""A bit form's decode emitted inline (docs/design/algebra.md §7.4): each output register of a `bits.BitForm`
over a register tile of stored words, one PTX block a word (`BitForm.ptx`), through Mosaic GPU's inline
assembly (`plgpu.inline_mgpu`, LLVM's `inline_asm` on each 32-bit word of each register), so the decoder is
the form's own instructions on the pipes its steps name rather than XLA's lowering of their `jax.numpy`
reading (`BitForm.traced`, which an interpreter runs, where no inline assembly does); and a register tile's
IEEE fp32 quotients the same way (`quotients`: PTX's `div.rn.f32`), and its 4-bit codes two a byte
(`paired`: PTX's `bfi.b32`)."""


import numpy as np
from jax import numpy as jnp
from jax.experimental.mosaic.gpu import fragmented_array as mgpu_fragments
from jax.experimental.pallas import mosaic_gpu as plgpu
from jax.extend.mlir import ir as mgpu_ir
from jax.extend.mlir.dialects import arith as mgpu_arith, vector as mgpu_vector
from jaxlib.mlir.dialects import llvm as mgpu_llvm

from dew.algebra import bits


def registers(form: bits.BitForm, words, layout) -> tuple:
    """Each output register of `form` over `words` (uint32, a register tile in `layout`), as `form.traced`
    gives them, each a uint32 tile in `layout`: one PTX block a word computing every register
    (`BitForm.ptx_registers`), each step once, which a kernel's emission counts as emitted
    (`inline_work.body_work`)."""
    typed = plgpu.ShapeDtypeStruct(words.shape, jnp.dtype(jnp.uint32), layout=layout)

    def decoded(ctx, words):
        return _asm((words,), form.ptx_registers(), len(form.outputs))

    return plgpu.inline_mgpu(arg_types=(layout,), return_type=(typed,) * len(form.outputs))(decoded)(words)


def quotients(numerator, denominator, layout):
    """`numerator / denominator` (an fp32 register tile in `layout`, by one of its shape and layout or by a
    Python number, the instruction's immediate) as IEEE fp32 division rounds it, subnormals kept: PTX's
    `div.rn.f32` an element, the correctly rounded quotient (`precision.correctly_rounded_divide`'s), where
    its software form issued dozens of instructions an element and spilled a quantizing epilogue on tcgen05
    (row 12721); a kernel's emission counts it as emitted (`inline_work.body_work`). A constant's tile is no
    operand: Mosaic GPU casts no splat to a register layout for an inline body (witness/b1-probe-6, sm100)."""
    typed = plgpu.ShapeDtypeStruct(numerator.shape, jnp.dtype(jnp.float32), layout=layout)
    constant = isinstance(denominator, (int, float))
    divisor = f"0f{int(np.float32(denominator).view(np.uint32)):08X}" if constant else "$2"
    operands = (numerator,) if constant else (numerator, denominator)

    def divided(ctx, *tiles):
        del ctx
        return _asm(tiles, f"div.rn.f32 $0, $1, {divisor};", 1, kind="f")

    inlined = plgpu.inline_mgpu(arg_types=(layout,) * len(operands), return_type=(typed,))(divided)
    return inlined(*operands)[0]


def paired(codes, layout):
    """Each two adjacent columns' 4-bit codes of `codes` (a uint8 tile of codes below 16 in `layout`, whose
    registers hold column pairs: WGMMA's and tcgen05's) one byte, the even column's in its low nibble, as a
    sub-byte array packs them (XLA's float4): `[rows, columns / 2]` uint8 in the layout halved along its
    columns (`halved`), each byte in the thread that held its pair, one PTX bit-field insert a byte (row
    12743). Mosaic GPU takes no strided slice of a register tile (witness/packing-1), so the pairs are read
    in the registers that hold them."""
    packed = halved(layout)
    rows, columns = codes.shape
    typed = plgpu.ShapeDtypeStruct((rows, columns // 2), jnp.dtype(jnp.uint8), layout=packed)

    def pairs(ctx, codes):
        del ctx
        i8, i32 = mgpu_ir.IntegerType.get_signless(8), mgpu_ir.IntegerType.get_signless(32)
        byte = mgpu_ir.VectorType.get([1], i8)
        out = np.empty_like(codes.registers)
        for index, register in np.ndenumerate(codes.registers):
            low, high = (mgpu_arith.extui(i32, mgpu_vector.extract(
                register, dynamic_position=[], static_position=mgpu_ir.DenseI64ArrayAttr.get([at])))
                for at in (0, 1))
            # The odd column's code into bits 4 to 7 of the even column's.
            both = mgpu_llvm.inline_asm(i32, [low, high], "bfi.b32 $0, $2, $1, 4, 4;", "=r,r,r")
            out[index] = mgpu_vector.from_elements(byte, [mgpu_arith.trunci(i8, both)])
        return (mgpu_fragments.FragmentedArray(_registers=out, _layout=packed.to_mgpu(), _is_signed=False),)

    return plgpu.inline_mgpu(arg_types=(layout,), return_type=(typed,))(pairs)(codes)[0]


def halved(layout):
    """`layout` (a register layout whose vectors are two adjacent columns, its tiling's last tile `(2,)`)
    with each tile's columns halved and a vector of one: the layout of a tile of each pair's one value, held
    by the thread that held the pair (`paired`)."""
    tiled = layout.to_mgpu()
    *tiles, vector = tiled.tiling.tiles
    if vector != (2,) or any(tile[-1] % 2 for tile in tiles):
        raise ValueError(f"{layout}'s registers hold no column pairs to halve")
    narrow = tuple((*tile[:-1], tile[-1] // 2) for tile in tiles)
    return plgpu.Layout.TILED(mgpu_fragments.Tiling((*narrow, (1,))), warp_dims=tiled.warp_dims,
                              lane_dims=tiled.lane_dims, vector_dim=tiled.vector_dim)


def _asm(arrays: tuple, ptx: str, registers_out: int, *, kind: str = "r") -> tuple:
    """`ptx` (`registers_out` 32-bit registers of one 32-bit element of each of `arrays`, of one shape and
    layout; constraints `=k,...,k`, `k` its `kind`, `r` an integer word's and `f` an fp32's) on each element
    of `arrays`' registers, an element a block: each output register's array."""
    element, name = ((mgpu_ir.IntegerType.get_signless(32), "i32") if kind == "r"
                     else (mgpu_ir.F32Type.get(), "f32"))
    # One output register is its block's value, several a struct of them (as Mosaic GPU's `inline_ptx`
    # returns them): LLVM refuses a struct of one output.
    returned = (element if registers_out == 1
                else mgpu_ir.Type.parse(f"!llvm.struct<({', '.join([name] * registers_out)})>"))
    constraints = ",".join([f"={kind}"] * registers_out + [kind] * len(arrays))
    first = arrays[0]
    outs = [np.empty_like(first.registers) for _ in range(registers_out)]
    for index, register in np.ndenumerate(first.registers):
        [count] = mgpu_ir.VectorType(register.type).shape
        decoded = [mgpu_llvm.mlir_undef(register.type) for _ in range(registers_out)]
        for at in range(count):
            position = mgpu_ir.DenseI64ArrayAttr.get([at])
            taken = [mgpu_vector.extract(array.registers[index], dynamic_position=[],
                                         static_position=position) for array in arrays]
            done = mgpu_llvm.inline_asm(returned, taken, ptx, constraints)
            for output in range(registers_out):
                value = done if registers_out == 1 else mgpu_llvm.extractvalue(element, done, [output])
                decoded[output] = mgpu_vector.insert(value, decoded[output], dynamic_position=[],
                                                     static_position=position)
        for output in range(registers_out):
            outs[output][index] = decoded[output]
    return tuple(mgpu_fragments.FragmentedArray(_registers=out, _layout=first.layout,
                                                _is_signed=False if kind == "r" else None)
                 for out in outs)
