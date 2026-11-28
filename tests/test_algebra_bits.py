"""Bit-level forms (docs/design/algebra.md §7, theme T82): a format's decode declared as the placement of its
codes' fields, its cheapest instruction form found by the price-ordered search and proven on every word, and
the decoded registers read as the consumer reads them, values its format defines."""

import dataclasses
import functools
import itertools

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from arm_arithmetic import drawn_on_the_host
from low_precision_support import spread
from map_judge import sharded
from reference_error import ahead_arm, ran_here

from dew.algebra import bits

# E2M1 (OCP Microscaling Formats v1.0, §5.3.3): sign, two exponent bits of bias 1, one mantissa bit; a zero
# exponent is subnormal. Its eight magnitudes by code.
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _e4m3(code: int) -> float:
    """An E4M3 code's value (OCP 8-bit Floating Point v1.0: bias 7, a zero exponent subnormal)."""
    sign, exponent, mantissa = code >> 7, code >> 3 & 15, code & 7
    magnitude = (2.0 ** -6 * mantissa / 8 if exponent == 0
                 else 2.0 ** (exponent - 7) * (1 + mantissa / 8))
    return -magnitude if sign else magnitude


def _bfloat16(halves: np.ndarray) -> np.ndarray:
    """16-bit words read as bfloat16s, widened to float32."""
    return (halves.astype(np.uint32) << 16).view(np.float32)


def test_a_formats_decode_is_found_at_the_cheapest_price_and_reads_back_its_values():
    """The NVFP4 decode into bfloat16 pairs, its layout free, searched over shifts of at most 6 places: the
    form found costs no more than the fp4 challenge's 4 LOP3 + 1 SHF + 3 IMAD.SHL (its busiest pipe's 5
    issues, 8 instructions), is proven on every basis word (`bits.proven`), and every code of every place,
    encoded and decoded, reads back as the bfloat16 whose value times 2^126 is the code's (E2M1's own
    values)."""
    form = bits.synthesized(bits.NVFP4_BFLOAT16, reach=6)
    assert form.price <= (5, 8)
    # The form the kernels hold is proven and no dearer than the one the search finds, whichever it finds.
    bits.proven(bits.NVFP4_BFLOAT16, bits.NVFP4_BFLOAT16_FORM)
    assert bits.NVFP4_BFLOAT16_FORM.price <= form.price
    rng = np.random.default_rng(3731)
    codes = rng.integers(0, 16, size=(4096, 8))
    codes[:16, :] = np.arange(16)[:, None]
    words = (codes.astype(np.uint64) << (4 * np.arange(8, dtype=np.uint64))).sum(axis=1).astype(np.uint64)
    registers = form.decoded(form.encoded(words))
    for register, value in enumerate(registers):
        for half in range(2):
            code = codes[:, 2 * register + half]
            expected = np.where(code >= 8, -1.0, 1.0) * np.asarray(E2M1)[code & 7]
            halves = (value >> np.uint64(16 * half)) & np.uint64(0xFFFF)
            got = _bfloat16(halves).astype(np.float64) * 2.0 ** 126
            np.testing.assert_array_equal(got, expected)


def test_nvfp4_words_stored_in_the_forms_layout_decode_to_their_codes_in_order():
    """NVFP4 words packed in their natural order (code `i` at bits `4i`), stored in the checked-in form's
    layout on the host (`BitForm.encoded`) and decoded to bfloat16 elements as the GEMM's portable path
    reads them (`BitForm.values`), under a jit: every code of every place reads back in its place as its
    E2M1 value times 2^-126, the bit pattern NumPy's encoder stores."""
    form = bits.NVFP4_BFLOAT16_FORM
    rng = np.random.default_rng(3621)
    codes = rng.integers(0, 16, size=(512, 8))
    codes[:16, :] = np.arange(16)[:, None]
    words = (codes.astype(np.uint64) << (4 * np.arange(8, dtype=np.uint64))).sum(axis=1)
    stored = form.encoded(words).astype(np.uint32).reshape(64, 8)
    decoded = ahead_arm(jax.jit(lambda stored: form.values(stored, jnp.bfloat16)),
                        "NVFP4 words decoded")(stored)
    assert decoded.shape == (64, 64)
    expected = np.where(codes >= 8, -1.0, 1.0) * np.asarray(E2M1)[codes & 7]
    got = np.asarray(decoded).astype(np.float64).reshape(512, 8) * 2.0 ** 126
    np.testing.assert_array_equal(got, expected)


def test_a_decoded_value_restored_narrow_is_its_float32_restoring_rounded_once():
    """`bits.restored` in float16 and bfloat16 is its float32 restoring rounded once to the format, at every
    16-bit pattern, a NaN as a NaN: formed in float32 and cast, where a fraction field cast to float16 first
    overflowed (the half code, `0x0040`, read as infinity, not 0.5: row 8660's head review)."""
    def restorings(patterns):
        narrowed = (bits.restored(patterns, dtype) for dtype in (jnp.float16, jnp.bfloat16))
        return bits.restored(patterns), *narrowed

    narrow, *made_each = (np.asarray(x) for x in ahead_arm(jax.jit(restorings), "restorings narrowed")(
        np.arange(1 << 16, dtype=np.uint32)))
    for dtype, made in zip((jnp.float16, jnp.bfloat16), made_each, strict=True):
        numbers = ~np.isnan(narrow)
        np.testing.assert_array_equal(np.isnan(made.astype(np.float32)), ~numbers)
        np.testing.assert_array_equal(made[numbers].view(np.uint16),
                                      narrow[numbers].astype(dtype).view(np.uint16))


def test_a_decoded_value_restored_wide_is_its_float32_restoring_widened():
    """`bits.restored` in float64, a float64 truth's, is its float32 restoring widened exactly (NumPy's
    widening on the host, a float32 subnormal kept), bit for bit, at every 16-bit pattern, a NaN as a NaN:
    one definition of the decoded value at every dtype (row 8661)."""
    patterns = np.arange(1 << 16, dtype=np.uint32)
    with jax.enable_x64(new_val=True):
        narrow, wide = (np.asarray(x) for x in ahead_arm(jax.jit(
            lambda pattern: (bits.restored(pattern), bits.restored(pattern, jnp.float64))),
            "restorings widened")(patterns))
    assert wide.dtype == np.float64
    numbers = ~np.isnan(narrow)
    np.testing.assert_array_equal(np.isnan(wide), ~numbers)
    np.testing.assert_array_equal(wide[numbers].view(np.uint64),
                                  narrow[numbers].astype(np.float64).view(np.uint64))


def test_a_decoded_value_restored_narrow_is_of_the_dtype_asked_for():
    """`bits.restored` in bfloat16 is bfloat16, its every pattern's float32 restoring cast: the dtype a caller
    asks for is the one returned (row 8661's review)."""
    def asked_and_cast(pattern):
        return bits.restored(pattern, jnp.bfloat16), bits.restored(pattern).astype(jnp.bfloat16)

    narrow, cast = ahead_arm(jax.jit(asked_and_cast), "a restoring asked narrow")(
        np.arange(1 << 16, dtype=np.uint32))
    assert narrow.dtype == jnp.bfloat16
    cast = np.asarray(cast)
    numbers = ~np.isnan(cast.astype(np.float32))
    np.testing.assert_array_equal(np.isnan(np.asarray(narrow, np.float32)), ~numbers)
    np.testing.assert_array_equal(np.asarray(narrow)[numbers].view(np.uint16), cast[numbers].view(np.uint16))


def test_an_nvfp4_tile_decodes_to_its_codes_times_their_block_scales():
    """An NVFP4 tile (`nvfp4_decoded`): every code in every place of a word, under E4M3 block scales from the
    least subnormal to the largest, decodes to the code times its block's scale over `NVFP4_OWED`, exactly,
    under a jit."""
    rng = np.random.default_rng(3621)
    depth, columns = 64, 32
    codes = rng.integers(0, 16, size=(depth, columns))
    codes[:16, :16] = np.arange(16)[:, None]
    scale_bits = rng.integers(0, 0x7F, size=(depth // 16, columns), dtype=np.uint8)
    scale_bits[0, :3] = (0x01, 0x7E, 0x08)
    scales = scale_bits.view(ml_dtypes.float8_e4m3fn)
    packed = codes.reshape(depth // 8, 8, columns).astype(np.uint64) << (
        4 * np.arange(8, dtype=np.uint64))[None, :, None]
    stored = bits.NVFP4_BFLOAT16_FORM.encoded(packed.sum(axis=1)).astype(np.uint32)
    decoded = ahead_arm(jax.jit(bits.nvfp4_decoded), "an NVFP4 tile decoded")(stored, scales)
    values = np.where(codes >= 8, -1.0, 1.0) * np.asarray(E2M1)[codes & 7]
    block = np.repeat(scales.astype(np.float64), 16, axis=0)
    np.testing.assert_array_equal(np.asarray(decoded.astype(jnp.float32), np.float64) * bits.NVFP4_OWED,
                                  values * block)


def test_a_block_scales_declared_decode_is_the_formats_on_every_code():
    """The block scale's declared decode (`bits.E4M3_SCALE`, a fixed layout), its checked-in form proven
    (`bits.proven`) and no dearer than the search's, restored by its one rule (`bits.scales_decoded`), is
    the format's integer decode (`bits.decoded`) on every finite E4M3 code, bit for bit under a jit: subnormal
    scales and both zeros among them; and NaN at both NaN codes, 0x7F and 0xFF, which its linear form's
    fields read as 480 and -480."""
    bits.proven(bits.E4M3_SCALE, bits.E4M3_SCALE_FORM)
    assert bits.E4M3_SCALE_FORM.price <= bits.synthesized(bits.E4M3_SCALE).price
    declared, integer = (np.asarray(x) for x in ahead_arm(jax.jit(lambda codes: (
        bits.scales_decoded(codes), bits.decoded(codes, bits.E4M3))), "a block scale's decodes")(
        np.arange(256, dtype=np.uint32)))
    finite = (np.arange(256) & 0x7F) != 0x7F
    np.testing.assert_array_equal(declared.view(np.uint32), integer.view(np.uint32))
    # Every reader of a block scale reads it so (row 8675): an NVFP4 operand's block values
    # (`formats.NvFp4Block.decoded`), each of whose blocks codes 1 under one scale, are ml_dtypes' FP8 values,
    # which they were read as before, on every code: a block of a NaN scale NaN (the review of 3580, F1: read
    # through the fields' 480 its values were finite).
    from dew.nn.kernels.formats import NvFp4Block

    scales = np.arange(256, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn)[None]
    read = np.asarray(ahead_arm(jax.jit(lambda ones, scales: NvFp4Block(ones, scales, 1.0).decoded()),
                                "a block's values")(np.ones((1, 256 * 16), ml_dtypes.float4_e2m1fn), scales))
    read, cast = read[0, ::16], scales[0].astype(np.float32)
    np.testing.assert_array_equal(read[finite].view(np.uint32), cast[finite].view(np.uint32))
    np.testing.assert_array_equal(np.isnan(read), np.isnan(cast))


@pytest.mark.parametrize(("dtype", "minifloat"), [(jnp.float8_e4m3fn, bits.E4M3),
                                                  (jnp.float8_e5m2, bits.E5M2)],
                         ids=["e4m3fn", "e5m2"])
def test_decoded_matches_xla_on_every_fp8_code(dtype, minifloat):
    """Row 8549: `bits.decoded` matches XLA's `astype(float32)` bitwise on all 256 codes of `e4m3fn` and
    `e5m2` (including -0.0, subnormals, +-inf and NaN)."""
    ours, xla = (np.asarray(x) for x in ahead_arm(jax.jit(lambda c: (
        bits.decoded(c.astype(jnp.uint32), minifloat),
        jax.lax.bitcast_convert_type(c, dtype).astype(jnp.float32))), f"{minifloat} decoded beside XLA's")(
        np.arange(256, dtype=np.uint8)))
    np.testing.assert_array_equal(ours.view(np.uint32), xla.view(np.uint32))


def test_a_forms_version_is_its_stored_layouts():
    """A form's version (`BitForm.version`, what a packed checkpoint records) is its encoder's: forms of one
    encoder share it whatever their steps, and another encoder's differs, so words stored in another
    layout are told apart from the checked-in form's."""
    form = bits.NVFP4_BFLOAT16_FORM
    restepped = dataclasses.replace(form, steps=form.steps[:-1])
    reencoded = dataclasses.replace(form, encoder=(*form.encoder[1:], form.encoder[0]))
    assert restepped.version == form.version != reencoded.version


def test_an_fp8_decode_reads_back_every_finite_code_in_every_place():
    """Four E4M3 codes of a 32-bit word decoded into two registers of two float16s (sm80's FP8, which its
    tensor cores do not multiply), code `2i` the low half of register `i` and `2i + 1` the high, its layout
    free: the form found is proven (`bits.proven`), and every finite code in every place reads back as the
    float16 whose value times 2^8 is the code's (E4M3's bias folded into the scale; its NaN, exponent and
    mantissa all ones, no value of the decode's)."""
    fields = {field: 7 + field for field in range(7)} | {7: 15}
    placement = {8 * code + field: (code // 2, 16 * (code % 2) + bit)
                 for code in range(4) for field, bit in fields.items()}
    transform = bits.BitTransform.placed("e4m3-f16", 32, (0xBF80BF80,) * 2, placement)
    form = bits.synthesized(transform)
    codes = np.stack([np.roll(np.arange(256), 61 * place) for place in range(4)], axis=1)
    words = (codes.astype(np.uint64) << (8 * np.arange(4, dtype=np.uint64))).sum(axis=1).astype(np.uint64)
    registers = form.decoded(form.encoded(words))
    for place in range(4):
        code = codes[:, place]
        finite = (code & 0x7F) != 0x7F
        half = (registers[place // 2] >> np.uint64(16 * (place % 2))) & np.uint64(0xFFFF)
        value = half.astype(np.uint16).view(np.float16)
        expected = np.array([_e4m3(int(c)) for c in code[finite]])
        np.testing.assert_array_equal(value[finite].astype(np.float64) * 2.0 ** 8, expected)


def test_a_fixed_layout_reads_the_input_as_stored():
    """A fixed layout's 16-bit transform, each byte's high nibble moved to its low one: the form reads the
    input as it is stored (its encoder the identity), one shift and one mask on separate pipes, proven on all
    65,536 words, which it computes as the shift and mask do."""
    placement = {4 + 8 * byte + bit: (0, 8 * byte + bit) for byte in range(2) for bit in range(4)}
    transform = bits.BitTransform.placed("high-nibbles", 16, (0x0F0F,), placement, free=False)
    form = bits.synthesized(transform)
    assert form.price == (1, 2)
    words = np.arange(1 << 16, dtype=np.uint64)
    np.testing.assert_array_equal(form.encoded(words), words)
    np.testing.assert_array_equal(form.decoded(words)[0], (words >> np.uint64(4)) & np.uint64(0x0F0F))


@pytest.mark.parametrize("fault", ["mask", "shift"])
def test_a_form_wrong_on_one_bit_is_unproven(fault):
    """The proof refuses a form one bit off its transform: the NVFP4 decode's first register masked one bit
    short, or a shift one place over, each wrong on some basis word."""
    transform, form = bits.NVFP4_BFLOAT16, bits.NVFP4_BFLOAT16_FORM
    steps = list(form.steps)
    index = (form.outputs[0] - 1 if fault == "mask"
             else next(i for i, step in enumerate(steps) if step.op is not bits.BitOp.LOP3))
    step = steps[index]
    moved = step.immediate & ~(1 << 6) if fault == "mask" else step.immediate + 1
    steps[index] = step._replace(immediate=moved)
    with pytest.raises(bits.Unproven):
        bits.proven(transform, dataclasses.replace(form, steps=tuple(steps)))


def test_a_fixed_layouts_zero_field_takes_no_instruction_of_its_own():
    """A fixed layout's register whose low byte is its input's and whose high byte the consumer reads as zero:
    one LOP3 over the stored word, its mask narrowed to the byte it reads, no shift made to produce the
    zeros."""
    transform = bits.BitTransform.placed("low-byte", 16, (0xFFFF,), {bit: (0, bit) for bit in range(8)},
                                         free=False)
    form = bits.synthesized(transform)
    assert form.price == (1, 1)
    words = np.arange(1 << 16, dtype=np.uint64)
    np.testing.assert_array_equal(form.decoded(words)[0], words & np.uint64(0xFF))


def test_a_fixed_layouts_form_reads_the_input_as_stored():
    """A fixed layout's form proven on the input as it is stored: a decoder reading input bit 1 behind an
    encoder that moves bit 0 there computes the transform end to end, but not on the stored input, and is
    refused."""
    transform = bits.BitTransform.placed("bit-0", 2, (1,), {0: (0, 0)}, free=False)
    swapped = bits.BitForm((bits.BitStep(bits.BitOp.SHR, (0,), 1, bits.Unit.ALU),
                         bits.BitStep(bits.BitOp.LOP3, (1,), 1, bits.Unit.ALU)), (2,), (2, 1))
    with pytest.raises(bits.Unproven):
        bits.proven(transform, swapped)


def test_a_forms_depth_counts_a_constant_as_the_start_of_a_chain():
    """A form's depth is its dependent instructions' longest chain, and an instruction of no operand (a
    constant) starts one. The word shifted and a constant, joined, is two deep. Each minifloat's encoding
    form (`encode_form`, whose constants are such instructions) has a depth within its steps."""
    alu = bits.Unit.ALU
    form = bits.BitForm((bits.BitStep(bits.BitOp.IADD3, (), 3, alu),
                         bits.BitStep(bits.BitOp.SHR, (0,), 1, alu),
                         bits.BitStep(bits.BitOp.LOGIC, (1, 2), 0, alu, 0xFC)), (3,), (1, 2))
    assert form.depth == 2
    assert dataclasses.replace(form, outputs=(1,)).depth == 1
    for minifloat in (bits.E4M3, bits.E5M2, bits.E2M1):
        encoding = bits.encode_form(minifloat)
        assert 1 < encoding.depth <= len(encoding.steps)


def test_a_forms_live_registers_hold_each_value_until_its_last_reader_and_each_output_to_the_end():
    """A form's live registers are the most values its decode holds at once: the word and each step's value
    until the step that last reads it, whose result may take its register, an output register's to the
    end. A word, its shift and a constant all read by one join hold three before it; with the shift an
    output too, it is held past the join beside the join's result, still three at most; a chain whose every
    value is read once, by the next step, holds one."""
    alu = bits.Unit.ALU
    joined = bits.BitForm((bits.BitStep(bits.BitOp.SHR, (0,), 1, alu),
                           bits.BitStep(bits.BitOp.IADD3, (), 3, alu),
                           bits.BitStep(bits.BitOp.LOGIC, (0, 1, 2), 0, alu, 0xFC)), (3,), (1, 2))
    assert joined.live == 3
    assert dataclasses.replace(joined, outputs=(1, 3)).live == 3
    chain = bits.BitForm((bits.BitStep(bits.BitOp.SHR, (0,), 1, alu),
                          bits.BitStep(bits.BitOp.SHR, (1,), 1, alu)), (2,), (1, 2))
    assert chain.live == 1


def test_a_transform_kept_off_the_fma_pipe_takes_its_shifts_on_the_alu():
    """The NVFP4 decode declared for a consumer whose FMA pipe is its own (`units`): its form's every shift
    issues on the ALU, its price the ALU's, and it is proven."""
    transform = dataclasses.replace(bits.NVFP4_BFLOAT16, units=frozenset({bits.Unit.ALU}))
    form = bits.synthesized(transform, reach=6)
    assert form.issues[bits.Unit.FMA] == 0
    assert form.price == (len(form.steps), len(form.steps))


def _nibbles(units=frozenset(bits.Unit)) -> bits.BitTransform:
    """A fixed 16-bit layout's transform, each byte's high nibble moved to its low one, its form issued on
    `units`."""
    placement = {4 + 8 * byte + bit: (0, 8 * byte + bit) for byte in range(2) for bit in range(4)}
    return dataclasses.replace(
        bits.BitTransform.placed("high-nibbles", 16, (0x0F0F,), placement, free=False), units=units)


ALU = frozenset({bits.Unit.ALU})
FORMS = {
    "nvfp4": lambda: bits.NVFP4_BFLOAT16_FORM,
    "nvfp4-alu": lambda: bits.synthesized(dataclasses.replace(bits.NVFP4_BFLOAT16, units=ALU), reach=6),
    "nibbles": lambda: bits.synthesized(_nibbles()),
    "nibbles-alu": lambda: bits.synthesized(_nibbles(ALU)),
    "encode-e4m3": lambda: bits.encode_form(bits.E4M3),
    "encode-e5m2": lambda: bits.encode_form(bits.E5M2),
    "encode-e2m1": lambda: bits.encode_form(bits.E2M1),
    "e4m3-scale": lambda: bits.E4M3_SCALE_FORM,
}
"""Forms whose steps together take every instruction a form's PTX writes (`BitForm.ptx_registers`): a
rotation, left and right shifts on either pipe, and a mask of one value and of two values' XOR; and an
encode's (every instruction that is not linear: logic of any table, three-way adds, minima, compares,
selects and shifts by a value)."""


@pytest.mark.parametrize("name", FORMS)
def test_a_forms_traced_decode_is_its_decode(name):
    """A form traced in JAX's integer operations (`BitForm.traced`, a kernel's portable path and its
    interpreter's) gives every register of every drawn stored word as the form's own decode does, under a
    jit."""
    form = FORMS[name]()
    words = np.random.default_rng(3733).integers(0, 1 << 32, size=4096, dtype=np.uint64)
    traced = ahead_arm(jax.jit(form.traced), f"{name}'s traced decode")(words.astype(np.uint32))
    for got, expected in zip(traced, form.decoded(words), strict=True):
        np.testing.assert_array_equal(np.asarray(got).astype(np.uint64), expected)


DECODED = (64, 16)
"""The stored words a test's kernel decodes."""


def _inlined(form: bits.BitForm, generation: str):
    """A Mosaic GPU kernel decoding `DECODED` stored words (uint32) through the form's PTX, every register in
    one inline block (`inline_forms.registers`, `BitForm.ptx_registers`), in `generation`'s family's register
    layout, as a packed GEMM's mainloop inlines it."""
    from jax.experimental.pallas import mosaic_gpu as plgpu

    from dew.nn.kernels.templates import bundles, inline_forms

    layout = bundles.of(generation, jnp.bfloat16).layout

    def body(words_ref, *out_refs):
        words = plgpu.layout_cast(words_ref[...], layout)
        for out_ref, register in zip(out_refs, inline_forms.registers(form, words, layout), strict=True):
            out_ref[...] = register

    return plgpu.kernel(body, out_type=(jax.ShapeDtypeStruct(DECODED, jnp.uint32),) * len(form.outputs),
                        num_threads=1, thread_name="wg")


@pytest.mark.parametrize("name", FORMS)
@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_a_forms_ptx_compiles_for_the_gpu(generation, name):
    """A form's PTX inlined in a Mosaic GPU kernel (`inline_forms.registers`, a compiled epilogue's packing,
    `gemm.quantized.encode_tile`), lowered and compiled by XLA for a compile-only GPU of `generation`
    (`tests/kernel_compile.py`): ptxas takes every instruction PTX writes for a form (`FORMS` take them all).
    What a form computes on a GPU is held where a kernel consumes it, never a unit test on a GPU (the owner's
    rule): the NVFP4 form in the packed GEMM's mainloop, decoded by `traced` compiled (row 3733), through
    tests/test_gpu_gemm.py's NVFP4 product cells (every order, sm80 and sm90) and its NVFP4 matrix (a
    training step with the matrix decoded in the mainloop, check 1)."""
    from kernel_compile import lowered_for

    from dew.nn.kernels import generation as generations

    with generations.assumed(generation):
        lowered_for(_inlined(FORMS[name](), generation), (jax.ShapeDtypeStruct(DECODED, jnp.uint32),),
                    generation).compile()


@pytest.mark.parametrize("name", FORMS)
def test_a_kernel_inlining_a_forms_decode_counts_the_forms_issues_on_each_pipe(name):
    """A kernel that inlines a form's decode (`inline_forms.registers`) and writes every register of it
    counts what the form's PTX issues over its words, the form's own work (`BitForm.work`: its steps' issues
    on the ALU, `Work.integer_operations`, and the FMA pipe, `Work.integer_multiply_adds`), where the count
    refused the body as unpriced (row 5386): a compare's and a select's ISETP and SEL each, a three-way add
    one IADD3, a constant one (the review of train 1261, F1: setp and selp counted free, an add a PTX
    statement); traced alone, nothing compiled or run."""
    from dew.nn import programs
    from dew.nn.kernels.templates import emission
    from dew.nn.kernels.templates.cost import Stated

    form = FORMS[name]()
    traced = jax.make_jaxpr(_inlined(form, "sm90"))(jnp.zeros(DECODED, jnp.uint32)).jaxpr
    (kernel,) = [equation for equation in programs.equations(traced)
                 if equation.primitive in programs.NATIVE_KERNELS]
    counted = emission.launch_work(kernel, Stated(steps=0, bytes_accessed=0))
    stated = form.work(DECODED[0] * DECODED[1])
    assert (counted.integer_operations, counted.integer_multiply_adds) == (
        stated.integer_operations, stated.integer_multiply_adds), (counted, stated)


MINIFLOATS = {"e4m3": (bits.E4M3, ml_dtypes.float8_e4m3fn), "e5m2": (bits.E5M2, ml_dtypes.float8_e5m2),
              "e2m1": (bits.E2M1, ml_dtypes.float4_e2m1fn)}
"""Each encode's minifloat and ml_dtypes' type of it, the reference's."""

SLICES = 64
"""All 2^32 fp32 words, cut into this many slices of 2^26 a map task's unit."""


def reference_codes(words: np.ndarray, minifloat: bits.Minifloat, dtype) -> np.ndarray:
    """ml_dtypes' round-to-nearest-even cast of fp32 `words` (uint32 bits) saturated first to the largest
    finite value (the satfinite cvt's semantics), as uint32 codes; a NaN the minifloat's NaN code (whose
    payload ml_dtypes chooses otherwise: e5m2's quiet 0x7E) with its sign, e2m1's its largest."""
    x = words.view(np.float32)
    largest = np.uint32(minifloat.largest_float32).view(np.float32)
    codes = np.clip(x, -largest, largest).astype(dtype).view(np.uint8).astype(np.uint32)
    codes &= np.uint32((1 << minifloat.width) - 1)
    nan = minifloat.largest if minifloat.nan is None else minifloat.nan
    signed = np.uint32(nan) | (words >> np.uint32(31)) << np.uint32(minifloat.width - 1)
    return np.where(np.isnan(x), signed, codes)


@functools.cache
def _encodes(name: str):
    """A slice of fp32 words (`SLICES`) through `name`'s encode and its instruction form as a kernel traces
    them (`bits.encoded`, `BitForm.traced`), one compiled program: each word's code, whether the form
    differs from the encode on any, and the first word it does on."""
    minifloat, _ = MINIFLOATS[name]
    form = bits.encode_form(minifloat)

    @jax.jit
    def run(words):
        codes = bits.encoded(words, minifloat)
        (formed,) = form.traced(words)
        apart = formed != codes
        return codes.astype(jnp.uint8), jnp.any(apart), words[jnp.argmax(apart)]

    return run


@pytest.mark.shards(units=192, per_task=32)
def test_each_encode_is_the_reference_cast_on_every_fp32_word():
    """Each of e4m3's, e5m2's and e2m1's encodes (`bits.encoded`) is ml_dtypes' round-to-nearest-even cast
    saturated to its largest finite value on every one of the 2^32 fp32 words, a NaN as its NaN code, and its
    instruction form (`bits.encode_form`) is the encode on every word: a proof by exhaustion (row 4050), of
    the encode and the form as a kernel traces them, each slice of words one compiled program's."""
    cases = list(itertools.product(MINIFLOATS, range(SLICES)))
    assert len(cases) == 192, "the shards mark counts each slice of each encode"

    def checked(start: int, stop: int) -> int:
        for name, index in cases[start:stop]:
            minifloat, dtype = MINIFLOATS[name]
            words = np.arange(index << 26, (index + 1) << 26, dtype=np.uint64).astype(np.uint32)
            codes, apart, first = _encodes(name)(words)
            assert not apart, (name, "form", hex(int(first)))
            got = np.asarray(codes).astype(np.uint32)
            differ = np.flatnonzero(got != reference_codes(words, minifloat, dtype))
            assert differ.size == 0, (name, [(hex(int(words[i])), int(got[i])) for i in differ[:4]])
        return stop - start

    sharded("every fp32 word's encode", checked, len(cases))


@pytest.mark.parametrize("e2m1", [bits.E2M1, bits.Minifloat(*bits.E2M1)], ids=["the format", "an equal one"])
def test_an_nvfp4_tile_packed_from_its_encode_decodes_to_its_rounded_values(e2m1):
    """fp32 values encoded to e2m1 (`bits.encoded`) and packed eight a word (`bits.packed`), with a block
    scale of 1 (an e4m3 scale of 2^8, `NVFP4_OWED`), decode (`nvfp4_decoded`) to their e2m1 rounding: the
    pack is the stored NVFP4 order the decode reads, in jax.numpy as a kernel's portable path runs it,
    whichever object states the format: one built equal to `bits.E2M1` packs as it does, where it packed
    its codes in their natural order, which the decode read as other values."""
    rng = np.random.default_rng(4050)
    x = rng.standard_normal((64, 8)).astype(np.float32) * 3

    def decoded(x, scales):
        codes = bits.encoded(x.view(jnp.uint32), e2m1)
        return bits.nvfp4_decoded(bits.packed(codes.T, e2m1).T, scales)

    scales = np.full((4, 8), 2.0 ** 8, ml_dtypes.float8_e4m3fn)
    found = [run(x, scales) for run in ran_here(decoded, f"an NVFP4 tile packed from {e2m1}")]
    rounded = np.clip(x, -6, 6).astype(ml_dtypes.float4_e2m1fn).astype(np.float32)
    for each in found:
        np.testing.assert_array_equal(np.asarray(each, np.float32), rounded)


def _spread_tile() -> np.ndarray:
    """A tile whose values span six decades, a block of zeros among them."""
    rng = np.random.default_rng(4050)
    x = (rng.standard_normal((64, 8)) * 10.0 ** rng.integers(-3, 3, (64, 8))).astype(np.float32)
    x[16:32, 3] = 0
    return x


def _quantized_values() -> np.ndarray:
    """An NVFP4 tile's values (each code times its block's scale and the tensor scale, rounded to fp32):
    quotients on and beside e2m1's midpoints and e4m3's values, where an inexact division lands across (XLA's
    took 186 of its 1024 scales a step and 2760 of its codes across a midpoint)."""
    # Drawn on the host: a GPU tier part's prepare computes nothing on its compile-only GPU (row 8617).
    drawn = drawn_on_the_host(lambda seed: np.asarray(spread(seed, (256, 64), 16), np.float32), 0)
    words, scales, tensor = ahead_arm(jax.jit(bits.nvfp4_quantized), "an NVFP4 tile's values, quantized")(
        drawn)
    owed = np.float32(bits.NVFP4_OWED) * np.asarray(tensor)
    return np.asarray(ahead_arm(jax.jit(bits.nvfp4_decoded), "an NVFP4 tile's values, decoded")(
        np.asarray(words), np.asarray(scales)), np.float32) * owed


@pytest.mark.parametrize("tile", [_spread_tile, _quantized_values], ids=["spread", "quantized values"])
def test_an_nvfp4_quantized_tile_takes_each_blocks_least_scale_and_decodes_to_its_rounding(tile):
    """An fp32 tile quantized to NVFP4 (`bits.nvfp4_quantized`): its tensor scale its largest magnitude over
    6 * 448, each block's scale the least e4m3 at or above the block's largest over it over 6 (at most
    e4m3's largest), and its decode (`nvfp4_decoded`, times `NVFP4_OWED`) each value over the tensor scale
    over its block's scale rounded to e2m1 times its block's scale, exactly, every quotient IEEE fp32
    division's as NumPy divides (row 4050; row 5456: an NVFP4 tile's own values, whose quotients land on
    the formats' boundaries, took codes and scales across them by XLA's inexact division)."""
    x = tile()

    def quantized(x):
        words, scales, tensor = bits.nvfp4_quantized(x)
        return scales, tensor, bits.nvfp4_decoded(words, scales)

    # Every way run before any is judged (`ran_here`).
    found = [run(x) for run in ran_here(quantized, f"a tile quantized to NVFP4, {tile.__name__}")]
    for scales, tensor, decoded in found:
        _holds_its_least_scales_and_rounding(x, scales, tensor, decoded)


def _holds_its_least_scales_and_rounding(x, scales, tensor, decoded):
    """`x` quantized to NVFP4 (its block `scales`, its `tensor` scale and its `decoded` codes) as
    test_an_nvfp4_quantized_tile_takes_each_blocks_least_scale_and_decodes_to_its_rounding states it."""
    largest = np.float32(np.abs(x).max())
    np.testing.assert_array_equal(np.asarray(tensor), largest / np.float32(6 * 448))
    table = np.arange(0x7F, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    blocks = np.abs(x).reshape(x.shape[0] // 16, 16, x.shape[1]).max(axis=1)
    ratio = blocks / np.asarray(tensor) / np.float32(6)
    # The largest block's ratio is 448 up to its rounding, which may pass it: its scale is e4m3's largest.
    least = table[np.minimum(np.searchsorted(table, ratio), table.size - 1)]
    np.testing.assert_array_equal(np.asarray(scales, np.float32), least)
    each = np.repeat(least, 16, axis=0)
    quotient = np.where(each > 0, x / np.asarray(tensor) / np.where(each > 0, each, 1), 0).astype(np.float32)
    rounded = np.clip(quotient, -6, 6).astype(ml_dtypes.float4_e2m1fn).astype(np.float32)
    decoded = np.asarray(decoded, np.float32) * np.float32(bits.NVFP4_OWED)
    np.testing.assert_array_equal(decoded, rounded * np.repeat(least, 16, axis=0))


def test_a_small_nonzero_tile_keeps_its_nvfp4_codes():
    """A tile of small normal values, whose largest over 6 * 448 is no normal fp32, quantizes to codes that
    decode to the values' rounding (`bits.nvfp4_quantized`, then `nvfp4_decoded` times `NVFP4_OWED` and the
    tensor scale): nonzero, each within e2m1's half step of its value, not a tile erased by its scale's
    underflow."""
    x = np.full((32, 8), np.float32(3e-36))
    x[::2] *= np.float32(-0.5)

    def decoded(x):
        words, scales, tensor = bits.nvfp4_quantized(x)
        return bits.nvfp4_decoded(words, scales), tensor

    found = [run(x) for run in ran_here(decoded, "a small nonzero tile quantized to NVFP4")]
    for codes, tensor in found:
        values = np.asarray(codes, np.float64) * bits.NVFP4_OWED * np.float64(np.asarray(tensor))
        assert np.all(np.sign(values) == np.sign(x))


def test_nvfp4_does_not_classify_a_subnormal_tile_as_zero():
    x = np.zeros((16, 2), np.float32)
    x[:2, 0] = [2.0 ** -130, -(2.0 ** -132)]
    codes, scales, tensor = jax.jit(bits.nvfp4_coded)(x)
    # 2^-130 / 2^-126 / 6 = 1/96, whose e4m3 ceiling is 6/512; the other block is empty.
    expected = np.zeros_like(x, dtype=np.uint32)
    expected[:2, 0] = [7, 11]  # e2m1's +6 and -1.5
    np.testing.assert_array_equal(codes, expected)
    np.testing.assert_array_equal(np.asarray(scales).view(np.uint8), [[6, 0]])
    np.testing.assert_array_equal(tensor, np.float32(2.0 ** -126))


def test_a_one_operand_lop3_reads_its_immediate_as_both_other_operands():
    """A truth table over one value and the immediate (`BitOp.LOGIC` with one operand) runs as PTX writes it
    and the synthesizer proves it, `lop3.b32 d, a, imm, imm`, under every table (row 4647): each result
    bit the table's bit at the minterm of the value's bit and the immediate's twice, the PTX ISA's
    definition."""
    words = np.random.default_rng(4647).integers(0, 1 << 32, 4096, dtype=np.uint64).astype(np.uint32)
    immediate = 0x0F0F_A5C3
    lanes = np.arange(32, dtype=np.uint32)
    held = (words[:, None] >> lanes) & 1
    other = (np.uint32(immediate) >> lanes) & 1
    for table in range(256):
        form = bits.BitForm((bits.BitStep(bits.BitOp.LOGIC, (0,), immediate, bits.Unit.ALU, table),), (1,),
                            tuple(1 << bit for bit in range(32)))
        minterms = (held << 2) | (other << 1) | other
        expected = np.sum(((np.uint32(table) >> minterms) & 1) << lanes, axis=-1, dtype=np.uint64)
        np.testing.assert_array_equal(form.decoded(words)[0], expected, err_msg=f"table {table:#04x}")


def seeded_as_computed() -> None:
    """`tools/superoptimize_bits.seed` against its references, in the tool's own environment (run by
    test_the_bit_tracer_seeds_a_booleans_not_and_refuses_an_arithmetic_shift_by_a_value): a boolean's NOT,
    held as 0 or 1, is its other value on every word, and a word's conversion to a boolean whether it is not
    0; an arithmetic shift by a value, and a conversion through a byte, are refused."""
    from tools import superoptimize_bits

    words = np.array([0, 1, 5, 0x80000000, 0xFFFFFFFF], np.uint32)

    def nonzero(words):
        return jnp.logical_not(words == 0).astype(jnp.uint32)

    def shifted_by_its_low_bits(words):
        return (words.astype(jnp.int32) >> (words & 7).astype(jnp.int32)).astype(jnp.uint32)

    def held(words):
        return words.astype(jnp.bool_).astype(jnp.uint32)

    def through_a_byte(words):
        return words.astype(jnp.uint8).astype(jnp.uint32)

    for reference in (nonzero, held):
        [seeded] = superoptimize_bits.seed(reference).decoded(words)
        np.testing.assert_array_equal(seeded, np.asarray(reference(jnp.asarray(words))))
    for refused in (shifted_by_its_low_bits, through_a_byte):
        with pytest.raises(ValueError):
            superoptimize_bits.seed(refused)


def test_the_bit_tracer_seeds_a_booleans_not_and_refuses_an_arithmetic_shift_by_a_value():
    """The nonlinear search's tracer (`tools/superoptimize_bits.seed`, §7.3), run in a fresh interpreter under
    its own PEP 723 environment (its Z3, no dependency of Dew's, layered by `uv run --with`): a boolean's NOT
    seeds its other value (it seeded the word's complement, 0 and 1 to 0xffffffff and 0xfffffffe), a word
    converted to a boolean whether it is not 0, and an arithmetic shift by a value and a conversion through a
    byte, which no step of the vocabulary states, are refused (each seeded another function: a left shift,
    the word itself)."""
    from release_support import ROOT, under_release

    from tools import check_tool_types

    [(_, [requirement])] = check_tool_types.environments_of(ROOT / "tools" / "superoptimize_bits.py", {})
    under_release(requirement, seeded_as_computed)
