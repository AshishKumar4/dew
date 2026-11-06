"""Bit-level forms (docs/design/algebra.md §7, theme T82): a format's decode declared as the placement of its
codes' fields, its cheapest instruction form found by the price-ordered search and proven on every word, and
the decoded registers read as the consumer reads them, values its format defines."""

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import pytest
from map_judge import sharded

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
    assert form == bits.NVFP4_BFLOAT16_FORM
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
    layout as a prepare step stores them on the device (`BitForm.prepared`) and decoded to bfloat16 elements
    as the GEMM's portable path reads them (`BitForm.values`), under a jit: every code of every place reads
    back in its place as its E2M1 value times 2^-126, the bit pattern NumPy's encoder stores."""
    form = bits.NVFP4_BFLOAT16_FORM
    rng = np.random.default_rng(3621)
    codes = rng.integers(0, 16, size=(512, 8))
    codes[:16, :] = np.arange(16)[:, None]
    words = (codes.astype(np.uint64) << (4 * np.arange(8, dtype=np.uint64))).sum(axis=1)
    natural = jnp.asarray(words.astype(np.uint32).reshape(64, 8))
    stored = jax.jit(form.prepared)(natural)
    np.testing.assert_array_equal(np.asarray(stored).astype(np.uint64).ravel(), form.encoded(words))
    decoded = jax.jit(lambda stored: form.values(stored, jnp.bfloat16))(stored)
    assert decoded.shape == (64, 64)
    expected = np.where(codes >= 8, -1.0, 1.0) * np.asarray(E2M1)[codes & 7]
    got = np.asarray(decoded).astype(np.float64).reshape(512, 8) * 2.0 ** 126
    np.testing.assert_array_equal(got, expected)


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
    scales = jnp.asarray(scale_bits).view(jnp.float8_e4m3fn)
    packed = codes.reshape(depth // 8, 8, columns).astype(np.uint64) << (
        4 * np.arange(8, dtype=np.uint64))[None, :, None]
    natural = jnp.asarray(packed.sum(axis=1).astype(np.uint32))
    stored = bits.NVFP4_BFLOAT16_FORM.prepared(natural)
    decoded = jax.jit(bits.nvfp4_decoded)(stored, scales)
    values = np.where(codes >= 8, -1.0, 1.0) * np.asarray(E2M1)[codes & 7]
    block = np.repeat(np.asarray(scales.astype(jnp.float32), np.float64), 16, axis=0)
    np.testing.assert_array_equal(np.asarray(decoded.astype(jnp.float32), np.float64) * bits.NVFP4_OWED,
                                  values * block)


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
}
"""Forms whose steps together take every instruction a form's PTX writes (`BitForm.ptx`): a rotation, left
and right shifts on either pipe, and a mask of one value and of two values' XOR; and an encode's (every
instruction that is not linear: logic of any table, three-way adds, minima, compares, selects and shifts
by a value)."""


@pytest.mark.parametrize("name", FORMS)
def test_a_forms_traced_decode_is_its_decode(name):
    """A form traced in JAX's integer operations (`BitForm.traced`, a kernel's portable path and its
    interpreter's) gives every register of every drawn stored word as the form's own decode does, under a
    jit."""
    form = FORMS[name]()
    words = np.random.default_rng(3733).integers(0, 1 << 32, size=4096, dtype=np.uint64)
    traced = jax.jit(form.traced)(jnp.asarray(words.astype(np.uint32)))
    for got, expected in zip(traced, form.decoded(words), strict=True):
        np.testing.assert_array_equal(np.asarray(got).astype(np.uint64), expected)


def _inline_decode(form: bits.BitForm):
    """A Triton kernel decoding a block of stored words (uint32) through the form's PTX (`BitForm.ptx`), a
    register's block apiece, its words int32 inside, as Triton's inline assembly types a 32-bit register (a
    signless integer)."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    count = len(form.outputs)

    def body(words_ref, *out_refs):
        shape = jax.ShapeDtypeStruct(words_ref.shape, jnp.int32)
        words = words_ref[...]
        for output, ref in enumerate(out_refs):
            ref[...] = plgpu.elementwise_inline_asm(form.ptx(output), args=(words,), constraints="=r,r",
                                                    pack=1, result_shape_dtypes=(shape,))[0]

    def decode(words):
        shape = jax.ShapeDtypeStruct(words.shape, jnp.int32)
        signed = jax.lax.bitcast_convert_type(words, jnp.int32)
        return tuple(jax.lax.bitcast_convert_type(value, jnp.uint32)
                     for value in pl.pallas_call(body, out_shape=(shape,) * count)(signed))

    return decode


@pytest.mark.parametrize("name", FORMS)
@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_a_forms_ptx_compiles_for_the_gpu(generation, name):
    """A form's PTX in a Triton kernel's inline assembly, lowered and compiled by XLA for a compile-only GPU
    of `generation` (`tests/kernel_compile.py`): ptxas takes every instruction PTX writes for a form
    (`FORMS` take them all)."""
    from kernel_compile import lowered_for

    words = jax.ShapeDtypeStruct((1024,), jnp.uint32)
    lowered_for(_inline_decode(FORMS[name]()), (words,), generation).compile()


@pytest.mark.gpu(reason="runs a form's PTX on this host's GPU")
@pytest.mark.parametrize("name", FORMS)
def test_a_forms_ptx_decodes_on_the_gpu_as_its_form_does(name):
    """A form's PTX run on the GPU (`BitForm.ptx` in a Triton kernel's inline assembly) gives every register
    of every drawn stored word as the form's own decode does, each instruction PTX writes among them."""
    form = FORMS[name]()
    words = np.random.default_rng(3734).integers(0, 1 << 32, size=1 << 16, dtype=np.uint64)
    decoded = jax.jit(_inline_decode(form))(jnp.asarray(words.astype(np.uint32)))
    for got, expected in zip(decoded, form.decoded(words), strict=True):
        np.testing.assert_array_equal(np.asarray(got).astype(np.uint64), expected)


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


@pytest.mark.shards(units=192, per_task=8)
def test_each_encode_is_the_reference_cast_on_every_fp32_word():
    """Each of e4m3's, e5m2's and e2m1's encodes (`bits.encoded`) is ml_dtypes' round-to-nearest-even cast
    saturated to its largest finite value on every one of the 2^32 fp32 words, a NaN as its NaN code, and its
    instruction form (`bits.encode_form`) is the encode on every word: a proof by exhaustion (row 4050)."""
    cases = list(itertools.product(MINIFLOATS, range(SLICES)))
    assert len(cases) == 192, "the shards mark counts each slice of each encode"

    def checked(start: int, stop: int) -> int:
        for name, index in cases[start:stop]:
            minifloat, dtype = MINIFLOATS[name]
            words = np.arange(index << 26, (index + 1) << 26, dtype=np.uint64).astype(np.uint32)
            got = bits.encoded(words, minifloat, np)
            want = reference_codes(words, minifloat, dtype)
            differ = np.flatnonzero(got != want)
            assert differ.size == 0, (name, [(hex(int(words[i])), int(got[i]), int(want[i]))
                                             for i in differ[:4]])
            (formed,) = bits.encode_form(minifloat).decoded(words)
            differ = np.flatnonzero(formed != got)
            assert differ.size == 0, (name, "form", [(hex(int(words[i])), int(formed[i]), int(got[i]))
                                                     for i in differ[:4]])
        return stop - start

    sharded("every fp32 word's encode", checked, len(cases))


def test_an_nvfp4_tile_packed_from_its_encode_decodes_to_its_rounded_values():
    """fp32 values encoded to e2m1 (`bits.encoded`) and packed eight a word (`bits.packed`), with a block
    scale of 1 (an e4m3 scale of 2^8, `NVFP4_OWED`), decode (`nvfp4_decoded`) to their e2m1 rounding: the
    pack is the stored NVFP4 order the decode reads, in jax.numpy as a kernel's portable path runs it."""
    rng = np.random.default_rng(4050)
    x = rng.standard_normal((64, 8)).astype(np.float32) * 3
    codes = bits.encoded(jnp.asarray(x).view(jnp.uint32), bits.E2M1)
    words = bits.packed(codes.T, bits.E2M1).T
    scales = jnp.full((4, 8), 2.0 ** 8, ml_dtypes.float8_e4m3fn)
    decoded = np.asarray(bits.nvfp4_decoded(words, scales), np.float32)
    rounded = np.clip(x, -6, 6).astype(ml_dtypes.float4_e2m1fn).astype(np.float32)
    np.testing.assert_array_equal(decoded, rounded)


def test_an_nvfp4_quantized_tile_takes_each_blocks_least_scale_and_decodes_to_its_rounding():
    """An fp32 tile quantized to NVFP4 (`bits.nvfp4_quantized`), a block of zeros and a block whose values
    span six decades among them: its tensor scale its largest magnitude over 6 * 448, each block's scale the
    least e4m3 at or above the block's largest over 6 times it (at most e4m3's largest), and its decode
    (`nvfp4_decoded`, times `NVFP4_OWED`) each value over its scales rounded to e2m1 times its block's
    scale, exactly (row 4050)."""
    rng = np.random.default_rng(4050)
    x = (rng.standard_normal((64, 8)) * 10.0 ** rng.integers(-3, 3, (64, 8))).astype(np.float32)
    x[16:32, 3] = 0
    words, scales, tensor = bits.nvfp4_quantized(jnp.asarray(x))
    largest = np.float32(np.abs(x).max())
    np.testing.assert_array_equal(np.asarray(tensor), largest / np.float32(6 * 448))
    table = np.arange(0x7F, dtype=np.uint8).view(ml_dtypes.float8_e4m3fn).astype(np.float32)
    ratio = np.abs(x).reshape(4, 16, 8).max(axis=1) / (np.float32(6) * np.asarray(tensor))
    # The largest block's ratio is 448 up to its rounding, which may pass it: its scale is e4m3's largest.
    least = table[np.minimum(np.searchsorted(table, ratio), table.size - 1)]
    np.testing.assert_array_equal(np.asarray(scales, np.float32), least)
    each = np.repeat(least, 16, axis=0) * np.asarray(tensor)
    quotient = np.where(each > 0, x / np.where(each > 0, each, 1), 0).astype(np.float32)
    rounded = np.clip(quotient, -6, 6).astype(ml_dtypes.float4_e2m1fn).astype(np.float32)
    decoded = np.asarray(bits.nvfp4_decoded(words, scales), np.float32) * np.float32(bits.NVFP4_OWED)
    np.testing.assert_array_equal(decoded, rounded * np.repeat(least, 16, axis=0))
