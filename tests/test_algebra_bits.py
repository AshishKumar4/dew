"""Bit-level forms (docs/design/algebra.md §7, theme T82): a format's decode declared as the placement of its
codes' fields, its cheapest instruction form found by the price-ordered search and proven on every word, and
the decoded registers read as the consumer reads them, values its format defines."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def _nvfp4():
    """Eight E2M1 codes of a 32-bit word (code `i` at bits `4i` to `4i + 3`: mantissa, the exponent's two
    bits, sign) decoded into four registers of two bfloat16s, code `2i` the low half of register `i` and code
    `2i + 1` the high: its mantissa at bit 6, its exponent at 7 and 8, its sign at 15, so the bfloat16 times
    2^126 is the code's value (the exponent's bias, folded into the block scale's multiply)."""
    fields = {0: 6, 1: 7, 2: 8, 3: 15}
    placement = {4 * code + field: (code // 2, 16 * (code % 2) + bit)
                 for code in range(8) for field, bit in fields.items()}
    return bits.BitTransform.placed("nvfp4-bf16", 32, (0x81C081C0,) * 4, placement)


@functools.cache
def _nvfp4_form() -> bits.BitForm:
    """The NVFP4 decode's form over shifts of at most 6 places, found once for the tests that read it."""
    return bits.synthesized(_nvfp4(), reach=6)


def _bfloat16(halves: np.ndarray) -> np.ndarray:
    """16-bit words read as bfloat16s, widened to float32."""
    return (halves.astype(np.uint32) << 16).view(np.float32)


def test_a_formats_decode_is_found_at_the_cheapest_price_and_reads_back_its_values():
    """The NVFP4 decode into bfloat16 pairs, its layout free, searched over shifts of at most 6 places: the
    form found costs no more than the fp4 challenge's 4 LOP3 + 1 SHF + 3 IMAD.SHL (its busiest pipe's 5
    issues, 8 instructions), is proven on every basis word (`bits.proven`), and every code of every place,
    encoded and decoded, reads back as the bfloat16 whose value times 2^126 is the code's (E2M1's own
    values)."""
    form = _nvfp4_form()
    assert form.price <= (5, 8)
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
    transform, form = _nvfp4(), _nvfp4_form()
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
    transform = dataclasses.replace(_nvfp4(), units=frozenset({bits.Unit.ALU}))
    form = bits.synthesized(transform, reach=6)
    assert form.issues[bits.Unit.FMA] == 0
    assert form.price == (len(form.steps), len(form.steps))


def test_a_forms_traced_decode_is_its_decode():
    """The NVFP4 decode's form traced in JAX's integer operations (`BitForm.traced`, a kernel's portable
    path and its interpreter's) gives every register of every drawn stored word as the form's own decode
    does, under a jit."""
    form = _nvfp4_form()
    words = np.random.default_rng(3733).integers(0, 1 << 32, size=4096, dtype=np.uint64)
    traced = jax.jit(form.traced)(jnp.asarray(words.astype(np.uint32)))
    for got, expected in zip(traced, form.decoded(words), strict=True):
        np.testing.assert_array_equal(np.asarray(got).astype(np.uint64), expected)


def _inline_decode(form: bits.BitForm):
    """A Triton kernel decoding a block of stored words through the form's PTX (`BitForm.ptx`), a register's
    block apiece."""
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu

    count = len(form.outputs)

    def body(words_ref, *out_refs):
        shape = jax.ShapeDtypeStruct(words_ref.shape, jnp.uint32)
        words = words_ref[...]
        for output, ref in enumerate(out_refs):
            ref[...] = plgpu.elementwise_inline_asm(form.ptx(output), args=(words,), constraints="=r,r",
                                                    pack=1, result_shape_dtypes=(shape,))[0]

    def decode(words):
        shape = jax.ShapeDtypeStruct(words.shape, jnp.uint32)
        return pl.pallas_call(body, out_shape=(shape,) * count)(words)

    return decode


@pytest.mark.parametrize("generation", ["sm80", "sm90"])
def test_a_forms_ptx_compiles_for_the_gpu(generation):
    """The NVFP4 decode's PTX in a Triton kernel's inline assembly, lowered and compiled by XLA for a
    compile-only GPU of `generation` (`tests/kernel_compile.py`): ptxas takes every instruction of it, each
    shift on its pipe's (SHF, or IMAD by a power of two) and each register a LOP3."""
    from kernel_compile import lowered_for

    words = jax.ShapeDtypeStruct((1024,), jnp.uint32)
    lowered_for(_inline_decode(_nvfp4_form()), (words,), generation).compile()


@pytest.mark.gpu(reason="runs a form's PTX on this host's GPU")
def test_a_forms_ptx_decodes_on_the_gpu_as_its_form_does():
    """The NVFP4 decode's PTX run on the GPU (`BitForm.ptx` in a Triton kernel's inline assembly) gives
    every register of every drawn stored word as the form's own decode does."""
    form = _nvfp4_form()
    words = np.random.default_rng(3734).integers(0, 1 << 32, size=1 << 16, dtype=np.uint64)
    decoded = jax.jit(_inline_decode(form))(jnp.asarray(words.astype(np.uint32)))
    for got, expected in zip(decoded, form.decoded(words), strict=True):
        np.testing.assert_array_equal(np.asarray(got).astype(np.uint64), expected)
