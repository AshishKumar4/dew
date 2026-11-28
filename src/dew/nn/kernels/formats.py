"""The low-precision operand formats the GEMM template multiplies (row 3621, 4519): an NVFP4 matrix as Dew
stores it (`NvFp4`, its E2M1 codes in the staged order the mainloop decodes, its block and tensor scales), an
MXFP8 operand (`MxFp8`, E4M3 codes under E8M0 scales every 32 of its depth) and an NVFP4 operand of the
block-scaled products (`NvFp4Block`): their codecs and values alone. Their products and selectors are the
GEMM's (`gemm`), their recipes' projections a layer's (`projections`)."""

import dataclasses
import functools
import hashlib
import math
from collections.abc import Sequence
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero
from jax.typing import DTypeLike

from dew.algebra import bits
from dew.nn.derivatives import Protected, contracted, declared_tangent
from dew.nn.float_bits import exact_power2, quiet_nan, raw_fields, rounded_binary
from dew.nn.kernels import generation
from dew.nn.kernels.templates import product as formats_of
from dew.nn.kernels.templates.gemm import packed
from dew.nn.precision import at_least_fp32, power2

NVFP4_STAGE = packed.NVFP4_STEP
"""The depths a stage of the mainloop's NVFP4 decode reads: two groups, so each column's scales are a
16-byte row (a TMA copy's least)."""


def _staged_rows(depth: int, group: int, code_width: int, depth_fn, label: str) -> np.ndarray:
    """Shared staged-depth permutation for packed weight matrices (`staged_rows`, `ternary_staged_rows`)."""
    if depth % group:
        raise ValueError(f"a {label} matrix's depth is whole {group}-deep groups, not {depth}")
    groups, within = np.divmod(np.arange(depth), group)
    word, code = np.divmod(within, 32 // code_width)
    return depth_fn(groups, word, code)


def staged_rows(depth: int) -> np.ndarray:
    """The depth each code of the staged order holds (`NvFp4`): packed row `8w + j` of a group (word `w`'s
    code `j`) holds the depth `packed.nvfp4_staged_depth` gives it, so a stage's word rows decode, code by
    code, into its sixteen-row depth slices, the block of one scale each (the mainloop's stage decode)."""
    return _staged_rows(depth, packed.NVFP4_GROUP, bits.E2M1.width, packed.nvfp4_staged_depth, "NVFP4")

def form_version(encoder: str, order: np.ndarray) -> bytes:
    """The 8-byte digest of a packed encoding: its bit-form `encoder` (`bits.BitForm.version`) and the
    staged depth `order` (`staged_rows(NVFP4_STAGE)`). Changing either invalidates stored words."""
    return hashlib.sha256(bytes.fromhex(encoder) + np.asarray(order, "<i4").tobytes()).digest()[:8]


FORM = form_version(bits.NVFP4_BFLOAT16_FORM.version, staged_rows(NVFP4_STAGE))
"""The version of the form an `NvFp4`'s words are packed by (`form_version`: its encoder's digest and its
staged depth order, `staged_rows(NVFP4_STAGE)`), as the bytes a matrix states beside its words
(`NvFp4.form`): words packed by another encoder or staged order decode wrong by this Dew's form, so a load
refuses them (`NvFp4.checked`; algebra.md 7.4, rows 6101, 8662)."""


def _checked_form(found: set[bytes], expected: bytes, label: str, factory: str) -> None:
    """Refuse a packed matrix whose words were packed by a form (`found`, each digest it states) other than
    this Dew's (`expected`), rows 8958 and 13700."""
    if found != {expected}:
        raise ValueError(f"{label} words packed by form version {', '.join(sorted(f.hex() for f in found))}, "
                         f"where this Dew decodes form version {expected.hex()}: repack them from the source "
                         f"weights (`{factory}`)")


def _digests(stored, size: int) -> set[bytes]:
    """The digests of a stored `form` leaf this process holds (its shards' rows; a stack's one a layer),
    each a row of `size` bytes, none where it holds no shard; refused where a shard holds no whole row."""
    pieces = ([stored] if not isinstance(stored, jax.Array)
              else jax.device_get([shard.data for shard in stored.addressable_shards]))
    found: set[bytes] = set()
    for piece in pieces:
        held = np.asarray(piece)
        if held.ndim == 0 or held.shape[-1] != size or held.size == 0:
            raise ValueError(f"a form digest of {size} bytes stored as {held.shape}")
        found.update(bytes(row) for row in held.reshape(-1, size))
    return found


def _form_field(default: bytes):
    """A packed matrix's `form` field: its encoder's digest, immutable bytes held in its type's tree
    structure, so a program it enters reads it concrete and its product checks it wherever it runs, traced
    or not (`checked`; a transformation keeps it unless code replaces it, row 13700). A checkpoint holds it
    as a leaf of the matrix's record (`stored`)."""
    return dataclasses.field(default=default, metadata={"static": True})


class _Packed:
    """The form check a packed matrix's types share (`NvFp4`, `TernaryWeights`, `Bf16Window`): `FORM_OF`,
    this Dew's digest of its encoding, and `FACTORY`, what packs its words."""

    FORM_OF: ClassVar[bytes]
    FACTORY: ClassVar[str]
    form: bytes

    def checked(self):
        """Itself where its words are this Dew's form's (`FORM_OF`), its form a static field's, so under a
        transformation too; refused otherwise: words another encoder packed read wrong through this form's
        decode, so they are repacked from the source weights, never decoded."""
        _checked_form({self.form}, self.FORM_OF, type(self).__name__, self.FACTORY)
        return self

    def admitted(self):
        """Itself, checked (`checked`): the one entry every product reading its words passes."""
        return self.checked()


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NvFp4(_Packed):
    """An NVFP4 matrix `[depth, width]` as the GEMM stores it (row 3621), K-major, each column's depths
    along its row (as a Dense kernel read transposed): its E2M1 codes eight a uint32 in the GEMM's staged
    depth order (`words`, `[width, depth / 8]`: `staged_rows`, each word the `bits.NVFP4_BFLOAT16_FORM` word
    of its eight codes, `_staged_words`), each sixteen consecutive depths' E4M3
    scale (`scales`, `[width, depth / 16]`, in the depths' own order), and the tensor's fp32 scale
    (`scale`, a scalar): each value its code times its block's scale times the tensor's. Its depth is whole
    stages (`NVFP4_STAGE`): the mainloop's stage decode reads each column's words and scales as rows a TMA
    copy takes, at least 16 bytes. A bank of experts' matrices (`[groups, depth, width]`, row 4516) leads
    each field with its group axis: `words` `[groups, width, depth / 8]`, `scales` `[groups, width,
    depth / 16]`, `scale` `[groups]`, each expert's matrix its own tensor scale. Its fields are NumPy
    arrays where it was packed on the host (`of`)."""

    words: jax.Array | np.ndarray
    scales: jax.Array | np.ndarray
    scale: jax.Array | np.ndarray
    form: bytes = _form_field(FORM)
    """The version of the form its words were packed by (`FORM`'s bytes, a stack of banks' one), stored with
    them, so a checkpoint's words carry the encoder that packed them (`checked`, `stored`)."""
    FORM_OF: ClassVar[bytes] = FORM
    FACTORY: ClassVar[str] = "NvFp4.of"

    @classmethod
    def quantized(cls, x: jax.Array) -> "NvFp4":
        """`x` (`[depth, width]`, or a bank's `[groups, depth, width]`) quantized to NVFP4
        (`bits.nvfp4_quantized`, each matrix: its block scales and tensor scale, each value rounded to E2M1),
        its codes packed in the staged order."""
        def one(x):
            codes, scales, scale = bits.nvfp4_coded(jnp.asarray(x, jnp.float32))
            return _e2m1_values(codes), scales, scale

        return cls.of(*(jax.vmap(one) if x.ndim == 3 else one)(x))

    @classmethod
    def of(cls, codes, scales, scale) -> "NvFp4":
        """The matrix of `codes` (`[depth, width]`, or a bank's `[groups, depth, width]`, in the depths' own
        order: each an E2M1 value, or its 4-bit code as a uint8 nibble, its sign bit 3), its block scales
        `scales` (`[..., depth / 16, width]` float8_e4m3fn) and its tensor scale `scale` (one a matrix): a
        checkpoint's NVFP4 weight, its codes packed in the staged order, K-major. NumPy codes, scales and
        scale give a NumPy matrix (packed on the host, nothing copied to a device: a load places each
        expert shard's itself)."""
        if (why := unstaged(codes.shape[-2])) is not None:
            raise ValueError(f"an NVFP4 matrix: {why}")
        host = isinstance(codes, np.ndarray) and isinstance(scales, np.ndarray)
        xp = np if host and not isinstance(scale, jax.Array) else jnp
        words = _staged_words(codes, xp)
        return cls(xp.swapaxes(words, -1, -2), xp.swapaxes(scales, -1, -2), xp.asarray(scale, xp.float32))

    def codes(self, dtype: DTypeLike = jnp.float32) -> jax.Array:
        """Its codes' values (`[..., depth, width]` in `dtype`, float32 by default, each code's E2M1 value) in
        the depths' own order."""
        words = jnp.swapaxes(self.words, -1, -2)
        staged = functools.partial(_codes, dtype=dtype)
        # Each matrix's own, over every leading axis: a bank's groups, a stack of banks' layers before them.
        for _ in range(words.ndim - 2):
            staged = jax.vmap(staged)
        return staged(words)

    def decoded(self) -> jax.Array:
        """Its values over `bits.NVFP4_OWED` and the tensor scale, exactly, in bfloat16 (`[..., depth,
        width]`): each code times its block's scale over 2^8, as the mainloop's stage decode forms it."""
        owed = bits.scales_decoded(jax.lax.bitcast_convert_type(jnp.swapaxes(self.scales, -1, -2), jnp.uint8)
                                   ) * jnp.float32(1 / bits.NVFP4_OWED)
        return (self.codes() * jnp.repeat(owed, 16, axis=-2)).astype(jnp.bfloat16)

    @property
    def dtype(self) -> jnp.dtype:
        """The dtype its decode yields (`decoded`), which it is multiplied as: bfloat16."""
        return jnp.dtype(jnp.bfloat16)

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the matrix it stands for: `[depth, width]`, a bank's `[groups, depth, width]`."""
        *groups, width, words = self.words.shape
        return (*groups, 8 * words, width)

    @staticmethod
    def leaf_axes(axes: Sequence[str | None]) -> dict[str, tuple[str | None, ...]]:
        """The logical axes of each of its leaves, in their order, where the matrix it stands for is laid out
        along `axes` (`[..., depth, width]`, any axes before them a bank's or a stack's): its words' and
        block scales' `[..., width, depth / 8 or depth / 16]`, its tensor scales' the leading ones alone,
        its form none (a static field). The one statement of how its layout follows the matrix's,
        which placing a packed bank reads (row 11734). By each leaf's field name, in its fields' order (the
        leaves', as `jax.tree_util.register_dataclass` flattens it)."""
        *leading, depth, width = axes
        return {"words": (*leading, width, depth), "scales": (*leading, width, depth),
                "scale": tuple(leading)}

    def layer(self, index: jax.Array) -> "NvFp4":
        """A stack of banks' bank at layer `index` (a traced index): its fields led by a layers axis before
        their group axis (a scanned load's stacked banks, row 4861), each field's slice at it."""
        return NvFp4(*(jax.lax.dynamic_index_in_dim(jnp.asarray(field), index, keepdims=False)
                       for field in (self.words, self.scales, self.scale)), form=self.form)

    @property
    def factor(self) -> jax.Array:
        """The factor its decode defers to the product's accumulator (`bits.NVFP4_OWED` times the tensor's
        scale, a bank's one a matrix): `bits.nvfp4_decoded` keeps each value within bfloat16's normal range
        by owing 2^8."""
        return jnp.float32(bits.NVFP4_OWED) * jax.lax.stop_gradient(jnp.asarray(self.scale, jnp.float32))

def unstaged(depth: int) -> str | None:
    """Why a matrix of `depth` inputs is no NVFP4 matrix (`NvFp4`: its depth whole `NVFP4_STAGE`s), or None:
    what a load asks of a weight's shape before it reads any of its values (row 11643's review: an MXFP4
    bank's powers and conversion, gigabytes on a host, were made for banks no stage holds)."""
    return f"its depth {depth} is no whole {NVFP4_STAGE}-deep stages" if depth % NVFP4_STAGE else None


def _e2m1_values(codes: jax.Array) -> jax.Array:
    """E2M1 codes' values (`bits.nvfp4_coded`'s uint32 codes), float32, read from the codes' table
    (`fake_quant.E2M1`): a selection, no arithmetic."""
    from dew.nn import fake_quant

    return jnp.asarray(fake_quant.E2M1)[codes]


def _codes(words: jax.Array, dtype: DTypeLike = jnp.float32) -> jax.Array:
    """The codes' values of staged `words` (each its E2M1 value in `dtype`, `[depth, width]`) in the depths'
    own order (`staged_rows` undone): `bits.nvfp4_codes`, the one word-to-code decode (row 8718). The staged
    order permutes the depths within each `packed.NVFP4_GROUP`-deep group alone (its 16 words' 8 codes, code
    `j` of word `w` depth `16 j + w`), so it is undone within each group, a transpose of its words and codes,
    the groups where they lie: a depth an Explicit axis splits by whole groups keeps its placement, where a
    gather across the depth had none (row 14097)."""
    from dew.nn.sharding import explicit_spec

    codes = bits.nvfp4_codes(words, dtype)
    depth, width = codes.shape
    if depth % packed.NVFP4_GROUP:
        raise ValueError(f"an NVFP4 matrix's depth is whole {packed.NVFP4_GROUP}-deep groups, not {depth}")
    # A group's codes as the staged order lays them (`packed.nvfp4_staged_depth`): its words, a block's
    # depth of them, by each word's codes.
    grouped = (depth // packed.NVFP4_GROUP, formats_of.NVFP4.depth, 32 // bits.E2M1.width, width)
    # A split depth's placement stated where its reshapes split and join it, by whole groups; any other
    # placement JAX keeps through them itself.
    placed = explicit_spec(codes)
    deep = placed is not None and placed.spec[0] is not None
    split = jax.sharding.NamedSharding(placed.mesh, jax.sharding.PartitionSpec(
        placed.spec[0], None, None, placed.spec[1])) if deep else None
    held = jnp.swapaxes(codes.reshape(grouped, out_sharding=split), 1, 2)
    return held.reshape(depth, width, out_sharding=placed if deep else None)


def _staged_words(codes, xp=jnp):
    """`codes` (`[..., depth, width]` in the depths' own order: E2M1 values, or uint8 nibbles, the codes
    themselves) packed in the staged order (`NvFp4`), uint32 `[..., depth / 8, width]`, by `xp`'s arrays
    (NumPy's on the host)."""
    ordered = codes[..., staged_rows(codes.shape[-2]), :]
    if xp.dtype(ordered.dtype) == xp.dtype(xp.uint8):
        nibbles = ordered.astype(xp.uint32)
    else:
        fp32 = ordered.astype(xp.float32)
        words = fp32.view(np.uint32) if xp is np else jax.lax.bitcast_convert_type(fp32, jnp.uint32)
        nibbles = bits.encoded(words, bits.E2M1, xp)
    return xp.swapaxes(bits.packed(xp.swapaxes(nibbles, -1, -2), bits.E2M1, xp), -1, -2)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MxFp8:
    """An MXFP8 operand (OCP MX: E4M3 codes, a power-of-two E8M0 scale every 32 depths, `product.MXFP8`):
    `codes` float8_e4m3fn `[..., depth]`, the depth K-major (last), and `scales` float8_e8m0fnu
    `[..., depth / 32]`, each value its code times its block's scale. Rows `[rows, depth]` or a grouped
    product's matrices `[groups, width, depth]`, as Blackwell's block-scaled tensor cores read both."""

    codes: jax.Array
    scales: jax.Array

    @classmethod
    def quantized(cls, x: jax.Array) -> "MxFp8":
        """`x` (`[..., depth]`) as MXFP8, by OCP MX's rule: each 32 depths' scale 2 to the power of its
        largest magnitude's exponent less E4M3's largest's (8), so that magnitude over the scale lies in
        [256, 512); each value over its scale rounded to E4M3, ties to even, saturating at +-448. A block of
        zeros takes the least scale. (A scale the ceiling of the largest magnitude over 448, DeepSeek's
        rule, rounds a code up past the largest float32 at float32's top binade: 256 x 2^120.) The largest
        magnitude's exponent is its binary exponent read from its bits (`jnp.frexp`), exact where a float32
        `log2` rounds a magnitude just under a power of two up onto it, and every step is exact or a rounding
        to E4M3 at the operand's width (`at_least_fp32`): a float64 truth's float32 values quantize to the
        same operand."""
        from dew.nn import fake_quant

        x = jnp.asarray(x)
        x = x.astype(at_least_fp32(x.dtype))
        per = formats_of.MXFP8.depth
        blocks = x.reshape(*x.shape[:-1], x.shape[-1] // per, per)
        largest = jnp.max(jnp.abs(blocks), axis=-1, keepdims=True)
        # Its scale's exponent and E8M0 byte from the largest magnitude's own bits (`mxfp8_scale_bytes`, the
        # GEMM epilogue's quantizer's too; row 13249).
        exponent, stored = mxfp8_scale_bytes(largest)
        scales = jax.lax.bitcast_convert_type(stored.astype(jnp.uint8), formats_of.MXFP8.scale)
        # Each value over its scale as its exponent (`precision.power2`, the one owner of scaling by a power
        # of two; row 8971), as `decoded` applies it: the least scale, a block of zeros' or of magnitudes
        # under 2^-118, is float32's subnormal 2^-127, which a program that flushes subnormals (XLA's CPU, op
        # by op or with its simplifier off) reads as zero, its zeros over it NaN (row 6931). A subnormal
        # value a scale below 1 raises is kept where the backend keeps subnormals (`jnp.ldexp` flushed it).
        codes = fake_quant.fp8_codes(power2(blocks, -exponent)).reshape(x.shape)
        return cls(codes.astype(jnp.float8_e4m3fn), scales[..., 0])

    def decoded(self, dtype=jnp.float32) -> jax.Array:
        """Each E4M3 code times its E8M0 scale in `dtype`, with one destination rounding. The least
        scale byte and subnormal result bits are kept on FTZ backends. A NaN code keeps its sign at a valid
        scale; scale byte 255 gives a positive quiet NaN (OCP MX 5.4.1).
        """
        exponent, invalid = e8m0_exponents(self.scales)
        exponent, invalid = (jnp.repeat(x, formats_of.MXFP8.depth, axis=-1) for x in (exponent, invalid))
        return _decode_mxfp8(jax.dtypes.canonicalize_dtype(dtype), self.codes, exponent, invalid)


def _mxfp8_scale_arithmetic(dtype, codes, exponents, invalid):
    """The floating linear rule; two normal factors even for the least E8M0 scale."""
    wide = jnp.promote_types(dtype, jnp.float32)
    values = codes.astype(wide)
    values = jnp.where(exponents == -127, values * jnp.asarray(0.5, wide), values)
    powers = exact_power2(jnp.maximum(exponents, -126), jnp.finfo(wide).dtype)
    return (values * jnp.where(invalid, jnp.asarray(jnp.nan, wide), powers)).astype(dtype)


def _mxfp8_decoded(dtype, codes, exponents, invalid):
    if dtype == jnp.float64:
        landed = _mxfp8_scale_arithmetic(dtype, codes, exponents, invalid)
    else:
        # Every nonzero finite E4M3 code first widens to a normal fp32. Only the final exponent
        # application can enter the subnormals, so round the stored product at its destination.
        _, sign, biased, fraction = raw_fields(codes.astype(jnp.float32))
        held = jnp.where(biased != 0, fraction | np.uint32(1 << 23), fraction)
        power = biased.astype(jnp.int32) - 150 + exponents
        landing = dtype if dtype in (jnp.float16, jnp.bfloat16, jnp.float32) else jnp.float32
        place = power + 23 - jnp.finfo(landing).nmant
        landed = rounded_binary(sign, held, power, landing, last=place)

    # Read the E4M3 sign from storage; an invalid E8M0 scale instead denotes an unsigned NaN.
    stored = jax.lax.bitcast_convert_type(codes, jnp.uint8)
    special = (stored & np.uint8(0x7F)) == np.uint8(0x7F)
    negative = (stored == np.uint8(0xFF)) & ~invalid
    return jnp.where(special | invalid, quiet_nan(negative, landed.dtype), landed).astype(dtype)


def _mxfp8_decode_tangent(dtype, primals, tangents):
    codes, exponents, invalid = primals
    moved, _, _ = tangents
    value = _decode_mxfp8(dtype, *primals)
    if isinstance(moved, SymbolicZero) or not jnp.issubdtype(dtype, jnp.inexact):
        return value, SymbolicZero(jax.typeof(value).to_tangent_aval())

    def applied(scales, direction):
        return _decode_mxfp8(dtype, direction, *scales)

    def pulled(scales, cotangent):
        # Keep the established floating transpose, narrowing only to the stored code type.
        wide = jnp.promote_types(dtype, jnp.float32)
        return _mxfp8_scale_arithmetic(wide, cotangent, *scales).astype(codes.dtype)

    direction = generation.linear_map(applied, pulled, (exponents, invalid), moved,
                                      out_aval=jax.typeof(value))
    return value, direction


_decode_mxfp8 = contracted("test_a_decoded_mxfp8_operand_preserves_its_linear_rule")(
    declared_tangent(Protected(), _mxfp8_decoded, _mxfp8_decode_tangent,
                     nondiff_argnums=(0,), symbolic_zeros=True))


def mxfp8_scale_bytes(largest: jax.Array) -> tuple[jax.Array, jax.Array]:
    """A 32-value block's MXFP8 scale from its largest magnitude `largest` (float32's, or a float64 truth's),
    read from its own exponent field (`float_bits.raw_fields`): the exponent its codes are scaled by (int32,
    the magnitude's binade less E4M3's largest's 8, clipped to E8M0's -127 to 127: a zero's or a subnormal's
    -127) and its stored E8M0 byte (uint32, the exponent plus 127); a block whose largest is an infinity or a
    NaN (the field all ones) its codes at a unit scale (0) and its byte the NaN scale `0xFF` (row 13164).
    Integer arithmetic on the bits alone: the one mapping `MxFp8.quantized` and the GEMM's quantizing
    epilogue (`quantized._mxfp8_quantize_epilogue`, in a kernel) share (row 13249)."""
    from dew.nn.float_bits import raw_fields

    limits = jnp.finfo(largest.dtype)
    biased = raw_fields(largest)[2].astype(jnp.int32)
    finite = biased != (1 << limits.nexp) - 1
    top = jnp.finfo(jnp.float8_e4m3fn).maxexp - 1
    exponent = jnp.where(finite, jnp.clip(biased - (limits.maxexp - 1) - top, -127, 127), 0)
    return exponent, jnp.where(finite, exponent + 127, 0xFF).astype(jnp.uint32)


def e8m0_exponents(scales: jax.Array) -> tuple[jax.Array, jax.Array]:
    """E8M0 `scales` as their powers' exponents (int32: a byte less 127, `0x00`'s -127 too) and the mask of
    the NaN byte `0xFF`: a block-scaled product's scales as they meet in a sum of exponents
    (`xla_products.exponent_scaled`), no power decoded alone."""
    held = jax.lax.bitcast_convert_type(scales, jnp.uint8).astype(jnp.int32)
    return held - 127, held == 0xFF


def scale_checked(scale_shape: tuple[int, ...], weights_shape: tuple[int, ...]) -> None:
    """Validate that `scale_shape` satisfies the single unambiguous broadcast contract for a scaled packed
    matrix (`TernaryWeights`) of `weights_shape` (`[depth, width]` or `[groups, depth, width]`): scalar
    `()`/`(1,)`, per-column `(width,)` or `(1, width)` / `(1, 1, width)`, or grouped per-group
    `(groups, 1, 1)` / `(groups, 1, width)`."""
    if len(weights_shape) not in (2, 3):
        raise ValueError(f"a scaled packed matrix is 2D [depth, width] or 3D [groups, depth, width], got "
                         f"{weights_shape}")
    width = weights_shape[-1]
    if len(weights_shape) == 2:
        allowed = {(), (1,), (width,), (1, width)}
    else:
        groups = weights_shape[0]
        allowed = {(), (1,), (width,), (1, 1, width), (groups, 1, 1), (groups, 1, width)}
    if scale_shape not in allowed:
        raise ValueError(f"scale shape {scale_shape} is invalid for packed weights {weights_shape}; "
                         f"expected one of {sorted(allowed)}")

def scale_factor(scale: jax.Array, sizes: jax.Array, count: int, weights_shape: tuple[int, ...],
                 dense: bool) -> jax.Array:
    """Broadcast a scaled packed matrix's `scale` to match `[count, width]` output rows under
    `scale_checked`."""
    scale_checked(scale.shape, weights_shape)
    if scale.ndim == 0 or scale.shape == (1,) or dense:
        return scale
    if scale.ndim == 1:
        return scale[None, :]
    if scale.shape[0] == 1:
        return scale[0]
    return jnp.repeat(scale[:, 0, :], sizes, axis=0, total_repeat_length=count)


def _block_values(codes: jax.Array, scales: jax.Array, block: int, dtype) -> jax.Array:
    """Block-scaled codes' values in `dtype`: each code times its `block` depths' E4M3 scale, read by the one
    decode of a block scale's (`bits.scales_decoded`, row 8675), exact in float32 (an E2M1 code times an E4M3
    scale: 6 bits)."""
    held = bits.scales_decoded(jax.lax.bitcast_convert_type(scales, jnp.uint8))
    return codes.astype(dtype) * jnp.repeat(held.astype(dtype), block, axis=-1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NvFp4Block:
    """An NVFP4 operand as Blackwell's block-scaled tensor cores read it (row 4519's NVFP4 arm): `codes`
    float4_e2m1fn `[..., depth]`, the depth K-major (last), each sixteen depths' E4M3 scale (`scales`,
    `[..., depth / 16]`) and the tensor's fp32 `scale`: each value its code times its block's scale times
    the tensor's (`NvFp4`'s values, laid as the products read them). Rows `[rows, depth]` or a grouped
    product's matrices `[groups, width, depth]`."""

    codes: jax.Array
    scales: jax.Array
    scale: jax.Array

    @classmethod
    def quantized(cls, x: jax.Array, tensor_scale: jax.Array | None = None) -> "NvFp4Block":
        """`x` (`[..., depth]`) as NVFP4 along its depth, `bits.nvfp4_quantized`'s rule (one tensor scale,
        its largest magnitude over 6 x 448; each block's the least E4M3 at or above its largest magnitude
        over 6 times it; each value rounded to E2M1, ties to even), at the operand's width (`at_least_fp32`:
        a float64 truth's float32 values quantize to the same operand). `tensor_scale`, the tensor scale in
        place of `x`'s own (a delayed-scaling recipe's, `bits.nvfp4_coded`'s `tensor`, row 12721)."""
        x = jnp.asarray(x)
        x = x.astype(at_least_fp32(x.dtype))
        codes, scales, scale = bits.nvfp4_coded(x.reshape(-1, x.shape[-1]).T, tensor_scale)
        values = _e2m1_values(codes.T).reshape(x.shape).astype(jnp.float4_e2m1fn)
        return cls(values, scales.T.reshape(*x.shape[:-1], -1), scale)

    @classmethod
    def encoded(cls, codes: jax.Array, scales: jax.Array, scale: jax.Array) -> "NvFp4Block":
        """An operand of E2M1 `codes` one a byte (uint8 codes of `bits.nvfp4_coded`'s rule, as a GEMM
        epilogue stores them: `quantized.nvfp4_epilogue`), its block `scales` and tensor `scale`: each code's
        value as float4, which XLA packs two a byte (row 12721; packed 4-bit stores of its own, row 12743);
        its tensor scale one scalar (`bits.nvfp4_given_scale`)."""
        return cls(_e2m1_values(codes).astype(jnp.float4_e2m1fn), scales, bits.nvfp4_given_scale(scale))

    @classmethod
    def packed(cls, codes: jax.Array, scales: jax.Array, scale: jax.Array) -> "NvFp4Block":
        """An operand of E2M1 `codes` two a byte (`[..., depth / 2]` uint8, the even depth's in its low
        nibble, as a GEMM epilogue stores them packed: `quantized.Form`'s `packed`, row 12743), its block
        `scales` and tensor `scale` (one scalar, `bits.nvfp4_given_scale`): the bytes read as the float4
        codes they hold, no pass over them."""
        values = jax.lax.bitcast_convert_type(codes, jnp.float4_e2m1fn)
        return cls(values.reshape(*codes.shape[:-1], -1), scales, bits.nvfp4_given_scale(scale))

    @classmethod
    def of(cls, matrix: NvFp4) -> "NvFp4Block":
        """A stored NVFP4 matrix (`NvFp4`, `[depth, width]`, its codes in the packed kernel's staged order)
        as the block-scaled products read it: `[1, width, depth]`, its depths in their own order, once its
        form is this Dew's (`NvFp4.admitted`)."""
        matrix.admitted()
        return cls(matrix.codes().T.astype(jnp.float4_e2m1fn)[None], jnp.asarray(matrix.scales)[None],
                   jnp.asarray(matrix.scale))

    def decoded(self, dtype=jnp.float32) -> jax.Array:
        """Its block-scaled values, exactly, in `dtype` (`[..., depth]`, float32 or a float64 truth's): each
        code times its block's scale, the tensor's apart (`scale`)."""
        return _block_values(self.codes, self.scales, formats_of.NVFP4.depth, dtype)


TERNARY_STAGE = packed.TERNARY_STEP
"""Depth unit of 2-bit packed BitNet b1.58 ternary weights (`TernaryWeights`, 256 depths = 16 words)."""

def ternary_staged_rows(depth: int) -> np.ndarray:
    """Depth permutation for `TernaryWeights` (`packed.ternary_staged_depth`): word `w` (`0..15`) of each
    256-depth group holds 16 2-bit codes at depths `16 * code + w` (row 1464)."""
    return _staged_rows(depth, packed.TERNARY_GROUP, 2, packed.ternary_staged_depth, "ternary weight")


def ternary_form_version() -> bytes:
    """8-byte digest identifying the BitNet b1.58 2-bit ternary staging layout (`TernaryWeights`)."""
    return form_version(b"ternary-b158-v1\x00".hex(), ternary_staged_rows(TERNARY_STAGE))


TERNARY_FORM = ternary_form_version()
"""This Dew's ternary encoding's digest (`ternary_form_version`), every `TernaryWeights`' form."""


def ternary_packed(codes: jax.Array) -> jax.Array:
    """Pack 2-bit ternary `codes` (`[..., n]` uint32, 0->0, 1->+1, 2->-1) 16 per uint32 word."""
    grouped = codes.reshape(*codes.shape[:-1], codes.shape[-1] // 16, 16)
    shifts = jnp.arange(16, dtype=jnp.uint32) * jnp.uint32(2)
    return jnp.sum((grouped & jnp.uint32(3)) << shifts, axis=-1, dtype=jnp.uint32)


def ternary_code_value(code2: jax.Array, dtype=jnp.float32) -> jax.Array:
    """Decode 2-bit BitNet b1.58 code (`0 -> 0`, `1 -> +1`, `2 -> -1`, `3 -> 0`) to `dtype`."""
    nz = (code2 ^ (code2 >> jnp.uint32(1))) & jnp.uint32(1)
    neg = (code2 >> jnp.uint32(1)) & nz
    f32_bits = (neg << jnp.uint32(31)) | (nz * jnp.uint32(0x3F800000))
    return jax.lax.bitcast_convert_type(f32_bits, jnp.float32).astype(dtype)


def ternary_codes(words: jax.Array, dtype=jnp.float32) -> jax.Array:
    """Decode 2-bit packed ternary `words` (`[depth / 16, columns]` uint32) to `[depth, columns]`."""
    shifts = jnp.arange(16, dtype=jnp.uint32) * jnp.uint32(2)
    code2 = (words[:, None, :] >> shifts[None, :, None]) & jnp.uint32(3)
    return ternary_code_value(code2, dtype).reshape(words.shape[0] * 16, words.shape[-1])


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TernaryWeights(_Packed):
    """BitNet b1.58 ternary weight matrix `[..., depth, width]` packed 16 2-bit codes per `uint32` word in
    staged depth order (`ternary_staged_rows`, row 1464): `words` `[..., width, depth // 16]` uint32,
    absmean/per-column/per-group `scale` fp32, and `form` digest (`_form_field`)."""
    words: jax.Array | np.ndarray
    scale: jax.Array | np.ndarray
    form: bytes = _form_field(TERNARY_FORM)
    fmt: str = dataclasses.field(metadata={"static": True}, default="ternary")
    FORM_OF: ClassVar[bytes] = TERNARY_FORM
    FACTORY: ClassVar[str] = "TernaryWeights.of"

    @classmethod
    def of(cls, weights: jax.Array | np.ndarray, *,
           scale: jax.Array | np.ndarray | float | None = None) -> "TernaryWeights":
        """Quantize `weights` (`[..., depth, width]`, `depth % 256 == 0`) to BitNet b1.58 `{-1, 0, +1}`
        under absmean `scale` (computed automatically when None) and pack into staged `uint32` words."""
        source = jnp.asarray(weights, jnp.float32)
        if source.ndim not in (2, 3) or source.shape[-2] % TERNARY_STAGE != 0:
            raise ValueError(
                f"TernaryWeights expects 2D/3D with depth divisible by {TERNARY_STAGE}, got {source.shape}")
        if scale is None:
            if source.size == 0:
                scale_shape = source.shape[:-2] + ((1, 1) if source.ndim == 3 else ())
                scale_arr = jnp.full(scale_shape, jnp.float32(1e-5))
            else:
                abs_w = jnp.abs(source)
                peak = jnp.max(abs_w, axis=(-2, -1), keepdims=(source.ndim == 3))
                safe_peak = jnp.where(peak > 0, peak, jnp.float32(1.0))
                absmean = peak * jnp.mean(abs_w / safe_peak, axis=(-2, -1), keepdims=(source.ndim == 3))
                scale_arr = jnp.maximum(absmean, jnp.float32(1e-5))
        else:
            scale_arr = jnp.asarray(scale, jnp.float32)
        scale_checked(scale_arr.shape, source.shape)
        quantized = jnp.clip(jnp.round(source / jnp.where(scale_arr > 0, scale_arr, jnp.float32(1.0))),
                             -1.0, 1.0)
        codes = jnp.where(quantized > 0, jnp.uint32(1),
                          jnp.where(quantized < 0, jnp.uint32(2), jnp.uint32(0)))
        staged = codes[..., ternary_staged_rows(source.shape[-2]), :]
        packed_words = ternary_packed(jnp.swapaxes(staged, -1, -2))
        return cls(packed_words, scale_arr)

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.dtype(jnp.bfloat16)

    @property
    def shape(self) -> tuple[int, ...]:
        *groups, width, words = self.words.shape
        return (*groups, 16 * words, width)

    def checked(self) -> "TernaryWeights":
        """Itself, its form this Dew's encoder's (`_Packed.checked`) and its scale's shape its words'."""
        super().checked()
        scale_checked(jnp.shape(self.scale), self.shape)
        return self

    def unscaled(self, dtype=jnp.bfloat16) -> jax.Array:
        """Decode packed `words` (`[..., width, depth // 16]`) back to `[..., depth, width]` ternary
        `{-1, 0, +1}` in `dtype`."""
        words = jnp.asarray(self.words)
        *batches, width, wdepth = words.shape
        depth = 16 * wdepth
        batch = math.prod(batches)
        if batch == 0 or width == 0 or wdepth == 0:
            return jnp.zeros((*batches, depth, width), dtype)
        inv = np.argsort(ternary_staged_rows(depth))
        flat = words.reshape(batch, width, wdepth)
        decoded = jax.vmap(lambda w: ternary_codes(w.T, jnp.float32)[inv])(flat)
        return decoded.reshape(*batches, depth, width).astype(dtype)

    def decoded(self, dtype=jnp.bfloat16) -> jax.Array:
        """Decode the packed values in `dtype`, including a representable subnormal stored scale."""
        scale_checked(jnp.shape(self.scale), self.shape)
        dtype = jax.dtypes.canonicalize_dtype(dtype)
        codes, scale = self.unscaled(jnp.float32), jnp.asarray(self.scale)
        if scale.dtype == jnp.float32 and dtype in (jnp.float16, jnp.bfloat16, jnp.float32, jnp.float64):
            return _decode_ternary(dtype, codes, scale)
        return _ternary_arithmetic(dtype, codes, scale)


def _ternary_arithmetic(dtype, codes, scale):
    wide = jnp.promote_types(dtype, jnp.float32)
    return (codes.astype(wide) * scale.astype(wide)).astype(dtype)


def _ternary_decoded(dtype, codes, scale):
    ordinary = _ternary_arithmetic(dtype, codes, scale)
    _, sign, biased, fraction = raw_fields(scale)
    power = jnp.full(fraction.shape, -149, jnp.int32)
    if dtype == jnp.float64:
        widened = fraction.astype(dtype) * exact_power2(power, dtype)
    else:
        widened = rounded_binary(jnp.zeros_like(sign), fraction, power, dtype)
    width = jnp.finfo(dtype).bits
    word = jnp.dtype(f"uint{width}")
    bits = jax.lax.bitcast_convert_type(widened, word)
    # Stored zero codes have positive sign; multiplying by a negative scale gives them negative zero.
    negative = ((sign != 0) ^ (codes < 0)).astype(word) << word.type(width - 1)
    magnitude = jnp.where(codes != 0, bits, word.type(0))
    exact = jax.lax.bitcast_convert_type(magnitude | negative, dtype)
    return jnp.where((biased == 0) & (fraction != 0), exact, ordinary)


def _ternary_decode_tangent(dtype, primals, tangents):
    codes, scale = primals
    _, moving = tangents
    value = _decode_ternary(dtype, codes, scale)
    if isinstance(moving, SymbolicZero):
        return value, SymbolicZero(jax.typeof(value).to_tangent_aval())

    def applied(held, direction):
        return _decode_ternary(dtype, held[0], direction)

    def pulled(held, cotangent):
        # Let the floating linear rule reduce broadcasts in the scale's original layout.
        codes, scale = held
        _, pull = jax.vjp(lambda weights: _ternary_arithmetic(dtype, codes, weights), scale)
        return pull(cotangent)[0]

    direction = generation.linear_map(applied, pulled, primals, moving, out_aval=jax.typeof(value))
    return value, direction


_decode_ternary = contracted("test_decoded_ternary_preserves_its_linear_scale_rule")(
    declared_tangent(Protected(), _ternary_decoded, _ternary_decode_tangent,
                     nondiff_argnums=(0,), symbolic_zeros=True))


def window_byte_rows(depth: int, layout: packed.WindowLayout = packed.WINDOW) -> np.ndarray:
    """The depth each byte of a `Bf16Window`'s staged byte plane holds (`packed.WindowLayout.byte_depth`)."""
    return _staged_rows(depth, layout.byte_block, 8, layout.byte_depth, "bfloat16 window")


def window_nibble_rows(depth: int, layout: packed.WindowLayout = packed.WINDOW) -> np.ndarray:
    """The depth each nibble of a `Bf16Window`'s staged offset plane holds (`WindowLayout.nibble_depth`)."""
    return _staged_rows(depth, layout.group, 4, layout.nibble_depth, "bfloat16 window")


def window_form_version(layout: packed.WindowLayout = packed.WINDOW) -> bytes:
    """The 8-byte digest of the `Bf16Window` encoding at `layout`: this Dew's served layout's the digest its
    windows have always stored (its encoder's name, `bf16-window-v2`, and both planes' staged orders), so
    every saved window restores; another layout's folds its stored fields in too
    (`packed.WindowLayout.fingerprint`, its meta's words among them), so its encoding reads as another's."""
    order = np.concatenate([window_byte_rows(layout.group, layout), window_nibble_rows(layout.group, layout)])
    if layout == packed.WINDOW:
        return form_version(b"bf16-window-v2\x00".hex(), order)
    return form_version(b"bf16-window-v3".hex() + layout.fingerprint().hex(), order)


WINDOW_FORM = window_form_version()
"""This Dew's served window encoding's digest (`window_form_version` at `packed.WINDOW`), every
`Bf16Window`'s form."""


def window_planes(weights: jax.Array | np.ndarray, layout: packed.WindowLayout = packed.WINDOW, *,
                  site: str = "a matrix") -> tuple[np.ndarray, np.ndarray]:
    """The `Bf16Window` codec's planes of `weights` (`[..., depth, width]` bfloat16, `depth` whole
    `layout.group`s) at `layout` (this Dew's, the one served; another's a geometry's witness, which its
    `window_weights` and its stage reader, `packed.weight_format`, are given): its words and its meta, each
    group's base its largest normal exponent less 14; refused where a group of `site`'s holds more than
    `layout.escapes` nonzeros below its window."""
    source = np.asarray(weights)
    group = layout.group
    if source.dtype != np.dtype(jnp.bfloat16) or source.ndim not in (2, 3) or source.shape[-2] % group:
        raise ValueError(f"a bfloat16 window encodes a 2D/3D bfloat16 matrix of whole {group}-deep "
                         f"groups, not {site}'s {source.dtype} {source.shape}")
    *lead, depth, width = source.shape
    held = np.swapaxes(source.view(np.uint16).astype(np.uint32), -1, -2)
    grouped = held.reshape(*lead, width, depth // group, group)
    # Its exponents signed: a group whose largest is below the window's binades floors its base at 1.
    sign, exponent, mantissa = grouped >> 15, ((grouped >> 7) & 0xFF).astype(np.int32), grouped & 0x7F
    zero = (grouped & 0x7FFF) == 0
    normal = exponent > 0
    base = np.maximum(np.where(normal, exponent, 0).max(-1, keepdims=True) - (layout.binades - 1), 1)
    window = normal & (exponent >= base)
    escape = ~zero & ~window
    count = escape.sum(-1)
    if (count > layout.escapes).any():
        column, at = np.argwhere(count > layout.escapes)[0][-2:]
        raise ValueError(f"{site}'s column {column}, depths {at * group} to {(at + 1) * group}, holds "
                         f"{count.max()} weights below its window of {layout.binades} binades, past "
                         f"the {layout.escapes} exact escapes a group stores: serve it as its bfloat16 "
                         "weights")
    slot = np.cumsum(escape, axis=-1) * escape
    byte = np.where(escape, slot, (sign << 7) | np.where(window, mantissa, 0))
    offset = np.where(window, exponent - base + 1, 0)
    table = np.zeros((*grouped.shape[:-1], layout.escapes), np.uint32)
    found = np.nonzero(escape)
    table[(*found[:-1], slot[found] - 1)] = grouped[found]
    # Each meta word where the layout places it: the base, the escapes two a word, the rest zero.
    meta = np.zeros((*grouped.shape[:-1], layout.meta_words), np.uint32)
    meta[..., layout.base_word] = base[..., 0]
    for pair, word in enumerate(layout.escape_words):
        meta[..., word] = table[..., 2 * pair] | table[..., 2 * pair + 1] << 16
    bytes_ = byte.reshape(*lead, width, depth)[..., window_byte_rows(depth, layout)]
    nibbles = offset.reshape(*lead, width, depth)[..., window_nibble_rows(depth, layout)]
    low = (bytes_.reshape(*lead, width, depth // 4, 4) << (8 * np.arange(4, dtype=np.uint32))).sum(
        -1, dtype=np.uint32)
    offsets = (nibbles.reshape(*lead, width, depth // 8, 8) << (4 * np.arange(8, dtype=np.uint32))).sum(
        -1, dtype=np.uint32)
    groups = depth // group
    words = np.concatenate([low.reshape(*lead, width, groups, layout.byte_words),
                            offsets.reshape(*lead, width, groups, layout.words - layout.byte_words)],
                           axis=-1)
    return (words.reshape(*lead, width, groups * layout.words),
            meta.reshape(*lead, width, groups * layout.meta_words))


def window_weights(words, meta, dtype=jnp.bfloat16,
                   layout: packed.WindowLayout = packed.WINDOW) -> jax.Array:
    """The `Bf16Window` codec's matrix `[..., depth, width]` in `dtype` from its planes at `layout`
    (`window_planes`'), each weight its stored bits again, a bit operation a weight, traced or not, each
    plane's words unpacked into its depths' order by shifts and reshapes alone (`WindowLayout.unpacked`).
    It reads no index: a gather by the planes' staged orders, the decode before it, read an index a weight
    beside two uint32 planes the matrix's size, and took an A100 0.75 ms for Qwen3-4B's gate and up
    projections a call where their bfloat16 product took 0.078 (M-NG-49-a100's window-xla arm, row 13647;
    M-NG-59 times this one)."""
    words, meta = (jnp.asarray(plane, jnp.uint32) for plane in (words, meta))
    *lead, width, count = words.shape
    groups = count // layout.words
    grouped_shape = (*lead, width, groups)
    grouped = words.reshape(*grouped_shape, layout.words)
    byte = layout.unpacked(grouped[..., :layout.byte_words], 8)
    offset = layout.unpacked(grouped[..., layout.byte_words:], 4)
    words4 = meta.reshape(*grouped_shape, layout.meta_words)
    held = window_bits(byte, offset, words4[..., layout.base_word, None],
                       tuple(words4[..., word, None] for word in layout.escape_words))
    stored = held.astype(jnp.uint16).reshape(*lead, width, groups * layout.group)
    values = jax.lax.bitcast_convert_type(stored, jnp.bfloat16)
    return jnp.swapaxes(values, -1, -2).astype(dtype)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Bf16Window(_Packed):
    """A frozen bfloat16 matrix `[..., depth, width]` stored lossless in 13 bits a weight (row 5387): each
    column's `packed.WINDOW.group` depths share a base exponent `b`, each weight its sign, its 7 mantissa bits
    and a 4-bit offset `1..15` of its exponent from `b - 1`; offset 0 is a signed zero where its mantissa bits
    are, else the index `1..4` of the group's exact escape (`WindowLayout.escapes`: a nonzero below the
    window). Its geometry is `packed.WINDOW`'s alone (`packed.WindowLayout`, row 13707). K-major, each group's
    48 words a column in the order a stage decodes them a lane at a time (`words`, `[..., width, 3 * depth //
    8]` uint32): its bytes (sign and mantissa, or an escape's index), 32 words in two 64-deep blocks of 16
    (`WindowLayout.byte_depth`), then its nibbles, 16 words (`WindowLayout.nibble_depth`); `meta` each group's
    four words `[..., width, depth // 32]` uint32 (its base, its escapes' bits two a word, a zero word);
    `form` its encoder's digest (`_form_field`). Decoded, every weight is its bits again, signed zeros,
    subnormals, infinities and NaNs among them (`decoded`, `window_bits`)."""

    words: jax.Array | np.ndarray
    meta: jax.Array | np.ndarray
    form: bytes = _form_field(WINDOW_FORM)
    fmt: str = dataclasses.field(metadata={"static": True}, default="window")
    FORM_OF: ClassVar[bytes] = WINDOW_FORM
    FACTORY: ClassVar[str] = "Bf16Window.of"

    @classmethod
    def of(cls, weights: jax.Array | np.ndarray, *, site: str = "a matrix") -> "Bf16Window":
        """`weights` (`[..., depth, width]` bfloat16, `depth` whole `packed.WINDOW.group`s) encoded on the
        host at this Dew's layout (`window_planes`), each group's base its largest normal exponent less 14;
        refused where a group of `site`'s holds more than `WindowLayout.escapes` nonzeros below its window (a
        heavy tail no window of 15 binades holds), its weights then served as they are."""
        return cls(*window_planes(weights, site=site))

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.dtype(jnp.bfloat16)

    @property
    def shape(self) -> tuple[int, ...]:
        *lead, width, words = jnp.shape(self.words)
        return (*lead, words // packed.WINDOW.words * packed.WINDOW.group, width)

    @property
    def meta_halves(self) -> jax.Array:
        """Its meta words as their 16-bit halves, low half first (`[..., width, depth // 16]` uint16: a
        group's base, a zero, its four escapes' bits and two zeros), the stream a stage decode reads beside
        its words, a 256-deep step's a 32-byte row (`packed.weight_format('window')`)."""
        meta = jnp.asarray(self.meta, jnp.uint32)
        halves = jnp.stack([meta & 0xFFFF, meta >> 16], axis=-1).astype(jnp.uint16)
        return halves.reshape(*meta.shape[:-1], 2 * meta.shape[-1])


    def decoded(self, dtype=jnp.bfloat16) -> jax.Array:
        """The matrix `[..., depth, width]` in `dtype`, each weight its stored bits again (a bit operation a
        weight, traced or not, `window_weights`): its base and offset's exponent beside its sign and
        mantissa, a signed zero, or its group's escape."""
        return window_weights(self.words, self.meta, dtype)


def window_bits(byte: jax.Array, offset: jax.Array, base: jax.Array, escape_words: tuple) -> jax.Array:
    """Each `Bf16Window` weight's bfloat16 bits (a uint32's low 16 bits, its high half unread) from its
    stored byte (sign and mantissa, or an escape's index), its 4-bit offset, its group's base and its group's
    two escape words (`WindowLayout.escape_words`: escapes one to four, two a word, low half first; each
    broadcast against the weights): the one decode a matrix decoded whole (`Bf16Window.decoded`) and a
    stage decoded a lane at a time (`packed`'s window lanes) read, in integer operations alone (escape `i`
    its word's half by a select and a shift, no gather)."""
    mantissa, signed = byte & 0x7F, (byte & 0x80) << 8
    coded = signed | ((base + offset - 1) << 7) | mantissa
    low, high = escape_words
    escaped = jnp.where(mantissa <= 2, low, high) >> (((mantissa - 1) & 1) << 4)
    return jnp.where(offset > 0, coded, jnp.where(mantissa > 0, escaped, signed))


PACKED_FORMATS = (NvFp4, TernaryWeights, Bf16Window)
"""The packed matrices a variables tree holds in its `QUANTIZATION` collection, a layer's served as its
codes (`dense.packed_matrix`), each stating its encoder's `form`, which its products check (`checked`), and
a checkpoint holding it as its record (`stored`), the digest beside its words that a restore checks
(`restored`)."""


@dataclasses.dataclass(frozen=True)
class _Record:
    """A packed matrix type's checkpoint record (`stored`) and the one split of its fields both directions
    read: its data fields (`leaves`, the matrix's array leaves) and its static ones but its digest
    (`static`). The record type holds the leaves, then the digest `form` a leaf, then the static fields as
    its structure, keyed and ordered as the matrix type registered them while its digest was a leaf (before
    row 13700): every checkpoint, an earlier Dew's too, holds the same leaves at the same paths, and a
    collection's integrity digest reads the same."""

    type: type
    leaves: tuple[str, ...]
    static: tuple[str, ...]

    @classmethod
    def of(cls, matrix_type) -> "_Record":
        fields = dataclasses.fields(matrix_type)
        leaves = tuple(field.name for field in fields if not field.metadata.get("static"))
        static = tuple(field.name for field in fields
                       if field.metadata.get("static") and field.name != "form")
        record = dataclasses.make_dataclass(
            f"Stored{matrix_type.__name__}",
            [*((name, object) for name in leaves), ("form", object),
             *((name, object, dataclasses.field(metadata={"static": True})) for name in static)], frozen=True)
        return cls(jax.tree_util.register_dataclass(record), leaves, static)


RECORDS = {cls: _Record.of(cls) for cls in PACKED_FORMATS}
"""Each packed matrix type's checkpoint record (`_Record`), by the type."""

RECORD_TYPES = tuple(record.type for record in RECORDS.values())
"""Every packed matrix's checkpoint record type (`RECORDS`), a node of a tree a checkpoint holds."""


def record_of(matrix, form):
    """Packed `matrix`'s record (`RECORDS`) holding `form` as its digest leaf (an array, or a restore's asked
    one), `matrix`'s other leaves and static fields its own (`matrix` a restore's template too)."""
    record = RECORDS[type(matrix)]
    held = {name: getattr(matrix, name) for name in (*record.leaves, *record.static)}
    return record.type(**held, form=form)


def digest_placement(words) -> jax.sharding.Sharding | None:
    """Where a packed matrix's digest leaf lies beside its `words` (an array, or a restore's asked one): whole
    on every device of their mesh where a mesh places them, on their device where one holds them, on this
    process's first device where none places them; None for host words, whose digest stays on the host."""
    placed = getattr(words, "sharding", None)
    if isinstance(placed, jax.sharding.NamedSharding):
        return jax.sharding.NamedSharding(placed.mesh, jax.sharding.PartitionSpec(),
                                          memory_kind=placed.memory_kind)
    if placed is not None:
        return placed if isinstance(placed, jax.sharding.SingleDeviceSharding) else None
    return jax.sharding.SingleDeviceSharding(jax.local_devices()[0]) if isinstance(
        words, jax.ShapeDtypeStruct) else None


def stored(matrix):
    """The record a checkpoint holds of packed `matrix` (`record_of`): its form's bytes its digest leaf, a
    uint8 array placed beside its words (`digest_placement`; a checkpoint's local tier writes device arrays
    alone)."""
    digest = np.frombuffer(matrix.form, np.uint8).copy()
    placed = digest_placement(matrix.words)
    if isinstance(placed, jax.sharding.NamedSharding):
        digest = jax.make_array_from_callback(digest.shape, placed, lambda index: digest[index],
                                              dtype=np.uint8)
    elif placed is not None:
        digest = jax.device_put(digest, placed)
    return record_of(matrix, digest)


def restored(template, record):
    """The packed matrix of `template`'s type and static fields a checkpoint's `record` (`stored`'s, read
    back) holds: refused where a row of its digest this process holds (an earlier Dew's stack, a row a
    layer; none where its words lie on other processes' devices alone) is not this Dew's form, or the matrix
    fails its own check (`checked`), before any product reads its words."""
    name = type(template).__name__
    if found := _digests(record.form, len(template.form)):
        _checked_form(found, template.FORM_OF, name, template.FACTORY)
    leaves = {name: getattr(record, name) for name in RECORDS[type(template)].leaves}
    return dataclasses.replace(template, **leaves).checked()
