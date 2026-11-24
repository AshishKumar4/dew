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

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike

from dew.algebra import bits
from dew.nn.float_bits import exact_power2
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

def form_version(encoder: str, order: np.ndarray) -> np.ndarray:
    """The 8-byte digest of an `NvFp4` encoding: its bit-form `encoder` (`bits.BitForm.version`) and the
    staged depth `order` (`staged_rows(NVFP4_STAGE)`). Changing either invalidates stored words."""
    digest = hashlib.sha256(bytes.fromhex(encoder) + np.asarray(order, "<i4").tobytes()).digest()[:8]
    return np.frombuffer(digest, np.uint8).copy()


FORM = form_version(bits.NVFP4_BFLOAT16_FORM.version, staged_rows(NVFP4_STAGE))
"""The version of the form an `NvFp4`'s words are packed by (`form_version`: its encoder's digest and its
staged depth order, `staged_rows(NVFP4_STAGE)`), as the bytes a matrix stores beside its words
(`NvFp4.form`): words packed by another encoder or staged order decode wrong by this Dew's form, so a load
refuses them (`NvFp4.checked`; algebra.md 7.4, rows 6101, 8662)."""

def _checked_form(stored, expected: np.ndarray, label: str, factory: str, *, form=None) -> None:
    """Verify `stored` (or gathered `form`) matches `expected` across every addressable shard (row 8958)."""
    target = stored if form is None else form
    pieces = ([shard.data for shard in target.addressable_shards]
              if isinstance(target, jax.Array) else [target])
    found: set[str] = set()
    for piece in pieces:
        held = np.asarray(piece)
        if held.ndim == 0 or held.shape[-1] != expected.size:
            raise ValueError(f"{label} form digest sharded across its {expected.size} bytes "
                             f"(shard shape {held.shape}); pass the gathered form to checked()")
        found.update(bytes(row).hex() for row in held.reshape(-1, expected.size))
    if found != {expected.tobytes().hex()}:
        raise ValueError(f"{label} words packed by form version {', '.join(sorted(found))}, where this Dew "
                         f"decodes form version {expected.tobytes().hex()}: repack them from the source "
                         f"weights (`{factory}`)")


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NvFp4:
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
    form: jax.Array | np.ndarray = dataclasses.field(default_factory=FORM.copy)
    """The version of the form its words were packed by (`FORM`'s bytes; a stack of banks' one a layer),
    stored with them, so a checkpoint's words carry the encoder that packed them (`checked`)."""

    def checked(self, form=None) -> "NvFp4":
        """Itself, where its words are the checked-in form's (`FORM`), every layer's; refused otherwise:
        words another encoder packed (a checkpoint an earlier Dew wrote) read wrong through this form's
        decode, so they are repacked from the source weights, never decoded. A local host check of its
        stored bytes (every addressable shard, or the gathered `form` a collective restore hands it,
        row 8958)."""
        _checked_form(self.form, FORM, "NVFP4", "NvFp4.of", form=form)
        return self

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
        its form's bytes whole on every device. The one statement of how its layout follows the matrix's,
        which placing a packed bank reads (row 11734). By each leaf's field name, in its fields' order (the
        leaves', as `jax.tree_util.register_dataclass` flattens it)."""
        *leading, depth, width = axes
        named = {"words": (*leading, width, depth), "scales": (*leading, width, depth),
                 "scale": tuple(leading), "form": (None,)}
        return {field.name: named[field.name] for field in dataclasses.fields(NvFp4)}

    def layer(self, index: jax.Array) -> "NvFp4":
        """A stack of banks' bank at layer `index` (a traced index): its fields led by a layers axis before
        their group axis (a scanned load's stacked banks, row 4861), each field's slice at it."""
        return NvFp4(*(jax.lax.dynamic_index_in_dim(jnp.asarray(field), index, keepdims=False)
                       for field in (self.words, self.scales, self.scale, self.form)))

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
    own order (`staged_rows` undone): `bits.nvfp4_codes`, the one word-to-code decode (row 8718)."""
    codes = bits.nvfp4_codes(words, dtype)
    return codes[np.argsort(staged_rows(codes.shape[0]))]


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
        # The largest magnitude is a fraction in [1/2, 1) times 2 to `binade`: its exponent is `binade` - 1.
        binade = jnp.frexp(largest)[1]
        top = jnp.finfo(jnp.float8_e4m3fn).maxexp - 1
        scale = formats_of.MXFP8.scale
        bias = -jnp.finfo(scale).minexp
        exponent = jnp.clip(jnp.where(largest > 0, binade - 1 - top, -bias), -bias, bias)
        # The power its scale is, set in E8M0's bits by the one owner of an exact power (`exact_power2`: every
        # biased exponent a power, none a zero; its byte the exponent plus 127, which its int8 sum wraps to
        # past 127 and its bitcast reads back; row 8832).
        scales = exact_power2(exponent, scale)
        # Each value over its scale as its exponent (`precision.power2`, the one owner of scaling by a power
        # of two; row 8971), as `decoded` applies it: the least scale, a block of zeros' or of magnitudes
        # under 2^-118, is float32's subnormal 2^-127, which a program that flushes subnormals (XLA's CPU, op
        # by op or with its simplifier off) reads as zero, its zeros over it NaN (row 6931). A subnormal
        # value a scale below 1 raises is kept where the backend keeps subnormals (`jnp.ldexp` flushed it).
        codes = fake_quant.fp8_codes(power2(blocks, -exponent)).reshape(x.shape)
        return cls(codes.astype(jnp.float8_e4m3fn), scales[..., 0])

    def decoded(self, dtype=jnp.float32) -> jax.Array:
        """Its values, exactly, in `dtype` (`[..., depth]`, float32 or a float64 truth's): each code times
        its block's E8M0 scale decoded through `bits.E8M0_SCALE_FORM` (OCP MX 5.4.1, row 9040), preserving
        both `0x00` (`2^-127`) without FTZ flush and `0xFF` NaN."""
        wide = jnp.promote_types(dtype, jnp.float32)
        scales_f32, subnormal = e8m0_scales(self.scales)
        rep_scale = jnp.repeat(scales_f32.astype(wide), formats_of.MXFP8.depth, axis=-1)
        rep_sub = jnp.repeat(subnormal, formats_of.MXFP8.depth, axis=-1)
        codes = self.codes.astype(wide)
        # For 0x00 (2^-127), split into two normal powers (0.5 * 2^-126) so FTZ hardware never flushes the
        # scale before multiplying normal codes (`448 * 2^-127 = 1.75 * 2^-119` stays normal).
        sub_vals = (codes * jnp.asarray(0.5, wide)) * jnp.asarray(2.0 ** -126, wide)
        return jnp.where(rep_sub, sub_vals, codes * rep_scale).astype(dtype)


def e8m0_scales(scales: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Decode E8M0 `scales` via `bits.e8m0_decoded` into float32 powers (`2^(e-127)` for `1..254`, NaN
    for `0xFF`, `2^-127` for `0x00`) plus the boolean mask of `0x00` subnormal blocks (row 9040)."""
    held = jax.lax.bitcast_convert_type(scales, jnp.uint8).astype(jnp.uint32)
    return bits.e8m0_decoded(held), (held & jnp.uint32(0xFF)) == 0

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
    def of(cls, matrix: NvFp4) -> "NvFp4Block":
        """A stored NVFP4 matrix (`NvFp4`, `[depth, width]`, its codes in the packed kernel's staged order)
        as the block-scaled products read it: `[1, width, depth]`, its depths in their own order."""
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


def ternary_form_version() -> np.ndarray:
    """8-byte digest identifying the BitNet b1.58 2-bit ternary staging layout (`TernaryWeights`)."""
    return form_version(b"ternary-b158-v1\x00".hex(), ternary_staged_rows(TERNARY_STAGE))


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
class TernaryWeights:
    """BitNet b1.58 ternary weight matrix `[..., depth, width]` packed 16 2-bit codes per `uint32` word in
    staged depth order (`ternary_staged_rows`, row 1464): `words` `[..., width, depth // 16]` uint32,
    absmean/per-column/per-group `scale` fp32, and `form` digest."""
    words: jax.Array | np.ndarray
    scale: jax.Array | np.ndarray
    form: jax.Array | np.ndarray
    fmt: str = dataclasses.field(metadata={"static": True}, default="ternary")

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
        return cls(packed_words, scale_arr, ternary_form_version())

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.dtype(jnp.bfloat16)

    @property
    def shape(self) -> tuple[int, ...]:
        *groups, width, words = self.words.shape
        return (*groups, 16 * words, width)

    def checked(self, form=None) -> "TernaryWeights":
        """Verify stored `form` version (every addressable shard) and scale shape match this Dew's encoder."""
        _checked_form(self.form, ternary_form_version(), "TernaryWeights", "TernaryWeights.of", form=form)
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
        """Decode `words` back to `[..., depth, width]` in `dtype`, scaled once in `at_least_fp32`."""
        scale_checked(jnp.shape(self.scale), self.shape)
        wide = jnp.promote_types(dtype, jnp.float32)
        return (self.unscaled(wide) * jnp.asarray(self.scale, wide)).astype(dtype)
