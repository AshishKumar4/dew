"""The low-precision operand formats the GEMM template multiplies (row 3621, 4519): an NVFP4 matrix as Dew
stores it (`NvFp4`, its E2M1 codes in the staged order the mainloop decodes, its block and tensor scales), an
MXFP8 operand (`MxFp8`, E4M3 codes under E8M0 scales every 32 of its depth) and an NVFP4 operand of the
block-scaled products (`NvFp4Block`): their codecs and values alone. Their products and selectors are the
GEMM's (`gemm`), their recipes' projections a layer's (`projections`)."""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np

from dew.algebra import bits
from dew.nn.kernels.templates import product as formats_of
from dew.nn.kernels.templates.gemm import packed
from dew.nn.precision import at_least_fp32


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class NvFp4:
    """An NVFP4 matrix `[depth, width]` as the GEMM stores it (row 3621), K-major, each column's depths
    along its row (as a Dense kernel read transposed): its E2M1 codes eight a uint32 in the GEMM's staged
    depth order (`words`, `[width, depth / 8]`: `staged_rows`, each word the
    `bits.NVFP4_BFLOAT16_FORM.prepared` word of its eight codes), each sixteen consecutive depths' E4M3
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
        depth = codes.shape[-2]
        if depth % NVFP4_STAGE:
            raise ValueError(f"an NVFP4 matrix's depth is whole {NVFP4_STAGE}s, not {depth}")
        host = isinstance(codes, np.ndarray) and isinstance(scales, np.ndarray)
        xp = np if host and not isinstance(scale, jax.Array) else jnp
        words = _staged_words(codes, xp)
        return cls(xp.swapaxes(words, -1, -2), xp.swapaxes(scales, -1, -2), xp.asarray(scale, xp.float32))

    def codes(self) -> jax.Array:
        """Its codes' values (`[..., depth, width]` float32, each code's E2M1 value) in the depths' own
        order."""
        words = jnp.swapaxes(self.words, -1, -2)
        staged = functools.partial(_codes, staged=True)
        # Each matrix's own, over every leading axis: a bank's groups, a stack of banks' layers before them.
        for _ in range(words.ndim - 2):
            staged = jax.vmap(staged)
        return staged(words)

    def decoded(self) -> jax.Array:
        """Its values over `bits.NVFP4_OWED` and the tensor scale, exactly, in bfloat16 (`[..., depth,
        width]`): each code times its block's scale over 2^8, as the mainloop's stage decode forms it."""
        owed = jnp.swapaxes(self.scales, -1, -2).astype(jnp.float32) * jnp.float32(1 / bits.NVFP4_OWED)
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

    def layer(self, index: jax.Array) -> "NvFp4":
        """A stack of banks' bank at layer `index` (a traced index): its fields led by a layers axis before
        their group axis (a scanned load's stacked banks, row 4861), each field's slice at it."""
        return NvFp4(*(jax.lax.dynamic_index_in_dim(jnp.asarray(field), index, keepdims=False)
                       for field in (self.words, self.scales, self.scale)))

    @property
    def factor(self) -> jax.Array:
        """The factor its decode defers to the product's accumulator (`bits.NVFP4_OWED` times the tensor's
        scale, a bank's one a matrix): `bits.nvfp4_decoded` keeps each value within bfloat16's normal range
        by owing 2^8."""
        return jnp.float32(bits.NVFP4_OWED) * jax.lax.stop_gradient(jnp.asarray(self.scale, jnp.float32))

NVFP4_STAGE = packed.NVFP4_STEP
"""The depths a stage of the mainloop's NVFP4 decode reads: two groups, so each column's scales are a
16-byte row (a TMA copy's least)."""


def staged_rows(depth: int) -> np.ndarray:
    """The depth each code of the staged order holds (`NvFp4`): packed row `8w + j` of a group (word `w`'s
    code `j`) holds the depth `packed.nvfp4_staged_depth` gives it, so a stage's word rows decode, code by
    code, into its sixteen-row depth slices, the block of one scale each (the mainloop's stage decode)."""
    group = packed.NVFP4_GROUP
    if depth % group:
        raise ValueError(f"an NVFP4 matrix's depth is whole {group}-deep groups, not {depth}")
    groups, within = np.divmod(np.arange(depth), group)
    word, code = np.divmod(within, 32 // bits.E2M1.width)
    return packed.nvfp4_staged_depth(groups, word, code)


def _e2m1_values(codes: jax.Array) -> jax.Array:
    """E2M1 codes' values (`bits.nvfp4_coded`'s uint32 codes), float32, read from the codes' table
    (`fake_quant.E2M1`): a selection, no arithmetic."""
    from dew.nn import fake_quant

    return jnp.asarray(fake_quant.E2M1)[codes]


def _codes(words: jax.Array, *, staged: bool = False) -> jax.Array:
    """The codes' values of `words` (each its E2M1 value, float32 `[depth, width]`), in the depths' own order
    where `staged` (`staged_rows`), else the words' own: `bits.nvfp4_decoded` at a block scale of 2^8."""
    unit = jnp.full((words.shape[0] // 2, words.shape[1]), bits.NVFP4_OWED, jnp.float8_e4m3fn)
    codes = bits.nvfp4_decoded(words, unit).astype(jnp.float32)
    return codes[np.argsort(staged_rows(codes.shape[0]))] if staged else codes


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
        scales = jax.lax.bitcast_convert_type((exponent + bias).astype(jnp.uint8), scale)
        codes = fake_quant.fp8_codes(blocks / scales.astype(x.dtype)).reshape(x.shape)
        return cls(codes.astype(jnp.float8_e4m3fn), scales[..., 0])

    def decoded(self, dtype=jnp.float32) -> jax.Array:
        """Its values, exactly, in `dtype` (`[..., depth]`, float32 or a float64 truth's): each code times
        its block's scale."""
        return _block_values(self.codes, self.scales, formats_of.MXFP8.depth, dtype)


def _block_values(codes: jax.Array, scales: jax.Array, block: int, dtype) -> jax.Array:
    """Block-scaled codes' values in `dtype`: each code times its `block` depths' scale, exact in float32
    (a code's 4 bits times a power of two, or an E2M1 code times an E4M3 scale: 6 bits)."""
    return codes.astype(dtype) * jnp.repeat(scales.astype(dtype), block, axis=-1)


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
    def quantized(cls, x: jax.Array) -> "NvFp4Block":
        """`x` (`[..., depth]`) as NVFP4 along its depth, `bits.nvfp4_quantized`'s rule (one tensor scale,
        its largest magnitude over 6 x 448; each block's the least E4M3 at or above its largest magnitude
        over 6 times it; each value rounded to E2M1, ties to even), at the operand's width (`at_least_fp32`:
        a float64 truth's float32 values quantize to the same operand)."""
        x = jnp.asarray(x)
        x = x.astype(at_least_fp32(x.dtype))
        codes, scales, scale = bits.nvfp4_coded(x.reshape(-1, x.shape[-1]).T)
        values = _e2m1_values(codes.T).reshape(x.shape).astype(jnp.float4_e2m1fn)
        return cls(values, scales.T.reshape(*x.shape[:-1], -1), scale)

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
