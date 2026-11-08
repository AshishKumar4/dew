"""float32 dense products on a CPU, summed no worse than a vectorized dot sums them (row 4863).

XLA's CPU dot sums each output's float32 products as one sequential FMA chain (all 15 of a llama
decoder's dense sites matched it output for output, row 4735's receipts), so its rounding path grows
with the contraction: 2.0 to 2.7 times libggml's local error at 64- and 128-long sums, and a decoder's
logits 1.77 times llama.cpp's distance from float64. A vectorized dot (libggml's, a BLAS kernel's)
keeps one partial sum a lane of each FMA it holds in flight and sums them as a tree at the end, a path
of `depth / partials` FMAs and `log2(partials)` adds. Here the contraction is split into blocks, each
block a dot of its own (XLA's chain, `block` FMAs long), and the blocks' sums are added as a pairwise
tree (`log2(blocks)` adds): the block is the longest power of two whose path is no longer than the
CPU's own vectorized accumulation (`contraction_block`), read from the CPU's vector lanes and its FMA
latency. Every row sums alike whatever rows share its call, so a row's product is its own: a decode
step's the prefill's, a packed document's its own row's (its partial sums move `depth / block` outputs'
bytes: on armada's EPYC, NS4863-PROF2, 0.55-0.67 of XLA's time at 1-16 rows of 1024-4096 deep, 5-11 times
it at 256 rows; a float64 sum for many rows, cheaper there, made a row's product depend on its batch's
rows, row 4863's base reds). A product's gradients are products too, each summed
over its own contraction (`_matmul`'s reverse pass). On another platform the product is XLA's own
(`accumulated`'s `platform_dependent`).
"""

import functools
import math
import os
import platform
from pathlib import Path

import jax
import jax.numpy as jnp

from dew.nn.derivatives import Protected, contracted

FMA_LATENCY = 4
"""Cycles from an FMA's issue to its result on the cores Dew runs on (Intel Skylake-SP through Sapphire
Rapids, AMD Zen 2 through Zen 4, Arm Neoverse N1, V1 and V2: Agner Fog's instruction tables, Arm's
software optimization guides): the FMAs a dot keeps in flight to issue one a cycle, each with a vector of
partial sums of its own."""


@functools.cache
def vector_lanes() -> int:
    """The float32 lanes of this CPU's widest vector register: 16 with AVX-512 (`avx512f` among its flags),
    8 with AVX or AVX2, 4 for Arm's NEON (`asimd`) and any CPU naming none of them."""
    flags: set[str] = set()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith(("flags", "Features")):
                flags = set(line.split(":", 1)[1].split())
                break
    if "avx512f" in flags:
        return 16
    if flags & {"avx", "avx2"} or (not flags and platform.machine() in ("x86_64", "AMD64")):
        return 8
    return 4


def contraction_block(depth: int, lanes: int | None = None) -> int:
    """The products a block sums in one chain, for a contraction `depth` long: the longest power of two
    whose rounding path, `block` FMAs and `ceil(log2(blocks))` adds, is no longer than a vectorized dot's
    on this CPU, `ceil(depth / partials)` FMAs and `ceil(log2(partials))` adds over its `partials` (its
    `vector_lanes` times `FMA_LATENCY`). A depth one block holds, or none, is the plain dot's."""
    if depth <= 1:
        return max(depth, 1)
    partials = min(depth, (lanes or vector_lanes()) * FMA_LATENCY)
    budget = -(-depth // partials) + math.ceil(math.log2(partials))
    block = 1 << (depth.bit_length() - 1)
    while block > 1 and block + math.ceil(math.log2(-(-depth // block))) > budget:
        block //= 2
    return block


@functools.cache
def workspace_bytes() -> int:
    """The bytes a product's blocks' sums may hold at once (`_blocked`): a 64th of this machine's memory,
    or 256 MiB where it states none."""
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 64
    except (ValueError, OSError):
        return 256 << 20


def _tree(parts: jax.Array) -> jax.Array:
    """`parts`' sum over its axis 1, pairwise: each level adds neighbours, an odd one out carried up."""
    while parts.shape[1] > 1:
        carried = parts[:, -1:] if parts.shape[1] % 2 else None
        even = parts[:, :parts.shape[1] - (carried is not None)]
        parts = even[:, 0::2] + even[:, 1::2]
        if carried is not None:
            parts = jnp.concatenate([parts, carried], axis=1)
    return parts[:, 0]


def _summed(a: jax.Array, b: jax.Array, precision) -> jax.Array:
    """`_blocked`'s sum of `a @ b`, `[batch, rows, depth]` and `[batch, depth, columns]` float32 arrays, over
    the shares of the depth an Explicit mesh axis splits across devices (a projection's rows, which its
    matrix's gradient contracts, split over a data axis): each device's share summed as one product
    (`_blocked`), the shares then added across the devices, as the plain product's partials are; a depth no
    axis splits, one share."""
    sharding = jax.typeof(a).sharding
    names = sharding.spec[2] if len(sharding.spec) > 2 else None
    names = () if names is None else names if isinstance(names, tuple) else (names,)
    shares = math.prod(sharding.mesh.shape[name] for name in names)
    if shares == 1:
        return _blocked(a, b, precision)
    batch, rows, depth = a.shape
    columns = b.shape[2]
    # The shares a batch of their own, leading, so each device's lies whole on it.
    a = jnp.moveaxis(a.reshape(batch, rows, shares, depth // shares), 2, 0).reshape(
        shares * batch, rows, depth // shares)
    b = jnp.moveaxis(b.reshape(batch, shares, depth // shares, columns), 1, 0).reshape(
        shares * batch, depth // shares, columns)
    return _blocked(a, b, precision).reshape(shares, batch, rows, columns).sum(axis=0)


def _blocked(a: jax.Array, b: jax.Array, precision) -> jax.Array:
    """`a @ b` of `[batch, rows, depth]` and `[batch, depth, columns]` float32 arrays, the depth in blocks
    (`contraction_block`), each block's dot XLA's own, their sums added as a tree (`_tree`); a zero
    product pads the last block. Where the blocks' sums would hold more than `workspace_bytes`, a run of the
    columns at a time (`lax.map`), the fewest runs that fit: a CPU vocabulary head's 64 blocks at once
    (512 tokens of 151,936 columns) would hold 18.5 GiB. An empty operand's product is the dot's own."""
    depth, columns = a.shape[2], b.shape[2]
    block = contraction_block(depth)
    if block >= depth or 0 in a.shape + b.shape:
        return jax.lax.dot_general(a, b, (((2,), (1,)), ((0,), (0,))), precision=precision,
                                   preferred_element_type=jnp.float32)
    count = -(-depth // block)
    pad = count * block - depth
    a = jnp.pad(a, ((0, 0), (0, 0), (0, pad))).reshape(a.shape[0], a.shape[1], count, block)
    held = a.shape[0] * count * a.shape[1] * columns * 4
    runs = min(columns, max(1, -(-held // workspace_bytes())))
    width = -(-columns // runs)
    b = jnp.pad(b, ((0, 0), (0, pad), (0, runs * width - columns)))
    b = b.reshape(b.shape[0], count, block, runs, width).transpose(3, 0, 1, 2, 4)

    def run(columns_run):
        return _tree(jax.lax.dot_general(a, columns_run, (((3,), (2,)), ((0, 2), (0, 1))),
                                         precision=precision, preferred_element_type=jnp.float32))

    summed = jax.lax.map(run, b) if runs > 1 else run(b[0])[None]
    return jnp.moveaxis(summed, 0, 2).reshape(a.shape[0], a.shape[1], runs * width)[..., :columns]


def _matmul(a: jax.Array, b: jax.Array, precision) -> jax.Array:
    """`_blocked`, its reverse pass blocked over each cotangent's own contraction: the rows' cotangent sums
    over the columns, the matrix's over the rows, each a product of its own (`generation.linear_tangent`,
    Protected: a reassociation would undo the tree it sums by); forward mode and a higher order through
    `_blocked`'s own tangent."""
    from dew.nn.kernels.generation import linear_tangent

    def pulled(saved, operands, cotangent, live):
        left, right = operands
        gradients = {0: lambda: _summed(cotangent, jnp.swapaxes(right, 1, 2), precision),
                     1: lambda: _summed(jnp.swapaxes(left, 1, 2), cotangent, precision)}
        return tuple(gradients[index]() for index in live)

    summed = functools.partial(_summed, precision=precision)
    product = contracted("test_a_cpu_pairwise_dense_product_holds_every_order")(
        linear_tangent(lambda *operands: (summed(*operands), ()), pulled, summed, live_cotangents=True,
                       kind=Protected()))
    return product(a, b)


def pairwise_dot_general(lhs: jax.Array, rhs: jax.Array, dimension_numbers, precision=None) -> jax.Array:
    """`jax.lax.dot_general` of float32 `lhs` and `rhs` into float32, as `_matmul` sums it: each operand
    laid out as `[batch, free, contracted]` and `[batch, contracted, free]`, the result in `dot_general`'s
    order (batch, `lhs`'s free axes, `rhs`'s)."""
    (contracted, against), (batched, batched_rhs) = (tuple(map(tuple, pair)) for pair in dimension_numbers)
    free = tuple(axis for axis in range(lhs.ndim) if axis not in contracted + batched)
    free_rhs = tuple(axis for axis in range(rhs.ndim) if axis not in against + batched_rhs)

    def size(array, axes):
        return math.prod(array.shape[axis] for axis in axes)

    a = jnp.transpose(lhs, batched + free + contracted).reshape(
        size(lhs, batched), size(lhs, free), size(lhs, contracted))
    b = jnp.transpose(rhs, batched_rhs + against + free_rhs).reshape(
        size(rhs, batched_rhs), size(rhs, against), size(rhs, free_rhs))
    return _matmul(a, b, precision).reshape(*(lhs.shape[axis] for axis in batched + free),
                                            *(rhs.shape[axis] for axis in free_rhs))
