"""Muon's Newton-Schulz orthogonalization (`dew.optim`'s Muon and M3 iterate through it): optax's iteration
over each update's matrices as they are stored (`newton_schulz`, `as_matrices`), a head's alone where a run
names its width (`headed`), each update's stack spread whole across the mesh's devices and walked in place a
chunk of matrices at a time (`orthogonalized`), in the run's numerics or a stated dot algorithm
(`ns_operands`), at Jordan's coefficients or Polar Express's (`polar_express`)."""

import functools
import math
from collections.abc import Callable, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax

from dew.mesh import STAGE_AXIS
from dew.nn.precision import configured_precision, iteration_dtype

Preconditioning = Literal["frobenius", "spectral", "aol", "schatten"]
"""How optax's Newton-Schulz scales a matrix before it iterates (`optax.contrib.muon`'s `preconditioning`)."""


NEWTON_SCHULZ = (3.4445, -4.7750, 2.0315)
"""Muon's quintic Newton-Schulz coefficients, Jordan's (modded-nanogpt), optax's default."""


POLAR_EXPRESS_TABLE = (
    (8.28721201814563, -23.595886519098837, 17.300387312530933),
    (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
    (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
    (3.3184196573706015, -2.488488024314874, 0.51004894012372),
    (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
    (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
    (1.875, -1.25, 0.375))
"""Polar Express's quintic per iteration (arXiv 2505.16932, Appendix A's `coeffs_list`), designed for
singular values in [1e-3, 1], before its safety factor."""


def polar_express(steps: int, safety: float = 1.01) -> tuple[tuple[float, float, float], ...]:
    """Polar Express's coefficients for `steps` iterations as its code applies them (arXiv 2505.16932,
    Appendix A): each but the last triple's polynomial taken at `x / safety`, `(a / s, b / s^3, c / s^5)`,
    the first's again for its input's own scaling, `X / (safety * ||X||_F)`, which optax's iteration, over
    `X / ||X||_F`, then reproduces; the table's first `steps`, its last repeated past it."""
    def scaled(triple, by):
        a, b, c = triple
        return a / by, b / by ** 3, c / by ** 5
    table = [scaled(triple, safety) if index < len(POLAR_EXPRESS_TABLE) - 1 else triple
             for index, triple in enumerate(POLAR_EXPRESS_TABLE)]
    table[0] = scaled(table[0], safety)
    return tuple(table[:steps] + [table[-1]] * (steps - len(table)))


def muon_leaf(node) -> bool:
    """A leaf of a tree of Muon's dimension numbers: a parameter's, or None for AdamW's."""
    return node is None or isinstance(node, optax.contrib.MuonDimensionNumbers)


def headed(stack: jax.Array, width: int | None) -> tuple[jax.Array, Callable[[jax.Array], jax.Array]]:
    """`stack`, `[batch, reduction, output]`, as one matrix a head where `width` is given, its output side
    `width` wide heads, `[batch * heads, reduction, width]`, and the reshape back; else as it is."""
    if width is None:
        return stack, lambda matrices: matrices
    batch, reduction, output = stack.shape
    if output % width:
        raise ValueError(f"an output side of {output} is no stack of heads {width} wide (`per_head`)")
    heads = jnp.transpose(stack.reshape(batch, reduction, output // width, width), (0, 2, 1, 3))
    return heads.reshape(batch * (output // width), reduction, width), lambda matrices: jnp.transpose(
        matrices.reshape(batch, output // width, reduction, width), (0, 2, 1, 3)).reshape(stack.shape)


def matrix_axes(x: jax.Array, numbers: optax.contrib.MuonDimensionNumbers
                 ) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The reduction and output axes `numbers` name of `x`, each a tuple of axes counted from its first."""
    def axes(given: int | Sequence[int]) -> tuple[int, ...]:
        return tuple(axis % x.ndim for axis in ((given,) if isinstance(given, int) else given))
    return axes(numbers.reduction_axis), axes(numbers.output_axis)


def as_matrices(x: jax.Array, numbers: optax.contrib.MuonDimensionNumbers
                 ) -> tuple[jax.Array, Callable[[jax.Array], jax.Array]]:
    """`x` as a stack of its matrices, `[batch, reduction, output]`, and the reshape back, as optax lays a
    parameter out for its Newton-Schulz: its other axes the batch's, in order."""
    reduction, output = matrix_axes(x, numbers)
    batch = tuple(axis for axis in range(x.ndim) if axis not in (*reduction, *output))
    order = (*batch, *reduction, *output)
    moved = jnp.transpose(x, order)
    stack = moved.reshape(*(math.prod(x.shape[axis] for axis in side) for side in (batch, reduction, output)))
    back = tuple(sorted(range(x.ndim), key=order.__getitem__))
    return stack, lambda matrices: jnp.transpose(matrices.reshape(moved.shape), back)


def orthogonalized(updates: optax.Updates, numbers, coeffs: jax.Array, steps: int, eps: float,
                   preconditioning: Preconditioning = "frobenius",
                   algorithm: jax.lax.DotAlgorithmPreset | None = None, widths=None,
                   chunk_bytes: int | None = None) -> optax.Updates:
    """Each update `numbers` makes a stack of matrices of (`as_matrices`) orthogonalized by optax's
    Newton-Schulz on the matrices as they are stored (`newton_schulz`), in the run's numerics or `algorithm`
    (`ns_operands`), as Moonlight distributes it (arXiv 2502.16982, 3): each stack spread across the mesh's
    devices whole, each device iterating its own share, and handed back as the update lies. A matrix split
    across devices would exchange partial products at each of the iteration's products; spread whole, a
    step moves each matrix twice, there and back. A stack its devices divide spreads alone; stacks of one
    shape that they do not are gathered into one, which they do once padded, so no device idles on a
    stack's remainder. Each device walks its share in place a chunk of matrices at a time, as many as
    `chunk_bytes` holds of one's working set (`working_bytes`), its whole share where none is given, so the
    iteration holds the update and one chunk's products, not every matrix's at once (row 5669). On a mesh
    with Explicit axes the spreading runs with them Auto (`jax.sharding.auto_axes`), where a constraint may
    name them and a split stack slices, each update leaving where it came; on one device, or none in
    context, each stack iterates whole. A tree of head `widths` (`head_widths`) splits an update's output
    side into heads, each one matrix."""
    mesh = jax.sharding.get_abstract_mesh()
    spread = functools.partial(_spread_orthogonalized, numbers=numbers, coeffs=coeffs, steps=steps, eps=eps,
                               preconditioning=preconditioning, algorithm=algorithm, chunk_bytes=chunk_bytes,
                               widths=jax.tree.map(lambda _: None, updates) if widths is None else widths)
    if mesh.empty or not mesh.explicit_axes:
        return spread(updates)
    places = jax.tree.map(lambda leaf: jax.typeof(leaf).sharding.spec, updates)
    with_auto = jax.sharding.auto_axes(axes=mesh.explicit_axes, out_sharding=places)
    return with_auto(spread)(updates)


def _spread_orthogonalized(updates, *, numbers, coeffs: jax.Array, steps: int, eps: float,
                           algorithm: jax.lax.DotAlgorithmPreset | None, widths,
                           preconditioning: Preconditioning, chunk_bytes: int | None) -> optax.Updates:
    """`orthogonalized`'s iteration over a mesh of Auto axes: each stack laid out `[devices, share, rows,
    columns]`, its first axis split across every axis of the mesh but a pipeline's stages, so each device
    walks its own share (`_walked`), which no exchange crosses; stacks of a shape the devices do not divide
    gathered and padded with zero matrices first."""
    mesh = jax.sharding.get_abstract_mesh()
    axes = () if mesh.empty else tuple(axis for axis, size in mesh.shape.items()
                                       if size > 1 and axis != STAGE_AXIS)
    devices = math.prod(mesh.shape[axis] for axis in axes)

    def split(stack: jax.Array) -> jax.Array:
        return stack if not axes else jax.lax.with_sharding_constraint(
            stack, jax.sharding.PartitionSpec(axes, None, None, None))

    def iterated(stack: jax.Array) -> jax.Array:
        if algorithm is not None and jnp.iscomplexobj(stack):
            raise ValueError(f"ns_precision={algorithm.name} states real products, which a complex update's "
                             f"Newton-Schulz cannot take: state none, so it multiplies in complex64")
        work = ns_operands(stack.dtype, preconditioning, algorithm)
        shared = split(stack.reshape(devices, stack.shape[0] // devices, *stack.shape[1:]))
        rows, columns = stack.shape[1:]
        count = shared.shape[1] if chunk_bytes is None else max(
            1, chunk_bytes // working_bytes(rows, columns, jnp.dtype(work)))

        def chunk(matrices: jax.Array) -> jax.Array:
            with configured_precision(algorithm):
                return newton_schulz(matrices.astype(work), coeffs, steps, eps, preconditioning
                                     ).astype(stack.dtype)
        return split(_walked(shared, count, chunk)).reshape(stack.shape)

    leaves, structure = jax.tree.flatten(updates)
    stacks = []
    for leaf, shape, width in zip(leaves, jax.tree.leaves(numbers, is_leaf=muon_leaf),
                                  jax.tree.leaves(widths, is_leaf=lambda node: node is None), strict=True):
        matrices, back = as_matrices(leaf, shape)
        heads, apart = headed(matrices, width)
        stacks.append((heads, lambda done, back=back, apart=apart: back(apart(done))))
    done: list[jax.Array | None] = [None] * len(leaves)
    # A stack its devices divide alone; the others of one shape together.
    gathered: dict[tuple[tuple[int, ...], str], list[int]] = {}
    for index, (stack, back) in enumerate(stacks):
        if stack.shape[0] % devices:
            gathered.setdefault((stack.shape[1:], str(stack.dtype)), []).append(index)
        else:
            done[index] = back(iterated(stack))
    for members in gathered.values():
        joined = jnp.concatenate([stacks[index][0] for index in members])
        orthogonal = iterated(jnp.pad(joined, ((0, -joined.shape[0] % devices), (0, 0), (0, 0))))
        starts = np.cumsum([0, *(stacks[index][0].shape[0] for index in members)])
        for index, start, end in zip(members, starts[:-1], starts[1:], strict=True):
            done[index] = stacks[index][1](orthogonal[start:end])
    return jax.tree.unflatten(structure, done)


def _walked(shared: jax.Array, count: int, chunk: Callable[[jax.Array], jax.Array]) -> jax.Array:
    """`shared`, `[devices, share, rows, columns]`, each device's share mapped by `chunk` `count` matrices at
    a time in place: a loop whose carry is the stack, each step's chunk read out and written back where it
    was (XLA updates a carried buffer in place), the share's remainder after it; the whole share at once
    where `count` covers it."""
    share = shared.shape[1]
    if count >= share:
        return chunk(shared)
    whole = share // count

    def step(index, held):
        start = index * count
        return jax.lax.dynamic_update_slice_in_dim(
            held, chunk(jax.lax.dynamic_slice_in_dim(held, start, count, axis=1)), start, axis=1)
    held = jax.lax.fori_loop(0, whole, step, shared)
    if share % count:
        rest = whole * count
        held = jax.lax.dynamic_update_slice_in_dim(held, chunk(held[:, rest:]), rest, axis=1)
    return held


def working_bytes(rows: int, columns: int, dtype: jnp.dtype) -> int:
    """The bytes one `rows` by `columns` matrix's Newton-Schulz step holds at once in `dtype`
    (`newton_schulz`): its iterate, its Gram product on the shorter side, that product's square, the
    polynomial of them, the polynomial applied to the iterate, and the next iterate."""
    side = min(rows, columns)
    return (3 * rows * columns + 3 * side * side) * dtype.itemsize


def newton_schulz(x: jax.Array, coeffs: jax.Array, steps: int, eps: float,
                  preconditioning: Preconditioning = "frobenius") -> jax.Array:
    """optax's Newton-Schulz (`orthogonalize_via_newton_schulz`, optax 0.2.8) of each of `x`'s matrices, its
    last two axes, as they are stored: optax transposes a matrix taller than it is wide, iterates, and
    transposes back, which, since a step satisfies f(X) = f(Xᵀ)ᵀ (optax's own note), is the iteration with
    its Gram product taken on the shorter side, `X Xᴴ` or `Xᴴ X`, and the polynomial applied from that side,
    `B X` or `X B`: so the products contract the stored layout, and no matrix is transposed. Its
    preconditioning optax's: the Frobenius or spectral norm, or the AOL or Schatten-4 rescaling of the first
    step; `coeffs` one triple, `steps` times, or a schedule of them."""
    rows, columns = x.shape[-2:]
    tall = rows > columns
    batch = tuple(range(x.ndim - 2))
    last = x.ndim - 1

    def product(a: jax.Array, b: jax.Array, contract: tuple[int, int]) -> jax.Array:
        return jax.lax.dot_general(a, b, (((contract[0],), (contract[1],)), (batch, batch)))

    def gram(matrices: jax.Array, conjugate: bool = True) -> jax.Array:
        held = matrices.conj() if conjugate else matrices
        return (product(held, matrices, (last - 1, last - 1)) if tall
                else product(matrices, held, (last, last)))

    def applied(polynomial: jax.Array, matrices: jax.Array) -> jax.Array:
        return product(matrices, polynomial, (last, last - 1)) if tall else product(
            polynomial, matrices, (last, last - 1))

    def stepped(matrices: jax.Array, a: jax.Array, triple: jax.Array) -> jax.Array:
        polynomial = triple[1] * a + triple[2] * product(a, a, (last, last - 1))
        return triple[0] * matrices + applied(polynomial, matrices)

    def base(matrices: jax.Array, triple: jax.Array) -> jax.Array:
        return stepped(matrices, gram(matrices), triple)

    def aol(matrices: jax.Array, triple: jax.Array) -> jax.Array:
        a = gram(matrices)
        scale = jax.lax.rsqrt(jnp.clip(jnp.abs(a).sum(axis=-1), min=eps))
        side = scale[..., None, :] if tall else scale[..., :, None]
        return stepped(matrices * side, a * scale[..., :, None] * scale[..., None, :], triple)

    def schatten(matrices: jax.Array, triple: jax.Array) -> jax.Array:
        a = gram(matrices, conjugate=False)
        scale = jax.lax.rsqrt(jnp.clip(jnp.linalg.norm(a, axis=(-2, -1), keepdims=True), min=eps))
        return stepped(matrices * scale, a * scale ** 2, triple)

    if preconditioning == "frobenius":
        x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)
    elif preconditioning == "spectral":
        x = x / (jnp.linalg.norm(x, ord=2, axis=(-2, -1), keepdims=True) + eps)
    first = {"aol": aol, "schatten": schatten}.get(preconditioning, base)
    triples = coeffs.astype(x.dtype)
    if triples.ndim == 1:
        triples = jnp.broadcast_to(triples, (steps, 3))
    x = first(x, triples[0])
    for triple in range(1, triples.shape[0]):
        x = base(x, triples[triple])
    return x


def dot_algorithm(named: jax.lax.DotAlgorithmPreset | str | None) -> jax.lax.DotAlgorithmPreset | None:
    """A dot algorithm as a record or a flag names it (`Numerics.matmul`'s vocabulary), or None."""
    return jax.lax.DotAlgorithmPreset[named] if isinstance(named, str) else named


def ns_operands(dtype, preconditioning: Preconditioning = "frobenius",
                algorithm: jax.lax.DotAlgorithmPreset | None = None) -> jnp.dtype:
    """The dtype Muon's Newton-Schulz takes its operands in: an `algorithm`'s own, where one states the
    iteration's products (`BF16_BF16_F32`'s bfloat16, `TF32_TF32_F32`'s and `F32_F32_F32`'s float32, which a
    platform that has not the algorithm refuses in JAX's own words, and a complex update, whose products no
    real algorithm holds, refuses by name); else the run's numerics' (`dew.nn.kernels.generation.numerics`),
    each product at its precision (`Numerics.matmul`), as `dew.nn.precision.iteration_dtype` declares it:
    in its fast forms bfloat16 for a 16-bit compute dtype (16-mixed's float16 among them), as Jordan's Muon
    (modded-nanogpt) and Moonlight (arXiv 2502.16982) iterate, and the compute dtype otherwise; at least
    float32 of it in its accurate ones, TF32 products on a GPU
    at the default precision and IEEE float32 under the accurate preset (`precision="accurate"`);
    complex64 for a complex update, or its own complex128; and at least float32 of the iteration's dtype for
    a spectral preconditioning, whose norm is an SVD of at least float32 operands (a float64 run's in
    float64).

    The fast forms' bfloat16, where optax iterates in its momentum's own float32: the quintic brings the
    singular values only near one, roughly within 0.7 to 1.2 (optax `orthogonalize_via_newton_schulz`), far
    coarser than bfloat16's rounding. A Qwen3.5-4B decision fine-tune on one H100 (node job LR-h100-muon-2752c
    at 699d038ca, 8192 tokens a step) stepped in 844 ms with its iteration in bfloat16 (`BF16_BF16_F32`), 1287
    ms at TF32 (`TF32_TF32_F32`) and 4641 ms at IEEE float32 (`F32_F32_F32`), against AdamW's 495 ms, its loss
    at each of 30 steps within 0.07 of the float32 iteration's (0.02 on average). Accumulated, its update's
    microbatch takes 860 ms against 516 and the others about AdamW's: an update of 65,536 tokens 9.4% over
    AdamW's and one of 262,144 2.5% (`dew.optim.muon`; ledger 2685, 2752)."""
    if jnp.issubdtype(dtype, jnp.complexfloating):
        return jnp.promote_types(dtype, jnp.complex64)
    if preconditioning == "spectral":
        return jnp.promote_types(iteration_dtype(), jnp.float32)
    if algorithm is not None and algorithm.supported_lhs_types:
        return jnp.dtype(algorithm.supported_lhs_types[0])
    return iteration_dtype()


def orthogonalized_leaf(update: jax.Array, steps: int,
                    dimension_numbers: optax.contrib.MuonDimensionNumbers | None, *,
                    coeffs=NEWTON_SCHULZ, eps: float = 1e-8,
                    preconditioning: Preconditioning = "frobenius") -> jax.Array:
    """`update` orthogonalized by Muon's `steps` quintic Newton-Schulz iterations at `coeffs`, one triple
    or a schedule of them, after `preconditioning`, optax's own: its Muon transform with no momentum
    (`scale_by_muon`, `beta` 0 and no Nesterov step) hands back the update it is given orthogonalized, in
    its own dtype, bit for bit its `orthogonalize_via_newton_schulz`, which optax 0.2.8 does not export.
    A leaf with no `dimension_numbers` is one matrix, its rows the reduction axis (a Linen kernel's), its
    other axes a stack of such matrices."""
    numbers = optax.contrib.MuonDimensionNumbers() if dimension_numbers is None else dimension_numbers
    muon = optax.contrib.scale_by_muon(coeffs, steps, beta=0.0, eps=eps, nesterov=False,
                                       preconditioning=preconditioning, weight_dimension_numbers=numbers)
    orthogonal, _ = muon.update(update, muon.init(update))
    assert isinstance(orthogonal, jax.Array), "one array's update is an array"
    return orthogonal


__all__ = ["NEWTON_SCHULZ", "POLAR_EXPRESS_TABLE", "Preconditioning", "newton_schulz", "ns_operands",
           "orthogonalized", "orthogonalized_leaf", "polar_express", "working_bytes"]
