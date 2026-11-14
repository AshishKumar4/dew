"""Muon's Newton-Schulz orthogonalization (`dew.optim`'s Muon and M3 iterate through it): optax's iteration
over each update's matrices as they are stored (`newton_schulz`, `as_matrices`), a head's alone where a run
names its width (`headed`), each update's stack spread whole across the mesh's devices and walked in place a
chunk of matrices at a time (`orthogonalized`), in the run's numerics or a stated dot algorithm
(`ns_operands`), at Jordan's coefficients or Polar Express's (`polar_express`)."""

import functools
import inspect
import math
from collections.abc import Callable, Sequence
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
import optax
from optax.contrib import _muon

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


_FIRST_STEP_EPS = inspect.signature(_muon._aol_first_newton_schulz_iteration).parameters["eps"].default
"""The clamp optax's AOL and Schatten first steps take on their rescaling: their own default, which optax's
iterators call them at, whatever the transform's `eps` (optax 0.2.8)."""


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


def as_matrices(x: jax.Array, numbers: optax.contrib.MuonDimensionNumbers
                ) -> tuple[jax.Array, Callable[[jax.Array], jax.Array]]:
    """`x` as a stack of its matrices, `[batch, reduction, output]`, and the reshape back: optax's own layout
    for its Newton-Schulz (`_compute_muon_reshape`), its other axes the batch's, in order."""
    reshape, back = _muon._compute_muon_reshape(x, numbers)
    return reshape(x), back


def orthogonalized(updates: optax.Updates, numbers, coeffs, steps: int, eps: float,
                   preconditioning: Preconditioning = "frobenius",
                   algorithm: jax.lax.DotAlgorithmPreset | None = None, widths=None,
                   group: int | None = None) -> optax.Updates:
    """Each update `numbers` makes a stack of matrices of (`as_matrices`) orthogonalized by optax's
    Newton-Schulz on the matrices as they are stored (`newton_schulz`), in the run's numerics or `algorithm`
    (`ns_operands`), as Moonlight distributes it (arXiv 2502.16982, 3): each stack spread across the mesh's
    devices whole, each device iterating its own share, and handed back as the update lies. A matrix split
    across devices would exchange partial products at each of the iteration's products; spread whole, a
    step moves each matrix twice, there and back. The stacks of one shape are joined `group` at a time, all
    where None (`spread_stacks`), and padded so no device idles on a remainder: on a GH200 a 4B decision
    step's update ran its 222 TFLOP of products as 31 joined stacks at 0.74 of peak and as 317 stacks one at
    a time (`group=1`) at 0.58, 82 ms a step slower, where the joined copies held 11.3 GB more at the step's
    peak (row 5669); a run states its `group` (`OptimConfig`'s `ns_group`), which no plan sets. On a mesh
    with Explicit axes the spreading runs with them Auto (`jax.sharding.auto_axes`), where a constraint may
    name them and a split stack slices, each update leaving where it came; on one device, or none in
    context, each stack iterates whole. A tree of head `widths` (`head_widths`) splits an update's output
    side into heads, each one matrix."""
    mesh = jax.sharding.get_abstract_mesh()
    spread = functools.partial(_spread_orthogonalized, numbers=numbers, coeffs=coeffs, steps=steps, eps=eps,
                               preconditioning=preconditioning, algorithm=algorithm, group=group,
                               widths=jax.tree.map(lambda _: None, updates) if widths is None else widths)
    if mesh.empty or not mesh.explicit_axes:
        return spread(updates)
    places = jax.tree.map(lambda leaf: jax.typeof(leaf).sharding.spec, updates)
    with_auto = jax.sharding.auto_axes(axes=mesh.explicit_axes, out_sharding=places)
    return with_auto(spread)(updates)


def _spread_orthogonalized(updates, *, numbers, coeffs, steps: int, eps: float,
                           algorithm: jax.lax.DotAlgorithmPreset | None, widths,
                           preconditioning: Preconditioning, group: int | None) -> optax.Updates:
    """`orthogonalized`'s iteration over a mesh of Auto axes: each stack laid out `[devices, share, rows,
    columns]`, its first axis split across every axis of the mesh but a pipeline's stages, so each device
    iterates its own share, which no exchange crosses; the stacks of one shape joined `group` at a time
    (`spread_stacks`)."""
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
        # Its devices' shares as one stack, each device's still its own (its first axis the major one), so
        # each product batches over one axis, as optax's do.
        flat = shared.reshape(-1, *shared.shape[2:]).astype(work)
        with configured_precision(algorithm):
            done = newton_schulz(flat, coeffs, steps, eps, preconditioning)
        return split(done.astype(stack.dtype).reshape(shared.shape)).reshape(stack.shape)

    leaves, structure = jax.tree.flatten(updates)
    stacks = []
    for leaf, shape, width in zip(leaves, jax.tree.leaves(numbers, is_leaf=muon_leaf),
                                  jax.tree.leaves(widths, is_leaf=lambda node: node is None), strict=True):
        matrices, back = as_matrices(leaf, shape)
        heads, apart = headed(matrices, width)
        stacks.append((heads, lambda done, back=back, apart=apart: back(apart(done))))
    done: list[jax.Array | None] = [None] * len(leaves)
    for members in spread_stacks([stack for stack, _ in stacks], group):
        if len(members) == 1 and not stacks[members[0]][0].shape[0] % devices:
            (index,) = members
            done[index] = stacks[index][1](iterated(stacks[index][0]))
            continue
        joined = jnp.concatenate([stacks[index][0] for index in members])
        orthogonal = iterated(jnp.pad(joined, ((0, -joined.shape[0] % devices), (0, 0), (0, 0))))
        starts = np.cumsum([0, *(stacks[index][0].shape[0] for index in members)])
        for index, start, end in zip(members, starts[:-1], starts[1:], strict=True):
            done[index] = stacks[index][1](orthogonal[start:end])
    return jax.tree.unflatten(structure, done)


def spread_stacks(stacks: Sequence[jax.Array | jax.ShapeDtypeStruct], group: int | None) -> list[list[int]]:
    """The stacks `orthogonalized` iterates as one, by their places in `stacks`: those of one matrix shape and
    dtype together, `group` of them at a time (all where None), each set joined into one stack (a copy) and
    padded with zero matrices to a multiple of the devices it spreads over, a set of one its devices divide
    iterated as it lies. Every product of a set is one batched product: the fewer sets, the fewer and larger
    the products and the more the copies hold (row 5669)."""
    if group is not None and group < 1:
        raise ValueError(f"Muon groups at least one stack, got {group}")
    alike: dict[tuple[tuple[int, ...], str], list[int]] = {}
    for index, stack in enumerate(stacks):
        alike.setdefault((tuple(stack.shape[1:]), str(stack.dtype)), []).append(index)
    return [members[start:start + (group or len(members))] for members in alike.values()
            for start in range(0, len(members), group or len(members))]


def newton_schulz_flops(matrices: int, rows: int, columns: int, steps: int) -> float:
    """The FLOPs of `steps` iterations of `newton_schulz` over `matrices` matrices of `rows` by `columns`, its
    products' (each multiply and add two): of a short side m and a long side n, the Gram product `2 m² n`, its
    square `2 m³` and the polynomial applied to the iterate `2 m² n` an iteration."""
    short, long = sorted((rows, columns))
    return float(matrices * steps * (4 * short * short * long + 2 * short ** 3))


def newton_schulz(x: jax.Array, coeffs, steps: int, eps: float,
                  preconditioning: Preconditioning = "frobenius") -> jax.Array:
    """optax's Newton-Schulz (`orthogonalize_via_newton_schulz`, optax 0.2.8) of each of `x`'s matrices, its
    last two axes, as they are stored: optax transposes a matrix taller than it is wide, iterates, and
    transposes back, which, since a step satisfies f(X) = f(Xᵀ)ᵀ (optax's own note), is the iteration with
    its Gram product taken on the shorter side, `X Xᴴ` or `Xᴴ X`, and the polynomial applied from that side,
    `B X` or `X B`: so the products contract the stored layout, and no matrix is transposed. Its
    preconditioning optax's: the Frobenius or spectral norm, or the AOL or Schatten-4 rescaling of the first
    step; `coeffs` one triple, `steps` times, or a schedule of them, rounded to the iteration's dtype as optax
    rounds them: constants of the program where the caller's are (`dew.optim.scale_by_muon`'s, not its
    state's), which a product's scaling and its added term may fold into (`a X + B X` one product's
    epilogue; with them read from the state at run time, the scalings ran as passes of their own, 50 ms of
    row 5669's step)."""
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
        scale = jax.lax.rsqrt(jnp.clip(jnp.abs(a).sum(axis=-1), min=_FIRST_STEP_EPS))
        side = scale[..., None, :] if tall else scale[..., :, None]
        return stepped(matrices * side, a * scale[..., :, None] * scale[..., None, :], triple)

    def schatten(matrices: jax.Array, triple: jax.Array) -> jax.Array:
        a = gram(matrices, conjugate=False)
        scale = jax.lax.rsqrt(jnp.clip(jnp.linalg.norm(a, axis=(-2, -1), keepdims=True), min=_FIRST_STEP_EPS))
        return stepped(matrices * scale, a * scale ** 2, triple)

    firsts = {"frobenius": base, "spectral": base, "aol": aol, "schatten": schatten}
    if preconditioning not in firsts:
        raise ValueError(f"Muon's preconditioning is one of {sorted(firsts)}, got {preconditioning!r}")
    if preconditioning == "frobenius":
        x = x / (jnp.linalg.norm(x, axis=(-2, -1), keepdims=True) + eps)
    elif preconditioning == "spectral":
        x = x / (jnp.linalg.norm(x, ord=2, axis=(-2, -1), keepdims=True) + eps)
    table = jnp.asarray(coeffs).astype(x.dtype)
    triples = [table] * steps if table.ndim == 1 else [table[index] for index in range(table.shape[0])]
    for index, triple in enumerate(triples):
        x = (firsts[preconditioning] if index == 0 else base)(x, triple)
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


__all__ = ["NEWTON_SCHULZ", "POLAR_EXPRESS_TABLE", "Preconditioning", "newton_schulz", "newton_schulz_flops",
           "ns_operands", "orthogonalized", "polar_express", "spread_stacks"]
