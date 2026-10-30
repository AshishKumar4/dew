"""Float-range and nested-derivative contracts of Dew's custom-derivative seams (ledger row 2283, theme
G12): each seam's primal, its forward and reverse derivatives, a forward derivative of its gradient and a
second forward derivative, the last two again under a rematerialized call (`jax.checkpoint`), run at a
normal draw and at draws at float32's exponent edges. Each order runs (a custom derivative's tangent tree
and dtypes as JAX requires: rows 1861 and 1862, a boolean residual's tangent and a cotangent tree of the
wrong length), and each of its leaves is finite wherever the same order's float64 run is a normal float32
(row 1768, a contraction that overflowed before its division). Behaviour alone: no route, structure or bits
are read, and each seam is its public call."""

import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import widened

from dew.nn import precision
from dew.nn.blocks import table_rows
from dew.nn.kernels import recurrence_chunk
from dew.nn.kernels.templates.gemm import template

TINY, LARGEST = float(np.finfo(np.float32).tiny), float(np.finfo(np.float32).max)


class LessFinite(AssertionError):
    """A seam's order not finite where its float64 run is a normal float32 (and the plain formula's is
    finite): what the contract refuses, apart from an order that does not run."""


def normal_float32(values) -> np.ndarray:
    """Whether each float64 value is a float32 a few roundings inside its normal range: from just above
    the least normal to just below the largest, or zero."""
    size = np.abs(np.asarray(values, np.float64))
    return (size == 0) | ((size >= TINY * (1 + 2.0**-16)) & (size <= LARGEST * (1 - 2.0**-16)))


def pulled(fn, primals, cotangent):
    """`fn`'s reverse derivative at `primals` pulling back `cotangent` in the output's own dtype."""
    out, pull = jax.vjp(fn, *primals)
    return pull(jax.tree.map(lambda given, made: jnp.asarray(given, made.dtype), cotangent, out))


def orders(fn, primals, direction, cotangent):
    """`fn` at each order the contract holds, by name: the primal, a forward and a reverse derivative, a
    forward derivative of the gradient of the output's sum against `cotangent` and a second forward
    derivative, the last two again with `fn` rematerialized; and the gradient's forward derivative traced
    as programs nest it (ledger 2641): under a jit within a jit, within a scan's body, and a reverse
    derivative of the gradient along `direction` (reverse over reverse)."""
    def scalar(f):
        return lambda *x: jnp.sum(f(*x) * cotangent)

    def forward(f):
        return lambda *x: jax.jvp(f, x, direction)[1]

    def gradient(f):
        return jax.grad(scalar(f), argnums=tuple(range(len(primals))))

    remat = jax.checkpoint(fn)
    return {
        "primal": lambda: fn(*primals),
        "jvp": lambda: forward(fn)(*primals),
        "vjp": lambda: pulled(fn, primals, cotangent),
        "jvp of grad": lambda: jax.jvp(gradient(fn), primals, direction)[1],
        "second jvp": lambda: jax.jvp(forward(fn), primals, direction)[1],
        "jvp of grad, remat": lambda: jax.jvp(gradient(remat), primals, direction)[1],
        "second jvp, remat": lambda: jax.jvp(forward(remat), primals, direction)[1],
        "jvp of grad, nested jit": lambda: jax.jit(
            lambda *x: jax.jvp(jax.jit(gradient(fn)), x, direction)[1])(*primals),
        "jvp of grad, scanned": lambda: jax.lax.scan(
            lambda carry, _: (carry, jax.jvp(gradient(fn), primals, direction)[1]), 0, None, length=1)[1],
        "grad of grad": lambda: jax.grad(
            lambda *x: sum(jnp.sum(jnp.asarray(along, part.dtype) * part)
                           for along, part in zip(direction, gradient(fn)(*x), strict=True)),
            argnums=tuple(range(len(primals))))(*primals),
    }


def ran(order, label: str, name: str):
    """`order()`'s leaves as arrays, a failure to run named by the seam and the order that raised it."""
    try:
        return jax.tree.map(np.asarray, order())
    except Exception as error:
        raise AssertionError(f"{label}: its {name} does not run: {type(error).__name__}: {error}") from error


def assert_contract(fn, primals, direction, cotangent, label: str, *, truth_fn=None, plain_fn=None,
                    held: tuple[str, ...] | None = None) -> None:
    """Every order of `fn` runs at the draw, and each leaf is finite wherever the same order of its float64
    run is a normal float32 and, given `plain_fn` (the plain float32 formula, JAX's own derivatives of it),
    the plain formula's is finite: no less finite than the plain arithmetic (row 1768, a contraction Dew
    overflowed where the plain formula did not), where float32 itself overflows a partial sum the float64
    run cancels. The float64 run is `fn`'s own, or `truth_fn`'s, the plain float64 formula, where `fn`
    computes in a fixed precision whatever its inputs' (a declared dot algorithm). `held`, where given, names
    the orders a decision its caller cites holds the call to: each order it drops must break, so the
    exemption ends when the call holds it."""
    narrow = orders(fn, primals, direction, cotangent)
    plain_orders = {} if plain_fn is None else orders(plain_fn, primals, direction, cotangent)
    plain = {name: ran(order, f"{label}, plainly", name) for name, order in plain_orders.items()}
    with jax.enable_x64(new_val=True):
        wide_primals, wide_direction, wide_cotangent = widened((primals, direction, cotangent))
        wide = orders(truth_fn or fn, wide_primals, wide_direction, wide_cotangent)
        truths = {name: ran(order, f"{label} in float64", name) for name, order in wide.items()}
    for name, order in narrow.items():
        values = jax.tree.leaves(ran(order, label, name))
        plains = jax.tree.leaves(plain[name]) if plain else [None] * len(values)
        broken = []
        for index, (value, truth, plainly) in enumerate(zip(values, jax.tree.leaves(truths[name]), plains,
                                                            strict=True)):
            if not np.issubdtype(value.dtype, np.inexact):
                continue
            kept = normal_float32(truth) & np.isfinite(truth)
            if plainly is not None:
                kept &= np.isfinite(plainly)
            lost = kept & ~np.isfinite(value)
            if lost.any():
                broken.append(f"{label}, {name}, leaf {index}: {int(lost.sum())} values not finite where "
                              f"float64's are normal float32s, e.g. {value[lost][:3]}")
        if held is not None and name not in held:
            if not broken:
                raise AssertionError(f"{label}, {name}: holds, so the exemption that drops it is over")
        elif broken:
            raise LessFinite(broken[0])


EDGES = {"normal": 1.0, "near the largest": 2.0**63, "near the least normal": 2.0**-63}
"""The draws' scales: unit values, and values whose products (a squared distance, a second derivative's
pairs) reach float32's largest or its least normal."""


DELTA_RULE_EDGES = {"normal": 1.0, "the largest its chunk holds": 2.0**62, "near the largest": 2.0**63,
                    "near the least normal": 2.0**-63}
"""The chunked delta rule's edges. Near float32's largest its chunk's own products overflow at entries the
token recurrence keeps finite, as fla's chunked form's do at the same entries (`precedent_chunked`, on an
H100: ledger 2470), which that edge compares it with in place of the recurrence."""

PACKED_OVERFLOW_ORDERS = ("primal", "jvp", "vjp", "second jvp", "second jvp, remat")
"""The orders a packed row of the chunked delta rule holds from 2^62: all but the second orders through its
gradient, forward over reverse and reverse over reverse, which the owner's 10-30 decision accepts there (one
cause, documents sharing chunks, an overflow-only difference: ledger 2470, Main 10-30 on reverse over
reverse; `test_the_chunked_delta_rule_holds_every_order`)."""


def precedent_chunked(query, key, value, g, beta, *, chunk_size: int, starts=None):
    """fla's upstream chunked gated delta rule as it is written (`naive_chunk_gated_delta_rule`,
    fla/ops/gated_delta_rule/naive.py), in JAX for its derivatives, over each packed document alone as fla
    runs a varlen batch's sequences: the precedent the chunked rule's largest edge compares with. Its row
    correction one row at a time and its masks filled after their products, as fla's are; `g` `[B, S, H, 1]`,
    from a zero memory."""
    if starts is not None:
        begun = np.flatnonzero(np.asarray(starts)[0])
        cuts = [0, *begun[begun > 0].tolist(), query.shape[1]]
        pieces = [precedent_chunked(*(x[:, a:b] for x in (query, key, value, g, beta)), chunk_size=chunk_size)
                  for a, b in itertools.pairwise(cuts) if b > a]
        return jnp.concatenate(pieces, axis=1)
    batch, length, heads, width = query.shape
    pad = (-length) % chunk_size
    q, k, v = (jnp.pad(jnp.moveaxis(x, 1, 2), [(0, 0), (0, 0), (0, pad), (0, 0)])
               for x in (query, key, value))
    beta, g = (jnp.pad(jnp.moveaxis(x, 1, 2), [(0, 0), (0, 0), (0, pad)]) for x in (beta, g[..., 0]))
    q, v, keyed = q * width ** -0.5, v * beta[..., None], k * beta[..., None]
    chunks = (length + pad) // chunk_size
    q, k, v, keyed = (x.reshape(batch, heads, chunks, chunk_size, -1) for x in (q, k, v, keyed))
    decay = jnp.cumsum(g.reshape(batch, heads, chunks, chunk_size), axis=-1)
    upper = jnp.triu(jnp.ones((chunk_size, chunk_size), bool))
    weights = jnp.tril(jnp.exp(jnp.tril(decay[..., :, None] - decay[..., None, :])))
    system = -jnp.where(upper, 0.0, (keyed @ jnp.swapaxes(k, -1, -2)) * weights)
    for row in range(1, chunk_size):
        system = system.at[..., row, :row].set(
            system[..., row, :row] + jnp.sum(system[..., row, :row, None] * system[..., :row, :row], axis=-2))
    system = system + jnp.eye(chunk_size)
    v, written = system @ v, system @ (keyed * jnp.exp(decay)[..., None])
    memory = jnp.zeros((batch, heads, width, v.shape[-1]))
    strict = jnp.triu(jnp.ones((chunk_size, chunk_size), bool), 1)
    outputs = []
    for index in range(chunks):
        scores = jnp.where(strict, 0.0, (q[:, :, index] @ jnp.swapaxes(k[:, :, index], -1, -2))
                           * weights[:, :, index])
        corrected = v[:, :, index] - written[:, :, index] @ memory
        read = (q[:, :, index] * jnp.exp(decay[:, :, index, :, None])) @ memory
        outputs.append(read + scores @ corrected)
        last = decay[:, :, index, -1]
        memory = memory * jnp.exp(last)[..., None, None] + jnp.swapaxes(
            k[:, :, index] * jnp.exp(last[..., None] - decay[:, :, index])[..., None], -1, -2) @ corrected
    out = jnp.stack(outputs, axis=2).reshape(batch, heads, chunks * chunk_size, -1)[:, :, :length]
    return jnp.moveaxis(out, 1, 2)


@pytest.mark.parametrize(("packed", "edge"), list(itertools.product((False, True), DELTA_RULE_EDGES)),
                         ids=lambda value: {False: "one sequence", True: "packed"}.get(value, value))
def test_the_chunked_delta_rule_holds_every_order(packed, edge):
    """`recurrence_chunk.chunk_delta_rule` (the portable rule's custom derivative through
    `generation.linear_tangent`, rows 1861 and 1862): every order runs, its boolean document starts a
    residual with no tangent space, and each is finite wherever float64 is a normal float32 and JAX's own
    derivatives of the token recurrence are (`recurrent_delta_rule`, an independent formula of the same
    rule, in float64 the truth too: the chunk's own float64 run excused its float32 NaNs wherever it was NaN
    itself), near float32's largest of fla's own chunked form (`precedent_chunked`), at queries, keys and
    values whose products near float32's ends, the decays and gates as drawn (`DELTA_RULE_EDGES`). A pair
    the chunk weighs zero once multiplied its overflowed tangent by zero, the queries' second derivative NaN
    from 2^48 (ledger 2470).

    A packed row's documents share chunks, so from 2^62 one document's overflow reaches another's forward
    over reverse (ledger 2608); its other orders hold. The owner accepted that under the accuracy-for-speed
    rule (ledger 2470, 10-30): documents in chunks of their own held every order and cost the portable
    packed gradient 25% (JS-cost2470d-h100-80113cc1c) for a difference at overflow alone, where the
    parameters' gradient is NaN either way."""
    keys = jax.random.split(jax.random.key(1861), 8)
    rows, length, heads, width = 1, 37, 2, 8
    scale = DELTA_RULE_EDGES[edge]
    query, key, value = (jax.random.normal(k, (rows, length, heads, width)) * scale for k in keys[:3])
    g = -jax.nn.softplus(jax.random.normal(keys[3], (rows, length, heads, 1)))
    beta = jax.nn.sigmoid(jax.random.normal(keys[4], (rows, length, heads)))
    starts = jax.random.bernoulli(keys[5], 0.1, (rows, length)) if packed else None

    def rule(query, key, value, g, beta):
        return recurrence_chunk.chunk_delta_rule(query, key, value, g, beta, chunk_size=16, starts=starts)[0]

    def recurrence(query, key, value, g, beta):
        return recurrence_chunk.recurrent_delta_rule(query, key, value, g, beta, starts=starts)[0]

    plain = (functools.partial(precedent_chunked, chunk_size=16, starts=starts) if edge == "near the largest"
             else recurrence)
    primals = (query, key, value, g, beta)
    direction = tuple(jax.random.normal(keys[6], x.shape) * (scale if at < 3 else 1.0)
                      for at, x in enumerate(primals))
    cotangent = jax.random.normal(keys[7], query.shape)
    # A packed row's second orders through its gradient from 2^62: the owner's accuracy-for-speed acceptance
    # (ledger 2470, 10-30).
    accepted = packed and scale >= 2.0**62
    assert_contract(rule, primals, direction, cotangent, f"the chunked delta rule {edge}",
                    truth_fn=recurrence, plain_fn=plain, held=PACKED_OVERFLOW_ORDERS if accepted else None)


def normal_draw(seed: int, *shapes, scale: float = 1.0) -> tuple:
    """Standard normal float32 arrays of `shapes` at `scale`, from `seed`."""
    rng = np.random.default_rng(seed)
    return tuple(jnp.asarray(rng.standard_normal(shape) * scale, jnp.float32) for shape in shapes)


@pytest.mark.parametrize("edge", EDGES)
def test_an_embedding_lookup_holds_every_order(edge):
    """`blocks.table_rows` (a gather whose reverse mode sums each token's cotangent into its row, row
    1869): every order runs in the table, the ids not differentiable, and each is finite wherever float64
    is a normal float32."""
    (table,) = normal_draw(1869, (11, 6), scale=EDGES[edge])
    ids = jnp.asarray([[0, 3, 3, 10], [7, 0, 2, 2]], jnp.int32)
    (direction,) = normal_draw(1870, table.shape, scale=EDGES[edge])
    (cotangent,) = normal_draw(1871, (2, 4, 6))
    assert_contract(lambda table: table_rows(table, ids, table.dtype), (table,), (direction,), cotangent,
                    f"an embedding lookup {edge}")


@pytest.mark.parametrize("edge", EDGES)
def test_a_bf16_dot_holds_every_order(edge):
    """`precision.bf16_dot_general` (the BF16_BF16_F32 algorithm's dot and its cotangent's one rounding
    to bf16): every order runs, finite wherever float64 is a normal float32, at values whose products
    near float32's ends."""
    primals = normal_draw(956, (5, 8), (8, 3), scale=EDGES[edge])
    direction = normal_draw(957, (5, 8), (8, 3), scale=EDGES[edge])
    (cotangent,) = normal_draw(958, (5, 3))

    def dot(lhs, rhs):
        return precision.bf16_dot_general(lhs, rhs, (((1,), (0,)), ((), ())))

    def exact(lhs, rhs):
        # The operands as the algorithm reads them, rounded to bf16, then multiplied in float64.
        return jnp.matmul(*(jnp.asarray(jnp.asarray(x, jnp.bfloat16), x.dtype) for x in (lhs, rhs)))

    def plain(lhs, rhs):
        # The operands rounded to bf16, their products summed in float32 by XLA's own dot.
        return jnp.matmul(*(jnp.asarray(jnp.asarray(x, jnp.bfloat16), x.dtype) for x in (lhs, rhs)),
                          preferred_element_type=jnp.float32)

    assert_contract(dot, primals, direction, cotangent, f"a bf16 dot {edge}", truth_fn=exact,
                    plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_a_wide_cpu_sum_of_products_holds_every_order(monkeypatch, edge):
    """`precision.accumulated` in a process running wide CPU arithmetic (`WIDE_CPU_ARITHMETIC`, its own
    custom derivative of the float64-summed product): every order runs, finite wherever float64 is a
    normal float32 and wherever XLA's plain product is, at values whose products near float32's ends."""
    monkeypatch.setenv(precision.WIDE_CPU_ARITHMETIC, "1")
    primals = normal_draw(343, (4, 16), (16, 5), scale=EDGES[edge])
    direction = normal_draw(344, (4, 16), (16, 5), scale=EDGES[edge])
    (cotangent,) = normal_draw(345, (4, 5))
    assert_contract(lambda lhs, rhs: precision.accumulated(jnp.matmul, lhs, rhs), primals, direction,
                    cotangent, f"a wide CPU sum of products {edge}", plain_fn=jnp.matmul)


@pytest.mark.parametrize("edge", EDGES)
def test_a_grouped_gemm_holds_every_order(edge):
    """The GEMM template's grouped product (`template.gemm`, its custom derivative through
    `generation.linear_tangent`): every order runs in the rows and the matrices, the group sizes not
    differentiable, finite wherever float64 is a normal float32 and wherever the plain grouped product is
    (each row's product with its group's matrix, by a mask, its float64 run the truth: XLA's ragged product
    takes no reverse over reverse), at values whose products near float32's ends."""
    sizes = jnp.asarray([3, 0, 5], jnp.int32)
    groups = jnp.repeat(jnp.arange(3), sizes, total_repeat_length=8)
    primals = normal_draw(146, (8, 16), (3, 16, 4), scale=EDGES[edge])
    direction = normal_draw(147, (8, 16), (3, 16, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(148, (8, 4))

    def grouped(rows, matrices):
        return template.gemm(rows, matrices, sizes, out_dtype=rows.dtype)

    def plain(rows, matrices):
        return jnp.einsum("rg,rk,gkn->rn", jax.nn.one_hot(groups, 3, dtype=rows.dtype), rows, matrices)

    assert_contract(grouped, primals, direction, cotangent, f"a grouped GEMM {edge}", truth_fn=plain,
                    plain_fn=plain)


def test_the_contract_refuses_a_formula_less_finite_than_the_plain_one():
    """The rule's validation: a formula overflowing where the plain one does not (|x| as the root of its
    square, past 2**64) is refused, at its primal and its derivatives; the plain formula against itself
    holds, at the same edge, where float32's own arithmetic holds every value."""
    x = (jnp.asarray([2.0**70, -3.0, 2.0**-70], jnp.float32),)
    direction = (jnp.ones((3,), jnp.float32),)
    cotangent = jnp.ones((3,), jnp.float32)
    assert_contract(jnp.abs, x, direction, cotangent, "the plain formula", plain_fn=jnp.abs)
    with pytest.raises(LessFinite):
        assert_contract(lambda x: jnp.sqrt(x * x), x, direction, cotangent, "a squared root",
                        plain_fn=jnp.abs)

