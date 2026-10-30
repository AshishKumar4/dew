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

from dew.diffusion.schedules.flow import time_shift
from dew.interop import torchax_fallback
from dew.nn import attention, categorical, kv_cache, precision
from dew.nn.activations import exact_gelu
from dew.nn.blocks import table_rows
from dew.nn.kernels import fla_gdn, fla_kda, mamba_ssd, recurrence, recurrence_chunk, short_conv, ssd
from dew.nn.kernels.templates.gemm import template
from dew.nn.kernels.templates.recurrence import mainloop
from dew.nn.protocols import OutputTable
from dew.objectives import base, residuals
from dew.objectives.lm import chunked

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


FORWARD_ORDERS = ("jvp", "second jvp", "jvp of grad", "jvp of grad, remat", "second jvp, remat",
                  "jvp of grad, nested jit", "jvp of grad, scanned")
"""The orders a forward derivative takes part in: those a custom VJP, which has no forward mode, does not
take (a row names each such seam's gap)."""
GRADIENT_ORDERS = ("jvp of grad", "jvp of grad, remat", "jvp of grad, nested jit", "jvp of grad, scanned",
                   "grad of grad")
"""The orders that differentiate a gradient."""
REVERSE_ORDERS = ("vjp", *GRADIENT_ORDERS)
"""The orders a reverse derivative takes part in: those a forward mode alone, whose reverse mode is another
seam's, does not take."""


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
                    held: tuple[str, ...] | None = None, without: tuple[str, ...] = ()) -> None:
    """Every order of `fn` runs at the draw, and each leaf is finite wherever the same order of its float64
    run is a normal float32 and, given `plain_fn` (the plain float32 formula, JAX's own derivatives of it),
    the plain formula's is finite: no less finite than the plain arithmetic (row 1768, a contraction Dew
    overflowed where the plain formula did not), where float32 itself overflows a partial sum the float64
    run cancels. The float64 run is `fn`'s own, or `truth_fn`'s, the plain float64 formula, where `fn`
    computes in a fixed precision whatever its inputs' (a declared dot algorithm). `held`, where given, names
    the orders a decision its caller cites holds the call to: each order it drops must break, so the
    exemption ends when the call holds it. `without` names orders the contract does not take yet, each
    joining with its row's fix (a row's head brings its own test)."""

    def orders_of(f, *drawn):
        return {name: order for name, order in orders(f, *drawn).items() if name not in without}

    narrow = orders_of(fn, primals, direction, cotangent)
    plain_orders = {} if plain_fn is None else orders_of(plain_fn, primals, direction, cotangent)
    plain = {name: ran(order, f"{label}, plainly", name) for name, order in plain_orders.items()}
    with jax.enable_x64(new_val=True):
        wide_primals, wide_direction, wide_cotangent = widened((primals, direction, cotangent))
        wide = orders_of(truth_fn or fn, wide_primals, wide_direction, wide_cotangent)
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

    # Its reverse over reverse does not run (ledger 2650); that row's head adds the order with its fix.
    assert_contract(grouped, primals, direction, cotangent, f"a grouped GEMM {edge}", truth_fn=plain,
                    plain_fn=plain, without=("grad of grad",))


@pytest.mark.parametrize("edge", EDGES)
def test_an_exact_gelu_holds_every_order(edge):
    """`activations.exact_gelu` (rounded once to its dtype in both directions): every order finite wherever
    float64 is a normal float32 and the plain formula, x (1 + erf(x / sqrt 2)) / 2, is."""
    (x,) = normal_draw(2641, (4, 7), scale=EDGES[edge])
    (direction,) = normal_draw(2642, x.shape, scale=EDGES[edge])
    (cotangent,) = normal_draw(2643, x.shape)

    def plain(x):
        return 0.5 * x * (1 + jax.scipy.special.erf(x / np.sqrt(2.0)))

    assert_contract(exact_gelu, (x,), (direction,), cotangent, f"an exact gelu {edge}", truth_fn=plain,
                    plain_fn=plain)


@pytest.mark.parametrize("edge", ["normal", "near the least normal"])
def test_a_time_shift_holds_every_order(edge):
    """`schedules.flow.time_shift` (its derivative in the regular form): every order finite wherever float64
    is a normal float32 and the plain shifted time, shift t / (1 + (shift - 1) t), is, over times in (0, 1],
    its domain, and near its least normal; a time past 1 is no time of a schedule's."""
    rng = np.random.default_rng(2644)
    t = jnp.asarray(rng.uniform(1e-3, 1.0, (9,)) * EDGES[edge], jnp.float32)
    direction = jnp.asarray(rng.standard_normal(t.shape) * EDGES[edge], jnp.float32)
    cotangent = jnp.asarray(rng.standard_normal(t.shape), jnp.float32)

    def plain(t):
        return 3.0 * t / (1 + 2.0 * t)

    assert_contract(lambda t: time_shift(3.0, t), (t,), (direction,), cotangent, f"a time shift {edge}",
                    truth_fn=plain, plain_fn=plain)


def categorical_draw(seed: int, edge: str) -> tuple:
    """Logits [3, 6] at `edge`'s scale, labels, soft targets and a reference distribution's logits."""
    rng = np.random.default_rng(seed)
    logits = jnp.asarray(rng.standard_normal((3, 6)) * EDGES[edge], jnp.float32)
    labels = jnp.asarray(rng.integers(0, 6, (3,)), jnp.int32)
    targets = jnp.asarray(rng.dirichlet(np.ones(6), 3), jnp.float32)
    reference = jnp.asarray(rng.standard_normal((3, 6)), jnp.float32)
    return logits, labels, targets, reference


@pytest.mark.parametrize("edge", EDGES)
def test_the_categorical_losses_hold_every_order(edge):
    """`categorical`'s probabilities, cross entropy, soft cross entropy and KL divergence (each a custom
    derivative): every order in the logits finite wherever float64 is a normal float32 and the plain formula,
    jax.nn's softmax and log-softmax as written, is."""
    logits, labels, targets, reference = categorical_draw(2645, edge)
    (direction,) = normal_draw(2646, logits.shape, scale=EDGES[edge])
    rows = jnp.arange(3)
    sites = {
        "probabilities": (categorical.probabilities, lambda x: jax.nn.softmax(x, axis=-1), (3, 6)),
        "cross entropy": (lambda x: categorical.cross_entropy(x, labels),
                          lambda x: -jax.nn.log_softmax(x, axis=-1)[rows, labels], (3,)),
        "soft cross entropy": (lambda x: categorical.soft_cross_entropy(x, targets),
                               lambda x: -jnp.sum(targets * jax.nn.log_softmax(x, axis=-1), axis=-1), (3,)),
        "kl divergence": (lambda x: categorical.kl_divergence(x, reference), lambda x: jnp.sum(
            jax.nn.softmax(reference, axis=-1)
            * (jax.nn.log_softmax(reference, axis=-1) - jax.nn.log_softmax(x, axis=-1)), axis=-1), (3,)),
    }
    for name, (fn, plain, shape) in sites.items():
        (cotangent,) = normal_draw(2647, shape)
        assert_contract(fn, (logits,), (direction,), cotangent, f"{name} {edge}", truth_fn=plain,
                        plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_the_precision_seams_hold_every_order(edge):
    """`precision`'s correctly rounded division, compensated sum and rounded operand (each a custom
    derivative): every order finite wherever float64 is a normal float32 and the plain formula, the
    division, the sum and the identity as JAX writes them, is."""
    numerator, denominator = normal_draw(2648, (5, 4), (5, 4), scale=EDGES[edge])
    numerator_direction, denominator_direction = normal_draw(2649, (5, 4), (5, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(2650, (5, 4))
    # Denominators a quarter of the scale or more from zero, so the quotient is a normal float32 wherever
    # float64's is.
    denominator = jnp.copysign(jnp.abs(denominator) + 0.25 * EDGES[edge], denominator)
    assert_contract(precision.correctly_rounded_divide, (numerator, denominator),
                    (numerator_direction, denominator_direction), cotangent,
                    f"a correctly rounded division {edge}", truth_fn=jnp.divide, plain_fn=jnp.divide)
    (summed,) = normal_draw(2651, (3, 64), scale=EDGES[edge])
    (summed_direction,) = normal_draw(2652, summed.shape, scale=EDGES[edge])
    (summed_cotangent,) = normal_draw(2653, (3,))

    def plain_sum(values):
        return jnp.sum(values, axis=1)

    assert_contract(lambda values: precision.compensated_sum(values, (1,)), (summed,), (summed_direction,),
                    summed_cotangent, f"a compensated sum {edge}", truth_fn=plain_sum, plain_fn=plain_sum)
    assert_contract(lambda x: precision.rounded_operand(x, jnp.bfloat16), (numerator,),
                    (numerator_direction,), cotangent, f"a rounded operand {edge}",
                    plain_fn=lambda x: x.astype(jnp.bfloat16).astype(x.dtype))


def plain_attention(query, key, value):
    """Softmax attention as its formula reads, `[B, S, H, D]` each: every query's logits over every key at
    the width's inverse root, their softmax's weights on the values (none of Dew's attention code)."""
    logits = jnp.einsum("bqhd,bkhd->bhqk", query, key) * query.shape[-1] ** -0.5
    return jnp.einsum("bhqk,bkhd->bqhd", jax.nn.softmax(logits, axis=-1), value)


REVERSE_ONLY = ("jvp", "jvp of grad", "second jvp", "jvp of grad, remat", "second jvp, remat",
                "jvp of grad, nested jit", "jvp of grad, scanned", "grad of grad")
"""The orders a fused attention kernel's reverse-only derivative (`jax.custom_vjp`) does not run: every
order through a forward derivative, which `forward_mode_attention` serves, and reverse over reverse, its
backward kernel's own derivative (ledger 2654)."""


def attention_draw(seed: int, shape: tuple, edge: str) -> tuple:
    """A query, key and value of `shape` at `edge`'s scale, a direction for each and an output cotangent."""
    primals = normal_draw(seed, shape, shape, shape, scale=EDGES[edge])
    direction = normal_draw(seed + 1, shape, shape, shape, scale=EDGES[edge])
    (cotangent,) = normal_draw(seed + 2, shape)
    return primals, direction, cotangent


@pytest.mark.parametrize("edge", EDGES)
def test_a_fused_attention_call_in_forward_mode_holds_every_order(edge):
    """`attention.forward_differentiable` (a fused call under `forward_mode_attention`: the template's value,
    the reference path's tangent, a custom JVP): every order runs, finite wherever float64 is a normal
    float32 and the plain attention formula is, at queries, keys and values whose products near float32's
    ends."""
    primals, direction, cotangent = attention_draw(2654, (1, 16, 2, 16), edge)

    def fused(query, key, value):
        with attention.forward_mode_attention():
            return attention.scaled_dot_product_attention(query, key, value, implementation="template")

    assert_contract(fused, primals, direction, cotangent, f"a fused call in forward mode {edge}",
                    truth_fn=plain_attention, plain_fn=plain_attention)


@pytest.mark.parametrize(("implementation", "edge"), list(itertools.product(("template", "tpu"), EDGES)))
def test_a_fused_attention_kernel_holds_its_reverse_orders(implementation, edge):
    """The template's route (`attention_route._attend`) and splash (`splash._attention`, interpreted off a
    TPU), each a reverse-only custom derivative: its primal and its reverse derivative run, finite wherever
    float64 is a normal float32 and the plain attention formula is, at queries, keys and values whose
    products near float32's ends. Its other orders do not run (`REVERSE_ONLY`, ledger 2654)."""
    primals, direction, cotangent = attention_draw(2657, (1, 128, 2, 64), edge)

    def fused(query, key, value):
        return attention.scaled_dot_product_attention(query, key, value, implementation=implementation)

    assert_contract(fused, primals, direction, cotangent, f"{implementation} attention {edge}",
                    truth_fn=plain_attention, plain_fn=plain_attention, without=REVERSE_ONLY)


@pytest.mark.parametrize("edge", EDGES)
def test_the_paged_decode_kernel_holds_every_order(edge):
    """`kv_cache.gpu_paged` (the Pallas paged decode kernel's value, the gathered attention's derivatives
    through `generation.linear_tangent`), interpreted off a GPU: every order runs in the query, the key
    and value pages and the sinks, finite wherever float64 is a normal float32 and the plain formula, one
    query a row over its row's first `lengths` keys gathered from its pages with the sinks in the
    softmax, is."""
    table = jnp.asarray([[2, 0], [0, 3]], jnp.int32)
    lengths = jnp.asarray([21, 32], jnp.int32)
    shapes = ((2, 4, 64), (2, 4, 16, 64), (2, 4, 16, 64), (4,))
    primals = normal_draw(2660, *shapes, scale=EDGES[edge])
    direction = normal_draw(2661, *shapes, scale=EDGES[edge])
    (cotangent,) = normal_draw(2662, shapes[0])

    def paged(query, key, value, sinks):
        return kv_cache.gpu_paged(None)(query, key, value, table, lengths, None, sinks)

    def plain(query, key, value, sinks):
        # Each row's pages in its table's order, `[rows, keys, key heads, D]`, each key head read by two
        # query heads.
        keys, values = (jnp.repeat(jnp.moveaxis(pool[:, table], 0, 3).reshape(2, 32, 2, 64), 2, axis=2)
                        for pool in (key, value))
        logits = jnp.einsum("rhd,rkhd->rhk", query, keys) * 64 ** -0.5
        logits = jnp.where(jnp.arange(keys.shape[1]) < lengths[:, None, None], logits, -jnp.inf)
        sunk = jnp.broadcast_to(sinks[None, :, None], (2, 4, 1))
        weights = jax.nn.softmax(jnp.concatenate([logits, sunk], axis=-1), axis=-1)[..., :-1]
        return jnp.einsum("rhk,rkhd->rhd", weights, values)

    assert_contract(paged, primals, direction, cotangent, f"the paged decode kernel {edge}",
                    truth_fn=plain, plain_fn=plain)


def joined(seam):
    """`seam`'s two outputs, the stream and its norm, side by side along the features: one output a contract
    pulls back."""
    return lambda *inputs: jnp.concatenate(seam(*inputs), axis=-1)


def plain_rms(x, epsilon: float = 1e-6):
    """x over its root mean square, the RMS norm as written."""
    return x / jnp.sqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + epsilon)


@pytest.mark.parametrize("edge", EDGES)
def test_the_fused_norms_and_products_hold_every_order(edge):
    """`norm_swiglu`'s fused pre-norm seam, RMS norm, gated norm, SiLU product and packed SiLU product (each
    through `generation.linear_tangent`), interpreted on a CPU as the norm tests call them: every order finite
    wherever float64 is a normal float32 and the plain formula, the norm and the product as written, is."""
    import test_norm_swiglu as norms

    x, branch, gate, up = normal_draw(2661, (3, 16), (3, 16), (3, 16), (3, 16), scale=EDGES[edge])
    (scale,) = normal_draw(2662, (16,))
    dx, dbranch, dgate, dup = normal_draw(2663, (3, 16), (3, 16), (3, 16), (3, 16), scale=EDGES[edge])
    (dscale,) = normal_draw(2664, (16,))
    (cotangent,) = normal_draw(2665, (3, 16))
    sites = {
        "a pre-norm seam": (joined(norms.fused_seam(1e-6, jnp.float32, offset=False, after_cast=False)),
                            joined(lambda x, branch, scale: (x + branch, plain_rms(x + branch) * scale)),
                            (x, branch, scale), (dx, dbranch, dscale),
                            jnp.concatenate([cotangent, cotangent], axis=-1)),
        "an unweighted RMS norm": (norms.fused_unweighted(1e-6), plain_rms, (x,), (dx,), cotangent),
        "a gated RMS norm": (norms.fused_gated(1e-6, jnp.float32, "silu"),
                             lambda x, gate, scale: plain_rms(x) * scale * jax.nn.silu(gate),
                             (x, gate, scale), (dx, dgate, dscale), cotangent),
        "a SiLU product": (norms.fused_swiglu(), lambda gate, up: jax.nn.silu(gate) * up, (gate, up),
                           (dgate, dup), cotangent),
        "a packed SiLU product": (norms.fused_packed(),
                                  lambda gate_up: jax.nn.silu(gate_up[..., 0, :]) * gate_up[..., 1, :],
                                  (jnp.stack([gate, up], axis=-2),), (jnp.stack([dgate, dup], axis=-2),),
                                  cotangent),
    }
    if edge == "near the largest":
        # The gated norm's reverse mode is NaN there where the plain formula is finite (ledger 2692); that
        # row's head adds the edge with its fix.
        del sites["a gated RMS norm"]
    for name, (fn, plain, primals, direction, pulled_back) in sites.items():
        assert_contract(fn, primals, direction, pulled_back, f"{name} {edge}", truth_fn=plain, plain_fn=plain)


GROUPS = jnp.asarray([3, 0, 5], jnp.int32)
"""A grouped product's sizes: eight rows, an empty middle expert."""


def masked_grouped_product(rows, matrices):
    """Each row's product with its group's matrix (`GROUPS`), by a mask: the grouped product as written."""
    groups = jnp.repeat(jnp.arange(3), GROUPS, total_repeat_length=8)
    return jnp.einsum("rg,rk,gkn->rn", jax.nn.one_hot(groups, 3, dtype=rows.dtype), rows, matrices)


@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_a_portable_grouped_projection_holds_every_order(edge):
    """`grouped_matmul.portable_projection` (its custom derivative, the rounded operands' products): every
    order finite wherever float64 is a normal float32 and the masked grouped product as written is. Near the
    largest its primal overflows a partial sum the masked product does not (ledger 2711), that row's head
    adds the edge; it takes no reverse over reverse (XLA's ragged product, ledger 2650)."""
    from dew.nn.kernels import grouped_matmul

    primals = normal_draw(1564, (8, 16), (3, 16, 4), scale=EDGES[edge])
    direction = normal_draw(1565, (8, 16), (3, 16, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(1566, (8, 4))
    def projected(rows, matrices):
        return grouped_matmul.portable_projection(rows, matrices, GROUPS, None, None)

    assert_contract(projected, primals, direction, cotangent, f"a portable grouped projection {edge}",
                    truth_fn=masked_grouped_product, plain_fn=masked_grouped_product,
                    without=("grad of grad",))


def rms_producer(lhs, scale):
    """An RMS norm as a GEMM's producer writes it: the operand over its root mean square, times its scale."""
    return lhs * jax.lax.rsqrt(jnp.mean(jnp.square(lhs), axis=-1, keepdims=True) + 1e-6) * scale


@pytest.mark.parametrize("edge", EDGES)
def test_a_gemm_prologue_holds_every_order(edge):
    """The GEMM template's prologue (`template._prologued`, a custom JVP whose tangent is the producer's own,
    as the fused GEMM hands it, `fused._whole_jvp`): every order finite wherever float64 is a normal float32
    and the producer as written is."""
    from dew.nn.kernels.templates.gemm.template import _prologued

    prologued = _prologued(rms_producer, lambda *primals: lambda *tangents: jax.jvp(
        rms_producer, primals, tangents)[1])
    primals = normal_draw(394, (6, 16), (16,), scale=EDGES[edge])
    direction = normal_draw(395, (6, 16), (16,), scale=EDGES[edge])
    (cotangent,) = normal_draw(396, (6, 16))
    assert_contract(prologued, primals, direction, cotangent, f"a GEMM prologue {edge}",
                    truth_fn=rms_producer, plain_fn=rms_producer)


def softly_rectified(scores):
    """A score hook through a custom derivative entry by entry (jnp.logaddexp's)."""
    return jnp.logaddexp(scores, 0.5 * scores)


@pytest.mark.parametrize("edge", EDGES)
def test_a_transposed_hooks_custom_derivative_holds_every_order_it_takes(edge):
    """A custom derivative within a hook a mainloop runs on transposed tiles (`layout.transposed`, its rule
    held entry by entry, `_custom.held`): every order it takes finite wherever float64 is a normal float32
    and the hook on the tile as its caller holds it is; it takes no forward mode (ledger 2658)."""
    from dew.nn.kernels.templates import layout

    (tile,) = normal_draw(105, (6, 4), scale=EDGES[edge])
    (direction,) = normal_draw(106, (6, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(107, (6, 4))

    def plain(tile):
        return softly_rectified(tile.T).T

    assert_contract(layout.transposed(softly_rectified), (tile,), (direction,), cotangent,
                    f"a transposed hook's custom derivative {edge}", truth_fn=plain, plain_fn=plain,
                    without=FORWARD_ORDERS)


@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_a_stored_expert_primal_holds_every_order(edge):
    """An expert layer's primal over its parameters as stored (`moe._stored_primal`, differentiated over them
    widened): every order finite wherever float64 is a normal float32 and the run itself is. Near the
    largest its kernel gradient sums its products in an order that overflows where the run's own does not
    (ledger 2711), that row's head adds the edge."""
    from dew.nn import moe

    indices = jnp.zeros((6,), jnp.int32)

    def run(x, indices, parameters, weights):
        return (x @ parameters["kernel"]) * weights[:, None]

    def stored(x, kernel, weights):
        return moe._stored_primal(run, x, indices, {"kernel": kernel}, weights)

    def plain(x, kernel, weights):
        return run(x, indices, {"kernel": kernel}, weights)

    primals = normal_draw(597, (6, 16), (16, 4), (6,), scale=EDGES[edge])
    direction = normal_draw(598, (6, 16), (16, 4), (6,), scale=EDGES[edge])
    (cotangent,) = normal_draw(599, (6, 4))
    assert_contract(stored, primals, direction, cotangent, f"a stored expert primal {edge}", truth_fn=plain,
                    plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_a_default_precision_call_holds_every_order_it_takes(edge):
    """A function traced at the default matmul precision (`precision.at_default_precision`, a custom VJP):
    every order it takes finite wherever float64 is a normal float32 and the product as written is; it takes
    no forward mode (ledger 2658)."""
    def product(x, y):
        return x @ y

    primals = normal_draw(1208, (5, 16), (16, 3), scale=EDGES[edge])
    direction = normal_draw(1209, (5, 16), (16, 3), scale=EDGES[edge])
    (cotangent,) = normal_draw(1210, (5, 3))
    assert_contract(precision.at_default_precision(product), primals, direction, cotangent,
                    f"a default-precision call {edge}", truth_fn=product, plain_fn=product,
                    without=FORWARD_ORDERS)


@pytest.mark.parametrize("edge", EDGES)
def test_a_frozen_fp8_product_holds_every_order(edge):
    """A frozen FP8 kernel's product (`quantization.frozen_fp8_product`, its tangent the decoded kernel's):
    every order in the input finite wherever float64 is a normal float32 and the product with the decoded
    kernel as written is."""
    from dew.quantization import FROZEN_FP8, frozen_fp8_kernel, frozen_fp8_leaves

    (kernel,) = normal_draw(475, (16, 8))
    leaves = frozen_fp8_leaves(kernel)
    decoded = frozen_fp8_kernel(leaves)

    def frozen(x):
        from dew.quantization import frozen_fp8_product

        return frozen_fp8_product(x, *(leaves[name] for name in FROZEN_FP8))

    def plain(x):
        return x @ decoded.astype(x.dtype)

    (x,) = normal_draw(476, (5, 16), scale=EDGES[edge])
    (direction,) = normal_draw(477, (5, 16), scale=EDGES[edge])
    (cotangent,) = normal_draw(478, (5, 8))
    assert_contract(frozen, (x,), (direction,), cotangent, f"a frozen FP8 product {edge}", truth_fn=plain,
                    plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_a_quantized_convolution_holds_every_order_it_takes(edge):
    """A quantized convolution under Qwix's quantized training (`quantization`'s grouped convolution, a
    custom VJP whose backward is the float convolution's transpose at the dequantized operands): every order
    it takes finite wherever float64 is a normal float32 and the float convolution is; it takes no forward
    mode (ledger 2658)."""
    pytest.importorskip("qwix")
    from dew.nn.conv import Conv
    from dew.quantization import Quantization

    conv = Conv(features=8, kernel_size=(3, 3), padding="SAME", use_bias=False)
    quantized = Quantization().apply(conv)

    def convolved(x, kernel):
        return quantized.apply({"params": {"kernel": kernel}}, x)

    def plain(x, kernel):
        return conv.clone(dtype=x.dtype).apply({"params": {"kernel": kernel}}, x)

    primals = normal_draw(1134, (1, 6, 6, 4), (3, 3, 4, 8), scale=EDGES[edge])
    direction = normal_draw(1135, (1, 6, 6, 4), (3, 3, 4, 8), scale=EDGES[edge])
    (cotangent,) = normal_draw(1136, (1, 6, 6, 8))
    assert_contract(convolved, primals, direction, cotangent, f"a quantized convolution {edge}",
                    truth_fn=plain, plain_fn=plain, without=FORWARD_ORDERS)


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


@pytest.mark.parametrize("edge", EDGES)
def test_a_broadcast_summing_its_gradient_wide_holds_every_order(edge):
    """`torchax_fallback._accumulated` (a broadcast whose gradient sums in float32 and rounds once to its
    value's dtype, as torch's autograd reduces a bf16 gradient): every order runs, finite wherever float64
    is a normal float32, at values near float32's ends, beside the plain broadcast; but forward mode,
    which a custom_vjp takes none of, the orders ledger 2653 adds with its fix."""
    value, direction = normal_draw(959, (3, 1, 4), (3, 1, 4), scale=EDGES[edge])
    (cotangent,) = normal_draw(961, (3, 5, 4))

    def broadcast(value):
        return torchax_fallback._accumulated(value, (3, 5, 4), (0, 1, 2))

    def plain(value):
        return jnp.broadcast_to(value, (3, 5, 4))

    # Forward mode's orders (`orders`), which ledger 2653 lists.
    forward = ("jvp", "jvp of grad", "second jvp", "jvp of grad, remat", "second jvp, remat",
               "jvp of grad, nested jit", "jvp of grad, scanned")
    assert_contract(broadcast, (value,), (direction,), cotangent, f"a wide-summed broadcast {edge}",
                    truth_fn=plain, plain_fn=plain, without=forward)



INTERPRETED = jax.default_backend() != "gpu"
"""The ported kernels interpret off a GPU."""

REVERSE_ONLY = ("jvp", "jvp of grad", "second jvp", "jvp of grad, remat", "second jvp, remat",
                "jvp of grad, nested jit", "jvp of grad, scanned", "grad of grad")
"""The orders a custom VJP over kernels with no derivative of their own does not run: forward mode, and a
reverse derivative of its reverse rule's kernels (ledger 2655)."""


def plain_delta_rule(query, key, value, g, beta, state, scale):
    """The gated delta rule a token at a time as its paper states it: the memory `[B, Hv, Dk, Dv]` decays by
    `exp(g)` (a head's `[B, S, Hv]` or a key dimension's `[B, S, Hv, Dk]`), the value corrected against the
    decayed memory is written by `beta k`, and the scaled query reads the memory written. Key heads serve
    `Hv // Hk` value heads each. Returns the output `[B, S, Hv, Dv]` and the final memory."""
    group = value.shape[2] // key.shape[2]
    query, key = (jnp.repeat(x, group, axis=2) for x in (query, key))
    decay = jnp.exp(g)[..., None] if g.ndim == 3 else jnp.exp(g)

    def token(memory, step):
        q, k, v, d, b = step
        memory = memory * d[..., :, None]
        corrected = v - jnp.einsum("bhk,bhkv->bhv", k, memory)
        memory = memory + jnp.einsum("bhk,bhv->bhkv", k * b[..., None], corrected)
        return memory, jnp.einsum("bhk,bhkv->bhv", q * scale, memory)

    final, out = jax.lax.scan(token, state, tuple(jnp.moveaxis(x, 1, 0) for x in (query, key, value, decay,
                                                                                   beta)))
    return jnp.moveaxis(out, 0, 1), final


def plain_ssd(x, dt, a, b, c, skip, state):
    """Mamba2's state space recurrence a token at a time: `h_t = exp(dt_t a) h_{t-1} + dt_t x_t b_t`,
    `y_t = h_t c_t + skip x_t`, `x` `[B, S, H, P]`, `dt` `[B, S, H]`, `a` and `skip` `[H]`, `b` and `c`
    `[B, S, G, N]` each serving `H // G` heads, `state` `[B, H, P, N]`. Returns `y` and the final state."""
    group = x.shape[2] // b.shape[2]
    b, c = (jnp.repeat(t, group, axis=2) for t in (b, c))

    def token(h, step):
        x_t, dt_t, b_t, c_t = step
        h = jnp.exp(dt_t * a)[..., None, None] * h + (dt_t[..., None] * x_t)[..., None] * b_t[..., None, :]
        return h, jnp.einsum("bhpn,bhn->bhp", h, c_t) + skip[:, None] * x_t

    final, y = jax.lax.scan(token, state, tuple(jnp.moveaxis(t, 1, 0) for t in (x, dt, b, c)))
    return jnp.moveaxis(y, 0, 1), final


def flattened(fn):
    """`fn`'s outputs as one vector, which the orders through its gradient sum against one cotangent."""
    return lambda *operands: jnp.concatenate([out.ravel() for out in jax.tree.leaves(fn(*operands))])


def delta_rule_draw(seed: int, scale: float, *, key_heads: int, value_heads: int, width: int, length: int,
                    per_key: bool, unit_keys: bool = False):
    """A delta rule's operands `(query, key, value, g, beta, state)`, their direction and the outputs'
    cotangents: queries, values and memory at `scale`, keys at it or of unit norm, a log decay in (-inf, 0)
    a head or a key dimension, write gates in (0, 1)."""
    query, key, value, decay, gate, state = normal_draw(
        seed, (1, length, key_heads, width), (1, length, key_heads, width), (1, length, value_heads, width),
        (1, length, value_heads, *((width,) if per_key else ())), (1, length, value_heads),
        (1, value_heads, width, width))
    key = key / jnp.linalg.norm(key, axis=-1, keepdims=True) if unit_keys else key * scale
    primals = (query * scale, key, value * scale, -jax.nn.softplus(decay), jax.nn.sigmoid(gate),
               state * scale)
    drawn = normal_draw(seed + 1, *(x.shape for x in primals))
    direction = tuple(along * (scale if at in (0, 2, 5) or (at == 1 and not unit_keys) else 1.0)
                      for at, along in enumerate(drawn))
    cotangent = normal_draw(seed + 2, value.shape, state.shape)
    return primals, direction, cotangent


# The ports near float32's largest: their reverse derivatives are NaN at entries the recurrence keeps finite,
# the gated delta rule's where fla's own chunked form keeps them too (ledger 2656).
@pytest.mark.parametrize("per_key", [False, True], ids=["fla_gdn", "fla_kda"])
@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_the_delta_rule_ports_hold_their_orders(per_key, edge):
    """fla's chunked gated delta rule and Kimi delta attention on the ported kernels
    (`fla_gdn.chunk_gated_delta_rule`, `fla_kda.chunk_kda`, custom VJPs over Triton kernels, interpreted off a
    GPU), from a zero memory: the primal and reverse derivative of the output finite wherever float64 is a
    normal float32 and the delta rule a token at a time is (`plain_delta_rule`, in float64 the truth), at
    queries, keys and values whose products near float32's least normal. Forward mode and reverse over
    reverse do not run (`REVERSE_ONLY`)."""
    (query, key, value, g, beta, _), direction, _ = delta_rule_draw(
        2661 + per_key, EDGES[edge], key_heads=2, value_heads=2, width=64, length=64, per_key=per_key)
    (cotangent,) = normal_draw(2663, value.shape)
    memory = jnp.zeros((1, 2, 64, 64), jnp.float32)
    scale = 64 ** -0.5
    port = fla_kda.chunk_kda if per_key else fla_gdn.chunk_gated_delta_rule

    def ported(*operands):
        return port(*operands, memory, None, scale, 64, INTERPRETED)[0]

    def recurrent(*operands):
        return plain_delta_rule(*operands, memory.astype(operands[0].dtype), scale)[0]

    assert_contract(ported, (query, key, value, g, beta), direction[:5], cotangent,
                    f"the {'kda' if per_key else 'gdn'} port {edge}", truth_fn=recurrent, plain_fn=recurrent,
                    without=REVERSE_ONLY)


# The template rule near float32's largest: its reverse derivative is NaN at entries fla's chunked form keeps
# finite (ledger 2656).
@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_the_template_rule_holds_its_orders(edge):
    """The recurrence template's chunked rule (`recurrence_chunk.mosaic_rule`, a custom VJP over the
    mainloop's launches, interpreted off a GPU), a 16-bit call with its chunk inverse's declared bound (keys
    of unit norm, sigmoid write gates: its domain), from a zero memory: the primal and reverse derivative of
    the output finite wherever float64 is a normal float32 and the delta rule a token at a time is
    (`plain_delta_rule`), at queries and values whose products near float32's least normal. Forward mode and
    reverse over reverse do not run (`REVERSE_ONLY`)."""
    (query, key, value, g, beta, _), direction, _ = delta_rule_draw(
        2664, EDGES[edge], key_heads=1, value_heads=1, width=64, length=64, per_key=False, unit_keys=True)
    primals = (query.astype(jnp.bfloat16), key.astype(jnp.bfloat16), value.astype(jnp.bfloat16), g,
               beta.astype(jnp.bfloat16))
    direction = tuple(along.astype(operand.dtype) for along, operand in zip(direction, primals, strict=False))
    (cotangent,) = normal_draw(2665, value.shape)

    def rule(*operands):
        return recurrence_chunk.mosaic_rule(*operands, tiles=mainloop.Tiles(32, 1), bound=1.0)[0]

    def recurrent(*operands):
        operands = tuple(x.astype(jnp.promote_types(x.dtype, jnp.float32)) for x in operands)
        return plain_delta_rule(*operands, jnp.zeros((1, 1, 64, 64), operands[0].dtype), 64 ** -0.5)[0]

    assert_contract(rule, primals, direction, cotangent, f"the template rule {edge}", truth_fn=recurrent,
                    plain_fn=recurrent, without=REVERSE_ONLY)


@pytest.mark.parametrize("edge", EDGES)
def test_the_chunked_op_holds_every_order(edge):
    """The chunked op (`recurrence_chunk.chunk`: its kernel a product, `_per_product`, differentiated
    through the portable math, and the skip `D v`, `recurrence.skipped`, each through
    `generation.linear_tangent`), the gated delta rule with a decay a head, its queries scaled, and the skip,
    from a zero memory: every order of its output finite wherever float64 is a normal float32 and the rule a
    token at a time with its skip is (`plain_delta_rule`, in float64 the truth), at queries, keys and values
    whose products near float32's ends; near float32's largest wherever fla's own chunked form with the skip
    is (`precedent_chunked`, as the portable rule's edge: ledger 2470)."""
    (query, key, value, g, beta, _), direction, _ = delta_rule_draw(
        2664, EDGES[edge], key_heads=2, value_heads=2, width=8, length=12, per_key=False)
    (skip,) = normal_draw(2667, (2,))
    (skip_direction,) = normal_draw(2668, (2,))
    (cotangent,) = normal_draw(2669, value.shape)
    hooks = recurrence.Hooks(decay="scalar", skip=True, scale=8 ** -0.5)

    def chunked(query, key, value, g, beta, skip):
        operands = recurrence_chunk.Chunk(query, key, value, log_decay=g, beta=beta, skip=skip)
        return recurrence_chunk.chunk(hooks, operands, chunk_size=4)[0]

    def recurrent(query, key, value, g, beta, skip):
        out, _ = plain_delta_rule(query, key, value, g, beta, jnp.zeros((1, 2, 8, 8), query.dtype),
                                  8 ** -0.5)
        return out + skip[:, None] * value

    def precedent(query, key, value, g, beta, skip):
        return precedent_chunked(query, key, value, g[..., None], beta, chunk_size=4) + skip[:, None] * value

    assert_contract(chunked, (query, key, value, g, beta, skip), (*direction[:5], skip_direction), cotangent,
                    f"the chunked op {edge}", truth_fn=recurrent,
                    plain_fn=precedent if edge == "near the largest" else recurrent)


@pytest.mark.parametrize("edge", EDGES)
def test_the_decode_step_holds_every_order(edge):
    """A decode token in one launch (`recurrence.step`, the launch differentiated through its reference by
    `generation.linear_tangent`; interpreted off a GPU), the gated delta rule with a decay a head and the
    skip: every order of its output and updated memory finite wherever float64 is a normal float32 and the
    token's update as the rule states it is (`plain_delta_rule` over one token), at queries, keys, values
    and memory whose products near float32's ends."""
    (query, key, value, g, beta, state), direction, cotangent = delta_rule_draw(
        2669, EDGES[edge], key_heads=1, value_heads=2, width=8, length=1, per_key=False)
    (skip,) = normal_draw(2672, (2,))
    (skip_direction,) = normal_draw(2673, (2,))
    hooks = recurrence.Hooks(decay="scalar", skip=True)

    def stepped(query, key, value, g, beta, state, skip):
        operands = recurrence.Step(state, query[:, 0], key[:, 0], value[:, 0], log_decay=g[:, 0],
                                   beta=beta[:, 0], skip=skip)
        out, _, updated = recurrence.step(hooks, operands)
        return out, updated

    def plain(query, key, value, g, beta, state, skip):
        out, final = plain_delta_rule(query, key, value, g, beta, state, 1.0)
        return out[:, 0] + skip[:, None] * value[:, 0], final

    flat_cotangent = jnp.concatenate([cotangent[0][:, 0].ravel(), cotangent[1].ravel()])
    assert_contract(flattened(stepped), (query, key, value, g, beta, state, skip),
                    (*direction, skip_direction), flat_cotangent, f"the decode step {edge}",
                    truth_fn=flattened(plain), plain_fn=flattened(plain))


def ssd_draw(seed: int, scale: float):
    """Mamba2's operands `(x, dt, a, b, c, skip, state)`, their direction and the outputs' cotangents: inputs,
    the groups' input gates and the state at `scale`, positive steps, decays under zero."""
    x, steps, decay, b, c, skip, state = normal_draw(
        seed, (1, 64, 2, 16), (1, 64, 2), (2,), (1, 64, 1, 16), (1, 64, 1, 16), (2,), (1, 2, 16, 16))
    primals = (x * scale, jax.nn.softplus(steps), -jnp.exp(decay), b * scale, c, skip, state * scale)
    drawn = normal_draw(seed + 1, *(t.shape for t in primals))
    direction = tuple(along * (scale if at in (0, 3, 6) else 1.0) for at, along in enumerate(drawn))
    return primals, direction, normal_draw(seed + 2, x.shape, state.shape)


# mamba_ssm's port near float32's largest: no transcription of mamba_ssm's chunked form to hold it to yet
# (ledger 2656).
@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_the_ssd_port_holds_its_orders(edge):
    """Mamba2's chunked scan on the port of mamba_ssm's kernels (`mamba_ssd.ssd_scan`, a custom VJP over
    Triton kernels, interpreted off a GPU): its primal and reverse derivative finite wherever float64 is a
    normal float32 and the recurrence a token at a time is (`plain_ssd`), at inputs, gates and states whose
    products near float32's least normal. Forward mode and reverse over reverse do not run
    (`REVERSE_ONLY`)."""
    primals, direction, cotangent = ssd_draw(2674, EDGES[edge])

    def ported(*operands):
        return mamba_ssd.ssd_scan(*operands, None, 64, INTERPRETED)

    assert_contract(ported, primals, direction, cotangent, f"the ssd port {edge}", truth_fn=plain_ssd,
                    plain_fn=plain_ssd, without=REVERSE_ONLY)


# A TPU's scan near float32's largest: its output is infinite at an entry the recurrence keeps finite
# (ledger 2656).
@pytest.mark.parametrize("edge", [edge for edge in EDGES if edge != "near the largest"])
def test_the_tpu_ssd_scan_holds_its_orders(edge):
    """Mamba2's chunked scan on a TPU's Pallas kernel (`ssd.ssd_chunk_scan`, a custom VJP, interpreted off a
    TPU), its operands the inputs scaled by their steps, each head's gates and each step's decay `A dt`: its
    primal and reverse derivative finite wherever float64 is a normal float32 and the recurrence a token at a
    time is (`_plain_scaled_ssd`), at inputs, gates and states whose products near float32's least normal.
    Forward mode and reverse over reverse do not run (`REVERSE_ONLY`)."""
    (x, steps, decay, b, c, _, state), direction, cotangent = ssd_draw(2677, EDGES[edge])

    def tpu(x, b, c, a, state):
        return ssd.ssd_chunk_scan(x, b, c, a, state, 64, "tpu")

    heads = tuple(jnp.repeat(t, 2, axis=2) for t in (b, c))
    primals = (x * steps[..., None], *heads, steps * decay, state)
    along = (direction[0], *(jnp.repeat(t, 2, axis=2) for t in direction[3:5]), direction[1], direction[6])
    assert_contract(tpu, primals, along, cotangent, f"the TPU ssd scan {edge}", truth_fn=_plain_scaled_ssd,
                    plain_fn=_plain_scaled_ssd, without=REVERSE_ONLY)


def _plain_scaled_ssd(x, b, c, a, state):
    """`plain_ssd` of inputs already scaled by their steps `x` `[B, S, H, P]`, each head's `b` and `c` `[B, S,
    H, N]` and each step's decay `a` `[B, S, H]` (`A dt`): `h_t = exp(a_t) h_{t-1} + x_t b_t`, `y_t = h_t
    c_t`."""
    def token(h, step):
        x_t, b_t, c_t, a_t = step
        h = jnp.exp(a_t)[..., None, None] * h + x_t[..., None] * b_t[..., None, :]
        return h, jnp.einsum("bhpn,bhn->bhp", h, c_t)

    final, y = jax.lax.scan(token, state, tuple(jnp.moveaxis(t, 1, 0) for t in (x, b, c, a)))
    return jnp.moveaxis(y, 0, 1), final


@pytest.mark.parametrize("edge", EDGES)
def test_the_short_conv_holds_every_order(edge):
    """The short conv on its kernels (`short_conv.conv`, differentiated through XLA's conv by
    `generation.linear_tangent`; interpreted off a GPU), with its bias and silu: every order finite wherever
    float64 is a normal float32 and the causal depthwise conv as its definition sums it is, at inputs and
    taps whose products near float32's ends."""
    scale = EDGES[edge]
    x, taps, bias = normal_draw(2677, (1, 12, 16), (16, 4), (16,))
    primals = (x * scale, taps, bias * scale)
    direction = tuple(along * (scale if at != 1 else 1.0)
                      for at, along in enumerate(normal_draw(2678, x.shape, taps.shape, bias.shape)))
    (cotangent,) = normal_draw(2679, x.shape)

    def kernel(x, taps, bias):
        return short_conv.conv(x, taps, bias, None, None, ((16, None),), activation=True, eps=1e-6,
                               dtype=x.dtype, interpret=INTERPRETED)[0]

    def plain(x, taps, bias):
        width = taps.shape[1]
        padded = jnp.pad(x, ((0, 0), (width - 1, 0), (0, 0)))
        summed = sum(padded[:, tap:tap + x.shape[1]] * taps[:, tap] for tap in range(width)) + bias
        return jax.nn.silu(summed)

    assert_contract(kernel, primals, direction, cotangent, f"the short conv {edge}", truth_fn=plain,
                    plain_fn=plain)


FORWARD_ORDERS = ("jvp", "jvp of grad", "second jvp", "jvp of grad, remat", "second jvp, remat",
                  "jvp of grad, nested jit", "jvp of grad, scanned")
"""The orders through a forward derivative, which a custom VJP with no forward rule does not run: Training &
Data's (`chunked`'s heads, `residuals`' pullbacks, a node perturbation's estimate), each joining with row
2690's fix."""


@pytest.mark.parametrize("edge", EDGES)
def test_a_ruled_loss_holds_every_order(edge):
    """`Objective.with_gradients`'s seam (`base._ruled`, a custom JVP): a value whose derivative in the
    parameters is a stated rule, every order finite wherever float64 is a normal float32 and the plain
    formula, the value plus the rule's inner product with the parameters' move, is."""
    value, first, second = normal_draw(2691, (3,), (4, 5), (6,), scale=EDGES[edge])
    rule = dict(zip(("first", "second"), normal_draw(2692, (4, 5), (6,)), strict=True))
    direction = normal_draw(2693, (3,), (4, 5), (6,), scale=EDGES[edge])
    (cotangent,) = normal_draw(2694, (3,))

    def ruled(value, first, second):
        return base._ruled(value, rule, {"first": first, "second": second})

    def plain(value, first, second):
        moved = {"first": first, "second": second}
        return value + sum(jnp.vdot(rule[name], leaf - jax.lax.stop_gradient(leaf))
                           for name, leaf in moved.items())

    assert_contract(ruled, (value, first, second), direction, cotangent, f"a ruled loss {edge}",
                    truth_fn=plain, plain_fn=plain)


@pytest.mark.parametrize("edge", EDGES)
def test_the_residual_pullbacks_hold_their_reverse_orders(edge):
    """The Gauss-Newton step's two custom VJPs (`residuals._pulled`, `residuals._stepped`): a microbatch's
    J^T r whose pullback hands the parameters their share of the cotangent, and the zero whose derivative
    in the pooled J^T r is the damped step, its dual's J^T (J J^T + damping I)^-1 r. Their reverse orders
    finite wherever float64 is a normal float32 and the plain formulas' are, the step solved as JAX solves
    a linear system; forward mode, row 2690's."""
    scale = EDGES[edge]
    first, second, pulled = normal_draw(2695, (2, 3), (4,), (10,), scale=scale)
    direction = normal_draw(2696, (2, 3), (4,), (10,), scale=scale)
    (cotangent,) = normal_draw(2697, (10,))

    def pullback(first, second, pulled):
        return residuals._pulled({"first": first, "second": second}, pulled, 0.25)

    def plain_pullback(first, second, pulled):
        raveled = jnp.concatenate([first.ravel(), second.ravel()])
        return jax.lax.stop_gradient(pulled) + 0.25 * (raveled - jax.lax.stop_gradient(raveled))

    assert_contract(pullback, (first, second, pulled), direction, cotangent,
                    f"a microbatch's pullback {edge}", truth_fn=plain_pullback, plain_fn=plain_pullback,
                    without=FORWARD_ORDERS)
    jacobian, residual, pulled = normal_draw(2698, (3, 10), (3,), (10,), scale=scale)
    directions = normal_draw(2699, (10,), (3, 10), (3,), scale=scale)
    (stepped,) = normal_draw(2700, ())
    rows = jnp.asarray(3, jnp.int32)

    def step(pulled, jacobian, residual):
        return residuals._stepped(0.5, 3, rows, pulled, (jacobian, residual))

    def plain_step(pulled, jacobian, residual):
        solved = jnp.linalg.solve(jacobian @ jacobian.T + 0.5 * jnp.eye(3, dtype=jacobian.dtype), residual)
        # The step's own derivatives in J and r stay: the pulled gradient moves with them.
        return jnp.vdot(pulled - jax.lax.stop_gradient(pulled), jacobian.T @ solved)

    assert_contract(step, (pulled, jacobian, residual), directions, stepped, f"a damped step {edge}",
                    truth_fn=plain_step, plain_fn=plain_step, without=FORWARD_ORDERS)


@pytest.mark.parametrize("edge", EDGES)
def test_the_whole_logits_losses_hold_every_order(edge):
    """`chunked._whole_losses` (a custom JVP, the categorical's own derivative): each row's cross entropy
    and log partition of whole logits, every order finite wherever float64 is a normal float32 and the
    plain formula, jax.nn's log-softmax and logsumexp as written, is."""
    (logits,) = normal_draw(2701, (4, 7), scale=EDGES[edge])
    (direction,) = normal_draw(2702, logits.shape, scale=EDGES[edge])
    (cotangent,) = normal_draw(2703, (2, 4))
    targets = jnp.asarray([0, 3, 6, 2], jnp.int32)

    def whole(logits):
        return jnp.stack(chunked._whole_losses(logits, targets))

    def plain(logits):
        rows = jnp.arange(logits.shape[0])
        return jnp.stack([-jax.nn.log_softmax(logits, axis=-1)[rows, targets],
                          jax.nn.logsumexp(logits, axis=-1)])

    assert_contract(whole, (logits,), (direction,), cotangent, f"whole logits' losses {edge}", truth_fn=plain,
                    plain_fn=plain)


@pytest.mark.parametrize("head", ["whole", "tiled", "fused"])
@pytest.mark.parametrize("edge", EDGES)
def test_the_lm_heads_hold_their_reverse_orders(head, edge):
    """The LM head's three custom VJPs (`chunked._whole_head`, `_bounded_head`, `_fused`): the head's
    per-token cross entropy and log partition of `hidden @ table.T`, whole, in tiles, or summed with its
    gradient taken in the forward. Their reverse orders finite wherever float64 is a normal float32 and the
    plain formula, the whole logits' log-softmax, is; forward mode and reverse over reverse, whose backward
    meets a symbolic zero it reshapes, row 2690's."""
    scale = EDGES[edge]
    hidden, table = normal_draw(2704, (2, 4, 8), (11, 8), scale=scale)
    direction = normal_draw(2705, hidden.shape, table.shape, scale=scale)
    targets = jnp.asarray([[0, 3, 10, 2], [7, 1, 4, 9]], jnp.int32)
    weights = jnp.ones(targets.shape, jnp.float32)

    def heads(hidden, table):
        head_table = OutputTable(table, vocab_major=True, precision=jax.lax.Precision.HIGHEST)
        if head == "fused":
            total, (_, _, log_z) = chunked.fused_head_loss(hidden, head_table, targets, weights, tile=(4, 4),
                                                           predict=False)
            return jnp.concatenate([total[None], log_z.ravel()])
        losses, _, log_z = chunked.chunked_cross_entropy(hidden, head_table, targets, 2, predict=False,
                                                         tile=None if head == "whole" else (4, 4))
        return jnp.stack([losses, log_z])

    def plain(hidden, table):
        logits = jnp.einsum("btf,vf->btv", hidden, table, precision=jax.lax.Precision.HIGHEST)
        scored = jax.nn.log_softmax(logits, axis=-1)
        losses = -jnp.take_along_axis(scored, targets[..., None], axis=-1)[..., 0]
        log_z = jax.nn.logsumexp(logits, axis=-1)
        if head == "fused":
            return jnp.concatenate([jnp.sum(losses * weights)[None], log_z.ravel()])
        return jnp.stack([losses, log_z])

    (cotangent,) = normal_draw(2706, jax.eval_shape(plain, hidden, table).shape)
    assert_contract(heads, (hidden, table), direction, cotangent, f"the {head} LM head {edge}",
                    truth_fn=plain, plain_fn=plain, without=(*FORWARD_ORDERS, "grad of grad"))


@pytest.mark.parametrize("edge", EDGES)
def test_a_node_perturbation_estimate_holds_its_reverse_orders(edge):
    """A node perturbation objective's loss (`NodePerturbation.loss.estimated`, a custom VJP whose reverse
    rule is its estimate of the gradient from the model's forwards at perturbed sites): over a two-layer
    linear model's squared error, its primal, its estimate and the estimate's own reverse derivative finite
    wherever its float64 run, the same perturbations drawn at float64, is a normal float32; forward mode,
    which an estimate from forwards alone has none of, row 2690's."""
    from flax import linen as nn
    from model_support import parameter_loss

    from dew.inputs import Field, InputSpec
    from dew.objectives.base import Step
    from dew.objectives.perturbation import NodePerturbation
    from dew.objectives.supervised import Supervised

    class Linear(nn.Module):
        @nn.compact
        def __call__(self, x):
            return nn.Dense(2)(nn.Dense(4)(x))

    class Perturbed(NodePerturbation, Supervised):
        """`Supervised` trained by node perturbation."""

    def squared(outputs, batch):
        return jnp.sum(jnp.square(outputs.astype(jnp.float32) - batch["y"]), -1)

    objective = Perturbed(Linear(), squared, inputs=InputSpec(Field("x", (3,))), draws=4)
    variables = objective.init(jax.random.key(0))
    leaves, structure = jax.tree.flatten(variables["params"])
    x, y = normal_draw(2707, (6, 3), (6, 2))
    step = Step(jnp.asarray(0), jax.random.key(1), None)

    def loss(*params):
        batch = {"x": x.astype(params[0].dtype), "y": y.astype(params[0].dtype)}
        return parameter_loss(objective, variables, batch, step)(jax.tree.unflatten(structure, params))

    primals = tuple(leaf * EDGES[edge] for leaf in normal_draw(2708, *(leaf.shape for leaf in leaves)))
    direction = normal_draw(2709, *(leaf.shape for leaf in leaves), scale=EDGES[edge])
    (cotangent,) = normal_draw(2710, ())
    assert_contract(loss, primals, direction, cotangent, f"a node perturbation estimate {edge}",
                    without=FORWARD_ORDERS)


def positive_draw(seed: int, shape: tuple, scale: float) -> jax.Array:
    """A float32 variance a row, positive, at `scale`: a draw's magnitude, at least a tenth of the scale."""
    rng = np.random.default_rng(seed)
    return jnp.asarray((np.abs(rng.standard_normal(shape)) + 0.1) * scale, jnp.float32)


def half_squared_over(value, mean, variance):
    """`|value - mean|**2 / (2 variance)` a row, as written: a Gaussian transition's quadratic term, and its
    KL to a reference of the same variance."""
    return jnp.sum(jnp.square(value - mean), axis=tuple(range(1, value.ndim))) / (2 * variance)


@pytest.mark.parametrize("edge", EDGES)
def test_a_transitions_quadratic_term_holds_every_order(edge):
    """`flow._half_mahalanobis` (a custom JVP whose tangent is the plain formula's, scaled to range): every
    order finite wherever float64 is a normal float32, past where the plain formula overflows, at values,
    means and variances near float32's ends. A value equal to its mean, whose second forward derivative is NaN,
    is ledger 2383's, whose head adds it."""
    from dew.sampling import flow

    value, mean = normal_draw(2283, (3, 4), (3, 4), scale=EDGES[edge])
    variance = positive_draw(2284, (3,), EDGES[edge])
    direction = (*normal_draw(2285, (3, 4), (3, 4), scale=EDGES[edge]),
                 positive_draw(2286, (3,), EDGES[edge]))
    (cotangent,) = normal_draw(2287, (3,))
    assert_contract(flow._half_mahalanobis, (value, mean, variance), direction, cotangent,
                    f"a transition's quadratic term {edge}", truth_fn=half_squared_over)


@pytest.mark.parametrize("edge", EDGES)
def test_a_transitions_kl_holds_every_order(edge):
    """`GaussianTransition.kl` (`flow._gaussian_kl`, Dew's seam over `generation.linear_tangent`: its forward
    mode `_directional`'s, its reverse mode `_kl_pullback`'s): every order it takes finite wherever float64 is
    a normal float32 and the KL as written is, in the mean, the reference and the variance; its gradient takes
    no derivative (ledger 2383)."""
    from dew.sampling import flow

    mean, reference = normal_draw(2288, (3, 4), (3, 4), scale=EDGES[edge])
    variance = positive_draw(2289, (3,), EDGES[edge])
    direction = (*normal_draw(2290, (3, 4), (3, 4), scale=EDGES[edge]),
                 positive_draw(2291, (3,), EDGES[edge]))
    (cotangent,) = normal_draw(2292, (3,))

    def kl(mean, reference, variance):
        return flow.GaussianTransition(mean, variance).kl(reference)

    assert_contract(kl, (mean, reference, variance), direction, cotangent, f"a transition's KL {edge}",
                    truth_fn=half_squared_over, plain_fn=half_squared_over, without=GRADIENT_ORDERS)


@pytest.mark.parametrize("edge", EDGES)
def test_a_kls_forward_mode_holds_its_orders(edge):
    """`flow._directional` (the KL's forward mode, a custom JVP whose tangent reads its tangents'
    exponents): each forward order finite wherever float64 is a normal float32 and the KL as written is; it
    is forward mode alone by design, its reverse mode `_kl_pullback`'s, which the KL's contract holds."""
    from dew.sampling import flow

    mean, reference = normal_draw(2293, (3, 4), (3, 4), scale=EDGES[edge])
    variance = positive_draw(2294, (3,), EDGES[edge])
    direction = (*normal_draw(2295, (3, 4), (3, 4), scale=EDGES[edge]),
                 positive_draw(2296, (3,), EDGES[edge]))
    (cotangent,) = normal_draw(2297, (3,))
    assert_contract(flow._directional, (mean, reference, variance), direction, cotangent,
                    f"a KL's forward mode {edge}", truth_fn=half_squared_over, plain_fn=half_squared_over,
                    without=REVERSE_ORDERS)


@pytest.mark.parametrize("edge", EDGES)
def test_a_guided_prediction_holds_every_order(edge):
    """`guidance.guide` (`_guided`, a custom JVP: the conditional exactly at a unit scale, the affine
    combination's derivative at every scale): every order finite wherever float64 is a normal float32 and the
    combination as written, unconditional + scale (conditional - unconditional), is, a row's scale one."""
    from dew.sampling import guidance

    conditional, unconditional = normal_draw(2298, (4, 5), (4, 5), scale=EDGES[edge])
    scale = jnp.asarray([1.0, 3.5, -0.5, 7.5], jnp.float32)
    direction = (*normal_draw(2299, (4, 5), (4, 5), scale=EDGES[edge]), *normal_draw(2300, (4,)))
    (cotangent,) = normal_draw(2301, (4, 5))

    def affine(conditional, unconditional, scale):
        return unconditional + scale[:, None] * (conditional - unconditional)

    assert_contract(guidance.guide, (conditional, unconditional, scale), direction, cotangent,
                    f"a guided prediction {edge}", truth_fn=affine, plain_fn=affine)
