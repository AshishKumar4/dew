"""jax.nn's range-safe form's derivatives (`dew.nn.attention.range_safe`): its tangent and transpose at every
level, nested forward and reverse over forward, and its pullback under a checked map."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def test_jax_nns_recovered_logits_tangent_keeps_its_directions_bits_at_every_level():
    """jax.nn's form at width 16 in bf16: a query `(2^127, 0, ...)` against a key `(-4, 2^126, ...)`, its
    logit recovered (-2^127, its raw product overflowing), moving along `(0, (1 + 2^-7) 2^-126, ...)`: its
    tangent `(1 + 2^-7) / 4` as the direction's raw product forms it, directly and as an outer derivative
    of an inner one's primal, once and twice nested, the rule's value its declared form at every level
    (`attention.range_safe`), where the outer level differentiated the prescaled product as written and
    lost the direction's low bit (train 1430's review)."""
    from dew.nn.attention import reference_scores

    width = 16
    query = jnp.zeros((1, 1, width), jnp.bfloat16).at[0, 0, 0].set(2.0 ** 127)
    key = jnp.zeros((1, 1, width), jnp.bfloat16).at[0, 0, 0].set(-4.0).at[0, 0, 1].set(2.0 ** 126)
    direction = jnp.zeros_like(query).at[0, 0, 1].set((1.0 + 2.0 ** -7) * 2.0 ** -126)

    def scores(query):
        return reference_scores(query, key, "bqd,bkd->bqk", scale_logits=True)

    def primal_of(function):
        return lambda query: jax.jvp(function, (query,), (direction,))[0]

    logits, direct = jax.jvp(scores, (query,), (direction,))
    np.testing.assert_array_equal(np.asarray(logits)[0, 0, 0], -(2.0 ** 127))
    _, outer = jax.jvp(primal_of(scores), (query,), (direction,))
    _, twice = jax.jvp(primal_of(primal_of(scores)), (query,), (direction,))
    for tangent in (direct, outer, twice):
        np.testing.assert_array_equal(np.asarray(tangent, np.float32)[0, 0, 0],
                                      np.float32((1.0 + 2.0 ** -7) / 4))


@pytest.mark.parametrize("projection", ["primal", "tangent"])
def test_jax_nns_reverse_over_a_jvp_transposes_the_rule_at_both_projections(projection):
    """jax.nn's form at width 16 in fp32 on all-ones operands, reverse over forward: the query's gradient of
    the summed JVP's primal (`[0]`) and of its tangent (`[1]`) along a direction, the tangent's terms behind
    one linear map whose transpose is the rule's own (`attention.range_safe`), where the primal's tangent
    staged the rule's conditional on its linear direction and JAX could not transpose it (train 1440's
    review); each the plain scaled product's own: `scale · 3 · 1` a key for the primal, `scale · Σ d` for
    the tangent."""
    from dew.nn.attention import reference_scores

    width = 16
    query = jnp.ones((1, 2, width), jnp.float32)
    key = jnp.ones((1, 3, width), jnp.float32)
    direction = jnp.ones((1, 2, width), jnp.float32).at[0, 1].set(2.0)
    index = 0 if projection == "primal" else 1

    def projected(query):
        return jax.jvp(lambda query: reference_scores(query, key, "bqd,bkd->bqk", scale_logits=True),
                       (query,), (direction,))[index].sum()

    gradient = np.asarray(jax.grad(projected)(query))
    expected = np.full((1, 2, width), 3.0 / 4) if projection == "primal" else np.zeros((1, 2, width))
    np.testing.assert_array_equal(gradient, expected)


@pytest.mark.mesh(devices=2)
def test_jax_nns_form_pulls_back_alike_under_a_checked_map():
    """jax.nn's form at width 16 in fp32 on all-ones operands under a checked two-device `shard_map`, the
    query replicated and the keys split by token: the operands' cotangents those of one device's call,
    each conditional's branches typed alike (`attention.range_safe`: the operands varied alike first, each
    placement restored after), where a replicated query's raw partial was summed over the map while its
    recovered one varied and JAX refused the conditional (train 1440's review)."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    from dew.nn.attention import reference_scores

    width = 16
    query = jnp.ones((1, 2, width), jnp.float32)
    key = jnp.ones((1, 4, width), jnp.float32)
    mesh = jax.make_mesh((2,), ("token",), devices=jax.devices()[:2])

    def scores(query, key):
        return reference_scores(query, key, "bqd,bkd->bqk", scale_logits=True)

    def pulled(query, key):
        logits, pull = jax.vjp(scores, query, key)
        return pull(jnp.ones_like(logits))

    mapped = jax.jit(jax.shard_map(pulled, mesh=mesh, in_specs=(P(), P(None, "token")),
                                   out_specs=(P(), P(None, "token"))))
    placed = (jax.device_put(query, NamedSharding(mesh, P())),
              jax.device_put(key, NamedSharding(mesh, P(None, "token"))))
    for found, expected in zip(mapped(*placed), pulled(query, key), strict=True):
        np.testing.assert_array_equal(np.asarray(found), np.asarray(expected))


def test_jax_nns_forward_tangent_transposes_at_reverse_over_forward():
    """jax.nn's form at width 16 in fp32, reverse over forward: the key's gradient of the summed tangent of
    its logits along a query direction, `scale · Σ dq` at each key, each bilinear term of the tangent
    through the range-safe rule with its declared transpose (`attention.range_safe`), where its terms'
    conditional on the tangent's own finiteness had no transpose and JAX raised (train 1416's review)."""
    from dew.nn.attention import reference_scores

    width = 16
    query = jnp.ones((1, 2, width), jnp.float32)
    key = jnp.ones((1, 3, width), jnp.float32)
    direction = jnp.ones((1, 2, width), jnp.float32).at[0, 1].set(2.0)

    def tangent(key):
        return jax.jvp(lambda query: reference_scores(query, key, "bqd,bkd->bqk", scale_logits=True),
                       (query,), (direction,))[1].sum()

    gradient = jax.grad(tangent)(key)
    np.testing.assert_array_equal(np.asarray(gradient), np.full((1, 3, width), 3.0 / 4))


def test_a_keys_tangent_moved_along_a_large_query_keeps_its_scale():
    """jax.nn's form at width 16 in fp32, forward over forward: a query of `2^125` against a key of 0 moving
    along -4 has the tangent `-2^125` (its raw product -2^127, finite), and that tangent moving along a
    query direction of `2^127` reads `-2^127`, each order's bilinear terms through the range-safe form
    again (`attention.range_safe`), where plain differentiation of the raw arm formed `-2^129` and read
    -inf (train 1368's review)."""
    from dew.nn.attention import reference_scores

    width = 16
    query = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set(2.0 ** 125)
    key = jnp.zeros((1, 1, width), jnp.float32)

    def tangent(query):
        return jax.jvp(lambda key: reference_scores(query, key, "bqd,bkd->bqk", scale_logits=True), (key,),
                       (key.at[0, 0, 0].set(-4.0),))[1]

    inner = tangent(query)
    np.testing.assert_array_equal(np.asarray(inner)[0, 0, 0], -(2.0 ** 125))
    _, outer = jax.jvp(tangent, (query,), (jnp.zeros_like(query).at[0, 0, 0].set(2.0 ** 127),))
    np.testing.assert_array_equal(np.asarray(outer)[0, 0, 0], -(2.0 ** 127))


def test_jax_nns_form_keeps_a_large_directions_tangent_from_a_zero_query():
    """jax.nn's form's forward-mode derivative at width 16 (`attention.range_safe`): a query of 0 moving along
    `2^127` in one entry against a key of -4 there has the tangent `-2^127`, each term of the derivative
    guarded on its own operands (the direction's among them), where the primal's row factor of 1 formed
    `-2^129` first."""
    from dew.nn.attention import reference_scores

    width = 16
    query = jnp.zeros((1, 1, width), jnp.float32)
    key = jnp.zeros((1, 1, width), jnp.float32).at[0, 0, 0].set(-4.0)

    def scores(query):
        return reference_scores(query, key, "bqd,bkd->bqk", scale_logits=True)

    _, tangent = jax.jvp(scores, (query,), (query.at[0, 0, 0].set(2.0 ** 127),))
    np.testing.assert_array_equal(np.asarray(tangent)[0, 0, 0], -(2.0 ** 127))


def test_a_range_safe_decode_step_is_handed_past_xla():
    """An fp32 decode step on an H100 (one query position over key lengths): the declared cell's xla
    attention (`attention_kernels.DECODE_DECLARED`) where nothing asks range safety, and range-safe the
    decode kernel, which prescales its query, xla's unscaled products handing it on as its whole-call entry
    does (`attention_kernels.DECODE_XLA`, `AttentionCase.range_safe`), where the decode selector's own xla
    entry took it and read a query of `2^127` against a key of 4 as infinity (train 1440's review)."""
    from dew.nn.attention import resolve_implementation
    from dew.nn.kernels import generation

    query = jnp.zeros((1, 1, 8, 16), jnp.float32)
    key = jnp.zeros((1, 64, 8, 16), jnp.float32)
    lengths = jnp.full((1,), 64, jnp.int32)
    with generation.assumed("sm90"):
        plain, safe = (resolve_implementation("auto", query, key, key_value_seq_lengths=lengths,
                                              range_safe=range_safe, published=False)
                       for range_safe in (False, True))
    assert (plain.kernel, safe.kernel) == ("xla", "decode")


@pytest.mark.parametrize(("subscripts", "shapes", "expected"), [
    ("bdq,bkd->bqk", ((1, 16, 16), (1, 2, 16)), (0.5, 4.0)),
    ("bdq,bkd", ((1, 16, 16), (1, 2, 16)), (0.5, 4.0)),
    ("q...d,...kd->...qk", ((2, 3, 16), (3, 4, 16)), (1.0, 0.5)),
], ids=["contracted-first", "implicit", "ellipsis"])
def test_jax_nns_form_pulls_back_a_query_contracted_in_any_place(subscripts, shapes, expected):
    """jax.nn's form at width 16 in fp32 (`attention.range_safe`), its query laid `[B, D, Q]`, contracted
    on its middle axis, an implicit output's (`kq`, its batch summed), and a nonleading ellipsis's (`q...d`
    against `...kd`): the summed logits' gradients exact (`scale · Σ` of the other operand), the recovered
    pullback's query rows read through the product itself (`attention._rows_reached`), where taking the
    query's last label for the contracted one, or its ellipsis for its leading axes, failed its trace
    (train 1455's reviews)."""
    from dew.nn.attention import reference_scores

    query, key = (jnp.ones(shape, jnp.float32) for shape in shapes)
    grads = jax.grad(lambda query, key: reference_scores(query, key, subscripts, scale_logits=True).sum(),
                     argnums=(0, 1))(query, key)
    np.testing.assert_array_equal(np.asarray(grads[0]), np.full(query.shape, expected[0], np.float32))
    np.testing.assert_array_equal(np.asarray(grads[1]), np.full(key.shape, expected[1], np.float32))


def test_jax_nns_form_moves_by_its_live_directions_alone():
    """jax.nn's form's forward-mode derivative along the query alone at width 16 (`attention.range_safe`), a
    key of ones beside a query holding an infinity: the tangent `scale · Σ dq k = 4`, the key's direction a
    symbolic zero whose term is never formed, where the zero direction against the infinite query read NaN
    (train 1455's review)."""
    from dew.nn.attention import reference_scores

    query = jnp.zeros((1, 1, 16), jnp.float32).at[0, 0, 0].set(jnp.inf)
    key = jnp.ones((1, 1, 16), jnp.float32)
    _, tangent = jax.jvp(lambda query: reference_scores(query, key, "bqd,bkd->bqk", scale_logits=True),
                         (query,), (jnp.ones_like(query),))
    np.testing.assert_array_equal(np.asarray(tangent), np.full((1, 1, 1), 4.0, np.float32))
