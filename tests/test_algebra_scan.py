"""A recurrence's chunk forms derived from its token update (`dew.algebra.scan`, docs/design/algebra.md §6.9):
Mamba-2's state-space dual (a scalar decay, a rank-one input, a contraction readout and a skip), the gated
delta rule (a decayed, corrected state: the compact WY form) and Kimi's delta attention (its decay each key's
own), each stated token by token, walked in chunks
through its derived step and outputs (and Mamba-2's two walks merged), each held to the float64 truth
(check 1) against the token-by-token scan, with packed documents' resets too (`Scan.reset`), across which no
gradient flows; and updates no chunk form here holds, refused."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jcore
from reference_error import assert_as_exact

from dew.algebra import SATURATION, cost, linear_scan, scan
from dew.nn.kernels import generation
from dew.nn.precision import float32_matmul

TOKENS, STATE, WIDTH, CHUNK = 128, 16, 32, 32


def update(h, x, dt, B, C, rate, skip):
    """Mamba-2's token update: the state decayed by exp(dt * rate), plus B (x) x dt."""
    del C, skip
    return jnp.exp(dt * rate) * h + B[:, None] * (x * dt)[None, :]


def readout(h, x, dt, B, C, rate, skip):
    """The token's output: C contracted with the state, plus the skip of its input."""
    del dt, B, rate
    return C @ h + skip * x


OPERANDS = {"x": ("time", "width"), "dt": ("time",), "B": ("time", "state"), "C": ("time", "state"),
            "rate": (), "skip": ()}
DECLARED = scan.Scan(update, readout, OPERANDS, walked="time", state=((STATE, WIDTH), jnp.float32))


def tokenwise(x, dt, B, C):
    """The recurrence token by token (`lax.scan`), its outputs and final state."""
    def one(h, token):
        x_t, dt_t, B_t, C_t = token
        h = update(h, x_t, dt_t, B_t, C_t, RATE, SKIP)
        return h, readout(h, x_t, dt_t, B_t, C_t, RATE, SKIP)

    final, outputs = jax.lax.scan(one, jnp.zeros((STATE, WIDTH), x.dtype), (x, dt, B, C))
    return outputs, final


RATE, SKIP = -0.7, 0.3


def draw(seed):
    """Inputs of a token's scale, its steps spanning three decades so some decays reach near zero."""
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.standard_normal((TOKENS, WIDTH)), jnp.float32),
            jnp.asarray(10.0 ** rng.uniform(-3, 0, TOKENS), jnp.float32),
            jnp.asarray(rng.standard_normal((TOKENS, STATE)) / np.sqrt(STATE), jnp.float32),
            jnp.asarray(rng.standard_normal((TOKENS, STATE)) / np.sqrt(STATE), jnp.float32))


def deep_draw(seed):
    """`draw`'s inputs with steps of one to two: each token's log decay -0.7 to -1.4, so a chunk's reach
    e^-45, where a log decay's fp32 ulp is 4e-6 and an exponent rounded whole before its exponential loses
    that much of every decay."""
    x, _, B, C = draw(seed)
    return x, jnp.asarray(np.random.default_rng(seed + 1).uniform(1, 2, TOKENS), jnp.float32), B, C


def operands(x, dt, B, C) -> dict:
    return {"x": x, "dt": dt, "B": B, "C": C, "rate": jnp.float32(RATE), "skip": jnp.float32(SKIP)}


def chunk_of(values: dict, index: int, axes: dict = OPERANDS) -> dict:
    tokens = slice(index * CHUNK, (index + 1) * CHUNK)
    return {name: value[tokens] if axes[name][:1] == ("time",) else value
            for name, value in values.items()}


def walk(forms, values, chunks: range, state, axes: dict = OPERANDS):
    outputs = []
    # A walk that carries its transfer composes each chunk's transition.
    transition = isinstance(state, scan.Walk) and state.transfer is not None
    for index in chunks:
        chunk = chunk_of(values, index, axes)
        prepared = forms.prepare(chunk, transition=transition)
        outputs.append(forms.mapped(prepared, state))
        state = forms.step(state, prepared)
    return outputs, state


def test_the_derived_chunks_compute_the_token_scan():
    """The derived chunk outputs and step over every chunk in order: on par with the token-by-token scan,
    outputs and final state, against the float64 truth (check 1)."""
    forms = scan.derive_scan(DECLARED, saturation=SATURATION)

    def derived(x, dt, B, C):
        values = operands(x, dt, B, C)
        outputs, state = walk(forms, values, range(TOKENS // CHUNK), forms.init({}))
        return jnp.concatenate(outputs), forms.finish(state, values)

    assert_as_exact(jax.jit(derived), jax.jit(tokenwise), draw, "ssd chunks", seeds=16)


def test_two_walks_merged_give_the_final_state():
    """The tokens walked in two halves, the second from a zero state, merged: the final state on par with
    the token-by-token scan's."""
    forms = scan.derive_scan(DECLARED, saturation=SATURATION)
    half = TOKENS // CHUNK // 2

    def merged(x, dt, B, C):
        values = operands(x, dt, B, C)
        _, first = walk(forms, values, range(half), forms.init(chunk_of(values, 0)))
        _, second = walk(forms, values, range(half, 2 * half), forms.init(chunk_of(values, 0)))
        return forms.finish(forms.merge(first, second, values), values)

    assert_as_exact(jax.jit(merged), jax.jit(lambda *xs: tokenwise(*xs)[1]), draw, "ssd merged", seeds=16)


def weighted_readout(h, x, dt, B, C, a, rate, skip):
    """A readout that weighs the state's rows by the token's own vector `a` before contracting it with C."""
    del dt, B, rate
    return C @ (a[:, None] * h) + skip * x


def weighted_update(h, x, dt, B, C, a, rate, skip):
    del a
    return update(h, x, dt, B, C, rate, skip)


def test_a_readout_that_weighs_the_states_rows_computes_the_token_scan():
    """Mamba-2's update read out through the state's rows weighed by a vector of the token's
    (`C @ (a * h)`), walked in chunks: outputs on par with the token-by-token scan (check 1)."""
    weighted = {**OPERANDS, "a": ("time", "state")}
    forms = scan.derive_scan(scan.Scan(weighted_update, weighted_readout, weighted, walked="time",
                                       state=((STATE, WIDTH), jnp.float32)), saturation=SATURATION)

    def derived(x, dt, B, C, a):
        values = {**operands(x, dt, B, C), "a": a}
        outputs = []
        state = forms.init({})
        for index in range(TOKENS // CHUNK):
            tokens = slice(index * CHUNK, (index + 1) * CHUNK)
            chunk = {name: value[tokens] if weighted[name][:1] == ("time",) else value
                     for name, value in values.items()}
            prepared = forms.prepare(chunk)
            outputs.append(forms.mapped(prepared, state))
            state = forms.step(state, prepared)
        return jnp.concatenate(outputs)

    def tokens(x, dt, B, C, a):
        def one(h, token):
            h = weighted_update(h, *token, RATE, SKIP)
            return h, weighted_readout(h, *token, RATE, SKIP)

        return jax.lax.scan(one, jnp.zeros((STATE, WIDTH), x.dtype), (x, dt, B, C, a))[1]

    def weights(seed):
        return (*draw(seed), jnp.asarray(np.random.default_rng(seed + 1).uniform(0.1, 2.0, (TOKENS, STATE)),
                                         jnp.float32))

    assert_as_exact(jax.jit(derived), jax.jit(tokens), weights, "a readout weighing the rows", seeds=16)


def columnwise(h, x, dt, B, C, rate, skip):
    del C, skip
    return jnp.exp(dt * rate * jnp.arange(WIDTH, dtype=jnp.float32))[None, :] * h + B[:, None] * x[None, :]


def squared(h, x, dt, B, C, rate, skip):
    del C, skip
    return jnp.exp(dt * rate) * h * h + B[:, None] * x[None, :]


@pytest.mark.parametrize("refused", [columnwise, squared])
def test_an_update_no_chunk_form_holds_is_refused(refused):
    """A decay along the state's second axis (each value column its own) and a state read twice have no
    chunk form here: refused before a form is built."""
    forms = scan.derive_scan(scan.Scan(refused, readout, OPERANDS, walked="time",
                                       state=((STATE, WIDTH), jnp.float32)), saturation=SATURATION)
    with pytest.raises(scan.Refused):
        forms.prepare(chunk_of(operands(*draw(0)), 0))


def gated_delta(h, q, k, v, g, beta, exponential=jnp.exp):
    """The gated delta rule: the state decayed and its key's old value corrected toward the new one,
    `h = exp(g) (h - beta k (k^T h)) + beta k v^T`, its decay `exponential(g)` (`jnp.exp2`: in base 2, fla's
    gates)."""
    del q
    return exponential(g) * (h - beta * k[:, None] * (k @ h)[None, :]) + beta * k[:, None] * v[None, :]


def gated_delta_written_as_error(h, q, k, v, g, beta):
    """The same rule as an error correction: `h = exp(g) h + beta k (v - exp(g) k^T h)^T`."""
    del q
    decayed = jnp.exp(g)
    return decayed * h + beta * jnp.outer(k, v - decayed * (k @ h))


def gated_readout(h, q, k, v, g, beta):
    del k, v, g, beta
    return (q @ h) * KEY ** -0.5


KEY, VALUE = 16, 32
GATED = {"q": ("time", "key"), "k": ("time", "key"), "v": ("time", "value"), "g": ("time",),
         "beta": ("time",)}


def gated_draw(seed):
    """Unit keys (a correction of strength beta < 1 stays a contraction), decays to 1e-3 a token."""
    rng = np.random.default_rng(seed)
    k = rng.standard_normal((TOKENS, KEY))
    return (jnp.asarray(rng.standard_normal((TOKENS, KEY)), jnp.float32),
            jnp.asarray(k / np.linalg.norm(k, axis=-1, keepdims=True), jnp.float32),
            jnp.asarray(rng.standard_normal((TOKENS, VALUE)), jnp.float32),
            jnp.asarray(-(10.0 ** rng.uniform(-3, 0, TOKENS)), jnp.float32),
            jnp.asarray(rng.uniform(0.05, 0.95, TOKENS), jnp.float32))


def gated_tokenwise(update_fn):
    def run(q, k, v, g, beta):
        def one(h, token):
            h = update_fn(h, *token)
            return h, gated_readout(h, *token)

        final, outputs = jax.lax.scan(one, jnp.zeros((KEY, VALUE), q.dtype), (q, k, v, g, beta))
        return outputs, final

    return run


def kimi_delta(h, q, k, v, g, beta):
    """Kimi's delta attention: each key's row decayed by its own gate, then the key's old value corrected,
    `h = (I - beta k k^T) diag(exp(g)) h + beta k v^T`."""
    del q
    decayed = jnp.exp(g)[:, None] * h
    return decayed - beta * k[:, None] * (k @ decayed)[None, :] + beta * k[:, None] * v[None, :]


def kimi_delta_written_as_error(h, q, k, v, g, beta):
    del q
    decayed = jnp.exp(g)[:, None] * h
    return decayed + beta * jnp.outer(k, v - k @ decayed)


KIMI = {**GATED, "g": ("time", "key")}


def kimi_draw(seed):
    """As `gated_draw`, each key's decay its own, down to 1e-3 a token."""
    q, k, v, _, beta = gated_draw(seed)
    rng = np.random.default_rng(seed + 1)
    return q, k, v, jnp.asarray(-(10.0 ** rng.uniform(-3, 0, (TOKENS, KEY))), jnp.float32), beta


def gated_walk(forms, axes: dict):
    """`forms`' chunks walked over a gated rule's operands (`axes`, `GATED`'s or `KIMI`'s) from a zero state:
    the outputs and the final state, as `gated_tokenwise` gives them."""
    def walked(q, k, v, g, beta):
        values = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
        outputs, state = walk(forms, values, range(TOKENS // CHUNK), forms.init({}), axes)
        return jnp.concatenate(outputs), forms.finish(state, values)

    return walked


@pytest.mark.parametrize(("update_fn", "operands", "draw_fn"), [
    (gated_delta, GATED, gated_draw), (gated_delta_written_as_error, GATED, gated_draw),
    (kimi_delta, KIMI, kimi_draw), (kimi_delta_written_as_error, KIMI, kimi_draw)],
    ids=["gated_delta", "gated_delta_as_error", "kimi_delta", "kimi_delta_as_error"])
def test_the_delta_rules_derived_chunks_compute_their_token_scans(update_fn, operands, draw_fn):
    """The gated delta rule (a scalar decay) and Kimi's delta attention (each key's decay its own), each
    written as a decayed corrected state and as an error correction, walked in chunks through the derived
    WY form: on par with the token-by-token scan, outputs and final state, against the float64 truth (check
    1), decays down to 1e-3 a token."""
    forms = scan.derive_scan(scan.Scan(update_fn, gated_readout, operands, walked="time",
                                       state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
    assert_as_exact(jax.jit(gated_walk(forms, operands)), jax.jit(gated_tokenwise(update_fn)), draw_fn,
                    update_fn.__name__, seeds=16)


@pytest.mark.parametrize(("update_fn", "operands", "draw_fn"), [
    (gated_delta, GATED, gated_draw), (kimi_delta, KIMI, kimi_draw)], ids=["gated_delta", "kimi_delta"])
def test_two_delta_rule_walks_merged_give_the_final_state(update_fn, operands, draw_fn):
    """The gated delta rule's and Kimi's delta attention's tokens walked in two halves, the second from a zero
    state carrying its transfer (the corrections' keys-by-keys map of the state entering it), merged: the
    final state on par with the token-by-token scan's, against the float64 truth (check 1)."""
    forms = scan.derive_scan(scan.Scan(update_fn, gated_readout, operands, walked="time",
                                       state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
    half = TOKENS // CHUNK // 2

    def merged(q, k, v, g, beta):
        values = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
        states = [walk(forms, values, chunks, forms.init(chunk_of(values, 0, operands)), operands)[1]
                  for chunks in (range(half), range(half, 2 * half))]
        return forms.finish(forms.merge(*states, values), values)

    assert_as_exact(jax.jit(merged), jax.jit(lambda *xs: gated_tokenwise(update_fn)(*xs)[1]), draw_fn,
                    f"{update_fn.__name__} merged", seeds=16)


@pytest.mark.parametrize(("update_fn", "operands", "draw_fn"), [
    (gated_delta, GATED, gated_draw), (kimi_delta, KIMI, kimi_draw)], ids=["gated_delta", "kimi_delta"])
def test_two_delta_rule_walks_merged_through_an_issuer_give_the_final_state(update_fn, operands, draw_fn):
    """The two halves' walks of the delta rules stepped, their transfers composed from each chunk's prepared
    transition and joined, every product a mainloop's issuer's by its role (`issue`, Dew's own products
    here): the final state on par with the token-by-token scan's against the float64 truth (check 1), the
    transfer held as the issuer's own value, which no operation but the issuer's reads."""
    forms = scan.derive_scan(scan.Scan(update_fn, gated_readout, operands, walked="time",
                                       state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
    half = TOKENS // CHUNK // 2
    roles = set()

    class Held:
        """A transfer as a mainloop's issuer holds it (its own tiles), which only the issuer reads."""

        def __init__(self, value):
            self.value = value

    def issue(role, left, right):
        roles.add(role)
        left, right = (x.value if isinstance(x, Held) else x for x in (left, right))
        made = right * left[None, :] if role == "transfer_decay" else float32_matmul(left, right)
        return Held(made) if role in ("transfer_compose", "transfer_decay") else made

    def merged(q, k, v, g, beta):
        values = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
        states = []
        for chunks in (range(half), range(half, 2 * half)):
            state = forms.init(chunk_of(values, 0, operands))
            for index in chunks:
                prepared = forms.prepare(chunk_of(values, index, operands), issue, transition=True)
                state = forms.step(state, prepared, issue)
            states.append(state)
        return forms.finish(forms.merge(*states, values, issue), values)

    assert_as_exact(jax.jit(merged), jax.jit(lambda *xs: gated_tokenwise(update_fn)(*xs)[1]), draw_fn,
                    f"{update_fn.__name__} merged through an issuer", seeds=16)
    assert {"transfer_compose", "transfer_apply"} <= roles, roles


@pytest.mark.parametrize("target", ["sm86", "sm90"])
def test_a_chunk_form_any_target_extracts_computes_the_token_scan(target):
    """The gated delta rule's chunks derived under an A10 (its step's latency unstated, so the token loop is
    priced at its work alone and extracted) and an H100 (WY's): on par with the token scan, outputs and
    final state, against the float64 truth (check 1)."""
    with generation.assumed(target):
        forms = scan.derive_scan(scan.Scan(gated_delta, gated_readout, GATED, walked="time",
                                           state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
        assert_as_exact(jax.jit(gated_walk(forms, GATED)), jax.jit(gated_tokenwise(gated_delta)), gated_draw,
                        f"gated delta on {target}", seeds=8)


def test_a_chunks_form_is_the_one_its_target_derives():
    """A chunk's form asked for under one target and then another is each target's own derivation: an
    H100's (WY with its inverse by doubling) and an A10's (by substitution, its span unpriced)."""
    extents = linear_scan.Extents(64, 128, 128, 1)
    for target in ("sm90", "sm86", "sm90"):
        with generation.assumed(target):
            assert scan._form(extents, corrected=True) == linear_scan.derive(
                extents, corrects=True, schedule=cost.fused(cost.target()))


def test_a_scans_carry_derived_for_one_target_serves_no_other():
    """One derived scan's chunk prepared under an H100, an A100 and an H100 again takes each target's own
    carry split (their chunk form alike), and the H100's again where its target returns: a split priced for
    one GPU is not another's."""
    forms = scan.derive_scan(scan.Scan(gated_delta, gated_readout, GATED, walked="time",
                                       state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
    chunk = chunk_of(dict(zip(GATED, gated_draw(0), strict=True)), 0, GATED)
    splits = []
    for target in ("sm90", "sm80", "sm90"):
        with generation.assumed(target):
            splits.append(forms.prepare(chunk).split)
    assert splits[1] is not splits[0]
    assert splits[2] is splits[0]


@pytest.mark.parametrize(("update_fn", "operands", "draw_fn"), [
    (gated_delta, GATED, gated_draw), (kimi_delta, KIMI, kimi_draw)], ids=["gated_delta", "kimi_delta"])
def test_a_derived_scans_products_are_its_issuers_each_by_its_role(update_fn, operands, draw_fn):
    """A derived scan's chunk walked through a mainloop's issuer (`issue(role, lhs, rhs)`) at prepare, step
    and mapped: every product of its forms is the issuer's, under the roles a chunk's form makes (its gram,
    a per-key decay's factored around each sub-chunk's reference among them, its inverse's doubling, its
    correction, write and readouts), and the walk computes what Dew's own products do, bit for bit."""
    forms = scan.derive_scan(scan.Scan(update_fn, gated_readout, operands, walked="time",
                                       state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
    roles = []

    def issue(role, left, right):
        roles.append(role)
        return float32_matmul(left, right)

    chunk = chunk_of(dict(zip(operands, draw_fn(0), strict=True)), 0, operands)
    state = forms.init(())
    plain = forms.prepare(chunk)
    expected = forms.mapped(plain, state), forms.step(state, plain)
    issued = forms.prepare(chunk, issue)
    made = forms.mapped(issued, state, issue), forms.step(state, issued, issue)
    np.testing.assert_array_equal(np.asarray(made[0]), np.asarray(expected[0]))
    np.testing.assert_array_equal(np.asarray(made[1][1]), np.asarray(expected[1][1]))
    assert {"gram", "inverse_merge", "correct", "carry", "readout_memory"} <= set(roles), roles


def test_a_derived_scan_holds_its_latest_splits_alone():
    """A derived scan keeps its latest carry splits alone (`splits_held`), the least recently prepared
    dropped: held to one, an H100's split prepared after an A100's is derived again."""
    forms = scan.derive_scan(scan.Scan(gated_delta, gated_readout, GATED, walked="time",
                                       state=((KEY, VALUE), jnp.float32)),
                             saturation=SATURATION, splits_held=1)
    chunk = chunk_of(dict(zip(GATED, gated_draw(0), strict=True)), 0, GATED)
    splits = []
    for target in ("sm90", "sm90", "sm80", "sm90"):
        with generation.assumed(target):
            splits.append(forms.prepare(chunk).split)
    assert splits[1] is splits[0]
    assert splits[3] is not splits[0]


@pytest.mark.parametrize(("update_fn", "operands", "draw_fn"),
                         [(gated_delta, GATED, gated_draw), (kimi_delta, KIMI, kimi_draw)],
                         ids=["gated_delta", "kimi_delta"])
def test_the_delta_rules_carry_reads_the_state_through_one_product_where_its_state_is_bounded(
        update_fn, operands, draw_fn):
    """The delta rule's split (`carry.split`), its keys, write gates, log decays and values ranged (unit keys,
    so tokens' corrections overlap by at most their gates, under 1; decays at most 1; values under the draws'
    8): where its entering state is ranged too
    (`Scan.ranges`' "state"), its carry (`step` over a chunk's prepared values) reads the state through one
    product, a matrix the chunk prepared times the state, the solve, the decays and every other product among
    the prepared values. With no range on the state, the solve's distribution over `W h` makes an
    intermediate of the state's magnitude no number bounds (§6.3), and the carry keeps it: more than one
    product reads the state."""
    names = ("q", "k", "v", "g", "beta")
    chunk = {name: value[:CHUNK] for name, value in zip(names, draw_fn(0), strict=True)}
    ranges = {"k": 1.0, "beta": 1.0, "overlap": 1.0, "g": (-np.inf, 0.0), "v": 8.0}

    def carried_products(declared: dict) -> list[str]:
        forms = scan.derive_scan(scan.Scan(update_fn, gated_readout, operands, walked="time",
                                           state=((KEY, VALUE), jnp.float32), ranges=declared),
                                 saturation=SATURATION)
        prepared = forms.prepare(chunk)
        carry = jax.make_jaxpr(lambda h: forms.step((jnp.zeros((KEY,), jnp.float32), h), prepared))(
            jnp.zeros((KEY, VALUE)))
        return list(_primitives(carry.jaxpr))

    bounded = carried_products({**ranges, "state": 64.0})
    assert bounded.count("dot_general") == 1, bounded
    assert not any(name.startswith("custom_vjp") for name in bounded), bounded
    assert carried_products(ranges).count("dot_general") > 1


def test_a_delta_rules_carry_stays_finite_wherever_the_token_scan_is():
    """The gated delta rule's two-token chunk, its keys sixteen ones (each element under its declared 1, two
    tokens' overlap 16), unit write gates and decays, values of 2^124 and an entering state of 2^120, each
    ranged (`Scan.ranges`): the keys read the state as the values exactly, so every write is zero and the
    token scan finite. The derived chunk is finite too (bible §6): its solve's inverse is bounded over the
    keys' contraction, 17 a pair here, so distributing it over the values, past float32's largest, is not
    taken."""
    tokens = 2
    chunk = {"q": jnp.full((tokens, KEY), 2.0 ** -10), "k": jnp.ones((tokens, KEY)),
             "v": jnp.full((tokens, VALUE), 2.0 ** 124), "g": jnp.zeros((tokens,)),
             "beta": jnp.ones((tokens,))}
    ranges = {"k": 1.0, "beta": 1.0, "g": (-np.inf, 0.0), "v": 2.0 ** 124, "state": 2.0 ** 120}
    declared = scan.Scan(gated_delta, gated_readout, GATED, walked="time", state=((KEY, VALUE), jnp.float32),
                         ranges=ranges)
    forms = scan.derive_scan(declared, saturation=SATURATION)
    h = jnp.full((KEY, VALUE), 2.0 ** 120)
    expected = []
    for index in range(tokens):
        token = [chunk[name][index] for name in GATED]
        expected.append(gated_readout(h := gated_delta(h, *token), *token))
    assert np.all(np.isfinite(np.asarray(expected))) and np.all(np.isfinite(np.asarray(h)))
    entering = (jnp.zeros((KEY,), jnp.float32), jnp.full((KEY, VALUE), 2.0 ** 120))
    prepared = forms.prepare(chunk)
    outputs, (_, leaving) = forms.mapped(prepared, entering), forms.step(entering, prepared)
    assert np.all(np.isfinite(np.asarray(outputs))) and np.all(np.isfinite(np.asarray(leaving)))


def test_a_chunks_decay_gradient_reads_no_pair_of_its_future():
    """The gated delta rule's two-token chunk whose first query and second key are parallel at 2^128 (a pair
    no decay joins: the second token is the first's future) and every pair a decay joins orthogonal or small:
    the token scan's outputs and decay gradients are finite, and the derived chunk's are too, its future
    pair's product selected away before its decay's derivative reads it."""
    tokens = 2
    q = jnp.zeros((tokens, KEY)).at[0, 0].set(2.0 ** 127).at[1, 1].set(1.0)
    k = jnp.zeros((tokens, KEY)).at[0, 1].set(1.0).at[1, 0].set(2.0)
    chunk = {"q": q, "k": k, "v": jnp.ones((tokens, VALUE)), "g": jnp.zeros((tokens,)),
             "beta": jnp.ones((tokens,))}
    forms = scan.derive_scan(scan.Scan(gated_delta, gated_readout, GATED, walked="time",
                                       state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)

    def derived(g):
        prepared = forms.prepare({**chunk, "g": g})
        return jnp.sum(forms.mapped(prepared, forms.init({})))

    def tokenwise(g):
        return jnp.sum(gated_tokenwise(gated_delta)(chunk["q"], chunk["k"], chunk["v"], g, chunk["beta"])[0])

    for fn in (tokenwise, derived):
        value, gradient = jax.value_and_grad(fn)(chunk["g"])
        assert np.isfinite(value) and np.all(np.isfinite(gradient)), fn.__name__


@pytest.mark.parametrize("tokens", [2, 256])
def test_a_chunk_of_any_finite_nonpositive_decays_stays_finite(tokens):
    """The gated delta rule's chunk whose log decays are each -2^127 (a decay of zero), its keys, values and
    write gates unit, over two tokens, and over 256 of zero operands under declared unit keys and gates
    (whose solve's inverse bound, 17^255, passes float64's range): the token scan is finite, and so is the
    derived chunk, its decays' prefix held finite (`scan.FLOOR`) and its derivation unbounded where its bound
    passes the range."""
    ranges = {"k": 1.0, "beta": 1.0, "g": (-np.inf, 0.0)}
    declared = scan.Scan(gated_delta, gated_readout, GATED, walked="time", state=((KEY, VALUE), jnp.float32),
                         ranges=ranges)
    forms = scan.derive_scan(declared, saturation=SATURATION)
    ones = 1.0 if tokens == 2 else 0.0
    chunk = {"q": jnp.full((tokens, KEY), ones / 4), "k": jnp.full((tokens, KEY), ones / 4),
             "v": jnp.full((tokens, VALUE), ones), "g": jnp.full((tokens,), -(2.0 ** 127)),
             "beta": jnp.full((tokens,), ones)}
    h, expected = jnp.zeros((KEY, VALUE)), []
    for index in range(tokens):
        token = [chunk[name][index] for name in GATED]
        expected.append(gated_readout(h := gated_delta(h, *token), *token))
    assert np.all(np.isfinite(np.asarray(expected)))
    assert np.all(np.isfinite(np.asarray(forms.mapped(forms.prepare(chunk), forms.init({})))))


def test_a_chunks_later_decays_keep_their_digits_after_a_vanished_one():
    """A scalar SSD chunk whose log decays are -2^127, -2^70 and -1 (two vanished, then e^-1), its writes
    `2^127` from the second token on: the token scan's output is `(1 + e^-1) 2^127`, finite; the derived
    chunk's is too, its last decay e^-1, not the 1 a prefix holding -2^70 rounds its -1 into, whose two
    writes sum past float32's largest."""
    forms = scalar_ssd()
    writes = jnp.asarray([[0.0], [2.0 ** 127], [2.0 ** 127]])
    chunk = {"q": jnp.ones((3, 1)), "b": jnp.ones((3, 1)), "v": writes,
             "g": jnp.asarray([-(2.0 ** 127), -(2.0 ** 70), -1.0])}
    h, expected = jnp.zeros((1, 1)), []
    for index in range(3):
        token = [chunk[name][index] for name in SCALAR_SSD]
        expected.append(scalar_read(h := scalar_step(h, *token), *token))
    outputs = forms.mapped(forms.prepare(chunk), forms.init({}))
    assert np.all(np.isfinite(np.asarray(expected))) and np.all(np.isfinite(np.asarray(outputs)))


def test_a_chunk_of_a_vast_declared_step_derives():
    """A float64 SSD scan whose step is declared at 2^1023 and its input at 2: its token values' bounds pass
    float64's range, so the derivation holds none of them (no `OverflowError`), and a chunk of zero steps maps
    finite."""
    def ssd(h, q, b, x, dt):
        return jnp.exp(-dt) * h + b[:, None] * (x * dt)[None, :]

    def read(h, q, b, x, dt):
        return q @ h

    operands = {"q": ("time", "key"), "b": ("time", "key"), "x": ("time", "value"), "dt": ("time",)}
    with jax.enable_x64(new_val=True):
        forms = scan.derive_scan(scan.Scan(ssd, read, operands, walked="time", state=((2, 2), jnp.float64),
                                           ranges={"dt": 2.0 ** 1023, "x": 2.0, "q": 1.0, "b": 1.0}),
                                 saturation=SATURATION)
        chunk = {"q": jnp.ones((4, 2)), "b": jnp.ones((4, 2)), "x": jnp.ones((4, 2)), "dt": jnp.zeros((4,))}
        assert np.all(np.isfinite(np.asarray(forms.mapped(forms.prepare(chunk), forms.init({})))))


def scalar_step(h, q, b, v, g):
    """A scalar SSD's token update, `h = exp(g) h + b v`."""
    return jnp.exp(g) * h + b[:, None] * v[None, :]


def scalar_read(h, q, b, v, g):
    """A scalar SSD's readout, `q h`."""
    return q @ h


SCALAR_SSD = {"q": ("time", "key"), "b": ("time", "key"), "v": ("time", "value"), "g": ("time",)}
"""A scalar SSD's operands and their axes."""


def scalar_ssd(dtype=jnp.float32):
    """A scalar SSD's derived forms (`scalar_step`, `scalar_read`) of a `dtype` state."""
    return scan.derive_scan(scan.Scan(scalar_step, scalar_read, SCALAR_SSD, walked="time",
                                      state=((1, 1), dtype), ranges={"g": (-np.inf, 0.0)}),
                            saturation=SATURATION)


def ones_chunk(g, dtype=jnp.float32):
    """A chunk of unit readers, keys and writes at log decays `g`."""
    tokens = g.shape[0]
    ones = jnp.ones((tokens, 1), dtype)
    return {"q": ones, "b": ones, "v": ones, "g": g}


@pytest.mark.parametrize("decays", [[-100.0, -(2.0 ** 127)], [-(2.0 ** 127), -(2.0 ** 127)]])
def test_a_walk_through_a_vanished_decay_differentiates_and_merges_finite(decays):
    """A two-token scalar SSD chunk whose last decay vanishes (after a live one, or both): its final state 1,
    its decays' gradient and their gradient's tangent (forward over reverse, through the portable pullback),
    and its walk merged after another's, all finite, as the token scan's are."""
    forms = scalar_ssd()

    def final(g):
        return jnp.sum(forms.step(forms.init({}), forms.prepare(ones_chunk(g)))[1])

    g = jnp.asarray(decays, jnp.float32)
    value, gradient = jax.value_and_grad(final)(g)
    _, tangent = jax.jvp(jax.grad(final), (g,), (jnp.ones_like(g),))
    start = forms.init(ones_chunk(g))
    merged = forms.merge(start, forms.step(start, forms.prepare(ones_chunk(g))), None)
    leaves = (value, gradient, tangent, *jax.tree.leaves(merged))
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in leaves)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float16, jnp.bfloat16], ids=lambda d: jnp.dtype(d).name)
def test_a_walk_carries_its_state_in_one_dtype(dtype):
    """A scalar SSD of a float64, float16 or bfloat16 state walked over two chunks by `lax.scan`: its carry
    keeps its type through the steps (the walk's decay and state at their accumulation's, a float64 state's
    float64), its final state finite and of the declared dtype."""
    with jax.enable_x64(new_val=True):
        forms = scalar_ssd(dtype)
        prepared = jax.vmap(forms.prepare)(jax.tree.map(lambda leaf: jnp.stack([leaf, leaf]),
                                                       ones_chunk(jnp.zeros((4,), dtype), dtype)))
        state, _ = jax.lax.scan(lambda carried, chunk: (forms.step(carried, chunk), None), forms.init({}),
                                prepared)
        final = forms.finish(state, {})
        assert final.dtype == dtype and np.isfinite(np.asarray(final, np.float64)).all()
        assert state[0].dtype == jnp.dtype(jnp.float64 if dtype == jnp.float64 else jnp.float32)


def _primitives(jaxpr):
    """Each equation's primitive in `jaxpr` and in every jaxpr its equations hold (a jit's, a custom
    derivative's), a cond's one branch (Dew's product a choice by platform, each branch one product)."""
    for equation in jaxpr.eqns:
        yield equation.primitive.name
        inner = ([equation.params["branches"][-1].jaxpr] if equation.primitive.name == "cond"
                 else jcore.jaxprs_in_params(equation.params))
        for held in inner:
            yield from _primitives(held)


def adjoint(forms, chunk_of_values, count: int, issue=None, kept: bool = False):
    """A loss's gradient through the derived forms: the chunks walked forward from a zero state (each
    entering state kept), then back through each chunk's `backward`, the cotangents of the outputs and the
    final state the loss's (`COTANGENT`, a fixed draw), every product a mainloop's `issue`'s where given,
    each chunk's solve the forward's inverse where `kept`, as a training forward keeps it."""
    def run(values, cotangents):
        d_outputs, d_final = cotangents
        states, inverses = [forms.init({})], []
        for index in range(count):
            prepared = forms.prepare(chunk_of_values(values, index))
            inverses.append(prepared.inverse if kept else None)
            states.append(forms.step(states[-1], prepared))
        d_state, grads = d_final, []
        for index in reversed(range(count)):
            chunk = chunk_of_values(values, index)
            d_state, d_chunk = forms.backward(chunk, states[index], d_state,
                                              d_outputs[index * CHUNK:(index + 1) * CHUNK], issue,
                                              inverse=inverses[index])
            grads.append(d_chunk)
        # A token operand's gradient each chunk's; a constant's (a rate, a skip) summed over them.
        return {name: jnp.concatenate([grad[name] for grad in reversed(grads)]) if grads[0][name].ndim
                else sum(grad[name] for grad in grads) for name in grads[0]}

    return run


def gradient_of(tokenwise_fn, names):
    """The token scan's gradient of the same loss, each operand's."""
    def run(values, cotangents):
        _, pullback = jax.vjp(lambda *xs: tokenwise_fn(*xs), *[values[name] for name in names])
        return dict(zip(names, pullback(cotangents), strict=True))

    return run


def cotangents_of(seed, outputs_shape, state_shape):
    rng = np.random.default_rng(seed + 7)
    return (jnp.asarray(rng.standard_normal(outputs_shape), jnp.float32),
            jnp.asarray(rng.standard_normal(state_shape), jnp.float32))


@pytest.mark.parametrize("draw_fn", [draw, deep_draw], ids=["steps over three decades", "deep decays"])
def test_the_ssd_chunks_adjoint_is_the_token_scans_gradient(draw_fn):
    """Mamba-2's chunk adjoint (`backward`), walked back over the chunks: each token operand's gradient on par
    with the token-by-token scan's against the float64 truth (check 1), its steps' too where a chunk's
    decays reach e^-45."""
    forms = scan.derive_scan(DECLARED, saturation=SATURATION)
    names = ("x", "dt", "B", "C")

    def chunked(values, index):
        return chunk_of({**values, "rate": jnp.float32(RATE), "skip": jnp.float32(SKIP)}, index)

    derived = adjoint(forms, chunked, TOKENS // CHUNK)

    def draw_with_cotangents(seed):
        return (*draw_fn(seed), *cotangents_of(seed, (TOKENS, WIDTH), (STATE, WIDTH)))

    def as_derived(x, dt, B, C, d_out, d_final):
        grads = derived({"x": x, "dt": dt, "B": B, "C": C}, (d_out, d_final))
        return tuple(grads[name] for name in names)

    def as_tokens(x, dt, B, C, d_out, d_final):
        grads = gradient_of(tokenwise, names)({"x": x, "dt": dt, "B": B, "C": C}, (d_out, d_final))
        return tuple(grads[name] for name in names)

    assert_as_exact(jax.jit(as_derived), jax.jit(as_tokens), draw_with_cotangents, "ssd adjoint", seeds=16)


def assert_delta_adjoint(forms, update_fn, draw_fn, label: str, seeds: int = 16, issue=None,
                         kept: bool = False) -> None:
    """`forms`' chunk adjoint of the delta rule `update_fn`, walked back from each chunk's entering state:
    each token operand's gradient on par with the token-by-token scan's against the float64 truth (check
    1), over `draw_fn`'s draws and a fixed cotangent's."""
    names = ("q", "k", "v", "g", "beta")

    def chunked(values, index):
        return {name: value[index * CHUNK:(index + 1) * CHUNK] for name, value in values.items()}

    derived = adjoint(forms, chunked, TOKENS // CHUNK, issue, kept)

    def draw_with_cotangents(seed):
        return (*draw_fn(seed), *cotangents_of(seed, (TOKENS, VALUE), (KEY, VALUE)))

    def as_derived(q, k, v, g, beta, d_out, d_final):
        grads = derived(dict(zip(names, (q, k, v, g, beta), strict=True)), (d_out, d_final))
        return tuple(grads[name] for name in names)

    def as_tokens(q, k, v, g, beta, d_out, d_final):
        values = dict(zip(names, (q, k, v, g, beta), strict=True))
        grads = gradient_of(gated_tokenwise(update_fn), names)(values, (d_out, d_final))
        return tuple(grads[name] for name in names)

    assert_as_exact(jax.jit(as_derived), jax.jit(as_tokens), draw_with_cotangents, label, seeds=seeds)


@pytest.mark.parametrize(("update_fn", "operands", "draw_fn"), [
    (gated_delta, GATED, gated_draw), (kimi_delta, KIMI, kimi_draw)], ids=["gated_delta", "kimi_delta"])
def test_the_delta_rules_chunk_adjoint_is_the_token_scans_gradient(update_fn, operands, draw_fn):
    """The gated delta rule's and Kimi's delta attention's chunk adjoints, walked back: each token operand's
    gradient on par with the token-by-token scan's against the float64 truth (check 1)."""
    forms = scan.derive_scan(scan.Scan(update_fn, gated_readout, operands, walked="time",
                                       state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
    assert_delta_adjoint(forms, update_fn, draw_fn, f"{update_fn.__name__} adjoint")


@pytest.mark.parametrize(("update_fn", "operands", "draw_fn"), [
    (gated_delta, GATED, gated_draw), (kimi_delta, KIMI, kimi_draw)], ids=["gated_delta", "kimi_delta"])
def test_the_delta_rules_chunk_adjoint_through_an_issuer_is_the_token_scans_gradient(update_fn, operands,
                                                                                      draw_fn):
    """The delta rules' chunk adjoints walked back through a mainloop's issuer: each product of the forward
    the issuer's, each of its adjoint's the issuer's under the product's role (`_dlhs`, `_drhs`), and each
    token operand's gradient on par with the token-by-token scan's against the float64 truth (check 1)."""
    forms = scan.derive_scan(scan.Scan(update_fn, gated_readout, operands, walked="time",
                                       state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
    roles = set()

    def issue(role, left, right):
        roles.add(role)
        return float32_matmul(left, right)

    assert_delta_adjoint(forms, update_fn, draw_fn, f"{update_fn.__name__} adjoint through an issuer",
                         issue=issue)
    assert {"correct_dlhs", "correct_drhs", "gram_dlhs", "readout_memory_drhs"} <= roles, sorted(roles)


def test_the_delta_rules_adjoint_from_the_kept_inverse_solves_nothing():
    """The gated delta rule's chunk adjoint reading each chunk's solve's inverse as its training forward kept
    it (`Prepared.inverse`, `backward`'s `inverse`): each token operand's gradient on par with the
    token-by-token scan's against the float64 truth (check 1), and no product of the solve's doubling issued
    again."""
    forms = scan.derive_scan(scan.Scan(gated_delta, gated_readout, GATED, walked="time",
                                       state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
    roles = set()

    def issue(role, left, right):
        roles.add(role)
        return float32_matmul(left, right)

    assert_delta_adjoint(forms, gated_delta, gated_draw, "gated_delta adjoint from its kept inverse",
                         issue=issue, kept=True)
    assert "correct_dlhs" in roles and not any(role.startswith("inverse_merge") for role in roles), roles


def kimi_draw_float16(seed):
    """As `kimi_draw`, each key's decay's exponent in float16 (a narrow gate's own dtype)."""
    q, k, v, g, beta = kimi_draw(seed)
    return q, k, v, g.astype(jnp.float16), beta


RESETTING = {
    "ssd": (update, readout, OPERANDS, (STATE, WIDTH), draw, ("x", "dt", "B", "C"),
            {"rate": jnp.float32(RATE), "skip": jnp.float32(SKIP)}),
    "gated_delta": (gated_delta, gated_readout, GATED, (KEY, VALUE), gated_draw, tuple(GATED), {}),
    "gated_delta_base2": (functools.partial(gated_delta, exponential=jnp.exp2), gated_readout, GATED,
                          (KEY, VALUE), gated_draw, tuple(GATED), {}),
    "kimi_delta": (kimi_delta, gated_readout, KIMI, (KEY, VALUE), kimi_draw, tuple(GATED), {}),
    "kimi_delta_float16": (kimi_delta, gated_readout, KIMI, (KEY, VALUE), kimi_draw_float16, tuple(GATED),
                           {}),
}
"""Each rule a reset walk is derived for: its update and readout, operands, state, draw, token operands and
constants. A decay in base 2, and exponents in float16, whose partials' sum of several resets passes its
largest."""


def resets_of(seed) -> jax.Array:
    """Each token a document's first with probability 1/6, and so the first token of the second chunk and two
    consecutive tokens inside it: up to several resets a chunk, its partials' sum many times `RESET_DECAY`."""
    first = np.random.default_rng(seed + 11).uniform(size=TOKENS) < 1 / 6
    first[[CHUNK, CHUNK + 9, CHUNK + 10]] = True
    return jnp.asarray(first)


def resetting(rule: str):
    """`rule`'s chunk forms with a reset operand (`first`), its token scan whose state each reset token zeroes
    before its update, and each as a function of the resets and the token operands."""
    update_fn, readout_fn, declared, shape, _, names, constants = RESETTING[rule]
    operands = {**declared, "first": ("time",)}
    forms = scan.derive_scan(scan.Scan(update_fn, readout_fn, operands, walked="time",
                                       state=(shape, jnp.float32), reset="first"), saturation=SATURATION)

    def values(first, tokens):
        return {**dict(zip(names, tokens, strict=True)), **constants, "first": first}

    def chunked(given, index):
        return {name: value[index * CHUNK:(index + 1) * CHUNK] if operands[name][:1] == ("time",) else value
                for name, value in given.items()}

    def tokens_run(first, *tokens):
        def one(h, token):
            first_t, *xs = token
            named = {**dict(zip(names, xs, strict=True)), **constants}
            h = update_fn(jnp.where(first_t, jnp.zeros((), h.dtype), h), **named)
            return h, readout_fn(h, **named)

        final, outputs = jax.lax.scan(one, jnp.zeros(shape, tokens[0].dtype), (first, *tokens))
        return outputs, final

    return forms, values, chunked, tokens_run


@pytest.mark.parametrize("rule", RESETTING)
def test_a_reset_walk_in_chunks_computes_the_token_scan_of_each_document(rule):
    """A walk whose resets (`Scan.reset`) fall on a chunk's first token, inside chunks and back to back, its
    chunks through the derived step and outputs: on par with the token scan that zeroes the state at each,
    outputs and final state, against the float64 truth (check 1). A chunk's exponents sum several resets, so
    each decay within a document is a difference of two partials past `-3e4`: a plain prefix held it to about
    `3e4 * 6e-8`, an error of 2e-3 in each decay."""
    forms, values, chunked, tokens_run = resetting(rule)

    def derived(first, *tokens):
        given = values(first, tokens)
        state, outputs = forms.init({}), []
        for index in range(TOKENS // CHUNK):
            chunk = chunked(given, index)
            prepared = forms.prepare(chunk)
            outputs.append(forms.mapped(prepared, state))
            state = forms.step(state, prepared)
        return jnp.concatenate(outputs), forms.finish(state, given)

    assert_as_exact(jax.jit(derived), jax.jit(tokens_run),
                    lambda seed: (resets_of(seed), *RESETTING[rule][4](seed)), f"{rule} reset walk", seeds=16)


def test_two_reset_walks_merged_give_the_final_state():
    """Mamba-2's tokens with resets in both halves (`resets_of`), walked in two halves, the second from a zero
    state, merged: the final state on par with the token scan's that zeroes the state at each, against the
    float64 truth (check 1), a reset in the second half leaving the first's state none of its weight."""
    forms, values, chunked, tokens_run = resetting("ssd")
    half = TOKENS // CHUNK // 2

    def merged(first, *tokens):
        given = values(first, tokens)
        states = []
        for chunks in (range(half), range(half, 2 * half)):
            state = forms.init(chunked(given, 0))
            for index in chunks:
                state = forms.step(state, forms.prepare(chunked(given, index)))
            states.append(state)
        return forms.finish(forms.merge(*states, given), given)

    assert_as_exact(jax.jit(merged), jax.jit(lambda *xs: tokens_run(*xs)[1]),
                    lambda seed: (resets_of(seed), *draw(seed)), "ssd reset merged", seeds=16)


@pytest.mark.parametrize("rule", RESETTING)
def test_a_reset_walks_chunk_adjoint_is_the_token_scans_gradient(rule):
    """The chunk adjoint of a walk with resets as the forward test's (`resets_of`), walked back: each token
    operand's gradient on par with the token scan's against the float64 truth (check 1)."""
    forms, values, chunked, tokens_run = resetting(rule)
    names = RESETTING[rule][5]
    derived = adjoint(forms, chunked, TOKENS // CHUNK)
    width = RESETTING[rule][3][1]

    def draws(seed):
        return (resets_of(seed), *RESETTING[rule][4](seed), *cotangents_of(seed, (TOKENS, width),
                                                                           RESETTING[rule][3]))

    def as_derived(first, *drawn):
        *tokens, d_out, d_final = drawn
        grads = derived(values(first, tokens), (d_out, d_final))
        return tuple(grads[name] for name in names)

    def as_tokens(first, *drawn):
        *tokens, d_out, d_final = drawn
        _, pullback = jax.vjp(lambda *xs: tokens_run(first, *xs), *tokens)
        return pullback((d_out, d_final))

    assert_as_exact(jax.jit(as_derived), jax.jit(as_tokens), draws, f"{rule} reset adjoint", seeds=16)


@pytest.mark.parametrize("rule", RESETTING)
def test_no_gradient_crosses_a_reset(rule):
    """A walk with one reset inside its second chunk, the loss reading only the outputs from it on and the
    final state: the chunk adjoint gives every operand of every token before it, and the reset token's own
    decay where it is an operand of its own (the delta rules' `g`; Mamba-2's `dt` writes its token too), a
    gradient of exactly zero (every decay spanning it `exp(RESET_DECAY)`, 0 in fp32)."""
    forms, values, chunked, _ = resetting(rule)
    names, (rows, width) = RESETTING[rule][5], RESETTING[rule][3]
    reset = CHUNK + 9
    first = jnp.zeros(TOKENS, bool).at[reset].set(True)
    d_out, d_final = cotangents_of(0, (TOKENS, width), (rows, width))
    d_out = d_out.at[:reset].set(0.0)
    run = jax.jit(adjoint(forms, chunked, TOKENS // CHUNK))
    grads = run(values(first, RESETTING[rule][4](0)), (d_out, d_final))
    for name in names:
        assert not np.any(np.asarray(grads[name][:reset])), f"{rule}: {name}'s gradient crosses the reset"
    if "g" in grads:
        assert not np.any(np.asarray(grads["g"][reset])), f"{rule}: the reset token's decay has a gradient"


def _cast(h, g, b, v, q):
    return jnp.exp(g) * h + b.astype(jnp.int32).astype(h.dtype)[:, None] * v[None, :]


def _huge(h, g, b, v, q):
    return jnp.exp(g) * h + b[:, None] * (v * jnp.float64(1e100))[None, :]


@pytest.mark.parametrize(("update", "dtype"), [(_cast, jnp.float32), (_huge, jnp.float64)],
                         ids=["integer_conversion", "float64_literal"])
def test_a_derived_chunk_computes_what_its_token_update_states(update, dtype):
    """A chunk's derived form computes its declared token scan exactly where the update's terms hold more
    than their algebra's: a key truncated to an integer and back (1.9 counts 1), a float64 literal past
    float32's range (1e100): the conversion and the literal's dtype kept, not the identity and a float32's
    infinity."""
    def readout(h, g, b, v, q):
        return q @ h

    with jax.enable_x64(new_val=True):
        operands = {"g": ("time",), "b": ("time", "keys"), "v": ("time", "values"), "q": ("time", "keys")}
        made = scan.derive_scan(scan.Scan(update, readout, operands, walked="time", state=((2, 3), dtype)),
                                saturation=SATURATION)
        chunk = {"g": jnp.zeros((2,), dtype), "b": jnp.full((2, 2), 1.9, dtype), "v": jnp.ones((2, 3), dtype),
                 "q": jnp.ones((2, 2), dtype)}
        prepared = made.prepare(chunk)
        outputs, final = made.mapped(prepared, made.init({})), made.step(made.init({}), prepared)[1]

        def token(h, values):
            h = update(h, **values)
            return h, readout(h, **values)

        declared_final, declared_outputs = jax.lax.scan(token, jnp.zeros((2, 3), dtype), chunk)
        # Every value here is exact (counts, doublings of one rounded 1.9e100), so the two agree exactly.
        np.testing.assert_array_equal(np.asarray(final), np.asarray(declared_final))
        np.testing.assert_array_equal(np.asarray(outputs), np.asarray(declared_outputs))
