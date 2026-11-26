"""A recurrence's chunk forms derived from its token update (`dew.algebra.scan`, docs/design/algebra.md §6.9):
Mamba-2's state-space dual (a scalar decay, a rank-one input, a contraction readout and a skip), the gated
delta rule (a decayed, corrected state: the compact WY form) and Kimi's delta attention (its decay each key's
own), each stated token by token, walked in chunks
through its derived step and outputs (and Mamba-2's two walks merged), each held to the float64 truth
(check 1) against the token-by-token scan, with packed documents' resets too (`Scan.reset`), across which no
gradient flows; and updates no chunk form here holds, refused."""

import dataclasses
import functools
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.algebra import SATURATION, linear_scan, program, scan
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


def tokenwise(x, dt, B, C, reads: str = "after", group: int = 1):
    """The recurrence token by token (`lax.scan`), its outputs and final state, each token reading the state
    after it or entering it (`reads`, `Scan.reads`: RWKV-6's, before its decay and its write), or after its
    decay apart from every write of its `group` (Mamba-3's ranks: the state its group entered, decayed by
    each of the group's tokens up to it)."""
    def one(carry, token):
        h, held, index = carry
        x_t, dt_t, B_t, C_t = token
        entering = h
        after = update(h, x_t, dt_t, B_t, C_t, RATE, SKIP)
        held = jnp.exp(dt_t * RATE) * jnp.where(index % group == 0, h, held)
        read = {"before": entering, "after": after, "apart": held}[reads]
        return (after, held, index + 1), readout(read, x_t, dt_t, B_t, C_t, RATE, SKIP)

    zero = jnp.zeros((STATE, WIDTH), x.dtype)
    (final, _, _), outputs = jax.lax.scan(one, (zero, zero, 0), (x, dt, B, C))
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


def stepped_at(tokens: slice, step: float, doc: str):
    """`draw`'s inputs with the steps of `tokens` set to `step`, a draw `doc` states."""
    def drawn(seed):
        x, dt, B, C = draw(seed)
        return x, dt.at[tokens].set(step), B, C

    drawn.__doc__ = doc
    return drawn


vanishing_draw = stepped_at(slice(3, None, 8), 2e3, """`draw`'s inputs, every eighth token's step so long its
decay vanishes in float32 (`e^-1400`), with no reset: the state it reads entering it is whole, the one it
leaves its own write alone.""")


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
        state = forms.step(state, prepared)[0]
    return outputs, state


ranked_draw = stepped_at(slice(1, None, 2), 0.0, """`draw`'s inputs as Mamba-3's ranks of two: each token's
two ranks consecutive, its step (its decay) on its first rank alone, the second's zero, so the second reads
the state its token's first decayed.""")


def ssd_walked(forms, axes: dict = OPERANDS):
    """`forms`' chunk outputs and steps over every chunk in order from a zero state, finished: a function of
    the SSD's inputs."""
    def derived(x, dt, B, C):
        values = operands(x, dt, B, C)
        outputs, state = walk(forms, values, range(TOKENS // CHUNK), forms.init({}), axes)
        return jnp.concatenate(outputs), forms.finish(state, values)

    return derived


READING_DRAWS = [("after", 1, draw), ("before", 1, draw), ("before", 1, vanishing_draw), ("apart", 1, draw),
                 ("apart", 2, ranked_draw), ("apart", 4, draw), ("apart", 2, vanishing_draw)]
"""The SSD's walks each token reading the state after it or entering it (`Scan.reads`, RWKV-6's), over
`draw`'s inputs, and reading before over `vanishing_draw`'s, a decay vanishing at a token with no reset; and
each reading after its decay apart from its group's writes, a group of one, Mamba-3's ranks of two, four
tokens, and two over vanishing decays."""


@pytest.mark.parametrize(("reads", "group", "draw_fn"), READING_DRAWS,
                         ids=["after", "before", "before-vanishing", "apart", "apart-ranks", "apart-4",
                              "apart-vanishing"])
def test_the_derived_chunks_compute_the_token_scan(reads, group, draw_fn):
    """The derived chunk outputs and step over every chunk in order, each token reading the state after it or
    entering it, or after its decay apart from every write of its group (`Scan.reads`, `Scan.group`): on par
    with the token-by-token scan reading alike, outputs and final state, against the float64 truth (check 1);
    a decay vanishing at a token erases nothing that token reads."""
    forms = scan.derive_scan(dataclasses.replace(DECLARED, reads=reads, group=group), saturation=SATURATION)
    derived = ssd_walked(forms)
    label = (f"ssd chunks{READING[reads]}{f' a group of {group}' if reads == 'apart' else ''}"
             f"{', ranks' if draw_fn is ranked_draw else ''}"
             f"{', vanishing decays' if draw_fn is vanishing_draw else ''}")
    scanned = functools.partial(tokenwise, reads=reads, group=group)
    assert_as_exact(jax.jit(derived), jax.jit(scanned), draw_fn, label, seeds=16)


@pytest.mark.parametrize("target", ["sm86", "v5e"])
def test_an_apart_readouts_replay_and_token_loop_read_apart_from_its_group(target):
    """A readout apart from its group's writes (Mamba-3's ranks of two) replayed token by token
    (`Derived.replay`, which the walk's derivative reads) and its chunk derived where the token loop is
    extracted (a v5e's step latency unstated) or the dual form (an A10's): each on par with the token scan,
    outputs and state (check 1)."""
    declared = dataclasses.replace(DECLARED, reads="apart", group=2)
    # The target held through each check, where the forms are prepared and traced.
    with program.passes(target=target):
        forms = scan.derive_scan(declared, saturation=SATURATION)
        derived = ssd_walked(forms)

        def replayed(x, dt, B, C):
            return forms.replay(operands(x, dt, B, C), jnp.zeros((STATE, WIDTH), x.dtype))

        scanned = functools.partial(tokenwise, reads="apart", group=2)
        for form, name in ((derived, "derived"), (replayed, "replayed")):
            assert_as_exact(jax.jit(form), jax.jit(scanned), ranked_draw,
                            f"ssd apart ranks, {name} on {target}", seeds=16)


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
            state = forms.step(state, prepared)[0]
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


def test_a_scans_derivations_stop_at_limits_of_their_inputs_size():
    """A scan's token and chunk e-graphs saturate within limits derived from their size
    (`program.Saturation.sized`), as a region's does: under limits of one node a row and one iteration a
    level, which stop the token programs' saturation before it reaches the affine form, the declaration that
    derives at the engine's limits is refused, and each derivation's saturation is reported with the
    kernels' rewrites, none of the engine's as limited. `run_within` runs a rule set to its fixpoint where
    its limits allow, and stops an iteration short of it, or before its first at a graph past its nodes,
    saying so. Both saturations ran to their fixpoints whatever limits the derivation was given, which
    reached its carry's split alone."""
    from egglog import EGraph, rewrite, ruleset

    from dew.algebra import program
    from dew.algebra.linear_scan import Chunk
    from dew.nn.kernels.templates import report

    chunk = chunk_of(operands(*draw(0)), 0)
    # Limits one node an equation above the engine's, so the chunk's form, kept by its limits, derives here.
    roomy = program.Saturation(growth=SATURATION.growth + 1, depth=SATURATION.depth)
    with report.refusals():
        scan.derive_scan(DECLARED, saturation=roomy).prepare(chunk)
        derived = [made for made in report.rewrites()
                   if any(name.startswith("a scan's") for name in made.admitted)]
        assert {made.admitted for made in derived} == {("a scan's token",), ("a scan's chunk",)}
        assert not any(made.limited for made in derived)
    with report.refusals(), pytest.raises(scan.Refused):
        scan.derive_scan(DECLARED, saturation=program.Saturation(growth=1, depth=1)).prepare(chunk)

    def chained() -> EGraph:
        graph = EGraph()
        graph.register(Chunk.token("a"))
        return graph

    # Each iteration reaches one name further along the chain.
    names = "abcdef"
    chain = ruleset(*(rewrite(Chunk.token(left)).to(Chunk.token(right))
                      for left, right in itertools.pairwise(names)))
    whole = program.run_within(chained(), chain, (10**6, 10**6), "a chain")
    assert not whole.limited and whole.iterations >= len(names) - 1
    short = program.run_within(chained(), chain, (10**6, whole.iterations - 1), "a chain")
    assert (short.iterations, short.limited) == (whole.iterations - 1, True)
    unrun = program.run_within(chained(), chain, (0, 10**6), "a chain")
    assert (unrun.iterations, unrun.limited) == (0, True)


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


def gated_tokenwise(update_fn, reads: str = "after", readout_fn=gated_readout):
    """A gated rule `update_fn` token by token, its outputs and final state, each token reading the state
    (`readout_fn`) after its update or, `reads` "apart", after its decay before its own write (`Scan.reads`
    at a group of one: `exp(g) h`, each key's row by its own decay where `g` is a key's)."""
    def run(q, k, v, g, beta):
        def one(h, token):
            decayed = jnp.reshape(jnp.exp(token[3]), (-1, 1)) * h
            h = update_fn(h, *token)
            return h, readout_fn(h if reads == "after" else decayed, *token)

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


def delta_before(h, q, k, v, g, beta):
    """The delta rule correcting the state a token enters, before its decay (RWKV7's erase read):
    `h = D h + beta k (v - k^T h)^T`, `D = exp(g)` a scalar's or each key's."""
    del q
    return jnp.reshape(jnp.exp(g), (-1, 1)) * h + beta * jnp.outer(k, v - k @ h)


KIMI = {**GATED, "g": ("time", "key")}


def kimi_draw(seed):
    """As `gated_draw`, each key's decay its own, down to 1e-3 a token."""
    q, k, v, _, beta = gated_draw(seed)
    rng = np.random.default_rng(seed + 1)
    return q, k, v, jnp.asarray(-(10.0 ** rng.uniform(-3, 0, (TOKENS, KEY))), jnp.float32), beta


def rwkv_erase(h, q, k, v, g, beta, a, c):
    """RWKV7's token: the state decayed, erased along its own vector `a` by its read `c^T h` of the state the
    token enters, and written: `h = D h + a (c^T h)^T + beta k v^T`, `D = exp(g)` a scalar's or each key's."""
    del q
    return jnp.reshape(jnp.exp(g), (-1, 1)) * h + jnp.outer(a, c @ h) + beta * jnp.outer(k, v)


def rwkv_readout(h, q, k, v, g, beta, a, c):
    del k, v, g, beta, a, c
    return (q @ h) * KEY ** -0.5


RWKV = {**GATED, "a": ("time", "key"), "c": ("time", "key")}
"""RWKV7's operands: `GATED`'s and its erase's vector `a` and read `c`."""

RWKV_KEYED = {**RWKV, "g": ("time", "key")}


def rwkv_draw(seed, per_key: bool = False):
    """`gated_draw`'s (`kimi_draw`'s where `per_key`) and an erase of unit vectors' own: `c = -kk`, `a = kk
    alpha`, `alpha` up to 1/2, each token's erase a contraction."""
    q, k, v, g, beta = (kimi_draw if per_key else gated_draw)(seed)
    rng = np.random.default_rng([seed, 9670])
    kk = rng.standard_normal((TOKENS, KEY))
    kk /= np.linalg.norm(kk, axis=-1, keepdims=True)
    alpha = rng.uniform(0.0, 0.5, (TOKENS, 1))
    return q, k, v, g, beta, jnp.asarray(kk * alpha, jnp.float32), jnp.asarray(-kk, jnp.float32)


VANISHED = -200.0
"""A log decay that vanishes alone (`exp(-200)`, zero in float32), no document's start."""

SATURATED = scan.RESET_DECAY
"""A gate saturated at a document's start's own exponent (`scan.RESET_DECAY`), no document's start: RWKV7's
`-exp(x)` past it at `x` of about 9.2."""


def rwkv_forms(operands: dict, **declared):
    """RWKV7's token's derived forms over `operands` (`RWKV`'s or `RWKV_KEYED`'s), `declared` the rest of its
    declaration (`scan.Scan`'s)."""
    return scan.derive_scan(scan.Scan(rwkv_erase, rwkv_readout, operands, walked="time",
                                      state=((KEY, VALUE), jnp.float32), **declared), saturation=SATURATION)


def rwkv_tokenwise(q, k, v, g, beta, a, c):
    """RWKV7's token scan from a zero state: its outputs and final state."""
    return scanned(rwkv_erase, rwkv_readout)(q, k, v, g, beta, a, c, jnp.zeros((KEY, VALUE), q.dtype))


def scanned(update_fn, readout_fn):
    """`update_fn`'s token scan, each token reading the state after it (`readout_fn`), from the state given
    after its operands: its outputs and final state."""
    def run(*xs):
        *tokens, h0 = xs

        def one(h, token):
            h = update_fn(h, *token)
            return h, readout_fn(h, *token)

        final, outputs = jax.lax.scan(one, h0, tuple(tokens))
        return outputs, final

    return run


def walked_of(forms, operands: dict, entering: bool = False):
    """`forms`' chunks walked over `operands` from a zero state, or from the state given after them where
    `entering`: outputs and final state."""
    def walked(*xs):
        values = dict(zip(operands, xs[:len(operands)], strict=True))
        state = forms.init({})
        state = (state[0], xs[-1]) if entering else state
        outputs, state = walk(forms, values, range(TOKENS // CHUNK), state, operands)
        return jnp.concatenate(outputs), forms.finish(state, values)

    return walked


def replayed_of(forms, operands: dict):
    """`forms`' token walk (`Derived.replay`) chunk by chunk over `operands` from the state given after them,
    as the chunked op's walk takes its tangent: outputs and final state."""
    def replayed(*xs):
        values, h = dict(zip(operands, xs[:-1], strict=True)), xs[-1]
        outputs = []
        for index in range(TOKENS // CHUNK):
            out, h = forms.replay(chunk_of(values, index, operands), h)
            outputs.append(out)
        return jnp.concatenate(outputs), h

    return replayed


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
                state = forms.step(state, prepared, issue)[0]
            states.append(state)
        return forms.finish(forms.merge(*states, values, issue), values)

    assert_as_exact(jax.jit(merged), jax.jit(lambda *xs: gated_tokenwise(update_fn)(*xs)[1]), draw_fn,
                    f"{update_fn.__name__} merged through an issuer", seeds=16)
    assert {"transfer_compose", "transfer_apply"} <= roles, roles


def long_draw(seed):
    """`draw`'s inputs over two of its draws: 256 tokens, one chunk of them."""
    return tuple(jnp.concatenate(pair) for pair in zip(draw(2 * seed), draw(2 * seed + 1), strict=True))


@pytest.mark.parametrize("target", ["sm86", "v5e"])
def test_a_long_chunks_token_loop_is_split_by_its_construction(target):
    """Mamba-2's chunk of 256 tokens derived priced for a target whose step latency is unstated (an A10, a
    v5e), so the token loop's steps are priced by no time and it is extracted: its split is the loop's own,
    never its 256 steps unrolled into the split's e-graph (row 7201: 6,917 equations, 20 minutes, then
    egglog's extraction past Python's stack), its outputs and final state on par with the token scan
    (check 1)."""
    with program.passes(target=target):
        forms = scan.derive_scan(DECLARED, saturation=SATURATION)

        def derived(x, dt, B, C):
            values = operands(x, dt, B, C)
            prepared = forms.prepare(values)
            state = forms.init({})
            return forms.mapped(prepared, state), forms.finish(forms.step(state, prepared)[0], values)

        assert_as_exact(jax.jit(derived), jax.jit(tokenwise), long_draw, f"ssd's 256 tokens on {target}",
                        seeds=8)


def test_a_token_loop_carries_its_state_in_its_operands_wider_type():
    """A float32 state walked over float64 operands (x64's) by a token loop (a v5e's, `_looped_split`): its
    outputs and the state it leaves are float64, the token scan's types, its carry widened once."""
    with jax.enable_x64(new_val=True), program.passes(target="v5e"):
        forms = scan.derive_scan(DECLARED, saturation=SATURATION)
        wide = tuple(value.astype(jnp.float64) for value in long_draw(0))
        values = operands(*wide)
        prepared = forms.prepare(values)
        outputs, (_, state) = forms.mapped(prepared, forms.init({})), forms.step(forms.init({}), prepared)[0]
        expected = tokenwise(*wide)
    assert (outputs.dtype, state.dtype) == (expected[0].dtype, expected[1].dtype) == (jnp.float64,) * 2


@pytest.mark.parametrize("target", ["sm86", "sm90"])
@pytest.mark.parametrize(("update_fn", "operands", "draw_fn"), [
    (gated_delta, GATED, gated_draw), (delta_before, GATED, gated_draw), (delta_before, KIMI, kimi_draw)],
    ids=["gated_delta", "before its decay, a head's", "before its decay, a key's"])
def test_a_chunk_form_any_target_extracts_computes_the_token_scan(update_fn, operands, draw_fn, target):
    """The gated delta rule's chunks, and the delta rule's correcting the state a token enters
    (`delta_before`, a decay a head's or each key's: the WY system's decays a token back), derived under an
    A10 (its step's latency unstated, so the token loop is priced at its work alone and extracted) and an
    H100 (WY's): on par with the token scan, outputs and final state, against the float64 truth (check
    1)."""
    with generation.assumed(target):
        forms = scan.derive_scan(scan.Scan(update_fn, gated_readout, operands, walked="time",
                                           state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
        assert_as_exact(jax.jit(gated_walk(forms, operands)), jax.jit(gated_tokenwise(update_fn)), draw_fn,
                        f"{update_fn.__name__} on {target}", seeds=8)


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
    expected = forms.mapped(plain, state), forms.step(state, plain)[0]
    issued = forms.prepare(chunk, issue)
    made = forms.mapped(issued, state, None, issue), forms.step(state, issued, issue)[0]
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
def test_the_delta_rules_carry_from_a_ranged_state_computes_its_token_scan(update_fn, operands, draw_fn):
    """The delta rule's split (`carry.split`), its keys, write gates, log decays and values ranged (unit keys,
    so tokens' corrections overlap by at most their gates, under 1; decays at most 1; values under the draws'
    8), its entering state ranged too (`Scan.ranges`' "state", where the solve need not distribute over
    `W h`, §6.3) or not: a chunk's carry (`step` over its prepared values) from a drawn state within that
    range on par with the token-by-token scan over the chunk, against the float64 truth (check 1)."""
    names = ("q", "k", "v", "g", "beta")
    ranges = {"k": 1.0, "beta": 1.0, "overlap": 1.0, "g": (-np.inf, 0.0), "v": 8.0}

    def draw(seed):
        entering = np.random.default_rng([374, seed]).uniform(-64.0, 64.0, (KEY, VALUE)).astype(np.float32)
        return (*(value[:CHUNK] for value in draw_fn(seed)), entering)

    def tokens(q, k, v, g, beta, h):
        for token in zip(q, k, v, g, beta, strict=True):
            h = update_fn(h, *token)
        return h

    for declared in ({**ranges, "state": 64.0}, ranges):
        forms = scan.derive_scan(scan.Scan(update_fn, gated_readout, operands, walked="time",
                                           state=((KEY, VALUE), jnp.float32), ranges=declared),
                                 saturation=SATURATION)

        def carried(*values, forms=forms):
            prepared = forms.prepare(dict(zip(names, values[:-1], strict=True)))
            return forms.step((jnp.zeros((KEY,), jnp.float32), values[-1]), prepared)[0][1]

        label = f"{update_fn.__name__} carry, {'a ranged' if 'state' in declared else 'an unranged'} state"
        assert_as_exact(jax.jit(carried), jax.jit(tokens), draw, label, seeds=16)


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
    outputs, (_, leaving) = forms.mapped(prepared, entering), forms.step(entering, prepared)[0]
    assert np.all(np.isfinite(np.asarray(outputs))) and np.all(np.isfinite(np.asarray(leaving)))


def test_a_chunks_adjoint_meets_its_decay_once():
    """The gated delta rule's one-token chunk, its key and write gate unit on one key, its query and value
    zero, its decay e^-1, entering a state of 2^120 on that key alone, with a cotangent of 2^20 on the state
    it leaves: the token's correction cancels its write, so the state's cotangent through the decay is zero
    on the key the state holds and so is the decay's gradient, and the derived chunk's adjoint
    (`Derived.backward`, the token update replayed) keeps the entering state's cotangent and the decay's
    gradient finite, the latter exactly zero (row 1966's head review). The key's and write gate's gradients
    exceed float32's range at this state and cotangent."""
    chunk = {"q": jnp.zeros((1, KEY)), "k": jnp.zeros((1, KEY)).at[0, 0].set(1.0),
             "v": jnp.zeros((1, VALUE)), "g": jnp.full((1,), -1.0), "beta": jnp.ones((1,))}
    forms = scan.derive_scan(scan.Scan(gated_delta, gated_readout, GATED, walked="time",
                                       state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
    entering = jnp.zeros((KEY, VALUE)).at[0].set(2.0 ** 120)
    d_state, d_operands = forms.backward(chunk, (None, entering), jnp.full((KEY, VALUE), 2.0 ** 20),
                                         jnp.zeros((1, VALUE)))
    assert np.all(np.isfinite(np.asarray(d_state)))
    assert np.all(np.isfinite(np.asarray(d_operands["g"])))
    np.testing.assert_array_equal(d_operands["g"], jnp.zeros_like(chunk["g"]))


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


def _scaled_first_read(scale, query, memory):
    """A one-token scalar recurrence's read of the state entering it (`reads="before"`), `scale (q h)`, its
    reader's literal `scale` (`linear_scan.Parts.reader_scale` where it shrinks), from `memory`."""
    def read(h, q, b, v, g):
        del b, v, g
        return scale * (q @ h)

    forms = scan.derive_scan(scan.Scan(scalar_step, read, SCALAR_SSD, walked="time",
                                       state=(memory.shape, memory.dtype), reads="before"),
                             saturation=SATURATION)
    chunk = {"q": query, "b": jnp.zeros((1, memory.shape[0]), memory.dtype),
             "v": jnp.zeros((1, memory.shape[1]), memory.dtype), "g": jnp.zeros((1,), memory.dtype)}
    return forms.replay(chunk, memory)[0]


def test_a_magnifying_reader_keeps_a_representable_query_cotangent():
    """A reader scale of 2^64 over a memory of 2^64 and a cotangent of 2^-64: the query's cotangent 2^64,
    where factoring the scale out formed scale times the memory, 2^128, before the cotangent (row 8428's
    head review, P2): a magnifying scale stays in the reader."""
    query = jnp.full((1, 1), 2.0 ** -64, jnp.float32)
    memory = jnp.full((1, 1), 2.0 ** 64, jnp.float32)
    cotangent = jnp.full((1, 1), 2.0 ** -64, jnp.float32)
    gradient = jax.jit(jax.grad(lambda q: jnp.sum(
        _scaled_first_read(np.float32(2.0 ** 64), q, memory) * cotangent)))(query)
    np.testing.assert_array_equal(np.asarray(gradient), np.full((1, 1), 2.0 ** 64, np.float32))


@pytest.mark.parametrize("scale", [np.float64(1e100), np.float64(0.1)],
                         ids=["magnifying", "shrinking"])
def test_a_reader_literal_preserves_its_float64_promotion(scale):
    """A float64 literal scale over a float32 query reads in float64, `2 scale` exactly, where a factored
    scale read as a Python float multiplied the query in float32 first, 2e100 to inf (row 8428's head review,
    P2)."""
    with jax.enable_x64(new_val=True):
        query = jnp.full((1, 1), 2.0, jnp.float32)
        memory = jnp.ones((1, 1), jnp.float64)
        output = jax.jit(lambda q: _scaled_first_read(scale, q, memory))(query)
        # Multiplication by two is exact for both literals, with no nontrivial sum.
        np.testing.assert_array_equal(np.asarray(output), np.full((1, 1), 2.0 * scale, np.float64))


def test_equal_reader_scales_keep_their_own_dtypes_and_query_cotangents():
    """Scales of 0.5 in float32 and in float64, equal in value: each read in its own dtype and its query's
    cotangent 0.5, the scaled contraction cached by the scale's dtype as well as its value."""
    with jax.enable_x64(new_val=True):
        query = jnp.full((1, 1), 2.0, jnp.float32)
        memory = jnp.ones((1, 1), jnp.float32)
        for dtype in (np.float32, np.float64):
            read = functools.partial(_scaled_first_read, dtype(0.5), memory=memory)
            output = jax.jit(read)(query)
            assert output.dtype == dtype
            np.testing.assert_array_equal(np.asarray(output), np.ones((1, 1), dtype))
            gradient = jax.jit(jax.grad(lambda q, read=read: jnp.sum(read(q))))(query)
            np.testing.assert_array_equal(np.asarray(gradient), np.full((1, 1), 0.5, np.float32))


@pytest.mark.parametrize("reads", ["before", "after"])
@pytest.mark.parametrize("large", ["reader", "state"])
def test_a_prepared_chunk_keeps_a_subnormal_read_scale_until_the_state_contraction(reads, large):
    def read(h, q, b, v, g):
        return np.float32(2.0 ** -128) * (q @ h)

    declaration = scan.Scan(scalar_step, read, SCALAR_SSD, walked="time", state=((1, 1), jnp.float32),
                            reads=reads)
    forms = scan.derive_scan(declaration, saturation=SATURATION)
    giant = np.float32(2.0 ** 127)
    reader = np.full((2, 1), giant if large == "reader" else 1, np.float32)
    memory = np.full((1, 1), giant if large == "state" else 1, np.float32)
    cotangent = np.full_like(reader, 2.0 ** 126)

    def run(q, h):
        chunk = {"q": q, "b": jnp.zeros_like(q), "v": jnp.zeros_like(q), "g": jnp.zeros(2, q.dtype)}
        state = (jnp.asarray(0, q.dtype), h)
        prepared = forms.prepare(chunk)
        _, held = forms.step(state, prepared)
        mapped = forms.mapped(prepared, state, held)
        replayed, _ = forms.replay(chunk, h)
        return mapped, replayed

    @jax.jit
    def orders(q, h, ct):
        outputs, pull = jax.vjp(run, q, h)
        zero = jnp.zeros_like(ct)
        return outputs, (pull((ct, zero)), pull((zero, ct)))

    outputs, pullbacks = orders(reader, memory, cotangent)
    for output, (dq, dh) in zip(outputs, pullbacks, strict=True):
        np.testing.assert_array_equal(output, np.full_like(reader, 0.5))
        np.testing.assert_array_equal(dq, np.broadcast_to(memory * 0.25, reader.shape))
        np.testing.assert_array_equal(dh, reader[:1] * 0.5)


def test_a_tiny_prepared_reader_leaves_a_steep_decays_factors_on_the_state():
    def step(h, q, b, v, g):
        return jnp.exp2(g) * h + b[:, None] * v[None, :]

    def read(h, q, b, v, g):
        return np.float32(2.0 ** -128) * (q @ h)

    declaration = scan.Scan(step, read, SCALAR_SSD, walked="time", state=((1, 1), jnp.float32))
    forms = scan.derive_scan(declaration, saturation=SATURATION)

    @jax.jit
    def run(q, h, g):
        chunk = {"q": q, "b": jnp.zeros_like(q), "v": jnp.zeros_like(q), "g": g}
        state = (jnp.zeros((), q.dtype), h)
        prepared = forms.prepare(chunk)
        leaving, kept = forms.step(state, prepared)
        return forms.mapped(prepared, state, kept), leaving[1]

    giant = np.float32(2.0 ** 127)
    rows = np.full((2, 1), giant, np.float32)
    memory = np.full((1, 1), giant, np.float32)
    output, leaving = run(rows, memory, np.asarray([-149.5, 0], np.float32))
    np.testing.assert_array_equal(output, np.full((2, 1), np.float32(2.0 ** -23.5)))
    np.testing.assert_array_equal(leaving, np.full((1, 1), np.float32(2.0 ** -22.5)))


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
        return jnp.sum(forms.step(forms.init({}), forms.prepare(ones_chunk(g)))[0][1])

    g = jnp.asarray(decays, jnp.float32)
    value, gradient = jax.value_and_grad(final)(g)
    _, tangent = jax.jvp(jax.grad(final), (g,), (jnp.ones_like(g),))
    start = forms.init(ones_chunk(g))
    merged = forms.merge(start, forms.step(start, forms.prepare(ones_chunk(g)))[0], None)
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
        state, _ = jax.lax.scan(lambda carried, chunk: (forms.step(carried, chunk)[0], None), forms.init({}),
                                prepared)
        final = forms.finish(state, {})
        assert final.dtype == dtype and np.isfinite(np.asarray(final, np.float64)).all()
        assert state[0].dtype == jnp.dtype(jnp.float64 if dtype == jnp.float64 else jnp.float32)


def adjoint(forms, chunk_of_values, count: int):
    """A loss's gradient through the derived forms: the chunks walked forward from a zero state, or from
    `values`' "h0" where given, whose cotangent it returns too (each entering state kept), then back through
    each chunk's `backward`, the cotangents of the outputs and the final state the loss's (`COTANGENT`, a
    fixed draw)."""
    def run(values, cotangents):
        values = dict(values)
        entering = values.pop("h0", None)
        d_outputs, d_final = cotangents
        init = forms.init({})
        states = [init if entering is None else (init[0], entering)]
        for index in range(count):
            states.append(forms.step(states[-1], forms.prepare(chunk_of_values(values, index)))[0])
        d_state, grads = d_final, []
        for index in reversed(range(count)):
            chunk = chunk_of_values(values, index)
            d_state, d_chunk = forms.backward(chunk, states[index], d_state,
                                              d_outputs[index * CHUNK:(index + 1) * CHUNK])
            grads.append(d_chunk)
        # A token operand's gradient each chunk's; a constant's (a rate, a skip) summed over them.
        found = {name: jnp.concatenate([grad[name] for grad in reversed(grads)]) if grads[0][name].ndim
                 else sum(grad[name] for grad in grads) for name in grads[0]}
        return found if entering is None else {**found, "h0": d_state}

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


@pytest.mark.parametrize(("reads", "group", "draw_fn"),
                         [("after", 1, draw), ("after", 1, deep_draw), *READING_DRAWS[1:]],
                         ids=["after", "after-deep", "before", "before-vanishing", "apart", "apart-ranks",
                              "apart-4", "apart-vanishing"])
def test_the_ssd_chunks_adjoint_is_the_token_scans_gradient(reads, group, draw_fn):
    """Mamba-2's chunk adjoint (`backward`), walked back over the chunks, each token reading the state after
    it or entering it, or after its decay apart from its group's writes (`Scan.reads`): each token operand's
    gradient on par with the token-by-token scan's reading alike against the float64 truth (check 1), its
    steps' too where a chunk's decays reach e^-45 or vanish at a token."""
    forms = scan.derive_scan(dataclasses.replace(DECLARED, reads=reads, group=group), saturation=SATURATION)
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
        scanned = functools.partial(tokenwise, reads=reads, group=group)
        grads = gradient_of(scanned, names)({"x": x, "dt": dt, "B": B, "C": C}, (d_out, d_final))
        return tuple(grads[name] for name in names)

    label = f"ssd adjoint{READING[reads]}"
    assert_as_exact(jax.jit(as_derived), jax.jit(as_tokens), draw_with_cotangents, label, seeds=16)


def assert_adjoint(forms, tokenwise_fn, names: tuple[str, ...], chunked, draw_fn, shapes: tuple,
                   label: str, seeds: int = 16) -> None:
    """`forms`' chunk adjoint, walked back from each chunk's entering state (`chunked(values, index)` a
    chunk's operands): each of the token operands `names`' gradient on par with the token-by-token scan's
    (`tokenwise_fn` of them in order) against the float64 truth (check 1), over `draw_fn`'s draws and a fixed
    cotangent's of the outputs' and the final state's `shapes`."""
    derived = adjoint(forms, chunked, TOKENS // CHUNK)

    def draw_with_cotangents(seed):
        return (*draw_fn(seed), *cotangents_of(seed, *shapes))

    def as_derived(*drawn):
        grads = derived(dict(zip(names, drawn[:len(names)], strict=True)), drawn[len(names):])
        return tuple(grads[name] for name in names)

    def as_tokens(*drawn):
        values = dict(zip(names, drawn[:len(names)], strict=True))
        grads = gradient_of(tokenwise_fn, names)(values, drawn[len(names):])
        return tuple(grads[name] for name in names)

    assert_as_exact(jax.jit(as_derived), jax.jit(as_tokens), draw_with_cotangents, label, seeds=seeds)


def assert_delta_adjoint(forms, update_fn, draw_fn, label: str, seeds: int = 16,
                         reads: str = "after") -> None:
    """`forms`' chunk adjoint of the delta rule `update_fn`, on par with its token scan's, each token reading
    as `reads` (`gated_tokenwise`, `assert_adjoint`)."""
    assert_adjoint(forms, gated_tokenwise(update_fn, reads), ("q", "k", "v", "g", "beta"),
                   functools.partial(chunk_of, axes=GATED), draw_fn, ((TOKENS, VALUE), (KEY, VALUE)), label,
                   seeds)


@pytest.mark.parametrize(("update_fn", "operands", "draw_fn"), [
    (gated_delta, GATED, gated_draw), (kimi_delta, KIMI, kimi_draw)], ids=["gated_delta", "kimi_delta"])
def test_the_delta_rules_chunk_adjoint_is_the_token_scans_gradient(update_fn, operands, draw_fn):
    """The gated delta rule's and Kimi's delta attention's chunk adjoints, walked back: each token operand's
    gradient on par with the token-by-token scan's against the float64 truth (check 1)."""
    forms = scan.derive_scan(scan.Scan(update_fn, gated_readout, operands, walked="time",
                                       state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
    assert_delta_adjoint(forms, update_fn, draw_fn, f"{update_fn.__name__} adjoint")


def update_parts(update_fn, operands, readout_fn=gated_readout) -> linear_scan.Parts:
    """`update_fn`'s token as the rules read it (`linear_scan.parts`, its readout `readout_fn`'s) over a chunk
    of `operands`' shapes."""
    declaration = scan.Scan(update_fn, readout_fn, operands, walked="time", state=((KEY, VALUE), jnp.float32))
    widths = {"key": KEY, "value": VALUE}
    shapes = {name: jax.ShapeDtypeStruct((CHUNK, *(widths[axis] for axis in axes[1:])), jnp.float32)
              for name, axes in operands.items()}
    names = tuple(declaration.operands)
    return linear_scan.parts(scan._traced(declaration, update_fn, names, shapes),
                             scan._traced(declaration, readout_fn, names, shapes), SATURATION)


@pytest.mark.parametrize(("update_fn", "operands", "before"), [
    (delta_before, GATED, True), (delta_before, KIMI, True), (gated_delta_written_as_error, GATED, False),
    (kimi_delta_written_as_error, KIMI, False)],
    ids=["before, a head's decay", "before, a key's", "after, a head's", "after, a key's"])
def test_a_corrections_read_before_or_after_its_decay_reaches_its_own_form(update_fn, operands, before):
    """A delta rule's correction reading the state a token enters (`delta_before`) reaches the pre-decay
    form (`linear_scan.Affine.corrected_before`), and one reading its decayed state the after-decay form
    (`Affine.corrected`), neither the other's: a decay a head's commuting into the read, the after-decay
    form priced first."""
    assert update_parts(update_fn, operands).before == before


PRE_DECAY = {
    "rwkv7": (rwkv_erase, rwkv_readout, RWKV, RWKV_KEYED, rwkv_draw),
    "delta_before": (delta_before, gated_readout, GATED, KIMI,
                     lambda seed, per_key: (kimi_draw if per_key else gated_draw)(seed)),
}
"""The forms correcting the state a token enters: RWKV7's erase along its own vector (`rwkv_erase`, rank two,
`linear_scan.Affine.erased_before`) and the delta rule's along its write's (`delta_before`, rank one,
`Affine.corrected_before`), each its update, readout, operands (a head's decay, a key's) and draw."""


def pre_decay(rule: str, per_key: bool):
    """`rule`'s (`PRE_DECAY`) derived forms, operands, token scan from a state entering (`scanned`) and its
    draw from one (`entering_of`)."""
    update_fn, readout_fn, scalar, keyed, draw_fn = PRE_DECAY[rule]
    operands = keyed if per_key else scalar
    forms = scan.derive_scan(scan.Scan(update_fn, readout_fn, operands, walked="time",
                                       state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
    return forms, operands, scanned(update_fn, readout_fn), entering_of(draw_fn, per_key)


def entering_of(draw_fn, per_key: bool):
    """`draw_fn`'s operands (its log decays the fourth) and a state entering of unit normals, a decay
    vanishing alone (`VANISHED`, one key's where `per_key`) at the second chunk's first token and at its
    sub-chunk's start, and a gate saturated at a reset's exponent (`SATURATED`, no reset) inside it."""
    def drawn(seed):
        q, k, v, g, *rest = draw_fn(seed, per_key)
        g = np.array(g)
        for token, key, exponent in ((CHUNK, 0, VANISHED), (CHUNK + 5, 3, SATURATED),
                                     (CHUNK + CHUNK // 2, KEY - 1, VANISHED)):
            g[(token, key) if per_key else token] = exponent
        h0 = np.random.default_rng([seed, 6755]).standard_normal((KEY, VALUE))
        return q, k, v, jnp.asarray(g, jnp.float32), *rest, jnp.asarray(h0, jnp.float32)

    return drawn


@pytest.mark.parametrize(("update_fn", "operands", "readout_fn", "erased"), [
    (rwkv_erase, RWKV, rwkv_readout, True), (rwkv_erase, RWKV_KEYED, rwkv_readout, True),
    (delta_before, GATED, gated_readout, False)],
    ids=["an erase's own vector, a head's decay", "a key's", "the write's vector"])
def test_an_erase_along_its_own_vector_reaches_the_rank_two_form(update_fn, operands, readout_fn, erased):
    """RWKV7's token (`rwkv_erase`: an erase along its own vector beside the write) reaches the rank-two
    pre-decay form (`linear_scan.Affine.erased_before`, `Parts.erase`), and a correction along the write's
    own vector (`delta_before`) the shared form, neither the other's."""
    parts = update_parts(update_fn, operands, readout_fn)
    assert (parts.erase is not None, parts.before) == (erased, True)


@pytest.mark.parametrize("target", ["sm86", "sm90"])
@pytest.mark.parametrize("per_key", [False, True], ids=["a head's decay", "a key's"])
def test_an_erase_along_its_own_vectors_chunks_compute_their_token_scan(per_key, target):
    """RWKV7's token (`rwkv_erase`) walked in chunks through its rank-two form (one triangular system for its
    erased values, its writes known), the dual form an H100's extraction takes and the token loop an A10's:
    outputs and final state on par with the token scan's from float64 (check 1)."""
    operands = RWKV_KEYED if per_key else RWKV
    with generation.assumed(target):
        forms = rwkv_forms(operands)
        assert_as_exact(jax.jit(walked_of(forms, operands)), jax.jit(rwkv_tokenwise),
                        functools.partial(rwkv_draw, per_key=per_key), f"rwkv7 on {target}", seeds=16)


@pytest.mark.parametrize("per_key", [False, True], ids=["a head's decay", "a key's"])
def test_an_erase_along_its_own_vectors_chunk_adjoint_is_the_token_scans_gradient(per_key):
    """RWKV7's chunk adjoint (`backward`, its token update replayed: its erase and its write each its own
    outer product), walked back: each token operand's gradient on par with the token scan's from float64
    (check 1)."""
    operands = RWKV_KEYED if per_key else RWKV
    assert_adjoint(rwkv_forms(operands), rwkv_tokenwise, tuple(operands),
                   functools.partial(chunk_of, axes=operands), functools.partial(rwkv_draw, per_key=per_key),
                   ((TOKENS, VALUE), (KEY, VALUE)), "rwkv7 adjoint", 16)


def pulled(run):
    """`run`'s value and its gradient along every primal under the cotangents drawn after them (the drawn
    operands, then the outputs' and the final state's cotangents)."""
    def ran(*drawn):
        value, pull = jax.vjp(run, *drawn[:-2])
        return value, pull(drawn[-2:])

    return ran


def pushed(run):
    """`run`'s value and its tangent along every primal (`jax.jvp`), the tangents drawn after them."""
    def ran(*drawn):
        half = len(drawn) // 2
        return jax.jvp(run, drawn[:half], drawn[half:])

    return ran


@pytest.mark.parametrize("target", ["sm86", "sm90"])
@pytest.mark.parametrize("per_key", [False, True], ids=["a head's decay", "a key's"])
@pytest.mark.parametrize("rule", PRE_DECAY)
def test_a_pre_decay_walk_from_a_state_entering_holds_its_token_scans_gradient(rule, per_key, target):
    """Each form correcting the state a token enters (`PRE_DECAY`: RWKV7's rank-two erase, the delta rule's
    rank-one correction), its chunks (the token loop an A10's extraction takes, the dual or WY form an
    H100's) walked from a state entering, its decays vanishing alone and saturated at a reset's exponent
    with no reset declared (`entering_of`), differentiated through the forms' own derivatives (the decays'
    `_decays_backward`, the decays a token back transposed as their shift): outputs, final state and every
    gradient, the entering state's among them, on par with the token scan's from float64 (check 1). A key
    whose decay vanishes at the correcting token is read from the state it enters, as the recurrence reads
    it: a declared document's start alone reads it zero."""
    with generation.assumed(target):
        forms, operands, tokenwise, drawn = pre_decay(rule, per_key)

        def draws(seed):
            return *drawn(seed), *cotangents_of(seed, (TOKENS, VALUE), (KEY, VALUE))

        assert_as_exact(jax.jit(pulled(walked_of(forms, operands, entering=True))),
                        jax.jit(pulled(tokenwise)), draws, f"{rule} entering on {target}", seeds=16)


@pytest.mark.parametrize("per_key", [False, True], ids=["a head's decay", "a key's"])
@pytest.mark.parametrize("rule", PRE_DECAY)
def test_a_pre_decay_adjoint_from_a_state_entering_is_the_token_scans_gradient(rule, per_key):
    """Each pre-decay form's chunk adjoint (`backward`, its token update replayed) walked back to a state
    entering, its decays vanishing alone and saturated with no reset (`entering_of`): each token operand's
    gradient and the entering state's on par with the token scan's from float64 (check 1)."""
    forms, operands, tokenwise, drawn = pre_decay(rule, per_key)
    assert_adjoint(forms, tokenwise, (*operands, "h0"), functools.partial(chunk_of, axes=operands), drawn,
                   ((TOKENS, VALUE), (KEY, VALUE)), f"{rule} adjoint entering", 16)


@pytest.mark.parametrize("per_key", [False, True], ids=["a head's decay", "a key's"])
@pytest.mark.parametrize("rule", PRE_DECAY)
def test_a_pre_decay_replay_takes_its_token_scans_tangent(rule, per_key):
    """Each pre-decay form's token walk (`Derived.replay`) chunk by chunk from a state entering, as the
    chunked op's walk takes its tangent at every order (`recurrence_chunk._walk`), its decays vanishing alone
    and saturated with no reset (`entering_of`): outputs, final state and their tangent along every operand
    on par with the token scan's from float64 (check 1)."""
    forms, operands, tokenwise, drawn = pre_decay(rule, per_key)

    def draws(seed):
        primals = drawn(seed)
        rng = np.random.default_rng([seed, 1])
        return *primals, *(jnp.asarray(rng.standard_normal(x.shape), jnp.float32) for x in primals)

    assert_as_exact(jax.jit(pushed(replayed_of(forms, operands))), jax.jit(pushed(tokenwise)), draws,
                    f"{rule} replay tangent", seeds=16)


def rwkv_values_readout(h, q, k, v, g, beta, a, c):
    del k, v, g, beta, a, c
    return h @ q


@pytest.mark.parametrize("prepared", [True, False], ids=["prepared", "replayed"])
@pytest.mark.parametrize("declared", [{"readout": rwkv_values_readout}, {"reads": "apart"}],
                         ids=["a readout along the values", "a readout apart from its writes"])
def test_an_erase_along_its_own_vector_beside_a_readout_no_form_takes_is_refused(declared, prepared):
    """RWKV7's token beside a readout along the state's values (its query a value's) or apart from its
    group's writes, which no chunk form takes: refused, named, whether a chunk is prepared or replayed
    (`scan._apart`, which both ask), never computed by forms that do not hold it."""
    operands = {**RWKV, "q": ("time", "value")} if "readout" in declared else RWKV
    forms = scan.derive_scan(scan.Scan(rwkv_erase, declared.get("readout", rwkv_readout), operands,
                                       walked="time", state=((KEY, VALUE), jnp.float32),
                                       reads=declared.get("reads", "after")), saturation=SATURATION)
    widths = {"key": KEY, "value": VALUE}
    chunk = {name: jnp.zeros((CHUNK, *(widths[axis] for axis in axes[1:])), jnp.float32)
             for name, axes in operands.items()}
    with pytest.raises(scan.Refused):
        if prepared:
            forms.prepare(chunk)
        else:
            forms.replay(chunk, jnp.zeros((KEY, VALUE), jnp.float32))


def rwkv_widened(h, q, k, v, g, beta, a, c):
    """`rwkv_erase` erasing along `a` times a float64 vector it closes over (x64's): its erase part wider than
    every operand."""
    return rwkv_erase(h, q, k, v, g, beta, a * np.ones(KEY, np.float64), c)


@pytest.mark.parametrize(("update_fn", "wide", "carried"), [
    (rwkv_erase, "a", jnp.float64), (rwkv_widened, None, jnp.float64), (rwkv_erase, "q", jnp.float32)],
    ids=["an erase vector given wide", "an erase part widened in the update", "a query given wide"])
def test_a_pre_decay_walks_carry_follows_its_updates_parts(update_fn, wide, carried):
    """RWKV7's token over float32 operands, its erase vector given in float64, or widened in its update by a
    float64 vector it closes over, or its query given in float64 (x64's), walked by a token loop (an A10's,
    its step unpriced, the extraction guarded) and replayed (`Derived.replay`): each carries its state in the
    type its update's parts widen it to (`scan._carry_type`), float64 where its erase part is, float32 where
    a readout's operand alone is, where the replay's carry typed by its state alone changed type mid-scan,
    which `lax.scan` refuses."""
    extents = linear_scan.Extents(CHUNK, KEY, VALUE, 1, before=True, erased=True)
    with jax.enable_x64(new_val=True), generation.assumed("sm86"):
        form = scan._form(extents, corrected=True, saturation=SATURATION)
        assert linear_scan.looped((form.outputs, form.state))
        forms = scan.derive_scan(scan.Scan(update_fn, rwkv_readout, RWKV, walked="time",
                                           state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
        drawn = dict(zip(RWKV, rwkv_draw(0), strict=True))
        if wide is not None:
            drawn[wide] = drawn[wide].astype(jnp.float64)
        chunk = chunk_of(drawn, 0, RWKV)
        looped = forms.step(forms.init({}), forms.prepare(chunk))[0][1]
        _, replayed = forms.replay(chunk, jnp.zeros((KEY, VALUE), jnp.float32))
    assert (looped.dtype, replayed.dtype) == (jnp.dtype(carried),) * 2


def test_an_erases_document_after_a_nonfinite_one_is_its_own():
    """RWKV7's tokens from a state entering of inf, a document starting at the fifth token (`Scan.reset`),
    walked by a token loop (an A10's extraction) and replayed: the outputs from the start on and the final
    state on par with the token scan's, which zeroes the state at the start, from float64 (check 1), where
    the loop decayed the earlier document's state by its start's zero before selecting it out, `0 * inf`,
    NaN in every later token."""
    operands = {**RWKV, "first": ("time",)}
    start = 5

    def draw_inf(seed):
        first = np.zeros(TOKENS, bool)
        first[start] = True
        return (*rwkv_draw(seed), jnp.asarray(first), jnp.full((KEY, VALUE), jnp.inf, jnp.float32))

    def tokens(q, k, v, g, beta, a, c, first, h0):
        def one(h, token):
            *xs, first_t = token
            h = rwkv_erase(jnp.where(first_t, jnp.zeros((), h.dtype), h), *xs)
            return h, rwkv_readout(h, *xs)

        final, outputs = jax.lax.scan(one, h0, (q, k, v, g, beta, a, c, first))
        return outputs[start:], final

    def later(run):
        def ran(*xs):
            outputs, final = run(*xs)
            return outputs[start:], final

        return ran

    with generation.assumed("sm86"):
        forms = rwkv_forms(operands, reset="first")
        for arm, label in ((walked_of(forms, operands, entering=True), "loop"),
                           (replayed_of(forms, operands), "replay")):
            assert_as_exact(jax.jit(later(arm)), jax.jit(tokens), draw_inf, f"rwkv7 after inf, {label}",
                            seeds=16)


def values_readout(h, q, k, v, g, beta):
    """A read of the state along its values, `h q`, its query a value's."""
    del k, v, g, beta
    return h @ q


def valued(draw_fn):
    """`draw_fn`'s operands, its queries along the state's values."""
    def drawn(seed):
        q = np.random.default_rng([seed, 2]).standard_normal((TOKENS, VALUE))
        return jnp.asarray(q, jnp.float32), *draw_fn(seed)[1:]

    return drawn


@pytest.mark.parametrize(("operands", "draw_fn", "along"), [
    (GATED, gated_draw, "keys"), (KIMI, kimi_draw, "keys"), (GATED, gated_draw, "values")],
    ids=["a head's decay", "a key's", "a head's decay, read along the values"])
def test_a_pre_decay_corrections_chunk_adjoint_is_the_token_scans_gradient(operands, draw_fn, along):
    """The pre-decay delta rule (`delta_before`), its state read along its keys or its values
    (`values_readout`, the replay's read of its whole state): its chunks' outputs and final state and its
    chunk adjoint's (`backward`, its token update replayed, its correction read before the decay) each token
    operand's gradient on par with the token scan's from float64 (check 1), its decays' through the shifted
    parts' rows."""
    readout_fn, draw_fn = (gated_readout, draw_fn) if along == "keys" else (values_readout, valued(draw_fn))
    operands = operands if along == "keys" else {**operands, "q": ("time", "value")}
    forms = scan.derive_scan(scan.Scan(delta_before, readout_fn, operands, walked="time",
                                       state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
    tokenwise = gated_tokenwise(delta_before, readout_fn=readout_fn)
    assert_as_exact(jax.jit(gated_walk(forms, operands)), jax.jit(tokenwise), draw_fn,
                    f"delta_before along its {along}", seeds=16)
    assert_adjoint(forms, tokenwise, ("q", "k", "v", "g", "beta"), functools.partial(chunk_of, axes=operands),
                   draw_fn, ((TOKENS, VALUE if along == "keys" else KEY), (KEY, VALUE)),
                   f"delta_before adjoint along its {along}", 16)


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
    "delta_before": (delta_before, gated_readout, GATED, (KEY, VALUE), gated_draw, tuple(GATED), {}),
    "delta_before_per_key": (delta_before, gated_readout, KIMI, (KEY, VALUE), kimi_draw, tuple(GATED), {}),
    "rwkv7": (rwkv_erase, rwkv_readout, RWKV, (KEY, VALUE), rwkv_draw, tuple(RWKV), {}),
    "rwkv7_per_key": (rwkv_erase, rwkv_readout, RWKV_KEYED, (KEY, VALUE),
                      functools.partial(rwkv_draw, per_key=True), tuple(RWKV), {}),
}
"""Each rule a reset walk is derived for: its update and readout, operands, state, draw, token operands and
constants. A decay in base 2, and exponents in float16, whose partials' sum of several resets passes its
largest; corrections reading the state a token enters (`delta_before`), a reset's zero, through the decays
a token back."""


READING = {"after": "", "before": ", reading the state entering each token",
           "apart": ", reading apart from its group's writes"}
"""A reset site's label's suffix for where its tokens read the state (`Scan.reads`)."""


def resets_of(seed) -> jax.Array:
    """Each token a document's first with probability 1/6, and so the first token of the second chunk and two
    consecutive tokens inside it: up to several resets a chunk, its partials' sum many times `RESET_DECAY`."""
    first = np.random.default_rng(seed + 11).uniform(size=TOKENS) < 1 / 6
    first[[CHUNK, CHUNK + 9, CHUNK + 10]] = True
    return jnp.asarray(first)


def resetting(rule: str, reads: str = "after"):
    """`rule`'s chunk forms with a reset operand (`first`), each token reading the state after it or entering
    it (`reads`), its token scan whose state each reset token zeroes before its update, and each as a
    function of the resets and the token operands."""
    update_fn, readout_fn, declared, shape, _, names, constants = RESETTING[rule]
    operands = {**declared, "first": ("time",)}
    forms = scan.derive_scan(scan.Scan(update_fn, readout_fn, operands, walked="time",
                                       state=(shape, jnp.float32), reset="first", reads=reads),
                             saturation=SATURATION)

    def values(first, tokens):
        return {**dict(zip(names, tokens, strict=True)), **constants, "first": first}

    def chunked(given, index):
        return {name: value[index * CHUNK:(index + 1) * CHUNK] if operands[name][:1] == ("time",) else value
                for name, value in given.items()}

    def tokens_run(first, *tokens):
        def one(h, token):
            first_t, *xs = token
            named = {**dict(zip(names, xs, strict=True)), **constants}
            entering = jnp.where(first_t, jnp.zeros((), h.dtype), h)
            h = update_fn(entering, **named)
            return h, readout_fn(entering if reads == "before" else h, **named)

        final, outputs = jax.lax.scan(one, jnp.zeros(shape, tokens[0].dtype), (first, *tokens))
        return outputs, final

    return forms, values, chunked, tokens_run


@pytest.mark.parametrize("reads", ["after", "before"])
@pytest.mark.parametrize("rule", RESETTING)
def test_a_reset_walk_in_chunks_computes_the_token_scan_of_each_document(rule, reads):
    """A walk whose resets (`Scan.reset`) fall on a chunk's first token, inside chunks and back to back, its
    chunks through the derived step and outputs, each token reading the state after it or entering it
    (`Scan.reads`, RWKV-6's): on par with the token scan that zeroes the state at each, outputs and final
    state, against the float64 truth (check 1). A chunk's exponents sum several resets, so
    each decay within a document is a difference of two partials past `-3e4`: a plain prefix held it to about
    `3e4 * 6e-8`, an error of 2e-3 in each decay."""
    forms, values, chunked, tokens_run = resetting(rule, reads)

    def derived(first, *tokens):
        given = values(first, tokens)
        state, outputs = forms.init({}), []
        for index in range(TOKENS // CHUNK):
            chunk = chunked(given, index)
            prepared = forms.prepare(chunk)
            outputs.append(forms.mapped(prepared, state))
            state = forms.step(state, prepared)[0]
        return jnp.concatenate(outputs), forms.finish(state, given)

    def draws(seed):
        return resets_of(seed), *RESETTING[rule][4](seed)

    assert_as_exact(jax.jit(derived), jax.jit(tokens_run), draws, f"{rule} reset walk{READING[reads]}",
                    seeds=16)


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
                state = forms.step(state, forms.prepare(chunked(given, index)))[0]
            states.append(state)
        return forms.finish(forms.merge(*states, given), given)

    assert_as_exact(jax.jit(merged), jax.jit(lambda *xs: tokens_run(*xs)[1]),
                    lambda seed: (resets_of(seed), *draw(seed)), "ssd reset merged", seeds=16)


@pytest.mark.parametrize("reads", ["after", "before"])
@pytest.mark.parametrize("rule", RESETTING)
def test_a_reset_walks_chunk_adjoint_is_the_token_scans_gradient(rule, reads):
    """The chunk adjoint of a walk with resets as the forward test's (`resets_of`), each token reading the
    state after it or entering it (`Scan.reads`), walked back: each token operand's gradient on par with the
    token scan's against the float64 truth (check 1)."""
    forms, values, chunked, tokens_run = resetting(rule, reads)
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

    assert_as_exact(jax.jit(as_derived), jax.jit(as_tokens), draws, f"{rule} reset adjoint{READING[reads]}",
                    seeds=16)


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
        outputs, final = made.mapped(prepared, made.init({})), made.step(made.init({}), prepared)[0][1]

        def token(h, values):
            h = update(h, **values)
            return h, readout(h, **values)

        declared_final, declared_outputs = jax.lax.scan(token, jnp.zeros((2, 3), dtype), chunk)
        # Every value here is exact (counts, doublings of one rounded 1.9e100), so the two agree exactly.
        np.testing.assert_array_equal(np.asarray(final), np.asarray(declared_final))
        np.testing.assert_array_equal(np.asarray(outputs), np.asarray(declared_outputs))


@pytest.mark.parametrize(("reverse", "chunk"), [(False, None), (True, None), (False, 24)],
                         ids=["prefixes", "suffixes", "prefixes walked in chunks"])
def test_a_segmented_compensated_scan_of_double_floats_along_an_axis_holds_each_segments_exact_sum(reverse,
                                                                                                  chunk):
    """`scan.compensated` over double floats (`low` each value's low part) along a later axis, restarting
    at each segment's first: each prefix's pair, and each suffix's where `reverse`, is its segment's exact
    sum, the high half that sum rounded once (a suffix's within its segment to its last element, where a
    reverse scan that restarted at segment firsts summed [1, 2, 3, 4] from [T, F, T, F] to [1, 5, 3, 4],
    not [3, 2, 7, 4]: the base review of row 7432's head). Whole numbers with low parts in multiples of
    2^-30 keep every pair's arithmetic exact, so the pairs equal float64's sums, where a plain fp32 sum drops
    every low part. Each value a double float as an
    exact product leaves it, its low part within half its high part's ulp (none beside a zero). Walked in
    chunks of 24 (`chunk`, Mamba-3's frames' walk), the axis of 64 padded to three, a segment crossing a
    chunk's edge sums the same."""
    rng = np.random.default_rng(7432)
    shape = (2, 64, 3)
    high = rng.integers(-8, 9, shape).astype(np.float32)
    low = np.where(high == 0, 0.0, rng.integers(-16, 17, shape) * 2.0 ** -30).astype(np.float32)
    restarts = rng.random(shape[:2]) < 0.1
    restarts[:, 0] = True
    restarts = np.broadcast_to(restarts[:, :, None], shape)
    summed_high, summed_low = jax.jit(functools.partial(scan.compensated, axis=1, reverse=reverse,
                                                        chunk=chunk))(high, low=low, restarts=restarts)
    exact, steps = np.zeros(shape), shape[1]
    for t in reversed(range(steps)) if reverse else range(steps):
        # The sum beside `t` so far, its segment's where `t` opens none: a prefix's where `t` starts no
        # segment, a suffix's where the next token starts none.
        if reverse:
            beside = 0.0 if t == steps - 1 else np.where(restarts[:, t + 1], 0.0, exact[:, t + 1])
        else:
            beside = 0.0 if t == 0 else np.where(restarts[:, t], 0.0, exact[:, t - 1])
        exact[:, t] = beside + high[:, t].astype(np.float64) + low[:, t]
    np.testing.assert_array_equal(np.float64(summed_high) + np.float64(summed_low), exact)
    np.testing.assert_array_equal(np.asarray(summed_high), exact.astype(np.float32))
    # An empty axis scans to empty sums either way (the review: a reverse scan's restarts shifted along it).
    empty = np.zeros((2, 0, 3), np.float32)
    assert [x.shape for x in scan.compensated(empty, axis=1, reverse=reverse, chunk=chunk,
                                              restarts=np.zeros(empty.shape, bool))] == [empty.shape] * 2


@pytest.mark.parametrize("reads", ["after", "before"])
def test_a_group_beside_a_readout_that_reads_none_of_it_is_refused(reads):
    """A group of tokens (`Scan.group`) is what a readout apart from its writes reads none of: declared
    beside a readout after or before its token, which would read it whole, its forms are refused, never
    derived as if no group were declared (train 1004's review F3)."""
    forms = scan.derive_scan(dataclasses.replace(DECLARED, reads=reads, group=2), saturation=SATURATION)
    with pytest.raises(scan.Refused):
        forms.prepare(chunk_of(operands(*draw(0)), 0))


@pytest.mark.parametrize(("update_fn", "operands", "draw_fn"), [
    (gated_delta, GATED, gated_draw), (kimi_delta, KIMI, kimi_draw)], ids=["gated_delta", "kimi_delta"])
def test_a_delta_rule_reading_apart_from_its_own_write_reads_its_decayed_entering_state(update_fn, operands,
                                                                                       draw_fn):
    """The gated delta rule and Kimi's delta attention, each token reading the state after its decay before
    its own corrected write (`Scan.reads` "apart", a group of one: a single-rank Mamba-3 under the delta
    rule, row 7608): their chunks walked and their adjoint walked back on par with the token-by-token scan
    reading `exp(g) h`, outputs, final state and every operand's gradient, against the float64 truth (check
    1); a group of two beside the correction is refused."""
    declared = scan.Scan(update_fn, gated_readout, operands, walked="time", state=((KEY, VALUE), jnp.float32),
                         reads="apart")
    forms = scan.derive_scan(declared, saturation=SATURATION)
    assert_as_exact(jax.jit(gated_walk(forms, operands)), jax.jit(gated_tokenwise(update_fn, "apart")),
                    draw_fn, f"{update_fn.__name__} apart", seeds=16)
    assert_delta_adjoint(forms, update_fn, draw_fn, f"{update_fn.__name__} apart adjoint", reads="apart")
    grouped = scan.derive_scan(dataclasses.replace(declared, group=2), saturation=SATURATION)
    with pytest.raises(scan.Refused):
        gated_walk(grouped, operands)(*draw_fn(0))


def two_sided(h, q, k, v, g, e, beta, exponential=jnp.exp):
    """Two-sided GLA's token: the state decayed along its keys and its values, `exp(g)` each key's and
    `exp(e)` each value's, then written, `h = diag(exp(g)) h diag(exp(e)) + beta k v^T`; its decays
    `exponential`'s (`jnp.exp2`: in base 2, exact at integer exponents)."""
    del q
    return exponential(g)[:, None] * h * exponential(e)[None, :] + beta * k[:, None] * v[None, :]


def two_sided_readout(h, q, k, v, g, e, beta):
    del k, v, g, e, beta
    return (q @ h) * KEY ** -0.5


TWO_SIDED = {**{name: KIMI[name] for name in ("q", "k", "v", "g")}, "e": ("time", "value"), "beta": ("time",)}
"""Its operands in its update's order, as its draw gives them."""


def two_sided_draw(seed, steep: bool = False):
    """`kimi_draw`'s, each value's decay its own beside each key's, down to 1e-3 a token, or `steep`, its
    values' decays of `exp(-30)` to `exp(-60)` a token at every eighth token (a chunk's span past fp32's
    exponents), and a state entering of entries about 0.3."""
    q, k, v, g, beta = kimi_draw(seed)
    rng = np.random.default_rng(seed + 2)
    e = -(10.0 ** rng.uniform(-3, 0, (TOKENS, VALUE)))
    if steep:
        e[::8] = -rng.uniform(30, 60, (TOKENS // 8, VALUE))
    h0 = 0.3 * rng.standard_normal((KEY, VALUE))
    return q, k, v, g, jnp.asarray(e, jnp.float32), beta, jnp.asarray(h0, jnp.float32)


def two_sided_tokenwise(q, k, v, g, e, beta, h0):
    def one(h, token):
        h = two_sided(h, *token)
        return h, two_sided_readout(h, *token)

    final, outputs = jax.lax.scan(one, h0, (q, k, v, g, e, beta))
    return outputs, final


TWO_SIDED_FORMS = scan.Scan(two_sided, two_sided_readout, TWO_SIDED, walked="time",
                            state=((KEY, VALUE), jnp.float32))

ALGORITHMS = {"sm90": False, "sm86": True}
"""The targets each two-sided form is derived under and whether its extraction takes the token loop: an
H100's step latency prices its loop past the dual form, an A10's states none, its loop priced at its work."""


def two_sided_forms(target: str):
    """Two-sided GLA's forms derived under `target`, its extraction the algorithm `ALGORITHMS` names (a
    guard: a test of both forms that ran one twice would hold neither)."""
    extents = linear_scan.Extents(CHUNK, KEY, VALUE, KEY, across=True)
    form = scan._form(extents, corrected=False, saturation=SATURATION)
    assert linear_scan.looped((form.outputs, form.state)) == ALGORITHMS[target]
    return scan.derive_scan(TWO_SIDED_FORMS, saturation=SATURATION)


def two_sided_walked(forms, q, k, v, g, e, beta, h0):
    """`forms`' chunks walked from `h0`: the outputs and the final state."""
    values = {"q": q, "k": k, "v": v, "g": g, "e": e, "beta": beta}
    init = forms.init({})
    outputs, state = walk(forms, values, range(TOKENS // CHUNK), (init[0], h0), TWO_SIDED)
    return jnp.concatenate(outputs), forms.finish(state, values)


@pytest.mark.parametrize("target", list(ALGORITHMS))
@pytest.mark.parametrize("steep", [False, True], ids=["mild", "steep values"])
def test_a_two_sided_decays_chunks_compute_their_token_scan(target, steep):
    """Two-sided GLA (`two_sided`: a decay along the state's keys and its values) walked in chunks from a
    state entering through its derived forms, the dual form and the token loop (`ALGORITHMS`): outputs and
    final state on par with the token scan's from float64 (check 1), its values' decays mild or steep past
    fp32's exponents across a chunk."""
    with generation.assumed(target):
        forms = two_sided_forms(target)
        assert_as_exact(jax.jit(functools.partial(two_sided_walked, forms)), jax.jit(two_sided_tokenwise),
                        functools.partial(two_sided_draw, steep=steep), f"two-sided on {target}", seeds=16)


@pytest.mark.parametrize("target", list(ALGORITHMS))
@pytest.mark.parametrize("steep", [False, True], ids=["mild", "steep values"])
def test_a_two_sided_decays_chunk_adjoint_is_the_token_scans_gradient(target, steep):
    """Two-sided GLA's chunk adjoint (`backward`, its token update replayed) walked back from the states
    each form's walk entered (`ALGORITHMS`): each token operand's gradient, its values' decays' among them,
    and the entering state's, on par with the token scan's from float64 (check 1)."""
    with generation.assumed(target):
        forms = two_sided_forms(target)
        sliced = functools.partial(chunk_of, axes=TWO_SIDED)
        assert_adjoint(forms, two_sided_tokenwise, (*TWO_SIDED, "h0"), sliced,
                       functools.partial(two_sided_draw, steep=steep), ((TOKENS, VALUE), (KEY, VALUE)),
                       f"two-sided adjoint on {target}")


def gla(h, q, k, v, g, beta):
    """GLA's token: the state decayed along its keys alone, then written."""
    del q
    return jnp.exp(g)[:, None] * h + beta * k[:, None] * v[None, :]


def test_two_sided_and_one_sided_walks_merged_in_either_grouping_give_the_final_state():
    """Three walks merged in either grouping, `(a b) c` and `a (b c)`: two-sided GLA's first chunk from the
    state entering (`a`), its next two from a one-sided walk's start (`b`: a walk that decays no values
    stepped through chunks that do), and GLA's last chunk (`c`, one-sided, its walk's decay along the values
    none): the final state on par with the token scan's whose last chunk's value gates are the identity's
    from float64 (check 1), each walk's decay along the values summed, a walk of none the zero sum's."""
    two_sided = scan.derive_scan(TWO_SIDED_FORMS, saturation=SATURATION)
    one_sided = scan.derive_scan(scan.Scan(gla, gated_readout, KIMI, walked="time",
                                           state=((KEY, VALUE), jnp.float32)), saturation=SATURATION)
    last = TOKENS // CHUNK - 1

    def merged(q, k, v, g, e, beta, h0):
        values = {"q": q, "k": k, "v": v, "g": g, "e": e, "beta": beta}
        plain = {name: value for name, value in values.items() if name != "e"}
        start = one_sided.init(chunk_of(plain, 0, KIMI))
        a = walk(two_sided, values, range(1), two_sided.init(chunk_of(values, 0, TWO_SIDED))._replace(h=h0),
                 TWO_SIDED)[1]
        b = walk(two_sided, values, range(1, last), start, TWO_SIDED)[1]
        c = walk(one_sided, plain, range(last, last + 1), start, KIMI)[1]
        return (two_sided.finish(two_sided.merge(two_sided.merge(a, b, values), c, values), values),
                two_sided.finish(two_sided.merge(a, two_sided.merge(b, c, values), values), values))

    def twice(q, k, v, g, e, beta, h0):
        final = two_sided_tokenwise(q, k, v, g, e.at[last * CHUNK:].set(0.0), beta, h0)[1]
        return final, final

    assert_as_exact(jax.jit(merged), jax.jit(twice), two_sided_draw, "two-sided and one-sided merged",
                    seeds=16)


def plain_readout(h, q, k, v, g, e, beta):
    del k, v, g, e, beta
    return q @ h


def test_a_two_sided_decays_pairs_across_a_sub_chunk_are_decayed_before_they_are_summed():
    """Two-sided GLA's dual form (an H100's), two writes of a key of 2^63 and a unit value before a
    sub-chunk's reference (`recurrence_algebra.SUB`) read by a query of 2^64 after it, the value's gate 1/4
    at the reference's next token: the reader's output 2^126 exactly, the token scan's (its every term
    normal), each pair's read of 2^127 decayed before the pairs are summed, where summed first they passed
    float32's largest (row 9029's head review)."""
    from dew.nn.kernels.recurrence_algebra import SUB

    def at(shape, *entries):
        x = np.zeros(shape, np.float32)
        for index, value in entries:
            x[index] = value
        return jnp.asarray(x)

    reader = SUB + 1
    values = {"q": at((CHUNK, KEY), ((reader, 0), 2.0 ** 64)),
              "k": at((CHUNK, KEY), ((0, 0), 2.0 ** 63), ((1, 0), 2.0 ** 63)),
              "v": at((CHUNK, VALUE), ((0, 0), 1.0), ((1, 0), 1.0)), "g": jnp.zeros((CHUNK, KEY)),
              "e": at((CHUNK, VALUE), ((SUB, 0), -2.0)), "beta": jnp.ones((CHUNK,))}
    with generation.assumed("sm90"):
        update = functools.partial(two_sided, exponential=jnp.exp2)
        base2 = dataclasses.replace(TWO_SIDED_FORMS, update=update, readout=plain_readout)
        forms = scan.derive_scan(base2, saturation=SATURATION)
        outputs = forms.mapped(forms.prepare(values), forms.init({}))
    assert float(outputs[reader, 0]) == 2.0 ** 126


@pytest.mark.parametrize("crossed", [False, True], ids=["cancelling", "crossed-key"])
def test_a_portable_per_key_pair_keeps_the_finite_sum_and_left_adjoint(crossed):
    from dew.nn.precision import FP32_SUMS

    left = np.asarray([[np.inf if crossed else 2.0 ** 127, 1 if crossed else 2.0 ** 127]], np.float32)
    right = np.asarray([[1, 1] if crossed else [2, -2]], np.float32)
    decays = np.asarray([[[-0.0 if crossed else 1.0, 1.0]]], np.float32)
    direction = np.asarray([[0, 1]] if crossed else [[1, 0]], np.float32)

    def orders(a, b, d, da):
        def pair(x):
            return scan._paired(x, b, d).sum()

        result, tangent = jax.jvp(pair, (a,), (da,))
        return result, tangent, jax.grad(pair)(a)

    with jax.default_matmul_precision(FP32_SUMS):
        result, tangent, adjoint = jax.jit(orders)(left, right, decays, direction)
    assert float(result) == (1 if crossed else 0)
    assert float(tangent) == (1 if crossed else 2)
    np.testing.assert_array_equal(adjoint, [[0, 1]] if crossed else [[2, -2]])


def test_a_portable_per_key_transpose_sums_opposite_cotangents_before_they_land():
    row = np.ones((2, 1), np.float32)
    keys = np.asarray([[2.0 ** 127], [-2.0 ** 127]], np.float32)
    decay = np.asarray([[[1.0], [-0.0]], [[1.0], [1.0]]], np.float32)
    cotangent = np.asarray([[0, 0], [2, 2]], np.float32)

    @jax.jit
    def pull(q, k, factors, ct):
        _, back = jax.vjp(lambda x: scan._portable_pairs(x, k, factors, None), q)
        return back(ct)[0]

    np.testing.assert_array_equal(pull(row, keys, decay, cotangent), np.zeros_like(row))


@pytest.mark.parametrize("side", ["reader", "state"])
def test_a_subnormal_read_scale_keeps_the_primal_and_pullback_in_range(side):
    scale = np.float32(2.0 ** -128)
    giant = np.float32(2.0 ** 127)
    reader = np.asarray([giant if side == "reader" else 1], np.float32)
    state = np.asarray([[giant if side == "state" else 1]], np.float32)
    cotangent = np.asarray([giant], np.float32)

    def read(q, h):
        return scan._read(h, q, values_read=False, scale=scale)

    def orders(q, h, ct):
        out, pull = jax.vjp(read, q, h)
        moved = jax.jvp(lambda q: read(q, h), (q,), (ct,))[1]
        mixed = jax.jvp(lambda h: jax.grad(lambda x: (read(x, h) * ct).sum())(q), (h,), (ct[:, None],))[1]
        return out, pull(ct), moved, mixed

    out, (dq, dh), moved, mixed = jax.jit(orders)(reader, state, cotangent)
    np.testing.assert_array_equal(out, [0.5])
    np.testing.assert_array_equal(dq, state[:, 0] * 0.5)
    np.testing.assert_array_equal(dh, reader[:, None] * 0.5)
    np.testing.assert_array_equal(moved, state[:, 0] * 0.5)
    np.testing.assert_array_equal(mixed, [2.0 ** 126])


def test_a_mixed_width_subnormal_read_uses_the_first_multiply_dtype_for_admission():
    scale = np.float32(2.0 ** -149)

    def read(q, h):
        return scan._read(h, q, values_read=False, scale=scale)

    def orders(q, h, cotangent):
        value, pull = jax.vjp(read, q, h)
        return value, pull(cotangent)

    with jax.enable_x64():
        inputs = (np.asarray([0.25], np.float32), np.asarray([[2.0 ** 127]], np.float64),
                  np.asarray([2.0 ** 100], np.float64))
        value, (dq, dh) = jax.jit(orders)(*inputs)
    assert value.dtype == jnp.float64
    np.testing.assert_array_equal(value, [2.0 ** -24])
    np.testing.assert_array_equal(dq, [2.0 ** 78])
    np.testing.assert_array_equal(dh, [[2.0 ** -51]])


def test_a_rank_one_subnormal_read_keeps_the_explicit_value_axis():
    from jax.sharding import AbstractMesh, AxisType, NamedSharding, PartitionSpec, use_abstract_mesh

    mesh = AbstractMesh((2,), ("value",), axis_types=(AxisType.Explicit,))
    whole = jax.ShapeDtypeStruct((3,), jnp.float32, sharding=NamedSharding(mesh, PartitionSpec()))
    values = jax.ShapeDtypeStruct((4,), jnp.float32, sharding=NamedSharding(mesh, PartitionSpec("value")))

    def read(q, left, right):
        return scan._read((left, right), q, values_read=False, scale=np.float32(2.0 ** -128))

    with use_abstract_mesh(mesh):
        out = jax.eval_shape(read, whole, whole, values)
    assert out.shape == (4,)
    assert out.sharding == values.sharding


DEPTHS = {"past the least normal": (2.0 ** 100, None), "past two normal factors": (2.0 ** 127, -58.375)}
"""The isolated write's size and its three gates (`isolated_draw`): about 2^100 through gates of -39 to -41
(`e^-120`, past fp32's least normal, row 9660), or about 2^127 through three of -58.375 (`e^-175.125`, past
two normal factors' reach of 2^-252, the decayed write about 1.5e-38, normal)."""


def isolated_draw(seed: int, side: str, merged: bool = False, depth: str = "past the least normal"):
    """Two-sided GLA's operands of one write (its key's where `side` is "keys", its value's where "values")
    and three gates on that side after it (`DEPTHS`' `depth`), every other write none and every other decay
    at most 1e-2 a token: the write decayed to a normal value, as the token scan keeps it, where each gate's
    span past the write alone is past fp32's least normal (row 9660), or past two normal factors' reach. The
    gated side's exponents multiples of 2^-10, so every span's sum is exact in fp32 and the decays' own
    rounding (an exponent of -120 rounded, `|x|` ulps of its decay) is no part of the witness. The write in
    chunk 1 and its readers in the chunks after it, or where `merged`, the write the first half's last token
    and its gates the second half's first tokens."""
    rng = np.random.default_rng([9660, seed])
    write = TOKENS // 2 - 1 if merged else CHUNK + int(rng.integers(0, CHUNK - 4))
    size, gate = DEPTHS[depth]
    # At 2^127 the write and the unit it multiplies stay below fp32's largest (1.5 times 1.3).
    large, unit = (size * rng.uniform(1, 2), rng.uniform(1, 2)) if gate is None else (
        size * rng.uniform(1, 1.5), rng.uniform(1, 1.3))
    k, v = np.zeros((TOKENS, KEY)), np.zeros((TOKENS, VALUE))
    k[write, 0], v[write, 0] = (large, unit) if side == "keys" else (unit, large)
    g, e = -rng.uniform(0, 1e-2, (TOKENS, KEY)), -rng.uniform(0, 1e-2, (TOKENS, VALUE))
    gated = g if side == "keys" else e
    gated[:] = np.round(gated * 1024) / 1024
    gated[write + 1:write + 4] = -rng.integers(39, 42, (3, gated.shape[1])) if gate is None else gate
    q = rng.standard_normal((TOKENS, KEY))
    q[:(write // CHUNK + 1) * CHUNK] = 0.0
    drawn = (q, k, v, g, e, np.ones(TOKENS), np.zeros((KEY, VALUE)))
    return tuple(jnp.asarray(x, jnp.float32) for x in drawn)


@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("target", list(ALGORITHMS))
@pytest.mark.parametrize("side", ["keys", "values"])
def test_a_large_write_through_gates_past_the_least_normal_is_kept(target, side, depth):
    """Two-sided GLA's forms (`ALGORITHMS`) over one large write decayed by three gates on its key's or its
    value's side (`isolated_draw`, `DEPTHS`): outputs of its readers in later chunks and the final state on
    par with the token scan's from float64 (check 1), each decay a value takes alone applied as normal
    factors in turn (`scan._Split`), where its span's decay alone (`e^-120`) underflowed to zero (row 9660),
    and two factors' second past their reach (`e^-175.125`) was subnormal and flushed to zero."""
    with generation.assumed(target):
        forms = two_sided_forms(target)
        assert_as_exact(jax.jit(functools.partial(two_sided_walked, forms)), jax.jit(two_sided_tokenwise),
                        functools.partial(isolated_draw, side=side, depth=depth),
                        f"a large write's {side} {depth} on {target}", seeds=16)


@pytest.mark.parametrize("depth", DEPTHS)
@pytest.mark.parametrize("side", ["keys", "values"])
def test_a_large_write_through_a_later_walks_gates_is_kept_by_the_merge(side, depth):
    """Two-sided GLA's tokens walked in two halves, a large write the first's last token and three gates on
    its key's or its value's side the second's first (`isolated_draw`, `DEPTHS`), merged: the final state on
    par with the token scan's from float64 (check 1), the later walk's decay applied as normal factors in
    turn (`scan._scaled_exp`), where a floor at the vanished exponent (-105) zeroed `e^-120` (row 9660) and
    two factors' second past their reach (`e^-175.125`) was subnormal and flushed to zero."""
    forms = scan.derive_scan(TWO_SIDED_FORMS, saturation=SATURATION)
    half = TOKENS // CHUNK // 2

    def merged(q, k, v, g, e, beta, h0):
        values = {"q": q, "k": k, "v": v, "g": g, "e": e, "beta": beta}
        init = forms.init(chunk_of(values, 0, TWO_SIDED))
        a, b = (walk(forms, values, chunks, init._replace(h=h0) if not chunks.start else init, TWO_SIDED)[1]
                for chunks in (range(half), range(half, 2 * half)))
        return forms.finish(forms.merge(a, b, values), values)

    assert_as_exact(jax.jit(merged), jax.jit(lambda *xs: two_sided_tokenwise(*xs)[1]),
                    functools.partial(isolated_draw, side=side, merged=True, depth=depth),
                    f"a large write's {side} {depth} merged", seeds=16)


def scalar_gla(h, q, k, v, g, beta):
    """GLA's token under a decay a head's: the state decayed by `exp(g)`, then written."""
    del q
    return jnp.exp(g) * h + beta * k[:, None] * v[None, :]


def steep_last_draw(seed: int):
    """`scalar_gla`'s operands and a state entering of about 2^90 (within 2^96 of its values of about one,
    row 2344's range): the last chunk no write and three gates of -39 to -41 at its second to fourth tokens,
    so its readers after them read the state through a span past fp32's least normal (`e^-120`), about
    2^-83, normal, and the chunk leaves it so. Its exponents multiples of 2^-10, every span's sum exact in
    fp32 (row 9667's rounding none of the witness)."""
    rng = np.random.default_rng([9660, seed, 2])
    last = TOKENS - CHUNK
    k = rng.standard_normal((TOKENS, KEY))
    g = -np.round(rng.uniform(0, 1e-2, TOKENS) * 1024) / 1024
    g[last + 1:last + 4] = -rng.integers(39, 42, 3)
    beta = rng.uniform(0.05, 0.95, TOKENS)
    beta[last:] = 0.0
    drawn = (rng.standard_normal((TOKENS, KEY)), k / np.linalg.norm(k, axis=-1, keepdims=True),
             rng.standard_normal((TOKENS, VALUE)), g, beta, 2.0 ** 90 * rng.standard_normal((KEY, VALUE)))
    return tuple(jnp.asarray(x, jnp.float32) for x in drawn)


def late_reads(run):
    """`run`'s outputs of the last chunk's readers after its gates and its final state, their tangent along
    every operand (`jax.jvp`) and their gradient by the log decays (`jax.vjp`): the split decays' own
    (`scan._split_term`). A reader's gradient by its query, its cotangent times the decay before the state it
    reads, is a product's range, fla's (row 2344)."""
    def late(*primals):
        outputs, final = run(*primals)
        return outputs[TOKENS - CHUNK + 4:], final

    def ran(*primals):
        value, pull = jax.vjp(late, *primals)
        tangents = tuple(jnp.full_like(x, 1e-2) for x in primals)
        decays = pull(tuple(map(jnp.ones_like, value)))[3]
        return {"late": value[0], "final": value[1], "decays' gradient": decays,
                "tangent": jax.jvp(late, primals, tangents)[1]}

    return ran


def test_a_steep_chunks_split_decays_keep_the_state_it_enters_and_leaves():
    """GLA under a decay a head's (`scalar_gla`), its dual form (an H100's), the state entering its last chunk
    about 2^90 and that chunk's three gates of -40 (`steep_last_draw`): its readers after the gates (through
    `entering`) and the state it leaves (through `whole`), their tangent and their gradient by the decays
    through the forms' own derivatives (`late_reads`), on par with the token scan's from float64 (check 1),
    each standalone decay (`e^-120`) taken as two normal factors (`scan._Split`), where whole it underflowed
    to zero and read the state as none (row 9660)."""
    declared = scan.Scan(scalar_gla, gated_readout, GATED, walked="time", state=((KEY, VALUE), jnp.float32))

    def tokenwise(q, k, v, g, beta, h0):
        def one(h, token):
            h = scalar_gla(h, *token)
            return h, gated_readout(h, *token)

        final, outputs = jax.lax.scan(one, h0, (q, k, v, g, beta))
        return outputs, final

    with generation.assumed("sm90"):
        forms = scan.derive_scan(declared, saturation=SATURATION)

        def walked(q, k, v, g, beta, h0):
            values = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
            outputs, state = walk(forms, values, range(TOKENS // CHUNK), (forms.init({})[0], h0), GATED)
            return jnp.concatenate(outputs), forms.finish(state, values)

        assert_as_exact(jax.jit(late_reads(walked)), jax.jit(late_reads(tokenwise)), steep_last_draw,
                        "a steep last chunk's split decays", seeds=16)


@pytest.mark.parametrize(("update_fn", "operands"), [(scalar_gla, GATED), (gla, KIMI)],
                         ids=["a decay a head's", "a decay a key's"])
def test_a_nan_key_reads_nan_through_a_decay_that_underflows(update_fn, operands):
    """GLA's dual form (an H100's) over one chunk whose every log decay is -80, so the span from token 0 to
    token 2 underflows to a live zero, and whose first key is NaN: token 2 reads NaN, as the recurrence does
    (its state NaN from token 0 on), the pair kept where only a crossed decay (`scan._crossed`'s negative
    zero) drops one, never one that underflows (row 8808: a pair dropped on any zero read finite)."""
    declared = scan.Scan(update_fn, gated_readout, operands, walked="time", state=((KEY, VALUE), jnp.float32))
    rng = np.random.default_rng(8808)
    keys = rng.standard_normal((CHUNK, KEY))
    keys[0] = np.nan
    values = {"q": jnp.asarray(rng.standard_normal((CHUNK, KEY)), jnp.float32),
              "k": jnp.asarray(keys, jnp.float32),
              "v": jnp.asarray(rng.standard_normal((CHUNK, VALUE)), jnp.float32),
              "g": jnp.full((CHUNK,) + (() if operands is GATED else (KEY,)), -80.0, jnp.float32),
              "beta": jnp.full((CHUNK,), 0.5, jnp.float32)}
    with generation.assumed("sm90"):
        forms = scan.derive_scan(declared, saturation=SATURATION)
        outputs = forms.mapped(forms.prepare(values), forms.init({}))
    assert np.isnan(np.asarray(outputs[2])).all()


def corrected_two_sided(h, q, k, v, g, e, beta):
    """A two-sided decay beside the delta rule's correction, which no chunk form takes."""
    del q
    decayed = jnp.exp(g)[:, None] * h * jnp.exp(e)[None, :]
    return decayed + beta * k[:, None] * (v - k @ decayed)[None, :]


def two_sided_values_read(h, q, k, v, g, e, beta):
    del k, v, g, e, beta
    return h @ q


APART = dataclasses.replace(TWO_SIDED_FORMS, reads="apart")
"""Two-sided GLA read apart from its own write, which no form takes."""


@pytest.mark.parametrize(("declared", "path"), [
    (APART, "prepare"), (APART, "replay"),
    (dataclasses.replace(TWO_SIDED_FORMS, update=corrected_two_sided), "prepare"),
    (dataclasses.replace(TWO_SIDED_FORMS, readout=two_sided_values_read,
                         operands={**TWO_SIDED, "q": ("time", "value")}), "prepare")],
    ids=["apart, prepared", "apart, replayed", "corrected", "read along the values"])
def test_a_two_sided_decay_outside_its_forms_is_refused(declared, path):
    """A decay along the state's values beside a readout apart from its group's writes, prepared or replayed,
    beside the delta rule's correction, or beside a readout along the values: refused (`scan.Refused`)."""
    q, k, v, g, e, beta, h0 = two_sided_draw(0)
    values = {"q": q, "k": k, "v": v, "g": g, "e": e, "beta": beta}
    if "q" in declared.operands and declared.operands["q"][1:] == ("value",):
        values["q"] = v
    chunk = chunk_of(values, 0, TWO_SIDED)
    forms = scan.derive_scan(declared, saturation=SATURATION)
    with pytest.raises(scan.Refused):
        forms.prepare(chunk) if path == "prepare" else forms.replay(chunk, h0)
