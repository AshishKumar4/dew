"""A recurrence's chunk forms derived from its token update (`dew.algebra.scan`, docs/design/algebra.md §6.9):
Mamba-2's state-space dual (a scalar decay, a rank-one input, a contraction readout and a skip), the gated
delta rule (a decayed, corrected state: the compact WY form) and Kimi's delta attention (its decay each key's
own), each stated token by token, walked in chunks
through its derived step and outputs (and Mamba-2's two walks merged), each held to the float64 truth
(check 1) against the token-by-token scan, with packed documents' resets too (`Scan.reset`), across which no
gradient flows; and updates no chunk form here holds, refused."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.algebra import scan

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


def operands(x, dt, B, C) -> dict:
    return {"x": x, "dt": dt, "B": B, "C": C, "rate": jnp.float32(RATE), "skip": jnp.float32(SKIP)}


def chunk_of(values: dict, index: int) -> dict:
    tokens = slice(index * CHUNK, (index + 1) * CHUNK)
    return {name: value[tokens] if OPERANDS[name][:1] == ("time",) else value
            for name, value in values.items()}


def walk(forms, values, chunks: range, state):
    outputs = []
    for index in chunks:
        chunk = chunk_of(values, index)
        outputs.append(forms.mapped(chunk, state))
        state = forms.step(state, chunk)
    return outputs, state


def test_the_derived_chunks_compute_the_token_scan():
    """The derived chunk outputs and step over every chunk in order: on par with the token-by-token scan,
    outputs and final state, against the float64 truth (check 1)."""
    forms = scan.derive_scan(DECLARED)

    def derived(x, dt, B, C):
        values = operands(x, dt, B, C)
        outputs, state = walk(forms, values, range(TOKENS // CHUNK), forms.init({}))
        return jnp.concatenate(outputs), forms.finish(state, values)

    assert_as_exact(jax.jit(derived), jax.jit(tokenwise), draw, "ssd chunks", seeds=16)


def test_two_walks_merged_give_the_final_state():
    """The tokens walked in two halves, the second from a zero state, merged: the final state on par with
    the token-by-token scan's."""
    forms = scan.derive_scan(DECLARED)
    half = TOKENS // CHUNK // 2

    def merged(x, dt, B, C):
        values = operands(x, dt, B, C)
        _, first = walk(forms, values, range(half), forms.init({}))
        _, second = walk(forms, values, range(half, 2 * half), forms.init({}))
        return forms.finish(forms.merge(first, second), values)

    assert_as_exact(jax.jit(merged), jax.jit(lambda *xs: tokenwise(*xs)[1]), draw, "ssd merged", seeds=16)


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
                                       state=((STATE, WIDTH), jnp.float32)))
    with pytest.raises(scan.Refused):
        forms.step(forms.init({}), chunk_of(operands(*draw(0)), 0))


def gated_delta(h, q, k, v, g, beta):
    """The gated delta rule: the state decayed and its key's old value corrected toward the new one,
    `h = exp(g) (h - beta k (k^T h)) + beta k v^T`."""
    del q
    return jnp.exp(g) * (h - beta * k[:, None] * (k @ h)[None, :]) + beta * k[:, None] * v[None, :]


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
                                       state=((KEY, VALUE), jnp.float32)))

    def derived(q, k, v, g, beta):
        values = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
        state, outputs = forms.init({}), []
        for index in range(TOKENS // CHUNK):
            chunk = {name: value[index * CHUNK:(index + 1) * CHUNK] for name, value in values.items()}
            outputs.append(forms.mapped(chunk, state))
            state = forms.step(state, chunk)
        return jnp.concatenate(outputs), forms.finish(state, values)

    assert_as_exact(jax.jit(derived), jax.jit(gated_tokenwise(update_fn)), draw_fn, update_fn.__name__,
                    seeds=16)


def adjoint(forms, chunk_of_values, count: int):
    """A loss's gradient through the derived forms: the chunks walked forward from a zero state (each
    entering state kept), then back through each chunk's `backward`, the cotangents of the outputs and the
    final state the loss's (`COTANGENT`, a fixed draw)."""
    def run(values, cotangents):
        d_outputs, d_final = cotangents
        states = [forms.init({})]
        for index in range(count):
            states.append(forms.step(states[-1], chunk_of_values(values, index)))
        d_state, grads = d_final, []
        for index in reversed(range(count)):
            chunk = chunk_of_values(values, index)
            d_state, d_chunk = forms.backward(chunk, states[index], d_state,
                                              d_outputs[index * CHUNK:(index + 1) * CHUNK])
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


def test_the_ssd_chunks_adjoint_is_the_token_scans_gradient():
    """Mamba-2's chunk adjoint (`backward`), walked back over the chunks: each token operand's gradient on par
    with the token-by-token scan's against the float64 truth (check 1)."""
    forms = scan.derive_scan(DECLARED)
    names = ("x", "dt", "B", "C")

    def chunked(values, index):
        return chunk_of({**values, "rate": jnp.float32(RATE), "skip": jnp.float32(SKIP)}, index)

    derived = adjoint(forms, chunked, TOKENS // CHUNK)

    def draw_with_cotangents(seed):
        return (*draw(seed), *cotangents_of(seed, (TOKENS, WIDTH), (STATE, WIDTH)))

    def as_derived(x, dt, B, C, d_out, d_final):
        grads = derived({"x": x, "dt": dt, "B": B, "C": C}, (d_out, d_final))
        return tuple(grads[name] for name in names)

    def as_tokens(x, dt, B, C, d_out, d_final):
        grads = gradient_of(tokenwise, names)({"x": x, "dt": dt, "B": B, "C": C}, (d_out, d_final))
        return tuple(grads[name] for name in names)

    assert_as_exact(jax.jit(as_derived), jax.jit(as_tokens), draw_with_cotangents, "ssd adjoint", seeds=16)


@pytest.mark.parametrize(("update_fn", "operands", "draw_fn"), [
    (gated_delta, GATED, gated_draw), (kimi_delta, KIMI, kimi_draw)], ids=["gated_delta", "kimi_delta"])
def test_the_delta_rules_chunk_adjoint_is_the_token_scans_gradient(update_fn, operands, draw_fn):
    """The gated delta rule's and Kimi's delta attention's chunk adjoints, walked back: each token operand's
    gradient on par with the token-by-token scan's against the float64 truth (check 1)."""
    forms = scan.derive_scan(scan.Scan(update_fn, gated_readout, operands, walked="time",
                                       state=((KEY, VALUE), jnp.float32)))
    names = ("q", "k", "v", "g", "beta")

    def chunked(values, index):
        return {name: value[index * CHUNK:(index + 1) * CHUNK] for name, value in values.items()}

    derived = adjoint(forms, chunked, TOKENS // CHUNK)

    def draw_with_cotangents(seed):
        return (*draw_fn(seed), *cotangents_of(seed, (TOKENS, VALUE), (KEY, VALUE)))

    def as_derived(q, k, v, g, beta, d_out, d_final):
        grads = derived(dict(zip(names, (q, k, v, g, beta), strict=True)), (d_out, d_final))
        return tuple(grads[name] for name in names)

    def as_tokens(q, k, v, g, beta, d_out, d_final):
        values = dict(zip(names, (q, k, v, g, beta), strict=True))
        grads = gradient_of(gated_tokenwise(update_fn), names)(values, (d_out, d_final))
        return tuple(grads[name] for name in names)

    assert_as_exact(jax.jit(as_derived), jax.jit(as_tokens), draw_with_cotangents,
                    f"{update_fn.__name__} adjoint", seeds=16)


RESETTING = {
    "ssd": (update, readout, OPERANDS, (STATE, WIDTH), draw, ("x", "dt", "B", "C"),
            {"rate": RATE, "skip": SKIP}),
    "gated_delta": (gated_delta, gated_readout, GATED, (KEY, VALUE), gated_draw, tuple(GATED), {}),
    "kimi_delta": (kimi_delta, gated_readout, KIMI, (KEY, VALUE), kimi_draw, tuple(GATED), {}),
}
"""Each rule a reset walk is derived for: its update and readout, operands, state, draw, token operands and
constants."""


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
                                       state=(shape, jnp.float32), reset="first"))

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
            outputs.append(forms.mapped(chunk, state))
            state = forms.step(state, chunk)
        return jnp.concatenate(outputs), forms.finish(state, given)

    assert_as_exact(jax.jit(derived), jax.jit(tokens_run),
                    lambda seed: (resets_of(seed), *RESETTING[rule][4](seed)), f"{rule} reset walk", seeds=16)


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
        assert not np.any(np.asarray(grads["g"][reset])), f"{rule}: the reset token's own decay has a gradient"
