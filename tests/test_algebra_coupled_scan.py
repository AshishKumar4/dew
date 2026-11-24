"""A coupled pair's chunk forms derived from its token update (`dew.algebra.coupled_scan`, `scan.Scan` of a
pair state): Momentum DeltaNet's memory and momentum, stated token by token, walked in chunks through the
derived step and outputs and held to the float64 truth (check 1) against the token-by-token scan, one
sequence and packed documents' resets; and pairs no coupled form holds, or whose ranges no declaration
bounds, refused."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from reference_error import assert_as_exact

from dew.algebra import SATURATION, scan
from dew.algebra.forms import Refused
from dew.nn.kernels.recurrence_coupled import declaration

TOKENS, STATE, WIDTH, CHUNK = 96, 16, 8, 32

DECLARED = declaration(1.0, STATE, WIDTH, jnp.float32, reset=True, bound=1.0)
"""Momentum DeltaNet's pair as the chunked op declares it (`recurrence_coupled.declaration`): the error of the
erase key's read of the memory gathered into the momentum, the memory stepping along it, packed documents'
resets marked."""
OPERANDS = tuple(DECLARED.operands)


def step(pair, token):
    """One token of the declared update and its readout, a document's first entering a zero pair."""
    *operands, first = token
    named = dict(zip(OPERANDS, operands, strict=False))
    pair = DECLARED.update(tuple(jnp.where(first, 0.0, s) for s in pair), **named)
    return pair, DECLARED.readout(pair, **named)


def tokenwise(*tokens):
    """The recurrence token by token (`lax.scan`): its outputs and the final pair `[2, STATE, WIDTH]`."""
    zero = jnp.zeros((STATE, WIDTH), tokens[0].dtype)
    final, outputs = jax.lax.scan(step, (zero, zero), tokens)
    return outputs, jnp.stack(final)


def draws(packed: bool):
    """A seed's tokens as fla's layer forms them (unit queries, keys and erase keys; log decays `log sigmoid`
    of normals, the memory's near one; strengths a sigmoid's), and each document's first token where
    `packed` (the first among them)."""
    def draw(seed):
        rng = np.random.default_rng([1454, seed])
        q, k, p = (x / np.linalg.norm(x, axis=-1, keepdims=True)
                   for x in (rng.standard_normal((TOKENS, STATE)) for _ in range(3)))
        v = rng.standard_normal((TOKENS, WIDTH))
        log_alpha = -np.logaddexp(0.0, -(rng.standard_normal(TOKENS) + 3.0))
        log_mu = -np.logaddexp(0.0, -(rng.standard_normal(TOKENS) + 1.0))
        beta = 1.0 / (1.0 + np.exp(-rng.standard_normal(TOKENS)))
        reset = np.zeros(TOKENS, bool)
        if packed:
            reset[[0, 17, 40, 63]] = True
        return (*(jnp.asarray(x, jnp.float32) for x in (q, k, v, log_alpha, beta, p, log_mu)),
                jnp.asarray(reset))
    return draw


def walked(forms):
    """`forms`' chunk outputs and steps over every chunk in order from a zero pair, finished."""
    def derived(*tokens):
        values = dict(zip(OPERANDS, tokens, strict=True))
        state, outputs = forms.init({}), []
        for index in range(TOKENS // CHUNK):
            chunk = {name: value[index * CHUNK:(index + 1) * CHUNK] for name, value in values.items()}
            prepared = forms.prepare(chunk)
            outputs.append(forms.mapped(prepared, state))
            state = forms.step(state, prepared)
        return jnp.concatenate(outputs), forms.finish(state, values)

    return derived


@pytest.mark.parametrize("packed", [False, True], ids=["one sequence", "packed"])
def test_a_coupled_pairs_derived_chunks_compute_its_token_scan(packed):
    """Momentum DeltaNet's pair walked in chunks of its derived forms: its outputs and final memory and
    momentum on par with the token-by-token scan against the float64 truth (check 1), one sequence and
    packed documents, each document's first token reading neither state before it."""
    forms = scan.derive_scan(DECLARED, saturation=SATURATION)
    assert_as_exact(jax.jit(walked(forms)), jax.jit(tokenwise), draws(packed),
                    f"coupled momentum chunks, packed {packed}", seeds=16)


def test_a_coupled_chunks_replay_is_its_token_scan():
    """A chunk's replay (`Derived.replay`, the adjoint's walk) from an entering pair: each output and the pair
    leaving it the token-by-token scan's from that pair, bit for bit."""
    forms = scan.derive_scan(DECLARED, saturation=SATURATION)
    tokens = draws(packed=True)(0)
    entering = jnp.asarray(np.random.default_rng(3).standard_normal((2, STATE, WIDTH)), jnp.float32)
    chunk = {name: value[:CHUNK] for name, value in zip(OPERANDS, tokens, strict=True)}
    outputs, leaving = forms.replay(chunk, entering)
    pair, expected = jax.lax.scan(step, tuple(entering), tuple(chunk.values()))
    np.testing.assert_array_equal(np.asarray(outputs), np.asarray(expected))
    np.testing.assert_array_equal(np.asarray(leaving), np.asarray(jnp.stack(pair)))


def uncoupled(pair, q, k, v, g, beta, p, mu):
    """A pair whose memory steps along the momentum entering the token, not the one it leaves: no coupled
    form's."""
    del q
    h, m = pair
    return jnp.exp(g) * h - beta * m, jnp.exp(mu) * m + k[:, None] * (p @ h - v)[None, :]


def per_key(pair, q, k, v, g, beta, p, mu):
    """The declared update with each key's own memory decay."""
    del q
    h, m = pair
    m = jnp.exp(mu) * m + k[:, None] * (p @ h - v)[None, :]
    return jnp.exp(g * k)[:, None] * h - beta * m, m


@pytest.mark.parametrize(("declared", "why"), [
    (dataclasses.replace(DECLARED, update=uncoupled), "no coupled form"),
    (dataclasses.replace(DECLARED, update=per_key), "a key's own decay"),
    (dataclasses.replace(DECLARED, ranges={}), "no declared ranges")],
    ids=["uncoupled", "per-key decay", "unranged"])
def test_a_pair_no_coupled_form_holds_is_refused(declared, why):
    """A pair whose update is not the coupled form (its memory stepping along the momentum entering the
    token), whose decay is a key's own, or whose decays and weight no declared range bounds is refused when
    its chunk is prepared, never walked token by token in its place."""
    del why
    forms = scan.derive_scan(declared, saturation=SATURATION)
    chunk = {name: value[:CHUNK] for name, value in zip(OPERANDS, draws(packed=False)(0), strict=True)}
    with pytest.raises(Refused):
        forms.prepare(chunk)


def test_a_state_of_no_rank_is_refused_named():
    """A single state of rank zero, no keys by values, is refused, named, as any rank but two is."""
    declared = scan.Scan(lambda h, x: h + x, lambda h, x: h, {"x": ("time",)}, walked="time",
                         state=((), jnp.float32))
    with pytest.raises(Refused):
        scan.derive_scan(declared, saturation=SATURATION)


def test_a_float32_pairs_walk_over_float64_tokens_carries_it_wide():
    """ReviewMomentum1454's x64 case: a pair declared float32 replayed token by token (`Derived.replay`) and
    differentiated (`Derived.backward`) over float64 tokens carries the pair in float64, its outputs and the
    pair leaving it float64, where a carry typed as the entering float32 pair failed the scan."""
    with jax.enable_x64(new_val=True):
        forms = scan.derive_scan(DECLARED, saturation=SATURATION)
        tokens = draws(packed=True)(0)
        chunk = {name: value[:CHUNK].astype(jnp.float64) if jnp.issubdtype(value.dtype, jnp.floating)
                 else value[:CHUNK] for name, value in zip(OPERANDS, tokens, strict=True)}
        entering = jnp.zeros((2, STATE, WIDTH), jnp.float32)
        outputs, leaving = forms.replay(chunk, entering)
        d_entering, _ = forms.backward(chunk, (None, entering), jnp.ones_like(leaving), jnp.ones_like(outputs))
    assert {outputs.dtype, leaving.dtype, d_entering.dtype} == {jnp.dtype(jnp.float64)}
