import sys; sys.path[:0]=["src","tests","."]
import os
import lane_environment
os.environ.update(lane_environment.configured(os.environ))
import functools
import math
import jax
import jax.numpy as jnp
import numpy as np
from dew.algebra import scan
from dew.nn.precision import compensated_add, vanishing
from test_recurrence_chunk import chunk_on, token_by_token, CHUNK
from test_recurrence_stability import SITES, cotangents
from test_recurrence_properties import FORMS, _drawn
from reference_error import assert_as_exact, with_gradients
import reference_error

original_judgement = reference_error._judgement
def logged_judgement(draws, label, *args, **kwargs):
    for i, name in enumerate(draws.names):
        if name.endswith(".rate") or (label == "rated norm 96/8" and name == "[1]"):
            print("K metrics", mode, label, name, "seeds", len(draws.mine),
                  "RMS ratio", reference_error._pooled(draws.mine[:, i], draws.theirs[:, i]),
                  "worst rows", np.exp(np.max(draws.worst[:, i], axis=0)), flush=True)
    return original_judgement(draws, label, *args, **kwargs)
reference_error._judgement = logged_judgement


def split_decays(exponents, base2):
    vanished = exponents <= vanishing(exponents.dtype, base2)
    high, low = scan._compensated(jnp.where(vanished, 0., exponents), restarts=vanished)
    segments = jnp.cumsum(vanished.astype(jnp.int32), axis=0)
    count = exponents.shape[0]
    below = (jnp.arange(count)[:, None] >= jnp.arange(count)[None, :])[:, :, None]
    joined = below & (segments[:, None] == segments[None, :])
    started, ending = segments == 0, segments == segments[-1]
    def exp_pair(hi, lo, mask):
        hi, lo = jnp.where(mask, hi, 0.), jnp.where(mask, lo, 0.)
        if base2:
            n = jnp.floor(hi)
            return jnp.where(mask, jnp.ldexp(jnp.exp2((hi-n)+lo), n.astype(jnp.int32)), 0.)
        grown = jnp.exp(hi)
        return jnp.where(mask, grown + grown * lo, 0.)
    gh, gl = compensated_add((high[:, None], low[:, None]), (-high[None, :], -low[None, :]))
    th, tl = compensated_add((high[-1], low[-1]), (-high, -low))
    return scan._Decays(exp_pair(high, low, started), exp_pair(th, tl, ending),
                        exp_pair(high[-1], low[-1], started[-1]), exp_pair(gh, gl, joined))


def judge(label, f):
    try:
        f()
        print("K", label, "PASS", flush=True)
    except Exception as exc:
        print("K", label, type(exc).__name__, str(exc), flush=True)


def normalized(values, *, reverse=False, restarts=None):
    def combine(left, right):
        summed = compensated_add(left[:2], right[:2])
        if restarts is None:
            return summed
        return jnp.where(right[2], right[0], summed[0]), jnp.where(right[2], right[1], summed[1]), left[2] | right[2]
    if restarts is None:
        return jax.lax.associative_scan(combine, (values, jnp.zeros_like(values)), reverse=reverse)
    hi, lo, _ = jax.lax.associative_scan(combine, (values, jnp.zeros_like(values), restarts), reverse=reverse)
    return hi, lo


original_decays = scan._decays_value
for mode in ("barrier",):
    if mode == "split":
        scan._decays_value = split_decays
        scan._decays_rule.cache_clear()
    if mode == "barrier":
        scan._decays_value = lambda exponents, base2: original_decays(jax.lax.optimization_barrier(exponents), base2)
        scan._decays_rule.cache_clear()
    if mode.startswith("normalized"):
        scan._compensated = normalized
        scan._decays_value = split_decays if mode.endswith("_split") else original_decays
        scan._decays_rule.cache_clear()
    jax.clear_caches()
    hooks = SITES["rated"]
    judge(mode + " gradient", lambda: assert_as_exact(
        with_gradients(functools.partial(chunk_on, "xla", hooks, chunk_size=CHUNK)),
        with_gradients(functools.partial(token_by_token, hooks)),
        functools.partial(cotangents, hooks, "decays near zero"), "rated near zero", seeds=128))
    hooks = FORMS[17]
    judge(mode + " property", lambda: assert_as_exact(
        functools.partial(chunk_on, "xla", hooks, chunk_size=8),
        functools.partial(chunk_on, "xla", hooks), _drawn(hooks, False, 96),
        "rated norm 96/8", seeds=8, truth_fn=functools.partial(token_by_token, hooks)))
