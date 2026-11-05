"""Scratch probe: where the derivative contracts' time goes, eager orders against jitted ones."""
import collections
import time

import contract_support
import jax
import jax.numpy as jnp
import pytest
from jax._src import compiler

import test_contracts_objectives as obj
import test_contracts_recurrence as rec

COMPILES = collections.Counter()
_real = compiler.backend_compile_and_load


def counted(*args, **kwargs):
    start = time.perf_counter()
    try:
        return _real(*args, **kwargs)
    finally:
        COMPILES["n"] += 1
        COMPILES["s"] += time.perf_counter() - start


def jitted(fn, primals, direction, cotangent):
    def scalar(f):
        return lambda *x: jnp.sum(f(*x) * cotangent)

    def forward(f):
        return lambda *x: jax.jvp(f, x, direction)[1]

    def gradient(f):
        return jax.grad(scalar(f), argnums=tuple(range(len(primals))))

    remat = jax.checkpoint(fn)
    made = {
        "primal": fn,
        "jvp": forward(fn),
        "vjp": lambda *x: contract_support.pulled(fn, x, cotangent),
        "jvp of grad": lambda *x: jax.jvp(gradient(fn), x, direction)[1],
        "second jvp": lambda *x: jax.jvp(forward(fn), x, direction)[1],
        "jvp of grad, remat": lambda *x: jax.jvp(gradient(remat), x, direction)[1],
        "second jvp, remat": lambda *x: jax.jvp(forward(remat), x, direction)[1],
        "jvp of grad, nested jit": lambda *x: jax.jvp(jax.jit(gradient(fn)), x, direction)[1],
        "jvp of grad, scanned": lambda *x: jax.lax.scan(
            lambda carry, _: (carry, jax.jvp(gradient(fn), x, direction)[1]), 0, None, length=1)[1],
        "grad of grad": lambda *x: jax.grad(
            lambda *y: sum(jnp.sum(jnp.asarray(along, part.dtype) * part)
                           for along, part in zip(direction, gradient(fn)(*y), strict=True)),
            argnums=tuple(range(len(primals))))(*x),
    }
    return {name: (lambda f=f: jax.jit(f)(*primals)) for name, f in made.items()}


CASES = {
    "chunked": lambda edge: rec.test_the_chunked_op_holds_every_order(edge, False),
    "decode": rec.test_the_decode_step_holds_every_order,
    "short": rec.test_the_short_conv_holds_every_order,
}


@pytest.mark.parametrize("mode", ["eager", "jit"])
@pytest.mark.parametrize("case", sorted(CASES))
def test_probe(case, mode, monkeypatch):
    monkeypatch.setattr(compiler, "backend_compile_and_load", counted)
    if mode == "jit":
        monkeypatch.setattr(contract_support, "orders", jitted)
    for edge in contract_support.EDGES:
        COMPILES.clear()
        start = time.perf_counter()
        failed = ""
        try:
            CASES[case](edge)
        except Exception as error:
            failed = f" FAILED {type(error).__name__}: {str(error)[:120]}"
        print(f"\nPROBE {case} {mode} {edge}: {time.perf_counter() - start:.1f} s, {COMPILES['n']} compiles "
              f"{COMPILES['s']:.1f} s{failed}")
