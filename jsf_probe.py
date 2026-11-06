import sys
sys.path[:0] = ['src', 'tests', 'tools']
import dataclasses as D
import jax
import jax.numpy as jnp
import numpy as np
import test_recurrence_chunk as T
from dew.nn.kernels.templates.recurrence import mainloop as ML
from dew.nn.kernels import generation

kw, vw, heads, length, form, seed, mode = sys.argv[1:]
kw, vw, heads, length, seed = map(int, (kw, vw, heads, length, seed))
T.KEY_WIDTH, T.VALUE_WIDTH = kw, vw
hooks = T.FORMED[0] if form == 'formed' else T.HOOKS[4]
(x,) = T.draw(hooks, False, seed, length=length, value_heads=heads)
x = D.replace(x, **{n: getattr(x, n).astype(jnp.bfloat16) for n in
                   ('query', 'key', 'value', 'beta') if getattr(x, n) is not None})
prepared = ML.prepared
def recorded_prepared(*a, **kw):
    run = prepared(*a, **kw)
    print('K effective', run.call.given, run.call.held, 'ranged', run.call.ranged, flush=True)
    return run
ML.prepared = recorded_prepared
if mode == 'merge32':
    ML._merged = lambda call: ML.Operand(call.rows, 3, jnp.dtype(jnp.float32))
if mode == 'merge2':
    ML._merged = lambda call: ML.Operand(call.rows, 2, jnp.dtype(jnp.float32))
if mode == 'held32':
    plain = ML._parts
    ML._parts = lambda dtype, family, fast=False: 3
if mode == 'given32':
    ML._parts = lambda dtype, family, fast=False: 3
if mode in ('given32', 'heldonly32'):
    Call = ML.Call
    def call(*a, **kw):
        c = Call(*a, **kw)
        return D.replace(c, held=ML.Operand(c.rows, 2 if mode == 'given32' else 3,
                                         jnp.dtype(jnp.float32)))
    ML.Call = call
if mode.startswith('exact-'):
    from dew.nn.kernels.templates.operands import Issuer, Parts
    from dew.nn.kernels.templates import product
    plain = Issuer.__call__
    selected = mode[6:].split(',')
    def value(x):
        if isinstance(x, Parts):
            return x.value()
        if isinstance(x, product.Transposed):
            return value(x.value).T
        return x
    def exact(self, name, lhs, rhs):
        if name in selected or 'all' in selected:
            return jnp.dot(value(lhs).astype(jnp.float32), value(rhs).astype(jnp.float32),
                           precision=jax.lax.Precision.HIGHEST)
        return plain(self, name, lhs, rhs)
    Issuer.__call__ = exact
def run(kernel):
    def f(x):
        return T.chunk_on(kernel, hooks, x)
    # A reverse-mode forward, which keeps the template's intermediates.
    if mode == 'accurate':
        with generation.run_precision(D.replace(generation.numerics(), forms='accurate')):
            return jax.jit(lambda x: jax.vjp(f, x)[0])(x)[1]
    return jax.jit(lambda x: jax.vjp(f, x)[0])(x)[1]

with jax.experimental.enable_x64():
    truth = T.token_by_token(hooks, jax.tree.map(
        lambda y: y.astype(jnp.float64) if hasattr(y, 'dtype') and jnp.issubdtype(y.dtype, jnp.inexact) else y,
        x))[1]
truth = np.asarray(truth, np.float64)
for kernel in ('xla', 'mosaic_gpu'):
    a = np.asarray(run(kernel), np.float64)
    delta = a - truth
    rms = np.sqrt(np.mean(delta ** 2))
    rel = np.max(np.sqrt(np.mean(delta ** 2, axis=-1)) /
                 np.maximum(np.sqrt(np.mean(truth ** 2, axis=-1)), 1e-30))
    print('K', sys.argv[1:], kernel, 'rms', rms, 'worst-row', rel, flush=True)
