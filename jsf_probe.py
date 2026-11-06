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
check = mode.endswith('-check')
mode = mode.removesuffix('-check')
direct = mode.startswith('direct-')
mode = mode.removeprefix('direct-')
kw, vw, heads, length, seed = map(int, (kw, vw, heads, length, seed))
T.KEY_WIDTH, T.VALUE_WIDTH = kw, vw
hooks = T.FORMED[0] if form == 'formed' else T.HOOKS[4]
if form == 'l2plain':
    hooks = D.replace(T.FORMED[0], gates=None, write_bound=1.0)
if form == 'gateonly':
    hooks = D.replace(T.FORMED[0], l2norm=False)
(x,) = T.draw(hooks, False, seed, length=length, value_heads=heads)
x = D.replace(x, **{n: getattr(x, n).astype(jnp.bfloat16) for n in
                   ('query', 'key', 'value', 'beta') if getattr(x, n) is not None})
prepared = ML.prepared
def recorded_prepared(*a, **kw):
    run = prepared(*a, **kw)
    if not recorded_prepared.printed:
        print('K effective', run.call.given, run.call.held, 'ranged', run.call.ranged,
              'tiles', run.tiles, flush=True)
        recorded_prepared.printed = True
    return run
recorded_prepared.printed = False
ML.prepared = recorded_prepared
if mode == 'merge32':
    ML._merged = lambda call: ML.Operand(call.rows, 3, jnp.dtype(jnp.float32))
if mode == 'merge2':
    ML._merged = lambda call: ML.Operand(call.rows, 2, jnp.dtype(jnp.float32))
if mode in ('held32', 'storage32'):
    plain = ML._parts
    ML._parts = lambda dtype, family, fast=False: 3
if mode == 'storage32':
    from dew.nn.kernels.templates import product
    terms = product.terms
    product.terms = lambda left, right: [(i, j) for i, j in terms(left, right) if i+j < 2]
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
    if direct and kernel == 'mosaic_gpu':
        from dew.nn.kernels import recurrence
        from dew.nn.precision import l2norm
        from dew.nn.kernels.templates import bundles
        def memory(x):
            q, k = (l2norm(y.astype(jnp.float32)) for y in (x.query, x.key))
            given = recurrence.read(hooks, x, optional=('state',))
            g, beta = recurrence.log_gates(hooks, given, jnp.float32)
            return ML.prepared(
                q, k, x.value, g, beta, x.state, starts=None, max_segments=None,
                scale=kw**-0.5, delta=True, tiles=ML.Tiles(64, 1),
                family=bundles.INTERPRETED, interpret=True, rounded=True,
                ranged=False, states=True, keeps=True).final
        return jax.jit(memory)(x)
    # A reverse-mode forward, which keeps the template's intermediates.
    if mode == 'accurate':
        with generation.run_precision(D.replace(generation.numerics(), forms='accurate')):
            return jax.jit(lambda x: jax.vjp(f, x)[0])(x)[1]
    return jax.jit(lambda x: jax.vjp(f, x)[0])(x)[1]

with jax.enable_x64():
    truth = T.token_by_token(hooks, jax.tree.map(
        lambda y: y.astype(jnp.float64) if hasattr(y, 'dtype') and jnp.issubdtype(y.dtype, jnp.inexact) else y,
        x))[1]
truth = np.asarray(truth, np.float64)
summaries = {}
for kernel in ('xla', 'mosaic_gpu'):
    a = np.asarray(run(kernel), np.float64)
    delta = a - truth
    rms = np.sqrt(np.mean(delta ** 2))
    rel = np.max(np.sqrt(np.mean(delta ** 2, axis=-1)) /
                 np.maximum(np.sqrt(np.mean(truth ** 2, axis=-1)), 1e-30))
    print('K', sys.argv[1:], kernel, 'rms', rms, 'worst-row', rel, flush=True)
    summaries[kernel] = rms, rel
if check:
    mine, reference = summaries['mosaic_gpu'], summaries['xla']
    passed = mine[0] <= 1.5 * reference[0] and mine[1] <= 4 * reference[1]
    print('K', 'PASS' if passed else 'FAIL', 'RMS ratio', mine[0]/reference[0],
          'worst-row ratio', mine[1]/reference[1], flush=True)
    assert passed, 'final memory is not on par with the portable math'

if mode == 'capture':
    captured = {}
    def capture(name, labels):
        plain = getattr(ML, name)
        def wrapped(*args, **kwargs):
            out = plain(*args, **kwargs)
            call = kwargs.get('call')
            if call is None:
                call = next(a for a in args if isinstance(a, ML.Call))
            used = ['A'] if name == '_solved' and len(out) == 1 else labels
            for label, value in zip(used, out):
                if value is None:
                    continue
                parts = 1 if label == 'F' else call.held.parts
                def save(v, label=label, parts=parts):
                    v = np.asarray(v, np.float64)
                    captured[label] = v.reshape(parts, -1, v.shape[-1]).sum(0)
                jax.debug.callback(save, value)
            return out
        setattr(ML, name, wrapped)
    capture('_solved', ['A', 'W', 'U'])
    capture('_weighted', ['W', 'U'])
    capture('_states', ['H', 'C', 'F'])
    # New jit identity, because the earlier plain run had no callbacks.
    def captured_run():
        return jax.jit(lambda y: jax.vjp(lambda z: T.chunk_on('mosaic_gpu', hooks, z), y)[0])(x)
    captured_run()
    jax.effects_barrier()
    baseline = dict(captured)
    ML._parts = lambda dtype, family, fast=False: 3
    captured.clear()
    captured_run()
    jax.effects_barrier()
    for label in baseline:
        a, b = baseline[label], captured[label]
        print('K launch', label, a.shape, 'base-full3 rms',
              np.sqrt(np.mean((a-b)**2)), 'size', np.sqrt(np.mean(b*b)), flush=True)
