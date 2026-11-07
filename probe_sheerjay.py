import sys; sys.path[:0] = ["src", "tests", "."]
import functools
import dataclasses
import jax
import jax.numpy as jnp
import numpy as np
from dew.algebra import scan
from dew.nn.kernels import recurrence_chunk as rc
from test_recurrence_stability import SITES, cotangents, CHUNK
from test_recurrence_chunk import chunk_on, token_by_token
from reference_error import with_gradients, widened, without_ynn, row_errors

jax.config.update("jax_enable_x64", True)
mode = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "x" else "base"
original_chunk = scan._chunk
original_derive = scan.derive_scan

def peeled(form, held, h, entering, to_last, whole, between, *values):
    outputs, final = original_chunk(form, held, h, entering, to_last, whole, between, *values)
    count = entering.shape[0]
    if count == 1:
        prefix = h
    else:
        ending = between[-2] * (jnp.arange(count)[:, None] < count - 1)
        _, prefix = original_chunk(form, held, h, entering, ending, entering[-2], between, *values)
    B, U, s, C, Q, R = values[:6]
    decay = entering[0] if count == 1 else between[-1, -2]
    decayed = prefix * decay[:, None]
    final = decayed + B[-1, :, None] * (U[-1] - s[-1] * (C[-1] @ decayed))[None, :]
    return outputs, final

if mode in ("peel", "both"):
    scan._chunk = peeled

def replay_forms(declaration, **kwargs):
    forms = original_derive(declaration, **kwargs)
    names = tuple(name for name in declaration.operands if name != declaration.reset)
    def replay(h, chunk):
        def token(h, xs):
            if declaration.reset:
                h = jnp.where(xs[declaration.reset], 0.0, h)
            args = [xs[name] for name in names]
            h = declaration.update(h, *args)
            return h, declaration.readout(h, *args)
        h, outputs = jax.lax.scan(token, h, chunk)
        return outputs, h
    def body(h, chunk):
        prepared = forms.prepare(chunk)
        state = (jnp.zeros((h.shape[0],), h.dtype), h)
        return forms.mapped(prepared, state), forms.step(state, prepared)[1]
    rule = jax.custom_vjp(body)
    def forward(h, chunk):
        return body(h, chunk), (h, chunk)
    def backward(saved, cotangent):
        h, chunk = saved
        return jax.vjp(replay, h, chunk)[1](cotangent)
    rule.defvjp(forward, backward)
    def prepare(chunk, *args, **kwargs):
        return chunk
    def step(state, chunk, *args, **kwargs):
        return state[0], rule(state[1], chunk)[1]
    def mapped(chunk, state, *args, **kwargs):
        return rule(state[1], chunk)[0]
    return forms._replace(prepare=prepare, step=step, mapped=mapped)

if mode in ("adjoint", "both"):
    scan.derive_scan = replay_forms

unit = jnp.ones((1, 4, 1, 1), jnp.float32)
value = jnp.array([.314159, -.271828, .577216, 1.414214], jnp.float32).reshape(unit.shape)
decay = jnp.array([-.17, -.23, -.31, -.41], jnp.float32).reshape(unit.shape)
beta = jnp.ones((1, 4, 1), jnp.float32)
zero = jax.jit(jax.grad(lambda g: rc.xla_mainloop(rc.Tiles(4), unit, unit, value, g[..., 0], beta,
                                               scale=1., delta=True, dtype=jnp.float32)[0].sum()))(decay)
print("K", mode, "exactzero", np.asarray(zero).ravel().tolist(), flush=True)

cases = [("scalar delta", "aligned keys", "log_decay"),
         ("gated", "aligned keys", "gate"),
         ("vector delta", "steep then mild", "state")]
for name, regime, leaf in cases:
    hooks = SITES[name]
    candidate = without_ynn(with_gradients(functools.partial(chunk_on, "xla", hooks, chunk_size=CHUNK)))
    reference = without_ynn(with_gradients(functools.partial(token_by_token, hooks)))
    truth = jax.jit(with_gradients(functools.partial(token_by_token, hooks)))
    ratios, mine, theirs, rawmine, rawtheirs = [], [], [], [], []
    for seed in range(8):
        args = cotangents(hooks, regime, seed)
        a, b, t = candidate(*args), reference(*args), truth(*widened(args))
        av, bv, tv = (np.asarray(getattr(x[1], leaf), np.float64) for x in (a, b, t))
        size, ea = row_errors(av, tv)
        _, eb = row_errors(bv, tv)
        mask = size > 0
        mine.append(np.mean((ea[mask] / size[mask])**2))
        theirs.append(np.mean((eb[mask] / size[mask])**2))
        rawmine.append(float(ea.max()))
        rawtheirs.append(float(eb.max()))
        print("K", mode, name, regime, seed, "relative_ratio",
              float(np.sqrt(mine[-1] / theirs[-1])), "worst", rawmine[-1], rawtheirs[-1], flush=True)
    print("K", mode, name, regime, leaf, "pooled",
          float(np.sqrt(np.mean(mine) / np.mean(theirs))),
          "worst", max(rawmine), max(rawtheirs), flush=True)
