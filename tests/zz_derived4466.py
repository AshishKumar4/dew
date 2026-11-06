"""Row 4466: the derived forms' fp32 distance from float64 against the portable math's, a form's training
step, its log decays as drawn or zeroed (argv: form, decay drawn|zero, packed one|packed): `K ... leaf ratio`."""
import dataclasses, functools, sys
sys.path[:0] = ['src', 'tests', 'tools']
import jax, jax.numpy as jnp, numpy as np
import recurrence_matrix as M
from reference_error import with_gradients
from test_recurrence_chunk import chunk_on, token_by_token

form, decay, documents = sys.argv[1:4]
case = {'generation': 'sm90', 'form': form, 'kernel': 'derived', 'dtype': 'float32', 'documents': documents}
hooks, starts = M.FORMS[form], M._starts(case)

def drawn(seed):
    operands, cotangents = M._drawn(case, seed)
    if decay == 'zero':
        replaced = {name: jnp.zeros_like(getattr(operands, name)) for name in ('log_decay',)
                    if getattr(operands, name) is not None}
        if 'gate' in [f.name for f in dataclasses.fields(operands)] and operands.gate is not None:
            replaced['gate'] = jnp.full_like(operands.gate, 30.0)
        operands = dataclasses.replace(operands, **replaced)
    return operands, cotangents

def arm(kernel):
    def run(operands, cotangents):
        def ours(operands):
            return M._wide(chunk_on(kernel, hooks, dataclasses.replace(operands, starts=starts)))
        return with_gradients(ours)(operands, cotangents)
    return jax.jit(run)

def truth(operands, cotangents):
    def ours(operands):
        return M._wide(token_by_token(hooks, dataclasses.replace(operands, starts=starts)))
    with jax.enable_x64(True):
        wide = jax.tree.map(lambda x: x.astype(jnp.float64) if hasattr(x, 'dtype') and jnp.issubdtype(x.dtype, jnp.floating) else x, (operands, cotangents))
        return jax.jit(with_gradients(ours))(*wide)

errors = {'derived': [], 'xla': []}
for seed in range(12):
    operands, cotangents = drawn(seed)
    t = jax.tree.leaves(truth(operands, cotangents))
    for kernel in errors:
        got = jax.tree.leaves(arm(kernel)(operands, cotangents))
        errors[kernel].append([float(np.sqrt(np.mean((np.float64(a) - np.float64(b)) ** 2))) for a, b in zip(got, t)])
names = [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(truth(*drawn(0)))[0]]
d, x = np.array(errors['derived']), np.array(errors['xla'])
for i, name in enumerate(names):
    ratio = np.sqrt(np.mean(d[:, i] ** 2)) / max(np.sqrt(np.mean(x[:, i] ** 2)), 1e-300)
    print('K %s decay=%s %s %s derived/xla %.3f' % (form, decay, documents, name, ratio), flush=True)
