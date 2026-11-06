"""Storage is not multiply precision: a diagnostic, not a product change."""
import sys
sys.path[:0] = ['src', 'tests', 'tools']
import jax
import jax.numpy as jnp
import numpy as np
from dew.nn.kernels.templates.recurrence.mainloop import Operand

x = jnp.array([[1.0 + 2**-8 + 2**-17]], jnp.float32)
for parts in (2, 3):
    operand = Operand(1, parts, jnp.dtype(jnp.float32))
    found = jax.jit(lambda x: operand.joined(operand.split(x)))(x)
    print('K parts', parts, 'input', np.asarray(x).item(), 'joined', np.asarray(found).item(),
          'error', np.asarray(found-x).item(), flush=True)
    assert bool(jnp.all(found == x)) == (parts == 3)
