"""Debug (not for landing): which exported names are retained, and how embed leaves map."""
import sys
sys.path.insert(0, "tests")
import jax, numpy as np
from test_gemma3n import clipped_gemma3n
loaded, model, clip = clipped_gemma3n()
print("RETAINED", sorted(loaded.retained_tensors)[:10] if loaded.retained_tensors else None)
params = loaded.variables["params"]
print("TOP", sorted(params)[:12], "COLLECTIONS", sorted(loaded.variables))
zeros = jax.tree.map(np.zeros_like, params)
exported = loaded.export({**loaded.variables, "params": zeros})
for name, value in exported.items():
    if np.any(value != 0):
        print("NONZERO-FROM-ZEROS", name, value.shape, float(np.abs(value).max()))
