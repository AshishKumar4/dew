import sys
from pathlib import Path
sys.path.insert(0, "tests"); sys.path.insert(0, "tools")
import lane_environment  # noqa: F401
import jax, jax.numpy as jnp, numpy as np
from lpips_reference import drawn_weights
from reference_error import chain_roundings, distance
from dew.eval.lpips import LPIPSNetwork, variables_from_torch
reference = dict(np.load(Path("tests/fixtures/lpips/drawn.npz")))
variables = variables_from_torch(*drawn_weights())
network = LPIPSNetwork()
with jax.enable_x64(new_val=True):
    wide = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
    first, second = (jnp.asarray(reference[key], jnp.float64) for key in ("images", "references"))
    def mean(images):
        return jnp.mean(network.apply(wide, images, second))
    gradient = np.asarray(jax.grad(mean)(first))
    roundings = chain_roundings(jax.make_jaxpr(jax.grad(mean))(first))
truth = reference["gradient_f64"]
scale = float(np.sqrt(np.mean(truth ** 2)))
eps = float(np.finfo(np.float64).eps)
forward = None
print(f"@@ apart {distance(gradient, truth):.3e} bound {2 * roundings * eps * scale:.3e} roundings {roundings} ulps-of-scale {distance(gradient, truth) / (eps * scale):.1f} gradient dtype {gradient.dtype}")

with jax.enable_x64(new_val=True):
    distance64 = np.asarray(network.apply(wide, first, second))
print("@@ forward rel", np.abs(distance64 - reference["distance_f64"]) / reference["distance_f64"])
