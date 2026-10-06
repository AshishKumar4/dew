import hashlib
import sys
from pathlib import Path

sys.path.insert(0, "tests"); sys.path.insert(0, "tools")
import lane_environment  # noqa: F401
import jax, jax.numpy as jnp, numpy as np
from lpips_reference import drawn_weights
from dew.eval.lpips import LPIPSNetwork, VGG16, SHIFT, SCALE, _unit, variables_from_torch


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array)).tobytes()).hexdigest()[:16]


reference = dict(np.load(Path("tests/fixtures/lpips/drawn.npz")))
vgg, linear = drawn_weights()
print("@@ weights", digest(np.concatenate([v.ravel() for _, v in sorted({**vgg, **linear}.items())])))
print("@@ fixture", {k: digest(v) for k, v in sorted(reference.items()) if k in ("images", "references", "gradient_f64", "distance_f64")})
variables = variables_from_torch(vgg, linear)
with jax.enable_x64(new_val=True):
    wide = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables)
    first, second = (jnp.asarray(reference[key], jnp.float64) for key in ("images", "references"))
    shift, scale = jnp.asarray(SHIFT, jnp.float64), jnp.asarray(SCALE, jnp.float64)
    for name, images in (("first", first), ("second", second)):
        scaled = (images - shift) / scale
        print("@@ scaled", name, digest(scaled))
        outputs = VGG16().apply({"params": wide["params"]["vgg"]}, scaled)
        print("@@ stages", name, [digest(o) for o in outputs])
    # One convolution alone, on the scaled first image.
    kernel = wide["params"]["vgg"]["conv_0"]["kernel"]
    conv = jax.lax.conv_general_dilated((first - shift) / scale, kernel, (1, 1), ((1, 1), (1, 1)),
                                        dimension_numbers=("NHWC", "HWIO", "NHWC"))
    print("@@ conv0", digest(conv))
    big = jax.random.normal(jax.random.key(0), (64, 576), jnp.float64)
    print("@@ matmul", digest(big @ big.T), "sum", float(jnp.sum(big)).hex())
