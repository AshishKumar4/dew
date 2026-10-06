"""Where the drawn LPIPS weights differ across machines: the float32 normal
draw's tail values (numpy's ziggurat takes them through log1pf), and libm's
log1pf and expf themselves over a fixed grid of inputs."""
import ctypes
import ctypes.util
import hashlib
import platform

import numpy as np

libm = ctypes.CDLL(ctypes.util.find_library("m"))
for name in ("log1pf", "expf", "logf"):
    getattr(libm, name).restype = ctypes.c_float
    getattr(libm, name).argtypes = [ctypes.c_float]
print("@@", platform.libc_ver(), np.__version__)
grid = np.random.default_rng(5).random(200_000, dtype=np.float32)
for name in ("log1pf", "expf", "logf"):
    function = getattr(libm, name)
    values = np.array([function(float(-x if name == "log1pf" else (-8 * x if name == "expf" else x)))
                       for x in grid], np.float32)
    print("@@", name, hashlib.sha256(values.tobytes()).hexdigest()[:16])
rng = np.random.default_rng(0)
shapes = [(64, 3), (64,), (64, 64), (64,), (128, 64), (128,), (128, 128), (128,), (256, 128), (256,),
          (256, 256), (256,), (256, 256), (256,), (512, 256)]
for index, shape in enumerate(shapes):
    draw = rng.standard_normal((*shape, 3, 3) if len(shape) == 2 else shape, dtype=np.float32)
    if index == len(shapes) - 1:
        tail = np.flatnonzero(np.abs(draw.ravel()) > 3.4426198558966521)
        print("@@ features.17 digest", hashlib.sha256(draw.tobytes()).hexdigest()[:16], "tail count", tail.size)
        print("@@ tail", [(int(i), float(draw.ravel()[i]).hex()) for i in tail[:12]])
