import sys; sys.path[:0]=["src","tests","."]
import os
import tempfile
import pathlib
import subprocess
import lane_environment
os.environ.update(lane_environment.configured(os.environ))
dump = tempfile.mkdtemp(prefix="rated-prefix-codegen-")
os.environ["XLA_FLAGS"] += f" --xla_dump_to={dump} --xla_dump_hlo_as_text=true"
import jax
import jax.numpy as jnp
import numpy as np
from dew.algebra import scan
jax.config.update("jax_enable_compilation_cache", False)

rate = np.float32(-1.7891662120819092)
beta = np.linspace(14, 20, 16, dtype=np.float32)
for barrier in (False, True):
    def prefix(r, b):
        g = r * b
        if barrier:
            g = jax.lax.optimization_barrier(g)
        return scan._compensated(g)
    hi, lo = jax.jit(prefix)(rate, beta)
    truth = np.cumsum((rate * beta).astype(np.float64))
    print("K prefix error", barrier, np.max(np.abs(np.asarray(hi).astype(np.float64)+np.asarray(lo)-truth)), flush=True)
print("K dump files", [p.name for p in pathlib.Path(dump).rglob("*")][:20], flush=True)
for path in pathlib.Path(dump).rglob("*.ll"):
    lines = path.read_text().splitlines()
    contracted = [line.strip() for line in lines if "llvm.fma" in line or "llvm.fmuladd" in line]
    print("K codegen", path.name, "FMA lines", len(contracted), flush=True)
    for line in contracted[:4]:
        print("K", line, flush=True)
for path in pathlib.Path(dump).rglob("*.o"):
    disassembled = subprocess.run(["objdump", "-d", str(path)], capture_output=True, text=True, check=True).stdout
    contracted = [line.strip() for line in disassembled.splitlines() if "fmadd" in line or "fmsub" in line or "fnmadd" in line or "fnmsub" in line]
    if contracted:
        print("K native contraction", path.name, "count", len(contracted), flush=True)
        for line in contracted[:3]:
            print("K", line, flush=True)
