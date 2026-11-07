"""Temporary instrumentation for row 1701; not product code."""
import dataclasses
import os
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from dew.interop import Pretrained

jax.config.update('jax_enable_compilation_cache', False)
loaded = Pretrained.load('tests/fixtures/hf/glm5-next-tiny', dtype='float32', param_dtype='float32')
tokens = (jnp.arange(24, dtype=jnp.int32).reshape(2, 12) * 7) % 31 + 1
unrolled = np.asarray(jax.jit(loaded.model.apply)(loaded.variables, tokens))
scanned = dataclasses.replace(loaded.model, scan_layers=True)
lowered = jax.jit(scanned.apply).lower(loaded.variables, tokens)
artifacts = Path(os.environ.get('PROBE_ARTIFACTS', '/tmp/probe1701'))
artifacts.mkdir(parents=True, exist_ok=True)
(artifacts / 'input.hlo').write_text(lowered.compiler_ir('hlo').as_hlo_text())
compiled = lowered.compile()
(artifacts / 'optimized.hlo').write_text(compiled.as_text())
print('VERSIONS', jax.__version__, jax.lib.__version__, flush=True)
for draw in range(4):
    got = np.asarray(compiled(loaded.variables, tokens))
    print('DRAW', draw, 'error', np.max(np.abs(got-unrolled)), 'finite', np.isfinite(got).all(), flush=True)
