"""The index a scatter with `mode='drop'` gives the entries it must not write.

JAX drops an index at or past its axis. XLA's deterministic scatter on GPU
(`--xla_gpu_deterministic_ops`, which the CUDA test lane sets) linearizes a
multi-dimensional index row-major and drops only a position past the whole
index space, so a column index equal to its axis size lands on the next row's
first element and can overwrite that row's own write there (jax 0.11.2, XLA
at jaxlib 0.11.2; tools/xla_scatter_drop_repro.py, openxla/xla#49380). An index of `DROPPED`
lies past the whole space for any scatter indexing fewer than 2**30
positions, so it is dropped on every backend.
"""

DROPPED = 2**30
