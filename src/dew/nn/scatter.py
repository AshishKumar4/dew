"""The index a scatter with `mode='drop'` gives the entries it must not write.

JAX drops an index at or past its axis. XLA's deterministic scatter on GPU
(`--xla_gpu_deterministic_ops`, which the CUDA test lane sets) linearizes a
multi-dimensional index row-major and drops only a position past the whole
index space, so a column index equal to its axis size lands on the next row's
first element and can overwrite that row's own write there (jax 0.11.2, XLA
at jaxlib 0.11.2; openxla/xla#49380, fixed by openxla/xla#49498, which no jaxlib release
carries yet: ledger 951). It reaches a scatter of single elements by a row index against
columns: sampling's present, count and ban masks (tests/test_decoding_components.py's
repetition penalty over padded rows fails on the CUDA lane with the vocabulary's size
in place of `DROPPED`) and the decision head's score layout. An index of `DROPPED` lies
past the whole space of any scatter indexing fewer than 2**30 positions, so those drop
it on every backend. Neither sparse selection's three-index mask nor a scatter of whole
windows, as the KV cache's token writes are, is reached (measured on an A100 with
jaxlib 0.11.2). The paged pool's write into heads it leaves implicit is a second case,
which `DROPPED` alone does not cover: `KVStore.write_tokens` maps it over its group.
"""

DROPPED = 2**30
