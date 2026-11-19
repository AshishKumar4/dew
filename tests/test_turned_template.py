"""The recurrence template's turned calls (`recurrence.Hooks.turns`, Mamba-3's rotating basis) as a GPU
compiles them: a forked call's load rotation placed on a mesh, and a forward-only call's temporaries
(`templates.recurrence.turned`). Their numbers are the turned sites' (tests/test_recurrence_chunk.py,
tests/test_recurrence_stability.py)."""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from generation_support import COMPILES_ON_CPU
from test_recurrence_chunk import chunk_on

from dew.nn.kernels import generation, recurrence
from dew.nn.kernels.recurrence_chunk import Chunk


def test_a_turned_forked_call_lowers_for_a_gpu_under_a_mesh():
    """A turned forked call (`Hooks.turns`, Mamba-3's: its documents' load rotation, `Forked.turning`) with
    entry writes, through the public chunked op on the template, lowered for an H100 under a mesh, as
    training runs every call (`jax.set_mesh`): each operand its shard spec (`_template_launch`), where the
    load rotation took none and the specs' zip with the operands refused the call under any mesh (row
    7973)."""
    from test_recurrence_chunk import TURNED, entered_draw, entering_documents

    operands = entering_documents(entered_draw(TURNED, 0, entries=True)[0])
    mesh = generation.gpu_mesh(jax.sharding.Mesh(np.array(jax.devices()[:1]), ("data",)), "sm90")
    with generation.assumed("sm90"), jax.sharding.use_abstract_mesh(mesh):
        jax.jit(lambda given: chunk_on('mosaic_gpu', TURNED, given)).trace(operands).lower(
            lowering_platforms=("cuda",))


@COMPILES_ON_CPU
def test_a_forward_only_turned_call_keeps_no_more_than_its_frames():
    """A bf16 turned call (`Hooks.turns`, Mamba-3's) without a gradient, compiled for a compile-only H100: its
    temporaries at most the same call's unturned and its frames' own: its keys and queries turned, in their
    dtype, and per token and head its frames' double float, their relative one and their rotation (six fp32
    numbers a pair, `turned.frames`, `_in_last_frame`), where the turned wrapper traced its launch's
    pullback for any call and its forward took the training launches, which keep each chunk's memories and
    inverses for a backward no one runs (audit KPM-04)."""
    from kernel_compile import lowered_for

    rows, length, heads, keys, values = 1, 1024, 2, 128, 128
    hooks = dataclasses.replace(recurrence.Hooks(decay='scalar', delta=False, write_gate=True, skip=True),
                                turns=2)
    pairs = (jax.ShapeDtypeStruct((rows, length, heads, keys), jnp.bfloat16),) * 2
    rest = (jax.ShapeDtypeStruct((rows, length, heads, values), jnp.bfloat16),
            jax.ShapeDtypeStruct((rows, length, heads), jnp.float32),
            jax.ShapeDtypeStruct((rows, length, heads), jnp.float32),
            jax.ShapeDtypeStruct((heads,), jnp.float32))
    turn = jax.ShapeDtypeStruct((rows, length, heads, 2), jnp.float32)
    framed = 2 * math.prod(pairs[0].shape) * 2 + 6 * math.prod(turn.shape) * 4

    def temporaries(fn, *operands) -> int:
        """The compiled program's temporaries' bytes (`memory_analysis`)."""
        analysis = lowered_for(fn, operands, "sm90").compile().memory_analysis()
        assert analysis is not None
        return analysis.temp_size_in_bytes

    def called(hooks):
        def call(query, key, value, g, beta, skip, *turn):
            operands = Chunk(query, key, value, log_decay=g, beta=beta, skip=skip,
                             turn=turn[0] if turn else None)
            return chunk_on('mosaic_gpu', hooks, operands)[0]
        return call

    whole = temporaries(called(hooks), *pairs, *rest, turn)
    unturned = temporaries(called(dataclasses.replace(hooks, turns=0)), *pairs, *rest)
    assert whole <= unturned + framed
