"""A training step compiled ahead of time for GPUs this CPU host does not have (`dew.training.ahead`).

The step is compiled by XLA's CUDA backend for compile-only GPUs (the `kernel-compile` extra), the
step `fit` would compile on them, and is held to the step the same trainer compiles on this host's
eight CPU devices: the bytes each device reads and writes are the state's and the batch's shares
under one placement, whichever backend compiles it. What it costs and its executable are XLA's own;
a GPU loads and runs the saved executable (TL-*-compile-ahead-881)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from model_support import TINY_DECODER
from token_support import token_rows

from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer

VOCAB, SEQ_LEN, BATCH = 96, 16, 8
TINY_SHARD = 256


def fine_tune(mesh) -> Trainer:
    model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, num_kv_heads=2)
    return Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0), mesh=mesh,
                   layout=Layout(min_shard=TINY_SHARD), split=1)


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("spec", [MeshSpec(fsdp=8), MeshSpec(), MeshSpec(fsdp=4, tensor=2)],
                         ids=["fsdp8", "data8", "fsdp4_tensor2"])
def test_a_step_compiled_ahead_for_eight_gpus_reads_and_writes_what_the_run_on_eight_devices_does(spec):
    """Compiled ahead for eight compile-only H100s, the step reads and writes on each device the bytes
    the same trainer's step reads and writes on eight CPU devices, under each layout: its share of
    the state, whole or split, and of the batch. XLA states what it costs and every byte count of its
    memory, and its peak holds its arguments, its outputs that alias none, and its temporaries."""
    from dew.training.ahead import compile_ahead, compile_only_gpus

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    ahead = compile_ahead(fine_tune(spec.build(compile_only_gpus("sm90", 8))), batch)
    run = fine_tune(spec.build())
    state, _, _ = run.place()
    run.compile(state, batch)
    assert run.executable is not None
    here = run.executable.memory_analysis()
    assert (ahead.argument_bytes, ahead.output_bytes, ahead.alias_bytes) == (
        here.argument_size_in_bytes, here.output_size_in_bytes, here.alias_size_in_bytes)
    assert (ahead.generation, ahead.devices, ahead.mesh) == ("sm90", 8, dict(run.device_mesh.shape))
    assert ahead.executable and ahead.temp_bytes > 0
    assert ahead.flops is not None and ahead.flops.hardware > 0
    beside = ahead.output_bytes - ahead.alias_bytes + ahead.temp_bytes
    assert ahead.peak_bytes == ahead.argument_bytes + beside


def test_a_step_compiled_ahead_saves_and_loads_whole_and_runs_only_on_its_own_gpus(tmp_path):
    """A step saved is the step loaded, executable and what XLA stated of it; this CPU host, which is
    not the topology the step was compiled for, refuses to load it."""
    from dew.training.ahead import AheadOfTime, compile_ahead, compile_only_gpus

    ahead = compile_ahead(fine_tune(MeshSpec(fsdp=2).build(compile_only_gpus("sm80", 2))),
                          token_rows(VOCAB, BATCH, SEQ_LEN + 1))
    ahead.save(tmp_path / "step.dew-aot")
    loaded = AheadOfTime.load(tmp_path / "step.dew-aot")
    assert loaded == ahead
    with pytest.raises(ValueError):
        loaded.loaded()


@pytest.mark.mesh(devices=8)
def test_compiling_ahead_takes_a_mesh_of_compile_only_gpus():
    """A trainer whose mesh is this host's own devices runs its step here; compiling it ahead is
    refused, as it compiles for no other topology."""
    from dew.training.ahead import compile_ahead

    with pytest.raises(ValueError):
        compile_ahead(fine_tune(MeshSpec(fsdp=8).build()), token_rows(VOCAB, BATCH, SEQ_LEN + 1))


@pytest.mark.gpu(reason="loads and runs a step compiled ahead on this host's own GPU")
def test_a_step_compiled_ahead_and_saved_runs_on_its_gpu_in_the_run_s_place(tmp_path):
    """A step compiled ahead for one compile-only GPU of this GPU's generation, saved, and loaded
    again, runs on this GPU in the place of the run's own compile of it: handed the arguments the
    run's step is (the state the trainer places, the batch, the first part), it makes outputs of
    the same tree, shapes, dtypes and placements, the same step and microstep counters, a finite
    loss, and parameters it has updated. Its floats are not held to the run's own bit for bit: XLA
    tunes a GEMM's algorithm on the GPU it compiles on and takes its default with none attached."""
    from dew.nn.kernels.generation import chosen, device_generation
    from dew.state import apart
    from dew.training.ahead import AheadOfTime, compile_ahead, compile_only_gpus

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    # A step that reads its parameters whole, so its arguments are the placed state's own.
    with chosen(narrow_copies="off"):
        compile_ahead(fine_tune(MeshSpec().build(compile_only_gpus(device_generation(), 1))), batch).save(
            tmp_path / "step.dew-aot")
        run = fine_tune(MeshSpec().build(jax.devices()[:1]))
        state, _, _ = run.place()
        run.compile(state, batch)
    ahead = AheadOfTime.load(tmp_path / "step.dew-aot")
    assert run.executable is not None
    here = run.executable.memory_analysis()
    assert (ahead.argument_bytes, ahead.output_bytes, ahead.alias_bytes) == (
        here.argument_size_in_bytes, here.output_size_in_bytes, here.alias_size_in_bytes)
    shardings, _ = run.executable.input_shardings

    def arguments():
        # Each call takes the state it is handed, so each is handed its own placing of the same one.
        given, _, _ = run.place()
        return jax.device_put((*apart(dataclasses.replace(given, compute=None)), batch, np.int32(0)),
                              shardings)

    theirs, ours = run.executable(*arguments()), ahead.loaded()(*arguments())
    assert jax.tree.structure(ours) == jax.tree.structure(theirs)
    for left, right in zip(jax.tree.leaves(ours), jax.tree.leaves(theirs), strict=True):
        assert (left.shape, left.dtype, left.sharding) == (right.shape, right.dtype, right.sharding)
    moving, *outputs = ours
    np.testing.assert_array_equal(np.asarray(moving.step), np.asarray(theirs[0].step))
    np.testing.assert_array_equal(np.asarray(moving.microstep), np.asarray(theirs[0].microstep))
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(outputs)
               if jnp.issubdtype(leaf.dtype, jnp.floating))
    initial = jax.tree.leaves(state.variables["params"])
    assert any(not np.array_equal(np.asarray(new), np.asarray(old))
               for new, old in zip(jax.tree.leaves(moving.variables["params"]), initial, strict=True))
