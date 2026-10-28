"""A training step compiled ahead of time for GPUs this CPU host does not have (`dew.training.ahead`).

The step is compiled by XLA's CUDA backend for compile-only GPUs (the `kernel-compile` extra), the
step `fit` would compile on them, and is held to the step the same trainer compiles on this host's
eight CPU devices: the bytes each device reads and writes are the state's and the batch's shares
under one placement, whichever backend compiles it. What it costs and its executable are XLA's own;
a GPU loads and runs the saved executable (TL-*-compile-ahead-881)."""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from model_support import TINY_DECODER
from token_support import token_rows

from dew.objectives.lm import LMObjective
from dew.optim import OptimConfig
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer

VOCAB, SEQ_LEN, BATCH = 96, 16, 8
TINY_SHARD = 256


def fine_tune(mesh, *, ahead=None, width: int | None = None) -> Trainer:
    fields = TINY_DECODER if width is None else {**TINY_DECODER, "emb_features": width}
    model = models.build("causal_transformer", vocab_size=VOCAB, **fields, num_kv_heads=2)
    return Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0), mesh=mesh,
                   layout=Layout(min_shard=TINY_SHARD), split=1, ahead=ahead)


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("spec", [MeshSpec(fsdp=8), MeshSpec(), MeshSpec(fsdp=4, tensor=2)],
                         ids=["fsdp8", "data8", "fsdp4_tensor2"])
def test_a_step_compiled_ahead_for_eight_gpus_reads_and_writes_what_the_run_on_eight_devices_does(spec):
    """Compiled ahead for eight compile-only H100s, the step reads and writes on each device the bytes
    the same trainer's step reads and writes on eight CPU devices, under each layout: its share of
    the state, whole or split, and of the batch. XLA states what it costs and every byte count of its
    memory, and its peak holds its arguments, its outputs that alias none, and its temporaries."""
    from dew.training.ahead import compile_ahead, compile_only_gpus, step_batch

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    ahead = compile_ahead(fine_tune(spec.build(compile_only_gpus("sm90", 8))), batch)
    run = fine_tune(spec.build())
    state, _, _ = run.place()
    run.compile(state, step_batch(run, batch))
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
def test_a_run_given_its_step_compiled_ahead_runs_it_and_trains_as_one_that_compiled_it(tmp_path):
    """Compiled ahead over this host's own eight devices and saved, the step runs as a fresh run's first
    step in place of its compile (`Trainer(ahead=...)`): the run trains to the parameters the same run
    that compiled its own step trains to, bit for bit, one program having run both. A step compiled
    ahead for a wider model is another program, refused before the run trains (row 1915)."""
    from dew.training.ahead import AheadOfTime, compile_ahead

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    compile_ahead(fine_tune(MeshSpec(fsdp=8).build()), batch).save(tmp_path / "step.dew-aot")
    compile_ahead(fine_tune(MeshSpec(fsdp=8).build(), width=48), batch).save(tmp_path / "wide.dew-aot")

    def trained(ahead):
        trainer = fine_tune(MeshSpec(fsdp=8).build(), ahead=ahead)
        return trainer.fit(batch, steps=2, log_every=1).variables["params"]

    taken, compiled = trained(AheadOfTime.load(tmp_path / "step.dew-aot")), trained(None)
    for ours, theirs in zip(jax.tree.leaves(taken), jax.tree.leaves(compiled), strict=True):
        np.testing.assert_array_equal(np.asarray(ours), np.asarray(theirs))
    with pytest.raises(ValueError):
        trained(AheadOfTime.load(tmp_path / "wide.dew-aot"))


def test_a_configured_run_compiles_its_step_ahead_for_gpus_it_lacks_and_trains_nothing(tmp_path):
    """`dew train` with `trainer.compile_ahead` set compiles the run's first step for one compile-only
    A100, writes it where it names, and ends with status 0, the step loadable and what it costs
    stated; it writes no checkpoint or tracking, as it trains nothing (row 1915)."""
    from test_run_options import lm

    from dew.cli.main import main
    from dew.training.ahead import AheadOfTime, CompiledAhead

    record, runs, path = tmp_path / "record", tmp_path / "runs", tmp_path / "step.dew-aot"
    lm(tmp_path, drawn=False).save(str(record))
    target = {"generation": "sm80", "devices": 1, "path": str(path)}
    settings = {"trainer.name": "ahead", "trainer.checkpoint_dir": str(runs), "trainer.steps": "2",
                "trainer.compile_ahead": json.dumps(target)}
    with pytest.raises(CompiledAhead) as ended:
        main(["train", str(record / "run.json"),
              *(flag for name, value in settings.items() for flag in ("--set", f"{name}={value}"))])
    assert ended.value.code == 0
    saved = AheadOfTime.load(path)
    assert (saved.generation, saved.devices, saved.platform) == ("sm80", 1, "gpu") and saved.peak_bytes > 0
    assert not runs.exists()


ACCUMULATION_CELLS = [("sm80", "bfloat16"), ("sm90", "float32")]
"""Each generation and each compute dtype once (testing.md's pairwise settings): a bfloat16 step reads
its parameters through narrow copies on a GPU (`dew.training.narrow`), a float32 step reads them whole."""

WIDE_MLP = {**TINY_DECODER, "emb_features": 64, "mlp_features": 2048}
"""A decoder whose parameters, 3.3 MB in float32, dwarf what a row of 16 tokens keeps for its backward:
each buffer of a value a parameter is then larger than one microbatch's activations."""


@pytest.mark.parametrize(("generation", "dtype"), ACCUMULATION_CELLS)
def test_an_accumulating_step_holds_its_gradient_sum_and_no_other_buffer_a_parameter(generation, dtype):
    """Row 1900: a step that pools four microbatches into a commit, compiled ahead for a GPU, holds
    beyond the step that commits each one its window's gradient sum, a value a parameter in the
    parameter's dtype, among its arguments, and no more temporaries than one microbatch's activations,
    nor more at its peak, outputs included, as XLA's memory analysis of both programs states them. A
    resident copy of the parameters, a second gradient buffer or a narrow copy kept beside the sum,
    each two bytes a parameter or more, exceeds that: row 1661's 14 GB on a 4B run surfaced only on a
    GPU, and this finds its kind on a CPU host."""
    from dew.training.ahead import compile_ahead, compile_only_gpus

    model = models.build("causal_transformer", vocab_size=VOCAB, **WIDE_MLP, num_kv_heads=2, dtype=dtype)

    def ahead(accumulation: int, rows: int):
        trainer = Trainer(LMObjective(model, SEQ_LEN), OptimConfig(optimizer="adamw", learning_rate=1e-3),
                          key=jax.random.key(0), mesh=MeshSpec().build(compile_only_gpus(generation, 1)),
                          layout=Layout(min_shard=TINY_SHARD), split=1, accumulation=accumulation)
        trainer.validate(steps=8)
        return compile_ahead(trainer, token_rows(VOCAB, rows, SEQ_LEN + 1))

    one, window, two_rows = ahead(1, 1), ahead(4, 1), ahead(1, 2)
    shapes = jax.eval_shape(model.init, jax.random.key(0), jnp.zeros((1, SEQ_LEN), jnp.int32))["params"]
    gradient = sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(shapes))
    held = 4 * token_rows(VOCAB, 1, SEQ_LEN + 1)["text"].nbytes
    activations = two_rows.temp_bytes - one.temp_bytes
    # The cell can tell: half the sum, a narrow copy's two bytes a parameter, is more than it allows.
    assert 0 < activations < gradient / 2
    # The window's arguments: its gradient sum, and at most a token row a microbatch it holds to replay.
    assert window.argument_bytes - one.argument_bytes <= gradient + held
    assert window.temp_bytes - one.temp_bytes <= activations
    # Its whole peak, outputs that alias no argument among it.
    assert window.peak_bytes - one.peak_bytes <= gradient + held + activations


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
    from dew.training.ahead import AheadOfTime, compile_ahead, compile_only_gpus, step_batch

    # A step that reads its parameters whole, so its arguments are the placed state's own.
    with chosen(narrow_copies="off"):
        compile_ahead(fine_tune(MeshSpec().build(compile_only_gpus(device_generation(), 1))),
                      token_rows(VOCAB, BATCH, SEQ_LEN + 1)).save(tmp_path / "step.dew-aot")
        run = fine_tune(MeshSpec().build(jax.devices()[:1]))
        batch = step_batch(run, token_rows(VOCAB, BATCH, SEQ_LEN + 1))
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
