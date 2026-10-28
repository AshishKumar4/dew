"""A training step compiled ahead of time for GPUs this CPU host does not have (`dew.training.ahead`).

The step is compiled by XLA's CUDA backend for compile-only GPUs (the `kernel-compile` extra), the
step `fit` would compile on them, and is held to the step the same trainer compiles on this host's
eight CPU devices: the bytes each device reads and writes are the state's and the batch's shares
under one placement, whichever backend compiles it. What it costs and its executable are XLA's own;
a GPU loads and runs the saved executable (TL-*-compile-ahead-881)."""

import dataclasses
import logging

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from model_support import TINY_DECODER
from token_support import token_rows

from dew._records import listed, record
from dew.objectives.lm import LMObjective
from dew.optim import OptimConfig
from dew.registry import models
from dew.training import Layout, MeshSpec, Trainer

VOCAB, SEQ_LEN, BATCH = 96, 16, 8
TINY_SHARD = 256


def fine_tune(mesh, *, ahead=None, width: int | None = None, xla: bool = False, links=None):
    fields = {**TINY_DECODER, **({} if width is None else {"emb_features": width}),
              **({"attention_impl": "xla"} if xla else {})}
    model = models.build("causal_transformer", vocab_size=VOCAB, **fields, num_kv_heads=2)
    return Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0), mesh=mesh,
                   layout=Layout(min_shard=TINY_SHARD), split=1, ahead=ahead, links=links)


def test_compiling_ahead_plans_once_and_saves_the_selected_program(tmp_path, monkeypatch, caplog):
    """A compile-only GPU with stated memory takes the run's own fit search, whose one fitting
    program is the saved executable, not another compile outside the report. The report records
    that compile's time and memory, and planning creates no device state or runner to execute."""
    from dew.training.ahead import AheadOfTime, compile_ahead, compile_only_gpus

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "rungs"))
    trainer = fine_tune(MeshSpec().build(compile_only_gpus("sm90", 1)))
    original_state_mesh = trainer.state_mesh
    with caplog.at_level(logging.DEBUG, logger="jax._src.dispatch"):
        ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1), capacity=2**30)
    assert tuple(trainer.device_mesh.devices.flat) == compile_only_gpus("sm90", 1, capacity=2**30)
    assert trainer.state_mesh is trainer.device_mesh and trainer.state_mesh is not original_state_mesh
    step_compiles = [message for message in caplog.messages
                     if message.startswith("Finished XLA compilation of jit(step)")]
    assert len(step_compiles) == 1
    plan = record(ahead.plan, "plan")
    (probe,) = listed(plan["probes"], "probes", read=record, what="probes")
    assert probe["fits"] is True and probe["temp"] == ahead.temp_bytes
    assert probe["argument"] == ahead.argument_bytes
    assert ahead.capacity == 2**30
    from dew.training.memory import device_memory

    stated = device_memory(trainer.device_mesh.devices.flat[0], ahead.capacity, ahead.argument_bytes)
    assert stated["pool_bytes"] == ahead.capacity
    assert stated["largest_free_block_bytes"] == ahead.capacity - ahead.argument_bytes
    seconds = probe["compile_seconds"]
    assert isinstance(seconds, float) and ahead.planned_seconds is not None
    assert ahead.planned_seconds >= seconds > 0
    destination = tmp_path / "planned.dew-aot"
    ahead.save(destination)
    saved = AheadOfTime.load(destination)
    assert saved.plan == ahead.plan and saved.program == ahead.program


def test_an_ahead_plan_that_fits_no_device_reports_every_refused_program(tmp_path, monkeypatch):
    """An absent GPU with less memory than even the arguments gets no executable to run. Its
    refusal keeps every candidate's compile evidence, so the offline gate can check refusals too."""
    from dew.training.ahead import compile_ahead, compile_only_gpus
    from dew.training.planning import PlanRefused

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    trainer = fine_tune(MeshSpec().build(compile_only_gpus("sm90", 1)))
    with pytest.raises(PlanRefused) as refused:
        compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1), capacity=1024)
    probes = refused.value.probes
    assert probes and all(not probe.fits for probe in probes)
    assert all(probe.compile_seconds > 0 for probe in probes)
    assert any(probe.headroom is not None and probe.headroom < 0 for probe in probes)


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
    links = None
    if spec.tensor == 2:
        from dew.nn.sharding import Link

        links = {"tensor": Link(None, None)}
    absent = fine_tune(spec.build(compile_only_gpus("sm90", 8)), links=links)
    ahead = compile_ahead(absent, batch)
    run = fine_tune(spec.build())
    state, _, _ = run.place()
    run.compile(state, step_batch(run, batch))
    assert run.executable is not None
    here = run.executable.memory_analysis()
    assert here is not None
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
    refused, as XLA's CPU executable, loaded again, finds none of its fused functions."""
    from dew.training.ahead import compile_ahead

    with pytest.raises(ValueError):
        compile_ahead(fine_tune(MeshSpec(fsdp=8).build()), token_rows(VOCAB, BATCH, SEQ_LEN + 1))


def test_a_step_compiled_ahead_is_one_program_and_a_run_of_another_refuses_it(tmp_path):
    """Two trainers of one config compile ahead one program, the digest a run lowers its first step
    to (`AheadOfTime.program`); a wider model's is another. A run handed another program's step
    refuses it before it trains, and refuses it again on its next fit, rather than compiling its own
    (row 1915)."""
    from dew.training.ahead import AheadOfTime, compile_ahead, compile_only_gpus

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)

    def ahead(width=None):
        return compile_ahead(fine_tune(MeshSpec().build(compile_only_gpus("sm80", 1)), width=width), batch)

    first, again, wide = ahead(), ahead(), ahead(width=48)
    assert first.program == again.program != wide.program
    wide.save(tmp_path / "wide.dew-aot")
    run = fine_tune(MeshSpec().build(jax.devices()[:1]), ahead=AheadOfTime.load(tmp_path / "wide.dew-aot"))
    for _ in range(2):
        with pytest.raises(ValueError):
            run.fit(batch, steps=1, log_every=1)


def test_a_configured_run_compiles_its_step_ahead_for_gpus_it_lacks_and_trains_nothing(tmp_path):
    """A run parsed with `--trainer.compile-ahead.generation`, `.devices` and `.path` (`RunConfig.cli`,
    its default the run's own config) compiles its first step for one compile-only A100, writes it
    where it names, and ends with status 0, the step loadable and what it costs stated; it writes no
    checkpoint or tracking, as it trains nothing (row 1915)."""
    from test_run_options import lm

    from dew.config.lm import LMRunConfig
    from dew.training.ahead import AheadOfTime, CompiledAhead

    runs, path = tmp_path / "runs", tmp_path / "step.dew-aot"
    default = lm(tmp_path, drawn=False)
    assert isinstance(default, LMRunConfig)
    run = LMRunConfig.cli(["--trainer.name", "ahead", "--trainer.checkpoint-dir", str(runs),
                           "--trainer.steps", "2", "--trainer.compile-ahead.generation", "sm80",
                           "--trainer.compile-ahead.devices", "1", "--trainer.compile-ahead.path", str(path),
                           "--trainer.compile-ahead.capacity-gib", "1"],
                          default=default)
    with pytest.raises(CompiledAhead) as ended:
        run.run()
    assert ended.value.code == 0
    saved = AheadOfTime.load(path)
    assert (saved.generation, saved.devices) == ("sm80", 1) and saved.peak_bytes > 0
    assert saved.capacity == 2**30 and saved.plan is not None
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
    assert here is not None
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


@pytest.mark.gpu(reason="runs a step compiled ahead as a run's first step on this host's own GPU")
def test_a_run_handed_its_step_compiled_ahead_trains_from_it_on_its_gpu(tmp_path):
    """A step compiled ahead for one compile-only GPU of this GPU's generation, saved and handed to
    the run (`Trainer(ahead=...)`), is the program the run lowers its first step to, and the run
    trains from it, its plan's first compile the step compiled ahead: its steps advance and its
    parameters move (row 1915). A wider model's step for this GPU is another program, refused. The
    model's attention takes XLA's path, which a step compiled with no GPU attached traces too."""
    from dew.nn.kernels.generation import chosen, device_generation
    from dew.training.ahead import AheadOfTime, compile_ahead, compile_only_gpus

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    with chosen(narrow_copies="off"):
        for name, width in (("step", None), ("wide", 48)):
            compile_ahead(fine_tune(MeshSpec().build(compile_only_gpus(device_generation(), 1)), xla=True,
                                    width=width), batch).save(tmp_path / f"{name}.dew-aot")
        here = MeshSpec().build(jax.devices()[:1])
        run = fine_tune(here, ahead=AheadOfTime.load(tmp_path / "step.dew-aot"), xla=True)
        initial = jax.tree.leaves(run.initial_state().variables["params"])
        state = run.fit(batch, steps=2, log_every=1)
        other = fine_tune(here, ahead=AheadOfTime.load(tmp_path / "wide.dew-aot"), xla=True)
        with pytest.raises(ValueError):
            other.fit(batch, steps=1, log_every=1)
    first, *_ = listed(record(run.plan, "plan")["probes"], "probes", read=record, what="probes")
    assert int(state.step) == 2 and first["ahead"] is True
    assert any(not np.array_equal(np.asarray(new), np.asarray(old))
               for new, old in zip(jax.tree.leaves(state.variables["params"]), initial, strict=True))
