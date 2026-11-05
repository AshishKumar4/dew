"""A training step compiled ahead of time for GPUs this CPU host does not have (`dew.training.ahead`).

The step is compiled by XLA's CUDA backend for compile-only GPUs (the `kernel-compile` extra), the
step `fit` would compile on them, and is held to the step the same trainer compiles on this host's
eight CPU devices: the bytes each device reads and writes are the state's and the batch's shares
under one placement, whichever backend compiles it. What it costs and its executable are XLA's own;
a GPU loads and runs the saved executable (TL-*-compile-ahead-881)."""

import dataclasses
import itertools
import json
import logging
import os

import lane_environment
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from lane_environment import without_flags
from model_support import TINY_DECODER
from token_support import token_rows

from dew._records import listed, record
from dew.nn.sharding import Link
from dew.objectives.lm import LMObjective
from dew.optim import OptimConfig
from dew.registry import models
from dew.telemetry.compiles import Compiles
from dew.training import Layout, MeshSpec, Trainer

VOCAB, SEQ_LEN, BATCH = 96, 16, 8
TINY_SHARD = 256


def fine_tune(mesh, *, ahead=None, width: int | None = None, xla: bool = False, split=1, remat=None,
              inspection=None):
    fields = {**TINY_DECODER, **({} if width is None else {"emb_features": width}),
              **({"attention_impl": "xla"} if xla else {}), **({} if remat is None else {"remat": remat})}
    model = models.build("causal_transformer", vocab_size=VOCAB, **fields, num_kv_heads=2)
    return Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0), mesh=mesh,
                   layout=Layout(min_shard=TINY_SHARD), split=split, ahead=ahead, inspection=inspection)


def test_compiling_ahead_plans_once_and_saves_the_selected_program(tmp_path, monkeypatch):
    """A compile-only GPU with stated memory takes the run's own fit search, whose one fitting
    program is the saved executable, not another compile outside the report. The report records
    that compile's time and memory, and planning creates no device state or runner to execute."""
    from dew.training.ahead import AheadOfTime, compile_ahead, compile_only_gpus

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "rungs"))
    trainer = fine_tune(MeshSpec().build(compile_only_gpus("sm90", 1)))
    original_state_mesh = trainer.state_mesh
    with Compiles("step") as compiles:
        ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1), capacity=2**30)
    assert tuple(trainer.device_mesh.devices.flat) == compile_only_gpus("sm90", 1, 2**30)
    assert trainer.state_mesh is trainer.device_mesh and trainer.state_mesh is not original_state_mesh
    assert compiles.count == 1
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


NEEDING_MICROBATCHES = 700_000
"""Bytes a compile-only GPU states for `fine_tune`'s step to need several microbatches."""


@pytest.fixture(scope="module")
def admitted(tmp_path_factory):
    """`fine_tune`'s step compiled ahead on GPUs of `NEEDING_MICROBATCHES`, its plan kept beside the
    compilation cache as every search keeps its own (`DEW_RUNG_RECORDS`): the records' directory, the
    trainer and its `AheadOfTime`, admitted once for the module."""
    from dew.training.ahead import compile_ahead, compile_only_gpus

    records = tmp_path_factory.mktemp("rungs")
    with pytest.MonkeyPatch.context() as patch:
        # The module's own data and Hub caches, as each test's (`own_settings`), its rung records `records`.
        lane_environment.per_test_settings(patch, tmp_path_factory.mktemp)
        patch.setenv("DEW_RUNG_RECORDS", str(records))
        trainer = fine_tune(MeshSpec().build(compile_only_gpus("sm90", 1)), split=None)
        ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1), capacity=NEEDING_MICROBATCHES)
    return records, trainer, ahead


@pytest.mark.shared_setup("admitted")
def test_compiling_ahead_on_gpus_its_plan_needs_microbatches_for_admits_it(admitted):
    """A step compiled ahead on GPUs whose memory its plan needs several microbatches for searches for its
    plan, at several microbatches, and saves it, the run left at its plan, its rung its last probe's."""
    _, trainer, ahead = admitted
    plan = record(ahead.plan, "plan")
    probes = listed(plan["probes"], "probes", read=record, what="probes")
    assert plan["split"] > 1 and len(probes) > 1 and probes[-1]["fits"] is True
    assert trainer.plan == ahead.plan and plan["remat"] == probes[-1]["remat"]


@pytest.mark.shared_setup("admitted")
def test_compiling_ahead_with_its_frontier_lists_each_rung_above_its_plan_at_fewer_microbatches(
        admitted, monkeypatch):
    """With its frontier asked, a step compiled ahead on GPUs whose memory its plan needs several
    microbatches for reports, past the plan it takes and saves, each rung above it that fits at fewer
    microbatches than every plan before it, at the fewest that fit there (rows 2115, 3108, 3110): the plans
    a run's speed is chosen among on its GPUs (tools/benchmark_step.py times each), the run naming the one it
    takes. A rung at as many microbatches as one below it is none of them. The run is left at its plan. Its
    search starts at the plan admitted before it (`admitted`), kept beside the compilation cache, as a run
    after another starts at its recorded plan: that plan's compile alone, the frontier read above it."""
    from dew.training.ahead import compile_ahead, compile_only_gpus

    records, _, earlier = admitted
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(records))
    trainer = fine_tune(MeshSpec().build(compile_only_gpus("sm90", 1)), split=None)
    ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1), capacity=NEEDING_MICROBATCHES,
                          frontier=True)
    plan = record(ahead.plan, "plan")
    probes = listed(plan["probes"], "probes", read=record, what="probes")
    # A sibling that offloads what the ladder's top recomputes sits at that top's split (row 1932).
    from dew.nn.backbones.decoder_block import REMAT_POLICIES, remat_record

    offloaded = remat_record(REMAT_POLICIES["minimal_offloaded"])
    frontier = [entry for entry in listed(plan["frontier"], "frontier", read=record, what="frontier")
                if entry["remat"] != offloaded]
    rung = ("split", "remat", "head_tile", "head_fused", "xla_defaults", "narrow")
    admitted_plan = record(earlier.plan, "admitted plan")
    assert {name: plan[name] for name in rung} == {name: admitted_plan[name] for name in rung}
    assert len(probes) == 1
    splits = [plan["split"], *(entry["split"] for entry in frontier)]
    assert len(frontier) >= 2 and all(entry["fits"] is True for entry in frontier)
    assert all(isinstance(more, int) and isinstance(fewer, int) and fewer < more
               for more, fewer in itertools.pairwise(splits))
    rungs = [json.dumps({name: entry[name] for name in ("remat", "head_tile", "head_fused")}, sort_keys=True)
             for entry in (plan, *frontier)]
    assert len(set(rungs)) == len(rungs)
    assert trainer.plan == ahead.plan and plan["remat"] == probes[-1]["remat"]


@pytest.mark.parametrize("capacity", [2**30, 520_000], ids=["one microbatch", "several"])
def test_a_frontier_at_the_ladders_top_lists_its_offloaded_sibling_at_its_last_fit(tmp_path, monkeypatch,
                                                                                    capacity):
    """A step whose plan is the remat ladder's top ('full') lists, past it, the sibling that keeps what
    'minimal' saves in pinned host memory in place of recomputing it ('minimal_offloaded'): as many device
    bytes as 'full', so no fit rung, but another speed, which the GPUs' timing chooses (row 1932). It is
    compiled at the last configuration at 'full' that fit, its head policy put back, whatever head policies
    the frontier then tried at fewer microbatches; and the run, which times nothing, is left at its plan.
    At 520 kB the plan at 'full' takes four microbatches and the head policies tried past it fit at no fewer
    (a sweep at 83bc7370bc: from 600 kB to 700 kB the sibling fits at no plan's split on a device with no
    host memory of its own, and past 850 kB the plan takes one)."""
    from dew.nn.backbones.decoder_block import REMAT_POLICIES, remat_record
    from dew.training.ahead import compile_ahead, compile_only_gpus
    from dew.training.memory import head_fused_of, head_tile_of, recompute_record

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "rungs"))
    trainer = fine_tune(MeshSpec().build(compile_only_gpus("sm90", 1)), split=None, remat="full")
    ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1), capacity=capacity, frontier=True)
    plan = record(ahead.plan, "plan")
    frontier = listed(plan["frontier"], "frontier", read=record, what="frontier")
    full, offloaded = (remat_record(REMAT_POLICIES[name]) for name in ("full", "minimal_offloaded"))
    *rungs, sibling = frontier
    last = [entry for entry in (plan, *rungs) if entry["remat"] == full][-1]
    assert plan["remat"] == full
    assert sibling["remat"] == offloaded and sibling["fits"] is True
    assert (sibling["split"], sibling["head_tile"], sibling["head_fused"]) == (
        last["split"], last["head_tile"], last["head_fused"])
    assert recompute_record(trainer.objective) == plan["remat"]
    tile = head_tile_of(trainer.objective)
    assert ([*tile] if tile else None, head_fused_of(trainer.objective)) == (plan["head_tile"],
                                                                             plan["head_fused"])


def test_a_step_compiled_for_gpus_none_attached_skips_the_autotuning_nothing_can_time(monkeypatch):
    """A step compiled for compile-only GPUs takes autotuning off (`deviceless_options`), its programs' bytes
    the same: XLA's config assigner gives each matmul its default config where no device is attached, while
    its autotuner would still compile every candidate it cannot time. A run's own XLA_FLAGS setting, and a
    mesh of devices that run, keep the options as given."""
    from dew.nn.kernels.options import compiler_options
    from dew.training.ahead import compile_only_gpus
    from dew.training.memory import deviceless_options
    from dew.training.planning import FitSearch

    monkeypatch.setenv("XLA_FLAGS", without_flags(os.environ.get("XLA_FLAGS", ""),
                                                  "--xla_gpu_autotune_level"))
    deviceless = MeshSpec().build(compile_only_gpus("sm90", 1))
    assert deviceless_options(deviceless, None) == {"xla_gpu_autotune_level": 0}
    assert deviceless_options(deviceless, {"xla_gpu_enable_triton_gemm": False}) == {
        "xla_gpu_enable_triton_gemm": False, "xla_gpu_autotune_level": 0}
    running = MeshSpec().build(jax.devices()[:1])
    assert deviceless_options(running, {"xla_gpu_enable_triton_gemm": False}) == {
        "xla_gpu_enable_triton_gemm": False}
    # The fit search alone applies the policy, to the step's options and its towers' alike (row 3506): a
    # step whose options are its towers' own compiles as the half rung of XLA's defaults does, one program.
    objective = fine_tune(deviceless, split=None).objective
    search = FitSearch(lower=None, options=lambda candidate: compiler_options(*objective.towers()) or None,
                       objective=objective, mesh=deviceless, xla_defaults=False, narrow=False, base=None,
                       resumed=None, capacity=2**30)
    assert search.compiled_options(None) == search.required == deviceless_options(
        deviceless, compiler_options(*objective.towers()) or None)
    assert dataclasses.replace(search, xla_defaults=True).compiled_options(None) == search.required
    monkeypatch.setenv("XLA_FLAGS", f"{os.environ['XLA_FLAGS']} --xla_gpu_autotune_level=2")
    assert deviceless_options(deviceless, None) is None


def test_compiling_ahead_with_its_frontier_on_gpus_of_no_stated_memory_is_refused(tmp_path, monkeypatch):
    """A frontier asked of compile-only GPUs that state no memory is refused before anything compiles
    (row 3107): their memory unknown, every rung would fit there, a frontier of the whole ladder."""
    from dew.training.ahead import compile_ahead, compile_only_gpus

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path / "rungs"))
    trainer = fine_tune(MeshSpec().build(compile_only_gpus("sm90", 1)))
    with Compiles("step") as compiles, pytest.raises(ValueError):
        compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1), frontier=True)
    assert compiles.count == 0


def test_a_run_compiled_ahead_on_gpus_given_no_capacity_hears_they_state_no_memory(tmp_path, monkeypatch,
                                                                                    caplog):
    """A run that sets its remat, compiled ahead on GPUs given no capacity, whose memory nothing states,
    has no rung judged ("unstated", one answer for the walk and the footprint, `memory.judged`; rows 3667,
    3668), compiles at its own rung alone, and hears once that its devices state no memory, not that its
    layout cannot be judged ("unknown")."""
    from dew.nn.backbones.decoder_block import REMAT_POLICIES
    from dew.training.ahead import compile_ahead, compile_only_gpus

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    trainer = fine_tune(MeshSpec().build(compile_only_gpus("sm90", 1)), remat=REMAT_POLICIES["full"])
    with caplog.at_level(logging.WARNING, logger="dew"), Compiles("step") as compiles:
        ahead = compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1))
    plan = record(ahead.plan, "plan")
    derived = listed(plan["derived"], "derived", read=record, what="derived rungs")
    assert compiles.count == 1 and [entry["label"] for entry in derived] == ["unstated"]
    assert len([entry for entry in caplog.records if entry.name == "dew.training.planning"]) == 1


def test_an_ahead_plan_that_fits_no_device_reports_every_refused_program(tmp_path, monkeypatch):
    """An absent GPU with less memory than even the arguments gets no executable to run. Its
    refusal keeps every candidate's compile evidence, so the offline gate can check refusals too. The
    step starts at its ladder's last remat rung and its split is pinned, so the refusal comes after the
    few rungs left rather than after every policy's compile."""
    from dew.nn.backbones.decoder_block import REMAT_POLICIES
    from dew.training.ahead import compile_ahead, compile_only_gpus
    from dew.training.planning import PlanRefused

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    trainer = fine_tune(MeshSpec().build(compile_only_gpus("sm90", 1)), remat=REMAT_POLICIES["full"])
    with pytest.raises(PlanRefused) as refused:
        compile_ahead(trainer, token_rows(VOCAB, BATCH, SEQ_LEN + 1), capacity=1024)
    probes = refused.value.probes
    assert probes and all(not probe.fits for probe in probes)
    assert all(probe.compile_seconds > 0 for probe in probes)
    assert any(probe.headroom is not None and probe.headroom < 0 for probe in probes)


def test_a_compile_only_gpu_holds_the_accumulator_a_step_builds_once(tmp_path, monkeypatch):
    """On a compile-only GPU of stated memory, a step read in two microbatches, whose first call builds
    its window's gradient sum, needs room for that sum once: the stated pool holds the step's arguments,
    the sum among them, and beside them the outputs that alias none, the temporaries and the batches
    prefetched next to the step (the 2113 review: the sum was counted among the arguments and again
    beside them, so an offline plan could refuse a step that fits)."""
    from dew.distributed.batches import batch_shardings, prefetched_bytes
    from dew.training.ahead import compile_ahead, compile_only_gpus, step_batch
    from dew.training.memory import temporaries_factor

    monkeypatch.setenv("DEW_RUNG_RECORDS", str(tmp_path))
    capacity = 2**30
    trainer = fine_tune(MeshSpec().build(compile_only_gpus("sm90", 1)), split=2)
    rows = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    ahead = compile_ahead(trainer, rows, capacity=capacity)
    (probe,) = listed(record(ahead.plan, "plan")["probes"], "probes", read=record, what="probes")
    batch = step_batch(trainer, rows)
    prefetched = prefetched_bytes(batch, batch_shardings(trainer.device_mesh, batch))
    temporaries = ahead.temp_bytes * temporaries_factor(trainer.device_mesh, capacity)
    assert probe["fits"] is True
    assert probe["headroom"] == (capacity - ahead.argument_bytes - (ahead.output_bytes - ahead.alias_bytes)
                                 - prefetched - temporaries)


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
    absent = fine_tune(spec.build(compile_only_gpus("sm90", 8)))
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
    """A step saved is the step loaded, executable and what XLA stated of it, with the target it was built
    for (`BuildTarget`), the one rule every loader holds: this CPU host, which is not the topology the step
    was compiled for, refuses to load it, and a run reproducing it derives its launches for its GPU only on
    one of as many multiprocessors, as a balanced tail's grid past the GPU's residency can wait on a
    program that never starts (row 3606)."""
    from dew.training.ahead import AheadOfTime, compile_ahead, compile_only_gpus

    ahead = compile_ahead(fine_tune(MeshSpec(fsdp=2).build(compile_only_gpus("sm80", 2))),
                          token_rows(VOCAB, BATCH, SEQ_LEN + 1))
    ahead.save(tmp_path / "step.dew-aot")
    loaded = AheadOfTime.load(tmp_path / "step.dew-aot")
    assert loaded == ahead
    with pytest.raises(ValueError):
        loaded.loaded()
    assert loaded.derived_for(loaded.target) == loaded.target
    fewer = dataclasses.replace(loaded.target, multiprocessors=loaded.target.multiprocessors - 6)
    with pytest.raises(ValueError):
        loaded.derived_for(fewer)


GATED_DELTA_AHEAD = """
import sys
sys.path[:0] = [f"{sys.argv[1]}/src", f"{sys.argv[1]}/tests"]
import jax
import optax
from token_support import token_rows

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.objectives.lm import LMObjective
from dew.training import Layout, MeshSpec, Trainer
from dew.training.ahead import compile_ahead, compile_only_gpus

mixer = GatedDeltaNetMixer(linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=64,
                           linear_value_head_dim=64)
model = CausalTransformer(vocab_size=64, emb_features=128, num_layers=1, num_heads=2, head_dim=64,
                          mlp_features=256, max_seq_len=128, dtype="bfloat16", mixer=mixer)
trainer = Trainer(LMObjective(model, 128), optax.adam(1e-3), key=jax.random.key(0),
                  mesh=MeshSpec().build(compile_only_gpus("sm80", 1)), layout=Layout(min_shard=256))
compile_ahead(trainer, token_rows(64, 2, 129), capacity=2**34)
"""
"""A bf16 gated-delta decoder's step compiled ahead for a compile-only A100, in a process of its own: its
checkout is its argument."""


def test_a_gated_delta_steps_mosaic_kernels_compile_ahead_for_the_gpu_it_targets():
    """A bf16 gated-delta decoder's step compiled ahead for a compile-only A100 (`compile_only_gpus("sm80")`)
    lowers its Mosaic GPU kernels for the architecture its trace's abstract device states, the A100's 8.0
    (row 3885): a mesh's abstract device stated none, so Mosaic lowered for an H100's, and its
    `nvvm.elect.sync` failed sm80's LLVM instruction selection and aborted the process, every A100 prepare
    of a model with a Mosaic route among them. The compile runs in a process of its own, as Mosaic compiles
    for one capability a process (`kernel_compile.compile_mosaic_kernels`)."""
    import subprocess
    import sys
    from pathlib import Path

    root = str(Path(__file__).resolve().parents[1])
    # Its compile took 45 to 75 s on armada; past ten minutes it is stuck, not slow.
    ran = subprocess.run([sys.executable, "-c", GATED_DELTA_AHEAD, root], capture_output=True, text=True,
                         timeout=600, check=False)
    assert ran.returncode == 0, ran.stderr[-3000:]


@pytest.mark.parametrize("generation", ["sm90", "sm100", "sm120"])
def test_a_compile_only_gpu_assembles_mosaic_gpus_architecture_specific_code(generation):
    """A Mosaic GPU kernel compiled for a compile-only GPU of each generation from Hopper on, in a process
    that assembles for that generation alone (`compile_mosaic_kernels`): its PTX targets the
    architecture-specific variant (sm_90a, sm_100a, sm_120a), which the device's capability states, so it
    assembles. An sm120 stating no feature extension assembled for plain sm_120 and refused its own PTX
    ('.target sm_120a cannot be compiled for architecture sm_120')."""
    from fla_gdn_support import KernelCase
    from kernel_compile import compile_mosaic_kernels, staged_copy

    kernel = staged_copy((64, 64), jnp.bfloat16)
    spec = jax.ShapeDtypeStruct((64, 64), jnp.bfloat16)
    record, = compile_mosaic_kernels([KernelCase("copy", lambda: (kernel, (spec,)), {})], generation)
    assert "error" not in record, record["error"]


@pytest.mark.mesh(devices=8)
def test_programs_saved_before_executables_were_one_type_are_refused_to_compile_again(tmp_path):
    """A file of programs compiled ahead whose programs are not executables of this Dew's (an `AheadPrograms`
    saved before row 3670 made `Executable` its programs' one form) is refused as it loads, asking for them
    compiled again (row 3671), never an AttributeError as it runs."""
    import pickle

    from dew.training.ahead import AheadPrograms, BuildTarget

    target = BuildTarget("sm90", 132, 1, "0", "0", (), "x86_64")
    path = tmp_path / "old.dew-aot"
    path.write_bytes(pickle.dumps(AheadPrograms(target, {"step": ("program", b"executable", ())})))
    with pytest.raises(ValueError):
        AheadPrograms.load(path)


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
    from dew.nn.kernels.generation import assumed
    from dew.training.ahead import AheadOfTime, compile_ahead, compile_only_gpus

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)

    def ahead(width=None):
        return compile_ahead(fine_tune(MeshSpec().build(compile_only_gpus("sm80", 1)), width=width), batch)

    first, again, wide = ahead(), ahead(), ahead(width=48)
    assert first.program == again.program != wide.program
    wide.save(tmp_path / "wide.dew-aot")
    run = fine_tune(MeshSpec().build(jax.devices()[:1]), ahead=AheadOfTime.load(tmp_path / "wide.dew-aot"))
    # On the GPU it was compiled for, which a CPU is not, its program is what refuses it (train 332's review).
    with assumed(wide.target):
        for _ in range(2):
            with pytest.raises(ValueError):
                run.fit(batch, steps=1, log_every=1)


def test_a_configured_run_compiles_its_step_ahead_for_gpus_it_lacks_and_trains_nothing(tmp_path):
    """A run parsed with `--trainer.compile-ahead.generation`, `.devices` and `.path` (`RunConfig.cli`,
    its default the run's own config) compiles its first step for one compile-only A100, writes it
    where it names, and ends with status 0, the step loadable and what it costs stated; it writes no
    checkpoint or tracking, as it trains nothing (row 1915)."""
    from run_kinds import lm

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


def test_a_configured_run_compiles_ahead_for_the_multiprocessors_it_states_and_loads_only_there(
        tmp_path, monkeypatch):
    """`--trainer.compile-ahead.multiprocessors` states the GPU a step is compiled ahead for within its
    generation (an H100 PCIe's 114 against the canonical H100's 132, rows 3606 and 3669): XLA compiles the
    step and its sampled step over compile-only GPUs of 114 (as they state), and the saved build target and
    the facts its launches derived for are that GPU's, so a GPU of 114 runs it and the canonical one is
    refused; the rung its plan recorded is that GPU's, which a second build for it reuses, apart from the
    canonical build's beside it in one record directory."""
    import dataclasses

    from run_kinds import lm

    from dew.config.lm import LMRunConfig
    from dew.nn.kernels.generation import gpu_target
    from dew.training.ahead import AheadOfTime, CompiledAhead

    records = tmp_path / "rungs"
    monkeypatch.setenv("DEW_RUNG_RECORDS", str(records))

    builds = iter(range(3))

    def compiled(*stated: str) -> AheadOfTime:
        root = tmp_path / f"run{next(builds)}"
        root.mkdir()
        path = root / "step.dew-aot"
        run = LMRunConfig.cli(["--trainer.name", "ahead", "--trainer.checkpoint-dir", str(root / "runs"),
                               "--trainer.steps", "2", "--trainer.compile-ahead.generation", "sm90",
                               "--trainer.compile-ahead.devices", "1",
                               "--trainer.compile-ahead.path", str(path),
                               "--trainer.compile-ahead.capacity-gib", "0.0007",
                               "--trainer.inspection.every", "1", *stated],
                              default=lm(root, drawn=False))
        with pytest.raises(CompiledAhead):
            run.run()
        return AheadOfTime.load(path)

    pcie = compiled("--trainer.compile-ahead.multiprocessors", "114")
    again = compiled("--trainer.compile-ahead.multiprocessors", "114")
    canonical = compiled()
    assert (pcie.built.multiprocessors, pcie.target.multiprocessors) == (114, 114)
    assert pcie.sampled is not None and again.built == pcie.built
    assert canonical.built.multiprocessors == gpu_target("sm90").multiprocessors
    held = dataclasses.replace(gpu_target("sm90"), device_kind="NVIDIA H100 PCIe", multiprocessors=114)
    assert pcie.derived_for(held) == pcie.target
    with pytest.raises(ValueError):
        pcie.derived_for(gpu_target("sm90"))
    assert len(list(records.rglob("*.json"))) == 2


def test_a_configured_run_compiling_ahead_with_its_frontier_records_it_in_the_saved_plan(tmp_path):
    """`--trainer.compile-ahead.frontier` (`RunConfig.cli`) asks the step compiled ahead for its frontier,
    which its saved plan records (rows 3108, 3147): on GPUs of 0.0007 GiB, where the plan needs two
    microbatches, a rung above it that fits at one."""
    from run_kinds import lm

    from dew.config.lm import LMRunConfig
    from dew.training.ahead import AheadOfTime, CompiledAhead

    runs, path = tmp_path / "runs", tmp_path / "step.dew-aot"
    run = LMRunConfig.cli(["--trainer.name", "ahead", "--trainer.checkpoint-dir", str(runs),
                           "--trainer.steps", "2", "--trainer.compile-ahead.generation", "sm80",
                           "--trainer.compile-ahead.devices", "1", "--trainer.compile-ahead.path", str(path),
                           "--trainer.compile-ahead.capacity-gib", "0.0007",
                           "--trainer.compile-ahead.frontier"],
                          default=lm(tmp_path, drawn=False))
    with pytest.raises(CompiledAhead) as ended:
        run.run()
    assert ended.value.code == 0
    plan = record(AheadOfTime.load(path).plan, "plan")
    frontier = listed(plan["frontier"], "frontier", read=record, what="frontier")
    assert plan["split"] == 2 and frontier
    assert all(entry["fits"] is True and entry["split"] < plan["split"] for entry in frontier)


def test_a_configured_run_compiles_its_step_ahead_for_a_tpu_topology_with_its_stated_links(tmp_path):
    """`--trainer.compile-ahead.generation` names a TPU topology as well as a CUDA generation: the run
    compiles its first step for that topology's compile-only devices here, through the plan a run takes,
    and writes it as any step compiled ahead (ledger 2551). The links it compiles with are the run's own
    (`trainer.links`), which the step records (ledger 2552)."""
    from run_kinds import lm

    from dew.config.lm import LMRunConfig
    from dew.training.ahead import AheadOfTime, CompiledAhead

    runs, path = tmp_path / "runs", tmp_path / "step.dew-aot"
    default = lm(tmp_path, drawn=False)
    assert isinstance(default, LMRunConfig)
    links = {"tensor": {"bytes_per_second": 4.5e11, "flops_per_second": 9.9e14}}
    run = LMRunConfig.cli(["--trainer.name", "ahead", "--trainer.checkpoint-dir", str(runs),
                           "--trainer.steps", "2", "--trainer.compile-ahead.generation", "v5e:2x2",
                           "--trainer.compile-ahead.devices", "4", "--trainer.compile-ahead.path", str(path),
                           "--trainer.mesh.tensor", "2", "--set", f"trainer.links={json.dumps(links)}"],
                          default=default)
    with pytest.raises(CompiledAhead) as ended:
        run.run()
    assert ended.value.code == 0
    saved = AheadOfTime.load(path)
    assert (saved.generation, saved.devices) == ("v5e", 4) and saved.peak_bytes > 0
    assert saved.links["tensor"].bytes_per_second == 4.5e11
    assert not runs.exists()


def test_a_run_on_the_examples_clock_compiles_its_step_ahead_reading_its_progress(tmp_path):
    """A decision run bounded in examples, which its objective expands after its bound so no pass counts
    its steps, keeps its progress on the examples clock: compiled ahead for a GPU it lacks, its step reads
    that progress as fit's does (`Trainer.reading`), under a schedule over the run (`Linear`), and the run
    ends with the step saved, where it was compiled over no clock and refused the schedule (row 2434)."""
    from run_kinds import decision

    from dew.decision.config import DecisionRunConfig
    from dew.training.ahead import AheadOfTime, CompiledAhead

    runs, path = tmp_path / "runs", tmp_path / "step.dew-aot"
    default = decision(tmp_path, drawn=False)
    assert isinstance(default, DecisionRunConfig)
    run = DecisionRunConfig.cli(["--trainer.name", "ahead", "--trainer.checkpoint-dir", str(runs),
                                 "--trainer.examples", "16", "--optim.learning-rate",
                                 '{"class": "linear", "fields": {"peak_value": 0.001}}',
                                 "--trainer.compile-ahead.generation", "sm80",
                                 "--trainer.compile-ahead.devices", "1",
                                 "--trainer.compile-ahead.path", str(path),
                                 "--trainer.compile-ahead.capacity-gib", "4"],
                                default=default)
    with pytest.raises(CompiledAhead) as ended:
        run.run()
    assert ended.value.code == 0
    assert AheadOfTime.load(path).peak_bytes > 0


CLOCKED = ["--trainer.examples", "48", "--trainer.eval-every", "3", "--trainer.checkpoint-every", "3",
           "--trainer.multi-host", "False", "--trainer.compilation-cache-dir", "None",
           "--trainer.cooldowns.at", "50%", "--trainer.cooldowns.length", "25%",
           "--optim.learning-rate", '{"class": "linear", "fields": {"peak_value": 0.001}}']
"""The clocked decision run's flags (the clocked GPU cell's): its tickets stream, of no record count, so its
evaluations and checkpoints are counted in steps (ledger 3153)."""


def test_a_decision_runs_step_compiled_ahead_is_the_program_its_run_lowers_as_fit_hands_its_step(tmp_path):
    """The clocked GPU cell's property, read where a host has no GPU (ledger 3153): the decision run compiled
    ahead for eight sm90 GPUs (`--trainer.compile-ahead`) and the same run lowered for them as `fit` hands it
    its step, its batch placed on the devices (`--trainer.lower-for`), are one program, so a run handed the
    step it compiled ahead takes it. A step lowered over the host batch the loader gave and over that batch
    placed came to two module texts, a helper call's function told apart by its arguments' shardings."""
    from run_kinds import decision

    from dew.decision.config import DecisionRunConfig
    from dew.training._lowering import Lowered
    from dew.training.ahead import AheadOfTime, CompiledAhead, program_of

    path = tmp_path / "ahead.dew-aot"
    stated = ["--trainer.checkpoint-dir", str(tmp_path / "runs"), *CLOCKED]
    target = ["--trainer.compile-ahead.generation", "sm90", "--trainer.compile-ahead.devices", "8",
              "--trainer.compile-ahead.path", str(path)]
    ahead = DecisionRunConfig.cli(["--trainer.name", "ahead", *stated, *target],
                                  default=decision(tmp_path, drawn=False))
    with pytest.raises(CompiledAhead):
        ahead.run()
    lowering = DecisionRunConfig.cli(["--trainer.name", "lowered", *stated, "--trainer.lower-for", "sm90"],
                                     default=decision(tmp_path, drawn=False))
    with pytest.raises(Lowered) as lowered:
        lowering.run()
    assert program_of(lowered.value.lowered) == AheadOfTime.load(path).program


def a_clocked_run_with_a_cooldown_trains_from_its_step_compiled_ahead_compiling_no_other(tmp_path):
    """A GPU flow's cell (tests/test_gpu_training.py; it trains from a step compiled ahead on this host's own
    GPU): a decision run on the examples clock under a schedule over the run, with a cooldown, compiled
    ahead through its command line for this GPU's generation (`--trainer.compile-ahead`) and then handed
    the step it saved (`--trainer.ahead`), trains from that step through its cooldown's branch, its last
    window's too, and compiles no step program of its own: each batch it steps closes its window as the
    one it compiled ahead over did (ledgers 2434, 2527)."""
    from run_kinds import decision

    from dew.decision.config import DecisionRunConfig
    from dew.nn.kernels.generation import device_generation
    from dew.training.ahead import CompiledAhead

    runs, path = tmp_path / "runs", tmp_path / "step.dew-aot"
    default = decision(tmp_path, drawn=False)
    stated = ["--trainer.checkpoint-dir", str(runs), *CLOCKED]
    ahead = ["--trainer.compile-ahead.generation", device_generation(),
             "--trainer.compile-ahead.devices", "1", "--trainer.compile-ahead.path", str(path)]
    compiling = DecisionRunConfig.cli(["--trainer.name", "ahead", *stated, *ahead], default=default)
    with pytest.raises(CompiledAhead):
        compiling.run()
    handed = DecisionRunConfig.cli(["--trainer.name", "handed", *stated, "--trainer.ahead", str(path)],
                                   default=default)
    with Compiles("step") as compiles:
        state = handed.run()
    assert compiles.count == 0 and int(state.updates) > 0


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


def a_step_compiled_ahead_and_saved_runs_on_its_gpu_in_the_run_s_place(tmp_path):
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


def a_run_handed_its_step_compiled_ahead_trains_from_it_on_its_gpu(tmp_path):
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


def test_a_watched_runs_step_compiled_ahead_carries_its_sampled_step():
    """A run that inspects compiles its sampled step beside its step (`Candidate.sampled`), and a step
    compiled ahead for it carries both (`AheadOfTime.sampled`), each its own program, so the run handed it
    compiles neither on its target (train 332's review)."""
    from dew.training import Inspection
    from dew.training.ahead import compile_ahead, compile_only_gpus

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    devices = compile_only_gpus("sm90", 1)
    watched = compile_ahead(fine_tune(MeshSpec().build(devices), inspection=Inspection(every=1)), batch)
    plain = compile_ahead(fine_tune(MeshSpec().build(devices)), batch)
    assert watched.sampled is not None and plain.sampled is None
    assert watched.sampled.program != watched.program and watched.program == plain.program


def a_watched_run_handed_its_steps_compiled_ahead_compiles_neither(tmp_path):
    """A GPU flow's cell (tests/test_gpu_training.py): a run that inspects, handed its step and sampled step
    compiled ahead for this host's own GPU, takes both (`taken`) and compiles no step program on its GPU
    (`Compiles`): its first plan's compile is the step compiled ahead, and it trains from it (train 332's
    review)."""
    from dew.nn.kernels.generation import chosen, device_generation
    from dew.telemetry.compiles import Compiles
    from dew.training import Inspection
    from dew.training.ahead import AheadOfTime, compile_ahead, compile_only_gpus

    batch, watched = token_rows(VOCAB, BATCH, SEQ_LEN + 1), Inspection(every=1)
    with chosen(narrow_copies="off"):
        compile_ahead(fine_tune(MeshSpec().build(compile_only_gpus(device_generation(), 1)), xla=True,
                                inspection=watched), batch).save(tmp_path / "step.dew-aot")
        run = fine_tune(MeshSpec().build(jax.devices()[:1]), xla=True, inspection=watched,
                        ahead=AheadOfTime.load(tmp_path / "step.dew-aot"))
        with Compiles("step") as compiles:
            state = run.fit(batch, steps=1, log_every=1)
    first, *_ = listed(record(run.plan, "plan")["probes"], "probes", read=record, what="probes")
    assert int(state.step) == 1 and first["ahead"] is True and compiles.count == 0


@pytest.mark.tpu(reason="runs a step compiled ahead for this host's TPU topology as a run's first step")
def test_a_run_handed_its_step_compiled_ahead_for_a_tpu_topology_trains_from_it(tmp_path):
    """A step compiled ahead for this host's one-chip TPU topology (`compile_only_tpus`), saved and handed
    to the run on the TPU itself, is the program the run lowers its first step to, and the run trains
    from it, its plan's first compile the step compiled ahead: the TPU run row 2090 asks for."""
    from dew.nn.kernels.generation import gpu_target
    from dew.training.ahead import AheadOfTime, compile_ahead, compile_only_tpus

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    chip = gpu_target(jax.devices()[0].device_kind).generation
    compile_ahead(fine_tune(MeshSpec().build(compile_only_tpus(f"{chip}:1x1")), xla=True),
                  batch).save(tmp_path / "step.dew-aot")
    run = fine_tune(MeshSpec().build(jax.devices()[:1]), ahead=AheadOfTime.load(tmp_path / "step.dew-aot"),
                    xla=True)
    state = run.fit(batch, steps=2, log_every=1)
    first, *_ = listed(record(run.plan, "plan")["probes"], "probes", read=record, what="probes")
    assert int(state.step) == 2 and first["ahead"] is True


def test_a_run_naming_its_input_alone_compiles_its_step_ahead(tmp_path):
    """An objective that names its input field alone (`Supervised(inputs="x")`) compiles its step ahead
    (`AheadTarget.compiled`) as `fit` would start it: its input declared at the shape the loader's first
    batch gives (`Trainer._declare_input`) before the run is checked (train 325's review of ledger
    2551)."""
    from flax import linen as nn

    from dew.data.dataset import Loader
    from dew.objectives.supervised import Supervised
    from dew.training.ahead import AheadTarget, compile_only_gpus

    data = Loader(train=lambda partition: iter([{"x": np.ones((BATCH, 8), np.float32),
                                                  "y": np.zeros((BATCH, 4), np.float32)}] * 4),
                  val=None, batch=BATCH, records=None)
    trainer = Trainer(Supervised(nn.Dense(4), lambda outputs, batch: (outputs - batch["y"]) ** 2, inputs="x"),
                      optax.sgd(0.1), key=jax.random.key(0),
                      mesh=MeshSpec().build(compile_only_gpus("sm90", 1)))
    ended = AheadTarget("sm90", 1, str(tmp_path / "step.dew-aot")).compiled(trainer, data, steps=1)
    assert ended.ahead.devices == 1 and trainer.objective.inputs is not None


FAST = {"tensor": Link(1e18, 1e9)}
"""A tensor axis whose link moves a projection's bytes in no time beside its FLOPs: any projection
spreads over it, as over a measured NVLink one may."""


def latent(mesh, *, ahead=None, links=None, checkpoints=None):
    """A decoder of latent attention, whose down projection a tensor axis's link may spread
    (`dew.nn.sharding.down_projection`), trained over `mesh`."""
    from dew.nn.mla import MLAMixer

    model = models.build("causal_transformer", vocab_size=VOCAB, **TINY_DECODER, attention_impl="xla",
                         mixer=MLAMixer(q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4,
                                        qk_rope_head_dim=4, v_head_dim=4))
    return Trainer(LMObjective(model, SEQ_LEN), optax.adam(1e-3), key=jax.random.key(0), mesh=mesh,
                   layout=Layout(min_shard=TINY_SHARD), ahead=ahead, links=links, checkpoints=checkpoints)


def test_a_step_compiled_ahead_records_the_links_its_program_was_decided_by():
    """A latent-attention step compiled ahead over a tensor axis of two compile-only GPUs, which no
    process holds to measure, spreads no projection over it and records that link
    (`AheadOfTime.links`); the same step over a link that spreads one is another program, which a run
    on GPUs that measure such a link would lower in its place (row 2229)."""
    from dew.training.ahead import compile_ahead, compile_only_gpus

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    target = MeshSpec(tensor=2).build(compile_only_gpus("sm90", 2))
    saved = compile_ahead(latent(target), batch)
    assert saved.links["tensor"].bytes_per_second is None
    assert compile_ahead(latent(target, links=FAST), batch).program != saved.program


def test_a_stated_link_is_the_one_a_step_compiled_ahead_records_and_a_handed_run_traces_with(caplog):
    """A step compiled ahead over a link its trainer states, as `trainer.links` states one, records
    that link whole, bandwidth and peak, and is the program a stated link spreads by; a run
    handed it that states another traces with the step's, and says so once (train 300's review: the
    step recorded the bandwidth alone, so a run stating the same link traced with its own peak)."""
    from dew.training.ahead import compile_ahead, compile_only_gpus

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    target = MeshSpec(tensor=2).build(compile_only_gpus("sm90", 2))
    stated = compile_ahead(latent(target, links=FAST), batch)
    (recorded,) = stated.links.values()
    assert (recorded.bytes_per_second, recorded.flops_per_second, recorded.spread) == (1e18, 1e9, True)
    assert stated.program != compile_ahead(latent(target), batch).program
    other = {"tensor": Link(1.0, 1.0)}
    with caplog.at_level("WARNING", logger="dew.training.ahead"):
        assert stated.over(other) == stated.over(FAST) == stated.links
    assert len(caplog.records) == 1


def a_run_handed_a_step_compiled_ahead_traces_with_the_links_it_was_compiled_with(tmp_path):
    """A run on two GPUs whose link would spread a projection (`FAST`), handed a step compiled ahead
    over compile-only GPUs that spread none, traces with the link the step records and trains from it,
    rather than lowering another program and refusing it; resumed from its checkpoint without the step,
    it traces with that link again, not one it measures (row 2229)."""
    from dew.checkpoints import Checkpoints
    from dew.nn.kernels.generation import chosen, device_generation
    from dew.training.ahead import AheadOfTime, compile_ahead, compile_only_gpus

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    spec, path = MeshSpec(tensor=2), tmp_path / "step.dew-aot"
    here = spec.build(jax.devices()[:2])
    with chosen(narrow_copies="off"):
        target = spec.build(compile_only_gpus(device_generation(), 2))
        compile_ahead(latent(target), batch).save(path)
        saved = str(tmp_path / "run")
        run = latent(here, ahead=AheadOfTime.load(path), links=FAST, checkpoints=Checkpoints(saved))
        state = run.fit(batch, steps=2, log_every=1)
        resumed = latent(here, checkpoints=Checkpoints(saved))
        again = resumed.fit(batch, steps=3, log_every=1)
    first, *_ = listed(record(run.plan, "plan")["probes"], "probes", read=record, what="probes")
    assert int(state.step) == 2 and first["ahead"] is True
    assert int(again.step) == 3 and resumed.links["tensor"].bytes_per_second is None


def a_run_takes_the_saved_offline_split_before_loading_its_executable(tmp_path):
    """An offline plan for two microbatches is handed to a live run with its split unpinned. The
    live planner takes the saved split before lowering, so the artifact's program is reused and its
    two donated microbatch calls produce one committed update, not a mismatch or a recompile."""
    from dew.nn.kernels.generation import chosen, device_generation
    from dew.training.ahead import AheadOfTime, compile_ahead, compile_only_gpus

    batch = token_rows(VOCAB, BATCH, SEQ_LEN + 1)
    with chosen(narrow_copies="off"):
        target = MeshSpec().build(compile_only_gpus(device_generation(), 1))
        saved = compile_ahead(fine_tune(target, xla=True, split=2), batch)
        path = tmp_path / "split.dew-aot"
        saved.save(path)
        here = MeshSpec().build(jax.devices()[:1])
        run = fine_tune(here, ahead=AheadOfTime.load(path), xla=True, split=None)
        state = run.fit(batch, steps=1, log_every=1)
    plan = record(run.plan, "plan")
    first, *_ = listed(plan["probes"], "probes", read=record, what="probes")
    assert int(state.step) == 1 and int(state.updates) == 1
    assert plan["split"] == 2 and first["ahead"] is True


def test_programs_compiled_ahead_compile_at_once():
    """Named programs compiled ahead (`compile_programs`) are lowered in turn and compiled at once, each in a
    thread of its own (`report.compiled_at_once`), each with the trees it has compiled alone."""
    from dew.telemetry.compiles import Compiles
    from dew.training.ahead import compile_programs

    square = jax.ShapeDtypeStruct((128, 128), jnp.float32)
    programs = {"dot": (lambda x, y: x @ y, [square, square]),
                "sum": (lambda x: jnp.sum(jnp.tanh(x)), [square])}
    with Compiles(None) as compiles:
        built = compile_programs(programs, "sm90")
    threads = {program.thread for program in compiles.programs if program.kind == "compiled"}
    assert threads and all(thread.startswith("dew-compile") for thread in threads), threads
    alone = {name: compile_programs({name: program}, "sm90").programs[name]
             for name, program in programs.items()}
    assert {name: each.trees for name, each in built.programs.items()} == {
        name: each.trees for name, each in alone.items()}


def test_programs_compiled_ahead_save_and_load_whole_and_load_only_on_their_own_target(tmp_path):
    """Named programs compiled ahead for a GPU this CPU host lacks (`compile_programs`), one of them reading
    no argument, are the programs loaded after saving, each an executable this host's CPU is no target of:
    loading them here is refused, naming the GPU kind it is not (ledger 2716: an H100 node loads what a CPU
    host built, and runs it alone)."""
    from dew.training.ahead import AheadPrograms, BuildTarget, compile_programs

    square = jax.ShapeDtypeStruct((128, 128), jnp.float32)
    built = compile_programs({"dot": (lambda x, y: x @ y, [square, square]),
                              "iota": (lambda: jnp.arange(8.0), [])}, "sm90")
    assert set(built.programs) == {"dot", "iota"}
    assert (built.target.generation, built.target.devices) == ("sm90", 1)
    built.save(tmp_path / "programs.dew-aot")
    loaded = AheadPrograms.load(tmp_path / "programs.dew-aot")
    assert loaded == built
    with pytest.raises(ValueError):
        loaded.loaded()
    here = BuildTarget.of(jax.devices()[:1])
    assert [line.split()[0] for line in built.target.differences(here)] == ["generation", "multiprocessors"]
    # A kind of the generation with its multiprocessors is its target, whatever its memory or its name: the
    # build's target is sm90's facts, 132 multiprocessors.
    assert (built.target.generation, built.target.multiprocessors) == ("sm90", 132)


def test_a_devices_own_target_is_its_kinds_with_the_facts_it_reports():
    """A device's own target, what a program compiled ahead is checked against, is the one `gpu_target`
    resolves for it unassumed: an H100 PCIe its own kind and peaks with the multiprocessors it reports, and an
    sm90 kind the table names no SKU of its own kind, its peaks unknown, never the generation's canonical
    device's."""
    from types import SimpleNamespace

    from dew.nn.kernels.generation import device_target

    def device(kind, cores):
        return SimpleNamespace(platform="gpu", device_kind=kind, compute_capability="9.0", core_count=cores,
                               local_hardware_id=0)

    pcie = device_target(device("NVIDIA H100 PCIe", 114))
    unnamed = device_target(device("NVIDIA H20", 78))
    assert (pcie.generation, pcie.device_kind, pcie.bf16_peak, pcie.multiprocessors) == (
        "sm90", "NVIDIA H100 PCIe", 756e12, 114)
    assert (unnamed.generation, unnamed.device_kind, unnamed.bf16_peak, unnamed.multiprocessors) == (
        "sm90", "NVIDIA H20", None, 78)


def test_a_runs_compilers_dumps_hold_what_it_compiled_and_nothing_where_it_compiled_nothing(tmp_path,
                                                                                           monkeypatch):
    """A run given `dumps` writes its compiles under that directory, a child process's too from another
    working directory, whatever the environment said of XLA's dumping (elsewhere, or filtered to no
    module), and nothing where it compiled nothing (`compiled_in`); a directory dumps did not make is
    refused, never read as empty: the gate of a run of programs compiled ahead (ledger 2716)."""
    import subprocess
    import sys
    from pathlib import Path

    from dew.training.ahead import compiled_in, dumps

    child = "import jax; jax.jit(lambda x: x + 1)(1.0)"
    elsewhere = f"[sys.executable, '-c', {child!r}], cwd='/', check=True"
    compiling = f"import subprocess, sys; subprocess.run({elsewhere})"
    monkeypatch.chdir(tmp_path)
    given = (("elsewhere", f"--xla_dump_to={tmp_path / 'other'}"),
             ("filtered", "--xla_dump_hlo_module_re=^never$"))
    for name, flags in given:
        environment = {**os.environ, "JAX_PLATFORMS": "cpu", "XLA_FLAGS": flags}
        subprocess.run([sys.executable, "-c", compiling], env=dumps(Path(name), environment), check=True)
        assert compiled_in(tmp_path / name), name
    computing = dumps(Path("computes"), {**os.environ, "JAX_PLATFORMS": "cpu"})
    subprocess.run([sys.executable, "-c", "import numpy; numpy.arange(8.0).sum()"], env=computing, check=True)
    assert compiled_in(tmp_path / "computes") == []
    with pytest.raises(FileNotFoundError):
        compiled_in(tmp_path / "never-made")
    # A dump directory that is a link elsewhere is refused at both ends, and one that takes no file at the
    # start.
    (tmp_path / "linked").mkdir()
    (tmp_path / "link-target").mkdir()
    (tmp_path / "linked" / "xla").symlink_to(tmp_path / "link-target")
    with pytest.raises(ValueError):
        dumps(tmp_path / "linked", {})
    (tmp_path / "linked" / "mosaic").mkdir()
    with pytest.raises(ValueError):
        compiled_in(tmp_path / "linked")
    (tmp_path / "locked" / "xla").mkdir(parents=True)
    (tmp_path / "locked" / "xla").chmod(0o500)
    with pytest.raises(PermissionError):
        dumps(tmp_path / "locked", {})


def test_a_flag_given_twice_is_read_by_its_order():
    """XLA reads a flag given twice by its last value, so the two orders of one pair are two targets
    (`computation_flags`, review)."""
    from dew.training.ahead import computation_flags

    first = "--xla_allow_excess_precision=false --xla_allow_excess_precision=true --xla_dump_to=/tmp/x"
    second = "--xla_allow_excess_precision=true --xla_allow_excess_precision=false"
    assert computation_flags(first) != computation_flags(second)
    excess = "--xla_allow_excess_precision"
    assert computation_flags(first) == (f"{excess}=false", f"{excess}=true")
