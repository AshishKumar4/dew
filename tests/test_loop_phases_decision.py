"""A decision run whose implicit loop's pass count grows with its data's phases (row 5480(a)'s
acceptance): the tiny Qwen 3.5, its rows batched by their tokens after a step that drops some, so each
phase reads per share (`dew.data.dataset.StepPhases`), two sources in two phases on the step clock, its
block [1, 3) run once in the first and twice in the second (`DecoderRunConfig.loop_steps`, `Trainer`'s
`phase_passes`)."""

import csv
import dataclasses
import itertools
from pathlib import Path

import jax
import numpy as np
import pytest
from run_kinds import MostlyKept, trained

from dew.config import ModelConfig
from dew.data import Files, Loading, phases
from dew.decision import Decide, decisions
from dew.decision.config import DecisionRunConfig

QWEN35 = Path(__file__).parent / "fixtures" / "hf" / "qwen35-native-tiny"
"""The tiny Qwen 3.5, with its tokenizer, which lays a decision's rows out."""
MAX_LEN = 256
"""A decision row's tokens: room for the tokenizer's chat template, the instructions, the options and a
ticket (at 64 every row was refused as too long, and the read went on without one)."""
TOKENS = 8 * MAX_LEN
"""A batch's token budget: eight rows, which the run's eight row shards divide."""
STEPS = (1, 2)
"""The block's passes a phase."""
PHASE = 2
"""The step the first phase ends at."""
TEAMS = ("billing", "technical", "sales")
"""The team a ticket asks for, its question's options, which a table read by index states."""


def tickets(root: Path, name: str, team: str) -> Files:
    """A source of 32 tickets, each its own text, labelled by its team: its rows are this source's."""
    table = root / f"{name}.csv"
    asked = [(f"{name} charged twice for March", "billing"), (f"{name} the site is down", "technical"),
             (f"{name} how much is the pro plan", "sales"), (f"{name} refund my invoice", team)]
    with table.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows((f"{text}, ticket {index}", label) for index in range(8) for text, label in asked)
    return Files(str(table), format="csv")


def asked(rows: Files):
    """`rows` as decision rows, some drawn out, read on past their end, batched by their tokens: a read
    per share."""
    return (rows.pipe(decisions, question="team", options=TEAMS).read("indexed")
            .random_filter_map(MostlyKept()).repeat()
            .batch(tokens=TOKENS).loading(Loading(workers=0)))


def run(root: Path, steps, data) -> DecisionRunConfig:
    del root
    return DecisionRunConfig(pretrained=str(QWEN35),
                             model=ModelConfig("causal_transformer", {"dtype": "float32"}), max_len=MAX_LEN,
                             data=data, validation=None, loop_layers=(1, 3), loop_steps=steps,
                             loop_implicit=True)


def test_a_phased_decision_run_resumes_with_its_loops_passes_recomputed(tmp_path):
    """A run stopped inside its second phase resumes with its loop's passes recomputed in the backward
    (`loop_remat`), memory alone: the resume takes it and trains on through the phase, where a change of
    what the run computes (its phases' passes) it refuses (row 5558)."""
    data = phased(tmp_path)
    trained(run(tmp_path, STEPS, data), tmp_path, "stopped", trainer={"steps": 3, "checkpoint_every": 1})
    recomputed = dataclasses.replace(run(tmp_path, STEPS, data), loop_remat=True)
    resumed, _ = trained(recomputed, tmp_path, "stopped", trainer={"steps": 4, "checkpoint_every": 1})
    assert int(resumed.step) == 4
    with pytest.raises(ValueError):
        trained(run(tmp_path, (1, 3), data), tmp_path, "stopped", trainer={"steps": 5, "checkpoint_every": 1})


def phased(root: Path):
    return phases((asked(tickets(root, "first", "billing")), PHASE),
                  (asked(tickets(root, "second", "sales")), None))


def parameters(state) -> dict:
    return {jax.tree_util.keystr(path): np.asarray(leaf)
            for path, leaf in jax.tree_util.tree_leaves_with_path(state.variables["params"])}


def theta(state) -> np.ndarray:
    (found,) = [leaf for name, leaf in parameters(state).items() if "loop_theta" in name]
    return found


def test_a_decision_run_whose_passes_grow_by_phase_is_its_phases_runs_and_resumes_as_one(tmp_path):
    """The run's first phase is a one-pass run's over the first source alone, bit for bit (its rows that
    source's, its steps at one pass), its implicit loop's θ untouched
    while every step runs one pass (its gradient zero) and moved from the second phase's first step; stopped
    after step 3, inside its second phase, and resumed, it lands where the run nobody stopped lands; its saved
    run loads with the last phase's two passes. A count list whose length is not the data's phases is
    refused before anything loads."""
    with pytest.raises(ValueError):
        run(tmp_path, (1, 2, 3), phased(tmp_path))
    data = phased(tmp_path)
    first, _ = trained(run(tmp_path, STEPS, data), tmp_path, "first", trainer={"steps": PHASE})
    # The comparisons below hold on numbers: NaN equals NaN to them.
    assert all(np.all(np.isfinite(leaf)) for leaf in parameters(first).values())
    alone, _ = trained(run(tmp_path, 1, asked(tickets(tmp_path, "first", "billing"))), tmp_path, "alone",
                       trainer={"steps": PHASE})
    assert parameters(first).keys() == parameters(alone).keys()
    for name, leaf in parameters(first).items():
        np.testing.assert_array_equal(leaf, parameters(alone)[name], err_msg=name)
    assert not np.any(theta(first)), "θ's gradient is zero while each step runs one pass"

    whole, _ = trained(run(tmp_path, STEPS, data), tmp_path, "whole", trainer={"steps": 4})
    assert all(np.all(np.isfinite(leaf)) for leaf in parameters(whole).values())
    assert np.any(theta(whole)), "θ moves once a step runs two passes"
    trained(run(tmp_path, STEPS, data), tmp_path, "stopped", trainer={"steps": 3, "checkpoint_every": 1})
    resumed, _ = trained(run(tmp_path, STEPS, data), tmp_path, "stopped",
                         trainer={"steps": 4, "checkpoint_every": 1})
    assert int(resumed.step) == 4
    for name, leaf in parameters(whole).items():
        np.testing.assert_array_equal(leaf, parameters(resumed)[name], err_msg=name)
    loaded = Decide.from_run(str(tmp_path / "runs" / "stopped"))
    assert loaded.model.backbone.loop.steps == STEPS[-1]



def test_each_phases_step_of_the_decision_run_is_a_fixed_count_runs_step_and_counts_its_flops(tmp_path):
    """From the same state over the same batch, each phase's step of the phased decision run computes what
    a run whose loop runs that phase's count computes: its loss and its stepped parameters the same bits (the
    ask's invariant 1); and the FLOPs it counts are that run's, so its FLOP count follows the phase's count,
    two passes counting more than one (invariant 5); and its offline view of an update is that run's."""
    from dew.batch import PHASE as READ_IN
    from dew.config import TrainerConfig
    from dew.data.dataset import DataPartition
    from dew.distributed.batches import split_fields
    from dew.training import Layout, MeshSpec
    from dew.training.inspection import Inspection, inspected

    settings = TrainerConfig(checkpoint_dir=str(tmp_path / "runs"), steps=4, log_every=1, eval_every=None,
                             checkpoint_every=None, checkpoint_final=False, multi_host=False,
                             mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=64))

    def built(steps, data, name: str):
        """The run's trainer and its training read, as `train` builds them."""
        prepared = dataclasses.replace(run(tmp_path, steps, data), trainer=settings).prepare()
        readied, loader, _, _, _ = prepared.run.readied(prepared.objective, prepared.dataset, None, (), name)
        trainer = readied.built(prepared.objective, readied.trainer.mesh, None)
        # Its optimizer built over the run's length, as `fit` builds it.
        trainer.scheduled(settings.steps)
        return trainer, loader

    stepping, loader = built(STEPS, phased(tmp_path), "phased")
    stream = loader.train(DataPartition())
    try:
        batches = {}
        for batch in itertools.islice(stream, 4):
            batches.setdefault(batch[READ_IN].index, batch)
    finally:
        stream.close()
    assert sorted(batches) == [0, 1]
    counted = []
    for phase, passes in enumerate(STEPS):
        # The count the step runs is its batch's phase's, read on the host; the step reads the rest.
        assert stepping.phase_passes.at(batches[phase]) == passes
        batch, _ = split_fields(batches[phase])
        state, _, _ = stepping.place()
        stepped, loss, *_ = stepping.compile(state, batch)(state, batch)
        alone, _ = built(passes, asked(tickets(tmp_path, "first", "billing")), f"alone-{passes}")
        start, _, _ = alone.place()
        own, own_loss, *_ = alone.compile(start, batch)(start, batch)
        assert np.isfinite(np.asarray(loss)) and np.array_equal(np.asarray(loss), np.asarray(own_loss)), phase
        assert jax.tree.all(jax.tree.map(lambda a, b: bool(np.array_equal(a, b)),
                                         jax.device_get(stepped.variables), jax.device_get(own.variables)))
        assert stepping.step_flops == alone.step_flops is not None
        counted.append(stepping.step_flops)
        # Its offline view of one update over the phase's batch is that run's view (`inspected`, at the
        # count of the phase the batch was read in, not the run's last).
        for trainer in (stepping, alone):
            trainer.inspection = Inspection(paths=("*",), targets=("updates",))
        assert inspected(stepping, batches[phase]) == inspected(alone, batch), phase
    assert counted[0] != counted[1]


def test_a_phased_runs_later_phases_are_named_from_the_first_batch_its_loop_read(tmp_path):
    """A run's later phases are named (`later_phases`) from the batch its loop hands its first step, its
    work counted and its rows marked as the loop reads it (`marked`, `Trainer._counted_batch`), counted no
    more: the second phase, whose loop runs another pass count, is another program. Counted again, its
    telemetry columns were refused as reserved before a Qwen3.5-4B run's first step on a GH200."""
    from dew.data.dataset import DataPartition
    from dew.training.ahead import first_step_batch, later_phases
    from dew.training.distributed import marked

    stated = run(tmp_path, STEPS, phased(tmp_path))
    prepared = dataclasses.replace(stated, trainer=dataclasses.replace(stated.trainer, steps=4)).prepare()
    readied = prepared.run.readied(prepared.objective, prepared.dataset, None, (), "named")
    config, loader, steps = readied[0], readied[1], readied[4]
    trainer = config.built(prepared.objective, config.trainer.mesh, prepared.rollout)
    first = first_step_batch(trainer, loader, steps, loader.examples if steps is None else None)
    assert later_phases(trainer, marked(DataPartition(), trainer._counted_batch)(first)) == (1,)


def test_a_phased_runs_later_phase_compiles_ahead_as_the_run_lowers_it_at_its_boundary(tmp_path):
    """A run's later phases compile ahead through its record, trained by `dew train` with its target the
    phases after the first (`AheadTarget.phases`, `later_phases`, by the trainer's own
    key of each phase's batch), over its host's GPUs (here a serialized topology of eight compile-only
    H100s, as many as the run's row shards, so each reads its phases' rows as the run reads them),
    ending with status 0 and training nothing; and the step it compiles for the second phase is the
    program the run lowers at that phase's boundary: the run stopped inside its second phase, compiled
    ahead as its resume would step, lowers the same program, digest for digest."""
    from jax.experimental import topologies

    from dew.nn.kernels.generation import compile_only_gpus
    from dew.training.ahead import AheadTarget, CompiledAhead

    topology = tmp_path / "topology.bin"
    host = compile_only_gpus("sm90", 8, 40 * 2**30)
    topology.write_bytes(topologies.TopologyDescription(list(host)).serialize())
    data = phased(tmp_path)
    phases_target = AheadTarget("sm90", 8, topology=str(topology), phases=(1,))
    # The topology states its GPUs' memory: a capacity beside it, which nothing would read, is refused.
    with pytest.raises(ValueError):
        dataclasses.replace(phases_target, capacity_gib=30.0).compile_only()
    with pytest.raises(CompiledAhead) as later:
        trained(run(tmp_path, STEPS, data), tmp_path, "ahead",
                trainer={"steps": 4, "compile_ahead": phases_target})
    assert later.value.code == 0 and later.value.path is None
    assert not (tmp_path / "runs" / "ahead").exists()
    trained(run(tmp_path, STEPS, data), tmp_path, "stopped", trainer={"steps": 3, "checkpoint_every": 1})
    resumed_target = AheadTarget("sm90", 8, path=str(tmp_path / "resumed.ahead"), topology=str(topology))
    with pytest.raises(CompiledAhead) as resumed:
        trained(run(tmp_path, STEPS, data), tmp_path, "stopped", trainer={"steps": 4,
                                                                         "compile_ahead": resumed_target})
    assert resumed.value.ahead.program == later.value.ahead.program
