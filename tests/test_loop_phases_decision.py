"""A decision run whose implicit loop's pass count grows with its data's phases (row 5480(a)'s
acceptance): the tiny Qwen 3.5, its rows batched by their tokens after a step that drops some, so each
phase reads per share (`dew.data.dataset.StepPhases`), two sources in two phases on the step clock, its
block [1, 3) run once in the first and twice in the second (`DecoderRunConfig.loop_steps`, `Trainer`'s
`phase_passes`)."""

import csv
import dataclasses
import itertools
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from run_kinds import MostlyKept, trained

from dew.config import ModelConfig
from dew.data import Files, Loading, Phase, Steps, phases
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
    trained(run(tmp_path, STEPS, data), tmp_path, "stopped", trainer={"steps": 3,
                                                                      "checkpoint_every": 1})
    recomputed = dataclasses.replace(run(tmp_path, STEPS, data), loop_remat=True)
    resumed, _ = trained(recomputed, tmp_path, "stopped", trainer={"steps": 4,
                                                                   "checkpoint_every": 1})
    assert int(resumed.step) == 4
    with pytest.raises(ValueError):
        trained(run(tmp_path, (1, 3), data), tmp_path, "stopped", trainer={"steps": 5,
                                                                           "checkpoint_every": 1})


def phased(root: Path):
    return phases(Phase(asked(tickets(root, "first", "billing")), Steps(PHASE)),
                  Phase(asked(tickets(root, "second", "sales"))))


def named_leaves_of(variables) -> list:
    """Each leaf of `variables`' parameters by its path, shapes held as they are."""
    return [(jax.tree_util.keystr(path), leaf)
            for path, leaf in jax.tree_util.tree_leaves_with_path(variables["params"])]


def named_leaves(variables) -> dict:
    """Each leaf of `variables`' parameters by its path."""
    return {jax.tree_util.keystr(path): np.asarray(leaf)
            for path, leaf in jax.tree_util.tree_leaves_with_path(variables["params"])}


def parameters(state) -> dict:
    return named_leaves(state.variables)


def theta(state) -> np.ndarray:
    (found,) = [leaf for name, leaf in parameters(state).items() if "loop_theta" in name]
    return found


def test_a_decision_runs_first_phase_is_a_one_pass_runs_over_its_first_source(tmp_path):
    """The run's first phase is a one-pass run's over the first source alone, bit for bit (its rows that
    source's, its steps at one pass), its implicit loop's θ untouched while every step runs one pass (its
    gradient zero). A count list whose length is not the data's phases is refused before anything loads.
    Its own cell apart from the resume's: one cell of both trained five runs (row 5531, 311 s in CI)."""
    with pytest.raises(ValueError):
        run(tmp_path, (1, 2, 3), phased(tmp_path))
    first, _ = trained(run(tmp_path, STEPS, phased(tmp_path)), tmp_path, "first",
                       trainer={"steps": PHASE})
    # The comparisons below hold on numbers: NaN equals NaN to them.
    assert all(np.all(np.isfinite(leaf)) for leaf in parameters(first).values())
    alone, _ = trained(run(tmp_path, 1, asked(tickets(tmp_path, "first", "billing"))), tmp_path, "alone",
                       trainer={"steps": PHASE})
    assert parameters(first).keys() == parameters(alone).keys()
    for name, leaf in parameters(first).items():
        np.testing.assert_array_equal(leaf, parameters(alone)[name], err_msg=name)
    assert not np.any(theta(first)), "θ's gradient is zero while each step runs one pass"


def test_a_decision_run_whose_passes_grow_by_phase_resumes_as_one(tmp_path):
    """The phased run's implicit loop's θ moves from the second phase's first step; stopped after step 3,
    inside its second phase, and resumed, it lands where the run nobody stopped lands, bit for bit; its
    saved run loads with the last phase's two passes."""
    data = phased(tmp_path)
    whole, _ = trained(run(tmp_path, STEPS, data), tmp_path, "whole", trainer={"steps": 4})
    assert all(np.all(np.isfinite(leaf)) for leaf in parameters(whole).values())
    assert np.any(theta(whole)), "θ moves once a step runs two passes"
    trained(run(tmp_path, STEPS, data), tmp_path, "stopped", trainer={"steps": 3,
                                                                      "checkpoint_every": 1})
    resumed, _ = trained(run(tmp_path, STEPS, data), tmp_path, "stopped",
                         trainer={"steps": 4, "checkpoint_every": 1})
    assert int(resumed.step) == 4
    for name, leaf in parameters(whole).items():
        np.testing.assert_array_equal(leaf, parameters(resumed)[name], err_msg=name)
    loaded = Decide.from_run(str(tmp_path / "runs" / "stopped"))
    assert loaded.model.backbone.loop.steps == STEPS[-1]


def test_a_phased_run_with_a_theta_per_pass_keeps_its_rows_through_its_phases_and_resumes(tmp_path):
    """The phased run with a θ per later pass (`loop_theta_per_pass`, rows for its largest count): its θ
    holds its rows through the phase of one pass, whose steps move none, and moves from the second; stopped
    inside the second phase and resumed, it lands where the run nobody stopped lands, bit for bit; its
    record states the choice and its saved run loads with its rows; and the choice without an implicit loop
    is refused before anything loads (row 7661)."""
    from dew.checkpoints import RUN_FILE
    from dew.training.budget import Pause

    data = phased(tmp_path)

    def per_pass(name: str, **trainer):
        return trained(dataclasses.replace(run(tmp_path, STEPS, data), loop_theta_per_pass=True), tmp_path,
                       name, trainer={"steps": 4, "checkpoint_every": 1, **trainer})[0]

    whole = per_pass("whole")
    assert theta(whole).shape[0] == STEPS[-1] - 1 and np.any(theta(whole))
    # Preempted at its third step, as a run stopped mid-schedule is: built for the whole run's four steps,
    # it steps the whole run's programs (row 8407).
    stopped = per_pass("stopped", stop=Pause(.75))
    # It stops where it is resumed from, at its third step: a pause that rounded elsewhere fails here, not
    # by a resume from another step.
    assert int(stopped.step) == 3, int(stopped.step)
    resumed = per_pass("stopped")
    for name, leaf in parameters(whole).items():
        np.testing.assert_array_equal(leaf, parameters(resumed)[name], err_msg=name)
    # The run's record states the choice (its data built in Python, which no record reads back).
    assert recorded(json.loads((tmp_path / "runs" / "stopped" / RUN_FILE).read_text()), "loop_theta_per_pass")
    loaded = Decide.from_run(str(tmp_path / "runs" / "stopped"))
    assert loaded.model.backbone.loop.theta_passes == STEPS[-1]


def recorded(record, name: str):
    """The value a run's record (`run.json`'s JSON) holds for its field `name`, wherever it states it."""
    if isinstance(record, dict):
        if name in record:
            return record[name]
        found = [recorded(value, name) for value in record.values()]
    elif isinstance(record, list):
        if len(record) == 2 and record[0] == name:
            return record[1]
        found = [recorded(value, name) for value in record]
    else:
        return None
    return next((value for value in found if value is not None), None)


def test_a_theta_per_pass_over_a_checkpoints_implicit_loop_starts_each_row_at_its_theta():
    """A run asking only for a θ per pass over a checkpoint whose implicit loop holds one θ takes it at the
    checkpoint's own count, each row the checkpoint's θ, so it computes what the checkpoint computed; over a
    checkpoint whose loop is not implicit it is refused, by the loop (row 7661)."""
    from dew.config import looping
    from dew.interop.pretrained import Pretrained

    loaded = Pretrained.load(str(QWEN35), dtype="float32", attention_impl="reference")
    implicit = looping(loaded, 3, (1, 3), implicit=True)
    (held,) = [leaf for name, leaf in named_leaves(implicit.variables).items() if "loop_theta" in name]
    theta = np.random.default_rng(7661).uniform(0.1, 0.9, held.shape).astype(np.float32)
    implicit = dataclasses.replace(implicit, variables=jax.tree_util.tree_map_with_path(
        lambda path, leaf: theta if "loop_theta" in jax.tree_util.keystr(path) else leaf, implicit.variables))
    asked = dataclasses.replace(run(Path(), None, None), loop_layers=None, loop_implicit=False,
                                loop_theta_per_pass=True)
    per_pass = asked.looped(implicit)
    assert per_pass.model.loop.theta_passes == 3 and per_pass.model.loop.steps == 3
    (rows,) = [leaf for name, leaf in named_leaves(per_pass.variables).items() if "loop_theta" in name]
    np.testing.assert_array_equal(np.asarray(rows), np.stack([theta] * 2))
    ids = jnp.arange(12, dtype=jnp.int32).reshape(2, 6)
    np.testing.assert_array_equal(np.asarray(per_pass.model.apply(per_pass.variables, ids)),
                                  np.asarray(implicit.model.apply(implicit.variables, ids)))
    plain = looping(loaded, 3, (1, 3))
    with pytest.raises(ValueError):
        asked.looped(plain)
    # Held as shapes alone, as a resume loads its source, its θ rows as its shape.
    shaped = dataclasses.replace(implicit, variables=jax.tree.map(
        lambda leaf: jax.ShapeDtypeStruct(np.shape(leaf), np.asarray(leaf).dtype), implicit.variables))
    rowed = asked.looped(shaped).variables
    (shape,) = [leaf for name, leaf in named_leaves_of(rowed) if "loop_theta" in name]
    assert isinstance(shape, jax.ShapeDtypeStruct) and shape.shape == (2, *held.shape)
    # Its channels placed as its one θ's, over a mesh, its rows whole.
    mesh = jax.make_mesh((2,), ("x",), devices=jax.devices()[:2])
    channels = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("x"))
    placed = dataclasses.replace(implicit, variables=jax.tree_util.tree_map_with_path(
        lambda path, leaf: jax.ShapeDtypeStruct(np.shape(leaf), np.asarray(leaf).dtype, sharding=channels)
        if "loop_theta" in jax.tree_util.keystr(path) else leaf, shaped.variables))
    (rowed_placed,) = [leaf for name, leaf in named_leaves_of(asked.looped(placed).variables)
                       if "loop_theta" in name]
    assert rowed_placed.sharding.spec == jax.sharding.PartitionSpec(None, "x")
    # A checkpoint holding rows for three passes keeps them, run at two.
    fewer = dataclasses.replace(asked, loop_steps=2).looped(per_pass)
    assert fewer.model.loop.theta_passes == 3 and fewer.model.loop.steps == 2
    np.testing.assert_array_equal(np.asarray(fewer.model.apply(fewer.variables, ids)),
                                  np.asarray(per_pass.model.with_passes(2).apply(per_pass.variables, ids)))
    with pytest.raises(ValueError):
        dataclasses.replace(asked, loop_steps=4).looped(per_pass)


@pytest.mark.parametrize("phase", range(len(STEPS)), ids=[f"{passes}-passes" for passes in STEPS])
def test_each_phases_step_of_the_decision_run_is_a_fixed_count_runs_step_and_counts_its_flops(tmp_path,
                                                                                               phase):
    """From the same state over the same batch, each phase's step of the phased decision run computes what
    a run whose loop runs that phase's count computes: its loss and its stepped parameters the same bits (the
    ask's invariant 1); and the FLOPs it counts are that run's, so its FLOP count follows the phase's count
    (invariant 5); and its offline view of an update is that run's. A cell a phase: each compiles its own
    phase's programs and its fixed-count run's, four, where one cell of both compiled eight (300 s in CI)."""
    from dew.batch import PHASE as READ_IN
    from dew.config import TrainerConfig
    from dew.data.dataset import DataPartition
    from dew.distributed.batches import split_fields
    from dew.training import Layout, MeshSpec
    from dew.training.inspection import Inspection, inspected

    settings = TrainerConfig(checkpoint_dir=str(tmp_path / "runs"), steps=4, log_every=1,
                             eval_every=None, checkpoint_every=None, checkpoint_final=False, multi_host=False,
                             mesh=MeshSpec(fsdp=2), layout=Layout(min_shard=64))

    def built(steps, data, name: str):
        """The run's trainer and its training read, as `train` builds them."""
        prepared = dataclasses.replace(run(tmp_path, steps, data), trainer=settings).prepare()
        readied, loader, _, _, _ = prepared.run.readied(prepared.objective, prepared.dataset, None, (), name)
        trainer = readied.built(prepared.objective, readied.trainer.mesh, None)
        # Its optimizer built over the run's length, as `fit` builds it.
        trainer.scheduled(settings.horizon.steps)
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
    passes = STEPS[phase]
    # The count the step runs is its batch's phase's, read on the host; the step reads the rest.
    assert stepping.phase_passes.at(batches[phase]) == passes
    batch, _ = split_fields(batches[phase])
    state, _, _ = stepping.place()
    stepped, loss, *_ = stepping.compile(state, batch)(state, batch)
    alone, _ = built(passes, asked(tickets(tmp_path, "first", "billing")), f"alone-{passes}")
    start, _, _ = alone.place()
    own, own_loss, *_ = alone.compile(start, batch)(start, batch)
    assert np.isfinite(np.asarray(loss)) and np.array_equal(np.asarray(loss), np.asarray(own_loss))
    assert jax.tree.all(jax.tree.map(lambda a, b: bool(np.array_equal(a, b)),
                                     jax.device_get(stepped.variables), jax.device_get(own.variables)))
    assert stepping.step_flops == alone.step_flops is not None
    # Its offline view of one update over the phase's batch is that run's view (`inspected`, at the
    # count of the phase the batch was read in, not the run's last).
    for trainer in (stepping, alone):
        trainer.inspection = Inspection(paths=("*",), targets=("updates",))
    assert inspected(stepping, batches[phase]) == inspected(alone, batch)


def test_a_phased_runs_later_phases_are_named_from_the_first_batch_its_loop_read(tmp_path):
    """A run's later phases are named (`later_phases`) from the batch its loop hands its first step, its
    work counted and its rows marked as the loop reads it (`marked`, `Trainer._counted_batch`), counted no
    more: the second phase, whose loop runs another pass count, is another program. Counted again, its
    telemetry columns were refused as reserved before a Qwen3.5-4B run's first step on a GH200."""
    from dew.data.dataset import DataPartition
    from dew.training.ahead import first_step_batch, later_phases
    from dew.training.distributed import marked

    stated = run(tmp_path, STEPS, phased(tmp_path))
    prepared = dataclasses.replace(stated, trainer=dataclasses.replace(stated.trainer,
                                                                       steps=4)).prepare()
    readied = prepared.run.readied(prepared.objective, prepared.dataset, None, (), "named")
    config, loader, steps = readied[0], readied[1], readied[4]
    trainer = config.built(prepared.objective, config.trainer.mesh, prepared.rollout)
    first = first_step_batch(trainer, loader, steps, loader.examples if steps is None else None)
    assert later_phases(trainer, marked(DataPartition(), trainer._counted_batch)(first)) == (1,)


def test_a_phased_runs_later_phase_compiles_ahead_as_the_run_lowers_it_at_its_boundary(tmp_path):
    """A run's later phases compile ahead through its record, trained by `dew train` with its target the
    phases after the first (`AheadTarget.phases`, `later_phases`, by the trainer's own key of each phase's
    batch), over its host's GPUs (here a serialized topology of eight compile-only H100s, as many as the
    run's row shards, so each reads its phases' rows as the run reads them), ending with status 0 and
    training nothing. Into the run's persistent cache, which a device's run reads as tuned, a step compiles
    only with every choice the GPU kind's runs tuned (`report.for_devices`): a host none ran on prepares
    none, its run compiling them on the device (row 6046), each said by its program (`CompiledAhead.untuned`).
    Into a file over the host's own topology, which a run takes, the step its resume would take is refused
    alike, not preparable until such a run has tuned it (`report.Untuned`): the program the prepare compiled
    for its phase."""
    from jax.experimental import topologies

    from dew.nn.kernels.generation import compile_only_gpus
    from dew.nn.kernels.options import program_digest
    from dew.nn.kernels.templates import report
    from dew.training.ahead import CompiledAhead
    from dew.training.targets import AheadTarget

    topology = tmp_path / "topology.bin"
    host = compile_only_gpus("sm90", 8, 40 * 2**30)
    topology.write_bytes(topologies.TopologyDescription(list(host)).serialize())
    data = phased(tmp_path)
    phases_target = AheadTarget("sm90", 8, topology=str(topology), phases=(1,))
    # The topology states its GPUs' memory: a capacity beside it, which nothing would read, is refused.
    with pytest.raises(ValueError):
        dataclasses.replace(phases_target, capacity_gib=30.0).compile_only()
    # Into the cache, the phases named and, with no path, the first step and each later phase's: none
    # preparable on a host no H100's run left its tuned choices beside.
    prepared_target = AheadTarget("sm90", 8, topology=str(topology))
    untuned = {}
    for name, target in (("ahead", phases_target), ("prepared", prepared_target)):
        with pytest.raises(CompiledAhead) as cached:
            trained(run(tmp_path, STEPS, data), tmp_path, name, lowering=target,
                    trainer={"steps": 4})
        assert cached.value.code == 0 and cached.value.path is None and cached.value.ahead is None
        assert not (tmp_path / "runs" / name).exists()
        untuned[name] = cached.value.untuned
    # The phase's step, and the first step beside it.
    assert len(untuned["ahead"]) == 1 and len(set(untuned["prepared"])) == 2
    trained(run(tmp_path, STEPS, data), tmp_path, "stopped", trainer={"steps": 3,
                                                                      "checkpoint_every": 1})
    resumed_target = AheadTarget("sm90", 8, path=str(tmp_path / "resumed.ahead"), topology=str(topology))
    with pytest.raises(ValueError) as resumed:
        trained(run(tmp_path, STEPS, data), tmp_path, "stopped", lowering=resumed_target,
                trainer={"steps": 4})
    refused = resumed.value.__cause__
    assert isinstance(refused, report.Untuned)
    assert untuned["ahead"] == (program_digest(refused.lowered),)
    assert untuned["ahead"][0] in untuned["prepared"]
