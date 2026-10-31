"""Runs saved by earlier mains, read at this one (row 2282).

The corpus (tests/fixtures/saved_runs, written by tools/saved_runs_reference.py) holds, for each main that
recorded it, a saved run of each run kind (`saved_run_kinds.SAVED_KINDS`): its run.json and its last
checkpoint, a recording that is committed or pinned in the fetched store (`reference_error.recording`).
This main reads each as a user's process would: the record rebuilds its run (`RunConfig.load`), the
checkpoint's inference declaration pins (`SavedRun.pinned`) where its objective declares a saved task,
which loads it (`from_run`) and answers one request, and the run resumes from its checkpoint for one step
more. Its data is the kind's own, written here, since a record names its data where the run that wrote it
read them. What a run serialized, recorded beside it where its main recorded it (`evidence`: its
parameters' and compute dtypes, its loss scale, the examples it consumed and its step's rung), is what
this main reads back: the record's compute, and the resumed run's parameters, scale, clock and rung."""

import dataclasses
import io
import json
import re
import tarfile
from pathlib import Path

import jax
import pytest
from reference_error import recording
from saved_run_kinds import SAVED_KINDS, answered, computed

from dew.checkpoints import Checkpoints
from dew.config import RunConfig
from dew.inference.tasks import SavedRun
from dew.registry import objectives

CORPUS = Path(__file__).with_name("fixtures") / "saved_runs"
SAVED = [(recorded.name, kind) for recorded in sorted(CORPUS.iterdir()) if recorded.is_dir()
         for kind in json.loads((recorded / "source.json").read_text())["kinds"]]
"""Each saved run: the main that recorded it, by its abbreviated commit, and its kind."""

IDS = [f"{recorded}-{kind}" for recorded, kind in SAVED]

ABSOLUTE = re.compile(rb"/(?:home|Users|root|tmp|mnt|private|var)/")
"""An absolute place of the machine that wrote a file: a home, a scratch or a mounted directory."""


PRODUCTS = 2
"""The evidence version from which a generation records the dtypes one request of each run's task
multiplies in, by module path (`saved_run_kinds.computed`); an earlier one's are not compared."""


def evidence(recorded: str, kind: str) -> dict | None:
    """What `kind`'s run the main `recorded` saved serialized (`saved_runs_reference.evidence`), beside
    the version its generation states (`version`), None where that main's generation records none: one
    recorded before its version (`source.json`'s `evidence`). A generation that states one holds every
    kind's."""
    source = json.loads((CORPUS / recorded / "source.json").read_text())
    if "evidence" not in source:
        return None
    seen = json.loads((CORPUS / recorded / f"{kind}.evidence.json").read_text())
    return {**seen, "version": source["evidence"]}


def archived(recorded: str, kind: str) -> tarfile.TarFile:
    """The archive of `kind`'s run the main `recorded` saved (`tools/saved_runs_reference.py`)."""
    stored = recording(f"tests/fixtures/saved_runs/{recorded}/{kind}.npz")["run"]
    return tarfile.open(fileobj=io.BytesIO(stored.tobytes()), mode="r:xz")


@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=IDS)
def test_a_saved_run_names_no_owner_time_or_place_of_the_machine_that_wrote_it(recorded, kind):
    """Each archive of the corpus names nothing of its writer's machine, which the fetched store
    publishes: its members hold no owner, user or group, and one time, and no file of it names an
    absolute directory, a scratch directory, a checkout or a home (`saved_runs_reference.anonymous`)."""
    with archived(recorded, kind) as packed:
        members = packed.getmembers()
        assert {(member.uid, member.gid, member.uname, member.gname, member.mtime) for member in members} == {
            (0, 0, "", "", 0)}
        for member in members:
            if member.isfile():
                found = packed.extractfile(member)
                assert found is not None and not ABSOLUTE.search(found.read()), member.name


def extracted(tmp_path: Path, recorded: str, kind: str) -> Path:
    """The run directory of `kind`'s saved run, unpacked under `tmp_path`."""
    with archived(recorded, kind) as packed:
        packed.extractall(tmp_path / "runs", filter="data")
    return tmp_path / "runs" / kind


@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=IDS)
def test_a_run_an_earlier_main_saved_loads_and_answers(tmp_path, recorded, kind):
    """A kind's run saved at an earlier main rebuilds from its record, in the compute it recorded, and,
    where its objective declares a saved task, pins its last checkpoint and loads as that task, which
    answers a request; an objective that declares none (a JEPA encoder's) records no inference
    declaration."""
    run = extracted(tmp_path, recorded, kind)
    recorded_run = RunConfig.load(str(run))
    seen = evidence(recorded, kind)
    task = objectives[recorded_run.objective.name].saved_task
    if seen is not None:
        assert recorded_run.compute_dtype() == seen["compute"]
        if task is not None and seen["version"] >= PRODUCTS:
            # The task loaded multiplies as the one the run saved did: one request's products, module by
            # module, in the dtypes they multiplied in then.
            assert computed(run) == seen["products"]
    with Checkpoints(str(run)) as checkpoints:
        last = checkpoints.latest
        declared = checkpoints.artifact(last) is not None
    assert declared == (task is not None)
    if task is not None:
        saved = SavedRun.pinned(str(run))
        assert saved.step == last
        answered(task.from_run(str(run)))


@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=IDS)
def test_a_run_an_earlier_main_saved_resumes_from_its_checkpoint(tmp_path, monkeypatch, recorded, kind):
    """A kind's run saved at an earlier main resumes from its last checkpoint for one step more, in the
    compute it trained in (a main that recorded no evidence: every corpus run's model draws fresh or loads
    a decision backbone, in float32 unless its model names a dtype; train 328's review of ledger 2500):
    the fit starts at that step, from that checkpoint, and trains the one step past it. Where its main
    recorded what it serialized (`evidence`), the resumed run holds its parameters in the dtypes it
    stored them in, scales its loss as it did, its count of finite steps going on, counts on from the
    examples it consumed (a run on the examples clock paused before its bound trains on to it), and
    compiles its step at the rung it was admitted at."""
    run = extracted(tmp_path, recorded, kind)
    recorded_run = RunConfig.load(str(run))
    seen = evidence(recorded, kind)
    compute = recorded_run.model.fields.get("dtype", "float32") if seen is None else seen["compute"]
    assert recorded_run.compute_dtype() == compute
    with Checkpoints(str(run)) as checkpoints:
        last = checkpoints.latest
    here = SAVED_KINDS[kind](tmp_path, drawn=False)
    # A source the repo holds is recorded by its path in the repo (a decision run's backbone), read from
    # the checkout's root as the run that wrote it read it.
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    # A run on the examples clock paused before its bound (`Pause`) trains on to it, its progress the
    # examples it consumed; a run counted in steps trains one step more.
    clocked = seen is not None and recorded_run.trainer.examples is not None
    trainer = dataclasses.replace(recorded_run.trainer, checkpoint_dir=str(tmp_path / "runs"), stop=None,
                                  **({} if clocked else {"steps": last + 1}), resume="require",
                                  compilation_cache_dir=None, multi_host=False)
    sources = {"data": here.data, "validation": here.validation}
    state = dataclasses.replace(recorded_run, trainer=trainer, **sources).prepare().train(kind)
    records = [json.loads(line) for line in (run / "tracking" / "records.jsonl").read_text().splitlines()]
    resumed = max(index for index, row in enumerate(records) if row["type"] == "FitStarted")
    started = records[resumed]["value"]
    assert (started["start_step"], started["resumed_from"] is not None) == (last, True)
    scalars = (run / "tracking" / "scalars.jsonl").read_text().splitlines()
    stepped = [json.loads(line)["step"] for line in scalars]
    assert max(stepped) > last if clocked else max(stepped) == last + 1
    if seen is not None:
        stored = sorted({str(leaf.dtype) for leaf in jax.tree.leaves(state.variables["params"])})
        assert stored == seen["params"]
        # The scale as it was, and its count of finite steps on from the recorded one, not from none.
        assert (None if state.scale is None else float(state.scale.scale)) == seen["scale"]
        assert (None if state.scale is None else int(state.scale.fin_steps)) == (
            None if seen["finite_steps"] is None else seen["finite_steps"] + int(state.step) - last)
        # The examples it consumed count on from the recorded ones to the bound.
        consumed = None if state.consumed is None else int(state.consumed)
        assert (consumed is None) == (seen["consumed"] is None)
        assert consumed is None or consumed > seen["consumed"]
        assert not clocked or consumed == recorded_run.trainer.examples
        # The model the resumed fit trained computes as the saved one did: the task its new step saves.
        task = objectives[recorded_run.objective.name].saved_task
        if task is not None and seen["version"] >= PRODUCTS:
            assert computed(run, int(state.step)) == seen["products"]
        # The step the resumed fit compiled, at the rung it was admitted at.
        compiled = [row["value"] for row in records[resumed:] if row["type"] == "StepCompiled"][-1]
        assert compiled["remat"] == seen["remat"]
        assert {name: (compiled["plan"] or {}).get(name) for name in seen["plan"]} == seen["plan"]


def test_roots_applied_one_after_another_are_each_named_by_their_own_branch():
    """Roots applied one after another, each over its own branch of the task's tree, each freed before the
    next is made, which Python may make at the freed one's address: each root's products are named by its
    own branch, never an earlier root's (row 2867: the reading kept each root's branch by its address,
    and a later root made there read an earlier one's)."""
    import types

    import jax.numpy as jnp
    import numpy as np
    from saved_run_products import products, scoping

    branches = {f"r{index}": {"w": np.zeros((index + 1, 2), np.float32)} for index in range(12)}
    scoped = scoping({"params": branches})

    class Root:
        """A root scope as the reading reads one: the variables it was applied over."""

        def __init__(self, params):
            self.params = params

        def variables(self):
            return {"params": self.params}

    def request(x):
        total = jnp.zeros_like(x)
        for params in branches.values():
            root = Root(params)
            context = types.SimpleNamespace(module=types.SimpleNamespace(
                scope=types.SimpleNamespace(root=root, path=())))
            total = total + scoped(lambda: x @ x, (), {}, context)
            del root, context
        return total

    found = products([jax.jit(request).lower(jnp.ones((2, 2))).as_text(debug_info=True)])
    assert found == {name: ["float32"] for name in branches}


def test_a_requests_products_are_named_by_their_variable_scopes():
    """The products the saved-run evidence compares (`saved_run_products`) are named by the variable scope
    of the module that runs them in the task's tree, whichever method or class runs them: a helper
    method's product is its owner's, a shared scope's its sharer's, two siblings' apart, two roots of one
    class applied over two branches of the tree apart, a renamed class's the same; and a projection whose
    dtype moves moves alone. Two branches alike in every leaf, which no root's reading tells apart, are
    refused."""
    import jax.numpy as jnp
    from flax import linen as nn
    from saved_run_products import products, scoping

    class Pair(nn.Module):
        right: jnp.dtype
        width: int = 4

        def setup(self):
            self.left = nn.Dense(self.width, dtype=jnp.float32, param_dtype=jnp.float32)
            self.right_ = nn.Dense(self.width, dtype=self.right, param_dtype=jnp.float32, name="right")

        def project(self, x):
            return self.right_(x)

        def __call__(self, x):
            return self.left(x)[..., :4] + self.project(x)[..., :4]

    class Shared(nn.Module):
        """A `Pair` in this module's own scope (`nn.share_scope`), as a release's decoder shares its text
        model's."""

        right: jnp.dtype

        def setup(self):
            self.inner = Pair(self.right)
            nn.share_scope(self, self.inner)

        def __call__(self, x):
            return self.inner(x)

    class Renamed(Pair):
        """`Pair` under another name, over the same variables."""

    x = jnp.ones((2, 4), jnp.float32)

    def read(roots: dict[str, nn.Module]) -> dict[str, list[str]]:
        tree = {"params": {name: root.init(jax.random.key(0), x)["params"] for name, root in roots.items()}}

        def request(tree, x):
            return sum(root.apply({"params": tree["params"][name]}, x) for name, root in roots.items())

        with nn.intercept_methods(scoping(tree)):
            lowered = jax.jit(request).lower(tree, x)
        return products([lowered.as_text(debug_info=True)])

    assert read({"pair": Pair(jnp.bfloat16)}) == {"pair/left": ["float32"], "pair/right": ["bfloat16"]}
    assert read({"pair": Shared(jnp.bfloat16)}) == read({"pair": Pair(jnp.bfloat16)})
    assert read({"pair": Renamed(jnp.bfloat16)}) == read({"pair": Pair(jnp.bfloat16)})
    assert read({"first": Pair(jnp.float32), "second": Pair(jnp.bfloat16, width=8)}) == {
        "first/left": ["float32"], "first/right": ["float32"], "second/left": ["float32"],
        "second/right": ["bfloat16"]}
    with pytest.raises(ValueError):
        read({"first": Pair(jnp.float32), "second": Pair(jnp.bfloat16)})
