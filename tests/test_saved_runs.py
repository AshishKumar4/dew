"""Runs saved by earlier mains, read at this one (row 2282).

The corpus (tests/fixtures/saved_runs, written by tools/saved_runs_reference.py) holds, for each main that
recorded it, a saved run of each run kind (`saved_run_kinds.SAVED_KINDS`): its run.json and its last
checkpoint, a recording that is committed or pinned in the fetched store (`reference_error.recording`), and
the inputs its runs read that the repo has since changed (`INPUTS`), read as they were (`checkout_of`).
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
from process_support import assert_exits_cleanly
from reference_error import recording
from saved_run_kinds import SAVED_KINDS, answered, computed, trained_finite
from saved_run_products import unpacked

from dew.checkpoints import Checkpoints
from dew.config import RunConfig
from dew.inference.tasks import SavedRun
from dew.registry import objectives
from dew.training import LocalTracker

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

USERS_PRODUCTS = 3
"""The evidence version from which a generation records its products in a user's process
(`saved_runs_reference.EVIDENCE`); an earlier one recorded them in the suite's wide CPU arithmetic, and
they are read again in it (`computed`'s `wide`): a product of float32 operands there multiplies in
float64 (row 5442)."""


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


def test_every_task_a_registered_objective_saves_its_runs_as_is_a_recorded_kinds(tmp_path):
    """Every kind the corpus's writer saves (`SAVED_KINDS`) has a run a main recorded (`SAVED`), and every
    task a registered objective saves its runs as (`dew.registry.objectives`' `saved_task`) is the task of
    one of them, so a run of it a main saved loads back and answers
    (`test_a_run_an_earlier_main_saved_loads_and_answers`); a masked-diffusion run's task had no kind, and
    a kind the writer saves with no recording would pass a gate that read the writer's list (train 669's
    review, row 4674)."""
    def built(kind: str):
        # Each kind's run built in a directory of its own, as the corpus's writer builds it.
        (tmp_path / kind).mkdir()
        return SAVED_KINDS[kind](tmp_path / kind, drawn=False)

    recorded = {kind for _, kind in SAVED}
    assert set(SAVED_KINDS) <= recorded, sorted(set(SAVED_KINDS) - recorded)
    declared = {objectives[name].saved_task for name in objectives} - {None}
    saved = {objectives[built(kind).objective.name].saved_task for kind in SAVED_KINDS}
    assert declared <= saved, sorted(task.__name__ for task in declared - saved)


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


INPUTS = "inputs"
"""A generation's directory of the inputs its runs read that the repo has since changed, each at its path
in the checkout (the Gemma 3 fixture's tokenizer, which the repair of its mask token's row changed after
8a7909b0f068 trained over it): what was saved with a run, as it read them, is part of its record."""


def checkout_of(tmp_path: Path, recorded: str) -> Path:
    """The checkout as the main `recorded` read its runs' sources: this one, where its generation keeps no
    inputs of its own (`INPUTS`); else a mirror of it under `tmp_path`, every entry the checkout's own but
    those inputs, each its generation's copy. A record names a source by its path in the repo, which a
    run reads from the checkout's root."""
    root = Path(__file__).resolve().parents[1]
    kept = CORPUS / recorded / INPUTS
    if not kept.is_dir():
        return root
    mirror = tmp_path / "checkout"
    if mirror.is_dir():
        return mirror

    def opened(directory: Path, original: Path) -> None:
        # A directory of the mirror whose every entry is the checkout's, to replace one at a time.
        directory.mkdir()
        for entry in original.iterdir():
            (directory / entry.name).symlink_to(entry)

    opened(mirror, root)
    for given in sorted(path for path in kept.rglob("*") if path.is_file()):
        place = given.relative_to(kept)
        here, there = mirror, root
        for part in place.parts[:-1]:
            here, there = here / part, there / part
            if here.is_symlink():
                here.unlink()
                opened(here, there)
        (here / place.name).unlink(missing_ok=True)
        (here / place.name).write_bytes(given.read_bytes())
    return mirror


def extracted(tmp_path: Path, recorded: str, kind: str) -> Path:
    """The run directory of `kind`'s saved run, unpacked under `tmp_path`."""
    with archived(recorded, kind) as packed:
        packed.extractall(tmp_path / "runs", filter="data")
    return tmp_path / "runs" / kind


@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=IDS)
def test_a_run_an_earlier_main_saved_trained_to_finite_losses(tmp_path, recorded, kind):
    """Each run of the corpus logged finite losses and gradient norms at every step it logged them
    (`saved_run_kinds.trained_finite`): a run trained to NaN is no reference, and every later main's reads
    of it would compare NaN to NaN."""
    trained_finite(extracted(tmp_path, recorded, kind))


def test_a_run_that_logged_a_loss_not_finite_is_refused_as_a_reference(tmp_path):
    """`trained_finite` refuses a run whose logged loss or gradient norm is NaN or infinite, at any step,
    and one that logged none, which no reference can be; a run of finite ones reads back its steps'."""
    for logged in ({1: {"train/loss": 1.4}, 2: {"train/loss": float("nan")}},
                   {1: {"train/loss": 1.4, "train/grad_norm": float("inf")}}, {1: {"train/accuracy": 0.5}}):
        run = tmp_path / f"run-{len(list(tmp_path.iterdir()))}"
        with LocalTracker(run / "tracking") as tracker:
            for step, scalars in logged.items():
                tracker.log(scalars, step)
        with pytest.raises(ValueError):
            trained_finite(run)
    with LocalTracker(tmp_path / "finite" / "tracking") as tracker:
        tracker.log({"train/loss": 1.4, "train/grad_norm": 0.45, "train/accuracy": 0.3}, 1)
    assert trained_finite(tmp_path / "finite") == {1: {"train/loss": 1.4, "train/grad_norm": 0.45}}


@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=IDS)
def test_a_run_an_earlier_main_saved_loads_and_answers(tmp_path, monkeypatch, recorded, kind):
    """A kind's run saved at an earlier main rebuilds from its record, in the compute it recorded, and,
    where its objective declared a saved task then and the run recorded that declaration, pins its last
    checkpoint and loads as that task, which answers a request. A run whose objective declared none at its
    main recorded none (the corpus's JEPA runs, saved before a JEPA encoder had a release: row 4313,
    `dew.inference.models.SavedModel`), so pinning it is refused; resuming it saves a step that declares
    one (`test_a_run_an_earlier_main_saved_resumes_from_its_checkpoint`)."""
    run = extracted(tmp_path, recorded, kind)
    monkeypatch.chdir(checkout_of(tmp_path, recorded))
    recorded_run = RunConfig.load(str(run))
    seen = evidence(recorded, kind)
    task = objectives[recorded_run.objective.name].saved_task
    with Checkpoints(str(run)) as checkpoints:
        last = checkpoints.latest
        declared = checkpoints.artifact(last) is not None
    if seen is not None:
        assert recorded_run.compute_dtype() == seen["compute"]
        if seen["version"] >= PRODUCTS:
            # A run that declared its task recorded what one request multiplied, and one that declared
            # none recorded nothing.
            assert (seen["products"] is not None) == declared
        if declared and seen["version"] >= PRODUCTS:
            # The task loaded multiplies as the one the run saved did: one request's products, module by
            # module, in the dtypes they multiplied in then.
            found = computed(run, wide=seen["version"] < USERS_PRODUCTS)
            assert found["products"] == unpacked(seen["products"], found["packs"])
    if declared:
        saved = SavedRun.pinned(str(run))
        assert saved.step == last
        answered(task.from_run(str(run)))
    else:
        with pytest.raises(ValueError):
            SavedRun.pinned(str(run))


@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=IDS)
def test_a_run_an_earlier_main_saved_exports_and_loads_back_answering_as_its_run(tmp_path, recorded, kind):
    """A kind's run saved at an earlier main, exported by `dew export` where its checkpoint declared its task,
    loads back through `dew.pipeline` as the task its run loads (`from_run`) and answers one request as that
    task does, leaf for leaf (MF01). A run that declared none (the corpus's JEPA runs, saved before a JEPA
    encoder had a release) is refused by the export, as `SavedRun.pinned` refuses it."""
    from option_witness import answered_alike, answers

    from dew._cli.main import main
    from dew.inference.loading import pipeline

    run = extracted(tmp_path, recorded, kind)
    task = objectives[RunConfig.load(str(run)).objective.name].saved_task
    with Checkpoints(str(run)) as checkpoints:
        declared = checkpoints.artifact(checkpoints.latest) is not None
    release = tmp_path / "release"
    if not declared:
        with pytest.raises(ValueError):
            main(["export", str(run), str(release)])
        return
    assert main(["export", str(run), str(release)]) == 0
    assert answered_alike(answers(pipeline(str(release))), answers(task.from_run(str(run))))

@pytest.mark.parametrize(("recorded", "kind"), SAVED, ids=IDS)
def test_a_run_an_earlier_main_saved_is_refused_where_it_resumes_its_data(tmp_path, monkeypatch, recorded,
                                                                          kind):
    """A kind's run saved at an earlier main, resumed from its last checkpoint, is refused before it trains a
    step: its data position was written under a data plan this Dew reads none of (`dew.data.plan.Plan.check`,
    `dew._position.decode`), which the Dew that wrote it resumes, or a new run goes on from its weights."""
    run = extracted(tmp_path, recorded, kind)
    recorded_run = RunConfig.load(str(run))
    with Checkpoints(str(run)) as checkpoints:
        last = checkpoints.latest
    here = SAVED_KINDS[kind](tmp_path, drawn=False)
    # A source the repo holds is recorded by its path in the repo (a decision run's backbone), read from
    # the checkout's root as the run that wrote it read it.
    monkeypatch.chdir(checkout_of(tmp_path, recorded))
    # A run counted in examples reads on to its bound; one counted in steps, one step more.
    steps = {} if recorded_run.trainer.examples is not None else {"steps": last + 1}
    trainer = dataclasses.replace(recorded_run.trainer, checkpoint_dir=str(tmp_path / "runs"), stop=None,
                                  resume="require", compilation_cache_dir=None, multi_host=False, **steps)
    sources = {"data": here.data, "validation": here.validation}
    with pytest.raises(ValueError):
        dataclasses.replace(recorded_run, trainer=trainer, **sources).prepare().train(kind)
    stepped = [step for step, _ in LocalTracker(run / "tracking").scalars()]
    assert max(stepped) <= last


def rebuilt(tmp_path: Path, monkeypatch, recorded: str, kind: str, **stated) -> tuple[Path, int, object]:
    """`kind`'s saved run of `recorded` unpacked, its last step, and the run that made it built again, as
    `dew train` builds it from its Python or its flags (`SAVED_KINDS`), not read back from its record,
    `stated` set on it, to resume that step and train one more: a source the repo holds named by its
    path in the repo, as the corpus records it, read from the checkout's root."""
    run = extracted(tmp_path, recorded, kind)
    recorded_run = RunConfig.load(str(run))
    with Checkpoints(str(run)) as checkpoints:
        last = checkpoints.latest
    monkeypatch.chdir(checkout_of(tmp_path, recorded))
    clocked = recorded_run.trainer.examples is not None
    trainer = dataclasses.replace(recorded_run.trainer, checkpoint_dir=str(tmp_path / "runs"), stop=None,
                                  **({} if clocked else {"steps": last + 1}), resume="require",
                                  compilation_cache_dir=None, multi_host=False)
    here = SAVED_KINDS[kind](tmp_path, drawn=False)
    sourced = {"pretrained": recorded_run.pretrained} if hasattr(recorded_run, "pretrained") else {}
    return run, last, dataclasses.replace(here, trainer=trainer, **sourced, **stated)


def test_reading_a_saved_run_opens_no_device_before_its_process_is_prepared(tmp_path):
    """`dew train run.json` reads the run before it prepares its process (`RunConfig.run`'s
    `prepare_process`, which joins a pool of processes before any device is opened), so reading a saved
    decision run's record opens none, whatever its record leaves to its checkpoints: the writing a run saved
    before row 3867 left to its layout is read where the run resumes (`resumed`, `check_resumed`), after
    the process is prepared, in a fresh interpreter (train 589's review)."""
    # A saved decision run's record as main before row 3867 wrote it, its writing left to its layout.
    run = extracted(tmp_path, *next((recorded, kind) for recorded, kind in SAVED if kind == "decision"))
    record = json.loads((run / "run.json").read_text())
    (run / "run.json").write_text(json.dumps({**record, "fields": {**record["fields"], "option_text": None}}))
    assert_exits_cleanly(f"""
import socket

import jax
from dew.config import RunConfig

RunConfig.load({str(run)!r})
# A pool of one process joined after the read, as `prepare_process` joins one: refused once a device is open.
with socket.socket() as free:
    free.bind(("localhost", 0))
    port = free.getsockname()[1]
jax.distributed.initialize(f"localhost:{{port}}", num_processes=1, process_id=0)
jax.distributed.shutdown()
""")


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("kind", ["decision", "decision-lettered"])
def test_a_decision_run_this_dew_saves_resumes_when_its_command_runs_again_or_from_its_record(tmp_path,
                                                                                              kind):
    """A decision run saved by this Dew, its option text left to its layout (a release's named, a fresh
    lettered head's described), records the writing its build resolved, as a pinned source's step is, so
    a later Dew's default writes none of its rows otherwise (row 3867); it resumes when its command runs
    again and when read back from its record. A record a Dew before that wrote holds None, which reading
    it leaves (no checkpoint is read before its process is prepared, train 589's review) and its resume
    reads off its checkpoint's layout (`resumed`, `dew.config.earlier_option_writing`), and it resumes
    either way."""
    from run_kinds import trained

    from dew.decision.config import DecisionRunConfig

    first = SAVED_KINDS[kind](tmp_path, drawn=False)
    assert isinstance(first, DecisionRunConfig) and first.option_text is None
    trained(first, tmp_path, kind, trainer={"name": kind, "checkpoint_final": True})
    directory = tmp_path / "runs" / kind
    written = {"decision": "named", "decision-lettered": "described"}[kind]
    saved = json.loads((directory / "run.json").read_text())
    assert saved["fields"]["option_text"] == written
    unresolved = json.dumps({**saved, "fields": {**saved["fields"], "option_text": None}})
    for steps in (2, 3):
        (directory / "run.json").write_text(unresolved)
        loaded = DecisionRunConfig.load(str(directory))
        assert loaded.option_text is None and loaded.resumed().option_text == written
        run = first if steps == 2 else loaded
        trained(run, tmp_path, kind, trainer={"name": kind, "steps": steps, "checkpoint_final": True})
        assert str(steps) in {path.name for path in directory.iterdir()}
        assert json.loads((directory / "run.json").read_text())["fields"]["option_text"] == written


def test_a_packed_projections_products_count_at_each_of_its_members():
    """A task that serves a group of projections packed (`dew.nn.protocols.ProjectionGroup`: a decoder's
    q, k and v as `qkv_proj`) multiplies them in one product, which the evidence counts at each member
    (`unpacked`), so a run a main saved before the task packed them compares with one it packs: the same
    dtypes at the same members. A member whose dtype moved still differs; a module of the same name that
    no group declares (a linear attention's own `qkv_proj`) keeps its path; and two groups of one packed
    name unpack each to its own members."""
    from saved_run_products import unpacked

    packs = {"layers_0/self_attn/qkv_proj": ["layers_0/self_attn/q_proj", "layers_0/self_attn/k_proj",
                                             "layers_0/self_attn/v_proj"],
             "layers_0/mlp/gate_up_proj": ["layers_0/mlp/gate_proj", "layers_0/mlp/up_proj"],
             "layers_1/self_attn/qkv_proj": ["layers_1/self_attn/q_proj", "layers_1/self_attn/kv_proj"]}
    then = {"layers_0/self_attn/q_proj": ["float64"], "layers_0/self_attn/k_proj": ["float64"],
            "layers_0/self_attn/v_proj": ["float64"], "layers_0/mlp/gate_proj": ["float32"],
            "layers_0/mlp/up_proj": ["float32"], "layers_1/self_attn/q_proj": ["float64"],
            "layers_1/self_attn/kv_proj": ["float64"], "layers_2/lightning/qkv_proj": ["float64"]}
    now = {"layers_0/self_attn/qkv_proj": ["float64"], "layers_0/mlp/gate_up_proj": ["float32"],
           "layers_1/self_attn/qkv_proj": ["float64"], "layers_2/lightning/qkv_proj": ["float64"]}
    assert unpacked(now, packs) == unpacked(then, packs) == then
    moved = {**then, "layers_0/mlp/up_proj": ["bfloat16"]}
    assert unpacked(moved, packs) != unpacked(now, packs)


def test_a_called_product_is_read_at_the_scope_that_called_it(tmp_path, monkeypatch):
    """A product in a called program is read at the scope of the call that runs it, as XLA inlines it: a wide
    sum's (`precision.accumulated` on a CPU in the wide arithmetic the corpus's earlier generations recorded
    in, one private function traced once for its types, its own location naming no scope) at its caller's,
    and one under a scope of the called program's own at that (row 3193)."""
    import jax.numpy as jnp
    from saved_run_products import SCOPE, products

    from dew.nn.precision import WIDE_CPU_ARITHMETIC, accumulated

    monkeypatch.setenv(WIDE_CPU_ARITHMETIC, "1")

    @jax.jit
    def block(x, weight):
        with jax.named_scope(SCOPE + "layer2|inner"):
            return x @ weight

    def program(x, first, second):
        with jax.named_scope(SCOPE + "layer1"):
            hidden = accumulated(jnp.dot, x, first)
        with jax.named_scope(SCOPE + "layer2"):
            return block(hidden.astype(jnp.bfloat16), second)

    with jax.default_device(jax.devices("cpu")[0]):
        x, w = jnp.ones((4, 8), jnp.float32), jnp.ones((8, 8), jnp.float32)
        jax.config.update("jax_dump_ir_to", str(tmp_path))
        try:
            jax.jit(program)(x, w, w.astype(jnp.bfloat16)).block_until_ready()
        finally:
            jax.config.update("jax_dump_ir_to", "")
    texts = [path.read_text() for path in sorted(tmp_path.glob("*.mlir"))]
    assert products(texts) == {"layer1": ["float64"], "layer2/inner": ["bfloat16"]}


def test_a_requests_products_are_named_by_their_variable_scopes():
    """The products the saved-run evidence compares (`saved_run_products`) are named by the variable scope
    of the module that runs them in the task's tree, whichever method or class runs them: a helper
    method's product is its owner's, a shared scope's its sharer's, two siblings' apart, roots of one
    class applied over branches of the tree apart, a dozen in one request, a renamed class's the same;
    and a projection whose dtype moves moves alone, and a branch whose trunk is held frozen, read from two
    collections, its own. Two branches alike in every leaf, which no root's reading tells apart, are
    refused."""
    import jax.numpy as jnp
    from flax import linen as nn
    from saved_run_products import products, scoping

    from dew.batch import freeze, part, thaw

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

    def read(roots: dict[str, nn.Module], frozen: bool = False) -> dict[str, list[str]]:
        tree = {"params": {name: root.init(jax.random.key(0), x)["params"] for name, root in roots.items()}}
        if frozen:
            # Each root's right projection held frozen, as a run's frozen trunk is (`dew.batch.freeze`).
            tree = freeze(tree, lambda path: "right" not in path)

        def request(tree, x):
            # Each root reads its trunk whole (`thaw`), as a run's model reads its frozen trunk.
            return sum(root.apply(thaw(part(tree, name)), x) for name, root in roots.items())

        with nn.intercept_methods(scoping(tree)):
            lowered = jax.jit(request).lower(tree, x)
        return products([lowered.as_text(debug_info=True)])

    assert read({"pair": Pair(jnp.bfloat16)}) == {"pair/left": ["float32"], "pair/right": ["bfloat16"]}
    assert read({"pair": Shared(jnp.bfloat16)}) == read({"pair": Pair(jnp.bfloat16)})
    assert read({"pair": Renamed(jnp.bfloat16)}) == read({"pair": Pair(jnp.bfloat16)})
    assert read({"first": Pair(jnp.float32), "second": Pair(jnp.bfloat16, width=8)}) == {
        "first/left": ["float32"], "first/right": ["float32"], "second/left": ["float32"],
        "second/right": ["bfloat16"]}
    # A dozen roots of one class applied one after another in one request, each over its own branch, each
    # freed before the next is made (which Python may make at a freed one's address): each named by its
    # own branch, never an earlier root's (row 2867).
    dtypes = [jnp.float32 if index % 2 else jnp.bfloat16 for index in range(12)]
    assert read({f"r{index}": Pair(dtype, width=4 + index) for index, dtype in enumerate(dtypes)}) == {
        f"r{index}/{side}": ["float32" if side == "left" else jnp.dtype(dtype).name]
        for index, dtype in enumerate(dtypes) for side in ("left", "right")}
    with pytest.raises(ValueError):
        read({"first": Pair(jnp.float32), "second": Pair(jnp.bfloat16)})
    # A branch read from two collections, its trunk held frozen, is the branch its root reads.
    assert read({"first": Pair(jnp.float32), "second": Pair(jnp.bfloat16, width=8)}, frozen=True) == read(
        {"first": Pair(jnp.float32), "second": Pair(jnp.bfloat16, width=8)})



@pytest.mark.parametrize(("recorded", "kind"),
                         [(recorded, kind) for recorded, kind in SAVED if kind == "diffusion"],
                         ids=lambda part: str(part))
def test_a_diffusion_runs_release_stores_its_encoders_parameters_in_the_dtype_a_load_asks(tmp_path, recorded,
                                                                                            kind):
    """A diffusion run's native release loaded with a storage dtype stores every parameter its run's load does
    in it, its condition encoders' among them, and the rest as written: leaf for leaf the dtypes of the run
    loaded so (review of the run release)."""
    from dew._cli.main import main
    from dew.inference.loading import pipeline

    run = extracted(tmp_path, recorded, kind)
    assert main(["export", str(run), str(tmp_path / "release")]) == 0
    loaded, released = (pipeline(str(path), param_dtype="bfloat16") for path in (run, tmp_path / "release"))
    assert (jax.tree.map(lambda leaf: leaf.dtype, released.variables)
            == jax.tree.map(lambda leaf: leaf.dtype, loaded.variables))
