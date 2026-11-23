"""A run that starts from a saved Dew run (`--pretrained <run directory>`, ledger 2713): its source is the
release `dew export` writes of the run's checkpoint (`dew.interop.saved_sources`), or a diffusion run's
own sampling task, at the step its selection pins, which its record names."""

import dataclasses
import json
import shutil
from pathlib import Path

import jax
import numpy as np
import pytest
from process_support import run_pool
from run_kinds import SEQ, captioned, decision, diffusion, lm, pictures, trained
from saved_run_kinds import over, windows
from token_support import write_token_corpus

from dew.batch import FROZEN
from dew.config import Diagnostics, ObjectiveConfig, SourceOptions
from dew.config.diffusion import DEFAULT_MODEL_CONFIG, DiffusionRunConfig
from dew.decision.config import DecisionRunConfig
from dew.inference import TextToImage
from dew.inference.tasks import SavedRun
from dew.interop.pretrained import load_diffusion_source
from dew.interop.saved_sources import export_run, released_run
from dew.kernels import Kernels
from dew.lora import LoRA
from dew.registry import ModelConfig
from dew.sampling.solvers import Euler


@pytest.mark.mesh(devices=8)
def test_a_decision_run_starts_from_a_saved_decision_run_as_from_its_export(tmp_path, monkeypatch):
    """A decision run named on the command line by a saved decision run (`--pretrained <run>`,
    `--source.step latest`) starts from the weights its export does (`dew export`):
    one step from each, alike in every other option, lands every leaf bit for bit alike, and its record
    names the run it started from and the step it read. Its record resumes on a host whose cache lacks
    the export (another node's, ledger 2774). A step beside a source that is no run is refused. A run
    naming a head of its own over the run draws it over the run's backbone and trains."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    first = decision(tmp_path, drawn=False)
    trained(first, tmp_path, "first", trainer={"checkpoint_final": True})
    source = tmp_path / "runs" / "first"
    exported = tmp_path / "exported"
    export_run(SavedRun.pinned(str(source)), exported)

    def started(pretrained, name: str, *flags: str):
        run = DecisionRunConfig.cli(["--pretrained", str(pretrained), *flags], default=first)
        return trained(run, tmp_path, name, trainer={"checkpoint_final": True})[0]

    from_run = started(source, "from_run", "--source.step", "latest")
    from_export = started(exported, "from_export")
    leaves_alike(from_run.variables, from_export.variables)
    recorded = DecisionRunConfig.load(str(tmp_path / "runs" / "from_run"))
    assert recorded.pretrained == str(source)
    assert recorded.source.step == SavedRun.pinned(str(source)).step
    # Named, the run resumes from its own checkpoint (`TrainerConfig.resumes`), reading its source without
    # weights, on a host whose cache holds no release.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "another_cache"))
    trained(recorded, tmp_path, "from_run",
            trainer={"name": "from_run", "steps": 2, "checkpoint_final": True})
    assert "2" in {path.name for path in (tmp_path / "runs" / "from_run").iterdir()}
    assert released_run(source, step="latest") == released_run(source)
    with pytest.raises(ValueError):
        released_run(exported, step="latest")
    with pytest.raises(ValueError):
        load_diffusion_source(str(source))
    # A run naming a head of its own over the Laya release draws it over the release's backbone, its rows
    # laid out by the release's own tokenizer, which the backbone alone does not hold, and trains.
    joint = ["head:joint-shape", "--head.width", "16", "--head.routing-layers", "1", "--head.layers", "1",
             "--head.heads", "2", "--head.feedforward", "32", "--max-len", "768"]
    drawn = started(source, "fresh_head", "--source.step", "latest", *joint)
    assert int(drawn.step) == 1


@pytest.fixture(scope="module")
def native_release_run(tmp_path_factory):
    """A decision run whose head's format holds no backbone of its own family (Laya's fresh head, whose
    format holds ModernBERT alone, over Qwen 3.5's MoE text decoder), trained a step and saved: its root,
    its config, its directory and its state, trained once for the cells reading it, which their mark keeps
    in one group (`shared_setup`, row 11722: one cell training it and the run from it took 131 s cold)."""
    root = tmp_path_factory.mktemp("native-release")
    decoder = Path(__file__).parent / "fixtures" / "hf" / "qwen38-moe-tiny"
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CACHE_HOME", str(root / "cache"))
        first = dataclasses.replace(decision(root, drawn=False), pretrained=str(decoder))
        state, _ = trained(first, root, "first", trainer={"checkpoint_final": True})
    return root, first, root / "runs" / "first", state


@pytest.mark.shared_setup("native_release_run")
def test_a_decision_run_whose_head_holds_no_backbone_of_its_family_exports_as_its_native_release(
        native_release_run, monkeypatch):
    """Such a run exports as its native release, which a run naming it (`--pretrained <run>`) starts from
    as from the run's own task: its head and backbone the first run's final ones; loaded without its
    weights, it holds their shapes, its projections packed as its weights are (row 4776: the release's
    record states no dtypes a family's load reads, and the start raised KeyError)."""
    from dew.decision.formats import format_unstated
    from dew.inference.tasks import saved_model
    from dew.interop import Pretrained

    root, first, source, state = native_release_run
    # The fixture's own cache, which its run and the run from it publish their release in.
    monkeypatch.setenv("XDG_CACHE_HOME", str(root / "cache"))
    assert format_unstated("laya", saved_model(SavedRun.pinned(str(source)).record, None).build()) is not None
    shapes = Pretrained.load(str(source), weights=False)
    assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(shapes.variables))
    run = DecisionRunConfig.cli(["--pretrained", str(source), "--source.step", "latest"], default=first)
    started = run.prepare().objective.init(jax.random.key(0))
    leaves_alike(started["params"], state.variables["params"])


@pytest.mark.shared_setup("native_release_run")
def test_a_decision_run_started_from_a_native_release_trains_records_it_and_answers(native_release_run,
                                                                                     monkeypatch):
    """A run started from that native release trains, its record naming the run and step it read, and its
    task answers (row 4776)."""
    from dew.decision import Choice, Decide

    root, first, source, _ = native_release_run
    monkeypatch.setenv("XDG_CACHE_HOME", str(root / "cache"))
    run = DecisionRunConfig.cli(["--pretrained", str(source), "--source.step", "latest"], default=first)
    second, _ = trained(run, root, "second", trainer={"checkpoint_final": True})
    assert int(second.step) == 1
    recorded = DecisionRunConfig.load(str(root / "runs" / "second"))
    assert recorded.pretrained == str(source) and recorded.source.step == SavedRun.pinned(str(source)).step
    team = Choice("Which team?", {"billing": "payments", "technical": "outages", "sales": "pricing"})
    answer = Decide.from_run(str(root / "runs" / "second"))("charged twice", {"team": team})["team"]
    assert answer.probabilities.shape == (3,) and np.all(answer.probabilities > 0)


@pytest.mark.parametrize("record", [
    {"run": "/runs/first", "step": "latest", "ema": None},
    {"run": "/runs/first", "step": True, "ema": None},
    {"run": "/runs/first", "step": 3, "ema": "false"},
    {"run": "/runs/first", "step": 3},
    {"run": "/runs/first", "step": None, "ema": None},
    {"run": "/runs/first", "step": [3], "ema": None},
    {"run": "/runs/first", "step": {"function": "dew.registry:to_record"}, "ema": None},
])
def test_a_release_record_of_another_shape_is_refused(record):
    """A cache release's record of the saved run it holds (`Held`) is a run's directory, a step (never a
    moving selection) and an EMA choice (a boolean or none): any other is refused, as a record its own
    write never makes; one of its shape reads back as written."""
    from dew.interop.saved_sources import Held

    with pytest.raises(ValueError):
        Held.read(record)
    held = Held("/runs/first", 3, None)
    assert held.record() == {"run": "/runs/first", "step": 3, "ema": None}
    assert Held.read(held.record()) == held


def test_a_run_started_from_a_saved_run_resumes_and_serves_on_hosts_whose_caches_lack_its_releases(
        tmp_path, monkeypatch):
    """A decision run started from a saved decision run (`--pretrained <run>`) reads its tokenizer within its
    source's release, a key under its training host's Dew cache, which each step's record names beside the
    saved run that release holds (`RELEASE_SOURCES`, the release's own `HELD_FILE`). Resumed on another host,
    whose cache lacks it, the run trains on, reading its own release of the source, and every step serves on
    a third host lacking both, by each reader of a run (`Decide.from_run`, `dew.pipeline`), as it served
    where it trained: the pin they share writes this host's release of the same saved run
    (`SavedRun.pinned`, `releases_here`), where the record's paths raised Hugging Face's repo-id error. On a
    host that lacks the saved run too, a reader of the weights alone (`dew.checkpoints.averaged`) reads them,
    and a reader that opens the record's paths is refused, naming the saved run."""
    from dew.checkpoints import averaged
    from dew.decision import Choice, Decide
    from dew.inference.loading import pipeline
    from dew.interop.saved_sources import HELD_FILE, RELEASE_SOURCES, RELEASES

    def cache(name: str) -> Path:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / name))
        return tmp_path / name / "dew" / RELEASES

    trained_on = cache("training_host")
    first = decision(tmp_path, drawn=False)
    trained(first, tmp_path, "first", trainer={"checkpoint_final": True})
    source = tmp_path / "runs" / "first"
    run = DecisionRunConfig.cli(["--pretrained", str(source), "--source.step", "latest"], default=first)
    trained(run, tmp_path, "second", trainer={"checkpoint_final": True})
    second = str(tmp_path / "runs" / "second")
    held = {"run": str(source), "step": SavedRun.pinned(str(source)).step, "ema": None}
    (release,) = SavedRun.at(second, 1).record[RELEASE_SOURCES]
    assert Path(release).parent == trained_on
    assert json.loads((Path(release) / HELD_FILE).read_text()) == held
    team = Choice("Which team?", {"billing": "payments", "technical": "outages", "sales": "pricing"})

    def answered(task) -> np.ndarray:
        return np.asarray(task("charged twice", {"team": team})["team"].probabilities)

    served = answered(Decide.from_run(second, step=1))
    # Another host, the training host's cache nowhere: the run resumes there and trains a second step.
    shutil.move(trained_on.parent.parent, tmp_path / "kept")
    resumed_on = cache("resuming_host")
    trained(run, tmp_path, "second", trainer={"steps": 2, "checkpoint_final": True})
    (later,) = SavedRun.at(second, 2).record[RELEASE_SOURCES]
    assert Path(later).parent == resumed_on
    stepped = answered(Decide.from_run(second, step=2))
    # A third host, lacking both releases: each step serves as it did, by each reader.
    cache("serving_host")
    np.testing.assert_array_equal(answered(Decide.from_run(second, step=1)), served)
    np.testing.assert_array_equal(answered(pipeline(second, step=2)), stepped)
    # A host lacking the saved run too: a reader of the weights alone reads them, and one that opens the
    # record's paths is refused, naming the saved run.
    shutil.move(source, tmp_path / "moved")
    cache("bare_host")
    # The run averaged with itself: its weights in fp32.
    soup = averaged([second, second], step=1)
    leaves_alike(soup.variables["params"], jax.tree.map(lambda leaf: np.asarray(leaf, np.float32),
                                                        SavedRun.at(second, 1).variables(ema=None)["params"]))
    with pytest.raises(FileNotFoundError) as refused:
        Decide.from_run(second, step=1)
    assert str(source) in str(refused.value)


@pytest.mark.mesh(devices=8)
def test_a_language_model_run_starts_from_a_saved_run_no_familys_layout_writes(tmp_path, monkeypatch):
    """A language model run whose decoder no family's layout holds (GPT-J's parallel layout under a gated
    MLP, its positions a learned table) exports as its native release, and loads (`Pretrained.load(run)`,
    at the attention a run asks and twice the context, its learned table keeping its rows, as every
    task resizes a cache) as its model in Dew's own layout on JAX's default device, its parameters the
    run's final ones, whose own export refuses, stating why; asked no weights, it reads their shapes and
    dtypes alone; on the host, its
    weights as host arrays read from storage, nothing transferred to a device on the way; a second run
    naming it (`--pretrained <run>`) starts from it and trains its step (row 4785: a token run's native
    release declared no task a run starts from)."""
    from flax import linen as nn

    from dew.config.lm import LMRunConfig
    from dew.inference.tasks import saved_model
    from dew.interop import Pretrained
    from dew.interop.decoder_parts import ExportRefused
    from dew.interop.layouts import source_layout

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    gated = ModelConfig("causal_transformer", {
        "emb_features": 16, "num_layers": 1, "num_heads": 2, "dtype": "float32", "parallel_residual": True,
        "shared_parallel_norm": True, "head_bias": True, "attention_bias": False, "mlp": "swiglu",
        "position_embedding": "learned"})
    first = dataclasses.replace(lm(tmp_path, drawn=False), model=gated)
    state, _ = trained(first, tmp_path, "first", trainer={"checkpoint_final": True})
    source = tmp_path / "runs" / "first"
    assert not source_layout(saved_model(SavedRun.pinned(str(source)).record, None).build())
    loaded = Pretrained.load(str(source), max_seq_len=2 * SEQ, attention_impl="xla")
    assert (loaded.model.max_seq_len, loaded.model.attention_impl) == (2 * SEQ, "xla")
    # Asked no mesh, its weights lie on JAX's default device, as a family's load places them, not on
    # every device of a default mesh (ledger 4798).
    assert {frozenset(leaf.sharding.device_set) for leaf in jax.tree.leaves(loaded.variables)} == {
        frozenset({jax.devices()[0]})}
    leaves_alike(loaded.variables["params"], state.variables["params"])
    # The model built at the larger context holds the parameters the run trained, its table's rows kept.
    ids = jax.ShapeDtypeStruct((1, 2), "int32")
    built = nn.unbox(jax.eval_shape(loaded.model.init, jax.random.key(0), ids))
    assert jax.tree.map(lambda leaf: leaf.shape, built["params"]) == jax.tree.map(
        lambda leaf: leaf.shape, loaded.variables["params"])
    with pytest.raises(ExportRefused):
        loaded.export()
    shapes = Pretrained.load(str(source), weights=False)
    assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(shapes.variables))
    assert jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), shapes.variables["params"]) == jax.tree.map(
        lambda leaf: (leaf.shape, leaf.dtype), state.variables["params"])
    # On the host, its weights read from storage as host arrays, no device holding them on the way (a
    # load placed them on the accelerator and fetched them back).
    with jax.transfer_guard_host_to_device("disallow_explicit"):
        host = Pretrained.load(str(source), on_host=True)
    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(host.variables))
    leaves_alike(host.variables["params"], state.variables["params"])
    run = dataclasses.replace(first, pretrained=str(source), source=SourceOptions(step="latest"),
                              model=ModelConfig("causal_transformer", {"dtype": "float32"}))
    second, _ = trained(run, tmp_path, "second", trainer={"checkpoint_final": True})
    assert int(second.step) == 1
    assert LMRunConfig.load(str(tmp_path / "runs" / "second")).pretrained == str(source)


def joint_flags() -> list[str]:
    """A chat-format decision run's flags: a joint head over Qwen 3.5's chat backbone, its rows framed as a
    conversation by the backbone's chat template."""
    from run_kinds import QWEN_TINY

    return ["--pretrained", str(QWEN_TINY), "head:joint-shape", "--head.width", "16",
            "--head.routing-layers", "1", "--head.layers", "1", "--head.heads", "2",
            "--head.feedforward", "32", "--max-len", "768"]


@pytest.fixture(scope="module")
def chat_decision_run(tmp_path_factory):
    """A chat-format decision run (`joint_flags`) trained a step and saved, its directory and its state:
    the source each of the run-from-run cells below names, trained once for them, which their mark keeps in
    one group of a split file (`shared_setup`, row 10777: main's cut, tests/test_cuts.json, parted the three
    into three groups, each training it again, 139 s for one in train 1139's CI)."""
    root = tmp_path_factory.mktemp("chat-decision")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("XDG_CACHE_HOME", str(root / "cache"))
        first = DecisionRunConfig.cli(joint_flags(), default=decision(root, drawn=False))
        state, _ = trained(first, root, "first", trainer={"checkpoint_final": True})
    return root / "runs" / "first", state


TEAM = {"billing": "payments", "technical": "outages", "sales": "pricing"}
"""The question the chat-format decision runs answer, `Choice("Which team?", TEAM)`."""


@pytest.mark.shared_setup("chat_decision_run")
def test_a_saved_chat_format_decision_runs_release_answers_as_its_run(chat_decision_run, tmp_path,
                                                                      monkeypatch):
    """A chat-format decision run's release, written by `dew export` and by its task's `save_pretrained`,
    reloads answering as the run's task serves its written weights, bit for bit, on the host (the run's
    own task is placed on its training mesh)."""
    from dew._cli.main import main
    from dew.decision import Choice, Decide
    from dew.decision.sources import PretrainedDecision

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    source, _ = chat_decision_run
    team = Choice("Which team?", TEAM)
    task = Decide.from_run(str(source))
    assert main(["export", str(source), str(tmp_path / "exported")]) == 0
    task.save_pretrained(tmp_path / "saved")
    served = Decide.of(PretrainedDecision.of(dataclasses.replace(task, variables=jax.tree.map(
        np.asarray, task.variables))))
    for written in ("exported", "saved"):
        reloaded = Decide.from_pretrained(tmp_path / written, attention_impl="xla")
        for state_text in ("charged twice", "is the site down?"):
            np.testing.assert_array_equal(reloaded(state_text, {"team": team})["team"].probabilities,
                                          served(state_text, {"team": team})["team"].probabilities)


@pytest.mark.shared_setup("chat_decision_run")
def test_a_decision_run_starts_from_a_saved_chat_format_decision_run(chat_decision_run, tmp_path,
                                                                    monkeypatch):
    """A saved chat-format decision run is a source a second decision run names (`--pretrained <run>`):
    the second run trains on from the first's final weights, head and backbone, its record names the run
    and step it read, and its task loads and answers (a run from a finished chat-format run, which the
    export's Laya-only layout refused: "Laya's layout is a marker row over a Hugging Face tokenizer", row
    4310)."""
    from dew.decision import Choice, Decide

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    source, state = chat_decision_run
    run = DecisionRunConfig.cli(["--pretrained", str(source), "--source.step", "latest", "--max-len", "768"],
                                default=decision(tmp_path, drawn=False))
    started = run.prepare().objective.init(jax.random.key(0))
    leaves_alike(started["params"]["head"], state.variables["params"]["head"])
    second, _ = trained(run, tmp_path, "second", trainer={"checkpoint_final": True})
    assert int(second.step) == 1
    recorded = DecisionRunConfig.load(str(tmp_path / "runs" / "second"))
    assert recorded.pretrained == str(source) and recorded.source.step == SavedRun.pinned(str(source)).step
    answer = Decide.from_run(str(tmp_path / "runs" / "second"))("charged twice",
                                                                {"team": Choice("Which team?", TEAM)})["team"]
    assert answer.probabilities.shape == (3,) and np.all(answer.probabilities > 0)


@pytest.mark.shared_setup("chat_decision_run")
def test_a_decision_run_naming_its_own_head_draws_it_fresh_over_a_saved_chat_format_runs_backbone(
        chat_decision_run, tmp_path, monkeypatch):
    """A run naming a head of its own over a saved chat-format decision run draws it fresh over the run's
    backbone, its head, layout and calibration set aside: it starts from the run's backbone and from the
    head a run drawing the same head over the base draws (row 4310's run-from-run, a fresh head), and
    trains, its task loading and answering (it was refused: "a decision release reads with its own")."""
    from dew.decision import Choice, Decide

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    source, state = chat_decision_run
    joint = joint_flags()
    fresh = ["--pretrained", str(source), "--source.step", "latest", *joint[2:]]
    drawn = DecisionRunConfig.cli(fresh, default=decision(tmp_path, drawn=False)).prepare().objective
    over_base = DecisionRunConfig.cli(joint, default=decision(tmp_path, drawn=False)).prepare().objective
    started = drawn.init(jax.random.key(0))
    leaves_alike(started["params"]["backbone"], state.variables["params"]["backbone"])
    leaves_alike(started["params"]["head"], over_base.init(jax.random.key(0))["params"]["head"])
    third, _ = trained(DecisionRunConfig.cli(fresh, default=decision(tmp_path, drawn=False)), tmp_path,
                       "third", trainer={"checkpoint_final": True})
    assert int(third.step) == 1
    answer = Decide.from_run(str(tmp_path / "runs" / "third"))("charged twice",
                                                               {"team": Choice("Which team?", TEAM)})["team"]
    assert answer.probabilities.shape == (3,) and np.all(answer.probabilities > 0)


@pytest.mark.mesh(devices=8)
def test_a_run_whose_mixture_no_family_writes_exports_and_loads_back_answering_as_its_run(tmp_path):
    """A language model run whose decoder routes one layer of two through a mixture of experts exports
    through `dew export`, in its family's source layout where one states the model and as its native
    release, stating why, where none does (export review #1), and `dew.pipeline` loads the release back
    answering as the run's own task, where the export was refused."""
    from option_witness import answered_alike, answers

    from dew._cli.main import main
    from dew.inference.loading import pipeline
    from dew.inference.models import ReleasedRun
    from dew.inference.tasks import SavedRun, TextGeneration, saved_model
    from dew.interop.layouts import source_layout

    routed = ModelConfig("causal_transformer", {"emb_features": 16, "num_layers": 2, "num_heads": 2,
                                                "mlp_features": 32, "dtype": "float32",
                                                "mixture": {"experts": 4, "top_k": 2, "layers": [1]}})
    trained(dataclasses.replace(lm(tmp_path, drawn=False), model=routed), tmp_path, "routed",
            trainer={"checkpoint_final": True})
    run, release = tmp_path / "runs" / "routed", tmp_path / "release"
    assert main(["export", str(run), str(release)]) == 0
    # The release is the one the family's layout answers for the model: its source layout, or the native
    # release stating the layout's reason.
    layout = source_layout(saved_model(SavedRun.pinned(str(run)).record, None).build())
    if layout:
        assert (release / "config.json").is_file()
    else:
        assert ReleasedRun.of(release).unstated == layout.reason
    assert answered_alike(answers(pipeline(str(release))), answers(TextGeneration.from_run(str(run))))


@pytest.mark.mesh(devices=8)
def test_a_lettered_run_over_a_qwen35_backbone_exports_a_release_a_run_starts_from(tmp_path, monkeypatch):
    """`dew export` of a lettered decision run over a Qwen3.5 text backbone (qwen3_5_text, its delta
    nets' geometry and its rotary's fraction the generic export writes none of) writes a release that
    loads: a run started from it lands every leaf as a run started from the saved run does, one step from
    each, bit for bit (ledger 4533's repro through its public path, row 4563)."""
    from saved_run_kinds import lettered

    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    first = lettered(tmp_path, drawn=False)
    trained(first, tmp_path, "first", trainer={"checkpoint_final": True})
    source = tmp_path / "runs" / "first"
    exported = tmp_path / "exported"
    export_run(SavedRun.pinned(str(source)), exported)
    assert json.loads((exported / "config.json").read_text())["model_type"] == "qwen3_5_text"

    def started(pretrained, name: str, *flags: str):
        run = DecisionRunConfig.cli(["--pretrained", str(pretrained), *flags], default=first)
        # The saved run and its release are each the run's decision task whole, its head among it and its
        # backbone no wrapper's: a run from either reads it as it is (no fresh head, no text_only).
        run = dataclasses.replace(run, head=None, source=dataclasses.replace(run.source, text_only=False))
        return trained(run, tmp_path, name, trainer={"checkpoint_final": True})[0]

    leaves_alike(started(source, "from_run", "--source.step", "latest").variables,
                 started(exported, "from_export").variables)


@pytest.mark.mesh(devices=8)
def test_a_run_started_from_a_saved_run_resumes_its_step_after_that_run_trains_on(tmp_path):
    """A named run started from a saved Dew run (`--pretrained <run>`) at the step its selection names by
    default (a diffusion run's, its latest) records that step; when the saved run has trained on, so that
    its selection names a later one, the same command resuming takes the step it read before it pins
    (`RunConfig.BUILT`, `source.step`, `resumed`), and its source compares alike with the record's, so
    it trains on past its checkpoint, sampling by the policy it recorded (row 3864); the command naming
    the later step keeps it, and its resume is refused."""
    from dew.config import earlier_run

    first = diffusion(tmp_path, drawn=False)
    trained(first, tmp_path, "first", trainer={"name": "first", "checkpoint_final": True})
    source = tmp_path / "runs" / "first"
    read = SavedRun.pinned(str(source)).step
    command = dataclasses.replace(first, pretrained=str(source), text=None,
                                  model=ModelConfig("unet", dict(DEFAULT_MODEL_CONFIG)),
                                  objective=ObjectiveConfig("diffusion", {}))
    trained(command, tmp_path, "after", trainer={"name": "after", "checkpoint_final": True})
    trained(first, tmp_path, "first", trainer={"name": "first", "steps": 2,
                                               "checkpoint_final": True})
    later = SavedRun.pinned(str(source)).step
    assert later != read
    again = dataclasses.replace(command, trainer=dataclasses.replace(
        first.trainer, name="after", checkpoint_dir=str(tmp_path / "runs")))
    saved = earlier_run(json.loads((tmp_path / "runs" / "after" / "run.json").read_text()))
    recorded = DiffusionRunConfig.read(saved)
    assert recorded.source.step == read
    assert again.resumed().pinned().source == recorded.source
    stated = dataclasses.replace(again, source=SourceOptions(step=later))
    assert stated.resumed().pinned().source.step == later
    with pytest.raises(ValueError):
        dataclasses.replace(recorded, source=SourceOptions(step=later)).check_resume(saved)
    trained(command, tmp_path, "after", trainer={"name": "after", "steps": 2,
                                                 "checkpoint_final": True})
    assert "2" in {path.name for path in (tmp_path / "runs" / "after").iterdir()}


@pytest.mark.mesh(devices=8)
def test_a_run_resumes_whether_its_decoder_scans_its_layers_or_not(tmp_path):
    """How a decoder executes its layers (`scan_layers`, `RunConfig.EXECUTED`) is no part of what a run
    computes or stores, its checkpoint holding every layer `layers_N` either way: a run saved running a
    layer at a time resumes scanning them, and back, where its resume refused it as another model (row
    4104)."""
    first = lm(tmp_path, drawn=False)
    deeper = dataclasses.replace(first, model=ModelConfig("causal_transformer",
                                                          {**first.model.fields, "num_layers": 2}))
    trained(deeper, tmp_path, "run", trainer={"name": "run", "checkpoint_final": True})
    for scanned, steps in ((True, 2), (False, 3)):
        run = dataclasses.replace(deeper, model=ModelConfig("causal_transformer", {
            **deeper.model.fields, "scan_layers": scanned}))
        trained(run, tmp_path, "run", trainer={"name": "run", "steps": steps,
                                               "checkpoint_final": True})
        assert str(steps) in {path.name for path in (tmp_path / "runs" / "run").iterdir()}


def test_a_run_resumes_under_another_kernel_choice_and_refuses_its_experts_packed_otherwise(tmp_path):
    """A run over an MXFP4 GPT-OSS source, its experts decoded, resumes on another kernel for its attention
    (`RunConfig.kernels`, overridable on whatever hardware the resume lands on, training.md) and trains a
    step past its saved one (`RESUMABLE`); a resume asking its experts packed, whose banks are no trainable
    parameters and hold no optimizer moments, is refused before it restores or writes its record
    (`check_resume`; the promotion's review: it passed the check and failed at restore)."""
    from test_hf_decoder_moe import _mxfp4_gpt_oss

    source = _mxfp4_gpt_oss(tmp_path / "source")
    # A tokenizer whose ids fit the source's 96, which its fixture carries none of, as its own, and windows
    # of ids it holds.
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(Path(__file__).parent / "fixtures" / "hf" / "dory-tiny" / name, source / name)
    ids = np.random.default_rng(0).integers(1, 56, (2, 40 * SEQ))
    corpus = write_token_corpus(tmp_path / "held-tokens", ids[0], ids[1, :8 * SEQ], tokenizer=str(source),
                                vocab_size=96)
    first = dataclasses.replace(over(str(source), tmp_path, drawn=False), data=windows(corpus.train),
                                validation=windows(corpus.val),
                                diagnostics=Diagnostics(kernels=Kernels(attention="reference")))
    trained(first, tmp_path, "run", trainer={"name": "run", "checkpoint_final": True})
    rerun = dataclasses.replace(first, diagnostics=Diagnostics(kernels=Kernels(attention="xla")))
    again, _ = trained(rerun, tmp_path, "run",
                       trainer={"name": "run", "steps": 2, "checkpoint_final": True})
    assert int(again.step) == 2
    record = (tmp_path / "runs" / "run" / "run.json").read_text()
    packed = dataclasses.replace(first, source=dataclasses.replace(first.source, packed_weights=True))
    with pytest.raises(ValueError):
        trained(packed, tmp_path, "run", trainer={"name": "run", "steps": 3,
                                                  "checkpoint_final": True})
    assert (tmp_path / "runs" / "run" / "run.json").read_text() == record


@pytest.mark.mesh(devices=8)
def test_a_saved_decision_runs_step_reads_the_calibration_fitted_on_it_after_a_later_one_is(tmp_path):
    """A decision run calibrated at its first step and trained on, calibrated again at its second, reads
    each step with the calibration fitted on it (`Decide.save` keeps each, `CALIBRATIONS`, and a latest an
    earlier Dew kept alone by its own weights before replacing it): a reader pinned to the first step, a
    run started from it among them, loads it after the second was fitted, where only the latest was kept
    and refused it (rows 3865, 3877)."""
    import shutil

    from dew.decision.calibration import Calibration
    from dew.decision.task import Decide
    from dew.registry import from_record

    first = decision(tmp_path, drawn=False)
    trained(first, tmp_path, "first", trainer={"name": "first", "checkpoint_final": True})
    source = tmp_path / "runs" / "first"
    read = SavedRun.pinned(str(source)).step
    # The run as a Dew that kept no calibration by step left it: its latest alone, which the next fit
    # keeps by its own weights before it replaces it (row 3877).
    shutil.rmtree(source / "calibrations")
    trained(first, tmp_path, "first", trainer={"name": "first", "steps": 2,
                                               "checkpoint_final": True})
    later = SavedRun.pinned(str(source)).step
    assert later != read
    tasks = {step: Decide.from_saved(SavedRun.pinned(str(source), step)) for step in (read, later)}
    from dew.decision.task import step_calibrations

    for step, task in tasks.items():
        assert task.weights is not None and task.weights.step == step
        (kept,) = step_calibrations(str(source), step)
        assert task.calibration == from_record(Calibration, json.loads(kept.read_text())["calibration"])
    # A run started from the first step trains, as `dew train --pretrained <run> --source.step` starts it
    # (row 3880).
    pinned = DecisionRunConfig.cli(["--pretrained", str(source), "--source.step", str(read)], default=first)
    trained(pinned, tmp_path, "from_first", trainer={"checkpoint_final": True})
    assert DecisionRunConfig.load(str(tmp_path / "runs" / "from_first")).source.step == read
    # The release of the first step is written again where its own calibration changes, the run's latest
    # as it was (`released_run`'s fingerprint holds the step's calibrations).
    release = released_run(source, step=read)
    (kept,) = step_calibrations(str(source), read)
    changed = json.loads(kept.read_text())
    changed["name"] = "recalibrated"
    kept.write_text(json.dumps(changed))
    assert released_run(source, step=read) != release


def leaves_alike(tree, alike) -> None:
    for leaf, other in zip(jax.tree.leaves(jax.device_get(tree)), jax.tree.leaves(jax.device_get(alike)),
                           strict=True):
        np.testing.assert_array_equal(np.asarray(leaf), np.asarray(other), strict=True)


@pytest.mark.mesh(devices=8)
def test_a_diffusion_run_starts_from_a_saved_diffusion_run_as_from_its_task(tmp_path):
    """A diffusion run that starts from a saved diffusion run (`--pretrained <run>`, `--source.step latest`)
    starts from the weights the run's own task restores (`TextToImage.from_run`), bit for bit, and samples
    by that task's policy where it states none. It trains a step, and its record names the run it started
    from and the step it read; an adapter over it trains its factors over the run's frozen weights. The
    source binds at the run's own size and at another, from which a run over larger pictures trains (ledger
    2792, where another size was refused), and a language model's loader refuses a diffusion run, which has
    no release. A run that trained an adapter starts another with its base frozen
    (`test_a_diffusion_run_starts_from_an_adapted_run_with_its_base_frozen`; the two took 135 s as one
    test, ledger 3899)."""
    first = diffusion(tmp_path, drawn=False)
    trained(first, tmp_path, "first", trainer={"checkpoint_final": True})
    source = tmp_path / "runs" / "first"
    run = dataclasses.replace(first, pretrained=str(source), source=SourceOptions(step="latest"), text=None,
                              model=ModelConfig("unet", dict(DEFAULT_MODEL_CONFIG)),
                              objective=ObjectiveConfig("diffusion", {}))
    task = TextToImage.from_run(str(source))

    objective = run.build()
    leaves_alike(objective.variables["params"], task.variables["params"])
    assert (objective.steps, objective.solver) == (task.steps, task.solver) == (2, Euler())
    trained(run, tmp_path, "from_run")
    recorded = DiffusionRunConfig.load(str(tmp_path / "runs" / "from_run"))
    assert recorded.pretrained == str(source)
    assert recorded.source.step == SavedRun.pinned(str(source)).step
    adapted, _ = trained(run, tmp_path, "adapted", lora=LoRA(rank=2, modules=("ada_proj",)))
    assert FROZEN in adapted.variables
    factors = sum(leaf.size for leaf in jax.tree.leaves(adapted.variables["params"]))
    assert 0 < factors < sum(leaf.size for leaf in jax.tree.leaves(task.variables["params"]))

    sized = load_diffusion_source(str(source), dtype="float32", size=task.inputs.sample.shape[:-1])
    assert sized.inputs.sample == task.inputs.sample
    larger = load_diffusion_source(str(source), dtype="float32", size=(8, 8))
    assert larger.inputs.sample == dataclasses.replace(task.inputs.sample, shape=(8, 8, 3))
    train, held = pictures()
    state, _ = trained(dataclasses.replace(run, data=captioned(train, 8, drawn=False),
                                           validation=captioned(held, 8, drawn=False)), tmp_path, "larger")
    assert int(state.step) == 1
    with pytest.raises(ValueError):
        released_run(source)


PUBLISHED = {"xl": ("tiny_diffusers", "xl"), "flux": ("flux_source", "pipeline")}
"""Each published pipeline a run trains from (its fixture's archive and folder): SDXL's, whose time ids
state its size, and Flux's, whose grid's tokens set its resolution shift."""


@pytest.mark.parametrize("pipeline", sorted(PUBLISHED))
def test_a_run_from_a_published_pipeline_binds_at_another_size_as_the_pipeline_does(pipeline, tmp_path):
    """A run trained a step from a published pipeline and saved binds at another size as the pipeline binds
    there (ledger 2792, where another size was refused): its sample, an XL composition's size conditioning,
    its grid's tokens and a resolution shift at that size are the pipeline's own there, each moved from
    its own size's."""
    from diffusion_stubs import batch_for
    from diffusion_support import images_of, saved_after_one_step
    from interop_support import extract_fixture

    from dew.config import TrainerConfig
    from dew.interop.conditioners import DiffusionConditioner
    from dew.interop.pretrained import PretrainedPipeline
    from dew.interop.saved_sources import RunPipeline

    archive, folder = PUBLISHED[pipeline]
    directory = extract_fixture(Path(__file__).parent / "fixtures" / f"{archive}.tar.xz", tmp_path / archive)
    directory = directory / folder
    own = load_diffusion_source(str(directory), dtype="float32", attention_impl="xla")
    height, width = own.inputs.sample.shape[-3:-1]
    config = DiffusionRunConfig(
        pretrained=str(directory), data=images_of("tfds://oxford_flowers102", height), preset=None,
        model=ModelConfig("unet", {**DEFAULT_MODEL_CONFIG, "dtype": "float32", "attention_impl": "xla"}),
        val_metrics=(),
        trainer=TrainerConfig(checkpoint_dir=str(tmp_path)),
        objective=ObjectiveConfig("diffusion", {"steps": 2, "ema_decay": None}))
    saved_after_one_step(config, lambda task: batch_for(task, 4), tmp_path / "run", 1e-2)
    # Wider and taller apart: a rectangle's two extents each read where they are.
    size = (2 * height, width + width // 2)
    saved = load_diffusion_source(str(tmp_path / "run"), dtype="float32", size=size)
    published = load_diffusion_source(str(directory), dtype="float32", attention_impl="xla", size=size)
    assert isinstance(saved, RunPipeline) and saved.grid is not None
    assert isinstance(published, PretrainedPipeline) and isinstance(own, PretrainedPipeline)
    assert saved.inputs.sample == published.inputs.sample != own.inputs.sample

    def time_ids(source) -> dict:
        return {keyword: condition.encoder.time_id_values for keyword, condition in
                source.inputs.conditions.items() if isinstance(condition.encoder, DiffusionConditioner)}

    assert time_ids(saved) == time_ids(published)
    if pipeline == "xl":
        assert time_ids(saved) and time_ids(saved) != time_ids(own)
    assert ((saved.grid.tokens, saved.grid.origin, dict(saved.grid.config)) ==
            (published.task.grid.tokens, published.task.grid.origin, dict(published.task.grid.config)))
    assert saved.grid.tokens != own.task.grid.tokens
    # The run named no preset, so it trained the pipeline's own process (`grid_process`), which follows the
    # grid to the size.
    assert saved.grid_process
    if pipeline == "flux":
        assert flow_shift(saved) == flow_shift(published) != flow_shift(own)


def flow_shift(source) -> float:
    """The shift of `source`'s rectified-flow process."""
    from dew.diffusion.schedules.flow import FlowMatchingScheduler

    schedule = source.process.schedule
    assert isinstance(schedule, FlowMatchingScheduler)
    return schedule.shift


def test_a_run_naming_its_own_resolution_shift_keeps_its_process_at_another_size(tmp_path):
    """A run from a published Flux pipeline naming a resolution shift of its own keeps its process at another
    size, where its source's follows its grid there: the run's shift equals the grid's at its own size,
    its own shift's constants a mu flat in the token count at the grid's own mu there, so no shift's value
    tells the two apart, and the record its objective writes does (`DiffusionObjective.grid_process`,
    ledger 2792), as it does for an objective given a grid of its own."""
    from diffusion_stubs import batch_for
    from diffusion_support import images_of, saved_after_one_step
    from interop_support import extract_fixture

    from dew.config import TrainerConfig
    from dew.diffusion.presets import Flow, ResolutionShift
    from dew.interop.pretrained import PretrainedPipeline
    from dew.interop.saved_sources import RunPipeline
    from dew.objectives.diffusion import DiffusionObjective

    archive, folder = PUBLISHED["flux"]
    directory = extract_fixture(Path(__file__).parent / "fixtures" / f"{archive}.tar.xz", tmp_path / archive)
    directory = directory / folder
    own = load_diffusion_source(str(directory), dtype="float32", attention_impl="xla")
    assert isinstance(own, PretrainedPipeline)
    # An objective over the pipeline trains its process over its grid; one given another grid, even an
    # equal one, trains the pipeline's process over a grid of its own.
    assert DiffusionObjective(own, ema_decay=None).grid_process
    assert not DiffusionObjective(own, ema_decay=None, grid=dataclasses.replace(own.grid)).grid_process
    flow = own.grid.schedule.policy.flow
    assert flow is not None and flow.dynamic and own.grid.tokens is not None
    mu = flow.mu(own.grid.tokens)
    height, width = own.inputs.sample.shape[-3:-1]
    config = DiffusionRunConfig(
        pretrained=str(directory), data=images_of("tfds://oxford_flowers102", height),
        preset=Flow(resolution_shift=ResolutionShift(base_shift=mu, max_shift=mu)),
        model=ModelConfig("unet", {**DEFAULT_MODEL_CONFIG, "dtype": "float32", "attention_impl": "xla"}),
        val_metrics=(), trainer=TrainerConfig(checkpoint_dir=str(tmp_path)),
        objective=ObjectiveConfig("diffusion", {"steps": 2, "ema_decay": None}))
    task, _, _ = saved_after_one_step(config, lambda task: batch_for(task, 4), tmp_path / "run", 1e-2)
    # At its own size the run's process is the grid's, value for value.
    assert flow_shift(task) == flow_shift(own)
    size = (2 * height, width + width // 2)
    saved = load_diffusion_source(str(tmp_path / "run"), dtype="float32", size=size)
    published = load_diffusion_source(str(directory), dtype="float32", attention_impl="xla", size=size)
    assert isinstance(saved, RunPipeline)
    assert flow_shift(saved) == flow_shift(own) != flow_shift(published)
    assert not task.grid_process and not saved.grid_process


@pytest.mark.mesh(devices=8)
def test_a_diffusion_run_starts_from_an_adapted_run_with_its_base_frozen(tmp_path):
    """A run that trained an adapter starts another (`--pretrained <run>`) with its base frozen and its
    factors training: as many as the adapter it starts from trained."""
    first = diffusion(tmp_path, drawn=False)
    adapter, _ = trained(first, tmp_path, "adapted_first", trainer={"checkpoint_final": True},
                         lora=LoRA(rank=2, modules=("ada_proj",)))
    run = dataclasses.replace(first, pretrained=str(tmp_path / "runs" / "adapted_first"),
                              source=SourceOptions(step="latest"), text=None,
                              model=ModelConfig("unet", dict(DEFAULT_MODEL_CONFIG)),
                              objective=ObjectiveConfig("diffusion", {}))
    from_adapted, _ = trained(run, tmp_path, "from_adapted")
    assert FROZEN in from_adapted.variables
    factors = sum(leaf.size for leaf in jax.tree.leaves(adapter.variables["params"]))
    assert 0 < sum(leaf.size for leaf in jax.tree.leaves(from_adapted.variables["params"])) == factors


WORKER = Path(__file__).with_name("saved_export_worker.py")


@pytest.mark.distributed
@pytest.mark.mesh
@pytest.mark.parametrize("scan", [False, True], ids=["looped", "scanned"])
def test_a_pool_exports_a_model_spanning_its_processes_a_shard_at_a_time(tmp_path, scan):
    """Two real processes export a decoder whose kernels span both, looped or scanning its layers over a
    banked store (each run's row read home as it is written, `stored_layers`, ledger 4103), every layer's
    attention extended by DEX, as a saved run's release (`export_run`, ledger 2874), in a score of shards,
    each output projection folded as it is written: each reads one tensor home at a time, together, so
    neither host's allocated bytes rise during the export by half the model, where the same measure over
    that tree held home rises at least half of it on each host, so the measure sees the bytes in that run
    (`saved_export_worker.allocated_growth`, glibc's own count, a decoder of 256 KiB tensors): resident
    pages missed what the allocator served from pages freed before (rows 5266, 5562), and
    tracemalloc never saw the buffers a tensor is read home into, and held a 1.3 MB decoder's export to
    its own fixed cost, which CI's process state crossed (ledger 3794). The publishing process writes the
    tensors one process writes of the same weights unplaced, byte for byte, though the restored tree's
    key order may place one in another shard; the other writes the release's small files alone, no
    weights. Outside the export's scope the write is refused rather than left waiting on the other
    process."""
    publisher, reader = run_pool("exported", tmp_path / "pool", processes=2, script=WORKER, devices=2,
                                 fsdp_size=4, run_dir=str(tmp_path), **({"scan": True} if scan else {}))
    for report in (publisher, reader):
        assert report["processes"] == 2 and report["spanning"] > 0
        assert report["refused_outside"]
        assert report["exported"] < report["whole"] / 2 <= report["held"]
    assert publisher["differing"] == {} and len(publisher["release"]) > 10
    assert publisher["alone"] == reader["alone"]
    assert reader["release"] == {} and reader["small_files"] == publisher["small_files"]


@pytest.mark.distributed
@pytest.mark.mesh
def test_a_pool_reads_a_saved_runs_release_through_one_publisher(tmp_path, monkeypatch):
    """Two real processes sharing Dew's cache read a saved language model run as a source (`released_run`,
    ledger 2992): the release is written once, the first process publishing it and the other reading
    every tensor as it does, writing none, each host's allocated bytes rising inside the export by less
    than half the release, written in shards a fraction of it, where the same measure over the release
    held home rises at least half of it (`saved_export_worker.allocated_growth`): at one shard, the 5 GB
    default, the writer held the whole release, which tracemalloc did not see (ledger 3794). Both read
    the one release. `dew export` to a destination they share writes it once
    likewise, the tensors one process's export writes.

    The release is sixteen layers deep, 67 MB, its tensors a megabyte at most, so the shards a writer
    holds are a small fraction of it (row 4868)."""
    from dew._cli.main import main
    from dew.registry import ModelConfig

    run = dataclasses.replace(lm(tmp_path, drawn=False), model=ModelConfig(
        "causal_transformer", {"emb_features": 256, "num_layers": 16, "num_heads": 4, "dtype": "float32"}))
    trained(run, tmp_path, "first", trainer={"checkpoint_final": True})
    source = tmp_path / "runs" / "first"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "shared_cache"))
    publisher, reader = run_pool("released", tmp_path / "pool", processes=2, script=WORKER, devices=2,
                                 fsdp_size=4, run_dir=str(source))
    assert publisher["processes"] == 2 and publisher["release"] == reader["release"]
    assert publisher["release_files"] == reader["release_files"] and publisher["release_files"]
    whole = sum(path.stat().st_size for path in Path(publisher["release"]).glob("*.safetensors"))
    assert [scope["writes"] for scope in publisher["read"]] == [True]
    assert [scope["writes"] for scope in reader["read"]] == [False]
    for report in (publisher, reader):
        assert report["read"][0]["growth"] < whole / 2 <= report["held"]

    assert publisher["export_code"] == reader["export_code"] == 0
    assert [scope["writes"] for scope in reader["exported"]] == [False]
    assert main(["export", str(source), str(tmp_path / "alone")]) == 0
    assert exported_tensors(source.parent / "exported") == exported_tensors(tmp_path / "alone")


@pytest.mark.distributed
@pytest.mark.mesh
def test_a_pool_reads_a_saved_decision_runs_release_through_one_publisher(tmp_path, monkeypatch):
    """Two real processes sharing Dew's cache read a saved decision run as a source (`released_run`), its
    release its task's own (a Laya checkpoint, `run_plan`'s decision branch), as they read a language model
    run's (ledger 3102): the release is written once, the first process publishing it and the other
    reading every tensor as it does, writing none, and both read the one release; `dew export` to a
    destination they share writes it once likewise, the tensors one process's export writes. The tiny Laya
    backbone's release is a few megabytes, below what a host's allocated bytes resolve, so the bound on
    what a host holds is the language model run's to show (above), over the writer both share."""
    from dew._cli.main import main

    trained(decision(tmp_path, drawn=False), tmp_path, "first", trainer={"checkpoint_final": True})
    source = tmp_path / "runs" / "first"
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "shared_cache"))
    publisher, reader = run_pool("released", tmp_path / "pool", processes=2, script=WORKER, devices=2,
                                 fsdp_size=4, run_dir=str(source))
    assert publisher["processes"] == 2 and publisher["release"] == reader["release"]
    assert publisher["release_files"] == reader["release_files"] and publisher["release_files"]
    assert [scope["writes"] for scope in publisher["read"]] == [True]
    assert [scope["writes"] for scope in reader["read"]] == [False]
    assert publisher["export_code"] == reader["export_code"] == 0
    assert [scope["writes"] for scope in reader["exported"]] == [False]
    assert main(["export", str(source), str(tmp_path / "alone")]) == 0
    assert exported_tensors(source.parent / "exported") == exported_tensors(tmp_path / "alone")


def exported_tensors(directory: Path) -> dict[str, bytes]:
    """Each tensor an export's weight files hold, by name, as bytes beside its dtype and shape."""
    from safetensors.numpy import load_file

    return {name: (str(array.dtype), array.shape, array.tobytes())
            for path in sorted(directory.glob("*.safetensors")) for name, array in load_file(path).items()}


@pytest.mark.mesh(devices=8)
def test_a_multimodal_run_passing_messages_reloads_from_its_native_release_as_its_run(tmp_path):
    """A multimodal LM run whose loop passes messages (`Loop.messages`), which no family's layout writes,
    releases in Dew's own layout (`released_run`), which loads back (`Pretrained.from_saved`'s wrapper
    branch) as the run's model: its towers' and decoder's weights the run's, its loop the run's messages;
    its export is refused, as the source's own loop is once it gains the same messages, which exports
    without them; and a change of its merges' rank refused on the reloaded bundle, whose merges it holds,
    not draws (rows 10839, 11755)."""
    from saved_run_kinds import MULTIMODAL, multimodal

    from dew.interop import Pretrained
    from dew.nn.cross_pass import CrossPass
    from dew.nn.multimodal import MultimodalTransformer

    looped = dataclasses.replace(multimodal(tmp_path, drawn=False), loop_steps=2, loop_layers=(0, 2),
                                 loop_messages=True, loop_message_block_size=1)
    state, _ = trained(looped, tmp_path, "messaged", trainer={"checkpoint_final": True})
    loaded = Pretrained.load(str(released_run(tmp_path / "runs" / "messaged")), dtype="float32")
    assert isinstance(loaded.model, MultimodalTransformer)
    assert loaded.model.loop is not None and loaded.model.loop.messages == CrossPass(block_size=1)
    leaves_alike(loaded.variables["params"], state.variables["params"])
    with pytest.raises(ValueError):
        loaded.export()
    # The control: the run's source, looped as the run is, exports; gaining the run's messages, it is
    # refused as the run's release is, so the messages are what no family's format writes.
    source = Pretrained.load(MULTIMODAL, dtype="float32")
    looping = source.with_model(source.model.with_passes(2, (0, 2)), key=0)
    assert looping.export()
    messaged = source.with_model(source.model.with_passes(2, (0, 2), messages=CrossPass(block_size=1)), key=0)
    with pytest.raises(ValueError):
        messaged.export()
    with pytest.raises(ValueError):
        loaded.with_model(loaded.model.with_passes(messages=CrossPass(block_size=1, rank=8)), key=0)

