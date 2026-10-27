"""A decision run end to end on Laya's toy checkpoint: a CSV in, a calibrated run out."""

import csv
import dataclasses
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from dew.config import RunConfig
from dew.data import Files, Loading, Memory
from dew.decision import Choice, Decide, decisions
from dew.decision.config import DecisionRunConfig
from dew.decision.scoring import ScoringRule
from dew.registry import from_record
from dew.training.evaluation import Summary

pytestmark = pytest.mark.mesh

TINY = Path(__file__).parent / "fixtures" / "laya" / "tiny"


def test_a_csv_fine_tunes_a_laya_checkpoint_into_a_calibrated_run(tmp_path):
    table = tmp_path / "tickets.csv"
    tickets = [("charged twice for March", "billing"), ("the site is down", "technical"),
               ("how much is the pro plan", "sales"), ("refund my last invoice", "billing"),
               ("error 500 on login", "technical"), ("can I upgrade", "sales")] * 4
    with table.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows(tickets)
    # The command line over a table's chain: half its rows held out by the hash of their content.
    train, held = Files(str(table), format="csv").holdout(0.5)

    def asked(rows):
        return rows.pipe(decisions, question="team").read("memory").batch(8).loading(Loading(workers=0))

    config = DecisionRunConfig.cli([
        "--pretrained", str(TINY), "--objective.loss", json.dumps(
            {"class": "dew.decision.scoring:Combined",
             "fields": {"terms": [[1.0, {"class": "log_loss"}], [0.5, {"class": "brier"}]]}}),
        "--trainer.name", "tickets", "--trainer.checkpoint-dir", str(tmp_path / "runs"),
        "--trainer.steps", "2", "--trainer.eval-every", "2",
        "--trainer.checkpoint-every", "2", "--trainer.log-every", "1", "--trainer.multi-host", "False",
        "--trainer.compilation-cache-dir", "None", "--model.dtype", "float32"],
        default=DecisionRunConfig(data=asked(train), validation=asked(held)))
    assert config.data.facts()["batch"] == 8
    assert from_record(ScoringRule, config.objective.fields["loss"]).name == "log_loss+0.5*brier"
    config.run()
    run = tmp_path / "runs" / "tickets"
    assert (run / "decide.json").is_file()
    decide = Decide.from_run(str(run))
    answer = decide("I was billed twice", {"team": Choice("Which team?", ["billing", "technical", "sales"])})
    assert answer["team"].choice in ("billing", "technical", "sales")
    assert decide.name == "tickets" and set(decide.calibration.temperatures.types) == {"choice"}


def test_a_decision_run_scores_each_split_with_its_own_metrics_and_ranks_by_their_summary(tmp_path):
    """A decision run names its validation splits and each one's metrics (`--val-metrics`): accuracy on
    every split and the Brier score on `test` alone, in place of the task's own suite. A summary weighs
    both splits' accuracy into one (`--val-summaries`), formed at each evaluation, which scores both, and
    the run ranks its checkpoints by it (`--trainer.best evaluation/summary/board`)."""
    board = f"{Summary.SPLIT}/board"
    table = tmp_path / "tickets.csv"
    tickets = [("charged twice for March", "billing"), ("the site is down", "technical"),
               ("how much is the pro plan", "sales"), ("refund my last invoice", "billing")] * 6
    with table.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows(tickets)
    train, held = Files(str(table), format="csv").holdout(0.5)

    def asked(rows):
        return rows.pipe(decisions, question="team").read("memory").batch(8).loading(Loading(workers=0))

    config = DecisionRunConfig.cli([
        "--pretrained", str(TINY), "--trainer.name", "splits",
        "--trainer.checkpoint-dir", str(tmp_path / "runs"),
        "--trainer.steps", "2", "--trainer.eval-every", "1", "--trainer.checkpoint-every", "1",
        "--trainer.log-every", "1", "--trainer.multi-host", "False",
        "--trainer.compilation-cache-dir", "None",
        "--model.dtype", "float32", "--trainer.best", board,
        "--val-metrics", '[{"name": "accuracy"}, {"name": "brier", "split": "test"}]',
        "--val-summaries", json.dumps([{"name": "board", "weights": {"val/accuracy": 1, "test/accuracy": 3},
                                        "better": "higher"}])],
        default=DecisionRunConfig(data=asked(train), validation={"val": asked(held), "test": asked(held)}))
    config.run()
    run = tmp_path / "runs" / "splits"
    logged = [json.loads(line) for line in (run / "tracking" / "scalars.jsonl").read_text().splitlines()]
    scores = {}
    for row in logged:
        scores.setdefault(row["step"], {}).update(row["scalars"])
    for step in (1, 2):
        names = {name for name in scores[step] if name.rpartition("/")[0] in ("val", "test", Summary.SPLIT)}
        assert names == {"val/loss", "val/accuracy", "test/loss", "test/accuracy", "test/brier", board}
        assert scores[step][board] == (scores[step]["val/accuracy"] + 3 * scores[step]["test/accuracy"]) / 4
    recorded = RunConfig.load(str(run))
    assert (recorded.val_metrics, recorded.val_summaries) == (config.val_metrics, config.val_summaries)


def test_a_lora_run_over_a_decision_release_trains_its_factors_and_reloads(tmp_path):
    """A decision release takes the backbone's transformations as a fresh backbone does: a LoRA
    run over a slot release starts from the release's own readout and its question layout, trains
    the adapter's factors and the readout over a frozen base, and its saved run loads back as a
    task answering with the trained probabilities."""
    import jax
    import numpy as np
    import optax
    from test_slot_sources import slot_release

    from dew.batch import FROZEN
    from dew.checkpoints import Checkpoints
    from dew.decision.layout import QuestionLayout
    from dew.lora import LoRA
    from dew.training import Trainer

    release, stored = slot_release(tmp_path, "head")
    rows = Memory([{"text": text, "label": label} for text, label in
                   [("charged twice", "billing"), ("site down", "technical"), ("pro price", "sales")] * 4])
    asked = rows.pipe(decisions, question="team").read("memory").batch(8).loading(Loading(workers=0))
    default = Fitted(data=asked)
    # The release serves long rows; training pads each to the layout's length, so the run budgets it.
    config = dataclasses.replace(default, pretrained=str(release), lora=LoRA(rank=2, modules=("q_proj",)),
                                 max_len=512, trainer=dataclasses.replace(default.trainer, multi_host=False,
                                                                          compilation_cache_dir=None))
    objective, dataset = config.prepare().train("adapted")
    assert objective.layout.max_len == 512
    assert isinstance(objective.layout, QuestionLayout)
    head = objective.held_variables()["params"]["head"]
    for native, name in (("slots", "slot"), ("q", "q.weight"), ("k", "k.weight")):
        np.testing.assert_array_equal(np.asarray(head[native]), stored[name])
    start = objective.init(jax.random.key(0))
    run = tmp_path / "run"
    checkpoints = Checkpoints(str(run), keep=1)
    state = Trainer(objective, optax.adam(1e-2), key=0, checkpoints=checkpoints).fit(
        dataset, steps=2, log_every=100, checkpoint_every=2)
    checkpoints.wait()

    def moved(before, after):
        return [not np.array_equal(np.asarray(a), np.asarray(b))
                for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after), strict=True)]

    assert all(not change for change in moved(start[FROZEN], state.variables[FROZEN]))
    assert any(moved(start["params"]["head"], state.variables["params"]["head"]))
    assert any(moved(start["params"]["backbone"], state.variables["params"]["backbone"]))
    live, reloaded = objective.pipeline(state), Decide.from_run(str(run))
    question = {"team": Choice("Which team?", ["billing", "technical", "sales"])}
    first, again = live("charged twice", question), reloaded("charged twice", question)
    np.testing.assert_array_equal(first["team"].probabilities, again["team"].probabilities)


QWEN = Path(__file__).parent / "fixtures" / "hf" / "qwen38-dense-tiny"
TICKETS = Memory([{"text": text, "label": label} for text, label in
                  [("charged twice", "billing"), ("site down", "technical"), ("pro price", "sales")] * 4]
                 ).pipe(decisions, question="team").read("memory").batch(8).loading(Loading(workers=0))


@dataclasses.dataclass(frozen=True)
class _NamedSplits(Memory):
    """Memory rows with native named splits, as a Hub or TFDS source has them."""

    splits: Mapping[str, tuple] = dataclasses.field(default_factory=dict)

    def with_split(self, name: str) -> "_NamedSplits":
        return dataclasses.replace(self, values=self.splits[name])


def test_decisions_read_their_table_as_the_stages_before_them_make_it():
    """A table renamed to the decision columns before `decisions` reads its options from the renamed
    rows, in a memory read: the stages before it run before its options are inferred."""
    from dew.data.compile import iterate

    rows = [{"body": f"ticket {index}", "category": "ab"[index % 2]} for index in range(4)]
    renamed = Memory(rows).rename_columns({"body": "text", "category": "label"})
    examples = [row["example"] for row in iterate(renamed.pipe(decisions, question="team"), read="memory")]
    assert list(examples[0].questions["team"].options) == ["a", "b"]
    assert [example.answers["team"] for example in examples] == ["a", "b", "a", "b"]


@dataclasses.dataclass(frozen=True)
class _ClassLabels(Memory):
    """Memory rows whose store states its label column's class names, as a Hub ClassLabel does."""

    names: tuple[str, ...] = ()

    def metadata(self):
        return {**super().metadata(), "class_names": {"label": list(self.names)}}


def test_a_relabelled_table_reads_its_new_labels_not_the_store_s_class_names():
    """The store's class names name the options where the stages before `decisions` keep its labels,
    and a map that relabels the column is read from its rows instead."""
    from dew.data.compile import iterate

    rows = [{"text": f"ticket {index}", "label": index % 2} for index in range(4)]
    store = _ClassLabels(rows, names=("billing", "technical"))

    def options(chain):
        examples = [row["example"] for row in iterate(chain.pipe(decisions, question="team"), read="memory")]
        return list(examples[0].questions["team"].options), [example.answers["team"] for example in examples]

    assert options(store) == (["billing", "technical"], [0, 1, 0, 1])
    renamed = store.map(lambda row: {**row, "label": "xy"[row["label"]]})
    assert options(renamed) == (["x", "y"], ["x", "y", "x", "y"])


TEAM = {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "technical"]}


def asked(state: str, **gold) -> dict:
    """A row asking which team takes `state`, with its `answers` or `targets` for that question."""
    stated = {kind: {"team": value} for kind, value in gold.items()}
    return {"state": state, "questions": {"team": TEAM}, **stated}


ROWS = {
    "tickets/0": [asked("charged twice", answers="billing"), asked("site down", answers="technical")],
    "tickets/1": [asked("refund please", targets={"technical": 0.25, "billing": 0.75}),
                  asked("error 500", answers=1)],
    "tickets.held": [asked("billed again", answers="billing"), asked("login broken", answers="technical")],
    "billing": [asked("invoice wrong", answers="billing"), asked("double charge", answers=0)],
    "legacy": [asked("fax broken", answers="technical")],
}
"""A decision mixture's files by their names in its directory, each a JSONL file of these rows: the
`tickets` corpus's two training shards and its held rows, and the `billing` and `legacy` corpora."""


def decision_mixture(root: Path) -> dict:
    """The mixture directory `root` of `ROWS`, every corpus of weight one, and its manifest: `tickets`
    trains on its two shards, which one index of their rows lists, and holds out its held file; the
    others train on their one file, as a corpus the manifest lists no shards of does. Beside it, the
    sha256 of the canonical source manifest it was made from, which a descriptor replacing it names."""
    facts = {}
    for name, rows in ROWS.items():
        path = root / f"{name}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        facts[name] = {"path": f"{name}.jsonl", "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    # The shards' one index: each shard's own offsets (`Files.provision`) under its place in the list.
    offsets = []
    for number, name in enumerate(("tickets/0", "tickets/1")):
        Files(str(root / f"{name}.jsonl"), format="jsonl").provision()
        listed = np.load(root / f"{name}.index.npy")
        listed["shard"] = number
        offsets.append(listed)
    np.save(root / "tickets" / "train.index.npy", np.concatenate(offsets), allow_pickle=False)
    shards = [facts["tickets/0"], facts["tickets/1"]]
    index = {"path": "tickets/train.index.npy",
             "bytes": (root / "tickets" / "train.index.npy").stat().st_size}
    manifest = {"weights": {"tickets": 1.0, "billing": 1.0, "legacy": 1.0},
                "slots": {"questions": 1, "options": 2},
                "files": {"tickets": {"train": {"sha256": "".join(shard["sha256"] for shard in shards),
                                                "shards": shards, "index": index},
                                      "held": {"sha256": facts["tickets.held"]["sha256"]}},
                          "billing": {"train": {"sha256": facts["billing"]["sha256"]}},
                          "legacy": {"train": {"sha256": facts["legacy"]["sha256"]}}}}
    (root / "mixture.json").write_text(json.dumps(manifest))
    (root / "manifest.sha256").write_text(hashlib.sha256(b"the canonical source manifest").hexdigest() + "\n")
    return manifest


def replacing(root: Path, manifest: dict, name: str, **changes) -> str:
    """A descriptor at `name` beside `root` that replaces its manifest by `manifest` with `changes`,
    naming the canonical manifest `root`'s was made from; its URL."""
    provenance = {"original_manifest_sha": (root / "manifest.sha256").read_text().strip()}
    path = root.parent / name
    path.write_text(json.dumps({"provenance": provenance, **manifest, **changes}))
    return str(path)


def test_a_decision_run_fine_tunes_on_a_mixture_by_the_weights_its_descriptor_and_the_run_state(tmp_path):
    """A decision run trains on a mixture directory's corpora by the weights of the descriptor that
    replaces its manifest, and the run's own over them (`mixture`): the shards its index lists, a
    corpus's one file, none of no weight, each row the example it states. Its summary reports the
    weights it mixed by and the manifest that made them, and the held rows of the corpora that have
    any validate it."""
    from dew.data.compile import iterate
    from dew.decision import mixture

    manifest = decision_mixture(tmp_path / "mixture")
    descriptor = replacing(tmp_path / "mixture", manifest, "descriptor.json",
                           weights={"tickets": 3.0, "billing": 1.0, "legacy": 0.0})
    train, held = mixture(str(tmp_path / "mixture"), weights={"billing": 2.0}, descriptor=descriptor)
    assert held is not None
    assert [row["example"].state for row in iterate(held, read="memory")] == ["billed again", "login broken"]

    def loaded(rows):
        return rows.read("memory").batch(8).loading(Loading(workers=0))

    config = DecisionRunConfig.cli([
        "--pretrained", str(TINY), "--trainer.name", "mixed",
        "--trainer.checkpoint-dir", str(tmp_path / "runs"),
        "--trainer.steps", "2", "--trainer.eval-every", "2", "--trainer.checkpoint-every", "2",
        "--trainer.log-every", "1", "--trainer.multi-host", "False",
        "--trainer.compilation-cache-dir", "None", "--model.dtype", "float32"],
        default=DecisionRunConfig(data=loaded(train), validation=loaded(held)))
    config.run()
    tracking = tmp_path / "runs" / "mixed" / "tracking"
    recorded = [json.loads(line) for line in (tracking / "records.jsonl").read_text().splitlines()]
    [summary] = [entry["value"]["summary"] for entry in recorded if entry["type"] == "RunRecord"]
    assert summary["mixture"] == {"tickets": 3.0, "billing": 2.0}
    assert summary["made"] == json.loads(Path(descriptor).read_text())
    logged = [json.loads(line) for line in (tracking / "scalars.jsonl").read_text().splitlines()]
    assert any("val/loss" in row["scalars"] for row in logged)
    with pytest.raises(ValueError):
        mixture(str(tmp_path / "mixture"), weights={"tickets": 0.0, "billing": 0.0}, descriptor=descriptor)


def test_a_benchmark_reads_one_corpus_split_whatever_its_weight_under_the_descriptor_that_pins_it(tmp_path):
    """`corpus` reads one corpus's split as a decision run reads it, whatever its weight: the shards its
    manifest lists, in their order, through the index of their rows, each row the example it states, a
    target keyed by option in its question's order; a corpus's one held file; no such split, None. A
    descriptor of the directory reads alike, and one that names another canonical manifest, or changes a
    corpus's raw rows, is refused."""
    from dew.data.compile import iterate
    from dew.decision import corpus

    root = tmp_path / "mixture"
    manifest = decision_mixture(root)
    pinned = replacing(root, manifest, "descriptor.json",
                       weights={"tickets": 1.0, "billing": 0.0, "legacy": 0.0})
    for descriptor in (None, pinned):
        shards = corpus(str(root), "tickets", "train", descriptor=descriptor)
        assert shards is not None
        examples = [row["example"] for row in iterate(shards, read="indexed")]
        assert [example.state for example in examples] == [
            row["state"] for row in ROWS["tickets/0"] + ROWS["tickets/1"]]
        assert [example.labels()["team"] for example in examples] == [0, 1, 0, 1]
        assert examples[2].targets == {"team": (0.75, 0.25)}
        for name, split, rows in (("tickets", "held", ROWS["tickets.held"]),
                                  ("billing", "train", ROWS["billing"])):
            read = corpus(str(root), name, split, descriptor=descriptor)
            assert read is not None
            assert [row["example"].state for row in iterate(read, read="memory")] == [
                row["state"] for row in rows]
        assert corpus(str(root), "billing", "held", descriptor=descriptor) is None
    elsewhere = replacing(root, manifest, "elsewhere.json", provenance={"original_manifest_sha": "0" * 64})
    files = json.loads(json.dumps(manifest["files"]))
    files["billing"]["train"]["sha256"] = "0" * 64
    rewritten = replacing(root, manifest, "rewritten.json", files=files)
    for refused in (elsewhere, rewritten):
        with pytest.raises(ValueError):
            corpus(str(root), "tickets", "train", descriptor=refused)


def test_a_tables_float_labels_name_their_options_by_their_text():
    """A flat table whose label column holds floats, over options it is given, names the option of a
    label's text, as a string label does, and not the option at an index."""
    from dew.data.compile import iterate

    rows = [{"text": f"rating {index}", "label": (0.5, 1.5)[index % 2]} for index in range(4)]
    examples = [row["example"] for row in iterate(Memory(rows).pipe(decisions, options=("1.5", "0.5")),
                                                  read="memory")]
    assert [example.labels()["label"] for example in examples] == [1, 0, 1, 0]


def test_a_named_validation_split_reads_the_source_s_own_split():
    """A holdout of the store's own named split (`holdout(split="validation")`) opens it through the
    source's `with_split`, and the decisions over it read that split's rows; a source with no native
    splits refuses it."""
    from dew.data.compile import iterate

    train = tuple({"text": f"train {index}", "label": "ab"[index % 2]} for index in range(4))
    held = tuple({"text": f"held {index}", "label": "a"} for index in range(2))
    named = _NamedSplits(train, splits={"train": train, "validation": held})
    _, evaluated = named.holdout(split="validation")
    rows = iterate(evaluated.pipe(decisions, options=("a", "b"), question="team"), read="memory")
    assert [row["example"].state for row in rows] == [row["text"] for row in held]
    with pytest.raises(ValueError):
        Memory(train).holdout(split="validation")


def test_a_tables_own_integer_labels_name_their_options_by_value():
    """A flat table whose options are its label column's own values: every
    label names the option of its value, the integers 2 and 10 included,
    rather than the option at that position. The options sort as text."""
    from dew.data.compile import held as held_reader
    from dew.data.dataset import DataPartition
    from dew.data.text import ByteTokenizer
    from dew.decision import Specials
    from dew.decision.layout import StateFirstLayout
    from dew.decision.objective import DecisionObjective
    from dew.nn.backbones.causal_transformer import CausalTransformer

    rows = [{"text": f"row {index}", "label": 10 if index < 3 else 2} for index in range(8)]
    layout = StateFirstLayout(max_len=64, head_max_len=32, option_tokens=4)
    objective = DecisionObjective(
        CausalTransformer(vocab_size=256, emb_features=16, num_layers=1, num_heads=2, mlp_features=32,
                          max_seq_len=layout.max_len),
        tokenizer=ByteTokenizer(), specials=Specials(begin=None, separator=10, marker=0, marker_text="\x00",
                                                     pad=255),
        layout=layout, shuffle_options=False)
    table = Memory(rows).pipe(decisions).read("memory")
    (batch,) = list(held_reader(table.batch(8), objective=objective.data())(DataPartition()))
    scored = batch["scored"].ravel()
    labels = [int(label) for label, real in zip(batch["labels"].ravel(), scored, strict=True) if real]
    # Options ("10", "2"): the three tens are option 0 and the five twos option 1.
    assert sorted(labels) == [0, 0, 0, 1, 1, 1, 1, 1]


def test_a_checkpoint_loops_at_the_passes_and_over_the_block_asked():
    """--loop-steps sets the passes of Ouro's tiny looped decoder, leaving its
    weights as they were; --loop-layers gives the tiny Qwen 3.5, which does not
    loop, a loop over a block of its own layers; a block with no passes, or
    passes with no block, on a decoder that does not loop is refused."""
    from dew.config import looping
    from dew.interop import Pretrained
    from dew.nn.backbones.decoder_stack import Loop

    ouro = Pretrained.load(Path(__file__).parent / "fixtures" / "hf" / "ouro-tiny", dtype="float32")
    once = looping(ouro, 1, None)
    assert once.model.loop == Loop(1, exit_gate=True) and ouro.model.loop == Loop(3, exit_gate=True)
    assert once.variables is ouro.variables
    qwen = Pretrained.load(QWEN, dtype="float32")
    retrofit = looping(qwen, 2, (1, 2))
    assert retrofit.model.language_model.loop == Loop(2, step_norm=False, layers=(1, 2))
    assert qwen.model.language_model.loop is None
    assert retrofit.variables is qwen.variables
    for steps, layers in ((2, None), (None, (1, 2))):
        with pytest.raises(ValueError):
            looping(qwen, steps, layers)


@pytest.mark.parametrize("fields,checkpoint,message", [
    ({"loop_backprop_steps": 1}, QWEN, "has no loop"),
    ({"loop_remat": True}, QWEN, "has no loop"),
    ({"loop_backprop_steps": 1}, TINY, "has no loop"),
    ({"loop_remat": True}, TINY, "has no loop"),
    ({"remat": "unknown"}, QWEN, "remat names one of"),
])
def test_a_decision_run_refuses_memory_options_without_their_backbone(fields, checkpoint, message):
    with pytest.raises(ValueError):
        DecisionRunConfig(pretrained=str(checkpoint), data=TICKETS, **fields).prepare()


def test_a_decision_release_takes_its_backbones_layer_recomputation():
    """`--remat` chooses a decision release's backbone layer policy as a fresh backbone's: the
    prepared objective's backbone recomputes its layers under that policy, under the release's own
    head and layout."""
    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.backbones.decoder_block import remat_policy

    default = Fitted(data=TICKETS)
    objective, _ = dataclasses.replace(default, pretrained=str(TINY), remat="full",
                                       trainer=dataclasses.replace(default.trainer, multi_host=False,
                                                                   compilation_cache_dir=None)).prepare().train("x")
    released = Decide.from_pretrained(TINY, attention_impl="xla")
    backbone = objective.model.backbone
    assert isinstance(backbone, CausalTransformer) and backbone.remat == remat_policy("full")
    assert released.model.backbone.remat != backbone.remat
    assert type(objective.model.head) is type(released.model.head)
    assert type(objective.layout) is type(released.layout)


class Started(DecisionRunConfig):
    """A decision run that hands back the objective it would train, rather than training it."""

    def train(self, objective, dataset, **settings):
        return objective


class Fitted(DecisionRunConfig):
    """A decision run that hands back the objective it would train and the data it would train on."""

    def train(self, objective, dataset, **settings):
        return objective, dataset


def test_a_decision_run_builds_its_source_as_its_source_options_say():
    """A decision run's `--source.text-only` and `--source.prediction-depths` reach its source's load:
    over Qwen3.5, a wrapper with a vision tower and a prediction depth, the run's backbone is the
    language model alone, with no tower and no depth, where by default it builds both."""
    from dew.config import SourceOptions
    from dew.nn.backbones.causal_transformer import CausalTransformer

    default = Fitted(data=TICKETS)
    trainer = dataclasses.replace(default.trainer, multi_host=False, compilation_cache_dir=None)

    def backbone(options: SourceOptions):
        run = dataclasses.replace(default, pretrained=str(QWEN), trainer=trainer, source=options)
        objective, _ = run.prepare().train("x")
        return objective.model.backbone, objective.variables

    whole, _ = backbone(SourceOptions())
    assert not isinstance(whole, CausalTransformer) and whole.language_model.num_nextn_predict_layers == 1
    text, variables = backbone(SourceOptions(text_only=True, prediction_depths=0))
    assert isinstance(text, CausalTransformer) and text.num_nextn_predict_layers == 0
    import jax

    paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_flatten_with_path(variables)[0]]
    assert paths and not any("mtp_" in path or "vision" in path or "visual" in path for path in paths)


def test_a_lora_decision_runs_source_loads_onto_its_trainers_mesh():
    """A decision run that trains an adapter over a source reads the source onto its trainer's mesh
    (`TrainerConfig.source_placement`), as an LM run does: its objective's frozen base is device
    arrays, which the start shares as they are."""
    import jax
    import optax

    from dew.batch import FROZEN
    from dew.decision import DecisionObjective
    from dew.lora import LoRA
    from dew.training import Trainer

    rows = Memory([{"text": text, "label": label} for text, label in
                   [("charged twice", "billing"), ("site down", "technical"), ("pro price", "sales")] * 4])
    asked = rows.pipe(decisions, question="team").read("memory").batch(4).loading(Loading(workers=0))
    default = Started(data=asked)
    config = dataclasses.replace(default, pretrained=str(QWEN), lora=LoRA(rank=2, modules=("q_proj",)),
                                 trainer=dataclasses.replace(default.trainer, multi_host=False,
                                                             compilation_cache_dir=None))
    objective = config.prepare().train("placed")
    assert isinstance(objective, DecisionObjective)
    held = objective.held_variables()
    assert held is not None and FROZEN in held
    base = jax.tree.leaves(held[FROZEN])
    assert base and all(isinstance(leaf, jax.Array) for leaf in base)
    state = Trainer(objective, optax.sgd(0.1), key=jax.random.key(0), mesh=config.trainer.mesh,
                    layout=config.trainer.layout).place()[0]
    assert {id(leaf) for leaf in jax.tree.leaves(state.variables[FROZEN])} == {id(leaf) for leaf in base}
