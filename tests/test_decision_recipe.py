"""A decision run end to end on Laya's toy checkpoint: a CSV in, a calibrated run out."""

import csv
import dataclasses
import json
from collections.abc import Mapping
from pathlib import Path

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


def test_a_decision_mixture_states_its_weights_and_the_manifest_that_made_it(tmp_path):
    """`mixture` mixes the corpora of a positive weight and states, through the chain's facts, the
    weights it mixes them by and the manifest it read, which a run's summary reports."""
    from dew.decision.data import mixture

    question = {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "technical"]}
    row = {"state": "charged twice", "questions": {"team": question}, "answers": {"team": "billing"}}
    manifest = {"weights": {"a": 2.0, "b": 1.0, "c": 0.0}, "slots": {"questions": 1, "options": 2}}
    (tmp_path / "mixture.json").write_text(json.dumps(manifest))
    for name in ("a.jsonl", "a.held.jsonl", "b.jsonl", "c.jsonl"):
        (tmp_path / name).write_text(json.dumps(row) + "\n")
    train, held = mixture(str(tmp_path))
    facts = train.facts()
    assert facts["mixture_weights"] == {"a": 2.0, "b": 1.0}
    assert facts["mixture_made"] == manifest
    assert held is not None


def test_a_decision_corpus_of_one_file_reads_through_every_loading_worker(tmp_path):
    """A corpus whose examples are one JSON-lines file, one physical shard, streams through the run's
    loading workers, each reading the lines of its own part of the file (data.md §10), and a pass
    reads each example once, where a read by more workers than shards was refused."""
    from dew.batch import VALID_ROWS
    from dew.data.compile import compile
    from dew.data.dataset import DataPartition
    from dew.decision.data import corpus

    question = {"type": "choice", "instructions": "Which team?", "criteria": ["billing", "technical"]}
    rows = [{"state": f"ticket {index}", "questions": {"team": question},
             "answers": {"team": ["billing", "technical"][index % 2]}} for index in range(24)]
    (tmp_path / "mixture.json").write_text(json.dumps({"weights": {"a": 1.0}, "slots": {"questions": 1,
                                                                                         "options": 2}}))
    (tmp_path / "a.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    examples = corpus(str(tmp_path), "a", "train")
    assert examples is not None
    chain = examples.read("streamed").repeat(1).batch(4, drop_remainder=False)
    loading = Loading(workers=2, threads=1, read_buffer=1, worker_buffer=1)
    reader = compile(chain, loading=loading).train(DataPartition())
    try:
        # Each worker batches its own part, the last batch of each filled out with rows it marks.
        states = [example.state for batch in reader
                  for example, real in zip(batch["example"], batch.get(VALID_ROWS, [True] * 4), strict=True)
                  if real]
    finally:
        reader.close()
    assert sorted(states) == sorted(row["state"] for row in rows)


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


def test_a_decision_run_reads_letters_as_its_letter_readout():
    """`head:letters` reads each option as the fresh backbone's own logit at its label's token, at the
    answer cue of a row lettered as its labels letter it, as a slot release reads it: the run's head
    is the letter readout of its labels' token ids under the objective's tokenizer, and its rows a
    question to a row, lettered by those labels alone, each bare letter one token of the fixture's
    byte-level vocabulary. A question of more options than the head holds labels is refused."""
    from dew.data.dataset import DataPartition
    from dew.decision.config import Letters
    from dew.decision.head import LabelReadout
    from dew.decision.layout import Labels, QuestionLayout
    from dew.decision.objective import tokenizer_of
    from dew.interop import Pretrained

    default = Fitted(data=TICKETS)
    trainer = dataclasses.replace(default.trainer, multi_host=False, compilation_cache_dir=None)
    labels = Labels(variants=("{}",))
    run = dataclasses.replace(default, pretrained=str(QWEN), trainer=trainer,
                              head=Letters(labels=labels, count=4))
    objective, _ = run.prepare().train("x")
    head, layout = objective.model.head, objective.layout
    tokenizer = tokenizer_of(Pretrained.load(str(QWEN), dtype="float32").text_processor)
    assert tokenizer is not None
    assert isinstance(head, LabelReadout) and head.label_ids == labels.ids(tokenizer, 4)
    assert isinstance(layout, QuestionLayout)
    assert layout.labels == dataclasses.replace(labels, names=labels.names_for(4))
    with pytest.raises(ValueError):
        _, few = dataclasses.replace(run, head=Letters(labels=labels, count=2)).prepare().train("x")
        next(iter(few.train(DataPartition())))


def test_a_decision_run_trains_its_backbone_whole_beside_its_context_layers_where_asked():
    """`--no-context-frozen` trains the backbone whole beside its context layers' blocks, a full
    fine-tune, where by default it holds frozen and the blocks train alone: the objective's held tree
    then splits off no frozen backbone, every backbone weight trained. With LoRA the adapter trains
    beside the blocks over its frozen base; without context layers the flag is refused."""
    from model_support import flat_tree

    from dew.decision.objective import DecisionObjective
    from dew.lora import LoRA
    from dew.nn.cross_attention import crossing
    from dew.objectives.base import FROZEN

    default = Fitted(data=TICKETS)
    trainer = dataclasses.replace(default.trainer, multi_host=False, compilation_cache_dir=None)
    run = dataclasses.replace(default, pretrained=str(QWEN), trainer=trainer, context_layers=(0,),
                              context_length=16)
    frozen, _ = run.prepare().train("x")
    assert isinstance(frozen, DecisionObjective) and frozen.variables is not None
    assert all(crossing(path) for path in flat_tree(frozen.variables["params"]["backbone"]))
    assert "backbone" in frozen.variables[FROZEN]
    whole, _ = dataclasses.replace(run, context_frozen=False).prepare().train("x")
    assert isinstance(whole, DecisionObjective) and whole.variables is not None
    assert FROZEN not in whole.variables
    trained = flat_tree(whole.variables["params"]["backbone"])
    assert any(crossing(path) for path in trained) and any(not crossing(path) for path in trained)
    adapted, _ = dataclasses.replace(run, context_frozen=False,
                                     lora=LoRA(rank=2, modules=("q_proj",))).prepare().train("x")
    assert isinstance(adapted, DecisionObjective) and adapted.variables is not None
    held = flat_tree(adapted.variables["params"]["backbone"])
    assert any(crossing(path) for path in held) and any("lora_A" in path for path in held)
    assert "backbone" in adapted.variables[FROZEN]
    with pytest.raises(ValueError):
        dataclasses.replace(run, context_frozen=False, context_layers=None).prepare()


def test_a_backbone_reading_its_own_media_in_blocks_keeps_a_joint_layout():
    """Gemma 4 Unified reads its vision tokens' blocks whole, as its checkpoint states: that asks no
    block of the run's rows, so a fresh joint head over it keeps the joint layout, a row of every
    question, as a run that asks for no block-bidirectional attention."""
    from dew.decision.config import JointShape
    from dew.decision.layout import JointLayout

    default = Fitted(data=TICKETS)
    trainer = dataclasses.replace(default.trainer, multi_host=False, compilation_cache_dir=None)
    run = dataclasses.replace(default, pretrained=str(Path(__file__).parent / "fixtures" / "hf" /
                                                       "gemma4-unified-native-tiny"),
                              trainer=trainer, head=JointShape(), max_len=128)
    objective, _ = run.prepare().train("x")
    assert isinstance(objective.layout, JointLayout)


@pytest.mark.parametrize("scope", ["wrapper", "text_config"])
def test_a_checkpoint_declaring_block_attention_lays_a_fresh_heads_rows_in_blocks(tmp_path, scope):
    """A checkpoint whose own config declares its full-attention layers block-bidirectional
    (`modeling_overrides`), at the wrapper's scope or its `text_config`'s, lays a fresh head's rows in
    blocks for them, with no run option asking it, as the run's `--source.attention` would."""
    import shutil

    declared = tmp_path / "declared"
    shutil.copytree(QWEN, declared)
    config = json.loads((declared / "config.json").read_text())
    stated = config if scope == "wrapper" else config["text_config"]
    stated["modeling_overrides"] = {"attention": {"full_attention": "block_bidirectional"}}
    (declared / "config.json").write_text(json.dumps(config))
    default = Fitted(data=TICKETS)
    trainer = dataclasses.replace(default.trainer, multi_host=False, compilation_cache_dir=None)
    run = dataclasses.replace(default, pretrained=str(declared), trainer=trainer)
    objective, _ = run.prepare().train("x")
    assert objective.layout.blocks


def test_a_decision_run_lays_its_rows_in_blocks_its_source_reads_whole():
    """`--source.attention full_attention block_bidirectional` loads the backbone's full-attention
    layers reading blocks (`Pretrained.load`'s `modeling_overrides`), and the run's rows lie in blocks
    for them to read; a joint row of every question lies in none, and is refused."""
    from dew.config import SourceOptions
    from dew.decision.config import JointShape
    from dew.interop import Pretrained

    default = Fitted(data=TICKETS)
    trainer = dataclasses.replace(default.trainer, multi_host=False, compilation_cache_dir=None)
    source = SourceOptions(text_only=True, attention={"full_attention": "block_bidirectional"})
    run = dataclasses.replace(default, pretrained=str(QWEN), trainer=trainer, source=source)
    objective, _ = run.prepare().train("x")
    assert objective.model.backbone.kind_of("full_attention").mixer.bidirectional_blocks
    assert objective.layout.blocks
    plain, _ = dataclasses.replace(run, source=SourceOptions(text_only=True)).prepare().train("x")
    release = Pretrained.load(str(QWEN), dtype="float32", text_only=True).model
    assert plain.model.backbone.kind_of("full_attention") == release.kind_of("full_attention")
    assert not plain.layout.blocks
    with pytest.raises(ValueError):
        dataclasses.replace(run, head=JointShape()).prepare()


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
