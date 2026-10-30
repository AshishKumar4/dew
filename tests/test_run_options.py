"""Options of the common run layer reach what consumes them in every kind of run, part of row 1901.

Seven of twenty defects a run's consumers found on GPUs were options a kind of run or the command line
did not hand on (rows 1610, 1772, 1786, 1884, 1889, 1894, 1895). Every kind builds its objective, its
data and its metrics from `RunConfig`'s fields (`prepare`), and trains through `RunConfig.train`, where
alone the trainer, `fit`'s arguments, the checkpoints and the tracker are built from its
`TrainerConfig`. Each kind here, a language model's, a decision's, a diffusion model's and a JEPA's, is
set these and observed where each acts, in the scalars and checkpoints its run writes and the state it
ends on: on the command line, its name and checkpoint directory, steps, accumulation, cadences of logs,
evaluations and checkpoints, checkpoints kept, dynamic loss scale, learning rate, validation metrics and
summaries; from Python, its key, data seed, mesh, layout and adapter. An option a kind builds nothing
from (source options of a fresh model, a model name its source decides, process options its process
does not carry) is refused before it trains, rather than dropped. The trainer's other options, each kind
over, are row 1901's still. Nothing here reads a config's fields back.

Each option a saved task is built with (its objective's `settings`) is derived from its type and passed
through `dew train` to the task the run saves (row 2285, which would have caught row 1061's blockwise
walk, an option with no public path)."""

import csv
import dataclasses
import inspect
import json
import types
import typing
from collections.abc import Mapping
from pathlib import Path

import jax
import numpy as np
import pytest
from objective_data import IMAGES
from test_lm_recipe import SEQ, write_token_files

from dew import registry
from dew.batch import FROZEN
from dew.cli.main import main
from dew.config import ObjectiveConfig, RunConfig, SourceOptions, TrainerConfig
from dew.config.diffusion import DiffusionRunConfig
from dew.config.jepa import JepaRunConfig
from dew.config.lm import LMRunConfig
from dew.data import TFDS, Files, Loading, images as image_steps, presets
from dew.data.pipeline import Rejected
from dew.data.sources.tokens import TokenFiles
from dew.decision import decisions
from dew.decision.config import DecisionRunConfig
from dew.decision.layout import QuestionSegment, Reads, built_in
from dew.diffusion.presets import Flow
from dew.inputs import TextCondition
from dew.lora import LoRA
from dew.registry import ModelConfig, objectives
from dew.sampling import Euler
from dew.training import Layout, MeshSpec
from dew.training.evaluation import Summary

pytestmark = pytest.mark.mesh

TINY_LAYA = Path(__file__).parent / "fixtures" / "laya" / "tiny"
QWEN_TINY = Path(__file__).parent / "fixtures" / "hf" / "qwen38-dense-tiny"
RATE = 0.0123
"""A learning rate no kind takes by default, which the run's optimizer logs as it steps."""


@dataclasses.dataclass(frozen=True)
class MostlyKept:
    """A record kept by its draw three times in four (`Pipeline.random_filter_map`): which records a
    batch holds is then the run's data seed's."""

    identity = "mostly-kept@1"

    def __call__(self, row, rng: np.random.Generator):
        return row if rng.random() < 0.75 else Rejected("drawn out")


@dataclasses.dataclass(frozen=True)
class Dimmed:
    """An image dimmed by a factor its record draws (`presets.images`' `augment`): what a batch's
    pixels hold is then the run's data seed's."""

    identity = "dimmed@1"

    def __call__(self, row, rng: np.random.Generator):
        image = np.asarray(row["image"])
        return {**row, "image": (image * rng.uniform(0.5, 1.0)).astype(image.dtype)}

    def declared(self, facts):
        # The image keeps its shape and dtype.
        return facts


def read(chain):
    return chain.read("memory").loading(Loading(workers=0))


def lm(root: Path, drawn: bool) -> RunConfig:
    corpus = write_token_files(root / "tokens", 40 * SEQ, 8 * SEQ)

    def windows(path, *, drawn: bool):
        chain = TokenFiles(path).pipe(presets.lm, None, seq_len=SEQ, batch=None, pack=None)
        return read((chain.random_filter_map(MostlyKept()) if drawn else chain).batch(8))
    return LMRunConfig(model=ModelConfig("causal_transformer", {"emb_features": 16, "num_layers": 1,
                                                                "num_heads": 2, "dtype": "float32"}),
                       data=windows(corpus.train, drawn=drawn), validation=windows(corpus.val, drawn=False))


def decision(root: Path, drawn: bool) -> RunConfig:
    table = root / "tickets.csv"
    asked = [("charged twice for March", "billing"), ("the site is down", "technical"),
             ("how much is the pro plan", "sales"), ("refund my last invoice", "billing")]
    # Each ticket its own text, so the held-out split, cut by a row's content, holds some of each.
    tickets = [(f"{text}, ticket {index}", label) for index in range(8) for text, label in asked]
    with table.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["text", "label"])
        writer.writerows(tickets)
    train, held = Files(str(table), format="csv").holdout(0.25)

    def asked(rows, *, drawn: bool):
        chain = rows.pipe(decisions, question="team").read("memory")
        kept = chain.random_filter_map(MostlyKept()) if drawn else chain
        return kept.batch(8).loading(Loading(workers=0))
    return DecisionRunConfig(pretrained=str(TINY_LAYA), model=ModelConfig("causal_transformer",
                                                                          {"dtype": "float32"}),
                             data=asked(train, drawn=drawn), validation=asked(held, drawn=False))


def pictures():
    return TFDS("dew_images", path=str(IMAGES), split="all").holdout(records=4)


def diffusion(root: Path, drawn: bool) -> RunConfig:
    del root
    train, held = pictures()

    def captioned(rows, *, drawn: bool):
        return read(rows.pipe(presets.images, 4, batch=8, caption=image_steps.caption(),
                              augment=Dimmed() if drawn else False))
    return DiffusionRunConfig(
        model=ModelConfig("simple_dit", {"patch_size": 2, "emb_features": 16, "num_layers": 1,
                                         "num_heads": 2}),
        data=captioned(train, drawn=drawn), validation=captioned(held, drawn=False), preset=Flow(),
        val_metrics=(),
        text=TextCondition(encoder="char_table", checkpoint="char_table"),
        objective=ObjectiveConfig("diffusion", {"solver": Euler(), "steps": 2}))


def jepa(root: Path, drawn: bool) -> RunConfig:
    del root
    train, held = pictures()

    def patched(rows, *, drawn: bool):
        return read(rows.pipe(presets.images, 16, batch=8, augment=Dimmed() if drawn else False))
    return JepaRunConfig(
        model=ModelConfig("jepa_encoder", {"patch_size": 2, "emb_features": 16, "num_layers": 1,
                                           "num_heads": 2, "mlp_ratio": 2, "dtype": "float32",
                                           "attention_impl": "xla"}),
        predictor={"num_layers": 1, "predictor_features": 8}, num_target_blocks=2,
        data=patched(train, drawn=drawn), validation=patched(held, drawn=False))


KINDS = {"lm": lm, "decision": decision, "diffusion": diffusion, "jepa": jepa}
ADAPTED = {"lm": ("q_proj",), "decision": ("q_proj",), "diffusion": ("ada_proj",), "jepa": ("up_proj",)}
"""A module of each kind's model a `lora` adapts."""
DEFAULT_METRICS = {"lm": "val/perplexity", "decision": "val/accuracy"}
"""A metric each kind scores where `val_metrics` names none, for the kinds that score one."""


def scalars(run: Path) -> dict[int, dict[str, float]]:
    """Each step's scalars the run logged, joined."""
    logged = {}
    for line in (run / "tracking" / "scalars.jsonl").read_text().splitlines():
        row = json.loads(line)
        logged.setdefault(row["step"], {}).update(row["scalars"])
    return logged


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("kind", list(KINDS))
def test_a_kinds_run_set_on_the_command_line_pools_logs_scores_and_keeps_as_set(kind, tmp_path):
    """`dew train` over a kind's run.json with its options set (`--set`): four steps pooled two to an
    update commit at the second's and the fourth's end (`train/grad_norm`), under a dynamic loss
    scale (`train/loss_scale`) and the learning rate set (`train/learning_rate`); every step is
    logged and scored on the held split, with no metric the kind would score by default and the
    summary named formed from the scores; and every step's checkpoint is kept, four, where the
    default keeps the latest two and the best, in the directory named."""
    record = tmp_path / "record"
    KINDS[kind](tmp_path, drawn=False).save(str(record))
    runs = tmp_path / "runs"
    settings = {"trainer.name": "set", "trainer.checkpoint_dir": str(runs), "trainer.steps": "4",
                "trainer.accumulation": "2", "trainer.log_every": "1", "trainer.eval_every": "1",
                "trainer.checkpoint_every": "1", "trainer.keep": "4", "model.dtype": "float16",
                "trainer.multi_host": "False", "trainer.compilation_cache_dir": "None",
                "optim.learning_rate": str(RATE), "val_metrics": "[]",
                "val_summaries": json.dumps([{"name": "board", "weights": {"val/loss": 1},
                                              "better": "lower"}])}
    assert main(["train", str(record / "run.json"),
                 *(flag for path, value in settings.items() for flag in ("--set", f"{path}={value}"))]) == 0
    run = runs / "set"
    logged = scalars(run)
    assert {step for step, row in logged.items() if "train/loss" in row} == {1, 2, 3, 4}
    committed = sorted(step for step, row in logged.items() if "train/grad_norm" in row)
    # Its model computes in float16, which scales the loss (`DynamicScale`): a commit whose gradients
    # overflow skips its update and lowers the scale, as a loaded backbone's first may.
    assert set(committed) <= {2, 4} and 4 in committed
    if 2 not in committed:
        assert logged[2]["train/loss_scale"] < logged[1]["train/loss_scale"]
    for step in (1, 2, 3, 4):
        row = logged[step]
        # The optimizer holds its rate in float32, which the log reads back.
        assert row["train/learning_rate"] == float(np.float32(RATE))
        assert "train/loss_scale" in row
        assert row[f"{Summary.SPLIT}/board"] == row["val/loss"]
        if kind in DEFAULT_METRICS:
            assert DEFAULT_METRICS[kind] not in row
    assert sorted(path.name for path in run.iterdir() if path.name.isdigit()) == ["1", "2", "3", "4"]


def trained(run: RunConfig, root: Path, name: str, **fields):
    """`run` trained one step from Python (`prepare().train`) over a mesh of four data shards by two
    fsdp shards, its trainer the fields given, over its defaults here: its state, and its step's
    scalars."""
    trainer = TrainerConfig(**{"checkpoint_dir": str(root / "runs"), "steps": 1, "log_every": 1,
                               "eval_every": None, "checkpoint_every": None, "checkpoint_final": False,
                               "multi_host": False, "mesh": MeshSpec(fsdp=2), "layout": Layout(min_shard=64),
                               **fields.pop("trainer", {})})
    state = dataclasses.replace(run, trainer=trainer, **fields).prepare().train(name)
    return state, scalars(root / "runs" / name)[1]


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("kind", list(KINDS))
def test_a_kinds_key_data_seed_mesh_and_adapter_reach_what_it_trains(kind, tmp_path):
    """From Python, a kind's run trains under each option it states: another `trainer.key` is another
    run key in its state, and draws other parameters where the kind draws them fresh (a decision run
    reads its release's), another `data_seed` reads another first batch from parameters drawn alike,
    its mesh and layout split a parameter over its fsdp shards, and a `lora` trains its factors over a
    frozen base."""
    run = KINDS[kind](tmp_path, drawn=True)
    base, first = trained(run, tmp_path, "base", trainer={"key": 1})
    keyed, _ = trained(run, tmp_path, "keyed", trainer={"key": 2})
    _, reordered = trained(run, tmp_path, "seeded", trainer={"key": 1}, data_seed=5)
    held = [np.asarray(leaf) for leaf in jax.tree.leaves(base.variables["params"])]
    assert not np.array_equal(jax.random.key_data(base.key), jax.random.key_data(keyed.key))
    drawn = any(not np.array_equal(mine, np.asarray(theirs))
                for mine, theirs in zip(held, jax.tree.leaves(keyed.variables["params"]), strict=True))
    assert drawn is (kind != "decision")
    assert first["train/loss"] != reordered["train/loss"]
    assert any(len(leaf.sharding.device_set) > 1 and not leaf.sharding.is_fully_replicated
               for leaf in jax.tree.leaves(base.variables["params"]))
    adapted, _ = trained(run, tmp_path, "adapted", trainer={"key": 1},
                         lora=LoRA(rank=2, modules=ADAPTED[kind]))
    assert FROZEN in adapted.variables
    factors = sum(leaf.size for leaf in jax.tree.leaves(adapted.variables["params"]))
    assert 0 < factors < sum(leaf.size for leaf in held)


@pytest.mark.parametrize("kind", ["lm", "diffusion", "jepa"])
def test_a_kind_that_builds_its_model_fresh_refuses_source_options(kind, tmp_path):
    """`source` says what a run builds of the source it loads, and a run whose model is built from
    `model` loads none: its source options are refused as it is prepared, not dropped."""
    run = KINDS[kind](tmp_path, drawn=False)
    with pytest.raises(ValueError):
        dataclasses.replace(run, source=SourceOptions(text_only=True)).prepare()


@pytest.mark.parametrize("kind", ["lm", "decision"])
def test_a_kind_whose_source_decides_its_model_refuses_another_model_name(kind, tmp_path):
    """A run that starts from a source takes its architecture from it, and its model's fields'
    few choices alone: a model it names is refused as it is prepared, not dropped."""
    run = KINDS[kind](tmp_path, drawn=False)
    model = ModelConfig("simple_dit", {})
    pretrained = {"pretrained": str(TINY_LAYA)} if kind == "lm" else {}
    with pytest.raises(ValueError):
        dataclasses.replace(run, model=model, **pretrained).prepare()


@pytest.mark.parametrize("kind", list(KINDS))
@pytest.mark.parametrize("option", ["xla_flags", "compilation_cache_dir", "multi_host"])
def test_a_process_option_the_process_does_not_carry_is_refused_before_training(kind, option, tmp_path):
    """XLA's flags, the compilation cache and the process pool are set before JAX opens a backend,
    as `run()` and `dew train` do: a run prepared and trained in a process that carries none of the
    one it states is refused before it trains, not trained without it."""
    # A flag value no lane sets (tests/lane_environment.py), so this process carries none of it.
    stated = {"xla_flags": "--xla_dump_hlo_pass_re=run-options-1901",
              "compilation_cache_dir": str(tmp_path / "cache"), "multi_host": True}[option]
    run = KINDS[kind](tmp_path, drawn=False)
    trainer = TrainerConfig(**{"checkpoint_dir": str(tmp_path / "runs"), "steps": 1, "eval_every": None,
                               "multi_host": False, option: stated})
    prepared = dataclasses.replace(run, trainer=trainer).prepare()
    with pytest.raises(ValueError):
        prepared.train("refused")
    assert not (tmp_path / "runs" / "refused").exists()



# ----------------------------------------------------------- options a saved task is built with (row 2285)

def masked(root: Path, drawn: bool) -> RunConfig:
    """`lm`'s run as a masked-diffusion one, its model bidirectional with a mask id, which previews
    nothing."""
    run = lm(root, drawn)
    model = ModelConfig("causal_transformer", {**run.model.fields, "causal": False, "mask_token_id": 0})
    return dataclasses.replace(run, objective=ObjectiveConfig("masked_diffusion"), model=model,
                               max_new_tokens=0)


def members_of(annotation: type, base=None) -> list[object]:
    """The values of Dew's classes that are `annotation`s: every member of a registry kind
    (`registry.KINDS`) and every class its package exports (`dew.sampling`'s CFG, APG, ...) that derives
    from `annotation`, or, for a protocol, holds each method it declares, each built from its defaults,
    its required fields `base`'s of those names, the value a run saved, which one without them lacks."""
    import importlib

    package = importlib.import_module(".".join(annotation.__module__.split(".")[:2]))
    found = [registry.imported(path) for kind in registry.KINDS for path in kind.paths.values()]
    found += [getattr(package, name) for name in getattr(package, "__all__", ())]
    declared = {name for name in vars(annotation) if not name.startswith("_")}
    protocol = registry.is_protocol(annotation)
    values = []
    for member in dict.fromkeys(found):
        if not isinstance(member, type):
            continue
        fits = member is not annotation and (
            all(callable(getattr(member, name, None)) for name in declared) if protocol
            else issubclass(member, annotation))
        if not fits:
            continue
        required = [field.name for field in dataclasses.fields(member) if field.init and field.default
                    is dataclasses.MISSING and field.default_factory is dataclasses.MISSING] if (
            dataclasses.is_dataclass(member)) else []
        if not all(hasattr(base, name) for name in required):
            continue
        try:
            values.append(member(**{name: getattr(base, name) for name in required}))
        except (TypeError, ValueError):
            continue
    return values


WITNESSES = {Reads: (Reads((("row", "self", "whole"),)),),
             tuple[QuestionSegment, ...]: (built_in(None, "question_first"),)}
"""Values of Dew's classes that build from no defaults, and of tuples of them, each one a run here trains
with: a row's reads of its streams (`Reads`), here its one stream reading its whole document both ways,
and a question layout's own row written as its segments (`built_in`)."""


def members_built(member: type) -> list[object]:
    """`member` built from its defaults, where it has them, else its `WITNESSES`."""
    try:
        return [member()]
    except (TypeError, ValueError):
        return list(WITNESSES.get(member, ()))


def alternatives(annotation, base=None) -> list[object]:
    """The values `annotation` declares as alternatives, in its order: a literal's values, a flag's two, an
    optional's None and its member's (an optional number's beside the saved one), Dew's classes that are
    one (`members_of`), and a dataclass's instances: `base`, the value a run saved, with one of its fields
    changed to each of that field's alternatives, or, with no base, built over its fields' alternatives. A
    plain number or text declares none, nor a record (a union holding a mapping or a list), whose values
    are its writer's."""
    if isinstance(annotation, typing.TypeAliasType):
        return alternatives(annotation.__value__, base)
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return alternatives(typing.get_args(annotation)[0], base)
    if origin is typing.Literal:
        return list(typing.get_args(annotation))
    if annotation is bool:
        return [False, True]
    if origin is tuple:
        return list(WITNESSES.get(annotation, ()))
    if origin in (types.UnionType, typing.Union):
        if record(annotation):
            return []
        members = typing.get_args(annotation)
        found = []
        for member in members:
            # A count's witnesses: one, and one past the saved count, which a budget that one would cut
            # to nothing takes (a layout's option tokens); a number's, a half and twice the saved one.
            counts = [1, *([base + 1] if type(base) is int else [])]
            numbers = [0.5, *([base * 2] if type(base) is float else [])]
            values = ([None] if member is type(None) else counts if member is int
                      else numbers if member is float else alternatives(member, base))
            if typing.get_origin(member) is tuple:
                values = [value for value in values if value != base]
            if dataclasses.is_dataclass(member) and not isinstance(base, member):
                values = [*members_built(member), *values]
            found += [value for value in values if value not in found]
        return found
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        if isinstance(base, annotation):
            # The saved value with one field changed, each field's alternatives in turn: the fields of
            # its own class, a subclass's beside the annotation's.
            saved = type(base)
            hints = typing.get_type_hints(saved)
            changed = []
            for name in (field.name for field in dataclasses.fields(saved) if field.init):
                for value in alternatives(hints[name], getattr(base, name)):
                    try:
                        changed.append(dataclasses.replace(base, **{name: value}))
                    except (TypeError, ValueError):
                        continue
            return [base, *(value for value in changed if value != base)]
        hints = typing.get_type_hints(annotation)
        fields = [field.name for field in dataclasses.fields(annotation) if field.init]
        declared = {name: alternatives(hints[name]) for name in fields}
        declared = {name: values for name, values in declared.items() if len(values) > 1}
        try:
            built = [annotation(**{name: values[index] for name, values in declared.items()})
                     for index in (0, 1)]
        except (TypeError, ValueError):
            return members_of(annotation, base)
        return list(built) if declared and built[0] != built[1] else []
    if isinstance(annotation, type) and annotation.__module__.startswith("dew."):
        return members_of(annotation, base)
    return []


def record(annotation) -> bool:
    """Whether `annotation` is a record's: a union holding a mapping or a list, whose values its writer's
    are."""
    if isinstance(annotation, typing.TypeAliasType):
        return record(annotation.__value__)
    if typing.get_origin(annotation) is typing.Annotated:
        return record(typing.get_args(annotation)[0])
    return typing.get_origin(annotation) in (types.UnionType, typing.Union) and any(
        typing.get_origin(member) in (dict, list, typing.Mapping) or member in (dict, list)
        for member in typing.get_args(annotation))


def dews(annotation) -> bool:
    """Whether `annotation` names one of Dew's types anywhere within it: alone, in a union, an alias or a
    container's arguments (a tuple of a layout's segments)."""
    if isinstance(annotation, typing.TypeAliasType):
        return dews(annotation.__value__)
    if isinstance(annotation, type) and typing.get_origin(annotation) is None:
        return annotation.__module__.startswith("dew.")
    return any(dews(argument) for argument in typing.get_args(annotation) if argument is not Ellipsis)


def bare(annotation):
    """`annotation` as a field of its type declares it: an alias's value, an annotated type's own, an
    optional's member."""
    if isinstance(annotation, typing.TypeAliasType):
        return bare(annotation.__value__)
    if typing.get_origin(annotation) is typing.Annotated:
        return bare(typing.get_args(annotation)[0])
    members = [member for member in typing.get_args(annotation) if member is not type(None)]
    if typing.get_origin(annotation) in (types.UnionType, typing.Union) and len(members) == 1:
        return bare(members[0])
    return annotation


@dataclasses.dataclass(frozen=True)
class Reach:
    """Where `dew train` sets an option (`reach`): the run's field `path`, of type `annotation`, whose value
    is the option, or, where `builds`, a builder whose call makes it (a preset its process)."""

    path: str
    annotation: object
    builds: bool = False

    def made(self, value):
        """The option a value of this field gives the task."""
        return value() if self.builds and value is not None else value

    def makes(self, value) -> bool:
        """Whether a value of this field gives the task an option: a builder may refuse its own numbers."""
        try:
            self.made(value)
        except ValueError:
            return False
        return True


def builds(annotation, option) -> bool:
    """Whether a value of `annotation` is a builder whose call returns an `option` (a preset its process)."""
    members = [member for member in typing.get_args(bare(annotation)) or (bare(annotation),)
               if member is not type(None)]
    def made(member) -> set:
        returned = typing.get_type_hints(member.__call__).get("return") if callable(member) else None
        return set(typing.get_args(returned) or (returned,))
    return any(isinstance(member, type) and bare(option) in made(member) for member in members)


def reach(run: RunConfig, objective: str, name: str, annotation, argument: bool = True) -> Reach | None:
    """Where `dew train` sets a saved task's option `name` of `annotation`: a field of the run, or of a
    section it holds, declared of that type (a number or a text's of that name) or a builder of it
    (`builds`), else, with `argument`, its objective's argument of that name; None where it has none."""
    # A number's or a text's field is the option's by its name; any other type's, by its type.
    named = bare(annotation) in (int, float, str, bool)

    def paths(config, prefix: str):
        hints = typing.get_type_hints(type(config), include_extras=True)
        for field in dataclasses.fields(config):
            value, hint = getattr(config, field.name), hints[field.name]
            if field.name == name if named else bare(hint) == bare(annotation):
                yield Reach(prefix + field.name, hint)
            elif not named and builds(hint, annotation):
                yield Reach(prefix + field.name, hint, builds=True)
            elif dataclasses.is_dataclass(value) and type(value).__module__.startswith("dew.config"):
                yield from paths(value, f"{prefix}{field.name}.")
    found = next(paths(run, ""), None)
    if found is None and argument and name in inspect.signature(objectives[objective]).parameters:
        return Reach(f"objective.{name}", annotation)
    return found


@dataclasses.dataclass(frozen=True)
class Option:
    """An option a saved task is built with: its name in the settings, dotted where it is a field of one
    of theirs (`layout.reads`), its alternatives (`witnessed`), where `dew train` sets it (`reach`), None
    where nothing does, and the field that declares it."""

    name: str
    values: list[object]
    found: Reach | None
    field: dataclasses.Field


def unwitnessed(kind: type, options) -> list[str]:
    """The fields of `kind` of Dew's types (`dews`) that declare fewer than two alternatives, so no option
    of `options` witnesses them, none declared derived from the run nor a record, its writer's."""
    hints = typing.get_type_hints(kind)
    return [field.name for field in dataclasses.fields(kind) if field.name not in options
            and dews(hints[field.name]) and "derived" not in field.metadata and not record(hints[field.name])]


def options_of(run: RunConfig, objective: str, kind: type, base, prefix: str = "") -> list[Option]:
    """The options `base`, a `kind` a run saved, holds (`witnessed`), each where `dew train` sets it. A
    field its class declares derived is the run's, set by no objective argument, which the run fills
    (`ObjectiveConfig.build`); where its saved value is a dataclass the run builds from fields of its own
    (a decision run's layout from its budgets, `_budgeted`), its options are that value's fields', each
    reached as the run sets it. A field of Dew's types that no alternative witnesses fails."""
    hints = typing.get_type_hints(kind)
    fields = {field.name: field for field in dataclasses.fields(kind)}
    declared = witnessed(kind, base)
    missing = unwitnessed(kind, declared)
    assert not missing, f"{kind.__name__}'s {', '.join(missing)} declares no alternative to witness it by"
    found = []
    for name, values in declared.items():
        derived = "derived" in fields[name].metadata
        path = reach(run, objective, name, hints[name], argument=not derived)
        value = getattr(base, name)
        if path is None and derived and dataclasses.is_dataclass(value) and not isinstance(value, type):
            found += options_of(run, objective, type(value), value, f"{prefix}{name}.")
        else:
            found.append(Option(prefix + name, values, path, fields[name]))
    return found


def held_at(value, name: str):
    """The option a dotted `name` names in `value`, a field of a field where it is dotted."""
    for part in name.split("."):
        value = getattr(value, part)
    return value


def replaced(value, name: str, option):
    """`value` with the option a dotted `name` names in it replaced by `option`."""
    head, _, rest = name.partition(".")
    inner = replaced(getattr(value, head), rest, option) if rest else option
    return dataclasses.replace(value, **{head: inner})


def flag(value, annotation) -> str:
    """`value` as `--set` writes it for a field of `annotation`: a record's JSON, a tuple's values, or its
    text."""
    if value is None or isinstance(value, bool | int | str):
        return str(value)
    # A tuple of numbers or texts as its values in turn, as the command line reads one.
    if isinstance(value, tuple) and all(isinstance(item, bool | int | str) for item in value):
        return " ".join(map(str, value))
    return json.dumps(registry.to_record(value, bare(annotation)))


def prefixed(root: Path, drawn: bool) -> RunConfig:
    """`lm`'s run over a decoder whose attention reads a prefix's blocks, which a prefix run needs."""
    from dew.nn.backbones.causal_transformer import CausalTransformer, LayerKind
    from dew.nn.mixers.attention import AttentionMixer

    run = lm(root, drawn)
    kinds = {"blocks": LayerKind(mixer=AttentionMixer(bidirectional_blocks=True))}
    record = registry.to_record(kinds, typing.get_type_hints(CausalTransformer)["kinds"])
    return dataclasses.replace(run, model=ModelConfig("causal_transformer", {
        **run.model.fields, "layer_types": ["blocks"], "kinds": record}))


BESIDE = {"layout.framing": {"segments": built_in(None, "question_first"), "max_len": 256},
          "context_encoder": {"context_layers": (0,), "context_length": 16}}
"""What an option's reader takes it only beside, by the run's path, set with it: a framing frames a row of
its own segments (`_budgeted`), here the question layout's own row as segments (`built_in`), in a row long
enough for its conversation's turns; a context's encoder encodes it for context layers, which read a
row's context."""

def blocked(root: Path, drawn: bool) -> RunConfig:
    """`decision`'s run as a fresh head over a tiny causal backbone whose full attention reads blocks
    (`--source.attention`), whose rows a split takes, which a release's rows never are."""
    return dataclasses.replace(decision(root, drawn), pretrained=str(QWEN_TINY), source=SourceOptions(
        text_only=True, attention={"full_attention": "block_bidirectional"}))


TASK_RUNS = {"TextSettings": ((prefixed, "lm"),), "MaskedSettings": ((masked, "masked_diffusion"),),
             "ImageSettings": ((diffusion, "diffusion"),),
             "DecideSettings": ((decision, "decision"), (blocked, "decision"))}
"""The runs here of an objective that records each saved task's settings, and the objective's name: a
decision release's, which exports, and a fresh head's over a backbone that reads blocks."""


def witnessed(settings: type, base=None) -> dict[str, list[object]]:
    """Each field of `settings` that declares alternatives (`alternatives`), with them: about `base`'s
    value of it, the settings a run saved, where given."""
    hints = typing.get_type_hints(settings)
    declared = {field.name: alternatives(hints[field.name], getattr(base, field.name, None))
                for field in dataclasses.fields(settings)}
    return {name: values for name, values in declared.items() if len(values) > 1}


SETTINGS = sorted({cls.settings for cls in (objectives[name] for name in objectives)
                   if cls.settings is not None and witnessed(cls.settings)},
                  key=lambda settings: settings.__name__)


def held_by(task, depth: int = 3) -> dict[str, tuple[object, str]]:
    """Where the loaded `task` holds each of its values, by path: the object and its attribute's name, for
    its attributes, and theirs where they are Dew's objects or dataclasses (its model's fields), `depth`
    deep."""
    found, frontier = {}, [("", task)]
    for _ in range(depth):
        following = []
        for prefix, value in frontier:
            names = ([field.name for field in dataclasses.fields(value)] if dataclasses.is_dataclass(value)
                     else list(vars(value)) if type(value).__module__.startswith("dew.") else [])
            for name in names:
                inner = getattr(value, name, None)
                found[prefix + name] = (value, name)
                if dataclasses.is_dataclass(inner) or type(inner).__module__.startswith("dew."):
                    following.append((f"{prefix}{name}.", inner))
        frontier = following
    return found


def held(task) -> dict[str, object]:
    """What the loaded `task` holds, by path (`held_by`)."""
    return {path: getattr(holder, name, None) for path, (holder, name) in held_by(task).items()}


def requested(task) -> object:
    """One request of a loaded task, as its kind's users make it, at the task's own defaults but a
    generation's budget of two tokens, its answers: a decision's with a context beside its state, which
    a backbone reading none leaves unread."""
    from dew.decision import Choice
    from dew.decision.task import Decide
    from dew.inference.images import TextToImage
    from dew.inference.tasks import BlockGeneration, MaskedGeneration, TextGeneration

    if isinstance(task, TextGeneration):
        return task("ab", max_new_tokens=2, key=0)
    if isinstance(task, MaskedGeneration | BlockGeneration):
        # Token ids, which a source with no tokenizer of its own (dream-tiny) reads too.
        return task([[5, 17, 33, 2]], max_new_tokens=2, key=0)
    if isinstance(task, Decide):
        question = {"team": Choice("which team", ["billing", "technical", "sales"])}
        return task([({"text": "the site is down"}, question)], contexts=[{"plan": "enterprise"}])
    if isinstance(task, TextToImage):
        return task(["a cat"], key=0)
    raise AssertionError(f"no request is written for a loaded {type(task).__name__}")


def answers(task) -> list[object]:
    """What one request of `task` answers (`requested`), each leaf a host value, in order."""
    def leaves(value) -> list[object]:
        if isinstance(value, Mapping):
            return [leaf for name in sorted(value) for leaf in leaves(value[name])]
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return [leaf for field in dataclasses.fields(value)
                    for leaf in leaves(getattr(value, field.name))]
        if isinstance(value, list | tuple):
            return [leaf for item in value for leaf in leaves(item)]
        return [np.asarray(value) if isinstance(value, jax.Array | np.ndarray) else value]
    return leaves(requested(task))


def answered_alike(first: list[object], second: list[object]) -> bool:
    """Whether two requests' answers (`answers`) are one: each leaf equal, an array's in shape and in
    every entry."""
    def alike(a, b) -> bool:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape
                    and np.array_equal(a, b, equal_nan=True))
        return bool(a == b)
    return len(first) == len(second) and all(alike(a, b) for a, b in zip(first, second, strict=True))


def trained_by_cli(run: RunConfig, root: Path, name: str, assigned: dict[str, str]) -> Path:
    """The run directory one step of `dew train` writes for `run` with each `path=value` of `assigned`."""
    record, runs = root / f"{name}-record", root / "runs"
    run.save(str(record))
    given = {"trainer.name": name, "trainer.checkpoint_dir": str(runs), "trainer.steps": "1",
             "trainer.eval_every": "None", "trainer.checkpoint_every": "None",
             "trainer.multi_host": "False", "trainer.compilation_cache_dir": "None", **assigned}
    assert main(["train", str(record / "run.json"),
                 *(part for path, value in given.items() for part in ("--set", f"{path}={value}"))]) == 0
    return runs / name


def saved_with(run: RunConfig, root: Path, name: str, settings: type, assigned: dict[str, str]):
    """The `settings` the task of `run` saves after one step of `dew train` with each `path=value` of
    `assigned` set (`trained_by_cli`), and the task loaded from them (`dew.pipeline`)."""
    import dew
    from dew.inference.tasks import SavedRun, saved_settings

    directory = trained_by_cli(run, root, name, assigned)
    return saved_settings(SavedRun.pinned(str(directory)).record, settings), dew.pipeline(str(directory))


def same(first, second) -> bool:
    """Whether two values are one option: of one type and one record (`registry.to_record`), as a run
    saves and reads them, where a schedule compares by its numbers, not its identity."""
    if type(first) is not type(second):
        return False
    try:
        return registry.to_record(first, type(first)) == registry.to_record(second, type(second))
    except (TypeError, ValueError):
        return bool(first == second)


def accepted(saved, name: str, value) -> bool:
    """Whether `saved` settings take `value` for the option `name` names beside the rest they hold."""
    try:
        replaced(saved, name, value)
    except (TypeError, ValueError):
        return False
    return True


def reached_by(run: RunConfig, root: Path, settings: type, flags, queued: dict[str, list[object]], holds):
    """Each option of `queued` set alone through `dew train` (`flags` writes its `--set`s), so what its
    task holds differs from the run that sets none by it alone: at its first alternative, or its next where
    a reader refuses one or the saved settings do not hold it (`holds`: a run's None that keeps the
    layout's own). The task each loads and the value it was set to, for each that trains so, and why
    each other alternative did not."""
    reached, refused = {}, {}
    for name, values in queued.items():
        for index, value in enumerate(values):
            try:
                saved, task = saved_with(run, root, f"{name}{index}", settings, flags({name: value}))
            except ValueError as error:  # A reader refused it, which names what refused it: the next.
                refused.setdefault(name, []).append(f"{value!r}: {error}")
                continue
            if holds(saved, name, value):
                reached[name] = (task, value)
                break
            refused.setdefault(name, []).append(f"{value!r}: saved {held_at(saved, name)!r}")
    return reached, refused


def witnessed_on(builder, objective: str, settings: type, root: Path
                 ) -> tuple[dict[str, Option], dict[str, str], bool]:
    """The options of `settings` the run `builder` makes saves (`options_of`), and why each the run does not
    witness is not: one with no path its settings do not declare derived, or derived from data the run
    leaves unfilled; one set alone to each alternative (`reached_by`) that no run trains with, whose
    loaded task holds it nowhere it held another before (`held`), or whose request answers as the same
    task does with what it held before in its place, the consumer's work, not its record (`unread`): a
    released task's, its export's too (`dew export`), where the run exports, which it says."""
    import dew
    from dew.objectives.base import ReleasedTask

    root.mkdir(parents=True)
    run = builder(root, drawn=False)
    saved_task = objectives[objective].saved_task
    released = saved_task is not None and issubclass(saved_task, ReleasedTask)
    base, unset = saved_with(run, root, "unset", settings, {})
    before = held(unset)
    # A released task's run exports where its model's format holds it (a model with cross-attention
    # layers its checkpoint keeps alone): its witnesses' exports then hold them too.
    exports = released and exported(root / "runs" / "unset", root / "exported" / "unset")
    options = {option.name: option for option in options_of(run, objective, settings, base)}
    failures = {}
    for name, option in options.items():
        if option.found is None and "derived" not in option.field.metadata:
            failures[name] = "it has no path"
        # A derived field the run fills, from its data or the option its declaration names (`--head`'s
        # labels): on one run at least, where it holds other than its default.
        elif (option.found is None and option.field.default is not dataclasses.MISSING
              and same(held_at(base, name), option.field.default)):
            failures[name] = "the run fills none"
    # A builder's path is witnessed by its own alternatives, each the option its call makes; an option
    # set beside what its reader takes it with (`BESIDE`) is judged by the run alone.
    queued = {name: [value for value in (alternatives(found.annotation) if found.builds else option.values)
                     if found.makes(value) and not same(found.made(value), held_at(base, name))
                     and (name in BESIDE or accepted(base, name, found.made(value)))]
              for name, option in options.items() if (found := option.found) is not None}
    reachable = {name: option.found for name, option in options.items() if option.found is not None}
    hints = typing.get_type_hints(type(run), include_extras=True)

    def flags(assigned: dict[str, object]) -> dict[str, str]:
        """The `--set`s of each option `assigned`, and of what its reader takes it only beside (`BESIDE`)."""
        beside = {path: flag(value, hints[path])
                  for name in assigned for path, value in BESIDE.get(name, {}).items()}
        return {**beside, **{reachable[name].path: flag(value, reachable[name].annotation)
                             for name, value in assigned.items()}}
    reached, refused = reached_by(run, root, settings, flags, queued, lambda saved, name, value: same(
        held_at(saved, name), reachable[name].made(value)))
    for name in queued:
        if name not in reached:
            failures[name] = f"no alternative trains: {refused.get(name)}"
            continue
        task, value = reached[name]
        option = reachable[name].made(value)
        missed = unread(task, before, option)
        destination = root / "exported" / name
        # A released task's export (`dew export`), reloaded as its users load one, holds it too; a format
        # that holds no such option refuses it by name (a release's holds its budgets alone), never drops it.
        if missed is None and exports and exported(root / "runs" / f"{name}{queued[name].index(value)}",
                                                   destination):
            missed = unread(dew.pipeline(str(destination)), before, option)
            missed = None if missed is None else f"its export: {missed}"
        if missed is not None:
            failures[name] = missed
    return options, failures, exports


UNHELD = ("states a layout's window, budgets and option layout alone",
          "Laya's layout holds a ModernBERT encoder")
"""What `dew export` says refusing a run whose format holds not all it saved, rather than dropping it: a
release's config (Laya's) its layout's window, budgets and option layout alone, and its weights a
ModernBERT encoder's."""


def exported(run: Path, destination: Path) -> bool:
    """Whether `dew export` writes the saved `run` to `destination`, False where it refuses the run as a
    format that holds not all it saved (`UNHELD`); any other failure is the export's own, raised."""
    try:
        return main(["export", str(run), str(destination)]) == 0
    except ValueError as error:
        if not any(said in str(error) for said in UNHELD):
            raise
        return False


def unread(task, before: dict[str, object], option) -> str | None:
    """Why the loaded `task` does not consume `option`, None where it does: it holds it nowhere it held
    another before (`held`), or a request of it answers as the same task does holding there what it held
    before (`answers`): the same weights, the option alone apart, so what differs is the consumer's
    work with it, not its record."""
    places = [path for path, inner in held(task).items()
              if same(inner, option) and path in before and not same(before[path], inner)]
    if not places:
        return "the task loaded holds none set"
    swapped = task
    for path in places:
        swapped = replaced(swapped, path, before[path])
    if answered_alike(answers(task), answers(swapped)):
        return f"a request of it answers as with {[before[path] for path in places]} in its place"
    return None


@pytest.mark.parametrize("release", ["dream", "diffucoder"])
def test_a_masked_run_over_a_release_saves_and_reads_the_walk_the_release_declares(release, tmp_path):
    """A masked-diffusion run over a Dream release (dream-tiny as the release ships it, its generation
    config and its own sampling code, `test_dream_walk.snapshot`), setting nothing of its task, saves the
    walk the release's own task declares (`Pretrained.text_generation`): its process, the grid's endpoint
    among it (ledger 2563), its solver, its steps and its blocks; and the walk its loaded task takes
    reads that endpoint: the same task with no endpoint answers apart (`answers`), a loaded source's
    default reaching its consumer."""
    from test_dream_walk import snapshot
    from token_support import write_token_corpus

    from dew.inference.tasks import MaskedSettings
    from dew.interop import Pretrained

    source = snapshot(release, tmp_path)
    declared = Pretrained.load(source, dtype="float32", attention_impl="reference").text_generation()
    corpus = write_token_corpus(tmp_path / "tokens", np.random.RandomState(0).randint(4, 100, 400),
                                tokenizer=str(source), vocab_size=128, eos_id=1)
    data = TokenFiles(corpus.train).pipe(presets.lm, None, seq_len=11, batch=8, pack=None).loading(
        Loading(workers=0))
    model = ModelConfig("causal_transformer", {"dtype": "float32", "attention_impl": "reference"})
    run = LMRunConfig(data=data, pretrained=str(source), objective=ObjectiveConfig("masked_diffusion"),
                      tokenizer=str(source), max_new_tokens=0, model=model)
    _, task = saved_with(run, tmp_path, "over", MaskedSettings, {})
    options = {name: getattr(declared, name) for name in ("process", "solver", "steps", "block")}
    assert {name: same(getattr(task, name), option) for name, option in options.items()} == dict.fromkeys(
        options, True)
    assert task.process.endpoint is not None
    unended = dataclasses.replace(task, process=dataclasses.replace(task.process, endpoint=None))
    assert not answered_alike(answers(task), answers(unended))


def test_a_decision_run_over_a_release_saves_and_reads_the_task_the_release_declares(tmp_path):
    """A decision run over a release (tiny Laya), setting nothing of its task, saves the task the
    release's own declares (`Decide.from_pretrained`): its layout, its special tokens and its confidence,
    its budget the run's data's (`DecideSettings.questions`). What reads each, the gate above witnesses
    (`witnessed_on`)."""
    from dew.decision.task import Decide, DecideSettings

    declared = Decide.from_pretrained(TINY_LAYA)
    _, task = saved_with(decision(tmp_path, drawn=False), tmp_path, "over", DecideSettings, {})
    options = {name: getattr(declared, name) for name in ("layout", "specials", "confidence")}
    assert {name: same(getattr(task, name), option) for name, option in options.items()} == dict.fromkeys(
        options, True)


def the_attention_forms_a_run_names_reach_its_step(tmp_path):
    """A GPU flow's cell (tests/test_gpu_training.py): the attention forms a run's numerics name on the
    command line (`--set trainer.precision`, `Numerics.forms`) reach its step through the default route: a
    step in the accurate forms (IEEE fp32 products) and one in the fast (a GPU's TF32), the model's
    compute float32 in both, score its batch apart, so the route reads the run's forms, not its record
    alone (ledger 2350). A CPU multiplies fp32 one way in both, so only a GPU tells them apart."""
    from process_support import logged_losses

    from dew.nn.precision import Numerics

    run = lm(tmp_path, drawn=False)
    losses = {forms: logged_losses(trained_by_cli(run, tmp_path, forms, {
        "trainer.precision": flag(Numerics(forms=forms), Numerics)})) for forms in ("fast", "accurate")}
    assert losses["fast"] and losses["fast"] != losses["accurate"]
