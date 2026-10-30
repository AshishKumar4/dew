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
from dew.diffusion.presets import Flow
from dew.inputs import TextCondition
from dew.lora import LoRA
from dew.registry import ModelConfig, objectives
from dew.sampling import Euler
from dew.training import Layout, MeshSpec
from dew.training.evaluation import Summary

pytestmark = pytest.mark.mesh

TINY_LAYA = Path(__file__).parent / "fixtures" / "laya" / "tiny"
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
                "trainer.checkpoint_every": "1", "trainer.keep": "4", "trainer.dynamic_scale": "True",
                "trainer.multi_host": "False", "trainer.compilation_cache_dir": "None",
                "optim.learning_rate": str(RATE), "val_metrics": "[]",
                "val_summaries": json.dumps([{"name": "board", "weights": {"val/loss": 1},
                                              "better": "lower"}])}
    assert main(["train", str(record / "run.json"),
                 *(flag for path, value in settings.items() for flag in ("--set", f"{path}={value}"))]) == 0
    run = runs / "set"
    logged = scalars(run)
    assert {step for step, row in logged.items() if "train/loss" in row} == {1, 2, 3, 4}
    assert sorted(step for step, row in logged.items() if "train/grad_norm" in row) == [2, 4]
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


def registered_alternatives(annotation: type) -> list[object]:
    """The members of the registry `dew.registry` declares for `annotation` (`solvers` for `Solver`), each
    built with its defaults where it has them."""
    for name, declared in registry.__annotations__.items():
        built = str(declared).partition(", ")[2].partition("[")[0].rstrip("]")
        if declared.startswith("Aliases[") and built == annotation.__name__:
            table = getattr(registry, name)
            members = []
            for alias in table:
                try:
                    members.append(table[alias]())
                except TypeError:  # A member with required fields has no default to witness.
                    continue
            return members
    return []


def alternatives(annotation) -> list[object]:
    """The values `annotation` declares as alternatives, in its order: a literal's values, a flag's two, an
    optional's None and its member's, a registered type's members (`registered_alternatives`), and a
    dataclass's instances over its own fields' alternatives. A plain number or text declares none, nor a
    record (a union holding a mapping or a list), whose values are its writer's."""
    if isinstance(annotation, typing.TypeAliasType):
        return alternatives(annotation.__value__)
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        return alternatives(typing.get_args(annotation)[0])
    if origin is typing.Literal:
        return list(typing.get_args(annotation))
    if annotation is bool:
        return [False, True]
    if origin in (types.UnionType, typing.Union):
        members = typing.get_args(annotation)
        if any(typing.get_origin(member) in (dict, list, typing.Mapping) or member in (dict, list)
               for member in members):
            return []
        found = []
        for member in members:
            values = [None] if member is type(None) else [1] if member is int else alternatives(member)
            found += [value for value in values if value not in found]
        return found
    if dataclasses.is_dataclass(annotation) and isinstance(annotation, type):
        hints = typing.get_type_hints(annotation)
        declared = {field.name: alternatives(hints[field.name]) for field in dataclasses.fields(annotation)
                    if field.init}
        declared = {name: values for name, values in declared.items() if len(values) > 1}
        try:
            built = [annotation(**{name: values[index] for name, values in declared.items()})
                     for index in (0, 1)]
        except (TypeError, ValueError):
            return []
        return built if declared and built[0] != built[1] else []
    if isinstance(annotation, type) and annotation.__module__.startswith("dew."):
        return registered_alternatives(annotation)
    return []


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


def reach(run: RunConfig, objective: str, name: str, annotation) -> str | None:
    """The path `dew train` sets a saved task's option `name` of `annotation` through: a field of the run,
    or of a section it holds, declared of that type (a number or a text's of that name), else its
    objective's argument of that name; None where it has neither."""
    # A number's or a text's field is the option's by its name; any other type's, by its type.
    named = bare(annotation) in (int, float, str, bool)

    def paths(config, prefix: str):
        hints = typing.get_type_hints(type(config), include_extras=True)
        for field in dataclasses.fields(config):
            value = getattr(config, field.name)
            if field.name == name if named else bare(hints[field.name]) == bare(annotation):
                yield prefix + field.name
            elif dataclasses.is_dataclass(value) and type(value).__module__.startswith("dew.config"):
                yield from paths(value, f"{prefix}{field.name}.")
    found = next(paths(run, ""), None)
    if found is None and name in inspect.signature(objectives[objective]).parameters:
        return f"objective.{name}"
    return found


def flag(value, annotation) -> str:
    """`value` as `--set` writes it for a field of `annotation`: a record's JSON, or its text."""
    if value is None or isinstance(value, bool | int | str):
        return str(value)
    return json.dumps(registry.to_record(value, bare(annotation)))


TASK_RUNS = {"TextSettings": (lm, "lm"), "MaskedSettings": (masked, "masked_diffusion"),
             "ImageSettings": (diffusion, "diffusion"), "DecideSettings": (decision, "decision")}
"""A run here of an objective that records each saved task's settings, and the objective's name."""


def witnessed(settings: type) -> dict[str, list[object]]:
    """Each field of `settings` that declares alternatives (`alternatives`), with them."""
    hints = typing.get_type_hints(settings)
    declared = {field.name: alternatives(hints[field.name]) for field in dataclasses.fields(settings)}
    return {name: values for name, values in declared.items() if len(values) > 1}


SETTINGS = sorted({cls.settings for cls in (objectives[name] for name in objectives)
                   if cls.settings is not None and witnessed(cls.settings)},
                  key=lambda settings: settings.__name__)


def saved_with(run: RunConfig, root: Path, name: str, settings: type, assigned: dict[str, str]):
    """The `settings` the task of `run` saves after one step of `dew train` with each `path=value` of
    `assigned` set, its task built from them (`dew.pipeline`)."""
    import dew
    from dew.inference.tasks import SavedRun, saved_settings

    record, runs = root / f"{name}-record", root / "runs"
    run.save(str(record))
    given = {"trainer.name": name, "trainer.checkpoint_dir": str(runs), "trainer.steps": "1",
             "trainer.eval_every": "None", "trainer.checkpoint_every": "None",
             "trainer.multi_host": "False", "trainer.compilation_cache_dir": "None", **assigned}
    assert main(["train", str(record / "run.json"),
                 *(part for path, value in given.items() for part in ("--set", f"{path}={value}"))]) == 0
    dew.pipeline(str(runs / name))
    return saved_settings(SavedRun.pinned(str(runs / name)).record, settings)


def witness_runs(settings: type, options: dict[str, list[object]], recorded) -> dict[str, object]:
    """What the second of two runs sets each option to, the first having set each to its first
    alternative and a registered type's (`registered_alternatives`) to none, saving `recorded`: its second
    alternative, and a registered type's first member other than the one the first run saved that the
    settings accept beside the rest it saved."""
    hints, second = typing.get_type_hints(settings), {}
    for name, values in options.items():
        if not registered_alternatives(bare(hints[name])):
            second[name] = values[1]
            continue
        for value in values:
            if value == getattr(recorded, name):
                continue
            try:
                dataclasses.replace(recorded, **{name: value})
            except (TypeError, ValueError):
                continue
            second[name] = value
            break
    return second


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("settings", SETTINGS, ids=lambda settings: settings.__name__)
def test_each_option_a_saved_task_is_built_with_reaches_it_from_the_command_line(settings, tmp_path):
    """Every option a saved task is built with that its settings' types declare alternatives for
    (`alternatives`: a literal's values, an optional's, a flag's, a registered type's members, a
    dataclass's instances over its own) reaches the task `dew train` saves from its path (`reach`): two
    runs set every one at its first and at its second alternative (`witness_runs`), and the task each
    saves holds each as set, or a run refuses one, judged by what reads it. One with no path is filled by
    the run itself, not left at its default: a task option nothing sets is row 1061's blockwise walk.
    Every settings type an objective declares alternatives in has a run here (`TASK_RUNS`)."""
    assert settings.__name__ in TASK_RUNS, f"no run here trains an objective whose task reads {settings}"
    builder, objective = TASK_RUNS[settings.__name__]
    hints = typing.get_type_hints(settings)
    run, options = builder(tmp_path, drawn=False), witnessed(settings)
    paths = {name: reach(run, objective, name, hints[name]) for name in options}
    defaults = {field.name: field.default for field in dataclasses.fields(settings)}
    first = {name: values[0] for name, values in options.items()
             if paths[name] is not None and not registered_alternatives(bare(hints[name]))}
    saved = None
    for index in (0, 1):
        assigned = first if saved is None else {name: value for name, value in witness_runs(
            settings, options, saved).items() if paths[name] is not None}
        flags = {paths[name]: flag(value, hints[name]) for name, value in assigned.items()}
        try:
            saved = saved_with(run, tmp_path, f"set{index}", settings, flags)
        except ValueError:
            # A refusal names what refused it; each option alone shows which reached a reader that did.
            for name, value in assigned.items():
                try:
                    alone = saved_with(run, tmp_path, f"{name}{index}", settings,
                                       {paths[name]: flag(value, hints[name])})
                except ValueError:
                    continue
                assert getattr(alone, name) == value, (name, value)
                saved = saved or alone
            assert saved is not None, f"no option of {settings.__name__} saves at its first alternative"
            continue
        assert {name: getattr(saved, name) for name in assigned} == assigned
        for name in options:
            if paths[name] is None and defaults[name] is not dataclasses.MISSING:
                assert getattr(saved, name) != defaults[name], f"{settings.__name__}.{name} has no path"
