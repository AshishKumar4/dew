"""How a run saved by an earlier Dew reads now: each record shape an earlier release wrote, read as the run
trained (`earlier_run`, the one place earlier record shapes are read), and the defaults an argument a class
gained since keeps for runs that predate it (`_earlier_default`)."""

import dataclasses
import sys
from collections.abc import Mapping

from dew import _records, registry
from dew._records import JSON, earlier_fields
from dew.data import records
from dew.nn.precision import Numerics, numerics_of
from dew.registry import from_record, models, objectives


def _chains() -> tuple[str, ...]:
    """The run's fields that hold chains today (`dew.config.CHAINS`), which an earlier record's chains are
    read under: read where `earlier_run` runs, as `dew.config` imports this module."""
    from dew.config import CHAINS

    return CHAINS


def earlier_run(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved run's record (its run.json, `{"class": ..., "fields": ...}`) with its fields in the shapes
    Dew writes now, every earlier shape of a saved run read here and nowhere else (`RunConfig.load`,
    `check_resume`). A record written before its trainer held a precision policy (`dew.nn.precision.
    Numerics`, ledger 2500) reads as its main trained it (train 328's review): its trainer's
    `attention_form` as the policy's `forms`; a loaded checkpoint's `pretrained_param_dtype`, and a
    decision's `param_dtype`, as the storage of the source alone (`SourceOptions.param_dtype`), the head
    and adapter it drew fresh kept in float32; its compute as its main's (`earlier_compute`); and its
    `dynamic_scale` as it trained under it; a Muon run's Newton-Schulz precision as a dot algorithm
    (`earlier_newton_schulz`); a profile window's capture options as its capture's (`earlier_profile`); a
    decision run's option text as its rows wrote it (`earlier_option_text`), and the fewest options its
    rows drew "none of the above" on as they drew it (`earlier_none_of_the_above`); a world model's
    tokenizer's and its dynamics model's positions as they learned them (`earlier_tokenizer_positions`,
    `earlier_dynamics_positions`); a trainer layout recorded at `Layout()`'s defaults as unset
    (`earlier_trainer_layout`); its data plan version
    as the one it read by (`earlier_data_plan`); a chain's store under the fields its class renamed since,
    by today's names (`dew.data.records.earlier_chains`); an encoder or conditioner by the import path it
    has now (`dew._records.moved_paths`, row 5996); a decision layout of a class it was before by the
    segment layout it is now, wherever the record holds it (`dew._records.earlier_layout`, row 6140); a
    source's `packed_experts` as `packed_weights` (`earlier_packed_experts`, row 8550); and wherever it holds
    a run or trainer, a trainer's retired `export_model` taken off and a looped decoder's
    `loop_message_additive` true read as unset where it could not be unset (`earlier_retired`, rows 11636,
    11742).
    A configuration written today that names the earlier knobs is refused by name, as any field its class
    does not declare. A record in today's shapes passes as it is. Carried to 1.0, when it is deleted
    (ledger 1552)."""
    unsettable, additive = (record.get(key) is True for key in (LAYOUT_UNSETTABLE, ADDITIVE_UNSETTABLE))
    moved = _records.moved_record({key: value for key, value in record.items() if key not in MARKERS})
    earlier = earlier_none_of_the_above(earlier_profile(earlier_newton_schulz(earlier_trainer(
        moved if unsettable else earlier_trainer_layout(moved)))))
    earlier = earlier_option_text(earlier_dynamics_positions(earlier_tokenizer_positions(earlier)))
    earlier = earlier_retired(earlier_packed_experts(earlier), additive_unset=additive)
    # A decision layout recorded by one of the classes it was before (decision-inputs.md §1, row 6140).
    earlier = {name: _records.earlier_layout(_records.json_value(value, name))
               for name, value in earlier.items()}
    return records.earlier_chains(earlier_data_plan(earlier), _chains())


LAYOUT_UNSETTABLE = "layout_unsettable"
"""The key beside a run record's fields (`RunConfig.record`) that says it states its trainer's layout as
this Dew does, unset as null: a record written while it could not be unset (`Layout()` its default) lacks
it and reads `Layout()` as unset (`earlier_trainer_layout`), where one written since keeps a stated
`Layout()` (row 9664). Beside the fields, where no field of the run's holds it (train 1109's review, F3)."""

ADDITIVE_UNSETTABLE = "additive_unsettable"
"""The key beside a run record's fields (`RunConfig.record`) that says it states `loop_message_additive` as
this Dew does, unset as null: a record written while it could not be unset (true its default) lacks it and
reads its true as unset (`earlier_retired`), where one written since keeps a stated true (row 11742)."""

MARKERS = (LAYOUT_UNSETTABLE, ADDITIVE_UNSETTABLE)
"""Every key a run record holds beside its fields, each saying how it states a field (`RunConfig.record`)."""


def earlier_retired(record: Mapping[str, object], *, additive_unset: bool) -> Mapping[str, object]:
    """A saved run's record with each run and trainer it holds, the root and every one a field holds as its
    class record, in today's shapes (`_retire`): a trainer's `export_model`, the field that wrote each
    persistent step's model beside it (row 11636), taken off, `dew export` the one export since; and a
    looped decoder's `loop_message_additive` true read as unset where the record could not leave it unset
    (no `ADDITIVE_UNSETTABLE`, `additive_unset` false; row 11742), every such record writing true for its
    default, a checkpoint's choice kept, its false a choice. A record written since states its own. Untyped
    data, and a class no module imported yet holds, pass as they are. Carried to 1.0 (ledger 1552)."""
    from dew.config import RunConfig

    # The record read as JSON once, here, a tree of its own the walk below migrates in place (row 11824).
    retired = {key: _records.json_value(value, key) for key, value in record.items()}
    _retire(retired, additive_unset, RunConfig)
    return retired


def _retire(value: _records.JSON, additive_unset: bool, default: type | None = None) -> None:
    """Each run and trainer `value`'s class records name, at any depth, taken to today's shapes in place
    (`_retire_fields`), `default` the class of a record that names none it resolves (the root's, a run):
    data no migration reads stays as it is (a payload naming a class with no fields keeps none)."""
    if isinstance(value, list):
        for held in value:
            _retire(held, additive_unset)
        return
    if not isinstance(value, dict):
        return
    for entry in value.values():
        _retire(entry, additive_unset)
    named = registry.class_record(value)
    kind = (_recorded_class(named[0]) or default) if named is not None else None
    fields = value.get("fields")
    if kind is not None and isinstance(fields, dict):
        _retire_fields(kind, fields, additive_unset)


def _retire_fields(kind: type, fields: dict[str, _records.JSON], additive_unset: bool) -> None:
    """The fields of a record of `kind` taken to today's shapes in place (`earlier_retired`)."""
    from dew.config import DecoderRunConfig, RunConfig
    from dew.config.trainer import TrainerConfig

    if issubclass(kind, TrainerConfig):
        fields.pop("export_model", None)
    # A trainer recorded flat, by its run's class, is read here; one recorded by its own class as a record.
    trainer = fields.get("trainer")
    if issubclass(kind, RunConfig) and isinstance(trainer, dict) and registry.class_record(trainer) is None:
        trainer.pop("export_model", None)
    if (issubclass(kind, DecoderRunConfig) and not additive_unset
            and fields.get("loop_message_additive") is True):
        fields["loop_message_additive"] = None


def _recorded_class(name: str) -> type | None:
    """The class a record names, a run's alias or an import path, where a module imported already holds it;
    None else (a class a record imports is `registry.imported`'s, as it is built)."""
    module, _, qualified = registry.runs.paths.get(name, name).partition(":")
    found = sys.modules.get(module)
    for part in qualified.split(".") if qualified else ():
        found = getattr(found, part, None)
    return found if isinstance(found, type) else None


def earlier_packed_experts(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved run's record whose source states `packed_experts`, the field `SourceOptions.packed_weights`
    was while it packed an MXFP4 source's experts alone (row 8550): the same choice under today's name.
    Carried to 1.0 (ledger 1552)."""
    fields = record.get("fields")
    source = fields.get("source") if isinstance(fields, Mapping) else None
    if not isinstance(fields, Mapping) or not isinstance(source, Mapping) or "packed_experts" not in source:
        return record
    renamed = {("packed_weights" if key == "packed_experts" else key): value for key, value in source.items()}
    return {**record, "fields": {**fields, "source": renamed}}



def earlier_trainer_layout(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved run's record whose trainer layout is `Layout()`'s, which every run recorded before its
    trainer's layout could be unset (`TrainerConfig.layout`) recorded for its default: unset, the layout its
    resume takes its checkpoint's (`dew.training.streaming.laid_start`), as the run it continues chose it
    (row 9534: read as a stated device layout, a streamed run's checkpoint resumed on its devices). Carried
    to 1.0 (ledger 1552)."""
    from dew.distributed.placement import Layout

    fields = record.get("fields")
    trainer = fields.get("trainer") if isinstance(fields, Mapping) else None
    if not isinstance(fields, Mapping) or not isinstance(trainer, Mapping):
        return record
    # A record written while `layout_unsettable` was a trainer field (row 9664, until train 1109's review)
    # states its `Layout()`: the field taken off.
    if "layout_unsettable" in trainer:
        return {**record, "fields": {**fields, "trainer": {key: value for key, value in trainer.items()
                                                           if key != "layout_unsettable"}}}
    if not isinstance(trainer.get("layout"), Mapping) or from_record(Layout, trainer["layout"]) != Layout():
        return record
    return {**record, "fields": {**fields, "trainer": {**trainer, "layout": None}}}


def earlier_data_plan(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved run's record that states no data plan version (`RunConfig.data_plan_version`): one a Dew
    wrote while the field was `data_plan` states it there, None where the run was trained in Python,
    which that Dew compiled at version 9; one written before either, under version 8, which read a
    mixture's corpora shuffled by one seed in one permutation, reads by 8 (row 3840). Carried to 1.0
    (ledger 1552)."""
    named = registry.class_record(record)
    if named is None or named[0] not in registry.runs or "data_plan_version" in named[1]:
        return record
    fields = {name: value for name, value in named[1].items() if name != "data_plan"}
    landed = named[1].get("data_plan", 8)
    return {**record, "fields": {**fields, "data_plan_version": 9 if landed is None else landed}}


def earlier_option_text(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved run's record of a class that states how its rows write a choice's option (`option_text`,
    `dew.decision.DecisionRunConfig`) where it states none: its rows wrote each option's key beside its
    description, as every layout did before the field (`SegmentLayout.option_text`), so it reads
    `named`. Carried to 1.0 (ledger 1552)."""
    named = registry.class_record(record)
    if named is None or named[0] not in registry.runs or "option_text" in named[1]:
        return record
    run_class = registry.runs[named[0]]
    if not (isinstance(run_class, type) and dataclasses.is_dataclass(run_class)) or "option_text" not in {
            field.name for field in dataclasses.fields(run_class)}:
        return record
    return {**record, "fields": {**named[1], "option_text": "named"}}


def _earlier_default(record: Mapping[str, object], field: str, kinds: registry.Aliases, kind: str,
                     argument: str, earlier: JSON) -> Mapping[str, object]:
    """A saved run's record whose `field` (its objective, its model), named in `kinds` as whatever alias or
    import path it saved, the class at import path `kind` (`registry.import_path`), states `argument`
    neither in its fields nor in the defaults it recorded: saved before the class took it, it trained as
    `earlier` computes, which its defaults read now; a run saved since records its own. The class is named
    by its path, so reading a record imports only the class it names."""
    named = registry.class_record(record)
    if named is None:
        return record
    _, fields = named
    held = fields.get(field)
    if not isinstance(held, Mapping) or not isinstance(held.get("name"), str):
        return record
    stated, defaulted = held.get("fields", {}), held.get("defaults", {})
    if (registry.import_path(kinds[held["name"]]) != kind or not isinstance(stated, Mapping)
            or not isinstance(defaulted, Mapping) or argument in {*stated, *defaulted}):
        return record
    return {**record, "fields": {**fields, field: {**held, "defaults": {**defaulted, argument: earlier}}}}


def earlier_none_of_the_above(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved decision run's record whose objective states the fewest options its rows draw "none of the
    above" on (`none_of_the_above_options`) neither in its fields nor in the defaults it recorded: saved
    before the argument, its rows drew it on every choice, a yes or a no among them, so it reads 0, as
    they drew (`_earlier_default`; row 2938: the default became 3). Carried to 1.0 (ledger 1552)."""
    return _earlier_default(record, "objective", objectives, "dew.decision.objective:DecisionObjective",
                            "none_of_the_above_options", 0)


def earlier_tokenizer_positions(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved world-model run's record whose tokenizer (`CausalTokenizer`) states its `positions`
    nowhere: saved before rotary positions, its tokens knew their places by learned vectors alone, so it
    reads "learned", as it trained (`_earlier_default`; row 4669: the default became "rotary"). Carried to
    1.0 (ledger 1552)."""
    return _earlier_default(record, "model", models, "dew.objectives.world_model:CausalTokenizer",
                            "positions", "learned")


def earlier_dynamics_positions(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved world-model run's record whose dynamics model (`Dynamics`) states its `positions` nowhere:
    saved before rotary positions, its tokens knew their frame and place by learned vectors alone, so it
    reads "learned", as it trained (`_earlier_default`; row 4802: the default became "rotary"). Carried
    to 1.0 (ledger 1552)."""
    return _earlier_default(record, "model", models, "dew.objectives.world_model:Dynamics", "positions",
                            "learned")


def earlier_trainer(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved run's record with its decision segments and its trainer's numerics in today's shapes
    (`earlier_run`), its optimizer's as it was."""
    fields = record.get("fields")
    segments = fields.get("segments") if isinstance(fields, Mapping) else None
    if isinstance(fields, Mapping) and isinstance(segments, list):
        # A decision run's row of segments, each component in today's fields (`earlier_fields`): the
        # segments alone, where a layout's components are, never a field of data a row holds.
        migrated = earlier_fields(_records.json_value(segments, "segments"))
        assert isinstance(migrated, list)
        fields = {**fields, "segments": migrated}
        record = {**record, "fields": fields}
    if not isinstance(fields, Mapping) or not isinstance(fields.get("trainer"), Mapping):
        return record
    trainer, fields = dict(fields["trainer"]), dict(fields)
    stated = {name: fields.pop(name) for name in ("pretrained_param_dtype", "param_dtype") if name in fields}
    if "precision" in trainer:
        if "attention_form" in trainer or stated:
            raise ValueError(f"a saved run records {sorted({'attention_form'} & set(trainer) | set(stated))} "
                             "beside its trainer's precision, which holds them now")
        return record
    if "attention_form" not in trainer and "dynamic_scale" not in trainer and not stated:
        return record
    policy: dict[str, object] = {"dtypes": {"compute": earlier_compute(record)}}
    if "attention_form" in trainer:
        policy["forms"] = trainer.pop("attention_form")
    stored = next(iter(stated.values()), None)
    # The storage named a loaded checkpoint's alone, which a run that loads none never read.
    if stored is not None and fields.get("pretrained") is not None:
        source = fields.get("source")
        fields["source"] = {**(source if isinstance(source, Mapping) else {}), "param_dtype": stored}
    return {**record, "fields": {**fields, "trainer": {**trainer, "precision": policy}}}


EARLIER_CAPTURE = ("host_tracer_level", "python_tracer_level", "activity_cap")
"""The capture options a profile window held as its own fields until it held its capture's
(`dew.training.ProfileWindow.capture`, row 2932)."""


def earlier_profile(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved run's record whose trainer's profile window states its capture's options as its own fields
    (`EARLIER_CAPTURE`), with them its capture's, as it captured with them; and one saved before a run's
    own window was its `run_profile` (row 4255): its window naming no directory (PF01 to train 610's) as
    its `run_profile`, its steps, warmup and capture kept, and no window at all as none of its own, as it
    traced none. A window that states its capture beside them is as it was, its fields refused by name.
    Carried to 1.0 (ledger 1552)."""
    fields = record.get("fields")
    trainer = fields.get("trainer") if isinstance(fields, Mapping) else None
    inner = trainer.get("fields", trainer) if isinstance(trainer, Mapping) else None
    window = inner.get("profile") if isinstance(inner, Mapping) else None
    stated = window.get("fields", window) if isinstance(window, Mapping) else None
    if not isinstance(fields, Mapping) or not isinstance(trainer, Mapping) or not isinstance(inner, Mapping):
        return record
    own_window = isinstance(stated, Mapping) and stated.get("directory", "") is None
    if "run_profile" not in inner and "profile" in inner and (window is None or own_window):
        # Saved before a run's own window was its `run_profile`: no window traced none, and a window
        # naming no directory was the run's own, its steps, warmup and capture its own's (its reports
        # none, a run's own window's trace and manifest alone).
        own = ({name: stated[name] for name in ("steps", "warmup", "capture") if name in stated}
               if isinstance(stated, Mapping) else None)
        inner = {**inner, "profile": None, "run_profile": own}
        trainer = {**trainer, "fields": inner} if "fields" in trainer else inner
        return {**record, "fields": {**fields, "trainer": trainer}}
    if (not isinstance(window, Mapping) or not isinstance(stated, Mapping) or "capture" in stated
            or not set(EARLIER_CAPTURE) & set(stated)):
        return record
    capture = {name: stated[name] for name in EARLIER_CAPTURE if name in stated}
    stated = {**{name: value for name, value in stated.items() if name not in EARLIER_CAPTURE},
              "capture": capture}
    window = {**window, "fields": stated} if "fields" in window else stated
    inner = {**inner, "profile": window}
    trainer = {**trainer, "fields": inner} if "fields" in trainer else inner
    return {**record, "fields": {**fields, "trainer": trainer}}


EARLIER_NS_PRECISIONS = {"bfloat16": None, "tensorfloat32": "TF32_TF32_F32", "float32": "F32_F32_F32"}
"""A Muon or M3 run's `ns_precision` as it was recorded while it named a matmul precision level (ledger
2637), by the dot algorithm it names now: bfloat16 the run's numerics' own (None), the others their
algorithms."""


def earlier_newton_schulz(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved run's record whose optimizer's options name Muon's Newton-Schulz precision by an earlier
    level's name (`EARLIER_NS_PRECISIONS`), with it named as a dot algorithm, or dropped where it was the
    numerics' own: bfloat16 is the fast forms' own, and stated as `BF16_BF16_F32` where the run's forms
    are accurate (`saved_forms`), whose own is float32 (row 2753). Carried to 1.0 (ledger 1552)."""
    fields = record.get("fields")
    optim = fields.get("optim") if isinstance(fields, Mapping) else None
    inner = optim.get("fields", optim) if isinstance(optim, Mapping) else None
    opts = inner.get("optimizer_opts") if isinstance(inner, Mapping) else None
    named = opts.get("ns_precision") if isinstance(opts, Mapping) else None
    if not isinstance(fields, Mapping) or not isinstance(optim, Mapping) or not isinstance(inner, Mapping) \
            or not isinstance(opts, Mapping) or named not in EARLIER_NS_PRECISIONS:
        return record
    trainer = fields.get("trainer")
    accurate = isinstance(trainer, Mapping) and saved_forms(trainer) == "accurate"
    algorithm = "BF16_BF16_F32" if named == "bfloat16" and accurate else EARLIER_NS_PRECISIONS[named]
    opts = {name: value for name, value in opts.items() if name != "ns_precision"}
    opts = opts if algorithm is None else {**opts, "ns_precision": algorithm}
    inner = {**inner, "optimizer_opts": opts}
    optim = {**optim, "fields": inner} if "fields" in optim else inner
    return {**record, "fields": {**fields, "optim": optim}}


def saved_forms(trainer: Mapping[str, object]) -> str:
    """The forms a saved trainer's numerics name, in today's shapes (`earlier_trainer`): its precision's,
    a preset's name or a `Numerics` record; the fast ones where it names none."""
    precision = trainer.get("precision")
    if isinstance(precision, str):
        return numerics_of(precision).forms
    if isinstance(precision, Mapping):
        return from_record(Numerics, precision).forms
    return "fast"


def earlier_compute(record: Mapping[str, object]) -> str:
    """The dtype a run saved before its numerics computed in: its model's stated dtype, where it states
    one; else bfloat16 where it loaded a language model or a diffusion pipeline, as those mains loaded
    them; else float32, Flax's own for a model drawn fresh and a decision run's backbone."""
    from dew.config.diffusion import DiffusionRunConfig
    from dew.config.lm import LMRunConfig

    fields = record.get("fields")
    assert isinstance(fields, Mapping)
    model = fields.get("model")
    stated = model.get("fields", {}).get("dtype") if isinstance(model, Mapping) else None
    if isinstance(stated, str):
        return stated
    named = record.get("class")
    kind = registry.runs[named] if isinstance(named, str) else None
    loads = fields.get("pretrained") is not None and isinstance(kind, type) and issubclass(
        kind, LMRunConfig | DiffusionRunConfig)
    return "bfloat16" if loads else "float32"
