"""How a run saved by an earlier Dew reads now: each record shape an earlier release wrote, read as the run
trained (`earlier_run`, the one place earlier record shapes are read), and the defaults an argument a class
gained since keeps for runs that predate it (`_earlier_default`)."""

import dataclasses
from collections.abc import Mapping

from dew import _records, registry
from dew._records import JSON, earlier_fields
from dew.checkpoints import Checkpoints
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
    `earlier_dynamics_positions`); its data plan version
    as the one it read by (`earlier_data_plan`); a chain's store under the fields its class renamed since,
    by today's names (`dew.data.records.earlier_chains`).
    A configuration written today that names the earlier knobs is refused by name, as any field its class
    does not declare. A record in today's shapes passes as it is. Carried to 1.0, when it is deleted
    (ledger 1552)."""
    earlier = earlier_none_of_the_above(earlier_profile(earlier_newton_schulz(earlier_trainer(record))))
    earlier = earlier_option_text(earlier_dynamics_positions(earlier_tokenizer_positions(earlier)))
    return records.earlier_chains(earlier_data_plan(earlier), _chains())


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
    description, as every layout did before the field (`QuestionLayout.option_text`), so it reads
    `named`. Carried to 1.0 (ledger 1552)."""
    named = registry.class_record(record)
    if named is None or named[0] not in registry.runs or "option_text" in named[1]:
        return record
    run_class = registry.runs[named[0]]
    if not (isinstance(run_class, type) and dataclasses.is_dataclass(run_class)) or "option_text" not in {
            field.name for field in dataclasses.fields(run_class)}:
        return record
    return {**record, "fields": {**named[1], "option_text": "named"}}


DECISION_OBJECTIVE = "dew.decision.objective:DecisionObjective"
"""The decision objective's import path, as a saved run's record names its objective
(`ObjectiveConfig.name`)."""

CAUSAL_TOKENIZER = "dew.objectives.world_model:CausalTokenizer"
"""Dreamer 4's tokenizer's import path, as a saved run's record names its model (`ModelConfig.name`)."""

DYNAMICS = "dew.objectives.world_model:Dynamics"
"""Dreamer 4's dynamics model's import path, as a saved run's record names its model (`ModelConfig.name`)."""


def _earlier_default(record: Mapping[str, object], field: str, kinds: registry.Aliases, path: str,
                     argument: str, earlier: JSON) -> Mapping[str, object]:
    """A saved run's record whose `field` (its objective, its model), a `path` class of `kinds`, states
    `argument` neither in its fields nor in the defaults it recorded: saved before the class took it, it
    trained as `earlier` computes, which its defaults read now; a run saved since records its own."""
    named = registry.class_record(record)
    if named is None:
        return record
    _, fields = named
    held = fields.get(field)
    if not isinstance(held, Mapping) or not isinstance(held.get("name"), str):
        return record
    stated, defaulted = held.get("fields", {}), held.get("defaults", {})
    if (registry.import_path(kinds[held["name"]]) != path or not isinstance(stated, Mapping)
            or not isinstance(defaulted, Mapping) or argument in {*stated, *defaulted}):
        return record
    return {**record, "fields": {**fields, field: {**held, "defaults": {**defaulted, argument: earlier}}}}


def earlier_none_of_the_above(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved decision run's record whose objective states the fewest options its rows draw "none of the
    above" on (`none_of_the_above_options`) neither in its fields nor in the defaults it recorded: saved
    before the argument, its rows drew it on every choice, a yes or a no among them, so it reads 0, as
    they drew (`_earlier_default`; row 2938: the default became 3). Carried to 1.0 (ledger 1552)."""
    return _earlier_default(record, "objective", objectives, DECISION_OBJECTIVE, "none_of_the_above_options",
                            0)


def earlier_tokenizer_positions(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved world-model run's record whose tokenizer (`CausalTokenizer`) states its `positions`
    nowhere: saved before rotary positions, its tokens knew their places by learned vectors alone, so it
    reads "learned", as it trained (`_earlier_default`; row 4669: the default became "rotary"). Carried to
    1.0 (ledger 1552)."""
    return _earlier_default(record, "model", models, CAUSAL_TOKENIZER, "positions", "learned")


def earlier_dynamics_positions(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved world-model run's record whose dynamics model (`Dynamics`) states its `positions` nowhere:
    saved before rotary positions, its tokens knew their frame and place by learned vectors alone, so it
    reads "learned", as it trained (`_earlier_default`; row 4802: the default became "rotary"). Carried
    to 1.0 (ledger 1552)."""
    return _earlier_default(record, "model", models, DYNAMICS, "positions", "learned")


def earlier_option_writing(record: Mapping[str, object], checkpoints: Checkpoints) -> Mapping[str, object]:
    """A saved decision run's record (`earlier_run`'s) whose option text is None, which a Dew before row
    3867 recorded where the run left it to its layout: the writing the layout its last checkpoint in
    `checkpoints` (its tiers, as the run keeps them) records wrote (`dew.decision.layout.layout_writing`),
    so a resume writes its rows as it did, whatever a later Dew's layout leaves; as it is where no
    checkpoint records one (a run that trained no step, or a layout no record described), whose writing
    its build resolves again (`RunConfig.check_resume`). Carried to 1.0 (ledger 1552)."""
    named = registry.class_record(record)
    if named is None or "option_text" not in named[1] or named[1]["option_text"] is not None:
        return record
    from dew.decision.layout import Layout, layout_writing
    from dew.inference.tasks import saved_artifact

    step = checkpoints.latest
    artifact = None if step is None else saved_artifact(checkpoints, step)
    laid = None if artifact is None else artifact.get("layout")
    if not isinstance(laid, Mapping):
        return record
    writing = layout_writing(registry.from_record(Layout, laid))
    return {**record, "fields": {**named[1], "option_text": writing}}


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
