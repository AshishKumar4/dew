"""How a run saved by an earlier Dew reads now: each record shape an earlier release wrote, read as the run
trained (`earlier_run`, the one place earlier record shapes are read), and the defaults an argument a class
gained since keeps for runs that predate it (`_earlier_default`)."""

import dataclasses
import sys
from collections.abc import Mapping, Sequence

from dew import _records, registry
from dew._records import JSON, earlier_fields
from dew.data import records
from dew.nn.precision import Numerics, numerics_of
from dew.registry import from_record, models, objectives

type _Fields = dict[str, _records.JSON]


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
    11742); its model's attention kernel as the run's (`earlier_attention_kernel`, row 11938); a pin of the
    grouped products' kernel written before SYRK took a choice of its own as SYRK's too
    (`earlier_grouped_pins`, row 12457); and every
    field a class lost since, its model's, its source options' and any class record's, left out
    (`earlier_removed`, `dew._records.REMOVED_FIELDS`).
    A configuration written today that names the earlier knobs is refused by name, as any field its class
    does not declare. A record in today's shapes passes as it is. Carried to 1.0, when it is deleted
    (ledger 1552)."""
    unsettable, additive, split = (record.get(key) is True
                                   for key in (LAYOUT_UNSETTABLE, ADDITIVE_UNSETTABLE, KERNELS_SPLIT))
    moved = earlier_removed(earlier_attention_kernel(earlier_layer_scan(_records.moved_record(
        {key: value for key, value in record.items() if key not in MARKERS}))))
    moved = moved if split else earlier_grouped_pins(moved)
    earlier = earlier_none_of_the_above(earlier_profile(earlier_trainer(
        moved if unsettable else earlier_trainer_layout(moved))))
    earlier = earlier_option_text(earlier_dynamics_positions(earlier_tokenizer_positions(earlier)))
    earlier = earlier_newton_schulz(earlier_retired(earlier_packed_experts(earlier), additive_unset=additive))
    # Its optimizer's and trainer's settings by the names a run's surface gives them now (`earlier_surface`).
    earlier = earlier_surface(earlier)
    # A decision layout recorded by one of the classes it was before (decision-inputs.md §1, row 6140).
    earlier = {name: _records.earlier_layout(_records.json_value(value, name))
               for name, value in earlier.items()}
    return records.earlier_chains(earlier_data_plan(earlier), _chains())


def earlier_attention_kernel(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved run's record whose model named its attention kernel (`attention_impl`, a model's field
    until row 11938) with the kernel the run's own (`chosen_kernel`'s `attention`), where it named one but
    'auto' and the run named none: the model's own, else a decoder's it wraps (`saved_decoders`,
    `DiffusionGemma.text`, `MultimodalTransformer.language_model`, the API02 review: the root alone missed
    a wrapped decoder's); the field itself is left out after (`earlier_removed`). Carried to 1.0 (ledger
    1552)."""
    from dew.inference.tasks import earlier_attention_pin

    fields = record.get("fields")
    model = fields.get("model") if isinstance(fields, Mapping) else None
    kernel = earlier_attention_pin(model) if isinstance(model, Mapping) else None
    return record if kernel is None else chosen_kernel(record, "attention", kernel)


def earlier_layer_scan(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved run's record whose model or source options scanned its stack (`scan_layers=True`, a field
    of each until row 4074) with the scan the run's own kernel choice (`chosen_kernel`'s `layer_scan`), so
    a resumed run, a host-streamed one among them, scans as it trained (the API02 review: a saved
    streamed rung resumed a layer at a time and its host placement raised); the fields themselves are left
    out after (`earlier_removed`). Carried to 1.0 (ledger 1552)."""
    fields = record.get("fields")
    if not isinstance(fields, Mapping):
        return record
    from dew.inference.tasks import saved_decoders

    scanned = any(decoder.get("scan_layers") is True for part in (fields.get("model"), fields.get("source"))
                  if isinstance(part, Mapping) for decoder in saved_decoders(part))
    return chosen_kernel(record, "layer_scan", "scanned") if scanned else record


def earlier_grouped_pins(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved run's record written before Muon's symmetric products took a kernel choice of their own
    (`syrk`, rows 12457, 12485; no `KERNELS_SPLIT` beside its fields), whose `grouped_matmul` pin of
    `mosaic` or `xla` chose SYRK's kernel too, refusals and all: that pin SYRK's as well, where the run named
    none of its own, so a resumed run takes its Muon steps as it trained. Fused linear cross-entropy's is
    not carried: its pin read the grouped products' only where its loss's dtype held the kernel, falling to
    the chunked program elsewhere, a handoff no pin of `linear_ce` states (row 12457's defect, the coupling
    itself), and both its arms compute the loss alike (check 1, tests/linear_ce_matrix.py), so the resumed
    run's loss is its chunked program's. A pin of another of the grouped products' kernels named none of
    SYRK's. Carried to 1.0 (ledger 1552)."""
    fields = record.get("fields")
    if not isinstance(fields, Mapping):
        return record
    diagnostics = fields.get("diagnostics")
    named = _records.class_record(diagnostics) if isinstance(diagnostics, Mapping) else None
    pins = named[1] if named is not None else diagnostics if isinstance(diagnostics, Mapping) else fields
    kernels = pins.get("kernels")
    pinned = kernels.get("grouped_matmul") if isinstance(kernels, Mapping) else None
    if pinned not in ("mosaic", "xla"):
        return record
    return chosen_kernel(record, "syrk", pinned)


def chosen_kernel(record: Mapping[str, object], choice: str, kernel: str) -> Mapping[str, object]:
    """A saved run's record whose kernel choices name `kernel` for `choice` (`Diagnostics.kernels`, or the
    run's `kernels` a record of before its diagnostics states, which `earlier_retired` moves there), where
    the run named none of its own for it, which stands."""
    fields = record.get("fields")
    if not isinstance(fields, Mapping):
        return record
    diagnostics = fields.get("diagnostics")
    named = _records.class_record(diagnostics) if isinstance(diagnostics, Mapping) else None
    pins = named[1] if named is not None else diagnostics if isinstance(diagnostics, Mapping) else fields
    kernels = pins.get("kernels")
    chosen = {choice: kernel, **(kernels if isinstance(kernels, Mapping) else {})}
    if not isinstance(diagnostics, Mapping):
        return {**record, "fields": {**fields, "kernels": chosen}}
    stated = {**pins, "kernels": chosen}
    return {**record, "fields": {**fields, "diagnostics": (stated if named is None
                                                            else {"class": named[0], "fields": stated})}}


def earlier_removed(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved run's record without the fields their classes lost since it was saved
    (`dew._records.REMOVED_FIELDS`: how a model executes, never what it computes): its source options'
    (`SourceOptions`, which its run states by its type alone) and every class record's
    (`dew._records.earlier_fields`); its model's are its model record's to drop
    (`dew.inference.tasks.earlier_model`, which every reader of a saved model reads it through). Carried to
    1.0 (ledger 1552)."""
    fields = record.get("fields")
    if not isinstance(fields, Mapping):
        return record
    kept = dict(fields)
    source = kept.get("source")
    if isinstance(source, Mapping) and _records.class_record(source) is None:
        kept["source"] = _without_removed(source, _SOURCE_OPTIONS)
    walked = earlier_fields(_records.json_value({**record, "fields": kept}, "run"))
    assert isinstance(walked, Mapping)
    return walked


_SOURCE_OPTIONS = "dew.interop.source_options:SourceOptions"
"""The class a run's source options are (`RunConfig.source`), stated by the run's type, never by a record."""


def _without_removed(fields: Mapping[str, object], name: str) -> Mapping[str, object]:
    """`fields`, a record of the class `name`, without the fields it lost since (`REMOVED_FIELDS`)."""
    removed = _records.REMOVED_FIELDS.get(name, frozenset())
    return fields if removed.isdisjoint(fields) else {key: value for key, value in fields.items()
                                                       if key not in removed}


LAYOUT_UNSETTABLE = "layout_unsettable"
"""The key beside a run record's fields (`RunConfig.record`) that says it states its trainer's layout as
this Dew does, unset as null: a record written while it could not be unset (`Layout()` its default) lacks
it and reads `Layout()` as unset (`earlier_trainer_layout`), where one written since keeps a stated
`Layout()` (row 9664). Beside the fields, where no field of the run's holds it (train 1109's review, F3)."""

ADDITIVE_UNSETTABLE = "additive_unsettable"
"""The key beside a run record's fields (`RunConfig.record`) that says it states `loop_message_additive` as
this Dew does, unset as null: a record written while it could not be unset (true its default) lacks it and
reads its true as unset (`earlier_retired`), where one written since keeps a stated true (row 11742)."""

KERNELS_SPLIT = "kernels_split"
"""The key beside a run record's fields (`RunConfig.record`) that says its kernel pins name Muon's symmetric
products' own choice (`syrk`): a record written before it had one lacks it, and its `grouped_matmul` pin
reads as SYRK's too (`earlier_grouped_pins`), where one written since keeps them apart (rows 12457,
12485)."""

MARKERS = (LAYOUT_UNSETTABLE, ADDITIVE_UNSETTABLE, KERNELS_SPLIT)
"""Every key a run record holds beside its fields, each saying how it states a field (`RunConfig.record`)."""


def earlier_retired(record: Mapping[str, object], *, additive_unset: bool) -> Mapping[str, object]:
    """A saved run's record with each run and trainer it holds, the root and every one a field holds as its
    class record, in today's shapes (`_retired`): a trainer's `export_model`, the field that wrote each
    persistent step's model beside it (row 11636), taken off, `dew export` the one export since; and a
    looped decoder's `loop_message_additive` true read as unset where the record could not leave it unset
    (no `ADDITIVE_UNSETTABLE`, `additive_unset` false; row 11742), every such record writing true for its
    default, a checkpoint's choice kept, its false a choice. A record written since states its own. Untyped
    data, and a class no module imported yet holds, pass as they are. Carried to 1.0 (ledger 1552)."""

    # The record read as JSON once, here, walked by the one walk of a record's class records, rebuilt only
    # where a record migrates (`_records.class_records`; rows 11824, 11838).
    retired = _records.class_records({key: _records.json_value(value, key) for key, value in record.items()},
                                     lambda name, fields, root: _retired(name, fields, root, additive_unset))
    assert isinstance(retired, Mapping)
    return retired


def _retired(name: str, fields: Mapping[str, _records.JSON], root: bool,
             additive_unset: bool) -> dict[str, _records.JSON] | None:
    """The record of `name` and `fields` in today's shapes (`earlier_retired`), the root's a run where
    it names no class an imported module holds; None where nothing of it changes."""
    from dew.config import DecoderRunConfig, RunConfig
    from dew.config.trainer import TrainerConfig
    from dew.optim import OptimConfig

    kind = _recorded_class(name) or (RunConfig if root else None)
    if kind is None:
        return None
    if issubclass(kind, OptimConfig):
        spec = _optimizer_spec(fields)
        return None if spec is None else {"class": name, "fields": spec}
    optim = fields.get("optim")
    # An optimizer recorded flat, by its run's class, is read here; one by its own class as a record.
    spec = (_optimizer_spec(optim) if issubclass(kind, RunConfig) and isinstance(optim, Mapping)
            and _records.class_record(optim) is None else None)
    trainer = fields.get("trainer")
    # Each retirement asked of the record as it is, the record copied once where one applies (row 11892).
    exported = issubclass(kind, TrainerConfig) and "export_model" in fields
    batched = issubclass(kind, TrainerConfig) and any(key in fields for key in EARLIER_TRAINER_FIELDS)
    flat = (issubclass(kind, RunConfig) and isinstance(trainer, Mapping) and "export_model" in trainer
            and _records.class_record(trainer) is None)
    unset = (issubclass(kind, DecoderRunConfig) and not additive_unset
             and fields.get("loop_message_additive") is True)
    # A trainer's own fields, flat or under its class record's, which the pins are taken out of.
    tagged = _records.class_record(trainer) if isinstance(trainer, Mapping) else None
    trainer_fields = tagged[1] if tagged is not None else trainer
    held_pins = isinstance(trainer_fields, Mapping) and any(key in trainer_fields
                                                            for key in EARLIER_DIAGNOSTICS)
    pinned = issubclass(kind, RunConfig) and ("kernels" in fields or held_pins)
    listed = fields.get("val_metrics")
    split_metrics = isinstance(listed, list) and any(
        isinstance(metric, Mapping) and "split" in metric for metric in listed)
    validated = issubclass(kind, RunConfig) and ("val_every" in fields or split_metrics)
    flat_batched = (issubclass(kind, RunConfig) and isinstance(trainer, Mapping)
                    and _records.class_record(trainer) is None
                    and any(key in trainer for key in EARLIER_TRAINER_FIELDS))
    if not (exported or flat or unset or spec is not None or batched or flat_batched or pinned or validated):
        return None
    migrated = {key: value for key, value in fields.items() if not (exported and key == "export_model")}
    if batched:
        migrated = _update_and_fit(migrated)
    if spec is not None:
        migrated["optim"] = spec
    if flat:
        # A trainer recorded flat, by its run's class, is read here; one by its own class as a record.
        assert isinstance(trainer, Mapping)
        migrated["trainer"] = {key: entry for key, entry in trainer.items() if key != "export_model"}
    if flat_batched:
        held = migrated["trainer"]
        assert isinstance(held, Mapping)
        migrated["trainer"] = _update_and_fit(held)
    if pinned:
        held = migrated.get("trainer")
        held = held if isinstance(held, Mapping) else {}
        named = _records.class_record(held)
        read = _records.json_value(named[1] if named is not None else held, "trainer")
        inner: Mapping[str, _records.JSON] = read if isinstance(read, Mapping) else {}
        diagnostics: dict[str, _records.JSON] = {
            **({"kernels": migrated.pop("kernels")} if "kernels" in migrated else {}),
            **{key: inner[key] for key in EARLIER_DIAGNOSTICS if key in inner}}
        if "trainer" in migrated:
            kept = {key: value for key, value in inner.items() if key not in EARLIER_DIAGNOSTICS}
            # A trainer recorded by its class keeps its class, its fields less the pins.
            migrated["trainer"] = {"class": named[0], "fields": kept} if named is not None else kept
        migrated["diagnostics"] = diagnostics
    if validated:
        migrated = _suites(migrated)
    if unset:
        migrated["loop_message_additive"] = None
    return {"class": name, "fields": migrated}


def earlier_surface(record: Mapping[str, object]) -> Mapping[str, object]:
    """A saved run's record with each optimizer and trainer config it holds, by its class record or flat in
    its run's, under the names a run's surface gives its settings now (API03): an optimizer's
    `learning_rate` as its `lr`, a number, or its `schedule`, a schedule's record; its `spared` as
    `decay_norms_and_biases`, its inverse, a param group's too; its `clip_grads` as `clip`; a trainer's
    `horizon`, `update` and `fit` records as its own fields (`steps`, `accumulation`, `update_tokens`,
    `split`, ...); its `keep` of a count as its latest kept; its `best` of a metric's name as that metric's
    policy, one policy listed as that one, several refused, as a run's config ranks by one; and its split
    of None as the planner's, "auto". A preset's name for its numerics reads as the config reads it
    (`TrainerConfig.precision`). Carried to 1.0 (ledger 1552)."""
    surfaced = _records.class_records({key: _records.json_value(value, key) for key, value in record.items()},
                                      _surfaced)
    assert isinstance(surfaced, Mapping)
    return surfaced


def _surfaced(name: str, fields: Mapping[str, _records.JSON], root: bool) -> dict[str, _records.JSON] | None:
    """The record of `name` and `fields` under today's names (`earlier_surface`), None where none changes."""
    from dew.config import RunConfig
    from dew.config.trainer import TrainerConfig
    from dew.optim import OptimConfig

    kind = _recorded_class(name) or (RunConfig if root else None)
    if kind is None:
        return None
    if issubclass(kind, OptimConfig):
        optim = _optim_surface(fields)
        return None if optim is None else {"class": name, "fields": optim}
    if issubclass(kind, TrainerConfig):
        trainer = _trainer_surface(fields)
        return None if trainer is None else {"class": name, "fields": trainer}
    if not issubclass(kind, RunConfig):
        return None
    # An optimizer or trainer recorded flat, by its run's class, is read here; one by its own class as a
    # record of its own.
    flat = {key: surfaced for key, surface in (("optim", _optim_surface), ("trainer", _trainer_surface))
            if isinstance(held := fields.get(key), Mapping) and _records.class_record(held) is None
            and (surfaced := surface(held)) is not None}
    return {"class": name, "fields": {**fields, **flat}} if flat else None


EARLIER_RATE_FIELDS = ("learning_rate", "spared", "clip_grads")
"""The fields an optimizer config held its rate, its decay's sparing and its clip in before a run's surface
named them (`OptimConfig.lr`, `OptimConfig.schedule`, `OptimConfig.decay_norms_and_biases`,
`OptimConfig.clip`)."""


def _optim_surface(fields: Mapping[str, _records.JSON]) -> _Fields | None:
    """An optimizer config's fields under today's names (`earlier_surface`), None where it states them so."""
    groups = fields.get("param_groups")
    grouped = isinstance(groups, list) and any(isinstance(group, Mapping) and "spared" in group
                                               for group in groups)
    if not grouped and not any(key in fields for key in EARLIER_RATE_FIELDS):
        return None
    kept = {key: value for key, value in fields.items() if key not in EARLIER_RATE_FIELDS}
    rate = fields.get("learning_rate")
    if isinstance(rate, Mapping):
        kept["schedule"] = rate
    elif rate is not None:
        kept["lr"] = rate
    if "spared" in fields:
        kept["decay_norms_and_biases"] = not fields["spared"]
    if "clip_grads" in fields:
        kept["clip"] = fields["clip_grads"]
    if grouped:
        assert isinstance(groups, list)
        kept["param_groups"] = [
            {**{key: value for key, value in group.items() if key != "spared"},
             "decay_norms_and_biases": not group["spared"]}
            if isinstance(group, Mapping) and "spared" in group else group for group in groups]
    return kept


EARLIER_TRAINER_RECORDS = {"horizon": {}, "update": {"batches": "accumulation", "tokens": "update_tokens"},
                           "fit": {}}
"""The records a trainer config held its length, its update's size and its step's fit in before it held
each setting as its own field (`TrainerConfig.steps`, ...), each record's fields by the name it has now
where that differs."""


def _trainer_surface(fields: Mapping[str, _records.JSON]) -> _Fields | None:
    """A trainer config's fields under today's names (`earlier_surface`), None where it states them so."""
    keep, best = fields.get("keep"), fields.get("best")
    counted = isinstance(keep, int) and not isinstance(keep, bool)
    # A split of None, the planner's: "auto" now (row 11934's flat trainer held None).
    unsplit = "split" in fields and fields["split"] is None
    if not (counted or unsplit or isinstance(best, (str, list))
            or any(key in fields for key in EARLIER_TRAINER_RECORDS)):
        return None
    kept: _Fields = {key: value for key, value in fields.items()
                     if key not in EARLIER_TRAINER_RECORDS and not (unsplit and key == "split")}
    for key, renamed in EARLIER_TRAINER_RECORDS.items():
        held = fields.get(key)
        tagged = _records.class_record(held) if isinstance(held, Mapping) else None
        inner = tagged[1] if tagged is not None else held
        if isinstance(inner, Mapping):
            kept.update({renamed.get(field, field): value for field, value in inner.items()})
    if counted:
        kept["keep"] = {"latest": keep}
    if isinstance(best, str):
        kept["best"] = {"metric": best}
    elif isinstance(best, list):
        if len(best) != 1:
            raise ValueError(f"a saved run's trainer ranks its checkpoints by {len(best)} policies, where "
                             "a run's config ranks by one (TrainerConfig.best); rank by several in "
                             "Trainer.fit's best=")
        kept["best"] = best[0]
    return kept


def _suites(fields: Mapping[str, _records.JSON]) -> dict[str, _records.JSON]:
    """A run record's fields with its split's metrics and cadences, each earlier stated beside the others'
    (a metric's `split`, `val_every`), as the split's suite (`dew.config.Suite`, `RunConfig.suites`, row
    11935): a metric of no split stays the run's (`val_metrics`), and every split some metric names scores
    those beside its own. Carried to 1.0 (ledger 1552)."""
    kept = dict(fields)
    metrics, every = kept.pop("val_metrics", None), kept.pop("val_every", None) or {}
    assert isinstance(every, Mapping), every
    suites: dict[str, _records.JSON] = {split: {"every": value} for split, value in every.items()}
    if isinstance(metrics, list):
        # Each metric as its config is now, the split it named or a null one (an earlier serializer's)
        # its suite's alone.
        shared: list[_records.JSON] = [
            {key: value for key, value in metric.items() if key != "split"} if isinstance(metric, Mapping)
            else metric for metric in metrics if not (isinstance(metric, Mapping) and metric.get("split"))]
        kept["val_metrics"] = shared
        for metric in metrics:
            if isinstance(metric, Mapping) and isinstance(split := metric.get("split"), str):
                suite = suites.setdefault(split, {})
                assert isinstance(suite, dict)
                own = suite.setdefault("metrics", list(shared))
                assert isinstance(own, list)
                own.append({key: value for key, value in metric.items() if key != "split"})
    elif metrics is not None:
        kept["val_metrics"] = metrics
    if suites:
        kept["suites"] = suites
    return kept


EARLIER_DIAGNOSTICS = ("links", "tune_kernels", "xla_flags")
"""The fields a trainer's config held its run's pins in before the run held them apart
(`dew.config.Diagnostics`, row 11933), beside the run's own `kernels`."""


EARLIER_TRAINER_FIELDS = ("dynamic_scale", "lower_for", "compile_ahead", "publish")
"""The fields a trainer's config held its loss scaling in before its numerics held it
(`Numerics.loss_scaling`, row 11935), a target to lower its step for in place of training before a run was
lowered by its own call (`RunConfig.lower`, row 11933), which a saved record, a run that trained, never acts
on, and its publishing before its W&B sink held it. Its length, its update's size and its step's split it held
as it holds them again (`TrainerConfig.steps`, `TrainerConfig.accumulation`, `TrainerConfig.split`, ...; row
11934 held them in records of their own between, `earlier_surface`)."""


def _update_and_fit(fields: Mapping[str, _records.JSON]) -> dict[str, _records.JSON]:
    """A trainer config's fields with its earlier fields (`EARLIER_TRAINER_FIELDS`) as today's: its dynamic
    scale stated as its numerics' loss scaling, unset as theirs, "auto"; its publishing its W&B sink's; and
    no target to lower for. Carried to 1.0 (ledger 1552)."""
    kept: dict[str, _records.JSON] = {key: value for key, value in fields.items()
                                       if key not in EARLIER_TRAINER_FIELDS}
    scaled = fields.get("dynamic_scale")
    if scaled is not None:
        given = kept.get("precision", "default")
        recorded = registry.to_record(numerics_of(given), Numerics) if isinstance(given, str) else given
        precision = dict(recorded) if isinstance(recorded, Mapping) else {}
        kept["precision"] = {**precision, "loss_scaling": "dynamic" if scaled else "none"}
    wandb = kept.get("wandb")
    if fields.get("publish") and isinstance(wandb, Mapping):
        # Publishing is the W&B sink's (`Wandb.publish`), the final checkpoint the run's save policy keeps.
        record, tagged = _records.json_value(wandb, "wandb"), _records.class_record(wandb)
        held = _records.json_value(tagged[1] if tagged is not None else wandb, "wandb")
        assert isinstance(record, dict) and isinstance(held, dict)
        published: dict[str, _records.JSON] = {**held, "publish": True}
        kept["wandb"] = {**record, "fields": published} if tagged is not None else published
    return kept


EARLIER_OPTIMIZER_FIELDS = ("optimizer_opts", "b1", "b2", "weight_decay", "state_dtype")
"""The fields an optimizer's config held its optimizer's settings in before each optimizer held its own
(`dew.optim.OptimizerSpec`, row 11933)."""


MOMENTS_HELD = ("adam", "adamw", "muon", "muonclip")
"""The optimizers whose moments a float32 `state_dtype` held in float32 before their own `state_dtype`, a
`mu_dtype` or `nu_dtype` among their options holding a moment otherwise."""


def _optimizer_spec(fields: Mapping[str, _records.JSON]) -> dict[str, _records.JSON] | None:
    """An optimizer config's fields with its optimizer named by a string and its settings in its earlier
    fields (`EARLIER_OPTIMIZER_FIELDS`) as its optimizer's record holding them (`registry.optimizers`): one of
    Dew's by its alias or its factory's import path (`EARLIER_OPTIMIZER_PATHS`), any other by its import
    path as an `Optax` spec; None where it holds them so already. Its moments' dtypes, a float32
    `state_dtype` with `mu_dtype` and `nu_dtype` among its options, are its spec's `state_dtype`: float32
    both, the default; the parameters' own both ("params"); else the two as they were. Carried to 1.0
    (ledger 1552)."""
    named = fields.get("optimizer", "adamw")
    if not isinstance(named, str) and not any(key in fields for key in EARLIER_OPTIMIZER_FIELDS):
        return None
    given = fields.get("optimizer_opts") or {}
    assert isinstance(given, Mapping), given
    opts: dict[str, _records.JSON] = dict(given)
    assert isinstance(named, str), named
    typed = earlier_typed(named) is not None
    named = EARLIER_OPTIMIZER_PATHS.get(named, named)
    # Dew's own optimizers take no mask; an Optax factory takes its options as they were given. Muon's took
    # its matrices' decay mask by its own keyword, which is the decay's own mask now (`decay_mask`).
    mask = opts.pop("mask", None) if typed else None
    muon_mask = opts.pop("weight_decay_mask", None) if named in MUON_GROUPS else None
    clipped = fields.get("clip_grads")
    if named == "gauss_newton" and isinstance(clipped, (int, float)) and clipped > 0:
        # Its own cap the config's clip wrote over (`OptimConfig.build`), which caps it once now.
        opts.pop("clip", None)
    stated = {key: fields[key] for key in ("b1", "b2", "weight_decay") if fields.get(key) is not None}
    if fields.get("state_dtype", "float32") != "float32":
        stated["state_dtype"] = fields["state_dtype"]
    elif named in MOMENTS_HELD and ({"mu_dtype", "nu_dtype"} & opts.keys()):
        moments = (opts.pop("mu_dtype", "float32"), opts.pop("nu_dtype", "float32"))
        if moments == (None, None):
            stated["state_dtype"] = "params"
        elif moments != ("float32", "float32"):
            stated["state_dtype"] = list(moments)
    settings = {**opts, **stated}
    kept = {key: value for key, value in fields.items() if key not in EARLIER_OPTIMIZER_FIELDS}
    if typed:
        kept = _masked(mask, kept)
        kept = _muon_masked(muon_mask, kept)
        optimizer = {"class": named, "fields": settings}
    else:
        optimizer = {"class": "optax", "fields": {"path": named, "settings": settings}}
    return {**kept, "optimizer": optimizer}


MUON_GROUPS = ("muon", "muonclip")
"""The optimizers an earlier run named whose matrices' decay took its mask by `weight_decay_mask`
(`dew.optim.Muon.masked_by`)."""


def _muon_masked(mask: _records.JSON, config: _Fields) -> _Fields:
    """An earlier Muon's matrices' decay `mask` (its options' `weight_decay_mask`: a boolean or a tree of
    them) as its config's decay mask (`OptimConfig.decay_mask`), or each of its param groups' where it
    groups (`ParamGroup.decay_mask`), each unspared, as Muon spared none of them; none where it masked by
    none. The mask's own checks refuse any other."""
    if mask is None:
        return config
    groups = config.get("param_groups") or []
    assert isinstance(groups, list), groups
    if not groups:
        return {**config, "decay_mask": mask}
    return {**config, "param_groups": [{**group, "decay_mask": mask, "spared": False}
                                       if isinstance(group, Mapping) else group for group in groups]}


EARLIER_OPTIMIZER_PATHS = {"dew.optim:_adam": "adam", "dew.optim:_adamw": "adamw", "dew.optim:_lamb": "lamb",
                           "dew.optim:_muon_groups": "muon", "dew.optim:_muonclip_groups": "muonclip",
                           "dew.laprop:laprop": "laprop", "dew.gauss_newton:gauss_newton": "gauss_newton"}
"""Dew's own optimizers by the import paths of the factories a config could name them by before each held
its own settings (row 11933's review): each the optimizer of its alias, never a generic `Optax` spec."""


def earlier_typed(named: str) -> str | None:
    """The alias of the Dew optimizer an earlier optimizer config named, by its alias or its factory's import
    path (`EARLIER_OPTIMIZER_PATHS`); None for any other, an Optax factory's."""
    named = EARLIER_OPTIMIZER_PATHS.get(named, named)
    return named if named in registry.optimizers.paths and named != "optax" else None


def mask_readers(config: Mapping[str, JSON]) -> tuple[list[JSON], Sequence[Mapping[str, JSON]]]:
    """An earlier optimizer config's param groups, and the parts of it that read a mask among its optimizer's
    options: where it spared its decay (its `spared`, on by default; `OptimConfig.decay_norms_and_biases` off
    now) its own mask was written over the options', as a group's that spared (its `spared`, off by default;
    `ParamGroup.decay_norms_and_biases` on now) was, every group reading it alike; so the config where it
    neither groups nor spares, else each group that does not spare."""
    groups = config.get("param_groups") or []
    assert isinstance(groups, list), groups
    held: list[Mapping[str, JSON]] = [group for group in groups
                                      if isinstance(group, Mapping) and not group.get("spared", False)]
    return groups, held if groups else ([] if config.get("spared", True) else [config])


MASKED_DECAY_MARK = "earlier-masked-decay.json"
"""The file an intermediate Dew (main f124c76b5 until row 12338's deletion of its restore translation) kept
beside a run whose earlier record masked its decay off as it rewrote that record, its masked parts' decay
written as 0 (`{"unmasked_all": true}` or `{"unmasked_groups": [...]}`): read by `unmarked`, where the run
resumes (`dew.config.RunConfig.resumed`)."""

INERT_DECAY = 1e-4
"""The decay a part masked off is given where its rewritten record holds 0: one its mask makes inert,
which builds the masked decay step its checkpoints hold (`unmarked`)."""


def unmarked(record: Mapping[str, object], mark: Mapping[str, object]) -> Mapping[str, object]:
    """A run record an intermediate Dew rewrote beside its `MASKED_DECAY_MARK` (`mark`), as its earlier run
    trained: each part the mark names (its optimizer config ungrouped, or each named param group) masked
    off by its own mask (`OptimConfig.decay_mask`, `ParamGroup.decay_mask`), decaying norms and biases too
    (`decay_norms_and_biases`), its decay of 0,
    which that Dew wrote in its place, given `INERT_DECAY`, so its optimizer holds the masked decay step its
    checkpoints hold. The decay it trained at is not recorded: one of 0 held no step, so a checkpoint of
    such a run is refused at its restore, its tree another, never read across (Main, review of d22)."""
    named = _records.class_record({key: value for key, value in record.items() if key not in MARKERS})
    if named is None:
        return record
    fields = dict(_records.record(named[1], "run fields"))
    optim = fields.get("optim")
    tagged = _records.class_record(optim) if isinstance(optim, Mapping) else None
    config = dict(_records.record(tagged[1] if tagged is not None else optim, "optim"))

    def masked(part: Mapping[str, object], decay_of) -> Mapping[str, object]:
        return {**decay_of(part), "decay_mask": False, "decay_norms_and_biases": True}

    def decayed(fields_: Mapping[str, object]) -> Mapping[str, object]:
        return {**fields_, "weight_decay": fields_.get("weight_decay") or INERT_DECAY}

    if mark.get("unmasked_all") is True:
        optimizer = config.get("optimizer")
        held = _records.class_record(optimizer) if isinstance(optimizer, Mapping) else None
        if held is not None:
            config["optimizer"] = {"class": held[0], "fields": decayed(_records.record(held[1], "optimizer"))}
        config = masked(config, lambda part: dict(part))
    else:
        names = set(_records.listed(mark.get("unmasked_groups"), "unmasked groups", read=_records.text,
                                    what="the names of the param groups the mark unmasked"))
        groups = config.get("param_groups") or []
        assert isinstance(groups, list), groups
        config["param_groups"] = [masked(group, decayed) if isinstance(group, Mapping)
                                  and group.get("name") in names else group for group in groups]
    fields["optim"] = {"class": tagged[0], "fields": config} if tagged is not None else config
    return {**record, "fields": fields}


def _masked(mask: _records.JSON, config: _Fields) -> _Fields:
    """An earlier optimizer's `mask` among its options (`optimizer_opts`), which Dew's own optimizers take no
    longer, as its config's decay mask (`OptimConfig.decay_mask`) or each param group's that read it
    (`ParamGroup.decay_mask`, `mask_readers`): a boolean or a tree of them, Optax's decay mask, which builds
    the decay step, and its state, the earlier Dew built, so its checkpoints restore as they were written.
    Where the config spared its decay (its `spared`, on by default; `OptimConfig.decay_norms_and_biases` off
    now) its own mask was written over it, as a group's that spared (its `spared`, off by default;
    `ParamGroup.decay_norms_and_biases` on now) was: it changed nothing, as a mask of None did not. Any other
    mask, which no record of this Dew states, is refused."""
    if mask is None:
        return config
    groups, read = mask_readers(config)
    if not read:
        return config
    if type(mask) is not bool and not isinstance(mask, Mapping):
        raise ValueError(f"an earlier run masked its decay by {mask!r}, which no record of this Dew states: "
                         "set OptimConfig.decay_mask or its param groups' in its place")
    if not groups:
        return {**config, "decay_mask": mask}
    return {**config, "param_groups": [{**group, "decay_mask": mask}
                                       if isinstance(group, Mapping) and group in read else group
                                       for group in groups]}


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
    """A saved run's record whose optimizer (its record, `_optimizer_spec`) names Muon's Newton-Schulz
    precision by an earlier level's name (`EARLIER_NS_PRECISIONS`), with it named as a dot algorithm, or
    dropped where it was the numerics' own: bfloat16 is the fast forms' own, and stated as `BF16_BF16_F32`
    where the run's forms are accurate (`saved_forms`), whose own is float32 (row 2753). Carried to 1.0
    (ledger 1552)."""
    fields = record.get("fields")
    optim = fields.get("optim") if isinstance(fields, Mapping) else None
    inner = optim.get("fields", optim) if isinstance(optim, Mapping) else None
    spec = inner.get("optimizer") if isinstance(inner, Mapping) else None
    opts = spec.get("fields") if isinstance(spec, Mapping) else None
    named = opts.get("ns_precision") if isinstance(opts, Mapping) else None
    if not isinstance(fields, Mapping) or not isinstance(optim, Mapping) or not isinstance(inner, Mapping) \
            or not isinstance(spec, Mapping) or not isinstance(opts, Mapping) \
            or named not in EARLIER_NS_PRECISIONS:
        return record
    trainer = fields.get("trainer")
    accurate = isinstance(trainer, Mapping) and saved_forms(trainer) == "accurate"
    algorithm = "BF16_BF16_F32" if named == "bfloat16" and accurate else EARLIER_NS_PRECISIONS[named]
    opts = {name: value for name, value in opts.items() if name != "ns_precision"}
    opts = opts if algorithm is None else {**opts, "ns_precision": algorithm}
    inner = {**inner, "optimizer": {**spec, "fields": opts}}
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
