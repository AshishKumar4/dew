"""A registered kind's records as a command line chooses them (`dew._flags.record_flags`, api03): each
member a subcommand by its alias, its arguments plain flags, no record written as JSON, the records the
flags make exactly the ones a record states."""

import contextlib
import dataclasses
import functools
import io

import pytest
import tyro

from dew import registry
from dew._flags import implied_subcommand
from dew.config.env import EnvChoice, EnvConfig, EnvRunConfig, RolloutChoice, RolloutConfig
from dew.objectives.rl.control import ControlRollout
from dew.registry import envs, from_record, rollouts, to_record


@dataclasses.dataclass(frozen=True)
class Chosen:
    """A config choosing an optional environment and a rollout, as a run's does."""

    rollout: RolloutChoice = dataclasses.field(default_factory=lambda: RolloutConfig().flagged())
    env: EnvChoice | None = None


@pytest.mark.parametrize(("record", "aliases"), [(RolloutConfig, rollouts), (EnvConfig, envs)],
                         ids=["rollouts", "envs"])
def test_a_kinds_flagged_records_are_the_records_a_record_states(record, aliases):
    """Each aliased member's record, flagged (`Bound.flagged`), writes the record its plain record
    writes (`to_record`) and reads back as it (`from_record`): a command line's subcommand changes no run's
    record."""
    for alias in aliases:
        plain = record(functools.partial(aliases[alias]))
        flagged = plain.flagged()
        assert type(flagged) is not record and isinstance(flagged, record)
        assert to_record(flagged, record) == to_record(plain, record)
        assert to_record(from_record(record, to_record(flagged, record)), record) == to_record(plain, record)


def test_a_rollout_is_chosen_by_its_subcommand_and_set_by_plain_flags():
    """`rollout:control --rollout.unroll-length 64` is the control rollout bound by `unroll_length=64`'s
    record, no flag left at its default stated; none given, the default's subcommand, its record stating
    none; an environment chosen by its own (`env:flight --env.count 4`) holds its settings beside its
    arguments."""
    parsed = tyro.cli(Chosen, args=["rollout:control", "--rollout.unroll-length", "64"])
    stated = RolloutConfig(functools.partial(ControlRollout, unroll_length=64))
    assert to_record(parsed.rollout, RolloutConfig) == to_record(stated, RolloutConfig)
    assert to_record(tyro.cli(Chosen, args=[]).rollout, RolloutConfig) == to_record(RolloutConfig(),
                                                                                     RolloutConfig)
    flown = tyro.cli(Chosen, args=["env:flight", "--env.count", "4"]).env
    assert flown is not None and (flown.name, flown.count) == (envs.paths["flight"], 4)
    assert implied_subcommand(RolloutConfig().flagged()) == "dreamer"
    assert implied_subcommand(RolloutConfig()) is None


def test_an_environment_runs_help_names_its_choices_and_their_flags_and_no_json():
    """An environment run's command line (`EnvRunConfig.cli`) offers its rollouts and environments as
    subcommands, and no flag takes a rollout's or an environment's fields as a JSON record."""
    shown = io.StringIO()
    with contextlib.redirect_stdout(shown), pytest.raises(SystemExit):
        EnvRunConfig.cli(["--help"])
    text = shown.getvalue()
    assert "rollout:control" in text and "rollout:dreamer" in text
    hidden = [f"--{kind}.{held}" for kind in ("rollout", "env") for held in ("bound", "defaults")]
    assert not any(flag in text for flag in hidden), text


@pytest.mark.parametrize("run", [EnvRunConfig(),
                                 EnvRunConfig(rollout=RolloutConfig(functools.partial(
                                     registry.rollouts["control"])))],
                         ids=["declared", "control"])
def test_an_env_run_saved_and_read_back_is_the_run(run, tmp_path):
    """An environment run, its rollout a kind its command line chooses by subcommands, saved and read back
    is the run it saved (dataclass equality), its rollout its kind's own record: the subcommands' generated
    classes are its parser's alone (the API03 review: read back, a run held them and was another run)."""
    run.save(str(tmp_path))
    read = EnvRunConfig.load(str(tmp_path))
    assert read == run
    assert type(read.rollout) is RolloutConfig


def test_an_env_run_parsed_from_its_command_line_holds_its_kinds_own_records():
    """A run parsed from its command line (`EnvRunConfig.cli`) holds each record as its kind's own class,
    equal to the one a caller builds with the same settings (`Bound.recorded`)."""
    parsed = EnvRunConfig.cli(["rollout:control", "--rollout.unroll-length", "64"])
    assert type(parsed.rollout) is RolloutConfig
    assert parsed.rollout == RolloutConfig(functools.partial(ControlRollout, unroll_length=64))
    assert EnvRunConfig.cli([]) == EnvRunConfig()


def test_a_flag_set_to_its_members_default_over_a_recipes_value_takes_it():
    """A rollout flag given its member's default over a recipe that binds another value runs the default
    the command line asked for (row 14127: `--rollout.train-ratio 1024.0`, `DreamerRollout`'s own, over a
    recipe's 16.0 kept 16.0, the flag dropped as unchanged from its default); a flag left unset keeps the
    recipe's value, and one set elsewhere takes its own."""
    from dew.objectives.rl.dreamer import DreamerRollout

    recipe = EnvRunConfig(rollout=RolloutConfig(functools.partial(DreamerRollout, train_ratio=16.0)))
    default = DreamerRollout.__dataclass_fields__["train_ratio"].default
    for args, ratio in ((["--rollout.train-ratio", str(default)], default), ([], 16.0),
                        (["--rollout.train-ratio", "64.0"], 64.0)):
        parsed = EnvRunConfig.cli(args, default=recipe)
        assert parsed.rollout.fields.get("train_ratio", default) == ratio
