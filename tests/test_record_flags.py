"""A registered kind's records as a command line chooses them (`dew.registry.record_flags`, api03): each
member a subcommand by its alias, its arguments plain flags, no record written as JSON, the records the
flags make exactly the ones a record states."""

import contextlib
import dataclasses
import io

import pytest
import tyro

from dew.config.env import EnvChoice, EnvConfig, EnvRunConfig, RolloutChoice, RolloutConfig
from dew.registry import envs, from_record, implied_subcommand, rollouts, to_record


@dataclasses.dataclass(frozen=True)
class Chosen:
    """A config choosing an optional environment and a rollout, as a run's does."""

    rollout: RolloutChoice = dataclasses.field(default_factory=lambda: RolloutConfig().flagged())
    env: EnvChoice | None = None


@pytest.mark.parametrize(("record", "aliases"), [(RolloutConfig, rollouts), (EnvConfig, envs)],
                         ids=["rollouts", "envs"])
def test_a_kinds_flagged_records_are_the_records_a_record_states(record, aliases):
    """Each aliased member's record, flagged (`Registered.flagged`), writes the record its plain record
    writes (`to_record`) and reads back as it (`from_record`): a command line's subcommand changes no run's
    record."""
    for alias in aliases:
        plain = record(aliases.paths[alias])
        flagged = plain.flagged()
        assert type(flagged) is not record and isinstance(flagged, record)
        assert to_record(flagged, record) == to_record(plain, record)
        assert to_record(from_record(record, to_record(flagged, record)), record) == to_record(plain, record)


def test_a_rollout_is_chosen_by_its_subcommand_and_set_by_plain_flags():
    """`rollout:control --rollout.unroll-length 64` is `RolloutConfig("control", {"unroll_length": 64})`'s
    record, no flag left at its default stated; none given, the default's subcommand, its record stating
    none; an environment chosen by its own (`env:flight --env.count 4`) holds its settings beside its
    arguments."""
    parsed = tyro.cli(Chosen, args=["rollout:control", "--rollout.unroll-length", "64"])
    stated = RolloutConfig("control", {"unroll_length": 64})
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
    assert "--rollout.fields" not in text and "--env.fields" not in text, text
