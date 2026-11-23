"""Completed tools survive SIGKILL; pending actions and policy provenance survive too."""

import dataclasses
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import jax
import numpy as np
import pytest
from episode_support import (
    IDENTITY,
    SAMPLING,
    Harness,
    SquareSession,
    action,
    build,
    cohort_snapshot,
    collect,
    environment,
)
from process_support import device_flags
from tool_policy import CALL_THREE, EOS, GROUPS, NINE, START

from dew.objectives.rl import EpisodeJournal, RecoverableEnvironment, Transcript, session_of
from dew.sampling import Sampling

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("episode_recovery_worker.py")


def test_journal_refuses_a_different_policy_at_identical_clocks(tmp_path):
    trainer, collector = build(environment(), journal=EpisodeJournal(str(tmp_path)))
    state = trainer.initial_state()
    complete = collect(collector, state)
    assert collect(collector, state) == complete
    changed = replace(state, variables=jax.tree.map(lambda value: value + .001, state.variables))
    with pytest.raises(ValueError):
        collect(collector, changed)


def test_a_cooldown_branchs_cohorts_are_its_own_beside_the_runs_at_the_same_clocks(tmp_path):
    """A cooldown branch draws at the run's steps and keys under its own weights: its cohorts are its
    own (`Snapshot.branch`), so the run, drawing at those clocks after it, is not refused as another
    policy, and a resumed branch reads its own turns back."""
    from dew.objectives.rl.sessions import Task

    trainer, collector = build(environment(), journal=EpisodeJournal(str(tmp_path)))
    state = trainer.initial_state()
    decayed = replace(state, variables=jax.tree.map(lambda value: value + .001, state.variables))
    snapshot = cohort_snapshot(decayed, 23)
    branch = collector.episodes(replace(snapshot, branch="cooldown-4"), [Task("31")], GROUPS)
    assert collector.episodes(replace(snapshot, branch="cooldown-4"), [Task("31")], GROUPS) == branch
    run = collect(collector, state)
    assert collect(collector, state) == run
    with pytest.raises(ValueError):
        collect(collector, decayed)


@dataclasses.dataclass(frozen=True)
class Raised:
    """A logits transform's state held by its instance, which a bound method reads."""

    by: float

    def apply(self, state, logits):
        del state
        return logits + self.by


def test_a_journaled_policy_holding_a_bound_method_is_refused_where_it_is_built(tmp_path):
    """A journaled collection whose policy's chain holds a method bound to an instance (`Raised(2).apply`)
    is refused where its collector is built: a record names the method by its path alone, which a reload
    imports unbound, its receiver lost (the API02 review)."""
    from dew.registry import Unrecordable

    with pytest.raises(Unrecordable):
        build(environment(), journal=EpisodeJournal(str(tmp_path)),
              sampling=replace(SAMPLING, logits=(Raised(2.0).apply,)))


def test_a_journaled_cohort_records_its_tasks_inherited_controls_and_reads_them_back(tmp_path):
    """A journaled cohort whose task binds stopping the collector's sampling leaves at its default records
    the policy its turns drew under, the task's criteria among it (`EpisodeCollector._effective`), and reads
    the same episodes back from its journal; where its episodes and journal held the collector's own,
    dropping the inherited controls (the API02 review)."""
    from dew.sampling.decoding import MaxNewTokens

    _, collector = build(environment(), journal=EpisodeJournal(str(tmp_path)))
    bound = Sampling(stopping=(MaxNewTokens(1),))
    collector = replace(collector, policy=replace(collector.policy, sampling=bound))
    trainer, _ = build(environment())
    state = trainer.initial_state()
    episodes = collect(collector, state)
    assert {episode.sampling for episode in episodes} == {SAMPLING.with_defaults_of(bound)}
    assert collect(collector, state) == episodes


def test_a_journaled_cohort_resumed_under_another_effective_policy_is_refused(tmp_path):
    """A journaled cohort's fingerprint states the policy it drew under, its pad id among it, which no
    component holds (nor its stop strings, which the fingerprint states beside it): a resume whose policy
    differs in its pad id alone is another cohort, refused, where its turns were read back as the same
    (the API02 review); the same policy reads them back."""
    _, collector = build(environment(), journal=EpisodeJournal(str(tmp_path)))
    trainer, _ = build(environment())
    state = trainer.initial_state()
    episodes = collect(collector, state)
    assert collect(collector, state) == episodes
    _, moved = build(environment(), journal=EpisodeJournal(str(tmp_path)),
                     sampling=replace(SAMPLING, pad_token_id=SAMPLING.pad + 1))
    with pytest.raises(ValueError):
        collect(moved, state)


def test_journal_tells_apart_policies_whose_weights_are_equal_bytes_of_another_dtype():
    """A policy's weights in float8_e4m3fn and the same bytes read as float8_e4m3fnuz are two
    policies, which NumPy names by one '<V1' code: the journal's policy fingerprint keys each by its
    named dtype, so a recovery under the other refuses, while the same weights match again."""
    import ml_dtypes

    from dew.objectives.rl.journal import policy_digest

    bits = np.arange(64, dtype=np.uint8).reshape(8, 8)
    weights = {"params": {"kernel": bits.view(ml_dtypes.float8_e4m3fn)}}
    other = {"params": {"kernel": bits.view(ml_dtypes.float8_e4m3fnuz)}}
    again = {"params": {"kernel": bits.copy().view(ml_dtypes.float8_e4m3fn)}}
    assert policy_digest(weights) == policy_digest(again)
    assert policy_digest(weights) != policy_digest(other)


class Saying(SquareSession):
    """The square tool, whose observations also say the chat messages they added, as read-only mappings."""

    def reset(self):
        return replace(super().reset(), messages=(MappingProxyType({"role": "user", "content": "square"}),))

    def step(self, action):
        seen = super().step(action)
        return replace(seen, messages=(MappingProxyType({"role": "tool", "content": seen.detail}),))

    def get_state(self):
        return json.dumps({"answer": self.answer}).encode()

    def set_state(self, state):
        self.answer = json.loads(state)["answer"]


class Sayings(Harness):
    @contextmanager
    def __call__(self, identity):
        yield Saying(identity, self)


def test_a_journaled_episode_keeps_the_messages_its_environment_said(tmp_path):
    """Messages an environment gives in any mapping are kept as JSON: a journaled collection writes
    them and reads the same episodes back, and each call's transcript lists those before it."""
    trainer, collector = build(Sayings(), journal=EpisodeJournal(str(tmp_path)))
    state = trainer.initial_state()
    episodes = collect(collector, state)
    assert collect(collector, state) == episodes
    for episode in episodes:
        said = [{"role": "user", "content": "square"},
                *({"role": "tool", "content": turn.observation.detail} for turn in episode.transitions)]
        calls = session_of(episode, group="0").calls
        assert [call.transcript for call in calls] == [Transcript(tuple(said[:turn + 1]))
                                                       for turn in range(len(calls))]


def kept_scores(state, logits):
    """A logits transform that keeps the scores: a function a journaled policy holds by its import path."""
    del state
    return logits


def test_a_journaled_policy_holding_a_function_is_recorded_by_its_path_and_read_back(tmp_path):
    """A journaled collection whose policy holds a function in its chain (the documented `favor_short`
    kind) writes each episode's policy as the registry records it, the function by its import path, and
    reads the same episodes back; a function no record can name (a lambda) is refused where the collector
    is built, not at its first turn's commit (Train 1208's review F2: the commit crashed in `asdict`'s
    JSON)."""
    from dew.registry import Unrecordable

    sampling = replace(SAMPLING, logits=(kept_scores,))
    trainer, collector = build(environment(), journal=EpisodeJournal(str(tmp_path)), sampling=sampling)
    state = trainer.initial_state()
    episodes = collect(collector, state)
    assert collect(collector, state) == episodes
    assert {episode.sampling.logits for episode in episodes} == {(kept_scores,)}
    with pytest.raises(Unrecordable):
        build(environment(), journal=EpisodeJournal(str(tmp_path / "unnamed")),
              sampling=replace(SAMPLING, logits=(lambda state, logits: logits,)))


def test_a_cohort_of_one_process_draws_under_a_notebooks_function(monkeypatch):
    """A cohort of one process, journaling nothing, whose policy holds a function a notebook defined (its
    module holds no source file) draws its episodes: the policy's identity, whose statement asks each
    function's code of a file, is stated only where processes agree on it or a journal keeps it, where it
    refused the function at the cohort's start (the models review)."""
    import types

    cell = types.ModuleType("a_notebooks_cell")
    exec("def kept(state, logits):\n    return logits\n", cell.__dict__)
    monkeypatch.setitem(sys.modules, cell.__name__, cell)
    trainer, collector = build(environment(), sampling=replace(SAMPLING, logits=(cell.kept,)))
    episodes = collect(collector, trainer.initial_state())
    assert len(episodes) == GROUPS and {episode.sampling.logits for episode in episodes} == {(cell.kept,)}

@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True, eq=False)
class FreshlyFlattened:
    """A logits transform adding `bias` to the scores, whose flatten hands a fresh array each walk."""

    bias: np.ndarray

    def __call__(self, state, scores):
        del state
        return scores + self.bias

    def tree_flatten(self):
        return (np.array(self.bias, copy=True),), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        del aux
        return cls(*children)


def test_a_policy_whose_transform_flattens_its_arrays_afresh_states_one_identity():
    """A policy whose transform hands its arrays afresh at each flatten (a pytree's own `tree_flatten`) is
    stated (`policy_identity`), its arrays fingerprinted from the one walk of its leaves that states it,
    and alike for an equal policy, where a second walk read arrays the first never saw and raised a
    KeyError (the models review)."""
    from dew.sampling.text import policy_identity

    def policy(bias):
        return replace(SAMPLING, logits=(FreshlyFlattened(np.asarray([bias, 0.0], np.float32)),))

    assert policy_identity(policy(1.0)) == policy_identity(policy(1.0)) != policy_identity(policy(2.0))

PLUGIN = """import dataclasses

import jax

from . import settings as cfg

_OPAQUE = object()


def biased(state, scores):
    return scores + cfg.BIAS


@dataclasses.dataclass(frozen=True)
class Scaled:
    scale: float

    def __call__(self, state, scores):
        return self.scaled(scores) + cfg.BIAS

    def scaled(self, scores):
        return scores * self.scale

    def shifted(self, state, scores):
        return scores + cfg.BIAS

    def unrelated(self):
        return _OPAQUE


class Base:
    def __call__(self, state, scores):
        return self.helper(scores)

    def helper(self, scores):
        return scores + cfg.BIAS


@dataclasses.dataclass(frozen=True)
class Derived(Base):
    scale: float

    def __call__(self, state, scores):
        return super().__call__(state, scores) * self.scale


def bound_shift(self, state, scores):
    return scores + cfg.BIAS


@dataclasses.dataclass(frozen=True)
class Helped:
    def helper(self, scores):
        return scores + cfg.BIAS


class OpaqueBase:
    def helper(self, scores):
        return _OPAQUE


@dataclasses.dataclass(frozen=True)
class Overriding(OpaqueBase):
    def __call__(self, state, scores):
        return self.helper(scores)

    def helper(self, scores):
        return scores + cfg.BIAS


class Classy:
    @classmethod
    def transform(cls, state, scores):
        return cls.helper(scores)

    @classmethod
    def helper(cls, scores):
        return scores + cfg.BIAS


class MixedBase:
    def offset(self, scores):
        return scores

    def helper(self, scores):
        return _OPAQUE


@dataclasses.dataclass(frozen=True)
class Mixed(MixedBase):
    def __call__(self, state, scores):
        return super().offset(scores) + self.helper(scores)

    def helper(self, scores):
        return scores + cfg.BIAS


@dataclasses.dataclass(frozen=True)
class NamedSuper(MixedBase):
    def __call__(self, state, scores):
        return super(NamedSuper, self).offset(scores)


class LexBase:
    def helper(self, scores):
        return scores + cfg.BIAS


class LexMid(LexBase):
    def transform(self, state, scores):
        return super().helper(scores)

    def helper(self, scores):
        return _OPAQUE


@dataclasses.dataclass(frozen=True)
class LexDerived(LexMid):
    pass


LexDerived.__call__ = LexMid.transform


jitted = jax.jit(biased)
"""
"""An installed policy plugin's code: a function, a dataclass transform (whose `__call__` reaches a helper
and whose bound `shifted` is a transform of its own, beside a method no call reaches reading an opaque
sentinel) and a jitted function, each reading its settings module's bias through an alias."""


PLUGIN_FILES = ["policy_plugin/__init__.py", "policy_plugin/settings.py", "policy_plugin/alt.py"]
"""The installed plugin's files, as its distribution's record lists them."""


def recorded(monkeypatch, tmp_path, files: dict[str, list[str]]) -> None:
    """Installed distributions' records of their files under `tmp_path` (`importlib.metadata.distribution`),
    by distribution: what tells a module's distribution (`dew._stated._owners`)."""
    import importlib.metadata
    from pathlib import PurePosixPath

    class Installed:
        def __init__(self, held):
            self.files = [PurePosixPath(each) for each in held]

        def locate_file(self, held):
            return tmp_path / held

    installed = {name: Installed(held) for name, held in files.items()}
    real = importlib.metadata.distribution
    monkeypatch.setattr(importlib.metadata, "distribution",
                        lambda name: installed[name] if name in installed else real(name))


@pytest.fixture
def dependencies(monkeypatch):
    """Dew's published distribution's requirements as a test declares them (`importlib.metadata.requires`
    of `dewml`, the name `pyproject.toml` publishes), its statement's closure of them read afresh."""
    import importlib.metadata

    from dew import _stated

    declared: dict[str, list[str]] = {"dewml": []}
    real = importlib.metadata.requires
    monkeypatch.setattr(importlib.metadata, "requires",
                        lambda name: declared[name] if name in declared else real(name))
    _stated._dependencies.cache_clear()
    yield declared
    _stated._dependencies.cache_clear()


@pytest.fixture
def installed_plugin(tmp_path, monkeypatch):
    """A policy plugin installed as a distribution's (`_sources`: its file under an install directory, its
    top-level package a distribution's outside Dew's dependencies), its bias in a module of its own read
    through an alias (`from . import settings as cfg`), as a plugin's configuration may be."""
    import importlib

    from dew import _sources

    package = tmp_path / "policy_plugin"
    package.mkdir()
    (package / "settings.py").write_text("BIAS = 1.0\n")
    (package / "alt.py").write_text("def bound_shift(self, state, scores):\n    return scores * 2\n\n\n"
                                    "def transform(self, state, scores):\n    return self.helper(scores)\n")
    (package / "__init__.py").write_text(PLUGIN)
    monkeypatch.syspath_prepend(str(tmp_path))
    recorded(monkeypatch, tmp_path, {"policy-plugin": PLUGIN_FILES})
    # This test's plugin, never an earlier test's of another directory, and none left after it.
    for name in ("policy_plugin", "policy_plugin.settings", "policy_plugin.alt"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    importlib.invalidate_caches()
    plugin = importlib.import_module("policy_plugin")
    monkeypatch.setitem(sys.modules, "policy_plugin", plugin)
    monkeypatch.setitem(sys.modules, "policy_plugin.settings", plugin.cfg)
    # Its directory an install's, its package a distribution's (`_sources.environment`).
    installed = _sources.Environment((tmp_path.resolve(),), {"policy_plugin": ["policy-plugin"]})
    with _sources.environment(installed):
        assert not _sources.own_source("policy_plugin")
        yield plugin


def identities_across_bias(plugin, monkeypatch, transform) -> tuple[str, str]:
    """The policy identities (`policy_identity`) of a policy holding `transform` before and after the
    plugin's settings move its bias (as a process's environment may set it)."""
    from dew.sampling.text import policy_identity

    policy = replace(SAMPLING, logits=(transform,))
    first = policy_identity(policy)
    monkeypatch.setattr(plugin.cfg, "BIAS", 2.0)
    return first, policy_identity(policy)


@pytest.mark.parametrize("held", ["function", "dataclass", "bound method", "derived", "overriding",
                                  "bound elsewhere", "classmethod", "super and self", "lexical super",
                                  "jitted"])
def test_an_installed_plugins_policy_states_its_modules_data(installed_plugin, monkeypatch, held):
    """A policy holding an installed plugin's transform (a distribution's outside Dew's dependencies): a
    function, a dataclass whose `__call__` reads it, a jitted function; each states the data the plugin's
    modules hold that it reads (`stated`'s `installed`): its settings module's bias, read through an alias,
    set apart (as a process's environment may set it) states two identities (`policy_identity`), which the
    journal's resume refuses, where the two agreed (the models review; a dataclass's class body and a
    jitted function were stated by name alone, the confirm review). A method no call reaches, reading an
    opaque sentinel, is no part of it: the dataclass states (the confirm review: every method was walked and
    the sentinel refused it); a subclass's `super().__call__()` reaches its base's helper (the bias read
    there), a definition past its own class followed where it calls `super` alone: an override of a base
    reading the sentinel states the override (the confirm review: every definition followed refused it); an
    installed function bound to an object of no member of its name reaches what its own body reads
    (`self.helper`, the confirm review: its name alone seeded none); a classmethod reaches its class's
    members (`cls.helper`), its receiver the class itself; a method reading `super().offset` beside
    `self.helper` dispatches each by its own receiver, the shadowed base's `helper` reading the sentinel no
    part of it; a `super()` dispatches past the class its function was written in, whatever slot holds it
    (`LexDerived.__call__` is `LexMid.transform`: past `LexMid`, never its sentinel `helper`; the confirm
    review)."""
    import importlib
    import types

    from dew.sampling.text import policy_identity

    elsewhere = importlib.import_module("policy_plugin.alt").transform
    transform = {"function": installed_plugin.biased, "dataclass": installed_plugin.Scaled(2.0),
                 "bound method": installed_plugin.Scaled(2.0).shifted,
                 "derived": installed_plugin.Derived(2.0), "overriding": installed_plugin.Overriding(),
                 "bound elsewhere": types.MethodType(elsewhere, installed_plugin.Helped()),
                 "classmethod": installed_plugin.Classy.transform, "super and self": installed_plugin.Mixed(),
                 "lexical super": installed_plugin.LexDerived(),
                 "jitted": installed_plugin.jitted}[held]
    first, moved = identities_across_bias(installed_plugin, monkeypatch, transform)
    assert moved != first
    monkeypatch.setattr(installed_plugin.cfg, "BIAS", 1.0)
    assert policy_identity(replace(SAMPLING, logits=(transform,))) == first


def test_a_plugin_among_dews_published_requirements_is_named(installed_plugin, dependencies, monkeypatch):
    """A plugin's distribution among Dew's published requirements (`dewml`'s metadata) is a dependency, which
    its version states: its module's bias is not walked, so the identity holds across it (an installed Dew's
    closure read of `dew`, no distribution, fell back or refused, the confirm review)."""
    dependencies["dewml"] = ["policy-plugin>=1"]
    dependencies["policy-plugin"] = []
    first, moved = identities_across_bias(installed_plugin, monkeypatch, installed_plugin.biased)
    assert moved == first


def test_dews_dependencies_are_its_published_distributions_requirements_alone(dependencies):
    """Dew's dependency closure (`dew._stated._dependencies`), where Dew is installed, is its published
    distribution's (`dewml`, `pyproject.toml`'s name) requirements and theirs, transitively, each by its
    normalized name, markers and versions dropped: where it read `dew`, no distribution, it fell back to the
    source tree's every extra, a closure of most of the environment, which named a plugin a dependency (the
    confirm review)."""
    from dew import _stated

    dependencies["dewml"] = ["A_Dependency>=1; python_version >= '3'"]
    dependencies["a-dependency"] = ["b.dependency"]
    dependencies["b-dependency"] = []
    assert _stated._dependencies() == {"a-dependency", "b-dependency"}

def test_dews_dependencies_hold_a_requirement_where_its_marker_holds_for_the_extras_asked(dependencies):
    """A requirement's marker is evaluated for the extras its requirer asked (`packaging`), none at Dew's
    root: one an inactive extra names is no dependency, one an extra a dependency was asked for names is;
    where every marker was stripped an inactive extra's plugin was exempted (the confirm review)."""
    from dew import _stated

    dependencies["dewml"] = ["a-dependency[fast]", "policy-plugin; extra == 'gpu'"]
    dependencies["a-dependency"] = ["b-dependency; extra == 'fast'", "c-dependency; extra == 'slow'"]
    dependencies["b-dependency"] = []
    assert _stated._dependencies() == {"a-dependency", "b-dependency"}


def test_a_plugin_whose_file_no_record_lists_is_walked(installed_plugin, dependencies, monkeypatch, tmp_path):
    """A plugin among Dew's requirements whose module's file its distribution's record does not list is
    walked, no dependency vouching for it (`_owners`, fail closed), where a dependency sharing its top-level
    name vouched for it whole (the confirm review)."""
    recorded(monkeypatch, tmp_path, {"policy-plugin": ["policy_plugin/elsewhere.py"]})
    dependencies["dewml"] = ["policy-plugin"]
    dependencies["policy-plugin"] = []
    first, moved = identities_across_bias(installed_plugin, monkeypatch, installed_plugin.biased)
    assert moved != first

def test_a_plugin_installed_in_a_dependencys_namespace_is_walked(installed_plugin, dependencies, monkeypatch,
                                                                 tmp_path):
    """A plugin installed beside a dependency in one top-level namespace (`policy_plugin`, which a dependency
    `policy-core` shares) is the distribution's whose files hold its modules (`_owners`): its bias walked and
    two identities stated, where any dependency sharing the namespace named the whole of it (the confirm
    review)."""
    from dew import _sources

    recorded(monkeypatch, tmp_path, {"policy-core": ["policy_plugin/core.py"], "policy-plugin": PLUGIN_FILES})
    dependencies["dewml"] = ["policy-core"]
    dependencies["policy-core"] = []
    shared = _sources.Environment((tmp_path.resolve(),), {"policy_plugin": ["policy-core", "policy-plugin"]})
    with _sources.environment(shared):
        first, moved = identities_across_bias(installed_plugin, monkeypatch, installed_plugin.biased)
    assert moved != first


class UserBase:
    """A base of the user's own code."""

    def shift(self, scores):
        return scores + 1


@dataclasses.dataclass(frozen=True)
class UserChild(UserBase):
    """A user's class whose method reaches its base through a `super()` it holds a moment."""

    def transform(self, state, scores):
        parent = super()
        return parent.shift(scores)


def test_a_users_own_bound_callback_holding_its_super_is_stated():
    """A bound method of the user's own class whose body holds `super()` before reading through it is stated
    (`policy_identity`), its class stated whole as before: no dispatch of the user's own code is read from its
    bytecode, which an installed plugin's reachability alone needs (the confirm review: it was refused)."""
    from dew.sampling.text import policy_identity

    policy = replace(SAMPLING, logits=(UserChild().transform,))
    assert policy_identity(policy) == policy_identity(replace(SAMPLING, logits=(UserChild().transform,)))

def test_a_plugin_whose_super_names_its_receiver_is_refused(installed_plugin):
    """An installed plugin's method reading through a two-argument `super(...)`, whose dispatch starts past a
    class the code names rather than its own, is refused (`policy_identity`) rather than stated by a guess
    (the confirm review)."""
    from dew.sampling.text import policy_identity

    with pytest.raises(ValueError):
        policy_identity(replace(SAMPLING, logits=(installed_plugin.NamedSuper(),)))

def test_two_installed_functions_of_one_name_bound_to_one_object_state_apart(installed_plugin):
    """Two installed functions of one name bound to one object (`types.MethodType`) are two policies: each
    bound method states the function it binds beside its object, where the object and the name alone
    agreed (the confirm review)."""
    import importlib
    import types

    from dew.sampling.text import policy_identity

    receiver = installed_plugin.Scaled(2.0)
    alt = importlib.import_module("policy_plugin.alt")
    one, other = (replace(SAMPLING, logits=(types.MethodType(function, receiver),))
                  for function in (installed_plugin.bound_shift, alt.bound_shift))
    assert policy_identity(one) != policy_identity(other)

def test_a_statement_walks_installed_code_only_where_it_is_asked_to_in_either_order(installed_plugin):
    """One installed function stated with and without its installed code walked (`stated`'s `installed`), in
    either order, states its name alone without and its reads with: the walk's choice holds for its one
    statement, never served to another (no statement is cached across them)."""
    from dew._stated import stated

    plain = stated(installed_plugin.biased, "a plugin")
    walked = stated(installed_plugin.biased, "a plugin", installed=True)
    assert walked != plain
    assert stated(installed_plugin.biased, "a plugin", installed=True) == walked
    assert stated(installed_plugin.biased, "a plugin") == plain


def test_a_plugin_module_whose_state_no_statement_reads_is_refused(installed_plugin, monkeypatch):
    """A module of the installed plugin holding no file (as an extension's state is no statement's) reached
    by its function is refused, the policy with it (`policy_identity`), rather than stated by its name."""
    import types

    from dew.sampling.text import policy_identity

    held = types.ModuleType("policy_plugin.settings")
    vars(held).update(BIAS=1.0)
    monkeypatch.setitem(sys.modules, "policy_plugin.settings", held)
    monkeypatch.setattr(installed_plugin, "cfg", held)
    with pytest.raises(ValueError):
        policy_identity(replace(SAMPLING, logits=(installed_plugin.biased,)))

def launch(directory, mode):
    env = {**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'tests'}",
           "XLA_FLAGS": device_flags(2), "OMP_NUM_THREADS": "1"}
    return subprocess.Popen([sys.executable, str(WORKER), str(directory), mode],
                            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def finish(process):
    try:
        output, _ = process.communicate(timeout=90)
    except BaseException:
        process.kill()
        process.wait()
        raise
    assert process.returncode == 0, output


def test_sigkill_resumes_completed_turns_and_produces_the_same_update(tmp_path):
    """Two real CPU-device runs: committed tools once, actions and update exact."""
    baseline, resumed = tmp_path / "baseline", tmp_path / "resumed"
    finish(launch(baseline, "ok"))
    process = launch(resumed, "pause_final")
    try:
        deadline = time.monotonic() + 60
        while not (resumed / "ready").exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert (resumed / "ready").exists(), (
            process.communicate(timeout=1)[0] if process.poll() is not None else "no pending tool"
        )
    finally:
        process.kill()
        process.wait()
    first_calls = (resumed / "calls.jsonl").read_text().splitlines()
    assert len(first_calls) == 8
    finish(launch(resumed, "ok"))
    assert (resumed / "calls.jsonl").read_text() == (baseline / "calls.jsonl").read_text()
    np.testing.assert_array_equal(np.load(resumed / "parameters.npy"), np.load(baseline / "parameters.npy"))
    original = json.loads((baseline / "episodes.json").read_text())
    restored = json.loads((resumed / "episodes.json").read_text())
    for before, after in zip(original, restored, strict=True):
        assert before["identity"] == after["identity"]
        assert before["reward"] == after["reward"]
        assert before["transitions"] == after["transitions"]

    with environment()(IDENTITY) as first:
        assert isinstance(first, RecoverableEnvironment)
        first.reset()
        observation = first.step(action((START,), CALL_THREE, EOS))
        snapshot = first.get_state()
    with environment()(IDENTITY) as second:
        assert isinstance(second, RecoverableEnvironment)
        second.set_state(snapshot)
        final = second.step(action(observation.context, 5, EOS))
        assert observation.context[-1] == NINE
        assert json.loads(final.detail) == {"answer": 9, "expected": 9}
