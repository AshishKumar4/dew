"""dew.nn.kernels.tuning's picks in Dew's own JSON: every launch a step traces, its op, its arguments and
its timings, written and read back equal, through tokamax's public API alone."""

import dataclasses
import functools
import itertools
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tokamax
from reference_error import assert_computes_the_oracle

from dew.nn.kernels import generation, norm_swiglu, tuning
from dew.nn.kernels.templates import attention, report
from dew.nn.precision import Numerics


def test_a_steps_launches_and_their_timings_round_trip_through_dews_json():
    """Each launch of a norm seam and SwiGLU step, written as JSON (`dumps_launches`, `dumps_picks`)
    and read back (`loads_launches`, `_read`), is the same op at the same key, with its configs'
    times; tokamax's own BoundArguments key them, so a read launch looks its pick up as a traced one
    does."""
    from test_tuning import operands, step

    with tuning.launches():
        found = tokamax.autotuning.get_bound_args(step(), *operands(15))
    assert found
    keyed = [ba.op.keyed(ba) for ba in found]
    again = tuning.loads_launches(tuning.dumps_launches(keyed))
    assert [(ba.op, ba.autotuning_cache_key) for ba in again] == [
        (tuning._identity(ba.op), ba.autotuning_cache_key) for ba in keyed]
    timing = tokamax.BenchmarkData(compile_time_ms=1.0, lower_time_ms=2.0, evaluation_times_ms=(3.0, 4.0),
                                   metadata={})
    picks = [(ba, tuning.Timings({ba.heuristics_config: timing})) for ba in keyed]
    kind = jax.devices()[0].device_kind
    entries = json.loads(tuning.dumps_picks(kind, picks))["data"]
    read = tuning._keyed([(tuning._launch(entry), tuning._timings(entry["timings"])) for entry in entries])
    assert set(read) == {(tuning._identity(ba.op), ba.autotuning_cache_key) for ba in keyed}
    for (op, _), (ba, timings) in read.items():
        [config] = timings
        assert config == ba.heuristics_config and timings.fastest_config == config
        assert timings[config] == timing
        assert isinstance(op, (norm_swiglu.RmsForward, norm_swiglu.RmsBackward, norm_swiglu.SwigluForward,
                               norm_swiglu.SwigluBackward))


def _norm_launch_and_two_configs():
    """A norm launch of `test_tuning`'s step, its derived config and another of its space."""
    from test_tuning import operands, step

    with tuning.launches():
        found = tokamax.autotuning.get_bound_args(step(), *operands(15))
    ba = next(ba.op.keyed(ba) for ba in found if len(ba.autotuning_configs) > 1)
    other = next(config for config in ba.autotuning_configs if config != ba.heuristics_config)
    return ba, ba.heuristics_config, other


def _timed(ms):
    return tokamax.BenchmarkData(compile_time_ms=0.0, lower_time_ms=0.0, evaluation_times_ms=(ms,),
                                 metadata={})


def test_a_pick_measured_under_another_jax_is_not_read(tmp_path, monkeypatch):
    """Picks persist per device, shape and JAX version (bible §6): Pallas lowers a kernel in jax itself,
    which a release moves without jaxlib, so a launch measured under another jax is read by none, kept
    in the store for its own checkout, and measured anew here."""
    monkeypatch.setattr(tuning.cache, "kernel_picks_path", lambda kind: str(tmp_path / f"{kind}.json"))
    ba, derived, other = _norm_launch_and_two_configs()
    with monkeypatch.context() as patched:
        patched.setattr(jax, "__version__", "0.0.0.other")
        tuning._store("gpu", [(ba, tuning.Timings({derived: _timed(2.0), other: _timed(1.0)}))])
    read, foreign = tuning._read(str(tmp_path / "gpu.json"))
    assert (read, len(foreign)) == ({}, 1)
    tuning._store("gpu", [(ba, tuning.Timings({derived: _timed(2.0), other: _timed(1.0)}))])
    read, foreign = tuning._read(str(tmp_path / "gpu.json"))
    assert len(read) == 1 and len(foreign) == 1


def test_a_launch_whose_derived_config_lost_names_its_gap(tmp_path, monkeypatch):
    """The tuner's picks feed a maintainer report (bible §6): each stored launch whose fastest config is
    not its derived one names the launch, its derived config and the pick, and their times; a launch
    whose derived config won names none."""
    monkeypatch.setattr(tuning.cache, "kernel_picks_path", lambda kind: str(tmp_path / f"{kind}.json"))
    ba, derived, other = _norm_launch_and_two_configs()
    tuning._store("gpu", [(ba, tuning.Timings({derived: _timed(3.0), other: _timed(2.0)}))])
    [gap] = tuning.gaps("gpu")
    assert (gap.derived, gap.picked, gap.derived_ms, gap.picked_ms) == (derived, other, 3.0, 2.0)
    assert gap.gain == 1.5 and gap.key == str(dict(ba.autotuning_cache_key))
    tuning._store("gpu", [(ba, tuning.Timings({derived: _timed(1.0), other: _timed(2.0)}))])
    assert tuning.gaps("gpu") == []


def test_a_timing_mapping_takes_its_fastest_run_and_drops_its_failures():
    def timed(ms):
        return tokamax.BenchmarkData(compile_time_ms=0.0, lower_time_ms=0.0, evaluation_times_ms=(ms,),
                                     metadata={})

    fast, slow = timed(1.0), timed(2.0)
    timings = tuning.Timings({"a": slow, "b": fast, "c": ValueError("past the GPU's shared memory")})
    assert timings.fastest_config == "b"
    assert dict(timings.prune_errors()) == {"a": slow, "b": fast}
    assert dict(timings.prune()) == {"b": fast}


def test_a_launch_of_numpy_descriptors_keys_and_round_trips_as_jaxs():
    """The gated delta rule unpacked passes its chunk and span descriptors as NumPy arrays; each launch,
    recorded as the tuner records it while the rule traces (`tuning.seen`), keys them as shapes and
    dtypes (`KernelOp.keyed`), as it does JAX's arrays, so its launches round-trip through Dew's JSON
    as every other's."""
    import jax.numpy as jnp

    from dew.nn.kernels import fla_gdn

    q = jax.ShapeDtypeStruct((1, 64, 1, 64), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((1, 64, 2, 64), jnp.bfloat16)
    g = jax.ShapeDtypeStruct((1, 64, 2), jnp.float32)
    beta = jax.ShapeDtypeStruct((1, 64, 2), jnp.bfloat16)
    h0 = jax.ShapeDtypeStruct((1, 2, 64, 64), jnp.float32)
    with tuning.seen() as found:
        jax.make_jaxpr(lambda *args: fla_gdn.chunk_gated_delta_rule(*args, None, 0.125, 64,
                                                                    interpret=True))(q, q, v, g, beta, h0)
    # The rule's launches as `KernelOp.keyed` gives them from the NumPy descriptors (`fla_gdn.sequences`),
    # which JSON has no form for unless they key as shapes.
    keyed = [ba for _, ba in found]
    assert keyed
    again = tuning.loads_launches(tuning.dumps_launches(keyed))
    assert [(ba.op, ba.autotuning_cache_key) for ba in again] == [
        (tuning._identity(ba.op), ba.autotuning_cache_key) for ba in keyed]


def test_a_stored_class_this_checkout_lacks_is_unreadable_and_an_imports_own_failure_propagates(
        tmp_path, monkeypatch):
    """A class name whose module, outer class or class this checkout lacks is `Unreadable`, which the
    store keeps aside as another checkout's entry; a module that fails as it imports, here on a
    dependency it lacks, is this checkout's error, and propagates."""
    import pytest

    for name in ("dew.nn.kernels.removed_module:Kernel", "dew.nn.kernels.tuning:Removed",
                 "dew.nn.kernels.tuning:RemovedNamespace.Config", "dew.nn.kernels.tuning:Timings.Removed"):
        with pytest.raises(tuning.Unreadable):
            tuning._located(name)
    assert tuning._located("dew.nn.kernels.tuning:Timings") is tuning.Timings
    (tmp_path / "kernel_needing_a_dependency.py").write_text("import a_dependency_nobody_installed\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    with pytest.raises(ModuleNotFoundError) as failed:
        tuning._located("kernel_needing_a_dependency:Kernel")
    assert not isinstance(failed.value, tuning.Unreadable)


@dataclasses.dataclass(frozen=True)
class _Hooks:
    """A gemm-like launch's hooks: a function, and axes that decide its side operands."""

    epilogue: object
    axes: tuple[int, ...]


TILE = jax.ShapeDtypeStruct((8, 16), jnp.float32)


@dataclasses.dataclass(frozen=True)
class _HookedKernel(tuning.KernelOp):
    """A launch whose op holds a traced function, and whose argument `hooks` holds another, each traced
    at its tile for its picks' key."""

    HOOKS = ("prologue", "hooks")
    name = "hooked"
    config_cls = int
    prologue: object = None

    def _fwd(self, x, hooks, *, config, return_residuals=False):
        return x, None

    def _get_heuristics_config(self, ba):
        return 1

    def _get_autotuning_configs(self, ba):
        return {1, 2}

    def traced(self, name, value):
        if value is None or isinstance(value, str):
            return value
        if name == "prologue":
            return tuning.program(value, TILE)
        return dataclasses.replace(value, epilogue=self.traced("prologue", value.epilogue))


def test_a_hooked_launch_keys_by_what_its_hooks_compute_and_round_trips():
    """A template's traced functions, closures among them, key its picks by what they compute (their
    traced programs): two closures over equal values key alike and store and read back as the same
    launch; a closure over another value, another computation of one layout (a sum against a product,
    as two fused epilogues differ), hooks whose other fields differ (axes), and a launch without the
    function key apart."""
    def closure(scale):
        return lambda tile: tile * scale

    def keyed(prologue, epilogue, axes):
        op = _HookedKernel(prologue=prologue)
        return op.keyed(op.bind(TILE, _Hooks(epilogue, axes)))

    def key(launch):
        return tuning._identity(launch.op), launch.autotuning_cache_key

    first, again_alike = keyed(closure(2.0), closure(3.0), (0,)), keyed(closure(2.0), closure(3.0), (0,))
    assert key(first) == key(again_alike)
    [again] = tuning.loads_launches(tuning.dumps_launches([first]))
    assert (again.op, again.autotuning_cache_key) == key(first)
    added = functools.partial(lambda combine, tile: combine(tile, tile), jnp.add)
    multiplied = functools.partial(lambda combine, tile: combine(tile, tile), jnp.multiply)
    assert key(keyed(closure(2.0), added, (0,))) != key(keyed(closure(2.0), multiplied, (0,)))
    for other in (keyed(closure(5.0), closure(3.0), (0,)), keyed(closure(2.0), closure(7.0), (0,)),
                  keyed(closure(2.0), closure(3.0), (1,)), keyed(None, closure(3.0), (0,))):
        assert key(other) != key(first)


def test_a_hook_keys_by_the_values_of_the_arrays_it_closes_over():
    """Two hooks of one code closing over different same-shaped arrays (a flag choosing a branch), at
    the hook's top or inside a jitted helper it calls, key apart; over equal arrays, alike."""
    def hook(flag):
        return lambda tile: jax.lax.cond(flag[0], jnp.tanh, lambda x: x, tile)

    def through_a_jitted_helper(flag):
        inner = jax.jit(hook(flag))
        return lambda tile: inner(tile) + 1

    for factory in (hook, through_a_jitted_helper):
        keys = [tuning.program(factory(jnp.asarray([value])), TILE) for value in (True, False, True)]
        assert keys[0] != keys[1]
        assert keys[0] == keys[2]


def _times(scale, tile):
    """A module-level hook body: the tile times `scale`."""
    return tile * scale


@dataclasses.dataclass(frozen=True)
class _Times:
    """A module-level hook class: the tile times its scale, or plus it."""

    scale: object
    added: bool = False

    def __call__(self, tile):
        return tile + self.scale if self.added else tile * self.scale


def _hooks() -> list:
    """Hooks of every kind a launch holds: module-level functions, dataclasses of one and partials of them
    over numbers, arrays, lists and tuples and in either keyword order, closures over each, callable objects
    equal whatever they compute, hooks whose state is no field, and partials over random keys."""
    row = np.arange(16, dtype=np.float32)
    scales = (2.0, 3.0, 2, row, row[::-1].copy(), [2.0], (2.0,))

    def closure(scale):
        return lambda tile: tile * (scale[0] if isinstance(scale, (list, tuple)) else scale)

    def branching(scale):
        return lambda tile: tile * (2 if isinstance(scale, list) else 3)

    def first_keyword(tile, **settings):
        return tile * next(iter(settings.values()))

    def keyed(tile, key):
        return tile + jax.random.key_data(key)[0].astype(tile.dtype)

    return [_scaled_hook, *(_Times(scale) for scale in scales if not isinstance(scale, (list, tuple))),
            _Times(2.0, added=True), *(functools.partial(_times, scale) for scale in scales[:5]),
            *(closure(scale) for scale in scales), branching([1]), branching((1,)),
            functools.partial(jnp.add, 1.0), functools.partial(jnp.multiply, 2.0),
            functools.partial(first_keyword, a=2.0, b=3.0), functools.partial(first_keyword, b=3.0, a=2.0),
            _Initialized(2.0), _Initialized(3.0), _Equal(2.0), _Equal(3.0),
            functools.partial(keyed, key=jax.random.key(0)), functools.partial(keyed, key=jax.random.key(1))]


def test_two_hooks_share_an_identity_only_where_they_lower_to_one_program():
    """The identity's invariant (`tuning.KernelOp.identified`): two hooks of one launch's identity lower to
    one program, the same equations and constants, over every pair of hooks of every kind (`_hooks`)."""
    op = _HookedKernel()
    tile = jnp.ones((8, 16), jnp.float32)
    hooks = _hooks()
    keys = [op.identified("prologue", hook) for hook in hooks]

    def lowered(hook):
        closed = jax.make_jaxpr(lambda value: hook(value))(tile)
        return str(closed.jaxpr), [np.asarray(constant).tobytes() for constant in closed.consts]

    for (first, one), (second, other) in itertools.combinations(zip(keys, hooks, strict=True), 2):
        if first == second:
            assert lowered(one) == lowered(other), f"{one} and {other} share {first}"
            assert_computes_the_oracle(one(tile), other(tile), f"{one} and {other}", roundings=0)


TRACES: list[str] = []
"""Each field `_CountingKernel` traced, in order."""


@dataclasses.dataclass(frozen=True)
class _CountingKernel(_HookedKernel):
    """`_HookedKernel`, each trace of a hook field counted (`TRACES`): JAX's own tracing cache serves a
    function's second trace without calling it, so the hook's calls do not count the identity's traces."""

    def traced(self, name, value):
        TRACES.append(name)
        return super().traced(name, value)


def test_a_hook_is_traced_once_while_it_lives_under_one_configuration():
    """A launch's identity traces a hook once for its object while it lives (a model's layers share it), at
    the op's layouts and the trace's configuration: twice asked, traced once; another object of one code,
    apart; a bundle rebuilt around the same callable (a bind's), traced once for it; arbitrary callable
    dataclasses, once an object; under another random bits' construction, apart where its program differs. The
    whole identity, its kernels' names (`report.launching`), reads it alike."""
    def scaling(scale):
        def hook(tile):
            return tile * scale
        return hook

    TRACES.clear()
    op, shared = _CountingKernel(), scaling(2.0)
    first = op.identified("prologue", shared)
    assert first == op.identified("prologue", shared) and TRACES == ["prologue"]
    assert op.identified("prologue", scaling(3.0)) != first
    assert op.identified("hooks", _Hooks(shared, (0,))) == op.identified("hooks", _Hooks(shared, (0,)))
    # The bundle's trace traces its function within it.
    assert TRACES == ["prologue", "prologue", "hooks", "prologue"]
    TRACES.clear()
    assert op.identified("prologue", _Times(2.0)) == op.identified("prologue", _Times(2.0))
    assert op.identified("prologue", _Times(2.0, added=True)) != op.identified("prologue", _Times(2.0))
    assert op.identified("prologue", _Initialized(2.0)) == op.identified("prologue", _Initialized(2.0))
    assert TRACES == ["prologue"] * 6
    hooked = _HookedKernel(prologue=shared)
    whole = report._identity(hooked, hooked.bind(TILE, _Hooks(shared, (0,))))
    assert whole == report._identity(hooked, hooked.bind(TILE, _Hooks(shared, (0,))))
    assert whole != report._identity(hooked, hooked.bind(TILE, _Hooks(scaling(4.0), (0,))))

    def drawing(tile):
        return tile + jax.random.bits(jax.random.key(0), tile.shape).astype(tile.dtype)

    with jax.threefry_partitionable(new_val=True):
        partitionable = op.identified("prologue", drawing)
    with jax.threefry_partitionable(new_val=False):
        assert op.identified("prologue", drawing) != partitionable


@pytest.mark.parametrize("biased", (False, True))
def test_rebuilt_softcap_and_bias_hooks_share_their_analysis(biased):
    from dew.nn.kernels.templates.attention_mosaic import MosaicForward

    calls = []

    def cap():
        hook = attention.Capped(2.0)
        return attention.Biased(hook) if biased else hook

    @dataclasses.dataclass(frozen=True)
    class Counted(MosaicForward):
        score_kinds: tuple[attention.Kind, ...] = (attention.Kind.KEY,) if biased else ()
        score_dtypes: tuple[str, ...] = ("float32",) if biased else ()

        def traced(self, name, value):
            calls.append(name)
            return super().traced(name, value)

    op = Counted()
    keys = [op.identified("score_mod", cap()) for _ in range(3)]
    assert keys[0] == keys[1] == keys[2]
    assert calls == ["score_mod"]
    changed = attention.Capped(3.0)
    changed = attention.Biased(changed) if biased else changed
    assert op.identified("score_mod", changed) != keys[0]


@pytest.mark.parametrize("forms", (("fast", "accurate"), ("accurate", "fast")))
def test_a_retained_score_hook_keys_each_run_precisions_program(forms):
    from dew.nn.kernels.templates.attention_mosaic import MosaicForward, Tiles

    def contextual(scores, queries, keys):
        del queries, keys
        return scores * (2.0 if generation.numerics().forms == "accurate" else 3.0)

    op = MosaicForward(score_mod=contextual)
    q = jax.ShapeDtypeStruct((1, 16, 1, 16), jnp.bfloat16)
    starts = jax.ShapeDtypeStruct((1, 16), jnp.int32)
    bound = op.bind(q, q, q, starts, starts, None, None, (), (), None,
                    residuals=attention.Kept.NONE, interpret=True)
    identities, programs, names = [], [], []
    for form in forms:
        with generation.run_precision(Numerics(forms=form)):
            direct = op.traced("score_mod", contextual)
            held = op.identified("score_mod", contextual)
            assert held == direct
            programs.append(held)
            identities.append(report._identity(op, bound))
            with report.launching(op, bound, Tiles(16, 16, 2, 1)):
                names.append(report.named("attention"))
            value = contextual(jnp.ones((1, 1)), None, None)
            expected = np.full((1, 1), 2.0 if form == "accurate" else 3.0)
            assert_computes_the_oracle(value, expected, form, roundings=0)
    assert programs[0] != programs[1]
    assert identities[0] != identities[1]
    assert names[0] != names[1]


@dataclasses.dataclass(frozen=True, eq=False)
class _Initialized:
    """A hook class whose scale is no field of it (`InitVar`): no description of its fields says what its
    call reads."""

    scale: dataclasses.InitVar[float]

    def __post_init__(self, scale):
        object.__setattr__(self, "_scale", scale)

    def __call__(self, tile):
        return tile * self._scale


class _Equal:
    """A callable object equal to every other of its class and hashing alike, whatever its scale."""

    def __init__(self, scale):
        self.scale = scale

    def __eq__(self, other):
        return isinstance(other, _Equal)

    def __hash__(self):
        return 0

    def __call__(self, tile):
        return tile * self.scale


class _ScaleSlot:
    __slots__ = ("scale",)


@dataclasses.dataclass(frozen=True, slots=True, init=False, eq=False)
class _SlottedScale(_ScaleSlot):
    def __init__(self, scale):
        object.__setattr__(self, "scale", scale)

    def __call__(self, tile):
        return tile * self.scale


@pytest.mark.parametrize("factory", (_Equal, _SlottedScale, _Initialized))
def test_hooks_with_distinct_outputs_have_distinct_program_and_launch_keys(factory):
    hooks = [factory(scale) for scale in (2.0, 3.0)]
    tile = jnp.ones(TILE.shape, TILE.dtype)
    for scale, hook in zip((2.0, 3.0), hooks, strict=True):
        assert_computes_the_oracle(hook(tile), np.full(TILE.shape, scale), "scaled tile", roundings=0)

    op = _HookedKernel()
    direct = [tuning.program(hook, TILE) for hook in hooks]
    held = [op.identified("prologue", hook) for hook in hooks]
    identities, names = [], []
    for hook in hooks:
        launch = _HookedKernel(prologue=hook)
        bound = launch.bind(TILE, _Hooks(None, ()))
        identities.append(report._identity(launch, bound))
        with report.launching(launch, bound, 1):
            names.append(report.named("hooked"))
    assert direct[0] != direct[1]
    assert held == direct
    assert identities[0] != identities[1]
    assert names[0] != names[1]


@dataclasses.dataclass(frozen=True, eq=False)
class _CyclicScale:
    scale: float
    context: list

    def __call__(self, tile):
        return tile * self.scale


@dataclasses.dataclass(frozen=True, eq=False)
class _CyclicBundle:
    epilogue: object
    context: list


@pytest.mark.parametrize("bundled", (False, True))
def test_a_cyclic_hook_graph_names_its_finite_program(bundled):
    context = []
    hook = _CyclicScale(2.0, context)
    graph = _CyclicBundle(hook, context) if bundled else hook
    context.append(graph)
    op = _HookedKernel()
    direct = tuning.program(lambda tile: hook(tile), TILE)
    held = op.identified("hooks" if bundled else "prologue", graph)
    assert (held.epilogue if bundled else held) == direct
    assert_computes_the_oracle(hook(jnp.ones(TILE.shape, TILE.dtype)),
                              np.full(TILE.shape, 2.0), "cyclic hook", roundings=0)
    launch = _HookedKernel(prologue=None if bundled else hook)
    bound = launch.bind(TILE, graph if bundled else _Hooks(None, ()))
    with report.launching(launch, bound, 1):
        assert report.named("hooked") is not None


def _scaled_hook(tile):
    """A module-level hook, as a template's prologue is."""
    return tile * 2


_shadowed_hook = json.dumps
"""Another module's function under this module's name: no hook of this module."""


class _HookHolder:
    """A function a class holds: its qualified name, dotted, is no module attribute's."""

    @staticmethod
    def hook(tile):
        return tile


def test_a_module_level_hook_round_trips_and_any_other_callable_is_refused():
    """A template launch's hook enters its stored key by its module and name and comes back as that very
    function; a lambda, a closure and a local function have no name to store; a stored name that a
    module attribute, another module's function or nothing answers to is unreadable."""
    stored = json.loads(json.dumps(tuning.encoded({"prologue": _scaled_hook})))
    assert tuning.decoded(stored) == {"prologue": _scaled_hook}
    assert tuning.decoded({"function": "json:dumps"}) is json.dumps

    scale = 3.0

    def local(tile):
        return tile * scale

    for hook in (lambda tile: tile, local, _HookHolder.hook):
        with pytest.raises(TypeError):
            tuning.encoded(hook)
    for name in ("json", "_shadowed_hook", "_no_such_hook"):
        with pytest.raises(tuning.Unreadable):
            tuning.decoded({"function": f"{__name__}:{name}"})


def test_a_programs_key_holds_what_its_equations_evaluate_under():
    """A key's bits drawn under Threefry's partitionable construction and under its default are different
    streams of one equation (its context, `JaxprEqnContext`): `tuning.program` keys them apart."""
    key = jax.ShapeDtypeStruct((), jax.random.key(0).dtype)

    def bits(seed):
        return jax.random.bits(seed, shape=(4,))

    with jax.threefry_partitionable(new_val=True):
        partitionable = tuning.program(bits, key)
    with jax.threefry_partitionable(new_val=False):
        default = tuning.program(bits, key)
    assert partitionable != default


UNORDERED = """
from dew.algebra.terms import parameter_key
print(repr(parameter_key(frozenset({"data", "model", "expert", "pipeline"}))))
"""
"""A parameter JAX holds as a set of axis names (`shard_map`'s manual axes), as `program` reads it
(`dew.algebra.terms.parameter_key`)."""


CALLED_BACK = """
import functools
import jax
import jax.numpy as jnp
import numpy as np
from dew.nn.kernels import tuning

def scaled(x, factor):
    return np.asarray(x) * factor

def hook(x):
    shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
    return jax.pure_callback(functools.partial(scaled, factor=2.0), shape, x)

print(tuning.program(hook, jax.ShapeDtypeStruct((4,), jnp.float32)))
"""
"""A launch's hook calling back into a module's function with an argument bound (`functools.partial`), as
`program` keys it."""


def test_a_programs_key_reads_a_callback_alike_in_every_process():
    """A hook calling back into a function with an argument bound is keyed alike in every process, whose
    functions sit at other addresses: every process keys the launch alike and a pick persists across runs."""
    read = {subprocess.run([sys.executable, "-c", CALLED_BACK], capture_output=True, text=True, check=True,
                           env={**os.environ, "JAX_PLATFORMS": "cpu"}).stdout
            for _ in range(2)}
    assert len(read) == 1


def test_a_programs_key_reads_a_set_alike_in_every_process():
    """A set of axis names reads alike in processes of different hash seeds, whose sets iterate in different
    orders: every process keys a launch alike, so each takes process 0's picks (`_settle`)."""
    read = {subprocess.run([sys.executable, "-c", UNORDERED], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": seed, "JAX_PLATFORMS": "cpu"}).stdout
            for seed in ("1", "2", "3", "4")}
    assert len(read) == 1
