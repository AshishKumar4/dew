"""dew.nn.kernels.tuning's picks in Dew's own JSON: every launch a step traces, its op, its arguments and
its timings, written and read back equal, through tokamax's public API alone."""

import dataclasses
import functools
import json
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import pytest
import tokamax

from dew.nn.kernels import norm_swiglu, tuning
from dew.nn.kernels.templates import report


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


def _refusing_hook(tile):
    """A module-level hook that refuses to run: a launch's identity names it without running it."""
    raise AssertionError("a launch's identity ran its hook")


@dataclasses.dataclass(frozen=True)
class _Scaled:
    """A module-level hook class whose call refuses to run: named by its class and its fields."""

    scale: float

    def __call__(self, tile):
        raise AssertionError(f"a launch's identity ran its hook at {self.scale}")


def test_a_launchs_identity_names_its_named_hooks_without_running_them():
    """A launch's identity (`KernelOp.identified`, every launch's, tuning or not) names a module-level hook,
    a dataclass of one and a partial of them by their parts, running none: equal parts alike, another field's
    value or a list for a tuple apart. The whole identity, its kernels' names (`report.launching`), reads an
    argument's hooks alike."""
    op = _HookedKernel()
    named = op.identified("prologue", _Scaled(2.0))
    assert named == op.identified("prologue", _Scaled(2.0))
    assert str(named).startswith("described:")
    assert named != op.identified("prologue", _Scaled(3.0))
    partial = functools.partial(_Scaled, 2.0)
    assert op.identified("prologue", partial) != op.identified("prologue", functools.partial(_Scaled, 3.0))
    assert tuning.described(([1],)) != tuning.described(((1,),))
    hooked = _HookedKernel(prologue=_refusing_hook)
    whole = report._identity(hooked, hooked.bind(TILE, _Hooks(_Scaled(3.0), (0,))))
    assert whole == report._identity(hooked, hooked.bind(TILE, _Hooks(_Scaled(3.0), (0,))))
    for changed in (_Hooks(_Scaled(4.0), (0,)), _Hooks(_Scaled(3.0), (1,))):
        assert whole != report._identity(hooked, hooked.bind(TILE, changed))


def test_a_closure_hook_is_traced_once_while_it_lives():
    """A closure's parts do not say what its call reads (`tuning.described`), so a launch's identity traces
    it, once for the object while it lives (a model's layers share it): twice asked, traced once; two
    closures of one line computing apart, apart."""
    traced = []

    def counting(scale):
        def hook(tile):
            traced.append(scale)
            return tile * scale
        return hook

    op, shared = _HookedKernel(), counting(2.0)
    first = op.identified("prologue", shared)
    assert first == op.identified("prologue", shared) and traced == [2.0]
    assert op.identified("prologue", counting(3.0)) != first
    doubled, tripled = lambda tile: tile * 2, lambda tile: tile * 3
    assert op.identified("prologue", doubled) != op.identified("prologue", tripled)


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
from dew.nn.kernels import tuning
print(repr(tuning._parameter(frozenset({"data", "model", "expert", "pipeline"}))))
"""
"""A parameter JAX holds as a set of axis names (`shard_map`'s manual axes), as `program` reads it."""


def test_a_programs_key_reads_a_set_alike_in_every_process():
    """A set of axis names reads alike in processes of different hash seeds, whose sets iterate in different
    orders: every process keys a launch alike, so each takes process 0's picks (`_settle`)."""
    read = {subprocess.run([sys.executable, "-c", UNORDERED], capture_output=True, text=True, check=True,
                           env={**os.environ, "PYTHONHASHSEED": seed, "JAX_PLATFORMS": "cpu"}).stdout
            for seed in ("1", "2", "3", "4")}
    assert len(read) == 1
