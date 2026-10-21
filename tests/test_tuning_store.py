"""dew.nn.kernels.tuning's picks in Dew's own JSON: every launch a step traces, its op, its arguments and
its timings, written and read back equal, through tokamax's public API alone."""

import json

import jax
import pytest
import tokamax

from dew.nn.kernels import norm_swiglu, tuning


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


def test_a_stored_timing_keeps_every_field_tokamax_measures():
    """`tuning.TIMING_FIELDS` is every field of `tokamax.BenchmarkData`, so a field a later tokamax adds
    fails here rather than going unstored."""
    import dataclasses

    assert {field.name for field in dataclasses.fields(tokamax.BenchmarkData)} == set(tuning.TIMING_FIELDS)


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
