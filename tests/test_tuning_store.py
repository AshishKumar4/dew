"""dew.nn.kernels.tuning's picks in Dew's own JSON: every launch a step traces, its op, its arguments and
its timings, written and read back equal, through tokamax's public API alone."""

import json

import jax
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
