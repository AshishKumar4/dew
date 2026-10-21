"""dew.nn.kernels.tuning's picks in Dew's own JSON: every launch a step traces, its op, its arguments and
its timings, written and read back equal, through tokamax's public API alone."""

import dataclasses
import functools
import json

import jax
import jax.numpy as jnp
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


@dataclasses.dataclass(frozen=True)
class _Hooks:
    """A gemm-like launch's hooks: a function, and axes that decide its side operands."""

    epilogue: object
    axes: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class _HookedKernel(tuning.KernelOp):
    """A launch whose op holds a traced function, and whose argument `hooks` holds another."""

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


def test_a_hooked_launch_keys_by_what_its_hooks_compute_and_round_trips():
    """A template's traced functions, closures among them, key its picks by what they compute: two
    closures over equal values key alike and store and read back as the same launch; a closure over
    another value, hooks whose other fields differ (axes), and a launch without the function key
    apart."""
    def closure(scale):
        return lambda tile: tile * scale

    x = jax.ShapeDtypeStruct((8, 16), jnp.float32)

    def keyed(prologue, epilogue, axes):
        op = _HookedKernel(prologue=prologue)
        return op.keyed(op.bind(x, _Hooks(epilogue, axes)))

    def key(launch):
        return tuning._identity(launch.op), launch.autotuning_cache_key

    first, again_alike = keyed(closure(2.0), closure(3.0), (0,)), keyed(closure(2.0), closure(3.0), (0,))
    assert key(first) == key(again_alike)
    [again] = tuning.loads_launches(tuning.dumps_launches([first]))
    assert (again.op, again.autotuning_cache_key) == key(first)
    for other in (keyed(closure(5.0), closure(3.0), (0,)), keyed(closure(2.0), closure(7.0), (0,)),
                  keyed(closure(2.0), closure(3.0), (1,)), keyed(None, closure(3.0), (0,))):
        assert key(other) != key(first)


def _fused(jaxpr, tile):
    """A fused consumer's stand-in: the generic code a fuser's epilogue runs, over the jaxpr it is
    partial in."""
    return tile


def test_fused_consumers_of_one_layout_key_by_the_jaxprs_they_close_over():
    """Two hooks of one code and layout that close over different jaxprs (a fused bias and a fused
    scale, as the fuser's consumers do) key apart; the same jaxpr keys alike."""
    def consumer(jaxpr):
        return functools.partial(_fused, jaxpr)

    accumulator = jax.ShapeDtypeStruct((8, 16), jnp.float32)
    bias = jax.make_jaxpr(lambda acc, b: acc + b)(accumulator, accumulator)
    scale = jax.make_jaxpr(lambda acc, b: acc * b)(accumulator, accumulator)
    digests = [tuning._traced(consumer(jaxpr)) for jaxpr in (bias, scale, bias)]
    assert digests[0] != digests[1]
    assert digests[0] == digests[2]

