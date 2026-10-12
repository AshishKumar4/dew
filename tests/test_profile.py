"""Public native profiling lifecycle and retained capture artifacts."""

import gzip
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest
from steady_state import guarded

import dew
from dew.telemetry.profile import active_profile


@pytest.fixture
def native_reports():
    return pytest.importorskip("xprof.convert.raw_to_tool_data", reason="optional profile extra")


def work():
    value = jnp.ones((32, 32), jnp.float32)
    return jax.jit(lambda x: jnp.sin(x @ x))(value)


def captures(root):
    return sorted(root.glob("capture-*"))


def manifest(capture):
    return json.loads((capture / "manifest.json").read_text())


def event_names(capture):
    names = []
    for path in capture.glob("plugins/profile/*/*.trace.json.gz"):
        with gzip.open(path, "rt") as stream:
            names.extend(event.get("name", "") for event in json.load(stream)["traceEvents"])
    return names


def test_manual_context_and_restart_preserve_each_native_capture(tmp_path, native_reports):
    sentinel = tmp_path / "existing.txt"
    sentinel.write_text("keep")
    profiler = dew.Profiler(tmp_path)
    assert profiler.directory == tmp_path and not profiler.running
    assert captures(tmp_path) == []
    assert profiler.start() is profiler
    with jax.profiler.TraceAnnotation("manual_capture_marker"):
        result = work()
    profiler.stop()
    first = captures(tmp_path)[0]
    before = {path: path.read_bytes() for path in first.rglob("*") if path.is_file()}
    with profiler as entered:
        assert entered is profiler and active_profile() is profiler
        with jax.profiler.TraceAnnotation("context_capture_marker"):
            result = work()
    assert not profiler.running and active_profile() is None
    assert len(captures(tmp_path)) == 2
    assert sentinel.read_text() == "keep"
    assert all(path.read_bytes() == content for path, content in before.items())
    assert "manual_capture_marker" in event_names(first)
    assert any("context_capture_marker" in event_names(path) for path in captures(tmp_path))
    for capture in captures(tmp_path):
        record = manifest(capture)
        assert record["body_status"] == "completed"
        assert record["capture_status"] == record["export_status"] == "completed"
        xplanes = list(capture.glob("plugins/profile/*/*.xplane.pb"))
        assert xplanes and all(path.stat().st_size for path in xplanes)
        for report in record["reports"]:
            if report["status"] == "completed":
                for relative in report["artifacts"]:
                    artifact = capture / relative
                    assert artifact.stat().st_size
                    if artifact.suffix == ".json":
                        json.loads(artifact.read_bytes())
    assert result.is_ready()


def test_body_failure_keeps_trace_and_releases_for_next_capture(tmp_path, native_reports):
    failure = ValueError("user computation failed")
    with pytest.raises(ValueError) as raised, dew.Profiler(tmp_path):
        work()
        raise failure
    assert raised.value is failure and active_profile() is None
    assert manifest(captures(tmp_path)[0])["body_status"] == "failed"
    with dew.Profiler(tmp_path):
        work()
    assert len(captures(tmp_path)) == 2


def test_nested_and_concurrent_refusals_do_not_stop_owner(tmp_path, native_reports):
    with dew.Profiler(tmp_path / "outer") as outer:
        with pytest.raises(RuntimeError, match="already running"), dew.Profiler(tmp_path / "nested"):
            pytest.fail("nested capture must not run its body")
        other = dew.Profiler(tmp_path / "concurrent")
        with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(RuntimeError, match="already running"):
            pool.submit(other.start).result()
        with pytest.raises(RuntimeError, match="does not own"):
            other.stop()
        assert outer.running
        with jax.profiler.TraceAnnotation("outer_survived"):
            work()
    assert "outer_survived" in event_names(captures(tmp_path / "outer")[0])


def test_external_jax_capture_survives_failed_dew_start(tmp_path, native_reports):
    native = tmp_path / "native"
    jax.profiler.start_trace(native)
    try:
        with pytest.raises(RuntimeError, match="already been started"):
            dew.Profiler(tmp_path / "refused").start()
        assert active_profile() is None
        with jax.profiler.TraceAnnotation("external_survived"):
            work().block_until_ready()
    finally:
        jax.profiler.stop_trace()
    assert "external_survived" in event_names(native)


def test_manual_stop_inside_context_does_not_stop_another_owner(tmp_path, native_reports):
    second = dew.Profiler(tmp_path / "second")
    with dew.Profiler(tmp_path / "first") as first:
        work()
        first.stop()
        second.start()
    try:
        assert second.running
        with jax.profiler.TraceAnnotation("second_survived"):
            work()
    finally:
        second.stop()
    assert "second_survived" in event_names(captures(tmp_path / "second")[0])


def test_capture_drains_async_arrays_and_effects_without_host_copies(tmp_path, native_reports):
    observed = []

    @jax.jit
    def compute(value):
        result = value @ value
        jax.debug.callback(lambda: observed.append("effect"), ordered=True)
        return result

    value = jnp.ones((64, 64), jnp.float32)
    compute(value).block_until_ready()
    observed.clear()
    with guarded(allow=("host_to_device",)), dew.Profiler(tmp_path):
        pending = compute(value)
    assert pending.is_ready()
    assert observed == ["effect"]


def test_capture_does_not_retain_donated_arrays(tmp_path, native_reports):
    update = jax.jit(lambda value: value + 1, donate_argnums=(0,))
    value = jnp.ones((16,), jnp.float32)
    with dew.Profiler(tmp_path):
        previous = value
        value = update(value)
    assert previous.is_deleted() and value.is_ready()


@pytest.mark.mesh(devices=2)
def test_a_capture_drains_past_an_array_a_donation_consumed(tmp_path, native_reports):
    """An array still held after a step donated it can list as live with
    only its first buffer left: XLA counts an array deleted by that buffer,
    and a host read of a replicated array keeps it from the donation (jax
    0.11.2 on CPU). The capture's drain waits on the work in flight without
    waiting on the consumed array."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = jax.make_mesh((jax.device_count(),), ("data",))
    consumed = jax.device_put(jnp.asarray(0, jnp.int32), NamedSharding(mesh, P()))
    assert int(consumed) == 0
    stepped = jax.jit(lambda value: value + 1, donate_argnums=0)(consumed)
    assert any(array is consumed for array in jax.live_arrays())
    assert consumed.addressable_shards[1].data.is_deleted()
    with dew.Profiler(tmp_path):
        stepped = stepped + 1
    assert stepped.is_ready() and int(stepped) == 2


def test_native_export_failure_keeps_body_exception_and_trace(tmp_path, native_reports, monkeypatch):
    original = native_reports.xspace_to_tool_data

    def broken(*args, **kwargs):
        raise RuntimeError("native conversion failed")

    monkeypatch.setattr(native_reports, "xspace_to_tool_data", broken)
    failure = ValueError("primary user error")
    with pytest.raises(ValueError) as raised, dew.Profiler(tmp_path / "body-failed"):
        work()
        raise failure
    assert raised.value is failure and failure.__notes__
    assert list((tmp_path / "body-failed").glob("capture-*/plugins/profile/*/*.xplane.pb"))
    assert manifest(captures(tmp_path / "body-failed")[0])["export_status"] == "failed"
    with pytest.raises(ExceptionGroup, match="cleanup or export"), dew.Profiler(tmp_path / "export-failed"):
        work()
    assert active_profile() is None
    monkeypatch.setattr(native_reports, "xspace_to_tool_data", original)
    with dew.Profiler(tmp_path / "recovered"):
        work()


def test_default_directory_is_allocated_on_start_and_survives_stop(native_reports):
    profiler = dew.Profiler()
    assert profiler.directory is None
    with pytest.raises(RuntimeError, match="does not own"):
        profiler.stop()
    try:
        with profiler:
            work()
        root = profiler.directory
        assert isinstance(root, Path) and root.is_dir()
        assert len(captures(root)) == 1
        assert manifest(captures(root)[0])["capture_status"] == "completed"
    finally:
        if profiler.directory is not None:
            shutil.rmtree(profiler.directory)


def test_empty_native_tool_discovery_is_an_export_failure(tmp_path, native_reports, monkeypatch):
    monkeypatch.setattr(native_reports, "xspace_to_tool_names", lambda paths: [])
    with pytest.raises(ExceptionGroup), dew.Profiler(tmp_path):
        work()
    capture = captures(tmp_path)[0]
    record = manifest(capture)
    assert record["export_status"] == "failed"
    assert record["reports"][0]["status"] == "error"
    assert list(capture.glob("plugins/profile/*/*.xplane.pb"))
    assert active_profile() is None
