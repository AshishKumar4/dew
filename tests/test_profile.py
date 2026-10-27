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
from dew.telemetry.profile import active_profile, program_kernels


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
    assert profiler.directory == tmp_path and not profiler.running and profiler.captures == ()
    assert captures(tmp_path) == []
    assert profiler.start() is profiler
    with jax.profiler.TraceAnnotation("manual_capture_marker"):
        result = work()
    profiler.stop()
    first = captures(tmp_path)[0]
    assert profiler.captures == (first,)
    before = {path: path.read_bytes() for path in first.rglob("*") if path.is_file()}
    with profiler as entered:
        assert entered is profiler and active_profile() is profiler
        with jax.profiler.TraceAnnotation("context_capture_marker"):
            result = work()
    assert not profiler.running and active_profile() is None
    assert len(captures(tmp_path)) == 2
    assert profiler.captures[0] == first and set(profiler.captures) == set(captures(tmp_path))
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
        with pytest.raises(RuntimeError), dew.Profiler(tmp_path / "nested"):
            pytest.fail("nested capture must not run its body")
        other = dew.Profiler(tmp_path / "concurrent")
        with ThreadPoolExecutor(max_workers=1) as pool, pytest.raises(RuntimeError):
            pool.submit(other.start).result()
        with pytest.raises(RuntimeError):
            other.stop()
        assert outer.running
        with jax.profiler.TraceAnnotation("outer_survived"):
            work()
    assert "outer_survived" in event_names(captures(tmp_path / "outer")[0])


def test_external_jax_capture_survives_failed_dew_start(tmp_path, native_reports):
    native = tmp_path / "native"
    jax.profiler.start_trace(native)
    try:
        with pytest.raises(RuntimeError):
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
def test_a_capture_drains_past_a_live_array_with_a_deleted_buffer(tmp_path, native_reports):
    """An array can list as live with one of its buffers deleted, and waiting on it raises: one whose
    shard's buffer its holder deleted (on jax 0.11.2, too, one a donation consumed but a host read kept
    listed). The capture's drain waits on the work in flight without waiting on such an array."""
    from jax.sharding import NamedSharding, PartitionSpec as P

    mesh = jax.make_mesh((jax.device_count(),), ("data",))
    held = jax.device_put(jnp.asarray(0, jnp.int32), NamedSharding(mesh, P()))
    held.addressable_shards[1].data.delete()
    assert any(array is held for array in jax.live_arrays())
    with pytest.raises(jax.errors.JaxRuntimeError):
        jax.block_until_ready(held)
    stepped = jax.jit(lambda value: value + 1)(jnp.asarray(1, jnp.int32))
    with dew.Profiler(tmp_path):
        stepped = stepped + 1
    assert stepped.is_ready() and int(stepped) == 3


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
    with pytest.raises(ExceptionGroup), dew.Profiler(tmp_path / "export-failed"):
        work()
    assert active_profile() is None
    monkeypatch.setattr(native_reports, "xspace_to_tool_data", original)
    with dew.Profiler(tmp_path / "recovered"):
        work()


def test_default_directory_is_allocated_on_start_and_survives_stop(native_reports):
    profiler = dew.Profiler()
    assert profiler.directory is None
    with pytest.raises(RuntimeError):
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


def test_a_captures_kernels_are_filed_under_the_programs_that_ran_them(tmp_path):
    """`program_kernels` files each kernel_stats row under the first named program its op name holds, the
    rest under '', longest total time first; a capture whose export wrote no kernel_stats raises naming it."""
    capture = tmp_path / "capture-x"
    (capture / "reports").mkdir(parents=True)
    rows = [("a", "jit(step)/a", 3.0), ("b", "jit(step_eval)/b", 9.0), ("c", "jit(step)/c", 5.0),
            ("d", "jit(other)/d", 1.0)]
    table = {"cols": [{"id": "kernel_name"}, {"id": "op_name"}, {"id": "total_duration_us"}],
             "rows": [{"c": [{"v": name}, {"v": op}, {"v": time}]} for name, op, time in rows]}
    (capture / "reports" / "kernel_stats.json").write_text(json.dumps([table]))
    found = program_kernels(capture, ["step_eval", "step"])
    assert [row["kernel_name"] for row in found["step_eval"]] == ["b"]
    assert [row["kernel_name"] for row in found["step"]] == ["c", "a"]
    assert [row["kernel_name"] for row in found[""]] == ["d"]
    with pytest.raises(FileNotFoundError):
        program_kernels(tmp_path / "capture-y", ["step"])


def test_a_profile_counts_overlapped_collectives_once_and_the_gaps_as_idle():
    """Two steps, each 10 ms of compute with an all-reduce over the same 10 ms, 5 ms apart, on each of two
    devices: the window runs 25 ms, 12.5 ms a step, of which 10 ms is compute, 10 ms communication, none of
    it exposed, and 2.5 ms the gap no kernel ran in: the overlap counted once (row 1853, where summed self
    times made 20 ms of device time a step and so no idle). The report's lines say the same."""
    from dew.training.profiling import ProfileReport, device_figures

    step = [("gemm_fusion_dot", 0, 10_000_000), ("ncclDevKernel_AllReduce_Sum_f32_RING_LL", 0, 10_000_000)]
    later = [(name, start + 15_000_000, end + 15_000_000) for name, start, end in step]
    figures = device_figures([step + later, later + step], steps=2)
    assert figures == {"device_window_ms": 12.5, "compute_ms": 10.0, "communication_ms": 10.0,
                       "exposed_communication_ms": 0.0, "device_idle_ms": 2.5}
    report = ProfileReport("profile", 2, 2, (2, 3), "capture", 15.0, 1.0, 3.0, **figures, flops=None,
                           mfu=None, hfu=None, tokens_per_s=None, device=None)
    # The lines carry the figures: the idle gap, and compute's share of a step, the largest.
    text = report.text()
    assert "2.50 ms" in text and "67%" in text
    assert device_figures([[], []], steps=2)["device_idle_ms"] is None


def test_a_capture_that_holds_less_than_its_run_says_why():
    """A capture's status comes from what its trace holds (row 1893): a local device with no kernel track, or
    as many GPU activity events as CUPTI's cap (within the buffer CUPTI drops whole), is a partial capture
    with its reason; a full one has none. A CPU run, with no device track to hold, is never partial."""
    from dew.telemetry.profile import held_back, missing_tracks

    assert missing_tracks({"/device:GPU:0": 19160, "/device:GPU:1": 18800}, 2, 2_097_152) == []
    [untracked] = missing_tracks({"/device:GPU:0": 7451}, 2, 2_097_152)
    assert "1 of 2" in untracked
    # The cap's boundary: within 1% of it is reached (CUPTI drops whole buffers), below that is whole.
    [capped] = missing_tracks({"/device:GPU:0": 1987}, 1, 2000)
    assert "2000" in capped and "1987" in capped
    assert missing_tracks({"/device:GPU:0": 1979}, 1, 2000) == []
    # XLA's no-limit cap (-1) drops nothing, however many events the capture holds.
    assert missing_tracks({"/device:GPU:0": 19160}, 1, -1) == []
    assert held_back(Path("absent"), {"backend": "cpu", "device_kinds": ["cpu"]}) == []


@pytest.mark.gpu(reason="CUPTI's activity cap, which a GPU capture alone has")
def test_a_gpu_window_past_cuptis_cap_reports_no_device_figure_its_kernels_cannot_establish(tmp_path):
    """A window whose kernels pass CUPTI's activity cap keeps only the events before it: its capture is
    partial, the cap its reason, and the report gives no device figure the surviving kernels cannot
    establish, its host figures kept; with no cap (-1) the capture is whole and its device figures measured
    (row 1893: a capped H100 capture reported 0% duty cycle as measured)."""
    pytest.importorskip("xprof.convert.raw_to_tool_data", reason="optional profile extra")
    import autoencoder_experiment

    from dew.cli.main import main

    # A cap of one event: a step's few launches (XLA runs its kernels as command buffers) pass it.
    for cap, partial in ((1, True), (-1, False)):
        out = tmp_path / f"cap{cap}"
        assert main(["profile", autoencoder_experiment.__file__, "--warmup", "2", "--steps", "2", "--out",
                     str(out), "--activity-cap", str(cap)]) == 0
        report = json.loads((out / "report.json").read_text())
        manifest = json.loads((Path(report["capture"]) / "manifest.json").read_text())
        assert (manifest["capture_status"] == "partial") == partial, manifest
        assert (report["device_window_ms"] is None) == partial and report["step_ms"] > 0
