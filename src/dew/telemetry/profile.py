"""Explicit process-local JAX capture and unmodified native XProf reports.

Only start/stop drain the default backend's live arrays and effects. This is
not a distributed barrier. Normal execution installs no instrumentation hooks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import importlib
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import shlex
import tempfile
import threading
from types import TracebackType
from typing import Literal, Protocol, Self, runtime_checkable

import jax


@runtime_checkable
class _Converter(Protocol):
    def xspace_to_tool_names(self, paths: Sequence[str]) -> list[str]: ...

    def xspace_to_tool_data(
        self, paths: Sequence[str], tool: str, params: Mapping[str, object]
    ) -> tuple[object, str]: ...


_active: Profiler | None = None
_lifecycle = threading.Lock()
_logger = logging.getLogger(__name__)


def require_profile_support() -> _Converter:
    """Resolve the optional native report converter before starting work."""
    try:
        converter = importlib.import_module("xprof.convert.raw_to_tool_data")
    except ImportError as error:
        raise ImportError("Profiling requires XProf; install 'dew-ml[profile]'.") from error
    if not isinstance(converter, _Converter):
        raise ImportError("Installed XProf lacks native converters; install 'dew-ml[profile]'.")
    return converter


def active_profile() -> Profiler | None:
    """The explicitly enabled process-local profiler, or None."""
    owner = _active
    return owner if owner is not None and owner._phase == "running" else None


def _drain() -> None:
    jax.block_until_ready(jax.live_arrays())
    jax.effects_barrier()


def _reports(directory: Path, converter: _Converter, metadata: dict[str, object]) -> None:
    paths = sorted(directory.glob("plugins/profile/*/*.xplane.pb"))
    failures: list[Exception] = []
    records: list[dict[str, object]] = []
    metadata["reports"] = records
    reports = directory / "reports"
    reports.mkdir()
    if not paths:
        failures.append(RuntimeError("JAX capture produced no native XPlane files"))
    sessions = sorted({path.parent for path in paths})
    for index, session in enumerate(sessions):
        hosts = [str(path) for path in paths if path.parent == session]
        try:
            tools = converter.xspace_to_tool_names(hosts)
            if not tools:
                raise RuntimeError("Native XProf tool discovery returned no tools")
        except Exception as error:
            records.append({"session": str(session.relative_to(directory)),
                            "status": "error", "error": str(error)})
            failures.append(error)
            continue
        for tool in tools:
            record: dict[str, object] = {"tool": tool, "session": str(session.relative_to(directory))}
            records.append(record)
            if tool in ("graph_viewer", "memory_viewer", "trace_viewer@", "trace_viewer"):
                record.update(status="interactive", reason="Use the native XProf viewer and retained trace/HLO files")
                continue
            if not tool or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in tool):
                error = ValueError(f"Unexpected native tool name: {tool!r}")
                record.update(status="error", error=str(error))
                failures.append(error)
                continue
            groups = [[host] for host in hosts] if tool == "memory_profile" else [hosts]
            artifacts: list[str] = []
            record["artifacts"] = artifacts
            record["status"] = "completed"
            for host_index, group in enumerate(groups):
                try:
                    payload, mime = converter.xspace_to_tool_data(group, tool, {})
                    if payload is None:
                        record.update(status="unavailable", reason="Native converter returned no data")
                        continue
                    if not isinstance(payload, (bytes, str)):
                        raise TypeError(f"Native {tool} returned {type(payload).__name__}; expected bytes or str")
                    extension = {"application/json": ".json", "text/html": ".html",
                                 "text/plain": ".txt", "application/octet-stream": ".bin"}.get(mime, ".bin")
                    suffix = (f"-session-{index}" if len(sessions) > 1 else "")
                    suffix += f"-host-{host_index}" if len(groups) > 1 else ""
                    destination = reports / (tool + suffix + extension)
                    destination.write_bytes(payload.encode("utf-8") if isinstance(payload, str) else payload)
                    artifacts.append(str(destination.relative_to(directory)))
                    record["content_type"] = mime
                except Exception as error:
                    record.update(status="error", error=str(error))
                    failures.append(error)
    metadata["artifacts"] = [str(path.relative_to(directory)) for path in sorted(directory.rglob("*")) if path.is_file()]
    metadata["export_status"] = "failed" if failures else "completed"
    (directory / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if failures:
        raise ExceptionGroup("Native profile report export failed", failures)


class Profiler:
    """One reusable explicitly enabled profiler for context or manual use.

    Each start creates a fresh capture child below directory. Without a supplied
    directory, the first start allocates a persistent temporary root. Collection
    covers this process and drains only JAX's default backend, without copying
    array values to the host. Other processes must enter their own captures.
    """

    def __init__(self, directory: str | os.PathLike[str] | None = None, *,
                 options: jax.profiler.ProfileOptions | None = None) -> None:
        self._directory = None if directory is None else Path(directory)
        self._options = options
        self._capture: Path | None = None
        self._converter: _Converter | None = None
        self._metadata: dict[str, object] = {}
        self._temporary = directory is None
        self._phase: Literal["starting", "running", "stopping"] | None = None

    @property
    def directory(self) -> Path | None:
        return self._directory

    @property
    def running(self) -> bool:
        return active_profile() is self

    def start(self) -> Self:
        global _active
        with _lifecycle:
            if _active is not None:
                raise RuntimeError("A Dew profile is already running in this process")
            _active = self
            self._phase = "starting"
        owned_start = False
        try:
            converter = require_profile_support()
            backend = jax.default_backend()
            options = self._options
            if options is None:
                options = jax.profiler.ProfileOptions()
                options.host_tracer_level = 2
                options.python_tracer_level = 0
                options.enable_hlo_proto = True
                if backend == "tpu":
                    options.advanced_configuration = {"tpu_trace_mode": "TRACE_COMPUTE_AND_SYNC"}
            _drain()
            if self._directory is None:
                self._directory = Path(tempfile.mkdtemp(prefix="dew-profile-"))
            self._directory = self._directory.expanduser().resolve()
            root = self._directory
            count, rank = jax.process_count(), jax.process_index()
            if count > 1:
                root = root / f"process-{rank}"
            root.mkdir(parents=True, exist_ok=True)
            capture = Path(tempfile.mkdtemp(prefix="capture-", dir=root))
            metadata: dict[str, object] = {
                "versions": {name: version(name) for name in ("jax", "jaxlib", "xprof")},
                "backend": backend, "devices": [str(device) for device in jax.local_devices()],
                "device_kinds": [device.device_kind for device in jax.local_devices()],
                "process_index": rank, "process_count": count,
                "scope": "process-local; default-backend drain; no distributed barrier",
                "default_matmul_precision": jax.config.jax_default_matmul_precision,
                "jax_enable_x64": jax.config.jax_enable_x64,
                "options": {"host_tracer_level": options.host_tracer_level,
                            "python_tracer_level": options.python_tracer_level,
                            "enable_hlo_proto": options.enable_hlo_proto,
                            "advanced_configuration": options.advanced_configuration},
                "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
                "view_command": "xprof --logdir=" + shlex.quote(str(capture)),
            }
            jax.profiler.start_trace(capture, profiler_options=options)
            owned_start = True
            with _lifecycle:
                self._capture, self._converter, self._metadata = capture, converter, metadata
                self._phase = "running"
        except BaseException as primary:
            try:
                if owned_start:
                    try:
                        jax.profiler.stop_trace()
                    except BaseException as cleanup:
                        primary.add_note(f"Owned profile start cleanup failed: {cleanup}")
            finally:
                with _lifecycle:
                    _active = None
                    self._phase = None
                    self._capture = None
                    self._converter = None
                    self._metadata = {}
            raise
        return self

    def stop(self) -> None:
        self._stop("completed")

    def _stop(self, body: str) -> None:
        global _active
        failures: list[BaseException] = []
        with _lifecycle:
            if _active is not self or self._phase != "running":
                raise RuntimeError("This profiler does not own a running capture")
            self._phase = "stopping"
            capture, converter, metadata = self._capture, self._converter, self._metadata
        try:
            try:
                _drain()
            except BaseException as error:
                failures.append(error)
            try:
                jax.profiler.stop_trace()
            except BaseException as error:
                failures.append(error)
        finally:
            with _lifecycle:
                _active = None
                self._phase = None
                self._capture = None
                self._converter = None
                self._metadata = {}
        assert capture is not None and converter is not None
        metadata["body_status"] = body
        metadata["capture_status"] = "failed" if failures else "completed"
        metadata["capture_errors"] = [str(error) for error in failures]
        try:
            _reports(capture, converter, metadata)
        except BaseException as error:
            failures.append(error)
        if self._temporary:
            _logger.info("Profile saved under %s", self._directory)
        if failures:
            raise BaseExceptionGroup("Profile cleanup or export failed", failures)

    def __enter__(self) -> Self:
        return self.start()

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, traceback: TracebackType | None) -> None:
        if self.running:
            try:
                self._stop("failed" if exc is not None else "completed")
            except BaseException as error:
                if exc is None:
                    raise
                exc.add_note(f"Profile cleanup/report failure: {error}")


def profile(directory: str | os.PathLike[str] | None = None, *,
            options: jax.profiler.ProfileOptions | None = None) -> Profiler:
    """Configure native profiling; capture starts only on enter or start()."""
    return Profiler(directory, options=options)
