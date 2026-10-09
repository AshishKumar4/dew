"""Jupyter Kernel Gateway owns contexts; this launcher owns their Linux limits.

Every kernel has a unique non-root uid, no capabilities, a read-only root and
small private tmpfs scratch mounts. Models and compiled programs belong to the
separate serving uid and are never loaded into these kernel processes.
"""

import json
import os
import queue
import shlex
import subprocess
import tempfile
from pathlib import Path

from kernel_gateway.services.kernels.manager import (
    KernelGatewayIOLoopKernelManager,
    SeedingMappingKernelManager,
)
from traitlets import default

_uids = queue.SimpleQueue()
for _uid in range(6100, 6200):
    _uids.put(_uid)


def _private_tmpfs(directory, uid, size):
    """Mount a tmpfs of `size` on `directory` that only `uid` may use."""
    subprocess.run(['mount', '-t', 'tmpfs', '-o', f'size={size},uid={uid},gid={uid},mode=0700,nosuid,nodev',
                    'tmpfs', str(directory)], check=True)


class LimitedKernelManager(KernelGatewayIOLoopKernelManager):
    @default("transport")
    def _transport_default(self):
        return "ipc"

    @default("ip")
    def _ip_default(self):
        if not self.connection_file:
            raise ValueError("the private connection file must be assigned before IPC startup")
        return str(Path('/sessions/ipc') / Path(self.connection_file).stem / 'kernel')

    @property
    def overlay(self):
        """Where a training context's writes to /opt/train go: a small tmpfs of its own."""
        return Path('/sessions/overlay') / Path(self.connection_file).stem

    def cleanup_ipc_files(self):
        super().cleanup_ipc_files()
        for directory in (Path(self.ip).parent, self.overlay):
            if directory.is_mount():
                subprocess.run(['umount', str(directory)], check=True)
            if directory.exists():
                directory.rmdir()

    def cleanup_connection_file(self):
        super().cleanup_connection_file()
        if self.connection_file:
            connection = Path('/sessions/connections') / Path(self.connection_file).name
            connection.unlink(missing_ok=True)
        uid = getattr(self, "guest_uid", None)
        if uid is not None:
            _uids.put(uid)
            self.guest_uid = None

    async def _async_launch_kernel(self, kernel_cmd, **kwargs):
        try:
            uid = _uids.get_nowait()
        except queue.Empty:
            raise RuntimeError("the shared container has no free isolated kernel uid") from None
        self.guest_uid = uid
        if len(kernel_cmd) < 3 or kernel_cmd[1:3] != ["-m", "ipykernel_launcher"]:
            raise ValueError("only the Python demo kernel may be started")
        connection = Path('/sessions/connections') / Path(self.connection_file).name
        connection.parent.mkdir(mode=0o711, exist_ok=True)
        data = json.loads(Path(self.connection_file).read_text())
        data['ip'] = '/work/ipc/kernel'
        descriptor, temporary = tempfile.mkstemp(dir=connection.parent)
        try:
            with os.fdopen(descriptor, 'w') as stream:
                json.dump(data, stream)
                os.fchown(stream.fileno(), uid, uid)
                os.fchmod(stream.fileno(), 0o400)
            os.replace(temporary, connection)
        finally:
            Path(temporary).unlink(missing_ok=True)
        directory = Path(self.ip).parent
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chown(directory, uid, uid)
        _private_tmpfs(directory, uid, "1m")
        if self.kernel_name == "dew-train":
            # A training context reads the prepared cells' inputs (prepare-cells.py) through an
            # overlay of its own: the Hub and datasets libraries write locks and records beside
            # what they read, and no context may change what the next one reads. Its writes land
            # in this tmpfs, which bounds them: the page's cells write only empty locks.
            self.overlay.mkdir(mode=0o700, exist_ok=True)
            _private_tmpfs(self.overlay, uid, "64m")
            for name in ("upper", "work"):
                (self.overlay / name).mkdir(mode=0o700)
                os.chown(self.overlay / name, uid, uid)
        arguments = ["/kernel.json" if arg == self.connection_file else arg for arg in kernel_cmd[3:]]
        command = [
            "bwrap", "--unshare-user", "--die-with-parent", "--new-session",
            "--ro-bind", "/", "/", "--dev", "/dev",
            "--size", str(64 * 1024 * 1024), "--tmpfs", "/work",
            "--size", str(16 * 1024 * 1024), "--tmpfs", "/tmp",
            "--ro-bind", str(connection), "/kernel.json",
            "--bind", str(directory), "/work/ipc",
            "--ro-bind", "/run/dew/model/model.sock", "/work/model.sock",
            *(["--overlay-src", "/opt/train", "--overlay", str(self.overlay / "upper"),
               str(self.overlay / "work"), "/opt/train"]
              if self.kernel_name == "dew-train" else []),
            "--chdir", "/work", "--cap-drop", "ALL",
            "/opt/venv/bin/python", "/opt/live/guest_entry.py", *arguments,
        ]
        limited = ["/usr/sbin/capsh", "--drop=all", "--no-new-privs", f"--user=ctx{uid - 6100}",
                   "--", "-c", "exec " + shlex.join(command)]
        env = {
            "PATH": "/opt/venv/bin:/usr/bin:/bin", "HOME": "/work", "TMPDIR": "/tmp",
            "LANG": "C.UTF-8", "PYTHONPATH": "/opt/live", "IPYTHONDIR": "/work/ipython",
            "JUPYTER_RUNTIME_DIR": "/work/jupyter",
            "HF_HOME": "/work/hf", "HF_HUB_OFFLINE": "1", "JAX_PLATFORMS": "cpu",
            "JAX_COMPILATION_CACHE_DIR": "/work/xla", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
            "MALLOC_ARENA_MAX": "2",
            **({"DEW_GUEST_TRACE": "1"} if os.environ.get("DEW_GUEST_TRACE") == "1" else {}),
        }
        if self.kernel_name == "dew-train":
            # Dew runs in the context, reading the model and the corpus prepared under
            # /opt/train (warm-managed.py); its writes there land in its own overlay.
            env.update({"DEW_GUEST_PROFILE": "train", "HF_HOME": "/opt/train/hf",
                        "XDG_CACHE_HOME": "/opt/train/cache", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                        "TRANSFORMERS_VERBOSITY": "error", "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1"})
        # A privileged write sets the OOM score's floor as well as the score, so the guest
        # cannot lower itself back to the gateway's -1000 (start-shared.sh) and make a trusted
        # process the OOM killer's choice; the kernel then picks the largest guest.
        isolated = ["sh", "-c", 'echo 1000 > /proc/self/oom_score_adj && exec "$@"', "oom",
                    "unshare", "--mount", "--pid", "--fork", "--kill-child", "--mount-proc",
                    "--ipc", "--uts", "--net", *limited]
        await super()._async_launch_kernel(isolated, **{**kwargs, "env": env, "cwd": "/"})


class LimitedMappingKernelManager(SeedingMappingKernelManager):
    @default("kernel_manager_class")
    def _kernel_manager_class_default(self):
        return "gateway_manager.LimitedKernelManager"
