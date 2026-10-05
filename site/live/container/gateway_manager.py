"""Jupyter Kernel Gateway owns contexts; this launcher owns their Linux limits.

Every kernel has a unique non-root uid, no capabilities, a read-only root and
small private tmpfs scratch mounts. Models and compiled programs belong to the
separate serving uid and are never loaded into these kernel processes.
"""

import itertools
import os
import shlex
import shutil
from pathlib import Path

from kernel_gateway.services.kernels.manager import (
    KernelGatewayIOLoopKernelManager,
    SeedingMappingKernelManager,
)
from traitlets import default

_uids = itertools.count(6100)


class LimitedKernelManager(KernelGatewayIOLoopKernelManager):
    def cleanup_connection_file(self):
        super().cleanup_connection_file()
        if self.connection_file:
            connection = Path('/sessions/connections') / Path(self.connection_file).name
            connection.unlink(missing_ok=True)

    async def _async_launch_kernel(self, kernel_cmd, **kwargs):
        uid = next(_uids)
        if uid >= 6200:
            raise RuntimeError("the container's kernel uid allotment is exhausted; retire this container")
        if len(kernel_cmd) < 3 or kernel_cmd[1:3] != ["-m", "ipykernel_launcher"]:
            raise ValueError("only the Python demo kernel may be started")
        connection = Path('/sessions/connections') / Path(self.connection_file).name
        connection.parent.mkdir(mode=0o711, exist_ok=True)
        shutil.copyfile(self.connection_file, connection)
        os.chown(connection, uid, uid)
        os.chmod(connection, 0o400)
        arguments = ["/kernel.json" if arg == self.connection_file else arg for arg in kernel_cmd[3:]]
        command = [
            "bwrap", "--unshare-user", "--die-with-parent", "--new-session", "--ro-bind", "/", "/",
            "--size", str(64 * 1024 * 1024), "--tmpfs", "/work",
            "--size", str(16 * 1024 * 1024), "--tmpfs", "/tmp",
            "--ro-bind", str(connection), "/kernel.json", "--chdir", "/work", "--cap-drop", "ALL",
            "/opt/venv/bin/python", "/opt/live/guest_entry.py", *arguments,
        ]
        limited = ["/usr/sbin/capsh", "--drop=all", "--no-new-privs", f"--user=ctx{uid - 6100}",
                   "--", "-c", "exec " + shlex.join(command)]
        env = {
            "PATH": "/opt/venv/bin:/usr/bin:/bin", "HOME": "/work", "TMPDIR": "/tmp",
            "LANG": "C.UTF-8", "IPYTHONDIR": "/work/ipython", "JUPYTER_RUNTIME_DIR": "/work/jupyter",
            "HF_HOME": "/opt/hf", "HF_HUB_OFFLINE": "1", "JAX_PLATFORMS": "cpu",
            "JAX_COMPILATION_CACHE_DIR": "/work/xla", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
            "MALLOC_ARENA_MAX": "2",
            **({"DEW_GUEST_TRACE": "1"} if os.environ.get("DEW_GUEST_TRACE") == "1" else {}),
        }
        await super()._async_launch_kernel(limited, **{**kwargs, "env": env, "cwd": "/"})


class LimitedMappingKernelManager(SeedingMappingKernelManager):
    @default("kernel_manager_class")
    def _kernel_manager_class_default(self):
        return "gateway_manager.LimitedKernelManager"
