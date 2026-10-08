"""Linux execution limits for lightweight shared-compute kernels.

The trusted launcher drops capabilities and mounts the read-only namespace.
This process then forbids creating processes (threads share its address-space
limit), changing namespaces, and inspecting another process. Limits are not a
replacement for the namespace or for the supervisor's wall-clock deadline.
"""

import ctypes
import errno
import resource
from typing import ClassVar

GIB = 1024 * 1024 * 1024
INFINITY = resource.RLIM_INFINITY
# A page cell's context only sends requests to the model process, in a small
# address space. A training context runs Dew itself, and XLA reserves far more
# than it touches: one step of the page's train.py asks for a 6.2 GB buffer on
# these hosts and keeps 1.6 GiB resident. Its memory is bounded where it is
# used instead: the bridge kills the largest guest when the host runs short
# (shared_bridge.py), and the OOM killer takes guests first (gateway_manager.py).
PROFILES = {
    "cell": {"address_space": GIB * 3 // 4, "data": INFINITY, "cpu_seconds": 10, "processes": 32,
             "files": 128},
    "train": {"address_space": INFINITY, "data": INFINITY, "cpu_seconds": 600, "processes": 256,
              "files": 1024},
}


class Comparison(ctypes.Structure):
    _fields_: ClassVar = [("arg", ctypes.c_uint), ("op", ctypes.c_int),
                ("first", ctypes.c_uint64), ("second", ctypes.c_uint64)]


def install(profile="cell"):
    limits = PROFILES[profile]
    resource.setrlimit(resource.RLIMIT_AS, (limits["address_space"], limits["address_space"]))
    resource.setrlimit(resource.RLIMIT_DATA, (limits["data"], limits["data"]))
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
    # Stock ipykernel creates more than 64 descriptors while starting its shell channels.
    resource.setrlimit(resource.RLIMIT_NOFILE, (limits["files"], limits["files"]))
    resource.setrlimit(resource.RLIMIT_NPROC, (limits["processes"], limits["processes"]))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_CPU, (limits["cpu_seconds"], limits["cpu_seconds"] + 1))
    lib = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_rule_add_array.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int,
                                         ctypes.c_uint, ctypes.POINTER(Comparison)]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x7FFF0000)  # allow unless a rule below refuses it
    if not context:
        raise RuntimeError("could not create the kernel syscall policy")

    def deny(name, error=errno.EPERM, comparison=None):
        syscall = lib.seccomp_syscall_resolve_name(name.encode())
        if syscall < 0:
            return
        pointer = ctypes.pointer(comparison) if comparison is not None else None
        result = lib.seccomp_rule_add_array(context, 0x00050000 | error, syscall,
                                           int(comparison is not None), pointer)
        if result < 0:
            raise RuntimeError(f"could not restrict {name}: {result}")

    try:
        for name in ("fork", "vfork", "unshare", "setns", "ptrace", "process_vm_readv", "process_vm_writev",
                     "io_uring_setup", "io_uring_enter", "io_uring_register", "bpf", "perf_event_open",
                     "userfaultfd", "keyctl", "add_key", "request_key"):
            deny(name)
        # libc retries clone for threads when clone3 is unavailable. A clone
        # without CLONE_THREAD could multiply the per-process memory allowance.
        deny("clone3", errno.ENOSYS)
        deny("clone", comparison=Comparison(0, 7, 0x00010000, 0))
        if lib.seccomp_load(context) < 0:
            raise RuntimeError("could not load the kernel syscall policy")
    finally:
        lib.seccomp_release(context)
