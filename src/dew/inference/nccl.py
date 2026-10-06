"""Publish a policy version to vLLM replicas by NCCL broadcast from a trainer GPU.

`SafetensorsReload` writes the policy to disk and every engine reads it back.
`NCCLPush` sends the same tensors, `Pretrained.export`'s in the source layout,
from one trainer GPU into each replica's GPUs over an NCCL group the two sides
open together. vLLM (checked against v0.30.0) serves that group when launched
with `--weight-transfer-config '{"backend": "nccl"}'` and
`VLLM_SERVER_DEV_MODE=1`: its workers join a group of `1 + world size` ranks at
`rank_offset=1` and receive each tensor by broadcast from rank 0, in the order
`/update_weights` lists them, loading each as it lands.

The trainer holds no torch, so this module drives NCCL through ctypes. It must
load the NCCL library the engine runs (`library`, usually the engine
environment's `nvidia/nccl/lib/libnccl.so.2`): NCCL refuses a peer of another
version when the group opens. The trainer mints the group's unique id and posts
it base64-encoded (`nccl_unique_id_b64`), and both sides enter
`ncclCommInitRank` together, since that rendezvous has no store to wait on.
The workers' communicator then runs a one-element all-reduce, which this side
matches before its first broadcast.

A push pauses each replica with `mode=wait` and posts `/update_weights`, which
returns once every tensor has landed, while this side broadcasts them; it then
sets the version, resets the prefix cache and resumes, with the pause and
failure rules of `SafetensorsReload`. Every process of a pool calls the push,
and the policy is gathered to process 0 as `SafetensorsReload` gathers it, in
groups of at most `dew.coordination.GATHER_BYTES`.
"""

from __future__ import annotations

import base64
import ctypes
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import SingleDeviceSharding

from dew.coordination import agreed, collective_host
from dew.nn.inputs import mesh_of
from dew.objectives.base import Variables
from dew.records import JSON
from dew.telemetry.devices import primary_context

from .rollouts import _post, _served

if TYPE_CHECKING:
    from dew.interop.pretrained import Pretrained

_DTYPES = {"int8": 0, "uint8": 1, "int32": 2, "uint32": 3, "int64": 4, "uint64": 5, "float16": 6,
           "float32": 7, "float64": 8, "bfloat16": 9}
"""`ncclDataType_t` by numpy dtype name (nccl.h)."""

_SUM = 0
"""`ncclSum` (nccl.h)."""


class _UniqueId(ctypes.Structure):
    _fields_: ClassVar = [("internal", ctypes.c_byte * 128)]


class _Library:
    """The NCCL and CUDA driver calls a push makes.

    A communicator binds to the device whose context is current on the calling
    thread when it opens, and every call runs on that thread, so each call
    makes the sender device's primary context current first. Collectives run on
    the default stream; `synchronize` waits for them before a buffer is freed.
    """

    def __init__(self, library: str, ordinal: int) -> None:
        self.nccl = ctypes.CDLL(library)
        self.nccl.ncclGetErrorString.restype = ctypes.c_char_p
        self.nccl.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId,
                                               ctypes.c_int]
        collective = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                      ctypes.c_void_p, ctypes.c_void_p]
        self.nccl.ncclAllReduce.argtypes = collective
        self.nccl.ncclBroadcast.argtypes = collective
        self.nccl.ncclCommAbort.argtypes = [ctypes.c_void_p]
        self.cuda, self.context, _ = primary_context(ordinal)

    def _driver(self, status: int) -> None:
        if status != 0:
            raise RuntimeError(f"CUDA driver error {status}")

    def _checked(self, status: int) -> None:
        if status != 0:
            raise RuntimeError(f"NCCL error {status}: {self.nccl.ncclGetErrorString(status).decode()}")

    def current(self) -> None:
        self._driver(self.cuda.cuCtxSetCurrent(self.context))

    def unique_id(self) -> _UniqueId:
        uid = _UniqueId()
        self._checked(self.nccl.ncclGetUniqueId(ctypes.byref(uid)))
        return uid

    def open(self, uid: _UniqueId, rank: int, world: int) -> ctypes.c_void_p:
        self.current()
        comm = ctypes.c_void_p()
        self._checked(self.nccl.ncclCommInitRank(ctypes.byref(comm), world, uid, rank))
        return comm

    def all_reduce(self, comm: ctypes.c_void_p, pointer: int, count: int, dtype: str) -> None:
        self.current()
        self._checked(self.nccl.ncclAllReduce(pointer, pointer, count, _DTYPES[dtype], _SUM, comm, None))

    def broadcast(self, comm: ctypes.c_void_p, pointer: int, count: int, dtype: str) -> None:
        self.current()
        self._checked(self.nccl.ncclBroadcast(pointer, pointer, count, _DTYPES[dtype], 0, comm, None))

    def synchronize(self) -> None:
        """Wait for the collectives on the default stream, not the trainer's own streams."""
        self.current()
        self._driver(self.cuda.cuStreamSynchronize(None))

    def abort(self, comm: ctypes.c_void_p) -> None:
        """Free `comm` without waiting for its peers or its operations."""
        self._checked(self.nccl.ncclCommAbort(comm))


def _world(root: str, timeout: float) -> int:
    """The replica's worker count, which its NCCL group holds beside the sender."""
    import httpx

    response = httpx.get(root + "/get_world_size", params={"include_dp": "true"}, timeout=timeout)
    answer = response.json() if response.status_code == 200 else None
    size = answer.get("world_size") if isinstance(answer, dict) else None
    if type(size) is not int or size < 1:
        raise RuntimeError(f"/get_world_size answered {response.status_code}: {response.text}")
    return size


@dataclass
class NCCLPush:
    """Publishes a policy version to vLLM replicas by NCCL broadcast from a trainer GPU.

    The replicas run vLLM (checked against v0.30.0) launched with
    `--weight-transfer-config '{"backend": "nccl"}'` and
    `VLLM_SERVER_DEV_MODE=1`. The module docstring of `dew.inference.nccl`
    describes the handshake and the push in full.

    - `source` is the `Pretrained` the replicas were launched from; its
      `export` gives the tensors.
    - `engines` are the replicas' root URLs, not their `/v1` API URLs.
    - `library` is the engine's own libnccl.so.2, because NCCL refuses a
      peer of another version when the group opens.
    - `chunk` bounds the bytes of tensors placed on the sender device beside
      the trainer's state. The next chunk is copied over while the current
      one broadcasts, so a push holds at most two chunks there, or two of its
      largest tensor.

    Floating-point leaves are cast to `dtype` before the push, and `timeout`
    bounds each HTTP call in seconds. Groups open on the first push and stay
    open until `close`. Every process of a pool calls the push.

    `timings` holds the seconds each phase of the last push took: `gather`
    (the cast and the gather to process 0's host, on every process), and on
    process 0, which sends, `export`, `send` (copies and broadcasts until
    every replica has loaded the tensors) and `total`.
    """

    source: Pretrained
    engines: tuple[str, ...]
    library: str
    dtype: str = "bfloat16"
    chunk: int = 512 * 2 ** 20
    timeout: float = 600.0
    timings: dict[str, float] = field(default_factory=dict, init=False)
    _library: _Library | None = field(default=None, init=False, repr=False)
    _groups: dict[str, ctypes.c_void_p] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if len(set(self.engines)) != len(self.engines):
            raise ValueError("every replica is published to once: a replica listed twice would wait on its "
                             "second /update_weights for a broadcast that never comes")
        if not Path(self.library).is_file():
            raise ValueError(f"no NCCL library at {self.library}; pass the engine's own libnccl.so.2")

    def __call__(self, variables: Variables, version: int) -> None:
        began = time.perf_counter()
        mesh = mesh_of(variables)
        # Process 0 holds the gathered policy, so it sends, from its first device of the mesh.
        sender = jax.local_devices()[0] if mesh is None else next(
            device for device in mesh.devices.flat if device.process_index == 0)
        served = collective_host(_served(variables, jnp.dtype(self.dtype)), phase="weight push gather",
                                 held_by="first")
        self.timings["gather"] = time.perf_counter() - began
        agreed("weight push", lambda: None if served is None else self._push(
            served, SingleDeviceSharding(sender), sender.local_hardware_id, version))
        self.timings["total"] = time.perf_counter() - began

    def close(self) -> None:
        """Abort the NCCL groups this side opened. A later push opens them again.

        Groups are aborted rather than destroyed, because NCCL's destroy
        finalizes the group, which waits on the engine's side, and an engine
        keeps its side open until it exits. A failed push also calls `close`,
        because a group whose broadcast or replica failed cannot deliver the
        next version.
        """
        if self._library is not None:
            for comm in self._groups.values():
                self._library.abort(comm)
        self._groups.clear()

    def _push(self, served: Variables, target: SingleDeviceSharding, ordinal: int, version: int) -> None:
        began = time.perf_counter()
        exported = self.source.export(served)
        names = sorted(exported)
        tensors = [exported[name] for name in names]
        self.timings["export"] = time.perf_counter() - began
        library = self._opened(target, ordinal)
        began = time.perf_counter()
        for root in self.engines:
            _post(root, "/pause?mode=wait", None, self.timeout)
            _post(root, "/start_weight_update", None, self.timeout)
        listing: JSON = {
            "update_info": {
                "names": list(names),
                "dtype_names": [str(tensor.dtype) for tensor in tensors],
                "shapes": [[int(size) for size in tensor.shape] for tensor in tensors],
            }
        }
        failures: list[BaseException] = []

        def receive(root: str) -> None:
            try:
                _post(root, "/update_weights", listing, self.timeout)
            except BaseException as failure:
                failures.append(failure)

        receivers = [threading.Thread(target=receive, args=(root,)) for root in self.engines]
        for receiver in receivers:
            receiver.start()
        try:
            self._broadcast(library, target, tensors)
        except BaseException:
            self.close()
            raise
        finally:
            for receiver in receivers:
                receiver.join()
        if failures:
            self.close()
            raise RuntimeError(f"version {version} did not reach every replica") from failures[0]
        for root in self.engines:
            _post(root, "/finish_weight_update", {"weight_version": str(version)}, self.timeout)
            _post(root, "/reset_prefix_cache", None, self.timeout, reports=True)
            _post(root, "/resume", None, self.timeout)
        self.timings["send"] = time.perf_counter() - began

    def _opened(self, target: SingleDeviceSharding, ordinal: int) -> _Library:
        """The library bound to the sending device, CUDA `ordinal`, with a group open to every replica."""
        if self._library is None:
            self._library = _Library(self.library, ordinal)
        library = self._library
        for root in self.engines:
            if root in self._groups:
                continue
            workers = _world(root, self.timeout)
            uid = library.unique_id()
            body: JSON = {"init_info": {"nccl_unique_id_b64": base64.b64encode(bytes(uid.internal)).decode(),
                                        "rank_offset": 1, "world_size": 1 + workers, "packed": False}}
            failures: list[BaseException] = []

            def join(root: str = root, body: JSON = body, failures=failures) -> None:
                try:
                    _post(root, "/init_weight_transfer_engine", body, self.timeout)
                except BaseException as failure:
                    failures.append(failure)

            joining = threading.Thread(target=join)
            joining.start()
            comm = library.open(uid, 0, 1 + workers)
            # The workers' communicator all-reduces one float32 as it opens.
            warm = jax.device_put(np.zeros(1, np.float32), target).block_until_ready()
            library.all_reduce(comm, warm.unsafe_buffer_pointer(), 1, "float32")
            library.synchronize()
            joining.join()
            if failures:
                library.abort(comm)
                raise RuntimeError(f"{root} did not open the weight-transfer group") from failures[0]
            self._groups[root] = comm
        return library

    def _broadcast(
        self, library: _Library, target: SingleDeviceSharding, tensors: Sequence[np.ndarray]
    ) -> None:
        """Broadcast `tensors` in order to every group, `chunk` bytes resident at a time.

        The next chunk's copies are issued before the current chunk broadcasts,
        so the host-to-device copies overlap the transfers; a chunk is released
        only after the default stream has finished with it.
        """
        chunks: list[list[np.ndarray]] = [[]]
        size = 0
        for tensor in tensors:
            if chunks[-1] and size + tensor.nbytes > self.chunk:
                chunks.append([])
                size = 0
            chunks[-1].append(tensor)
            size += tensor.nbytes
        placed = jax.device_put(chunks[0], target)
        for index in range(len(chunks)):
            following = jax.device_put(chunks[index + 1], target) if index + 1 < len(chunks) else None
            for buffer in jax.block_until_ready(placed):
                for comm in self._groups.values():
                    library.broadcast(comm, buffer.unsafe_buffer_pointer(), buffer.size, str(buffer.dtype))
            library.synchronize()
            if following is not None:
                placed = following

