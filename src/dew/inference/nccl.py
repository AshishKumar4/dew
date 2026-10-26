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
matches before its first broadcast. This side's communicators are nonblocking,
so while it waits on the rendezvous or on a broadcast it also watches the
HTTP calls the engine answers, and aborts the group when one of them fails
rather than waiting on a peer that never comes.

A push runs the vLLM publication lifecycle `SafetensorsReload` runs: it pauses
each replica with `mode=wait`, delivers, resets the prefix cache and resumes,
and stops at a phase that fails on any replica. Its delivery starts the update,
posts `/update_weights`, which returns once every tensor has landed, while this
side broadcasts them, and finishes the update under the version. Every process of a pool calls the push,
and the policy is gathered to process 0 as `SafetensorsReload` gathers it, in
groups of at most `dew._coordination.GATHER_BYTES`.
"""

from __future__ import annotations

import base64
import ctypes
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import SingleDeviceSharding

from dew._coordination import agreed, byte_groups, collective_host
from dew._records import JSON
from dew.batch import Variables
from dew.devices import primary_context
from dew.nn.inputs import mesh_of

from .rollouts import _on_every, _post, _served, _vllm_publication

if TYPE_CHECKING:
    import httpx

    from dew.interop.pretrained import Pretrained

_DTYPES = {"int8": 0, "uint8": 1, "int32": 2, "uint32": 3, "int64": 4, "uint64": 5, "float16": 6,
           "float32": 7, "float64": 8, "bfloat16": 9}
"""`ncclDataType_t` by numpy dtype name (nccl.h)."""

_SUM = 0
"""`ncclSum` (nccl.h)."""

_IN_PROGRESS = 7
"""`ncclInProgress` (nccl.h): a nonblocking communicator's call that has not finished."""

_NOT_READY = 600
"""`CUDA_ERROR_NOT_READY` (cuda.h): a stream with work still running."""


class _UniqueId(ctypes.Structure):
    _fields_: ClassVar = [("internal", ctypes.c_byte * 128)]


class _Config(ctypes.Structure):
    """`ncclConfig_t` as NCCL 2.17 lays it out (nccl.h.in, v2.17.1-1). A later NCCL reads the
    fields this size and version declare and keeps its own defaults for the rest
    (init.cc `parseCommConfig`, v2.27.7-1)."""

    _fields_: ClassVar = [
        ("size", ctypes.c_size_t), ("magic", ctypes.c_uint), ("version", ctypes.c_uint),
        ("blocking", ctypes.c_int), ("cga_cluster_size", ctypes.c_int), ("min_ctas", ctypes.c_int),
        ("max_ctas", ctypes.c_int), ("net_name", ctypes.c_char_p),
    ]


def _nonblocking() -> _Config:
    """A config that opens a nonblocking communicator, every other field NCCL's default."""
    undefined = -2 ** 31
    return _Config(ctypes.sizeof(_Config), 0xCAFEBEEF, 21700, 0, undefined, undefined, undefined, None)


class _Library:
    """The NCCL and CUDA driver calls a push makes.

    A communicator binds to the device whose context is current on the calling
    thread when it opens, and every call runs on that thread, so each call
    makes the sender device's primary context current first. Collectives run on
    the default stream; `settled` waits for them before a buffer is freed.
    """

    def __init__(self, library: str, ordinal: int) -> None:
        self.nccl = ctypes.CDLL(library)
        self.nccl.ncclGetErrorString.restype = ctypes.c_char_p
        self.nccl.ncclCommInitRankConfig.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, _UniqueId,
                                                     ctypes.c_int, ctypes.POINTER(_Config)]
        self.nccl.ncclCommGetAsyncError.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
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

    def _issued(self, status: int) -> None:
        """Check a nonblocking communicator's call, which may still be running (`settled` waits)."""
        if status != _IN_PROGRESS:
            self._checked(status)

    def current(self) -> None:
        self._driver(self.cuda.cuCtxSetCurrent(self.context))

    def unique_id(self) -> _UniqueId:
        uid = _UniqueId()
        self._checked(self.nccl.ncclGetUniqueId(ctypes.byref(uid)))
        return uid

    def open(self, uid: _UniqueId, rank: int, world: int) -> ctypes.c_void_p:
        """Start opening a nonblocking communicator; it is open once `settled` returns."""
        self.current()
        comm = ctypes.c_void_p()
        config = _nonblocking()
        status = self.nccl.ncclCommInitRankConfig(ctypes.byref(comm), world, uid, rank, ctypes.byref(config))
        try:
            self._issued(status)
        except RuntimeError:
            # NCCL may hand out the communicator before its start fails, and only an abort frees it.
            if comm.value is not None:
                self.abort(comm)
            raise
        return comm

    def all_reduce(self, comm: ctypes.c_void_p, pointer: int, count: int, dtype: str) -> None:
        self.current()
        self._issued(self.nccl.ncclAllReduce(pointer, pointer, count, _DTYPES[dtype], _SUM, comm, None))

    def broadcast(self, comm: ctypes.c_void_p, pointer: int, count: int, dtype: str) -> None:
        self.current()
        self._issued(self.nccl.ncclBroadcast(pointer, pointer, count, _DTYPES[dtype], 0, comm, None))

    def ready(self, comms: Sequence[ctypes.c_void_p], peers: Sequence[Future[httpx.Response]]) -> None:
        """Wait until none of `comms` is still issuing a call, as `settled` waits, leaving its
        collectives running on the stream. A nonblocking communicator takes no call while its last
        is in progress (NCCL's `ncclCommEnsureReady` refuses it), so each collective waits here
        before the next goes to the same communicator."""
        self._waited(comms, peers, stream=False)

    def settled(self, comms: Sequence[ctypes.c_void_p], peers: Sequence[Future[httpx.Response]]) -> None:
        """Wait until every one of `comms` has finished its calls and the default stream its
        collectives (not the trainer's own streams), polling NCCL between waits on `peers`, the
        replicas' HTTP calls that run meanwhile. A call that fails raises; the caller aborts the
        comms."""
        self._waited(comms, peers, stream=True)

    def _waited(self, comms: Sequence[ctypes.c_void_p], peers: Sequence[Future[httpx.Response]], *,
                stream: bool) -> None:
        """Poll `comms`, and with `stream` the default stream, until no call is in progress, watching
        `peers` between polls (`settled`)."""
        state = ctypes.c_int()
        while True:
            self.current()
            issuing = False
            for comm in comms:
                self._checked(self.nccl.ncclCommGetAsyncError(comm, ctypes.byref(state)))
                if state.value != _IN_PROGRESS:
                    self._checked(state.value)
                issuing = issuing or state.value == _IN_PROGRESS
            running = self.cuda.cuStreamQuery(None) if stream else 0
            if running != _NOT_READY:
                self._driver(running)
            if not issuing and running == 0:
                return
            answered, _ = wait(peers, timeout=0.001, return_when=FIRST_EXCEPTION)
            failure = next((peer.exception() for peer in answered if peer.exception() is not None), None)
            if failure is not None:
                raise RuntimeError("a replica's HTTP call failed while its NCCL group waited") from failure

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
        _vllm_publication(self.engines, version, self.timeout,
                          lambda: self._deliver(library, target, names, tensors, version), stamped=True)
        self.timings["send"] = time.perf_counter() - began

    def _deliver(self, library: _Library, target: SingleDeviceSharding, names: Sequence[str],
                 tensors: Sequence[np.ndarray], version: int) -> Mapping[str, BaseException]:
        """Load `tensors` into every paused replica over its group and finish the update under `version`.

        Each replica's `/update_weights` waits for the broadcast this side
        runs meanwhile. A failed broadcast, or a replica that did not take the
        tensors, closes the groups, since a group whose broadcast or replica
        failed cannot deliver the next version.
        """
        def start(root: str) -> None:
            _post(root, "/start_weight_update", None, self.timeout)

        def finish(root: str) -> None:
            _post(root, "/finish_weight_update", {"weight_version": str(version)}, self.timeout)

        failures = _on_every(self.engines, start)
        if failures:
            return failures
        listing: JSON = {
            "update_info": {
                "names": list(names),
                "dtype_names": [str(tensor.dtype) for tensor in tensors],
                "shapes": [[int(size) for size in tensor.shape] for tensor in tensors],
            }
        }
        with ThreadPoolExecutor(max_workers=len(self.engines), thread_name_prefix="dew-weight-push") as pool:
            receives = [(root, pool.submit(_post, root, "/update_weights", listing, self.timeout))
                        for root in self.engines]
            try:
                self._broadcast(library, target, tensors, [receive for _, receive in receives])
            except BaseException:
                self.close()
                raise
        failures = {root: failure for root, receive in receives
                    if (failure := receive.exception()) is not None}
        if failures:
            self.close()
            return failures
        return _on_every(self.engines, finish)

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
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="dew-weight-join") as pool:
                joining = pool.submit(_post, root, "/init_weight_transfer_engine", body, self.timeout)
                comm = library.open(uid, 0, 1 + workers)
                try:
                    library.settled([comm], [joining])
                    # The workers' communicator all-reduces one float32 as it opens.
                    warm = jax.device_put(np.zeros(1, np.float32), target).block_until_ready()
                    library.all_reduce(comm, warm.unsafe_buffer_pointer(), 1, "float32")
                    library.settled([comm], [joining])
                    if (failure := joining.exception()) is not None:
                        raise RuntimeError(f"{root} did not open the weight-transfer group") from failure
                except BaseException:
                    library.abort(comm)
                    raise
            self._groups[root] = comm
        return library

    def _broadcast(
        self, library: _Library, target: SingleDeviceSharding, tensors: Sequence[np.ndarray],
        receives: Sequence[Future[httpx.Response]],
    ) -> None:
        """Broadcast `tensors` in order to every group, `chunk` bytes resident at a time.

        The next chunk's copies are issued before the current chunk broadcasts,
        so the host-to-device copies overlap the transfers; a chunk is released
        only after the default stream has finished with it, or a replica's
        receive (`receives`) has failed. Each tensor goes to every group, and
        the next waits until each group has issued it (`_Library.ready`).
        """
        chunks = [[tensors[index] for index in group]
                  for group in byte_groups(tensors, self.chunk, nbytes=lambda tensor: tensor.nbytes)] or [[]]
        groups = list(self._groups.values())
        placed = jax.device_put(chunks[0], target)
        for index in range(len(chunks)):
            following = jax.device_put(chunks[index + 1], target) if index + 1 < len(chunks) else None
            for buffer in jax.block_until_ready(placed):
                for comm in groups:
                    library.broadcast(comm, buffer.unsafe_buffer_pointer(), buffer.size, str(buffer.dtype))
                library.ready(groups, receives)
            library.settled(groups, receives)
            if following is not None:
                placed = following

