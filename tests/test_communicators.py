"""The room NCCL's communicators take outside XLA's pool (ledger 2901): measured once per configuration by a
child process before the backend opens and kept in Dew's cache, the pool's share held to leave it at NCCL's
full speed, the run's share its upper bound, and a compiled program whose communicators need more room than
Dew left them refused before it runs, never left to stall in a collective. The measurement itself needs two
GPUs, which a two-GPU run of the check makes; here it is the child's report, as the parent reads it."""

import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import jax
import pytest

from dew import communicators
from dew._cache import package_digest
from dew._devices import POOL_NAMES, pool_bound
from dew.communicators import Collectives, CommunicatorRoom, Needs

SOURCE = Path(__file__).resolve().parents[1] / "src"
GIB = 2**30

def connected(gpus: int, send: int, receive: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Each directed pair of `gpus` GPUs connected by `send` bytes on its source and `receive` on its
    target and none on any other GPU, as NCCL's direct peer transports allocate."""
    return dict.fromkeys(itertools.permutations(range(gpus), 2), (send, receive, 0))


NEEDS = Needs(collective={2: GIB}, exchanged={2: GIB // 4}, permuted={2: GIB},
              connections=connected(2, GIB // 2, GIB // 2), runtime={0: GIB // 2, 1: GIB // 2}, seconds=4.0)
"""Two GPUs' measurement: a collective's communicator over both takes 1 GiB on each, an all-to-all's
connections to its peer a quarter more, a permute's communicator 1 GiB on each member and half a GiB for
each send and each receive, the runtime half."""

EVERY = Collectives(groups=frozenset({(0, 1)}), exchanges=frozenset({(0, 1)}),
                    permutes={(0, 1): frozenset({(0, 1), (1, 0)})})
"""What Dew leaves room for before a program compiles: a collective's communicator over both GPUs, its
all-to-all's connections to every peer, and a permute along every pair."""

MEMORY = {0: (79 * GIB, 80 * GIB), 1: (76 * GIB, 80 * GIB)}
"""Each GPU's free and total bytes before the backend opens: 79 and 76 of 80 GiB free."""

FOUR = Needs(collective={2: GIB, 4: 2 * GIB}, exchanged={2: GIB // 4, 4: GIB // 2}, permuted={4: GIB},
             connections=connected(4, GIB // 4, GIB // 8), runtime=dict.fromkeys(range(4), GIB // 2),
             seconds=8.0)
"""Four GPUs' measurement: a collective's communicator over two of them 1 GiB on each, over all four 2 GiB;
a permute's over four 1 GiB on each member, a quarter for each send and an eighth for each receive."""


def test_the_pool_takes_at_most_the_runs_share_and_leaves_the_communicators_room():
    """The pool's share of each GPU is the run's (`dew._devices.pool_bound`) as an upper bound: it leaves the
    room its communicators and the runtime take outside the pool at NCCL's full speed, the tightest GPU
    deciding, to a thousandth below, and NCCL is never slowed to fit. A pool the communicators leave no room
    for at all is refused, naming each side's bytes."""
    tightest = 76 * GIB - (GIB + GIB // 4 + GIB + GIB // 2 + GIB // 2 + GIB // 2)
    for bound in (0.95, 0.97):
        share = communicators.pool_share(NEEDS, MEMORY, bound, EVERY)
        assert share * 80 * GIB <= tightest < (share + 0.001) * 80 * GIB
    roomy = {0: (79 * GIB, 80 * GIB), 1: (79 * GIB, 80 * GIB)}
    assert communicators.pool_share(NEEDS, roomy, 0.75, EVERY) == 0.75
    with pytest.raises(CommunicatorRoom) as refused:
        communicators.pool_share(NEEDS, {0: (3 * GIB, 80 * GIB)}, 0.97, EVERY)
    assert refused.value.needed == {0: 3 * GIB + 3 * GIB // 4} and refused.value.room == {0: 3 * GIB}


@pytest.mark.parametrize(("named", "bound"), [
    ({}, 0.95), ({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.97"}, 0.97),
    ({"XLA_CLIENT_MEM_FRACTION": "0.5", "XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, 0.5),
    ({"XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, 0.75), ({"XLA_PYTHON_CLIENT_ALLOCATOR": "bfc"}, 0.75),
    ({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, None), ({"XLA_PYTHON_CLIENT_ALLOCATOR": "platform"}, None)])
def test_one_owner_reads_the_pools_bound_from_how_the_run_named_it(named, bound, monkeypatch):
    """The pool's upper bound, as the run names it: Dew's 95% where it names nothing, its fraction where it
    names one under either of JAX's names, JAX's own 75% where it names preallocation or the allocator
    without one, and none where it holds no fixed pool (a pool that grows, the platform's allocator), which
    Dew neither sizes nor admits programs against."""
    for name in POOL_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in named.items():
        monkeypatch.setenv(name, value)
    assert pool_bound(cuda=True) == bound
    assert pool_bound(cuda=False) is None


def test_a_programs_communicators_are_charged_by_their_groups_exchanges_and_permute_pairs():
    """Each GPU is charged for every distinct group its collectives take, at that group's size; for the
    connections an all-to-all makes to every peer of its group; and, on each permute clique it is a member
    of, the communicator's base, a send for each pair it sends along and a receive for each it receives
    along. A permute clique already made adds only its new pairs; another clique over some of the same GPUs
    is a communicator of its own."""
    every = (0, 1, 2, 3)
    held = Collectives(groups=frozenset({(0, 1), (2, 3), every}), exchanges=frozenset({(0, 1)}),
                       permutes={every: frozenset({(0, 1), (0, 2), (1, 0)})})
    assert FOUR.needed(held, range(4), runtime=False) == {
        0: GIB + 2 * GIB + GIB // 4 + GIB + 2 * (GIB // 4) + GIB // 8,
        1: GIB + 2 * GIB + GIB // 4 + GIB + GIB // 4 + GIB // 8,
        2: GIB + 2 * GIB + GIB + GIB // 8,
        3: GIB + 2 * GIB + GIB}
    pair = Collectives(permutes={(0, 1): frozenset({(0, 1)})})
    assert FOUR.needed(Collectives(permutes={every: frozenset({(0, 1)})}), range(4), runtime=False,
                       made=pair) == {0: GIB + GIB // 4, 1: GIB + GIB // 8, 2: GIB, 3: GIB}
    assert FOUR.needed(Collectives(permutes={every: frozenset({(0, 1), (2, 3)})}), range(4), runtime=False,
                       made=Collectives(permutes={every: frozenset({(0, 1)})})) == {
        0: 0, 1: 0, 2: GIB // 4, 3: GIB // 8}


def test_a_compiled_programs_communicators_are_read_through_the_flop_counts_parser():
    """A compiled program's communicators, read by the parser that counts its FLOPs: each distinct group of
    its collectives however XLA prints it, listed, as an iota or over a mesh's named axes (an fsdp x
    sequence step's two groupings are two), its all-to-alls' and ragged all-to-alls' groups, and its
    permutes' directed pairs on one clique of its devices, the module's positions read as the GPUs its
    device assignment names."""
    program = ("HloModule step, entry_computation_layout={()->f32[8]}\n\nENTRY %main () -> f32[8] {\n"
               "  %x = f32[8] parameter(0)\n"
               "  %a = f32[8] all-gather(f32[8] %x), replica_groups={{0,1},{2,3}}, dimensions={0}\n"
               "  %b = f32[8] all-reduce(f32[8] %x), replica_groups=[2,2]<=[4], to_apply=%add\n"
               "  %c = f32[8] all-reduce(f32[8] %x), replica_groups=mesh['fsdp'=2,'sequence'=2] {'fsdp'}, "
               "to_apply=%add\n"
               "  %d = f32[8] all-to-all(f32[8] %x), replica_groups={}, dimensions={0}\n"
               "  %r = f32[8] ragged-all-to-all(f32[8] %x), replica_groups={{0,2},{1,3}}\n"
               "  ROOT %e = f32[8] collective-permute(f32[8] %x), source_target_pairs={{0,1},{1,2}}\n}\n")
    gpus = (3, 2, 1, 0)
    assert communicators.collectives_of(program, gpus) == Collectives(
        groups=frozenset({(3, 2), (1, 0), (3, 1), (2, 0), gpus}),
        exchanges=frozenset({gpus, (3, 1), (2, 0)}),
        permutes={gpus: frozenset({(3, 2), (2, 1)})})


def test_a_processs_communicators_are_held_together_to_the_room_dew_left_outside_the_pool():
    """Every communicator a process has made stays made (XLA caches them), so each compiled program is
    admitted only if its communicators and those of the programs before it, each counted once, and the
    runtime fit the room Dew left outside the pool, and the communicators it adds fit what its GPUs have
    free there now; one beyond either is refused before it runs, naming the bytes, never left to stall in a
    collective."""
    room = dict.fromkeys(range(4), 3 * GIB + 3 * GIB // 4)
    every = (0, 1, 2, 3)
    resident = communicators.Resident()

    def roomy(gpu: int) -> int:
        return 4 * GIB

    resident.admit(Collectives(groups=frozenset({every, (0, 1), (2, 3)})), FOUR, room, roomy)
    resident.admit(Collectives(groups=frozenset({every})), FOUR, room, lambda gpu: 0)
    with pytest.raises(CommunicatorRoom) as refused:
        resident.admit(Collectives(exchanges=frozenset({every})), FOUR, room, roomy)
    assert refused.value.needed == dict.fromkeys(range(4), 2 * GIB + GIB + GIB // 2 + GIB // 2)
    with pytest.raises(CommunicatorRoom):
        resident.admit(Collectives(exchanges=frozenset({(0, 1)})), FOUR, room, lambda gpu: GIB // 8)
    assert resident.held == Collectives(groups=frozenset({every, (0, 1), (2, 3)}))


def test_what_a_gpu_has_spent_already_is_not_charged_to_a_program_again():
    """The free memory a program is admitted against is read after the runtime started and after the
    programs before it made their communicators, so neither is charged to it again: a program whose new
    communicators take exactly what its GPUs have free is admitted, the runtime warmed before it, and so is
    one adding a pair's communicator beside one already made."""
    room = dict.fromkeys(range(4), 3 * GIB + 3 * GIB // 4)
    resident = communicators.Resident()
    resident.admit(Collectives(groups=frozenset({(0, 1, 2, 3), (0, 1)})), FOUR, room, lambda gpu: 3 * GIB)
    resident.admit(Collectives(groups=frozenset({(2, 3)})), FOUR, room, lambda gpu: GIB)
    assert resident.held == Collectives(groups=frozenset({(0, 1, 2, 3), (0, 1), (2, 3)}))


def test_a_programs_code_is_held_to_what_is_free_beside_its_new_communicators():
    """A program's code takes its GPUs' memory outside XLA's pool as it first runs and leaves as its
    executable is released, neither of which a process sees: at every admission, a program whose new
    communicators fit what its GPUs have free now but not with its code is refused, naming the bytes its code
    and communicators need and the bytes free, an executable admitted again (loaded again, perhaps) charged
    its code again; the room, which holds what stays made, never holds code."""
    room = dict.fromkeys(range(4), 4 * GIB)
    every = Collectives(groups=frozenset({(0, 1, 2, 3)}))
    pair = Collectives(groups=frozenset({(0, 1)}))
    resident = communicators.Resident()
    step = communicators.Loaded(b"step", (0, 1, 2, 3), GIB)
    resident.admit(every, FOUR, room, lambda gpu: 3 * GIB, step)
    with pytest.raises(CommunicatorRoom) as again:
        resident.admit(every, FOUR, room, lambda gpu: GIB // 2, step)
    assert again.value.needed == dict.fromkeys(range(4), GIB)
    assert again.value.room == dict.fromkeys(range(4), GIB // 2)
    resident.admit(every, FOUR, room, lambda gpu: GIB, step)
    evaluation = communicators.Loaded(b"eval", (0, 1), GIB // 2)
    with pytest.raises(CommunicatorRoom):
        resident.admit(pair, FOUR, room, lambda gpu: GIB, evaluation)
    resident.admit(pair, FOUR, room, lambda gpu: GIB + GIB // 2, evaluation)


def test_a_programs_communicators_are_kept_under_the_digest_of_dews_code(tmp_path, monkeypatch):
    """A program's communicators are kept under the digest of Dew's own code (`dew._cache.package_digest`),
    the key every cache of what Dew's code computed reads: those kept under this Dew's digest are taken as
    they are kept, its text unread, and those kept by another Dew's code, under its digest or none, are read
    again from the program's text."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    kept = tmp_path / "dew" / "communicators" / "programs"
    planted = Collectives(groups=frozenset({(0, 1, 2, 3)}))

    def unread() -> str:
        raise AssertionError("the text of a program whose communicators this Dew kept is read again")

    ours = kept / package_digest() / f"{b'kept'.hex()}.json"
    ours.parent.mkdir(parents=True)
    ours.write_text(json.dumps(planted.to_json()))
    assert communicators.collectives_kept(b"kept", (0, 1, 2, 3), unread) == planted
    for earlier in (kept / f"{b'step'.hex()}.json", kept / ("0" * 64) / f"{b'step'.hex()}.json"):
        earlier.parent.mkdir(parents=True, exist_ok=True)
        earlier.write_text(json.dumps(planted.to_json()))
    program = ("HloModule step, entry_computation_layout={()->f32[8]}\n\nENTRY %main () -> f32[8] {\n"
               "  %x = f32[8] parameter(0)\n"
               "  ROOT %a = f32[8] all-gather(f32[8] %x), replica_groups={{0,1},{2,3}}, dimensions={0}\n}\n")
    assert communicators.collectives_kept(b"step", (0, 1, 2, 3), lambda: program) == Collectives(
        groups=frozenset({(0, 1), (2, 3)}))


def test_a_programs_communicators_are_read_once_per_executable(tmp_path, monkeypatch):
    """A program's communicators are read from its text once per executable, kept in Dew's cache by its
    fingerprint as positions in its device assignment, so a warm start that loads it from the compilation
    cache reads no text, and the same executable over other GPUs maps them to those."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    program = ("HloModule step, entry_computation_layout={()->f32[8]}\n\nENTRY %main () -> f32[8] {\n"
               "  %x = f32[8] parameter(0)\n"
               "  ROOT %a = f32[8] all-gather(f32[8] %x), replica_groups={{0,1},{2,3}}, dimensions={0}\n}\n")
    read = []

    def text():
        read.append(1)
        return program

    first = communicators.collectives_kept(b"step", (0, 1, 2, 3), text)
    again = communicators.collectives_kept(b"step", (3, 2, 1, 0), text)
    assert read == [1]
    assert first == Collectives(groups=frozenset({(0, 1), (2, 3)}))
    assert again == Collectives(groups=frozenset({(3, 2), (1, 0)}))


def test_programs_admitted_at_once_are_held_together():
    """Two programs compiled at once, each of whose communicators fits the room alone and not with the
    other's, are admitted one at a time: one is admitted and the other refused."""
    room = dict.fromkeys(range(4), 2 * GIB + GIB // 2 + GIB // 4)
    resident = communicators.Resident()
    outcomes = []
    barrier = threading.Barrier(2)

    def compile_one(group):
        barrier.wait()
        try:
            resident.admit(Collectives(groups=frozenset({group})), FOUR, room, lambda gpu: None)
        except CommunicatorRoom:
            outcomes.append("refused")
        else:
            outcomes.append("admitted")

    threads = [threading.Thread(target=compile_one, args=(group,)) for group in ((0, 1, 2, 3), (3, 2, 1, 0))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["admitted", "refused"]


ADMITTED = """
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, PartitionSpec
from dew import communicators
from dew.communicators import Needs, Sizing
from dew.nn.kernels import options

mesh = Mesh(np.array(jax.devices()[:2]), ("pair",))
summed = jax.shard_map(lambda part: jax.lax.psum(part, "pair"), mesh=mesh, in_specs=PartitionSpec("pair"),
                       out_specs=PartitionSpec())
ids = [communicators.ordinal(device) for device in mesh.devices.flat]
options.program(summed).compiled(jnp.ones(4))
tight = Needs(collective={2: 2**30}, exchanged={}, permuted={}, connections={}, runtime=dict.fromkeys(ids, 0),
              seconds=1.0)
communicators.hold(Sizing(tight, 0.95, 0.9, dict.fromkeys(ids, 2**30), dict.fromkeys(ids, 2**29)))
options.program(summed).compiled(jnp.ones(4))
print("compiled")
"""
"""A fresh process of two CPU devices compiling one all-reduce through `options.program`, then holding a
sizing that left too little room for its communicator over two GPUs and compiling it again."""


def test_a_cpu_program_takes_none_of_the_gpus_room():
    """A program loaded on CPU devices makes no NCCL communicator and holds no GPU code, so a process whose
    sizing leaves its GPUs too little room for the same all-reduce over two of them compiles it on two CPU
    devices, admitted against nothing."""
    done = subprocess.run([sys.executable, "-c", ADMITTED], capture_output=True, text=True, check=True,
                          env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE),
                               "XLA_FLAGS": "--xla_force_host_platform_device_count=2"})
    assert done.stdout.strip().splitlines()[-1] == "compiled"


AHEAD = """
import jax, jax.numpy as jnp, numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from dew import communicators
from dew.communicators import Needs, Sizing
from dew.nn.kernels import options
from dew.training.ahead import compile_only_gpus

mesh = Mesh(np.array(compile_only_gpus("sm90", 2)), ("pair",))
summed = jax.shard_map(lambda part: jax.lax.psum(part, "pair"), mesh=mesh, in_specs=PartitionSpec("pair"),
                       out_specs=PartitionSpec())
tight = Needs(collective={2: 2**30}, exchanged={}, permuted={}, connections={}, runtime={0: 0, 1: 0},
              seconds=1.0)
communicators.hold(Sizing(tight, 0.95, 0.9, {0: 2**30, 1: 2**30}, {0: 2**29, 1: 2**29}))
rows = jax.ShapeDtypeStruct((4,), jnp.float32, sharding=NamedSharding(mesh, PartitionSpec("pair")))
options.program(summed).compiled(rows)
print("compiled ahead")
"""
"""A process holding a sizing too small for a pair's communicator, compiling an all-reduce ahead for two
compile-only GPUs that this host does not have."""


def test_a_program_compiled_ahead_for_absent_gpus_is_not_admitted_here():
    """A program compiled ahead for GPUs this process has not (`compile_only_gpus`) is never loaded here, so
    its communicators take none of this host's room: a process whose sizing would refuse the same program on
    its own GPUs compiles it ahead."""
    done = subprocess.run([sys.executable, "-c", AHEAD], capture_output=True, text=True, check=True,
                          env={**os.environ, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)})
    assert done.stdout.strip().splitlines()[-1] == "compiled ahead"


def test_a_process_prepared_again_keeps_the_sizing_its_pool_was_made_with():
    """Once the backend has opened, the pool keeps the share its first preparation sized, so a second
    preparation (a sweep's next run, a profile after a run) keeps that sizing and admits every later
    program against it, never holding nothing."""
    room = dict.fromkeys(range(2), GIB)
    sizing = communicators.Sizing(NEEDS, 0.95, 0.9, room, room)
    held = communicators.held()
    communicators.hold(sizing)
    try:
        assert communicators.sized_pool(cuda=True, opened=True) is sizing
    finally:
        communicators.hold(held)


def test_the_sized_pool_is_recorded_with_what_it_left_room_for(monkeypatch):
    """A pool Dew sized states the run's bound, the share it took, each GPU's bytes its communicators need
    and the room it left them, which the fit check's record carries into the run record; the pool's share
    goes to XLA through the environment."""
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.97")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    sized = communicators.sized_pool(measure=lambda: NEEDS, memories=lambda: MEMORY, cuda=True, opened=False)
    assert sized is not None and sized.bound == 0.97
    assert float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]) == sized.share < 0.97
    assert sized.needed == NEEDS.needed(EVERY, MEMORY, runtime=True)
    assert sized.room == {gpu: free - int(sized.share * total) for gpu, (free, total) in MEMORY.items()}
    assert sized.record()["room"] == {str(gpu): room for gpu, room in sized.room.items()}


def test_a_measurement_is_paid_once_per_configuration(tmp_path, monkeypatch):
    """A configuration's measurement is kept in Dew's cache and read back without a child process; another
    NCCL setting, another XLA NCCL flag, another JAX build or the same model of GPUs at other places is
    another configuration, measured afresh."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    taken = []

    def measure(configuration):
        taken.append(configuration)
        return NEEDS

    configuration = {"gpus": ["NVIDIA H100 80GB HBM3"] * 2, "placement": ["0000:18:00.0", "0000:2a:00.0"],
                     "driver": 12090, "nccl": 23203, "nccl_env": {}, "xla_nccl": [], "jax": "0.12.0"}
    assert communicators.measured(configuration, measure=measure) == NEEDS
    assert communicators.measured(configuration, measure=measure) == NEEDS
    others = [{**configuration, "nccl_env": {"NCCL_NVLS_ENABLE": "0"}},
              {**configuration, "xla_nccl": ["--xla_gpu_nccl_collective_max_nchannels=4"]},
              {**configuration, "jax": "0.12.1"},
              {**configuration, "placement": ["0000:3a:00.0", "0000:5d:00.0"]}]
    for other in others:
        assert communicators.measured(other, measure=measure) == NEEDS
    assert taken == [configuration, *others]


def test_a_measurement_kept_in_an_earlier_form_is_measured_afresh(tmp_path, monkeypatch):
    """A host's measurement kept by an earlier Dew, whose child wrote a permute's costs as two numbers, is
    not read as this one's: the configuration is measured afresh."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    configuration = {"gpus": ["NVIDIA H100 80GB HBM3"] * 2, "placement": ["0000:18:00.0", "0000:2a:00.0"],
                     "driver": 12090, "nccl": 23203, "nccl_env": {}, "xla_nccl": [], "jax": "0.12.0"}
    digest = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:24]
    earlier = tmp_path / "dew" / "communicators" / f"{digest}.json"
    earlier.parent.mkdir(parents=True)
    earlier.write_text(json.dumps({"configuration": configuration, "needs": {
        "collective": {"2": GIB}, "exchanged": {"2": GIB // 4}, "permuted": {"2": [GIB, GIB // 2]},
        "runtime": {"0": GIB // 2, "1": GIB // 2}, "seconds": 4.0}}))
    taken = []

    def measure(configured):
        taken.append(configured)
        return NEEDS

    assert communicators.measured(configuration, measure=measure) == NEEDS
    assert taken == [configuration]


def test_a_permutes_connections_are_charged_to_every_gpu_that_may_hold_their_buffers():
    """A permute's communicator is measured connecting every directed pair of its clique in turn, the first
    making the communicator, as NCCL picks each peer's transport apart: GPU buffers on a pair over NVLink or
    PCIe peer access, host memory for one through shared memory, and a third GPU's buffers for one it
    routes through that GPU (`P2P_INTERMEDIATE`). Each later pair keeps its bytes on its source, on its
    target and the most on any other GPU; the first pair's own connection is its reverse's, and the base what
    the first run took beside it. A clique charges each new pair's source and
    target theirs, and every other member the most a GPU beside the pair took, as NCCL may route the pair
    through any of them, so pairs routed through GPU 2 accumulate there."""
    mib = 2**20
    runs = [(0, 1, {0: 65 * mib, 1: 65 * mib, 2: 64 * mib, 3: 64 * mib}),
            (1, 0, {0: 0, 1: 0, 2: 8 * mib, 3: 0}),
            (2, 3, {0: 0, 1: 0, 2: 40 * mib, 3: 24 * mib}),
            (0, 3, {0: mib, 1: 0, 2: 8 * mib, 3: mib})]
    base, connections = communicators.peer_costs(runs)
    assert base == 65 * mib
    assert connections == {(0, 1): (0, 0, 8 * mib), (1, 0): (0, 0, 8 * mib), (2, 3): (40 * mib, 24 * mib, 0),
                           (0, 3): (mib, mib, 8 * mib)}
    needs = Needs(collective={}, exchanged={}, permuted={4: base}, connections=connections,
                  runtime={}, seconds=1.0)
    routed = Collectives(permutes={(0, 1, 2, 3): frozenset({(1, 0), (0, 3)})})
    assert needs.needed(routed, range(4), runtime=False) == {
        0: 65 * mib + mib, 1: 65 * mib + 8 * mib, 2: 65 * mib + 16 * mib, 3: 65 * mib + 8 * mib + mib}


def test_a_pair_is_charged_its_direct_route_where_its_clique_holds_no_gpu_to_route_it_through():
    """A pair routed through a third GPU in a clique of every GPU is measured on its own too, in the
    communicator of its two GPUs, where NCCL connects it directly, with a device FIFO on each end
    (`NCCL_SHM_USE_CUDA_MEMCPY`): a clique of the two charges each end the more of its two routes, and
    no other GPU anything, and charges a clique measured alone what its runs took in all, its first
    connection once."""
    mib = 2**20
    every = [(0, 1, {0: 9 * mib, 1: 9 * mib, 2: 8 * mib}), (0, 2, {0: mib, 1: 6 * mib, 2: mib}),
             (1, 0, {0: mib, 1: mib, 2: 0}), (1, 2, {0: 0, 1: mib, 2: mib}),
             (2, 0, {0: mib, 1: 6 * mib, 2: mib}), (2, 1, {0: 0, 1: mib, 2: mib})]
    first = [(0, 1, {0: 9 * mib, 1: 9 * mib}), (1, 0, {0: mib, 1: mib})]
    alone = [(0, 2, {0: 12 * mib, 2: 12 * mib}), (2, 0, {0: 4 * mib, 2: 4 * mib})]
    bases, connections = communicators.measured_connections([every, first, alone])
    assert bases == {3: 8 * mib, 2: 8 * mib}
    assert connections[(0, 2)] == connections[(2, 0)] == (4 * mib, 4 * mib, 6 * mib)
    needs = Needs(collective={}, exchanged={}, permuted=bases, connections=connections, runtime={},
                  seconds=1.0)
    assert needs.needed(Collectives(permutes={(0, 2): frozenset({(0, 2), (2, 0)})}), range(3),
                        runtime=False) == {0: 12 * mib + 4 * mib, 1: 0, 2: 12 * mib + 4 * mib}


def test_a_hosts_gpus_are_counted_by_its_cuda_driver_without_nvidia_smi(tmp_path):
    """A container whose CUDA driver makes two GPUs visible and that ships no nvidia-smi counts both, so its
    pool is sized for their communicators; one whose driver finds no GPU counts none."""
    compiler = shutil.which("cc")
    assert compiler is not None, "the stand-in driver library is built by the host's C compiler, cc"
    for found, status, expected in ((2, 0, 2), (0, 100, 0)):
        directory = tmp_path / str(found)
        directory.mkdir()
        (directory / "libcuda.c").write_text(
            f"int cuInit(unsigned flags) {{ return {status}; }}\n"
            f"int cuDeviceGetCount(int *count) {{ *count = {found}; return 0; }}\n")
        subprocess.run([compiler, "-shared", "-fPIC", "-o", str(directory / "libcuda.so.1"),
                        str(directory / "libcuda.c")], check=True)
        environment = {name: value for name, value in os.environ.items() if name != "CUDA_VISIBLE_DEVICES"}
        counted = "from dew._pool import local_gpu_count; print(local_gpu_count())"
        done = subprocess.run([sys.executable, "-c", counted], capture_output=True, text=True, check=True,
                              env={**environment, "PATH": str(directory), "PYTHONPATH": str(SOURCE),
                                   "LD_LIBRARY_PATH": str(directory)})
        assert done.stdout.strip() == str(expected)


def test_the_gpus_counted_are_those_jax_opens_where_its_own_setting_selects_some(tmp_path):
    """JAX's own selection of a process's GPUs (`jax_cuda_visible_devices`, set in the process) is what the
    process opens: on a host whose driver finds four, a process selecting two measures its communicators'
    room on those two, by the driver's ordinals (its configuration's places, which its pool's sizing reads),
    and one selecting none on every GPU."""
    compiler = shutil.which("cc")
    assert compiler is not None, "the stand-in driver library is built by the host's C compiler, cc"
    # A driver of four GPUs, each placed at a PCI bus id of its ordinal.
    (tmp_path / "libcuda.c").write_text(
        "int cuInit(unsigned flags) { return 0; }\n"
        "int cuDeviceGetCount(int *count) { *count = 4; return 0; }\n"
        "int cuDeviceGet(int *device, int ordinal) { *device = ordinal; return 0; }\n"
        "int cuDeviceGetName(char *name, int size, int device) { name[0] = 'G'; name[1] = 0; return 0; }\n"
        "int cuDeviceGetPCIBusId(char *id, int size, int device) {\n"
        "  id[0] = '0' + device; id[1] = 0; return 0; }\n"
        "int cuDriverGetVersion(int *version) { *version = 13000; return 0; }\n")
    subprocess.run([compiler, "-shared", "-fPIC", "-o", str(tmp_path / "libcuda.so.1"),
                    str(tmp_path / "libcuda.c")], check=True)
    environment = {name: value for name, value in os.environ.items() if name != "CUDA_VISIBLE_DEVICES"}
    environment.update(PATH=str(tmp_path), PYTHONPATH=str(SOURCE), LD_LIBRARY_PATH=str(tmp_path),
                       JAX_PLATFORMS="cpu")
    counted = ("import jax, sys\n"
               "if len(sys.argv) > 1: jax.config.update('jax_cuda_visible_devices', sys.argv[1])\n"
               "from dew.communicators import configuration\n"
               "import json; print(json.dumps(configuration()['placement']))\n")
    for setting, expected in ((None, ["0", "1", "2", "3"]), ("1,2", ["1", "2"]), ("3", ["3"])):
        done = subprocess.run([sys.executable, "-c", counted, *([setting] if setting else [])],
                              capture_output=True, text=True, check=True, env=environment)
        assert json.loads(done.stdout.strip().splitlines()[-1]) == expected


def test_a_configuration_names_the_xla_nccl_flags_the_build_and_the_gpus_places(monkeypatch):
    """A host's configuration holds each XLA flag that sets NCCL's communicators (their channels), the JAX
    build that makes them and each GPU's place, as well as NCCL's own settings."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_nccl_collective_max_nchannels=4 --xla_dump_to=/tmp/x")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "0")
    configured = communicators.configuration()
    assert configured["xla_nccl"] == ["--xla_gpu_nccl_collective_max_nchannels=4"]
    assert configured["nccl_env"]["NCCL_NVLS_ENABLE"] == "0"
    assert configured["jax"] == jax.__version__


def test_a_host_of_one_gpu_or_none_or_a_cpu_process_measures_nothing(monkeypatch):
    """A host with fewer than two GPUs makes no communicator, nor does a process that asks JAX for its CPU
    alone, so neither measures anything."""
    assert communicators.needs_of(gpus=1) is None
    assert communicators.needs_of(gpus=0) is None
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    assert communicators.needs_of(gpus=2) is None


@pytest.mark.parametrize(("set_by_user", "expected"), [(None, "0"), ("1", "1")])
def test_import_dew_connects_nccls_peers_at_creation_unless_told_otherwise(set_by_user, expected):
    """`import dew` has NCCL connect its rings, trees and NVLS peers as it creates a communicator
    (NCCL_RUNTIME_CONNECT=0), where XLA returns an allocation's failure to every rank; a value the user set
    is theirs."""
    env = {key: value for key, value in os.environ.items() if key != "NCCL_RUNTIME_CONNECT"}
    if set_by_user is not None:
        env["NCCL_RUNTIME_CONNECT"] = set_by_user
    script = "import json, os\nimport dew\nprint(json.dumps(os.environ.get('NCCL_RUNTIME_CONNECT')))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True,
                          env={**env, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)})
    assert json.loads(done.stdout.strip().splitlines()[-1]) == expected

def a_program_on_this_gpu_is_admitted_against_the_room_its_pool_left(tmp_path):
    """A GPU flow's cell (tests/test_gpu_training.py): a program compiled through Dew's compile point
    (`options.program`) on this host's GPU is admitted against the sizing the process holds: refused before
    it runs, naming the GPU's bytes, where the room left outside XLA's pool holds less than the runtime
    needs, and admitted with its code held to what the GPU has free where the room holds it."""
    import jax.numpy as jnp

    from dew.nn.kernels import options

    gpu = jax.devices("gpu")[0]
    ordinal = communicators.ordinal(gpu)
    needs = Needs(collective={}, exchanged={}, permuted={}, connections={}, runtime={ordinal: 2**20},
                  seconds=1.0)
    rows = jax.device_put(jnp.ones(8), gpu)
    held = communicators.held()
    try:
        communicators.hold(communicators.Sizing(needs, 0.95, 0.9, {ordinal: 1}, {ordinal: 1}))
        with pytest.raises(CommunicatorRoom) as refused:
            options.program(lambda x: x * 3.0).compiled(rows)
        assert refused.value.room == {ordinal: 1}
        communicators.hold(communicators.Sizing(needs, 0.95, 0.9, {ordinal: GIB}, {ordinal: GIB}))
        assert float(options.program(lambda x: x * 5.0)(rows)[0]) == 5.0
    finally:
        communicators.hold(held)


COMPILED_ON_GPU = """\
import json

import jax
import jax.numpy as jnp

events = []
jax.monitoring.register_event_listener(lambda event, **_: events.append(event))
from dew import communicators
from dew.communicators import Needs
from dew.nn.kernels import options


def tripled(rows):
    return rows * 3.0


gpu = jax.devices("gpu")[0]
ordinal = communicators.ordinal(gpu)
needs = Needs(collective={}, exchanged={}, permuted={}, connections={}, runtime={ordinal: 2**20}, seconds=1.0)
communicators.hold(communicators.Sizing(needs, 0.95, 0.9, {ordinal: 2**30}, {ordinal: 2**30}))
assert float(options.program(tripled)(jax.device_put(jnp.ones(8), gpu))[0]) == 3.0
print(json.dumps([events.count("/jax/compilation_cache/compile_requests_use_cache"),
                  events.count("/jax/compilation_cache/cache_hits")]))
"""
"""A process that compiles a program on its GPU through Dew's compile point under a sizing that admits it,
printing its compilation-cache requests and hits."""


def a_program_loaded_from_the_compilation_cache_is_admitted_from_its_kept_communicators(tmp_path):
    """A GPU flow's cell (tests/test_gpu_training.py): a program a process compiled and admitted on its GPU
    keeps its communicators in Dew's cache under its executable's fingerprint (`collectives_kept`), and a
    second process that loads it from the compilation cache, every request a hit, admits it from that
    record as it is, the file neither rewritten nor replaced (row 3316, 3474)."""
    cache, compiled = tmp_path / "cache", tmp_path / "xla"
    environment = {**os.environ, "XDG_CACHE_HOME": str(cache), "JAX_COMPILATION_CACHE_DIR": str(compiled),
                   "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
                   "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "0"}
    kept = cache / "dew" / "communicators" / "programs" / package_digest()

    def ran() -> tuple[list[int], dict[str, tuple[int, int]]]:
        done = subprocess.run([sys.executable, "-c", COMPILED_ON_GPU], check=True, env=environment,
                              capture_output=True, text=True)
        requests = json.loads(done.stdout.strip().splitlines()[-1])
        return requests, {path.name: (path.stat().st_ino, path.stat().st_mtime_ns) for path in kept.iterdir()}

    _, written = ran()
    (requests, hits), read = ran()
    assert len(written) == 1 and read == written
    assert requests and hits == requests
