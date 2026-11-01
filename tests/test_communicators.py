"""The room NCCL's communicators take outside XLA's pool (ledger 2901): measured once per configuration by a
child process before the backend opens and kept in Dew's cache, the pool's share held to leave it at NCCL's
full speed, the run's share its upper bound, and a compiled program whose communicators need more room than
Dew left them refused before it runs, never left to stall in a collective. The measurement itself needs two
GPUs, which a two-GPU run of the check makes; here it is the child's report, as the parent reads it."""

import json
import os
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from dew.nn.kernels import options
from dew.training import communicators
from dew.training.communicators import Collectives, CommunicatorRoom, Needs

SOURCE = Path(__file__).resolve().parents[1] / "src"
GIB = 2**30

NEEDS = Needs(collective={2: GIB}, exchanged={2: GIB // 4}, permuted={2: (2 * GIB, 2 * GIB)},
              runtime={0: GIB // 2, 1: GIB // 2}, seconds=4.0)
"""Two GPUs' measurement: a collective's communicator over both takes 1 GiB on each, an all-to-all's
connections to its peer a quarter more, a permute's communicator to its one peer 2 GiB, the runtime half."""

EVERY = Collectives(groups=frozenset({(0, 1)}), exchanges=frozenset({(0, 1)}),
                    peers={0: frozenset({1}), 1: frozenset({0})})
"""What Dew leaves room for before a program compiles: a collective's communicator over both GPUs, its
all-to-all's connections to every peer, and a permute's to every peer."""

MEMORY = {0: (79 * GIB, 80 * GIB), 1: (76 * GIB, 80 * GIB)}
"""Each GPU's free and total bytes before the backend opens: 79 and 76 of 80 GiB free."""


@pytest.fixture
def unsized(monkeypatch):
    """The module's sized pool and resident communicators, put back as they were after the test."""
    monkeypatch.setattr(communicators, "NEEDS", None)
    monkeypatch.setattr(communicators, "SIZED", None)
    monkeypatch.setattr(communicators, "RESIDENT", communicators.Resident())


def test_the_pool_takes_at_most_the_runs_share_and_leaves_the_communicators_room():
    """The pool's share of each GPU is the run's (`dew.devices.pool_bound`) as an upper bound: it leaves the
    room its communicators and the runtime take outside the pool at NCCL's full speed, the tightest GPU
    deciding, to a thousandth below, and NCCL is never slowed to fit. A pool the communicators leave no room
    for at all is refused, naming each side's bytes."""
    tightest = 76 * GIB - (GIB + GIB // 4 + 2 * GIB + GIB // 2)
    for bound in (0.95, 0.97):
        share = communicators.pool_share(NEEDS, MEMORY, bound, EVERY)
        assert share * 80 * GIB <= tightest < (share + 0.001) * 80 * GIB
    roomy = {0: (79 * GIB, 80 * GIB), 1: (79 * GIB, 80 * GIB)}
    assert communicators.pool_share(NEEDS, roomy, 0.75, EVERY) == 0.75
    with pytest.raises(CommunicatorRoom) as refused:
        communicators.pool_share(NEEDS, {0: (3 * GIB, 80 * GIB)}, 0.97, EVERY)
    assert refused.value.needed == {0: 3 * GIB + GIB // 4 + GIB // 2} and refused.value.room == {0: 3 * GIB}


@pytest.mark.parametrize(("named", "bound"), [
    ({}, 0.95), ({"XLA_PYTHON_CLIENT_MEM_FRACTION": "0.97"}, 0.97),
    ({"XLA_PYTHON_CLIENT_PREALLOCATE": "true"}, 0.75), ({"XLA_PYTHON_CLIENT_ALLOCATOR": "bfc"}, 0.75),
    ({"XLA_PYTHON_CLIENT_PREALLOCATE": "false"}, None), ({"XLA_PYTHON_CLIENT_ALLOCATOR": "platform"}, None)])
def test_one_owner_reads_the_pools_bound_from_how_the_run_named_it(named, bound, monkeypatch):
    """The pool's upper bound, as the run names it: Dew's 95% where it names nothing, its fraction where it
    names one, JAX's own 75% where it names preallocation or the allocator without one, and none where it
    holds no fixed pool (a pool that grows, the platform's allocator), which Dew neither sizes nor admits
    programs against."""
    for name in ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE",
                 "XLA_PYTHON_CLIENT_ALLOCATOR"):
        monkeypatch.delenv(name, raising=False)
    for name, value in named.items():
        monkeypatch.setenv(name, value)
    assert communicators.pool_bound(cuda=True) == bound
    assert communicators.pool_bound(cuda=False) is None


def test_a_programs_communicators_are_charged_by_their_groups_exchanges_and_peers():
    """Each GPU is charged for every distinct group its collectives take, at that group's size; for the
    connections an all-to-all makes to every peer of its group; and for the permute peers it sends to or
    receives from, one peer's communicator and each further peer's connections, measured at one peer and at
    every peer (each NCCL peer connection holds buffers of its own)."""
    four = Needs(collective={2: GIB, 4: 2 * GIB}, exchanged={2: GIB // 4, 4: GIB // 2},
                 permuted={4: (GIB, 3 * GIB)}, runtime=dict.fromkeys(range(4), 0), seconds=8.0)
    held = Collectives(groups=frozenset({(0, 1), (2, 3), (0, 1, 2, 3)}), exchanges=frozenset({(0, 1)}),
                       peers={0: frozenset({1, 2}), 1: frozenset({0})})
    assert four.needed(held, range(4), runtime=False) == {
        0: GIB + 2 * GIB + GIB // 4 + 2 * GIB, 1: GIB + 2 * GIB + GIB // 4 + GIB,
        2: GIB + 2 * GIB, 3: GIB + 2 * GIB}


def test_a_compiled_programs_communicators_are_read_through_the_flop_counts_parser():
    """A compiled program's communicators, read by the parser that counts its FLOPs: each distinct group of
    its collectives however XLA prints it, listed, as an iota or over a mesh's named axes (an fsdp x
    sequence step's two groupings are two), an all-to-all's groups, and each GPU's permute peers, the
    module's positions read as the GPUs its device assignment names."""
    program = ("HloModule step, entry_computation_layout={()->f32[8]}\n\nENTRY %main () -> f32[8] {\n"
               "  %x = f32[8] parameter(0)\n"
               "  %a = f32[8] all-gather(f32[8] %x), replica_groups={{0,1},{2,3}}, dimensions={0}\n"
               "  %b = f32[8] all-reduce(f32[8] %x), replica_groups=[2,2]<=[4], to_apply=%add\n"
               "  %c = f32[8] all-reduce(f32[8] %x), replica_groups=mesh['fsdp'=2,'sequence'=2] {'fsdp'}, "
               "to_apply=%add\n"
               "  %d = f32[8] all-to-all(f32[8] %x), replica_groups={}, dimensions={0}\n"
               "  ROOT %e = f32[8] collective-permute(f32[8] %x), source_target_pairs={{0,1},{1,2}}\n}\n")
    gpus = (3, 2, 1, 0)
    assert communicators.collectives_of(program, gpus) == Collectives(
        groups=frozenset({(3, 2), (1, 0), (3, 1), (2, 0), (3, 2, 1, 0)}),
        exchanges=frozenset({(3, 2, 1, 0)}),
        peers={3: frozenset({2}), 2: frozenset({3, 1}), 1: frozenset({2})})


def test_a_processs_communicators_are_held_together_to_the_room_dew_left_outside_the_pool():
    """Every communicator a process has made stays made (XLA caches them), so each compiled program is
    admitted only if its communicators and those of the programs before it, each counted once, and the
    runtime fit the room Dew left outside the pool; one beyond it is refused before it runs, naming the
    bytes, never left to stall in a collective."""
    four = Needs(collective={2: GIB, 4: 2 * GIB}, exchanged={2: GIB // 4, 4: GIB // 2},
                 permuted={4: (GIB, 3 * GIB)}, runtime=dict.fromkeys(range(4), GIB // 2), seconds=8.0)
    room = dict.fromkeys(range(4), 3 * GIB + 3 * GIB // 4)
    every = (0, 1, 2, 3)
    resident = communicators.Resident()
    resident.admit(Collectives(groups=frozenset({every, (0, 1), (2, 3)})), four, room)
    resident.admit(Collectives(groups=frozenset({every})), four, room)
    with pytest.raises(CommunicatorRoom) as refused:
        resident.admit(Collectives(exchanges=frozenset({every})), four, room)
    assert refused.value.needed == dict.fromkeys(range(4), 2 * GIB + GIB + GIB // 2 + GIB // 2)
    assert refused.value.room == room
    assert resident.held == Collectives(groups=frozenset({every, (0, 1), (2, 3)}))


def test_every_program_dew_compiles_is_admitted_where_it_sized_the_pool(unsized, monkeypatch):
    """A program compiled through Dew's one compile point (`dew.nn.kernels.templates.report.compiled`, which
    every outer program, step and evaluation passes) is admitted against the room Dew left; a pool it never
    sized admits nothing."""
    mesh = Mesh(np.array(jax.devices()[:2]), ("pair",))
    summed = jax.shard_map(lambda part: jax.lax.psum(part, "pair"), mesh=mesh,
                           in_specs=PartitionSpec("pair"), out_specs=PartitionSpec())
    ids = tuple(communicators.ordinal(device) for device in mesh.devices.flat)
    tight = Needs(collective={2: GIB}, exchanged={}, permuted={}, runtime=dict.fromkeys(ids, 0), seconds=1.0)
    options.program(summed).compiled(jnp.ones(4))
    monkeypatch.setattr(communicators, "NEEDS", tight)
    monkeypatch.setattr(communicators, "SIZED", {"room": dict.fromkeys(ids, GIB // 2)})
    with pytest.raises(CommunicatorRoom):
        options.program(summed).compiled(jnp.ones(4))


def test_the_sized_pool_is_recorded_with_what_it_left_room_for(unsized, monkeypatch):
    """A pool Dew sized is recorded as the run's bound, the share it took and each GPU's room it left, which
    the fit check's record carries into the run record."""
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.97")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    communicators.sized_pool(measure=lambda: NEEDS, memories=lambda: MEMORY, cuda=True, opened=False)
    sized = communicators.SIZED
    assert sized is not None and sized["bound"] == 0.97
    assert float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]) == sized["share"] < 0.97
    assert sized["needed"] == NEEDS.needed(EVERY, MEMORY, runtime=True)


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


def test_a_configuration_names_the_xla_nccl_flags_the_build_and_the_gpus_places(monkeypatch):
    """A host's configuration holds each XLA flag that sets NCCL's communicators (their channels), the JAX
    build that makes them and each GPU's place, as well as NCCL's own settings."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_nccl_collective_max_nchannels=4 --xla_dump_to=/tmp/x")
    monkeypatch.setenv("NCCL_NVLS_ENABLE", "0")
    configured = communicators.configuration()
    assert configured["xla_nccl"] == ["--xla_gpu_nccl_collective_max_nchannels=4"]
    assert configured["nccl_env"] == {"NCCL_NVLS_ENABLE": "0"}
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
