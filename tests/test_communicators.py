"""The room NCCL's communicators take outside XLA's pool (ledger 2901): measured once per configuration by a
child process before the backend opens and kept in Dew's cache, the pool's share held to leave it at NCCL's
full speed, the user's share its upper bound, and a compiled step whose collectives need more room than its
GPUs have outside the pool refused before it runs, never left to stall in a collective. The measurement
itself needs two GPUs, which a two-GPU run of the check makes; here it is the child's report, as the parent
reads it."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dew.training import communicators
from dew.training.communicators import CommunicatorRoom, Needs

SOURCE = Path(__file__).resolve().parents[1] / "src"
GIB = 2**30

NEEDS = Needs(communicators={0: {2: GIB}, 1: {2: 2 * GIB}}, runtime={0: GIB // 2, 1: GIB // 2}, seconds=4.0)
"""Two GPUs' measurement: a collective's communicator over both takes 1 GiB on each, a permute's 2, the
runtime half."""

BOTH = frozenset({(0, "all", 2), (1, "all", 2)})
"""A collective's and a permute's communicator over both GPUs, which Dew leaves room for before a step
compiles."""

MEMORY = {0: (79 * GIB, 80 * GIB), 1: (76 * GIB, 80 * GIB)}
"""Each GPU's free and total bytes before the backend opens: 79 and 76 of 80 GiB free."""


def test_the_pool_takes_at_most_the_users_share_and_leaves_the_communicators_room():
    """The pool's share of each GPU is the user's (XLA_PYTHON_CLIENT_MEM_FRACTION, Dew's 95% where they named
    none) as an upper bound: it leaves the room its communicators and the runtime take outside the pool at
    NCCL's full speed, the tightest GPU deciding, to a thousandth below, and NCCL is never slowed to fit. A
    pool the communicators leave no room for at all is refused, naming each side's bytes."""
    tightest = 76 * GIB - (3 * GIB + GIB // 2)
    for requested in (None, 0.97):
        share = communicators.pool_share(NEEDS, MEMORY, requested=requested, cliques=BOTH)
        assert share * 80 * GIB <= tightest < (share + 0.001) * 80 * GIB
    roomy = {0: (79 * GIB, 80 * GIB)}
    assert communicators.pool_share(NEEDS, roomy, requested=None, cliques=frozenset({(0, "all", 2)})) == 0.95
    assert communicators.pool_share(NEEDS, MEMORY, requested=0.5, cliques=BOTH) == 0.5
    with pytest.raises(CommunicatorRoom) as refused:
        communicators.pool_share(NEEDS, {0: (3 * GIB, 80 * GIB)}, requested=0.97, cliques=BOTH)
    assert refused.value.needed == {0: 3 * GIB + GIB // 2} and refused.value.room == {0: 3 * GIB}


def test_a_measurement_is_paid_once_per_configuration(tmp_path, monkeypatch):
    """A configuration's measurement is kept in Dew's cache and read back without a child process; another
    NCCL setting is another configuration, measured afresh."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    taken = []

    def measure(configuration):
        taken.append(configuration)
        return NEEDS

    configuration = {"gpus": ["NVIDIA H100 80GB HBM3"] * 2, "driver": 12090, "nccl": 23203, "nccl_env": {}}
    assert communicators.measured(configuration, measure=measure) == NEEDS
    assert communicators.measured(configuration, measure=measure) == NEEDS
    other = {**configuration, "nccl_env": {"NCCL_NVLS_ENABLE": "0"}}
    assert communicators.measured(other, measure=measure) == NEEDS
    assert taken == [configuration, other]


def test_the_sized_pool_is_recorded_with_what_it_left_room_for(monkeypatch):
    """A pool Dew sized is recorded as the user's bound, the share it took and each GPU's room it left, which
    the fit check's record carries into the run record."""
    monkeypatch.setenv("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.97")
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    communicators.sized_pool(measure=lambda: NEEDS, memories=lambda: MEMORY)
    sized = communicators.SIZED
    assert sized is not None and sized["requested"] == 0.97
    assert float(os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]) == sized["share"] < 0.97
    assert sized["needed"] == NEEDS.needed(BOTH, MEMORY, runtime=True)


def test_a_compiled_steps_communicators_are_read_by_kind_and_device_grouping():
    """A compiled step makes a communicator for each kind and device grouping its collectives take (XLA keys a
    clique by its devices and its kind): two all-gathers over one grouping share one, another grouping of the
    same size makes another, and a permute makes its own."""
    program = ("%a = f32[8] all-gather(%x), replica_groups={{0,1},{2,3}}\n"
               "%b = f32[8] all-reduce(%y), replica_groups={{0,1},{2,3}}\n"
               "%c = f32[8] all-reduce(%z), replica_groups={{0,2},{1,3}}\n"
               "%d = f32[8] all-to-all(%w), replica_groups=[1,4]<=[4]\n"
               "%e = f32[8] collective-permute(%v), source_target_pairs={{0,1},{1,0}}\n"
               "%f = f32[8] add(%u, %t)\n")
    assert communicators.cliques_of(program, 4) == frozenset({
        (0, "{{0,1},{2,3}}", 2), (0, "{{0,2},{1,3}}", 2), (0, "[1,4]<=[4]", 4), (1, "all", 4)})
    assert communicators.cliques_of("%f = f32[8] add(%u, %t)", 4) == frozenset()


def test_a_compiled_steps_communicators_are_held_to_the_room_dew_left_outside_the_pool():
    """A compiled step whose communicators need more than the room Dew left outside the pool is refused before
    it runs, naming the bytes, never left to stall in a collective; one within it passes, its communicators
    already made by earlier steps counted against that room, not against what is free after them. A
    grouping the host was not measured for is charged as its size's measured communicator."""
    room = {0: 4 * GIB, 1: 4 * GIB}
    communicators.checked_room(BOTH, NEEDS, room)
    with pytest.raises(CommunicatorRoom) as refused:
        communicators.checked_room(BOTH | {(0, "{{0,1}}", 2)}, NEEDS, room)
    assert refused.value.needed == {0: 4 * GIB + GIB // 2, 1: 4 * GIB + GIB // 2} and refused.value.room == room


def test_a_host_of_one_gpu_or_none_or_a_cpu_process_measures_nothing(monkeypatch):
    """A host with fewer than two GPUs makes no communicator, nor does a process that asks JAX for its CPU
    alone, so neither measures anything and its share stays Dew's own."""
    assert communicators.needs_of(gpus=1) is None
    assert communicators.needs_of(gpus=0) is None
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    assert communicators.needs_of(gpus=2) is None


@pytest.mark.parametrize(("set_by_user", "expected"), [(None, "0"), ("1", "1")])
def test_import_dew_connects_nccls_peers_at_creation_unless_told_otherwise(set_by_user, expected):
    """`import dew` has NCCL connect every peer as it creates a communicator (NCCL_RUNTIME_CONNECT=0), where
    XLA returns an allocation's failure to every rank; a value the user set is theirs."""
    env = {key: value for key, value in os.environ.items() if key != "NCCL_RUNTIME_CONNECT"}
    if set_by_user is not None:
        env["NCCL_RUNTIME_CONNECT"] = set_by_user
    script = "import json, os\nimport dew\nprint(json.dumps(os.environ.get('NCCL_RUNTIME_CONNECT')))\n"
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True,
                          env={**env, "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(SOURCE)})
    assert json.loads(done.stdout.strip().splitlines()[-1]) == expected
