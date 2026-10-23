"""The choice layer's features (dew.nn.kernels.templates.features): the analytic estimate reproduced from a
row's legacy inputs bitwise, the predictions derived from them and the target's facts, a launch without a
costed record left without features."""

import dataclasses
import functools
import json

import jax
import jax.numpy as jnp
import pytest
from test_gemm_template import biased, normed

from dew.nn.kernels import generation, grouped_matmul
from dew.nn.kernels.templates import base, cost, features

ROWS, DEPTH, WIDTH, GROUPS = 4096, 1024, 768, 8


def _launches():
    """The grouped launches, plain, with a norm prologue and bias epilogue, and summing exactly."""
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    f32 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.float32)
    sizes = jax.ShapeDtypeStruct((GROUPS,), jnp.int32)
    hooks = grouped_matmul.Hooks(normed, (("rows",), ("depth",)), biased, (("columns",),))
    prologue = (f32((ROWS,)), f32((DEPTH,)))
    forward = {"out_dtype": "bfloat16", "trans_rhs": False, "interpret": False}
    gradient = {"out_dtype": "float32", "interpret": False}
    for launch in (grouped_matmul.Gmm(), grouped_matmul.MosaicGmm()):
        yield launch, (bf16((ROWS, DEPTH)), bf16((GROUPS, DEPTH, WIDTH)), sizes, (), ()), forward
        hooked = (bf16((ROWS, DEPTH)), bf16((GROUPS, DEPTH, WIDTH)), sizes, prologue, (f32((WIDTH,)),))
        yield launch, hooked, {**forward, "hooks": hooks}
    for launch in (grouped_matmul.Tgmm(), grouped_matmul.MosaicTgmm()):
        yield launch, (bf16((ROWS, DEPTH)), bf16((ROWS, WIDTH)), sizes, ()), gradient
        yield launch, (bf16((ROWS, DEPTH)), bf16((ROWS, WIDTH)), sizes, ()), {**gradient, "exact_sums": True}


@pytest.mark.parametrize("generation_name", ["sm80", "sm90"])
def test_a_rows_legacy_inputs_give_the_estimate_bitwise(generation_name):
    """For every fitting config of the grouped launches, the estimate (`Template.cost`) is `cost.seconds`
    of the row's legacy inputs, to the bit, and the row is plain JSON."""
    with generation.assumed(generation_name):
        target = generation.gpu_target()
        for launch, args, options in _launches():
            ba = launch.bind(*args, **options)
            if not launch.eligible(ba, target):
                continue
            for config in launch.fitting(ba, target):
                found = features.features(launch, ba, config, target)
                assert cost.seconds(found.legacy, target) == launch.cost(ba, config, target)
                assert json.loads(json.dumps(found.row())) == found.row()


class _Costed(base.Template):
    """A template whose every config costs `legacy`."""
    def __init__(self, legacy: cost.Cost):
        super().__init__()
        object.__setattr__(self, 'legacy', legacy)

    def costed(self, ba, config, target):
        return self.legacy

    def _fwd(self, *args, **kwargs):
        raise AssertionError("never launched here")


def _sm80():
    with generation.assumed("sm80"):
        return generation.gpu_target()


def test_the_predictions_follow_from_the_legacy_inputs_and_the_facts():
    """An A100's 108 multiprocessors, one program each, hold 216 programs in two waves with no tail; a
    launch of 1 GFLOP over 1 MB is 1000 FLOPs a byte, against the A100's balance."""
    target = _sm80()
    facts = generation.effective(target)
    found = features.features(_Costed(cost.Cost(programs=216, device_bytes=1e6, program_flops=1e9 / 216,
                                                steps=8, stages=3)), None, None, target)
    balance = facts.tensor_rate('16bit') * facts.multiprocessors * facts.clock_hz / facts.memory_bandwidth
    assert found.predictions['waves'] == 2 and found.predictions['tail'] == 0
    assert found.predictions['intensity'] == 1e9 / 1e6
    assert found.predictions['balance'] == balance and found.predictions['hidden_steps'] == 2
    assert found.geometry == {}


def test_a_half_filled_last_wave_leaves_half_its_slots_idle():
    """162 programs on 108 slots: one full wave and one half full, whose idle share is a half, however
    many full waves come before it."""
    target = _sm80()
    for programs, waves in ((162, 1.5), (270, 2.5)):
        found = features.features(_Costed(cost.Cost(programs=programs, device_bytes=1.0)), None, None, target)
        assert found.predictions['waves'] == waves and found.predictions['tail'] == 0.5


def test_the_waiting_share_reads_the_estimates_service_time():
    """A program bound by shared memory, not its tensor work: the waiting share sets the waiting term
    beside that service time, `cost.program_seconds` as the estimate reads it."""
    target = _sm80()
    facts = generation.effective(target)
    legacy = cost.Cost(programs=108, device_bytes=1.0, program_flops=1.0, program_staged=1e9, steps=4,
                       stages=2)
    found = features.features(_Costed(legacy), None, None, target)
    program = cost.program_seconds(legacy, facts.tensor_rate('16bit'), facts.clock_hz)
    waiting = cost.waiting_seconds(legacy, facts.load_latency, facts.clock_hz)
    assert program == legacy.program_staged / (cost.SHARED_BYTES_PER_CLOCK * facts.clock_hz)
    assert found.predictions['waiting_share'] == waiting / (waiting + program)


def test_a_bulk_loaded_programs_waiting_share_follows_from_its_logged_facts():
    """A Hopper program staging by bulk copies waits each stage's bulk-copy latency and its L2 transfer:
    the waiting share reconstructs from the row's legacy inputs and the facts it logs."""
    target = generation.TARGETS["sm90"]
    legacy = cost.Cost(programs=132, device_bytes=1.0, program_flops=1.0, program_staged=1e9, steps=8,
                       stages=3, l2_bytes=132 * 8 * 49152.0, bulk_loads=True)
    found = features.features(_Costed(legacy), None, None, target)
    facts = found.facts
    clock, multiprocessors = facts['clock_hz'], facts['multiprocessors']
    stage = facts['stage_latency_clocks'] + 49152.0 / (facts['l2_bandwidth'] / (multiprocessors * clock))
    waiting = legacy.steps * stage / (legacy.stages - 1) / clock
    program = legacy.program_staged / (cost.SHARED_BYTES_PER_CLOCK * clock)
    assert found.predictions['waiting_share'] == pytest.approx(waiting / (waiting + program), rel=1e-12)


def test_a_launch_without_a_costed_record_has_no_features():
    """A kernel op that is no template, and a template costing by an override of `cost`, have none."""
    class Uncosted(base.Template):
        def cost(self, ba, config, target):
            return 1.0

        def _fwd(self, *args, **kwargs):
            raise AssertionError("never launched here")

    with generation.assumed("sm80"):
        target = generation.gpu_target()
    assert features.features(object(), None, None, target) is None
    assert features.features(Uncosted(), None, None, target) is None


def test_a_configs_fields_are_its_geometry():
    tile = grouped_matmul.Tile(128, 256, 64, 8, 3)
    with generation.assumed("sm80"):
        target = generation.gpu_target()
    ba = grouped_matmul.Gmm().bind(*next(_launches())[1], **next(_launches())[2])
    found = features.features(grouped_matmul.Gmm(), ba, tile, target)
    assert found.geometry == dataclasses.asdict(tile)
