"""The choice layer's features (dew.nn.kernels.templates.features): the analytic estimate reproduced from a
row's legacy inputs bitwise, the predictions derived from them and the target's facts, a launch without a
costed record left without features."""

import dataclasses
import functools
import json

import jax
import jax.numpy as jnp
import pytest

from dew.nn.kernels import generation, grouped_matmul
from dew.nn.kernels.templates import base, cost, features

ROWS, DEPTH, WIDTH, GROUPS = 4096, 1024, 768, 8


def _launches():
    bf16 = functools.partial(jax.ShapeDtypeStruct, dtype=jnp.bfloat16)
    sizes = jax.ShapeDtypeStruct((GROUPS,), jnp.int32)
    forward = {"out_dtype": "bfloat16", "trans_rhs": False, "interpret": False}
    for launch in (grouped_matmul.Gmm(), grouped_matmul.MosaicGmm()):
        yield launch, (bf16((ROWS, DEPTH)), bf16((GROUPS, DEPTH, WIDTH)), sizes, (), ()), forward
    for launch in (grouped_matmul.Tgmm(), grouped_matmul.MosaicTgmm()):
        yield launch, (bf16((ROWS, DEPTH)), bf16((ROWS, WIDTH)), sizes, ()), {"out_dtype": "float32",
                                                                              "interpret": False}


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


def test_the_predictions_follow_from_the_legacy_inputs_and_the_facts():
    """An A100's 108 multiprocessors, one program each, hold 216 programs in two waves with no tail; a
    launch of 1 GFLOP over 1 MB is 1000 FLOPs a byte, against the A100's balance."""
    with generation.assumed("sm80"):
        target = generation.gpu_target()
    facts = generation.effective(target)
    legacy = cost.Cost(programs=216, device_bytes=1e6, program_flops=1e9 / 216, steps=8, stages=3)

    class Costed:
        def costed(self, ba, config, target):
            return legacy

    found = features.features(Costed(), None, None, target)
    balance = facts.tensor_rate('16bit') * facts.multiprocessors * facts.clock_hz / facts.memory_bandwidth
    assert found.predictions['waves'] == 2 and found.predictions['tail'] == 0
    assert found.predictions['intensity'] == 1e9 / 1e6
    assert found.predictions['balance'] == balance and found.predictions['hidden_steps'] == 2
    assert found.geometry == {}


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
