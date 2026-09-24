"""Every model family on every objective class it takes, split by the layouts
that split it differently, against one device on the eight-device CPU mesh.

tools/layout_parity.py judges a layout's step-one loss and gradient against
one device's, each leaf within FLOOR_FACTOR of the reference's own deviation
under reassociating its sums (its rows reordered and pooled, and the step's
distance from fp64), and holds the layout's FLOPs to an even split. A cell
works, or is refused by design with its reason, as a stage axis over a model
that runs no pipeline is; any other status is a defect. The models are the
tool's zoo, sized so every layout divides them. Each row lists the layouts
that split its model in a way no other row covers: a pair's second axis, a
packed or masked column under a sequence split, a bidirectional exchange, a
vision tower beside a pipelined decoder."""

import jax
import pytest
from test_tools import load

CELLS: dict[str, tuple[str, ...]] = {
    "dense_sft": ("fsdp4", "tensor2_sequence2"),
    "dense_dpo": ("fsdp4", "sequence4"),
    "dense_grpo": ("tensor4", "sequence4"),
    "dense_mdlm": ("sequence4", "fsdp2_tensor2"),
    "moe_grpo": ("expert4", "expert2_fsdp2"),
    "mla_dpo": ("fsdp4", "sequence4"),
    "hybrid_sft": ("sequence4", "stage4"),
    "mmdit": ("tensor4", "fsdp2_tensor2"),
    "dit": ("sequence4", "tensor4"),
    "jepa": ("fsdp4", "sequence4"),
    "multimodal": ("sequence4", "stage4"),
}


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize("model", sorted(CELLS))
def test_each_layout_of_a_model_matches_one_device_or_is_refused(model):
    tool = load("layout_parity")
    with jax.enable_x64(True):
        rows = tool.run([model], CELLS[model], dtype="float32", steps=1, anchor=True, mixture={},
                        objective={}, references=tool.References(), speak=lambda line: None,
                        keep=lambda rows: None)
    assert tool.verdict(rows) == 0, [
        {key: row.get(key) for key in ("layout", "status", "worst_leaf", "worst_ratio", "loss_error",
                                       "loss_bound", "flops_ratio", "reason", "error")}
        for row in rows]
