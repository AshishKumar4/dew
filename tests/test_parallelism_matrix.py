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
vision tower beside a pipelined decoder, a tensor or sequence axis over
experts.

The tool draws every parameter an initializer leaves all zeros, so a
DiT's zeroed output and modulations do not hide the layers above them
from the first step's gradient (2 of its 70 leaves were judged before).

A layout's data axis takes the devices its other axes leave, so on eight
devices every pipeline below splits the rows over data. The rows of
CELLS_ON_FOUR run on four of the eight, where four stages, or two beside a
sequence axis of two, leave no axis for the rows: every device holds all of
them."""

import jax
import pytest
from test_tools import load

CELLS: dict[str, tuple[str, ...]] = {
    "dense_sft": ("fsdp4", "tensor2_sequence2"),
    "dense_dpo": ("fsdp4", "sequence4"),
    "dense_grpo": ("tensor4", "sequence4"),
    "dense_mdlm": ("sequence4", "fsdp2_tensor2"),
    "moe_grpo": ("expert4", "expert2_fsdp2", "tensor4", "sequence4"),
    "mla_dpo": ("fsdp4", "sequence4"),
    "hybrid_sft": ("sequence4", "stage4"),
    "mmdit": ("tensor4", "fsdp2_tensor2"),
    "dit": ("sequence4", "tensor4"),
    "unet": ("tensor4", "fsdp2_tensor2", "sequence4"),
    "jepa": ("fsdp4", "sequence4", "tensor4"),
    "multimodal": ("sequence4", "stage4"),
}
CELLS_ON_FOUR: dict[str, tuple[str, ...]] = {
    "dense": ("stage4", "stage2_sequence2"),
}


@pytest.mark.mesh(devices=8)
@pytest.mark.parametrize(("model", "layouts", "devices"), [
    *(pytest.param(model, layouts, None, id=model) for model, layouts in sorted(CELLS.items())),
    *(pytest.param(model, layouts, 4, id=f"{model}-on-four")
      for model, layouts in sorted(CELLS_ON_FOUR.items()))])
def test_each_layout_of_a_model_matches_one_device_or_is_refused(model, layouts, devices):
    tool = load("layout_parity")
    with jax.enable_x64(new_val=True):
        rows = tool.run([model], layouts, dtype="float32", steps=1, anchor=True, mixture={},
                        objective={}, references=tool.References(), speak=lambda line: None,
                        keep=lambda rows: None, devices=devices)
    assert tool.verdict(rows) == 0, [
        {key: row.get(key) for key in ("layout", "status", "worst_leaf", "worst_ratio", "loss_error",
                                       "loss_bound", "flops_ratio", "reason", "error")}
        for row in rows]
