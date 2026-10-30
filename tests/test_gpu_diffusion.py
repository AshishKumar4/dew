"""The image and diffusion models' GPU flow (G14, the owner's rule: no unit tests on GPUs): each one that
runs a GPU kernel of its own, run through it in one process (`gpu_flow`), each cell its own file's
property:

- MeanFlow trains through a fused attention kernel, its JVP through it (test_mean_flow).
- Qwen-Image's block attends a padded batch through cuDNN as XLA does (test_qwen_image_source).
"""

import pytest
import test_mean_flow as mean_flow
import test_qwen_image_source as qwen_image
from gpu_flow import Cell, run_cells

pytestmark = pytest.mark.gpu(reason="the models' fused attention kernels: flash and cuDNN")


def test_the_image_models_run_their_gpu_kernels(without_deterministic_ops):
    """Every image and diffusion cell above on this GPU (`gpu_flow.run_cells`), each one's failure kept
    and all raised together; the cuDNN cell runs without XLA's deterministic ops, which Dew refuses
    cuDNN attention under (`without_deterministic_ops`)."""
    run_cells([
        Cell("MeanFlow trains through a fused attention kernel",
             mean_flow.meanflow_trains_through_a_fused_attention_kernel),
        Cell("Qwen-Image attends a padded batch through cuDNN as XLA does",
             lambda: qwen_image.cudnn_attends_a_padded_batch_as_xla_does(without_deterministic_ops)),
    ])
