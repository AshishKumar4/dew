"""Kimi Linear against its released remote code.

kimi-linear-source holds the released config of
moonshotai/Kimi-Linear-48B-A3B-Instruct at e1df551 and the name, dtype and
shape of every tensor (read off the 20 shard headers). kimi-linear-tiny is
written by tools/kimi_linear_reference.py from the pinned modeling_kimi.py
with fla-core 0.4.0's kernels in fp32 on CUDA, its gate weighing the
released selection by the unbiased scores, as K3's revision of the file and
vLLM do. Its numerics.npz is the same released model evaluated in float64
on the CPU, fla's kernels answered by its own torch references
(tools/kimi_linear_reference.py truth), which Dew and the reference are both
measured from (tests/reference_error.py). No model weights are downloaded
at test time.
"""

from pathlib import Path

import numpy as np
import pytest
from kimi_support import assert_forward_as_exact, assert_greedy_as_exact, exported, kimi_source

ROOT = Path(__file__).parent / "fixtures" / "hf"
TINY = ROOT / "kimi-linear-tiny"


@pytest.fixture(scope="module")
def source():
    return kimi_source(TINY)


def test_forward_matches_the_reference_over_left_padding(source):
    """fp32 logits over 70 tokens (past one KDA chunk), one row left-padded
    by 9, against fla 0.4.0 on CUDA in IEEE fp32
    (tools/kimi_linear_reference.py): Dew's RMS distance from the float64
    logits at most twice the reference's, and argmax exact. The reference
    sits 1.04e-6 from float64 (RMS); Dew at 0.88 of that on CPU, 0.79 on an
    AVX-512 CPU (Intel SDE's Ice Lake server, the CI runners' ISA) and 0.68
    on an RTX 4080."""
    assert_forward_as_exact(source)


def test_update_exports_the_trained_model_back_in_the_source_layout(source, tmp_path):
    """One all-parameter SGD step at the reference's learning rate (1e-2,
    which moves the logits by up to 3.8). The loss is within 1e-5 of the
    reference's (4.8e-7 apart). The step concentrates its rounding in a few
    directions, so one run's distance of the updated logits from a float64
    step is a draw of a few degrees of freedom, and one draw against one
    reference draw is a coin flip at the factor: Dew's draws here run from
    0.32 to 2.1 times the reference's first. They are held instead over the
    fixture's 52 residual orders (orders.npz, tests/residual_orders.py), the
    reference's own from tools/kimi_linear_reference.py, by
    `assert_as_exact_over_orders`. Dew's RMS over them is 0.95 of the
    reference's on CPU, 0.73 on an AVX-512 CPU (Intel SDE's Ice Lake server,
    the CI runners' ISA) and 0.56 on an RTX 4080. The gradient through the
    KDA decays is where `recurrence_algebra.decays`' compensated sums matter.

    The export writes every source name back at its stored shape, A_log as
    [1, 1, heads, 1], and reloading restores the trained weights exactly."""
    trained, restored, written, shipped = exported(source, TINY, tmp_path)
    assert {name: value.shape for name, value in written.items()} == {
        name: value.shape for name, value in shipped.items()
    }
    assert written["model.layers.0.self_attn.A_log"].shape == (1, 1, 2, 1)
    for name, after in restored.items():
        np.testing.assert_array_equal(np.asarray(after), np.asarray(trained[name]), err_msg=name)


def test_greedy_generation_and_decode_steps_match_the_reference(source):
    """Four greedy tokens from both rows through the cached KDA recurrence and
    MLA cache, held as tests/test_generate_reference.py holds greedy decoding.
    E is the reference's largest decode logit error from the float64 logits
    teacher-forced over the same path (5.26e-6). Every step's float64 top-2
    margin exceeds 2 FACTOR E (smallest 0.034 against 2.1e-5), so the ids
    must be exact, and each chosen token's log probability is within 2
    FACTOR E of float64's (observed 1.8e-6)."""
    assert_greedy_as_exact(source)
