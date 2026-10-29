"""Kimi K3's text decoder against the released remote code.

kimi-k3-source holds the released config and the name, dtype and shape of
every tensor of moonshotai/Kimi-K3 at f831ab6 (read off the 96 shard
headers). kimi-k3-tiny is written by tools/kimi_k3_reference.py from the
pinned modeling_kimi_linear.py with fla-core 0.5.2's kernels in fp32 on
CUDA; its routed experts are compressed-tensors MXFP4 as in the release.
Its numerics.npz is the same released model evaluated in float64 on the
CPU, fla's kernels answered by its own torch references
(tools/kimi_k3_reference.py truth), which Dew and the reference are both
measured from (tests/reference_error.py). No model weights are downloaded
at test time.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from kimi_support import assert_forward_as_exact, assert_greedy_as_exact, exported, kimi_source
from reference_error import assert_as_exact_as_the_reference, ieee_fixture

from dew.interop import Pretrained
from dew.interop.codecs import decode_e2m1, quantize_packed_mxfp4
from dew.nn.activations import Situ

ROOT = Path(__file__).parent / "fixtures" / "hf"
TINY = ROOT / "kimi-k3-tiny"


@pytest.fixture(scope="module")
def source():
    return kimi_source(TINY)


def test_situ_matches_the_released_activation():
    """SituAndMul at the released betas (4, 25) over gate and up in
    [-80, 80]: the fp32 product is reproduced to 1 ulp-scale rounding."""
    reference = ieee_fixture(TINY / "reference.npz")
    situ = Situ(4.0, 25.0)
    actual = situ(jnp.asarray(reference["situ_gate"]), jnp.asarray(reference["situ_up"]))
    np.testing.assert_allclose(np.asarray(actual), reference["situ"], rtol=2e-6, atol=1e-6)


def test_forward_matches_the_reference_over_left_padding(source):
    """fp32 logits over 70 tokens (past one KDA chunk), one row left-padded
    by 9, against fla 0.5.2 on CUDA with its intra-chunk solve in IEEE fp32
    (tools/kimi_k3_reference.py): Dew's RMS distance from the float64
    logits at most twice the reference's, and argmax exact. Observed RMS
    from float64: Dew 1.67e-6, the reference 1.97e-6 (ratio 0.85 on CPU,
    0.87 on an RTX 4080, 0.89 on an RTX 3090)."""
    assert_forward_as_exact(source)


def test_update_exports_the_trained_model_back_in_the_source_layout(source, tmp_path):
    """One all-parameter SGD step at the reference's learning rate (1e-2,
    which moves the logits by up to 4.2). The loss is within 1e-5 of the
    reference's (9.5e-7 apart). One run's distance of the updated logits
    from a float64 step is a draw of a few degrees of freedom (Dew's draws
    here run from 0.54 to 1.98 times the reference's first), so they are
    held over the fixture's 52 residual orders, each moving MXFP4's groups
    of 32 whole (orders.npz, tests/residual_orders.py), by
    `assert_as_exact_over_orders`. Dew's RMS over them is 0.75 of the
    reference's on CPU, 0.71 on an AVX-512 CPU (Intel SDE's Ice Lake server)
    and 0.69 on an RTX 4080.

    The export writes every source name back: towers byte-exact, A_log
    zero-padded to its stored length, trained experts as MXFP4 pairs,
    everything else as trained; reloading reproduces that."""
    trained, restored, written, shipped = exported(source, TINY, tmp_path)
    assert set(written) == set(shipped)
    for name in shipped:
        if name.startswith(("vision_tower.", "mm_projector.")):
            np.testing.assert_array_equal(written[name], shipped[name])
        if name.endswith("A_log"):
            assert written[name].shape == shipped[name].shape and not written[name][2:].any()
    assert (
        written["language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed"].dtype == np.uint8
    )
    for name, after in restored.items():
        before = np.asarray(trained[name])
        if ".experts." in name:
            # Stacked [E, in, out]; compressed-tensors groups each [out, in] weight along its input.
            before = decode_e2m1(*quantize_packed_mxfp4(before.swapaxes(-1, -2))).swapaxes(-1, -2)
        np.testing.assert_array_equal(np.asarray(after), before, err_msg=name)


def test_greedy_generation_and_decode_steps_match_the_reference(source):
    """Four greedy tokens from both rows through the cached KDA recurrence and
    MLA cache, held as tests/test_generate_reference.py holds greedy decoding.
    E is the reference's largest decode logit error from the float64 logits
    teacher-forced over the same path (2.28e-5). Every step's float64 top-2
    margin exceeds 2 FACTOR E (smallest 6.2e-4 against 9.1e-5), so the ids
    must be exact, and each chosen token's log probability is within 2
    FACTOR E of float64's (observed 3.1e-6)."""
    assert_greedy_as_exact(source)


def test_cached_prefill_and_token_steps_are_as_exact_as_the_reference(source):
    """Row 0 prefilled for 40 tokens, then decoded one token at a time through
    the KDA state and the MLA cache: its logits are held to the reference
    forward's distance from float64, as the parallel forward is. Observed
    RMS from float64: Dew 1.39e-6, the reference 2.09e-6 (ratio 0.66 on CPU)."""
    loaded, inputs, reference = source
    model, variables = loaded.model, loaded.variables
    ids = inputs.tokens[:1]
    state = model.apply(variables, 1, method="init_cache", mutable=["cache"])[1]
    out, state = model.apply({**variables, **state}, ids[:, :40], decode=True, mutable=["cache"])
    pieces = [np.asarray(out)]
    for index in range(40, ids.shape[1]):
        out, state = model.apply(
            {**variables, **state}, ids[:, index : index + 1], decode=True, mutable=["cache"]
        )
        pieces.append(np.asarray(out))
    assert_as_exact_as_the_reference(np.concatenate(pieces, axis=1)[0], reference["logits"][0],
                                     reference["logits_f64"][0], "cached decode")


@pytest.mark.parametrize("param_dtype", ["bfloat16", "auto"])
def test_a_narrow_storage_load_holds_the_float32_loads_weights_rounded_once(source, param_dtype):
    """K3 nests its decoder under text_config and reads it whole, as a
    registered family rather than a media wrapper, so its narrow load's
    alias check reads it the same way: the MXFP4 experts decode in float32
    and every leaf is the float32 load's, rounded once to the storage dtype
    (bfloat16, the config's own for 'auto')."""
    loaded, _, _ = source
    narrow = Pretrained.load(TINY, dtype="float32", param_dtype=param_dtype, attention_impl="reference")
    assert jax.tree.structure(narrow.variables) == jax.tree.structure(loaded.variables)
    stored = set()
    for (path, ours), wide in zip(jax.tree_util.tree_leaves_with_path(narrow.variables),
                                  jax.tree.leaves(loaded.variables), strict=True):
        ours = np.asarray(ours)
        stored.add(ours.dtype)
        np.testing.assert_array_equal(ours, np.asarray(wide).astype(ours.dtype),
                                      err_msg=jax.tree_util.keystr(path))
    assert np.dtype(jnp.bfloat16) in stored
