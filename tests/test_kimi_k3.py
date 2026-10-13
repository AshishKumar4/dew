"""Kimi K3's text decoder against the released remote code.

kimi-k3-source holds the released config and the name, dtype and shape of
every tensor of moonshotai/Kimi-K3 at f831ab6 (read off the 96 shard
headers). kimi-k3-tiny is written by tools/kimi_k3_reference.py from the
pinned modeling_kimi_linear.py with fla-core 0.5.2's kernels in fp32 on
CUDA; its routed experts are compressed-tensors MXFP4 as in the release.
Its numerics.npz is the same model evaluated in float64
(tools/kimi_float64_reference.py), which Dew and the reference are both
measured from (tests/reference_error.py). No model weights are downloaded
at test time.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax.traverse_util import flatten_dict
from reference_error import (
    FACTOR,
    assert_as_exact_as_the_reference,
    assert_as_exact_over_orders,
    distance,
    ieee_fixture,
)
from residual_orders import permuted
from safetensors.numpy import load_file
from scipy.special import log_softmax

from dew.interop import Pretrained
from dew.interop.codecs import decode_e2m1, quantize_packed_mxfp4
from dew.nn.inputs import ModelInputs
from dew.nn.moe import Situ
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling, generate

ROOT = Path(__file__).parent / "fixtures" / "hf"
TINY = ROOT / "kimi-k3-tiny"


@pytest.fixture(scope="module")
def source():
    loaded = Pretrained.load(TINY, dtype="float32", attention_impl="reference")
    reference = ieee_fixture(TINY / "reference.npz") | ieee_fixture(TINY / "numerics.npz")
    inputs = ModelInputs(jnp.asarray(reference["input_ids"], jnp.int32),
                         {"attention_mask": jnp.asarray(reference["attention_mask"], bool)})
    return loaded, inputs, reference


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
    loaded, inputs, reference = source
    valid = reference["attention_mask"].astype(bool)
    logits = jax.jit(lambda variables: loaded.model.apply(variables, inputs.tokens, **inputs.kwargs()))(
        loaded.variables
    )
    assert_as_exact_as_the_reference(np.asarray(logits)[valid], reference["logits"][valid],
                                     reference["logits_f64"][valid], "logits")
    np.testing.assert_array_equal(np.asarray(logits)[valid].argmax(-1), reference["logits"][valid].argmax(-1))


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
    loaded, inputs, reference = source
    objective = LMObjective(loaded, inputs.tokens.shape[1] - 1,
                            ema_decay=None, pad_id=0)
    step = Step(step=jnp.int32(0), key=jax.random.key(0), ema=None)

    def loss(variables):
        statistics, _ = objective.loss({**loaded.variables, "params": variables}, {"text": inputs}, step)
        return objective.reduce_loss(statistics)[0]

    def stepped(params):
        gradient = jax.grad(loss)(params)
        return jax.tree.map(lambda weight, grad: weight - reference["learning_rate"] * grad, params, gradient)

    value = jax.jit(loss)(loaded.variables["params"])
    np.testing.assert_allclose(value, reference["loss"], atol=1e-5, rtol=0)
    valid = reference["attention_mask"].astype(bool)
    updated = jax.jit(lambda params: loaded.model.apply(
        {**loaded.variables, "params": stepped(params)}, inputs.tokens, **inputs.kwargs()))
    drawn = ieee_fixture(TINY / "orders.npz")
    orders, theirs = drawn["orders"], drawn["updated_logits"]
    truth = reference["updated_logits_f64"][valid]
    mine = [distance(np.asarray(updated(permuted(loaded.variables, order)["params"]))[valid], truth)
            for order in orders]
    assert_as_exact_over_orders(mine, theirs, "updated logits")

    variables = {**loaded.variables, "params": jax.jit(stepped)(loaded.variables["params"])}

    loaded.save(tmp_path, variables=variables)
    written, shipped = (
        load_file(str(tmp_path / "model.safetensors")),
        load_file(str(TINY / "model.safetensors")),
    )
    assert set(written) == set(shipped)
    for name in shipped:
        if name.startswith(("vision_tower.", "mm_projector.")):
            np.testing.assert_array_equal(written[name], shipped[name])
        if name.endswith("A_log"):
            assert written[name].shape == shipped[name].shape and not written[name][2:].any()
    assert (
        written["language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed"].dtype == np.uint8
    )
    assert json.loads((tmp_path / "config.json").read_text()) == json.loads(
        (TINY / "config.json").read_text()
    )
    restored = Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")
    trained = flatten_dict(variables, sep=".")
    for name, after in flatten_dict(dict(restored.variables), sep=".").items():
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
    loaded, inputs, reference = source
    error = float(np.max(np.abs(reference["step_logits"] - reference["step_logits_f64"])))
    ranked = np.sort(reference["step_logits_f64"], -1)
    assert np.min(ranked[..., -1] - ranked[..., -2]) > 2 * FACTOR * error
    generated = generate(loaded.model, loaded.variables, inputs, 4, key=jax.random.key(1),
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(generated.tokens)[:, -4:], reference["generated"][:, -4:])
    exact = np.take_along_axis(log_softmax(reference["step_logits_f64"], -1),
                               reference["generated"][:, -4:, None], -1)[..., 0]
    assert (
        np.max(np.abs(np.asarray(generated.raw_log_probs, np.float64)[:, :4] - exact)) <= 2 * FACTOR * error
    )


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
