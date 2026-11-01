"""What the Kimi Linear and Kimi K3 suites hold their fixtures to alike: each fixture directory holds the
reference's inputs and outputs (reference.npz), the same model in float64 (numerics.npz) and the
reference's updated logits over residual orders (orders.npz), and each suite's docstrings give its
family's own measurements."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from model_support import as_stepped, parameter_loss
from reference_error import (
    FACTOR,
    as_given,
    assert_as_exact,
    assert_as_exact_over_orders,
    distance,
    ieee_fixture,
    indexed_recording,
    recording,
)
from residual_orders import permuted
from scipy.special import log_softmax

from dew.batch import Variables
from dew.interop import Pretrained
from dew.nn.inputs import ModelInputs
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling, generate


def kimi_source(directory: Path) -> tuple[Pretrained, ModelInputs, dict[str, np.ndarray]]:
    """The fixture loaded in float32, its two rows (one left-padded) and its reference arrays."""
    loaded = Pretrained.load(directory, dtype="float32", attention_impl="reference")
    reference = ieee_fixture(directory / "reference.npz") | ieee_fixture(directory / "numerics.npz")
    inputs = ModelInputs(jnp.asarray(reference["input_ids"], jnp.int32),
                         {"attention_mask": jnp.asarray(reference["attention_mask"], bool)})
    return loaded, inputs, reference


RECORDING = "recording.npz"
"""Each fixture's check-1 evidence of the reference's logits over drawn batches (`drawn_batch`), written by
its tool's `seeds` command (tools/kimi_k3_reference.py's `record_forward_seeds`)."""

STEPPED_PREFILL = 40
"""The prefill of a cached decode the recording's 'stepped' leaf scores (`stepped_logits`)."""


def drawn_batch(seed: int) -> dict[str, np.ndarray]:
    """A seed's batch of the fixtures' geometry: two rows of 70 ids past the specials, past one 64-token KDA
    chunk, the second left-padded by 9, a stream of its own for each seed."""
    ids = np.random.default_rng([0x3755, seed]).integers(3, 512, (2, 70)).astype(np.int32)
    mask = np.ones_like(ids)
    ids[1, :9], mask[1, :9] = 0, 0
    return {"input_ids": ids, "attention_mask": mask}


def _forward(model, variables, inputs) -> jax.Array:
    """Each valid position's logits of `model` over a batch (`drawn_batch`)."""
    logits = model.apply(variables, jnp.asarray(inputs["input_ids"]),
                         attention_mask=jnp.asarray(inputs["attention_mask"], bool))
    return logits[jnp.asarray(inputs["attention_mask"], bool)]


def stepped_logits(model, variables, row) -> jax.Array:
    """`row`'s logits `[tokens, vocab]` from a cached decode: its first `STEPPED_PREFILL` tokens prefilled,
    then a token a step through the cache."""
    state = model.apply(variables, 1, method="init_cache", mutable=["cache"])[1]
    out, state = model.apply({**variables, **state}, row[None, :STEPPED_PREFILL], decode=True,
                             mutable=["cache"])
    pieces = [out[0]]
    for index in range(STEPPED_PREFILL, row.shape[0]):
        out, state = model.apply({**variables, **state}, row[None, index:index + 1], decode=True,
                                 mutable=["cache"])
        pieces.append(out[0])
    return jnp.concatenate(pieces)


def assert_logits_as_recorded(source, directory: Path, leaf: str) -> None:
    """Check 1 of Dew's float32 logits against the reference's recorded over drawn batches (`RECORDING`):
    'logits', each valid position's of a batch's forward; 'stepped', the first row's from Dew's cached
    decode (`stepped_logits`), against the reference's forwards at the shapes the decode computes them
    (ledger 1543). Dew's own float64 forward is the truth, held to the recorded truth's sketch."""
    loaded = source[0]
    model, variables = loaded.model, loaded.variables
    stored = recording(directory / RECORDING)
    seeds = int(stored["forward.seeds"])
    forward = jax.jit(lambda inputs: _forward(model, variables, inputs))
    if leaf == "logits":
        dew = as_given(lambda index, inputs: {"logits": forward(inputs)})
    else:
        dew = as_given(lambda index, inputs: {"stepped": stepped_logits(
            model, variables, jnp.asarray(inputs["input_ids"][0]))})
    with jax.enable_x64(new_val=True):
        wide = Pretrained.load(directory, dtype="float64", param_dtype="float64", attention_impl="reference")
        wide_forward = jax.jit(lambda inputs: _forward(wide.model, wide.variables, inputs))
        wide_rows = jax.jit(lambda ids: wide.model.apply(wide.variables, ids))

    def truth(inputs):
        with jax.enable_x64(new_val=True):
            if leaf == "logits":
                return {"logits": wide_forward(inputs)}
            row = jnp.asarray(inputs["input_ids"][:1])
            return {"stepped": as_stepped(wide_rows, row, STEPPED_PREFILL)[0]}

    drawn, reference, recorded_truth = indexed_recording(stored, "forward", seeds, inputs_of=drawn_batch,
                                                         truth_fn=as_given(truth), leaves=(leaf,))
    assert_as_exact(dew, reference, drawn, f"{directory.name} {leaf}", seeds=seeds, truth_fn=recorded_truth)


def assert_update_as_exact(source, directory: Path) -> Variables:
    """One all-parameter SGD step at the reference's learning rate: the loss within 1e-5 of the
    reference's, and the updated logits as exact over the fixture's residual orders as the
    reference's own (`assert_as_exact_over_orders`). Returns the stepped variables."""
    loaded, inputs, reference = source
    objective = LMObjective(loaded, inputs.tokens.shape[1] - 1, ema_decay=None, pad_id=0)
    step = Step(step=jnp.int32(0), key=jax.random.key(0), ema=None)

    loss = parameter_loss(objective, loaded.variables, {"text": inputs}, step)

    def stepped(params):
        gradient = jax.grad(loss)(params)
        return jax.tree.map(lambda weight, grad: weight - reference["learning_rate"] * grad, params, gradient)

    value = jax.jit(loss)(loaded.variables["params"])
    np.testing.assert_allclose(value, reference["loss"], atol=1e-5, rtol=0)
    valid = reference["attention_mask"].astype(bool)
    updated = jax.jit(lambda params: loaded.model.apply(
        {**loaded.variables, "params": stepped(params)}, inputs.tokens, **inputs.kwargs()))
    drawn = ieee_fixture(directory / "orders.npz")
    truth = reference["updated_logits_f64"][valid]
    mine = [distance(np.asarray(updated(permuted(loaded.variables, order)["params"]))[valid], truth)
            for order in drawn["orders"]]
    assert_as_exact_over_orders(mine, drawn["updated_logits"], "updated logits")
    return {**loaded.variables, "params": jax.jit(stepped)(loaded.variables["params"])}


def exported(source, directory: Path, written: Path) -> tuple[dict, dict, dict, dict]:
    """The model stepped once (`assert_update_as_exact`) and saved to `written` with the config
    `directory` ships: the stepped weights and the reloaded ones, flat by dotted name, and the tensors
    it wrote and the ones `directory` ships, by name."""
    from flax.traverse_util import flatten_dict
    from safetensors.numpy import load_file

    loaded, _, _ = source
    variables = assert_update_as_exact(source, directory)
    loaded.save(written, variables=variables)
    assert (json.loads((written / "config.json").read_text())
            == json.loads((directory / "config.json").read_text()))
    restored = Pretrained.load(written, dtype="float32", attention_impl="reference")
    return (flatten_dict(dict(variables), sep="."), flatten_dict(dict(restored.variables), sep="."),
            load_file(str(written / "model.safetensors")), load_file(str(directory / "model.safetensors")))


def assert_greedy_as_exact(source) -> None:
    """Four greedy tokens from both rows, held as tests/test_generate_reference.py holds greedy
    decoding: with E the reference's largest decode logit error from float64, every step's float64
    top-2 margin exceeds 2 FACTOR E, so the ids are exact, and each chosen token's log probability is
    within 2 FACTOR E of float64's."""
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
