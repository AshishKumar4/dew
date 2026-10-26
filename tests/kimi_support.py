"""What the Kimi Linear and Kimi K3 suites hold their fixtures to alike: each fixture directory holds the
reference's inputs and outputs (reference.npz), the same model in float64 (numerics.npz) and the
reference's updated logits over residual orders (orders.npz), and each suite's docstrings give its
family's own measurements."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from model_support import parameter_loss
from reference_error import (
    FACTOR,
    assert_as_exact_as_the_reference,
    assert_as_exact_over_orders,
    distance,
    ieee_fixture,
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


def assert_forward_as_exact(source) -> None:
    """The real tokens' logits as exact as the reference's from float64, with its argmax."""
    loaded, inputs, reference = source
    valid = reference["attention_mask"].astype(bool)
    logits = jax.jit(lambda variables: loaded.model.apply(variables, inputs.tokens, **inputs.kwargs()))(
        loaded.variables
    )
    assert_as_exact_as_the_reference(np.asarray(logits)[valid], reference["logits"][valid],
                                     reference["logits_f64"][valid], "logits")
    np.testing.assert_array_equal(np.asarray(logits)[valid].argmax(-1), reference["logits"][valid].argmax(-1))


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
