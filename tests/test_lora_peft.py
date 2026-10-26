"""LoRA's rsLoRA scaling and its branch dropout in training against PEFT
0.20.0's own forward, backward and AdamW steps
(`tools/lora_peft_reference.py`), and a target that contracts and produces
several axes against a float64 oracle of its forward and its factors'
gradient."""

import contextlib
import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from reference_error import (
    assert_as_exact,
    assert_as_exact_as_the_reference,
    assert_as_exact_over_orders,
    with_gradients,
)
from safetensors.numpy import load_file
from sampling_gate import assert_draws_as_declared

from dew import lora
from dew.batch import merge
from dew.interop.pretrained import Pretrained
from dew.lora import Adapter, LoRA, Target
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.optim import OPTIMIZER_MAP

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "tests" / "fixtures" / "hf" / "llama-tiny"
FIXTURES = ROOT / "tests" / "fixtures" / "lora_peft"


STEP = Step(step=jnp.int32(0), key=jax.random.key(1), ema=None)
"""The training step whose loss the fixtures' gradients are, and whose draw `dew_masks` wrote
masks.npz from."""


def adapted(case: str):
    """The case's PEFT adapter on the tiny Llama, read by `LoRA.load`, and its reference arrays."""
    with np.load(FIXTURES / case / "reference.npz") as data:
        reference = {key: data[key] for key in data}
    decoder = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    return LoRA.load(decoder.model, decoder.variables, FIXTURES / case / "adapter",
                     layouts=decoder.layouts), reference


def assert_trains_as_peft(adapter, reference, case: str, tmp_path, masks=None):
    """One training step's mean next-token cross entropy and its gradient in all twelve factors, and
    what two steps of Dew's AdamW (`OPTIMIZER_MAP`, at PEFT's run's settings `ADAMW`) move each
    factor by, the factors exported in PEFT's layout (`Adapter.save`): each held to PEFT's float64
    run by the float64 rule against its float32 one, each branch's dropout given `masks`, PEFT's
    run's (`given`). The loss, one number, is held over the case's residual orders
    (`assert_loss_over_orders`)."""
    from tools.lora_peft_reference import ADAMW

    tokens = jnp.asarray(reference["input_ids"])
    objective = LMObjective(adapter.model, tokens.shape[1] - 1, variables=adapter.variables, ema_decay=None)
    params = objective.init(jax.random.key(0))

    def loss(moving):
        with given(adapter, masks):
            stats, _ = objective.loss({**params, "params": moving}, {"text": tokens}, STEP)
        return objective.reduce_loss(stats)[0]

    gradient = jax.grad(loss)(params["params"])
    assert_loss_over_orders(adapter, objective, params, reference, case, masks)
    names = sorted(key.removeprefix("grad/") for key in reference
                   if key.startswith("grad/") and not key.endswith("_f64"))
    exported = peft_factors(adapter, gradient, tmp_path / "gradient")
    assert sorted(exported) == names and len(names) == 12
    for name in names:
        assert_as_exact_as_the_reference(exported[name], reference[f"grad/{name}"],
                                         reference[f"grad/{name}_f64"], f"gradient {name}")

    adamw = OPTIMIZER_MAP["adamw"](ADAMW["lr"], *ADAMW["betas"], ADAMW["eps"],
                                   weight_decay=ADAMW["weight_decay"])
    moving, state = params["params"], adamw.init(params["params"])
    for _ in range(2):
        updates, state = adamw.update(jax.grad(loss)(moving), state, moving)
        moving = optax.apply_updates(moving, updates)
    start = peft_factors(adapter, params["params"], tmp_path / "start")
    stepped = peft_factors(adapter, moving, tmp_path / "stepped")
    # Each factor's move, from the float32 factors both runs start at: exact in float64.
    for name in names:
        assert_as_exact_as_the_reference(stepped[name] - start[name],
                                         reference[f"stepped/{name}"] - start[name],
                                         reference[f"stepped/{name}_f64"] - start[name], f"update {name}")


def assert_loss_over_orders(adapter, objective, params, reference, case: str, masks) -> None:
    """The training loss over the case's residual orders (`orders.npz`, tools/lora_peft_reference.py's
    `in_order`), its distance from PEFT's float64 loss, which no order moves, as exact as PEFT's
    float32 run's over the same orders (`assert_as_exact_over_orders`): one loss against one is a
    single rounding draw against another, and a draw of PEFT's sits a few ulps from float64 or far
    nearer by chance. Order 0, the identity, computes the loss the plain run does."""
    from tools.lora_peft_reference import adapter_orders, in_order

    with np.load(FIXTURES / case / "orders.npz") as recorded:
        drawn, theirs = recorded["orders"], recorded["loss"]
    whole, residual_inputs = adapter_orders(adapter)
    tokens = jnp.asarray(reference["input_ids"])

    def loss(variables, masks):
        with given(adapter, masks):
            stats, _ = objective.loss({**params, **variables}, {"text": tokens}, STEP)
        return objective.reduce_loss(stats)[0]

    mine = []
    for order in drawn:
        variables, moved = in_order(adapter, whole, residual_inputs, np.asarray(order, np.int64), masks)
        mine.append(abs(np.float64(loss(variables, moved)) - np.float64(reference["loss_f64"])))
    assert mine[0] == abs(np.float64(loss(adapter.variables, masks)) - np.float64(reference["loss_f64"]))
    assert_as_exact_over_orders(np.asarray(mine), theirs, "the loss")


@contextlib.contextmanager
def given(adapter, masks: dict[str, np.ndarray] | None):
    """Each target's branch dropout given its mask in `masks`, by PEFT module path
    (`branch_dropout`), keeping its input where the mask does at 1 / (1 - p) of the rate the
    branch drops at, as torch's dropout keeps it; no change where `masks` is None."""
    from tools.lora_peft_reference import branch_dropout

    def dropped(next_fun, args, kwargs, context):
        owner = branch_dropout(adapter, context)
        if owner is None:
            return next_fun(*args, **kwargs)
        return jnp.where(masks[owner], args[0] / (1 - context.module.rate), 0)

    if masks is None:
        yield
        return
    with nn.intercept_methods(dropped):
        yield


def peft_factors(adapter, factors, directory) -> dict[str, np.ndarray]:
    """The tree `factors`, shaped as the adapter's own, exported in PEFT's layout (`Adapter.save`)
    and read back, by PEFT's module path, as float64."""
    adapter.save(merge(adapter.variables, {"params": factors}), directory)
    return {key.removeprefix(lora.PEFT_PREFIX): np.asarray(value, np.float64)
            for key, value in load_file(directory / lora.PEFT_WEIGHTS).items()}


def test_an_rslora_adapter_loads_trains_and_exports_as_peft_trains_it(tmp_path):
    """rsLoRA: each target scales by its own alpha over the square root of its
    own rank (layer 1's v_proj at rank 2, down_proj at alpha 3). The adapted
    logits, and one training step, are PEFT's (`assert_trains_as_peft`)."""
    adapter, reference = adapted("rslora")
    assert (adapter.rslora, adapter.dropout) == (True, 0.0)
    logits = adapter.model.apply(adapter.variables, jnp.asarray(reference["input_ids"]))
    assert_as_exact_as_the_reference(logits, reference["adapted_logits"], reference["adapted_logits_f64"],
                                     "the adapted logits")
    # The adapter moves the logits far past rounding: the base model's are nowhere near the adapted ones.
    with pytest.raises(AssertionError):
        assert_as_exact_as_the_reference(reference["base_logits"], reference["adapted_logits"],
                                         reference["adapted_logits_f64"], "the base logits as adapted ones")
    assert_trains_as_peft(adapter, reference, "rslora", tmp_path)


def test_a_dropout_adapter_trains_as_peft_trains_it_on_the_masks_peft_ran_on(tmp_path):
    """Dropout at 0.25: a training step drops each target's branch input,
    `B A dropout(x)`, and not the base path. Each branch's dropout given the
    mask PEFT's run was given (masks.npz), keeping its input where the mask
    does at 1 / (1 - p) of the rate the branch drops at, as torch's dropout
    keeps it (`given`), the steps' loss, gradients and updates are PEFT's
    (`assert_trains_as_peft`), and every target's branch drops."""
    from tools.lora_peft_reference import MASKS, branch_draws

    adapter, reference = adapted("dropout")
    assert (adapter.rslora, adapter.dropout) == (False, 0.25)
    with np.load(FIXTURES / "dropout" / MASKS) as saved:
        masks = {name: saved[name] for name in saved.files}
    assert sorted(masks) == sorted(branch_draws(adapter, reference["input_ids"], STEP)) and len(masks) == 6
    assert_trains_as_peft(adapter, reference, "dropout", tmp_path, masks)


DROPOUT_SEEDS = 4
"""The dropout site's declared training steps (the sampling gate's): a kept share off 1 - p by six of one
step's standard deviations, the bound the site held before the gate, fails."""


def test_a_training_step_drops_each_branch_input_at_the_adapters_rate():
    """A training step draws a dropout of its own on each of the six targets'
    branch inputs at the adapter's rate: each nonzero input entry kept at
    1 - p as a fair draw would (the sampling gate, over DROPOUT_SEEDS steps),
    each target's mask its own and each step's its own, and the same step the
    same masks."""
    from tools.lora_peft_reference import branch_draws

    adapter, reference = adapted("dropout")
    tokens, rate = reference["input_ids"], adapter.dropout
    first, again = branch_draws(adapter, tokens, STEP), branch_draws(adapter, tokens, STEP)
    later = branch_draws(adapter, tokens, Step(step=jnp.int32(1), key=jax.random.key(2), ema=None))
    assert len(first) == 6
    for name, (x, output) in first.items():
        mask = output != 0
        # A zero input would hide whether the dropout kept it.
        assert np.all(x != 0), name
        np.testing.assert_array_equal(again[name][1], output, err_msg=name)
        assert not np.array_equal(later[name][1] != 0, mask), name

    def counts(seed):
        # A zero input is dropped or kept alike, so only the nonzero entries are draws.
        drawn = branch_draws(adapter, tokens, Step(step=jnp.int32(seed), key=jax.random.key(seed), ema=None))
        kept = sum(int(np.count_nonzero(output)) for _, output in drawn.values())
        return np.asarray([kept, sum(int(np.count_nonzero(x)) for x, _ in drawn.values()) - kept])

    assert_draws_as_declared(counts, (1 - rate, rate), "branch dropout", seeds=DROPOUT_SEEDS)
    masks = [output != 0 for _, output in first.values()]
    for one, other in itertools.combinations(masks, 2):
        assert one.shape != other.shape or not np.array_equal(one, other)


class Host(nn.Module):
    """One DenseGeneral that contracts the input's last two axes, listed out
    of order, into a (5, 6) output."""

    @nn.compact
    def __call__(self, x):
        return nn.DenseGeneral((5, 6), axis=(-1, -2), name="proj")(x)


MULTI_AXIS_SEEDS = 16
"""Check 1's declared seeds for the multi-axis branch, sized by `seeds_for` (2026-10-21, on armada)."""


def test_a_multi_axis_target_steps_as_its_flattened_matrix_does():
    """The branch of a target contracting (8, 6) input axes into (5, 6)
    output ones is the matrix LoRA of the flattened kernel: with x as
    [batch, 48] in the kernel's sorted axis order, A as [48, r] and B as
    [r, 30], the output is x K + bias + s x A B, and a fixed cotangent G
    gives A's gradient s x^T G B^T and B's s (x A)^T G. The float64 oracle
    computes that flattened form; numpy in float32 is the reference Dew's
    forward and gradients are on par with (check 1), over drawn inputs,
    weights, factors and cotangents."""
    model = Host()
    target = Target(rank=4, alpha=3.0)

    def draw(seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(16, 6, 8)).astype(np.float32)
        params = jax.tree.map(np.asarray, model.init(jax.random.key(seed), jnp.asarray(x)))["params"]["proj"]
        return (x, params["kernel"], params["bias"], rng.normal(size=(6, 8, 4)).astype(np.float32),
                rng.normal(size=(4, 5, 6)).astype(np.float32), rng.normal(size=(16, 5, 6)).astype(np.float32))

    _, kernel, bias, a, b, _ = draw(3)
    tree = {"params": {"proj": {"kernel": kernel, "bias": bias, "lora_A": a, "lora_B": b}}}
    adapter = Adapter.bound(model, tree, {("params", "proj"): target}, rslora=False, dropout=0.0, layouts={})
    assert tree["params"]["proj"]["kernel"].shape == (6, 8, 5, 6)
    scale = adapter.scale(target)

    @jax.jit
    def dew(x, kernel, bias, a, b, cotangent):
        def output(factors):
            node = {"kernel": kernel, "bias": bias, **factors}
            return adapter.model.apply({"params": {"proj": node}}, x)

        forward, gradient = with_gradients(output)({"lora_A": a, "lora_B": b}, cotangent)
        return {"the output": forward, "lora_A's gradient": gradient["lora_A"],
                "lora_B's gradient": gradient["lora_B"]}

    def oracle(x, kernel, bias, a, b, cotangent):
        dtype = x.dtype.type
        rows, seed = x.reshape(16, 48), cotangent.reshape(16, 30)
        down, up = a.reshape(48, 4), b.reshape(4, 30)
        s = dtype(scale)
        out = rows @ kernel.reshape(48, 30) + bias.reshape(30) + s * ((rows @ down) @ up)
        return {"the output": out.reshape(16, 5, 6),
                "lora_A's gradient": (s * (rows.T @ (seed @ up.T))).reshape(6, 8, 4),
                "lora_B's gradient": (s * ((rows @ down).T @ seed)).reshape(4, 5, 6)}

    assert_as_exact(dew, oracle, draw, "multi-axis LoRA", seeds=MULTI_AXIS_SEEDS)
