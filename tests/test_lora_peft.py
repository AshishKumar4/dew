"""LoRA's rsLoRA scaling and its branch dropout in training against PEFT's
own forward, backward and SGD steps over drawn token ids, factors and
masks (`tools/lora_peft_reference.py`'s runs), and a target that contracts
and produces several axes against a float64 oracle of its forward and its
factors' gradient."""

import contextlib
import functools
import itertools
import json
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
from flax import linen as nn
from peft import LoraConfig, inject_adapter_in_model
from reference_error import (
    GateFailure,
    Inconclusive,
    as_given,
    assert_as_exact,
    assert_as_exact_over_orders,
    with_gradients,
)
from safetensors.numpy import load_file
from sampling_gate import assert_draws_as_declared
from tool_support import run_from

from dew import lora
from dew.batch import merge
from dew.interop.pretrained import Pretrained
from dew.lora import Adapter, LoRA, Target
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.optim import OptimConfig

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "tests" / "fixtures" / "hf" / "llama-tiny"
FIXTURES = ROOT / "tests" / "fixtures" / "lora_peft"


STEP = Step(step=jnp.int32(0), key=jax.random.key(1), ema=None)
"""The training step whose loss the fixtures' gradients are, and whose draw `dew_masks` wrote
masks.npz from."""


@functools.cache
def peft_reference():
    """tools/lora_peft_reference.py, PEFT's runs and the draws of Dew's branch dropout, loaded as a script
    from its directory."""
    with run_from("lora_peft_reference", "tools") as module:
        return module


@functools.cache
def adapted(case: str):
    """The case's PEFT adapter on the tiny Llama, read by `LoRA.load`, and its reference arrays."""
    with np.load(FIXTURES / case / "reference.npz") as data:
        reference = {key: data[key] for key in data}
    decoder = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    return LoRA.load(decoder.model, decoder.variables, FIXTURES / case / "adapter",
                     layouts=decoder.layouts), reference


PEFT_SEEDS = {"rslora": 64, "dropout": 256}
"""Each case's declared seeds, sized by `seeds_for` paired (armada, 11-06)."""


@functools.cache
def trainer(case: str):
    """The case's adapter in the LM objective over the fixture's token shape, its initialized tree, and
    one training step's logits, gradient in every factor and the factors after two steps of SGD at PEFT's
    run's rate (tools/lora_peft_reference.py's `SGD`), each branch's dropout given its mask (`given`),
    compiled."""
    rate = peft_reference().SGD["lr"]
    adapter, reference = adapted(case)
    objective = LMObjective(adapter.model, reference["input_ids"].shape[1] - 1, variables=adapter.variables,
                            ema_decay=None)
    params = nn.unbox(objective.init(jax.random.key(0)))
    sgd = OptimConfig(optimizer="optax:sgd", learning_rate=rate, clip_grads=0.0).build(2)

    @jax.jit
    def run(tokens, factors, masks):
        def loss(moving):
            with given(adapter, masks):
                stats, _ = objective.loss({**params, "params": moving}, {"text": tokens}, STEP)
            return objective.reduce_loss(stats)[0]

        moving, state = factors, sgd.init(factors)
        for _ in range(2):
            updates, state = sgd.update(jax.grad(loss)(moving), state, moving)
            moving = optax.apply_updates(moving, updates)
        logits = adapter.model.apply(merge(adapter.variables, {"params": factors}), tokens)
        return logits, jax.grad(loss)(factors), moving

    return adapter, objective, params, run


def drawn(case: str):
    """The case's draw: a seed's token ids of the fixture's shape; its adapter's factors as PEFT draws an
    A, uniform within one over the root of its fan-in, and each B normal at 0.2, the fixture's own scale,
    so the adapter moves the logits, in Dew's layout and exported in PEFT's (`peft_factors`); and where
    the adapter drops its branch inputs, each target's keep mask by PEFT module path, kept at 1 - p."""
    adapter, _, params, _ = trainer(case)
    shape = adapted(case)[1]["input_ids"].shape
    vocabulary = json.loads((LLAMA / "config.json").read_text())["vocab_size"]
    with np.load(FIXTURES / "dropout" / "masks.npz") as saved:
        shapes = {name: saved[name].shape for name in saved.files}

    def draw(seed):
        rng = np.random.default_rng([2126, seed])
        tokens = rng.integers(0, vocabulary, shape, dtype=np.int32)

        def factor(path, leaf):
            if path[-1].key == "lora_A":
                bound = 1 / np.sqrt(np.prod(leaf.shape[:-1]))
                return rng.uniform(-bound, bound, leaf.shape).astype(np.float32)
            return (rng.normal(size=leaf.shape) * 0.2).astype(np.float32)

        factors = jax.tree_util.tree_map_with_path(factor, params["params"])
        with tempfile.TemporaryDirectory() as directory:
            peft = {name: value.astype(np.float32)
                    for name, value in peft_factors(adapter, factors, Path(directory)).items()}
        masks = (None if not adapter.dropout else
                 {name: rng.random(size) >= adapter.dropout for name, size in shapes.items()})
        return tokens, factors, peft, masks
    return draw


def trained(case: str):
    """Dew's arm: one training step's gradient in every factor and every factor after two SGD steps, as a
    run builds its optimizer (`OptimConfig`), the factors exported in PEFT's layout (`Adapter.save`):
    what a user trains, not each step's move; and where the adapter drops no branch input, its logits,
    which PEFT's training forward gives under its masks where it does (`adapted_logits`)."""
    adapter, _, _, run = trainer(case)

    @as_given
    def arm(tokens, factors, peft, masks):
        logits, gradient, moving = run(tokens, factors, masks)
        with tempfile.TemporaryDirectory() as directory:
            exported = peft_factors(adapter, gradient, Path(directory) / "gradient")
            stepped = peft_factors(adapter, moving, Path(directory) / "stepped")
        return {**adapted_logits(adapter, logits), "gradient": exported, "trained": stepped}
    return arm


def adapted_logits(adapter, logits) -> dict:
    """The logits leaf of an arm whose adapter drops no branch input: PEFT's run gives the logits of its
    training forward, which drops a branch's inputs where the adapter does, and Dew's are its forward's."""
    return {} if adapter.dropout else {"logits": logits}


def peft_trained(case: str):
    """PEFT's arm and its float64 truth: the case's adapter on the tiny Llama (`peft_model`) given the
    draw's factors through PEFT's own state loading, run as `trained` is, in the draw's precision
    (`tools/lora_peft_reference.py`'s `run`, its float64 twin `widened`)."""
    from peft import set_peft_model_state_dict

    model = peft_model(case)

    @as_given
    def arm(tokens, factors, peft, masks):
        dtype = torch.float64 if next(iter(peft.values())).dtype == np.float64 else torch.float32
        loaded = set_peft_model_state_dict(model, {lora.PEFT_PREFIX + name: torch.as_tensor(np.float32(value))
                                                   for name, value in peft.items()})
        assert not loaded.unexpected_keys, loaded.unexpected_keys
        ran = peft_reference().run(model, torch.as_tensor(tokens).long(), dtype, masks,
                                   adapted(case)[0].dropout)
        tail = "_f64" if dtype == torch.float64 else ""
        return {**adapted_logits(adapted(case)[0], ran[f"adapted_logits{tail}"]),
                "gradient": {name: ran[f"grad/{name}{tail}"] for name in peft},
                "trained": {name: ran[f"stepped/{name}{tail}"] for name in peft}}
    return arm


@functools.cache
def peft_model(case: str):
    """PEFT's model of the case's adapter on the tiny Llama, trainable."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float32)
    return PeftModel.from_pretrained(base, FIXTURES / case / "adapter", is_trainable=True)


def assert_trains_as_peft(case: str):
    """One training step's gradient in all twelve factors, every factor after two SGD steps,
    the factors exported in PEFT's layout, and an adapter without dropout's logits, as exact as PEFT's
    float32 run against its float64 run over drawn token ids, factors and masks (check 1); and the loss,
    one number, held on the fixture's own adapter over the case's residual orders
    (`assert_loss_over_orders`)."""
    adapter, reference = adapted(case)
    assert len(drawn(case)(0)[2]) == 12
    reference_arm = peft_trained(case)
    assert_as_exact(trained(case), reference_arm, drawn(case), f"one {case} training step",
                    seeds=PEFT_SEEDS[case], truth_fn=reference_arm)
    masks = None
    if adapter.dropout:
        with np.load(FIXTURES / case / "masks.npz") as saved:
            masks = {name: saved[name] for name in saved.files}
    _, objective, params, _ = trainer(case)
    assert_loss_over_orders(adapter, objective, params, reference, case, masks)


def assert_loss_over_orders(adapter, objective, params, reference, case: str, masks) -> None:
    """The training loss over the case's residual orders (`orders.npz`, tools/lora_peft_reference.py's
    `in_order`), its distance from PEFT's float64 loss, which no order moves, as exact as PEFT's
    float32 run's over the same orders (`assert_as_exact_over_orders`): one loss against one is a
    single rounding draw against another, and a draw of PEFT's sits a few ulps from float64 or far
    nearer by chance. Order 0, the identity, computes the loss the plain run does."""
    adapter_orders, in_order = peft_reference().adapter_orders, peft_reference().in_order
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
    branch_dropout = peft_reference().branch_dropout

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


def test_an_rslora_adapter_loads_trains_and_exports_as_peft_trains_it():
    """rsLoRA: each target scales by its own alpha over the square root of its
    own rank (layer 1's v_proj at rank 2, down_proj at alpha 3). The adapted
    logits, and one training step, are PEFT's (`assert_trains_as_peft`); the
    base model's logits are not, check 1 deciding against them."""
    adapter, _ = adapted("rslora")
    assert (adapter.rslora, adapter.dropout) == (True, 0.0)
    assert_trains_as_peft("rslora")
    decoder = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    base = jax.jit(lambda tokens: decoder.model.apply(decoder.variables, tokens))
    reference_arm = peft_trained("rslora")

    def logits(arm):
        return as_given(lambda *inputs: {"logits": arm(*inputs)["logits"]})

    with pytest.raises(GateFailure) as failed:
        assert_as_exact(as_given(lambda tokens, *_: {"logits": base(tokens)}), logits(reference_arm),
                        drawn("rslora"), "the base logits as adapted ones", seeds=PEFT_SEEDS["rslora"],
                        truth_fn=logits(reference_arm))
    assert not isinstance(failed.value, Inconclusive)


def test_a_dropout_adapter_trains_as_peft_trains_it_on_the_masks_peft_ran_on():
    """Dropout at 0.25: a training step drops each target's branch input,
    `B A dropout(x)`, and not the base path. Each branch's dropout given the
    mask PEFT's run is given, keeping its input where the mask does at
    1 / (1 - p) of the rate the branch drops at, as torch's dropout keeps it
    (`given`), the step's gradients and the factors two steps train are
    PEFT's (`assert_trains_as_peft`), and every target's branch drops."""
    adapter, reference = adapted("dropout")
    assert (adapter.rslora, adapter.dropout) == (False, 0.25)
    with np.load(FIXTURES / "dropout" / peft_reference().MASKS) as saved:
        masks = {name: saved[name] for name in saved.files}
    drawn_by = peft_reference().branch_draws(adapter, reference["input_ids"], STEP)
    assert sorted(masks) == sorted(drawn_by) and len(masks) == 6
    assert_trains_as_peft("dropout")


DROPOUT_SEEDS = 4
"""The dropout site's declared training steps (the sampling gate's): a kept share off 1 - p by six of one
step's standard deviations, the bound the site held before the gate, fails."""


def test_a_training_step_drops_each_branch_input_at_the_adapters_rate():
    """A training step draws a dropout of its own on each of the six targets'
    branch inputs at the adapter's rate: each nonzero input entry kept at
    1 - p as a fair draw would (the sampling gate, over DROPOUT_SEEDS steps),
    each target's mask its own and each step's its own, and the same step the
    same masks."""
    branch_draws = peft_reference().branch_draws
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


class Flattened(torch.nn.Module):
    """`Host`'s target as PEFT adapts it: one Linear of its flattened (48, 30) kernel, `proj`."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(48, 30)

    def forward(self, x):
        return self.proj(x)


MULTI_AXIS_SEEDS = 16
"""Check 1's declared seeds for the multi-axis branch, sized by `seeds_for` (2026-10-21, on armada)."""


def test_a_multi_axis_target_steps_as_its_flattened_matrix_does():
    """The branch of a target contracting (8, 6) input axes into (5, 6)
    output ones is the matrix LoRA of the flattened kernel: with x as
    [batch, 48] in the kernel's sorted axis order, A as [48, r] and B as
    [r, 30], the output is x K + bias + s x A B, and a fixed cotangent G
    gives A's gradient s x^T G B^T and B's s (x A)^T G. The float64 truth
    computes that flattened form; PEFT's LoRA of that Linear in torch, float32,
    is the reference Dew's forward and gradients are on par with (check 1),
    over drawn inputs, weights, factors and cotangents. The branch at twice
    its scale fails the site."""
    model = Host()
    target = Target(rank=4, alpha=3.0)

    def draw(seed):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(16, 6, 8)).astype(np.float32)
        drawn = nn.unbox(model.init(jax.random.key(seed), jnp.asarray(x)))
        params = jax.tree.map(np.asarray, drawn)["params"]["proj"]
        return (x, params["kernel"], params["bias"], rng.normal(size=(6, 8, 4)).astype(np.float32),
                rng.normal(size=(4, 5, 6)).astype(np.float32), rng.normal(size=(16, 5, 6)).astype(np.float32))

    _, kernel, bias, a, b, _ = draw(3)
    tree = {"params": {"proj": {"kernel": kernel, "bias": bias, "lora_A": a, "lora_B": b}}}
    adapter = Adapter.bound(model, tree, {("params", "proj"): target}, rslora=False, dropout=0.0, layouts={})
    assert tree["params"]["proj"]["kernel"].shape == (6, 8, 5, 6)
    scale = adapter.scale(target)

    def adapted(adapter: Adapter):
        @jax.jit
        def dew(x, kernel, bias, a, b, cotangent):
            def output(factors):
                node = {"kernel": kernel, "bias": bias, **factors}
                return adapter.model.apply({"params": {"proj": node}}, x)

            forward, gradient = with_gradients(output)({"lora_A": a, "lora_B": b}, cotangent)
            return {"the output": forward, "lora_A's gradient": gradient["lora_A"],
                    "lora_B's gradient": gradient["lora_B"]}
        return dew

    @as_given
    def peft(x, kernel, bias, a, b, cotangent):
        dtype = torch.float64 if x.dtype == np.float64 else torch.float32
        config = LoraConfig(r=target.rank, lora_alpha=target.alpha, target_modules=["proj"])
        layer = inject_adapter_in_model(config, Flattened()).to(dtype).proj
        with torch.no_grad():
            layer.base_layer.weight.copy_(torch.tensor(kernel.reshape(48, 30).T))
            layer.base_layer.bias.copy_(torch.tensor(bias.reshape(30)))
            layer.lora_A["default"].weight.copy_(torch.tensor(a.reshape(48, 4).T))
            layer.lora_B["default"].weight.copy_(torch.tensor(b.reshape(4, 30).T))
        out = layer(torch.tensor(x.reshape(16, 48)))
        out.backward(torch.tensor(cotangent.reshape(16, 30)))
        return {"the output": out.detach().numpy().reshape(16, 5, 6),
                "lora_A's gradient": layer.lora_A["default"].weight.grad.numpy().T.reshape(6, 8, 4),
                "lora_B's gradient": layer.lora_B["default"].weight.grad.numpy().T.reshape(4, 5, 6)}

    @as_given
    def flattened(x, kernel, bias, a, b, cotangent):
        dtype = x.dtype.type
        rows, seed = x.reshape(16, 48), cotangent.reshape(16, 30)
        down, up = a.reshape(48, 4), b.reshape(4, 30)
        s = dtype(scale)
        out = rows @ kernel.reshape(48, 30) + bias.reshape(30) + s * ((rows @ down) @ up)
        return {"the output": out.reshape(16, 5, 6),
                "lora_A's gradient": (s * (rows.T @ (seed @ up.T))).reshape(6, 8, 4),
                "lora_B's gradient": (s * ((rows @ down).T @ seed)).reshape(4, 5, 6)}

    assert_as_exact(adapted(adapter), peft, draw, "multi-axis LoRA", seeds=MULTI_AXIS_SEEDS,
                    truth_fn=flattened)
    doubled = Target(rank=4, alpha=6.0)
    faulted = Adapter.bound(model, tree, {("params", "proj"): doubled}, rslora=False, dropout=0.0, layouts={})
    with pytest.raises(GateFailure):
        assert_as_exact(adapted(faulted), peft, draw, "multi-axis LoRA at twice its scale",
                        seeds=MULTI_AXIS_SEEDS, truth_fn=flattened)
