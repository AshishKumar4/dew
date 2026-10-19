"""LoRA's rsLoRA scaling and its branch dropout in training against PEFT
0.20.0's own forward, backward and AdamW steps
(`tools/lora_peft_reference.py`), and a target that contracts and produces
several axes against a float64 oracle of its forward and its factors'
gradient."""

import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from reference_error import assert_as_exact_as_the_reference, assert_leaves_as_exact
from safetensors.numpy import load_file

from dew import lora
from dew.interop.pretrained import Pretrained
from dew.lora import Adapter, LoRA, Target
from dew.objectives.base import Step, merge
from dew.objectives.lm import LMObjective
from dew.training.optim import OPTIMIZER_MAP

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


def assert_trains_as_peft(adapter, reference, tmp_path):
    """One training step's mean next-token cross entropy and its gradient in all twelve factors, and
    what two steps of Dew's AdamW (`OPTIMIZER_MAP`, at PEFT's run's settings `ADAMW`) move each
    factor by, the factors exported in PEFT's layout (`Adapter.save`): each held to PEFT's float64
    run by the float64 rule against its float32 one."""
    from tools.lora_peft_reference import ADAMW

    tokens = jnp.asarray(reference["input_ids"])
    objective = LMObjective(adapter.model, tokens.shape[1] - 1, variables=adapter.variables, ema_decay=None)
    params = objective.init(jax.random.key(0))

    def loss(moving):
        stats, _ = objective.loss({**params, "params": moving}, {"text": tokens}, STEP)
        return objective.reduce_loss(stats)[0]

    value, gradient = jax.value_and_grad(loss)(params["params"])
    assert_as_exact_as_the_reference(np.asarray(value), reference["loss"], reference["loss_f64"], "the loss")
    names = sorted(key.removeprefix("grad/") for key in reference
                   if key.startswith("grad/") and not key.endswith("_f64"))
    exported = peft_factors(adapter, gradient, tmp_path / "gradient")
    assert sorted(exported) == names and len(names) == 12
    assert_leaves_as_exact(exported, {name: reference[f"grad/{name}"] for name in names},
                           {name: reference[f"grad/{name}_f64"] for name in names})

    adamw = OPTIMIZER_MAP["adamw"](ADAMW["lr"], *ADAMW["betas"], ADAMW["eps"],
                                   weight_decay=ADAMW["weight_decay"])
    moving, state = params["params"], adamw.init(params["params"])
    for _ in range(2):
        updates, state = adamw.update(jax.grad(loss)(moving), state, moving)
        moving = optax.apply_updates(moving, updates)
    start = peft_factors(adapter, params["params"], tmp_path / "start")
    stepped = peft_factors(adapter, moving, tmp_path / "stepped")
    # Each factor's move, from the float32 factors both runs start at: exact in float64.
    assert_leaves_as_exact({name: stepped[name] - start[name] for name in names},
                           {name: reference[f"stepped/{name}"] - start[name] for name in names},
                           {name: reference[f"stepped/{name}_f64"] - start[name] for name in names})


def peft_factors(adapter, factors, directory) -> dict[str, np.ndarray]:
    """The tree `factors`, shaped as the adapter's own, exported in PEFT's layout (`Adapter.save`)
    and read back, by PEFT's module path, as float64."""
    adapter.save(merge(adapter.variables, {"params": factors}), directory)
    return {key.removeprefix(lora.PEFT_PREFIX): np.asarray(value, np.float64)
            for key, value in load_file(directory / lora.PEFT_WEIGHTS).items()}


def branch_draws(adapter, tokens: np.ndarray, step: Step) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each target branch's dropout in the LM objective's training loss over `tokens` at `step`,
    keyed by the target's PEFT module path (`branch_dropout`): its input and its output."""
    from tools.lora_peft_reference import branch_dropout

    objective = LMObjective(adapter.model, tokens.shape[1] - 1, variables=adapter.variables, ema_decay=None)
    params = objective.init(jax.random.key(0))
    drawn = {}

    def capture(next_fun, args, kwargs, context):
        output = next_fun(*args, **kwargs)
        owner = branch_dropout(adapter, context)
        if owner is not None:
            drawn[owner] = (np.asarray(args[0]), np.asarray(output))
        return output

    with nn.intercept_methods(capture):
        objective.loss(params, {"text": jnp.asarray(tokens)}, step)
    return drawn


def test_an_rslora_adapter_loads_trains_and_exports_as_peft_trains_it(tmp_path):
    """rsLoRA: each target scales by its own alpha over the square root of its
    own rank (layer 1's v_proj at rank 2, down_proj at alpha 3). The adapted
    logits, and one training step, are PEFT's (`assert_trains_as_peft`)."""
    adapter, reference = adapted("rslora")
    assert (adapter.rslora, adapter.dropout) == (True, 0.0)
    logits = adapter.model.apply(adapter.variables, jnp.asarray(reference["input_ids"]))
    assert_as_exact_as_the_reference(logits, reference["adapted_logits"], reference["adapted_logits_f64"],
                                     "the adapted logits")
    assert np.abs(reference["adapted_logits_f64"] - reference["base_logits"]).max() > 1
    assert_trains_as_peft(adapter, reference, tmp_path)


def test_a_dropout_adapter_trains_as_peft_trains_it_on_the_masks_peft_ran_on(tmp_path):
    """Dropout at 0.25: a training step drops each target's branch input,
    `B A dropout(x)`, and not the base path. Each branch's dropout given the
    mask PEFT's run was given (masks.npz), keeping its input where the mask
    does at 1 / (1 - p) of the rate the branch drops at, as torch's dropout
    keeps it, the steps' loss, gradients and updates are PEFT's
    (`assert_trains_as_peft`)."""
    from tools.lora_peft_reference import MASKS, branch_dropout

    adapter, reference = adapted("dropout")
    assert (adapter.rslora, adapter.dropout) == (False, 0.25)
    with np.load(FIXTURES / "dropout" / MASKS) as given:
        masks = {name: given[name] for name in given.files}
    dropped = set()

    def given_masks(next_fun, args, kwargs, context):
        owner = branch_dropout(adapter, context)
        if owner is None:
            return next_fun(*args, **kwargs)
        dropped.add(owner)
        return jnp.where(masks[owner], args[0] / (1 - context.module.rate), 0)

    with nn.intercept_methods(given_masks):
        assert_trains_as_peft(adapter, reference, tmp_path)
    assert dropped == set(masks) and len(dropped) == 6


def test_a_training_step_drops_each_branch_input_at_the_adapters_rate():
    """A training step draws a dropout of its own on each of the six targets'
    branch inputs at the adapter's rate: the kept count within six standard
    deviations of (1 - p) n, which a dropout at rate p misses about once in
    5e8 draws, each target's mask its own and each step's its own, and the
    same step the same masks."""
    adapter, reference = adapted("dropout")
    tokens, rate = reference["input_ids"], adapter.dropout
    first, again = branch_draws(adapter, tokens, STEP), branch_draws(adapter, tokens, STEP)
    later = branch_draws(adapter, tokens, Step(step=jnp.int32(1), key=jax.random.key(2), ema=None))
    assert len(first) == 6
    kept, entries = 0, 0
    for name, (x, output) in first.items():
        mask = output != 0
        # A zero input would hide whether the dropout kept it.
        assert np.all(x != 0), name
        np.testing.assert_array_equal(again[name][1], output, err_msg=name)
        assert not np.array_equal(later[name][1] != 0, mask), name
        kept, entries = kept + int(mask.sum()), entries + mask.size
    assert abs(kept - (1 - rate) * entries) <= 6 * np.sqrt(rate * (1 - rate) * entries)
    masks = [output != 0 for _, output in first.values()]
    for one, other in itertools.combinations(masks, 2):
        assert one.shape != other.shape or not np.array_equal(one, other)


class Host(nn.Module):
    """One DenseGeneral that contracts the input's last two axes, listed out
    of order, into a (5, 6) output."""

    @nn.compact
    def __call__(self, x):
        return nn.DenseGeneral((5, 6), axis=(-1, -2), name="proj")(x)


def test_a_multi_axis_target_steps_as_its_flattened_matrix_does():
    """The branch of a target contracting (8, 6) input axes into (5, 6)
    output ones is the matrix LoRA of the flattened kernel: with x as
    [batch, 48] in the kernel's sorted axis order, A as [48, r] and B as
    [r, 30], the output is x K + bias + s x A B, and a fixed cotangent G
    gives A's gradient s x^T G B^T and B's s (x A)^T G. The float64 oracle
    computes that flattened form; numpy in float32 is the reference whose
    rounding Dew's forward and gradients are held to by the float64 rule,
    over 480, 192 and 120 entries."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(16, 6, 8)).astype(np.float32)
    model = Host()
    params = jax.tree.map(np.asarray, model.init(jax.random.key(0), jnp.asarray(x)))["params"]
    target = Target(rank=4, alpha=3.0)
    a = rng.normal(size=(6, 8, 4)).astype(np.float32)
    b = rng.normal(size=(4, 5, 6)).astype(np.float32)
    cotangent = rng.normal(size=(16, 5, 6)).astype(np.float32)
    tree = {"params": {"proj": {**params["proj"], "lora_A": a, "lora_B": b}}}
    adapter = Adapter.bound(model, tree, {("params", "proj"): target}, rslora=False, dropout=0.0, layouts={})
    scale = adapter.scale(target)

    def output(factors):
        node = {**params["proj"], **factors}
        return adapter.model.apply({"params": {"proj": node}}, jnp.asarray(x))

    def functional(factors):
        return jnp.sum(output(factors) * cotangent)

    forward = output({"lora_A": a, "lora_B": b})
    gradient = jax.grad(functional)({"lora_A": jnp.asarray(a), "lora_B": jnp.asarray(b)})

    def oracle(dtype):
        rows = x.reshape(16, 48).astype(dtype)
        kernel = params["proj"]["kernel"].reshape(48, 30).astype(dtype)
        down, up = a.reshape(48, 4).astype(dtype), b.reshape(4, 30).astype(dtype)
        seed = cotangent.reshape(16, 30).astype(dtype)
        s = dtype(scale)
        out = rows @ kernel + params["proj"]["bias"].reshape(30).astype(dtype) + s * ((rows @ down) @ up)
        return (out.reshape(16, 5, 6), (s * (rows.T @ (seed @ up.T))).reshape(6, 8, 4),
                (s * ((rows @ down).T @ seed)).reshape(4, 5, 6))

    assert tree["params"]["proj"]["kernel"].shape == (6, 8, 5, 6)
    references, truths = oracle(np.float32), oracle(np.float64)
    for label, dew, reference, truth in zip(
            ("the output", "lora_A's gradient", "lora_B's gradient"),
            (forward, gradient["lora_A"], gradient["lora_B"]), references, truths, strict=True):
        assert_as_exact_as_the_reference(np.asarray(dew), reference, truth, label)
