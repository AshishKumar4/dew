"""LoRA's rsLoRA scaling and its branch dropout in training against PEFT's own
forward, backward and AdamW steps, run in the test on the adapters
`tools/lora_peft_reference.py` wrote, and a target that contracts and produces
several axes against a float64 oracle of its forward and its factors'
gradient."""

import contextlib
import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from interop_support import fixture_ids, transformers_arms
from reference_error import as_given, assert_as_exact, with_gradients
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
"""The training step whose loss the checks' gradients are, and whose draw `dew_masks` gives PEFT's
dropout."""

TRAINING_SEEDS = {"rslora": 0, "dropout": 0}
"""Each case's declared draws of ids against PEFT's training steps (check 1), sized by `seeds_for`
paired with Dew's steps on the CPU."""

LOGITS_SEEDS = 0
"""The rsLoRA adapter's declared draws of ids against PEFT's adapted logits (check 1), sized by
`seeds_for` paired with Dew's logits on the CPU."""


def adapted(case: str):
    """The case's PEFT adapter on the tiny Llama, read by `LoRA.load`, and by PEFT itself over
    transformers' Llama, its factors trainable, as the recorder ran it."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    decoder = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    base = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float32)
    return (LoRA.load(decoder.model, decoder.variables, FIXTURES / case / "adapter", layouts=decoder.layouts),
            PeftModel.from_pretrained(base, FIXTURES / case / "adapter", is_trainable=True))


def assert_trains_as_peft(adapter, peft, case: str, tmp_path):
    """One training step's mean next-token cross entropy and its gradient in all twelve factors, and
    what two steps of Dew's AdamW (`OPTIMIZER_MAP`, at PEFT's run's settings `ADAMW`) move each factor
    by, the factors exported in PEFT's layout (`Adapter.save`): each on par with PEFT's run of the same
    steps (tools/lora_peft_reference.py `run_at`) from its float64 twin's, over drawn ids (check 1).
    Each branch's dropout, where the adapter drops, keeps the mask Dew's own first training loss draws
    over a draw's ids (`dew_masks`), which PEFT's run is given (`given`). Each move is from the float32
    factors both runs start at, so exact in float64."""
    import torch

    from tools.lora_peft_reference import ADAMW, dew_masks, run_at, twin
    from tools.torch_precision import float64

    rate = adapter.dropout
    draw = fixture_ids(LLAMA)
    objective = LMObjective(adapter.model, draw(0)[0].shape[1] - 1, variables=adapter.variables,
                            ema_decay=None)
    params = objective.init(jax.random.key(0))
    adamw = OPTIMIZER_MAP["adamw"](ADAMW["lr"], *ADAMW["betas"], ADAMW["eps"],
                                   weight_decay=ADAMW["weight_decay"])
    start = peft_factors(adapter, params["params"], tmp_path / "start")

    def masks_of(ids):
        return dew_masks(adapter, np.asarray(ids)) if rate else None

    @as_given
    def ours(ids):
        masks, tokens = masks_of(ids), jnp.asarray(ids)

        def loss(moving):
            with given(adapter, masks):
                stats, _ = objective.loss({**params, "params": moving}, {"text": tokens}, STEP)
            return objective.reduce_loss(stats)[0]

        value, gradient = jax.value_and_grad(loss)(params["params"])
        moving, state = params["params"], adamw.init(params["params"])
        for _ in range(2):
            updates, state = adamw.update(jax.grad(loss)(moving), state, moving)
            moving = optax.apply_updates(moving, updates)
        stepped = peft_factors(adapter, moving, tmp_path / "stepped")
        return {"loss": np.asarray(value),
                "gradient": peft_factors(adapter, gradient, tmp_path / "gradient"),
                "update": {name: stepped[name] - start[name] for name in start}}

    def theirs(model, precision, dtype):
        tail = "_f64" if dtype == torch.float64 else ""

        @as_given
        def run(ids):
            masks = masks_of(ids)
            with precision():
                arrays = run_at(model, torch.from_numpy(np.asarray(ids, np.int64)), dtype, masks, rate)
            return {"loss": arrays[f"loss{tail}"],
                    "gradient": {name: arrays[f"grad/{name}{tail}"].astype(np.float64) for name in start},
                    "update": {name: arrays[f"stepped/{name}{tail}"].astype(np.float64) - start[name]
                               for name in start}}
        return run

    assert len(start) == 12
    assert_as_exact(ours, theirs(peft, contextlib.nullcontext, torch.float32), draw,
                    f"{case} training steps", seeds=TRAINING_SEEDS[case],
                    truth_fn=theirs(twin(peft), float64, torch.float64))


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
    logits are on par with PEFT's from its float64 twin's over drawn ids
    (check 1), the base model's far from them, and one training step is
    PEFT's (`assert_trains_as_peft`)."""
    import torch

    from tools.lora_peft_reference import twin

    adapter, peft = adapted("rslora")
    assert (adapter.rslora, adapter.dropout) == (True, 0.0)
    draw = fixture_ids(LLAMA)
    theirs, truth = transformers_arms(peft.eval(), lambda model, ids: model(ids).logits, twin=twin(peft))
    assert_as_exact(lambda ids: adapter.model.apply(adapter.variables, ids), theirs, draw,
                    "the adapted logits", seeds=LOGITS_SEEDS, truth_fn=truth)
    # The adapter moves the logits far past rounding: the base model's are nowhere near the adapted ones.
    (ids,) = draw(0)
    with peft.disable_adapter(), torch.no_grad():
        base = peft(torch.from_numpy(ids.astype(np.int64))).logits.numpy()
    assert np.abs(base - theirs(ids)).max() > 1
    assert_trains_as_peft(adapter, peft, "rslora", tmp_path)


def test_a_dropout_adapter_trains_as_peft_trains_it_on_the_masks_peft_ran_on(tmp_path):
    """Dropout at 0.25: a training step drops each target's branch input,
    `B A dropout(x)`, and not the base path. Each branch's dropout given the
    mask Dew's own step draws, given to PEFT's run too, keeping its input
    where the mask does at 1 / (1 - p) of the rate the branch drops at, as
    torch's dropout keeps it (`given`), the steps' loss, gradients and updates
    are PEFT's (`assert_trains_as_peft`), and every target's branch drops."""
    from tools.lora_peft_reference import branch_draws, dew_masks

    adapter, peft = adapted("dropout")
    assert (adapter.rslora, adapter.dropout) == (False, 0.25)
    (ids,) = fixture_ids(LLAMA)(0)
    masks = dew_masks(adapter, ids)
    assert sorted(masks) == sorted(branch_draws(adapter, ids, STEP)) and len(masks) == 6
    assert all(not mask.all() for mask in masks.values())
    assert_trains_as_peft(adapter, peft, "dropout", tmp_path)


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

    adapter, _ = adapted("dropout")
    tokens, rate = np.load(LLAMA / "input_ids.npy"), adapter.dropout
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

    @as_given
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
