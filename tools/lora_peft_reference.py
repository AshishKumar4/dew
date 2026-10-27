#!/usr/bin/env python3
"""Write tests/fixtures/lora_peft: PEFT's rsLoRA scaling and its branch
dropout in training, run by PEFT 0.20.0 on tests/fixtures/hf/llama-tiny.

- `rslora`: a rank 4, alpha 8 adapter on q_proj, v_proj and down_proj with
  `use_rslora=True`, a rank 2 on layer 1's v_proj and alpha 3 on down_proj,
  so each target scales by its own alpha over the square root of its own
  rank. Its B factors are drawn so the adapter moves the logits.
- `dropout`: the same targets at `lora_dropout=0.25`, the model in training
  mode. PEFT's `LoraLayer.forward` drops the branch's input,
  `lora_B(lora_A(dropout(x))) * scaling`; here each target's dropout
  module keeps the mask Dew's own branch draws in the LM objective's first
  training loss (`dew_masks`), torch's `nn.Dropout` with its random bits
  fixed to Dew's: `x * mask / (1 - p)`.

Each case writes PEFT's adapter directory, the fixture's input ids, the
base and adapted logits, the mean next-token cross entropy and its
gradient with respect to every adapter factor, and every factor after two
torch AdamW steps (`ADAMW`) on that loss, in float32 and in float64.

The loss is one number, so one run of it against PEFT's is one rounding draw
against another. `orders.npz` beside each case holds PEFT's float32 loss's
distance from its float64 one over the K-order rule's residual orders
(tests/reference_error.py, `ORDERS`): Dew writes the base model and the adapter
in each order (`write_orders`, tests/residual_orders.py, the factors' residual
sides read off their declared axes), with the dropout case's masks moved with
the branch inputs that carry the residual, and PEFT's run reads them back
(`record_orders`), on an AVX-512 (x86-64-v4) CPU, the CI runners' ISA: torch's
float32 products sum in other orders on an AVX2 one.
The dropout case reads its masks, one per target keyed by the target's
PEFT module path, from `masks.npz` beside it, which Dew writes first over
the committed adapter (`masks`, in Dew's environment); the PEFT run (the
`peft` cohort of tools/reference_releases.py) follows, both steps on
those masks:

    PYTHONPATH=src python tools/lora_peft_reference.py masks
    ~/.cache/dew/reference-venvs/peft/bin/python tools/lora_peft_reference.py
    PYTHONPATH=src python tools/lora_peft_reference.py write-orders <dir>
    ~/.cache/dew/reference-venvs/peft/bin/python tools/lora_peft_reference.py orders <dir>
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LLAMA = ROOT / "tests" / "fixtures" / "hf" / "llama-tiny"
FIXTURE = ROOT / "tests" / "fixtures" / "lora_peft"
TARGETS = ["q_proj", "v_proj", "down_proj"]
CASES = {"rslora": {"use_rslora": True, "lora_dropout": 0.0},
         "dropout": {"use_rslora": False, "lora_dropout": 0.25}}
PREFIX = "base_model.model."


class Masked(torch.nn.Module):
    """torch's dropout on a recorded mask: `x * mask / (1 - p)`."""

    def __init__(self, mask: torch.Tensor, rate: float):
        super().__init__()
        self.mask, self.rate = mask, rate

    def forward(self, x):
        return x * self.mask.to(x.dtype) / (1 - self.rate)


def adapter(case: str, settings: dict, out: Path):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float32)
    config = LoraConfig(r=4, lora_alpha=8, target_modules=TARGETS,
                        rank_pattern={"layers.1.self_attn.v_proj": 2}, alpha_pattern={"down_proj": 3.0},
                        **settings)
    model = get_peft_model(base, config)
    generator = torch.Generator().manual_seed(1)
    for name, parameter in model.named_parameters():
        if "lora_B" in name:
            parameter.data = torch.randn(parameter.shape, generator=generator) * 0.2
    directory = out / case / "adapter"
    model.save_pretrained(directory)
    (directory / "README.md").unlink()
    saved = json.loads((directory / "adapter_config.json").read_text())
    saved["base_model_name_or_path"] = str(LLAMA.relative_to(ROOT))
    # PEFT holds the targets as a set, which a run lists in its own order.
    saved["target_modules"] = sorted(saved["target_modules"])
    (directory / "adapter_config.json").write_text(json.dumps(saved, indent=2, sort_keys=True) + "\n")
    return model


MASKS = "masks.npz"
"""The dropout case's keep masks, Dew's own (`dew_masks`), which the PEFT run reads."""

ADAMW = {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 0.01}
"""The optimizer of the two training steps each case records: torch's AdamW, its decay decoupled."""


def branch_dropout(adapter, context) -> str | None:
    """The PEFT module path of the target whose branch dropout a Flax method interception's
    `context` calls, or None for any other call."""
    from flax import linen as nn

    module = context.module
    if not (isinstance(module, nn.Dropout) and context.method_name == "__call__"
            and ("params", *module.path[:-1]) in adapter.targets):
        return None
    return "model." + ".".join(module.path[:-1]).replace("layers_", "layers.")


def branch_draws(adapter, tokens: np.ndarray, step) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Each target branch's dropout in the LM objective's training loss over `tokens` at `step`,
    keyed by the target's PEFT module path (`branch_dropout`): its input and its output, read
    through Flax's method interception, the objective reading every token but the last."""
    import jax
    import jax.numpy as jnp
    from flax import linen as nn

    from dew.objectives.lm import LMObjective

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


def dew_masks(adapter, tokens: np.ndarray) -> dict[str, np.ndarray]:
    """The keep mask each target's branch dropout draws in the LM objective's first training
    loss over `tokens` (`Step(0, key(1))`, `branch_draws`), keyed by the target's PEFT module
    path: `[batch, inputs, width]`. A kept entry is one the dropout passed, as no input entry is
    zero."""
    import jax
    import jax.numpy as jnp

    from dew.objectives.base import Step

    masks = {}
    for owner, (x, output) in branch_draws(adapter, tokens, Step(step=jnp.int32(0), key=jax.random.key(1),
                                                                 ema=None)).items():
        if not np.all(x != 0):
            raise ValueError("a zero in a branch's input hides whether the dropout kept it")
        masks[owner] = output != 0
    return masks


ORDER_SEED = 6
"""The seed of the residual orders `write_orders` draws."""


def write_orders(out: Path) -> None:
    """Each case's base model and adapter written under `out`/<case>/<k>/ in each of `ORDERS`
    residual orders (tests/residual_orders.py, `in_order`), order 0 the identity: the base
    through `Pretrained.save`, the adapter through `Adapter.save`, and the dropout case's masks
    in `MASKS`."""
    sys.path.insert(0, str(ROOT / "tests"))
    from reference_error import ORDERS
    from residual_orders import orders, residual_width

    from dew.interop.pretrained import Pretrained
    from dew.lora import LoRA

    decoder = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    for case in CASES:
        adapter = LoRA.load(decoder.model, decoder.variables, FIXTURE / case / "adapter",
                            layouts=decoder.layouts)
        whole, residual_inputs = adapter_orders(adapter)
        drawn = orders(residual_width(whole), ORDERS, ORDER_SEED)
        (out / case).mkdir(parents=True, exist_ok=True)
        np.save(out / case / "orders.npy", drawn)
        masks = None
        if CASES[case]["lora_dropout"]:
            with np.load(FIXTURE / case / MASKS) as saved:
                masks = {key: saved[key] for key in saved}
        for k, order in enumerate(drawn):
            variables, moved_masks = in_order(adapter, whole, residual_inputs, order, masks)
            decoder.save(out / case / str(k) / "base",
                         variables={**decoder.variables, "params": variables["frozen"]})
            adapter.save(variables, out / case / str(k) / "adapter")
            if moved_masks is not None:
                np.savez(out / case / str(k) / MASKS, **moved_masks)
        print(f"{out / case}: {len(drawn)} orders, residual branch inputs {sorted(residual_inputs)}")


def adapter_orders(adapter) -> tuple[dict, set[str]]:
    """The adapter's frozen base and its factors as one `{"params": ...}` tree, which an order
    permutes whole (tests/residual_orders.py, the factors' residual sides read off their declared
    axes), and the PEFT module paths of the targets whose branch input is the residual stream:
    those whose `lora_A`'s first declared axis is 'embed'."""
    import jax

    from dew.batch import thaw
    from dew.nn.sharding import declarations, declared_axes, parameter_path

    whole = thaw(adapter.variables)
    declared = declarations(whole["params"])
    residual_inputs = {
        "model." + ".".join(parameter_path(path)[:-1]).replace("layers_", "layers.")
        for path, leaf in jax.tree_util.tree_leaves_with_path(whole["params"])
        if parameter_path(path)[-1] == "lora_A" and declared_axes(path, leaf.ndim, declared)[0] == "embed"}
    return whole, residual_inputs


def in_order(adapter, whole: dict, residual_inputs: set[str], order: np.ndarray,
             masks: dict[str, np.ndarray] | None) -> tuple[dict, dict[str, np.ndarray] | None]:
    """The adapter's variables with the residual stream's units in `order` (`adapter_orders`),
    and `masks`, a dropout mask's last axis moved with its target's branch input where that input
    is the residual stream."""
    sys.path.insert(0, str(ROOT / "tests"))
    from residual_orders import permuted

    from dew.batch import freeze
    from dew.lora import FACTORS

    variables = freeze(permuted(whole, order), lambda path: path[-1] in FACTORS)
    if masks is None:
        return variables, None
    return variables, {key: mask[..., order] if key in residual_inputs else mask
                       for key, mask in masks.items()}


def record_orders(drawn: Path) -> None:
    """PEFT's float32 loss over each case's residual orders under `drawn` (`write_orders`), its
    distance from the case's float64 loss, written to the case's `orders.npz` beside the orders.
    A permutation of the residual stream leaves the loss exact, so every order's truth is the
    identity's; order 0 must reproduce the case's recorded float32 loss bit for bit."""
    from reference_releases import require

    require("peft")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    input_ids = torch.from_numpy(np.load(LLAMA / "input_ids.npy")).long()
    for case, settings in CASES.items():
        with np.load(FIXTURE / case / "reference.npz") as stored:
            recorded, truth = stored["loss"], np.float64(stored["loss_f64"])
        drawn_orders = np.load(drawn / case / "orders.npy")
        distances = []
        for k in range(len(drawn_orders)):
            base = AutoModelForCausalLM.from_pretrained(drawn / case / str(k) / "base", dtype=torch.float32)
            model = PeftModel.from_pretrained(base, drawn / case / str(k) / "adapter", is_trainable=True)
            masks = None
            if settings["lora_dropout"]:
                with np.load(drawn / case / str(k) / MASKS) as saved:
                    masks = {key: saved[key] for key in saved}
            loss = run_at(model, input_ids, torch.float32, masks, settings["lora_dropout"])["loss"]
            if k == 0:
                assert np.array_equal(loss, recorded), "order 0 must reproduce the case's recorded loss"
            distances.append(abs(np.float64(loss) - truth))
        np.savez(FIXTURE / case / "orders.npz",
                 orders=drawn_orders.astype(np.min_scalar_type(drawn_orders.max())),
                 loss=np.asarray(distances))
        print(f"{case}: {len(distances)} orders, loss distances / order 0's",
              np.round(np.asarray(distances) / distances[0], 2))


def write_masks() -> None:
    """Write the dropout case's masks: Dew's draw over the committed adapter."""
    from dew.interop.pretrained import Pretrained
    from dew.lora import LoRA

    decoder = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    adapter = LoRA.load(decoder.model, decoder.variables, FIXTURE / "dropout" / "adapter",
                        layouts=decoder.layouts)
    drawn = dew_masks(adapter, np.load(LLAMA / "input_ids.npy"))
    np.savez(FIXTURE / "dropout" / MASKS, **drawn)
    print(f"{FIXTURE / 'dropout' / MASKS}: {len(drawn)} masks")


@contextlib.contextmanager
def widened(model):
    """`model`'s float64 twin (`float64_twin`): the same base and adapter built inside `float64`, so
    every value their constructors compute is float64's (the rotary tables' among them), loaded with
    `model`'s weights and run inside `float64`, every float32 pin of transformers' Llama (5.16.1)
    widened: its RMSNorm's cast (modeling_llama.py:64), its rotary tables' (:108, :115-117) and its
    attention softmax's (:208)."""
    import copy

    from peft import get_peft_model
    from torch_precision import float64, float64_twin
    from transformers import AutoModelForCausalLM

    def build():
        base = AutoModelForCausalLM.from_config(copy.deepcopy(model.get_base_model().config),
                                                dtype=torch.float64)
        return get_peft_model(base, copy.deepcopy(model.peft_config["default"]))

    twin = float64_twin(model, build)
    with float64():
        yield twin


def run(model, input_ids: torch.Tensor, dtype, drawn: dict[str, np.ndarray] | None, rate: float) -> dict:
    if dtype == torch.float64:
        with widened(model) as wide:
            return run_at(wide, input_ids, dtype, drawn, rate)
    return run_at(model.to(dtype), input_ids, dtype, drawn, rate)


def run_at(model, input_ids: torch.Tensor, dtype, drawn: dict[str, np.ndarray] | None, rate: float) -> dict:
    """The adapted logits, the loss and every factor's gradient of `model` as it stands at `dtype`,
    and every factor after two AdamW steps from there, which leave `model`'s factors as they were."""
    if drawn is None:
        model.eval()
    else:
        model.train()
        for name, module in model.named_modules():
            key = name.removeprefix(PREFIX)
            if key in drawn:
                # The last position's output reaches no next-token loss, so its branch keeps all.
                mask = np.ones((*input_ids.shape, drawn[key].shape[-1]), bool)
                mask[:, :drawn[key].shape[1]] = drawn[key]
                module.lora_dropout["default"] = Masked(torch.as_tensor(mask), rate)
    model.zero_grad(set_to_none=True)
    logits, loss = next_token_loss(model, input_ids)
    loss.backward()
    tail = "_f64" if dtype == torch.float64 else ""
    arrays = {f"adapted_logits{tail}": logits.detach().numpy(), f"loss{tail}": loss.detach().numpy()}
    factors = {name.removeprefix(PREFIX).replace(".default", ""): parameter
               for name, parameter in model.named_parameters() if parameter.requires_grad}
    for key, parameter in factors.items():
        arrays[f"grad/{key}{tail}"] = parameter.grad.numpy().copy()
    initial = {key: parameter.detach().clone() for key, parameter in factors.items()}
    optimizer = torch.optim.AdamW(factors.values(), **ADAMW)
    optimizer.step()
    model.zero_grad(set_to_none=True)
    next_token_loss(model, input_ids)[1].backward()
    optimizer.step()
    with torch.no_grad():
        for key, parameter in factors.items():
            arrays[f"stepped/{key}{tail}"] = parameter.detach().numpy().copy()
            parameter.copy_(initial[key])
    return arrays


def next_token_loss(model, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`model`'s logits over `input_ids` and their mean next-token cross entropy."""
    logits = model(input_ids).logits
    return logits, torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), input_ids[:, 1:].reshape(-1))


def main() -> None:
    from reference_releases import require

    require("peft")
    from transformers import AutoModelForCausalLM

    input_ids = torch.from_numpy(np.load(LLAMA / "input_ids.npy")).long()
    with torch.no_grad():
        base = AutoModelForCausalLM.from_pretrained(LLAMA, dtype=torch.float32)
        base_logits = base(input_ids).logits.numpy()
    for case, settings in CASES.items():
        model = adapter(case, settings, FIXTURE)
        rate = settings["lora_dropout"]
        drawn = None
        if rate:
            with np.load(FIXTURE / case / MASKS) as saved:
                drawn = {key: saved[key] for key in saved}
        arrays = {"input_ids": input_ids.numpy(), "base_logits": base_logits}
        for dtype in (torch.float32, torch.float64):
            arrays.update(run(model, input_ids, dtype, drawn, rate))
        np.savez(FIXTURE / case / "reference.npz", **arrays)
        moved = np.abs(arrays["adapted_logits_f64"] - base_logits).max()
        print(f"{FIXTURE / case}: logits moved {moved:.3f}, "
              f"{sum(1 for key in arrays if key.startswith('grad/') and key.endswith('_f64'))} gradients")


if __name__ == "__main__":
    if sys.argv[1:] == ["masks"]:
        write_masks()
    elif sys.argv[1:2] == ["write-orders"]:
        write_orders(Path(sys.argv[2]))
    elif sys.argv[1:2] == ["orders"]:
        record_orders(Path(sys.argv[2]))
    else:
        main()
