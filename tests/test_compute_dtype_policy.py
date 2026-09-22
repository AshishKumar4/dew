"""What `--model.dtype bfloat16` promises the arithmetic, family by family.

bf16 is a compute dtype: the parameters stay fp32 and the head the loss reads
stays fp32 (test_precision_policy.py), so the whole of what the flag buys is
the dtype the matmuls run in. test_causal_transformer.py states that for one
module by capturing an intermediate
(`test_bf16_compute_runs_the_feed_forward_in_bf16`); an intermediate is only
visible where a module sows one, and a backward pass sows nothing at all. The
compiler sees every product, forward and backward, so this file states the
same promise where every family can be held to it: the matmul FLOPs of a
compiled loss-and-grad, split by the dtype of the operands the machine
multiplies.

An fp32 operand is what costs the rate. XLA promotes the other side of a
mixed dot, so an f32 x bf16 product runs at the fp32 (on Ada, TF32) rate, half
of bf16's; that is findings 1 and 3 of the MFU audit
(mfu-audit-571d8d3b/REPORT.json), where a dense feed-forward left in fp32 cost
23.6% of a step and MoE tangents cast to fp32 landed on TF32 tensor cores for
another 24.85%. The FLOP formulas are dew's own
(`dew.telemetry.instrumentation`), so the denominator here is the number the
trainer reports as step FLOPs; the private helpers are imported because the
public `hlo_flops` sums the split this file is about away.

Which HLO carries the answer depends on the backend. The CPU backend cannot
emit a bf16 dot at all - `float-normalization-bf16` rewrites every one of them
to f32 before codegen, and disabling that pass makes CPU compilation fail in
`dot_op_emitter.cc` - so on CPU every family's optimized module reads ~99%
fp32 whatever the model asked for, and the module the compiler was *given* is
the one that still holds the model's own dtypes. On an accelerator the
optimized module keeps them, and is the better text because it is what runs.
The compile happens either way: a graph that does not compile is not a graph
whose dtypes are worth counting.
"""

import math
import re
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax
import pytest
from test_precision_policy import build_model, tiny_inputs

import dew.nn.backbones.flux  # noqa: F401  registers "flux_transformer"
import dew.nn.backbones.sd3  # noqa: F401  registers "sd3_transformer"
from dew import models  # the attribute import is what registers every family
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.base import Step, scalar_loss
from dew.objectives.lm import LMObjective
from dew.objectives.lm.objective import TEXT_KEY
from dew.telemetry.instrumentation import (
    _COMPUTATION,
    _INSTRUCTION,
    _Instruction,
    _dims,
    _instruction_flops,
    _parse,
    _split_operands,
    _weights,
    hlo_flops,
)

# The element type of an HLO shape: `f32[4,8]{1,0}` and `bf16[]` both start
# with it, and a tuple shape starts with its first member's.
_ELEMENT = re.compile(r"([a-z]\w*)\[")
# The flax module path XLA carries from the jaxpr, which is what names the
# module that upcast.
_OP_NAME = re.compile(r'op_name="([^"]*)"')

# An operand the machine multiplies at the fp32 rate. f64 is here for
# completeness; nothing in dew asks for it.
FP32 = ("f32", "f64")

# How much of a family's matmul work may still be fp32. The fp32 boundaries a
# bf16 run keeps on purpose - the output head, the timestep embedding - are
# rounding error at a trained width; at the toy width these cases build they
# are not, which is what a failure here reports.
BUDGET = 0.01

# One dot big enough to matter in the language-model head test below. The
# vocabulary head is the widest matmul in a language model, so an fp32 one is
# never small; a tiny fp32 dot (a scalar scale, a probe) is not what that test
# is about.
LARGE = 1e6

# The families whose loss is the cross entropy of a logits tensor; the rest
# predict a sample and take a mean square.
LOGITS = ("causal_transformer", "diffusion_gemma", "multimodal_transformer")

# A family whose tiny build needs something this file cannot make: the value
# names the fixture or checkpoint, and the case skips saying so rather than
# disappearing. Every registered family builds from its own fields today, so
# the table is empty and the parametrization covers all of them.
FIXTURES: dict[str, str] = {}

# A family is registered by importing the module that defines it, so the two
# published transformers are imported above: the set of cases is then the same
# whether this file runs alone or after a session that happened to import them.
FAMILIES = sorted(models)


@dataclass(frozen=True)
class Matmul:
    """One matmul of a compiled module: what it multiplies and what it costs."""

    computation: str
    name: str
    op: str
    result: str
    operands: tuple[str, ...]
    module: str
    """The flax module path XLA kept in the instruction's metadata, empty
    where the text carries none."""
    flops: float
    """FLOPs per call of the entry computation, the way
    `dew.telemetry.instrumentation.hlo_flops` counts them: an instruction a
    fusion or an annotated loop runs many times counts as many times. A loop
    with no trip count in the text counts its body once (see `matmuls`)."""

    @property
    def fp32(self) -> bool:
        return any(operand.split("[")[0] in FP32 for operand in self.operands)

    def __str__(self) -> str:
        return (f"{self.flops:>15,.0f} FLOPs  %{self.name} = {self.result} "
                f"{self.op}({', '.join(self.operands)})"
                + (f"\n{'':>21}{self.module}" if self.module else ""))


def _typed(dtype: str, dims: tuple[int, ...]) -> str:
    return f"{dtype}[{','.join(str(dim) for dim in dims)}]"


def matmuls(text: str) -> list[Matmul]:
    """Every matmul of one HLO module, by name, operand dtype and FLOPs.

    `hlo_flops` reads the same instructions with the same formulas and returns
    their sum; this keeps each one, which is what a dtype split needs, and its
    name, which is what a failure has to point at.
    """
    computations, entry = _parse(text)
    weights = _weights(computations, entry)
    found: list[Matmul] = []
    current, dtypes = None, {}
    for line in text.splitlines():
        stripped = line.strip()
        header = _COMPUTATION.match(line)
        if header and " = " not in stripped:
            current, dtypes = header.group("name"), {}
            continue
        if stripped == "}":
            current = None
            continue
        match = _INSTRUCTION.match(line)
        if match is None or current is None:
            continue
        name = match.group("name")
        element = _ELEMENT.search(match.group("shape"))
        # Every operand is defined before its use inside its own computation,
        # a fused computation's parameters included, so the table built as the
        # lines go by resolves them.
        dtypes[name] = element.group(1) if element else "?"
        weight = weights.get(current, 0.0)
        if not weight:
            continue
        operands, attributes = _split_operands(match.group("rest"))
        instruction = _Instruction(match.group("op"), _dims(match.group("shape")),
                                   operands, attributes)
        shapes = computations[current].shapes
        flops = _instruction_flops(instruction, shapes)
        if not flops:
            continue
        module = _OP_NAME.search(attributes)
        found.append(Matmul(
            current, name, instruction.op, _typed(dtypes[name], instruction.dims),
            tuple(_typed(dtypes.get(operand, "?"), shapes.get(operand, ()))
                  for operand in operands),
            module.group(1) if module else "",
            # A loop XLA has not yet annotated with a trip count weighs
            # infinity, which no share can be read from. The module given to
            # the compiler is such a module, and counting its body once is
            # what a share of it means: every matmul in one body, fp32 and
            # bf16 alike, is counted the same number of times.
            (1.0 if not math.isfinite(weight) else weight) * flops))
    return found


def fp32_share(found: list[Matmul]) -> tuple[float, float, float]:
    """The fp32-operand FLOPs, the total, and the fraction."""
    total = sum(matmul.flops for matmul in found)
    fp32 = sum(matmul.flops for matmul in found if matmul.fp32)
    return fp32, total, (fp32 / total if total else 0.0)


def listing(found: list[Matmul]) -> str:
    """The fp32 matmuls, heaviest first, one per line with its module path."""
    return "\n".join(f"  {matmul}"
                     for matmul in sorted(found, key=lambda m: -m.flops) if matmul.fp32)


def matmul_text(function, *arguments) -> str:
    """The HLO of `function` whose dtypes are the ones the model asked for.

    Compiled first, because a family whose loss-and-grad does not compile has
    no dtypes worth counting. The text is the optimized module wherever the
    backend can run a bf16 dot, and the module the compiler was given on CPU,
    where it cannot (see this file's docstring).
    """
    lowered = jax.jit(function).lower(*arguments)
    compiled = lowered.compile()
    if jax.default_backend() != "cpu":
        return compiled.as_text()
    return lowered.compiler_ir(dialect="hlo").get_hlo_module().to_string()


def family_variables(family, model, args, kwargs, rng):
    """The variables a forward of `model` needs, its cache primed where the
    family refines against one."""
    variables = model.init(rng, *args, **kwargs)
    if family != "diffusion_gemma":
        return variables
    # DiffusionGemma refines a canvas against an encoded prompt, so its
    # forward reads a cache the prompt was written into first.
    prompt = jnp.zeros((1, 8), jnp.int32)
    cache = model.apply(variables, 1, method=model.init_cache, mutable=["cache"])[1]["cache"]
    cache = model.apply({**variables, "cache": cache}, prompt, method=model.encode,
                        mutable=["cache"])[1]["cache"]
    return {**variables, "cache": cache}


def family_loss(family: str, rng):
    """One family at the smallest size its tests build, under bf16 compute,
    as a loss of its parameters and the fp32 parameters themselves."""
    model = build_model(family, dtype="bfloat16")
    inputs = tiny_inputs(family, rng)
    args, kwargs = inputs if isinstance(inputs[0], tuple) else (inputs, {})
    variables = family_variables(family, model, args, kwargs, rng)
    held = {name: value for name, value in variables.items() if name != "params"}

    def loss(params):
        prediction = model.apply({**held, "params": params}, *args, **kwargs)
        prediction = prediction.astype(jnp.float32)
        if family in LOGITS:
            targets = jnp.zeros(prediction.shape[:-1], jnp.int32)
            return optax.softmax_cross_entropy_with_integer_labels(prediction, targets).mean()
        return jnp.mean(prediction ** 2)

    return loss, variables["params"]


@pytest.mark.parametrize("family", FAMILIES)
def test_a_bf16_family_runs_its_matmuls_in_bf16(family, rng):
    """Under bf16 compute with fp32 parameters, the fp32 share of a family's
    loss-and-grad matmuls is rounding error.

    The gradient is half the point: a forward can be bf16 throughout and still
    hand its backward an fp32 cotangent, which promotes both operands of every
    product that cotangent enters, and a backward is twice the forward's
    matmuls (audit finding 11: 108 forward against 220 backward dots).
    """
    fixture = FIXTURES.get(family)
    if fixture is not None:
        pytest.skip(f"{family} cannot be built at a tiny size without {fixture}")
    loss, params = family_loss(family, rng)
    found = matmuls(matmul_text(jax.value_and_grad(loss), params))
    assert found, f"{family} compiled to no matmul at all"

    fp32, total, share = fp32_share(found)
    assert share < BUDGET, (
        f"{family}: {fp32:,.0f} of {total:,.0f} matmul FLOPs ({100 * share:.2f}%) have an "
        f"fp32 operand, over the {100 * BUDGET:.0f}% budget. The dots and the modules "
        f"they came from:\n{listing(found)}")


def test_the_split_counts_every_flop_dew_counts(rng):
    """The denominator is dew's own step-FLOP number, not a second opinion:
    splitting the module by operand dtype adds back up to `hlo_flops`."""
    loss, params = family_loss("simple_dit", rng)
    text = matmul_text(jax.value_and_grad(loss), params)
    _, total, _ = fp32_share(matmuls(text))
    assert total == pytest.approx(hlo_flops(text))


def lm_loss_and_grad(rng):
    """A tiny bf16 language model under the real LM objective, as a loss of
    its variables. The vocabulary is wide enough, and the backbone narrow
    enough, that the head's own dot is the only fp32 matmul in the graph that
    can pass `LARGE`: the next one is the attention product at 262,144."""
    model = CausalTransformer(vocab_size=512, emb_features=64, num_layers=1, num_heads=2,
                              mlp_features=128, max_seq_len=32, dtype=jnp.bfloat16)
    # One chunk, so the head is one dot rather than four small ones: the
    # question is the dtype of the product, not how it is tiled.
    objective = LMObjective(model, seq_len=32, head_chunks=1, ema_decay=None)
    variables = objective.init(rng)
    batch = {TEXT_KEY: jax.random.randint(rng, (2, 33), 0, 512)}
    step = Step(step=jnp.asarray(0), key=rng, ema=None)

    def loss(params):
        return scalar_loss(objective, params, batch, step)[0]

    return jax.value_and_grad(loss), variables


@pytest.mark.xfail(strict=True, reason="chunked head upcasts hidden; fix pending")
def test_the_lm_head_multiplies_bf16_states(rng):
    """The vocabulary head is the widest matmul a language model runs, and
    `dew.objectives.lm.chunked.head_logits` casts the states to fp32 before it
    (chunked.py:63), so both operands of that dot are fp32 and it runs at half
    rate - 7.66% of a dense step, 7.96% of the MoE step (audit finding 7).
    fp32 accumulation is what the head needs for an exact logsumexp, and
    `preferred_element_type=jnp.float32` already asks for it; the operand cast
    is the separate thing, and this fails until it goes.
    """
    value_and_grad, variables = lm_loss_and_grad(rng)
    found = matmuls(matmul_text(value_and_grad, variables))
    large = [matmul for matmul in found if matmul.fp32 and matmul.flops > LARGE]
    assert not large, (
        f"{len(large)} fp32 matmul(s) above {LARGE:,.0f} FLOPs in the LM objective's "
        f"loss and grad:\n{listing(large)}")
