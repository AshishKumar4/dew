"""The elementwise activations, one arithmetic each, and the names that pick them.

The gelus and silu compute in at least float32 and round once to the input's
dtype, as torch's bf16 kernels do (docs/performance.md): a bf16 polynomial or
erf moves the negative tail before it rounds. The exact gelu has two forms,
each kept for a measured reason. `gelu_exact` is the erfc form, the closest
to float64, which the towers' 'gelu' and the gated 'geglu_exact' compute;
`gelu_exact_torch` is torch's `1 + erf`, which the decoders' ungated
'gelu_exact' and ModernBERT's head compute, since their converted
checkpoints were trained with its rounding, and DiT's exact MLP, where it
made an A100 training step 9.8% faster (`dew.nn.dit`). `gelu_tanh` is the tanh
approximation, torch's `gelu(approximate='tanh')`. `quick_gelu` stays in
the input's dtype, as transformers' `QuickGELUActivation` does, and relu and
relu2 are exact in any dtype but for relu2's one rounded product.

Two tables name them. `activation` reads Hugging Face's `hidden_act` names
(transformers 5.16.1 `activations.ACT2CLS`): 'gelu' is the erfc form there.
Every tower checks its own subset of names before asking, and refuses the
rest in its own words. `UNGATED` holds Dew's names for an ungated decoder
or encoder feed-forward, which saved run records store: 'gelu' is the tanh
form there and 'gelu_exact' torch's erf form. The gated names map onto these in
`gated_product`, which owns the product's rounding and its kernel choice, and which
the dense and the expert MLPs both take their product from.
"""

import dataclasses
import functools
import math
from collections.abc import Callable

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.typing import Dtype

from .derivatives import Protected, contracted, declared_tangent
from .kernels import norm_swiglu
from .precision import at_least_fp32, capped_logits, rounded_to


def gelu_exact(x: jax.Array) -> jax.Array:
    """`jax.nn.gelu(approximate=False)`, `0.5 * x * erfc(-x / sqrt(2))`, in at least fp32, in
    `x`'s dtype. Over 2^20 normal fp32 inputs at scale 3 it lies 5.6e-8 RMS from float64 where
    the `1 + erf` form and torch's gelu lie 1.0e-7, and 4.5e-10 against 1.3e-7 below x = -3.
    In bf16 both forms lie 1.76e-3 RMS from float64."""
    return jax.nn.gelu(x.astype(at_least_fp32(x.dtype)), approximate=False).astype(x.dtype)


def gelu_exact_torch(x: jax.Array) -> jax.Array:
    """Torch's `0.5 * x * (1 + erf(x / sqrt(2)))` in at least fp32, in `x`'s dtype, for converted
    decoders: in fp32 its `1 + erf` rounds to zero from x = -5.21 down (torch's from -5.54), where
    the erfc form keeps a tail that a trained layer amplifies (tests/test_hf_gpt_neox.py)."""
    work = x.astype(at_least_fp32(x.dtype))
    return (.5 * work * (1 + jax.lax.erf(work * math.sqrt(.5)))).astype(x.dtype)


def gelu_tanh(x: jax.Array) -> jax.Array:
    """`0.5 * x * (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))` in at least fp32, in `x`'s dtype, as
    its identity `x * sigmoid(sqrt(8 / pi) (x + 0.044715 x^3))`, which XLA:GPU leaves as written: the
    tanh form after a biased matmul becomes cuBLASLt's GELU epilogue, which lay 2.4e-6 from float64
    relative to its size on an A100 where this form lay 1.1e-7 (each alone 5.6e-8 and 4.9e-8). The
    sigmoid also keeps the tail `1 + tanh` cancels: on a CPU the form lies 3.5e-8 from float64 against
    `jax.nn.gelu`'s 4.5e-8 and torch's 2.8e-8, and 3.1e-7 against 1.3e-4 and 3.4e-5 below x = -3,
    over 2^22 normal inputs at scale 3. The sigmoid is `lax.logistic` at its highest accuracy, whose
    derivative is `s(z) s(-z)`: the default `s (1 - s)` reads the rounded `s(z)`, which from x near 5
    in fp32 rounds to 1 and drops the derivative's `x s(z) s(-z) z'`, a derivative of 1 where
    float64's is 1.0000016. The forward is the default's on a CPU and a GPU, which lower either alike."""
    work = x.astype(at_least_fp32(x.dtype))
    inner = math.sqrt(8 / math.pi) * (work + 0.044715 * work ** 3)
    return (work * jax.lax.logistic(inner, accuracy=jax.lax.AccuracyMode.HIGHEST)).astype(x.dtype)


def quick_gelu(x: jax.Array) -> jax.Array:
    """CLIP's activation, `x * sigmoid(1.702 x)`, ACT2FN's `quick_gelu`."""
    return x * jax.nn.sigmoid(1.702 * x)


def silu(x: jax.Array) -> jax.Array:
    """`x * sigmoid(x)` in at least fp32, in `x`'s dtype."""
    return jax.nn.silu(x.astype(at_least_fp32(x.dtype))).astype(x.dtype)


def relu2(x: jax.Array) -> jax.Array:
    """The squared relu, Nemotron-H's and ACT2FN's `relu2`."""
    return jnp.square(jax.nn.relu(x))


Activation = Callable[[jax.Array], jax.Array]
HIDDEN_ACT: dict[str, Activation] = {'gelu': gelu_exact, 'gelu_pytorch_tanh': gelu_tanh,
                                     'quick_gelu': quick_gelu, 'silu': silu, 'relu': jax.nn.relu}
"""The activations Hugging Face's `hidden_act` names, as Dew computes them."""


def activation(name: str) -> Activation:
    """The activation Hugging Face's `hidden_act` `name` computes (`HIDDEN_ACT`)."""
    return HIDDEN_ACT[name]


UNGATED: dict[str, Activation] = {
    'gelu': gelu_tanh, 'gelu_exact': gelu_exact_torch, 'relu': jax.nn.relu, 'relu2': relu2,
    'quick_gelu': quick_gelu, 'silu': silu}
"""The activations an ungated feed-forward takes, by Dew's names."""


_SLOPES: dict[Activation, tuple[float, float | None]] = {
    gelu_exact: (1.1289122, 0.7978924), gelu_exact_torch: (1.1289122, 0.7978924),
    gelu_tanh: (1.1290011, 0.7978924), quick_gelu: (1.0998479, 0.851009), silu: (1.0998444, 0.5000031),
    jax.nn.relu: (1.0, None)}
"""Upper bounds on each activation's `sup |phi'|` and `sup |phi''|` over the reals, which a curvature bound
through it reads (`dew.objectives.lm.adaptation`): the exact GELU's are Phi(sqrt 2) + sqrt 2 phi(sqrt 2) and
2 phi(0), SiLU's sigma(x)(1 + x(1 - sigma(x))) at its root 2 + x(1 - 2 sigma(x)) = 0 and 1/2, quick GELU's
SiLU's at 1.702 x, and 1.702 / 2, each its float64 maximum on a grid of step 2e-5 plus half the step times
the next derivative's maximum there, rounded up at the seventh decimal (`tests/test_chunk_adaptation.py`).
ReLU's `phi''` is a point mass at zero, which no number bounds (None)."""


XIELU = 'xielu'
"""The ungated feed-forward's learned activation (`Xielu`), by the name a decoder's `mlp` gives it."""


class Xielu(nn.Module):
    """Apertus's xIELU (transformers' `XIELUActivation`, arXiv 2411.13010): `alpha_p * x**2 + beta * x` for
    a positive input, `(expm1(min(x, eps)) - x) * alpha_n + beta * x` otherwise, where `alpha_p` is the
    softplus of its learned `alpha_p` and `alpha_n` is `beta` plus the softplus of its own, each a `[1]`
    parameter, computed in the input's dtype as the reference's Python form computes it: Apertus holds
    the activation's parameters in the model's dtype (`XIELUActivation(dtype=config.dtype)`), which Dew's
    parameters, cast to the compute dtype as every layer's are, stand for; each softplus runs in at least
    float32 and rounds once, as torch's does. `beta` and `eps` are the reference's buffers, which its
    checkpoint stores and no step trains: Dew holds them as constants (`constants` collection), their
    defaults the reference's."""
    alpha_p_init: float = 0.8
    alpha_n_init: float = 0.8
    beta: float = 0.5
    eps: float = -1e-6
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        def inverse_softplus(value: float):
            return lambda key, shape, dtype: jnp.full(shape, math.log(math.expm1(value)), dtype)

        alpha_p = self.param('alpha_p', inverse_softplus(self.alpha_p_init), (1,), self.param_dtype)
        alpha_n = self.param('alpha_n', inverse_softplus(self.alpha_n_init - self.beta), (1,),
                             self.param_dtype)
        beta = self.variable('constants', 'beta', lambda: jnp.asarray(self.beta, self.param_dtype)).value
        eps = self.variable('constants', 'eps', lambda: jnp.asarray(self.eps, self.param_dtype)).value
        alpha_p, alpha_n, beta, eps = (value.astype(x.dtype) for value in (alpha_p, alpha_n, beta, eps))
        wide = at_least_fp32(x.dtype)
        alpha_p = jax.nn.softplus(alpha_p.astype(wide)).astype(x.dtype)
        alpha_n = beta + jax.nn.softplus(alpha_n.astype(wide)).astype(x.dtype)
        return jnp.where(x > 0, alpha_p * x * x + beta * x,
                         (jnp.expm1(jnp.minimum(x, eps)) - x) * alpha_n + beta * x)


def ungated_activation(name: str, x: jax.Array) -> jax.Array:
    """`x` through the `UNGATED` activation `name`, in `x`'s dtype; callers check `name` first."""
    return UNGATED[name](x)


def _exact_gelu(x: jax.Array) -> jax.Array:
    """`dew.nn.activations.gelu_exact`, rounded once to `x`'s dtype in both
    differentiation directions: a bf16 gate's activation rounds once, wherever
    the compiler places it. A protected span (`exact_gelu`,
    `dew.nn.derivatives.declared_tangent`): its barriers hold the roundings the
    algebra pass must not move."""
    return gelu_exact(jax.lax.optimization_barrier(x))


def _exact_gelu_jvp(primals: tuple[jax.Array], tangents: tuple[jax.Array]
                    ) -> tuple[jax.Array, jax.Array]:
    x, = primals
    dx, = tangents
    output = exact_gelu(x)
    work = x.astype(at_least_fp32(x.dtype))
    _, derivative = jax.jvp(gelu_exact, (work,), (jnp.ones_like(work),))
    tangent = derivative * jax.lax.optimization_barrier(dx).astype(work.dtype)
    return output, jax.lax.optimization_barrier(tangent.astype(x.dtype))


exact_gelu = contracted("test_an_exact_gelu_holds_every_order", "numerics")(
    declared_tangent(Protected(), _exact_gelu, _exact_gelu_jvp))


@dataclasses.dataclass(frozen=True)
class Situ:
    """Moonshot's SiTU gated product, `SituAndMul` (modeling_kimi_linear.py:64-82
    of moonshotai/Kimi-K3 at f831ab6): `beta * tanh(gate / beta) * sigmoid(gate)`
    times the up projection, which `linear_beta` soft-caps as `linear_beta *
    tanh(up / linear_beta)` when set. Both halves run in fp32 and the product
    returns in the gate's dtype, as the reference computes it. The released
    text config names the two `activation_situ_beta` and
    `activation_situ_linear_beta`. A gated MLP takes one of these as its
    activation in place of a name, since it transforms the up projection too.
    """
    beta: float = 1.0
    linear_beta: float | None = None

    def __call__(self, gate: jax.Array, up: jax.Array) -> jax.Array:
        wide = at_least_fp32(gate.dtype)
        work_gate, work_up = gate.astype(wide), up.astype(wide)
        activated = capped_logits(work_gate, self.beta) * jax.nn.sigmoid(work_gate)
        if self.linear_beta is not None:
            work_up = capped_logits(work_up, self.linear_beta)
        return (activated * work_up).astype(gate.dtype)


GatedActivation = str | Situ
"""A gated MLP's activation: 'swiglu', 'geglu' or 'geglu_exact' by name, or a `Situ`."""


def _gate_function(activation: str, implementation: str) -> Callable[[jax.Array], jax.Array]:
    """The gate's activation `activation` names, refusing an unknown name or implementation."""
    if implementation not in norm_swiglu.IMPLEMENTATIONS:
        raise ValueError(f"implementation must be one of {norm_swiglu.IMPLEMENTATIONS}, "
                         f"got {implementation!r}")
    gates = {"swiglu": silu, "geglu": gelu_tanh, "geglu_exact": exact_gelu}
    if activation not in gates:
        raise ValueError(f"mlp must be 'swiglu', 'geglu', 'geglu_exact' or a Situ, got {activation!r}")
    return gates[activation]


def _rounded(activate: Callable[[jax.Array], jax.Array]) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """`gated_product`'s rounded arithmetic over `activate`."""
    def rounded(gate: jax.Array, up: jax.Array) -> jax.Array:
        dtype = jnp.result_type(gate, up)
        gate_fp32 = rounded_to(gate.astype(jnp.float32), gate.dtype)
        up_fp32 = rounded_to(up.astype(jnp.float32), up.dtype)
        activated = rounded_to(activate(gate_fp32), gate.dtype)
        return rounded_to(activated * up_fp32, dtype).astype(dtype)

    return rounded


# Recomputed in the backward pass, as the norms are (`normalized_in_fp32`):
# a scanned stack otherwise stores five fp32 copies of the MLP width per
# token (TPU v6e, scanned 12-layer 512-wide decoder: 10.51 ms against 9.06
# and 1.33x the memory).
_recomputed = functools.partial(jax.checkpoint, policy=jax.checkpoint_policies.nothing_saveable)


def swiglu_clamped(gate: jax.Array, up: jax.Array, limit: float | None) -> tuple[jax.Array, jax.Array]:
    """A gated MLP's gate and up projections clamped before their product, the gate from above and
    the up projection both ways by `limit` (GLM-5.3-Flash's, modeling_glm5_next.py:98-104, and
    DeepSeek V4's experts'); unclamped where `limit` is None."""
    if limit is None:
        return gate, up
    return jnp.minimum(gate, limit), jnp.clip(up, -limit, limit)


def gated_product(activation: GatedActivation, *, implementation: str = 'auto',
                  axes=None) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """The product a gated MLP takes of its gate and up projections, rounded
    where torch rounds `act_fn(gate) * up`: the activation (silu 'swiglu', tanh
    gelu 'geglu', erf gelu 'geglu_exact') runs in fp32 and rounds to the gate's
    dtype, and the product rounds again. Every rounding is explicit
    (`rounded_to`), the inputs' included: left to XLA, a third of Qwen3-0.6B's
    bf16 products differed from transformers' by up to 2 ulp. fp32 and wider
    round nowhere. A `Situ` computes its own product. SwiGLU's
    `implementation` defaults to the generation's measured kernel; off a GPU,
    'pallas' too runs XLA's product. GeGLU and SiTU remain XLA products. `axes`,
    the operands' logical axes, place the kernel on a mesh."""
    if isinstance(activation, Situ):
        return activation
    portable = _portable(_gate_function(activation, implementation), lambda gate, up: (gate, up))
    return _dispatched(activation, portable,
                       lambda gate, up: norm_swiglu.swiglu_kernel(implementation, gate, up, axes=axes),
                       lambda gate, up, portable: norm_swiglu.swiglu(gate, up, portable, axes=axes))


def packed_gated_product(activation: str, *, implementation: str = 'auto',
                         axes=None) -> Callable[[jax.Array], jax.Array]:
    """`gated_product` of one packed projection's `[..., 2, H]` output, gate
    then up, read where the projection wrote them: SwiGLU's kernel reads and
    writes the packed layout (`norm_swiglu.packed_swiglu`), and XLA's product
    recomputes from the packed output itself, so neither copies a half out.
    `axes` are the product's logical axes, `[..., H]`'s."""
    portable = _portable(_gate_function(activation, implementation),
                         lambda gate_up: (gate_up[..., 0, :], gate_up[..., 1, :]))
    return _dispatched(activation, portable,
                       lambda gate_up: norm_swiglu.packed_swiglu_kernel(implementation, gate_up, axes=axes),
                       lambda gate_up, portable: norm_swiglu.packed_swiglu(gate_up, portable, axes=axes))


def _portable(activate, halves):
    """The XLA product of a gate and up projection that `halves(*operands)` reads from the operands
    (the two projections, or one packed output's two halves): `activate(gate) * up` at fp32 and
    wider, else rounded where torch rounds it (`_rounded`) and recomputed from the operands
    themselves in the backward, so a packed output is never copied into halves."""
    rounded = _rounded(activate)
    rounded_product = _recomputed(lambda *operands: rounded(*halves(*operands)))

    def portable(*operands: jax.Array) -> jax.Array:
        if jnp.finfo(jnp.result_type(*operands)).bits >= 32:
            gate, up = halves(*operands)
            return activate(gate) * up
        return rounded_product(*operands)

    return portable


def _dispatched(activation, portable, kernel_choice, kernel):
    """The product over the operands: SwiGLU's Pallas `kernel(*operands, portable)` where
    `kernel_choice(*operands)` chooses it (every order of its derivative is the portable product's;
    every level runs the kernels), else `portable`, which every other activation runs."""
    def product(*operands: jax.Array) -> jax.Array:
        chosen = 'xla' if activation != 'swiglu' else kernel_choice(*operands)
        return kernel(*operands, portable) if chosen == 'pallas' else portable(*operands)

    return product
