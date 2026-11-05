"""Every custom derivative of Dew's under a contract (ledger 2641): each seam in src/dew (a custom_jvp or
custom_vjp, and each call of generation.linear_tangent, which builds three) named in `CONTRACTED` with the
test of tests/test_contracts_*.py that holds it to its contract (`assert_contract`: every order
finite wherever its float64 truth is a normal float32, at every exponent edge, beside an independent plain
formula). A new seam fails here until a contract names it. The harness's own validation is here too."""

import ast
import importlib
from pathlib import Path

import jax.numpy as jnp
import pytest
from contract_support import LessFinite, assert_contract

SOURCE = Path(__file__).resolve().parents[1] / "src" / "dew"

CUSTOM = frozenset({"custom_jvp", "custom_vjp", "surrogate", "declared_derivative", "declared_tangent"})
"""JAX's custom derivatives and Dew's declaring functions (`dew.nn.derivatives`: a surrogate, a derivative and
a tangent of a declared kind), as a decorator, a call, or within functools.partial."""

KERNEL, NUMERICS, SURROGATE = "kernel", "numerics", "surrogate"
"""Each seam's kind (`CONTRACTED`)."""

CONTRACTED = {
    "src/dew/nn/kernels/generation.py::linear_tangent.produced":
        ("test_an_embedding_lookup_holds_every_order", KERNEL),
    "src/dew/nn/kernels/generation.py::linear_tangent.pulled":
        ("test_an_embedding_lookup_holds_every_order", KERNEL),
    "src/dew/nn/kernels/generation.py::linear_tangent.bounded":
        ("test_an_embedding_lookup_holds_every_order", KERNEL),
    "src/dew/nn/blocks.py::_rows.linear_tangent":
        ("test_an_embedding_lookup_holds_every_order", NUMERICS),
    "src/dew/nn/precision.py::bf16_dot_general.linear_tangent":
        ("test_a_bf16_dot_holds_every_order", NUMERICS),
    "src/dew/nn/precision.py::_summed_wide.linear_tangent":
        ("test_a_wide_cpu_sum_of_products_holds_every_order", NUMERICS),
    "src/dew/nn/kernels/templates/gemm/template.py::gemm.linear_tangent":
        ("test_a_grouped_gemm_holds_every_order", KERNEL),
    "src/dew/nn/kernels/templates/recurrence/selective.py::_scan":
        ("test_the_selective_scan_holds_its_orders", KERNEL),
    "src/dew/algebra/scan.py::_decays_rule.linear_tangent":
        ("test_the_derived_delta_rule_chunks_hold_their_orders", NUMERICS),
    "src/dew/algebra/scan.py::_lowered_rule.linear_tangent":
        ("test_the_derived_delta_rule_chunks_hold_their_orders", NUMERICS),
    "src/dew/nn/kernels/recurrence_chunk.py::_portable_rule.linear_tangent":
        ("test_the_chunked_delta_rule_holds_every_order", NUMERICS),
    "src/dew/nn/kernels/recurrence_algebra.py::decays":
        ("test_the_chunked_delta_rule_holds_every_order", NUMERICS),
    "src/dew/nn/kernels/recurrence_algebra.py::remembered":
        ("test_the_chunked_delta_rule_holds_every_order", NUMERICS),
    "src/dew/nn/kernels/recurrence_algebra.py::_decayed_product":
        ("test_the_chunked_delta_rule_holds_every_order", NUMERICS),
    "src/dew/nn/activations.py::exact_gelu":
        ("test_an_exact_gelu_holds_every_order", NUMERICS),
    "src/dew/diffusion/schedules/flow.py::time_shift":
        ("test_a_time_shift_holds_every_order", NUMERICS),
    "src/dew/nn/inspection.py::tapped":
        ("test_a_probes_tap_holds_its_orders", SURROGATE),
    "src/dew/nn/categorical.py::_probabilities":
        ("test_the_categorical_losses_hold_every_order", NUMERICS),
    "src/dew/nn/categorical.py::label_terms":
        ("test_the_label_terms_hold_every_order", NUMERICS),
    "src/dew/nn/categorical.py::_soft_loss":
        ("test_the_categorical_losses_hold_every_order", NUMERICS),
    "src/dew/nn/categorical.py::_divergence":
        ("test_the_categorical_losses_hold_every_order", NUMERICS),
    "src/dew/nn/precision.py::correctly_rounded_divide":
        ("test_the_precision_seams_hold_every_order", NUMERICS),
    "src/dew/nn/precision.py::_compensated_sum":
        ("test_the_precision_seams_hold_every_order", NUMERICS),
    "src/dew/nn/precision.py::rounded_operand":
        ("test_the_precision_seams_hold_every_order", SURROGATE),
    "src/dew/nn/attention.py::forward_differentiable.attend":
        ("test_a_fused_attention_call_in_forward_mode_holds_every_order", KERNEL),
    "src/dew/nn/kernels/splash.py::_attention":
        ("test_a_fused_attention_kernel_holds_its_reverse_orders", KERNEL),
    "src/dew/nn/kernels/templates/attention_route.py::_attend":
        ("test_a_fused_attention_kernel_holds_its_reverse_orders", KERNEL),
    "src/dew/nn/kv_cache.py::gpu_paged.linear_tangent":
        ("test_the_paged_decode_kernel_holds_every_order", KERNEL),
    "src/dew/nn/kv_cache.py::template_paged.linear_tangent":
        ("test_the_paged_decode_kernel_holds_every_order", KERNEL),
    "src/dew/nn/kernels/fla_gdn.py::chunk_gated_delta_rule":
        ("test_the_delta_rule_ports_hold_their_orders", KERNEL),
    "src/dew/nn/kernels/fla_kda.py::chunk_kda":
        ("test_the_delta_rule_ports_hold_their_orders", KERNEL),
    "src/dew/nn/kernels/mamba_ssd.py::ssd_scan":
        ("test_the_ssd_port_holds_its_orders", KERNEL),
    "src/dew/nn/kernels/ssd.py::ssd_chunk_scan":
        ("test_the_tpu_ssd_scan_holds_its_orders", KERNEL),
    "src/dew/nn/kernels/recurrence.py::linear_tangent":
        ("test_the_decode_step_holds_every_order", NUMERICS),
    "src/dew/nn/kernels/recurrence.py::step.linear_tangent":
        ("test_the_decode_step_holds_every_order", KERNEL),
    "src/dew/nn/kernels/recurrence_chunk.py::_per_product.linear_tangent":
        ("test_the_chunked_op_holds_every_order", KERNEL),
    "src/dew/nn/kernels/recurrence_chunk.py::mosaic_rule.rule":
        ("test_the_template_rule_holds_its_orders", KERNEL),
    "src/dew/nn/kernels/short_conv.py::conv.linear_tangent":
        ("test_the_short_conv_holds_every_order", KERNEL),
    "src/dew/objectives/base.py::_ruled":
        ("test_a_ruled_loss_holds_every_order", SURROGATE),
    "src/dew/objectives/lm/chunked.py::_bounded_head.linear_tangent":
        ("test_the_lm_heads_hold_every_order_of_their_gradient", KERNEL),
    "src/dew/objectives/lm/chunked.py::_whole_head.linear_tangent":
        ("test_the_lm_heads_hold_every_order_of_their_gradient", NUMERICS),
    "src/dew/objectives/lm/chunked.py::_fused.linear_tangent":
        ("test_the_lm_heads_hold_every_order_of_their_gradient", KERNEL),
    "src/dew/nn/kernels/norm_swiglu.py::stream_rms_norm.linear_tangent":
        ("test_the_fused_norms_and_products_hold_every_order", KERNEL),
    "src/dew/nn/kernels/norm_swiglu.py::rms_norm.linear_tangent":
        ("test_the_fused_norms_and_products_hold_every_order", KERNEL),
    "src/dew/nn/ngpt.py::stepped.linear_tangent":
        ("test_ngpts_join_and_normed_hidden_hold_every_order", KERNEL),
    "src/dew/nn/ngpt.py::normed_hidden.linear_tangent":
        ("test_ngpts_join_and_normed_hidden_hold_every_order", KERNEL),
    "src/dew/nn/kernels/norm_swiglu.py::gated_rms_norm.linear_tangent":
        ("test_the_fused_norms_and_products_hold_every_order", KERNEL),
    "src/dew/nn/kernels/norm_swiglu.py::swiglu.linear_tangent":
        ("test_the_fused_norms_and_products_hold_every_order", KERNEL),
    "src/dew/nn/kernels/norm_swiglu.py::packed_swiglu.linear_tangent":
        ("test_the_fused_norms_and_products_hold_every_order", KERNEL),
    "src/dew/objectives/perturbation.py::NodePerturbation.loss.estimated":
        ("test_a_node_perturbation_estimate_holds_every_order", SURROGATE),
    "src/dew/nn/kernels/grouped_matmul.py::portable_projection":
        ("test_a_portable_grouped_projection_holds_every_order", NUMERICS),
    "src/dew/nn/kernels/templates/gemm/template.py::_prologued.prologued":
        ("test_a_gemm_prologue_holds_every_order", KERNEL),
    "src/dew/nn/kernels/templates/layout.py::_custom.held":
        ("test_a_transposed_hooks_custom_derivative_holds_every_order_it_takes", KERNEL),
    "src/dew/nn/moe.py::_stored_primal":
        ("test_a_stored_expert_primal_holds_every_order", NUMERICS),
    "src/dew/nn/precision.py::at_default_precision.run":
        ("test_a_default_precision_call_holds_every_order_it_takes", NUMERICS),
    "src/dew/quantization.py::frozen_fp8_product":
        ("test_a_frozen_fp8_product_holds_every_order", SURROGATE),
    "src/dew/quantization.py::_grouped_convolution_gradient.quantized_training":
        ("test_a_quantized_convolution_holds_every_order_it_takes", SURROGATE),
    "src/dew/sampling/flow.py::_gaussian_kl":
        ("test_a_transitions_kl_holds_every_order", NUMERICS),
    "src/dew/sampling/flow.py::_half_mahalanobis":
        ("test_a_transitions_quadratic_term_holds_every_order", NUMERICS),
    "src/dew/sampling/guidance.py::_guided":
        ("test_a_guided_prediction_holds_every_order", NUMERICS),
    "src/dew/nn/derivatives.py::surrogate.rule":
        ("test_a_surrogate_holds_every_order", SURROGATE),
    "src/dew/nn/derivatives.py::declared_derivative.rule":
        ("test_a_declared_derivative_holds_every_order", KERNEL),
    "src/dew/nn/derivatives.py::declared_tangent.rule":
        ("test_a_declared_derivative_holds_every_order", NUMERICS),
}
"""Each contracted seam, `path::qualified name`, the test of tests/test_contracts_*.py that holds it, and its
kind: a kernel's derivative (`KERNEL`: a launch's own backward, or the tangent of the code it stands for),
a derivative of the value written for its rounding or range (`NUMERICS`: time_shift's regular form, a
guided prediction's affine derivative among them), or a declared derivative that is not the value's
(`SURROGATE`, `dew.nn.derivatives.surrogate`)."""


def _named(node) -> str | None:
    return getattr(node, "attr", None) or getattr(node, "id", None)


DECLARING = frozenset({"declared_derivative", "declared_tangent"})
"""Dew's declaring functions that take the kind of what they declare first (`dew.nn.derivatives`)."""


def _maker(node) -> str | None:
    """The custom derivative `node`, a decorator or a call, makes (`CUSTOM`), or None: a derivative or
    tangent declared of the kind `Surrogate()` a surrogate, as `surrogate` declares one."""
    if not isinstance(node, ast.Call):
        name = _named(node)
        return name if name in CUSTOM else None
    partial = _named(node.func) == "partial" and node.args
    maker = _maker(node.args[0]) if partial else _maker(node.func)
    kind = node.args[1:2] if partial else node.args[:1]
    if (maker in DECLARING and kind and isinstance(kind[0], ast.Call)
            and _named(kind[0].func) == "Surrogate"):
        return "surrogate"
    return maker


def seams_of(relative: str, text: str) -> dict[str, str]:
    """`relative`'s seams in `text`, each `relative::qualified name`, and what makes each: a function
    decorated a custom derivative, a name assigned one (`CUSTOM`'s maker), and each enclosing function's
    call of linear_tangent (`<scope>.linear_tangent`)."""
    found: dict[str, str] = {}

    def walk(node, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                makers = [] if isinstance(child, ast.ClassDef) else [
                    maker for maker in map(_maker, child.decorator_list) if maker]
                if makers:
                    found.setdefault(".".join([*scope, child.name]), makers[0])
                walk(child, [*scope, child.name])
                continue
            target = (child.targets[0] if isinstance(child, ast.Assign)
                      else child.target if isinstance(child, ast.AnnAssign) else None)
            if (isinstance(target, ast.Name) and isinstance(child.value, ast.Call)
                    and (maker := _maker(child.value))):
                found.setdefault(".".join([*scope, target.id]), maker)
            if isinstance(child, ast.Call) and _named(child.func) == "linear_tangent":
                found.setdefault(".".join([*scope, "linear_tangent"]), "linear_tangent")
            walk(child, scope)

    walk(ast.parse(text), [])
    return {f"{relative}::{name}": maker for name, maker in found.items()}


def seams() -> dict[str, str]:
    """Every seam of src/dew and what makes it (`seams_of`)."""
    root = SOURCE.parents[1]
    return {seam: maker for path in sorted(SOURCE.rglob("*.py"))
            for seam, maker in seams_of(str(path.relative_to(root)), path.read_text()).items()}


def test_every_custom_derivative_is_under_a_contract():
    """Each seam is contracted (`CONTRACTED`, its test one of tests/test_contracts_*.py's), and each named
    seam exists, so the list never outlives its code."""
    found = set(seams())
    assert found - set(CONTRACTED) == set(), "a seam with no contract"
    assert set(CONTRACTED) - found == set(), "a named seam no longer in src/dew"
    contracts = [importlib.import_module(path.stem)
                 for path in Path(__file__).parent.glob("test_contracts_*.py")]
    assert all(any(callable(getattr(module, test, None)) for module in contracts)
               for test, _ in CONTRACTED.values())


def test_each_seams_kind_is_its_declaration():
    """A seam is a surrogate exactly where it is declared one (`surrogate`, or a derivative or tangent
    declared of the kind `Surrogate()`), every other a kernel's or a numerics derivative: a derivative
    that is not its value's cannot hide in a custom rule of its own, and the safely written true
    derivatives (time_shift, a guided prediction, a held hook's, a gemm prologue's) are not surrogates.
    The declaration's own custom rule is the surrogate kind's."""
    kinds = {seam: kind for seam, (_, kind) in CONTRACTED.items()}
    assert set(kinds.values()) <= {KERNEL, NUMERICS, SURROGATE}
    marked = {seam for seam, kind in kinds.items() if kind == SURROGATE}
    declared = {seam for seam, maker in seams().items() if maker == "surrogate"}
    assert marked == declared | {"src/dew/nn/derivatives.py::surrogate.rule"}


def test_the_scan_finds_each_form_of_a_custom_derivative():
    """`seams_of` on each form a seam takes, and its maker: a decorator, one within functools.partial, an
    assignment of a call, an annotated one, a nested function's, a linear_tangent call within an
    expression, a surrogate's declaration, assigned or decorating, and a declared derivative or tangent,
    a surrogate where its kind is `Surrogate()`."""
    text = (
        "import functools\nimport jax\n"
        "@jax.custom_jvp\ndef a(x): return x\n"
        "@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))\ndef b(x, n): return x\n"
        "c = jax.custom_jvp(lambda x: x)\n"
        "d: Callable = jax.custom_vjp(lambda x: x)\n"
        "def outer(x):\n    @jax.custom_vjp\n    def inner(x): return x\n    return inner(x)\n"
        "def tangent(x):\n    return linear_tangent(f, g, h)(x)\n"
        "e = surrogate(lambda x: x, lambda primals, tangents, value: tangents[0])\n"
        "@functools.partial(surrogate, derivative=lambda primals, tangents, value: tangents[0])\n"
        "def f(x): return x\n"
        "g = declared_derivative(Surrogate(), lambda x: x, fwd, bwd)\n"
        "h = declared_tangent(Protected(), lambda x: x, jvp)\n"
        "@functools.partial(declared_derivative, Surrogate(), forward=fwd, backward=bwd)\n"
        "def k(x): return x\n"
        "def plain(x): return x\n")
    assert seams_of("m.py", text) == {
        "m.py::a": "custom_jvp", "m.py::b": "custom_vjp", "m.py::c": "custom_jvp", "m.py::d": "custom_vjp",
        "m.py::outer.inner": "custom_vjp", "m.py::tangent.linear_tangent": "linear_tangent",
        "m.py::e": "surrogate", "m.py::f": "surrogate", "m.py::g": "surrogate",
        "m.py::h": "declared_tangent", "m.py::k": "surrogate"}


def test_the_contract_refuses_a_formula_less_finite_than_the_plain_one():
    """The rule's validation: a formula overflowing where the plain one does not (|x| as the root of its
    square, past 2**64) is refused, at its primal and its derivatives; the plain formula against itself
    holds, at the same edge, where float32's own arithmetic holds every value."""
    x = (jnp.asarray([2.0**70, -3.0, 2.0**-70], jnp.float32),)
    direction = (jnp.ones((3,), jnp.float32),)
    cotangent = jnp.ones((3,), jnp.float32)
    assert_contract(jnp.abs, x, direction, cotangent, "the plain formula", plain_fn=jnp.abs)
    with pytest.raises(LessFinite):
        assert_contract(lambda x: jnp.sqrt(x * x), x, direction, cotangent, "a squared root",
                        plain_fn=jnp.abs)
