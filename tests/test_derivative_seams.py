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

CUSTOM = frozenset({"custom_jvp", "custom_vjp"})
"""JAX's custom derivatives, as a decorator, a call, or within functools.partial."""

CONTRACTED = {
    "src/dew/interop/torchax_fallback.py::_accumulated":
        "test_a_broadcast_summing_its_gradient_wide_holds_every_order",
    "src/dew/nn/kernels/generation.py::linear_tangent.produced":
        "test_an_embedding_lookup_holds_every_order",
    "src/dew/nn/kernels/generation.py::linear_tangent.pulled":
        "test_an_embedding_lookup_holds_every_order",
    "src/dew/nn/kernels/generation.py::linear_tangent.bounded":
        "test_an_embedding_lookup_holds_every_order",
    "src/dew/nn/blocks.py::_rows.linear_tangent":
        "test_an_embedding_lookup_holds_every_order",
    "src/dew/nn/precision.py::bf16_dot_general.linear_tangent":
        "test_a_bf16_dot_holds_every_order",
    "src/dew/nn/precision.py::accumulated.linear_tangent":
        "test_a_wide_cpu_sum_of_products_holds_every_order",
    "src/dew/nn/kernels/templates/gemm/template.py::gemm.linear_tangent":
        "test_a_grouped_gemm_holds_every_order",
    "src/dew/nn/kernels/recurrence_chunk.py::_portable_rule.linear_tangent":
        "test_the_chunked_delta_rule_holds_every_order",
    "src/dew/nn/kernels/recurrence_algebra.py::decays":
        "test_the_chunked_delta_rule_holds_every_order",
    "src/dew/nn/kernels/recurrence_algebra.py::remembered":
        "test_the_chunked_delta_rule_holds_every_order",
    "src/dew/nn/kernels/recurrence_algebra.py::_decayed_product":
        "test_the_chunked_delta_rule_holds_every_order",
    "src/dew/nn/activations.py::exact_gelu":
        "test_an_exact_gelu_holds_every_order",
    "src/dew/diffusion/schedules/flow.py::time_shift":
        "test_a_time_shift_holds_every_order",
    "src/dew/nn/categorical.py::_probabilities":
        "test_the_categorical_losses_hold_every_order",
    "src/dew/nn/categorical.py::_label_loss":
        "test_the_categorical_losses_hold_every_order",
    "src/dew/nn/categorical.py::_soft_loss":
        "test_the_categorical_losses_hold_every_order",
    "src/dew/nn/categorical.py::_divergence":
        "test_the_categorical_losses_hold_every_order",
    "src/dew/nn/precision.py::correctly_rounded_divide":
        "test_the_precision_seams_hold_every_order",
    "src/dew/nn/precision.py::_compensated_sum":
        "test_the_precision_seams_hold_every_order",
    "src/dew/nn/precision.py::rounded_operand":
        "test_the_precision_seams_hold_every_order",
    "src/dew/nn/attention.py::forward_differentiable.attend":
        "test_a_fused_attention_call_in_forward_mode_holds_every_order",
    "src/dew/nn/kernels/splash.py::_attention":
        "test_a_fused_attention_kernel_holds_its_reverse_orders",
    "src/dew/nn/kernels/templates/attention_route.py::_attend":
        "test_a_fused_attention_kernel_holds_its_reverse_orders",
    "src/dew/nn/kv_cache.py::gpu_paged.linear_tangent":
        "test_the_paged_decode_kernel_holds_every_order",
    "src/dew/nn/kernels/fla_gdn.py::chunk_gated_delta_rule":
        "test_the_delta_rule_ports_hold_their_orders",
    "src/dew/nn/kernels/fla_kda.py::chunk_kda":
        "test_the_delta_rule_ports_hold_their_orders",
    "src/dew/nn/kernels/mamba_ssd.py::ssd_scan":
        "test_the_ssd_port_holds_its_orders",
    "src/dew/nn/kernels/ssd.py::ssd_chunk_scan":
        "test_the_tpu_ssd_scan_holds_its_orders",
    "src/dew/nn/kernels/recurrence.py::linear_tangent":
        "test_the_decode_step_holds_every_order",
    "src/dew/nn/kernels/recurrence.py::step.linear_tangent":
        "test_the_decode_step_holds_every_order",
    "src/dew/nn/kernels/recurrence_chunk.py::_per_product.linear_tangent":
        "test_the_chunked_op_holds_every_order",
    "src/dew/nn/kernels/recurrence_chunk.py::mosaic_rule.rule":
        "test_the_template_rule_holds_its_orders",
    "src/dew/nn/kernels/short_conv.py::conv.linear_tangent":
        "test_the_short_conv_holds_every_order",
    "src/dew/objectives/base.py::_ruled":
        "test_a_ruled_loss_holds_every_order",
    "src/dew/objectives/lm/chunked.py::_bounded_head":
        "test_the_lm_heads_hold_every_order_of_their_gradient",
    "src/dew/objectives/lm/chunked.py::_whole_head":
        "test_the_lm_heads_hold_every_order_of_their_gradient",
    "src/dew/objectives/lm/chunked.py::_fused":
        "test_the_lm_heads_hold_every_order_of_their_gradient",
    "src/dew/objectives/lm/chunked.py::_whole_losses":
        "test_the_whole_logits_losses_hold_every_order",
    "src/dew/objectives/residuals.py::_pulled":
        "test_the_residual_pullbacks_hold_every_order",
    "src/dew/objectives/residuals.py::_stepped":
        "test_the_residual_pullbacks_hold_every_order",
    "src/dew/nn/kernels/norm_swiglu.py::stream_rms_norm.linear_tangent":
        "test_the_fused_norms_and_products_hold_every_order",
    "src/dew/nn/kernels/norm_swiglu.py::rms_norm.linear_tangent":
        "test_the_fused_norms_and_products_hold_every_order",
    "src/dew/nn/kernels/norm_swiglu.py::gated_rms_norm.linear_tangent":
        "test_the_fused_norms_and_products_hold_every_order",
    "src/dew/nn/kernels/norm_swiglu.py::swiglu.linear_tangent":
        "test_the_fused_norms_and_products_hold_every_order",
    "src/dew/nn/kernels/norm_swiglu.py::packed_swiglu.linear_tangent":
        "test_the_fused_norms_and_products_hold_every_order",
    "src/dew/objectives/perturbation.py::NodePerturbation.loss.estimated":
        "test_a_node_perturbation_estimate_holds_its_reverse_orders",
    "src/dew/nn/kernels/grouped_matmul.py::portable_projection":
        "test_a_portable_grouped_projection_holds_every_order",
    "src/dew/nn/kernels/templates/gemm/template.py::_prologued.prologued":
        "test_a_gemm_prologue_holds_every_order",
    "src/dew/nn/kernels/templates/layout.py::_custom.held":
        "test_a_transposed_hooks_custom_derivative_holds_every_order_it_takes",
    "src/dew/nn/moe.py::_stored_primal":
        "test_a_stored_expert_primal_holds_every_order",
    "src/dew/nn/precision.py::at_default_precision.run":
        "test_a_default_precision_call_holds_every_order_it_takes",
    "src/dew/quantization.py::frozen_fp8_product":
        "test_a_frozen_fp8_product_holds_every_order",
    "src/dew/quantization.py::_grouped_convolution_gradient.grouped_conv_qt":
        "test_a_quantized_convolution_holds_every_order_it_takes",
    "src/dew/sampling/flow.py::_directional": "test_a_kls_forward_mode_holds_its_orders",
    "src/dew/sampling/flow.py::_half_mahalanobis": "test_a_transitions_quadratic_term_holds_every_order",
    "src/dew/sampling/flow.py::linear_tangent": "test_a_transitions_kl_holds_every_order",
    "src/dew/sampling/guidance.py::_guided": "test_a_guided_prediction_holds_every_order",
}
"""Each contracted seam, `path::qualified name`, and the test of tests/test_contracts_*.py that holds it."""


def _named(node) -> str | None:
    return getattr(node, "attr", None) or getattr(node, "id", None)


def _custom(node) -> bool:
    """Whether `node`, a decorator or a call's function, makes a custom derivative (`CUSTOM`)."""
    if isinstance(node, ast.Call):
        if _named(node.func) == "partial" and node.args:
            return _custom(node.args[0])
        return _custom(node.func)
    return _named(node) in CUSTOM


def seams_of(relative: str, text: str) -> list[str]:
    """`relative`'s seams in `text`, each `relative::qualified name`: a function decorated a custom
    derivative, a name assigned one, and each enclosing function's call of linear_tangent
    (`<scope>.linear_tangent`)."""
    found = []

    def walk(node, scope: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if not isinstance(child, ast.ClassDef) and any(_custom(d) for d in child.decorator_list):
                    found.append(".".join([*scope, child.name]))
                walk(child, [*scope, child.name])
                continue
            if (isinstance(child, ast.Assign) and isinstance(child.value, ast.Call)
                    and _custom(child.value.func) and isinstance(child.targets[0], ast.Name)):
                found.append(".".join([*scope, child.targets[0].id]))
            if isinstance(child, ast.Call) and _named(child.func) == "linear_tangent":
                found.append(".".join([*scope, "linear_tangent"]))
            walk(child, scope)

    walk(ast.parse(text), [])
    return [f"{relative}::{name}" for name in dict.fromkeys(found)]


def seams() -> list[str]:
    """Every seam of src/dew (`seams_of`)."""
    root = SOURCE.parents[1]
    return [seam for path in sorted(SOURCE.rglob("*.py"))
            for seam in seams_of(str(path.relative_to(root)), path.read_text())]


def test_every_custom_derivative_is_under_a_contract():
    """Each seam is contracted (`CONTRACTED`, its test one of tests/test_contracts_*.py's), and each named
    seam exists, so the list never outlives its code."""
    found = set(seams())
    assert found - set(CONTRACTED) == set(), "a seam with no contract"
    assert set(CONTRACTED) - found == set(), "a named seam no longer in src/dew"
    contracts = [importlib.import_module(path.stem)
                 for path in Path(__file__).parent.glob("test_contracts_*.py")]
    assert all(any(callable(getattr(module, test, None)) for module in contracts)
               for test in CONTRACTED.values())


def test_the_scan_finds_each_form_of_a_custom_derivative():
    """`seams_of` on each form a seam takes: a decorator, one within functools.partial, an assignment of a
    call, a nested function's, and a linear_tangent call within an expression."""
    text = (
        "import functools\nimport jax\n"
        "@jax.custom_jvp\ndef a(x): return x\n"
        "@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))\ndef b(x, n): return x\n"
        "c = jax.custom_jvp(lambda x: x)\n"
        "def outer(x):\n    @jax.custom_vjp\n    def inner(x): return x\n    return inner(x)\n"
        "def tangent(x):\n    return linear_tangent(f, g, h)(x)\n"
        "def plain(x): return x\n")
    assert seams_of("m.py", text) == ["m.py::a", "m.py::b", "m.py::c", "m.py::outer.inner",
                                      "m.py::tangent.linear_tangent"]


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
