"""Every custom derivative of Dew's under a contract (ledger 2641): each seam in src/dew (a custom_jvp or
custom_vjp, and each call of generation.linear_tangent, which builds three) named in `CONTRACTED` with the
test of tests/test_derivative_contracts.py that holds it to its contract (`assert_contract`: every order
finite wherever its float64 truth is a normal float32, at every exponent edge, beside an independent plain
formula), or in `PENDING`, the seams row 2641 has yet to contract, which only shrinks. A new seam fails here
until a contract or that row names it."""

import ast
from pathlib import Path

import test_derivative_contracts

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
}
"""Each contracted seam, `path::qualified name`, and the test of tests/test_derivative_contracts.py that holds
it."""

PENDING = frozenset({
    "src/dew/nn/attention.py::forward_differentiable.attend",
    "src/dew/nn/kernels/fla_gdn.py::chunk_gated_delta_rule",
    "src/dew/nn/kernels/fla_kda.py::chunk_kda",
    "src/dew/nn/kernels/grouped_matmul.py::portable_projection",
    "src/dew/nn/kernels/mamba_ssd.py::ssd_scan",
    "src/dew/nn/kernels/norm_swiglu.py::stream_rms_norm.linear_tangent",
    "src/dew/nn/kernels/norm_swiglu.py::rms_norm.linear_tangent",
    "src/dew/nn/kernels/norm_swiglu.py::gated_rms_norm.linear_tangent",
    "src/dew/nn/kernels/norm_swiglu.py::swiglu.linear_tangent",
    "src/dew/nn/kernels/norm_swiglu.py::packed_swiglu.linear_tangent",
    "src/dew/nn/kernels/recurrence.py::linear_tangent",
    "src/dew/nn/kernels/recurrence.py::step.linear_tangent",
    "src/dew/nn/kernels/recurrence_chunk.py::_per_product.linear_tangent",
    "src/dew/nn/kernels/recurrence_chunk.py::mosaic_rule.rule",
    "src/dew/nn/kernels/short_conv.py::conv.linear_tangent",
    "src/dew/nn/kernels/splash.py::_attention",
    "src/dew/nn/kernels/ssd.py::ssd_chunk_scan",
    "src/dew/nn/kernels/templates/attention_route.py::_attend",
    "src/dew/nn/kernels/templates/gemm/template.py::_prologued.prologued",
    "src/dew/nn/kernels/templates/layout.py::_custom.held",
    "src/dew/nn/kv_cache.py::gpu_paged.linear_tangent",
    "src/dew/nn/moe.py::_stored_primal",
    "src/dew/nn/precision.py::at_default_precision.run",
    "src/dew/objectives/base.py::_ruled",
    "src/dew/objectives/lm/chunked.py::_bounded_head",
    "src/dew/objectives/lm/chunked.py::_whole_head",
    "src/dew/objectives/lm/chunked.py::_fused",
    "src/dew/objectives/lm/chunked.py::_whole_losses",
    "src/dew/objectives/perturbation.py::NodePerturbation.loss.estimated",
    "src/dew/objectives/residuals.py::_pulled",
    "src/dew/objectives/residuals.py::_stepped",
    "src/dew/quantization.py::frozen_fp8_product",
    "src/dew/quantization.py::_grouped_convolution_gradient.grouped_conv_qt",
    "src/dew/sampling/flow.py::_directional",
    "src/dew/sampling/flow.py::_half_mahalanobis",
    "src/dew/sampling/flow.py::linear_tangent",
    "src/dew/sampling/guidance.py::_guided",
})
"""The seams row 2641 has yet to contract: each leaves this set for `CONTRACTED` with its contract, and no
seam joins it."""


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


def test_every_custom_derivative_is_under_a_contract_or_pending_its_row():
    """Each seam is contracted (`CONTRACTED`, its test one of tests/test_derivative_contracts.py's) or pending
    row 2641 (`PENDING`), never both; and each named seam exists, so neither list outlives its code."""
    found = set(seams())
    assert not set(CONTRACTED) & PENDING
    assert found - set(CONTRACTED) - PENDING == set(), "a seam with no contract and no row"
    assert (set(CONTRACTED) | PENDING) - found == set(), "a named seam no longer in src/dew"
    assert all(callable(getattr(test_derivative_contracts, test, None)) for test in CONTRACTED.values())


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
