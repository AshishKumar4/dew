"""CPU first filters for the fla port's declared configs: each compiles for its GPU within the block's
shared memory, and its dots refuse mixed dtypes. A GPU's own tests are the verdict."""

import fla_gdn_support as support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from kernel_compile import compile_kernels

from dew.nn.kernels import fla_gdn, generation
from dew.nn.linear import chunk_gated_delta_rule, fla_refusal, l2norm

pytestmark = pytest.mark.kernel


def test_the_training_shapes_reach_the_ports_widest_heads():
    assert max(shape[side] for shape in support.TRAINING_SHAPES.values()
               for side in ("width", "columns")) == fla_gdn.WIDEST == 128


def test_each_triton_gpu_takes_the_table_its_block_holds():
    """An A100's configs on every generation whose block holds as much shared memory, the 99 KB
    tables on the rest, and the RTX PRO 6000's own bf16 picks, measured on a G4, on sm120."""
    assert fla_gdn.CONFIGS[("sm80", "bfloat16")] is fla_gdn._SM80_BF16
    assert fla_gdn.CONFIGS[("sm90", "bfloat16")] is fla_gdn._SM80_BF16
    for target in ("sm86", "sm89"):
        assert fla_gdn.CONFIGS[(target, "bfloat16")] is fla_gdn._SMALL_BF16
        assert fla_gdn.CONFIGS[(target, "float32")] is fla_gdn._SMALL_FP32
    assert fla_gdn.CONFIGS[("sm120", "bfloat16")] is fla_gdn._BLACKWELL_BF16
    assert fla_gdn.CONFIGS[("sm120", "float32")] is fla_gdn._SMALL_FP32
    assert ("sm75", "bfloat16") not in fla_gdn.CONFIGS


def test_a_config_takes_the_labs_overrides_and_clips_to_the_operands():
    shape = {"width": 64, "columns": 32, "chunk": 64}
    assert fla_gdn.kernel_config("chunk_fwd_o", shape, jnp.bfloat16, target="sm80") == {
        "bk": 64, "bv": 32, "warps": 8, "stages": 3}
    overridden = fla_gdn.kernel_config("chunk_bwd_dhu", shape, jnp.float32, target="sm120",
                                       stages=3, warps=None)
    assert overridden == {"bv": 32, "warps": 4, "stages": 3}
    assert fla_gdn.CONFIGS[("sm120", "float32")]["chunk_bwd_dhu"]["stages"] == 1


@pytest.mark.parametrize("target", ["sm80", "sm120"])
def test_the_kernel_cases_lower_each_kernel_on_abstract_training_operands(target):
    cases = list(support.kernel_cases(target))
    # Nine kernels at two dtypes and two shapes, plus the four reverse cumsums.
    assert len(cases) == 40 and len({case.name for case in cases}) == 40
    for case in cases:
        fn, args = case.call()
        assert fn.keywords["interpret"] is False
        assert args and all(isinstance(arg, jax.ShapeDtypeStruct) for arg in args)
        assert args[0].shape[0] in (4, 8) and args[0].shape[1] == 3072
        assert case.name.startswith(target + "/")


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("left,right", [(jnp.bfloat16, jnp.float32), (jnp.float32, jnp.bfloat16),
                                       (jnp.float16, jnp.bfloat16)])
def test_the_shared_dot_rejects_mismatches_before_interpret_widening(exact, left, right):
    a, b = jnp.ones((2, 2), left), jnp.ones((2, 2), right)
    with pytest.raises(TypeError, match="same dtype"):
        fla_gdn._dot(a, b, exact=exact)


def test_the_shared_dot_preserves_narrow_products_with_fp32_accumulation():
    a = jnp.asarray([[1.0, 0.5], [-0.25, 2.0]], jnp.bfloat16)
    got = fla_gdn._dot(a, a.T, exact=True)
    assert got.dtype == jnp.float32
    expected = a.astype(jnp.float32) @ a.T.astype(jnp.float32)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_the_fla_rule_refuses_keys_wider_than_its_configs_and_falls_back_cleanly(monkeypatch):
    monkeypatch.setattr(generation, "device_generation", lambda: "sm120")
    q = l2norm(jax.random.normal(jax.random.key(83), (1, 64, 1, 256)))
    k = l2norm(jax.random.normal(jax.random.key(89), q.shape))
    v = jax.random.normal(jax.random.key(97), (1, 64, 2, 128))
    g, beta = jnp.full(v.shape[:3], -0.1), jnp.full(v.shape[:3], 0.5)
    assert fla_gdn.rule_refusal(k, k, v, 64) == (
        "keys 256 and values 128 wide; the port's configs compile for at most 128")
    assert fla_refusal(k, k, v, 64) is not None
    got = chunk_gated_delta_rule(q, k, v, g, beta, implementation="fla")
    want = chunk_gated_delta_rule(q, k, v, g, beta, implementation="xla")
    for mine, theirs in zip(got, want, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


def test_the_fla_rule_refuses_a_key_head_serving_no_power_of_two_value_heads(monkeypatch):
    monkeypatch.setattr(generation, "device_generation", lambda: "sm120")
    k, v = jnp.zeros((1, 64, 1, 128), jnp.bfloat16), jnp.zeros((1, 64, 3, 128), jnp.bfloat16)
    assert fla_gdn.rule_refusal(k, k, v, 64) == "3 value heads a key head; dqkwg's block holds a power of two"


def test_a_rule_with_float32_queries_over_bf16_keys_runs_xla_not_the_port():
    q = l2norm(jax.random.normal(jax.random.key(101), (1, 64, 1, 64)))
    k = l2norm(jax.random.normal(jax.random.key(103), q.shape)).astype(jnp.bfloat16)
    v = jax.random.normal(jax.random.key(107), (1, 64, 2, 64)).astype(jnp.bfloat16)
    g, beta = jnp.full(v.shape[:3], -0.1), jnp.full(v.shape[:3], 0.5, jnp.bfloat16)
    got = chunk_gated_delta_rule(q, k, v, g, beta, implementation="fla")
    want = chunk_gated_delta_rule(q, k, v, g, beta, implementation="xla")
    for mine, theirs in zip(got, want, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


def test_the_fla_rule_refuses_a_gpu_without_its_configs(monkeypatch):
    monkeypatch.setattr(generation, "device_generation", lambda: "sm75")
    k, v = jnp.zeros((1, 64, 1, 128), jnp.bfloat16), jnp.zeros((1, 64, 2, 128), jnp.bfloat16)
    assert fla_gdn.rule_refusal(k, k, v, 64) == "the port has no bfloat16 configs for sm75"


def test_the_fla_rule_refuses_a_dtype_without_configs_where_it_interprets_too(monkeypatch):
    monkeypatch.setattr(generation, "device_generation", lambda: "cpu")
    with jax.enable_x64(new_val=True):
        k, v = jnp.zeros((1, 64, 1, 64), jnp.float64), jnp.zeros((1, 64, 2, 64), jnp.float64)
        assert fla_gdn.rule_refusal(k, k, v, 64) == "the port has no float64 configs for cpu"
        assert fla_refusal(k, k, v, 64) == "the port has no float64 configs for cpu"


@pytest.mark.parametrize("target", ["sm80", "sm89", "sm120"])
def test_every_declared_kernel_config_compiles_on_tritons_compiler_within_its_block(target):
    """Each declared config at each declared shape, lowered as on the GPU and compiled by pip's Triton
    for its capability on this CPU (tests/kernel_compile.py): it compiles, within the bound, to
    no more shared memory than its GPU's block holds. A kernel that does not compile (a two-result
    inline asm, a mixed-dtype dot) or compiles pathologically slowly (an IEEE fp32 dot on sm80) fails
    here and never reaches a GPU."""
    pytest.importorskip("triton")
    found = compile_kernels(support.kernel_cases(target), target)
    assert found, "the compile gate must not be vacuous"
    failures = [f"{record['name']}: {record['error']}" if "error" in record else
                f"{record['name']}: {record['shared']} shared bytes, limit {record['limit']}"
                for record in found if "error" in record or not record["shared"] <= record["limit"]]
    assert not failures, "\n".join(failures)
