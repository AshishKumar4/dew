"""CPU first filters for declared Pallas kernel configs: each compiles for its GPU within the block's
shared memory, and the dots the kernels share refuse mixed dtypes. A GPU's own tests are the verdict."""

import ast
import importlib
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dew.nn import kernels
from dew.nn.kernels import fla_gdn, generation
from dew.nn.linear import chunk_gated_delta_rule, fla_refusal, l2norm


def configured_modules():
    """Kernel modules declaring CONFIGS, found from source without importing the ones that do not."""
    for root in kernels.__path__:
        for path in sorted(Path(root).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            declared = any((isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == "CONFIGS"
                                    for target in node.targets))
                           or (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
                               and node.target.id == "CONFIGS") for node in tree.body)
            if declared:
                yield importlib.import_module(f"{kernels.__name__}.{path.stem}")


def test_the_declared_shapes_are_the_training_shapes():
    assert fla_gdn.CONFIG_SHAPES == {
        "0.8b": {"batch": 8, "length": 3072, "key_heads": 16, "value_heads": 16,
                 "width": 128, "columns": 128, "chunk": 64},
        "9b": {"batch": 4, "length": 3072, "key_heads": 16, "value_heads": 32,
               "width": 128, "columns": 128, "chunk": 64},
    }


def test_each_triton_gpu_takes_the_table_its_block_holds():
    """An A100's configs on every generation whose block holds as much shared memory, the 99 KB
    tables on the rest; fla's measured A100 choices on sm80, and Blackwell's two-warp forward h and
    two-warp, four-stage WY backward on sm120."""
    assert fla_gdn.CONFIGS[("sm80", "bfloat16")] is fla_gdn._SM80_BF16
    assert fla_gdn.CONFIGS[("sm90", "bfloat16")] is fla_gdn._SM80_BF16
    for target in ("sm86", "sm89"):
        assert fla_gdn.CONFIGS[(target, "bfloat16")] is fla_gdn._SMALL_BF16
        assert fla_gdn.CONFIGS[(target, "float32")] is fla_gdn._SMALL_FP32
    assert fla_gdn.CONFIGS[("sm120", "bfloat16")] == {**fla_gdn._SMALL_BF16, **fla_gdn._BLACKWELL}
    assert ("sm75", "bfloat16") not in fla_gdn.CONFIGS
    assert fla_gdn.CONFIGS[("sm120", "bfloat16")]["chunk_fwd_h"]["warps"] == 2
    for dtype in ("bfloat16", "float32"):
        wy = fla_gdn.CONFIGS[("sm120", dtype)]["prepare_wy_repr_bwd"]
        assert (wy["warps"], wy["stages"]) == (2, 4)


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
    cases = list(fla_gdn.kernel_cases(target))
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
        generation.strict_dot(a, b, exact=exact)
    with pytest.raises(TypeError, match="same dtype"):
        fla_gdn._dot(a, b, exact=exact)


def test_the_shared_dot_preserves_narrow_products_with_fp32_accumulation():
    a = jnp.asarray([[1.0, 0.5], [-0.25, 2.0]], jnp.bfloat16)
    got = generation.strict_dot(a, a.T, exact=True)
    assert got.dtype == jnp.float32
    expected = a.astype(jnp.float32) @ a.T.astype(jnp.float32)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_the_fla_rule_refuses_keys_wider_than_its_configs_and_falls_back_cleanly(monkeypatch):
    monkeypatch.setattr(generation, "device_generation", lambda: "sm120")
    q = l2norm(jax.random.normal(jax.random.key(83), (1, 64, 1, 256)))
    k = l2norm(jax.random.normal(jax.random.key(89), q.shape))
    v = jax.random.normal(jax.random.key(97), (1, 64, 2, 128))
    g, beta = jnp.full(v.shape[:3], -0.1), jnp.full(v.shape[:3], 0.5)
    assert fla_gdn.rule_refusal(k, v, 64) == (
        "keys 256 and values 128 wide; the port's configs compile for at most 128")
    assert fla_refusal(k, v, 64) is not None
    got = chunk_gated_delta_rule(q, k, v, g, beta, implementation="fla")
    want = chunk_gated_delta_rule(q, k, v, g, beta, implementation="xla")
    for mine, theirs in zip(got, want, strict=True):
        np.testing.assert_array_equal(np.asarray(mine), np.asarray(theirs))


def test_the_fla_rule_refuses_a_key_head_serving_no_power_of_two_value_heads(monkeypatch):
    monkeypatch.setattr(generation, "device_generation", lambda: "sm120")
    k, v = jnp.zeros((1, 64, 1, 128), jnp.bfloat16), jnp.zeros((1, 64, 3, 128), jnp.bfloat16)
    assert fla_gdn.rule_refusal(k, v, 64) == "3 value heads a key head; dqkwg's block holds a power of two"


def test_the_fla_rule_refuses_a_gpu_without_its_configs(monkeypatch):
    monkeypatch.setattr(generation, "device_generation", lambda: "sm75")
    k, v = jnp.zeros((1, 64, 1, 128), jnp.bfloat16), jnp.zeros((1, 64, 2, 128), jnp.bfloat16)
    assert fla_gdn.rule_refusal(k, v, 64) == "the port has no configs for sm75"


@pytest.mark.parametrize("target", ["sm80", "sm89", "sm120"])
def test_every_declared_kernel_config_compiles_on_tritons_compiler_within_its_block(target):
    """Each declared config at each declared shape, lowered as on the GPU and compiled by pip's Triton
    for its capability on this CPU (`generation.compile_kernels`): it compiles, within the bound, to
    no more shared memory than its GPU's block holds. A kernel that does not compile (a two-result
    inline asm, a mixed-dtype dot) or compiles pathologically slowly (an IEEE fp32 dot on sm80) fails
    here and never reaches a GPU."""
    pytest.importorskip("triton", reason="the test extra pins Triton on Linux x86_64")
    found = [record for module in configured_modules()
             for record in generation.compile_kernels(module, [target])]
    assert found, "the compile gate must not be vacuous"
    failures = [f"{record['name']}: {record['error']}" if "error" in record else
                f"{record['name']}: {record['shared']} shared bytes, limit {record['limit']}"
                for record in found if "error" in record or not record["shared"] <= record["limit"]]
    assert not failures, "\n".join(failures)
