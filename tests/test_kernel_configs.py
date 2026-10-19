"""CPU first filters for the kernels: each of the fla port's declared configs, and the delta rule's and
the short conv's kernels at their training shapes, compiles for its GPU within the block's shared
memory, and the port's dots refuse mixed dtypes. None runs a kernel, so the GPU tier, whose kernel tests
are the verdict, runs none of them: the file carries no `kernel` mark."""

import fla_gdn_support as support
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import tokamax
from fla_gdn_support import CONV_SHAPES, conv_config_cases, conv_training_cases, kda_training_cases
from kernel_compile import compile_kernels, failures

from dew.nn.kernels import fla_gdn, fla_kda, generation, short_conv, tuning

LAUNCHES = {launch for module in (fla_gdn, fla_kda) for launch in vars(module).values()
            if isinstance(launch, fla_gdn._Launch)}
"""Every launch of the two ports."""


def test_the_compiled_shapes_reach_the_widest_keys_and_values_the_port_takes():
    shapes = support.COMPILED_SHAPES.values()
    assert max(shape["width"] for shape in shapes) == fla_gdn.WIDEST_KEYS == 256
    assert max(shape["columns"] for shape in shapes) == fla_gdn.WIDEST == 128


@pytest.mark.parametrize("target, wide", [("sm80", True), ("sm90", True), ("sm86", False), ("sm89", False),
                                          ("sm120", False), ("sm100", False), ("cpu", True)])
def test_each_gpu_takes_the_picks_its_block_holds_and_measures_a_space_holding_them(target, wide):
    """An A100's picks on every GPU whose block holds as much shared memory, the 99 KB ones elsewhere,
    on a GPU the targets do not list (sm100) too, and an A100's where a CPU interprets; every kernel's
    16-bit space holds its unmeasured pick, Kimi delta attention's own kernels' too, and fp32 has a pick
    for every kernel."""
    with generation.assumed(target):
        assert fla_gdn._wide() is wide
    for launch in LAUNCHES:
        assert launch.PICKS["bfloat16", wide] in launch.SPACES[wide], launch.NAME
        assert {"bfloat16", "float32"} <= {dtype for dtype, _ in launch.PICKS}, launch.NAME


@pytest.mark.parametrize("module", [fla_gdn, fla_kda])
def test_the_tuner_finds_each_of_the_rules_launches_in_its_lowering(module):
    """The rule's forward and backward, traced as the tuner reads a program (`tuning.launches`): each
    of its kernels is a launch tokamax describes, its arguments serializable into its key; the gated
    delta rule's nine, and Kimi delta attention's seven with the cumsum and memory kernels it shares."""
    per_key = module is fla_kda
    q = jax.ShapeDtypeStruct((1, 64, 1, 64), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((1, 64, 2, 64), jnp.bfloat16)
    g = jax.ShapeDtypeStruct((1, 64, 2, *(64,) * per_key), jnp.float32)
    h0 = jax.ShapeDtypeStruct((1, 2, 64, 64), jnp.float32)
    port = fla_kda.chunk_kda if per_key else fla_gdn.chunk_gated_delta_rule
    shared = {fla_gdn.ChunkLocalCumsum, fla_gdn.ChunkFwdH, fla_gdn.ChunkBwdDhu} if per_key else set()

    def rule(*args):
        return port(*args, None, 0.125, 64, interpret=True)

    def both(q, k, v, g, beta, h0, do):
        (_, final), pull = jax.vjp(rule, q, k, v, g, beta, h0)
        return pull((do, jnp.zeros_like(final)))

    with tuning.launches():
        beta = g.update(shape=g.shape[:3], dtype=jnp.bfloat16)
        found = tokamax.autotuning.get_bound_args(jax.jit(both), q, q, v, g, beta, h0, v)
    assert {type(ba.op) for ba in found} == shared | {type(op) for op in vars(module).values()
                                                      if isinstance(op, fla_gdn._Launch)}


def test_a_launch_clips_its_blocks_to_the_operands():
    """Blocks wider than a call's keys or values shrink to them."""
    q = jax.ShapeDtypeStruct((1, 64, 1, 64), jnp.bfloat16)
    v = jax.ShapeDtypeStruct((1, 64, 1, 32), jnp.bfloat16)
    h = jax.ShapeDtypeStruct((1, 1, 64, 32), jnp.bfloat16)
    h0 = jax.ShapeDtypeStruct((1, 1, 64, 32), jnp.float32)
    g = jax.ShapeDtypeStruct((1, 64, 1), jnp.float32)
    spans = fla_gdn.sequences(1, 64)
    ba = fla_gdn._CHUNK_FWD_O.bind(q, q, v, h, h0, g, *spans, scale=1.0, chunk=64, interpret=True)
    with generation.assumed("sm80"):
        assert fla_gdn._CHUNK_FWD_O._get_heuristics_config(ba) == fla_gdn.Tiles(64, 32, 8, 3)


@pytest.mark.parametrize("target", ["sm80", "sm120"])
def test_the_kernel_cases_lower_each_kernel_on_abstract_operands(target):
    cases = list(support.kernel_cases(target))
    # Nine kernels at two dtypes and two shapes and at bf16's widest keys, plus the five reverse cumsums.
    assert len(cases) == 50 and len({case.name for case in cases}) == 50
    for case in cases:
        fn, args = case.call()
        assert fn.keywords["interpret"] is False
        assert args and all(isinstance(arg, jax.ShapeDtypeStruct) for arg in args)
        assert (args[0].shape[0], args[0].shape[1]) in ((8, 3072), (4, 3072), (1, 512))
        assert case.name.startswith(target + "/")


@pytest.mark.parametrize("exact", [False, True])
@pytest.mark.parametrize("left,right", [(jnp.bfloat16, jnp.float32), (jnp.float32, jnp.bfloat16),
                                       (jnp.float16, jnp.bfloat16)])
def test_the_shared_dot_rejects_mismatches_before_interpret_widening(exact, left, right):
    a, b = jnp.ones((2, 2), left), jnp.ones((2, 2), right)
    with pytest.raises(TypeError):
        fla_gdn._dot(a, b, exact=exact)


def test_the_shared_dot_preserves_narrow_products_with_fp32_accumulation():
    a = jnp.asarray([[1.0, 0.5], [-0.25, 2.0]], jnp.bfloat16)
    got = fla_gdn._dot(a, a.T, exact=True)
    assert got.dtype == jnp.float32
    expected = a.astype(jnp.float32) @ a.T.astype(jnp.float32)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


@pytest.mark.parametrize("target", ["sm80", "sm89", "sm120"])
def test_every_declared_kernel_config_compiles_on_tritons_compiler_within_its_block(target):
    """Each declared config at each declared shape, lowered as on the GPU and compiled by pip's Triton
    for its capability on this CPU (tests/kernel_compile.py): it compiles, to no more shared memory
    than its GPU's block holds. A kernel that does not compile (a two-result inline asm, a mixed-dtype
    dot) fails here and never reaches a GPU."""
    pytest.importorskip("triton")
    found = compile_kernels(support.kernel_cases(target), target)
    assert found, "the compile gate must not be vacuous"
    assert not failures(found), "\n".join(failures(found))


@pytest.mark.parametrize("target", ["sm80", "sm89", "sm120"])
def test_kimi_delta_attentions_kernels_compile_on_tritons_compiler_within_each_gpus_block(target):
    """Every kernel of the port's forward and backward at the training shape, lowered as on the GPU and
    compiled by pip's Triton for its capability on this CPU (tests/kernel_compile.py), within the
    compile bound and the block's shared memory."""
    pytest.importorskip("triton")
    found = compile_kernels(kda_training_cases(), target)
    assert len(found) == 2
    assert not failures(found), "\n".join(failures(found))


@pytest.mark.parametrize("target", ["sm80", "sm89", "sm120"])
def test_the_short_convs_kernels_compile_on_tritons_compiler_within_each_gpus_block(target):
    """Each of the short conv's training cases lowered as on the GPU and compiled by pip's Triton for its
    capability on this CPU (tests/kernel_compile.py), within the compile bound and the block's shared
    memory."""
    pytest.importorskip("triton")
    found = compile_kernels(conv_training_cases(), target)
    assert len(found) == 2 * len(CONV_SHAPES) + 2
    assert not failures(found), "\n".join(failures(found))


@pytest.mark.parametrize("target", ["sm80", "sm89", "sm120"])
def test_every_short_conv_config_compiles_on_tritons_compiler_within_each_gpus_block(target):
    """Each config the tuner may give either launch of the short conv (`short_conv.TILES`,
    `short_conv.WALKS`), pinned at the 9B's packed delta conv, compiled for each GPU within the compile
    bound and the block's shared memory."""
    pytest.importorskip("triton")
    found = compile_kernels(conv_config_cases(), target)
    assert len(found) == len(short_conv.TILES) + len(short_conv.WALKS)
    assert not failures(found), "\n".join(failures(found))
