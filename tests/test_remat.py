"""Rematerialization must be invisible to everything except memory use.

Same parameter tree, same forward values, same gradients: a difference in
any of them would invalidate checkpoints or change what a run converges to.
"""

import contextlib
import io

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from jax.ad_checkpoint import checkpoint_name, print_saved_residuals

from dew.nn.attention import cudnn_runs
from dew.nn.dit import (
    FUSED_ATTENTION_FORWARD, TextContext, remat_block, saved_through_remat,
)
from dew.nn.backbones.dit import SimpleDiT
from dew.nn.backbones.mmdit import SimpleMMDiT, HierarchicalMMDiT
from dew.nn.backbones.uvit import SimpleUDiT
from dew.nn.backbones.ssm_dit import HybridSSMAttentionDiT
from dew.nn.backbones.video_dit import VideoDiT

RES = 32

BUILDERS = {
    'simple_dit': lambda remat: SimpleDiT(
        patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2, remat=remat),
    'simple_udit': lambda remat: SimpleUDiT(
        patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2, remat=remat),
    'simple_mmdit': lambda remat: SimpleMMDiT(
        patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2, remat=remat),
    'hierarchical_mmdit': lambda remat: HierarchicalMMDiT(
        base_patch_size=2, emb_features=(32, 64, 96), num_layers=(1, 1, 1),
        num_heads=(2, 2, 2), mlp_ratio=2, remat=remat),
    'hybrid_dit': lambda remat: HybridSSMAttentionDiT(
        patch_size=4, emb_features=64, num_layers=2, num_heads=2, mlp_ratio=2, remat=remat),
}


def image_inputs(rng):
    return (jax.random.normal(rng, (2, RES, RES, 3)), jnp.ones((2,)),
            TextContext(jnp.ones((2, 77, 768), jnp.float32), jnp.ones((2, 77), bool)))


def saved_residuals(f, *args):
    """The lines jax prints for the values a remat policy keeps in `f`."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        print_saved_residuals(f, *args)
    return printed.getvalue().splitlines()


def compare_forward_and_backward(plain, remat, params, *inputs):
    def loss(model, p):
        return jnp.sum(model.apply(p, *inputs) ** 2)

    assert jnp.allclose(loss(plain, params), loss(remat, params), rtol=1e-5, atol=1e-4)
    g_plain = jax.grad(lambda p: loss(plain, p))(params)
    g_remat = jax.grad(lambda p: loss(remat, p))(params)
    for a, b in zip(jax.tree.leaves(g_plain), jax.tree.leaves(g_remat), strict=True):
        assert jnp.allclose(a, b, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize('arch', sorted(BUILDERS))
def test_remat_keeps_parameter_tree_identical(rng, arch):
    x, temb, ctx = image_inputs(rng)
    plain = BUILDERS[arch](False).init(rng, x, temb, ctx)
    remat = BUILDERS[arch](True).init(rng, x, temb, ctx)

    def paths(tree):
        return [jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_leaves_with_path(tree)]

    assert paths(plain) == paths(remat), "remat changed the checkpoint layout"


@pytest.mark.parametrize('arch', sorted(BUILDERS))
def test_remat_preserves_outputs_and_gradients(rng, arch):
    x, temb, ctx = image_inputs(rng)
    plain, remat = BUILDERS[arch](False), BUILDERS[arch](True)
    compare_forward_and_backward(plain, remat, plain.init(rng, x, temb, ctx), x, temb, ctx)


@pytest.mark.skipif(jax.default_backend() != 'cpu',
                    reason='fp32 remat is bit-exact on the CPU backend, not across GPU fusions')
def test_remat_recompute_is_bit_exact(rng):
    """The recomputed forward runs the same ops on the same saved values, so
    on CPU fp32 nothing is merely close: a single differing bit would mean
    the policy dropped something the backward pass then rebuilt differently."""
    x, temb, ctx = image_inputs(rng)
    plain, remat = BUILDERS['simple_dit'](False), BUILDERS['simple_dit'](True)
    params = plain.init(rng, x, temb, ctx)

    def loss(model, p):
        return jnp.sum(model.apply(p, x, temb, ctx) ** 2)

    assert jnp.allclose(loss(plain, params), loss(remat, params), rtol=0, atol=0)
    g_plain = jax.grad(lambda p: loss(plain, p))(params)
    g_remat = jax.grad(lambda p: loss(remat, p))(params)
    for a, b in zip(jax.tree.leaves(g_plain), jax.tree.leaves(g_remat), strict=True):
        assert jnp.allclose(a, b, rtol=0, atol=0)


class BatchedDotBlock(nn.Module):
    """A batched einsum, the shape attention scores have, optionally carrying
    the name `scaled_dot_product_attention` puts on its output."""

    named: bool

    @nn.compact
    def __call__(self, x):
        scores = jnp.einsum('bhqd,bhkd->bhqk', x, x)
        return jnp.tanh(checkpoint_name(scores, 'attention_output') if self.named else scores)


@pytest.mark.parametrize('named', [True, False])
def test_remat_policy_saves_what_the_attention_name_marks(rng, named):
    """A batched dot is not a dot the policy saves on its own and it is no
    fused kernel either, so whether this value survives the backward pass is
    decided by the name alone."""
    x = jax.random.normal(rng, (2, 2, 8, 4))
    block = remat_block(BatchedDotBlock, True)(named=named)
    params = block.init(rng, x)
    lines = saved_residuals(lambda arr: jnp.sum(block.apply(params, arr) ** 2), x)

    inside = [line for line in lines if 'BatchedDotBlock' in line]
    assert bool(inside) == named
    assert all("f32[2,2,8,8] named 'attention_output'" in line for line in inside)


def test_remat_policy_keeps_attention_output_and_drops_the_scores(rng):
    """On the reference attention path the scores are a plain batched einsum,
    which is the [B, H, Q, K] residual remat exists to avoid; the kernel's
    [B, S, H, D] output is the cheap thing to keep, once per layer."""
    layers, heads, features = 2, 2, 64
    tokens, head_dim = (RES // 4) ** 2, features // 2
    model = SimpleDiT(patch_size=4, emb_features=features, num_layers=layers,
                      num_heads=heads, mlp_ratio=2, remat=True)
    x, temb, ctx = image_inputs(rng)
    params = model.init(rng, x, temb, ctx)
    lines = saved_residuals(lambda p: jnp.sum(model.apply(p, x, temb, ctx) ** 2), params)

    named = [line for line in lines if "named 'attention_output'" in line]
    assert len(named) == layers
    assert all(f'f32[2,{tokens},{heads},{head_dim}]' in line for line in named)
    assert not [line for line in lines if f'f32[2,{heads},{tokens},{tokens}]' in line]


def test_fused_attention_forward_is_named_as_the_policy_matches_it():
    """The policy recognises the fused forward by the primitive's name, so a
    rename in jax would silently cost a second flash forward per layer rather
    than fail. Read the names off jax's own primitives instead."""
    from jax._src.cudnn.fused_attention_stablehlo import (
        _dot_product_attention_fwd_p, _dot_product_attention_fwd_p_wrapper,
    )

    assert {str(_dot_product_attention_fwd_p),
            str(_dot_product_attention_fwd_p_wrapper)} == set(FUSED_ATTENTION_FORWARD)
    assert saved_through_remat(_dot_product_attention_fwd_p_wrapper)
    assert not saved_through_remat(jax.lax.tanh_p)


@pytest.mark.skipif(not cudnn_runs(jnp.zeros((1, 8, 2, 64), jnp.bfloat16)),
                    reason='needs the fused cudnn kernel, which this backend will not run')
def test_fused_attention_runs_once_per_layer_under_remat():
    """What the policy is for: the flash forward stays out of the backward
    pass, so a rematerialized step holds one fused call per layer, not two."""
    layers = 2
    model = SimpleDiT(patch_size=4, emb_features=256, num_layers=layers, num_heads=4,
                      mlp_ratio=4, remat=True, dtype=jnp.bfloat16, attention_impl='auto')
    x, temb = jnp.zeros((2, RES, RES, 3)), jnp.ones((2,))
    params = model.init(jax.random.PRNGKey(0), x, temb, None)
    step = jax.jit(jax.grad(
        lambda p: jnp.sum(model.apply(p, x, temb, None).astype(jnp.float32) ** 2)))
    text = step.lower(params).compile().as_text()

    assert text.count('custom_call_target="__cudnn$fmhaSoftmax"') == layers
    assert text.count('custom_call_target="__cudnn$fmhaSoftmaxBackward"') == layers


def test_video_dit_remat_matches():
    """Remat only rewrites the backward pass, so the gradients are where a
    dropped residual or a mismatched body would show, not the forward."""
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 3, 16, 16, 3))
    temb = jnp.ones((1,))
    ctx = TextContext(jnp.ones((1, 77, 768), jnp.float32), jnp.ones((1, 77), bool))
    plain = VideoDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1)
    remat = VideoDiT(patch_size=4, emb_features=32, num_layers=1, num_heads=2, mlp_ratio=1,
                     remat=True)
    compare_forward_and_backward(plain, remat, plain.init(rng, x, temb, ctx), x, temb, ctx)
