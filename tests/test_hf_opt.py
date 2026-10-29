"""OPT's learned-position offset and pre-norm block against transformers."""

from pathlib import Path

import numpy as np
import pytest
from interop_support import fixture_ids, token_draw, transformers_arms, transformers_model
from reference_error import assert_as_exact, seeds_for

from dew.interop import Pretrained
from dew.sampling import Sampling, generate

DIRECTORY = Path(__file__).parent / 'fixtures' / 'hf' / 'opt-tiny'

OPT_SEEDS = 8
"""The logits' declared draws of ids against transformers' (check 1)."""


def test_opt_logits_follow_transformers_and_its_greedy_continuation():
    """OPT's logits, its learned positions read two places on (OPT's offset), are on par with
    transformers' from its float64 twin over drawn ids (check 1), and a cached greedy continuation draws
    the tokens transformers drew."""
    loaded = Pretrained.load(DIRECTORY, dtype='float32', attention_impl='reference')
    reference, truth = transformers_arms(transformers_model(DIRECTORY),
                                         lambda model, ids: model(ids, use_cache=False).logits)
    assert_as_exact(lambda ids: loaded.model.apply(loaded.variables, ids), reference, fixture_ids(DIRECTORY),
                    'OPT logits', seeds=OPT_SEEDS, truth_fn=truth)
    ids = np.load(DIRECTORY / 'input_ids.npy')
    generated = generate(loaded.model, loaded.variables, ids[:1, :4], 6, key=0,
                         sampling=Sampling(temperature=0))
    np.testing.assert_array_equal(np.asarray(generated.tokens), np.load(DIRECTORY / 'generated.npy'))


@pytest.mark.network
def test_released_opt125m_logits_and_cached_greedy_generation(tmp_path):
    from tools.classic_gpt_reference import check_checkpoint

    check_checkpoint('facebook/opt-125m', tmp_path / 'parity.json',
                     revision='27dcfa74d334bc871f3234de431e71c6eeba5dd6')


@pytest.mark.network
def test_released_opt350m_logits_and_cached_greedy_generation(tmp_path):
    """OPT-350m as released, a 1024-wide post-LN stream over a 512-wide embedding projected in and out
    of it, 24 layers and no final norm: its logits and a cached greedy continuation against
    transformers' (`check_checkpoint`)."""
    from tools.classic_gpt_reference import check_checkpoint

    check_checkpoint('facebook/opt-350m', tmp_path / 'parity.json',
                     revision='08ab08cc4b72ff5593870b5d527cf4230323703c')


@pytest.mark.parametrize("variant", [
    {"do_layer_norm_before": False, "word_embed_proj_dim": 32},
    {"do_layer_norm_before": True, "_remove_final_layer_norm": True}],
    ids=["opt-350m post-LN with projected embeddings", "pre-norm without a final norm"])
def test_an_opt_350m_shaped_decoder_computes_transformers_logits(variant, tmp_path):
    """OPT-350m's decoder (post-LN blocks, no final norm, a 32-wide embedding projected in and out of a
    64-wide stream by bias-free projections), and a pre-norm one that drops its final norm, each drawn by
    transformers' own init: Dew loads them and computes transformers' logits on par from its float64
    twin (check 1), its export reloads in transformers to the same logits, bit for bit, and a cached
    greedy continuation, the prompt's ids projected in on the decode path too, draws the tokens
    transformers' generate draws."""
    import torch
    from transformers import OPTConfig, OPTForCausalLM

    config = OPTConfig(vocab_size=96, hidden_size=64, num_hidden_layers=2, ffn_dim=128,
                       num_attention_heads=4, max_position_embeddings=32, **variant)
    torch.manual_seed(0)
    OPTForCausalLM(config).eval().save_pretrained(tmp_path / "source")
    loaded = Pretrained.load(tmp_path / "source", dtype="float32", attention_impl="reference")
    reference, truth = transformers_arms(transformers_model(tmp_path / "source"),
                                         lambda model, ids: model(ids, use_cache=False).logits)
    draw = token_draw(0x350, 96, (2, 9))
    seeds = seeds_for(reference, draw, "OPT-350m logits", truth_fn=truth)
    assert_as_exact(lambda ids: loaded.model.apply(loaded.variables, ids), reference, draw,
                    "OPT-350m logits", seeds=seeds, truth_fn=truth)
    loaded.save(tmp_path / "export")
    # Both read by the same class on the same attention, so the export's weights and config alone differ.
    exported, source = (transformers_model(tmp_path / name) for name in ("export", "source"))
    ids = torch.as_tensor(np.asarray(draw(0)[0]), dtype=torch.long)
    with torch.no_grad():
        np.testing.assert_array_equal(exported(ids, use_cache=False).logits.numpy(),
                                      source(ids, use_cache=False).logits.numpy())
        drawn = source.generate(ids[:1, :4], max_new_tokens=6, min_new_tokens=6, do_sample=False)
    generated = generate(loaded.model, loaded.variables, np.asarray(ids[:1, :4]), 6, key=0,
                         sampling=Sampling(temperature=0))
    # Both hold the prompt and its continuation.
    np.testing.assert_array_equal(np.asarray(generated.tokens), drawn.numpy())


def test_an_opt_350m_shaped_decoder_reads_its_head_in_the_streams_coordinates_and_keeps_loop_norms():
    """Past an embedding's own width the head's vocabulary rows are given in the stream's coordinates,
    through `project_out`: the final states times those rows are the logits, bit for bit in float64 on
    integer-valued weights. A decoder that ends in no final norm still norms its loop's passes, so its
    passes differ from a loop that norms none."""
    import jax
    import jax.numpy as jnp

    from dew.nn.backbones.causal_transformer import CausalTransformer, Loop

    with jax.enable_x64(new_val=True):
        model = CausalTransformer(vocab_size=16, emb_features=8, num_layers=1, num_heads=2, mlp_features=16,
                                  max_seq_len=8, embedding_features=4, post_residual_norms=True,
                                  pre_norms=False, final_norm=False, dtype=jnp.float64,
                                  attention_impl="reference")
        ids = jnp.arange(6, dtype=jnp.int32).reshape(1, 6) % 16
        variables = model.init(jax.random.key(0), ids)
        # Integer-valued weights and final states: both products are exact in float64, whatever their
        # order.
        whole = jax.tree.map(lambda leaf: jnp.round(leaf * 4), variables)
        hidden = jnp.asarray(np.random.default_rng(0).integers(-3, 4, (1, 6, 8)), jnp.float64)
        rows = model.apply(whole, method=CausalTransformer.vocabulary_rows)
        logits = model.apply(whole, hidden, method=CausalTransformer.logits_from_hidden)
        np.testing.assert_array_equal(np.asarray(logits), np.asarray(hidden @ rows.T))
    looped = {"num_layers": 2, "loop": Loop(2, layers=(0, 2), step_norm=True), "final_norm": False}
    base = CausalTransformer(vocab_size=16, emb_features=8, num_heads=2, mlp_features=16, max_seq_len=8,
                             attention_impl="reference")
    normed = base.clone(**looped)
    plain = base.clone(**{**looped, "loop": Loop(2, layers=(0, 2), step_norm=False)})
    params = normed.init(jax.random.key(1), ids)
    assert not np.array_equal(np.asarray(normed.apply(params, ids)), np.asarray(plain.apply(params, ids)))


PROJECTED_TWIN_SEEDS = 16
"""The draws a projected decoder's depth and context are held to its stream-width twin over (check 1)."""


def test_a_projected_decoders_prediction_depth_and_table_context_read_the_stream_embedding():
    """A decoder whose embedding has a width of its own (`embedding_features`), with a prediction depth and
    a cross-attention layer reading a context by the table, pairs its depth with and encodes its context
    by the token embedding at the stream's width (`stream_embeddings`, through `project_in`): its trunk's
    and its depth's logits over a context, and their loss's gradient in its layers', depth's and
    cross-attention's parameters, are on par with those of its twin at the stream's width, whose table is
    the projected one and whose untied head is `project_out` and the table together (check 1, the twin
    in float64 the truth)."""
    import jax
    import jax.numpy as jnp
    import optax
    from reference_error import as_given

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.cross_attention import CrossAttention, encode_context

    shape = {"vocab_size": 61, "emb_features": 16, "num_layers": 2, "num_heads": 2, "mlp_features": 32,
             "max_seq_len": 8, "attention_impl": "reference", "num_nextn_predict_layers": 1,
             "cross_attention": CrossAttention((1,), 4)}
    projected = CausalTransformer(**shape, embedding_features=8)
    twin = CausalTransformer(**shape, tie_embeddings=False)
    stream = ("embed_tokens", "project_in", "project_out")

    def twinned(params):
        table = params["embed_tokens"]["embedding"]
        held = {name: value for name, value in params.items() if name not in stream}
        return {**held, "embed_tokens": {"embedding": table @ params["project_in"]["kernel"]},
                "lm_head": {"kernel": params["project_out"]["kernel"] @ table.T}}

    def draw(seed):
        rng = np.random.default_rng([2409, seed])
        ids = rng.integers(0, 61, (2, 6)).astype(np.int32)
        params = projected.init(jax.random.key(seed), ids)["params"]
        # The gates drawn away from zero, so the context reaches the states.
        params = jax.tree_util.tree_map_with_path(
            lambda path, leaf: (rng.uniform(-1, 1, leaf.shape).astype(np.float32)
                                if path[-1].key.startswith("alpha_") else leaf), params)
        return params, ids, rng.integers(0, 61, (2, 4)).astype(np.int32), rng.integers(0, 61, (2, 6))

    def scored(model, convert):
        def run(params, ids, context, targets):
            whole = convert(params)
            shared = {name: value for name, value in whole.items() if name not in (*stream, "lm_head")}

            def outputs(shared):
                variables = {"params": {**whole, **shared}}
                conditioning = encode_context(model, variables, context, jnp.full((2,), 4, jnp.int32),
                                              encoder="table")
                hidden, prediction = model.apply(variables, ids, conditioning=conditioning,
                                                 method="hidden_and_mtp_inputs")
                (depth,) = model.apply(variables, prediction, ids, conditioning=conditioning,
                                       method="mtp_logits")
                return model.apply(variables, hidden, method="logits_from_hidden"), depth

            def loss(shared):
                logits, depth = outputs(shared)
                return jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits, targets)
                               + optax.softmax_cross_entropy_with_integer_labels(depth, targets))
            return outputs(shared), jax.grad(loss)(shared)

        def held(*inputs):
            with jax.default_matmul_precision("highest"):
                return run(*inputs)
        return held

    @as_given
    def truth(params, ids, context, targets):
        with jax.enable_x64(new_val=True):
            wide = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), params)
            return jax.device_get(scored(twin.clone(dtype=jnp.float64), twinned)(wide, ids, context, targets))

    assert_as_exact(jax.jit(scored(projected, lambda params: params)), jax.jit(scored(twin, twinned)), draw,
                    "projected depth and context", seeds=PROJECTED_TWIN_SEEDS, truth_fn=truth)
