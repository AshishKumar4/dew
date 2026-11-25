"""Looped decoders that compute only what they need (theme M70): a loop's exits stop its passes, a gate
on the block's output runs the coda once, rows exit whole and leave the batch a pass computes over, a
cached call keeps each row's exiting pass's cache, and a block's providers publish across the loop's parts
(Gemma 4's shared keys and values)."""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.extend import core as jex_core
from model_support import attending, decode_steps, model_draw, with_logit_cotangent
from reference_error import assert_as_exact, with_gradients

from dew.interop.hf_decoders import translate_config
from dew.nn.backbones.call_metadata import call_metadata
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.backbones.embeddings import prepared_embeddings

FIXTURES = Path(__file__).with_name("fixtures") / "hf"

SEEDS = {"rows": 8, "first pass": 8, "last pass": 8, "sharing": 8, "sharing backward": 8,
         "sharing decode": 8}
"""Check 1's declared seeds for each site of the file."""


def _decoder(name: str) -> CausalTransformer:
    """A tiny release fixture as Dew builds it (`translate_config`), in float32."""
    config = json.loads((FIXTURES / name / "config.json").read_text())
    return CausalTransformer(**{**translate_config(config), "dtype": "float32"})


def _rows_exiting(**loop) -> CausalTransformer:
    """Qwen 3.5's tiny decoder looping its block [1, 3) three times before its coda, its gate on the
    block's output exiting each row whole at its threshold, the loop's other fields as `loop` states."""
    looped = _decoder("qwen35-tiny").with_passes(3, (1, 3), exit_gate=True, exit_threshold=0.6)
    return looped.clone(loop=dataclasses.replace(looped.loop, exit_at="block", exit_by="row", **loop))


def _gated(draw, width: int, salt: int, bias: float | None = None):
    """`draw`'s parameters with an exit gate drawn off its zeros (rows exiting at different passes), or its
    kernel zero and its bias `bias` (every row's hazard alike: certain to exit, or never)."""
    def drawn(seed):
        variables, *rest = draw(seed)
        gate = np.random.default_rng([salt, seed]).normal(0, 1.0, (width + 1,))
        if bias is not None:
            gate = np.concatenate([np.zeros(width), [bias]])
        params = {**variables["params"], "early_exit_gate": {
            "kernel": jnp.asarray(gate[:-1, None], jnp.float32), "bias": jnp.asarray(gate[-1:], jnp.float32)}}
        return {**variables, "params": params}, *rest
    return drawn


def _inputs(module, ids) -> dict:
    return {"train": False, "decode": False, "per_layer_input": None,
            "attention_metadata": call_metadata(module, ids, *(False, *(None,) * 9))}


def _rows_by_hand(module, ids):
    """`module`'s logits with its block's passes unrolled from the stack's own spans, each row's block
    output its exiting pass's: the first whose cumulative exit probability, each pass's hazard
    `sigmoid(gate(mean over the row of norm(h)))` times the chance no pass before it exited, reaches the
    threshold, the last pass's where none does; the coda run once, on the chosen outputs."""
    loop, inputs = module.loop, _inputs(module, ids)
    first, end = loop.layers
    x = prepared_embeddings(module.scaled_embeddings(module.token_embeddings(ids)), None)
    h = module.stack(module.stack(x, span=(0, first), **inputs), span=(first, end), **inputs)
    leaving, kept = [], 1.0
    for _ in range(loop.steps - 1):
        pooled = jnp.mean(module.norm(h), axis=1, keepdims=True)
        hazard = jax.nn.sigmoid(module.early_exit_gate(pooled)[..., 0])
        leaving.append((h, 1 - kept * (1 - hazard) >= loop.exit_threshold))
        kept = kept * (1 - hazard)
        h = module.stack(h, span=(first, end), **inputs)
    chosen = h
    for state, reached in reversed(leaving):
        chosen = jnp.where(reached[..., None], state, chosen)
    out = module.stack(chosen, span=(end, module.num_layers), **inputs)
    return module.logits_from_hidden(module.norm(out))


@attending("reference")
def test_a_row_exits_whole_at_its_threshold_and_its_coda_runs_once():
    """A gate on the block's output exiting each row whole (`Loop.exit_at='block'`, `exit_by='row'`):
    every row's logits read its exiting pass's block output through one coda, on par with the passes
    unrolled here from the stack's own spans (`_rows_by_hand`, check 1 against it in float64), rows of a
    batch exiting at different passes and those still looping run apart (row 12927)."""
    looped = _rows_exiting()
    draw = _gated(model_draw(looped, (8, 12), looped.vocab_size), looped.emb_features, 0xE91)

    def unrolled(model):
        return jax.jit(lambda variables, ids: model.apply(variables, ids, method=_rows_by_hand))

    assert_as_exact(jax.jit(looped.apply), unrolled(looped), draw, "rows exiting at the block",
                    seeds=SEEDS["rows"], truth_fn=unrolled(looped.clone(dtype=jnp.float64)))


def _conds(closed) -> list:
    """Every conditional of `closed`'s jaxpr, those of the jaxprs it holds among them."""
    found = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "cond":
                found.append(eqn)
            for value in eqn.params.values():
                for each in value if isinstance(value, (tuple, list)) else (value,):
                    if isinstance(each, jex_core.ClosedJaxpr):
                        walk(each.jaxpr)
                    elif isinstance(each, jex_core.Jaxpr):
                        walk(each)
    walk(closed.jaxpr)
    return found


def _rows_of_products(jaxpr) -> set[int]:
    """The leading sizes of every dot product's operands in `jaxpr`, those it holds among them."""
    sizes = set()
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "dot_general":
            sizes |= {var.aval.shape[0] for var in eqn.invars if var.aval.shape}
        for value in eqn.params.values():
            for each in value if isinstance(value, (tuple, list)) else (value,):
                if isinstance(each, jex_core.ClosedJaxpr):
                    sizes |= _rows_of_products(each.jaxpr)
                elif isinstance(each, jex_core.Jaxpr):
                    sizes |= _rows_of_products(each)
    return sizes


def test_a_loop_runs_no_pass_after_every_token_exits():
    """Scoring under a threshold stops its passes once every token has exited (`_Passes.stopping`, row
    12924): with a gate certain to exit at the first pass, more of its layers' outputs are what a pass that
    did not run leaves (zero, `_sown_switch`) than with a gate that never exits, which runs every pass."""
    looped = _decoder("qwen35-tiny").with_passes(3, (1, 3), exit_gate=True, exit_threshold=0.6)

    def zeros(bias: float) -> int:
        variables, ids = _gated(model_draw(looped, (2, 12), looped.vocab_size), looped.emb_features, 0xE92,
                                bias=bias)(0)
        _, captured = looped.apply(variables, ids, capture_intermediates=True, mutable=["intermediates"])
        return sum(not np.any(np.asarray(leaf)) for leaf in jax.tree.leaves(captured))
    assert zeros(30.0) > zeros(-30.0)


def test_a_pass_runs_over_the_rows_still_looping_alone():
    """Scoring whose rows exit whole (`exit_by='row'`) runs each later pass over the batch's halvings, the
    smallest that holds its rows still looping (`compacted`, row 12927): eight rows' pass a program for
    eight, four, two and one, each branch's products over its own rows."""
    looped = _rows_exiting()
    variables, ids = _gated(model_draw(looped, (8, 12), looped.vocab_size), looped.emb_features, 0xE95)(0)
    buckets = [eqn for eqn in _conds(jax.make_jaxpr(looped.apply)(variables, ids))
               if len(eqn.params["branches"]) == 4]
    # One a later pass, each branch's products over its bucket's rows (or its rows' tokens, flattened).
    assert len(buckets) == looped.loop.steps - 1
    for eqn in buckets:
        for size, branch in zip((8, 4, 2, 1), eqn.params["branches"], strict=True):
            rows = _rows_of_products(branch.jaxpr)
            assert size in rows or size * ids.shape[1] in rows


def _decoded(model, prompt: int):
    """`model`'s logits over `ids`, its first `prompt` tokens prefilled into an allocated cache and each
    later one decoded alone."""
    made, step = decode_steps(model)

    def run(variables, ids):
        cache = made(variables, ids.shape[0])
        logits, cache = step(variables, cache, ids[:, :prompt])
        each = [logits]
        for index in range(prompt, ids.shape[1]):
            logits, cache = step(variables, cache, ids[:, index:index + 1])
            each.append(logits)
        return jnp.concatenate(each, axis=1)
    return run


@attending("reference")
@pytest.mark.parametrize(("bias", "site"), [(30.0, "first pass"), (-30.0, "last pass")])
def test_a_shared_cache_decodes_each_row_at_its_exiting_pass(bias, site):
    """The loop decoding under a cache its passes share (`Loop.shared_cache`), its gate on the block's
    output and its rows exiting whole, a row's cache its exiting pass's writes (row 12928): a gate certain
    to exit decodes what the decoder looping once does, and one that never exits what the loop running
    every pass without a threshold does (check 1, each against its own in float64)."""
    looped = _rows_exiting(shared_cache=True)
    reference = (looped.clone(loop=dataclasses.replace(looped.loop, steps=1)) if bias > 0
                 else looped.clone(loop=dataclasses.replace(looped.loop, exit_threshold=None)))
    draw = _gated(model_draw(looped, (2, 10), looped.vocab_size), looped.emb_features, 0xE93, bias=bias)
    assert_as_exact(_decoded(looped, 6), _decoded(reference, 6), draw, f"decoding exiting at the {site}",
                    seeds=SEEDS[site], truth_fn=_decoded(reference.clone(dtype=jnp.float64), 6))


def test_a_gate_on_the_block_exits_by_row():
    """A gate on the block's output runs the coda once on its tokens' states, which a token's own exit
    would make a causal prefix of mixed passes: refused, as a gate placement without a gate is."""
    with pytest.raises(ValueError):
        Loop(3, exit_gate=True, exit_at="block")
    with pytest.raises(ValueError):
        Loop(3, exit_by="row")
    Loop(3, exit_gate=True, exit_at="block", exit_by="row")


def _sharing(steps: int) -> CausalTransformer:
    """Gemma 4's tiny release sharing its keys and values (its layers 2 and 3 read layers 0 and 1's),
    looping its block [1, 3) `steps` times: a provider before the block every pass reads, one in it its
    coda reads."""
    return _decoder("gemma4-kvshare").with_passes(steps, (1, 3))


def _sharing_by_hand(module, ids):
    """`module`'s logits with its block's passes unrolled from the stack's own spans, each part's layers
    reading its providers' publications as their lifetimes give them: the prelude's in every pass and the
    coda, a pass's own in it, the last pass's in the coda."""
    loop, inputs = module.loop, _inputs(module, ids)
    first, end = loop.layers
    x = prepared_embeddings(module.scaled_embeddings(module.token_embeddings(ids)), None)
    prelude = {}
    h = module.stack(x, span=(0, first), kv_store=prelude, **inputs)
    for _ in range(loop.steps):
        left = dict(prelude)
        h = module.stack(h, span=(first, end), kv_store=left, **inputs)
    out = module.stack(h, span=(end, module.num_layers), kv_store=dict(left), **inputs)
    return module.logits_from_hidden(module.norm(out))


@attending("reference")
def test_a_loop_over_shared_keys_and_values_computes_its_parts_lifetimes():
    """Gemma 4 sharing its keys and values, looped over a block a provider before it feeds and one in it
    feeds the coda after it (row 12930): once, the decoder it loops; twice, its logits and every
    parameter's gradient on par with the parts unrolled here (`_sharing_by_hand`), check 1 against them in
    float64."""
    once, unlooped = _sharing(1), _decoder("gemma4-kvshare")
    draw = model_draw(once, (2, 12), once.vocab_size)
    assert_as_exact(jax.jit(once.apply), jax.jit(unlooped.apply), draw, "a sharing loop of one pass",
                    seeds=SEEDS["sharing"], truth_fn=jax.jit(unlooped.clone(dtype=jnp.float64).apply))
    twice = _sharing(2)

    def unrolled(model):
        return lambda variables, ids: model.apply(variables, ids, method=_sharing_by_hand)

    def pulled(forward):
        return jax.jit(lambda variables, ids, cotangent:
                       with_gradients(lambda v: forward(v, ids))(variables, cotangent))

    drawn = with_logit_cotangent(draw, twice.vocab_size, 0xE94)
    assert_as_exact(pulled(twice.apply), pulled(unrolled(twice)), drawn, "a sharing loop's backward",
                    seeds=SEEDS["sharing backward"],
                    truth_fn=pulled(unrolled(twice.clone(dtype=jnp.float64))))


@attending("reference")
def test_a_loop_over_shared_keys_and_values_decodes_as_it_prefills():
    """Gemma 4's sharing loop of two passes, its cache a pass's own, decoded a token at a time after a
    prompt of eight, on par with its prefill over every token (check 1 against it in float64)."""
    twice = _sharing(2)
    draw = model_draw(twice, (2, 12), twice.vocab_size)
    assert_as_exact(_decoded(twice, 8), jax.jit(twice.apply), draw, "a sharing loop's decode",
                    seeds=SEEDS["sharing decode"], truth_fn=jax.jit(twice.clone(dtype=jnp.float64).apply))
