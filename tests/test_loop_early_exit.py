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
from flax import linen as nn
from jax.extend import core as jex_core
from model_support import TINY_DECODER, attending, decode_steps, model_draw, stacked, with_logit_cotangent
from reference_error import assert_as_exact, with_gradients

from dew.interop.hf_decoders import translate_config
from dew.nn.backbones.call_metadata import call_metadata
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.backbones.embeddings import prepared_embeddings
from dew.nn.backbones.looping import gate_logit
from dew.nn.cross_attention import CrossAttention
from dew.nn.inputs import CONTEXT, CONTEXT_LENGTHS

FIXTURES = Path(__file__).with_name("fixtures") / "hf"

SEEDS = {"rows": 8, "first pass": 8, "last pass": 8, "sharing": 8, "sharing backward": 8,
         "sharing decode": 8, "sharing parts": 8, "counted sharing": 8, "cross rows": 8}
"""Check 1's declared seeds for each site of the file."""


def _decoder(name: str) -> CausalTransformer:
    """A tiny release fixture as Dew builds it (`translate_config`), in float32."""
    config = json.loads((FIXTURES / name / "config.json").read_text())
    return CausalTransformer(**{**translate_config(config), "dtype": "float32"})


def _from(name: str, **config) -> CausalTransformer:
    """A tiny release fixture as Dew builds it with `config`'s fields over its own, in float32."""
    source = json.loads((FIXTURES / name / "config.json").read_text())
    return CausalTransformer(**{**translate_config({**source, **config}), "dtype": "float32"})


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


def _tokens_exiting(**loop) -> CausalTransformer:
    """Qwen 3.5's tiny decoder looping its block [1, 3) three times before its coda, its gate on the
    block's output exiting each token at its threshold, the loop's other fields as `loop` states."""
    looped = _decoder("qwen35-tiny").with_passes(3, (1, 3), exit_gate=True, exit_threshold=0.6)
    return looped.clone(loop=dataclasses.replace(looped.loop, exit_at="block", **loop))


@attending("reference")
@pytest.mark.parametrize(("bias", "site"), [(30.0, "first pass"), (-30.0, "last pass")])
def test_a_shared_cache_decodes_each_token_at_its_exiting_pass(bias, site):
    """The loop decoding ten tokens in sequence, a token a call, under a cache its passes share
    (`Loop.shared_cache`), its gate on the block's output exiting each token, a row's cache its token's
    exiting pass's writes and the coda's its one run's (row 12928): a gate certain to exit decodes what the
    decoder looping once does, and one that never exits what the loop running every pass without a
    threshold does (check 1, each against its own in float64)."""
    looped = _tokens_exiting(shared_cache=True)
    reference = (looped.clone(loop=dataclasses.replace(looped.loop, steps=1)) if bias > 0
                 else looped.clone(loop=dataclasses.replace(looped.loop, exit_threshold=None)))
    draw = _gated(model_draw(looped, (2, 10), looped.vocab_size), looped.emb_features, 0xE93, bias=bias)
    assert_as_exact(_decoded(looped, 1), _decoded(reference, 1), draw, f"decoding exiting at the {site}",
                    seeds=SEEDS[site], truth_fn=_decoded(reference.clone(dtype=jnp.float64), 1))


@attending("reference")
def test_a_tokens_exit_and_output_read_no_later_token():
    """A loop exiting each token at its threshold, its gate on the block's output and its coda run once on
    the tokens' chosen states: a row's tokens after `t` changed, its tokens up to `t` keep their exit
    gates' logits, the passes they exit at and their logits, bit for bit (a causal LM's every token scores
    its prefix alone)."""
    looped = _tokens_exiting()
    variables, ids = _gated(model_draw(looped, (2, 12), looped.vocab_size), looped.emb_features, 0xE96)(0)
    changed = ids.at[:, 7:].set((ids[:, 7:] + 1) % looped.vocab_size)
    run = jax.jit(looped.apply)
    np.testing.assert_array_equal(np.asarray(run(variables, ids))[:, :7],
                                  np.asarray(run(variables, changed))[:, :7])

    def gates(tokens):
        _, exits = looped.apply(variables, tokens, mutable=["exits"])
        return [np.asarray(each)[:, :7] for each in exits["exits"]["logits"]]
    for kept, moved in zip(gates(ids), gates(changed), strict=True):
        np.testing.assert_array_equal(kept, moved)


@attending("reference")
def test_a_packed_documents_exit_and_output_read_no_neighbour():
    """A row packing a second document after the first, under a loop exiting each token at its threshold:
    the first document's logits stay as they are, bit for bit, whatever the second holds; a row's exit
    (`exit_by='row'`) would pool its documents, and a packed call refuses it."""
    looped = _tokens_exiting()
    variables, ids = _gated(model_draw(looped, (2, 12), looped.vocab_size), looped.emb_features, 0xE97)(0)
    segments = jnp.asarray([[1] * 7 + [2] * 5] * 2, jnp.int32)
    neighbour = ids.at[:, 7:].set((ids[:, 7:] + 1) % looped.vocab_size)
    packed = jax.jit(lambda v, tokens, s: looped.apply(v, tokens, segment_ids=s))
    np.testing.assert_array_equal(np.asarray(packed(variables, ids, segments))[:, :7],
                                  np.asarray(packed(variables, neighbour, segments))[:, :7])
    rows = _rows_exiting()
    with pytest.raises(ValueError):
        rows.apply(variables, ids, segment_ids=segments)


def test_a_rows_exit_is_refused_where_a_token_is_scored_causally():
    """A row's exit gate pools the row's tokens, its later ones deciding its earlier ones' passes: an LM
    objective refuses it, whether or not its loss reads each pass's exit (its threshold's evaluation leaks
    alike), and so does decode (its row the tokens so far); a row's gate without a gate, refused as any gate
    placement is."""
    from dew.objectives.lm import LMObjective

    rows = _rows_exiting()
    variables, ids = model_draw(rows, (2, 13), rows.vocab_size)(0)
    for exit_entropy in (None, 0.25):
        with pytest.raises(ValueError):
            LMObjective(rows, 12, exit_entropy=exit_entropy, token_accuracy=False)
    made, decode = decode_steps(rows)
    with pytest.raises(ValueError):
        decode(variables, made(variables, 2), ids[:, :1])
    with pytest.raises(ValueError):
        Loop(3, exit_by="row")


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


def test_every_passes_gate_reads_its_placement_and_pooling():
    """A row's gate pools the row's tokens at every pass, the last pass's among them, so each logit is one
    a row; a gate on the block's output reads the block's at every pass, the last's among them, not the
    coda's a gate after the coda reads."""
    rows = _rows_exiting()
    variables, ids = _gated(model_draw(rows, (2, 12), rows.vocab_size), rows.emb_features, 0xE98)(0)
    _, exits = jax.jit(lambda v, tokens: rows.apply(v, tokens, mutable=["exits"]))(variables, ids)
    for logit in exits["exits"]["logits"]:
        held = np.asarray(logit)
        np.testing.assert_array_equal(held, np.broadcast_to(held[:, :1], held.shape))
    block = _tokens_exiting()
    coda = block.clone(loop=dataclasses.replace(block.loop, exit_at="coda"))
    variables, ids = _gated(model_draw(block, (2, 12), block.vocab_size), block.emb_features, 0xE99)(0)

    def last(model):
        _, sown = jax.jit(lambda v, tokens: model.apply(v, tokens, mutable=["exits"]))(variables, ids)
        return np.asarray(sown["exits"]["logits"][-1])
    assert not np.array_equal(last(block), last(coda))


def test_a_rows_gate_pools_its_real_tokens_alone():
    """A row's gate reads the mean of its real tokens' states (`gate_logit`): a padding token's state,
    unspecified, a NaN among them, leaves the row's logit as it is, bit for bit."""
    rows = _rows_exiting()
    variables, _ = _gated(model_draw(rows, (2, 12), rows.vocab_size), rows.emb_features, 0xE9B)(0)
    states = jax.random.normal(jax.random.key(1), (2, 12, rows.emb_features))
    valid = jnp.arange(12)[None] < jnp.asarray([[9], [12]])
    poisoned = jnp.where(valid[..., None], states, jnp.nan)

    def gate(module, states, valid):
        return gate_logit(module, states, valid)
    run = jax.jit(lambda v, s, real: rows.apply(v, s, real, method=gate))
    np.testing.assert_array_equal(np.asarray(run(variables, poisoned, valid)),
                                  np.asarray(run(variables, states, valid)))


def test_a_block_gate_refuses_what_its_one_coda_cannot_read():
    """A gate on the block's output runs the coda once over its tokens' exiting states: a coda reading the
    block's published keys and values (Gemma 4's sharing layer after the block) would read the last pass's
    for every token, and a cached call collecting every pass's exit would run codas it keeps no cache for,
    each refused."""
    sharing = _sharing(2)
    gated = sharing.clone(loop=dataclasses.replace(sharing.loop, exit_gate=True, exit_threshold=0.6,
                                                   exit_at="block"))
    with pytest.raises(ValueError):
        model_draw(gated, (2, 12), gated.vocab_size)(0)
    block = _tokens_exiting(shared_cache=True)
    variables, ids = model_draw(block, (2, 12), block.vocab_size)(0)
    made, _ = decode_steps(block)
    with pytest.raises(ValueError):
        block.apply({**variables, "cache": made(variables, 2)}, ids[:, :1], decode=True,
                    mutable=["cache", "exits"])


def test_a_stopping_loop_runs_under_a_capture_of_every_method():
    """A loop under a threshold, which stops its passes where every token has exited, under a capture that
    keeps every method's output, whose helpers return no array: its passes run one by one, keeping each
    capture as Flax keeps it, no switch returning a non-array."""
    rows = _rows_exiting()
    variables, ids = _gated(model_draw(rows, (2, 12), rows.vocab_size), rows.emb_features, 0xE9C)(0)
    _, captured = rows.apply(variables, ids, capture_intermediates=lambda module, method: True,
                             mutable=["intermediates"])
    assert "looped" in captured["intermediates"], sorted(captured["intermediates"])


@attending("reference")
def test_a_row_exit_over_cross_attention_stops_its_passes_over_every_row():
    """A loop exiting by row over cross-attending layers, its call's context `[B, C, D]` and lengths `[B]`
    holding its rows but not its tokens: scoring under its threshold stops its passes over every row,
    compacting none (`looping._token_aligned`), its logits on par with its passes unrolled under a capture of
    every method, which stops none (check 1 against them in float64)."""
    model = CausalTransformer(**{**TINY_DECODER, "vocab_size": 64, "num_layers": 4, "dtype": "float32",
                                 "cross_attention": CrossAttention(layer_ids=(1, 2), context_length=6,
                                                                   context_features=24),
                                 "loop": Loop(3, layers=(1, 3), exit_gate=True, exit_threshold=0.6,
                                              exit_at="block", exit_by="row")})

    def plain(seed):
        rng = np.random.default_rng([0xE9D, seed])
        ids = jnp.asarray(rng.integers(1, 64, (2, 10)), jnp.int32)
        context = jnp.asarray(rng.normal(size=(2, 6, 24)), jnp.float32)
        lengths = jnp.asarray([4, 6], jnp.int32)
        conditioning = {CONTEXT: context, CONTEXT_LENGTHS: lengths}
        variables = nn.unbox(model.init(jax.random.key(seed), ids, conditioning=conditioning))
        return variables, ids, context, lengths

    def scored(module, *, captured: bool):
        def run(variables, ids, context, lengths):
            conditioning = {CONTEXT: context, CONTEXT_LENGTHS: lengths}
            if not captured:
                return module.apply(variables, ids, conditioning=conditioning)
            return module.apply(variables, ids, conditioning=conditioning, mutable=["intermediates"],
                                capture_intermediates=lambda each, method: True)[0]
        return jax.jit(run)
    assert_as_exact(scored(model, captured=False), scored(model, captured=True),
                    _gated(plain, model.emb_features, 0xE9E),
                    "a row exit over cross-attention", seeds=SEEDS["cross rows"],
                    truth_fn=scored(model.clone(dtype=jnp.float64), captured=True))


@attending("reference")
@pytest.mark.parametrize("looped", [
    pytest.param(lambda: _from("gemma-tiny", num_hidden_layers=6).clone(kv_shared_layers=(4, 5))
                 .with_passes(2, (4, 6)), id="a scanned block opening on a reader"),
    pytest.param(lambda: _from("gemma-tiny", num_hidden_layers=6).clone(kv_shared_layers=(5,))
                 .with_passes(2, (5, 6)), id="a provider closing a scanned prelude"),
])
def test_a_loops_parts_carry_their_providers_publications(looped):
    """A loop whose block opens on a layer reading a provider before it (its first layer's dtype read with
    that provider's publications), or whose prelude scans a run of layers before the provider its block
    reads (the scanned run's initialization writing the publications the block reads): its logits on par
    with its parts unrolled here (`_sharing_by_hand`), check 1 against them in float64."""
    model = looped()
    draw = model_draw(model, (2, 12), model.vocab_size)

    def unrolled(module):
        return jax.jit(lambda variables, ids: module.apply(variables, ids, method=_sharing_by_hand))
    scanned = stacked(scanned=True)(jax.jit(model.apply))
    assert_as_exact(scanned, unrolled(model), draw, "a loop's parts' publications",
                    seeds=SEEDS["sharing parts"], truth_fn=unrolled(model.clone(dtype=jnp.float64)))


@attending("reference")
def test_a_counted_loop_runs_where_its_coda_reads_none_of_its_blocks_publications():
    """A loop whose block holds a provider and the layer reading it, its coda reading neither: given a
    count of its passes per call, it runs every pass under one scan, the block's publications dropped
    across it, and computes the loop of that many passes (check 1 against it in float64)."""
    counted = _from("gemma-tiny", num_hidden_layers=6).clone(kv_shared_layers=(2,)).with_passes(3, (1, 3))
    fixed = counted.clone(loop=dataclasses.replace(counted.loop, steps=2))
    draw = model_draw(counted, (2, 12), counted.vocab_size)
    run = jax.jit(lambda variables, ids: counted.apply(variables, ids, loop_passes=jnp.int32(2)))
    assert_as_exact(run, jax.jit(fixed.apply), draw, "a counted loop over a block's own sharing",
                    seeds=SEEDS["counted sharing"], truth_fn=jax.jit(fixed.clone(dtype=jnp.float64).apply))


def _reindexed(**fields) -> CausalTransformer:
    """DeepSeek V4.1's tiny release cut to six layers on a plain residual (no mHC, engram or DSpark, which
    a loop refuses): a CSA2 Full layer at 2 publishing its entries (compressed two to one) and Reindex
    layers after it reading them and writing their selections, `fields` over its own."""
    config = json.loads((FIXTURES / "deepseek-v41-tiny" / "config.json").read_text())
    text = config["text_config"]
    ratios = text["compress_ratios"][:4] + [2] * 2 + text["compress_ratios"][text["num_hidden_layers"]:]
    config = {**config, "text_config": {
        **text, "num_hidden_layers": 6, "compress_ratios": ratios, "kv_source_layer_ids": [2],
        "index_source_layer_ids": [2, 3, 4, 5], "candidate_source_layer_id": 2,
        "engram_layer_ids": [1], "engram_num_embeddings": text["engram_num_embeddings"][:1],
        "dspark_target_layer_ids": [3, 4, 5]}}
    translated = {key: value for key, value in translate_config(config).items()
                  if key not in ("hyper_connections", "engram", "dspark")}
    return CausalTransformer(**{**translated, "dtype": "float32", **fields})


def _block_gated(model: CausalTransformer, layers: tuple[int, int], steps: int = 2) -> CausalTransformer:
    gated = model.with_passes(steps, layers, exit_gate=True, exit_threshold=0.6)
    return gated.clone(loop=dataclasses.replace(gated.loop, exit_at="block"))


def test_a_block_gate_refuses_a_coda_reading_its_blocks_publications():
    """A gate on the block's output runs the coda once over its tokens' exiting states, so a coda reading a
    key its block's last writer left (`MixerBase.publications`) is refused: Reindex layers after a looped
    Full layer reading its entries, and a Reuse layer after a looped Reindex layer reading its selection.
    A coda that reads only what it writes itself (each Reindex after a looped Reindex writing its own
    selection), and a single pass, which hands on its own publications, run, the latter as the loop
    without a threshold does, bit for bit."""
    ids = jnp.ones((2, 8), jnp.int32)
    plain = _reindexed()
    for refused in (_block_gated(plain, (2, 3)), _block_gated(plain.clone(kv_shared_layers=(5,)), (3, 5))):
        with pytest.raises(ValueError):
            refused.init(jax.random.key(0), ids)
    _block_gated(plain, (3, 4)).init(jax.random.key(0), ids)
    once = _block_gated(plain, (2, 3), steps=1)
    variables = once.init(jax.random.key(0), ids)
    free = once.clone(loop=dataclasses.replace(once.loop, exit_threshold=None))
    np.testing.assert_array_equal(np.asarray(jax.jit(once.apply)(variables, ids)),
                                  np.asarray(jax.jit(free.apply)(variables, ids)))


def test_a_row_exit_over_compressed_publications_stops_over_every_row():
    """A row-exit loop over a CSA2 Full layer, whose entries `[B, S // 2, D]` hold its rows but not its
    tokens: scoring under its threshold stops its passes over every row, compacting none, its logits
    finite."""
    model = _reindexed().with_passes(3, (2, 3), exit_gate=True, exit_threshold=0.6)
    model = model.clone(loop=dataclasses.replace(model.loop, exit_by="row", exit_at="coda"))
    variables, ids = _gated(model_draw(model, (2, 12), model.vocab_size), model.emb_features, 0xEA0)(0)
    assert np.all(np.isfinite(np.asarray(jax.jit(model.apply)(variables, ids))))


def test_a_rows_host_mask_is_gathered_with_its_rows():
    """A row-exit loop over softmax attention scored under a host (NumPy) pairwise mask, its rows exiting at
    different passes: each compacted pass reads its rows' own mask as it reads a JAX array's, its logits the
    same, bit for bit."""
    rows = _from("gemma-tiny", num_hidden_layers=4).with_passes(3, (1, 3), exit_gate=True, exit_threshold=0.6)
    rows = rows.clone(loop=dataclasses.replace(rows.loop, exit_at="block", exit_by="row"))
    variables, ids = _gated(model_draw(rows, (2, 12), rows.vocab_size), rows.emb_features, 0xEA1)(0)
    mask = np.broadcast_to(np.tril(np.ones((12, 12), bool)), (2, 12, 12))
    host = rows.apply(variables, ids, attention_pairwise_mask=mask)
    device = rows.apply(variables, ids, attention_pairwise_mask=jnp.asarray(mask))
    np.testing.assert_array_equal(np.asarray(host), np.asarray(device))


def test_a_cached_block_gate_collects_exits_where_no_earlier_coda_runs():
    """A cached call collecting every pass's exit under a gate on the block's output runs where no earlier
    pass runs a coda (a loop over the stack's last layers); one capturing every method's output, which no
    switch returns, is refused where it keeps each row's exiting pass's cache."""
    looped = _decoder("qwen35-tiny")
    whole = looped.with_passes(3, (1, looped.num_layers), exit_gate=True, exit_threshold=0.6)
    whole = whole.clone(loop=dataclasses.replace(whole.loop, exit_at="block", shared_cache=True))
    variables, ids = model_draw(whole, (2, 12), whole.vocab_size)(0)
    made, _ = decode_steps(whole)
    _, sown = whole.apply({**variables, "cache": made(variables, 2)}, ids[:, :1], decode=True,
                          mutable=["cache", "exits"])
    assert sown["exits"]["logits"]
    block = _tokens_exiting(shared_cache=True)
    variables, ids = model_draw(block, (2, 12), block.vocab_size)(0)
    made, _ = decode_steps(block)
    with pytest.raises(ValueError):
        block.apply({**variables, "cache": made(variables, 2)}, ids[:, :1], decode=True,
                    mutable=["cache", "intermediates"], capture_intermediates=lambda module, method: True)
