"""Cross-pass messages (`Loop.messages`, `dew.nn.cross_pass`): a looped block whose passes each enter at the
block's input and couple through the bank of block summaries the pass before left, merged into every
sublayer input of the next. A merge is its formula; at its gates' draw (zero) the loop computes the decoder
it repeats; at drawn gates it computes the passes unrolled from the stack's own spans, scanned, cut,
recomputed and decoded alike; and it trains, resumes and loads as a loop does (check 1 throughout)."""

import dataclasses
import functools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from map_judge import marked_seeds
from model_support import TINY_DECODER, as_stepped, decode_steps, model_draw, with_logit_cotangent
from reference_error import assert_as_exact, with_gradients
from test_loop_sampled import counts_compiled_off_the_loop

from dew.nn.attention_residuals import AttentionResiduals
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.backbones.embeddings import prepared_embeddings
from dew.nn.cross_pass import MERGE_PREFIX, MESSAGES_UNWRITTEN, CrossPass, CrossPassMerge

VOCAB = 37

DELTA_NETS = {'linear_attention': {'mixer': {'class': 'gated_delta_net', 'fields': {
    'linear_num_key_heads': 2, 'linear_num_value_heads': 4, 'linear_key_head_dim': 6,
    'linear_value_head_dim': 8, 'linear_conv_kernel_dim': 4}}}}
"""Gated delta nets, a hybrid's recurrent layers, whose one input site their mixer reads."""

SEEDS = {"merge": 8, "zero gates": 32, "decode": 32, "packed": 8, "retrofitted": 8}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` on armada (XP-SIZE, the most
any of a site's cases asked)."""

UNROLLED_SEEDS = {"two": 8, "four": 512, "the last trained": 8, "recomputed": 512, "attention": 512,
                  "depth residuals": 8, "scanned layers": 128}
"""Check 1's declared seeds for each unrolled case, sized by `seeds_for` on armada (XP-SIZE); those of 512
drawn 128 a map task (`check1_shards`), the scanned layers' 128 drawn 32 a task (237 s whole in train 1156's
CI, row 10946)."""

MESSAGES = CrossPass(block_size=2, rank=4)
"""Messages over blocks of two layers through a controller of rank 4."""


def decoder(form: str, **loop) -> CausalTransformer:
    """A tiny decoder of four layers in `form`: attention alone ('attention'), a hybrid of gated delta nets
    and attention ('hybrid'), attention under Kimi K3's attention residuals in blocks of two ('depth'), or
    six attention layers scanned as one run (`scan_layers`, 'scanned'), or a hybrid whose block of four is
    two periods of a delta net and an attention layer, scanned (a periodic run, 'periodic'); its block
    [`first`, end) looped as `loop` says where it says anything, passing `MESSAGES` (blocks of four where
    scanned)."""
    fields = {"vocab_size": VOCAB, **TINY_DECODER, "num_layers": 4, "attention_impl": "reference",
              "tie_embeddings": False}
    if form == "scanned":
        fields |= {"num_layers": 6, "scan_layers": True}
    if form == "periodic":
        fields |= {"num_layers": 5, "scan_layers": True, "kinds": DELTA_NETS,
                   "layer_types": ("full_attention",) + ("linear_attention", "full_attention") * 2}
    if form == "hybrid":
        fields |= {"layer_types": ("linear_attention", "full_attention") * 2, "kinds": DELTA_NETS}
    if form == "depth":
        fields |= {"attention_residuals": AttentionResiduals(block_size=2)}
    if loop:
        block = {"depth": (2, 4), "scanned": (1, 6), "periodic": (1, 5)}.get(form, (1, 4))
        scanned = form in ("scanned", "periodic")
        messages = dataclasses.replace(MESSAGES, block_size=4) if scanned else MESSAGES
        loop = {"messages": messages, "step_norm": False, "layers": block, **loop}
        fields["loop"] = Loop(**loop)
    return CausalTransformer(**fields)


def gated(variables, seed: int):
    """`variables` with every merge's gates drawn off zero, small normals (`CrossPassMerge`'s forget, gain
    and added projections, its depth mixture's kernel), so each later pass reads the bank."""
    rng = np.random.default_rng(seed)

    def drawn(path, leaf):
        names = [getattr(key, "key", "") for key in path]
        if any(str(name).startswith("cross_pass_") for name in names) and names[-1] in (
                "forget", "gain", "added", "kernel"):
            return jnp.asarray(rng.normal(scale=0.3, size=leaf.shape), leaf.dtype)
        return leaf
    return jax.tree_util.tree_map_with_path(drawn, variables)


def gated_draw(model, shape=(2, 12), salt=0):
    """`model_draw`'s draw with the merges' gates drawn off zero (`gated`)."""
    draw = model_draw(model, shape, VOCAB)

    def drawn(seed):
        variables, ids = draw(seed)
        return gated(variables, seed + salt), ids
    return drawn


def merged_by_hand(params, read, bank, spec: CrossPass, eps: float, dtype=jnp.float64):
    """A merge's formula (the module doc of `dew.nn.cross_pass`) in `dtype`, from its parameters: in
    float32 the controller as Inkling 2 Mini's reference writes it, in float64 its truth."""
    with jax.default_matmul_precision("highest"):
        wide = {name: jnp.asarray(value, dtype) for name, value in params.items() if name != "depth"}
        read, bank = jnp.asarray(read, dtype), jnp.asarray(bank, dtype)
        keys = bank * jax.lax.rsqrt(jnp.mean(bank * bank, axis=-1, keepdims=True) + eps)
        depth = {name: jnp.asarray(value, dtype) for name, value in params["depth"].items()}
        query = depth["scale"] * depth["kernel"][:, 0]
        weights = jax.nn.softmax(keys @ query, axis=-1)
        mixed = jnp.einsum("bsn,bsnd->bsd", weights, keys if spec.normalize_values else bank)
        message = mixed * jax.lax.rsqrt(jnp.mean(mixed * mixed, axis=-1, keepdims=True) + eps)
        sigma = jnp.sqrt(jnp.mean(read * read, axis=-1, keepdims=True) + eps)
        state = (read / sigma) @ wide["input_down"] + message @ wide["message_down"]
        sent = jnp.tanh(state @ wide["gain"]) * message
        if spec.additive:
            sent = sent + jnp.tanh(state @ wide["added"])
        kept = (1 - jnp.tanh(state @ wide["forget"])) * read
        return kept + (sigma * sent if spec.scale_by_input_rms else sent)


@pytest.mark.parametrize("spec", [MESSAGES, dataclasses.replace(MESSAGES, additive=False),
                                  dataclasses.replace(MESSAGES, scale_by_input_rms=False,
                                                      normalize_values=False)],
                         ids=["additive", "gained alone", "unit scale, raw values"])
def test_a_merge_is_its_formula(spec):
    """A merge of a bank of three summaries into a sublayer input, every parameter drawn off its init,
    computes its formula (`merged_by_hand`) on par with the formula written in float32 (check 1 against
    it in float64), whether its message adds a term of its own, joins at the input's RMS and mixes the
    summaries normed or raw."""
    merge = CrossPassMerge(spec, 16, 1e-6)

    def draw(seed):
        rng = np.random.default_rng([0xC405, seed])
        read = rng.normal(size=(2, 5, 16)).astype(np.float32)
        bank = rng.normal(size=(2, 5, 3, 16)).astype(np.float32)
        # Every parameter drawn off its init, its gates and its depth query among them.
        params = jax.tree.map(lambda leaf: jnp.asarray(rng.normal(scale=0.3, size=leaf.shape), leaf.dtype),
                              nn.unbox(merge.init(jax.random.key(seed), read, bank)))
        return params, read, bank

    def by_hand(dtype):
        def formula(params, read, bank):
            return merged_by_hand(params["params"], read, bank, spec, 1e-6, dtype)
        return jax.jit(formula)

    assert_as_exact(jax.jit(merge.apply), by_hand(jnp.float32), draw, "a merge", seeds=SEEDS["merge"],
                    truth_fn=by_hand(jnp.float64))


def _messages_by_hand(module, ids):
    """`module`'s logits with its messages' passes unrolled from the stack's own spans: the prelude once,
    then each pass from the block's input a message block at a time, each block's summary banked for the
    next pass, the bank the pass `backprop_steps` cuts at held from the gradient, and the coda on the last
    pass's output."""
    from dew.nn.backbones.call_metadata import call_metadata

    loop = module.loop
    first, end = loop.layers or (0, module.num_layers)
    inputs = {"train": False, "decode": False, "per_layer_input": None,
              "attention_metadata": call_metadata(module, ids, *(False, *(None,) * 9))}
    x = module._expand(prepared_embeddings(module.scaled_embeddings(module.token_embeddings(ids)), None))
    entered = module.stack(x, span=(0, first), **inputs) if first else x
    cut = None if loop.backprop_steps is None else loop.steps - loop.backprop_steps
    bank, h = None, entered
    for step in range(loop.steps):
        bank = jax.lax.stop_gradient(bank) if step == cut else bank
        h, summaries = entered, []
        for span in loop.messages.blocks(first, end):
            h = module.stack(h, span=span, prior_bank=bank, **inputs)
            summaries.append(h if module.attention_residuals is None else h[:, :, -1])
        bank = jnp.stack(summaries, axis=2)
    if end < module.num_layers:
        h = module.stack(h, span=(end, module.num_layers), **inputs)
    return module.logits_from_hidden(module.head_states(module.exit_states(module._collapse(h)[1])))


def _results(forward):
    """`forward`'s logits and every parameter's gradient under a cotangent of them, one program."""
    def run(variables, ids, cotangent):
        logits, gradients = with_gradients(lambda variables: forward(variables, ids))(variables, cotangent)
        return {"logits": logits, "gradients": gradients}
    return jax.jit(run)


@pytest.mark.parametrize("form", ["attention", "hybrid", "depth", "one pass gaining messages"])
@pytest.mark.parametrize("steps", [1, 2, 3])
def test_at_its_gates_draw_a_loop_passing_messages_computes_the_decoder_it_repeats(form, steps):
    """A decoder whose block passes messages, its merges as drawn (gates at zero), computes the decoder
    without the loop: its logits and the gradient of every parameter the decoder holds, the merges'
    gradients apart, on par with the decoder's own in float64 (check 1), whether its layers attend, mix a
    hybrid's delta nets or read Kimi K3's depth residuals, one pass or more (two a later pass traced
    alone, three a scan of them); at one pass bit for bit. A one-pass loop gaining messages
    (`with_passes`) of more passes is the attention decoder's loop of them, which that case judges."""
    if form == "one pass gaining messages":
        # A loop of one pass, its step norm as `Loop` defaults it, gaining messages (`with_passes`).
        plain = decoder("attention")
        looped = dataclasses.replace(plain, loop=Loop(1, layers=(1, 4))).with_passes(steps, messages=MESSAGES)
        if steps > 1:
            assert looped.loop == decoder("attention", steps=steps).loop
            return
    else:
        plain, looped = decoder(form), decoder(form, steps=steps)
    draw = with_logit_cotangent(model_draw(looped, (2, 12), VOCAB), VOCAB, 0xC401 + steps)
    if steps == 1:
        variables, ids, _ = draw(0)
        np.testing.assert_array_equal(
            np.asarray(looped.apply(variables, ids)),
            np.asarray(plain.apply({"params": _without_merges(variables["params"])}, ids)))
        return
    looped_results = _results(looped.apply)

    def merged(variables, ids, cotangent):
        result = looped_results(variables, ids, cotangent)
        return {"logits": result["logits"], "gradients": {
            **result["gradients"], "params": _without_merges(result["gradients"]["params"])}}

    assert_as_exact(merged, _unlooped(form, jnp.float32), draw, f"{form} at zero gates, {steps} passes",
                    seeds=SEEDS["zero gates"], truth_fn=_unlooped(form, jnp.float64))


@functools.cache
def _unlooped(form: str, dtype):
    """The decoder of `form` without its loop in `dtype`, its logits and the gradients of its own
    parameters, compiled once for every pass count the zero-gates cases judge against it."""
    model = decoder(form).clone(dtype=dtype)

    def run(variables, ids, cotangent):
        logits, gradients = with_gradients(lambda tree: model.apply(tree, ids))(
            {**variables, "params": _without_merges(variables["params"])}, cotangent)
        return {"logits": logits, "gradients": gradients}
    return jax.jit(run)


def _without_merges(params):
    """`params` without the merges' parameters, the decoder's own alone."""
    def kept(tree):
        if not isinstance(tree, dict):
            return tree
        return {name: kept(value) for name, value in tree.items() if not str(name).startswith("cross_pass_")}
    return kept(dict(params))


@pytest.mark.parametrize("case, seeds", [
    pytest.param(("hybrid", {"steps": 2}), UNROLLED_SEEDS["two"], id="two"),
    pytest.param(("hybrid", {"steps": 4}), UNROLLED_SEEDS["four"], id="four",
                 marks=pytest.mark.check1_shards(seeds=512, per_task=128)),
    pytest.param(("hybrid", {"steps": 3, "backprop_steps": 1}), UNROLLED_SEEDS["the last trained"],
                 id="the last trained"),
    pytest.param(("hybrid", {"steps": 3, "remat": True}), UNROLLED_SEEDS["recomputed"], id="recomputed",
                 marks=pytest.mark.check1_shards(seeds=512, per_task=128)),
    pytest.param(("attention", {"steps": 3}), UNROLLED_SEEDS["attention"], id="attention",
                 marks=pytest.mark.check1_shards(seeds=512, per_task=128)),
    pytest.param(("depth", {"steps": 3}), UNROLLED_SEEDS["depth residuals"], id="depth residuals"),
    pytest.param(("scanned", {"steps": 3}), UNROLLED_SEEDS["scanned layers"], id="scanned-layers",
                 marks=pytest.mark.check1_shards(seeds=128, per_task=32))])
def test_a_loop_passing_messages_computes_its_passes_unrolled(case, seeds, request):
    """At gates drawn off zero, a loop passing messages computes its passes unrolled pass by pass from the
    stack's own spans (`_messages_by_hand`): its logits and every parameter's gradient, the prelude's and
    the merges' among them, on par in float64 (check 1), its later passes traced alone (two), scanned
    (four), cut where `backprop_steps` says (its bank held from the gradient, the prelude still training
    through every pass after the cut) and recomputed (`remat`)."""
    form, loop = case
    looped = decoder(form, **loop)
    draw = with_logit_cotangent(gated_draw(looped, salt=0xC402), VOCAB, 0xC403 + loop["steps"])

    def by_hand(dtype):
        return _results(lambda variables, ids: looped.clone(dtype=dtype).apply(
            variables, ids, method=_messages_by_hand))

    name = f"{form}, {loop}"
    marked = request.node.get_closest_marker("check1_shards")
    assert_as_exact(_results(looped.apply), by_hand(jnp.float32), draw, name,
                    seeds=marked_seeds(request) if marked else seeds, truth_fn=by_hand(jnp.float64))
    if loop.get("backprop_steps"):
        variables, ids, cotangent = draw(0)
        gradients = _results(looped.apply)(variables, ids, cotangent)["gradients"]["params"]
        assert np.any(np.asarray(gradients["layers_0"]["mlp"]["down_proj"]["kernel"]) != 0)


@pytest.mark.check1_shards(seeds=32, per_task=16)
def test_a_loop_passing_messages_decodes_as_it_prefills():
    """A hybrid decoder passing messages over three passes, its gates drawn off zero, each pass's cache its
    own, decodes token by token (a prefill of four, then one at a time) what its full passes over each
    prefix compute, its logits on par with theirs in float64 (check 1), its 32 seeds drawn 16 a map task
    (132 s whole in train 1180's CI, row 11793)."""
    looped = decoder("hybrid", steps=3)
    draw = gated_draw(looped, salt=0xC404)
    made, step = decode_steps(looped)

    def decoded(variables, ids, prompt=4):
        logits, cache = step(variables, made(variables, ids.shape[0]), ids[:, :prompt])
        steps = [logits]
        for position in range(prompt, ids.shape[1]):
            logits, cache = step(variables, cache, ids[:, position:position + 1])
            steps.append(logits)
        return jnp.concatenate(steps, axis=1)

    def passes(model):
        forward = jax.jit(model.apply)
        return lambda variables, ids: as_stepped(lambda held: forward(variables, held), ids, 4)

    assert_as_exact(decoded, passes(looped), draw, "decode", seeds=SEEDS["decode"],
                    truth_fn=passes(looped.clone(dtype=jnp.float64)))


@pytest.mark.parametrize("form", ["hybrid", "periodic"])
def test_a_shared_cache_prefills_as_a_cache_a_pass_does(form):
    """A hybrid decoder passing messages over four passes, its gates drawn off zero, prefills with one
    cache its passes share (`shared_cache`, each pass starting from the cache the step found) what it
    prefills with a cache a pass, bit for bit: the bank crosses its scanned passes as it crosses them
    apart, its block's layers one at a time or a periodic run of two periods (`PipelineStage`); and the
    shared cache it leaves holds, layer by layer, its last pass's entries, those the cache a pass keeps
    of its last pass (`pass_offset`), and the prelude's alike."""
    from dew.nn.backbones.looping import pass_offset

    models = [decoder(form, steps=4, shared_cache=shared) for shared in (True, False)]
    variables, ids = gated_draw(models[0], salt=0xC407)(0)
    prefilled, caches = [], []
    for model in models:
        made, step = decode_steps(model)
        logits, cache = step(variables, made(variables, ids.shape[0]), ids)
        prefilled.append(np.asarray(logits))
        caches.append(cache)
    np.testing.assert_array_equal(*prefilled)
    if form == "periodic":
        return
    shared, apart = caches
    loop = models[1].loop
    first, end = loop.layers
    offset = pass_offset(loop, models[1].num_layers, loop.steps - 1)
    named = {f"layers_{index}": f"layers_{index}" for index in range(first)} | {
        f"layers_{first + index}": f"layers_{offset + index}" for index in range(end - first)}
    for kept, last in named.items():
        for one, other in zip(jax.tree.leaves(shared[kept]), jax.tree.leaves(apart[last]), strict=True):
            np.testing.assert_array_equal(np.asarray(one), np.asarray(other), err_msg=kept)


def test_a_packed_attention_projects_each_site_apart_as_its_own_projections_do():
    """A decoder passing messages packs none of its block's attention projections anew (`pending`: a later
    pass would run the packed product once a site), the prelude's alone, its feed-forwards' as any; a
    tree a server packed as the decoder without the loop packs (`qkv_proj`) runs each site's input apart
    through the packed product (`ProjectionGroup.apart`), computing what its own three projections compute
    at drawn gates (check 1 against them in float64), and unpacks back to them (`unpacked`)."""
    from dew.nn.projections import pack_projections, pending, unpacked

    looped = decoder("attention", steps=3)
    draw = gated_draw(looped, salt=0xC406)
    variables, _ = draw(0)
    def attending(groups):
        return {group.path[0] for group in groups if group.packed == "qkv_proj"}

    assert attending(pending(looped, variables)) == {"layers_0"}
    plain = decoder("attention")
    groups = pending(plain, {"params": _without_merges(variables["params"])})
    assert attending(groups) == {f"layers_{index}" for index in range(4)}
    packed_tree = pack_projections(variables, groups)
    for one, other in zip(jax.tree.leaves(unpacked(looped, packed_tree)), jax.tree.leaves(variables),
                          strict=True):
        np.testing.assert_array_equal(np.asarray(one), np.asarray(other))

    def packed(variables, ids):
        return looped.apply(pack_projections(variables, groups), ids)

    assert_as_exact(jax.jit(packed), jax.jit(looped.apply), draw, "packed attention",
                    seeds=SEEDS["packed"], truth_fn=jax.jit(looped.clone(dtype=jnp.float64).apply))


def test_a_decoder_drawn_at_one_pass_holds_every_merge_a_later_pass_reads():
    """A decoder passing messages drawn at one pass, which reads no bank, holds every merge a later pass
    reads (each core layer's sites', their depth mixtures' among them), so the same parameters run it at
    three passes; the layers before its block hold none."""
    once, thrice = decoder("hybrid", steps=1), decoder("hybrid", steps=3)
    ids = jnp.zeros((1, 6), jnp.int32)
    params = nn.unbox(once.init(jax.random.key(0), ids))["params"]
    assert not any(name.startswith("cross_pass_") for name in params["layers_0"])
    assert {name for name in params["layers_1"] if name.startswith("cross_pass_")} == {
        "cross_pass_q", "cross_pass_k", "cross_pass_v", "cross_pass_mlp"}
    assert {name for name in params["layers_2"] if name.startswith("cross_pass_")} == {
        "cross_pass_mixer", "cross_pass_mlp"}
    drawn = nn.unbox(thrice.init(jax.random.key(0), ids))["params"]
    assert jax.tree.structure(params) == jax.tree.structure(drawn)
    assert np.all(np.isfinite(np.asarray(thrice.apply({"params": params}, ids))))


@pytest.mark.parametrize("refused", [
    lambda: Loop(2, messages=MESSAGES),
    lambda: Loop(2, step_norm=False, implicit=True, messages=MESSAGES),
    lambda: Loop(2, step_norm=False, layers=(1, 1), messages=MESSAGES),
    lambda: CrossPass(block_size=0),
    lambda: decoder("depth", steps=2, layers=(1, 4)).init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)),
    lambda: dataclasses.replace(decoder("depth", steps=2), loop=Loop(
        2, step_norm=False, layers=(2, 4), messages=CrossPass(block_size=1))).init(
        jax.random.key(0), jnp.zeros((1, 4), jnp.int32)),
    lambda: dataclasses.replace(decoder("attention"), kv_shared_layers=(3,), loop=Loop(
        2, step_norm=False, messages=MESSAGES)).init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32)),
    lambda: decoder("attention", steps=2, messages=None).with_passes(3, messages=MESSAGES),
    lambda: dataclasses.replace(decoder("attention", steps=2), normed_residual=True).init(
        jax.random.key(0), jnp.zeros((1, 4), jnp.int32)),
    lambda: decoder("depth", steps=2, exit_gate=True).init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))],
    ids=["a step norm", "implicit", "an empty block", "an empty message block", "depth blocks misaligned",
         "a message block not the depth block", "shared keys", "messages joining a loop of passes",
         "a normed residual", "exits over depth residuals"])
def test_a_loop_whose_messages_cannot_summarize_its_blocks_is_refused(refused):
    """A loop passing messages is refused where its passes would not each enter at the block's input (a
    step norm, the θ-method), where it banks nothing (an empty block or message block), where under
    attention residuals a summary would not be a whole depth block's partial sum or a pass would exit
    through a depth state uncollapsed, where a layer's keys would cross a message block's end, and where
    it would join a loop already of several passes, changing what it computes at the merges' draw, and
    where a normed residual makes a sublayer's normed input the stream its merge would leave as it is."""
    with pytest.raises(ValueError):
        refused()


def test_a_loaded_decoder_gains_messages_that_leave_it_as_it_was_and_train_fresh():
    """A loaded decoder that gains a loop passing messages (`with_passes`, `Pretrained.with_model`) draws
    each merge fresh (`Pretrained.fresh`, the fresh group's alone) beside its loaded weights, which stay as
    they were, bit for bit, and at the merges' draw computes what the source computes (check 1 against it
    in float64); it exports to the source's format no more, its merges the source's format cannot hold."""
    from test_decision_context import LLAMA

    from dew.interop import Pretrained
    from dew.optim import OptimConfig, ParamGroup

    source = Pretrained.load(LLAMA, dtype="float32", attention_impl="reference")
    layers = source.model.num_layers
    looped = source.with_model(source.model.with_passes(3, (1, layers), messages=MESSAGES), key=3)
    assert looped.fresh and all("cross_pass_" in path for path in looped.fresh)
    fresh = ParamGroup("fresh", fresh=True, learning_rate_multiplier=10.0, spared=True)
    claimed = OptimConfig(param_groups=(fresh, ParamGroup("base", ("*",), spared=True))).with_fresh(
        (), looped.fresh).param_groups
    assert set(claimed[0].patterns) == set(looped.fresh)

    kept = _without_merges(looped.variables["params"])
    loaded = jax.tree_util.tree_leaves_with_path(source.variables["params"])
    for (path, held), (_, was) in zip(jax.tree_util.tree_leaves_with_path(kept), loaded, strict=True):
        np.testing.assert_array_equal(np.asarray(held), np.asarray(was), err_msg=jax.tree_util.keystr(path))

    def draw(seed):
        ids = jax.random.randint(jax.random.key(seed), (2, 8), 0, source.model.vocab_size)
        return looped.variables, ids

    def plain(dtype):
        return jax.jit(lambda _, ids: source.model.clone(dtype=dtype).apply(source.variables, ids))

    assert_as_exact(jax.jit(looped.model.apply), plain(jnp.float32), draw, "a loaded decoder retrofitted",
                    seeds=SEEDS["retrofitted"], truth_fn=plain(jnp.float64))
    with pytest.raises(ValueError):
        looped.export()


def test_a_source_written_in_its_own_bindings_refuses_its_export_once_it_passes_messages():
    """A source whose family writes it back in the bindings it was read from (Mixtral's,
    `preserve_source_layout`) exports as it loaded; gaining a loop passing messages (`with_passes`), whose
    merges its format holds none of, its export is refused rather than written without them."""
    from dew.interop import Pretrained

    source = Pretrained.load(Path(__file__).parent / "fixtures" / "hf" / "mixtral-tiny", dtype="float32",
                             attention_impl="reference")
    assert source.export()
    layers = source.model.num_layers
    looped = source.with_model(source.model.with_passes(2, (1, layers), messages=MESSAGES), key=4)
    with pytest.raises(ValueError) as refused:
        looped.export()
    assert refused.value.args == (MESSAGES_UNWRITTEN,)


@pytest.mark.usefixtures(counts_compiled_off_the_loop.__name__)
def test_a_loop_passing_messages_trains_sampled_counts_exits_and_narrow_copies():
    """A bf16 decoder passing messages over a block every pass exits through (an exit gate), its passes
    recomputed and drawn per microbatch (`SampledPasses`), its parameters read through narrow copies,
    trains: each step's loss finite, every count drawn its own program, and its merges' gates moved off
    zero, the bank reaching them."""
    from test_loop_sampled import TRAINED, token_batch
    from test_pass_weights import phased_trainer

    from dew.nn.kernels.generation import chosen
    from dew.training.interval import Interval

    model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=3, num_heads=2, mlp_features=48,
                              max_seq_len=128, attention_impl="xla", dtype=jnp.bfloat16,
                              loop=Loop(TRAINED.cap, step_norm=False, layers=(1, 3), exit_gate=True,
                                        remat=True, messages=CrossPass(block_size=1, rank=4)))
    batch = token_batch()
    with chosen(narrow_copies="on"):
        trainer = phased_trainer(model, 0, None, sampled_passes=TRAINED, accumulation=2)
        state, _, _ = trainer.place()
        step = trainer.compile(state, batch)
        interval = Interval(0.0, None, microbatches=0)
        drawn = []
        for index in range(6):
            step.drawn(interval, index, state)
            drawn.extend(step.counts)
            state, loss, *_ = step(state, batch)
            assert np.isfinite(float(loss)), (index, step.counts)
    assert len(set(drawn)) > 1, drawn
    named = [(jax.tree_util.keystr(path), leaf)
             for path, leaf in jax.tree_util.tree_leaves_with_path(state.variables["params"])]
    gains = [np.asarray(leaf) for name, leaf in named if "cross_pass_" in name and name.endswith("['gain']")]
    assert gains and any(np.any(gain != 0) for gain in gains)


def test_a_loop_passing_messages_resumes_as_it_never_stopped(tmp_path):
    """A run of a decoder passing messages stopped after two steps and resumed to four lands, bit for bit,
    where the run never stopped lands: its merges are parameters it checkpoints and resumes (`Loop.PASSED`
    holds no messages, so a resume compares them)."""
    from test_data_mixtures import assert_same, parameters
    from test_pass_weights import phased_draw, phased_loader, phased_trainer

    from dew.training import Checkpoints

    tokens, _ = phased_draw(0)
    model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=3, num_heads=2, mlp_features=48,
                              max_seq_len=128, attention_impl="xla",
                              loop=Loop(3, step_norm=False, layers=(1, 3), messages=CrossPass(block_size=1,
                                                                                               rank=4)))
    assert "messages" not in Loop.PASSED

    def fitted(directory, steps):
        return phased_trainer(model, 0, None, checkpoints=Checkpoints(str(directory), keep=8)).fit(
            phased_loader(tokens), steps=steps, log_every=100, checkpoint_every=1)

    whole = fitted(tmp_path / "whole", 4)
    fitted(tmp_path / "resumed", 2)
    resumed = fitted(tmp_path / "resumed", 4)
    assert int(resumed.step) == 4
    assert_same(parameters(whole), parameters(resumed), "the resumed run is not the run unstopped")


def test_a_loop_passing_messages_bounds_its_step_memory_across_unrolled_passes():
    """A bf16 decoder passing messages over an unrolled exit-gated loop (`Loop.messages`, `exit_gate=True`),
    every block rematerialized (`full`), compiled ahead on two compile-only H100s splitting its rows: its
    live memory past its arguments grows from two passes to six by no more than a message-free loop's growth
    plus the added passes' saved banks, rather than holding deferred fp32 merge intermediates or un-summed
    bank cotangents live across unrolled passes (row 11713)."""
    from step_memory import MIXER_HEAD, MIXER_LENGTH, MIXER_ROWS, MIXER_WIDTH, STEP_LAYOUT
    from token_support import token_rows

    from dew.nn.backbones.decoder_block import REMAT_POLICIES
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.objectives.lm import LMObjective
    from dew.optim import OptimConfig
    from dew.training import MeshSpec, Trainer
    from dew.training.ahead import compile_ahead

    devices = 2
    spec = CrossPass(block_size=2, rank=64)

    def live_past_args(steps: int, messages: CrossPass | None) -> int:
        model = CausalTransformer(
            vocab_size=512, emb_features=MIXER_WIDTH, num_layers=4, num_heads=8, head_dim=MIXER_HEAD,
            mlp_features=4 * MIXER_WIDTH, max_seq_len=MIXER_LENGTH, dtype="bfloat16",
            remat=REMAT_POLICIES["full"],
            loop=Loop(steps, step_norm=False, layers=(0, 4), exit_gate=True, messages=messages))
        trainer = Trainer(LMObjective(model, MIXER_LENGTH, exit_entropy=0.1),
                          OptimConfig(optimizer="adamw", learning_rate=1e-3).build(None),
                          key=jax.random.key(0),
                          mesh=MeshSpec(fsdp=devices).build(compile_only_gpus("sm90", devices, 2**36)),
                          layout=STEP_LAYOUT, split=1)
        ahead = compile_ahead(trainer, token_rows(512, MIXER_ROWS, MIXER_LENGTH + 1))
        return ahead.live_bytes - ahead.argument_bytes

    msg_two, msg_six = live_past_args(2, spec), live_past_args(6, spec)
    plain_two, plain_six = live_past_args(2, None), live_past_args(6, None)
    msg_growth, plain_growth = msg_six - msg_two, plain_six - plain_two
    # Four added passes each hold beyond a message-free loop's its saved bf16 bank [B/fsdp, S, blocks, D]
    # and that bank's one summed cotangent: two banks a pass.
    added_banks = 4 * (MIXER_ROWS // devices) * MIXER_LENGTH * 2 * MIXER_WIDTH * 2
    assert msg_growth <= plain_growth + 2 * added_banks, (
        f"cross-pass live growth {msg_growth} B exceeded plain growth {plain_growth} B + "
        f"2 * added_banks {2 * added_banks} B "
        f"(msg: {msg_two} -> {msg_six}, plain: {plain_two} -> {plain_six})")


@pytest.mark.parametrize("watch", ["captured", "inspected"])
def test_a_watched_merge_runs_its_depth_mixture_as_its_own_call(watch):
    """A caller watching a merge's calls, capturing every module's outputs (`capture_intermediates`) or
    summarizing its depth mixture's (`inspection.watching`, `INSPECTED`), sees each merge's depth mixture
    called, as a merge no one watches computes it through its reverse-mode rule (`portable_formed`, row
    11713's review): the merge runs as Flax sees it, its gates off zero (`gated`) so its output reads the
    mixture, its logits the unwatched run's."""
    from dew.nn.inspection import INSPECTED, watching

    model = decoder("attention", steps=2)
    ids = jnp.asarray(np.random.default_rng(11713).integers(0, VOCAB, (2, 12)), jnp.int32)
    variables = gated(model.init(jax.random.key(0), ids), 11713)
    if watch == "captured":
        logits, seen = model.apply(variables, ids, capture_intermediates=True, mutable=["intermediates"])
    else:
        with watching(("DepthAttention",)):
            logits, seen = model.apply(variables, ids, mutable=[INSPECTED])
    paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_leaves_with_path(seen)]
    assert any(MERGE_PREFIX in path and "'depth'" in path for path in paths)
    np.testing.assert_array_equal(np.asarray(logits), np.asarray(model.apply(variables, ids)))


def qwix_owned(provider: str, rule: dict, spec: CrossPass = MESSAGES) -> list:
    """The gradients of a two-pass decoder passing messages by `spec`, quantized by a direct Qwix provider of
    `rule` (`QtProvider`'s or `OdmlQatProvider`'s), its gates off zero (`gated`), each beside the same
    model's run as a watched merge runs (captured,
    `capture_intermediates`) and the unquantized model's: `(owned, watched, plain)` leaves."""
    qwix = pytest.importorskip("qwix")

    model = decoder("attention", steps=2, messages=spec)
    ids = jnp.asarray(np.random.default_rng(11716).integers(0, VOCAB, (2, 12)), jnp.int32)
    made = {"qt": qwix.QtProvider, "odml": qwix.OdmlQatProvider}[provider]
    # A QAT provider's jit off where Qwix finds a weight by its value; ODML's activation check off, Dew's ops
    # being more than its converter's (the provider's own refusal of an op it does not convert).
    options = {"disable_jit": True} if provider == "qt" else {"strict": False}
    quantized = qwix.quantize_model(model, made([qwix.QtRule(**rule)], **options))
    variables = gated(quantized.init(jax.random.key(0), ids), 11716)

    def gradients(applied, **watched):
        def loss(params):
            out = applied({**variables, "params": params}, ids, **watched)
            logits = out[0] if watched else out
            return jnp.sum(jnp.square(logits.astype(jnp.float32)))
        return [np.asarray(leaf) for leaf in jax.tree.leaves(jax.grad(loss)(variables["params"]))]

    return (gradients(quantized.apply), gradients(quantized.apply, capture_intermediates=True,
                                                  mutable=["intermediates"]), gradients(model.apply))


@pytest.mark.parametrize(("provider", "rule", "spec"), [
    ("qt", {"module_path": r".*cross_pass_[^/]+/depth", "weight_qtype": jnp.int8, "act_qtype": jnp.int8,
            "op_names": ("einsum",)}, MESSAGES),
    ("odml", {"module_path": r".*cross_pass_[^/]+/depth", "act_qtype": jnp.int8, "op_names": ("softmax",)},
     MESSAGES),
    ("qt", {"module_path": r".*cross_pass_[^/]+", "weight_qtype": jnp.int8, "act_qtype": jnp.int8,
            "op_names": ("dot_general",)}, dataclasses.replace(MESSAGES, additive=False))],
    ids=["depth einsums", "depth softmax", "merge matmuls"])
def test_a_merge_a_direct_qwix_rule_owns_takes_its_gradients_through_the_rule(provider, rule, spec):
    """A merge whose ops a direct `qwix.quantize_model` rule owns, its depth child's alone (`QtProvider`'s
    einsums, `OdmlQatProvider`'s softmax) or its own alone (its controller's matmuls: a merge of no added
    projection, whose two same-shaped gates Qwix tells apart by their values), runs as Flax and Qwix see it
    in both modes (row 11713's review: recomputed apart, its reverse mode bypassed the rule): its gradients
    the ones of the same model run as a watched merge runs, and the rule's, not the unquantized model's."""
    owned, watched, plain = qwix_owned(provider, rule, spec)
    for got, seen in zip(owned, watched, strict=True):
        np.testing.assert_array_equal(got, seen)
    assert any(not np.array_equal(got, unquantized) for got, unquantized in zip(owned, plain, strict=True))
