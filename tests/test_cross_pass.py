"""Cross-pass messages (`Loop.messages`, `dew.nn.cross_pass`): a looped block whose passes each enter at the
block's input and couple through the bank of block summaries the pass before left, merged into every
sublayer input of the next. A merge is its formula; at its gates' draw (zero) the loop computes the decoder
it repeats; at drawn gates it computes the passes unrolled from the stack's own spans, scanned, cut,
recomputed and decoded alike; and it trains, resumes and loads as a loop does (check 1 throughout)."""

import dataclasses
import functools
from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from jax.extend import core as jex_core
from map_judge import marked_seeds
from model_support import (
    TINY_DECODER,
    as_stepped,
    attending,
    compiled_apply,
    decode_steps,
    model_draw,
    with_logit_cotangent,
)
from reference_error import assert_as_exact, with_gradients

from dew.nn.attention_residuals import AttentionResiduals
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.backbones.embeddings import prepared_embeddings
from dew.nn.cross_pass import MERGE_PREFIX, MESSAGES_UNWRITTEN, CrossPass, CrossPassMerge, merged_sites
from dew.optim import AdamW
from dew.training import TrainerAdvanced

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


SCANNED_FORMS = ("scanned", "periodic")
"""The forms whose decoder scans its stack (`decoder`)."""


@pytest.fixture(autouse=True)
def scanned_forms(request):
    """A test of a form `decoder` scans (`SCANNED_FORMS`, its `form` or its case's first element) runs its
    decoders' stacks scanned (`generation.chosen(layer_scan="scanned")`), every other form's a layer at a
    time."""
    from dew.nn.kernels import generation

    named = getattr(getattr(request.node, "callspec", None), "params", {})
    case = named.get("case")
    form = named.get("form", case[0] if isinstance(case, tuple) else None)
    with generation.chosen(layer_scan="scanned" if form in SCANNED_FORMS else "plain"):
        yield


def decoder(form: str, **loop) -> CausalTransformer:
    """A tiny decoder of four layers in `form`: attention alone ('attention'), a hybrid of gated delta nets
    and attention ('hybrid'), attention under Kimi K3's attention residuals in blocks of two ('depth'), or
    six attention layers scanned as one run ('scanned'), or a hybrid whose block of four is
    two periods of a delta net and an attention layer, scanned (a periodic run, 'periodic'), each scanned
    as its test runs it (`scanned_forms`); its block
    [`first`, end) looped as `loop` says where it says anything, passing `MESSAGES` (blocks of four where
    scanned)."""
    fields = {"vocab_size": VOCAB, **TINY_DECODER, "num_layers": 4,
              "tie_embeddings": False}
    if form == "scanned":
        fields |= {"num_layers": 6}
    if form == "periodic":
        fields |= {"num_layers": 5, "kinds": DELTA_NETS,
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


def equations(closed) -> list:
    """Every equation of `closed`'s jaxpr, those of the jaxprs its equations hold among them."""
    found = []

    def walk(jaxpr):
        for eqn in jaxpr.eqns:
            found.append(eqn)
            for value in eqn.params.values():
                for each in value if isinstance(value, (tuple, list)) else (value,):
                    if isinstance(each, jex_core.ClosedJaxpr):
                        walk(each.jaxpr)
                    elif isinstance(each, jex_core.Jaxpr):
                        walk(each)
    walk(closed.jaxpr)
    return found


def products(closed) -> list:
    """`closed`'s dot products, every jaxpr's it holds."""
    return [eqn for eqn in equations(closed) if eqn.primitive.name == "dot_general"]


def test_an_attention_layers_sites_read_their_bank_once_each_by_its_formula():
    """Three merges of one bank into one sublayer input, as an attention layer's query, key and value sites
    run them (`merged_sites`), each its own parameters drawn off its init: each computes its own formula
    (`merged_by_hand`, check 1 against it in float64), the bank read by the products one site's read takes
    for them all (its keys' scores and its mixture, each over every site's query: row 12923), and each
    site's controller as many as that read (its down and up projections over the joined kernels, where
    apart it took five: row 12922)."""
    merges = tuple(CrossPassMerge(MESSAGES, 16, 1e-6, name=f"site_{index}") for index in range(3))
    shape = (2, 5, 3, 16)

    def draw(seed):
        rng = np.random.default_rng([0xC406, seed])
        read = rng.normal(size=shape[:2] + shape[3:]).astype(np.float32)
        bank = rng.normal(size=shape).astype(np.float32)
        def drawn(leaf):
            return jnp.asarray(rng.normal(scale=0.3, size=leaf.shape), leaf.dtype)
        params = tuple(jax.tree.map(drawn, nn.unbox(merge.init(jax.random.key(seed), read, bank)))["params"]
                       for merge in merges)
        return params, read, bank

    def ours(params, read, bank):
        bound = tuple(merge.bind({"params": each}) for merge, each in zip(merges, params, strict=True))
        return merged_sites(bound, read, bank)

    def by_hand(dtype):
        def formula(params, read, bank):
            return tuple(merged_by_hand(each, read, bank, MESSAGES, 1e-6, dtype) for each in params)
        return jax.jit(formula)

    assert_as_exact(jax.jit(ours), by_hand(jnp.float32), draw, "an attention layer's sites",
                    seeds=SEEDS["merge"], truth_fn=by_hand(jnp.float64))
    # Counted against one site traced alike, however this platform's programs form a product (`accumulated`).
    params, read, bank = draw(0)

    def reading(dots):
        return sum(any(var.aval.shape == shape for var in eqn.invars) for eqn in dots)
    one = products(jax.make_jaxpr(merges[0].apply)({"params": params[0]}, read, bank))
    together = products(jax.make_jaxpr(ours)(params, read, bank))
    # A site's controller as many products as its depth read: one down, one up (two scores and a sum).
    assert len(one) - reading(one) == reading(one)
    assert reading(together) == reading(one)
    assert len(together) == reading(one) + len(merges) * (len(one) - reading(one))


def test_a_bf16_merge_contracts_in_bf16_under_the_runs_precision():
    """A merge whose sublayer computes in bf16 takes every product of its depth read and its controller in
    bf16, at the precision the run gives bf16 operands (`dew.nn.precision.operand_precision`: a bf16 run's
    Tensor Cores), not in float32 at `Precision.HIGHEST` (row 12918)."""
    merge = CrossPassMerge(MESSAGES, 16, 1e-6)
    read = jnp.ones((2, 5, 16), jnp.bfloat16)
    bank = jnp.ones((2, 5, 3, 16), jnp.bfloat16)
    variables = jax.tree.map(lambda leaf: leaf.astype(jnp.bfloat16),
                             nn.unbox(merge.init(jax.random.key(0), read, bank)))
    dots = products(jax.make_jaxpr(merge.apply)(variables, read, bank))
    assert dots
    assert {str(var.aval.dtype) for eqn in dots for var in eqn.invars} == {"bfloat16"}
    highest = jax.lax.Precision.HIGHEST
    assert not any(highest in (precision if isinstance(precision := eqn.params["precision"], tuple)
                               else (precision,)) for eqn in dots)


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


@attending("reference")
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
            np.asarray(compiled_apply(looped, variables, ids)),
            np.asarray(compiled_apply(plain, {"params": _without_merges(variables["params"])}, ids)))
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


@attending("reference")
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


@attending("reference")
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


@attending("reference")
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


@attending("reference")
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
    def attended_by(groups):
        return {group.path[0] for group in groups if group.packed == "qkv_proj"}

    assert attended_by(pending(looped, variables)) == {"layers_0"}
    plain = decoder("attention")
    groups = pending(plain, {"params": _without_merges(variables["params"])})
    assert attended_by(groups) == {f"layers_{index}" for index in range(4)}
    packed_tree = pack_projections(variables, groups)
    for one, other in zip(jax.tree.leaves(unpacked(looped, packed_tree)), jax.tree.leaves(variables),
                          strict=True):
        np.testing.assert_array_equal(np.asarray(one), np.asarray(other))

    def packed(variables, ids):
        return looped.apply(pack_projections(variables, groups), ids)

    assert_as_exact(jax.jit(packed), jax.jit(looped.apply), draw, "packed attention",
                    seeds=SEEDS["packed"], truth_fn=jax.jit(looped.clone(dtype=jnp.float64).apply))


@attending("reference")
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


@attending("reference")
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


@attending("reference")
def test_a_loaded_decoder_gains_messages_that_leave_it_as_it_was_and_train_fresh():
    """A loaded decoder that gains a loop passing messages (`with_passes`, `Pretrained.with_model`) draws
    each merge fresh (`Pretrained.fresh`, the fresh group's alone) beside its loaded weights, which stay as
    they were, bit for bit, and at the merges' draw computes what the source computes (check 1 against it
    in float64); it exports to the source's format no more, its merges the source's format cannot hold."""
    from test_decision_context import LLAMA

    from dew.interop import Pretrained
    from dew.optim import OptimConfig, ParamGroup

    source = Pretrained.load(LLAMA, dtype="float32")
    layers = source.model.num_layers
    looped = source.with_model(source.model.with_passes(3, (1, layers), messages=MESSAGES), key=3)
    assert looped.fresh and all("cross_pass_" in path for path in looped.fresh)
    fresh = ParamGroup("fresh", fresh=True, learning_rate_multiplier=10.0, decay_norms_and_biases=False)
    claimed = OptimConfig(param_groups=(fresh, ParamGroup("base", ("*",),
                                                          decay_norms_and_biases=False))).with_fresh(
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


@attending("reference")
def test_a_source_written_in_its_own_bindings_refuses_its_export_once_it_passes_messages():
    """A source whose family writes it back in the bindings it was read from (Mixtral's,
    `preserve_source_layout`) exports as it loaded; gaining a loop passing messages (`with_passes`), whose
    merges its format holds none of, its export is refused rather than written without them."""
    from dew.interop import Pretrained

    source = Pretrained.load(Path(__file__).parent / "fixtures" / "hf" / "mixtral-tiny", dtype="float32")
    assert source.export()
    layers = source.model.num_layers
    looped = source.with_model(source.model.with_passes(2, (1, layers), messages=MESSAGES), key=4)
    with pytest.raises(ValueError) as refused:
        looped.export()
    assert refused.value.args == (MESSAGES_UNWRITTEN,)


@attending("xla")
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
                              max_seq_len=128, dtype=jnp.bfloat16,
                              loop=Loop(TRAINED.cap, step_norm=False, layers=(1, 3), exit_gate=True,
                                        remat=True, messages=CrossPass(block_size=1, rank=4)))
    batch = token_batch()
    with chosen(narrow_copies="on"):
        trainer = phased_trainer(model, 0, None, advanced=TrainerAdvanced(sampled_passes=TRAINED),
                                 accumulation=2)
        state, _, _ = trainer.place()
        step = trainer.compile(state, batch)
        interval = Interval(0.0, None, microbatches=0)
        drawn = []
        for index in range(6):
            step.drawn(interval, index)
            drawn.extend(step.counts)
            state, loss, *_ = step(state, batch)
            assert np.isfinite(float(loss)), (index, step.counts)
    assert len(set(drawn)) > 1, drawn
    named = [(jax.tree_util.keystr(path), leaf)
             for path, leaf in jax.tree_util.tree_leaves_with_path(state.variables["params"])]
    gains = [np.asarray(leaf) for name, leaf in named if "cross_pass_" in name and name.endswith("['gain']")]
    assert gains and any(np.any(gain != 0) for gain in gains)


@attending("xla")
def test_a_loop_passing_messages_resumes_as_it_never_stopped(tmp_path):
    """A run of a decoder passing messages stopped after two steps and resumed to four lands, bit for bit,
    where the run never stopped lands: its merges are parameters it checkpoints and resumes (`Loop.PASSED`
    holds no messages, so a resume compares them)."""
    from test_data_mixtures import assert_same, parameters
    from test_pass_weights import phased_draw, phased_loader, phased_trainer

    from dew.training import Checkpoints

    tokens, _ = phased_draw(0)
    model = CausalTransformer(vocab_size=256, emb_features=32, num_layers=3, num_heads=2, mlp_features=48,
                              max_seq_len=128,
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


@pytest.mark.parametrize("watched", [False, True], ids=["unwatched", "watching its MLPs"])
def test_a_loop_passing_messages_bounds_its_step_memory_across_unrolled_passes(watched):
    """A bf16 decoder passing messages over an exit-gated loop (`Loop.messages`, `exit_gate=True`), every
    block rematerialized (`full`), compiled ahead on two compile-only H100s splitting its rows: its live
    memory past its arguments grows from three passes to seven by no more than a message-free loop's growth
    plus the added passes' saved banks, rather than holding deferred fp32 merge intermediates or un-summed
    bank cotangents live across its passes (row 11713); so too where the step watches its MLPs. Each a cell
    of its own (row 11902: their six compiles in one cell took 360 s)."""
    from step_memory import MIXER_HEAD, MIXER_LENGTH, MIXER_ROWS, MIXER_WIDTH, STEP_LAYOUT
    from token_support import token_rows

    from dew.nn.backbones.decoder_block import REMAT_POLICIES
    from dew.nn.kernels.generation import compile_only_gpus
    from dew.objectives.lm import LMObjective
    from dew.optim import OptimConfig
    from dew.training import MeshSpec, Trainer
    from dew.training.ahead import compile_ahead
    from dew.training.inspection import Inspection

    devices = 2
    spec = CrossPass(block_size=2, rank=64)

    def live_past_args(steps: int, messages: CrossPass | None, inspection=None) -> int:
        model = CausalTransformer(
            vocab_size=512, emb_features=MIXER_WIDTH, num_layers=4, num_heads=8, head_dim=MIXER_HEAD,
            mlp_features=4 * MIXER_WIDTH, max_seq_len=MIXER_LENGTH, dtype="bfloat16",
            remat=REMAT_POLICIES["full"],
            loop=Loop(steps, step_norm=False, layers=(0, 4), exit_gate=True, messages=messages))
        trainer = Trainer(LMObjective(model, MIXER_LENGTH, exit_entropy=0.1),
                          OptimConfig(optimizer=AdamW(), lr=1e-3).build(None),
                          key=jax.random.key(0),
                          mesh=MeshSpec(fsdp=devices).build(compile_only_gpus("sm90", devices, 2**36)),
                          advanced=TrainerAdvanced(layout=STEP_LAYOUT, inspection=inspection), split=1)
        ahead = compile_ahead(trainer, token_rows(512, MIXER_ROWS, MIXER_LENGTH + 1))
        if inspection is None:
            return ahead.live_bytes - ahead.argument_bytes
        # A run that inspects samples its watches in a step of its own, compiled ahead beside its step.
        memory = ahead.sampled.executable.memory
        return memory.live - memory.argument

    # Three passes to seven, each program's later passes under one scan: two passes' one later pass is
    # traced alone, a scan of one being no program of its own (`looping._Passes.fixed`).
    plain_two, plain_six = live_past_args(3, None), live_past_args(7, None)
    plain_growth = plain_six - plain_two
    # Four added passes each hold beyond a message-free loop's its saved bf16 bank [B/fsdp, S, blocks, D]
    # and that bank's one summed cotangent: two banks a pass.
    added_banks = 4 * (MIXER_ROWS // devices) * MIXER_LENGTH * 2 * MIXER_WIDTH * 2
    # So too where the step watches its MLPs (row 11820: a watch of other modules leaves the merges to their
    # own rules), its sampled step the one compiled.
    inspection = Inspection(modules=("*MLP",), every=1) if watched else None
    msg_two, msg_six = live_past_args(3, spec, inspection), live_past_args(7, spec, inspection)
    msg_growth = msg_six - msg_two
    assert msg_growth <= plain_growth + 2 * added_banks, (
        f"cross-pass live growth {msg_growth} B, watching {inspection}, exceeded plain growth "
        f"{plain_growth} B + 2 * added_banks {2 * added_banks} B "
        f"(msg: {msg_two} -> {msg_six}, plain: {plain_two} -> {plain_six})")


@attending("reference")
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


@attending("reference")
@pytest.mark.parametrize("watch", ["capture filter", "interceptor"])
def test_a_merge_runs_under_a_callers_unhashable_capture_filter_or_interceptor(watch):
    """A caller's `capture_intermediates` filter and method interceptor may be any callable, an unhashable
    one too (a dataclass's instance): a merge deciding whether it is watched matches Dew's own by identity
    (rows 11876, the filter, and 11877, the interceptor: hashing them raised), so the model runs under
    each, its logits the unwatched run's."""

    @dataclasses.dataclass
    class Calls:
        def __call__(self, module, method):
            return method == "__call__"

    @dataclasses.dataclass
    class Passing:
        def __call__(self, next_fun, args, kwargs, context):
            return next_fun(*args, **kwargs)

    model = decoder("attention", steps=2)
    ids = jnp.asarray(np.random.default_rng(11876).integers(0, VOCAB, (2, 12)), jnp.int32)
    variables = gated(model.init(jax.random.key(0), ids), 11876)
    expected = np.asarray(model.apply(variables, ids))
    if watch == "capture filter":
        logits, _ = model.apply(variables, ids, capture_intermediates=Calls(), mutable=["intermediates"])
    else:
        with nn.intercept_methods(Passing()):
            logits = model.apply(variables, ids)
    np.testing.assert_array_equal(np.asarray(logits), expected)


@attending("reference")
def test_a_looped_decoder_runs_under_a_capture_of_every_method():
    """A capture filter may keep every method's output (`capture_intermediates=lambda module, method: True`),
    a method returning a submodule among them: a looped decoder's later pass reads its layers' stored
    variables alone, never what the capture sowed, so it runs and captures methods beside the calls (row
    11882: the layers' structures read the captured submodule as an array and refused it)."""
    model = decoder("attention", steps=2)
    ids = jnp.asarray(np.random.default_rng(11882).integers(0, VOCAB, (2, 12)), jnp.int32)
    variables = gated(model.init(jax.random.key(0), ids), 11882)
    _, captured = model.apply(variables, ids, capture_intermediates=lambda module, method: True,
                              mutable=["intermediates"])
    # Each capture is kept under its method's name: the decoder's `looped`, which runs its passes, among them.
    assert "looped" in captured["intermediates"], sorted(captured["intermediates"])


def qwix_owned(provider: str, rules: Sequence[dict], spec: CrossPass = MESSAGES) -> tuple:
    """The gradients of a two-pass decoder passing messages by `spec`, quantized by a direct Qwix provider of
    `rules` (`QtProvider`'s or `OdmlQatProvider`'s), its gates off zero (`gated`), each beside the same
    model's run as a watched merge runs (captured, `capture_intermediates`) and the unquantized model's:
    `(owned, watched, plain)` leaves, and the rules no op matched (`get_unused_rules`)."""
    qwix = pytest.importorskip("qwix")

    model = decoder("attention", steps=2, messages=spec)
    ids = jnp.asarray(np.random.default_rng(11716).integers(0, VOCAB, (2, 12)), jnp.int32)
    made = {"qt": qwix.QtProvider, "odml": qwix.OdmlQatProvider}[provider]
    # A QAT provider's jit off where Qwix finds a weight by its value; ODML's activation check off, Dew's ops
    # being more than its converter's (the provider's own refusal of an op it does not convert).
    options = {"disable_jit": True} if provider == "qt" else {"strict": False}
    made = made([qwix.QtRule(**rule) for rule in rules], **options)
    quantized = qwix.quantize_model(model, made)
    variables = gated(quantized.init(jax.random.key(0), ids), 11716)

    def gradients(applied, **watched):
        def loss(params):
            out = applied({**variables, "params": params}, ids, **watched)
            logits = out[0] if watched else out
            return jnp.sum(jnp.square(logits.astype(jnp.float32)))
        return [np.asarray(leaf) for leaf in jax.tree.leaves(jax.grad(loss)(variables["params"]))]

    leaves = (gradients(quantized.apply), gradients(quantized.apply, capture_intermediates=True,
                                                    mutable=["intermediates"]), gradients(model.apply))
    return leaves, made.get_unused_rules()


@attending("reference")
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
    (owned, watched, plain), unused = qwix_owned(provider, [rule], spec)
    assert not unused
    for got, seen in zip(owned, watched, strict=True):
        np.testing.assert_array_equal(got, seen)
    assert any(not np.array_equal(got, unquantized) for got, unquantized in zip(owned, plain, strict=True))


@attending("reference")
def test_a_rule_over_a_merges_own_path_owns_its_matmuls_and_not_its_depth_mixtures_einsums():
    """Rules over a merge's own path (`cross_pass_<k>`, its depth child's not), one over its matmuls and one
    over einsums, which only its depth child computes: the matmuls' rule is used, the einsums' is not (a
    negative control of the merge-matmuls case, row 11805: its path owns no depth mixture)."""
    rules = [{"module_path": r".*cross_pass_[^/]+", "weight_qtype": jnp.int8, "act_qtype": jnp.int8,
              "op_names": (op,)} for op in ("dot_general", "einsum")]
    _, unused = qwix_owned("qt", rules, dataclasses.replace(MESSAGES, additive=False))
    assert [tuple(rule.op_names) for rule in unused] == [("einsum",)]
