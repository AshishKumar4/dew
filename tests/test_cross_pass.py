"""Cross-pass messages (`Loop.messages`, `dew.nn.cross_pass`): a looped block whose passes each enter at the
block's input and couple through the bank of block summaries the pass before left, merged into every
sublayer input of the next. A merge is its formula; at its gates' draw (zero) the loop computes the decoder
it repeats; at drawn gates it computes the passes unrolled from the stack's own spans, scanned, cut,
recomputed and decoded alike; and it trains, resumes and loads as a loop does (check 1 throughout)."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from map_judge import marked_seeds
from model_support import TINY_DECODER, as_stepped, decode_steps, model_draw, with_logit_cotangent
from reference_error import assert_as_exact, with_gradients

from dew.nn.attention_residuals import AttentionResiduals
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import Loop
from dew.nn.backbones.embeddings import prepared_embeddings
from dew.nn.cross_pass import CrossPass, CrossPassMerge

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
drawn 128 a map task (`check1_shards`)."""

MESSAGES = CrossPass(block_size=2, rank=4)
"""Messages over blocks of two layers through a controller of rank 4."""


def decoder(form: str, **loop) -> CausalTransformer:
    """A tiny decoder of four layers in `form`: attention alone ('attention'), a hybrid of gated delta nets
    and attention ('hybrid'), attention under Kimi K3's attention residuals in blocks of two ('depth'), or
    six attention layers scanned as one run (`scan_layers`, 'scanned'); its block [`first`, end) looped as
    `loop` says where it says anything, passing `MESSAGES`."""
    fields = {"vocab_size": VOCAB, **TINY_DECODER, "num_layers": 4, "attention_impl": "reference",
              "tie_embeddings": False}
    if form == "scanned":
        fields |= {"num_layers": 6, "scan_layers": True}
    if form == "hybrid":
        fields |= {"layer_types": ("linear_attention", "full_attention") * 2, "kinds": DELTA_NETS}
    if form == "depth":
        fields |= {"attention_residuals": AttentionResiduals(block_size=2)}
    if loop:
        block = {"depth": (2, 4), "scanned": (1, 6)}.get(form, (1, 4))
        messages = dataclasses.replace(MESSAGES, block_size=4) if form == "scanned" else MESSAGES
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


@pytest.mark.parametrize("form", ["attention", "hybrid", "depth"])
@pytest.mark.parametrize("steps", [1, 2, 3])
def test_at_its_gates_draw_a_loop_passing_messages_computes_the_decoder_it_repeats(form, steps):
    """A decoder whose block passes messages, its merges as drawn (gates at zero), computes the decoder
    without the loop: its logits and the gradient of every parameter the decoder holds, the merges'
    gradients apart, on par with the decoder's own in float64 (check 1), whether its layers attend, mix a
    hybrid's delta nets or read Kimi K3's depth residuals, one pass or more (two a later pass traced
    alone, three a scan of them); at one pass bit for bit."""
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

    def unlooped(dtype):
        model = plain.clone(dtype=dtype)

        def run(variables, ids, cotangent):
            logits, gradients = with_gradients(lambda tree: model.apply(tree, ids))(
                {**variables, "params": _without_merges(variables["params"])}, cotangent)
            return {"logits": logits, "gradients": gradients}
        return jax.jit(run)

    assert_as_exact(merged, unlooped(jnp.float32), draw, f"{form} at zero gates, {steps} passes",
                    seeds=SEEDS["zero gates"], truth_fn=unlooped(jnp.float64))


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
    pytest.param(("scanned", {"steps": 3}), UNROLLED_SEEDS["scanned layers"], id="scanned layers")])
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


def test_a_loop_passing_messages_decodes_as_it_prefills():
    """A hybrid decoder passing messages over three passes, its gates drawn off zero, each pass's cache its
    own, decodes token by token (a prefill of four, then one at a time) what its full passes over each
    prefix compute, its logits on par with theirs in float64 (check 1)."""
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


def test_a_shared_cache_prefills_as_a_cache_a_pass_does():
    """A hybrid decoder passing messages over four passes, its gates drawn off zero, prefills with one
    cache its passes share (`shared_cache`, each pass starting from the cache the step found) what it
    prefills with a cache a pass, bit for bit: the bank crosses its scanned passes as it crosses them
    apart."""
    models = [decoder("hybrid", steps=4, shared_cache=shared) for shared in (True, False)]
    variables, ids = gated_draw(models[0], salt=0xC407)(0)
    prefilled = []
    for model in models:
        made, step = decode_steps(model)
        prefilled.append(np.asarray(step(variables, made(variables, ids.shape[0]), ids)[0]))
    np.testing.assert_array_equal(*prefilled)


def test_a_packed_attention_projects_each_site_apart_as_its_own_projections_do():
    """A decoder passing messages whose attention's projections a server packed (`qkv_proj`), each site's
    input projected apart through the packed projection (`ProjectionGroup.apart`), computes what its own
    three projections compute at drawn gates (check 1 against them in float64)."""
    from dew.nn.projections import pack_projections
    from dew.nn.protocols import ProjectionGroup

    looped = decoder("attention", steps=3)
    draw = gated_draw(looped, salt=0xC406)

    def packed(variables, ids):
        members = ("q_proj", "k_proj", "v_proj")

        def widths(index):
            held = variables["params"][f"layers_{index}"]["self_attn"]
            return tuple(held[member]["kernel"].shape[-1] for member in members)

        groups = [ProjectionGroup((f"layers_{index}", "self_attn"), "qkv_proj", members, widths(index))
                  for index in range(looped.num_layers)]
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
    lambda: decoder("depth", steps=2, exit_gate=True).init(jax.random.key(0), jnp.zeros((1, 4), jnp.int32))],
    ids=["a step norm", "implicit", "an empty block", "an empty message block", "depth blocks misaligned",
         "a message block not the depth block", "shared keys", "messages joining a loop of passes",
         "exits over depth residuals"])
def test_a_loop_whose_messages_cannot_summarize_its_blocks_is_refused(refused):
    """A loop passing messages is refused where its passes would not each enter at the block's input (a
    step norm, the θ-method), where it banks nothing (an empty block or message block), where under
    attention residuals a summary would not be a whole depth block's partial sum or a pass would exit
    through a depth state uncollapsed, where a layer's keys would cross a message block's end, and where
    it would join a loop already of several passes, changing what it computes at the merges' draw."""
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
            step.drawn(interval, index)
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
