"""A decoder's cache opens as its holders state it, tracing no forward: every mixer kind, every decoder
family the capability matrix builds, a store banked by run and the prediction depths, each opened cache
the one its decode opens and the one its decode leaves, leaf for leaf in tree, shape and dtype; and a
server's start traces only what it runs."""

import jax
import jax.numpy as jnp
import pytest
from flax import linen as nn
from model_support import attending, decoder
from test_capability_matrix import ALL, CACHE, CAUSAL, SEQ, loaded
from test_mixers import COMPRESSED, KIND_FIELDS, tiny

from dew.inference.banks import HeldBanks
from dew.nn.backbones.causal_transformer import LayerKind
from dew.nn.backbones.decoder_stack import Loop, stacked_rows
from dew.registry import mixers

ROWS, PROMPT = 2, 4

PRODUCTS = frozenset({"dot_general", "conv_general_dilated", "ragged_dot_general"})
"""The primitives a forward computes by, none of which opening a cache traces."""


def primitives(jaxpr) -> set[str]:
    """Every primitive `jaxpr` and the jaxprs it holds bind."""
    return {eqn.primitive.name for eqn in jaxpr.eqns} | {
        name for eqn in jaxpr.eqns for inner in jax.extend.core.jaxprs_in_params(eqn.params)
        for name in primitives(inner)}


def typed(cache) -> dict[str, tuple[tuple[int, ...], jnp.dtype]]:
    """Each leaf of `cache` by its path: its shape and dtype."""
    return {jax.tree_util.keystr(path): (tuple(leaf.shape), jnp.dtype(leaf.dtype))
            for path, leaf in jax.tree_util.tree_leaves_with_path(cache)}


def assert_opens_as_it_decodes(model, variables, method: str = "init_cache", decoded=None):
    """`model`'s cache opened by `method` for `ROWS` rows holds no product in its program, and is, leaf for
    leaf, the cache a decode over none opens, the one a prefill over it leaves and the one a step after
    that leaves (`decoded(variables, cache, tokens)`, the model's own cached call by default)."""
    def opened(variables):
        return model.apply(variables, ROWS, method=method, mutable=["cache"])[1]["cache"]

    def call(variables, cache, tokens):
        held = variables if cache is None else {**variables, "cache": cache}
        return model.apply(held, tokens, decode=True, mutable=["cache"])[1]["cache"]

    decoded = decoded or call
    program = jax.make_jaxpr(opened)(variables).jaxpr
    assert not primitives(program) & PRODUCTS, primitives(program) & PRODUCTS
    cache = jax.eval_shape(opened, variables)
    tokens = jnp.ones((ROWS, PROMPT), jnp.int32)
    first = jax.eval_shape(decoded, variables, None, tokens)
    prefilled = jax.eval_shape(decoded, variables, cache, tokens)
    stepped = jax.eval_shape(decoded, variables, prefilled, tokens[:, :1])
    assert typed(cache), "an opened cache holds leaves"
    for name, made in (("a decode over none", first), ("a prefill", prefilled), ("a step", stepped)):
        assert typed(made) == typed(cache), (name, typed(made), typed(cache))


V4_FIELDS, V4_OVERRIDES = KIND_FIELDS["deepseek_v4"]
KINDS = {**{kind: KIND_FIELDS.get(kind, ({}, {})) for kind in sorted(set(mixers.paths) - {"mlp"})},
         **{f"deepseek_v4 {name}": ({**V4_FIELDS, **fields}, V4_OVERRIDES)
            for name, fields in COMPRESSED.items()}}
"""Every registered mixer kind but the tokenwise one, which keeps no cache, and DeepSeek V4's compressed
layers, at the fields each needs to build tiny (`test_mixers.KIND_FIELDS`)."""


@pytest.mark.parametrize("kind", sorted(KINDS))
def test_each_mixers_cache_opens_as_its_decode_opens_and_writes_it(kind):
    """A decoder of each mixer kind opens its cache from its holders' specs, tracing no product, as its
    decode opens it over none and as its prefill and its step leave it: the spec states every leaf its
    forward keeps, at the shape and dtype its writes carry."""
    fields, overrides = KINDS[kind]
    model = tiny(mixer=mixers.build(kind.split()[0], **fields), max_seq_len=32, **overrides)
    variables = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, PROMPT), jnp.int32)))
    assert_opens_as_it_decodes(model, variables)


CACHED = tuple(family for family in ALL if {CAUSAL, CACHE} <= family.facts)
"""Every decoder family the capability matrix builds that decodes over a cache."""


@attending("reference")
@pytest.mark.parametrize("family", CACHED, ids=str)
def test_each_familys_cache_opens_as_its_decode_opens_and_writes_it(family):
    """Each cached decoder family the capability matrix builds (its looped stacks, its multimodal wrappers,
    its latent and recurrent hybrids) opens its cache as its decode opens it and as its prefill and step
    leave it."""
    held = loaded(family)
    variables = held.variables or nn.unbox(held.model.init(jax.random.key(0), jnp.ones((1, SEQ), jnp.int32)))
    assert_opens_as_it_decodes(held.model, variables)


BANKED = {"attention": ("attention", None), "looped": ("attention", Loop(steps=2)),
          "looped-shared": ("attention", Loop(steps=2, shared_cache=True)),
          "gated_delta_net": ("gated_delta_net", None), "memory": ("memory", None)}
"""A banked store's cases: a stack of attention, looped whole over its own caches or one shared, and of a
recurrent and a memory mixer, whose specs read their weights' shapes off the bank."""


@pytest.mark.usefixtures("scanned_stacks")
@pytest.mark.parametrize("case", sorted(BANKED))
def test_a_banked_stores_cache_opens_banked_as_its_decode_opens_and_writes_it(case):
    """A scanned stack over a store banked by run opens its cache banked, each run's layers one array
    under the run's name, as its decode over none opens it and as its prefill and step leave it; looped
    whole, each later pass's runs under their pass's names, but where the passes share one cache."""
    kind, loop = BANKED[case]
    fields, overrides = KINDS[kind]
    mixer = mixers.build(kind, **fields)
    plain = tiny(num_layers=4, max_seq_len=16, loop=loop, mixer=mixer, **overrides)
    variables = plain.init(jax.random.key(0), jnp.ones((1, PROMPT), jnp.int32))
    scanned = tiny(num_layers=4, max_seq_len=16, loop=loop, mixer=mixer, **overrides)
    banked = HeldBanks(variables).place(scanned)
    cache = jax.eval_shape(lambda store: scanned.apply(store, ROWS, method="init_cache", mutable=["cache"]),
                           banked)[1]["cache"]
    passes = 1 if loop is None or loop.shared_cache else loop.steps
    assert sorted(cache) == [f"layers_{4 * index}_{4 * index + 3}" for index in range(passes)], list(cache)
    assert_opens_as_it_decodes(scanned, banked)


STAGED = ("attention", "gated_delta_net", "memory", "hybrid")
"""A one-stage state's stacks: alike layers of attention, of a delta net and of a memory, and a hybrid
period of attention and a delta net, which the stage's run holds by position of its period."""


@pytest.mark.usefixtures("scanned_stacks")
@pytest.mark.parametrize("case", STAGED)
def test_a_stage_local_stores_cache_opens_a_layer_at_a_time_as_its_decode_opens_and_writes_it(case):
    """A scanned stack's one-stage state (`stage_sites`, `StackView.stage_form`), its runs stacked by stage
    under `stages` and its layers' own scopes empty, opens its cache a layer at a time as its decode over
    none opens it (the stack reads the store off its stages, `off_stages`) and as its prefill and step
    leave it: each layer's spec read where its stage's run holds it, a periodic run's at its position."""
    kind = "gated_delta_net" if case == "hybrid" else case
    fields, overrides = KINDS[kind]
    layer = {"mixer": mixers.build(kind, **fields), **overrides}
    layout = (layer if case != "hybrid" else
              {"layer_types": ("full_attention", "linear") * 2, "kinds": {"linear": LayerKind(**layer)}})
    scanned = tiny(num_layers=4, max_seq_len=16, **layout)
    params = nn.unbox(scanned.init(jax.random.key(0), jnp.ones((1, PROMPT), jnp.int32)))["params"]
    (site,) = scanned.stage_sites({"params": params}, 1)
    staged = {"params": site.view.stage_form("params", dict(params), stacked_rows)}
    held = staged["params"]
    assert "stages" in held and not any(f"layers_{index}" in held for index in range(4)), list(held)
    cache = jax.eval_shape(lambda store: scanned.apply(store, ROWS, method="init_cache", mutable=["cache"]),
                           staged)[1]["cache"]
    assert {f"layers_{index}" for index in range(4)} <= set(cache), list(cache)
    assert_opens_as_it_decodes(scanned, staged)


@pytest.mark.parametrize("kind", ["attention", "mla", "recurrent"])
def test_prediction_depths_open_their_caches_as_their_steps_write_them(kind):
    """A decoder's prediction depths open their caches apart from the trunk's (`init_mtp_cache`), tracing no
    product, as a depth's step over none opens them and as its steps leave them."""
    model = decoder(kind).clone(num_nextn_predict_layers=1, max_seq_len=16)
    variables = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, PROMPT), jnp.int32)))

    def stepped(variables, cache, tokens):
        held = variables if cache is None else {**variables, "cache": cache}
        hidden = jnp.zeros((*tokens.shape, model.emb_features), jnp.float32)
        return model.apply(held, hidden, tokens, depth=0, decode=True, mutable=["cache"],
                           method="mtp_step")[1]["cache"]

    assert_opens_as_it_decodes(model, variables, method="init_mtp_cache", decoded=stepped)


def test_a_servers_free_slots_open_with_no_forward_traced():
    """The program a server compiles and runs to open its free slots (`serving_kernel.opened`) holds no
    product, and traces none: their cache opens as the model states it, and the rest of their state takes
    the types its admissions' prefill leaves (`StateTypes`, traced once at an admission the server runs,
    `Server._admitted_types`), where each trace of the program traced a blank prompt's prefill for them
    within it."""
    from jax.experimental import checkify

    from dew.inference.serving_kernel import StateTypes, opened
    from dew.nn.inputs import ModelInputs
    from dew.nn.kernels import generation
    from dew.sampling.text import decode_ops, prefill_state

    model, slots = decoder().clone(max_seq_len=32), 4
    variables = nn.unbox(model.init(jax.random.key(0), jnp.ones((1, PROMPT), jnp.int32)))
    prompts = ModelInputs(jnp.ones((ROWS, PROMPT), jnp.int32))

    def prefilled(variables):
        return prefill_state(model, variables, prompts, decode_ops(model, variables, 0, None))[0]

    _, state = jax.eval_shape(checkify.checkify(prefilled, errors=checkify.user_checks), variables)
    with generation.routes() as routes:
        program = jax.make_jaxpr(lambda variables: opened(model, variables, 0, slots, 32,
                                                          types=StateTypes.of(state)))(variables).jaxpr
    assert not primitives(program) & PRODUCTS, primitives(program) & PRODUCTS
    # No forward is traced within it, abstractly or not: no layer selects a kernel.
    assert not routes, routes
