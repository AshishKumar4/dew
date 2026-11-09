"""A looped stack (`Loop`): passes that share the layers' parameters, each
with its own entries in the decode cache. A loop computes the stack it
unrolls to as that stack does, to the bit; decoding, scanned runs and
checkpointed passes, which compute it another way, are on par with it from
float64 over drawn models and tokens (check 1)."""

import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import linen as nn
from model_support import TINY_DECODER, as_stepped, decode_steps, model_draw, with_logit_cotangent
from reference_error import assert_as_exact, with_gradients

from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.backbones.decoder_stack import Loop

VOCAB = 37


def tiny(**overrides):
    return CausalTransformer(**{"vocab_size": VOCAB, **TINY_DECODER, "attention_impl": "reference",
                                "tie_embeddings": False, "loop": Loop(3, exit_gate=True), **overrides})


def tokens(rng, length=12, rows=2):
    return jax.random.randint(rng, (rows, length), 0, VOCAB)


SEEDS = {'decode': 64, 'scanned': 64, 'checkpointed': 512, 'block decode': 32}
"""Check 1's declared seeds for each site of the file, sized by `seeds_for` (on armada):
the most any of a site's cases took."""


def drawn(model, length=12, rows=2):
    """Check 1's draw for `model`: a seed's parameters and `rows` rows of `length` tokens."""
    return model_draw(model, (rows, length), VOCAB)


def wide(model):
    """`model` in float64, a function of its parameters and tokens: the truth of a site that compares
    two ways of computing it."""
    return jax.jit(model.clone(dtype=jnp.float64).apply)


def decoded(model, params, ids, prompt=4):
    """Prefill `prompt` tokens, then decode the rest one at a time: each
    position's logits and the cache after the last."""
    made, step = decode_steps(model)
    logits, cache = step(params, made(params, ids.shape[0]), ids[:, :prompt])
    steps = [logits]
    for position in range(prompt, ids.shape[1]):
        logits, cache = step(params, cache, ids[:, position:position + 1])
        steps.append(logits)
    return jnp.concatenate(steps, axis=1), cache


def passes(model, prompt=4):
    """`model`'s full passes over a draw's ids at the shapes `decoded` computes them (`as_stepped`): the
    prompt's from one pass over it, each later position's from one over the ids up to it."""
    forward = jax.jit(model.apply)
    return lambda params, ids: as_stepped(lambda held: forward(params, held), ids, prompt)


def unrolled(params, steps):
    """The parameters of a stack `steps` times as deep whose layer i is layer i mod depth."""
    depth = TINY_DECODER["num_layers"]
    tree = dict(params['params'])
    for index in range(depth, steps * depth):
        tree[f'layers_{index}'] = tree[f'layers_{index % depth}']
    return {'params': tree}


def test_a_loop_without_its_norm_is_the_stack_repeated_cache_and_all(rng):
    """Two passes of two layers compute four layers whose last two repeat the
    first two, and their decode cache holds the same entries under the same names."""
    looped = tiny(loop=Loop(2, step_norm=False))
    deep = tiny(num_layers=4, loop=None)
    ids = tokens(rng)
    params = nn.unbox(looped.init(rng, ids))
    np.testing.assert_array_equal(looped.apply(params, ids), deep.apply(unrolled(params, 2), ids))
    ours, cache = decoded(looped, params, ids)
    theirs, deep_cache = decoded(deep, unrolled(params, 2), ids)
    np.testing.assert_array_equal(ours, theirs)
    assert jax.tree.structure(cache) == jax.tree.structure(deep_cache)
    jax.tree.map(np.testing.assert_array_equal, cache, deep_cache)


@pytest.mark.parametrize("scan_layers", [False, True])
def test_decoding_a_loop_matches_its_prefill(rng, scan_layers):
    """A loop decoded a token at a time is on par with its prefills, each step's at the width it reads
    (check 1), and its cache holds each pass's layers."""
    model = tiny(scan_layers=scan_layers)
    assert_as_exact(lambda params, ids: decoded(model, params, ids)[0], passes(model),
                    drawn(model), f'{scan_layers} loop decode', seeds=SEEDS['decode'], truth_fn=wide(model))
    ids = tokens(rng)
    _, cache = decoded(model, nn.unbox(model.init(rng, ids)), ids)
    assert sorted(cache) == [f'layers_{index}' for index in range(6)]


def test_scanned_runs_loop_as_the_plain_layers_do():
    """A loop's layers scanned are on par with them run one by one (check 1)."""
    assert_as_exact(jax.jit(tiny(scan_layers=True).apply), jax.jit(tiny().apply), drawn(tiny()),
                    'scanned loop', seeds=SEEDS['scanned'], truth_fn=wide(tiny()))


def test_backprop_steps_train_through_the_last_passes_alone(rng):
    ids = tokens(rng)
    params = nn.unbox(tiny().init(rng, ids))

    def gradient(backprop_steps):
        model = tiny(loop=Loop(3, backprop_steps=backprop_steps))

        def loss(tree):
            return jnp.mean(model.apply(tree, ids) ** 2)

        return jax.value_and_grad(loss)({'params': {name: value for name, value in params['params'].items()
                                                    if name != 'early_exit_gate'}})

    (whole, every), (cut, last) = gradient(None), gradient(1)
    assert whole == cut
    assert np.any(every['params']['embed_tokens']['embedding'])
    assert not np.any(last['params']['embed_tokens']['embedding'])
    assert np.any(last['params']['layers_0']['mlp']['down_proj']['kernel'])


@pytest.mark.parametrize("backprop_steps", [None, 1])
@pytest.mark.parametrize("scan_layers", [False, True])
def test_checkpointed_passes_preserve_logits_and_gradients(backprop_steps, scan_layers):
    """Three passes over layers [1, 3) of four, checkpointed, keep the fp32
    logits and gradients of the plain passes on par (check 1), including the
    cut before the last pass: a pass recomputed instead of retained may fuse
    its backward another way.
    """
    loop = Loop(3, layers=(1, 3), backprop_steps=backprop_steps, remat=False)
    plain = tiny(num_layers=4, loop=loop, scan_layers=scan_layers, dtype="float32")
    checkpointed = plain.clone(loop=dataclasses.replace(loop, remat=True))

    def result(model):
        def run(tree, ids, cotangent):
            pulled = with_gradients(lambda tree: model.apply(tree, ids, train=True))
            logits, gradients = pulled(tree, cotangent)
            return {'logits': logits, 'gradients': gradients}
        return jax.jit(run)

    # Eight rows of sixteen tokens: a norm's scale sums its gradient over enough of them to resolve.
    draw = with_logit_cotangent(drawn(plain, 16, 8), VOCAB, 0x100)
    assert_as_exact(result(checkpointed), result(plain), draw,
                    f'{scan_layers} {backprop_steps} checkpointed', seeds=SEEDS['checkpointed'],
                    truth_fn=result(plain.clone(dtype=jnp.float64)))
    gradients = result(checkpointed)(*draw(0))['gradients']['params']
    assert bool(jnp.any(gradients['embed_tokens']['embedding'])) == (backprop_steps is None)
    assert bool(jnp.any(gradients['layers_1']['mlp']['down_proj']['kernel']))


@pytest.mark.parametrize("scan_layers", [False, True])
def test_checkpointed_passes_leave_decode_and_its_collections_unchanged(rng, scan_layers):
    """Decode bypasses the pass checkpoints and retains each pass's cache
    names and values, under both the plain stack and its scanned runs.
    """
    loop = Loop(3, layers=(1, 3), remat=False)
    plain = tiny(num_layers=4, loop=loop, scan_layers=scan_layers)
    checkpointed = plain.clone(loop=dataclasses.replace(loop, remat=True))
    ids = tokens(rng)
    params = nn.unbox(plain.init(rng, ids))
    expected, actual = decoded(plain, params, ids), decoded(checkpointed, params, ids)
    jax.tree.map(np.testing.assert_array_equal, actual, expected)
    assert sorted(actual[1]) == [f'layers_{index}' for index in range(8)]
    boundaries = []
    for model in (plain, checkpointed):
        program = jax.make_jaxpr(lambda tree, model=model: model.apply(
            tree, ids[:, :1], decode=True, mutable=['cache']))(params).jaxpr
        boundaries.append(sum(equation.primitive.name == "remat2" for equation in program.eqns))
    assert boundaries[0] == boundaries[1]


def test_a_loop_over_a_block_is_the_stack_with_the_block_repeated_cache_and_all(rng):
    """Layers 1 and 2 of four, run twice with no norm between, compute six
    layers whose fourth and fifth repeat the second and third, to the bit,
    and keep the second pass's entries after the first's; decoded, and
    scanned, they are on par with their prefill (check 1)."""
    looped = tiny(num_layers=4, loop=Loop(2, step_norm=False, layers=(1, 3)))
    ids = tokens(rng)
    params = nn.unbox(looped.init(rng, ids))
    tree = dict(params['params'])
    shared = {name: value for name, value in tree.items() if not name.startswith('layers_')}
    order = [0, 1, 2, 1, 2, 3]
    repeated = {f'layers_{index}': tree[f'layers_{source}'] for index, source in enumerate(order)}
    deep_params = {'params': {**shared, **repeated}}
    deep = tiny(num_layers=6, loop=None)
    np.testing.assert_array_equal(looped.apply(params, ids), deep.apply(deep_params, ids))
    ours, cache = decoded(looped, params, ids)
    theirs, deep_cache = decoded(deep, deep_params, ids)
    np.testing.assert_array_equal(ours, theirs)
    # The second pass's block sits after the decoder's own layers: deep layers 3, 4 at stored 4, 5.
    for stored, deep_index in ((0, 0), (1, 1), (2, 2), (4, 3), (5, 4), (3, 5)):
        jax.tree.map(np.testing.assert_array_equal, cache[f'layers_{stored}'],
                     deep_cache[f'layers_{deep_index}'])
    assert_as_exact(lambda params, ids: decoded(looped, params, ids)[0], passes(looped), drawn(looped),
                    'block loop decode', seeds=SEEDS['block decode'], truth_fn=wide(looped))
    scanned = tiny(num_layers=4, loop=Loop(2, step_norm=False, layers=(1, 3)), scan_layers=True)
    assert_as_exact(jax.jit(scanned.apply), jax.jit(looped.apply), drawn(looped), 'scanned block loop',
                    seeds=SEEDS['scanned'], truth_fn=wide(looped))


def test_a_decoder_gains_a_loop_over_its_own_block_that_one_pass_leaves_as_it_was(rng):
    plain = tiny(num_layers=4, loop=None)
    ids = tokens(rng)
    params = nn.unbox(plain.init(rng, ids))
    once, twice = plain.with_passes(1, (1, 3)), plain.with_passes(2, (1, 3))
    assert twice.loop == Loop(2, step_norm=False, layers=(1, 3))
    np.testing.assert_array_equal(once.apply(params, ids), plain.apply(params, ids))
    assert not np.array_equal(twice.apply(params, ids), plain.apply(params, ids))
    assert plain.with_passes(2) is None and plain.with_passes(None, (1, 3)) is None
    assert twice.with_passes(4).loop == Loop(4, step_norm=False, layers=(1, 3))


DELTA_NETS = {'linear_attention': {'mixer': {'class': 'gated_delta_net', 'fields': {
    'linear_num_key_heads': 2, 'linear_num_value_heads': 4, 'linear_key_head_dim': 6,
    'linear_value_head_dim': 8, 'linear_conv_kernel_dim': 4}}}}
"""A stack of gated delta nets, whose decode state is a recurrent memory and a convolution tail."""


def test_each_pass_of_a_shared_cache_step_starts_from_the_cache_the_step_found(rng):
    """Every pass of a `Loop(shared_cache=True)` step starts from the cache as the step found it
    (`SharedPassView`), its recurrent memory and convolution tail with its cursors: allocating the
    cache leaves every entry zero, and a stack of gated delta nets prefills, bit for bit, what the same
    loop with a cache of its own per pass prefills, where the second pass took the prompt again over
    the state the first pass left (audit NN-A1)."""
    models = [tiny(loop=Loop(2, shared_cache=shared), layer_types=('linear_attention',) * 2, kinds=DELTA_NETS)
              for shared in (True, False)]
    ids = tokens(rng)
    params = nn.unbox(models[0].init(rng, ids))
    prefilled = []
    for model in models:
        made, step = decode_steps(model)
        cache = made(params, ids.shape[0])
        assert all(not jnp.any(leaf) for leaf in jax.tree.leaves(cache))
        prefilled.append(step(params, cache, ids)[0])
    np.testing.assert_array_equal(*prefilled)


QWEN35 = Path(__file__).with_name("fixtures") / "hf" / "qwen35-tiny"
"""Qwen 3.5's tiny release: three gated delta nets and one attention layer."""

IMPLICIT_SEEDS = {2: 8, 3: 8, 4: 8}
"""Check 1's declared seeds for an implicit loop at zero θ against the decoder it loops, by its pass count,
sized by `seeds_for` on armada."""

THETA_SEEDS = {"two passes": 8, "three passes": 8, "three passes, the last trained": 8}
"""Check 1's declared seeds for an implicit loop at drawn θ against the θ-method unrolled here, sized by
`seeds_for` on armada."""


def _qwen35(steps: int | None = None, **loop):
    """Qwen 3.5's tiny release as Dew builds it (`translate_config`), its block [1, 4) run `steps` times
    with each later pass by the θ-method (`Loop.implicit`), as `dew.config.looping` gives a loaded
    decoder; the decoder alone where `steps` is None."""
    from dew.interop.hf_decoders import translate_config

    plain = CausalTransformer(**{**translate_config(json.loads((QWEN35 / "config.json").read_text())),
                                 "dtype": "float32", "attention_impl": "reference"})
    return plain if steps is None else plain.with_passes(steps, (1, 4), implicit=True, **loop)


def _results(forward):
    """`forward`'s logits and the gradient of every parameter under a cotangent of them, jitted."""
    def run(variables, ids, cotangent):
        pulled = with_gradients(lambda variables: forward(variables, ids))
        logits, gradients = pulled(variables, cotangent)
        return {"logits": logits, "gradients": gradients}
    return jax.jit(run)


@pytest.mark.parametrize("steps", [2, 3, 4])
def test_an_implicit_loop_at_zero_theta_computes_the_decoder_it_loops(steps):
    """Qwen 3.5 with a block of its own run `steps` times, each later pass at `a + θ⊙(h - a)` leaving
    `block(x) + (a - x)` (`Loop.implicit`), its θ as the loop draws it, at zero: its logits, and the
    gradient of every parameter the decoder has, on par with the decoder's own, and θ's, which the
    decoder has none of, on par with the θ-method unrolled at zero (`_theta_method`; check 1, against
    both in float64), over drawn parameters and tokens, whatever the count, two passes one later pass
    traced alone and more a scan of them; and θ's own gradient nonzero, so a step moves it off the
    decoder it started as."""
    plain, looped = _qwen35(), _qwen35(steps)
    vocab = plain.vocab_size
    # Eight rows of sixteen tokens: a norm's scale sums its gradient over enough of them to resolve.
    draw = with_logit_cotangent(model_draw(looped, (8, 16), vocab), vocab, 0x7E7A + steps)

    def decoder(dtype):
        """The decoder's logits and its parameters' gradients, θ's from the θ-method unrolled."""
        unlooped, unrolled = plain.clone(dtype=dtype), looped.clone(dtype=dtype)

        def run(variables, ids, cotangent):
            held = {name: value for name, value in variables["params"].items() if name != "loop_theta"}
            logits, gradients = with_gradients(lambda tree: unlooped.apply(tree, ids))(
                {**variables, "params": held}, cotangent)
            _, through = with_gradients(lambda tree: unrolled.apply(tree, ids, method=_theta_method))(
                variables, cotangent)
            theta = through["params"]["loop_theta"]
            return {"logits": logits,
                    "gradients": {**gradients, "params": {**gradients["params"], "loop_theta": theta}}}
        return jax.jit(run)

    assert_as_exact(_results(looped.apply), decoder(jnp.float32), draw, f"an implicit loop of {steps}",
                    seeds=IMPLICIT_SEEDS[steps], truth_fn=decoder(jnp.float64))
    variables, ids, cotangent = draw(0)
    assert not np.any(np.asarray(variables["params"]["loop_theta"]))
    _, pull = jax.vjp(lambda params: looped.apply({"params": params}, ids), variables["params"])
    assert np.any(np.asarray(pull(cotangent)[0]["loop_theta"]) != 0)


def _theta_method(module, ids):
    """`module`'s logits with its block taken by the θ-method pass by pass, unrolled here from the stack's
    own spans: `h = block(a)`, then each later pass `z = a + θ⊙(h - a)` and `h = block(z) + (a - z)`, the
    state entering the pass `backprop_steps` cuts held from the gradient."""
    from dew.nn.backbones.call_metadata import call_metadata

    loop = module.loop
    first, end = loop.layers
    inputs = {"train": False, "decode": False, "per_layer_input": None,
              "attention_metadata": call_metadata(module, ids, *(False, *(None,) * 9))}
    x = module._prepared(module.scaled_embeddings(module.token_embeddings(ids)), None)
    a = module.stack(x, span=(0, first), **inputs)
    h = module.stack(a, span=(first, end), **inputs)
    cut = None if loop.backprop_steps is None else loop.steps - loop.backprop_steps
    for step in range(1, loop.steps):
        h = jax.lax.stop_gradient(h) if step == cut else h
        z = a + module.loop_theta * (h - a)
        h = module.stack(z, span=(first, end), **inputs) + (a - z)
    return module._logits(module.norm(module.stack(h, span=(end, module.num_layers), **inputs)))


@pytest.mark.parametrize("name, steps, loop", [("two passes", 2, {}), ("three passes", 3, {}),
                                               ("three passes, the last trained", 3, {"backprop_steps": 1})],
                         ids=["two passes", "three passes", "the last trained"])
def test_an_implicit_loop_at_drawn_theta_takes_the_theta_method(name, steps, loop):
    """Qwen 3.5 with an implicit loop over its block at a drawn θ: its logits, and the gradient of every
    parameter, θ's and the prelude's among them, on par with the θ-method unrolled pass by pass from the
    stack's own spans (`_theta_method`, check 1 against it in float64), its later passes traced alone (two
    passes) or scanned (three), the last alone trained where `backprop_steps` says."""
    looped = _qwen35(steps, **loop)
    vocab = looped.vocab_size
    drawn = with_logit_cotangent(model_draw(looped, (8, 16), vocab), vocab, 0x7E7B + steps)

    def draw(seed):
        variables, ids, cotangent = drawn(seed)
        theta = np.random.default_rng([0x7E7C, seed]).uniform(0.1, 0.9, looped.emb_features)
        params = {**variables["params"], "loop_theta": jnp.asarray(theta, jnp.float32)}
        return {**variables, "params": params}, ids, cotangent

    def unrolled(model):
        return lambda variables, ids: model.apply(variables, ids, method=_theta_method)

    assert_as_exact(_results(looped.apply), _results(unrolled(looped)), draw, name, seeds=THETA_SEEDS[name],
                    truth_fn=_results(unrolled(looped.clone(dtype=jnp.float64))))


def test_a_loaded_decoder_gains_an_implicit_loop_at_zero_theta():
    """A loaded Qwen 3.5 given an implicit loop over its own block, as a run asks for one
    (`dew.config.looping`), draws the loop's θ at zero beside the parameters it loaded, among what it drew
    fresh (`Pretrained.with_model`), so it runs and starts as the decoder it loaded."""
    from dew.config import looping
    from dew.interop.pretrained import Pretrained

    loaded = Pretrained.load(str(QWEN35), dtype="float32", attention_impl="reference")
    looped = looping(loaded, 3, (1, 4), implicit=True)
    theta = looped.variables["params"]["loop_theta"]
    assert looped.model.loop == Loop(3, step_norm=False, layers=(1, 4), implicit=True)
    assert np.shape(theta) == (loaded.model.emb_features,) and not np.any(np.asarray(theta))
    assert set(looped.variables["params"]) - set(loaded.variables["params"]) == {"loop_theta"}
    assert "loop_theta" in looped.fresh
    ids = jnp.arange(12, dtype=jnp.int32).reshape(2, 6)
    assert np.all(np.isfinite(np.asarray(looped.model.apply(looped.variables, ids))))


EXIT_SEEDS = {"whole stack": 128, "a block before a coda": 256, "a block before a coda with a z-loss": 256,
              "a block before a coda exiting at a threshold": 256, "the last pass trained": 256,
              "a saturated gate": 128}
"""Check 1's declared seeds for the exit objective against its formula, sized by `seeds_for` on armada."""


def _unrolled_passes(module, ids):
    """`module`'s logits and exit-gate logits at each pass of its loop, unrolled here from the stack's own
    spans: the block run `steps` times on its own output, the coda run on each pass's output, then the
    final norm, the gate and the head. The passes before the cut (`backprop_steps`) leave for their exits
    held from the gradient, as the state entering the pass it cuts is."""
    from dew.nn.backbones.call_metadata import call_metadata

    loop = module.loop
    first, end = loop.layers
    inputs = {"train": False, "decode": False, "per_layer_input": None,
              "attention_metadata": call_metadata(module, ids, *(False, *(None,) * 9))}
    x = module._prepared(module.scaled_embeddings(module.token_embeddings(ids)), None)
    h = module.stack(module.stack(x, span=(0, first), **inputs), span=(first, end), **inputs)
    cut = None if loop.backprop_steps is None else loop.steps - loop.backprop_steps
    logits, gates = [], []
    for step in range(loop.steps):
        held = jax.lax.stop_gradient(h) if cut is not None and step < cut else h
        out = module.norm(held if end == module.num_layers else
                          module.stack(held, span=(end, module.num_layers), **inputs))
        logits.append(module._logits(out))
        gates.append(module.early_exit_gate(out)[..., 0])
        if step + 1 < loop.steps:
            h = module.stack(jax.lax.stop_gradient(h) if step + 1 == cut else h, span=(first, end), **inputs)
    return logits, gates


@pytest.mark.parametrize("name, layers, loop, z_loss, shift", [
    ("whole stack", (0, 4), {}, 0.0, 0.0), ("a block before a coda", (1, 3), {}, 0.0, 0.0),
    ("a block before a coda with a z-loss", (1, 3), {}, 1e-2, 0.0),
    ("a block before a coda exiting at a threshold", (1, 3), {"exit_threshold": 0.5}, 0.0, 0.0),
    ("the last pass trained", (1, 3), {"backprop_steps": 1}, 0.0, 0.0),
    ("a saturated gate", (1, 3), {}, 0.0, 30.0)],
    ids=["whole stack", "coda", "z-loss", "threshold", "the last trained", "saturated"])
def test_the_exit_objective_is_the_expected_cross_entropy_less_its_entropy(name, layers, loop, z_loss, shift):
    """A two-pass Qwen 3.5 with an exit gate (`Loop.exit_gate`, its block the whole stack or one before a
    coda, each pass's coda its own) trained under Ouro's objective (`LMObjective.exit_entropy`): its loss
    and every parameter's gradient, the gate's and the passes' states' through it, on par with the formula
    over the passes unrolled here (`_unrolled_passes`): each pass's cross entropy through the head,
    weighted by the probability it exits there, less β times that distribution's entropy, plus the last
    pass's z-loss where one is asked, by check 1 against the formula in float64. A threshold chooses no
    pass the objective scores, the last pass scored as it is; the passes the cut holds back pass no
    gradient back through their exits; and a gate whose hazard rounds to one in float32 (its bias moved
    by `shift`) leaves every gradient finite."""
    from dew.batch import TEXT_KEY
    from dew.objectives.base import Step
    from dew.objectives.lm import LMObjective

    beta, length = 0.25, 12
    looped = _qwen35().with_passes(2, layers, exit_gate=True, **loop)
    objective = LMObjective(looped, length, exit_entropy=beta, z_loss=z_loss, token_accuracy=False)
    step = Step(step=jnp.asarray(0), key=jax.random.key(0), ema=None)
    drawn = model_draw(looped, (4, length + 1), looped.vocab_size)

    def draw(seed):
        variables, ids = drawn(seed)
        gate = np.random.default_rng([0xE817, seed]).normal(0, 0.5, (looped.emb_features + 1,))
        drawn_gate = {"kernel": jnp.asarray(gate[:-1, None], jnp.float32),
                      "bias": jnp.asarray(gate[-1:] + shift, jnp.float32)}
        params = {**variables["params"], "early_exit_gate": drawn_gate}
        return {**variables, "params": params}, ids

    def ours(variables, ids):
        return jax.value_and_grad(lambda v: objective.loss(v, {TEXT_KEY: ids}, step)[0].mean()[0])(variables)

    def formula(model):
        def loss(variables, ids):
            passes, gates = model.apply(variables, ids[:, :-1], method=_unrolled_passes)
            targets = ids[:, 1:]
            scored = [jnp.take_along_axis(jax.nn.log_softmax(each, -1), targets[..., None], -1)[..., 0]
                      for each in passes]
            log_chances = jnp.stack([jax.nn.log_sigmoid(gates[0]), jax.nn.log_sigmoid(-gates[0])])
            chances = jnp.exp(log_chances)
            term = -jnp.sum(chances * jnp.stack(scored), 0) + beta * jnp.sum(chances * log_chances, 0)
            return jnp.mean(term + z_loss * jnp.square(jax.nn.logsumexp(passes[-1], -1)))
        return lambda variables, ids: jax.value_and_grad(lambda v: loss(v, ids))(variables)

    assert_as_exact(jax.jit(ours), jax.jit(formula(looped)), draw, f"the exit objective over {name}",
                    seeds=EXIT_SEEDS[name], truth_fn=jax.jit(formula(looped.clone(dtype=jnp.float64))))


EXITING_SEEDS = {"forward": 8, "decode": 8, "shared": 8}
"""Check 1's declared seeds for a loop that exits each token at its threshold, sized by `seeds_for` on
armada."""


def _exiting(**loop):
    """Qwen 3.5's tiny decoder looping its block [1, 3) three times before its coda, a gate weighing each
    pass's exit through its own coda, each token leaving at the first pass whose cumulative exit
    probability reaches 0.6, the loop's other fields as `loop` states them."""
    looped = _qwen35().with_passes(3, (1, 3), exit_gate=True, exit_threshold=0.6)
    return looped.clone(loop=dataclasses.replace(looped.loop, **loop))


def _with_gate(draw, width: int, seed_salt: int):
    """`draw`'s parameters with an exit gate drawn apart from its zeros, so tokens exit at different
    passes."""
    def drawn(seed):
        variables, *rest = draw(seed)
        gate = np.random.default_rng([seed_salt, seed]).normal(0, 1.0, (width + 1,))
        params = {**variables["params"], "early_exit_gate": {
            "kernel": jnp.asarray(gate[:-1, None], jnp.float32), "bias": jnp.asarray(gate[-1:], jnp.float32)}}
        return {**variables, "params": params}, *rest
    return drawn


def _passes_exiting(module, ids):
    """`module`'s logits with its block's passes and each pass's coda unrolled here from the stack's own
    spans, each token's state its exiting pass's: the first whose cumulative exit probability, each
    pass's hazard `sigmoid(gate(norm(coda(h))))` times the chance no pass before it exited, reaches the
    threshold, the last pass's where none does."""
    from dew.nn.backbones.call_metadata import call_metadata

    loop = module.loop
    first, end = loop.layers
    inputs = {"train": False, "decode": False, "per_layer_input": None,
              "attention_metadata": call_metadata(module, ids, *(False, *(None,) * 9))}
    x = module._prepared(module.scaled_embeddings(module.token_embeddings(ids)), None)
    h = module.stack(module.stack(x, span=(0, first), **inputs), span=(first, end), **inputs)
    leaving, kept, chosen = [], 1.0, None
    for _ in range(loop.steps - 1):
        y = module.stack(h, span=(end, module.num_layers), **inputs)
        hazard = jax.nn.sigmoid(module.early_exit_gate(module.norm(y))[..., 0])
        reached = 1 - kept * (1 - hazard) >= loop.exit_threshold
        chosen = reached if chosen is None else chosen | reached
        leaving.append((y, reached))
        kept = kept * (1 - hazard)
        h = module.stack(h, span=(first, end), **inputs)
    out = module.stack(h, span=(end, module.num_layers), **inputs)
    for y, reached in reversed(leaving):
        out = jnp.where(reached[..., None], y, out)
    return module._logits(module.norm(out))


def test_a_loop_exits_each_token_at_its_threshold_through_each_passs_coda():
    """Qwen 3.5 looping a block before its coda, each pass weighed by its exit gate through its own coda:
    its logits, each token at the pass its threshold exits at, on par with the passes and codas unrolled
    here from the stack's own spans (`_passes_exiting`, check 1 against it in float64)."""
    looped = _exiting()
    draw = _with_gate(model_draw(looped, (8, 16), looped.vocab_size), looped.emb_features, 0xE81)

    def unrolled(model):
        return jax.jit(lambda variables, ids: model.apply(variables, ids, method=_passes_exiting))

    assert_as_exact(jax.jit(looped.apply), unrolled(looped), draw, "a loop exiting at its threshold",
                    seeds=EXITING_SEEDS["forward"], truth_fn=unrolled(looped.clone(dtype=jnp.float64)))


def test_decoding_a_loop_that_exits_through_each_passs_coda_matches_its_prefill():
    """The loop exiting at its threshold through each pass's coda, decoded a token at a time, is on par
    with its prefills (check 1), its cache holding each pass's block and each earlier pass's coda beside
    the layers' own: Qwen 3.5's four layers, the block's two later passes over its two layers and the
    coda's two earlier passes over its one."""
    looped = _exiting()
    draw = _with_gate(model_draw(looped, (2, 12), looped.vocab_size), looped.emb_features, 0xE82)
    assert_as_exact(lambda params, ids: decoded(looped, params, ids)[0], passes(looped), draw,
                    "an exiting loop's decode", seeds=EXITING_SEEDS["decode"],
                    truth_fn=passes(looped.clone(dtype=jnp.float64)))
    variables, ids = draw(0)
    _, cache = decoded(looped, variables, ids)
    assert sorted(cache) == sorted(f"layers_{index}" for index in range(4 + 2 * 2 + 2 * 1))


def test_a_shared_cache_keeps_each_passs_coda_entries_as_the_coda_wrote_them():
    """The exiting loop with a cache its passes share (`Loop.shared_cache`), prefilled into an allocated
    cache, computes what the loop with a cache of its own per pass does there, its logits and each
    earlier pass's coda entries alike (check 1): the block's passes in between leave the coda's entries
    as it wrote them (`SharedPassView`), at slots after the layers' own, the block's passes keeping
    none apart."""
    own, shared = _exiting(), _exiting(shared_cache=True)
    draw = _with_gate(model_draw(own, (2, 12), own.vocab_size), own.emb_features, 0xE83)

    def prefilled(model, slots):
        made, step = decode_steps(model)

        def run(params, ids):
            logits, cache = step(params, made(params, ids.shape[0]), ids)
            return {"logits": logits, "codas": [cache[f"layers_{slot}"] for slot in slots]}
        return run

    assert_as_exact(prefilled(shared, (4, 5)), prefilled(own, (8, 9)), draw,
                    "a shared cache's coda entries", seeds=EXITING_SEEDS["shared"],
                    truth_fn=prefilled(own.clone(dtype=jnp.float64), (8, 9)))
