"""Every model that climbs a remat ladder steps at its rungs as it steps without them.

A model that recomputes (`dew.nn.protocols.Recomputing`) climbs a ladder of rungs (`recompute_more`), each
keeping fewer of its forward's residuals for the backward than the rung below it, and a rung's record
restores it (`recompute_record`, `restore_recompute`). These tests hold every rung to the model: each keeps
fewer residual bytes, its record restores it from any rung below and moves none above, and its top rung's
forward, and for one member of each architecture class its gradients, hold check 1 against the model's from
the model's float64 twin, a rung's checkpoints making it another program, which a GPU rounds apart. The
models are the registry-wide checkpoints (tests/checkpoint_cases.py's `PATHS`), every architecture
test_architectures.py trains that climbs a ladder, and a decoder with a continuum level scanned at its top
rung."""

import dataclasses
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from architecture_cases import CASES as ARCHITECTURES, LM, SEQ_LEN, TEXT_FEATURES, TEXT_TOKENS, VOCAB, Case
from checkpoint_cases import LENGTH, PATHS, ROWS, loaded, reads
from flax.traverse_util import flatten_dict, unflatten_dict
from model_support import float64_twin
from reference_error import assert_as_exact, widened
from tool_support import load

from dew.nn.attention_residuals import AttentionResiduals
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.dit import TextContext
from dew.nn.protocols import Recomputing
from dew.registry import models


def _call(model, variables, inputs, **apply) -> jax.Array:
    """A forward without dropout, whose float64 twin would draw other masks:
    a token model's logits, a denoiser's output."""
    return model.apply(variables, *inputs, **apply)


def _saved(model, variables, inputs, **apply) -> int:
    """The device bytes `model`'s forward keeps for its backward: the residuals its linearization holds
    (`jax.linearize`) in device memory, traced abstractly. A remat rung recomputes what it does not keep
    and an offloading one holds it on the host, so a stronger rung keeps fewer."""
    def forward(params):
        return _call(model, {**variables, "params": params}, inputs, **apply)

    kept = jax.eval_shape(lambda params: jax.linearize(forward, params)[1], variables["params"])
    return sum(leaf.size * leaf.dtype.itemsize for leaf in jax.tree.leaves(kept)
               if jax.typeof(leaf).memory_space == jax.memory.Space.Device)


def _stepped(model, variables, inputs, **apply) -> dict:
    """A step's gradient in every floating parameter, along a cotangent drawn at its output's shape."""
    output = jax.eval_shape(lambda: _call(model, variables, inputs, **apply))
    cotangent = jax.random.normal(jax.random.key(1), output.shape, jnp.float32).astype(output.dtype)
    flat = flatten_dict(variables["params"])
    floats = {path: leaf for path, leaf in flat.items() if jnp.issubdtype(leaf.dtype, jnp.inexact)}
    held = {path: leaf for path, leaf in flat.items() if path not in floats}

    def loss(trained):
        params = unflatten_dict({**held, **trained})
        return jnp.sum(_call(model, {**variables, "params": params}, inputs, **apply) * cotangent)

    return jax.grad(loss)(floats)


def _rungs(model) -> list:
    """The model, then every rung `recompute_more` climbs to."""
    rungs = [model]
    while (stronger := rungs[-1].recompute_more()) is not None:
        rungs.append(stronger)
        assert len(rungs) <= 16, "the ladder does not end"
    return rungs


RUNG_SEEDS = 8
"""Check 1's declared seeds for a rung's forward against the model's, sized paired by `seeds_for` on
an A100, where a rung's checkpoint barriers change the forward program."""
ZEROS_DRAWN = load("layout_parity").zeros_drawn
"""A parameter tree with each all-zero leaf drawn (tools/layout_parity.py)."""
GRADIENT_COVERING: dict[str, tuple[str, int]] = {
    "dense attention": ("causal_transformer", 8),
    "GQA and latent attention": ("causal_transformer+mla", 8),
    "MoE router": ("causal_transformer+moe", 8),
    "linear and recurrent mixers": ("mamba2-tiny", 8),
    "hybrid stacks": ("qwen35-tiny", 8),
    "vision towers": ("gemma4-unified-native-tiny", 8),
    "hyper-connections": ("deepseek-v4-tiny", 32),
    "attention residuals": ("causal_transformer+attention_residuals", 32),
    "looped (Dory)": ("dory-tiny", 64),
    "diffusion language models": ("dream-tiny", 8),
    "per-layer inputs and AltUp (Gemma 3n)": ("causal_transformer+gemma3n", 32),
    "image and video diffusion transformers": ("simple_dit", 8),
}
"""The architecture classes whose top remat rung's gradient holds check 1 against the model's, each by one
member, the gate's covering (ledger 1385), with the seeds `seeds_for` sized it at, paired, on
armada. Each member stands for the others of its class:

- dense attention: gpt-neo, gptj, falcon, phi, phi3, starcoder2, gpt-bigcode, olmo3-yarn, gemma3-native,
  k2-horizon, modernbert;
- GQA and latent attention: deepseek-v3, deepseek-v32, deepseek-v41, kimi-k25, mimo-v2-flash,
  causal_transformer+deepseek_v2;
- MoE router: mixtral, granitemoe, qwen38-moe, qwen35-moe-native, gemma4-moe, minimax-m2-fp8, llama4,
  nemotron-h-moe-latent, causal_transformer+gemma4_moe, +glm4_moe, +gpt_oss and +llama4;
- linear and recurrent mixers: nemotron-h, kimi-linear;
- hybrid stacks: qwen35-native, qwen35-blocks, causal_transformer+qwen35, glm5-next;
- vision towers: gemma4-tiny-mm, gemma3n-vision, llama4-native;
- hyper-connections: deepseek-v41;
- attention residuals (depth attention over earlier layers' outputs): kimi-k3, whose KDA leaves need
  2048 paired seeds (A_log and dt_bias, a recomputed scan's rounding apart from the saved one's);
- looped (Dory): none other;
- diffusion language models: llada2, diffusion-gemma-sft, diffusion-gemma-workflow;
- per-layer inputs and AltUp: gemma3n-vision;
- image and video diffusion transformers: hybrid_dit, simple_mmdit, simple_udit, video_dit,
  hierarchical_mmdit.

Every other model's top rung holds its forward to check 1 alone."""
GRADIENT_SEEDS = dict(GRADIENT_COVERING.values())
"""Each covering member's declared seeds for its gradient."""
RESOLVED_SINKHORN = {"hc_sinkhorn_iters": 3, "hc_eps": 1e-2}
"""The Sinkhorn configuration the hyper-connection member's gradient is judged at (`judged`): the same
code, five alternating normalisations through its loop, at a fixed point its float64 truth resolves. At
the release's (hc_eps 1e-6, 20 iterations) the comb logits' gradient on a layer whose streams are copies
is hc_eps times the derivative of the Sinkhorn residual S - 1, below float64's own resolution of the
column sums: JAX's and torch's float64 put it 2x apart (1.2677e-23 and 6.3385e-24) and every float32
arm, torch's included, leaves it as noise (ledger 1856). Over 64 drawn 2x2 comb logits (scales 0.02 to
3) under a stream-constant cotangent, JAX's and torch's float64 gradients agree here within 2.7e-12 of
the row's size and float32's error is 2.2e-5 of it at the median and 6.5e-4 at most, where at the
release's setting they sit up to 9.8e5 apart (armada IG1856cfg; 2 iterations at 1e-3 agree within
2.0e-11 but keep one alternation fewer, 5 iterations at 1e-2 already part by 1.9e-8). The release's
setting keeps a finiteness check, its truth beyond float64 (ledger 1864)."""


def judged(model):
    """`model` as its gradient is judged: a hyper-connection model at `RESOLVED_SINKHORN`, every stack
    of streams it holds alike; any other as it is."""
    held = {name: dataclasses.replace(getattr(model, name), **RESOLVED_SINKHORN)
            for name in ("hyper_connections", "mtp_hyper_connections")
            if getattr(model, name, None) is not None}
    return model.clone(**held) if held else model


def _redrawn(inputs, seed: int):
    """`inputs` drawn afresh at `seed` with their shapes and kinds: integer ids below the largest given
    (a token model's ids stay in its vocabulary), floats from a normal, the rest as given."""
    rng = np.random.default_rng([0x2A, seed])

    def leaf(value):
        value = np.asarray(value)
        if np.issubdtype(value.dtype, np.integer):
            return rng.integers(0, int(value.max()) + 1, value.shape).astype(value.dtype)
        if np.issubdtype(value.dtype, np.floating):
            return rng.standard_normal(value.shape).astype(value.dtype)
        return value
    return jax.tree.map(leaf, inputs)


def _climbs_and_restores(name: str, model, variables, inputs, **apply) -> None:
    """Each rung holds less than the one below it, and its top rung steps the
    forward the model does. Each rung keeps fewer residual bytes for its
    backward than the rung below it (`_saved`), so no rung is a remat that
    recomputes nothing. The top rung's checkpoint makes its forward another
    program, which check 1 holds against the model's from the model's float64
    twin over drawn inputs. Its backward recomputes, which can round apart
    (gemma2-tiny's gradients by up to 1.2e-4, kimi-k3-tiny's 8.3e-3 of 10.8;
    test_decoder_remat.py found the same), so a covering class's member
    (`GRADIENT_COVERING`) holds its gradients to check 1 too, each parameter
    its own leaf. Both are taken over the variables with every
    all-zero leaf drawn (tools/layout_parity.py's `zeros_drawn`): a zero output
    projection or an identity modulation, as a fresh DiT holds them, would
    leave its trunk's gradient zero and a wrong recomputed backward unseen.
    A rung's record restores it from any rung below
    it and moves no rung above it, and the top rung runs over the model's
    variables, every collection where it was.

    Of testing.md's memory settings, no remat and a rung of the planner's, a
    model meets each once here: its top rung, which recomputes every name its
    layers mark, so every checkpoint site the architecture has runs under
    remat. Each named policy below it steps on its own in
    test_decoder_remat.py (`test_named_policies_train_the_same_model_as_full`),
    which no architecture here repeats."""
    rungs = _rungs(model)
    records = [rung.recompute_record() for rung in rungs]
    assert len(rungs) > 1
    assert all(records[index] not in records[:index] for index in range(len(records)))
    for low, rung in enumerate(rungs):
        for high in range(len(rungs)):
            restored = rung.restore_recompute(records[high]).recompute_record()
            assert restored == records[max(low, high)], (records[low], records[high], restored)
    assert all(rung.restore_recompute("no rung").recompute_record() == record
               for rung, record in zip(rungs, records, strict=True))
    variables = {**variables, "params": ZEROS_DRAWN(variables["params"], jax.random.key(7))}
    saved = [_saved(rung, variables, inputs, **apply) for rung in rungs]
    assert all(above < below for below, above in itertools.pairwise(saved)), list(zip(records, saved,
                                                                                         strict=True))
    written = model.apply(variables, *inputs, mutable=True, **apply)[1].keys()
    # Device arrays, made under x64 so they stay float64: a layer that indexes a weight by its ids
    # (DeepSeek V4's hash routing, Gemma 4's per-expert scales) indexes a device array in a program.
    with jax.enable_x64(new_val=True):
        twin, wide_variables = float64_twin(model), jax.tree.map(jnp.asarray, widened(variables))

    # Plain functions of the inputs, which the gate compiles, the truths under x64 over inputs it widens.
    def forward(rung, variables=variables):
        return lambda *inputs: _call(rung, variables, inputs, **apply)

    def stepped(rung, variables=variables):
        return lambda *inputs: _stepped(rung, variables, inputs, **apply)

    top = rungs[-1]
    record = top.recompute_record()

    def draw(seed):
        return _redrawn(inputs, seed)

    assert_as_exact(forward(top), forward(model), draw, f"forward at {record}", seeds=RUNG_SEEDS,
                    truth_fn=forward(twin, wide_variables))
    if name in GRADIENT_SEEDS:
        if judged(model) is not model:
            # The release's setting, whose gradient no float64 truth judges, steps finitely.
            assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(stepped(top)(*inputs)))
        assert_as_exact(stepped(judged(top)), stepped(judged(model)), draw, f"gradient at {record}",
                        seeds=GRADIENT_SEEDS[name], truth_fn=stepped(judged(twin), wide_variables))
    assert top.apply(variables, *inputs, mutable=True, **apply)[1].keys() == written


WHOLE_SUITE_PATHS = ("deepseek-v4-tiny", "deepseek-v41-tiny", "qwen35-tiny", "glm5-next-tiny")
"""The checkpoints whose rungs take past a group's 120 s to step, each run by the whole suite alone
(`whole_suite`): DeepSeek V4's and V4.1's hyper-connected layers (130 s and 126 s on armada at 727e70cb1
and d00c68f8f; 256 s and 187 s in the durations record), Qwen 3.5's and GLM-5 Next's (182 s and 162 s
there)."""


@pytest.mark.parametrize("name", [pytest.param(name, marks=pytest.mark.whole_suite)
                                  if name in WHOLE_SUITE_PATHS else name for name in PATHS])
def test_a_checkpoints_remat_rungs_step_as_it_does(name):
    source, read = loaded(name), reads(name, "float32")
    _climbs_and_restores(name, source.model, source.variables, (read.tokens,), method="logits")


CONTINUUM_SEEDS = 8
"""Check 1's declared seeds for a continuum decoder scanned at its top remat rung, and decoding, against
its plain forward (`seeds_for`)."""


def test_a_continuum_level_composes_with_the_scan_the_ladder_and_the_cache():
    """A decoder with a continuum level (`dew.nn.continuum`) scanned at its top remat rung computes the
    logits and gradients the same decoder does unscanned and unrematerialized, and its cached decode
    the logits its own full forward gives each prefix, both on par (check 1) from its float64 twin over
    drawn variables and tokens: the level stands after the stack, so a scan, a rung's checkpoints and a
    cache leave it as the head reads it."""
    from dew.nn.continuum import ContinuumLevel

    plain = CausalTransformer(vocab_size=VOCAB, emb_features=16, num_layers=3, num_heads=2, head_dim=8,
                              mlp_features=32, max_seq_len=32, attention_impl="xla",
                              continuum=ContinuumLevel(hidden=4))
    scanned = _rungs(plain.clone(scan_layers=True))[-1]
    tokens = jnp.zeros((ROWS, LENGTH), jnp.int32)
    shapes = jax.eval_shape(lambda: plain.init(jax.random.key(0), tokens))

    def draw(seed):
        rng = np.random.default_rng(seed)
        variables = jax.tree.map(lambda leaf: (0.3 * rng.standard_normal(leaf.shape)).astype(np.float32),
                                 shapes)
        drawn = rng.integers(0, VOCAB, (ROWS, LENGTH)).astype(np.int32)
        return variables, drawn, rng.standard_normal((ROWS, LENGTH, VOCAB)).astype(np.float32)

    def stepped(model):
        def run(variables, drawn, cotangent):
            output, pull = jax.vjp(lambda params: model.apply({"params": params}, drawn), variables["params"])
            return output, pull(cotangent)[0]
        return run

    assert_as_exact(jax.jit(stepped(scanned)), jax.jit(stepped(plain)), draw,
                    "a scanned, rematerialized continuum decoder", seeds=CONTINUUM_SEEDS,
                    truth_fn=stepped(float64_twin(plain)))

    def decoded(model):
        def run(variables, drawn, _):
            cache = model.apply(variables, ROWS, method="init_cache", mutable=["cache"])[1]["cache"]
            steps = []
            for position in range(LENGTH):
                logits, held = model.apply({**variables, "cache": cache}, drawn[:, position:position + 1],
                                           decode=True, mutable=["cache"])
                cache = held["cache"]
                steps.append(logits[:, 0])
            return jnp.stack(steps, axis=1)
        return run

    def whole(model):
        return lambda variables, drawn, _: model.apply(variables, drawn)

    assert_as_exact(jax.jit(decoded(plain)), jax.jit(whole(plain)), draw,
                    "a continuum decoder's cached decode", seeds=CONTINUUM_SEEDS,
                    truth_fn=whole(float64_twin(plain)))


ATTENTION_RESIDUALS = Case("causal_transformer", {**LM, "num_layers": 4,
                                                  "attention_residuals": AttentionResiduals(block_size=2)},
                           seq_len=SEQ_LEN, label="attention_residuals")
"""Kimi K3's attention residuals (the one `AttentionResiduals` module kimi-k3-tiny runs) over plain
attention at the language cases' size, two blocks of two layers: the attention-residual class's member of
`GRADIENT_COVERING`, standing for kimi-k3-tiny, whose KDA recurrence's gradient needs 2048 paired
seeds."""
RECOMPUTED = [*(case for case in ARCHITECTURES if not case.is_jepa
                and isinstance(models.build(case.architecture, **case.config), Recomputing)),
              ATTENTION_RESIDUALS]


def _inputs(case) -> tuple:
    """A batch of what the case's model takes, at test_architectures.py's sizes."""
    if case.is_lm:
        return (jnp.asarray(np.random.RandomState(0).randint(0, VOCAB, (ROWS, case.seq_len)), jnp.int32),)
    sample = jax.random.normal(jax.random.key(0), (ROWS, *case.sample_shape))
    text = jax.random.normal(jax.random.key(3), (ROWS, TEXT_TOKENS, TEXT_FEATURES))
    return sample, jnp.asarray([0.3, 0.7]), TextContext(text, jnp.ones((ROWS, TEXT_TOKENS), bool))


@pytest.mark.parametrize("case", RECOMPUTED, ids=[case.name for case in RECOMPUTED])
def test_an_architectures_remat_rungs_step_as_it_does(case):
    """Every architecture test_architectures.py trains that climbs a remat
    ladder, decoders and DiT stacks, at its size there."""
    model = models.build(case.architecture, **case.config)
    inputs = _inputs(case)
    _climbs_and_restores(case.name, model, model.init(jax.random.key(0), *inputs), inputs)


def test_the_gradient_covering_names_tested_models():
    """Each covering member is a model the remat tests run, so no class is covered by a name that tests
    nothing."""
    tested = {*PATHS, *(case.name for case in RECOMPUTED)}
    assert {member for member, _ in GRADIENT_COVERING.values()} <= tested
