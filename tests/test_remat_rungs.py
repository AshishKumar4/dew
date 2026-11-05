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
from arm_arithmetic import float64_held
from checkpoint_cases import LENGTH, PATHS, ROWS, loaded, tokens
from flax import linen as nn
from model_support import float64_twin, floating_gradient, stepped_outputs
from reference_error import assert_as_exact
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
    return floating_gradient(lambda held: _call(model, held, inputs, **apply), variables, cotangent)


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
    (a token model's ids stay in its vocabulary), floats from a normal, the rest as given; JAX arrays,
    whose arms the gate compiles before it runs them (a NumPy draw had it run each model arm once op by
    op, a minute of a rung's check on armada)."""
    rng = np.random.default_rng([0x2A, seed])

    def leaf(value):
        value = np.asarray(value)
        if np.issubdtype(value.dtype, np.integer):
            return jnp.asarray(rng.integers(0, int(value.max()) + 1, value.shape).astype(value.dtype))
        if np.issubdtype(value.dtype, np.floating):
            return jnp.asarray(rng.standard_normal(value.shape).astype(value.dtype))
        return value
    return jax.tree.map(leaf, inputs)


def _climbs_and_restores(model, variables, inputs, **apply) -> None:
    """Each rung holds less than the one below it, and its record restores it. Each rung keeps fewer
    residual bytes for its backward than the rung below it (`_saved`), over the variables with every
    all-zero leaf drawn (`_zeros_drawn`), so no rung is a remat that recomputes nothing. A rung's record
    restores it from any rung below it and moves no rung above it, and the top rung runs over the model's
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
    variables = _zeros_drawn(variables)
    saved = [_saved(rung, variables, inputs, **apply) for rung in rungs]
    assert all(above < below for below, above in itertools.pairwise(saved)), list(zip(records, saved,
                                                                                         strict=True))
    # The collections each call writes, traced abstractly: op by op the two calls took most of the
    # test (ledger 3905).
    def written(rung):
        return jax.eval_shape(lambda variables: rung.apply(variables, *inputs, mutable=True, **apply)[1],
                              variables).keys()

    assert written(rungs[-1]) == written(model)


def _zeros_drawn(variables):
    """`variables` with every all-zero leaf drawn (tools/layout_parity.py's `zeros_drawn`): a zero output
    projection or an identity modulation, as a fresh DiT holds them, would leave its trunk's gradient zero
    and a wrong recomputed backward unseen."""
    return {**variables, "params": ZEROS_DRAWN(variables["params"], jax.random.key(7))}


def _against_its_twin(model, variables):
    """`variables` with every all-zero leaf drawn (`_zeros_drawn`), the model's top rung, its record, and
    its float64 twin over those variables widened: the arms and the truth check 1 holds the top rung's step
    to, over the drawn variables both arms read. Device arrays, made under x64 so they stay float64: a layer
    that indexes a weight by its ids (DeepSeek V4's hash routing, Gemma 4's per-expert scales) indexes a
    device array in a program (row 3655)."""
    variables = _zeros_drawn(variables)
    top = _rungs(model)[-1]
    with jax.enable_x64(new_val=True):
        wide = float64_held(variables)
        return variables, top, top.recompute_record(), float64_twin(model), wide


def _steps_its_forward(model, variables, inputs, **apply) -> None:
    """The top rung's checkpoint makes its forward another program, which check 1 holds against the model's
    from the model's float64 twin over drawn inputs (`_against_its_twin`)."""
    variables, top, record, twin, wide = _against_its_twin(model, variables)
    assert_as_exact(lambda *drawn: _call(top, variables, drawn, **apply),
                    lambda *drawn: _call(model, variables, drawn, **apply),
                    lambda seed: _redrawn(inputs, seed), f"forward at {record}", seeds=RUNG_SEEDS,
                    truth_fn=lambda *drawn: _call(twin, wide, drawn, **apply))


def _steps_its_gradient(name: str, model, variables, inputs, **apply) -> None:
    """The top rung's backward recomputes, which can round apart (gemma2-tiny's gradients by up to 1.2e-4,
    kimi-k3-tiny's 8.3e-3 of 10.8; test_decoder_remat.py found the same), so a covering class's member
    (`GRADIENT_COVERING`) holds its gradients to check 1 against the model's from its float64 twin, each
    parameter its own leaf (`_against_its_twin`), at the setting its truth resolves (`judged`)."""
    variables, top, record, twin, wide = _against_its_twin(model, variables)
    if judged(model) is not model:
        # The release's setting, whose gradient no float64 truth judges, steps finitely: one compiled step,
        # where run op by op it took 16 s of DeepSeek V4's 87 on armada.
        stepped = jax.jit(lambda *drawn: _stepped(top, variables, drawn, **apply))(*inputs)
        assert all(np.isfinite(leaf).all() for leaf in jax.tree.leaves(stepped))
    assert_as_exact(lambda *drawn: _stepped(judged(top), variables, drawn, **apply),
                    lambda *drawn: _stepped(judged(model), variables, drawn, **apply),
                    lambda seed: _redrawn(inputs, seed), f"gradient at {record}", seeds=GRADIENT_SEEDS[name],
                    truth_fn=lambda *drawn: _stepped(judged(twin), wide, drawn, **apply))


@pytest.mark.parametrize("name", PATHS)
def test_a_checkpoints_remat_rungs_climb_and_restore(name):
    source = loaded(name)
    _climbs_and_restores(source.model, source.variables, (tokens(name),), method="logits")


@pytest.mark.parametrize("name", PATHS)
def test_a_checkpoints_top_rung_steps_its_forward_as_it_does(name):
    source = loaded(name)
    _steps_its_forward(source.model, source.variables, (tokens(name),), method="logits")


@pytest.mark.parametrize("name", [name for name in PATHS if name in GRADIENT_SEEDS])
def test_a_covering_checkpoints_top_rung_steps_its_gradient_as_it_does(name):
    source = loaded(name)
    _steps_its_gradient(name, source.model, source.variables, (tokens(name),), method="logits")


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
    shapes = jax.eval_shape(lambda: nn.unbox(plain.init(jax.random.key(0), tokens)))

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

    forward = jax.jit(plain.apply)

    def passes(variables, drawn, _):
        # The full passes at the shapes the decode's steps compute, one token each from an empty cache.
        return stepped_outputs(lambda length: forward(variables, drawn[:, :length]), LENGTH, 1)

    assert_as_exact(jax.jit(decoded(plain)), passes, draw,
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


def _built(case) -> tuple:
    """The case's model, its variables and its inputs (`_inputs`)."""
    model = models.build(case.architecture, **case.config)
    inputs = _inputs(case)
    return model, nn.unbox(model.init(jax.random.key(0), *inputs)), inputs


COVERED = [case for case in RECOMPUTED if case.name in GRADIENT_SEEDS]
"""The architectures that stand for a class of the gradient's covering (`GRADIENT_COVERING`)."""


@pytest.mark.parametrize("case", RECOMPUTED, ids=[case.name for case in RECOMPUTED])
def test_an_architectures_remat_rungs_climb_and_restore(case):
    """Every architecture test_architectures.py trains that climbs a remat
    ladder, decoders and DiT stacks, at its size there."""
    _climbs_and_restores(*_built(case))


@pytest.mark.parametrize("case", RECOMPUTED, ids=[case.name for case in RECOMPUTED])
def test_an_architectures_top_rung_steps_its_forward_as_it_does(case):
    _steps_its_forward(*_built(case))


@pytest.mark.parametrize("case", COVERED, ids=[case.name for case in COVERED])
def test_a_covering_architectures_top_rung_steps_its_gradient_as_it_does(case):
    _steps_its_gradient(case.name, *_built(case))


def test_the_gradient_covering_names_tested_models():
    """Each covering member is a model the remat tests run, so no class is covered by a name that tests
    nothing."""
    tested = {*PATHS, *(case.name for case in RECOMPUTED)}
    assert {member for member, _ in GRADIENT_COVERING.values()} <= tested
