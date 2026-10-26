"""Every registered architecture, trained through the trainer, on both meshes.

test_models.py proves each architecture has a working forward pass, and
test_parallelism.py proves the trainer's sharding on one tiny DiT. This file
puts the other architectures' real parameter trees through `Trainer.fit`,
where an unshardable or unsharded architecture shows.

Each case trains two steps on the simulated 8-device CPU mesh, once as pure
data parallelism (8x1) and once as data x fsdp (2x4), and checks what only a
real fit can check: finite losses out of the compiled step, parameters and
their optimizer moments genuinely split over the fsdp axis, the objective's
evaluation running against the sharded EMA copy, and a checkpoint on disk
afterwards.
"""

import functools
import json
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import selection_margins
from diffusion_stubs import StubText
from diffusion_support import images_of
from recording import RecordingTracker
from reference_error import as_given, assert_as_exact, assert_computes_the_oracle, chain_roundings, widened
from residual_orders import permuted, residual_width

from dew.artifacts import ImageGrid, Representations, TokenScores, VideoGrid
from dew.config import MetricConfig, ObjectiveConfig
from dew.config.diffusion import DiffusionRunConfig
from dew.data.dataset import Loader
from dew.data.images import CAPTIONS
from dew.diffusion import presets
from dew.inputs import Condition, ConditionEncoder, Field, InputSpec
from dew.nn.attention import Stage
from dew.nn.text_encoders import DEFAULT_MODEL as CLIP_CHECKPOINT
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.jepa import JepaObjective, MultiBlockMask
from dew.objectives.lm import LMObjective
from dew.registry import ModelConfig, from_record, metrics, models, to_record
from dew.sampling import CFG, Euler
from dew.training import Checkpoints, Layout, MeshSpec, Trainer
from dew.training.distributed import shard_batch
from dew.training.model_flops import prepare_counts

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh


RES = 16
FRAMES = 2
PATCH = 4
GRID = (RES // PATCH, RES // PATCH)
# One batch element per simulated device, so the 8x1 and 2x4 meshes see the
# same global batch.
BATCH = 8
# These models hold thousands of parameters, orders below the production shard
# threshold, so the threshold is lowered; at the production value "fsdp on"
# would replicate everything.
TINY_SHARD = 256
# Enough to run the sampler loop end to end: sample quality is not what a
# two-step run can be about.
SAMPLER_STEPS = 2
# 2x2 target blocks on the 4x4 grid, which leaves 8 context tokens.
MASK = MultiBlockMask.for_grid(GRID, num_targets=2, scale=(0.2, 0.3))

TEXT_TOKENS = 8
TEXT_FEATURES = 32
TEXT_VOCAB = 16


@dataclass(frozen=True)
class Case:
    """One architecture at the smallest size that still exercises it."""

    architecture: str
    config: dict
    frames: int = 0
    """Video architectures take (frames, H, W, C) samples; 0 means images."""
    predictor: dict | None = None
    """Set for JEPA: `architecture` is the encoder and this builds its predictor."""
    seq_len: int = 0
    """Set for language models: batches are token windows, not images."""
    label: str = ""
    """A second case of one architecture, named apart from the first."""

    @property
    def is_jepa(self) -> bool:
        return self.predictor is not None

    @property
    def is_lm(self) -> bool:
        return self.seq_len > 0

    @property
    def sample_shape(self) -> tuple:
        square = (RES, RES, 3)
        return square if self.frames == 0 else (self.frames, *square)

    @property
    def sample_key(self) -> str:
        return "video" if self.frames else "image"

    @property
    def name(self) -> str:
        return self.architecture + (f"+{self.label}" if self.label else "")


DIT = {"patch_size": PATCH, "emb_features": 64, "num_layers": 2, "num_heads": 2,
       "mlp_ratio": 2}
UNET = {"emb_features": 64, "feature_depths": [16, 32],
        "attention_configs": [None, {"heads": 2, "use_projection": False,
                                     "use_self_and_cross": False}],
        "num_res_blocks": 1, "num_middle_res_blocks": 1}
ENCODER = {"patch_size": PATCH, "emb_features": 32, "num_layers": 2, "num_heads": 2,
           "mlp_ratio": 2}
PREDICTOR = {"grid": GRID, "emb_features": 32, "predictor_features": 16,
             "num_layers": 1, "num_heads": 2, "mlp_ratio": 2}

VOCAB = 64
SEQ_LEN = 16
LONG_SEQ_LEN = 128
"""The rows of a cell whose float32 error is heavy-tailed over draws, a few draws holding most of it
(Gemma 3n's and Qwen 3.5's): longer rows put more such events in each draw,
so check 1's pooled ratio resolves over a sized count of draws, where SEQ_LEN's rows left the
reference compared with itself unresolved within 512. Qwen 3.5's rows are twice as long: at 128 its
embedding's momentum still did not resolve within 512. Gemma 3n's draws also clear the sparsity
cutoff by a margin (`selection_margin`)."""
LM = {"vocab_size": VOCAB, "emb_features": 32, "num_layers": 2, "num_heads": 2,
      "num_kv_heads": 1, "mlp_features": 64, "max_seq_len": SEQ_LEN}

CASES = [
    Case("unet", UNET),
    # Both U-shaped stacks split their layers into a down and an up half
    Case("uvit", {"patch_size": PATCH, "emb_features": 64, "num_layers": 4,
                  "num_heads": 2}),
    Case("simple_udit", {**DIT, "num_layers": 2}),
    Case("simple_dit", DIT),
    Case("simple_mmdit", DIT),
    Case("hierarchical_mmdit", {"base_patch_size": 2, "emb_features": (32, 64, 96),
                                "num_layers": (1, 1, 1), "num_heads": (2, 2, 2),
                                "mlp_ratio": 2}),
    Case("hybrid_dit", {**DIT, "num_layers": 4, "ssm_state_dim": 8,
                        "ssm_attention_ratio": "3:1"}),
    Case("video_dit", {**DIT, "num_layers": 1}, frames=FRAMES),
    Case("unet_3d", {**UNET, "attention_configs": [None, None], "temporal_heads": 2},
         frames=FRAMES),
    Case("jepa_encoder", ENCODER, predictor=PREDICTOR),
    Case("jepa_video_encoder", {**ENCODER, "num_layers": 1},
         predictor={**PREDICTOR, "factorized": True}, frames=FRAMES),
    Case("causal_transformer", LM, seq_len=SEQ_LEN),
    # Eight experts so the expert dimension divides every mesh this file
    # builds, on the second layer only, so one dense and one sparse
    # feed-forward go through the same run.
    Case("causal_transformer", {**LM, "mixture": {"experts": 8, "top_k": 2, "layers": (1,)}},
         seq_len=SEQ_LEN, label="moe"),
    # GPT OSS at toy width: sink attention on both kinds, the fused biased
    # experts on every layer, and YaRN over grouped-query heads.
    Case("causal_transformer", {
        **LM, "mlp": "swigluoai", "attention_sinks": True, "qk_norm": False,
        "layer_types": ("sliding_attention", "full_attention"),
        "kinds": {"sliding_attention": {"window": 4}},
        "yarn": {"factor": 4.0, "original_max_position_embeddings": 8, "truncate": False},
        "mixture": {"experts": 8, "top_k": 2},
    }, seq_len=SEQ_LEN, label="gpt_oss"),
    # DeepSeek V3.2's stack at toy width: the mla mixer with its sparse
    # indexer on every layer, and the routed layer with the balancing bias,
    # the group limit and a shared expert. The low ranks are 32 wide, so the
    # latent norms' scales, whose rounding over a draw varies most, hold
    # enough values a draw for the layout cell to resolve.
    Case("causal_transformer", {
        **LM, "head_dim": 16,
        "mixer": {"class": "mla", "fields": {"q_lora_rank": 32, "kv_lora_rank": 32,
                  "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8,
                  "index_topk": 4, "index_n_heads": 2, "index_head_dim": 16}},
        "mixture": {"experts": 8, "top_k": 2, "layers": (1,), "score_function": "sigmoid",
                    "groups": 4, "groups_per_token": 2, "bias": True,
                    "expert_features": 16, "shared_features": 16},
    }, seq_len=SEQ_LEN, label="mla"),
    # DeepSeek V2 Lite's stack at toy width: the mla mixer without the query
    # LoRA, and a routed layer under softmax group-limited routing scored by
    # the best expert, unnormalised, with a shared expert and no bias.
    Case("causal_transformer", {
        **LM, "head_dim": 16,
        "mixer": {"class": "mla", "fields": {"kv_lora_rank": 8, "qk_nope_head_dim": 8,
                  "qk_rope_head_dim": 8, "v_head_dim": 8}},
        "mixture": {"experts": 8, "top_k": 4, "layers": (1,), "score_function": "softmax",
                    "norm_topk_prob": False, "groups": 4, "groups_per_token": 2,
                    "group_score": "max", "expert_features": 16, "shared_features": 32},
    }, seq_len=SEQ_LEN, label="deepseek_v2"),
    # GLM 4.5's stack at toy width: biased q/k/v over a bias-free o_proj, a
    # half rotary, DeepSeek V3 routing with a shared expert on the last
    # layer, and one MTP depth that routes like it.
    Case("causal_transformer", {
        **LM, "attention_bias": True, "o_proj_bias": False, "qk_norm": False,
        "head_dim": 8, "partial_rotary_factor": 0.5, "partial_rotary_type": "default",
        "mixture": {"experts": 8, "top_k": 2, "layers": (1,), "score_function": "sigmoid",
                    "bias": True, "expert_features": 16, "shared_features": 16},
        "num_nextn_predict_layers": 1,
    }, seq_len=SEQ_LEN, label="glm4_moe"),
    # Llama 4's stack at toy width: chunked rotated layers, adjacent pairs under the weightless L2
    # norm, around a global layer without positions tuning its temperature; the routed layers scale
    # their inputs.
    Case("causal_transformer", {
        **LM, "head_dim": 8, "qk_norm": False,
        "layer_types": ("chunked_attention", "full_attention"),
        "kinds": {"chunked_attention": {"chunk": 4, "qk_norm": True, "mixer": {
                      "class": "attention", "fields": {"rotary_pairs": "adjacent", "qk_norm_weight": False}}},
                  "full_attention": {"mixer": {"class": "attention", "fields": {
                      "nope": True, "temperature_tuning": (4.0, 0.1)}}}},
        "mixture": {"experts": 8, "top_k": 1, "layers": (1,), "score_function": "sigmoid",
                    "norm_topk_prob": False, "scale_inputs": True, "expert_features": 16,
                    "shared_features": 16},
    }, seq_len=SEQ_LEN, label="llama4"),
    # Gemma 4's 26B-A4B at toy width: the routed branch beside every layer's
    # dense MLP, the global layer reading its values off the keys with its
    # own head dim and key/value count, and the per-layer output scalar.
    Case("causal_transformer", {
        **LM, "num_heads": 4, "num_kv_heads": 2, "head_dim": 8, "mlp": "geglu",
        "sandwich_norms": True, "qk_norm": True, "v_norm": True,
        "embedding_scale": True, "attention_scale": 1.0, "partial_rotary_factor": 0.25,
        "layer_types": ("sliding_attention", "full_attention"),
        "kinds": {"sliding_attention": {"window": 4},
                  "full_attention": {"head_dim": 16, "num_kv_heads": 1}},
        "attention_k_eq_v": True, "layer_scalar": "frozen",
        "mixture": {"experts": 8, "top_k": 2, "expert_features": 16, "parallel": True},
    }, seq_len=SEQ_LEN, label="gemma4_moe"),
    # Gemma 3n at toy width: three copies of the residual stream under
    # AltUp, the LAuReL block, gaussian top-k on the first layer, widths
    # that differ per layer, per-layer inputs and the last layer sharing the
    # first's keys and values. The first layer's 192 units keep about ten a
    # token at 0.95 sparsity, where the released models keep hundreds: at 48,
    # about two, a token whose kept units all sit just above the cutoff gives
    # the feed-forward's post-norm a near-zero input, whose direction float32
    # resolves poorly, and four draws in 320 held 60% of a layer-0 leaf's
    # pooled error (at 192, at most 22%; at 384, 6%).
    Case("causal_transformer", {
        **LM, "num_heads": 4, "num_kv_heads": 2, "head_dim": 8, "mlp": "geglu",
        "mlp_features": (192, 64), "activation_sparsity_pattern": (0.95, 0.0),
        "sandwich_norms": True, "qk_norm": True, "v_norm": True,
        "embedding_scale": True, "attention_scale": 1.0, "final_logit_softcap": 30.0,
        "layer_types": ("sliding_attention", "sliding_attention"),
        "kinds": {"sliding_attention": {"window": 4, "rope_theta": 1e4}},
        "altup": {"num_inputs": 3}, "laurel_rank": 8,
        "per_layer_input_dim": 8, "kv_shared_layers": (1,), "max_seq_len": LONG_SEQ_LEN,
    }, seq_len=LONG_SEQ_LEN, label="gemma3n"),
    # Qwen3.5's stack: gated delta net layers on the linear_attention kind,
    # one gated full-attention layer with the sliced partial rotary. The
    # delta net's projections are wide enough to cross the shard threshold,
    # so its declarations and its heuristic conv taps are what the layout
    # places here.
    Case("causal_transformer", {
        **LM, "num_layers": 4, "head_dim": 8, "output_gate": True,
        "partial_rotary_factor": 0.5, "partial_rotary_type": "default",
        "layer_types": ("linear_attention",) * 3 + ("full_attention",),
        "kinds": {"linear_attention": {"mixer": {
            "class": "gated_delta_net", "fields": {"linear_num_key_heads": 2,
            "linear_num_value_heads": 4, "linear_key_head_dim": 8,
            "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4}}}},
        "max_seq_len": 2 * LONG_SEQ_LEN}, seq_len=2 * LONG_SEQ_LEN, label="qwen35"),
]

IDS = [case.name for case in CASES]


def batches(case: Case, encoder: ConditionEncoder | None, seed: int = 0):
    """uint8-range samples, as the data pipeline delivers them, with labels for
    the probes and tokenized text for the conditioned models, drawn from `seed`."""
    rng = np.random.default_rng(seed)
    if case.is_lm:
        batch = {"text": rng.integers(0, VOCAB, size=(BATCH, case.seq_len + 1)).astype(np.int32)}
    else:
        batch = {case.sample_key: rng.integers(0, 256, size=(BATCH, *case.sample_shape)).astype(np.uint8),
                 "label": np.arange(BATCH) % 4}
        if encoder is not None:
            batch["text"] = encoder.tokenize([f"sample {index}" for index in range(BATCH)])

    def source():
        while True:
            yield batch

    return source


class Spread:
    """A real metric over the objective's evaluation: the artifact's spread.

    Recording the shapes here and asserting afterwards makes an evaluation
    that never ran a failure.
    """

    def __init__(self, seen, reads):
        self.seen = seen
        self.reads = reads
        self.name = "artifact_spread"

    def __call__(self, artifact, batch):
        values = np.asarray(jax.tree.leaves(artifact)[0])
        self.seen.append(values.shape)
        return float(values.std()), 1

    def merge(self, accumulated, contribution):
        return accumulated[0] + contribution[0], accumulated[1] + contribution[1]

    def finalize(self, accumulated):
        return accumulated[0] / accumulated[1]


def make_objective(case: Case, model, encoder):
    if case.is_lm:
        return LMObjective(model, case.seq_len)
    sample = Field(case.sample_key, case.sample_shape)
    if case.is_jepa:
        return JepaObjective(model, models.build("jepa_predictor", **case.predictor),
                             MASK, sample=sample)
    inputs = InputSpec(sample, {"textcontext": Condition(encoder, field="text")})
    return DiffusionObjective(model, presets.EDM(regime="pixel"), inputs, steps=SAMPLER_STEPS,
                              guidance=CFG(2.0), solver=Euler())


LEARNING_RATE, MOMENTUM = 1e-2, 0.9
LINEAR = optax.sgd(LEARNING_RATE, momentum=MOMENTUM)
"""The optimizer the layouts are compared under: SGD with momentum moves the
parameters and the momentum by each gradient linearly, so a layout's state
differs from one device's by its gradients' rounding alone. Adam divides by
each gradient's own size, so where a gradient is zero in exact arithmetic
(a zero-initialised modulation's shift, a key bias) its rounding becomes a
full step whose sign no order of summation fixes."""


def make_trainer(case: Case, tmp_path, fsdp, tracker=None, devices=None, optimizer=LINEAR):
    """The trainer a recipe would build for this case: a registry model, the
    real objective, a real optimizer, no wandb, over `devices` devices (all
    by default)."""
    encoder = None if case.is_lm or case.is_jepa else StubText.from_pretrained(
        "stub", tokens=TEXT_TOKENS, features=TEXT_FEATURES, vocab=TEXT_VOCAB)
    model = models.build(case.architecture, **case.config)
    trainer = Trainer(
        make_objective(case, model, encoder), optimizer, key=jax.random.key(0),
        mesh=MeshSpec(fsdp=fsdp), layout=Layout(min_shard=TINY_SHARD),
        checkpoints=Checkpoints(str(tmp_path / f"{case.name}-fsdp{fsdp}")), tracker=tracker)
    if devices is not None:
        trainer.device_mesh = trainer.mesh.build(jax.devices()[:devices])
    return trainer, encoder


def expected_artifact(case: Case):
    if case.is_lm:
        return TokenScores, (BATCH, case.seq_len)
    if case.is_jepa:
        return Representations, (BATCH, case.config["emb_features"])
    if case.frames:
        return VideoGrid, (BATCH, *case.sample_shape)
    return ImageGrid, (BATCH, *case.sample_shape)


def fsdp_leaves(tree):
    return [leaf for leaf in jax.tree.leaves(tree) if 'fsdp' in str(leaf.sharding.spec)]


def run_case(case: Case, tmp_path, fsdp, devices=None, state=None, steps=2):
    """One run of `steps` steps on the mesh over `devices` devices (all by
    default), from `state` (the trainer's own initial state by default),
    with everything the assertions read: the trainer, the state and the
    training losses."""
    seen = []
    tracker = RecordingTracker()
    trainer, encoder = make_trainer(case, tmp_path / f"{devices}-{fsdp}-{steps}", fsdp, tracker, devices)
    artifact, shape = expected_artifact(case)
    source = batches(case, encoder)

    data = Loader(train=lambda partition: source(),
                   val=lambda partition: (batch for batch in [next(source())]),
                   batch=BATCH, records=None)
    state = trainer.fit(data, steps=steps, log_every=1, eval_every=1,
                        metrics=(Spread(seen, artifact),), preview=True, state=state)

    assert dict(trainer.device_mesh.shape) == {"data": (devices or jax.device_count()) // fsdp, "expert": 1,
                                               "fsdp": fsdp, "tensor": 1, "sequence": 1, "stage": 1}
    assert int(state.step) == steps
    losses = [s["train/loss"] for _, s in tracker.scalars if "train/loss" in s]
    assert len(losses) == steps and all(np.isfinite(loss) for loss in losses), losses
    # Every pass scores every row with the same EMA weight selection, the one that warms the
    # evaluations before the first step included where one runs before the end.
    assert seen == [shape] * (steps + (steps > 1)), seen
    scores = [s["val/artifact_spread"] for _, s in tracker.scalars if "val/artifact_spread" in s]
    assert len(scores) == steps and all(np.isfinite(score) for score in scores)
    assert [type(value) for _, value in tracker.artifacts
            if isinstance(value, (ImageGrid, Representations, TokenScores, VideoGrid))] == (
                [] if case.is_lm else [artifact] * steps)
    assert Checkpoints(trainer.checkpoints.directory).latest == steps
    return trainer, state, losses


def wide(case: Case, batch):
    """A batch as the float64 runs read it: its samples float64, so their
    conversion and the diffusion noising do not share the float32 runs' roundings."""
    return batch if case.is_lm else {**batch, case.sample_key: batch[case.sample_key].astype(np.float64)}


def widened_state(state):
    return jax.tree.map(lambda leaf: leaf.astype(jnp.float64) if jnp.issubdtype(leaf.dtype, jnp.floating)
                        else jnp.copy(leaf), state)


LAYOUT_SEEDS: dict[str, int] = {
    "unet": 64, "uvit": 8, "simple_udit": 8, "simple_dit": 8, "simple_mmdit": 8,
    "hierarchical_mmdit": 8, "hybrid_dit": 8, "video_dit": 8, "unet_3d": 32, "jepa_encoder": 128,
    "jepa_video_encoder": 128, "causal_transformer": 32, "causal_transformer+moe": 32,
    "causal_transformer+gpt_oss": 256, "causal_transformer+deepseek_v2": 64,
    "causal_transformer+glm4_moe": 128, "causal_transformer+llama4": 32,
    "causal_transformer+gemma4_moe": 64, "causal_transformer+gemma3n": 64,
    "causal_transformer+mla": 32, "causal_transformer+qwen35": 64,
}
"""Each cell's declared seeds for check 1 (`assert_as_exact`), the most any layout and step count took,
sized paired by `seeds_for` (`dew_fn`, the layout's arm) on armada: every cell's arms share their
upstream rounding (the same state, batch and draws), so the paired ratio the verdict judges resolves in
the draws it needs, as few as the reference compared with itself needs or up to 16 times fewer
(testing.md states how paired counts were validated). For three cells it is the only count that
resolves at all: Gemma 3n's and Qwen 3.5's errors are heavy-tailed across draws (the four largest of
64 draws hold half of a layer's pooled error, LONG_SEQ_LEN), and MLA's (one draw in 1024 holds all of
one device's pooled error, seed 247, an indexer selection the draw filter lets through: ledger 1527),
where the reference compared with itself resolves no momentum within 4096 draws. Every cell is named:
one added is sized."""

def selects(case: Case) -> bool:
    """Whether the cell's model selects discontinuously (`selection_margins.selects`)."""
    return case.is_lm and selection_margins.selects(case.config)


@functools.cache
def selections_forward(name: str):
    """A cell's forward capturing its discontinuous selections (`selection_margins.capture`), compiled
    once a cell."""
    case = next(case for case in CASES if case.name == name)
    return selection_margins.capture(models.build(case.architecture, **case.config))


def selection_margin(case: Case, variables, tokens) -> float:
    """The smallest margin of the model's discontinuous selections over the forward of `variables` on
    the batch's input tokens, relative to their scale (`selection_margins.margin`)."""
    captured = selections_forward(case.name)(variables, jnp.asarray(tokens[:, :-1]))
    return selection_margins.margin(case.config, captured, case.name)


def clears(case: Case, raw, initial, stepped) -> bool:
    """Whether the selections of each step's forward on the batch `raw` clear the margin
    (`selection_margin`): the first step's from `initial`, then the second's from the state
    `stepped(raw)` makes, made only once the first's clear."""
    return (selection_margin(case, initial, raw["text"]) > selection_margins.MARGIN
            and selection_margin(case, stepped(raw), raw["text"]) > selection_margins.MARGIN)


def reordered(state, order):
    """`state` with its residual stream's units in `order` (`tests/residual_orders.permuted`), its
    variables and EMA alike: the same function in other coordinates, every contraction inside the
    model summed in another order. Its momentum starts at zero, which no order moves. An empty
    `order` is a model with no residual stream to permute, and leaves `state` as it is."""
    if not len(order):
        return state
    assert all(not np.any(np.asarray(leaf)) for leaf in jax.tree.leaves(state.opt_state)
               if jnp.issubdtype(leaf.dtype, jnp.floating))
    ema = None if state.ema is None else permuted(state.ema, order)
    return state.replace(variables=permuted(state.variables, order), ema=ema)


def in_identity_order(stepped: dict, order):
    """A drawn step's variables, optimizer state and EMA (`drawn_steps`) with the residual stream's
    units taken back out of `order`: each tree that mirrors the parameters (the parameters, the EMA's,
    the optimizer's momentum) permuted by its inverse, so that every order of one draw is judged against
    that draw's one truth (`ORBIT`). A permutation moves no leaf's RMS nor any row's, so a cell judged
    at one order alone keeps its verdict."""
    if not len(order):
        return stepped
    inverse = np.argsort(order)
    params = jax.tree.structure(stepped["variables"]["params"])

    def mirrors(node) -> bool:
        return jax.tree.structure(node) == params

    return jax.tree.map(lambda node: permuted({"params": node}, inverse)["params"] if mirrors(node) else node,
                        stepped, is_leaf=mirrors)


ORBIT = 4
"""The residual orders each draw of a mixture cell runs at (`assert_as_exact`'s `orders`): its own and
ORBIT - 1 more, each arm's worst row its worst over them. A routed expert's row of a step's momentum
can nearly cancel on one draw, its rounding then set by the order its few tokens sum in; one order's
worst row of either arm is then an extreme of that order alone (ledger 1531), where over the same
orders both arms meet the same cancellation. A mitigation measured, not a bound: over fresh windows of
the one-step 8x1-data site, the seven mixture cells' own arms failed 2 of 898 verdicts over four orders
where they failed 12 at the draw's order alone, and every arm scaled 1.6 times or given an independent
error 2.5 times its own failed at both."""


def orbit(width: int):
    """`ORBIT` orders of each draw (`assert_as_exact`'s `orders`): the draw's own, and its order
    composed with ORBIT - 1 fixed shuffles of the `width` residual units, each as uniform over the orders
    as the draw's own."""
    shuffles = [np.random.default_rng([0x0B17, width, at]).permutation(width) for at in range(1, ORBIT)]

    def orders(inputs):
        raw, order = inputs
        return [inputs, *((raw, order[shuffle]) for shuffle in shuffles)]
    return orders


def drawn_steps(case: Case, tmp_path, fsdp, devices, initial, steps: int, *, exact: bool = False):
    """`steps` steps of a layout over the batch a draw gives, from `initial` in the residual order
    it gives (`reordered`), every step on the same rows, as `run_case`'s `fit` takes them, the step
    compiled once for every draw at the plan fit takes, a frontier's first, untimed in Dew's tests
    (`dew.training.planning.UNTIMED_FRONTIER`): a function of the raw batch and the order returning
    the stepped state's variables, optimizer state and EMA taken back to the identity order
    (`in_identity_order`). `exact` runs one device in float64 from the state widened, the batch's
    samples widened (`wide`), built at its first call, so under the caller's float64."""
    built = []

    @as_given
    def stepped(raw, order):
        if not built:
            trainer, _ = make_trainer(case, tmp_path / f"drawn-{devices}-{fsdp}-{steps}-{exact}", fsdp,
                                      RecordingTracker(), devices)
            built.append(trainer)
        trainer = built[0]
        start = reordered(widened_state(initial) if exact else jax.tree.map(jnp.copy, initial), order)
        # A copy each draw, as each step donates the state it is handed.
        state = jax.device_put(start, trainer.shardings(start))
        batch = shard_batch(trainer.device_mesh, prepare_counts(
            trainer.objective, wide(case, raw) if exact else raw, count_model=False))
        if len(built) == 1:
            built.append(trainer.compile(state, batch))
        for _ in range(steps):
            state, *_ = built[1](state, batch)
        return in_identity_order(
            jax.device_get({name: getattr(state, name) for name in ("variables", "opt_state", "ema")}), order)

    return stepped


@pytest.mark.parametrize("case", [case for case in CASES if case.is_lm],
                         ids=[name for case, name in zip(CASES, IDS, strict=True) if case.is_lm])
def test_a_residual_order_is_the_same_decoder(case):
    """The residual orders the layout cells draw (`permuted`) reorder sums only: each language cell's
    float64 logits under an order other than the identity are the identity's within float64's
    rounding of their own longest chain (`assert_computes_the_oracle`), so a draw over orders is a
    draw of roundings and never another model. Every parameter is drawn apart from its initial value,
    a norm's or a router's scale as uneven as a trained one, so a residual-wide parameter `permuted`
    leaves in place moves the logits; a step's state, which `in_identity_order` takes back, holds no
    other."""
    model = models.build(case.architecture, **case.config)
    ids = np.random.default_rng(3).integers(0, LM["vocab_size"], (2, SEQ_LEN)).astype(np.int32)
    initial = jax.jit(model.init)(jax.random.key(0), ids)
    rng = np.random.default_rng(4)
    variables = jax.tree.map(lambda leaf: leaf + rng.normal(0, 0.1, leaf.shape).astype(leaf.dtype)
                             if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, initial)
    width = residual_width(variables)
    with jax.enable_x64(new_val=True):
        wide = widened(variables)
        identity = np.asarray(model.clone(dtype=jnp.float64).apply(wide, ids))
        for seed in (1, 2):
            order = np.random.default_rng([0x0BB17, seed]).permutation(width)
            reordered = model.clone(dtype=jnp.float64).apply(permuted(wide, order), ids)
            roundings = chain_roundings(jax.make_jaxpr(
                lambda variables: model.clone(dtype=jnp.float64).apply(variables, ids))(wide))
            assert_computes_the_oracle(np.asarray(reordered), identity, f"{case.name} order {seed}",
                                       roundings=roundings)


LAYOUTS = {"8x1 data": 1, "2x4 fsdp": 4}
"""Each layout a cell trains on, by its fsdp width over every device."""

LAYOUT_STEPS = (((1, "8x1 data"), (2, "2x4 fsdp")), ((1, "2x4 fsdp"), (2, "8x1 data")))
"""The pairs of step count and layout a case trains, alternating by case: each case meets each step
count and each layout, and each step count meets each layout over the cases, as testing.md's
settings pairwise asks."""

LAYOUT_CELLS = [(case, steps, layout) for index, case in enumerate(CASES)
                for steps, layout in LAYOUT_STEPS[index % 2]]


@pytest.mark.parametrize(("case", "steps", "layout"), LAYOUT_CELLS,
                         ids=[f"{case.name}-{steps}-step{'s' * (steps > 1)}-{layout.split()[1]}"
                              for case, steps, layout in LAYOUT_CELLS])
def test_architecture_trains_on_every_layout_as_on_one_device(case, steps, layout, tmp_path):
    """One device and a layout, 8x1 data parallelism (every parameter
    replicated, the batch split across every device) or 2x4 fsdp (the
    parameter tree split four ways, the optimizer's state and the EMA with
    it), train the same steps from the same state: after `steps` steps (one,
    or two, each case training each layout once and each step count once,
    `LAYOUT_STEPS`), the layout's variables, momentum and EMA are as near a
    float64 run as one device's are, over the cell's declared batches (check
    1, `assert_as_exact`), so
    their summing a batch's rows and positions in other orders is the only
    difference, where a layout that dropped or repeated a row would fail by
    orders of magnitude. The steps check 1 draws are `fit`'s own on the
    run's batch, bit for bit, each compiled once for every draw. Adam's two
    moments and its count follow the parameters' placement too."""
    trainer, encoder = make_trainer(case, tmp_path / "initial", 1, devices=1)
    initial = trainer.initial_state()

    def copied():
        # Each fit donates the state it was handed.
        return jax.tree.map(jnp.copy, initial)

    width = residual_width(initial.variables) if case.is_lm else 0
    one_step = drawn_steps(case, tmp_path, 1, 1, initial, 1)

    def keyed(key):
        """A batch drawn from the key and, independently, an order of the residual stream's units
        (`reordered`), the cell's own batch and the identity at key 0: the fresh batch varies the
        values, the order every contraction's summation, and the reference and each layout start
        from the same reordered state, so all are judged in its coordinates. A model with no
        token embedding has no residual stream to permute, and draws batches alone."""
        order = np.arange(width) if key == 0 else np.random.default_rng(key).permutation(width)
        return next(batches(case, encoder, key)()), order

    def cleared(inputs):
        # The selections of each step's forward clear the margin: the first step's from the initial
        # state, the second's from the state the first makes.
        return clears(case, inputs[0], initial.variables,
                      lambda batch: one_step(batch, np.arange(width))["variables"])

    # A model that selects discontinuously takes the seed's first batch whose selections clear
    # `selection_margins.MARGIN` (`selection_margins.clearing`).
    draw = selection_margins.clearing(keyed, cleared) if selects(case) else keyed

    seeds = LAYOUT_SEEDS[case.name]
    fsdp = LAYOUTS[layout]
    _, alone, _ = run_case(case, tmp_path, fsdp=1, devices=1, state=copied(), steps=steps)
    _, state, _ = run_case(case, tmp_path, fsdp=fsdp, state=copied(), steps=steps)
    drawn = {"one device": drawn_steps(case, tmp_path, 1, 1, initial, steps),
             layout: drawn_steps(case, tmp_path, fsdp, None, initial, steps)}
    # The drawn steps are fit's own on its batch, bit for bit, so what check 1 judges is the run's:
    # the cell's own batch in its own order, whatever selection margin a draw then asks.
    own = (next(batches(case, encoder)()), np.arange(width))
    for fitted, stepped in zip((alone, state), drawn.values(), strict=True):
        ours = stepped(*own)
        for name, tree in ours.items():
            held = jax.device_get(getattr(fitted, name))
            assert jax.tree.structure(tree) == jax.tree.structure(held), name
            for mine, theirs in zip(jax.tree.leaves(tree), jax.tree.leaves(held), strict=True):
                mine, theirs = np.asarray(mine), np.asarray(theirs)
                assert mine.dtype == theirs.dtype and mine.shape == theirs.shape, name
                assert np.array_equal(mine.view(np.uint8), theirs.view(np.uint8)), name
    truth = drawn_steps(case, tmp_path, 1, 1, initial, steps, exact=True)
    # A mixture's rows are judged over each draw's orbit of residual orders (`ORBIT`).
    orders = orbit(width) if case.is_lm and selection_margins.routes(case.config) else None
    assert_as_exact(drawn[layout], drawn["one device"], draw, f"{case.name} {layout}, {steps} step(s)",
                    seeds=seeds, truth_fn=truth, orders=orders)
    if fsdp == 1:
        assert not fsdp_leaves(state.variables), "nothing may shard on a 1-wide fsdp axis"
        return
    sharded = fsdp_leaves(state.variables)
    assert sharded, "no parameter was sharded over the fsdp axis"
    for param in sharded:
        assert param.addressable_shards[0].data.size == param.size // 4, \
            "shard is not a quarter of the global parameter"
    # The optimizer's state and the EMA copy follow the params they track,
    # without the optimizer or the model ever describing a layout.
    param_specs = [leaf.sharding.spec for leaf in jax.tree.leaves(state.variables["params"])]
    assert param_specs == [leaf.sharding.spec for leaf in jax.tree.leaves(state.opt_state[0].trace)]
    assert {leaf.sharding.spec for leaf in jax.tree.leaves(state.ema)} <= set(param_specs)
    adam = make_trainer(case, tmp_path / "adam", 4, optimizer=optax.adam(1e-3))[0].place()[0].opt_state[0]
    for moment in (adam.mu, adam.nu):
        assert param_specs == [leaf.sharding.spec for leaf in jax.tree.leaves(moment)]


# --------------------------------------------------------------------------
# The declarations as the fsdp axis widens
# --------------------------------------------------------------------------

# A unet the way a command line and a run.json carry it: every sequence a
# list, every stage a record. Two stages over 8-pixel images is the smallest
# unet that still downsamples once and attends on the coarse stage.
JSON_UNET = {"emb_features": 32, "feature_depths": [8, 16], "norm_groups": 4,
             "num_res_blocks": 1, "num_middle_res_blocks": 1,
             "attention_configs": [None, {"heads": 2, "use_projection": True}]}


def unet_run(fields):
    return DiffusionRunConfig(
        model=ModelConfig("unet", {**fields, "dtype": "float32", "attention_impl": "reference"}),
        data=images_of("tfds://oxford_flowers102", 8), text=None,
        objective=ObjectiveConfig("diffusion", {"guidance": None, "solver": Euler(), "steps": SAMPLER_STEPS}))


def test_a_unet_from_a_json_record_generates_what_its_value_twin_does():
    """JSON reconstruction preserves sampling with the same weights and key."""
    value_unet = {**JSON_UNET, "feature_depths": (8, 16),
                  "attention_configs": (None, Stage(heads=2, use_projection=True))}

    record = json.loads(json.dumps(unet_run(JSON_UNET).to_dict()))
    from_json = DiffusionRunConfig.from_dict(record).build()
    from_values = unet_run(value_unet).build()

    state = Trainer(from_json, optax.adam(1e-3), key=jax.random.key(0)).initial_state()
    key = jax.random.key(3)
    prompts = ["a water lily", "a sunflower"]
    drawn = from_json.pipeline(state, ema=False)(
        prompts, steps=SAMPLER_STEPS, key=key).host().images
    np.testing.assert_array_equal(
        drawn, from_values.pipeline(state, ema=False)(
            prompts, steps=SAMPLER_STEPS, key=key).host().images)


def test_a_val_metric_no_registry_knows_is_refused():
    """A run's metric names the metrics registry, so an unknown name is
    refused where it is written, and every registered one is held as its
    import path."""
    with pytest.raises(KeyError):
        MetricConfig("fid_score")

    for name in sorted(metrics):
        (held,) = replace(unet_run(JSON_UNET), val_metrics=(MetricConfig(name),)).val_metrics
        assert metrics[held.name] is metrics[name]


def test_a_val_metric_records_its_own_constructors_defaults():
    """A metric whose class has a constructor of its own over its base's fields, as CLIP's has,
    records that constructor's defaults, so a run holds the CLIP it scored with and reads it
    back. A run's metrics are such records, so a bare alias in their place is refused where the
    run is written: in its record, on its command line and in its fields."""
    clip = MetricConfig("clip")
    assert clip.defaults == {"modelname": CLIP_CHECKPOINT, "field": CAPTIONS}
    assert from_record(MetricConfig, to_record(clip, MetricConfig)) == clip
    run = unet_run(JSON_UNET)
    with pytest.raises(TypeError):
        DiffusionRunConfig.from_dict({**run.to_dict(), "val_metrics": ["psnr"]})
    with pytest.raises(TypeError):
        DiffusionRunConfig.cli(["--val-metrics", '["psnr"]'])
    with pytest.raises(TypeError):
        replace(run, val_metrics=("psnr",))
