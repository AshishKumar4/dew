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

import json
import math
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from flax import linen as nn
from recording import RecordingTracker
from reference_error import assert_leaves_as_exact, float64_terms

from dew.artifacts import ImageGrid, Representations, TokenScores, VideoGrid
from dew.config import ModelConfig, ObjectiveConfig
from dew.config.data import ImageData
from dew.data.dataset import Loader
from dew.diffusion import presets
from dew.inputs import Condition, ConditionEncoder, Field, InputSpec
from dew.nn.attention import Stage
from dew.nn.dit import TextContext
from dew.objectives.base import Step
from dew.objectives.diffusion import DiffusionObjective, DiffusionRunConfig
from dew.objectives.jepa import JepaObjective, MultiBlockMask
from dew.objectives.lm import LMObjective
from dew.registry import metrics, models
from dew.sampling import CFG, Euler
from dew.training import Checkpoints, Layout, MeshSpec, Trainer
from dew.training.model_flops import model_batch

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
class StubText(ConditionEncoder):
    """Stands in for CLIP-L/14, whose weights would put a download in every
    case here. Pretokenized text of a fixed width, embedded from a table."""

    params: dict

    @classmethod
    def from_pretrained(cls, checkpoint: str, *, params=None, **fields):
        if params is None:
            params = {"table": jnp.asarray(
                np.random.RandomState(0).normal(size=(TEXT_VOCAB, TEXT_FEATURES)).astype(np.float32))}
        return cls(params=params)

    def tokenize(self, texts):
        ids = np.zeros((len(texts), TEXT_TOKENS), np.int32)
        mask = np.zeros((len(texts), TEXT_TOKENS), np.int32)
        for row, text in enumerate(texts):
            codes = [1] + [2 + (ord(char) % (TEXT_VOCAB - 2)) for char in text[:TEXT_TOKENS - 1]]
            ids[row, :len(codes)] = codes
            mask[row, :len(codes)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def encode(self, params, tokens):
        return TextContext(hidden=params["table"][jnp.asarray(tokens["input_ids"])],
                           mask=jnp.asarray(tokens["attention_mask"]))

    def captions(self, tokens):
        return tuple("".join(chr(97 + int(i)) for i in row[row > 1])
                     for row in np.asarray(tokens["input_ids"]))

    def to_json(self):
        return {"checkpoint": "stub"}


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
    # the group limit and a shared expert.
    Case("causal_transformer", {
        **LM, "head_dim": 16,
        "mixer": {"class": "mla", "fields": {"q_lora_rank": 8, "kv_lora_rank": 8,
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
    # Llama 4's stack at toy width: chunked rotated layers around a global
    # layer under the llama4 kind, the routed layers scaling their inputs.
    Case("causal_transformer", {
        **LM, "head_dim": 8, "qk_norm": False,
        "layer_types": ("chunked_attention", "full_attention"),
        "kinds": {"chunked_attention": {"chunk": 4, "mixer": {
                      "class": "llama4", "fields": {"use_rope": True, "floor_scale": 4.0}}},
                  "full_attention": {"mixer": {"class": "llama4", "fields": {"use_rope": False,
                                               "floor_scale": 4.0}}}},
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
    # first's keys and values.
    Case("causal_transformer", {
        **LM, "num_heads": 4, "num_kv_heads": 2, "head_dim": 8, "mlp": "geglu",
        "mlp_features": (48, 64), "activation_sparsity_pattern": (0.95, 0.0),
        "sandwich_norms": True, "qk_norm": True, "v_norm": True,
        "embedding_scale": True, "attention_scale": 1.0, "final_logit_softcap": 30.0,
        "layer_types": ("sliding_attention", "sliding_attention"),
        "kinds": {"sliding_attention": {"window": 4, "rope_theta": 1e4}},
        "altup": {"num_inputs": 3}, "laurel_rank": 8,
        "per_layer_input_dim": 8, "kv_shared_layers": (1,),
    }, seq_len=SEQ_LEN, label="gemma3n"),
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
            "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4}}}}},
        seq_len=SEQ_LEN, label="qwen35"),
]

IDS = [case.name for case in CASES]


def batches(case: Case, encoder: ConditionEncoder | None):
    """uint8-range samples, as the data pipeline delivers them, with labels for
    the probes and tokenized text for the conditioned models."""
    rng = np.random.default_rng(0)
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
    encoder = None if case.is_lm or case.is_jepa else StubText.from_pretrained("stub")
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
    # Every pass scores every row with the same EMA weight selection.
    assert seen == [shape] * steps, seen
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


def truth(case: Case, tmp_path, initial, steps=2):
    """The case's `steps` steps on one device in float64, from `initial`
    widened: the same rows, keys and draws, every float wider."""
    trainer, encoder = make_trainer(case, tmp_path / f"truth-{steps}", 1, RecordingTracker(), 1)
    source = batches(case, encoder)

    data = Loader(train=lambda partition: (wide(case, batch) for batch in source()),
                  val=lambda partition: (wide(case, batch) for batch in [next(source())]),
                  batch=BATCH, records=None)
    return trainer.fit(data, steps=steps, state=widened_state(initial))


def update_terms(case: Case, tmp_path, initial, steps=2):
    """The terms of each leaf the case's `steps` steps write (`float64_terms`), in float64 from
    `initial`: the trainer's own step body, its transaction (the loss's gradient at each step's
    parameters, the momentum update, the EMA), over the rows `truth` trains on. A parameter's
    terms are its updates' sums and the momentum's the gradients', never the parameter it started
    from, whose own size floors it."""
    trainer, encoder = make_trainer(case, tmp_path / "terms", 1, devices=1)
    source = batches(case, encoder)()
    rows = [model_batch(wide(case, next(source))) for _ in range(steps)]
    state = widened_state(initial)
    with trainer._traced_on(trainer.device_mesh):
        body = trainer._step_body(trainer._loss_shape(state, rows[0]), {}, 1)

        def stepped(state):
            for batch in rows:
                advanced, *_ = body(state, batch, 0)
                state = replace(advanced, step=state.step + 1)
            return state

        return float64_terms(stepped, state)[1]


def decay_terms(objective, variables, batch, index: int) -> dict[tuple[str, ...], np.ndarray]:
    """Each gated delta net layer's decay parameters' gradients as their
    terms summed absolutely, by head, in float64, keyed by the parameter's
    path. The decay g = -exp(A_log) softplus(z), z = a + dt_bias, enters the
    rule at every row and position, so `dt_bias`'s gradient sums dL/dz and
    `A_log`'s sums dL/dg g = dL/dz softplus(z) / sigmoid(z) there, both read
    through the `a` projection, whose cotangent is dL/dz."""
    step = Step(step=jnp.asarray(index), key=jax.random.key(0), ema=None)
    values: dict[tuple[str, ...], jax.Array] = {}

    def loss(offsets, record=None):
        def project(next_fun, args, kwargs, context):
            value = next_fun(*args, **kwargs)
            path = context.module.scope.path
            if context.method_name == "__call__" and path[-1:] == ("in_proj_a",):
                if record is not None:
                    record[path[:-1]] = value
                return value + offsets.get(path[:-1], 0)
            return value

        with nn.intercept_methods(project):
            return objective.scalar_loss(variables, batch, step)[0]

    loss({}, values)
    cotangents = jax.grad(loss)({path: jnp.zeros_like(value) for path, value in values.items()})
    terms = {}
    for path, cotangent in cotangents.items():
        layer = variables["params"]
        for part in path:
            layer = layer[part]
        z = values[path] + layer["dt_bias"]
        decay = cotangent * jax.nn.softplus(z) / jax.nn.sigmoid(z)
        for name, term in (("dt_bias", cotangent), ("A_log", decay)):
            terms[(*path, name)] = np.sum(np.abs(np.asarray(term)), axis=(0, 1))
    return terms


def decay_invariants(case: Case, tmp_path, initial,
                     steps=2) -> dict[str, dict[tuple[str, ...], np.ndarray]]:
    """The declared terms (`raised`) of each gated delta net's decay
    parameters over a language model's one or two SGD steps, by state tree.
    After two the momentum holds momentum g1 + g2, and a parameter that
    starts at zero moves to -lr ((1 + momentum) g1 + g2); after one they
    hold g1 and -lr g1. So each is a sum of the steps' gradients' terms
    (`decay_terms`), measured in float64 at the parameters each step reads.
    A parameter that starts elsewhere is no such sum, and its own size
    floors it."""
    if not case.is_lm:
        return {}

    def starts_at_zero(path):
        leaf = initial.variables["params"]
        for part in path:
            leaf = leaf[part]
        return not np.any(np.asarray(leaf))

    trainer, _ = make_trainer(case, tmp_path / "terms", 1, devices=1)
    batch = {"text": jnp.asarray(next(batches(case, None)())["text"])}
    with jax.enable_x64(new_val=True):
        wide = jax.tree.map(lambda leaf: leaf.astype(jnp.float64) if jnp.issubdtype(leaf.dtype, jnp.floating)
                            else leaf, initial.variables)
        first = decay_terms(trainer.objective, wide, batch, 0)
        if not first:
            return {}
        if steps == 1:
            return {"variables": {path: LEARNING_RATE * first[path]
                                  for path in first if starts_at_zero(path)}, "opt_state": first}
        step = Step(step=jnp.asarray(0), key=jax.random.key(0), ema=None)
        gradient = jax.grad(lambda params: trainer.objective.scalar_loss(
            {**wide, "params": params}, batch, step)[0])(wide["params"])
        moved = {**wide, "params": jax.tree.map(lambda leaf, step: leaf - LEARNING_RATE * step,
                                                wide["params"], gradient)}
        second = decay_terms(trainer.objective, moved, batch, 1)

    return {"variables": {path: LEARNING_RATE * ((1 + MOMENTUM) * first[path] + second[path])
                          for path in first if starts_at_zero(path)},
            "opt_state": {path: MOMENTUM * first[path] + second[path] for path in first}}


def raised(terms, scales: dict[str, dict[tuple[str, ...], np.ndarray]]):
    """`terms` with each leaf whose path ends in a declared parameter's at least
    its declared terms (`decay_invariants`), leaf by leaf."""
    def raise_tree(name, tree):
        def leaf(key, value):
            text = jax.tree_util.keystr(key)
            for path, scale in scales.get(name, {}).items():
                if text.endswith("".join(f"['{part}']" for part in path)):
                    value = np.maximum(value, scale)
            return value
        return jax.tree_util.tree_map_with_path(leaf, tree)
    return replace(terms, **{name: raise_tree(name, getattr(terms, name)) for name in scales})


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_architecture_trains_on_every_layout_as_on_one_device(case, tmp_path):
    """One device, 8x1 data parallelism (every parameter replicated, the
    batch split across every device) and 2x4 fsdp (the parameter tree split
    four ways, the optimizer's state and the EMA with it) train the same
    steps from the same state: each layout's variables, momentum and EMA
    are as near a float64 run as one device's are
    (`assert_leaves_as_exact`), so their summing a batch's rows and
    positions in other orders is the only difference, where a layout that
    dropped or repeated a row would fail by orders of magnitude. After one
    step every row is held to its own size; after two each tensor is, as a
    second step's rows inherit the first's rounding through the parameters
    they read, which their own terms do not see (Gemma 3n's altup output
    scale, a cancelling gradient's smallest element, carries it into a
    per-layer gate's row). Adam's two moments and its count follow the
    parameters' placement too."""
    initial = make_trainer(case, tmp_path / "initial", 1, devices=1)[0].initial_state()

    def copied():
        # Each fit donates the state it was handed.
        return jax.tree.map(jnp.copy, initial)

    # The most terms a gradient sums: every row's positions, and over every
    # layer for a weight the layers share (Gemma 3n's per-layer input norm).
    positions = case.seq_len or math.prod(case.sample_shape[:-1])
    layers = case.config.get("num_layers", 1)
    reductions = BATCH * positions * int(np.sum(layers))

    def as_near(steps, rows):
        _, alone, _ = run_case(case, tmp_path, fsdp=1, devices=1, state=copied(), steps=steps)
        layouts = [run_case(case, tmp_path, fsdp=fsdp, state=copied(), steps=steps)[1] for fsdp in (1, 4)]
        with jax.enable_x64(new_val=True):
            exact = jax.device_get(truth(case, tmp_path, copied(), steps))
            terms = jax.device_get(update_terms(case, tmp_path, copied(), steps))
        # A delta net's decay terms stay declared: its float32 rounding runs through the chunked
        # rule's cumulative sums of the decay, which the step's own terms do not see.
        terms = raised(terms, decay_invariants(case, tmp_path, initial, steps))
        for layout in layouts:
            for name in ("variables", "opt_state", "ema"):
                assert_leaves_as_exact(*jax.device_get((getattr(layout, name), getattr(alone, name))),
                                       getattr(exact, name), reductions=reductions,
                                       terms=getattr(terms, name), rows=rows)
        return layouts

    as_near(1, rows=True)
    replicated, state = as_near(2, rows=False)
    assert not fsdp_leaves(replicated.variables), "nothing may shard on a 1-wide fsdp axis"
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
        data=ImageData(source="tfds://oxford_flowers102", image_size=8), text=None,
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
    """`val_metrics` names the metrics registry, so an unknown name is
    refused where it is written, with the registered names in the message."""
    with pytest.raises(ValueError):
        replace(unet_run(JSON_UNET), val_metrics=("fid_score",))

    for name in sorted(metrics):
        assert replace(unet_run(JSON_UNET), val_metrics=(name,)).val_metrics == (name,)
