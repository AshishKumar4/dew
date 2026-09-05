"""Every registered architecture, trained through the trainer, on both meshes.

test_models.py proves each architecture has a working forward pass, and
test_parallelism.py proves the trainer's sharding on one tiny DiT. Between
them nothing else puts the other architectures' real parameter trees through
`Trainer.fit`, so an architecture could be unshardable, or silently never
sharded, and only a production run would find out.

Each case trains two steps on the simulated 8-device CPU mesh, once as pure
data parallelism (8x1) and once as data x fsdp (2x4), and checks what only a
real fit can check: finite losses out of the compiled step, parameters and
their optimizer moments genuinely split over the fsdp axis, the objective's
evaluation running against the sharded EMA copy, and a checkpoint on disk
afterwards. The declarations behind the layout are checked too: every matrix
parameter of every registered model is declared or listed as heuristic, and
every declared name is carried by some parameter.
"""

import fnmatch
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh
from jax.sharding import PartitionSpec as P

from dew.artifacts import ImageGrid, Representations, TokenScores, VideoGrid
from dew.diffusion import presets
from dew.inputs import Condition, ConditionEncoder, Field, InputSpec
from dew.nn.dit import TextContext
from dew.nn.sharding import DECLARED, HEURISTIC, declared_axes, is_heuristic, parameter_path
from dew.objectives.diffusion import DiffusionObjective
from dew.objectives.jepa import JepaObjective, multi_block_mask
from dew.objectives.lm import LMObjective
from dew.registry import models
from dew.sampling import CFG, Euler
from dew.training import Checkpoints, Layout, MeshSpec, Trainer, build_mesh

RES = 16
FRAMES = 2
PATCH = 4
GRID = (RES // PATCH, RES // PATCH)
# One batch element per simulated device, so the 8x1 and 2x4 meshes see the
# same global batch.
BATCH = 8
# These models hold thousands of parameters, orders below the production shard
# threshold, so lower it or "fsdp on" would mean "everything replicated".
TINY_SHARD = 256
# Enough to run the sampler loop end to end and nothing more: sample quality
# is not what a two-step run can be about.
SAMPLER_STEPS = 2
# 2x2 target blocks on the 4x4 grid, which leaves 8 context tokens.
MASK = multi_block_mask(GRID, num_targets=2, scale=(0.2, 0.3))

TEXT_TOKENS = 8
TEXT_FEATURES = 32
TEXT_VOCAB = 16


@dataclass(frozen=True)
class StubText(ConditionEncoder):
    """Stands in for CLIP-L/14, whose weights would put a download in every
    case here. Pretokenized text of a fixed width, embedded from a table."""

    params: dict

    @classmethod
    def from_pretrained(cls, checkpoint: str, **fields):
        return cls(params={"table": jnp.asarray(
            np.random.RandomState(0).normal(size=(TEXT_VOCAB, TEXT_FEATURES)).astype(np.float32))})

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
    predictor: Optional[dict] = None
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
    # DeepSeek V3.2's stack at toy width: the mla mixer with its sparse
    # indexer on every layer, and the routed layer with the balancing bias,
    # the group limit and a shared expert.
    Case("causal_transformer", {
        **LM, "head_dim": 16,
        "mixer": {"kind": "mla", "q_lora_rank": 8, "kv_lora_rank": 8,
                  "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8,
                  "index_topk": 4, "index_n_heads": 2, "index_head_dim": 16},
        "mixture": {"experts": 8, "top_k": 2, "layers": (1,), "score_function": "sigmoid",
                    "groups": 4, "groups_per_token": 2, "bias": True,
                    "expert_features": 16, "shared_features": 16},
    }, seq_len=SEQ_LEN, label="mla"),
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
            "kind": "gated_delta_net", "linear_num_key_heads": 2,
            "linear_num_value_heads": 4, "linear_key_head_dim": 8,
            "linear_value_head_dim": 8, "linear_conv_kernel_dim": 4}}}},
        seq_len=SEQ_LEN, label="qwen35"),
]

# jepa_predictor has no training step of its own: it is built through the
# registry and trained inside the two JEPA cases.
COVERED = {case.architecture for case in CASES} | {"jepa_predictor"}

IDS = [case.name for case in CASES]


def test_every_registry_architecture_is_trained_here():
    """The point of the file: a new architecture must arrive with a trained case."""
    assert COVERED == set(models)


def model_variables(case: Case, concrete: bool = False):
    """The case's variables as shapes, or as arrays when `concrete` (the
    hilbert scan builds its permutation on the host and does not trace)."""
    model = models.build(case.architecture, **case.config)
    rng = jax.random.key(0)
    init = model.init if concrete else (lambda *args: jax.eval_shape(model.init, *args))
    if case.is_lm:
        return init(rng, jnp.ones((1, case.seq_len), jnp.int32))
    sample = jnp.ones((1, *case.sample_shape), jnp.float32)
    if case.is_jepa:
        return init(rng, sample, jnp.arange(MASK.num_context, dtype=jnp.int32)[None])
    return init(rng, sample, jnp.ones((1,)),
                TextContext(jnp.ones((1, TEXT_TOKENS, TEXT_FEATURES)), jnp.ones((1, TEXT_TOKENS), bool)))


# Options the trained cases leave off but whose modules are declared: shapes
# only, so the declarations behind them are checked without a training run.
VARIANTS = [
    replace(case, config={**case.config, "tie_embeddings": False}, label="untied")
    for case in CASES if case.is_lm and "num_experts" not in case.config
] + [
    Case("simple_dit", {**DIT, "scan_order": "hilbert"}, label="hilbert"),
    Case("hybrid_dit", {**DIT, "num_layers": 4, "ssm_state_dim": 8,
                        "ssm_attention_ratio": "3:1", "use_2d_fusion": True}, label="fusion"),
    Case("uvit", {"patch_size": PATCH, "emb_features": 64, "num_layers": 4, "num_heads": 2,
                  "add_residualblock_output": True}, label="residual"),
    # The Gemma 4 gaps leave the default tree untouched, so a case with them
    # on carries their declarations: per-layer table, projection, gate and
    # projection, with the second layer sharing the first's K/V.
    Case("causal_transformer", {**LM, "per_layer_input_dim": 8,
                                "num_kv_shared_layers": 1}, seq_len=SEQ_LEN, label="gemma4"),
    # A multi-token-prediction depth adds its projection and block beside the
    # backbone, so the declarations behind them are checked here.
    Case("causal_transformer", {**LM, "num_nextn_predict_layers": 1}, seq_len=SEQ_LEN, label="mtp"),
    # Qwen2 biases q, k and v while o_proj stays bias-free, so the split
    # dial's declarations are walked with the odd projection left out.
    Case("causal_transformer", {**LM, "attention_bias": True, "o_proj_bias": False},
         seq_len=SEQ_LEN, label="qwen2"),
]


def frozen_towers():
    """The condition encoders' towers, from the committed tiny configs: they
    are placed on the mesh beside the model, so their declarations are
    checked like a model's."""
    import json

    from dew.nn.text_encoders import (
        CLIP, CLIPTextTransformer, CLIPVisionTransformer, T5EncoderTransformer,
        translate_clip_config, translate_t5_config,
    )
    fixtures = Path(__file__).resolve().parent / "fixtures"
    clip_config = translate_clip_config(
        json.loads((fixtures / "clip" / "tiny" / "config.json").read_text()))
    clip = CLIP(text_model=CLIPTextTransformer(**clip_config["text"]),
                vision_model=CLIPVisionTransformer(**clip_config["vision"]),
                projection_dim=clip_config["projection_dim"])
    vision = clip_config["vision"]
    pixels = jnp.zeros((1, vision["num_channels"], vision["image_size"], vision["image_size"]))
    yield clip, (pixels, jnp.ones((1, 4), jnp.int32))
    t5_config = translate_t5_config(
        json.loads((fixtures / "t5" / "tiny" / "config.json").read_text()))
    tokens = (jnp.ones((1, 4), jnp.int32),)
    yield T5EncoderTransformer(**t5_config), tokens
    # The tiny fixture is gated-gelu; the relu feed-forward declares a module
    # of its own, so a tower with it is walked too.
    yield T5EncoderTransformer(**{**t5_config, "feed_forward_proj": "relu"}), tokens


def every_leaf():
    """Every parameter leaf of every case and variant, the predictor and the
    frozen towers included."""
    for case in CASES:
        yield from jax.tree_util.tree_flatten_with_path(model_variables(case))[0]
    for case in VARIANTS:
        yield from jax.tree_util.tree_flatten_with_path(model_variables(case, concrete=True))[0]
    model = models.build("jepa_predictor", **PREDICTOR)
    variables = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, MASK.num_context, 32)),
        jnp.arange(MASK.num_context, dtype=jnp.int32)[None],
        jnp.arange(MASK.block_area, dtype=jnp.int32)[None])
    yield from jax.tree_util.tree_flatten_with_path(variables)[0]
    for tower, inputs in frozen_towers():
        variables = jax.eval_shape(tower.init, jax.random.key(0), *inputs)
        yield from jax.tree_util.tree_flatten_with_path(variables)[0]


def test_every_matrix_parameter_is_declared_or_listed_as_heuristic():
    """A parameter of rank two or more is placed by a declaration on its
    module, or its module says it takes the shape heuristic; nothing is
    placed by a name that happened to match."""
    undeclared = sorted({
        "/".join(parameter_path(path)) for path, leaf in every_leaf()
        if leaf.ndim >= 2 and declared_axes(path, leaf.ndim) is None
        and not is_heuristic(path)})
    assert undeclared == []


def test_every_declared_name_is_carried_by_a_parameter():
    """A renamed module has to break its declaration, not silently stop
    matching it: every declared suffix and every heuristic pattern names some
    parameter of some registered model."""
    modules = {parameter_path(path)[:-1] for path, _ in every_leaf()}
    unmatched = [key for key in DECLARED
                 if not any(module[-len(key):] == key for module in modules)]
    assert unmatched == []
    paths = {parameter_path(path) for path, _ in every_leaf()}
    unused = [pattern for pattern in HEURISTIC if not any(
        all(fnmatch.fnmatchcase(name, glob) for name, glob in zip(names[start:], pattern))
        for names in paths for start in range(len(names) - len(pattern) + 1))]
    assert unused == []


def text_condition() -> Condition:
    return Condition(StubText.from_pretrained("stub"), field="text", unconditional="")


def batches(case: Case, encoder: Optional[ConditionEncoder]):
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
        return float(values.std())

    def reduce(self, values):
        return float(np.mean(values))


class RecordingTracker:
    def __init__(self):
        self.scalars = []
        self.artifacts = []

    def log(self, scalars, step):
        self.scalars.append((step, dict(scalars)))

    def artifact(self, value, step):
        self.artifacts.append((step, value))


def make_objective(case: Case, model, encoder):
    if case.is_lm:
        return LMObjective(model, case.seq_len)
    sample = Field(case.sample_key, case.sample_shape)
    if case.is_jepa:
        return JepaObjective(model, models.build("jepa_predictor", **case.predictor),
                             MASK, sample=sample)
    inputs = InputSpec(sample, {"textcontext": Condition(encoder, field="text")})
    return DiffusionObjective(model, presets.EDM()(), inputs, steps=SAMPLER_STEPS,
                              guidance=CFG(2.0), sampler=Euler())


def make_trainer(case: Case, tmp_path, fsdp, tracker=None):
    """The trainer a recipe would build for this case: a registry model, the
    real objective, the real optimizer, no wandb."""
    encoder = None if case.is_lm or case.is_jepa else StubText.from_pretrained("stub")
    model = models.build(case.architecture, **case.config)
    trainer = Trainer(
        make_objective(case, model, encoder), optax.adam(1e-3), key=jax.random.key(0),
        mesh=MeshSpec(fsdp=fsdp), layout=Layout(min_shard=TINY_SHARD),
        checkpoints=Checkpoints(str(tmp_path / f"{case.name}-fsdp{fsdp}")), tracker=tracker)
    return trainer, encoder


def expected_artifact(case: Case):
    if case.is_lm:
        return TokenScores, (BATCH, case.seq_len)
    if case.is_jepa:
        return Representations, (BATCH, case.config["emb_features"])
    if case.frames:
        return VideoGrid, (4, *case.sample_shape)
    return ImageGrid, (4, *case.sample_shape)


def fsdp_leaves(tree):
    return [leaf for leaf in jax.tree.leaves(tree) if 'fsdp' in str(leaf.sharding.spec)]


def run_case(case: Case, tmp_path, fsdp):
    """One two-step run on the mesh, with everything the assertions read."""
    seen = []
    tracker = RecordingTracker()
    trainer, encoder = make_trainer(case, tmp_path, fsdp, tracker)
    artifact, shape = expected_artifact(case)
    source = batches(case, encoder)

    class Data:
        train = staticmethod(source)
        val = staticmethod(lambda: (batch for batch in [next(source())]))
        batch, records, steps_per_epoch = BATCH, None, None

    state = trainer.fit(Data(), steps=2, log_every=1, eval_every=1,
                        metrics=(Spread(seen, artifact),))

    assert dict(trainer.device_mesh.shape) == {"data": jax.device_count() // fsdp, "expert": 1,
                                               "fsdp": fsdp, "tensor": 1, "sequence": 1}
    assert int(state.step) == 2
    losses = [s["train/loss"] for _, s in tracker.scalars if "train/loss" in s]
    assert len(losses) == 2 and all(np.isfinite(loss) for loss in losses), losses
    # A pass after step 1 and one at the end, both from the EMA parameters as
    # they sit on the mesh. A diffusion objective samples four images
    # whatever the batch holds, since each is a full sampler pass.
    assert seen == [shape] * 2, seen
    scores = [s["val/artifact_spread"] for _, s in tracker.scalars if "val/artifact_spread" in s]
    assert len(scores) == 2 and all(np.isfinite(score) for score in scores)
    assert [type(value) for _, value in tracker.artifacts] == [artifact] * 2
    assert Checkpoints(trainer.checkpoints.directory).latest == 2
    return trainer, state


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_architecture_trains_data_parallel(case, tmp_path):
    """8x1: every parameter replicated, the batch split across every device."""
    _, state = run_case(case, tmp_path, fsdp=1)
    assert not fsdp_leaves(state.params), "nothing may shard on a 1-wide fsdp axis"


@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_architecture_trains_under_fsdp(case, tmp_path):
    """2x4: the parameter tree really split four ways, moments and EMA with it."""
    _, state = run_case(case, tmp_path, fsdp=4)

    sharded = fsdp_leaves(state.params)
    assert sharded, "no parameter was sharded over the fsdp axis"
    for param in sharded:
        assert param.addressable_shards[0].data.size == param.size // 4, \
            "shard is not a quarter of the global parameter"

    # Adam's moments and the EMA copy follow the params they track, without
    # the optimizer or the model ever describing a layout.
    param_specs = [leaf.sharding.spec for leaf in jax.tree.leaves(state.params["params"])]
    assert param_specs == [leaf.sharding.spec
                           for leaf in jax.tree.leaves(state.opt_state[0].mu)]
    ema_specs = {leaf.sharding.spec for leaf in jax.tree.leaves(state.ema)}
    assert ema_specs <= set(param_specs)
    assert fsdp_leaves(state.opt_state[0].mu), "no optimizer moment was sharded"


# --------------------------------------------------------------------------
# The declarations as the fsdp axis widens
# --------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("fsdp_size", [2, 4, 8])
@pytest.mark.parametrize("case", CASES, ids=IDS)
def test_every_architecture_shards_within_the_tolerance_at_every_width(case, fsdp_size):
    """Each architecture on a 2, 4 and 8 wide fsdp axis, not just the 4 above.

    Three properties at once, because they share one derivation. The layout
    has to be reproducible, or two processes deriving it would disagree and
    the run would deadlock on mismatched collectives. Every dimension a spec
    names has to be splittable that many ways, or jit rejects the layout. And
    what the rules could not place has to stay inside the tolerance, or the
    run is quietly training a replicated model on every device.
    """
    variables = model_variables(case)
    mesh = build_mesh(MeshSpec(fsdp=fsdp_size))
    layout = Layout(min_shard=TINY_SHARD)
    shardings = layout.shardings(mesh, variables)
    specs = jax.tree.map(lambda sharding: sharding.spec, shardings)
    assert specs == jax.tree.map(lambda s: s.spec, layout.shardings(mesh, variables)), \
        "the derivation is not stable"

    leaves = jax.tree_util.tree_flatten_with_path(variables["params"])[0]
    for (path, value), sharding in zip(
            leaves, jax.tree.leaves(shardings["params"]), strict=True):
        for dimension, entry in enumerate(sharding.spec):
            if not entry:
                continue
            size = value.shape[dimension]
            where = f"{jax.tree_util.keystr(path)} {value.shape} -> {sharding.spec}"
            assert size % fsdp_size == 0, f"{where} cannot split {fsdp_size} ways"
            assert size > 1, f"{where} shards a dimension of one"

    layout.check(variables["params"], shardings["params"], mesh)


# One named parameter per case whose spec the declarations decide, written
# out rather than derived, so a declaration that stops matching a module
# shows up here.
NAMED_LEAF = {
    "causal_transformer": (("embed_tokens", "embedding"), P("fsdp")),
    "simple_dit": (("dit_block_0", "attention", "to_q", "kernel"), P("fsdp")),
}


@pytest.mark.slow
@pytest.mark.parametrize("fsdp_size", [2, 4, 8])
@pytest.mark.parametrize("case", [case for case in CASES if case.name in NAMED_LEAF],
                         ids=[case.name for case in CASES if case.name in NAMED_LEAF])
def test_a_placed_state_carries_the_layout_the_declarations_derive(case, tmp_path, fsdp_size):
    """The derivation through the path a run takes, not called on its own.

    Every other sharding test calls the layout and reads its return value, so
    a layout that is derived correctly and never reaches the state leaves all
    of them green. Dropping out_shardings from the jit that materialises the
    state is that mutation. Placing the state is what these assertions read.
    """
    trainer, _ = make_trainer(case, tmp_path, fsdp_size)
    state, shardings, _ = trainer.place()
    path, expected = NAMED_LEAF[case.name]
    leaf = state.params["params"]
    for key in path:
        leaf = leaf[key]
    assert leaf.sharding.spec == expected, f"{path} {leaf.shape}"
    assert leaf.addressable_shards[0].data.size == leaf.size // fsdp_size

    for placed, derived in zip(jax.tree.leaves(state), jax.tree.leaves(shardings), strict=True):
        assert placed.sharding == derived
