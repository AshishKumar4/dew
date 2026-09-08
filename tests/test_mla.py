"""Multi-head latent attention parity against transformers 5.16.1.

The fixtures come from tools/mla_reference.py: tiny DeepseekV3Attention
blocks with and without the query LoRA (YaRN rope at the released
spelling, interleaved pairs), a tiny DeepseekV32Attention with its DSA
indexer and biased projections, and the YaRN derivation standalone.
Everything runs at fp32 on CPU, and each parity test states its tolerance
and the largest difference observed. The middle section runs the same
blocks under `attention_impl='auto'`, whose kernels take one head width for
the keys and the values where MLA's differ. The last section trains the
V3.2 stack, mixer and routed experts together, on the simulated mesh.
"""

import functools
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from jax.sharding import PartitionSpec as P

from dew.interop.hf_decoders import _yarn_record, translate_weights
from dew.nn.attention import scaled_dot_product_attention
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers import mixers
from dew.nn.mla import (
    MLAMixer,
    MultiHeadLatentAttention,
    YarnScaling,
    mla_rope_freqs,
    yarn_attention_factor,
    yarn_inv_freq,
    yarn_query_scale,
)
from dew.objectives.lm import LMObjective
from dew.training import Layout, MeshSpec, Trainer

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "mla"
CONFIG = json.loads((FIXTURES / "config.json").read_text())


def fixture(name: str) -> dict:
    with np.load(FIXTURES / f"{name}.npz") as data:
        return {key: np.asarray(value) for key, value in data.items()}


def yarn_of(settings: dict) -> YarnScaling:
    """The fixture's rope spelling as the mixer's yarn record."""
    record = _yarn_record(
        dict(settings["rope_scaling"], rope_theta=settings["rope_theta"]),
        "rope_scaling", float(settings["rope_theta"]),
        int(settings["max_position_embeddings"]))
    return YarnScaling(**record)


def mla_module(settings: dict, max_seq_len: int = 64,
               attention_impl: str | None = None
               ) -> MultiHeadLatentAttention:
    """The fixture's reference config as a dew mixer."""
    return MultiHeadLatentAttention(
        emb_features=settings["hidden_size"],
        num_heads=settings["num_attention_heads"],
        max_seq_len=max_seq_len,
        q_lora_rank=settings["q_lora_rank"],
        kv_lora_rank=settings["kv_lora_rank"],
        qk_nope_head_dim=settings["qk_nope_head_dim"],
        qk_rope_head_dim=settings["qk_rope_head_dim"],
        v_head_dim=settings["v_head_dim"],
        rope_theta=float(settings["rope_theta"]),
        rope_interleave=settings.get("rope_interleave", True),
        yarn=yarn_of(settings),
        norm_eps=float(settings["rms_norm_eps"]),
        attention_bias=bool(settings["attention_bias"]),
        index_topk=settings.get("index_topk"),
        index_n_heads=settings.get("index_n_heads"),
        index_head_dim=settings.get("index_head_dim"),
        attention_impl=attention_impl)


def block_variables(tensors: dict) -> dict:
    """Block-relative fixture tensors as the mixer's parameter tree.

    The fixtures hold one attention block's tensors under their torch
    names; a checkpoint carries them under model.layers.N.self_attn.*,
    so the test prefixes that path and runs the real weight translation.
    """
    names = {f"model.layers.0.self_attn.{name}": tensor
             for name, tensor in tensors.items()
             if name not in ("hidden", "output")}
    tree = translate_weights(names, {"tie_embeddings": False})["params"]
    return {"params": tree["layers_0"]["self_attn"]}


def block_output(name: str, settings: dict,
                 attention_impl: str | None = None) -> float:
    """Largest difference between the dew mixer and the fixture block."""
    tensors = fixture(name)
    module = mla_module(settings, attention_impl=attention_impl)
    output = module.apply(block_variables(tensors),
                          jnp.asarray(tensors["hidden"]))
    return float(np.max(np.abs(np.asarray(output) - tensors["output"])))


def test_yarn_matches_the_reference_derivation():
    """Inverse frequencies, cos/sin amplitude and positions, standalone.

    Tolerance rtol 1e-6 on the inverse frequencies and 1e-5 on cos and sin;
    observed 0 on the inverse frequencies, the attention factor and the
    query scale, 6.0e-8 on cos and 3.0e-8 on sin.
    """
    tensors = fixture("yarn")
    yarn = yarn_of(CONFIG["v3"])

    np.testing.assert_allclose(
        np.asarray(yarn_inv_freq(8, 10000.0, yarn)), tensors["inv_freq"],
        rtol=1e-6, atol=1e-7)
    assert yarn_attention_factor(yarn) == pytest.approx(
        float(tensors["attention_scaling"]), rel=1e-6)
    cos, sin = mla_rope_freqs(
        jnp.arange(CONFIG["length"]), 8, 10000.0, yarn)
    half = tensors["cos"].shape[-1] // 2
    # The reference batches identical position rows; the first row compares.
    np.testing.assert_allclose(np.asarray(cos), tensors["cos"][0, ..., :half],
                               rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.asarray(sin), tensors["sin"][0, ..., :half],
                               rtol=1e-5, atol=1e-6)
    # (0.1 * ln(40) + 1) ** 2.
    assert yarn_query_scale(yarn) == pytest.approx(
        (0.1 * math.log(40.0) + 1.0) ** 2, rel=1e-6)


def decode_loop(module, variables, tokens):
    """Prefill plus one-token steps, the sampler's decode protocol."""
    state = module.init(jax.random.key(0), tokens[:, :1], decode=True)
    state = {**state, "params": variables["params"]}
    outputs = []
    for position in range(tokens.shape[1]):
        out, mutated = module.apply(
            state, tokens[:, position:position + 1],
            decode=True, mutable=["cache"])
        state = {**state, "cache": mutated["cache"]}
        outputs.append(out)
    return jnp.concatenate(outputs, axis=1)


SETTINGS = {"mla_v3": "v3", "mla_v3n": "v3n", "mla_v32": "v32"}


@pytest.mark.parametrize("name", ["mla_v3", "mla_v3n", "mla_v32"])
def test_decode_matches_prefill(name):
    """Tolerance 1e-5; observed 2.4e-6, 9.5e-7 and 1.9e-6."""
    settings = CONFIG[SETTINGS[name]]
    tensors = fixture(name)
    module = mla_module(settings)
    variables = block_variables(tensors)
    hidden = jnp.asarray(tensors["hidden"])
    full = module.apply(variables, hidden)
    assert np.max(np.abs(np.asarray(decode_loop(module, variables, hidden))
                         - np.asarray(full))) < 1e-5


def test_mla_reproduces_the_v3_block():
    """DeepseekV3Attention: q LoRA, compressed KV, interleaved YaRN rope.

    Tolerance 2e-5; observed 2.9e-6.
    """
    assert block_output("mla_v3", CONFIG["v3"]) < 2e-5


def test_mla_reproduces_the_v3_block_without_the_query_lora():
    """The plain-q_proj variant no released checkpoint uses, same reference.

    Tolerance 2e-5; observed 2.1e-6.
    """
    assert block_output("mla_v3n", CONFIG["v3n"]) < 2e-5


def test_mla_reproduces_the_v32_block():
    """DeepseekV32Attention: biased projections, indexer top-k mask fold.

    Tolerance 2e-5; observed 1.9e-6.

    The first fixture was the dense block, not the sparse one. The
    reference folds the indexer's top-k into the attention mask only when
    its config names the eager or sdpa attention path, and a standalone
    DeepseekV32Config leaves `_attn_implementation` None, so the generator
    handed the indices to the eager kernel as a keyword it ignores. The
    port, which always selects, differed from that fixture by 3.42 while
    every intermediate up to the selection matched to 1e-6. The generator
    names the eager path now (5aa94b9). The dense mixer on the same weights
    still differs from the fixture by 3.4, so a fixture that lost the
    selection again fails on the second assertion.
    """
    assert block_output("mla_v32", CONFIG["v32"]) < 2e-5
    dense = dict(CONFIG["v32"], index_topk=None, index_n_heads=None,
                 index_head_dim=None)
    assert block_output("mla_v32", dense) > 1.0


# --------------------------------------------------------------------------
# The kernel the head widths select
# --------------------------------------------------------------------------


def readout_gradients(module, variables, hidden):
    """Gradients of one fixed non-uniform readout of the block's output, so a
    value column that came back in the wrong place moves them."""
    def loss(params):
        out = module.apply(params, hidden)
        weights = jax.random.normal(jax.random.key(0), out.shape, out.dtype)
        return jnp.sum(out * weights)
    return jax.grad(loss)(variables)


@pytest.mark.parametrize("name", ["mla_v3", "mla_v3n", "mla_v32"])
def test_auto_takes_the_unequal_query_and_value_head_widths(name):
    """`attention_impl='auto'` computes the reference block on the shape it
    used to refuse.

    MLA's values are narrower than its queries (`v_head_dim` 8 against
    `qk_nope_head_dim + qk_rope_head_dim` 16 in these fixtures, 128 against
    192 in every released V3 config), and `jax.nn.dot_product_attention`
    checks the value against the key's whole shape, so both fused paths
    raised before the value was padded to the query's width. Tolerance 2e-5,
    the bound of the parity tests above; observed 2.9e-6 (v3) and 1.9e-6
    (v3n, v32) on CPU, where `auto` resolves to the xla kernel.
    """
    assert block_output(name, CONFIG[SETTINGS[name]], attention_impl="auto") < 2e-5


@pytest.mark.parametrize("name", ["mla_v3", "mla_v3n", "mla_v32"])
def test_auto_and_the_reference_agree_on_the_mla_gradients(name):
    """One backward pass through each implementation on the same weights.

    The trees hold gradients up to 4.7e+01, so the bound is relative to the
    largest of them: tolerance 1e-6, observed 1.6e-7 (v3), 7.7e-8 (v3n) and
    1.6e-7 (v32) on CPU. A crop that kept the padded columns instead of the
    value's own moves these by their own size.
    """
    settings = CONFIG[SETTINGS[name]]
    tensors = fixture(name)
    variables, hidden = block_variables(tensors), jnp.asarray(tensors["hidden"])
    reference = jax.tree.leaves(
        readout_gradients(mla_module(settings), variables, hidden))
    auto = jax.tree.leaves(readout_gradients(
        mla_module(settings, attention_impl="auto"), variables, hidden))
    largest = max(float(np.max(np.abs(np.asarray(leaf)))) for leaf in reference)
    difference = max(float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
                     for a, b in zip(reference, auto))
    assert difference < 1e-6 * largest


def test_the_fused_path_returns_the_value_columns_and_not_the_padding():
    """A value narrower than the query rides into the fused kernel padded to
    the query's width, so the call has to hand back the value's own columns:
    the same numbers as the call where the caller writes that padding out,
    and the reference kernel's answer within fp32 rounding (tolerance 1e-6,
    observed 1.2e-7). A value wider than the query has no such rewrite,
    since padding the query would move the kernel's own 1/sqrt(d) scale, and
    is refused.
    """
    rng = np.random.default_rng(0)
    query = jnp.asarray(rng.standard_normal((2, 7, 4, 16)), jnp.float32)
    key = jnp.asarray(rng.standard_normal((2, 7, 4, 16)), jnp.float32)
    value = jnp.asarray(rng.standard_normal((2, 7, 4, 8)), jnp.float32)
    fused = functools.partial(
        scaled_dot_product_attention, query, key, implementation="xla", causal=True)
    narrow = fused(value)
    assert narrow.shape == (2, 7, 4, 8)
    assert jnp.array_equal(
        narrow, fused(jnp.pad(value, ((0, 0), (0, 0), (0, 0), (0, 8))))[..., :8])
    reference = scaled_dot_product_attention(query, key, value, causal=True)
    assert float(jnp.max(jnp.abs(narrow - reference))) < 1e-6
    with pytest.raises(ValueError):
        fused(jnp.pad(value, ((0, 0), (0, 0), (0, 0), (0, 16))))


def mla_record(settings: dict) -> dict:
    """The fixture's reference config as the `mla` kind's record."""
    return {
        "kind": "mla",
        "q_lora_rank": settings["q_lora_rank"],
        "kv_lora_rank": settings["kv_lora_rank"],
        "qk_nope_head_dim": settings["qk_nope_head_dim"],
        "qk_rope_head_dim": settings["qk_rope_head_dim"],
        "v_head_dim": settings["v_head_dim"],
        "rope_interleave": settings.get("rope_interleave", True),
        "yarn": _yarn_record(
            dict(settings["rope_scaling"], rope_theta=settings["rope_theta"]),
            "rope_scaling", float(settings["rope_theta"]),
            int(settings["max_position_embeddings"])),
        "index_topk": settings.get("index_topk"),
        "index_n_heads": settings.get("index_n_heads"),
        "index_head_dim": settings.get("index_head_dim"),
    }


def mla_model(settings: dict, **overrides) -> CausalTransformer:
    """A one-layer decoder whose mixer is the fixture's block."""
    fields = dict(
        vocab_size=37, emb_features=settings["hidden_size"], num_layers=1,
        num_heads=settings["num_attention_heads"],
        head_dim=settings["qk_nope_head_dim"] + settings["qk_rope_head_dim"],
        mlp_features=48, max_seq_len=64, rope_theta=float(settings["rope_theta"]),
        norm_eps=float(settings["rms_norm_eps"]), qk_norm=False,
        attention_bias=bool(settings["attention_bias"]),
        mixer=mla_record(settings))
    return CausalTransformer(**{**fields, **overrides})


def test_the_mla_record_and_value_agree():
    """`mixer={"kind": "mla", ...}` from a config is the dataclass from code,
    yarn record included."""
    settings = CONFIG["v32"]
    built = mla_model(settings).mixer
    assert built == MLAMixer(
        q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=8, qk_rope_head_dim=8,
        v_head_dim=8, rope_interleave=True, yarn=yarn_of(settings),
        index_topk=4, index_n_heads=2, index_head_dim=16)
    assert mixers["mla"] is MLAMixer


@pytest.mark.parametrize("name", ["mla_v3", "mla_v32"])
def test_a_decoder_on_the_mla_kind_runs_the_reference_block(name):
    """The kind lands the block at layers_0/self_attn with the translated
    tree's names, and there it computes what the standalone block does on
    the same weights, bit for bit, and the fixture's output within the
    block's tolerance. Tolerance 2e-5; observed 2.9e-6 (v3) and 1.9e-6 (v32).
    """
    settings = CONFIG[SETTINGS[name]]
    tensors = fixture(name)
    model = mla_model(settings)
    tokens = jnp.zeros((2, 7), jnp.int32)
    variables = model.init(jax.random.key(0), tokens)
    translated = block_variables(tensors)["params"]
    layer = variables["params"]["layers_0"]
    assert jax.tree_util.tree_structure(layer["self_attn"]) == (
        jax.tree_util.tree_structure(translated))
    variables = {**variables, "params": {
        **variables["params"],
        "layers_0": {**layer, "self_attn": translated}}}
    hidden = jnp.asarray(tensors["hidden"])
    inside = model.apply(
        variables, hidden, method=lambda m, x: m.layers[0].self_attn(x))
    standalone = mla_module(settings).apply(block_variables(tensors), hidden)
    assert jnp.array_equal(inside, standalone)
    assert float(np.max(np.abs(np.asarray(inside) - tensors["output"]))) < 2e-5
    logits = model.apply(variables, jnp.arange(14, dtype=jnp.int32).reshape(2, 7))
    assert logits.shape == (2, 7, 37)


def test_the_mla_kind_refuses_the_dials_it_cannot_honour():
    """A dial the standard attention reads and this kind would drop raises
    at build, naming the dial, and no model is built."""
    settings = CONFIG["v3"]
    tokens = jnp.zeros((1, 4), jnp.int32)
    with pytest.raises(ValueError, match="no attention_scale, v_norm"):
        mla_model(settings, v_norm=True, attention_scale=0.25).init(
            jax.random.key(0), tokens)
    mismatched = dict(mla_record(settings))
    mismatched["yarn"] = dict(mismatched["yarn"], rope_theta=5000.0)
    with pytest.raises(ValueError, match="rope_theta .* disagree"):
        mla_model(settings, mixer=mismatched).init(jax.random.key(0), tokens)


# --------------------------------------------------------------------------
# The stack on the mesh
# --------------------------------------------------------------------------

TINY_SHARD = 256
"""The architecture sweep's threshold: these models hold thousands of
parameters, orders below the production shard threshold."""


def deepseek_stack() -> CausalTransformer:
    """DeepSeek V3.2 at toy width: the sparse mla mixer on both layers, the
    second layer routed with the balancing bias, the group limit and a
    shared expert; the architecture sweep trains the same case."""
    return CausalTransformer(
        vocab_size=64, emb_features=32, num_layers=2, num_heads=2,
        head_dim=16, mlp_features=64, max_seq_len=16,
        mixer={"kind": "mla", "q_lora_rank": 8, "kv_lora_rank": 8,
               "qk_nope_head_dim": 8, "qk_rope_head_dim": 8, "v_head_dim": 8,
               "index_topk": 4, "index_n_heads": 2, "index_head_dim": 16},
        mixture={"experts": 8, "top_k": 2, "layers": (1,),
                 "score_function": "sigmoid", "groups": 4, "groups_per_token": 2,
                 "bias": True, "expert_features": 16, "shared_features": 16})


class RecordingTracker:
    def __init__(self):
        self.scalars = []

    def log(self, scalars, step):
        self.scalars.append(dict(scalars))

    def artifact(self, value, step):
        pass


def token_batches():
    rng = np.random.default_rng(0)
    while True:
        yield {"text": rng.integers(0, 64, size=(8, 17)).astype(np.int32)}


class Data:
    train = staticmethod(token_batches)
    val, batch, records, steps_per_epoch = None, 8, None, None


@pytest.mark.mesh
@pytest.mark.parametrize("fsdp,expert", [(2, 1), (2, 2)])
def test_the_deepseek_stack_fits_on_the_mesh(fsdp, expert):
    """Two steps through the trainer on the simulated eight devices, with
    the parameters two ways over fsdp and, in the second case, the experts
    two ways over the expert axis as well. The loss is finite and falls, the
    balancing bias moves, and the layout check passes: every rank two or
    more parameter of the mixer, the indexer and the experts is placed by
    its declaration, so nothing above the shard threshold stays replicated.
    Expert parallelism alone leaves the mixer replicated by design, which
    test_moe covers, so the expert case rides an fsdp axis."""
    model = deepseek_stack()
    tracker = RecordingTracker()
    trainer = Trainer(
        LMObjective(model, 16, balance_rate=0.01), optax.adam(1e-3),
        key=jax.random.key(0), mesh=MeshSpec(fsdp=fsdp, expert=expert),
        layout=Layout(min_shard=TINY_SHARD), tracker=tracker)

    state = trainer.fit(Data(), steps=2, log_every=1)

    assert dict(trainer.device_mesh.shape) == {
        "data": 8 // (fsdp * expert), "expert": expert, "fsdp": fsdp,
        "tensor": 1, "sequence": 1, "stage": 1}
    losses = [entry["train/loss"] for entry in tracker.scalars if "train/loss" in entry]
    assert len(losses) == 2 and all(np.isfinite(losses)), losses
    assert losses[1] < losses[0], losses
    bias = state.params["moe"]["layers_1"]["mlp"]["gate"]["e_score_correction_bias"]
    assert np.any(np.asarray(bias) != 0), "the bias never moved"
    attention = state.params["params"]["layers_0"]["self_attn"]
    experts = state.params["params"]["layers_1"]["mlp"]["experts"]
    assert attention["kv_b_proj"]["kernel"].sharding.spec == P(None, "fsdp")
    assert attention["indexer"]["wq_b"]["kernel"].sharding.spec == P(None, "fsdp")
    assert experts["gate_proj"]["kernel"].sharding.spec == P(
        "expert" if expert > 1 else None, None, "fsdp")
    abstract = jax.eval_shape(
        model.init, jax.random.key(0), jnp.ones((1, 16), jnp.int32))
    shardings = trainer.layout.shardings(trainer.device_mesh, abstract)
    trainer.layout.check(abstract["params"], shardings["params"], trainer.device_mesh)
