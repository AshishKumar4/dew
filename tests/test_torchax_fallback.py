"""Tier 3: `Pretrained.load(..., fallback="torchax")` over transformers' own forward.

The offline cases build a tiny random GPT-NeoX, an architecture Dew has no
family for and whose rotary `inv_freq` is a buffer, save it the way
transformers saves any checkpoint, and load it back through the fallback.
The network case runs the released GPT-2, pinned to a commit.

Bounds. The fp32 forward is transformers' own op sequence run by torchax as
XLA ops, so only summation order separates the two. On the tiny model over
its fixed rows the logits differ by at most 2.8e-7 (magnitude 0.67), and
`LOGITS` is about twice that. The objective's loss (5.561) scores the
captured head input through `chunked_cross_entropy`'s own logsumexp and
lands one fp32 ulp (4.8e-7) from torch's; `LOSS` is two.
"""

import copy
import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
import transformers
from jax.sharding import PartitionSpec as P
from reference_error import assert_leaves_as_exact, float64_terms, widened

from dew.data.dataset import Loader
from dew.interop import Pretrained
from dew.interop.torchax_fallback import TorchLayout
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.training import MeshSpec, Trainer
from tools.diffusers_wan_reference import float64

# CI runs this file in tasks of its own: torchax renames torch's privateuse1 backend process-wide.
pytestmark = pytest.mark.torchax

pytest.importorskip("torchax")

SEQ = 16
ROWS = 8
TINY = {"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4,
        "intermediate_size": 256, "vocab_size": 256, "max_position_embeddings": 64}
LOGITS = 6e-7
LOSS = 1e-6
GRADIENT = 1.7e-6
"""Twice the worst leaf's max |Δgrad| over its max |grad| measured on CPU,
8.3e-7 at the first layer's post-attention norm bias."""


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """A random GPT-NeoX checkpoint on disk, and the torch model that wrote it."""
    torch.manual_seed(0)
    model = transformers.GPTNeoXForCausalLM(transformers.GPTNeoXConfig(**TINY)).eval()
    directory = tmp_path_factory.mktemp("neox")
    model.save_pretrained(directory)
    return directory, model


@pytest.fixture(scope="module")
def loaded(source):
    with pytest.warns(UserWarning):
        return Pretrained.load(source[0], fallback="torchax", dtype="float32")


@pytest.fixture(scope="module")
def tokens():
    return np.random.RandomState(0).randint(0, TINY["vocab_size"], (ROWS, SEQ + 1)).astype(np.int32)


def torch_logits(model, ids) -> np.ndarray:
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(ids, np.int64))).logits.float().numpy()


def objective_loss(objective, variables, tokens, *, scalar: bool = False):
    stats, _ = objective.loss(variables, {"text": tokens},
                              Step(step=jnp.int32(0), key=jax.random.key(1), ema=None))
    loss = objective.reduce_loss(stats)[0]
    return loss if scalar else float(loss)


def test_the_logits_and_the_objective_loss_are_transformers(source, loaded, tokens):
    """`apply` is the torch forward, and LMObjective's chunked loss over the
    captured head input and the head weight is transformers' own loss."""
    _, model = source
    logits = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(tokens)))
    assert np.abs(logits - torch_logits(model, tokens)).max() <= LOGITS
    objective = LMObjective(loaded, seq_len=SEQ, ema_decay=None)
    ids = torch.from_numpy(tokens.astype(np.int64))
    with torch.no_grad():
        expected = float(model(ids, labels=ids).loss)
    assert abs(objective_loss(objective, loaded.variables, tokens) - expected) <= LOSS


def test_the_gradients_are_transformers(source, loaded, tokens):
    """jax.grad of LMObjective's loss is torch autograd of transformers' own
    loss, leaf for leaf, so every path to the head and the embedding trains."""
    _, model = source
    objective = LMObjective(loaded, seq_len=SEQ, ema_decay=None)
    ids = torch.from_numpy(tokens.astype(np.int64))
    model.zero_grad()
    model(ids, labels=ids).loss.backward()
    grads = jax.grad(lambda params: objective_loss(objective, {**loaded.variables, "params": params}, tokens,
                                                   scalar=True))(
        {name: jnp.asarray(leaf) for name, leaf in loaded.variables["params"].items()})
    for name, parameter in model.named_parameters():
        expected = parameter.grad.numpy()
        assert np.abs(np.asarray(grads[name]) - expected).max() <= GRADIENT * np.abs(expected).max(), name


@pytest.mark.mesh
def test_trainer_steps_move_the_params_and_place_them_by_name(loaded, tokens):
    """AdamW through the Trainer on a fsdp x tensor mesh: the loss falls,
    every parameter moves, the rotary buffers come back bitwise, and the
    torch names take the specs the name table gives them."""
    params, buffers = loaded.variables["params"], loaded.variables["buffers"]
    assert set(buffers) == {"gpt_neox.rotary_emb.inv_freq", "gpt_neox.rotary_emb.original_inv_freq"}
    assert not set(params) & set(buffers)
    objective = LMObjective(loaded, seq_len=SEQ, ema_decay=None)
    rows = math.lcm(ROWS, jax.device_count())
    batch = tokens[np.arange(rows) % ROWS]
    data = Loader(train=lambda partition: iter([{"text": batch}] * 4), val=None, records=rows, batch=rows)
    trainer = Trainer(objective, optax.adamw(1e-2), key=jax.random.key(0),
                      mesh=MeshSpec(fsdp=2, tensor=2), layout=TorchLayout(min_shard=1))

    state = trainer.fit(data, steps=4, log_every=1)

    assert objective_loss(objective, state.variables, batch) < objective_loss(
        objective, loaded.variables, batch)
    moved = state.variables["params"]
    assert all(bool(jnp.any(jnp.asarray(params[name]) != moved[name])) for name in params)
    for name, buffer in buffers.items():
        np.testing.assert_array_equal(np.asarray(state.variables["buffers"][name]), buffer)
    layer = "gpt_neox.layers.0"
    assert moved[f"{layer}.attention.query_key_value.weight"].sharding.spec == P("tensor", "fsdp")
    # The row-parallel projection, [embed, mlp] in torch: the residual width
    # takes fsdp and the 4h width the tensor axis (8a6ff3e8's Megatron 2-D).
    assert moved[f"{layer}.mlp.dense_4h_to_h.weight"].sharding.spec == P("fsdp", "tensor")
    assert moved["gpt_neox.embed_in.weight"].sharding.spec == P(("fsdp", "tensor"))


def test_a_frozen_torch_base_kept_whole_drops_fsdp_from_the_name_tables_placement(loaded, tokens):
    """`Layout.frozen_whole` reaches a torch-named tree: TorchLayout's name table picks each
    tensor's axes, and the one placement keeps the frozen tensors whole over fsdp, split over the
    tensor axis as the table splits them, while the trained ones, their moments and one step's
    update lie as the split layout lays them."""
    from dew.nn.sharding import mesh_axes
    from dew.objectives.base import FROZEN, freeze
    from dew.training.distributed import shard_batch

    variables = freeze(loaded.variables, lambda path: ".layers.1." in path[1])
    mesh = MeshSpec(fsdp=2, tensor=2)
    rows = math.lcm(ROWS, jax.device_count())
    batch = {"text": tokens[np.arange(rows) % ROWS]}

    def stepped(frozen_whole: bool):
        objective = LMObjective(loaded, seq_len=SEQ, ema_decay=None, variables=variables)
        trainer = Trainer(objective, optax.adamw(1e-2), key=jax.random.key(0), mesh=mesh, checkpoints=None,
                          tracker=None, layout=TorchLayout(min_shard=1, frozen_whole=frozen_whole))
        trainer.device_mesh = mesh.build(jax.devices()[:4])
        state, shardings, _ = trainer.place()
        placed = shard_batch(trainer.device_mesh, batch)
        trained, *_ = trainer.compile(state, placed)(state, placed)
        return shardings, jax.device_get(trained.variables["params"])

    def unsplit(entry):
        kept = [axis for axis in mesh_axes(entry) if axis != "fsdp"]
        return tuple(kept) if kept else None

    split, split_trained = stepped(frozen_whole=False)
    whole, whole_trained = stepped(frozen_whole=True)
    split_frozen = {name: sharding.spec for name, sharding in split.variables[FROZEN].items()}
    whole_frozen = {name: sharding.spec for name, sharding in whole.variables[FROZEN].items()}
    assert any("fsdp" in mesh_axes(entry) for spec in split_frozen.values() for entry in spec)
    assert whole_frozen == {name: P(*map(unsplit, spec)) for name, spec in split_frozen.items()}
    assert whole_frozen["gpt_neox.layers.0.attention.query_key_value.weight"] == P("tensor", None)
    for field in ("params", "opt_state"):
        tree = (lambda placement: placement.variables["params"]) if field == "params" else (
            lambda placement: placement.opt_state)
        assert [sharding.spec for sharding in jax.tree.leaves(tree(whole))] == [
            sharding.spec for sharding in jax.tree.leaves(tree(split))]
    for got, want in zip(jax.tree.leaves(whole_trained), jax.tree.leaves(split_trained), strict=True):
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)


GEMMA2 = {**TINY, "intermediate_size": 128, "num_key_value_heads": 2, "head_dim": 16, "sliding_window": 32,
          "final_logit_softcapping": 2.0, "query_pre_attn_scalar": 16}
LLAMA = {**TINY, "num_key_value_heads": 2, "tie_word_embeddings": True}
HEADS = {"zero bias": ("TorchCausalLM", "float32"), "bias": ("TorchCausalLM", "float32"),
         "softcap": ("TorchCausalLM", "float32"), "tied": ("TorchTableLM", "float32"),
         "bf16 tied": ("TorchTableLM", "bfloat16")}
"""Each head below, the module it loads as and the dtype its checkpoint stores.
Every one computes in float32. Over bf16 compute torchax's XLA ops and torch's
part by their own bf16 roundings, which a scalar loss over 128 tokens cannot
average out (measured: logits 8.3e-4 and 8.0e-4 from float64 in RMS, the
losses 1.5e-4 and 3.0e-5)."""


def head_model(head):
    """A random causal LM: Phi's lm_head with its bias as initialized (zero)
    or drawn, Gemma 2's softcapped head, or Llama's head tied to its
    embedding, a plain product, stored in float32 or bf16."""
    torch.manual_seed(0)
    if head == "softcap":
        return transformers.Gemma2ForCausalLM(transformers.Gemma2Config(**GEMMA2)).eval()
    if head.endswith("tied"):
        model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**LLAMA)).eval()
        return model.to(torch.bfloat16) if head.startswith("bf16") else model
    model = transformers.PhiForCausalLM(transformers.PhiConfig(**TINY)).eval()
    if head == "bias":
        with torch.no_grad():
            model.lm_head.bias.normal_(0, 0.5)
    return model


def torch_step(model, tokens, stored=torch.float32) -> dict[str, np.ndarray]:
    """transformers' own loss over `tokens` and torch autograd's gradient of
    every parameter, rounded to the `stored` dtype a parameter is kept in,
    as float64 arrays."""
    ids = torch.from_numpy(tokens.astype(np.int64))
    model.zero_grad()
    loss = model(ids, labels=ids).loss
    loss.backward()
    return {"loss": np.asarray([loss.item()])} | {
        name: parameter.grad.to(stored).double().numpy() for name, parameter in model.named_parameters()}


@pytest.mark.parametrize("head", HEADS)
def test_a_head_trains_as_transformers_does_by_what_it_is(head, tokens, tmp_path):
    """A head that adds anything to the product, even a bias that is zero
    now, loads as a model a loss scores whole; a plain head, tied or bf16,
    as one whose head input and weight it scores. Either way LMObjective's
    loss and its gradient of every parameter, the head bias included, are
    as exact as transformers' own over the same stored weights, both measured
    from transformers in float64 (tests/reference_error.py); a bf16 store's
    gradients round to bf16 on both sides, as the parameters they update are
    kept. A leaf whose sum cancels by design (a key bias's gradient, under the
    softmax's shift invariance) is sized by its float64 terms, which Dew's own
    forward sums in float64."""
    module, stored = HEADS[head]
    model = head_model(head)
    model.save_pretrained(tmp_path)
    with pytest.warns(UserWarning):
        loaded = Pretrained.load(tmp_path, fallback="torchax", dtype="float32", param_dtype=stored)
    assert type(loaded.model).__name__ == module

    def step(source):
        objective = LMObjective(source, seq_len=SEQ, ema_decay=None)
        loss = jax.value_and_grad(lambda params: objective_loss(
            objective, {**source.variables, "params": params}, tokens, scalar=True))
        return loss, {name: jnp.asarray(leaf) for name, leaf in source.variables["params"].items()}

    names = [name for name, _ in model.named_parameters()]
    loss, params = step(loaded)
    value, grads = loss(params)
    dew = {"loss": np.asarray([value], np.float64)} | {
        name: np.asarray(grads[name], np.float64) for name in names}
    reference = torch_step(copy.deepcopy(model).float(), tokens, getattr(torch, stored))
    # Truth: the same weights in a model built in float64, its float32 pins
    # (the loss's logits.float(), an RMS norm's upcast) widened and its
    # rotary tables rebuilt; Dew's own float64 terms under the same widening.
    with float64():
        wide_model = transformers.AutoModelForCausalLM.from_config(model.config, dtype=torch.float64,
                                                                   attn_implementation="eager").eval()
        wide_model.load_state_dict(model.state_dict())
        truth = torch_step(wide_model, tokens)
        with jax.enable_x64(new_val=True):
            wide = replace(loaded, model=loaded.model.clone(dtype=jnp.float64),
                           variables=widened(loaded.variables))
            loss, params = step(wide)
            _, (loss_terms, grad_terms) = float64_terms(loss, params)
    terms = {"loss": np.asarray([loss_terms])} | {name: np.asarray(grad_terms[name]) for name in names}
    assert_leaves_as_exact(dew, reference, truth, reductions=ROWS * SEQ, terms=terms)
    if head in ("zero bias", "bias"):
        assert np.abs(dew["lm_head.bias"]).max() > 0


def test_save_writes_the_source_names_back(source, loaded, tokens, tmp_path):
    """The saved directory is a transformers checkpoint of the given weights."""
    variables = {**loaded.variables, "params": {name: np.asarray(leaf) * 0.5
                                                 for name, leaf in loaded.variables["params"].items()}}
    loaded.save(tmp_path, variables=variables)
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    logits = np.asarray(loaded.model.apply(variables, jnp.asarray(tokens)))
    assert np.abs(logits - torch_logits(reloaded, tokens)).max() <= LOGITS


@pytest.mark.parametrize("selector", ["gguf_file", "single_file", "dduf_file"])
def test_a_payload_selector_is_refused_where_the_fallback_reads_the_repos_own_weights(source, selector):
    """The fallback builds the transformers model the repo's config.json and
    weights describe. A GGUF, single-file or DDUF selector names another
    payload, so it is refused rather than dropped."""
    with pytest.raises(ValueError):
        Pretrained.load(source[0], fallback="torchax", dtype="float32", **{selector: "model.bin"})


def test_the_output_table_scores_as_the_bf16_forward_whatever_dtype_the_states_arrive_in(source, tokens):
    """In a bf16 forward the head multiplies bf16 states by the bf16 weight and
    rounds the logits to bf16. The output table carries that compute dtype,
    so states handed to it in fp32, as a loss may hold them, score as the
    forward's own logits do: bf16 values, each within one bf16 step of the
    forward's, which sums the same products in another order."""
    loaded = Pretrained.load(source[0], fallback="torchax", dtype="bfloat16")
    table = loaded.model.apply(loaded.variables, method="output_table")
    hidden = loaded.model.apply(loaded.variables, jnp.asarray(tokens), method="hidden_states")
    logits = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(tokens)))
    scored = np.asarray(table.logits(hidden.astype(jnp.float32)), np.float32)
    np.testing.assert_array_equal(scored, np.asarray(jnp.asarray(scored, jnp.bfloat16), np.float32))
    # A bf16 value is an fp32 one's upper 16 bits, so neighbours differ by one there.
    apart = (scored.view(np.int32) >> 16) - (logits.astype(np.float32).view(np.int32) >> 16)
    assert np.abs(apart).max() <= 1
