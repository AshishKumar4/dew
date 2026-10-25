"""Tier 3: transformers' own forward, run by torchax: `Pretrained.load` of an unregistered causal LM
no registered convention computes, and `Pretrained.load(..., fallback="torchax")` of any.

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

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
import transformers
from jax.sharding import PartitionSpec as P
from reference_error import (
    assert_as_exact,
    assert_as_exact_as_the_reference,
    equations,
    torch_with_gradients,
    with_gradients,
)

from dew.data.dataset import Loader
from dew.interop import Pretrained
from dew.interop.torchax_fallback import TorchLayout, TransformersTierWarning
from dew.nn.inputs import coordinates
from dew.nn.protocols import Lacks
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.training import MeshSpec, Trainer
from tools.torch_precision import float64

# CI runs this file in tasks of its own: torchax renames torch's privateuse1 backend process-wide.
pytestmark = pytest.mark.torchax

SEQ = 16
ROWS = 8
TINY = {"num_hidden_layers": 2, "hidden_size": 64, "num_attention_heads": 4,
        "intermediate_size": 256, "vocab_size": 256, "max_position_embeddings": 64}
LOGITS = 6e-7
LOSS = 1e-6
GRADIENT = 1.7e-6
"""Twice the worst leaf's max |Δgrad| over its max |grad| measured on CPU,
8.3e-7 at the first layer's post-attention norm bias."""
HEAD_SEEDS, BF16_SEEDS = 512, 512
"""Each site's check-1 seeds, sized by `seeds_for` on armada: the most any of its cases took (the
softcap head's and GPT-NeoX's)."""


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



def token_draws(tokens):
    """Check 1's draw of token rows: `tokens` (the fixture's) at seed 0, else rows of their shape
    drawn uniformly from the vocabulary; and the loss's cotangent, one, a step's."""
    def draw(seed):
        if seed == 0:
            return tokens, np.float32(1)
        rows = np.random.default_rng(seed).integers(0, TINY["vocab_size"], tokens.shape).astype(np.int32)
        return rows, np.float32(1)

    return draw


def loss_and_gradients(loaded, model, params):
    """A draw's LM loss of the fallback-loaded model and its gradient for each parameter `model`
    (its torch twin) names under the draw's cotangent (`with_gradients`), the loss as `[1]`: the
    function check 1 holds against torch's step."""
    objective = LMObjective(loaded, seq_len=SEQ, ema_decay=None)

    @jax.jit
    def step(params, ids, cotangent):
        return with_gradients(lambda params: objective_loss(
            objective, {**loaded.variables, "params": params}, ids, scalar=True))(params, cotangent)

    names = [name for name, _ in model.named_parameters()]

    def dew_fn(ids, cotangent):
        value, grads = step(params, ids, cotangent)
        return {"loss": value[None]} | {name: grads[name] for name in names}

    return dew_fn

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


@pytest.mark.mesh(devices=4)
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
        np.testing.assert_array_equal(got, want)


GEMMA2 = {**TINY, "intermediate_size": 128, "num_key_value_heads": 2, "head_dim": 16, "sliding_window": 32,
          "final_logit_softcapping": 2.0, "query_pre_attn_scalar": 16}
LLAMA = {**TINY, "num_key_value_heads": 2, "tie_word_embeddings": True}
HEADS = {"zero bias": ("TorchCausalLM", "float32"), "bias": ("TorchCausalLM", "float32"),
         "softcap": ("TorchCausalLM", "float32"), "tied": ("TorchTableLM", "float32"),
         "bf16 tied": ("TorchTableLM", "bfloat16")}
"""Each head below, the module it loads as and the dtype its checkpoint stores.
Every one computes in float32; a bf16 checkpoint computing in bf16 is
`test_bf16_compute_gradients_are_as_exact_as_torchs_bf16`'s."""


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


def torch_step(model, tokens, cotangent, stored=torch.float32) -> dict[str, np.ndarray]:
    """transformers' own loss over `tokens` and the gradient of every parameter under the draw's
    `cotangent` (`torch_with_gradients`), rounded to the `stored` dtype a parameter is kept in, as
    float64 arrays. Each parameter enters at float32 (float64 in a float64 model) and is cast to its
    own dtype inside, so a bf16 model's gradients are its own, exactly widened."""
    ids = torch.from_numpy(tokens.astype(np.int64))
    entry = torch.float64 if next(model.parameters()).dtype == torch.float64 else torch.float32
    own = {name: parameter.dtype for name, parameter in model.named_parameters()}
    weights = {name: parameter.detach().to(entry).numpy() for name, parameter in model.named_parameters()}

    def loss(params):
        typed = {name: value.to(own[name]) for name, value in params.items()}
        return torch.func.functional_call(model, typed, (ids,), {"labels": ids}).loss

    value, gradients = torch_with_gradients(loss)(weights, cotangent)
    return {"loss": np.asarray([value], np.float64)} | {
        name: torch.from_numpy(gradient).to(stored).double().numpy() for name, gradient in gradients.items()}


@pytest.mark.parametrize("head", HEADS)
def test_a_head_trains_as_transformers_does_by_what_it_is(head, tokens, tmp_path):
    """A head that adds anything to the product, even a bias that is zero
    now, loads as a model a loss scores whole; a plain head, tied or bf16,
    as one whose head input and weight it scores. Either way LMObjective's
    loss and its gradient of every parameter, the head bias included, are
    as exact as transformers' own over the same stored weights, both measured
    from transformers in float64 (tests/reference_error.py); a bf16 store's
    gradients round to bf16 on both sides, as the parameters they update are
    kept."""
    module, stored = HEADS[head]
    model = head_model(head)
    model.save_pretrained(tmp_path)
    with pytest.warns(UserWarning):
        loaded = Pretrained.load(tmp_path, fallback="torchax", dtype="float32", param_dtype=stored)
    assert type(loaded.model).__name__ == module

    params = {name: jnp.asarray(leaf) for name, leaf in loaded.variables["params"].items()}
    dew_fn = loss_and_gradients(loaded, model, params)

    reference_model = copy.deepcopy(model).float()

    def reference_fn(ids, cotangent):
        return torch_step(reference_model, ids, cotangent, getattr(torch, stored))

    # Truth: the same weights in a model built in float64, its float32 pins
    # (the loss's logits.float(), an RMS norm's upcast) widened and its
    # rotary tables rebuilt.
    with float64():
        wide_model = transformers.AutoModelForCausalLM.from_config(model.config, dtype=torch.float64,
                                                                   attn_implementation="eager").eval()
        wide_model.load_state_dict(model.state_dict())

    def truth_fn(ids, cotangent):
        with float64():
            return torch_step(wide_model, ids, cotangent, torch.float64)

    dew = dew_fn(tokens, np.float32(1))
    assert_as_exact(dew_fn, reference_fn, token_draws(tokens), f"torchax {head} LM step", seeds=HEAD_SEEDS,
                    truth_fn=truth_fn)
    if head in ("zero bias", "bias"):
        assert np.abs(dew["lm_head.bias"]).max() > 0


COHERE = {**TINY, "num_key_value_heads": 2}
"""A tiny Cohere: parallel attention and feed-forward under one bias-free layer norm, rotary pairs
interleaved and a scaled head, a type Dew registers no family for and no registered family's
convention computes."""
UNREGISTERED_SEEDS = 256
"""The unregistered type's check-1 seeds, sized by `seeds_for` on armada."""


def saved_cohere(directory):
    """A tiny Cohere saved to `directory`, and the torch model it holds."""
    torch.manual_seed(0)
    model = transformers.CohereForCausalLM(transformers.CohereConfig(**COHERE)).eval()
    model.save_pretrained(directory)
    return model


def test_an_unregistered_causal_lm_loads_as_transformers_own_forward(tokens, tmp_path):
    """A causal model_type Dew registers no family for, which no registered family's convention
    computes, loads without `fallback` as tier 3, warning so (`TransformersTierWarning`), its every
    layer's attention computed by Dew's, as its layers all ask transformers' attention interface:
    its LM loss and the gradient of every parameter are as exact as transformers' own over the same
    stored weights, measured from transformers in float64 (check 1)."""
    model = saved_cohere(tmp_path)
    with pytest.warns(TransformersTierWarning):
        loaded = Pretrained.load(tmp_path, dtype="float32")
    assert loaded.model.graph.attention_impl == "auto"

    dew_fn = loss_and_gradients(loaded, model, {name: jnp.asarray(leaf)
                                                for name, leaf in loaded.variables["params"].items()})
    reference = copy.deepcopy(model)
    with float64():
        wide = transformers.AutoModelForCausalLM.from_config(model.config, dtype=torch.float64,
                                                             attn_implementation="eager").eval()
        wide.load_state_dict(model.state_dict())

    def truth_fn(ids, cotangent):
        with float64():
            return torch_step(wide, ids, cotangent, torch.float64)

    assert_as_exact(dew_fn, lambda ids, cotangent: torch_step(reference, ids, cotangent), token_draws(tokens),
                    "tier 3 cohere LM step", seeds=UNREGISTERED_SEEDS, truth_fn=truth_fn)


PACKED_SEEDS = {"cohere": 256, "cohere2": 256}
"""Each model's packed rows' check-1 seeds, sized by `seeds_for` on armada."""

PACKED_MODELS = {
    "cohere": lambda: transformers.CohereForCausalLM(transformers.CohereConfig(**COHERE)),
    "cohere2": lambda: transformers.Cohere2ForCausalLM(transformers.Cohere2Config(
        **COHERE, sliding_window=4, layer_types=["sliding_attention", "full_attention"])),
}
"""Unregistered types tier 3 runs on Dew's attention: Cohere's every layer full, Cohere 2's first a
window of 4 keys, which counts each document's real tokens."""


def packed_rows(seed):
    """Two rows of SEQ + 1 token ids: a drawn run of padding first, numbered with the first document but
    no real token, then two documents cut at a drawn place, the second holding a padding slot among
    its tokens; their documents, validity, and the loss's cotangent, one."""
    rng = np.random.default_rng([0xBAC, seed])
    ids = rng.integers(0, TINY["vocab_size"], (2, SEQ + 1)).astype(np.int32)
    places = np.arange(SEQ + 1)[None, :]
    lead = rng.integers(0, 3, (2, 1))
    cuts = rng.integers(lead + 3, SEQ - 5)
    gaps = rng.integers(cuts + 2, SEQ - 1)
    segments = np.where(places < cuts, 1, 2).astype(np.int32)
    valid = (places >= lead) & (places != gaps)
    return ids, segments, valid, np.float32(1)


def documents_alone(model, ids, segments, valid, cotangent, stored=torch.float32):
    """transformers' run of each document's real tokens on its own: the mean next-token loss over the
    targets `LMObjective` scores in the rows, a real token following the real one beside it in the same
    document, and each parameter's gradient under the cotangent (`torch_with_gradients`)."""
    entry = torch.float64 if next(model.parameters()).dtype == torch.float64 else torch.float32
    weights = {name: parameter.detach().to(entry).numpy() for name, parameter in model.named_parameters()}
    runs = []
    for row in range(ids.shape[0]):
        for document in np.unique(segments[row][valid[row]]):
            at = np.flatnonzero((segments[row] == document) & valid[row])
            run = torch.from_numpy(ids[row, at].astype(np.int64))[None]
            runs.append((run, torch.from_numpy(np.flatnonzero(np.diff(at) == 1))))

    def loss(params):
        total = sum(torch.nn.functional.cross_entropy(
            torch.func.functional_call(model, params, (run,)).logits[0, :-1][scored], run[0, 1:][scored],
            reduction="sum") for run, scored in runs)
        return total / sum(len(scored) for _, scored in runs)

    value, gradients = torch_with_gradients(loss)(weights, cotangent)
    return {"loss": np.asarray([value], np.float64)} | {
        name: torch.from_numpy(gradient).to(stored).double().numpy() for name, gradient in gradients.items()}


@pytest.mark.parametrize("family", sorted(PACKED_MODELS))
def test_a_packed_row_trains_each_document_as_transformers_runs_it_alone(family, tmp_path):
    """On Dew's attention a tier-3 model trains on packed rows with padding: each document attends
    within its real tokens, a window counting them alone, and rotates from its own first, so
    `LMObjective`'s loss over the rows and every parameter's gradient are as exact as transformers'
    run of each document's real tokens on its own, measured from that run in float64 (check 1)."""
    torch.manual_seed(0)
    model = PACKED_MODELS[family]().eval()
    model.save_pretrained(tmp_path)
    with pytest.warns(TransformersTierWarning):
        loaded = Pretrained.load(tmp_path, dtype="float32")
    assert loaded.model.graph.attention_impl == "auto"
    objective = LMObjective(loaded, seq_len=SEQ, ema_decay=None)
    names = [name for name, _ in model.named_parameters()]

    @jax.jit
    def stepped(params, ids, segments, valid, cotangent):
        def loss(params):
            laid = coordinates(segments=segments, valid=valid)
            stats, _ = objective.loss({**loaded.variables, "params": params},
                                      {"text": ids, "text_coordinates": laid},
                                      Step(step=jnp.int32(0), key=jax.random.key(1), ema=None))
            return objective.reduce_loss(stats)[0]
        value, gradients = with_gradients(loss)(params, cotangent)
        return {"loss": value[None]} | {name: gradients[name] for name in names}

    def ours(ids, segments, valid, cotangent):
        return stepped(loaded.variables["params"], jnp.asarray(ids), jnp.asarray(segments),
                       jnp.asarray(valid), cotangent)

    reference = copy.deepcopy(model)
    with float64():
        wide = transformers.AutoModelForCausalLM.from_config(model.config, dtype=torch.float64,
                                                             attn_implementation="eager")
        wide.load_state_dict(model.state_dict())

    def truth(ids, segments, valid, cotangent):
        with float64():
            return documents_alone(wide, ids, segments, valid, cotangent, torch.float64)

    def theirs(ids, segments, valid, cotangent):
        return documents_alone(reference, ids, segments, valid, cotangent)

    assert_as_exact(ours, theirs, packed_rows, f"tier 3 {family} packed rows", seeds=PACKED_SEEDS[family],
                    truth_fn=truth)


def test_a_training_call_that_drops_attention_needs_a_dropout_rng(tmp_path):
    """Dew's attention draws a training call's attention dropout from the call's 'dropout' rng, so one
    given none is refused rather than run without the dropout the model asks for."""
    torch.manual_seed(0)
    config = transformers.CohereConfig(**COHERE, attention_dropout=0.1)
    transformers.CohereForCausalLM(config).save_pretrained(tmp_path)
    with pytest.warns(TransformersTierWarning):
        loaded = Pretrained.load(tmp_path, dtype="float32")
    ids = jnp.asarray(packed_rows(0)[0])
    with pytest.raises(ValueError):
        loaded.model.apply(loaded.variables, ids, train=True, method="logits")
    trained = loaded.model.apply(loaded.variables, ids, train=True, method="logits",
                                 rngs={"dropout": jax.random.key(0)})
    assert np.all(np.isfinite(np.asarray(trained)))


DROPPING_SEEDS = 8
"""The dropping call's check-1 seeds, sized by `seeds_for` on armada."""


def test_a_training_call_that_drops_attention_reads_each_documents_real_tokens(tmp_path):
    """A training call that drops attention probabilities reads the keys an eval call does: over packed
    rows with padding among their tokens, Cohere 2's window counts each document's real tokens.
    Dropping one probability in a billion keeps every one in all but a billionth of draws, so the
    training call's logits on the real tokens are as exact as transformers' eval of each document's
    real tokens on its own, from its float64 run (check 1)."""
    torch.manual_seed(0)
    model = transformers.Cohere2ForCausalLM(transformers.Cohere2Config(
        **COHERE, sliding_window=4, layer_types=["sliding_attention", "full_attention"],
        attention_dropout=1e-9)).eval()
    model.save_pretrained(tmp_path)
    with pytest.warns(TransformersTierWarning):
        loaded = Pretrained.load(tmp_path, dtype="float32")

    @jax.jit
    def trained(ids, segments, valid):
        logits = loaded.model.apply(loaded.variables, ids, train=True, segment_ids=segments,
                                    attention_mask=valid, method="logits",
                                    rngs={"dropout": jax.random.key(5)})
        return {"logits": jnp.where(valid[..., None], logits, 0)}

    def ours(ids, segments, valid, _):
        return trained(jnp.asarray(ids), jnp.asarray(segments), jnp.asarray(valid))

    def alone(model, ids, segments, valid):
        logits = np.zeros((*ids.shape, TINY["vocab_size"]))
        with torch.no_grad():
            for row in range(ids.shape[0]):
                for document in np.unique(segments[row][valid[row]]):
                    at = np.flatnonzero((segments[row] == document) & valid[row])
                    run = torch.from_numpy(ids[row, at].astype(np.int64))[None]
                    logits[row, at] = model(run).logits[0].double()
        return {"logits": logits}

    with float64():
        wide = transformers.AutoModelForCausalLM.from_config(model.config, dtype=torch.float64,
                                                             attn_implementation="eager").eval()
        wide.load_state_dict(model.state_dict())

    def truth(ids, segments, valid, _):
        with float64():
            return alone(wide, ids, segments, valid)

    assert_as_exact(ours, lambda ids, segments, valid, _: alone(model, ids, segments, valid), packed_rows,
                    "tier 3 cohere2 dropping call", seeds=DROPPING_SEEDS, truth_fn=truth)


def test_a_type_whose_attention_asks_no_interface_runs_its_own_and_reads_no_packing(tmp_path):
    """OpenAI GPT's layers compute their attention in their own code, asking transformers' attention
    interface nothing: tier 3 runs that attention as transformers does, so it names no Dew kernel
    (`attention_impl`) and reads no packed rows (`Lacks`)."""
    torch.manual_seed(0)
    config = transformers.OpenAIGPTConfig(n_embd=64, n_layer=2, n_head=4, vocab_size=TINY["vocab_size"],
                                          n_positions=64)
    transformers.OpenAIGPTLMHeadModel(config).save_pretrained(tmp_path)
    with pytest.warns(TransformersTierWarning):
        loaded = Pretrained.load(tmp_path, dtype="float32")
    assert loaded.model.graph.attention_impl is None
    ids, segments, _, _ = packed_rows(0)
    with pytest.raises(Lacks):
        loaded.model.apply(loaded.variables, jnp.asarray(ids), segment_ids=jnp.asarray(segments),
                           method="logits")
    # Refused before the load warns of its tier.
    with pytest.raises(ValueError):
        Pretrained.load(tmp_path, dtype="float32", attention_impl="reference")
@pytest.mark.parametrize("route", ["tier 3", "fallback"])
def test_a_load_without_weights_reads_none_and_states_their_shapes(route, tmp_path):
    """Without its weights a tier-3 source is built from its config alone, its weight files never read
    (here removed), and its variables state every leaf's shape and dtype as the load with weights
    holds it: what a run resuming from its own checkpoint builds its state around. The explicit route
    and the automatic one read it alike."""
    import shutil

    torch.manual_seed(0)
    transformers.CohereForCausalLM(transformers.CohereConfig(**COHERE)).save_pretrained(tmp_path / "whole")
    options = {"fallback": "torchax"} if route == "fallback" else {}
    with pytest.warns(TransformersTierWarning):
        whole = Pretrained.load(tmp_path / "whole", dtype="float32", **options)
    shutil.copytree(tmp_path / "whole", tmp_path / "shapes")
    for weights in (tmp_path / "shapes").glob("*.safetensors"):
        weights.unlink()
    with pytest.warns(TransformersTierWarning):
        shapes = Pretrained.load(tmp_path / "shapes", dtype="float32", weights=False, **options)
    assert type(shapes.model) is type(whole.model)
    assert jax.tree.map(lambda leaf: (leaf.shape, leaf.dtype), shapes.variables) == jax.tree.map(
        lambda leaf: (leaf.shape, jnp.dtype(leaf.dtype)), whole.variables)
    assert all(isinstance(leaf, jax.ShapeDtypeStruct) for leaf in jax.tree.leaves(shapes.variables))


_SHAPES = """
import json
import sys
import warnings
import jax
from dew.interop import Pretrained
warnings.simplefilter("ignore")
loaded = Pretrained.load("dew/tiny-cohere", dtype="float32", fallback="torchax", weights=False)
print(json.dumps(jax.tree.map(lambda leaf: [list(leaf.shape), str(leaf.dtype)], loaded.variables)))
"""


def test_a_shapes_only_fallback_load_reads_no_weights_from_the_hub(tmp_path):
    """`fallback='torchax'` without weights reads metadata alone: from a Hub cache holding the
    commit's metadata and not its weight files, offline, it states every leaf's shape and dtype
    as the load with weights holds it, as a run resuming from its own checkpoint needs."""
    import json
    import os
    import subprocess
    import sys

    from interop_support import hub_cache

    torch.manual_seed(0)
    transformers.CohereForCausalLM(transformers.CohereConfig(**COHERE)).save_pretrained(tmp_path / "whole")
    with pytest.warns(TransformersTierWarning):
        whole = Pretrained.load(tmp_path / "whole", dtype="float32", fallback="torchax")
    hub_cache(tmp_path / "hub", "dew/tiny-cohere", tmp_path / "whole")
    for weights in (tmp_path / "hub").rglob("*.safetensors"):
        weights.unlink()
    env = {**os.environ, "HF_HUB_CACHE": str(tmp_path / "hub"), "HF_HUB_OFFLINE": "1", "JAX_PLATFORMS": "cpu"}
    run = subprocess.run([sys.executable, "-c", _SHAPES], capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr[-2000:]
    assert json.loads(run.stdout.splitlines()[-1]) == jax.tree.map(
        lambda leaf: [list(leaf.shape), str(jnp.dtype(leaf.dtype))], whole.variables)


@pytest.mark.parametrize("route", ["tier 3", "fallback"])
def test_a_selected_folder_loads_its_own_model_through_tier_3(route, tmp_path):
    """A causal LM stored in a folder below the source's root, selected by `subfolder`, loads the
    folder's own weights through tier 3, whichever route takes it: the logits are the folder
    model's as loaded from the folder itself."""
    torch.manual_seed(0)
    model = transformers.CohereForCausalLM(transformers.CohereConfig(**COHERE))
    model.save_pretrained(tmp_path / "root" / "lm")
    options = {"fallback": "torchax"} if route == "fallback" else {}
    with pytest.warns(TransformersTierWarning):
        selected = Pretrained.load(tmp_path / "root", subfolder="lm", dtype="float32", **options)
    with pytest.warns(TransformersTierWarning):
        direct = Pretrained.load(tmp_path / "root" / "lm", dtype="float32", **options)
    ids = jnp.asarray(np.random.default_rng(0).integers(0, COHERE["vocab_size"], (2, 8)), jnp.int32)
    np.testing.assert_array_equal(selected.model.apply(selected.variables, ids),
                                  direct.model.apply(direct.variables, ids))


@pytest.mark.parametrize("family", ["llama", "neox"])
def test_bf16_compute_gradients_are_as_exact_as_torchs_bf16(family, tokens, tmp_path):
    """A bf16 checkpoint computing in bf16: LMObjective's loss and its gradient of every parameter
    are as exact as transformers' bf16 loss and torch autograd's gradient over the same weights,
    each measured from torch in float64 (tests/reference_error.py). A norm weight's and a bias's
    gradient is the sum over the tokens of a broadcast's cotangent, which jax sums in the
    broadcast value's dtype (jax-ml/jax#30310) and torch in fp32; summed in bf16 they sat 2.5-3
    and 3.7-4.7 times torch's distance. torchax lowers gelu, silu, the softmaxes and the layer
    norms into a bf16 rounding a step, where torch rounds each once (`_OPMATH`): the losses sat
    1.5e-4 and 1.7e-4 from float64 against torch's 3.0e-5 and 2.1e-5, and at torch's opmath
    5.4e-5 and 4.1e-5."""
    torch.manual_seed(0)
    if family == "llama":
        model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**LLAMA))
    else:
        model = transformers.GPTNeoXForCausalLM(transformers.GPTNeoXConfig(**TINY))
    model = model.eval().to(torch.bfloat16)
    model.save_pretrained(tmp_path)
    with pytest.warns(UserWarning):
        loaded = Pretrained.load(tmp_path, fallback="torchax", dtype="bfloat16", param_dtype="bfloat16")

    dew_fn = loss_and_gradients(loaded, model, loaded.variables["params"])

    reference_model = copy.deepcopy(model)

    def reference_fn(ids, cotangent):
        return torch_step(reference_model, ids, cotangent, torch.bfloat16)

    with float64():
        wide = transformers.AutoModelForCausalLM.from_config(model.config, dtype=torch.float64,
                                                             attn_implementation="eager").eval()
        wide.load_state_dict(model.state_dict())

    def truth_fn(ids, cotangent):
        with float64():
            return torch_step(wide, ids, cotangent, torch.float64)

    assert_as_exact(dew_fn, reference_fn, token_draws(tokens), f"torchax bf16 {family} LM step",
                    seeds=BF16_SEEDS, truth_fn=truth_fn)


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


def tied_float32(tmp_path):
    """Llama's tied head stored and computing in float32, as `TorchTableLM` loads it."""
    head_model("tied").save_pretrained(tmp_path)
    with pytest.warns(UserWarning):
        loaded = Pretrained.load(tmp_path, fallback="torchax", dtype="float32")
    assert type(loaded.model).__name__ == "TorchTableLM"
    return loaded


def test_a_float32_forward_and_its_output_table_multiply_float32_products(tokens, tmp_path):
    """A float32 torchax model multiplies every product of its forward, torch's operators that
    state no precision among them, at HIGHEST, as its output table does, under JAX's own default
    precision: the two reads of the same logits take the same arithmetic, where the forward ran
    TF32 on a GPU and the table float32 products."""
    loaded = tied_float32(tmp_path)
    ids = jnp.asarray(tokens[:, :SEQ])
    highest = (jax.lax.Precision.HIGHEST, jax.lax.Precision.HIGHEST)
    for read in (lambda variables: loaded.model.apply(variables, ids),
                 lambda variables: loaded.model.apply(variables, method="output_table").logits(
                     loaded.model.apply(variables, ids, method="hidden_states"))):
        with jax.default_matmul_precision(None):
            graph = jax.make_jaxpr(read)(loaded.variables)
        products = [equation.params["precision"] for equation in equations(graph)
                    if equation.primitive.name == "dot_general"]
        assert products and all(precision == highest for precision in products), products


def test_a_float64_forwards_logits_are_as_wide_as_its_output_tables(tokens, tmp_path):
    """Under x64, a float64 torchax model's whole logits stay float64, as its output table's
    logits over the same states do (`at_least_fp32`): neither read narrows to float32."""
    loaded = tied_float32(tmp_path)
    ids = jnp.asarray(tokens[:, :SEQ])
    with jax.enable_x64(new_val=True):
        model = loaded.model.clone(dtype=jnp.float64)
        variables = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64)
                                 if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating) else leaf,
                                 loaded.variables)
        direct = model.apply(variables, ids)
        hidden = model.apply(variables, ids, method="hidden_states")
        table = model.apply(variables, method="output_table").logits(hidden)
    assert direct.dtype == table.dtype == jnp.float64


@pytest.mark.gpu(reason="TF32 is a GPU's default float32 product")
def test_a_float32_forwards_logits_and_its_output_tables_are_as_exact_as_torchs(tokens, tmp_path):
    """On a GPU, a float32 torchax model's own logits and its output table's over the same
    states are each as exact as torch's float32 forward, from torch's in float64, under JAX's own
    default precision (the test lane sets 'highest' for every product that states none)."""
    loaded = tied_float32(tmp_path)
    ids = tokens[:, :SEQ]
    model = head_model("tied")
    with jax.default_matmul_precision(None):
        direct = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(ids)), np.float64)
        hidden = loaded.model.apply(loaded.variables, jnp.asarray(ids), method="hidden_states")
        table = np.asarray(loaded.model.apply(loaded.variables, method="output_table").logits(hidden),
                           np.float64)
    reference = torch_logits(model, ids).astype(np.float64)
    with float64(), torch.no_grad():
        wide = transformers.AutoModelForCausalLM.from_config(model.config, dtype=torch.float64,
                                                             attn_implementation="eager").eval()
        wide.load_state_dict(model.state_dict())
        truth = wide(torch.from_numpy(np.asarray(ids, np.int64))).logits.numpy()
    assert_as_exact_as_the_reference(direct, reference, truth, "the forward's logits")
    assert_as_exact_as_the_reference(table, reference, truth, "the output table's logits")
