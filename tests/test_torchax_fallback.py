"""Tier 3: `load_pretrained(..., fallback="torchax")` over transformers' own forward.

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

import json
import math
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from jax.sharding import PartitionSpec as P

from dew.data.dataset import Dataset
from dew.interop import load_pretrained
from dew.interop.torchax_fallback import TorchLayout
from dew.objectives.base import Step
from dew.objectives.lm import LMObjective
from dew.training import MeshSpec, Trainer

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
    with pytest.warns(UserWarning, match="tier 3"):
        return load_pretrained(source[0], fallback="torchax", dtype="float32")


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
    objective = LMObjective(loaded.model, seq_len=SEQ, ema_decay=None, pretrained=loaded.variables)
    ids = torch.from_numpy(tokens.astype(np.int64))
    with torch.no_grad():
        expected = float(model(ids, labels=ids).loss)
    assert abs(objective_loss(objective, loaded.variables, tokens) - expected) <= LOSS


def test_the_gradients_are_transformers(source, loaded, tokens):
    """jax.grad of LMObjective's loss is torch autograd of transformers' own
    loss, leaf for leaf, so every path to the head and the embedding trains."""
    _, model = source
    objective = LMObjective(loaded.model, seq_len=SEQ, ema_decay=None, pretrained=loaded.variables)
    ids = torch.from_numpy(tokens.astype(np.int64))
    model.zero_grad()
    model(ids, labels=ids).loss.backward()
    grads = jax.grad(lambda params: objective_loss(objective, {**loaded.variables, "params": params}, tokens,
                                                   scalar=True))(
        {name: jnp.asarray(leaf) for name, leaf in loaded.variables["params"].items()})
    for name, parameter in model.named_parameters():
        expected = parameter.grad.numpy()
        assert np.abs(np.asarray(grads[name]) - expected).max() <= GRADIENT * np.abs(expected).max(), name


def test_a_load_puts_nothing_on_a_device(source):
    """The weights come to the host as views of torch's storage; only the
    caller's placement moves them, so a model larger than one device loads.
    Any transfer to a device during the load raises under the guard."""
    with jax.transfer_guard_host_to_device("disallow"), pytest.warns(UserWarning, match="tier 3"):
        load_pretrained(source[0], fallback="torchax", dtype="float32")


@pytest.mark.mesh
def test_trainer_steps_move_the_params_and_place_them_by_name(loaded, tokens):
    """AdamW through the Trainer on a fsdp x tensor mesh: the loss falls,
    every parameter moves, the rotary buffers come back bitwise, and the
    torch names take the specs the name table gives them."""
    params, buffers = loaded.variables["params"], loaded.variables["buffers"]
    assert set(buffers) == {"gpt_neox.rotary_emb.inv_freq", "gpt_neox.rotary_emb.original_inv_freq"}
    assert not set(params) & set(buffers)
    objective = LMObjective(loaded.model, seq_len=SEQ, ema_decay=None, pretrained=loaded.variables)
    rows = math.lcm(ROWS, jax.device_count())
    batch = tokens[np.arange(rows) % ROWS]
    data = Dataset(train=lambda partition: iter([{"text": batch}] * 4), val=None, records=rows, batch=rows)
    trainer = Trainer(objective, optax.adamw(1e-2), key=jax.random.key(0),
                      mesh=MeshSpec(fsdp=2, tensor=2), layout=TorchLayout(min_shard=1))

    state = trainer.fit(data, steps=4, log_every=1)

    assert objective_loss(objective, state.params, batch) < objective_loss(objective, loaded.variables, batch)
    moved = state.params["params"]
    assert all(bool(jnp.any(jnp.asarray(params[name]) != moved[name])) for name in params)
    for name, buffer in buffers.items():
        np.testing.assert_array_equal(np.asarray(state.params["buffers"][name]), buffer)
    layer = "gpt_neox.layers.0"
    assert moved[f"{layer}.attention.query_key_value.weight"].sharding.spec == P("tensor", "fsdp")
    assert moved[f"{layer}.mlp.dense_4h_to_h.weight"].sharding.spec == P(None, ("fsdp", "tensor"))
    assert moved["gpt_neox.embed_in.weight"].sharding.spec == P(("fsdp", "tensor"))


def test_save_writes_the_source_names_back(source, loaded, tokens, tmp_path):
    """The saved directory is a transformers checkpoint of the given weights."""
    variables = {**loaded.variables, "params": {name: np.asarray(leaf) * 0.5
                                                 for name, leaf in loaded.variables["params"].items()}}
    loaded.save(tmp_path, variables=variables)
    reloaded = transformers.AutoModelForCausalLM.from_pretrained(tmp_path).eval()
    logits = np.asarray(loaded.model.apply(variables, jnp.asarray(tokens)))
    assert np.abs(logits - torch_logits(reloaded, tokens)).max() <= LOGITS


def test_the_fallback_refuses_what_it_cannot_run(source, loaded, tmp_path):
    with pytest.raises(ValueError, match="the one fallback is 'torchax'"):
        load_pretrained("nobody/nothing", fallback="torch")
    config = {**transformers.GPTNeoXConfig(**TINY).to_dict(),
              "quantization_config": {"quant_method": "awq", "bits": 4}}
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=r"float weights only.*awq"):
        load_pretrained(tmp_path, fallback="torchax")
    with pytest.raises(TypeError, match="generate with transformers"):
        loaded.text_generation()


GPT2 = ("openai-community/gpt2", "607a30d783dfa663caf39e06633721c8d4cfcd7e")
GPT2_FP32 = 6e-4


@pytest.mark.network
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_released_gpt2_matches_transformers(dtype):
    """GPT-2 has no Dew family; the fallback loads it at a pinned commit.

    fp32 is held to twice the measured 2.9e-4 (logits reach 156, where an
    fp32 ulp is 1.5e-5). A bf16 load, bf16 storage and compute, is held
    to transformers' own bf16 run: no further from the fp32 reference than
    that run is (measured 1.12 against 2.07).
    """
    repo, revision = GPT2
    if (os.environ.get("DEW_NETWORK_TESTS") != "1"
            and not isinstance(try_to_load_from_cache(repo, "model.safetensors", revision=revision), str)):
        pytest.skip(f"{repo} at {revision[:8]} is neither cached nor DEW_NETWORK_TESTS=1")
    with pytest.warns(UserWarning, match="tier 3"):
        loaded = load_pretrained(repo, revision=revision, fallback="torchax", dtype=dtype, param_dtype=dtype)
    assert loaded.processor is not None
    ids = np.asarray(loaded.processor("The Cascade Range runs from northern California through").tokens)
    expected = torch_logits(transformers.AutoModelForCausalLM.from_pretrained(
        str(loaded.source), dtype=torch.float32, local_files_only=True).eval(), ids)
    bound = GPT2_FP32
    if dtype == "bfloat16":
        bound = np.abs(torch_logits(transformers.AutoModelForCausalLM.from_pretrained(
            str(loaded.source), dtype=torch.bfloat16, local_files_only=True).eval(), ids) - expected).max()
    logits = np.asarray(loaded.model.apply(loaded.variables, jnp.asarray(ids)))
    assert np.abs(logits - expected).max() <= bound
