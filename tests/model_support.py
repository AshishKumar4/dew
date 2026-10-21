"""The tiny decoder and the parameter-tree view several model tests share."""

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from dew.inference import RunProcessor, TextGeneration
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.nn.mixers.gated_delta_net import GatedDeltaNetMixer
from dew.nn.mla import MLAMixer
from dew.sampling import Sampling

TINY_DECODER = {"emb_features": 32, "num_layers": 2, "num_heads": 4, "mlp_features": 64, "max_seq_len": 16}
"""A causal transformer 32 wide, two layers of four heads; each test file adds
its own vocabulary and any field its claim turns on."""


def two_documents(rng: np.random.Generator, rows: int, length: int):
    """The `Coordinates` of `rows` packed rows of `length` tokens: two documents a row, the second
    starting at a column `rng` draws for each row, and the last row's final three slots padding."""
    from dew.nn.inputs import coordinates

    segments = np.ones((rows, length), np.int32)
    positions = np.zeros((rows, length), np.int32)
    for row in range(rows):
        boundary = int(rng.integers(3, length - 3))
        segments[row, boundary:] = 2
        positions[row, :boundary] = np.arange(boundary)
        positions[row, boundary:] = np.arange(length - boundary)
    segments[-1, -3:] = 0
    return coordinates(segments=segments, positions=positions, max_segments=length, right_padded=True, xp=np)


def decode_through(module: nn.Module, variables, hidden, prefill: int):
    """`module` over `hidden` through its decode cache: the call that allocates the cache, then the
    first `prefill` tokens at once, then one token a step. Returns the outputs joined along the
    sequence, the cache as allocated and the cache after the last step."""
    from dew.nn.inputs import MixerCall

    step = MixerCall(decode=True)
    allocated = module.apply(variables, hidden[:, :1], step, mutable=["cache"])[1]
    out, cache = module.apply({**variables, **allocated}, hidden[:, :prefill], step, mutable=["cache"])
    outputs = [out]
    for position in range(prefill, hidden.shape[1]):
        out, cache = module.apply({**variables, **cache}, hidden[:, position:position + 1], step,
                                  mutable=["cache"])
        outputs.append(out)
    return jnp.concatenate(outputs, axis=1), allocated["cache"], cache["cache"]


def flat_tree(tree) -> dict:
    """`tree`'s leaves by their dictionary keys joined with dots, the way a
    checkpoint names them: 'params.layers_0.mlp.gate.kernel'."""
    return {".".join(str(entry.key) for entry in path): leaf
            for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]}


def decoder(kind: str = "attention") -> CausalTransformer:
    mixer = None
    if kind == "mla":
        mixer = MLAMixer(q_lora_rank=8, kv_lora_rank=8, qk_nope_head_dim=4,
                         qk_rope_head_dim=4, v_head_dim=8)
    elif kind == "recurrent":
        mixer = GatedDeltaNetMixer(linear_num_key_heads=2, linear_num_value_heads=2,
                                  linear_key_head_dim=8, linear_value_head_dim=8)
    return CausalTransformer(vocab_size=13, emb_features=16, num_layers=1, num_heads=2,
                             head_dim=8, mlp_features=32, max_seq_len=12,
                             dtype="float32", mixer=mixer)


class Digits:
    def encode(self, text):
        return [int(character) for character in text]

    def decode(self, ids):
        return "".join(str(int(token)) for token in ids)


_DEFAULT_TASK_SAMPLING = Sampling(temperature=0, eos_token_ids=12)


def serving_task(sampling: Sampling = _DEFAULT_TASK_SAMPLING, capacity: int = 128) -> TextGeneration:
    model = decoder().clone(max_seq_len=capacity)
    params = model.init(jax.random.key(0), jnp.ones((1, 2), jnp.int32))
    return TextGeneration(model, params, RunProcessor(Digits()), sampling=sampling)


def greedy_walk(model: nn.Module, params: dict, prompt, steps: int,
                transform: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None) -> np.ndarray:
    """Greedy continuation with a fresh full forward for every token."""
    sequence = np.asarray(prompt)
    for _ in range(steps):
        logits = np.asarray(model.apply(params, jnp.asarray(sequence))[:, -1])
        if transform is not None:
            logits = transform(sequence, logits)
        sequence = np.concatenate([sequence, logits.argmax(-1)[:, None].astype(np.int32)], axis=1)
    return sequence


def streamed_export(writer: str, argument: Path, directory: Path, shard_size: str) -> dict:
    """The streaming export's measurements (tools/benchmark_streaming_export.py, its last line) of the
    checkpoint the script `writer` writes from `argument` into `directory`/source, saved in `shard_size`
    shards into `directory`/out. The writer and the export each run in a process of their own on one
    CPU device, as a user's has (the suite's simulated mesh would place every leaf eight times over),
    so no heap the writer freed is there for the save to reuse."""
    root = Path(__file__).resolve().parents[1]
    env = {**{name: value for name, value in os.environ.items() if name != "XLA_FLAGS"},
           "JAX_PLATFORMS": "cpu", "PYTHONPATH": str(root / "src")}
    written = subprocess.run([sys.executable, "-c", writer, str(argument), str(directory / "source")],
                             capture_output=True, text=True, env=env)
    assert written.returncode == 0, written.stderr[-2000:]
    run = subprocess.run([sys.executable, str(root / "tools" / "benchmark_streaming_export.py"),
                          str(directory / "source"), shard_size, str(directory / "out")],
                         capture_output=True, text=True, env=env)
    assert run.returncode == 0, run.stderr[-2000:]
    return json.loads(run.stdout.splitlines()[-1])


def tokenized_qwen3(directory: Path) -> Path:
    """`directory`, made to hold the tiny Qwen3 checkpoint (fixtures/hf/qwen3-tiny) beside a
    tokenizer it ships (fixtures/hf/qwen38-dense-tiny's): a decoder source whose processor is a
    tokenizer, not bytes."""
    import shutil

    fixtures = Path(__file__).resolve().parent / "fixtures" / "hf"
    directory.mkdir()
    for name in ("config.json", "model.safetensors"):
        shutil.copyfile(fixtures / "qwen3-tiny" / name, directory / name)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(fixtures / "qwen38-dense-tiny" / name, directory / name)
    return directory


LLADA2_BLOCK = 4
"""The block of LLaDA2's block-causal mask that its recordings' forwards run under."""


def llada_draw(seed: int, vocabulary: int) -> dict[str, np.ndarray]:
    """A LLaDA-MoE recording's draw at `seed` (tools/llada_moe_reference.py), which its test draws
    again: two rows of 16 token ids, whole blocks of LLaDA2.2's 4-token routing block."""
    return {"ids": np.random.default_rng(seed).integers(0, vocabulary, (2, 16)).astype(np.int32)}


def llada_fields(name: str, rows: int, length: int) -> dict[str, jax.Array]:
    """The forward's fields of a LLaDA-MoE recording over `rows` of `length` tokens: for LLaDA2's
    releases, the explicit block-causal mask of `LLADA2_BLOCK`-token blocks its 2.0 forward requires
    and its 2.1 and 2.2 take; none for the others, whose forward is bidirectional."""
    if name not in ("llada2-tiny", "llada21-tiny", "llada22-tiny"):
        return {}
    blocks = np.arange(length) // LLADA2_BLOCK
    return {"attention_pairwise_mask": jnp.asarray(np.broadcast_to(
        blocks[None, :] <= blocks[:, None], (rows, length, length)))}


def cached_round_trips(round_trip, cases, workspace) -> Callable[[str], object]:
    """A function of a case's name to its round trip (`round_trip(cases[name], workspace)`, a family
    trained and exported once), run the first time a test asks for it and shared after."""
    cache = {}

    def trip(name: str):
        if name not in cache:
            cache[name] = round_trip(cases[name], workspace)
        return cache[name]

    return trip


def parameter_loss(objective, variables, batch, step) -> Callable:
    """`objective`'s reduced loss of `batch` at `step` as a function of the parameters alone, the rest
    of `variables` held: the function a reference's SGD step differentiates."""
    def loss(params):
        statistics, _ = objective.loss({**variables, "params": params}, batch, step)
        return objective.reduce_loss(statistics)[0]

    return loss
