"""A run whose model computes inside wrappers (a bound LoRA, a run's quantized
training, a checkpoint's input quantization) records them in the order they
were applied (`ModelConfig.wrappers`), and its public loader rebuilds the same
model: the same wrappers, in the same order, computing what the run computed."""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import dew
from dew.config import ModelConfig, TrainerConfig
from dew.data import Dataset
from dew.interop import Pretrained
from dew.lora import AdapterRecord, LoRA
from dew.nn.backbones.causal_transformer import CausalTransformer
from dew.objectives.lm import LMObjective, LMRunConfig
from dew.training import Checkpoints, Trainer
from dew.training.quantization import InputQuantization, Quantization, quantize_trunk

pytestmark = pytest.mark.mesh

FIXTURES = Path(__file__).parent / "fixtures"
BATCH, SEQ = 8, 8
IDS = jnp.asarray([[3, 4, 5, 6, 7, 8, 9, 10]], jnp.int32)


class Stream:
    """The same batch of token windows each step, below `vocab`."""

    def __init__(self, vocab: int):
        self.vocab, self.position = vocab, 0

    def __iter__(self):
        return self

    def __next__(self):
        self.position += 1
        return {"text": np.random.RandomState(0).randint(1, self.vocab, (BATCH, SEQ + 1)).astype(np.int32)}

    def get_state(self):
        return str(self.position).encode()

    def set_state(self, state):
        self.position = int(state.decode())


def trained_run(directory, objective, vocab, quantization=None):
    """Train `objective` one step as a run under `directory` and load the run back as `dew.pipeline` does."""
    config = LMRunConfig(trainer=TrainerConfig(checkpoint_dir=str(directory), batch_size=BATCH, steps=1,
                                               eval_every=None, checkpoint_every=1,
                                               compilation_cache_dir=None, quantization=quantization))
    state = config.train(objective, Dataset(lambda partition: Stream(vocab), None, None, BATCH), name="run")
    Checkpoints(str(directory / "run")).wait()
    return state, dew.pipeline(str(directory / "run"))


def wrappers(model) -> tuple[type, ...]:
    return tuple(type(wrapper) for wrapper in ModelConfig.from_model(model).wrappers)


def rebuilt_with(model, *kept: type):
    """`model` rebuilt from its record with only the wrappers of the `kept` types."""
    config = ModelConfig.from_model(model)
    return dataclasses.replace(
        config, wrappers=tuple(wrapper for wrapper in config.wrappers if isinstance(wrapper, kept))).build()


def logits(model, variables):
    return np.asarray(jax.jit(model.apply)(variables, IDS))


def decoder() -> CausalTransformer:
    return CausalTransformer(vocab_size=64, emb_features=16, num_layers=1, num_heads=2, head_dim=8,
                             mlp_features=32, max_seq_len=16, attention_impl="reference")


@pytest.mark.parametrize("order", ["lora, then the trainer's quantization", "quantization, then lora"])
def test_a_run_reloads_its_lora_and_quantization_in_the_order_it_trained_them(tmp_path, order):
    """A LoRA run the trainer quantizes holds the adapter inside the
    quantization; a quantized model adapted afterwards holds them the other way
    round. Each reloads with its own order, computing the run's logits bit for
    bit, and the quantization is live: without it the same weights compute
    other logits."""
    pytest.importorskip("qwix")
    base = decoder()
    variables = base.init(jax.random.key(0), IDS)
    lora = LoRA(rank=2, modules=("q_proj",))
    if order.startswith("lora"):
        adapter = lora.apply(base, variables, key=jax.random.key(1))
        objective = LMObjective(adapter.model, SEQ, variables=adapter.variables)
        state, task = trained_run(tmp_path, objective, 64, quantization=Quantization())
        expected = (AdapterRecord, Quantization)
    else:
        adapter = lora.apply(Quantization().apply(base), variables, key=jax.random.key(1))
        objective = LMObjective(adapter.model, SEQ, variables=adapter.variables)
        state, task = trained_run(tmp_path, objective, 64)
        expected = (Quantization, AdapterRecord)

    assert wrappers(objective.model) == wrappers(task.model) == expected
    np.testing.assert_array_equal(logits(task.model, task.variables),
                                  logits(objective.model, state.variables))
    unquantized = rebuilt_with(task.model, AdapterRecord)
    assert float(np.max(np.abs(logits(unquantized, task.variables) - logits(task.model, task.variables)))) > 0


def test_a_quantized_run_trains_on_from_its_own_record(tmp_path):
    """`--trainer.quantization` is the trainer's: the run's own run.json
    builds the model unwrapped and the trainer wraps it once more, so a run
    stopped after one step and resumed from its record ends where two
    uninterrupted steps end, bit for bit."""
    pytest.importorskip("qwix")
    from dew.config import RunConfig

    def run(directory, steps, config=None):
        fields = {"vocab_size": 64, "emb_features": 16, "num_layers": 1, "num_heads": 2, "head_dim": 8,
                  "mlp_features": 32, "max_seq_len": 16, "attention_impl": "reference"}
        config = config or LMRunConfig(
            model=ModelConfig("causal_transformer", fields),
            trainer=TrainerConfig(checkpoint_dir=str(directory), batch_size=BATCH, steps=steps,
                                  eval_every=None, checkpoint_every=1, compilation_cache_dir=None,
                                  quantization=Quantization()))
        objective = LMObjective(config.model.build(), SEQ)
        state = config.train(objective, Dataset(lambda partition: Stream(64), None, None, BATCH), name="run")
        Checkpoints(str(directory / "run")).wait()
        return state

    whole = run(tmp_path / "whole", 2)
    run(tmp_path / "stopped", 1)
    record = RunConfig.load(str(tmp_path / "stopped" / "run"))
    assert isinstance(record, LMRunConfig) and record.model.wrappers == ()
    resumed = run(tmp_path / "stopped", 2, dataclasses.replace(
        record, trainer=dataclasses.replace(record.trainer, steps=2)))
    assert int(resumed.step) == int(whole.step) == 2
    for before, after in zip(jax.tree.leaves(whole.variables), jax.tree.leaves(resumed.variables),
                             strict=True):
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))


def test_a_second_quantization_is_refused_where_qwix_would_replace_the_first():
    """Qwix holds one provider per model, and a second call drops the first
    and every wrapper made over it, an adapter included; it is refused."""
    pytest.importorskip("qwix")
    base = decoder()
    adapted = LoRA(rank=2, modules=("q_proj",)).apply(
        Quantization().apply(base), base.init(jax.random.key(0), IDS), key=jax.random.key(1)).model
    with pytest.raises(ValueError, match="already computes under a Qwix quantization"):
        Quantization(dtype="fp8").apply(adapted)


@pytest.mark.parametrize("fixture", ["modelopt/mixed", "modelopt/tiny", "nvfp4_qdq/e4m3"],
                         ids=["modelopt fp8 and nvfp4", "modelopt nvfp4", "compressed-tensors nvfp4"])
def test_a_run_of_an_input_quantized_checkpoint_reloads_with_its_input_quantization(tmp_path, fixture):
    """A ModelOpt or compressed-tensors FP8 or NVFP4 checkpoint computes with
    its input quantization (`InputQuantization`); a run of it records the
    per-Linear scales and reloads computing the run's logits bit for bit,
    which the same weights without them do not."""
    pytest.importorskip("qwix")
    source = Pretrained.load(FIXTURES / "codecs" / fixture, dtype="float32", attention_impl="reference")
    objective = LMObjective(source.model, SEQ, variables=source.variables)
    state, task = trained_run(tmp_path, objective, 90)

    recorded = ModelConfig.from_model(task.model).wrappers
    assert recorded == ModelConfig.from_model(source.model).wrappers
    assert len(recorded) == 1 and isinstance(recorded[0], InputQuantization) and recorded[0].inputs
    np.testing.assert_array_equal(logits(task.model, task.variables), logits(source.model, state.variables))
    plain = rebuilt_with(task.model)
    assert float(np.max(np.abs(logits(plain, task.variables) - logits(task.model, task.variables)))) > 0


def test_a_block_run_the_trainer_quantizes_reloads_quantized(tmp_path):
    """The trainer quantizes the block objective's training view; the model
    its record and task read is wrapped the same way at its serving
    capacity, so the saved run reloads quantized and computes the objective's
    canvas logits bit for bit."""
    pytest.importorskip("qwix")
    from test_block_diffusion import prefill

    from dew.data.text import tokenizer_for
    from dew.inference import RunProcessor
    from dew.objectives.diffusion.block import BlockDiffusionObjective

    bundle = Pretrained.load(str(FIXTURES / "hf" / "diffusion-gemma-workflow"), dtype="float32",
                             attention_impl="xla", max_seq_len=32)
    objective = BlockDiffusionObjective(bundle.model, prompt_length=3, variables=bundle.variables,
                                        processor=RunProcessor(tokenizer_for("byte")))
    serving = objective.model.text.max_seq_len
    quantize_trunk(objective, Quantization())
    assert wrappers(objective.model) == wrappers(objective.training_model) == (Quantization,)
    assert objective.model.text.max_seq_len == serving
    assert objective.training_model.text.max_seq_len == objective.sequence_length

    state = Trainer(objective, optax.sgd(0.01), key=jax.random.PRNGKey(2)).initial_state()
    checkpoints = Checkpoints(str(tmp_path))
    checkpoints.save(0, state, None, artifact=objective.inference_record())
    checkpoints.wait()
    task = dew.pipeline(str(tmp_path))

    assert wrappers(task.model) == (Quantization,)
    prompt = jnp.asarray([[5, 6, 7]], jnp.int32)
    canvas = jnp.arange(8, 8 + objective.canvas_size, dtype=jnp.int32)[None]

    def canvas_logits(model, variables):
        cache = prefill(model, variables, prompt)
        return np.asarray(model.apply({**variables, "cache": cache}, canvas))

    np.testing.assert_array_equal(canvas_logits(task.model, task.variables),
                                  canvas_logits(objective.model, state.variables))
    plain = rebuilt_with(task.model)
    assert float(np.max(np.abs(canvas_logits(plain, task.variables)
                               - canvas_logits(task.model, task.variables)))) > 0
