"""The language-model recipe: a config in, a trained trainer out.

The first test drives the recipe with a stub model and a stub loader, which is
exactly the wiring the recipe owns: the vocabulary read off meta.json, the
sequence length written onto the data config, the objective, the perplexity
metric and the trainer with no input config. The second runs the whole real
path over token files on disk.
"""

import importlib.util
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from dew.config import DataConfig, ModelConfig, OptimConfig, TrainerConfig
from test_lm_objective import BATCH, SEQ, VOCAB, TinyCausalLM, cycle_batches

RECIPE = Path(__file__).resolve().parents[1] / "recipes" / "lm" / "train.py"


def load_recipe():
    spec = importlib.util.spec_from_file_location("lm_train_recipe", RECIPE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_token_files(directory, vocab_size, train_tokens=4096, val_tokens=1024):
    """A token directory shaped like the one tools/tokenize_text.py writes."""
    directory.mkdir(parents=True, exist_ok=True)
    rs = np.random.RandomState(0)
    for name, count in (("train.bin", train_tokens), ("val.bin", val_tokens)):
        rs.randint(0, vocab_size, count).astype(np.uint16).tofile(directory / name)
    (directory / "meta.json").write_text(json.dumps({
        "tokenizer": "byte", "vocab_size": vocab_size, "dtype": "uint16",
        "train_tokens": train_tokens, "val_tokens": val_tokens,
    }))
    return directory


def test_the_run_writes_its_sequence_length_onto_the_data_config():
    recipe = load_recipe()
    config = recipe.LmRunConfig(
        data=DataConfig(dataset="tokens", batch_size=4),
        trainer=TrainerConfig(compilation_cache_dir=None),
        sequence_length=128, tokenizer="gpt2")

    data_config = recipe.token_data_config(config)

    assert data_config.sequence_length == 128 and data_config.tokenizer == "gpt2"
    assert data_config.dataset == "tokens" and data_config.batch_size == 4


def trainer_config(tmp_path, **kwargs):
    """Trainer settings for a recipe run inside the test process.

    multi_host is off because prepare_process otherwise asks
    jax.distributed.initialize() for a process pool, which a test that has
    already touched JAX cannot join.
    """
    return TrainerConfig(distributed_training=False, multi_host=False,
                         checkpoint_dir=str(tmp_path / "checkpoints"),
                         compilation_cache_dir=None, **kwargs)


def test_the_sampling_budget_decides_the_context_the_model_is_built_for():
    """generate decodes into a cache sized at build time, so a long sample has
    to fit in it even when it outruns the training context."""
    recipe = load_recipe()
    config = recipe.LmRunConfig(sequence_length=64)

    assert recipe.context_length(config, None) == 64
    assert recipe.context_length(config, {"prompt": [1, 2, 3], "max_new_tokens": 8}) == 64
    assert recipe.context_length(config, {"prompt": [1, 2, 3], "max_new_tokens": 100}) == 103


def test_the_recipe_wires_the_objective_and_the_trainer(tmp_path, monkeypatch):
    recipe = load_recipe()
    dataset = write_token_files(tmp_path / "tokens", vocab_size=VOCAB)
    built = {}

    def fake_build_model(architecture, config):
        built.update(architecture=architecture, config=config)
        return TinyCausalLM(vocab_size=config["vocab_size"])

    def fake_load_data(data_config):
        built["data_config"] = data_config
        return {"train": cycle_batches, "val": cycle_batches, "train_len": 64,
                "local_batch_size": BATCH}

    monkeypatch.setattr(recipe, "build_model", fake_build_model)
    monkeypatch.setattr(recipe, "load_data", fake_load_data)

    config = recipe.LmRunConfig(
        model=ModelConfig("causal_transformer", {"emb_features": 16, "num_layers": 1}),
        data=DataConfig(dataset=str(dataset), batch_size=BATCH, val_steps_per_epoch=1),
        optim=OptimConfig(learning_rate=3e-3),
        trainer=trainer_config(tmp_path, epochs=1, steps_per_epoch=3),
        sequence_length=SEQ,
        sample_tokens=0,
    )
    trainer = recipe.main(config)

    # The data decides the vocabulary, and the run decides the context
    assert built["architecture"] == "causal_transformer"
    assert built["config"]["vocab_size"] == VOCAB
    assert built["config"]["max_seq_len"] == SEQ
    assert built["config"]["dtype"] == "bfloat16", "the precision policy was skipped"
    assert built["data_config"].sequence_length == SEQ
    assert built["data_config"].tokenizer == "byte"

    assert trainer.objective.tag == "lm"
    assert trainer.objective.seq_len == SEQ and trainer.objective.vocab_size == VOCAB
    assert trainer.objective.samples is None
    assert trainer.input_shapes == {"tokens": ((SEQ,), jnp.int32)}
    assert [metric.name for metric in trainer.eval_metrics] == ["perplexity"]
    assert trainer.best_tracker_metric == "val/perplexity"
    assert "val/perplexity" in trainer.best_val_metrics, "validation never scored"
    assert int(trainer.state.step) == 3
    # No wandb project configured, so the run never opened one
    assert trainer.wandb is None


def test_a_dataset_that_is_not_a_token_directory_says_so(tmp_path):
    recipe = load_recipe()
    with pytest.raises(FileNotFoundError, match="meta.json"):
        recipe.read_meta(str(tmp_path))


def test_a_tokenizer_that_does_not_match_the_token_files_is_rejected(tmp_path, monkeypatch):
    recipe = load_recipe()
    dataset = write_token_files(tmp_path / "tokens", vocab_size=VOCAB)
    monkeypatch.setattr(recipe, "load_data", lambda config: {})

    config = recipe.LmRunConfig(
        data=DataConfig(dataset=str(dataset)),
        trainer=trainer_config(tmp_path),
        tokenizer="gpt2",
    )
    with pytest.raises(ValueError, match="written with byte"):
        recipe.main(config)


def test_the_recipe_trains_on_tokenized_files(tmp_path):
    """The real path end to end: registry model, token loader, sampled text."""
    recipe = load_recipe()
    dataset = write_token_files(tmp_path / "tokens", vocab_size=256)
    config = recipe.LmRunConfig(
        model=ModelConfig("causal_transformer",
                          {"emb_features": 32, "num_layers": 2, "num_heads": 2}),
        data=DataConfig(dataset=str(dataset), batch_size=4, val_steps_per_epoch=1,
                        worker_count=0),
        trainer=trainer_config(tmp_path, epochs=1, steps_per_epoch=2),
        sequence_length=32,
        sample_prompt="the ",
        sample_tokens=8,
    )
    trainer = recipe.main(config)

    assert trainer.objective.tag == "lm"
    assert int(trainer.state.step) == 2
    assert "val/perplexity" in trainer.best_val_metrics, "validation never scored"
    trainer.wait_for_checkpoints()
    assert trainer.checkpointer.latest_step() == 2
