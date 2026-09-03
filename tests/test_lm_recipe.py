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

# Needs the eight simulated CPU devices conftest configures; the GPU lane skips it.
pytestmark = pytest.mark.mesh

from dew.config import DataConfig, ModelConfig, OptimConfig, TrainerConfig
from test_lm_objective import (
    BATCH, SEQ, VOCAB, PackedTinyLM, TinyCausalLM, cycle_batches,
)

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


def write_packed_token_files(directory, vocab_size, train_lengths, val_lengths):
    """A token directory shaped like tools/tokenize_text.py --pack writes it:
    every document closed by the eos id meta.json records."""
    directory.mkdir(parents=True, exist_ok=True)
    rs = np.random.RandomState(0)
    eos_id = vocab_size - 1

    def stream(lengths):
        return np.concatenate(
            [np.append(rs.randint(0, eos_id, length - 1), eos_id)
             for length in lengths]).astype(np.uint16)

    train, val = stream(train_lengths), stream(val_lengths)
    train.tofile(directory / "train.bin")
    val.tofile(directory / "val.bin")
    (directory / "meta.json").write_text(json.dumps({
        "tokenizer": "byte", "vocab_size": vocab_size, "dtype": "uint16",
        "train_tokens": int(train.size), "val_tokens": int(val.size),
        "eos_id": eos_id,
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


def test_the_computed_context_cannot_be_overridden_by_model_config():
    recipe = load_recipe()
    config = recipe.LmRunConfig(
        model=ModelConfig("causal_transformer", {"max_seq_len": 8}),
        sequence_length=64,
    )

    _, model_config = recipe.build_lm(config, vocab_size=256, max_seq_len=103)

    assert model_config["max_seq_len"] == 103


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


def packed_run(tmp_path, monkeypatch, train_lengths, val_lengths, batch_size):
    """A run over a packed corpus with the step count left to the loader."""
    recipe = load_recipe()
    dataset = write_packed_token_files(
        tmp_path / "tokens", VOCAB, train_lengths, val_lengths)
    monkeypatch.setattr(
        recipe, "build_model",
        lambda architecture, config: PackedTinyLM(vocab_size=config["vocab_size"]))
    return recipe, recipe.LmRunConfig(
        model=ModelConfig("causal_transformer", {"emb_features": 16, "num_layers": 1}),
        data=DataConfig(dataset=str(dataset), batch_size=batch_size,
                        val_steps_per_epoch=1, worker_count=0, pack_sequences=True),
        trainer=trainer_config(tmp_path, epochs=1),
        sequence_length=SEQ,
        sample_tokens=0,
    )


def test_a_packed_corpus_of_one_document_still_has_steps_in_an_epoch(tmp_path, monkeypatch):
    """steps_per_epoch comes from the loader's length when a run does not set
    it, and a packed split is windows, not documents: one long document is an
    epoch of several batches."""
    # 16 windows of SEQ + 1, four rows to a batch.
    recipe, config = packed_run(tmp_path, monkeypatch, [16 * (SEQ + 1)],
                                [4 * (SEQ + 1)], batch_size=4)

    trainer = recipe.main(config)

    assert int(trainer.state.step) == 4
    assert "val/perplexity" in trainer.best_val_metrics, "validation never scored"


def test_a_corpus_too_small_for_one_batch_is_refused(tmp_path, monkeypatch):
    """An epoch of zero steps trains nothing and says nothing about it."""
    recipe, config = packed_run(tmp_path, monkeypatch, [2 * (SEQ + 1)],
                                [4 * (SEQ + 1)], batch_size=4)

    with pytest.raises(ValueError, match="do not fill one batch"):
        recipe.main(config)
def tiny_export(directory, tokenizer_name="byte"):
    """A local checkpoint in the HF layout, as --pretrained takes one."""
    from dew.interop.hf_decoders import (
        load_pretrained_decoder, save_pretrained_decoder,
    )

    fixture = Path(__file__).resolve().parent / "fixtures" / "hf" / "qwen3-tiny"
    model, variables, _ = load_pretrained_decoder(
        str(fixture), dtype='float32', attention_impl='reference')
    save_pretrained_decoder(model, variables, directory,
                            tokenizer_name=tokenizer_name)
    return directory, variables


def test_the_recipe_continues_training_from_a_pretrained_checkpoint(tmp_path):
    """--pretrained decides the architecture and the initial weights: the run
    starts from the checkpoint's parameters, not from a fresh init."""
    recipe = load_recipe()
    dataset = write_token_files(tmp_path / "tokens", vocab_size=256)
    export, variables = tiny_export(tmp_path / "checkpoint")

    config = recipe.LmRunConfig(
        model=ModelConfig("causal_transformer", {}, dtype="float32"),
        data=DataConfig(dataset=str(dataset), batch_size=4, val_steps_per_epoch=1,
                        worker_count=0),
        trainer=trainer_config(tmp_path, epochs=1, steps_per_epoch=2),
        sequence_length=32,
        sample_tokens=0,
        pretrained=str(export),
    )
    trainer = recipe.main(config)

    # the checkpoint's own shape, none of the recipe's defaults
    assert (trainer.model.emb_features, trainer.model.num_layers) == (64, 2)
    assert trainer.model.vocab_size == 256 and trainer.model.qk_norm
    started_from = trainer.objective.pretrained
    assert started_from is not None
    assert np.array_equal(
        np.asarray(started_from["params"]["embed_tokens"]["embedding"]),
        np.asarray(variables["params"]["embed_tokens"]["embedding"]))
    assert int(trainer.state.step) == 2


def test_a_pretrained_run_logs_the_config_it_built_the_model_from(tmp_path):
    """model_config is what lands in wandb and what run_summary spreads into
    arguments, so it has to be the dew config the model ran with, compute dtype
    and attention kernel included, and it has to rebuild that model through the
    config parser every other run's config goes through."""
    from dew.registry import build_model

    recipe = load_recipe()
    dataset = write_token_files(tmp_path / "tokens", vocab_size=256)
    export, _ = tiny_export(tmp_path / "checkpoint")
    config = recipe.LmRunConfig(
        model=ModelConfig("causal_transformer", {}, dtype="float32",
                          attention_impl="reference"),
        data=DataConfig(dataset=str(dataset), worker_count=0),
        trainer=trainer_config(tmp_path),
        sequence_length=32,
        pretrained=str(export),
    )

    model, _, model_config = recipe.load_pretrained(
        config, vocab_size=256, max_seq_len=32, meta=recipe.read_meta(str(dataset)))

    assert model_config["dtype"] == "float32"
    assert model_config["attention_impl"] is None, "'reference' is the kernel None"
    assert build_model("causal_transformer", model_config) == model


def test_a_pretrained_run_refuses_architecture_overrides(tmp_path):
    recipe = load_recipe()
    dataset = write_token_files(tmp_path / "tokens", vocab_size=256)
    export, _ = tiny_export(tmp_path / "checkpoint")

    config = recipe.LmRunConfig(
        model=ModelConfig("causal_transformer", {"emb_features": 128}),
        data=DataConfig(dataset=str(dataset), worker_count=0),
        trainer=trainer_config(tmp_path),
        sequence_length=32,
        pretrained=str(export),
    )
    with pytest.raises(ValueError, match="emb_features"):
        recipe.main(config)


def test_a_pretrained_checkpoint_from_another_tokenizer_is_rejected(tmp_path):
    """The ids on disk have to come from the checkpoint's own vocabulary."""
    recipe = load_recipe()
    dataset = write_token_files(tmp_path / "tokens", vocab_size=256)
    export, _ = tiny_export(tmp_path / "checkpoint", tokenizer_name="gpt2")

    config = recipe.LmRunConfig(
        model=ModelConfig("causal_transformer", {}),
        data=DataConfig(dataset=str(dataset), worker_count=0),
        trainer=trainer_config(tmp_path),
        sequence_length=32,
        pretrained=str(export),
    )
    with pytest.raises(ValueError, match="expects gpt2"):
        recipe.main(config)


def test_a_padded_embedding_table_is_not_a_vocabulary_mismatch(tmp_path):
    """Qwen3 stores 151936 rows for 151669 tokens, and every real decoder pads
    like that, so covering the ids is the requirement rather than matching."""
    recipe = load_recipe()
    dataset = write_token_files(tmp_path / "tokens", vocab_size=250)
    export, _ = tiny_export(tmp_path / "checkpoint")

    config = recipe.LmRunConfig(
        model=ModelConfig("causal_transformer", {}, dtype="float32"),
        data=DataConfig(dataset=str(dataset), batch_size=4, val_steps_per_epoch=1,
                        worker_count=0),
        trainer=trainer_config(tmp_path, epochs=1, steps_per_epoch=1),
        sequence_length=32,
        sample_tokens=0,
        pretrained=str(export),
    )
    trainer = recipe.main(config)

    assert trainer.model.vocab_size == 256 and trainer.objective.vocab_size == 250
    assert int(trainer.state.step) == 1
