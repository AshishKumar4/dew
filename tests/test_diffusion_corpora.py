"""The corpora recipes/diffusion/train.py trains on: values of core specs,
named on the command line, recorded as the core spec they are."""

import importlib.util
import json
import sys
from pathlib import Path

from dew.data import ArrayRecordImages, OnlineImages, TFDSImages
from dew.objectives.diffusion import DiffusionRunConfig
from dew.registry import import_path

ROOT = Path(__file__).resolve().parents[1]


def recipe():
    """The diffusion recipe, loaded once as `recipe_diffusion`."""
    module = sys.modules.get("recipe_diffusion")
    if module is None:
        spec = importlib.util.spec_from_file_location(
            "recipe_diffusion", ROOT / "recipes" / "diffusion" / "train.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    return module


def parse(args):
    return recipe().DiffusionRecipeConfig.cli(
                    [*args, "--trainer.steps", "1"])


def test_a_corpus_is_a_subcommand_and_its_record_reads_back_without_the_recipe():
    """A run's record names the core spec, so dew.pipeline and from_run read
    a recipe run with nothing from this file imported."""
    for args, kind in ((["data:cc12m", "--data.path", "/mnt/gcs"], ArrayRecordImages),
                       (["data:combined-online"], OnlineImages),
                       (["--data.path", "flowers"], TFDSImages)):
        config = parse(args)
        record = json.loads(json.dumps(config.to_dict()))

        assert type(config.data) is kind
        assert record["data"]["class"] == import_path(kind)
        assert DiffusionRunConfig.from_dict(record).data == config.data
        assert type(config).from_dict(record) == config


def test_the_default_corpus_is_flowers_with_flower_captions():
    config = parse(["--data.path", "p"])

    assert config.data.path == "p" and "a photo of a {} flower" in config.data.caption_templates
    assert recipe().corpus_name(config.data) == "oxford-flowers102"


def test_the_experiment_names_the_corpus_a_run_read():
    """Two corpora of one spec are two experiments; a spec no corpus sets
    keeps its registry name."""
    module = recipe()
    names = {module.corpus_name(parse(args).data) for args in (
        ["data:cc12m", "--data.path", "/mnt"], ["data:cc3m", "--data.path", "/mnt"])}

    assert names == {"cc12m", "cc3m"}
    assert module.corpus_name(ArrayRecordImages(path="/mnt", shards=("mine",))) == "array_record_images"
