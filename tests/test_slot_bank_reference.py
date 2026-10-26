"""The independent author's float64 model receives exact tensors through the existing bank source."""

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest
import torch
from tool_support import run_from
from transformers import AutoConfig, AutoModelForCausalLM

from dew.inference.banks import SafetensorsBanks
from dew.interop import Pretrained

SOURCE = Path(__file__).parent / "fixtures/hf/qwen38-dense-tiny"


@pytest.mark.parametrize("changed", [None, "pool", "source", "tokenizer", "device", "dtype", "phase"])
def test_a_runtime_verdict_refuses_records_from_a_different_experiment(tmp_path, changed):
    """A verdict judges the three phases' records only where each was recorded over this pool, from this
    source's bytes (its tokenizer files among them), at the declared device, dtype and phase; any one of
    them changed refuses the verdict before it writes one."""
    from reference_error import REDRAW, SIZING_FROM

    source = tmp_path / "source"
    source.mkdir()
    config = source / "config.json"
    config.write_text('{"model_type":"qwen3_5"}')
    count = 8
    verdict, pilot = list(range(count * (1 + REDRAW))), list(range(SIZING_FROM, SIZING_FROM + 2 * count))
    # One option a question: its log-probability is exactly zero in every precision, so the drawn
    # errors are zero and the admitted control sizes at its fewest seeds.
    pool = {"count": count, "verdict": verdict, "pilot": pilot, "sizing_from": SIZING_FROM,
            "requests": {str(seed): {"questions": {f"case_{seed}_{index}": {"criteria": ["only"]}
                                                   for index in range(2)}}
                         for seed in (*verdict, *pilot)}}
    pool_file = tmp_path / "pool.json"
    pool_file.write_text(json.dumps(pool))
    identity = {"files": {
        "config.json": hashlib.sha256(config.read_bytes()).hexdigest()}}
    for phase in ("native", "author", "truth"):
        seeds = pool["verdict"] if phase == "native" else [*pool["pilot"], *pool["verdict"]]
        dtype = "float64" if phase == "truth" else "float32"
        np.savez(tmp_path / f"{phase}.npz", **{name: np.zeros(1, dtype) for seed in seeds
                                                for name in pool["requests"][str(seed)]["questions"]})
        metadata = {"phase": phase, "source": identity,
                    "pool_sha256": hashlib.sha256(pool_file.read_bytes()).hexdigest(),
                    "compute_dtype": dtype, "storage": "auto" if phase == "native" else dtype,
                    "parameter_dtypes": [dtype], "platforms": ["cpu"], "device": "cpu",
                    "precision": "ieee", "floating_operations": 1, "cuda_floating_operations": 0}
        (tmp_path / f"{phase}.json").write_text(json.dumps(metadata))
    args = SimpleNamespace(source=source, pool=pool_file, records=tmp_path, dtype="float32", device="cpu",
                           output=tmp_path / "verdict.json")
    if changed == "pool":
        pool_file.write_text(json.dumps({**pool, "population": "a different request set"}))
    elif changed == "source":
        config.write_text('{"model_type":"a different source"}')
    elif changed == "tokenizer":
        (source / "vocab.txt").write_text("a changed tokenizer\n")
    elif changed is not None:
        filename = tmp_path / "author.json"
        metadata = json.loads(filename.read_text())
        field = "compute_dtype" if changed == "dtype" else changed
        metadata[field] = {"device": "cuda:0", "dtype": "bfloat16", "phase": "truth"}[changed]
        filename.write_text(json.dumps(metadata))
    with run_from("slot_runtime_qualification", "tools") as tool:
        if changed is None:
            tool.verdict(args, pool)
            assert args.output.is_file()
        else:
            with pytest.raises(ValueError):
                tool.verdict(args, pool)
            assert not args.output.exists()


@pytest.fixture
def reference_tool():
    with run_from("slot_runtime_qualification", "tools") as module:
        from block_override_reference import oracle_precision

        yield module, oracle_precision


def test_author_float64_offload_receives_bit_identical_weights_and_prefill_suffix_results(reference_tool):
    """The author's float64 model, its weights streamed from the native banks and kept at their stored
    precision once recovered, receives each weight bit for bit as the resident model loads it, and
    computes a prefill and its continuation as the resident model does; a forward after the first reads
    no bank."""
    from accelerate.hooks import ModelHook, add_hook_to_module

    tool, precision = reference_tool
    with jax.enable_x64(new_val=True):
        bundle = Pretrained.load(SOURCE, weights=False, dtype="float64", param_dtype="float64",
                                  attention_impl="reference")
        config = AutoConfig.from_pretrained(SOURCE, local_files_only=True).get_text_config()
        stored = AutoModelForCausalLM.from_pretrained(
            SOURCE, config=config, dtype=torch.float32, attn_implementation="eager").eval()
        with precision():
            resident = AutoModelForCausalLM.from_config(
                copy.deepcopy(config), dtype=torch.float64, attn_implementation="eager").eval()
            resident.load_state_dict(stored.state_dict())
        del stored
        with SafetensorsBanks(SOURCE, param_dtype="float64", cache_bytes=0, read_ahead=False) as source:
            streamed, weights = tool.offloaded_model(SOURCE, source, bundle, device=torch.device("cpu"))
            assert weights.reads == 0
            original = resident.state_dict()
            buffers = dict(resident.named_buffers())
            for name, value in streamed.named_buffers():
                np.testing.assert_array_equal(value.numpy(), buffers[name].numpy(), err_msg=name)
            for name in weights:
                np.testing.assert_array_equal(weights[name].numpy(), original[name].numpy())
            weights.clear()
            seen = set()

            class Delivered(ModelHook):
                def __init__(self, prefix):
                    self.prefix = prefix

                def pre_forward(self, module, *args, **kwargs):
                    for name, parameter in module.named_parameters():
                        key = f"{self.prefix}.{name}"
                        np.testing.assert_array_equal(parameter.detach().cpu().numpy(), original[key].numpy())
                        seen.add(key)
                    return args, kwargs

            layers = tuple(streamed.base_model.layers)
            for name, module in list(streamed.named_modules()):
                if module in layers or any(True for _ in module.parameters(recurse=False)):
                    add_hook_to_module(module, Delivered(name), append=True)
            tokens = torch.tensor([[1, 7, 5, 9, 4]], dtype=torch.long)
            with precision(), torch.no_grad():
                expected = resident(tokens, use_cache=True)
                actual = streamed(tokens, use_cache=True)
                assert actual.logits.dtype == torch.float64
                np.testing.assert_array_equal(actual.logits.numpy(), expected.logits.numpy())
                suffix = torch.tensor([[8, 6]], dtype=torch.long)
                expected = resident(suffix, past_key_values=expected.past_key_values, use_cache=True)
                reads = weights.reads
                actual = streamed(suffix, past_key_values=actual.past_key_values, use_cache=True)
                np.testing.assert_array_equal(actual.logits.numpy(), expected.logits.numpy())
            # The prefill recovered every tensor; the suffix's forward reads no bank and widens each kept one.
            assert weights.reads == reads
            assert weights.peak_bank_bytes < weights.whole_bytes
            assert seen == set(original)
            assert not weights.held
            assert all(parameter.device.type == "meta" for parameter in streamed.parameters())
