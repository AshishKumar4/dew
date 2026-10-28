"""The independent author's float64 model receives exact tensors through the existing bank source."""

import copy
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


COMMIT = "0" * 40
"""The commit the synthetic records name, one qualification's."""


def recorded_runs(tool, tmp_path, options, logits, *, count=8, pool_commit=COMMIT):
    """A verdict's local source (a config and a float32 safetensors shard), request pool and three phase
    records over it, made on the CPU at float32 in one run of COMMIT: `count` verdict draws and their
    redraw, twice `count` pilots from SIZING_FROM, two questions of `options` options a request, and
    each question's logits `logits(phase, seed, question)`; the pool names `pool_commit`, where one."""
    from reference_error import REDRAW, SIZING_FROM
    from safetensors.numpy import save_file

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text('{"model_type":"qwen3_5"}')
    save_file({"model.embed_tokens.weight": np.zeros((4, 2), np.float32)}, source / "model.safetensors")
    verdict, pilot = list(range(count * (1 + REDRAW))), list(range(SIZING_FROM, SIZING_FROM + 2 * count))
    pool = {**({} if pool_commit is None else {"commit": pool_commit}), "count": count, "verdict": verdict,
            "pilot": pilot,
            "requests": {str(seed): {"questions": {
                f"case_{seed}_{index}": {"criteria": [f"option {option}" for option in range(options)]}
                for index in range(2)}} for seed in (*verdict, *pilot)}}
    pool_file = tmp_path / "pool.json"
    pool_file.write_text(json.dumps(pool))
    run = {"commit": COMMIT, "pool_sha256": tool.file_sha256(pool_file), "device": "cpu",
           "source_sha256": tool.manifest_sha256(tool.source_identity(source))}
    for phase in ("native", "author", "truth"):
        dtype = "float64" if phase == "truth" else "float32"
        np.savez(tmp_path / f"{phase}.npz", **{
            name: np.asarray(logits(phase, seed, index), dtype) for seed in (*pilot, *verdict)
            for index, name in enumerate(pool["requests"][str(seed)]["questions"])})
        made = {**run, "phase": phase, "compute_dtype": dtype}
        metadata = {**made, "parameter_dtypes": [dtype], "precision": tool.IEEE, "floating_operations": 1,
                    "cuda_floating_operations": 0}
        if phase != "native":
            metadata["requests"] = [{"seed": seed, **made} for seed in (*pilot, *verdict)]
        (tmp_path / f"{phase}.json").write_text(json.dumps(metadata))
    return SimpleNamespace(source=source, pool=pool_file, records=tmp_path, dtype="float32", device="cpu",
                           output=tmp_path / "verdict.json"), pool


def edited(path, **fields):
    """The record at `path` with `fields` set, None deleting one."""
    record = json.loads(path.read_text())
    record.update(fields)
    path.write_text(json.dumps({name: value for name, value in record.items() if value is not None}))


@pytest.mark.parametrize("changed", [None, "pool", "source", "tokenizer", "device", "dtype", "phase",
                                     "native commit", "author commit", "truth commit", "pool commit",
                                     "mixed commits", "request commit", "requests", "dropped request",
                                     "pilots", "parameters", "precision"])
def test_a_runtime_verdict_refuses_records_from_a_different_experiment(tmp_path, changed):
    """A verdict judges the three phases' records only where each, and each request record within them,
    was made in one run of the commit the pool names: over this pool, from this source's bytes (its
    tokenizer files among them), at the declared device, dtype and phase, native's pilots and verdict
    draws in one record, each arm's parameters at its storage policy over the release's stored dtypes
    and the author's products at IEEE float32, the author's and truth's request records one for each
    pilot and verdict draw in order. Any one of them changed refuses the verdict before it writes one; a
    verdict it writes names its own commit."""
    with run_from("slot_runtime_qualification", "tools") as tool:
        from reference_source import dew_revision

        # One option a question: its log-probability is exactly zero in every precision, so the drawn
        # errors are zero and the admitted records size at their fewest seeds.
        args, pool = recorded_runs(tool, tmp_path, 1, lambda phase, seed, question: np.zeros(1),
                                   pool_commit=None if changed == "pool commit" else COMMIT)
        if changed == "pool":
            args.pool.write_text(json.dumps({**pool, "population": "a different request set"}))
        elif changed == "source":
            (args.source / "config.json").write_text('{"model_type":"a different source"}')
        elif changed == "tokenizer":
            (args.source / "vocab.txt").write_text("a changed tokenizer\n")
        elif changed in ("native commit", "author commit", "truth commit"):
            edited(tmp_path / f"{changed.split()[0]}.json", commit=None)
        elif changed == "mixed commits":
            edited(tmp_path / "truth.json", commit="1" * 40)
        elif changed == "request commit":
            metadata = json.loads((tmp_path / "author.json").read_text())
            metadata["requests"][3]["commit"] = "1" * 40
            (tmp_path / "author.json").write_text(json.dumps(metadata))
        elif changed == "requests":
            edited(tmp_path / "author.json", requests=None)
        elif changed == "dropped request":
            metadata = json.loads((tmp_path / "truth.json").read_text())
            del metadata["requests"][5]
            (tmp_path / "truth.json").write_text(json.dumps(metadata))
        elif changed == "pilots":
            with np.load(tmp_path / "native.npz") as held:
                verdict = {name for seed in pool["verdict"]
                           for name in pool["requests"][str(seed)]["questions"]}
                np.savez(tmp_path / "native.npz", **{name: held[name] for name in verdict})
        elif changed == "parameters":
            edited(tmp_path / "native.json", parameter_dtypes=["bfloat16"])
        elif changed == "precision":
            edited(tmp_path / "author.json", precision={**tool.IEEE, "float32_matmul": "high"})
        elif changed in ("device", "dtype", "phase"):
            field = "compute_dtype" if changed == "dtype" else changed
            edited(tmp_path / "author.json", **{field: {"device": "cuda", "dtype": "bfloat16",
                                                        "phase": "truth"}[changed]})
        if changed is None:
            tool.verdict(args, pool)
            assert json.loads(args.output.read_text())["commit"] == dew_revision()
        else:
            with pytest.raises(ValueError):
                tool.verdict(args, pool)
            assert not args.output.exists()


@pytest.mark.parametrize("changed", [None, "phase", "commit", "source"])
def test_a_resumed_phase_reads_back_only_the_requests_its_own_run_recorded(tmp_path, changed):
    """A stopped phase resumes from the request records it wrote, kept apart by phase; a record made
    under another phase, commit or source, in this phase's place, is refused rather than read back
    under this run's provenance."""
    with run_from("slot_runtime_qualification", "tools") as tool:
        run = {"phase": "author", "commit": COMMIT, "pool_sha256": "1" * 64, "source_sha256": "2" * 64,
               "compute_dtype": "bfloat16", "device": "cuda"}
        recorded = {**run, **{"phase": {"phase": "truth"}, "commit": {"commit": "3" * 40},
                              "source": {"source_sha256": "4" * 64}}.get(changed, {})}
        values = {"case_0_0": np.arange(3, dtype=np.float32)}
        tool.Partial(tmp_path, recorded).write(0, values, {"seed": 0, **recorded})
        if changed == "phase":
            # Each phase reads its own records alone; one copied into another phase's place is refused.
            assert tool.Partial(tmp_path, run).read(0) is None
            (tmp_path / "truth" / "0.npz").rename(tmp_path / "author" / "0.npz")
            (tmp_path / "truth" / "0.json").rename(tmp_path / "author" / "0.json")
        resumed = tool.Partial(tmp_path, run)
        assert resumed.read(1) is None
        if changed is None:
            held, spent = resumed.read(0)
            np.testing.assert_array_equal(held["case_0_0"], values["case_0_0"])
            assert spent["commit"] == COMMIT
        else:
            with pytest.raises(ValueError):
                resumed.read(0)


@pytest.mark.parametrize("native", ["tracking", "worse", "independent"])
def test_a_runtime_verdict_sizes_native_paired_with_the_author_on_their_shared_rounding(tmp_path, native):
    """The author's error from the float64 truth spans three decades across draws, so the author
    compared with itself resolves nothing on the pilots it holds. Native read the same weights and
    inputs, its error the author's own a draw, so paired with the author it sizes at the fewest seeds
    and the verdict judges it there: within the author's error it is green, at three times it fails,
    and an error that does not track the author's leaves the count unresolved."""
    def logits(phase, seed, question):
        rng = np.random.default_rng([0x1510, seed, question])
        truth, scale = rng.standard_normal(2), 10 ** rng.uniform(-4, -1)
        apart = 10 ** rng.uniform(-4, -1)
        error = {"truth": 0.0, "author": scale, "native": {"tracking": 1.01 * scale, "worse": 3 * scale,
                                                           "independent": apart}[native]}[phase]
        return truth + error * np.asarray([1.0, -1.0])

    with run_from("slot_runtime_qualification", "tools") as tool:
        args, pool = recorded_runs(tool, tmp_path, 2, logits)
        if native == "tracking":
            tool.verdict(args, pool)
            assert json.loads(args.output.read_text())["verdict"] == "green"
        else:
            with pytest.raises(AssertionError):
                tool.verdict(args, pool)
            assert not args.output.exists()


def test_a_common_shift_of_a_phases_logits_moves_none_of_its_probabilities(tmp_path):
    """The claim is each question's options' probabilities, so a shift every option of a phase shares
    is out of it: native's equal options at 2**24, exact in float32, are the author's and the truth's
    equal options at zero, one half each, and the verdict holds them equal."""
    def logits(phase, seed, question):
        return np.full(2, 2.0 ** 24 if phase == "native" else 0.0)

    with run_from("slot_runtime_qualification", "tools") as tool:
        args, pool = recorded_runs(tool, tmp_path, 2, logits)
        tool.verdict(args, pool)
        assert json.loads(args.output.read_text())["verdict"] == "green"


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
