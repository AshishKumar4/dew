#!/usr/bin/env python3
"""Write the llada-tiny, dream-tiny and mimo-v2-tiny logits with the released model code.

LLaDA and Dream ship no transformers class: their Hub repos carry their own
configuration and modeling files, loaded with trust_remote_code. This fetches
those files at a pinned commit, builds the model from the fixture's tiny
config (the released config's auto_map added, so transformers finds the
classes), loads the fixture's committed weights and runs its committed token
ids, in fp32 for logits.npy and in float64 for logits_f64.npy, the exact
value the tests measure both fp32 runs from (tests/reference_error.py).

The weights and ids are inputs, read here as they are: drawn by
tools/hf_reference.py's write_diffusion_tiny as of commit 18511496038f
(dew.interop.verify's scatter_weights at seed 1234 over a port of each
release's module order, which drawing over the release's own order does
not reproduce, and probe_ids).

The remote code was written against transformers 4.46 (both released
config.json files record 4.46.x, and later releases drop attributes it
reads), so run it there:
  uv venv /tmp/remote --python 3.12
  uv pip install --python /tmp/remote/bin/python torch==2.5.1 \
      --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python /tmp/remote/bin/python transformers==4.46.3 \
      safetensors numpy huggingface_hub
  /tmp/remote/bin/python tools/remote_code_reference.py

MiMo-V2.6's code reads transformers 5's masking and rotary utilities, calling
`create_causal_mask(input_embeds=...)`, which 5.2.0 renamed (its config records
5.3.0, which refuses it), so mimo-v2-tiny runs under transformers 5.1.0, with
--draw the first time: the release's own class
draws the fixture's weights (every parameter, the sinks and the balancing
bias too, from N(0, 0.1^2) at seed 0, as the class leaves the sinks empty)
and its token ids, written in the class's split layout:
  python tools/remote_code_reference.py mimo-v2-tiny --draw

`--padding` records instead the rows each release's own token lookup leaves
untrained, under the fixture's config with its pad id stated, left out and
null (`reference_source.record_padding`), as padding.json beside it. It reads
the release's code and config alone, under each release's transformers as
above:
  /tmp/remote/bin/python tools/remote_code_reference.py --padding llada-tiny dream-tiny

LFM3's code (lfm3-tiny, lfm3-biased-tiny) runs under transformers 5.18.0 on
(tests/fixtures/hf/lfm3-tiny/source.json).

`--record` records each release over seeds at its sites (tests/remote_code_draws.py's `SITES`,
`record_case(..., summary=True)`), as `released.npz` beside the fixture, which
tests/test_remote_code_releases.py holds Dew to by check 1: Dew's side runs in Dew's environment, and
the release's, served over a pipe (`--serve`), under `--release-python`, its own transformers':
  PYTHONPATH=src python tools/remote_code_reference.py lfm3-tiny lfm3-biased-tiny --record \
      --release-python <a Python with transformers 5.18.0>
  PYTHONPATH=src python tools/remote_code_reference.py mimo-v2-tiny --record --release-python <5.1.0>
  PYTHONPATH=src python tools/remote_code_reference.py dream-tiny llada-tiny --record \
      --release-python <4.46.3>
`--opens <export>` checks a Dew export of the fixture opens through the release's own loader as the
release does (`opens`).
"""

import argparse
import io
import json
import runpy
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download
from reference_source import config_lookups, record_padding
from safetensors.torch import load_file, save_file
from torch_precision import float64, float64_twin
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
# NumPy alone, as the release's environment holds no JAX.
from remote_code_draws import CHUNKS, PADDED, SITES, draw, positions

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
_PINNED = runpy.run_path(str(Path(__file__).resolve().parents[1] / "src" / "dew" / "interop"
                             / "release_pins.py"))
"""The releases whose own code is a Dew family's reference (dew.interop.release_pins), read by path: this
tool runs in the release's own environment, which need not hold Dew's."""

RELEASES = {
    "llada-tiny": ("GSAI-ML/LLaDA-8B-Base", "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
                   ("configuration_llada.py", "modeling_llada.py"), "AutoModel"),
    "dream-tiny": ("Dream-org/Dream-v0-Base-7B", "6572adb5535263e4d1a337b56942ba48b6dee2a9",
                   ("configuration_dream.py", "modeling_dream.py", "generation_utils.py"),
                   "AutoModel"),
    "mimo-v2-tiny": ("XiaomiMiMo/MiMo-V2.6-Pro-MOPD", "6a7f202e8dbd0e57b3fe1da03917d08d75abc4b8",
                     ("configuration_mimo_v2.py", "modeling_mimo_v2.py"), "AutoModelForCausalLM"),
    # The family's own pin (dew.interop.release_pins). modeling_lfm3.py imports its
    # Triton conv from lfm3_conv.py, which its CPU run passes by.
    "lfm3-tiny": (*_PINNED["LFM3"][:2],
                  ("configuration_lfm3.py", "modeling_lfm3.py", "lfm3_conv.py"), "AutoModelForCausalLM"),
    # The same release with its convolutions' bias (`conv_bias`), which its padding masks around.
    "lfm3-biased-tiny": (*_PINNED["LFM3"][:2], ("configuration_lfm3.py", "modeling_lfm3.py", "lfm3_conv.py"),
                         "AutoModelForCausalLM"),
}
LICENSES = {"lfm3-tiny": "lfm1.0", "lfm3-biased-tiny": "lfm1.0", "mimo-v2-tiny": "mit",
            "dream-tiny": "apache-2.0", "llada-tiny": "mit"}
"""The license of each release a recording (`--record`) runs, as its Hub card states it."""
"""Each fixture's release: the Hub repo, the commit its code is read at, the
files that code is, and the auto class its language model registers as. Each
release's float64 run is built and run under `torch_precision.float64`, which
widens the pins its code carries (MiMo's router casts its logits with
`.type(torch.float32)`) and computes the tables its constructor builds (a
rotary `inv_freq`) in float64."""


def released_classes(name: str, workspace: Path):
    """The release's own config of a dict (its code's files at its commit, beside the dict with the
    release's `auto_map`, in `workspace`) and its model class over such a config."""
    repo, revision, files, auto = RELEASES[name]
    directory = workspace / name
    directory.mkdir(parents=True)
    for file in files:
        shutil.copy(hf_hub_download(repo, file, revision=revision), directory / file)
    released = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())

    def configured(config: dict):
        (directory / "config.json").write_text(json.dumps({**config, "auto_map": released["auto_map"]}))
        return AutoConfig.from_pretrained(str(directory), trust_remote_code=True)

    def model(config) -> torch.nn.Module:
        return getattr(transformers, auto).from_config(config, trust_remote_code=True,
                                                       attn_implementation="eager")
    return configured, model


def released_model(name: str, scratch: Path, draw: bool = False) -> torch.nn.Module:
    """The release's own model class over the fixture's config and weights,
    or, with `draw`, weights it draws and writes as the fixture's."""
    configured, built = released_classes(name, scratch)
    model = built(configured(json.loads((FIXTURES / name / "config.json").read_text())))
    if draw:
        torch.manual_seed(0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.1)
        save_file(_stored_once(model.state_dict()), str(FIXTURES / name / "model.safetensors"))
        vocabulary = json.loads((FIXTURES / name / "config.json").read_text())["vocab_size"]
        ids = torch.randint(0, vocabulary, (2, 16)).numpy().astype(np.int32)
        np.save(FIXTURES / name / "input_ids.npy", ids)
    state = model.state_dict()
    missing, unexpected = model.load_state_dict(load_file(str(FIXTURES / name / "model.safetensors")),
                                                strict=False)
    # A tied head is its embedding's tensor, which the file stores once (`_stored_once`).
    shared = set(state) - set(_stored_once(state))
    if unexpected or set(missing) != shared:
        raise ValueError(f"{name}: the file's tensors differ from the model's: missing {missing}, "
                         f"unexpected {unexpected}")
    return model.eval()


def _stored_once(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """`state` with each tensor once, under its first name, as `save_pretrained` writes a tied head:
    the embedding's alone."""
    kept: dict[str, torch.Tensor] = {}
    seen = set()
    for key, value in state.items():
        if value.data_ptr() not in seen:
            seen.add(value.data_ptr())
            kept[key] = value.contiguous()
    return kept


OPENED = """import json, sys, numpy, torch, transformers
loader = getattr(transformers, sys.argv[1])
ids = torch.from_numpy(numpy.load(sys.argv[2]).astype('int64'))
opened = []
for directory in sys.argv[3:]:
    model, info = loader.from_pretrained(directory, trust_remote_code=True, dtype=torch.float32,
                                         output_loading_info=True)
    with torch.no_grad():
        logits = model(input_ids=ids).logits
    report = {key: sorted(value) for key, value in info.items()}
    opened.append((type(model).__name__, report, model.state_dict(), logits))
(name, info, state, logits), (other, other_info, other_state, other_logits) = opened
matched = info == other_info and not any(info.values())
print(json.dumps({"class": name == other, "every weight matched": matched,
                  "state": sorted(state) == sorted(other_state)
                  and all(torch.equal(state[key], other_state[key]) for key in state),
                  "logits bitwise": bool(torch.equal(logits, other_logits))}))
"""
"""A fresh interpreter's offline opens of two directories through their own code (`opens`), compared."""


def opens(name: str, export: Path, scratch: Path) -> None:
    """Whether `export`, a Dew save of `name`'s fixture (`PretrainedDecoder.from_model(...).save`), opens
    offline through the release's own transformers loader as the release does: one fresh interpreter, the
    hub offline and its module cache empty, opens the export, which reads its own code alone, and the
    release (the fixture's config and weights beside the release's code and auto_map at its commit), and
    finds the same class, every weight matched, the same state tensor for tensor and the same logits bit
    for bit over the fixture's ids (ledger 2038). Run in the release's environment, after Dew's writes
    the export:
      python -c 'from dew.interop import Pretrained, PretrainedDecoder; loaded = Pretrained.load(
          "tests/fixtures/hf/lfm3-tiny"); PretrainedDecoder.from_model(loaded.model, loaded.variables).save(
          "/tmp/lfm3-export")'
      <the release's Python> tools/remote_code_reference.py lfm3-tiny --opens /tmp/lfm3-export"""
    import os
    import subprocess
    import sys

    from huggingface_hub import hf_hub_download

    repo, revision, files, auto = RELEASES[name]
    release = scratch / "release"
    release.mkdir()
    released = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
    config = json.loads((FIXTURES / name / "config.json").read_text())
    (release / "config.json").write_text(json.dumps({**config, "auto_map": released["auto_map"]}))
    shutil.copy(FIXTURES / name / "model.safetensors", release / "model.safetensors")
    for file in files:
        shutil.copy(hf_hub_download(repo, file, revision=revision), release / file)
    # The release's own loader, the auto class its config registers it under.
    run = subprocess.run([sys.executable, "-c", OPENED, auto, str(FIXTURES / name / "input_ids.npy"),
                          str(export), str(release)], check=True, capture_output=True, text=True,
                         env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                              "HF_MODULES_CACHE": str(scratch / "modules")})
    checks = json.loads(run.stdout.strip().splitlines()[-1])
    print(name, "opens as the release does" if all(checks.values()) else f"differs: {checks}")
    if not all(checks.values()):
        raise SystemExit(1)


def send(stream, arrays: dict) -> None:
    """`arrays` over a pipe to the other side of a recording (`received`)."""
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    stream.write(struct.pack("<Q", len(buffer.getvalue())) + buffer.getvalue())
    stream.flush()


def received(stream) -> dict:
    """The arrays the other side of a recording sent (`send`)."""
    size = struct.unpack("<Q", stream.read(8))[0]
    with np.load(io.BytesIO(stream.read(size))) as arrays:
        return dict(arrays)


def released_outputs(model, site: str, inputs: dict) -> dict[str, np.ndarray]:
    """The release's outputs at a recorded `site` (tests/remote_code_draws.py), as Dew's forward
    gives them (tests/model_support.py's `remote_code_site`), the gradients by the release's own tensor
    names."""
    ids = torch.from_numpy(inputs["ids"].astype(np.int64))
    if site == "gradients":
        model.zero_grad()
        with torch.enable_grad():
            logits = model(input_ids=ids).logits
            (logits * torch.from_numpy(inputs["cotangent"]).to(logits.dtype)).sum().backward()
        return {name: parameter.grad.detach().numpy() for name, parameter in model.named_parameters()}
    with torch.no_grad():
        if site == "packed":
            return {"logits": model(input_ids=ids, position_ids=torch.from_numpy(
                positions().astype(np.int64))).logits.numpy()}
        if site == "decode":
            cache, steps, start = None, [], 0
            for chunk in CHUNKS:
                out = model(input_ids=ids[:, start:start + chunk], past_key_values=cache, use_cache=True)
                cache, start = out.past_key_values, start + chunk
                steps.append(out.logits)
            return {"logits": torch.cat(steps, dim=1).numpy()}
        if site == "biased":
            logits = model(input_ids=ids, attention_mask=torch.from_numpy(inputs["attention_mask"])).logits
            return {"row0": logits[0].numpy(), "row1": logits[1, PADDED:].numpy()}
        return {"logits": model(input_ids=ids).logits.numpy()}


def serve(name: str) -> None:
    """The release's side of a recording (`--record`): its fp32 outputs and their float64 truth at each
    site and draw the pipe brings."""
    with tempfile.TemporaryDirectory() as scratch:
        native = released_model(name, Path(scratch) / "fp32").float()
        # Its float64 twin (`float64_twin`): built inside `float64`, its constructor's tables float64's.
        wide = float64_twin(native, lambda: released_model(name, Path(scratch) / "f64"))
        while True:
            request = received(sys.stdin.buffer)
            site = str(request.pop("site"))
            reference = released_outputs(native, site, request)
            with float64():
                truth = released_outputs(wide, site, request)
            send(sys.stdout.buffer, {**{f"reference.{leaf}": value for leaf, value in reference.items()},
                                     **{f"truth.{leaf}": value for leaf, value in truth.items()}})


def record(name: str, release_python: str) -> None:
    """`name`'s recording over seeds at each of its sites (tests/remote_code_draws.py's `SITES`,
    `record_case(..., summary=True)`), the release served by `release_python` in the environment its code
    runs under, as `released.npz` beside the fixture: what its test holds Dew to by check 1. The
    gradients' leaves are the release's tensors, each read into Dew's parameter it loads as (its
    `WeightLayout.restore`)."""
    # Dew's side, in Dew's environment: the release's runs in `release_python`'s.
    from model_support import remote_code_site
    from recorded_references import provenance, record_case, save

    from dew.interop import Pretrained

    source = Pretrained.load(FIXTURES / name, dtype="float32", attention_impl="reference")
    vocabulary = int(json.loads((FIXTURES / name / "config.json").read_text())["vocab_size"])
    layouts = {layout.name: layout for layout in source.weight_layouts}
    shapes = {".".join(path): leaf.shape for path, leaf in _leaves(source.variables)}
    worker = subprocess.Popen([release_python, __file__, name, "--serve"], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE)
    arrays = {}
    try:
        for site in SITES[name]:
            def evaluate(inputs, site=site):
                send(worker.stdin, {"site": np.asarray(site), **inputs})
                returned = received(worker.stdout)
                arms = ({leaf.removeprefix(f"{kind}."): value for leaf, value in returned.items()
                         if leaf.startswith(f"{kind}.")} for kind in ("reference", "truth"))
                if site != "gradients":
                    return tuple(arms)
                return tuple({".".join(layouts[tensor].paths[0][1:]): layouts[tensor].restore(
                    value, shapes[".".join(layouts[tensor].paths[0])]) for tensor, value in arm.items()}
                    for arm in arms)

            arrays.update(record_case(
                f"{name} {site}", lambda seed, site=site: draw(site, seed, vocabulary), evaluate,
                remote_code_site(source, site), truth_proxy=remote_code_site(source, site, wide=True),
                store_inputs=False, summary=True))
    finally:
        worker.kill()
    repo, revision = RELEASES[name][:2]
    versions = subprocess.run([release_python, "-c", "import torch, transformers; "
                               "print(transformers.__version__, torch.__version__)"],
                              capture_output=True, text=True, check=True).stdout.split()
    arrays[f"{name}.provenance"] = provenance(
        repo, revision, "cpu",
        f"python tools/remote_code_reference.py {name} --record --release-python <python>",
        transformers=versions[0], torch=versions[1])
    print(save(FIXTURES / name / "released.npz", arrays, reference=repo, license=LICENSES[name]), flush=True)


def _leaves(tree, path=()):
    """`tree`'s leaves with their key paths."""
    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _leaves(value, (*path, key))
    else:
        yield path, tree


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=sorted(RELEASES))
    parser.add_argument("--check", action="store_true", help="compare with logits.npy instead of writing")
    parser.add_argument("--draw", action="store_true", help="draw the fixture's weights and ids first")
    parser.add_argument("--padding", action="store_true", help="record the rows each lookup leaves untrained")
    parser.add_argument("--opens", type=Path,
                        help="check that a Dew export of the fixture opens as the release does")
    parser.add_argument("--record", action="store_true", help="record the release over seeds at its sites")
    parser.add_argument("--serve", action="store_true", help="the release's side of --record's pipe")
    parser.add_argument("--release-python", default=sys.executable,
                        help="the Python whose transformers runs the release's code, for --record")
    arguments = parser.parse_args()
    if arguments.serve:
        serve(arguments.names[0])
        return
    if arguments.record:
        for name in arguments.names:
            record(name, arguments.release_python)
        return
    with tempfile.TemporaryDirectory() as scratch:
        if arguments.opens is not None:
            for name in arguments.names:
                opens(name, arguments.opens, Path(scratch))
            return
        if arguments.padding:
            for name in arguments.names:
                record_padding(*config_lookups(*released_classes(name, Path(scratch) / "padding")),
                               json.loads((FIXTURES / name / "config.json").read_text()), FIXTURES / name)
            return
        for name in arguments.names:
            model = released_model(name, Path(scratch) / "fp32", arguments.draw)
            ids = torch.from_numpy(np.load(FIXTURES / name / "input_ids.npy").astype(np.int64))
            with torch.no_grad():
                fp32 = model.float()(input_ids=ids).logits.numpy()
                # Its float64 twin (`float64_twin`), so the rotary tables its constructor computes are
                # float64's (a float32 build cast afterwards keeps float32's, Dream's `inv_freq`), run
                # inside `float64`, so a pin its forward carries widens too.
                wide = float64_twin(model, lambda: released_model(name, Path(scratch) / "f64"))
                with float64():
                    f64 = wide(input_ids=ids).logits.numpy()
            if arguments.check:
                same = np.array_equal(np.load(FIXTURES / name / "logits.npy"), fp32)
                print(name, "logits.npy bitwise" if same else "logits.npy differs")
            else:
                np.save(FIXTURES / name / "logits.npy", fp32)
                np.save(FIXTURES / name / "logits_f64.npy", f64)
                print(name, fp32.shape)


if __name__ == "__main__":
    main()
