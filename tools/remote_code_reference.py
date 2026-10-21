#!/usr/bin/env python3
"""Write the llada-tiny, dream-tiny and mimo-v2-tiny logits, and the LLaDA-MoE and LLaDA2 recordings,
with the released model code.

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

MiMo-V2.6's code reads transformers 5's masking and rotary utilities (its
config records 5.3.0), so mimo-v2-tiny runs under Dew's locked environment
(transformers 5.16.1), with --draw the first time: the release's own class
draws the fixture's weights (every parameter, the sinks and the balancing
bias too, from N(0, 0.1^2) at seed 0, as the class leaves the sinks empty)
and its token ids, written in the class's split layout:
  python tools/remote_code_reference.py mimo-v2-tiny --draw

LLaDA-MoE's and LLaDA2's fixtures (`RECORDED`) are recordings over seeds for
check 1 (tests/reference_error.py `assert_as_exact`, tools/recorded_references.py):
each seed draws its token ids again (`token_draw`), and the release's code
writes its fp32 logits and their float64 truth beside each. The release's
class draws the fixture's weights first, every floating tensor it registers
(the parameters and LLaDA2's expert_bias buffer) from N(0, 0.1^2) at seed 0,
the norms' scales about one:
  python tools/remote_code_reference.py llada2-tiny --draw
Recording runs in Dew's locked environment, which plans the seed count on
Dew's own forward (`record_case`); the release's code runs in a process of
its own, under the transformers its code reads. LLaDA2.1's and 2.2's read
transformers 5's masking and rotary records, so Dew's environment serves:
  python tools/remote_code_reference.py llada2-tiny llada22-tiny --record
LLaDA-MoE's read transformers 4's rotary table (its 'default' entry, which 5
drops), so they run in a second environment of the transformers their
configs record:
  uv venv /tmp/remote4 --python 3.12
  uv pip install --python /tmp/remote4/bin/python torch==2.5.1 \
      --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python /tmp/remote4/bin/python transformers==4.53.2 \
      diffusers==0.34.0 safetensors numpy huggingface_hub
  /tmp/remote4/bin/python tools/remote_code_reference.py llada-moe-tiny --draw
  python tools/remote_code_reference.py llada-moe-tiny llada-moe-v2-tiny --record \
      --python /tmp/remote4/bin/python
LLaDA2.0's code reads the same table, and computes 2.1's model: the two
files differ in the mask factory and the rotary plumbing alone.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"
RELEASES = {
    "llada-tiny": ("GSAI-ML/LLaDA-8B-Base", "0f2787f2d87eac5eed8a087d5ecd24277e6255b2",
                   ("configuration_llada.py", "modeling_llada.py"), "AutoModel", False),
    "dream-tiny": ("Dream-org/Dream-v0-Base-7B", "6572adb5535263e4d1a337b56942ba48b6dee2a9",
                   ("configuration_dream.py", "modeling_dream.py", "generation_utils.py"),
                   "AutoModel", False),
    "mimo-v2-tiny": ("XiaomiMiMo/MiMo-V2.6-Pro-MOPD", "6a7f202e8dbd0e57b3fe1da03917d08d75abc4b8",
                     ("configuration_mimo_v2.py", "modeling_mimo_v2.py"), "AutoModelForCausalLM", True),
    "llada-moe-tiny": ("inclusionAI/LLaDA-MoE-7B-A1B-Base", "daccaf1ccbf263a427bc9ac2a23b0b004fb275bf",
                       ("configuration_lladamoe.py", "modeling_lladamoe.py"), "AutoModel", True),
    "llada-moe-v2-tiny": ("GSAI-ML/LLaDA-MoE-v2-30B-A3B-Base", "2a513f7a4634e426b2ab030e6defb24001d0c847",
                          ("configuration_lladamoe_v2.py", "modeling_lladamoe_v2.py"), "AutoModel", True),
    "llada2-tiny": ("inclusionAI/LLaDA2.1-mini", "20e64e2ad21644d0e5248586ed9c942cdd45de0f",
                    ("configuration_llada2_moe.py", "modeling_llada2_moe.py"), "AutoModelForCausalLM", True),
    "llada22-tiny": ("inclusionAI/LLaDA2.2-flash", "9b78da3f9149f1e239fb6d15f20a326fde574b9f",
                     ("configuration_llada2_moe.py", "modeling_llada2_moe.py"), "AutoModelForCausalLM", True),
}
"""Each fixture's release: the Hub repo, the commit its code is read at, the
files that code is, the auto class its language model registers as, and
whether the code pins float32 (MiMo's router casts its logits with
`.type(torch.float32)`), so its float64 run is built and run under
`diffusers_wan_reference.float64`, which widens those pins and the rotary
tables built at init."""


@dataclass(frozen=True)
class Recorded:
    """A fixture recorded over seeds: `length` tokens a draw (`token_draw`), and the `block` LLaDA2's
    generate masks attention by (`block_mask`), which the test's model reads too, or None for a model
    that reads its whole row."""

    length: int = 16
    block: int | None = None


RECORDED = {"llada-moe-tiny": Recorded(), "llada-moe-v2-tiny": Recorded(),
            "llada2-tiny": Recorded(block=4), "llada22-tiny": Recorded(block=4)}
"""The fixtures recorded over seeds. LLaDA2's draws hold four blocks of four, where its releases'
generate cuts blocks of 32, so the blocks the mask keeps apart fit a draw; LLaDA2.2's routing
blocks are its config's block_size (8)."""


def token_draw(name: str):
    """Seed by seed, a recorded fixture's token ids `[1, length]`: uniform over its vocabulary, half of
    them, drawn apart, the mask id, as a masked-diffusion step reads them."""
    config = json.loads((FIXTURES / name / "config.json").read_text())
    length = RECORDED[name].length

    def draw(seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        ids = rng.integers(0, config["vocab_size"], (1, length))
        masked = np.where(rng.random((1, length)) < 0.5, config["mask_token_id"], ids)
        return {"input_ids": masked.astype(np.int32)}

    return draw


def block_mask(length: int, block: int, dtype: torch.dtype) -> torch.Tensor:
    """LLaDA2MoeModelLM.generate's attention mask over `length` tokens in blocks of `block`: the log of
    a block-lower-triangular one, 0 where a query's block reads a key's and -inf elsewhere."""
    blocks = -(-length // block)
    tril = torch.tril(torch.ones(blocks, blocks, dtype=dtype))
    return tril.repeat_interleave(block, 0).repeat_interleave(block, 1)[None, None, :length, :length].log()


def _scattered(model: torch.nn.Module) -> None:
    """Every floating tensor `model` registers, its buffers too, drawn from N(0, 0.1^2) at seed 0, a
    norm's scale about one, in the class's own order."""
    torch.manual_seed(0)
    with torch.no_grad():
        for key, tensor in model.state_dict(keep_vars=True).items():
            if tensor.is_floating_point():
                tensor.copy_(torch.randn_like(tensor) * 0.1 + (1.0 if key.endswith("norm.weight") else 0.0))


def released_model(name: str, workspace: Path, draw: bool = False) -> torch.nn.Module:
    """The release's own model class over the fixture's config and weights,
    or, with `draw`, weights it draws and writes as the fixture's."""
    repo, revision, files, auto, _ = RELEASES[name]
    directory = workspace / name
    directory.mkdir(parents=True)
    for file in files:
        shutil.copy(hf_hub_download(repo, file, revision=revision), directory / file)
    released = json.loads(Path(hf_hub_download(repo, "config.json", revision=revision)).read_text())
    config = json.loads((FIXTURES / name / "config.json").read_text())
    (directory / "config.json").write_text(json.dumps({**config, "auto_map": released["auto_map"]}))
    reader = getattr(transformers, auto)
    model = reader.from_config(AutoConfig.from_pretrained(str(directory), trust_remote_code=True),
                               trust_remote_code=True, attn_implementation="eager")
    if draw and name in RECORDED:
        _scattered(model)
        save_file({key: value.contiguous() for key, value in model.state_dict().items()},
                  str(FIXTURES / name / "model.safetensors"))
    elif draw:
        torch.manual_seed(0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.copy_(torch.randn_like(parameter) * 0.1)
        save_file({key: value.contiguous() for key, value in model.state_dict().items()},
                  str(FIXTURES / name / "model.safetensors"))
        vocabulary = json.loads((FIXTURES / name / "config.json").read_text())["vocab_size"]
        ids = torch.randint(0, vocabulary, (2, 16)).numpy().astype(np.int32)
        np.save(FIXTURES / name / "input_ids.npy", ids)
    model.load_state_dict(load_file(str(FIXTURES / name / "model.safetensors")), strict=True)
    return model.eval()


def released_logits(name: str, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The release's fp32 logits for each draw's token ids in `ids` `[draws, 1, length]`, and their
    float64 truth, under its generate's block mask where the fixture names a block."""
    block = RECORDED[name].block

    def run(model: torch.nn.Module, dtype: torch.dtype) -> np.ndarray:
        given = torch.from_numpy(ids.astype(np.int64))
        mask = {} if block is None else {"attention_mask": block_mask(given.shape[-1], block, dtype)}
        return np.stack([model(input_ids=row, **mask).logits.to(dtype).numpy() for row in given])

    from diffusers_wan_reference import float64

    with tempfile.TemporaryDirectory() as workspace, torch.no_grad():
        fp32 = run(released_model(name, Path(workspace) / "fp32").float(), torch.float32)
        with float64():
            f64 = run(released_model(name, Path(workspace) / "f64").double(), torch.float64)
    return fp32, f64


def record(name: str, python: str) -> Path:
    """Record `name` over seeds (`record_case`) beside its provenance: the release's code run by
    `python`, the seeds planned on Dew's own forward of the loaded fixture, in fp32 and in float64."""
    import jax.numpy as jnp

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from recorded_references import provenance, record_case, save
    from reference_error import widened

    from dew.interop import Pretrained

    source = Pretrained.load(FIXTURES / name, dtype="float32", attention_impl="reference")
    block = RECORDED[name].block
    model = source.model if block is None else source.model.clone(diffusion_block=block)
    wide, wide_variables = model.clone(dtype=jnp.float64), widened(source.variables)
    draw, outputs = token_draw(name), {}

    def evaluate(inputs):
        # The release runs a process per batch of draws: the seeds record_case asks for next.
        key = inputs["input_ids"].tobytes()
        if key not in outputs:
            first = len(outputs)
            batch = [draw(seed)["input_ids"] for seed in range(first, first + 40)]
            with tempfile.TemporaryDirectory() as workspace:
                np.save(Path(workspace) / "ids.npy", np.stack(batch))
                subprocess.run([python, __file__, name, "--evaluate", workspace], check=True)
                with np.load(Path(workspace) / "logits.npz") as written:
                    for index, ids in enumerate(batch):
                        outputs[ids.tobytes()] = (written["fp32"][index], written["f64"][index])
        fp32, f64 = outputs[key]
        return {"logits": fp32}, {"logits": f64}

    recording = record_case(
        name, draw, evaluate,
        lambda inputs: {"logits": model.apply(source.variables, inputs["input_ids"])},
        truth_proxy=lambda inputs: {"logits": wide.apply(wide_variables, inputs["input_ids"])},
        store_inputs=False)
    repo, revision = RELEASES[name][:2]
    reader = subprocess.run([python, "-c", "import transformers; print(transformers.__version__)"],
                            check=True, capture_output=True, text=True).stdout.strip()
    recording["provenance"] = provenance(
        repo, revision, "cpu", f"python tools/remote_code_reference.py {name} --record",
        transformers=reader, code=list(RELEASES[name][2]))
    return save(FIXTURES / name / "recording.npz", recording, reference=repo, license="apache-2.0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*", default=sorted(set(RELEASES) - set(RECORDED)))
    parser.add_argument("--check", action="store_true", help="compare with logits.npy instead of writing")
    parser.add_argument("--draw", action="store_true", help="draw the fixture's weights and ids first")
    parser.add_argument("--record", action="store_true", help="record a RECORDED fixture over seeds")
    parser.add_argument("--python", default=sys.executable, help="the interpreter the release's code runs in")
    parser.add_argument("--evaluate", help="a directory whose ids.npy the release's code runs (`record`)")
    arguments = parser.parse_args()
    if arguments.evaluate is not None:
        (name,) = arguments.names
        fp32, f64 = released_logits(name, np.load(Path(arguments.evaluate) / "ids.npy"))
        np.savez(Path(arguments.evaluate) / "logits.npz", fp32=fp32, f64=f64)
        return
    if arguments.record:
        for name in arguments.names:
            print(name, record(name, arguments.python))
        return
    with tempfile.TemporaryDirectory() as workspace:
        for name in arguments.names:
            if name in RECORDED:
                released_model(name, Path(workspace) / name, arguments.draw)
                continue
            model = released_model(name, Path(workspace) / "fp32", arguments.draw)
            ids = torch.from_numpy(np.load(FIXTURES / name / "input_ids.npy").astype(np.int64))
            with torch.no_grad():
                fp32 = model.float()(input_ids=ids).logits.numpy()
                if RELEASES[name][4]:
                    from diffusers_wan_reference import float64

                    with float64():
                        wide = released_model(name, Path(workspace) / "f64").double()
                        f64 = wide(input_ids=ids).logits.numpy()
                else:
                    f64 = model.double()(input_ids=ids).logits.numpy()
            if arguments.check:
                same = np.array_equal(np.load(FIXTURES / name / "logits.npy"), fp32)
                print(name, "logits.npy bitwise" if same else "logits.npy differs")
            else:
                np.save(FIXTURES / name / "logits.npy", fp32)
                np.save(FIXTURES / name / "logits_f64.npy", f64)
                print(name, fp32.shape)


if __name__ == "__main__":
    main()
