#!/usr/bin/env python3
"""Record the LLaDA-MoE and LLaDA2 fixtures over seeds with each release's own modeling code.

LLaDA-MoE and LLaDA2 ship no transformers class: their Hub repos carry their
own configuration and modeling files, loaded with trust_remote_code. This
fetches those files at a pinned commit and builds the model from the fixture's
tiny config, the released config's auto_map added so transformers finds the
classes. Each fixture is a recording for check 1 (tests/reference_error.py
`assert_as_exact`, tools/recorded_references.py): each seed draws its token
ids again (`token_draw`), and the release's code writes its fp32 logits and
their float64 truth beside each.

The release's class draws the fixture's weights first (--draw), every floating
tensor it registers (the parameters and LLaDA2's expert_bias buffer) from
N(0, 0.1^2) at seed 0, a norm's scale about one:
  python tools/masked_moe_reference.py llada2-tiny llada22-tiny --draw
Recording runs in Dew's locked environment, which plans the seed count on
Dew's own forward (`record_case`); the release's code runs in a process of its
own, under the transformers its code reads. LLaDA2.1's and 2.2's read
transformers 5's masking and rotary records, so Dew's environment serves:
  python tools/masked_moe_reference.py llada2-tiny llada22-tiny --record
LLaDA-MoE's read transformers 4's rotary table (its 'default' entry, which 5
drops), so they run in a second environment of the transformers their configs
record:
  uv venv /tmp/remote4 --python 3.12
  uv pip install --python /tmp/remote4/bin/python torch==2.5.1 \\
      --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python /tmp/remote4/bin/python transformers==4.53.2 \\
      diffusers==0.34.0 safetensors numpy huggingface_hub
  /tmp/remote4/bin/python tools/masked_moe_reference.py llada-moe-tiny llada-moe-v2-tiny --draw
  python tools/masked_moe_reference.py llada-moe-tiny llada-moe-v2-tiny --record \\
      --python /tmp/remote4/bin/python
LLaDA2.0's code reads the same table, and computes 2.1's model: the two files
differ in the mask factory and the rotary plumbing alone.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import transformers
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "hf"


@dataclass(frozen=True)
class Recorded:
    """A fixture's release, the Hub repo and the commit its code is read at, the files that code is
    and the auto class its model registers as; the tokens a draw holds (`token_draw`); and the
    `block` LLaDA2's generate masks attention by (`block_mask`), which the test's model reads too,
    or None for a model that reads its whole row."""

    repo: str
    revision: str
    files: tuple[str, ...]
    auto: str
    length: int = 16
    block: int | None = None


_LLADA2_FILES = ("configuration_llada2_moe.py", "modeling_llada2_moe.py")
RECORDED = {
    "llada-moe-tiny": Recorded(
        "inclusionAI/LLaDA-MoE-7B-A1B-Base", "daccaf1ccbf263a427bc9ac2a23b0b004fb275bf",
        ("configuration_lladamoe.py", "modeling_lladamoe.py"), "AutoModel"),
    "llada-moe-v2-tiny": Recorded(
        "GSAI-ML/LLaDA-MoE-v2-30B-A3B-Base", "2a513f7a4634e426b2ab030e6defb24001d0c847",
        ("configuration_lladamoe_v2.py", "modeling_lladamoe_v2.py"), "AutoModel"),
    "llada2-tiny": Recorded("inclusionAI/LLaDA2.1-mini", "20e64e2ad21644d0e5248586ed9c942cdd45de0f",
                            _LLADA2_FILES, "AutoModelForCausalLM", block=4),
    "llada22-tiny": Recorded("inclusionAI/LLaDA2.2-flash", "9b78da3f9149f1e239fb6d15f20a326fde574b9f",
                             _LLADA2_FILES, "AutoModelForCausalLM", block=4),
}
"""The fixtures recorded over seeds. LLaDA2's draws hold four blocks of four, where its releases'
generate cuts blocks of 32, so the blocks the mask keeps apart fit a draw; LLaDA2.2's routing
blocks are its config's block_size (8)."""

DRAWS_A_PROCESS = 40
"""The draws one process of the release's code runs while recording: a process starts in seconds,
so a batch of draws keeps the recording a few processes long, and record_case asks for at most five
times the seeds it plans."""


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


def released_class(name: str, workspace: Path) -> torch.nn.Module:
    """The release's own model class over the fixture's config, its weights as the class draws them."""
    release = RECORDED[name]
    directory = Path(snapshot_download(release.repo, revision=release.revision, local_dir=workspace / name,
                                       allow_patterns=[*release.files, "config.json"]))
    auto_map = json.loads((directory / "config.json").read_text())["auto_map"]
    fixture = json.loads((FIXTURES / name / "config.json").read_text())
    (directory / "config.json").write_text(json.dumps({**fixture, "auto_map": auto_map}))
    config = AutoConfig.from_pretrained(str(directory), trust_remote_code=True)
    return getattr(transformers, release.auto).from_config(config, trust_remote_code=True,
                                                           attn_implementation="eager")


def draw_weights(name: str) -> None:
    """Write the fixture's weights: every floating tensor the release's class registers, its buffers
    too, drawn from N(0, 0.1^2) at seed 0, a norm's scale about one, in the class's own order."""
    with tempfile.TemporaryDirectory() as workspace:
        model = released_class(name, Path(workspace))
    torch.manual_seed(0)
    with torch.no_grad():
        for key, tensor in model.state_dict(keep_vars=True).items():
            if tensor.is_floating_point():
                tensor.copy_(torch.randn_like(tensor) * 0.1 + (1.0 if key.endswith("norm.weight") else 0.0))
    save_file({key: value.contiguous() for key, value in model.state_dict().items()},
              str(FIXTURES / name / "model.safetensors"))


def released_logits(name: str, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The release's fp32 logits for each draw's token ids in `ids` `[draws, 1, length]`, and their
    float64 truth, under its generate's block mask where the fixture names a block."""
    from diffusers_wan_reference import float64

    block = RECORDED[name].block
    weights = load_file(str(FIXTURES / name / "model.safetensors"))

    def run(workspace: Path, dtype: torch.dtype) -> np.ndarray:
        model = released_class(name, workspace)
        model.load_state_dict(weights, strict=True)
        model = model.eval().to(dtype)
        given = torch.from_numpy(ids.astype(np.int64))
        mask = {} if block is None else {"attention_mask": block_mask(given.shape[-1], block, dtype)}
        return np.stack([model(input_ids=row, **mask).logits.to(dtype).numpy() for row in given])

    with tempfile.TemporaryDirectory() as workspace, torch.no_grad():
        fp32 = run(Path(workspace) / "fp32", torch.float32)
        with float64():
            f64 = run(Path(workspace) / "f64", torch.float64)
    return fp32, f64


def record(name: str, python: str) -> Path:
    """Record `name` over seeds (`record_case`) beside its provenance: the release's code run by
    `python`, the seeds planned on Dew's own forward of the loaded fixture, in fp32 and in float64."""
    import jax.numpy as jnp
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
            batch = [draw(seed)["input_ids"] for seed in range(first, first + DRAWS_A_PROCESS)]
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
    release = RECORDED[name]
    reader = subprocess.run([python, "-c", "import transformers; print(transformers.__version__)"],
                            check=True, capture_output=True, text=True).stdout.strip()
    recording["provenance"] = provenance(
        release.repo, release.revision, "cpu", f"python tools/masked_moe_reference.py {name} --record",
        transformers=reader, code=list(release.files))
    return save(FIXTURES / name / "recording.npz", recording, reference=release.repo, license="apache-2.0")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="+", choices=sorted(RECORDED))
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--draw", action="store_true", help="draw the weights with the release's class")
    mode.add_argument("--record", action="store_true", help="record the fixtures over seeds")
    mode.add_argument("--evaluate", help="a directory whose ids.npy the release's code runs (`record`)")
    parser.add_argument("--python", default=sys.executable, help="the interpreter the release's code runs in")
    arguments = parser.parse_args()
    for name in arguments.names:
        if arguments.draw:
            draw_weights(name)
        elif arguments.record:
            print(name, record(name, arguments.python))
        else:
            fp32, f64 = released_logits(name, np.load(Path(arguments.evaluate) / "ids.npy"))
            np.savez(Path(arguments.evaluate) / "logits.npz", fp32=fp32, f64=f64)


if __name__ == "__main__":
    main()
