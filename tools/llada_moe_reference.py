"""The pinned LLaDA MoE releases over synthetic checkpoints, recorded over seeds for check 1.

Each family's site (its fixture directory's name) draws two rows of 16 token ids a seed
(tests/model_support.py's `llada_draw`, which its test draws again) and records the release's fp32
forward's evidence against the release run in float64 (`record_case(..., summary=True)`): kilobytes,
committed as `released.npz` beside the fixture. Dew's own fp32 forward plans the seed count, and the
recorded release confirms it. LLaDA2's forwards run under its explicit block-causal mask of 4-token
blocks (`llada_fields`), which its 2.0 forward requires. Sampling-recipe parity is separate.

The releases' code reads its own transformers, LLaDA-MoE's, v2's and 2.0's 4.57.1 (which refuses the
huggingface_hub Dew reads) and 2.1's and 2.2's 5.16.1, so the release runs in a process of its own
(`--serve`, under `--release-python`), fed each draw over a pipe, beside Dew's: it imports torch and
the release loader alone. remote_code_reference's loader fetches config and Python only, never
weights; --draw replaces the committed synthetic weights first.
  python tools/llada_moe_reference.py llada-moe-tiny llada-moe-v2-tiny llada2-tiny \\
      --release-python <a Python with transformers 4.57.1>
  python tools/llada_moe_reference.py llada21-tiny llada22-tiny --release-python <transformers 5.16.1>

`--generate` records instead the release's own blockwise generation (its `generate`) of a few drawn
prompts under `GENERATE`, as generate.json beside the fixture, which tests/test_blockwise_walk.py holds
Dew's blockwise walk to; it runs the release's code alone, under the release's Python:
  <a Python with transformers 4.57.1> tools/llada_moe_reference.py --generate llada2-tiny

`--padding` records instead the rows each release's own token lookup leaves untrained under the
fixture's config with its pad id stated, left out and null (`reference_source.record_padding`), as
padding.json beside it; it runs the release's code alone, so the Python running it is the release's:
  <a Python with transformers 4.57.1> tools/llada_moe_reference.py --padding llada-moe-tiny ...
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from reference_source import config_lookups, record_padding
from remote_code_reference import (
    FIXTURES,
    RELEASES as REGISTERED,
    received,
    released_classes,
    released_model,
    send,
)
from safetensors.torch import save_file
from torch_precision import float64, float64_twin

RELEASES = {
    "llada-moe-tiny": ("inclusionAI/LLaDA-MoE-7B-A1B-Base", "daccaf1ccbf263a427bc9ac2a23b0b004fb275bf",
                       ("configuration_lladamoe.py", "modeling_lladamoe.py"), "AutoModelForCausalLM"),
    "llada-moe-v2-tiny": ("GSAI-ML/LLaDA-MoE-v2-30B-A3B-Base", "2a513f7a4634e426b2ab030e6defb24001d0c847",
                          ("configuration_lladamoe_v2.py", "modeling_lladamoe_v2.py"),
                          "AutoModelForCausalLM"),
    "llada2-tiny": ("inclusionAI/LLaDA2.0-mini", "dad945cac317da394b390f82c7b40691d8a881ed",
                    ("configuration_llada2_moe.py", "modeling_llada2_moe.py"), "AutoModelForCausalLM"),
    "llada21-tiny": ("inclusionAI/LLaDA2.1-mini", "20e64e2ad21644d0e5248586ed9c942cdd45de0f",
                     ("configuration_llada2_moe.py", "modeling_llada2_moe.py"), "AutoModelForCausalLM"),
    "llada22-tiny": ("inclusionAI/LLaDA2.2-flash", "9b78da3f9149f1e239fb6d15f20a326fde574b9f",
                     ("configuration_llada2_moe.py", "modeling_llada2_moe.py"), "AutoModelForCausalLM"),
}
LICENSES = {"llada-moe-tiny": "apache-2.0", "llada-moe-v2-tiny": "apache-2.0", "llada2-tiny": "apache-2.0",
            "llada21-tiny": "apache-2.0", "llada22-tiny": "apache-2.0"}
"""Each release's license, as its Hub card states it."""


def draw(model, directory):
    """Redraw every weight and balancing buffer, including the norms, at seed zero."""
    torch.manual_seed(0)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            noise = torch.randn_like(parameter) * 0.2
            parameter.copy_(noise + 1 if 'norm' in name else noise)
        for name, buffer in model.named_buffers():
            if name.endswith('expert_bias'):
                buffer.copy_(torch.randn_like(buffer) * 0.3)
    save_file({name: value.contiguous() for name, value in model.state_dict().items()},
              str(directory / 'model.safetensors'))


def released_logits(model, ids, visible):
    """The release's forward of `ids` `[2, 16]`, under `visible` `[16, 16]` as an additive mask where
    one is given (LLaDA2's block-causal mask)."""
    fields = {}
    if visible is not None:
        fields['attention_mask'] = torch.where(torch.from_numpy(visible), 0.0, -torch.inf).to(model.dtype)[
            None, None].expand(ids.shape[0], 1, -1, -1)
    return model(input_ids=ids, **fields).logits.detach().numpy()


def serve(name: str) -> None:
    """The release's side: its fp32 and float64 forwards of each draw the pipe brings."""
    REGISTERED.update(RELEASES)
    with tempfile.TemporaryDirectory() as workspace, torch.no_grad():
        native = released_model(name, Path(workspace) / 'fp32').float()
        # Its float64 twin (`float64_twin`): built inside `float64`, its constructor's tables float64's.
        wide = float64_twin(native, lambda: released_model(name, Path(workspace) / 'f64'))
        while True:
            request = received(sys.stdin.buffer)
            ids = torch.from_numpy(request['ids'].astype(np.int64))
            visible = request.get('visible')
            reference = released_logits(native, ids, visible)
            with float64():
                truth = released_logits(wide, ids, visible)
            send(sys.stdout.buffer, {'reference': reference, 'truth': truth})


def record(name: str, release_python: str) -> None:
    """`name`'s recording (`record_case`), the release served by `release_python`."""
    import jax.numpy as jnp
    from recorded_references import provenance, record_case, save

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tests'))
    from model_support import llada_draw, llada_fields
    from reference_error import widened

    from dew.interop import Pretrained

    repo, revision = RELEASES[name][:2]
    worker = subprocess.Popen([release_python, __file__, '--serve', name], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE)
    fields = llada_fields(name, 2, 16)
    mask = {} if not fields else {'visible': np.asarray(fields['attention_pairwise_mask'])[0]}

    def evaluate(inputs):
        send(worker.stdin, {'ids': inputs['ids'], **mask})
        returned = received(worker.stdout)
        return {'logits': returned['reference']}, {'logits': returned['truth']}

    source = Pretrained.load(FIXTURES / name, dtype='float32', attention_impl='reference')

    def forward(inputs):
        return {'logits': source.model.apply(source.variables, inputs['ids'], **fields)}

    wide_model, wide_variables = source.model.clone(dtype=jnp.float64), widened(source.variables)

    def wide_forward(inputs):
        return {'logits': wide_model.apply(wide_variables, inputs['ids'], **fields)}

    try:
        arrays = record_case(name, lambda seed: llada_draw(seed, source.model.vocab_size), evaluate, forward,
                             truth_proxy=wide_forward, store_inputs=False, summary=True)
    finally:
        worker.kill()
    versions = subprocess.run([release_python, '-c', 'import torch, transformers; '
                               'print(transformers.__version__, torch.__version__)'],
                              capture_output=True, text=True, check=True).stdout.split()
    arrays[f'{name}.provenance'] = provenance(
        repo, revision, 'cpu', f'python tools/llada_moe_reference.py {name} --release-python <python>',
        transformers=versions[0], torch=versions[1])
    print(save(FIXTURES / name / 'released.npz', arrays, reference=repo, license=LICENSES[name]), flush=True)


GENERATE = {"block_length": 4, "steps": 4, "gen_length": 8, "top_k": 1, "threshold": 0.95, "eos_id": 1}
"""The release's generation a recording runs (`generated`): blocks of four tokens over four steps each,
eight tokens, its best token alone (`top_k` 1, its one setting that draws nothing, each block's every
position then confident), token 1 its end."""

PROMPTS = (3, 5, 8)
"""The lengths of the prompts a generation recording draws: inside a block, across one, and whole blocks."""


def generated(name: str, workspace: Path) -> None:
    """The release's own `generate` of prompts drawn at `PROMPTS`' lengths (ids past the end token, the mask
    token aside) under `GENERATE`, written as generate.json beside the fixture with the releases it ran.

    Its head's mask-token logit is held at -inf (a forward hook on `lm_head`, which its forward calls): the
    release's `generate` draws over the whole vocabulary, so a position whose best token is the mask token
    keeps it, a mask in its output, where Dew's walk draws no mask token (`DiscreteDenoiser`, MDLM's
    carry-over); with it held out, the two walks draw from one distribution, every block, visibility,
    position and reveal the release's."""
    import transformers

    config = json.loads((FIXTURES / name / 'config.json').read_text())
    model = released_model(name, workspace).float()
    mask = config['mask_token_id']

    def no_mask(module, inputs, logits):
        return logits.index_fill(-1, torch.tensor([mask]), float('-inf'))

    model.lm_head.register_forward_hook(no_mask)
    ids = np.setdiff1d(np.arange(GENERATE["eos_id"] + 1, config['vocab_size']), [config['mask_token_id']])
    rng = np.random.default_rng(1061)
    rows = []
    for length in PROMPTS:
        prompt = rng.choice(ids, length)
        tokens = model.generate(inputs=torch.from_numpy(prompt[None]).long(), temperature=0.0, mask_id=mask,
                                **GENERATE)
        rows.append({"prompt": prompt.tolist(), "generated": tokens[0].tolist()})
    repo, revision = RELEASES[name][:2]
    record = {"reference": f"{repo}@{revision}", "transformers": transformers.__version__,
              "torch": torch.__version__, "command": f"tools/llada_moe_reference.py --generate {name}",
              "settings": GENERATE, "mask_id": mask, "mask_logit": "held at -inf", "rows": rows}
    (FIXTURES / name / 'generate.json').write_text(json.dumps(record, indent=1) + '\n')
    print(name, rows, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+', choices=RELEASES)
    parser.add_argument('--draw', action='store_true', help='redraw the committed synthetic checkpoint')
    parser.add_argument('--serve', action='store_true', help="the release's side of the pipe")
    parser.add_argument('--padding', action='store_true', help='record the rows each lookup leaves untrained')
    parser.add_argument('--generate', action='store_true', help="record the release's blockwise generation")
    parser.add_argument('--release-python', default=sys.executable,
                        help="the Python that runs the release's code, with its transformers")
    args = parser.parse_args()
    if args.serve:
        serve(args.names[0])
        return
    REGISTERED.update(RELEASES)
    if args.generate:
        with tempfile.TemporaryDirectory() as workspace:
            for name in args.names:
                generated(name, Path(workspace))
        return
    if args.padding:
        with tempfile.TemporaryDirectory() as workspace:
            for name in args.names:
                record_padding(*config_lookups(*released_classes(name, Path(workspace))),
                               json.loads((FIXTURES / name / 'config.json').read_text()), FIXTURES / name)
        return
    for name in args.names:
        if args.draw:
            with tempfile.TemporaryDirectory() as workspace, torch.no_grad():
                draw(released_model(name, Path(workspace) / 'draw').float(), FIXTURES / name)
        record(name, args.release_python)


if __name__ == '__main__':
    main()
