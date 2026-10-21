"""The pinned LLaDA MoE releases over synthetic checkpoints, recorded over seeds for check 1.

Each family's site (its fixture directory's name) draws two rows of 16 token ids a seed
(tests/model_support.py's `llada_draw`, which its test draws again) and records the release's fp32
forward's evidence against the release run in float64 (`record_case(..., summary=True)`): kilobytes,
committed as `released.npz` beside the fixture. Dew's own fp32 forward plans the seed count, and the
recorded release confirms it. LLaDA2's forwards run under its explicit block-causal mask of 4-token
blocks (`llada_fields`), which its 2.0 forward requires. Sampling-recipe parity is separate.

Uses remote_code_reference's loader, which fetches config and Python only, never weights; --draw
replaces the committed synthetic weights first. The releases' code reads different transformers:
LLaDA-MoE, v2 and 2.0 run under 4.57.1, 2.1 and 2.2 under 5.16.1, each in an environment holding Dew
too, as record_case runs Dew's forward beside the release's:
  python tools/llada_moe_reference.py llada-moe-tiny llada-moe-v2-tiny llada2-tiny   # 4.57.1
  python tools/llada_moe_reference.py llada21-tiny llada22-tiny                      # 5.16.1
"""

import argparse
import sys
import tempfile
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import torch
import transformers
from diffusers_wan_reference import float64
from recorded_references import provenance, record_case, save
from remote_code_reference import FIXTURES, RELEASES as REGISTERED, released_model
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from model_support import llada_draw, llada_fields
from reference_error import widened

from dew.interop import Pretrained

RELEASES = {
    "llada-moe-tiny": ("inclusionAI/LLaDA-MoE-7B-A1B-Base", "daccaf1ccbf263a427bc9ac2a23b0b004fb275bf",
                       ("configuration_lladamoe.py", "modeling_lladamoe.py"), "AutoModelForCausalLM", True),
    "llada-moe-v2-tiny": ("GSAI-ML/LLaDA-MoE-v2-30B-A3B-Base", "2a513f7a4634e426b2ab030e6defb24001d0c847",
                          ("configuration_lladamoe_v2.py", "modeling_lladamoe_v2.py"),
                          "AutoModelForCausalLM", True),
    "llada2-tiny": ("inclusionAI/LLaDA2.0-mini", "dad945cac317da394b390f82c7b40691d8a881ed",
                    ("configuration_llada2_moe.py", "modeling_llada2_moe.py"), "AutoModelForCausalLM", True),
    "llada21-tiny": ("inclusionAI/LLaDA2.1-mini", "20e64e2ad21644d0e5248586ed9c942cdd45de0f",
                     ("configuration_llada2_moe.py", "modeling_llada2_moe.py"), "AutoModelForCausalLM", True),
    "llada22-tiny": ("inclusionAI/LLaDA2.2-flash", "9b78da3f9149f1e239fb6d15f20a326fde574b9f",
                     ("configuration_llada2_moe.py", "modeling_llada2_moe.py"), "AutoModelForCausalLM", True),
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


def released_logits(model, ids, name):
    """The release's forward of `ids` `[2, 16]`, under LLaDA2's block-causal mask."""
    fields = {}
    mask = llada_fields(name, ids.shape[-1])
    if mask:
        visible = torch.from_numpy(np.asarray(mask['attention_pairwise_mask']))
        fields['attention_mask'] = torch.where(visible, 0.0, -torch.inf).to(model.dtype)[None, None].expand(
            ids.shape[0], 1, -1, -1)
    return model(input_ids=ids, **fields).logits.detach().numpy()


def record(name: str, workspace: Path) -> None:
    repo, revision = RELEASES[name][:2]
    native = released_model(name, workspace / 'fp32').float()
    with float64():
        wide = released_model(name, workspace / 'f64').double()

    def evaluate(inputs):
        ids = torch.from_numpy(inputs['ids'].astype(np.int64))
        reference = {'logits': released_logits(native, ids, name)}
        with float64():
            truth = {'logits': released_logits(wide, ids, name)}
        return reference, truth

    source = Pretrained.load(FIXTURES / name, dtype='float32', attention_impl='reference')
    fields = llada_fields(name, 16)

    def forward(inputs):
        return {'logits': source.model.apply(source.variables, inputs['ids'], **fields)}

    wide_model, wide_variables = source.model.clone(dtype=jnp.float64), widened(source.variables)

    def wide_forward(inputs):
        return {'logits': wide_model.apply(wide_variables, inputs['ids'], **fields)}

    arrays = record_case(name, lambda seed: llada_draw(seed, source.model.vocab_size), evaluate, forward,
                         truth_proxy=wide_forward, store_inputs=False, summary=True)
    arrays[f'{name}.provenance'] = provenance(
        repo, revision, 'cpu', f'python tools/llada_moe_reference.py {name}',
        transformers=transformers.__version__, torch=torch.__version__)
    print(save(FIXTURES / name / 'released.npz', arrays, reference=repo, license=LICENSES[name]), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+', choices=RELEASES)
    parser.add_argument('--draw', action='store_true', help='redraw the committed synthetic checkpoint')
    args = parser.parse_args()
    REGISTERED.update(RELEASES)
    with tempfile.TemporaryDirectory() as workspace, torch.no_grad():
        for name in args.names:
            if args.draw:
                draw(released_model(name, Path(workspace) / 'draw').float(), FIXTURES / name)
            record(name, Path(workspace) / name)


if __name__ == '__main__':
    main()
