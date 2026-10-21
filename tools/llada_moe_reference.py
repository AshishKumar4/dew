"""The pinned LLaDA MoE releases over synthetic checkpoints and forty token grids.

Uses remote_code_reference's loader, which fetches config and Python only,
never weights. --draw replaces the committed synthetic weights and ids.
LLaDA-MoE, v2 and 2.0 use transformers 4.57.1; 2.1 and 2.2 use 5.16.1.
The float64 arm widens explicit fp32 pins through the shared torch helpers.
LLaDA2's probes carry its release's explicit block-causal mask. Sampling
recipe parity is separate from this forward parity.
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
from diffusers_wan_reference import float64
from remote_code_reference import FIXTURES, RELEASES as REGISTERED, released_model
from safetensors.torch import save_file

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
    ids = torch.randint(0, model.config.vocab_size, (40, 2, 16)).numpy().astype(np.int32)
    np.save(directory / 'probe_ids.npy', ids)
    np.save(directory / 'input_ids.npy', ids[0])


def ensemble(model, ids, name):
    """The released forward on every grid, with LLaDA2's block-causal mask."""
    fields = {}
    if name in ('llada2-tiny', 'llada21-tiny', 'llada22-tiny'):
        blocks = torch.arange(ids.shape[-1]) // 4
        visible = blocks[None, :] <= blocks[:, None]
        fields['attention_mask'] = torch.where(visible, 0.0, -torch.inf)[None, None].expand(
            ids.shape[1], 1, -1, -1)
    return np.stack([model(input_ids=tokens, **fields).logits.detach().numpy() for tokens in ids])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('names', nargs='+', choices=RELEASES)
    parser.add_argument('--draw', action='store_true', help='redraw the committed synthetic checkpoint')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    REGISTERED.update(RELEASES)
    with tempfile.TemporaryDirectory() as workspace, torch.no_grad():
        for name in args.names:
            directory = FIXTURES / name
            model = released_model(name, Path(workspace) / 'fp32').float()
            if args.draw:
                draw(model, directory)
            ids = torch.from_numpy(np.load(directory / 'probe_ids.npy').astype(np.int64))
            native = ensemble(model, ids, name)
            with float64():
                wide = released_model(name, Path(workspace) / 'f64').double()
                truth = ensemble(wide, ids, name)
            if args.check:
                print(name, 'bitwise' if np.array_equal(np.load(directory / 'probe_logits.npy'), native)
                      else 'differs')
            else:
                for file, values in (('probe_logits.npy', native), ('probe_logits_f64.npy', truth),
                                     ('logits.npy', native[0]), ('logits_f64.npy', truth[0])):
                    np.save(directory / file, values)
                print(name, native.shape)


if __name__ == '__main__':
    main()
