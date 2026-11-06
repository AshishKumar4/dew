"""A joint-head release's own code on fixed requests, for dew.decision's parity tests.

The release's code is its joint_schema_model.py at the repo and revision
tests/fixtures/joint_checkpoint/tiny/source.json records (`model` names its
model class there), read from the Hub, beside transformers 5.16.1. Its backbone here is the tiny
Qwen 3.5 of tests/fixtures/hf/qwen38-dense-tiny (two Gated DeltaNet layers
and a full-attention one at width 32, with Qwen's chat tokens), and its head
a `JointSchemaHead` of width 24 with every tensor scattered. For each request
of tests/fixtures/laya/cases.json this runs what the release's `systemone`
runs, `encode_record`, `collate_records` and its model's `forward`, in fp32,
and writes to tests/fixtures/joint_checkpoint/tiny:

- joint_head_config.json and joint_head.safetensors, the head as the
  release stores it;
- layouts.json: each request's token ids, and each question's instruction
  span, option spans and option ids, which Dew's `JointLayout` must
  reproduce exactly;
- states.npz: the backbone's fp32 last hidden states per request, which the
  head-alone arms read;
- logits.npz: each question's fp32 option logits in the release's option order, under `<case>/<id>`.

A choice whose criteria are a list is given to the release's code as a
mapping of each option to None, which is what the list means and what that
code reads.

With `--images`, it reads that head back and lays out the quickstart request
with the backbone fixture's two images (images.npy) as the release's
`systemone` lays one out, its processor laying the images out before the state, and writes
images.json (the row's token ids).

The release over drawn head weights, in float32 and float64, is tools/joint_worker.py's.

    PYTHONPATH=src:tools python tools/joint_checkpoint_reference.py [--images]
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = ROOT / "tests" / "fixtures" / "hf" / "qwen38-dense-tiny"
OUT = ROOT / "tests" / "fixtures" / "joint_checkpoint" / "tiny"
CPU = torch.device("cpu")
HEAD = {"hidden_size": 32, "width": 24, "routing_layers": 2, "layers": 2, "heads": 2, "feedforward": 40}


def upstream_module():
    """joint_schema_model.py as the release the fixtures reproduce publishes it
    (source.json: its repo and revision), and its model class (`model`)."""
    from huggingface_hub import hf_hub_download
    from reference_source import loaded

    source = json.loads((OUT / "source.json").read_text())
    module = loaded("joint_schema_model",
                    hf_hub_download(source["repo"], "joint_schema_model.py", revision=source["revision"]))
    return module, getattr(module, source["model"])


def as_upstream(request: dict) -> dict:
    questions = {name: ({**question, "criteria": dict.fromkeys(question["criteria"])}
                        if question["type"] == "choice" and isinstance(question["criteria"], list)
                        else question)
                 for name, question in request["questions"].items()}
    return {**request, "questions": questions}


def joint_twin(model_class, upstream, backbone, head, head_config: dict):
    """The joint model's float64 truth: the backbone's and the head's each their float64 twin
    (`torch_precision.float64_twin`), the backbone read from its fixture at float64 and the head built
    from its config, each holding the float32 model's weights. Run it inside `float64` too."""
    from torch_precision import float64_twin
    from transformers import Qwen3_5ForConditionalGeneration

    wide_backbone = float64_twin(backbone, lambda: Qwen3_5ForConditionalGeneration.from_pretrained(
        BACKBONE, dtype=torch.float64, attn_implementation="eager"))
    wide_head = float64_twin(head, lambda: upstream.JointSchemaHead(**head_config))
    return model_class(wide_backbone, wide_head).eval()


def main() -> None:
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    from dew.interop.verify import scatter_weights

    upstream, model_class = upstream_module()
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(BACKBONE, dtype=torch.float32,
                                                               attn_implementation="eager").eval()
    torch.manual_seed(0)
    head = upstream.JointSchemaHead(**HEAD).eval()
    scatter_weights(head, 1234)
    with torch.no_grad():
        head.prior_logit_scale.fill_(0.7)
        head.joint_logit_scale.fill_(1.3)
        head.residual_gate.fill_(-0.4)
    exact = joint_twin(model_class, upstream, backbone, head, HEAD)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "joint_head_config.json").write_text(json.dumps(HEAD, indent=1) + "\n")
    save_file({name: tensor.contiguous() for name, tensor in head.state_dict().items()},
              OUT / "joint_head.safetensors")

    cases = json.loads((ROOT / "tests" / "fixtures" / "laya" / "cases.json").read_text())
    layouts, states, logits = case_logits(upstream, tokenizer, backbone, head, exact, cases)
    (OUT / "layouts.json").write_text(json.dumps(layouts) + "\n")
    np.savez(OUT / "states.npz", **states)
    np.savez(OUT / "logits.npz", **{key: value for key, value in logits.items() if key.count("/") == 1})


def case_logits(upstream, tokenizer, backbone, head, exact, cases: dict, *, states: dict | None = None,
                device: torch.device = CPU) -> tuple[dict, dict, dict]:
    """Each request's layout, its backbone's fp32 last hidden states, and each question's option logits
    keyed `<case>/<id>`: end to end in fp32, and from `exact`, the joint model's float64 twin, end to end
    (`/f64`); and from `head` alone on `states[case]`, the backbone's own fp32 states where None, in fp32
    (`/head`) and from its float64 twin on the same states (`/head64`); every model on `device`."""
    from torch_precision import float64

    layouts, hidden_states, logits = {}, {}, {}
    table = backbone.get_output_embeddings().weight
    text = backbone.model.language_model
    for case, request in cases.items():
        encoded = upstream.encode_record(tokenizer, as_upstream(request))
        layouts[case] = {"ids": list(encoded.input_ids), "questions": [
            {"id": question.question_id, "type": question.question_type,
             "span": list(question.question_span), "options": [list(span) for span in question.option_spans],
             "option_ids": list(question.option_ids)}
            for question in encoded.questions]}
        batch = upstream.collate_records([encoded], tokenizer.pad_token_id, device)
        read = (batch["input_ids"], batch["attention_mask"], batch["records"])
        with torch.no_grad():
            hidden = text(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                          use_cache=False).last_hidden_state
            given = hidden if states is None else torch.from_numpy(states[case])[None].to(device)
            whole, alone = head(hidden, *read, table)[0], head(given, *read, table)[0]
        with float64(), torch.no_grad():
            exact_table = exact.language_model.get_output_embeddings().weight
            whole64, alone64 = exact(batch)[0], exact.head(given.double(), *read, exact_table)[0]
        hidden_states[case] = hidden[0].cpu().numpy()
        for question, *found in zip(encoded.questions, whole, whole64, alone, alone64, strict=True):
            for suffix, value in zip(("", "/f64", "/head", "/head64"), found, strict=True):
                logits[f"{case}/{question.question_id}{suffix}"] = value.cpu().numpy()
    return layouts, hidden_states, logits



def images() -> None:
    from PIL import Image
    from transformers import AutoProcessor

    upstream, _ = upstream_module()
    processor = AutoProcessor.from_pretrained(BACKBONE)
    request = json.loads((ROOT / "tests" / "fixtures" / "laya" / "cases.json").read_text())["quickstart"]
    pictures = [Image.fromarray(array) for array in np.load(BACKBONE / "images.npy")]
    encoded = upstream.encode_record(processor.tokenizer, {**as_upstream(request), "images": pictures},
                                     processor=processor)
    (OUT / "images.json").write_text(json.dumps({"ids": list(encoded.input_ids)}) + "\n")


if __name__ == "__main__":
    {"--images": images}.get(next(iter(sys.argv[1:]), ""), main)()
