"""Clef's own code on fixed requests, for dew.decision's parity tests.

Clef's code is joint_schema_model.py at Cloudflare/clef CLEF_REVISION, read
from the Hub, beside transformers 5.16.1. Its backbone here is the tiny
Qwen 3.5 of tests/fixtures/hf/qwen38-dense-tiny (two Gated DeltaNet layers
and a full-attention one at width 32, with Qwen's chat tokens), and its head
a `JointSchemaHead` of width 24 with every tensor scattered. For each request
of tests/fixtures/laya/cases.json this runs what Clef's `systemone` runs,
`encode_record`, `collate_records` and `ClefModel.forward`, in fp32 and
again in float64, and writes to tests/fixtures/clef/tiny:

- joint_head_config.json and joint_head.safetensors, the head as a Clef
  release stores it;
- layouts.json: each request's token ids, and each question's instruction
  span, option spans and option ids, which Dew's `JointLayout` must
  reproduce exactly;
- states.npz: the backbone's fp32 last hidden states per request, which the
  head-alone references read;
- logits.npz: each question's option logits in Clef's option order, from
  the head alone on the fp32 states (fp32 under `<case>/<id>`, float64 under
  `<case>/<id>/head64`) and end to end in float64 (`<case>/<id>/f64`).

A choice whose criteria are a list is given to Clef as a mapping of each
option to None, which is what the list means and what Clef's code reads.

With `--images`, it reads that head back and runs the quickstart request
with the backbone fixture's two images (images.npy) as Clef's `systemone`
runs one, its processor laying the images out before the state, and writes
images.json (the row's token ids) and images.npz (each question's fp32
logits under `<id>` and float64 ones under `<id>/f64`, the vision tower
included).

    PYTHONPATH=src:tools python tools/clef_reference.py [--images]
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

CLEF_REPO = "Cloudflare/clef"
CLEF_REVISION = "2f3de3dd85f379784083b0814d997ab627200f0c"
ROOT = Path(__file__).resolve().parents[1]
BACKBONE = ROOT / "tests" / "fixtures" / "hf" / "qwen38-dense-tiny"
OUT = ROOT / "tests" / "fixtures" / "clef" / "tiny"
HEAD = {"hidden_size": 32, "width": 24, "routing_layers": 2, "layers": 2, "heads": 2, "feedforward": 40}


def clef_module():
    """joint_schema_model.py as Clef publishes it."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(CLEF_REPO, "joint_schema_model.py", revision=CLEF_REVISION)
    spec = importlib.util.spec_from_file_location("joint_schema_model", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def as_clef(request: dict) -> dict:
    questions = {name: ({**question, "criteria": dict.fromkeys(question["criteria"])}
                        if question["type"] == "choice" and isinstance(question["criteria"], list)
                        else question)
                 for name, question in request["questions"].items()}
    return {**request, "questions": questions}


def main() -> None:
    from diffusers_wan_reference import float64
    from safetensors.torch import save_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    from dew.interop.verify import scatter_weights

    clef = clef_module()
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(BACKBONE, dtype=torch.float32,
                                                               attn_implementation="eager").eval()
    torch.manual_seed(0)
    head = clef.JointSchemaHead(**HEAD).eval()
    scatter_weights(head, 1234)
    with torch.no_grad():
        head.prior_logit_scale.fill_(0.7)
        head.joint_logit_scale.fill_(1.3)
        head.residual_gate.fill_(-0.4)
    model = clef.ClefModel(backbone, head).eval()
    exact = copy.deepcopy(model).double()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "joint_head_config.json").write_text(json.dumps(HEAD, indent=1) + "\n")
    save_file({name: tensor.contiguous() for name, tensor in head.state_dict().items()},
              OUT / "joint_head.safetensors")

    cases = json.loads((ROOT / "tests" / "fixtures" / "laya" / "cases.json").read_text())
    layouts, states, logits = {}, {}, {}
    table = backbone.get_output_embeddings().weight
    for case, request in cases.items():
        encoded = clef.encode_record(tokenizer, as_clef(request))
        layouts[case] = {"ids": list(encoded.input_ids), "questions": [
            {"id": question.question_id, "type": question.question_type,
             "span": list(question.question_span), "options": [list(span) for span in question.option_spans],
             "option_ids": list(question.option_ids)}
            for question in encoded.questions]}
        batch = clef.collate_records([encoded], tokenizer.pad_token_id, torch.device("cpu"))
        text = backbone.model.language_model
        with torch.no_grad():
            hidden = text(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                          use_cache=False).last_hidden_state
            head_logits = head(hidden, batch["input_ids"], batch["attention_mask"], batch["records"],
                               table)[0]
        with float64(), torch.no_grad():
            exact_table = exact.language_model.get_output_embeddings().weight
            head64 = exact.head(hidden.double(), batch["input_ids"], batch["attention_mask"],
                                batch["records"], exact_table)[0]
            whole64 = exact(batch)[0]
        states[case] = hidden[0].numpy()
        answers = zip(encoded.questions, head_logits, head64, whole64, strict=True)
        for question, fp32, on_fp32, whole in answers:
            logits[f"{case}/{question.question_id}"] = fp32.numpy()
            logits[f"{case}/{question.question_id}/head64"] = on_fp32.double().numpy()
            logits[f"{case}/{question.question_id}/f64"] = whole.double().numpy()
    (OUT / "layouts.json").write_text(json.dumps(layouts) + "\n")
    np.savez(OUT / "states.npz", **states)
    np.savez(OUT / "logits.npz", **logits)
    (OUT / "source.json").write_text(json.dumps({"repo": CLEF_REPO, "revision": CLEF_REVISION}) + "\n")


def images() -> None:
    from diffusers_wan_reference import float64
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    clef = clef_module()
    processor = AutoProcessor.from_pretrained(BACKBONE)
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(BACKBONE, dtype=torch.float32,
                                                               attn_implementation="eager").eval()
    head = clef.JointSchemaHead(**json.loads((OUT / "joint_head_config.json").read_text())).eval()
    head.load_state_dict(load_file(OUT / "joint_head.safetensors"), strict=True)
    model = clef.ClefModel(backbone, head).eval()
    exact = copy.deepcopy(model).double()
    request = json.loads((ROOT / "tests" / "fixtures" / "laya" / "cases.json").read_text())["quickstart"]
    pictures = [Image.fromarray(array) for array in np.load(BACKBONE / "images.npy")]
    encoded = clef.encode_record(processor.tokenizer, {**as_clef(request), "images": pictures},
                                 processor=processor)
    batch = clef.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cpu"))
    with torch.no_grad():
        found = model(batch)[0]
    exact_batch = {**batch, "media": {key: value.double() if value.is_floating_point() else value
                                      for key, value in batch["media"].items()}}
    with float64(), torch.no_grad():
        whole64 = exact(exact_batch)[0]
    logits = {}
    for question, fp32, whole in zip(encoded.questions, found, whole64, strict=True):
        logits[question.question_id] = fp32.numpy()
        logits[f"{question.question_id}/f64"] = whole.double().numpy()
    (OUT / "images.json").write_text(json.dumps({"ids": list(encoded.input_ids)}) + "\n")
    np.savez(OUT / "images.npz", **logits)


if __name__ == "__main__":
    images() if "--images" in sys.argv[1:] else main()
