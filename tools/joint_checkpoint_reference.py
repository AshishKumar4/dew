"""A joint-head release's own code on fixed requests, for dew.decision's parity tests.

The release's code is its joint_schema_model.py at the repo and revision
tests/fixtures/joint_checkpoint/tiny/source.json records (`model` names its
model class there), read from the Hub, beside transformers 5.16.1. Its backbone here is the tiny
Qwen 3.5 of tests/fixtures/hf/qwen38-dense-tiny (two Gated DeltaNet layers
and a full-attention one at width 32, with Qwen's chat tokens), and its head
a `JointSchemaHead` of width 24 with every tensor scattered. For each request
of tests/fixtures/laya/cases.json this runs what the release's `systemone`
runs, `encode_record`, `collate_records` and its model's `forward`, in fp32
and again in float64, and writes to tests/fixtures/joint_checkpoint/tiny:

- joint_head_config.json and joint_head.safetensors, the head as the
  release stores it;
- layouts.json: each request's token ids, and each question's instruction
  span, option spans and option ids, which Dew's `JointLayout` must
  reproduce exactly;
- states.npz: the backbone's fp32 last hidden states per request, which the
  head-alone references read;
- logits.npz: each question's option logits in the release's option order, from
  the head alone on the fp32 states (fp32 under `<case>/<id>`, float64 under
  `<case>/<id>/head64`) and end to end in float64 (`<case>/<id>/f64`).

A choice whose criteria are a list is given to the release's code as a
mapping of each option to None, which is what the list means and what that
code reads.

With `--images`, it reads that head back and runs the quickstart request
with the backbone fixture's two images (images.npy) as the release's
`systemone` runs one, its processor laying the images out before the state, and writes
images.json (the row's token ids) and images.npz (each question's fp32
logits under `<id>` and float64 ones under `<id>/f64`, the vision tower
included).

    PYTHONPATH=src:tools python tools/joint_checkpoint_reference.py [--images]
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = ROOT / "tests" / "fixtures" / "hf" / "qwen38-dense-tiny"
OUT = ROOT / "tests" / "fixtures" / "joint_checkpoint" / "tiny"
SOURCE = json.loads((OUT / "source.json").read_text())
"""The release the fixtures reproduce: its repo, revision and model class."""
HEAD = {"hidden_size": 32, "width": 24, "routing_layers": 2, "layers": 2, "heads": 2, "feedforward": 40}


def upstream_module():
    """joint_schema_model.py as the release publishes it."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(SOURCE["repo"], "joint_schema_model.py", revision=SOURCE["revision"])
    spec = importlib.util.spec_from_file_location("joint_schema_model", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def as_upstream(request: dict) -> dict:
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

    upstream = upstream_module()
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
    model = getattr(upstream, SOURCE["model"])(backbone, head).eval()
    exact = copy.deepcopy(model).double()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "joint_head_config.json").write_text(json.dumps(HEAD, indent=1) + "\n")
    save_file({name: tensor.contiguous() for name, tensor in head.state_dict().items()},
              OUT / "joint_head.safetensors")

    cases = json.loads((ROOT / "tests" / "fixtures" / "laya" / "cases.json").read_text())
    layouts, states, logits = {}, {}, {}
    table = backbone.get_output_embeddings().weight
    for case, request in cases.items():
        encoded = upstream.encode_record(tokenizer, as_upstream(request))
        layouts[case] = {"ids": list(encoded.input_ids), "questions": [
            {"id": question.question_id, "type": question.question_type,
             "span": list(question.question_span), "options": [list(span) for span in question.option_spans],
             "option_ids": list(question.option_ids)}
            for question in encoded.questions]}
        batch = upstream.collate_records([encoded], tokenizer.pad_token_id, torch.device("cpu"))
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


def images() -> None:
    from diffusers_wan_reference import float64
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    upstream = upstream_module()
    processor = AutoProcessor.from_pretrained(BACKBONE)
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(BACKBONE, dtype=torch.float32,
                                                               attn_implementation="eager").eval()
    head = upstream.JointSchemaHead(**json.loads((OUT / "joint_head_config.json").read_text())).eval()
    head.load_state_dict(load_file(OUT / "joint_head.safetensors"), strict=True)
    model = getattr(upstream, SOURCE["model"])(backbone, head).eval()
    exact = copy.deepcopy(model).double()
    request = json.loads((ROOT / "tests" / "fixtures" / "laya" / "cases.json").read_text())["quickstart"]
    pictures = [Image.fromarray(array) for array in np.load(BACKBONE / "images.npy")]
    encoded = upstream.encode_record(processor.tokenizer, {**as_upstream(request), "images": pictures},
                                 processor=processor)
    batch = upstream.collate_records([encoded], processor.tokenizer.pad_token_id, torch.device("cpu"))
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
