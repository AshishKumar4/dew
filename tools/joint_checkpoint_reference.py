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

With `--fla`, on a CUDA GPU with fla installed, it reads that head back and
runs each request through the same release with HF's Qwen 3.5 Gated DeltaNet
on fla's own chunked rule, and writes each question's fp32 logits to
logits_fla.npz under `<case>/<id>`, with `precision` its matmuls': the
reference Dew's model is held to where its delta rule runs on its port, fla's
algorithm (tests/test_decision_joint_checkpoint.py). Every product is IEEE
fp32, as tests/reference_error.py's `ieee_fixture` requires: run it with
TRITON_F32_DEFAULT=ieee for the dots that leave their precision unset, and it
sets the one constant that fixes fla-core 0.5.2's intra-chunk solve at TF32 on
a GPU that has it (fla/ops/gated_delta_rule/chunk_fwd.py:20-23) to 'ieee',
the value fla takes on a GPU without TF32. Triton keys its disk cache on a
kernel's source and not on the globals it reads, so these kernels compile into
a cache directory of their own. Its environment is pinned: torch 2.8.0 for
CUDA 12.8, transformers 5.16.1 and fla-core 0.5.2, which `on_fla` checks
before it runs.

With `--images`, it reads that head back and runs the quickstart request
with the backbone fixture's two images (images.npy) as the release's
`systemone` runs one, its processor laying the images out before the state, and writes
images.json (the row's token ids) and images.npz (each question's fp32
logits under `<id>` and float64 ones under `<id>/f64`, the vision tower
included).

    PYTHONPATH=src:tools python tools/joint_checkpoint_reference.py [--fla | --images]
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
BACKBONE = ROOT / "tests" / "fixtures" / "hf" / "qwen38-dense-tiny"
OUT = ROOT / "tests" / "fixtures" / "joint_checkpoint" / "tiny"
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
    from torch_precision import float64
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


def on_fla() -> None:
    import inspect

    from reference_releases import require
    from safetensors.torch import load_file
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5 import modeling_qwen3_5

    require("fla")
    if os.environ.get("TRITON_F32_DEFAULT") != "ieee":
        raise SystemExit("run with TRITON_F32_DEFAULT=ieee so fla's kernels keep fp32")
    if os.environ.get("FLA_TRIL_PRECISION", "ieee") != "ieee":
        raise SystemExit("leave FLA_TRIL_PRECISION at its 'ieee' default")
    os.environ["TRITON_CACHE_DIR"] = str(Path.home() / ".cache" / "dew" / "triton-ieee")
    from fla.ops.gated_delta_rule import chunk_fwd, chunk_gated_delta_rule
    from triton.language import constexpr

    chunk_fwd.SOLVE_TRIL_DOT_PRECISION = constexpr("ieee")

    # HF's Qwen 3.5 takes fla's rule only where its kernel lookup resolves it, which it did not in this
    # environment; its Gated DeltaNet calls the rule by this module name, so it is fla's here, on the
    # arguments fla's signature takes.
    accepted, calls = inspect.signature(chunk_gated_delta_rule).parameters, []

    def on_fla_rule(*args, **kwargs):
        calls.append(None)
        taken = {name: value for name, value in kwargs.items() if name in accepted}
        return chunk_gated_delta_rule(*args, **taken)

    modeling_qwen3_5.torch_chunk_gated_delta_rule = on_fla_rule
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    upstream, _ = upstream_module()
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(BACKBONE, dtype=torch.float32,
                                                               attn_implementation="eager").to(device).eval()
    head = upstream.JointSchemaHead(**json.loads((OUT / "joint_head_config.json").read_text())).eval()
    head.load_state_dict(load_file(OUT / "joint_head.safetensors"), strict=True)
    head = head.to(device)
    cases = json.loads((ROOT / "tests" / "fixtures" / "laya" / "cases.json").read_text())
    table, text, logits = backbone.get_output_embeddings().weight, backbone.model.language_model, {}
    for case, request in cases.items():
        encoded = upstream.encode_record(tokenizer, as_upstream(request))
        batch = upstream.collate_records([encoded], tokenizer.pad_token_id, device)
        with torch.no_grad():
            hidden = text(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                          use_cache=False).last_hidden_state
            found = head(hidden, batch["input_ids"], batch["attention_mask"], batch["records"], table)[0]
        for question, fp32 in zip(encoded.questions, found, strict=True):
            logits[f"{case}/{question.question_id}"] = fp32.cpu().numpy()
    assert calls, "no layer ran fla's rule"
    np.savez(OUT / "logits_fla.npz", **logits, precision=fla_precision())


def fla_precision() -> str:
    """What logits_fla.npz records of its run's matmuls (`recorded_matmuls`), fla's two triangular
    solves among them."""
    from fla.ops.gated_delta_rule import chunk_fwd
    from torch_precision import recorded_matmuls

    return recorded_matmuls(FLA_TRIL_PRECISION=os.environ.get("FLA_TRIL_PRECISION", "ieee"),
                            **{"the intra-chunk solve": str(getattr(chunk_fwd.SOLVE_TRIL_DOT_PRECISION, "value",
                                                                    chunk_fwd.SOLVE_TRIL_DOT_PRECISION))})


def images() -> None:
    from torch_precision import float64
    from PIL import Image
    from safetensors.torch import load_file
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    upstream, model_class = upstream_module()
    processor = AutoProcessor.from_pretrained(BACKBONE)
    backbone = Qwen3_5ForConditionalGeneration.from_pretrained(BACKBONE, dtype=torch.float32,
                                                               attn_implementation="eager").eval()
    head_config = json.loads((OUT / "joint_head_config.json").read_text())
    head = upstream.JointSchemaHead(**head_config).eval()
    head.load_state_dict(load_file(OUT / "joint_head.safetensors"), strict=True)
    model = model_class(backbone, head).eval()
    exact = joint_twin(model_class, upstream, backbone, head, head_config)
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
    {"--images": images, "--fla": on_fla}.get(next(iter(sys.argv[1:]), ""), main)()
