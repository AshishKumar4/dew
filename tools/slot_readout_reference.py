"""Record the released renderer, block adapter and readout on the tiny native Qwen source.

Run in the author's environment (`python tools/slot_runtime_qualification.py environment` prints its
Python), then use --calibrate in the project's gate environment. The output belongs under
tests/fixtures for fetched storage.
"""

import argparse
import copy
import dataclasses
import importlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch
from block_override_reference import (
    LICENSE,
    PATCH_SHA256,
    PROMPT_FILES,
    PROMPT_REPO,
    PROMPT_REVISION,
    REFERENCE,
    calibrate,
    oracle_precision,
    release_package,
    require_reference_transformers,
)
from fetched_fixtures import save
from reference_source import git_head

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests/fixtures/hf/qwen38-dense-tiny"
SITES = ("causal_slot_lm", "causal_head", "block_slot_lm", "block_head")
WIDTH = 512


def draw(seed: int, tokenizer, prompt, schema, readout):
    """Unequal choice/score rows, structured and raw UTF-8 states, and asymmetric VJP probes."""
    rng = np.random.default_rng(seed)
    words = ("refund", "broken", "invoice", "urgent", "late", "help", "café", "delta")

    def text(count):
        return " ".join(rng.choice(words, count).tolist())

    requests = [
        {"state": text(int(rng.integers(3, 8))), "questions": {"asked": {
            "type": "choice", "instructions": text(3),
            "criteria": {"left": text(2), "right": {"detail": text(2)}}}}},
        {"state": {"event": text(2), "notes": [text(2), text(1)]}, "questions": {"asked": {
            "type": "score", "instructions": text(3), "criteria": [text(2) for _ in range(4)]}}},
    ]
    tokens = np.full((len(requests), WIDTH), tokenizer.pad_token_id, np.int32)
    spans = np.zeros((len(requests), 4, 2), np.int32)
    options = np.zeros((len(requests), 4), bool)
    lengths, prefixes = [], []
    for index, request in enumerate(requests):
        parsed = schema.SystemOneRequest.model_validate(request)
        question = prompt.render_question("asked", parsed.questions["asked"], style=prompt.DEFAULT_STYLE)
        messages = prompt.build_messages(parsed.state, question.suffix, style=prompt.DEFAULT_STYLE)
        rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=False)
        start = rendered.rindex(question.suffix)
        characters = [(start + begin, start + end) for begin, end in question.option_spans]
        ids, prefix, positions = readout.token_layout(tokenizer, rendered, start, characters)
        if len(ids) > WIDTH:
            raise ValueError(f"the recorded row needs {len(ids)} tokens, over its width {WIDTH}")
        tokens[index, :len(ids)] = ids
        spans[index, :len(positions)] = positions
        options[index, :len(positions)] = True
        lengths.append(len(ids))
        prefixes.append(prefix)
    return ({"tokens": tokens, "lengths": np.asarray(lengths, np.int32), "spans": spans,
             "options": options, "blocks": readout.block_ids(prefixes, lengths, WIDTH, "cpu").numpy(),
             "probe": rng.standard_normal(int(options.sum())).astype(np.float32)}, requests)


def observed(model, head, capture, readout, inputs, labels, *, blocked: bool, dtype) -> dict[str, dict]:
    tokens = torch.from_numpy(inputs["tokens"])
    lengths = torch.from_numpy(inputs["lengths"])
    mask = torch.arange(tokens.shape[1])[None] < lengths[:, None]
    groups = torch.from_numpy(inputs["blocks"]) if blocked else None
    with readout.blocks(groups):
        output = model(input_ids=tokens, attention_mask=mask, use_cache=False)
    hidden = capture.pop()
    if hidden.dtype != dtype or output.logits.dtype != dtype:
        raise RuntimeError("the reference hidden states and vocabulary logits must retain the oracle dtype")
    rows = torch.arange(len(lengths))
    answer = hidden[rows, lengths - 1]
    positions = [spans[valid].tolist()
                 for spans, valid in zip(inputs["spans"], inputs["options"], strict=True)]
    pooled = readout.pool_options(hidden, positions, inputs["options"].shape[1])
    selected = torch.from_numpy(inputs["options"])
    probe = torch.from_numpy(inputs["probe"]).to(dtype)
    result = {}
    for mode in ("slot_lm", "head"):
        local = tuple(head.named_parameters()) if mode == "head" else ()
        scores = (head(answer, pooled) if mode == "head" else
                  output.logits[rows, lengths - 1][:, labels[:selected.shape[1]]])
        scores = scores[selected]
        gradients = torch.autograd.grad(scores, (*[value for _, value in local], hidden),
                                        grad_outputs=probe, retain_graph=mode == "slot_lm")
        values = {"scores": scores, "gradient.hidden": gradients[-1]}
        values.update({f"gradient.{name}": value
                       for (name, _), value in zip(local, gradients[:-1], strict=True)})
        if any(value.dtype != dtype for value in values.values()) or pooled.dtype != dtype:
            raise RuntimeError("the reference readout and every VJP must retain the oracle dtype")
        result[mode] = {name: value.detach().numpy() for name, value in values.items()}
    return result


def main():
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int)
    parser.add_argument("--redraw", type=int)
    parser.add_argument("--calibrate", action="store_true")
    args = parser.parse_args()
    if args.calibrate:
        calibrate(args.output, SITES)
        return
    if args.seeds is None or args.redraw is None or args.seeds <= 0 or args.redraw < 0:
        parser.error("seeds must be positive and redraw nonnegative")
    require_reference_transformers()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(43)
    tokenizer = AutoTokenizer.from_pretrained(SOURCE, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        SOURCE, dtype=torch.float32, attn_implementation="eager").eval()
    with release_package() as (prompt, schema, readout):
        labels = [tokenizer.encode(label, add_special_tokens=False) for label in prompt.SLOT_LABELS[:16]]
        if any(len(ids) != 1 or ids[0] >= model.config.text_config.vocab_size
               or tokenizer.decode(ids) != label
               for ids, label in zip(labels, prompt.SLOT_LABELS[:16], strict=True)):
            raise ValueError("every recorded slot must be one round-tripping token in the tiny vocabulary")
        labels = [ids[0] for ids in labels]
        head = readout.ReadoutHead.from_lm_head(model.get_output_embeddings().weight, labels, 8)
        torch.nn.init.normal_(head.k.weight, std=0.2)
        with oracle_precision():
            truth = AutoModelForImageTextToText.from_config(copy.deepcopy(model.config), dtype=torch.float64,
                                                           attn_implementation="eager").double().eval()
            truth.load_state_dict(model.state_dict())
            exact_head = readout.ReadoutHead(head.slot.shape[1], len(labels), 8).double()
            exact_head.load_state_dict(head.state_dict())
        readout.install_noncausal(model)
        readout.install_noncausal(truth)
        capture, exact_capture = (readout.HiddenCapture(value) for value in (model, truth))
        args.output.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.output)
        tokenizer.save_pretrained(args.output)
        torch.save(head.state_dict(), args.output / "readout_head.pt")
        (args.output / "readout.json").write_text(json.dumps(
            {"readout": "head", "head_dim": 8, "n_slots": len(labels), "attention": "noncausal"}))
        draws, requests, outputs = [], [], {site: {"reference": [], "truth": []} for site in SITES}
        for seed in range(args.seeds * (1 + args.redraw)):
            inputs, request = draw(seed, tokenizer, prompt, schema, readout)
            draws.append(inputs)
            requests.append(request)
            for attention, blocked in (("causal", False), ("block", True)):
                reference = observed(model, head, capture, readout, inputs, labels,
                                     blocked=blocked, dtype=torch.float32)
                with oracle_precision():
                    exact = observed(truth, exact_head, exact_capture, readout, inputs, labels,
                                     blocked=blocked, dtype=torch.float64)
                for mode in reference:
                    outputs[f"{attention}_{mode}"]["reference"].append(reference[mode])
                    outputs[f"{attention}_{mode}"]["truth"].append(exact[mode])
        arrays = {"precision": np.asarray("ieee")}
        for site, paired in outputs.items():
            arrays[f"{site}.redraw"] = np.asarray(args.redraw)
            arrays.update({f"{site}.input.{name}": np.stack([given[name] for given in draws])
                           for name in draws[0]})
            for arm, values in paired.items():
                arrays.update({f"{site}.{arm}.{name}": np.stack([value[name] for value in values])
                               for name in values[0]})
        save(args.output / "reference.npz", arrays, reference=REFERENCE, license=LICENSE)
        (args.output / "requests.json").write_text(json.dumps(requests, ensure_ascii=False) + "\n")
        provenance = {"generator": "tools/slot_readout_reference.py", "generator_revision": git_head(ROOT),
                      "prompt_repo": PROMPT_REPO, "prompt_revision": PROMPT_REVISION,
                      "prompt_files": PROMPT_FILES, "patch_sha256": PATCH_SHA256,
                      "style": dataclasses.asdict(prompt.DEFAULT_STYLE), "thinking": False,
                      "source": SOURCE.relative_to(ROOT).as_posix(), "seeds": args.seeds,
                      "redraw": args.redraw, "head_seed": 43, "precision": "ieee", "device": "cpu",
                      "packages": {name: importlib.metadata.version(name) for name in
                                   ("transformers", "torch", "numpy", "safetensors", "pydantic")}}
        (args.output / "source.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
