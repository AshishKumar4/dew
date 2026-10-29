"""Laya's own code on fixed requests, for dew.decision's parity tests.

Laya is NandhaKishorM/laya at LAYA_COMMIT; install it with

    uv pip install "laya @ git+https://github.com/NandhaKishorM/laya@a4a8921afebfd852bba0000475cfb6ab737a124c"

beside transformers 5.16.1. For each request in CASES and each option layout
(sequential, the published one, and parallel), this runs what `Agent`
runs: `Agent._check_question` and `_to_internal`, `build_sequence` over
the state tokenized once, `collate_items`, and `DecisionModel.forward`, in
fp32 and again in float64, and writes

- layouts.json: each question's token ids and option markers, which Dew's
  layout must reproduce exactly;
- logits.npz, for the released checkpoint: each question's option logits, fp32
  under `<layout>/<case>/<id>` and float64 under `<layout>/<case>/<id>/f64`;
- answers.json: `Agent.system_one` on each request, the published wire
  answers, in fp32 on the CPU;

and the requests themselves to tests/fixtures/laya/cases.json.

Two checkpoints: `tiny`, a random three-layer encoder and two-layer head
with a byte-level BPE vocabulary trained here, written whole under
tests/fixtures/laya/tiny; and with `--real`, the released English
checkpoint at LAYA_REVISION, whose references go to tests/fixtures/laya/real
for the network test. `--record` records instead the tiny checkpoint's
option logits, both layouts, fp32 and float64, over check 1's seeds of
requests whose texts are drawn again (`drawn_cases`), as
tests/fixtures/laya/tiny/recorded_logits.npz, which tests/test_decision_laya.py
holds Dew to.

    PYTHONPATH=src:tools python tools/laya_reference.py [--real | --record]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

LAYA_COMMIT = "a4a8921afebfd852bba0000475cfb6ab737a124c"
LAYA_REPO = "convaiinnovations/laya"
LAYA_REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "laya"

TICKET = ("Hi team, we were billed twice for March and the second charge bounced our rent payment. "
          "I need this reversed today or I am cancelling the account.")
CASES = {
    "quickstart": {"state": TICKET, "questions": {
        "department": {"type": "choice", "instructions": "Which department should handle this?",
                       "criteria": {"billing": "invoices, refunds and charges",
                                    "technical": "bugs and outages", "sales": "pricing and upgrades"}},
        "urgency": {"type": "score", "instructions": "How urgent is this?",
                    "criteria": ["not urgent", "soon", "today", "blocking"]},
        "churn": {"type": "noul", "instructions": "Does the customer threaten to leave?"}}},
    "structured": {"state": {"customer": "Ana", "plan": "pro",
                             "messages": ["refund please", "[MASK] it now"]}, "questions": {
        "refund": {"type": "noul", "instructions": {"question": "Is this a refund request?", "strict": True},
                   "criteria": {"true": "asks for money back", "false": {"note": "anything else"}}},
        "tone": {"type": "choice", "instructions": "What is the [MASK] tone?",
                 "criteria": ["calm", "angry", "neutral"]}}},
    "conversation": {"state": [{"role": "user", "content": f"message {turn}: " + TICKET}
                               for turn in range(6)], "questions": {
        "resolved": {"type": "noul", "instructions": "Has the issue been resolved?",
                     "criteria": {"true": "the agent fixed it", "false": ""}}}},
    "many_options": {"state": TICKET, "questions": {
        "intent": {"type": "choice", "instructions": "Which request is this?",
                   "criteria": {f"intent_{index}": f"request type number {index} about accounts"
                                for index in range(14)}}}},
    "single_option": {"state": "short", "questions": {
        "only": {"type": "choice", "instructions": "Pick it.", "criteria": {"yes": None}}}},
}


def _laya():
    """Laya's modules, which this tool runs as they are published."""
    from laya import agent, common
    return agent, common


def write_tokenizer(directory: Path):
    """A byte-level BPE of 320 tokens with BERT's specials, trained on the cases."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    corpus = [json.dumps(case, ensure_ascii=False) for case in CASES.values()] * 4
    tokenizer.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=320, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(directory / "tokenizer")
    return fast


def write_tiny(directory: Path) -> None:
    """A Laya checkpoint at toy size, in Laya's own layout."""
    from safetensors.torch import save_file
    from transformers import ModernBertConfig, ModernBertModel

    from dew.interop.verify import scatter_weights

    _, common = _laya()
    directory.mkdir(parents=True, exist_ok=True)
    tokenizer = write_tokenizer(directory)
    torch.manual_seed(0)
    encoder = ModernBertModel(ModernBertConfig(
        vocab_size=len(tokenizer), hidden_size=128, intermediate_size=96, num_hidden_layers=3,
        num_attention_heads=2, global_attn_every_n_layers=3, local_attention=8, max_position_embeddings=256,
        pad_token_id=tokenizer.pad_token_id, cls_token_id=tokenizer.cls_token_id,
        sep_token_id=tokenizer.sep_token_id, bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id))
    model = common.DecisionModel(encoder, head_layers=2, n_act=2)
    scatter_weights(model, 1234)
    encoder.config.save_pretrained(directory / "encoder")
    save_file({name: tensor.contiguous() for name, tensor in model.state_dict().items()},
              directory / "model.safetensors")
    (directory / "rl_agent_config.json").write_text(json.dumps({
        "encoder": "tiny", "head_layers": 2, "max_len": 96, "head_max_len": 48, "max_prefixes": 6,
        "act_costs": {"escalate": 0.5}, "cost_wrong_act": 3.0, "amp_dtype": "bf16", "model_name": "rl-agent",
        "temperature": [1.3, 0.9, 1.7],
        "temperature_by_options": {"choice:3-5": 1.2, "choice:11+": 0.3, "noul:2": 2.0}}, indent=2) + "\n")


def runs(directory: Path):
    """Laya's agent over `directory`, as a function of requests and a layout: each question's option
    logits by `<layout>/<case>/<id>` (what `Agent` runs: `_check_question` and `_to_internal`,
    `build_sequence` over the state tokenized once, `collate_items` and `DecisionModel.forward`), in
    fp32, or `wide` by its float64 twin inside `float64`; and each question's layout by the same key.
    Returns the agent, Laya's common module and the function."""
    from torch_precision import float64, float64_twin

    agent_module, common = _laya()
    agent = agent_module.Agent(str(directory), device="cpu")
    tokenizer, config = agent.tok, agent.cfg
    model = agent.model.float().eval()
    # Its float64 twin (`float64_twin`), the decision model built again inside the widening by Laya's
    # own builder, uninitialised for a checkpoint to fill (`common.build_model`), so its inverse
    # frequencies are float64's, the float32 model's weights in it.
    encoder = directory / "encoder"
    exact = float64_twin(model, lambda: common.build_model(
        config, encoder_dir=str(encoder) if encoder.exists() else None, pretrained=False))

    def run(requests: dict, parallel: bool, wide: bool = False) -> tuple[dict, dict]:
        name = "parallel" if parallel else "sequential"
        layouts, logits = {}, {}
        for case, request in requests.items():
            state = request["state"]
            state_ids = tokenizer(common.serialize_state(state).replace(tokenizer.mask_token, " "),
                                  add_special_tokens=False)["input_ids"]
            for qid, question in request["questions"].items():
                agent_module.Agent._check_question(qid, question)
                internal = agent_module.Agent._to_internal(question)
                ids, markers, *layout = common.build_sequence(
                    tokenizer, state, internal, config["max_len"], config["head_max_len"],
                    truncate_left=isinstance(state, list), state_ids=state_ids, return_layout=parallel)
                item = {"ids": ids, "markers": markers, "qtype": common.QTYPES[internal["t"]]}
                if parallel:
                    item["layout"] = layout[0]
                key = f"{name}/{case}/{qid}"
                layouts[key] = {"ids": ids, "markers": markers,
                                **({"positions": layout[0]["position_ids"], "slots": layout[0]["option_ids"]}
                                   if parallel else {})}
                batch = common.collate_items([[item]], tokenizer.pad_token_id)
                arguments = (batch["input_ids"], batch["attention_mask"], batch["marker_pos"],
                             batch["marker_mask"], batch["qtype"])
                extra = ({"position_ids": batch["position_ids"], "option_ids": batch["option_ids"]}
                         if parallel else {})
                if wide:
                    with float64(), torch.no_grad():
                        logits[key] = exact(*arguments, **extra)[0][0].double().numpy()
                else:
                    with torch.no_grad():
                        logits[key] = model(*arguments, **extra)[0][0].numpy()
        return logits, layouts

    return agent, run


def references(directory: Path, out: Path, *, logits: bool = True) -> None:
    """Laya's layouts, logits in fp32 and float64 unless not `logits`, and agent answers for
    `directory`."""
    agent, run = runs(directory)
    layouts, found = {}, {}
    for parallel in (False, True):
        fp32, laid = run(CASES, parallel)
        wide, _ = run(CASES, parallel, wide=True)
        layouts.update(laid)
        found.update({**fp32, **{f"{key}/f64": value for key, value in wide.items()}})
    answers = {case: agent.system_one(request["state"], request["questions"])
               for case, request in CASES.items()}
    out.mkdir(parents=True, exist_ok=True)
    (out / "layouts.json").write_text(json.dumps(layouts) + "\n")
    if logits:
        np.savez(out / "logits.npz", **found)
    (out / "answers.json").write_text(json.dumps(answers, indent=1) + "\n")


WORDS = sorted({word for case in CASES.values() for word in json.dumps(case).replace("\\", " ").split()
                if word.isalpha()})
"""The words a drawn request's text is made of: those the committed cases hold, which the tiny
tokenizer was trained on."""


def drawn_cases(seed: int) -> dict[str, np.ndarray]:
    """Check 1's draw of the requests: `CASES` with every free text (a state's string, a message's
    content, an instruction's question, an option's description) drawn again from `WORDS` at its own
    word count, a bracketed marker kept where it stands; their structure (the question types, the
    option labels, the conversation's turns, a None) kept, so each case lays out as it does, cut where
    it is. As JSON's UTF-8 bytes, under `cases` (an array a program can take beside the draw's other
    inputs, `requests`), the same on every run."""
    generator = np.random.default_rng([0x1A7A, seed])

    def text(value: str) -> str:
        return " ".join(word if word.startswith("[") else str(generator.choice(WORDS))
                        for word in value.split())

    def drawn(value, key=None):
        if isinstance(value, str):
            return value if key in ("type", "role") else text(value)
        if isinstance(value, dict):
            return {name: drawn(item, name) for name, item in value.items()}
        if isinstance(value, list):
            return [drawn(item) for item in value]
        return value

    return {"cases": np.frombuffer(json.dumps(drawn(CASES)).encode(), np.uint8)}


def requests(inputs) -> dict:
    """The requests a draw (`drawn_cases`) holds."""
    return json.loads(np.asarray(inputs["cases"], np.uint8).tobytes().decode())


def record(directory: Path) -> Path:
    """Laya's option logits over check 1's seeds of requests (`drawn_cases`, `record_case`), both
    layouts, fp32 and float64, its count sized on Laya alone, as recorded_logits.npz in `directory`."""
    import transformers
    from recorded_references import provenance, record_case, save

    _, run = runs(directory)

    def evaluate(inputs):
        drawn = requests(inputs)
        fp32 = {key: value for parallel in (False, True) for key, value in run(drawn, parallel)[0].items()}
        wide = {key: value for parallel in (False, True)
                for key, value in run(drawn, parallel, wide=True)[0].items()}
        return fp32, wide

    recorded = record_case("laya", drawn_cases, evaluate, store_inputs=False)
    made = provenance("NandhaKishorM/laya", LAYA_COMMIT, "cpu",
                      "PYTHONPATH=src:tools python tools/laya_reference.py --record",
                      transformers=transformers.__version__, torch=torch.__version__)
    return save(directory / "recorded_logits.npz", {**recorded, "laya.provenance": made},
                reference="NandhaKishorM/laya", license="Apache-2.0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real", action="store_true", help="also the released English checkpoint")
    parser.add_argument("--record", action="store_true", help="only the tiny checkpoint's logits over seeds")
    args = parser.parse_args()
    if args.record:
        print(record(FIXTURES / "tiny"))
        return
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "cases.json").write_text(json.dumps(CASES, indent=1, ensure_ascii=False) + "\n")
    tiny = FIXTURES / "tiny"
    write_tiny(tiny)
    references(tiny, tiny, logits=False)
    if args.real:
        from huggingface_hub import snapshot_download

        directory = Path(snapshot_download(LAYA_REPO, revision=LAYA_REVISION, allow_patterns=[
            "model.safetensors", "encoder/*", "tokenizer/*", "rl_agent_config.json"]))
        references(directory, FIXTURES / "real")
        (FIXTURES / "real" / "source.json").write_text(
            json.dumps({"repo": LAYA_REPO, "revision": LAYA_REVISION}) + "\n")
    print(sorted(path.name for path in FIXTURES.rglob("*") if path.is_file()), file=sys.stderr)


if __name__ == "__main__":
    main()
