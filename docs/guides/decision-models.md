# Decision models

A decision model answers typed questions about a state in one forward pass. It generates no text: each question names its options, the model scores every option at once, and the answer is a probability distribution over exactly those options. `dew.decision` covers the three question types of TypeSafe's Jev API, and it loads [Laya](https://huggingface.co/convaiinnovations/laya), an open decision model built on ModernBERT-large, as a native Dew model.

| Question | Options | Answer |
|---|---|---|
| `Noul(instructions)` | false, true | `noul`, the probability of yes |
| `Choice(instructions, criteria)` | 1 to 255 named options, each with an optional description | `choice`, the most likely option |
| `Score(instructions, criteria)` | 2 to 10 ordered levels, lowest first | `score`, the expected level |

## Answer questions

```python
from dew.decision import Choice, Decide, Noul, Score

decide = Decide.from_pretrained("convaiinnovations/laya")
answers = decide("Hi, we were billed twice for March and I want it reversed today.", {
    "department": Choice("Which department should handle this?",
                         criteria={"billing": "invoices, refunds", "technical": "bugs, outages"}),
    "urgency": Score("How urgent is this?", criteria=["not urgent", "soon", "blocking"]),
    "churn_risk": Noul("Does the user threaten to leave?"),
})
answers["department"].choice        # "billing"
answers["urgency"].score            # the expected level, between 0 and 2
answers["churn_risk"].noul          # P(yes)
answers["department"].probabilities # one probability per option, in the question's order
```

Each question and the state form one row. Laya's layout puts the question first, then a marker token before each option, then the state, and the model reads each option's score at its marker. To answer many requests together, use `decide.batch(...)`, which packs question rows, shortest first, into passes under a token budget. A choice with more options than one row holds comfortably, such as a 77-way intent question, can be decided in rounds with `decide.tournament(state, questions)`: the options are split into groups of 16, and each group's winner goes to a final.

`systemone` takes and returns Jev's wire format, so a client written for Jev talks to a `Decide` task unchanged:

```python
decide.systemone({
    "state": "I was charged twice for March, fix it today.",
    "questions": {
        "team": {"type": "choice", "instructions": "Which team handles this?",
                 "criteria": {"billing": "payments", "technical": "outages"}},
        "urgent": {"type": "noul", "instructions": "Is this urgent?"},
    },
})
# {"model": ..., "answers": {"team": {"type": "choice", "choice": "billing",
#   "probabilities": {"billing": 0.97, "technical": 0.03}, "confidence": 0.94},
#   "urgent": {"type": "noul", "noul": 0.91}}, "usage": {"input_tokens": ..., "output_tokens": 0}}
```

The response has exactly Jev's fields, with numbers rounded to four places. `details=True` adds what Laya's server adds: a noul's confidence, whether a gated answer abstained, and how much of the state the rows kept. A request may also carry `images`, as Clef's does, PIL images or encoded images, bare or as `data:` URLs, which a task whose backbone reads images lays out with the state, as a Clef release does (`decide(state, questions, images=[...])` takes them too); `strict=True` answers exactly as Jev's endpoint does, dropping such extension fields.

Dew ships no HTTP server of its own, but `systemone` is all one needs. `examples/serve_decisions.py` serves it with Starlette in about forty lines, at Jev's `POST /v1/systemone`, with Jev's 422 for a request it refuses and an optional bearer key:

```bash
uv pip install "dewml[serve] @ git+https://github.com/AshishKumar4/dew"
python examples/serve_decisions.py --model convaiinnovations/laya --port 8000
```

llama.cpp (release b11445) serves decision models natively at the same `POST /v1/systemone`, with images as `data:` URLs: its converter reads Laya's checkpoint layout (ModernBERT with Laya's head) and Clef's (a Qwen 3.5 backbone with `joint_head.safetensors`). `decide.save_pretrained(directory)` writes a ModernBERT task in Laya's layout, temperatures included, which `Decide.from_pretrained` reads back and llama.cpp converts:

```bash
python convert_hf_to_gguf.py directory --outfile laya.gguf
llama-server -m laya.gguf -ub 4096
```

Laya's release written this way converts to the same GGUF tensors as the release itself. On the five requests Dew's tests ask, llama.cpp's server counts the same tokens and gives the same choices as Dew, with every probability within 0.0014 of Dew's. The temperatures are written as Laya's agent applies them, held within [0.5, 5], because llama.cpp applies what it reads: the release ships 0.1 for choices of eleven options or more, which Laya's agent and Dew raise to 0.5. A calibration's binning map or abstention thresholds have no place in that layout, so a task carrying them is refused.

vLLM (v0.31.0) cannot serve these: its pooling heads reduce a row to one vector and at most one linear layer, with no per-request positions or attention masks, so a decision head would have to be rewritten as a vLLM plugin.

## Confidence and calibration

A confidence is a statistic of the answer's distribution, and the task decides which one:

- `JevConfidence` (the default) is Jev's: for a choice of n options, how far the top probability sits above an even split, (p_max - 1/n) / (1 - 1/n); for a score, one minus the spread of probability around the most likely level, measured against an even spread.
- `EntropyConfidence` is Laya's, one minus the entropy over log n.
- `TopProbability` is the answer's own probability, the quantity calibration fits.

`replace(decide, confidence=EntropyConfidence())` switches it. A model's raw probabilities are rarely well calibrated. `decide.calibrated(held_out)` fits a temperature per question type and per option-count bucket on labelled examples, as Laya does. It can also add histogram binning, and abstention thresholds that keep the error of the accepted answers at a target. `decide.gated(0.8)` abstains below a fixed top probability. `decide.score(examples)` reports accuracy, expected calibration error, the area under the risk-coverage curve and the log loss.

## Fine-tune

`DecisionObjective` trains a decision head over any backbone that has hidden states. The backbone can be Laya's own checkpoint (head and layout included), a pretrained language model, adapted with LoRA or partly frozen, or a model built from scratch:

```python
from dew import Trainer
from dew.config import OptimConfig
from dew.decision import ECE, Accuracy, Brier, Choice, DecisionObjective, DecisionTable, LogLoss
from dew.interop import Pretrained
from dew.lora import LoRA

qwen = Pretrained.load("Qwen/Qwen3-0.6B").adapt(LoRA(rank=16, modules=("q_proj", "v_proj")), key=0)
banking77 = DecisionTable(path="mteb/banking77", label="label_text", question="intent",
                          instructions="Which banking request is this?")

objective = DecisionObjective(qwen, loss=LogLoss() + 0.5 * Brier())
data = objective.dataset(banking77, batch=32)  # a tenth held out, for validation and calibration
state = Trainer(objective, OptimConfig(learning_rate=1e-4), key=0).fit(
    data, steps=2_000, eval_every=500, metrics=[Accuracy(), ECE()])

decide = objective.pipeline(state).calibrated(data.val)
```

Examples need not come from a table: `objective.dataset` takes any labelled examples, `Example`s or mappings with a `state`, `questions` (as `Question`s or in Jev's wire form) and `answers`, an answer naming its option by key or index.

<!-- BANKING77 results (Laya zero-shot, Laya fine-tuned, Qwen3-0.6B + LoRA; accuracy and ECE) go here when the Colab runs finish. -->

Any model that gives its final states is a backbone (`dew.nn.protocols.HiddenStates`): a `CausalTransformer`, causal or bidirectional as ModernBERT is, the text path of a multimodal model such as Qwen 3.5, or DiffusionGemma, whose encoder reads each clean row once whole. A bidirectional backbone reads Laya's layout. A causal one reads `StateFirstLayout`, which puts the state first and each marker after its option, so every marker has read the state, the question and its option. The backbone and the head are applied as separate modules over one variables tree, so a LoRA backbone trains its factors and the head while the base weights stay fixed.

The head is a value too. Laya's `DecisionHead` scores one question per row at its markers. Clef's `JointSchemaHead`, with its `JointLayout`, reads every question of a request in one row and decides them together: it pools each question's instructions and each option's tokens, adds a lexical prior from the backbone's output table, and lets the questions attend to each other. `Decide.from_pretrained("Cloudflare/clef-flash")` reads a Clef release, its Qwen 3.5 backbone with the head beside it and the backbone's processor, and answers as Clef's own model does, images included, which the processor lays out before the state (`tests/test_decision_clef.py` holds the layout, the head and the two together to Clef's code, with and without images). Clef reports an answer's top probability as its confidence, where `systemone` reports Jev's; `replace(decide, confidence=TopProbability())` answers with Clef's. A release fine-tunes like Laya's, its head and layout coming with it. Training pads every row to the layout's `max_len`, and Clef's is the 16384 tokens it serves, so set one that fits the data:

```python
objective = DecisionObjective(Decide.from_pretrained("Cloudflare/clef-flash"), layout=JointLayout(max_len=2048))
```

The loss is a proper scoring rule: its expected value is smallest when the forecast is the true distribution, so it rewards honest probabilities. The rules are `LogLoss`, `Brier`, `Spherical` and `RankedProbability` (for score questions, whose levels are ordered), and they can be added and scaled. `label_smoothing` spreads part of the target over every option. During training a choice's options are reshuffled every time a row is read, so the model cannot learn their positions, and `none_of_the_above=p` adds a "none of the above" option to a share of rows, half of which lose their right answer to it. The same rules, with `Accuracy`, `ECE` and `AURC`, score the validation pass.

Other backbones are one line each:

```python
DecisionObjective(Decide.from_pretrained("convaiinnovations/laya"))           # Laya, fully fine-tuned
DecisionObjective(CausalTransformer(vocab_size=50368, num_layers=12, causal=False),
                  tokenizer=HFTokenizer("answerdotai/ModernBERT-base"))       # from scratch
DecisionObjective(Pretrained.load("Qwen/Qwen3-8B").adapt(LoRA(rank=256, modules=("q_proj", "v_proj")), key=0))
```

A run saves like any other: `Decide.from_run(run)` and `dew.pipeline(run)` rebuild the task from its record, and `decide.save(run)` writes the fitted calibration into it.

## From a table, on the command line

`recipes/decision/train.py` is `laya-train --data tickets.csv`:

```bash
python recipes/decision/train.py --data.path tickets.csv --data.text text --data.label label \
    --trainer.batch-size 32 --trainer.epochs 4
```

It starts from Laya's checkpoint, or from `--pretrained Qwen/Qwen3-0.6B --lora.rank 16 --lora.modules q_proj v_proj`. `DecisionTable` reads CSV, JSON, JSONL and parquet files and Hub datasets: each row's text is the state, and its label is the answer to one choice question. A tenth of the rows, at most 400, are held out. Validation scores them, and after training they are used to fit the temperatures saved into the run.

## Train a general decision model

Two recipes train a model that answers questions it has not seen, as Jev, Laya and Clef do:

- `recipes/decision/encoder.py` trains ModernBERT-large whole under a fresh Laya head, as Laya and OpenDecider-nano are built.
- `recipes/decision/clef.py` trains a fresh joint schema head with LoRA factors over a frozen Qwen3.5-4B, as Clef is built. Rows of up to 2,048 tokens, with every block recomputed, fit a 40 GB A100. `--pretrained Qwen/Qwen3.5-9B --lora.rank 256` gives Clef-flash's size.

```bash
python recipes/decision/contamination.py --out eval-index.npz --jsonl suite-rows.jsonl.gz \
    --hub LocalLLaMA/typed-decisions@d0e2f0c4:all/test-00000-of-00001.parquet
python recipes/decision/encoder.py --decontaminate eval-index.npz --trainer.checkpoint-dir runs
```

Both are `DecisionRunConfig` subclasses, so every value is a flag. Both read `recipes/decision/sources.py`'s `Mixture`, each set at its share of every step:

- Open-Jev's typed decisions;
- typed-decisions' train split;
- gliclass;
- BANKING77, CLINC150, WANLI, HellaSwag, ARC, BoolQ and GSM8K;
- your own rows (`--mixture.rows.path`).

Each set is pinned to a commit. Benchmark-style items are framed in both shapes requests take: the content as the state, or inside the question, and the options by name or by letter. Where a set's gold is a distribution, as Open-Jev's and typed-decisions' are, the example carries it as `targets`. Training then scores against the distribution rather than its most likely option. `DecisionObjective.dataset` takes a mapping of names to `Weighted` sets the same way.

`contamination.py` reduces evaluation items to hashes, and `--decontaminate` drops every training example whose content an evaluation item shares:
- A synthetic set's examples are compared whole.
- Natural text is also compared line by line and by 13-word runs.

The run records what it dropped from each set.

The loss is the log loss plus half the Brier score, plus the ranked probability score on ordered levels: the proper scoring rules that Laya's RLCD and Clef's post-training reward.

A trained run is measured where a decision model is used, at `/v1/systemone`. Serve it with `examples/serve_decisions.py --run runs/... --whole`; `--whole` refuses rather than cuts a request too long to answer whole, as Decision Index requires. Then:
- Decision Index runs through its own kit (`python -m decision_index pipeline --engine http`).
- typed-decisions, Open-Jev's test and OOD splits, and Laya's application battery run through `recipes/decision/benchmark.py`. Its typed-decisions scorer reproduces the dataset card's reference row.

## How this relates to Laya

Laya's encoder loads through Dew's ModernBERT family and its head maps onto `DecisionHead`. For the same requests, Dew lays out the same tokens as Laya's own code, and its option logits are within the 2x reference-error rule of Laya's (`tests/test_decision_laya.py`, `tests/reference_error.py`). Laya's inference options are available here as values: per-bucket temperatures, binning, abstention, the parallel option layout and tournaments. Two parts are not. Laya's action head is left out, because its own model card reports that it carries no signal; gate on confidence instead. Laya's router, which picks its English or multilingual checkpoint per request, is shown in `examples/route_decisions.py` and is not built in.
