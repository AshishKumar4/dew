# Decision models

A decision model answers typed questions about a state in one forward pass. It generates no text: each question names its options, the model scores every option at once, and the answer is a probability distribution over exactly those options. `dew.decision` covers the three question types of TypeSafe's Jev API, and it loads [Laya](https://huggingface.co/convaiinnovations/laya), an open decision model built on ModernBERT-large, as a native Dew model.

| Question | Options | Answer |
|---|---|---|
| `Noul(instructions)` | false, true | `noul`, the probability of yes |
| `Choice(instructions, criteria)` | 1 to 255 named options, each with an optional description | `choice`, the most likely option |
| `Score(instructions, criteria)` | 2 to 10 ordered levels, lowest first | `score`, the expected level |

## Answer questions

<!-- not run: loads the Laya release, near 4 GiB on CPU, past the docs test's budget -->
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

<!-- not run: asks the Laya task the block above loads -->
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

The response has exactly Jev's fields, with numbers rounded to four places. `details=True` adds what Laya's server adds: a noul's confidence, whether a gated answer abstained, and how much of the state the rows kept. A request may also carry `context`, any JSON a task with context layers reads beside the state (refused past its `context_length`, and dropped under `strict`), and `images`, PIL images or encoded images, bare or as `data:` URLs, which a task whose backbone reads images lays out with the state, as a joint-head release does (`decide(state, questions, images=[...])` takes them too); `strict=True` answers exactly as Jev's endpoint does, dropping such extension fields.

`systemone` also takes a list of request bodies and returns their responses in order, each as it would answer the body alone, the bodies' rows scored together in one pass per budget's worth (the task's `Budget`). A pass packs whole rows into a few fixed-length windows, the way a training run packs them, each row attending only its own tokens, so a pass computes little padding and each row is scored as it is alone. Rows that hold images share a pass where they are as long as each other and their images alike, and a body Jev would refuse takes the `ValueError` it would raise alone in its place, the others answered all the same.

Dew ships no HTTP server of its own, but `systemone` is all one needs. `examples/serve_decisions.py` serves it with Starlette, at Jev's `POST /v1/systemone`, with Jev's 422 for a request it refuses and an optional bearer key. It answers concurrent requests together through Dew's serving engine (`dew.inference.Engine`), whose scoring kind (`dew.inference.ScoringServer`) runs one forward pass a turn over the rows waiting, at most `--batch-rows` rows and `--batch-tokens` padded tokens a pass; a request whose rows outrun one pass carries the rest to the next turn, so a long backlog never holds the engine's other kinds back:

```bash
uv pip install "dewml[serve] @ git+https://github.com/AshishKumar4/dew" \
    --extra-index-url https://pypi.dewml.dev/simple
python examples/serve_decisions.py --model convaiinnovations/laya --port 8000
```

llama.cpp (release b11445) serves decision models natively at the same `POST /v1/systemone`, with images as `data:` URLs: its converter reads Laya's checkpoint layout (ModernBERT with Laya's head) and the joint-head one (a Qwen 3.5 backbone with `joint_head.safetensors`). `decide.save_pretrained(directory)` writes a ModernBERT task in Laya's layout, temperatures included, which `Decide.from_pretrained` reads back and llama.cpp converts:

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

<!-- not run: fine-tunes Qwen3-0.6B on BANKING77 for 2,000 steps, a GPU job -->
```python
from dew import Trainer
from dew.optim import OptimConfig
from dew.data import Hub
from dew.data.compile import held
from dew.decision import ECE, Accuracy, Brier, DecisionObjective, LogLoss, decisions
from dew.interop import Pretrained
from dew.lora import LoRA

qwen = Pretrained.load("Qwen/Qwen3-0.6B").adapt(LoRA(rank=16, modules=("q_proj", "v_proj")), key=0)
# A tenth of the rows, by the hash of each one's content, the same in every run.
train, val = Hub("mteb/banking77").holdout(0.1)

def asked(rows):
    return rows.pipe(decisions, label="label_text", question="intent",
                     instructions="Which banking request is this?").read("memory")

objective = DecisionObjective(qwen, loss=LogLoss() + 0.5 * Brier())
state = Trainer(objective, OptimConfig(learning_rate=1e-4), key=0).fit(
    asked(train).shuffle(0).repeat().batch(32), steps=2_000, eval_every=500,
    validation={"val": asked(val).batch(32)}, metrics=[Accuracy(), ECE()])

decide = objective.pipeline(state).calibrated(held(asked(val).batch(32), objective=objective.data()))
```

Examples need not come from a table: `objective.dataset` takes any labelled examples, `Example`s or mappings with a `state`, `questions` (as `Question`s or in Jev's wire form) and `answers`, an answer naming its option by key or index.

<!-- BANKING77 results (Laya zero-shot, Laya fine-tuned, Qwen3-0.6B + LoRA; accuracy and ECE) go here when the Colab runs finish. -->

Any model that gives its final states is a backbone (`dew.nn.protocols.HiddenStates`): a `CausalTransformer`, causal or bidirectional as ModernBERT is, the text path of a multimodal model such as Qwen 3.5, or DiffusionGemma, whose encoder reads each clean row once whole. A bidirectional backbone reads Laya's layout. A causal one reads `StateFirstLayout`, which puts the state first and each marker after its option, so every marker has read the state, the question and its option. Either layout writes its row in another order where `order=` says: `question_first`, `state_first`, or `re_read`, the state, the question and the state again, so a causal backbone reads the state after the question too; each order keeps the built-in row's begin token, separators, header, declared budgets and a conversation's kept end. A decision run names it by `--order`. Over a backbone that reads a row's blocks whole (a `block_bidirectional` `--source.attention`), `--split whole` keeps the row one block read both ways, and `--split part` splits a question-first row at its state. The backbone and the head are applied as separate modules over one variables tree, so a LoRA backbone trains its factors and the head while the base weights stay fixed.

The head is a value too. Laya's `DecisionHead` scores one question per row at its markers. The `JointSchemaHead`, with its `JointLayout`, reads every question of a request in one row and decides them together: it pools each question's instructions and each option's tokens, adds a lexical prior from the backbone's vocabulary rows, and lets the questions attend to each other. A backbone gives those rows through `dew.nn.protocols.VocabularyRows`: `vocabulary_rows()` returns each token's `[vocab, features]` row as its head scores it, read from the bound variables, so past a nonlinear head or with an adapter on the head the rows are the ones the head applies. `CausalTransformer`, `MultimodalTransformer`, DiffusionGemma and the torchax fallback implement it, and a custom backbone under the joint head implements it too; the head refuses one that does not. `DecisionRunConfig(pretrained=backbone, head=JointShape())` draws one over a fresh backbone, which `backbone` names: a Hugging Face model, not a Laya checkpoint, since a Laya checkpoint trains under its own head. `Decide.from_pretrained` also reads a released joint-head checkpoint (through `Pretrained.load`), a Qwen 3.5 backbone with `joint_head_config.json` and `joint_head.safetensors` beside it, and the backbone's processor, and answers as the release's own model does, images included, which the processor lays out before the state (`tests/test_decision_joint_checkpoint.py` holds the layout, the head and the two together to the release's code, with and without images). Such a release reports an answer's top probability as its confidence, where `systemone` reports Jev's; `replace(decide, confidence=TopProbability())` answers with the top probability. A release fine-tunes like Laya's, its head and layout coming with it. Training pads every row to the layout's `max_len`, and a release's is the 16384 tokens it serves, so set one that fits the data:

A head may instead read the backbone's own answer. A question layout given `labels=Labels()` writes each option as a lettered line, `A. billing`, and ends its row with an answer cue, `Answer:`; `LabelReadout` scores each option as the backbone's own logit for its letter there, so before any training the decision is what the language model would answer. `untied=True` gives each letter a row of its own, drawn from the backbone's, to train. `SlotReadout` is the head of a published slot release, read by its own law. A decision run over a fresh backbone chooses its head by the `head` subcommand: `head:joint-shape` draws the joint schema head, `head:letters` the letter readout over rows a question to a row, lettered by its `labels` (`dew.decision.config.Letters`, up to `count` options, a none-of-the-above option among them), and none Laya's head. `--source.attention full_attention block_bidirectional` loads the backbone reading each of a row's blocks whole, and a fresh head's rows then lie in blocks, as a slot release's noncausal attention lays its own; a joint row lies in no blocks and is refused.

<!-- not run: a fragment; the release is the reader's own, a directory or a Hub repo id -->
```python
objective = DecisionObjective(Decide.from_pretrained(release), layout=JointLayout(max_len=2048))
```

Training and validation pack rows first-fit into windows of the layout's `max_len` tokens, each row a segment the backbone and the head read apart from the others, so a step pays for its rows' tokens rather than `max_len` per row and every batch has one shape: a run compiles one step. A window holds as many rows, questions and options as the layout's token costs let `max_len` tokens hold (`Layout.bounds`), and at most the objective's `segments` rows, 16 by default, which the backbone's attention kernel works on a window whether or not they come; the packer closes a window at whichever comes first. A mix of rows shorter than `max_len / 16` tokens fills its windows better with a larger `segments`. The loss is the mean over the batch's scored questions: every scored question in a batch weighs the same, however many tokens its row spends, where a token budget's loss once weighed a question by its batch's budget. `tests/test_decision_packing.py` holds each row of a window to what it decides alone, through Qwen 3.5's Gated DeltaNet and attention.

The loss is a proper scoring rule: its expected value is smallest when the forecast is the true distribution, so it rewards honest probabilities. The rules are `LogLoss`, `Brier`, `Spherical` and `RankedProbability` (for score questions, whose levels are ordered), and they can be added and scaled. `label_smoothing` spreads part of the target over every option. During training a choice's options are reshuffled every time a row is read, so the model cannot learn their positions, and `none_of_the_above=p` adds a "none of the above" option to a share of rows, half of which lose their right answer to it. A row the added option would leave past its window, or its options alike under a declared budget, is read as it was admitted instead, and counted under that reason in the run's data counters (`data/declined/<label>/<reason>`) and its summary (`none_of_the_above_declined`). The same rules, with `Accuracy`, `ECE` and `AURC`, score the validation pass.

Other backbones are one line each:

<!-- not run: loads Qwen3-8B, beyond a CI container's memory -->
```python
DecisionObjective(Decide.from_pretrained("convaiinnovations/laya"))           # Laya, fully fine-tuned
DecisionObjective(CausalTransformer(vocab_size=50368, num_layers=12, causal=False),
                  tokenizer=HFTokenizer("answerdotai/ModernBERT-base"))       # from scratch
DecisionObjective(Pretrained.load("Qwen/Qwen3-8B").adapt(LoRA(rank=256, modules=("q_proj", "v_proj")), key=0))
```

A run saves like any other: `Decide.from_run(run)` and `dew.pipeline(run)` rebuild the task from its record, and `decide.save(run)` writes the fitted calibration into it. Runs of one task trained apart from one start (the same held head, or a fresh head drawn from the same trainer key, over the same backbone and layout) average: `Decide.averaged([run_a, run_b], calibrate_on=held_out)` returns the task over the fp32 mean of their weights (`dew.checkpoints.averaged`, which averages any runs of one model), its temperatures refit on the held-out answers, since a calibration belongs to one set of weights. Its `weights` record every run it averaged, so no one run's directory holds it: `averaged.save("runs/averaged")` writes it to a directory of its own as a run, which `Decide.from_run` and `dew.pipeline` load and serve as any run, whatever its layout.

## From a table, on the command line

`recipes/decision/train.py` fine-tunes on BANKING77 by default:

```bash
python recipes/decision/train.py --trainer.epochs 3
```

It starts from Laya's checkpoint, or from `--pretrained Qwen/Qwen3-0.6B --lora.rank 16 --lora.modules q_proj v_proj`. A table of your own is a run file that names its chains, which `dew train` reads. `decisions` reads CSV, JSON, JSONL and parquet files and Hub datasets: each row's `text` is the state, and its `label` the answer to one choice question.

<!-- not run: the file `dew train tickets.py` reads -->
```python
# tickets.py
from dew.data import Files
from dew.decision import decisions
from dew.decision.config import DecisionRunConfig

train, held = Files("tickets.csv").holdout(0.1)
run = DecisionRunConfig(data=train.pipe(decisions, question="team").read("memory").shuffle().repeat().batch(32),
                        validation=held.pipe(decisions, question="team").read("memory").batch(32))
```

```bash
dew train tickets.py --set trainer.epochs=4
```

A tenth of the rows, by the hash of each, are held out. Validation scores them, and after training they are used to fit the temperatures saved into the run. A memory read takes the options from the whole table's labels; another read needs them given (`options=`) or a store's class names.

## How this relates to Laya

Laya's encoder loads through Dew's ModernBERT family and its head maps onto `DecisionHead`. For the same requests, Dew lays out the same tokens as Laya's own code, and its option logits are within the 2x reference-error rule of Laya's (`tests/test_decision_laya.py`, `tests/reference_error.py`). Laya's inference options are available here as values: per-bucket temperatures, binning, abstention, the parallel option layout and tournaments. Two parts are not. Laya's action head is left out, because its own model card reports that it carries no signal; gate on confidence instead. Laya's router, which picks its English or multilingual checkpoint per request, is shown in `examples/route_decisions.py` and is not built in.
