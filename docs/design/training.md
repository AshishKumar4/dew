# Training in Dew: an objective, an optimizer, a mesh, and `fit`

Design record, 2026-10-12. This is the binding design for Dew's training and model API. Code blocks show the target API; implementation and qualification work lives in the [roadmap](../roadmap.md). Its data counterpart is [the data design](data.md). Shared code ownership is in [architecture](architecture.md), and behavioral qualification is in [testing](testing.md).

## 0. The idea in one paragraph

A `Trainer` runs an `Objective` (the loss and metrics for one kind of training: language modelling, diffusion, decision, preference, RL) with an Optax optimizer on a device mesh. `fit` takes a data chain and a budget. Everything between (sharding, micro-batches, rematerialization, kernels, compilation, checkpoints, resume) is chosen by Dew from the model, the mesh and the hardware, and every one of those choices can be overridden. The model is a plain Flax module, the optimizer is plain Optax, checkpoints are plain Orbax, and the mesh can be a plain `jax.Mesh`.

```python
model = CausalTransformer(vocab=50_257, width=768, depth=12, heads=12)
trainer = Trainer(LMObjective(model, seq_len=1024), optax.adamw(3e-4), key=0)
state = trainer.fit(train, validation=val, tokens=10e9)
```

```mermaid
flowchart LR
  M["Flax module"] --> O["Objective<br/>loss · metrics · data contract"]
  O --> T["Trainer"]
  X["Optax optimizer"] --> T
  Me["Mesh<br/>MeshSpec or jax.Mesh"] --> T
  C["Checkpoints<br/>Orbax"] -.-> T
  K["Tracker<br/>W&B · MLflow"] -.-> T
  D["Dataset chain"] --> F["fit(budget)"]
  T --> F --> S["TrainState"]
```

## 1. Models

Any Flax module trains. The objective calls the module's forward through Flax's `apply`. Binding a different forward method is tracked in the [roadmap](../roadmap.md#models-and-interop).

```python
class MyLM(nn.Module):
    def __call__(self, tokens, *, positions=None, segment_ids=None): ...

LMObjective(MyLM(...), seq_len=2048)
```

- Flax NNX modules go through Flax's own NNX-to-Linen bridge; Dew has no second module system.
- Variables are whatever Flax returns (plain dicts, or `FrozenDict` from older code). The Trainer normalizes the tree once at its boundary, so training, saving and restoring see one form.
- Pretrained weights come from `Pretrained.load("Qwen/Qwen3.5-0.8B")`, which returns the module, its variables and its processor.

### A context read through cross-attention

A decision (`dew.decision`) may read a context beside its state: `Example.context`, any JSON value, rendered as text by the decision's tokenizer and cut to its first `context_length` tokens. A source takes the gated cross-attention blocks that read it with `Pretrained.with_cross_attention(CrossAttention(layers, context_length), key=...)` (`DecisionRunConfig.context_layers` and `context_length` on the command line). The blocks start with zero gates, so the model computes what the source computes until they train. The variables hold the blocks under `params` and the source's weights under `FROZEN`, so the objective trains the blocks alone (and an adapter's factors, where `lora` adds them), and the source stays as loaded. A multimodal source takes the blocks on its language model and reads a context beside its images.

- **Packing.** Rows that carry a context pack one to a window (`DecisionObjective.bounds`), so each window's context is its row's; the batch holds each row's context tokens and their count (`context_lengths`).
- **Work.** `train/flops` counts the blocks' work over each row's real context tokens (`CrossAttending.cross_flops`): a frozen source's training work, plus three passes of the blocks and the head.
- **Serving.** `Decide(state, questions, context=...)` and `Decide.batch(requests, contexts=...)` read a request's context as training did, its images too.
- **No blocks attached.** Training over a source without cross-attention leaves `Example.context` unread: the rows carry no context, as a table with no context column trains. Serving refuses a context such a task cannot read (ValueError), and a model refuses rows that carry one (TypeError), so no answer silently ignores a context it was given.

## 2. The mesh

Two ways to describe the devices, one path underneath.

```python
# Convenient: counts per role; the data axis takes the devices left over.
Trainer(objective, opt, key=0, mesh=MeshSpec(fsdp=8, tensor=2))

# Native: your own jax.Mesh, any axis names, and how the model maps onto them.
mesh = jax.make_mesh((4, 8, 2), ("replica", "fsdp", "context"))
Trainer(objective, opt, key=0, mesh=mesh,
        layout=Layout(rules={"embed": "fsdp", "mlp": "fsdp", "length": "context"},
                      data_axes=("replica", "fsdp")))
```

`MeshSpec` builds Dew's six named axes (`data`, `fsdp`, `expert`, `tensor`, `sequence`, `stage`). It is serializable, so a run's `run.json` and the command line (`--mesh.fsdp 8`) can store it and a resumed run rebuilds the mesh on whatever hardware it lands on. Its `replicas=` orders devices for hybrid sharded data parallelism, so FSDP's gathers stay inside a host's NVLink domain and only the gradient all-reduce crosses the slower network.

A native `jax.Mesh` needs two facts from you in `Layout`: `rules`, which map the model's logical axes (`embed`, `mlp`, `heads`, `vocab`, `experts`, `length`, `layers`) onto mesh axes, and `data_axes`, which say which axes split the batch. `data_axes=` is shorthand that writes the one batch rule everything else reads: each host's data share, gradient reduction, and token accounting. Model-parallel algorithms keep their role names: expert exchange runs on an axis called `expert`, sequence attention on `sequence`, pipeline stages on `stage`. Any other axis, a second FSDP level or a context axis, is free for parameters and data through the rules. Dew refuses by name only what cannot work: a rule naming an axis the mesh lacks, or a parameter placed on a data axis.

```mermaid
flowchart LR
  subgraph Model["Logical axes (from the module)"]
    E["embed"]; Mm["mlp"]; H["heads"]; V["vocab"]; X["experts"]; L["length"]; B["batch"]
  end
  subgraph Mesh["Mesh axes"]
    F["fsdp"]; T["tensor"]; Ex["expert"]; S["sequence"]; D["data"]
  end
  E --> F; Mm --> F; Mm --> T; H --> T; V --> T; X --> Ex; L --> S; B --> D; B --> F
```

`microbatches=` for pipeline stages is a Trainer argument (default: one per stage), the same for both mesh forms.

Where the layout splits a leaf, the shared rule table and that leaf's declared logical axes place it; a leaf that declares none falls back to a shape heuristic, and refusing it instead is not built yet. A rule outside Dew's defaults (`DEFAULT_RULES`) that names an axis the mesh lacks is refused, a role axis among them, since it would split nothing; Dew's own rules take a missing role axis as size 1. A placement rule that deviates for one model family is a separate rule (expert parallelism's, tensor parallelism's), never a branch in the default.

On a native mesh, the layout is to report what it is about to do before the first step, as `Layout.validate(mesh)`, which is not built yet: which leaves split over which axes, the per-device bytes, the batch split, the collectives each logical axis implies, and an unbound logical axis, which would replicate that dimension. A mismatch between the rules and the mesh's axis names is to fail there, not in GSPMD.

## 3. Memory and speed are chosen for you

The memory planner picks, per model, mesh and GPU, the cheapest combination that fits: micro-batch size, rematerialization policy, LM-head tiling and host offload. The planner checks each candidate against its compiled program's memory analysis. Where none fits, `fit` raises naming the split, rematerialization and head options it tried, and the closest candidate's temporary and argument bytes where the analysis gave them. The same `batch(1024)` works on one GPU or 256. You override any one of them:

```python
Trainer(objective, opt, key=0, accumulation=4)                    # fix the micro-batch split
Trainer(objective, opt, key=0, layout=Layout(host=("optimizer",)))  # keep optimizer state in host memory
```

Every eligible model takes the planner's path: no model family needs a planner of its own. Kernels are chosen per GPU generation from one table (`dew.nn.kernels`): FlashAttention 3 on Hopper, FlashAttention 2 on Ampere and Blackwell workstations, fla's gated delta rule, fused norms and SwiGLU. A kernel replaces the XLA path only when it is measured faster and passes the float64 accuracy gate on that GPU. Each kernel derives its tile sizes from the GPU's facts and the call's shape (`dew.nn.kernels.tuning`); nothing is measured on the user's side.

## 4. `fit`

```python
state = trainer.fit(
    train,                                    # a Dataset chain, a Grain dataset, or arrays
    validation={"web": val_web, "code": val_code},
    tokens=50e9,                              # or steps= or epochs=
    eval_every=1000,
    checkpoint_every=datetime.timedelta(minutes=30),
    best="validation/web/loss", restore_best=True,
    stop=Plateau(patience=5),
)
```

- One budget: `steps=`, `epochs=` or `tokens=` (committed valid target tokens, the same clock the data `schedule` reads). A run's tokens grow only at a committed update. `fit` checks the budget between steps, so after a rejected microbatch a split step can reach it at a commit and keep a later partial window; ending at that commit is not built yet.
- `validation` is one chain or a dict of named chains, each reported separately.
- `fit` returns the final `TrainState`. `state=` supplies initial state rather than a continuation; checkpoint restore also restores the data position.

### The batch contract

A data mismatch fails within seconds of launch, on every host at once, never hours into a run.

1. The data chain declares what it yields: each stream's fields, per-row shapes and dtypes, with the batch sharding from the mesh. This is MaxText's `get_shaped_batch`, the producer's shape. It is known before any read where the chain fixes it (presets, `pack`, buckets, `batch`); a chain with your own `map`s takes it from its first batch.
2. The step compiles from those shapes, `jit(step).lower(abstract_state, abstract_batch).compile()`, while the readers open. A `schedule` gives one shape per stage. The readers' shapes are compared with what the run already declares it reads, with no new declaration: `InputSpec`'s sample and mask, each condition encoder's token length, a supervised model's input shape, the model's vocabulary for id fields, and the host-only fields a metric names (such as the captions a CLIP score compares). Lowering then catches any other missing field. A wrong image size is caught here, because a convolution would trace at the wrong size without complaint.
3. Each host reads only its share and checks it as numpy, before `jax.make_array_from_process_local_data` turns it into a global array: field names, per-row shapes, dtypes, the share's row count, and the bounds an id field declares (a gather past the vocabulary would clamp silently on the device). At start, `fit` reads the first batch of the training stream and of every validation stream while it initializes, restores and compiles. These checks run inside `agreed()`, so all hosts report every mismatch in every stream in one error before step 1. The batches it read are used, not read again. `objective.check(batch)` runs the same path on a laptop before a costly launch.
4. Later batches take the same local comparison before transfer, with no collective. Shapes are fixed by construction (presets, `pack`, `batch` without a partial last batch), so a mismatch means a step broke one: the host raises naming the step and records, and JAX's coordination service stops the others. A `schedule` stage's new shape is expected and compiled once.

## 5. Checkpoints and resume

```python
Trainer(objective, opt, key=0,
        checkpoints=Checkpoints("gs://runs/lm-1b", keep=Keep(latest=2, every=10_000)))
```

- A run with a checkpoint directory resumes automatically from its latest checkpoint: parameters, optimizer state, step, RNG and each data share's position. Resume is exact on the same mesh.
- A checkpoint directory belongs to one run. A fresh run takes a new directory; a directory holding another run's checkpoints refuses to resume, naming what differs.
- A resumed run skips loading pretrained weights it is about to overwrite, and its first step hits the compile cache.
- `Keep` decides retention; `where=` sees each checkpoint's real record (kind, step, metrics).
- The directory is a plain Orbax `CheckpointManager` directory, readable without Dew.

## 6. Compilation

Compiled programs depend only on the model, shapes, dtypes, shardings and configuration, never on where a run happens to be: a compiled program captures no run path, temporary directory or checkout location. The persistent compile cache is shared across runs, processes and checkouts. A resumed or repeated run reuses its programs. `lower_for=` lowers a step for a GPU generation on any host. Compiling that lowered step before a GPU is held and carrying its executable into the run is tracked in the [roadmap](../roadmap.md#training-and-parallelism).

## 7. Seeing what happens

Every run logs canonical MFU (analytic model FLOPs on real tokens over wall time) and where step time goes. A model reports its FLOPs through `model_flops`; a model without them reports no MFU rather than an estimate. Each kernel dispatch records which kernel it chose and why (`dew.nn.kernels.generation.routes`); collecting those choices into a run's report is not built yet. Trackers (W&B, MLflow) take the same metrics; the local metric journal is JSONL. The [roadmap](../roadmap.md#kernels-and-performance) tracks a `dew profile` command that reports the step breakdown, chosen kernels and top three losses from the existing attribution tools.

## 8. After training

```python
pipe = dew.pipeline("runs/lm-1b")                 # the trained model with its processor
pipe("The capital of France is", max_new_tokens=32, key=0)
pipe([{"role": "user", "content": "Hi"}], key=1)  # chat, no PyTorch needed

Pretrained.from_run("runs/lm-1b").save("out/")     # loads in transformers and vLLM
adapter.save("out/lora")                          # a complete PEFT adapter
```

A reloaded run behaves like the live one: the same processor, chat template and generation defaults.

## 9. The shared seams, as a user sees them

| Seam | What it guarantees |
|---|---|
| Token coordinates | Packed, padded and cached rows give the same result as separate ones, in every mixer (attention, GDN, KDA, Mamba). |
| Layer identity | Quantization survives fused projections, scanned layers and pipeline stages. |
| One LM head | Full logits, the chunked loss and scoring compute the head the same way, quantized or not, sharded or not. |
| Loading readers | Every model family loads and exports through one reversible weight-layout plan. |
| Checkpoint state | Saving and restoring derive from the training state's own schema; restore casts and places once. |

Each seam requires one standing end-to-end test that runs every implementation registered for it. The [roadmap](../roadmap.md#ci-and-tooling) tracks the registry-wide coverage still to add.

## 10. Examples

### LoRA fine-tuning a chat model, exported for vLLM

```python
bundle = Pretrained.load("Qwen/Qwen3.5-0.8B")
adapted, adapter = bundle.adapt(LoRA(rank=16), key=0)
chats = Hub("HuggingFaceH4/ultrachat_200k", split="train_sft").map(chat.template(bundle.processor)).pack(seq_len=4096)
trainer = Trainer(LMObjective(adapted, seq_len=4096, loss_role=Role.ASSISTANT), optax.adamw(1e-4), key=0)
trainer.fit(chats.shuffle(seed=0).batch(64), epochs=2)
adapter.save("out/lora")
```

### A diffusion model, then samples

```python
model = DiT.preset("dit-b/2")
data = Files("gs://data/laion/*.array_record").filter_map(images.DecodeImage()).map(images.Resize(256)).batch(256)
trainer = Trainer(DiffusionObjective(model, process=FlowMatching(), vae="stabilityai/sd-vae-ft-ema"), optax.adamw(1e-4), key=0)
state = trainer.fit(data, steps=400_000)
images = dew.pipeline(trainer)("a red fox in snow", steps=50, guidance=4.0, key=0)
```

### The same LM script on 64 GPUs

```python
trainer = Trainer(LMObjective(model, seq_len=8192), optax.adamw(3e-4), key=0,
                  mesh=MeshSpec(fsdp=8, replicas=8),
                  checkpoints=Checkpoints("gs://runs/lm-8b"))
trainer.fit(data, tokens=2e12)    # launched on every host with `dew launch`
```

