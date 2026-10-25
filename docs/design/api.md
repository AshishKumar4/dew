# The API: one registry, one objective, one trainer

This page records the original API design and how it was built. The statistics-first C01/C02 cutover replaced its scalar-loss, MultiSteps, mutable-router and checkpoint-field examples. Its registry decorators and name records were later replaced by import paths and static aliases (`dew.registry`). For contracts you can run, read the [current core API](../reference/core-api.md) and the [objective guide](../concepts/objectives.md).

Design record, 2026-09-03. The code blocks are design sketches, not runnable code. The examples describe the revisions they cite.

## 0. The decisions

| # | Decision | Where |
| --- | --- | --- |
| 1 | `Trainer` and `Objective` stay the two nouns. No `TrainProgram`, `Execution`, `Runner` layer. | 3.3, 3.4 |
| 2 | Everything extensible has one `Registry`: a decorator, a `Mapping` and an attribute view. `models["simple_dit"]`, `models.SimpleDiT` and the class are the same object. | 3.1 |
| 3 | Values cross `jit`, objects author them, protocols grant effects. A `flax.struct.dataclass` for state, a frozen dataclass for configuration, a `Protocol` for anything with two implementations. | 2 |
| 4 | Randomness is a `key` argument. `RandomMarkovState` goes. Step keys are `fold_in(run_key, step)`. | 3.2 |
| 5 | Effects live in capabilities handed to the trainer (`Tracker`, `Checkpoints`). Constructing anything opens nothing. | 3.4 |
| 6 | An unknown name or field raises. No dropped keys, no swallowed keyword arguments, no `+suffix` string DSL. | 3.1, 3.9 |
| 7 | No migrations and no converters. Dew is unpublished; the checkpoint layout, the state tree and the config shape change outright. `CONTRIBUTING.md`'s "frozen" list becomes "frozen at 1.0". | [Contribution rules](../../CONTRIBUTING.md#design) |
| 8 | Diffusion vocabulary stays (`NoiseScheduler`, `PredictionTransform`, the sampler names). Ownership moves into one `Process` value per convention; presets are dataclasses in a registry. | 3.5 |
| 9 | Discrete diffusion gets its own small algebra in `dew.diffusion.discrete`, shaped like the Gaussian one. It is not a special case of it. | 6.1 |
| 10 | The objective's loss returns an `Aux` that can carry updated non-parameter collections. That is the one channel for MoE balancing, batch statistics and sown values. | 3.3, 6.2 |
| 11 | A custom compiled step is the one escape hatch (`Trainer(step=...)`). There is no second trainer. | 6.5 |
| 12 | Sharding is declared on the module that owns the parameters, with a decorator, and merged into a `Layout`. The table in `training/distributed.py` moves out of the training package. | 3.8 |

## 1. What is wrong today

The review found the mechanics sound and the ownership wrong. The compiled step (`src/dew/training/objective_trainer.py:237-299`), the sharded materialisation of the state (`trainer.py:226-241`), cross-mesh restore (`trainer.py:281-309`), the device prefetch iterator (`distributed.py:312-371`), the attention kernel seam and KV cache (`nn/attention.py:33-224`), the chunked cross entropy, `generate` and the HF decoder translation are correct where checked, and the suite asserts values against references.

The seams named in `CONTRIBUTING.md:9` are crossed in these places, all confirmed by running the import graph:

| Seam | Evidence at `102baa4` |
| --- | --- |
| The trainer knows diffusion | `ObjectiveTrainer.__init__` takes `input_config: DiffusionInputConfig`, `noise_schedule`, `model_output_transform`, `autoencoder`, `unconditional_prob`, `native_resolution`, `loss_fn`, `name="GeneralDiffusion"`, and builds a `DiffusionObjective` when given none (`objective_trainer.py:70-146`). `import dew.training` loads `dew.diffusion`, `dew.inputs`, `dew.nn.autoencoders` and `dew.sampling`. |
| Other objectives pay for that | JEPA constructs a `DiffusionInputConfig(conditions=[])` to hand the trainer shapes (`examples/train_jepa.py:59-60`). The LM example passes `input_config=None` and the model twice (`examples/train_lm.py:57-62`). `ObjectiveTrainer(model=None, objective=...)` fails at `model.apply` (`objective_trainer.py:204`). |
| Validation options tunnel through `fit` | `fit(**validation_step_args)` reaches `objective.make_validation_step(**kwargs)` (`objective_trainer.py:215-226, 301-302`); the diffusion example passes `sampler_class` and `sampling_noise_schedule` to `fit` (`examples/train_diffusion.py:66-67`). |
| The objective holds the tracker | `log_validation_artifacts(self, wandb, ...)` (`objectives/base.py:66`); both objectives import wandb. |
| The trainer publishes and deletes | `save()` pushes to the W&B registry, ranks the run against the W&B API and removes the local checkpoint (`objective_trainer.py:531-565, 388-529`). |
| Constructors have effects | `SimpleTrainer.__init__` enables the compile cache, builds the mesh, opens W&B, downloads an artifact, creates the Orbax manager and restores state (`trainer.py:89-217`). `ConditionalInputConfig.__post_init__` runs the encoder (`inputs/__init__.py:26-31`). |
| Data is a dict protocol | Six loaders return `{"train", "train_len", "val", "val_len", "local_batch_size", "global_batch_size"}`; `fit` reads them by name (`trainer.py:599-604`). |
| Sharding is a string table | `DEFAULT_LOGICAL_PARAM_AXES` keyed on trailing module names (`training/distributed.py:63-91`), read again by Muon grouping (`optim.py:95`). A rename in a model file changes both, silently. |
| One fact, two owners | The training schedule holds the transform for min-SNR (`schedules/common.py:22,58`) while the objective receives the transform again; presets return a 3-tuple (`transforms.py:157`). Gradient accumulation lives in `OptimConfig`, `optax.MultiSteps` and the trainer. |
| Randomness is an object | `RandomMarkovState` (`random_state.py`) appears in schedules, every sampler, the objective and the trainer state, and is checkpointed. |

## 2. The six rules

1. Everything extensible has one `Registry`: a decorator, a `Mapping`, an attribute view.
2. Everything that crosses `jit` is a PyTree (`flax.struct.dataclass`). Everything that configures is a frozen dataclass. Everything with interchangeable implementations is a `Protocol`. Nothing is two of these.
3. Randomness is a `key` argument. Per-step keys are `jax.random.fold_in(run_key, step)`.
4. Effects (disk, network, W&B) live only in capabilities handed to the `Trainer`.
5. An unknown name or field raises.
6. A schedule is a value, a sampler is a solver step, a preset is a `Process`.

Every surface below follows all six rules, and a reviewer can check a new module against the list.

## 3. The surfaces

### 3.1 `Registry`

```python
# dew/registry.py
class Registry(Mapping[str, T]):
    """Names for one kind of thing. Decorator, mapping and attribute view in one."""
    def __init__(self, kind: str): ...
    def __call__(self, name: str, /) -> Callable[[T], T]:      # @models("simple_dit")
    def __getitem__(self, name: str) -> T                       # models["simple_dit"]
    def __getattr__(self, attr: str) -> T                       # models.SimpleDiT
    def __iter__(self), __len__(self)
    def build(self, name: str, /, **fields) -> Any              # unknown field raises
    @property
    def union(self) -> type                                     # Union[...] of members, for tyro

models     = Registry[type[nn.Module]]("model")
presets    = Registry[type[Preset]]("preset")
samplers   = Registry[type[Solver]]("sampler")
encoders   = Registry[type[ConditionEncoder]]("encoder")
metrics    = Registry[Callable[..., Metric]]("metric")
objectives = Registry[type[Objective]]("objective")
```

```python
@models("simple_dit")
class SimpleDiT(nn.Module):
    scan_order: Literal["raster", "hilbert", "zigzag"] = "raster"   # replaces use_hilbert, use_zigzag and "+hilbert"
```

`build` validates against `dataclasses.fields`, which the fix in `8b34869` already does for `build_model`. `canonicalize_architecture` and `ARCHITECTURE_SUFFIX_FLAGS` (`registry.py:43-86`) go; `SimpleDiT.scan_order` is already a property (`nn/backbones/dit.py:33-37`) and becomes the field.

### 3.2 Values that cross `jit`

```python
@struct.dataclass
class TrainState:
    step: jax.Array
    params: Variables            # the objective's whole tree, every collection
    opt_state: optax.OptState
    ema: Variables | None
    key: jax.Array               # the run key; step keys are fold_in(key, step)

@struct.dataclass
class Step:                      # what an objective sees in one call
    step: jax.Array
    key: jax.Array
    ema: Variables | None

@struct.dataclass
class Aux:
    metrics: dict[str, jax.Array]
    variables: Variables | None = None     # non-parameter collections to write back
```

`metrics`, `dynamic_scale`, `rngs`, `best_loss` and `epoch` leave the checkpoint (decision 7). `dynamic_scale` becomes a trainer setting and its state a trainer-owned leaf outside the objective's tree.

### 3.3 `Objective`

```python
class Objective(ABC):
    """What is being learned: parameters, loss, what evaluation produces."""
    inputs: InputSpec                    # per-example shapes and dtypes the tree is initialised from
    ema: EMASpec | None = None

    @abstractmethod
    def init(self, key: jax.Array, variables: Variables | None = None) -> Variables: ...
    @abstractmethod
    def loss(self, params: Variables, batch: Batch, step: Step) -> tuple[jax.Array, Aux]: ...
    def evaluate(self, params: Variables, batch: Batch, step: Step) -> Artifact | None:
        return None                      # pure, jit-able; reads step.ema for the averaged weights

@dataclass(frozen=True)
class EMASpec:
    decay: optax.Schedule
    select: PathFilter = everything      # JEPA: under("params", "context_encoder")

@dataclass(frozen=True)
class InputSpec(Mapping[str, jax.ShapeDtypeStruct]): ...
```

`make_validation_step(**kwargs)` and `log_validation_artifacts(wandb, ...)` go. Evaluation options are constructor arguments of the objective, so nothing tunnels through `fit`. The trainer writes the collections in `Aux.variables` back into `state.variables`; that rule serves MoE balancing, batch statistics and sown values (section 6.2). One `PathFilter` selects the EMA subtree, `optax.multi_transform` labels and frozen subtrees; today those are three conventions.

### 3.4 `Trainer`

```python
trainer = Trainer(
    objective, optimizer,
    key=jax.random.key(0),
    mesh=MeshSpec(fsdp=4, expert=1),          # data fills the rest, as build_mesh does now
    layout=Layout(rules=DEFAULT_RULES, min_shard=2**16, tolerance=0.02),
    accumulation=4,                            # the one owner; wraps MultiSteps itself
    dynamic_scale=False,
    checkpoints=Checkpoints("runs/flowers", keep=2),
    tracker=WandbTracker(project="dew", name="flowers"),   # or None
)
state = trainer.fit(data, steps=100_000, log_every=100, eval_every=2_000,
                    checkpoint_every=5_000, metrics=(metrics["fid"](), metrics["clip_score"]()))
```

The trainer keeps what `CONTRIBUTING.md:9` gives it: mesh, compiled step, EMA, checkpoints, logging. The compiled step is `objective_trainer.py:237-299` with `Step` in place of positional `ema_params` and `rng`, plus the `Aux.variables` write-back. `fit` is step-based; `steps = data.epoch_steps(epochs)` is a recipe's line, counting the step an epoch's short last batch takes where its order reads on (`Loader.tail`). Publishing becomes `dew.io.publish(checkpoints.latest(), ...)`, called by a recipe after `fit`. Validation exceptions propagate (fixed in `6b747dc`).

```python
class Tracker(Protocol):
    def log(self, scalars: Mapping[str, float], step: int) -> None: ...
    def artifact(self, value: Artifact, step: int) -> None: ...

class WandbTracker:
    render = functools.singledispatch(lambda value: None)
    @render.register(ImageGrid) ...  @render.register(TextSamples) ...  @render.register(VideoGrid) ...
```

`singledispatch` on the artifact type is the whole "objectives return typed artifacts, the tracker draws them" mechanism. `Checkpoints` is a thin class over the Orbax manager (`save(step, state, position, metrics)`, `restore(template)`, `latest`, `wait`) with the preservation policy of `trainer.py:172-181`.

### 3.5 Diffusion

```python
class NoiseScheduler(ABC):                           # a value: no RNG object, no transform reference
    T: float
    def rates(self, t) -> tuple[Array, Array]: ...   # alpha, sigma
    def sample_t(self, key, n) -> Array: ...
    def weight(self, t) -> Array: ...                # the schedule's own weighting
    def model_time(self, t) -> Array: ...            # was transform_inputs' second output

@dataclass(frozen=True)
class Process:
    """One convention a model is trained and sampled with."""
    schedule: NoiseScheduler
    prediction: PredictionTransform
    weighting: Weighting = ScheduleWeighting()       # or MinSNR(gamma), computed here from both parts
    sampling: NoiseScheduler | None = None           # EDM samples on Karras; None means the same schedule

@presets("edm")
@dataclass(frozen=True)
class EDM:
    sigma_min: float = 0.002; sigma_max: float = 80.0; rho: float = 7.0
    sigma_data: float = 0.5
    regime: Literal["pixel", "latent"] | None = None  # Karras 2022's or EDM2's training sigmas
    P_mean: float | None = None; P_std: float | None = None  # stated, they override the regime's
    min_snr_gamma: float | None = None
    def __call__(self) -> Process: ...

process = presets.EDM(sigma_data=0.5, regime="pixel")()
```

`get_diffusion_preset` and its 3-tuple go. min-SNR no longer needs the schedule to hold the transform (`schedules/common.py:28-31, 55-58`). The preset dataclass is what the run manifest stores and the pipeline rebuilds, which is what the bug fixed in `a65d447` needed.

```python
class Solver(Protocol):
    State: type                                              # () for one-step solvers; history for multistep
    def init(self, x, times, process, *, key) -> State: ...          # times: the grid the walk takes; key: the walk root
    def step(self, x, t, t_next, denoised, eps, state, key, process, denoise) -> tuple[Array, State]: ...

samples = sample(denoise, x_T, steps, solver=samplers.Heun(), guidance=CFG(4.0, interval=(0.4, 0.6)), key=key)
```

`sample` is one `lax.scan`. `denoise` is `process.denoiser(model, params, conditions)`; `CFG` wraps a denoiser. No tqdm, no default seed, no `self.history`, no ignored arguments in subclass signatures. A solver states which schedules it integrates (`e25bba9` added that guard to the two sigma integrators). As built, `step` takes the denoiser as its last argument, which the first draft of this signature left out: Heun's corrector and RK4's stages evaluate the model again inside one step, and the one-evaluation solvers ignore it. `init` takes the concrete time grid and process beside x_T: the Diffusers-parity solvers whose order depends on where a step is in the walk (the singlestep DPM-Solver's order list, the multistep ones' final-step lowering, UniPC's remaining-step cap, consistency sampling's noiseless last step) read its length there, and a one-step solver ignores it.

### 3.6 Inputs and conditioning

```python
@dataclass(frozen=True)
class Condition:
    encoder: ConditionEncoder     # tokenize on the host, encode on device; params explicit
    field: str = "text"           # batch key
    unconditional: Any = ""

@dataclass(frozen=True)
class InputSpec:                  # replaces DiffusionInputConfig
    sample: Field                 # Field("image", (H, W, C))
    conditions: Mapping[str, Condition] = {}    # keyed by the model's keyword: {"textcontext": ...}
```

Conditions are a named mapping, so `model_key_override` and positional `*all_conditional_inputs` go; `objective.py:86-89` becomes `model.apply(params, x, temb, **conds)`. `ConditionEncoder.params` is explicit and placed by the layout as replicated, which takes the frozen encoder's weights out of the jit constants (review finding 19). Nothing runs a model in `__post_init__`.

### 3.7 Data

```python
@dataclass(frozen=True)
class Dataset:
    train: Callable[[], Iterator[Batch]]        # each iterator has get_state and set_state
    val: Callable[[], Iterator[Batch]] | None
    records: int | None
    batch: int                                  # global
    @property
    def steps_per_epoch(self) -> int | None: ...

from dew.data import TFDS, presets
from dew.data.compile import compile

train, val = TFDS("oxford_flowers102").holdout(split="validation")
data = compile(train.shuffle(0).repeat().pipe(presets.images, 128, batch=32, augment=True),
               validation=val.pipe(presets.images, 128, batch=32), read="memory", provision=True)
```

The chain replaces `datasetMap`, `onlineDatasetMap` and `mediaDatasetMap`: a store and the presets and steps it is piped through ([the data design](data.md)). The per-process batch size is computed in one place that raises on `batch % processes` (done in `f3ea98b`). A stream without `get_state` trains, and its resume is refused unless it declares replay (`replayed`, `from_iter(replay=True)`).

### 3.8 Distributed

```python
@models("causal_transformer")
@logical_axes({
    ("embed_tokens",): ("vocab", "embed"),
    ("q_proj",): ("embed", "heads"), ...
})
class CausalTransformer(nn.Module): ...
```

Same table and the same `logical_axes` and `_mesh_spec` machinery (`training/distributed.py:115-217`), declared on the module it names. Each class keeps its own declarations, and a suffix has one set of axes across the process: a second class that declares it otherwise is refused at the decorator, both named, and a class outside Dew qualifies a one-name suffix with its holder's name (`("head", "readout")`), so it places no other model's `readout`. `Layout` merges the declarations of the modules in the tree; Muon reads them through the same function. Models stay plain Flax modules with no partitioning metadata in `init`, which is the reason the table exists (`distributed.py:57-62`). A test asserts that every parameter of every registered model is declared or explicitly heuristic.

### 3.9 Config and CLI

`RunConfig` stays tyro-driven. `ModelConfig.architecture` resolves through `models`; `preset: presets.union` and `sampler: samplers.union` make tyro subcommands out of the registries. A run writes one `Manifest` (the resolved `RunConfig`, the preset dataclass, the `InputSpec`, the model fields) next to its checkpoints, and `Pipeline.from_run(dir)` rebuilds from it. `parse_config`'s layered `.get` fallbacks (`sampling/loading.py:43-146`) and `serialize_model`'s `__dict__` walk (`checkpoints/utils.py:8-30`) go.

### 3.10 Inference

```python
pipe = inference.TextToImage.from_run("runs/flowers")            # manifest and checkpoint, EMA by default
pipe = inference.TextToImage.from_pretrained("user/flowers-dit")  # hub export, same manifest
images = pipe(["a water lily", "a sunflower"], steps=40, guidance=4.0, sampler=samplers.Heun(), key=key)
text = generate(model, params, prompt, 300, key=key, sampling=Sampling(temperature=0.8)).tokens
```

A loaded model is ready to run. `Pretrained.load` without a mesh or layout places its variables on JAX's
default device, as a `jnp` array is placed, one leaf at a time; with a mesh it places them there.
Keeping them on the host is an explicit request (`on_host=True`), for a caller that places them
itself, such as an export or a run that trains the source. A run that starts from a source reads it where
its start keeps it (`TrainerConfig.source_placement`). A source the run keeps frozen, under an adapter,
loads onto the trainer's mesh, sharded as the layout shards the state, and the start shares those arrays.
A decoder's leaves are read shard by shard; a quantized or packed checkpoint, a tower or a projector is
decoded whole on the host first. A source the run trains stays on the host, since the start copies what
it trains; so does one for a state that lives on the host or keeps its frozen tree whole. A run that
resumes from its own checkpoint reads no weights.

Serving is one engine of kinds (`dew.inference.Engine`). A kind (`Kind`) admits waiting requests
within its own bounds, runs them with one bounded device call a `step`, and settles each request's
`Timed` future. The engine steps each busy kind once a turn from one thread, so a long queue of one
kind never holds another back, and it orders a kind's weight reload with its steps. Two kinds ship:
autoregressive generation's `Server`, a slot scheduler over a resident cache, and `ScoringServer`, a
task answered by forward passes over its requests' rows. A scoring task (`Scoring`) holds no
weights. It lays a request out as rows at submission (`admit`, refusing one alone, a ValueError), says
which rows one pass holds (`fits`), runs a pass over the weights the server hands it (`forward`), and
reads a request's answer off its rows' outputs (`answer`). The server runs one pass a step over the
waiting rows oldest first, so a request whose rows outrun a pass carries the rest to the next turn.
It binds each request to the (task, weights) pair that admitted it, so a reload swaps both at once
for what is submitted after. `Decide.scoring()` gives a decision task as such a task, and
`Decide.systemone` over a list of bodies is served by the same server.

## 4. The three examples in this API

```python
# diffusion
inputs = InputSpec(sample=Field("image", (128, 128, 3)),
                   conditions={"textcontext": Condition(encoders.CLIPText.from_pretrained("openai/clip-vit-large-patch14"))})
objective = DiffusionObjective(models.SimpleDiT(patch_size=4, emb_features=512, num_layers=12, num_heads=8),
                               presets.EDM(regime="pixel"), inputs, sampler=samplers.Heun(), guidance=3.0, steps=40)
state = Trainer(objective, optax.adamw(2e-4), key=jax.random.key(0), mesh=MeshSpec(fsdp=1),
                checkpoints=Checkpoints("runs/flowers")).fit(data, steps=steps)

# language model
objective = LMObjective(models.CausalTransformer(vocab_size=V, emb_features=384, num_layers=6, num_heads=6,
                                                 max_seq_len=556), seq_len=256, samples=TextSamples(prompt, 300))
state = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0), checkpoints=...).fit(data, steps=1200)

# JEPA
objective = JepaObjective(models.JepaEncoder(...), models.JepaPredictor(...), mask=multi_block_mask(grid),
                          sample=Field("image", (224, 224, 3)))
state = Trainer(objective, optax.adamw(1e-3), key=jax.random.key(0), checkpoints=...).fit(
    data, steps=steps, metrics=(metrics.linear_probe(102), metrics.knn_probe(102)))
```

JEPA imports nothing from diffusion.

## 5. What is deleted

`SimpleTrainer`'s W&B and publishing methods, `ObjectiveTrainer`'s diffusion parameters and default objective, `RandomMarkovState`, `get_diffusion_preset`, the `+suffix` DSL and `canonicalize_architecture`, `apply_precision_policy`'s string round trip (dtype and attention_impl become fields the registry validates), `serialize_model`, `parse_config`, `DiffusionInputConfig` and `ConditionalInputConfig`, `model_key_override`, the three dataset maps, `Metrics`, `FLAXDIFF_AUGMENT_MODE` (augmentation becomes a field of the dataset spec), `__encode__` and `__decode__` in favour of `encode` and `decode`.

## 6. Stress test

Four things the design was not written for, in the proposed API, with the friction named.

### 6.1 Masked diffusion LM

```python
class CausalTransformer(...):
    causal: bool = True                  # False: full attention, no cache

@presets("mdlm")
@dataclass(frozen=True)
class MDLM:
    schedule: MaskingSchedule = LogLinear()      # alpha(t): fraction unmasked, in dew.diffusion.discrete
    def __call__(self) -> DiscreteProcess: ...

class MaskedDiffusionObjective(Objective):
    def loss(self, params, batch, step):
        t = self.process.sample_t(step.key, batch["text"].shape[0])
        masked, is_masked = self.process.corrupt(step.key, batch["text"], t)
        hidden = self.model.apply(params, masked, method=type(self.model).hidden_states)
        losses, _ = chunked_cross_entropy(hidden, self.model.head_weight(params["params"]), batch["text"], self.head_chunks)
        return jnp.sum(losses * is_masked * self.process.weight(t)[:, None]) / jnp.sum(is_masked), Aux({})
```

The Gaussian `Process` does not describe this. `dew.diffusion.discrete` gets `MaskingSchedule`, `DiscreteProcess` and an unmasking `Solver`, shaped like the continuous ones. The trainer is untouched; `chunked_cross_entropy` is reused.

### 6.2 MoE training

```python
def loss(self, params, batch, step) -> tuple[jax.Array, Aux]:
    (hidden, sown), _ = self.model.apply(params, tokens, mutable=["router"], method=...)
    ce = ...
    balance = switch_balance_loss(sown["router"])                 # differentiable aux loss
    bias = calculate_load_balance_updates(sown["router"], rate)   # DeepSeek's non-differentiable bias
    return ce + self.balance_weight * balance, Aux(metrics={"ce": ce, "balance": balance}, variables={"moe": bias})
```

`Aux.variables` is what MoE balancing needs; without it the balancing code at `nn/moe.py:51` has no caller, which is review finding 18. The router already keeps its bias in the `moe` collection (`moe.py:142-149`); `nn.sow` carries the loads.

### 6.3 A novel LM

A new mixer drops into `DecoderBlock.mixer`, which already takes `(x, decode, positions, segment_ids)` (`causal_transformer.py:262-287`); a multi-token objective subclasses `LMObjective` and reuses `hidden_states`, `head_weight` and `chunked_cross_entropy`; a per-group optimizer is `optax.multi_transform` labelled by the same `PathFilter`. Nothing at the seams changes.

### 6.4 Mamba

The mixer keeps its recurrent state in the `cache` collection, exactly as the KV cache does (`nn/attention.py:50-94`), so `generate`, `init_cache` and `decode=True` apply unchanged. `positions` is ignored; `segment_ids` becomes a reset mask in the scan. Sharding follows the `@logical_axes` declaration; the state is batch-sharded by construction.

### 6.5 What the design does not do

| Need | Boundary |
| --- | --- |
| Alternating or multi-optimizer updates (GAN, actor-critic) | Not one `loss`. `Trainer(step=make_my_step)` replaces the compiled step and is the one documented place for a custom step. |
| Tensor or sequence parallel | `MeshSpec(tensor=N)` and rules are cheap; correct TP needs `with_sharding_constraint` on activations inside each model. Per-architecture work. |
| Pipeline parallel | Out of scope. |
| Non-grain data | `Dataset.train` is any callable returning an iterator; without `get_state` the run cannot checkpoint its position and `fit` says so. |

## 8. How to build against this design

- A new model is a module with `@models(name)` and `@logical_axes({...})`, and needs nothing else.
- A new modality is an `Objective`. It may add a preset registry and a discrete or continuous process; it never touches the trainer.
- A new sampler is a `Solver` with a `step`. It states which schedules it integrates.
- A new metric is a function `(artifact, batch) -> value` behind `@metrics(name)`.
- A new tracker implements `Tracker` and registers renderers with `singledispatch`.
- A new data preset is a function that takes a dataset, then its settings, and returns a dataset (`dew.data.presets`); a run pipes its store through it. All execution goes through Pipeline and compile.

## 10. What may be condensed

Measured on the tree that landed: 53 constructors take more than eight fields, `CausalTransformer` takes 42, and five of `Router`'s routing options, each already pinned against transformers, could not be named from a config at all. Both of those are worth fixing, and only one of them is fixed by grouping.

The rule is deliberately narrow:

1. Condense what is redundant: a field another field derives, the same dial declared a second time under a second name, the same dials declared again on a second base, and a sentinel value standing in for a feature being off when the rest of that feature's fields mean nothing without it.
2. Cohesion is not a reason. Dials that one model family happens to set together stay separate dials, because the next run wants three of them and not the fourth. A family's convention is not a type.
3. A feature that is off by default becomes `Value | None` only in the sentinel case above, where its other fields depend on it. Then nothing is taken away: every dial is still set by name, inside the value.
4. Every value stays a dict. `dew.registry.from_record` builds a declared value from a record at the build boundary, so `mixture={"experts": 8}` from the command line and `mixture=Mixture(experts=8)` from code agree, mappings and tuples of values are walked, and a run record round-trips without a second serialiser.

Every independent dial stays flat, and that is most of the fields. `qk_norm`, `v_norm`, `attention_bias`, `attention_scale`, `sandwich_norms`, `scale_offset`, `scale_after_cast`, `norm_eps`, `embedding_scale`, `final_logit_softcap` and `tie_embeddings` can each be set on its own and each means something on its own, so each is a field. Grouping them would read tidier, but a user would lose the ability to vary one.

| condensed | why it is redundant |
|---|---|
|`Mixture`: experts, top_k, layers, every, score_function, scaling, groups, groups_per_token, bias|`num_experts: int = 0` is a sentinel for dense, and `top_k`, `moe_every`, `moe_layers` and `expert_bias` mean nothing with no experts. The five routing options nothing can name today come with it, at no width.|
|`LayerKind`: window, rope_theta, head_dim, keyed by the names in the layer pattern|`rope_local_theta` beside `rope_theta` and `global_head_dim` beside `head_dim` are one dial named twice, with the local and global convention carried in the names instead of by the pattern that already names each layer's kind. `sliding_window` is that kind's window.|
|`Loading`: workers, threads, read_buffer, worker_buffer|The same four grain knobs are declared on three unrelated dataset bases.|
|`Wandb`: project, entity, offline|`wandb_entity` and `wandb_offline` mean nothing with no project, and an unset project is today's sentinel for running without a tracker.|

Deleted rather than grouped: `mlp_ratio` beside `mlp_features`, because the intermediate size is the primitive and the ratio restates it; `profile_steps`, which restates one field of the `ProfileWindow` value the trainer already takes, so the config carries the value; `per_layer_input_dim: int = 0`, whose zero becomes None; and `use_double_wide_mlp`, which today does nothing at all unless layers share their keys and values, so it is refused by name in that combination.

Not condensed, on the second reading: `norm_eps` with the three norm-placement switches, and the attention dials with the head geometry. Each is an independent dial, so each stays a field.
