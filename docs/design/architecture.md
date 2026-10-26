# Dew's architecture and shared contracts

Design record, consolidated 2026-10-19. This page owns the cross-cutting code boundaries. [Training](training.md), [data](data.md) and [post-training](post-training.md) own their topic contracts; [testing](testing.md) owns behavioral qualification. The [roadmap](../roadmap.md) lists work still to complete or qualify.

A model is a plain Flax module. An objective defines variables, loss and validation. The trainer runs the objective with an Optax optimizer and owns execution, placement, checkpoints and logging. Sources and transforms compose through Grain. Post-training composes Dew's own objectives, rollouts and session sources.

## One owner per fact

| Fact or operation | Code owner | Contract |
|---|---|---|
| Device capabilities and peaks | `GpuTarget` and `TARGETS`, `src/dew/nn/kernels/generation.py`, interim | Device-specific choices read this table until the device facts read at runtime replace it ([kernel templates](kernel-templates.md#7-choice-the-config-chooser-inside-each-template)). A second hardware table is a second source of truth. |
| Kernel selection and routes | `GpuTarget.kernels`, `Route` and the selection interface in `generation.py`, interim | A layer requests an operation; the selector records the implementation that ran and the reason for any generic route. Kernel modules declare support as data. Family claims and the cost-ranked chooser replace the selector and its tables ([kernel templates](kernel-templates.md#7-choice-the-config-chooser-inside-each-template)). |
| Launch configuration | `src/dew/nn/kernels/tuning.py`, on Tokamax's public API | Each launch takes the config its template derives from device facts (or its heuristic). An opt-in tuner times a few tested variants of it per device and shape and stores picks; a tuner win is a derivation gap. |
| Numerical acceptance | `tests/reference_error.py` | Every numerical consumer uses the [testing contract](testing.md#numerical-accuracy). No local tolerance policy. |
| Model work and MFU | `src/dew/nn/flops.py` and `WorkloadWork`, `src/dew/telemetry/workload.py` | One analytic FLOP formula per operation. MFU uses real, unpadded work over wall time and the measured device's peak. |
| Memory plan | `src/dew/training/memory.py` and `planning.py` | One search chooses a fitting micro-batch, rematerialization and head plan for the model and mesh. |
| Compilation cache | `src/dew/cache.py` | One cache dir and one autotune dir per (GPU model, jaxlib). A program's identity follows its computation and environment. |
| Data execution | `src/dew/data/pipeline.py`, `analysis.py` and `lowering.py` | One `Dataset` chain, one analyzed plan, and the lowering that plan declares. |
| Weight layout | `src/dew/interop/` | One reversible source-to-model binding plan serves loading and export. Family adapters describe a source's schema. |
| Model capabilities | `src/dew/nn/protocols.py` | Objectives, tasks, serving and export ask for capabilities and call through Flax `apply`. |
| Training state and checkpoints | The training-state schema and `src/dew/checkpoints/` | Save and restore derive fields from the state. Casting and placement happen once. |
| Run configuration | `RunConfig.record()`, `src/dew/config/__init__.py` | A record pins every value its run built with. Resume and export read it rather than reconstructing a second configuration. |
| Outcomes and telemetry | `src/dew/telemetry/records.py` | Trackers, display and reports consume the same outcome records. |
| Environments and suite selection | Project extras and lock files; `.armada.json` + `tools/armada/ci.py` | Installers and CI read their package and test selections from these owners. No other package list anywhere. |
| Reinforcement-learning imports | `src/dew/rl` | `dew.rl` imports nothing from `dew.training`. Nothing under `src/dew` outside `src/dew/rl` and `src/dew/objectives/rl` may import `dew.rl`. A test walks the module tree and asserts it. |
| Staged training | `Recipe` and `Stage`, `recipes/chain.py` | Stages compose existing data, objectives and rollouts. `Objective.model_variables` defines what passes to the next stage. |

## Interfaces stay shared

Activations cross every layer-to-kernel boundary physically as `[batch, time, heads, head_dim]`. A kernel adapts its tiling to that interface. Its callers do not carry layout conversions for it. Physical layout checked in optimized HLO (no copies at the edge). Other tensors keep their own meaning.

Coordinates carry validity, positions, document identity and cache extent. Layers use those facts directly. A layer or kernel that receives a contract argument (segments, masks, validity, positions, cache, mesh) either uses it, refuses by name when it is set, or is independent of it by construction (a position-wise MLP needs no segments; say so in one comment). Refuse only where the alternative is a silent wrong result at a trust boundary (user input, files, external APIs). It is never silently dropped. Kernel-route qualification follows every registered model, target, mode and cache layout.

Layer identity survives fusion, scans and pipeline stages. The shared LM head owns logits, loss and scoring, including quantized and sharded forms. Guidance has one denoising interface; checkpoint restoration has one state schema. The guarantees exposed by these interfaces are described in [training's shared seams](training.md#9-the-shared-seams-as-a-user-sees-them).

A new family configures the shared attention, MLP, MoE, normalization, position and quantization modules. A missing capability belongs in the shared module. A source adapter maps source configuration and weight paths; it does not implement a second model. A new objective composes existing math and leaves model placement and execution to the trainer.

## Keep and compose

Start from the maintained implementation's public interface: JAX, Flax, Optax, Orbax, Grain, Tokamax or the system being integrated. Use it directly when it fits. A replacement needs a reason in the direction (not taste), and a parity list: every capability of the old code mapped to the new code or to a drop the user approved, checked before the old code is deleted.

A replacement is complete only when its callers move together. Before 1.0, the API changes directly, without aliases, migrations or parallel implementations.

Tests enumerate implementations from the registry that declares them. A new implementation joins the standing behavioral contracts in the same change. Its siblings are checked against the same contracts. Performance choices need same-session measurements at real shapes, including compilation cost, and [numerical qualification](testing.md#numerical-accuracy).

The [kernel roadmap](../roadmap.md#kernels-and-performance) records the approved template work. `TARGETS`, `GpuTarget.kernels` and the selector are interim: device facts read at runtime, each template family's claims and the cost-ranked chooser replace them ([kernel templates](kernel-templates.md#7-choice-the-config-chooser-inside-each-template)), as one owner each, never beside them.
