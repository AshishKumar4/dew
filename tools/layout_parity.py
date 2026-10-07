#!/usr/bin/env python3
"""Every layout of a model against one device: the loss and each gradient leaf.

Each process builds the one-device reference on its first local device, then
each layout's trainer on the global mesh, from one key and one global batch,
and runs the trainer's own compiled step. The optimizer is `stash` before
adam, so the gradient the step handed the optimizer is kept whole and
compared leaf by leaf.

A layout changes the order of the sums over rows, tokens and split widths
and nothing else, so its error is held to the reference's own deviation
under that kind of change: the batch with its rows in PERMUTATIONS orders,
and pooled from 2 and 4 accumulated slices of consecutive rows and of
strided rows, which is the per-device shapes and partial sums a split batch
or a pipeline's microbatches compute. The floor is the largest deviation per
leaf, and a layout `works` when every leaf is within `FLOOR_FACTOR` of it
and the loss within as much of its own.

A split sequence, a tensor axis and an expert axis also reassociate sums
inside a row (over positions, heads and widths), which no reordering of rows
samples, and an objective that draws per row (a diffusion's noise, a masked
objective's masks) has no reordering to sample at all. With `--anchor` (and
JAX_ENABLE_X64=1), each leaf's floor is at least the reference's own
distance from the same step computed in fp64, the same draws included, so
any reassociation of fp32 sums is held to fp32's own rounding of the step,
while a defect lands orders of magnitude past it; the loss's floor likewise
takes the reference loss's distance from the fp64 one.

For a decoder whose individual gradient error is a coin flip, use
`--rounding-orders --steps 1 --out DIR` with JAX_ENABLE_X64=1. It evaluates
52 exact residual-unit permutations against fp64 and uses
tests/reference_error.py's K-order rule per gradient leaf and for the loss.
Each order's fp64 function invariance is checked; all raw distances and the
expert gate_proj inventory are kept in DIR. Initialization stays at the
default fp32 draw. This multi-draw mode belongs in the Colab queue.

A layout also has to split one device's work rather than repeat it: its
devices' FLOPs together, over the reference's, are at most `flops_bound`, an
even split and a pipeline's bubble, else the layout is REDUNDANT.

An objective that draws noise per row (a DiT's diffusion, a DiffusionGemma
canvas's masks) draws other noise for permuted or pooled rows, so neither is
a reassociation of its step. Its floor is data parallelism over every device
instead, the layout that splits the batch sum and nothing else; the
next-token models check that layout against the permutation floor.

The reference side (the one-device step, its permutation floor and the fp64
anchor) runs on one device while the others wait. With `--references DIR`,
`--prepare` computes each model's reference and anchor into DIR in a job of
one device, and a run of layouts reads them back and computes none; a
missing one is refused, naming `--prepare`. A reference is keyed by all its
numbers depend on: the case, the batch, the steps, the device kind and
backend that ran it, jax, x64 and Dew's source. The fp64 anchor rounds
nowhere a device's kind shows, so it is keyed without one, and any machine
may compute it. A model whose noise is drawn per row takes its floor from
data parallelism over the run's own devices, which the run of layouts
computes.

The same command runs in one process or under `dew launch`, where the global
batch is placed from every process alike:

    python tools/layout_parity.py --models dense --layouts data4,fsdp4,tensor4
    python tools/layout_parity.py --models dense moe --references refs --prepare  # one device
    python tools/layout_parity.py --models dense moe --references refs              # every device
    python tools/layout_parity.py --models moe --layouts expert4,data2_expert2,expert2_fsdp2 \\
        --mixture '{"experts": 32, "top_k": 8, "dispatch": "exchange"}' --objective '{"aux_loss_alpha": 0.01}'
    python tools/layout_parity.py --models dense --layouts stage4,stage2_sequence2 --devices 4
    dew launch --processes-per-host 4 --devices-per-process 1 -- \\
        python tools/layout_parity.py --models dense,dit --out parity.json

A layout's axes other than data are its MeshSpec fields and data takes the
devices left over, so the same layout splits differently on another device
count: `--devices 4` runs it on the process's first four, where four stages
leave no axis for the rows.

Process 0 prints a line per layout and writes the rows. A layout Dew refuses
by design (`dew.nn.sharding.LayoutRefused`: a stage axis over a model that
runs no pipeline) is a row with its reason, listed again at the end, and
passes the run; the exit status is nonzero when a layout mismatches, repeats
work or fails in any other way.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, TypedDict

import tyro

if TYPE_CHECKING:
    from numpy.typing import NDArray

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

PASSING = ("works", "refused")
"""The statuses of a row that pass a run: a layout that matches one device,
and a layout Dew refuses by design, with its reason."""

FLOOR_FACTOR = 4.0
PERMUTATIONS = 16
FLOPS_SLACK = 1.05
"""How far past an even split of one device's FLOPs a layout may compute:
the elementwise work sharding adds, such as collectives' local sums and the
update of parameters a layout keeps whole on several devices. On 4x RTX 3090
every layout of the zoo that repeats no matmul measured within 1.6% of an
even split (the DiT's fsdp2_sequence2 1.016), and a repeated matmul path 8%
and more (the DiT's text projection under fsdp2_tensor2 9%, the dense
decoder's head under fsdp2_tensor2 18%)."""

LAYOUTS: dict[str, dict[str, int]] = {
    "data4": {},
    "fsdp4": {"fsdp": 4},
    "tensor4": {"tensor": 4},
    "expert4": {"expert": 4},
    "sequence4": {"sequence": 4},
    "stage4": {"stage": 4, "microbatches": 4},
    "replicas4": {"replicas": 4},
    "fsdp2_tensor2": {"fsdp": 2, "tensor": 2},
    "replicas2_fsdp2": {"fsdp": 2, "replicas": 2},
    "data2_expert2": {"expert": 2},
    "expert2_fsdp2": {"expert": 2, "fsdp": 2},
    "sequence2": {"sequence": 2},
    "fsdp2_sequence2": {"fsdp": 2, "sequence": 2},
    "tensor2_sequence2": {"tensor": 2, "sequence": 2},
    "stage2_sequence2": {"stage": 2, "sequence": 2, "microbatches": 4},
    "stage2_fsdp2": {"fsdp": 2, "stage": 2, "microbatches": 4},
}
"""Four-device layouts: every axis alone and the combinations worth running."""


def _fixture(name: str) -> dict[str, Any]:
    """The decoder config Dew builds from a model's Hugging Face config."""
    from dew.interop.hf_decoders import translate_config

    config = json.loads((REPO / "tests/fixtures/hf" / name / "config.json").read_text())
    return dict(translate_config(config.get("text_config", config)))


def _nemotron_h_moe() -> dict[str, Any]:
    """The fp32 MoE row declares K-order judgment in its zoo entry.

    At f7f85cef1b8c83a85c74ed93be22928c3d400801 its sequence4 gradient was
    0.720 of the K-order bound over 52 exact residual orders, and expert4
    was 0.596. Float64 invariance drift was at most 1.121e-14; no expert
    gate_proj exists. The two-layout Colab CPU job took 172 seconds. This
    multi-draw comparison belongs in Colab, not on the shared workstation.
    Reproduce with `--models nemotron_h_moe --layouts sequence4 expert4
    --dtype float32 --steps 1 --devices 4 --rounding-orders --out DIR`,
    JAX_PLATFORMS=cpu, JAX_ENABLE_X64=1 and four virtual CPU devices.
    Raw distances live under
    ~/.cache/dew/integration/a906f0111d586f67996850929c23b80343345891/ExcitedRook/
    dew-aux-nemotron-orders-retry-job-1/outputs/out/nemotron_h_orders/;
    the sibling nemotron_h_orders.log records the run.
    """
    config = _fixture('nemotron-h-moe-tiny')
    config.update(vocab_size=512, num_layers=20, max_seq_len=33,
                  layer_types=tuple(config['layer_types']) * 4)
    return config


class TokenRows(TypedDict):
    """The batch every decoder of the zoo trains on."""
    batch_size: int
    seq_len: int
    fsdp_min_param_size: int


class ImageRows(TypedDict):
    """The batch every image model of the zoo trains on."""
    batch_size: int
    channels: int
    fsdp_min_param_size: int


def zoo() -> dict[str, Any]:
    """Small models of every family a layout splits differently: a dense
    decoder, MoE decoders with 8 and with Qwen3-30B-A3B's 128 experts, a
    Mamba-2 hybrid, a DiT and DiffusionGemma, each at widths every layout
    above divides. The DiT's rows hold 256 patches, as a DiT's smallest
    release does (DiT-B/2 at 256 px): a sequence or tensor split repeats each
    row's conditioning on every shard by design, which the FLOPs bound would
    count against a row of only a few patches. The last two MoE models and DiffusionGemma keep their
    released configs' routing and layer kinds. The layers a sequence axis
    splits differently come too: latent attention (MLA), a window (12 rows:
    a sequence split two ways reads it from one neighbour, four ways through
    the exchange), Mamba-2 alone, and packed rows of three documents whose
    boundaries fall inside the shards, in the dense decoder and in Rigel's
    three Mamba-2 layers to one windowed layer.

    The dense, MoE, hybrid and MLA decoders also train every other objective
    a decoder takes: SFT over the assistant's turns (`_sft`), DPO over pairs
    (`_dpo`), GRPO over rollouts with a KL to the frozen reference (`_grpo`)
    and, made bidirectional, masked diffusion (`_mdlm`), which the hybrid's
    Mamba-2 refuses on one device (its recurrence runs one way). The other families
    train their own: a conditional UNet and an MMDiT (SD3's) denoise latents,
    a JEPA encoder predicts its masked patches' features, and a decoder reads
    an image through a vision tower (`multimodal`), each with heads and
    tokens every layout above divides."""
    from benchmark_cases import Case

    dense: dict[str, object] = {"vocab_size": 512, "emb_features": 64, "num_layers": 4, "num_heads": 8,
             "num_kv_heads": 4, "head_dim": 8, "mlp_features": 128, "max_seq_len": 33}
    moe = {**dense, "mixture": {"experts": 8, "top_k": 2, "expert_features": 32,
                                "layers": (0, 1, 2, 3)}}
    moe128 = {**dense, "mixture": {**_fixture("qwen3-30b-a3b")["mixture"], "expert_features": 32,
                                   "layers": (0, 1, 2, 3)}}
    # Four periods of five sliding layers and one global one, so each stage
    # of two or four holds whole periods; the trunk is the causal encoder
    # view DiffusionGemma clones its bidirectional decoder from.
    released = _fixture("diffusiongemma-26b")
    dgemma = {**released, "vocab_size": 512, "emb_features": 64, "num_heads": 4,
              "num_kv_heads": 2, "head_dim": 16, "mlp_features": 64, "num_layers": 24,
              "layer_types": tuple(released["layer_types"][:6]) * 4, "max_seq_len": 32,
              "layer_scalar": "frozen", "causal": True,
              "kinds": {"sliding_attention": {"window": 8, "rope_theta": 10000.0},
                        "full_attention": {"head_dim": 32, "num_kv_heads": 1}},
              "mixture": {**released["mixture"], "experts": 8, "top_k": 2, "expert_features": 32}}
    mamba = {"mixer": {"class": "mamba2", "fields": {"num_heads": 4, "head_dim": 16, "state_size": 8,
                       "n_groups": 1, "chunk_size": 8}}}
    hybrid = {**dense, "layer_types": ("mamba", "attention") * 2,
              "kinds": {"mamba": mamba, "attention": {}}}
    nemotron_h = _fixture("nemotron-h-tiny")
    nemotron_h.update(vocab_size=512, num_layers=20, max_seq_len=33,
                     layer_types=tuple(nemotron_h["layer_types"]) * 4)
    window = {**dense, "layer_types": ("sliding",) * 4, "kinds": {"sliding": {"window": 12}}}
    mla = {**dense, "mixer": {"class": "mla", "fields": {"q_lora_rank": 48, "kv_lora_rank": 32, "qk_nope_head_dim": 16,
                              "qk_rope_head_dim": 8, "v_head_dim": 8}}}
    mamba2 = {**dense, "layer_types": ("mamba",) * 4, "kinds": {"mamba": mamba}}
    rigel = {**dense, "layer_types": ("mamba",) * 3 + ("sliding",),
             "kinds": {"mamba": mamba, "sliding": {"window": 12}}}
    dit: dict[str, object] = {"patch_size": 2, "emb_features": 64, "num_layers": 4, "num_heads": 4, "mlp_ratio": 2,
           "output_channels": 4}
    # Sixteen rows: a pipeline's microbatch has to take a share of every
    # device's rows, and stage2_fsdp2 on eight devices splits the rows four
    # ways into four microbatches.
    lm: TokenRows = {"batch_size": 16, "seq_len": 32, "fsdp_min_param_size": 256}
    decoders = {"dense": dense, "moe": moe, "hybrid": hybrid, "mla": mla}
    trained = {f"{name}_{kind}": Case("causal_transformer", config, decoder_objective=kind,
                                      objective={"beta": 0.01} if kind == "grpo" else {}, **lm)
               for name, config in decoders.items() for kind in ("sft", "dpo", "grpo")}
    # Mamba-2's recurrence has no bidirectional mode, which the hybrid's
    # model refuses on one device; masked diffusion takes the others.
    def bidirectional(config: Mapping[str, object]) -> dict[str, object]:
        return {**config, "causal": False}

    diffused = {f"{name}_mdlm": Case("causal_transformer", bidirectional(config),
                                     decoder_objective="mdlm", **lm)
                for name, config in decoders.items() if name != "hybrid"}
    images: ImageRows = {"batch_size": 8, "channels": 4, "fsdp_min_param_size": 256}
    return {
        "dense": Case("causal_transformer", dense, **lm),
        "moe": Case("causal_transformer", moe, **lm),
        "moe128": Case("causal_transformer", moe128, **lm),
        "hybrid": Case("causal_transformer", hybrid, **lm),
        "nemotron_h": Case("causal_transformer", nemotron_h, **lm),
        "nemotron_h_moe": Case("causal_transformer", _nemotron_h_moe(), orders=True, **lm),
        "window": Case("causal_transformer", window, **lm),
        "mla": Case("causal_transformer", mla, **lm),
        "mamba2": Case("causal_transformer", mamba2, **lm),
        "dense_packed": Case("causal_transformer", dense, packed_documents=3, **lm),
        "rigel_packed": Case("causal_transformer", rigel, packed_documents=3, **lm),
        "dit": Case("simple_dit", dit, batch_size=8, image_size=32, channels=4,
                    fsdp_min_param_size=256),
        "dgemma": Case("diffusion_gemma", dgemma, canvas={"prompt_length": 16, "canvas_size": 8},
                       batch_size=8, seq_len=31, fsdp_min_param_size=256),
        **trained,
        **diffused,
        "unet": Case("unet_2d_condition", {"stages": [{"features": 32, "heads": 4},
                                                      {"features": 64, "heads": 4}],
                                           "blocks_per_level": 1, "in_channels": 4, "out_channels": 4},
                     image_size=16, **images),
        "mmdit": Case("sd3_transformer", {"in_channels": 4, "out_channels": 4, "num_layers": 2, "heads": 4,
                                          "head_dim": 8, "joint_attention_dim": 16,
                                          "caption_projection_dim": 32, "pooled_projection_dim": 32,
                                          "sample_size": 16, "pos_embed_max_size": 8},
                      image_size=16, **images),
        "jepa": Case("jepa_encoder", {"patch_size": 4, "emb_features": 32, "num_layers": 2, "num_heads": 4,
                                      "mlp_ratio": 2},
                     predictor={"grid": (4, 4), "emb_features": 32, "predictor_features": 16,
                                "num_layers": 1, "num_heads": 4, "mlp_ratio": 2},
                     batch_size=8, image_size=16, fsdp_min_param_size=256),
        "multimodal": Case("multimodal_transformer", dense, media={
            "family": "gemma3", "image_token_id": 511, "images": 1, "pixels": [3, 16, 16],
            "tower": {"class": "siglip", "fields": {"hidden_size": 32, "intermediate_size": 64, "num_layers": 1,
                      "num_heads": 4, "image_size": 16, "patch_size": 8}},
            "projector": {"class": "gemma", "fields": {"text_width": 64, "patches_per_side": 2,
                          "tokens_per_side": 2}}}, **lm),
    }


def reassociates(case) -> bool:
    """Whether reordering or pooling the batch's rows only reassociates the
    step's sums: a next-token loss draws nothing per row (SFT, DPO and GRPO
    among them), a diffusion objective or a masked-diffusion decoder draws
    its noise by row, and JEPA its masks."""
    return case.is_lm and case.canvas is None and case.decoder_objective != "mdlm"


def stash():
    """Pass the updates on unchanged and keep them as the optimizer state."""
    import jax
    import optax

    return optax.GradientTransformation(
        lambda params: {"gradient": jax.tree.map(jax.numpy.zeros_like, params)},
        lambda updates, state, params=None: (updates, {"gradient": updates}))


def placed(batch, mesh):
    """The global batch on `mesh`, the same on every process."""
    import jax
    import numpy as np

    from dew.training.distributed import batch_shardings

    def put(leaf, sharding):
        array = np.asarray(leaf)
        return jax.make_array_from_callback(array.shape, sharding, lambda index: array[index])
    return jax.tree.map(put, batch, batch_shardings(mesh, batch))


def reordered(batch, seed: int):
    """The batch with its rows reversed (seed 0) or drawn in another order."""
    import jax
    import numpy as np

    rows = len(jax.tree.leaves(batch)[0])
    order = np.arange(rows)[::-1] if seed == 0 else np.random.default_rng(seed).permutation(rows)
    return jax.tree.map(lambda leaf: np.asarray(leaf)[order], batch)


def _drawn(params, key):
    """`params` with every leaf its initializer left all zeros drawn from a
    normal of std 0.02 instead, each from its own fold of `key`.

    A DiT zero-initializes its output projection and its modulations, so its
    first step's gradient reaches those layers alone: 2 of the zoo DiT's 70
    leaves, and a layout that split anything above them wrongly would pass.
    Every state of a comparison, the reference's, a layout's, a floor's and
    the fp64 anchor's, comes through here from one key, so they start alike."""
    import jax
    import jax.numpy as jnp

    def draw(index, leaf):
        if not jnp.issubdtype(leaf.dtype, jnp.floating):
            return leaf
        drawn = (0.02 * jax.random.normal(jax.random.fold_in(key, index), leaf.shape)).astype(leaf.dtype)
        return jnp.where(jnp.any(leaf != 0), leaf, drawn)

    leaves, tree = jax.tree.flatten(params)
    return jax.tree.unflatten(tree, [draw(index, leaf) for index, leaf in enumerate(leaves)])


def _trainer(case, fields: dict[str, int], *, one_device: bool = False, accumulation: int = 1,
             devices: int | None = None):
    """The trainer of `case` on the layout `fields` names over the first
    `devices` devices (every device by default), or on this process's first
    device, stashing each gradient the optimizer is handed. Its initial
    parameters have no all-zero leaf (`_drawn`). In a one-process run, once
    the case's one-device state is computed (`_initial`), a trainer whose
    mesh shards the variables in a way the case has not yet drawn them
    computes its own state and must find it bitwise equal to that one, and
    every other trainer copies it: the same state, without a compile of the
    model's initialization on every layout."""
    import benchmark_models as bench
    import jax
    import numpy as np
    import optax

    from dew.training import Layout, MeshSpec, Trainer
    from dew.training.trainer import refuse_wide_floats

    class Drawn(Trainer):
        def initial_state(self, initializer=None, key=None):
            state = super().initial_state(initializer, key)
            params = {**state.variables, "params": _drawn(state.variables["params"], jax.random.key(7))}
            return dataclasses.replace(state, variables=params, opt_state=self.optimizer.init(params["params"]))

        def place(self):
            held = _initial.get((repr(case), accumulation)) if jax.process_count() == 1 else None
            # Copies every time, in and out: the step donates the state it is handed.
            if held is None:
                state, shardings, position = super().place()
                if one_device:
                    _initial[(repr(case), accumulation)] = jax.tree.map(lambda leaf: leaf.copy(), state)
                return state, shardings, position
            abstract = jax.tree.map(lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), held)
            refuse_wide_floats(abstract, self.device_mesh)
            shardings = self.shardings(abstract)
            self.layout.check(abstract.variables, shardings.variables, self.device_mesh)
            # The mesh axes the variables split over, with their sizes: an
            # initializer drawn under a split the case has not drawn under
            # yet could come out different, so that split draws its own.
            split = tuple(sorted({(axis, self.device_mesh.shape[axis])
                                  for sharding in jax.tree.leaves(shardings.variables)
                                  for entry in sharding.spec if entry is not None
                                  for axis in ((entry,) if isinstance(entry, str) else entry)
                                  if self.device_mesh.shape[axis] > 1}))
            if split and (repr(case), split) not in _drawn_splits:
                state, shardings, position = super().place()
                drawn, _ = jax.tree_util.tree_flatten_with_path(jax.device_get(state.variables))
                wanted = jax.tree.leaves(jax.device_get(held.variables))
                differing = [jax.tree_util.keystr(path)
                             for (path, here), there in zip(drawn, wanted, strict=True)
                             if np.asarray(here).tobytes() != np.asarray(there).tobytes()]
                if differing:
                    raise ValueError(f"the variables drawn split over {split} differ from one device's "
                                     f"at {differing[:4]}")
                _drawn_splits.add((repr(case), split))
                return state, shardings, position
            return jax.device_put(jax.tree.map(lambda leaf: leaf.copy(), held), shardings), shardings, None

    trainer = Drawn(bench.build_objective(case), optax.chain(stash(), optax.adam(1e-3)),
                      key=jax.random.key(0), mesh=bench.mesh_spec(fields),
                      layout=Layout(min_shard=case.fsdp_min_param_size, tolerance=1.0),
                      accumulation=accumulation, checkpoints=None, tracker=None)
    if one_device:
        trainer.device_mesh = MeshSpec().build([jax.local_devices()[0]])
    elif devices is not None:
        trainer.device_mesh = trainer.mesh.build(jax.devices()[:devices])
    return trainer


_initial: dict[tuple[str, int], Any] = {}
"""Each case's initial state by its accumulation window, as the first
one-device trainer placed it on this process's first device: every layout's
trainer draws the same state from the same key, so later trainers copy it
onto their mesh."""
_drawn_splits: set[tuple[str, tuple[tuple[str, int], ...]]] = set()
"""Each case's variable splits whose own draw was found equal to `_initial`'s."""


def _gradient(state) -> dict[str, NDArray]:
    """The gradient the optimizer was handed, whole on every process, by leaf
    name. A one-device reference's leaves are already whole where they are,
    and gathering those would stack every process's copy."""
    import jax
    import numpy as np
    from jax.experimental import multihost_utils

    return {jax.tree_util.keystr(path): np.asarray(
        leaf if leaf.is_fully_addressable else multihost_utils.process_allgather(leaf, tiled=True))
        for path, leaf in jax.tree_util.tree_flatten_with_path(state.opt_state[0]["gradient"])[0]}


def compiled_step(case, fields: dict[str, int], batch, *, one_device: bool = False,
                  devices: int | None = None):
    """The trainer's step and its inputs, reusable across exact parameter orders."""
    trainer = _trainer(case, fields, one_device=one_device, devices=devices)
    state, _, _ = trainer.place()
    data = placed(batch, trainer.device_mesh)
    step = trainer.compile(state, data)
    compiled = {"flops_per_device": trainer.flops_per_step,
                "mesh": {axis: int(size) for axis, size in trainer.device_mesh.shape.items()}}
    return state, data, step, compiled


def trained(case, fields: dict[str, int], batch, *, steps: int, one_device: bool = False,
            devices: int | None = None) -> tuple[list[float], dict[str, NDArray], dict[str, Any]]:
    """The losses of `steps` steps on the layout `fields` names (over the
    first `devices` devices, every device by default), step one's gradient
    gathered whole, and what the compiler says of the step."""
    state, data, step, compiled = compiled_step(case, fields, batch, one_device=one_device, devices=devices)
    losses, gradient = [], {}
    for _ in range(steps):
        state, loss, _, _, _ = step(state, data)
        losses.append(float(loss))
        gradient = gradient or _gradient(state)
    return losses, gradient, compiled


def pooled(case, batches, pieces: int) -> list[tuple[float, dict[str, NDArray]]]:
    """Step one's loss and gradient on one device for each batch, each pooled
    from `pieces` consecutive slices of its rows by accumulation, from one
    initial state through one compiled step."""
    import jax
    import numpy as np

    trainer = _trainer(case, {}, one_device=True, accumulation=pieces)
    step, results = None, []
    for batch in batches:
        state, _, _ = trainer.place()
        rows = len(jax.tree.leaves(batch)[0]) // pieces
        slices = [placed(jax.tree.map(lambda leaf: np.asarray(leaf)[i * rows:(i + 1) * rows], batch),
                         trainer.device_mesh) for i in range(pieces)]
        step = trainer.compile(state, slices[0]) if step is None else step
        losses = []
        for data in slices:
            state, loss, _, _, _ = step(state, data)
            losses.append(float(loss))
        results.append((float(np.mean(losses)), _gradient(state)))
    return results


def leaf_errors(reference: Mapping[str, NDArray], other: Mapping[str, NDArray],
                dtype: str) -> dict[str, float]:
    """Each leaf's L2 distance from the reference over the reference leaf's
    norm, or over the compute dtype's rounding of the whole gradient's norm
    where the leaf is smaller than that. Both hold the same leaves by name.

    A leaf's reassociation error scales with its terms, not with their sum,
    and a leaf whose terms cancel has a gradient below the rounding of the
    step: a scale just ahead of a normalisation that undoes it, as a
    decoder's last layer scalar is ahead of the final RMSNorm. Relative to
    its own norm that is noise over noise (DiffusionGemma's read 2.2), so it
    is measured against what rounding the whole step moves instead; every
    other leaf is measured against itself.
    """
    import numpy as np

    if reference.keys() != other.keys():
        raise ValueError(f"the gradients hold different leaves: {sorted(reference.keys() ^ other.keys())[:6]}")
    pairs = [(name, np.asarray(want, np.float64), np.asarray(other[name], np.float64))
             for name, want in reference.items()]
    noise = rounding_limit(dtype) * float(np.sqrt(sum(np.sum(want ** 2) for _, want, _ in pairs)))
    return {name: float(np.linalg.norm(got - want) / max(float(np.linalg.norm(want)), noise))
            for name, want, got in pairs}


def strided(batch, pieces: int):
    """The batch with rows m, m + pieces, m + 2 * pieces, ... gathered into
    its m-th consecutive slice: a pipeline's microbatch m."""
    import jax
    import numpy as np

    rows = len(jax.tree.leaves(batch)[0])
    order = np.concatenate([np.arange(start, rows, pieces) for start in range(pieces)])
    return jax.tree.map(lambda leaf: np.asarray(leaf)[order], batch)


def data_parallel_floor(case, batch, reference: Mapping[str, NDArray], reference_loss: float,
                        devices: int | None = None) -> tuple[dict[str, float], float]:
    """Per leaf, the deviation from the reference of data parallelism over
    every device of the run (its first `devices`), the layout that splits
    the batch sum and nothing else, and the same of the step-one loss: the
    floor of an objective that draws noise per row, which no reordering of
    rows keeps."""
    losses, gradient, _ = trained(case, {}, batch, steps=1, devices=devices)
    return leaf_errors(reference, gradient, case.dtype), abs(losses[0] - reference_loss)


def permutation_floor(case, batch, reference: Mapping[str, NDArray],
                      reference_loss: float) -> tuple[dict[str, float], float]:
    """Per leaf, the largest deviation of the reference from itself under a
    reassociation of the batch's sums, and the same of the step-one loss,
    on the reference's own device."""
    runs = pooled(case, [reordered(batch, seed) for seed in range(PERMUTATIONS)], 1)
    loss = max(abs(moved - reference_loss) for moved, _ in runs)
    for pieces in (2, 4):
        runs += pooled(case, [batch, strided(batch, pieces)], pieces)
    leaves: dict[str, float] = {}
    for _, gradient in runs:
        for leaf, error in leaf_errors(reference, gradient, case.dtype).items():
            leaves[leaf] = max(leaves.get(leaf, 0.0), error)
    return leaves, loss


def anchor_program(case, batch, state):
    """The same fp64 loss/gradient executable reused for each residual order."""
    import benchmark_models as bench
    import jax
    import jax.numpy as jnp

    from dew.objectives.base import Step
    from dew.training.transaction import with_ema

    def widened(tree):
        return jax.tree.map(lambda leaf: leaf.astype(jnp.float64)
                            if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, tree)

    wide = widened(state.variables)
    objective = bench.build_objective(case, widened=True)
    # The trainer's first step: its key folded with the step, which draws a
    # diffusion objective's noise and a masked one's masks, and the frozen
    # EMA DPO and GRPO score against.
    step = Step(state.microstep, jax.random.fold_in(state.key, state.step),
                with_ema(wide, None if state.ema is None else widened(state.ema)))

    def loss(variables, batch):
        return objective.scalar_loss({**wide, "params": variables}, batch, step)[0]

    return jax.jit(jax.value_and_grad(loss)).lower(wide["params"], batch).compile()


def anchor_step(case, batch) -> tuple[float, dict[str, NDArray]]:
    """Step one's loss and its gradient by leaf name in fp64: the model's
    float64 twin (`benchmark_models.build_objective(widened=True)`), which computes in float64
    throughout, on the reference's own initial variables widened to fp64.
    The trainer draws those from its key's first split, so they come from
    the trainer, not from the objective's `init` on the key itself."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    state, _, _ = _trainer(case, {}, one_device=True).place()
    wide = jax.tree.map(lambda leaf: leaf.astype(jnp.float64)
                        if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, state.variables["params"])
    value, gradient = anchor_program(case, batch, state)(wide, batch)
    return float(value), {jax.tree_util.keystr(path): np.asarray(leaf)
                          for path, leaf in jax.tree_util.tree_flatten_with_path(gradient)[0]}


@dataclasses.dataclass(frozen=True)
class Reference:
    """A model's step on one device, which its layouts are judged against:
    the step losses, step one's gradient by leaf name and the FLOPs of the
    device, and, where reordering the batch only reassociates the step, each
    leaf's floor and the loss's (`permutation_floor`)."""

    losses: list[float]
    gradient: dict[str, NDArray]
    flops_per_device: float | None
    floors: dict[str, float] | None
    loss_floor: float | None


def computed_reference(case, batch, steps: int) -> Reference:
    """The reference of `case`, computed on this process's first device."""
    losses, gradient, compiled = trained(case, {}, batch, steps=steps, one_device=True)
    floors, loss_floor = (permutation_floor(case, batch, gradient, losses[0]) if reassociates(case)
                          else (None, None))
    return Reference(losses, gradient, compiled["flops_per_device"], floors, loss_floor)


@functools.cache
def source_digest() -> str:
    """The Dew this process imported and the tools that build and judge a
    reference, wherever each was imported from."""
    import benchmark_cases
    import benchmark_models
    import benchmark_step

    import dew

    package = Path(dew.__file__).parent
    files = [(str(path.relative_to(package)), path) for path in sorted(package.glob("**/*.py"))]
    files += [(f"tools/{Path(tool).name}", Path(tool)) for tool in (
        benchmark_step.__file__, __file__, benchmark_cases.__file__, benchmark_models.__file__)]
    digest = hashlib.sha256()
    for name, path in files:
        digest.update(name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _cache_name(kind: str, case, batch, **parts: object) -> tuple[str, dict[str, Any]]:
    """The file a cached `kind` of `case` is kept in, and what it depends on."""
    import importlib.metadata

    import jax
    import numpy as np

    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(batch):
        array = np.ascontiguousarray(leaf)
        digest.update(f"{array.dtype}{array.shape}".encode())
        digest.update(array.tobytes())
    inputs = {"case": dataclasses.asdict(case), "batch": digest.hexdigest(), "jax": jax.__version__,
              "jaxlib": importlib.metadata.version("jaxlib"), "x64": jax.config.jax_enable_x64,
              "source": source_digest(), **parts}
    name = hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()[:24]
    return f"{kind}-{name}.npz", inputs


def _write(path: Path, arrays: Mapping[str, NDArray], meta: dict[str, Any]) -> None:
    """`arrays`, widened to fp64 (exactly), and `meta` in one file, written
    whole or not at all."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}")
    with partial.open("wb") as file:
        # Stored as arr_0 (the metadata), arr_1, ... in the names' order.
        np.savez(file, np.array(json.dumps({**meta, "names": list(arrays)})),
                 *[np.asarray(array, np.float64) for array in arrays.values()])
    partial.replace(path)


def _read(path: Path) -> tuple[dict[str, NDArray], dict[str, Any]]:
    import numpy as np

    with np.load(path) as stored:
        meta = json.loads(str(stored["arr_0"]))
        return {name: stored[f"arr_{index}"] for index, name in enumerate(meta["names"], start=1)}, meta


@dataclasses.dataclass(frozen=True)
class References:
    """Where a model's reference and fp64 anchor come from: computed here,
    or, with a `directory`, read from the files a `prepare` run wrote there;
    see the module docstring."""

    directory: Path | None = None
    prepare: bool = False

    def reference(self, case, batch, steps: int) -> Reference:
        if self.directory is None:
            return computed_reference(case, batch, steps)
        import jax
        from jax.extend import backend

        path, inputs = self._path("reference", case, batch, steps=steps,
                                  device=jax.local_devices()[0].device_kind,
                                  backend=backend.get_backend().platform_version)
        if path.exists():
            gradient, meta = _read(path)
            return Reference(meta["losses"], gradient, meta["flops_per_device"], meta["floors"],
                             meta["loss_floor"])
        computed = computed_reference(case, batch, steps)
        _write(path, computed.gradient, {
            "inputs": inputs, "losses": computed.losses, "flops_per_device": computed.flops_per_device,
            "floors": computed.floors, "loss_floor": computed.loss_floor})
        return computed

    def anchor(self, case, batch) -> tuple[float, dict[str, NDArray]]:
        if self.directory is None:
            return anchor_step(case, batch)
        path, inputs = self._path("anchor", case, batch)
        if path.exists():
            gradient, meta = _read(path)
            return meta["loss"], gradient
        loss, gradient = anchor_step(case, batch)
        _write(path, gradient, {"inputs": inputs, "loss": loss})
        return loss, gradient

    def _path(self, kind: str, case, batch, **parts: object) -> tuple[Path, dict[str, Any]]:
        """The cached `kind`'s file, refused where a run of layouts would
        have to compute it."""
        assert self.directory is not None
        name, inputs = _cache_name(kind, case, batch, **parts)
        path = self.directory / name
        if not self.prepare and not path.exists():
            raise FileNotFoundError(
                f"no {kind} of this case in {self.directory} ({name}); compute it first with "
                f"--prepare in a job of one device")
        return path, inputs


def rounding_limit(dtype: str) -> float:
    """The farthest the reference may sit from the fp64 step, and the widest
    a floor may be, and still be the compute dtype's rounding of the step:
    the square root of its machine epsilon, where a reassociation has cost
    half the significand (3.5e-4 for fp32, 8.8e-2 for bf16). Past it, two
    runs compute different steps, which no floor may absorb."""
    import jax.numpy as jnp

    return float(jnp.finfo(dtype).eps) ** 0.5


def flops_bound(fields: Mapping[str, int]) -> float:
    """The FLOPs a layout's devices may compute together over one device's:
    an even split, and a pipeline of S stages and M microbatches holds each
    stage's work for M + S - 1 microbatches' time (GPipe's bubble), times
    FLOPS_SLACK."""
    stages, microbatches = fields.get("stage", 1), fields.get("microbatches", 1)
    return (microbatches + stages - 1) / microbatches * FLOPS_SLACK


def widest_floor(floors: dict[str, float], dtype: str) -> str:
    """The leaf with the widest floor, refused past `rounding_limit`: a floor
    that wide passes any layout at its leaf, and where the floor is data
    parallelism's own deviation, it would pass that layout's own defect."""
    widest = max(floors, key=floors.__getitem__)
    limit = rounding_limit(dtype)
    if floors[widest] > limit:
        raise ValueError(
            f"the floor at {widest} is {floors[widest]:.2e}, past {dtype} rounding "
            f"({limit:.1e}), so no layout of this model can be judged by it")
    return widest


def contraction_floor(terms: int, magnitude: float) -> float:
    """How far two fp32 sums of the same `terms` products may land apart,
    whose magnitudes add to `magnitude`: each is within gamma_N times it of
    the exact sum in any order (one product and at most N - 1 adds a path),
    gamma_N = N u / (1 - N u) with u fp32's unit roundoff, so two are within
    twice that. tests/test_packed_grpo.py bounds its fp64 contraction alike."""
    import numpy as np

    u = float(np.finfo(np.float32).eps) / 2
    return 2 * terms * u / (1 - terms * u) * magnitude


def contracted_terms(batch) -> int:
    """The terms a step's loss sums, at most: the elements of the batch's
    largest leaf, a decoder's tokens and a denoiser's pixels."""
    import jax
    import numpy as np

    return max(int(np.size(leaf)) for leaf in jax.tree.leaves(batch))


def judged(errors: dict[str, float], floors: dict[str, float], loss: float,
           loss_floor: float, reference_loss: float, terms: int) -> dict[str, Any]:
    """Every leaf against FLOOR_FACTOR times its floor, fp32 epsilon at
    least, and the loss against FLOOR_FACTOR times its floor or the
    `contraction_floor` of its `terms`, whichever is wider.

    The loss floor comes from reordering the batch, which reassociates the
    sum over rows; a layout that splits a row's own contraction, a sequence
    or a tensor axis over the output, reassociates the sum over its elements
    too, which no reordering of rows moves. The magnitude is |loss|, which is
    the sum of the terms' magnitudes for a loss of nonnegative terms (cross
    entropy, squared error) and a lower estimate of it for a signed one (an
    advantage-weighted GRPO loss); the floor only ever widens the bound, so
    the estimate errs toward judging more strictly, never less."""
    import numpy as np

    eps = float(np.finfo(np.float32).eps)
    ratios = {leaf: error / (FLOOR_FACTOR * max(floors[leaf], eps)) for leaf, error in errors.items()}
    worst = max(ratios, key=ratios.__getitem__)
    loss_bound = max(FLOOR_FACTOR * max(loss_floor, eps * abs(reference_loss)),
                     contraction_floor(terms, abs(reference_loss)))
    return {"worst_leaf": worst, "worst_leaf_error": errors[worst], "worst_leaf_floor": floors[worst],
            "worst_ratio": ratios[worst], "loss_error": loss, "loss_bound": loss_bound,
            "status": "works" if ratios[worst] <= 1.0 and loss <= loss_bound else "MISMATCH"}


def judged_orders(reference: Mapping[str, Sequence[float]], layout: Mapping[str, Sequence[float]]) -> dict:
    """K-order RMS distances to fp64, held to the shared reference rule per leaf.

    ORDERS=52 is the smallest K for which F(K,K) exceeds FACTOR² with
    probability at most FALSE_FAILURE=1e-6, even when a squared error has
    only one degree of freedom. tests/reference_error.py owns that derivation
    and assertion; row-permutation floors and picked leaf tolerances play no
    part in this comparison. Zero-error leaves pass only at zero layout error.
    """
    import numpy as np

    sys.path.insert(0, str(REPO / "tests"))
    from reference_error import FACTOR, assert_as_exact_over_orders

    if reference.keys() != layout.keys():
        raise ValueError('ordered gradients hold different leaves')
    failed, ratios = {}, {}
    for leaf in reference:
        mine, theirs = (float(np.sqrt(np.mean(np.square(values))))
                        for values in (layout[leaf], reference[leaf]))
        ratios[leaf] = mine / (FACTOR * theirs) if theirs else 0.0 if mine == 0 else float('inf')
        try:
            assert_as_exact_over_orders(layout[leaf], reference[leaf], leaf)
        except AssertionError as error:
            failed[leaf] = str(error)
    worst = max(ratios, key=ratios.__getitem__)
    return {"worst_leaf": worst, "worst_ratio": ratios[worst], "ratios": ratios,
            "failed_leaves": failed, "status": "MISMATCH" if failed else "works"}


def order_invariance(actual: Mapping[str, NDArray], truth: Mapping[str, NDArray],
                     loss: float, truth_loss: float) -> float:
    """A residual permutation preserves the entire fp64 gradient and loss to 1e-12 relative.

    The gradient is one vector over all parameters, so cancelled leaves are
    measured on its scale instead of being divided by their own near-zero norm.
    This is the exact-symmetry check required before an order counts as a
    rounding draw; it is independent of the fp32 K-order acceptance bound.
    """
    import numpy as np

    if actual.keys() != truth.keys():
        raise ValueError('float64 orders hold different leaves')
    apart = math.sqrt(sum(float(np.sum(np.square(actual[name] - truth[name]))) for name in truth))
    scale = math.sqrt(sum(float(np.sum(np.square(values))) for values in truth.values()))
    relative = apart / scale if scale else 0.0 if apart == 0 else float('inf')
    if relative > 1e-12 or abs(loss - truth_loss) > 1e-12 * abs(truth_loss):
        raise ValueError(
            f'float64 residual order changes the function: gradient relative error {relative:.3e}')
    return relative


def run_orders(models: Sequence[str], layouts: Sequence[str], *, dtype: str, devices: int | None,
               mixture: dict[str, Any], objective: dict[str, Any], out: Path,
               speak: Callable[[str], None]) -> list[dict[str, Any]]:
    """One compiled step per exact residual order, with a shared fp64 oracle.

    Initialization and native executables are built with x64 disabled, so
    enabling the oracle cannot change the original fp32 draw. Every native
    executable is reused for all ORDERS, and every input state is copied
    because the trainer donates it. This is a multi-draw validation job for
    the Colab queue; it must not be run on the shared workstation.
    """
    import benchmark_models as bench
    import jax
    import jax.numpy as jnp
    import numpy as np

    sys.path.insert(0, str(REPO / "tests"))
    from reference_error import ORDERS, distance
    from residual_orders import orders, permuted, residual_width

    if jax.process_count() != 1:
        raise ValueError('residual-order comparisons run in one process')
    if not jax.config.jax_enable_x64:
        raise ValueError('residual-order comparisons need JAX_ENABLE_X64=1')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in models:
        case, reference = model_case(model, dtype, mixture, objective)
        if not case.is_lm or case.architecture != 'causal_transformer':
            raise ValueError('residual orders require a CausalTransformer decoder')
        batch = bench.global_batch(case)
        with jax.enable_x64(new_val=False):
            base, ref_data, ref_step, ref_compiled = compiled_step(reference, {}, batch, one_device=True)
            programs = {name: compiled_step(case, LAYOUTS[name], batch, devices=devices) for name in layouts}
        truth_step = anchor_program(reference, batch, base)
        drawn = orders(residual_width(base.variables), ORDERS, seed=2024)
        ref_shardings = jax.tree.map(lambda leaf: leaf.sharding, base)
        model_out = out / model
        model_out.mkdir(exist_ok=True)
        (model_out / 'metadata.json').write_text(json.dumps({
            'model': model, 'dtype': dtype, 'orders': ORDERS, 'seed': 2024,
            'layouts': list(layouts), 'device_kind': jax.devices()[0].device_kind,
            'source': source_digest(), 'x64_initialization': False,
        }, indent=1))
        np.save(model_out / 'orders.npy', drawn)
        ref_distances, layout_distances = {}, {name: {} for name in layouts}
        invariance, identity, identity_loss = [], None, None

        def flat(tree, order):
            canonical = permuted({'params': jax.device_get(tree)}, np.argsort(order))['params']
            return {jax.tree_util.keystr(path): np.asarray(leaf)
                    for path, leaf in jax.tree_util.tree_flatten_with_path(canonical)[0]}

        def ordered_state(template, order, shardings):
            copied = jax.tree.map(lambda leaf: leaf.copy(), template)
            copied = dataclasses.replace(copied, variables=permuted(copied.variables, order))
            return jax.device_put(copied, shardings)

        for index, order in enumerate(drawn):
            variables = permuted(base.variables, order)
            params = jax.tree.map(lambda leaf: jnp.asarray(leaf, jnp.float64), variables['params'])
            truth_loss, truth_tree = truth_step(params, batch)
            truth_loss, truth = float(truth_loss), flat(truth_tree, order)
            if identity is None:
                identity, identity_loss = truth, truth_loss
            invariance.append(order_invariance(truth, identity, truth_loss, identity_loss))
            ref_state, ref_loss, _, _, _ = ref_step(ordered_state(base, order, ref_shardings), ref_data)
            ref_gradient = flat(ref_state.opt_state[0]['gradient'], order)
            keys = (*truth, '@loss')
            for path in truth:
                ref_distances.setdefault(path, []).append(distance(ref_gradient[path], truth[path]))
            ref_distances.setdefault('@loss', []).append(abs(float(ref_loss) - truth_loss))
            saved = {'reference': np.asarray([ref_distances[path][-1] for path in keys]),
                     'truth_loss': np.asarray(truth_loss), 'leaves': np.asarray(keys),
                     'invariance': np.asarray(invariance[-1])}
            for name, (template, data, step, _compiled) in programs.items():
                shardings = jax.tree.map(lambda leaf: leaf.sharding, template)
                state, loss, _, _, _ = step(ordered_state(template, order, shardings), data)
                gradient = flat(state.opt_state[0]['gradient'], order)
                for path in truth:
                    layout_distances[name].setdefault(path, []).append(distance(gradient[path], truth[path]))
                layout_distances[name].setdefault('@loss', []).append(abs(float(loss) - truth_loss))
                saved[name] = np.asarray([layout_distances[name][path][-1] for path in keys])
            np.savez(model_out / f'order-{index:02}.npz', **saved)
            speak(f'[{model}] residual order {index + 1}/{ORDERS}, fp64 invariance {invariance[-1]:.3e}')
        for name, (_, _, _, compiled) in programs.items():
            row = {"model": model, "layout": name, "dtype": dtype, "orders": ORDERS,
                   "gate_proj_inventory": [path for path in ref_distances if 'gate_proj' in path],
                   "invariance_max": max(invariance),
                   "reference_distances": ref_distances, "layout_distances": layout_distances[name],
                   **compiled, **judged_orders(ref_distances, layout_distances[name])}
            if compiled['flops_per_device'] and ref_compiled['flops_per_device']:
                row['flops_ratio'] = (compiled['flops_per_device'] * math.prod(compiled['mesh'].values())
                                      / ref_compiled['flops_per_device'])
                row['flops_bound'] = flops_bound(LAYOUTS[name])
                if row['status'] == 'works' and row['flops_ratio'] > row['flops_bound']:
                    row['status'] = 'REDUNDANT'
            rows.append(row)
            speak(f'[{model}/{name}] {row["status"]} K-order ratio {row["worst_ratio"]:.3f} '
                  f'at {row["worst_leaf"]}')
        (out / 'summary.json').write_text(json.dumps(rows, indent=1))
    return rows


def model_case(model: str, dtype: str, mixture: dict[str, Any], objective: dict[str, Any]):
    """`model`'s zoo case in `dtype`, with the flags' mixture and objective
    keywords merged in, and the case its one-device reference runs: the
    exchange needs an expert axis, so one device computes the same layer
    through the global dispatch."""
    case = dataclasses.replace(zoo()[model], dtype=dtype)
    if dtype == "bfloat16" and case.architecture == "causal_transformer":
        # At the default precision a bf16 vocabulary head rounds its logits
        # and their cotangent to bf16 (`dew.nn.precision.head_product`). With
        # the cotangent rounded once, 4 RTX 3090s read 1.75 of the bound at a
        # dense model's final norm and 5758 of it at an MoE's expert
        # gate_proj, against 0.47 and 0.41 with it carried in fp32. "highest"
        # keeps the head fp32.
        case = dataclasses.replace(case, matmul_precision="highest")
    if mixture:
        if "mixture" not in case.config:
            raise ValueError(f"{model} has no mixture for --mixture to change")
        case = dataclasses.replace(case, config={
            **case.config, "mixture": {**case.config["mixture"], **mixture}})
    case = dataclasses.replace(case, objective={**case.objective, **objective})
    reference = case if case.config.get("mixture", {}).get("dispatch") != "exchange" else (
        dataclasses.replace(case, config={
            **case.config, "mixture": {**case.config["mixture"], "dispatch": "global"}}))
    return case, reference


def prepared(models: Sequence[str], *, dtype: str, steps: int, anchor: bool, mixture: dict[str, Any],
             objective: dict[str, Any], references: References, speak: Callable[[str], None]) -> bool:
    """Each model's reference, and with `anchor` its fp64 anchor, computed
    into `references` where missing; whether every model's is there."""
    import benchmark_models as bench

    whole = True
    for model in models:
        started = time.perf_counter()
        try:
            case, reference = model_case(model, dtype, mixture, objective)
            batch = bench.global_batch(case)
            judge = references.reference(reference, batch, steps)
            if anchor:
                references.anchor(reference, batch)
        except Exception as error:  # the other models' references are still worth keeping
            whole = False
            speak(f"[{model}] reference error {type(error).__name__}: {error}"[:400])
            continue
        floor = "" if judge.floors is None else f", largest floor {max(judge.floors.values()):.2e}"
        speak(f"[{model}] reference losses {judge.losses}{floor}, {time.perf_counter() - started:.0f} s")
    return whole


def run(models: Sequence[str], layouts: Sequence[str], *, dtype: str, steps: int, anchor: bool,
        mixture: dict[str, Any], objective: dict[str, Any], references: References,
        speak: Callable[[str], None], keep: Callable[[list[dict[str, Any]]], None],
        devices: int | None = None, orders_out: Path | None = None) -> list[dict[str, Any]]:
    """Every layout of every model, one row each, `keep` handed the rows so
    far after each, each layout over the process's first `devices` devices
    (every device by default). A reference and a layout run as agreed
    phases: a failure on one process fails that row on every process, or,
    where the others wait in a collective it left, ends the pool within the
    failure grace (dew.artifacts) with the rows kept so far."""
    import benchmark_models as bench
    import jax

    from dew.coordination import agreed
    from dew.nn.sharding import LayoutRefused

    rows = []
    for model in models:
        case, reference = model_case(model, dtype, mixture, objective)
        if case.orders:
            with jax.enable_x64(new_val=True):
                rows.extend(run_orders(
                    [model], layouts, dtype=dtype, devices=devices, mixture=mixture, objective=objective,
                    out=Path('out/layout-parity-orders') if orders_out is None else orders_out, speak=speak))
            keep(rows)
            continue
        batch = bench.global_batch(case)
        try:
            judge = agreed(f"reference of {model}", lambda: references.reference(reference, batch, steps))
            ref_losses, ref_gradient = judge.losses, judge.gradient
            if judge.floors is None or judge.loss_floor is None:
                floors, loss_floor = agreed(f"floor of {model}", lambda: data_parallel_floor(
                    reference, batch, ref_gradient, ref_losses[0], devices))
            else:
                floors, loss_floor = judge.floors, judge.loss_floor
            if anchor:
                anchor_loss, anchor_gradient = agreed(f"anchor of {model}",
                                                      lambda: references.anchor(reference, batch))
                rounding = leaf_errors(anchor_gradient, ref_gradient, dtype)
                farthest = max(rounding, key=rounding.__getitem__)
                if rounding[farthest] > rounding_limit(dtype):
                    raise ValueError(
                        f"the {dtype} reference is {rounding[farthest]:.2e} from the fp64 step at "
                        f"{farthest}, past {dtype} rounding ({rounding_limit(dtype):.1e}): the two "
                        f"compute different steps, so the anchor cannot bound the layouts")
                floors = {leaf: max(value, rounding[leaf]) for leaf, value in floors.items()}
                # The loss rounds the same sums, inside rows too: DPO's first
                # step scores each pair from two log-likelihood sums that
                # cancel, which no reordering of the rows moves.
                loss_floor = max(loss_floor, abs(ref_losses[0] - anchor_loss))
            widest = widest_floor(floors, dtype)
        except Exception as error:  # no reference judges no layout: the model's one row
            rows.append({"model": model, "layout": "reference", "processes": jax.process_count(),
                         "status": "error", "error": f"{type(error).__name__}: {error}"[:2000],
                         "traceback": traceback.format_exc()[-4000:]})
            speak(f"[{model}] reference error {rows[-1]['error'][:300]}")
            keep(rows)
            continue
        speak(f"[{model}] reference losses {ref_losses}, largest floor {floors[widest]:.2e} at {widest}")
        for name in layouts:
            row: dict[str, Any] = {"model": model, "mixture": case.config.get("mixture"),
                                   "objective": case.objective, "layout": name,
                                   "processes": jax.process_count(),
                                   "reference_losses": ref_losses}
            started = time.perf_counter()
            try:
                losses, gradient, compiled = agreed(
                    f"{model} on {name}",
                    lambda: trained(case, LAYOUTS[name], batch, steps=steps, devices=devices))
                row.update(compiled, losses=losses, **judged(
                    leaf_errors(ref_gradient, gradient, dtype), floors, abs(losses[0] - ref_losses[0]),
                    loss_floor, ref_losses[0], contracted_terms(batch)))
                if compiled["flops_per_device"] and judge.flops_per_device:
                    # Above one when devices compute what one device need not.
                    size = math.prod(compiled["mesh"].values())
                    row["flops_ratio"] = compiled["flops_per_device"] * size / judge.flops_per_device
                    row["flops_bound"] = flops_bound(LAYOUTS[name])
                    if row["status"] == "works" and row["flops_ratio"] > row["flops_bound"]:
                        row["status"] = "REDUNDANT"
            except LayoutRefused as refusal:
                row.update(status="refused", reason=str(refusal))
            except Exception as error:  # a failing layout is a row of the matrix
                row.update(status="error", error=f"{type(error).__name__}: {error}"[:2000],
                           traceback=traceback.format_exc()[-4000:])
            row["seconds"] = round(time.perf_counter() - started, 1)
            rows.append(row)
            keep(rows)
            speak(f"[{model}/{name}] {row['status']} "
                  + (f"leaf {row['worst_ratio']:.2f} of bound at {row['worst_leaf']}, "
                     f"loss {row['loss_error']:.1e} of {row['loss_bound']:.1e}, "
                     f"flops x{row.get('flops_ratio', float('nan')):.2f} of x{row.get('flops_bound', float('nan')):.2f}"
                     if "worst_ratio" in row else row.get("reason", row.get("error", ""))[:300]))
    return rows


def verdict(rows: Sequence[Mapping[str, Any]]) -> int:
    """The run's exit status: 1 when a row is past `PASSING`, else 0."""
    return int(any(row["status"] not in PASSING for row in rows))


def summary(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """How many rows took each status, then each refusal with its reason."""
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    lines = [", ".join(f"{status} {count}" for status, count in counts.items())]
    lines += [f"refused {row['model']}/{row['layout']}: {row['reason']}"
              for row in rows if row["status"] == "refused"]
    return lines


def main(models: Annotated[tuple[str, ...], tyro.conf.arg(help="zoo() names")] = ("dense",),
         layouts: Annotated[tuple[str, ...], tyro.conf.arg(help="LAYOUTS names")] = tuple(LAYOUTS),
         dtype: str = "float32", steps: int = 3, anchor: bool = False,
         rounding_orders: Annotated[bool, tyro.conf.arg(
             help="52 exact residual orders against fp64; one step, --out is a directory; Colab")] = False,
         out: Path | None = None,
         mixture: Annotated[str, tyro.conf.arg(
             help="JSON merged into each model's mixture, e.g. '{\"dispatch\": \"exchange\"}'")] = "{}",
         objective: Annotated[str, tyro.conf.arg(
             help="JSON of LMObjective keywords, e.g. '{\"aux_loss_alpha\": 0.01}'")] = "{}",
         references: Annotated[Path | None, tyro.conf.arg(
             help="a directory of references a --prepare run wrote, read instead of computed")] = None,
         prepare: Annotated[bool, tyro.conf.arg(
             help="compute each model's missing references into --references and run no layout")] = False,
         devices: Annotated[int | None, tyro.conf.arg(
             help="run each layout on this process's first N devices: four stages on four devices "
                  "leave the rows unsplit, which eight, putting the rest on data, never show")] = None,
         ) -> None:
    """Run the layouts of each model against one device; see the module docstring."""
    from dew.training.runtime import prepare_process

    prepare_process()
    import jax

    if anchor and not jax.config.jax_enable_x64:
        raise SystemExit("--anchor computes the step in fp64, which needs JAX_ENABLE_X64=1")
    if prepare and (references is None or jax.process_count() > 1):
        raise SystemExit("--prepare writes the --references directory from one process")
    if devices is not None and (jax.process_count() > 1 or not 0 < devices <= jax.device_count()):
        raise SystemExit(f"--devices takes the first N of one process's {jax.device_count()} devices; "
                         "a pool runs every layout on all of its devices")
    jax.config.update("jax_default_matmul_precision", "highest")
    speaker = jax.process_index() == 0
    if rounding_orders:
        if steps != 1 or out is None or prepare or references is not None or anchor:
            raise SystemExit('--rounding-orders needs --steps 1 and --out DIR; '
                             '--prepare/--references/--anchor use the single-draw path')
        rows = run_orders(models, layouts, dtype=dtype, devices=devices, mixture=json.loads(mixture),
                          objective=json.loads(objective), out=out,
                          speak=lambda line: print(line, flush=True))
        print('\n'.join(summary(rows)), flush=True)
        raise SystemExit(verdict(rows))
    store = References(references, prepare)
    if prepare:
        if not prepared(models, dtype=dtype, steps=steps, anchor=anchor, mixture=json.loads(mixture),
                        objective=json.loads(objective), references=store,
                        speak=lambda line: print(line, flush=True)):
            raise SystemExit(1)
        return

    def keep(rows: list[dict[str, Any]]) -> None:
        if speaker and out is not None:
            out.write_text(json.dumps(rows, indent=1))

    rows = run(models, layouts, dtype=dtype, steps=steps, anchor=anchor, mixture=json.loads(mixture),
               objective=json.loads(objective), references=store,
               speak=lambda line: print(line, flush=True) if speaker else None, keep=keep,
               devices=devices,
               orders_out=None if out is None else out.parent / f'{out.stem}-orders')
    if speaker:
        print("\n".join(summary(rows)), flush=True)
    raise SystemExit(verdict(rows))


if __name__ == "__main__":
    tyro.cli(main)
