#!/usr/bin/env python3
"""Every layout of a model against one device: the loss and each gradient leaf, by the floor judge or
check 1.

Each process builds the one-device reference on its first local device, then
each layout's trainer on the global mesh, from one key and one initial state,
and runs the trainer's own compiled first step. The optimizer is `stash`
before adam, so the gradient the step handed the optimizer is kept whole.

By default a layout's first step is judged by the floor judge (`floored`), the parallelism matrix's
accuracy judge while check 1's draws cannot shard across its tasks: each gradient leaf and the loss
within FLOOR_FACTOR of the reference's own deviation. `--seeds N` judges it by the gate's check 1
instead (tests/reference_error.py, `assert_as_exact`): over the model's N draws, each a seed of its
global batch (`benchmark_models.global_batch`), every gradient leaf and the loss within PARITY of
the one-device reference's own error from the step computed in float64, with check 1's redraw and
its row rule. `works`, MISMATCH, or INCONCLUSIVE where the draws do not resolve it, which never
passes. `--seeds 0` runs each layout once, its sample held to the reference's (`ran`).

The float64 truth is the anchor (`Anchor`): the case's float64 twin, its
model's fields the native model's but its dtype, over the reference's own
initial state widened, through the Step the trainer's first step makes. Every
arm runs untaped and records what its loss drew (`Step.sample`) and started
its model computation from (`Step.boundary`), by each example's place, which
the harness's column of the batch (`EXAMPLE_ROWS`) carries through any
layout; each layout's record and the anchor's must be the reference's, bit
for bit, else the arm is refused (`coupled`, `CouplingRefused`). The loss is
audited once (`audited`): randomness reaching it outside its draws (dropout, a
draw from `Step.key`) refuses the model, and in the anchor every floating
equation past its boundaries must be float64, while what feeds only its
boundaries runs native, witnessed by them. A refused reference or anchor
makes the model's row INCONCLUSIVE.

For a decoder whose individual gradient error is a coin flip, use
`--rounding-orders --out DIR` with JAX_ENABLE_X64=1. It evaluates 52 exact
residual-unit permutations against fp64 and uses tests/reference_error.py's
K-order rule per gradient leaf and for the loss. Each order's fp64 function
invariance is checked; all raw distances and the expert gate_proj inventory
are kept in DIR. Initialization stays at the default fp32 draw. This
multi-draw mode belongs in the Colab queue.

A layout also has to split one device's work rather than repeat it: its
devices' FLOPs together, over the reference's, are at most `flops_bound`, an
even split and a pipeline's bubble, else the layout is REDUNDANT.

The reference side (the one-device step and the float64 anchor on each draw)
runs on one device while the others wait. With `--references DIR`,
`--prepare` computes each model's draws, its seeds and the redraw's, into DIR
in a job of one device, and a run of layouts reads them back and computes
none; a missing one is refused, naming `--prepare`. A draw is keyed by all its
numbers depend on (`_cache_name`): the case, the seed and its batch, the
producer, the device kind and backend that ran it, jax, x64 and Dew's source.

The same command runs in one process or under `dew launch`, where the global
batch is placed from every process alike:

    python tools/layout_parity.py --models dense --layouts data4,fsdp4,tensor4
    python tools/layout_parity.py --models dense moe --seeds 8 --references refs --prepare  # one device
    python tools/layout_parity.py --models dense moe --seeds 8 --references refs              # every device
    python tools/layout_parity.py --models moe --layouts expert4,data2_expert2,expert2_fsdp2 \\
        --mixture '{"experts": 32, "top_k": 8, "dispatch": "exchange"}' \\
        --objective '{"balance": {"aux_loss_alpha": 0.01}}'
    python tools/layout_parity.py --models dense --layouts stage4,stage2_sequence2 --devices 4
    dew launch --processes-per-host 4 --devices-per-process 1 -- \\
        python tools/layout_parity.py --models dense,dit --out parity.json

A layout's axes other than data are its MeshSpec fields and data takes the
devices left over, so the same layout splits differently on another device
count: `--devices 4` runs it on the process's first four, where four stages
leave no axis for the rows.

Process 0 prints a line per layout and writes the rows. A layout Dew refuses
by design (`dew.mesh.LayoutRefused`: a stage axis over a model that
runs no pipeline) is a row with its reason, listed again at the end, and
passes the run; the exit status is nonzero when a layout mismatches, is
inconclusive, repeats work or fails in any other way.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import hashlib
import json
import math
import shutil
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
    "data4_explicit": {"explicit": ("data",)},
    "fsdp2_explicit": {"fsdp": 2, "explicit": ("data",)},
}
"""Four-device layouts: every axis alone and the combinations worth running,
and the data axis Explicit (`MeshSpec.explicit`) alone and beside fsdp."""


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


GROUPED_NORMS = dict.fromkeys(("input_layernorm", "post_attention_layernorm", "norm"), {"groups": 4})
"""K2 Horizon's norms, each over four groups of the width."""
OWN_GATE = {"output_gate": True, "layer_types": ("attention",) * 4,
            "kinds": {"attention": {"mixer": {"class": "attention", "fields": {"gate_projection": "own"}}}}}
"""Muse Glimmer's attention gate, a projection of its own over the layer's input."""


def zoo() -> dict[str, Any]:
    """Small models of every family a layout splits differently: a dense decoder (also under
    `GROUPED_NORMS` and `OWN_GATE`), MoE decoders with 8 and with Qwen3-30B-A3B's 128 experts, a
    Mamba-2 hybrid, a DiT and DiffusionGemma, each at widths every layout above divides. The DiT's
    rows hold 256 patches, as a DiT's smallest release does (DiT-B/2 at 256 px): a sequence or tensor
    split repeats each row's conditioning on every shard by design, which the FLOPs bound would
    count against a row of only a few patches. The last two MoE models and DiffusionGemma keep their
    released configs' routing and layer kinds. The layers a sequence axis
    splits differently come too: a dense layer before routed ones and a tail
    sharing the keys and values one layer stored, which a pipeline runs on
    every stage, latent attention (MLA), a window (12 rows:
    a sequence split two ways reads it from one neighbour, four ways through
    the exchange), Mamba-2 alone, Mamba-1 alone, and packed rows of three
    documents whose boundaries fall inside the shards, in the dense decoder,
    in Mamba-1 and in Rigel's three Mamba-2 layers to one windowed layer.

    The dense, MoE, hybrid and MLA decoders also train every other objective
    a decoder takes: SFT over the assistant's turns (`_sft`), DPO over pairs
    (`_dpo`), GRPO over rollouts with a KL to the frozen reference (`_grpo`)
    and, made bidirectional, masked diffusion (`_mdlm`), which the hybrid's
    Mamba-2 refuses on one device (its recurrence runs one way). The other families
    train their own: a conditional UNet, EDM2's magnitude-preserving U-Net
    and an MMDiT (SD3's) denoise latents,
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
    mamba1 = {**dense, "layer_types": ("mamba",) * 4, "kinds": {"mamba": {"mixer": {
        "class": "mamba", "fields": {"intermediate_size": 128, "state_size": 8, "time_step_rank": 4,
                                     "chunk_size": 8}}}}}
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
    return {
        "dense": Case("causal_transformer", dense, **lm),
        "grouped": Case("causal_transformer", {**dense, "norms": GROUPED_NORMS}, **lm),
        "gated": Case("causal_transformer", {**dense, **OWN_GATE}, **lm),
        "moe": Case("causal_transformer", moe, **lm),
        "moe128": Case("causal_transformer", moe128, **lm),
        "hybrid": Case("causal_transformer", hybrid, **lm),
        **{name: Case("causal_transformer", config, **lm)
           for name, config in _placed_zoo(dense, moe).items()},
        "nemotron_h": Case("causal_transformer", nemotron_h, **lm),
        "nemotron_h_moe": Case("causal_transformer", _nemotron_h_moe(), orders=True, **lm),
        "window": Case("causal_transformer", window, **lm),
        "mla": Case("causal_transformer", mla, **lm),
        "mamba2": Case("causal_transformer", mamba2, **lm),
        "dense_packed": Case("causal_transformer", dense, packed_documents=3, **lm),
        "rigel_packed": Case("causal_transformer", rigel, packed_documents=3, **lm),
        "mamba1": Case("causal_transformer", mamba1, **lm),
        "mamba1_packed": Case("causal_transformer", mamba1, packed_documents=3, **lm),
        "dit": Case("simple_dit", dit, batch_size=8, image_size=32, channels=4,
                    fsdp_min_param_size=256),
        "dgemma": Case("diffusion_gemma", dgemma, canvas={"prompt_length": 16, "canvas_size": 8},
                       batch_size=8, seq_len=31, fsdp_min_param_size=256),
        **trained,
        **diffused,
        **_image_zoo(),
        "multimodal": Case("multimodal_transformer", dense, media={
            "family": "gemma3", "image_token_id": 511, "images": 1, "pixels": [3, 16, 16],
            "tower": {"class": "siglip", "fields": {"hidden_size": 32, "intermediate_size": 64, "num_layers": 1,
                      "num_heads": 4, "image_size": 16, "patch_size": 8}},
            "projector": {"class": "gemma", "fields": {"text_width": 64, "patches_per_side": 2,
                          "tokens_per_side": 2}}}, **lm),
    }


def _placed_zoo(dense: dict[str, object], moe: dict[str, object]) -> dict[str, dict[str, object]]:
    """The zoo's decoders a layout places apart (`zoo`), built on its `dense` and `moe` ones.

    The expert exchange runs under a capacity, whose buckets hold what each
    device's sequences keep, one round, so its program splits evenly;
    dropless, it also compiles the rounds a skewed routing needs, which a
    static count of the step's FLOPs reads whether or not they run.
    DeepSeek's class puts a dense first layer before eight routed ones, each
    layer reading its own per-layer input (Gemma's): a pipeline runs the
    dense layer on every stage and slices the inputs past it, each of four
    stages scanning two. Gemma 3n's has its last two layers read the keys
    and values the fourth stored, which a pipeline runs on every stage."""
    mixture = moe["mixture"]
    assert isinstance(mixture, dict)
    return {
        "moe_exchange": {**moe, "mixture": {**mixture, "dispatch": "exchange", "capacity_factor": 2.0}},
        "routed_tail": {**dense, "num_layers": 9, "per_layer_input_dim": 8,
                        "mixture": {**mixture, "layers": tuple(range(1, 9))}},
        "shared_tail": {**dense, "num_layers": 6, "kv_shared_layers": (4, 5)},
    }


def _image_zoo() -> dict[str, Any]:
    """The zoo's image models (`zoo`): a conditional UNet, EDM2's magnitude-preserving U-Net and an
    MMDiT (SD3's) denoising latents, and a JEPA encoder predicting its masked patches' features."""
    from benchmark_cases import Case

    images: ImageRows = {"batch_size": 8, "channels": 4, "fsdp_min_param_size": 256}
    return {
        "unet": Case("unet_2d_condition", {"stages": [{"features": 32, "heads": 4},
                                                      {"features": 64, "heads": 4}],
                                           "blocks_per_level": 1, "in_channels": 4, "out_channels": 4},
                     image_size=16, **images),
        "edm2": Case("edm2_unet", {"model_channels": 32, "channel_mult": [1, 2], "num_blocks": 1,
                                   "attn_resolutions": [8], "channels_per_head": 16, "output_channels": 4},
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
    }


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

    from dew.distributed.batches import batch_shardings

    def put(leaf, sharding):
        array = np.asarray(leaf)
        return jax.make_array_from_callback(array.shape, sharding, lambda index: array[index])
    return jax.tree.map(put, batch, batch_shardings(mesh, batch))


def zeros_drawn(params, key):
    """`params` with every leaf its initializer left all zeros drawn from a
    normal of std 0.02 instead, each from its own fold of `key`.

    A DiT zero-initializes its output projection and its modulations, so its
    first step's gradient reaches those layers alone: 2 of the zoo DiT's 70
    leaves, and a layout that split anything above them wrongly would pass.
    Every state of a comparison, the reference's, a layout's, a floor's and
    the fp64 anchor's, comes through here from one key, so they start alike.

    `key` is concrete, and each draw runs eagerly even inside a trace
    (`jax.ensure_compile_time_eval`), so an initializer holds the draws as
    constants: staged, a normal for every leaf made the initializer's compile
    most of a cell's time (the UNet's 286 leaves, 187 s of its 302)."""
    import jax
    import jax.numpy as jnp

    def draw(index, leaf):
        if not jnp.issubdtype(leaf.dtype, jnp.floating):
            return leaf
        with jax.ensure_compile_time_eval():
            drawn = (0.02 * jax.random.normal(jax.random.fold_in(key, index), leaf.shape)).astype(leaf.dtype)
        return jnp.where(jnp.any(leaf != 0), leaf, drawn)

    leaves, tree = jax.tree.flatten(params)
    return jax.tree.unflatten(tree, [draw(index, leaf) for index, leaf in enumerate(leaves)])


def _trainer(case, fields: dict[str, int], *, one_device: bool = False, accumulation: int = 1,
             devices: int | None = None, record: bool = False):
    """The trainer of `case` on the layout `fields` names over the first
    `devices` devices (every device by default), or on this process's first
    device, stashing each gradient the optimizer is handed. Its initial
    parameters have no all-zero leaf (`zeros_drawn`). In a one-process run, once
    the case's one-device state is computed (`_initial`), every other
    trainer copies it onto its mesh: every layout starts from the very state
    one device does, without a compile of the model's initialization on each.
    A layout's own draw of it is another program, whose rounding a bitwise
    comparison would judge (docs/design/testing.md). `record` makes each
    step return what its loss drew and started from (`Arm`)."""
    import benchmark_models as bench
    import jax
    import optax

    from dew.mesh import SEQUENCE_AXIS, TENSOR_AXIS
    from dew.nn.kernels.generation import gpu_target
    from dew.nn.sharding import Link
    from dew.training import Layout, MeshSpec, Trainer

    class Drawn(Trainer):
        def initial_state(self, initializer=None, key=None):
            state = super().initial_state(initializer, key)
            with jax.ensure_compile_time_eval():
                drawing = jax.random.key(7)
            params = {**state.variables, "params": zeros_drawn(state.variables["params"], drawing)}
            return dataclasses.replace(state, variables=params,
                                       opt_state=self.optimizer.init(params["params"]))

        def place(self):
            held = _initial.get((repr(case), accumulation)) if jax.process_count() == 1 else None
            # Copies every time, in and out: the step donates the state it is handed.
            if held is None:
                state, shardings, position = super().place()
                if one_device:
                    _initial[(repr(case), accumulation)] = jax.tree.map(lambda leaf: leaf.copy(), state)
                return state, shardings, position
            shardings, _ = self._validated(jax.tree.map(
                lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), held))
            return jax.device_put(jax.tree.map(lambda leaf: leaf.copy(), held), shardings), shardings, None

    # Devices the peak table does not name (a CPU's) measure no link, so no projection would spread
    # over them; the layouts state one fast enough that every projection that can spread does, as
    # it would over NVLink, so the matrix holds the spread path wherever it runs.
    unmeasured = gpu_target(jax.devices()[0].device_kind).bf16_peak is None
    links = {axis: Link(1e15, 1e12) for axis in (TENSOR_AXIS, SEQUENCE_AXIS)} if unmeasured else None
    trainer = Drawn(bench.build_objective(case), optax.chain(stash(), optax.adam(1e-3)),
                      key=jax.random.key(0), **bench.trainer_mesh(fields),
                      layout=Layout(min_shard=case.fsdp_min_param_size, tolerance=1.0),
                      accumulation=accumulation, checkpoints=None, tracker=None, links=links, record=record)
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
    compiled = {"flops_per_device": trainer.step_flops and trainer.step_flops.hardware,
                "mesh": {axis: int(size) for axis, size in trainer.device_mesh.shape.items()},
                "replicated": replicated_share(case, trainer.objective, fields.get("stage", 1))}
    return state, data, step, compiled


def replicated_share(case, objective, stages: int) -> float:
    """The share of `objective`'s model's forward work over the case's rows a pipeline of `stages`
    runs on every stage (`replicated_flops`); none without a pipeline, and none for a model with no
    decoder to pipeline, whose work the even split then judges whole."""
    import numpy as np

    from dew.nn.backbones.causal_transformer import CausalTransformer
    from dew.nn.wrapping import DecoderWrapper

    model = next(iter(objective.program_key()), None)
    if stages == 1 or model is None or not isinstance(model.module, (CausalTransformer, DecoderWrapper)):
        return 0.0
    lengths = np.full(case.batch_size, case.seq_len)
    return model.module.replicated_flops(stages, lengths) / model.module.model_flops(lengths)


def trained(case, fields: dict[str, int], batch, *, steps: int, one_device: bool = False,
            devices: int | None = None) -> tuple[list[float], dict[str, NDArray], dict[str, Any]]:
    """The losses of `steps` steps on the layout `fields` names (over the
    first `devices` devices, every device by default), step one's gradient
    gathered whole, and what the compiler says of the step."""
    state, data, step, compiled = compiled_step(case, fields, batch, one_device=one_device, devices=devices)
    losses, gradient = [], {}
    for _ in range(steps):
        state, loss, _, _, _, _ = step(state, data)
        losses.append(float(loss))
        gradient = gradient or _gradient(state)
    return losses, gradient, compiled


class CouplingRefused(Exception):
    """An arm of a comparison computed from another sample than the one-device reference's: a value
    its loss drew (`Step.sample`) or started its model computation from (`Step.boundary`) differs from
    the reference's at an example, is missing or extra, a row sat at a place its slot and the batch's
    column do not give it, randomness reached the loss outside a draw, or the float64 anchor rounds
    past its boundaries. A ratio would then judge two samples or two functions, not two roundings, so
    none is taken: the arm is a MISMATCH, or the model's row INCONCLUSIVE where its reference or
    anchor is the one refused."""


Witnessed = dict[str, dict[int, Any]]
"""What an arm's loss drew and started from: each `Step.sample` and `Step.boundary` name's value at
each example, by the example's place among the update's examples (`Step.example_index`)."""


@dataclasses.dataclass(frozen=True)
class Drawn:
    """One arm's first step on one draw: its loss, the gradient it handed the optimizer by leaf name,
    and what its loss drew and started from (`witnessed`)."""
    loss: float
    gradient: dict[str, NDArray]
    record: Witnessed

    def leaves(self) -> dict[str, NDArray]:
        """What check 1 judges of the step: every gradient leaf and the loss."""
        import numpy as np

        return {**self.gradient, "@loss": np.asarray(self.loss, np.float64)}


def drawn_batch(case, seed: int):
    """Draw `seed` of the case's global batch (`benchmark_models.global_batch`), with the harness's
    column of each row's place among the examples (`EXAMPLE_ROWS`), which a recording step holds its
    rows' places to, however a layout places and slices them."""
    import benchmark_models as bench
    import jax
    import numpy as np

    from dew.batch import EXAMPLE_ROWS

    batch = bench.global_batch(case, seed)
    return {**batch, EXAMPLE_ROWS: np.arange(len(jax.tree.leaves(batch)[0]), dtype=np.int32)}


def _same(left, right) -> bool:
    """Whether two values are one value: dtype, shape and every byte."""
    return left.dtype == right.dtype and left.shape == right.shape and left.tobytes() == right.tobytes()


def witnessed(record, label: str) -> Witnessed:
    """`record` by name and example: a recording step's (`Recorded`, its evaluations stacked on a first
    axis, `evaluated` saying which ran) or one evaluation's (`Evaluated`). A place given twice holds
    one value, a replay's equal to its first evaluation's, and an evaluated row its slot and the
    batch's column place apart refuses the arm (`CouplingRefused`)."""
    import numpy as np

    values = {name: np.asarray(value) for name, value in record["values"].items()}
    places = np.asarray(record["example_index"])
    if "evaluated" in record:
        ran = np.flatnonzero(np.asarray(record["evaluated"], bool))
        if not np.all(np.asarray(record["column_ok"])[ran]):
            raise CouplingRefused(f"{label}: a row sat at a place its slot and the batch's column do not "
                                  "give it")
        evaluations = [(places[index], {name: value[index] for name, value in values.items()})
                       for index in ran]
    else:
        evaluations = [(places, values)]
    found: Witnessed = {}
    for rows, by_name in evaluations:
        for name, value in by_name.items():
            held = found.setdefault(name, {})
            for row, place in enumerate(np.asarray(rows).tolist()):
                if place in held and not _same(held[place], value[row]):
                    raise CouplingRefused(f"{label}: {name!r} at example {place} differs between its "
                                          "evaluations")
                held.setdefault(place, value[row])
    return found


def coupled(reference: Witnessed, arm: Witnessed, label: str) -> None:
    """That the arm drew and started from what the reference did: every name at every example, at its
    own dtype, bit for bit, and nothing more; else `CouplingRefused`, naming the name, the example and
    the arm."""
    if reference.keys() != arm.keys():
        raise CouplingRefused(f"{label}: records {sorted(arm)} where the reference records "
                              f"{sorted(reference)}")
    for name, examples in reference.items():
        theirs = arm[name]
        if examples.keys() != theirs.keys():
            apart = sorted(set(examples) ^ set(theirs))[:8]
            raise CouplingRefused(f"{label}: {name!r} holds examples {apart} on one side only")
        for place, value in examples.items():
            if not _same(value, theirs[place]):
                raise CouplingRefused(f"{label}: {name!r} at example {place} is not the reference's "
                                      f"({theirs[place].dtype}{theirs[place].shape} against "
                                      f"{value.dtype}{value.shape})")


def taped(record: Witnessed) -> dict[str, Any]:
    """`record` as a `Step.tape`: each name's values stacked at their examples' places."""
    import numpy as np

    tape = {}
    for name, examples in record.items():
        first = np.asarray(next(iter(examples.values())))
        held = np.zeros((max(examples) + 1, *first.shape), first.dtype)
        for place, value in examples.items():
            held[place] = value
        tape[name] = held
    return tape


def _gathered(tree):
    """`tree`'s leaves whole on every process, as NumPy."""
    import jax
    import numpy as np
    from jax.experimental import multihost_utils

    return jax.tree.map(lambda leaf: np.asarray(
        leaf if not isinstance(leaf, jax.Array) or leaf.is_fully_addressable
        else multihost_utils.process_allgather(leaf, tiled=True)), tree)


class Arm:
    """One layout's trainer and its first step, compiled once and run on each draw (`drawn_batch`) from
    the one initial state every layout starts from (`_initial`), placed once on the layout's mesh and
    each run on a fresh copy of it there (the step donates the state it is handed), recording what its
    loss drew and started from (`Trainer(record=True)`)."""

    def __init__(self, case, fields: dict[str, int], *, one_device: bool = False, devices: int | None = None):
        import jax
        import jax.numpy as jnp

        self.case = case
        self.trainer = _trainer(case, fields, one_device=one_device, devices=devices, record=True)
        self.initial, _, _ = self.trainer.place()
        # One program copies the placed state on its devices, where placing it again from the host
        # each draw recomputed its shardings and moved every leaf (most of a small model's draw).
        self.fresh = jax.jit(lambda state: jax.tree.map(jnp.copy, state))
        self.step = self.trainer.compile(self.initial, placed(drawn_batch(case, 0), self.trainer.device_mesh))
        self.compiled = {"flops_per_device": self.trainer.step_flops and self.trainer.step_flops.hardware,
                         "mesh": {axis: int(size) for axis, size in self.trainer.device_mesh.shape.items()},
                         "replicated": replicated_share(case, self.trainer.objective, fields.get("stage", 1))}

    def __call__(self, seed: int) -> Drawn:
        from dew.objectives.base import taken_record

        data = placed(drawn_batch(self.case, seed), self.trainer.device_mesh)
        state, loss, metrics, *_ = self.step(self.fresh(self.initial), data)
        record, _ = taken_record(metrics)
        assert record is not None, "a recording trainer's step returns its record"
        return Drawn(float(loss), _gradient(state), witnessed(_gathered(record), f"draw {seed}"))


RANDOM_PRIMITIVES = frozenset({"random_bits", "threefry2x32"})
"""The equations that turn a key into random numbers: past a draw (`Step.sample`), where an arm's
loss reads them, they are a draw no witness holds (dropout, a posterior's noise, a draw from
`Step.key`)."""


UNROUNDED = frozenset({
    "reshape", "squeeze", "expand_dims", "broadcast_in_dim", "transpose", "concatenate", "slice",
    "dynamic_slice", "dynamic_update_slice", "gather", "select_n", "copy", "copy_p", "rev", "pad",
    "optimization_barrier", "stop_gradient", "max", "min", "neg", "abs", "sign", "reduce_max", "reduce_min",
    "argmax", "argmin", "sort", "jit", "pjit", "cond", "while", "scan", "remat", "checkpoint",
    "custom_jvp_call", "custom_vjp_call", "custom_vjp_call_jaxpr", "shard_map", "closed_call",
})
"""Equations that move, select or order values without rounding them, or run programs whose own
equations the audits read (`reached`, through the gate's `equations`): a float32 boundary value they
carry past the boundaries stays the witnessed value, and only an equation that rounds past them must
be float64."""


def _rounds(eqn) -> bool:
    """Whether an equation can round a floating value: not one of UNROUNDED, nor an identity barrier
    (a primitive whose name says so), nor a conversion between equal dtypes or of a boolean, nor a
    product with or quotient by the literal one, nor a sum with the literal zero."""
    import numpy as np
    from jax.extend import core

    name = eqn.primitive.name
    if name in UNROUNDED or name.endswith("barrier"):
        return False
    if name == "convert_element_type":
        source = eqn.invars[0].aval.dtype
        return source != eqn.outvars[0].aval.dtype and source != np.bool_
    identity = {"mul": 1, "div": 1, "add": 0, "sub": 0}.get(name)
    operands = eqn.invars if name in ("mul", "add") else eqn.invars[1:]
    return identity is None or not any(
        isinstance(var, core.Literal) and np.all(np.asarray(var.val) == identity) for var in operands)


def reached(jaxpr, cuts: set) -> list:
    """The equations whose outputs `jaxpr`'s first output, the loss, reads other than through `cuts`
    (outputs of the equations it stops at), and every equation inside a reached one's programs, which
    run whole."""
    from jax.extend import core

    producer = {var: eqn for eqn in jaxpr.eqns for var in eqn.outvars}
    stack = [jaxpr.outvars[0]] if isinstance(jaxpr.outvars[0], core.Var) else []
    seen, found = set(), {}
    while stack:
        var = stack.pop()
        if var in seen or var in cuts:
            continue
        seen.add(var)
        eqn = producer.get(var)
        if eqn is None or id(eqn) in found:
            continue
        found[id(eqn)] = eqn
        stack.extend(v for v in eqn.invars if isinstance(v, core.Var))
    sys.path.insert(0, str(REPO / "tests"))
    from reference_error import equations

    return [inner for eqn in found.values() for inner in (eqn, *equations(eqn.params))]


def _named(eqn) -> str:
    """An equation as a refusal names it: its primitive, its scope and its outputs."""
    outputs = ", ".join(f"{var.aval}" for var in eqn.outvars)
    return f"{eqn.primitive.name} in {eqn.source_info.name_stack or 'the loss'} -> {outputs}"


def recording_loss(objective, variables, step_fields: Mapping[str, Any]):
    """The loss of `objective` over `variables`' other collections as a function of the parameters and
    the batch, and of a `Step.tape` to read its draws and boundaries from where one is given, returning
    beside it what its `Step.sample` and `Step.boundary` calls produced by name, and the kind of each
    name once traced (`Witness.kinds`)."""
    from dew.objectives.base import Step, Witness

    kinds: dict[str, str] = {}

    def loss(params, batch, tape=None):
        witness = Witness()
        step = Step(**step_fields, tape=tape, witness=witness)
        value, _ = objective.scalar_loss({**variables, "params": params}, batch, step)
        kinds.update(witness.kinds)
        return value, dict(witness.values)

    return loss, kinds


def audited(loss, kinds: Mapping[str, str], params, batch, label: str, *, wide: bool) -> set[str]:
    """Refuse (`CouplingRefused`) a loss whose value reads randomness outside its draws: an equation of
    RANDOM_PRIMITIVES the loss reaches past every recorded value. `wide`, the float64 anchor's, also
    holds its partition: every equation the loss reaches past its boundaries that rounds a floating
    value (`_rounds`) outputs float64, and the prefix (what feeds only boundaries) runs native,
    witnessed by the boundary records. Returns
    the batch leaves the loss reads past its boundaries, which the anchor takes widened."""
    import jax
    import jax.numpy as jnp
    from jax.extend import core

    closed = jax.make_jaxpr(loss)(params, batch)
    jaxpr = closed.jaxpr
    # A value the loss records as a constant is no program's output, and cuts nothing.
    recorded = {name: var for name, var in zip(sorted(kinds), jaxpr.outvars[1:], strict=True)
                if isinstance(var, core.Var)}
    drawn = reached(jaxpr, set(recorded.values()))
    loose = [eqn for eqn in drawn if eqn.primitive.name in RANDOM_PRIMITIVES]
    if loose:
        raise CouplingRefused(f"{label}: randomness reaches the loss outside its draws: {_named(loose[0])}")
    paths = [jax.tree_util.keystr(path) for path, _ in jax.tree_util.tree_flatten_with_path(batch)[0]]
    inputs = dict(zip(jaxpr.invars[len(jax.tree.leaves(params)):], paths, strict=True))
    past = reached(jaxpr, {var for name, var in recorded.items() if kinds[name] == "boundary"})
    read = {inputs[var] for eqn in past for var in eqn.invars if isinstance(var, core.Var) and var in inputs}
    if wide:
        narrow = [eqn for eqn in past if _rounds(eqn) and any(
            jnp.issubdtype(var.aval.dtype, jnp.floating) and var.aval.dtype != jnp.float64
            for var in eqn.outvars)]
        if narrow:
            raise CouplingRefused(f"{label}: the float64 anchor rounds past its boundaries in "
                                  f"{len(narrow)} equations, the first {_named(narrow[0])}")
    return read


def anchor_loss(case, batch, state):
    """The fp64 loss of the trainer's first step over `state`'s widened variables, as a function of
    the parameters and the batch, and those widened parameters."""
    import benchmark_models as bench
    import jax

    from dew.objectives.base import Step
    from dew.training.transaction import example_fields, with_ema

    sys.path.insert(0, str(REPO / "tests"))
    from reference_error import widened

    wide = widened(state.variables)
    objective = bench.build_objective(case, widened=True)
    # The trainer's first step: its key folded with the step, its examples' (`example_fields`), which
    # draw a diffusion objective's noise and a masked one's masks, and the frozen EMA DPO and GRPO
    # score against.
    step = Step(state.microstep, jax.random.fold_in(state.key, state.step),
                with_ema(wide, None if state.ema is None else widened(state.ema)),
                **example_fields(state, 0, 1, batch))

    def loss(variables, batch):
        return objective.scalar_loss({**wide, "params": variables}, batch, step)[0]

    return loss, wide["params"]


def anchor_program(case, batch, state):
    """The same fp64 loss/gradient executable reused for each residual order."""
    import jax

    loss, params = anchor_loss(case, batch, state)
    return jax.jit(jax.value_and_grad(loss)).lower(params, batch).compile()


def _model_record(value):
    """`value` as a record its float64 twin shares: a module's or a dataclass's fields, its dtypes and
    a module's parent and name aside, every nested one the same way."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return (type(value).__name__, {field.name: _model_record(getattr(value, field.name))
                                       for field in dataclasses.fields(value)
                                       if field.name not in ("parent", "name", "dtype", "param_dtype")})
    if isinstance(value, (list, tuple)):
        return type(value)(_model_record(item) for item in value)
    if isinstance(value, dict):
        return {key: _model_record(item) for key, item in value.items()}
    return value


class ReferenceRefused(CouplingRefused):
    """The one-device reference or its float64 anchor refused (`CouplingRefused`): no layout of the
    model can be judged, so its row is INCONCLUSIVE, never a layout's MISMATCH."""


class Anchor:
    """The one-device reference's first step in float64 on each draw: the case's float64 twin
    (`build_objective(widened=True)`), its model's fields the native model's but its dtype, over the
    reference's own initial state widened, through the Step the trainer's first step makes, reading
    the reference's draws and boundaries from its record (`taped`), so it computes from exactly the
    sample the reference computed from, however a native prefix of its own would round it (EDM2's
    noised input, from a target it reads widened). Its loss is audited once (`audited`): the batch
    leaves it reads past its boundaries enter widened, exactly, and every floating equation past them
    is float64."""

    def __init__(self, case, label: str):
        import benchmark_models as bench
        import jax

        from dew.training.transaction import example_fields, with_ema

        sys.path.insert(0, str(REPO / "tests"))
        from reference_error import widened

        self.case, self.label = case, label
        state, _, _ = _trainer(case, {}, one_device=True).place()
        native, objective = bench.build_objective(case), bench.build_objective(case, widened=True)
        for mine, twin in zip(native.program_key(), objective.program_key(), strict=True):
            if _model_record(mine.module) != _model_record(twin.module):
                raise CouplingRefused(f"{label}: the float64 twin's {type(twin.module).__name__} is not "
                                      "the model's but its dtype")
        wide = widened(state.variables)
        first = drawn_batch(case, 0)
        self.places = example_fields(state, 0, 1, first)["example_index"]
        fields = {"step": state.microstep, "key": jax.random.fold_in(state.key, state.step),
                  "ema": with_ema(wide, None if state.ema is None else widened(state.ema)),
                  **example_fields(state, 0, 1, first)}
        self.loss, self.kinds = recording_loss(objective, wide, fields)
        self.params = wide["params"]
        read = audited(self.loss, self.kinds, self.params, first, label, wide=False)
        self.read = frozenset(read)
        audited(self.loss, self.kinds, self.params, self.entered(first), label, wide=True)
        self.program = jax.jit(jax.value_and_grad(self.loss, has_aux=True))

    def entered(self, batch):
        """`batch` as the anchor takes it: the floating leaves its loss reads past its boundaries
        widened to float64, exactly, the rest as drawn."""
        import jax
        import numpy as np

        def enter(path, leaf):
            leaf = np.asarray(leaf)
            wide = jax.tree_util.keystr(path) in self.read and np.issubdtype(leaf.dtype, np.floating)
            return leaf.astype(np.float64) if wide else leaf

        return jax.tree_util.tree_map_with_path(enter, batch)

    def __call__(self, seed: int, sample: Witnessed) -> Drawn:
        """The draw `seed`'s step, from `sample`, the reference's record of what that step drew and
        started from."""
        import jax
        import numpy as np

        (value, values), gradient = self.program(self.params, self.entered(drawn_batch(self.case, seed)),
                                                 taped(sample))
        record = {"values": values, "example_index": self.places}
        return Drawn(float(value), {jax.tree_util.keystr(path): np.asarray(leaf)
                                    for path, leaf in jax.tree_util.tree_flatten_with_path(gradient)[0]},
                     witnessed(_gathered(record), f"{self.label}'s float64 anchor, draw {seed}"))


def _decoded_tokens(case, batch):
    """The tokens a decoder's forward reads from `batch`, one row a sequence: a DPO pair's two rows,
    each without its last token, as the objective scores them."""
    import numpy as np

    from dew.data import preferences
    from dew.objectives.rl import sessions

    match case.decoder_objective:
        case "dpo":
            ids = np.asarray(batch[preferences.IDS_KEY])
            return ids.reshape(-1, ids.shape[-1])[:, :-1]
        case "grpo":
            return np.asarray(batch[sessions.IDS_KEY])[:, :-1]
        case "mdlm":
            return np.asarray(batch["text"])
    # A media row's tokens sit beside its conditioning.
    text = batch["text"]
    return np.asarray(getattr(text, "tokens", text))[:, :-1]


class Judge:
    """A model's one-device reference (`Arm`) and float64 anchor (`Anchor`) on each draw, run when check
    1 reads the draw, or read from `references` where it holds them, so a run of hundreds of draws
    holds one at a time; the anchor held to the reference's sample (`coupled`, the reference's record
    kept) and the reference's loss audited (`audited`) before any is judged. A model that
    selects discontinuously (`selection_margins.selects`) takes, for each seed, the first batch whose
    selections on the initial state clear `selection_margins.MARGIN` (`selection_margins.clearing`), so
    every float32 arm and the float64 anchor select alike, as a check-1 site's draw does."""

    def __init__(self, case, references: References, label: str):
        self.case, self.references, self.label = case, references, label
        self._arm: Arm | None = None
        self._anchor: Anchor | None = None
        self._batches: dict[int, int] = {}
        self._selections: tuple[Any, Callable] | None = None
        self._ahead: collections.OrderedDict[int, Any] = collections.OrderedDict()
        self._pool: Any = None

    def batch_seed(self, seed: int) -> int:
        """The seed of the batch the gate's draw `seed` takes: itself, or for a selecting model the
        first `selection_margins.clearing` gives whose selections clear the margin."""
        import benchmark_models as bench

        sys.path.insert(0, str(REPO / "tests"))
        import selection_margins

        if not (self.case.is_lm and selection_margins.selects(self.case.config)):
            return seed
        if seed not in self._batches:
            if self._selections is None and self.case.canvas:
                raise ReferenceRefused(
                    f"{self.label} selects discontinuously, and its selections are read from a plain "
                    "forward of its model, which a canvas objective's model (decoding against its "
                    "prompt's cache) does not run: its selection preflight is unbuilt")
            if self._selections is None:
                # The placed initial state and the compiled capture of its selections, once a judge: every
                # seed's batches are read against the same state by the same program.
                state, _, _ = _trainer(self.case, {}, one_device=True).place()
                (model, *_), *_ = bench.build_objective(self.case).program_key()
                self._selections = state.variables, selection_margins.capture(model)
            variables, forward = self._selections

            def clears(chosen):
                tokens = _decoded_tokens(self.case, drawn_batch(self.case, chosen))
                margin = selection_margins.margin(self.case.config, forward(variables, tokens), self.label)
                return margin > selection_margins.MARGIN

            self._batches[seed] = selection_margins.clearing(lambda chosen: chosen, clears)(seed)
        return self._batches[seed]

    @property
    def arm(self) -> Arm:
        if self._arm is None:
            self._arm = Arm(self.case, {}, one_device=True)
        return self._arm

    def audit(self) -> None:
        """Refuse a reference whose loss reads randomness outside its draws (`audited`)."""
        import benchmark_models as bench
        import jax

        from dew.training.transaction import example_fields, with_ema

        state, _, _ = _trainer(self.case, {}, one_device=True).place()
        first = drawn_batch(self.case, 0)
        fields = {"step": state.microstep, "key": jax.random.fold_in(state.key, state.step),
                  "ema": with_ema(state.variables, state.ema), **example_fields(state, 0, 1, first)}
        loss, kinds = recording_loss(bench.build_objective(self.case), state.variables, fields)
        audited(loss, kinds, state.variables["params"], first, self.label, wide=False)

    @property
    def flops_per_device(self) -> float | None:
        """The one-device reference's FLOPs, read with its draws where they are kept (`References.noted`),
        so a run of prepared draws compiles no reference."""
        return self.references.noted("flops", self.case, lambda: self.arm.compiled["flops_per_device"])

    AHEAD = 8
    """How many draws past the one read the reference and the anchor run ahead (`ahead`)."""

    def ahead(self, seed: int) -> None:
        """Draw `seed`'s reference and float64 anchor, and the `AHEAD` draws' after it, on two threads
        beside the caller, which runs a layout's draw meanwhile: each of a draw's three programs keeps
        about half of a CI task's four cores busy, so the three together run in about half the time.
        The pairs are held in the order they were last asked for, the least recently asked dropped past
        twice `AHEAD` once it has finished, so a pending draw is never dropped and recomputed; each
        layout of a model restarts its draws at 0 and finds the earlier layout's pairs still held, or
        held by `references`."""
        import concurrent.futures

        import jax

        if self._pool is None:
            self._pool = concurrent.futures.ThreadPoolExecutor(2)

        def both(at: int) -> tuple[Drawn, Drawn]:
            with jax.enable_x64(new_val=True):
                reference = self._drawn_reference(at)
                return reference, self._drawn_anchor(at, reference.record)

        for at in range(seed, seed + self.AHEAD + 1):
            if at in self._ahead:
                self._ahead.move_to_end(at)
            else:
                self._ahead[at] = self._pool.submit(both, at)
        finished = [at for at, pair in self._ahead.items() if pair.done()]
        for at in finished[:max(0, len(self._ahead) - 2 * self.AHEAD)]:
            del self._ahead[at]

    def _from_ahead(self, seed: int, anchor: bool) -> Drawn:
        """Draw `seed`'s reference, or its anchor where `anchor`: ahead (`ahead`) where it runs there,
        else drawn here."""
        if seed in self._ahead:
            return self._ahead[seed].result()[anchor]
        return self._drawn_anchor(seed) if anchor else self._drawn_reference(seed)

    def reference(self, seed: int) -> Drawn:
        return self._from_ahead(seed, anchor=False)

    def anchor(self, seed: int) -> Drawn:
        return self._from_ahead(seed, anchor=True)

    def _drawn_reference(self, seed: int) -> Drawn:
        chosen = self.batch_seed(seed)
        return self.references.held("reference", self.case, chosen, lambda: self.arm(chosen))

    def _drawn_anchor(self, seed: int, witnessed: Witnessed | None = None) -> Drawn:
        """Draw `seed`'s float64 anchor from `witnessed`, the reference's record of that draw (drawn
        here where not given), held by the caller beside the reference rather than kept here."""
        chosen = self.batch_seed(seed)
        if witnessed is None:
            witnessed = self._drawn_reference(seed).record

        def computed():
            if self._anchor is None:
                self._anchor = Anchor(self.case, self.label)
            return self._anchor(chosen, witnessed)

        try:
            drawn = self.references.held("anchor", self.case, chosen, computed)
            coupled(witnessed, drawn.record, f"{self.label}'s float64 anchor, draw {chosen}")
        except CouplingRefused as refusal:
            raise ReferenceRefused(str(refusal)) from refusal
        return drawn


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


CACHE_SCHEMA = 2
"""The layout of a cached draw (`References`): a gradient, its loss and its record; 1 held a
reference's floors."""


def _cache_name(kind: str, case, seed: int) -> tuple[str, dict[str, Any]]:
    """The file a cached `kind` of `case`'s draw `seed` is kept in, and what it depends on: the case,
    the draw and its batch, the producer and, for the one-device reference, the device and backend
    that rounded it, jax, x64 and Dew's source. A float64 anchor is not assumed bitwise portable
    either, so its backend is in its key too."""
    import importlib.metadata

    import jax
    import numpy as np
    from jax.extend import backend

    digest = hashlib.sha256()
    for leaf in jax.tree.leaves(drawn_batch(case, seed)):
        array = np.ascontiguousarray(leaf)
        digest.update(f"{array.dtype}{array.shape}".encode())
        digest.update(array.tobytes())
    from dew.nn.precision import wide_cpu_arithmetic

    inputs = {"schema": CACHE_SCHEMA, "kind": kind, "case": dataclasses.asdict(case), "seed": seed,
              "batch": digest.hexdigest(), "jax": jax.__version__,
              "jaxlib": importlib.metadata.version("jaxlib"),
              "x64": jax.config.jax_enable_x64, "wide_cpu_arithmetic": wide_cpu_arithmetic(),
              "source": source_digest(), "device": jax.local_devices()[0].device_kind,
              "backend": backend.get_backend().platform_version}
    name = hashlib.sha256(json.dumps(inputs, sort_keys=True, default=str).encode()).hexdigest()[:24]
    return f"{kind}-{name}.npz", inputs


def _write(path: Path, drawn: Drawn, meta: dict[str, Any]) -> None:
    """`drawn` and `meta` in one file, written whole or not at all, every array at its own dtype."""
    import numpy as np

    from dew.files import replacing

    arrays = {f"gradient/{name}": value for name, value in drawn.gradient.items()}
    arrays |= {f"record/{name}/{place}": value for name, examples in drawn.record.items()
               for place, value in examples.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with replacing(path) as partial, partial.open("wb") as file:
        # Stored as arr_0 (the metadata), arr_1, ... in the names' order.
        np.savez(file, np.array(json.dumps({**meta, "loss": drawn.loss, "names": list(arrays)})),
                 *[np.asarray(array) for array in arrays.values()])


def _read(path: Path) -> Drawn:
    import numpy as np

    with np.load(path) as stored:
        meta = json.loads(str(stored["arr_0"]))
        arrays = {name: stored[f"arr_{index}"] for index, name in enumerate(meta["names"], start=1)}
    record: Witnessed = {}
    for name, value in arrays.items():
        if name.startswith("record/"):
            kept, place = name.removeprefix("record/").rsplit("/", 1)
            record.setdefault(kept, {})[int(place)] = value
    return Drawn(meta["loss"], {name.removeprefix("gradient/"): value for name, value in arrays.items()
                                if name.startswith("gradient/")}, record)


@dataclasses.dataclass(frozen=True)
class References:
    """Where a model's reference and float64 anchor draws come from: computed here, or, with a
    `directory`, read from the files a `prepare` run wrote there (`_cache_name`); see the module
    docstring."""

    directory: Path | None = None
    prepare: bool = False
    reserve: int | None = None
    """With `prepare`, the bytes a kept draw must leave free on the directory's filesystem: past them a
    draw is computed and not kept, so a cache that one task's cells share never fills its disk (moe_grpo's
    4096 draws filled a CI task's 6 GB tmpfs); None keeps every draw, as a prepare run for a later run
    must."""

    def held(self, kind: str, case, seed: int, compute: Callable[[], Drawn]) -> Drawn:
        """Draw `seed`'s `kind` of `case`: computed, or with a directory, read where it holds it, else
        computed and kept there by a prepare run while its disk keeps `reserve` free; a run of layouts
        that finds it missing is refused."""
        import jax
        import numpy as np

        if self.directory is None:
            return compute()
        name, inputs = _cache_name(kind, case, seed)
        path = self.directory / name
        if path.exists():
            return _read(path)
        if not self.prepare:
            raise FileNotFoundError(
                f"no {kind} of this case's draw {seed} in {self.directory} ({name}); compute it first with "
                f"--prepare in a job of one device")
        drawn = compute()
        size = sum(np.asarray(leaf).nbytes for leaf in jax.tree.leaves((drawn.gradient, drawn.record)))
        if self.reserve is None or shutil.disk_usage(self.directory).free - size > self.reserve:
            _write(path, drawn, {"inputs": inputs})
        return drawn

    def noted(self, kind: str, case, compute: Callable[[], Any]) -> Any:
        """A model's `kind` of JSON value kept beside its draws (the reference's FLOPs): computed, or with
        a directory, read where it holds it, else computed and, by a prepare run, kept there."""
        if self.directory is None:
            return compute()
        name, inputs = _noted_name(kind, case)
        path = self.directory / name
        if path.exists():
            return json.loads(path.read_text())["value"]
        value = compute()
        if self.prepare:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"inputs": inputs, "value": value}))
        return value


def _noted_name(kind: str, case) -> tuple[str, dict[str, Any]]:
    name, inputs = _cache_name(kind, case, 0)
    return str(Path(name).with_suffix(".json")), inputs


def judged(model: str, name: str, arm: Arm, judge: Judge, seeds: int, window: int = 1) -> dict[str, Any]:
    """`arm`'s verdict against `judge`'s draws, check 1's (`assert_as_exact`): each gradient leaf and
    the loss within PARITY of the one-device reference's error from the float64 anchor, over `seeds`
    draws and the redraw it takes, each layout draw first held to the reference's sample (`coupled`).
    With a `window` of more than one, each of check 1's draws is that many consecutive draws, judged
    as one (`over_windows`): a model whose single draws' errors are heavy-tailed, as a loss summed over
    terms that nearly cancel, is resolved over windows, as the gate's windowed sites are.
    `works`, MISMATCH (over PARITY, or a sample apart, naming what) or INCONCLUSIVE (unresolved, or the
    model's reference refused), never passing."""
    import numpy as np

    sys.path.insert(0, str(REPO / "tests"))
    from reference_error import (
        GateFailure,
        Inconclusive,
        as_given,
        assert_as_exact,
        dew_programs,
        over_windows,
    )

    label = f"{model} on {name}"

    @dew_programs
    def layout(seed):
        judge.ahead(int(seed))
        drawn = arm(judge.batch_seed(int(seed)))
        coupled(judge.reference(int(seed)).record, drawn.record, f"{label}, draw {int(seed)}")
        return drawn.leaves()

    try:
        # Every arm runs a trainer's compiled step, or the anchor's float64 program, of its own.
        def windowed(fn):
            return fn if window == 1 else over_windows(fn)

        def draw(seed):
            return ((np.asarray(seed),) if window == 1
                    else (np.arange(seed * window, (seed + 1) * window),))

        def read(drawn):
            # Whichever arm check 1 reads first starts the draws ahead (`Judge.ahead`).
            def of(seed):
                judge.ahead(int(seed))
                return drawn(int(seed)).leaves()
            return of

        reference = dew_programs(read(judge.reference))
        assert_as_exact(windowed(layout), windowed(reference), draw, label, seeds=seeds,
                        truth_fn=windowed(as_given(read(judge.anchor))))
    except ReferenceRefused as refusal:
        return {"status": "INCONCLUSIVE", "reason": f"the model's reference is refused: {refusal}"[:2000]}
    except CouplingRefused as refusal:
        return {"status": "MISMATCH", "reason": str(refusal)[:2000]}
    except Inconclusive as unresolved:
        return {"status": "INCONCLUSIVE", "reason": str(unresolved)[:2000]}
    except GateFailure as over:
        return {"status": "MISMATCH", "reason": str(over)[:2000]}
    return {"status": "works"}


def ran(model: str, name: str, arm: Arm, judge: Judge, reference: Drawn) -> dict[str, Any]:
    """`arm` run once, on the first draw, its sample held to `reference`'s, the one-device reference's
    first draw (`coupled`), and its loss and every gradient leaf finite where the reference's are:
    whether the layout runs where the model's reference does, its accuracy left to `judged`'s many
    draws (check 1). `works`, or MISMATCH where its sample is apart or it computes a non-finite value
    the reference does not."""
    import numpy as np

    drawn = arm(judge.batch_seed(0))
    try:
        coupled(reference.record, drawn.record, f"{model} on {name}, draw 0")
    except CouplingRefused as refusal:
        return {"status": "MISMATCH", "reason": str(refusal)[:2000]}
    theirs, mine = reference.leaves(), drawn.leaves()
    broken = sorted(leaf for leaf, value in mine.items()
                    if np.all(np.isfinite(theirs[leaf])) and not np.all(np.isfinite(value)))
    if broken:
        return {"status": "MISMATCH", "reason": f"{model} on {name}, draw 0: not finite where the one-device "
                                                f"reference is: {', '.join(broken[:8])}"}
    return {"status": "works"}


FLOOR_FACTOR = 4.0
"""How far past its floor a layout's leaf or loss may sit under the floor judge (`floored`)."""
PERMUTATIONS = 16
"""The row orders the floor judge's permutation floor draws (`permutation_floor`)."""


_floor_held: dict[tuple[str, str, int | None, str], tuple[Any, ...]] = {}
"""Each floor-judged model's batch, one-device first step and floors by its dtype and device count, as
the first layout's `floored` computed them in this process: a model's layouts judge against the same
reference, so its permutation floor's runs and float64 anchor run once for them all."""


def _floor_reference(model: str, case, reference, dtype: str, devices: int | None) -> tuple[Any, ...]:
    """`model`'s batch, its one-device first step's losses and gradient, each leaf's floor and the
    loss's, and the step's FLOPs a device (`_floor_held`): the reference's own deviation under
    reassociations of its batch (`permutation_floor`, or data parallelism's for an objective that
    draws by row), widened by its distance from the float64 step (`anchor_step`), whose rounding a
    reference past `rounding_limit` refuses, as does a floor past it (`widest_floor`)."""
    import benchmark_models as bench

    # The case as resolved, its mixture and objective among it, as `_initial` keys a reference.
    key = (model, dtype, devices, repr(case))
    if key not in _floor_held:
        batch = bench.global_batch(case)
        ref_losses, ref_gradient, compiled = trained(reference, {}, batch, steps=1, one_device=True)
        if reassociates(reference):
            floors, loss_floor = permutation_floor(reference, batch, ref_gradient, ref_losses[0])
        else:
            floors, loss_floor = data_parallel_floor(reference, batch, ref_gradient, ref_losses[0],
                                                     devices)
        anchored, anchor_gradient = anchor_step(reference, batch)
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
        loss_floor = max(loss_floor, abs(ref_losses[0] - anchored))
        widest_floor(floors, dtype)
        _floor_held[key] = (batch, ref_losses, ref_gradient, floors, loss_floor,
                            compiled["flops_per_device"])
    return _floor_held[key]


def floored(model: str, layouts: Sequence[str], *, dtype: str, devices: int | None = None,
            mixture: dict[str, Any] | None = None, objective: dict[str, Any] | None = None,
            out: Path = Path("out/layout-parity-orders"),
            speak: Callable[[str], None] = lambda line: None) -> list[dict[str, Any]]:
    """Each of `layouts`' first step on `model` held to the one-device reference by the floor judge
    (`_floored`), a row each; a model with residual orders judged over them once for all its layouts
    instead (`run_orders`, its files under `out`), as `run` judges it. `mixture` and `objective` are
    merged into the model's as `run` merges them."""
    import jax

    mixture, objective = mixture or {}, objective or {}
    if zoo()[model].orders:
        with jax.enable_x64(new_val=True):
            return run_orders([model], tuple(layouts), dtype=dtype, devices=devices, mixture=mixture,
                              objective=objective, out=out, speak=speak)
    return [_floored(model, layout, dtype=dtype, devices=devices, mixture=mixture, objective=objective)
            for layout in layouts]


def _floored(model: str, layout: str, *, dtype: str, devices: int | None, mixture: dict[str, Any],
             objective: dict[str, Any]) -> dict[str, Any]:
    """`layout`'s first step on `model` held to the one-device reference by the floor judge: each
    gradient leaf and the loss within FLOOR_FACTOR of the reference's floors (`_floor_reference`),
    and its devices' FLOPs to an even split (`flops_bound`), REDUNDANT past it. A layout Dew refuses
    by design is a refused row, and a failing layout, or a reference no layout can be judged by, an
    error row, as `run` reports them."""
    import jax

    from dew._coordination import agreed
    from dew.mesh import LayoutRefused

    case, reference = model_case(model, dtype, mixture, objective)
    try:
        batch, ref_losses, ref_gradient, floors, loss_floor, reference_flops = agreed(
            f"reference of {model}",
            lambda: _floor_reference(model, case, reference, dtype, devices))
    except Exception as error:  # no reference judges no layout: the model's one row
        return {"model": model, "layout": "reference", "processes": jax.process_count(),
                "status": "error", "error": f"{type(error).__name__}: {error}"[:2000],
                "traceback": traceback.format_exc()[-4000:]}
    row: dict[str, Any] = {"model": model, "mixture": case.config.get("mixture"),
                           "objective": case.objective, "layout": layout,
                           "processes": jax.process_count(), "reference_losses": ref_losses}
    started = time.perf_counter()
    try:
        losses, gradient, compiled = agreed(
            f"{model} on {layout}",
            lambda: trained(case, LAYOUTS[layout], batch, steps=1, devices=devices))
        row.update(compiled, losses=losses, **floor_judged(
            leaf_errors(ref_gradient, gradient, dtype), floors, abs(losses[0] - ref_losses[0]),
            loss_floor, ref_losses[0], contracted_terms(batch)))
        count_flops(row, compiled, LAYOUTS[layout], reference_flops)
    except LayoutRefused as refusal:
        row.update(status="refused", reason=str(refusal))
    except Exception as error:  # a failing layout is a row of the matrix
        row.update(status="error", error=f"{type(error).__name__}: {error}"[:2000],
                   traceback=traceback.format_exc()[-4000:])
    row["seconds"] = round(time.perf_counter() - started, 1)
    return row


def reassociates(case) -> bool:
    """Whether reordering or pooling the batch's rows only reassociates the step's sums: a next-token
    loss draws nothing per row, a diffusion objective or a masked-diffusion decoder draws by row."""
    return case.is_lm and case.canvas is None and case.decoder_objective != "mdlm"


def reordered(batch, seed: int):
    """The batch with its rows reversed (seed 0) or drawn in another order."""
    import jax
    import numpy as np

    rows = len(jax.tree.leaves(batch)[0])
    order = np.arange(rows)[::-1] if seed == 0 else np.random.default_rng(seed).permutation(rows)
    return jax.tree.map(lambda leaf: np.asarray(leaf)[order], batch)


def strided(batch, pieces: int):
    """The batch with rows m, m + pieces, ... gathered into its m-th consecutive slice: a pipeline's
    microbatch m."""
    import jax
    import numpy as np

    rows = len(jax.tree.leaves(batch)[0])
    order = np.concatenate([np.arange(start, rows, pieces) for start in range(pieces)])
    return jax.tree.map(lambda leaf: np.asarray(leaf)[order], batch)


def pooled(case, batches, pieces: int) -> list[tuple[float, dict[str, NDArray]]]:
    """Step one's loss and gradient on one device for each batch, each pooled from `pieces`
    consecutive slices of its rows by accumulation, from one initial state through one compiled step."""
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
            state, loss, *_ = step(state, data)
            losses.append(float(loss))
        results.append((float(np.mean(losses)), _gradient(state)))
    return results


def leaf_errors(reference: Mapping[str, NDArray], other: Mapping[str, NDArray],
                dtype: str) -> dict[str, float]:
    """Each leaf's L2 distance from the reference over the reference leaf's norm, or over the compute
    dtype's rounding of the whole gradient's norm where the leaf is smaller than that."""
    import numpy as np

    if reference.keys() != other.keys():
        apart = sorted(reference.keys() ^ other.keys())[:6]
        raise ValueError(f"the gradients hold different leaves: {apart}")
    pairs = [(name, np.asarray(want, np.float64), np.asarray(other[name], np.float64))
             for name, want in reference.items()]
    noise = rounding_limit(dtype) * float(np.sqrt(sum(np.sum(want ** 2) for _, want, _ in pairs)))
    return {name: float(np.linalg.norm(got - want) / max(float(np.linalg.norm(want)), noise))
            for name, want, got in pairs}


def data_parallel_floor(case, batch, reference: Mapping[str, NDArray], reference_loss: float,
                        devices: int | None = None) -> tuple[dict[str, float], float]:
    """Per leaf, the deviation from the reference of data parallelism over every device of the run, and
    the same of the step-one loss: the floor of an objective that draws noise by row."""
    losses, gradient, _ = trained(case, {}, batch, steps=1, devices=devices)
    return leaf_errors(reference, gradient, case.dtype), abs(losses[0] - reference_loss)


def permutation_floor(case, batch, reference: Mapping[str, NDArray],
                      reference_loss: float) -> tuple[dict[str, float], float]:
    """Per leaf, the largest deviation of the reference from itself under a reassociation of the batch's
    sums, and the same of the step-one loss, on the reference's own device."""
    runs = pooled(case, [reordered(batch, seed) for seed in range(PERMUTATIONS)], 1)
    loss = max(abs(moved - reference_loss) for moved, _ in runs)
    for pieces in (2, 4):
        runs += pooled(case, [batch, strided(batch, pieces)], pieces)
    leaves: dict[str, float] = {}
    for _, gradient in runs:
        for leaf, error in leaf_errors(reference, gradient, case.dtype).items():
            leaves[leaf] = max(leaves.get(leaf, 0.0), error)
    return leaves, loss


def anchor_step(case, batch) -> tuple[float, dict[str, NDArray]]:
    """Step one's loss and its gradient by leaf name in fp64, the model's float64 twin on the
    reference's own initial variables widened."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    state, _, _ = _trainer(case, {}, one_device=True).place()
    wide = jax.tree.map(lambda leaf: leaf.astype(jnp.float64)
                        if jnp.issubdtype(leaf.dtype, jnp.floating) else leaf, state.variables["params"])
    value, gradient = anchor_program(case, batch, state)(wide, batch)
    return float(value), {jax.tree_util.keystr(path): np.asarray(leaf)
                          for path, leaf in jax.tree_util.tree_flatten_with_path(gradient)[0]}


def rounding_limit(dtype: str) -> float:
    """The farthest the reference may sit from the fp64 step, and the widest a floor may be: the square
    root of the compute dtype's machine epsilon."""
    import jax.numpy as jnp

    return float(jnp.finfo(dtype).eps) ** 0.5


def widest_floor(floors: dict[str, float], dtype: str) -> str:
    """The leaf with the widest floor, refused past `rounding_limit`."""
    widest = max(floors, key=floors.__getitem__)
    limit = rounding_limit(dtype)
    if floors[widest] > limit:
        raise ValueError(f"the floor at {widest} is {floors[widest]:.2e}, past {dtype} rounding "
                         f"({limit:.1e}), so no layout of this model can be judged by it")
    return widest


def contraction_floor(terms: int, magnitude: float) -> float:
    """How far two fp32 sums of the same `terms` products may land apart, whose magnitudes add to
    `magnitude`: twice gamma_N times it."""
    import numpy as np

    u = float(np.finfo(np.float32).eps) / 2
    return 2 * terms * u / (1 - terms * u) * magnitude


def contracted_terms(batch) -> int:
    """The terms a step's loss sums, at most: the elements of the batch's largest leaf."""
    import jax
    import numpy as np

    return max(int(np.size(leaf)) for leaf in jax.tree.leaves(batch))


def floor_judged(errors: dict[str, float], floors: dict[str, float], loss: float, loss_floor: float,
                 reference_loss: float, terms: int) -> dict[str, Any]:
    """Every leaf against FLOOR_FACTOR times its floor, fp32 epsilon at least, and the loss against
    FLOOR_FACTOR times its floor or the `contraction_floor` of its `terms`, whichever is wider."""
    import numpy as np

    eps = float(np.finfo(np.float32).eps)
    ratios = {leaf: error / (FLOOR_FACTOR * max(floors[leaf], eps)) for leaf, error in errors.items()}
    worst = max(ratios, key=ratios.__getitem__)
    loss_bound = max(FLOOR_FACTOR * max(loss_floor, eps * abs(reference_loss)),
                     contraction_floor(terms, abs(reference_loss)))
    return {"worst_leaf": worst, "worst_ratio": ratios[worst], "loss_error": loss, "loss_bound": loss_bound,
            "status": "works" if ratios[worst] <= 1.0 and loss <= loss_bound else "MISMATCH"}


def flops_bound(fields: Mapping[str, int], replicated: float = 0.0) -> float:
    """The FLOPs a layout's devices may compute together over one device's:
    an even split, and a pipeline of S stages and M microbatches holds each
    stage's work for M + S - 1 microbatches' time (GPipe's bubble) and runs
    the `replicated` share of the model's work, its layers outside the run
    the stages repeat (`CausalTransformer.replicated_flops`), on every stage,
    times FLOPS_SLACK."""
    stages, microbatches = fields.get("stage", 1), fields.get("microbatches", 1)
    return ((1 - replicated) * (microbatches + stages - 1) / microbatches + replicated * stages) * FLOPS_SLACK


def count_flops(row: dict[str, Any], compiled: Mapping[str, Any], fields: Mapping[str, int],
                reference_flops: float | None) -> None:
    """`row` takes its layout's FLOPs against one device's: its devices' together over the
    reference's `reference_flops`, held to `flops_bound`, REDUNDANT past it where the row works."""
    if not (compiled["flops_per_device"] and reference_flops):
        return
    # Above one when devices compute what one device need not.
    ratio = compiled["flops_per_device"] * math.prod(compiled["mesh"].values()) / reference_flops
    row.update(flops_ratio=ratio, flops_bound=flops_bound(fields, compiled["replicated"]))
    if row["status"] == "works" and ratio > row["flops_bound"]:
        row["status"] = "REDUNDANT"


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
                     loss: float, truth_loss: float, *, roundings: int) -> float:
    """A residual permutation preserves the entire fp64 gradient and loss, each within float64's
    rounding of the computation (`reference_error.assert_computes_the_oracle`) over `roundings`, its
    program's longest chain (`chain_roundings`); a value that is not finite fails it. Returns the
    gradient's distance relative to its scale.

    The gradient is one vector over all parameters, so cancelled leaves are
    measured on its scale instead of being divided by their own near-zero norm.
    This is the exact-symmetry check required before an order counts as a
    rounding draw; it is independent of the fp32 K-order acceptance bound.
    """
    import numpy as np

    sys.path.insert(0, str(REPO / "tests"))
    from reference_error import assert_computes_the_oracle

    if actual.keys() != truth.keys():
        raise ValueError('float64 orders hold different leaves')
    ordered, expected = (np.concatenate([np.ravel(tree[name]) for name in truth]) for tree in (actual, truth))
    if not (np.isfinite(ordered).all() and np.isfinite(loss)):
        raise ValueError('a float64 residual order computes a value that is not finite')
    assert_computes_the_oracle(ordered, expected, 'float64 residual order', roundings=roundings)
    assert_computes_the_oracle(np.asarray([loss]), np.asarray([truth_loss]), 'float64 residual order loss',
                               roundings=roundings)
    apart, scale = float(np.linalg.norm(ordered - expected)), float(np.linalg.norm(expected))
    return apart / scale if scale else 0.0


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
    from reference_error import ORDERS, chain_roundings, distance
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
        anchor, wide = anchor_loss(reference, batch, base)
        roundings = chain_roundings(jax.make_jaxpr(jax.value_and_grad(anchor))(wide, batch))
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
            invariance.append(order_invariance(truth, identity, truth_loss, identity_loss,
                                               roundings=roundings))
            ref_state, ref_loss, _, _, _, _ = ref_step(ordered_state(base, order, ref_shardings), ref_data)
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
                state, loss, _, _, _, _ = step(ordered_state(template, order, shardings), data)
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
            if compiled['flops_per_device']:
                count_flops(row, compiled, LAYOUTS[name], ref_compiled['flops_per_device'])
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


def prepared(models: Sequence[str], *, dtype: str, seeds: int | Mapping[str, int], mixture: dict[str, Any],
             objective: dict[str, Any], references: References, speak: Callable[[str], None],
             windows: int | Mapping[str, int] = 1) -> bool:
    """Each model's reference and float64 anchor on every draw check 1 may read, its seeds and the
    redraw's (`REDRAW`), each a window of draws where the model's window is more than one, or with no
    seeds the reference's first draw alone (`ran`), with the reference's FLOPs, computed into
    `references` where missing; whether every model's is there."""
    sys.path.insert(0, str(REPO / "tests"))
    from reference_error import REDRAW

    whole = True
    for model in models:
        started = time.perf_counter()
        count = seeds if isinstance(seeds, int) else seeds[model]
        draws = (1 + REDRAW) * count * (windows if isinstance(windows, int) else windows.get(model, 1))
        try:
            _, reference = model_case(model, dtype, mixture, objective)
            judge = Judge(reference, references, model)
            judge.audit()
            flops = judge.flops_per_device
            if count == 0:
                judge.reference(0)
            for seed in range(draws):
                judge.anchor(seed)
        except Exception as error:  # the other models' references are still worth keeping
            whole = False
            speak(f"[{model}] reference error {type(error).__name__}: {error}"[:400])
            continue
        speak(f"[{model}] {draws or 1} draws, the reference's {flops} FLOPs a device, "
              f"{time.perf_counter() - started:.0f} s")
    return whole


def run(models: Sequence[str], layouts: Sequence[str], *, dtype: str, seeds: int | Mapping[str, int],
        mixture: dict[str, Any], objective: dict[str, Any], references: References,
        speak: Callable[[str], None], keep: Callable[[list[dict[str, Any]]], None],
        devices: int | None = None, orders_out: Path | None = None,
        windows: int | Mapping[str, int] = 1) -> list[dict[str, Any]]:
    """Every layout of every model, one row each, `keep` handed the rows so far after each, each layout
    over the process's first `devices` devices (every device by default) and judged over the model's
    `seeds` draws (`judged`), or with none run once, its sample held to the reference's (`ran`). A
    reference and a layout run as agreed phases: a failure on one process fails that row on every
    process, or, where the others wait in a collective it left, ends the pool within the failure grace
    (dew.artifacts) with the rows kept so far."""
    import jax

    from dew._coordination import agreed
    from dew.mesh import LayoutRefused

    rows = []
    for model in models:
        case, reference = model_case(model, dtype, mixture, objective)
        count = seeds if isinstance(seeds, int) else seeds[model]
        # A run of no draws runs each layout once (`ran`), a model with residual orders as any other.
        if case.orders and count:
            with jax.enable_x64(new_val=True):
                rows.extend(run_orders(
                    [model], layouts, dtype=dtype, devices=devices, mixture=mixture, objective=objective,
                    out=Path('out/layout-parity-orders') if orders_out is None else orders_out, speak=speak))
            keep(rows)
            continue
        window = windows if isinstance(windows, int) else windows.get(model, 1)
        judge = Judge(reference, references, model)

        def first_draw() -> Drawn:
            # A run of no draws holds each layout's sample to the reference's alone: no anchor is built.
            judge.audit()
            return judge.reference(0) if count == 0 else judge.anchor(0)

        try:
            first = agreed(f"reference of {model}", first_draw)
        except Exception as error:  # no reference judges no layout: the model's one row
            refused = isinstance(error, CouplingRefused)
            rows.append({"model": model, "layout": "reference", "processes": jax.process_count(),
                         "status": "INCONCLUSIVE" if refused else "error",
                         ("reason" if refused else "error"): f"{type(error).__name__}: {error}"[:2000],
                         "traceback": traceback.format_exc()[-4000:]})
            speak(f"[{model}] reference {rows[-1]['status']}: {str(error)[:300]}")
            keep(rows)
            continue
        widened = f", float64 {first.loss}" if count else ""
        speak(f"[{model}] reference loss {first.loss if count == 0 else judge.reference(0).loss}"
              f"{widened}, {count} draws")
        for name in layouts:
            row: dict[str, Any] = {"model": model, "mixture": case.config.get("mixture"),
                                   "objective": case.objective, "layout": name, "seeds": count,
                                   "processes": jax.process_count()}
            started = time.perf_counter()
            try:
                arm = agreed(f"{model} on {name}", lambda: Arm(case, LAYOUTS[name], devices=devices))
                row.update(arm.compiled, **agreed(f"{model} on {name}, its draws", lambda: (
                    ran(model, name, arm, judge, first) if count == 0
                    else judged(model, name, arm, judge, count, window))))
                if arm.compiled["flops_per_device"]:
                    count_flops(row, arm.compiled, LAYOUTS[name], judge.flops_per_device)
            except LayoutRefused as refusal:
                row.update(status="refused", reason=str(refusal))
            except Exception as error:  # a failing layout is a row of the matrix
                row.update(status="error", error=f"{type(error).__name__}: {error}"[:2000],
                           traceback=traceback.format_exc()[-4000:])
            row["seconds"] = round(time.perf_counter() - started, 1)
            rows.append(row)
            keep(rows)
            judged_whole = row["status"] in ("works", "REDUNDANT")
            said = (f"over {count} draws, flops x{row.get('flops_ratio', float('nan')):.2f} "
                    f"of x{row.get('flops_bound', float('nan')):.2f}" if judged_whole
                    else str(row.get("reason", row.get("error", "")))[:300])
            speak(f"[{model}/{name}] {row['status']} {said}")
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
         dtype: str = "float32",
         seeds: Annotated[int | None, tyro.conf.arg(
             help="the draws check 1 judges each model over; 0 runs each layout once; unset, each "
                  "layout is judged by the floor judge CI runs (`floored`)")] = None,
         window: Annotated[int, tyro.conf.arg(
             help="consecutive draws each of check 1's draws holds")] = 1,
         rounding_orders: Annotated[bool, tyro.conf.arg(
             help="52 exact residual orders against fp64; one step, --out is a directory; Colab")] = False,
         out: Path | None = None,
         mixture: Annotated[str, tyro.conf.arg(
             help="JSON merged into each model's mixture, e.g. '{\"dispatch\": \"exchange\"}'")] = "{}",
         objective: Annotated[str, tyro.conf.arg(
             help="JSON of LMObjective keywords, e.g. '{\"balance\": {\"aux_loss_alpha\": 0.01}}'")] = "{}",
         references: Annotated[Path | None, tyro.conf.arg(
             help="a directory of references a --prepare run wrote, read instead of computed")] = None,
         prepare: Annotated[bool, tyro.conf.arg(
             help="compute each model's missing draws into --references and run no layout")] = False,
         devices: Annotated[int | None, tyro.conf.arg(
             help="run each layout on this process's first N devices: four stages on four devices "
                  "leave the rows unsplit, which eight, putting the rest on data, never show")] = None,
         ) -> None:
    """Run the layouts of each model against one device; see the module docstring."""
    from dew.training.runtime import prepare_process

    prepare_process()
    import jax

    # The float64 anchor computes in float64; every arm names its own dtypes.
    jax.config.update("jax_enable_x64", val=True)
    if prepare and (references is None or jax.process_count() > 1):
        raise SystemExit("--prepare writes the --references directory from one process")
    if devices is not None and (jax.process_count() > 1 or not 0 < devices <= jax.device_count()):
        raise SystemExit(f"--devices takes the first N of one process's {jax.device_count()} devices; "
                         "a pool runs every layout on all of its devices")
    jax.config.update("jax_default_matmul_precision", "highest")
    speaker = jax.process_index() == 0
    if rounding_orders:
        if out is None or prepare or references is not None:
            raise SystemExit("--rounding-orders needs --out DIR; --prepare and --references are the "
                             "draws' path")
        rows = run_orders(models, layouts, dtype=dtype, devices=devices, mixture=json.loads(mixture),
                          objective=json.loads(objective), out=out,
                          speak=lambda line: print(line, flush=True))
        print('\n'.join(summary(rows)), flush=True)
        raise SystemExit(verdict(rows))
    def keep(rows: list[dict[str, Any]]) -> None:
        if speaker and out is not None:
            out.write_text(json.dumps(rows, indent=1))

    mixed, aimed = json.loads(mixture), json.loads(objective)
    orders_out = Path('out/layout-parity-orders') if out is None else out.parent / f'{out.stem}-orders'
    if seeds is None:
        # The floor judge, which CI's matrix and its multi-host gangs run until check 1's draws shard
        # across tasks (`floored`): a layout at a time, through agreed phases on a pool, a model with
        # residual orders over them once for all its layouts (`run_orders`), as `run` judges it.
        if prepare or references is not None or window != 1:
            raise SystemExit("--prepare, --references and --window are check 1's draws; give --seeds")
        rows: list[dict[str, Any]] = []
        for model in models:
            rows += floored(model, layouts, dtype=dtype, devices=devices, mixture=mixed, objective=aimed,
                            out=orders_out, speak=lambda line: print(line, flush=True) if speaker else None)
            keep(rows)
        if speaker:
            print("\n".join(summary(rows)), flush=True)
        raise SystemExit(verdict(rows))
    store = References(references, prepare)
    if prepare:
        if not prepared(models, dtype=dtype, seeds=seeds, windows=window, mixture=mixed, objective=aimed,
                        references=store,
                        speak=lambda line: print(line, flush=True)):
            raise SystemExit(1)
        return

    rows = run(models, layouts, dtype=dtype, seeds=seeds, windows=window, mixture=mixed, objective=aimed,
               references=store, speak=lambda line: print(line, flush=True) if speaker else None, keep=keep,
               devices=devices, orders_out=orders_out)
    if speaker:
        print("\n".join(summary(rows)), flush=True)
    raise SystemExit(verdict(rows))


if __name__ == "__main__":
    tyro.cli(main)
