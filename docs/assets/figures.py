"""Draw the docs' and the tutorials' figures.

The docs' figures come in a light and a dark variant. The tutorials' figures
are one file each, whose colors follow the theme of the page that shows them
(dew.figures), and most of them animate.

The sharding and diffusion figures are computed by Dew itself: the mesh
figure reads the placements `MeshSpec.build`, `Layout` and `batch_shardings`
give on eight simulated CPU devices, and the diffusion figure noises an
image with the rates of the `Cosine` and `Flow` presets. The data and
training-step figures draw the order of calls in `Trainer.fit`
(src/dew/training/trainer.py) and in its compiled step
(src/dew/training/transaction.py).

    XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
        python docs/assets/figures.py [NAME_PREFIX ...]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=8")

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

from dew.diffusion.presets import Cosine, Flow
from dew.figures import THEMES, Figure
from dew.nn.backbones import CausalTransformer, Mixture
from dew.training import Layout, MeshSpec
from dew.training.distributed import batch_shardings

HERE = Path(__file__).resolve().parent


def box(svg: Figure, x, y, w, h, title, lines=(), accent=False):
    svg.rect(x, y, w, h, fill="accent_fill" if accent else "surface", stroke="accent" if accent else "rule")
    svg.text(x + 12, y + 23, title, size=14, mono=True, weight=600)
    for i, line in enumerate(lines):
        svg.text(x + 12, y + 45 + 19 * i, line, size=13, color="muted")


def mesh_figure():
    """A (data=2, fsdp=2, tensor=2) mesh, a batch and an MLP kernel placed on it."""
    mesh = MeshSpec(fsdp=2, tensor=2).build()
    ids = np.vectorize(lambda d: d.id)(mesh.devices)  # (data, expert, fsdp, tensor, sequence, stage)
    coords = {int(ids[d, 0, f, k, 0, 0]): (d, f, k) for d in range(2) for f in range(2) for k in range(2)}

    model = CausalTransformer(vocab_size=512, emb_features=256, num_layers=1,
                              num_heads=4, mlp_features=1024, max_seq_len=64)
    shapes = jax.eval_shape(lambda: model.init(jax.random.key(0), jnp.zeros((1, 64), jnp.int32)))
    kernel_shape = shapes["params"]["layers_0"]["mlp"]["up_proj"]["kernel"].shape
    kernel = Layout().shardings(mesh, shapes)["params"]["layers_0"]["mlp"]["up_proj"]["kernel"]
    batch = batch_shardings(mesh, {"text": np.zeros((16, 65), np.int32)})["text"]

    def holders(sharding, shape):
        """Map each distinct block (as index tuples) to the devices that hold it."""
        blocks: dict[tuple, list[int]] = {}
        for device, index in sharding.devices_indices_map(shape).items():
            key = tuple((s.start or 0, s.stop if s.stop is not None else n) for s, n in zip(index, shape))
            blocks.setdefault(key, []).append(device.id)
        return blocks

    batch_blocks = holders(batch, (16, 65))
    kernel_blocks = holders(kernel, kernel_shape)

    for variant, t in THEMES.items():
        svg = Figure(760, 540, palette=t)
        # The mesh: data picks the panel, fsdp the row, tensor the column.
        svg.text(24, 30, "MeshSpec(fsdp=2, tensor=2) on 8 devices", size=14, weight=600)
        svg.text(24, 50, "data takes the rest: 8 / (2 × 2) = 2", size=13, color="muted")
        for d in range(2):
            ox, oy = 24 + d * 172, 80
            svg.text(ox, oy - 8, f"data {d}", size=13, color="muted")
            for f in range(2):
                for k in range(2):
                    device = next(i for i, c in coords.items() if c == (d, f, k))
                    x, y = ox + k * 80, oy + f * 64
                    svg.rect(x, y, 74, 56, fill=f"tint{2 * f + k}", stroke="rule")
                    svg.text(x + 37, y + 25, f"device {device}", size=13, mono=True, anchor="middle", weight=600)
                    svg.text(x + 37, y + 44, f"f{f} t{k}", size=12, color="muted", anchor="middle")
        svg.text(24, 228, "f = fsdp index, t = tensor index", size=13, color="muted")

        # The batch: rows split over data and fsdp; tensor pairs hold the same rows.
        bx, by = 400, 80
        svg.text(bx, 30, "batch: text (16, 65) int32", size=14, weight=600)
        svg.text(bx, 50, "rows over (data, expert, fsdp)", size=13, color="muted")
        for (rows, _), devices in sorted(batch_blocks.items()):
            y = by + rows[0] * 8
            f = coords[devices[0]][1]
            svg.rect(bx, y, 336, 29, fill=f"tint{2 * f}", rx=3)
            svg.text(bx + 12, y + 20, f"rows {rows[0]}–{rows[1] - 1}", size=13, mono=True)
            svg.text(bx + 324, y + 20, "devices " + ", ".join(map(str, sorted(devices))), size=13,
                     color="muted", anchor="end")
        svg.text(bx, 228, "Both devices of a tensor pair hold the same rows.", size=13, color="muted")

        # The kernel: (embed, mlp) split fsdp × tensor; each block on one device per data index.
        kx, ky = 24, 300
        svg.text(kx, 280, f"MLP up_proj kernel {kernel_shape}, PartitionSpec{tuple(kernel.spec)}",
                 size=14, weight=600)
        for (rows, cols), devices in sorted(kernel_blocks.items()):
            f, k = coords[devices[0]][1], coords[devices[0]][2]
            x, y = kx + (cols[0] // 512) * 216, ky + (rows[0] // 128) * 76
            svg.rect(x, y, 208, 70, fill=f"tint{2 * f + k}", rx=3)
            svg.text(x + 12, y + 26, f"[{rows[0]}:{rows[1]}, {cols[0]}:{cols[1]}]", size=13, mono=True)
            svg.text(x + 12, y + 50, "devices " + ", ".join(map(str, sorted(devices))), size=13,
                     color="muted")
        svg.text(476, 326, "Rows (embed) split over fsdp,", size=13, color="muted")
        svg.text(476, 346, "columns (mlp) over tensor.", size=13, color="muted")
        svg.text(476, 376, "Each block is held by one", size=13, color="muted")
        svg.text(476, 396, "device of each data index.", size=13, color="muted")
        svg.text(24, 500, "Placements read from MeshSpec.build, Layout().shardings and batch_shardings",
                 size=13, color="muted")
        svg.text(24, 520, "(dew.training) on 8 simulated CPU devices.", size=13, color="muted")
        svg.save(HERE / f"mesh-{variant}.svg")


def training_step_figure():
    """The order of work in Trainer.fit and in one compiled step."""
    for variant, t in THEMES.items():
        svg = Figure(760, 610, palette=t)
        svg.text(24, 30, "Trainer.fit(dataset, steps=...)", size=15, mono=True, weight=600)
        box(svg, 24, 50, 224, 86, "Trainer.place()", ["init or restore TrainState,", "placed on the mesh"])
        box(svg, 24, 162, 224, 86, "dataset.train(partition)", ["this process's rows of", "every global batch"])
        box(svg, 24, 274, 224, 86, "DevicePrefetchIterator", ["shard_batch: host arrays", "to one global jax.Array"])
        svg.arrow(136, 136, 136, 160)
        svg.arrow(136, 248, 136, 272)

        # The compiled step.
        svg.rect(272, 50, 464, 360, fill="raised", stroke="accent", rx=10)
        svg.text(288, 74, "compiled step (jax.jit), once per step", size=14, weight=600, color="accent")
        box(svg, 288, 90, 432, 66, "objective.loss(variables, batch, step)",
            ["→ Ratio(total, mass), Aux(metrics)"], accent=True)
        box(svg, 288, 172, 432, 66, "gradient", ["of the mean over the accumulation window"])
        box(svg, 288, 254, 432, 66, "optimizer.update, optax.apply_updates",
            ["when the window closes"])
        box(svg, 288, 336, 432, 62, "ema_update", ["of the leaves objective.ema selects"])
        for y in (156, 238, 320):
            svg.arrow(504, y, 504, y + 15)
        svg.arrow(248, 93, 286, 110)
        svg.arrow(248, 317, 286, 130)

        # After the step, on the host.
        svg.arrow(504, 410, 504, 452)
        svg.text(24, 446, "then, on the host, between steps", size=14, weight=600)
        rows = [("log_every", "loss, metrics, speed"), ("eval_every", "evaluate on val"),
                ("checkpoint_every", "state, data position")]
        for i, (name, what) in enumerate(rows):
            box(svg, 24 + i * 242, 462, 228, 66, name, [what])
        svg.text(24, 568, "TrainState holds params, opt_state, ema, key and three counters:", size=13,
                 color="muted")
        svg.text(24, 588, "step counts attempts, microstep accepted microbatches, updates optimizer updates.",
                 size=13, color="muted")
        svg.save(HERE / f"training-loop-{variant}.svg")


def data_figure():
    """A Dataset's iterators, the process's share and the global batch."""
    for variant, t in THEMES.items():
        svg = Figure(760, 390, palette=t)
        box(svg, 24, 24, 280, 104, "Dataset", ["train(partition) → iterator", "val(partition) → one pass",
                                               "records, batch"])
        box(svg, 456, 24, 280, 104, "DataPartition", ["index, count: which share of", "every global batch",
                                                      "this process reads"])
        box(svg, 24, 196, 280, 104, "host batch", ['{"text": (B / count, S)}', "NumPy arrays,",
                                                   "one dict per step"])
        box(svg, 456, 196, 280, 104, "global batch", ["jax.Array (B, S)", "rows split over",
                                                      "data × expert × fsdp"], accent=True)
        svg.arrow(304, 76, 454, 76)
        svg.arrow(456, 128, 306, 196)
        svg.arrow(304, 248, 454, 248)
        svg.text(380, 66, "DataPartition.of(mesh)", size=12, mono=True, color="muted", anchor="middle")
        svg.text(400, 170, "next(train(partition))", size=12, mono=True, color="muted")
        svg.text(380, 238, "shard_batch", size=12, mono=True, color="muted", anchor="middle")
        svg.text(24, 336, "One process: DataPartition() reads every row; the host batch is the global batch.",
                 size=13, color="muted")
        svg.text(24, 358, "Several processes: each reads B / count rows of every step, and", size=13,
                 color="muted")
        svg.text(24, 378, "jax.make_array_from_process_local_data joins them.", size=13, color="muted")
        svg.save(HERE / f"data-pipeline-{variant}.svg")


def diffusion_figure():
    """x_t = alpha_t x_0 + sigma_t epsilon at six times, with the rates of two presets."""
    tile = Image.open(HERE / "gallery/heun.png").convert("RGB").crop((267, 502, 463, 698))
    x0 = np.asarray(tile.resize((64, 64), Image.BILINEAR), np.float32) / 127.5 - 1
    eps = np.asarray(jax.random.normal(jax.random.key(0), x0.shape))
    cosine, flow = Cosine()().schedule, Flow()().schedule
    fractions = np.linspace(0, 1, 6)

    def noised(schedule, fraction):
        alpha, sigma = (float(v) for v in schedule.rates(jnp.asarray(fraction * schedule.T)))
        x = alpha * x0 + sigma * eps
        return alpha, sigma, np.clip((x + 1) * 127.5, 0, 255).astype(np.uint8)

    for variant, t in THEMES.items():
        svg = Figure(760, 700, palette=t)
        for row, (name, schedule) in enumerate([("Cosine()", cosine), ("Flow()", flow)]):
            y = 30 + row * 200
            svg.text(24, y, f"{name}, T = {schedule.T:g}", size=14, mono=True, weight=600)
            for i, fraction in enumerate(fractions):
                alpha, sigma, image = noised(schedule, fraction)
                x = 24 + i * 120
                svg.image(x, y + 12, 112, image)
                svg.text(x + 56, y + 146, f"t = {fraction * schedule.T:g}", size=13, mono=True, anchor="middle")
                svg.text(x + 56, y + 165, f"α {alpha:.2f}  σ {sigma:.2f}", size=12, color="muted",
                         anchor="middle")

        # alpha_t and sigma_t against t / T for both presets.
        px, py, pw, ph = 24, 444, 280, 220
        svg.text(px, py - 14, "α and σ against t / T", size=14, weight=600)
        svg.rect(px, py, pw, ph, fill="raised", rx=4)
        grid = np.linspace(0, 1, 101)
        for schedule, dash in [(cosine, None), (flow, "5 4")]:
            rates = np.array([[float(v) for v in schedule.rates(jnp.asarray(g * schedule.T))] for g in grid])
            for column, color in [(0, "accent"), (1, "muted")]:
                svg.polyline([(px + g * pw, py + ph - r * ph) for g, r in zip(grid, rates[:, column])],
                             color=color, dash=dash)
        svg.text(px, py + ph + 20, "0", size=12, color="muted")
        svg.text(px + pw, py + ph + 20, "1", size=12, color="muted", anchor="end")
        legend = ["α: accent, σ: grey", "solid: Cosine(), dashed: Flow()", "",
                  "Each image is α x₀ + σ ε with the same ε,", "at the rates schedule.rates(t) returns.",
                  "", "x₀ is a 64 × 64 image sampled from a model", "trained with Dew, scaled to [-1, 1]."]
        for i, line in enumerate(legend):
            svg.text(340, py + 20 + 22 * i, line, size=13, color="muted")
        svg.save(HERE / f"diffusion-forward-{variant}.svg")


def moe_routing_figure():
    """The router's choices in a tiny mixture decoder, read from its sown `router` collection."""
    model = CausalTransformer(vocab_size=32, emb_features=16, num_layers=2, num_heads=2,
                              mlp_features=32, max_seq_len=64, mixture=Mixture(experts=4, top_k=2),
                              dtype=jnp.float32, attention_impl="xla")
    tokens = jax.random.randint(jax.random.key(1), (1, 64), 0, 32)
    variables = model.init(jax.random.key(0), tokens)
    _, sown = model.apply(variables, tokens, mutable=["router"])
    layers = []
    for name in ("layers_0", "layers_1"):
        gate = sown["router"][name]["mlp"]["gate"]
        indices, scores = np.asarray(gate["indices"][0][0]), np.asarray(gate["scores"][0][0])
        chosen = np.take_along_axis(scores, indices, axis=-1)
        layers.append((indices, chosen / chosen.sum(-1, keepdims=True)))
    shown = 22
    for variant, t in THEMES.items():
        svg = Figure(760, 490, palette=t)
        svg.text(20, 30, 'mixture={"experts": 4, "top_k": 2}, layer 0', size=14, mono=True, weight=600)
        svg.text(20, 52, f"The router's choices for the first {shown} of 64 tokens.", size=13,
                 color="muted")
        indices, weights = layers[0]
        cell, x0, y0 = 28, 100, 70
        for expert in range(4):
            svg.text(x0 - 10, y0 + expert * (cell + 4) + 19, f"expert {expert}", size=12, mono=True,
                     anchor="end", color="muted")
            for token in range(shown):
                x, y = x0 + token * cell, y0 + expert * (cell + 4)
                hit = np.nonzero(indices[token] == expert)[0]
                if hit.size:
                    svg.rect(x, y, cell - 2, cell, fill=f"tint{expert}", rx=3)
                    svg.text(x + (cell - 2) / 2, y + 19, f"{weights[token, hit[0]]:.2f}"[1:], size=12,
                             mono=True, anchor="middle")
                else:
                    svg.rect(x, y, cell - 2, cell, fill="surface", rx=3)
        for token in range(0, shown, 4):
            svg.text(x0 + token * cell + (cell - 2) / 2, y0 + 4 * (cell + 4) + 16, str(token), size=12,
                     mono=True, anchor="middle", color="muted")
        svg.text(x0, y0 + 4 * (cell + 4) + 36, "token position", size=12, color="muted")

        # Tokens per expert over all 64 tokens, per layer.
        by, bh = 300, 80
        svg.text(20, by - 26, "Tokens routed to each expert: 64 tokens × top 2 = 128 slots", size=14,
                 weight=600)
        for layer, (indices, _) in enumerate(layers):
            counts = np.bincount(indices.ravel(), minlength=4)
            lx = 100 + layer * 330
            for expert, count in enumerate(counts):
                h = bh * count / 64
                x = lx + expert * 60
                svg.rect(x, by + bh - h, 44, h, fill=f"tint{expert}", rx=3)
                svg.text(x + 22, by + bh - h - 6, str(int(count)), size=13, mono=True, anchor="middle")
                svg.text(x + 22, by + bh + 18, f"e{expert}", size=12, mono=True, anchor="middle",
                         color="muted")
            svg.text(lx, by + bh + 42, f"layer {layer}", size=13, mono=True)
        svg.text(20, 455, "Cell numbers: the token's normalized routing weight for that expert.",
                 size=13, color="muted")
        svg.text(20, 476, "Read from the 'router' collection of a fresh init: random weights and tokens, "
                 "so the load is uneven.", size=13, color="muted")
        svg.save(HERE / f"moe-routing-{variant}.svg")


def post_training_figure():
    """The data each post-training objective reads, from dew.data, dew.objectives.rl and pack."""
    for variant, t in THEMES.items():
        svg = Figure(760, 372, palette=t)
        lanes = [
            ("SFT", "Chat", ["conversations, rendered by", "the chat template"],
             "text, text_roles", ["[B, L + 1]"], "LMObjective", ["loss_role=Role.ASSISTANT"]),
            ("DPO", "Preferences", ["chosen, rejected and", "their completion masks"],
             "input_ids", ["completion_mask", "both [B, 2, S]"], "DPOObjective",
             ["policy against the frozen", "reference in TrainState.ema"]),
            ("GRPO", "Prompts", ["left-padded prompt,", "prompt_length, reward fields"],
             "SampledRollout → pack", ["G completions per prompt,", "reward → advantages"],
             "GRPOObjective", ["clipped ratio + beta · KL", "to the reference in .ema"]),
        ]
        for row, (name, source, source_lines, fields, field_lines, objective, objective_lines) in enumerate(lanes):
            y = 20 + row * 104
            svg.text(20, y + 45, name, size=14, mono=True, weight=600)
            box(svg, 70, y, 200, 84, source, source_lines)
            box(svg, 290, y, 210, 84, fields, field_lines)
            box(svg, 520, y, 220, 84, objective, objective_lines, accent=True)
            svg.arrow(270, y + 42, 288, y + 42)
            svg.arrow(500, y + 42, 518, y + 42)
        # GRPO samples with the policy it trains.
        svg.polyline([(630, 312), (630, 336), (395, 336)], color="muted", width=1.25)
        svg.arrow(395, 336, 395, 314)
        svg.text(512, 360, "the current policy samples the next batch", size=13, color="muted",
                 anchor="middle")
        svg.save(HERE / f"post-training-{variant}.svg")


FIGURES = {"mesh": mesh_figure, "training-loop": training_step_figure, "data-pipeline": data_figure,
           "diffusion-forward": diffusion_figure, "moe-routing": moe_routing_figure,
           "post-training": post_training_figure}


if __name__ == "__main__":
    prefixes = tuple(sys.argv[1:]) or ("",)
    for name, draw in FIGURES.items():
        if name.startswith(prefixes):
            draw()
