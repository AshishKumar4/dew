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
(src/dew/training/transaction.py). The RLVR figures score replies to the
GSM8K question of tutorials/10-rl-with-verifiable-rewards.ipynb, and take
their advantages and clipped objective from dew.rl.

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
from PIL import Image, ImageFilter

from dew.diffusion.presets import EDM, Cosine, Flow
from dew.figures import THEMES, Figure
from dew.nn.backbones import CausalTransformer, Mixture
from dew.rl import clipped_surrogate_terms, group_advantage, k3_kl
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
        box(svg, 288, 254, 432, 66, "optimizer.update, dew.optim.stepped",
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
             "PolicyCollector → pack", ["G completions per prompt,", "reward → advantages"],
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


# The task of tutorials/10: a GSM8K question (the first of its training split), four of a model's
# replies, and the answer each puts in \boxed{}, which the checker compares with the reference, 72.
QUESTION = ("Natalia sold 48", "clips in April and", "half as many in", "May. How many did", "she sell in all?")
REPLIES = (("48/2 = 24, 48+24 =", "72"), ("48×2 = 96, 48+96 =", "144"), ("April 48, May 24:", "72"),
           ("half of 48 is", "24"))
TRUTH = "72"


def save_both(svg: Figure, name: str, narrow: bool):
    """A figure's wide layout is <name>.svg, which notebooks embed; the site swaps in <name>-narrow.svg
    on a phone."""
    svg.save(HERE / f"{name}{'-narrow' if narrow else ''}.svg")


def reply_row(svg: Figure, x, y, width, text, answer, fill, size=13):
    """One reply: its working, then the answer it boxes, in a chip at the row's end."""
    svg.rect(x, y, width, 30, fill="surface", stroke="rule", rx=6)
    svg.text(x + 10, y + 20, text, size=size, mono=True, color="muted")
    chip = 12 + 9 * len(answer)
    svg.rect(x + width - chip - 4, y + 4, chip, 22, fill=fill, stroke="none" if fill != "surface" else "rule",
             rx=4)
    svg.text(x + width - 4 - chip / 2, y + 20, answer, size=size + 1, mono=True, weight=600, anchor="middle")


def rlvr_loop_figure():
    """The loop: the policy writes several replies, the checker scores them, and the update makes the
    replies that beat their group more likely."""
    rewards = np.array([float(answer == TRUTH) for _, answer in REPLIES], np.float32)
    advantages = np.asarray(group_advantage(jnp.asarray(rewards), len(REPLIES)))
    after = np.exp(0.6 * advantages) / np.exp(0.6 * advantages).sum()
    label = ("The policy writes four replies to a maths question, a checker compares each boxed answer with "
             "the reference, and the update makes the replies that scored above the group's mean more likely.")
    for narrow in (False, True):
        # Where each part sits in the wide (left to right) and narrow (top to bottom) layouts.
        width, height = (360, 420) if narrow else (720, 316)
        card_x, card_y = (12, 30) if narrow else (14, 92)
        box_x, box_y = (196, 14) if narrow else (172, 86)
        rows_x, rows_y, row_step, row_w = (30, 196, 44, 198) if narrow else (360, 44, 58, 196)
        checker_x, checker_w, reward_x = (236, 56, 344) if narrow else (572, 58, 700)
        svg = Figure(width, height, seconds=11, label=label)
        svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)
        svg.text(card_x, card_y - 10, "question", size=13, color="muted")
        svg.rect(card_x, card_y, 140 if not narrow else 160, 106, fill="raised", stroke="rule", rx=6)
        for k, line in enumerate(QUESTION):
            svg.text(card_x + 9, card_y + 21 + 19 * k, line, size=12.5)
        if narrow:
            svg.arrow(card_x + 162, card_y + 53, box_x - 4, card_y + 53)
        else:
            svg.arrow(card_x + 142, card_y + 53, box_x - 4, card_y + 53)
        svg.rect(box_x, box_y, 124, 140 if not narrow else 124, fill="raised", stroke="accent", rx=8)
        svg.text(box_x + 62, box_y + 24, "policy", size=15, weight=600, anchor="middle")
        svg.text(box_x + 62, box_y + 44, "P(reply)", size=12, color="muted", anchor="middle")
        bars_y = box_y + (54 if narrow else 60)
        for k, letter in enumerate("abcd"):
            x = box_x + 18 + k * 24
            svg.rect(x, bars_y, 16, 46, fill="rule", stroke="none", rx=2)
            column = svg.rect(x, bars_y, 16, 46, fill="accent", stroke="none", rx=2)
            svg.animate(column, {6.4: {"transform": "scaleY(0.25)"},
                                 7.6: {"transform": f"scaleY({after[k]:.3f})"}}, origin="50% 100%")
            svg.text(x + 8, bars_y + 64, letter, size=12, mono=True, color="muted", anchor="middle")
        svg.text(rows_x, rows_y - 14, "replies", size=13, color="muted")
        svg.text(reward_x, rows_y - 14, "reward", size=13, color="muted", anchor="end")
        rows_end = rows_y + 3 * row_step + 30
        with svg.group() as checker:
            svg.rect(checker_x, rows_y, checker_w, rows_end - rows_y, fill="raised", stroke="text", rx=8)
            svg.text(checker_x + checker_w / 2, (rows_y + rows_end) / 2 - 4, "= 72 ?", size=12, mono=True,
                     anchor="middle")
            svg.text(checker_x + checker_w / 2, (rows_y + rows_end) / 2 + 14, "checker", size=11, color="muted",
                     anchor="middle")
        svg.appear(checker, 2.9)
        if narrow:
            # One trunk from the policy down the left edge, a branch to each reply.
            svg.polyline([(box_x + 62, box_y + 124), (box_x + 62, rows_y - 36), (12, rows_y - 36),
                          (12, rows_y + 3 * row_step + 15)], color="muted", width=1.25)
        for i, ((text, answer), reward) in enumerate(zip(REPLIES, rewards)):
            y = rows_y + i * row_step
            with svg.group() as row:
                if narrow:
                    svg.arrow(12, y + 15, rows_x - 4, y + 15)
                else:
                    svg.arrow(box_x + 126, box_y + 70, rows_x - 24, y + 15)
                    svg.text(rows_x - 8, y + 20, "abcd"[i], size=13, mono=True, color="muted", anchor="end")
                reply_row(svg, rows_x, y, row_w, text, answer, "surface", size=12)
            svg.appear(row, 0.8 + 0.45 * i)
            with svg.group() as scored:
                reply_row(svg, rows_x, y, row_w, text, answer, "accent_fill" if reward else "warm_fill", size=12)
                svg.arrow(checker_x + checker_w, y + 15, checker_x + checker_w + 8, y + 15)
                svg.text(reward_x, y + 21, f"{reward:.0f}", size=15, mono=True, weight=600, anchor="end",
                         color="accent" if reward > rewards.mean() else "warm")
            svg.appear(scored, 3.4 + 0.35 * i)
        if narrow:
            back = [(reward_x - 6, rows_end + 6), (reward_x - 6, rows_end + 20), (352, rows_end + 20),
                    (352, box_y + 62), (box_x + 128, box_y + 62)]
            tip_x, tip_y = box_x + 125, box_y + 62
            head = f"M{tip_x},{tip_y} L{tip_x + 7},{tip_y - 3.5} L{tip_x + 7},{tip_y + 3.5} z"
            note_x, note_y = 180, rows_end + 46
        else:
            back = [(692, 274), (692, 298), (240, 298), (240, 232)]
            head = "M240,229 L236.5,236 L243.5,236 z"
            note_x, note_y = 466, 290
        svg.draw(svg.polyline(back, color="accent", width=1.5), 5.2, 6.2)
        svg.appear(svg.add("path", {"d": head}, {"fill": svg.palette["accent"]}), 6.1, 0.2)
        note = "update: replies that beat the group's mean get more likely"
        if narrow:
            note = "update: replies above the mean get more likely"
        svg.appear(svg.text(note_x, note_y, note, size=13, color="accent", anchor="middle"), 5.8)
        save_both(svg, "rlvr-loop", narrow)


def rlvr_advantage_figure():
    """GRPO's advantage: each reward against its group's mean, in units of the group's deviation. A
    maths checker's rewards are 0 or 1, so the advantages take two values."""
    rewards = np.array([1, 0, 0, 1, 1, 0, 0, 0], np.float32)
    advantages = np.asarray(group_advantage(jnp.asarray(rewards), len(rewards)))
    mean, deviation = float(rewards.mean()), float(rewards.std(ddof=1))
    label = ("Eight replies to one question scored 1 or 0, their mean, and the advantages: each reward minus "
             "the mean, divided by the group's standard deviation.")
    for narrow in (False, True):
        width, height = (360, 560) if narrow else (720, 300)
        # Each panel's top-left corner: rewards, then advantages beside them or below.
        rx_, ry_ = 20, 0
        ax_, ay_ = (20, 270) if narrow else (380, 0)
        svg = Figure(width, height, seconds=11, label=label)
        svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)
        floor, tall, slot = ry_ + 236, 150, 38
        svg.text(rx_ + 8, ry_ + 32, "reward R of 8 replies to one question", size=13, color="muted")
        svg.line(rx_ + 8, floor, rx_ + 320, floor, color="rule")
        level = floor - tall * mean
        for i, reward in enumerate(rewards):
            x = rx_ + 16 + i * slot
            up = reward > mean
            # The bar grows in a neutral color, then takes the color of its side of the mean.
            for fill, at in (("tint1", 0.0), ("accent" if up else "warm", 3.6 + 0.1 * i)):
                with svg.group() as colored:
                    column = svg.rect(x, floor - tall, 26, tall, fill=fill, stroke="none", rx=3)
                    svg.animate(column, {0.3 + 0.15 * i: {"transform": "scaleY(0)"},
                                         0.9 + 0.15 * i: {"transform": f"scaleY({reward:.3f})"}},
                                origin="50% 100%")
                if at:
                    svg.appear(colored, at)
            svg.text(x + 13, floor + 18, f"{reward:.0f}", size=12, mono=True, color="muted", anchor="middle")
            gap = svg.line(x + 13, floor - tall * reward, x + 13, level, color="accent" if up else "warm",
                           width=2, dash="2 3")
            svg.appear(gap, 3.6 + 0.1 * i)
        with svg.group() as average:
            svg.line(rx_ + 8, level, rx_ + 324, level, color="text", width=1.5, dash="5 4")
            svg.text(rx_ + 324, level - 8, f"mean {mean:.3f}", size=13, mono=True, anchor="end")
        svg.animate(average, {2.4: {"opacity": 0, "transform": "translateX(-24px)"},
                              3.0: {"opacity": 1, "transform": "translateX(0px)"}})
        zero, unit = ay_ + 146, 44
        svg.text(ax_ + 12, ay_ + 32, f"advantage A = (R − {mean:.3f}) / {deviation:.2f}", size=13,
                 color="muted")
        svg.line(ax_ + 8, zero, ax_ + 320, zero, color="rule")
        for i, advantage in enumerate(advantages):
            x = ax_ + 16 + i * slot
            up = advantage > 0
            column = svg.rect(x, zero - unit * advantage if up else zero, 26, unit * abs(advantage),
                              fill="accent" if up else "warm", stroke="none", rx=3)
            svg.animate(column, {4.8 + 0.1 * i: {"transform": "scaleY(0)"},
                                 5.6 + 0.1 * i: {"transform": "scaleY(1)"}},
                        origin="50% 100%" if up else "50% 0%")
            y = zero - unit * advantage - 8 if up else zero + unit * abs(advantage) + 18
            svg.appear(svg.text(x + 13, y, f"{advantage:+.2f}", size=11, mono=True, anchor="middle",
                                color="accent" if up else "warm"), 6.2)
        svg.appear(svg.text(ax_ + 164, ay_ + 280, "right replies are pushed up, wrong ones down",
                            size=13, color="muted", anchor="middle"), 6.6)
        save_both(svg, "rlvr-advantage", narrow)


def rlvr_clip_figure():
    """PPO's clipped objective against the ratio, from dew.rl's terms, with its flat stretches shaded."""
    ratios = np.linspace(0.02, 3.5, 351)
    log_ratio = jnp.log(jnp.asarray(ratios, jnp.float32))
    label = ("The clipped objective against the ratio for a positive and a negative advantage. Where it is "
             "flat, the clipped term gives the token no gradient.")
    for narrow in (False, True):
        width, height = (360, 590) if narrow else (720, 304)
        svg = Figure(width, height, seconds=9, label=label)
        svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)
        for panel, advantage in enumerate((1.0, -1.0)):
            left, top = (40, 58 + panel * 270) if narrow else (52 + panel * 352, 58)
            wide, high = 300, 170

            def objective(log_ratio, advantage=advantage):
                terms, _ = clipped_surrogate_terms(log_ratio[None, :], jnp.asarray([advantage]),
                                                   jnp.ones((1, log_ratio.size)))
                return -terms[0]

            values = np.asarray(objective(log_ratio))
            slope = jax.grad(lambda r, objective=objective: objective(r[None])[0])
            slopes = np.asarray(jax.vmap(slope)(log_ratio))
            low, high_value = (0.0, 1.5) if advantage > 0 else (-3.5, 0.0)

            def at(r, v, left=left, top=top, low=low, high_value=high_value):
                return left + wide * r / 3.5, top + high - high * (v - low) / (high_value - low)

            flat = slopes == 0
            edges = np.flatnonzero(np.diff(np.concatenate([[0], flat.astype(int), [0]])))
            for start, stop in zip(edges[::2], edges[1::2]):
                x0, _ = at(ratios[start], low)
                x1, _ = at(ratios[stop - 1], low)
                with svg.group() as shade:
                    svg.rect(x0, top, x1 - x0, high, fill="tint1", stroke="none", rx=0).style["opacity"] = "0.6"
                    name = "dual clip" if ratios[start] > 2 else "clipped"
                    svg.text(min((x0 + x1) / 2, left + wide - 26), top - 7, name, size=12, color="muted",
                             anchor="middle")
                svg.appear(shade, 3.6)
            svg.line(left, top + high, left + wide, top + high, color="rule")
            svg.line(left, top, left, top + high, color="rule")
            for tick in (0, 1, 2, 3):
                x, _ = at(tick, low)
                svg.text(x, top + high + 18, str(tick), size=12, mono=True, color="muted", anchor="middle")
            x1, _ = at(1.0, low)
            svg.line(x1, top, x1, top + high, color="muted", dash="3 4")
            title = ("A > 0: raise π(y), up to ρ = 1 + ε" if advantage > 0
                     else "A < 0: lower π(y), down to ρ = 1 − ε")
            svg.text(left, top - 28, title, size=13, weight=600)
            svg.text(left + wide, top + high + 38, "ratio ρ = π / π_old", size=12, color="muted",
                     anchor="end")
            name = svg.text(left - 12, top + high / 2, "objective", size=12, color="muted", anchor="middle")
            name.attributes["transform"] = f"rotate(-90 {left - 12} {top + high / 2})"
            line = svg.polyline([at(r, v) for r, v in zip(ratios, values)], color="accent", width=2.5)
            svg.draw(line, 0.4, 3.2)
        svg.appear(svg.text(width / 2, height - 12, "shaded: flat, so the clipped term gives no gradient",
                            size=12, color="muted", anchor="middle"), 4.0)
        save_both(svg, "rlvr-clip", narrow)


def rlvr_sequence_figure():
    """One reply as the loss reads it: its tokens, the mask that keeps the ones the policy wrote, and the
    reply's advantage on each of them."""
    prompt, written = ["in", "all", "?"], ["48", "+", "24", "=", "72"]
    rewards = np.array([float(answer == TRUTH) for _, answer in REPLIES], np.float32)
    advantage = float(np.asarray(group_advantage(jnp.asarray(rewards), len(REPLIES)))[0])
    width, height, left, gap = 430, 236, 92, 5
    svg = Figure(width, height, seconds=9, label=f"The tokens of one reply: the end of the question, "
                 f"{' '.join(prompt)}, with mask 0, and the tokens the policy wrote, {' '.join(written)}, with "
                 f"mask 1, each carrying the reply's advantage {advantage:+.2f}.")
    svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)
    rows = [("tokens", 30), ("mask m", 82), ("A × m", 134)]
    for name, y in rows:
        svg.text(left - 12, y + 20, name, size=14, color="muted", anchor="end")
    x = left
    for j, token in enumerate(prompt + written):
        mine = j >= len(prompt)
        wide = max(38 if mine else 26, 10 + 9 * len(token))
        svg.rect(x, rows[0][1], wide, 30, fill="tint1" if mine else "surface", stroke="none" if mine else "rule",
                 rx=4)
        svg.text(x + wide / 2, rows[0][1] + 20, token, size=14, mono=True, anchor="middle",
                 color="text" if mine else "muted")
        mask = svg.text(x + wide / 2, rows[1][1] + 21, "1" if mine else "0", size=15, mono=True,
                        anchor="middle", color="text" if mine else "muted")
        svg.appear(mask, 1.2 + 0.08 * j)
        with svg.group() as credit:
            if mine:
                svg.rect(x, rows[2][1], wide, 30, fill="accent_fill", stroke="none", rx=4)
            svg.text(x + wide / 2, rows[2][1] + 20, f"{advantage:+.2f}" if mine else "0", size=11, mono=True,
                     anchor="middle", color="accent" if mine else "muted")
        svg.appear(credit, 3.2 + 0.1 * j)
        x += wide + gap
    svg.appear(svg.text(width / 2, 204, f"reward 1 gives A = {advantage:+.2f} on each written token, "
                        "0 on the question", size=13, color="muted", anchor="middle"), 4.6)
    svg.save(HERE / "rlvr-sequence.svg")


def rlvr_kl_figure():
    """The k3 estimate against π / π_ref, from dew.rl.k3_kl: zero at 1, and a slope that pulls back to 1."""
    ratios = np.geomspace(0.25, 4, 161)
    values = np.asarray(k3_kl(jnp.log(jnp.asarray(ratios, jnp.float32)), jnp.zeros(ratios.size)))
    width, height = 440, 280
    left, top, wide, high = 56, 44, 360, 180
    svg = Figure(width, height, seconds=9, label="The k3 estimate of the KL divergence against the ratio of "
                 "the policy to the reference: zero where they agree and rising on both sides, so its "
                 "gradient pulls the policy back toward the reference.")
    svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)

    def at(r, v):
        return left + wide * np.log(r / 0.25) / np.log(16), top + high - high * v / values.max()

    svg.line(left, top + high, left + wide, top + high, color="rule")
    svg.line(left, top, left, top + high, color="rule")
    for tick in (0.25, 0.5, 1, 2, 4):
        x, _ = at(tick, 0)
        svg.text(x, top + high + 20, f"{tick:g}", size=13, mono=True, color="muted", anchor="middle")
    svg.text(left + wide, top + high + 42, "π / π_ref, the policy over the reference", size=13, color="muted",
             anchor="end")
    svg.text(left, 28, "k3 per token", size=14, weight=600)
    x1, _ = at(1, 0)
    svg.line(x1, top, x1, top + high, color="muted", dash="3 4")
    svg.draw(svg.polyline([at(r, v) for r, v in zip(ratios, values)], color="accent", width=2.5), 0.4, 2.8)
    # A token that drifted to twice the reference's probability slides back toward it.
    slide = np.geomspace(2.6, 1.0, 9)
    heights = np.asarray(k3_kl(jnp.log(jnp.asarray(slide, jnp.float32)), jnp.zeros(slide.size)))
    path = [at(r, v) for r, v in zip(slide, heights)]
    ball = svg.circle(*path[0], 6, fill="text")
    moves = {3.2 + 0.45 * i: {"transform": f"translate({x - path[0][0]:.1f}px,{y - path[0][1]:.1f}px)",
                              "opacity": 1} for i, (x, y) in enumerate(path)}
    svg.animate(ball, moves | {2.8: {"transform": "translate(0px,0px)", "opacity": 0}})
    svg.appear(svg.text(x1 + 8, top + 16, "zero where π = π_ref", size=13, color="muted"), 3.0)
    svg.save(HERE / "rlvr-kl.svg")


def agent_episode_figure():
    """A tool-using episode as one sequence: the question, the model's first action (code), the tool's
    output, the model's answer, then the verifier's one reward, which flows back to the actions alone."""
    width, height = 440, 290
    svg = Figure(width, height, seconds=12, label="An agent episode. The model writes code, a tool runs it and "
                 "answers, the model writes its answer, and the verifier scores the whole episode once; the "
                 "reward reaches only the tokens the model wrote, not the question or the tool's output.")
    svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)
    segments = [("question", ["Natalia sold", "48 clips…"], "none", False),
                ("action", ["python call:", "print(48+24)"], "accent_fill", True),
                ("tool", ["response:", "72"], "tint1", False),
                ("action", ["So the answer", "is \\boxed{72}."], "accent_fill", True)]
    widths, y, h = [98, 112, 76, 112], 46, 64
    svg.text(18, 30, "one episode, read left to right", size=13, color="muted")
    x, lefts = 18, []
    for k, ((who, lines, fill, trained), w) in enumerate(zip(segments, widths, strict=True)):
        with svg.group() as part:
            svg.rect(x, y, w, h, fill=fill, stroke="rule" if fill == "none" else "none", rx=6)
            svg.text(x + 8, y + 20, who, size=12, weight=600, color="text" if trained else "muted")
            for number, line in enumerate(lines):
                svg.text(x + 8, y + 38 + 14 * number, line, size=12, mono=True,
                         color="text" if trained else "muted")
        svg.appear(part, 0.4 + 0.8 * k)
        lefts.append((x, w, trained))
        x += w + 6
    last_x, last_w, _ = lefts[-1]
    with svg.group() as verdict:
        svg.arrow(last_x + last_w / 2, y + h + 2, last_x + last_w / 2, 140)
        svg.rect(last_x - 20, 142, last_w + 20, 40, fill="raised", stroke="text", rx=6)
        svg.text(last_x - 10 + last_w / 2, 167, "verifier: R = 1", size=12, mono=True, weight=600,
                 anchor="middle")
    svg.appear(verdict, 3.8)
    back_y = 206
    flow = svg.polyline([(last_x - 20, 162), (28, 162), (28, back_y)], color="accent", width=1.5)
    svg.draw(flow, 4.6, 5.6)
    svg.appear(svg.text(36, 154, "one reward for the whole episode", size=12, color="accent"), 4.8)
    for k, (left, w, trained) in enumerate(lefts):
        with svg.group() as credit:
            if trained:
                svg.rect(left, back_y + 10, w, 28, fill="accent", stroke="none", rx=4)
                svg.text(left + w / 2, back_y + 29, "trained on A", size=12, color="surface", anchor="middle",
                         weight=600)
            else:
                svg.rect(left, back_y + 10, w, 28, fill="none", stroke="rule", rx=4)
                svg.text(left + w / 2, back_y + 29, "read only", size=12, color="muted", anchor="middle")
        svg.appear(credit, 5.8 + 0.3 * k)
    svg.appear(svg.text(width - 18, height - 16, "only the model's own tokens carry loss", size=12,
                        color="muted", anchor="end"), 7.2)
    svg.save(HERE / "agent-episode.svg")

def agent_lanes_figure():
    """The rollout scheduler between the server and the trainer: it publishes the weights, starts a group of
    episodes for each task, waits for whole groups, packs them, and hands the batch to the trainer."""
    width, height = 440, 300
    svg = Figure(width, height, seconds=12, label="Three lanes. The scheduler publishes the trainer's weights to "
                 "the server, starts a group of episodes for each task, waits for the whole groups, packs them "
                 "into a batch and hands it to the trainer, whose update makes the next version of the weights.")
    svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)
    lanes = [("server", 40), ("scheduler", 140), ("trainer", 240)]
    for name, y in lanes:
        svg.line(110, y + 20, width - 20, y + 20, color="rule", width=1)
        svg.text(20, y + 25, name, size=14, weight=600)
    times = [0.4, 1.6, 3.0, 4.6, 6.0, 7.4]
    # 1. publish the weights, version 0
    with svg.group() as publish:
        svg.arrow(130, 258, 130, 64, color="accent", width=1.5)
        svg.text(138, 200, "weights v0", size=12, color="accent")
    svg.appear(publish, times[0])
    # 2. a group of episodes per task on the server
    with svg.group() as episodes:
        for k in range(4):
            svg.rect(160 + 18 * k, 48 + (k % 2) * 4, 14, 24, fill="accent_fill", stroke="none", rx=3)
        svg.text(160, 40, "a group of episodes per task", size=12, color="muted")
    svg.appear(episodes, times[1])
    # 3. whole groups come back to the scheduler
    with svg.group() as collected:
        svg.arrow(240, 76, 240, 146, width=1.25)
        for k in range(4):
            svg.rect(200 + 18 * k, 148, 14, 24, fill="accent_fill", stroke="none", rx=3)
        svg.text(276, 165, "whole groups", size=12, color="muted")
    svg.appear(collected, times[2])
    # 4. packed into a batch for the trainer
    with svg.group() as packed:
        svg.arrow(262, 176, 300, 246, width=1.25)
        svg.rect(300, 246, 90, 28, fill="raised", stroke="accent", rx=5)
        svg.text(345, 265, "packed batch", size=12, anchor="middle")
    svg.appear(packed, times[3])
    # 5. the update makes the next version
    with svg.group() as update:
        svg.text(345, 290, "update → v1", size=12, color="accent", weight=600, anchor="middle")
    svg.appear(update, times[4])
    with svg.group() as again:
        svg.polyline([(395, 260), (424, 260), (424, 30), (300, 30)], color="accent", width=1.5, dash="4 4")
        svg.text(418, 150, "next batch", size=12, color="accent", anchor="end")
    svg.appear(again, times[5])
    svg.save(HERE / "agent-lanes.svg")


def lm_next_token_figure():
    """Next-token prediction on one sentence: at each position the model reads the tokens so far, scores
    every possible next token, and training pushes up the one that actually came next."""
    sentence = ["Plants", "need", "light", "to", "grow"]
    # Four candidates at each position, the true next token first, with a model's probabilities before and
    # after training: made up for the drawing, shaped like a small model's.
    candidates = [(["need", "are", "and", "can"], [0.18, 0.34, 0.27, 0.21], [0.58, 0.18, 0.12, 0.12]),
                  (["light", "water", "to", "the"], [0.22, 0.31, 0.27, 0.20], [0.52, 0.30, 0.10, 0.08]),
                  (["to", "and", "for", "."], [0.24, 0.30, 0.19, 0.27], [0.61, 0.17, 0.12, 0.10]),
                  (["grow", "live", "make", "be"], [0.20, 0.25, 0.22, 0.33], [0.66, 0.14, 0.08, 0.12])]
    width, height = 440, 330
    box_w, gap, left, row_y = 70, 8, 24, 52
    svg = Figure(width, height, seconds=12, label="Next-token prediction on 'Plants need light to grow'. At "
                 "each position the model reads the tokens so far and scores the candidates for the next one; "
                 "training pushes up the token that actually came next.")
    svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)
    svg.text(24, 30, "the model reads the tokens so far", size=13, color="muted")
    for j, token in enumerate(sentence):
        x = left + j * (box_w + gap)
        svg.rect(x, row_y, box_w, 30, fill="raised", stroke="rule", rx=5)
        svg.text(x + box_w / 2, row_y + 20, token, size=14, mono=True, anchor="middle", color="muted")
    times = [0.4 + 2.2 * i for i in range(len(candidates))]
    windows, targets, panels = [], [], []
    floor, tall, bar_w = 290, 130, 56
    for i, (names, before, after) in enumerate(candidates):
        with svg.group() as window:
            svg.rect(left - 3, row_y - 3, (i + 1) * (box_w + gap) - gap + 6, 36, fill="none", stroke="accent",
                     rx=7, width=2)
        windows.append(window)
        with svg.group() as target:
            x = left + (i + 1) * (box_w + gap)
            svg.rect(x, row_y, box_w, 30, fill="accent_fill", stroke="none", rx=5)
            svg.text(x + box_w / 2, row_y + 20, sentence[i + 1], size=14, mono=True, anchor="middle",
                     weight=600)
            svg.text(x + box_w / 2, row_y + 52, "next?", size=12, color="accent", anchor="middle")
        targets.append(target)
        with svg.group() as panel:
            svg.text(24, 132, "P(next token)", size=13, color="muted")
            svg.line(24, floor, width - 24, floor, color="rule")
            for k, (name, low, high) in enumerate(zip(names, before, after)):
                x = 40 + k * 96
                bar = svg.rect(x, floor - tall, bar_w, tall, fill="accent" if k == 0 else "tint1", stroke="none",
                               rx=3)
                start = times[i] + 0.5
                svg.animate(bar, {start: {"transform": f"scaleY({low:.2f})"},
                                  start + 1.0: {"transform": f"scaleY({high:.2f})"}}, origin="50% 100%")
                svg.text(x + bar_w / 2, floor + 22, name, size=14, mono=True, anchor="middle",
                         color="text" if k == 0 else "muted")
        panels.append(panel)
    svg.show_in_turn(windows, times)
    svg.show_in_turn(targets, times)
    svg.show_in_turn(panels, times)
    svg.appear(svg.text(width - 24, 132, "training pushes the true one up", size=13, color="accent",
                        anchor="end"), 1.4)
    svg.save(HERE / "lm-next-token.svg")


def pretrained_round_trip_figure():
    """The round trip of a pretrained decoder: load a Hugging Face checkpoint into Dew, check it against
    transformers, train it, export it, and read the export back with transformers."""
    width, height = 440, 300
    svg = Figure(width, height, seconds=12, label="The round trip. A Hugging Face checkpoint loads into Dew, "
                 "Dew's next-token probabilities match transformers', Dew trains it on Python code, and the "
                 "export loads back into transformers with the same prediction.")
    svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)

    def station(x, y, title, lines, stroke):
        svg.rect(x, y, 150, 74, fill="raised", stroke=stroke, rx=8)
        svg.text(x + 12, y + 24, title, size=14, weight=600)
        for k, line in enumerate(lines):
            svg.text(x + 12, y + 44 + 17 * k, line, size=12, mono=True, color="muted")

    station(20, 30, "Hugging Face", ["config.json", "model.safetensors"], "rule")
    with svg.group() as loaded:
        svg.arrow(172, 67, 266, 67)
        svg.text(219, 58, "load", size=13, color="muted", anchor="middle")
        station(270, 30, "Dew", ["CausalTransformer", "params tree"], "accent")
    svg.appear(loaded, 0.6)
    with svg.group() as compared:
        svg.text(345, 126, "same P(next token)", size=12, color="accent", anchor="middle")
        svg.text(345, 142, "as transformers", size=12, color="accent", anchor="middle")
    svg.appear(compared, 2.0)
    with svg.group() as trained:
        svg.arrow(345, 150, 345, 192)
        svg.text(335, 176, "train on Python", size=13, color="muted", anchor="end")
        station(270, 196, "Dew, trained", ["lower perplexity", "on held-out code"], "accent")
    svg.appear(trained, 3.4)
    with svg.group() as exported:
        svg.arrow(268, 233, 174, 233)
        svg.text(221, 224, "export", size=13, color="muted", anchor="middle")
        station(20, 196, "Hugging Face", ["config.json", "model.safetensors"], "rule")
    svg.appear(exported, 4.8)
    with svg.group() as read:
        svg.arrow(95, 192, 95, 108)
        svg.text(105, 140, "transformers", size=13, color="muted")
        svg.text(105, 156, "reads it back:", size=13, color="muted")
        svg.text(105, 172, "same next token", size=13, color="accent", weight=600)
    svg.appear(read, 6.2)
    svg.save(HERE / "pretrained-round-trip.svg")


def sft_mask_figure():
    """One conversation through a chat template: the messages, the tokens the template writes with each
    one's role, and the loss mask, 1 on the assistant's reply and its closing marker."""
    width, height = 440, 336
    svg = Figure(width, height, seconds=11, label="A conversation through the chat template. The user's "
                 "question and the assistant's reply become one row of tokens with their roles, and the loss "
                 "mask is 1 only on the reply and its end-of-turn marker.")
    svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)
    svg.text(20, 30, "messages", size=13, color="muted")
    svg.rect(20, 40, 196, 30, fill="tint1", stroke="none", rx=6)
    svg.text(30, 60, "user: Why is the sky blue?", size=12)
    svg.rect(226, 40, 196, 30, fill="accent_fill", stroke="none", rx=6)
    svg.text(236, 60, "assistant: Light scatters.", size=12)
    rows = [("<|im_start|>", "none"), ("user", "none"), ("Why", "tint1"), ("is", "tint1"), ("the", "tint1"),
            ("sky", "tint1"), ("blue", "tint1"), ("?", "tint1"), ("<|im_end|>", "none"),
            ("<|im_start|>", "none"), ("assistant", "none"), ("Light", "accent_fill"),
            ("scatters", "accent_fill"), (".", "accent_fill"), ("<|im_end|>", "accent_fill")]
    x, y, line = 20, 112, 0
    svg.text(20, 100, "tokens the template writes, and the loss mask m under each", size=13, color="muted")
    placed = []
    for k, (token, fill) in enumerate(rows):
        w = 12 + 7.2 * len(token)
        if x + w > width - 20:
            x, line = 20, line + 1
        top = y + line * 66
        with svg.group() as box:
            svg.rect(x, top, w, 26, fill=fill, stroke="rule" if fill == "none" else "none", rx=4)
            svg.text(x + w / 2, top + 18, token, size=12, mono=True, anchor="middle",
                     color="muted" if fill == "none" else "text")
        svg.appear(box, 0.6 + 0.12 * k)
        placed.append((x, top, w, fill == "accent_fill"))
        x += w + 4
    for k, (bx, top, w, scored) in enumerate(placed):
        mark = svg.text(bx + w / 2, top + 44, "1" if scored else "0", size=13, mono=True, anchor="middle",
                        weight=600 if scored else 400, color="accent" if scored else "muted")
        svg.appear(mark, 3.8 + 0.1 * k)
    svg.appear(svg.text(width - 20, height - 14, "only the reply is trained", size=13, color="accent",
                        anchor="end"), 5.6)
    svg.save(HERE / "sft-mask.svg")


# The flower of tutorials/02: a 64 x 64 sample from a model Dew trained on Oxford Flowers, in [-1, 1].
def flower() -> np.ndarray:
    tile = Image.open(HERE / "gallery/heun.png").convert("RGB").crop((267, 502, 463, 698))
    return np.asarray(tile.resize((64, 64), Image.BILINEAR), np.float32) / 127.5 - 1


def pixels(x: np.ndarray) -> np.ndarray:
    return np.clip((x + 1) * 127.5, 0, 255).astype(np.uint8)


def edm_step_figure():
    """One training step in two lanes: noise a clean image to a random sigma, then have the model
    guess the clean image back and score the guess against it."""
    x0 = flower()
    sigma = 0.5
    eps = np.asarray(jax.random.normal(jax.random.key(0), x0.shape))
    # A model's guess at this noise level is a smoothed version of the image; a blur stands in for it.
    guess = np.asarray(Image.fromarray(pixels(x0)).filter(ImageFilter.GaussianBlur(2.2)), np.float32) / 127.5 - 1
    label = ("One training step. A clean flower x0 gets noise at a random level sigma to make x_t; the model "
             "sees x_t and sigma and guesses x0; the loss is how far the guess is from x0, and the update "
             "makes the next guess closer.")
    size = 80
    for narrow in (False, True):
        # The wide layout has the loss beside the guess; the narrow one has it under the second lane.
        width, height = (360, 520) if narrow else (720, 350)
        x_clean, x_noise, x_noisy = (24, 140, 256) if narrow else (34, 176, 318)
        lane2 = 300 if narrow else 196
        x_model, x_guess = (246, 40) if narrow else (308, 460)
        top1, top2 = 56, lane2 + 22
        svg = Figure(width, height, seconds=11, label=label)
        svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)
        svg.text(16 if narrow else 20, 34, "1  noise a clean image", size=14, weight=600)
        svg.text(16 if narrow else 20, lane2, "2  guess it back", size=14, weight=600)
        with svg.group() as clean:
            svg.image(x_clean, top1, size, pixels(x0), photo=True)
            svg.text(x_clean + size / 2, top1 + size + 20, "x₀", size=14, mono=True, anchor="middle")
        svg.appear(clean, 0.3)
        with svg.group() as noise:
            svg.text((x_clean + size + x_noise) / 2, top1 + size / 2 + 6, "+", size=20, anchor="middle")
            svg.image(x_noise, top1, size, pixels(eps / 3), photo=True)
            svg.text(x_noise + size / 2, top1 + size + 20, f"σ ε, σ = {sigma:g}", size=13, mono=True,
                     anchor="middle")
        svg.appear(noise, 1.1)
        with svg.group() as noisy:
            svg.text((x_noise + size + x_noisy) / 2, top1 + size / 2 + 6, "=", size=20, anchor="middle")
            svg.image(x_noisy, top1, size, pixels(x0 + sigma * eps), photo=True)
            svg.text(x_noisy + size / 2, top1 + size + 20, "xₜ", size=14, mono=True, anchor="middle")
        svg.appear(noisy, 1.9)
        # The noisy image drops into the second lane, to the model straight below it.
        middle = x_model + 50
        feed = svg.polyline([(x_noisy + size / 2, top1 + size + 30), (x_noisy + size / 2, lane2 - 24),
                             (middle, lane2 - 24), (middle, top2 - 4)], color="muted", width=1.25)
        svg.draw(feed, 2.6, 3.2)
        with svg.group() as model:
            svg.rect(x_model, top2, 100, size, fill="raised", stroke="accent", rx=8)
            svg.text(middle, top2 + 34, "DiT", size=15, weight=600, anchor="middle")
            svg.text(middle, top2 + 54, "(xₜ, σ)", size=13, mono=True, color="muted", anchor="middle")
        svg.appear(model, 3.0)
        with svg.group() as guessed:
            if narrow:
                svg.arrow(x_model - 4, top2 + size / 2, x_guess + size + 8, top2 + size / 2)
            else:
                svg.arrow(x_model + 104, top2 + size / 2, x_guess - 8, top2 + size / 2)
            svg.image(x_guess, top2, size, pixels(guess), photo=True)
            svg.text(x_guess + size / 2, top2 + size + 20, "guess x̂₀", size=14, mono=True, anchor="middle")
        svg.appear(guessed, 3.8)
        # The loss compares the guess with x0 from the first lane, and its gradient updates the model.
        with svg.group() as loss:
            if narrow:
                loss_x, loss_bottom = width / 2, top2 + size + 64
                svg.text(loss_x, loss_bottom - 14, "loss = w(σ) · ‖x̂₀ − x₀‖²", size=14, mono=True,
                         anchor="middle", color="accent")
            else:
                loss_x, loss_bottom = 630, top2 + 66
                svg.text(566, top2 + 30, "loss =", size=14, mono=True, color="accent")
                svg.text(566, top2 + 52, "w(σ) · ‖x̂₀ − x₀‖²", size=14, mono=True, color="accent")
        svg.appear(loss, 4.8)
        floor = height - 30
        update = svg.polyline([(loss_x, loss_bottom), (loss_x, floor), (middle, floor), (middle, top2 + size + 4)],
                              color="accent", width=1.5)
        svg.draw(update, 5.6, 6.4)
        svg.appear(svg.text((loss_x + middle) / 2, floor + 18, "update the weights", size=13, color="accent",
                            anchor="middle"), 6.3)
        save_both(svg, "edm-step", narrow)


def edm_noise_figure():
    """One flower at eight of EDM's noise levels, from the least noise to the most, at the rates the
    preset's sampler uses."""
    x0 = flower()
    eps = np.asarray(jax.random.normal(jax.random.key(0), x0.shape))
    process = EDM(regime="pixel")()
    _, sigmas = (np.asarray(v) for v in process.sampler_schedule.rates(process.times(8)))
    sigmas = sigmas[::-1]
    width, height, size = 440, 272, 168
    svg = Figure(width, height, seconds=10, label="One flower with more and more noise: x_t = x0 + sigma "
                 "epsilon at eight of EDM's noise levels, from sigma 0.002, where it is clean, to sigma 80, "
                 "where only noise is left.")
    svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)
    svg.text(24, 34, "xₜ = x₀ + σ ε", size=15, mono=True, weight=600)
    times = [0.4 + 0.8 * i for i in range(len(sigmas))]
    frames, captions = [], []
    for sigma in sigmas:
        with svg.group() as frame:
            svg.image(24, 52, size, pixels(x0 + sigma * eps), photo=True)
        frames.append(frame)
        captions.append(svg.text(24 + size / 2, 52 + size + 24, f"σ = {sigma:.3g}", size=14, mono=True,
                                 anchor="middle"))
    svg.show_in_turn(frames, times)
    svg.show_in_turn(captions, times)
    axis_x, axis_w, axis_y = 224, 192, 196
    svg.image(axis_x, 52, 64, pixels(x0), photo=True)
    svg.text(axis_x + 76, 90, "x₀, the clean image", size=13, color="muted")
    # Where each level sits on a log scale, the current one filled.
    low, high = np.log10(sigmas[0]), np.log10(sigmas[-1])

    def at(sigma: float) -> float:
        return axis_x + axis_w * (np.log10(sigma) - low) / (high - low)

    svg.line(axis_x, axis_y, axis_x + axis_w, axis_y, color="rule")
    svg.text(axis_x, axis_y + 26, f"{sigmas[0]:.3g}", size=12, mono=True, color="muted", anchor="middle")
    svg.text(axis_x + axis_w, axis_y + 26, f"{sigmas[-1]:.3g}", size=12, mono=True, color="muted",
             anchor="middle")
    svg.text(axis_x + axis_w / 2, axis_y - 40, "σ, log scale", size=13, color="muted", anchor="middle")
    for sigma in sigmas:
        svg.circle(at(sigma), axis_y, 4, fill="tint1")
    svg.show_in_turn([svg.circle(at(sigma), axis_y, 7, fill="accent") for sigma in sigmas], times)
    svg.save(HERE / "edm-noise.svg")


def samplers_paths_figure():
    """Euler and Heun on the probability-flow ODE of a five-point data set, against a fine-step reference path.

    The data are five points on a circle, like petals, so the denoiser D(x; sigma), the mean of the data
    under the posterior, has a closed form. Both solvers take three steps down EDM's Karras grid (rho = 7)
    from sigma = 4, along dx / dsigma = (x - D(x; sigma)) / sigma; the reference path takes 2,000 Heun steps.
    """
    petals = np.stack([np.cos(np.linspace(0, 2 * np.pi, 6)[:5] + 0.3),
                       np.sin(np.linspace(0, 2 * np.pi, 6)[:5] + 0.3)], axis=1)

    def denoised(x, sigma):
        logits = -np.sum((x - petals) ** 2, axis=1) / (2 * sigma ** 2)
        weights = np.exp(logits - logits.max())
        return weights @ petals / weights.sum()

    def slope(x, sigma):
        return (x - denoised(x, sigma)) / sigma

    def grid(steps, high=4.0, low=0.01, rho=7.0):
        ramp = np.linspace(0, 1, steps + 1)
        return (high ** (1 / rho) + ramp * (low ** (1 / rho) - high ** (1 / rho))) ** rho

    def euler(x, sigmas):
        path = [x]
        for now, then in zip(sigmas[:-1], sigmas[1:]):
            x = x + (then - now) * slope(x, now)
            path.append(x)
        return np.array(path)

    def heun(x, sigmas):
        path = [x]
        for now, then in zip(sigmas[:-1], sigmas[1:]):
            first = slope(x, now)
            guess = x + (then - now) * first
            x = x + (then - now) * (first + slope(guess, then)) / 2
            path.append(x)
        return np.array(path)

    start = np.array([1.35, 1.86])
    reference, coarse = heun(start, grid(2000)), grid(3)
    paths = {"Euler, 3 calls": euler(start, coarse), "Heun, 6 calls": heun(start, coarse)}
    width, height = 440, 320
    view_x, view_y, scale = 150, 205, 88

    def at(point):
        return view_x + scale * point[0], view_y - scale * point[1]

    svg = Figure(width, height, seconds=11, label="Five data points on a circle and the paths three solvers take "
                 "to them from the same starting noise: a fine-step reference path, Euler in three steps, which stops short "
                 "between the points, and Heun in three steps, which lands on one.")
    svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)
    # The direction the ODE pushes, at a middling noise level, as faint arrows.
    for gx in np.linspace(-1.3, 1.7, 8):
        for gy in np.linspace(-1.1, 2.0, 8):
            point = np.array([gx, gy])
            push = -slope(point, 0.6)
            push = 0.16 * push / (np.linalg.norm(push) + 1e-9)
            (x1, y1), (x2, y2) = at(point), at(point + push)
            svg.arrow(x1, y1, x2, y2, color="rule", width=1)
    for petal in petals:
        svg.circle(*at(petal), 6, fill="accent")
    svg.text(*at(petals[3] + [-0.1, -0.32]), "data", size=13, color="accent")
    svg.circle(*at(start), 5, fill="text")
    svg.text(at(start)[0] + 10, at(start)[1] + 5, "x_T", size=13, mono=True)
    svg.draw(svg.polyline([at(p) for p in reference], color="muted", width=1.5, dash=None), 0.4, 2.2)
    styles = (("text", "5 4", 2.6), ("accent", None, 5.6))
    for (name, path), (color, dash, begin) in zip(paths.items(), styles):
        line = svg.polyline([at(p) for p in path], color=color, width=2, dash=dash)
        svg.appear(line, begin, 0.3)
        for index, point in enumerate(path[1:]):
            svg.appear(svg.circle(*at(point), 4, fill=color), begin + 0.4 * index, 0.2)
    legend_x = 286
    rows = [("reference path", "muted", None), *((name, color, dash) for name, (color, dash, _) in
                                             zip(paths, styles))]
    for row, (name, color, dash) in enumerate(rows):
        y = 250 + 24 * row
        svg.line(legend_x, y - 5, legend_x + 20, y - 5, color=color, width=2, dash=dash)
        svg.text(legend_x + 28, y, name, size=13, color=color)
    svg.save(HERE / "samplers-paths.svg")


def jepa_step_figure():
    """One I-JEPA step: the context encoder sees part of a flower, the predictor guesses the embeddings of
    the hidden blocks, and the target encoder, an average of the context encoder's weights, supplies the
    embeddings to compare with."""
    x0 = flower()
    grid, size = 8, 112
    patch = size / grid
    rng = np.random.default_rng(3)
    target = [(r, c) for r in range(1, 4) for c in range(4, 7)]
    context = [(r, c) for r in range(grid) for c in range(grid)
               if (r, c) not in target and rng.random() < 0.45]
    width, height = 440, 330
    svg = Figure(width, height, seconds=12, label="One I-JEPA step on a flower. The context encoder sees the "
                 "unmasked patches, the predictor guesses the embeddings of the hidden block, and the target "
                 "encoder, an average of the context encoder's weights, reads the whole image and gives the "
                 "embeddings to match.")
    svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)
    ix, iy = 20, 40
    svg.text(ix, 28, "the image", size=13, color="muted")
    svg.image(ix, iy, size, pixels(x0), photo=True)
    with svg.group() as hidden:
        for r in range(grid):
            for c in range(grid):
                if (r, c) not in context and (r, c) not in target:
                    svg.rect(ix + c * patch, iy + r * patch, patch, patch, fill="surface", stroke="none", rx=0)
    svg.appear(hidden, 0.6)
    with svg.group() as block:
        top_r, left_c = target[0]
        svg.rect(ix + left_c * patch, iy + top_r * patch, 3 * patch, 3 * patch, fill="surface", stroke="accent",
                 rx=2, width=2.5)
        svg.text(ix + (left_c + 1.5) * patch, iy + (top_r + 1.5) * patch + 5, "?", size=16, weight=600,
                 color="accent", anchor="middle")
    svg.appear(block, 1.2)
    svg.appear(svg.text(ix, iy + size + 20, "context, and a hidden block", size=12, color="muted"), 1.2)

    def chips(x, y, fill):
        for k in range(4):
            svg.rect(x + k * 16, y, 13, 13, fill=fill, stroke="none", rx=2)

    def box(x, y, w, title, note, stroke):
        svg.rect(x, y, w, 46, fill="raised", stroke=stroke, rx=7)
        svg.text(x + w / 2, y + 20, title, size=13, weight=600, anchor="middle")
        svg.text(x + w / 2, y + 37, note, size=11, color="muted", anchor="middle")

    with svg.group() as encoded:
        svg.arrow(ix + size + 6, iy + size * 0.7, 168, 228)
        box(168, 210, 120, "context encoder", "sees the context", "accent")
    svg.appear(encoded, 2.0)
    with svg.group() as predicted:
        svg.arrow(290, 233, 312, 233)
        box(314, 210, 106, "predictor", "+ block's positions", "accent")
        svg.arrow(367, 258, 367, 274)
        chips(335, 280, "accent_fill")
        svg.text(367, 314, "guess", size=12, color="accent", anchor="middle")
    svg.appear(predicted, 3.0)
    with svg.group() as targeted:
        # The target encoder reads the whole image, unmasked.
        svg.image(214, 30, 56, pixels(x0), photo=True)
        svg.text(242, 100, "whole image", size=11, color="muted", anchor="middle")
        svg.arrow(272, 58, 310, 62)
        box(314, 46, 106, "target encoder", "sees everything", "rule")
        svg.arrow(367, 94, 367, 110)
        chips(335, 116, "tint1")
        svg.text(367, 150, "target", size=12, color="muted", anchor="middle")
    svg.appear(targeted, 4.2)
    with svg.group() as loss:
        svg.polyline([(401, 122), (428, 122), (428, 286), (401, 286)], color="accent", width=1.5, dash="4 4")
        svg.text(424, 190, "loss", size=12, color="accent", anchor="end")
    svg.appear(loss, 5.4)
    with svg.group() as average:
        svg.polyline([(228, 210), (228, 176), (296, 176), (296, 84), (312, 84)], color="muted", width=1.25,
                     dash="2 4")
        svg.text(236, 168, "average of weights", size=11, color="muted")
    svg.appear(average, 6.4)
    svg.save(HERE / "jepa-step.svg")


def cfg_step_figure():
    """Classifier-free guidance as arrows: the model's prediction with the caption and without it, the
    difference between them, and the guided prediction that follows the difference further."""
    width, height = 440, 300
    origin, uncond, cond, scale = np.array([60.0, 230.0]), np.array([150.0, -40.0]), np.array([170.0, -95.0]), 2.2
    guided = uncond + scale * (cond - uncond)
    svg = Figure(width, height, seconds=11, label="Classifier-free guidance. From the noisy image, the model's "
                 "prediction without a caption and with the caption 'a red rose'; their difference points where "
                 "the caption pulls, and the guided prediction follows it further, here at scale 2.2.")
    svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)
    svg.circle(*origin, 6, fill="text")
    svg.text(origin[0] - 8, origin[1] + 24, "xₜ", size=14, mono=True)
    with svg.group() as plain:
        svg.arrow(*origin, *(origin + uncond), color="muted", width=2)
        svg.text(*(origin + uncond + [8, 18]), "no caption", size=13, color="muted")
    svg.appear(plain, 0.4)
    with svg.group() as captioned:
        svg.arrow(*origin, *(origin + cond), color="text", width=2)
        svg.text(*(origin + cond + [20, 6]), "“a red rose”", size=13)
    svg.appear(captioned, 1.4)
    pull = svg.line(*(origin + uncond), *(origin + cond), color="accent", width=1.5, dash="4 4")
    svg.appear(pull, 2.6)
    svg.appear(svg.text(*(origin + (uncond + cond) / 2 + [10, 10]), "the caption's pull", size=13,
                        color="accent"), 2.8)
    with svg.group() as followed:
        svg.line(*(origin + cond), *(origin + guided), color="accent", width=1.5, dash="4 4")
        svg.arrow(*origin, *(origin + guided), color="accent", width=2.5)
        svg.text(*(origin + guided + [-40, -14]), f"guided, s = {scale:g}", size=13, color="accent", weight=600)
    svg.appear(followed, 4.2)
    svg.appear(svg.text(24, 32, "ε_guided = ε_uncond + s (ε_cond − ε_uncond)", size=13, mono=True), 5.2)
    svg.save(HERE / "cfg-step.svg")


def cfg_sweep_figure():
    """The guided prediction as the scale s sweeps from 0 to 6: at 0 the plain prediction, at 1 the captioned
    one, past 1 further along the caption's pull."""
    width, height = 440, 270
    origin, uncond, cond = np.array([40.0, 236.0]), np.array([100.0, -10.0]), np.array([125.0, -40.0])
    scales = [0, 1, 2, 3, 4, 6]
    svg = Figure(width, height, seconds=11, label="The guided prediction as the guidance scale grows from 0 to 6: "
                 "at 0 it is the prediction without a caption, at 1 the captioned one, and past 1 it reaches "
                 "further along the caption's pull.")
    svg.rect(0.5, 0.5, width - 1, height - 1, fill="surface", rx=10)
    svg.circle(*origin, 6, fill="text")
    far = uncond + scales[-1] * (cond - uncond)
    svg.line(*(origin + uncond), *(origin + far), color="rule", width=1.25, dash="4 4")
    svg.arrow(*origin, *(origin + uncond), color="muted", width=1.5)
    svg.arrow(*origin, *(origin + cond), color="text", width=1.5)
    svg.text(*(origin + uncond + [6, 20]), "s = 0", size=12, mono=True, color="muted")
    svg.text(*(origin + cond + [-52, -8]), "s = 1", size=12, mono=True)
    times = [0.5 + 1.1 * i for i in range(len(scales))]
    arrows, readouts = [], []
    for s_ in scales:
        tip = origin + uncond + s_ * (cond - uncond)
        arrows.append(svg.arrow(*origin, *tip, color="accent", width=2.5))
        readouts.append(svg.text(width - 24, 36, f"s = {s_:g}", size=16, mono=True, weight=600, color="accent",
                                 anchor="end"))
    svg.show_in_turn(arrows, times)
    svg.show_in_turn(readouts, times)
    svg.text(24, 36, "guidance scale", size=13, color="muted")
    svg.save(HERE / "cfg-sweep.svg")


FIGURES = {"mesh": mesh_figure, "training-loop": training_step_figure, "data-pipeline": data_figure,
           "diffusion-forward": diffusion_figure, "moe-routing": moe_routing_figure,
           "post-training": post_training_figure, "rlvr-loop": rlvr_loop_figure,
           "rlvr-advantage": rlvr_advantage_figure, "rlvr-sequence": rlvr_sequence_figure,
           "rlvr-clip": rlvr_clip_figure, "rlvr-kl": rlvr_kl_figure, "edm-step": edm_step_figure,
           "edm-noise": edm_noise_figure, "samplers-paths": samplers_paths_figure,
           "cfg-step": cfg_step_figure, "cfg-sweep": cfg_sweep_figure, "sft-mask": sft_mask_figure,
           "pretrained-round-trip": pretrained_round_trip_figure, "jepa-step": jepa_step_figure,
           "lm-next-token": lm_next_token_figure, "agent-episode": agent_episode_figure,
           "agent-lanes": agent_lanes_figure}


if __name__ == "__main__":
    prefixes = tuple(sys.argv[1:]) or ("",)
    for name, draw in FIGURES.items():
        if name.startswith(prefixes):
            draw()
