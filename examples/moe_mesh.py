"""Train a small mixture-of-experts decoder on a device mesh, generate from it, and draw where its tokens go.

On eight CPU devices simulated in one process, as a mesh of data 2, expert 2 and fsdp 2:

    XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu \
        python examples/moe_mesh.py --out runs/moe-mesh

On one accelerator, where the experts cannot be split, each device computes every expert:

    python examples/moe_mesh.py --out runs/moe-mesh-gpu --expert 1 --fsdp 1 --dispatch global

The training text mixes three kinds of line: short English sentences, sums and assignments.
Before every training step the script runs the step's batch through the step's parameters
once more and reads the experts each token's router picked. For each MoE layer it counts
the slots each device sends to every other device, and predicts, by the exchange's own
rule, the most exchange rounds any expert group needs for them. The drawing reads the
placement from the objects the run used: the device grid from the mesh,
each device's experts and kernel slice from the expert kernel's sharding, each device's
rows from the sharding the layout gives the layer's input, and the all-to-all operations
the compiled training step's program holds.

After training, `dew.inference.serving.Server` generates greedy continuations of eight
prompts on the same mesh, with the model's own dispatch. (`dew.sampling.generate` refuses
the exchange dispatch until jax-ml/jax#40907 is fixed.)

Writes OUT/moe-mesh.json with every step's traffic, and OUT/moe-mesh.svg, an animation
with a frame every `snapshot_every` steps and a last frame that routes the generated text.
"""
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import tyro
from jax.sharding import NamedSharding

from dew import Layout, MeshSpec, Trainer
from dew.config import OptimConfig
from dew.data import ByteTokenizer, DataPartition, Loading, TokenWindows
from dew.inference import RunProcessor, TextGeneration
from dew.inference.serving import Server
from dew.nn.backbones import CausalTransformer, Mixture
from dew.nn.sharding import RESIDUAL, logical_spec
from dew.objectives.lm import LMObjective
from dew.sampling import Sampling
from dew.training.distributed import DevicePrefetchIterator, batch_shardings


@dataclass
class Config:
    out: Path = Path("runs/moe-mesh")
    """A new directory for the data, the drawing and the recorded numbers."""
    expert: int = 2
    """Devices the experts are split over."""
    fsdp: int = 2
    """Devices the parameter widths are split over; data parallelism takes the rest."""
    dispatch: str = "exchange"
    """`exchange` sends each token to the device that holds its expert (all-to-all);
    `global` computes every expert where the token is, and needs no expert axis."""
    experts: int = 8
    top_k: int = 2
    steps: int = 300
    snapshot_every: int = 25
    batch_size: int = 16
    sequence_length: int = 32
    learning_rate: float = 3e-3
    seed: int = 0


CLASSES = ("letter", "digit", "space", "symbol")
PALETTE = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2", "#eeca3b", "#9d755d",
           "#bab0ac", "#ff9da6", "#79706e", "#d67195", "#8cd17d", "#b6992d", "#499894", "#86bcb6")


def corpus(rng: np.random.Generator, lines: int) -> str:
    """Sentences, sums and assignments in random order, one per line."""
    subjects = ["the cat", "a dog", "the bird", "my friend", "the child"]
    verbs = ["sees", "likes", "finds", "wants", "hears"]
    objects = ["the ball", "a tree", "the river", "some food", "the moon"]
    names = ["x", "y", "z", "total", "count"]
    out = []
    for kind in rng.integers(0, 3, lines):
        if kind == 0:
            out.append(f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)}.")
        elif kind == 1:
            a, b = rng.integers(0, 50, 2)
            out.append(f"{a}+{b}={a + b}")
        else:
            out.append(f"{rng.choice(names)} = {rng.choice(names)} * {rng.integers(2, 10)};")
    return "\n".join(out) + "\n"


def write_tokens(directory: Path, seed: int) -> None:
    """Byte token files in the layout `TokenWindows` reads."""
    directory.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    for split, lines in (("train", 20000), ("val", 400)):
        np.asarray(ByteTokenizer().encode(corpus(rng, lines)), np.uint8).tofile(directory / f"{split}.bin")
    sizes = {split: (directory / f"{split}.bin").stat().st_size for split in ("train", "val")}
    (directory / "meta.json").write_text(json.dumps({
        "tokenizer": "byte", "vocab_size": 256, "dtype": "uint8",
        "train_tokens": sizes["train"], "val_tokens": sizes["val"], "eos_id": None}))


def token_class(tokens: np.ndarray) -> np.ndarray:
    """The index into CLASSES of every byte."""
    text = np.asarray(tokens, np.uint8)
    letter = ((text >= ord("a")) & (text <= ord("z"))) | ((text >= ord("A")) & (text <= ord("Z")))
    digit = (text >= ord("0")) & (text <= ord("9"))
    space = (text == ord(" ")) | (text == ord("\n"))
    return np.select([letter, digit, space], [0, 1, 2], 3)


@dataclass
class Placement:
    """Where the mesh and the shardings put things, one entry per device in mesh order."""
    mesh: jax.sharding.Mesh
    devices: list
    held: list[range]
    """The experts each device stores."""
    kernel_slices: list[str]
    """The part of an expert kernel each device stores, as an index expression."""
    rows: list[slice]
    """The rows of a batch each device holds at an MoE layer's input."""
    owner: np.ndarray
    """[device, expert]: the device a token on `device` is computed on for `expert`."""


def placement(mesh, kernel: jax.Array, batch_shape: tuple[int, int, int], layout: Layout,
              dispatch: str) -> Placement:
    """Read the placement off the mesh, the expert kernel and the layout's activation rule."""
    devices = list(mesh.devices.flat)
    stored = kernel.sharding.devices_indices_map(kernel.shape)
    held = [range(*stored[device][0].indices(kernel.shape[0])) for device in devices]
    kernel_slices = [
        "[" + ", ".join(":" if part == slice(None) else f"{part.start}:{part.stop}"
                        for part in stored[device]) + "]"
        for device in devices]
    activation = NamedSharding(mesh, logical_spec(RESIDUAL, batch_shape, rules=layout.axis_rules, mesh=mesh))
    rows = [activation.devices_indices_map(batch_shape)[device][0] for device in devices]
    rows = [slice(*part.indices(batch_shape[0])[:2]) for part in rows]
    experts = kernel.shape[0]
    owner = np.tile(np.arange(len(devices))[:, None], (1, experts))
    if dispatch == "exchange":
        # The exchange's all-to-all runs over the expert axis alone: a device
        # trades with the devices that share its place on every other axis.
        position = {device: np.argwhere(mesh.devices == device)[0] for device in devices}
        axis = mesh.axis_names.index("expert")
        for index, device in enumerate(devices):
            group = [other for other, peer in enumerate(devices)
                     if np.array_equal(np.delete(position[peer], axis), np.delete(position[device], axis))]
            for expert in range(experts):
                (owner[index, expert],) = [peer for peer in group if expert in held[peer]]
    return Placement(mesh, devices, held, kernel_slices, rows, owner)


def routing_record(selections, tokens: np.ndarray, where: Placement, experts: int, shards: int,
                   dispatch: str) -> list[dict]:
    """Per MoE layer: slots sent between devices, the most exchange rounds they
    take, slots per expert and each byte class's slots per expert."""
    classes = token_class(tokens)
    layers = []
    for name in sorted(selections, key=lambda name: int(name.split("_")[1])):
        (indices,) = selections[name]["mlp"]["gate"]["indices"]
        sent = np.zeros((len(where.devices),) * 2, np.int64)
        for device, rows in enumerate(where.rows):
            np.add.at(sent[device], where.owner[device, indices[rows].ravel()], 1)
        by_class = np.zeros((len(CLASSES), experts), np.int64)
        np.add.at(by_class, (np.repeat(classes.ravel(), indices.shape[-1]), indices.ravel()), 1)
        layers.append(
            {
                "layer": name,
                "sent": sent.tolist(),
                "rounds": exchange_rounds(sent, shards, dispatch),
                "per_expert": np.bincount(indices.ravel(), minlength=experts).tolist(),
                "by_class": by_class.tolist(),
            }
        )
    return layers


def exchange_rounds(sent: np.ndarray, shards: int, dispatch: str) -> int:
    """The most all-to-all rounds any expert group runs in one MoE layer's
    forward, predicted by the rule in `dew.nn.moe._exchange_shard`: a first
    round sends every peer a bucket of ceil(slots / shards) rows, and the
    overflow of the fullest bucket in an expert group takes further rounds of
    the same size. Each group runs its own count; 0 under `global`."""
    if dispatch != "exchange":
        return 0
    first = -(-sent.sum(axis=1, keepdims=True) // shards)
    overflow = -(-np.maximum(sent - first, 0) // np.maximum(first, 1))
    return 1 + int(overflow.max())


def routing(selections, params, tokens: jax.Array, trainer: Trainer) -> dict:
    """The routers' sown selections for `tokens`, computed on the trainer's mesh."""
    with jax.set_mesh(trainer.device_mesh), nn.logical_axis_rules(trainer.layout.axis_rules):
        return jax.tree.map(np.asarray, selections(params, tokens))


def all_to_all_ops(hlo: str) -> tuple[int, str]:
    """How many all-to-all operations a compiled program holds, and the
    replica groups of the first. The count is of the program's text: an op
    inside a loop or a conditional runs as many times as the loop or the
    branch does."""
    ops = [line for line in hlo.splitlines() if re.search(r"= .*\ball-to-all\(", line)]
    groups = re.search(r"replica_groups=(\S+(?: \{[^}]*\})?)", ops[0]) if ops else None
    return len(ops), groups.group(1) if groups else ""


def main(config: Config) -> None:
    config.out.mkdir(parents=True)
    write_tokens(config.out / "tokens", config.seed)
    data = TokenWindows(path=str(config.out / "tokens"), seq_len=config.sequence_length, val_batches=1,
                        loading=Loading(workers=0, threads=1, read_buffer=2)).load(batch=config.batch_size)
    model = CausalTransformer(
        vocab_size=256, emb_features=64, num_layers=2, num_heads=4,
        mlp_features=128, max_seq_len=64, dtype=jnp.float32, attention_impl="xla",
        mixture=Mixture(experts=config.experts, top_k=config.top_k, dispatch=config.dispatch))
    objective = LMObjective(model, config.sequence_length, aux_loss_alpha=0.01)
    optimizer = OptimConfig(optimizer="adam", learning_rate=config.learning_rate).build(config.steps)
    trainer = Trainer(objective, optimizer, key=jax.random.key(config.seed),
                      mesh=MeshSpec(fsdp=config.fsdp, expert=config.expert), layout=Layout(min_shard=1))
    mesh = trainer.device_mesh
    print(f"Mesh {dict(mesh.shape)} over {mesh.devices.size} {jax.devices()[0].device_kind} device(s)")

    state, _, _ = trainer.place()
    kernel = state.variables["params"]["layers_0"]["mlp"]["experts"]["gate_proj"]["kernel"]
    print(f"Expert kernel {kernel.shape} placed as {kernel.sharding.spec}")
    shards = mesh.shape["expert"]
    where = placement(mesh, kernel, (config.batch_size, config.sequence_length, 64), trainer.layout,
                      config.dispatch)
    # The compiled step returns no routing, so a second forward over the
    # step's parameters and inputs reads the routers' sown selections.
    selections = jax.jit(lambda params, tokens: model.apply(params, tokens, mutable=["router"])[1]["router"])

    steps, frames = [], []
    with DevicePrefetchIterator(data.train(DataPartition.of(mesh)), mesh) as source:
        batch = next(source)
        step = trainer.compile(state, batch)
        operations, groups = all_to_all_ops(trainer.executable.as_text())
        print(f"The compiled training step holds {operations} all-to-all operations"
              + (f" over {groups}" if groups else ""))
        for current in range(1, config.steps + 1):
            # LMObjective predicts text[:, 1:] from text[:, :-1].
            inputs = batch["text"][:, :-1]
            layers = routing_record(routing(selections, state.variables, inputs, trainer), np.asarray(inputs),
                                    where, config.experts, shards, config.dispatch)
            state, loss, _, _, _ = step(state, batch)
            steps.append({"step": current, "loss": float(loss),
                          "sent": [layer["sent"] for layer in layers],
                          "rounds": [layer["rounds"] for layer in layers]})
            if current == 1 or current % config.snapshot_every == 0 or current == config.steps:
                print(f"step {current}: loss {float(loss):.3f}" + (
                    f", predicted exchange rounds per layer (max over expert groups) "
                    f"{[layer['rounds'] for layer in layers]}"
                    if config.dispatch == "exchange" else ""))
                frames.append({"label": f"training step {current} of {config.steps}", "step": current,
                               "loss": float(loss), "rows": [[rows.start, rows.stop] for rows in where.rows],
                               "layers": layers})
            batch = next(source)

    # Serve on the same mesh: the weights keep their placement, the server's
    # rows split over the batch axes, and under `exchange` its decode steps
    # run the same dispatch.
    tokenizer = ByteTokenizer()
    starts = ["the cat ", "a dog li", "my frien", "12+30=", "7+41=", "x = y * ", "total = ", "count = "]
    width = max(len(start) for start in starts)
    starts = [start.rjust(width, "\n") for start in starts]
    task = TextGeneration(model, state.variables, RunProcessor(tokenizer), sampling=Sampling(temperature=0.0))
    server = Server.from_task(task, slots=len(starts), capacity=64)
    generations = [generation.host() for generation in
                   server(starts, config.sequence_length - width, key=config.seed)]
    for start, generation in zip(starts, generations, strict=True):
        print(f"{start.lstrip()!r} -> {generation.text[0].split(chr(10))[0]!r}")
    sequences = np.concatenate([np.asarray(generation.tokens) for generation in generations])
    sequences_on_mesh = jax.device_put(sequences, batch_shardings(mesh, {"text": sequences})["text"])
    inference = placement(mesh, kernel, (*sequences.shape, 64), trainer.layout, config.dispatch)
    frames.append({
        "label": "after training: one forward over the prompts and their greedy continuations", "step": None,
        "loss": None,
        "rows": [[rows.start, rows.stop] for rows in inference.rows],
        "layers": routing_record(routing(selections, state.variables, sequences_on_mesh, trainer), sequences,
                                 inference, config.experts, shards, config.dispatch)})

    record = {
        "mesh": dict(mesh.shape), "device_kind": jax.devices()[0].device_kind,
        "devices": [str(device) for device in where.devices],
        "experts_held": [[r.start, r.stop] for r in where.held],
        "kernel_spec": str(kernel.sharding.spec), "kernel_shape": list(kernel.shape),
        "kernel_slices": where.kernel_slices,
        "dispatch": config.dispatch, "all_to_all_ops": operations, "all_to_all_groups": groups,
        "steps": steps, "frames": frames,
        "generated": [generation.text[0] for generation in generations],
    }
    (config.out / "moe-mesh.json").write_text(json.dumps(record))
    (config.out / "moe-mesh.svg").write_text(drawing(record, where, config))
    print(f"Wrote {config.out / 'moe-mesh.svg'} and {config.out / 'moe-mesh.json'}")


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

BOX_W, BOX_H, GAP = 190, 112, 34
SECONDS_PER_FRAME = 1.6


def text(x, y, body, size=12, anchor="start", weight="normal", fill="#222") -> str:
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" '
            f'font-weight="{weight}" fill="{fill}">{html.escape(str(body))}</text>')


def drawing(record: dict, where: Placement, config: Config) -> str:
    """An SVG of the device grid with one animated frame per recorded snapshot."""
    mesh = where.mesh
    shape = {axis: mesh.shape[axis] for axis in ("data", "expert", "fsdp")}
    if any(size > 1 for axis, size in mesh.shape.items() if axis not in shape):
        raise ValueError(
            f"the drawing lays out the data, expert and fsdp axes; this mesh is {dict(mesh.shape)}"
        )
    layers = len(record["frames"][0]["layers"])
    # One panel per data index, wrapped to rows about four devices wide;
    # inside a panel the expert axis runs down and fsdp across.
    panel_w = shape["fsdp"] * BOX_W + (shape["fsdp"] - 1) * GAP + 2 * GAP
    panel_h = shape["expert"] * BOX_H + (shape["expert"] - 1) * 2 * GAP + 2 * GAP
    per_row = min(shape["data"], max(1, 4 // shape["fsdp"]))
    panel_rows = -(-shape["data"] // per_row)
    grid_w = per_row * panel_w + (per_row - 1) * GAP
    grid_top = 112
    grid_h = panel_rows * panel_h + (panel_rows - 1) * GAP
    chart_x = grid_w + 60
    width = chart_x + 40 + config.experts * 34 + 60
    height = max(grid_top + grid_h + 300, grid_top + layers * 250 + 40)

    def panel(data: int) -> tuple[int, int]:
        return ((data % per_row) * (panel_w + GAP),
                grid_top + (data // per_row) * (panel_h + GAP))

    centre = {}
    static = []
    for index, device in enumerate(where.devices):
        data, expert, fsdp = (int(np.argwhere(mesh.devices == device)[0][mesh.axis_names.index(axis)])
                              for axis in shape)
        left, top = panel(data)
        x = left + GAP + fsdp * (BOX_W + GAP)
        y = top + GAP + expert * (BOX_H + 2 * GAP)
        centre[index] = (x + BOX_W / 2, y + BOX_H / 2)
        held = where.held[index]
        static += [
            f'<rect x="{x}" y="{y}" width="{BOX_W}" height="{BOX_H}" rx="6" fill="#fafafa" stroke="#888"/>',
            text(x + 8, y + 17, f"{device.platform}:{device.id}", 13, weight="bold"),
            text(x + BOX_W - 8, y + 17, f"d{data} e{expert} f{fsdp}", 11, "end", fill="#666"),
            text(x + 8, y + 36, f"experts {held.start}-{held.stop - 1}", 12),
            text(x + 8, y + 52, f"kernel{where.kernel_slices[index]}", 11, fill="#555"),
        ]
        for slot, expert_id in enumerate(held):
            static.append(f'<rect x="{x + 8 + slot * 14}" y="{y + 78}" width="11" height="11" '
                          f'fill="{PALETTE[expert_id % len(PALETTE)]}"/>')
    for data in range(shape["data"]):
        x, y = panel(data)
        static += [f'<rect x="{x}" y="{y}" width="{panel_w}" height="{panel_h}" rx="10" '
                   f'fill="none" stroke="#ccc" stroke-dasharray="4 3"/>',
                   text(x + 10, y + 16, f"data {data}", 12, fill="#666")]
    ops = (f"compiled step: its program holds {record['all_to_all_ops']} all-to-all ops"
           + (f" over {record['all_to_all_groups']}" if record["all_to_all_groups"] else ""))
    # Several CPU devices in one run are XLA's host platform split by
    # --xla_force_host_platform_device_count, not separate hardware.
    kind = ("simulated CPU devices" if where.devices[0].platform == "cpu" and len(where.devices) > 1
            else f"{record['device_kind']} device(s)")
    static += [
        text(
            0,
            20,
            f"MoE decoder on mesh {shape} of {len(where.devices)} {kind}, "
            f"{config.experts} experts, top-{config.top_k}, dispatch={config.dispatch}",
            15,
            weight="bold",
        ),
        text(0, 62, ops, 11, fill="#555"),
        text(
            0,
            grid_top + grid_h + 24,
            "Arrows: routed slots a device sends to another in this step's forward, "
            "summed over the MoE layers (d, e, f = data, expert, fsdp coordinate).",
            11,
            fill="#555",
        ),
        text(
            0,
            grid_top + grid_h + 40,
            "Each training frame routes that step's own batch through the "
            "step's parameters; the last frame routes the generated text.",
            11,
            fill="#555",
        ),
    ]
    if config.dispatch == "global":
        static.append(text(0, 80, "dispatch=global: each device computes every expert for its own rows, "
                                  "so no token leaves its device", 11, fill="#555"))
    charts = [
        series(
            [step["loss"] for step in record["steps"]], "training loss", 0, grid_top + grid_h + 80, grid_w, 80
        ),
        series(
            [int(np.sum(step["sent"]) - np.trace(np.sum(step["sent"], axis=0))) for step in record["steps"]],
            "slots sent to another device per step, all MoE layers",
            0,
            grid_top + grid_h + 200,
            grid_w,
            80,
        ),
    ]
    static += [chart.static for chart in charts]
    frames = [frame_group(frame, centre, config, chart_x, grid_top)
              + "".join(chart.marker(frame["step"]) for chart in charts)
              for frame in record["frames"]]
    count = len(frames)
    total = count * SECONDS_PER_FRAME
    share = 100 / count
    style = (f"@keyframes show {{ 0% {{ visibility: visible }} {share:.4f}% {{ visibility: hidden }} "
             f"100% {{ visibility: hidden }} }}\n"
             f".frame {{ visibility: hidden; animation: show {total:.1f}s step-end infinite }}\n"
             + "".join(f".f{i} {{ animation-delay: {i * SECONDS_PER_FRAME:.1f}s }}\n" for i in range(count)))
    body = "\n".join(static + [f'<g class="frame f{i}">{group}</g>' for i, group in enumerate(frames)])
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" '
        f'height="{height}" font-family="ui-sans-serif, system-ui, sans-serif">\n'
        f'<style>{style}</style>\n<rect width="100%" height="100%" fill="#fff"/>\n'
        f'<defs><marker id="head" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="9" markerHeight="9" '
        f'markerUnits="userSpaceOnUse" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" fill="#333"/>'
        f"</marker></defs>\n{body}\n</svg>\n"
    )


@dataclass
class Series:
    static: str
    points: list[tuple[float, float]]

    def marker(self, step: int | None) -> str:
        """A dot on the line at training step `step` (1-based); none for the generation frame."""
        if step is None:
            return ""
        x, y = self.points[step - 1]
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#e45756"/>'


def series(values: list[float], label: str, x: float, y: float, width: float, height: float) -> Series:
    """One value per training step as a line, with its range labelled."""
    low, high = min(values), max(values)
    points = [
        (
            x + 40 + (width - 40) * index / max(1, len(values) - 1),
            y + height - height * (value - low) / max(high - low, 1e-9),
        )
        for index, value in enumerate(values)
    ]
    line = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
    static = "".join([
        text(x, y - 6, label, 11, fill="#555"),
        text(x + 34, y + 10, f"{high:.4g}", 10, "end", fill="#555"),
        text(x + 34, y + height, f"{low:.4g}", 10, "end", fill="#555"),
        f'<polyline points="{line}" fill="none" stroke="#4c78a8" stroke-width="1.5"/>',
    ])
    return Series(static, points)


def frame_group(frame: dict, centre: dict, config: Config, chart_x: float, top: float) -> str:
    """One frame: the title, the traffic arrows, each device's slot counts and the per-layer charts."""
    label = frame["label"] + (f", loss {frame['loss']:.3f}" if frame["loss"] is not None else "")
    if config.dispatch == "exchange":
        label += (", predicted exchange rounds per MoE layer (max over expert groups) "
                  + ", ".join(str(layer["rounds"]) for layer in frame["layers"]))
    parts = [text(0, 42, label, 13, fill="#333")]
    sent = np.sum([layer["sent"] for layer in frame["layers"]], axis=0)
    away = sent - np.diag(np.diag(sent))
    peak = max(1, int(away.max()))
    for device, (cx, cy) in centre.items():
        start, stop = frame["rows"][device]
        parts += [
            text(cx - BOX_W / 2 + 8, cy - BOX_H / 2 + 68, f"batch rows {start}:{stop}", 11, fill="#555"),
            text(
                cx - BOX_W / 2 + 8,
                cy + BOX_H / 2 - 8,
                f"keeps {sent[device, device]} slots, sends {away[device].sum()}",
                11,
            ),
        ]
    for source, target in zip(*np.nonzero(away), strict=True):
        (x0, y0), (x1, y1) = centre[source], centre[target]
        # Each direction bends to its own side, so a pair's two arrows stay apart.
        bend = 30 if y1 > y0 or (y1 == y0 and x1 > x0) else -30
        mid_x, mid_y = (x0 + x1) / 2 + bend, (y0 + y1) / 2
        direction = np.sign(y1 - y0)
        y0, y1 = y0 + direction * BOX_H / 2, y1 - direction * BOX_H / 2
        stroke = 1 + 7 * away[source, target] / peak
        parts += [
            f'<path d="M{x0 + bend / 2:.1f} {y0:.1f} Q{mid_x:.1f} {mid_y:.1f} {x1 + bend / 2:.1f} {y1:.1f}" '
            f'fill="none" stroke="#333" stroke-opacity="0.55" stroke-width="{stroke:.1f}" '
            f'marker-end="url(#head)"/>',
            text(
                mid_x + (6 if bend > 0 else -6),
                mid_y + 4,
                int(away[source, target]),
                11,
                "start" if bend > 0 else "end",
                "bold",
            ),
        ]
    for index, layer in enumerate(frame["layers"]):
        parts.append(layer_chart(layer, config, chart_x, top + index * 250))
    return "".join(parts)


def layer_chart(layer: dict, config: Config, x: float, y: float) -> str:
    """Slots per expert as bars, and each token class's share of slots per expert as a heatmap."""
    parts = [text(x, y + 4, f"{layer['layer']}: slots per expert", 12, weight="bold")]
    per_expert = np.asarray(layer["per_expert"])
    peak = max(1, per_expert.max())
    for expert, count in enumerate(per_expert):
        bar = 70 * count / peak
        bx = x + 40 + expert * 34
        parts += [f'<rect x="{bx}" y="{y + 90 - bar:.1f}" width="26" height="{bar:.1f}" '
                  f'fill="{PALETTE[expert % len(PALETTE)]}"/>',
                  text(bx + 13, y + 104, expert, 10, "middle"),
                  text(bx + 13, y + 86 - bar, count, 9, "middle", fill="#555")]
    by_class = np.asarray(layer["by_class"], np.float64)
    shares = by_class / np.maximum(by_class.sum(axis=1, keepdims=True), 1)
    parts.append(text(x, y + 126, "share of each byte class's slots", 11, fill="#555"))
    for row, name in enumerate(CLASSES):
        ry = y + 134 + row * 22
        parts.append(text(x + 36, ry + 15, name, 10, "end"))
        for expert in range(config.experts):
            value = shares[row, expert]
            parts += [f'<rect x="{x + 40 + expert * 34}" y="{ry}" width="32" height="20" '
                      f'fill="#08519c" fill-opacity="{value:.3f}" stroke="#ddd"/>',
                      text(x + 56 + expert * 34, ry + 14, f"{value:.2f}", 9, "middle",
                           fill="#fff" if value > 0.5 else "#333")]
    return "".join(parts)


if __name__ == "__main__":
    main(tyro.cli(Config))
