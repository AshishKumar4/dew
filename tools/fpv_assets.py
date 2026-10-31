"""The gate race's renderer's assets (`dew.rl.hyperlux`), built as GLB files in OUT with their manifest:

    python tools/fpv_assets.py OUT

- Ground: a 400 m square of 8 m tiles, each a CC0 photograph of terrain from Poly Haven (grass and rocks,
  forest floor, asphalt from above, beach, rocky terrain, muddy leaves).
- Sky: a dome 350 m in radius textured inside with a CC0 sky from Poly Haven (its tonemapped panorama), its
  lower half the landscape at the horizon.
- Gates: the race's gate (`dew.rl.flight.Racing`: its opening, frame and depth) in four looks (orange, blue,
  white, black-and-red padding), and lit in four (green, yellow-and-black chevrons, magenta checks, cyan
  stripes) for the gate a drone heads for.
- Pillars: a cylinder of unit radius at each of the race's aspects (`dew.rl.flight.ASPECTS`), in concrete and
  in rusted metal (Poly Haven, CC0).
- Props, distractors outside the course: a tree, a rock (Poly Haven's rocky terrain, CC0) and a banner on two
  poles (Poly Haven's red brick, CC0).
- A quadrotor for a chase camera's view: a body, four arms and four rotor discs, 0.24 m across its arms.

Every texture is CC0 (polyhaven.com/license); the files are fetched from Poly Haven's API and its CDN and
shrunk to 256 pixels (a sky's to 1024 by 512): the race's camera sees 96. A GLB here is written by
`glb`, each part a node of its own with its own base-color texture (glTF 2.0, binary)."""

import io
import json
import math
import struct
import sys
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

GROUNDS = ("aerial_grass_rock", "forrest_ground_01", "aerial_asphalt_01", "aerial_beach_01",
           "rocky_terrain_02", "brown_mud_leaves_01")
SKIES = ("kloofendal_48d_partly_cloudy_puresky", "lilienstein", "the_sky_is_on_fire", "kloppenheim_02")
PILLARS = ("concrete_floor_worn_001", "rust_coarse_01")
ROCK, BRICK = "rocky_terrain_02", "red_brick"
TEXTURES = "https://api.polyhaven.com/files/{name}"
TONEMAPPED = "https://dl.polyhaven.org/file/ph-assets/HDRIs/extra/Tonemapped%20JPG/{name}.jpg"


def fetched(url: str) -> bytes:
    """`url`'s body, asked for by this tool's name: Poly Haven's API refuses urllib's agent (HTTP 403)."""
    request = urllib.request.Request(url, headers={"User-Agent": "dew-fpv-assets/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def texture(name: str, size: int = 256) -> Image.Image:
    """A Poly Haven texture's diffuse map at 1k, shrunk to `size` pixels square."""
    files = json.loads(fetched(TEXTURES.format(name=name)))
    url = files["Diffuse"]["1k"]["jpg"]["url"]
    return Image.open(io.BytesIO(fetched(url))).convert("RGB").resize((size, size), Image.LANCZOS)


def sky(name: str) -> Image.Image:
    """A Poly Haven sky's tonemapped panorama, shrunk to 1024 by 512."""
    return Image.open(io.BytesIO(fetched(TONEMAPPED.format(name=name)))).convert("RGB").resize(
        (1024, 512), Image.LANCZOS)


def flat(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (4, 4), color)


def striped(first: tuple[int, int, int], second: tuple[int, int, int], stripes: int = 8,
            diagonal: bool = False) -> Image.Image:
    """Stripes of two colours, `stripes` across a 128-pixel square, straight or diagonal (chevrons)."""
    y, x = np.mgrid[0:128, 0:128]
    band = ((x + y) if diagonal else x) * stripes // 128 % 2
    return Image.fromarray(np.where(band[..., None] == 0, first, second).astype(np.uint8))


def checked(first: tuple[int, int, int], second: tuple[int, int, int], squares: int = 4) -> Image.Image:
    y, x = np.mgrid[0:128, 0:128]
    band = (x * squares // 128 + y * squares // 128) % 2
    return Image.fromarray(np.where(band[..., None] == 0, first, second).astype(np.uint8))


Part = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Image.Image]
"""A mesh's positions `[V, 3]`, normals `[V, 3]`, texture coordinates `[V, 2]` in [0, 1], triangles `[F, 3]`,
and its base-color image."""


def box(low, high, image: Image.Image) -> Part:
    """An axis-aligned box from `low` to `high`, each face its own four corners mapped to the whole image."""
    low, high = np.asarray(low, np.float32), np.asarray(high, np.float32)
    positions, normals, uvs, faces = [], [], [], []
    for axis in range(3):
        for side in (0, 1):
            normal = np.zeros(3, np.float32)
            normal[axis] = 1.0 if side else -1.0
            u_axis, v_axis = [a for a in range(3) if a != axis]
            corners = []
            for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
                point = np.where(np.arange(3) == axis, high if side else low, 0.0)
                point[u_axis] = (low, high)[u][u_axis]
                point[v_axis] = (low, high)[v][v_axis]
                corners.append(point)
            base = len(positions)
            positions += corners
            normals += [normal] * 4
            uvs += [(0, 0), (1, 0), (1, 1), (0, 1)]
            # Wound counterclockwise seen from outside.
            order = (0, 1, 2, 0, 2, 3) if (side == 1) == ((u_axis, v_axis) in ((0, 1), (1, 2), (2, 0))) else \
                (0, 2, 1, 0, 3, 2)
            faces += [(base + order[0], base + order[1], base + order[2]),
                      (base + order[3], base + order[4], base + order[5])]
    return (np.asarray(positions, np.float32), np.asarray(normals, np.float32), np.asarray(uvs, np.float32),
            np.asarray(faces, np.uint32), image)


def cylinder(radius: float, height: float, image: Image.Image, segments: int = 16, base: float = 0.0,
             top_radius: float | None = None) -> Part:
    """A cylinder (or a cone, with `top_radius`) standing on z = `base`, its side mapped once round the
    image, capped."""
    top_radius = radius if top_radius is None else top_radius
    positions, normals, uvs, faces = [], [], [], []
    for index in range(segments + 1):
        angle = 2 * math.pi * index / segments
        c, s = math.cos(angle), math.sin(angle)
        slope = (radius - top_radius) / height
        normal = np.asarray([c, s, slope]) / math.sqrt(1 + slope * slope)
        positions += [(radius * c, radius * s, base), (top_radius * c, top_radius * s, base + height)]
        normals += [normal, normal]
        uvs += [(index / segments, 0.0), (index / segments, 1.0)]
    for index in range(segments):
        a, b = 2 * index, 2 * index + 2
        faces += [(a, b, a + 1), (a + 1, b, b + 1)]
    for z, r, up in ((base, radius, -1.0), (base + height, top_radius, 1.0)):
        if r <= 0:
            continue
        center = len(positions)
        positions.append((0.0, 0.0, z))
        normals.append((0.0, 0.0, up))
        uvs.append((0.5, 0.5))
        for index in range(segments):
            angle = 2 * math.pi * index / segments
            positions.append((r * math.cos(angle), r * math.sin(angle), z))
            normals.append((0.0, 0.0, up))
            uvs.append((0.5 + 0.5 * math.cos(angle), 0.5 + 0.5 * math.sin(angle)))
        for index in range(segments):
            a, b = center + 1 + index, center + 1 + (index + 1) % segments
            faces.append((center, b, a) if up < 0 else (center, a, b))
    return (np.asarray(positions, np.float32), np.asarray(normals, np.float32), np.asarray(uvs, np.float32),
            np.asarray(faces, np.uint32), image)


def ground(image: Image.Image, extent: float = 400.0, tile: float = 8.0) -> Part:
    """A square of ground `extent` metres a side about the origin at z = 0, each `tile` its own image."""
    count = int(extent / tile)
    positions, uvs, faces = [], [], []
    for row in range(count):
        for column in range(count):
            x, y = -extent / 2 + row * tile, -extent / 2 + column * tile
            base = len(positions)
            positions += [(x, y, 0.0), (x + tile, y, 0.0), (x + tile, y + tile, 0.0), (x, y + tile, 0.0)]
            uvs += [(0, 0), (1, 0), (1, 1), (0, 1)]
            faces += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
    positions = np.asarray(positions, np.float32)
    return (positions, np.tile(np.asarray([[0, 0, 1]], np.float32), (len(positions), 1)),
            np.asarray(uvs, np.float32), np.asarray(faces, np.uint32), image)


def dome(image: Image.Image, radius: float = 350.0, columns: int = 64, rows: int = 32) -> Part:
    """A sphere of `radius` about the origin seen from inside, its panorama mapped by longitude and
    latitude."""
    positions, normals, uvs, faces = [], [], [], []
    for row in range(rows + 1):
        polar = math.pi * row / rows
        for column in range(columns + 1):
            azimuth = 2 * math.pi * column / columns
            direction = (math.sin(polar) * math.cos(azimuth), math.sin(polar) * math.sin(azimuth),
                         math.cos(polar))
            positions.append(tuple(radius * d for d in direction))
            normals.append(tuple(-d for d in direction))
            uvs.append((1 - column / columns, row / rows))
    for row in range(rows):
        for column in range(columns):
            a = row * (columns + 1) + column
            b = a + columns + 1
            faces += [(a, a + 1, b), (a + 1, b + 1, b)]
    return (np.asarray(positions, np.float32), np.asarray(normals, np.float32), np.asarray(uvs, np.float32),
            np.asarray(faces, np.uint32), image)


def rock(image: Image.Image, seed: int = 0) -> Part:
    """A rock: a sphere of 1.2 m squashed and roughened, mapped by longitude and latitude."""
    rng = np.random.default_rng(seed)
    positions, normals, uvs, faces = [], [], [], []
    columns, rows = 16, 8
    bumps = rng.uniform(0.75, 1.15, (rows + 1, columns + 1))
    bumps[:, -1] = bumps[:, 0]
    for row in range(rows + 1):
        polar = math.pi * row / rows
        for column in range(columns + 1):
            azimuth = 2 * math.pi * column / columns
            direction = np.asarray([math.sin(polar) * math.cos(azimuth), math.sin(polar) * math.sin(azimuth),
                                    math.cos(polar)])
            point = 1.2 * bumps[row, column] * direction * np.asarray([1.3, 1.0, 0.7])
            positions.append(point + np.asarray([0.0, 0.0, 0.5]))
            normals.append(direction)
            uvs.append((column / columns, row / rows))
    for row in range(rows):
        for column in range(columns):
            a = row * (columns + 1) + column
            b = a + columns + 1
            faces += [(a, b, a + 1), (a + 1, b, b + 1)]
    return (np.asarray(positions, np.float32), np.asarray(normals, np.float32), np.asarray(uvs, np.float32),
            np.asarray(faces, np.uint32), image)


def glb(path: Path, parts: list[Part]) -> None:
    """`parts` written as a binary glTF 2.0 file, each a node and mesh of its own, its image a PNG base-color
    texture (repeating, linear) on an unlit-looking rough dielectric."""
    binary = bytearray()
    views, accessors, meshes, nodes, materials, textures, images = [], [], [], [], [], [], []

    def view(data: bytes, target: int | None = None) -> int:
        while len(binary) % 4:
            binary.append(0)
        entry = {"buffer": 0, "byteOffset": len(binary), "byteLength": len(data)}
        if target is not None:
            entry["target"] = target
        binary.extend(data)
        views.append(entry)
        return len(views) - 1

    def accessor(array: np.ndarray, kind: str, component: int, target: int) -> int:
        entry = {"bufferView": view(array.tobytes(), target), "componentType": component, "count": len(array),
                 "type": kind}
        if kind == "VEC3" and component == 5126:
            entry["min"] = array.min(0).tolist()
            entry["max"] = array.max(0).tolist()
        accessors.append(entry)
        return len(accessors) - 1

    for positions, normals, uvs, faces, image in parts:
        encoded = io.BytesIO()
        image.save(encoded, format="PNG")
        images.append({"bufferView": view(encoded.getvalue()), "mimeType": "image/png"})
        textures.append({"sampler": 0, "source": len(images) - 1})
        materials.append({"pbrMetallicRoughness": {"baseColorTexture": {"index": len(textures) - 1},
                                                   "metallicFactor": 0.0, "roughnessFactor": 1.0}})
        primitive = {"attributes": {"POSITION": accessor(positions, "VEC3", 5126, 34962),
                                    "NORMAL": accessor(normals, "VEC3", 5126, 34962),
                                    "TEXCOORD_0": accessor(uvs, "VEC2", 5126, 34962)},
                     "indices": accessor(faces.reshape(-1), "SCALAR", 5125, 34963),
                     "material": len(materials) - 1}
        meshes.append({"primitives": [primitive]})
        nodes.append({"mesh": len(meshes) - 1})
    while len(binary) % 4:
        binary.append(0)
    document = {"asset": {"version": "2.0", "generator": "dew tools/fpv_assets.py"}, "scene": 0,
                "scenes": [{"nodes": list(range(len(nodes)))}], "nodes": nodes, "meshes": meshes,
                "materials": materials, "textures": textures, "images": images,
                "samplers": [{"magFilter": 9729, "minFilter": 9729, "wrapS": 10497, "wrapT": 10497}],
                "accessors": accessors, "bufferViews": views, "buffers": [{"byteLength": len(binary)}]}
    text = json.dumps(document, separators=(",", ":")).encode()
    text += b" " * (-len(text) % 4)
    chunks = (struct.pack("<II", len(text), 0x4E4F534A) + text + struct.pack("<II", len(binary), 0x004E4942)
              + binary)
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(chunks)) + chunks)


def gate(opening: float, frame: float, depth: float, image: Image.Image) -> list[Part]:
    """A square gate facing along x about its centre: four bars `frame` wide and `depth` deep about an opening
    `opening` wide (the race's `bars`, ungrown)."""
    half, edge, deep = opening / 2, opening / 2 + frame, depth / 2
    return [box((-deep, -edge, half), (deep, edge, edge), image),
            box((-deep, -edge, -edge), (deep, edge, -half), image),
            box((-deep, half, -half), (deep, edge, half), image),
            box((-deep, -edge, -half), (deep, -half, half), image)]


def turned(part: Part, angle: float, offset=(0.0, 0.0, 0.0)) -> Part:
    """`part` turned by `angle` degrees about z, then moved by `offset`."""
    c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    rotation = np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], np.float32)
    positions, normals, uvs, faces, image = part
    return (positions @ rotation.T + np.asarray(offset, np.float32), normals @ rotation.T, uvs, faces, image)


def drone() -> list[Part]:
    """A quadrotor 0.24 m across its arms: a dark body with a red nose, four arms in an X, four grey rotor
    discs."""
    dark, arm, disc = flat((30, 30, 34)), flat((60, 60, 66)), flat((150, 150, 160))
    parts = [box((-0.06, -0.04, -0.02), (0.06, 0.04, 0.03), dark),
             box((0.04, -0.015, 0.0), (0.07, 0.015, 0.03), flat((200, 40, 30)))]
    for angle in (45, 135, 225, 315):
        c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
        parts.append(turned(box((0.0, -0.008, -0.005), (0.12, 0.008, 0.005), arm), angle))
        parts.append(turned(cylinder(0.06, 0.004, disc, base=0.015), 0.0, (0.12 * c, 0.12 * s, 0.0)))
    return parts


def main() -> None:
    from dew.rl.flight import ASPECTS, Racing

    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    racing = Racing()
    manifest: dict = {"license": "CC0 (polyhaven.com/license) textures; geometry Dew's", "sources": {}}
    manifest["ground"] = []
    for name in GROUNDS:
        glb(out / f"ground_{name}.glb", [ground(texture(name))])
        manifest["ground"].append(f"ground_{name}.glb")
        manifest["sources"][name] = TEXTURES.format(name=name)
    manifest["sky"] = []
    for name in SKIES:
        glb(out / f"sky_{name}.glb", [dome(sky(name))])
        manifest["sky"].append(f"sky_{name}.glb")
        manifest["sources"][name] = TONEMAPPED.format(name=name)
    looks = {"orange": flat((235, 110, 25)), "blue": flat((30, 80, 200)), "white": flat((225, 225, 220)),
             "padded": striped((20, 20, 20), (200, 30, 30), stripes=6)}
    lit = {"green": flat((40, 255, 60)), "chevrons": striped((255, 220, 0), (15, 15, 15), 8, diagonal=True),
           "magenta": checked((255, 30, 220), (255, 255, 255)),
           "cyan": striped((0, 240, 255), (255, 255, 255), 10)}
    for kind, images in (("gate", looks), ("lit", lit)):
        manifest[kind] = []
        for name, image in images.items():
            glb(out / f"{kind}_{name}.glb", gate(racing.opening, racing.frame, racing.depth, image))
            manifest[kind].append(f"{kind}_{name}.glb")
    manifest["pillar"] = []
    for aspect in ASPECTS:
        variants = []
        for name in PILLARS:
            path = f"pillar_{aspect:g}_{name}.glb"
            glb(out / path, [cylinder(1.0, aspect, texture(name))])
            variants.append(path)
            manifest["sources"][name] = TEXTURES.format(name=name)
        manifest["pillar"].append(variants)
    bark, leaves = flat((90, 60, 35)), flat((40, 110, 45))
    glb(out / "prop_tree.glb", [cylinder(0.25, 3.0, bark, segments=10), cylinder(2.2, 4.0, leaves, base=2.5,
                                                                              top_radius=0.0, segments=12),
                                cylinder(1.6, 3.0, leaves, base=5.0, top_radius=0.0, segments=12)])
    glb(out / "prop_rock.glb", [rock(texture(ROCK))])
    banner = texture(BRICK)
    pole = cylinder(0.06, 3.0, flat((180, 180, 185)), segments=8)
    glb(out / "prop_banner.glb", [pole, turned(pole, 0.0, (0.0, 4.0, 0.0)),
                                  box((-0.03, 0.0, 1.6), (0.03, 4.0, 3.0), banner)])
    manifest["sources"][ROCK] = TEXTURES.format(name=ROCK)
    manifest["sources"][BRICK] = TEXTURES.format(name=BRICK)
    manifest["prop"] = ["prop_tree.glb", "prop_rock.glb", "prop_banner.glb"]
    glb(out / "drone.glb", drone())
    manifest["drone"] = "drone.glb"
    manifest["aspects"] = list(ASPECTS)
    manifest["gate_geometry"] = {"opening": racing.opening, "frame": racing.frame, "depth": racing.depth}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({key: value for key, value in manifest.items() if key != "sources"}, indent=1))


if __name__ == "__main__":
    main()
