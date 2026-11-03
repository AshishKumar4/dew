"""A tutorial's recorded film as the landing page shows it (site/public/gallery/<name>/).

A tutorial that films its trained model embeds the film in its notebook's output (an `IPython.display.Video`
with `embed=True`, a data URL in the output's HTML). This tool reads that film back out of the notebook,
cuts its first `SECONDS` at `FPS` frames a second, encodes it small enough for the landing page (H.264,
under `FILM_BYTES`), takes its first frame as the poster (WebP, under `POSTER_BYTES`), and writes
`manifest.json` beside them: the film's and poster's files and the film's real width and height, the
notebook and the Dew commit it installed, and the scores its evaluation cell printed, each read from
the notebook's own outputs. Nothing is typed by hand, so a re-run notebook gives a new manifest by
running this again. A film or poster over its limit fails, writing nothing.

    python tools/tutorial_film.py tutorials/13-camera-only-drone-racing.ipynb flydreamer

needs imageio with its ffmpeg (imageio-ffmpeg) and Pillow.
"""

import argparse
import base64
import io
import json
import re
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
GALLERY = ROOT / "site" / "public" / "gallery"

SECONDS, FPS = 8, 24
"""The landing film: the film's first `SECONDS`, at `FPS` frames a second."""

FILM_BYTES, POSTER_BYTES = 600_000, 60_000
"""The most bytes the landing film and its poster may take (Platform's budget for a landing band)."""

CRF = 33
"""The film's H.264 constant rate factor: the size it is held under `FILM_BYTES` at."""

_VIDEO = re.compile(r'src="data:video/mp4;base64,([A-Za-z0-9+/=]+)"')
_COMMIT = re.compile(r"github\.com/AshishKumar4/dew@([0-9a-f]{40})")
_SCORE = re.compile(r"^\s*(\w+)\s+(\w+):\s+([-\d.]+) gates, return\s+([-\d.]+),\s+(\d+) steps of (\d+)\s*$")


def _outputs(notebook: dict) -> list[tuple[str, str]]:
    """Every code cell's source beside each of its outputs' text, in order."""
    found = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        for output in cell.get("outputs", []):
            text = output.get("text") or output.get("data", {}).get("text/html") or ""
            found.append((source, "".join(text) if isinstance(text, list) else text))
    return found


def film(notebook: dict) -> bytes:
    """The one film the notebook embeds, as its MP4 bytes; a notebook embedding none or several is
    refused."""
    films = [match for _, text in _outputs(notebook) for match in _VIDEO.findall(text)]
    if len(films) != 1:
        raise ValueError(f"the notebook embeds {len(films)} films; the landing page shows exactly one")
    return base64.b64decode(films[0])


def scores(notebook: dict) -> list[dict]:
    """The evaluation cell's printed lines, each a policy's score over a judged race: `<policy> <race>:
    <gates> gates, return <return>, <steps> steps of <episode>`."""
    found = []
    for _, text in _outputs(notebook):
        for line in text.splitlines():
            match = _SCORE.match(line)
            if match:
                policy, race, gates, returned, steps, episode = match.groups()
                found.append({"policy": policy, "race": race, "gates": float(gates),
                              "return": float(returned), "steps": int(steps), "episode": int(episode)})
    if not found:
        raise ValueError("the notebook printed no evaluation scores")
    return found


def commit(notebook: dict) -> str:
    """The Dew commit the notebook installed."""
    commits = {match for source, _ in _outputs(notebook) for match in _COMMIT.findall(source)}
    commits |= {match for cell in notebook["cells"] for match in _COMMIT.findall("".join(cell["source"]))}
    if len(commits) != 1:
        raise ValueError(f"the notebook installs {len(commits)} Dew commits; its film names exactly one")
    return commits.pop()


def landing(source: bytes) -> tuple[bytes, bytes, int, int, float]:
    """The landing film's MP4 and its poster's WebP, the film's width and height, and the source's rate."""
    with tempfile.TemporaryDirectory() as scratch:
        original, cut = Path(scratch) / "film.mp4", Path(scratch) / "landing.mp4"
        original.write_bytes(source)
        reader = imageio.get_reader(original)
        rate = float(reader.get_meta_data()["fps"])
        every = max(1, round(rate / FPS))
        frames = [frame for index, frame in enumerate(reader) if index % every == 0][:SECONDS * FPS]
        reader.close()
        height, width = frames[0].shape[:2]
        imageio.mimwrite(cut, frames, fps=FPS, codec="libx264", macro_block_size=1,
                         output_params=["-crf", str(CRF), "-preset", "slow", "-pix_fmt", "yuv420p",
                                        "-movflags", "+faststart"])
        encoded = cut.read_bytes()
    poster = io.BytesIO()
    Image.fromarray(np.asarray(frames[0])).save(poster, "WEBP", quality=70, method=6)
    return encoded, poster.getvalue(), width, height, rate


def write(notebook_path: Path, name: str) -> Path:
    """Write `name`'s landing film, poster and manifest from the notebook at `notebook_path`."""
    notebook = json.loads(notebook_path.read_text())
    encoded, poster, width, height, rate = landing(film(notebook))
    for kind, data, limit in (("film", encoded, FILM_BYTES), ("poster", poster, POSTER_BYTES)):
        if len(data) > limit:
            raise ValueError(f"the landing {kind} is {len(data)} bytes, over its {limit}")
    directory = GALLERY / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.mp4").write_bytes(encoded)
    (directory / f"{name}.webp").write_bytes(poster)
    manifest = {
        "video": f"{name}.mp4", "poster": f"{name}.webp", "width": width, "height": height,
        "seconds": SECONDS, "fps": FPS, "source_fps": rate,
        "notebook": notebook_path.resolve().relative_to(ROOT).as_posix(), "dew_commit": commit(notebook),
        "scores": scores(notebook),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return directory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("notebook", type=Path, help="the tutorial notebook whose film to show")
    parser.add_argument("name", help="the gallery folder under site/public/gallery/")
    arguments = parser.parse_args()
    directory = write(arguments.notebook, arguments.name)
    print({path.name: path.stat().st_size for path in sorted(directory.iterdir())})


if __name__ == "__main__":
    main()
