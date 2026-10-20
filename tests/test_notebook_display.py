"""The training display in a notebook, executed in a real Jupyter kernel (nbclient) and saved.

While the run lasts its panel redraws in one output widget, as on a terminal; a saved notebook,
and the site that renders it (site/scripts/notebook-outputs.mjs), keep no widget's state, so the
run ends with the panel's last frame and the summary as static outputs. A notebook gets no line
per logging interval.
"""

import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient

import dew

TESTS = Path(__file__).resolve().parent

FIT = """\
import jax
import optax
from affine_run import TimingRegression, batches

from dew.data.dataset import Loader
from dew.training import Trainer

trainer = Trainer(TimingRegression(), optax.sgd(0.01), key=jax.random.key(0))
state = trainer.fit(Loader(lambda partition: batches(), None, None, 8), steps=4, log_every=1)
"""


def text(output) -> str:
    if output.output_type == "stream":
        return output.get("text", "")
    return output.get("data", {}).get("text/plain", "")


def test_a_notebook_redraws_the_panel_in_one_widget_and_keeps_its_last_frame_and_the_summary(tmp_path):
    notebook = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(FIT)])
    # The kernel imports the Dew under test, wherever this process found it.
    path = os.pathsep.join(filter(None, [str(Path(dew.__file__).parents[1]), os.environ.get("PYTHONPATH")]))
    NotebookClient(notebook, kernel_name="python3", resources={"metadata": {"path": str(TESTS)}}).execute(
        env={**os.environ, "PYTHONPATH": path})
    nbformat.write(notebook, tmp_path / "fit.ipynb")
    notebook = nbformat.read(tmp_path / "fit.ipynb", as_version=4)
    outputs = notebook.cells[0].outputs

    assert not [output for output in outputs if output.output_type == "error"], outputs
    widgets = [output for output in outputs
               if "application/vnd.jupyter.widget-view+json" in output.get("data", {})]
    assert len(widgets) == 1, outputs
    # The widget's saved state holds no frame of its own beside the static one.
    (model,) = [state for state in notebook.metadata.widgets["application/vnd.jupyter.widget-state+json"]
                ["state"].values() if state["model_name"] == "OutputModel"]
    assert model["state"]["outputs"] == []
    assert not [output for output in outputs if "step 1/4" in text(output)], "a line per logging interval"
    frame, summary = [output for output in outputs if output.get("data", {}).get("text/html")][-2:]
    assert "dew" in text(frame) and "Affine" in text(frame) and "finished" in text(frame)
    assert "loss" in text(frame) and "Affine" in text(summary) and "4 steps" in text(summary)
    # Styled, so the site renders each as the panel it is, not as one more line of a log.
    assert all("<span" in "".join(output.data["text/html"]) for output in (frame, summary))
